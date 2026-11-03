"""Tests for the chat page's streaming proxy: payloads, stream parsing, stats."""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from chat_proxy import (  # noqa: E402
    EndpointError, build_payload, compute_stats, probe, resolve_endpoint, stream_chat,
)


# ---------- payload ----------

def test_payload_drops_blank_sampling_and_maps_repeat_penalty():
    msgs = [{"role": "user", "content": "hi"}]
    sampling = {"temperature": 0.7, "top_k": 40, "min_p": "", "seed": None, "repeat_penalty": 1.1}

    llama = build_payload("llamacpp", "alias", msgs, sampling)
    assert llama["repeat_penalty"] == 1.1 and "repetition_penalty" not in llama
    assert llama["temperature"] == 0.7 and llama["top_k"] == 40
    assert "min_p" not in llama and "seed" not in llama
    assert llama["stream"] is True and llama["stream_options"] == {"include_usage": True}
    assert llama["model"] == "alias"

    vllm = build_payload("vllm", "/models/x", msgs, sampling)
    assert vllm["repetition_penalty"] == 1.1 and "repeat_penalty" not in vllm


def test_payload_asks_each_engine_for_running_stats():
    msgs = [{"role": "user", "content": "hi"}]
    llama = build_payload("llamacpp", None, msgs, {})
    vllm = build_payload("vllm", None, msgs, {})
    other = build_payload("openai", None, msgs, {})
    assert llama["timings_per_token"] is True
    assert "continuous_usage_stats" not in llama["stream_options"]
    assert vllm["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}
    assert "timings_per_token" not in vllm
    # unknown servers get neither, a strict one could reject the field
    assert "timings_per_token" not in other
    assert other["stream_options"] == {"include_usage": True}


def test_payload_unknown_engine_gets_no_repeat_penalty():
    body = build_payload("openai", "m", [{"role": "user", "content": "x"}], {"repeat_penalty": 1.1})
    assert "repeat_penalty" not in body and "repetition_penalty" not in body


def test_payload_omits_model_when_unknown():
    assert "model" not in build_payload("llamacpp", None, [{"role": "user", "content": "x"}], {})


# ---------- stats ----------

def test_engine_timings_win():
    s = compute_stats(
        t_start=0.0, t_first=0.5, t_last=4.5, t_end=4.6, usage={"prompt_tokens": 999},
        timings={"prompt_n": 1000, "cache_n": 200, "prompt_ms": 400.0, "prompt_per_second": 2500.0,
                 "predicted_n": 500, "predicted_ms": 4000.0, "predicted_per_second": 125.0},
        delta_count=500, finish_reason="stop")
    assert s.source == "engine"
    assert (s.prompt_tokens, s.cached_tokens, s.prefill_tps) == (1000, 200, 2500.0)
    assert s.prefill_s == pytest.approx(0.4) and s.gen_s == pytest.approx(4.0)
    assert s.gen_tps == 125.0 and s.ttft_s == pytest.approx(0.5)


def test_measured_from_usage_excludes_cache_and_first_token():
    s = compute_stats(
        t_start=10.0, t_first=10.5, t_last=14.5, t_end=14.6,
        usage={"prompt_tokens": 1200, "completion_tokens": 401,
               "prompt_tokens_details": {"cached_tokens": 200}},
        timings=None, delta_count=380, finish_reason="length")
    assert s.source == "measured"
    assert s.prompt_tokens == 1000 and s.cached_tokens == 200
    assert s.prefill_tps == pytest.approx(1000 / 0.5)
    assert s.gen_tps == pytest.approx(400 / 4.0)   # (401 - 1) tokens over first->last
    assert s.finish_reason == "length" and s.approx is False


def test_measured_without_usage_falls_back_to_chunk_count():
    s = compute_stats(t_start=0.0, t_first=1.0, t_last=3.0, t_end=3.0, usage=None,
                      timings=None, delta_count=21, finish_reason=None)
    assert s.gen_tokens == 21 and s.gen_tps == pytest.approx(10.0)
    assert s.approx is True
    assert s.prompt_tokens is None and s.prefill_tps is None


def test_no_tokens_at_all_does_not_divide_by_zero():
    s = compute_stats(t_start=0.0, t_first=None, t_last=None, t_end=1.0, usage=None,
                      timings=None, delta_count=0, finish_reason=None)
    assert s.ttft_s is None and s.gen_tps is None and s.total_s == 1.0


# ---------- stream ----------

def _sse(*chunks):
    return "".join(f"data: {json.dumps(c) if not isinstance(c, str) else c}\n\n" for c in chunks)


def _run(handler):
    clock_vals = iter([0.0] + [0.1 * i for i in range(1, 100)])
    async def go():
        out = []
        async for line in stream_chat("http://127.0.0.1:9999", {"messages": []},
                                      clock=lambda: next(clock_vals),
                                      transport=httpx.MockTransport(handler)):
            out.append(json.loads(line))
        return out
    return asyncio.run(go())


def test_stream_vllm_reasoning_content_usage():
    body = _sse(
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"reasoning": "think"}}]},          # newer vLLM field name
        {"choices": [{"delta": {"reasoning_content": " more"}}]},  # older field name
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4}},
        "[DONE]",
    )
    events = _run(lambda req: httpx.Response(200, text=body))
    kinds = [e["type"] for e in events if not (e["type"] == "stats" and not e["final"])]
    assert kinds == ["reasoning", "reasoning", "content", "content", "stats"]
    assert "".join(e["text"] for e in events if e["type"] == "content") == "Hello"
    st = events[-1]
    assert st["final"] is True
    assert st["source"] == "measured" and st["prompt_tokens"] == 10 and st["gen_tokens"] == 4
    assert st["finish_reason"] == "stop" and st["gen_tps"] > 0


def test_stream_llama_timings_and_request_shape():
    seen = {}
    def handler(req):
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, text=_sse(
            {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}],
             "timings": {"prompt_n": 5, "prompt_ms": 10.0, "prompt_per_second": 500.0,
                         "predicted_n": 1, "predicted_ms": 1.0, "predicted_per_second": 1000.0}},
            "[DONE]"))
    events = _run(handler)
    assert seen["url"] == "http://127.0.0.1:9999/v1/chat/completions"
    assert events[-1]["source"] == "engine" and events[-1]["prefill_tps"] == 500.0


def test_stream_http_error_surfaces_message():
    events = _run(lambda req: httpx.Response(
        400, json={"error": {"message": "maximum context length exceeded"}}))
    assert events == [{"type": "error", "message": "HTTP 400: maximum context length exceeded"}]


def test_stream_inline_error_chunk():
    events = _run(lambda req: httpx.Response(200, text=_sse({"error": {"message": "boom"}})))
    assert events == [{"type": "error", "message": "boom"}]


def test_stream_connect_error():
    def handler(req):
        raise httpx.ConnectError("refused", request=req)
    events = _run(handler)
    assert events[0]["type"] == "error" and "127.0.0.1:9999" in events[0]["message"]


# ---------- endpoint validation ----------

@pytest.mark.parametrize("typed, base, label", [
    ("localhost:8302", "http://127.0.0.1:8302", "localhost:8302"),
    ("http://127.0.0.1:8080/v1", "http://127.0.0.1:8080", "127.0.0.1:8080"),
    ("192.168.4.200:8110/", "http://192.168.4.200:8110", "192.168.4.200:8110"),
    ("10.0.0.5:9000", "http://10.0.0.5:9000", "10.0.0.5:9000"),
    ("[::1]:8000", "http://[::1]:8000", "[::1]:8000"),
])
def test_resolve_accepts_local_and_private(typed, base, label):
    assert asyncio.run(resolve_endpoint(typed)) == (base, label)


@pytest.mark.parametrize("typed", [
    "",                          # empty
    "localhost",                 # no port
    "localhost:99999",           # bad port
    "8.8.8.8:80",                # public address
    "169.254.169.254:80",        # link-local / cloud metadata
    "localhost:8080/admin",      # arbitrary path
    "ftp://localhost:21",        # scheme
])
def test_resolve_rejects(typed):
    with pytest.raises(EndpointError):
        asyncio.run(resolve_endpoint(typed))


# ---------- engine detection ----------

def _probe(routes):
    def handler(req):
        if req.url.path in routes:
            return routes[req.url.path]
        return httpx.Response(404)
    return asyncio.run(probe("http://127.0.0.1:1", transport=httpx.MockTransport(handler)))


def test_probe_vllm_by_owner():
    info = _probe({"/v1/models": httpx.Response(200, json={"data": [
        {"id": "ornith-122b", "owned_by": "vllm"}]})})
    assert info == {"models": ["ornith-122b"], "engine": "vllm"}


def test_probe_llamacpp_by_owner_and_by_props():
    by_owner = _probe({"/v1/models": httpx.Response(200, json={"data": [{"id": "a", "owned_by": "llamacpp"}]})})
    by_props = _probe({"/v1/models": httpx.Response(200, json={"data": [{"id": "a"}]}),
                       "/props": httpx.Response(200, json={})})
    assert by_owner["engine"] == by_props["engine"] == "llamacpp"


def test_probe_generic_openai_server():
    info = _probe({"/v1/models": httpx.Response(200, json={"data": [{"id": "x", "owned_by": "someone"}, {"id": "y"}]})})
    assert info == {"models": ["x", "y"], "engine": "openai"}


# ---------- running stats (live t/s, and what a stopped reply shows) ----------

def _usage_stream(n):
    # vLLM with continuous_usage_stats: usage on every chunk
    chunks = [{"choices": [{"delta": {"content": f"t{i} "}}],
               "usage": {"prompt_tokens": 50, "completion_tokens": i + 1}} for i in range(n)]
    return _sse(*chunks, "[DONE]")


def test_running_stats_are_sent_while_streaming():
    # the fake clock advances 0.1s per read, so a 0.5s interval fires several times
    events = _run(lambda req: httpx.Response(200, text=_usage_stream(30)))
    progress = [e for e in events if e["type"] == "stats" and not e["final"]]
    assert len(progress) >= 3
    counts = [p["gen_tokens"] for p in progress]
    assert counts == sorted(counts) and counts[0] >= 1
    assert all(p["gen_tps"] is None or p["gen_tps"] > 0 for p in progress)
    assert events[-1]["final"] is True and events[-1]["gen_tokens"] == 30


def test_stopped_stream_still_has_recent_stats():
    """The client stops reading partway; it must already hold real figures."""
    clock_vals = iter([0.1 * i for i in range(1000)])

    async def go():
        seen, content = [], 0
        agen = stream_chat("http://127.0.0.1:9999", {"messages": []},
                           clock=lambda: next(clock_vals),
                           transport=httpx.MockTransport(lambda r: httpx.Response(200, text=_usage_stream(40))))
        async for line in agen:
            ev = json.loads(line)
            seen.append(ev)
            content += ev["type"] == "content"
            if content == 20:
                break          # the user pressed Stop
        await agen.aclose()
        return seen

    seen = asyncio.run(go())
    last = [e for e in seen if e["type"] == "stats"][-1]
    assert last["final"] is False
    assert last["gen_tokens"] >= 15 and last["prompt_tokens"] == 50
