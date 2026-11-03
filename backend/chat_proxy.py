"""Streaming chat proxy for the built-in test chat page.

The browser never talks to an engine directly: it posts to Llama Studio, which
forwards to an OpenAI-compatible /v1/chat/completions and re-emits a small
normalized NDJSON stream. One code path serves llama-server and vLLM, and
timing is measured next to the engine instead of across the browser's network.

The target is any server the user types in (Studio-managed or not), so the
endpoint is validated first: Llama Studio listens on 0.0.0.0 without auth, and
an unrestricted forwarder would let anyone on the network use it to reach
arbitrary hosts. Only loopback / private-network addresses are accepted, the
connection goes to the resolved IP (no re-resolution between check and use),
and only the fixed /v1/models, /props and /v1/chat/completions paths are ever
requested.

Normalized events (one JSON object per line):
    {"type": "content",   "text": "..."}
    {"type": "reasoning", "text": "..."}
    {"type": "stats", ...}   # see ChatStats.to_dict()
    {"type": "error", "message": "..."}

Stats sources differ by engine:
  - llama-server sends a `timings` object in its final chunk: exact prefill
    and generation figures from the engine itself ("engine" source).
  - vLLM sends only token counts (`usage`, requested via stream_options), so
    prefill time is the time to the first streamed token and generation time
    is first-to-last token, both measured here ("measured" source).

Both engines are asked to report those figures on every chunk, not just the
last (llama-server `timings_per_token`, vLLM `continuous_usage_stats`), and a
running {"type": "stats", "final": false} event goes out every
PROGRESS_INTERVAL_S. That gives the page accurate live t/s, and something to
show when the user stops a reply: a stopped stream never reaches the final
chunk where the engines would otherwise put their numbers.
"""

import asyncio
import ipaddress
import json
import logging
import socket
import time
from dataclasses import dataclass, asdict
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

# Sampling fields the page may send. Values left blank on the page are omitted,
# so the engine's (or the model's generation_config's) defaults apply.
COMMON_SAMPLING = ("temperature", "top_p", "max_tokens", "presence_penalty",
                   "frequency_penalty", "seed", "top_k", "min_p")
# Same knob, different name per engine. Unknown OpenAI-compatible servers get
# neither: a strict server may reject a field it doesn't know.
REPEAT_PENALTY_KEY = {"llamacpp": "repeat_penalty", "vllm": "repetition_penalty"}

# How often a running stats event is sent while tokens are streaming.
PROGRESS_INTERVAL_S = 0.5

# Prefill on a huge prompt can take minutes before the first byte, so no read
# timeout; connecting to a local port should be instant.
_TIMEOUT = httpx.Timeout(connect=5.0, read=None, write=30.0, pool=5.0)


@dataclass
class ChatStats:
    source: str                         # "engine" | "measured"
    prompt_tokens: Optional[int] = None   # tokens actually prefilled (excl. cache)
    cached_tokens: Optional[int] = None   # prompt tokens served from prefix cache
    prefill_s: Optional[float] = None
    prefill_tps: Optional[float] = None
    gen_tokens: Optional[int] = None
    gen_s: Optional[float] = None
    gen_tps: Optional[float] = None
    ttft_s: Optional[float] = None      # always measured here, both engines
    total_s: Optional[float] = None
    finish_reason: Optional[str] = None
    # gen_tokens counted from stream chunks because the server sent no usage;
    # with speculative decoding one chunk can carry several tokens.
    approx: bool = False
    final: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "stats", **asdict(self)}


def build_payload(engine: str, served_model: Optional[str], messages: List[Dict[str, Any]],
                  sampling: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI chat request body for the given engine, dropping blank sampling fields."""
    body: Dict[str, Any] = {
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if served_model:
        body["model"] = served_model
    for key in COMMON_SAMPLING:
        val = sampling.get(key)
        if val is not None and val != "":
            body[key] = val
    rp = sampling.get("repeat_penalty")
    rp_key = REPEAT_PENALTY_KEY.get(engine)
    if rp_key and rp is not None and rp != "":
        body[rp_key] = rp
    # Running figures on every chunk. Only for engines known to accept the
    # option: a strict OpenAI-compatible server may reject an unknown field.
    if engine == "llamacpp":
        body["timings_per_token"] = True
    elif engine == "vllm":
        body["stream_options"]["continuous_usage_stats"] = True
    return body


def compute_stats(*, t_start: float, t_first: Optional[float], t_last: Optional[float],
                  t_end: float, usage: Optional[Dict[str, Any]], timings: Optional[Dict[str, Any]],
                  delta_count: int, finish_reason: Optional[str], final: bool = True) -> ChatStats:
    """Turn what the stream has reported so far into prefill/generation figures."""
    ttft = (t_first - t_start) if t_first is not None else None
    total = t_end - t_start

    if timings:
        # llama-server: prompt_n excludes cache_n; *_ms and *_per_second are engine-side.
        p_ms = timings.get("prompt_ms")
        g_ms = timings.get("predicted_ms")
        return ChatStats(
            source="engine",
            prompt_tokens=timings.get("prompt_n"),
            cached_tokens=timings.get("cache_n"),
            prefill_s=p_ms / 1000 if p_ms is not None else None,
            prefill_tps=timings.get("prompt_per_second"),
            gen_tokens=timings.get("predicted_n"),
            gen_s=g_ms / 1000 if g_ms is not None else None,
            gen_tps=timings.get("predicted_per_second"),
            ttft_s=ttft, total_s=total, finish_reason=finish_reason, final=final,
        )

    usage = usage or {}
    prompt = usage.get("prompt_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    gen = usage.get("completion_tokens")
    approx = False
    if gen is None and delta_count:
        gen, approx = delta_count, True
    new_prompt = (prompt - (cached or 0)) if prompt is not None else None

    prefill_tps = None
    if new_prompt and ttft and ttft > 0:
        prefill_tps = new_prompt / ttft

    gen_s = gen_tps = None
    if t_first is not None and t_last is not None:
        gen_s = t_last - t_first
        # The first token is part of the TTFT window, so it doesn't count here.
        if gen and gen > 1 and gen_s > 0:
            gen_tps = (gen - 1) / gen_s

    return ChatStats(
        source="measured",
        prompt_tokens=new_prompt, cached_tokens=cached,
        prefill_s=ttft, prefill_tps=prefill_tps,
        gen_tokens=gen, gen_s=gen_s, gen_tps=gen_tps,
        ttft_s=ttft, total_s=total, finish_reason=finish_reason,
        approx=approx, final=final,
    )


def _line(obj: Dict[str, Any]) -> str:
    return json.dumps(obj) + "\n"


class EndpointError(ValueError):
    """The typed server:port can't be parsed or isn't an allowed address."""


def _allowed_ip(ip: ipaddress._BaseAddress) -> bool:
    # Link-local is excluded even though Python counts it private: it holds
    # cloud metadata services, never a model server.
    return (ip.is_loopback or ip.is_private) and not ip.is_link_local


async def resolve_endpoint(text: str) -> Tuple[str, str]:
    """Parse "host:port" (scheme and a trailing /v1 tolerated) into a base URL.

    Returns (base_url, label): base_url targets the resolved IP and is what
    requests use; label is the normalized "host:port" to show the user.
    """
    raw = (text or "").strip()
    if not raw:
        raise EndpointError("Enter a server as host:port, e.g. localhost:8080")
    if "://" not in raw:
        raw = "http://" + raw
    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https"):
        raise EndpointError(f"Unsupported scheme: {parts.scheme}")
    if parts.path.rstrip("/") not in ("", "/v1"):
        raise EndpointError("Give just host:port — the chat paths are fixed")
    host = parts.hostname
    try:
        port = parts.port
    except ValueError:
        raise EndpointError("Port must be a number between 1 and 65535")
    if not host or not port:
        raise EndpointError("Enter a server as host:port, e.g. localhost:8080")

    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, port, 0, socket.SOCK_STREAM)
    except socket.gaierror:
        raise EndpointError(f"Can't resolve host: {host}")
    ips = [ipaddress.ip_address(info[4][0]) for info in infos]
    allowed = [ip for ip in ips if _allowed_ip(ip)]
    if not allowed or len(allowed) != len(ips):
        raise EndpointError(f"{host} isn't a local or private-network address; "
                            "the chat page only connects to servers on your LAN")
    ip = allowed[0]
    ip_str = f"[{ip}]" if ip.version == 6 else str(ip)
    label_host = f"[{host}]" if ":" in host else host
    return f"{parts.scheme}://{ip_str}:{port}", f"{label_host}:{port}"


async def probe(base_url: str, transport=None) -> Dict[str, Any]:
    """Ask a server what it serves: model ids and which engine it is.

    Engine comes from /v1/models' owned_by ("vllm" / "llamacpp"); failing that,
    a llama-server-only /props endpoint identifies llama.cpp. Anything else is
    reported as "openai" (generic OpenAI-compatible).
    """
    async with httpx.AsyncClient(timeout=5.0, transport=transport) as client:
        r = await client.get(f"{base_url}/v1/models")
        r.raise_for_status()
        data = r.json().get("data") or []
        models = [m.get("id") for m in data if m.get("id")]
        owners = {str(m.get("owned_by", "")).lower() for m in data}
        if "vllm" in owners:
            engine = "vllm"
        elif "llamacpp" in owners:
            engine = "llamacpp"
        else:
            try:
                props = await client.get(f"{base_url}/props")
                engine = "llamacpp" if props.status_code == 200 else "openai"
            except httpx.HTTPError:
                engine = "openai"
    return {"models": models, "engine": engine}


async def stream_chat(base_url: str, payload: Dict[str, Any],
                      clock=time.perf_counter, transport=None) -> AsyncIterator[str]:
    """Forward one chat request and yield normalized NDJSON lines.

    If the browser disconnects, Starlette cancels this generator; leaving the
    `async with` closes the upstream connection, and both engines abort
    generation when the client goes away.
    """
    url = f"{base_url}/v1/chat/completions"
    t_start = clock()
    t_first = t_last = None
    usage = timings = finish_reason = None
    delta_count = 0
    last_progress = None

    def snapshot(final: bool, t_end: float) -> str:
        return _line(compute_stats(
            t_start=t_start, t_first=t_first, t_last=t_last, t_end=t_end,
            usage=usage, timings=timings, delta_count=delta_count,
            finish_reason=finish_reason, final=final).to_dict())

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, transport=transport) as client:
            async with client.stream("POST", url, json=payload) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")
                    yield _line({"type": "error", "message": _error_text(resp.status_code, body)})
                    return

                async for raw in resp.aiter_lines():
                    if not raw.startswith("data:"):
                        continue
                    data = raw[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    if chunk.get("error"):
                        err = chunk["error"]
                        yield _line({"type": "error",
                                     "message": err.get("message", str(err)) if isinstance(err, dict) else str(err)})
                        return
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    if chunk.get("timings"):
                        timings = chunk["timings"]

                    for choice in chunk.get("choices") or []:
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                        delta = choice.get("delta") or {}
                        # vLLM has used both field names across versions.
                        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                        content = delta.get("content")
                        if reasoning or content:
                            now = clock()
                            if t_first is None:
                                t_first = now
                            t_last = now
                            delta_count += 1
                        if reasoning:
                            yield _line({"type": "reasoning", "text": reasoning})
                        if content:
                            yield _line({"type": "content", "text": content})

                    if t_last is not None:
                        now = clock()
                        if last_progress is None or now - last_progress >= PROGRESS_INTERVAL_S:
                            last_progress = now
                            yield snapshot(False, now)
    except httpx.ConnectError:
        yield _line({"type": "error", "message": f"Could not connect to {base_url}. Is the server still running?"})
        return
    except httpx.HTTPError as e:
        yield _line({"type": "error", "message": f"Stream from the model failed: {type(e).__name__}: {e}"})
        return

    yield snapshot(True, clock())


def _error_text(status: int, body: str) -> str:
    """Pull the message out of an OpenAI-style error body when there is one."""
    try:
        parsed = json.loads(body)
        err = parsed.get("error", parsed)
        msg = err.get("message") if isinstance(err, dict) else str(err)
        if msg:
            return f"HTTP {status}: {msg}"
    except (json.JSONDecodeError, AttributeError):
        pass
    return f"HTTP {status}: {body[:500]}"
