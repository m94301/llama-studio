"""Tests for vLLM engine support in backend/launch_script.py (doc 21 Phase 1)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from launch_script import (  # noqa: E402
    parse_script,
    patch_meta_fence,
    render_vllm_skeleton_script,
    resolve_var_refs,
    META_BEGIN,
    META_END,
)


VLLM_SCRIPT = """#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# display_name: DeepSeek V4 Flash
# engine: vllm
# health_timeout: 900
# vram_gb: 380.5
# === llama-studio:meta END ===
set -euo pipefail
VENV=/opt/vllm/venv
export PATH="$VENV/bin:$PATH"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export VLLM_PP_LAYER_PARTITION="${PART:-15,16,12}"

MODEL="${DSV4_MODEL:-/models/dsv4-flash}"
MAXLEN="${MAXLEN:-524288}"
GPUUTIL="${GPUUTIL:-0.93}"
SPEC='{"method":"dspark","num_speculative_tokens":5}'

exec "$VENV/bin/vllm" serve "$MODEL" --served-model-name dsv4s DeepSeek-V4 "DeepSeek V4 Flash" \\
  --pipeline-parallel-size 3 --kv-cache-dtype fp8 \\
  --max-model-len "$MAXLEN" --trust-remote-code \\
  --gpu-memory-utilization "$GPUUTIL" \\
  --speculative-config "$SPEC" \\
  --host 0.0.0.0 --port 8098
"""


class TestVllmParse:
    def test_engine_and_meta(self):
        ps = parse_script(VLLM_SCRIPT)
        assert ps.engine == "vllm"
        assert ps.engine_declared is True
        assert ps.health_timeout == 900
        assert ps.vram_gb == 380.5

    def test_positional_model_with_var_resolution(self):
        ps = parse_script(VLLM_SCRIPT)
        assert ps.model_path == "/models/dsv4-flash"
        assert ps.llama_bin == "/opt/vllm/venv/bin/vllm"

    def test_args_and_port(self):
        ps = parse_script(VLLM_SCRIPT)
        assert ps.port == 8098
        assert ps.host == "0.0.0.0"
        assert ps.args["--max-model-len"] == "524288"  # self-referencing default
        assert ps.args["--kv-cache-dtype"] == "fp8"
        assert ps.args["--trust-remote-code"] is None  # boolean flag
        assert ps.args["--speculative-config"] == '{"method":"dspark","num_speculative_tokens":5}'

    def test_multi_value_flag_accumulates(self):
        ps = parse_script(VLLM_SCRIPT)
        assert ps.args["--served-model-name"] == "dsv4s DeepSeek-V4 DeepSeek V4 Flash"

    def test_gpu_count_and_util(self):
        ps = parse_script(VLLM_SCRIPT)
        assert ps.gpu_count == 3  # pp=3 × tp=1
        assert ps.tensor_parallel_size == 1
        assert ps.pipeline_parallel_size == 3
        assert ps.gpu_memory_utilization == 0.93
        assert ps.is_flexible_split is False

    def test_guarded_cvd(self):
        ps = parse_script(VLLM_SCRIPT)
        assert ps.cuda_visible_devices == [0, 1, 2]

    def test_pp_layer_partition(self):
        ps = parse_script(VLLM_SCRIPT)
        assert ps.pp_layer_partition == [15, 16, 12]
        assert not any("VLLM_PP_LAYER_PARTITION" in w for w in ps.warnings)

    def test_pp_partition_mismatch_warns(self):
        bad = VLLM_SCRIPT.replace("15,16,12", "15,16")
        ps = parse_script(bad)
        assert any("VLLM_PP_LAYER_PARTITION" in w for w in ps.warnings)

    def test_engine_inferred_without_meta(self):
        # A raw vllm script with no meta fence (e.g. dropped in from elsewhere)
        body = VLLM_SCRIPT.split(META_END)[1]
        ps = parse_script("#!/usr/bin/env bash" + body)
        assert ps.engine == "vllm"
        assert ps.engine_declared is False

    def test_invalid_engine_warns_and_falls_back(self):
        text = VLLM_SCRIPT.replace("# engine: vllm", "# engine: banana")
        ps = parse_script(text)
        assert ps.engine == "llamacpp" or ps.engine == "vllm"  # inferred from invocation
        assert any("banana" in w for w in ps.warnings)

    def test_is_configured_needs_port(self, tmp_path):
        model_dir = tmp_path / "dsv4-flash"
        model_dir.mkdir()
        text = VLLM_SCRIPT.replace("--port 8098", "").replace(
            "/models/dsv4-flash", str(model_dir)
        )
        ps = parse_script(text)
        ok, reason = ps.is_configured()
        assert not ok and "--port" in reason

    def test_is_configured_accepts_directory(self, tmp_path):
        model_dir = tmp_path / "dsv4-flash"
        model_dir.mkdir()
        text = VLLM_SCRIPT.replace("/models/dsv4-flash", str(model_dir)).replace(
            "--host 0.0.0.0 --port 8098",
            "--host 0.0.0.0 --port 8098 --max-num-seqs 8 --kv-cache-memory-bytes 1000000",
        )
        ps = parse_script(text)
        ok, reason = ps.is_configured()
        assert ok, reason

    def test_decision_7_blocks_a_script_without_a_kv_budget(self, tmp_path):
        """The reference DSV4 script pre-dates decision 7 and is now unconfigured
        until it states concurrency and a KV budget."""
        model_dir = tmp_path / "dsv4-flash"
        model_dir.mkdir()
        ps = parse_script(VLLM_SCRIPT.replace("/models/dsv4-flash", str(model_dir)))
        ok, reason = ps.is_configured()
        assert not ok and "--max-num-seqs" in reason


class TestLlamacppUnchanged:
    LLAMA = """#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# display_name: Test
# block_count: 32
# === llama-studio:meta END ===

LLAMA_BIN="/usr/bin/llama-server"

exec "$LLAMA_BIN" \\
  -m /tmp/model.gguf \\
  --host 0.0.0.0 \\
  --port 8100
"""

    def test_undeclared_parses_as_llamacpp(self):
        ps = parse_script(self.LLAMA)
        assert ps.engine == "llamacpp"
        assert ps.engine_declared is False
        assert ps.model_path == "/tmp/model.gguf"

    def test_engine_backfill_via_patch(self):
        ps = parse_script(self.LLAMA)
        patched = patch_meta_fence(
            self.LLAMA,
            display_name=ps.display_name,
            block_count=ps.block_count,
            engine=ps.engine,
        )
        ps2 = parse_script(patched)
        assert ps2.engine_declared is True
        assert ps2.engine == "llamacpp"
        assert ps2.block_count == 32
        # Body untouched
        assert patched.split(META_END)[1] == self.LLAMA.split(META_END)[1]


class TestVarResolution:
    def test_plain_and_braced(self):
        text = 'FOO=bar\nBAZ="qux"\n'
        assert resolve_var_refs("$FOO", text) == "bar"
        assert resolve_var_refs("${BAZ}", text) == "qux"

    def test_default_used_when_unset(self):
        assert resolve_var_refs("${NOPE:-fallback}", "") == "fallback"

    def test_self_reference_uses_default(self):
        text = 'X="${X:-42}"\n'
        assert resolve_var_refs("$X", text) == "42"

    def test_unresolved_warns_and_stays_literal(self):
        warnings = []
        out = resolve_var_refs("$MISSING", "", warnings)
        assert out == "$MISSING"
        assert warnings


class TestVllmSkeleton:
    def test_skeleton_round_trips(self, tmp_path):
        model_dir = tmp_path / "my-model"
        model_dir.mkdir()
        text = render_vllm_skeleton_script(
            display_name="My Model",
            model_path=str(model_dir),
            vllm_bin="/opt/venv/bin/vllm",
            available_flags={"--host", "--port", "--gpu-memory-utilization"},
        )
        ps = parse_script(text)
        assert ps.engine == "vllm" and ps.engine_declared
        assert ps.model_path == str(model_dir)
        assert ps.health_timeout == 600
        assert ps.args["--gpu-memory-utilization"] == "0.9"
        # No port → unconfigured, matching the llama.cpp skeleton convention
        ok, reason = ps.is_configured()
        assert not ok and "--port" in reason

    def test_optional_flags_filtered_by_manifest(self, tmp_path):
        """A skeleton must never contain a flag absent from the engine's manifest."""
        model_dir = tmp_path / "m"
        model_dir.mkdir()
        # Manifest without --disable-log-requests (modern vllm builds)
        text = render_vllm_skeleton_script(
            display_name="M",
            model_path=str(model_dir),
            vllm_bin="/opt/venv/bin/vllm",
            available_flags={"--host", "--port", "--gpu-memory-utilization"},
        )
        assert "--disable-log-requests" not in text
        assert "--gpu-memory-utilization" in text
        # Manifest that does have it (older builds) → included
        text2 = render_vllm_skeleton_script(
            display_name="M",
            model_path=str(model_dir),
            vllm_bin="/opt/venv/bin/vllm",
            available_flags={"--gpu-memory-utilization", "--disable-log-requests"},
        )
        assert "--disable-log-requests" in text2

    def test_no_manifest_emits_no_optional_flags(self, tmp_path):
        model_dir = tmp_path / "m"
        model_dir.mkdir()
        text = render_vllm_skeleton_script(
            display_name="M",
            model_path=str(model_dir),
            vllm_bin="/opt/venv/bin/vllm",
            available_flags=None,
        )
        ps = parse_script(text)
        assert set(ps.args.keys()) == {"--host"}


class TestVarResolutionShellSemantics:
    """Audit finding #2: resolution picked the first match per quoting style,
    so which assignment won depended on quote characters rather than position."""

    def test_last_assignment_wins(self):
        assert resolve_var_refs("$MODEL", "MODEL=/old\nMODEL=/new\n") == "/new"

    @pytest.mark.parametrize("text,expected", [
        ('MODEL=/bare/first\nMODEL="/quoted/second"\n', "/quoted/second"),
        ('MODEL="/quoted/first"\nMODEL=/bare/second\n', "/bare/second"),
        ("MODEL='/single/first'\nMODEL=/bare/second\n", "/bare/second"),
        ('MODEL=/bare/first\nMODEL=\'/single/second\'\n', "/single/second"),
    ])
    def test_quoting_style_does_not_decide_the_winner(self, text, expected):
        assert resolve_var_refs("$MODEL", text) == expected

    def test_export_prefix_and_override(self):
        text = 'export MODEL="/first"\nexport MODEL="/second"\n'
        assert resolve_var_refs("$MODEL", text) == "/second"

    def test_override_of_a_guarded_default(self):
        # `MAXLEN="${MAXLEN:-1000}"` then a later hard override
        text = 'MAXLEN="${MAXLEN:-1000}"\nMAXLEN=2000\n'
        assert resolve_var_refs("$MAXLEN", text) == "2000"


class TestContextConcurrencyDerivation:
    """Context and concurrency are read from arguments, never metadata (decision 7)."""

    def _vllm(self, extra: str) -> str:
        return (f"#!/usr/bin/env bash\n{META_BEGIN}\n# engine: vllm\n{META_END}\n"
                f"exec /opt/venv/bin/vllm serve /models/m --host 0.0.0.0 --port 8000 {extra}\n")

    def _llama(self, extra: str) -> str:
        return (f"#!/usr/bin/env bash\n{META_BEGIN}\n# engine: llamacpp\n{META_END}\n"
                f'exec /usr/bin/llama-server -m /m.gguf --host 0.0.0.0 --port 8100 {extra}\n')

    def test_vllm_reads_its_flags(self):
        ps = parse_script(self._vllm("--max-model-len 32768 --max-num-seqs 4"))
        assert ps.context_length == 32768
        assert ps.concurrency == 4

    def test_vllm_kv_budget(self):
        ps = parse_script(self._vllm("--kv-cache-memory-bytes 402653184"))
        assert ps.kv_cache_memory_bytes == 402653184

    def test_unset_concurrency_is_none_not_a_guess(self):
        """Neither engine states a static default, so we must not invent one."""
        assert parse_script(self._vllm("--max-model-len 4096")).concurrency is None
        assert parse_script(self._llama("--ctx-size 4096")).concurrency is None

    def test_llamacpp_auto_parallel_means_unified_and_full_context(self):
        # server.cpp: n_parallel < 0 -> n_parallel = 4, kv_unified = true
        ps = parse_script(self._llama("--ctx-size 262144"))
        assert ps.kv_unified is True
        assert ps.context_length == 262144
        assert parse_script(self._llama("--ctx-size 4096 --parallel -1")).kv_unified is True

    def test_llamacpp_explicit_parallel_divides_the_pool(self):
        # llama-context.cpp: n_ctx_seq = n_ctx / n_seq_max when not unified
        ps = parse_script(self._llama("--ctx-size 131072 --parallel 4"))
        assert ps.kv_unified is False
        assert ps.concurrency == 4
        assert ps.context_length == 32768  # 131072 / 4

    def test_explicit_kv_unified_keeps_the_full_context(self):
        ps = parse_script(self._llama("--ctx-size 131072 --parallel 4 --kv-unified"))
        assert ps.kv_unified is True
        assert ps.context_length == 131072


class TestVllmRequiredArguments:
    """Decision 7: a vLLM model must state context, concurrency, and budget."""

    def _script(self, tmp_path, extra: str) -> str:
        d = tmp_path / "m"
        d.mkdir(exist_ok=True)
        return (f"#!/usr/bin/env bash\n{META_BEGIN}\n# engine: vllm\n{META_END}\n"
                f"exec /opt/venv/bin/vllm serve {d} --host 0.0.0.0 --port 8000 {extra}\n")

    @pytest.mark.parametrize("extra,missing", [
        ("--max-num-seqs 1 --kv-cache-memory-bytes 1000", "--max-model-len"),
        ("--max-model-len 4096 --kv-cache-memory-bytes 1000", "--max-num-seqs"),
        ("--max-model-len 4096 --max-num-seqs 1", "--kv-cache-memory-bytes"),
    ])
    def test_each_missing_argument_blocks_the_load(self, tmp_path, extra, missing):
        ok, reason = parse_script(self._script(tmp_path, extra)).is_configured()
        assert not ok and missing in reason

    def test_all_three_present_is_configured(self, tmp_path):
        ok, reason = parse_script(self._script(
            tmp_path, "--max-model-len 4096 --max-num-seqs 1 --kv-cache-memory-bytes 1000"
        )).is_configured()
        assert ok, reason

    def test_llamacpp_is_not_subject_to_the_requirement(self, tmp_path):
        """Requiring --parallel would unconfigure the existing fleet AND flip it
        out of unified KV, dividing tuned contexts. See the deviation note."""
        gguf = tmp_path / "m.gguf"
        gguf.write_bytes(b"x")
        text = (f"#!/usr/bin/env bash\n{META_BEGIN}\n# engine: llamacpp\n{META_END}\n"
                f"exec /usr/bin/llama-server -m {gguf} --host 0.0.0.0 --port 8100\n")
        ok, reason = parse_script(text).is_configured()
        assert ok, reason


class TestCvdWithTrailingComment:
    """A CVD line with a trailing comment must still be replaced, not duplicated.

    Real case: the DSV4 script carried `export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"   # 3 cards`.
    Selecting 4 GPUs added a second line and left the old one, after which the
    parser read the stale value and reported 3 GPUs for a 4-GPU model.
    """

    SCRIPT = (
        "#!/usr/bin/env bash\n"
        f"{META_BEGIN}\n# engine: vllm\n{META_END}\n"
        'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"   # 3 cards\n'
        "VLLM_BIN=/opt/venv/bin/vllm\n\n"
        'exec "$VLLM_BIN" serve /models/m --host 0.0.0.0 --port 8000\n'
    )

    def test_commented_line_is_replaced_not_duplicated(self):
        from launch_script import patch_cuda_visible_devices
        out = patch_cuda_visible_devices(self.SCRIPT, [0, 1, 2, 3])
        assert out.count("CUDA_VISIBLE_DEVICES") == 1
        assert parse_script(out).cuda_visible_devices == [0, 1, 2, 3]

    def test_plain_commented_line_is_parsed(self):
        text = self.SCRIPT.replace(
            'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"   # 3 cards',
            "export CUDA_VISIBLE_DEVICES=2,3   # the big cards",
        )
        assert parse_script(text).cuda_visible_devices == [2, 3]
