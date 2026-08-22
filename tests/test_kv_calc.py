"""Tests for KV sizing and the context/concurrency/VRAM relation (doc 21 §2.4)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from kv_calc import (  # noqa: E402
    CONTEXT_GRANULARITY,
    DEFAULT_OVERHEAD_GB,
    GIB,
    bytes_per_token_from_config,
    calibrate_from_log,
    context_for_vram,
    pool_tokens,
    vram_gb_for,
)


def _write_config(tmp_path: Path, **overrides) -> str:
    cfg = {
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "hidden_size": 896,
        "torch_dtype": "bfloat16",
    }
    cfg.update(overrides)
    d = tmp_path / "model"
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps(cfg))
    return str(d)


class TestBytesPerTokenFromConfig:
    def test_matches_the_measured_qwen_figure(self, tmp_path):
        """Qwen2.5-0.5B: vLLM reported 55.42 GiB holding 4,842,496 tokens."""
        assert bytes_per_token_from_config(_write_config(tmp_path)) == 12288

    def test_head_dim_derived_when_absent(self, tmp_path):
        d = _write_config(tmp_path, hidden_size=896, num_attention_heads=14)
        Path(d, "config.json").write_text(
            json.dumps({"num_hidden_layers": 24, "num_attention_heads": 14,
                        "num_key_value_heads": 2, "hidden_size": 896,
                        "torch_dtype": "bfloat16"})
        )
        assert bytes_per_token_from_config(d) == 12288  # 896/14 = 64

    def test_kv_cache_dtype_overrides_model_dtype(self, tmp_path):
        d = _write_config(tmp_path)
        assert bytes_per_token_from_config(d, "fp8") == 6144  # half of bf16

    def test_auto_dtype_falls_back_to_model_dtype(self, tmp_path):
        assert bytes_per_token_from_config(_write_config(tmp_path), "auto") == 12288

    def test_nested_text_config(self, tmp_path):
        d = tmp_path / "mm"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"text_config": {
            "num_hidden_layers": 24, "num_key_value_heads": 2,
            "head_dim": 64, "torch_dtype": "bfloat16"}}))
        assert bytes_per_token_from_config(str(d)) == 12288

    def test_declines_on_mla_rather_than_guessing(self, tmp_path):
        """A compressed-latent KV isn't 2*kv_heads*head_dim; calibrate instead."""
        assert bytes_per_token_from_config(_write_config(tmp_path, kv_lora_rank=512)) is None

    def test_declines_when_unreadable(self, tmp_path):
        assert bytes_per_token_from_config(str(tmp_path / "nope")) is None


class TestCalibrationFromLog:
    LOG = """
    (EngineCore pid=1) INFO [gpu_worker.py:563] Available KV cache memory: 55.42 GiB
    (EngineCore pid=1) INFO [kv_cache_utils.py] GPU KV cache size: 4,842,496 tokens
    (EngineCore pid=1) INFO Maximum concurrency for 32,768 tokens per request: 147.78x
    """

    def test_reads_the_exact_figure(self):
        assert calibrate_from_log(self.LOG) == 12288

    def test_agrees_with_the_config_estimate(self, tmp_path):
        assert calibrate_from_log(self.LOG) == bytes_per_token_from_config(_write_config(tmp_path))

    @pytest.mark.parametrize("text", ["", "no numbers here",
                                      "Available KV cache memory: 55.42 GiB"])
    def test_none_when_incomplete(self, text):
        assert calibrate_from_log(text) is None


class TestForwardSolve:
    def test_single_conversation(self):
        total, tokens = vram_gb_for(
            context_length=32768, concurrency=1, bytes_per_token=12288,
            weights_gb=1.0, overhead_gb=0.5,
        )
        assert tokens == 32768
        assert total == pytest.approx(1.0 + 0.5 + 32768 * 12288 / GIB, abs=0.01)

    def test_concurrency_multiplies_the_pool(self):
        _, one = vram_gb_for(context_length=32768, concurrency=1,
                             bytes_per_token=12288, weights_gb=1.0)
        _, four = vram_gb_for(context_length=32768, concurrency=4,
                              bytes_per_token=12288, weights_gb=1.0)
        assert four == 4 * one

    def test_headroom_applies(self):
        assert pool_tokens(1000, 2, headroom=1.2) == 2400


class TestInverseSolve:
    ARGS = dict(bytes_per_token=12288, weights_gb=1.0, overhead_gb=0.5)

    def test_round_trip_returns_to_the_same_slider_position(self):
        """Acceptance #8: drag, read VRAM, type it back, land within one step."""
        for ctx in (8192, 32768, 131072):
            total, _ = vram_gb_for(context_length=ctx, concurrency=2, **self.ARGS)
            back = context_for_vram(vram_gb=total, concurrency=2, **self.ARGS)
            assert abs(back - ctx) <= CONTEXT_GRANULARITY

    def test_snaps_down_so_the_result_always_fits(self):
        ctx = context_for_vram(vram_gb=10.0, concurrency=1, **self.ARGS)
        assert ctx % CONTEXT_GRANULARITY == 0
        total, _ = vram_gb_for(context_length=ctx, concurrency=1, **self.ARGS)
        assert total <= 10.0

    def test_clamps_to_trained_context(self):
        assert context_for_vram(vram_gb=64.0, concurrency=1,
                                max_context=8192, **self.ARGS) == 8192

    def test_zero_when_budget_cannot_cover_weights(self):
        assert context_for_vram(vram_gb=1.0, concurrency=1, **self.ARGS) == 0

    def test_more_concurrency_means_less_context_for_the_same_budget(self):
        one = context_for_vram(vram_gb=20.0, concurrency=1, **self.ARGS)
        four = context_for_vram(vram_gb=20.0, concurrency=4, **self.ARGS)
        assert four == pytest.approx(one / 4, rel=0.01)


class TestCalibrationWithExplicitBudget:
    """With --kv-cache-memory-bytes set, vLLM stops printing 'Available KV cache
    memory', so the known budget over the reported token count is the source.
    This is the configuration decision 7 requires, so it must be the one that works."""

    LOG = "INFO [kv_cache_utils.py] GPU KV cache size: 65,536 tokens\n"

    def test_derives_from_the_budget(self):
        assert calibrate_from_log(self.LOG, 805306368) == 12288

    def test_none_without_a_budget_to_divide(self):
        assert calibrate_from_log(self.LOG) is None

    def test_memory_line_still_wins_when_present(self):
        text = ("Available KV cache memory: 55.42 GiB\n"
                "GPU KV cache size: 4,842,496 tokens\n")
        assert calibrate_from_log(text, 999) == 12288


class TestLoadRejectionIsTerminal:
    """A rejected load must not leave the row stuck in LOADING.

    load_model runs as a FastAPI BackgroundTask after trigger_load has already
    set LOADING, and a background task's exception is swallowed — so admission
    failures have to set FAILED themselves.
    """

    def test_rejection_sets_failed(self):
        import asyncio
        import sys
        from pathlib import Path as _P
        sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "backend"))
        from gpu_manager import GpuManager, ModelState

        class _Cfg:
            models = {}
            def get_all_models(self): return {}
            def get_model_config(self, name): raise FileNotFoundError(name)
            project_root = _P("/tmp")

        gm = GpuManager(_Cfg())
        asyncio.run(gm.load_model("nope", [0]))          # model not found
        assert gm.state_info["nope"].state == ModelState.FAILED
        assert "nope" in (gm.state_info["nope"].error_msg or "")


class TestDerivedUtilization:
    """--gpu-memory-utilization is a startup gate as well as a sizing knob:
    vLLM refuses to launch unless free >= util x capacity, however little it then
    uses. Deriving it from the need keeps the gate and real consumption in step."""

    def test_tracks_the_need_rather_than_the_card(self):
        from kv_calc import utilization_for
        # 52 GB needed on a 64 GB card -> ~0.85, not 0.98
        assert utilization_for(52.0, 64.0) == pytest.approx(0.853, abs=0.002)

    def test_a_small_model_does_not_claim_the_whole_card(self):
        from kv_calc import utilization_for
        util = utilization_for(2.2, 64.0)
        assert util < 0.05
        assert util * 64.0 >= 2.2  # still covers actual usage

    def test_covers_usage_with_margin(self):
        from kv_calc import UTILIZATION_SAFETY, utilization_for
        for need in (1.0, 10.0, 52.0):
            assert utilization_for(need, 64.0) * 64.0 >= need * (UTILIZATION_SAFETY - 0.01)

    def test_never_exceeds_the_maximum(self):
        from kv_calc import UTILIZATION_MAX, utilization_for
        assert utilization_for(999.0, 64.0) == UTILIZATION_MAX

    def test_never_zero(self):
        from kv_calc import utilization_for
        assert utilization_for(0.0, 64.0) > 0


class TestPickerMatchesTheStartupGate:
    """Regression for the failed 4-card DSV4 start: our picker asked for the
    predicted usage (52.12 GB) and passed a GPU with 62.07 GB free, while vLLM's
    gate demanded 62.24 GB and refused."""

    def _model(self, util, kv_bytes, size_gb, gpus=1):
        import sys
        from pathlib import Path as _P
        sys.path.insert(0, str(_P(__file__).resolve().parent.parent / "backend"))
        from config_manager import ModelConfig
        from launch_script import META_BEGIN, META_END, parse_script
        text = (
            f"#!/usr/bin/env bash\n{META_BEGIN}\n# engine: vllm\n{META_END}\n"
            f"exec /opt/venv/bin/vllm serve /m --host 0.0.0.0 --port 8000 "
            f"--gpu-memory-utilization {util} --kv-cache-memory-bytes {kv_bytes} "
            f"--pipeline-parallel-size {gpus}\n"
        )
        mc = ModelConfig(name="m", script_path=_P("/tmp/m.sh"), parsed=parse_script(text))
        mc.size_gb = size_gb
        return mc

    def test_hand_set_high_util_is_demanded_in_full(self):
        mc = self._model(0.98, 1_000_000, size_gb=155.4, gpus=4)
        assert mc.needed_on_gpu(63.51) == pytest.approx(0.98 * 63.51, abs=0.01)

    def test_derived_util_demands_only_what_is_used(self):
        from kv_calc import GIB, utilization_for
        kv = int(3.0 * GIB)
        usage = 2.0 + 1.5 + 3.0
        util = utilization_for(usage, 64.0)
        mc = self._model(util, kv, size_gb=2.0)
        need = mc.needed_on_gpu(64.0)
        assert need >= usage           # covers the load
        assert need < usage * 1.15     # without reserving the card


class TestPerGpuKvSemantics:
    """--kv-cache-memory-bytes is 'Size of KV Cache per GPU' — only the weights
    divide across cards. Dividing the KV too under-predicted a 4-card DSV4 at
    42 GB per GPU when it actually used 52-53."""

    # The real 4-card DSV4 load: 262k context, budget sized from the 44,032
    # bytes/token estimate that was current at the time.
    ARGS = dict(context_length=262144, concurrency=1, bytes_per_token=44032,
                weights_gb=155.4, overhead_gb=1.5, headroom=1.2)

    def test_matches_the_measured_four_card_load(self):
        per_gpu, tokens = vram_gb_for(gpu_count=4, **self.ARGS)
        assert tokens == 314572
        assert per_gpu == pytest.approx(52.5, abs=1.5)   # measured 51.2-53.3

    def test_only_weights_divide(self):
        one, _ = vram_gb_for(gpu_count=1, **self.ARGS)
        four, _ = vram_gb_for(gpu_count=4, **self.ARGS)
        kv_plus_overhead = 314572 * 44032 / GIB + 1.5
        assert one - four == pytest.approx(155.4 - 155.4 / 4, abs=0.05)
        assert four - 155.4 / 4 == pytest.approx(kv_plus_overhead, abs=0.05)

    def test_inverse_accounts_for_the_split_too(self):
        per_gpu, _ = vram_gb_for(gpu_count=4, **self.ARGS)
        back = context_for_vram(
            vram_gb=per_gpu, concurrency=1, bytes_per_token=44032,
            weights_gb=155.4, overhead_gb=1.5, headroom=1.2, gpu_count=4,
        )
        assert abs(back - 262144) <= CONTEXT_GRANULARITY


class TestMeasuredLoadProfile:
    """Pipeline ranks are not equal — vLLM's own log showed 37.07 / 37.09 /
    38.30 / 46.19 GiB on a 4-card DSV4. Predicting against the average
    under-shot the card that actually decides the fit by ~9 GiB."""

    LOG = """
    (Worker_PP0) INFO Model loading took 37.07 GiB and 60 seconds
    (Worker_PP1) INFO Model loading took 37.09 GiB and 61 seconds
    (Worker_PP2) INFO Model loading took 38.30 GiB and 59 seconds
    (Worker_PP3) INFO Model loading took 46.19 GiB and 62 seconds
    """

    def test_reads_the_heaviest_rank(self):
        from kv_calc import heaviest_rank_weights_from_log
        assert heaviest_rank_weights_from_log(self.LOG) == 46.19

    def test_none_without_the_lines(self):
        from kv_calc import heaviest_rank_weights_from_log
        assert heaviest_rank_weights_from_log("nothing here") is None

    def test_measured_weights_override_the_even_split(self):
        args = dict(context_length=1048576, concurrency=1, bytes_per_token=2252,
                    weights_gb=155.4, overhead_gb=4.01, headroom=1.2, gpu_count=4)
        even, _ = vram_gb_for(**args)
        measured, _ = vram_gb_for(**args, weights_per_gpu_gb=46.19)
        assert even == pytest.approx(45.5, abs=0.5)      # 155.4/4 + 4.01 + 2.64
        assert measured == pytest.approx(52.84, abs=0.1)  # heaviest rank
        assert measured > even

    def test_inverse_uses_the_measured_share_too(self):
        common = dict(concurrency=1, bytes_per_token=2252, weights_gb=155.4,
                      overhead_gb=4.01, headroom=1.2, gpu_count=4,
                      weights_per_gpu_gb=46.19)
        per_gpu, _ = vram_gb_for(context_length=1048576, **common)
        back = context_for_vram(vram_gb=per_gpu, **common)
        assert abs(back - 1048576) <= CONTEXT_GRANULARITY


class TestHeadroomFloor:
    """A config must leave MIN_HEADROOM_GB free on the tightest card.

    The 640K attempt projected under a gigabyte of headroom, drove rank 1 to
    99.2%, and left the GPU unresponsive to NVML until the host rebooted — the
    one hardware incident of the tuning session. This is the guard that would
    have refused to write it.
    """

    def test_accepts_the_verified_768k_config(self):
        from kv_calc import headroom_ok
        ok, free = headroom_ok(62.06, 64.0)      # measured peak at 768K
        assert ok and free == 1.94

    def test_refuses_the_config_that_wedged_the_gpu(self):
        from kv_calc import headroom_ok
        ok, free = headroom_ok(63.5, 64.0)       # measured at the 640K failure
        assert not ok and free < 1.5

    def test_boundary_is_inclusive(self):
        from kv_calc import MIN_HEADROOM_GB, headroom_ok
        ok, _ = headroom_ok(64.0 - MIN_HEADROOM_GB, 64.0)
        assert ok

    def test_smaller_cards_scale(self):
        from kv_calc import headroom_ok
        assert headroom_ok(22.0, 24.0)[0]        # 2.0 free on a 24 GB card
        assert not headroom_ok(23.0, 24.0)[0]    # 1.0 free
