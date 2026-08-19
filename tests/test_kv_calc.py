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
