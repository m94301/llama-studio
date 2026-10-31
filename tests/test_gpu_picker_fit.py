"""Regression tests for per-GPU VRAM 'need' shown by the single-GPU picker.

Guards audit finding #1: the picker used to compute need as
`total_vram or size_gb`, which for a vLLM model (total_vram is None by design)
fell back to the weights size — under-reporting by ~60x, because vLLM claims
`--gpu-memory-utilization × the GPU's capacity` regardless of model size.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from config_manager import ModelConfig  # noqa: E402
from launch_script import parse_script  # noqa: E402
from templates import jinja_env  # noqa: E402


def _model(text: str, *, size_gb=None, total_vram=None) -> ModelConfig:
    mc = ModelConfig(name="m", script_path=Path("/tmp/m.sh"), parsed=parse_script(text))
    mc.size_gb = size_gb
    mc.total_vram = total_vram
    return mc


def _render(mc: ModelConfig, gpu_free_gb: float, gpu_total_gb: float = 64.0) -> str:
    """Render the picker the way /api/gpu-selector does, incl. its need math."""
    gpu_data = {
        0: {
            "name": "TestGPU",
            "memory": gpu_total_gb,
            "allocated": gpu_total_gb - gpu_free_gb,  # NVML real usage
            "memory_pct": (gpu_total_gb - gpu_free_gb) / gpu_total_gb * 100,
        }
    }
    need_by_gpu = {g: mc.need_per_gpu([i["memory"]])[0] for g, i in gpu_data.items()}
    return jinja_env.get_template("modals/gpu_selector.html").render(
        model_name="m", model_config=mc, display_name="M",
        gpu_data=gpu_data, need_by_gpu=need_by_gpu,
    )


VLLM = """#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# engine: vllm
# === llama-studio:meta END ===
exec /opt/venv/bin/vllm serve /models/tiny \\
  --host 0.0.0.0 --port 8999 --gpu-memory-utilization 0.9
"""

LLAMACPP = """#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# engine: llamacpp
# === llama-studio:meta END ===
LLAMA_BIN="/usr/bin/llama-server"
exec "$LLAMA_BIN" -m /models/x.gguf --host 0.0.0.0 --port 8100
"""


class TestVllmNeed:
    def test_need_is_util_times_capacity_not_model_size(self):
        # A 0.92 GB model that will still claim 0.9 * 64 = 57.6 GB.
        mc = _model(VLLM, size_gb=0.92)
        assert mc.needed_on_gpu(64.0) == pytest.approx(57.6)
        html = _render(mc, gpu_free_gb=5.0)
        assert "57.6 GB needed" in html
        assert "0.9 GB needed" not in html

    def test_blocks_load_when_gpu_lacks_the_full_claim(self):
        mc = _model(VLLM, size_gb=0.92)
        html = _render(mc, gpu_free_gb=5.0)  # plenty for weights, far too little for the claim
        assert "disabled" in html
        assert "Not enough VRAM" in html

    def test_allows_load_on_an_empty_gpu(self):
        mc = _model(VLLM, size_gb=0.92)
        html = _render(mc, gpu_free_gb=63.0)
        assert "disabled" not in html

    def test_need_scales_with_the_target_gpu(self):
        mc = _model(VLLM, size_gb=0.92)
        assert mc.needed_on_gpu(24.0) == pytest.approx(21.6)
        assert mc.needed_on_gpu(80.0) == pytest.approx(72.0)


class TestLlamacppNeedUnchanged:
    def test_uses_total_vram(self):
        mc = _model(LLAMACPP, size_gb=12.0, total_vram=20.0)
        html = _render(mc, gpu_free_gb=40.0)
        assert "20.0 GB needed" in html
        assert "disabled" not in html

    def test_falls_back_to_size_when_total_vram_unknown(self):
        mc = _model(LLAMACPP, size_gb=12.0, total_vram=None)
        html = _render(mc, gpu_free_gb=40.0)
        assert "12.0 GB needed" in html


def _llama(extra_args: str = "", **kw) -> ModelConfig:
    return _model(LLAMACPP.replace("--port 8100", "--port 8100 " + extra_args), **kw)


class TestNeedPerGpu:
    """ModelConfig.need_per_gpu is the one rule for pickers, CVD check and load."""

    def test_vllm_need_follows_each_gpu_capacity(self):
        mc = _model(VLLM, size_gb=0.92)
        assert mc.need_per_gpu([64.0, 24.0]) == pytest.approx([57.6, 21.6])

    def test_single_gpu_needs_the_whole_total(self):
        assert _llama(size_gb=12.0, total_vram=20.0).need_per_gpu([64.0]) == [20.0]

    def test_tensor_split_follows_the_weights(self):
        mc = _llama("--tensor-split 3,1", size_gb=12.0, total_vram=20.0)
        assert mc.need_per_gpu([64.0, 64.0]) == [15.0, 5.0]

    def test_tensor_split_with_unknown_total_splits_the_file_size(self):
        mc = _llama("--tensor-split 1,1", size_gb=12.0, total_vram=None)
        assert mc.need_per_gpu([64.0, 64.0]) == [6.0, 6.0]

    def test_flexible_split_divides_evenly_over_the_chosen_gpus(self):
        mc = _llama("--split-mode layer", size_gb=12.0, total_vram=30.0)
        assert mc.need_per_gpu([64.0, 64.0, 64.0]) == [10.0, 10.0, 10.0]
        assert mc.need_per_gpu([64.0]) == [30.0]
