"""Tests for the vLLM flag manifest (audit #8 and the -tp/-pp bug it exposed).

The rule this enforces: what `vllm serve` accepts is read from the engine's own
--help, never from a list written from memory. A hardcoded flag set is what
broke a real launch (`--disable-log-requests` doesn't exist in every build),
and a hardcoded "vLLM has no short aliases" comment hid a worse bug: `-tp 2`
was parsed as an unknown flag, so a 2-GPU model reported a GPU count of 1.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from engine_flags import parse_vllm_help  # noqa: E402
from launch_script import parse_script, set_vllm_manifest  # noqa: E402


# Verbatim excerpt of `vllm serve --help=all` rendering, covering each shape.
HELP = """
Config Groups:
  positional arguments
  --enforce-eager, --no-enforce-eager
                        Whether to always use eager-mode PyTorch. If True, we
  --trust-remote-code, --no-trust-remote-code
                        Trust remote code (e.g., from HuggingFace) when
  --max-model-len MAX_MODEL_LEN
                        Model context length (prompt and output). If
  --gpu-memory-utilization GPU_MEMORY_UTILIZATION
                        The fraction of GPU memory to be used for the model
  --pipeline-parallel-size PIPELINE_PARALLEL_SIZE, -pp PIPELINE_PARALLEL_SIZE
                        Number of pipeline parallel groups. (default: 1)
  --tensor-parallel-size TENSOR_PARALLEL_SIZE, -tp TENSOR_PARALLEL_SIZE
                        Number of tensor parallel groups. (default: 1)
  --speculative-config SPECULATIVE_CONFIG, -sc SPECULATIVE_CONFIG
                        Speculative decoding configuration.
  --kda-prefill-backend {auto,triton,flashkda}
                        Backend selection.
  --tokenizer-mode ['auto', 'deepseek_v4', 'mistral', 'slow']
                        Tokenizer mode.
  --hf-token [HF_TOKEN]
                        HuggingFace token.
  --served-model-name SERVED_MODEL_NAME [SERVED_MODEL_NAME ...]
                        The model name(s) used in the API.
  --host HOST           Host name.
  --port PORT           Port number.
"""


@pytest.fixture
def manifest():
    return parse_vllm_help(HELP)


class TestHelpParsing:
    def test_boolean_flags(self, manifest):
        assert manifest["flags"]["--enforce-eager"]["takes_value"] is False
        assert manifest["flags"]["--trust-remote-code"]["takes_value"] is False

    def test_value_flags(self, manifest):
        for flag in ("--max-model-len", "--gpu-memory-utilization", "--host", "--port",
                     "--served-model-name", "--speculative-config"):
            assert manifest["flags"][flag]["takes_value"] is True, flag

    @pytest.mark.parametrize("flag", [
        "--kda-prefill-backend",  # {auto,triton,flashkda}
        "--tokenizer-mode",       # ['auto', 'deepseek_v4', ...]
        "--hf-token",             # [HF_TOKEN]
    ])
    def test_every_metavar_style_counts_as_a_value(self, manifest, flag):
        assert manifest["flags"][flag]["takes_value"] is True

    def test_short_aliases_map_to_long_forms(self, manifest):
        assert manifest["aliases"] == {
            "-pp": "--pipeline-parallel-size",
            "-tp": "--tensor-parallel-size",
            "-sc": "--speculative-config",
        }

    def test_negated_form_is_not_an_alias(self, manifest):
        """--no-x means the opposite of --x; collapsing them would invert config."""
        assert "--no-enforce-eager" not in manifest["aliases"]
        assert manifest["flags"]["--no-enforce-eager"]["takes_value"] is False


class TestParserUsesManifest:
    SCRIPT = """#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# engine: vllm
# === llama-studio:meta END ===
exec /opt/venv/bin/vllm serve /models/m -tp 2 -pp 3 \\
  --enforce-eager --max-model-len 4096 --host 0.0.0.0 --port 8000
"""

    def test_short_forms_normalize_and_count_gpus(self, manifest):
        set_vllm_manifest(manifest)
        try:
            ps = parse_script(self.SCRIPT)
            assert ps.args["--tensor-parallel-size"] == "2"
            assert ps.args["--pipeline-parallel-size"] == "3"
            assert ps.gpu_count == 6  # tp 2 x pp 3
        finally:
            set_vllm_manifest(None)

    def test_without_a_manifest_short_forms_are_not_understood(self):
        """Documents why the manifest matters: bare heuristics miscount GPUs."""
        set_vllm_manifest(None)
        ps = parse_script(self.SCRIPT)
        assert ps.gpu_count == 1  # -tp/-pp unrecognized
        assert "--tensor-parallel-size" not in ps.args

    def test_boolean_flag_does_not_swallow_the_next_value(self, manifest):
        set_vllm_manifest(manifest)
        try:
            ps = parse_script(self.SCRIPT)
            assert ps.args["--enforce-eager"] is None
            assert ps.args["--max-model-len"] == "4096"
            assert ps.port == 8000
        finally:
            set_vllm_manifest(None)

    def test_unknown_option_warns(self, manifest):
        set_vllm_manifest(manifest)
        try:
            ps = parse_script(self.SCRIPT.replace("--enforce-eager", "--totally-made-up"))
            assert any("unknown vllm option: --totally-made-up" in w for w in ps.warnings)
        finally:
            set_vllm_manifest(None)


VLLM_BIN = Path("/home/m6/projects/vllm-dsv4/venv/bin/vllm")


@pytest.mark.skipif(not VLLM_BIN.exists(), reason="local vllm build not present")
class TestAgainstTheRealBuild:
    """Known-truth check against the installed engine.

    If a vLLM upgrade changes the help format, this fails loudly instead of
    letting the parser silently misread every flag.
    """

    EXPECTED = {
        "--trust-remote-code": False, "--enforce-eager": False,
        "--enable-auto-tool-choice": False,
        "--gpu-memory-utilization": True, "--max-model-len": True,
        "--kv-cache-dtype": True, "--speculative-config": True,
        "--served-model-name": True, "--pipeline-parallel-size": True,
        "--tensor-parallel-size": True, "--tool-call-parser": True,
        "--block-size": True, "--host": True, "--port": True,
        # Quoted-choice-list metavar — the style that made the parser drop
        # `--tokenizer-mode deepseek_v4`'s value on the real DSV4 script.
        "--tokenizer-mode": True, "--distributed-executor-backend": True,
        "--config-format": True, "--model-impl": True,
    }

    @pytest.fixture(scope="class")
    def real(self, tmp_path_factory):
        from engine_flags import get_vllm_manifest
        m = get_vllm_manifest(str(VLLM_BIN), tmp_path_factory.mktemp("cfg"))
        if not m:
            pytest.skip("could not capture manifest from the local build")
        return m

    def test_known_flags_classified_correctly(self, real):
        wrong = {
            f: real["flags"].get(f, {}).get("takes_value", "MISSING")
            for f, expected in self.EXPECTED.items()
            if real["flags"].get(f, {}).get("takes_value", "MISSING") != expected
        }
        assert not wrong, f"misclassified: {wrong}"

    def test_real_short_aliases(self, real):
        assert real["aliases"].get("-tp") == "--tensor-parallel-size"
        assert real["aliases"].get("-pp") == "--pipeline-parallel-size"

    def test_manifest_is_substantial(self, real):
        assert len(real["flags"]) > 100
