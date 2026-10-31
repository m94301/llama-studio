"""engine_flags.flag_manifest — the one per-binary flag lookup for both engines.

Uses a fake llama-server (a shell script) so the probe, the disk cache, the
in-memory memo and rebuild detection are exercised for real.
"""

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import engine_flags  # noqa: E402
from engine_flags import flag_manifest, install_vllm_parse_manifests, validate_llama_option  # noqa: E402
import launch_script as L  # noqa: E402

FAKE_LLAMA = """#!/bin/sh
echo run >> "{calls}"
case "$1" in
  --version) echo "version: 0.4.1-dev (build {build}, commit abc1234)" ;;
  -h) cat <<'EOF'
  -c,    --ctx-size N                     size of the prompt context (default: 4096)
  -ngl,  --gpu-layers N                   number of layers to store in VRAM
         --no-mmap                        do not memory-map model
EOF
  ;;
esac
"""


@pytest.fixture(autouse=True)
def fresh_memo():
    engine_flags._MEM.clear()
    yield
    engine_flags._MEM.clear()


def _fake_llama(tmp_path, build=11067):
    binary = tmp_path / "llama-server"
    calls = tmp_path / "calls.log"
    binary.write_text(FAKE_LLAMA.format(calls=calls, build=build))
    binary.chmod(0o755)
    return binary, calls


def _calls(calls: Path) -> int:
    return len(calls.read_text().splitlines()) if calls.exists() else 0


class TestLlamaCpp:
    def test_probe_returns_version_and_flags_and_caches_to_disk(self, tmp_path):
        binary, _ = _fake_llama(tmp_path)
        m = flag_manifest("llamacpp", str(binary), tmp_path / "config")
        assert m["version"] == "11067_abc1234"
        assert m["flags"]["--ctx-size"]["type"] == "int"
        assert "-c" in m["flags"] and "--no-mmap" in m["flags"]
        from schema_parser import SCHEMA_FORMAT
        assert (tmp_path / "config" / "llama-server" / f"11067_abc1234.f{SCHEMA_FORMAT}.json").exists()

    def test_second_lookup_does_not_rerun_the_binary(self, tmp_path):
        binary, calls = _fake_llama(tmp_path)
        flag_manifest("llamacpp", str(binary), tmp_path / "config")
        before = _calls(calls)
        flag_manifest("llamacpp", str(binary), tmp_path / "config")
        assert _calls(calls) == before

    def test_binary_rebuilt_in_place_is_reread(self, tmp_path):
        binary, _ = _fake_llama(tmp_path, build=11067)
        assert flag_manifest("llamacpp", str(binary), tmp_path / "config")["version"] == "11067_abc1234"
        _fake_llama(tmp_path, build=11100)
        later = time.time() + 5
        os.utime(binary, (later, later))
        assert flag_manifest("llamacpp", str(binary), tmp_path / "config")["version"] == "11100_abc1234"

    def test_missing_binary_is_none(self, tmp_path):
        assert flag_manifest("llamacpp", str(tmp_path / "nope"), tmp_path / "config") is None
        assert flag_manifest("llamacpp", None, tmp_path / "config") is None


class TestValidateLlamaOption:
    FLAGS = {"--ctx-size": {"type": "int"}, "--alias": {"type": "str"}}

    def test_int_option(self):
        assert validate_llama_option("--ctx-size", "8192", self.FLAGS) == (True, "")
        assert validate_llama_option("--ctx-size", "big", self.FLAGS)[0] is False

    def test_unknown_option_or_no_schema_is_accepted(self):
        assert validate_llama_option("--brand-new", "x", self.FLAGS) == (True, "")
        assert validate_llama_option("--ctx-size", "big", None) == (True, "")


class TestVllmParseResolver:
    def test_falls_back_to_the_default_engine(self, tmp_path, monkeypatch):
        default = {"version": "d", "flags": {"--x": {"takes_value": True}}, "aliases": {}}
        lookups = []

        def fake(engine, binary, config_dir):
            lookups.append(binary)
            return default if binary == "/default/vllm" else None

        monkeypatch.setattr(engine_flags, "flag_manifest", fake)
        try:
            install_vllm_parse_manifests(tmp_path, lambda: "/default/vllm")
            assert L.manifest_for_binary("/custom/vllm") is default
            assert lookups == ["/custom/vllm", "/default/vllm"]
        finally:
            L.set_vllm_manifest_resolver(None)


class TestLlamaHelpParsing:
    """schema_parser._parse_help_text on help layouts the first parser mishandled."""

    HELP = (
        "-c,    --ctx-size N                     size of the prompt context (default: 4096)\n"
        "                                        (env: LLAMA_ARG_CTX_SIZE)\n"
        "-kvo,  --kv-offload, -nkvo, --no-kv-offload\n"
        "                                        whether to enable KV cache offloading (default: enabled)\n"
        "--spec-draft-ngl, -ngld, --gpu-layers-draft, --n-gpu-layers-draft N\n"
        "                                        max. number of draft model layers to store in VRAM\n"
        "-hf,   -hfr, --hf-repo <user>/<model>[:quant]\n"
        "                                        Hugging Face model repository\n"
        "--spec-ngram-mod-n-min N                minimum number of ngram tokens to use\n"
        "--draft-min, --draft-n-min N            the argument has been removed. use --spec-draft-n-min or\n"
        "                                        --spec-ngram-mod-n-min\n"
    )

    @pytest.fixture
    def schema(self):
        from schema_parser import _parse_help_text
        return _parse_help_text(self.HELP)

    def test_inline_description_unchanged(self, schema):
        assert schema["--ctx-size"] == {"type": "int", "description": "size of the prompt context", "category": "context"}
        assert schema["-c"] is schema["--ctx-size"]

    def test_every_spelling_on_a_line_is_kept(self, schema):
        for key in ("--spec-draft-ngl", "-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft"):
            assert schema[key]["type"] == "int", key
        for key in ("-kvo", "--kv-offload", "-nkvo", "--no-kv-offload"):
            assert key in schema, key

    def test_description_on_the_next_line(self, schema):
        assert schema["--kv-offload"]["description"] == "whether to enable KV cache offloading"
        assert schema["--hf-repo"]["description"] == "Hugging Face model repository"

    def test_wrapped_description_mentioning_a_flag_is_not_an_option(self, schema):
        assert schema["--spec-ngram-mod-n-min"]["description"] == "minimum number of ngram tokens to use"
