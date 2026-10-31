"""vLLM scripts are parsed against the manifest of the engine named in their own exec line."""
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import engine_flags  # noqa: E402
import launch_script as L  # noqa: E402

DEFAULT = {"version": "default", "flags": {"--host": {"takes_value": True}, "--port": {"takes_value": True}}, "aliases": {}}
CUSTOM = {"version": "custom", "flags": {**DEFAULT["flags"], "--only-in-custom": {"takes_value": False}}, "aliases": {}}

SCRIPT = textwrap.dedent('''\
    #!/usr/bin/env bash
    # === llama-studio:meta BEGIN ===
    # display_name: t
    # engine: vllm
    # === llama-studio:meta END ===
    export CUDA_VISIBLE_DEVICES=0
    exec "/engines/custom/bin/vllm" serve /models/m --host 0.0.0.0 --port 8000 --only-in-custom
''')


def _reset():
    L.set_vllm_manifest(None)
    L.set_vllm_manifest_resolver(None)


def test_script_uses_its_own_engines_manifest():
    try:
        L.set_vllm_manifest(DEFAULT)
        L.set_vllm_manifest_resolver(lambda b: CUSTOM if b == "/engines/custom/bin/vllm" else None)
        ps = L.parse_script(SCRIPT)
        assert not any("unknown vllm option" in w for w in ps.warnings), ps.warnings
        assert "--only-in-custom" in ps.args
    finally:
        _reset()


def test_falls_back_to_default_manifest_without_resolver_hit():
    try:
        L.set_vllm_manifest(DEFAULT)
        L.set_vllm_manifest_resolver(lambda b: None)   # capture failed / unknown binary
        ps = L.parse_script(SCRIPT)
        assert any("unknown vllm option" in w and "--only-in-custom" in w for w in ps.warnings), ps.warnings
    finally:
        _reset()


def test_resolver_exception_does_not_break_parsing():
    try:
        L.set_vllm_manifest(DEFAULT)
        def boom(b):
            raise RuntimeError("no help output")
        L.set_vllm_manifest_resolver(boom)
        ps = L.parse_script(SCRIPT)
        assert ps.model_path == "/models/m"
    finally:
        _reset()


def test_cache_key_distinguishes_engines_with_identical_version_strings(tmp_path):
    a = engine_flags._cache_path(tmp_path, "0.0.0+precompiled", "/engines/a/bin/vllm")
    b = engine_flags._cache_path(tmp_path, "0.0.0+precompiled", "/engines/b/bin/vllm")
    assert a != b
    assert a == engine_flags._cache_path(tmp_path, "0.0.0+precompiled", "/engines/a/bin/vllm")
