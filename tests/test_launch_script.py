"""Tests for backend/launch_script.py — parser, renderer, fence patchers."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from launch_script import (  # noqa: E402
    parse_script,
    parse_pasted_script,
    render_script,
    render_skeleton_script,
    patch_meta_fence,
    patch_launch_args_fence,
    patch_cuda_visible_devices,
    META_BEGIN,
    META_END,
    ARGS_BEGIN,
    ARGS_END,
)


# ---------- helpers ----------

CANONICAL = """#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# display_name: Qwen 3.6 27B Q4_K_M
# block_count: 64
# max_context: 262144
# kv_cache_multiplier: 213
# === llama-studio:meta END ===

LLAMA_BIN=/home/m6/servers/llamacpp/bin/llama-server

# === llama-studio:launch-args BEGIN ===
exec "$LLAMA_BIN" \\
  -m /home/m6/models/Qwen3.6-27B-Q4_K_M.gguf \\
  --host 0.0.0.0 \\
  --port 8100 \\
  --ctx-size 262144 \\
  --cache-type-k q8_0 \\
  --cache-type-v q8_0 \\
  --no-mmap
# === llama-studio:launch-args END ===
"""


# ---------- meta fence ----------

def test_meta_fence_parses():
    ps = parse_script(CANONICAL)
    assert ps.has_meta_fence is True
    assert ps.display_name == "Qwen 3.6 27B Q4_K_M"
    assert ps.block_count == 64
    assert ps.max_context == 262144
    assert ps.kv_cache_multiplier == 213


def test_meta_fence_missing_warns():
    text = CANONICAL.replace(META_BEGIN, "# (no meta here)").replace(META_END, "# (end)")
    ps = parse_script(text)
    assert ps.has_meta_fence is False
    assert any("meta fence missing" in w for w in ps.warnings)


# ---------- launch-args fence ----------

def test_args_fence_parses_canonical():
    ps = parse_script(CANONICAL)
    assert ps.has_args_fence is True
    assert ps.llama_bin == "/home/m6/servers/llamacpp/bin/llama-server"
    assert ps.model_path == "/home/m6/models/Qwen3.6-27B-Q4_K_M.gguf"
    assert ps.host == "0.0.0.0"
    assert ps.port == 8100
    assert ps.args["--ctx-size"] == "262144"
    assert ps.args["--cache-type-k"] == "q8_0"
    assert ps.args["--no-mmap"] is None  # flag-only


def test_args_short_form_aliases():
    text = CANONICAL.replace("--ctx-size 262144", "-c 262144")
    ps = parse_script(text)
    assert ps.args.get("--ctx-size") == "262144"
    assert "-c" not in ps.args  # normalized to long form


def test_args_flag_only_at_end():
    text = CANONICAL.replace("--no-mmap", "--direct-io")
    ps = parse_script(text)
    assert ps.args["--direct-io"] is None


def test_args_chat_template_kwargs_with_json_value():
    text = CANONICAL.replace(
        "--no-mmap",
        "--chat-template-kwargs '{\"enable_thinking\":true}'",
    )
    ps = parse_script(text)
    assert ps.args["--chat-template-kwargs"] == '{"enable_thinking":true}'


def test_llama_bin_var_resolves_from_assignment():
    ps = parse_script(CANONICAL)
    assert ps.llama_bin == "/home/m6/servers/llamacpp/bin/llama-server"


def test_llama_bin_var_quoted_assignment():
    text = CANONICAL.replace(
        "LLAMA_BIN=/home/m6/servers/llamacpp/bin/llama-server",
        'LLAMA_BIN="/opt/llama cpp/llama-server"',
    )
    ps = parse_script(text)
    assert ps.llama_bin == "/opt/llama cpp/llama-server"


# ---------- configured check ----------

def test_is_configured_happy(tmp_path):
    model_file = tmp_path / "m.gguf"
    model_file.write_text("fake")
    text = CANONICAL.replace(
        "/home/m6/models/Qwen3.6-27B-Q4_K_M.gguf",
        str(model_file),
    )
    ps = parse_script(text)
    ok, reason = ps.is_configured()
    assert ok, reason


def test_is_configured_missing_port(tmp_path):
    model_file = tmp_path / "m.gguf"
    model_file.write_text("fake")
    text = CANONICAL.replace(
        "/home/m6/models/Qwen3.6-27B-Q4_K_M.gguf", str(model_file)
    ).replace("  --port 8100 \\\n", "")
    ps = parse_script(text)
    ok, reason = ps.is_configured()
    assert not ok
    assert "port" in reason.lower()


def test_is_configured_missing_model_file():
    ps = parse_script(CANONICAL)
    ok, reason = ps.is_configured()
    assert not ok
    assert "not found" in reason or "missing" in reason.lower()


# ---------- pasted scripts (Unsloth-style) ----------

UNSLOTH = """./llama.cpp/llama-server \\
    --model unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \\
    --mmproj unsloth/gemma-4-26B-A4B-it-GGUF/mmproj-BF16.gguf \\
    --temp 1.0 \\
    --top-p 0.95 \\
    --top-k 64 \\
    --alias "unsloth/gemma-4-26B-A4B-it-GGUF" \\
    --port 8001 \\
    --chat-template-kwargs '{"enable_thinking":true}'
"""


def test_pasted_unsloth_parses():
    ps = parse_pasted_script(UNSLOTH)
    assert ps.llama_bin == "./llama.cpp/llama-server"
    assert ps.model_path == "unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
    assert ps.args["--temp"] == "1.0"
    assert ps.args["--top-p"] == "0.95"
    assert ps.args["--top-k"] == "64"
    assert ps.args["--port"] == "8001"
    assert ps.args["--mmproj"] == "unsloth/gemma-4-26B-A4B-it-GGUF/mmproj-BF16.gguf"
    assert ps.args["--chat-template-kwargs"] == '{"enable_thinking":true}'


def test_pasted_unsloth_alias_becomes_display_name():
    ps = parse_pasted_script(UNSLOTH)
    assert ps.display_name == "unsloth/gemma-4-26B-A4B-it-GGUF"


def test_pasted_no_binary_token():
    text = "--port 8080 --ctx-size 4096 --no-mmap"
    ps = parse_pasted_script(text)
    assert ps.args["--port"] == "8080"
    assert ps.args["--no-mmap"] is None


# ---------- render / round-trip ----------

def test_render_basic():
    text = render_script(
        display_name="Test",
        block_count=64,
        max_context=262144,
        kv_cache_multiplier=213,
        llama_bin="/bin/llama-server",
        model_path="/m.gguf",
        args={"--host": "0.0.0.0", "--port": "8100", "--ctx-size": "4096", "--no-mmap": None},
    )
    assert "#!/usr/bin/env bash" in text
    assert META_BEGIN in text
    assert META_END in text
    # New format: no launch-args fence markers
    assert ARGS_BEGIN not in text
    assert ARGS_END not in text
    assert "display_name: Test" in text
    assert "-m /m.gguf" in text
    assert "--port 8100" in text
    assert "--no-mmap" in text
    assert 'exec "$LLAMA_BIN"' in text


def test_render_then_parse_roundtrip():
    args_in = {
        "--host": "0.0.0.0",
        "--port": "8100",
        "--ctx-size": "262144",
        "--cache-type-k": "q8_0",
        "--cache-type-v": "q8_0",
        "--no-mmap": None,
    }
    text = render_script(
        display_name="X",
        block_count=64,
        max_context=262144,
        kv_cache_multiplier=213,
        llama_bin="/bin/llama-server",
        model_path="/m.gguf",
        args=args_in,
    )
    ps = parse_script(text)
    assert ps.display_name == "X"
    assert ps.block_count == 64
    assert ps.model_path == "/m.gguf"
    assert ps.port == 8100
    for k, v in args_in.items():
        assert ps.args.get(k) == v


def test_render_quotes_paths_with_spaces():
    text = render_script(
        display_name="X",
        block_count=None,
        max_context=None,
        kv_cache_multiplier=None,
        llama_bin="/bin/llama-server",
        model_path="/path with spaces/m.gguf",
        args={"--port": "8100"},
    )
    ps = parse_script(text)
    assert ps.model_path == "/path with spaces/m.gguf"


def test_render_preserves_json_arg_value():
    args_in = {"--port": "8100", "--chat-template-kwargs": '{"enable_thinking":true}'}
    text = render_script(
        display_name="X",
        block_count=None,
        max_context=None,
        kv_cache_multiplier=None,
        llama_bin="/bin/llama-server",
        model_path="/m.gguf",
        args=args_in,
    )
    ps = parse_script(text)
    assert ps.args["--chat-template-kwargs"] == '{"enable_thinking":true}'


def test_skeleton_has_no_port(tmp_path):
    model_file = tmp_path / "m.gguf"
    model_file.write_text("fake")
    text = render_skeleton_script(
        display_name="Skel",
        model_path=str(model_file),
        llama_bin="/bin/llama-server",
        block_count=32,
        max_context=4096,
        kv_cache_multiplier=128,
    )
    ps = parse_script(text)
    assert ps.port is None
    ok, reason = ps.is_configured()
    assert not ok
    assert "port" in reason.lower()


# ---------- fence patching ----------

def test_patch_meta_fence_updates_in_place():
    text = render_script(
        display_name="Old",
        block_count=1,
        max_context=2,
        kv_cache_multiplier=3,
        llama_bin="/bin/llama-server",
        model_path="/m.gguf",
        args={"--port": "8100"},
    )
    new = patch_meta_fence(
        text,
        display_name="New",
        block_count=10,
        max_context=20,
        kv_cache_multiplier=30,
    )
    ps = parse_script(new)
    assert ps.display_name == "New"
    assert ps.block_count == 10
    assert ps.max_context == 20
    assert ps.kv_cache_multiplier == 30
    # Launch args fence untouched
    assert ps.port == 8100


def test_patch_args_fence_updates_in_place():
    text = render_script(
        display_name="X",
        block_count=64,
        max_context=4096,
        kv_cache_multiplier=128,
        llama_bin="/bin/llama-server",
        model_path="/m.gguf",
        args={"--port": "8100"},
    )
    new = patch_launch_args_fence(
        text,
        llama_bin="/bin/llama-server",
        model_path="/other.gguf",
        args={"--port": "9000", "--ctx-size": "8192"},
    )
    ps = parse_script(new)
    assert ps.model_path == "/other.gguf"
    assert ps.port == 9000
    assert ps.args["--ctx-size"] == "8192"
    # Meta fence untouched
    assert ps.display_name == "X"
    assert ps.block_count == 64


def test_patch_overwrites_body_below_meta_fence():
    """New contract: form-saves rewrite everything below the meta fence wholesale.

    User content (custom env exports, sourced files) between meta and the exec
    line is intentionally clobbered — preserved decision when dropping the
    launch-args fence markers. The meta fence itself must survive untouched.
    """
    text = render_script(
        display_name="X",
        block_count=42,
        max_context=None,
        kv_cache_multiplier=None,
        llama_bin="/bin/llama-server",
        model_path="/m.gguf",
        args={"--port": "8100"},
    )
    new = patch_launch_args_fence(
        text,
        llama_bin="/bin/llama-server",
        model_path="/m.gguf",
        args={"--port": "9999"},
    )
    # Meta fence preserved
    ps = parse_script(new)
    assert ps.display_name == "X"
    assert ps.block_count == 42
    # New port reflected in the rewritten invocation
    assert ps.port == 9999
    # No legacy fence markers in output
    assert ARGS_BEGIN not in new
    assert ARGS_END not in new


# ---------- migration round-trip: legacy JSON shapes ----------

# ---------- multi-GPU: tensor-split + CUDA_VISIBLE_DEVICES ----------

def _script_with(extras: str = "", cvd_line: str = "") -> str:
    return f"""#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# display_name: T
# === llama-studio:meta END ===

LLAMA_BIN=/bin/llama-server
{cvd_line}
exec "$LLAMA_BIN" \\
  -m /m.gguf \\
  --port 8001{extras}
"""


def test_tensor_split_weights_uniform():
    text = _script_with(extras=" \\\n  --tensor-split 1,1,1")
    ps = parse_script(text)
    assert ps.tensor_split_weights == [1.0, 1.0, 1.0]
    assert ps.gpu_count == 3


def test_tensor_split_weights_asymmetric():
    text = _script_with(extras=" \\\n  --tensor-split 2,1")
    ps = parse_script(text)
    assert ps.tensor_split_weights == [2.0, 1.0]
    assert ps.gpu_count == 2


def test_gpu_count_single_when_absent():
    ps = parse_script(_script_with())
    assert ps.tensor_split_weights is None
    assert ps.gpu_count == 1


def test_gpu_count_single_when_split_one():
    text = _script_with(extras=" \\\n  --tensor-split 1")
    ps = parse_script(text)
    assert ps.gpu_count == 1


def test_tensor_split_malformed():
    text = _script_with(extras=" \\\n  --tensor-split abc")
    ps = parse_script(text)
    assert ps.tensor_split_weights is None
    assert ps.gpu_count == 1


def test_cvd_canonical_export_form():
    text = _script_with(cvd_line="export CUDA_VISIBLE_DEVICES=2,3")
    ps = parse_script(text)
    assert ps.cuda_visible_devices == [2, 3]


def test_cvd_plain_assignment():
    text = _script_with(cvd_line="CUDA_VISIBLE_DEVICES=0,1,2")
    ps = parse_script(text)
    assert ps.cuda_visible_devices == [0, 1, 2]


def test_cvd_quoted_value():
    text = _script_with(cvd_line='export CUDA_VISIBLE_DEVICES="1,2"')
    ps = parse_script(text)
    assert ps.cuda_visible_devices == [1, 2]


def test_cvd_inline_prefix():
    text = """#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# display_name: T
# === llama-studio:meta END ===

LLAMA_BIN=/bin/llama-server

CUDA_VISIBLE_DEVICES=1,2 exec "$LLAMA_BIN" \\
  -m /m.gguf \\
  --port 8001
"""
    ps = parse_script(text)
    assert ps.cuda_visible_devices == [1, 2]


def test_cvd_absent():
    ps = parse_script(_script_with())
    assert ps.cuda_visible_devices is None


def test_cvd_rightmost_wins():
    """Shell semantics: later assignment overrides earlier ones."""
    text = _script_with(cvd_line="export CUDA_VISIBLE_DEVICES=0,1\nexport CUDA_VISIBLE_DEVICES=2,3")
    ps = parse_script(text)
    assert ps.cuda_visible_devices == [2, 3]


def test_patch_cvd_insert():
    text = _script_with()
    new = patch_cuda_visible_devices(text, [2, 3])
    assert "export CUDA_VISIBLE_DEVICES=2,3" in new
    ps = parse_script(new)
    assert ps.cuda_visible_devices == [2, 3]


def test_patch_cvd_replace():
    text = _script_with(cvd_line="export CUDA_VISIBLE_DEVICES=0,1")
    new = patch_cuda_visible_devices(text, [4, 5])
    assert "0,1" not in new
    assert "export CUDA_VISIBLE_DEVICES=4,5" in new
    ps = parse_script(new)
    assert ps.cuda_visible_devices == [4, 5]


def test_patch_cvd_strip():
    text = _script_with(cvd_line="export CUDA_VISIBLE_DEVICES=0,1")
    new = patch_cuda_visible_devices(text, None)
    assert "CUDA_VISIBLE_DEVICES" not in new
    ps = parse_script(new)
    assert ps.cuda_visible_devices is None


def test_patch_cvd_strips_inline_prefix():
    text = """#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# === llama-studio:meta END ===

LLAMA_BIN=/bin/llama-server

CUDA_VISIBLE_DEVICES=1,2 exec "$LLAMA_BIN" \\
  -m /m.gguf \\
  --port 8001
"""
    new = patch_cuda_visible_devices(text, None)
    assert "CUDA_VISIBLE_DEVICES" not in new
    # The exec line should still execute correctly
    assert 'exec "$LLAMA_BIN"' in new


def test_patch_cvd_idempotent():
    text = _script_with()
    once = patch_cuda_visible_devices(text, [2, 3])
    twice = patch_cuda_visible_devices(once, [2, 3])
    assert once == twice


def test_patch_cvd_canonicalizes_when_inserting_over_inline_form():
    text = """#!/usr/bin/env bash
# === llama-studio:meta BEGIN ===
# === llama-studio:meta END ===

LLAMA_BIN=/bin/llama-server

CUDA_VISIBLE_DEVICES=0,1 exec "$LLAMA_BIN" \\
  -m /m.gguf \\
  --port 8001
"""
    new = patch_cuda_visible_devices(text, [4, 5])
    # Inline form stripped from exec line; canonical line above
    assert "CUDA_VISIBLE_DEVICES=0,1 exec" not in new
    assert "export CUDA_VISIBLE_DEVICES=4,5" in new
    ps = parse_script(new)
    assert ps.cuda_visible_devices == [4, 5]


# ---------- health_timeout meta field ----------

def test_health_timeout_default_absent():
    """Scripts without health_timeout parse to None."""
    ps = parse_script(CANONICAL)
    assert ps.health_timeout is None


def test_health_timeout_parses_when_present():
    text = CANONICAL.replace(
        "# kv_cache_multiplier: 213",
        "# kv_cache_multiplier: 213\n# health_timeout: 300",
    )
    ps = parse_script(text)
    assert ps.health_timeout == 300


def test_health_timeout_malformed_warns_and_keeps_none():
    text = CANONICAL.replace(
        "# kv_cache_multiplier: 213",
        "# kv_cache_multiplier: 213\n# health_timeout: not-a-number",
    )
    ps = parse_script(text)
    assert ps.health_timeout is None
    assert any("health_timeout" in w for w in ps.warnings)


def test_render_health_timeout_round_trip():
    text = render_script(
        display_name="X",
        block_count=64,
        max_context=4096,
        kv_cache_multiplier=128,
        llama_bin="/bin/llama-server",
        model_path="/m.gguf",
        args={"--port": "8100"},
        health_timeout=600,
    )
    assert "# health_timeout: 600" in text
    ps = parse_script(text)
    assert ps.health_timeout == 600


def test_render_health_timeout_omitted_when_none():
    text = render_script(
        display_name="X",
        block_count=64,
        max_context=4096,
        kv_cache_multiplier=128,
        llama_bin="/bin/llama-server",
        model_path="/m.gguf",
        args={"--port": "8100"},
        health_timeout=None,
    )
    assert "health_timeout" not in text


def test_patch_meta_fence_adds_health_timeout():
    text = render_script(
        display_name="X",
        block_count=64,
        max_context=4096,
        kv_cache_multiplier=128,
        llama_bin="/bin/llama-server",
        model_path="/m.gguf",
        args={"--port": "8100"},
    )
    new = patch_meta_fence(
        text,
        display_name="X",
        block_count=64,
        max_context=4096,
        kv_cache_multiplier=128,
        health_timeout=900,
    )
    ps = parse_script(new)
    assert ps.health_timeout == 900
    assert ps.port == 8100


def test_patch_meta_fence_strips_health_timeout_when_omitted():
    text = render_script(
        display_name="X",
        block_count=64,
        max_context=4096,
        kv_cache_multiplier=128,
        llama_bin="/bin/llama-server",
        model_path="/m.gguf",
        args={"--port": "8100"},
        health_timeout=300,
    )
    new = patch_meta_fence(
        text,
        display_name="X",
        block_count=64,
        max_context=4096,
        kv_cache_multiplier=128,
    )
    assert "health_timeout" not in new
    ps = parse_script(new)
    assert ps.health_timeout is None


# ---------- migration round-trip: legacy JSON shapes ----------

def test_migration_from_jsonish_launch_args():
    """Simulates: JSON launch_args dict → render → parse → equality."""
    launch_args = {
        "--host": "0.0.0.0",
        "--port": "8100",
        "--ctx-size": "262144",
        "--cache-type-k": "q8_0",
        "--cache-type-v": "q8_0",
        "--gpu-layers": "999",
        "--batch-size": "512",
        "--threads": "12",
        "--ubatch-size": "512",
        "--no-mmap": None,
        "--direct-io": None,
    }
    model_path = "/home/m6/models/Qwen3.6-27B-Q4_K_M.gguf"
    text = render_script(
        display_name="Qwen 3.6 27B",
        block_count=64,
        max_context=262144,
        kv_cache_multiplier=213,
        llama_bin="/home/m6/servers/llamacpp/bin/llama-server",
        model_path=model_path,
        args=launch_args,
    )
    ps = parse_script(text)
    assert ps.model_path == model_path
    for k, v in launch_args.items():
        assert ps.args.get(k) == v, f"mismatch for {k}: expected {v!r}, got {ps.args.get(k)!r}"


# ---------- split-mode / multi-GPU detection ----------

def _script_with_args(arg_lines: str) -> str:
    """Build a minimal script whose launch args are the given lines."""
    return (
        "#!/usr/bin/env bash\n"
        "# === llama-studio:meta BEGIN ===\n"
        "# display_name: T\n"
        "# === llama-studio:meta END ===\n"
        "LLAMA_BIN=/x/llama-server\n"
        'exec "$LLAMA_BIN" \\\n'
        "  -m /models/foo.gguf \\\n"
        "  --port 8120 \\\n"
        f"{arg_lines}"
        "  -fa on\n"
    )


def test_split_mode_layer_is_flexible():
    ps = parse_script(_script_with_args("  --split-mode layer \\\n"))
    assert ps.split_mode == "layer"
    assert ps.is_flexible_split is True
    assert ps.gpu_count == 1  # count is chosen at load time, not baked in


def test_split_mode_short_alias_row():
    ps = parse_script(_script_with_args("  -sm row \\\n"))
    assert ps.split_mode == "row"
    assert ps.is_flexible_split is True


def test_split_mode_none_is_single_gpu():
    ps = parse_script(_script_with_args("  --split-mode none \\\n"))
    assert ps.split_mode == "none"
    assert ps.is_flexible_split is False


def test_no_split_mode_is_single_gpu():
    ps = parse_script(_script_with_args("  --threads 8 \\\n"))
    assert ps.split_mode is None
    assert ps.is_flexible_split is False


def test_tensor_split_overrides_flexible():
    ps = parse_script(
        _script_with_args("  --split-mode layer \\\n  --tensor-split 1,1,1 \\\n")
    )
    assert ps.is_flexible_split is False  # explicit tensor-split wins
    assert ps.gpu_count == 3


def test_tensor_split_short_alias():
    ps = parse_script(_script_with_args("  -ts 2,1 \\\n"))
    assert ps.gpu_count == 2
    assert ps.tensor_split_weights == [2.0, 1.0]
