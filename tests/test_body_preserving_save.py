"""Tests for patch_exec_invocation and ConfigManager.save_script_edits — the
body-preserving structured save both engines use (doc 21 §2.2).

Scripts are mostly hand-written context (env vars, tuning comments, variable
defaults); a save rewrites the exec line, the meta fence and changed
assignments, and nothing else.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from launch_script import (  # noqa: E402
    META_BEGIN,
    META_END,
    find_invocation_span,
    parse_script,
    patch_exec_invocation,
)


DSV4_SHAPED = f"""#!/usr/bin/env bash
{META_BEGIN}
# display_name: DeepSeek V4 Flash
# engine: vllm
{META_END}
# Native (no-docker) launch on 3x CMP170HX. Tuned over several days.
set -euo pipefail
VENV=/opt/vllm/venv
export CUDA_HOME="$VENV/lib/python3.10/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"

export CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES:-0,1,2}}"   # 3 cards
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_PP_LAYER_PARTITION="${{PART:-15,16,12}}"   # rebalanced off rank2

MODEL="${{DSV4_MODEL:-/models/dsv4}}"
MAXLEN="${{MAXLEN:-524288}}"                            # 512K daily driver
GPUUTIL="${{GPUUTIL:-0.93}}"                            # 0.96 was too tight, crashed

exec "$VENV/bin/vllm" serve "$MODEL" \\
  --pipeline-parallel-size 3 \\
  --max-model-len "$MAXLEN" \\
  --gpu-memory-utilization "$GPUUTIL" \\
  --host 0.0.0.0 --port 8098
"""


def _preamble(text: str) -> str:
    """Everything before the invocation line."""
    span = find_invocation_span(text)
    return "\n".join(text.splitlines()[:span[0]])


class TestSpanDetection:
    def test_finds_the_full_continued_invocation(self):
        start, end = find_invocation_span(DSV4_SHAPED)
        lines = DSV4_SHAPED.splitlines()
        assert lines[start].startswith('exec "$VENV/bin/vllm" serve')
        assert lines[end - 1].strip().endswith("--port 8098")

    def test_ignores_comments_and_assignments(self):
        start, _ = find_invocation_span(DSV4_SHAPED)
        assert "MODEL=" not in DSV4_SHAPED.splitlines()[start]

    def test_none_when_absent(self):
        assert find_invocation_span("#!/bin/bash\nexport X=1\n") is None


class TestBodyPreservation:
    def test_everything_above_the_exec_line_is_untouched(self):
        ps = parse_script(DSV4_SHAPED)
        out = patch_exec_invocation(
            DSV4_SHAPED, args={**ps.args, "--max-num-seqs": "8"}, parsed=ps
        )
        assert _preamble(out) == _preamble(DSV4_SHAPED)

    def test_tuning_comments_survive(self):
        ps = parse_script(DSV4_SHAPED)
        out = patch_exec_invocation(DSV4_SHAPED, args=ps.args, parsed=ps)
        for comment in ("0.96 was too tight, crashed", "rebalanced off rank2",
                        "512K daily driver", "Tuned over several days"):
            assert comment in out

    def test_env_preamble_survives(self):
        ps = parse_script(DSV4_SHAPED)
        out = patch_exec_invocation(DSV4_SHAPED, args=ps.args, parsed=ps)
        for line in ("export VLLM_WORKER_MULTIPROC_METHOD=spawn",
                     'export VLLM_PP_LAYER_PARTITION="${PART:-15,16,12}"',
                     'export CUDA_HOME="$VENV/lib/python3.10/site-packages/nvidia/cu13"'):
            assert line in out

    def test_meta_fence_untouched(self):
        ps = parse_script(DSV4_SHAPED)
        out = patch_exec_invocation(DSV4_SHAPED, args=ps.args, parsed=ps)
        assert out.split(META_END)[0] == DSV4_SHAPED.split(META_END)[0]


class TestVariableRoundTrip:
    def test_unchanged_values_keep_their_variable_reference(self):
        ps = parse_script(DSV4_SHAPED)
        out = patch_exec_invocation(
            DSV4_SHAPED, args={**ps.args, "--max-num-seqs": "8"}, parsed=ps
        )
        assert '--max-model-len "$MAXLEN"' in out
        assert '--gpu-memory-utilization "$GPUUTIL"' in out
        # Never frozen to the resolved value in the invocation. (524288 still
        # appears in the preserved preamble's MAXLEN default, as it should.)
        span = find_invocation_span(out)
        invocation = "\n".join(out.splitlines()[span[0]:span[1]])
        assert "524288" not in invocation
        assert "0.93" not in invocation

    def test_changed_values_are_written_literally(self):
        ps = parse_script(DSV4_SHAPED)
        out = patch_exec_invocation(
            DSV4_SHAPED, args={**ps.args, "--max-model-len": "32768"}, parsed=ps
        )
        assert "--max-model-len 32768" in out
        assert '--max-model-len "$MAXLEN"' not in out

    def test_model_path_variable_survives(self):
        ps = parse_script(DSV4_SHAPED)
        out = patch_exec_invocation(DSV4_SHAPED, args=ps.args, parsed=ps)
        assert 'serve "$MODEL"' in out

    def test_binary_reference_survives(self):
        ps = parse_script(DSV4_SHAPED)
        out = patch_exec_invocation(DSV4_SHAPED, args=ps.args, parsed=ps)
        assert 'exec "$VENV/bin/vllm" serve' in out


class TestRoundTripSemantics:
    def test_reparse_yields_the_same_configuration(self):
        ps = parse_script(DSV4_SHAPED)
        out = patch_exec_invocation(DSV4_SHAPED, args=ps.args, parsed=ps)
        again = parse_script(out)
        assert again.model_path == ps.model_path
        assert again.port == ps.port
        assert again.args.get("--max-model-len") == ps.args.get("--max-model-len")
        assert again.gpu_count == ps.gpu_count
        assert again.cuda_visible_devices == ps.cuda_visible_devices

    def test_idempotent(self):
        ps = parse_script(DSV4_SHAPED)
        once = patch_exec_invocation(DSV4_SHAPED, args=ps.args, parsed=ps)
        twice = patch_exec_invocation(once, args=parse_script(once).args,
                                      parsed=parse_script(once))
        assert once == twice

    def test_added_argument_appears(self):
        ps = parse_script(DSV4_SHAPED)
        out = patch_exec_invocation(
            DSV4_SHAPED,
            args={**ps.args, "--kv-cache-memory-bytes": "805306368"},
            parsed=ps,
        )
        assert parse_script(out).kv_cache_memory_bytes == 805306368

    def test_removed_argument_disappears(self):
        ps = parse_script(DSV4_SHAPED)
        args = {k: v for k, v in ps.args.items() if k != "--gpu-memory-utilization"}
        out = patch_exec_invocation(DSV4_SHAPED, args=args, parsed=ps)
        assert "--gpu-memory-utilization" not in out

    def test_boolean_flag_emitted_without_a_value(self):
        ps = parse_script(DSV4_SHAPED)
        out = patch_exec_invocation(
            DSV4_SHAPED, args={**ps.args, "--trust-remote-code": None}, parsed=ps
        )
        assert "--trust-remote-code \\" in out or out.rstrip().endswith("--trust-remote-code")
        assert parse_script(out).args["--trust-remote-code"] is None


LLAMACPP = f"""#!/usr/bin/env bash
{META_BEGIN}
# engine: llamacpp
{META_END}
# my notes about this model
LLAMA_BIN="/usr/bin/llama-server"
export CUDA_VISIBLE_DEVICES=2

exec "$LLAMA_BIN" \\
  -m /models/x.gguf \\
  --host 0.0.0.0 \\
  --port 8100 \\
  --ctx-size 4096
"""


class TestLlamacppShape:
    def test_preserves_body_and_model_flag(self):
        ps = parse_script(LLAMACPP)
        out = patch_exec_invocation(
            LLAMACPP, args={**ps.args, "--ctx-size": "8192"}, parsed=ps
        )
        assert "# my notes about this model" in out
        assert "export CUDA_VISIBLE_DEVICES=2" in out
        again = parse_script(out)
        assert again.model_path == "/models/x.gguf"
        assert again.args["--ctx-size"] == "8192"
        assert again.port == 8100


# ---------- ConfigManager.save_script_edits: the one save path ----------

import asyncio  # noqa: E402
import difflib  # noqa: E402
import json  # noqa: E402

from config_manager import ConfigManager  # noqa: E402

NOTES_SCRIPT = f"""#!/usr/bin/env bash
{META_BEGIN}
# display_name: Muse Glimmer
# engine: llamacpp
{META_END}
# Notes: dense 30B, SWA. A/B by commenting out the --spec-* lines below.
# VRAM budget on one 64GB card: ~36GB loaded.
LLAMA_BIN=/opt/llama/bin/llama-server
DISPLAY_NAME='Muse Glimmer'
export CUDA_VISIBLE_DEVICES=0

exec "$LLAMA_BIN" \\
  -m {{model}} \\
  -a "$DISPLAY_NAME" \\
  --spec-draft-n-max 15 \\
  --host 0.0.0.0 \\
  --port 8135 \\
  --ctx-size 131072 \\
  -ngl 999
"""


@pytest.fixture
def studio(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    model = weights / "muse.gguf"
    model.write_bytes(b"not really a gguf")
    (tmp_path / "config" / "models").mkdir(parents=True)
    (tmp_path / "config" / "app.json").write_text(json.dumps({
        "models_directory": str(weights),
        "llama_server_binary": "/opt/llama/bin/llama-server",
    }))
    script = tmp_path / "config" / "models" / "muse.sh"
    script.write_text(NOTES_SCRIPT.format(model=model))
    cm = ConfigManager(tmp_path / "config" / "app.json", project_root=tmp_path)
    asyncio.run(cm.load_app_config())
    asyncio.run(cm.load_all_models())
    return cm, script


def _changed_lines(before: str, after: str) -> list:
    return [l for l in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0)
            if l[:1] in "+-" and l[:3] not in ("+++", "---")]


class TestStructuredSave:
    def _save(self, cm, dest="muse", **edits):
        mc = cm.get_model_config("muse")
        args = {**mc.parsed.args, **edits.pop("args", {})}
        return cm.save_script_edits("muse", dest, args=args,
                                    display_name=edits.pop("display_name", mc.display_name), **edits)

    def test_no_change_save_is_byte_identical(self, studio):
        cm, script = studio
        before = script.read_text()
        self._save(cm)
        assert script.read_text() == before

    def test_an_edit_changes_only_its_own_line(self, studio):
        cm, script = studio
        before = script.read_text()
        self._save(cm, args={"--ctx-size": "65536"})
        assert _changed_lines(before, script.read_text()) == [
            "-  --ctx-size 131072 \\", "+  --ctx-size 65536 \\"]

    def test_new_option_is_appended_and_spellings_kept(self, studio):
        cm, script = studio
        self._save(cm, args={"--no-mmap": None})
        text = script.read_text()
        assert text.rstrip().endswith("-ngl 999 \\\n  --no-mmap")
        assert '-a "$DISPLAY_NAME"' in text

    def test_new_binary_goes_to_the_llama_bin_assignment(self, studio):
        cm, script = studio
        before = script.read_text()
        self._save(cm, llama_bin="/opt/llama-new/bin/llama-server")
        assert _changed_lines(before, script.read_text()) == [
            "-LLAMA_BIN=/opt/llama/bin/llama-server", "+LLAMA_BIN=/opt/llama-new/bin/llama-server"]

    def test_display_name_updates_fence_and_variable(self, studio):
        cm, script = studio
        self._save(cm, display_name="Muse v2")
        text = script.read_text()
        assert "# display_name: Muse v2" in text and "DISPLAY_NAME='Muse v2'" in text
        assert "# Notes: dense 30B" in text

    def test_save_as_keeps_the_source_notes_and_leaves_source_alone(self, studio):
        cm, script = studio
        before = script.read_text()
        self._save(cm, dest="muse-fork", args={"--port": "8136"})
        fork = script.with_name("muse-fork.sh").read_text()
        assert script.read_text() == before
        assert "# Notes: dense 30B" in fork and "--port 8136" in fork
        assert cm.get_model_config("muse-fork").port == 8136
