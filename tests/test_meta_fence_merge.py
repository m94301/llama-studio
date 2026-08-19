"""Tests for meta-fence merge semantics (audit finding #4).

patch_meta_fence used to rebuild the fence from its kwargs, so anything it
didn't model — user comments, unrecognized keys — was deleted, and any caller
that forgot a field silently erased it (`engine` defaulted to "llamacpp",
which could flip a vLLM script to llama.cpp and hand it to the body-destroying
save path).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from launch_script import (  # noqa: E402
    META_BEGIN,
    META_END,
    parse_script,
    patch_meta_fence,
)


ANNOTATED = f"""#!/usr/bin/env bash
{META_BEGIN}
# display_name: My Model
# engine: vllm
# health_timeout: 900
# note: tuned for 3 cards, do not raise util above 0.93
# vram_gb: 380.5
{META_END}
export VLLM_PP_LAYER_PARTITION=15,16,12

exec /opt/venv/bin/vllm serve /models/m --host 0.0.0.0 --port 8098
"""


class TestPreservation:
    def test_unmodelled_lines_survive(self):
        new = patch_meta_fence(ANNOTATED, health_timeout=1200)
        assert "# note: tuned for 3 cards, do not raise util above 0.93" in new

    def test_engine_survives_a_caller_that_forgets_it(self):
        """The exact hazard: a llama.cpp-shaped call must not retag a vLLM model."""
        new = patch_meta_fence(ANNOTATED, block_count=64, max_context=4096)
        assert parse_script(new).engine == "vllm"

    def test_vram_gb_survives(self):
        new = patch_meta_fence(ANNOTATED, health_timeout=60)
        assert parse_script(new).vram_gb == 380.5

    def test_body_below_the_fence_untouched(self):
        new = patch_meta_fence(ANNOTATED, health_timeout=60)
        assert new.split(META_END)[1] == ANNOTATED.split(META_END)[1]


class TestThreeStateSemantics:
    def test_omitted_preserves(self):
        assert parse_script(patch_meta_fence(ANNOTATED, engine="vllm")).health_timeout == 900

    def test_value_sets(self):
        assert parse_script(patch_meta_fence(ANNOTATED, health_timeout=42)).health_timeout == 42

    def test_none_deletes(self):
        new = patch_meta_fence(ANNOTATED, health_timeout=None)
        assert parse_script(new).health_timeout is None
        assert "health_timeout" not in new

    def test_new_field_is_appended(self):
        assert "block_count" not in ANNOTATED
        new = patch_meta_fence(ANNOTATED, block_count=64)
        assert parse_script(new).block_count == 64

    def test_existing_key_keeps_its_position(self):
        new = patch_meta_fence(ANNOTATED, display_name="Renamed")
        inner = new.split(META_BEGIN)[1].split(META_END)[0].strip().splitlines()
        assert inner[0].strip() == "# display_name: Renamed"


class TestFenceCreation:
    def test_creates_fence_after_shebang_when_absent(self):
        text = "#!/usr/bin/env bash\nexec /usr/bin/llama-server -m /m.gguf --port 8100\n"
        new = patch_meta_fence(text, engine="llamacpp")
        assert new.splitlines()[0] == "#!/usr/bin/env bash"
        ps = parse_script(new)
        assert ps.has_meta_fence and ps.engine_declared and ps.engine == "llamacpp"
        assert ps.port == 8100

    def test_idempotent(self):
        once = patch_meta_fence(ANNOTATED, health_timeout=300)
        twice = patch_meta_fence(once, health_timeout=300)
        assert once == twice
