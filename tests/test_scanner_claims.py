"""Model discovery must not duplicate a model that a script already covers.

A vLLM weights directory used to count as covered only when a script was
named exactly `<dir name>.sh`, so any script with a different filename
(Save As, a rename, a hand-written script) got a duplicate skeleton next to it
on the next scan.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import engine_flags  # noqa: E402
from config_manager import ConfigManager  # noqa: E402


def _hf_dir(path: Path, arch: str = "Qwen3ForCausalLM") -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"architectures": [arch]}))
    (path / "model.safetensors").write_bytes(b"\0" * 16)
    return path


def _vllm_script(dest: Path, model_ref: str, preamble: str = "", extra: str = "") -> None:
    dest.write_text(
        "#!/usr/bin/env bash\n"
        "# === llama-studio:meta BEGIN ===\n"
        "# display_name: Custom\n"
        "# engine: vllm\n"
        "# === llama-studio:meta END ===\n"
        f"{preamble}"
        f'exec /venv/bin/vllm serve {model_ref} \\\n'
        f"  --port 8300{extra}\n"
    )


@pytest.fixture
def studio(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_flags, "get_vllm_serve_flags", lambda *a, **k: set())
    models = tmp_path / "weights"
    models.mkdir()
    (tmp_path / "config" / "models").mkdir(parents=True)
    app = tmp_path / "config" / "app.json"
    app.write_text(json.dumps({
        "models_directory": str(models),
        "logs_directory": str(tmp_path / "logs"),
        "llama_server_binary": "/nonexistent/llama-server",
        "vllm_binary": "/venv/bin/vllm",
    }))
    cm = ConfigManager(app, project_root=tmp_path)

    def scan():
        asyncio.run(cm.load_app_config())
        asyncio.run(cm.load_all_models())
        return sorted(p.name for p in (tmp_path / "config" / "models").glob("*.sh"))

    return models, tmp_path / "config" / "models", scan


def test_renamed_script_claims_its_weights(studio):
    models, scripts, scan = studio
    weights = _hf_dir(models / "org" / "Big-Model-NVFP4")
    _vllm_script(scripts / "big-model-2card.sh", str(weights))
    assert scan() == ["big-model-2card.sh"]


def test_claim_through_a_shell_variable_with_default(studio):
    # dsv4-flash-3card's form: MODEL="${X:-/path}" ... serve "$MODEL"
    models, scripts, scan = studio
    weights = _hf_dir(models / "org" / "DSV4")
    _vllm_script(scripts / "dsv4-3card.sh", '"$MODEL"',
                 preamble=f'MODEL="${{DSV4_MODEL:-{weights}}}"\n')
    assert scan() == ["dsv4-3card.sh"]


def test_claim_through_a_symlinked_path(studio):
    models, scripts, scan = studio
    weights = _hf_dir(models / "drive1" / "org" / "M")
    (models / "drive2").mkdir()
    (models / "drive2" / "M-link").symlink_to(weights)
    _vllm_script(scripts / "m.sh", str(models / "drive2" / "M-link"))
    assert scan() == ["m.sh"]


def test_disabled_script_still_claims(studio):
    # The sync pass re-enables it because its weights exist; what matters is
    # that no "Parked.sh" skeleton appears beside it.
    models, scripts, scan = studio
    weights = _hf_dir(models / "org" / "Parked")
    _vllm_script(scripts / "parked-custom.sh.old", str(weights))
    assert scan() == ["parked-custom.sh"]


def test_speculative_drafter_dir_is_claimed(studio):
    models, scripts, scan = studio
    target = _hf_dir(models / "org" / "Target")
    drafter = _hf_dir(models / "org" / "Target-Drafter")  # ordinary arch name
    spec = json.dumps({"method": "eagle3", "model": str(drafter), "num_speculative_tokens": 3})
    _vllm_script(scripts / "target.sh", str(target), extra=f" \\\n  --speculative-config '{spec}'")
    assert scan() == ["target.sh"]


def test_draft_architecture_gets_no_skeleton(studio):
    models, _, scan = studio
    _hf_dir(models / "org" / "GLM-Flash-DFlash2", arch="DFlash2DraftModel")
    assert scan() == []


def test_trash_and_hidden_dirs_are_ignored(studio):
    models, _, scan = studio
    _hf_dir(models / ".Trash-1000" / "files" / "org" / "Deleted")
    (models / ".cache").mkdir()
    (models / ".cache" / "x.gguf").write_bytes(b"\0" * 16)
    assert scan() == []


def test_unclaimed_new_download_still_gets_a_skeleton(studio):
    models, _, scan = studio
    _hf_dir(models / "org" / "Fresh-Download")
    assert scan() == ["Fresh-Download.sh"]
