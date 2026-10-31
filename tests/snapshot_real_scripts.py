"""Behavior snapshot over the real model scripts — the safety net for refactors.

Copies the active config/models/*.sh (not .old) plus app.json and the flag
schema/manifest caches into a temp project root, loads them the way startup
does, and records for every script:

  - the parse result (meta fields, engine, args, env, warnings)
  - derived values (size, total VRAM, GPU count, context, per-GPU VRAM need,
    CVD usability against synthetic GPU sets)
  - the script text after a no-change save through its engine's save endpoint

The real scripts are never touched.

    venv/bin/python tests/snapshot_real_scripts.py record   # write the baseline
    venv/bin/python tests/snapshot_real_scripts.py check    # diff against it

The baseline lives in tests/.snapshot/ (gitignored: it holds local paths).
"""

import asyncio
import dataclasses
import difflib
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

SNAPSHOT = PROJECT_ROOT / "tests" / ".snapshot" / "real_scripts.json"

# Synthetic machines for the GPU-dependent numbers: one tight, one roomy.
GPU_SETS = {
    "4x24": {i: {"name": "FAKE-24", "memory": 24.0, "allocated": 0.0} for i in range(4)},
    "4x96": {i: {"name": "FAKE-96", "memory": 96.0, "allocated": 0.0} for i in range(4)},
}


def _copy_config(dest: Path) -> list:
    """Copy the parts of config/ a load needs; return the copied script names."""
    src = PROJECT_ROOT / "config"
    (dest / "config" / "models").mkdir(parents=True)
    shutil.copy2(src / "app.json", dest / "config" / "app.json")
    for cache in ("llama-server", "vllm"):
        if (src / cache).is_dir():
            shutil.copytree(src / cache, dest / "config" / cache)
    names = []
    for sh in sorted((src / "models").glob("*.sh")):
        shutil.copy2(sh, dest / "config" / "models" / sh.name)
        names.append(sh.stem)
    return names


def _jsonable(value):
    return json.loads(json.dumps(value, default=str))


def _model_record(mc, main) -> dict:
    parsed = {k: v for k, v in dataclasses.asdict(mc.parsed).items() if k != "raw_text"}
    rec = {
        "parsed": parsed,
        "engine": mc.engine,
        "display_name": mc.display_name,
        "model_path": mc.model_path,
        "llama_path": mc.llama_path,
        "port": mc.port,
        "size_gb": mc.size_gb,
        "total_vram": mc.total_vram,
        "is_configured": mc.is_configured,
        "config_error": mc.config_error,
        "gpu_count": mc.gpu_count,
        "context_length": mc.context_length,
        "concurrency": mc.concurrency,
        "max_context": mc.max_context,
        "is_flexible_split": mc.is_flexible_split,
        "cuda_visible_devices": mc.cuda_visible_devices,
        "effective_health_timeout": mc.effective_health_timeout(),
        "per_gpu_vram": mc.per_gpu_vram(),
        "per_gpu_vram_by_count": {n: mc.per_gpu_vram(count=n) for n in (1, 2, 3, 4)},
        "needed_on_gpu": {cap: mc.needed_on_gpu(cap) for cap in (24.0, 96.0)},
        "cvd_usability": {
            label: main._stored_cvd_usability(mc, gpus) for label, gpus in GPU_SETS.items()
        },
    }
    return _jsonable(rec)


async def _noop_save(mc, main) -> dict:
    """Submit the model's current settings unchanged, as the edit modal would."""
    args = mc.launch_args or {}
    if mc.engine == "vllm":
        resp = await main.save_vllm_config(
            model_name=mc.name,
            display_name=mc.display_name or "",
            health_timeout=str(mc.health_timeout or ""),
            args_json=json.dumps(args),
            env_json=json.dumps(mc.parsed.env_assignments),
            source_model="",
        )
    else:
        # Mirrors config_modal_llama.html serializeFormData(): the five
        # permanent rows (with the modal's defaults) plus every other arg.
        get = lambda *keys, default=None: next((args[k] for k in keys if args.get(k)), default)
        advanced = {
            "--host": args.get("--host", "0.0.0.0"),
            "--port": str(args.get("--port", mc.port or "")),
            "--ctx-size": get("--ctx-size", "-c", default="4096"),
            "--cache-type-k": get("--cache-type-k", "-ctk", default="f16"),
            "--cache-type-v": get("--cache-type-v", "-ctv", default="f16"),
        }
        skip = {"--host", "--port", "-c", "--ctx-size", "-ctk", "--cache-type-k", "-ctv", "--cache-type-v"}
        advanced.update({k: v for k, v in args.items() if k not in skip})
        core = {
            "port": advanced["--port"] or None,
            "display_name": mc.display_name or None,
            "host": advanced["--host"],
            "llama_path": mc.llama_path or "default",
            "health_timeout": mc.health_timeout,
        }
        resp = await (getattr(main, "save_model_config_new", None) or main.save_model_config)(
            model_name=mc.name,
            core_json=json.dumps(core),
            advanced_json=json.dumps(advanced),
            source_model="",
        )
    body = resp.body.decode() if hasattr(resp, "body") else str(resp)
    return {
        "save_error": "Configuration Error" in body,
        "saved_text": mc.script_path.read_text(),
    }


async def build_snapshot() -> dict:
    logging.disable(logging.CRITICAL)
    import main
    from config_manager import ConfigManager
    from gpu_manager import GpuManager
    from engine_flags import get_vllm_manifest
    from launch_script import set_vllm_manifest, set_vllm_manifest_resolver

    tmp = Path(tempfile.mkdtemp(prefix="ls-snapshot-"))
    try:
        names = _copy_config(tmp)
        originals = {n: (tmp / "config" / "models" / f"{n}.sh").read_text() for n in names}

        cm = ConfigManager(tmp / "config" / "app.json", project_root=tmp)
        main.config_manager = cm
        main.gpu_manager = GpuManager(cm)
        await cm.load_app_config()
        await main.ensure_schema_for_binary(cm.app_config.llama_server_binary, cm, update_global=True)
        set_vllm_manifest_resolver(lambda b: get_vllm_manifest(b, tmp / "config"))
        if cm.app_config.vllm_binary:
            set_vllm_manifest(get_vllm_manifest(cm.app_config.vllm_binary, tmp / "config"))
        await cm.load_all_models()

        out = {}
        for name in names:
            mc = cm.models.get(name)
            if mc is None:
                out[name] = {"missing_after_load": True}
                continue
            rec = _model_record(mc, main)
            rec["load_rewrote_script"] = mc.script_path.read_text() != originals[name]
            rec.update(await _noop_save(mc, main))
            out[name] = rec
        return json.loads(json.dumps(out).replace(str(tmp), "<ROOT>"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _diff(old: dict, new: dict) -> list:
    lines = []
    for name in sorted(set(old) | set(new)):
        if name not in new:
            lines.append(f"- {name}: no longer present")
            continue
        if name not in old:
            lines.append(f"+ {name}: new script (not in baseline)")
            continue
        a, b = old[name], new[name]
        for key in sorted(set(a) | set(b)):
            if a.get(key) == b.get(key):
                continue
            if key == "saved_text":
                lines.append(f"~ {name}: no-change save output differs")
                lines.extend("    " + l for l in difflib.unified_diff(
                    a.get(key, "").splitlines(), b.get(key, "").splitlines(),
                    "baseline", "now", lineterm="", n=1))
            else:
                lines.append(f"~ {name}.{key}:\n    baseline: {json.dumps(a.get(key))}\n    now:      {json.dumps(b.get(key))}")
    return lines


def main_cli() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    snap = asyncio.run(build_snapshot())
    if mode == "record":
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(snap, indent=1, sort_keys=True))
        print(f"Recorded {len(snap)} scripts → {SNAPSHOT}")
        return 0
    if not SNAPSHOT.exists():
        print("No baseline yet; run with 'record' first.")
        return 2
    diff = _diff(json.loads(SNAPSHOT.read_text()), snap)
    if not diff:
        print(f"✓ {len(snap)} scripts: no difference from baseline")
        return 0
    print("\n".join(diff))
    return 1


if __name__ == "__main__":
    sys.exit(main_cli())
