"""Migrate legacy JSON model configs to shell scripts.

For each `config/models/<name>.json`:
  1. Read JSON.
  2. Render `<name>.sh` from launch_args + identity fields + GGUF cache metadata.
  3. chmod +x the script.
  4. Round-trip verify: parse the new .sh, assert recovered args match the JSON.
  5. Rename `<name>.json` -> `<name>.json.old` (never delete).

Idempotent: skips models that already have a .sh.
Dry-run mode: print plan without writing.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow running from project root or as a module
sys.path.insert(0, str(Path(__file__).resolve().parent))

from launch_script import render_script, parse_script  # noqa: E402

logger = logging.getLogger(__name__)


class MigrationError(Exception):
    pass


def _resolve_llama_bin(json_data: dict, app_config: dict) -> str:
    """Determine which llama-server binary path to bake into the script.

    Resolved decisions: bake absolute path at render time so the script is
    self-contained (matches the "pin model to llama-server version" goal).
    """
    llama_path = json_data.get("llama_path") or "default"
    if llama_path and llama_path != "default":
        return llama_path
    binary = app_config.get("llama_server_binary")
    if not binary:
        raise MigrationError(
            "app config has no llama_server_binary and model uses 'default'"
        )
    return binary


def migrate_one(
    json_path: Path,
    app_config: dict,
    *,
    dry_run: bool = False,
) -> Tuple[bool, str]:
    """Migrate a single JSON config to a shell script.

    Returns (changed, message). changed=False means skipped (already migrated or no-op).
    Raises MigrationError on hard failure.
    """
    model_name = json_path.stem
    sh_path = json_path.with_suffix(".sh")

    if sh_path.exists():
        return False, f"{model_name}: skip (sh already exists)"

    with open(json_path) as f:
        data = json.load(f)

    launch_args: Dict[str, Optional[str]] = dict(data.get("launch_args") or {})
    model_path = data.get("model_path")
    if not model_path:
        raise MigrationError(f"{model_name}: no model_path in JSON")

    # Strip -m / --model from launch_args if present (renderer emits separately)
    for k in ("-m", "--model"):
        launch_args.pop(k, None)

    # Coerce values to strings (JSON may have ints); preserve None for flag-only
    coerced: Dict[str, Optional[str]] = {}
    for k, v in launch_args.items():
        coerced[k] = None if v is None else str(v)

    llama_bin = _resolve_llama_bin(data, app_config)

    script_text = render_script(
        display_name=data.get("display_name") or model_name,
        block_count=data.get("block_count"),
        max_context=data.get("max_context"),
        kv_cache_multiplier=data.get("kv_cache_multiplier"),
        llama_bin=llama_bin,
        model_path=model_path,
        args=coerced,
    )

    # Round-trip verify
    ps = parse_script(script_text)
    if ps.model_path != model_path:
        raise MigrationError(
            f"{model_name}: round-trip model_path mismatch "
            f"({ps.model_path!r} != {model_path!r})"
        )
    for k, v in coerced.items():
        if ps.args.get(k) != v:
            raise MigrationError(
                f"{model_name}: round-trip arg {k!r} mismatch "
                f"({ps.args.get(k)!r} != {v!r})"
            )

    if dry_run:
        return True, f"{model_name}: would write {sh_path.name} + rename {json_path.name} -> {json_path.name}.old"

    sh_path.write_text(script_text)
    sh_path.chmod(0o755)
    backup = json_path.with_suffix(".json.old")
    json_path.rename(backup)
    return True, f"{model_name}: wrote {sh_path.name}, renamed {json_path.name} -> {backup.name}"


def find_legacy_configs(models_dir: Path) -> List[Path]:
    """Return all <name>.json files in config/models that aren't .json.old."""
    if not models_dir.exists():
        return []
    return sorted(
        p for p in models_dir.glob("*.json")
        if p.name != "default.json"  # template; not a real model config
    )


def migrate_all(
    project_root: Path,
    *,
    dry_run: bool = False,
) -> Tuple[List[str], List[str]]:
    """Migrate every legacy JSON in config/models/.

    Returns (successes, failures) as lists of human-readable messages.
    """
    config_dir = project_root / "config" / "models"
    app_config_path = project_root / "config" / "app.json"

    if not app_config_path.exists():
        raise MigrationError(f"app config not found: {app_config_path}")
    with open(app_config_path) as f:
        app_config = json.load(f)

    legacy = find_legacy_configs(config_dir)
    if not legacy:
        return [], []

    successes: List[str] = []
    failures: List[str] = []
    for json_path in legacy:
        try:
            changed, msg = migrate_one(json_path, app_config, dry_run=dry_run)
            successes.append(msg)
        except Exception as e:
            failures.append(f"{json_path.stem}: ✗ {type(e).__name__}: {e}")
    return successes, failures


def has_legacy_configs(project_root: Path) -> bool:
    """Quick check: are there any unmigrated JSONs? Used at startup."""
    return bool(find_legacy_configs(project_root / "config" / "models"))


def main():
    parser = argparse.ArgumentParser(description="Migrate JSON model configs to shell scripts.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--dry-run", action="store_true", help="Print plan without writing")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    print(f"Scanning {args.project_root}/config/models/ for legacy JSON configs...")
    successes, failures = migrate_all(args.project_root, dry_run=args.dry_run)

    if not successes and not failures:
        print("✓ Nothing to migrate.")
        return 0

    for s in successes:
        print(f"  ✓ {s}")
    for f in failures:
        print(f"  ✗ {f}")

    print()
    print(f"Result: {len(successes)} migrated, {len(failures)} failed.")
    if args.dry_run:
        print("(dry run — no files changed)")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
