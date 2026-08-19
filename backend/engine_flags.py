"""Capture and cache the flag set a vllm executable actually supports.

Generated args (skeletons, and later the Phase 2 structured modal) must only
ever emit flags present in this manifest — flags are read from the engine's
own `--help` output at registration time, never assumed. Cached per detected
version under config/vllm/flags-<version>.json.
"""

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional, Set

from llama_version import get_vllm_version

logger = logging.getLogger(__name__)

_FLAG_RE = re.compile(r"--[a-zA-Z0-9][a-zA-Z0-9-]*")


def _cache_path(config_dir: Path, version: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._+-]", "_", version)
    return config_dir / "vllm" / f"flags-{safe}.json"


def get_vllm_serve_flags(binary_path: str, config_dir: Path) -> Optional[Set[str]]:
    """Return the set of `vllm serve` flags this executable supports.

    Loads from the version-keyed cache when present; otherwise runs
    `vllm serve --help=all` (falling back to `--help`) and caches the result.
    Returns None when the flag set can't be determined — callers must then
    emit no optional flags at all.
    """
    version = get_vllm_version(binary_path)
    if not version:
        return None

    cache = _cache_path(config_dir, version)
    if cache.exists():
        try:
            return set(json.loads(cache.read_text()))
        except Exception as e:
            logger.warning(f"⚠ Corrupt vllm flag cache {cache.name}: {e}; regenerating")

    flags: Optional[Set[str]] = None
    for help_arg in ("--help=all", "--help"):
        try:
            result = subprocess.run(
                [binary_path, "serve", help_arg],
                capture_output=True,
                text=True,
                timeout=180,
            )
            found = set(_FLAG_RE.findall(result.stdout + result.stderr))
            # A real help dump has hundreds of flags; a handful means we got
            # an error/usage message, not the manifest.
            if len(found) >= 20:
                flags = found
                break
        except Exception as e:
            logger.warning(f"⚠ vllm serve {help_arg} failed: {e}")

    if flags is None:
        logger.warning(f"⚠ Could not capture flag manifest for {binary_path}")
        return None

    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(sorted(flags), indent=1))
        logger.info(f"✓ Cached {len(flags)} vllm serve flags → {cache.name}")
    except Exception as e:
        logger.warning(f"⚠ Could not write flag cache: {e}")
    return flags
