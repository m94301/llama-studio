"""Detect llama-server version via --version flag."""

import subprocess
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Two formats have been seen in the wild:
#   old: "version: 9030 (17df5830e)"
#   new: "version: 0.4.1-dev (build 11067, commit fe93f196b)"
# Try new first (current upstream), fall back to old (older/forked binaries).
_VERSION_RE_NEW = re.compile(r'build\s+(\d+),\s+commit\s+([a-f0-9]+)')
_VERSION_RE_OLD = re.compile(r'version:\s+(\d+)\s+\(([a-f0-9]+)\)')


def get_version(binary_path: str) -> Optional[str]:
    """
    Run llama-server --version and parse the version string.

    Handles both the current upstream format ("version: 0.4.1-dev (build
    11067, commit fe93f196b)") and the older one ("version: 9030
    (17df5830e)"). Either way the identifier stays "<build>_<commit>" so
    existing cache files and app.json's stored llama_schema_version keep
    working across the format change.

    Returns:
        "11067_fe93f196b" or None if detection fails
    """
    try:
        binary_path = Path(binary_path)
        if not binary_path.exists():
            logger.warning(f"⚠ Llama-server binary not found: {binary_path}")
            return None

        result = subprocess.run(
            [str(binary_path), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        combined_output = result.stdout + result.stderr
        match = _VERSION_RE_NEW.search(combined_output) or _VERSION_RE_OLD.search(combined_output)
        if match:
            version_num, commit = match.groups()
            version_str = f"{version_num}_{commit}"
            logger.info(f"✓ Detected llama-server version: {version_str}")
            return version_str
        else:
            logger.warning(f"⚠ Could not parse version from: {result.stdout}")
            return None

    except subprocess.TimeoutExpired:
        logger.warning("⚠ llama-server --version timed out")
        return None
    except Exception as e:
        logger.warning(f"⚠ Error detecting version: {e}")
        return None


# vllm --version imports torch on first run — slow — so cache per path.
_VLLM_VERSION_CACHE: dict = {}


def get_vllm_version(binary_path: str) -> Optional[str]:
    """Run `vllm --version` and return the version string (e.g. '0.11.2'), cached."""
    if binary_path in _VLLM_VERSION_CACHE:
        return _VLLM_VERSION_CACHE[binary_path]
    try:
        p = Path(binary_path)
        if not p.exists():
            logger.warning(f"⚠ vllm executable not found: {binary_path}")
            return None
        result = subprocess.run(
            [str(p), "--version"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        combined = result.stdout + result.stderr
        m = re.search(r"(\d+\.\d+\S*)", combined)
        version = m.group(1) if m else None
        if version:
            logger.info(f"✓ Detected vllm version: {version}")
            _VLLM_VERSION_CACHE[binary_path] = version
        else:
            logger.warning(f"⚠ Could not parse vllm version from: {combined[:200]}")
        return version
    except subprocess.TimeoutExpired:
        logger.warning("⚠ vllm --version timed out")
        return None
    except Exception as e:
        logger.warning(f"⚠ Error detecting vllm version: {e}")
        return None
