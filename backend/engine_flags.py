"""Capture and cache what an engine binary's CLI actually accepts.

Generated args and arg parsing must only ever use flags the engine itself
reports — never a list written from memory. A hardcoded `--disable-log-requests`
in a generated skeleton is what made a real launch fail: that flag does not
exist in every vLLM build.

The manifest is captured from `vllm serve --help=all` and records, per flag,
whether it takes a value and which long form a short alias maps to. Cached per
detected version under config/vllm/.

`vllm serve --help=all` renders options as::

    --enforce-eager, --no-enforce-eager
                          Whether to always use eager-mode PyTorch...
    --max-model-len MAX_MODEL_LEN
                          Model context length...
    --pipeline-parallel-size PIPELINE_PARALLEL_SIZE, -pp PIPELINE_PARALLEL_SIZE
                          Number of pipeline parallel groups...
    --kda-prefill-backend {auto,triton,flashkda}

so a flag takes a value iff it is followed by a metavar (uppercase token or a
{choice,list}), and short aliases share the entry with their long form.
"""

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional, Set

from llama_version import get_vllm_version

logger = logging.getLogger(__name__)

# Bumped when the manifest layout changes, so stale caches regenerate instead
# of being read with the wrong shape.
MANIFEST_FORMAT = 2

# One option entry: a flag, optionally followed by a metavar. vLLM renders
# metavars four ways, all of which mean "this flag takes a value":
#   --max-model-len MAX_MODEL_LEN          uppercase metavar
#   --kda-prefill-backend {auto,triton}    brace choice list
#   --tokenizer-mode ['auto', 'mistral']   quoted choice list
#   --hf-token [HF_TOKEN]                  optional-value metavar
_ENTRY_RE = re.compile(
    r"(-{1,2}[a-zA-Z0-9][\w-]*)"
    r"((?:[ =][A-Z][A-Z0-9_]*|\s*\{[^}]*\}|\s*\[[^\]]*\])?)"
)
# An option line: small indent, then the option list, then 2+ spaces before help.
_OPTION_LINE_RE = re.compile(r"^\s{1,6}(-{1,2}[a-zA-Z0-9].*)$")


def parse_vllm_help(help_text: str) -> Dict[str, dict]:
    """Parse `vllm serve --help` output into {flags: {...}, aliases: {...}}."""
    flags: Dict[str, dict] = {}
    aliases: Dict[str, str] = {}

    for line in help_text.splitlines():
        m = _OPTION_LINE_RE.match(line)
        if not m:
            continue
        # The option list ends at the first run of 2+ spaces (the help column).
        head = re.split(r"\s{2,}", m.group(1), maxsplit=1)[0]
        entries = _ENTRY_RE.findall(head)
        if not entries:
            continue

        takes_value = any(metavar.strip() for _, metavar in entries)
        names = [name for name, _ in entries]
        longs = [n for n in names if n.startswith("--")]
        canonical = longs[0] if longs else names[0]

        for name in names:
            flags[name] = {"takes_value": takes_value}
            # Only short forms are aliases. `--no-x` is a distinct flag with the
            # opposite meaning, never an alias of `--x`.
            if not name.startswith("--") and name != canonical:
                aliases[name] = canonical

    return {"flags": flags, "aliases": aliases}


def _cache_path(config_dir: Path, version: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._+-]", "_", version)
    return config_dir / "vllm" / f"manifest-{safe}.json"


def get_vllm_manifest(binary_path: str, config_dir: Path) -> Optional[dict]:
    """Return this vllm build's flag manifest, capturing + caching on first use.

    Returns None when it cannot be determined — callers must then emit no
    optional flags and fall back to heuristics rather than guessing.
    """
    version = get_vllm_version(binary_path)
    if not version:
        return None

    cache = _cache_path(config_dir, version)
    if cache.exists():
        try:
            data = json.loads(cache.read_text())
            if data.get("format") == MANIFEST_FORMAT and data.get("flags"):
                return data
            logger.info(f"↻ Regenerating vllm manifest ({cache.name}): stale format")
        except Exception as e:
            logger.warning(f"⚠ Corrupt vllm manifest {cache.name}: {e}; regenerating")

    parsed: Optional[Dict[str, dict]] = None
    for help_arg in ("--help=all", "--help"):
        try:
            result = subprocess.run(
                [binary_path, "serve", help_arg],
                capture_output=True, text=True, timeout=300,
            )
            candidate = parse_vllm_help(result.stdout + result.stderr)
            # A real help dump has hundreds of flags; a handful means we caught
            # a usage/error message instead of the manifest.
            if len(candidate["flags"]) >= 20:
                parsed = candidate
                break
        except Exception as e:
            logger.warning(f"⚠ `vllm serve {help_arg}` failed: {e}")

    if parsed is None:
        logger.warning(f"⚠ Could not capture a flag manifest from {binary_path}")
        return None

    data = {"format": MANIFEST_FORMAT, "version": version, **parsed}
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, indent=1, sort_keys=True))
        logger.info(
            f"✓ Cached vllm manifest: {len(data['flags'])} flags, "
            f"{len(data['aliases'])} aliases → {cache.name}"
        )
    except Exception as e:
        logger.warning(f"⚠ Could not write vllm manifest: {e}")
    return data


def get_vllm_serve_flags(binary_path: str, config_dir: Path) -> Optional[Set[str]]:
    """Just the set of accepted flag names (used to filter generated skeletons)."""
    manifest = get_vllm_manifest(binary_path, config_dir)
    return set(manifest["flags"]) if manifest else None
