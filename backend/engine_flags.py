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

import hashlib
import json
import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from llama_version import get_vllm_version

logger = logging.getLogger(__name__)

# Bumped when the manifest layout changes, so stale caches regenerate instead
# of being read with the wrong shape.
MANIFEST_FORMAT = 3  # added category + description

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


# Group headers sit at column 0 and end in a colon ("CacheConfig:", "Frontend:").
_GROUP_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 _-]*):\s*$")

# Human-facing category names for the Option Browser; anything unlisted keeps
# the group name vLLM printed.
_GROUP_LABELS = {
    "options": "general",
    "Frontend": "server",
    "ModelConfig": "model",
    "LoadConfig": "loading",
    "AttentionConfig": "attention",
    "ParallelConfig": "parallel",
    "CacheConfig": "cache",
    "SchedulerConfig": "scheduler",
    "LoRAConfig": "lora",
    "MultiModalConfig": "multimodal",
    "ObservabilityConfig": "observability",
    "CompilationConfig": "compilation",
    "StructuredOutputsConfig": "structured outputs",
}


def parse_vllm_help(help_text: str) -> Dict[str, dict]:
    """Parse `vllm serve --help` into {flags: {...}, aliases: {...}}.

    Each flag records whether it takes a value, its category (the config group
    it was printed under) and its description (the indented help text), which is
    what the Option Browser needs.
    """
    flags: Dict[str, dict] = {}
    aliases: Dict[str, str] = {}
    group = "general"
    pending: List[str] = []          # flag names awaiting their description
    description: List[str] = []

    def _flush() -> None:
        if not pending:
            return
        text = " ".join(" ".join(description).split())
        for name in pending:
            if name in flags:
                flags[name]["description"] = text
        pending.clear()
        description.clear()

    for line in help_text.splitlines():
        gm = _GROUP_RE.match(line)
        if gm:
            _flush()
            raw_group = gm.group(1)
            group = _GROUP_LABELS.get(raw_group, raw_group)
            continue

        m = _OPTION_LINE_RE.match(line)
        if m:
            _flush()
            # The option list ends at the first run of 2+ spaces (the help column).
            parts = re.split(r"\s{2,}", m.group(1), maxsplit=1)
            head = parts[0]
            entries = _ENTRY_RE.findall(head)
            if not entries:
                continue

            takes_value = any(metavar.strip() for _, metavar in entries)
            names = [name for name, _ in entries]
            longs = [n for n in names if n.startswith("--")]
            canonical = longs[0] if longs else names[0]

            for name in names:
                flags[name] = {
                    "takes_value": takes_value,
                    "category": group,
                    "description": "",
                }
                # Only short forms are aliases. `--no-x` is a distinct flag with
                # the opposite meaning, never an alias of `--x`.
                if not name.startswith("--") and name != canonical:
                    aliases[name] = canonical

            pending.extend(names)
            if len(parts) > 1 and parts[1].strip():
                description.append(parts[1].strip())
            continue

        # Indented continuation: part of the current flag's description.
        if pending and line.startswith("  ") and line.strip():
            description.append(line.strip())
        elif not line.strip():
            _flush()

    _flush()
    return {"flags": flags, "aliases": aliases}


def _binary_id(binary_path: str) -> str:
    """Short stable id of the binary's resolved path.

    Version strings alone are not unique: every precompiled-wheel build reports
    the same placeholder (e.g. '0.0.0+precompiled'), so two different engines
    would otherwise share (and corrupt) one cached manifest.
    """
    try:
        real = os.path.realpath(binary_path)
    except Exception:
        real = str(binary_path)
    return hashlib.sha1(real.encode()).hexdigest()[:8]


def _cache_path(config_dir: Path, version: str, binary_path: Optional[str] = None) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._+-]", "_", version)
    suffix = f"-{_binary_id(binary_path)}" if binary_path else ""
    return config_dir / "vllm" / f"manifest-{safe}{suffix}.json"


# Per-process memo so parsing many scripts that share an engine captures it once.
# Failures are remembered briefly so a broken binary is not re-probed (a 5 min
# timeout each) for every script, but a fixed binary is picked up without a restart.
_MANIFEST_MEM: Dict[str, tuple] = {}
_MANIFEST_LOCK = threading.Lock()
_FAILURE_RETRY_SECONDS = 60


def get_vllm_manifest(binary_path: str, config_dir: Path) -> Optional[dict]:
    """Return this vllm build's flag manifest, capturing + caching on first use.

    Keyed per binary (version + path id), so every engine gets its own flag list.
    Returns None when it cannot be determined — callers must then emit no
    optional flags and fall back to heuristics rather than guessing.
    """
    try:
        key = os.path.realpath(binary_path)
    except Exception:
        key = str(binary_path)
    with _MANIFEST_LOCK:
        hit = _MANIFEST_MEM.get(key)
        if hit is not None:
            data, ts = hit
            if data is not None or time.time() - ts < _FAILURE_RETRY_SECONDS:
                return data
        data = _load_or_capture_manifest(binary_path, config_dir)
        _MANIFEST_MEM[key] = (data, time.time())
        return data


def _load_or_capture_manifest(binary_path: str, config_dir: Path) -> Optional[dict]:
    version = get_vllm_version(binary_path)
    if not version:
        return None

    cache = _cache_path(config_dir, version, binary_path)
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

    data = {"format": MANIFEST_FORMAT, "version": version, "binary": os.path.realpath(binary_path), **parsed}
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
