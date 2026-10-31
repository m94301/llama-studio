"""Parse llama-server -h output and generate schema."""

import subprocess
import json
import logging
import re
from pathlib import Path
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

def parse_help(binary_path: str) -> Optional[Dict]:
    """
    Run llama-server -h and parse help text.

    Returns a dict like:
    {
        "-c": {
            "type": "int",
            "short": "-c",
            "long": "--ctx-size",
            "description": "context size",
            "category": "context"
        },
        ...
    }
    """
    try:
        binary_path = Path(binary_path)
        if not binary_path.exists():
            logger.warning(f"⚠ Binary not found: {binary_path}")
            return None

        result = subprocess.run(
            [str(binary_path), "-h"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            logger.warning(f"⚠ llama-server -h returned code {result.returncode}")
            return None

        # Combine stdout and stderr in case help text is on stderr
        combined_output = result.stdout + result.stderr
        schema = _parse_help_text(combined_output)
        logger.info(f"✓ Parsed {len(schema)} options from llama-server help")
        return schema

    except subprocess.TimeoutExpired:
        logger.warning("⚠ llama-server -h timed out (timeout=10s)")
        return None
    except Exception as e:
        logger.warning(f"⚠ Error parsing help: {e}")
        return None


# Bumped whenever parsing changes, so caches written by an older parser are
# regenerated instead of being trusted (the cache file name carries it).
SCHEMA_FORMAT = 2

# llama-server aligns descriptions at column 40; a line indented this far is a
# wrapped description, even when it begins with "--". Option lines start near
# column 0 (or a little further, under a short-option column).
_DESCRIPTION_INDENT = 30

# The option list at the start of a help line: one or more spellings separated
# by commas ("-c,    --ctx-size", "--spec-draft-ngl, -ngld, --gpu-layers-draft"),
# then whatever follows (metavar, description, or nothing).
_OPTION_LIST_RE = re.compile(r"\s*(-{1,2}[A-Za-z][\w-]*(?:,\s*-{1,2}[A-Za-z][\w-]*)*)(.*)$")


def _parse_help_text(help_text: str) -> Dict:
    """
    Parse help text output. Extract options and descriptions.

    Help format (typical llama.cpp):
      -c,    --ctx-size N                     size of the prompt context (default: 512)
      --spec-draft-ngl, -ngld, --gpu-layers-draft, --n-gpu-layers-draft N
                                              max. number of draft model layers ...

    When the option list fills the line, the description starts on the next
    (indented) line. Returns a dict keyed by primary option name (prefer long
    form); every other spelling points to the same entry.
    """
    schema = {}
    lines = help_text.split('\n')

    for idx, line in enumerate(lines):
        # A previous option's wrapped description, e.g. "... use
        # --spec-draft-n-min or\n<40 spaces>--spec-ngram-mod-n-min".
        if len(line) - len(line.lstrip()) >= _DESCRIPTION_INDENT:
            continue
        match = _OPTION_LIST_RE.match(line)
        if not match:
            continue
        rest = match.group(2)
        if rest and not rest[0].isspace():
            continue  # e.g. prose like "-1)" that only looks like an option
        options = [opt.strip() for opt in match.group(1).split(',')]

        # Separate short and long forms
        shorts = [o for o in options if not o.startswith('--')]
        longs = [o for o in options if o.startswith('--')]

        # Prefer long form for key; fall back to first short
        primary_key = longs[0] if longs else shorts[0]
        secondary_keys = (shorts + longs[1:]) if longs else shorts[1:]

        # Extract description (before any "default:" marker), dropping a
        # leading parameter indicator ("N", "INDEX", ...).
        inline = rest.strip()
        if len(inline.split()) <= 1:  # nothing, or a lone metavar ("N", "FNAME")
            following = lines[idx + 1] if idx + 1 < len(lines) else ""
            desc = following.strip() if following[:1].isspace() and not following.strip().startswith('-') else ""
        else:
            desc = re.sub(r'^[A-Z_0-9{}\[\],.\s]+\s+', '', inline).strip()
        desc = desc.split('(default:')[0].strip()
        desc = desc.split('(env:')[0].strip()

        # Infer type from the parameter indicator, else the description
        option_type = _infer_type(desc, inline)

        entry = {
            "type": option_type,
            "description": desc,
            "category": _infer_category(primary_key),
        }

        # Store under primary key and all secondary keys
        schema[primary_key] = entry
        for key in secondary_keys:
            if key not in schema:
                schema[key] = entry

    return schema


def _infer_type(description: str, full_text: str) -> str:
    """Infer whether an option takes an int or str value.

    Primary signal: the parameter indicator (the first whitespace-separated token
    after the flag in llama-server's help output). llama.cpp uses standardized
    indicators — N / INT / NUM / SIZE for numeric, HOST / PATH / STRING / FILE
    etc. for textual.

    Falls back to a tight keyword scan of the description only (the looser scan of
    full_text caused false positives like 'listen (' matching 'n ').
    """
    # Extract the parameter indicator (first token in the post-flag remainder)
    m = re.match(r"\s*(\S+)", full_text or "")
    if m:
        param = m.group(1)
        # All-caps numeric indicators
        if param in {"N", "INT", "NUM", "NUMBER", "COUNT", "SIZE", "INDEX", "PORT", "LAYERS"}:
            return "int"
        # All-caps single word that isn't a known int indicator → string-typed param
        # (HOST, PATH, STRING, FILE, NAME, TEMPLATE, ALIAS, URL, JSON, etc.)
        if len(param) >= 2 and param.isalpha() and param.isupper():
            return "str"

    # Tight description-only fallback: only well-known numeric keywords
    desc_lower = (description or "").lower()
    if any(kw in desc_lower for kw in ("number of", "count of", "size of", "layers to", "threads to")):
        return "int"
    return "str"


def _infer_category(option_key: str) -> str:
    """Categorize an option based on its name."""
    key_lower = option_key.lower()
    if any(x in key_lower for x in ["host", "port", "addr", "listen"]):
        return "networking"
    if any(x in key_lower for x in ["gpu", "ngl", "cuda", "vulkan"]):
        return "gpu"
    if any(x in key_lower for x in ["ctx", "context", "seq"]):
        return "context"
    if any(x in key_lower for x in ["thread", "parallel", "np"]):
        return "threading"
    if any(x in key_lower for x in ["batch", "ubatch", "prompt", "cache"]):
        return "performance"
    if any(x in key_lower for x in ["spec", "draft"]):
        return "speculative"
    return "other"


def save_schema(schema: Dict, version_str: str, config_dir: Path) -> Path:
    """
    Save schema to config/llama-server/{version_str}.f{SCHEMA_FORMAT}.json.

    Returns the path to the saved file.
    """
    schema_dir = config_dir / "llama-server"
    schema_dir.mkdir(parents=True, exist_ok=True)

    schema_file = schema_dir / f"{version_str}.f{SCHEMA_FORMAT}.json"

    try:
        with open(schema_file, "w") as f:
            json.dump(schema, f, indent=2)
        logger.info(f"✓ Saved schema: {schema_file}")
        return schema_file
    except Exception as e:
        logger.error(f"✗ Error saving schema: {e}")
        raise


def load_schema(version_str: str, config_dir: Path) -> Optional[Dict]:
    """
    Load cached schema for a given version.

    Returns the schema dict, or None if not found.
    """
    schema_file = config_dir / "llama-server" / f"{version_str}.f{SCHEMA_FORMAT}.json"

    if not schema_file.exists():
        return None

    try:
        with open(schema_file) as f:
            schema = json.load(f)
        logger.info(f"✓ Loaded cached schema: {schema_file}")
        return schema
    except Exception as e:
        logger.warning(f"⚠ Error loading schema: {e}")
        return None
