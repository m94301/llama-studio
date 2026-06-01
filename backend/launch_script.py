"""Parse and render llama-server launch shell scripts.

Script format:

    #!/usr/bin/env bash
    # === llama-studio:meta BEGIN ===
    # display_name: Human Name
    # block_count: 64
    # max_context: 262144
    # kv_cache_multiplier: 213
    # === llama-studio:meta END ===

    LLAMA_BIN="/path/to/llama-server"

    exec "$LLAMA_BIN" \
      -m /abs/path/to/model.gguf \
      --host 0.0.0.0 \
      --port 8100 \
      --ctx-size 262144 \
      --no-mmap

The meta fence holds user-owned display_name plus GGUF-derived cache fields,
and is patched in place by the UI. Everything after the meta fence is the
launch invocation; the UI rewrites it wholesale on save. The parser tolerates
legacy fenced launch-args blocks (used during the initial migration) but new
scripts are written without those markers.
"""

import logging
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

META_BEGIN = "# === llama-studio:meta BEGIN ==="
META_END = "# === llama-studio:meta END ==="
ARGS_BEGIN = "# === llama-studio:launch-args BEGIN ==="
ARGS_END = "# === llama-studio:launch-args END ==="

# Aliases: short form -> long form (normalize on parse, preserve user's choice when possible)
FLAG_ALIASES: Dict[str, str] = {
    "-m": "--model",
    "-c": "--ctx-size",
    "-ctk": "--cache-type-k",
    "-ctv": "--cache-type-v",
    "-ngl": "--gpu-layers",
    "-b": "--batch-size",
    "-ub": "--ubatch-size",
    "-t": "--threads",
}

# Known flag-only (boolean) options for llama-server. Used as a hint when
# heuristics ambiguous. Heuristic: a token starting with "-" after a flag is
# treated as a new flag, so these usually parse correctly without the list.
KNOWN_FLAG_ONLY = {
    "--no-mmap",
    "--mlock",
    "--no-warmup",
    "--flash-attn",
    "--cont-batching",
    "--embedding",
    "--metrics",
    "--direct-io",
    "--no-kv-offload",
    "--verbose",
    "-v",
}

# Meta fields that are GGUF-derived (integers); user-owned fields are str.
META_INT_FIELDS = {"block_count", "max_context", "kv_cache_multiplier"}

# Default health-check timeout in seconds. Used when a model's script doesn't
# declare an explicit `health_timeout` in the meta fence. 120 was the previous
# hardcoded value in llama_session.py.
DEFAULT_HEALTH_TIMEOUT = 120

# Bounds for user-set health timeout values (seconds).
HEALTH_TIMEOUT_MIN = 1
HEALTH_TIMEOUT_MAX = 1800


@dataclass
class ParsedScript:
    """Result of parsing a launch script."""

    # Meta fence contents
    display_name: Optional[str] = None
    block_count: Optional[int] = None
    max_context: Optional[int] = None
    kv_cache_multiplier: Optional[int] = None
    # Optional per-model overrides (None = use default at read time)
    health_timeout: Optional[int] = None

    # Launch-args fence contents
    llama_bin: Optional[str] = None  # binary path used in the exec line
    model_path: Optional[str] = None  # value of -m / --model
    args: Dict[str, Optional[str]] = field(default_factory=dict)  # all launch args incl. --port, --host

    # Shell-level state (read from script body, not args)
    cuda_visible_devices: Optional[List[int]] = None  # parsed CUDA_VISIBLE_DEVICES, None if absent

    # Original text and parse warnings
    raw_text: str = ""
    warnings: List[str] = field(default_factory=list)
    has_meta_fence: bool = False
    has_args_fence: bool = False

    @property
    def port(self) -> Optional[int]:
        v = self.args.get("--port")
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    @property
    def host(self) -> Optional[str]:
        return self.args.get("--host")

    def get(self, key: str, default=None):
        """Get an arg honoring aliases (caller can pass either short or long form)."""
        long = FLAG_ALIASES.get(key, key)
        return self.args.get(long, default)

    @property
    def tensor_split_weights(self) -> Optional[List[float]]:
        """Parse --tensor-split a,b,c into floats. None if absent or malformed."""
        raw = self.args.get("--tensor-split") or self.args.get("-ts")
        if raw is None:
            return None
        try:
            parts = [float(p.strip()) for p in str(raw).split(",") if p.strip()]
        except ValueError:
            return None
        if not parts:
            return None
        return parts

    @property
    def gpu_count(self) -> int:
        """Number of GPUs this model wants. 1 if no --tensor-split (or split=[1])."""
        weights = self.tensor_split_weights
        if weights is None:
            return 1
        return len(weights)

    def is_configured(self) -> Tuple[bool, Optional[str]]:
        """Return (configured, reason_if_not)."""
        if not self.has_args_fence:
            return False, "launch-args fence missing"
        if not self.model_path:
            return False, "no -m / --model in launch args"
        if not Path(self.model_path).exists():
            return False, f"model file not found: {self.model_path}"
        if self.port is None:
            return False, "--port not set"
        return True, None


# ---------- fence extraction ----------

def _extract_fence(text: str, begin: str, end: str) -> Optional[Tuple[int, int, str]]:
    """Return (begin_line_idx, end_line_idx, inner_text) or None.

    Indices point at the fence marker lines themselves. inner_text is the
    content strictly between the markers (no marker lines).
    """
    lines = text.splitlines()
    begin_idx = None
    end_idx = None
    for i, line in enumerate(lines):
        if line.strip() == begin and begin_idx is None:
            begin_idx = i
        elif line.strip() == end and begin_idx is not None:
            end_idx = i
            break
    if begin_idx is None or end_idx is None:
        return None
    inner = "\n".join(lines[begin_idx + 1:end_idx])
    return begin_idx, end_idx, inner


def _parse_meta_fence(inner: str) -> Dict[str, str]:
    """Parse '# key: value' comment lines from the meta fence."""
    meta: Dict[str, str] = {}
    for line in inner.splitlines():
        s = line.strip()
        if not s.startswith("#"):
            continue
        body = s.lstrip("#").strip()
        if ":" not in body:
            continue
        k, v = body.split(":", 1)
        meta[k.strip()] = v.strip()
    return meta


# ---------- launch-args tokenization ----------

def _collapse_continuations(text: str) -> str:
    """Collapse backslash-newline line continuations into single lines."""
    return re.sub(r"\\\s*\n\s*", " ", text)


def _strip_comments(text: str) -> str:
    """Remove # comments from shell text (naive; respects quoted strings)."""
    out_lines = []
    for line in text.splitlines():
        # Use shlex to find comment start while respecting quotes
        try:
            lex = shlex.shlex(line, posix=True)
            lex.commenters = "#"
            lex.whitespace_split = True
            # Reconstruct from tokens — drops comments
            tokens = list(lex)
            out_lines.append(" ".join(shlex.quote(t) for t in tokens))
        except ValueError:
            # Unbalanced quotes in this line; keep raw
            out_lines.append(line)
    return "\n".join(out_lines)


def _tokenize_args_fence(inner: str) -> Tuple[Optional[str], List[str]]:
    """Tokenize the launch-args fence body.

    Returns (binary_path, arg_tokens) where binary_path is the resolved path
    to llama-server (with $LLAMA_BIN expanded to the value of an assignment
    seen above, if present in the wider script — handled by caller) and
    arg_tokens is the list of tokens following the binary.

    If no recognizable llama-server invocation is found, returns (None, all_tokens).
    """
    collapsed = _collapse_continuations(inner)
    # Drop standalone comment lines but keep inline-quoted content intact
    no_comments = "\n".join(
        line for line in collapsed.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )

    try:
        tokens = shlex.split(no_comments, posix=True)
    except ValueError as e:
        raise ValueError(f"shell tokenization failed: {e}")

    # Drop leading 'exec' if present
    if tokens and tokens[0] == "exec":
        tokens = tokens[1:]

    if not tokens:
        return None, []

    binary = tokens[0]
    return binary, tokens[1:]


def _arg_tokens_to_dict(tokens: List[str], warnings: List[str]) -> Tuple[Dict[str, Optional[str]], Optional[str]]:
    """Walk a list of arg tokens into an ordered dict.

    Returns (args_dict, model_path). model_path is extracted from -m / --model
    and removed from args_dict (the renderer puts it back).
    """
    args: Dict[str, Optional[str]] = {}
    model_path: Optional[str] = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("-"):
            warnings.append(f"stray non-flag token in launch args: {tok!r}")
            i += 1
            continue

        # Normalize alias to canonical long form
        canonical = FLAG_ALIASES.get(tok, tok)

        # Determine if this is flag-only or takes a value
        is_flag_only = False
        if canonical in KNOWN_FLAG_ONLY:
            is_flag_only = True
        elif i + 1 >= len(tokens):
            is_flag_only = True
        else:
            nxt = tokens[i + 1]
            # Next token starts with '-' followed by a letter → new flag
            if nxt.startswith("--") or (
                len(nxt) >= 2 and nxt[0] == "-" and nxt[1].isalpha()
            ):
                is_flag_only = True

        if is_flag_only:
            if canonical == "--model":
                warnings.append("--model / -m present without a value")
            else:
                args[canonical] = None
            i += 1
        else:
            value = tokens[i + 1]
            if canonical == "--model":
                model_path = value
            else:
                args[canonical] = value
            i += 2

    return args, model_path


# ---------- public parse API ----------

def parse_script(text: str) -> ParsedScript:
    """Parse a llama-studio launch script (with fences).

    Designed for files we own. Missing fences are warnings, not errors.
    """
    ps = ParsedScript(raw_text=text)

    # Meta fence
    meta_fence = _extract_fence(text, META_BEGIN, META_END)
    if meta_fence is not None:
        ps.has_meta_fence = True
        meta = _parse_meta_fence(meta_fence[2])
        ps.display_name = meta.get("display_name") or None
        for f in META_INT_FIELDS:
            v = meta.get(f)
            if v is not None and v != "":
                try:
                    setattr(ps, f, int(v))
                except ValueError:
                    ps.warnings.append(f"meta {f}={v!r} is not an integer")
        # Optional user-overridable fields
        ht = meta.get("health_timeout")
        if ht is not None and ht != "":
            try:
                ps.health_timeout = int(ht)
            except ValueError:
                ps.warnings.append(f"meta health_timeout={ht!r} is not an integer")
    else:
        ps.warnings.append("meta fence missing")

    # Launch invocation: prefer the legacy fence if present (older scripts),
    # otherwise scan for the exec/llama-server line in the body below the meta fence.
    args_fence = _extract_fence(text, ARGS_BEGIN, ARGS_END)
    if args_fence is not None:
        ps.has_args_fence = True
        try:
            binary, tokens = _tokenize_args_fence(args_fence[2])
            ps.llama_bin = _resolve_llama_bin(binary, text) if binary else None
            ps.args, ps.model_path = _arg_tokens_to_dict(tokens, ps.warnings)
        except ValueError as e:
            ps.warnings.append(f"launch-args fence parse failed: {e}")
    else:
        # Fenceless: scan the body below the meta fence (if any) for the invocation
        body = text
        if meta_fence is not None:
            lines = text.splitlines()
            body = "\n".join(lines[meta_fence[1] + 1:])
        _populate_from_body(ps, body, text)

    # Shell-level CVD (read from the whole script regardless of fence presence)
    ps.cuda_visible_devices = _extract_cuda_visible_devices(text)

    return ps


def _populate_from_body(ps: ParsedScript, body: str, full_text: str) -> None:
    """Find the llama-server invocation in `body` and populate ps.{llama_bin, args, model_path}."""
    collapsed = _collapse_continuations(body)
    invocation = None
    for line in collapsed.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("#!"):
            continue
        if re.match(r"^(export\s+)?[A-Za-z_][A-Za-z0-9_]*\s*=", s):
            continue
        if s.startswith("exec ") or "$LLAMA_BIN" in s or "${LLAMA_BIN" in s or "llama-server" in s:
            invocation = s
            break

    if invocation is None:
        ps.has_args_fence = False  # no invocation found
        ps.warnings.append("no llama-server invocation found")
        return

    try:
        tokens = shlex.split(invocation, posix=True)
    except ValueError as e:
        ps.warnings.append(f"could not tokenize invocation: {e}")
        return

    if tokens and tokens[0] == "exec":
        tokens = tokens[1:]
    if not tokens:
        return

    ps.has_args_fence = True  # found a valid invocation; treat as "configured-able"
    ps.llama_bin = _resolve_llama_bin(tokens[0], full_text)
    ps.args, ps.model_path = _arg_tokens_to_dict(tokens[1:], ps.warnings)


def parse_pasted_script(text: str) -> ParsedScript:
    """Parse a foreign script (e.g. from Unsloth).

    Delegates to parse_script() in all cases — that function now handles both
    fenced (legacy) and fenceless scripts uniformly. After parsing, applies
    the paste-import display_name override from --alias.
    """
    ps = parse_script(text)

    # If no invocation was located, fall back to treating the whole text as
    # raw args (e.g. user pasted just the flag list without the binary).
    if ps.llama_bin is None and not ps.model_path and not ps.args:
        try:
            tokens = shlex.split(_collapse_continuations(text), posix=True)
            ps.args, ps.model_path = _arg_tokens_to_dict(tokens, ps.warnings)
            ps.warnings.append("no llama-server invocation found; parsed pasted text as args only")
        except ValueError as e:
            ps.warnings.append(f"could not tokenize pasted text: {e}")

    # Resolved decision #1: paste-import always overwrites display_name from --alias
    alias = ps.args.get("--alias")
    if alias:
        ps.display_name = alias

    return ps


_CVD_KEY = "CUDA_VISIBLE_DEVICES"


def _parse_cvd_value(raw: str) -> Optional[List[int]]:
    """Parse a 'CUDA_VISIBLE_DEVICES=...' value into a list of ints, or None on malformed."""
    s = raw.strip().strip('"').strip("'")
    if not s:
        return None
    try:
        return [int(p.strip()) for p in s.split(",") if p.strip()]
    except ValueError:
        return None


def _extract_cuda_visible_devices(text: str) -> Optional[List[int]]:
    """Find CUDA_VISIBLE_DEVICES in the script.

    Accepts three forms:
      1. `export CUDA_VISIBLE_DEVICES=2,3` on its own line (canonical)
      2. `CUDA_VISIBLE_DEVICES=2,3` (no export) on its own line
      3. Inline prefix: `CUDA_VISIBLE_DEVICES=2,3 exec ...`
    Returns None if absent or unparseable.
    """
    # Standalone assignment (with optional export). Match the rightmost occurrence
    # so a later override wins, mirroring shell behavior.
    pattern = re.compile(
        rf'^[ \t]*(?:export[ \t]+)?{_CVD_KEY}=("([^"]*)"|\'([^\']*)\'|(\S+))',
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if matches:
        m = matches[-1]
        # The capture group is whichever of the three forms matched
        value = m.group(2) if m.group(2) is not None else (m.group(3) if m.group(3) is not None else m.group(4))
        return _parse_cvd_value(value)

    # Inline prefix on an exec line: CUDA_VISIBLE_DEVICES=X command ...
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(_CVD_KEY + "="):
            # Find the value up to the next whitespace
            tail = s[len(_CVD_KEY) + 1:]
            value, _, _ = tail.partition(" ")
            return _parse_cvd_value(value)
    return None


def _resolve_llama_bin(token: str, full_text: str) -> str:
    """If token references $LLAMA_BIN, look up the assignment in the wider script."""
    if "$LLAMA_BIN" in token or "${LLAMA_BIN" in token:
        # Find LLAMA_BIN="..." assignment
        m = re.search(r'^\s*LLAMA_BIN\s*=\s*"([^"]+)"', full_text, re.MULTILINE)
        if m:
            return m.group(1)
        m = re.search(r"^\s*LLAMA_BIN\s*=\s*'([^']+)'", full_text, re.MULTILINE)
        if m:
            return m.group(1)
        m = re.search(r"^\s*LLAMA_BIN\s*=\s*(\S+)", full_text, re.MULTILINE)
        if m:
            return m.group(1)
    return token


# ---------- rendering ----------

def render_script(
    *,
    display_name: Optional[str],
    block_count: Optional[int],
    max_context: Optional[int],
    kv_cache_multiplier: Optional[int],
    llama_bin: str,
    model_path: Optional[str],
    args: Dict[str, Optional[str]],
    preserved_body: Optional[str] = None,
    health_timeout: Optional[int] = None,
) -> str:
    """Render a complete script.

    `args` should NOT include --model / -m (model_path is rendered separately).
    `preserved_body` is the content between the meta fence and the args fence;
    if None, a default LLAMA_BIN assignment is emitted.
    `health_timeout`, when set, adds a `# health_timeout: N` line to the meta
    fence. Pass None to omit (the load logic falls back to DEFAULT_HEALTH_TIMEOUT).
    """
    meta_lines = [META_BEGIN]
    if display_name:
        meta_lines.append(f"# display_name: {display_name}")
    if block_count is not None:
        meta_lines.append(f"# block_count: {block_count}")
    if max_context is not None:
        meta_lines.append(f"# max_context: {max_context}")
    if kv_cache_multiplier is not None:
        meta_lines.append(f"# kv_cache_multiplier: {kv_cache_multiplier}")
    if health_timeout is not None:
        meta_lines.append(f"# health_timeout: {health_timeout}")
    meta_lines.append(META_END)
    meta_block = "\n".join(meta_lines)

    if preserved_body is None:
        body = f'\nLLAMA_BIN={shlex.quote(llama_bin)}\n'
    else:
        body = preserved_body
        if "LLAMA_BIN" not in body:
            body = body + f'\nLLAMA_BIN={shlex.quote(llama_bin)}\n'

    invocation = _render_invocation(llama_bin, model_path, args)

    return f"#!/usr/bin/env bash\n{meta_block}\n{body}\n{invocation}\n"


def _render_invocation(llama_bin: str, model_path: Optional[str], args: Dict[str, Optional[str]]) -> str:
    """Render the bare exec line + args (no fence markers)."""
    lines = ['exec "$LLAMA_BIN" \\']
    arg_lines: List[str] = []
    if model_path:
        arg_lines.append(f"  -m {shlex.quote(model_path)}")
    for key, value in args.items():
        if key in ("-m", "--model"):
            continue  # already emitted
        if value is None or str(value).strip() == "":
            arg_lines.append(f"  {key}")
        else:
            arg_lines.append(f"  {key} {shlex.quote(str(value))}")
    if arg_lines:
        for line in arg_lines[:-1]:
            lines.append(line + " \\")
        lines.append(arg_lines[-1])
    return "\n".join(lines)


# ---------- fence patching (in-place rewrites) ----------

def patch_meta_fence(
    text: str,
    *,
    display_name: Optional[str] = None,
    block_count: Optional[int] = None,
    max_context: Optional[int] = None,
    kv_cache_multiplier: Optional[int] = None,
    health_timeout: Optional[int] = None,
) -> str:
    """Rewrite only the meta fence. Add it after the shebang if missing.

    Any of the kwargs left as None will be omitted from the new meta block.
    """
    new_lines = [META_BEGIN]
    if display_name:
        new_lines.append(f"# display_name: {display_name}")
    if block_count is not None:
        new_lines.append(f"# block_count: {block_count}")
    if max_context is not None:
        new_lines.append(f"# max_context: {max_context}")
    if kv_cache_multiplier is not None:
        new_lines.append(f"# kv_cache_multiplier: {kv_cache_multiplier}")
    if health_timeout is not None:
        new_lines.append(f"# health_timeout: {health_timeout}")
    new_lines.append(META_END)
    new_block = "\n".join(new_lines)

    fence = _extract_fence(text, META_BEGIN, META_END)
    if fence is not None:
        begin_idx, end_idx, _ = fence
        lines = text.splitlines()
        lines[begin_idx:end_idx + 1] = new_block.splitlines()
        return "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    # Insert after shebang (or at top)
    lines = text.splitlines()
    insert_at = 1 if lines and lines[0].startswith("#!") else 0
    lines[insert_at:insert_at] = new_block.splitlines()
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def patch_launch_args_fence(
    text: str,
    *,
    llama_bin: str,
    model_path: Optional[str],
    args: Dict[str, Optional[str]],
) -> str:
    """Replace everything below the meta fence with a fresh LLAMA_BIN + invocation.

    Any user content between the meta fence and the invocation is overwritten;
    this is the agreed trade-off for dropping the launch-args fence markers.
    Users who need custom shell content (pre-launch warmups, sourced files) can
    edit the script via the Edit raw-script flow.
    """
    body = f'\nLLAMA_BIN={shlex.quote(llama_bin)}\n\n' + _render_invocation(llama_bin, model_path, args) + "\n"

    meta_fence = _extract_fence(text, META_BEGIN, META_END)
    lines = text.splitlines()
    trailing_newline = "\n" if text.endswith("\n") else ""

    if meta_fence is not None:
        _, end_idx, _ = meta_fence
        # Keep everything through the meta END line; replace everything after.
        kept = lines[:end_idx + 1]
        return "\n".join(kept) + "\n" + body.lstrip("\n") + (trailing_newline if not body.endswith("\n") else "")

    # No meta fence: prepend shebang if missing, append body
    if not lines or not lines[0].startswith("#!"):
        prefix = "#!/usr/bin/env bash\n"
    else:
        prefix = ""
    return prefix + text + (trailing_newline if not text.endswith("\n") else "") + body


def patch_cuda_visible_devices(text: str, ids: Optional[List[int]]) -> str:
    """Insert, replace, or remove the canonical CUDA_VISIBLE_DEVICES line.

    - ids = [2, 3] → write `export CUDA_VISIBLE_DEVICES=2,3` between `LLAMA_BIN=`
      and the blank line preceding `exec`. If a CVD line already exists in any
      form (export, plain assignment, inline prefix), it's removed first.
    - ids = None → strip any existing CVD line(s) and leave nothing.
    Idempotent: if the script already has the canonical line with the same ids,
    no change is made.
    """
    lines = text.splitlines()
    trailing_newline = "\n" if text.endswith("\n") else ""

    # If the requested state already matches, no-op
    existing = _extract_cuda_visible_devices(text)
    if (ids is None and existing is None) or (
        ids is not None and existing is not None and list(ids) == list(existing)
    ):
        # Even on no-op, if there are duplicate/non-canonical lines we should clean up.
        # Detect by checking if a strict canonical pattern matches uniquely.
        if not _has_only_canonical_cvd(text, ids):
            pass  # fall through to rewrite
        else:
            return text

    # Strip all CVD declarations (standalone lines + inline prefixes)
    cleaned_lines: List[str] = []
    standalone_re = re.compile(
        rf'^[ \t]*(?:export[ \t]+)?{_CVD_KEY}=(?:"[^"]*"|\'[^\']*\'|\S+)[ \t]*$'
    )
    inline_prefix_re = re.compile(rf'^([ \t]*){_CVD_KEY}=(?:"[^"]*"|\'[^\']*\'|\S+)[ \t]+')
    for line in lines:
        if standalone_re.match(line):
            continue  # drop entire line
        # Strip an inline prefix but keep the rest of the command
        m = inline_prefix_re.match(line)
        if m:
            line = m.group(1) + line[m.end():]
        cleaned_lines.append(line)

    if ids is None:
        # Just remove; return cleaned text
        out = "\n".join(cleaned_lines)
        return out + trailing_newline

    # Insert canonical line after the LLAMA_BIN= line if present, else before the
    # first non-comment / non-shebang line.
    new_line = f"export {_CVD_KEY}=" + ",".join(str(i) for i in ids)
    insert_at: Optional[int] = None
    for i, line in enumerate(cleaned_lines):
        if re.match(r"^[ \t]*LLAMA_BIN[ \t]*=", line):
            insert_at = i + 1
            break
    if insert_at is None:
        # Fallback: after shebang + meta fence
        for i, line in enumerate(cleaned_lines):
            if line.strip() == META_END:
                insert_at = i + 1
                break
    if insert_at is None:
        insert_at = 1 if cleaned_lines and cleaned_lines[0].startswith("#!") else 0

    # Avoid stacking blank lines: if next line is already blank, replace one of them
    if insert_at < len(cleaned_lines) and cleaned_lines[insert_at].strip() == "":
        cleaned_lines[insert_at:insert_at] = [new_line]
    else:
        cleaned_lines[insert_at:insert_at] = [new_line, ""]

    out = "\n".join(cleaned_lines)
    return out + trailing_newline


def _has_only_canonical_cvd(text: str, ids: Optional[List[int]]) -> bool:
    """True if the script contains exactly the canonical form for the given ids and nothing else CVD-related."""
    if ids is None:
        return _CVD_KEY not in text
    canonical = f"export {_CVD_KEY}=" + ",".join(str(i) for i in ids)
    # Count occurrences of CVD anywhere
    cvd_lines = [ln for ln in text.splitlines() if _CVD_KEY in ln]
    return len(cvd_lines) == 1 and cvd_lines[0].strip() == canonical


# ---------- skeleton ----------

def render_skeleton_script(
    *,
    display_name: str,
    model_path: str,
    llama_bin: str,
    block_count: Optional[int] = None,
    max_context: Optional[int] = None,
    kv_cache_multiplier: Optional[int] = None,
) -> str:
    """Render a skeleton script for a newly-discovered GGUF.

    --port is intentionally omitted so the model shows as unconfigured (resolved
    decision #2). Defaults match the previous JSON skeleton.
    """
    args: Dict[str, Optional[str]] = {
        "--host": "0.0.0.0",
        "--gpu-layers": "999",
        "--ctx-size": "4096",
        "--cache-type-k": "f16",
        "--cache-type-v": "f16",
        "--batch-size": "512",
        "--threads": "12",
    }
    return render_script(
        display_name=display_name,
        block_count=block_count,
        max_context=max_context,
        kv_cache_multiplier=kv_cache_multiplier,
        llama_bin=llama_bin,
        model_path=model_path,
        args=args,
    )
