"""Parse and render model launch shell scripts (llama-server and vllm serve).

Script format (llama.cpp):

    #!/usr/bin/env bash
    # === llama-studio:meta BEGIN ===
    # display_name: Human Name
    # engine: llamacpp
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
and is patched in place by the UI. Below it, a save rewrites only the launch
invocation (patch_exec_invocation) and assignments that changed; comments and
every other line are the user's and are kept as written.

Script format (vLLM) — `# engine: vllm` in the meta fence, and the invocation
is `exec ".../vllm" serve <model-dir> --flags...` with the model as a
positional argument. vLLM scripts commonly carry an env-var preamble
(`export VLLM_*=...`, `MODEL=...` variable defaults); the parser resolves
`$VAR` / `${VAR}` / `${VAR:-default}` references against assignments found in
the script body, and the vLLM save patches only the exec line, env
assignments and meta fence (patch_exec_invocation), never the rest of the body.

The `engine` field is mandatory going forward. Scripts missing it parse as
llamacpp with `engine_declared=False`; the scanner backfills the tag.
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

# Aliases: short form -> long form (normalize on parse, preserve user's choice when possible)
FLAG_ALIASES: Dict[str, str] = {
    "-m": "--model",
    "-a": "--alias",
    "-c": "--ctx-size",
    "-ctk": "--cache-type-k",
    "-ctv": "--cache-type-v",
    "-ngl": "--gpu-layers",
    "-b": "--batch-size",
    "-ub": "--ubatch-size",
    "-t": "--threads",
    "-sm": "--split-mode",
    "-ts": "--tensor-split",
    "-np": "--parallel",
    "-kvu": "--kv-unified",
    "-no-kvu": "--no-kv-unified",
    "-mm": "--mmproj",
    "-md": "--model-draft",
    "--spec-draft-model": "--model-draft",
}


def canonical_args(args: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    """Rewrite short aliases to their long form, as the parser does (a later key wins)."""
    return {FLAG_ALIASES.get(k, k): v for k, v in args.items()}

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

# What `vllm serve` accepts is read from the engine's own --help (see
# engine_flags.py) — never hardcoded, because flag sets differ per build. The
# app installs a resolver (engine_flags.install_vllm_parse_manifests) so each
# script is parsed against its own engine; a fixed manifest set here is the
# fallback (tests use it). With neither, the parser falls back to the
# next-token heuristic instead of guessing from a list.
_VLLM_MANIFEST: Optional[Dict[str, dict]] = None


def set_vllm_manifest(manifest: Optional[Dict[str, dict]]) -> None:
    """Install a fixed `vllm serve` flag manifest for parsing."""
    global _VLLM_MANIFEST
    _VLLM_MANIFEST = manifest
    if manifest:
        logger.debug(
            f"✓ vllm manifest active: {len(manifest.get('flags', {}))} flags, "
            f"{len(manifest.get('aliases', {}))} aliases"
        )


_VLLM_MANIFEST_RESOLVER = None


def set_vllm_manifest_resolver(fn) -> None:
    """Install fn(binary_path) -> manifest|None.

    Each vLLM script runs the engine named in its own `exec` line, which may be
    a custom build whose flags differ from the default engine's. With a resolver
    installed, a script is parsed against that engine's own manifest; a fixed
    manifest (set_vllm_manifest) is only the fallback.
    """
    global _VLLM_MANIFEST_RESOLVER
    _VLLM_MANIFEST_RESOLVER = fn


def manifest_for_binary(binary: Optional[str]) -> Optional[Dict[str, dict]]:
    """Flag manifest of the engine at `binary`, else the default engine's."""
    if _VLLM_MANIFEST_RESOLVER:
        try:
            found = _VLLM_MANIFEST_RESOLVER(binary)
            if found:
                return found
        except Exception as e:  # never let manifest capture break parsing
            logger.warning(f"⚠ vllm manifest lookup failed for {binary}: {e}")
    return _VLLM_MANIFEST

VALID_ENGINES = ("llamacpp", "vllm")

# Distinguishes "caller didn't mention this field" from "caller passed None to
# clear it" in patch_meta_fence. Without it, a call site that simply forgets a
# field silently deletes (or worse, overwrites) it.
_UNSET = object()

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
    # Engine dispatch. Meta tag wins; else inferred from the invocation form;
    # else llamacpp. engine_declared=False marks a script the scanner should
    # backfill the tag into.
    engine: str = "llamacpp"
    engine_declared: bool = False
    # Optional user-declared model weights size (GB) for display; vLLM models
    # have no GGUF to size from when the dir path can't be summed.
    vram_gb: Optional[float] = None
    # Measured KV bytes per token. A cached *measurement* (estimated from
    # config.json, calibrated from a real load's log), not a setting — the same
    # category as kv_cache_multiplier. Settings live in the args, never here.
    kv_bytes_per_token: Optional[int] = None
    # Heaviest rank's weight footprint, and the runtime overhead observed on that
    # rank. Both measured from a real load — pipeline ranks are not equal, and
    # the heaviest one decides whether a placement fits.
    weights_max_gb: Optional[float] = None
    runtime_overhead_gb: Optional[float] = None
    # GPU count the measurements above were taken at. Weights per rank, runtime
    # overhead and KV bytes/token all scale with layers-per-rank, so a profile
    # measured on 4 cards is wrong for 3 — `load_profile` discards it rather
    # than predicting from stale numbers.
    calibrated_gpus: Optional[int] = None
    # Layer split the measurements were taken at. Weights per rank follow the
    # partition directly, so re-balancing invalidates them just as surely as
    # changing the card count does.
    calibrated_partition: Optional[str] = None
    # VLLM_PP_LAYER_PARTITION assignment from the body, parsed to ints.
    pp_layer_partition: Optional[List[int]] = None

    # Launch-args fence contents
    llama_bin: Optional[str] = None  # binary path used in the exec line
    model_path: Optional[str] = None  # value of -m / --model
    args: Dict[str, Optional[str]] = field(default_factory=dict)  # all launch args incl. --port, --host
    # The same args as written in the script, before $VAR resolution. A save
    # re-emits the raw form for any arg the user didn't change, so shell
    # indirection like --max-model-len "$MAXLEN" survives a structured edit.
    raw_args: Dict[str, Optional[str]] = field(default_factory=dict)
    raw_model_path: Optional[str] = None
    # Each canonical flag as it was spelled (-a for --alias, -tp for
    # --tensor-parallel-size), in script order. llama.cpp's -m / --model is
    # included at its position. A save re-emits arguments in this order and
    # spelling, so an edit changes only the lines it touches.
    raw_keys: Dict[str, str] = field(default_factory=dict)

    # Shell-level state (read from script body, not args)
    cuda_visible_devices: Optional[List[int]] = None  # parsed CUDA_VISIBLE_DEVICES, None if absent
    # Assignments above the invocation (VLLM_PP_LAYER_PARTITION, MODEL, …).
    # Real launch config that never appears in the argument list.
    env_assignments: Dict[str, str] = field(default_factory=dict)

    # Original text and parse warnings
    raw_text: str = ""
    warnings: List[str] = field(default_factory=list)
    has_meta_fence: bool = False
    has_invocation: bool = False

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
        raw = self.args.get("--tensor-split")
        if raw is None:
            return None
        try:
            parts = [float(p.strip()) for p in str(raw).split(",") if p.strip()]
        except ValueError:
            return None
        if not parts:
            return None
        return parts

    def _int_arg(self, key: str, default: int = 1) -> int:
        try:
            v = self.args.get(key)
            return int(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    @property
    def tensor_parallel_size(self) -> int:
        return self._int_arg("--tensor-parallel-size", 1) if self.engine == "vllm" else 1

    @property
    def pipeline_parallel_size(self) -> int:
        return self._int_arg("--pipeline-parallel-size", 1) if self.engine == "vllm" else 1

    @property
    def gpu_memory_utilization(self) -> float:
        """vLLM's per-GPU claim fraction (--gpu-memory-utilization, default 0.9)."""
        raw = self.args.get("--gpu-memory-utilization")
        try:
            return float(raw) if raw is not None else 0.9
        except (TypeError, ValueError):
            return 0.9

    @property
    def kv_cache_memory_bytes(self) -> Optional[int]:
        """Explicit vLLM KV pool budget, or None when sized by utilization."""
        raw = self.args.get("--kv-cache-memory-bytes")
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    # ---- context / concurrency (derived from arguments, never metadata) ----

    @property
    def concurrency(self) -> Optional[int]:
        """Conversations servable at once, or None when the engine decides.

        Neither engine states this statically: llama.cpp publishes --parallel as
        `-1 = auto` (resolved to 4 slots at runtime) and vLLM's --max-num-seqs
        help says `(default: None)`, computed in create_engine_config. So an
        absent flag means "engine decides" and must not be shown as a number.
        """
        key = "--max-num-seqs" if self.engine == "vllm" else "--parallel"
        raw = self.args.get(key)
        try:
            value = int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None
        if value is not None and value < 0:
            return None  # llama.cpp's -1 sentinel is "auto"
        return value

    @property
    def load_profile(self) -> Dict[str, Optional[float]]:
        """Measured figures from a real load, or empties when they don't apply.

        Everything here scales with layers-per-rank, so a profile captured at a
        different GPU count is discarded rather than used to predict.
        """
        current_partition = (
            ",".join(str(n) for n in self.pp_layer_partition)
            if self.pp_layer_partition else ""
        )
        stale_reason = None
        if self.calibrated_gpus is not None and self.calibrated_gpus != self.gpu_count:
            stale_reason = f"{self.calibrated_gpus} GPU(s)"
        elif (
            self.calibrated_partition is not None
            and self.calibrated_partition != current_partition
        ):
            stale_reason = f"layer split {self.calibrated_partition}"
        if stale_reason:
            return {"weights_max_gb": None, "runtime_overhead_gb": None,
                    "kv_bytes_per_token": None, "stale_from": stale_reason}
        return {"weights_max_gb": self.weights_max_gb,
                "runtime_overhead_gb": self.runtime_overhead_gb,
                "kv_bytes_per_token": self.kv_bytes_per_token,
                "stale_from": None}

    @property
    def kv_unified(self) -> bool:
        """llama.cpp shares one KV buffer across slots.

        Enabled implicitly when --parallel is absent (server.cpp sets
        n_parallel=4, kv_unified=true), or explicitly via --kv-unified. With it
        on, every slot sees the whole --ctx-size; with it off the pool is
        divided (n_ctx_seq = n_ctx / n_seq_max).
        """
        if self.engine == "vllm":
            return False
        if "--kv-unified" in self.args:
            return True
        if "--no-kv-unified" in self.args:
            return False
        raw = self.args.get("--parallel")
        if raw is None:
            return True  # absent -> auto
        try:
            return int(raw) < 0  # the explicit -1 "auto" sentinel
        except (TypeError, ValueError):
            return False

    @property
    def context_length(self) -> Optional[int]:
        """Context available to a single conversation.

        llama.cpp: --ctx-size when the KV buffer is unified, else divided by
        --parallel. vLLM: --max-model-len.
        """
        if self.engine == "vllm":
            raw = self.args.get("--max-model-len")
            try:
                return int(raw) if raw is not None else None
            except (TypeError, ValueError):
                return None
        raw = self.args.get("--ctx-size")
        try:
            ctx = int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None
        if ctx is None:
            return None
        if self.kv_unified:
            return ctx
        return ctx // max(1, self.concurrency or 1)

    @property
    def gpu_count(self) -> int:
        """Number of GPUs this model wants. 1 if no --tensor-split (or split=[1]).

        vLLM engine: tensor_parallel_size × pipeline_parallel_size (exact count;
        vLLM splits evenly, uneven pipeline splits go via VLLM_PP_LAYER_PARTITION
        in the preamble and don't change the count).

        Note: a flexible-split model (--split-mode layer/row without
        --tensor-split) reports 1 here because its count is not fixed by the
        script — it's chosen at load time. Use `is_flexible_split` to detect
        that case; the GPU count then comes from the user's selection / CVD.
        """
        if self.engine == "vllm":
            return max(1, self.tensor_parallel_size * self.pipeline_parallel_size)
        weights = self.tensor_split_weights
        if weights is None:
            return 1
        return len(weights)

    @property
    def split_mode(self) -> Optional[str]:
        """Value of --split-mode / -sm, lowercased. None if not set."""
        raw = self.args.get("--split-mode")
        return str(raw).strip().lower() if raw else None

    @property
    def is_flexible_split(self) -> bool:
        """True when the model splits across a user-chosen set of GPUs.

        This holds when --split-mode is explicitly `layer` or `row` (llama.cpp's
        multi-GPU modes) and no --tensor-split fixes the count/proportions. The
        model then splits evenly across whatever GPUs are made visible via
        CUDA_VISIBLE_DEVICES, so the GPU count is the user's selection rather
        than a value baked into the script. (`--split-mode none` and the absence
        of --split-mode both mean single-GPU here.)
        """
        if self.engine == "vllm":
            return False
        if self.tensor_split_weights is not None:
            return False
        return self.split_mode in ("layer", "row")

    def is_configured(self) -> Tuple[bool, Optional[str]]:
        """Return (configured, reason_if_not)."""
        if not self.has_invocation:
            return False, "no launch invocation found"
        if not self.model_path:
            return False, (
                "no model in launch invocation"
                if self.engine == "vllm"
                else "no -m / --model in launch args"
            )
        if not Path(self.model_path).exists():
            return False, f"model path not found: {self.model_path}"
        if self.port is None:
            return False, "--port not set"
        if self.engine == "vllm":
            # Decision 7: a vLLM model states what it wants rather than taking
            # 90% of the card. All three are required because L and C alone do
            # not size the pool — leave the budget out and the engine still
            # claims util × capacity regardless of what the other two say.
            if self.context_length is None:
                return False, "--max-model-len not set (context per conversation)"
            if self.concurrency is None:
                return False, "--max-num-seqs not set (concurrency)"
            if self.kv_cache_memory_bytes is None:
                return False, "--kv-cache-memory-bytes not set (KV pool budget)"
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


# ---------- shell variable resolution ----------

_VAR_REF_RE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}?")


def _lookup_assignment(name: str, text: str) -> Optional[str]:
    """Value of the *last* `NAME=value` assignment in the script.

    Shell runs top to bottom, so a later assignment wins — the same rule
    _extract_cuda_visible_devices already applies. Matching all three quoting
    forms in one pass matters: trying double-quoted, then single, then bare
    patterns in sequence would let quoting style decide the winner instead of
    position. (Assignments inside conditionals can't be resolved without
    evaluating the script; last-wins is the closest static approximation.)
    """
    pattern = re.compile(
        rf'^[ \t]*(?:export[ \t]+)?{re.escape(name)}[ \t]*=[ \t]*'
        rf'(?:"([^"]*)"|\'([^\']*)\'|(\S+))',
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    for group in (1, 2, 3):
        if last.group(group) is not None:
            return last.group(group)
    return None


def resolve_var_refs(token: str, full_text: str, warnings: Optional[List[str]] = None) -> str:
    """Substitute $NAME / ${NAME} / ${NAME:-default} using assignments in the script.

    Handles the common self-referencing default idiom (`MAXLEN="${MAXLEN:-524288}"`)
    by falling back to the default once a name is already being resolved.
    Unresolvable references are left literal (with a warning appended).
    """
    def _resolve(tok: str, seen: frozenset) -> str:
        def repl(m: re.Match) -> str:
            name, default = m.group(1), m.group(2)
            if name not in seen:
                val = _lookup_assignment(name, full_text)
                if val is not None:
                    return _resolve(val, seen | {name})
            if default is not None:
                return _resolve(default, seen | {name})
            if warnings is not None:
                warnings.append(f"unresolved shell variable ${name}")
            return m.group(0)
        return _VAR_REF_RE.sub(repl, tok)

    return _resolve(token, frozenset())


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


def _arg_tokens_to_dict(
    tokens: List[str], warnings: List[str]
) -> Tuple[Dict[str, Optional[str]], Optional[str], Dict[str, str]]:
    """Walk a list of arg tokens into an ordered dict.

    Returns (args_dict, model_path, raw_keys). model_path is extracted from
    -m / --model and removed from args_dict (the renderer puts it back).
    """
    args: Dict[str, Optional[str]] = {}
    raw_keys: Dict[str, str] = {}
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
        raw_keys.setdefault(canonical, tok)

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

    return args, model_path, raw_keys


def _set_llama_args(ps: ParsedScript, tokens: List[str]) -> None:
    """Populate ps from llama-server arg tokens. llama.cpp values are kept as
    written (no $VAR resolution), so the raw forms equal the parsed ones."""
    ps.args, ps.model_path, ps.raw_keys = _arg_tokens_to_dict(tokens, ps.warnings)
    ps.raw_args = dict(ps.args)
    ps.raw_model_path = ps.model_path


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
        eng = (meta.get("engine") or "").strip().lower()
        if eng in VALID_ENGINES:
            ps.engine = eng
            ps.engine_declared = True
        elif eng:
            ps.warnings.append(f"meta engine={eng!r} is not one of {VALID_ENGINES}; treating as llamacpp")
        cp = meta.get("calibrated_partition")
        if cp:
            ps.calibrated_partition = str(cp).strip()
        cg = meta.get("calibrated_gpus")
        if cg is not None and cg != "":
            try:
                ps.calibrated_gpus = int(cg)
            except ValueError:
                ps.warnings.append(f"meta calibrated_gpus={cg!r} is not an integer")
        for fname in ("weights_max_gb", "runtime_overhead_gb"):
            fv = meta.get(fname)
            if fv is not None and fv != "":
                try:
                    setattr(ps, fname, float(fv))
                except ValueError:
                    ps.warnings.append(f"meta {fname}={fv!r} is not a number")
        kvb = meta.get("kv_bytes_per_token")
        if kvb is not None and kvb != "":
            try:
                ps.kv_bytes_per_token = int(kvb)
            except ValueError:
                ps.warnings.append(f"meta kv_bytes_per_token={kvb!r} is not an integer")
        vg = meta.get("vram_gb")
        if vg is not None and vg != "":
            try:
                ps.vram_gb = float(vg)
            except ValueError:
                ps.warnings.append(f"meta vram_gb={vg!r} is not a number")
    else:
        ps.warnings.append("meta fence missing")

    # Launch invocation: the exec/llama-server line in the body below the meta fence.
    body = text
    if meta_fence is not None:
        lines = text.splitlines()
        body = "\n".join(lines[meta_fence[1] + 1:])
    _populate_from_body(ps, body, text)

    # Shell-level CVD (read from the whole script)
    ps.cuda_visible_devices = _extract_cuda_visible_devices(text)

    ps.env_assignments = extract_assignments(text)


    # vLLM uneven pipeline split (env var, preserved as preamble — doc 21 §4).
    raw_part = _lookup_assignment("VLLM_PP_LAYER_PARTITION", text)
    if raw_part is not None:
        resolved = resolve_var_refs(raw_part, text, ps.warnings)
        try:
            ps.pp_layer_partition = [int(p.strip()) for p in resolved.split(",") if p.strip()]
        except ValueError:
            ps.warnings.append(f"VLLM_PP_LAYER_PARTITION={resolved!r} is not a comma-separated int list")
        if (
            ps.pp_layer_partition is not None
            and ps.engine == "vllm"
            and len(ps.pp_layer_partition) != ps.pipeline_parallel_size
        ):
            ps.warnings.append(
                f"VLLM_PP_LAYER_PARTITION has {len(ps.pp_layer_partition)} entries but "
                f"--pipeline-parallel-size is {ps.pipeline_parallel_size} (vLLM will refuse to start)"
            )

    # Only now is pp_layer_partition populated, so the staleness comparison
    # against calibrated_partition is meaningful.
    if ps.engine == "vllm":
        stale_from = ps.load_profile["stale_from"]
        if stale_from:
            ps.warnings.append(
                f"VRAM measurements were taken with {stale_from}; this config has "
                f"changed, so they are ignored until the model is loaded again "
                f"and re-measured"
            )

    return ps


def _is_vllm_tokens(tokens: List[str]) -> bool:
    """True when the token list is a `<...>/vllm serve ...` invocation."""
    if len(tokens) < 2 or tokens[1] != "serve":
        return False
    return Path(tokens[0]).name == "vllm" or tokens[0].rstrip('"').endswith("/vllm") or tokens[0] == "vllm"


def _populate_from_body(ps: ParsedScript, body: str, full_text: str) -> None:
    """Find the launch invocation (llama-server or vllm serve) in `body` and
    populate ps.{llama_bin, args, model_path}."""
    collapsed = _collapse_continuations(body)
    invocation = None
    for line in collapsed.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("#!"):
            continue
        if re.match(r"^(export\s+)?[A-Za-z_][A-Za-z0-9_]*\s*=", s):
            continue
        if (
            s.startswith("exec ") or "$LLAMA_BIN" in s or "${LLAMA_BIN" in s
            or "llama-server" in s or re.search(r"vllm[\"']?\s+serve\b", s)
        ):
            invocation = s
            break

    if invocation is None:
        ps.has_invocation = False
        ps.warnings.append("no launch invocation found")
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

    # Resolve $VAR in the binary token so `"$VENV/bin/vllm"` is recognizable.
    binary = resolve_var_refs(tokens[0], full_text, ps.warnings)
    tokens = [binary] + tokens[1:]

    if _is_vllm_tokens(tokens):
        # Meta tag wins; an undeclared script with a vllm invocation is inferred.
        if not ps.engine_declared:
            ps.engine = "vllm"
        ps.has_invocation = True
        ps.llama_bin = binary
        ps.args, ps.model_path, ps.raw_args, ps.raw_model_path, ps.raw_keys = _vllm_arg_tokens_to_dict(
            tokens[2:], full_text, ps.warnings, manifest_for_binary(binary)
        )
        return

    ps.has_invocation = True
    ps.llama_bin = _resolve_llama_bin(binary, full_text)
    _set_llama_args(ps, tokens[1:])


def _vllm_arg_tokens_to_dict(
    tokens: List[str], full_text: str, warnings: List[str],
    manifest: Optional[Dict[str, dict]] = None,
) -> Tuple[Dict[str, Optional[str]], Optional[str], Dict[str, Optional[str]], Optional[str], Dict[str, str]]:
    """Walk `vllm serve` arg tokens (after 'serve') into a dict.

    The model is the positional token before any flag. Whether a flag takes a
    value, and how short forms (-tp, -pp, -sc) map to their long names, come
    from the captured manifest; without one we fall back to the same next-token
    heuristic llama.cpp parsing uses. Multi-value flags (nargs+, e.g.
    --served-model-name a b c) accumulate trailing non-flag tokens onto the
    last flag, space-joined. $VAR references are resolved against the body.
    """
    manifest = (manifest if manifest is not None else _VLLM_MANIFEST) or {}
    known_flags: Dict[str, dict] = manifest.get("flags", {})
    aliases: Dict[str, str] = manifest.get("aliases", {})

    args: Dict[str, Optional[str]] = {}
    raw: Dict[str, Optional[str]] = {}
    raw_keys: Dict[str, str] = {}
    model_path: Optional[str] = None
    raw_model_path: Optional[str] = None
    last_flag: Optional[str] = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("-"):
            value = resolve_var_refs(tok, full_text, warnings)
            if model_path is None and last_flag is None:
                model_path = value
                raw_model_path = tok
            elif last_flag is not None:
                prev = args.get(last_flag)
                args[last_flag] = f"{prev} {value}" if prev else value
                prev_raw = raw.get(last_flag)
                raw[last_flag] = f"{prev_raw} {tok}" if prev_raw else tok
            else:
                warnings.append(f"stray token in vllm invocation: {tok!r}")
            i += 1
            continue

        # Normalize short forms (-tp → --tensor-parallel-size) so downstream
        # lookups like gpu_count see one canonical key.
        canonical = aliases.get(tok, tok)
        raw_keys.setdefault(canonical, tok)
        next_tok = tokens[i + 1] if i + 1 < len(tokens) else None
        next_is_flag = next_tok is not None and (
            next_tok.startswith("--")
            or (len(next_tok) >= 2 and next_tok[0] == "-" and next_tok[1].isalpha())
        )

        spec = known_flags.get(canonical) or known_flags.get(tok)
        if spec is not None:
            takes_value = spec.get("takes_value", False)
            if takes_value and (next_tok is None or next_is_flag):
                warnings.append(f"{tok} expects a value but none was given")
                takes_value = False
        else:
            if known_flags:
                warnings.append(f"unknown vllm option: {tok}")
            takes_value = next_tok is not None and not next_is_flag

        if not takes_value:
            args[canonical] = None
            raw[canonical] = None
            last_flag = None  # boolean flags never accumulate values
            i += 1
        else:
            args[canonical] = resolve_var_refs(tokens[i + 1], full_text, warnings)
            raw[canonical] = tokens[i + 1]
            last_flag = canonical
            i += 2

    return args, model_path, raw, raw_model_path, raw_keys


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
            _set_llama_args(ps, tokens)
            ps.warnings.append("no llama-server invocation found; parsed pasted text as args only")
        except ValueError as e:
            ps.warnings.append(f"could not tokenize pasted text: {e}")

    # Resolved decision #1: paste-import always overwrites display_name from --alias
    alias = ps.args.get("--alias")
    if alias and alias != _DISPLAY_NAME_VAR_REF:
        ps.display_name = alias

    return ps


_CVD_KEY = "CUDA_VISIBLE_DEVICES"
_DISPLAY_NAME_KEY = "DISPLAY_NAME"
_DISPLAY_NAME_VAR_REF = f"${_DISPLAY_NAME_KEY}"


def _parse_cvd_value(raw: str) -> Optional[List[int]]:
    """Parse a 'CUDA_VISIBLE_DEVICES=...' value into a list of ints, or None on malformed."""
    s = raw.strip().strip('"').strip("'")
    # Guarded default form: ${CUDA_VISIBLE_DEVICES:-0,1,2} → parse the default.
    m = re.match(rf"^\$\{{{_CVD_KEY}:-([^}}]*)\}}$", s)
    if m:
        s = m.group(1)
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
    engine: str = "llamacpp",
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
    meta_lines.append(f"# engine: {engine}")
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

    # Default the API-visible model name (-a/--alias) to a $DISPLAY_NAME
    # reference so it tracks display_name forever without ever needing the
    # args themselves patched again. Never overrides an alias the user
    # already set explicitly (literal or otherwise).
    if engine == "llamacpp" and display_name and not any(k in args for k in ("-a", "--alias")):
        args = dict(args)
        args["--alias"] = _DISPLAY_NAME_VAR_REF

    invocation = _render_invocation(llama_bin, model_path, args)
    out = f"#!/usr/bin/env bash\n{meta_block}\n{body}\n{invocation}\n"

    # Keep the DISPLAY_NAME= body variable in sync whenever anything in the
    # script references it (fresh injection above, or preserved from before).
    if display_name and _DISPLAY_NAME_VAR_REF in out:
        out = patch_display_name_var(out, display_name)

    return out


_SHELL_VAR_REF_RE = re.compile(r'^\$\{?[A-Za-z_][A-Za-z0-9_]*\}?$')


def _is_shell_var_ref(value: str) -> bool:
    """True for a bare $NAME or ${NAME} reference and nothing else around it."""
    return bool(_SHELL_VAR_REF_RE.match(value))


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
        elif _is_shell_var_ref(str(value)):
            # A bare $NAME / ${NAME} reference: emit double-quoted so the shell
            # expands it, instead of shlex.quote turning it into a dead literal.
            arg_lines.append(f'  {key} "{value}"')
        else:
            arg_lines.append(f"  {key} {shlex.quote(str(value))}")
    if arg_lines:
        for line in arg_lines[:-1]:
            lines.append(line + " \\")
        lines.append(arg_lines[-1])
    return "\n".join(lines)


# ---------- fence patching (in-place rewrites) ----------

def _render_meta_line(key: str, value) -> str:
    """Render one `# key: value` meta line."""
    if key in ("vram_gb", "weights_max_gb", "runtime_overhead_gb"):
        return f"# {key}: {float(value):g}"
    return f"# {key}: {value}"


def patch_meta_fence(
    text: str,
    *,
    display_name=_UNSET,
    block_count=_UNSET,
    max_context=_UNSET,
    kv_cache_multiplier=_UNSET,
    health_timeout=_UNSET,
    engine=_UNSET,
    vram_gb=_UNSET,
    kv_bytes_per_token=_UNSET,
    weights_max_gb=_UNSET,
    runtime_overhead_gb=_UNSET,
    calibrated_gpus=_UNSET,
    calibrated_partition=_UNSET,
) -> str:
    """Merge the given fields into the meta fence, preserving everything else.

    Three distinct states, so that forgetting a field can't destroy it:
      - field omitted      → existing value left untouched
      - field = <value>    → set (added if absent, updated in place if present)
      - field = None       → its line is removed

    Lines the parser doesn't model — user comments, unrecognized `# key: value`
    entries — are preserved verbatim in their original positions. Existing keys
    keep their position; newly added keys are appended in canonical order.
    The fence is created after the shebang if the script has none.
    """
    updates = {
        "display_name": display_name,
        "engine": engine,
        "vram_gb": vram_gb,
        "block_count": block_count,
        "max_context": max_context,
        "kv_cache_multiplier": kv_cache_multiplier,
        "health_timeout": health_timeout,
        "kv_bytes_per_token": kv_bytes_per_token,
        "weights_max_gb": weights_max_gb,
        "runtime_overhead_gb": runtime_overhead_gb,
        "calibrated_gpus": calibrated_gpus,
        "calibrated_partition": calibrated_partition,
    }
    updates = {k: v for k, v in updates.items() if v is not _UNSET}
    # An empty string display_name behaves like a delete (legacy callers pass "").
    for k, v in list(updates.items()):
        if v == "":
            updates[k] = None

    fence = _extract_fence(text, META_BEGIN, META_END)
    existing_inner = fence[2].splitlines() if fence is not None else []

    out: List[str] = []
    seen = set()
    for line in existing_inner:
        stripped = line.strip()
        key = None
        if stripped.startswith("#") and ":" in stripped:
            candidate = stripped.lstrip("#").split(":", 1)[0].strip()
            if candidate in updates:
                key = candidate
        if key is None:
            out.append(line)  # unmodelled content — preserve verbatim
            continue
        if key in seen:
            continue  # collapse duplicate keys onto the first occurrence
        seen.add(key)
        value = updates[key]
        if value is not None:
            out.append(_render_meta_line(key, value))
        # value None → drop the line

    # Append fields that weren't already present, in canonical order.
    for key in ("display_name", "engine", "vram_gb", "kv_bytes_per_token",
                "weights_max_gb", "runtime_overhead_gb", "calibrated_gpus",
                "calibrated_partition",
                "block_count", "max_context", "kv_cache_multiplier", "health_timeout"):
        if key in updates and key not in seen and updates[key] is not None:
            out.append(_render_meta_line(key, updates[key]))

    new_block = "\n".join([META_BEGIN, *out, META_END])

    lines = text.splitlines()
    if fence is not None:
        begin_idx, end_idx, _ = fence
        lines[begin_idx:end_idx + 1] = new_block.splitlines()
    else:
        insert_at = 1 if lines and lines[0].startswith("#!") else 0
        lines[insert_at:insert_at] = new_block.splitlines()
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


# Shell variables the app manages itself; the user edits these through their own
# controls, not as free-form environment rows.
MANAGED_ASSIGNMENTS = {_CVD_KEY, _DISPLAY_NAME_KEY}

_ASSIGN_LINE_RE = re.compile(
    r'^[ \t]*(?P<export>export[ \t]+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)='
    r'(?P<value>"[^"]*"|\'[^\']*\'|[^\s#]*)(?P<rest>.*)$'
)


def extract_assignments(text: str) -> Dict[str, str]:
    """Shell assignments above the invocation, as written (quotes stripped).

    These carry real launch configuration that never appears in the argument
    list — VLLM_PP_LAYER_PARTITION, VLLM_WORKER_MULTIPROC_METHOD, the MODEL /
    MAXLEN style variable defaults — so the config UI has to be able to show and
    edit them. Later assignments win, matching the shell.
    """
    span = find_invocation_span(text)
    limit = span[0] if span else len(text.splitlines())
    found: Dict[str, str] = {}
    for line in text.splitlines()[:limit]:
        m = _ASSIGN_LINE_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        if name in MANAGED_ASSIGNMENTS:
            continue
        value = m.group("value")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        found[name] = value
    return found


def patch_env_assignment(text: str, name: str, value: Optional[str]) -> str:
    """Set, change, or remove one shell assignment, leaving the rest untouched.

    Rewrites the last existing assignment in place (preserving any trailing
    comment), inserts one just above the invocation when absent, and removes
    every assignment of `name` when value is None.
    """
    lines = text.splitlines()
    span = find_invocation_span(text)
    limit = span[0] if span else len(lines)
    trailing = "\n" if text.endswith("\n") else ""

    matches = [
        i for i in range(limit)
        if (m := _ASSIGN_LINE_RE.match(lines[i])) and m.group("name") == name
    ]

    if value is None:
        for i in reversed(matches):
            del lines[i]
        return "\n".join(lines) + trailing

    quoted = value if re.fullmatch(r"[A-Za-z0-9_./:,+-]*", value) else f'"{value}"'
    if matches:
        keep = matches[-1]
        m = _ASSIGN_LINE_RE.match(lines[keep])
        prefix = "export " if m.group("export") else ""
        indent = re.match(r"^[ \t]*", lines[keep]).group(0)
        comment = m.group("rest")
        lines[keep] = f"{indent}{prefix}{name}={quoted}{comment}"
        for i in reversed(matches[:-1]):
            del lines[i]
        return "\n".join(lines) + trailing

    insert_at = limit
    while insert_at > 0 and not lines[insert_at - 1].strip():
        insert_at -= 1
    lines[insert_at:insert_at] = [f"export {name}={quoted}"]
    return "\n".join(lines) + trailing


def find_invocation_span(text: str) -> Optional[Tuple[int, int]]:
    """Line span [start, end) of the launch invocation, continuations included.

    Returns None when no invocation line is found.
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if re.match(r"^(export\s+)?[A-Za-z_][A-Za-z0-9_]*\s*=", s):
            continue
        if (
            s.startswith("exec ") or "$LLAMA_BIN" in s or "${LLAMA_BIN" in s
            or "llama-server" in s or re.search(r"vllm[\"']?\s+serve\b", s)
        ):
            start = i
            break
    if start is None:
        return None
    end = start
    while end < len(lines) and lines[end].rstrip().endswith("\\"):
        end += 1
    return start, min(end + 1, len(lines))


def patch_exec_invocation(
    text: str,
    *,
    args: Dict[str, Optional[str]],
    model_path: Optional[str] = None,
    parsed: Optional[ParsedScript] = None,
    binary: Optional[str] = None,
) -> str:
    """Rewrite only the exec line, preserving every other byte of the script.

    The structured save for both engines. Arguments keep the script's own order
    and spelling (`-a` stays `-a`); new ones are appended. When `parsed` is
    supplied, an argument whose value is unchanged is re-emitted exactly as
    written — so `--max-model-len "$MAXLEN"` stays a variable reference instead
    of being frozen to the resolved number. Changed values are written literally.

    `binary` replaces the executable token when it differs from the parsed one;
    callers whose exec line names a variable (`"$LLAMA_BIN"`) patch that
    variable's assignment instead and leave this unset.
    """
    span = find_invocation_span(text)
    if span is None:
        raise ValueError("no launch invocation found to patch")
    start, end = span
    lines = text.splitlines()

    original = lines[start]
    # Preserve the original binary token (e.g. "$VLLM_BIN") rather than its
    # resolved path, and whether the line was exec'd.
    try:
        head_tokens = shlex.split(_collapse_continuations("\n".join(lines[start:end])), posix=True)
    except ValueError:
        head_tokens = []
    is_exec = original.strip().startswith("exec ")
    m = re.match(r'^\s*(?:exec\s+)?("[^"]*"|\'[^\']*\'|\S+)', original)
    binary_token = m.group(1) if m else ((parsed.llama_bin if parsed else None) or "")
    if binary and (parsed is None or binary != parsed.llama_bin):
        binary_token = shlex.quote(binary)
    is_vllm = bool(parsed and parsed.engine == "vllm") or (
        len(head_tokens) > 1 and head_tokens[1] == "serve"
    )
    raw_keys = parsed.raw_keys if parsed else {}

    indent = re.match(r"^(\s*)", original).group(1)

    def _requote(raw: str) -> str:
        """Re-emit a raw token. shlex stripped its quotes, so a variable
        reference must be re-wrapped — bare $VAR word-splits on spaces and
        vanishes when empty."""
        return f'"{raw}"' if "$" in raw else shlex.quote(raw)

    def _value_token(key: str, value: Optional[str]) -> Optional[str]:
        """Original spelling when unchanged, else a freshly quoted literal."""
        if value is None:
            return None
        if parsed is not None:
            prior = parsed.args.get(key)
            raw = parsed.raw_args.get(key)
            if raw is not None and str(prior) == str(value):
                return _requote(raw)
        if _is_shell_var_ref(str(value)):
            return f'"{value}"'
        return shlex.quote(str(value))

    model = model_path if model_path is not None else (parsed.model_path if parsed else None)
    model_token = None
    if model is not None:
        if parsed and parsed.raw_model_path and parsed.model_path == model:
            model_token = _requote(parsed.raw_model_path)
        else:
            model_token = shlex.quote(str(model))

    parts: List[str] = []
    if is_vllm:
        # The model is positional and belongs on the `serve` line, matching how
        # these scripts are written by hand.
        prefix = f'{"exec " if is_exec else ""}{binary_token} serve{" " + model_token if model_token else ""}'
    else:
        prefix = f'{"exec " if is_exec else ""}{binary_token}'

    # Script order first (llama.cpp's --model sits at its own position), then
    # anything new in the order given.
    order = [k for k in raw_keys if k in args or (k == "--model" and not is_vllm)]
    if not is_vllm and "--model" not in order:
        order.insert(0, "--model")
    order += [k for k in args if k not in order and k not in ("-m", "--model")]
    for key in order:
        spelled = raw_keys.get(key, key)
        if key == "--model":
            if model_token:
                parts.append(f"{raw_keys.get('--model', '-m')} {model_token}")
            continue
        token = _value_token(key, args[key])
        parts.append(spelled if token is None else f"{spelled} {token}")

    if not parts:
        rendered = [f"{indent}{prefix}"]
    else:
        rendered = [f"{indent}{prefix} \\"]
        for part in parts[:-1]:
            rendered.append(f"{indent}  {part} \\")
        rendered.append(f"{indent}  {parts[-1]}")

    lines[start:end] = rendered
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def patch_display_name_var(text: str, display_name: Optional[str]) -> str:
    """Insert, replace, or remove the canonical DISPLAY_NAME= line.

    Keeps a `DISPLAY_NAME='...'` shell assignment in the body in sync with the
    meta fence's display_name, so `-a "$DISPLAY_NAME"` in the invocation always
    reflects the current name without any patching of the args themselves —
    same idea as `patch_cuda_visible_devices` for CUDA_VISIBLE_DEVICES.
    Idempotent: no-ops if the script already has the canonical line.
    """
    lines = text.splitlines()
    trailing_newline = "\n" if text.endswith("\n") else ""

    standalone_re = re.compile(
        rf'^[ \t]*{_DISPLAY_NAME_KEY}=(?:"[^"]*"|\'[^\']*\'|\S*)[ \t]*(?:#.*)?$'
    )

    if display_name is None:
        cleaned = [ln for ln in lines if not standalone_re.match(ln)]
        return "\n".join(cleaned) + trailing_newline

    # single-quoted: the value is free text and must not itself be expanded
    escaped = display_name.replace("'", "'\\''")
    new_line = f"{_DISPLAY_NAME_KEY}='{escaped}'"

    existing_idx = next((i for i, ln in enumerate(lines) if standalone_re.match(ln)), None)
    if existing_idx is not None:
        if lines[existing_idx] == new_line:
            return text  # already canonical, no-op
        lines[existing_idx] = new_line
        return "\n".join(lines) + trailing_newline

    # Insert after LLAMA_BIN= if present, else after the meta fence, else at top.
    insert_at: Optional[int] = None
    for i, line in enumerate(lines):
        if re.match(r"^[ \t]*LLAMA_BIN[ \t]*=", line):
            insert_at = i + 1
            break
    if insert_at is None:
        for i, line in enumerate(lines):
            if line.strip() == META_END:
                insert_at = i + 1
                break
    if insert_at is None:
        insert_at = 1 if lines and lines[0].startswith("#!") else 0

    lines[insert_at:insert_at] = [new_line]
    return "\n".join(lines) + trailing_newline


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
    # A trailing comment still makes this a standalone assignment. Requiring the
    # line to end at the value left commented ones behind, so writing a new GPU
    # selection produced two CVD lines and the parser then read the stale one.
    standalone_re = re.compile(
        rf'^[ \t]*(?:export[ \t]+)?{_CVD_KEY}=(?:"[^"]*"|\'[^\']*\'|\S+)[ \t]*(?:#.*)?$'
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
    decision #2). --batch-size was raised from the original JSON skeleton's 512
    after a real config using it measured ~4x worse prompt-processing throughput
    than the llama-server upstream default (2048). KV stays f16 (faster than
    q8_0 - the dequant cost per token isn't worth the memory savings here).
    """
    args: Dict[str, Optional[str]] = {
        "--host": "0.0.0.0",
        "--gpu-layers": "999",
        "--ctx-size": "4096",
        "--cache-type-k": "f16",
        "--cache-type-v": "f16",
        "--batch-size": "2048",
        "--ubatch-size": "512",
        "--threads": "12",
        "-fa": "on",
    }
    return render_script(
        display_name=display_name,
        block_count=block_count,
        max_context=max_context,
        kv_cache_multiplier=kv_cache_multiplier,
        llama_bin=llama_bin,
        model_path=model_path,
        args=args,
        engine="llamacpp",
    )


# Optional skeleton args, emitted ONLY when present in the engine's captured
# flag manifest (engine_flags.get_vllm_serve_flags). Never assume a flag exists.
_VLLM_SKELETON_CANDIDATES: List[Tuple[str, Optional[str]]] = [
    ("--gpu-memory-utilization", "0.9"),
    ("--disable-log-requests", None),  # older builds; newer ones default quiet
]


def render_vllm_skeleton_script(
    *,
    display_name: str,
    model_path: str,
    vllm_bin: str,
    available_flags: Optional[set] = None,
) -> str:
    """Render a skeleton script for a newly-discovered vLLM model directory.

    --port is intentionally omitted so the model shows as unconfigured (same
    convention as llama.cpp skeletons). `available_flags` is the flag manifest
    captured from this executable's own `serve --help`; optional args are
    filtered against it, and with no manifest (None) none are emitted — a
    skeleton must never contain a flag the engine wasn't proven to accept.
    health_timeout defaults high because vLLM startup far exceeds llama.cpp's.
    """
    meta = "\n".join([
        META_BEGIN,
        f"# display_name: {display_name}",
        "# engine: vllm",
        "# health_timeout: 600",
        META_END,
    ])
    arg_lines = [f"  --host 0.0.0.0"]
    if available_flags:
        for flag, value in _VLLM_SKELETON_CANDIDATES:
            if flag in available_flags:
                arg_lines.append(f"  {flag} {value}" if value is not None else f"  {flag}")
    invocation = f'exec "$VLLM_BIN" serve {shlex.quote(model_path)} \\\n'
    invocation += " \\\n".join(arg_lines) + "\n"
    return (
        f"#!/usr/bin/env bash\n{meta}\n\n"
        f"VLLM_BIN={shlex.quote(vllm_bin)}\n\n"
        f"{invocation}"
    )
