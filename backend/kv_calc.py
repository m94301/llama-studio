"""KV-cache sizing for vLLM models (doc 21 §2.4).

Both engines allocate a KV pool of N tokens shared across conversations, so the
maximum simultaneous conversations at length L is N / L. The user states L
(context per conversation) and C (concurrency); everything else derives:

    N          = L × C  (+ headroom)
    kv_bytes   = N × bytes_per_token
    total VRAM = weights + overhead + kv_bytes

The forward direction (context → VRAM) and the inverse (VRAM → context) share
`vram_gb_for()` and `context_for_vram()` here so they cannot disagree — a
round-trip must land back on the same slider position.

`bytes_per_token` is estimated from the model's config.json and then calibrated
from a real load's log, which is the only trustworthy source for architectures
whose KV shape isn't the standard `2 × layers × kv_heads × head_dim × dtype`
(e.g. DeepSeek-family attention with a compressed latent or sparse indexer).
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

GIB = 1024 ** 3

# Bytes per element for the dtypes that appear in HF configs / --kv-cache-dtype.
DTYPE_BYTES = {
    "float32": 4.0, "float": 4.0, "fp32": 4.0,
    "bfloat16": 2.0, "float16": 2.0, "half": 2.0, "fp16": 2.0, "bf16": 2.0,
    "float8": 1.0, "fp8": 1.0, "fp8_e4m3": 1.0, "fp8_e5m2": 1.0, "int8": 1.0,
}

# Activation + CUDA-graph working memory, i.e. everything that is neither
# weights nor KV. Measured at 0.08 + 0.4 GiB on a Qwen2.5-0.5B load; it grows
# with model and batch size, so this is a deliberately loose default used only
# until a real load calibrates the model's actual figure.
DEFAULT_OVERHEAD_GB = 1.5

# Context is padded to this in llama.cpp (n_ctx_seq) — keeping every derived
# length on the same grid stops the engine silently rounding the pool down.
CONTEXT_GRANULARITY = 256

# Policy, not arithmetic: a pool sized to exactly L × C has no room for
# preemption or prefix caching, and vLLM refuses to start if the pool cannot
# hold one full request. The math helpers below default to 1.0 (no padding) so
# they stay honest; call sites that size a real budget apply this.
DEFAULT_KV_HEADROOM = 1.2


def max_context_from_config(model_dir: str) -> Optional[int]:
    """Trained context length from a HF config.json — the slider's upper bound.

    The vLLM counterpart of the GGUF `max_context` we cache for llama.cpp.
    """
    try:
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
    except Exception:
        return None
    if "max_position_embeddings" not in cfg and isinstance(cfg.get("text_config"), dict):
        cfg = cfg["text_config"]
    value = cfg.get("max_position_embeddings")
    try:
        return int(value) if value else None
    except (TypeError, ValueError):
        return None


def dtype_bytes(name: Optional[str], default: float = 2.0) -> float:
    if not name:
        return default
    return DTYPE_BYTES.get(str(name).strip().lower(), default)


def bytes_per_token_from_config(model_dir: str, kv_cache_dtype: Optional[str] = None) -> Optional[int]:
    """Estimate KV bytes per token from a HF model directory's config.json.

    Standard attention: 2 (K and V) × layers × kv_heads × head_dim × dtype.
    Returns None when the config doesn't describe a shape we can compute — the
    caller must then fall back to calibration from a real load rather than
    guessing.
    """
    try:
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
    except Exception as e:
        logger.debug(f"   kv_calc: no readable config.json in {model_dir}: {e}")
        return None

    # Some configs nest the real thing (multimodal wrappers, quantized repacks).
    if "num_hidden_layers" not in cfg and isinstance(cfg.get("text_config"), dict):
        cfg = cfg["text_config"]

    layers = cfg.get("num_hidden_layers")
    if not layers:
        return None

    heads = cfg.get("num_attention_heads")
    kv_heads = cfg.get("num_key_value_heads", heads)
    head_dim = cfg.get("head_dim")
    if not head_dim and heads and cfg.get("hidden_size"):
        head_dim = cfg["hidden_size"] // heads
    if not kv_heads or not head_dim:
        return None

    # A compressed-latent KV (DeepSeek-style MLA) is not 2 × kv_heads × head_dim;
    # decline rather than return a confidently wrong number.
    if "kv_lora_rank" in cfg:
        logger.debug("   kv_calc: MLA-style config; deferring to load-time calibration")
        return None

    per_element = dtype_bytes(kv_cache_dtype) if kv_cache_dtype and kv_cache_dtype != "auto" \
        else dtype_bytes(cfg.get("torch_dtype"))
    return int(2 * layers * kv_heads * head_dim * per_element)


# vLLM reports both numbers during startup; their ratio is the exact figure.
_KV_MEM_RE = re.compile(r"Available KV cache memory:\s*([\d.]+)\s*GiB")
_KV_TOK_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")


def calibrate_from_log(log_text: str, kv_budget_bytes: Optional[int] = None) -> Optional[int]:
    """Exact KV bytes per token, read back from a successful vLLM load's log.

    Two sources, because vLLM's reporting depends on how the pool was sized:
    with utilization-based sizing it prints "Available KV cache memory: X GiB",
    but with an explicit --kv-cache-memory-bytes that line is absent — so the
    known budget divided by the reported token count is the figure instead.
    """
    tok = _KV_TOK_RE.search(log_text)
    if not tok:
        return None
    tokens = int(tok.group(1).replace(",", ""))
    if tokens <= 0:
        return None
    mem = _KV_MEM_RE.search(log_text)
    if mem:
        return int(round(float(mem.group(1)) * GIB / tokens))
    if kv_budget_bytes:
        return int(round(kv_budget_bytes / tokens))
    return None


def pool_tokens(context_length: int, concurrency: int, headroom: float = 1.0) -> int:
    """Total KV tokens needed to serve `concurrency` conversations of `context_length`."""
    return int(max(1, context_length) * max(1, concurrency) * max(1.0, headroom))


def kv_gb(tokens: int, bytes_per_token: int) -> float:
    return tokens * bytes_per_token / GIB


def vram_gb_for(
    *,
    context_length: int,
    concurrency: int,
    bytes_per_token: int,
    weights_gb: float,
    overhead_gb: float = DEFAULT_OVERHEAD_GB,
    headroom: float = 1.0,
) -> Tuple[float, int]:
    """Forward solve: (total VRAM in GB, pool tokens) for a context/concurrency pair."""
    tokens = pool_tokens(context_length, concurrency, headroom)
    return round(weights_gb + overhead_gb + kv_gb(tokens, bytes_per_token), 2), tokens


def context_for_vram(
    *,
    vram_gb: float,
    concurrency: int,
    bytes_per_token: int,
    weights_gb: float,
    overhead_gb: float = DEFAULT_OVERHEAD_GB,
    headroom: float = 1.0,
    max_context: Optional[int] = None,
    granularity: int = CONTEXT_GRANULARITY,
) -> int:
    """Inverse solve: the largest context length that fits in `vram_gb`.

    Snaps *down* to `granularity` so the result is always achievable, and clamps
    to `max_context` when the model's trained length is known. Returns 0 when
    the budget cannot even cover weights + overhead.
    """
    available = vram_gb - weights_gb - overhead_gb
    if available <= 0 or bytes_per_token <= 0:
        return 0
    tokens = int(available * GIB / bytes_per_token / max(1.0, headroom))
    context = tokens // max(1, concurrency)
    context = (context // granularity) * granularity
    if max_context:
        context = min(context, (max_context // granularity) * granularity)
    return max(0, context)
