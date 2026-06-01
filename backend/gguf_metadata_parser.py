"""GGUF metadata parser for model architecture detection and KV cache multiplier computation.

This module handles the complexity of different model architectures:
- Standard multi-head attention
- GQA (Grouped Query Attention) and MQA (Multi-Query Attention)
- Hybrid attention+SSM models (e.g., Qwen3.6)

The parser extracts metadata once and returns a pre-adjusted kv_cache_multiplier
that works correctly with the standard VRAM formula, accounting for all architecture nuances.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# In-memory cache for GGUF metadata fields, keyed by (path, mtime).
# A full GGUFParser walk is CPU-bound and slow (~1.3s for large files), but the
# extracted field dict is small and reusable across rescans. The mtime invalidates
# automatically when the user replaces the model file.
_GGUF_FIELDS_CACHE: Dict[Tuple[str, int], dict] = {}


def _extract_gguf_fields(path: str) -> Optional[dict]:
    """Return the raw GGUF metadata dict for `path`, using a process-local cache.

    Cache key is (path, mtime), so editing the GGUF file invalidates the entry.
    Returns None if parsing fails.
    """
    try:
        mtime = int(Path(path).stat().st_mtime)
    except OSError:
        return None
    key = (path, mtime)
    # Negative entries (None) are cached too — a file that fails to parse will
    # keep failing until its mtime changes, so don't re-parse or re-log every scan.
    if key in _GGUF_FIELDS_CACHE:
        return _GGUF_FIELDS_CACHE[key]
    try:
        from gguf_parser import GGUFParser
        import time as _t
        t0 = _t.time()
        parser = GGUFParser(path)
        parser.parse()
        _GGUF_FIELDS_CACHE[key] = parser.metadata
        logger.debug(f"   [GGUF] Cached fields for {path} in {(_t.time()-t0)*1000:.0f}ms")
        return parser.metadata
    except Exception as e:
        logger.warning(f"⚠ GGUF parse failed for {path}: {type(e).__name__}: {e}")
        _GGUF_FIELDS_CACHE[key] = None
        return None


@dataclass
class GGUFMetadata:
    """Parsed GGUF metadata for VRAM calculation.

    All values are pre-adjusted for the model's architecture, so the VRAM formula
    can remain simple and uniform across all models.
    """
    block_count: Optional[int] = None
    max_context: Optional[int] = None
    kv_cache_multiplier: Optional[int] = None


def parse_gguf_metadata(
    model_path: str,
    reference_ctx: Optional[int] = None,
    reference_batch_size: Optional[int] = None,
    reference_k_bytes: float = 1.0,
    reference_v_bytes: float = 1.0,
) -> GGUFMetadata:
    """Extract and compute GGUF metadata for VRAM calculation.

    Handles:
    - Standard attention: multiplier = hidden_dim
    - GQA/MQA: multiplier = num_heads_kv * head_dim
    - Hybrid attention+SSM: multiplier = (num_heads_kv * head_dim) * (kv_active_layers / total_layers)
    - Sliding-window attention (e.g., Gemma 3+): multiplier folded so the standard
      formula gives the right total at a chosen reference context.
    - Compute-buffer overhead: an empirical estimate folded into the multiplier
      so the predicted total absorbs llama-server's per-batch scratch buffers.

    `reference_ctx`: the ctx_size to bake the SWA adjustment for. When set, the
    returned multiplier yields exact KV totals at that ctx. When None, falls back
    to `max_context` (so freshly-discovered models / skeletons get a sensible
    bake at their maximum supported window). For non-SWA models the value is
    ctx-independent and `reference_ctx` is ignored.

    `reference_batch_size`: the --batch-size to scale the compute-buffer estimate
    for. Larger batches use proportionally more scratch. Defaults to None (no
    overhead baked); pass the script's saved value to get an accurate total.

    `reference_k_bytes`, `reference_v_bytes`: KV element sizes for the saved
    cache types. Used to convert the compute-overhead bytes back into a multiplier
    addend so the standard formula's `× (k_bytes + v_bytes)` cancels out correctly
    at the saved cache types.

    Returns GGUFMetadata with pre-adjusted multiplier that works with standard formula:
        kv_cache = block_count * ctx_size * kv_cache_multiplier * (k_bytes + v_bytes) / 1024^3
    """
    try:
        metadata = _extract_gguf_fields(model_path)
        if metadata is None:
            return GGUFMetadata()

        # Detect architecture to use correct field name prefixes
        architecture = metadata.get("general.architecture", "llama")
        arch_prefix = f"{architecture}."

        logger.debug(f"   [GGUF] Architecture: {architecture}")

        # Extract required fields
        hidden_dim = (
            metadata.get(f"{arch_prefix}embedding_length") or
            metadata.get("llama.embedding_length")
        )

        block_count = (
            metadata.get(f"{arch_prefix}block_count") or
            metadata.get("llama.block_count")
        )

        # Extract max context length (supported by model during training)
        max_context = (
            metadata.get(f"{arch_prefix}context_length") or
            metadata.get("llama.context_length")
        )

        num_heads = (
            metadata.get(f"{arch_prefix}attention.head_count") or
            metadata.get("llama.attention.head_count")
        )

        num_heads_kv = (
            metadata.get(f"{arch_prefix}attention.head_count_kv") or
            metadata.get("llama.attention.head_count_kv")
        )

        # Check for direct KV embedding size (some models like Qwen3.6 have explicit values)
        n_embd_k_gqa = (
            metadata.get(f"{arch_prefix}embedding.length_kv") or
            metadata.get("llama.embedding.length_kv") or
            metadata.get("n_embd_k_gqa")
        )

        # Per-head KV dimension. When llama.cpp loads a model, it uses
        # `attention.key_length` for the cached KV head dim when present, falling
        # back to `hidden_dim / num_heads`. We mirror that, BUT only for
        # non-hybrid architectures — Qwen3 hybrid-SSM declares key_length=256
        # even though the actual cached dim equals hidden/heads. See doc 18.
        key_length = (
            metadata.get(f"{arch_prefix}attention.key_length")
            or metadata.get("llama.attention.key_length")
        )
        if isinstance(key_length, list):
            key_length = key_length[0] if key_length else None
        try:
            key_length = int(key_length) if key_length else None
        except (TypeError, ValueError):
            key_length = None

        # Detect hybrid attention+SSM models (e.g., Qwen3.6)
        # These have ssm_group_count indicating number of attention layers
        ssm_group_count = metadata.get(f"{arch_prefix}ssm.group_count")
        kv_active_layers = block_count

        if ssm_group_count and block_count and ssm_group_count < block_count:
            # Hybrid model: only some layers use attention+KV
            # ssm_group_count indicates the number of attention/KV layers
            # (remaining layers use SSM/Mamba, which don't use KV cache)
            kv_active_layers = ssm_group_count
            logger.debug(f"   [GGUF] Detected hybrid attention+SSM: block_count={block_count}, attention_layers={kv_active_layers}, ssm_group_count={ssm_group_count}")

        if block_count:
            logger.debug(f"   [GGUF] block_count={block_count}, kv_active_layers={kv_active_layers}")
        if max_context:
            logger.debug(f"   [GGUF] max_context={max_context}")

        # Compute KV cache multiplier based on attention architecture
        kv_cache_multiplier = None

        # If we have the actual KV embedding size, use it directly
        if n_embd_k_gqa:
            kv_cache_multiplier = n_embd_k_gqa
            logger.debug(f"   [GGUF] Using direct KV embedding: n_embd_k_gqa={n_embd_k_gqa}")
        elif hidden_dim and num_heads:
            # Handle per-layer arrays (e.g., Gemma-4, Deci models): use first element
            if isinstance(num_heads, list):
                num_heads = num_heads[0] if num_heads else None
            if isinstance(num_heads_kv, list):
                num_heads_kv = num_heads_kv[0] if num_heads_kv else None

            # Use key_length when present for non-hybrid architectures (matches
            # llama.cpp's KV cache sizing). Skip when:
            #  - Hybrid SSM (Qwen3.6 family): key_length disagrees with the
            #    actual cached dim there.
            #  - SWA (Gemma 4+): the model has separate key_length AND
            #    key_length_swa, and the SWA-adjustment downstream applies a
            #    uniform window correction — using either dim alone would be
            #    wrong. Stick with hidden/heads which is the architectural
            #    average and lets the SWA adjustment land near reality.
            is_hybrid_ssm = bool(ssm_group_count and block_count and ssm_group_count < block_count)
            _swa_marker = (
                metadata.get(f"{arch_prefix}attention.sliding_window")
                or metadata.get(f"{arch_prefix}attention.sliding_window_pattern")
                or metadata.get("llama.attention.sliding_window")
                or metadata.get("llama.attention.sliding_window_pattern")
            )
            has_swa = _swa_marker is not None
            head_dim = hidden_dim // num_heads
            if key_length and not is_hybrid_ssm and not has_swa:
                head_dim = key_length

            if num_heads and num_heads_kv and num_heads_kv < num_heads:
                # GQA or MQA detected
                kv_cache_multiplier = num_heads_kv * head_dim
                logger.debug(f"   [GGUF] Detected GQA: num_heads={num_heads}, num_heads_kv={num_heads_kv}, head_dim={head_dim} (key_length={key_length}, hybrid_ssm={is_hybrid_ssm}), multiplier={kv_cache_multiplier}")
            elif num_heads:
                # Standard multi-head attention
                kv_cache_multiplier = num_heads * head_dim
                logger.debug(f"   [GGUF] Detected standard attention: num_heads={num_heads}, head_dim={head_dim}, multiplier={kv_cache_multiplier}")

        # Pre-adjust multiplier for hybrid architectures
        if kv_cache_multiplier and block_count and kv_active_layers < block_count:
            # Hybrid model: adjust multiplier so formula uses total block_count
            # Formula: block_count * ctx * multiplier * bytes
            # Needs to account for: kv_active_layers * ctx * base_multiplier * bytes
            # So: multiplier = base_multiplier * (kv_active_layers / block_count)
            adjustment_ratio = kv_active_layers / block_count
            kv_cache_multiplier = int(kv_cache_multiplier * adjustment_ratio)
            logger.debug(f"   [GGUF] Adjusted multiplier for hybrid: {kv_cache_multiplier} (ratio={adjustment_ratio:.2f})")

        # Pre-adjust multiplier for sliding-window attention (e.g., Gemma 3+).
        # Only applied when SSM hybrid is *not* in play — the two architectures don't
        # combine in any model in the wild today. SWA bakes the per-layer cost so
        # the formula's `block_count * ctx * multiplier` matches the true
        # `(global_layers * ctx + swa_layers * min(ctx, window)) * base_mult`.
        if not (ssm_group_count and ssm_group_count < (block_count or 0)):
            swa_window_raw = (
                metadata.get(f"{arch_prefix}attention.sliding_window")
                or metadata.get("llama.attention.sliding_window")
            )
            swa_pattern_raw = (
                metadata.get(f"{arch_prefix}attention.sliding_window_pattern")
                or metadata.get("llama.attention.sliding_window_pattern")
            )

            # Normalize sliding_window to a scalar int.
            if isinstance(swa_window_raw, list):
                swa_window_raw = swa_window_raw[0] if swa_window_raw else None
            try:
                swa_window = int(swa_window_raw) if swa_window_raw else None
            except (TypeError, ValueError):
                swa_window = None

            # Normalize sliding_window_pattern. Two known forms:
            #   (a) integer stride (e.g., 6): every Nth layer is global
            #   (b) per-layer boolean list: True=SWA, False=global (Gemma 4)
            global_layers = None
            swa_layers = None
            if isinstance(swa_pattern_raw, list) and swa_pattern_raw:
                if all(isinstance(x, bool) for x in swa_pattern_raw):
                    # Boolean per-layer: True means SWA, False means global.
                    swa_layers = sum(1 for x in swa_pattern_raw if x)
                    global_layers = len(swa_pattern_raw) - swa_layers
                else:
                    # Numeric-but-list — collapse to first element and treat as stride.
                    try:
                        stride = int(swa_pattern_raw[0])
                        if block_count and stride > 1:
                            global_layers = max(1, block_count // stride)
                            swa_layers = block_count - global_layers
                    except (TypeError, ValueError):
                        pass
            elif swa_pattern_raw:
                try:
                    stride = int(swa_pattern_raw)
                    if block_count and stride > 1:
                        global_layers = max(1, block_count // stride)
                        swa_layers = block_count - global_layers
                except (TypeError, ValueError):
                    pass

            if (
                kv_cache_multiplier
                and block_count
                and swa_window
                and global_layers is not None
                and swa_layers is not None
                and swa_layers > 0
            ):
                ref = reference_ctx if reference_ctx else max_context
                if ref:
                    effective_sum = global_layers * ref + swa_layers * min(ref, swa_window)
                    base_mult = kv_cache_multiplier
                    kv_cache_multiplier = max(
                        1, int(round(base_mult * effective_sum / (block_count * ref)))
                    )
                    logger.debug(
                        f"   [GGUF] SWA adjustment: window={swa_window}, "
                        f"global={global_layers}, swa={swa_layers}, ref_ctx={ref}, "
                        f"base_mult={base_mult} → adjusted={kv_cache_multiplier}"
                    )

        # Compute-buffer overhead: fold an empirical scratch-space estimate into
        # the multiplier so the predicted total absorbs llama-server's per-batch
        # buffers. Linear-in-batch heuristic calibrated against four real models;
        # variance is ~1-2 GiB across the sample. The estimate is in raw bytes
        # and gets divided by (block_count × ctx × (k_bytes + v_bytes)) so it
        # disappears cleanly into the standard formula at the saved cache types.
        if (
            kv_cache_multiplier
            and block_count
            and reference_batch_size
            and reference_batch_size > 0
        ):
            ref = reference_ctx if reference_ctx else max_context
            total_bytes = reference_k_bytes + reference_v_bytes
            if ref and total_bytes > 0:
                # max(1.5 GiB, batch × 0.0017 GiB) — floor covers default batch overhead
                overhead_gib = max(1.5, reference_batch_size * 0.0017)
                overhead_bytes = overhead_gib * (1024 ** 3)
                addition = overhead_bytes / (block_count * ref * total_bytes)
                base = kv_cache_multiplier
                kv_cache_multiplier = max(1, int(round(base + addition)))
                logger.debug(
                    f"   [GGUF] Compute-buffer overhead: batch={reference_batch_size}, "
                    f"overhead={overhead_gib:.2f}GiB, ref_ctx={ref}, "
                    f"bytes={total_bytes}, base_mult={base} → adjusted={kv_cache_multiplier}"
                )

        if kv_cache_multiplier:
            logger.debug(f"   [GGUF] ✓ Extraction complete (block_count={block_count}, max_context={max_context}, kv_cache_multiplier={kv_cache_multiplier})")
        else:
            logger.debug(f"   [GGUF] ⚠ Could not compute kv_cache_multiplier (hidden_dim={hidden_dim}, num_heads={num_heads})")

        return GGUFMetadata(
            block_count=block_count,
            max_context=max_context,
            kv_cache_multiplier=kv_cache_multiplier,
        )
    except Exception as e:
        logger.warning(f"⚠ GGUF metadata extraction failed for {model_path}: {type(e).__name__}: {e}")
        return GGUFMetadata()
