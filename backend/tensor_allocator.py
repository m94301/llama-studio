# backend/tensor_allocator.py
# 
# Dynamic Tensor Split and Batch Configuration Engine for heterogeneous GPU setups.

from typing import Dict, List, Optional, Tuple


def compute_tensor_split(
    gpu_capacities_mb: Dict[int, float],
    headroom_mb: float = 768.0,
    integer_ratios: bool = False,
) -> Tuple[str, List[float]]:
    """
    Computes proportional split values for llama-server --tensor-split.
    
    :param gpu_capacities_mb: Map of GPU ID -> free/total VRAM in MB.
    :param headroom_mb: Safety margin in MB subtracted from each GPU's capacity.
    :param integer_ratios: If True, scale to small integers; if False, normalize to floats.
    :return: (split_string, list_of_weights)
    """
    if not gpu_capacities_mb:
        return "", []

    # Map effective capacity taking headroom into account
    effective_caps = {}
    for gid, cap in gpu_capacities_mb.items():
        effective = max(256.0, cap - headroom_mb)
        effective_caps[gid] = effective

    total_effective = sum(effective_caps.values())
    if total_effective <= 0:
        return "", []

    # Format weights matching the order of sorted GPU IDs (CVD order)
    sorted_ids = sorted(gpu_capacities_mb.keys())
    fractions = [effective_caps[gid] / total_effective for gid in sorted_ids]

    if integer_ratios:
        # Scale smallest non-zero weight to roughly 10
        min_f = min(f for f in fractions if f > 0)
        multiplier = 10.0 / min_f if min_f > 0 else 1.0
        int_weights = [max(1, round(f * multiplier)) for f in fractions]
        split_str = ",".join(str(w) for w in int_weights)
        return split_str, fractions

    # Default is 2-decimal rounded floats (normalized)
    float_weights = [round(f, 3) for f in fractions]
    split_str = ",".join(f"{w:.3f}".rstrip("0").rstrip(".") for w in float_weights)
    return split_str, fractions


def validate_batch_parameters(
    ctx_size: int,
    batch_size: Optional[int] = None,
    ubatch_size: Optional[int] = None,
    flash_attn: bool = False,
) -> Dict[str, any]:
    # Evaluates context size and batch configurations to prevent CUDA scratch buffer allocation failures
    b = batch_size or 2048
    ub = ubatch_size or 512

    issues = []
    warnings = []
    recommendations = {}

    # ubatch must never exceed batch
    if ub > b:
        issues.append(f"--ubatch-size ({ub}) cannot be larger than --batch-size ({b}).")
        recommendations["ubatch_size"] = min(ub, b)

    # Large context scaling rules (>= 64k)
    if ctx_size >= 65536:
        if b > 2048:
            warnings.append(
                f"Batch size {b} is high for context {ctx_size}. "
                "Large batch sizes can trigger CUDA scratch buffer OOM during prompt eval."
            )
            recommendations["batch_size"] = 2048

        if ub > 1024:
            warnings.append(
                f"Micro-batch size {ub} may exceed intermediate VRAM buffers on high-context runs."
            )
            recommendations["ubatch_size"] = 1024

        if not flash_attn:
            warnings.append(
                f"Context length is {ctx_size} but Flash Attention (--flash-attn) is not enabled. "
                "Enabling Flash Attention significantly reduces intermediate KV allocation."
            )
            recommendations["flash_attn"] = True

    # Extreme context scaling (>= 131k / 256k)
    if ctx_size >= 131072:
        if b > 2048:
            recommendations["batch_size"] = 2048
        if ub > 1024:
            recommendations["ubatch_size"] = 1024

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "recommended": recommendations,
    }