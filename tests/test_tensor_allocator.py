# tests/test_tensor_allocator.py

from backend.tensor_allocator import compute_tensor_split, validate_batch_parameters

def test_heterogeneous_tensor_split():
    # Simulate: 2x 20GB cards, 4x 16GB cards (in MB)
    capacities = {
        0: 20480.0,
        1: 20480.0,
        2: 16384.0,
        3: 16384.0,
        4: 16384.0,
        5: 16384.0,
    }
    split_str, fractions = compute_tensor_split(capacities, headroom_mb=768.0)
    assert len(fractions) == 6
    # First two GPUs should have higher weights than the remaining four
    assert fractions[0] > fractions[2]
    assert fractions[1] > fractions[3]
    assert len(split_str.split(",")) == 6


def test_batch_advisor_validation():
    # ubatch > batch should fail validation
    res1 = validate_batch_parameters(ctx_size=8192, batch_size=512, ubatch_size=1024)
    assert not res1["valid"]
    assert len(res1["issues"]) > 0

    # High context (>64k) with high batch should issue warnings
    res2 = validate_batch_parameters(ctx_size=131072, batch_size=4096, ubatch_size=2048, flash_attn=False)
    assert res2["valid"]
    assert len(res2["warnings"]) >= 2
    assert res2["recommended"]["batch_size"] == 2048
    assert res2["recommended"]["ubatch_size"] == 1024