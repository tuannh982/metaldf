import numpy as np
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")

OP_ADD = 16
OP_SUB = 17
OP_MUL = 18


def test_fused_reduce_sum_simple():
    """sum(col0 + col1) via fused kernel."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    ma = metaldf_engine.MetalSeries.from_numpy(a)
    mb = metaldf_engine.MetalSeries.from_numpy(b)
    program = bytes([0, 1, OP_ADD])  # col0 + col1
    result = metaldf_engine.eval_expression_reduce("sum", program, [ma, mb], len(a))
    expected = float(np.sum(a + b))  # 36.0
    assert abs(result - expected) < 0.01, f"got {result}, expected {expected}"


def test_fused_reduce_sum_chain():
    """sum((col0 + col1) * col2 - col3) via fused kernel."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    c = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    d = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    cols = [metaldf_engine.MetalSeries.from_numpy(x) for x in [a, b, c, d]]
    program = bytes([0, 1, OP_ADD, 2, OP_MUL, 3, OP_SUB])
    result = metaldf_engine.eval_expression_reduce("sum", program, cols, len(a))
    expected = float(np.sum((a + b) * c - d))
    assert abs(result - expected) < 0.1, f"got {result}, expected {expected}"


def test_fused_reduce_min():
    """min(col0 - col1) via fused kernel."""
    a = np.array([10.0, 2.0, 8.0, 5.0], dtype=np.float32)
    b = np.array([3.0, 7.0, 1.0, 9.0], dtype=np.float32)
    ma = metaldf_engine.MetalSeries.from_numpy(a)
    mb = metaldf_engine.MetalSeries.from_numpy(b)
    program = bytes([0, 1, OP_SUB])
    result = metaldf_engine.eval_expression_reduce("min", program, [ma, mb], len(a))
    expected = float(np.min(a - b))  # -5.0
    assert abs(result - expected) < 0.01, f"got {result}, expected {expected}"


def test_fused_reduce_max():
    """max(col0 * col1) via fused kernel."""
    a = np.array([1.0, 3.0, 2.0, 4.0], dtype=np.float32)
    b = np.array([5.0, 2.0, 8.0, 1.0], dtype=np.float32)
    ma = metaldf_engine.MetalSeries.from_numpy(a)
    mb = metaldf_engine.MetalSeries.from_numpy(b)
    program = bytes([0, 1, OP_MUL])
    result = metaldf_engine.eval_expression_reduce("max", program, [ma, mb], len(a))
    expected = float(np.max(a * b))  # 16.0
    assert abs(result - expected) < 0.01, f"got {result}, expected {expected}"


def test_fused_reduce_large():
    """sum((col0 - col1) * col2) at 1M elements."""
    rng = np.random.default_rng(42)
    a = rng.standard_normal(1_000_000).astype(np.float32)
    b = rng.standard_normal(1_000_000).astype(np.float32)
    c = rng.standard_normal(1_000_000).astype(np.float32)
    cols = [metaldf_engine.MetalSeries.from_numpy(x) for x in [a, b, c]]
    program = bytes([0, 1, OP_SUB, 2, OP_MUL])
    result = metaldf_engine.eval_expression_reduce("sum", program, cols, len(a))
    expected = float(np.sum((a - b) * c))
    assert abs(result - expected) / (abs(expected) + 1e-6) < 0.01
