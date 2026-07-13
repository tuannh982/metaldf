import numpy as np
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")

# Opcodes matching the Metal kernel
LOAD_COL = list(range(8))  # 0-7
OP_ADD = 16
OP_SUB = 17
OP_MUL = 18
OP_DIV = 19
OP_ABS = 32
OP_NEG = 33
OP_SQRT = 34


def test_simple_add():
    """col0 + col1"""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    ma = metaldf_engine.MetalSeries.from_numpy(a)
    mb = metaldf_engine.MetalSeries.from_numpy(b)
    program = bytes([0, 1, OP_ADD])  # LOAD_COL0, LOAD_COL1, ADD
    result = metaldf_engine.eval_expression(program, [ma, mb], len(a))
    np.testing.assert_allclose(result.to_numpy(), a + b, rtol=1e-5)


def test_chained_expression():
    """(col0 + col1) * col2 - col3"""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    c = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    d = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    cols = [metaldf_engine.MetalSeries.from_numpy(x) for x in [a, b, c, d]]
    # (col0 + col1) * col2 - col3
    program = bytes([0, 1, OP_ADD, 2, OP_MUL, 3, OP_SUB])
    result = metaldf_engine.eval_expression(program, cols, len(a))
    expected = (a + b) * c - d
    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)


def test_unary_in_expression():
    """sqrt(abs(col0))"""
    a = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
    ma = metaldf_engine.MetalSeries.from_numpy(a)
    program = bytes([0, OP_ABS, OP_SQRT])
    result = metaldf_engine.eval_expression(program, [ma], len(a))
    expected = np.sqrt(np.abs(a))
    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)


def test_large_array():
    """Verify correctness at scale: (col0 - col1) * col2, 1M elements"""
    rng = np.random.default_rng(42)
    a = rng.standard_normal(1_000_000).astype(np.float32)
    b = rng.standard_normal(1_000_000).astype(np.float32)
    c = rng.standard_normal(1_000_000).astype(np.float32)
    cols = [metaldf_engine.MetalSeries.from_numpy(x) for x in [a, b, c]]
    program = bytes([0, 1, OP_SUB, 2, OP_MUL])
    result = metaldf_engine.eval_expression(program, cols, len(a))
    expected = (a - b) * c
    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-4)
