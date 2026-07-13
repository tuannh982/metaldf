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


def test_codegen_simple_add():
    """Codegen path should produce same result as interpreter."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    ma = metaldf_engine.MetalSeries.from_numpy(a)
    mb = metaldf_engine.MetalSeries.from_numpy(b)
    program = bytes([0, 1, OP_ADD])

    interp = metaldf_engine.eval_expression(program, [ma, mb], len(a))
    codegen = metaldf_engine.eval_expression_codegen(program, [ma, mb], len(a))
    np.testing.assert_array_equal(interp.to_numpy(), codegen.to_numpy())


def test_codegen_chain():
    """(col0 + col1) * col2 - col3 via codegen."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    c = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    d = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    cols = [metaldf_engine.MetalSeries.from_numpy(x) for x in [a, b, c, d]]
    program = bytes([0, 1, OP_ADD, 2, OP_MUL, 3, OP_SUB])
    result = metaldf_engine.eval_expression_codegen(program, cols, len(a))
    expected = (a + b) * c - d
    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-5)


def test_codegen_cache_hit():
    """Second call with same program should use cached pipeline."""
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([3.0, 4.0], dtype=np.float32)
    ma = metaldf_engine.MetalSeries.from_numpy(a)
    mb = metaldf_engine.MetalSeries.from_numpy(b)
    program = bytes([0, 1, OP_ADD])

    r1 = metaldf_engine.eval_expression_codegen(program, [ma, mb], 2)
    r2 = metaldf_engine.eval_expression_codegen(program, [ma, mb], 2)
    np.testing.assert_array_equal(r1.to_numpy(), r2.to_numpy())
