import numpy as np
import pandas as pd
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")


@pytest.mark.parametrize("op,np_op", [
    ("sin", np.sin), ("cos", np.cos), ("tan", np.tan),
    ("asin", np.arcsin), ("acos", np.arccos), ("atan", np.arctan),
    ("sinh", np.sinh), ("cosh", np.cosh), ("tanh", np.tanh),
    ("log2", np.log2), ("log10", np.log10),
    ("round", np.round), ("trunc", np.trunc), ("cbrt", np.cbrt),
])
def test_unary_f32(op, np_op):
    if op in ("asin", "acos"):
        a = np.array([0.0, 0.1, 0.5, 0.9], dtype=np.float32)
    elif op in ("log2", "log10"):
        a = np.array([0.1, 1.0, 2.0, 10.0], dtype=np.float32)
    else:
        a = np.array([0.1, 0.5, 1.0, 2.0], dtype=np.float32)
    ma = metaldf_engine.MetalSeries.from_numpy(a)
    result = metaldf_engine.metal_unary_op(op, ma)
    expected = np_op(a)
    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-4)


def test_cbrt_negative_values():
    """cbrt of negative inputs should match numpy's real-valued cube root
    (MSL has no cbrt builtin -- see the copysign(pow(...)) formula in
    rust/metal/elementwise/unary.metal)."""
    a = np.array([-27.0, -8.0, -1.0, 0.0, 1.0, 8.0, 27.0], dtype=np.float32)
    ma = metaldf_engine.MetalSeries.from_numpy(a)
    result = metaldf_engine.metal_unary_op("cbrt", ma)
    expected = np.cbrt(a)
    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-4)


def test_round_half_to_even():
    """rint() (not round()) matches numpy's round-half-to-even behavior."""
    a = np.array([0.5, 1.5, 2.5, -0.5, -1.5], dtype=np.float32)
    ma = metaldf_engine.MetalSeries.from_numpy(a)
    result = metaldf_engine.metal_unary_op("round", ma)
    expected = np.round(a)
    np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-4)


def test_deferred_sin_fusion():
    from metaldf._wrappers import ProxySeries
    from metaldf._deferred import DeferredSeries

    s = ProxySeries(_pandas_obj=pd.Series([0.1, 0.5, 0.9], dtype=np.float32))
    result = (s + 0.1).sin()
    assert isinstance(result, DeferredSeries)
    materialized = result.to_pandas()
    expected = np.sin(np.array([0.2, 0.6, 1.0], dtype=np.float32))
    np.testing.assert_allclose(materialized.to_numpy(), expected, rtol=1e-4)


@pytest.mark.parametrize("op,np_op", [
    ("cos", np.cos), ("tan", np.tan), ("sinh", np.sinh), ("cosh", np.cosh),
    ("tanh", np.tanh), ("log2", np.log2), ("log10", np.log10),
    ("round", np.round), ("trunc", np.trunc), ("cbrt", np.cbrt),
])
def test_deferred_unary_fusion_all_ops(op, np_op):
    """Every new unary op should be chainable on a DeferredSeries and
    materialize to the same result as the equivalent numpy call."""
    from metaldf._wrappers import ProxySeries
    from metaldf._deferred import DeferredSeries

    base = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    s = ProxySeries(_pandas_obj=pd.Series(base))
    result = getattr(s + 0.0, op)()
    assert isinstance(result, DeferredSeries)
    materialized = result.to_pandas()
    expected = np_op(base)
    np.testing.assert_allclose(materialized.to_numpy(), expected, rtol=1e-4)


def test_codegen_matches_interpreter_for_new_ops():
    """The runtime-codegen path (eval_expression_codegen) and the bytecode
    interpreter (eval_expression) must agree for the new opcodes."""
    from metaldf._deferred import (
        OP_SIN, OP_LOG2, OP_ROUND, OP_CBRT,
    )

    a = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32)
    ma = metaldf_engine.MetalSeries.from_numpy(a)

    for opcode in (OP_SIN, OP_LOG2, OP_ROUND, OP_CBRT):
        program = bytes([0, opcode])
        interp = metaldf_engine.eval_expression(program, [ma], len(a))
        codegen = metaldf_engine.eval_expression_codegen(program, [ma], len(a))
        np.testing.assert_allclose(interp.to_numpy(), codegen.to_numpy(), rtol=1e-5)
