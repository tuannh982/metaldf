import numpy as np
import pandas as pd
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")


class TestBinaryOps:
    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
    def test_binary_f32(self, op):
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        ma = metaldf_engine.MetalSeries.from_numpy(a)
        mb = metaldf_engine.MetalSeries.from_numpy(b)
        result = metaldf_engine.metal_binary_op(op, ma, mb)
        got = result.to_numpy()
        ops = {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}
        np.testing.assert_allclose(got, ops[op], rtol=1e-5)

    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
    def test_binary_i32(self, op):
        a = np.array([10, 20, 30, 40], dtype=np.int32)
        b = np.array([1, 2, 3, 4], dtype=np.int32)
        ma = metaldf_engine.MetalSeries.from_numpy_i32(a)
        mb = metaldf_engine.MetalSeries.from_numpy_i32(b)
        result = metaldf_engine.metal_binary_op(op, ma, mb)
        got = result.to_numpy()
        ops = {"add": a + b, "sub": a - b, "mul": a * b, "div": a // b}
        np.testing.assert_array_equal(got, ops[op])

    def test_binary_dtype_mismatch_raises(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([1, 2], dtype=np.int32)
        ma = metaldf_engine.MetalSeries.from_numpy(a)
        mb = metaldf_engine.MetalSeries.from_numpy_i32(b)
        with pytest.raises(TypeError):
            metaldf_engine.metal_binary_op("add", ma, mb)

    def test_binary_length_mismatch_raises(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0], dtype=np.float32)
        ma = metaldf_engine.MetalSeries.from_numpy(a)
        mb = metaldf_engine.MetalSeries.from_numpy(b)
        with pytest.raises(ValueError):
            metaldf_engine.metal_binary_op("add", ma, mb)


class TestUnaryOps:
    @pytest.mark.parametrize("op", ["abs", "neg", "sqrt"])
    def test_unary_f32(self, op):
        a = np.array([1.0, 4.0, 9.0, 16.0], dtype=np.float32)
        ma = metaldf_engine.MetalSeries.from_numpy(a)
        result = metaldf_engine.metal_unary_op(op, ma)
        got = result.to_numpy()
        ops = {"abs": np.abs(a), "neg": -a, "sqrt": np.sqrt(a)}
        np.testing.assert_allclose(got, ops[op], rtol=1e-5)

    @pytest.mark.parametrize("op,expected_fn", [
        ("exp", np.exp),
        ("log", np.log),
        ("ceil", np.ceil),
        ("floor", np.floor),
    ])
    def test_unary_f32_transcendental(self, op, expected_fn):
        a = np.array([1.0, 2.5, 3.7, 4.2], dtype=np.float32)
        ma = metaldf_engine.MetalSeries.from_numpy(a)
        result = metaldf_engine.metal_unary_op(op, ma)
        got = result.to_numpy()
        np.testing.assert_allclose(got, expected_fn(a), rtol=1e-4)

    @pytest.mark.parametrize("op", ["abs", "neg"])
    def test_unary_i32(self, op):
        a = np.array([-3, 5, -7, 9], dtype=np.int32)
        ma = metaldf_engine.MetalSeries.from_numpy_i32(a)
        result = metaldf_engine.metal_unary_op(op, ma)
        got = result.to_numpy()
        ops = {"abs": np.abs(a), "neg": -a}
        np.testing.assert_array_equal(got, ops[op])

    def test_unary_large_array_not_multiple_of_threadgroup(self):
        # Regression check: dispatch must not depend on len being a
        # multiple of the threadgroup size (kernels have no bounds guard).
        n = 1000  # not a multiple of 256
        a = np.arange(n, dtype=np.float32) - 500.0
        ma = metaldf_engine.MetalSeries.from_numpy(a)
        result = metaldf_engine.metal_unary_op("abs", ma)
        got = result.to_numpy()
        np.testing.assert_allclose(got, np.abs(a), rtol=1e-5)
