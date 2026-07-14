"""Tests for the Bool dtype (uint8 storage) and logical AND/OR/NOT ops.

Task 2.3: Bool storage is 1 byte per element (uint8_t), NOT packed bits --
that's what distinguishes it from the null mask, which IS packed bits.
"""

import numpy as np
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")


class TestBoolRoundtrip:
    def test_from_numpy_bool_roundtrip(self):
        a = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        ms = metaldf_engine.MetalSeries.from_numpy_bool(a)
        assert ms.len == 5
        assert ms.dtype == "Bool"
        got = ms.to_numpy()
        np.testing.assert_array_equal(got, a)
        assert got.dtype == np.uint8

    def test_from_numpy_bool_empty(self):
        a = np.array([], dtype=np.uint8)
        ms = metaldf_engine.MetalSeries.from_numpy_bool(a)
        assert ms.len == 0
        assert ms.dtype == "Bool"
        got = ms.to_numpy()
        assert len(got) == 0

    def test_from_numpy_bool_all_true(self):
        a = np.ones(10, dtype=np.uint8)
        ms = metaldf_engine.MetalSeries.from_numpy_bool(a)
        np.testing.assert_array_equal(ms.to_numpy(), a)

    def test_from_numpy_bool_all_false(self):
        a = np.zeros(10, dtype=np.uint8)
        ms = metaldf_engine.MetalSeries.from_numpy_bool(a)
        np.testing.assert_array_equal(ms.to_numpy(), a)


class TestLogicalAnd:
    def test_and_basic(self):
        a = np.array([1, 1, 0, 0], dtype=np.uint8)
        b = np.array([1, 0, 1, 0], dtype=np.uint8)
        ma = metaldf_engine.MetalSeries.from_numpy_bool(a)
        mb = metaldf_engine.MetalSeries.from_numpy_bool(b)
        result = metaldf_engine.metal_logical_and(ma, mb)
        assert result.dtype == "Bool"
        np.testing.assert_array_equal(result.to_numpy(), [1, 0, 0, 0])

    def test_and_length_mismatch_raises(self):
        a = np.array([1, 0], dtype=np.uint8)
        b = np.array([1, 0, 1], dtype=np.uint8)
        ma = metaldf_engine.MetalSeries.from_numpy_bool(a)
        mb = metaldf_engine.MetalSeries.from_numpy_bool(b)
        with pytest.raises(ValueError):
            metaldf_engine.metal_logical_and(ma, mb)

    def test_and_rejects_non_bool_lhs(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([1, 0], dtype=np.uint8)
        ma = metaldf_engine.MetalSeries.from_numpy(a)
        mb = metaldf_engine.MetalSeries.from_numpy_bool(b)
        with pytest.raises(TypeError):
            metaldf_engine.metal_logical_and(ma, mb)

    def test_and_rejects_non_bool_rhs(self):
        a = np.array([1, 0], dtype=np.uint8)
        b = np.array([1.0, 0.0], dtype=np.float32)
        ma = metaldf_engine.MetalSeries.from_numpy_bool(a)
        mb = metaldf_engine.MetalSeries.from_numpy(b)
        with pytest.raises(TypeError):
            metaldf_engine.metal_logical_and(ma, mb)


class TestLogicalOr:
    def test_or_basic(self):
        a = np.array([1, 1, 0, 0], dtype=np.uint8)
        b = np.array([1, 0, 1, 0], dtype=np.uint8)
        ma = metaldf_engine.MetalSeries.from_numpy_bool(a)
        mb = metaldf_engine.MetalSeries.from_numpy_bool(b)
        result = metaldf_engine.metal_logical_or(ma, mb)
        assert result.dtype == "Bool"
        np.testing.assert_array_equal(result.to_numpy(), [1, 1, 1, 0])

    def test_or_rejects_non_bool(self):
        a = np.array([1, 2], dtype=np.int32)
        b = np.array([1, 0], dtype=np.uint8)
        ma = metaldf_engine.MetalSeries.from_numpy_i32(a)
        mb = metaldf_engine.MetalSeries.from_numpy_bool(b)
        with pytest.raises(TypeError):
            metaldf_engine.metal_logical_or(ma, mb)


class TestLogicalNot:
    def test_not_basic(self):
        a = np.array([1, 0, 1, 0], dtype=np.uint8)
        ma = metaldf_engine.MetalSeries.from_numpy_bool(a)
        result = metaldf_engine.metal_logical_not(ma)
        assert result.dtype == "Bool"
        np.testing.assert_array_equal(result.to_numpy(), [0, 1, 0, 1])

    def test_not_rejects_non_bool(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        ma = metaldf_engine.MetalSeries.from_numpy(a)
        with pytest.raises(TypeError):
            metaldf_engine.metal_logical_not(ma)

    def test_not_double_negation(self):
        a = np.array([1, 0, 1], dtype=np.uint8)
        ma = metaldf_engine.MetalSeries.from_numpy_bool(a)
        once = metaldf_engine.metal_logical_not(ma)
        twice = metaldf_engine.metal_logical_not(once)
        np.testing.assert_array_equal(twice.to_numpy(), a)


class TestLogicalLargeArray:
    def test_and_large_array_not_multiple_of_threadgroup(self):
        # Regression check mirroring test_elementwise.py: dispatch must not
        # depend on len being a multiple of the threadgroup size (kernels
        # have no bounds guard).
        n = 1000  # not a multiple of 256
        rng = np.random.default_rng(0)
        a = (rng.integers(0, 2, size=n)).astype(np.uint8)
        b = (rng.integers(0, 2, size=n)).astype(np.uint8)
        ma = metaldf_engine.MetalSeries.from_numpy_bool(a)
        mb = metaldf_engine.MetalSeries.from_numpy_bool(b)
        result = metaldf_engine.metal_logical_and(ma, mb)
        np.testing.assert_array_equal(result.to_numpy(), (a & b))
