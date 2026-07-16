"""Tests for GPU cumulative ops: cumsum, cummin, cummax."""

import numpy as np
import pandas as pd
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")

SCAN_TG_SIZE = 256


class TestCumsumFloat32:
    def test_small(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_cumsum(ms)
        np.testing.assert_allclose(result.to_numpy(), np.cumsum(arr), rtol=1e-5)

    def test_negative_values(self):
        arr = np.array([-1.5, 2.5, -3.5, 4.5], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_cumsum(ms)
        np.testing.assert_allclose(result.to_numpy(), np.cumsum(arr), rtol=1e-5)

    def test_single_element(self):
        arr = np.array([42.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_cumsum(ms)
        np.testing.assert_allclose(result.to_numpy(), np.cumsum(arr), rtol=1e-5)

    def test_cross_threadgroup(self):
        arr = np.arange(1, SCAN_TG_SIZE + 100, dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_cumsum(ms)
        np.testing.assert_allclose(result.to_numpy(), np.cumsum(arr), rtol=1e-4)

    def test_large(self):
        arr = np.ones(100_000, dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_cumsum(ms)
        np.testing.assert_allclose(result.to_numpy(), np.cumsum(arr), rtol=1e-3)


class TestCumsumInt32:
    def test_small(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        ms = metaldf_engine.MetalSeries.from_numpy_i32(arr)
        result = metaldf_engine.metal_cumsum(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.cumsum(arr))

    def test_negative(self):
        rng = np.random.default_rng(0)
        arr = rng.integers(-100, 100, size=37).astype(np.int32)
        ms = metaldf_engine.MetalSeries.from_numpy_i32(arr)
        result = metaldf_engine.metal_cumsum(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.cumsum(arr))

    def test_cross_threadgroup(self):
        arr = np.arange(1, SCAN_TG_SIZE + 2, dtype=np.int32)
        ms = metaldf_engine.MetalSeries.from_numpy_i32(arr)
        result = metaldf_engine.metal_cumsum(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.cumsum(arr))


class TestCumsumInt64:
    def test_small(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        ms = metaldf_engine.MetalSeries.from_numpy_i64(arr)
        result = metaldf_engine.metal_cumsum(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.cumsum(arr))

    def test_cross_threadgroup(self):
        arr = np.arange(1, SCAN_TG_SIZE + 50, dtype=np.int64)
        ms = metaldf_engine.MetalSeries.from_numpy_i64(arr)
        result = metaldf_engine.metal_cumsum(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.cumsum(arr))


class TestCumminFloat32:
    def test_small(self):
        arr = np.array([5.0, 3.0, 4.0, 1.0, 2.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_cummin(ms)
        np.testing.assert_allclose(result.to_numpy(), np.minimum.accumulate(arr), rtol=1e-5)

    def test_already_sorted_ascending(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_cummin(ms)
        np.testing.assert_allclose(result.to_numpy(), np.minimum.accumulate(arr), rtol=1e-5)

    def test_cross_threadgroup(self):
        rng = np.random.default_rng(42)
        arr = rng.random(SCAN_TG_SIZE + 100).astype(np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_cummin(ms)
        np.testing.assert_allclose(result.to_numpy(), np.minimum.accumulate(arr), rtol=1e-5)


class TestCumminInt32:
    def test_small(self):
        arr = np.array([5, 3, 4, 1, 2], dtype=np.int32)
        ms = metaldf_engine.MetalSeries.from_numpy_i32(arr)
        result = metaldf_engine.metal_cummin(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.minimum.accumulate(arr))


class TestCumminInt64:
    def test_small(self):
        arr = np.array([5, 3, 4, 1, 2], dtype=np.int64)
        ms = metaldf_engine.MetalSeries.from_numpy_i64(arr)
        result = metaldf_engine.metal_cummin(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.minimum.accumulate(arr))


class TestCummaxFloat32:
    def test_small(self):
        arr = np.array([1.0, 5.0, 3.0, 4.0, 2.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_cummax(ms)
        np.testing.assert_allclose(result.to_numpy(), np.maximum.accumulate(arr), rtol=1e-5)

    def test_already_sorted_descending(self):
        arr = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_cummax(ms)
        np.testing.assert_allclose(result.to_numpy(), np.maximum.accumulate(arr), rtol=1e-5)

    def test_cross_threadgroup(self):
        rng = np.random.default_rng(42)
        arr = rng.random(SCAN_TG_SIZE + 100).astype(np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_cummax(ms)
        np.testing.assert_allclose(result.to_numpy(), np.maximum.accumulate(arr), rtol=1e-5)


class TestCummaxInt32:
    def test_small(self):
        arr = np.array([1, 5, 3, 4, 2], dtype=np.int32)
        ms = metaldf_engine.MetalSeries.from_numpy_i32(arr)
        result = metaldf_engine.metal_cummax(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.maximum.accumulate(arr))


class TestCummaxInt64:
    def test_small(self):
        arr = np.array([1, 5, 3, 4, 2], dtype=np.int64)
        ms = metaldf_engine.MetalSeries.from_numpy_i64(arr)
        result = metaldf_engine.metal_cummax(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.maximum.accumulate(arr))
