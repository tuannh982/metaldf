"""Tests for GPU shift, diff, and pct_change."""

import numpy as np
import pandas as pd
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")


class TestShiftDirect:
    def test_shift_forward_float32(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_shift(ms, 1).to_numpy()
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1:], arr[:-1], rtol=1e-5)

    def test_shift_backward_float32(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_shift(ms, -1).to_numpy()
        np.testing.assert_allclose(result[:-1], arr[1:], rtol=1e-5)
        assert np.isnan(result[-1])

    def test_shift_zero_identity(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_shift(ms, 0).to_numpy()
        np.testing.assert_allclose(result, arr, rtol=1e-5)

    def test_shift_larger_than_length(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_shift(ms, 10).to_numpy()
        assert all(np.isnan(result))

    def test_shift_int32(self):
        arr = np.array([10, 20, 30, 40], dtype=np.int32)
        ms = metaldf_engine.MetalSeries.from_numpy_i32(arr)
        result = metaldf_engine.metal_shift(ms, 1).to_numpy()
        assert result[0] == 0
        np.testing.assert_array_equal(result[1:], arr[:-1])

    def test_shift_int64(self):
        arr = np.array([10, 20, 30, 40], dtype=np.int64)
        ms = metaldf_engine.MetalSeries.from_numpy_i64(arr)
        result = metaldf_engine.metal_shift(ms, 2).to_numpy()
        np.testing.assert_array_equal(result[:2], [0, 0])
        np.testing.assert_array_equal(result[2:], arr[:2])


class TestProxyShiftDiffPctChange:
    @pytest.fixture(autouse=True)
    def _install(self):
        import metaldf
        metaldf.install()
        yield
        metaldf.uninstall()

    def test_shift(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = s.shift(1)
        expected = pd.Series([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        pd.testing.assert_series_equal(result.reset_index(drop=True), expected, rtol=1e-5)

    def test_diff(self):
        s = pd.Series([1.0, 3.0, 6.0, 10.0], dtype=np.float32)
        result = s.diff(1)
        expected = pd.Series([np.nan, 2.0, 3.0, 4.0], dtype=np.float32)
        np.testing.assert_allclose(result.values[1:], expected.values[1:], rtol=1e-5)

    def test_pct_change(self):
        s = pd.Series([100.0, 110.0, 121.0], dtype=np.float32)
        result = s.pct_change(1)
        expected = pd.Series([np.nan, 0.1, 0.1], dtype=np.float32)
        np.testing.assert_allclose(result.values[1:], expected.values[1:], rtol=1e-4)
