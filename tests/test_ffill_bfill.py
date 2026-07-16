"""Tests for GPU ffill and bfill."""

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


class TestFfillDirect:
    def test_gap_in_middle(self):
        arr = np.array([1.0, np.nan, np.nan, 4.0, np.nan], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_ffill(ms).to_numpy()
        expected = np.array([1.0, 1.0, 1.0, 4.0, 4.0], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_nan_at_start(self):
        arr = np.array([np.nan, np.nan, 3.0, 4.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_ffill(ms).to_numpy()
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        np.testing.assert_allclose(result[2:], [3.0, 4.0], rtol=1e-5)

    def test_no_nan_identity(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_ffill(ms).to_numpy()
        np.testing.assert_allclose(result, arr, rtol=1e-5)

    def test_all_nan(self):
        arr = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_ffill(ms).to_numpy()
        assert all(np.isnan(result))

    def test_cross_threadgroup(self):
        arr = np.full(SCAN_TG_SIZE + 100, np.nan, dtype=np.float32)
        arr[0] = 42.0
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_ffill(ms).to_numpy()
        expected = np.full(SCAN_TG_SIZE + 100, 42.0, dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_large_multi_group(self):
        n = 10_000
        rng = np.random.default_rng(42)
        arr = rng.random(n).astype(np.float32)
        mask = rng.random(n) < 0.3
        arr[mask] = np.nan
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_ffill(ms).to_numpy()
        expected = pd.Series(arr).ffill().values
        # Compare only positions where expected is not NaN
        valid = ~np.isnan(expected)
        np.testing.assert_allclose(result[valid], expected[valid], rtol=1e-5)
        # Leading NaNs should remain NaN
        nan_expected = np.isnan(expected)
        assert np.all(np.isnan(result[nan_expected]))


class TestBfillDirect:
    def test_gap_in_middle(self):
        arr = np.array([1.0, np.nan, np.nan, 4.0, 5.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_bfill(ms).to_numpy()
        expected = np.array([1.0, 4.0, 4.0, 4.0, 5.0], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_nan_at_end(self):
        arr = np.array([1.0, 2.0, np.nan, np.nan], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_bfill(ms).to_numpy()
        np.testing.assert_allclose(result[:2], [1.0, 2.0], rtol=1e-5)
        assert np.isnan(result[2])
        assert np.isnan(result[3])

    def test_cross_threadgroup(self):
        arr = np.full(SCAN_TG_SIZE + 100, np.nan, dtype=np.float32)
        arr[-1] = 42.0
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_bfill(ms).to_numpy()
        expected = np.full(SCAN_TG_SIZE + 100, 42.0, dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestProxyFfillBfill:
    @pytest.fixture(autouse=True)
    def _install(self):
        import metaldf
        metaldf.install()
        yield
        metaldf.uninstall()

    def test_ffill(self):
        s = pd.Series([1.0, np.nan, np.nan, 4.0], dtype=np.float32)
        result = s.ffill()
        expected = pd.Series([1.0, 1.0, 1.0, 4.0], dtype=np.float32)
        np.testing.assert_allclose(result.values, expected.values, rtol=1e-5)

    def test_bfill(self):
        s = pd.Series([np.nan, np.nan, 3.0, 4.0], dtype=np.float32)
        result = s.bfill()
        expected = pd.Series([3.0, 3.0, 3.0, 4.0], dtype=np.float32)
        np.testing.assert_allclose(result.values, expected.values, rtol=1e-5)
