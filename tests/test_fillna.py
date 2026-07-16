"""Tests for GPU fillna."""

import numpy as np
import pandas as pd
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")


class TestFillnaDirect:
    def test_fill_nan(self):
        arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_fillna(ms, 0.0).to_numpy()
        expected = np.array([1.0, 0.0, 3.0, 0.0, 5.0], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_no_nan_identity(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_fillna(ms, 99.0).to_numpy()
        np.testing.assert_allclose(result, arr, rtol=1e-5)

    def test_all_nan(self):
        arr = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_fillna(ms, -1.0).to_numpy()
        expected = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_fill_with_nonzero(self):
        arr = np.array([np.nan, 2.0, np.nan], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_fillna(ms, 42.5).to_numpy()
        expected = np.array([42.5, 2.0, 42.5], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestProxyFillna:
    @pytest.fixture(autouse=True)
    def _install(self):
        import metaldf
        metaldf.install()
        yield
        metaldf.uninstall()

    def test_fillna_scalar(self):
        s = pd.Series([1.0, np.nan, 3.0], dtype=np.float32)
        result = s.fillna(0.0)
        expected = pd.Series([1.0, 0.0, 3.0], dtype=np.float32)
        np.testing.assert_allclose(result.values, expected.values, rtol=1e-5)

    def test_fillna_falls_back_for_int(self):
        s = pd.Series([1, 2, 3], dtype=np.int32)
        result = s.fillna(0)
        np.testing.assert_array_equal(result.values, [1, 2, 3])
