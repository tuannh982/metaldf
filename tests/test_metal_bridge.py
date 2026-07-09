import platform

import numpy as np
import pytest

from metaldf._engine._metal import is_metal_available
import metaldf_engine

# Skip all Metal tests if not on macOS
pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Metal tests only run on macOS",
)


def test_metal_available_on_macos():
    assert is_metal_available(), "Metal engine should be available after building"


def test_metal_series_from_numpy():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    series = metaldf_engine.MetalSeries.from_numpy(arr)
    assert series.len == 5
    assert series.dtype == "Float32"


def test_metal_series_to_numpy_roundtrip():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    series = metaldf_engine.MetalSeries.from_numpy(arr)
    out = series.to_numpy()
    np.testing.assert_array_equal(out, arr)


def test_metal_series_zero_copy():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ptr_before = arr.ctypes.data

    series = metaldf_engine.MetalSeries.from_numpy(arr)
    out = series.to_numpy()
    ptr_after = out.ctypes.data

    assert ptr_before == ptr_after, "MetalSeries should be zero-copy"
