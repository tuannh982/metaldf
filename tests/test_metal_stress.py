import platform

import numpy as np
import pytest

import metaldf_engine

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Metal stress tests only run on macOS",
)


def test_buffer_allocation_stress():
    """Allocate and deallocate many buffers to check for leaks."""
    for _ in range(1000):
        arr = np.random.randn(1000).astype(np.float32)
        series = metaldf_engine.MetalSeries.from_numpy(arr)
        _ = series.to_numpy()
        del series
        del arr


def test_large_buffer():
    """Allocate a large buffer (10M elements)."""
    arr = np.random.randn(10_000_000).astype(np.float32)
    series = metaldf_engine.MetalSeries.from_numpy(arr)
    out = series.to_numpy()
    assert len(out) == 10_000_000
    np.testing.assert_array_equal(out, arr)
