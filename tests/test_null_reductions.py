import math

import numpy as np
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")


def test_sum_with_nulls():
    arr = np.array([1.0, float('nan'), 3.0, float('nan'), 5.0], dtype=np.float32)
    ms = metaldf_engine.MetalSeries.from_numpy_with_nulls(arr)
    result = metaldf_engine.metal_sum(ms)
    np.testing.assert_allclose(result, 9.0, rtol=1e-5)


def test_min_with_nulls():
    arr = np.array([float('nan'), 3.0, 1.0, float('nan')], dtype=np.float32)
    ms = metaldf_engine.MetalSeries.from_numpy_with_nulls(arr)
    result = metaldf_engine.metal_min(ms)
    np.testing.assert_allclose(result, 1.0)


def test_max_with_nulls():
    arr = np.array([float('nan'), 3.0, 1.0, float('nan')], dtype=np.float32)
    ms = metaldf_engine.MetalSeries.from_numpy_with_nulls(arr)
    result = metaldf_engine.metal_max(ms)
    np.testing.assert_allclose(result, 3.0)


def test_mean_with_nulls():
    arr = np.array([2.0, float('nan'), 4.0], dtype=np.float32)
    ms = metaldf_engine.MetalSeries.from_numpy_with_nulls(arr)
    result = metaldf_engine.metal_mean(ms)
    np.testing.assert_allclose(result, 3.0, rtol=1e-5)


def test_all_nulls_returns_nan():
    arr = np.array([float('nan'), float('nan')], dtype=np.float32)
    ms = metaldf_engine.MetalSeries.from_numpy_with_nulls(arr)
    result = metaldf_engine.metal_sum(ms)
    assert math.isnan(result)


def test_all_nulls_min_max_mean_return_nan():
    arr = np.array([float('nan'), float('nan'), float('nan')], dtype=np.float32)
    ms = metaldf_engine.MetalSeries.from_numpy_with_nulls(arr)
    assert math.isnan(metaldf_engine.metal_min(ms))
    assert math.isnan(metaldf_engine.metal_max(ms))
    assert math.isnan(metaldf_engine.metal_mean(ms))


def test_no_nulls_matches_unmasked_path():
    """A series with no NaNs should get null_mask=None and behave exactly
    like the pre-existing unmasked reduction path."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    ms = metaldf_engine.MetalSeries.from_numpy_with_nulls(arr)
    assert ms.null_mask is None
    assert metaldf_engine.metal_sum(ms) == 15.0
    assert metaldf_engine.metal_min(ms) == 1.0
    assert metaldf_engine.metal_max(ms) == 5.0
    np.testing.assert_allclose(metaldf_engine.metal_mean(ms), 3.0)


def test_sum_with_nulls_large_array():
    """Exercise the multi-pass reduction path (more than one threadgroup
    worth of elements) with nulls scattered throughout."""
    n = 100_000
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(n).astype(np.float32)
    null_positions = rng.choice(n, size=n // 10, replace=False)
    arr[null_positions] = float('nan')

    ms = metaldf_engine.MetalSeries.from_numpy_with_nulls(arr)
    result = metaldf_engine.metal_sum(ms)

    expected = np.nansum(arr)
    np.testing.assert_allclose(result, expected, rtol=1e-3)


def test_mean_with_nulls_large_array():
    n = 50_000
    rng = np.random.default_rng(1)
    arr = rng.standard_normal(n).astype(np.float32)
    null_positions = rng.choice(n, size=n // 5, replace=False)
    arr[null_positions] = float('nan')

    ms = metaldf_engine.MetalSeries.from_numpy_with_nulls(arr)
    result = metaldf_engine.metal_mean(ms)

    expected = np.nanmean(arr)
    np.testing.assert_allclose(result, expected, rtol=1e-3)


def test_min_max_with_nulls_large_array():
    """Exercise the multi-pass path for min/max specifically, since it
    shares dispatch_reduction's masked-first-pass logic with sum/mean but
    instantiates a different Op (MinOp/MaxOp identity) template."""
    n = 100_000
    rng = np.random.default_rng(2)
    arr = rng.standard_normal(n).astype(np.float32)
    null_positions = rng.choice(n, size=n // 10, replace=False)
    arr[null_positions] = float('nan')

    ms = metaldf_engine.MetalSeries.from_numpy_with_nulls(arr)

    np.testing.assert_allclose(metaldf_engine.metal_min(ms), np.nanmin(arr), rtol=1e-5)
    np.testing.assert_allclose(metaldf_engine.metal_max(ms), np.nanmax(arr), rtol=1e-5)
