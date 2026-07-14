import numpy as np
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")


def test_sort_with_nulls_ascending():
    arr = np.array([3.0, float('nan'), 1.0, float('nan'), 2.0], dtype=np.float32)
    ms = metaldf_engine.MetalSeries.from_numpy_with_nulls(arr)
    result = metaldf_engine.metal_sort(ms)
    data = result.to_numpy()
    mask = result.null_mask
    # Valid values sorted first
    np.testing.assert_allclose(data[:3], [1.0, 2.0, 3.0])
    # Last 2 positions are null
    np.testing.assert_array_equal(mask[:3], [True, True, True])
    np.testing.assert_array_equal(mask[3:], [False, False])


def test_sort_no_nulls_unchanged():
    arr = np.array([3.0, 1.0, 2.0], dtype=np.float32)
    ms = metaldf_engine.MetalSeries.from_numpy(arr)
    result = metaldf_engine.metal_sort(ms)
    assert result.null_mask is None
    np.testing.assert_allclose(result.to_numpy(), [1.0, 2.0, 3.0])


def test_sort_no_nans_via_with_nulls_constructor():
    """A series built via from_numpy_with_nulls but containing no actual
    NaNs gets null_mask=None from the constructor — metal_sort should just
    take the null-free path and behave identically to the plain sort."""
    arr = np.array([5.0, 2.0, 4.0, 1.0, 3.0], dtype=np.float32)
    ms = metaldf_engine.MetalSeries.from_numpy_with_nulls(arr)
    assert ms.null_mask is None
    result = metaldf_engine.metal_sort(ms)
    assert result.null_mask is None
    np.testing.assert_allclose(result.to_numpy(), [1.0, 2.0, 3.0, 4.0, 5.0])


def test_sort_all_nulls():
    arr = np.array([float('nan'), float('nan'), float('nan')], dtype=np.float32)
    ms = metaldf_engine.MetalSeries.from_numpy_with_nulls(arr)
    result = metaldf_engine.metal_sort(ms)
    mask = result.null_mask
    assert mask is not None
    np.testing.assert_array_equal(mask, [False, False, False])


def test_sort_single_null():
    arr = np.array([3.0, 1.0, float('nan'), 2.0], dtype=np.float32)
    ms = metaldf_engine.MetalSeries.from_numpy_with_nulls(arr)
    result = metaldf_engine.metal_sort(ms)
    data = result.to_numpy()
    mask = result.null_mask
    np.testing.assert_allclose(data[:3], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(mask, [True, True, True, False])


def test_sort_with_nulls_large_array_radix_path():
    """Exercise the radix-sort path (N >= 100K) with nulls scattered
    throughout, verifying valid values end up sorted at the front and all
    null positions land contiguously at the tail."""
    n = 150_000
    rng = np.random.default_rng(3)
    arr = rng.standard_normal(n).astype(np.float32)
    null_positions = rng.choice(n, size=n // 10, replace=False)
    arr[null_positions] = float('nan')
    null_count = len(null_positions)
    valid_count = n - null_count

    ms = metaldf_engine.MetalSeries.from_numpy_with_nulls(arr)
    result = metaldf_engine.metal_sort(ms)
    data = result.to_numpy()
    mask = result.null_mask

    expected_valid_sorted = np.sort(arr[~np.isnan(arr)])
    np.testing.assert_allclose(data[:valid_count], expected_valid_sorted, rtol=1e-5)
    np.testing.assert_array_equal(mask[:valid_count], np.ones(valid_count, dtype=bool))
    np.testing.assert_array_equal(mask[valid_count:], np.zeros(null_count, dtype=bool))
    # Sanity: the front portion is non-decreasing.
    assert np.all(data[:valid_count][:-1] <= data[:valid_count][1:])


def test_sort_with_nulls_matches_numpy_bitonic_path():
    """Small-N (bitonic) path with a moderate fraction of nulls."""
    n = 5_000
    rng = np.random.default_rng(4)
    arr = rng.standard_normal(n).astype(np.float32)
    null_positions = rng.choice(n, size=n // 4, replace=False)
    arr[null_positions] = float('nan')
    valid_count = n - len(null_positions)

    ms = metaldf_engine.MetalSeries.from_numpy_with_nulls(arr)
    result = metaldf_engine.metal_sort(ms)
    data = result.to_numpy()
    mask = result.null_mask

    expected_valid_sorted = np.sort(arr[~np.isnan(arr)])
    np.testing.assert_allclose(data[:valid_count], expected_valid_sorted, rtol=1e-5)
    np.testing.assert_array_equal(mask[:valid_count], np.ones(valid_count, dtype=bool))
    np.testing.assert_array_equal(mask[valid_count:], np.zeros(len(null_positions), dtype=bool))
