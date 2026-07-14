import numpy as np
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")


def test_from_numpy_with_nulls_detects_nans():
    arr = np.array([1.0, float('nan'), 3.0, float('nan'), 5.0], dtype=np.float32)
    ms = metaldf_engine.MetalSeries.from_numpy_with_nulls(arr)
    mask = ms.null_mask
    assert mask is not None
    expected = [True, False, True, False, True]
    np.testing.assert_array_equal(mask, expected)


def test_from_numpy_with_nulls_no_nans():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ms = metaldf_engine.MetalSeries.from_numpy_with_nulls(arr)
    assert ms.null_mask is None


def test_from_numpy_with_nulls_all_nans():
    arr = np.array([float('nan'), float('nan')], dtype=np.float32)
    ms = metaldf_engine.MetalSeries.from_numpy_with_nulls(arr)
    mask = ms.null_mask
    assert mask is not None
    np.testing.assert_array_equal(mask, [False, False])


def test_from_numpy_with_nulls_data_cleaned():
    arr = np.array([1.0, float('nan'), 3.0], dtype=np.float32)
    ms = metaldf_engine.MetalSeries.from_numpy_with_nulls(arr)
    result = ms.to_numpy()
    assert not np.isnan(result[1])


def test_from_numpy_no_nulls_mask_is_none():
    """Series built via the plain from_numpy constructor never has nulls."""
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ms = metaldf_engine.MetalSeries.from_numpy(arr)
    assert ms.null_mask is None


def test_from_numpy_with_nulls_mask_length_matches_series():
    arr = np.array([1.0, float('nan'), 3.0, 4.0, float('nan'), 6.0, 7.0], dtype=np.float32)
    ms = metaldf_engine.MetalSeries.from_numpy_with_nulls(arr)
    mask = ms.null_mask
    assert len(mask) == ms.len
    assert mask.dtype == np.bool_
