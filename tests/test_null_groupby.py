import numpy as np
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")


def test_groupby_sum_null_keys_excluded():
    keys = np.array([1.0, float('nan'), 1.0, 2.0, float('nan')], dtype=np.float32)
    vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    mk = metaldf_engine.MetalSeries.from_numpy_with_nulls(keys)
    mv = metaldf_engine.MetalSeries.from_numpy(vals)
    unique_keys, agg = metaldf_engine.metal_groupby_sum(mk, mv)
    uk = unique_keys.to_numpy()
    av = agg.to_numpy()
    # NaN keys excluded; key 1.0 -> 10+30=40, key 2.0 -> 40
    assert len(uk) == 2
    av_dict = dict(zip(uk, av))
    np.testing.assert_allclose(av_dict[1.0], 40.0)
    np.testing.assert_allclose(av_dict[2.0], 40.0)


def test_groupby_sum_null_values_skipped():
    keys = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    vals = np.array([10.0, float('nan'), 30.0, 40.0], dtype=np.float32)
    mk = metaldf_engine.MetalSeries.from_numpy(keys)
    mv = metaldf_engine.MetalSeries.from_numpy_with_nulls(vals)
    unique_keys, agg = metaldf_engine.metal_groupby_sum(mk, mv)
    av_dict = dict(zip(unique_keys.to_numpy(), agg.to_numpy()))
    np.testing.assert_allclose(av_dict[1.0], 10.0)
    np.testing.assert_allclose(av_dict[2.0], 70.0)


def test_groupby_sum_both_keys_and_values_null():
    keys = np.array([1.0, float('nan'), 1.0, 2.0, 2.0], dtype=np.float32)
    vals = np.array([10.0, 999.0, float('nan'), 40.0, 50.0], dtype=np.float32)
    mk = metaldf_engine.MetalSeries.from_numpy_with_nulls(keys)
    mv = metaldf_engine.MetalSeries.from_numpy_with_nulls(vals)
    unique_keys, agg = metaldf_engine.metal_groupby_sum(mk, mv)
    uk = unique_keys.to_numpy()
    av_dict = dict(zip(uk, agg.to_numpy()))
    # Row 1 (nan key) dropped entirely; row 2 (key=1.0, val=nan) dropped too.
    assert len(uk) == 2
    np.testing.assert_allclose(av_dict[1.0], 10.0)
    np.testing.assert_allclose(av_dict[2.0], 90.0)


def test_groupby_mean_null_values():
    keys = np.array([1.0, 1.0, 1.0, 2.0], dtype=np.float32)
    vals = np.array([10.0, float('nan'), 30.0, 40.0], dtype=np.float32)
    mk = metaldf_engine.MetalSeries.from_numpy(keys)
    mv = metaldf_engine.MetalSeries.from_numpy_with_nulls(vals)
    unique_keys, agg = metaldf_engine.metal_groupby_mean(mk, mv)
    av_dict = dict(zip(unique_keys.to_numpy(), agg.to_numpy()))
    # mean(1.0) = (10+30)/2 = 20 (non-null sum / non-null count)
    np.testing.assert_allclose(av_dict[1.0], 20.0)
    np.testing.assert_allclose(av_dict[2.0], 40.0)


def test_groupby_min_null_values_skipped():
    keys = np.array([1.0, 1.0, 1.0, 2.0], dtype=np.float32)
    vals = np.array([5.0, float('nan'), 2.0, 40.0], dtype=np.float32)
    mk = metaldf_engine.MetalSeries.from_numpy(keys)
    mv = metaldf_engine.MetalSeries.from_numpy_with_nulls(vals)
    unique_keys, agg = metaldf_engine.metal_groupby_min(mk, mv)
    av_dict = dict(zip(unique_keys.to_numpy(), agg.to_numpy()))
    # min(1.0) should ignore the nan and be 2.0, not nan-propagated.
    np.testing.assert_allclose(av_dict[1.0], 2.0)
    np.testing.assert_allclose(av_dict[2.0], 40.0)


def test_groupby_max_null_values_skipped():
    keys = np.array([1.0, 1.0, 1.0, 2.0], dtype=np.float32)
    vals = np.array([5.0, float('nan'), 12.0, 40.0], dtype=np.float32)
    mk = metaldf_engine.MetalSeries.from_numpy(keys)
    mv = metaldf_engine.MetalSeries.from_numpy_with_nulls(vals)
    unique_keys, agg = metaldf_engine.metal_groupby_max(mk, mv)
    av_dict = dict(zip(unique_keys.to_numpy(), agg.to_numpy()))
    np.testing.assert_allclose(av_dict[1.0], 12.0)
    np.testing.assert_allclose(av_dict[2.0], 40.0)


def test_groupby_count_null_values_excluded():
    keys = np.array([1.0, 1.0, 1.0, 2.0], dtype=np.float32)
    vals = np.array([5.0, float('nan'), 12.0, 40.0], dtype=np.float32)
    mk = metaldf_engine.MetalSeries.from_numpy(keys)
    mv = metaldf_engine.MetalSeries.from_numpy_with_nulls(vals)
    unique_keys, agg = metaldf_engine.metal_groupby_count(mk, mv)
    av_dict = dict(zip(unique_keys.to_numpy(), agg.to_numpy()))
    # key 1.0 has 3 rows but one null value -> count 2 (pandas .count() semantics)
    assert av_dict[1.0] == 2
    assert av_dict[2.0] == 1


def test_groupby_count_null_keys_excluded():
    keys = np.array([1.0, float('nan'), 1.0, 2.0, float('nan')], dtype=np.float32)
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    mk = metaldf_engine.MetalSeries.from_numpy_with_nulls(keys)
    mv = metaldf_engine.MetalSeries.from_numpy(vals)
    unique_keys, agg = metaldf_engine.metal_groupby_count(mk, mv)
    uk = unique_keys.to_numpy()
    av_dict = dict(zip(uk, agg.to_numpy()))
    assert len(uk) == 2
    assert av_dict[1.0] == 2
    assert av_dict[2.0] == 1


def test_groupby_no_nulls_fast_path_unaffected():
    """A series built via from_numpy_with_nulls but with no actual NaNs gets
    null_mask=None — groupby should behave identically to the plain (no
    pre-filtering) path."""
    keys = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    vals = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    mk = metaldf_engine.MetalSeries.from_numpy_with_nulls(keys)
    mv = metaldf_engine.MetalSeries.from_numpy_with_nulls(vals)
    assert mk.null_mask is None
    assert mv.null_mask is None
    unique_keys, agg = metaldf_engine.metal_groupby_sum(mk, mv)
    av_dict = dict(zip(unique_keys.to_numpy(), agg.to_numpy()))
    np.testing.assert_allclose(av_dict[1.0], 30.0)
    np.testing.assert_allclose(av_dict[2.0], 70.0)


def test_groupby_all_keys_null_returns_empty():
    keys = np.array([float('nan'), float('nan'), float('nan')], dtype=np.float32)
    vals = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mk = metaldf_engine.MetalSeries.from_numpy_with_nulls(keys)
    mv = metaldf_engine.MetalSeries.from_numpy(vals)
    unique_keys, agg = metaldf_engine.metal_groupby_sum(mk, mv)
    assert unique_keys.len == 0
    assert agg.len == 0


def test_groupby_sum_null_keys_large_sort_path():
    """Exercise the sort-based groupby path (len > 500_000) with null keys
    scattered throughout, verifying null-key rows are excluded from the
    result while non-null groups still aggregate correctly."""
    n = 600_000
    rng = np.random.default_rng(7)
    keys = rng.integers(0, 100, size=n).astype(np.float32)
    vals = rng.standard_normal(n).astype(np.float32)

    null_positions = rng.choice(n, size=n // 20, replace=False)
    keys_with_nulls = keys.copy()
    keys_with_nulls[null_positions] = float('nan')

    mk = metaldf_engine.MetalSeries.from_numpy_with_nulls(keys_with_nulls)
    mv = metaldf_engine.MetalSeries.from_numpy(vals)
    unique_keys, agg = metaldf_engine.metal_groupby_sum(mk, mv)

    uk = unique_keys.to_numpy()
    av = agg.to_numpy()
    assert len(uk) == 100  # keys 0..99, none of them fully excluded

    valid_mask = ~np.isnan(keys_with_nulls)
    result = dict(zip(uk, av))
    for k in range(100):
        expected = vals[valid_mask & (keys == k)].sum()
        np.testing.assert_allclose(result[float(k)], expected, rtol=1e-3, atol=1e-2)
