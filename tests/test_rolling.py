"""Tests for the GPU rolling-window kernels (`metal_rolling_sum`/`_min`/
`_max`/`_count`/`_mean`).

Task 7.1 (Phase 7): naive parallel rolling window ops -- each GPU thread
computes exactly one output element by iterating over its own window
(`data[max(0, idx - window + 1) ..= idx]`). The prefix-sum-based strategy for
large windows (`window > 1024`) is deferred to a follow-up task; this kernel
is dispatched unconditionally regardless of `window` size.

f32 only for now (i32 rolling variants are deferred, same as
`rust/src/kernels/rolling.rs`).

Windows aren't specially masked for `min_periods`: every kernel operates
over however many elements are actually available near the start of the
series (`min(idx + 1, window)`), which is exactly pandas'
`rolling(window, min_periods=1)` behavior -- so results are compared
directly against `min_periods=1` throughout, with no NaN-masking needed on
either side.
"""

import numpy as np
import pandas as pd
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")

WINDOW_SIZES = [2, 10, 100]


def _series(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-100, 100, size=n).astype(np.float32)


class TestRollingSum:
    @pytest.mark.parametrize("window", WINDOW_SIZES)
    def test_matches_pandas(self, window):
        arr = _series(500, seed=window)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_rolling_sum(ms, window)
        expected = pd.Series(arr).rolling(window, min_periods=1).sum().to_numpy()
        assert result.dtype == "Float32"
        assert result.len == len(arr)
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-4, atol=1e-3)

    def test_window_one_is_identity(self):
        arr = _series(50, seed=1)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_rolling_sum(ms, 1)
        np.testing.assert_allclose(result.to_numpy(), arr, rtol=1e-5)

    def test_window_larger_than_series(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_rolling_sum(ms, 10)
        expected = pd.Series(arr).rolling(10, min_periods=1).sum().to_numpy()
        np.testing.assert_allclose(result.to_numpy(), expected)

    def test_early_positions_are_partial_sums(self):
        # Window not yet filled near the start -> partial sum of whatever
        # elements are actually available (pandas' min_periods=1 default).
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_rolling_sum(ms, 3).to_numpy()
        # idx=0: [1] -> 1; idx=1: [1,2] -> 3; idx=2: [1,2,3] -> 6;
        # idx=3: [2,3,4] -> 9; idx=4: [3,4,5] -> 12
        np.testing.assert_allclose(result, [1.0, 3.0, 6.0, 9.0, 12.0])


class TestRollingMin:
    @pytest.mark.parametrize("window", WINDOW_SIZES)
    def test_matches_pandas(self, window):
        arr = _series(500, seed=window + 100)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_rolling_min(ms, window)
        expected = pd.Series(arr).rolling(window, min_periods=1).min().to_numpy()
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-4, atol=1e-3)

    def test_basic(self):
        arr = np.array([5.0, 3.0, 8.0, 1.0, 9.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_rolling_min(ms, 2).to_numpy()
        expected = pd.Series(arr).rolling(2, min_periods=1).min().to_numpy()
        np.testing.assert_allclose(result, expected)


class TestRollingMax:
    @pytest.mark.parametrize("window", WINDOW_SIZES)
    def test_matches_pandas(self, window):
        arr = _series(500, seed=window + 200)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_rolling_max(ms, window)
        expected = pd.Series(arr).rolling(window, min_periods=1).max().to_numpy()
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-4, atol=1e-3)

    def test_basic(self):
        arr = np.array([5.0, 3.0, 8.0, 1.0, 9.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_rolling_max(ms, 2).to_numpy()
        expected = pd.Series(arr).rolling(2, min_periods=1).max().to_numpy()
        np.testing.assert_allclose(result, expected)


class TestRollingCount:
    @pytest.mark.parametrize("window", WINDOW_SIZES)
    def test_matches_pandas_no_nulls(self, window):
        # With no NaNs in the input, pandas' rolling().count() (non-NaN
        # count) matches our kernel's plain in-window-size count directly.
        arr = _series(500, seed=window + 300)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_rolling_count(ms, window)
        expected = pd.Series(arr).rolling(window, min_periods=1).count().to_numpy()
        np.testing.assert_allclose(result.to_numpy(), expected)

    def test_early_positions_ramp_up(self):
        arr = np.arange(5, dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_rolling_count(ms, 3).to_numpy()
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 3.0, 3.0])


class TestRollingMean:
    @pytest.mark.parametrize("window", WINDOW_SIZES)
    def test_matches_pandas(self, window):
        arr = _series(500, seed=window + 400)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_rolling_mean(ms, window)
        expected = pd.Series(arr).rolling(window, min_periods=1).mean().to_numpy()
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-4, atol=1e-3)


class TestRollingEdgeCases:
    def test_single_element_series(self):
        arr = np.array([42.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_rolling_sum(ms, 3)
        np.testing.assert_allclose(result.to_numpy(), [42.0])

    def test_window_zero_rejected(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        with pytest.raises(ValueError):
            metaldf_engine.metal_rolling_sum(ms, 0)

    def test_int32_supported(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        ms = metaldf_engine.MetalSeries.from_numpy_i32(arr)
        result = metaldf_engine.metal_rolling_sum(ms, 2)
        expected = np.array([1, 3, 5, 7, 9], dtype=np.int32)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_not_a_multiple_of_threadgroup_size(self):
        # THREADGROUP_SIZE is 256 in rust/src/kernels/rolling.rs -- exercise
        # the idx >= len bounds guard in the last (partial) threadgroup.
        arr = _series(1000, seed=42)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_rolling_sum(ms, 10)
        expected = pd.Series(arr).rolling(10, min_periods=1).sum().to_numpy()
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-4, atol=1e-3)
