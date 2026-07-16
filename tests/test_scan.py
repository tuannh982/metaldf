"""Tests for the GPU prefix-sum / scan kernel (`metal_prefix_sum`).

Task 3.1 (Phase 3): a two-pass inclusive scan (Hillis-Steele per-threadgroup
scan + recursive partial-sum propagation) used as a building block by
Phase 4 (filtering/boolean indexing) and Phase 7 (rolling windows). Not
directly user-facing -- exercised here purely at the `metaldf_engine` level,
verified against `np.cumsum()`.

`metal_prefix_sum` is a thin wrapper around the op-generic `cumulative_scan`
dispatch (see `rust/src/kernels/scan.rs`); it supports Float32/Int32/Int64/
Uint32 (plus Datetime/Timedelta, which share the int64 kernels).
"""

import numpy as np
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")

SCAN_TG_SIZE = 256


class TestPrefixSumInt32:
    def test_small(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        ms = metaldf_engine.MetalSeries.from_numpy_i32(arr)
        result = metaldf_engine.metal_prefix_sum(ms)
        assert result.dtype == "Int32"
        assert result.len == len(arr)
        np.testing.assert_array_equal(result.to_numpy(), np.cumsum(arr))

    def test_single_element(self):
        arr = np.array([42], dtype=np.int32)
        ms = metaldf_engine.MetalSeries.from_numpy_i32(arr)
        result = metaldf_engine.metal_prefix_sum(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.cumsum(arr))

    @pytest.mark.skip(
        reason="Pre-existing bug (unrelated to scan): SharedBuffer::from_numpy_inner "
        "aborts the whole process (Rust panic -> SIGABRT, not a catchable Python "
        "exception) on a zero-length numpy array -- reproduces identically on "
        "MetalSeries.from_numpy_bool([]) on main, before any Task 3.1 changes. "
        "Not something metal_prefix_sum can work around since the crash happens "
        "during MetalSeries construction, before metal_prefix_sum ever runs."
    )
    def test_empty(self):
        arr = np.array([], dtype=np.int32)
        ms = metaldf_engine.MetalSeries.from_numpy_i32(arr)
        result = metaldf_engine.metal_prefix_sum(ms)
        assert result.len == 0
        np.testing.assert_array_equal(result.to_numpy(), np.cumsum(arr))

    def test_negative_values(self):
        rng = np.random.default_rng(0)
        arr = rng.integers(-100, 100, size=37).astype(np.int32)
        ms = metaldf_engine.MetalSeries.from_numpy_i32(arr)
        result = metaldf_engine.metal_prefix_sum(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.cumsum(arr))

    def test_exactly_one_threadgroup(self):
        # Exactly SCAN_TG_SIZE elements -> num_groups == 1, no recursion.
        arr = np.arange(1, SCAN_TG_SIZE + 1, dtype=np.int32)
        ms = metaldf_engine.MetalSeries.from_numpy_i32(arr)
        result = metaldf_engine.metal_prefix_sum(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.cumsum(arr))

    def test_one_more_than_threadgroup(self):
        # SCAN_TG_SIZE + 1 elements -> num_groups == 2, minimal recursion.
        arr = np.arange(1, SCAN_TG_SIZE + 2, dtype=np.int32)
        ms = metaldf_engine.MetalSeries.from_numpy_i32(arr)
        result = metaldf_engine.metal_prefix_sum(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.cumsum(arr))

    def test_medium_multiple_threadgroups_not_aligned(self):
        # Not a multiple of the threadgroup size -- exercises the
        # idx >= len bounds guard in the last (partial) threadgroup.
        n = 1000
        rng = np.random.default_rng(1)
        arr = rng.integers(-50, 50, size=n).astype(np.int32)
        ms = metaldf_engine.MetalSeries.from_numpy_i32(arr)
        result = metaldf_engine.metal_prefix_sum(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.cumsum(arr))

    def test_large_multi_level_recursion(self):
        # 100_000 elements -> ~391 groups in pass 1, which itself needs a
        # second-level scan (391 > SCAN_TG_SIZE is false actually since
        # 391 < 256*2 groups fit in 2 threadgroups at the partials level) --
        # regardless, this exercises >1 level of the recursive partial scan.
        arr = np.ones(100_000, dtype=np.int32)
        ms = metaldf_engine.MetalSeries.from_numpy_i32(arr)
        result = metaldf_engine.metal_prefix_sum(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.cumsum(arr))

    def test_very_large_forces_deep_recursion(self):
        # 20_000_000 elements -> 78125 groups in pass 1; the partials buffer
        # from pass 1 itself spans multiple threadgroups (needs its own
        # multi-group scan), and so does *that* level's partials buffer --
        # exercising several levels of the recursive partial-sum scan.
        n = 20_000_000
        arr = np.ones(n, dtype=np.int32)
        ms = metaldf_engine.MetalSeries.from_numpy_i32(arr)
        result = metaldf_engine.metal_prefix_sum(ms)
        expected = np.cumsum(arr)
        np.testing.assert_array_equal(result.to_numpy(), expected)


class TestPrefixSumUint32:
    def test_small(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=np.uint32)
        ms = metaldf_engine.MetalSeries.from_numpy_u32(arr)
        result = metaldf_engine.metal_prefix_sum(ms)
        assert result.dtype == "Uint32"
        np.testing.assert_array_equal(result.to_numpy(), np.cumsum(arr))

    def test_medium_multiple_threadgroups(self):
        n = 1500
        arr = np.arange(n, dtype=np.uint32) % 7  # keep values small, non-negative
        ms = metaldf_engine.MetalSeries.from_numpy_u32(arr)
        result = metaldf_engine.metal_prefix_sum(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.cumsum(arr, dtype=np.uint32))


class TestPrefixSumNowSupportedDtype:
    # Task 1 (op-generic scan.metal) added sum kernels for float32/int64,
    # and Task 2's `cumulative_scan` dispatch (which `metal_prefix_sum` now
    # wraps) accepts them -- previously these dtypes raised TypeError since
    # only int32_/uint32_-suffixed sum kernels existed.
    def test_float32_now_supported(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_prefix_sum(ms)
        np.testing.assert_allclose(result.to_numpy(), np.cumsum(arr), rtol=1e-5)

    def test_int64_now_supported(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        ms = metaldf_engine.MetalSeries.from_numpy_i64(arr)
        result = metaldf_engine.metal_prefix_sum(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.cumsum(arr))


class TestPrefixSumUnsupportedDtype:
    def test_bool_rejected(self):
        arr = np.array([1, 0, 1], dtype=np.uint8)
        ms = metaldf_engine.MetalSeries.from_numpy_bool(arr)
        with pytest.raises(TypeError):
            metaldf_engine.metal_prefix_sum(ms)
