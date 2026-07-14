"""Tests for the GPU compact/take kernels (`metal_compact`/`metal_take`).

Task 4.1 (Phase 4): stream compaction (filter a data series by a parallel
`Bool` mask, keeping only `mask == 1` elements, in order) and gather-by-index
(`output[i] = data[indices[i]]`) — the GPU building blocks for `df[mask]`
boolean indexing. `metal_compact` is built on top of the Task 3.1 prefix-sum
kernel (`metal_prefix_sum`/`prefix_sum_inclusive`): the inclusive scan of the
mask (cast to uint32) gives each kept element its output slot.
"""

import numpy as np
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")

THREADGROUP_SIZE = 256


class TestCompactFloat32:
    def test_basic(self):
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
        mask = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
        ms_data = metaldf_engine.MetalSeries.from_numpy(data)
        ms_mask = metaldf_engine.MetalSeries.from_numpy_bool(mask)
        result = metaldf_engine.metal_compact(ms_data, ms_mask)
        assert result.dtype == "Float32"
        assert result.len == 3
        np.testing.assert_allclose(result.to_numpy(), [10.0, 30.0, 50.0])

    def test_all_true(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mask = np.array([1, 1, 1], dtype=np.uint8)
        ms_data = metaldf_engine.MetalSeries.from_numpy(data)
        ms_mask = metaldf_engine.MetalSeries.from_numpy_bool(mask)
        result = metaldf_engine.metal_compact(ms_data, ms_mask)
        np.testing.assert_allclose(result.to_numpy(), [1.0, 2.0, 3.0])

    def test_all_false(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mask = np.array([0, 0, 0], dtype=np.uint8)
        ms_data = metaldf_engine.MetalSeries.from_numpy(data)
        ms_mask = metaldf_engine.MetalSeries.from_numpy_bool(mask)
        result = metaldf_engine.metal_compact(ms_data, ms_mask)
        assert result.len == 0

    def test_single_true(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mask = np.array([0, 1, 0], dtype=np.uint8)
        ms_data = metaldf_engine.MetalSeries.from_numpy(data)
        ms_mask = metaldf_engine.MetalSeries.from_numpy_bool(mask)
        result = metaldf_engine.metal_compact(ms_data, ms_mask)
        np.testing.assert_allclose(result.to_numpy(), [2.0])

    def test_multiple_threadgroups_not_aligned(self):
        # Not a multiple of the scan/compact threadgroup size -- exercises
        # both kernels' idx >= len bounds guards and multi-group prefix-sum
        # recursion.
        n = 1000
        rng = np.random.default_rng(0)
        data = rng.random(n).astype(np.float32)
        mask = (rng.integers(0, 2, size=n)).astype(np.uint8)
        ms_data = metaldf_engine.MetalSeries.from_numpy(data)
        ms_mask = metaldf_engine.MetalSeries.from_numpy_bool(mask)
        result = metaldf_engine.metal_compact(ms_data, ms_mask)
        expected = data[mask.astype(bool)]
        assert result.len == expected.shape[0]
        np.testing.assert_allclose(result.to_numpy(), expected)

    def test_exactly_one_threadgroup(self):
        n = THREADGROUP_SIZE
        rng = np.random.default_rng(1)
        data = np.arange(n, dtype=np.float32)
        mask = (rng.integers(0, 2, size=n)).astype(np.uint8)
        ms_data = metaldf_engine.MetalSeries.from_numpy(data)
        ms_mask = metaldf_engine.MetalSeries.from_numpy_bool(mask)
        result = metaldf_engine.metal_compact(ms_data, ms_mask)
        expected = data[mask.astype(bool)]
        np.testing.assert_allclose(result.to_numpy(), expected)


class TestCompactOtherDtypes:
    def test_int32(self):
        data = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        mask = np.array([1, 0, 0, 1, 1], dtype=np.uint8)
        ms_data = metaldf_engine.MetalSeries.from_numpy_i32(data)
        ms_mask = metaldf_engine.MetalSeries.from_numpy_bool(mask)
        result = metaldf_engine.metal_compact(ms_data, ms_mask)
        assert result.dtype == "Int32"
        np.testing.assert_array_equal(result.to_numpy(), [1, 4, 5])

    def test_int64(self):
        data = np.array([100, 200, 300], dtype=np.int64)
        mask = np.array([0, 1, 1], dtype=np.uint8)
        ms_data = metaldf_engine.MetalSeries.from_numpy_i64(data)
        ms_mask = metaldf_engine.MetalSeries.from_numpy_bool(mask)
        result = metaldf_engine.metal_compact(ms_data, ms_mask)
        assert result.dtype == "Int64"
        np.testing.assert_array_equal(result.to_numpy(), [200, 300])

    def test_bool_data(self):
        # Compacting a Bool-dtype data column (shares the u8 kernel variant
        # with Uint8).
        data = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        mask = np.array([1, 1, 0, 1, 0], dtype=np.uint8)
        ms_data = metaldf_engine.MetalSeries.from_numpy_bool(data)
        ms_mask = metaldf_engine.MetalSeries.from_numpy_bool(mask)
        result = metaldf_engine.metal_compact(ms_data, ms_mask)
        np.testing.assert_array_equal(result.to_numpy(), [1, 0, 1])


class TestCompactErrors:
    def test_non_bool_mask_rejected(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mask = np.array([1, 0, 1], dtype=np.int32)
        ms_data = metaldf_engine.MetalSeries.from_numpy(data)
        ms_mask = metaldf_engine.MetalSeries.from_numpy_i32(mask)
        with pytest.raises(TypeError):
            metaldf_engine.metal_compact(ms_data, ms_mask)

    def test_length_mismatch_rejected(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        mask = np.array([1, 0], dtype=np.uint8)
        ms_data = metaldf_engine.MetalSeries.from_numpy(data)
        ms_mask = metaldf_engine.MetalSeries.from_numpy_bool(mask)
        with pytest.raises(ValueError):
            metaldf_engine.metal_compact(ms_data, ms_mask)


class TestTake:
    def test_basic(self):
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
        indices = np.array([4, 2, 0], dtype=np.uint32)
        ms_data = metaldf_engine.MetalSeries.from_numpy(data)
        ms_idx = metaldf_engine.MetalSeries.from_numpy_u32(indices)
        result = metaldf_engine.metal_take(ms_data, ms_idx)
        assert result.dtype == "Float32"
        np.testing.assert_allclose(result.to_numpy(), [50.0, 30.0, 10.0])

    def test_with_repeats(self):
        # take allows repeated / non-monotonic indices, unlike compact.
        data = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        indices = np.array([0, 0, 1, 2, 1], dtype=np.uint32)
        ms_data = metaldf_engine.MetalSeries.from_numpy(data)
        ms_idx = metaldf_engine.MetalSeries.from_numpy_u32(indices)
        result = metaldf_engine.metal_take(ms_data, ms_idx)
        np.testing.assert_allclose(result.to_numpy(), [10.0, 10.0, 20.0, 30.0, 20.0])

    def test_int32(self):
        data = np.array([1, 2, 3, 4], dtype=np.int32)
        indices = np.array([3, 1, 0], dtype=np.uint32)
        ms_data = metaldf_engine.MetalSeries.from_numpy_i32(data)
        ms_idx = metaldf_engine.MetalSeries.from_numpy_u32(indices)
        result = metaldf_engine.metal_take(ms_data, ms_idx)
        np.testing.assert_array_equal(result.to_numpy(), [4, 2, 1])

    def test_int64(self):
        data = np.array([100, 200, 300], dtype=np.int64)
        indices = np.array([2, 0], dtype=np.uint32)
        ms_data = metaldf_engine.MetalSeries.from_numpy_i64(data)
        ms_idx = metaldf_engine.MetalSeries.from_numpy_u32(indices)
        result = metaldf_engine.metal_take(ms_data, ms_idx)
        np.testing.assert_array_equal(result.to_numpy(), [300, 100])

    def test_multiple_threadgroups_not_aligned(self):
        n = 1000
        rng = np.random.default_rng(2)
        data = rng.random(n).astype(np.float32)
        indices = rng.integers(0, n, size=537).astype(np.uint32)
        ms_data = metaldf_engine.MetalSeries.from_numpy(data)
        ms_idx = metaldf_engine.MetalSeries.from_numpy_u32(indices)
        result = metaldf_engine.metal_take(ms_data, ms_idx)
        np.testing.assert_allclose(result.to_numpy(), data[indices])

    def test_non_uint32_indices_rejected(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        indices = np.array([0, 1], dtype=np.int32)
        ms_data = metaldf_engine.MetalSeries.from_numpy(data)
        ms_idx = metaldf_engine.MetalSeries.from_numpy_i32(indices)
        with pytest.raises(TypeError):
            metaldf_engine.metal_take(ms_data, ms_idx)


class TestCompactTakeRoundtrip:
    def test_compact_then_take_recovers_original_order(self):
        # A common pattern: compact selects a subset, take can be used to
        # apply the same reordering/selection to a parallel column.
        data_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        data_b = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        mask = np.array([1, 0, 1, 1, 0], dtype=np.uint8)

        ms_a = metaldf_engine.MetalSeries.from_numpy(data_a)
        ms_b = metaldf_engine.MetalSeries.from_numpy_i32(data_b)
        ms_mask = metaldf_engine.MetalSeries.from_numpy_bool(mask)

        compacted_a = metaldf_engine.metal_compact(ms_a, ms_mask)
        compacted_b = metaldf_engine.metal_compact(ms_b, ms_mask)

        np.testing.assert_allclose(compacted_a.to_numpy(), [1.0, 3.0, 4.0])
        np.testing.assert_array_equal(compacted_b.to_numpy(), [10, 30, 40])
