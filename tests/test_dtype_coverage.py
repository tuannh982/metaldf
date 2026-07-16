"""Coverage check: verify all numeric dtypes work across all kernel families.

Tests every combination of (dtype × operation) that we claim to support.
"""

import numpy as np
import pytest

import metaldf_engine


# --- dtype constructors and configs ---

DTYPES = {
    "float32": (np.float32, metaldf_engine.MetalSeries.from_numpy, 4),
    "int8":    (np.int8,    metaldf_engine.MetalSeries.from_numpy_i8, 1),
    "int16":   (np.int16,   metaldf_engine.MetalSeries.from_numpy_i16, 2),
    "int32":   (np.int32,   metaldf_engine.MetalSeries.from_numpy_i32, 4),
    "int64":   (np.int64,   metaldf_engine.MetalSeries.from_numpy_i64, 8),
    "uint8":   (np.uint8,   metaldf_engine.MetalSeries.from_numpy_u8, 1),
    "uint16":  (np.uint16,  metaldf_engine.MetalSeries.from_numpy_u16, 2),
    "uint32":  (np.uint32,  metaldf_engine.MetalSeries.from_numpy_u32, 4),
    "uint64":  (np.uint64,  metaldf_engine.MetalSeries.from_numpy_u64, 8),
}

SIGNED_DTYPES = ["float32", "int8", "int16", "int32", "int64"]
UNSIGNED_DTYPES = ["uint8", "uint16", "uint32", "uint64"]
ALL_DTYPES = list(DTYPES.keys())


def make_series(dtype_name, values):
    np_dtype, ctor, _ = DTYPES[dtype_name]
    arr = np.array(values, dtype=np_dtype)
    return ctor(arr), arr


# =====================================================================
# Elementwise binary ops
# =====================================================================

class TestBinaryOps:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("op", ["add", "sub", "mul"])
    def test_binary_ops(self, dtype, op):
        a, a_np = make_series(dtype, [10, 20, 30, 40])
        b, b_np = make_series(dtype, [1, 2, 3, 4])
        result = metaldf_engine.metal_binary_op(op, a, b)
        out = result.to_numpy()
        if op == "add":
            expected = a_np + b_np
        elif op == "sub":
            expected = a_np - b_np
        elif op == "mul":
            expected = a_np * b_np
        np.testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_div(self, dtype):
        a, _ = make_series(dtype, [10, 20, 30, 40])
        b, _ = make_series(dtype, [2, 4, 5, 8])
        result = metaldf_engine.metal_binary_op("div", a, b)
        out = result.to_numpy()
        np_dtype = DTYPES[dtype][0]
        expected = np.array([10, 20, 30, 40], dtype=np_dtype) // np.array([2, 4, 5, 8], dtype=np_dtype)
        if dtype == "float32":
            expected = np.array([10, 20, 30, 40], dtype=np.float32) / np.array([2, 4, 5, 8], dtype=np.float32)
            np.testing.assert_allclose(out, expected, rtol=1e-5)
        else:
            np.testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_mod(self, dtype):
        a, _ = make_series(dtype, [10, 21, 30, 43])
        b, _ = make_series(dtype, [3, 4, 7, 8])
        result = metaldf_engine.metal_binary_op("mod", a, b)
        out = result.to_numpy()
        np_dtype = DTYPES[dtype][0]
        a_np = np.array([10, 21, 30, 43], dtype=np_dtype)
        b_np = np.array([3, 4, 7, 8], dtype=np_dtype)
        if dtype == "float32":
            expected = np.fmod(a_np, b_np)
            np.testing.assert_allclose(out, expected, rtol=1e-5)
        else:
            expected = a_np % b_np
            np.testing.assert_array_equal(out, expected)


# =====================================================================
# Unary ops
# =====================================================================

class TestUnaryOps:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_neg(self, dtype):
        a, a_np = make_series(dtype, [1, 2, 3, 4])
        result = metaldf_engine.metal_unary_op("neg", a)
        out = result.to_numpy()
        expected = -a_np
        np.testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_abs(self, dtype):
        if dtype in UNSIGNED_DTYPES:
            a, a_np = make_series(dtype, [1, 2, 3, 4])
        else:
            a, a_np = make_series(dtype, [-1, 2, -3, 4])
        result = metaldf_engine.metal_unary_op("abs", a)
        out = result.to_numpy()
        expected = np.abs(a_np)
        np.testing.assert_array_equal(out, expected)


# =====================================================================
# Comparisons
# =====================================================================

class TestComparisons:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    @pytest.mark.parametrize("op", ["eq", "ne", "lt", "le", "gt", "ge"])
    def test_compare(self, dtype, op):
        a, _ = make_series(dtype, [1, 5, 3, 7])
        b, _ = make_series(dtype, [2, 5, 1, 7])
        result = metaldf_engine.metal_compare_op(op, a, b)
        out = result.to_numpy()
        np_dtype = DTYPES[dtype][0]
        a_np = np.array([1, 5, 3, 7], dtype=np_dtype)
        b_np = np.array([2, 5, 1, 7], dtype=np_dtype)
        ops = {"eq": a_np == b_np, "ne": a_np != b_np, "lt": a_np < b_np,
               "le": a_np <= b_np, "gt": a_np > b_np, "ge": a_np >= b_np}
        expected = ops[op].astype(np.int32)
        np.testing.assert_array_equal(out, expected)


# =====================================================================
# Reductions
# =====================================================================

class TestReductions:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_sum(self, dtype):
        a, a_np = make_series(dtype, [1, 2, 3, 4, 5])
        result = metaldf_engine.metal_sum(a)
        assert result == int(a_np.sum()), f"sum mismatch for {dtype}: {result} != {a_np.sum()}"

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_min(self, dtype):
        a, a_np = make_series(dtype, [5, 1, 3, 2, 4])
        result = metaldf_engine.metal_min(a)
        assert result == a_np.min()

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_max(self, dtype):
        a, a_np = make_series(dtype, [5, 1, 3, 2, 4])
        result = metaldf_engine.metal_max(a)
        assert result == a_np.max()

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_mean(self, dtype):
        a, a_np = make_series(dtype, [2, 4, 6, 8])
        result = metaldf_engine.metal_mean(a)
        np.testing.assert_allclose(result, float(a_np.mean()), rtol=1e-5)


# =====================================================================
# Sort + argsort
# =====================================================================

class TestSort:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_sort(self, dtype):
        a, a_np = make_series(dtype, [5, 3, 1, 4, 2])
        result = metaldf_engine.metal_sort(a)
        expected = np.sort(a_np)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_argsort(self, dtype):
        a, a_np = make_series(dtype, [5, 3, 1, 4, 2])
        result = metaldf_engine.metal_argsort(a)
        expected = np.argsort(a_np).astype(np.int32)
        np.testing.assert_array_equal(result.to_numpy(), expected)


# =====================================================================
# Cumulative ops
# =====================================================================

class TestCumulative:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_cumsum(self, dtype):
        a, a_np = make_series(dtype, [1, 2, 3, 4, 5])
        result = metaldf_engine.metal_cumsum(a)
        expected = np.cumsum(a_np)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_cummin(self, dtype):
        a, a_np = make_series(dtype, [5, 3, 4, 1, 2])
        result = metaldf_engine.metal_cummin(a)
        expected = np.minimum.accumulate(a_np)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_cummax(self, dtype):
        a, a_np = make_series(dtype, [1, 3, 2, 5, 4])
        result = metaldf_engine.metal_cummax(a)
        expected = np.maximum.accumulate(a_np)
        np.testing.assert_array_equal(result.to_numpy(), expected)


# =====================================================================
# Filter (compact + take)
# =====================================================================

class TestFilter:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_compact(self, dtype):
        data, data_np = make_series(dtype, [10, 20, 30, 40, 50])
        mask_arr = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
        mask = metaldf_engine.MetalSeries.from_numpy_bool(mask_arr)
        result = metaldf_engine.metal_compact(data, mask)
        expected = data_np[mask_arr.astype(bool)]
        np.testing.assert_array_equal(result.to_numpy(), expected)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_take(self, dtype):
        data, data_np = make_series(dtype, [10, 20, 30, 40, 50])
        indices = np.array([4, 2, 0], dtype=np.uint32)
        idx_series = metaldf_engine.MetalSeries.from_numpy_u32(indices)
        result = metaldf_engine.metal_take(data, idx_series)
        expected = data_np[[4, 2, 0]]
        np.testing.assert_array_equal(result.to_numpy(), expected)


# =====================================================================
# Shift
# =====================================================================

class TestShift:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_shift(self, dtype):
        a, a_np = make_series(dtype, [10, 20, 30, 40, 50])
        result = metaldf_engine.metal_shift(a, 2)
        out = result.to_numpy()
        assert len(out) == 5
        np.testing.assert_array_equal(out[2:], a_np[:3])


# =====================================================================
# Rolling
# =====================================================================

class TestRolling:
    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_rolling_sum(self, dtype):
        a, a_np = make_series(dtype, [1, 2, 3, 4, 5])
        result = metaldf_engine.metal_rolling_sum(a, 3)
        out = result.to_numpy()
        np_dtype = DTYPES[dtype][0]
        expected = np.array([1, 3, 6, 9, 12], dtype=np_dtype)
        np.testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_rolling_min(self, dtype):
        a, a_np = make_series(dtype, [5, 3, 4, 1, 2])
        result = metaldf_engine.metal_rolling_min(a, 3)
        out = result.to_numpy()
        np_dtype = DTYPES[dtype][0]
        expected = np.array([5, 3, 3, 1, 1], dtype=np_dtype)
        np.testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize("dtype", ALL_DTYPES)
    def test_rolling_max(self, dtype):
        a, a_np = make_series(dtype, [1, 5, 2, 4, 3])
        result = metaldf_engine.metal_rolling_max(a, 3)
        out = result.to_numpy()
        np_dtype = DTYPES[dtype][0]
        expected = np.array([1, 5, 5, 5, 4], dtype=np_dtype)
        np.testing.assert_array_equal(out, expected)


# =====================================================================
# Coverage matrix summary
# =====================================================================

def test_coverage_matrix():
    """Not a real test — prints the coverage matrix as a sanity check."""
    ops = {
        "binary(add/sub/mul/div/mod)": ALL_DTYPES,
        "unary(abs/neg)": ALL_DTYPES,
        "comparison(eq/ne/lt/le/gt/ge)": ALL_DTYPES,
        "reduction(sum/min/max/mean)": ALL_DTYPES,
        "sort/argsort": ALL_DTYPES,
        "cumulative(cumsum/cummin/cummax)": ALL_DTYPES,
        "filter(compact/take)": ALL_DTYPES,
        "shift": ALL_DTYPES,
        "rolling(sum/min/max)": ALL_DTYPES,
        "fillna": ["float32"],
        "ffill/bfill": ["float32"],
        "groupby": ["float32", "int32"],
        "join": ["float32", "int32"],
        "expression_eval": ["float32"],
    }
    total = 0
    for op, dtypes in ops.items():
        total += len(dtypes)
    assert total > 0
