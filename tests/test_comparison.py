import numpy as np
import pandas as pd
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")


@pytest.mark.parametrize("op", ["eq", "ne", "lt", "le", "gt", "ge"])
def test_compare_i64(op):
    a = np.array([1, 5, 3, 7, 2], dtype=np.int64)
    b = np.array([2, 5, 1, 8, 2], dtype=np.int64)
    ma = metaldf_engine.MetalSeries.from_numpy_i64(a)
    mb = metaldf_engine.MetalSeries.from_numpy_i64(b)
    result = metaldf_engine.metal_compare_op(op, ma, mb)
    got = result.to_numpy()
    ops = {"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b}
    np.testing.assert_array_equal(got, ops[op].astype(np.int32))


@pytest.mark.parametrize("op", ["eq", "ne", "lt", "le", "gt", "ge"])
def test_compare_i32(op):
    a = np.array([1, 5, 3, 7, 2], dtype=np.int32)
    b = np.array([2, 5, 1, 8, 2], dtype=np.int32)
    ma = metaldf_engine.MetalSeries.from_numpy_i32(a)
    mb = metaldf_engine.MetalSeries.from_numpy_i32(b)
    result = metaldf_engine.metal_compare_op(op, ma, mb)
    got = result.to_numpy()
    ops = {"eq": a == b, "ne": a != b, "lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b}
    np.testing.assert_array_equal(got, ops[op].astype(np.int32))


def test_compare_datetime_gt():
    """Datetime series (as viewed int64 nanoseconds) compare correctly via
    the i64 comparison kernels.

    NOTE: `MetalSeries.from_numpy_datetime` doesn't exist yet -- the
    "datetime dtype" task (Task 1) that adds it lands in a separate
    worktree/branch. This constructs the underlying int64-nanosecond view
    directly with `from_numpy_i64` so the i64 comparison path this task
    adds is exercised the same way Task 1's dispatch will use it once
    merged. TODO(datetime): once Task 1 lands, switch to
    `MetalSeries.from_numpy_datetime` if/when that constructor exists.
    """
    dates_a = pd.to_datetime(["2023-01-01", "2024-06-15", "2021-03-01"])
    dates_b = pd.to_datetime(["2022-12-31", "2024-06-15", "2022-01-01"])
    a_ns = dates_a.values.astype("datetime64[ns]").view(np.int64)
    b_ns = dates_b.values.astype("datetime64[ns]").view(np.int64)
    ma = metaldf_engine.MetalSeries.from_numpy_i64(a_ns)
    mb = metaldf_engine.MetalSeries.from_numpy_i64(b_ns)
    result = metaldf_engine.metal_compare_op("gt", ma, mb)
    got = result.to_numpy()
    expected = np.asarray(dates_a > dates_b).astype(np.int32)
    np.testing.assert_array_equal(got, expected)


def test_compare_f32():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([3.0, 2.0, 1.0], dtype=np.float32)
    ma = metaldf_engine.MetalSeries.from_numpy(a)
    mb = metaldf_engine.MetalSeries.from_numpy(b)
    result = metaldf_engine.metal_compare_op("lt", ma, mb)
    np.testing.assert_array_equal(result.to_numpy(), [1, 0, 0])


def test_compare_dtype_mismatch_raises():
    a = np.array([1.0], dtype=np.float32)
    b = np.array([1], dtype=np.int32)
    ma = metaldf_engine.MetalSeries.from_numpy(a)
    mb = metaldf_engine.MetalSeries.from_numpy_i32(b)
    with pytest.raises(TypeError):
        metaldf_engine.metal_compare_op("eq", ma, mb)


def test_compare_length_mismatch_raises():
    a = np.array([1, 2, 3], dtype=np.int64)
    b = np.array([1, 2], dtype=np.int64)
    ma = metaldf_engine.MetalSeries.from_numpy_i64(a)
    mb = metaldf_engine.MetalSeries.from_numpy_i64(b)
    with pytest.raises(ValueError):
        metaldf_engine.metal_compare_op("gt", ma, mb)


def test_compare_empty():
    a = np.array([], dtype=np.int64)
    b = np.array([], dtype=np.int64)
    ma = metaldf_engine.MetalSeries.from_numpy_i64(a)
    mb = metaldf_engine.MetalSeries.from_numpy_i64(b)
    result = metaldf_engine.metal_compare_op("eq", ma, mb)
    assert result.to_numpy().shape == (0,)


class TestPythonDispatch:
    """Exercises the Python-level registry dispatch added in
    `metaldf._engine._metal` / `metaldf._engine.__init__` (`execute("cmp_*",
    ...)`), as opposed to calling `metaldf_engine.metal_compare_op` directly.
    """

    @pytest.mark.parametrize("op,expected_fn", [
        ("cmp_eq", lambda a, b: a == b),
        ("cmp_ne", lambda a, b: a != b),
        ("cmp_lt", lambda a, b: a < b),
        ("cmp_le", lambda a, b: a <= b),
        ("cmp_gt", lambda a, b: a > b),
        ("cmp_ge", lambda a, b: a >= b),
    ])
    def test_execute_cmp_i64(self, op, expected_fn):
        from metaldf._engine import execute

        a = pd.Series(np.array([1, 5, 3, 7, 2], dtype=np.int64))
        b = pd.Series(np.array([2, 5, 1, 8, 2], dtype=np.int64))
        result = execute(op, a, b)
        expected = expected_fn(a, b)
        assert result.dtype == bool
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_metal_cmp_dtype_mismatch_raises_metal_not_available(self):
        """Mismatched dtypes should raise `MetalNotAvailable` at the Python
        dispatch layer (`_dispatch_compare`), the signal `execute()` uses
        to fall back to `PandasEngine` -- not propagate the Rust `TypeError`
        `metaldf_engine.metal_compare_op` itself would raise.
        """
        from metaldf._engine._metal import MetalEngine
        from metaldf.exceptions import MetalNotAvailable

        a = pd.Series(np.array([1.0, 2.0], dtype=np.float32))
        b = pd.Series(np.array([1, 2], dtype=np.int32))
        with pytest.raises(MetalNotAvailable):
            MetalEngine.metal_cmp_eq(a, b)

    def test_metal_cmp_unsupported_dtype_raises_metal_not_available(self):
        from metaldf._engine._metal import MetalEngine
        from metaldf.exceptions import MetalNotAvailable

        a = pd.Series(["a", "b"])
        b = pd.Series(["a", "c"])
        with pytest.raises(MetalNotAvailable):
            MetalEngine.metal_cmp_eq(a, b)
