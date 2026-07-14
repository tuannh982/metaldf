"""End-to-end tests for Task 1.6: Python layer null integration.

Verifies that a user writing plain pandas-shaped code (``df["a"] + df["b"]``,
``series.sum()``, etc.) against float32 data containing NaN gets the same
answer through the Metal engine as through real pandas -- i.e. NaN
propagation for elementwise ops and skip-null (``skipna=True``) semantics for
reductions -- rather than silently getting a wrong answer or an unexplained
fallback to pandas.
"""

import math

import numpy as np
import pandas as pd
import pytest

from metaldf._engine import execute

try:
    import metaldf_engine

    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")


@pytest.fixture(autouse=True)
def _ensure_metal_ops_registered():
    """Guard against cross-file registry pollution.

    ``metaldf._engine``'s op registry is a single process-wide dict, and
    ``tests/test_engine.py`` has tests that call ``clear_registry()`` and
    then only re-register a throwaway op name (e.g. ``test_execute_with_args_and_kwargs``
    registers only "greet"). Since ``metaldf._engine`` is only ever imported
    once per test session, that leaves "sum"/"add"/"sub"/"mul"/"div"/etc.
    unregistered for every test file that runs later in the same session
    (alphabetically after ``test_engine.py``, which includes this file) --
    silently falling through to ``PandasEngine``/``numpy`` instead of
    dispatching to Metal, which would let these tests "pass" without
    actually exercising the null-mask code paths under test. Re-registering
    the ops this file exercises before each test closes that gap
    independent of what ran before it or in what order.
    """
    from metaldf._engine import register
    from metaldf._engine._metal import MetalEngine

    register("sum", MetalEngine.metal_sum)
    register("min", MetalEngine.metal_min)
    register("max", MetalEngine.metal_max)
    register("mean", MetalEngine.metal_mean)
    register("add", MetalEngine.metal_add)
    register("sub", MetalEngine.metal_sub)
    register("mul", MetalEngine.metal_mul)
    register("div", MetalEngine.metal_div)


# ---------------------------------------------------------------------------
# _make_series_with_nulls / _result_to_series_with_nulls unit coverage
# ---------------------------------------------------------------------------


class TestMakeSeriesWithNulls:
    def test_float32_with_nan_builds_masked_series(self):
        from metaldf._engine._metal import _make_series_with_nulls

        arr = np.array([1.0, float("nan"), 3.0], dtype=np.float32)
        buf = _make_series_with_nulls(arr)
        assert buf.null_mask is not None
        np.testing.assert_array_equal(buf.null_mask, [True, False, True])

    def test_float32_without_nan_has_no_mask(self):
        from metaldf._engine._metal import _make_series_with_nulls

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        buf = _make_series_with_nulls(arr)
        assert buf.null_mask is None

    def test_int32_ignores_null_detection(self):
        from metaldf._engine._metal import _make_series_with_nulls

        arr = np.array([1, 2, 3], dtype=np.int32)
        buf = _make_series_with_nulls(arr)
        assert buf.null_mask is None
        np.testing.assert_array_equal(buf.to_numpy(), arr)

    def test_int64_ignores_null_detection(self):
        from metaldf._engine._metal import _make_series_with_nulls

        arr = np.array([1, 2, 3], dtype=np.int64)
        buf = _make_series_with_nulls(arr)
        assert buf.null_mask is None


class TestResultToSeriesWithNulls:
    def test_no_mask_passes_through_native_dtype(self):
        from metaldf._engine._metal import _result_to_series_with_nulls

        ms = metaldf_engine.MetalSeries.from_numpy(
            np.array([1.0, 2.0, 3.0], dtype=np.float32)
        )
        series = _result_to_series_with_nulls(ms, index=pd.RangeIndex(3), name="x")
        assert series.dtype == np.float32
        assert series.name == "x"
        np.testing.assert_allclose(series.values, [1.0, 2.0, 3.0])

    def test_mask_restores_nan_positions(self):
        from metaldf._engine._metal import _result_to_series_with_nulls

        ms = metaldf_engine.MetalSeries.from_numpy_with_nulls(
            np.array([1.0, float("nan"), 3.0], dtype=np.float32)
        )
        series = _result_to_series_with_nulls(ms)
        assert math.isnan(series.iloc[1])
        assert series.iloc[0] == 1.0
        assert series.iloc[2] == 3.0


# ---------------------------------------------------------------------------
# Elementwise binary ops via metaldf._engine.execute (the eager Metal path)
# ---------------------------------------------------------------------------


class TestBinaryOpsWithNulls:
    def test_add_with_nans_matches_pandas(self):
        a = pd.Series([1.0, float("nan"), 3.0, 4.0], dtype=np.float32)
        b = pd.Series([10.0, 20.0, float("nan"), 40.0], dtype=np.float32)
        expected = a + b
        result = execute("add", a, b)
        pd.testing.assert_series_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            rtol=1e-5,
        )

    @pytest.mark.parametrize(
        "op,pandas_fn",
        [
            ("sub", lambda x, y: x - y),
            ("mul", lambda x, y: x * y),
            ("div", lambda x, y: x / y),
        ],
    )
    def test_sub_mul_div_with_nans_match_pandas(self, op, pandas_fn):
        a = pd.Series([10.0, float("nan"), 30.0], dtype=np.float32)
        b = pd.Series([1.0, 2.0, float("nan")], dtype=np.float32)
        expected = pandas_fn(a, b)
        result = execute(op, a, b)
        pd.testing.assert_series_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            rtol=1e-5,
        )

    def test_add_no_nans_preserves_float32_dtype(self):
        """Regression: series with no NaN at all must not pay the float64 upcast.

        Output dtype should stay float32, matching pandas exactly.
        """
        a = pd.Series([1.0, 2.0, 3.0], dtype=np.float32)
        b = pd.Series([4.0, 5.0, 6.0], dtype=np.float32)
        expected = a + b
        result = execute("add", a, b)
        assert result.dtype == np.float32
        pd.testing.assert_series_equal(result, expected, check_dtype=True)

    def test_only_one_operand_has_nulls(self):
        a = pd.Series([1.0, float("nan"), 3.0, 4.0], dtype=np.float32)
        b = pd.Series([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        expected = a + b
        result = execute("add", a, b)
        pd.testing.assert_series_equal(
            result.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            rtol=1e-5,
        )

    def test_all_nan_operand_result_all_nan(self):
        a = pd.Series([float("nan"), float("nan")], dtype=np.float32)
        b = pd.Series([1.0, 2.0], dtype=np.float32)
        result = execute("add", a, b)
        assert result.isna().all()

    def test_index_and_name_preserved_with_nulls(self):
        a = pd.Series([1.0, float("nan"), 3.0], dtype=np.float32, index=[5, 6, 7], name="x")
        b = pd.Series([10.0, 20.0, float("nan")], dtype=np.float32, index=[5, 6, 7], name="x")
        result = execute("add", a, b)
        assert list(result.index) == [5, 6, 7]
        assert result.name == "x"


# ---------------------------------------------------------------------------
# Reductions via metaldf._engine.execute
# ---------------------------------------------------------------------------


class TestReductionsWithNulls:
    def test_sum_with_nans_matches_pandas(self):
        s = pd.Series([1.0, float("nan"), 3.0], dtype=np.float32)
        expected = s.sum()
        result = execute("sum", s)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_mean_with_nans_matches_pandas(self):
        s = pd.Series([2.0, float("nan"), 4.0], dtype=np.float32)
        expected = s.mean()
        result = execute("mean", s)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_min_with_nans_matches_pandas(self):
        s = pd.Series([float("nan"), 3.0, 1.0, float("nan")], dtype=np.float32)
        expected = s.min()
        result = execute("min", s)
        np.testing.assert_allclose(result, expected)

    def test_max_with_nans_matches_pandas(self):
        s = pd.Series([float("nan"), 3.0, 1.0, float("nan")], dtype=np.float32)
        expected = s.max()
        result = execute("max", s)
        np.testing.assert_allclose(result, expected)

    def test_all_nan_sum_returns_nan(self):
        s = pd.Series([float("nan"), float("nan")], dtype=np.float32)
        result = execute("sum", s)
        assert math.isnan(result)

    def test_all_nan_matches_pandas_skipna_false_semantics(self):
        """Document an intentional divergence from pandas for all-null input.

        pandas' own default ``sum()`` with all-NaN input returns 0.0
        (skipna=True, empty-after-dropna sums to 0); Metal's all-null
        reduction instead returns NaN by design (see
        tests/test_null_reductions.py). Assert that divergence explicitly
        rather than equality with pandas here.
        """
        s = pd.Series([float("nan"), float("nan")], dtype=np.float32)
        assert s.sum() == 0.0
        assert math.isnan(execute("sum", s))

    def test_no_nans_matches_unmasked_reduction(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        assert execute("sum", s) == s.sum()
        assert execute("min", s) == s.min()
        assert execute("max", s) == s.max()
        np.testing.assert_allclose(execute("mean", s), s.mean())


# ---------------------------------------------------------------------------
# Full user-facing stack: metaldf.install() + proxy dunder/methods
# ---------------------------------------------------------------------------


class TestEndToEndProxyStack:
    def test_dataframe_column_add_with_nans(self):
        import metaldf

        metaldf.install()
        try:
            df = pd.DataFrame(
                {
                    "a": pd.array([1.0, float("nan"), 3.0], dtype="float32"),
                    "b": pd.array([10.0, 20.0, float("nan")], dtype="float32"),
                }
            )
            result = df["a"] + df["b"]
            result = result.to_pandas() if hasattr(result, "to_pandas") else result

            plain_df = pd.DataFrame(
                {
                    "a": pd.array([1.0, float("nan"), 3.0], dtype="float32"),
                    "b": pd.array([10.0, 20.0, float("nan")], dtype="float32"),
                }
            )
            expected = plain_df["a"] + plain_df["b"]

            pd.testing.assert_series_equal(result, expected, check_dtype=False)
        finally:
            metaldf.uninstall()

    def test_series_sum_with_nans(self):
        import metaldf

        metaldf.install()
        try:
            s = pd.Series(np.array([1.0, float("nan"), 3.0], dtype=np.float32))
            result = s.sum()
            expected = pd.Series(np.array([1.0, float("nan"), 3.0], dtype=np.float32)).sum()
            np.testing.assert_allclose(result, expected)
        finally:
            metaldf.uninstall()

    def test_series_mean_min_max_with_nans(self):
        import metaldf

        metaldf.install()
        try:
            raw = np.array([5.0, float("nan"), 1.0, 9.0, float("nan")], dtype=np.float32)
            s = pd.Series(raw)
            plain = pd.Series(raw)

            np.testing.assert_allclose(s.mean(), plain.mean())
            np.testing.assert_allclose(s.min(), plain.min())
            np.testing.assert_allclose(s.max(), plain.max())
        finally:
            metaldf.uninstall()
