"""Datetime/timedelta arithmetic type inference (Task 3).

pandas >= 2.0 (esp. 3.x) infers non-nanosecond resolution (e.g.
``datetime64[us]``/``timedelta64[s]``) from ``pd.to_datetime``/
``pd.to_timedelta`` by default. Metal's Datetime/Timedelta dtypes only
recognize the nanosecond resolution (``_DATETIME_DTYPE``/``_TIMEDELTA_DTYPE``
in ``_metal.py``), so every fixture below pins the resolution explicitly to
``ns`` -- otherwise the operands would silently miss the Metal dispatch path
entirely and these tests would only ever exercise the plain pandas fallback
(see the same note in ``tests/test_datetime_dtype.py``).
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


def _dates(strs):
    return pd.Series(pd.to_datetime(strs).astype("datetime64[ns]"))


def _timedeltas(values, unit):
    return pd.Series(pd.to_timedelta(values, unit=unit).astype("timedelta64[ns]"))


class TestDatetimeArithRules:
    """Direct `execute()` dispatch through `metaldf._engine._metal`."""

    def test_datetime_sub_datetime_gives_timedelta(self):
        from metaldf._engine import execute
        a = _dates(["2024-01-10", "2024-06-15"])
        b = _dates(["2024-01-01", "2024-06-01"])
        result = execute("sub", a, b)
        expected = a - b
        assert result.dtype == np.dtype("timedelta64[ns]")
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_datetime_add_timedelta_gives_datetime(self):
        from metaldf._engine import execute
        dates = _dates(["2024-01-01", "2024-06-01"])
        deltas = _timedeltas([1, 7], "D")
        result = execute("add", dates, deltas)
        expected = dates + deltas
        assert result.dtype == np.dtype("datetime64[ns]")
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_datetime_sub_timedelta_gives_datetime(self):
        from metaldf._engine import execute
        dates = _dates(["2024-01-10", "2024-06-15"])
        deltas = _timedeltas([1, 7], "D")
        result = execute("sub", dates, deltas)
        expected = dates - deltas
        assert result.dtype == np.dtype("datetime64[ns]")
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_timedelta_add_datetime_gives_datetime(self):
        """timedelta + datetime (operands reversed) is also a valid pandas
        combination and must produce Datetime, not Timedelta.
        """
        from metaldf._engine import execute
        deltas = _timedeltas([1, 7], "D")
        dates = _dates(["2024-01-01", "2024-06-01"])
        result = execute("add", deltas, dates)
        expected = deltas + dates
        assert result.dtype == np.dtype("datetime64[ns]")
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_timedelta_add_timedelta(self):
        from metaldf._engine import execute
        a = _timedeltas([1, 2, 3], "h")
        b = _timedeltas([4, 5, 6], "h")
        result = execute("add", a, b)
        expected = a + b
        assert result.dtype == np.dtype("timedelta64[ns]")
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_timedelta_sub_timedelta(self):
        from metaldf._engine import execute
        a = _timedeltas([4, 5, 6], "h")
        b = _timedeltas([1, 2, 3], "h")
        result = execute("sub", a, b)
        expected = a - b
        assert result.dtype == np.dtype("timedelta64[ns]")
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_datetime_add_datetime_falls_back(self):
        """pandas itself rejects datetime + datetime -- not in
        _DATETIME_ARITH_RULES, so this must fall back to (and get the same
        TypeError as) plain pandas.
        """
        from metaldf._engine import execute
        a = _dates(["2024-01-01"])
        b = _dates(["2024-06-01"])
        with pytest.raises((TypeError,)):
            execute("add", a, b)

    def test_result_index_and_name_preserved(self):
        from metaldf._engine import execute
        idx = pd.Index([10, 20, 30], name="idx")
        a = pd.Series(
            pd.to_datetime(["2024-01-10", "2024-06-15", "2024-12-01"]).astype("datetime64[ns]"),
            index=idx, name="ts",
        )
        b = pd.Series(
            pd.to_datetime(["2024-01-01", "2024-06-01", "2024-11-01"]).astype("datetime64[ns]"),
            index=idx, name="ts",
        )
        result = execute("sub", a, b)
        expected = a - b
        pd.testing.assert_series_equal(result, expected)

    def test_mismatched_index_falls_back_to_pandas_alignment(self):
        """Index mismatch must trigger pandas-side union alignment (NaT
        filling on the non-overlapping positions), not Metal's positional
        zip. Uses "add" (a valid Datetime+Timedelta combo) rather than
        "sub" -- PandasEngine's numpy-name fallback only maps op names that
        match an actual numpy ufunc (``np.add`` exists, ``np.sub`` doesn't;
        the real name is ``np.subtract``), and ``np.add`` on two
        differently-indexed Series correctly dispatches through pandas'
        own ``__array_ufunc__`` alignment.
        """
        from metaldf._engine import execute
        a = pd.Series(
            pd.to_datetime(["2024-01-10", "2024-06-15"]).astype("datetime64[ns]"),
            index=[0, 1],
        )
        b = pd.Series(
            pd.to_timedelta([1, 7], unit="D").astype("timedelta64[ns]"),
            index=[1, 2],
        )
        result = execute("add", a, b)
        expected = a + b
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_add_actually_dispatches_to_metal_binary_op(self, monkeypatch):
        """Confirm the Metal kernel is genuinely exercised (int64 add),
        not just correct by luck / silently falling back to pandas.
        """
        calls = []
        original = metaldf_engine.metal_binary_op

        def spy(op, lhs, rhs):
            calls.append(op)
            return original(op, lhs, rhs)

        monkeypatch.setattr(metaldf_engine, "metal_binary_op", spy)

        from metaldf._engine import execute
        dates = _dates(["2024-01-01", "2024-06-01"])
        deltas = _timedeltas([1, 7], "D")
        result = execute("add", dates, deltas)

        assert calls == ["add"]
        pd.testing.assert_series_equal(result, dates + deltas, check_names=False)


class TestDatetimeReduction:
    def test_datetime_min_max(self):
        from metaldf._engine import execute
        dates = _dates(["2023-03-15", "2021-01-01", "2024-12-31"])
        result_min = execute("min", dates)
        result_max = execute("max", dates)
        assert result_min == dates.min().value
        assert result_max == dates.max().value

    def test_datetime_sum_rejected(self):
        from metaldf._engine import execute
        dates = _dates(["2024-01-01", "2024-06-01"])
        with pytest.raises(TypeError):
            execute("sum", dates)

    def test_timedelta_sum_still_allowed(self):
        """Only Datetime sum is nonsensical -- Timedelta sum is a
        meaningful Timedelta and pandas itself supports it, so it must not
        be blocked the way Datetime sum is.
        """
        from metaldf._engine import execute
        deltas = _timedeltas([1, 2, 3], "h")
        result = execute("sum", deltas)
        expected_ns = deltas.sum().value
        assert result == expected_ns


class TestProxySeriesDatetimeArithmetic:
    """End-to-end through the ProxySeries `+`/`-` operators."""

    def test_proxy_datetime_sub(self):
        from metaldf._wrappers import ProxySeries
        a = ProxySeries(_pandas_obj=_dates(["2024-01-10", "2024-06-15"]))
        b = ProxySeries(_pandas_obj=_dates(["2024-01-01", "2024-06-01"]))
        result = a - b
        result = result.to_pandas() if hasattr(result, "to_pandas") else result
        expected = a.to_pandas() - b.to_pandas()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_proxy_datetime_add_timedelta(self):
        from metaldf._wrappers import ProxySeries
        dates = ProxySeries(_pandas_obj=_dates(["2024-01-01", "2024-06-01"]))
        deltas = ProxySeries(_pandas_obj=_timedeltas([1, 7], "D"))
        result = dates + deltas
        result = result.to_pandas() if hasattr(result, "to_pandas") else result
        expected = dates.to_pandas() + deltas.to_pandas()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_proxy_datetime_add_datetime_falls_back_to_pandas_error(self):
        from metaldf._wrappers import ProxySeries
        a = ProxySeries(_pandas_obj=_dates(["2024-01-01"]))
        b = ProxySeries(_pandas_obj=_dates(["2024-06-01"]))
        with pytest.raises(TypeError):
            a + b

    def test_proxy_timedelta_sub_datetime_falls_back_to_pandas_error(self):
        """timedelta - datetime is nonsensical and absent from
        _DATETIME_ARITH_RULES; pandas itself raises TypeError for it, and
        that's what must surface here too.
        """
        from metaldf._wrappers import ProxySeries
        a = ProxySeries(_pandas_obj=_timedeltas([1], "D"))
        b = ProxySeries(_pandas_obj=_dates(["2024-01-01"]))
        with pytest.raises(TypeError):
            a - b
