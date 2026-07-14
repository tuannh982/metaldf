import numpy as np
import pandas as pd
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")


def test_from_numpy_datetime_roundtrip():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    ns_arr = dates.values.astype("datetime64[ns]").view(np.int64)
    ms = metaldf_engine.MetalSeries.from_numpy_datetime(ns_arr)
    assert ms.len == 5
    assert ms.dtype == "Datetime"
    result = ms.to_numpy()
    np.testing.assert_array_equal(result, ns_arr)


def test_from_numpy_timedelta_roundtrip():
    td = pd.to_timedelta([1, 2, 3, 4, 5], unit="h")
    ns_arr = td.values.astype("timedelta64[ns]").view(np.int64)
    ms = metaldf_engine.MetalSeries.from_numpy_timedelta(ns_arr)
    assert ms.len == 5
    assert ms.dtype == "Timedelta"
    result = ms.to_numpy()
    np.testing.assert_array_equal(result, ns_arr)


def test_datetime_sort_matches_pandas():
    # NOTE: pandas >= 2.0 (esp. 3.x) infers non-nanosecond resolution
    # (e.g. datetime64[us]) from date-only strings by default. Metal's
    # Datetime dtype only recognizes datetime64[ns] (see _DATETIME_DTYPE in
    # _metal.py), so pin the resolution explicitly to ns to actually
    # exercise the Metal dispatch path here instead of silently falling
    # back to pandas.
    from metaldf._engine import execute
    dates = pd.Series(
        pd.to_datetime(["2023-03-15", "2021-01-01", "2024-12-31", "2022-06-15"]).astype("datetime64[ns]")
    )
    result = execute("sort", dates)
    expected = dates.sort_values().reset_index(drop=True)
    pd.testing.assert_series_equal(result.reset_index(drop=True), expected, check_names=False)


def test_datetime_min_max():
    # See test_datetime_sort_matches_pandas for why ns resolution is pinned
    # explicitly.
    from metaldf._engine import execute
    dates = pd.Series(
        pd.to_datetime(["2023-03-15", "2021-01-01", "2024-12-31"]).astype("datetime64[ns]")
    )
    result_min = execute("min", dates)
    result_max = execute("max", dates)
    assert result_min == dates.min().value
    assert result_max == dates.max().value


def test_datetime_in_supported_dtypes():
    from metaldf._engine._metal import _SUPPORTED_DTYPES, _DATETIME_DTYPE, _TIMEDELTA_DTYPE
    assert _DATETIME_DTYPE in _SUPPORTED_DTYPES
    assert _TIMEDELTA_DTYPE in _SUPPORTED_DTYPES
