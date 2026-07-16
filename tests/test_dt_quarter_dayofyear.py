"""Tests for GPU datetime quarter and dayofyear extraction."""

import numpy as np
import pandas as pd
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")


def _to_metal_series(dates: pd.Series) -> tuple:
    ns = dates.values.astype("datetime64[ns]").view(np.int64)
    return metaldf_engine.MetalSeries.from_numpy_datetime(ns), ns


@pytest.fixture
def sample_dates():
    return pd.Series(pd.to_datetime([
        "2020-01-01 00:00:00",
        "2020-03-31 12:00:00",
        "2020-06-15 06:30:00",
        "2020-09-22 18:45:00",
        "2020-12-31 23:59:59",
    ]))


class TestQuarter:
    def test_all_quarters(self, sample_dates):
        ms, _ns = _to_metal_series(sample_dates)
        result = metaldf_engine.metal_dt_quarter(ms)
        expected = sample_dates.dt.quarter.values
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_quarter_boundaries(self):
        dates = pd.Series(pd.to_datetime([
            "2020-01-01", "2020-03-31",
            "2020-04-01", "2020-06-30",
            "2020-07-01", "2020-09-30",
            "2020-10-01", "2020-12-31",
        ]))
        ms, _ns = _to_metal_series(dates)
        result = metaldf_engine.metal_dt_quarter(ms)
        expected = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int32)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_pre_epoch(self):
        dates = pd.Series(pd.to_datetime(["1965-07-15", "1900-01-01"]))
        ms, _ns = _to_metal_series(dates)
        result = metaldf_engine.metal_dt_quarter(ms)
        expected = dates.dt.quarter.values
        np.testing.assert_array_equal(result.to_numpy(), expected)


class TestDayOfYear:
    def test_basic(self, sample_dates):
        ms, _ns = _to_metal_series(sample_dates)
        result = metaldf_engine.metal_dt_dayofyear(ms)
        expected = sample_dates.dt.dayofyear.values
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_jan1_is_day1(self):
        dates = pd.Series(pd.to_datetime(["2020-01-01", "2021-01-01"]))
        ms, _ns = _to_metal_series(dates)
        result = metaldf_engine.metal_dt_dayofyear(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.array([1, 1], dtype=np.int32))

    def test_dec31_leap_year(self):
        dates = pd.Series(pd.to_datetime(["2020-12-31"]))
        ms, _ns = _to_metal_series(dates)
        result = metaldf_engine.metal_dt_dayofyear(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.array([366], dtype=np.int32))

    def test_dec31_non_leap_year(self):
        dates = pd.Series(pd.to_datetime(["2021-12-31"]))
        ms, _ns = _to_metal_series(dates)
        result = metaldf_engine.metal_dt_dayofyear(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.array([365], dtype=np.int32))

    def test_feb29_leap_year(self):
        dates = pd.Series(pd.to_datetime(["2020-02-29"]))
        ms, _ns = _to_metal_series(dates)
        result = metaldf_engine.metal_dt_dayofyear(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.array([60], dtype=np.int32))

    def test_pre_epoch(self):
        dates = pd.Series(pd.to_datetime(["1965-07-15", "1900-03-01"]))
        ms, _ns = _to_metal_series(dates)
        result = metaldf_engine.metal_dt_dayofyear(ms)
        expected = dates.dt.dayofyear.values
        np.testing.assert_array_equal(result.to_numpy(), expected)


class TestProxyAccessor:
    def test_quarter_via_proxy(self):
        import metaldf
        metaldf.install()
        try:
            dates = pd.Series(pd.to_datetime(["2020-01-15", "2020-04-15", "2020-07-15", "2020-10-15"]))
            result = dates.dt.quarter
            expected = pd.Series([1, 2, 3, 4])
            np.testing.assert_array_equal(result.values, expected.values)
        finally:
            metaldf.uninstall()

    def test_dayofyear_via_proxy(self):
        import metaldf
        metaldf.install()
        try:
            dates = pd.Series(pd.to_datetime(["2020-01-01", "2020-12-31"]))
            result = dates.dt.dayofyear
            expected = pd.Series([1, 366])
            np.testing.assert_array_equal(result.values, expected.values)
        finally:
            metaldf.uninstall()
