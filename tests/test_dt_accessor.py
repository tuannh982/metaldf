import numpy as np
import pandas as pd
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")

# NOTE(DT-1 merge): this worktree branched before Task 1's `DType::Datetime` /
# `MetalSeries.from_numpy_datetime` landed, so tests build the GPU series via
# `MetalSeries.from_numpy_datetime` directly on the int64 nanosecond view (see the
# TODO in `rust/src/kernels/datetime.rs` and `ProxyDatetimeAccessor` in
# `src/metaldf/_wrappers.py`). Once merged onto a branch carrying Task 1,
# `_to_metal_series` below should switch to `MetalSeries.from_numpy_datetime`.
#
# NOTE(zero-copy lifetime): `MetalSeries.from_numpy_datetime` (like every
# `from_numpy_*` constructor -- see `SharedBuffer::from_numpy_inner` in
# `rust/src/buffer.rs`) wraps the numpy array's memory directly
# (`new_buffer_with_bytes_no_copy`) WITHOUT retaining a reference to keep it
# alive -- this is a pre-existing, cross-cutting property of the whole
# zero-copy buffer design (not something introduced by the datetime
# kernels), and it means the caller must keep the source numpy array alive
# for as long as the `MetalSeries` is used, or its backing memory can be
# freed and reused by an unrelated later allocation (observed directly:
# GPU reads then silently return whatever the freed memory now holds,
# rather than erroring). `_to_metal_series` therefore returns `(ms, ns)`
# rather than just `ms` -- every call site below keeps the `ns` half
# alive (even if never read again) for as long as any kernel dispatch on
# `ms` might still happen.


def _to_metal_series(dates: pd.Series) -> tuple:
    ns = dates.values.astype("datetime64[ns]").view(np.int64)
    return metaldf_engine.MetalSeries.from_numpy_datetime(ns), ns


@pytest.fixture
def sample_dates():
    return pd.Series(pd.to_datetime([
        "2020-01-15 10:30:45",
        "2023-06-20 14:15:00",
        "2024-12-31 23:59:59",
        "2021-02-28 00:00:00",
        "2000-03-01 12:00:00",
    ]))


def test_dt_year(sample_dates):
    ms, _ns = _to_metal_series(sample_dates)
    result = metaldf_engine.metal_dt_year(ms)
    expected = sample_dates.dt.year.values
    np.testing.assert_array_equal(result.to_numpy(), expected)


def test_dt_month(sample_dates):
    ms, _ns = _to_metal_series(sample_dates)
    result = metaldf_engine.metal_dt_month(ms)
    expected = sample_dates.dt.month.values
    np.testing.assert_array_equal(result.to_numpy(), expected)


def test_dt_day(sample_dates):
    ms, _ns = _to_metal_series(sample_dates)
    result = metaldf_engine.metal_dt_day(ms)
    expected = sample_dates.dt.day.values
    np.testing.assert_array_equal(result.to_numpy(), expected)


def test_dt_hour(sample_dates):
    ms, _ns = _to_metal_series(sample_dates)
    result = metaldf_engine.metal_dt_hour(ms)
    expected = sample_dates.dt.hour.values
    np.testing.assert_array_equal(result.to_numpy(), expected)


def test_dt_minute(sample_dates):
    ms, _ns = _to_metal_series(sample_dates)
    result = metaldf_engine.metal_dt_minute(ms)
    expected = sample_dates.dt.minute.values
    np.testing.assert_array_equal(result.to_numpy(), expected)


def test_dt_second(sample_dates):
    ms, _ns = _to_metal_series(sample_dates)
    result = metaldf_engine.metal_dt_second(ms)
    expected = sample_dates.dt.second.values
    np.testing.assert_array_equal(result.to_numpy(), expected)


def test_dt_dayofweek(sample_dates):
    ms, _ns = _to_metal_series(sample_dates)
    result = metaldf_engine.metal_dt_dayofweek(ms)
    expected = sample_dates.dt.dayofweek.values
    np.testing.assert_array_equal(result.to_numpy(), expected)


def test_dt_leap_year():
    dates = pd.Series(pd.to_datetime(["2024-02-29", "2023-03-01", "2000-02-29", "1900-03-01"]))
    ms, _ns = _to_metal_series(dates)
    days = metaldf_engine.metal_dt_day(ms).to_numpy()
    np.testing.assert_array_equal(days, dates.dt.day.values)
    months = metaldf_engine.metal_dt_month(ms).to_numpy()
    np.testing.assert_array_equal(months, dates.dt.month.values)
    years = metaldf_engine.metal_dt_year(ms).to_numpy()
    np.testing.assert_array_equal(years, dates.dt.year.values)


def test_dt_pre_epoch_dates():
    """Dates before 1970-01-01 exercise negative nanosecond counts, where
    naive truncating division (rather than floor division) gets every
    component wrong. Covers a date, a midday time, and a time in the last
    second before midnight UTC on the same pre-epoch day."""
    dates = pd.Series(pd.to_datetime([
        "1969-12-31 23:59:59",
        "1969-01-01 00:00:01",
        "1900-01-01 06:30:15",
        "1969-07-20 20:17:40",
    ]))
    ms, _ns = _to_metal_series(dates)

    np.testing.assert_array_equal(metaldf_engine.metal_dt_year(ms).to_numpy(), dates.dt.year.values)
    np.testing.assert_array_equal(metaldf_engine.metal_dt_month(ms).to_numpy(), dates.dt.month.values)
    np.testing.assert_array_equal(metaldf_engine.metal_dt_day(ms).to_numpy(), dates.dt.day.values)
    np.testing.assert_array_equal(metaldf_engine.metal_dt_hour(ms).to_numpy(), dates.dt.hour.values)
    np.testing.assert_array_equal(metaldf_engine.metal_dt_minute(ms).to_numpy(), dates.dt.minute.values)
    np.testing.assert_array_equal(metaldf_engine.metal_dt_second(ms).to_numpy(), dates.dt.second.values)
    np.testing.assert_array_equal(metaldf_engine.metal_dt_dayofweek(ms).to_numpy(), dates.dt.dayofweek.values)


def test_dt_epoch_boundary():
    """Nanosecond-exact epoch instant plus the instant immediately before
    and after it -- a boundary case for floor_div/floor_mod at zero.

    Built directly from int64 nanosecond offsets (rather than
    ``pd.to_datetime`` string parsing) since pandas' format inference
    chokes on mixing fractional-nanosecond and whole-second timestamp
    strings in the same list.
    """
    ns_values = np.array([0, -1, 1], dtype=np.int64)
    dates = pd.Series(ns_values.view("datetime64[ns]"))
    ms = metaldf_engine.MetalSeries.from_numpy_datetime(ns_values)
    np.testing.assert_array_equal(metaldf_engine.metal_dt_day(ms).to_numpy(), dates.dt.day.values)
    np.testing.assert_array_equal(metaldf_engine.metal_dt_hour(ms).to_numpy(), dates.dt.hour.values)
    np.testing.assert_array_equal(metaldf_engine.metal_dt_minute(ms).to_numpy(), dates.dt.minute.values)
    np.testing.assert_array_equal(metaldf_engine.metal_dt_second(ms).to_numpy(), dates.dt.second.values)


def test_dt_dayofweek_known_days():
    # 1970-01-01 was a Thursday (pandas dayofweek: Monday=0 .. Sunday=6).
    dates = pd.Series(pd.to_datetime([
        "1970-01-01",  # Thursday -> 3
        "1970-01-02",  # Friday   -> 4
        "1970-01-03",  # Saturday -> 5
        "1970-01-04",  # Sunday   -> 6
        "1970-01-05",  # Monday   -> 0
    ]))
    ms, _ns = _to_metal_series(dates)
    result = metaldf_engine.metal_dt_dayofweek(ms).to_numpy()
    np.testing.assert_array_equal(result, [3, 4, 5, 6, 0])
    np.testing.assert_array_equal(result, dates.dt.dayofweek.values)


def test_dt_extract_rejects_non_int64():
    """dispatch_dt_extract should reject non-Int64 series (see TODO in
    rust/src/kernels/datetime.rs about tightening this to DType::Datetime
    once Task 1 lands)."""
    ms = metaldf_engine.MetalSeries.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    with pytest.raises(Exception):
        metaldf_engine.metal_dt_year(ms)


def test_proxy_dt_accessor():
    from metaldf._wrappers import ProxySeries
    dates = pd.Series(pd.to_datetime(["2023-06-15", "2024-01-01", "2020-12-31"]))
    ps = ProxySeries(_pandas_obj=dates)
    years = ps.dt.year
    expected = dates.dt.year
    result = years.to_pandas() if hasattr(years, "to_pandas") else years
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_proxy_dt_accessor_all_components():
    from metaldf._wrappers import ProxySeries
    dates = pd.Series(pd.to_datetime([
        "2023-06-15 08:09:10", "2024-01-01 00:00:00", "2020-12-31 23:59:59",
    ]))
    ps = ProxySeries(_pandas_obj=dates)
    for component in ("year", "month", "day", "hour", "minute", "second", "dayofweek"):
        result = getattr(ps.dt, component)
        result = result.to_pandas() if hasattr(result, "to_pandas") else result
        expected = getattr(dates.dt, component)
        pd.testing.assert_series_equal(result, expected, check_names=False)


def test_proxy_dt_accessor_non_datetime_falls_back():
    """A non-datetime series' `.dt` should behave like plain pandas (raise
    the same AttributeError pandas itself raises), not route through
    ProxyDatetimeAccessor."""
    from metaldf._wrappers import ProxySeries
    ints = pd.Series([1, 2, 3])
    ps = ProxySeries(_pandas_obj=ints)
    with pytest.raises(AttributeError):
        ps.dt


def test_large_datetime_extraction():
    dates = pd.date_range("2000-01-01", periods=100_000, freq="h")
    dates_series = pd.Series(dates)
    ms, _ns = _to_metal_series(dates_series)
    years = metaldf_engine.metal_dt_year(ms).to_numpy()
    expected = dates_series.dt.year.values
    np.testing.assert_array_equal(years, expected)

    hours = metaldf_engine.metal_dt_hour(ms).to_numpy()
    np.testing.assert_array_equal(hours, dates_series.dt.hour.values)
