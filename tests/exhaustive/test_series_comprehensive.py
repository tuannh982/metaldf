"""Comprehensive Series proxy tests with real arguments.

Every test compares proxy output to real pandas output to prove 100% transparency.
"""

from __future__ import annotations

import pandas as pd
import pytest

from metaldf._wrappers import ProxySeries


@pytest.fixture
def real_s():
    return pd.Series([1, 2, 3, 4, 5], name="test")


@pytest.fixture
def proxy_s(real_s):
    return ProxySeries(_pandas_obj=real_s.copy())


@pytest.fixture
def real_s2():
    return pd.Series([10, 20, 30, 40, 50], name="other")


@pytest.fixture
def proxy_s2(real_s2):
    return ProxySeries(_pandas_obj=real_s2.copy())


@pytest.fixture
def real_s_str():
    return pd.Series(["hello", "world", "foo"])


@pytest.fixture
def proxy_s_str(real_s_str):
    return ProxySeries(_pandas_obj=real_s_str.copy())


@pytest.fixture
def real_s_dt():
    return pd.Series(pd.date_range("2020-01-01", periods=3))


@pytest.fixture
def proxy_s_dt(real_s_dt):
    return ProxySeries(_pandas_obj=real_s_dt.copy())


@pytest.fixture
def real_s_cat():
    return pd.Series(pd.Categorical(["a", "b", "a"]))


@pytest.fixture
def proxy_s_cat(real_s_cat):
    return ProxySeries(_pandas_obj=real_s_cat.copy())


def _assert_equal(proxy_result, real_result, msg=""):
    """Compare proxy output to real pandas output."""
    if hasattr(proxy_result, "to_pandas"):
        proxy_result = proxy_result.to_pandas()
    if isinstance(proxy_result, pd.DataFrame) and isinstance(real_result, pd.DataFrame):
        pd.testing.assert_frame_equal(proxy_result, real_result)
    elif isinstance(proxy_result, pd.Series) and isinstance(real_result, pd.Series):
        pd.testing.assert_series_equal(proxy_result, real_result)
    elif hasattr(proxy_result, "tolist") and hasattr(real_result, "tolist"):
        assert list(proxy_result) == list(real_result), msg
    else:
        assert proxy_result == real_result or (pd.isna(proxy_result) and pd.isna(real_result)), msg


class TestSeriesCreation:
    def test_isinstance(self, proxy_s, real_s):
        assert isinstance(proxy_s, pd.Series)

    def test_shape(self, proxy_s, real_s):
        assert proxy_s.shape == real_s.shape

    def test_index(self, proxy_s, real_s):
        assert list(proxy_s.index) == list(real_s.index)

    def test_values(self, proxy_s, real_s):
        assert proxy_s.values.tolist() == real_s.values.tolist()

    def test_name(self, proxy_s, real_s):
        assert proxy_s.name == real_s.name

    def test_len(self, proxy_s, real_s):
        assert len(proxy_s) == len(real_s)

    def test_dtype(self, proxy_s, real_s):
        assert proxy_s.dtype == real_s.dtype


class TestSeriesIndexing:
    def test_getitem_scalar(self, proxy_s, real_s):
        assert proxy_s[0] == real_s[0]

    def test_getitem_slice(self, proxy_s, real_s):
        _assert_equal(proxy_s[1:3], real_s[1:3])

    def test_getitem_bool_mask(self, proxy_s, real_s):
        mask = real_s > 2
        _assert_equal(proxy_s[mask], real_s[mask])

    def test_loc(self, proxy_s, real_s):
        assert proxy_s.loc[0] == real_s.loc[0]

    def test_iloc(self, proxy_s, real_s):
        assert proxy_s.iloc[0] == real_s.iloc[0]

    def test_at(self, proxy_s, real_s):
        assert proxy_s.at[0] == real_s.at[0]

    def test_iat(self, proxy_s, real_s):
        assert proxy_s.iat[0] == real_s.iat[0]

    def test_setitem_scalar(self, proxy_s, real_s):
        proxy_s[0] = 99
        assert proxy_s[0] == 99

    def test_iter(self, proxy_s, real_s):
        assert list(proxy_s) == list(real_s)

    def test_items(self, proxy_s, real_s):
        for (pk, pv), (rk, rv) in zip(
            proxy_s.items(), real_s.items(), strict=True
        ):
            assert pk == rk
            assert pv == rv


class TestSeriesArithmetic:
    def test_add_scalar(self, proxy_s, real_s):
        _assert_equal(proxy_s + 1, real_s + 1)

    def test_sub_scalar(self, proxy_s, real_s):
        _assert_equal(proxy_s - 1, real_s - 1)

    def test_mul_scalar(self, proxy_s, real_s):
        _assert_equal(proxy_s * 2, real_s * 2)

    def test_div_scalar(self, proxy_s, real_s):
        _assert_equal(proxy_s / 2, real_s / 2)

    def test_floordiv_scalar(self, proxy_s, real_s):
        _assert_equal(proxy_s // 2, real_s // 2)

    def test_mod_scalar(self, proxy_s, real_s):
        _assert_equal(proxy_s % 3, real_s % 3)

    def test_pow_scalar(self, proxy_s, real_s):
        _assert_equal(proxy_s ** 2, real_s ** 2)

    def test_add_series(self, proxy_s, real_s, proxy_s2, real_s2):
        _assert_equal(proxy_s + proxy_s2, real_s + real_s2)

    def test_sub_series(self, proxy_s, real_s, proxy_s2, real_s2):
        _assert_equal(proxy_s - proxy_s2, real_s - real_s2)

    def test_mul_series(self, proxy_s, real_s, proxy_s2, real_s2):
        _assert_equal(proxy_s * proxy_s2, real_s * real_s2)

    def test_neg(self, proxy_s, real_s):
        _assert_equal(-proxy_s, -real_s)

    def test_abs(self, proxy_s, real_s):
        _assert_equal(abs(proxy_s), abs(real_s))


class TestSeriesComparison:
    def test_eq_scalar(self, proxy_s, real_s):
        _assert_equal(proxy_s == 2, real_s == 2)

    def test_ne_scalar(self, proxy_s, real_s):
        _assert_equal(proxy_s != 2, real_s != 2)

    def test_lt_scalar(self, proxy_s, real_s):
        _assert_equal(proxy_s < 3, real_s < 3)

    def test_gt_scalar(self, proxy_s, real_s):
        _assert_equal(proxy_s > 2, real_s > 2)

    def test_le_scalar(self, proxy_s, real_s):
        _assert_equal(proxy_s <= 3, real_s <= 3)

    def test_ge_scalar(self, proxy_s, real_s):
        _assert_equal(proxy_s >= 2, real_s >= 2)


class TestSeriesAggregation:
    def test_sum(self, proxy_s, real_s):
        assert proxy_s.sum() == real_s.sum()

    def test_mean(self, proxy_s, real_s):
        assert proxy_s.mean() == real_s.mean()

    def test_min(self, proxy_s, real_s):
        assert proxy_s.min() == real_s.min()

    def test_max(self, proxy_s, real_s):
        assert proxy_s.max() == real_s.max()

    def test_std(self, proxy_s, real_s):
        assert proxy_s.std() == real_s.std()

    def test_var(self, proxy_s, real_s):
        assert proxy_s.var() == real_s.var()

    def test_count(self, proxy_s, real_s):
        assert proxy_s.count() == real_s.count()

    def test_describe(self, proxy_s, real_s):
        _assert_equal(proxy_s.describe(), real_s.describe())

    def test_nunique(self, proxy_s, real_s):
        assert proxy_s.nunique() == real_s.nunique()

    def test_quantile(self, proxy_s, real_s):
        assert proxy_s.quantile() == real_s.quantile()

    def test_rank(self, proxy_s, real_s):
        _assert_equal(proxy_s.rank(), real_s.rank())


class TestSeriesTransform:
    def test_apply(self, proxy_s, real_s):
        _assert_equal(proxy_s.apply(lambda x: x * 2), real_s.apply(lambda x: x * 2))

    def test_transform(self, proxy_s, real_s):
        _assert_equal(proxy_s.transform(lambda x: x * 2), real_s.transform(lambda x: x * 2))

    def test_map(self, proxy_s, real_s):
        _assert_equal(proxy_s.map(lambda x: x * 2), real_s.map(lambda x: x * 2))

    def test_pipe(self, proxy_s, real_s):
        _assert_equal(proxy_s.pipe(lambda x: x + 1), real_s.pipe(lambda x: x + 1))


class TestSeriesCleaning:
    def test_dropna(self, proxy_s, real_s):
        proxy_s.loc[0] = None
        real_s.loc[0] = None
        _assert_equal(proxy_s.dropna(), real_s.dropna())

    def test_fillna(self, proxy_s, real_s):
        proxy_s.loc[0] = None
        real_s.loc[0] = None
        _assert_equal(proxy_s.fillna(0), real_s.fillna(0))

    def test_replace(self, proxy_s, real_s):
        _assert_equal(proxy_s.replace(1, 100), real_s.replace(1, 100))

    def test_clip(self, proxy_s, real_s):
        _assert_equal(proxy_s.clip(lower=2, upper=4), real_s.clip(lower=2, upper=4))

    def test_round(self, proxy_s, real_s):
        _assert_equal(proxy_s.round(0), real_s.round(0))


class TestSeriesSelection:
    def test_head(self, proxy_s, real_s):
        _assert_equal(proxy_s.head(2), real_s.head(2))

    def test_tail(self, proxy_s, real_s):
        _assert_equal(proxy_s.tail(2), real_s.tail(2))

    def test_take(self, proxy_s, real_s):
        _assert_equal(proxy_s.take([0, 2]), real_s.take([0, 2]))

    def test_sample(self, proxy_s, real_s):
        _assert_equal(proxy_s.sample(n=2, random_state=42), real_s.sample(n=2, random_state=42))

    def test_isin(self, proxy_s, real_s):
        _assert_equal(proxy_s.isin([1, 2, 3]), real_s.isin([1, 2, 3]))

    def test_between(self, proxy_s, real_s):
        _assert_equal(proxy_s.between(2, 4), real_s.between(2, 4))

    def test_unique(self, proxy_s, real_s):
        assert set(proxy_s.unique()) == set(real_s.unique())

    def test_value_counts(self, proxy_s, real_s):
        _assert_equal(proxy_s.value_counts(), real_s.value_counts())


class TestSeriesSorting:
    def test_sort_values(self, proxy_s, real_s):
        _assert_equal(proxy_s.sort_values(), real_s.sort_values())

    def test_sort_index(self, proxy_s, real_s):
        idx = [2, 0, 1, 4, 3]
        proxy_s.index = idx
        real_s.index = idx
        _assert_equal(proxy_s.sort_index(), real_s.sort_index())

    def test_shift(self, proxy_s, real_s):
        _assert_equal(proxy_s.shift(1), real_s.shift(1))

    def test_diff(self, proxy_s, real_s):
        _assert_equal(proxy_s.diff(), real_s.diff())

    def test_pct_change(self, proxy_s, real_s):
        _assert_equal(proxy_s.pct_change(), real_s.pct_change())


class TestSeriesCumulative:
    def test_cumsum(self, proxy_s, real_s):
        _assert_equal(proxy_s.cumsum(), real_s.cumsum())

    def test_cumprod(self, proxy_s, real_s):
        _assert_equal(proxy_s.cumprod(), real_s.cumprod())

    def test_cummax(self, proxy_s, real_s):
        _assert_equal(proxy_s.cummax(), real_s.cummax())

    def test_cummin(self, proxy_s, real_s):
        _assert_equal(proxy_s.cummin(), real_s.cummin())


class TestSeriesProperties:
    def test_empty(self, proxy_s, real_s):
        assert proxy_s.empty == real_s.empty

    def test_ndim(self, proxy_s, real_s):
        assert proxy_s.ndim == real_s.ndim

    def test_size(self, proxy_s, real_s):
        assert proxy_s.size == real_s.size

    def test_isna(self, proxy_s, real_s):
        _assert_equal(proxy_s.isna(), real_s.isna())

    def test_notna(self, proxy_s, real_s):
        _assert_equal(proxy_s.notna(), real_s.notna())

    def test_duplicated(self, proxy_s, real_s):
        _assert_equal(proxy_s.duplicated(), real_s.duplicated())

    def test_memory_usage(self, proxy_s, real_s):
        assert proxy_s.memory_usage() == real_s.memory_usage()


class TestSeriesCopy:
    def test_copy(self, proxy_s, real_s):
        cp = proxy_s.copy()
        _assert_equal(cp, real_s)

    def test_astype(self, proxy_s, real_s):
        _assert_equal(proxy_s.astype("float64"), real_s.astype("float64"))


class TestSeriesStringAccessor:
    def test_str_len(self, proxy_s_str, real_s_str):
        _assert_equal(proxy_s_str.str.len(), real_s_str.str.len())

    def test_str_upper(self, proxy_s_str, real_s_str):
        _assert_equal(proxy_s_str.str.upper(), real_s_str.str.upper())

    def test_str_contains(self, proxy_s_str, real_s_str):
        _assert_equal(proxy_s_str.str.contains("o"), real_s_str.str.contains("o"))

    def test_str_replace(self, proxy_s_str, real_s_str):
        _assert_equal(proxy_s_str.str.replace("o", "0"), real_s_str.str.replace("o", "0"))

    def test_str_startswith(self, proxy_s_str, real_s_str):
        _assert_equal(proxy_s_str.str.startswith("h"), real_s_str.str.startswith("h"))

    def test_str_split(self, proxy_s_str, real_s_str):
        # split returns a DataFrame
        result = proxy_s_str.str.split("o", expand=True)
        expected = real_s_str.str.split("o", expand=True)
        _assert_equal(result, expected)


class TestSeriesDatetimeAccessor:
    def test_dt_year(self, proxy_s_dt, real_s_dt):
        _assert_equal(proxy_s_dt.dt.year, real_s_dt.dt.year)

    def test_dt_month(self, proxy_s_dt, real_s_dt):
        _assert_equal(proxy_s_dt.dt.month, real_s_dt.dt.month)

    def test_dt_day(self, proxy_s_dt, real_s_dt):
        _assert_equal(proxy_s_dt.dt.day, real_s_dt.dt.day)

    def test_dt_weekday(self, proxy_s_dt, real_s_dt):
        _assert_equal(proxy_s_dt.dt.weekday, real_s_dt.dt.weekday)

    def test_dt_is_month_start(self, proxy_s_dt, real_s_dt):
        _assert_equal(proxy_s_dt.dt.is_month_start, real_s_dt.dt.is_month_start)


class TestSeriesCategoricalAccessor:
    def test_cat_categories(self, proxy_s_cat, real_s_cat):
        assert list(proxy_s_cat.cat.categories) == list(real_s_cat.cat.categories)

    def test_cat_codes(self, proxy_s_cat, real_s_cat):
        _assert_equal(proxy_s_cat.cat.codes, real_s_cat.cat.codes)

    def test_cat_rename_categories(self, proxy_s_cat, real_s_cat):
        _assert_equal(
            proxy_s_cat.cat.rename_categories(["X", "Y"]),
            real_s_cat.cat.rename_categories(["X", "Y"]),
        )


class TestSeriesGroupBy:
    def test_groupby_sum(self):
        real = pd.Series([1, 2, 3, 4], index=["a", "a", "b", "b"])
        proxy = ProxySeries(_pandas_obj=real.copy())
        _assert_equal(
            proxy.groupby(level=0).sum(),
            real.groupby(level=0).sum(),
        )

    def test_groupby_mean(self):
        real = pd.Series([1, 2, 3, 4], index=["a", "a", "b", "b"])
        proxy = ProxySeries(_pandas_obj=real.copy())
        _assert_equal(
            proxy.groupby(level=0).mean(),
            real.groupby(level=0).mean(),
        )


class TestSeriesRolling:
    def test_rolling_sum(self):
        real = pd.Series([1, 2, 3, 4, 5])
        proxy = ProxySeries(_pandas_obj=real.copy())
        _assert_equal(
            proxy.rolling(window=2).sum(),
            real.rolling(window=2).sum(),
        )

    def test_rolling_mean(self):
        real = pd.Series([1, 2, 3, 4, 5])
        proxy = ProxySeries(_pandas_obj=real.copy())
        _assert_equal(
            proxy.rolling(window=2).mean(),
            real.rolling(window=2).mean(),
        )

    def test_rolling_std(self):
        real = pd.Series([1, 2, 3, 4, 5])
        proxy = ProxySeries(_pandas_obj=real.copy())
        _assert_equal(
            proxy.rolling(window=2).std(),
            real.rolling(window=2).std(),
        )


class TestSeriesCorr:
    def test_corr(self):
        real1 = pd.Series([1, 2, 3, 4, 5])
        real2 = pd.Series([5, 4, 3, 2, 1])
        proxy1 = ProxySeries(_pandas_obj=real1.copy())
        proxy2 = ProxySeries(_pandas_obj=real2.copy())
        assert proxy1.corr(proxy2) == real1.corr(real2)

    def test_cov(self):
        real1 = pd.Series([1, 2, 3, 4, 5])
        real2 = pd.Series([5, 4, 3, 2, 1])
        proxy1 = ProxySeries(_pandas_obj=real1.copy())
        proxy2 = ProxySeries(_pandas_obj=real2.copy())
        assert proxy1.cov(proxy2) == real1.cov(real2)


class TestSeriesIO:
    def test_to_list(self, proxy_s, real_s):
        assert proxy_s.tolist() == real_s.tolist()

    def test_to_dict(self, proxy_s, real_s):
        assert proxy_s.to_dict() == real_s.to_dict()

    def test_to_numpy(self, proxy_s, real_s):
        assert proxy_s.to_numpy().tolist() == real_s.to_numpy().tolist()
