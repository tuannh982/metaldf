"""Comprehensive DataFrame proxy tests with real arguments.

Every test compares proxy output to real pandas output to prove 100% transparency.
"""

from __future__ import annotations

import pandas as pd
import pytest

from metaldf._wrappers import ProxyDataFrame


# Fresh fixtures per test to avoid mutation issues
@pytest.fixture
def real_df():
    return pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        "c": [7, 8, 9, 10, 11],
    })


@pytest.fixture
def proxy_df(real_df):
    return ProxyDataFrame(_pandas_obj=real_df.copy())


@pytest.fixture
def real_df2():
    return pd.DataFrame({
        "a": [10, 20, 30, 40, 50],
        "b": [100.0, 200.0, 300.0, 400.0, 500.0],
    })


@pytest.fixture
def proxy_df2(real_df2):
    return ProxyDataFrame(_pandas_obj=real_df2.copy())


@pytest.fixture
def real_df_cat():
    """DataFrame with categorical column for groupby tests."""
    return pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        "grp": ["x", "y", "z", "x", "y"],
    })


@pytest.fixture
def proxy_df_cat(real_df_cat):
    return ProxyDataFrame(_pandas_obj=real_df_cat.copy())


def _assert_equal(proxy_result, real_result, msg=""):
    """Compare proxy output to real pandas output."""
    # Unwrap proxy if applicable
    if hasattr(proxy_result, "to_pandas"):
        proxy_result = proxy_result.to_pandas()
    # Compare based on type
    if isinstance(proxy_result, pd.DataFrame) and isinstance(real_result, pd.DataFrame):
        pd.testing.assert_frame_equal(proxy_result, real_result)
    elif isinstance(proxy_result, pd.Series) and isinstance(real_result, pd.Series):
        pd.testing.assert_series_equal(proxy_result, real_result)
    elif hasattr(proxy_result, "tolist") and hasattr(real_result, "tolist"):
        assert list(proxy_result) == list(real_result), msg
    else:
        assert proxy_result == real_result or (pd.isna(proxy_result) and pd.isna(real_result)), msg


class TestDataFrameCreation:
    def test_isinstance(self, proxy_df, real_df):
        assert isinstance(proxy_df, pd.DataFrame)

    def test_shape(self, proxy_df, real_df):
        assert proxy_df.shape == real_df.shape

    def test_columns(self, proxy_df, real_df):
        assert list(proxy_df.columns) == list(real_df.columns)

    def test_index(self, proxy_df, real_df):
        assert list(proxy_df.index) == list(real_df.index)

    def test_dtypes(self, proxy_df, real_df):
        assert list(proxy_df.dtypes) == list(real_df.dtypes)

    def test_values(self, proxy_df, real_df):
        assert proxy_df.values.tolist() == real_df.values.tolist()

    def test_len(self, proxy_df, real_df):
        assert len(proxy_df) == len(real_df)


class TestDataFrameIndexing:
    def test_getitem_column(self, proxy_df, real_df):
        _assert_equal(proxy_df["a"], real_df["a"])

    def test_getitem_multiple_columns(self, proxy_df, real_df):
        _assert_equal(proxy_df[["a", "b"]], real_df[["a", "b"]])

    def test_getitem_slice(self, proxy_df, real_df):
        _assert_equal(proxy_df[1:3], real_df[1:3])

    def test_getitem_bool_mask(self, proxy_df, real_df):
        mask = real_df["a"] > 2
        _assert_equal(proxy_df[mask], real_df[mask])

    def test_loc_row(self, proxy_df, real_df):
        _assert_equal(proxy_df.loc[0], real_df.loc[0])

    def test_loc_rows(self, proxy_df, real_df):
        _assert_equal(proxy_df.loc[1:3], real_df.loc[1:3])

    def test_loc_column(self, proxy_df, real_df):
        _assert_equal(proxy_df.loc[:, "a"], real_df.loc[:, "a"])

    def test_iloc_row(self, proxy_df, real_df):
        _assert_equal(proxy_df.iloc[0], real_df.iloc[0])

    def test_iloc_rows(self, proxy_df, real_df):
        _assert_equal(proxy_df.iloc[1:3], real_df.iloc[1:3])

    def test_iloc_cell(self, proxy_df, real_df):
        assert proxy_df.iloc[0, 0] == real_df.iloc[0, 0]

    def test_at(self, proxy_df, real_df):
        assert proxy_df.at[0, "a"] == real_df.at[0, "a"]

    def test_iat(self, proxy_df, real_df):
        assert proxy_df.iat[0, 0] == real_df.iat[0, 0]

    def test_setitem_column(self, proxy_df, real_df):
        # Mutating the proxy modifies its internal pandas state
        proxy_df["d"] = [10, 20, 30, 40, 50]
        assert "d" in proxy_df.columns
        assert list(proxy_df["d"]) == [10, 20, 30, 40, 50]

    def test_setitem_scalar(self, proxy_df, real_df):
        proxy_df["a"] = 99
        assert proxy_df["a"].tolist() == [99, 99, 99, 99, 99]

    def test_contains(self, proxy_df, real_df):
        assert ("a" in proxy_df) == ("a" in real_df)
        assert ("z" in proxy_df) == ("z" in real_df)

    def test_iter(self, proxy_df, real_df):
        assert list(proxy_df) == list(real_df)

    def test_items(self, proxy_df, real_df):
        for (pk, pv), (rk, rv) in zip(
            proxy_df.items(), real_df.items(), strict=True
        ):
            assert pk == rk
            _assert_equal(pv, rv)


class TestDataFrameArithmetic:
    def test_add_scalar(self, proxy_df, real_df):
        _assert_equal(proxy_df + 1, real_df + 1)

    def test_sub_scalar(self, proxy_df, real_df):
        _assert_equal(proxy_df - 1, real_df - 1)

    def test_mul_scalar(self, proxy_df, real_df):
        _assert_equal(proxy_df * 2, real_df * 2)

    def test_div_scalar(self, proxy_df, real_df):
        _assert_equal(proxy_df / 2, real_df / 2)

    def test_floordiv_scalar(self, proxy_df, real_df):
        _assert_equal(proxy_df // 2, real_df // 2)

    def test_mod_scalar(self, proxy_df, real_df):
        _assert_equal(proxy_df % 3, real_df % 3)

    def test_pow_scalar(self, proxy_df, real_df):
        _assert_equal(proxy_df ** 2, real_df ** 2)

    def test_add_dataframe(self, proxy_df, real_df, proxy_df2, real_df2):
        _assert_equal(proxy_df + proxy_df2, real_df + real_df2)

    def test_sub_dataframe(self, proxy_df, real_df, proxy_df2, real_df2):
        _assert_equal(proxy_df - proxy_df2, real_df - real_df2)

    def test_mul_dataframe(self, proxy_df, real_df, proxy_df2, real_df2):
        _assert_equal(proxy_df * proxy_df2, real_df * real_df2)

    def test_neg(self, proxy_df, real_df):
        _assert_equal(-proxy_df, -real_df)

    def test_abs(self, proxy_df, real_df):
        _assert_equal(abs(proxy_df), abs(real_df))


class TestDataFrameComparison:
    def test_eq_scalar(self, proxy_df, real_df):
        _assert_equal(proxy_df == 2, real_df == 2)

    def test_ne_scalar(self, proxy_df, real_df):
        _assert_equal(proxy_df != 2, real_df != 2)

    def test_lt_scalar(self, proxy_df, real_df):
        _assert_equal(proxy_df < 3, real_df < 3)

    def test_gt_scalar(self, proxy_df, real_df):
        _assert_equal(proxy_df > 2, real_df > 2)

    def test_le_scalar(self, proxy_df, real_df):
        _assert_equal(proxy_df <= 3, real_df <= 3)

    def test_ge_scalar(self, proxy_df, real_df):
        _assert_equal(proxy_df >= 2, real_df >= 2)


class TestDataFrameAggregation:
    def test_sum(self, proxy_df, real_df):
        _assert_equal(proxy_df.sum(), real_df.sum())

    def test_mean(self, proxy_df, real_df):
        _assert_equal(proxy_df.mean(numeric_only=True), real_df.mean(numeric_only=True))

    def test_min(self, proxy_df, real_df):
        _assert_equal(proxy_df.min(), real_df.min())

    def test_max(self, proxy_df, real_df):
        _assert_equal(proxy_df.max(), real_df.max())

    def test_std(self, proxy_df, real_df):
        _assert_equal(proxy_df.std(), real_df.std())

    def test_var(self, proxy_df, real_df):
        _assert_equal(proxy_df.var(), real_df.var())

    def test_count(self, proxy_df, real_df):
        _assert_equal(proxy_df.count(), real_df.count())

    def test_describe(self, proxy_df, real_df):
        _assert_equal(proxy_df.describe(), real_df.describe())

    def test_nunique(self, proxy_df, real_df):
        _assert_equal(proxy_df.nunique(), real_df.nunique())

    def test_idxmin(self, proxy_df, real_df):
        assert proxy_df.idxmin().tolist() == real_df.idxmin().tolist()

    def test_idxmax(self, proxy_df, real_df):
        assert proxy_df.idxmax().tolist() == real_df.idxmax().tolist()


class TestDataFrameTransform:
    def test_apply(self, proxy_df, real_df):
        _assert_equal(proxy_df.apply(lambda x: x + 1), real_df.apply(lambda x: x + 1))

    def test_transform(self, proxy_df, real_df):
        _assert_equal(proxy_df.transform(lambda x: x + 1), real_df.transform(lambda x: x + 1))

    def test_map(self, proxy_df, real_df):
        _assert_equal(proxy_df["a"].map(lambda x: x * 2), real_df["a"].map(lambda x: x * 2))

    def test_pipe(self, proxy_df, real_df):
        _assert_equal(proxy_df.pipe(lambda x: x + 1), real_df.pipe(lambda x: x + 1))


class TestDataFrameCleaning:
    def test_dropna(self, proxy_df, real_df):
        proxy_df.loc[0, "a"] = None
        real_df.loc[0, "a"] = None
        _assert_equal(proxy_df.dropna(), real_df.dropna())

    def test_fillna(self, proxy_df, real_df):
        proxy_df.loc[0, "a"] = None
        real_df.loc[0, "a"] = None
        _assert_equal(proxy_df.fillna(0), real_df.fillna(0))

    def test_drop_columns(self, proxy_df, real_df):
        _assert_equal(proxy_df.drop(columns=["a"]), real_df.drop(columns=["a"]))

    def test_drop_rows(self, proxy_df, real_df):
        _assert_equal(proxy_df.drop(index=[0, 1]), real_df.drop(index=[0, 1]))

    def test_rename(self, proxy_df, real_df):
        _assert_equal(
            proxy_df.rename(columns={"a": "A"}),
            real_df.rename(columns={"a": "A"}),
        )

    def test_replace(self, proxy_df, real_df):
        _assert_equal(proxy_df.replace(1, 100), real_df.replace(1, 100))

    def test_clip(self, proxy_df, real_df):
        _assert_equal(proxy_df.clip(lower=2, upper=4), real_df.clip(lower=2, upper=4))

    def test_round(self, proxy_df, real_df):
        _assert_equal(proxy_df.round(0), real_df.round(0))


class TestDataFrameSelection:
    def test_head(self, proxy_df, real_df):
        _assert_equal(proxy_df.head(2), real_df.head(2))

    def test_tail(self, proxy_df, real_df):
        _assert_equal(proxy_df.tail(2), real_df.tail(2))

    def test_sample(self, proxy_df, real_df):
        _assert_equal(proxy_df.sample(n=2, random_state=42), real_df.sample(n=2, random_state=42))

    def test_take(self, proxy_df, real_df):
        _assert_equal(proxy_df.take([0, 2]), real_df.take([0, 2]))

    def test_select_dtypes(self, proxy_df, real_df):
        _assert_equal(
            proxy_df.select_dtypes(include="number"),
            real_df.select_dtypes(include="number"),
        )

    def test_filter(self, proxy_df, real_df):
        _assert_equal(proxy_df.filter(items=["a", "b"]), real_df.filter(items=["a", "b"]))


class TestDataFrameSorting:
    def test_sort_values(self, proxy_df, real_df):
        _assert_equal(
            proxy_df.sort_values("a", ascending=False),
            real_df.sort_values("a", ascending=False),
        )

    def test_sort_index(self, proxy_df, real_df):
        idx = [2, 0, 1, 4, 3]
        proxy_df.index = idx
        real_df.index = idx
        _assert_equal(
            proxy_df.sort_index(),
            real_df.sort_index(),
        )


class TestDataFrameMerging:
    def test_merge(self, proxy_df, real_df, proxy_df2, real_df2):
        _assert_equal(
            proxy_df.merge(proxy_df2, on="a", how="inner"),
            real_df.merge(real_df2, on="a", how="inner"),
        )

    def test_join(self, proxy_df, real_df, proxy_df2, real_df2):
        _assert_equal(
            proxy_df.join(proxy_df2, lsuffix="_l", rsuffix="_r"),
            real_df.join(real_df2, lsuffix="_l", rsuffix="_r"),
        )

    def test_concat(self, proxy_df, real_df):
        _assert_equal(
            pd.concat([proxy_df, proxy_df]),
            pd.concat([real_df, real_df]),
        )


class TestDataFrameReshaping:
    def test_transpose(self, proxy_df, real_df):
        _assert_equal(proxy_df.T, real_df.T)

    def test_melt(self, proxy_df, real_df):
        _assert_equal(
            proxy_df.melt(id_vars=["a"]),
            real_df.melt(id_vars=["a"]),
        )

    def test_stack(self, proxy_df, real_df):
        _assert_equal(proxy_df.stack(), real_df.stack())

    def test_reset_index(self, proxy_df, real_df):
        _assert_equal(proxy_df.reset_index(), real_df.reset_index())

    def test_set_index(self, proxy_df, real_df):
        _assert_equal(proxy_df.set_index("a"), real_df.set_index("a"))


class TestDataFrameGroupBy:
    def test_groupby_sum(self, proxy_df_cat, real_df_cat):
        _assert_equal(
            proxy_df_cat.groupby("grp").sum(),
            real_df_cat.groupby("grp").sum(),
        )

    def test_groupby_mean(self, proxy_df_cat, real_df_cat):
        _assert_equal(
            proxy_df_cat.groupby("grp").mean(numeric_only=True),
            real_df_cat.groupby("grp").mean(numeric_only=True),
        )

    def test_groupby_count(self, proxy_df_cat, real_df_cat):
        _assert_equal(
            proxy_df_cat.groupby("grp").count(),
            real_df_cat.groupby("grp").count(),
        )

    def test_groupby_agg(self, proxy_df_cat, real_df_cat):
        _assert_equal(
            proxy_df_cat.groupby("grp").agg("sum"),
            real_df_cat.groupby("grp").agg("sum"),
        )

    def test_groupby_transform(self, proxy_df_cat, real_df_cat):
        _assert_equal(
            proxy_df_cat.groupby("grp").transform("sum"),
            real_df_cat.groupby("grp").transform("sum"),
        )


class TestDataFrameRolling:
    def test_rolling_sum(self, proxy_df, real_df):
        _assert_equal(
            proxy_df["a"].rolling(window=2).sum(),
            real_df["a"].rolling(window=2).sum(),
        )

    def test_rolling_mean(self, proxy_df, real_df):
        _assert_equal(
            proxy_df["a"].rolling(window=2).mean(),
            real_df["a"].rolling(window=2).mean(),
        )

    def test_rolling_std(self, proxy_df, real_df):
        _assert_equal(
            proxy_df["a"].rolling(window=2).std(),
            real_df["a"].rolling(window=2).std(),
        )


class TestDataFrameProperties:
    def test_empty(self, proxy_df, real_df):
        assert proxy_df.empty == real_df.empty

    def test_ndim(self, proxy_df, real_df):
        assert proxy_df.ndim == real_df.ndim

    def test_size(self, proxy_df, real_df):
        assert proxy_df.size == real_df.size

    def test_isna(self, proxy_df, real_df):
        _assert_equal(proxy_df.isna(), real_df.isna())

    def test_isnull(self, proxy_df, real_df):
        _assert_equal(proxy_df.isnull(), real_df.isnull())

    def test_notna(self, proxy_df, real_df):
        _assert_equal(proxy_df.notna(), real_df.notna())

    def test_notnull(self, proxy_df, real_df):
        _assert_equal(proxy_df.notnull(), real_df.notnull())

    def test_duplicated(self, proxy_df, real_df):
        _assert_equal(proxy_df.duplicated(), real_df.duplicated())


class TestDataFrameCopy:
    def test_copy(self, proxy_df, real_df):
        cp = proxy_df.copy()
        _assert_equal(cp, real_df)

    def test_astype(self, proxy_df, real_df):
        _assert_equal(proxy_df.astype("float64"), real_df.astype("float64"))


class TestDataFrameAssign:
    def test_assign(self, proxy_df, real_df):
        _assert_equal(
            proxy_df.assign(d=proxy_df["a"] + 1),
            real_df.assign(d=real_df["a"] + 1),
        )

    def test_insert(self, proxy_df, real_df):
        proxy_df.insert(0, "z", [10, 20, 30, 40, 50])
        assert "z" in proxy_df.columns
        assert proxy_df["z"].tolist() == [10, 20, 30, 40, 50]

    def test_pop(self, proxy_df, real_df):
        proxy_col = proxy_df.pop("a")
        assert "a" not in proxy_df.columns
        assert proxy_col.tolist() == [1, 2, 3, 4, 5]


class TestDataFrameDifferencing:
    def test_diff(self, proxy_df, real_df):
        _assert_equal(proxy_df.diff(), real_df.diff())

    def test_pct_change(self, proxy_df, real_df):
        _assert_equal(proxy_df.pct_change(), real_df.pct_change())

    def test_shift(self, proxy_df, real_df):
        _assert_equal(proxy_df.shift(1), real_df.shift(1))

    def test_cumsum(self, proxy_df, real_df):
        _assert_equal(proxy_df.cumsum(), real_df.cumsum())

    def test_cumprod(self, proxy_df, real_df):
        _assert_equal(proxy_df.cumprod(), real_df.cumprod())

    def test_cummax(self, proxy_df, real_df):
        _assert_equal(proxy_df.cummax(), real_df.cummax())

    def test_cummin(self, proxy_df, real_df):
        _assert_equal(proxy_df.cummin(), real_df.cummin())


class TestDataFrameRanking:
    def test_rank(self, proxy_df, real_df):
        _assert_equal(proxy_df.rank(), real_df.rank())

    def test_quantile(self, proxy_df, real_df):
        _assert_equal(proxy_df.quantile(), real_df.quantile())


class TestDataFrameCorrelation:
    def test_corr(self, proxy_df, real_df):
        _assert_equal(proxy_df.corr(), real_df.corr())

    def test_cov(self, proxy_df, real_df):
        _assert_equal(proxy_df.cov(), real_df.cov())


class TestDataFrameMemory:
    def test_memory_usage(self, proxy_df, real_df):
        assert proxy_df.memory_usage().tolist() == real_df.memory_usage().tolist()


class TestDataFrameQuery:
    def test_query(self, proxy_df, real_df):
        _assert_equal(
            proxy_df.query("a > 2"),
            real_df.query("a > 2"),
        )


class TestDataFrameWhere:
    def test_where(self, proxy_df, real_df):
        _assert_equal(
            proxy_df.where(proxy_df > 2, 0),
            real_df.where(real_df > 2, 0),
        )

    def test_mask(self, proxy_df, real_df):
        _assert_equal(
            proxy_df.mask(proxy_df > 2, 0),
            real_df.mask(real_df > 2, 0),
        )


class TestDataFrameCombine:
    def test_combine_first(self, proxy_df, real_df, proxy_df2, real_df2):
        _assert_equal(
            proxy_df.combine_first(proxy_df2),
            real_df.combine_first(real_df2),
        )


class TestDataFrameStringAccessor:
    def test_str_len(self):
        real = pd.DataFrame({"s": ["hello", "world", "foo"]})
        proxy = ProxyDataFrame(_pandas_obj=real.copy())
        _assert_equal(proxy["s"].str.len(), real["s"].str.len())

    def test_str_upper(self):
        real = pd.DataFrame({"s": ["hello", "world", "foo"]})
        proxy = ProxyDataFrame(_pandas_obj=real.copy())
        _assert_equal(proxy["s"].str.upper(), real["s"].str.upper())

    def test_str_contains(self):
        real = pd.DataFrame({"s": ["hello", "world", "foo"]})
        proxy = ProxyDataFrame(_pandas_obj=real.copy())
        _assert_equal(proxy["s"].str.contains("o"), real["s"].str.contains("o"))


class TestDataFrameDatetimeAccessor:
    def test_dt_year(self):
        real = pd.DataFrame({"d": pd.date_range("2020-01-01", periods=3)})
        proxy = ProxyDataFrame(_pandas_obj=real.copy())
        _assert_equal(proxy["d"].dt.year, real["d"].dt.year)

    def test_dt_month(self):
        real = pd.DataFrame({"d": pd.date_range("2020-01-01", periods=3)})
        proxy = ProxyDataFrame(_pandas_obj=real.copy())
        _assert_equal(proxy["d"].dt.month, real["d"].dt.month)

    def test_dt_day(self):
        real = pd.DataFrame({"d": pd.date_range("2020-01-01", periods=3)})
        proxy = ProxyDataFrame(_pandas_obj=real.copy())
        _assert_equal(proxy["d"].dt.day, real["d"].dt.day)


class TestDataFrameCategoricalAccessor:
    def test_cat_categories(self):
        real = pd.DataFrame({"c": pd.Categorical(["a", "b", "a"])})
        proxy = ProxyDataFrame(_pandas_obj=real.copy())
        assert list(proxy["c"].cat.categories) == list(real["c"].cat.categories)

    def test_cat_codes(self):
        real = pd.DataFrame({"c": pd.Categorical(["a", "b", "a"])})
        proxy = ProxyDataFrame(_pandas_obj=real.copy())
        _assert_equal(proxy["c"].cat.codes, real["c"].cat.codes)
