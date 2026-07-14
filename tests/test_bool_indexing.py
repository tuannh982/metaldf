"""Tests for Python-layer boolean indexing (`df[mask]` / `series[mask]`).

Task 4.2 (Phase 4): wires the Task 4.1 GPU stream-compaction kernel
(`metaldf_engine.metal_compact`) into `ProxyDataFrame.__getitem__` and
`ProxySeries.__getitem__` so a bool-dtype mask filters via Metal, with a
transparent fallback to plain pandas indexing for anything Metal can't
handle (non-bool keys, unsupported dtypes, index mismatches, ...).
"""

import numpy as np
import pandas as pd
import pytest

from metaldf._wrappers import ProxyDataFrame, ProxySeries

try:
    import metaldf_engine  # noqa: F401

    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")


def _make_df() -> ProxyDataFrame:
    return ProxyDataFrame(
        _pandas_obj=pd.DataFrame(
            {
                "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
                "b": np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32),
            }
        )
    )


class TestDataFrameBoolFilter:
    def test_basic_mask_from_comparison(self):
        df = _make_df()
        mask = df["a"] > 3.0
        result = df[mask]

        expected = pd.DataFrame({"a": [4.0, 5.0], "b": [40.0, 50.0]})
        pd.testing.assert_frame_equal(
            result.to_pandas().reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_result_is_proxy_dataframe(self):
        df = _make_df()
        mask = df["a"] > 3.0
        result = df[mask]
        assert isinstance(result, ProxyDataFrame)

    def test_matches_plain_pandas(self):
        """The Metal-filtered result must match plain pandas' own `df[mask]`."""
        rng = np.random.default_rng(0)
        data = pd.DataFrame(
            {
                "a": rng.random(50).astype(np.float32),
                "b": rng.random(50).astype(np.float32),
            }
        )
        df = ProxyDataFrame(_pandas_obj=data)
        mask = data["a"] > 0.5

        result = df[mask]
        expected = data[mask]

        pd.testing.assert_frame_equal(
            result.to_pandas().reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_all_true(self):
        df = _make_df()
        mask = pd.Series([True, True, True, True, True])
        result = df[mask]
        pd.testing.assert_frame_equal(
            result.to_pandas().reset_index(drop=True),
            df.to_pandas().reset_index(drop=True),
            check_dtype=False,
        )

    def test_all_false(self):
        df = _make_df()
        mask = pd.Series([False, False, False, False, False])
        result = df[mask]
        assert len(result.to_pandas()) == 0
        assert list(result.to_pandas().columns) == ["a", "b"]

    def test_multi_column_filter(self):
        df = ProxyDataFrame(
            _pandas_obj=pd.DataFrame(
                {
                    "a": np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
                    "b": np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32),
                    "c": np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float32),
                }
            )
        )
        mask = pd.Series([True, False, True, False, True])
        result = df[mask]

        expected = pd.DataFrame(
            {"a": [1.0, 3.0, 5.0], "b": [10.0, 30.0, 50.0], "c": [100.0, 300.0, 500.0]}
        )
        pd.testing.assert_frame_equal(
            result.to_pandas().reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_bool_ndarray_mask(self):
        """A bare numpy bool array (not a Series) is also recognized as a mask."""
        df = _make_df()
        mask = np.array([True, False, True, False, True])
        result = df[mask]
        expected = pd.DataFrame({"a": [1.0, 3.0, 5.0], "b": [10.0, 30.0, 50.0]})
        pd.testing.assert_frame_equal(
            result.to_pandas().reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_proxyseries_mask(self):
        """A ProxySeries-wrapped bool mask is also recognized."""
        df = _make_df()
        mask = ProxySeries(_pandas_obj=pd.Series([True, False, True, False, True]))
        result = df[mask]
        expected = pd.DataFrame({"a": [1.0, 3.0, 5.0], "b": [10.0, 30.0, 50.0]})
        pd.testing.assert_frame_equal(
            result.to_pandas().reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )

    def test_column_access_still_works(self):
        """Non-mask keys (plain column selection) must not be treated as a mask."""
        df = _make_df()
        col = df["a"]
        np.testing.assert_allclose(np.asarray(col), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_column_list_access_still_works(self):
        df = _make_df()
        result = df[["b", "a"]]
        assert list(result.columns) == ["b", "a"]


class TestSeriesBoolFilter:
    def test_basic(self):
        series = ProxySeries(
            _pandas_obj=pd.Series(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))
        )
        mask = pd.Series([False, False, False, True, True])
        result = series[mask]
        np.testing.assert_allclose(np.asarray(result.to_pandas()), [4.0, 5.0])

    def test_result_is_proxy_series(self):
        series = ProxySeries(
            _pandas_obj=pd.Series(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        )
        mask = pd.Series([True, False, True])
        result = series[mask]
        assert isinstance(result, ProxySeries)

    def test_matches_plain_pandas(self):
        rng = np.random.default_rng(1)
        data = pd.Series(rng.random(50).astype(np.float32))
        series = ProxySeries(_pandas_obj=data)
        mask = data > 0.5

        result = series[mask]
        expected = data[mask]

        pd.testing.assert_series_equal(
            result.to_pandas().reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_names=False,
        )

    def test_all_true(self):
        series = ProxySeries(
            _pandas_obj=pd.Series(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        )
        mask = pd.Series([True, True, True])
        result = series[mask]
        np.testing.assert_allclose(np.asarray(result.to_pandas()), [1.0, 2.0, 3.0])

    def test_all_false(self):
        series = ProxySeries(
            _pandas_obj=pd.Series(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        )
        mask = pd.Series([False, False, False])
        result = series[mask]
        assert len(result.to_pandas()) == 0

    def test_int32_series(self):
        series = ProxySeries(
            _pandas_obj=pd.Series(np.array([10, 20, 30, 40], dtype=np.int32))
        )
        mask = pd.Series([True, False, False, True])
        result = series[mask]
        np.testing.assert_array_equal(np.asarray(result.to_pandas()), [10, 40])

    def test_non_mask_key_still_works(self):
        """Non-bool keys (e.g. integer positional slicing) must fall through untouched."""
        series = ProxySeries(
            _pandas_obj=pd.Series(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        )
        result = series[1:3]
        np.testing.assert_allclose(np.asarray(result.to_pandas()), [2.0, 3.0])
