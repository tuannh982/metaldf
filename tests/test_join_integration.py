"""Tests for Task 6.2: Python layer join integration.

Verifies `ProxyDataFrame.merge()` dispatches equi-joins to
`metaldf_engine.metal_hash_join` + `metal_take` for supported dtypes
(float32/int32, single string key, inner join, no extra kwargs), and falls
back to plain `pd.DataFrame.merge` for everything else -- always producing
results that match `pd.merge`.

Row order from the GPU hash join has no defined relationship to pandas'
merge order, so comparisons sort both sides by every column before
`assert_frame_equal` (mirrors the "compare as an unordered set of pairs"
approach used in `tests/test_join.py` for the raw index-pair kernel tests).
"""

import numpy as np
import pandas as pd
import pytest

from metaldf._wrappers import ProxyDataFrame

try:
    import metaldf_engine

    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by every column value and reset the index for order-independent comparison."""
    df = df.reset_index(drop=True)
    if len(df) == 0:
        return df
    return df.sort_values(by=list(df.columns)).reset_index(drop=True)


def _assert_merge_matches_pandas(left: pd.DataFrame, right: pd.DataFrame, **kwargs):
    """Merge via ProxyDataFrame and assert it matches plain `pd.merge`, ignoring row order."""
    proxy_left = ProxyDataFrame(_pandas_obj=left.copy())
    expected = pd.merge(left, right, **kwargs)
    actual = proxy_left.merge(right, **kwargs)
    actual_pd = actual.to_pandas() if hasattr(actual, "to_pandas") else actual
    pd.testing.assert_frame_equal(_normalize(actual_pd), _normalize(expected), check_like=True)
    return actual_pd


def _spy_on_hash_join(monkeypatch):
    """Monkeypatch `metaldf_engine.metal_hash_join` with a call-counting wrapper.

    Returns the list that gets appended to on every real call -- `len(calls)`
    after the fact tells you whether the GPU path actually ran (vs. a silent
    pandas fallback that would still produce a correct result).
    """
    calls: list[int] = []
    original = metaldf_engine.metal_hash_join

    def spy(build, probe):
        calls.append(1)
        return original(build, probe)

    monkeypatch.setattr(metaldf_engine, "metal_hash_join", spy)
    return calls


# ---------------------------------------------------------------------------
# Metal path: basic equi-joins on `on=`
# ---------------------------------------------------------------------------


class TestMetalMergeBasic:
    def test_merge_on_float32_key(self, monkeypatch):
        calls = _spy_on_hash_join(monkeypatch)
        left = pd.DataFrame(
            {
                "key": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                "lval": np.array([10, 20, 30, 40], dtype=np.int32),
            }
        )
        right = pd.DataFrame(
            {
                "key": np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32),
                "rval": np.array([300, 400, 500, 600], dtype=np.int32),
            }
        )
        _assert_merge_matches_pandas(left, right, on="key", how="inner")
        assert len(calls) == 1

    def test_merge_on_int32_key(self, monkeypatch):
        calls = _spy_on_hash_join(monkeypatch)
        left = pd.DataFrame(
            {
                "key": np.array([1, 2, 3, 4], dtype=np.int32),
                "lval": np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
            }
        )
        right = pd.DataFrame(
            {
                "key": np.array([2, 3, 4, 5], dtype=np.int32),
                "rval": np.array([200.0, 300.0, 400.0, 500.0], dtype=np.float32),
            }
        )
        _assert_merge_matches_pandas(left, right, on="key", how="inner")
        assert len(calls) == 1

    def test_many_to_many(self, monkeypatch):
        calls = _spy_on_hash_join(monkeypatch)
        left = pd.DataFrame(
            {
                "key": np.array([1, 1, 2, 2], dtype=np.int32),
                "lval": np.array([1, 2, 3, 4], dtype=np.int32),
            }
        )
        right = pd.DataFrame(
            {
                "key": np.array([1, 1, 2], dtype=np.int32),
                "rval": np.array([10, 20, 30], dtype=np.int32),
            }
        )
        _assert_merge_matches_pandas(left, right, on="key", how="inner")
        assert len(calls) == 1

    def test_no_matches(self, monkeypatch):
        calls = _spy_on_hash_join(monkeypatch)
        left = pd.DataFrame(
            {"key": np.array([1, 2], dtype=np.int32), "lval": np.array([1, 2], dtype=np.int32)}
        )
        right = pd.DataFrame(
            {"key": np.array([3, 4], dtype=np.int32), "rval": np.array([3, 4], dtype=np.int32)}
        )
        result = _assert_merge_matches_pandas(left, right, on="key", how="inner")
        assert len(result) == 0
        assert len(calls) == 1

    def test_build_probe_side_chosen_by_size(self, monkeypatch):
        """Check that mapped-back left/right indices are correct.

        Regardless of which side is smaller (and thus 'build'), the
        mapped-back left/right indices must be correct.
        """
        calls = _spy_on_hash_join(monkeypatch)
        # left smaller than right
        left_small = pd.DataFrame(
            {"key": np.array([1, 2], dtype=np.int32), "lval": np.array([1, 2], dtype=np.int32)}
        )
        right_big = pd.DataFrame(
            {
                "key": np.array([1, 2, 2, 1, 3], dtype=np.int32),
                "rval": np.array([10, 20, 21, 11, 30], dtype=np.int32),
            }
        )
        _assert_merge_matches_pandas(left_small, right_big, on="key", how="inner")

        # left bigger than right
        _assert_merge_matches_pandas(right_big, left_small, on="key", how="inner")
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# Metal path: left_on/right_on with different column names
# ---------------------------------------------------------------------------


class TestMetalMergeLeftOnRightOn:
    def test_different_key_names(self, monkeypatch):
        calls = _spy_on_hash_join(monkeypatch)
        left = pd.DataFrame(
            {
                "lkey": np.array([1, 2, 3], dtype=np.int32),
                "lval": np.array([10, 20, 30], dtype=np.int32),
            }
        )
        right = pd.DataFrame(
            {
                "rkey": np.array([2, 3, 4], dtype=np.int32),
                "rval": np.array([200, 300, 400], dtype=np.int32),
            }
        )
        result = _assert_merge_matches_pandas(
            left, right, left_on="lkey", right_on="rkey", how="inner"
        )
        # Both key columns (lkey, rkey) must be present since they have
        # different names -- pandas doesn't merge them into one.
        assert set(result.columns) == {"lkey", "lval", "rkey", "rval"}
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Metal path: overlapping non-key column names get pandas' _x/_y suffixes
# ---------------------------------------------------------------------------


class TestMetalMergeOverlappingColumns:
    def test_overlapping_non_key_columns_get_suffixed(self, monkeypatch):
        calls = _spy_on_hash_join(monkeypatch)
        left = pd.DataFrame(
            {
                "key": np.array([1, 2, 3], dtype=np.int32),
                "val": np.array([10, 20, 30], dtype=np.int32),
            }
        )
        right = pd.DataFrame(
            {
                "key": np.array([2, 3, 4], dtype=np.int32),
                "val": np.array([200, 300, 400], dtype=np.int32),
            }
        )
        result = _assert_merge_matches_pandas(left, right, on="key", how="inner")
        assert set(result.columns) == {"key", "val_x", "val_y"}
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Metal path: mixed dtypes -- key must be float32/int32, but other columns
# of any dtype should still gather correctly (numpy fallback for dtypes
# `metal_take` doesn't support).
# ---------------------------------------------------------------------------


class TestMetalMergeMixedDtypeColumns:
    def test_extra_float64_and_string_columns(self, monkeypatch):
        """Check gathering of non-key columns with unsupported dtypes.

        Non-key columns with dtypes Metal's `take` doesn't support
        (float64, object/string) should still be gathered correctly, while
        the join itself still runs on the GPU (single metal_hash_join call).
        """
        calls = _spy_on_hash_join(monkeypatch)
        left = pd.DataFrame(
            {
                "key": np.array([1, 2, 3, 4], dtype=np.int32),
                "amount": np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64),
                "label": ["a", "b", "c", "d"],
            }
        )
        right = pd.DataFrame(
            {
                "key": np.array([2, 3, 4, 5], dtype=np.int32),
                "note": ["w", "x", "y", "z"],
            }
        )
        _assert_merge_matches_pandas(left, right, on="key", how="inner")
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Fallback to pandas: non-inner how, extra kwargs, unsupported dtypes
# ---------------------------------------------------------------------------


class TestMetalMergeFallback:
    def test_left_join_uses_metal(self, monkeypatch):
        calls = _spy_on_hash_join(monkeypatch)
        left = pd.DataFrame(
            {
                "key": np.array([1, 2, 3], dtype=np.int32),
                "lval": np.array([1, 2, 3], dtype=np.int32),
            }
        )
        right = pd.DataFrame(
            {
                "key": np.array([2, 3, 4], dtype=np.int32),
                "rval": np.array([2, 3, 4], dtype=np.int32),
            }
        )
        proxy_left = ProxyDataFrame(_pandas_obj=left.copy())
        expected = pd.merge(left, right, on="key", how="left")
        result = proxy_left.merge(right, on="key", how="left")
        result_pd = result.to_pandas() if hasattr(result, "to_pandas") else result
        pd.testing.assert_frame_equal(
            result_pd.sort_values("key").reset_index(drop=True),
            expected.sort_values("key").reset_index(drop=True),
            check_dtype=False,
        )
        assert len(calls) == 1

    def test_fallback_for_extra_kwargs(self, monkeypatch):
        calls = _spy_on_hash_join(monkeypatch)
        left = pd.DataFrame(
            {"key": np.array([1, 2, 3], dtype=np.int32), "val": np.array([1, 2, 3], dtype=np.int32)}
        )
        right = pd.DataFrame(
            {"key": np.array([2, 3, 4], dtype=np.int32), "val": np.array([2, 3, 4], dtype=np.int32)}
        )
        proxy_left = ProxyDataFrame(_pandas_obj=left.copy())
        expected = pd.merge(left, right, on="key", how="inner", suffixes=("_l", "_r"))
        result = proxy_left.merge(right, on="key", how="inner", suffixes=("_l", "_r"))
        result_pd = result.to_pandas() if hasattr(result, "to_pandas") else result
        pd.testing.assert_frame_equal(result_pd, expected)
        assert len(calls) == 0

    def test_fallback_for_unsupported_int64_key_dtype(self, monkeypatch):
        """int64 keys aren't supported by the GPU hash join -- must fall back."""
        calls = _spy_on_hash_join(monkeypatch)
        left = pd.DataFrame({"key": [1, 2, 3], "lval": [10, 20, 30]})  # int64 default
        right = pd.DataFrame({"key": [2, 3, 4], "rval": [200, 300, 400]})
        proxy_left = ProxyDataFrame(_pandas_obj=left.copy())
        expected = pd.merge(left, right, on="key", how="inner")
        result = proxy_left.merge(right, on="key", how="inner")
        result_pd = result.to_pandas() if hasattr(result, "to_pandas") else result
        pd.testing.assert_frame_equal(result_pd, expected)
        assert len(calls) == 0

    def test_fallback_for_string_key(self, monkeypatch):
        calls = _spy_on_hash_join(monkeypatch)
        left = pd.DataFrame({"key": ["a", "b", "c"], "lval": [1, 2, 3]})
        right = pd.DataFrame({"key": ["b", "c", "d"], "rval": [20, 30, 40]})
        proxy_left = ProxyDataFrame(_pandas_obj=left.copy())
        expected = pd.merge(left, right, on="key", how="inner")
        result = proxy_left.merge(right, on="key", how="inner")
        result_pd = result.to_pandas() if hasattr(result, "to_pandas") else result
        pd.testing.assert_frame_equal(result_pd, expected)
        assert len(calls) == 0

    def test_fallback_for_mismatched_key_dtypes(self, monkeypatch):
        """float32 left key vs int32 right key -- must fall back, not crash."""
        calls = _spy_on_hash_join(monkeypatch)
        left = pd.DataFrame(
            {"key": np.array([1.0, 2.0, 3.0], dtype=np.float32), "lval": [1, 2, 3]}
        )
        right = pd.DataFrame({"key": np.array([2, 3, 4], dtype=np.int32), "rval": [20, 30, 40]})
        proxy_left = ProxyDataFrame(_pandas_obj=left.copy())
        expected = pd.merge(left, right, on="key", how="inner")
        result = proxy_left.merge(right, on="key", how="inner")
        result_pd = result.to_pandas() if hasattr(result, "to_pandas") else result
        pd.testing.assert_frame_equal(result_pd, expected)
        assert len(calls) == 0

    def test_fallback_for_multi_column_on(self, monkeypatch):
        """Multi-column keys aren't supported by the single-key GPU path."""
        calls = _spy_on_hash_join(monkeypatch)
        left = pd.DataFrame(
            {
                "k1": np.array([1, 2, 3], dtype=np.int32),
                "k2": np.array([1, 1, 2], dtype=np.int32),
                "lval": [1, 2, 3],
            }
        )
        right = pd.DataFrame(
            {
                "k1": np.array([2, 3, 4], dtype=np.int32),
                "k2": np.array([1, 2, 2], dtype=np.int32),
                "rval": [20, 30, 40],
            }
        )
        proxy_left = ProxyDataFrame(_pandas_obj=left.copy())
        expected = pd.merge(left, right, on=["k1", "k2"], how="inner")
        result = proxy_left.merge(right, on=["k1", "k2"], how="inner")
        result_pd = result.to_pandas() if hasattr(result, "to_pandas") else result
        pd.testing.assert_frame_equal(result_pd, expected)
        assert len(calls) == 0

    def test_fallback_for_missing_key_spec(self, monkeypatch):
        """No `on`/`left_on`/`right_on` at all -- pandas infers common columns."""
        calls = _spy_on_hash_join(monkeypatch)
        left = pd.DataFrame(
            {"key": np.array([1, 2, 3], dtype=np.int32), "lval": [1, 2, 3]}
        )
        right = pd.DataFrame(
            {"key": np.array([2, 3, 4], dtype=np.int32), "rval": [20, 30, 40]}
        )
        proxy_left = ProxyDataFrame(_pandas_obj=left.copy())
        expected = pd.merge(left, right, how="inner")
        result = proxy_left.merge(right, how="inner")
        result_pd = result.to_pandas() if hasattr(result, "to_pandas") else result
        pd.testing.assert_frame_equal(result_pd, expected)
        assert len(calls) == 0


# ---------------------------------------------------------------------------
# pd.merge() module-level call form
# ---------------------------------------------------------------------------


class TestPdMergeFreeFunction:
    def test_pd_merge_with_proxy_operands_is_correct(self):
        """Check that `pd.merge` with proxy operands still produces a correct result.

        `pd.merge(proxy_df, other)` (real pandas, not the accelerator) must
        still produce a correct result even though pandas' merge algorithm
        doesn't call `ProxyDataFrame.merge` on its own.
        """
        left = pd.DataFrame(
            {
                "key": np.array([1, 2, 3], dtype=np.int32),
                "lval": np.array([10, 20, 30], dtype=np.int32),
            }
        )
        right = pd.DataFrame(
            {
                "key": np.array([2, 3, 4], dtype=np.int32),
                "rval": np.array([200, 300, 400], dtype=np.int32),
            }
        )
        proxy_left = ProxyDataFrame(_pandas_obj=left.copy())
        result = pd.merge(proxy_left, right, on="key", how="inner")
        result_pd = result.to_pandas() if hasattr(result, "to_pandas") else result
        expected = pd.merge(left, right, on="key", how="inner")
        pd.testing.assert_frame_equal(_normalize(result_pd), _normalize(expected), check_like=True)

    def test_pd_merge_dispatches_to_metal_when_accelerator_installed(self, monkeypatch):
        """Check that pd.merge() routes through the Metal fast path when installed.

        Under the real `import pandas` interception (metaldf._accelerator),
        `pd.merge(df1, df2, on='key')` should route through
        `ProxyDataFrame.merge`'s Metal fast path, not just produce a correct
        result via the plain pandas algorithm.
        """
        import sys

        from metaldf._accelerator import install, uninstall

        install()
        try:
            sys.modules.pop("pandas", None)
            import pandas as pd2

            df1 = pd2.DataFrame(
                {
                    "key": np.array([1, 2, 3], dtype=np.int32),
                    "lval": np.array([10, 20, 30], dtype=np.int32),
                }
            )
            df2 = pd2.DataFrame(
                {
                    "key": np.array([2, 3, 4], dtype=np.int32),
                    "rval": np.array([200, 300, 400], dtype=np.int32),
                }
            )
            assert type(df1).__name__ == "ProxyDataFrame"

            calls = _spy_on_hash_join(monkeypatch)
            result = pd2.merge(df1, df2, on="key", how="inner")
            assert len(calls) == 1, (
                "pd.merge() should dispatch through ProxyDataFrame.merge's Metal path"
            )

            result_pd = result.to_pandas() if hasattr(result, "to_pandas") else result
            got = set(
                zip(
                    result_pd["key"].tolist(),
                    result_pd["lval"].tolist(),
                    result_pd["rval"].tolist(),
                    strict=True,
                )
            )
            assert got == {(2, 20, 200), (3, 30, 300)}
        finally:
            uninstall()
            sys.modules.pop("pandas", None)


# ---------------------------------------------------------------------------
# Large-ish scale sanity check
# ---------------------------------------------------------------------------


class TestMetalMergeLarge:
    def test_large_partial_overlap(self, monkeypatch):
        calls = _spy_on_hash_join(monkeypatch)
        n = 50_000
        left = pd.DataFrame(
            {
                "key": np.arange(n, dtype=np.int32),
                "lval": np.arange(n, dtype=np.int32) * 2,
            }
        )
        right = pd.DataFrame(
            {
                "key": np.arange(n // 2, n + n // 2, dtype=np.int32),
                "rval": np.arange(n // 2, n + n // 2, dtype=np.int32) * 3,
            }
        )
        result = _assert_merge_matches_pandas(left, right, on="key", how="inner")
        assert len(result) == n // 2
        assert len(calls) == 1
