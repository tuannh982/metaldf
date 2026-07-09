import platform
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin", reason="Metal tests only run on macOS",
)

import metaldf_engine


def test_string_groupby_sum():
    keys = metaldf_engine.MetalSeries.from_strings(["a", "b", "a", "b", "a"])
    values = metaldf_engine.MetalSeries.from_numpy(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))
    result_keys, result_values = metaldf_engine.metal_string_groupby(keys, values, "sum")

    # Convert to dict for order-independent comparison
    rk = result_keys.to_strings()
    rv = list(result_values.to_numpy())
    result_dict = dict(zip(rk, rv))

    assert result_dict["a"] == pytest.approx(9.0)
    assert result_dict["b"] == pytest.approx(6.0)


def test_string_groupby_count():
    keys = metaldf_engine.MetalSeries.from_strings(["x", "y", "x", "z", "y", "x"])
    values = metaldf_engine.MetalSeries.from_numpy(np.array([1.0] * 6, dtype=np.float32))
    result_keys, result_values = metaldf_engine.metal_string_groupby(keys, values, "count")

    rk = result_keys.to_strings()
    rv = list(result_values.to_numpy())
    result_dict = dict(zip(rk, rv))

    assert result_dict["x"] == 3
    assert result_dict["y"] == 2
    assert result_dict["z"] == 1


def test_string_groupby_min_float():
    keys = metaldf_engine.MetalSeries.from_strings(["a", "b", "a", "b", "a"])
    values = metaldf_engine.MetalSeries.from_numpy(np.array([3.0, 2.0, 1.0, 4.0, 5.0], dtype=np.float32))
    rk, rv = metaldf_engine.metal_string_groupby(keys, values, "min")
    result = dict(zip(rk.to_strings(), rv.to_numpy()))
    assert result["a"] == pytest.approx(1.0)
    assert result["b"] == pytest.approx(2.0)


def test_string_groupby_min_int():
    keys = metaldf_engine.MetalSeries.from_strings(["a", "b", "a", "b", "a"])
    values = metaldf_engine.MetalSeries.from_numpy_i32(np.array([3, 2, 1, 4, 5], dtype=np.int32))
    rk, rv = metaldf_engine.metal_string_groupby(keys, values, "min")
    result = dict(zip(rk.to_strings(), rv.to_numpy()))
    assert result["a"] == 1
    assert result["b"] == 2


def test_string_groupby_max_float():
    keys = metaldf_engine.MetalSeries.from_strings(["a", "b", "a", "b", "a"])
    values = metaldf_engine.MetalSeries.from_numpy(np.array([3.0, 2.0, 1.0, 4.0, 5.0], dtype=np.float32))
    rk, rv = metaldf_engine.metal_string_groupby(keys, values, "max")
    result = dict(zip(rk.to_strings(), rv.to_numpy()))
    assert result["a"] == pytest.approx(5.0)
    assert result["b"] == pytest.approx(4.0)


def test_string_groupby_max_int():
    keys = metaldf_engine.MetalSeries.from_strings(["a", "b", "a", "b", "a"])
    values = metaldf_engine.MetalSeries.from_numpy_i32(np.array([3, 2, 1, 4, 5], dtype=np.int32))
    rk, rv = metaldf_engine.metal_string_groupby(keys, values, "max")
    result = dict(zip(rk.to_strings(), rv.to_numpy()))
    assert result["a"] == 5
    assert result["b"] == 4


def test_string_groupby_mean_float():
    keys = metaldf_engine.MetalSeries.from_strings(["a", "b", "a", "b", "a"])
    values = metaldf_engine.MetalSeries.from_numpy(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32))
    rk, rv = metaldf_engine.metal_string_groupby(keys, values, "mean")
    result = dict(zip(rk.to_strings(), rv.to_numpy()))
    assert result["a"] == pytest.approx(3.0)
    assert result["b"] == pytest.approx(3.0)


def test_string_groupby_mean_int():
    keys = metaldf_engine.MetalSeries.from_strings(["a", "b", "a", "b"])
    values = metaldf_engine.MetalSeries.from_numpy_i32(np.array([1, 2, 3, 4], dtype=np.int32))
    rk, rv = metaldf_engine.metal_string_groupby(keys, values, "mean")
    result = dict(zip(rk.to_strings(), rv.to_numpy()))
    # int32 mean returns float64 (matching pandas)
    assert rv.to_numpy().dtype == np.float64
    assert result["a"] == pytest.approx(2.0)
    assert result["b"] == pytest.approx(3.0)


def test_string_groupby_matches_pandas():
    import random
    random.seed(42)
    n = 10_000
    categories = ["alpha", "beta", "gamma", "delta", "epsilon"]
    key_list = [random.choice(categories) for _ in range(n)]
    value_arr = np.random.rand(n).astype(np.float32)

    # Pandas reference
    df = pd.DataFrame({"key": key_list, "val": value_arr})
    expected = df.groupby("key")["val"].sum()

    # Metal
    keys = metaldf_engine.MetalSeries.from_strings(key_list)
    values = metaldf_engine.MetalSeries.from_numpy(value_arr)
    rk, rv = metaldf_engine.metal_string_groupby(keys, values, "sum")
    metal_dict = dict(zip(rk.to_strings(), rv.to_numpy()))

    for cat in categories:
        assert metal_dict[cat] == pytest.approx(expected[cat], rel=1e-3)


def test_string_groupby_min_max_matches_pandas():
    import random
    random.seed(99)
    n = 5_000
    categories = ["foo", "bar", "baz"]
    key_list = [random.choice(categories) for _ in range(n)]
    value_arr = np.random.rand(n).astype(np.float32)

    df = pd.DataFrame({"key": key_list, "val": value_arr})

    keys = metaldf_engine.MetalSeries.from_strings(key_list)
    values = metaldf_engine.MetalSeries.from_numpy(value_arr)

    for agg in ("min", "max"):
        expected = getattr(df.groupby("key")["val"], agg)()
        rk, rv = metaldf_engine.metal_string_groupby(keys, values, agg)
        metal_dict = dict(zip(rk.to_strings(), rv.to_numpy()))
        for cat in categories:
            assert metal_dict[cat] == pytest.approx(expected[cat], rel=1e-5), \
                f"{agg}: {cat} expected {expected[cat]}, got {metal_dict[cat]}"


def test_string_groupby_mean_matches_pandas():
    import random
    random.seed(77)
    n = 5_000
    categories = ["one", "two", "three"]
    key_list = [random.choice(categories) for _ in range(n)]
    value_arr = np.random.rand(n).astype(np.float32)

    df = pd.DataFrame({"key": key_list, "val": value_arr})
    expected = df.groupby("key")["val"].mean()

    keys = metaldf_engine.MetalSeries.from_strings(key_list)
    values = metaldf_engine.MetalSeries.from_numpy(value_arr)
    rk, rv = metaldf_engine.metal_string_groupby(keys, values, "mean")
    metal_dict = dict(zip(rk.to_strings(), rv.to_numpy()))

    for cat in categories:
        assert metal_dict[cat] == pytest.approx(expected[cat], rel=1e-3)


def test_string_groupby_sort_path():
    """Test with >500K rows to exercise the sort-based fallback."""
    import random
    random.seed(123)
    n = 600_000
    categories = ["cat_a", "cat_b", "cat_c"]
    key_list = [random.choice(categories) for _ in range(n)]
    value_arr = np.random.rand(n).astype(np.float32)

    df = pd.DataFrame({"key": key_list, "val": value_arr})

    keys = metaldf_engine.MetalSeries.from_strings(key_list)
    values = metaldf_engine.MetalSeries.from_numpy(value_arr)

    for agg in ("sum", "min", "max", "count", "mean"):
        if agg == "mean":
            expected = df.groupby("key")["val"].mean()
        elif agg == "count":
            expected = df.groupby("key")["val"].count().astype(float)
        else:
            expected = getattr(df.groupby("key")["val"], agg)()

        rk, rv = metaldf_engine.metal_string_groupby(keys, values, agg)
        metal_dict = dict(zip(rk.to_strings(), rv.to_numpy()))

        for cat in categories:
            assert metal_dict[cat] == pytest.approx(float(expected[cat]), rel=1e-3), \
                f"sort path {agg}: {cat} expected {expected[cat]}, got {metal_dict[cat]}"
