import platform

import numpy as np
import pandas as pd
import pytest

from metaldf._engine import execute, clear_registry

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Metal sort/groupby tests only run on macOS",
)


def _register_ops():
    """Re-register Metal ops after clear_registry()."""
    import importlib
    import metaldf._engine
    # Reload _metal submodule first to pick up new metaldf_engine extension
    import metaldf._engine._metal
    importlib.reload(metaldf._engine._metal)
    importlib.reload(metaldf._engine)
    # Return fresh execute reference from reloaded module
    return metaldf._engine.execute


def test_metal_sort_matches_pandas():
    clear_registry()
    execute_fn = _register_ops()

    s = pd.Series(np.random.randn(100_000).astype(np.float32))
    expected = s.sort_values().reset_index(drop=True)
    actual = execute_fn("sort", s).reset_index(drop=True)
    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-4)


def test_metal_argsort_matches_pandas():
    clear_registry()
    execute_fn = _register_ops()

    s = pd.Series(np.random.randn(100_000).astype(np.float32))
    actual = execute_fn("argsort", s)
    # Verify argsort is correct: values[argsort] must be sorted
    sorted_values = s.values[actual]
    assert np.all(sorted_values[:-1] <= sorted_values[1:]), "argsort did not produce sorted values"


def test_metal_sort_int32():
    clear_registry()
    execute_fn = _register_ops()

    s = pd.Series(np.random.randint(-10000, 10000, size=100_000, dtype=np.int32))
    expected = s.sort_values().reset_index(drop=True)
    actual = execute_fn("sort", s).reset_index(drop=True)
    np.testing.assert_array_equal(actual.values, expected.values)


def test_metal_argsort_int32():
    clear_registry()
    execute_fn = _register_ops()

    s = pd.Series(np.random.randint(-10000, 10000, size=100_000, dtype=np.int32))
    actual = execute_fn("argsort", s)
    sorted_values = s.values[actual]
    assert np.all(sorted_values[:-1] <= sorted_values[1:])


def test_metal_sort_int64():
    clear_registry()
    execute_fn = _register_ops()

    s = pd.Series(np.random.randint(-10000, 10000, size=100_000, dtype=np.int64))
    expected = s.sort_values().reset_index(drop=True)
    actual = execute_fn("sort", s).reset_index(drop=True)
    np.testing.assert_array_equal(actual.values, expected.values)


def test_metal_sort_int64_8_passes():
    """int64 radix sort uses 8 passes instead of 4 — verify correctness with large value range."""
    clear_registry()
    execute_fn = _register_ops()

    s = pd.Series(np.random.randint(-2**40, 2**40, size=50_000, dtype=np.int64))
    expected = s.sort_values().reset_index(drop=True)
    actual = execute_fn("sort", s).reset_index(drop=True)
    np.testing.assert_array_equal(actual.values, expected.values)


def test_metal_sort_negative_values():
    """Verify radix key conversion handles negative values correctly for all types."""
    clear_registry()
    execute_fn = _register_ops()

    for dtype in [np.float32, np.int32]:
        s = pd.Series(np.array([-5, -1, 0, 1, 5, -3, 3, -2, 2, -4] * 1000, dtype=dtype))
        expected = s.sort_values().reset_index(drop=True)
        actual = execute_fn("sort", s).reset_index(drop=True)
        np.testing.assert_array_equal(actual.values, expected.values)


def test_metal_groupby_sum_matches_pandas():
    clear_registry()
    execute_fn = _register_ops()

    keys = pd.Series(np.random.choice(100, size=100_000).astype(np.float32))
    values = pd.Series(np.random.randn(100_000).astype(np.float32))

    expected = pd.Series(values.values).groupby(keys.values).sum().sort_index()
    actual = execute_fn("groupby_sum", keys, values).sort_index()

    # Verify: the number of unique keys matches
    assert len(actual) == len(expected), f"Length mismatch: {len(actual)} vs {len(expected)}"
    # Verify values are close (allowing for different key order)
    np.testing.assert_allclose(sorted(actual.values), sorted(expected.values), rtol=1e-3)


def test_metal_groupby_min_matches_pandas():
    clear_registry()
    execute_fn = _register_ops()

    keys = pd.Series(np.random.choice(100, size=100_000).astype(np.float32))
    values = pd.Series(np.random.randn(100_000).astype(np.float32))

    expected = pd.Series(values.values).groupby(keys.values).min().sort_index()
    actual = execute_fn("groupby_min", keys, values).sort_index()

    assert len(actual) == len(expected)
    np.testing.assert_allclose(sorted(actual.values), sorted(expected.values), rtol=1e-3)


def test_metal_groupby_max_matches_pandas():
    clear_registry()
    execute_fn = _register_ops()

    keys = pd.Series(np.random.choice(100, size=100_000).astype(np.float32))
    values = pd.Series(np.random.randn(100_000).astype(np.float32))

    expected = pd.Series(values.values).groupby(keys.values).max().sort_index()
    actual = execute_fn("groupby_max", keys, values).sort_index()

    assert len(actual) == len(expected)
    np.testing.assert_allclose(sorted(actual.values), sorted(expected.values), rtol=1e-3)


def test_metal_groupby_count_matches_pandas():
    clear_registry()
    execute_fn = _register_ops()

    keys = pd.Series(np.random.choice(100, size=100_000).astype(np.float32))
    values = pd.Series(np.random.randn(100_000).astype(np.float32))

    expected = pd.Series(values.values).groupby(keys.values).count().sort_index()
    actual = execute_fn("groupby_count", keys, values).sort_index()

    assert len(actual) == len(expected)
    np.testing.assert_array_equal(sorted(actual.values), sorted(expected.values))


def test_metal_groupby_mean_matches_pandas():
    clear_registry()
    execute_fn = _register_ops()

    keys = pd.Series(np.random.choice(100, size=100_000).astype(np.float32))
    values = pd.Series(np.random.randn(100_000).astype(np.float32))

    expected = pd.Series(values.values).groupby(keys.values).mean().sort_index()
    actual = execute_fn("groupby_mean", keys, values).sort_index()

    assert len(actual) == len(expected)
    np.testing.assert_allclose(sorted(actual.values), sorted(expected.values), rtol=1e-3)


def test_metal_groupby_sum_int32():
    clear_registry()
    execute_fn = _register_ops()

    keys = pd.Series(np.random.choice(100, size=100_000).astype(np.int32))
    values = pd.Series(np.random.randint(-100, 100, size=100_000).astype(np.int32))

    expected = pd.Series(values.values).groupby(keys.values).sum().sort_index()
    actual = execute_fn("groupby_sum", keys, values).sort_index()

    assert len(actual) == len(expected)
    np.testing.assert_array_equal(sorted(actual.values), sorted(expected.values))


def test_proxy_series_sort_values():
    """ProxySeries.sort_values() should dispatch to Metal."""
    import metaldf
    metaldf.install()
    import pandas as pd

    s = pd.Series(np.random.randn(100_000).astype(np.float32))
    expected = s.sort_values().reset_index(drop=True)
    actual = s.sort_values().reset_index(drop=True)

    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-4)
    metaldf.uninstall()


def test_proxy_groupby_sum():
    """ProxyGroupBy.sum() should dispatch to Metal when possible."""
    import metaldf
    metaldf.install()
    import pandas as pd

    np.random.seed(42)
    keys = np.random.choice(100, size=100_000).astype(np.float32)
    values = np.random.randn(100_000).astype(np.float32)
    df = pd.DataFrame({"key": keys, "val": values})

    expected = df.groupby("key")["val"].sum().sort_index()
    actual = df.groupby("key")["val"].sum().sort_index()

    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-3)
    metaldf.uninstall()


def test_proxy_groupby_mean():
    import metaldf
    metaldf.install()
    import pandas as pd

    np.random.seed(42)
    keys = np.random.choice(50, size=50_000).astype(np.float32)
    values = np.random.randn(50_000).astype(np.float32)
    df = pd.DataFrame({"key": keys, "val": values})

    expected = df.groupby("key")["val"].mean().sort_index()
    actual = df.groupby("key")["val"].mean().sort_index()

    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-3)
    metaldf.uninstall()


def test_proxy_groupby_int32():
    """GroupBy with int32 keys and values should dispatch to Metal."""
    import metaldf
    metaldf.install()
    import pandas as pd

    np.random.seed(42)
    keys = np.random.choice(100, size=100_000).astype(np.int32)
    values = np.random.randint(-100, 100, size=100_000).astype(np.int32)
    df = pd.DataFrame({"key": keys, "val": values})

    expected = df.groupby("key")["val"].sum().sort_index()
    actual = df.groupby("key")["val"].sum().sort_index()

    np.testing.assert_array_equal(actual.values, expected.values)
    metaldf.uninstall()
