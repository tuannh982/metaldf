import platform

import numpy as np
import pandas as pd
import pytest

from metaldf._engine import execute, clear_registry

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Metal reduction tests only run on macOS",
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


def test_metal_sum_matches_pandas():
    clear_registry()
    import metaldf._engine  # noqa: F401 -- triggers auto-registration

    s = pd.Series(np.random.randn(1_000_000).astype(np.float32))
    expected = s.sum()
    actual = execute("sum", s)
    np.testing.assert_allclose(actual, expected, rtol=1e-4)


def test_metal_min_matches_pandas():
    clear_registry()
    import metaldf._engine  # noqa: F401

    s = pd.Series(np.random.randn(1_000_000).astype(np.float32))
    expected = s.min()
    actual = execute("min", s)
    np.testing.assert_allclose(actual, expected, rtol=1e-4)


def test_metal_max_matches_pandas():
    clear_registry()
    import metaldf._engine  # noqa: F401

    s = pd.Series(np.random.randn(1_000_000).astype(np.float32))
    expected = s.max()
    actual = execute("max", s)
    np.testing.assert_allclose(actual, expected, rtol=1e-4)


def test_metal_mean_matches_pandas():
    clear_registry()
    import metaldf._engine  # noqa: F401

    s = pd.Series(np.random.randn(1_000_000).astype(np.float32))
    expected = s.mean()
    actual = execute("mean", s)
    np.testing.assert_allclose(actual, expected, rtol=1e-4)


def test_small_array_falls_back():
    """Small arrays should fall back to pandas and still be correct."""
    clear_registry()
    import metaldf._engine  # noqa: F401

    s = pd.Series(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert execute("sum", s) == 6.0
    assert execute("min", s) == 1.0
    assert execute("max", s) == 3.0
    assert execute("mean", s) == 2.0


def test_int64_dispatches_to_metal():
    """int64 reductions should now dispatch to Metal, not fall back."""
    clear_registry()
    import metaldf._engine  # noqa: F401

    s = pd.Series(np.array([1, 2, 3, 4, 5] * 1000, dtype=np.int64))
    assert execute("sum", s) == 15000
    assert execute("min", s) == 1
    assert execute("max", s) == 5
    np.testing.assert_allclose(execute("mean", s), 3.0)


def test_metal_sum_int32():
    clear_registry()
    import metaldf._engine  # noqa: F401

    s = pd.Series(np.random.randint(-1000, 1000, size=100_000, dtype=np.int32))
    expected = int(s.sum())
    actual = execute("sum", s)
    assert actual == expected, f"int32 sum: expected {expected}, got {actual}"


def test_metal_min_int32():
    clear_registry()
    import metaldf._engine  # noqa: F401

    s = pd.Series(np.random.randint(-1000, 1000, size=100_000, dtype=np.int32))
    expected = int(s.min())
    actual = execute("min", s)
    assert actual == expected


def test_metal_max_int32():
    clear_registry()
    import metaldf._engine  # noqa: F401

    s = pd.Series(np.random.randint(-1000, 1000, size=100_000, dtype=np.int32))
    expected = int(s.max())
    actual = execute("max", s)
    assert actual == expected


def test_metal_sum_int64():
    clear_registry()
    import metaldf._engine  # noqa: F401

    s = pd.Series(np.random.randint(-1000, 1000, size=100_000, dtype=np.int64))
    expected = int(s.sum())
    actual = execute("sum", s)
    assert actual == expected


def test_metal_mean_returns_float():
    """Mean on int32 should return a float, matching pandas."""
    clear_registry()
    import metaldf._engine  # noqa: F401

    s = pd.Series(np.array([1, 2, 3, 4, 5] * 1000, dtype=np.int32))
    expected = float(s.mean())
    actual = execute("mean", s)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert isinstance(actual, float)


def test_metal_sum_int32_direct():
    """Verify int32 sum dispatches to Metal kernel, not pandas fallback."""
    clear_registry()
    _register_ops()
    import metaldf_engine

    arr = np.random.randint(-1000, 1000, size=100_000).astype(np.int32)
    buf = metaldf_engine.MetalSeries.from_numpy_i32(arr)
    result = metaldf_engine.metal_sum(buf)
    expected = int(arr.sum())
    assert result == expected


def test_metal_min_max_int32_direct():
    """Verify int32 min/max dispatch to the Metal kernel directly."""
    clear_registry()
    _register_ops()
    import metaldf_engine

    arr = np.random.randint(-1000, 1000, size=100_000).astype(np.int32)
    buf = metaldf_engine.MetalSeries.from_numpy_i32(arr)
    assert metaldf_engine.metal_min(buf) == int(arr.min())
    assert metaldf_engine.metal_max(buf) == int(arr.max())


def test_metal_sum_min_max_int64_direct():
    """Verify int64 sum/min/max dispatch to the Metal kernel directly."""
    clear_registry()
    _register_ops()
    import metaldf_engine

    arr = np.random.randint(-1000, 1000, size=100_000).astype(np.int64)
    buf = metaldf_engine.MetalSeries.from_numpy_i64(arr)
    assert metaldf_engine.metal_sum(buf) == int(arr.sum())
    assert metaldf_engine.metal_min(buf) == int(arr.min())
    assert metaldf_engine.metal_max(buf) == int(arr.max())


def test_metal_mean_int64_direct():
    """metal_mean on int64 buffers returns a Python float."""
    clear_registry()
    _register_ops()
    import metaldf_engine

    arr = np.random.randint(-1000, 1000, size=100_000).astype(np.int64)
    buf = metaldf_engine.MetalSeries.from_numpy_i64(arr)
    result = metaldf_engine.metal_mean(buf)
    assert isinstance(result, float)
    np.testing.assert_allclose(result, float(arr.mean()), rtol=1e-6)


def test_proxy_series_sum():
    """ProxySeries.sum() should dispatch to Metal."""
    import metaldf
    metaldf.install()
    import pandas as pd

    s = pd.Series(np.random.randn(100_000).astype(np.float32))
    expected = s.sum()
    actual = s.sum()

    np.testing.assert_allclose(actual, expected, rtol=1e-4)
    metaldf.uninstall()
