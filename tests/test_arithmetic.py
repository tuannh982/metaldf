import platform

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Metal arithmetic tests only run on macOS",
)


def test_proxy_series_add():
    """ProxySeries __add__ falls back to pandas (Metal arithmetic removed)."""
    import metaldf

    metaldf.install()

    a = pd.Series(np.random.randn(100_000).astype(np.float32))
    b = pd.Series(np.random.randn(100_000).astype(np.float32))

    expected = a + b
    actual = a + b  # ProxySeries dunder dispatch

    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-5)
    metaldf.uninstall()


def test_proxy_series_sub():
    """ProxySeries __sub__ falls back to pandas."""
    import metaldf

    metaldf.install()

    a = pd.Series(np.random.randn(100_000).astype(np.float32))
    b = pd.Series(np.random.randn(100_000).astype(np.float32))

    expected = a - b
    actual = a - b

    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-5)
    metaldf.uninstall()


def test_proxy_series_mul():
    """ProxySeries __mul__ falls back to pandas."""
    import metaldf

    metaldf.install()

    a = pd.Series(np.random.randn(100_000).astype(np.float32))
    b = pd.Series(np.random.randn(100_000).astype(np.float32))

    expected = a * b
    actual = a * b

    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-5)
    metaldf.uninstall()


def test_proxy_series_div():
    """ProxySeries __truediv__ falls back to pandas."""
    import metaldf

    metaldf.install()

    a = pd.Series(np.random.randn(100_000).astype(np.float32))
    b = pd.Series(np.random.randn(100_000).astype(np.float32) + 1.0)

    expected = a / b
    actual = a / b

    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-5)
    metaldf.uninstall()


def test_proxy_series_reverse_sub():
    """ProxySeries __rsub__ must call pandas __rsub__, not __add__."""
    import metaldf

    metaldf.install()

    a = pd.Series(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    # 10.0 - a should give [9.0, 8.0, 7.0], NOT [11.0, 12.0, 13.0]
    result = 10.0 - a

    expected = pd.Series([9.0, 8.0, 7.0], dtype=np.float32)
    np.testing.assert_allclose(result.values, expected.values)
    metaldf.uninstall()


def test_proxy_series_int_add():
    """Integer arrays should work via pandas fallback."""
    import metaldf

    metaldf.install()

    a = pd.Series(np.array([1, 2, 3], dtype=np.int64))
    b = pd.Series(np.array([4, 5, 6], dtype=np.int64))

    expected = a + b
    actual = a + b
    np.testing.assert_array_equal(actual.values, expected.values)
    metaldf.uninstall()


def test_proxy_series_small_array_add():
    """Small arrays should work via pandas fallback."""
    import metaldf

    metaldf.install()

    a = pd.Series(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = pd.Series(np.array([4.0, 5.0, 6.0], dtype=np.float32))

    expected = a + b
    actual = a + b
    np.testing.assert_array_equal(actual.values, expected.values)
    metaldf.uninstall()
