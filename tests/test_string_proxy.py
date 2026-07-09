import platform
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin", reason="Metal tests only run on macOS",
)

from metaldf._wrappers import ProxySeries


def test_proxy_str_contains():
    data = ["hello world"] * 2000 + ["foobar"] * 2000
    s = ProxySeries(pd.Series(data))
    result = s.str.contains("world")
    expected = pd.Series(data).str.contains("world")
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_lower():
    data = ["HELLO"] * 2000 + ["World"] * 2000
    s = ProxySeries(pd.Series(data))
    result = s.str.lower()
    expected = pd.Series(data).str.lower()
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_fallback_small():
    data = ["hello", "world"]
    s = ProxySeries(pd.Series(data))
    result = s.str.upper()
    expected = pd.Series(data).str.upper()
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_startswith():
    data = ["hello world"] * 2000 + ["foobar"] * 2000
    s = ProxySeries(pd.Series(data))
    result = s.str.startswith("hello")
    expected = pd.Series(data).str.startswith("hello")
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_endswith():
    data = ["hello world"] * 2000 + ["foobar"] * 2000
    s = ProxySeries(pd.Series(data))
    result = s.str.endswith("world")
    expected = pd.Series(data).str.endswith("world")
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_upper():
    data = ["hello"] * 2000 + ["World"] * 2000
    s = ProxySeries(pd.Series(data))
    result = s.str.upper()
    expected = pd.Series(data).str.upper()
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_strip():
    data = ["  hello  "] * 2000 + ["\tworld\n"] * 2000
    s = ProxySeries(pd.Series(data))
    result = s.str.strip()
    expected = pd.Series(data).str.strip()
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_find():
    data = ["hello world"] * 2000 + ["foobar"] * 2000
    s = ProxySeries(pd.Series(data))
    result = s.str.find("world")
    expected = pd.Series(data).str.find("world")
    pd.testing.assert_series_equal(result.to_pandas(), expected, check_dtype=False)


def test_proxy_str_contains_case_insensitive_falls_back():
    """kwargs like case=False change semantics -- must not go through Metal."""
    data = ["Hello World"] * 2000 + ["foobar"] * 2000
    s = ProxySeries(pd.Series(data))
    result = s.str.contains("world", case=False)
    expected = pd.Series(data).str.contains("world", case=False)
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_unsupported_method_falls_back():
    """Methods without a Metal kernel (e.g. split) fall through to pandas."""
    data = ["a,b", "c,d"]
    s = ProxySeries(pd.Series(data))
    result = s.str.split(",")
    expected = pd.Series(data).str.split(",")
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_replace_literal_uses_metal():
    """regex=False (literal replacement) is the only path Metal supports."""
    data = ["hello world"] * 2000 + ["foo world bar"] * 2000 + ["no match"] * 2000
    s = ProxySeries(pd.Series(data))
    result = s.str.replace("world", "earth", regex=False)
    expected = pd.Series(data).str.replace("world", "earth", regex=False)
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_replace_default_regex_falls_back():
    """pandas' default is regex=True -- must not go through the literal Metal kernel."""
    data = ["hello world"] * 2000 + ["foo1 world bar"] * 2000
    s = ProxySeries(pd.Series(data))
    result = s.str.replace(r"\d", "#", regex=True)
    expected = pd.Series(data).str.replace(r"\d", "#", regex=True)
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_replace_small_array():
    """Small arrays now dispatch to Metal too (no size threshold) and stay correct."""
    data = ["hello world", "foo world bar", "no match"]
    s = ProxySeries(pd.Series(data))
    result = s.str.replace("world", "earth", regex=False)
    expected = pd.Series(data).str.replace("world", "earth", regex=False)
    pd.testing.assert_series_equal(result.to_pandas(), expected)
