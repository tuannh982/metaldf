import platform
import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Metal tests only run on macOS",
)

import metaldf_engine


def test_string_roundtrip():
    strings = ["hello", "world", "!"]
    series = metaldf_engine.MetalSeries.from_strings(strings)
    assert series.len == 3
    assert series.dtype == "Utf8"
    result = series.to_strings()
    assert result == strings


def test_empty_strings():
    strings = ["", "", ""]
    series = metaldf_engine.MetalSeries.from_strings(strings)
    assert series.len == 3
    result = series.to_strings()
    assert result == strings


def test_mixed_lengths():
    strings = ["a", "hello world", ""]
    series = metaldf_engine.MetalSeries.from_strings(strings)
    result = series.to_strings()
    assert result == strings


def test_unicode_roundtrip():
    strings = ["café", "éclair", "\U0001f600"]
    series = metaldf_engine.MetalSeries.from_strings(strings)
    result = series.to_strings()
    assert result == strings


def test_large_strings():
    strings = [f"string_{i:06d}" for i in range(10_000)]
    series = metaldf_engine.MetalSeries.from_strings(strings)
    assert series.len == 10_000
    result = series.to_strings()
    assert result == strings
