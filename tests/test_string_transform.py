import platform
import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin", reason="Metal tests only run on macOS",
)

import metaldf_engine

def test_lower():
    s = metaldf_engine.MetalSeries.from_strings(["Hello", "WORLD", "FoO123"])
    result = metaldf_engine.metal_string_lower(s)
    assert result.to_strings() == ["hello", "world", "foo123"]

def test_upper():
    s = metaldf_engine.MetalSeries.from_strings(["Hello", "world", "FoO123"])
    result = metaldf_engine.metal_string_upper(s)
    assert result.to_strings() == ["HELLO", "WORLD", "FOO123"]

def test_strip():
    s = metaldf_engine.MetalSeries.from_strings(["  hello  ", "\tworld\n", "  foo"])
    result = metaldf_engine.metal_string_strip(s)
    assert result.to_strings() == ["hello", "world", "foo"]

def test_strip_empty():
    s = metaldf_engine.MetalSeries.from_strings(["   ", "", "  a  "])
    result = metaldf_engine.metal_string_strip(s)
    assert result.to_strings() == ["", "", "a"]

def test_lower_preserves_non_ascii():
    s = metaldf_engine.MetalSeries.from_strings(["Café", "\U0001f600ABC"])
    result = metaldf_engine.metal_string_lower(s)
    assert result.to_strings() == ["café", "\U0001f600abc"]


def test_replace():
    s = metaldf_engine.MetalSeries.from_strings(["hello world", "foo world bar", "no match"])
    result = metaldf_engine.metal_string_replace(s, "world", "earth")
    assert result.to_strings() == ["hello earth", "foo earth bar", "no match"]


def test_replace_no_match():
    s = metaldf_engine.MetalSeries.from_strings(["abc", "def"])
    result = metaldf_engine.metal_string_replace(s, "xyz", "!!!")
    assert result.to_strings() == ["abc", "def"]


def test_replace_pattern_longer_than_string():
    s = metaldf_engine.MetalSeries.from_strings(["ab", "cd", "longer than pattern"])
    result = metaldf_engine.metal_string_replace(s, "xyzzy", "Q")
    assert result.to_strings() == ["ab", "cd", "longer than pattern"]


def test_swapcase():
    s = metaldf_engine.MetalSeries.from_strings(["Hello", "WORLD", "foo123"])
    result = metaldf_engine.metal_string_swapcase(s)
    assert result.to_strings() == ["hELLO", "world", "FOO123"]


def test_title():
    s = metaldf_engine.MetalSeries.from_strings(["hello world", "fOO BAR", "123abc"])
    result = metaldf_engine.metal_string_title(s)
    assert result.to_strings() == ["Hello World", "Foo Bar", "123Abc"]


def test_title_single_word():
    s = metaldf_engine.MetalSeries.from_strings(["hello", "HELLO", ""])
    result = metaldf_engine.metal_string_title(s)
    assert result.to_strings() == ["Hello", "Hello", ""]


def test_capitalize():
    s = metaldf_engine.MetalSeries.from_strings(["hello world", "HELLO", "fOO", ""])
    result = metaldf_engine.metal_string_capitalize(s)
    assert result.to_strings() == ["Hello world", "Hello", "Foo", ""]


def test_casefold():
    s = metaldf_engine.MetalSeries.from_strings(["Hello", "WORLD", "FoO123"])
    result = metaldf_engine.metal_string_casefold(s)
    assert result.to_strings() == ["hello", "world", "foo123"]


def test_lstrip():
    s = metaldf_engine.MetalSeries.from_strings(["  hello  ", "\tworld\n", "  foo"])
    result = metaldf_engine.metal_string_lstrip(s)
    assert result.to_strings() == ["hello  ", "world\n", "foo"]


def test_lstrip_empty():
    s = metaldf_engine.MetalSeries.from_strings(["   ", "", "  a  "])
    result = metaldf_engine.metal_string_lstrip(s)
    assert result.to_strings() == ["", "", "a  "]


def test_rstrip():
    s = metaldf_engine.MetalSeries.from_strings(["  hello  ", "\tworld\n", "foo  "])
    result = metaldf_engine.metal_string_rstrip(s)
    assert result.to_strings() == ["  hello", "\tworld", "foo"]


def test_rstrip_empty():
    s = metaldf_engine.MetalSeries.from_strings(["   ", "", "  a  "])
    result = metaldf_engine.metal_string_rstrip(s)
    assert result.to_strings() == ["", "", "  a"]
