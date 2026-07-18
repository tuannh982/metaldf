import platform
import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin", reason="Metal tests only run on macOS",
)

import metaldf_engine


def _to_bool_list(series):
    return [bool(x) for x in series.to_numpy()]


def test_isalpha():
    s = metaldf_engine.MetalSeries.from_strings(["hello", "hello123", "", "ABC"])
    result = metaldf_engine.metal_string_isalpha(s)
    assert _to_bool_list(result) == [True, False, False, True]


def test_isdigit():
    s = metaldf_engine.MetalSeries.from_strings(["123", "12a", "", "0"])
    result = metaldf_engine.metal_string_isdigit(s)
    assert _to_bool_list(result) == [True, False, False, True]


def test_isspace():
    s = metaldf_engine.MetalSeries.from_strings(["   ", " \t\n", "", "a "])
    result = metaldf_engine.metal_string_isspace(s)
    assert _to_bool_list(result) == [True, True, False, False]


def test_isalnum():
    s = metaldf_engine.MetalSeries.from_strings(["abc123", "abc!", "", "A1"])
    result = metaldf_engine.metal_string_isalnum(s)
    assert _to_bool_list(result) == [True, False, False, True]


def test_isupper():
    s = metaldf_engine.MetalSeries.from_strings(["ABC", "ABc", "123", "", "ABC123"])
    result = metaldf_engine.metal_string_isupper(s)
    assert _to_bool_list(result) == [True, False, False, False, True]


def test_islower():
    s = metaldf_engine.MetalSeries.from_strings(["abc", "aBc", "123", "", "abc123"])
    result = metaldf_engine.metal_string_islower(s)
    assert _to_bool_list(result) == [True, False, False, False, True]


def test_istitle():
    s = metaldf_engine.MetalSeries.from_strings(["Hello World", "hello world", "HELLO", "", "Hello"])
    result = metaldf_engine.metal_string_istitle(s)
    assert _to_bool_list(result) == [True, False, False, False, True]


def test_isnumeric():
    s = metaldf_engine.MetalSeries.from_strings(["123", "12a", ""])
    result = metaldf_engine.metal_string_isnumeric(s)
    assert _to_bool_list(result) == [True, False, False]


def test_isdecimal():
    s = metaldf_engine.MetalSeries.from_strings(["123", "12a", ""])
    result = metaldf_engine.metal_string_isdecimal(s)
    assert _to_bool_list(result) == [True, False, False]


def test_isalpha_boundary_bytes():
    s = metaldf_engine.MetalSeries.from_strings([
        "@",   # 0x40, just below 'A'
        "A",   # 0x41
        "[",   # 0x5B, just above 'Z'
        "`",   # 0x60, just below 'a'
        "a",   # 0x61
        "{",   # 0x7B, just above 'z'
    ])
    result = metaldf_engine.metal_string_isalpha(s)
    assert _to_bool_list(result) == [False, True, False, False, True, False]
