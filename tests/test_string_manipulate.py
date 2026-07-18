import platform
import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin", reason="Metal tests only run on macOS",
)

import metaldf_engine


def test_slice_basic():
    s = metaldf_engine.MetalSeries.from_strings(["hello", "world", "hi"])
    result = metaldf_engine.metal_string_slice(s, 1, 3)
    assert result.to_strings() == ["el", "or", "i"]


def test_slice_clamp():
    s = metaldf_engine.MetalSeries.from_strings(["hi", "hello"])
    result = metaldf_engine.metal_string_slice(s, 0, 100)
    assert result.to_strings() == ["hi", "hello"]


def test_slice_negative_start():
    s = metaldf_engine.MetalSeries.from_strings(["hello", "hi"])
    result = metaldf_engine.metal_string_slice(s, -3, 100)
    assert result.to_strings() == ["llo", "hi"]


def test_slice_empty():
    s = metaldf_engine.MetalSeries.from_strings(["hello", ""])
    result = metaldf_engine.metal_string_slice(s, 2, 4)
    assert result.to_strings() == ["ll", ""]


def test_get_basic():
    s = metaldf_engine.MetalSeries.from_strings(["hello", "world", ""])
    result = metaldf_engine.metal_string_get(s, 0)
    assert result.to_strings() == ["h", "w", ""]


def test_get_negative():
    s = metaldf_engine.MetalSeries.from_strings(["hello", "hi"])
    result = metaldf_engine.metal_string_get(s, -1)
    assert result.to_strings() == ["o", "i"]


def test_get_out_of_bounds():
    s = metaldf_engine.MetalSeries.from_strings(["hi", "hello"])
    result = metaldf_engine.metal_string_get(s, 10)
    assert result.to_strings() == ["", ""]


def test_repeat_basic():
    s = metaldf_engine.MetalSeries.from_strings(["ab", "cd", ""])
    result = metaldf_engine.metal_string_repeat(s, 3)
    assert result.to_strings() == ["ababab", "cdcdcd", ""]


def test_repeat_zero():
    s = metaldf_engine.MetalSeries.from_strings(["hello", "world"])
    result = metaldf_engine.metal_string_repeat(s, 0)
    assert result.to_strings() == ["", ""]


def test_repeat_one():
    s = metaldf_engine.MetalSeries.from_strings(["hello"])
    result = metaldf_engine.metal_string_repeat(s, 1)
    assert result.to_strings() == ["hello"]


def test_pad_left():
    s = metaldf_engine.MetalSeries.from_strings(["hi", "hello", "a"])
    result = metaldf_engine.metal_string_pad(s, 5, 0, ord(' '))
    assert result.to_strings() == ["   hi", "hello", "    a"]


def test_pad_right():
    s = metaldf_engine.MetalSeries.from_strings(["hi", "hello"])
    result = metaldf_engine.metal_string_pad(s, 5, 1, ord('-'))
    assert result.to_strings() == ["hi---", "hello"]


def test_pad_both():
    s = metaldf_engine.MetalSeries.from_strings(["hi", "hello"])
    result = metaldf_engine.metal_string_pad(s, 6, 2, ord('*'))
    assert result.to_strings() == ["**hi**", "hello*"]


def test_pad_no_change():
    s = metaldf_engine.MetalSeries.from_strings(["hello"])
    result = metaldf_engine.metal_string_pad(s, 3, 0, ord(' '))
    assert result.to_strings() == ["hello"]


def test_zfill():
    s = metaldf_engine.MetalSeries.from_strings(["42", "-42", "+5", "hello", ""])
    result = metaldf_engine.metal_string_zfill(s, 5)
    assert result.to_strings() == ["00042", "-0042", "+0005", "hello", "00000"]


def test_removeprefix():
    s = metaldf_engine.MetalSeries.from_strings(["hello world", "hello", "world"])
    result = metaldf_engine.metal_string_removeprefix(s, "hello")
    assert result.to_strings() == [" world", "", "world"]


def test_removeprefix_no_match():
    s = metaldf_engine.MetalSeries.from_strings(["abc", "def"])
    result = metaldf_engine.metal_string_removeprefix(s, "xyz")
    assert result.to_strings() == ["abc", "def"]


def test_removesuffix():
    s = metaldf_engine.MetalSeries.from_strings(["hello world", "world", "hello"])
    result = metaldf_engine.metal_string_removesuffix(s, "world")
    assert result.to_strings() == ["hello ", "", "hello"]


def test_removesuffix_no_match():
    s = metaldf_engine.MetalSeries.from_strings(["abc", "def"])
    result = metaldf_engine.metal_string_removesuffix(s, "xyz")
    assert result.to_strings() == ["abc", "def"]
