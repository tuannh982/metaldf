import platform
import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin", reason="Metal tests only run on macOS",
)

import metaldf_engine


def test_len_basic():
    s = metaldf_engine.MetalSeries.from_strings(["hello", "hi", ""])
    result = metaldf_engine.metal_string_len(s)
    assert list(result.to_numpy()) == [5, 2, 0]


def test_len_single_char():
    s = metaldf_engine.MetalSeries.from_strings(["a", "b", "c"])
    result = metaldf_engine.metal_string_len(s)
    assert list(result.to_numpy()) == [1, 1, 1]


def test_count_basic():
    s = metaldf_engine.MetalSeries.from_strings(["abcabc", "abc", "xyz"])
    result = metaldf_engine.metal_string_count(s, "abc")
    assert list(result.to_numpy()) == [2, 1, 0]


def test_count_empty_pattern():
    s = metaldf_engine.MetalSeries.from_strings(["hello", ""])
    result = metaldf_engine.metal_string_count(s, "")
    # pandas: empty pattern counts len+1 positions
    assert list(result.to_numpy()) == [6, 1]


def test_count_no_overlap():
    s = metaldf_engine.MetalSeries.from_strings(["aaa"])
    result = metaldf_engine.metal_string_count(s, "aa")
    # Non-overlapping: "aaa" has 1 non-overlapping "aa"
    assert list(result.to_numpy()) == [1]


def test_count_pattern_longer_than_string():
    s = metaldf_engine.MetalSeries.from_strings(["ab", "abcdef"])
    result = metaldf_engine.metal_string_count(s, "abcd")
    assert list(result.to_numpy()) == [0, 1]
