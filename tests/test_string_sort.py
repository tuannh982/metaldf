import platform
import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin", reason="Metal tests only run on macOS",
)

import metaldf_engine

def test_sort_ascending():
    s = metaldf_engine.MetalSeries.from_strings(["cherry", "apple", "banana"])
    result = metaldf_engine.metal_string_sort(s, True)
    assert result.to_strings() == ["apple", "banana", "cherry"]

def test_sort_descending():
    s = metaldf_engine.MetalSeries.from_strings(["cherry", "apple", "banana"])
    result = metaldf_engine.metal_string_sort(s, False)
    assert result.to_strings() == ["cherry", "banana", "apple"]

def test_sort_with_empty_strings():
    s = metaldf_engine.MetalSeries.from_strings(["b", "", "a", ""])
    result = metaldf_engine.metal_string_sort(s, True)
    assert result.to_strings() == ["", "", "a", "b"]

def test_sort_duplicates():
    s = metaldf_engine.MetalSeries.from_strings(["foo", "bar", "foo", "baz", "bar"])
    result = metaldf_engine.metal_string_sort(s, True)
    assert result.to_strings() == ["bar", "bar", "baz", "foo", "foo"]

def test_sort_single():
    s = metaldf_engine.MetalSeries.from_strings(["only"])
    result = metaldf_engine.metal_string_sort(s, True)
    assert result.to_strings() == ["only"]

def test_sort_matches_python():
    import random
    random.seed(42)
    strings = [f"str_{random.randint(0, 999):03d}" for _ in range(1000)]
    s = metaldf_engine.MetalSeries.from_strings(strings)
    result = metaldf_engine.metal_string_sort(s, True)
    assert result.to_strings() == sorted(strings)
