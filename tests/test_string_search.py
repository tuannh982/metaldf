import platform
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin", reason="Metal tests only run on macOS",
)

import metaldf_engine

def _to_int_list(series):
    return list(series.to_numpy())

def test_contains():
    s = metaldf_engine.MetalSeries.from_strings(["hello world", "foobar", "world cup"])
    result = metaldf_engine.metal_string_contains(s, "world")
    assert _to_int_list(result) == [1, 0, 1]

def test_contains_empty_pattern():
    s = metaldf_engine.MetalSeries.from_strings(["hello", ""])
    result = metaldf_engine.metal_string_contains(s, "")
    assert _to_int_list(result) == [1, 1]

def test_startswith():
    s = metaldf_engine.MetalSeries.from_strings(["hello world", "foobar", "help"])
    result = metaldf_engine.metal_string_startswith(s, "hel")
    assert _to_int_list(result) == [1, 0, 1]

def test_endswith():
    s = metaldf_engine.MetalSeries.from_strings(["hello world", "foobar", "new world"])
    result = metaldf_engine.metal_string_endswith(s, "world")
    assert _to_int_list(result) == [1, 0, 1]

def test_find():
    s = metaldf_engine.MetalSeries.from_strings(["hello world", "foobar", "world"])
    result = metaldf_engine.metal_string_find(s, "world")
    assert _to_int_list(result) == [6, -1, 0]

def test_find_not_found():
    s = metaldf_engine.MetalSeries.from_strings(["abc", "def"])
    result = metaldf_engine.metal_string_find(s, "xyz")
    assert _to_int_list(result) == [-1, -1]
