import platform
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Metal tests only run on macOS",
)

import metaldf_engine


def _to_int_list(series):
    return list(series.to_numpy())


def test_string_eq():
    a = metaldf_engine.MetalSeries.from_strings(["hello", "world", "foo"])
    b = metaldf_engine.MetalSeries.from_strings(["hello", "bar", "foo"])
    result = metaldf_engine.metal_string_eq(a, b)
    assert _to_int_list(result) == [1, 0, 1]


def test_string_ne():
    a = metaldf_engine.MetalSeries.from_strings(["hello", "world", "foo"])
    b = metaldf_engine.MetalSeries.from_strings(["hello", "bar", "foo"])
    result = metaldf_engine.metal_string_ne(a, b)
    assert _to_int_list(result) == [0, 1, 0]


def test_string_lt():
    a = metaldf_engine.MetalSeries.from_strings(["apple", "banana", "cherry"])
    b = metaldf_engine.MetalSeries.from_strings(["banana", "apple", "cherry"])
    result = metaldf_engine.metal_string_lt(a, b)
    assert _to_int_list(result) == [1, 0, 0]


def test_string_gt():
    a = metaldf_engine.MetalSeries.from_strings(["banana", "apple", "cherry"])
    b = metaldf_engine.MetalSeries.from_strings(["apple", "banana", "cherry"])
    result = metaldf_engine.metal_string_gt(a, b)
    assert _to_int_list(result) == [1, 0, 0]


def test_string_eq_scalar():
    series = metaldf_engine.MetalSeries.from_strings(["foo", "bar", "foo", "baz"])
    result = metaldf_engine.metal_string_eq_scalar(series, "foo")
    assert _to_int_list(result) == [1, 0, 1, 0]


def test_string_eq_empty_strings():
    a = metaldf_engine.MetalSeries.from_strings(["", "a", ""])
    b = metaldf_engine.MetalSeries.from_strings(["", "b", ""])
    result = metaldf_engine.metal_string_eq(a, b)
    assert _to_int_list(result) == [1, 0, 1]
