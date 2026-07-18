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


def test_proxy_str_len():
    data = ["hello", "hi", ""]
    s = ProxySeries(pd.Series(data))
    result = s.str.len()
    expected = pd.Series(data).str.len()
    pd.testing.assert_series_equal(result.to_pandas(), expected, check_dtype=False)


def test_proxy_str_count():
    data = ["abcabc", "abc", "xyz"]
    s = ProxySeries(pd.Series(data))
    result = s.str.count("abc")
    expected = pd.Series(data).str.count("abc")
    pd.testing.assert_series_equal(result.to_pandas(), expected, check_dtype=False)


def test_proxy_str_isalpha():
    data = ["hello", "hello123", "", "ABC"]
    s = ProxySeries(pd.Series(data))
    result = s.str.isalpha()
    expected = pd.Series(data).str.isalpha()
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_isdigit():
    data = ["123", "12a", "", "0"]
    s = ProxySeries(pd.Series(data))
    result = s.str.isdigit()
    expected = pd.Series(data).str.isdigit()
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_isspace():
    data = ["   ", " \t\n", "", "a "]
    s = ProxySeries(pd.Series(data))
    result = s.str.isspace()
    expected = pd.Series(data).str.isspace()
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_isupper():
    data = ["ABC", "ABc", "123", "", "ABC123"]
    s = ProxySeries(pd.Series(data))
    result = s.str.isupper()
    expected = pd.Series(data).str.isupper()
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_islower():
    data = ["abc", "aBc", "123", "", "abc123"]
    s = ProxySeries(pd.Series(data))
    result = s.str.islower()
    expected = pd.Series(data).str.islower()
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_istitle():
    data = ["Hello World", "hello world", "HELLO", "", "Hello"]
    s = ProxySeries(pd.Series(data))
    result = s.str.istitle()
    expected = pd.Series(data).str.istitle()
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_isalnum():
    data = ["abc123", "abc 123", "", "123"]
    s = ProxySeries(pd.Series(data))
    result = s.str.isalnum()
    expected = pd.Series(data).str.isalnum()
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_isnumeric():
    data = ["123", "12a", "", "0"]
    s = ProxySeries(pd.Series(data))
    result = s.str.isnumeric()
    expected = pd.Series(data).str.isnumeric()
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_isdecimal():
    data = ["123", "12a", "", "0"]
    s = ProxySeries(pd.Series(data))
    result = s.str.isdecimal()
    expected = pd.Series(data).str.isdecimal()
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_count_with_flags_falls_back():
    """kwargs like flags= change regex semantics -- must not go through Metal."""
    data = ["abcabc", "abc", "xyz"]
    s = ProxySeries(pd.Series(data))
    result = s.str.count("abc", flags=0)
    expected = pd.Series(data).str.count("abc", flags=0)
    pd.testing.assert_series_equal(result.to_pandas(), expected, check_dtype=False)


def test_proxy_str_swapcase():
    data = ["Hello", "WORLD", "foo123"]
    s = ProxySeries(pd.Series(data))
    result = s.str.swapcase()
    expected = pd.Series(data).str.swapcase()
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_title():
    data = ["hello world", "fOO BAR", "123abc"]
    s = ProxySeries(pd.Series(data))
    result = s.str.title()
    expected = pd.Series(data).str.title()
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_capitalize():
    data = ["hello world", "HELLO", "fOO", ""]
    s = ProxySeries(pd.Series(data))
    result = s.str.capitalize()
    expected = pd.Series(data).str.capitalize()
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_casefold():
    data = ["Hello", "WORLD", "FoO123"]
    s = ProxySeries(pd.Series(data))
    result = s.str.casefold()
    expected = pd.Series(data).str.casefold()
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_lstrip():
    data = ["  hello  ", "\tworld\n", "  foo"]
    s = ProxySeries(pd.Series(data))
    result = s.str.lstrip()
    expected = pd.Series(data).str.lstrip()
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_rstrip():
    data = ["  hello  ", "\tworld\n", "foo  "]
    s = ProxySeries(pd.Series(data))
    result = s.str.rstrip()
    expected = pd.Series(data).str.rstrip()
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_slice():
    data = ["hello", "world", "hi"]
    s = ProxySeries(pd.Series(data))
    result = s.str.slice(1, 3)
    expected = pd.Series(data).str.slice(1, 3)
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_slice_with_step_falls_back():
    data = ["hello", "world"]
    s = ProxySeries(pd.Series(data))
    result = s.str.slice(0, 5, 2)
    expected = pd.Series(data).str.slice(0, 5, 2)
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_get():
    data = ["hello", "world", ""]
    s = ProxySeries(pd.Series(data))
    result = s.str.get(0)
    expected = pd.Series(data).str.get(0)
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_repeat():
    data = ["ab", "cd", ""]
    s = ProxySeries(pd.Series(data))
    result = s.str.repeat(3)
    expected = pd.Series(data).str.repeat(3)
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_center():
    data = ["hi", "hello"]
    s = ProxySeries(pd.Series(data))
    result = s.str.center(6, '*')
    expected = pd.Series(data).str.center(6, '*')
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_ljust():
    data = ["hi", "hello"]
    s = ProxySeries(pd.Series(data))
    result = s.str.ljust(5, '-')
    expected = pd.Series(data).str.ljust(5, '-')
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_rjust():
    data = ["hi", "hello"]
    s = ProxySeries(pd.Series(data))
    result = s.str.rjust(5)
    expected = pd.Series(data).str.rjust(5)
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_zfill():
    data = ["42", "-42", "+5", "hello", ""]
    s = ProxySeries(pd.Series(data))
    result = s.str.zfill(5)
    expected = pd.Series(data).str.zfill(5)
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_removeprefix():
    data = ["hello world", "hello", "world"]
    s = ProxySeries(pd.Series(data))
    result = s.str.removeprefix("hello")
    expected = pd.Series(data).str.removeprefix("hello")
    pd.testing.assert_series_equal(result.to_pandas(), expected)


def test_proxy_str_removesuffix():
    data = ["hello world", "world", "hello"]
    s = ProxySeries(pd.Series(data))
    result = s.str.removesuffix("world")
    expected = pd.Series(data).str.removesuffix("world")
    pd.testing.assert_series_equal(result.to_pandas(), expected)


import random
import string as string_module


def _random_ascii_strings(n, max_len=20, seed=42):
    random.seed(seed)
    chars = string_module.ascii_letters + string_module.digits + " \t\n"
    return [
        "".join(random.choices(chars, k=random.randint(0, max_len)))
        for _ in range(n)
    ]


def test_proxy_str_ops_10k_match_pandas():
    data = _random_ascii_strings(10000)
    ps = pd.Series(data)
    ms = ProxySeries(ps)

    # len
    pd.testing.assert_series_equal(
        ms.str.len().to_pandas(), ps.str.len(), check_dtype=False
    )

    # count
    pd.testing.assert_series_equal(
        ms.str.count("a").to_pandas(), ps.str.count("a"), check_dtype=False
    )

    # isalpha
    pd.testing.assert_series_equal(
        ms.str.isalpha().to_pandas(), ps.str.isalpha()
    )

    # isdigit / isspace / isalnum / isupper / islower / istitle / isnumeric / isdecimal
    for op in [
        "isdigit", "isspace", "isalnum", "isupper", "islower",
        "istitle", "isnumeric", "isdecimal",
    ]:
        pd.testing.assert_series_equal(
            getattr(ms.str, op)().to_pandas(),
            getattr(ps.str, op)(),
            obj=op,
        )

    # lower / upper / swapcase / title / capitalize / casefold
    for op in ["lower", "upper", "swapcase", "title", "capitalize", "casefold"]:
        pd.testing.assert_series_equal(
            getattr(ms.str, op)().to_pandas(),
            getattr(ps.str, op)(),
            obj=op,
        )

    # strip / lstrip / rstrip
    for op in ["strip", "lstrip", "rstrip"]:
        pd.testing.assert_series_equal(
            getattr(ms.str, op)().to_pandas(),
            getattr(ps.str, op)(),
            obj=op,
        )

    # slice
    pd.testing.assert_series_equal(
        ms.str.slice(2, 8).to_pandas(), ps.str.slice(2, 8)
    )

    # repeat
    pd.testing.assert_series_equal(
        ms.str.repeat(2).to_pandas(), ps.str.repeat(2)
    )

    # zfill
    pd.testing.assert_series_equal(
        ms.str.zfill(25).to_pandas(), ps.str.zfill(25)
    )

    # get
    for i in [0, -1]:
        pd.testing.assert_series_equal(
            ms.str.get(i).to_pandas(), ps.str.get(i), obj=f"get({i})"
        )

    # center / ljust / rjust
    for op in ["center", "ljust", "rjust"]:
        pd.testing.assert_series_equal(
            getattr(ms.str, op)(25).to_pandas(),
            getattr(ps.str, op)(25),
            obj=op,
        )

    # removeprefix / removesuffix
    pd.testing.assert_series_equal(
        ms.str.removeprefix("a").to_pandas(), ps.str.removeprefix("a")
    )
    pd.testing.assert_series_equal(
        ms.str.removesuffix("a").to_pandas(), ps.str.removesuffix("a")
    )
