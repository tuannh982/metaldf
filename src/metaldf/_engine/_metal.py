"""Python bridge to the Rust Metal engine.

Imports the PyO3 extension module and provides a Python-friendly API
for buffer management and kernel dispatch (sum/mean/min/max, sort/argsort,
groupby aggregations, and vectorized string ops) via ``MetalEngine``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import NDArray

try:
    import metaldf_engine

    _METAL_AVAILABLE = True
except ImportError:
    _METAL_AVAILABLE = False
    metaldf_engine = None  # type: ignore[assignment]

from metaldf.exceptions import MetalNotAvailable


def is_metal_available() -> bool:
    """Return True if the Metal Rust extension is loaded and functional."""
    return _METAL_AVAILABLE


_SUPPORTED_DTYPES = {np.dtype(np.float32), np.dtype(np.int32), np.dtype(np.int64)}

_FROM_NUMPY = {
    np.dtype('float32'): lambda arr: metaldf_engine.MetalSeries.from_numpy(arr),
    np.dtype('int32'):   lambda arr: metaldf_engine.MetalSeries.from_numpy_i32(arr),
    np.dtype('int64'):   lambda arr: metaldf_engine.MetalSeries.from_numpy_i64(arr),
}


def _extract_array(data: Any) -> NDArray:
    """Get a C-contiguous numpy array without copying if possible.

    Handles pandas Series (data.values) and numpy arrays. Preserves the
    original dtype (float32/int32/int64) instead of coercing everything
    to float32.
    """
    # pandas Series: use _values (writeable) instead of values (read-only).
    # np.asarray on a read-only array silently copies even when dtype matches.
    if isinstance(data, pd.Series):
        vals = data._values
    else:
        vals = data
    arr = np.asarray(vals)

    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)

    return arr


def _make_series(arr: NDArray) -> object:
    """Create a Rust MetalSeries from a numpy array with the correct dtype."""
    ctor = _FROM_NUMPY.get(arr.dtype)
    if ctor is None:
        raise MetalNotAvailable(f"Unsupported dtype: {arr.dtype}")
    return ctor(arr)


def _has_metal() -> bool:
    """Check if Metal is available."""
    return _METAL_AVAILABLE


# ---------------------------------------------------------------------------
# Sort kernel dtype support
# ---------------------------------------------------------------------------

# NOTE: use np.dtype(...) instances here, not bare scalar types (np.float32).
# In this numpy version, `pandas_series.dtype == np.float32` is True but their
# `hash()` differs, so `dtype in {np.float32, ...}` (set/dict membership,
# which is hash-based) silently returns False even for a matching dtype —
# causing every Metal sort call to raise MetalNotAvailable and fall back to
# `np.sort`/`np.argsort` (masked by PandasEngine; see _pandas.py). Comparing
# dtype instances to dtype instances (as done here) hashes consistently.
_SORT_DTYPES = {np.dtype(np.float32), np.dtype(np.int32), np.dtype(np.int64)}

# ---------------------------------------------------------------------------
# GroupBy kernel dtype support
# ---------------------------------------------------------------------------

# NOTE: use np.dtype(...) instances here, not bare scalar types — see the
# _SORT_DTYPES comment above for why comparing dtype instances against bare
# numpy scalar types silently breaks set/dict membership checks.
_GROUPBY_DTYPES = {np.dtype(np.float32), np.dtype(np.int32)}


def _dispatch_reduction(op_name: str, data: Any) -> Any:
    """Try Metal reduction, fall back to pandas on any error."""
    if not _has_metal():
        raise MetalNotAvailable("Metal not available")

    from metaldf._wrappers import ProxySeries
    if isinstance(data, ProxySeries) and hasattr(data, "_pandas_obj"):
        data = data.to_pandas()

    if not hasattr(data, "dtype"):
        raise MetalNotAvailable("Operand must be a pandas Series or numpy array")

    if data.dtype not in _SUPPORTED_DTYPES:
        raise MetalNotAvailable(f"Unsupported dtype: {data.dtype}")

    arr = _extract_array(data)
    buf = _make_series(arr)
    rust_fn = getattr(metaldf_engine, f"metal_{op_name}")
    return rust_fn(buf)



# ---------------------------------------------------------------------------
# Elementwise binary op dtype support
# ---------------------------------------------------------------------------

# NOTE: use np.dtype(...) instances, not bare numpy scalar types -- see the
# _SORT_DTYPES comment above for why comparing dtype instances to bare
# numpy scalar types silently breaks set/dict membership checks.
_ELEMENTWISE_DTYPES = {np.dtype(np.float32), np.dtype(np.int32), np.dtype(np.int64)}

# pandas' true division (`/` / `__truediv__`) always promotes int Series to
# float64 (e.g. 10 / 3 == 3.333...). The Metal `div` kernel does same-dtype
# division -- floor division for int32/int64 (see
# tests/test_elementwise.py::TestBinaryOps::test_binary_i32, where the
# expected value for integer "div" is `a // b`, not `a / b`). Routing an
# int __truediv__ to Metal would therefore silently return the wrong
# (floor-divided) integers instead of pandas' float result, so "div" is
# restricted to float32, where Metal's division and pandas' true division
# agree.
_TRUEDIV_DTYPES = {np.dtype(np.float32)}


def _dispatch_binary(op_name: str, a: Any, b: Any) -> Any:
    """Try a Metal elementwise binary op (add/sub/mul/div), raising
    ``MetalNotAvailable`` for any condition Metal can't -- or, for
    integer true-division, *shouldn't* -- handle, so the caller falls back
    to pandas.
    """
    if not _has_metal():
        raise MetalNotAvailable("Metal not available")

    from metaldf._wrappers import ProxySeries
    if isinstance(a, ProxySeries) and hasattr(a, "_pandas_obj"):
        a = a.to_pandas()
    if isinstance(b, ProxySeries) and hasattr(b, "_pandas_obj"):
        b = b.to_pandas()

    if not hasattr(a, "dtype") or not hasattr(b, "dtype"):
        raise MetalNotAvailable("Operands must be pandas Series or numpy arrays")

    if a.dtype not in _ELEMENTWISE_DTYPES or b.dtype not in _ELEMENTWISE_DTYPES:
        raise MetalNotAvailable(f"Unsupported dtype: {a.dtype}, {b.dtype}")

    if a.dtype != b.dtype:
        raise MetalNotAvailable(f"dtype mismatch: {a.dtype} vs {b.dtype}")

    if op_name == "div" and a.dtype not in _TRUEDIV_DTYPES:
        raise MetalNotAvailable(
            f"True division of {a.dtype} needs float promotion; "
            "Metal 'div' does same-dtype (floor) division for integers"
        )

    # Metal has no notion of pandas' index-alignment semantics -- it just
    # zips positionally. Only dispatch when both operands already share the
    # same index (the overwhelmingly common case), otherwise fall back to
    # pandas so alignment (union index, NaN-filling on mismatches) happens.
    a_index = getattr(a, "index", None)
    b_index = getattr(b, "index", None)
    if a_index is not None and b_index is not None and not a_index.equals(b_index):
        raise MetalNotAvailable("Index mismatch requires pandas-side alignment")

    arr_a = _extract_array(a)
    arr_b = _extract_array(b)
    buf_a = _make_series(arr_a)
    buf_b = _make_series(arr_b)
    result = metaldf_engine.metal_binary_op(op_name, buf_a, buf_b)

    # Match pandas' name-inference rule for binary ops: if `b` is also a
    # Series, the result keeps the name only when both sides agree (e.g.
    # `pd.Series(..., name="x") + pd.Series(..., name="y")` -> name=None);
    # if `b` is a scalar/array (no `name` attribute), `a`'s name passes
    # through unchanged.
    a_name = getattr(a, "name", None)
    name = a_name if not hasattr(b, "name") or a_name == b.name else None
    return pd.Series(result.to_numpy(), index=a_index, name=name)


def _groupby_dispatch(agg_name: str, keys: Any, values: Any) -> Any:
    """Generic groupby dispatch to Metal."""
    if not _has_metal():
        raise MetalNotAvailable("Metal not available")

    from metaldf._wrappers import ProxySeries
    if isinstance(keys, ProxySeries) and hasattr(keys, "_pandas_obj"):
        keys = keys.to_pandas()
    if isinstance(values, ProxySeries) and hasattr(values, "_pandas_obj"):
        values = values.to_pandas()

    if not hasattr(keys, "dtype") or not hasattr(values, "dtype"):
        raise MetalNotAvailable("Keys and values must be pandas Series")

    if keys.dtype not in _GROUPBY_DTYPES or values.dtype not in _GROUPBY_DTYPES:
        raise MetalNotAvailable(f"Groupby not supported for key={keys.dtype} value={values.dtype}")

    # The Rust side only supports matching key/value dtypes (Float32/Float32
    # or Int32/Int32). Checking that here — rather than letting a mismatched
    # pair reach metaldf_engine — keeps the graceful-fallback-to-pandas
    # contract: every dtype combination metaldf_engine itself rejects should
    # already have been filtered out by this Python-level check.
    if keys.dtype != values.dtype:
        raise MetalNotAvailable(f"Groupby requires matching key/value dtypes, got key={keys.dtype} value={values.dtype}")

    keys_arr = _extract_array(keys)
    vals_arr = _extract_array(values)
    keys_buf = _make_series(keys_arr)
    vals_buf = _make_series(vals_arr)
    rust_fn = getattr(metaldf_engine, f"metal_groupby_{agg_name}")
    result = rust_fn(keys_buf, vals_buf)
    unique_keys, agg_values = result
    index = pd.Index(unique_keys.to_numpy(), name=getattr(keys, "name", None))
    return pd.Series(agg_values.to_numpy(), index=index, name=getattr(values, "name", None))


# ---------------------------------------------------------------------------
# String kernel helpers
# ---------------------------------------------------------------------------

def _is_string_dtype(data: Any) -> bool:
    """Return True if `data` holds pandas string data.

    Pandas string columns may be ``object`` dtype (pre-3.0 default, or an
    explicit ``dtype=object``) or the newer ``StringDtype`` (pandas>=3.0
    default for string literals). ``pd.api.types.is_string_dtype`` covers
    both, unlike a bare ``dtype == object`` check.
    """
    if not hasattr(data, "dtype"):
        return False
    return bool(pd.api.types.is_string_dtype(data))


def _make_string_series(data: Any) -> object:
    """Create a MetalSeries from a pandas string Series."""
    from metaldf._wrappers import ProxySeries
    if isinstance(data, ProxySeries) and hasattr(data, "_pandas_obj"):
        data = data.to_pandas()
    return metaldf_engine.MetalSeries.from_strings(list(data))


def _string_dtype_error(data: Any) -> MetalNotAvailable:
    return MetalNotAvailable(
        f"Unsupported dtype for string op: {getattr(data, 'dtype', type(data))}"
    )


def _dispatch_string_bool(rust_fn_name: str, data: Any, pat: str) -> Any:
    """Shared dispatch for contains/startswith/endswith (Int32 0/1 -> bool)."""
    if not _has_metal():
        raise MetalNotAvailable("Metal not available")
    if not _is_string_dtype(data):
        raise _string_dtype_error(data)
    series = _make_string_series(data)
    rust_fn = getattr(metaldf_engine, rust_fn_name)
    result = rust_fn(series, pat)
    return pd.Series(result.to_numpy().astype(bool), index=data.index, name=getattr(data, "name", None))


def _dispatch_string_transform(rust_fn_name: str, data: Any) -> Any:
    """Shared dispatch for lower/upper/strip (Utf8 -> Utf8)."""
    if not _has_metal():
        raise MetalNotAvailable("Metal not available")
    if not _is_string_dtype(data):
        raise _string_dtype_error(data)
    series = _make_string_series(data)
    rust_fn = getattr(metaldf_engine, rust_fn_name)
    result = rust_fn(series)
    return pd.Series(result.to_strings(), index=data.index, name=getattr(data, "name", None))


class MetalEngine:
    """Dispatches operations to the Metal GPU backend.

    Mirrors ``PandasEngine``'s shape: one staticmethod per operation, each
    raising ``MetalNotAvailable`` (caught by ``metaldf._engine.execute``) to
    signal that the caller should fall back to pandas. GPU dispatch happens
    regardless of input size -- there is no small-array threshold here; the
    kernels themselves are correct for any length, including zero/tiny
    arrays (see the direct-kernel tests in ``tests/test_metal_bridge.py`` and
    friends).
    """

    # -- Reductions ---------------------------------------------------------

    @staticmethod
    def metal_sum(data: Any) -> float:
        return _dispatch_reduction("sum", data)

    @staticmethod
    def metal_min(data: Any) -> float:
        return _dispatch_reduction("min", data)

    @staticmethod
    def metal_max(data: Any) -> float:
        return _dispatch_reduction("max", data)

    @staticmethod
    def metal_mean(data: Any) -> float:
        return _dispatch_reduction("mean", data)

    # -- Elementwise binary ops ------------------------------------------

    @staticmethod
    def metal_add(a: Any, b: Any) -> Any:
        return _dispatch_binary("add", a, b)

    @staticmethod
    def metal_sub(a: Any, b: Any) -> Any:
        return _dispatch_binary("sub", a, b)

    @staticmethod
    def metal_mul(a: Any, b: Any) -> Any:
        return _dispatch_binary("mul", a, b)

    @staticmethod
    def metal_div(a: Any, b: Any) -> Any:
        return _dispatch_binary("div", a, b)

    # -- Sort -----------------------------------------------------------

    @staticmethod
    def metal_sort(data: Any) -> Any:
        """Sort a float32/int32/int64 array using GPU bitonic/radix sort."""
        if not _has_metal():
            raise MetalNotAvailable("Metal not available")

        from metaldf._wrappers import ProxySeries
        if isinstance(data, ProxySeries) and hasattr(data, "_pandas_obj"):
            data = data.to_pandas()

        if not hasattr(data, "dtype"):
            raise MetalNotAvailable("Operand must be a pandas Series or numpy array")

        if data.dtype not in _SORT_DTYPES:
            raise MetalNotAvailable(f"Sort not supported for {data.dtype}")

        arr = _extract_array(data)
        buf = _make_series(arr)
        result_buf = metaldf_engine.metal_sort(buf)
        return pd.Series(result_buf.to_numpy(), index=data.index, name=getattr(data, "name", None))

    @staticmethod
    def metal_argsort(data: Any) -> Any:
        """Return indices that would sort the array."""
        if not _has_metal():
            raise MetalNotAvailable("Metal not available")

        from metaldf._wrappers import ProxySeries
        if isinstance(data, ProxySeries) and hasattr(data, "_pandas_obj"):
            data = data.to_pandas()

        if not hasattr(data, "dtype"):
            raise MetalNotAvailable("Operand must be a pandas Series or numpy array")

        if data.dtype not in _SORT_DTYPES:
            raise MetalNotAvailable(f"Argsort not supported for {data.dtype}")

        arr = _extract_array(data)
        buf = _make_series(arr)
        result_buf = metaldf_engine.metal_argsort(buf)
        return result_buf.to_numpy()

    # -- GroupBy --------------------------------------------------------

    @staticmethod
    def metal_groupby_sum(keys: Any, values: Any) -> Any:
        """GroupBy sum using GPU hash- or sort-based aggregation."""
        return _groupby_dispatch("sum", keys, values)

    @staticmethod
    def metal_groupby_mean(keys: Any, values: Any) -> Any:
        """GroupBy mean using GPU sum + count, divided on the CPU."""
        return _groupby_dispatch("mean", keys, values)

    @staticmethod
    def metal_groupby_min(keys: Any, values: Any) -> Any:
        """GroupBy min using GPU hash- or sort-based aggregation."""
        return _groupby_dispatch("min", keys, values)

    @staticmethod
    def metal_groupby_max(keys: Any, values: Any) -> Any:
        """GroupBy max using GPU hash- or sort-based aggregation."""
        return _groupby_dispatch("max", keys, values)

    @staticmethod
    def metal_groupby_count(keys: Any, values: Any) -> Any:
        """GroupBy count using GPU hash- or sort-based aggregation."""
        return _groupby_dispatch("count", keys, values)

    # -- Strings ---------------------------------------------------------

    @staticmethod
    def metal_string_contains(data: Any, pat: str) -> Any:
        return _dispatch_string_bool("metal_string_contains", data, pat)

    @staticmethod
    def metal_string_startswith(data: Any, pat: str) -> Any:
        return _dispatch_string_bool("metal_string_startswith", data, pat)

    @staticmethod
    def metal_string_endswith(data: Any, pat: str) -> Any:
        return _dispatch_string_bool("metal_string_endswith", data, pat)

    @staticmethod
    def metal_string_find(data: Any, pat: str) -> Any:
        """Return the index of the first occurrence of `pat`, or -1 if not found."""
        if not _has_metal():
            raise MetalNotAvailable("Metal not available")
        if not _is_string_dtype(data):
            raise _string_dtype_error(data)
        series = _make_string_series(data)
        result = metaldf_engine.metal_string_find(series, pat)
        return pd.Series(result.to_numpy(), index=data.index, name=getattr(data, "name", None))

    @staticmethod
    def metal_string_lower(data: Any) -> Any:
        return _dispatch_string_transform("metal_string_lower", data)

    @staticmethod
    def metal_string_upper(data: Any) -> Any:
        return _dispatch_string_transform("metal_string_upper", data)

    @staticmethod
    def metal_string_strip(data: Any) -> Any:
        return _dispatch_string_transform("metal_string_strip", data)

    @staticmethod
    def metal_string_replace(data: Any, pat: str, repl: str) -> Any:
        """Replace literal occurrences of `pat` with `repl` in each string.

        Only literal (non-regex) replacement is supported -- the Rust kernel
        does a plain substring search-and-replace. Callers must ensure `pat`
        isn't meant as a regex (see ``ProxyStringAccessor.replace``).
        """
        if not _has_metal():
            raise MetalNotAvailable("Metal not available")
        if not _is_string_dtype(data):
            raise _string_dtype_error(data)
        series = _make_string_series(data)
        result = metaldf_engine.metal_string_replace(series, pat, repl)
        return pd.Series(result.to_strings(), index=data.index, name=getattr(data, "name", None))
