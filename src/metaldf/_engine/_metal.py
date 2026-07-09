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
