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


_DATETIME_DTYPE = np.dtype('datetime64[ns]')
_TIMEDELTA_DTYPE = np.dtype('timedelta64[ns]')

_SUPPORTED_DTYPES = {np.dtype(np.float32), np.dtype(np.int32), np.dtype(np.int64),
                     _DATETIME_DTYPE, _TIMEDELTA_DTYPE}

_FROM_NUMPY = {
    np.dtype('float32'): lambda arr: metaldf_engine.MetalSeries.from_numpy(arr),
    np.dtype('int32'):   lambda arr: metaldf_engine.MetalSeries.from_numpy_i32(arr),
    np.dtype('int64'):   lambda arr: metaldf_engine.MetalSeries.from_numpy_i64(arr),
    _DATETIME_DTYPE:     lambda arr: metaldf_engine.MetalSeries.from_numpy_datetime(arr.view(np.int64)),
    _TIMEDELTA_DTYPE:    lambda arr: metaldf_engine.MetalSeries.from_numpy_timedelta(arr.view(np.int64)),
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


def _restore_datetime_dtype(arr: NDArray, original_dtype: np.dtype) -> NDArray:
    """View a raw int64 GPU result array back as datetime64[ns]/timedelta64[ns]
    when `original_dtype` was one of those (see ``DType::Datetime`` /
    ``DType::Timedelta`` -- both are stored as plain int64 on the Rust side,
    so ``SharedBuffer.to_numpy()`` always hands back an int64 array; callers
    that echo a Datetime/Timedelta series' *values* back out (e.g. sort,
    boolean-index compaction) must undo that here so the returned pandas
    Series' dtype matches what pandas itself would produce). No-op (returns
    `arr` unchanged) for every other dtype.
    """
    if original_dtype == _DATETIME_DTYPE or original_dtype == _TIMEDELTA_DTYPE:
        return arr.view(original_dtype)
    return arr


def _make_series(arr: NDArray) -> object:
    """Create a Rust MetalSeries from a numpy array with the correct dtype."""
    ctor = _FROM_NUMPY.get(arr.dtype)
    if ctor is None:
        raise MetalNotAvailable(f"Unsupported dtype: {arr.dtype}")
    return ctor(arr)


def _make_series_with_nulls(arr: NDArray) -> object:
    """Create a MetalSeries, detecting NaN as nulls for float32 data.

    ``float32`` is currently the only dtype with a NaN-aware constructor on
    the Rust side (``MetalSeries.from_numpy_with_nulls`` -- see Task 1.1).
    When such an array actually contains NaN, build the series through that
    constructor so the Rust/Metal side carries a validity mask through the
    kernel. Otherwise (non-float dtypes, or a float32 array with no NaNs at
    all) fall back to the plain ``_make_series`` constructor -- cheaper, and
    guaranteed to produce ``null_mask is None`` (see
    ``tests/test_null_mask.py::test_from_numpy_no_nulls_mask_is_none``).
    """
    if arr.dtype == np.dtype(np.float32) and np.any(np.isnan(arr)):
        return metaldf_engine.MetalSeries.from_numpy_with_nulls(arr)
    return _make_series(arr)


def _result_to_series_with_nulls(result: Any, index: Any = None, name: Any = None) -> pd.Series:
    """Convert a MetalSeries result to a pandas Series, restoring NaN for nulls.

    If the result carries a null mask (i.e. the op saw at least one null on
    either operand), the underlying data is upcast to float64 and NaN is
    written back at every invalid position -- matching pandas' own
    NaN-propagation semantics for the same operation. Series with no nulls
    (``result.null_mask is None``) are returned as-is, at their native dtype.
    """
    data = result.to_numpy()
    mask = result.null_mask
    if mask is not None:
        data = data.astype(np.float64, copy=True)
        data[~mask] = np.nan
    return pd.Series(data, index=index, name=name)


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
_SORT_DTYPES = {np.dtype(np.float32), np.dtype(np.int32), np.dtype(np.int64),
                _DATETIME_DTYPE, _TIMEDELTA_DTYPE}

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

    # pandas itself raises TypeError for `datetime_series.sum()` ("does not
    # support operation 'sum'") -- summing nanosecond timestamps isn't a
    # meaningful quantity. Reject it here (rather than letting it reach the
    # plain-int64 Metal sum kernel, which would happily add the raw epoch
    # values) so the caller falls back to pandas and gets that same
    # TypeError. Timedelta *is* meaningfully summable (pandas returns
    # another Timedelta), so only Datetime is blocked here.
    if data.dtype == _DATETIME_DTYPE and op_name == "sum":
        raise MetalNotAvailable("sum not meaningful for datetime")

    arr = _extract_array(data)
    buf = _make_series_with_nulls(arr)
    rust_fn = getattr(metaldf_engine, f"metal_{op_name}")
    # For an all-null reduction, the Rust kernel itself returns NaN (see
    # tests/test_null_reductions.py::test_all_nulls_returns_nan and friends)
    # -- no Python-side wrapping needed, this is already a plain scalar.
    return rust_fn(buf)


# ---------------------------------------------------------------------------
# Cumulative op dtype support
# ---------------------------------------------------------------------------

# NOTE: use np.dtype(...) instances, not bare numpy scalar types -- see the
# _SORT_DTYPES comment above for why comparing dtype instances to bare
# numpy scalar types silently breaks set/dict membership checks.
_CUMULATIVE_DTYPES = {np.dtype(np.float32), np.dtype(np.int32), np.dtype(np.int64),
                      _DATETIME_DTYPE, _TIMEDELTA_DTYPE}


def _dispatch_cumulative(op_name: str, data: Any) -> Any:
    """Try Metal cumulative scan, raise MetalNotAvailable on failure."""
    if not _has_metal():
        raise MetalNotAvailable("Metal not available")

    from metaldf._wrappers import ProxySeries
    if isinstance(data, ProxySeries) and hasattr(data, "_pandas_obj"):
        data = data.to_pandas()

    if not hasattr(data, "dtype"):
        raise MetalNotAvailable("Operand must be a pandas Series or numpy array")

    if data.dtype not in _CUMULATIVE_DTYPES:
        raise MetalNotAvailable(f"Unsupported dtype for cumulative: {data.dtype}")

    if data.dtype == _DATETIME_DTYPE and op_name == "cumsum":
        raise MetalNotAvailable("cumsum not meaningful for datetime")

    arr = _extract_array(data)
    if arr.dtype == np.dtype(np.float32) and np.any(np.isnan(arr)):
        raise MetalNotAvailable("cumulative scan on GPU does not yet support NaN (skipna)")

    buf = _make_series(arr)
    rust_fn = getattr(metaldf_engine, f"metal_{op_name}")
    result = rust_fn(buf)
    out_arr = result.to_numpy()
    out_arr = _restore_datetime_dtype(out_arr, data.dtype)
    return pd.Series(out_arr, index=data.index if hasattr(data, 'index') else None,
                     name=getattr(data, 'name', None))


def _dispatch_shift(data: Any, periods: int = 1) -> Any:
    """Try Metal shift, raise MetalNotAvailable on failure."""
    if not _has_metal():
        raise MetalNotAvailable("Metal not available")

    from metaldf._wrappers import ProxySeries
    if isinstance(data, ProxySeries) and hasattr(data, "_pandas_obj"):
        data = data.to_pandas()

    if not hasattr(data, "dtype"):
        raise MetalNotAvailable("Operand must be a pandas Series or numpy array")

    if data.dtype not in _SUPPORTED_DTYPES:
        raise MetalNotAvailable(f"Unsupported dtype for shift: {data.dtype}")

    arr = _extract_array(data)
    buf = _make_series(arr)
    periods = int(periods)
    result = metaldf_engine.metal_shift(buf, periods)
    out_arr = result.to_numpy()

    # The GPU kernel fills out-of-bounds positions with a raw 0 for every
    # integer dtype (see rust/metal/elementwise/shift.metal) -- it has no
    # notion of pandas' missing-value sentinels. Reconcile that here so the
    # result matches what `pd.Series.shift` itself would produce, and so
    # `diff`/`pct_change` (built on top of `shift`, see `ProxySeries`) don't
    # silently compute against a bogus `0` fill instead of a real "missing"
    # marker:
    #   - float32: the kernel already fills with NaN bit-for-bit -- no-op.
    #   - datetime64/timedelta64: stays as its own dtype, but the filled
    #     slots must read back as NaT rather than the 1970-01-01 epoch.
    #   - plain int32/int64: pandas upcasts the whole result to float64
    #     (ints can't represent NaN), whenever `periods != 0`.
    n = len(out_arr)
    fill_count = min(abs(periods), n) if periods != 0 else 0
    if data.dtype == _DATETIME_DTYPE or data.dtype == _TIMEDELTA_DTYPE:
        out_arr = _restore_datetime_dtype(out_arr, data.dtype)
        if fill_count > 0:
            nat = (np.datetime64('NaT', 'ns') if data.dtype == _DATETIME_DTYPE
                   else np.timedelta64('NaT', 'ns'))
            if periods > 0:
                out_arr[:fill_count] = nat
            else:
                out_arr[n - fill_count:] = nat
    elif periods != 0 and data.dtype != np.dtype(np.float32):
        out_arr = out_arr.astype(np.float64)
        if fill_count > 0:
            if periods > 0:
                out_arr[:fill_count] = np.nan
            else:
                out_arr[n - fill_count:] = np.nan

    return pd.Series(out_arr, index=data.index if hasattr(data, 'index') else None,
                     name=getattr(data, 'name', None))


def _dispatch_fillna(data: Any, fill_value: float) -> Any:
    """Try Metal fillna, raise MetalNotAvailable on failure."""
    if not _has_metal():
        raise MetalNotAvailable("Metal not available")

    from metaldf._wrappers import ProxySeries
    if isinstance(data, ProxySeries) and hasattr(data, "_pandas_obj"):
        data = data.to_pandas()

    if not hasattr(data, "dtype") or data.dtype != np.dtype(np.float32):
        raise MetalNotAvailable(f"fillna GPU only supports float32")

    arr = _extract_array(data)
    buf = _make_series(arr)
    result = metaldf_engine.metal_fillna(buf, float(fill_value))
    return pd.Series(result.to_numpy(), index=data.index if hasattr(data, 'index') else None,
                     name=getattr(data, 'name', None))


def _dispatch_ffill(data: Any) -> Any:
    """Try Metal ffill (forward-fill via parallel scan), raise MetalNotAvailable on failure."""
    if not _has_metal():
        raise MetalNotAvailable("Metal not available")

    from metaldf._wrappers import ProxySeries
    if isinstance(data, ProxySeries) and hasattr(data, "_pandas_obj"):
        data = data.to_pandas()

    if not hasattr(data, "dtype") or data.dtype != np.dtype(np.float32):
        raise MetalNotAvailable("ffill GPU only supports float32")

    arr = _extract_array(data)
    buf = _make_series(arr)
    result = metaldf_engine.metal_ffill(buf)
    return pd.Series(result.to_numpy(), index=data.index if hasattr(data, 'index') else None,
                     name=getattr(data, 'name', None))


def _dispatch_bfill(data: Any) -> Any:
    """Try Metal bfill (backward-fill via parallel scan), raise MetalNotAvailable on failure."""
    if not _has_metal():
        raise MetalNotAvailable("Metal not available")

    from metaldf._wrappers import ProxySeries
    if isinstance(data, ProxySeries) and hasattr(data, "_pandas_obj"):
        data = data.to_pandas()

    if not hasattr(data, "dtype") or data.dtype != np.dtype(np.float32):
        raise MetalNotAvailable("bfill GPU only supports float32")

    arr = _extract_array(data)
    buf = _make_series(arr)
    result = metaldf_engine.metal_bfill(buf)
    return pd.Series(result.to_numpy(), index=data.index if hasattr(data, 'index') else None,
                     name=getattr(data, 'name', None))


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


# ---------------------------------------------------------------------------
# Datetime/timedelta binary op type inference
# ---------------------------------------------------------------------------

# Maps (left_dtype, op_name, right_dtype) -> result_dtype for the six
# datetime/timedelta arithmetic combinations pandas itself supports. Both
# Datetime and Timedelta are stored as plain int64 nanoseconds on the Rust
# side (see DType::Datetime/DType::Timedelta), so the *kernel* dispatched is
# always the existing binary_add_i64/binary_sub_i64 -- this table only
# decides which dtype the raw int64 result should be view-cast back to.
# Combinations pandas itself rejects (e.g. datetime + datetime, timedelta -
# datetime) are intentionally absent, so `_dispatch_datetime_binary` raises
# MetalNotAvailable for them and the caller falls back to pandas, which
# raises the same TypeError pandas raises for those combinations natively.
_DATETIME_ARITH_RULES = {
    (_DATETIME_DTYPE, "sub", _DATETIME_DTYPE): _TIMEDELTA_DTYPE,
    (_DATETIME_DTYPE, "add", _TIMEDELTA_DTYPE): _DATETIME_DTYPE,
    (_DATETIME_DTYPE, "sub", _TIMEDELTA_DTYPE): _DATETIME_DTYPE,
    (_TIMEDELTA_DTYPE, "add", _DATETIME_DTYPE): _DATETIME_DTYPE,
    (_TIMEDELTA_DTYPE, "add", _TIMEDELTA_DTYPE): _TIMEDELTA_DTYPE,
    (_TIMEDELTA_DTYPE, "sub", _TIMEDELTA_DTYPE): _TIMEDELTA_DTYPE,
}


def _dispatch_datetime_binary(op_name: str, a: Any, b: Any) -> Any:
    """Dispatch datetime64[ns]/timedelta64[ns] add/sub to the plain int64
    Metal kernel, then view-cast the raw int64 result back to whichever
    dtype `_DATETIME_ARITH_RULES` says that combination should produce.

    Mirrors ``_dispatch_binary``'s guards (ProxySeries unwrapping, operand
    dtype/index checks) but keyed off the datetime arithmetic rule table
    instead of same-dtype elementwise rules, since here the two operands are
    deliberately allowed -- required, even -- to differ in dtype (e.g.
    Datetime - Timedelta = Datetime).
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

    result_dtype = _DATETIME_ARITH_RULES.get((a.dtype, op_name, b.dtype))
    if result_dtype is None:
        raise MetalNotAvailable(
            f"Unsupported datetime arithmetic: {a.dtype} {op_name} {b.dtype}"
        )

    # Same index-alignment guard as _dispatch_binary: Metal zips operands
    # positionally, so only dispatch when both sides already share an index.
    a_index = getattr(a, "index", None)
    b_index = getattr(b, "index", None)
    if a_index is not None and b_index is not None and not a_index.equals(b_index):
        raise MetalNotAvailable("Index mismatch requires pandas-side alignment")

    # Both Datetime and Timedelta MetalSeries are DType::Datetime/DType::
    # Timedelta on the Rust side, not DType::Int64 -- and `metal_binary_op`
    # rejects both mismatched dtypes (Datetime vs Timedelta, e.g. for
    # `datetime - timedelta`) *and* Datetime/Timedelta outright, since
    # `metal_suffix` only maps Float32/Int32/Int64 to a kernel suffix (see
    # rust/src/kernels/elementwise.rs::metal_suffix). Building both operand
    # buffers from the plain int64 nanosecond view instead -- via
    # `from_numpy_i64` rather than `_make_series`'s dtype-based dispatch --
    # sidesteps both restrictions: the kernel actually dispatched really is
    # just `binary_add_i64`/`binary_sub_i64`, exactly as the Rust side's own
    # DType::Datetime/Timedelta doc comments ("uses kernel_suffix = int64")
    # intend, and the result is view-cast back to the correct
    # datetime64/timedelta64 dtype below.
    arr_a = _extract_array(a).view(np.int64)
    arr_b = _extract_array(b).view(np.int64)
    buf_a = metaldf_engine.MetalSeries.from_numpy_i64(arr_a)
    buf_b = metaldf_engine.MetalSeries.from_numpy_i64(arr_b)
    result = metaldf_engine.metal_binary_op(op_name, buf_a, buf_b)

    result_np = result.to_numpy().view(result_dtype)

    # Same name-inference rule as _dispatch_binary.
    a_name = getattr(a, "name", None)
    name = a_name if not hasattr(b, "name") or a_name == b.name else None
    return pd.Series(result_np, index=a_index, name=name)


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

    # Datetime/Timedelta operands need pandas' type-inference rules (e.g.
    # datetime - datetime = timedelta) applied to the result, not the plain
    # same-dtype-in-same-dtype-out rule below -- route to the dedicated
    # dispatcher before the ELEMENTWISE_DTYPES check would otherwise reject
    # them outright (Datetime/Timedelta aren't elementwise-arithmetic dtypes
    # in their own right; they're only valid in the specific combinations
    # `_DATETIME_ARITH_RULES` lists).
    _dt_dtypes = (_DATETIME_DTYPE, _TIMEDELTA_DTYPE)
    if a.dtype in _dt_dtypes or b.dtype in _dt_dtypes:
        return _dispatch_datetime_binary(op_name, a, b)

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
    buf_a = _make_series_with_nulls(arr_a)
    buf_b = _make_series_with_nulls(arr_b)
    result = metaldf_engine.metal_binary_op(op_name, buf_a, buf_b)

    # Match pandas' name-inference rule for binary ops: if `b` is also a
    # Series, the result keeps the name only when both sides agree (e.g.
    # `pd.Series(..., name="x") + pd.Series(..., name="y")` -> name=None);
    # if `b` is a scalar/array (no `name` attribute), `a`'s name passes
    # through unchanged.
    a_name = getattr(a, "name", None)
    name = a_name if not hasattr(b, "name") or a_name == b.name else None
    return _result_to_series_with_nulls(result, index=a_index, name=name)


def _dispatch_compact(data: Any, mask: Any) -> Any:
    """Filter `data` by a parallel boolean `mask` using GPU stream compaction.

    Mirrors ``_dispatch_binary``'s guards: raises ``MetalNotAvailable`` (so
    the caller falls back to pandas) for unsupported dtypes and for a
    mismatched index between `data` and `mask` -- ``metal_compact``, like
    every other Metal kernel here, just zips the two positionally and has
    no notion of pandas' index-alignment semantics.
    """
    if not _has_metal():
        raise MetalNotAvailable("Metal not available")

    from metaldf._wrappers import ProxySeries
    if isinstance(data, ProxySeries) and hasattr(data, "_pandas_obj"):
        data = data.to_pandas()
    if isinstance(mask, ProxySeries) and hasattr(mask, "_pandas_obj"):
        mask = mask.to_pandas()

    if not hasattr(data, "dtype") or data.dtype not in _SUPPORTED_DTYPES:
        raise MetalNotAvailable(f"Unsupported dtype: {getattr(data, 'dtype', type(data))}")

    data_index = getattr(data, "index", None)
    mask_index = getattr(mask, "index", None)
    if data_index is not None and mask_index is not None and not data_index.equals(mask_index):
        raise MetalNotAvailable("Index mismatch requires pandas-side alignment")

    # Convert mask to a Bool-dtype (uint8, values 0/1) numpy array.
    mask_arr = np.asarray(mask)
    if mask_arr.dtype == np.dtype(np.bool_):
        mask_arr = mask_arr.astype(np.uint8)
    elif mask_arr.dtype != np.dtype(np.uint8):
        raise MetalNotAvailable(f"Mask must be bool, got {mask_arr.dtype}")

    data_arr = _extract_array(data)
    data_series = _make_series(data_arr)
    mask_series = metaldf_engine.MetalSeries.from_numpy_bool(mask_arr)

    result = metaldf_engine.metal_compact(data_series, mask_series)

    # Preserve pandas' index-preserving boolean-indexing semantics: the kept
    # elements retain their *original* index labels (e.g. filtering
    # `pd.Series([1,2,3], index=[7,8,9])` down to its last two elements
    # keeps index [8, 9], not a fresh RangeIndex(0, 2)). The Metal kernel
    # itself only returns compacted values with no index concept, so the
    # matching index subset is computed here, cheaply, via plain numpy
    # boolean-array indexing (no GPU work needed for the index itself).
    result_index = data_index[mask_arr.astype(bool)] if data_index is not None else None
    result_arr = _restore_datetime_dtype(result.to_numpy(), data.dtype)
    return pd.Series(result_arr, index=result_index, name=getattr(data, "name", None))


# ---------------------------------------------------------------------------
# Comparison op dtype support
# ---------------------------------------------------------------------------

# NOTE: use np.dtype(...) instances, not bare numpy scalar types -- see the
# _SORT_DTYPES comment above for why comparing dtype instances to bare
# numpy scalar types silently breaks set/dict membership checks.
#
# TODO(datetime): once the parallel "datetime dtype" task lands, Datetime
# and Timedelta series (both stored as int64 nanoseconds) should also be
# accepted here and treated as comparable against Int64/each other -- see
# the equivalent TODO in rust/src/kernels/comparison.rs::cmp_suffix.
_COMPARE_DTYPES = {np.dtype(np.float32), np.dtype(np.int32), np.dtype(np.int64),
                    _DATETIME_DTYPE, _TIMEDELTA_DTYPE}


def _dispatch_compare(op_name: str, a: Any, b: Any) -> Any:
    """Try a Metal comparison op (eq/ne/lt/le/gt/ge), raising
    ``MetalNotAvailable`` for any condition Metal can't handle so the
    caller falls back to pandas. Always returns a bool-dtype Series (the
    Rust side returns Int32 0/1, converted to bool here).
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

    if a.dtype not in _COMPARE_DTYPES or b.dtype not in _COMPARE_DTYPES:
        raise MetalNotAvailable(f"Comparison not supported for {a.dtype} vs {b.dtype}")

    if a.dtype != b.dtype:
        raise MetalNotAvailable(f"dtype mismatch: {a.dtype} vs {b.dtype}")

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
    result = metaldf_engine.metal_compare_op(op_name, buf_a, buf_b)

    a_name = getattr(a, "name", None)
    name = a_name if not hasattr(b, "name") or a_name == b.name else None
    return pd.Series(result.to_numpy().astype(bool), index=a_index, name=name)


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

    # -- Cumulative ops ----------------------------------------------------

    @staticmethod
    def metal_cumsum(data: Any) -> Any:
        return _dispatch_cumulative("cumsum", data)

    @staticmethod
    def metal_cummin(data: Any) -> Any:
        return _dispatch_cumulative("cummin", data)

    @staticmethod
    def metal_cummax(data: Any) -> Any:
        return _dispatch_cumulative("cummax", data)

    # -- Shift --------------------------------------------------------------

    @staticmethod
    def metal_shift(data: Any, periods: int = 1) -> Any:
        return _dispatch_shift(data, periods)

    # -- Fill ---------------------------------------------------------------

    @staticmethod
    def metal_fillna(data: Any, value: float = 0.0) -> Any:
        return _dispatch_fillna(data, value)

    @staticmethod
    def metal_ffill(data: Any) -> Any:
        return _dispatch_ffill(data)

    @staticmethod
    def metal_bfill(data: Any) -> Any:
        return _dispatch_bfill(data)

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

    # -- Boolean indexing -------------------------------------------------

    @staticmethod
    def metal_compact(data: Any, mask: Any) -> Any:
        """Filter `data` by a parallel boolean `mask` using GPU stream compaction."""
        return _dispatch_compact(data, mask)

    # -- Comparisons ------------------------------------------------------

    @staticmethod
    def metal_cmp_eq(a: Any, b: Any) -> Any:
        return _dispatch_compare("eq", a, b)

    @staticmethod
    def metal_cmp_ne(a: Any, b: Any) -> Any:
        return _dispatch_compare("ne", a, b)

    @staticmethod
    def metal_cmp_lt(a: Any, b: Any) -> Any:
        return _dispatch_compare("lt", a, b)

    @staticmethod
    def metal_cmp_le(a: Any, b: Any) -> Any:
        return _dispatch_compare("le", a, b)

    @staticmethod
    def metal_cmp_gt(a: Any, b: Any) -> Any:
        return _dispatch_compare("gt", a, b)

    @staticmethod
    def metal_cmp_ge(a: Any, b: Any) -> Any:
        return _dispatch_compare("ge", a, b)

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
        result_arr = _restore_datetime_dtype(result_buf.to_numpy(), data.dtype)
        return pd.Series(result_arr, index=data.index, name=getattr(data, "name", None))

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
