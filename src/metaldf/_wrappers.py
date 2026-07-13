"""Concrete proxy wrapper definitions for all pandas types.

Each proxy type wraps a real pandas object. Operations either delegate
straight to pandas via ``__getattr__``, or explicitly try the Metal
engine registry (see ``metaldf._engine``) first and fall back to pandas
on failure.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pandas as pd

from metaldf._deferred import DeferredSeries
from metaldf._proxy import _ProxyMeta


def _indexes_align(a: Any, b: Any) -> bool:
    """True if the deferred/Metal fast path is safe to zip `a` and `b` positionally.

    Mirrors the index-alignment guard in
    ``metaldf._engine._metal._dispatch_binary``: Metal (both the per-op
    eager kernels and the deferred bytecode-interpreter kernel) has no
    notion of pandas' index-alignment semantics -- it just zips operands
    positionally. Only take the fast path when indexes already match (the
    overwhelmingly common case); otherwise fall through so pandas performs
    real union-index alignment (NaN-filling on mismatches).

    ``DeferredSeries`` doesn't track an index at all (materialization
    always produces a fresh default `RangeIndex`), so there's nothing
    meaningful to compare -- and checking `.index` via its `__getattr__`
    would eagerly materialize it, defeating the point of deferring.
    """
    if isinstance(a, DeferredSeries) or isinstance(b, DeferredSeries):
        return True
    a_index = getattr(a, "index", None)
    b_index = getattr(b, "index", None)
    if a_index is not None and b_index is not None:
        return a_index.equals(b_index)
    return True


class ProxyDataFrame(pd.DataFrame, metaclass=_ProxyMeta):
    """Proxy for pandas DataFrame.

    Wraps a real pandas DataFrame. Operations dispatch to the Metal engine
    registry where supported (see ``metaldf._engine``), falling back to
    pandas otherwise.
    """

    _pandas_type = pd.DataFrame

    def __init__(
        self,
        *args: Any,
        _pandas_obj: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if _pandas_obj is not None:
            # Wrap existing DataFrame: initialize pandas base with the data
            super().__init__(_pandas_obj)
            object.__setattr__(self, "_pandas_obj", _pandas_obj)
        else:
            # Direct construction
            super().__init__(*args, **kwargs)
            object.__setattr__(self, "_pandas_obj", self)

    def __getattr__(self, name: str) -> Any:
        if name == "_pandas_obj":
            raise AttributeError(name)
        obj = object.__getattribute__(self, "_pandas_obj")
        return getattr(obj, name)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Assign a column, materializing DeferredSeries/ProxySeries values first.

        Writes through the real ``pd.DataFrame.__setitem__`` (not
        ``target[key] = value``) rather than through the instance's own
        inherited ``__setitem__``, since e.g. ``self[key] = value`` would
        just re-dispatch to this same override and recurse forever.

        Writes onto *both* ``self`` and the tracked ``_pandas_obj`` when
        they're different objects (wrapped construction, e.g.
        ``ProxyDataFrame(_pandas_obj=some_df)`` initializes ``self`` with
        its own copy of the data -- see ``__init__`` -- so ``self`` and
        ``_pandas_obj`` are separate ``BlockManager``s from that point on).
        Both need the new column: reads via the proxy itself (``proxy_df["a"]``,
        which resolves through ``self``'s own inherited ``__getitem__``)
        and reads via ``.to_pandas()`` (which returns ``_pandas_obj``) must
        both see it. For direct construction, ``_pandas_obj is self`` and
        the second write is a harmless no-op repeat of the first.

        Uses ``type(value) is ProxySeries`` (not ``isinstance``) for the
        ProxySeries check: ``_ProxyMeta.__instancecheck__`` makes
        ``isinstance(x, ProxySeries)`` true for *any* plain ``pd.Series``
        too (see ``_wrap_result``'s docstring below for the full
        rationale), which would otherwise make this branch re-fire on the
        already-materialized ``pd.Series`` from the ``DeferredSeries``
        branch above and crash calling ``.to_pandas()`` on a plain Series.
        """
        if isinstance(value, DeferredSeries):
            value = value.to_pandas()
        elif type(value) is ProxySeries:
            value = value.to_pandas()
        obj = object.__getattribute__(self, "_pandas_obj")
        pd.DataFrame.__setitem__(self, key, value)
        if obj is not self:
            pd.DataFrame.__setitem__(obj, key, value)

    def groupby(
        self,
        by: Any = None,
        level: Any = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool = True,
        dropna: bool = True,
    ) -> Any:
        """Group by a column/key, returning a ProxyGroupBy.

        The returned ProxyGroupBy tries Metal for the common
        ``df.groupby(key)[col].agg()`` pattern (sum/mean/min/max/count)
        before falling back to pandas.
        """
        return ProxyGroupBy(
            self,
            by=by,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            observed=observed,
            dropna=dropna,
        )

    def to_pandas(self) -> Any:
        """Unwrap to the real pandas object."""
        return object.__getattribute__(self, "_pandas_obj")

    def __repr__(self) -> str:
        obj = object.__getattribute__(self, "_pandas_obj")
        if obj is not None:
            return f"ProxyDataFrame(\n{repr(obj)}\n)"
        return "ProxyDataFrame(_pandas_obj=None)"

    def __str__(self) -> str:
        obj = object.__getattribute__(self, "_pandas_obj")
        if obj is not None:
            return str(obj)
        return repr(self)


class ProxySeries(pd.Series, metaclass=_ProxyMeta):
    """Proxy for pandas Series.

    Wraps a real pandas Series. Operations dispatch to the Metal engine
    registry where supported (see ``metaldf._engine``), falling back to
    pandas otherwise.
    """

    _pandas_type = pd.Series

    def __init__(
        self,
        *args: Any,
        _pandas_obj: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if _pandas_obj is not None:
            super().__init__(_pandas_obj)
            object.__setattr__(self, "_pandas_obj", _pandas_obj)
        else:
            super().__init__(*args, **kwargs)
            object.__setattr__(self, "_pandas_obj", self)
        # Lazily-built MetalSeries cache for `.str.*` dispatch -- avoids
        # rebuilding the offsets+chars GPU buffers (the ~570ms/5M-rows cost)
        # on every string method call on this series. Each ProxySeries gets
        # its own cache slot; results of string ops are brand-new ProxySeries
        # instances (see `_wrap_result`) with their own `None` cache, so
        # there's no shared mutable state between a series and its derived
        # results.
        object.__setattr__(self, "_metal_str_cache", None)

    def __getattr__(self, name: str) -> Any:
        if name == "_pandas_obj":
            raise AttributeError(name)
        obj = object.__getattribute__(self, "_pandas_obj")
        return getattr(obj, name)

    @property
    def _metal_string_series(self) -> Any:
        """Return the cached MetalSeries for this series' string data, building it once.

        Building a MetalSeries from a pandas string Series means copying every
        string into a GPU-visible offsets+chars buffer pair -- ~570ms for 5M
        rows, versus 2-3ms for the actual kernel. Caching it here means that
        cost is paid once per ``ProxySeries`` instance no matter how many
        ``.str.*`` calls are made on it.
        """
        cache = object.__getattribute__(self, "_metal_str_cache")
        if cache is None:
            import metaldf_engine

            pandas_obj = object.__getattribute__(self, "_pandas_obj")
            cache = metaldf_engine.MetalSeries.from_strings(list(pandas_obj))
            object.__setattr__(self, "_metal_str_cache", cache)
        return cache

    def _try_metal_or_fallback(self, op_name: str, other: Any, reverse: bool = False) -> Any:
        """Dispatch a binary arithmetic op, trying deferred fusion, then Metal
        eager dispatch (non-reverse only), then pandas.

        For add/sub/mul/div, if both operands are float32-eligible (see
        ``_can_defer`` -- ``ProxySeries``/``DeferredSeries``/plain
        scalars, but not int32/int64 series, since the bytecode
        interpreter kernel only supports f32 today), build a
        ``DeferredSeries`` instead of dispatching immediately. This lets
        chains like ``(a + b) * c`` fuse into a single Metal kernel launch
        at materialization time rather than three separate ones.

        Falls back to pandas by looking the dunder up on the real
        ``pd.Series`` class (rather than ``getattr(self._pandas_obj,
        dunder)``) because for a directly-constructed proxy
        ``self._pandas_obj is self`` -- instance-level lookup would just
        call this same override again and recurse forever.

        Reverse ops (radd/rsub/rmul/rtruediv) always go straight to
        pandas: the Metal registry's ``execute(op_name, lhs, rhs)`` has no
        notion of operand order beyond positional lhs/rhs, so there's no
        way to ask it for "rhs - lhs" without swapping arguments and
        re-deriving which side is genuinely `self` -- simpler and safer to
        just fall back for reverse ops here.
        """
        from metaldf._deferred import BinaryOp, LoadColumn, _as_node, _can_defer

        # Deferred path for float32 element-wise ops: build an expression
        # tree instead of dispatching a kernel immediately.
        if op_name in ("add", "sub", "mul", "div") and not reverse:
            if _can_defer(self) and _can_defer(other) and _indexes_align(self, other):
                size = len(self)
                if isinstance(other, DeferredSeries):
                    size = other.size
                return DeferredSeries(
                    root=BinaryOp(op_name, LoadColumn(self), _as_node(other)),
                    size=size,
                )

        # Reverse ops: scalar OP series (e.g. `5.0 + series`)
        if op_name in ("add", "sub", "mul", "div") and reverse:
            if _can_defer(self) and _can_defer(other) and _indexes_align(self, other):
                return DeferredSeries(
                    root=BinaryOp(op_name, _as_node(other), LoadColumn(self)),
                    size=len(self),
                )

        from metaldf._engine import execute
        from metaldf.exceptions import MetalNotAvailable

        if not reverse:
            try:
                result = execute(op_name, self._pandas_obj, other)
                return _wrap_result(result)
            except (MetalNotAvailable, KeyError, Exception):
                pass

        dunder = f"__r{op_name}__" if reverse else f"__{op_name}__"
        # Map engine op names to pandas dunder names
        if dunder == "__div__":
            dunder = "__truediv__"
        elif dunder == "__rdiv__":
            dunder = "__rtruediv__"
        pandas_op = getattr(pd.Series, dunder, None)
        if pandas_op is None:
            return NotImplemented
        result = pandas_op(self._pandas_obj, other)
        return _wrap_result(result)

    def __add__(self, other: Any) -> Any:
        return self._try_metal_or_fallback("add", other)

    def __sub__(self, other: Any) -> Any:
        return self._try_metal_or_fallback("sub", other)

    def __mul__(self, other: Any) -> Any:
        return self._try_metal_or_fallback("mul", other)

    def __truediv__(self, other: Any) -> Any:
        return self._try_metal_or_fallback("div", other)

    # Reverse operations -- CRITICAL: must call __radd__, __rsub__, etc.
    def __radd__(self, other: Any) -> Any:
        return self._try_metal_or_fallback("add", other, reverse=True)

    def __rsub__(self, other: Any) -> Any:
        return self._try_metal_or_fallback("sub", other, reverse=True)

    def __rmul__(self, other: Any) -> Any:
        return self._try_metal_or_fallback("mul", other, reverse=True)

    def __rtruediv__(self, other: Any) -> Any:
        return self._try_metal_or_fallback("div", other, reverse=True)

    def _try_metal_reduction(self, op_name: str, *args: Any, **kwargs: Any) -> Any:
        """Try Metal path for a reduction operation, fall back to pandas.

        The pandas fallback calls the real ``pd.Series`` implementation
        directly (rather than ``getattr(self._pandas_obj, op_name)``)
        because for a directly-constructed proxy ``self._pandas_obj is
        self`` -- looking the method up on the instance would just call
        this same override again. That matters in practice: numpy's
        reduction dispatch (`np.mean(obj)` etc, used by the PandasEngine
        fallback below) calls back into ``obj.mean(axis=None, dtype=None,
        out=None)``, so without this, a Metal-ineligible reduction (e.g. a
        too-small Series) recurses forever.
        """
        from metaldf._engine import execute
        from metaldf._wrappers import _wrap_result
        from metaldf.exceptions import MetalNotAvailable

        try:
            result = execute(op_name, self._pandas_obj, *args, **kwargs)
            return result
        except (MetalNotAvailable, Exception):
            pandas_method = getattr(pd.Series, op_name, None)
            if pandas_method is None:
                raise AttributeError(f"'{type(self).__name__}' has no attribute '{op_name}'")
            result = pandas_method(self._pandas_obj, *args, **kwargs)
            return _wrap_result(result)

    def sum(self, *args: Any, **kwargs: Any) -> Any:
        return self._try_metal_reduction("sum", *args, **kwargs)

    def min(self, *args: Any, **kwargs: Any) -> Any:
        return self._try_metal_reduction("min", *args, **kwargs)

    def max(self, *args: Any, **kwargs: Any) -> Any:
        return self._try_metal_reduction("max", *args, **kwargs)

    def mean(self, *args: Any, **kwargs: Any) -> Any:
        return self._try_metal_reduction("mean", *args, **kwargs)

    def sort_values(self, ascending: bool = True, **kwargs: Any) -> Any:
        """Sort values, trying Metal first.

        Falls back to the real ``pd.Series.sort_values`` (not
        ``self._pandas_obj.sort_values``) to avoid recursing back into this
        same override when ``self._pandas_obj is self`` (direct construction).
        """
        from metaldf._engine import execute
        from metaldf._wrappers import _wrap_result
        from metaldf.exceptions import MetalNotAvailable

        try:
            result = execute("sort", self._pandas_obj)
            if not ascending:
                result = result[::-1]
            return _wrap_result(result)
        except (MetalNotAvailable, Exception):
            result = pd.Series.sort_values(self._pandas_obj, ascending=ascending, **kwargs)
            return _wrap_result(result)

    def argsort(self, **kwargs: Any) -> Any:
        """Return indices that would sort the array, trying Metal first."""
        from metaldf._engine import execute
        from metaldf.exceptions import MetalNotAvailable

        try:
            return execute("argsort", self._pandas_obj)
        except (MetalNotAvailable, Exception):
            return pd.Series.argsort(self._pandas_obj, **kwargs)

    def groupby(
        self,
        by: Any = None,
        level: Any = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        observed: bool = True,
        dropna: bool = True,
    ) -> Any:
        """Group by a key, returning a ProxyGroupBy.

        The returned ProxyGroupBy tries Metal for sum/mean/min/max/count
        before falling back to pandas.
        """
        return ProxyGroupBy(
            self,
            by=by,
            level=level,
            as_index=as_index,
            sort=sort,
            group_keys=group_keys,
            observed=observed,
            dropna=dropna,
        )

    @property
    def str(self) -> ProxyStringAccessor:
        """Accessor for vectorized string methods, e.g. ``series.str.contains(...)``.

        Tries Metal GPU dispatch first (see ``ProxyStringAccessor``), falling
        back to the real ``pd.Series.str`` accessor for unsupported dtypes,
        small arrays, or methods without a Metal kernel.
        """
        return ProxyStringAccessor(self)

    def to_pandas(self) -> Any:
        return object.__getattribute__(self, "_pandas_obj")

    def __repr__(self) -> str:
        obj = object.__getattribute__(self, "_pandas_obj")
        if obj is not None:
            return f"ProxySeries(\n{repr(obj)}\n)"
        return "ProxySeries(_pandas_obj=None)"

    def __str__(self) -> str:
        obj = object.__getattribute__(self, "_pandas_obj")
        if obj is not None:
            return str(obj)
        return repr(self)


class ProxyStringAccessor:
    """Proxy for Series.str accessor -- tries Metal, falls back to pandas.

    Each explicit method here (contains/startswith/endswith/find/lower/upper/
    strip/replace) dispatches straight to a ``metaldf_engine.metal_string_*``
    Rust kernel, passing the series' cached ``MetalSeries`` (see
    ``ProxySeries._metal_string_series``) so the offsets+chars GPU buffers are
    only built once per series, not on every ``.str.*`` call. Unsupported
    dtypes, small arrays, or Metal unavailability all raise
    ``MetalNotAvailable`` (or any other error) which is caught here and re-run
    through the real ``pd.Series.str`` accessor. Any other ``.str`` method
    (e.g. ``split``, ``cat``) simply falls through to pandas via
    ``__getattr__``.
    """

    def __init__(self, series: ProxySeries) -> None:
        self._series = series

    def _real_str_accessor(self) -> Any:
        """Get the real ``pd.Series.str`` accessor, bypassing ``ProxySeries.str``.

        ``self._series._pandas_obj`` may itself *be* the ``ProxySeries``
        (direct construction, e.g. ``ProxySeries(pd.Series(data))`` sets
        ``_pandas_obj = self``). Accessing ``.str`` on it would then just
        call ``ProxySeries.str`` again -- constructing a new
        ``ProxyStringAccessor`` around the same series and recursing
        forever the first time a fallback is needed. Pulling the raw
        ``CachedAccessor`` descriptor straight off ``pd.Series`` and
        invoking it manually always binds to the real pandas
        ``StringMethods`` implementation, regardless of which case applies.
        """
        pandas_obj = self._series._pandas_obj
        return pd.Series.__dict__["str"].__get__(pandas_obj, pd.Series)

    # Ops whose Rust kernel returns an Int32 0/1 MetalSeries (cast to bool)
    # vs. a Utf8 MetalSeries (materialized back to Python strings).
    _BOOL_OPS = ("contains", "startswith", "endswith")
    _TRANSFORM_OPS = ("lower", "upper", "strip")

    def _metal_series_and_module(self) -> tuple[Any, Any]:
        """Return ``(cached MetalSeries, metaldf_engine module)`` for this series.

        Shared precondition check for every Metal string dispatch path
        (``_try_metal`` and ``replace``): raises ``MetalNotAvailable`` if
        Metal isn't loaded or the series isn't string-dtyped, otherwise
        returns the series' cached ``MetalSeries`` (building it once via
        ``ProxySeries._metal_string_series`` -- see that property's
        docstring for why caching it matters) plus the ``metaldf_engine``
        module for the caller to invoke a kernel on.
        """
        from metaldf._engine._metal import is_metal_available, _is_string_dtype
        from metaldf.exceptions import MetalNotAvailable

        if not is_metal_available():
            raise MetalNotAvailable("Metal not available")

        pandas_obj = self._series._pandas_obj
        if not _is_string_dtype(pandas_obj):
            raise MetalNotAvailable("Not string dtype")

        import metaldf_engine

        return self._series._metal_string_series, metaldf_engine

    def _try_metal(self, op_name: str, *args: Any, **kwargs: Any) -> Any:
        """Dispatch a string op straight to the cached MetalSeries + Rust kernel.

        Bypasses ``metaldf._engine.execute``/``MetalEngine`` (see ``_metal.py``)
        on purpose: that layer rebuilds a fresh MetalSeries -- i.e. re-copies
        every string into new offsets+chars GPU buffers -- on *every* call.
        Going straight to ``metaldf_engine.metal_string_*`` with
        ``self._series._metal_string_series`` (built once, cached on the
        series) means only the kernel itself runs on repeat calls.
        """
        from metaldf.exceptions import MetalNotAvailable

        try:
            metal_series, metaldf_engine = self._metal_series_and_module()
            pandas_obj = self._series._pandas_obj
            index = pandas_obj.index
            name = pandas_obj.name

            if op_name in self._BOOL_OPS:
                rust_fn = getattr(metaldf_engine, f"metal_string_{op_name}")
                result = rust_fn(metal_series, args[0])
                pandas_result = pd.Series(result.to_numpy().astype(bool), index=index, name=name)
            elif op_name == "find":
                result = metaldf_engine.metal_string_find(metal_series, args[0])
                pandas_result = pd.Series(result.to_numpy(), index=index, name=name)
            elif op_name in self._TRANSFORM_OPS:
                rust_fn = getattr(metaldf_engine, f"metal_string_{op_name}")
                result = rust_fn(metal_series)
                pandas_result = pd.Series(result.to_strings(), index=index, name=name)
            else:
                raise MetalNotAvailable(f"Unsupported string op for Metal dispatch: {op_name}")

            return _wrap_result(pandas_result)
        except (MetalNotAvailable, Exception):
            pandas_result = getattr(self._real_str_accessor(), op_name)(*args, **kwargs)
            return _wrap_result(pandas_result)

    def _fallback(self, op_name: str, *args: Any, **kwargs: Any) -> Any:
        """Call the real pandas ``.str`` accessor directly, bypassing Metal."""
        pandas_result = getattr(self._real_str_accessor(), op_name)(*args, **kwargs)
        return _wrap_result(pandas_result)

    def contains(self, pat: str, **kwargs: Any) -> Any:
        # Metal kernels only implement plain substring containment -- any
        # extra kwargs (case, na, regex, flags, ...) change semantics that
        # Metal doesn't replicate, so go straight to pandas rather than
        # silently ignoring them.
        if kwargs:
            return self._fallback("contains", pat, **kwargs)
        return self._try_metal("contains", pat)

    def startswith(self, pat: str, **kwargs: Any) -> Any:
        if kwargs:
            return self._fallback("startswith", pat, **kwargs)
        return self._try_metal("startswith", pat)

    def endswith(self, pat: str, **kwargs: Any) -> Any:
        if kwargs:
            return self._fallback("endswith", pat, **kwargs)
        return self._try_metal("endswith", pat)

    def find(self, sub: str, **kwargs: Any) -> Any:
        if kwargs:
            return self._fallback("find", sub, **kwargs)
        return self._try_metal("find", sub)

    def replace(self, pat: str, repl: str, **kwargs: Any) -> Any:
        """Replace literal occurrences of `pat` with `repl`.

        Only dispatches to Metal for plain literal replacement -- i.e.
        ``regex=False`` passed explicitly, and no other kwargs. pandas'
        default is ``regex=True``, and any other kwarg (``case``,
        ``flags``, ``n``, ...) changes semantics the Metal kernel doesn't
        replicate (same guard as contains/startswith/endswith above), so
        those go straight to pandas.

        Falls back via ``self._fallback`` (-> ``_real_str_accessor()``)
        rather than ``self._series._pandas_obj.str.replace(...)``: for a
        directly-constructed ``ProxySeries`` (e.g.
        ``ProxySeries(pd.Series(d))``), ``_pandas_obj is self``, so
        ``_pandas_obj.str`` would re-enter ``ProxyStringAccessor.replace``
        and recurse forever. See ``_real_str_accessor``'s docstring for the
        full explanation.

        Doesn't reuse the generic ``_try_metal`` helper here: on a genuine
        Metal failure, ``_try_metal``'s fallback calls
        ``getattr(accessor, op_name)(*args, **kwargs)`` with whatever kwargs
        the caller passed -- none, in the literal-replacement case -- which
        would hit pandas' ``regex=True`` default and silently reinterpret
        ``pat`` as a regex. Falling back explicitly with ``regex=False``
        below preserves the literal-replacement intent.
        """
        if kwargs.get("regex", True) or (set(kwargs) - {"regex"}):
            return self._fallback("replace", pat, repl, **kwargs)

        from metaldf.exceptions import MetalNotAvailable

        try:
            metal_series, metaldf_engine = self._metal_series_and_module()
            pandas_obj = self._series._pandas_obj
            result = metaldf_engine.metal_string_replace(metal_series, pat, repl)
            pandas_result = pd.Series(
                result.to_strings(), index=pandas_obj.index, name=pandas_obj.name
            )
            return _wrap_result(pandas_result)
        except (MetalNotAvailable, Exception):
            return self._fallback("replace", pat, repl, regex=False)

    def lower(self) -> Any:
        return self._try_metal("lower")

    def upper(self) -> Any:
        return self._try_metal("upper")

    def strip(self) -> Any:
        return self._try_metal("strip")

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._real_str_accessor(), name)
        if callable(attr):
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return _wrap_result(attr(*args, **kwargs))
            return wrapper
        return attr


class ProxyIndex(pd.Index, metaclass=_ProxyMeta):
    """Proxy for pandas Index.

    Wraps a real pandas Index. Used to keep index results consistent with
    the rest of the proxy chain (ProxyDataFrame/ProxySeries); currently
    delegates everything to pandas via ``__getattr__``.
    """

    _pandas_type = pd.Index

    def __init__(
        self,
        *args: Any,
        _pandas_obj: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if _pandas_obj is not None:
            super().__init__(_pandas_obj)
            object.__setattr__(self, "_pandas_obj", _pandas_obj)
        else:
            super().__init__(*args, **kwargs)
            object.__setattr__(self, "_pandas_obj", self)

    def __getattr__(self, name: str) -> Any:
        if name == "_pandas_obj":
            raise AttributeError(name)
        obj = object.__getattribute__(self, "_pandas_obj")
        return getattr(obj, name)

    def to_pandas(self) -> Any:
        return object.__getattribute__(self, "_pandas_obj")


# Intermediate types -- record method chain, replay on conversion
class ProxyGroupBy:
    """Proxy for DataFrame.groupby() / Series.groupby().

    Records the groupby call so it can be replayed on either side.
    Explicit aggregations (sum/mean/min/max/count) try Metal first for the
    common ``df.groupby(key)[col].agg()`` pattern; everything else is
    recorded via ``__getattr__``/``__getitem__`` and replayed on pandas.
    """

    def __init__(
        self,
        obj: Any,
        by: Any = None,
        axis: int = 0,
        level: Any = None,
        as_index: bool = True,
        sort: bool = True,
        group_keys: bool = True,
        squeeze: Any = False,
        observed: bool = False,
        dropna: bool = True,
    ) -> None:
        self._obj = obj
        self._by = by
        self._axis = axis
        self._level = level
        self._as_index = as_index
        self._sort = sort
        self._group_keys = group_keys
        self._squeeze = squeeze
        self._observed = observed
        self._dropna = dropna
        self._method_chain: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _groupby_kwargs(self, base_type: type) -> dict[str, Any]:
        """Build kwargs for the real pandas ``.groupby()`` call.

        ``axis`` and ``squeeze`` were deprecated then removed from pandas'
        groupby signature across the 2.x/3.x series -- filter the candidate
        kwargs down to whatever the installed pandas actually accepts so
        this works across the pandas>=2.0 range this project supports.
        """
        candidate = {
            "by": self._by,
            "axis": self._axis,
            "level": self._level,
            "as_index": self._as_index,
            "sort": self._sort,
            "group_keys": self._group_keys,
            "squeeze": self._squeeze,
            "observed": self._observed,
            "dropna": self._dropna,
        }
        try:
            params = inspect.signature(base_type.groupby).parameters
        except (TypeError, ValueError):
            return candidate
        return {k: v for k, v in candidate.items() if k in params}

    def _replay(self) -> Any:
        """Replay the recorded method chain on a real pandas groupby object.

        Calls the pandas base-class ``.groupby()`` directly (rather than
        ``self._obj.groupby(...)``) so that this bypasses ProxyDataFrame's/
        ProxySeries's own ``.groupby()`` override -- otherwise replaying
        here would just construct another ProxyGroupBy and recurse forever.
        """
        obj = self._obj
        if isinstance(obj, pd.DataFrame):
            gb = pd.DataFrame.groupby(obj, **self._groupby_kwargs(pd.DataFrame))
        elif isinstance(obj, pd.Series):
            gb = pd.Series.groupby(obj, **self._groupby_kwargs(pd.Series))
        else:
            gb = obj.groupby(**self._groupby_kwargs(type(obj)))
        for method_name, args, kwargs in self._method_chain:
            gb = getattr(gb, method_name)(*args, **kwargs)
        return gb

    def _replay_with_agg(self, agg_name: str) -> Any:
        """Replay the groupby chain and apply the aggregation on pandas."""
        gb = self._replay()
        return getattr(gb, agg_name)()

    def _replay_with_agg_full(
        self, agg_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        """Replay the groupby chain, passing explicit args/kwargs to the aggregation."""
        gb = self._replay()
        return getattr(gb, agg_name)(*args, **kwargs)

    def _try_metal_agg(self, agg_name: str) -> Any:
        """Try Metal groupby aggregation, fall back to pandas replay."""
        from metaldf._engine import execute
        from metaldf.exceptions import MetalNotAvailable

        try:
            obj = self._obj
            by = self._by

            if isinstance(obj, pd.DataFrame) and isinstance(by, str):
                # Only optimize df.groupby('key')['val'].agg() -- a single
                # grouping key and a single selected value column.
                pandas_obj = obj.to_pandas() if hasattr(obj, "to_pandas") else obj
                if not self._method_chain:
                    raise MetalNotAvailable("Multi-column groupby not optimized")

                first_method, first_args, _first_kwargs = self._method_chain[0]
                if (
                    first_method != "__getitem__"
                    or len(first_args) != 1
                    or not isinstance(first_args[0], str)
                ):
                    raise MetalNotAvailable("Complex groupby chain not optimized")

                col_name = first_args[0]
                keys = pandas_obj[by]
                values = pandas_obj[col_name]
                result = execute(f"groupby_{agg_name}", keys, values)
                return _wrap_result(result)

            elif isinstance(obj, pd.Series):
                # Series.groupby(by=<array-like>).agg()
                pandas_obj = obj.to_pandas() if hasattr(obj, "to_pandas") else obj
                if not isinstance(by, (pd.Series, np.ndarray)):
                    raise MetalNotAvailable("Series groupby requires array-like keys")
                by_arr = by.to_pandas() if hasattr(by, "to_pandas") else by
                result = execute(f"groupby_{agg_name}", pd.Series(by_arr), pandas_obj)
                return _wrap_result(result)

            raise MetalNotAvailable("Unsupported groupby pattern")

        except Exception:
            # Any failure (Metal unavailable, unsupported dtype, unsupported
            # shape, ...) falls back to plain pandas replay.
            result = self._replay_with_agg(agg_name)
            return _wrap_result(result)

    def sum(self, *args: Any, **kwargs: Any) -> Any:
        if args or kwargs:
            return _wrap_result(self._replay_with_agg_full("sum", args, kwargs))
        return self._try_metal_agg("sum")

    def mean(self, *args: Any, **kwargs: Any) -> Any:
        if args or kwargs:
            return _wrap_result(self._replay_with_agg_full("mean", args, kwargs))
        return self._try_metal_agg("mean")

    def min(self, *args: Any, **kwargs: Any) -> Any:
        if args or kwargs:
            return _wrap_result(self._replay_with_agg_full("min", args, kwargs))
        return self._try_metal_agg("min")

    def max(self, *args: Any, **kwargs: Any) -> Any:
        if args or kwargs:
            return _wrap_result(self._replay_with_agg_full("max", args, kwargs))
        return self._try_metal_agg("max")

    def count(self, *args: Any, **kwargs: Any) -> Any:
        if args or kwargs:
            return _wrap_result(self._replay_with_agg_full("count", args, kwargs))
        return self._try_metal_agg("count")

    def __getitem__(self, key: Any) -> ProxyGroupBy:
        """Record column selection (e.g. ``gb['val']``) for later replay/Metal dispatch."""
        new_gb = ProxyGroupBy.__new__(ProxyGroupBy)
        new_gb._obj = self._obj
        new_gb._by = self._by
        new_gb._axis = self._axis
        new_gb._level = self._level
        new_gb._as_index = self._as_index
        new_gb._sort = self._sort
        new_gb._group_keys = self._group_keys
        new_gb._squeeze = self._squeeze
        new_gb._observed = self._observed
        new_gb._dropna = self._dropna
        new_gb._method_chain = self._method_chain + [("__getitem__", (key,), {})]
        return new_gb

    def __getattr__(self, name: str) -> Any:
        """Record method calls and return a lazy wrapper."""

        def method_capturer(*args: Any, **kwargs: Any) -> Any:
            self._method_chain.append((name, args, kwargs))
            # Try to replay immediately so subsequent calls work
            result = self._replay()
            return _wrap_result(result)

        return method_capturer


class ProxyRolling:
    """Proxy for Series.rolling() / DataFrame.rolling().

    Similar to ProxyGroupBy: records the rolling window and replays.
    """

    def __init__(self, obj: Any, window: Any, **kwargs: Any) -> None:
        self._obj = obj
        self._window = window
        self._rolling_kwargs = kwargs
        self._method_chain: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _replay(self) -> Any:
        rolling = self._obj.rolling(window=self._window, **self._rolling_kwargs)
        for method_name, args, kwargs in self._method_chain:
            rolling = getattr(rolling, method_name)(*args, **kwargs)
        return rolling

    def __getattr__(self, name: str) -> Any:
        def method_capturer(*args: Any, **kwargs: Any) -> Any:
            self._method_chain.append((name, args, kwargs))
            result = self._replay()
            return _wrap_result(result)

        return method_capturer


class ProxyModule:
    """Module-level proxy that intercepts pandas type access.

    Returns proxy types for user code, real types for internal pandas/metaldf code.
    """

    def __init__(self, real_module: Any) -> None:
        self._real_module = real_module

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._real_module, name)

        # Intercept DataFrame, Series, Index
        if attr is pd.DataFrame:
            return ProxyDataFrame
        if attr is pd.Series:
            return ProxySeries
        if attr is pd.Index:
            return ProxyIndex

        # For functions, wrap their return values
        if callable(attr) and not isinstance(attr, type):
            return _wrap_callable(attr)

        return attr


def _wrap_callable(func: Any) -> Any:
    """Wrap a function so its return value is proxied if it's a pandas type."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        return _wrap_result(result)

    return wrapper


def _wrap_result(result: Any) -> Any:
    """Wrap a pandas result in the appropriate proxy type.

    Returns proxy for DataFrame/Series/Index, passes through everything else.
    Also passes through if already a proxy type to avoid double-wrapping.

    Note: the "already a proxy" check below uses ``type(result) in (...)``
    rather than ``isinstance(result, (...))`` on purpose. ``_ProxyMeta``
    overrides ``__instancecheck__`` so that, e.g., ``isinstance(x,
    ProxySeries)`` is also True for any *plain* ``pd.Series`` (that's what
    lets a real pandas object satisfy proxy-type isinstance checks
    elsewhere). Using ``isinstance`` here would make that same trick
    misfire: every plain ``pd.Series``/``pd.DataFrame``/``pd.Index`` result
    would match the "already wrapped" branch and get returned unwrapped,
    silently skipping ``ProxySeries``/``ProxyDataFrame``/``ProxyIndex``
    construction (and losing methods like ``.to_pandas()``).
    """
    if type(result) in (ProxyDataFrame, ProxySeries, ProxyIndex):
        return result
    if isinstance(result, pd.DataFrame):
        return ProxyDataFrame(_pandas_obj=result)
    if isinstance(result, pd.Series):
        return ProxySeries(_pandas_obj=result)
    if isinstance(result, pd.Index):
        return ProxyIndex(_pandas_obj=result)
    return result
