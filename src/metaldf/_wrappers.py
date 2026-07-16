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
        # Pending deferred column assignments for multi-output fusion
        # (Task 8.1). Queued by ``__setitem__`` when the value is a
        # ``DeferredSeries``; flushed by ``_flush_pending`` before any read.
        object.__setattr__(self, "_pending_assigns", [])
        # Guard flag: when True, ``_flush_pending`` becomes a no-op. Set
        # around calls into ``pd.DataFrame`` internals (e.g.
        # ``__getitem__``) that may re-enter our ``columns`` property and
        # cause a premature flush.
        object.__setattr__(self, "_suppress_flush", False)

    def _flush_pending(self) -> None:
        """Materialize all queued deferred column assignments at once.

        Tries the multi-output codegen path first (a single GPU kernel with
        one output buffer per pending assignment, sharing input column reads).
        Falls back to materializing each ``DeferredSeries`` individually if
        the fused path fails for any reason.
        """
        try:
            if object.__getattribute__(self, "_suppress_flush"):
                return
        except AttributeError:
            pass
        try:
            pending = object.__getattribute__(self, "_pending_assigns")
        except AttributeError:
            return
        if not pending:
            return

        # Clear the queue before doing any work so that any column access
        # triggered during materialization (e.g. __getitem__ inside
        # _collect_columns) doesn't re-enter _flush_pending.
        object.__setattr__(self, "_pending_assigns", [])

        try:
            import metaldf_engine
            from metaldf._engine._metal import _extract_array, _make_series

            # 1. Collect all unique LoadColumn series across ALL pending trees
            all_columns: list[Any] = []
            seen: set[int] = set()
            for _col_name, deferred in pending:
                for col in deferred._collect_columns():
                    obj_id = id(col)
                    if obj_id not in seen:
                        seen.add(obj_id)
                        all_columns.append(col)

            # 2. Build unified column_index shared by all programs
            column_index = {id(col): i for i, col in enumerate(all_columns)}

            # 3. Compile each deferred tree to bytecode using the shared mapping
            programs: list[bytes] = []
            for _col_name, deferred in pending:
                programs.append(deferred._compile_bytecode(column_index))

            # 4. Convert columns to MetalSeries
            metal_cols = []
            for col in all_columns:
                pandas_obj = object.__getattribute__(col, "_pandas_obj")
                arr = _extract_array(pandas_obj)
                metal_cols.append(_make_series(arr))

            size = pending[0][1].size

            # 5. Dispatch all programs in a single GPU kernel
            results = metaldf_engine.eval_multi_expression_codegen(
                programs, metal_cols, size,
            )

            # 6. Assign results as concrete columns
            obj = object.__getattribute__(self, "_pandas_obj")
            for (col_name, _deferred), result in zip(pending, results):
                value = pd.Series(result.to_numpy())
                pd.DataFrame.__setitem__(self, col_name, value)
                if obj is not self:
                    pd.DataFrame.__setitem__(obj, col_name, value)
        except Exception:
            # Fallback: materialize each deferred individually
            obj = object.__getattribute__(self, "_pandas_obj")
            for col_name, deferred in pending:
                value = deferred.to_pandas()
                pd.DataFrame.__setitem__(self, col_name, value)
                if obj is not self:
                    pd.DataFrame.__setitem__(obj, col_name, value)

    def __getattr__(self, name: str) -> Any:
        if name in ("_pandas_obj", "_pending_assigns"):
            raise AttributeError(name)
        self._flush_pending()
        obj = object.__getattribute__(self, "_pandas_obj")
        return getattr(obj, name)

    @property  # type: ignore[override]
    def columns(self) -> Any:
        """Flush pending assignments before returning columns.

        ``pd.DataFrame.columns`` is an ``AxisProperty`` resolved via the
        MRO, so it bypasses ``__getattr__``. Without this override,
        accessing ``df.columns`` after queuing deferred assignments would
        return the pre-assignment column list.
        """
        self._flush_pending()
        return pd.DataFrame.columns.__get__(self)

    @columns.setter
    def columns(self, value: Any) -> None:
        pd.DataFrame.columns.__set__(self, value)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Assign a column, queuing DeferredSeries for multi-output fusion.

        When ``value`` is a ``DeferredSeries``, the assignment is deferred:
        the ``(key, DeferredSeries)`` pair is appended to ``_pending_assigns``
        and will be materialized together with other queued assignments in a
        single GPU kernel dispatch the next time any read access triggers
        ``_flush_pending``.

        Non-deferred assignments are written through immediately (after
        removing any pending entry with the same key to prevent a stale
        deferred from overwriting the concrete value on the next flush).

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
            pending = object.__getattribute__(self, "_pending_assigns")
            # Remove any earlier pending entry for the same key so the latest
            # deferred assignment wins.
            pending[:] = [(k, v) for k, v in pending if k != key]
            pending.append((key, value))
            return

        # Non-deferred: remove any pending entry with the same key (it is
        # being overridden by a concrete value).
        try:
            pending = object.__getattribute__(self, "_pending_assigns")
            pending[:] = [(k, v) for k, v in pending if k != key]
        except AttributeError:
            pass

        if type(value) is ProxySeries:
            value = value.to_pandas()
        obj = object.__getattribute__(self, "_pandas_obj")
        # Suppress flush: pd.DataFrame.__setitem__ internally accesses
        # self.columns which would re-enter _flush_pending.
        object.__setattr__(self, "_suppress_flush", True)
        try:
            pd.DataFrame.__setitem__(self, key, value)
            if obj is not self:
                pd.DataFrame.__setitem__(obj, key, value)
        finally:
            object.__setattr__(self, "_suppress_flush", False)

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
        self._flush_pending()
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

    def rolling(self, window: Any, **kwargs: Any) -> ProxyRolling:
        """Return a rolling window over the DataFrame as a ``ProxyRolling``.

        The Metal rolling kernels (see ``ProxyRolling._try_metal_rolling``)
        only operate on a single float32 ``Series``, so a DataFrame-level
        rolling aggregation always falls back to plain pandas -- but it
        still needs to come back as a ``ProxyRolling`` (not a raw pandas
        ``Rolling``) so ``df.rolling(3).mean().<anything>`` keeps flowing
        through the proxy chain (e.g. so the result is re-wrapped).
        """
        self._flush_pending()
        return ProxyRolling(self, window=window, **kwargs)

    def __getitem__(self, key: Any) -> Any:
        """Column selection (``df["col"]``) falls straight through to pandas.

        Boolean-mask filtering (``df[mask]``, `mask` a bool-dtype Series or
        array) tries the GPU ``compact`` kernel column-by-column (see
        ``MetalEngine.metal_compact`` / Task 4.1's ``metal_compact``) before
        falling back to plain pandas indexing. Every other key (string
        column name, list of column names, slice, integer mask, ...) is
        left untouched and handed to the real ``pd.DataFrame.__getitem__``.

        Only flushes the pending queue when the requested key is actually
        one of the pending column names (or when the key is not a simple
        string, e.g. a boolean mask or list of columns). This avoids
        premature flush when reading existing input columns during
        expression building (e.g. ``df["a"]`` inside ``df["z"] = df["a"] + df["b"]``).
        """
        # Smart flush: only if key matches a pending column name
        if isinstance(key, str):
            try:
                pending = object.__getattribute__(self, "_pending_assigns")
                if any(k == key for k, _ in pending):
                    self._flush_pending()
            except AttributeError:
                pass
        else:
            self._flush_pending()
        from metaldf._deferred import DeferredSeries

        # If key is a DeferredSeries, materialize it first.
        if isinstance(key, DeferredSeries):
            key = key.to_pandas()

        # Check if key is a boolean mask (Series or array). `isinstance(key,
        # pd.Series)` already covers `ProxySeries` (it's a genuine subclass
        # of `pd.Series`); the explicit `ProxySeries` branch below exists
        # only as a defensive fallback in case that isn't true for some
        # unusual construction.
        is_bool_mask = False
        if isinstance(key, (pd.Series, np.ndarray)):
            arr = np.asarray(key)
            if arr.dtype == np.dtype(np.bool_):
                is_bool_mask = True
        elif type(key) is ProxySeries:
            pandas_key = key.to_pandas()
            arr = np.asarray(pandas_key)
            if arr.dtype == np.dtype(np.bool_):
                is_bool_mask = True
                key = pandas_key

        if is_bool_mask:
            try:
                from metaldf._engine import execute
                # Read columns off `self` (via the real, unbound
                # `pd.DataFrame.__getitem__`), not `self._pandas_obj`: any
                # in-place mutation that isn't routed through
                # `ProxyDataFrame.__setitem__` (e.g. `.insert()`, `.loc[...]
                # = ...`) only updates `self`'s own data, leaving a
                # wrapped-construction proxy's `_pandas_obj` stale (see
                # `tests/exhaustive/test_dataframe_comprehensive.py::
                # TestDataFrameAssign::test_insert`, which mutates via
                # `.insert()` and then immediately reads the new column
                # back through `__getitem__`).
                result_dict = {}
                # Suppress flush while iterating columns: pandas'
                # __getitem__ internally accesses self.columns (our
                # overridden property), which would otherwise re-enter
                # _flush_pending.
                object.__setattr__(self, "_suppress_flush", True)
                try:
                    for col in self.columns:
                        col_series = pd.DataFrame.__getitem__(self, col)
                        result_dict[col] = execute("compact", col_series, key)
                finally:
                    object.__setattr__(self, "_suppress_flush", False)
                result_df = pd.DataFrame(result_dict)
                return ProxyDataFrame(_pandas_obj=result_df)
            except Exception:
                object.__setattr__(self, "_suppress_flush", False)

        # Fallback to pandas (column selection, unsupported dtype, Metal
        # unavailable, index mismatch, ...). Reads/writes through `self`
        # (not `self._pandas_obj`) for the same read-after-write reason as
        # the Metal branch above.
        # Suppress flush: pd.DataFrame.__getitem__ internally accesses
        # self.columns which would re-enter _flush_pending.
        object.__setattr__(self, "_suppress_flush", True)
        try:
            result = pd.DataFrame.__getitem__(self, key)
        finally:
            object.__setattr__(self, "_suppress_flush", False)
        return _wrap_result(result)

    def merge(
        self,
        right: Any,
        how: str = "inner",
        on: Any = None,
        left_on: Any = None,
        right_on: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Merge two DataFrames, trying a GPU hash join for equi-joins.

        Only attempts the Metal path for ``how="inner"`` with no extra
        pandas kwargs (``suffixes``, ``indicator``, ``validate``,
        ``left_index``/``right_index``, ...) -- those change semantics the
        GPU join doesn't replicate, so they go straight to pandas. Within
        that, ``_metal_merge`` further restricts to a single string key
        column (via ``on`` or matching ``left_on``/``right_on``) present on
        both sides as float32 or int32 -- see its docstring. Any failure of
        the Metal path (unsupported dtype/shape, Metal unavailable, ...)
        falls back to plain ``pd.DataFrame.merge``, which always produces
        the exact pandas-reference result.
        """
        self._flush_pending()
        obj = object.__getattribute__(self, "_pandas_obj")

        def _pandas_fallback() -> Any:
            right_df = right.to_pandas() if hasattr(right, "to_pandas") else right
            result = pd.DataFrame.merge(
                obj, right_df, how=how, on=on, left_on=left_on, right_on=right_on, **kwargs
            )
            return _wrap_result(result)

        if how not in ("inner", "left", "right") or kwargs:
            return _pandas_fallback()

        try:
            return self._metal_merge(right, on=on, left_on=left_on, right_on=right_on, how=how)
        except Exception:
            return _pandas_fallback()

    def _metal_merge(
        self,
        right: Any,
        on: Any = None,
        left_on: Any = None,
        right_on: Any = None,
        how: str = "inner",
    ) -> Any:
        """Equi-join via ``metaldf_engine.metal_hash_join`` + ``metal_take``.

        Raises ``MetalNotAvailable`` (or lets any other exception propagate)
        for anything the GPU path can't handle, so ``merge()`` can catch it
        and fall back to a full pandas merge:

        - multi-column keys (``on``/``left_on``/``right_on`` not a plain
          string),
        - no key specified at all,
        - key dtype other than float32/int32, or mismatched key dtypes
          (``metal_hash_join`` only supports Float32/Float32 or
          Int32/Int32),
        - Metal not being available at all.

        The Rust kernel uses build/probe (not left/right) semantics and
        returns ``(build_indices, probe_indices)`` -- the smaller table is
        used as build here (better hash-table load factor), then the
        indices are mapped back to left/right order. Column values are
        gathered via ``metal_take`` for dtypes it supports (float32/int32/
        int64); any other column dtype (object/string, bool, float64,
        datetime, ...) is gathered with plain numpy fancy indexing instead
        of failing the whole merge -- the GPU-computed join indices are
        still reused, only that column's gather happens on the CPU.
        """
        import metaldf_engine

        from metaldf._engine._metal import (
            _FROM_NUMPY,
            _extract_array,
            _make_series,
            is_metal_available,
        )
        from metaldf.exceptions import MetalNotAvailable

        if not is_metal_available():
            raise MetalNotAvailable("Metal not available")

        obj = object.__getattribute__(self, "_pandas_obj")
        right_df = right.to_pandas() if hasattr(right, "to_pandas") else right

        # Resolve key column names -- single string key only.
        if on is not None:
            if not isinstance(on, str):
                raise MetalNotAvailable("Metal join only supports a single key column")
            left_key, right_key = on, on
        elif left_on is not None and right_on is not None:
            if not isinstance(left_on, str) or not isinstance(right_on, str):
                raise MetalNotAvailable("Metal join only supports a single key column")
            left_key, right_key = left_on, right_on
        else:
            raise MetalNotAvailable("Must specify 'on' or both 'left_on' and 'right_on'")

        left_keys_arr = _extract_array(obj[left_key])
        right_keys_arr = _extract_array(right_df[right_key])

        supported = {np.dtype(np.float32), np.dtype(np.int32)}
        if left_keys_arr.dtype not in supported or right_keys_arr.dtype not in supported:
            raise MetalNotAvailable(
                f"Join keys must be float32 or int32, got "
                f"{left_keys_arr.dtype}/{right_keys_arr.dtype}"
            )
        if left_keys_arr.dtype != right_keys_arr.dtype:
            raise MetalNotAvailable("Key dtypes must match")

        left_keys_ms = _make_series(left_keys_arr)
        right_keys_ms = _make_series(right_keys_arr)

        # Decide build vs probe: the smaller table is build, for a better
        # hash-table load factor (see Task 6.1's `metal_hash_join`).
        if len(left_keys_arr) <= len(right_keys_arr):
            build_ms, probe_ms = left_keys_ms, right_keys_ms
            left_is_build = True
        else:
            build_ms, probe_ms = right_keys_ms, left_keys_ms
            left_is_build = False

        build_idx, probe_idx = metaldf_engine.metal_hash_join(build_ms, probe_ms)

        if left_is_build:
            left_idx_ms, right_idx_ms = build_idx, probe_idx
        else:
            left_idx_ms, right_idx_ms = probe_idx, build_idx

        def _gather(df: Any, col: str, idx_ms: Any) -> Any:
            col_arr = _extract_array(df[col])
            if col_arr.dtype in _FROM_NUMPY:
                col_ms = _make_series(col_arr)
                return metaldf_engine.metal_take(col_ms, idx_ms).to_numpy()
            idx_np = idx_ms.to_numpy()
            return col_arr[idx_np]

        result_dict: dict[str, Any] = {}
        for col in obj.columns:
            result_dict[col] = _gather(obj, col, left_idx_ms)

        # Match pandas' default `_x`/`_y` suffixing for non-key columns that
        # appear on both sides. When both sides use the same key column name
        # (`on=...`), that column is emitted once (from the left side) and
        # excluded from suffixing/duplication.
        same_key_name = left_key == right_key
        overlap = set(obj.columns) & set(right_df.columns)
        if same_key_name:
            overlap.discard(left_key)

        if overlap:
            renamed: dict[str, Any] = {}
            for col in list(result_dict):
                renamed[f"{col}_x" if col in overlap else col] = result_dict.pop(col)
            result_dict = renamed

        for col in right_df.columns:
            if same_key_name and col == right_key:
                continue  # Already emitted from the left side above.
            dest_name = f"{col}_y" if col in overlap else col
            result_dict[dest_name] = _gather(right_df, col, right_idx_ms)

        if how == "inner":
            return ProxyDataFrame(_pandas_obj=pd.DataFrame(result_dict))

        left_idx_np = left_idx_ms.to_numpy()
        right_idx_np = right_idx_ms.to_numpy()

        if how == "left":
            matched = set(left_idx_np.tolist())
            unmatched = [i for i in range(len(obj)) if i not in matched]
            if unmatched:
                um = np.array(unmatched)
                for col in list(result_dict):
                    src_col = col.rstrip("_x") if col.endswith("_x") and col[:-2] in overlap else col
                    if src_col in obj.columns:
                        result_dict[col] = np.concatenate([result_dict[col], obj[src_col].values[um]])
                    else:
                        fill = np.full(len(um), np.nan)
                        result_dict[col] = np.concatenate([result_dict[col], fill])
            return ProxyDataFrame(_pandas_obj=pd.DataFrame(result_dict))

        if how == "right":
            matched = set(right_idx_np.tolist())
            unmatched = [i for i in range(len(right_df)) if i not in matched]
            if unmatched:
                um = np.array(unmatched)
                key_col_name = left_key if same_key_name else (f"{left_key}_x" if left_key in overlap else left_key)
                for col in list(result_dict):
                    src_col = col.rstrip("_y") if col.endswith("_y") and col[:-2] in overlap else col
                    if src_col in right_df.columns:
                        result_dict[col] = np.concatenate([result_dict[col], right_df[src_col].values[um]])
                    elif col == key_col_name and same_key_name:
                        result_dict[col] = np.concatenate([result_dict[col], right_df[right_key].values[um]])
                    else:
                        fill = np.full(len(um), np.nan)
                        result_dict[col] = np.concatenate([result_dict[col], fill])
            return ProxyDataFrame(_pandas_obj=pd.DataFrame(result_dict))

        return ProxyDataFrame(_pandas_obj=pd.DataFrame(result_dict))

    def to_pandas(self) -> Any:
        """Unwrap to the real pandas object."""
        self._flush_pending()
        return object.__getattribute__(self, "_pandas_obj")

    def __repr__(self) -> str:
        self._flush_pending()
        obj = object.__getattribute__(self, "_pandas_obj")
        if obj is not None:
            return f"ProxyDataFrame(\n{repr(obj)}\n)"
        return "ProxyDataFrame(_pandas_obj=None)"

    def __str__(self) -> str:
        self._flush_pending()
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

    def _try_metal_unary(self, op_name: str, np_func: Any) -> Any:
        from metaldf._deferred import DeferredSeries, LoadColumn, UnaryOp, _can_defer

        if _can_defer(self):
            return DeferredSeries(UnaryOp(op_name, LoadColumn(self)), len(self))

        try:
            import metaldf_engine
            from metaldf._engine._metal import _extract_array, _make_series

            arr = _extract_array(self._pandas_obj)
            ms = _make_series(arr)
            result = metaldf_engine.metal_unary_op(op_name, ms)
            return _wrap_result(
                pd.Series(result.to_numpy(), index=self._pandas_obj.index, name=self._pandas_obj.name)
            )
        except Exception:
            return _wrap_result(
                pd.Series(np_func(self._pandas_obj.values), index=self._pandas_obj.index, name=self._pandas_obj.name)
            )

    def sin(self) -> Any: return self._try_metal_unary("sin", np.sin)
    def cos(self) -> Any: return self._try_metal_unary("cos", np.cos)
    def tan(self) -> Any: return self._try_metal_unary("tan", np.tan)
    def asin(self) -> Any: return self._try_metal_unary("asin", np.arcsin)
    def acos(self) -> Any: return self._try_metal_unary("acos", np.arccos)
    def atan(self) -> Any: return self._try_metal_unary("atan", np.arctan)
    def sinh(self) -> Any: return self._try_metal_unary("sinh", np.sinh)
    def cosh(self) -> Any: return self._try_metal_unary("cosh", np.cosh)
    def tanh(self) -> Any: return self._try_metal_unary("tanh", np.tanh)
    def log2(self) -> Any: return self._try_metal_unary("log2", np.log2)
    def log10(self) -> Any: return self._try_metal_unary("log10", np.log10)
    def trunc(self) -> Any: return self._try_metal_unary("trunc", np.trunc)
    def cbrt(self) -> Any: return self._try_metal_unary("cbrt", np.cbrt)

    def round(self, *args: Any, **kwargs: Any) -> Any:
        if args or kwargs:
            return _wrap_result(pd.Series.round(self._pandas_obj, *args, **kwargs))
        return self._try_metal_unary("round", np.round)

    _UFUNC_MAP = {
        np.sin: "sin", np.cos: "cos", np.tan: "tan",
        np.arcsin: "asin", np.arccos: "acos", np.arctan: "atan",
        np.sinh: "sinh", np.cosh: "cosh", np.tanh: "tanh",
        np.log2: "log2", np.log10: "log10",
        np.sqrt: "sqrt", np.exp: "exp", np.log: "log",
        np.abs: "abs", np.ceil: "ceil", np.floor: "floor",
        np.trunc: "trunc", np.cbrt: "cbrt",
        np.round: "round",
    }

    def __array_ufunc__(self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if method == "__call__" and not kwargs:
            op_name = self._UFUNC_MAP.get(ufunc)
            if op_name is not None and len(inputs) == 1:
                return self._try_metal_unary(op_name, ufunc)
        return pd.Series.__array_ufunc__(self, ufunc, method, *inputs, **kwargs)

    def _try_metal_series_op(self, op_name: str, *args: Any, **kwargs: Any) -> Any:
        """Try Metal path for a series-returning operation, fall back to pandas."""
        from metaldf._engine import execute
        from metaldf._wrappers import _wrap_result
        from metaldf.exceptions import MetalNotAvailable

        try:
            result = execute(op_name, self._pandas_obj, *args, **kwargs)
            return _wrap_result(result)
        except (MetalNotAvailable, Exception):
            pandas_method = getattr(pd.Series, op_name, None)
            if pandas_method is None:
                raise AttributeError(f"'{type(self).__name__}' has no attribute '{op_name}'")
            result = pandas_method(self._pandas_obj, *args, **kwargs)
            return _wrap_result(result)

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

    def cumsum(self, *args: Any, **kwargs: Any) -> Any:
        return self._try_metal_series_op("cumsum", *args, **kwargs)

    def cummin(self, *args: Any, **kwargs: Any) -> Any:
        return self._try_metal_series_op("cummin", *args, **kwargs)

    def cummax(self, *args: Any, **kwargs: Any) -> Any:
        return self._try_metal_series_op("cummax", *args, **kwargs)

    def shift(self, periods: int = 1, **kwargs: Any) -> Any:
        return self._try_metal_series_op("shift", periods=periods)

    def diff(self, periods: int = 1, **kwargs: Any) -> Any:
        from metaldf._wrappers import _wrap_result
        try:
            shifted = self.shift(periods)
            return self - shifted
        except Exception:
            result = pd.Series.diff(self._pandas_obj, periods=periods, **kwargs)
            return _wrap_result(result)

    def pct_change(self, periods: int = 1, **kwargs: Any) -> Any:
        from metaldf._wrappers import _wrap_result
        try:
            shifted = self.shift(periods)
            return (self - shifted) / shifted
        except Exception:
            result = pd.Series.pct_change(self._pandas_obj, periods=periods, **kwargs)
            return _wrap_result(result)

    def fillna(self, value: Any = None, **kwargs: Any) -> Any:
        """Fill NaN values, trying Metal for a scalar float32 fill.

        Reads off ``self`` directly (not ``self._pandas_obj``) rather than
        going through ``_try_metal_series_op``: unlike ``ProxyDataFrame``,
        ``ProxySeries`` has no ``__setitem__``/``.loc`` override that keeps
        ``_pandas_obj`` in sync with in-place mutations (see
        ``ProxySeries.__getitem__``'s docstring for the same issue) -- an
        in-place ``.loc[i] = None`` before calling ``fillna`` would otherwise
        silently operate on stale data.
        """
        from metaldf._engine import execute
        from metaldf._wrappers import _wrap_result
        from metaldf.exceptions import MetalNotAvailable

        current = pd.Series(self._values, index=self.index, name=self.name)
        if value is not None and np.isscalar(value) and not kwargs.get("method"):
            try:
                result = execute("fillna", current, value=value)
                return _wrap_result(result)
            except (MetalNotAvailable, Exception):
                pass
        result = pd.Series.fillna(current, value=value, **kwargs)
        return _wrap_result(result)

    def ffill(self, **kwargs: Any) -> Any:
        return self._try_metal_series_op("ffill")

    def bfill(self, **kwargs: Any) -> Any:
        return self._try_metal_series_op("bfill")

    def pad(self, **kwargs: Any) -> Any:
        return self.ffill(**kwargs)

    def backfill(self, **kwargs: Any) -> Any:
        return self.bfill(**kwargs)

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

    def rolling(self, window: Any, **kwargs: Any) -> ProxyRolling:
        """Return a rolling window over the series as a ``ProxyRolling``.

        Without this override, ``series.rolling(...)`` would resolve via
        normal MRO straight to ``pd.Series.rolling`` (``ProxySeries``
        doesn't otherwise define it), returning a plain
        ``pandas.core.window.Rolling`` that never gets a chance to try the
        Metal kernels below -- see ``ProxyRolling._try_metal_rolling``.
        """
        return ProxyRolling(self, window=window, **kwargs)

    def __getitem__(self, key: Any) -> Any:
        """Filter by boolean mask (``series[mask]``), trying the GPU ``compact`` kernel first.

        Mirrors ``ProxyDataFrame.__getitem__``: any key that isn't a
        bool-dtype ``Series``/array (label, integer position, slice, ...) is
        left untouched and handed to the real ``pd.Series.__getitem__``.
        """
        from metaldf._deferred import DeferredSeries

        if isinstance(key, DeferredSeries):
            key = key.to_pandas()

        is_bool_mask = False
        if isinstance(key, (pd.Series, np.ndarray)):
            arr = np.asarray(key)
            if arr.dtype == np.dtype(np.bool_):
                is_bool_mask = True
        elif type(key) is ProxySeries:
            pandas_key = key.to_pandas()
            arr = np.asarray(pandas_key)
            if arr.dtype == np.dtype(np.bool_):
                is_bool_mask = True
                key = pandas_key

        if is_bool_mask:
            try:
                from metaldf._engine import execute
                # Read the current data off `self` (the real, unbound
                # `_values`/`index`/`name`), not `self._pandas_obj`: unlike
                # `ProxyDataFrame.__setitem__`, `ProxySeries` has no
                # `__setitem__` override that keeps `_pandas_obj` in sync,
                # so any in-place mutation (`.loc[...] = ...`, `series[i] =
                # ...`) only updates `self`'s own data -- `_pandas_obj`
                # would silently go stale (see
                # `tests/exhaustive/test_series_comprehensive.py::
                # TestSeriesIndexing::test_setitem_scalar` and
                # `TestSeriesCleaning::test_dropna`, both of which mutate
                # via `.loc`/`[]=` and then immediately read back through
                # `__getitem__`/a method that filters internally).
                current = pd.Series(self._values, index=self.index, name=self.name)
                result = execute("compact", current, key)
                return _wrap_result(result)
            except Exception:
                pass

        # Fallback to pandas -- through `self` for the same read-after-write
        # reason as above.
        result = pd.Series.__getitem__(self, key)
        return _wrap_result(result)

    @property
    def str(self) -> ProxyStringAccessor:
        """Accessor for vectorized string methods, e.g. ``series.str.contains(...)``.

        Tries Metal GPU dispatch first (see ``ProxyStringAccessor``), falling
        back to the real ``pd.Series.str`` accessor for unsupported dtypes,
        small arrays, or methods without a Metal kernel.
        """
        return ProxyStringAccessor(self)

    @property
    def dt(self) -> Any:
        """Accessor for vectorized datetime methods, e.g. ``series.dt.year``.

        Tries Metal GPU dispatch first (see ``ProxyDatetimeAccessor``),
        falling back to the real ``pd.Series.dt`` accessor for non-datetime
        dtypes or components without a Metal kernel. Mirrors the ``str``
        property above: only routes through ``ProxyDatetimeAccessor`` when
        ``_pandas_obj`` is actually datetime64-dtyped, otherwise hands back
        the real pandas accessor directly (bypassing the proxy layer
        entirely) so unsupported dtypes get pandas' own error messages
        rather than a confusing detour through this accessor.
        """
        pandas_obj = object.__getattribute__(self, "_pandas_obj")
        if hasattr(pandas_obj, "dtype") and pd.api.types.is_datetime64_any_dtype(pandas_obj):
            return ProxyDatetimeAccessor(self)
        return pd.Series.__dict__["dt"].__get__(pandas_obj, pd.Series)

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
        from metaldf._engine._metal import _is_string_dtype, is_metal_available
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


class ProxyDatetimeAccessor:
    """Proxy for ``Series.dt`` accessor -- tries Metal, falls back to pandas.

    Each explicit property here (year/month/day/hour/minute/second/
    dayofweek) dispatches to a ``metaldf_engine.metal_dt_*`` GPU kernel
    (see ``rust/src/kernels/datetime.rs``) that extracts the calendar
    component straight from the series' raw int64 nanosecond-since-epoch
    values -- civil calendar reconstruction (year/month/day) via Howard
    Hinnant's ``civil_from_days`` algorithm, everything else via plain
    floor-division/floor-modulo. Any failure (Metal not built, unsupported
    dtype, ...) is caught and re-run through the real ``pd.Series.dt``
    accessor. Any other ``.dt`` attribute (e.g. ``date``, ``strftime``)
    simply falls through to pandas via ``__getattr__``.

    NOTE(DT-1 merge): this worktree branched before Task 1's
    ``DType::Datetime``/``MetalSeries.from_numpy_datetime`` landed, so
    ``_dispatch`` below builds the GPU series manually via
    ``MetalSeries.from_numpy_i64`` on the datetime64 array's int64 view,
    rather than going through ``metaldf._engine._metal``'s
    ``_extract_array``/``_make_series`` (which don't yet know about
    datetime64 dtypes on this branch). Once merged onto a branch carrying
    Task 1, this can be simplified to reuse those helpers directly.
    """

    def __init__(self, series: ProxySeries) -> None:
        self._series = series

    def _real_dt_accessor(self) -> Any:
        """Get the real ``pd.Series.dt`` accessor, bypassing ``ProxySeries.dt``.

        Same reasoning as ``ProxyStringAccessor._real_str_accessor``: for a
        directly-constructed ``ProxySeries``, ``_pandas_obj is self``, so
        ``_pandas_obj.dt`` would re-enter ``ProxySeries.dt`` and recurse
        forever the first time a fallback is needed. Pulling the raw
        ``CachedAccessor`` descriptor straight off ``pd.Series`` always
        binds to the real pandas ``DatetimeProperties`` implementation.
        """
        pandas_obj = self._series._pandas_obj
        return pd.Series.__dict__["dt"].__get__(pandas_obj, pd.Series)

    def _dispatch(self, component: str) -> Any:
        try:
            import metaldf_engine
            from metaldf._engine._metal import is_metal_available

            if not is_metal_available():
                raise Exception("Metal not available")

            pandas_obj = self._series._pandas_obj
            if not hasattr(pandas_obj, "dtype") or not pd.api.types.is_datetime64_any_dtype(pandas_obj):
                raise Exception("Not datetime")

            ns = np.asarray(pandas_obj._values).astype("datetime64[ns]").view(np.int64)
            if not ns.flags["C_CONTIGUOUS"]:
                ns = np.ascontiguousarray(ns)
            ms = metaldf_engine.MetalSeries.from_numpy_i64(ns)
            fn = getattr(metaldf_engine, f"metal_dt_{component}")
            result = fn(ms)
            return _wrap_result(
                pd.Series(result.to_numpy(), index=pandas_obj.index, name=pandas_obj.name)
            )
        except Exception:
            return _wrap_result(getattr(self._real_dt_accessor(), component))

    @property
    def year(self) -> Any: return self._dispatch("year")

    @property
    def month(self) -> Any: return self._dispatch("month")

    @property
    def day(self) -> Any: return self._dispatch("day")

    @property
    def hour(self) -> Any: return self._dispatch("hour")

    @property
    def minute(self) -> Any: return self._dispatch("minute")

    @property
    def second(self) -> Any: return self._dispatch("second")

    @property
    def dayofweek(self) -> Any: return self._dispatch("dayofweek")

    @property
    def quarter(self) -> Any: return self._dispatch("quarter")

    @property
    def dayofyear(self) -> Any: return self._dispatch("dayofyear")

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._real_dt_accessor(), name)
        if callable(attr):
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return _wrap_result(attr(*args, **kwargs))
            return wrapper
        return _wrap_result(attr) if isinstance(attr, (pd.Series, pd.DataFrame)) else attr


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


def _unwrap_pandas(obj: Any) -> Any:
    """Return a genuinely plain (non-proxy-subclass) pandas Series/DataFrame.

    ``obj.to_pandas()`` returns ``self`` for a directly-constructed proxy
    (``_pandas_obj is self`` -- see ``ProxyDataFrame.__init__``/
    ``ProxySeries.__init__``), which is still an instance of the proxy
    subclass, not a real ``pd.DataFrame``/``pd.Series``. Handing such an
    instance to genuine pandas machinery (e.g. ``pd.DataFrame.rolling``)
    risks pandas' internal ``_constructor`` mechanism building further
    instances of the proxy subclass for intermediate/final results --
    bypassing our custom ``__init__`` (the only place ``_pandas_obj`` gets
    set) entirely, so any later ``.to_pandas()``/``__getattr__`` access on
    those results raises ``AttributeError: _pandas_obj``.

    Rebuilds a fresh, real ``pd.DataFrame``/``pd.Series`` from the same
    underlying column arrays/values, index and name/columns when `obj` is a
    proxy subclass instance (``type(obj) is not`` the real pandas type);
    passes through unchanged for anything else (plain pandas objects,
    scalars, arrays, ...).
    """
    if hasattr(obj, "to_pandas"):
        obj = obj.to_pandas()
    if isinstance(obj, pd.DataFrame) and type(obj) is not pd.DataFrame:
        cols = {col: np.asarray(pd.DataFrame.__getitem__(obj, col)) for col in obj.columns}
        obj = pd.DataFrame(cols, index=obj.index)
    elif isinstance(obj, pd.Series) and type(obj) is not pd.Series:
        obj = pd.Series(obj._values, index=obj.index, name=obj.name)
    return obj


class ProxyRolling:
    """Proxy for Series.rolling() / DataFrame.rolling().

    Similar to ProxyGroupBy: records the rolling window and replays.
    Explicit aggregations (sum/mean/min/max/count) try the Metal rolling
    kernels first (see ``_try_metal_rolling``) for the common
    ``series.rolling(window).mean()`` pattern; everything else
    (std/var/apply/median/...) is recorded via ``__getattr__`` and replayed
    on pandas.
    """

    # Ops with a Metal kernel: metaldf_engine.metal_rolling_<op>.
    _METAL_OPS = ("sum", "mean", "min", "max", "count")

    def __init__(self, obj: Any, window: Any, **kwargs: Any) -> None:
        self._obj = obj
        self._window = window
        self._rolling_kwargs = kwargs
        self._method_chain: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _real_rolling(self) -> Any:
        """Build a genuine pandas ``Rolling`` object for replay/fallback.

        Calls ``pd.Series.rolling``/``pd.DataFrame.rolling`` directly
        (rather than ``self._obj.rolling(...)``) because ``ProxySeries``
        and ``ProxyDataFrame`` both now define their own ``.rolling()``
        that returns a ``ProxyRolling`` -- going through ``self._obj``
        would just construct another ``ProxyRolling`` and recurse forever
        the moment any fallback/replay path here is exercised.

        Also rebuilds a genuinely plain (non-proxy-subclass) pandas object
        via ``_unwrap_pandas`` before handing it to pandas' own rolling
        machinery. ``obj.to_pandas()`` returns ``self`` for a
        directly-constructed proxy (``_pandas_obj is self`` -- see e.g.
        ``ProxyDataFrame.__init__``), which is still an instance of the
        proxy *subclass*. Running ``pd.DataFrame.rolling``/``.sum()`` on
        that risks pandas' internal ``_constructor`` machinery building
        further instances of the proxy subclass to hold intermediate/final
        results -- bypassing our custom ``__init__`` (the only place that
        sets ``_pandas_obj``) entirely, so anything later touching
        ``.to_pandas()``/``__getattr__`` on those results raises
        ``AttributeError: _pandas_obj``. Rebuilding a fresh, genuinely plain
        object from the same underlying data/index/(columns/name) up front
        means pandas only ever constructs real ``pd.DataFrame``/``pd.Series``
        instances downstream.
        """
        pandas_obj = _unwrap_pandas(self._obj)
        if isinstance(pandas_obj, pd.DataFrame):
            return pd.DataFrame.rolling(pandas_obj, window=self._window, **self._rolling_kwargs)
        if isinstance(pandas_obj, pd.Series):
            return pd.Series.rolling(pandas_obj, window=self._window, **self._rolling_kwargs)
        return pandas_obj.rolling(window=self._window, **self._rolling_kwargs)

    def _replay(self) -> Any:
        rolling = self._real_rolling()
        for method_name, args, kwargs in self._method_chain:
            rolling = getattr(rolling, method_name)(*args, **kwargs)
        return rolling

    def _try_metal_rolling(self, op_name: str) -> Any:
        """Try GPU rolling, fall back to pandas.

        Only attempts the Metal path for a plain float32 ``Series`` with no
        NaNs and no rolling kwargs beyond ``min_periods`` (``center``,
        ``win_type``, ``closed``, ``step``, ``method``, ... all change
        semantics the naive per-position GPU kernel doesn't replicate).

        The kernel itself always computes a partial aggregate for the
        ramp-up region at the start of the series (i.e. it behaves like
        pandas' ``min_periods=1``) -- see ``rust/src/kernels/rolling.rs``
        and ``tests/test_rolling.py``. But plain ``series.rolling(window)``
        (no explicit ``min_periods``) actually defaults to
        ``min_periods=window`` in pandas (NaN until the window fully
        fills), *not* 1. To reproduce that default (and any explicit
        ``min_periods`` the caller passed) exactly, the GPU result is
        post-masked to NaN wherever the number of observations actually
        available at that position (``min(idx + 1, window)``) is below the
        requested ``min_periods`` -- cheap, plain-numpy work, done once on
        the whole result array.

        NaN inputs are routed straight to pandas rather than through Metal:
        the kernel has no null-aware skip-and-recount logic (unlike the
        reduction/groupby kernels' ``_make_series_with_nulls`` path), so a
        NaN in a window would either propagate incorrectly or be silently
        counted as a real observation instead of being excluded the way
        pandas' default ``skipna=True`` rolling behavior requires.
        """
        try:
            import metaldf_engine

            from metaldf._engine._metal import _extract_array, _make_series, is_metal_available

            if not is_metal_available():
                raise Exception("Metal not available")

            extra_kwargs = set(self._rolling_kwargs) - {"min_periods"}
            if extra_kwargs:
                raise Exception(f"Unsupported rolling kwargs for Metal: {sorted(extra_kwargs)}")

            pandas_obj = _unwrap_pandas(self._obj)
            if not isinstance(pandas_obj, pd.Series):
                raise Exception("Metal rolling only supports Series")

            # Only float32 supported.
            arr = _extract_array(pandas_obj)
            if arr.dtype != np.dtype(np.float32):
                raise Exception("Only float32 supported")
            if np.isnan(arr).any():
                raise Exception("Metal rolling doesn't support NaN values")

            ms = _make_series(arr)
            fn = getattr(metaldf_engine, f"metal_rolling_{op_name}")
            result = fn(ms, self._window)
            result_arr = result.to_numpy()

            # Match pandas' actual min_periods semantics (see docstring
            # above) -- mask ramp-up positions the kernel filled in but
            # that don't meet the requested min_periods.
            min_periods = self._rolling_kwargs.get("min_periods")
            if min_periods is None:
                min_periods = self._window
            if min_periods > 1:
                valid_count = np.minimum(np.arange(1, len(result_arr) + 1), self._window)
                result_arr = result_arr.astype(np.float64, copy=True)
                result_arr[valid_count < min_periods] = np.nan

            return _wrap_result(
                pd.Series(result_arr, index=pandas_obj.index, name=pandas_obj.name)
            )
        except Exception:
            # Fall back to pandas.
            rolling = self._real_rolling()
            return _wrap_result(getattr(rolling, op_name)())

    def sum(self, *args: Any, **kwargs: Any) -> Any:
        if args or kwargs:
            return _wrap_result(self._replay_agg("sum", args, kwargs))
        return self._try_metal_rolling("sum")

    def mean(self, *args: Any, **kwargs: Any) -> Any:
        if args or kwargs:
            return _wrap_result(self._replay_agg("mean", args, kwargs))
        return self._try_metal_rolling("mean")

    def min(self, *args: Any, **kwargs: Any) -> Any:
        if args or kwargs:
            return _wrap_result(self._replay_agg("min", args, kwargs))
        return self._try_metal_rolling("min")

    def max(self, *args: Any, **kwargs: Any) -> Any:
        if args or kwargs:
            return _wrap_result(self._replay_agg("max", args, kwargs))
        return self._try_metal_rolling("max")

    def count(self, *args: Any, **kwargs: Any) -> Any:
        if args or kwargs:
            return _wrap_result(self._replay_agg("count", args, kwargs))
        return self._try_metal_rolling("count")

    def _replay_agg(self, agg_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        rolling = self._real_rolling()
        return getattr(rolling, agg_name)(*args, **kwargs)

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

        if name == "merge" and callable(attr):
            return _wrap_merge(attr)

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


def _wrap_merge(real_merge_func: Any) -> Any:
    """Route module-level ``pandas.merge`` through ``ProxyDataFrame.merge`` when possible.

    Also dispatches through the Metal hash-join fast path when the left
    operand is a proxy.

    pandas' ``DataFrame.merge()`` method is a thin wrapper that calls the
    module-level ``merge()`` function -- not the other way around -- so
    without this, ``pd.merge(proxy_df, other)`` would run pandas' merge
    algorithm directly against the two operands and never reach
    ``ProxyDataFrame.merge``'s GPU dispatch. Routing through
    ``left.merge(right, ...)`` whenever `left` exposes a callable ``.merge``
    (true for both ``ProxyDataFrame`` and plain ``pd.DataFrame``, since both
    inherit/define one) makes the module-level call form behave identically
    to ``left.merge(right, ...)``.
    """

    def wrapper(
        left: Any,
        right: Any,
        how: str = "inner",
        on: Any = None,
        left_on: Any = None,
        right_on: Any = None,
        **kwargs: Any,
    ) -> Any:
        merge_method = getattr(left, "merge", None)
        if callable(merge_method):
            result = merge_method(
                right, how=how, on=on, left_on=left_on, right_on=right_on, **kwargs
            )
        else:
            result = real_merge_func(
                left, right, how=how, on=on, left_on=left_on, right_on=right_on, **kwargs
            )
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
