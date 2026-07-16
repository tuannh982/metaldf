"""Lazy expression graph for fused GPU execution.

Builds an AST of element-wise operations. Materialization compiles the
tree to bytecode and dispatches it as a single Metal kernel, preferring
the runtime-compiled codegen path (``metaldf_engine.eval_expression_codegen``)
and falling back to the bytecode interpreter (``metaldf_engine.eval_expression``)
if codegen fails.
"""

from __future__ import annotations

import struct
from typing import Any

import numpy as np
import pandas as pd


# Opcodes — must match rust/metal/expression/eval.metal
OP_LOAD_SCALAR = 8
OP_ADD = 16
OP_SUB = 17
OP_MUL = 18
OP_DIV = 19
OP_MOD = 20
OP_EQ = 24
OP_NE = 25
OP_LT = 26
OP_LE = 27
OP_GT = 28
OP_GE = 29
OP_ABS = 32
OP_NEG = 33
OP_SQRT = 34
OP_EXP = 35
OP_LOG = 36
OP_CEIL = 37
OP_FLOOR = 38
OP_AND = 40
OP_OR = 41
OP_NOT = 42
OP_SIN = 43
OP_COS = 44
OP_TAN = 45
OP_ASIN = 46
OP_ACOS = 47
OP_ATAN = 48
OP_SINH = 49
OP_COSH = 50
OP_TANH = 51
OP_LOG2 = 52
OP_LOG10 = 53
OP_ROUND = 54
OP_TRUNC = 55
OP_CBRT = 56

_BINARY_OPS = {
    "add": OP_ADD, "sub": OP_SUB, "mul": OP_MUL, "div": OP_DIV, "mod": OP_MOD,
    "eq": OP_EQ, "ne": OP_NE, "lt": OP_LT, "le": OP_LE, "gt": OP_GT, "ge": OP_GE,
    "and": OP_AND, "or": OP_OR,
}

_UNARY_OPS = {
    "abs": OP_ABS, "neg": OP_NEG, "sqrt": OP_SQRT, "exp": OP_EXP, "log": OP_LOG,
    "ceil": OP_CEIL, "floor": OP_FLOOR, "not": OP_NOT,
    "sin": OP_SIN, "cos": OP_COS, "tan": OP_TAN,
    "asin": OP_ASIN, "acos": OP_ACOS, "atan": OP_ATAN,
    "sinh": OP_SINH, "cosh": OP_COSH, "tanh": OP_TANH,
    "log2": OP_LOG2, "log10": OP_LOG10,
    "round": OP_ROUND, "trunc": OP_TRUNC, "cbrt": OP_CBRT,
}


class ExprNode:
    pass


class LoadColumn(ExprNode):
    def __init__(self, series: Any) -> None:
        self.series = series


class LoadScalar(ExprNode):
    def __init__(self, value: float) -> None:
        self.value = float(value)


class BinaryOp(ExprNode):
    def __init__(self, op: str, left: ExprNode, right: ExprNode) -> None:
        self.op = op
        self.left = left
        self.right = right


class UnaryOp(ExprNode):
    def __init__(self, op: str, child: ExprNode) -> None:
        self.op = op
        self.child = child


def _can_defer(obj: Any) -> bool:
    """Check if an object can participate in deferred expressions.

    Conservative on purpose: only float32 ``ProxySeries`` (the only dtype
    the bytecode-interpreter Metal kernel currently supports), plain
    Python scalars, and other ``DeferredSeries`` qualify. int32/int64
    series, non-float scalars (e.g. numpy scalar types beyond
    int/float), and anything else fall back to eager dispatch/pandas.
    """
    if isinstance(obj, DeferredSeries):
        return True
    if isinstance(obj, bool):
        # bool is an int subclass -- exclude explicitly since it's not a
        # meaningful float32 scalar operand.
        return False
    if isinstance(obj, (int, float)):
        return True
    from metaldf._wrappers import ProxySeries
    if isinstance(obj, ProxySeries) and hasattr(obj, "_pandas_obj"):
        pandas_obj = object.__getattribute__(obj, "_pandas_obj")
        return hasattr(pandas_obj, "dtype") and pandas_obj.dtype in (
            np.dtype(np.float32),
        )
    return False


def _as_node(obj: Any) -> ExprNode:
    """Convert an object to an expression node."""
    if isinstance(obj, DeferredSeries):
        return obj.root
    if isinstance(obj, (int, float)):
        return LoadScalar(obj)
    from metaldf._wrappers import ProxySeries
    if isinstance(obj, ProxySeries) and hasattr(obj, "_pandas_obj"):
        return LoadColumn(obj)
    raise TypeError(f"Cannot convert {type(obj)} to expression node")


class DeferredSeries:
    """A series whose value is defined by an expression tree, not yet computed.

    Arithmetic on a ``DeferredSeries`` (or on a ``ProxySeries`` where both
    operands qualify -- see ``_can_defer``) builds a larger tree rather
    than dispatching a kernel immediately, so chains like ``(a + b) * c``
    fuse into a single Metal kernel launch. The tree is only compiled to
    bytecode and executed on data access (``to_pandas``/``to_numpy``/
    ``.values``/reductions/``repr``/``str``/``len``/attribute fallthrough).
    """

    def __init__(self, root: ExprNode, size: int) -> None:
        self.root = root
        self.size = size

    def _collect_columns(self) -> list[Any]:
        """Collect all unique LoadColumn series from the tree (in order of first appearance)."""
        columns: list[Any] = []
        seen: set[int] = set()

        def walk(node: ExprNode) -> None:
            if isinstance(node, LoadColumn):
                obj_id = id(node.series)
                if obj_id not in seen:
                    seen.add(obj_id)
                    columns.append(node.series)
            elif isinstance(node, BinaryOp):
                walk(node.left)
                walk(node.right)
            elif isinstance(node, UnaryOp):
                walk(node.child)

        walk(self.root)
        return columns

    def _compile_bytecode(self, column_index: dict[int, int]) -> bytes:
        """Compile the expression tree to bytecode (post-order: children before parent)."""
        program = bytearray()

        def emit(node: ExprNode) -> None:
            if isinstance(node, LoadColumn):
                idx = column_index[id(node.series)]
                program.append(idx)
            elif isinstance(node, LoadScalar):
                program.append(OP_LOAD_SCALAR)
                program.extend(struct.pack("<f", node.value))
            elif isinstance(node, BinaryOp):
                emit(node.left)
                emit(node.right)
                program.append(_BINARY_OPS[node.op])
            elif isinstance(node, UnaryOp):
                emit(node.child)
                program.append(_UNARY_OPS[node.op])

        emit(self.root)
        return bytes(program)

    def _materialize(self) -> pd.Series:
        """Compile and execute the expression tree on GPU.

        Tries codegen (runtime MSL compilation) first for best performance.
        Falls back to bytecode interpreter if codegen fails.
        """
        import metaldf_engine
        from metaldf._engine._metal import _extract_array, _make_series

        columns = self._collect_columns()
        column_index = {id(col): i for i, col in enumerate(columns)}
        program = self._compile_bytecode(column_index)

        metal_cols = []
        for col in columns:
            pandas_obj = object.__getattribute__(col, "_pandas_obj")
            arr = _extract_array(pandas_obj)
            metal_cols.append(_make_series(arr))

        # Try codegen first (compiled kernel), fall back to interpreter
        try:
            result = metaldf_engine.eval_expression_codegen(program, metal_cols, self.size)
        except Exception:
            result = metaldf_engine.eval_expression(program, metal_cols, self.size)

        return pd.Series(result.to_numpy())

    def to_pandas(self) -> pd.Series:
        return self._materialize()

    def to_numpy(self) -> np.ndarray:
        return self._materialize().to_numpy()

    @property
    def values(self) -> np.ndarray:
        return self.to_numpy()

    # Arithmetic — chain more deferred ops
    def __add__(self, other: Any) -> DeferredSeries:
        if _can_defer(other):
            return DeferredSeries(BinaryOp("add", self.root, _as_node(other)), self.size)
        return NotImplemented

    def __radd__(self, other: Any) -> DeferredSeries:
        if _can_defer(other):
            return DeferredSeries(BinaryOp("add", _as_node(other), self.root), self.size)
        return NotImplemented

    def __sub__(self, other: Any) -> DeferredSeries:
        if _can_defer(other):
            return DeferredSeries(BinaryOp("sub", self.root, _as_node(other)), self.size)
        return NotImplemented

    def __rsub__(self, other: Any) -> DeferredSeries:
        if _can_defer(other):
            return DeferredSeries(BinaryOp("sub", _as_node(other), self.root), self.size)
        return NotImplemented

    def __mul__(self, other: Any) -> DeferredSeries:
        if _can_defer(other):
            return DeferredSeries(BinaryOp("mul", self.root, _as_node(other)), self.size)
        return NotImplemented

    def __rmul__(self, other: Any) -> DeferredSeries:
        if _can_defer(other):
            return DeferredSeries(BinaryOp("mul", _as_node(other), self.root), self.size)
        return NotImplemented

    def __truediv__(self, other: Any) -> DeferredSeries:
        if _can_defer(other):
            return DeferredSeries(BinaryOp("div", self.root, _as_node(other)), self.size)
        return NotImplemented

    def __rtruediv__(self, other: Any) -> DeferredSeries:
        if _can_defer(other):
            return DeferredSeries(BinaryOp("div", _as_node(other), self.root), self.size)
        return NotImplemented

    # Unary math -- chain more deferred ops (Phase 5: trig/log/rounding).
    # Each wraps the current tree in a UnaryOp node rather than dispatching
    # immediately, so e.g. `(a + b).sin()` still fuses into a single kernel.
    def sin(self) -> DeferredSeries:
        return DeferredSeries(UnaryOp("sin", self.root), self.size)

    def cos(self) -> DeferredSeries:
        return DeferredSeries(UnaryOp("cos", self.root), self.size)

    def tan(self) -> DeferredSeries:
        return DeferredSeries(UnaryOp("tan", self.root), self.size)

    def asin(self) -> DeferredSeries:
        return DeferredSeries(UnaryOp("asin", self.root), self.size)

    def acos(self) -> DeferredSeries:
        return DeferredSeries(UnaryOp("acos", self.root), self.size)

    def atan(self) -> DeferredSeries:
        return DeferredSeries(UnaryOp("atan", self.root), self.size)

    def sinh(self) -> DeferredSeries:
        return DeferredSeries(UnaryOp("sinh", self.root), self.size)

    def cosh(self) -> DeferredSeries:
        return DeferredSeries(UnaryOp("cosh", self.root), self.size)

    def tanh(self) -> DeferredSeries:
        return DeferredSeries(UnaryOp("tanh", self.root), self.size)

    def log2(self) -> DeferredSeries:
        return DeferredSeries(UnaryOp("log2", self.root), self.size)

    def log10(self) -> DeferredSeries:
        return DeferredSeries(UnaryOp("log10", self.root), self.size)

    def round(self) -> DeferredSeries:
        return DeferredSeries(UnaryOp("round", self.root), self.size)

    def trunc(self) -> DeferredSeries:
        return DeferredSeries(UnaryOp("trunc", self.root), self.size)

    def cbrt(self) -> DeferredSeries:
        return DeferredSeries(UnaryOp("cbrt", self.root), self.size)

    def _fused_reduce(self, op: str) -> Any:
        """Try the fused expression-reduce kernel. Returns a scalar, or ``None``.

        ``None`` signals "not applicable / failed" -- either the root is a
        plain ``LoadColumn`` (no expression to fuse; the ordinary
        ``ProxySeries`` reduction already handles that case directly), any
        input column contains NaN (the fused kernel doesn't skip nulls, so
        we fall back to materialize-then-pandas-reduce which honours
        ``skipna=True``), or the fused kernel raised.
        """
        if isinstance(self.root, LoadColumn):
            return None

        import metaldf_engine
        from metaldf._engine._metal import _extract_array, _make_series

        columns = self._collect_columns()

        arrays = []
        for col in columns:
            pandas_obj = object.__getattribute__(col, "_pandas_obj")
            arr = _extract_array(pandas_obj)
            if arr.dtype == np.dtype(np.float32) and np.any(np.isnan(arr)):
                return None
            arrays.append(arr)

        column_index = {id(col): i for i, col in enumerate(columns)}
        program = self._compile_bytecode(column_index)

        metal_cols = [_make_series(arr) for arr in arrays]

        try:
            return metaldf_engine.eval_expression_reduce(op, program, metal_cols, self.size)
        except Exception:
            return None

    # Materialization triggers
    def sum(self, *args: Any, **kwargs: Any) -> Any:
        result = self._fused_reduce("sum")
        if result is not None:
            return result
        return self._materialize().sum(*args, **kwargs)

    def min(self, *args: Any, **kwargs: Any) -> Any:
        result = self._fused_reduce("min")
        if result is not None:
            return result
        return self._materialize().min(*args, **kwargs)

    def max(self, *args: Any, **kwargs: Any) -> Any:
        result = self._fused_reduce("max")
        if result is not None:
            return result
        return self._materialize().max(*args, **kwargs)

    def mean(self, *args: Any, **kwargs: Any) -> Any:
        result = self._fused_reduce("sum")
        if result is not None:
            return result / self.size
        return self._materialize().mean(*args, **kwargs)

    def cumsum(self, *args: Any, **kwargs: Any) -> Any:
        materialized = self._materialize()
        from metaldf._wrappers import ProxySeries
        ps = ProxySeries(_pandas_obj=materialized)
        return ps.cumsum(*args, **kwargs)

    def cummin(self, *args: Any, **kwargs: Any) -> Any:
        materialized = self._materialize()
        from metaldf._wrappers import ProxySeries
        ps = ProxySeries(_pandas_obj=materialized)
        return ps.cummin(*args, **kwargs)

    def cummax(self, *args: Any, **kwargs: Any) -> Any:
        materialized = self._materialize()
        from metaldf._wrappers import ProxySeries
        ps = ProxySeries(_pandas_obj=materialized)
        return ps.cummax(*args, **kwargs)

    def sort_values(self, ascending: bool = True, **kwargs: Any) -> Any:
        """Materialize the expression via fused codegen, then sort."""
        materialized = self._materialize()
        from metaldf._wrappers import ProxySeries
        ps = ProxySeries(_pandas_obj=materialized)
        return ps.sort_values(ascending=ascending, **kwargs)

    def __repr__(self) -> str:
        return repr(self._materialize())

    def __str__(self) -> str:
        return str(self._materialize())

    def __len__(self) -> int:
        return self.size

    def __getattr__(self, name: str) -> Any:
        return getattr(self._materialize(), name)
