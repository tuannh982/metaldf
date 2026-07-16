"""Engine registry for metaldf.

Operations are registered by name. execute() checks the registry first,
then falls back to PandasEngine.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from metaldf._engine._pandas import PandasEngine
from metaldf.exceptions import MetalNotAvailable

_registry: dict[str, Callable[..., Any]] = {}


def register(op_name: str, impl: Callable[..., Any]) -> None:
    """Register an implementation for an operation.

    Operations without a registered implementation fall through to
    PandasEngine. Metal kernels are auto-registered below when the
    ``metaldf_engine`` extension module is importable.
    """
    _registry[op_name] = impl


def execute(op_name: str, *args: Any, **kwargs: Any) -> Any:
    """Execute an operation, trying the registry first, then PandasEngine."""
    if op_name in _registry:
        try:
            return _registry[op_name](*args, **kwargs)
        except MetalNotAvailable:
            pass
    return PandasEngine.execute(op_name, *args, **kwargs)


def clear_registry() -> None:
    """Clear all registered operations. Used primarily in tests."""
    _registry.clear()


# Auto-register Metal kernels if available
try:
    from metaldf._engine._metal import MetalEngine

    register("sum", MetalEngine.metal_sum)
    register("min", MetalEngine.metal_min)
    register("max", MetalEngine.metal_max)
    register("mean", MetalEngine.metal_mean)
    register("cumsum", MetalEngine.metal_cumsum)
    register("cummin", MetalEngine.metal_cummin)
    register("cummax", MetalEngine.metal_cummax)
    register("shift", MetalEngine.metal_shift)
    register("fillna", MetalEngine.metal_fillna)
    register("ffill", MetalEngine.metal_ffill)
    register("bfill", MetalEngine.metal_bfill)
    register("add", MetalEngine.metal_add)
    register("sub", MetalEngine.metal_sub)
    register("mul", MetalEngine.metal_mul)
    register("div", MetalEngine.metal_div)
    register("cmp_eq", MetalEngine.metal_cmp_eq)
    register("cmp_ne", MetalEngine.metal_cmp_ne)
    register("cmp_lt", MetalEngine.metal_cmp_lt)
    register("cmp_le", MetalEngine.metal_cmp_le)
    register("cmp_gt", MetalEngine.metal_cmp_gt)
    register("cmp_ge", MetalEngine.metal_cmp_ge)
    register("sort", MetalEngine.metal_sort)
    register("argsort", MetalEngine.metal_argsort)
    register("groupby_sum", MetalEngine.metal_groupby_sum)
    register("groupby_mean", MetalEngine.metal_groupby_mean)
    register("groupby_min", MetalEngine.metal_groupby_min)
    register("groupby_max", MetalEngine.metal_groupby_max)
    register("groupby_count", MetalEngine.metal_groupby_count)
    register("str_contains", MetalEngine.metal_string_contains)
    register("str_startswith", MetalEngine.metal_string_startswith)
    register("str_endswith", MetalEngine.metal_string_endswith)
    register("str_find", MetalEngine.metal_string_find)
    register("str_lower", MetalEngine.metal_string_lower)
    register("str_upper", MetalEngine.metal_string_upper)
    register("str_strip", MetalEngine.metal_string_strip)
    register("str_replace", MetalEngine.metal_string_replace)
    register("compact", MetalEngine.metal_compact)
except ImportError:
    pass  # Metal not built yet
