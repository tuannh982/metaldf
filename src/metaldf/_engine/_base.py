"""Base engine protocol.

PandasEngine is the fallback engine; MetalEngine (see ``_metal.py``)
registers GPU-accelerated implementations for supported operations.
"""

from __future__ import annotations

from typing import Any, Protocol


class Engine(Protocol):
    """Protocol for compute engines.

    Each engine provides implementations for operations.
    """

    def execute(self, op_name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute an operation by name."""
        ...
