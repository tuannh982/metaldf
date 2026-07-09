"""PandasEngine — fallback engine.

Delegates all operations to numpy/pandas. Used whenever an operation has
no registered Metal implementation, or the Metal implementation raises
``MetalNotAvailable``.
"""

from __future__ import annotations

from typing import Any

import numpy as np



# Ops whose name happens to collide with a numpy function of different
# semantics: np.sort/np.argsort take a Series and hand back a bare ndarray,
# dropping the index/name -- silently "succeeding" with the wrong return
# type instead of signalling that PandasEngine can't handle this op.
# Callers of these ops (e.g. ProxySeries.sort_values/argsort) already
# implement their own correct pandas fallback (pd.Series.sort_values/
# argsort) and rely on getting an exception here so they know to use it.
_NO_NUMPY_FALLBACK = {"sort", "argsort"}


class PandasEngine:
    """Fallback engine that uses numpy/pandas for everything."""

    @classmethod
    def execute(cls, op_name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute an operation using numpy.

        This is a thin wrapper around numpy operations. In practice, most
        operations are handled by proxy __getattr__ delegating directly to
        the pandas object, so this registry is mainly used as the fallback
        for explicit Metal vs pandas dispatch (see ``metaldf._engine``).
        """
        if op_name not in _NO_NUMPY_FALLBACK:
            np_func = getattr(np, op_name, None)
            if np_func is not None:
                return np_func(*args, **kwargs)
        raise KeyError(f"Operation '{op_name}' not available in PandasEngine")
