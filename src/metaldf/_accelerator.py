"""Import interception layer for metaldf.

Installs a MetaPathFinder that intercepts `import pandas` and returns
a proxy module. Internal pandas/metaldf code gets real objects via a
caller-context denylist (checks sys._getframe().f_back.f_code.co_filename).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import Any

from metaldf._wrappers import ProxyDataFrame, ProxyIndex, ProxySeries

_METALDF_MARKER = "__metaldf_accelerator__"


class _ProxyPandasModule(types.ModuleType):
    """Custom module type that intercepts pandas attribute access.

    This replaces the module's __getattr__ to return proxy types
    for DataFrame, Series, and Index, and wraps callable return values.
    """

    def __init__(self, real_module: types.ModuleType) -> None:
        super().__init__(real_module.__name__)
        self._real_module = real_module

    def __getattribute__(self, name: str) -> Any:
        # Intercept attribute access at the __getattribute__ level
        # so that it works even for attributes in __dict__
        if name in ("_real_module", "__class__", "__dict__"):
            return object.__getattribute__(self, name)
        if name in ("DataFrame", "Series", "Index"):
            real_module = object.__getattribute__(self, "_real_module")
            attr = getattr(real_module, name)
            # Internal pandas code should see the real types so that
            # pandas internals work correctly (e.g., Index._simple_new).
            try:
                frame = sys._getframe().f_back
                if frame is not None:
                    filename = frame.f_code.co_filename
                    if "pandas" in filename:
                        return attr
            except (AttributeError, ValueError):
                pass
            if name == "DataFrame":
                return ProxyDataFrame
            if name == "Series":
                return ProxySeries
            if name == "Index":
                return ProxyIndex
        return object.__getattribute__(self, name)

    def __getattr__(self, name: str) -> Any:
        # Fallback for attributes not handled by __getattribute__
        real_module = object.__getattribute__(self, "_real_module")
        attr = getattr(real_module, name)

        # Internal pandas code should see the real types
        try:
            frame = sys._getframe().f_back
            if frame is not None:
                filename = frame.f_code.co_filename
                if "pandas" in filename:
                    return attr
        except (AttributeError, ValueError):
            pass

        # For functions, wrap their return values
        if callable(attr) and not isinstance(attr, type):
            return _wrap_callable(attr)

        return attr


def _wrap_callable(func: Any) -> Any:
    """Wrap a function so its return value is proxied if it's a pandas type."""
    from metaldf._wrappers import _wrap_result

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        return _wrap_result(result)

    return wrapper


class MetalAccelerator:
    """MetaPathFinder + Loader that intercepts `import pandas`."""

    _metaldf_marker = _METALDF_MARKER

    def __init__(self) -> None:
        self._real_pandas_module: types.ModuleType | None = None

    def find_module(self, fullname: str, path: Any = None) -> Any:
        return self.find_spec(fullname, path)

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:
        if fullname == "pandas":
            return importlib.util.spec_from_loader(fullname, self)  # type: ignore[arg-type]
        return None

    def create_module(self, spec: Any) -> Any:
        return None

    def exec_module(self, module: types.ModuleType) -> None:
        # Load the real pandas module
        if self._real_pandas_module is None:
            # Temporarily remove ourselves from meta_path to avoid recursion.
            # Also remove the cached pandas module so we can import the REAL
            # pandas (not the proxy that may already be in sys.modules).
            old_meta_path = sys.meta_path
            sys.meta_path = [f for f in sys.meta_path if f is not self]
            old_pandas = sys.modules.pop("pandas", None)
            try:
                import pandas as real_pandas

                self._real_pandas_module = real_pandas
            finally:
                sys.meta_path = old_meta_path
                # CRITICAL: Remove the real pandas from sys.modules so that
                # our proxy module (the one passed to exec_module) becomes
                # the module that user code gets when they import pandas.
                sys.modules.pop("pandas", None)
                # Restore cached pandas if it was present before
                if old_pandas is not None:
                    sys.modules["pandas"] = old_pandas

        # Set up proxy on the module: set _real_module and change __class__
        module.__dict__["_real_module"] = self._real_pandas_module
        module.__class__ = _ProxyPandasModule


def install() -> None:
    """Install the metaldf import interceptor.

    After calling install(), `import pandas` returns a proxy module
    whose DataFrame, Series, etc. are metaldf proxy types.
    """
    # Remove any existing metaldf accelerator
    uninstall()
    accelerator = MetalAccelerator()
    sys.meta_path.insert(0, accelerator)
    # `metaldf` itself imports pandas as part of its own initialization
    # (e.g. _wrappers.py does `import pandas as pd`), so the *real* pandas
    # module is already cached in sys.modules by the time install() runs.
    # Without dropping that cache entry, a subsequent `import pandas`
    # would short-circuit straight to the cached real module and never
    # reach our finder.
    sys.modules.pop("pandas", None)


def uninstall() -> None:
    """Remove the metaldf import interceptor."""
    sys.meta_path = [
        f for f in sys.meta_path if not hasattr(f, "_metaldf_marker")
    ]
    # Drop whatever pandas module (proxy or real) is currently cached so a
    # subsequent `import pandas` resolves cleanly against the real package
    # instead of keeping a stale proxy module bound under "pandas".
    sys.modules.pop("pandas", None)
