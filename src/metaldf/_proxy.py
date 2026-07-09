"""Base proxy type system for metaldf.

Provides metaclass machinery to make isinstance checks work,
and a base class for "final" proxy types (DataFrame, Series, Index)
that wrap a real pandas object. Metal dispatch is handled separately
by the engine registry (see ``metaldf._engine``).
"""

from __future__ import annotations

from typing import Any


class _ProxyMeta(type):
    """Metaclass for proxy types.

    Makes isinstance(proxy, pd.DataFrame) return True by
    overriding __instancecheck__ to also check the proxied type.
    """

    def __instancecheck__(cls, instance: object) -> bool:
        # First check the proxy type itself
        if super().__instancecheck__(instance):
            return True
        # Then check the underlying pandas type
        pandas_type = getattr(cls, "_pandas_type", None)
        return pandas_type is not None and isinstance(instance, pandas_type)

    def __subclasscheck__(cls, subclass: type) -> bool:
        pandas_type = getattr(cls, "_pandas_type", None)
        if pandas_type is not None and issubclass(subclass, pandas_type):
            return True
        return super().__subclasscheck__(subclass)


class _FinalProxy(metaclass=_ProxyMeta):
    """Base class for final proxy types.

    Holds:
      - _pandas_obj: the real pandas object

    Metal dispatch (when available) goes through the engine registry
    (see ``metaldf._engine``), not through this class.

    __getattr__ delegates to _pandas_obj.
    to_pandas() returns the real pandas object (escape hatch).
    """

    _pandas_type: type
    _pandas_obj: Any | None = None

    def __init__(
        self,
        *args: Any,
        _pandas_obj: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self._pandas_obj = _pandas_obj

    def __getattr__(self, name: str) -> Any:
        if self._pandas_obj is None:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self._pandas_obj, name)

    def to_pandas(self) -> Any:
        """Unwrap to the real pandas object."""
        return self._pandas_obj

    def __repr__(self) -> str:
        if self._pandas_obj is not None:
            return f"{type(self).__name__}(\n{repr(self._pandas_obj)}\n)"
        return f"{type(self).__name__}(_pandas_obj=None)"

    def __str__(self) -> str:
        if self._pandas_obj is not None:
            return str(self._pandas_obj)
        return repr(self)


def make_final_proxy_type(name: str, pandas_type: type) -> type:
    """Create a concrete final proxy type for a given pandas type.

    Returns a new class that inherits from _FinalProxy and has the
    correct _pandas_type set up for isinstance checks.
    """
    def _new(
        cls,
        *args: Any,
        _pandas_obj: Any | None = None,
        **kwargs: Any,
    ):
        if _pandas_obj is not None:
            # Wrapping an existing pandas object
            instance = object.__new__(cls)
            object.__setattr__(instance, "_pandas_obj", _pandas_obj)
            return instance
        # Direct construction: create real object and wrap it
        real_obj = pandas_type(*args, **kwargs)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_pandas_obj", real_obj)
        return instance

    def _init(
        self,
        *args: Any,
        _pandas_obj: Any | None = None,
        **kwargs: Any,
    ) -> None:
        # Skip pandas_type.__init__ — we're wrapping an existing object
        pass

    def _getattr(self: Any, name: str) -> Any:
        if name == "_pandas_obj":
            raise AttributeError(name)
        obj = object.__getattribute__(self, "_pandas_obj")
        return getattr(obj, name)

    def _to_pandas(self: Any) -> Any:
        return object.__getattribute__(self, "_pandas_obj")

    return _ProxyMeta(
        name,
        (_FinalProxy, pandas_type),
        {
            "_pandas_type": pandas_type,
            "__new__": _new,
            "__init__": _init,
            "__getattr__": _getattr,
            "to_pandas": _to_pandas,
        },
    )
