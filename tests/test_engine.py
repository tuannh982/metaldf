import pytest

from metaldf._engine import clear_registry, execute, register
from metaldf._engine._pandas import PandasEngine


def test_execute_falls_back_to_pandas_engine():
    clear_registry()
    result = execute("sum", [1, 2, 3])
    # PandasEngine.sum should delegate to numpy/pandas sum
    assert result == 6


def test_register_and_execute():
    clear_registry()
    register("double", lambda x: x * 2)
    result = execute("double", 5)
    assert result == 10


def test_clear_registry_removes_ops():
    clear_registry()
    register("foo_op", lambda x: x * 2)
    clear_registry()
    # After clear, should fall back to PandasEngine which doesn't know "foo_op"
    with pytest.raises(KeyError):
        execute("foo_op", 5)


def test_pandas_engine_execute_known_op():
    result = PandasEngine.execute("sum", [1, 2, 3])
    assert result == 6


def test_pandas_engine_execute_unknown_op_raises():
    with pytest.raises(KeyError):
        PandasEngine.execute("nonexistent_op", [1, 2, 3])


def test_pandas_engine_mean():

    result = PandasEngine.execute("mean", [1, 2, 3])
    assert result == pytest.approx(2.0)


def test_pandas_engine_min_max():
    result = PandasEngine.execute("min", [3, 1, 2])
    assert result == 1
    result = PandasEngine.execute("max", [3, 1, 2])
    assert result == 3


def test_pandas_engine_sum_with_series():
    import pandas as pd

    s = pd.Series([1, 2, 3])
    result = PandasEngine.execute("sum", s)
    assert result == 6


def test_pandas_engine_mean_with_series():
    import pandas as pd

    s = pd.Series([1, 2, 3, 4])
    result = PandasEngine.execute("mean", s)
    assert result == pytest.approx(2.5)


def test_execute_with_kwargs():
    clear_registry()
    register("add", lambda a, b: a + b)
    result = execute("add", a=3, b=4)
    assert result == 7


def test_execute_with_args_and_kwargs():
    clear_registry()
    register("greet", lambda name, greeting="Hello": f"{greeting} {name}")
    result = execute("greet", "World", greeting="Hi")
    assert result == "Hi World"
