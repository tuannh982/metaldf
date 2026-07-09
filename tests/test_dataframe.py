import pandas as pd

from metaldf._wrappers import ProxyDataFrame


def test_proxy_dataframe_creation():
    real_df = pd.DataFrame({"x": [1, 2, 3]})
    proxy = ProxyDataFrame(_pandas_obj=real_df)
    assert isinstance(proxy, pd.DataFrame)


def test_proxy_dataframe_column_access():
    real_df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    proxy = ProxyDataFrame(_pandas_obj=real_df)
    col = proxy["x"]
    assert isinstance(col, pd.Series)
    assert list(col) == [1, 2, 3]


def test_proxy_dataframe_arithmetic():
    real_df = pd.DataFrame({"x": [1, 2, 3]})
    proxy = ProxyDataFrame(_pandas_obj=real_df)
    result = proxy + 1
    # result should delegate to pandas; we check the value
    assert list(result["x"]) == [2, 3, 4]


def test_proxy_dataframe_shape():
    real_df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    proxy = ProxyDataFrame(_pandas_obj=real_df)
    assert proxy.shape == (3, 2)


def test_proxy_dataframe_to_pandas():
    real_df = pd.DataFrame({"x": [1, 2, 3]})
    proxy = ProxyDataFrame(_pandas_obj=real_df)
    assert proxy.to_pandas() is real_df


def test_proxy_dataframe_repr():
    real_df = pd.DataFrame({"x": [1, 2, 3]})
    proxy = ProxyDataFrame(_pandas_obj=real_df)
    r = repr(proxy)
    assert "ProxyDataFrame" in r


def test_proxy_dataframe_groupby():
    real_df = pd.DataFrame({"key": ["a", "a", "b"], "val": [1, 2, 3]})
    proxy = ProxyDataFrame(_pandas_obj=real_df)
    result = proxy.groupby("key")["val"].sum()
    assert result["a"] == 3
    assert result["b"] == 3


def test_proxy_dataframe_direct_construction():
    proxy = ProxyDataFrame({"x": [1, 2, 3]})
    assert isinstance(proxy, pd.DataFrame)
    assert list(proxy["x"]) == [1, 2, 3]
