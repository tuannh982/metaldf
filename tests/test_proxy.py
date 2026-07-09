import pandas as pd

from metaldf._proxy import make_final_proxy_type


def test_proxy_meta_isinstance_works():
    """isinstance(proxy_df, pd.DataFrame) must return True."""
    proxy_cls = make_final_proxy_type("ProxyDataFrame", pd.DataFrame)
    real_df = pd.DataFrame({"x": [1, 2, 3]})
    proxy = proxy_cls(_pandas_obj=real_df)
    assert isinstance(proxy, pd.DataFrame)


def test_proxy_delegates_getattr():
    """Attribute access on proxy delegates to wrapped pandas object."""
    proxy_cls = make_final_proxy_type("ProxyDataFrame", pd.DataFrame)
    real_df = pd.DataFrame({"x": [1, 2, 3]})
    proxy = proxy_cls(_pandas_obj=real_df)
    assert proxy.shape == (3, 1)


def test_proxy_to_pandas_returns_real_object():
    """to_pandas() unwraps to the real pandas object."""
    proxy_cls = make_final_proxy_type("ProxyDataFrame", pd.DataFrame)
    real_df = pd.DataFrame({"x": [1, 2, 3]})
    proxy = proxy_cls(_pandas_obj=real_df)
    assert proxy.to_pandas() is real_df


def test_proxy_meta_subclasscheck():
    """issubclass checks work for proxy types."""
    proxy_cls = make_final_proxy_type("ProxyDataFrame", pd.DataFrame)
    assert issubclass(proxy_cls, pd.DataFrame)


def test_proxy_repr():
    proxy_cls = make_final_proxy_type("ProxyDataFrame", pd.DataFrame)
    real_df = pd.DataFrame({"x": [1, 2, 3]})
    proxy = proxy_cls(_pandas_obj=real_df)
    r = repr(proxy)
    assert "ProxyDataFrame" in r
    assert "x" in r


def test_proxy_str():
    proxy_cls = make_final_proxy_type("ProxyDataFrame", pd.DataFrame)
    real_df = pd.DataFrame({"x": [1, 2, 3]})
    proxy = proxy_cls(_pandas_obj=real_df)
    s = str(proxy)
    assert "1" in s
    assert "x" in s
