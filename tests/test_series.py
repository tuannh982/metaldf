import pandas as pd

from metaldf._wrappers import ProxySeries


def test_proxy_series_creation():
    real_s = pd.Series([1, 2, 3])
    proxy = ProxySeries(_pandas_obj=real_s)
    assert isinstance(proxy, pd.Series)


def test_proxy_series_sum():
    real_s = pd.Series([1, 2, 3])
    proxy = ProxySeries(_pandas_obj=real_s)
    assert proxy.sum() == 6


def test_proxy_series_mean():
    real_s = pd.Series([1, 2, 3, 4])
    proxy = ProxySeries(_pandas_obj=real_s)
    assert proxy.mean() == 2.5


def test_proxy_series_to_pandas():
    real_s = pd.Series([1, 2, 3])
    proxy = ProxySeries(_pandas_obj=real_s)
    assert proxy.to_pandas() is real_s


def test_proxy_series_repr():
    real_s = pd.Series([1, 2, 3])
    proxy = ProxySeries(_pandas_obj=real_s)
    r = repr(proxy)
    assert "ProxySeries" in r


def test_proxy_series_direct_construction():
    proxy = ProxySeries([1, 2, 3])
    assert isinstance(proxy, pd.Series)
    assert list(proxy) == [1, 2, 3]
