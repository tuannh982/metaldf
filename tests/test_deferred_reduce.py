import numpy as np
import pandas as pd
import pytest
from metaldf._wrappers import ProxySeries

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")


def test_deferred_sum_fused():
    """(pa + pb).sum() should use fused reduce kernel."""
    a = pd.Series(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    b = pd.Series(np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32))
    pa = ProxySeries(_pandas_obj=a)
    pb = ProxySeries(_pandas_obj=b)
    result = (pa + pb).sum()
    expected = (a + b).sum()
    assert abs(float(result) - float(expected)) < 0.01


def test_deferred_min_fused():
    a = pd.Series(np.array([10.0, 2.0, 8.0, 5.0], dtype=np.float32))
    b = pd.Series(np.array([3.0, 7.0, 1.0, 9.0], dtype=np.float32))
    pa = ProxySeries(_pandas_obj=a)
    pb = ProxySeries(_pandas_obj=b)
    result = (pa - pb).min()
    expected = (a - b).min()
    assert abs(float(result) - float(expected)) < 0.01


def test_deferred_max_fused():
    a = pd.Series(np.array([1.0, 3.0, 2.0, 4.0], dtype=np.float32))
    b = pd.Series(np.array([5.0, 2.0, 8.0, 1.0], dtype=np.float32))
    pa = ProxySeries(_pandas_obj=a)
    pb = ProxySeries(_pandas_obj=b)
    result = (pa * pb).max()
    expected = (a * b).max()
    assert abs(float(result) - float(expected)) < 0.01


def test_deferred_mean_fused():
    a = pd.Series(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    b = pd.Series(np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32))
    pa = ProxySeries(_pandas_obj=a)
    pb = ProxySeries(_pandas_obj=b)
    result = (pa + pb).mean()
    expected = (a + b).mean()
    assert abs(float(result) - float(expected)) < 0.1


def test_deferred_sum_chain_fused():
    """sum((a + b) * c - d) end to end via DeferredSeries."""
    a = pd.Series(np.random.default_rng(42).standard_normal(10000).astype(np.float32))
    b = pd.Series(np.random.default_rng(43).standard_normal(10000).astype(np.float32))
    c = pd.Series(np.random.default_rng(44).standard_normal(10000).astype(np.float32))
    d = pd.Series(np.random.default_rng(45).standard_normal(10000).astype(np.float32))
    pa, pb = ProxySeries(_pandas_obj=a), ProxySeries(_pandas_obj=b)
    pc, pd_ = ProxySeries(_pandas_obj=c), ProxySeries(_pandas_obj=d)
    result = ((pa + pb) * pc - pd_).sum()
    expected = ((a + b) * c - d).sum()
    assert abs(float(result) - float(expected)) / (abs(float(expected)) + 1e-6) < 0.01
