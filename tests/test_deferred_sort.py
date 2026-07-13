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


def test_deferred_sort_values():
    """(pa + pb).sort_values() should materialize then sort."""
    a = pd.Series(np.array([3.0, 1.0, 4.0, 2.0], dtype=np.float32))
    b = pd.Series(np.array([1.0, 2.0, 0.0, 3.0], dtype=np.float32))
    pa = ProxySeries(_pandas_obj=a)
    pb = ProxySeries(_pandas_obj=b)
    result = (pa + pb).sort_values()
    expected_vals = np.sort((a + b).to_numpy())
    result_vals = result.to_numpy() if hasattr(result, 'to_numpy') else np.asarray(result)
    np.testing.assert_allclose(result_vals, expected_vals, rtol=1e-5)


def test_deferred_sort_descending():
    a = pd.Series(np.array([3.0, 1.0, 4.0, 2.0], dtype=np.float32))
    b = pd.Series(np.array([1.0, 2.0, 0.0, 3.0], dtype=np.float32))
    pa = ProxySeries(_pandas_obj=a)
    pb = ProxySeries(_pandas_obj=b)
    result = (pa + pb).sort_values(ascending=False)
    expected_vals = np.sort((a + b).to_numpy())[::-1]
    result_vals = result.to_numpy() if hasattr(result, 'to_numpy') else np.asarray(result)
    np.testing.assert_allclose(result_vals, expected_vals, rtol=1e-5)
