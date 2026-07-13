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


def test_deferred_add_materializes():
    a = pd.Series(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = pd.Series(np.array([4.0, 5.0, 6.0], dtype=np.float32))
    pa = ProxySeries(_pandas_obj=a)
    pb = ProxySeries(_pandas_obj=b)
    result = pa + pb
    # Result should be a DeferredSeries
    from metaldf._deferred import DeferredSeries
    assert isinstance(result, DeferredSeries)
    # Materializing should give correct values
    materialized = result.to_pandas()
    expected = a + b
    pd.testing.assert_series_equal(materialized, expected, check_dtype=False, check_names=False)


def test_chained_deferred():
    a = pd.Series(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    b = pd.Series(np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32))
    c = pd.Series(np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32))
    pa = ProxySeries(_pandas_obj=a)
    pb = ProxySeries(_pandas_obj=b)
    pc = ProxySeries(_pandas_obj=c)
    result = (pa + pb) * pc  # Should build a 2-deep expression tree
    from metaldf._deferred import DeferredSeries
    assert isinstance(result, DeferredSeries)
    materialized = result.to_pandas()
    expected = (a + b) * c
    np.testing.assert_allclose(materialized.to_numpy(), expected.to_numpy(), rtol=1e-5)


def test_deferred_triggers_on_sum():
    """Reduction on a deferred series should materialize then reduce."""
    a = pd.Series(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = pd.Series(np.array([4.0, 5.0, 6.0], dtype=np.float32))
    pa = ProxySeries(_pandas_obj=a)
    pb = ProxySeries(_pandas_obj=b)
    result = (pa + pb).sum()
    expected = (a + b).sum()
    assert abs(float(result) - float(expected)) < 0.01


def test_deferred_triggers_on_print():
    """str() on a deferred series should materialize."""
    a = pd.Series(np.array([1.0, 2.0], dtype=np.float32))
    b = pd.Series(np.array([3.0, 4.0], dtype=np.float32))
    pa = ProxySeries(_pandas_obj=a)
    pb = ProxySeries(_pandas_obj=b)
    result = pa + pb
    s = str(result)  # Should not crash
    assert "4.0" in s or "4" in s


def test_deferred_assign_to_df():
    """Assigning a deferred series to a DataFrame column should materialize."""
    from metaldf._wrappers import ProxyDataFrame
    df = pd.DataFrame({
        "revenue": np.array([100.0, 200.0, 300.0], dtype=np.float32),
        "cost": np.array([30.0, 60.0, 90.0], dtype=np.float32),
    })
    pdf = ProxyDataFrame(_pandas_obj=df)
    rev = ProxySeries(_pandas_obj=df["revenue"])
    cost = ProxySeries(_pandas_obj=df["cost"])
    profit = rev - cost  # DeferredSeries
    pdf["profit"] = profit  # Should materialize
    expected = df["revenue"] - df["cost"]
    pd.testing.assert_series_equal(
        pdf.to_pandas()["profit"], expected, check_dtype=False, check_names=False,
    )


def test_deferred_with_plain_series_operand():
    """Mixing ProxySeries + plain pd.Series should not crash."""
    a = pd.Series(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    pa = ProxySeries(_pandas_obj=a)
    b = pd.Series(np.array([4.0, 5.0, 6.0], dtype=np.float32))  # plain, not wrapped
    result = pa + b  # should not crash
    expected = a + b
    # Result may be DeferredSeries or ProxySeries depending on fallback
    if hasattr(result, 'to_pandas'):
        result = result.to_pandas()
    pd.testing.assert_series_equal(result, expected, check_dtype=False, check_names=False)
