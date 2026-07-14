import numpy as np
import pandas as pd
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")


# --- Fix 1: Empty-array crash ---

def test_from_numpy_empty_f32():
    arr = np.array([], dtype=np.float32)
    ms = metaldf_engine.MetalSeries.from_numpy(arr)
    assert ms.len == 0


def test_from_numpy_empty_i32():
    arr = np.array([], dtype=np.int32)
    ms = metaldf_engine.MetalSeries.from_numpy_i32(arr)
    assert ms.len == 0


def test_from_numpy_empty_i64():
    arr = np.array([], dtype=np.int64)
    ms = metaldf_engine.MetalSeries.from_numpy_i64(arr)
    assert ms.len == 0


# --- Fix 2: ProxySeries unary methods ---

from metaldf._wrappers import ProxySeries
from metaldf._deferred import DeferredSeries


@pytest.mark.parametrize("method,np_fn", [
    ("sin", np.sin), ("cos", np.cos), ("tan", np.tan),
    ("sinh", np.sinh), ("cosh", np.cosh), ("tanh", np.tanh),
    ("log2", np.log2), ("log10", np.log10),
    ("trunc", np.trunc), ("cbrt", np.cbrt),
])
def test_proxy_unary_method(method, np_fn):
    data = np.array([0.1, 0.5, 1.0, 2.0], dtype=np.float32)
    if method in ("log2", "log10"):
        data = np.array([0.5, 1.0, 2.0, 10.0], dtype=np.float32)
    s = ProxySeries(_pandas_obj=pd.Series(data))
    result = getattr(s, method)()
    if isinstance(result, DeferredSeries):
        result = result.to_pandas()
    expected = np_fn(data)
    np.testing.assert_allclose(result.values, expected, rtol=1e-4)


def test_numpy_ufunc_sin():
    s = ProxySeries(_pandas_obj=pd.Series([0.1, 0.5, 1.0], dtype=np.float32))
    result = np.sin(s)
    if isinstance(result, DeferredSeries):
        result = result.to_pandas()
    expected = np.sin(np.array([0.1, 0.5, 1.0], dtype=np.float32))
    np.testing.assert_allclose(result.values, expected, rtol=1e-4)


def test_numpy_ufunc_isnan_still_works():
    s = ProxySeries(_pandas_obj=pd.Series([1.0, float("nan"), 3.0], dtype=np.float32))
    result = np.isnan(s)
    assert result[1] == True


def test_deferred_fusion_via_method():
    s = ProxySeries(_pandas_obj=pd.Series([0.1, 0.5, 0.9], dtype=np.float32))
    result = (s + 0.1).sin()
    assert isinstance(result, DeferredSeries)


# --- Fix 3: Fused-reduce null awareness ---

def test_fused_reduce_with_nan_matches_pandas():
    a = pd.Series([1.0, float("nan"), 3.0, 4.0], dtype=np.float32)
    b = pd.Series([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    pa = ProxySeries(_pandas_obj=a)
    pb = ProxySeries(_pandas_obj=b)
    result = (pa + pb).sum()
    expected = (a + b).sum()
    np.testing.assert_allclose(result, expected, rtol=1e-4)


def test_fused_reduce_no_nan_still_fuses():
    a = pd.Series([1.0, 2.0, 3.0], dtype=np.float32)
    b = pd.Series([4.0, 5.0, 6.0], dtype=np.float32)
    pa = ProxySeries(_pandas_obj=a)
    pb = ProxySeries(_pandas_obj=b)
    result = (pa + pb).sum()
    expected = (a + b).sum()
    np.testing.assert_allclose(result, expected, rtol=1e-4)


def test_fused_reduce_mean_with_nan():
    a = pd.Series([1.0, float("nan"), 3.0], dtype=np.float32)
    b = pd.Series([10.0, 20.0, 30.0], dtype=np.float32)
    pa = ProxySeries(_pandas_obj=a)
    pb = ProxySeries(_pandas_obj=b)
    result = (pa + pb).mean()
    expected = (a + b).mean()
    np.testing.assert_allclose(result, expected, rtol=1e-4)


# --- Fix 4: Left/right join ---

from metaldf._wrappers import ProxyDataFrame


def test_left_join_matches_pandas():
    left = pd.DataFrame({"key": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                          "val_l": np.array([10.0, 20.0, 30.0], dtype=np.float32)})
    right = pd.DataFrame({"key": np.array([2.0, 3.0, 4.0], dtype=np.float32),
                           "val_r": np.array([200.0, 300.0, 400.0], dtype=np.float32)})
    proxy_left = ProxyDataFrame(_pandas_obj=left)
    result = proxy_left.merge(right, on="key", how="left")
    expected = pd.merge(left, right, on="key", how="left")
    pd.testing.assert_frame_equal(
        result.to_pandas().sort_values("key").reset_index(drop=True),
        expected.sort_values("key").reset_index(drop=True),
        check_dtype=False,
    )


def test_right_join_matches_pandas():
    left = pd.DataFrame({"key": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                          "val_l": np.array([10.0, 20.0, 30.0], dtype=np.float32)})
    right = pd.DataFrame({"key": np.array([2.0, 3.0, 4.0], dtype=np.float32),
                           "val_r": np.array([200.0, 300.0, 400.0], dtype=np.float32)})
    proxy_left = ProxyDataFrame(_pandas_obj=left)
    result = proxy_left.merge(right, on="key", how="right")
    expected = pd.merge(left, right, on="key", how="right")
    pd.testing.assert_frame_equal(
        result.to_pandas().sort_values("key").reset_index(drop=True),
        expected.sort_values("key").reset_index(drop=True),
        check_dtype=False,
    )


def test_left_join_all_matched():
    left = pd.DataFrame({"key": np.array([1.0, 2.0], dtype=np.float32),
                          "v": np.array([10.0, 20.0], dtype=np.float32)})
    right = pd.DataFrame({"key": np.array([1.0, 2.0], dtype=np.float32),
                           "w": np.array([100.0, 200.0], dtype=np.float32)})
    proxy = ProxyDataFrame(_pandas_obj=left)
    result = proxy.merge(right, on="key", how="left")
    assert len(result.to_pandas()) == 2
    assert not result.to_pandas()["w"].isna().any()


def test_left_join_no_matches():
    left = pd.DataFrame({"key": np.array([1.0, 2.0], dtype=np.float32),
                          "v": np.array([10.0, 20.0], dtype=np.float32)})
    right = pd.DataFrame({"key": np.array([3.0, 4.0], dtype=np.float32),
                           "w": np.array([100.0, 200.0], dtype=np.float32)})
    proxy = ProxyDataFrame(_pandas_obj=left)
    result = proxy.merge(right, on="key", how="left")
    assert len(result.to_pandas()) == 2
    assert result.to_pandas()["w"].isna().all()
