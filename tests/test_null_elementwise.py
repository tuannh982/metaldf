import numpy as np
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")


def test_binary_add_with_nulls():
    a = np.array([1.0, float('nan'), 3.0, 4.0], dtype=np.float32)
    b = np.array([10.0, 20.0, float('nan'), 40.0], dtype=np.float32)
    ma = metaldf_engine.MetalSeries.from_numpy_with_nulls(a)
    mb = metaldf_engine.MetalSeries.from_numpy_with_nulls(b)
    result = metaldf_engine.metal_binary_op("add", ma, mb)
    mask = result.null_mask
    assert mask is not None
    np.testing.assert_array_equal(mask, [True, False, False, True])
    data = result.to_numpy()
    np.testing.assert_allclose(data[0], 11.0)
    np.testing.assert_allclose(data[3], 44.0)


def test_binary_no_nulls_fast_path():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    ma = metaldf_engine.MetalSeries.from_numpy(a)
    mb = metaldf_engine.MetalSeries.from_numpy(b)
    result = metaldf_engine.metal_binary_op("add", ma, mb)
    assert result.null_mask is None
    np.testing.assert_allclose(result.to_numpy(), [5.0, 7.0, 9.0])


def test_unary_abs_with_nulls():
    a = np.array([-1.0, float('nan'), -3.0], dtype=np.float32)
    ma = metaldf_engine.MetalSeries.from_numpy_with_nulls(a)
    result = metaldf_engine.metal_unary_op("abs", ma)
    mask = result.null_mask
    np.testing.assert_array_equal(mask, [True, False, True])
    np.testing.assert_allclose(result.to_numpy()[0], 1.0)
    np.testing.assert_allclose(result.to_numpy()[2], 3.0)


def test_unary_no_nulls_fast_path():
    a = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    ma = metaldf_engine.MetalSeries.from_numpy(a)
    result = metaldf_engine.metal_unary_op("abs", ma)
    assert result.null_mask is None
    np.testing.assert_allclose(result.to_numpy(), [1.0, 2.0, 3.0])


def test_binary_mixed_one_side_has_mask():
    """Only one operand carries a null mask -- the other must behave as
    'always valid' (nullptr mask binding), not silently null out its side."""
    a = np.array([1.0, float('nan'), 3.0, 4.0], dtype=np.float32)
    b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    ma = metaldf_engine.MetalSeries.from_numpy_with_nulls(a)
    mb = metaldf_engine.MetalSeries.from_numpy(b)  # no mask at all
    result = metaldf_engine.metal_binary_op("add", ma, mb)
    mask = result.null_mask
    assert mask is not None
    np.testing.assert_array_equal(mask, [True, False, True, True])
    data = result.to_numpy()
    np.testing.assert_allclose(data[0], 11.0)
    np.testing.assert_allclose(data[2], 33.0)
    np.testing.assert_allclose(data[3], 44.0)

    # Same check with mask on the right-hand side instead.
    result2 = metaldf_engine.metal_binary_op("add", mb, ma)
    mask2 = result2.null_mask
    assert mask2 is not None
    np.testing.assert_array_equal(mask2, [True, False, True, True])


def test_binary_sub_mul_div_with_nulls():
    a = np.array([10.0, float('nan'), 30.0], dtype=np.float32)
    b = np.array([1.0, 2.0, float('nan')], dtype=np.float32)
    ma = metaldf_engine.MetalSeries.from_numpy_with_nulls(a)
    mb = metaldf_engine.MetalSeries.from_numpy_with_nulls(b)
    for op, expected_fn in [
        ("sub", lambda x, y: x - y),
        ("mul", lambda x, y: x * y),
        ("div", lambda x, y: x / y),
    ]:
        result = metaldf_engine.metal_binary_op(op, ma, mb)
        mask = result.null_mask
        np.testing.assert_array_equal(mask, [True, False, False])
        np.testing.assert_allclose(result.to_numpy()[0], expected_fn(10.0, 1.0))


def test_binary_i32_with_nulls():
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    b = np.array([10, 20, 30, 40], dtype=np.int32)
    ma = metaldf_engine.MetalSeries.from_numpy_i32(a)
    mb = metaldf_engine.MetalSeries.from_numpy_i32(b)
    # int series have no NaN-based null constructor; both plain -> fast path.
    result = metaldf_engine.metal_binary_op("add", ma, mb)
    assert result.null_mask is None
    np.testing.assert_array_equal(result.to_numpy(), [11, 22, 33, 44])


def test_unary_neg_sqrt_with_nulls():
    a = np.array([4.0, float('nan'), 9.0], dtype=np.float32)
    ma = metaldf_engine.MetalSeries.from_numpy_with_nulls(a)

    neg_result = metaldf_engine.metal_unary_op("neg", ma)
    np.testing.assert_array_equal(neg_result.null_mask, [True, False, True])
    np.testing.assert_allclose(neg_result.to_numpy()[0], -4.0)
    np.testing.assert_allclose(neg_result.to_numpy()[2], -9.0)

    sqrt_result = metaldf_engine.metal_unary_op("sqrt", ma)
    np.testing.assert_array_equal(sqrt_result.null_mask, [True, False, True])
    np.testing.assert_allclose(sqrt_result.to_numpy()[0], 2.0)
    np.testing.assert_allclose(sqrt_result.to_numpy()[2], 3.0)


def test_null_elementwise_large_array_not_multiple_of_threadgroup():
    n = 1000  # not a multiple of 256
    a = (np.arange(n, dtype=np.float32) - 500.0)
    a[3] = float('nan')
    a[999] = float('nan')
    ma = metaldf_engine.MetalSeries.from_numpy_with_nulls(a)
    result = metaldf_engine.metal_unary_op("abs", ma)
    mask = result.null_mask
    assert mask is not None
    assert not mask[3]
    assert not mask[999]
    assert mask.sum() == n - 2
    got = result.to_numpy()
    expected = np.abs(a)
    valid = mask
    np.testing.assert_allclose(got[valid], expected[valid], rtol=1e-5)
