import numpy as np
import pandas as pd
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")


def test_batched_binary_ops():
    """Batch 3 binary ops into one command buffer submission."""
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    c = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    ma = metaldf_engine.MetalSeries.from_numpy(a)
    mb = metaldf_engine.MetalSeries.from_numpy(b)
    mc = metaldf_engine.MetalSeries.from_numpy(c)

    batch = metaldf_engine.begin_batch()
    # (a + b) * c: two batched ops
    tmp = metaldf_engine.metal_binary_op_batched("add", ma, mb, batch)
    result = metaldf_engine.metal_binary_op_batched("mul", tmp, mc, batch)
    metaldf_engine.batch_commit(batch)

    got = result.to_numpy()
    expected = (a + b) * c
    np.testing.assert_allclose(got, expected, rtol=1e-5)


def test_batched_matches_unbatched():
    """Batched and unbatched produce identical results."""
    a = np.random.default_rng(42).standard_normal(1000).astype(np.float32)
    b = np.random.default_rng(43).standard_normal(1000).astype(np.float32)
    ma = metaldf_engine.MetalSeries.from_numpy(a)
    mb = metaldf_engine.MetalSeries.from_numpy(b)

    unbatched = metaldf_engine.metal_binary_op("add", ma, mb)

    batch = metaldf_engine.begin_batch()
    batched = metaldf_engine.metal_binary_op_batched("add", ma, mb, batch)
    metaldf_engine.batch_commit(batch)

    np.testing.assert_array_equal(unbatched.to_numpy(), batched.to_numpy())
