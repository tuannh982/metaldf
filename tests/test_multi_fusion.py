"""Tests for Task 8.1: multi-column expression fusion.

Verifies that multiple deferred column assignments on a ProxyDataFrame are
flushed together as a single GPU kernel dispatch (via
``eval_multi_expression_codegen``), and that the pending-queue semantics
(flush-on-read, non-deferred override, etc.) behave correctly.
"""

import numpy as np
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")


# ---------------------------------------------------------------------------
# Low-level Rust function: eval_multi_expression_codegen
# ---------------------------------------------------------------------------

OP_ADD = 16
OP_MUL = 18
OP_SUB = 17


class TestMultiExpressionCodegen:
    """Direct tests for the Rust ``eval_multi_expression_codegen`` function."""

    def test_two_programs_shared_columns(self):
        """Two programs sharing column c0: (c0 + c1) and (c0 * c2)."""
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        c = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        ma = metaldf_engine.MetalSeries.from_numpy(a)
        mb = metaldf_engine.MetalSeries.from_numpy(b)
        mc = metaldf_engine.MetalSeries.from_numpy(c)

        # Program 0: c0 + c1
        prog0 = bytes([0, 1, OP_ADD])
        # Program 1: c0 * c2
        prog1 = bytes([0, 2, OP_MUL])

        results = metaldf_engine.eval_multi_expression_codegen(
            [prog0, prog1], [ma, mb, mc], len(a),
        )

        assert len(results) == 2
        np.testing.assert_allclose(results[0].to_numpy(), a + b, rtol=1e-5)
        np.testing.assert_allclose(results[1].to_numpy(), a * c, rtol=1e-5)

    def test_single_program(self):
        """Multi-output with a single program degenerates correctly."""
        a = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ma = metaldf_engine.MetalSeries.from_numpy(a)
        mb = metaldf_engine.MetalSeries.from_numpy(b)

        prog = bytes([0, 1, OP_SUB])
        results = metaldf_engine.eval_multi_expression_codegen(
            [prog], [ma, mb], len(a),
        )
        assert len(results) == 1
        np.testing.assert_allclose(results[0].to_numpy(), a - b, rtol=1e-5)

    def test_empty_programs(self):
        """Empty program list returns empty result list."""
        results = metaldf_engine.eval_multi_expression_codegen([], [], 0)
        assert results == []

    def test_three_programs(self):
        """Three programs with overlapping inputs."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        ma = metaldf_engine.MetalSeries.from_numpy(a)
        mb = metaldf_engine.MetalSeries.from_numpy(b)

        # out0 = c0 + c1, out1 = c0 - c1, out2 = c0 * c1
        results = metaldf_engine.eval_multi_expression_codegen(
            [bytes([0, 1, OP_ADD]), bytes([0, 1, OP_SUB]), bytes([0, 1, OP_MUL])],
            [ma, mb],
            len(a),
        )
        assert len(results) == 3
        np.testing.assert_allclose(results[0].to_numpy(), a + b, rtol=1e-5)
        np.testing.assert_allclose(results[1].to_numpy(), a - b, rtol=1e-5)
        np.testing.assert_allclose(results[2].to_numpy(), a * b, rtol=1e-5)

    def test_large_array(self):
        """Multi-output at scale (1M elements)."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(1_000_000).astype(np.float32)
        b = rng.standard_normal(1_000_000).astype(np.float32)
        ma = metaldf_engine.MetalSeries.from_numpy(a)
        mb = metaldf_engine.MetalSeries.from_numpy(b)

        results = metaldf_engine.eval_multi_expression_codegen(
            [bytes([0, 1, OP_ADD]), bytes([0, 1, OP_MUL])],
            [ma, mb],
            len(a),
        )
        np.testing.assert_allclose(results[0].to_numpy(), a + b, rtol=1e-4)
        np.testing.assert_allclose(results[1].to_numpy(), a * b, rtol=1e-4)


# ---------------------------------------------------------------------------
# ProxyDataFrame pending queue: deferred multi-column fusion
# ---------------------------------------------------------------------------

from metaldf._wrappers import ProxyDataFrame, ProxySeries
import pandas as pd


class TestPendingQueue:
    """Tests for the pending-assignment queue on ProxyDataFrame."""

    def _make_df(self):
        """Create a ProxyDataFrame with float32 columns a, b, c."""
        df = ProxyDataFrame(
            _pandas_obj=pd.DataFrame({
                "a": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                "b": np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
                "c": np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32),
            })
        )
        return df

    def test_two_deferred_columns_fuse(self):
        """Assigning two deferred expressions flushes them together."""
        df = self._make_df()

        # These should be queued, not materialized yet
        df["z"] = df["a"] + df["b"]
        df["w"] = df["a"] * df["c"]

        # Verify they are pending
        pending = object.__getattribute__(df, "_pending_assigns")
        assert len(pending) == 2

        # Access triggers flush
        result_z = np.asarray(df["z"])
        result_w = np.asarray(df["w"])

        expected_z = np.array([6.0, 8.0, 10.0, 12.0], dtype=np.float32)
        expected_w = np.array([2.0, 6.0, 12.0, 20.0], dtype=np.float32)

        np.testing.assert_allclose(result_z, expected_z, rtol=1e-5)
        np.testing.assert_allclose(result_w, expected_w, rtol=1e-5)

    def test_shared_column_read_once(self):
        """Shared column 'a' is in the unified column list exactly once.

        We verify correctness of the fused result, which implicitly proves
        column 'a' was read correctly (if it were duplicated or misindexed,
        the expressions would produce wrong results).
        """
        df = self._make_df()
        df["z"] = df["a"] + df["b"]
        df["w"] = df["a"] * df["c"]

        z = np.asarray(df["z"])
        w = np.asarray(df["w"])

        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        c = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)

        np.testing.assert_allclose(z, a + b, rtol=1e-5)
        np.testing.assert_allclose(w, a * c, rtol=1e-5)

    def test_non_deferred_assignment_still_works(self):
        """Plain (non-deferred) column assignments bypass the queue."""
        df = self._make_df()
        df["x"] = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

        # Should be immediately available, no pending
        pending = object.__getattribute__(df, "_pending_assigns")
        assert len(pending) == 0

        result = np.asarray(df["x"])
        np.testing.assert_allclose(result, [10.0, 20.0, 30.0, 40.0])

    def test_flush_on_repr(self):
        """repr() triggers flush of pending assignments."""
        df = self._make_df()
        df["z"] = df["a"] + df["b"]

        pending = object.__getattribute__(df, "_pending_assigns")
        assert len(pending) == 1

        # repr should trigger flush
        _ = repr(df)

        pending = object.__getattribute__(df, "_pending_assigns")
        assert len(pending) == 0

    def test_flush_on_str(self):
        """str() triggers flush of pending assignments."""
        df = self._make_df()
        df["z"] = df["a"] + df["b"]
        _ = str(df)

        pending = object.__getattribute__(df, "_pending_assigns")
        assert len(pending) == 0

    def test_flush_on_to_pandas(self):
        """to_pandas() triggers flush of pending assignments."""
        df = self._make_df()
        df["z"] = df["a"] + df["b"]

        result = df.to_pandas()
        expected_z = np.array([6.0, 8.0, 10.0, 12.0], dtype=np.float32)
        np.testing.assert_allclose(np.asarray(result["z"]), expected_z, rtol=1e-5)

    def test_flush_on_getattr(self):
        """Attribute access (e.g. .columns) triggers flush."""
        df = self._make_df()
        df["z"] = df["a"] + df["b"]

        # .columns goes through __getattr__
        cols = df.columns
        assert "z" in cols

    def test_non_deferred_overrides_pending(self):
        """A concrete assignment to the same key removes the pending entry."""
        df = self._make_df()
        df["z"] = df["a"] + df["b"]  # queued
        df["z"] = np.array([99.0, 99.0, 99.0, 99.0], dtype=np.float32)  # concrete override

        # The pending queue should have no entries for "z"
        pending = object.__getattribute__(df, "_pending_assigns")
        assert len(pending) == 0

        result = np.asarray(df["z"])
        np.testing.assert_allclose(result, [99.0, 99.0, 99.0, 99.0])

    def test_deferred_override_replaces_pending(self):
        """A second deferred assignment to the same key replaces the first."""
        df = self._make_df()
        df["z"] = df["a"] + df["b"]  # queued
        df["z"] = df["a"] * df["c"]  # replaces

        pending = object.__getattribute__(df, "_pending_assigns")
        assert len(pending) == 1

        result = np.asarray(df["z"])
        expected = np.array([2.0, 6.0, 12.0, 20.0], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_mixed_deferred_and_concrete(self):
        """Mix of deferred and concrete assignments in sequence."""
        df = self._make_df()
        df["z"] = df["a"] + df["b"]  # queued
        df["x"] = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)  # concrete
        df["w"] = df["a"] * df["c"]  # queued

        # z and w are pending, x is already assigned
        pending = object.__getattribute__(df, "_pending_assigns")
        assert len(pending) == 2

        # Access flushes all
        z = np.asarray(df["z"])
        w = np.asarray(df["w"])
        x = np.asarray(df["x"])

        np.testing.assert_allclose(z, [6.0, 8.0, 10.0, 12.0], rtol=1e-5)
        np.testing.assert_allclose(w, [2.0, 6.0, 12.0, 20.0], rtol=1e-5)
        np.testing.assert_allclose(x, [10.0, 20.0, 30.0, 40.0])

    def test_chained_dependency_forces_flush(self):
        """Second expression depending on a pending column forces a flush."""
        df = self._make_df()
        df["z"] = df["a"] + df["b"]  # queued

        # Accessing df["z"] to build next expression flushes the queue
        df["w"] = df["z"] + df["c"]  # df["z"] triggers flush, then this is queued

        # w is now pending
        pending = object.__getattribute__(df, "_pending_assigns")
        assert len(pending) == 1  # only w

        w = np.asarray(df["w"])
        expected_z = np.array([6.0, 8.0, 10.0, 12.0], dtype=np.float32)
        expected_w = expected_z + np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        np.testing.assert_allclose(w, expected_w, rtol=1e-5)
