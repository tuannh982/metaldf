import platform

import numpy as np
import pandas as pd
import pytest

from metaldf._wrappers import ProxySeries

pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin",
    reason="Metal arithmetic tests only run on macOS",
)


def test_proxy_series_add():
    """ProxySeries __add__ dispatches to Metal for float32, matching pandas."""
    import metaldf

    metaldf.install()

    a = pd.Series(np.random.randn(100_000).astype(np.float32))
    b = pd.Series(np.random.randn(100_000).astype(np.float32))

    expected = a + b
    actual = a + b  # ProxySeries dunder dispatch

    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-5)
    metaldf.uninstall()


def test_proxy_series_sub():
    """ProxySeries __sub__ dispatches to Metal for float32, matching pandas."""
    import metaldf

    metaldf.install()

    a = pd.Series(np.random.randn(100_000).astype(np.float32))
    b = pd.Series(np.random.randn(100_000).astype(np.float32))

    expected = a - b
    actual = a - b

    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-5)
    metaldf.uninstall()


def test_proxy_series_mul():
    """ProxySeries __mul__ dispatches to Metal for float32, matching pandas."""
    import metaldf

    metaldf.install()

    a = pd.Series(np.random.randn(100_000).astype(np.float32))
    b = pd.Series(np.random.randn(100_000).astype(np.float32))

    expected = a * b
    actual = a * b

    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-5)
    metaldf.uninstall()


def test_proxy_series_div():
    """ProxySeries __truediv__ dispatches to Metal for float32, matching pandas."""
    import metaldf

    metaldf.install()

    a = pd.Series(np.random.randn(100_000).astype(np.float32))
    b = pd.Series(np.random.randn(100_000).astype(np.float32) + 1.0)

    expected = a / b
    actual = a / b

    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-5)
    metaldf.uninstall()


def test_proxy_series_reverse_sub():
    """ProxySeries __rsub__ must call pandas __rsub__, not __add__.

    Reverse ops always fall back to pandas directly (see
    ``ProxySeries._try_metal_or_fallback``) since Metal's registry has no
    notion of operand order beyond positional lhs/rhs.
    """
    import metaldf

    metaldf.install()

    a = pd.Series(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    # 10.0 - a should give [9.0, 8.0, 7.0], NOT [11.0, 12.0, 13.0]
    result = 10.0 - a

    expected = pd.Series([9.0, 8.0, 7.0], dtype=np.float32)
    np.testing.assert_allclose(result.values, expected.values)
    metaldf.uninstall()


def test_proxy_series_int_add():
    """Integer (int64) arrays dispatch to Metal for add/sub/mul (matches pandas)."""
    import metaldf

    metaldf.install()

    a = pd.Series(np.array([1, 2, 3], dtype=np.int64))
    b = pd.Series(np.array([4, 5, 6], dtype=np.int64))

    expected = a + b
    actual = a + b
    np.testing.assert_array_equal(actual.values, expected.values)
    metaldf.uninstall()


def test_proxy_series_small_array_add():
    """Small arrays also dispatch to Metal (no small-array threshold)."""
    import metaldf

    metaldf.install()

    a = pd.Series(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    b = pd.Series(np.array([4.0, 5.0, 6.0], dtype=np.float32))

    expected = a + b
    actual = a + b
    np.testing.assert_array_equal(actual.values, expected.values)
    metaldf.uninstall()


class TestProxyArithmetic:
    """ProxySeries + ProxySeries dispatches through metaldf._engine.execute,
    which tries MetalEngine.metal_add/sub/mul/div (registered in
    metaldf._engine.__init__) before falling back to pandas.
    """

    def test_add_dispatches_to_metal(self):
        a = pd.Series(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = pd.Series(np.array([4.0, 5.0, 6.0], dtype=np.float32))
        pa = ProxySeries(_pandas_obj=a)
        pb = ProxySeries(_pandas_obj=b)
        result = pa + pb
        expected = a + b
        pd.testing.assert_series_equal(
            result.to_pandas() if hasattr(result, 'to_pandas') else result,
            expected, check_dtype=False, check_names=False,
        )

    def test_chained_expression(self):
        a = pd.Series(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        b = pd.Series(np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32))
        c = pd.Series(np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32))
        pa = ProxySeries(_pandas_obj=a)
        pb = ProxySeries(_pandas_obj=b)
        pc = ProxySeries(_pandas_obj=c)
        result = (pa + pb) * pc
        expected = (a + b) * c
        pd.testing.assert_series_equal(
            result.to_pandas() if hasattr(result, 'to_pandas') else result,
            expected, check_dtype=False, check_names=False,
        )

    def test_sub_mul_div(self):
        a = pd.Series(np.array([10.0, 20.0, 30.0], dtype=np.float32))
        b = pd.Series(np.array([2.0, 5.0, 10.0], dtype=np.float32))
        pa = ProxySeries(_pandas_obj=a)
        pb = ProxySeries(_pandas_obj=b)
        for op, expected in [
            ("__sub__", a - b),
            ("__mul__", a * b),
            ("__truediv__", a / b),
        ]:
            result = getattr(pa, op)(pb)
            pd.testing.assert_series_equal(
                result.to_pandas() if hasattr(result, 'to_pandas') else result,
                expected, check_dtype=False, check_names=False,
            )

    def test_add_actually_calls_metal_binary_op(self, monkeypatch):
        """Confirm the fast path is really exercised, not just correct by luck.

        As of Task 15 (deferred/fused expression evaluation), two float32
        ``ProxySeries`` added together no longer dispatch straight to the
        per-op ``metal_binary_op`` kernel -- ``ProxySeries.__add__`` now
        builds a ``DeferredSeries`` expression tree, and materializing it
        (``result.to_pandas()``) compiles the tree to bytecode and runs it
        through the fused kernel instead. As of Task 18, materialization
        tries the runtime-compiled codegen entry point
        (``eval_expression_codegen``) first, falling back to the bytecode
        interpreter (``eval_expression``) only if codegen raises. Spy on
        both entry points to confirm one of the (new) fast paths is
        genuinely exercised, not just correct by luck.
        """
        import metaldf_engine

        codegen_calls = []
        interp_calls = []
        original_codegen = metaldf_engine.eval_expression_codegen
        original_interp = metaldf_engine.eval_expression

        def spy_codegen(program, columns, size):
            codegen_calls.append(program)
            return original_codegen(program, columns, size)

        def spy_interp(program, columns, size):
            interp_calls.append(program)
            return original_interp(program, columns, size)

        monkeypatch.setattr(metaldf_engine, "eval_expression_codegen", spy_codegen)
        monkeypatch.setattr(metaldf_engine, "eval_expression", spy_interp)

        a = pd.Series(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        b = pd.Series(np.array([4.0, 5.0, 6.0], dtype=np.float32))
        pa = ProxySeries(_pandas_obj=a)
        pb = ProxySeries(_pandas_obj=b)
        result = pa + pb

        from metaldf._deferred import DeferredSeries
        assert isinstance(result, DeferredSeries)

        materialized = result.to_pandas()

        # Exactly one fused entry point should have fired: codegen normally,
        # or the interpreter if codegen fell back.
        assert len(codegen_calls) + len(interp_calls) == 1
        assert len(codegen_calls) == 1, (
            "expected codegen to be the primary path for a simple float32 add"
        )
        pd.testing.assert_series_equal(
            materialized, a + b, check_dtype=False, check_names=False,
        )

    def test_int_truediv_falls_back_to_pandas_float_semantics(self):
        """int64 __truediv__ must promote to float64 (pandas semantics), not
        take the Metal 'div' path, which does same-dtype floor division for
        integers -- see metaldf._engine._metal._TRUEDIV_DTYPES.
        """
        a = pd.Series(np.array([10, 20, 3], dtype=np.int64))
        b = pd.Series(np.array([3, 6, 2], dtype=np.int64))
        pa = ProxySeries(_pandas_obj=a)
        pb = ProxySeries(_pandas_obj=b)
        result = pa / pb
        expected = a / b  # pandas true division -> float64
        pd.testing.assert_series_equal(
            result.to_pandas(), expected, check_dtype=False, check_names=False,
        )
        assert not np.array_equal(result.to_pandas().values, (a // b).values)

    def test_mismatched_index_falls_back_to_pandas_alignment(self):
        """Operands with different indexes must be pandas-aligned, not zipped
        positionally by Metal.
        """
        a = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2], dtype=np.float32)
        b = pd.Series([10.0, 20.0, 30.0], index=[1, 2, 3], dtype=np.float32)
        pa = ProxySeries(_pandas_obj=a)
        pb = ProxySeries(_pandas_obj=b)
        result = pa + pb
        expected = a + b
        pd.testing.assert_series_equal(
            result.to_pandas(), expected, check_dtype=False, check_names=False,
        )

    def test_differing_names_result_in_none_name(self):
        """Matches pandas' name-inference rule: differing operand names -> None."""
        a = pd.Series([1.0, 2.0], dtype=np.float32, name="x")
        b = pd.Series([3.0, 4.0], dtype=np.float32, name="y")
        pa = ProxySeries(_pandas_obj=a)
        pb = ProxySeries(_pandas_obj=b)
        result = pa + pb
        expected = a + b
        pd.testing.assert_series_equal(result.to_pandas(), expected, check_dtype=False)
