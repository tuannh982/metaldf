"""Integration tests for Task 7.2: ProxyRolling Metal GPU dispatch.

Exercises `series.rolling(window).{sum,mean,min,max,count}()` through the
full `metaldf.install()` proxy stack (not the raw `metaldf_engine` bridge --
see `tests/test_rolling.py` for that), verifying:

- Results match plain pandas exactly, including pandas' actual default
  `min_periods=window` semantics (NaN until the window fills) -- not the
  GPU kernel's native `min_periods=1` ramp-up behavior (see
  `ProxyRolling._try_metal_rolling`'s docstring in `src/metaldf/_wrappers.py`).
- The GPU kernel is actually invoked for the eligible (float32, no
  unsupported kwargs) case, via a call-counting monkeypatch -- pure value
  comparisons alone can't distinguish "dispatched to Metal and got it
  right" from "silently fell back to pandas".
- Ineligible cases (non-float32 dtype, extra rolling kwargs like
  `center=True`) fall back to pandas without ever touching the kernel.
"""

import numpy as np
import pandas as pd
import pytest

try:
    import metaldf_engine

    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")

WINDOW_SIZES = [3, 10, 100]


def _float32_array(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-100, 100, size=n).astype(np.float32)


class _CallSpy:
    """Monkeypatches a `metaldf_engine.metal_rolling_<op>` function to count calls."""

    def __init__(self, monkeypatch, op_name):
        self.calls = []
        self._orig = getattr(metaldf_engine, f"metal_rolling_{op_name}")

        def spy(ms, window):
            self.calls.append(window)
            return self._orig(ms, window)

        monkeypatch.setattr(metaldf_engine, f"metal_rolling_{op_name}", spy)


@pytest.fixture
def proxied_pandas():
    """Installs metaldf's import interceptor, yields the proxied `pandas` module."""
    import metaldf

    metaldf.install()
    try:
        import pandas as proxied

        yield proxied
    finally:
        metaldf.uninstall()


class TestRollingSumMatchesPandas:
    @pytest.mark.parametrize("window", WINDOW_SIZES)
    def test_default_min_periods(self, proxied_pandas, monkeypatch, window):
        """Verify `series.rolling(window).sum()` matches pandas and dispatches to Metal.

        No explicit `min_periods` is passed here, so this exercises pandas'
        actual default (NaN for the ramp-up region), not the GPU kernel's
        native `min_periods=1` behavior.
        """
        spy = _CallSpy(monkeypatch, "sum")
        arr = _float32_array(500, seed=window)
        s = proxied_pandas.Series(arr)

        actual = s.rolling(window).sum()
        expected = pd.Series(arr).rolling(window).sum()

        assert spy.calls == [window]
        np.testing.assert_allclose(
            np.asarray(actual), expected.to_numpy(), rtol=1e-4, atol=1e-3, equal_nan=True
        )

    def test_min_periods_one(self, proxied_pandas, monkeypatch):
        """Explicit min_periods=1 matches pandas and dispatches to Metal."""
        spy = _CallSpy(monkeypatch, "sum")
        arr = _float32_array(500, seed=1)
        s = proxied_pandas.Series(arr)

        actual = s.rolling(10, min_periods=1).sum()
        expected = pd.Series(arr).rolling(10, min_periods=1).sum()

        assert spy.calls == [10]
        np.testing.assert_allclose(np.asarray(actual), expected.to_numpy(), rtol=1e-4, atol=1e-3)


class TestRollingMeanMatchesPandas:
    def test_matches_pandas(self, proxied_pandas, monkeypatch):
        spy = _CallSpy(monkeypatch, "mean")
        arr = _float32_array(500, seed=2)
        s = proxied_pandas.Series(arr)

        actual = s.rolling(10).mean()
        expected = pd.Series(arr).rolling(10).mean()

        assert spy.calls == [10]
        np.testing.assert_allclose(
            np.asarray(actual), expected.to_numpy(), rtol=1e-4, atol=1e-3, equal_nan=True
        )


class TestRollingMinMaxMatchPandas:
    def test_min(self, proxied_pandas, monkeypatch):
        spy = _CallSpy(monkeypatch, "min")
        arr = _float32_array(500, seed=3)
        s = proxied_pandas.Series(arr)

        actual = s.rolling(5).min()
        expected = pd.Series(arr).rolling(5).min()

        assert spy.calls == [5]
        np.testing.assert_allclose(
            np.asarray(actual), expected.to_numpy(), rtol=1e-4, atol=1e-3, equal_nan=True
        )

    def test_max(self, proxied_pandas, monkeypatch):
        spy = _CallSpy(monkeypatch, "max")
        arr = _float32_array(500, seed=4)
        s = proxied_pandas.Series(arr)

        actual = s.rolling(5).max()
        expected = pd.Series(arr).rolling(5).max()

        assert spy.calls == [5]
        np.testing.assert_allclose(
            np.asarray(actual), expected.to_numpy(), rtol=1e-4, atol=1e-3, equal_nan=True
        )


class TestRollingCountMatchesPandas:
    def test_matches_pandas(self, proxied_pandas, monkeypatch):
        spy = _CallSpy(monkeypatch, "count")
        arr = _float32_array(500, seed=5)
        s = proxied_pandas.Series(arr)

        actual = s.rolling(7).count()
        expected = pd.Series(arr).rolling(7).count()

        assert spy.calls == [7]
        np.testing.assert_allclose(
            np.asarray(actual), expected.to_numpy(), rtol=1e-4, atol=1e-3, equal_nan=True
        )


class TestRollingFallback:
    def test_int32_series_falls_back(self, proxied_pandas, monkeypatch):
        """Non-float32 dtype: no Metal kernel support -- must fall back to pandas."""
        spy = _CallSpy(monkeypatch, "sum")
        arr = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
        s = proxied_pandas.Series(arr)

        actual = s.rolling(3).sum()
        expected = pd.Series(arr).rolling(3).sum()

        assert spy.calls == []
        np.testing.assert_allclose(np.asarray(actual), expected.to_numpy(), equal_nan=True)

    def test_center_kwarg_falls_back(self, proxied_pandas, monkeypatch):
        """Verify `center=True` falls back to pandas instead of dispatching to Metal.

        It changes windowing semantics the GPU kernel doesn't replicate, so
        silently dispatching would produce wrong results.
        """
        spy = _CallSpy(monkeypatch, "mean")
        arr = _float32_array(200, seed=6)
        s = proxied_pandas.Series(arr)

        actual = s.rolling(5, center=True).mean()
        expected = pd.Series(arr).rolling(5, center=True).mean()

        assert spy.calls == []
        np.testing.assert_allclose(
            np.asarray(actual), expected.to_numpy(), rtol=1e-4, atol=1e-3, equal_nan=True
        )

    def test_nan_input_falls_back(self, proxied_pandas, monkeypatch):
        """Verify a NaN in the data falls back to pandas' skipna semantics.

        The GPU kernel's window aggregation isn't null-aware, so it can't be
        trusted to reproduce pandas' NaN-skipping behavior.
        """
        spy = _CallSpy(monkeypatch, "sum")
        arr = _float32_array(50, seed=7)
        arr[5] = np.nan

        s = proxied_pandas.Series(arr)
        actual = s.rolling(3, min_periods=1).sum()
        expected = pd.Series(arr).rolling(3, min_periods=1).sum()

        assert spy.calls == []
        np.testing.assert_allclose(
            np.asarray(actual), expected.to_numpy(), rtol=1e-4, atol=1e-3, equal_nan=True
        )

    def test_extra_agg_args_fall_back(self, proxied_pandas, monkeypatch):
        """Verify args/kwargs passed straight to `.sum()` bypass the Metal path.

        e.g. `numeric_only` -- these replay on pandas via the recorded
        rolling call instead.
        """
        spy = _CallSpy(monkeypatch, "sum")
        arr = _float32_array(200, seed=8)
        s = proxied_pandas.Series(arr)

        actual = s.rolling(4).sum(numeric_only=True)
        expected = pd.Series(arr).rolling(4).sum(numeric_only=True)

        assert spy.calls == []
        np.testing.assert_allclose(
            np.asarray(actual), expected.to_numpy(), rtol=1e-4, atol=1e-3, equal_nan=True
        )

    def test_std_falls_back_via_getattr(self, proxied_pandas):
        """Verify a method with no explicit Metal-dispatching override still works.

        e.g. `std()` -- replayed on pandas via `ProxyRolling.__getattr__`.
        """
        arr = _float32_array(200, seed=9)
        s = proxied_pandas.Series(arr)

        actual = s.rolling(6).std()
        expected = pd.Series(arr).rolling(6).std()

        np.testing.assert_allclose(
            np.asarray(actual), expected.to_numpy(), rtol=1e-4, atol=1e-3, equal_nan=True
        )


class TestDataFrameRolling:
    def test_dataframe_rolling_falls_back(self, proxied_pandas):
        """Verify DataFrame-level rolling still returns correct, proxy-wrapped results.

        There's no Metal kernel for it (the kernels are Series-only), so
        this always falls back to pandas.
        """
        data = {
            "a": _float32_array(50, seed=10),
            "b": _float32_array(50, seed=11),
        }
        df = proxied_pandas.DataFrame(data)

        actual = df.rolling(4).sum()
        expected = pd.DataFrame(data).rolling(4).sum()

        actual_pd = actual.to_pandas() if hasattr(actual, "to_pandas") else actual
        pd.testing.assert_frame_equal(actual_pd, expected, check_exact=False, rtol=1e-4)
