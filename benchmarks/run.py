"""Benchmark suite for metaldf -- all operations, 5M elements, verified against pandas.

Dispatches through the public ``ProxySeries``/``ProxyDataFrame`` layer (not
raw ``metaldf_engine.*`` calls) for numeric reductions, sort/argsort,
numeric groupby, and the string accessor ops (``contains``, ``startswith``,
``endswith``, ``find``, ``lower``, ``upper``, ``strip``, ``replace``) --
so the numbers reflect what a real caller of ``metaldf`` sees, including
proxy/array-conversion overhead.

String *sort* and *groupby* are the exception: ``ProxySeries.sort_values()``
and ``ProxyGroupBy`` only dispatch to Metal for numeric dtypes (see
``_SORT_DTYPES``/``_GROUPBY_DTYPES`` in ``metaldf._engine._metal``) --
string keys/values fall back to plain pandas at the proxy layer. To still
benchmark the dedicated ``metal_string_sort``/``metal_string_groupby`` Rust
kernels, those two benchmarks call ``metaldf_engine.*`` directly (same
pattern as ``tests/test_string_sort.py``/``tests/test_string_groupby.py``).

Every benchmark verifies correctness first: the Metal (proxy) result is
compared against the plain-pandas result with an assertion, and only once
that passes does timing run. A mismatch raises immediately instead of being
silently recorded.

For the ``.str.*`` accessor benchmarks, a single ``ProxySeries`` instance is
reused across all string-accessor benchmarks. The first ``.str.*`` call on
it builds and caches a ``MetalSeries`` (offsets+chars GPU buffers -- the
expensive part, ~500ms/5M rows) on the series itself; every later ``.str.*``
call on that *same* instance (including later benchmarks in this file)
reuses the cached buffers, so only the kernel dispatch is timed. A freshly
constructed ``ProxySeries`` would pay that build cost on every call instead.
"""

from __future__ import annotations

import argparse
import gc
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Allow `python benchmarks/run.py` to find the sibling `data.py` module
# regardless of the current working directory (running a script directly
# puts its own directory on sys.path, not the repo root, so a package-style
# `from benchmarks.data import ...` import would fail).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import make_numeric_array, make_numeric_keys, make_string_list  # noqa: E402, I001
from metaldf._wrappers import ProxyDataFrame, ProxySeries  # noqa: E402

N = 10_000_000


# ---------------------------------------------------------------------------
# Verified benchmarking helper
# ---------------------------------------------------------------------------

def bench_verified(
    name: str,
    metal_fn: Callable[[], Any],
    pandas_fn: Callable[[], Any],
    verify_fn: Callable[[Any, Any], None],
    n_runs: int = 5,
) -> dict:
    """Verify correctness once, then time both paths (best-of-`n_runs`).

    ``verify_fn`` raises on mismatch -- a correctness failure aborts the run
    instead of being silently recorded. The verification call itself is not
    timed.
    """
    metal_result = metal_fn()
    pandas_result = pandas_fn()
    verify_fn(metal_result, pandas_result)

    times_metal: list[float] = []
    times_pandas: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        metal_fn()
        times_metal.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        pandas_fn()
        times_pandas.append(time.perf_counter() - t0)

    m, p = min(times_metal), min(times_pandas)
    speedup = p / m if m > 0 else float("inf")
    marker = "OK" if speedup > 1.0 else "ok"
    print(f"  {name:35s}  metal={m*1000:8.1f}ms  pandas={p*1000:8.1f}ms  {speedup:6.2f}x  {marker}")
    return {
        "name": name,
        "metal_ms": round(m * 1000, 2),
        "pandas_ms": round(p * 1000, 2),
        "speedup": round(speedup, 2),
    }


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------

def verify_scalar(
    metal: Any, pandas: Any, dtype: np.dtype | None = None, rtol: float = 1e-3
) -> None:
    """Verify a reduction scalar. Exact match for ints, tolerant for floats."""
    if dtype in (np.int32, np.int64):
        assert int(metal) == int(pandas), f"scalar mismatch: metal={metal!r} pandas={pandas!r}"
    else:
        diff = abs(float(metal) - float(pandas))
        tol = rtol * abs(float(pandas)) + 1e-6
        assert diff <= tol, (
            f"scalar mismatch: metal={metal!r} pandas={pandas!r} diff={diff} > tol={tol}"
        )


def _unwrap(x: Any) -> Any:
    return x.to_pandas() if hasattr(x, "to_pandas") else x


def verify_series(metal: Any, pandas: pd.Series) -> None:
    """Verify a Metal-produced Series matches the pandas result (dtype-agnostic)."""
    pd.testing.assert_series_equal(_unwrap(metal), pandas, check_names=False, check_dtype=False)


def verify_sort(metal: Any, pandas: pd.Series) -> None:
    """Compare sorted *values* only.

    Metal sort returns a fresh 0..n RangeIndex while ``pd.Series.sort_values``
    keeps the original (permuted) index -- ``assert_series_equal`` would flag
    that index mismatch even though the sorted values themselves match, so
    compare the underlying arrays directly instead.
    """
    metal_vals = np.asarray(_unwrap(metal))
    pandas_vals = pandas.to_numpy()
    assert np.array_equal(metal_vals, pandas_vals), "sort mismatch"


def verify_argsort(original: np.ndarray) -> Callable[[Any, Any], None]:
    """Return a verify_fn asserting that `metal` indices sort `original`.

    Doesn't compare index-for-index against pandas' own argsort: ties can
    legitimately be broken differently between a GPU sort and pandas', so
    only sortedness of the resulting permutation is checked (matching
    ``tests``' own convention for argsort).
    """

    def _verify(metal: Any, _pandas: Any) -> None:
        indices = np.asarray(_unwrap(metal))
        reordered = original[indices]
        assert bool(np.all(reordered[:-1] <= reordered[1:])), "argsort result is not sorted"

    return _verify


def verify_groupby(metal: Any, pandas: pd.Series, rtol: float = 5e-2) -> None:
    """Verify a groupby result, ignoring key order (Metal doesn't preserve pandas' key order).

    Tolerance is generous (5%) because float32 groupby accumulation order differs
    between GPU (hash/sort-based atomic adds) and CPU (sequential), and the error
    grows with element count (~100K elements per group at 10M/100 keys).
    """
    metal_s = _unwrap(metal).sort_index()
    pandas_s = pandas.sort_index()
    pd.testing.assert_series_equal(
        metal_s, pandas_s, check_names=False, check_dtype=False, rtol=rtol
    )


def verify_string_sort(metal: Any, pandas: pd.Series) -> None:
    """Compare sorted *values* only -- ties can land in either order."""
    metal_vals = metal.to_strings()
    pandas_vals = list(pandas)
    assert metal_vals == pandas_vals, "string sort mismatch"


def verify_string_groupby(metal: Any, pandas: pd.Series, rtol: float = 1e-3) -> None:
    """Verify a direct ``metal_string_groupby`` (keys, values) pair against pandas groupby."""
    keys_ms, vals_ms = metal
    metal_dict = dict(zip(keys_ms.to_strings(), vals_ms.to_numpy(), strict=True))
    for key, expected in pandas.items():
        assert key in metal_dict, f"string groupby: missing key {key!r}"
        got = metal_dict[key]
        diff = abs(float(got) - float(expected))
        tol = rtol * abs(float(expected)) + 1e-4
        assert diff <= tol, (
            f"string groupby mismatch key={key!r}: metal={got!r} pandas={expected!r}"
        )


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_reductions() -> list[dict]:
    """Benchmark sum/min/max/mean across float32/int32/int64."""
    print("\n=== Reductions (5M elements) ===")
    results = []
    for dtype in [np.float32, np.int32, np.int64]:
        arr = make_numeric_array(N, dtype)
        s = pd.Series(arr)
        proxy_s = ProxySeries(_pandas_obj=s)

        for op in ["sum", "min", "max", "mean"]:
            r = bench_verified(
                f"{op} {np.dtype(dtype).name}",
                lambda proxy_s=proxy_s, op=op: getattr(proxy_s, op)(),
                lambda s=s, op=op: getattr(s, op)(),
                lambda m, p, dtype=dtype: verify_scalar(m, p, dtype),
            )
            r.update({"category": "reduction", "op": op, "dtype": np.dtype(dtype).name})
            results.append(r)

        del arr, s, proxy_s
        gc.collect()
    return results


def bench_sort() -> list[dict]:
    """Benchmark sort_values/argsort across float32/int32/int64."""
    print("\n=== Sort / Argsort (5M elements) ===")
    results = []
    for dtype in [np.float32, np.int32, np.int64]:
        arr = make_numeric_array(N, dtype)
        s = pd.Series(arr)
        proxy_s = ProxySeries(_pandas_obj=s)

        r = bench_verified(
            f"sort {np.dtype(dtype).name}",
            lambda proxy_s=proxy_s: proxy_s.sort_values(),
            lambda s=s: s.sort_values(),
            verify_sort,
        )
        r.update({"category": "sort", "op": "sort", "dtype": np.dtype(dtype).name})
        results.append(r)

        r = bench_verified(
            f"argsort {np.dtype(dtype).name}",
            lambda proxy_s=proxy_s: proxy_s.argsort(),
            lambda s=s: s.to_numpy().argsort(),
            verify_argsort(arr),
        )
        r.update({"category": "sort", "op": "argsort", "dtype": np.dtype(dtype).name})
        results.append(r)

        del arr, s, proxy_s
        gc.collect()
    return results


def bench_groupby_numeric() -> list[dict]:
    """Benchmark groupby sum/mean/min/max/count at low and high key cardinality."""
    print("\n=== GroupBy Numeric (5M elements) ===")
    results = []
    # groupby only supports float32/int32 on the Metal side (no 64-bit atomics)
    dtypes = [np.float32, np.int32]
    cardinalities = [("low_card", 100), ("high_card", N)]
    agg_ops = ["sum", "mean", "min", "max", "count"]

    for card_name, n_unique in cardinalities:
        print(f"  -- {card_name} ({n_unique:,} unique keys) --")
        for dtype in dtypes:
            keys = make_numeric_keys(N, dtype, n_unique)
            values = make_numeric_array(N, dtype)
            pandas_df = pd.DataFrame({"key": keys, "val": values})
            proxy_df = ProxyDataFrame(_pandas_obj=pandas_df)

            for op in agg_ops:
                def metal_fn(proxy_df=proxy_df, op=op):
                    return getattr(proxy_df.groupby("key")["val"], op)()

                def pandas_fn(pandas_df=pandas_df, op=op):
                    return getattr(pandas_df.groupby("key")["val"], op)()

                r = bench_verified(
                    f"{card_name} {op} {np.dtype(dtype).name}", metal_fn, pandas_fn, verify_groupby
                )
                r.update({
                    "category": f"groupby_{card_name}",
                    "op": op,
                    "dtype": np.dtype(dtype).name,
                })
                results.append(r)

            del keys, values, pandas_df, proxy_df
            gc.collect()
    return results


def bench_strings() -> list[dict]:
    """Benchmark the .str accessor ops plus direct string sort/groupby kernels."""
    print("\n=== String Operations (5M elements, 100 categories) ===")
    results = []
    strings = make_string_list(N)
    pd_str = pd.Series(strings)
    proxy_str = ProxySeries(_pandas_obj=pd_str)

    # These all go through the `.str` accessor -- the first call below
    # builds `proxy_str`'s MetalSeries cache; every later `.str.*` call in
    # this function (including later benchmarks) reuses it.
    str_ops: list[tuple[str, Callable[[], Any], Callable[[], Any]]] = [
        ("str.contains('cat_0050')",
         lambda: proxy_str.str.contains("cat_0050"),
         lambda: pd_str.str.contains("cat_0050")),
        ("str.startswith('cat_0005')",
         lambda: proxy_str.str.startswith("cat_0005"),
         lambda: pd_str.str.startswith("cat_0005")),
        ("str.endswith('0050')",
         lambda: proxy_str.str.endswith("0050"),
         lambda: pd_str.str.endswith("0050")),
        ("str.find('cat_0050')",
         lambda: proxy_str.str.find("cat_0050"),
         lambda: pd_str.str.find("cat_0050")),
        ("str.lower()",
         lambda: proxy_str.str.lower(),
         lambda: pd_str.str.lower()),
        ("str.upper()",
         lambda: proxy_str.str.upper(),
         lambda: pd_str.str.upper()),
        ("str.strip()",
         lambda: proxy_str.str.strip(),
         lambda: pd_str.str.strip()),
        ("str.replace('cat_', 'category_')",
         lambda: proxy_str.str.replace("cat_", "category_", regex=False),
         lambda: pd_str.str.replace("cat_", "category_", regex=False)),
    ]
    for name, metal_fn, pandas_fn in str_ops:
        r = bench_verified(name, metal_fn, pandas_fn, verify_series)
        r.update({"category": "string_op", "op": name, "dtype": "object"})
        results.append(r)

    # String sort/groupby: the proxy layer falls back to pandas for object
    # dtype (see module docstring), so exercise the dedicated Rust kernels
    # directly, the same way tests/test_string_sort.py and
    # tests/test_string_groupby.py do.
    import metaldf_engine

    ms = metaldf_engine.MetalSeries.from_strings(strings)
    r = bench_verified(
        "string sort (direct)",
        lambda: metaldf_engine.metal_string_sort(ms, True),
        lambda: pd_str.sort_values(),
        verify_string_sort,
    )
    r.update({"category": "string_sort", "op": "sort", "dtype": "object"})
    results.append(r)

    values = np.random.default_rng(42).random(N).astype(np.float32)
    mv = metaldf_engine.MetalSeries.from_numpy(values)
    pd_values = pd.Series(values)
    for op in ["sum", "mean", "min", "max", "count"]:
        r = bench_verified(
            f"string groupby {op} (direct)",
            lambda op=op: metaldf_engine.metal_string_groupby(ms, mv, op),
            lambda op=op: getattr(pd_values.groupby(pd_str), op)(),
            verify_string_groupby,
        )
        r.update({"category": "string_groupby", "op": op, "dtype": "object"})
        results.append(r)

    return results


def bench_elementwise() -> list[dict]:
    """Benchmark element-wise ops: single binary and chained unfused."""
    print("\n=== Element-wise Operations (5M elements) ===")
    results = []
    a_arr = make_numeric_array(N, np.float32, seed=42)
    b_arr = make_numeric_array(N, np.float32, seed=43)
    c_arr = make_numeric_array(N, np.float32, seed=44)
    d_arr = make_numeric_array(N, np.float32, seed=45)
    a_s, b_s = pd.Series(a_arr), pd.Series(b_arr)
    c_s, d_s = pd.Series(c_arr), pd.Series(d_arr)
    pa = ProxySeries(_pandas_obj=a_s)
    pb = ProxySeries(_pandas_obj=b_s)
    pc = ProxySeries(_pandas_obj=c_s)
    pd_ = ProxySeries(_pandas_obj=d_s)
    for name, mfn, pfn in [
        ("add float32", lambda: pa + pb, lambda: a_s + b_s),
        ("sub float32", lambda: pa - pb, lambda: a_s - b_s),
        ("mul float32", lambda: pa * pb, lambda: a_s * b_s),
    ]:
        r = bench_verified(name, mfn, pfn, verify_series)
        r.update({"category": "elementwise", "op": name, "dtype": "float32"})
        results.append(r)
    r = bench_verified(
        "chained (a+b)*c-d float32",
        lambda: (pa + pb) * pc - pd_,
        lambda: (a_s + b_s) * c_s - d_s,
        verify_series,
    )
    r.update({"category": "elementwise_chain", "op": "(a+b)*c-d", "dtype": "float32"})
    results.append(r)
    return results


def bench_fused() -> list[dict]:
    """Compare fused vs unfused for chained expressions."""
    print("\n=== Fused Expression Evaluation (5M float32) ===")
    results = []
    import metaldf_engine

    a_arr = make_numeric_array(N, np.float32, seed=42)
    b_arr = make_numeric_array(N, np.float32, seed=43)
    c_arr = make_numeric_array(N, np.float32, seed=44)
    d_arr = make_numeric_array(N, np.float32, seed=45)
    ma = metaldf_engine.MetalSeries.from_numpy(a_arr)
    mb = metaldf_engine.MetalSeries.from_numpy(b_arr)
    mc = metaldf_engine.MetalSeries.from_numpy(c_arr)
    md = metaldf_engine.MetalSeries.from_numpy(d_arr)
    # (col0 + col1) * col2 - col3
    program = bytes([0, 1, 16, 2, 18, 3, 17])

    pandas_a, pandas_b = pd.Series(a_arr), pd.Series(b_arr)
    pandas_c, pandas_d = pd.Series(c_arr), pd.Series(d_arr)

    def verify_fused(metal, pandas):
        np.testing.assert_allclose(metal.to_numpy(), pandas.to_numpy(), rtol=1e-4)

    # Direct fused kernel
    r = bench_verified(
        "fused (a+b)*c-d direct",
        lambda: metaldf_engine.eval_expression(program, [ma, mb, mc, md], N),
        lambda: (pandas_a + pandas_b) * pandas_c - pandas_d,
        verify_fused,
    )
    r.update({"category": "fused", "op": "(a+b)*c-d", "dtype": "float32"})
    results.append(r)

    # Unfused (3 separate kernel launches)
    r = bench_verified(
        "unfused (a+b)*c-d direct",
        lambda: metaldf_engine.metal_binary_op(
            "sub",
            metaldf_engine.metal_binary_op(
                "mul",
                metaldf_engine.metal_binary_op("add", ma, mb),
                mc,
            ),
            md,
        ),
        lambda: (pandas_a + pandas_b) * pandas_c - pandas_d,
        verify_fused,
    )
    r.update({"category": "unfused", "op": "(a+b)*c-d", "dtype": "float32"})
    results.append(r)

    # End-to-end via DeferredSeries proxy
    pa = ProxySeries(_pandas_obj=pandas_a)
    pb = ProxySeries(_pandas_obj=pandas_b)
    pc = ProxySeries(_pandas_obj=pandas_c)
    pd_ = ProxySeries(_pandas_obj=pandas_d)

    def deferred_chain():
        return ((pa + pb) * pc - pd_).to_pandas()

    r = bench_verified(
        "deferred (a+b)*c-d e2e",
        deferred_chain,
        lambda: (pandas_a + pandas_b) * pandas_c - pandas_d,
        verify_series,
    )
    r.update({"category": "fused_e2e", "op": "(a+b)*c-d", "dtype": "float32"})
    results.append(r)

    # Size sweep
    for sz in [1_000_000, 5_000_000, 10_000_000, 20_000_000]:
        a = np.random.default_rng(42).standard_normal(sz).astype(np.float32)
        b = np.random.default_rng(43).standard_normal(sz).astype(np.float32)
        c = np.random.default_rng(44).standard_normal(sz).astype(np.float32)
        d = np.random.default_rng(45).standard_normal(sz).astype(np.float32)
        ma2 = metaldf_engine.MetalSeries.from_numpy(a)
        mb2 = metaldf_engine.MetalSeries.from_numpy(b)
        mc2 = metaldf_engine.MetalSeries.from_numpy(c)
        md2 = metaldf_engine.MetalSeries.from_numpy(d)
        pa2, pb2, pc2, pd2 = pd.Series(a), pd.Series(b), pd.Series(c), pd.Series(d)
        r = bench_verified(
            f"fused (a+b)*c-d {sz//1000}K",
            lambda ma2=ma2, mb2=mb2, mc2=mc2, md2=md2, sz=sz: metaldf_engine.eval_expression(program, [ma2, mb2, mc2, md2], sz),
            lambda pa2=pa2, pb2=pb2, pc2=pc2, pd2=pd2: (pa2 + pb2) * pc2 - pd2,
            verify_fused,
        )
        r.update({"category": "fused_sweep", "op": f"(a+b)*c-d_{sz}", "dtype": "float32"})
        results.append(r)
        del a, b, c, d, ma2, mb2, mc2, md2, pa2, pb2, pc2, pd2
        gc.collect()

    return results


def bench_codegen() -> list[dict]:
    """Compare interpreter vs codegen for expression evaluation."""
    print("\n=== Codegen vs Interpreter (5M float32) ===")
    results = []
    import metaldf_engine

    a_arr = make_numeric_array(N, np.float32, seed=42)
    b_arr = make_numeric_array(N, np.float32, seed=43)
    c_arr = make_numeric_array(N, np.float32, seed=44)
    d_arr = make_numeric_array(N, np.float32, seed=45)
    ma = metaldf_engine.MetalSeries.from_numpy(a_arr)
    mb = metaldf_engine.MetalSeries.from_numpy(b_arr)
    mc = metaldf_engine.MetalSeries.from_numpy(c_arr)
    md = metaldf_engine.MetalSeries.from_numpy(d_arr)
    program = bytes([0, 1, 16, 2, 18, 3, 17])  # (col0+col1)*col2-col3

    pandas_a = pd.Series(a_arr)
    pandas_b = pd.Series(b_arr)
    pandas_c = pd.Series(c_arr)
    pandas_d = pd.Series(d_arr)

    def verify_cg(metal, pandas):
        np.testing.assert_allclose(metal.to_numpy(), pandas.to_numpy(), rtol=1e-4)

    # Warm up codegen cache
    metaldf_engine.eval_expression_codegen(program, [ma, mb, mc, md], N)

    r = bench_verified(
        "interpreter (a+b)*c-d",
        lambda: metaldf_engine.eval_expression(program, [ma, mb, mc, md], N),
        lambda: (pandas_a + pandas_b) * pandas_c - pandas_d,
        verify_cg,
    )
    r.update({"category": "codegen", "op": "interpreter", "dtype": "float32"})
    results.append(r)

    r = bench_verified(
        "codegen cached (a+b)*c-d",
        lambda: metaldf_engine.eval_expression_codegen(program, [ma, mb, mc, md], N),
        lambda: (pandas_a + pandas_b) * pandas_c - pandas_d,
        verify_cg,
    )
    r.update({"category": "codegen", "op": "codegen_cached", "dtype": "float32"})
    results.append(r)

    # Compilation time measurement
    import time
    for n_ops, prog in [
        (1, bytes([0, 1, 16])),  # a+b
        (3, bytes([0, 1, 16, 2, 18, 3, 17])),  # (a+b)*c-d
        (5, bytes([0, 1, 16, 2, 18, 3, 17, 4, 16])),  # ((a+b)*c-d)+e
    ]:
        # Force cache miss by modifying program slightly
        test_prog = bytes(list(prog) + [32])  # append ABS to make unique
        t0 = time.perf_counter()
        metaldf_engine.eval_expression_codegen(test_prog, [ma, mb, mc, md, ma], N)
        compile_ms = (time.perf_counter() - t0) * 1000
        print(f"  codegen compile time ({n_ops} ops): {compile_ms:.1f}ms")

    return results


def bench_fused_reduce() -> list[dict]:
    """Compare fused expression-reduce vs separate codegen+reduce."""
    print("\n=== Fused Expression-Reduce (5M float32) ===")
    results = []
    import metaldf_engine

    a_arr = make_numeric_array(N, np.float32, seed=42)
    b_arr = make_numeric_array(N, np.float32, seed=43)
    c_arr = make_numeric_array(N, np.float32, seed=44)
    d_arr = make_numeric_array(N, np.float32, seed=45)
    ma = metaldf_engine.MetalSeries.from_numpy(a_arr)
    mb = metaldf_engine.MetalSeries.from_numpy(b_arr)
    mc = metaldf_engine.MetalSeries.from_numpy(c_arr)
    md = metaldf_engine.MetalSeries.from_numpy(d_arr)
    program = bytes([0, 1, 16, 2, 18, 3, 17])  # (col0+col1)*col2-col3

    pandas_a, pandas_b = pd.Series(a_arr), pd.Series(b_arr)
    pandas_c, pandas_d = pd.Series(c_arr), pd.Series(d_arr)

    def verify_reduce(metal, pandas):
        assert abs(float(metal) - float(pandas)) / (abs(float(pandas)) + 1e-6) < 0.01

    # Fused: one kernel for expr+reduce
    r = bench_verified(
        "fused sum((a+b)*c-d)",
        lambda: metaldf_engine.eval_expression_reduce("sum", program, [ma, mb, mc, md], N),
        lambda: float(((pandas_a + pandas_b) * pandas_c - pandas_d).sum()),
        verify_reduce,
    )
    r.update({"category": "fused_reduce", "op": "sum((a+b)*c-d)", "dtype": "float32"})
    results.append(r)

    # Separate: codegen then reduce
    r = bench_verified(
        "separate codegen+reduce sum((a+b)*c-d)",
        lambda: float(metaldf_engine.eval_expression_codegen(program, [ma, mb, mc, md], N).to_numpy().sum()),
        lambda: float(((pandas_a + pandas_b) * pandas_c - pandas_d).sum()),
        verify_reduce,
    )
    r.update({"category": "separate_reduce", "op": "sum((a+b)*c-d)", "dtype": "float32"})
    results.append(r)

    # End-to-end via DeferredSeries
    pa = ProxySeries(_pandas_obj=pandas_a)
    pb = ProxySeries(_pandas_obj=pandas_b)
    pc = ProxySeries(_pandas_obj=pandas_c)
    pd_ = ProxySeries(_pandas_obj=pandas_d)

    r = bench_verified(
        "deferred sum((a+b)*c-d) e2e",
        lambda: float(((pa + pb) * pc - pd_).sum()),
        lambda: float(((pandas_a + pandas_b) * pandas_c - pandas_d).sum()),
        verify_reduce,
    )
    r.update({"category": "fused_reduce_e2e", "op": "sum((a+b)*c-d)", "dtype": "float32"})
    results.append(r)

    return results


def bench_long_chains() -> list[dict]:
    """Benchmark longer expression chains and deferred sort."""
    print("\n=== Long Chain Benchmarks (5M float32) ===")
    results = []
    import metaldf_engine

    rng = np.random.default_rng(42)
    arrays = [rng.standard_normal(N).astype(np.float32) for _ in range(8)]
    metal_series = [metaldf_engine.MetalSeries.from_numpy(a) for a in arrays]
    pandas_series = [pd.Series(a) for a in arrays]
    proxy_series = [ProxySeries(_pandas_obj=s) for s in pandas_series]

    def verify_reduce(metal, pandas):
        assert abs(float(metal) - float(pandas)) / (abs(float(pandas)) + 1e-6) < 0.01

    def verify_sort(metal, pandas):
        m = metal.to_numpy() if hasattr(metal, 'to_numpy') else np.asarray(metal)
        p = pandas.to_numpy() if hasattr(pandas, 'to_numpy') else np.asarray(pandas)
        np.testing.assert_allclose(m, p, rtol=1e-4)

    def verify_codegen(metal, pandas):
        # atol handles near-zero results (catastrophic cancellation in
        # subtraction-heavy chains): GPU vs CPU float32 rounding differs at
        # the ~1e-7 absolute level, which blows up under pure rtol when the
        # true value is close to zero.
        np.testing.assert_allclose(metal.to_numpy(), pandas.to_numpy(), rtol=1e-4, atol=1e-4)

    # 5-op chain: a + b * c - d / e  → program: col0 col1 col2 MUL ADD col3 col4 DIV SUB
    prog_5 = bytes([0, 1, 2, 18, 16, 3, 4, 19, 17])
    pandas_5 = lambda: pandas_series[0] + pandas_series[1] * pandas_series[2] - pandas_series[3] / pandas_series[4]

    r = bench_verified(
        "5-op fused codegen",
        lambda: metaldf_engine.eval_expression_codegen(prog_5, metal_series[:5], N),
        pandas_5,
        verify_codegen,
    )
    r.update({"category": "long_chain", "op": "5op_codegen", "dtype": "float32"})
    results.append(r)

    # 5-op fused reduce: sum(a + b * c - d / e)
    r = bench_verified(
        "sum(5-op) fused",
        lambda: metaldf_engine.eval_expression_reduce("sum", prog_5, metal_series[:5], N),
        lambda: float(pandas_5().sum()),
        verify_reduce,
    )
    r.update({"category": "long_chain_reduce", "op": "sum(5op)", "dtype": "float32"})
    results.append(r)

    # 8-op chain: a + b * c - d / e + f * g - h
    prog_8 = bytes([0, 1, 2, 18, 16, 3, 4, 19, 17, 5, 6, 18, 16, 7, 17])
    pandas_8 = lambda: pandas_series[0] + pandas_series[1] * pandas_series[2] - pandas_series[3] / pandas_series[4] + pandas_series[5] * pandas_series[6] - pandas_series[7]

    r = bench_verified(
        "8-op fused codegen",
        lambda: metaldf_engine.eval_expression_codegen(prog_8, metal_series, N),
        pandas_8,
        verify_codegen,
    )
    r.update({"category": "long_chain", "op": "8op_codegen", "dtype": "float32"})
    results.append(r)

    # 8-op fused reduce: sum(8-op chain)
    r = bench_verified(
        "sum(8-op) fused",
        lambda: metaldf_engine.eval_expression_reduce("sum", prog_8, metal_series, N),
        lambda: float(pandas_8().sum()),
        verify_reduce,
    )
    r.update({"category": "long_chain_reduce", "op": "sum(8op)", "dtype": "float32"})
    results.append(r)

    # Deferred sort: (a + b).sort_values()
    pa, pb = proxy_series[0], proxy_series[1]
    r = bench_verified(
        "deferred (a+b).sort_values()",
        lambda: (pa + pb).sort_values(),
        lambda: (pandas_series[0] + pandas_series[1]).sort_values(),
        verify_sort,
    )
    r.update({"category": "deferred_sort", "op": "(a+b).sort()", "dtype": "float32"})
    results.append(r)

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(all_results: list[dict]) -> None:
    """Print the subset of benchmarks where Metal beat pandas, sorted by speedup."""
    print(f"\n{'=' * 80}")
    print("  SUMMARY: Metal wins (speedup > 1.0x), sorted by speedup")
    print(f"{'=' * 80}")

    wins = [r for r in all_results if r["speedup"] > 1.0]
    wins.sort(key=lambda r: -r["speedup"])

    if not wins:
        print("  (no wins)")
    else:
        print(f"{'Category':>18} {'Op':>35} {'Speedup':>10}")
        print("-" * 68)
        for r in wins:
            print(f"{r['category']:>18} {r['op']:>35} {r['speedup']:>8.2f}x")

    total = len(all_results)
    print(
        f"\nTotal: {total} benchmarks, {total}/{total} correct (all assertions passed), "
        f"{len(wins)} Metal wins ({100 * len(wins) // total}%)"
    )


def main() -> None:
    """Run the full benchmark suite and print a summary (optionally writing JSON)."""
    parser = argparse.ArgumentParser(
        description="Benchmark metaldf vs pandas (5M elements, all operations)"
    )
    parser.add_argument("--json", type=str, help="Write results to JSON file")
    args = parser.parse_args()

    if platform.system() != "Darwin":
        print("Benchmarks only run on macOS (Metal required).", file=sys.stderr)
        sys.exit(1)

    try:
        import metaldf_engine
    except ImportError:
        print("metaldf_engine not importable. Build the Rust extension first.", file=sys.stderr)
        sys.exit(1)

    gpu = metaldf_engine.metal_gpu_info()
    print(f"GPU: {gpu['name']}")
    print(f"  Family: {gpu['gpu_family']}")
    print(f"  Max threads/threadgroup: {gpu['max_threads_per_threadgroup']}")
    print(f"  Max threadgroup memory: {gpu['max_threadgroup_memory_bytes'] // 1024} KB")
    print(f"  Max buffer length: {gpu['max_buffer_length_bytes'] / (1024**3):.1f} GB")
    print(f"  Tuning: reduce_tg={gpu['tuning_reduce_threadgroup_size']}, "
          f"reduce_n_reads={gpu['tuning_reduce_n_reads']}, "
          f"local_sort={gpu['tuning_local_sort_size']}")
    print(f"\nElements: {N:,}")
    print("Dispatch: via ProxySeries/ProxyDataFrame (string sort/groupby use metaldf_engine)")

    all_results: list[dict] = []
    all_results.extend(bench_reductions())
    gc.collect()
    all_results.extend(bench_sort())
    gc.collect()
    all_results.extend(bench_groupby_numeric())
    gc.collect()
    all_results.extend(bench_strings())
    gc.collect()
    all_results.extend(bench_elementwise())
    gc.collect()
    all_results.extend(bench_fused())
    gc.collect()
    all_results.extend(bench_codegen())
    gc.collect()
    all_results.extend(bench_fused_reduce())
    gc.collect()
    all_results.extend(bench_long_chains())
    gc.collect()

    print_summary(all_results)

    if args.json:
        import json

        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":
    main()
