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

N = 5_000_000


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


def verify_groupby(metal: Any, pandas: pd.Series, rtol: float = 1e-2) -> None:
    """Verify a groupby result, ignoring key order (Metal doesn't preserve pandas' key order)."""
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

    print_summary(all_results)

    if args.json:
        import json

        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults written to {args.json}")


if __name__ == "__main__":
    main()
