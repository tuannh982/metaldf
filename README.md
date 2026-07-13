# MetalDF

GPU-accelerated pandas on Apple Silicon. Drop-in replacement -- no code changes required.

```bash
python -m metaldf my_script.py
```

MetalDF intercepts `import pandas` at runtime and returns proxy objects that dispatch operations to Metal compute shaders on the GPU. Reductions, sorting, groupby, and string operations are accelerated transparently. Unsupported operations fall back to pandas with identical results.

## Architecture

```
+------------------------------------------------------------------+
|  Import Proxy                                                     |
|  Intercepts `import pandas`, returns proxy objects                |
|  User code sees pandas types -- proxy dispatches transparently    |
+------------------------------------------------------------------+
                              |
                    arithmetic: build expression tree (lazy)
                    other ops: try Metal, fallback pandas
                              |
+------------------------------------------------------------------+
|  Proxy & Dispatch                                                 |
|  ProxyDataFrame, ProxySeries wrap Metal + pandas implementations  |
|  DeferredSeries: lazy expression tree, materializes on data access|
+------------------------------------------------------------------+
                              |
                    materialize: codegen -> fused GPU kernel
                    eager ops: direct Metal kernel dispatch
                              |
+------------------+-------------------+
|  PandasEngine    |  MetalEngine      |
|  numpy/pandas    |  Rust + metal-rs  |
|  CPU fallback    |  GPU kernels      |
|                  |  + MSL codegen    |
+------------------+-------------------+
```

The proxy layer is invisible to user code. `isinstance(df, pd.DataFrame)` returns `True`. Operations that have Metal kernels run on the GPU; everything else silently falls back to pandas.

### Lazy Evaluation and Kernel Fusion

Arithmetic operations (`+`, `-`, `*`, `/`) on float32 Series don't execute immediately. Instead, they return a `DeferredSeries` that records the operation as a node in an expression tree. Subsequent arithmetic extends the tree without launching any GPU work.

When the result is actually needed -- `print()`, `.sum()`, `.sort_values()`, assignment to a DataFrame column, or any non-arithmetic operation -- the tree is compiled and executed:

1. **Bytecode compilation**: the expression tree is walked in post-order and emitted as a compact bytecode program (1 byte per operation)
2. **MSL code generation**: the bytecode is decompiled into a Metal Shading Language expression and compiled at runtime via `MTLDevice.newLibraryWithSource`. The compiled pipeline is cached by expression hash (~1.5ms first compile, ~0ms on cache hit)
3. **Single kernel dispatch**: one GPU kernel evaluates the entire expression per-element, reading each input column exactly once

For reductions on expressions (`sum()`, `min()`, `max()`, `mean()`), a fused expression-reduce kernel evaluates the expression AND reduces in a single pass -- the intermediate column is never materialized.

**Without fusion** (naive GPU approach):
```
(a + b) * c - d
  Kernel 1: tmp1 = a + b       (read a,b → write 20MB tmp1)
  Kernel 2: tmp2 = tmp1 * c    (read tmp1,c → write 20MB tmp2)
  Kernel 3: result = tmp2 - d  (read tmp2,d → write 20MB result)
  Total: 3 kernel launches, 60MB intermediate writes
```

**With fusion** (what MetalDF does):
```
(a + b) * c - d
  1 kernel: result[i] = (a[i] + b[i]) * c[i] - d[i]
  Total: 1 kernel launch, 0 intermediate writes
```

NVIDIA's cuDF does not fuse expressions with sort, groupby, or reductions -- each is a separate kernel. MetalDF fuses across these operator boundaries.

## Supported Operations

| Category | Operations | DTypes |
|----------|------------|--------|
| Arithmetic | `+`, `-`, `*`, `/` (fused via lazy eval) | float32, int32, int64 |
| Reductions | `sum`, `min`, `max`, `mean` | float32, int32, int64 |
| Sort | `sort_values`, `argsort` | float32, int32, int64 |
| GroupBy | `sum`, `mean`, `min`, `max`, `count` | float32, int32 |
| String Search | `str.contains`, `str.startswith`, `str.endswith`, `str.find` | object (string) |
| String Transform | `str.lower`, `str.upper`, `str.strip`, `str.replace` | object (string) |
| String Sort | `sort` (via direct API) | object (string) |
| String GroupBy | `sum`, `min`, `max`, `mean`, `count` (via direct API) | object keys, float32/int32 values |

## Benchmarks

Apple M4 Pro, 10M elements. All 73 benchmarks verified correct against pandas. Reproduce with `python benchmarks/run.py`.

### String Operations (59-100x faster)

| Operation | Metal | Pandas | Speedup |
|-----------|-------|--------|---------|
| `str.find` | 7.4ms | 740ms | **100x** |
| `str.contains` | 7.6ms | 746ms | **99x** |
| `str.endswith` | 7.3ms | 436ms | **59x** |
| `str.startswith` | 7.4ms | 436ms | **59x** |
| `str.replace` | 531ms | 803ms | **1.5x** |

### GroupBy — High Cardinality (10-15x faster)

| Operation | Metal | Pandas | Speedup |
|-----------|-------|--------|---------|
| groupby max float32 | 101ms | 1529ms | **15x** |
| groupby sum float32 | 102ms | 1548ms | **15x** |
| groupby min float32 | 102ms | 1534ms | **15x** |
| groupby sum int32 | 95ms | 1282ms | **14x** |
| groupby min int32 | 93ms | 1251ms | **13x** |

### Sort (4-12x faster)

| Operation | Metal | Pandas | Speedup |
|-----------|-------|--------|---------|
| argsort float32 | 79ms | 913ms | **12x** |
| sort float32 | 82ms | 916ms | **11x** |
| `(df["a"] + df["b"]).sort_values()` | 89ms | 967ms | **11x** |
| sort int32 | 74ms | 619ms | **8x** |
| sort int64 | 141ms | 664ms | **5x** |
| string sort | 986ms | 4282ms | **4x** |

### Chained Expressions (1.3-2.7x faster)

| Code | Metal | Pandas | Speedup |
|------|-------|--------|---------|
| `df["a"] + df["b"]*df["c"] - df["d"]/df["e"] + df["f"]*df["g"] - df["h"]` | 2.8ms | 7.3ms | **2.7x** |
| `(df["a"] + df["b"]*df["c"] - df["d"]/df["e"] + df["f"]*df["g"] - df["h"]).sum()` | 3.7ms | 9.2ms | **2.5x** |
| `((df["a"] + df["b"]) * df["c"] - df["d"]).sum()` | 2.4ms | 5.0ms | **2.1x** |
| `(df["a"] + df["b"]*df["c"] - df["d"]/df["e"]).sum()` | 3.1ms | 6.1ms | **2.0x** |
| `df["a"] + df["b"]*df["c"] - df["d"]/df["e"]` | 2.9ms | 4.2ms | **1.5x** |
| `(df["a"] + df["b"]) * df["c"] - df["d"]` | 2.3ms | 3.1ms | **1.3x** |

### Reductions (1.1-2.6x faster)

| Operation | Metal | Pandas | Speedup |
|-----------|-------|--------|---------|
| mean float32 | 1.2ms | 3.2ms | **2.6x** |
| mean int64 | 1.3ms | 3.4ms | **2.6x** |
| mean int32 | 1.4ms | 3.4ms | **2.3x** |
| max float32 | 1.4ms | 2.5ms | **1.8x** |
| min float32 | 1.4ms | 2.5ms | **1.8x** |

## Requirements

- macOS on Apple Silicon
- Python 3.10+
- Rust toolchain

## Building

```bash
pip install -e .
```

## Development

### Prerequisites
- macOS on Apple Silicon
- Python 3.10+
- Rust toolchain (rustup)

### Setup
```bash
git clone <repo>
cd metaldf
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"   # or: pip install maturin numpy pandas pytest
```

### Building
```bash
# Development build (debug)
maturin develop

# Release build (optimized -- use this for benchmarks)
maturin develop --release
```

### Testing
```bash
# Run all tests
python -m pytest tests/ -x -q

# Run specific test file
python -m pytest tests/test_string_compare.py -v

# Run Rust tests
cargo test --manifest-path rust/Cargo.toml
```

### Running Benchmarks
```bash
# Full benchmark suite (5M elements, all operations)
python benchmarks/run.py

# Write results to JSON
python benchmarks/run.py --json results.json
```
