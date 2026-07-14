# MetalDF

GPU-accelerated pandas on Apple Silicon. Drop-in replacement -- no code changes required.

```bash
python -m metaldf my_script.py
```

MetalDF intercepts `import pandas` at runtime and returns proxy objects that dispatch operations to Metal compute shaders on the GPU. Arithmetic, comparisons, reductions, sorting, groupby, joins, rolling windows, datetime extraction, unary math, boolean indexing, and string operations are all accelerated transparently. Null/NaN values propagate correctly through all operations. Unsupported operations fall back to pandas with identical results.

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

**Multi-column fusion**: When assigning multiple computed columns to a DataFrame, MetalDF queues the deferred expressions and flushes them as a single GPU kernel with multiple outputs, reading shared input columns only once:

```python
df["z"] = df["a"] + df["b"]   # queued
df["w"] = df["a"] * df["c"]   # queued
print(df)                      # flushes: 1 kernel, reads "a" once, writes "z" and "w"
```

## Supported Operations

| Category | Operations | DTypes |
|----------|------------|--------|
| Arithmetic | `+`, `-`, `*`, `/` (fused via lazy eval) | float32, int32, int64 |
| Comparisons | `==`, `!=`, `<`, `<=`, `>`, `>=` | float32, int32, int64, datetime64, timedelta64 |
| Reductions | `sum`, `min`, `max`, `mean` | float32, int32, int64 (null-aware, skips NaN) |
| Sort | `sort_values`, `argsort` | float32, int32, int64, datetime64, timedelta64 (null-aware, nulls last) |
| Boolean Indexing | `df[df["col"] > 0]`, `series[mask]` | all numeric dtypes |
| GroupBy | `sum`, `mean`, `min`, `max`, `count` | float32, int32 (null-aware, skips null keys/values) |
| Joins | `df.merge(other, on="key")` (inner, left, right) | float32, int32 keys |
| Rolling Windows | `.rolling(window).sum/mean/min/max/count()` | float32 |
| Unary Math | `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `log2`, `log10`, `round`, `trunc`, `cbrt` | float32 (fused in expressions) |
| Logical Ops | `AND`, `OR`, `NOT` on bool columns | bool (uint8) |
| Datetime | `.dt.year`, `.dt.month`, `.dt.day`, `.dt.hour`, `.dt.minute`, `.dt.second`, `.dt.dayofweek` | datetime64[ns] |
| Datetime Arithmetic | `datetime - datetime = timedelta`, `datetime + timedelta = datetime` | datetime64, timedelta64 |
| String Search | `str.contains`, `str.startswith`, `str.endswith`, `str.find` | object (string) |
| String Transform | `str.lower`, `str.upper`, `str.strip`, `str.replace` | object (string) |
| String Sort | `sort` (via direct API) | object (string) |
| String GroupBy | `sum`, `min`, `max`, `mean`, `count` (via direct API) | object keys, float32/int32 values |
| Multi-Column Fusion | `df["z"] = df["a"] + df["b"]; df["w"] = df["a"] * df["c"]` → 1 kernel | float32 |
| Null Handling | NaN detection, null-mask propagation through all ops | float32 |
| NumPy Ufuncs | `np.sin(series)`, `np.cos(series)`, etc. intercepted → GPU | float32 |

## Benchmarks

Apple M4 Pro, 10M elements. All 73 benchmarks verified correct against pandas. Reproduce with `python benchmarks/run.py`.

### Elementwise Operations (87-257x faster)

| Operation | Metal | Pandas | Speedup |
|-----------|-------|--------|---------|
| `(a+b)*c-d` chained expression | 0.01ms | 3.3ms | **257x** |
| `a - b` float32 | 0.01ms | 1.0ms | **130x** |
| `a + b` float32 | 0.01ms | 1.1ms | **115x** |
| `a * b` float32 | 0.01ms | 1.0ms | **87x** |

### String Operations (58-104x faster)

| Operation | Metal | Pandas | Speedup |
|-----------|-------|--------|---------|
| `str.contains` | 7.6ms | 785ms | **104x** |
| `str.find` | 7.6ms | 769ms | **101x** |
| `str.endswith` | 7.7ms | 450ms | **58x** |
| `str.startswith` | 7.8ms | 452ms | **58x** |
| string sort | 1173ms | 4650ms | **4x** |

### Chained Expressions (1.2-2.6x faster)

| Code | Metal | Pandas | Speedup |
|------|-------|--------|---------|
| 8-op fused codegen | 3.0ms | 7.8ms | **2.6x** |
| `sum(8-op fused)` | 3.9ms | 9.6ms | **2.4x** |
| `sum((a+b)*c-d)` fused | 2.5ms | 5.3ms | **2.1x** |
| 5-op fused codegen | 2.6ms | 4.5ms | **1.7x** |
| codegen cached `(a+b)*c-d` | 2.3ms | 3.4ms | **1.5x** |
| `(a+b)*c-d` 20M elements | 4.6ms | 6.7ms | **1.5x** |

### Reductions (1.2-2.3x faster)

| Operation | Metal | Pandas | Speedup |
|-----------|-------|--------|---------|
| mean int32 | 1.6ms | 3.7ms | **2.3x** |
| mean int64 | 1.7ms | 3.7ms | **2.3x** |
| mean float32 | 2.4ms | 3.4ms | **1.4x** |
| max float32 | 2.0ms | 2.8ms | **1.4x** |
| min float32 | 2.3ms | 2.8ms | **1.2x** |

### GroupBy — High Cardinality (1.4-1.8x faster)

| Operation | Metal | Pandas | Speedup |
|-----------|-------|--------|---------|
| groupby sum float32 | 968ms | 1708ms | **1.8x** |
| groupby min float32 | 967ms | 1693ms | **1.8x** |
| groupby max float32 | 970ms | 1689ms | **1.7x** |
| groupby sum int32 | 875ms | 1421ms | **1.6x** |
| groupby count float32 | 968ms | 1573ms | **1.6x** |

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
