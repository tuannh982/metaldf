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
                    dispatch: try Metal, fallback pandas
                              |
+------------------------------------------------------------------+
|  Proxy & Dispatch                                                 |
|  ProxyDataFrame, ProxySeries wrap Metal + pandas implementations  |
|  Every operation: try Metal path -> if fails -> pandas path       |
+------------------------------------------------------------------+
                              |
                    compute operations
                              |
+------------------+-------------------+
|  PandasEngine    |  MetalEngine      |
|  numpy/pandas    |  Rust + metal-rs  |
|  CPU fallback    |  GPU kernels      |
+------------------+-------------------+
```

The proxy layer is invisible to user code. `isinstance(df, pd.DataFrame)` returns `True`. Operations that have Metal kernels run on the GPU; everything else silently falls back to pandas.

## Supported Operations

| Category | Operations | DTypes |
|----------|------------|--------|
| Reductions | `sum`, `min`, `max`, `mean` | float32, int32, int64 |
| Sort | `sort_values`, `argsort` | float32, int32, int64 |
| GroupBy | `sum`, `mean`, `min`, `max`, `count` | float32, int32 |
| String Search | `str.contains`, `str.startswith`, `str.endswith`, `str.find` | object (string) |
| String Transform | `str.lower`, `str.upper`, `str.strip`, `str.replace` | object (string) |
| String Sort | `sort` (via direct API) | object (string) |
| String GroupBy | `sum`, `min`, `max`, `mean`, `count` (via direct API) | object keys, float32/int32 values |

## Benchmarks

Apple M4 Pro, 5M elements. All 52 benchmarks verified correct against pandas. Reproduce with `python benchmarks/run.py`.

| Operation | Metal | Pandas | Speedup |
|-----------|-------|--------|---------|
| `str.contains` | 4.0ms | 388ms | **96x** |
| `str.find` | 4.1ms | 370ms | **90x** |
| `str.startswith` | 3.7ms | 219ms | **59x** |
| `str.endswith` | 4.4ms | 220ms | **51x** |
| groupby sum float32 (high card) | 55ms | 699ms | **13x** |
| groupby min int32 (high card) | 59ms | 739ms | **12x** |
| groupby min float32 (high card) | 55ms | 670ms | **12x** |
| groupby max float32 (high card) | 56ms | 666ms | **12x** |
| groupby sum int32 (high card) | 50ms | 585ms | **12x** |
| groupby max int32 (high card) | 51ms | 581ms | **11x** |
| groupby count float32 (high card) | 56ms | 611ms | **11x** |
| groupby mean float32 (high card) | 64ms | 690ms | **11x** |
| groupby count int32 (high card) | 51ms | 517ms | **10x** |
| groupby mean int32 (high card) | 60ms | 592ms | **10x** |
| sort float32 | 42ms | 409ms | **10x** |
| argsort float32 | 43ms | 376ms | **9x** |
| sort int32 | 39ms | 273ms | **7x** |
| argsort int32 | 37ms | 244ms | **7x** |
| string sort | 410ms | 2106ms | **5x** |
| sort int64 | 79ms | 306ms | **4x** |
| argsort int64 | 76ms | 272ms | **4x** |
| mean float32 | 0.7ms | 1.6ms | **2.4x** |
| mean int32 | 0.7ms | 1.6ms | **2.3x** |
| min float32 | 0.7ms | 1.2ms | **1.8x** |
| mean int64 | 1.0ms | 1.7ms | **1.8x** |
| max float32 | 0.7ms | 1.2ms | **1.7x** |
| `str.replace` | 274ms | 406ms | **1.5x** |
| sum float32 | 0.7ms | 0.8ms | **1.2x** |
| `str.upper` | 248ms | 289ms | **1.2x** |

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
