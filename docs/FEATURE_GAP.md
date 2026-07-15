# MetalDF Feature Gap Analysis

Gap analysis against **pandas** API coverage and Metal GPU architecture best practices.

Last updated: 2026-07-15

---

## 1. Operations Coverage

| Category | metaldf | pandas | Status |
|----------|---------|--------|--------|
| Elementwise arithmetic (add/sub/mul/div/mod) | f32/i32/i64 | Full (+ pow, floordiv, reverse ops) | Partial |
| Comparisons (eq/ne/lt/le/gt/ge) | f32/i32/i64/datetime/timedelta | Full (+ isin, between, where, mask) | Partial |
| Reductions: sum/min/max/mean | f32/i32/i64 | Full | Done |
| Reductions: std/var | None | Full | Missing |
| Reductions: median | None | Full | Missing |
| Reductions: prod | None | Full | Missing |
| Reductions: quantile/mode/nunique | None | Full | Missing |
| Cumulative: cumsum | Internal only (uint32/int32) | Full | Missing (public) |
| Cumulative: cumprod/cummin/cummax | None | Full | Missing |
| Sorting: sort_values/argsort | f32/i32/i64/datetime/timedelta | Full | Done |
| Sorting: rank | None | Full | Missing |
| Sorting: nlargest/nsmallest | None | Full | Missing |
| GroupBy: sum/mean/min/max/count | f32-f32 or i32-i32 only | Full (all types) | Partial |
| GroupBy: std/var/median | None | Full | Missing |
| GroupBy: nunique/first/last/nth | None | Full | Missing |
| GroupBy: mixed key/value dtypes | None | Full | Missing |
| GroupBy: int64 keys | None (no 64-bit Metal atomics) | Full | Missing |
| Joins: inner/left/right | f32/i32 keys only | Full | Partial |
| Joins: outer | None | Full | Missing |
| Joins: cross | None | Full | Missing |
| Rolling: sum/mean/min/max/count | f32 only | Full | Partial |
| Rolling: std/var/median | None | Full | Missing |
| Rolling: NaN-aware skip | None | Full | Missing |
| Expanding window | None | Full | Missing |
| EWM (exponentially weighted) | None | Full | Missing |
| String: contains/startswith/endswith/find | Literal only | Full (+ regex) | Partial |
| String: lower/upper/strip/replace | Literal replace only | Full (+ regex replace) | Partial |
| String: extract/split/pad/center | None | Full | Missing |
| String: len/count/repeat/slice | None | Full | Missing |
| String: isalpha/isdigit/isspace/etc | None | Full | Missing |
| Datetime: year/month/day/hour/min/sec/dow | Done | Full | Done |
| Datetime: quarter/dayofyear | None | Full | Missing |
| Datetime: is_month_start/end/leap_year | None | Full | Missing |
| Datetime: tz_localize/tz_convert | None | Full | Missing |
| Datetime: strftime/round/floor/ceil | None | Full | Missing |
| Datetime arithmetic (+/- timedelta) | Done | Full | Done |
| Missing data: fillna | None | Full | Missing |
| Missing data: ffill/bfill | None | Full | Missing |
| Missing data: dropna | None | Full | Missing |
| Missing data: interpolate | None | Full | Missing |
| Reshaping: pivot/pivot_table | None | Full | Missing |
| Reshaping: melt (unpivot) | None | Full | Missing |
| Reshaping: stack/unstack | None | Full | Missing |
| Reshaping: explode | None | Full | Missing |
| Reshaping: get_dummies | None | Full | Missing |
| Reshaping: crosstab | None | Full | Missing |
| Indexing: loc/iloc | None (positional only) | Full | Missing |
| Indexing: query/eval | None | Full | Missing |
| Boolean indexing | Stream compaction | Full | Done |
| Statistical: describe | None | Full | Missing |
| Statistical: corr/cov | None | Full | Missing |
| Statistical: skew/kurt | None | Full | Missing |
| Statistical: value_counts | None | Full | Missing |
| Duplicate handling: drop_duplicates | None | Full | Missing |
| Duplicate handling: duplicated | None | Full | Missing |
| Shift/diff/pct_change | None | Full | Missing |
| UDF (user-defined functions) | None | apply() | Missing |
| IO: CSV/Parquet (GPU-accelerated) | None (pandas fallback) | CPU | Missing |
| Unary math: abs/neg/sqrt/exp/log/ceil/floor | Done | Full | Done |
| Unary math: trig (sin/cos/tan/asin/acos/atan) | Done | Full | Done |
| Unary math: hyperbolic (sinh/cosh/tanh) | Done | Full | Done |
| Unary math: log2/log10/round/trunc/cbrt | Done | Full | Done |
| Logical ops: AND/OR/NOT | Done | Full | Done |
| Kernel fusion (lazy eval + codegen) | f32 arithmetic only | N/A | Partial |
| Fused expression-reduce | Done (sum/min/max) | N/A | Done |

## 2. Data Type Coverage

| Type | metaldf GPU | pandas | Status |
|------|-------------|--------|--------|
| float32 | Full | Full | Done |
| float64 | Storage only, no GPU ops | Full | Missing |
| int8/int16 | None | Full | Missing |
| int32 | Partial (some ops) | Full | Partial |
| int64 | Partial (no atomics) | Full | Partial |
| uint8 | String char storage only | Full | Missing |
| uint32/uint64 | Internal only | Full | Missing |
| bool | Basic | Full | Partial |
| datetime64 | Comparisons + dt accessor | Full | Partial |
| timedelta64 | Comparisons + arithmetic | Full | Partial |
| string (Utf8) | Offsets+chars GPU | Full | Partial |
| categorical | None | Full | Missing |
| Nullable integers | None (NaN-as-null f32 only) | Full (pd.Int32Dtype etc) | Missing |

## 3. GPU/Metal Architecture Gaps

| Aspect | metaldf | Best Practice | Status |
|--------|---------|---------------|--------|
| Memory pooling / buffer cache | None (raw MTLBuffer alloc) | Best-fit LRU cache with page-aligned alloc | Missing |
| Buffer donation (reuse input for output) | None | Refcount-1 buffer reuse for in-place ops | Missing |
| Graph-level lazy eval | Expression trees (f32 arith only) | Full subgraph fusion across all elementwise ops | Partial |
| Command buffer batching | One dispatch per op (+ BatchContext) | Auto-commit at op/size thresholds per GPU class | Missing |
| Non-contiguous/strided array support | Falls back to pandas | ndim-specialized strided kernels | Missing |
| Large array support (>2B elements) | uint32 indexing | Auto int64 index switching at threshold | Missing |
| Memory pressure handling / GC | None | Configurable limits with eval backpressure | Missing |
| Automatic barrier management | Manual per-op | Input/output set tracking with auto barriers | Missing |
| Per-architecture tuning | apple3-9 threadgroup sizes | Per-GPU-class op/memory budgets | Similar |
| Pre-compiled metallib (AOT) | Optional via xcrun | AOT default + JIT fallback | Similar |

## 4. Prioritized Improvement Proposals

### Tier 1: High Impact, Foundational

| ID | Proposal | Effort | Impact | Deps |
|----|----------|--------|--------|------|
| P1 | Proper nullable type system (bitmask for all dtypes) | Large | Critical | None |
| P2 | Buffer pool / memory cache | Medium | High | None |
| P3 | Broader reductions (std, var, median, prod) | Medium | High | None |
| P4 | Public cumulative ops (cumsum, cummin, cummax) | Small | High | None |

### Tier 2: High Impact, Moderate Effort

| ID | Proposal | Effort | Impact | Deps |
|----|----------|--------|--------|------|
| P5 | GroupBy: more aggs + wider dtype support | Medium | High | P1 helps |
| P6 | Graph-level lazy eval (all dtypes, all elementwise) | Large | High | None |
| P7 | String regex support (Thompson NFA on GPU) | Large | High | None |
| P8 | Outer join | Small | Medium | None |

### Tier 3: Medium Impact, Enables Ecosystem

| ID | Proposal | Effort | Impact | Deps |
|----|----------|--------|--------|------|
| P9 | fillna / ffill / bfill | Small | Medium | P1 helps |
| P10 | drop_duplicates / value_counts | Small | Medium | None |
| P11 | shift / diff / pct_change | Small | Medium | None |
| P12 | Non-contiguous array support (strided kernels) | Large | Medium | None |
| P13 | GPU-accelerated Parquet IO | Large | Medium | None |

### Tier 4: Nice to Have

| ID | Proposal | Effort | Impact | Deps |
|----|----------|--------|--------|------|
| P14 | Reshaping ops (pivot, melt, stack, unstack) | Medium | Low | None |
| P15 | loc/iloc indexing | Medium | Low | None |
| P16 | describe() | Small | Low | P3 |
| P17 | corr/cov | Medium | Low | P3 |
| P18 | EWM / expanding windows | Medium | Low | None |
| P19 | String: len/count/split/extract/pad | Medium | Medium | None |
| P20 | Datetime: quarter/dayofyear/strftime/round | Small | Low | None |

## 5. Tracking Notes

Use this section to record progress per commit:

| Date | Commit | Items Addressed | Notes |
|------|--------|-----------------|-------|
| | | | |
