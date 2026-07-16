# Low-Hanging Fruit: Easy Improvements Batch

**Date:** 2026-07-16
**FEATURE_GAP IDs:** P4 (partial), P9, P11, P20 (partial)
**Branch:** `feat/low-hanging-fruit`

## Scope

Six feature groups from FEATURE_GAP.md that require minimal new infrastructure because they build on existing patterns (scan, datetime extract, elementwise ops):

1. **Cumulative ops** — cumsum, cummin, cummax (public API for existing internal scan)
2. **Datetime accessors** — quarter, dayofyear
3. **Shift / diff / pct_change**
4. **fillna** (scalar)
5. **ffill / bfill** (forward/backward fill)

## Architecture Approach: Hybrid

- **Op-generic scan template** for cumsum/cummin/cummax — refactor `scan.metal` to accept Op structs (reusing `SumOp`/`MinOp`/`MaxOp` from `02_reduce_ops.h`).
- **Separate kernels** for ffill/bfill — their NaN-conditional propagation semantics don't fit the numeric Op abstraction cleanly.
- **Standalone kernels** for shift and fillna — simple elementwise operations.
- **No new kernels** for diff and pct_change — composed from shift + existing arithmetic.

---

## Feature 1: Cumulative Ops (cumsum, cummin, cummax)

### Metal Layer

**Modified: `rust/metal/scan/scan.metal`**

Refactor `scan_inclusive_impl` and `scan_propagate_impl` to be op-generic:

```metal
template <typename T, typename Op>
void scan_inclusive_impl(...) {
    // Out-of-bounds lanes use Op::identity (not T(0))
    shared[tid] = (base < len) ? input[base] : Op::identity;
    // ...
    // Hillis-Steele: replace `shared[tid - offset] + shared[tid]`
    // with `Op::apply(shared[tid - offset], shared[tid])`
    for (uint offset = 1; offset < group_size; offset *= 2) {
        T val = shared[tid];
        if (tid >= offset) {
            val = Op::apply(shared[tid - offset], shared[tid]);
        }
        // ...
    }
}

template <typename T, typename Op>
void scan_propagate_impl(...) {
    // Replace `data[base] += partials[group_id - 1]`
    // with `data[base] = Op::apply(partials[group_id - 1], data[base])`
}
```

The `Op` structs come from `02_reduce_ops.h` (already defined: `SumOp<T>`, `MinOp<T>`, `MaxOp<T>`).

Instantiate via macro:

```metal
#define INSTANTIATE_SCAN_OP(T, Op, suffix) \
    [[kernel]] void scan_inclusive_##suffix(...) { \
        scan_inclusive_impl<T, Op<T>>(...); \
    } \
    [[kernel]] void scan_propagate_##suffix(...) { \
        scan_propagate_impl<T, Op<T>>(...); \
    }

// cumsum
INSTANTIATE_SCAN_OP(float, SumOp, sum_float32)
INSTANTIATE_SCAN_OP(int,   SumOp, sum_int32)
INSTANTIATE_SCAN_OP(long,  SumOp, sum_int64)

// cummin
INSTANTIATE_SCAN_OP(float, MinOp, min_float32)
INSTANTIATE_SCAN_OP(int,   MinOp, min_int32)
INSTANTIATE_SCAN_OP(long,  MinOp, min_int64)

// cummax
INSTANTIATE_SCAN_OP(float, MaxOp, max_float32)
INSTANTIATE_SCAN_OP(int,   MaxOp, max_int32)
INSTANTIATE_SCAN_OP(long,  MaxOp, max_int64)

// backward compat: filter.rs uses these names
INSTANTIATE_SCAN_OP(uint,  SumOp, sum_uint32)
```

**Backward compatibility:** The old kernel names (`scan_inclusive_uint32`, `scan_inclusive_int32`) become `scan_inclusive_sum_uint32`, `scan_inclusive_sum_int32`. Update `filter.rs` and `scan.rs` references to use the new names.

### Rust Layer

**Modified: `rust/src/kernels/scan.rs`**

- Extend `check_scan_dtype` → rename to `check_cumulative_dtype`, accept Float32, Int32, Int64.
- Generalize `prefix_sum_inclusive` → `cumulative_scan(input, len, dtype, op: &str)` where `op` is `"sum"`, `"min"`, or `"max"`. The kernel name becomes `scan_inclusive_{op}_{suffix}` / `scan_propagate_{op}_{suffix}`.
- Add pyfunctions:
  - `metal_cumsum(input: &MetalSeries) -> MetalSeries` — calls `cumulative_scan(..., "sum")`
  - `metal_cummin(input: &MetalSeries) -> MetalSeries` — calls `cumulative_scan(..., "min")`
  - `metal_cummax(input: &MetalSeries) -> MetalSeries` — calls `cumulative_scan(..., "max")`
- Keep `metal_prefix_sum` as a thin wrapper around `metal_cumsum` for backward compat with any internal callers.

**Modified: `rust/src/kernels/filter.rs`**

Update the kernel names used for prefix sum from `scan_inclusive_uint32` → `scan_inclusive_sum_uint32` (and propagate variant).

**Modified: `rust/src/lib.rs`**

Register `metal_cumsum`, `metal_cummin`, `metal_cummax`.

### Python Layer

**Modified: `src/metaldf/_engine/_metal.py`**

Add static methods: `MetalEngine.metal_cumsum`, `.metal_cummin`, `.metal_cummax`. Pattern: extract numpy array → build MetalSeries → call `metaldf_engine.metal_cum*` → wrap result as ProxySeries.

**Modified: `src/metaldf/_engine/__init__.py`**

Register: `"cumsum"`, `"cummin"`, `"cummax"`.

**Modified: `src/metaldf/_wrappers.py`**

Add to `ProxySeries`:

```python
def cumsum(self, *args, **kwargs):
    return self._try_metal_series_op("cumsum", *args, **kwargs)

def cummin(self, *args, **kwargs):
    return self._try_metal_series_op("cummin", *args, **kwargs)

def cummax(self, *args, **kwargs):
    return self._try_metal_series_op("cummax", *args, **kwargs)
```

Where `_try_metal_series_op` is a new helper similar to `_try_metal_reduction` but returns a Series instead of a scalar.

**Modified: `src/metaldf/_deferred.py`**

Add `DeferredSeries.cumsum()`, `.cummin()`, `.cummax()` — these materialize the deferred expression first (scan is not fusible with elementwise), then dispatch the cumulative op.

### Supported dtypes

| Op | float32 | int32 | int64 | datetime | timedelta |
|----|---------|-------|-------|----------|-----------|
| cumsum | Yes | Yes | Yes | No | Yes (via int64) |
| cummin | Yes | Yes | Yes | Yes (via int64) | Yes (via int64) |
| cummax | Yes | Yes | Yes | Yes (via int64) | Yes (via int64) |

---

## Feature 2: Datetime quarter & dayofyear

### Metal Layer

**Modified: `rust/metal/datetime/01_calendar.h`**

Add `days_from_civil` (Hinnant's inverse algorithm):

```metal
inline long days_from_civil(int y, int m, int d) {
    y -= (m <= 2);
    long era = (y >= 0 ? y : y - 399) / 400;
    uint yoe = uint(y - era * 400);
    uint doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
    uint doe = yoe * 365 + yoe/4 - yoe/100 + doy;
    return era * 146097 + long(doe) - 719468;
}
```

**Modified: `rust/metal/datetime/extract.metal`**

Add two new kernels:

```metal
kernel void dt_quarter_i64(device const long* ns, device int* out,
                           device const uint* len_ptr, uint idx) {
    if (idx >= *len_ptr) return;
    long days = floor_div(ns[idx], NS_PER_DAY);
    int month = civil_from_days(days).month;
    out[idx] = (month - 1) / 3 + 1;
}

kernel void dt_dayofyear_i64(device const long* ns, device int* out,
                              device const uint* len_ptr, uint idx) {
    if (idx >= *len_ptr) return;
    long days = floor_div(ns[idx], NS_PER_DAY);
    CivilDate c = civil_from_days(days);
    long jan1 = days_from_civil(c.year, 1, 1);
    out[idx] = int(days - jan1) + 1;
}
```

### Rust Layer

**Modified: `rust/src/kernels/datetime.rs`**

Add two pyfunctions (same pattern as existing 7):

```rust
#[pyfunction]
pub fn metal_dt_quarter(data: &MetalSeries) -> PyResult<MetalSeries> {
    dispatch_dt_extract("dt_quarter_i64", data)
}

#[pyfunction]
pub fn metal_dt_dayofyear(data: &MetalSeries) -> PyResult<MetalSeries> {
    dispatch_dt_extract("dt_dayofyear_i64", data)
}
```

**Modified: `rust/src/lib.rs`** — register both.

### Python Layer

**Modified: `src/metaldf/_wrappers.py`**

Add to `ProxyDatetimeAccessor`:

```python
@property
def quarter(self): return self._dispatch("quarter")

@property
def dayofyear(self): return self._dispatch("dayofyear")
```

No engine registry changes needed — datetime accessors bypass the engine and call `metaldf_engine` directly.

---

## Feature 3: shift / diff / pct_change

### Metal Layer

**New: `rust/metal/elementwise/shift.metal`**

```metal
template <typename T>
void shift_impl(device const T* input, device T* output,
                device const int* periods_ptr, device const uint* len_ptr,
                uint idx) {
    uint len = *len_ptr;
    if (idx >= len) return;
    int p = *periods_ptr;
    int src = int(idx) - p;
    if (src >= 0 && src < int(len)) {
        output[idx] = input[src];
    } else {
        output[idx] = /* NaN for float, 0 for int */;
    }
}
```

Instantiate for float32 (NaN fill), int32 (0 fill), int64 (0 fill).

For float32, out-of-bounds fills with NaN (`as_type<float>(0x7FC00000u)`).
For int types, fills with 0 (matching pandas behavior for int shift).

### Rust Layer

**New: `rust/src/kernels/shift.rs`**

- `metal_shift(input: &MetalSeries, periods: i32) -> MetalSeries`
- Validates dtype (Float32, Int32, Int64).
- Dispatches `shift_<suffix>` kernel with periods as a buffer parameter.

**Modified: `rust/src/kernels/mod.rs`** — add `pub mod shift;` and `load_elementwise_library` (already exists, shift.metal gets compiled into the elementwise library).

**Modified: `rust/src/lib.rs`** — register `metal_shift`.

### Python Layer

**Modified: `src/metaldf/_engine/_metal.py`**

```python
@staticmethod
def metal_shift(series, periods=1):
    # extract array → MetalSeries → metaldf_engine.metal_shift → wrap
```

**Modified: `src/metaldf/_engine/__init__.py`** — register `"shift"`.

**Modified: `src/metaldf/_wrappers.py`**

```python
def shift(self, periods=1, **kwargs):
    # Try Metal for supported dtypes, fall back to pandas
    return self._try_metal_series_op("shift", periods=periods)

def diff(self, periods=1, **kwargs):
    try:
        shifted = self.shift(periods)
        return self - shifted  # uses existing GPU arithmetic
    except Exception:
        return _wrap_result(pd.Series.diff(self._pandas_obj, periods=periods))

def pct_change(self, periods=1, **kwargs):
    try:
        shifted = self.shift(periods)
        return (self - shifted) / shifted  # GPU arithmetic
    except Exception:
        return _wrap_result(pd.Series.pct_change(self._pandas_obj, periods=periods))
```

Note: `diff` and `pct_change` compose existing GPU ops — no new Metal kernels, no engine registration.

---

## Feature 4: fillna (scalar)

### Metal Layer

**New: `rust/metal/elementwise/fillna.metal`**

```metal
kernel void fillna_f32(device const float* input, device float* output,
                       device const float* fill_ptr, device const uint* len_ptr,
                       uint idx) {
    if (idx >= *len_ptr) return;
    output[idx] = isnan(input[idx]) ? *fill_ptr : input[idx];
}
```

For null-mask aware variant (all dtypes):

```metal
kernel void fillna_masked_<dtype>(device const T* input, device T* output,
                                   device const uint* mask_in,
                                   device atomic_uint* mask_out,
                                   device const T* fill_ptr, device const uint* len_ptr,
                                   uint idx) {
    if (idx >= *len_ptr) return;
    uint word = mask_in[idx / 32];
    bool is_null = !((word >> (idx % 32)) & 1u);
    output[idx] = is_null ? *fill_ptr : input[idx];
    // Set validity bit for filled positions (atomic — multiple threads
    // may update different bits in the same word concurrently)
    if (is_null) {
        atomic_fetch_or_explicit(&mask_out[idx / 32],
                                 1u << (idx % 32),
                                 memory_order_relaxed);
    }
}
```

Instantiate: float32 (NaN-based), int32 + int64 (mask-based only).

### Rust Layer

**New: `rust/src/kernels/fillna.rs`**

- `metal_fillna(input: &MetalSeries, fill_value: f64) -> MetalSeries`
- For Float32 without null mask: uses NaN-check kernel.
- For series with null mask: uses masked kernel variant.
- Casts fill_value to the target dtype.

**Modified: `rust/src/lib.rs`** — register `metal_fillna`.

### Python Layer

**Modified: `src/metaldf/_engine/_metal.py`** — `MetalEngine.metal_fillna(series, value)`.

**Modified: `src/metaldf/_engine/__init__.py`** — register `"fillna"`.

**Modified: `src/metaldf/_wrappers.py`**

```python
def fillna(self, value=None, method=None, **kwargs):
    if value is not None and np.isscalar(value) and method is None:
        return self._try_metal_series_op("fillna", value=value)
    # Non-scalar or method-based fill: fall back to pandas
    return _wrap_result(pd.Series.fillna(self._pandas_obj, value=value, method=method, **kwargs))
```

---

## Feature 5: ffill / bfill

### Metal Layer

**New: `rust/metal/scan/fill_scan.metal`**

Forward fill as a parallel scan with NaN-conditional propagation:

```metal
template <typename T>
void ffill_scan_impl(device const T* input, device T* output,
                     device T* partials, threadgroup T* shared,
                     uint tid, uint group_id, uint group_size,
                     device const uint* len_ptr) {
    uint len = *len_ptr;
    uint base = group_id * group_size + tid;
    
    // Identity is NaN (propagation should skip NaN, keep prior valid)
    shared[tid] = (base < len) ? input[base] : NAN;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Hillis-Steele with ffill-combine: take right if valid, else left
    for (uint offset = 1; offset < group_size; offset *= 2) {
        T val = shared[tid];
        if (tid >= offset) {
            T left = shared[tid - offset];
            val = isnan(shared[tid]) ? left : shared[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[tid] = val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (base < len) output[base] = shared[tid];
    if (tid == group_size - 1) partials[group_id] = shared[tid];
}

template <typename T>
void ffill_propagate_impl(device T* data, device const T* partials,
                          uint tid, uint group_id, uint group_size,
                          device const uint* len_ptr) {
    if (group_id == 0) return;
    uint len = *len_ptr;
    uint base = group_id * group_size + tid;
    if (base < len && isnan(data[base])) {
        // Propagate must also check: only fill if partial is valid
        T prev = partials[group_id - 1];
        if (!isnan(prev)) data[base] = prev;
    }
}
```

Instantiate for float32 only (int types need null-mask-based fill, deferred to later).

**bfill:** Same structure but reverses the read/write indices:
- `base = len - 1 - (group_id * group_size + tid)`
- Scan direction changes: `shared[tid + offset]` instead of `shared[tid - offset]`

### Rust Layer

**New: `rust/src/kernels/fill_scan.rs`**

- `metal_ffill(input: &MetalSeries) -> MetalSeries` — two-pass dispatch (same recursive pattern as `prefix_sum_inclusive`).
- `metal_bfill(input: &MetalSeries) -> MetalSeries` — same but dispatches bfill kernels.
- Both validate dtype == Float32.

**Modified: `rust/src/lib.rs`** — register both.

### Python Layer

**Modified: `src/metaldf/_engine/_metal.py`** — `MetalEngine.metal_ffill`, `MetalEngine.metal_bfill`.

**Modified: `src/metaldf/_engine/__init__.py`** — register `"ffill"`, `"bfill"`.

**Modified: `src/metaldf/_wrappers.py`**

```python
def ffill(self, **kwargs):
    return self._try_metal_series_op("ffill")

def bfill(self, **kwargs):
    return self._try_metal_series_op("bfill")

# pandas aliases
def pad(self, **kwargs):
    return self.ffill(**kwargs)

def backfill(self, **kwargs):
    return self.bfill(**kwargs)
```

---

## Testing Plan

Each feature gets its own test file in `tests/`:

| Test file | What it covers |
|-----------|----------------|
| `test_cumulative.py` | cumsum/cummin/cummax for float32, int32, int64. Empty arrays, single element, arrays larger than one threadgroup (>256), NaN handling, comparison vs pandas reference. Also: DeferredSeries.cumsum() materializes correctly. |
| `test_dt_quarter_dayofyear.py` | quarter/dayofyear for dates spanning multiple years, leap years, pre-epoch dates, edge cases (Jan 1, Dec 31, Feb 29). Comparison vs pandas `.dt.quarter`/`.dt.dayofyear`. |
| `test_shift.py` | shift with positive/negative periods, periods=0 (identity), periods > len (all fill), float32 NaN fill, int32 zero fill. diff and pct_change as compositions. |
| `test_fillna.py` | Scalar fillna for float32 (NaN → value), int32 with null mask. No-NaN series (identity). Mixed NaN positions. |
| `test_ffill_bfill.py` | ffill/bfill for float32 with NaN gaps at start, middle, end. All-NaN series. No-NaN series (identity). Arrays > 256 elements (cross-threadgroup propagation). |

All tests follow the existing pattern: compute on Metal, compute on pandas, assert matching results via `np.testing.assert_allclose` (float) or `np.testing.assert_array_equal` (int).

---

## Files Changed Summary

### New files (6)
- `rust/metal/elementwise/shift.metal`
- `rust/metal/elementwise/fillna.metal`
- `rust/metal/scan/fill_scan.metal`
- `rust/src/kernels/shift.rs`
- `rust/src/kernels/fillna.rs`
- `rust/src/kernels/fill_scan.rs`

### Modified files (12)
- `rust/metal/scan/scan.metal` — op-generic refactor
- `rust/metal/datetime/01_calendar.h` — add `days_from_civil`
- `rust/metal/datetime/extract.metal` — add quarter/dayofyear kernels
- `rust/src/kernels/scan.rs` — generalize to cumsum/cummin/cummax, extend dtypes
- `rust/src/kernels/datetime.rs` — add quarter/dayofyear dispatch
- `rust/src/kernels/filter.rs` — update kernel names after scan refactor
- `rust/src/kernels/mod.rs` — register new kernel modules
- `rust/src/lib.rs` — register new pyfunctions
- `rust/build.rs` — include new metal sources
- `src/metaldf/_engine/__init__.py` — register new ops
- `src/metaldf/_engine/_metal.py` — MetalEngine methods
- `src/metaldf/_wrappers.py` — ProxySeries methods + dt accessor properties

### New test files (5)
- `tests/test_cumulative.py`
- `tests/test_dt_quarter_dayofyear.py`
- `tests/test_shift.py`
- `tests/test_fillna.py`
- `tests/test_ffill_bfill.py`

---

## Implementation Order

1. **Cumulative ops** — refactors scan.metal which other features don't depend on, and validates the op-generic pattern
2. **Datetime quarter/dayofyear** — independent, smallest change
3. **shift/diff/pct_change** — independent, simple kernel
4. **fillna** — independent, simple kernel
5. **ffill/bfill** — last because it's the most complex scan variant and benefits from the scan refactor being validated in step 1

## FEATURE_GAP.md Updates

After implementation, update the tracking table:

| ID | Status change |
|----|---------------|
| P4 | cumsum: Missing (public) → Done. cummin/cummax: Missing → Done. cumprod: still Missing. |
| P9 | fillna: Missing → Done (scalar only). ffill/bfill: Missing → Partial (float32 only). |
| P11 | shift/diff/pct_change: Missing → Done. |
| P20 | quarter/dayofyear: Missing → Done. strftime/round: still Missing. |
