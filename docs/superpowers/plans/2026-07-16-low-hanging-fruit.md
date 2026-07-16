# Low-Hanging Fruit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add cumulative ops (cumsum/cummin/cummax), datetime quarter/dayofyear, shift/diff/pct_change, fillna, and ffill/bfill to metaldf's GPU-accelerated pandas proxy.

**Architecture:** Each feature follows the same 4-layer pipeline: Metal shader (MSL kernel) → Rust dispatch (`#[pyfunction]` via PyO3) → Python engine registry (`_engine/`) → ProxySeries method (`_wrappers.py`). The scan.metal template is refactored to be op-generic for cumulative ops; ffill/bfill get their own separate scan kernels. shift/fillna are standalone elementwise kernels. diff/pct_change compose from shift + existing arithmetic (no new kernels).

**Tech Stack:** Metal Shading Language (MSL), Rust (PyO3 + metal-rs), Python (numpy, pandas)

## Global Constraints

- macOS only (Metal GPU). All kernel tests use `pytest.mark.skipif(not HAS_METAL, ...)`.
- Zero-copy buffer pattern: callers must keep source numpy arrays alive while MetalSeries is in use.
- `build.rs` auto-discovers `rust/metal/` subdirectories — no build.rs changes needed for new `.metal` files in existing dirs.
- Kernel names follow `{operation}_{suffix}` where suffix comes from `DType::kernel_suffix()`: `float32`, `int32`, `int64`, `uint32`.
- All new Metal kernels use `dispatch_thread_groups` (not `dispatch_threads`) with explicit `idx >= len` bounds guard.
- Python engine uses `np.dtype(...)` instances in dtype sets (not bare `np.float32`), due to hash mismatch — see `_metal.py` comments.
- Test pattern: compute on Metal, compute on pandas/numpy reference, assert via `np.testing.assert_array_equal` (int) or `np.testing.assert_allclose` (float, `rtol=1e-5`).
- Build after each Rust task: `cd rust && maturin develop --release 2>&1 | tail -5`
- Run tests after each task: `cd /Users/atlnguyen/personal_git/metaldf && python -m pytest tests/<test_file>.py -v`

---

### Task 1: Refactor scan.metal to be op-generic + add cumsum/cummin/cummax Metal kernels

**Files:**
- Modify: `rust/metal/scan/scan.metal`
- Test: `tests/test_cumulative.py` (created in Task 3)

- [ ] **Step 1: Refactor scan.metal to use Op template parameter**

Replace the entire contents of `rust/metal/scan/scan.metal` with:

```metal
// GPU inclusive prefix-scan kernels — op-generic two-pass algorithm.
//
// Pass 1 (scan_inclusive_*): each threadgroup loads up to
// SCAN_THREADGROUP_SIZE elements, runs a Hillis-Steele inclusive scan
// in shared memory using Op::apply, writes scanned values to output,
// and publishes the group's total to partials[group_id].
//
// Between passes, the Rust dispatcher recursively scans partials.
//
// Pass 2 (scan_propagate_*): applies Op::apply(partials[group_id-1], elem)
// to each element of group group_id. Group 0 is left untouched.

#ifndef SCAN_THREADGROUP_SIZE
#define SCAN_THREADGROUP_SIZE 256
#endif

template <typename T, typename Op>
void scan_inclusive_impl(
    device const T* input,
    device T* output,
    device T* partials,
    threadgroup T* shared,
    uint tid,
    uint group_id,
    uint group_size,
    device const uint* len_ptr
) {
    uint len = *len_ptr;
    uint base = group_id * group_size + tid;

    shared[tid] = (base < len) ? input[base] : Op::identity;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = 1; offset < group_size; offset *= 2) {
        T val = shared[tid];
        if (tid >= offset) {
            val = Op::apply(shared[tid - offset], shared[tid]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[tid] = val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (base < len) {
        output[base] = shared[tid];
    }

    if (tid == group_size - 1) {
        partials[group_id] = shared[tid];
    }
}

template <typename T, typename Op>
void scan_propagate_impl(
    device T* data,
    device const T* partials,
    uint tid,
    uint group_id,
    uint group_size,
    device const uint* len_ptr
) {
    if (group_id == 0) return;
    uint len = *len_ptr;
    uint base = group_id * group_size + tid;
    if (base < len) {
        data[base] = Op::apply(partials[group_id - 1], data[base]);
    }
}

#define INSTANTIATE_SCAN_OP(T, Op, suffix) \
    [[kernel]] void scan_inclusive_##suffix( \
        device const T* input       [[buffer(0)]], \
        device T* output            [[buffer(1)]], \
        device T* partials          [[buffer(2)]], \
        device const uint* len_ptr  [[buffer(3)]], \
        threadgroup T* shared       [[threadgroup(0)]], \
        uint tid                    [[thread_position_in_threadgroup]], \
        uint group_id               [[threadgroup_position_in_grid]], \
        uint group_size             [[threads_per_threadgroup]] \
    ) { scan_inclusive_impl<T, Op<T>>(input, output, partials, shared, tid, group_id, group_size, len_ptr); } \
    [[kernel]] void scan_propagate_##suffix( \
        device T* data              [[buffer(0)]], \
        device const T* partials    [[buffer(1)]], \
        device const uint* len_ptr  [[buffer(2)]], \
        uint tid                    [[thread_position_in_threadgroup]], \
        uint group_id               [[threadgroup_position_in_grid]], \
        uint group_size             [[threads_per_threadgroup]] \
    ) { scan_propagate_impl<T, Op<T>>(data, partials, tid, group_id, group_size, len_ptr); }

// cumsum
INSTANTIATE_SCAN_OP(float, SumOp, sum_float32)
INSTANTIATE_SCAN_OP(int,   SumOp, sum_int32)
INSTANTIATE_SCAN_OP(long,  SumOp, sum_int64)
INSTANTIATE_SCAN_OP(uint,  SumOp, sum_uint32)

// cummin
INSTANTIATE_SCAN_OP(float, MinOp, min_float32)
INSTANTIATE_SCAN_OP(int,   MinOp, min_int32)
INSTANTIATE_SCAN_OP(long,  MinOp, min_int64)

// cummax
INSTANTIATE_SCAN_OP(float, MaxOp, max_float32)
INSTANTIATE_SCAN_OP(int,   MaxOp, max_int32)
INSTANTIATE_SCAN_OP(long,  MaxOp, max_int64)
```

- [ ] **Step 2: Build to verify Metal compilation**

Run: `cd /Users/atlnguyen/personal_git/metaldf/rust && maturin develop --release 2>&1 | tail -5`
Expected: Build succeeds. The scan library now includes `SumOp`/`MinOp`/`MaxOp` from `02_reduce_ops.h` (prepended by `build.rs` via the common preamble).

- [ ] **Step 3: Commit**

```bash
git add rust/metal/scan/scan.metal
git commit -m "refactor: make scan.metal op-generic for cumsum/cummin/cummax"
```

---

### Task 2: Update scan.rs for op-generic cumulative scan dispatch

**Files:**
- Modify: `rust/src/kernels/scan.rs`
- Modify: `rust/src/lib.rs`
- Test: `tests/test_cumulative.py` (created in Task 3)

**Interfaces:**
- Produces: `metal_cumsum(input: &MetalSeries) -> PyResult<MetalSeries>`, `metal_cummin(...)`, `metal_cummax(...)` — registered as pyfunctions in `metaldf_engine`
- Produces: `cumulative_scan(input: &Buffer, len: usize, dtype: DType, op: &str) -> PyResult<Buffer>` — internal, replaces `prefix_sum_inclusive`

- [ ] **Step 1: Rewrite scan.rs**

Replace the entire contents of `rust/src/kernels/scan.rs` with:

```rust
use pyo3::prelude::*;
use metal::{MTLSize, MTLResourceOptions};

use crate::backend::MetalBackend;
use crate::buffer::{SharedBuffer, DType};
use crate::kernels::{load_scan_library, get_pipeline_state};
use crate::series::MetalSeries;

const SCAN_TG_SIZE: u64 = 256;

fn check_cumulative_dtype(dtype: DType) -> PyResult<()> {
    match dtype {
        DType::Float32 | DType::Int32 | DType::Int64 | DType::Uint32
        | DType::Datetime | DType::Timedelta => Ok(()),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Cumulative scan not supported for {:?}", dtype
        ))),
    }
}

fn scan_kernel_suffix(dtype: DType) -> &'static str {
    match dtype {
        DType::Float32 => "float32",
        DType::Int32 => "int32",
        DType::Int64 | DType::Datetime | DType::Timedelta => "int64",
        DType::Uint32 => "uint32",
        _ => unreachable!(),
    }
}

pub fn cumulative_scan(
    input: &metal::Buffer,
    len: usize,
    dtype: DType,
    op: &str,
) -> PyResult<metal::Buffer> {
    check_cumulative_dtype(dtype)?;

    let (device, queue) = MetalBackend::device_and_queue()?;
    let elem_size = dtype.size_in_bytes() as u64;
    let suffix = scan_kernel_suffix(dtype);

    if len == 0 {
        return Ok(device.new_buffer(elem_size, MTLResourceOptions::StorageModeShared));
    }

    let library = load_scan_library(device)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let num_groups = (len as u64 + SCAN_TG_SIZE - 1) / SCAN_TG_SIZE;

    let output = device.new_buffer(len as u64 * elem_size, MTLResourceOptions::StorageModeShared);
    let partials = device.new_buffer(num_groups * elem_size, MTLResourceOptions::StorageModeShared);
    let len_buf = device.new_buffer(
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        *(len_buf.contents() as *mut u32) = len as u32;
    }

    // Pass 1: per-threadgroup local scan
    let scan_name = format!("scan_inclusive_{op}_{suffix}");
    let scan_pl = get_pipeline_state(device, &library, &scan_name)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&scan_pl);
    enc.set_buffer(0, Some(input), 0);
    enc.set_buffer(1, Some(&output), 0);
    enc.set_buffer(2, Some(&partials), 0);
    enc.set_buffer(3, Some(&len_buf), 0);
    enc.set_threadgroup_memory_length(0, SCAN_TG_SIZE * elem_size);
    enc.dispatch_thread_groups(
        MTLSize::new(num_groups, 1, 1),
        MTLSize::new(SCAN_TG_SIZE, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            format!("Scan pass 1 ({scan_name}) failed"),
        ));
    }

    if num_groups <= 1 {
        return Ok(output);
    }

    let scanned_partials = cumulative_scan(&partials, num_groups as usize, dtype, op)?;

    let prop_name = format!("scan_propagate_{op}_{suffix}");
    let prop_pl = get_pipeline_state(device, &library, &prop_name)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let cb2 = queue.new_command_buffer();
    let enc2 = cb2.new_compute_command_encoder();
    enc2.set_compute_pipeline_state(&prop_pl);
    enc2.set_buffer(0, Some(&output), 0);
    enc2.set_buffer(1, Some(&scanned_partials), 0);
    enc2.set_buffer(2, Some(&len_buf), 0);
    enc2.dispatch_thread_groups(
        MTLSize::new(num_groups, 1, 1),
        MTLSize::new(SCAN_TG_SIZE, 1, 1),
    );
    enc2.end_encoding();
    cb2.commit();
    cb2.wait_until_completed();

    if cb2.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            format!("Scan pass 2 ({prop_name}) failed"),
        ));
    }

    Ok(output)
}

/// Backward-compat wrapper used by filter.rs
pub fn prefix_sum_inclusive(
    input: &metal::Buffer,
    len: usize,
    dtype: DType,
) -> PyResult<metal::Buffer> {
    cumulative_scan(input, len, dtype, "sum")
}

#[pyfunction]
pub fn metal_prefix_sum(input: &MetalSeries) -> PyResult<MetalSeries> {
    let buf = input.as_numeric_checked()?;
    let result = cumulative_scan(buf.metal_buffer(), buf.len, buf.dtype, "sum")?;
    let result_buf = SharedBuffer::from_metal_buffer(result, buf.len, buf.dtype);
    Ok(MetalSeries::from_numeric(result_buf))
}

#[pyfunction]
pub fn metal_cumsum(input: &MetalSeries) -> PyResult<MetalSeries> {
    let buf = input.as_numeric_checked()?;
    let result = cumulative_scan(buf.metal_buffer(), buf.len, buf.dtype, "sum")?;
    let result_buf = SharedBuffer::from_metal_buffer(result, buf.len, buf.dtype);
    Ok(MetalSeries::from_numeric(result_buf))
}

#[pyfunction]
pub fn metal_cummin(input: &MetalSeries) -> PyResult<MetalSeries> {
    let buf = input.as_numeric_checked()?;
    let result = cumulative_scan(buf.metal_buffer(), buf.len, buf.dtype, "min")?;
    let result_buf = SharedBuffer::from_metal_buffer(result, buf.len, buf.dtype);
    Ok(MetalSeries::from_numeric(result_buf))
}

#[pyfunction]
pub fn metal_cummax(input: &MetalSeries) -> PyResult<MetalSeries> {
    let buf = input.as_numeric_checked()?;
    let result = cumulative_scan(buf.metal_buffer(), buf.len, buf.dtype, "max")?;
    let result_buf = SharedBuffer::from_metal_buffer(result, buf.len, buf.dtype);
    Ok(MetalSeries::from_numeric(result_buf))
}
```

- [ ] **Step 2: Register new pyfunctions in lib.rs**

In `rust/src/lib.rs`, add the import (alongside the existing `metal_prefix_sum` import on line 30):

```rust
use kernels::scan::{metal_prefix_sum, metal_cumsum, metal_cummin, metal_cummax};
```

And add these three lines after `m.add_wrapped(wrap_pyfunction!(metal_prefix_sum))?;` (line 123):

```rust
    m.add_wrapped(wrap_pyfunction!(metal_cumsum))?;
    m.add_wrapped(wrap_pyfunction!(metal_cummin))?;
    m.add_wrapped(wrap_pyfunction!(metal_cummax))?;
```

- [ ] **Step 3: Build and verify existing scan tests still pass**

Run:
```bash
cd /Users/atlnguyen/personal_git/metaldf/rust && maturin develop --release 2>&1 | tail -5
cd /Users/atlnguyen/personal_git/metaldf && python -m pytest tests/test_scan.py -v
```
Expected: All existing scan tests pass (kernel names changed internally but `metal_prefix_sum` still works via the backward-compat wrapper). Also verify filter tests still work:
```bash
python -m pytest tests/test_filter.py tests/test_bool_indexing.py -v
```

- [ ] **Step 4: Commit**

```bash
git add rust/src/kernels/scan.rs rust/src/lib.rs
git commit -m "feat: add metal_cumsum/cummin/cummax Rust dispatch"
```

---

### Task 3: Write cumulative ops tests

**Files:**
- Create: `tests/test_cumulative.py`

**Interfaces:**
- Consumes: `metaldf_engine.metal_cumsum(ms)`, `metaldf_engine.metal_cummin(ms)`, `metaldf_engine.metal_cummax(ms)`

- [ ] **Step 1: Write the test file**

Create `tests/test_cumulative.py`:

```python
"""Tests for GPU cumulative ops: cumsum, cummin, cummax."""

import numpy as np
import pandas as pd
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")

SCAN_TG_SIZE = 256


class TestCumsumFloat32:
    def test_small(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_cumsum(ms)
        np.testing.assert_allclose(result.to_numpy(), np.cumsum(arr), rtol=1e-5)

    def test_negative_values(self):
        arr = np.array([-1.5, 2.5, -3.5, 4.5], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_cumsum(ms)
        np.testing.assert_allclose(result.to_numpy(), np.cumsum(arr), rtol=1e-5)

    def test_single_element(self):
        arr = np.array([42.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_cumsum(ms)
        np.testing.assert_allclose(result.to_numpy(), np.cumsum(arr), rtol=1e-5)

    def test_cross_threadgroup(self):
        arr = np.arange(1, SCAN_TG_SIZE + 100, dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_cumsum(ms)
        np.testing.assert_allclose(result.to_numpy(), np.cumsum(arr), rtol=1e-4)

    def test_large(self):
        arr = np.ones(100_000, dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_cumsum(ms)
        np.testing.assert_allclose(result.to_numpy(), np.cumsum(arr), rtol=1e-3)


class TestCumsumInt32:
    def test_small(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        ms = metaldf_engine.MetalSeries.from_numpy_i32(arr)
        result = metaldf_engine.metal_cumsum(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.cumsum(arr))

    def test_negative(self):
        rng = np.random.default_rng(0)
        arr = rng.integers(-100, 100, size=37).astype(np.int32)
        ms = metaldf_engine.MetalSeries.from_numpy_i32(arr)
        result = metaldf_engine.metal_cumsum(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.cumsum(arr))

    def test_cross_threadgroup(self):
        arr = np.arange(1, SCAN_TG_SIZE + 2, dtype=np.int32)
        ms = metaldf_engine.MetalSeries.from_numpy_i32(arr)
        result = metaldf_engine.metal_cumsum(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.cumsum(arr))


class TestCumsumInt64:
    def test_small(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        ms = metaldf_engine.MetalSeries.from_numpy_i64(arr)
        result = metaldf_engine.metal_cumsum(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.cumsum(arr))

    def test_cross_threadgroup(self):
        arr = np.arange(1, SCAN_TG_SIZE + 50, dtype=np.int64)
        ms = metaldf_engine.MetalSeries.from_numpy_i64(arr)
        result = metaldf_engine.metal_cumsum(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.cumsum(arr))


class TestCumminFloat32:
    def test_small(self):
        arr = np.array([5.0, 3.0, 4.0, 1.0, 2.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_cummin(ms)
        np.testing.assert_allclose(result.to_numpy(), np.minimum.accumulate(arr), rtol=1e-5)

    def test_already_sorted_ascending(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_cummin(ms)
        np.testing.assert_allclose(result.to_numpy(), np.minimum.accumulate(arr), rtol=1e-5)

    def test_cross_threadgroup(self):
        rng = np.random.default_rng(42)
        arr = rng.random(SCAN_TG_SIZE + 100).astype(np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_cummin(ms)
        np.testing.assert_allclose(result.to_numpy(), np.minimum.accumulate(arr), rtol=1e-5)


class TestCumminInt32:
    def test_small(self):
        arr = np.array([5, 3, 4, 1, 2], dtype=np.int32)
        ms = metaldf_engine.MetalSeries.from_numpy_i32(arr)
        result = metaldf_engine.metal_cummin(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.minimum.accumulate(arr))


class TestCumminInt64:
    def test_small(self):
        arr = np.array([5, 3, 4, 1, 2], dtype=np.int64)
        ms = metaldf_engine.MetalSeries.from_numpy_i64(arr)
        result = metaldf_engine.metal_cummin(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.minimum.accumulate(arr))


class TestCummaxFloat32:
    def test_small(self):
        arr = np.array([1.0, 5.0, 3.0, 4.0, 2.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_cummax(ms)
        np.testing.assert_allclose(result.to_numpy(), np.maximum.accumulate(arr), rtol=1e-5)

    def test_already_sorted_descending(self):
        arr = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_cummax(ms)
        np.testing.assert_allclose(result.to_numpy(), np.maximum.accumulate(arr), rtol=1e-5)

    def test_cross_threadgroup(self):
        rng = np.random.default_rng(42)
        arr = rng.random(SCAN_TG_SIZE + 100).astype(np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_cummax(ms)
        np.testing.assert_allclose(result.to_numpy(), np.maximum.accumulate(arr), rtol=1e-5)


class TestCummaxInt32:
    def test_small(self):
        arr = np.array([1, 5, 3, 4, 2], dtype=np.int32)
        ms = metaldf_engine.MetalSeries.from_numpy_i32(arr)
        result = metaldf_engine.metal_cummax(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.maximum.accumulate(arr))


class TestCummaxInt64:
    def test_small(self):
        arr = np.array([1, 5, 3, 4, 2], dtype=np.int64)
        ms = metaldf_engine.MetalSeries.from_numpy_i64(arr)
        result = metaldf_engine.metal_cummax(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.maximum.accumulate(arr))
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/test_cumulative.py -v`
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cumulative.py
git commit -m "test: add GPU cumsum/cummin/cummax kernel tests"
```

---

### Task 4: Wire cumulative ops through Python engine + ProxySeries + DeferredSeries

**Files:**
- Modify: `src/metaldf/_engine/_metal.py`
- Modify: `src/metaldf/_engine/__init__.py`
- Modify: `src/metaldf/_wrappers.py`
- Modify: `src/metaldf/_deferred.py`

**Interfaces:**
- Consumes: `metaldf_engine.metal_cumsum(ms)`, `metaldf_engine.metal_cummin(ms)`, `metaldf_engine.metal_cummax(ms)` (from Task 2)
- Produces: `ProxySeries.cumsum()`, `.cummin()`, `.cummax()` returning ProxySeries; `DeferredSeries.cumsum()`, `.cummin()`, `.cummax()` that materialize first

- [ ] **Step 1: Add MetalEngine dispatch methods in _metal.py**

Add after the `_dispatch_reduction` function (around line 188) in `src/metaldf/_engine/_metal.py`:

```python
_CUMULATIVE_DTYPES = {np.dtype(np.float32), np.dtype(np.int32), np.dtype(np.int64),
                      _DATETIME_DTYPE, _TIMEDELTA_DTYPE}


def _dispatch_cumulative(op_name: str, data: Any) -> Any:
    """Try Metal cumulative scan, raise MetalNotAvailable on failure."""
    if not _has_metal():
        raise MetalNotAvailable("Metal not available")

    from metaldf._wrappers import ProxySeries
    if isinstance(data, ProxySeries) and hasattr(data, "_pandas_obj"):
        data = data.to_pandas()

    if not hasattr(data, "dtype"):
        raise MetalNotAvailable("Operand must be a pandas Series or numpy array")

    if data.dtype not in _CUMULATIVE_DTYPES:
        raise MetalNotAvailable(f"Unsupported dtype for cumulative: {data.dtype}")

    if data.dtype == _DATETIME_DTYPE and op_name == "cumsum":
        raise MetalNotAvailable("cumsum not meaningful for datetime")

    arr = _extract_array(data)
    buf = _make_series(arr)
    rust_fn = getattr(metaldf_engine, f"metal_{op_name}")
    result = rust_fn(buf)
    out_arr = result.to_numpy()
    out_arr = _restore_datetime_dtype(out_arr, data.dtype)
    return pd.Series(out_arr, index=data.index if hasattr(data, 'index') else None,
                     name=getattr(data, 'name', None))
```

Add these static methods to the `MetalEngine` class (after the reduction methods, around line 596):

```python
    # -- Cumulative ops ----------------------------------------------------

    @staticmethod
    def metal_cumsum(data: Any) -> Any:
        return _dispatch_cumulative("cumsum", data)

    @staticmethod
    def metal_cummin(data: Any) -> Any:
        return _dispatch_cumulative("cummin", data)

    @staticmethod
    def metal_cummax(data: Any) -> Any:
        return _dispatch_cumulative("cummax", data)
```

- [ ] **Step 2: Register in engine __init__.py**

Add these three lines after the existing `register("mean", ...)` line (around line 50) in `src/metaldf/_engine/__init__.py`:

```python
    register("cumsum", MetalEngine.metal_cumsum)
    register("cummin", MetalEngine.metal_cummin)
    register("cummax", MetalEngine.metal_cummax)
```

- [ ] **Step 3: Add _try_metal_series_op helper and cumulative methods to ProxySeries in _wrappers.py**

Add this method to the `ProxySeries` class, right before the `_try_metal_reduction` method (before line 810):

```python
    def _try_metal_series_op(self, op_name: str, *args: Any, **kwargs: Any) -> Any:
        """Try Metal path for a series-returning operation, fall back to pandas."""
        from metaldf._engine import execute
        from metaldf._wrappers import _wrap_result
        from metaldf.exceptions import MetalNotAvailable

        try:
            result = execute(op_name, self._pandas_obj, *args, **kwargs)
            return _wrap_result(result)
        except (MetalNotAvailable, Exception):
            pandas_method = getattr(pd.Series, op_name, None)
            if pandas_method is None:
                raise AttributeError(f"'{type(self).__name__}' has no attribute '{op_name}'")
            result = pandas_method(self._pandas_obj, *args, **kwargs)
            return _wrap_result(result)
```

Add these methods to the `ProxySeries` class, after the `mean` method (after line 847):

```python
    def cumsum(self, *args: Any, **kwargs: Any) -> Any:
        return self._try_metal_series_op("cumsum", *args, **kwargs)

    def cummin(self, *args: Any, **kwargs: Any) -> Any:
        return self._try_metal_series_op("cummin", *args, **kwargs)

    def cummax(self, *args: Any, **kwargs: Any) -> Any:
        return self._try_metal_series_op("cummax", *args, **kwargs)
```

- [ ] **Step 4: Add materialization triggers to DeferredSeries in _deferred.py**

Add these methods to `DeferredSeries`, after the `mean` method (after line 378):

```python
    def cumsum(self, *args: Any, **kwargs: Any) -> Any:
        materialized = self._materialize()
        from metaldf._wrappers import ProxySeries
        ps = ProxySeries(_pandas_obj=materialized)
        return ps.cumsum(*args, **kwargs)

    def cummin(self, *args: Any, **kwargs: Any) -> Any:
        materialized = self._materialize()
        from metaldf._wrappers import ProxySeries
        ps = ProxySeries(_pandas_obj=materialized)
        return ps.cummin(*args, **kwargs)

    def cummax(self, *args: Any, **kwargs: Any) -> Any:
        materialized = self._materialize()
        from metaldf._wrappers import ProxySeries
        ps = ProxySeries(_pandas_obj=materialized)
        return ps.cummax(*args, **kwargs)
```

- [ ] **Step 5: Run tests to verify end-to-end**

Run:
```bash
python -m pytest tests/test_cumulative.py tests/test_scan.py tests/test_filter.py tests/test_bool_indexing.py -v
```
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/metaldf/_engine/_metal.py src/metaldf/_engine/__init__.py src/metaldf/_wrappers.py src/metaldf/_deferred.py
git commit -m "feat: wire cumsum/cummin/cummax through engine, ProxySeries, and DeferredSeries"
```

---

### Task 5: Add datetime quarter and dayofyear Metal kernels

**Files:**
- Modify: `rust/metal/datetime/01_calendar.h`
- Modify: `rust/metal/datetime/extract.metal`
- Modify: `rust/src/kernels/datetime.rs`
- Modify: `rust/src/lib.rs`

**Interfaces:**
- Produces: `metaldf_engine.metal_dt_quarter(ms)`, `metaldf_engine.metal_dt_dayofyear(ms)` — take a Datetime MetalSeries, return Int32 MetalSeries

- [ ] **Step 1: Add days_from_civil to 01_calendar.h**

Add at the end of `rust/metal/datetime/01_calendar.h` (after the `civil_from_days` function, line 70):

```metal
// Howard Hinnant's inverse: (year, month, day) -> day count relative to epoch.
// See http://howardhinnant.github.io/date_algorithms.html
inline long days_from_civil(int y, int m, int d) {
    y -= (m <= 2);
    long era = (y >= 0 ? y : y - 399) / 400;
    uint yoe = uint(y - era * 400);
    uint doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
    uint doe = yoe * 365 + yoe/4 - yoe/100 + doy;
    return era * 146097 + long(doe) - 719468;
}
```

- [ ] **Step 2: Add quarter and dayofyear kernels to extract.metal**

Add at the end of `rust/metal/datetime/extract.metal` (after the `dt_dayofweek_i64` kernel, line 77):

```metal
kernel void dt_quarter_i64(device const long* ns [[buffer(0)]],
                           device int* out        [[buffer(1)]],
                           device const uint* len_ptr [[buffer(2)]],
                           uint idx [[thread_position_in_grid]]) {
    if (idx >= *len_ptr) return;
    long days = floor_div(ns[idx], NS_PER_DAY);
    int month = civil_from_days(days).month;
    out[idx] = (month - 1) / 3 + 1;
}

kernel void dt_dayofyear_i64(device const long* ns [[buffer(0)]],
                              device int* out        [[buffer(1)]],
                              device const uint* len_ptr [[buffer(2)]],
                              uint idx [[thread_position_in_grid]]) {
    if (idx >= *len_ptr) return;
    long days = floor_div(ns[idx], NS_PER_DAY);
    CivilDate c = civil_from_days(days);
    long jan1 = days_from_civil(c.year, 1, 1);
    out[idx] = int(days - jan1) + 1;
}
```

- [ ] **Step 3: Add Rust dispatch functions in datetime.rs**

Add at the end of `rust/src/kernels/datetime.rs` (after `metal_dt_dayofweek`, line 105):

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

- [ ] **Step 4: Register in lib.rs**

In `rust/src/lib.rs`, update the datetime import (line 34-37) to include the new functions:

```rust
use kernels::datetime::{
    metal_dt_year, metal_dt_month, metal_dt_day,
    metal_dt_hour, metal_dt_minute, metal_dt_second, metal_dt_dayofweek,
    metal_dt_quarter, metal_dt_dayofyear,
};
```

Add after `m.add_wrapped(wrap_pyfunction!(metal_dt_dayofweek))?;` (line 138):

```rust
    m.add_wrapped(wrap_pyfunction!(metal_dt_quarter))?;
    m.add_wrapped(wrap_pyfunction!(metal_dt_dayofyear))?;
```

- [ ] **Step 5: Build and verify**

Run:
```bash
cd /Users/atlnguyen/personal_git/metaldf/rust && maturin develop --release 2>&1 | tail -5
cd /Users/atlnguyen/personal_git/metaldf && python -m pytest tests/test_dt_accessor.py -v
```
Expected: Build succeeds, existing datetime tests pass.

- [ ] **Step 6: Commit**

```bash
git add rust/metal/datetime/01_calendar.h rust/metal/datetime/extract.metal rust/src/kernels/datetime.rs rust/src/lib.rs
git commit -m "feat: add GPU datetime quarter and dayofyear extraction kernels"
```

---

### Task 6: Wire datetime quarter/dayofyear into Python + write tests

**Files:**
- Modify: `src/metaldf/_wrappers.py`
- Create: `tests/test_dt_quarter_dayofyear.py`

**Interfaces:**
- Consumes: `metaldf_engine.metal_dt_quarter(ms)`, `metaldf_engine.metal_dt_dayofyear(ms)` (from Task 5)
- Produces: `ProxyDatetimeAccessor.quarter`, `.dayofyear` properties

- [ ] **Step 1: Add properties to ProxyDatetimeAccessor**

In `src/metaldf/_wrappers.py`, add after the `dayofweek` property (after line 1283):

```python
    @property
    def quarter(self) -> Any: return self._dispatch("quarter")

    @property
    def dayofyear(self) -> Any: return self._dispatch("dayofyear")
```

- [ ] **Step 2: Write the test file**

Create `tests/test_dt_quarter_dayofyear.py`:

```python
"""Tests for GPU datetime quarter and dayofyear extraction."""

import numpy as np
import pandas as pd
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")


def _to_metal_series(dates: pd.Series) -> tuple:
    ns = dates.values.astype("datetime64[ns]").view(np.int64)
    return metaldf_engine.MetalSeries.from_numpy_datetime(ns), ns


@pytest.fixture
def sample_dates():
    return pd.Series(pd.to_datetime([
        "2020-01-01 00:00:00",
        "2020-03-31 12:00:00",
        "2020-06-15 06:30:00",
        "2020-09-22 18:45:00",
        "2020-12-31 23:59:59",
    ]))


class TestQuarter:
    def test_all_quarters(self, sample_dates):
        ms, _ns = _to_metal_series(sample_dates)
        result = metaldf_engine.metal_dt_quarter(ms)
        expected = sample_dates.dt.quarter.values
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_quarter_boundaries(self):
        dates = pd.Series(pd.to_datetime([
            "2020-01-01", "2020-03-31",
            "2020-04-01", "2020-06-30",
            "2020-07-01", "2020-09-30",
            "2020-10-01", "2020-12-31",
        ]))
        ms, _ns = _to_metal_series(dates)
        result = metaldf_engine.metal_dt_quarter(ms)
        expected = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int32)
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_pre_epoch(self):
        dates = pd.Series(pd.to_datetime(["1965-07-15", "1900-01-01"]))
        ms, _ns = _to_metal_series(dates)
        result = metaldf_engine.metal_dt_quarter(ms)
        expected = dates.dt.quarter.values
        np.testing.assert_array_equal(result.to_numpy(), expected)


class TestDayOfYear:
    def test_basic(self, sample_dates):
        ms, _ns = _to_metal_series(sample_dates)
        result = metaldf_engine.metal_dt_dayofyear(ms)
        expected = sample_dates.dt.dayofyear.values
        np.testing.assert_array_equal(result.to_numpy(), expected)

    def test_jan1_is_day1(self):
        dates = pd.Series(pd.to_datetime(["2020-01-01", "2021-01-01"]))
        ms, _ns = _to_metal_series(dates)
        result = metaldf_engine.metal_dt_dayofyear(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.array([1, 1], dtype=np.int32))

    def test_dec31_leap_year(self):
        dates = pd.Series(pd.to_datetime(["2020-12-31"]))
        ms, _ns = _to_metal_series(dates)
        result = metaldf_engine.metal_dt_dayofyear(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.array([366], dtype=np.int32))

    def test_dec31_non_leap_year(self):
        dates = pd.Series(pd.to_datetime(["2021-12-31"]))
        ms, _ns = _to_metal_series(dates)
        result = metaldf_engine.metal_dt_dayofyear(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.array([365], dtype=np.int32))

    def test_feb29_leap_year(self):
        dates = pd.Series(pd.to_datetime(["2020-02-29"]))
        ms, _ns = _to_metal_series(dates)
        result = metaldf_engine.metal_dt_dayofyear(ms)
        np.testing.assert_array_equal(result.to_numpy(), np.array([60], dtype=np.int32))

    def test_pre_epoch(self):
        dates = pd.Series(pd.to_datetime(["1965-07-15", "1900-03-01"]))
        ms, _ns = _to_metal_series(dates)
        result = metaldf_engine.metal_dt_dayofyear(ms)
        expected = dates.dt.dayofyear.values
        np.testing.assert_array_equal(result.to_numpy(), expected)


class TestProxyAccessor:
    def test_quarter_via_proxy(self):
        import metaldf
        metaldf.install()
        try:
            dates = pd.Series(pd.to_datetime(["2020-01-15", "2020-04-15", "2020-07-15", "2020-10-15"]))
            result = dates.dt.quarter
            expected = pd.Series([1, 2, 3, 4])
            np.testing.assert_array_equal(result.values, expected.values)
        finally:
            metaldf.uninstall()

    def test_dayofyear_via_proxy(self):
        import metaldf
        metaldf.install()
        try:
            dates = pd.Series(pd.to_datetime(["2020-01-01", "2020-12-31"]))
            result = dates.dt.dayofyear
            expected = pd.Series([1, 366])
            np.testing.assert_array_equal(result.values, expected.values)
        finally:
            metaldf.uninstall()
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_dt_quarter_dayofyear.py -v`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/metaldf/_wrappers.py tests/test_dt_quarter_dayofyear.py
git commit -m "feat: add datetime quarter/dayofyear to ProxyDatetimeAccessor"
```

---

### Task 7: Add shift Metal kernel + Rust dispatch

**Files:**
- Create: `rust/metal/elementwise/shift.metal`
- Create: `rust/src/kernels/shift.rs`
- Modify: `rust/src/kernels/mod.rs`
- Modify: `rust/src/lib.rs`

**Interfaces:**
- Produces: `metaldf_engine.metal_shift(ms, periods)` — takes MetalSeries + i32, returns MetalSeries

- [ ] **Step 1: Create shift.metal**

Create `rust/metal/elementwise/shift.metal`:

```metal
// Shift kernel: copy with offset, fill out-of-bounds with NaN (float) or 0 (int).

template <typename T, T fill_value>
void shift_impl(
    device const T* input,
    device T* output,
    device const int* periods_ptr,
    device const uint* len_ptr,
    uint idx
) {
    uint len = *len_ptr;
    if (idx >= len) return;
    int p = *periods_ptr;
    int src = int(idx) - p;
    if (src >= 0 && src < int(len)) {
        output[idx] = input[src];
    } else {
        output[idx] = fill_value;
    }
}

#define SHIFT_KERNEL(suffix, metal_type, fill) \
    [[kernel]] void shift_##suffix( \
        device const metal_type* input [[buffer(0)]], \
        device metal_type* output      [[buffer(1)]], \
        device const int* periods_ptr  [[buffer(2)]], \
        device const uint* len_ptr     [[buffer(3)]], \
        uint idx [[thread_position_in_grid]] \
    ) { shift_impl<metal_type, fill>(input, output, periods_ptr, len_ptr, idx); }

SHIFT_KERNEL(float32, float, as_type<float>(0x7FC00000u))
SHIFT_KERNEL(int32, int, 0)
SHIFT_KERNEL(int64, long, 0L)
```

- [ ] **Step 2: Create shift.rs**

Create `rust/src/kernels/shift.rs`:

```rust
use pyo3::prelude::*;
use metal::{MTLSize, MTLResourceOptions};

use crate::backend::MetalBackend;
use crate::buffer::{SharedBuffer, DType};
use crate::kernels::{load_elementwise_library, get_pipeline_state};
use crate::series::MetalSeries;

const THREADGROUP_SIZE: u64 = 256;

fn shift_suffix(dtype: DType) -> PyResult<&'static str> {
    match dtype {
        DType::Float32 => Ok("float32"),
        DType::Int32 => Ok("int32"),
        DType::Int64 | DType::Datetime | DType::Timedelta => Ok("int64"),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "shift not supported for {:?}", dtype
        ))),
    }
}

#[pyfunction]
pub fn metal_shift(input: &MetalSeries, periods: i32) -> PyResult<MetalSeries> {
    let buf = input.as_numeric_checked()?;
    let dtype = input.dtype;
    let len = input.len;
    let suffix = shift_suffix(dtype)?;
    let elem_size = dtype.size_in_bytes() as u64;

    let (device, queue) = MetalBackend::device_and_queue()?;

    if len == 0 {
        let out = device.new_buffer(elem_size, MTLResourceOptions::StorageModeShared);
        let result_buf = SharedBuffer::from_metal_buffer(out, 0, dtype);
        return Ok(MetalSeries::from_numeric(result_buf));
    }

    let library = load_elementwise_library(device)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    let kernel_name = format!("shift_{suffix}");
    let pipeline = get_pipeline_state(device, &library, &kernel_name)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let out = device.new_buffer(len as u64 * elem_size, MTLResourceOptions::StorageModeShared);
    let periods_buf = device.new_buffer_with_data(
        &periods as *const i32 as *const _,
        std::mem::size_of::<i32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let len_buf = device.new_buffer(
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe { *(len_buf.contents() as *mut u32) = len as u32; }

    let num_groups = (len as u64 + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(buf.metal_buffer()), 0);
    enc.set_buffer(1, Some(&out), 0);
    enc.set_buffer(2, Some(&periods_buf), 0);
    enc.set_buffer(3, Some(&len_buf), 0);
    enc.dispatch_thread_groups(
        MTLSize::new(num_groups, 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("shift kernel failed"));
    }

    let result_buf = SharedBuffer::from_metal_buffer(out, len, dtype);
    Ok(MetalSeries::from_numeric(result_buf))
}
```

- [ ] **Step 3: Register module in mod.rs**

Add to `rust/src/kernels/mod.rs`, after the `pub mod datetime;` line (line 111):

```rust
pub mod shift;
```

Note: No new `include!` or `load_*_library` needed — shift.metal lives in `elementwise/` and is automatically compiled into the elementwise library by `build.rs`.

- [ ] **Step 4: Register pyfunction in lib.rs**

Add to `rust/src/lib.rs`:

Import (after the scan imports, around line 30):
```rust
use kernels::shift::metal_shift;
```

Registration (after `metal_dt_dayofyear`, around line 140):
```rust
    m.add_wrapped(wrap_pyfunction!(metal_shift))?;
```

- [ ] **Step 5: Build and test**

Run:
```bash
cd /Users/atlnguyen/personal_git/metaldf/rust && maturin develop --release 2>&1 | tail -5
cd /Users/atlnguyen/personal_git/metaldf && python -c "
import numpy as np, metaldf_engine
arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
ms = metaldf_engine.MetalSeries.from_numpy(arr)
result = metaldf_engine.metal_shift(ms, 1)
print('shift(1):', result.to_numpy())
result2 = metaldf_engine.metal_shift(ms, -1)
print('shift(-1):', result2.to_numpy())
"
```
Expected: `shift(1): [nan  1.  2.  3.  4.]` and `shift(-1): [ 2.  3.  4.  5. nan]`

- [ ] **Step 6: Commit**

```bash
git add rust/metal/elementwise/shift.metal rust/src/kernels/shift.rs rust/src/kernels/mod.rs rust/src/lib.rs
git commit -m "feat: add GPU shift kernel for float32/int32/int64"
```

---

### Task 8: Wire shift/diff/pct_change through Python engine + write tests

**Files:**
- Modify: `src/metaldf/_engine/_metal.py`
- Modify: `src/metaldf/_engine/__init__.py`
- Modify: `src/metaldf/_wrappers.py`
- Create: `tests/test_shift.py`

**Interfaces:**
- Consumes: `metaldf_engine.metal_shift(ms, periods)` (from Task 7)
- Produces: `ProxySeries.shift(periods)`, `.diff(periods)`, `.pct_change(periods)`

- [ ] **Step 1: Add MetalEngine.metal_shift in _metal.py**

Add this dispatch function (near the `_dispatch_cumulative` function):

```python
def _dispatch_shift(data: Any, periods: int = 1) -> Any:
    """Try Metal shift, raise MetalNotAvailable on failure."""
    if not _has_metal():
        raise MetalNotAvailable("Metal not available")

    from metaldf._wrappers import ProxySeries
    if isinstance(data, ProxySeries) and hasattr(data, "_pandas_obj"):
        data = data.to_pandas()

    if not hasattr(data, "dtype"):
        raise MetalNotAvailable("Operand must be a pandas Series or numpy array")

    if data.dtype not in _SUPPORTED_DTYPES:
        raise MetalNotAvailable(f"Unsupported dtype for shift: {data.dtype}")

    arr = _extract_array(data)
    buf = _make_series(arr)
    result = metaldf_engine.metal_shift(buf, int(periods))
    out_arr = result.to_numpy()
    out_arr = _restore_datetime_dtype(out_arr, data.dtype)
    return pd.Series(out_arr, index=data.index if hasattr(data, 'index') else None,
                     name=getattr(data, 'name', None))
```

Add this static method to `MetalEngine`:

```python
    # -- Shift --------------------------------------------------------------

    @staticmethod
    def metal_shift(data: Any, periods: int = 1) -> Any:
        return _dispatch_shift(data, periods)
```

- [ ] **Step 2: Register in engine __init__.py**

Add after the cumulative registrations:

```python
    register("shift", MetalEngine.metal_shift)
```

- [ ] **Step 3: Add shift/diff/pct_change to ProxySeries in _wrappers.py**

Add these methods to `ProxySeries`, after the `cummax` method:

```python
    def shift(self, periods: int = 1, **kwargs: Any) -> Any:
        return self._try_metal_series_op("shift", periods=periods)

    def diff(self, periods: int = 1, **kwargs: Any) -> Any:
        from metaldf._wrappers import _wrap_result
        try:
            shifted = self.shift(periods)
            return self - shifted
        except Exception:
            result = pd.Series.diff(self._pandas_obj, periods=periods, **kwargs)
            return _wrap_result(result)

    def pct_change(self, periods: int = 1, **kwargs: Any) -> Any:
        from metaldf._wrappers import _wrap_result
        try:
            shifted = self.shift(periods)
            return (self - shifted) / shifted
        except Exception:
            result = pd.Series.pct_change(self._pandas_obj, periods=periods, **kwargs)
            return _wrap_result(result)
```

- [ ] **Step 4: Write the test file**

Create `tests/test_shift.py`:

```python
"""Tests for GPU shift, diff, and pct_change."""

import numpy as np
import pandas as pd
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")


class TestShiftDirect:
    def test_shift_forward_float32(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_shift(ms, 1).to_numpy()
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1:], arr[:-1], rtol=1e-5)

    def test_shift_backward_float32(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_shift(ms, -1).to_numpy()
        np.testing.assert_allclose(result[:-1], arr[1:], rtol=1e-5)
        assert np.isnan(result[-1])

    def test_shift_zero_identity(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_shift(ms, 0).to_numpy()
        np.testing.assert_allclose(result, arr, rtol=1e-5)

    def test_shift_larger_than_length(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_shift(ms, 10).to_numpy()
        assert all(np.isnan(result))

    def test_shift_int32(self):
        arr = np.array([10, 20, 30, 40], dtype=np.int32)
        ms = metaldf_engine.MetalSeries.from_numpy_i32(arr)
        result = metaldf_engine.metal_shift(ms, 1).to_numpy()
        assert result[0] == 0
        np.testing.assert_array_equal(result[1:], arr[:-1])

    def test_shift_int64(self):
        arr = np.array([10, 20, 30, 40], dtype=np.int64)
        ms = metaldf_engine.MetalSeries.from_numpy_i64(arr)
        result = metaldf_engine.metal_shift(ms, 2).to_numpy()
        np.testing.assert_array_equal(result[:2], [0, 0])
        np.testing.assert_array_equal(result[2:], arr[:2])


class TestProxyShiftDiffPctChange:
    @pytest.fixture(autouse=True)
    def _install(self):
        import metaldf
        metaldf.install()
        yield
        metaldf.uninstall()

    def test_shift(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        result = s.shift(1)
        expected = pd.Series([np.nan, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        pd.testing.assert_series_equal(result.reset_index(drop=True), expected, rtol=1e-5)

    def test_diff(self):
        s = pd.Series([1.0, 3.0, 6.0, 10.0], dtype=np.float32)
        result = s.diff(1)
        expected = pd.Series([np.nan, 2.0, 3.0, 4.0], dtype=np.float32)
        np.testing.assert_allclose(result.values[1:], expected.values[1:], rtol=1e-5)

    def test_pct_change(self):
        s = pd.Series([100.0, 110.0, 121.0], dtype=np.float32)
        result = s.pct_change(1)
        expected = pd.Series([np.nan, 0.1, 0.1], dtype=np.float32)
        np.testing.assert_allclose(result.values[1:], expected.values[1:], rtol=1e-4)
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_shift.py -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/metaldf/_engine/_metal.py src/metaldf/_engine/__init__.py src/metaldf/_wrappers.py tests/test_shift.py
git commit -m "feat: add GPU shift/diff/pct_change to ProxySeries"
```

---

### Task 9: Add fillna Metal kernel + Rust dispatch + Python wiring + tests

**Files:**
- Create: `rust/metal/elementwise/fillna.metal`
- Create: `rust/src/kernels/fillna.rs`
- Modify: `rust/src/kernels/mod.rs`
- Modify: `rust/src/lib.rs`
- Modify: `src/metaldf/_engine/_metal.py`
- Modify: `src/metaldf/_engine/__init__.py`
- Modify: `src/metaldf/_wrappers.py`
- Create: `tests/test_fillna.py`

**Interfaces:**
- Produces: `metaldf_engine.metal_fillna(ms, fill_value)`, `ProxySeries.fillna(value)`

- [ ] **Step 1: Create fillna.metal**

Create `rust/metal/elementwise/fillna.metal`:

```metal
// fillna: replace NaN with a scalar fill value.

[[kernel]] void fillna_f32(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    device const float* fill   [[buffer(2)]],
    device const uint* len_ptr [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= *len_ptr) return;
    output[idx] = isnan(input[idx]) ? *fill : input[idx];
}
```

- [ ] **Step 2: Create fillna.rs**

Create `rust/src/kernels/fillna.rs`:

```rust
use pyo3::prelude::*;
use metal::{MTLSize, MTLResourceOptions};

use crate::backend::MetalBackend;
use crate::buffer::{SharedBuffer, DType};
use crate::kernels::{load_elementwise_library, get_pipeline_state};
use crate::series::MetalSeries;

const THREADGROUP_SIZE: u64 = 256;

#[pyfunction]
pub fn metal_fillna(input: &MetalSeries, fill_value: f64) -> PyResult<MetalSeries> {
    let buf = input.as_numeric_checked()?;
    let dtype = input.dtype;
    let len = input.len;

    if dtype != DType::Float32 {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "fillna currently only supports Float32, got {:?}", dtype
        )));
    }

    let (device, queue) = MetalBackend::device_and_queue()?;

    if len == 0 {
        let out = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        let result_buf = SharedBuffer::from_metal_buffer(out, 0, dtype);
        return Ok(MetalSeries::from_numeric(result_buf));
    }

    let library = load_elementwise_library(device)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    let pipeline = get_pipeline_state(device, &library, "fillna_f32")
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let out = device.new_buffer(len as u64 * 4, MTLResourceOptions::StorageModeShared);
    let fill_f32 = fill_value as f32;
    let fill_buf = device.new_buffer_with_data(
        &fill_f32 as *const f32 as *const _,
        std::mem::size_of::<f32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let len_buf = device.new_buffer(
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe { *(len_buf.contents() as *mut u32) = len as u32; }

    let num_groups = (len as u64 + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(buf.metal_buffer()), 0);
    enc.set_buffer(1, Some(&out), 0);
    enc.set_buffer(2, Some(&fill_buf), 0);
    enc.set_buffer(3, Some(&len_buf), 0);
    enc.dispatch_thread_groups(
        MTLSize::new(num_groups, 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("fillna kernel failed"));
    }

    let result_buf = SharedBuffer::from_metal_buffer(out, len, dtype);
    Ok(MetalSeries::from_numeric(result_buf))
}
```

- [ ] **Step 3: Register module and pyfunction**

In `rust/src/kernels/mod.rs`, add after `pub mod shift;`:
```rust
pub mod fillna;
```

In `rust/src/lib.rs`, add import:
```rust
use kernels::fillna::metal_fillna;
```

Add registration:
```rust
    m.add_wrapped(wrap_pyfunction!(metal_fillna))?;
```

- [ ] **Step 4: Build**

Run: `cd /Users/atlnguyen/personal_git/metaldf/rust && maturin develop --release 2>&1 | tail -5`
Expected: Build succeeds.

- [ ] **Step 5: Add Python wiring**

In `src/metaldf/_engine/_metal.py`, add dispatch function:

```python
def _dispatch_fillna(data: Any, fill_value: float) -> Any:
    """Try Metal fillna, raise MetalNotAvailable on failure."""
    if not _has_metal():
        raise MetalNotAvailable("Metal not available")

    from metaldf._wrappers import ProxySeries
    if isinstance(data, ProxySeries) and hasattr(data, "_pandas_obj"):
        data = data.to_pandas()

    if not hasattr(data, "dtype") or data.dtype != np.dtype(np.float32):
        raise MetalNotAvailable(f"fillna GPU only supports float32")

    arr = _extract_array(data)
    buf = _make_series(arr)
    result = metaldf_engine.metal_fillna(buf, float(fill_value))
    return pd.Series(result.to_numpy(), index=data.index if hasattr(data, 'index') else None,
                     name=getattr(data, 'name', None))
```

Add `MetalEngine` static method:

```python
    # -- Fill ---------------------------------------------------------------

    @staticmethod
    def metal_fillna(data: Any, value: float = 0.0) -> Any:
        return _dispatch_fillna(data, value)
```

In `src/metaldf/_engine/__init__.py`, add:
```python
    register("fillna", MetalEngine.metal_fillna)
```

In `src/metaldf/_wrappers.py`, add to `ProxySeries` (after `pct_change`):

```python
    def fillna(self, value: Any = None, **kwargs: Any) -> Any:
        if value is not None and np.isscalar(value) and not kwargs.get("method"):
            try:
                return self._try_metal_series_op("fillna", value=value)
            except Exception:
                pass
        from metaldf._wrappers import _wrap_result
        result = pd.Series.fillna(self._pandas_obj, value=value, **kwargs)
        return _wrap_result(result)
```

- [ ] **Step 6: Write tests**

Create `tests/test_fillna.py`:

```python
"""Tests for GPU fillna."""

import numpy as np
import pandas as pd
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")


class TestFillnaDirect:
    def test_fill_nan(self):
        arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_fillna(ms, 0.0).to_numpy()
        expected = np.array([1.0, 0.0, 3.0, 0.0, 5.0], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_no_nan_identity(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_fillna(ms, 99.0).to_numpy()
        np.testing.assert_allclose(result, arr, rtol=1e-5)

    def test_all_nan(self):
        arr = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_fillna(ms, -1.0).to_numpy()
        expected = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_fill_with_nonzero(self):
        arr = np.array([np.nan, 2.0, np.nan], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_fillna(ms, 42.5).to_numpy()
        expected = np.array([42.5, 2.0, 42.5], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestProxyFillna:
    @pytest.fixture(autouse=True)
    def _install(self):
        import metaldf
        metaldf.install()
        yield
        metaldf.uninstall()

    def test_fillna_scalar(self):
        s = pd.Series([1.0, np.nan, 3.0], dtype=np.float32)
        result = s.fillna(0.0)
        expected = pd.Series([1.0, 0.0, 3.0], dtype=np.float32)
        np.testing.assert_allclose(result.values, expected.values, rtol=1e-5)

    def test_fillna_falls_back_for_int(self):
        s = pd.Series([1, 2, 3], dtype=np.int32)
        result = s.fillna(0)
        np.testing.assert_array_equal(result.values, [1, 2, 3])
```

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/test_fillna.py -v`
Expected: All tests PASS.

- [ ] **Step 8: Commit**

```bash
git add rust/metal/elementwise/fillna.metal rust/src/kernels/fillna.rs rust/src/kernels/mod.rs rust/src/lib.rs src/metaldf/_engine/_metal.py src/metaldf/_engine/__init__.py src/metaldf/_wrappers.py tests/test_fillna.py
git commit -m "feat: add GPU fillna (scalar, float32) with engine + proxy wiring"
```

---

### Task 10: Add ffill/bfill Metal kernels + Rust dispatch + Python wiring + tests

**Files:**
- Create: `rust/metal/scan/fill_scan.metal`
- Create: `rust/src/kernels/fill_scan.rs`
- Modify: `rust/src/kernels/mod.rs`
- Modify: `rust/src/lib.rs`
- Modify: `src/metaldf/_engine/_metal.py`
- Modify: `src/metaldf/_engine/__init__.py`
- Modify: `src/metaldf/_wrappers.py`
- Create: `tests/test_ffill_bfill.py`

**Interfaces:**
- Produces: `metaldf_engine.metal_ffill(ms)`, `metaldf_engine.metal_bfill(ms)`, `ProxySeries.ffill()`, `.bfill()`, `.pad()`, `.backfill()`

- [ ] **Step 1: Create fill_scan.metal**

Create `rust/metal/scan/fill_scan.metal`:

```metal
// Forward-fill and backward-fill via parallel scan.
// ffill propagates the last valid (non-NaN) value forward.
// bfill propagates the next valid value backward.

#ifndef SCAN_THREADGROUP_SIZE
#define SCAN_THREADGROUP_SIZE 256
#endif

// --- Forward fill (ffill) ---

template <typename T>
void ffill_scan_impl(
    device const T* input,
    device T* output,
    device T* partials,
    threadgroup T* shared,
    uint tid,
    uint group_id,
    uint group_size,
    device const uint* len_ptr
) {
    uint len = *len_ptr;
    uint base = group_id * group_size + tid;

    shared[tid] = (base < len) ? input[base] : NAN;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = 1; offset < group_size; offset *= 2) {
        T val = shared[tid];
        if (tid >= offset) {
            val = isnan(shared[tid]) ? shared[tid - offset] : shared[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[tid] = val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (base < len) {
        output[base] = shared[tid];
    }

    if (tid == group_size - 1) {
        partials[group_id] = shared[tid];
    }
}

template <typename T>
void ffill_propagate_impl(
    device T* data,
    device const T* partials,
    uint tid,
    uint group_id,
    uint group_size,
    device const uint* len_ptr
) {
    if (group_id == 0) return;
    uint len = *len_ptr;
    uint base = group_id * group_size + tid;
    if (base < len && isnan(data[base])) {
        T prev = partials[group_id - 1];
        if (!isnan(prev)) {
            data[base] = prev;
        }
    }
}

#define INSTANTIATE_FFILL(T, suffix) \
    [[kernel]] void ffill_scan_##suffix( \
        device const T* input       [[buffer(0)]], \
        device T* output            [[buffer(1)]], \
        device T* partials          [[buffer(2)]], \
        device const uint* len_ptr  [[buffer(3)]], \
        threadgroup T* shared       [[threadgroup(0)]], \
        uint tid                    [[thread_position_in_threadgroup]], \
        uint group_id               [[threadgroup_position_in_grid]], \
        uint group_size             [[threads_per_threadgroup]] \
    ) { ffill_scan_impl<T>(input, output, partials, shared, tid, group_id, group_size, len_ptr); } \
    [[kernel]] void ffill_propagate_##suffix( \
        device T* data              [[buffer(0)]], \
        device const T* partials    [[buffer(1)]], \
        device const uint* len_ptr  [[buffer(2)]], \
        uint tid                    [[thread_position_in_threadgroup]], \
        uint group_id               [[threadgroup_position_in_grid]], \
        uint group_size             [[threads_per_threadgroup]] \
    ) { ffill_propagate_impl<T>(data, partials, tid, group_id, group_size, len_ptr); }

INSTANTIATE_FFILL(float, float32)

// --- Backward fill (bfill) ---

template <typename T>
void bfill_scan_impl(
    device const T* input,
    device T* output,
    device T* partials,
    threadgroup T* shared,
    uint tid,
    uint group_id,
    uint group_size,
    device const uint* len_ptr
) {
    uint len = *len_ptr;
    uint base = group_id * group_size + tid;
    uint ridx = (base < len) ? (len - 1 - base) : base;

    shared[tid] = (base < len) ? input[ridx] : NAN;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = 1; offset < group_size; offset *= 2) {
        T val = shared[tid];
        if (tid >= offset) {
            val = isnan(shared[tid]) ? shared[tid - offset] : shared[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[tid] = val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (base < len) {
        output[ridx] = shared[tid];
    }

    if (tid == group_size - 1) {
        partials[group_id] = shared[tid];
    }
}

template <typename T>
void bfill_propagate_impl(
    device T* data,
    device const T* partials,
    uint tid,
    uint group_id,
    uint group_size,
    device const uint* len_ptr
) {
    if (group_id == 0) return;
    uint len = *len_ptr;
    uint base = group_id * group_size + tid;
    if (base < len) {
        uint ridx = len - 1 - base;
        if (isnan(data[ridx])) {
            T prev = partials[group_id - 1];
            if (!isnan(prev)) {
                data[ridx] = prev;
            }
        }
    }
}

#define INSTANTIATE_BFILL(T, suffix) \
    [[kernel]] void bfill_scan_##suffix( \
        device const T* input       [[buffer(0)]], \
        device T* output            [[buffer(1)]], \
        device T* partials          [[buffer(2)]], \
        device const uint* len_ptr  [[buffer(3)]], \
        threadgroup T* shared       [[threadgroup(0)]], \
        uint tid                    [[thread_position_in_threadgroup]], \
        uint group_id               [[threadgroup_position_in_grid]], \
        uint group_size             [[threads_per_threadgroup]] \
    ) { bfill_scan_impl<T>(input, output, partials, shared, tid, group_id, group_size, len_ptr); } \
    [[kernel]] void bfill_propagate_##suffix( \
        device T* data              [[buffer(0)]], \
        device const T* partials    [[buffer(1)]], \
        device const uint* len_ptr  [[buffer(2)]], \
        uint tid                    [[thread_position_in_threadgroup]], \
        uint group_id               [[threadgroup_position_in_grid]], \
        uint group_size             [[threads_per_threadgroup]] \
    ) { bfill_propagate_impl<T>(data, partials, tid, group_id, group_size, len_ptr); }

INSTANTIATE_BFILL(float, float32)
```

- [ ] **Step 2: Create fill_scan.rs**

Create `rust/src/kernels/fill_scan.rs`:

```rust
use pyo3::prelude::*;
use metal::{MTLSize, MTLResourceOptions};

use crate::backend::MetalBackend;
use crate::buffer::{SharedBuffer, DType};
use crate::kernels::{load_scan_library, get_pipeline_state};
use crate::series::MetalSeries;

const SCAN_TG_SIZE: u64 = 256;

fn fill_scan_dispatch(
    input: &metal::Buffer,
    len: usize,
    scan_kernel: &str,
    propagate_kernel: &str,
) -> PyResult<metal::Buffer> {
    let (device, queue) = MetalBackend::device_and_queue()?;
    let elem_size = 4u64; // float32

    if len == 0 {
        return Ok(device.new_buffer(elem_size, MTLResourceOptions::StorageModeShared));
    }

    let library = load_scan_library(device)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let num_groups = (len as u64 + SCAN_TG_SIZE - 1) / SCAN_TG_SIZE;

    let output = device.new_buffer(len as u64 * elem_size, MTLResourceOptions::StorageModeShared);
    let partials = device.new_buffer(num_groups * elem_size, MTLResourceOptions::StorageModeShared);
    let len_buf = device.new_buffer(
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe { *(len_buf.contents() as *mut u32) = len as u32; }

    // Pass 1: local scan
    let scan_pl = get_pipeline_state(device, &library, scan_kernel)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&scan_pl);
    enc.set_buffer(0, Some(input), 0);
    enc.set_buffer(1, Some(&output), 0);
    enc.set_buffer(2, Some(&partials), 0);
    enc.set_buffer(3, Some(&len_buf), 0);
    enc.set_threadgroup_memory_length(0, SCAN_TG_SIZE * elem_size);
    enc.dispatch_thread_groups(
        MTLSize::new(num_groups, 1, 1),
        MTLSize::new(SCAN_TG_SIZE, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            format!("{scan_kernel} failed"),
        ));
    }

    if num_groups <= 1 {
        return Ok(output);
    }

    // Recursively fill-scan the partials
    let scanned_partials = fill_scan_dispatch(
        &partials, num_groups as usize, scan_kernel, propagate_kernel,
    )?;

    // Pass 2: propagate
    let prop_pl = get_pipeline_state(device, &library, propagate_kernel)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let cb2 = queue.new_command_buffer();
    let enc2 = cb2.new_compute_command_encoder();
    enc2.set_compute_pipeline_state(&prop_pl);
    enc2.set_buffer(0, Some(&output), 0);
    enc2.set_buffer(1, Some(&scanned_partials), 0);
    enc2.set_buffer(2, Some(&len_buf), 0);
    enc2.dispatch_thread_groups(
        MTLSize::new(num_groups, 1, 1),
        MTLSize::new(SCAN_TG_SIZE, 1, 1),
    );
    enc2.end_encoding();
    cb2.commit();
    cb2.wait_until_completed();

    if cb2.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            format!("{propagate_kernel} failed"),
        ));
    }

    Ok(output)
}

#[pyfunction]
pub fn metal_ffill(input: &MetalSeries) -> PyResult<MetalSeries> {
    if input.dtype != DType::Float32 {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "ffill currently only supports Float32, got {:?}", input.dtype
        )));
    }
    let buf = input.as_numeric_checked()?;
    let result = fill_scan_dispatch(
        buf.metal_buffer(), buf.len, "ffill_scan_float32", "ffill_propagate_float32",
    )?;
    let result_buf = SharedBuffer::from_metal_buffer(result, buf.len, DType::Float32);
    Ok(MetalSeries::from_numeric(result_buf))
}

#[pyfunction]
pub fn metal_bfill(input: &MetalSeries) -> PyResult<MetalSeries> {
    if input.dtype != DType::Float32 {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "bfill currently only supports Float32, got {:?}", input.dtype
        )));
    }
    let buf = input.as_numeric_checked()?;
    let result = fill_scan_dispatch(
        buf.metal_buffer(), buf.len, "bfill_scan_float32", "bfill_propagate_float32",
    )?;
    let result_buf = SharedBuffer::from_metal_buffer(result, buf.len, DType::Float32);
    Ok(MetalSeries::from_numeric(result_buf))
}
```

- [ ] **Step 3: Register module and pyfunctions**

In `rust/src/kernels/mod.rs`, add after `pub mod fillna;`:
```rust
pub mod fill_scan;
```

In `rust/src/lib.rs`, add imports:
```rust
use kernels::fill_scan::{metal_ffill, metal_bfill};
```

Add registrations:
```rust
    m.add_wrapped(wrap_pyfunction!(metal_ffill))?;
    m.add_wrapped(wrap_pyfunction!(metal_bfill))?;
```

- [ ] **Step 4: Build**

Run: `cd /Users/atlnguyen/personal_git/metaldf/rust && maturin develop --release 2>&1 | tail -5`
Expected: Build succeeds.

- [ ] **Step 5: Add Python wiring**

In `src/metaldf/_engine/_metal.py`, add dispatch functions:

```python
def _dispatch_ffill(data: Any) -> Any:
    if not _has_metal():
        raise MetalNotAvailable("Metal not available")

    from metaldf._wrappers import ProxySeries
    if isinstance(data, ProxySeries) and hasattr(data, "_pandas_obj"):
        data = data.to_pandas()

    if not hasattr(data, "dtype") or data.dtype != np.dtype(np.float32):
        raise MetalNotAvailable("ffill GPU only supports float32")

    arr = _extract_array(data)
    buf = _make_series(arr)
    result = metaldf_engine.metal_ffill(buf)
    return pd.Series(result.to_numpy(), index=data.index if hasattr(data, 'index') else None,
                     name=getattr(data, 'name', None))


def _dispatch_bfill(data: Any) -> Any:
    if not _has_metal():
        raise MetalNotAvailable("Metal not available")

    from metaldf._wrappers import ProxySeries
    if isinstance(data, ProxySeries) and hasattr(data, "_pandas_obj"):
        data = data.to_pandas()

    if not hasattr(data, "dtype") or data.dtype != np.dtype(np.float32):
        raise MetalNotAvailable("bfill GPU only supports float32")

    arr = _extract_array(data)
    buf = _make_series(arr)
    result = metaldf_engine.metal_bfill(buf)
    return pd.Series(result.to_numpy(), index=data.index if hasattr(data, 'index') else None,
                     name=getattr(data, 'name', None))
```

Add `MetalEngine` static methods:

```python
    @staticmethod
    def metal_ffill(data: Any) -> Any:
        return _dispatch_ffill(data)

    @staticmethod
    def metal_bfill(data: Any) -> Any:
        return _dispatch_bfill(data)
```

In `src/metaldf/_engine/__init__.py`, add:
```python
    register("ffill", MetalEngine.metal_ffill)
    register("bfill", MetalEngine.metal_bfill)
```

In `src/metaldf/_wrappers.py`, add to `ProxySeries` (after `fillna`):

```python
    def ffill(self, **kwargs: Any) -> Any:
        return self._try_metal_series_op("ffill")

    def bfill(self, **kwargs: Any) -> Any:
        return self._try_metal_series_op("bfill")

    def pad(self, **kwargs: Any) -> Any:
        return self.ffill(**kwargs)

    def backfill(self, **kwargs: Any) -> Any:
        return self.bfill(**kwargs)
```

- [ ] **Step 6: Write tests**

Create `tests/test_ffill_bfill.py`:

```python
"""Tests for GPU ffill and bfill."""

import numpy as np
import pandas as pd
import pytest

try:
    import metaldf_engine
    HAS_METAL = True
except ImportError:
    HAS_METAL = False

pytestmark = pytest.mark.skipif(not HAS_METAL, reason="Metal engine not built")

SCAN_TG_SIZE = 256


class TestFfillDirect:
    def test_gap_in_middle(self):
        arr = np.array([1.0, np.nan, np.nan, 4.0, np.nan], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_ffill(ms).to_numpy()
        expected = np.array([1.0, 1.0, 1.0, 4.0, 4.0], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_nan_at_start(self):
        arr = np.array([np.nan, np.nan, 3.0, 4.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_ffill(ms).to_numpy()
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        np.testing.assert_allclose(result[2:], [3.0, 4.0], rtol=1e-5)

    def test_no_nan_identity(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_ffill(ms).to_numpy()
        np.testing.assert_allclose(result, arr, rtol=1e-5)

    def test_all_nan(self):
        arr = np.array([np.nan, np.nan, np.nan], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_ffill(ms).to_numpy()
        assert all(np.isnan(result))

    def test_cross_threadgroup(self):
        arr = np.full(SCAN_TG_SIZE + 100, np.nan, dtype=np.float32)
        arr[0] = 42.0
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_ffill(ms).to_numpy()
        expected = np.full(SCAN_TG_SIZE + 100, 42.0, dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_large_multi_group(self):
        n = 10_000
        rng = np.random.default_rng(42)
        arr = rng.random(n).astype(np.float32)
        mask = rng.random(n) < 0.3
        arr[mask] = np.nan
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_ffill(ms).to_numpy()
        expected = pd.Series(arr).ffill().values
        # Compare only positions where expected is not NaN
        valid = ~np.isnan(expected)
        np.testing.assert_allclose(result[valid], expected[valid], rtol=1e-5)
        # Leading NaNs should remain NaN
        nan_expected = np.isnan(expected)
        assert np.all(np.isnan(result[nan_expected]))


class TestBfillDirect:
    def test_gap_in_middle(self):
        arr = np.array([1.0, np.nan, np.nan, 4.0, 5.0], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_bfill(ms).to_numpy()
        expected = np.array([1.0, 4.0, 4.0, 4.0, 5.0], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_nan_at_end(self):
        arr = np.array([1.0, 2.0, np.nan, np.nan], dtype=np.float32)
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_bfill(ms).to_numpy()
        np.testing.assert_allclose(result[:2], [1.0, 2.0], rtol=1e-5)
        assert np.isnan(result[2])
        assert np.isnan(result[3])

    def test_cross_threadgroup(self):
        arr = np.full(SCAN_TG_SIZE + 100, np.nan, dtype=np.float32)
        arr[-1] = 42.0
        ms = metaldf_engine.MetalSeries.from_numpy(arr)
        result = metaldf_engine.metal_bfill(ms).to_numpy()
        expected = np.full(SCAN_TG_SIZE + 100, 42.0, dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


class TestProxyFfillBfill:
    @pytest.fixture(autouse=True)
    def _install(self):
        import metaldf
        metaldf.install()
        yield
        metaldf.uninstall()

    def test_ffill(self):
        s = pd.Series([1.0, np.nan, np.nan, 4.0], dtype=np.float32)
        result = s.ffill()
        expected = pd.Series([1.0, 1.0, 1.0, 4.0], dtype=np.float32)
        np.testing.assert_allclose(result.values, expected.values, rtol=1e-5)

    def test_bfill(self):
        s = pd.Series([np.nan, np.nan, 3.0, 4.0], dtype=np.float32)
        result = s.bfill()
        expected = pd.Series([3.0, 3.0, 3.0, 4.0], dtype=np.float32)
        np.testing.assert_allclose(result.values, expected.values, rtol=1e-5)
```

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/test_ffill_bfill.py -v`
Expected: All tests PASS.

- [ ] **Step 8: Commit**

```bash
git add rust/metal/scan/fill_scan.metal rust/src/kernels/fill_scan.rs rust/src/kernels/mod.rs rust/src/lib.rs src/metaldf/_engine/_metal.py src/metaldf/_engine/__init__.py src/metaldf/_wrappers.py tests/test_ffill_bfill.py
git commit -m "feat: add GPU ffill/bfill via parallel scan for float32"
```

---

### Task 11: Update FEATURE_GAP.md + run full test suite

**Files:**
- Modify: `docs/FEATURE_GAP.md`

- [ ] **Step 1: Update FEATURE_GAP.md tracking table**

In `docs/FEATURE_GAP.md`, update these rows in the Operations Coverage table (Section 1):

Change the cumulative rows:
- `| Cumulative: cumsum | Internal only (uint32/int32) | Full | Missing (public) |` → `| Cumulative: cumsum | f32/i32/i64 | Full | Done |`
- `| Cumulative: cumprod/cummin/cummax | None | Full | Missing |` → `| Cumulative: cumprod | None | Full | Missing |` and add new row: `| Cumulative: cummin/cummax | f32/i32/i64 | Full | Done |`

Change the datetime rows:
- `| Datetime: quarter/dayofyear | None | Full | Missing |` → `| Datetime: quarter/dayofyear | Done | Full | Done |`

Change the missing data rows:
- `| Missing data: fillna | None | Full | Missing |` → `| Missing data: fillna | f32 scalar | Full | Partial |`
- `| Missing data: ffill/bfill | None | Full | Missing |` → `| Missing data: ffill/bfill | f32 | Full | Partial |`

Change the shift row:
- `| Shift/diff/pct_change | None | Full | Missing |` → `| Shift/diff/pct_change | f32/i32/i64 | Full | Done |`

Add a row to the Tracking Notes table (Section 5):

```
| 2026-07-16 | <commit> | P4, P9, P11, P20 | cumsum/cummin/cummax, quarter/dayofyear, shift/diff/pct_change, fillna, ffill/bfill |
```

- [ ] **Step 2: Run the full test suite**

Run:
```bash
python -m pytest tests/ -v --ignore=tests/exhaustive 2>&1 | tail -30
```
Expected: All tests pass including the new ones and all pre-existing ones.

- [ ] **Step 3: Commit**

```bash
git add docs/FEATURE_GAP.md
git commit -m "docs: update FEATURE_GAP.md with completed low-hanging fruit items"
```
