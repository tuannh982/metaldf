// Prefix-sum / scan kernel dispatch — GPU inclusive scan (cumulative sum).
//
// Two-pass algorithm (see `rust/metal/scan/scan.metal` for the kernel-level
// docs): pass 1 scans each `SCAN_TG_SIZE`-element chunk locally (in
// threadgroup shared memory) and emits one partial (that group's total) per
// threadgroup; the partials buffer is then recursively scanned (this
// function calling itself, bottoming out once a single threadgroup covers
// the whole buffer); pass 2 propagates each group's prefix (the scanned
// partial contributed by every group before it) into every element of that
// group. Group 0 has no predecessor and needs no propagation.
//
// Only `Int32`/`Uint32` are supported today (see Task 3.1 brief: this is an
// internal building block for Phase 4 filtering/boolean-indexing and
// Phase 7 rolling windows, not directly user-facing — float32/int64 support
// is deferred, and float64 isn't supported on Metal at all, per Task 2.1).

use pyo3::prelude::*;
use metal::{MTLSize, MTLResourceOptions};

use crate::backend::MetalBackend;
use crate::buffer::{SharedBuffer, DType};
use crate::kernels::{load_scan_library, get_pipeline_state};
use crate::series::MetalSeries;

const SCAN_TG_SIZE: u64 = 256;

fn check_scan_dtype(dtype: DType) -> PyResult<()> {
    match dtype {
        DType::Int32 | DType::Uint32 => Ok(()),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Prefix sum not supported for {:?} (only Int32/Uint32 today)",
            dtype
        ))),
    }
}

/// Computes the GPU inclusive prefix sum (cumulative sum) of `input` (`len`
/// elements of `dtype`), returning a freshly allocated buffer of the same
/// length and dtype. Internal-only — not exposed to Python directly; see
/// `metal_prefix_sum` below for the pyfunction wrapper.
pub fn prefix_sum_inclusive(
    input: &metal::Buffer,
    len: usize,
    dtype: DType,
) -> PyResult<metal::Buffer> {
    check_scan_dtype(dtype)?;

    let (device, queue) = MetalBackend::device_and_queue()?;
    let elem_size = dtype.size_in_bytes() as u64;

    // Empty input: nothing to scan. Matches np.cumsum(empty) -> empty.
    if len == 0 {
        return Ok(device.new_buffer(elem_size, MTLResourceOptions::StorageModeShared));
    }

    let library = load_scan_library(device)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    let suffix = dtype.kernel_suffix();

    // len >= 1 here, so num_groups >= 1 always.
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

    // Pass 1: per-threadgroup local scan, writing scanned output plus one
    // partial (group total) per threadgroup.
    let scan_pl = get_pipeline_state(device, &library, &format!("scan_inclusive_{suffix}"))
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
            "Scan pass 1 failed: Metal command buffer error",
        ));
    }

    if num_groups <= 1 {
        return Ok(output);
    }

    // Recursively scan the per-group partials: turns per-group totals into
    // per-group inclusive prefixes over all groups up to and including it.
    let scanned_partials = prefix_sum_inclusive(&partials, num_groups as usize, dtype)?;

    // Pass 2: propagate each group's exclusive prefix (the prior group's
    // scanned partial) into every element of that group.
    let prop_pl = get_pipeline_state(device, &library, &format!("scan_propagate_{suffix}"))
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
            "Scan pass 2 (propagate) failed: Metal command buffer error",
        ));
    }

    Ok(output)
}

/// Python-facing GPU inclusive prefix sum (`cumsum`). Supports `Int32`- and
/// `Uint32`-dtype series today (see module docs above).
#[pyfunction]
pub fn metal_prefix_sum(input: &MetalSeries) -> PyResult<MetalSeries> {
    let buf = input.as_numeric_checked()?;
    let result = prefix_sum_inclusive(buf.metal_buffer(), buf.len, buf.dtype)?;
    let result_buf = SharedBuffer::from_metal_buffer(result, buf.len, buf.dtype);
    Ok(MetalSeries::from_numeric(result_buf))
}
