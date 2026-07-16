// Forward-fill (ffill) / backward-fill (bfill) via parallel scan — same
// two-pass Hillis-Steele + recursive partial propagation architecture as
// `crate::kernels::scan::cumulative_scan` (see that module's doc comment for
// the full algorithm write-up), but with a NaN-conditional combine ("take
// the left value unless it's NaN, in which case take the right") instead of
// a numeric op. ffill/bfill kernels live in the same `scan.metal` MSL
// library (see `rust/metal/scan/fill_scan.metal`), so dispatch here reuses
// `load_scan_library`.
//
// bfill is implemented by mirroring indices (`len - 1 - base`) on both read
// and write within the same kernel that does ffill's forward pass — see the
// kernel-level docs in `rust/metal/scan/fill_scan.metal`.

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
