// eval_expression -- dispatches the bytecode-interpreter Metal kernel
// (`eval_expression_f32`, see rust/metal/expression/eval.metal) that
// evaluates a stack-based program of column loads / scalar loads / binary
// and unary ops over up to 8 input columns, one thread per row.
//
// Dispatch uses `dispatch_threads` (exact grid size = element count) rather
// than `dispatch_thread_groups` with a threadgroup-padded grid: like the
// elementwise kernels (see `crate::kernels::elementwise`), the interpreter
// kernel has no `idx >= len` bounds guard, so a padded grid would read/write
// out of bounds for lengths that aren't a multiple of the threadgroup size.

use pyo3::prelude::*;
use pyo3::IntoPy;
use metal::{MTLSize, MTLResourceOptions};

use crate::backend::MetalBackend;
use crate::buffer::{SharedBuffer, DType};
use crate::kernels::{load_expression_library, load_reductions_library, get_pipeline_state, tuning};
use crate::series::MetalSeries;

const THREADGROUP_SIZE: u64 = 256;

/// Evaluate a bytecode `program` (see rust/metal/expression/eval.metal for
/// the opcode set) against up to 8 input `columns`, producing a new
/// `MetalSeries` of `size` f32 elements.
#[pyfunction]
pub fn eval_expression(
    program: Vec<u8>,
    columns: Vec<PyRef<MetalSeries>>,
    size: usize,
) -> PyResult<MetalSeries> {
    if size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Cannot evaluate empty expression"));
    }
    if columns.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("No input columns"));
    }
    if columns.len() > 8 {
        return Err(pyo3::exceptions::PyValueError::new_err("Max 8 input columns"));
    }

    let (device, queue) = MetalBackend::device_and_queue()?;
    let library = load_expression_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let pipeline = get_pipeline_state(device, &library, "eval_expression_f32")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    // Allocate output buffer (float32).
    let out_buf = device.new_buffer(
        (size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Upload program bytecode.
    let prog_buf = device.new_buffer_with_data(
        program.as_ptr() as *const _,
        program.len().max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Upload program length as a uint.
    let prog_len = program.len() as u32;
    let len_buf = device.new_buffer_with_data(
        &prog_len as *const u32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);

    // Bind input columns (buffers 0-7). For unused slots, bind the first
    // column's buffer -- the kernel unconditionally reads `cols[N]` into a
    // local array for all N in 0..8, but the program bytecode never
    // references slots beyond `columns.len()`, so the fallback value is
    // never actually used.
    let fallback_buf = columns[0].as_numeric_checked()?.metal_buffer();
    for i in 0..8u64 {
        if (i as usize) < columns.len() {
            let col_buf = columns[i as usize].as_numeric_checked()?.metal_buffer();
            enc.set_buffer(i, Some(col_buf), 0);
        } else {
            enc.set_buffer(i, Some(fallback_buf), 0);
        }
    }

    // Buffer 8: output.
    enc.set_buffer(8, Some(&out_buf), 0);
    // Buffer 9: program bytecode.
    enc.set_buffer(9, Some(&prog_buf), 0);
    // Buffer 10: program length.
    enc.set_buffer(10, Some(&len_buf), 0);

    let tg_size = THREADGROUP_SIZE.min(size as u64);
    enc.dispatch_threads(
        MTLSize::new(size as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Expression evaluation failed"
        ));
    }

    let result_buf = SharedBuffer::from_metal_buffer(out_buf, size, DType::Float32);
    Ok(MetalSeries::from_numeric(result_buf))
}

/// Evaluate a bytecode `program` over up to 8 input `columns` and reduce the
/// per-element results to a single scalar with `op` ("sum", "min", or "max"),
/// without ever materializing the full-length intermediate expression result.
///
/// Pass 1 dispatches `eval_reduce_{op}_f32` (see
/// rust/metal/expression/eval_reduce.metal) with `dispatch_thread_groups`:
/// unlike `eval_expression`'s kernel, this one has an explicit `gid <
/// data_len` bounds guard (so out-of-range threads contribute the op's
/// identity element instead of reading OOB), because it needs
/// `threadgroup_position_in_grid` to write one partial per threadgroup.
/// Passes 2..N feed those partials through the existing `reduce_float32_{op}`
/// tree-reduction kernels (see `reductions.rs`) until a single scalar
/// remains.
#[pyfunction]
pub fn eval_expression_reduce(
    py: Python,
    op: &str,
    program: Vec<u8>,
    columns: Vec<PyRef<MetalSeries>>,
    size: usize,
) -> PyResult<PyObject> {
    if size == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Cannot reduce empty expression"));
    }
    if columns.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("No input columns"));
    }
    if columns.len() > 8 {
        return Err(pyo3::exceptions::PyValueError::new_err("Max 8 input columns"));
    }

    let kernel_name = format!("eval_reduce_{}_f32", op);
    let (device, queue) = MetalBackend::device_and_queue()?;
    let library = load_expression_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let pipeline = get_pipeline_state(device, &library, &kernel_name)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    // Upload program bytecode.
    let prog_buf = device.new_buffer_with_data(
        program.as_ptr() as *const _,
        program.len().max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let prog_len = program.len() as u32;
    let prog_len_buf = device.new_buffer_with_data(
        &prog_len as *const u32 as *const _,
        4, MTLResourceOptions::StorageModeShared,
    );
    let data_len = size as u32;
    let data_len_buf = device.new_buffer_with_data(
        &data_len as *const u32 as *const _,
        4, MTLResourceOptions::StorageModeShared,
    );

    let tg_size = THREADGROUP_SIZE;
    let num_groups = (size as u64 + tg_size - 1) / tg_size;

    // Allocate partials buffer (one float32 per threadgroup).
    let partials = device.new_buffer(
        num_groups * 4,
        MTLResourceOptions::StorageModeShared,
    );

    // Pass 1: fused expression-evaluate + reduce.
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);

    // Bind input columns (buffers 0-7); see `eval_expression` above for why
    // unused slots can safely fall back to column 0's buffer.
    let fallback_buf = columns[0].as_numeric_checked()?.metal_buffer();
    for i in 0..8u64 {
        if (i as usize) < columns.len() {
            let col_buf = columns[i as usize].as_numeric_checked()?.metal_buffer();
            enc.set_buffer(i, Some(col_buf), 0);
        } else {
            enc.set_buffer(i, Some(fallback_buf), 0);
        }
    }
    enc.set_buffer(8, Some(&partials), 0);
    enc.set_buffer(9, Some(&prog_buf), 0);
    enc.set_buffer(10, Some(&prog_len_buf), 0);
    enc.set_buffer(11, Some(&data_len_buf), 0);
    enc.set_threadgroup_memory_length(0, tg_size * 4);

    enc.dispatch_thread_groups(
        MTLSize::new(num_groups, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Fused expression-reduce failed"
        ));
    }

    // Passes 2..N: reduce the per-threadgroup partials down to one scalar
    // using the existing float32 tree-reduction kernels.
    let reduce_library = load_reductions_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let reduce_kernel = format!("reduce_float32_{}", op);
    let reduce_pipeline = get_pipeline_state(device, &reduce_library, &reduce_kernel)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let reduce_tg = tuning().reduce_threadgroup_size;
    let reduce_n_reads = tuning().reduce_n_reads;
    let mut current_len = num_groups;
    let mut src = partials;

    while current_len > 1 {
        let next_groups = (current_len + reduce_tg * reduce_n_reads - 1) / (reduce_tg * reduce_n_reads);
        let dst = device.new_buffer(
            next_groups * 4,
            MTLResourceOptions::StorageModeShared,
        );

        let len_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        unsafe { *(len_buf.contents() as *mut u32) = current_len as u32; }

        let cb2 = queue.new_command_buffer();
        let enc2 = cb2.new_compute_command_encoder();
        enc2.set_compute_pipeline_state(&reduce_pipeline);
        enc2.set_buffer(0, Some(&src), 0);
        enc2.set_buffer(1, Some(&dst), 0);
        enc2.set_buffer(2, Some(&len_buf), 0);
        enc2.set_threadgroup_memory_length(0, reduce_tg * 4);
        enc2.dispatch_thread_groups(
            MTLSize::new(next_groups, 1, 1),
            MTLSize::new(reduce_tg, 1, 1),
        );
        enc2.end_encoding();
        cb2.commit();
        cb2.wait_until_completed();

        if cb2.status() == metal::MTLCommandBufferStatus::Error {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Partial reduction pass failed"
            ));
        }

        current_len = next_groups;
        src = dst;
    }

    // Read back the final scalar.
    let result = unsafe { *(src.contents() as *const f32) };
    Ok(result.into_py(py))
}
