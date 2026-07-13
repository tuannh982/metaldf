// Elementwise kernel dispatch -- binary (add/sub/mul/div/mod) and unary
// (abs/neg/sqrt/exp/log/ceil/floor) ops on Metal GPU.
//
// One thread per element. Kernel names follow `binary_{op}_{f32,i32,i64}`
// and `unary_{op}_{f32,i32}` (see rust/metal/elementwise/*.metal) — note
// this differs from `DType::kernel_suffix()` (`float32`/`int32`/`int64`),
// which is used elsewhere (reductions/sort/groupby) for a different naming
// convention, so a local `metal_suffix()` maps dtype -> the short suffix
// these kernels use.
//
// Dispatch uses `dispatch_threads` (exact grid size = element count) rather
// than `dispatch_thread_groups` with a threadgroup-padded grid: the
// elementwise kernels have no `idx >= len` guard, so padding the grid up to
// a threadgroup multiple (e.g. 256) would read/write out of bounds for
// lengths that aren't a multiple of the threadgroup size. Non-uniform
// threadgroup dispatch is supported on all Apple Silicon GPUs.

use pyo3::prelude::*;
use metal::{MTLSize, MTLResourceOptions};

use crate::backend::{BatchContext, MetalBackend};
use crate::buffer::{SharedBuffer, DType};
use crate::kernels::{load_elementwise_library, get_pipeline_state};
use crate::series::MetalSeries;

const THREADGROUP_SIZE: u64 = 256;

/// Maps `DType` to the short suffix used by elementwise kernel names
/// (`f32`/`i32`/`i64`), as opposed to `DType::kernel_suffix()`'s
/// `float32`/`int32`/`int64` used by other kernel families.
fn metal_suffix(dtype: DType) -> PyResult<&'static str> {
    match dtype {
        DType::Float32 => Ok("f32"),
        DType::Int32 => Ok("i32"),
        DType::Int64 => Ok("i64"),
        other => Err(pyo3::exceptions::PyTypeError::new_err(
            format!("Elementwise ops not supported for dtype {:?}", other)
        )),
    }
}

/// Dispatch a compute kernel with one thread per element (`len` threads
/// total, no padding), reading `inputs` buffers and writing into `out_buf`.
fn dispatch_elementwise(
    kernel_name: &str,
    inputs: &[&metal::Buffer],
    out_buf: &metal::Buffer,
    len: usize,
) -> PyResult<()> {
    let (device, queue) = MetalBackend::device_and_queue()?;
    let library = load_elementwise_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let pipeline = get_pipeline_state(device, &library, kernel_name)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    for (i, buf) in inputs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(buf), 0);
    }
    enc.set_buffer(inputs.len() as u64, Some(out_buf), 0);

    if len > 0 {
        let tg_size = THREADGROUP_SIZE.min(len as u64);
        enc.dispatch_threads(
            MTLSize::new(len as u64, 1, 1),
            MTLSize::new(tg_size, 1, 1),
        );
    }
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            format!("{} failed: Metal command buffer error", kernel_name)
        ));
    }
    Ok(())
}

fn dispatch_binary_inner(
    op: &str,
    lhs: &MetalSeries,
    rhs: &MetalSeries,
) -> PyResult<MetalSeries> {
    let lhs_buf = lhs.as_numeric_checked()?;
    let rhs_buf = rhs.as_numeric_checked()?;

    if lhs.dtype != rhs.dtype {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            format!("dtype mismatch: {:?} vs {:?}", lhs.dtype, rhs.dtype)
        ));
    }
    if lhs.len != rhs.len {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("length mismatch: {} vs {}", lhs.len, rhs.len)
        ));
    }

    let dtype = lhs.dtype;
    let len = lhs.len;
    let elem_size = dtype.size_in_bytes();
    let kernel_name = format!("binary_{}_{}", op, metal_suffix(dtype)?);

    let device = MetalBackend::device()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal device"))?;
    let out_buf = device.new_buffer(
        (len.max(1) * elem_size) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    dispatch_elementwise(
        &kernel_name,
        &[lhs_buf.metal_buffer(), rhs_buf.metal_buffer()],
        &out_buf,
        len,
    )?;

    let result_buf = SharedBuffer::from_metal_buffer(out_buf, len, dtype);
    Ok(MetalSeries::from_numeric(result_buf))
}

fn dispatch_unary_inner(
    op: &str,
    input: &MetalSeries,
) -> PyResult<MetalSeries> {
    let in_buf = input.as_numeric_checked()?;
    let dtype = input.dtype;
    let len = input.len;
    let elem_size = dtype.size_in_bytes();
    let kernel_name = format!("unary_{}_{}", op, metal_suffix(dtype)?);

    let device = MetalBackend::device()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal device"))?;
    let out_buf = device.new_buffer(
        (len.max(1) * elem_size) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    dispatch_elementwise(
        &kernel_name,
        &[in_buf.metal_buffer()],
        &out_buf,
        len,
    )?;

    let result_buf = SharedBuffer::from_metal_buffer(out_buf, len, dtype);
    Ok(MetalSeries::from_numeric(result_buf))
}

/// Same as `dispatch_binary_inner`, but encodes into an existing
/// `BatchContext` (via `encode_threads`) instead of creating, committing,
/// and waiting on its own command buffer -- letting callers batch several
/// dispatches into one command-buffer submission (one commit/wait for the
/// whole batch, via `BatchContext::commit_and_wait`).
fn dispatch_binary_batched_inner(
    op: &str,
    lhs: &MetalSeries,
    rhs: &MetalSeries,
    batch: &BatchContext,
) -> PyResult<MetalSeries> {
    let lhs_buf = lhs.as_numeric_checked()?;
    let rhs_buf = rhs.as_numeric_checked()?;

    if lhs.dtype != rhs.dtype {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            format!("dtype mismatch: {:?} vs {:?}", lhs.dtype, rhs.dtype)
        ));
    }
    if lhs.len != rhs.len {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("length mismatch: {} vs {}", lhs.len, rhs.len)
        ));
    }

    let dtype = lhs.dtype;
    let len = lhs.len;
    let elem_size = dtype.size_in_bytes();
    let kernel_name = format!("binary_{}_{}", op, metal_suffix(dtype)?);

    let device = MetalBackend::device()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal device"))?;
    let library = load_elementwise_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let pipeline = get_pipeline_state(device, &library, &kernel_name)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let out_buf = device.new_buffer(
        (len.max(1) * elem_size) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    if len > 0 {
        let tg_size = THREADGROUP_SIZE.min(len as u64);
        batch.encode_threads(
            &pipeline,
            &[
                (lhs_buf.metal_buffer(), 0),
                (rhs_buf.metal_buffer(), 0),
                (&out_buf, 0),
            ],
            MTLSize::new(len as u64, 1, 1),
            MTLSize::new(tg_size, 1, 1),
        );
    }

    let result_buf = SharedBuffer::from_metal_buffer(out_buf, len, dtype);
    Ok(MetalSeries::from_numeric(result_buf))
}

#[pyfunction]
pub fn metal_binary_op(op: &str, lhs: &MetalSeries, rhs: &MetalSeries) -> PyResult<MetalSeries> {
    dispatch_binary_inner(op, lhs, rhs)
}

/// Batched variant of `metal_binary_op`: encodes the dispatch into `batch`
/// (see `BatchContext::encode_threads`) instead of committing its own
/// command buffer, so callers can chain several elementwise ops and pay a
/// single commit/wait round-trip via `metaldf_engine.batch_commit`.
#[pyfunction]
pub fn metal_binary_op_batched(
    op: &str, lhs: &MetalSeries, rhs: &MetalSeries, batch: &BatchContext,
) -> PyResult<MetalSeries> {
    dispatch_binary_batched_inner(op, lhs, rhs, batch)
}

#[pyfunction]
pub fn metal_unary_op(op: &str, input: &MetalSeries) -> PyResult<MetalSeries> {
    dispatch_unary_inner(op, input)
}
