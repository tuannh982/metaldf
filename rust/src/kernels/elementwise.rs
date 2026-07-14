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
use crate::buffer::{SharedBuffer, DType, NullMask};
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

/// Dispatch a null-aware `_masked` kernel variant with one thread per
/// element. Buffer bindings follow `BINARY_KERNEL_MASKED`/
/// `UNARY_KERNEL_MASKED` in `rust/metal/elementwise/01_types.h`: `inputs`
/// bind to buffer(0..), then `out_buf`, then `masks` (one per input operand,
/// in the same order), then `valid_out` last.
///
/// An entry in `masks` is `None` when that operand has no null mask —
/// leaving that buffer slot unbound makes the shader-side pointer `nullptr`,
/// which `is_valid()` (see `common/04_null_mask.h`) treats as "always
/// valid", so a partially-masked binary op (one operand nullable, the other
/// not) doesn't need a synthesized all-valid dummy buffer.
///
/// `valid_out` must be a `len`-byte buffer (one `uint8_t` per element, not
/// packed bits) — see the concurrency note on `BINARY_KERNEL_MASKED` for why
/// packed-bit output would race across threads.
fn dispatch_elementwise_masked(
    kernel_name: &str,
    inputs: &[&metal::Buffer],
    out_buf: &metal::Buffer,
    masks: &[Option<&metal::Buffer>],
    valid_out: &metal::Buffer,
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

    let mut idx = 0u64;
    for buf in inputs {
        enc.set_buffer(idx, Some(*buf), 0);
        idx += 1;
    }
    enc.set_buffer(idx, Some(out_buf), 0);
    idx += 1;
    for mask in masks {
        enc.set_buffer(idx, mask.map(|m| &**m), 0);
        idx += 1;
    }
    enc.set_buffer(idx, Some(valid_out), 0);

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

/// Pack a per-element `u8` validity buffer (0 = null, 1 = valid), as written
/// by a `_masked` kernel's `valid_out` argument, into a proper bit-packed
/// `NullMask` (1 bit/element — see `common/04_null_mask.h`). Done on the CPU
/// after the kernel completes: `valid_buf` lives in `StorageModeShared`
/// memory, so this is a plain sequential scan, and it's trivial next to the
/// cost of a kernel launch — this is what lets the masked kernels avoid GPU
/// atomics entirely (see the concurrency note on `BINARY_KERNEL_MASKED`).
fn pack_validity_to_mask(device: &metal::Device, valid_buf: &metal::Buffer, len: usize) -> NullMask {
    let valid_ptr = valid_buf.contents() as *const u8;
    let byte_len = (len + 7) / 8;
    let mut mask_bytes = vec![0u8; byte_len.max(1)];
    for i in 0..len {
        if unsafe { *valid_ptr.add(i) } != 0 {
            mask_bytes[i / 8] |= 1u8 << (i % 8);
        }
    }
    let mask_buf = device.new_buffer_with_data(
        mask_bytes.as_ptr() as *const _,
        byte_len.max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    NullMask::from_metal_buffer(mask_buf, len)
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
    let suffix = metal_suffix(dtype)?;

    let device = MetalBackend::device()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal device"))?;
    let out_buf = device.new_buffer(
        (len.max(1) * elem_size) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Fast path: neither operand carries a null mask -> dispatch the plain
    // (unmasked) kernel, identical to pre-null-support behavior/performance.
    if lhs.null_mask.is_none() && rhs.null_mask.is_none() {
        let kernel_name = format!("binary_{}_{}", op, suffix);
        dispatch_elementwise(
            &kernel_name,
            &[lhs_buf.metal_buffer(), rhs_buf.metal_buffer()],
            &out_buf,
            len,
        )?;

        let result_buf = SharedBuffer::from_metal_buffer(out_buf, len, dtype);
        return Ok(MetalSeries::from_numeric(result_buf));
    }

    // At least one operand has a null mask -> use the `_masked` kernel
    // variant. Operands without a mask pass `None`, which is treated as
    // "always valid" via `is_valid()`'s `nullptr` check.
    let kernel_name = format!("binary_{}_{}_masked", op, suffix);
    let valid_out = device.new_buffer(len.max(1) as u64, MTLResourceOptions::StorageModeShared);

    dispatch_elementwise_masked(
        &kernel_name,
        &[lhs_buf.metal_buffer(), rhs_buf.metal_buffer()],
        &out_buf,
        &[
            lhs.null_mask.as_ref().map(|m| m.metal_buffer()),
            rhs.null_mask.as_ref().map(|m| m.metal_buffer()),
        ],
        &valid_out,
        len,
    )?;

    let mask = pack_validity_to_mask(device, &valid_out, len);
    let result_buf = SharedBuffer::from_metal_buffer(out_buf, len, dtype);
    Ok(MetalSeries::from_numeric_with_mask(result_buf, mask))
}

fn dispatch_unary_inner(
    op: &str,
    input: &MetalSeries,
) -> PyResult<MetalSeries> {
    let in_buf = input.as_numeric_checked()?;
    let dtype = input.dtype;
    let len = input.len;
    let elem_size = dtype.size_in_bytes();
    let suffix = metal_suffix(dtype)?;

    let device = MetalBackend::device()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal device"))?;
    let out_buf = device.new_buffer(
        (len.max(1) * elem_size) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Fast path: no null mask on the input -> plain (unmasked) kernel.
    if input.null_mask.is_none() {
        let kernel_name = format!("unary_{}_{}", op, suffix);
        dispatch_elementwise(
            &kernel_name,
            &[in_buf.metal_buffer()],
            &out_buf,
            len,
        )?;

        let result_buf = SharedBuffer::from_metal_buffer(out_buf, len, dtype);
        return Ok(MetalSeries::from_numeric(result_buf));
    }

    let kernel_name = format!("unary_{}_{}_masked", op, suffix);
    let valid_out = device.new_buffer(len.max(1) as u64, MTLResourceOptions::StorageModeShared);

    dispatch_elementwise_masked(
        &kernel_name,
        &[in_buf.metal_buffer()],
        &out_buf,
        &[input.null_mask.as_ref().map(|m| m.metal_buffer())],
        &valid_out,
        len,
    )?;

    let mask = pack_validity_to_mask(device, &valid_out, len);
    let result_buf = SharedBuffer::from_metal_buffer(out_buf, len, dtype);
    Ok(MetalSeries::from_numeric_with_mask(result_buf, mask))
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

/// Verify `series` is `Bool`-dtype (the storage type for logical-op operands
/// and output — see `rust/metal/elementwise/logical.metal`), returning its
/// backing buffer on success.
fn check_bool_operand<'a>(series: &'a MetalSeries, op_name: &str) -> PyResult<&'a SharedBuffer> {
    if series.dtype != DType::Bool {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            format!("{} requires Bool dtype operands, got {:?}", op_name, series.dtype)
        ));
    }
    series.as_numeric_checked()
}

/// Shared dispatch for `metal_logical_and`/`metal_logical_or`: both operands
/// must be `Bool` dtype and the same length; result is a new `Bool` series.
fn dispatch_logical_binary(kernel_name: &str, a: &MetalSeries, b: &MetalSeries) -> PyResult<MetalSeries> {
    let a_buf = check_bool_operand(a, kernel_name)?;
    let b_buf = check_bool_operand(b, kernel_name)?;
    if a.len != b.len {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("length mismatch: {} vs {}", a.len, b.len)
        ));
    }

    let len = a.len;
    let device = MetalBackend::device()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal device"))?;
    let out_buf = device.new_buffer(len.max(1) as u64, MTLResourceOptions::StorageModeShared);

    dispatch_elementwise(kernel_name, &[a_buf.metal_buffer(), b_buf.metal_buffer()], &out_buf, len)?;

    let result_buf = SharedBuffer::from_metal_buffer(out_buf, len, DType::Bool);
    Ok(MetalSeries::from_numeric(result_buf))
}

/// Elementwise logical AND on two `Bool`-dtype series (see
/// `logical_and_bool` in `rust/metal/elementwise/logical.metal`).
#[pyfunction]
pub fn metal_logical_and(a: &MetalSeries, b: &MetalSeries) -> PyResult<MetalSeries> {
    dispatch_logical_binary("logical_and_bool", a, b)
}

/// Elementwise logical OR on two `Bool`-dtype series (see `logical_or_bool`
/// in `rust/metal/elementwise/logical.metal`).
#[pyfunction]
pub fn metal_logical_or(a: &MetalSeries, b: &MetalSeries) -> PyResult<MetalSeries> {
    dispatch_logical_binary("logical_or_bool", a, b)
}

/// Elementwise logical NOT on a `Bool`-dtype series (see `logical_not_bool`
/// in `rust/metal/elementwise/logical.metal`).
#[pyfunction]
pub fn metal_logical_not(input: &MetalSeries) -> PyResult<MetalSeries> {
    let in_buf = check_bool_operand(input, "logical_not_bool")?;
    let len = input.len;
    let device = MetalBackend::device()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal device"))?;
    let out_buf = device.new_buffer(len.max(1) as u64, MTLResourceOptions::StorageModeShared);

    dispatch_elementwise("logical_not_bool", &[in_buf.metal_buffer()], &out_buf, len)?;

    let result_buf = SharedBuffer::from_metal_buffer(out_buf, len, DType::Bool);
    Ok(MetalSeries::from_numeric(result_buf))
}
