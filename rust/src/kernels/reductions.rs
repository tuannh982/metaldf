// Reduction kernel dispatch -- sum, min, max, mean on Metal GPU.
//
// Multi-pass tree reduction: keep reducing partials until a single scalar
// remains. Supports float32, int32, and int64 buffers (Datetime/Timedelta
// ride along on int64's kernel_suffix/read_scalar path); mean always returns
// a Python float (matching pandas semantics).

use pyo3::prelude::*;
use pyo3::IntoPy;

use metal::{MTLSize, MTLResourceOptions};

use crate::backend::MetalBackend;
use crate::buffer::{SharedBuffer, DType, NullMask};
use crate::kernels::{load_reductions_library, get_pipeline_state, tuning};
use crate::series::MetalSeries;

fn num_threadgroups(len: u64) -> u64 {
    let epg = tuning().elements_per_reduce_group();
    (len + epg - 1) / epg
}

fn dispatch_reduce_pass(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    pipeline: &metal::ComputePipelineState,
    input: &metal::Buffer,
    output: &metal::Buffer,
    len: u64,
    elem_size: u64,
) -> PyResult<()> {
    let num_groups = num_threadgroups(len);

    let len_buffer = device.new_buffer(
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        *(len_buffer.contents() as *mut u32) = len as u32;
    }

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(input), 0);
    enc.set_buffer(1, Some(output), 0);
    enc.set_buffer(2, Some(&len_buffer), 0);
    let tg_size = tuning().reduce_threadgroup_size;
    enc.set_threadgroup_memory_length(0, tg_size * elem_size);
    enc.dispatch_thread_groups(
        MTLSize::new(num_groups, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Reduction pass failed: Metal command buffer error"
        ));
    }
    Ok(())
}

/// Null-aware counterpart of `dispatch_reduce_pass`. Binds an extra `mask`
/// buffer at slot 2 (shifting `len_ptr` to slot 3) so the `_masked` kernel
/// variants in `reduce.metal` can skip null elements (treating them as
/// `Op::identity`) instead of reading them. Only ever used for the *first*
/// reduction pass: partials produced by a masked pass already have nulls
/// folded away, so every subsequent pass reduces them with the plain
/// (unmasked) kernel via `dispatch_reduce_pass`.
fn dispatch_reduce_pass_masked(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    pipeline: &metal::ComputePipelineState,
    input: &metal::Buffer,
    output: &metal::Buffer,
    mask: &metal::Buffer,
    len: u64,
    elem_size: u64,
) -> PyResult<()> {
    let num_groups = num_threadgroups(len);

    let len_buffer = device.new_buffer(
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        *(len_buffer.contents() as *mut u32) = len as u32;
    }

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(pipeline);
    enc.set_buffer(0, Some(input), 0);
    enc.set_buffer(1, Some(output), 0);
    enc.set_buffer(2, Some(mask), 0);
    enc.set_buffer(3, Some(&len_buffer), 0);
    let tg_size = tuning().reduce_threadgroup_size;
    enc.set_threadgroup_memory_length(0, tg_size * elem_size);
    enc.dispatch_thread_groups(
        MTLSize::new(num_groups, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Masked reduction pass failed: Metal command buffer error"
        ));
    }
    Ok(())
}

fn read_scalar(py: Python, buffer: &metal::Buffer, dtype: DType) -> PyResult<PyObject> {
    unsafe {
        match dtype {
            DType::Float32 => Ok((*(buffer.contents() as *const f32)).into_py(py)),
            DType::Int8 => Ok((*(buffer.contents() as *const i8)).into_py(py)),
            DType::Int16 => Ok((*(buffer.contents() as *const i16)).into_py(py)),
            DType::Int32 => Ok((*(buffer.contents() as *const i32)).into_py(py)),
            DType::Int64 | DType::Datetime | DType::Timedelta => Ok((*(buffer.contents() as *const i64)).into_py(py)),
            DType::Uint8 => Ok((*(buffer.contents() as *const u8)).into_py(py)),
            DType::Uint16 => Ok((*(buffer.contents() as *const u16)).into_py(py)),
            DType::Uint32 => Ok((*(buffer.contents() as *const u32)).into_py(py)),
            DType::Uint64 => Ok((*(buffer.contents() as *const u64)).into_py(py)),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                format!("Reduction not supported for {:?}", dtype)
            )),
        }
    }
}

/// Orchestrates the multi-pass tree reduction for `sum`/`min`/`max`.
///
/// When `mask` is `Some`, the *first* pass dispatches the `_masked` kernel
/// variant (null elements folded into `Op::identity` instead of being read)
/// via `dispatch_reduce_pass_masked`; every subsequent pass reduces the
/// resulting (null-free) partials with the plain kernel, same as the
/// no-mask path. If every element is null (`count_valid() == 0`), the GPU
/// dispatch is skipped entirely and `f64::NAN` is returned directly,
/// matching pandas' all-null reduction semantics.
fn dispatch_reduction(
    py: Python,
    op_name: &str,
    data: &SharedBuffer,
    mask: Option<&NullMask>,
) -> PyResult<PyObject> {
    let dtype = data.dtype;
    match dtype {
        DType::Float32 | DType::Int8 | DType::Int16 | DType::Int32 | DType::Int64
        | DType::Uint8 | DType::Uint16 | DType::Uint32 | DType::Uint64
        | DType::Datetime | DType::Timedelta => {}
        _ => return Err(pyo3::exceptions::PyTypeError::new_err(
            format!("Reduction not supported for {:?}", dtype)
        )),
    }

    if data.len == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot reduce an empty array"
        ));
    }

    if let Some(m) = mask {
        if m.count_valid() == 0 {
            return Ok(f64::NAN.into_py(py));
        }
    }

    let elem_size = dtype.size_in_bytes() as u64;
    let kernel_name = format!("reduce_{}_{}", dtype.kernel_suffix(), op_name);

    let device = MetalBackend::device()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal device"))?;
    let queue = MetalBackend::queue()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal queue"))?;
    let library = load_reductions_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let pipeline = get_pipeline_state(device, &library, &kernel_name)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let len = data.len as u64;
    let num_groups = num_threadgroups(len);

    let partials = device.new_buffer(
        num_groups * elem_size,
        MTLResourceOptions::StorageModeShared,
    );

    if let Some(m) = mask {
        let masked_kernel_name = format!("reduce_{}_{}_masked", dtype.kernel_suffix(), op_name);
        let masked_pipeline = get_pipeline_state(device, &library, &masked_kernel_name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        dispatch_reduce_pass_masked(
            device, queue, &masked_pipeline,
            data.metal_buffer(), &partials, m.metal_buffer(),
            len, elem_size,
        )?;
    } else {
        dispatch_reduce_pass(device, queue, &pipeline, data.metal_buffer(), &partials, len, elem_size)?;
    }

    let mut current_len = num_groups;
    let mut src = partials;

    while current_len > 1 {
        let next_groups = num_threadgroups(current_len);
        let dst = device.new_buffer(
            next_groups * elem_size,
            MTLResourceOptions::StorageModeShared,
        );
        dispatch_reduce_pass(device, queue, &pipeline, &src, &dst, current_len, elem_size)?;
        current_len = next_groups;
        src = dst;
    }

    read_scalar(py, &src, dtype)
}

#[pyfunction]
pub fn metal_sum(py: Python, data: &MetalSeries) -> PyResult<PyObject> {
    let buf = data.as_numeric_checked()?;
    match buf.dtype {
        DType::Int8 | DType::Int16 | DType::Int32
        | DType::Uint8 | DType::Uint16 => {
            let sum = dispatch_widening_sum(buf)?;
            return Ok(sum.into_py(py));
        }
        _ => {}
    }
    dispatch_reduction(py, "sum", buf, data.null_mask.as_ref())
}

#[pyfunction]
pub fn metal_min(py: Python, data: &MetalSeries) -> PyResult<PyObject> {
    dispatch_reduction(py, "min", data.as_numeric_checked()?, data.null_mask.as_ref())
}

#[pyfunction]
pub fn metal_max(py: Python, data: &MetalSeries) -> PyResult<PyObject> {
    dispatch_reduction(py, "max", data.as_numeric_checked()?, data.null_mask.as_ref())
}

/// GPU widening sum: first pass reads narrow integers (8/16/32-bit), accumulates
/// as int64 via `reduce_widen_sum_{kernel_suffix}`. Subsequent passes reduce
/// int64 partials via `reduce_int64_sum`. Avoids overflow while staying fully
/// on GPU.
fn dispatch_widening_sum(data: &SharedBuffer) -> PyResult<i64> {
    let (device, queue) = MetalBackend::device_and_queue()?;
    let library = load_reductions_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let len = data.len as u64;
    let num_groups = num_threadgroups(len);

    let widen_name = format!("reduce_widen_sum_{}", data.dtype.kernel_suffix());
    let widen_pl = get_pipeline_state(device, &library, &widen_name)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let partials = device.new_buffer(num_groups * 8, MTLResourceOptions::StorageModeShared);
    dispatch_reduce_pass(device, queue, &widen_pl, data.metal_buffer(), &partials, len, 8)?;

    let i64_pl = get_pipeline_state(device, &library, "reduce_int64_sum")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let mut current_len = num_groups;
    let mut src = partials;
    while current_len > 1 {
        let next_groups = num_threadgroups(current_len);
        let dst = device.new_buffer(next_groups * 8, MTLResourceOptions::StorageModeShared);
        dispatch_reduce_pass(device, queue, &i64_pl, &src, &dst, current_len, 8)?;
        current_len = next_groups;
        src = dst;
    }

    Ok(unsafe { *(src.contents() as *const i64) })
}

#[pyfunction]
pub fn metal_mean(py: Python, data: &MetalSeries) -> PyResult<PyObject> {
    let buf = data.as_numeric_checked()?;
    if buf.len == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Cannot compute mean of empty array"));
    }

    // Null-aware path: sum only the valid elements on GPU, then divide by
    // the valid count (computed cheaply on CPU from the bitmask) rather
    // than `buf.len`. All-null returns NaN, matching pandas.
    if let Some(mask) = &data.null_mask {
        let valid_count = mask.count_valid();
        if valid_count == 0 {
            return Ok(f64::NAN.into_py(py));
        }
        let sum_obj = dispatch_reduction(py, "sum", buf, Some(mask))?;
        let sum: f64 = sum_obj.extract(py)?;
        return Ok((sum / valid_count as f64).into_py(py));
    }

    let sum: f64 = match buf.dtype {
        DType::Int8 | DType::Int16 | DType::Int32 | DType::Uint8 | DType::Uint16 => {
            dispatch_widening_sum(buf)? as f64
        }
        _ => {
            let sum_obj = dispatch_reduction(py, "sum", buf, None)?;
            sum_obj.extract(py)?
        }
    };
    Ok((sum / buf.len as f64).into_py(py))
}
