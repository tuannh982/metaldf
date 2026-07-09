// Reduction kernel dispatch -- sum, min, max, mean on Metal GPU.
//
// Multi-pass tree reduction: keep reducing partials until a single scalar
// remains. Supports float32, int32, and int64 buffers; mean always returns
// a Python float (matching pandas semantics).

use pyo3::prelude::*;
use pyo3::IntoPy;

use metal::{MTLSize, MTLResourceOptions};

use crate::backend::MetalBackend;
use crate::buffer::{SharedBuffer, DType};
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

fn read_scalar(py: Python, buffer: &metal::Buffer, dtype: DType) -> PyResult<PyObject> {
    unsafe {
        match dtype {
            DType::Float32 => Ok((*(buffer.contents() as *const f32)).into_py(py)),
            DType::Int32 => Ok((*(buffer.contents() as *const i32)).into_py(py)),
            DType::Int64 => Ok((*(buffer.contents() as *const i64)).into_py(py)),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                format!("Reduction not supported for {:?}", dtype)
            )),
        }
    }
}

fn dispatch_reduction(
    py: Python,
    op_name: &str,
    data: &SharedBuffer,
) -> PyResult<PyObject> {
    let dtype = data.dtype;
    match dtype {
        DType::Float32 | DType::Int32 | DType::Int64 => {}
        _ => return Err(pyo3::exceptions::PyTypeError::new_err(
            format!("Reduction not supported for {:?}", dtype)
        )),
    }

    if data.len == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Cannot reduce an empty array"
        ));
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
    dispatch_reduce_pass(device, queue, &pipeline, data.metal_buffer(), &partials, len, elem_size)?;

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
    if buf.dtype == DType::Int32 {
        let sum = dispatch_widening_sum_int32(buf)?;
        return Ok(sum.into_py(py));
    }
    dispatch_reduction(py, "sum", buf)
}

#[pyfunction]
pub fn metal_min(py: Python, data: &MetalSeries) -> PyResult<PyObject> {
    dispatch_reduction(py, "min", data.as_numeric_checked()?)
}

#[pyfunction]
pub fn metal_max(py: Python, data: &MetalSeries) -> PyResult<PyObject> {
    dispatch_reduction(py, "max", data.as_numeric_checked()?)
}

/// GPU widening sum: first pass reads int32, accumulates as int64 via
/// reduce_widen_sum_int32. Subsequent passes reduce int64 partials via
/// reduce_int64_sum. Avoids int32 overflow while staying fully on GPU.
fn dispatch_widening_sum_int32(data: &SharedBuffer) -> PyResult<i64> {
    let (device, queue) = MetalBackend::device_and_queue()?;
    let library = load_reductions_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let len = data.len as u64;
    let num_groups = num_threadgroups(len);

    let widen_pl = get_pipeline_state(device, &library, "reduce_widen_sum_int32")
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
    let sum: f64 = match buf.dtype {
        DType::Int32 => dispatch_widening_sum_int32(buf)? as f64,
        _ => {
            let sum_obj = dispatch_reduction(py, "sum", buf)?;
            sum_obj.extract(py)?
        }
    };
    Ok((sum / buf.len as f64).into_py(py))
}
