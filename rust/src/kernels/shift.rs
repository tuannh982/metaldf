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
        DType::Int8 => Ok("int8"),
        DType::Int16 => Ok("int16"),
        DType::Int32 => Ok("int32"),
        DType::Int64 | DType::Datetime | DType::Timedelta => Ok("int64"),
        DType::Uint8 => Ok("uint8"),
        DType::Uint16 => Ok("uint16"),
        DType::Uint32 => Ok("uint32"),
        DType::Uint64 => Ok("uint64"),
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
