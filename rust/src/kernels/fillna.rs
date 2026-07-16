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
