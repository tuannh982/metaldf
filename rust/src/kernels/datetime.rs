// Calendar-component extraction kernel dispatch — the Rust half of the
// `.dt` accessor (`.dt.year`/`.month`/`.day`/`.hour`/`.minute`/`.second`/
// `.dayofweek`). See `rust/metal/datetime/` for the MSL kernels and the
// floor-division/civil-calendar helpers they rely on.
//
// Each of the seven kernels takes one `device const long*` (int64
// nanoseconds-since-epoch) buffer and produces one `device int*` (int32
// component) buffer, dispatched with `dispatch_thread_groups` (grid padded
// to a threadgroup multiple) plus the `idx >= len` bounds guard the MSL
// kernels each contain.

use pyo3::prelude::*;
use metal::{MTLSize, MTLResourceOptions};

use crate::backend::MetalBackend;
use crate::buffer::{SharedBuffer, DType};
use crate::kernels::{load_datetime_library, get_pipeline_state};
use crate::series::MetalSeries;

const THREADGROUP_SIZE: u64 = 256;

fn dispatch_dt_extract(kernel_name: &str, data: &MetalSeries) -> PyResult<MetalSeries> {
    if data.dtype != DType::Datetime {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Expected Datetime series, got {:?}", data.dtype
        )));
    }
    let buf = data.as_numeric_checked()?;
    let len = data.len;

    let (device, queue) = MetalBackend::device_and_queue()?;
    let library = load_datetime_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let pipeline = get_pipeline_state(device, &library, kernel_name)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let out_buf = device.new_buffer(
        (len.max(1) * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let len_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe { *(len_buf.contents() as *mut u32) = len as u32; }

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(buf.metal_buffer()), 0);
    enc.set_buffer(1, Some(&out_buf), 0);
    enc.set_buffer(2, Some(&len_buf), 0);

    if len > 0 {
        let num_groups = (len as u64 + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
        enc.dispatch_thread_groups(
            MTLSize::new(num_groups, 1, 1),
            MTLSize::new(THREADGROUP_SIZE, 1, 1),
        );
    }
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            format!("{} failed", kernel_name)
        ));
    }

    let result_buf = SharedBuffer::from_metal_buffer(out_buf, len, DType::Int32);
    Ok(MetalSeries::from_numeric(result_buf))
}

#[pyfunction]
pub fn metal_dt_year(data: &MetalSeries) -> PyResult<MetalSeries> {
    dispatch_dt_extract("dt_year_i64", data)
}

#[pyfunction]
pub fn metal_dt_month(data: &MetalSeries) -> PyResult<MetalSeries> {
    dispatch_dt_extract("dt_month_i64", data)
}

#[pyfunction]
pub fn metal_dt_day(data: &MetalSeries) -> PyResult<MetalSeries> {
    dispatch_dt_extract("dt_day_i64", data)
}

#[pyfunction]
pub fn metal_dt_hour(data: &MetalSeries) -> PyResult<MetalSeries> {
    dispatch_dt_extract("dt_hour_i64", data)
}

#[pyfunction]
pub fn metal_dt_minute(data: &MetalSeries) -> PyResult<MetalSeries> {
    dispatch_dt_extract("dt_minute_i64", data)
}

#[pyfunction]
pub fn metal_dt_second(data: &MetalSeries) -> PyResult<MetalSeries> {
    dispatch_dt_extract("dt_second_i64", data)
}

#[pyfunction]
pub fn metal_dt_dayofweek(data: &MetalSeries) -> PyResult<MetalSeries> {
    dispatch_dt_extract("dt_dayofweek_i64", data)
}

#[pyfunction]
pub fn metal_dt_quarter(data: &MetalSeries) -> PyResult<MetalSeries> {
    dispatch_dt_extract("dt_quarter_i64", data)
}

#[pyfunction]
pub fn metal_dt_dayofyear(data: &MetalSeries) -> PyResult<MetalSeries> {
    dispatch_dt_extract("dt_dayofyear_i64", data)
}
