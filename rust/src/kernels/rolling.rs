// Rolling window kernel dispatch — `series.rolling(window).sum()/.mean()/
// .min()/.max()/.count()` (Phase 7, Task 7.1).
//
// Naive parallel strategy only for now: each GPU thread computes exactly one
// output element by iterating over its own `window`-sized (or shorter, near
// the start of the series) slice of `data` (see `rust/metal/rolling/
// rolling.metal` for the kernel-level docs). The prefix-sum-based strategy
// for large windows (`window > 1024`) described in the Task 7.1 design is
// deferred to a follow-up task — this dispatches the naive kernel
// unconditionally, regardless of `window` size.
//
// f32 only for now (i32 rolling variants are deferred, same as the MSL
// kernels). `metal_rolling_mean` is a thin wrapper: it reuses the sum kernel
// then divides by each position's actual in-window count (via the count
// kernel) rather than a plain GPU `sum / window`, since early positions
// (before the window is fully filled) have fewer than `window` elements
// contributing to their sum — matching pandas' default `min_periods=1`
// behavior.
//
// Null-masking is NOT handled here: `min_periods` and non-NaN-aware
// `count()` are left to the Python layer (see Task 7.1 brief / design doc).

use pyo3::prelude::*;
use metal::{MTLSize, MTLResourceOptions};

use crate::backend::MetalBackend;
use crate::buffer::{SharedBuffer, DType};
use crate::kernels::{load_rolling_library, get_pipeline_state};
use crate::series::MetalSeries;

const THREADGROUP_SIZE: u64 = 256;

fn num_threadgroups(len: usize) -> u64 {
    (len as u64 + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE
}

fn check_rolling_dtype(dtype: DType) -> PyResult<()> {
    match dtype {
        DType::Float32 => Ok(()),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Rolling ops not supported for {:?} (only Float32 today)",
            dtype
        ))),
    }
}

fn check_window(window: usize) -> PyResult<()> {
    if window == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window must be >= 1",
        ));
    }
    Ok(())
}

/// [len, window] packed as two consecutive `uint32_t`s — matches the
/// `device const uint* params [[buffer(2)]]` layout every `rolling_*_f32`
/// kernel expects (`params[0]` = len, `params[1]` = window).
fn make_params_buffer(device: &metal::Device, len: usize, window: usize) -> metal::Buffer {
    let params: [u32; 2] = [len as u32, window as u32];
    device.new_buffer_with_data(
        params.as_ptr() as *const _,
        (2 * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// Dispatches a `rolling_{op}_f32` naive-parallel kernel: one thread per
/// output element, reading directly from `data`. Shared by
/// sum/min/max/count.
fn dispatch_rolling(op_name: &str, data: &SharedBuffer, window: usize) -> PyResult<SharedBuffer> {
    check_rolling_dtype(data.dtype)?;
    check_window(window)?;

    let (device, queue) = MetalBackend::device_and_queue()?;
    let len = data.len;
    let elem_size = data.dtype.size_in_bytes();

    let out_buf = device.new_buffer(
        (len.max(1) * elem_size) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Empty input: nothing to compute.
    if len == 0 {
        return Ok(SharedBuffer::from_metal_buffer(out_buf, 0, data.dtype));
    }

    let params_buf = make_params_buffer(device, len, window);

    let library = load_rolling_library(device)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    let kernel_name = format!("rolling_{op_name}_f32");
    let pipeline = get_pipeline_state(device, &library, &kernel_name)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(data.metal_buffer()), 0);
    enc.set_buffer(1, Some(&out_buf), 0);
    enc.set_buffer(2, Some(&params_buf), 0);
    enc.dispatch_thread_groups(
        MTLSize::new(num_threadgroups(len), 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "rolling_{op_name}_f32 kernel failed: Metal command buffer error"
        )));
    }

    Ok(SharedBuffer::from_metal_buffer(out_buf, len, data.dtype))
}

/// GPU rolling sum: `output[i] = sum(data[max(0, i - window + 1) ..= i])`.
#[pyfunction]
pub fn metal_rolling_sum(data: &MetalSeries, window: usize) -> PyResult<MetalSeries> {
    let buf = data.as_numeric_checked()?;
    let result = dispatch_rolling("sum", buf, window)?;
    Ok(MetalSeries::from_numeric(result))
}

/// GPU rolling min: `output[i] = min(data[max(0, i - window + 1) ..= i])`.
#[pyfunction]
pub fn metal_rolling_min(data: &MetalSeries, window: usize) -> PyResult<MetalSeries> {
    let buf = data.as_numeric_checked()?;
    let result = dispatch_rolling("min", buf, window)?;
    Ok(MetalSeries::from_numeric(result))
}

/// GPU rolling max: `output[i] = max(data[max(0, i - window + 1) ..= i])`.
#[pyfunction]
pub fn metal_rolling_max(data: &MetalSeries, window: usize) -> PyResult<MetalSeries> {
    let buf = data.as_numeric_checked()?;
    let result = dispatch_rolling("max", buf, window)?;
    Ok(MetalSeries::from_numeric(result))
}

/// GPU rolling count: `output[i] = min(i + 1, window)` — the number of
/// elements actually in-window at each position. Does NOT consult a null
/// mask (see module docs above): this is a plain window-size count, not a
/// non-NaN count.
#[pyfunction]
pub fn metal_rolling_count(data: &MetalSeries, window: usize) -> PyResult<MetalSeries> {
    let buf = data.as_numeric_checked()?;
    let result = dispatch_rolling("count", buf, window)?;
    Ok(MetalSeries::from_numeric(result))
}

/// GPU rolling mean: computed as rolling sum divided (elementwise, on CPU —
/// `len` scalar divisions, negligible next to the two GPU dispatches) by
/// each position's actual in-window count (from `rolling_count_f32`), not a
/// flat `window`, so early positions (before the window is fully filled)
/// match pandas' default `min_periods=1` behavior directly.
#[pyfunction]
pub fn metal_rolling_mean(data: &MetalSeries, window: usize) -> PyResult<MetalSeries> {
    let buf = data.as_numeric_checked()?;
    check_rolling_dtype(buf.dtype)?;
    check_window(window)?;

    let sum_buf = dispatch_rolling("sum", buf, window)?;
    let len = buf.len;

    if len == 0 {
        return Ok(MetalSeries::from_numeric(sum_buf));
    }

    let count_buf = dispatch_rolling("count", buf, window)?;

    let device = MetalBackend::device()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal device"))?;
    let out_buf = device.new_buffer(
        (len * std::mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    unsafe {
        let sum_ptr = sum_buf.metal_buffer().contents() as *const f32;
        let count_ptr = count_buf.metal_buffer().contents() as *const f32;
        let out_ptr = out_buf.contents() as *mut f32;
        for i in 0..len {
            let count = *count_ptr.add(i);
            *out_ptr.add(i) = if count > 0.0 { *sum_ptr.add(i) / count } else { f32::NAN };
        }
    }

    let result = SharedBuffer::from_metal_buffer(out_buf, len, DType::Float32);
    Ok(MetalSeries::from_numeric(result))
}
