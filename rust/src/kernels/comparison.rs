// Comparison kernel dispatch -- eq/ne/lt/le/gt/ge on Metal GPU.
//
// Lives in the elementwise library alongside binary/unary ops (see
// `rust/metal/elementwise/comparison.metal`) and follows the exact same
// dispatch shape as `crate::kernels::elementwise`: one thread per element,
// `dispatch_threads` (not `dispatch_thread_groups`) because the CMP_KERNEL
// macro (see `rust/metal/elementwise/01_types.h`) has no `idx >= len` bounds
// guard, so a threadgroup-padded grid would read/write out of bounds for
// lengths that aren't a multiple of the threadgroup size.
//
// Kernel names follow `cmp_{op}_{f32,i32,i64}`. Output is always Int32
// (0/1), regardless of input dtype.

use pyo3::prelude::*;
use metal::{MTLSize, MTLResourceOptions};

use crate::backend::MetalBackend;
use crate::buffer::{SharedBuffer, DType};
use crate::kernels::{load_elementwise_library, get_pipeline_state};
use crate::series::MetalSeries;

const THREADGROUP_SIZE: u64 = 256;

/// Maps `DType` to the short suffix used by comparison kernel names
/// (`f32`/`i32`/`i64`).
///
/// Int64, Datetime, and Timedelta are all backed by int64 storage (the
/// latter two as nanosecond counts), so they all route to the `i64`
/// kernels and are treated as mutually comparable.
///
/// TODO(datetime): `DType::Datetime` and `DType::Timedelta` are added by
/// the parallel "datetime dtype" task and aren't present in this worktree
/// yet. Once merged, add them here:
///     DType::Int64 | DType::Datetime | DType::Timedelta => Ok("i64"),
fn cmp_suffix(dtype: DType) -> PyResult<&'static str> {
    match dtype {
        DType::Float32 => Ok("f32"),
        DType::Int32 => Ok("i32"),
        DType::Int64 | DType::Datetime | DType::Timedelta => Ok("i64"),
        other => Err(pyo3::exceptions::PyTypeError::new_err(
            format!("Comparison not supported for dtype {:?}", other)
        )),
    }
}

/// Returns true if `a` and `b` should be treated as dtype-compatible for
/// comparison purposes.
///
/// Today this is just dtype equality. Once Task 1's `DType::Datetime` /
/// `DType::Timedelta` land, this should also accept any pairing within the
/// int64-backed family (Int64/Datetime/Timedelta all compare via the `i64`
/// kernels, see `cmp_suffix`'s TODO above), e.g.:
///     matches!(a, DType::Int64 | DType::Datetime | DType::Timedelta)
///         && matches!(b, DType::Int64 | DType::Datetime | DType::Timedelta)
fn dtypes_comparable(a: DType, b: DType) -> bool {
    if a == b { return true; }
    matches!(a, DType::Int64 | DType::Datetime | DType::Timedelta)
        && matches!(b, DType::Int64 | DType::Datetime | DType::Timedelta)
}

#[pyfunction]
pub fn metal_compare_op(op: &str, lhs: &MetalSeries, rhs: &MetalSeries) -> PyResult<MetalSeries> {
    let lhs_buf = lhs.as_numeric_checked()?;
    let rhs_buf = rhs.as_numeric_checked()?;

    if !dtypes_comparable(lhs_buf.dtype, rhs_buf.dtype) {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            format!("dtype mismatch: {:?} vs {:?}", lhs_buf.dtype, rhs_buf.dtype)
        ));
    }
    if lhs.len != rhs.len {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("length mismatch: {} vs {}", lhs.len, rhs.len)
        ));
    }

    let suffix = cmp_suffix(lhs_buf.dtype)?;
    let kernel_name = format!("cmp_{}_{}", op, suffix);
    let len = lhs.len;

    let (device, queue) = MetalBackend::device_and_queue()?;
    let library = load_elementwise_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let pipeline = get_pipeline_state(device, &library, &kernel_name)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let out_buf = device.new_buffer(
        (len.max(1) * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(lhs_buf.metal_buffer()), 0);
    enc.set_buffer(1, Some(rhs_buf.metal_buffer()), 0);
    enc.set_buffer(2, Some(&out_buf), 0);

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
        return Err(pyo3::exceptions::PyRuntimeError::new_err("Comparison kernel failed"));
    }

    let result_buf = SharedBuffer::from_metal_buffer(out_buf, len, DType::Int32);
    Ok(MetalSeries::from_numeric(result_buf))
}
