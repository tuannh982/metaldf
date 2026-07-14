// Stream compaction (compact) and gather-by-index (take) kernel dispatch —
// the GPU building blocks for `df[mask]` boolean indexing (Phase 4).
//
// `metal_compact` filters a data series down to just the elements where a
// parallel `Bool` mask is `1`, preserving relative order. It works by first
// casting the `uint8_t` mask to `uint32_t` (0/1 per element) and running it
// through `prefix_sum_inclusive` (Task 3.1's GPU scan kernel): the resulting
// inclusive prefix sum gives each kept element its 1-based rank among kept
// elements so far (`prefix_sum[i] - 1` is its 0-based output slot), and the
// scan's last element is the total number of kept elements, used to size the
// output buffer before dispatching the `compact_{suffix}` kernel (see
// `rust/metal/filter/compact.metal`).
//
// `metal_take` is a plain GPU gather: `output[i] = data[indices[i]]`, no
// relation to the mask/prefix-sum machinery above.
//
// Both kernels use `dispatch_thread_groups` (grid padded to a threadgroup
// multiple) with an explicit `idx >= len` bounds guard, rather than
// `dispatch_threads` like the elementwise kernels — see the MSL file's docs.

use pyo3::prelude::*;
use metal::{MTLSize, MTLResourceOptions};

use crate::backend::MetalBackend;
use crate::buffer::{SharedBuffer, DType};
use crate::kernels::{load_filter_library, get_pipeline_state};
use crate::kernels::scan::prefix_sum_inclusive;
use crate::series::MetalSeries;

const THREADGROUP_SIZE: u64 = 256;

/// Maps `DType` to the short suffix used by compact/take kernel names
/// (`f32`/`i32`/`i64`/`u8`) — matches the elementwise kernel convention
/// (see `metal_suffix()` in `rust/src/kernels/elementwise.rs`), not
/// `DType::kernel_suffix()`'s `float32`/`int32`/`int64`/`uint8`. `Bool` data
/// (e.g. compacting/taking a boolean column itself) shares the `u8` kernel
/// with `Uint8`, since both are one `uint8_t` per element.
fn filter_suffix(dtype: DType) -> PyResult<&'static str> {
    match dtype {
        DType::Float32 => Ok("f32"),
        DType::Int32 => Ok("i32"),
        DType::Int64 | DType::Datetime | DType::Timedelta => Ok("i64"),
        DType::Uint8 | DType::Bool => Ok("u8"),
        other => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "compact/take not supported for dtype {:?}", other
        ))),
    }
}

/// Number of threadgroups needed to cover `len` elements at
/// `THREADGROUP_SIZE` threads/group (used with `dispatch_thread_groups`,
/// which the `idx >= len`-guarded compact/take kernels require).
fn num_threadgroups(len: usize) -> u64 {
    (len as u64 + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE
}

fn make_len_buffer(device: &metal::Device, len: usize) -> metal::Buffer {
    let len_u32 = len as u32;
    device.new_buffer_with_data(
        &len_u32 as *const u32 as *const _,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

/// GPU stream compaction: keeps only the elements of `data` where the
/// parallel `Bool` `mask` is `1`, preserving relative order. `mask` must be
/// `Bool`-dtype and the same length as `data`.
#[pyfunction]
pub fn metal_compact(data: &MetalSeries, mask: &MetalSeries) -> PyResult<MetalSeries> {
    let data_buf = data.as_numeric_checked()?;
    let mask_buf = mask.as_numeric_checked()?;

    if mask.dtype != DType::Bool {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "compact mask must be Bool dtype, got {:?}", mask.dtype
        )));
    }
    if data.len != mask.len {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "length mismatch: data has {} elements, mask has {}", data.len, mask.len
        )));
    }

    let dtype = data.dtype;
    let suffix = filter_suffix(dtype)?;
    let len = data.len;
    let elem_size = dtype.size_in_bytes();

    let device = MetalBackend::device()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal device"))?;

    // Empty input: nothing to compact. Matches np.array([])[mask] -> empty.
    if len == 0 {
        let out_buf = device.new_buffer(elem_size.max(1) as u64, MTLResourceOptions::StorageModeShared);
        let result_buf = SharedBuffer::from_metal_buffer(out_buf, 0, dtype);
        return Ok(MetalSeries::from_numeric(result_buf));
    }

    // Cast the uint8_t mask (0/1 per element) to uint32_t so it can be fed
    // through the Uint32 prefix-sum kernel. `mask_buf` is StorageModeShared,
    // so this is a plain CPU-side read -- cheap relative to the GPU scan and
    // compact dispatches this feeds.
    let mask_ptr = mask_buf.metal_buffer().contents() as *const u8;
    let mut mask_u32: Vec<u32> = Vec::with_capacity(len);
    for i in 0..len {
        mask_u32.push(unsafe { *mask_ptr.add(i) } as u32);
    }
    let mask_u32_buf = device.new_buffer_with_data(
        mask_u32.as_ptr() as *const _,
        (len * std::mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Inclusive prefix sum over the uint32 mask -> 1-based output rank for
    // each kept element; the last element is the total number kept.
    let prefix_sum = prefix_sum_inclusive(&mask_u32_buf, len, DType::Uint32)?;
    let total_count = unsafe { *(prefix_sum.contents() as *const u32).add(len - 1) } as usize;

    let out_buf = device.new_buffer(
        (total_count.max(1) * elem_size) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let len_buf = make_len_buffer(device, len);

    let library = load_filter_library(device)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    let kernel_name = format!("compact_{suffix}");
    let pipeline = get_pipeline_state(device, &library, &kernel_name)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let queue = MetalBackend::queue()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal queue"))?;
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(data_buf.metal_buffer()), 0);
    enc.set_buffer(1, Some(mask_buf.metal_buffer()), 0);
    enc.set_buffer(2, Some(&prefix_sum), 0);
    enc.set_buffer(3, Some(&out_buf), 0);
    enc.set_buffer(4, Some(&len_buf), 0);
    enc.dispatch_thread_groups(
        MTLSize::new(num_threadgroups(len), 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "compact kernel failed: Metal command buffer error"
        ));
    }

    let result_buf = SharedBuffer::from_metal_buffer(out_buf, total_count, dtype);
    Ok(MetalSeries::from_numeric(result_buf))
}

/// GPU gather by index: `output[i] = data[indices[i]]`. `indices` must be
/// `Uint32`-dtype; the output has the same length as `indices` (not `data`).
#[pyfunction]
pub fn metal_take(data: &MetalSeries, indices: &MetalSeries) -> PyResult<MetalSeries> {
    let data_buf = data.as_numeric_checked()?;
    let idx_buf = indices.as_numeric_checked()?;

    if indices.dtype != DType::Uint32 {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "take indices must be Uint32 dtype, got {:?}", indices.dtype
        )));
    }

    let dtype = data.dtype;
    let suffix = filter_suffix(dtype)?;
    let len = indices.len;
    let elem_size = dtype.size_in_bytes();

    let device = MetalBackend::device()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal device"))?;

    let out_buf = device.new_buffer(
        (len.max(1) * elem_size) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Empty indices: nothing to gather.
    if len == 0 {
        let result_buf = SharedBuffer::from_metal_buffer(out_buf, 0, dtype);
        return Ok(MetalSeries::from_numeric(result_buf));
    }

    let len_buf = make_len_buffer(device, len);

    let library = load_filter_library(device)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    let kernel_name = format!("take_{suffix}");
    let pipeline = get_pipeline_state(device, &library, &kernel_name)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let queue = MetalBackend::queue()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal queue"))?;
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(data_buf.metal_buffer()), 0);
    enc.set_buffer(1, Some(idx_buf.metal_buffer()), 0);
    enc.set_buffer(2, Some(&out_buf), 0);
    enc.set_buffer(3, Some(&len_buf), 0);
    enc.dispatch_thread_groups(
        MTLSize::new(num_threadgroups(len), 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "take kernel failed: Metal command buffer error"
        ));
    }

    let result_buf = SharedBuffer::from_metal_buffer(out_buf, len, dtype);
    Ok(MetalSeries::from_numeric(result_buf))
}
