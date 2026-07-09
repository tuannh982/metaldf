// Sort kernel dispatch — bitonic (small N) + CPU-histogram radix (large N).
//
// Selection heuristic:
//   - N < 100K: bitonic sort (single CB, no CPU histogram overhead)
//   - N >= 100K: radix sort (CPU histogram + GPU scatter per pass, 4 or 8 CBs)
//
// Supports Float32, Int32, and Int64. The CPU-side radix key conversion
// functions below must exactly match the Metal-side `RadixTraits<T>::to_key()`
// defined in rust/metal/types.h.

use pyo3::prelude::*;

use metal::{MTLSize, MTLResourceOptions};

use crate::backend::MetalBackend;
use crate::buffer::{SharedBuffer, DType};
use crate::kernels::{load_sort_library, get_pipeline_state, tuning};
use crate::series::MetalSeries;

const THREADGROUP_SIZE: u64 = 256;
const RADIX_BITS: u32 = 8;
const NUM_BUCKETS: usize = 1 << RADIX_BITS; // 256

const BITONIC_MAX_N: usize = 100_000;

/// Validate buffer dtype and return (length, dtype).
fn validate_sort_buffer(data: &SharedBuffer) -> PyResult<(usize, DType)> {
    match data.dtype {
        DType::Float32 | DType::Int32 | DType::Int64 => Ok((data.len, data.dtype)),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            format!("Sort not supported for {:?}", data.dtype)
        )),
    }
}

// ============================================================================
// CPU-side radix key conversion — must match RadixTraits<T>::to_key() in types.h
// ============================================================================

#[inline]
fn float_to_radix_key(f: f32) -> u32 {
    let u = f.to_bits();
    if (u & 0x80000000u32) != 0 { !u } else { u ^ 0x80000000u32 }
}

#[inline]
fn int32_to_radix_key(i: i32) -> u32 {
    (i as u32) ^ 0x80000000u32
}

#[inline]
fn int64_to_radix_key(i: i64) -> u64 {
    (i as u64) ^ 0x8000000000000000u64
}

// ============================================================================
// Radix sort (large N)
// ============================================================================

/// Allocate a padded key buffer (copy data, fill remainder with max) and an
/// identity index buffer. Shared by both radix and bitonic sort setup.
fn alloc_padded_key_buffer(
    device: &metal::Device,
    data: &SharedBuffer,
    len: usize,
    n: u64,
    dtype: DType,
) -> metal::Buffer {
    let elem_size = dtype.size_in_bytes();
    let buf = device.new_buffer(n * elem_size as u64, MTLResourceOptions::StorageModeShared);
    unsafe {
        std::ptr::copy_nonoverlapping(data.metal_buffer().contents(), buf.contents(), len * elem_size);
        dtype.fill_max(buf.contents() as *mut u8, len, n as usize);
    }
    buf
}

fn alloc_identity_indices(device: &metal::Device, n: u64) -> metal::Buffer {
    let buf = device.new_buffer(n * std::mem::size_of::<u32>() as u64, MTLResourceOptions::StorageModeShared);
    unsafe {
        let ptr = buf.contents() as *mut u32;
        for i in 0..n as usize { *ptr.add(i) = i as u32; }
    }
    buf
}

/// Setup working buffers for radix sort: padded keys + identity indices.
///
/// `pub(crate)` so groupby.rs's sort-based aggregation path can reuse the
/// exact same dtype-aware padding logic instead of duplicating it.
pub(crate) fn setup_radix_buffers(
    device: &metal::Device,
    data: &SharedBuffer,
    len: usize,
    dtype: DType,
) -> (metal::Buffer, metal::Buffer, metal::Buffer, metal::Buffer, u64) {
    let n = len.next_power_of_two() as u64;
    let keys0 = alloc_padded_key_buffer(device, data, len, n, dtype);
    let keys1 = device.new_buffer(n * dtype.size_in_bytes() as u64, MTLResourceOptions::StorageModeShared);
    let indices0 = alloc_identity_indices(device, n);
    let indices1 = device.new_buffer(n * std::mem::size_of::<u32>() as u64, MTLResourceOptions::StorageModeShared);
    (keys0, keys1, indices0, indices1, n)
}

/// Run CPU-histogram radix sort scatter passes on pre-populated GPU buffers.
///
/// Each pass: CPU reads keys from the shared GPU buffer, computes a
/// histogram + per-element local offsets for the current 8-bit digit, then
/// the GPU scatter kernel writes each element to its sorted position.
/// Also used directly by groupby.rs for its sort-based aggregation path.
pub fn run_radix_sort_on_buffers(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    keys_in: &metal::Buffer,
    keys_out: &metal::Buffer,
    indices_in: &metal::Buffer,
    indices_out: &metal::Buffer,
    n: u64,
    dtype: DType,
) -> PyResult<()> {
    let library = load_sort_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let scatter_name = format!("radix_scatter_{}", dtype.kernel_suffix());
    let scatter_pipeline = get_pipeline_state(device, &library, &scatter_name)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let num_passes = dtype.radix_passes();
    let num_groups = (n + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;

    let local_offsets_buf = device.new_buffer(
        n * std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let bucket_offsets_buf = device.new_buffer(
        NUM_BUCKETS as u64 * std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let uniform_buf = device.new_buffer(
        2 * std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let mut k_in = keys_in;
    let mut k_out = keys_out;
    let mut i_in = indices_in;
    let mut i_out = indices_out;

    for pass in 0..num_passes {
        let mut histogram = [0u32; NUM_BUCKETS];

        unsafe {
            let local_ptr = local_offsets_buf.contents() as *mut u32;
            let mut running = [0u32; NUM_BUCKETS];

            macro_rules! build_histogram {
                ($ty:ty, $key_fn:expr, $mask:expr) => {{
                    let keys_ptr = k_in.contents() as *const $ty;
                    for i in 0..n as usize {
                        let key = $key_fn(*keys_ptr.add(i));
                        let digit = ((key >> (pass * 8)) & $mask) as usize;
                        histogram[digit] += 1;
                        *local_ptr.add(i) = running[digit];
                        running[digit] += 1;
                    }
                }};
            }

            match dtype {
                DType::Float32 => build_histogram!(f32, float_to_radix_key, 0xFFu32),
                DType::Int32   => build_histogram!(i32, int32_to_radix_key, 0xFFu32),
                DType::Int64   => build_histogram!(i64, int64_to_radix_key, 0xFFu64),
                _ => unreachable!(),
            }

            let mut bucket_offsets = [0u32; NUM_BUCKETS];
            let mut sum = 0u32;
            for i in 0..NUM_BUCKETS {
                bucket_offsets[i] = sum;
                sum += histogram[i];
            }
            let b_ptr = bucket_offsets_buf.contents() as *mut u32;
            for i in 0..NUM_BUCKETS {
                *b_ptr.add(i) = bucket_offsets[i];
            }
            let u_ptr = uniform_buf.contents() as *mut u32;
            *u_ptr.add(0) = n as u32;
            *u_ptr.add(1) = pass;
        }

        let cb = queue.new_command_buffer();
        {
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&scatter_pipeline);
            enc.set_buffer(0, Some(k_in), 0);
            enc.set_buffer(1, Some(i_in), 0);
            enc.set_buffer(2, Some(k_out), 0);
            enc.set_buffer(3, Some(i_out), 0);
            enc.set_buffer(4, Some(&bucket_offsets_buf), 0);
            enc.set_buffer(5, Some(&local_offsets_buf), 0);
            enc.set_buffer(6, Some(&uniform_buf), 0);
            enc.set_buffer(7, Some(&uniform_buf), 4);
            enc.dispatch_thread_groups(
                MTLSize::new(num_groups, 1, 1),
                MTLSize::new(THREADGROUP_SIZE, 1, 1),
            );
            enc.end_encoding();
        }
        cb.commit();
        cb.wait_until_completed();

        if cb.status() == metal::MTLCommandBufferStatus::Error {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("Radix scatter failed on pass {}", pass)
            ));
        }

        std::mem::swap(&mut k_in, &mut k_out);
        std::mem::swap(&mut i_in, &mut i_out);
    }

    Ok(())
}

// ============================================================================
// Bitonic sort (small N)
// ============================================================================

fn setup_bitonic_buffers(
    device: &metal::Device,
    data: &SharedBuffer,
    len: usize,
    dtype: DType,
) -> (metal::Buffer, metal::Buffer, u64) {
    let n = len.next_power_of_two() as u64;
    let keys = alloc_padded_key_buffer(device, data, len, n, dtype);
    let indices = alloc_identity_indices(device, n);
    (keys, indices, n)
}

/// Hybrid bitonic sort: threadgroup-local sort + global steps + local merge.
///
/// Phase 1: Sort each `local_sort_size`-element block in threadgroup memory
///          (all stages < local_sort_stages, zero global memory traffic).
/// Phase 2: For stages >= local_sort_stages (cross-block merges):
///   a) Global steps (comparison distance >= local_sort_size) via global memory
///   b) Local merge (remaining steps) back in threadgroup memory
///
/// Block size and stage counts come from `tuning()` (GPU-family-specific).
/// All phases batched into a single command buffer.
fn run_bitonic_passes(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    library: &metal::Library,
    dtype: DType,
    work_buffer: &metal::Buffer,
    indices_buffer: &metal::Buffer,
    n: u64,
) -> PyResult<()> {
    let suffix = dtype.kernel_suffix();
    let elem_size = dtype.size_in_bytes() as u64;
    let num_stages = n.trailing_zeros() as u32;
    let local_sort_size = tuning().local_sort_size;
    let local_sort_stages = tuning().local_sort_stages;
    let num_local_groups = (n + local_sort_size - 1) / local_sort_size;

    let local_sort_pl = get_pipeline_state(device, library, &format!("bitonic_sort_local_{suffix}"))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let global_pl = get_pipeline_state(device, library, &format!("bitonic_sort_{suffix}_ascending"))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let n_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe { *(n_buf.contents() as *mut u32) = n as u32; }

    let cb = queue.new_command_buffer();

    // Phase 1: local sort (stages 0..9 in threadgroup memory)
    {
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&local_sort_pl);
        enc.set_buffer(0, Some(work_buffer), 0);
        enc.set_buffer(1, Some(indices_buffer), 0);
        enc.set_buffer(2, Some(&n_buf), 0);
        enc.set_threadgroup_memory_length(0, local_sort_size * elem_size);
        enc.set_threadgroup_memory_length(1, local_sort_size * 4);
        enc.dispatch_thread_groups(
            MTLSize::new(num_local_groups, 1, 1),
            MTLSize::new(local_sort_size, 1, 1),
        );
        enc.end_encoding();
    }

    // Phase 2: global merge stages (if n > 1024)
    if num_stages > local_sort_stages {
        let merge_local_pl = get_pipeline_state(device, library, &format!("bitonic_merge_local_{suffix}"))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        let num_global_threadgroups = (n + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;

        // Count passes for uniform buffer pre-allocation
        let mut total_global = 0usize;
        let total_merge = (num_stages - local_sort_stages) as usize;
        for stage in local_sort_stages..num_stages {
            total_global += (stage - local_sort_stages + 1) as usize;
        }

        let global_uniform = device.new_buffer(
            (total_global * 3 * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let merge_uniform = device.new_buffer(
            (total_merge * 2 * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        unsafe {
            let gp = global_uniform.contents() as *mut u32;
            let mut gi = 0usize;
            for stage in local_sort_stages..num_stages {
                for step in (local_sort_stages..=stage).rev() {
                    *gp.add(gi * 3) = n as u32;
                    *gp.add(gi * 3 + 1) = stage;
                    *gp.add(gi * 3 + 2) = step;
                    gi += 1;
                }
            }
            let mp = merge_uniform.contents() as *mut u32;
            for (mi, stage) in (local_sort_stages..num_stages).enumerate() {
                *mp.add(mi * 2) = n as u32;
                *mp.add(mi * 2 + 1) = stage;
            }
        }

        let mut gi = 0usize;
        let mut mi = 0usize;

        for stage in local_sort_stages..num_stages {
            // Global steps (comparison distance >= 1024)
            for _step in (local_sort_stages..=stage).rev() {
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&global_pl);
                enc.set_buffer(0, Some(work_buffer), 0);
                enc.set_buffer(1, Some(indices_buffer), 0);
                let off = (gi * 3 * 4) as u64;
                enc.set_buffer(2, Some(&global_uniform), off);
                enc.set_buffer(3, Some(&global_uniform), off + 4);
                enc.set_buffer(4, Some(&global_uniform), off + 8);
                enc.dispatch_thread_groups(
                    MTLSize::new(num_global_threadgroups, 1, 1),
                    MTLSize::new(THREADGROUP_SIZE, 1, 1),
                );
                enc.end_encoding();
                gi += 1;
            }

            // Local merge (steps 9..0 in threadgroup memory)
            {
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&merge_local_pl);
                enc.set_buffer(0, Some(work_buffer), 0);
                enc.set_buffer(1, Some(indices_buffer), 0);
                let off = (mi * 2 * 4) as u64;
                enc.set_buffer(2, Some(&merge_uniform), off);
                enc.set_buffer(3, Some(&merge_uniform), off + 4);
                enc.set_threadgroup_memory_length(0, local_sort_size * elem_size);
                enc.set_threadgroup_memory_length(1, local_sort_size * 4);
                enc.dispatch_thread_groups(
                    MTLSize::new(num_local_groups, 1, 1),
                    MTLSize::new(local_sort_size, 1, 1),
                );
                enc.end_encoding();
                mi += 1;
            }
        }
    }

    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Bitonic sort failed: Metal command buffer error"
        ));
    }
    Ok(())
}

/// Bitonic sort a buffer, returning the sorted values.
fn run_bitonic_sort(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    data: &SharedBuffer,
    len: usize,
    dtype: DType,
) -> PyResult<SharedBuffer> {
    let library = load_sort_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let (work_buffer, indices_buffer, n) = setup_bitonic_buffers(device, data, len, dtype);
    run_bitonic_passes(device, queue, &library, dtype, &work_buffer, &indices_buffer, n)?;

    Ok(SharedBuffer::from_metal_buffer(work_buffer, len, dtype))
}

/// Bitonic argsort: return indices that would sort the array.
fn run_bitonic_argsort(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    data: &SharedBuffer,
    len: usize,
    dtype: DType,
) -> PyResult<SharedBuffer> {
    let library = load_sort_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let (work_buffer, indices_buffer, n) = setup_bitonic_buffers(device, data, len, dtype);
    run_bitonic_passes(device, queue, &library, dtype, &work_buffer, &indices_buffer, n)?;

    Ok(SharedBuffer::from_metal_buffer(indices_buffer, len, DType::Int32))
}

// ============================================================================
// Public entry points
// ============================================================================

#[pyfunction]
pub fn metal_sort(data: &MetalSeries) -> PyResult<MetalSeries> {
    let buf = data.as_numeric_checked()?;
    let (len, dtype) = validate_sort_buffer(buf)?;
    let (device, queue) = MetalBackend::device_and_queue()?;

    let result = if len < BITONIC_MAX_N {
        run_bitonic_sort(device, queue, buf, len, dtype)
    } else {
        let (keys0, keys1, indices0, indices1, n) = setup_radix_buffers(device, buf, len, dtype);
        run_radix_sort_on_buffers(device, queue, &keys0, &keys1, &indices0, &indices1, n, dtype)?;
        Ok(SharedBuffer::from_metal_buffer(keys0, len, dtype))
    }?;
    Ok(MetalSeries::from_numeric(result))
}

#[pyfunction]
pub fn metal_argsort(data: &MetalSeries) -> PyResult<MetalSeries> {
    let buf = data.as_numeric_checked()?;
    let (len, dtype) = validate_sort_buffer(buf)?;
    let (device, queue) = MetalBackend::device_and_queue()?;

    let result = if len < BITONIC_MAX_N {
        run_bitonic_argsort(device, queue, buf, len, dtype)
    } else {
        let (keys0, keys1, indices0, indices1, n) = setup_radix_buffers(device, buf, len, dtype);
        run_radix_sort_on_buffers(device, queue, &keys0, &keys1, &indices0, &indices1, n, dtype)?;
        Ok(SharedBuffer::from_metal_buffer(indices0, len, DType::Int32))
    }?;
    Ok(MetalSeries::from_numeric(result))
}
