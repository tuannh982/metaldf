// Hash join kernel dispatch — GPU equi-join for inner joins.
//
// Algorithm (3 GPU passes + scan):
//   1. Build:        Hash the build-table keys into a linear-probing hash table
//   2. Probe-count:  Count matches per probe row
//   3. Prefix-sum:   Inclusive scan on count buffer → exclusive write offsets
//   4. Probe-write:  Write (left_idx, right_idx) pairs at pre-computed offsets
//
// Returns (left_indices, right_indices) as two MetalSeries of Uint32. The
// Python layer (Task 6.2) uses metal_take to gather actual columns.
//
// Only float32/int32 keys supported (32-bit atomics, same as groupby).
// Inner join only — left/right/outer joins deferred.

use pyo3::prelude::*;

use metal::{MTLSize, MTLResourceOptions};

use crate::backend::MetalBackend;
use crate::buffer::{SharedBuffer, DType};
use crate::kernels::{load_join_library, get_pipeline_state};
use crate::kernels::scan::prefix_sum_inclusive;
use crate::series::MetalSeries;

const THREADGROUP_SIZE: u64 = 256;

/// Validate that build and probe keys have a supported dtype for hash join.
fn validate_join_keys(build: &SharedBuffer, probe: &SharedBuffer) -> PyResult<DType> {
    match (build.dtype, probe.dtype) {
        (DType::Float32, DType::Float32) => Ok(DType::Float32),
        (DType::Int32, DType::Int32) => Ok(DType::Int32),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Hash join not supported for build={:?} probe={:?} (must both be Float32 or Int32)",
            build.dtype, probe.dtype
        ))),
    }
}

fn num_threadgroups(len: u64) -> u64 {
    (len + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE
}

fn make_u32_buffer(device: &metal::Device, value: u32) -> metal::Buffer {
    let buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe { *(buf.contents() as *mut u32) = value; }
    buf
}

/// GPU hash join: returns (left_indices, right_indices) as Uint32 MetalSeries.
///
/// `build_keys` is hashed into the table; `probe_keys` is probed against it.
/// For inner joins, the caller decides which side is build vs. probe.
#[pyfunction]
pub fn metal_hash_join(
    build_keys: &MetalSeries,
    probe_keys: &MetalSeries,
) -> PyResult<(MetalSeries, MetalSeries)> {
    let build_buf = build_keys.as_numeric_checked()?;
    let probe_buf = probe_keys.as_numeric_checked()?;
    let dtype = validate_join_keys(build_buf, probe_buf)?;

    let (device, queue) = MetalBackend::device_and_queue()?;
    let library = load_join_library(device)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    let suffix = dtype.kernel_suffix();
    let build_len = build_buf.len as u64;
    let probe_len = probe_buf.len as u64;

    // --- Handle empty inputs ---
    if build_len == 0 || probe_len == 0 {
        let empty_left = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        let empty_right = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        return Ok((
            MetalSeries::from_numeric(SharedBuffer::from_metal_buffer(empty_left, 0, DType::Uint32)),
            MetalSeries::from_numeric(SharedBuffer::from_metal_buffer(empty_right, 0, DType::Uint32)),
        ));
    }

    // --- Step 1: Allocate and initialize hash table ---
    // Table size: next power of 2 >= 2 * build_len (load factor <= 0.5).
    let table_size = (build_len * 2).next_power_of_two().max(64);

    let table_keys_buf = device.new_buffer(table_size * 4, MTLResourceOptions::StorageModeShared);
    let table_rows_buf = device.new_buffer(table_size * 4, MTLResourceOptions::StorageModeShared);

    // Initialize table_keys to sentinel (0xFFFFFFFF)
    unsafe {
        let ptr = table_keys_buf.contents() as *mut u32;
        for i in 0..table_size as usize {
            *ptr.add(i) = 0xFFFFFFFFu32;
        }
    }

    let build_len_buf = make_u32_buffer(device, build_len as u32);
    let probe_len_buf = make_u32_buffer(device, probe_len as u32);
    let table_size_buf = make_u32_buffer(device, table_size as u32);

    // --- Step 2: Build pass (insert build keys into hash table) ---
    {
        let pipeline = get_pipeline_state(device, &library, &format!("join_build_{suffix}"))
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(build_buf.metal_buffer()), 0);
        enc.set_buffer(1, Some(&table_keys_buf), 0);
        enc.set_buffer(2, Some(&table_rows_buf), 0);
        enc.set_buffer(3, Some(&build_len_buf), 0);
        enc.set_buffer(4, Some(&table_size_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(num_threadgroups(build_len), 1, 1),
            MTLSize::new(THREADGROUP_SIZE, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        if cb.status() == metal::MTLCommandBufferStatus::Error {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Hash join build pass failed: Metal command buffer error"
            ));
        }
    }

    // --- Step 3: Probe-count pass (count matches per probe row) ---
    let count_buf = device.new_buffer(probe_len * 4, MTLResourceOptions::StorageModeShared);
    // Initialize count buffer to zero
    unsafe {
        let ptr = count_buf.contents() as *mut u32;
        for i in 0..probe_len as usize {
            *ptr.add(i) = 0;
        }
    }

    {
        let pipeline = get_pipeline_state(device, &library, &format!("join_probe_count_{suffix}"))
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(probe_buf.metal_buffer()), 0);
        enc.set_buffer(1, Some(&table_keys_buf), 0);
        enc.set_buffer(2, Some(&table_rows_buf), 0);
        enc.set_buffer(3, Some(&count_buf), 0);
        enc.set_buffer(4, Some(&probe_len_buf), 0);
        enc.set_buffer(5, Some(&table_size_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(num_threadgroups(probe_len), 1, 1),
            MTLSize::new(THREADGROUP_SIZE, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        if cb.status() == metal::MTLCommandBufferStatus::Error {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Hash join probe-count pass failed: Metal command buffer error"
            ));
        }
    }

    // --- Step 4: Prefix-sum on count buffer → inclusive scan ---
    let inclusive_scan = prefix_sum_inclusive(&count_buf, probe_len as usize, DType::Uint32)?;

    // Read total matches from last element of inclusive scan
    let total_matches = unsafe {
        *(inclusive_scan.contents() as *const u32).add(probe_len as usize - 1) as usize
    };

    // No matches: return empty index arrays
    if total_matches == 0 {
        let empty_left = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        let empty_right = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        return Ok((
            MetalSeries::from_numeric(SharedBuffer::from_metal_buffer(empty_left, 0, DType::Uint32)),
            MetalSeries::from_numeric(SharedBuffer::from_metal_buffer(empty_right, 0, DType::Uint32)),
        ));
    }

    // Convert inclusive scan to exclusive offsets:
    // exclusive[i] = inclusive[i] - count[i] = inclusive[i-1] (or 0 for i=0)
    // We compute this on the CPU since it's a simple shift.
    let offset_buf = device.new_buffer(probe_len * 4, MTLResourceOptions::StorageModeShared);
    unsafe {
        let scan_ptr = inclusive_scan.contents() as *const u32;
        let count_ptr = count_buf.contents() as *const u32;
        let offset_ptr = offset_buf.contents() as *mut u32;
        for i in 0..probe_len as usize {
            // exclusive offset = inclusive_scan[i] - count[i]
            *offset_ptr.add(i) = *scan_ptr.add(i) - *count_ptr.add(i);
        }
    }

    // --- Step 5: Probe-write pass (write index pairs) ---
    let left_indices_buf = device.new_buffer(
        (total_matches as u64) * 4,
        MTLResourceOptions::StorageModeShared,
    );
    let right_indices_buf = device.new_buffer(
        (total_matches as u64) * 4,
        MTLResourceOptions::StorageModeShared,
    );

    {
        let pipeline = get_pipeline_state(device, &library, &format!("join_probe_write_{suffix}"))
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(probe_buf.metal_buffer()), 0);
        enc.set_buffer(1, Some(&table_keys_buf), 0);
        enc.set_buffer(2, Some(&table_rows_buf), 0);
        enc.set_buffer(3, Some(&offset_buf), 0);
        enc.set_buffer(4, Some(&left_indices_buf), 0);
        enc.set_buffer(5, Some(&right_indices_buf), 0);
        enc.set_buffer(6, Some(&probe_len_buf), 0);
        enc.set_buffer(7, Some(&table_size_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(num_threadgroups(probe_len), 1, 1),
            MTLSize::new(THREADGROUP_SIZE, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        if cb.status() == metal::MTLCommandBufferStatus::Error {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Hash join probe-write pass failed: Metal command buffer error"
            ));
        }
    }

    // --- Return (left_indices, right_indices) as Uint32 MetalSeries ---
    Ok((
        MetalSeries::from_numeric(SharedBuffer::from_metal_buffer(
            left_indices_buf, total_matches, DType::Uint32,
        )),
        MetalSeries::from_numeric(SharedBuffer::from_metal_buffer(
            right_indices_buf, total_matches, DType::Uint32,
        )),
    ))
}
