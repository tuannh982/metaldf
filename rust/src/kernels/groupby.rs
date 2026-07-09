// GroupBy kernel dispatch — hash-based (low cardinality) + sort-based (high cardinality).
//
// Strategy:
//   - Low cardinality (len <= 500_000): hash-based, 3-pass GPU
//   - High cardinality (len > 500_000): sort-based with a serial leader-scan
//     direct reduction (see groupby.metal for why this isn't yet a parallel
//     segmented reduction)
//
// Hash-based approach:
//   1. groupby_hash_build_<suffix>: insert keys into hash table, assign group_ids
//   2. groupby_hash_{sum,min,max,count}_<suffix>: accumulate values into per-group accumulators
//   3. groupby_hash_compact_{sum,minmax,count}_<suffix>: extract results from hash table
//
// All passes for a given aggregation are batched into a single command buffer.
//
// Supported dtypes: Float32/Float32 and Int32/Int32 (keys and values must
// match). Float64/Int64 values are excluded because Metal has no
// atomic<double>/atomic<long>, and Int64 keys are excluded to keep the
// hash table's key-bits storage a uniform 32 bits (see KeyBits<T> in
// types.h) — could be added later for the sort-based path only.
//
// Mean is computed as sum/count: both are dispatched on the GPU against a
// SHARED group-id assignment (one hash build, or one sort-direct kernel
// invocation) so the two results are guaranteed to line up group-for-group,
// then divided into a Float32 result on the CPU.

use pyo3::prelude::*;

use metal::{MTLSize, MTLResourceOptions};

use crate::backend::MetalBackend;
use crate::buffer::{SharedBuffer, DType};
use crate::kernels::{load_groupby_library, load_sort_library, get_pipeline_state};
use crate::kernels::sort::{run_radix_sort_on_buffers, setup_radix_buffers};
use crate::series::MetalSeries;

const THREADGROUP_SIZE: u64 = 256;

/// Selects hash-based (len <= HASH_MAX_LEN) or sort-based (len > HASH_MAX_LEN)
/// aggregation. Hash tables scale with `len`, so for very large inputs a
/// sort is cheaper than a huge (and increasingly collision-prone) table.
const HASH_MAX_LEN: usize = 500_000;

/// Validate that keys/values have matching length and a supported dtype
/// pairing, returning the shared dtype on success.
fn validate_groupby_buffers(keys: &SharedBuffer, values: &SharedBuffer) -> PyResult<DType> {
    if keys.len != values.len {
        return Err(pyo3::exceptions::PyValueError::new_err("keys and values must have same length"));
    }
    match (keys.dtype, values.dtype) {
        (DType::Float32, DType::Float32) | (DType::Int32, DType::Int32) => Ok(keys.dtype),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(
            format!("Groupby not supported for key={:?} value={:?}", keys.dtype, values.dtype)
        )),
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GroupOp {
    Sum,
    Min,
    Max,
    Count,
}

impl GroupOp {
    fn accumulate_kernel(self, suffix: &str) -> String {
        match self {
            GroupOp::Sum => format!("groupby_hash_sum_{suffix}"),
            GroupOp::Min => format!("groupby_hash_min_{suffix}"),
            GroupOp::Max => format!("groupby_hash_max_{suffix}"),
            GroupOp::Count => format!("groupby_hash_count_{suffix}"),
        }
    }

    fn compact_kernel(self, suffix: &str) -> String {
        match self {
            GroupOp::Sum => format!("groupby_hash_compact_sum_{suffix}"),
            GroupOp::Min | GroupOp::Max => format!("groupby_hash_compact_minmax_{suffix}"),
            GroupOp::Count => format!("groupby_hash_compact_count_{suffix}"),
        }
    }

    fn direct_kernel(self, suffix: &str) -> String {
        match self {
            GroupOp::Sum => format!("groupby_sum_direct_{suffix}"),
            GroupOp::Min => format!("groupby_min_direct_{suffix}"),
            GroupOp::Max => format!("groupby_max_direct_{suffix}"),
            GroupOp::Count => format!("groupby_count_direct_{suffix}"),
        }
    }

    /// dtype of the aggregated value output (before any post-processing,
    /// e.g. mean's sum/count -> float32 division happens on top of this).
    fn out_dtype(self, key_dtype: DType) -> DType {
        match self {
            GroupOp::Count => DType::Int64,
            _ => key_dtype,
        }
    }
}

fn init_accumulator(buf: &metal::Buffer, len: u64, dtype: DType, op: GroupOp) {
    unsafe {
        let n = len as usize;
        let fill_u32 = |ptr: *mut u32, val: u32| { for i in 0..n { *ptr.add(i) = val; } };
        let fill_i32 = |ptr: *mut i32, val: i32| { for i in 0..n { *ptr.add(i) = val; } };

        match (dtype, op) {
            (_, GroupOp::Count)              => fill_u32(buf.contents() as *mut u32, 0),
            (DType::Float32, GroupOp::Sum)   => fill_u32(buf.contents() as *mut u32, 0),
            (DType::Int32,   GroupOp::Sum)   => fill_i32(buf.contents() as *mut i32, 0),
            (DType::Float32, GroupOp::Min)   => fill_u32(buf.contents() as *mut u32, f32::INFINITY.to_bits()),
            (DType::Int32,   GroupOp::Min)   => fill_i32(buf.contents() as *mut i32, i32::MAX),
            (DType::Float32, GroupOp::Max)   => fill_u32(buf.contents() as *mut u32, f32::NEG_INFINITY.to_bits()),
            (DType::Int32,   GroupOp::Max)   => fill_i32(buf.contents() as *mut i32, i32::MIN),
            _ => unreachable!("groupby only supports Float32/Int32"),
        }
    }
}

/// Divide per-group sum by per-group count. Returns float32 for float32
/// input, float64 for integer input (matching pandas mean dtype promotion).
/// `count_elem_size`: 4 for hash path (u32 accumulator), 8 for sort path (i64 output).
fn compute_mean_buffer(
    device: &metal::Device,
    sum_buf: &metal::Buffer,
    count_buf: &metal::Buffer,
    num_groups: usize,
    dtype: DType,
    count_elem_size: usize,
) -> (metal::Buffer, DType) {
    unsafe {
        let read_count = |g: usize| -> f64 {
            if count_elem_size == 8 {
                *(count_buf.contents() as *const i64).add(g) as f64
            } else {
                *(count_buf.contents() as *const u32).add(g) as f64
            }
        };
        match dtype {
            DType::Float32 => {
                let buf = device.new_buffer(
                    (num_groups.max(1) * 4) as u64, MTLResourceOptions::StorageModeShared,
                );
                let mean_ptr = buf.contents() as *mut f32;
                let sum_ptr = sum_buf.contents() as *const f32;
                for g in 0..num_groups {
                    *mean_ptr.add(g) = (*sum_ptr.add(g) as f64 / read_count(g)) as f32;
                }
                (buf, DType::Float32)
            }
            DType::Int32 => {
                let buf = device.new_buffer(
                    (num_groups.max(1) * 8) as u64, MTLResourceOptions::StorageModeShared,
                );
                let mean_ptr = buf.contents() as *mut f64;
                let sum_ptr = sum_buf.contents() as *const i32;
                for g in 0..num_groups {
                    *mean_ptr.add(g) = *sum_ptr.add(g) as f64 / read_count(g);
                }
                (buf, DType::Float64)
            }
            _ => unreachable!("groupby only supports Float32/Int32"),
        }
    }
}

// ============================================================================
// Hash-based path (low cardinality)
// ============================================================================

/// Allocate a fresh hash table for `keys` and encode the build pass onto
/// `cb` (caller commits). Returns (table_keys, table_gids, group_counter,
/// table_size, len_buf, table_size_buf) for use by later passes encoded
/// onto the same command buffer.
fn setup_and_encode_hash_build(
    device: &metal::Device,
    library: &metal::Library,
    keys: &SharedBuffer,
    suffix: &str,
    len: u64,
    cb: &metal::CommandBufferRef,
) -> PyResult<(metal::Buffer, metal::Buffer, metal::Buffer, u64, metal::Buffer, metal::Buffer)> {
    // Hash table size: next power of two, at least 1024, at most 4M entries.
    // Use 2x the data size to keep load factor < 0.5 for linear probing.
    let table_size = (1024u64)
        .max((len * 2).next_power_of_two())
        .min(4_194_304);

    let table_keys = device.new_buffer(table_size * 4, MTLResourceOptions::StorageModeShared);
    let table_gids = device.new_buffer(table_size * 4, MTLResourceOptions::StorageModeShared);
    let group_counter = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe {
        let ptr = table_keys.contents() as *mut u32;
        for i in 0..table_size as usize { *ptr.add(i) = 0xFFFFFFFFu32; }
        *(group_counter.contents() as *mut u32) = 0;
    }

    let len_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let table_size_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe {
        *(len_buf.contents() as *mut u32) = len as u32;
        *(table_size_buf.contents() as *mut u32) = table_size as u32;
    }

    let build_pipeline = get_pipeline_state(device, library, &format!("groupby_hash_build_{suffix}"))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let num_groups = (len + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    {
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&build_pipeline);
        enc.set_buffer(0, Some(keys.metal_buffer()), 0);
        enc.set_buffer(1, Some(&table_keys), 0);
        enc.set_buffer(2, Some(&table_gids), 0);
        enc.set_buffer(3, Some(&group_counter), 0);
        enc.set_buffer(4, Some(&len_buf), 0);
        enc.set_buffer(5, Some(&table_size_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(num_groups, 1, 1),
            MTLSize::new(THREADGROUP_SIZE, 1, 1),
        );
        enc.end_encoding();
    }

    Ok((table_keys, table_gids, group_counter, table_size, len_buf, table_size_buf))
}

/// Hash-based groupby for a single aggregation (sum/min/max/count).
fn metal_groupby_hash(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    keys: &SharedBuffer,
    values: &SharedBuffer,
    dtype: DType,
    op: GroupOp,
) -> PyResult<(SharedBuffer, SharedBuffer)> {
    let library = load_groupby_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let suffix = dtype.kernel_suffix();
    let len = keys.len() as u64;
    let elem_size = dtype.size_in_bytes() as u64;

    let out_accum = device.new_buffer(len.max(1) * 4, MTLResourceOptions::StorageModeShared);
    init_accumulator(&out_accum, len, dtype, op);

    let out_dtype = op.out_dtype(dtype);
    let out_keys = device.new_buffer(len.max(1) * elem_size, MTLResourceOptions::StorageModeShared);
    let out_vals = device.new_buffer(
        len.max(1) * out_dtype.size_in_bytes() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cb = queue.new_command_buffer();

    // Pass 1: build hash table
    let (table_keys, table_gids, group_counter, table_size, len_buf, table_size_buf) =
        setup_and_encode_hash_build(device, &library, keys, suffix, len, cb)?;

    let num_groups_disp = (len + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    let table_groups = (table_size + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;

    // Pass 2: accumulate
    {
        let kname = op.accumulate_kernel(suffix);
        let pipeline = get_pipeline_state(device, &library, &kname)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        if op == GroupOp::Count {
            enc.set_buffer(0, Some(keys.metal_buffer()), 0);
            enc.set_buffer(1, Some(&table_keys), 0);
            enc.set_buffer(2, Some(&table_gids), 0);
            enc.set_buffer(3, Some(&out_accum), 0);
            enc.set_buffer(4, Some(&len_buf), 0);
            enc.set_buffer(5, Some(&table_size_buf), 0);
        } else {
            enc.set_buffer(0, Some(keys.metal_buffer()), 0);
            enc.set_buffer(1, Some(values.metal_buffer()), 0);
            enc.set_buffer(2, Some(&table_keys), 0);
            enc.set_buffer(3, Some(&table_gids), 0);
            enc.set_buffer(4, Some(&out_accum), 0);
            enc.set_buffer(5, Some(&len_buf), 0);
            enc.set_buffer(6, Some(&table_size_buf), 0);
        }
        enc.dispatch_thread_groups(
            MTLSize::new(num_groups_disp, 1, 1),
            MTLSize::new(THREADGROUP_SIZE, 1, 1),
        );
        enc.end_encoding();
    }

    // Pass 3: compact
    {
        let kname = op.compact_kernel(suffix);
        let pipeline = get_pipeline_state(device, &library, &kname)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&table_keys), 0);
        enc.set_buffer(1, Some(&table_gids), 0);
        enc.set_buffer(2, Some(&out_accum), 0);
        enc.set_buffer(3, Some(&out_keys), 0);
        enc.set_buffer(4, Some(&out_vals), 0);
        enc.set_buffer(5, Some(&table_size_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(table_groups, 1, 1),
            MTLSize::new(THREADGROUP_SIZE, 1, 1),
        );
        enc.end_encoding();
    }

    cb.commit();
    cb.wait_until_completed();
    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Hash groupby failed: Metal command buffer error"
        ));
    }

    let num_groups = unsafe { *(group_counter.contents() as *const u32) as usize };

    Ok((
        SharedBuffer::from_metal_buffer(out_keys, num_groups, dtype),
        SharedBuffer::from_metal_buffer(out_vals, num_groups, out_dtype),
    ))
}

/// Hash-based mean: one hash build, sum + count accumulate against the same
/// (frozen) table, compact the keys+sum, then read count directly (same
/// dense group-id ordering as the compacted sum) and divide on the CPU.
fn metal_groupby_mean_hash(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    keys: &SharedBuffer,
    values: &SharedBuffer,
    dtype: DType,
) -> PyResult<(SharedBuffer, SharedBuffer)> {
    let library = load_groupby_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let suffix = dtype.kernel_suffix();
    let len = keys.len() as u64;
    let elem_size = dtype.size_in_bytes() as u64;

    let sum_accum = device.new_buffer(len.max(1) * elem_size, MTLResourceOptions::StorageModeShared);
    init_accumulator(&sum_accum, len, dtype, GroupOp::Sum);
    let count_accum = device.new_buffer(len.max(1) * 4, MTLResourceOptions::StorageModeShared);
    init_accumulator(&count_accum, len, dtype, GroupOp::Count);

    let out_keys = device.new_buffer(len.max(1) * elem_size, MTLResourceOptions::StorageModeShared);
    let out_sum_vals = device.new_buffer(len.max(1) * elem_size, MTLResourceOptions::StorageModeShared);

    let cb = queue.new_command_buffer();

    // Pass 1: build hash table
    let (table_keys, table_gids, group_counter, table_size, len_buf, table_size_buf) =
        setup_and_encode_hash_build(device, &library, keys, suffix, len, cb)?;

    let num_groups_disp = (len + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    let table_groups = (table_size + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;

    // Sum accumulate
    {
        let pipeline = get_pipeline_state(device, &library, &format!("groupby_hash_sum_{suffix}"))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(keys.metal_buffer()), 0);
        enc.set_buffer(1, Some(values.metal_buffer()), 0);
        enc.set_buffer(2, Some(&table_keys), 0);
        enc.set_buffer(3, Some(&table_gids), 0);
        enc.set_buffer(4, Some(&sum_accum), 0);
        enc.set_buffer(5, Some(&len_buf), 0);
        enc.set_buffer(6, Some(&table_size_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(num_groups_disp, 1, 1),
            MTLSize::new(THREADGROUP_SIZE, 1, 1),
        );
        enc.end_encoding();
    }

    // Count accumulate (against the SAME table_keys/table_gids)
    {
        let pipeline = get_pipeline_state(device, &library, &format!("groupby_hash_count_{suffix}"))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(keys.metal_buffer()), 0);
        enc.set_buffer(1, Some(&table_keys), 0);
        enc.set_buffer(2, Some(&table_gids), 0);
        enc.set_buffer(3, Some(&count_accum), 0);
        enc.set_buffer(4, Some(&len_buf), 0);
        enc.set_buffer(5, Some(&table_size_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(num_groups_disp, 1, 1),
            MTLSize::new(THREADGROUP_SIZE, 1, 1),
        );
        enc.end_encoding();
    }

    // Compact sum + keys (count is already dense-indexed by the same group
    // ids, so it doesn't need its own compact pass).
    {
        let pipeline = get_pipeline_state(device, &library, &format!("groupby_hash_compact_sum_{suffix}"))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&table_keys), 0);
        enc.set_buffer(1, Some(&table_gids), 0);
        enc.set_buffer(2, Some(&sum_accum), 0);
        enc.set_buffer(3, Some(&out_keys), 0);
        enc.set_buffer(4, Some(&out_sum_vals), 0);
        enc.set_buffer(5, Some(&table_size_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(table_groups, 1, 1),
            MTLSize::new(THREADGROUP_SIZE, 1, 1),
        );
        enc.end_encoding();
    }

    cb.commit();
    cb.wait_until_completed();
    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Hash groupby mean failed: Metal command buffer error"
        ));
    }

    let num_groups = unsafe { *(group_counter.contents() as *const u32) as usize };
    let (mean_buf, mean_dtype) = compute_mean_buffer(device, &out_sum_vals, &count_accum, num_groups, dtype, 4);

    Ok((
        SharedBuffer::from_metal_buffer(out_keys, num_groups, dtype),
        SharedBuffer::from_metal_buffer(mean_buf, num_groups, mean_dtype),
    ))
}

// ============================================================================
// Sort-based path (high cardinality)
// ============================================================================

/// Sort `keys` (tracking a permutation), returning the buffers holding the
/// final sorted keys/indices plus the padded length `n`. Float32/Int32 both
/// use 4 radix passes, so the sorted result always round-trips back into
/// the first ("in") buffer of each ping-pong pair — see run_radix_sort_on_buffers.
fn sort_keys_with_indices(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    keys: &SharedBuffer,
    len: usize,
    dtype: DType,
) -> PyResult<(metal::Buffer, metal::Buffer, u64)> {
    let (keys0, keys1, indices0, indices1, n) = setup_radix_buffers(device, keys, len, dtype);
    run_radix_sort_on_buffers(device, queue, &keys0, &keys1, &indices0, &indices1, n, dtype)?;
    Ok((keys0, indices0, n))
}

/// Gather `values` (unsorted) into `sorted_values` using `indices` (as
/// produced by `sort_keys_with_indices`), encoded onto `cb` (caller commits).
fn gather_values(
    device: &metal::Device,
    values: &SharedBuffer,
    indices: &metal::Buffer,
    len: usize,
    n: u64,
    dtype: DType,
    cb: &metal::CommandBufferRef,
) -> PyResult<metal::Buffer> {
    let elem_size = dtype.size_in_bytes() as u64;
    let work_values = device.new_buffer(n * elem_size, MTLResourceOptions::StorageModeShared);
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.metal_buffer().contents(),
            work_values.contents(),
            len * elem_size as usize,
        );
    }
    let sorted_values = device.new_buffer(n * elem_size, MTLResourceOptions::StorageModeShared);

    let sort_lib = load_sort_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let gather_pipeline = get_pipeline_state(device, &sort_lib, &format!("gather_{}", dtype.kernel_suffix()))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let len_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe { *(len_buf.contents() as *mut u32) = n as u32; }

    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&gather_pipeline);
    enc.set_buffer(0, Some(&work_values), 0);
    enc.set_buffer(1, Some(indices), 0);
    enc.set_buffer(2, Some(&sorted_values), 0);
    enc.set_buffer(3, Some(&len_buf), 0);
    let num_groups = (n + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    enc.dispatch_thread_groups(
        MTLSize::new(num_groups, 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
    enc.end_encoding();

    Ok(sorted_values)
}

/// Sort-based groupby for a single aggregation (sum/min/max/count).
fn metal_groupby_sort(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    keys: &SharedBuffer,
    values: &SharedBuffer,
    dtype: DType,
    op: GroupOp,
) -> PyResult<(SharedBuffer, SharedBuffer)> {
    let library = load_groupby_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let suffix = dtype.kernel_suffix();
    let len = keys.len();
    let elem_size = dtype.size_in_bytes() as u64;

    let (sorted_keys, sorted_indices, n) = sort_keys_with_indices(device, queue, keys, len, dtype)?;

    let out_dtype = op.out_dtype(dtype);
    let out_keys = device.new_buffer((len.max(1) as u64) * elem_size, MTLResourceOptions::StorageModeShared);
    let out_vals = device.new_buffer(
        (len.max(1) as u64) * out_dtype.size_in_bytes() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let group_counter = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe { *(group_counter.contents() as *mut u32) = 0; }

    let direct_pipeline = get_pipeline_state(device, &library, &op.direct_kernel(suffix))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let cb = queue.new_command_buffer();

    let sorted_values = gather_values(device, values, &sorted_indices, len, n, dtype, cb)?;

    {
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&direct_pipeline);
        let len_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        unsafe { *(len_buf.contents() as *mut u32) = len as u32; }
        if op == GroupOp::Count {
            enc.set_buffer(0, Some(&sorted_keys), 0);
            enc.set_buffer(1, Some(&out_keys), 0);
            enc.set_buffer(2, Some(&out_vals), 0);
            enc.set_buffer(3, Some(&group_counter), 0);
            enc.set_buffer(4, Some(&len_buf), 0);
        } else {
            enc.set_buffer(0, Some(&sorted_keys), 0);
            enc.set_buffer(1, Some(&sorted_values), 0);
            enc.set_buffer(2, Some(&out_keys), 0);
            enc.set_buffer(3, Some(&out_vals), 0);
            enc.set_buffer(4, Some(&group_counter), 0);
            enc.set_buffer(5, Some(&len_buf), 0);
        }
        let num_groups = (len as u64 + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
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
            "Sort-based groupby failed: Metal command buffer error"
        ));
    }

    let num_groups = unsafe { *(group_counter.contents() as *const u32) as usize };

    Ok((
        SharedBuffer::from_metal_buffer(out_keys, num_groups, dtype),
        SharedBuffer::from_metal_buffer(out_vals, num_groups, out_dtype),
    ))
}

/// Sort-based mean: sort once, then a single fused sum+count leader-scan
/// kernel (see groupby_sum_count_direct_impl in groupby.metal) so sum and
/// count are guaranteed to share the same group ids, then divide on the CPU.
fn metal_groupby_mean_sort(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    keys: &SharedBuffer,
    values: &SharedBuffer,
    dtype: DType,
) -> PyResult<(SharedBuffer, SharedBuffer)> {
    let library = load_groupby_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let suffix = dtype.kernel_suffix();
    let len = keys.len();
    let elem_size = dtype.size_in_bytes() as u64;

    let (sorted_keys, sorted_indices, n) = sort_keys_with_indices(device, queue, keys, len, dtype)?;

    let out_keys = device.new_buffer((len.max(1) as u64) * elem_size, MTLResourceOptions::StorageModeShared);
    let out_sums = device.new_buffer((len.max(1) as u64) * elem_size, MTLResourceOptions::StorageModeShared);
    let out_counts = device.new_buffer((len.max(1) as u64) * 8, MTLResourceOptions::StorageModeShared);
    let group_counter = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe { *(group_counter.contents() as *mut u32) = 0; }

    let fused_pipeline = get_pipeline_state(device, &library, &format!("groupby_sum_count_direct_{suffix}"))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let cb = queue.new_command_buffer();

    let sorted_values = gather_values(device, values, &sorted_indices, len, n, dtype, cb)?;

    {
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&fused_pipeline);
        enc.set_buffer(0, Some(&sorted_keys), 0);
        enc.set_buffer(1, Some(&sorted_values), 0);
        enc.set_buffer(2, Some(&out_keys), 0);
        enc.set_buffer(3, Some(&out_sums), 0);
        enc.set_buffer(4, Some(&out_counts), 0);
        enc.set_buffer(5, Some(&group_counter), 0);
        let len_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
        unsafe { *(len_buf.contents() as *mut u32) = len as u32; }
        enc.set_buffer(6, Some(&len_buf), 0);
        let num_groups = (len as u64 + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
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
            "Sort-based groupby mean failed: Metal command buffer error"
        ));
    }

    let num_groups = unsafe { *(group_counter.contents() as *const u32) as usize };
    let (mean_buf, mean_dtype) = compute_mean_buffer(device, &out_sums, &out_counts, num_groups, dtype, 8);

    Ok((
        SharedBuffer::from_metal_buffer(out_keys, num_groups, dtype),
        SharedBuffer::from_metal_buffer(mean_buf, num_groups, mean_dtype),
    ))
}

// ============================================================================
// Public entry points
// ============================================================================

fn metal_groupby_dispatch_inner(keys: &SharedBuffer, values: &SharedBuffer, op: GroupOp) -> PyResult<(SharedBuffer, SharedBuffer)> {
    let dtype = validate_groupby_buffers(keys, values)?;
    let (device, queue) = MetalBackend::device_and_queue()?;

    if keys.len() <= HASH_MAX_LEN {
        metal_groupby_hash(device, queue, keys, values, dtype, op)
    } else {
        metal_groupby_sort(device, queue, keys, values, dtype, op)
    }
}

fn metal_groupby_dispatch(keys: &MetalSeries, values: &MetalSeries, op: GroupOp) -> PyResult<(MetalSeries, MetalSeries)> {
    let keys_buf = keys.as_numeric_checked()?;
    let values_buf = values.as_numeric_checked()?;
    let (k, v) = metal_groupby_dispatch_inner(keys_buf, values_buf, op)?;
    Ok((MetalSeries::from_numeric(k), MetalSeries::from_numeric(v)))
}

/// GroupBy sum. Selects hash-based (low cardinality) or sort-based (high
/// cardinality) aggregation based on input length.
#[pyfunction]
pub fn metal_groupby_sum(keys: &MetalSeries, values: &MetalSeries) -> PyResult<(MetalSeries, MetalSeries)> {
    metal_groupby_dispatch(keys, values, GroupOp::Sum)
}

/// GroupBy min.
#[pyfunction]
pub fn metal_groupby_min(keys: &MetalSeries, values: &MetalSeries) -> PyResult<(MetalSeries, MetalSeries)> {
    metal_groupby_dispatch(keys, values, GroupOp::Min)
}

/// GroupBy max.
#[pyfunction]
pub fn metal_groupby_max(keys: &MetalSeries, values: &MetalSeries) -> PyResult<(MetalSeries, MetalSeries)> {
    metal_groupby_dispatch(keys, values, GroupOp::Max)
}

/// GroupBy count (number of rows per group). `values` is only used for
/// dtype/length validation — the count itself only depends on `keys`.
#[pyfunction]
pub fn metal_groupby_count(keys: &MetalSeries, values: &MetalSeries) -> PyResult<(MetalSeries, MetalSeries)> {
    metal_groupby_dispatch(keys, values, GroupOp::Count)
}

/// GroupBy mean. Dispatches sum + count on the GPU against a shared
/// group-id assignment, then divides element-wise on the CPU, returning a
/// Float32 result (matching pandas' float-promoting mean behavior).
#[pyfunction]
pub fn metal_groupby_mean(keys: &MetalSeries, values: &MetalSeries) -> PyResult<(MetalSeries, MetalSeries)> {
    let keys_buf = keys.as_numeric_checked()?;
    let values_buf = values.as_numeric_checked()?;
    let dtype = validate_groupby_buffers(keys_buf, values_buf)?;
    let (device, queue) = MetalBackend::device_and_queue()?;

    let (k, v) = if keys_buf.len() <= HASH_MAX_LEN {
        metal_groupby_mean_hash(device, queue, keys_buf, values_buf, dtype)
    } else {
        metal_groupby_mean_sort(device, queue, keys_buf, values_buf, dtype)
    }?;
    Ok((MetalSeries::from_numeric(k), MetalSeries::from_numeric(v)))
}
