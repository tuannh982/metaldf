// String comparison kernel dispatch — eq/ne/lt/gt/le/ge (elementwise) and
// eq_scalar (each element vs. a single pattern string).
//
// All comparison kernels write int32 0/1 into a MetalSeries::Numeric(Int32)
// output, matching the boolean-mask convention used elsewhere in the crate.

use pyo3::prelude::*;
use metal::{Device, MTLSize, MTLResourceOptions};

use crate::backend::MetalBackend;
use crate::buffer::{SharedBuffer, DType};
use crate::series::{MetalSeries, SeriesData};
use crate::kernels::{load_strings_library, load_sort_library, get_pipeline_state};

const THREADGROUP_SIZE: u64 = 256;

/// Build a single-string offsets/chars Metal buffer pair for `s`.
///
/// Mirrors `MetalSeries::from_strings` for a series of length 1: guards
/// against zero-length Metal buffers (invalid) when `s` is empty. Used to
/// pass scalar pattern/replacement strings into kernels that otherwise
/// operate on `MetalSeries::Str` (offsets + chars) buffers.
fn build_scalar_string_buffers(device: &Device, s: &str) -> (metal::Buffer, metal::Buffer) {
    let bytes = s.as_bytes();
    let offsets_data: [i64; 2] = [0, bytes.len() as i64];
    let offsets_byte_len = std::mem::size_of::<[i64; 2]>();
    let offsets_buf = device.new_buffer(
        offsets_byte_len as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(
            offsets_data.as_ptr() as *const u8,
            offsets_buf.contents() as *mut u8,
            offsets_byte_len,
        );
    }
    let chars_buf = device.new_buffer(
        bytes.len().max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    if !bytes.is_empty() {
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                chars_buf.contents() as *mut u8,
                bytes.len(),
            );
        }
    }
    (offsets_buf, chars_buf)
}

fn dispatch_string_compare(
    kernel_name: &str,
    a: &MetalSeries,
    b: &MetalSeries,
) -> PyResult<MetalSeries> {
    let (a_offsets, a_chars) = a.as_str_checked()?;
    let (b_offsets, b_chars) = b.as_str_checked()?;

    if a.len != b.len {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "String series must have same length for comparison"
        ));
    }

    let (device, queue) = MetalBackend::device_and_queue()?;
    let library = load_strings_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let pipeline = get_pipeline_state(device, &library, kernel_name)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let n = a.len as u64;
    let output = device.new_buffer(n * 4, MTLResourceOptions::StorageModeShared);

    let len_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe { *(len_buf.contents() as *mut u32) = n as u32; }

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(a_offsets.metal_buffer()), 0);
    enc.set_buffer(1, Some(a_chars.metal_buffer()), 0);
    enc.set_buffer(2, Some(b_offsets.metal_buffer()), 0);
    enc.set_buffer(3, Some(b_chars.metal_buffer()), 0);
    enc.set_buffer(4, Some(&output), 0);
    enc.set_buffer(5, Some(&len_buf), 0);
    let num_groups = (n + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    enc.dispatch_thread_groups(
        MTLSize::new(num_groups, 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            format!("{} failed: Metal command buffer error", kernel_name)
        ));
    }

    Ok(MetalSeries::from_numeric(SharedBuffer::from_metal_buffer(output, n as usize, DType::Int32)))
}

fn dispatch_string_scalar_compare(
    kernel_name: &str,
    series: &MetalSeries,
    pattern: &str,
) -> PyResult<MetalSeries> {
    let (offsets, chars) = series.as_str_checked()?;

    let (device, queue) = MetalBackend::device_and_queue()?;
    let library = load_strings_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let pipeline = get_pipeline_state(device, &library, kernel_name)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let n = series.len as u64;

    // Build a single-string offsets/chars buffer pair for the pattern.
    let (pat_offsets_buf, pat_chars_buf) = build_scalar_string_buffers(device, pattern);

    let output = device.new_buffer(n * 4, MTLResourceOptions::StorageModeShared);
    let len_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe { *(len_buf.contents() as *mut u32) = n as u32; }

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(offsets.metal_buffer()), 0);
    enc.set_buffer(1, Some(chars.metal_buffer()), 0);
    enc.set_buffer(2, Some(&pat_offsets_buf), 0);
    enc.set_buffer(3, Some(&pat_chars_buf), 0);
    enc.set_buffer(4, Some(&output), 0);
    enc.set_buffer(5, Some(&len_buf), 0);
    let num_groups = (n + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    enc.dispatch_thread_groups(
        MTLSize::new(num_groups, 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            format!("{} failed: Metal command buffer error", kernel_name)
        ));
    }

    Ok(MetalSeries::from_numeric(SharedBuffer::from_metal_buffer(output, n as usize, DType::Int32)))
}

#[pyfunction]
pub fn metal_string_eq(a: &MetalSeries, b: &MetalSeries) -> PyResult<MetalSeries> {
    dispatch_string_compare("string_eq", a, b)
}

#[pyfunction]
pub fn metal_string_ne(a: &MetalSeries, b: &MetalSeries) -> PyResult<MetalSeries> {
    dispatch_string_compare("string_ne", a, b)
}

#[pyfunction]
pub fn metal_string_lt(a: &MetalSeries, b: &MetalSeries) -> PyResult<MetalSeries> {
    dispatch_string_compare("string_lt", a, b)
}

#[pyfunction]
pub fn metal_string_gt(a: &MetalSeries, b: &MetalSeries) -> PyResult<MetalSeries> {
    dispatch_string_compare("string_gt", a, b)
}

#[pyfunction]
pub fn metal_string_le(a: &MetalSeries, b: &MetalSeries) -> PyResult<MetalSeries> {
    dispatch_string_compare("string_le", a, b)
}

#[pyfunction]
pub fn metal_string_ge(a: &MetalSeries, b: &MetalSeries) -> PyResult<MetalSeries> {
    dispatch_string_compare("string_ge", a, b)
}

#[pyfunction]
pub fn metal_string_eq_scalar(series: &MetalSeries, pattern: &str) -> PyResult<MetalSeries> {
    dispatch_string_scalar_compare("string_eq_scalar", series, pattern)
}

#[pyfunction]
pub fn metal_string_contains(series: &MetalSeries, pattern: &str) -> PyResult<MetalSeries> {
    dispatch_string_scalar_compare("string_contains", series, pattern)
}

#[pyfunction]
pub fn metal_string_startswith(series: &MetalSeries, pattern: &str) -> PyResult<MetalSeries> {
    dispatch_string_scalar_compare("string_startswith", series, pattern)
}

#[pyfunction]
pub fn metal_string_endswith(series: &MetalSeries, pattern: &str) -> PyResult<MetalSeries> {
    dispatch_string_scalar_compare("string_endswith", series, pattern)
}

#[pyfunction]
pub fn metal_string_find(series: &MetalSeries, pattern: &str) -> PyResult<MetalSeries> {
    dispatch_string_scalar_compare("string_find", series, pattern)
}

// ---------------------------------------------------------------------------
// Transform kernels — lower/upper (same-length, single-pass) and
// strip/replace (variable-length, two-pass: GPU sizes -> CPU prefix-sum ->
// GPU write).
// ---------------------------------------------------------------------------

/// Same-length string transform (lower/upper): the output byte length equals
/// the input byte length, so offsets are reused unchanged and only the chars
/// buffer is recomputed.
fn dispatch_string_inplace_transform(
    kernel_name: &str,
    series: &MetalSeries,
) -> PyResult<MetalSeries> {
    let (offsets, chars) = series.as_str_checked()?;
    let (device, queue) = MetalBackend::device_and_queue()?;
    let library = load_strings_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let pipeline = get_pipeline_state(device, &library, kernel_name)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let n = series.len as u64;
    let total_chars = chars.len;

    let chars_out = device.new_buffer(
        total_chars.max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let len_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe { *(len_buf.contents() as *mut u32) = n as u32; }

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(offsets.metal_buffer()), 0);
    enc.set_buffer(1, Some(chars.metal_buffer()), 0);
    enc.set_buffer(2, Some(&chars_out), 0);
    enc.set_buffer(3, Some(&len_buf), 0);
    let num_groups = (n + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    enc.dispatch_thread_groups(
        MTLSize::new(num_groups, 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            format!("{} failed: Metal command buffer error", kernel_name)
        ));
    }

    let new_chars = SharedBuffer::from_metal_buffer(chars_out, total_chars, DType::Uint8);
    Ok(MetalSeries {
        data: SeriesData::Str { offsets: offsets.clone(), chars: new_chars },
        len: series.len,
        dtype: DType::Utf8,
    })
}

#[pyfunction]
pub fn metal_string_lower(series: &MetalSeries) -> PyResult<MetalSeries> {
    dispatch_string_inplace_transform("string_lower", series)
}

#[pyfunction]
pub fn metal_string_upper(series: &MetalSeries) -> PyResult<MetalSeries> {
    dispatch_string_inplace_transform("string_upper", series)
}

#[pyfunction]
pub fn metal_string_strip(series: &MetalSeries) -> PyResult<MetalSeries> {
    let (offsets, chars) = series.as_str_checked()?;
    let (device, queue) = MetalBackend::device_and_queue()?;
    let library = load_strings_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let n = series.len as u64;

    // Pass 1: compute sizes
    let sizes_buf = device.new_buffer(n * 8, MTLResourceOptions::StorageModeShared);
    let len_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe { *(len_buf.contents() as *mut u32) = n as u32; }

    let sizes_pl = get_pipeline_state(device, &library, "string_strip_sizes")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&sizes_pl);
    enc.set_buffer(0, Some(offsets.metal_buffer()), 0);
    enc.set_buffer(1, Some(chars.metal_buffer()), 0);
    enc.set_buffer(2, Some(&sizes_buf), 0);
    enc.set_buffer(3, Some(&len_buf), 0);
    let num_groups = (n + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    enc.dispatch_thread_groups(
        MTLSize::new(num_groups, 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "string_strip_sizes failed: Metal command buffer error".to_string()
        ));
    }

    // CPU prefix-sum to build new offsets
    let new_offsets_buf = device.new_buffer(
        (n as usize + 1) as u64 * 8,
        MTLResourceOptions::StorageModeShared,
    );
    let mut total_chars: i64 = 0;
    unsafe {
        let sizes_ptr = sizes_buf.contents() as *const i64;
        let offsets_ptr = new_offsets_buf.contents() as *mut i64;
        for i in 0..n as usize {
            *offsets_ptr.add(i) = total_chars;
            total_chars += *sizes_ptr.add(i);
        }
        *offsets_ptr.add(n as usize) = total_chars;
    }

    // Pass 2: write stripped chars
    let new_chars_buf = device.new_buffer(
        total_chars.max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let write_pl = get_pipeline_state(device, &library, "string_strip_write")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let cb2 = queue.new_command_buffer();
    let enc2 = cb2.new_compute_command_encoder();
    enc2.set_compute_pipeline_state(&write_pl);
    enc2.set_buffer(0, Some(offsets.metal_buffer()), 0);
    enc2.set_buffer(1, Some(chars.metal_buffer()), 0);
    enc2.set_buffer(2, Some(&new_offsets_buf), 0);
    enc2.set_buffer(3, Some(&new_chars_buf), 0);
    enc2.set_buffer(4, Some(&len_buf), 0);
    enc2.dispatch_thread_groups(
        MTLSize::new(num_groups, 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
    enc2.end_encoding();
    cb2.commit();
    cb2.wait_until_completed();

    if cb2.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "string_strip_write failed: Metal command buffer error".to_string()
        ));
    }

    let new_offsets = SharedBuffer::from_metal_buffer(new_offsets_buf, n as usize + 1, DType::Int64);
    let new_chars = SharedBuffer::from_metal_buffer(new_chars_buf, total_chars as usize, DType::Uint8);

    Ok(MetalSeries {
        data: SeriesData::Str { offsets: new_offsets, chars: new_chars },
        len: series.len,
        dtype: DType::Utf8,
    })
}

#[pyfunction]
pub fn metal_string_replace(series: &MetalSeries, pat: &str, repl: &str) -> PyResult<MetalSeries> {
    let (offsets, chars) = series.as_str_checked()?;
    let (device, queue) = MetalBackend::device_and_queue()?;
    let library = load_strings_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let n = series.len as u64;

    // Build pattern and replacement single-string buffers (same as scalar compare)
    let (pat_offsets_buf, pat_chars_buf) = build_scalar_string_buffers(device, pat);
    let (repl_offsets_buf, repl_chars_buf) = build_scalar_string_buffers(device, repl);

    let len_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe { *(len_buf.contents() as *mut u32) = n as u32; }

    // Pass 1: compute sizes
    let sizes_buf = device.new_buffer(n * 8, MTLResourceOptions::StorageModeShared);
    let sizes_pl = get_pipeline_state(device, &library, "string_replace_sizes")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&sizes_pl);
    enc.set_buffer(0, Some(offsets.metal_buffer()), 0);
    enc.set_buffer(1, Some(chars.metal_buffer()), 0);
    enc.set_buffer(2, Some(&pat_offsets_buf), 0);
    enc.set_buffer(3, Some(&pat_chars_buf), 0);
    enc.set_buffer(4, Some(&repl_offsets_buf), 0);
    enc.set_buffer(5, Some(&repl_chars_buf), 0);
    enc.set_buffer(6, Some(&sizes_buf), 0);
    enc.set_buffer(7, Some(&len_buf), 0);
    let num_groups = (n + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    enc.dispatch_thread_groups(
        MTLSize::new(num_groups, 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "string_replace_sizes failed: Metal command buffer error".to_string()
        ));
    }

    // CPU prefix-sum to build new offsets
    let new_offsets_buf = device.new_buffer(
        (n as usize + 1) as u64 * 8,
        MTLResourceOptions::StorageModeShared,
    );
    let mut total_chars: i64 = 0;
    unsafe {
        let sizes_ptr = sizes_buf.contents() as *const i64;
        let offsets_ptr = new_offsets_buf.contents() as *mut i64;
        for i in 0..n as usize {
            *offsets_ptr.add(i) = total_chars;
            total_chars += *sizes_ptr.add(i);
        }
        *offsets_ptr.add(n as usize) = total_chars;
    }

    // Pass 2: write replaced chars
    let new_chars_buf = device.new_buffer(
        total_chars.max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let write_pl = get_pipeline_state(device, &library, "string_replace_write")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let cb2 = queue.new_command_buffer();
    let enc2 = cb2.new_compute_command_encoder();
    enc2.set_compute_pipeline_state(&write_pl);
    enc2.set_buffer(0, Some(offsets.metal_buffer()), 0);
    enc2.set_buffer(1, Some(chars.metal_buffer()), 0);
    enc2.set_buffer(2, Some(&pat_offsets_buf), 0);
    enc2.set_buffer(3, Some(&pat_chars_buf), 0);
    enc2.set_buffer(4, Some(&repl_offsets_buf), 0);
    enc2.set_buffer(5, Some(&repl_chars_buf), 0);
    enc2.set_buffer(6, Some(&new_offsets_buf), 0);
    enc2.set_buffer(7, Some(&new_chars_buf), 0);
    enc2.set_buffer(8, Some(&len_buf), 0);
    enc2.dispatch_thread_groups(
        MTLSize::new(num_groups, 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
    enc2.end_encoding();
    cb2.commit();
    cb2.wait_until_completed();

    if cb2.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "string_replace_write failed: Metal command buffer error".to_string()
        ));
    }

    let new_offsets = SharedBuffer::from_metal_buffer(new_offsets_buf, n as usize + 1, DType::Int64);
    let new_chars = SharedBuffer::from_metal_buffer(new_chars_buf, total_chars as usize, DType::Uint8);

    Ok(MetalSeries {
        data: SeriesData::Str { offsets: new_offsets, chars: new_chars },
        len: series.len,
        dtype: DType::Utf8,
    })
}

// ---------------------------------------------------------------------------
// Sort — bitonic sort on an index array comparing via string_compare, then
// a two-pass gather (sizes on GPU, prefix-sum on CPU, write on GPU) to
// materialize the reordered string series. Mirrors the strip/replace
// two-pass gather pattern above.
// ---------------------------------------------------------------------------

/// Run every (stage, step) pass of the string-index bitonic sort network,
/// batched into a SINGLE command buffer (mirrors `run_bitonic_passes` in
/// sort.rs, which does the same for the numeric bitonic sort). Without this,
/// each pass would be its own command buffer + synchronous GPU round-trip —
/// O(log2(padded_n)^2 / 2) of them, e.g. ~253 for 5M rows padded to 2^23.
///
/// `indices_buf` holds `padded_n` u32 slots to be sorted in place, comparing
/// the strings each index points to via `string_compare`. Positions
/// `[real_n, padded_n)` must already hold the sentinel value `real_n` (an
/// out-of-bounds string index), which `string_bitonic_sort` recognizes as
/// padding that always sorts last.
fn run_string_bitonic_passes(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    library: &metal::Library,
    indices_buf: &metal::Buffer,
    offsets: &SharedBuffer,
    chars: &SharedBuffer,
    padded_n: u64,
    real_n: usize,
) -> PyResult<()> {
    let sort_pl = get_pipeline_state(device, library, "string_bitonic_sort")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let num_stages = padded_n.trailing_zeros();
    let half_n = padded_n / 2;
    let num_groups = (half_n + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;

    // Total (stage, step) passes, for uniform buffer pre-allocation: stage 0
    // has 1 step, stage 1 has 2 steps, ..., stage (num_stages-1) has
    // num_stages steps.
    let total_passes: usize = (1..=num_stages as usize).sum();

    // Uniform layout per pass: [padded_n, stage, step, real_n] (4 x u32 = 16 bytes).
    // Each buffer binding in the kernel dereferences a single uint, so we
    // point each of buffers 3..6 at the right 4-byte slot within this one
    // pre-populated buffer via offsets — exactly like sort.rs's global_uniform.
    let uniform_buf = device.new_buffer(
        (total_passes * 4 * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        let ptr = uniform_buf.contents() as *mut u32;
        let mut i = 0usize;
        for stage in 0..num_stages {
            for step in (0..=stage).rev() {
                *ptr.add(i * 4) = padded_n as u32;
                *ptr.add(i * 4 + 1) = stage;
                *ptr.add(i * 4 + 2) = step;
                *ptr.add(i * 4 + 3) = real_n as u32;
                i += 1;
            }
        }
    }

    let cb = queue.new_command_buffer();
    let mut i = 0usize;
    for stage in 0..num_stages {
        for _step in (0..=stage).rev() {
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&sort_pl);
            enc.set_buffer(0, Some(indices_buf), 0);
            enc.set_buffer(1, Some(offsets.metal_buffer()), 0);
            enc.set_buffer(2, Some(chars.metal_buffer()), 0);
            let off = (i * 4 * 4) as u64;
            enc.set_buffer(3, Some(&uniform_buf), off);
            enc.set_buffer(4, Some(&uniform_buf), off + 4);
            enc.set_buffer(5, Some(&uniform_buf), off + 8);
            enc.set_buffer(6, Some(&uniform_buf), off + 12);
            enc.dispatch_thread_groups(
                MTLSize::new(num_groups, 1, 1),
                MTLSize::new(THREADGROUP_SIZE, 1, 1),
            );
            enc.end_encoding();
            i += 1;
        }
    }

    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "string_bitonic_sort failed: Metal command buffer error"
        ));
    }
    Ok(())
}

#[pyfunction]
pub fn metal_string_sort(series: &MetalSeries, ascending: bool) -> PyResult<MetalSeries> {
    let (offsets, chars) = series.as_str_checked()?;
    let n = series.len;

    if n <= 1 {
        return Ok(MetalSeries {
            data: SeriesData::Str { offsets: offsets.clone(), chars: chars.clone() },
            len: n,
            dtype: DType::Utf8,
        });
    }

    let (device, queue) = MetalBackend::device_and_queue()?;
    let library = load_strings_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let padded_n = n.next_power_of_two() as u64;

    // Identity index array, padded with the sentinel value `n` (an
    // out-of-bounds string index) — the sort kernel treats any index
    // >= real_n as padding and sorts it last.
    let indices_buf = device.new_buffer(padded_n * 4, MTLResourceOptions::StorageModeShared);
    unsafe {
        let ptr = indices_buf.contents() as *mut u32;
        for i in 0..padded_n as usize {
            *ptr.add(i) = if i < n { i as u32 } else { n as u32 };
        }
    }

    run_string_bitonic_passes(device, queue, &library, &indices_buf, offsets, chars, padded_n, n)?;

    // If descending, reverse the (now ascending-sorted) first n indices.
    if !ascending {
        unsafe {
            let ptr = indices_buf.contents() as *mut u32;
            let slice = std::slice::from_raw_parts_mut(ptr, n);
            slice.reverse();
        }
    }

    // Gather pass 1: sizes
    let n64 = n as u64;
    let sizes_buf = device.new_buffer(n64 * 8, MTLResourceOptions::StorageModeShared);
    let len_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe { *(len_buf.contents() as *mut u32) = n as u32; }

    let sizes_pl = get_pipeline_state(device, &library, "string_gather_sizes")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let gather_num_groups = (n64 + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&sizes_pl);
    enc.set_buffer(0, Some(&indices_buf), 0);
    enc.set_buffer(1, Some(offsets.metal_buffer()), 0);
    enc.set_buffer(2, Some(&sizes_buf), 0);
    enc.set_buffer(3, Some(&len_buf), 0);
    enc.dispatch_thread_groups(
        MTLSize::new(gather_num_groups, 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "string_gather_sizes failed: Metal command buffer error".to_string()
        ));
    }

    // CPU prefix-sum to build new offsets
    let new_offsets_buf = device.new_buffer((n + 1) as u64 * 8, MTLResourceOptions::StorageModeShared);
    let mut total: i64 = 0;
    unsafe {
        let sp = sizes_buf.contents() as *const i64;
        let op = new_offsets_buf.contents() as *mut i64;
        for i in 0..n {
            *op.add(i) = total;
            total += *sp.add(i);
        }
        *op.add(n) = total;
    }

    // Gather pass 2: write chars
    let new_chars_buf = device.new_buffer(total.max(1) as u64, MTLResourceOptions::StorageModeShared);

    let write_pl = get_pipeline_state(device, &library, "string_gather_write")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let cb2 = queue.new_command_buffer();
    let enc2 = cb2.new_compute_command_encoder();
    enc2.set_compute_pipeline_state(&write_pl);
    enc2.set_buffer(0, Some(&indices_buf), 0);
    enc2.set_buffer(1, Some(offsets.metal_buffer()), 0);
    enc2.set_buffer(2, Some(chars.metal_buffer()), 0);
    enc2.set_buffer(3, Some(&new_offsets_buf), 0);
    enc2.set_buffer(4, Some(&new_chars_buf), 0);
    enc2.set_buffer(5, Some(&len_buf), 0);
    enc2.dispatch_thread_groups(
        MTLSize::new(gather_num_groups, 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
    enc2.end_encoding();
    cb2.commit();
    cb2.wait_until_completed();

    if cb2.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "string_gather_write failed: Metal command buffer error".to_string()
        ));
    }

    let new_offsets = SharedBuffer::from_metal_buffer(new_offsets_buf, n + 1, DType::Int64);
    let new_chars = SharedBuffer::from_metal_buffer(new_chars_buf, total as usize, DType::Uint8);

    Ok(MetalSeries {
        data: SeriesData::Str { offsets: new_offsets, chars: new_chars },
        len: n,
        dtype: DType::Utf8,
    })
}

// ---------------------------------------------------------------------------
// String-key groupby — hash-based (<=500K rows) + sort-based (>500K rows).
//
// Hash-based: 3-pass GPU (build, accumulate, compact) with FNV-1a hash +
// string_equals probing.  Sort-based: bitonic sort on string indices, then
// a parallel leader-scan that walks sorted keys and accumulates per-group.
//
// Supported aggregations: sum, min, max, count, mean.
// Gather pass reuses string_gather_sizes / string_gather_write from sort.metal.
// ---------------------------------------------------------------------------

/// Selects hash-based (len <= HASH_MAX_LEN) or sort-based (len > HASH_MAX_LEN)
/// aggregation for string groupby.
const STRING_HASH_MAX_LEN: usize = 500_000;

/// Return an empty groupby result (0 groups).
fn empty_string_groupby(
    device: &metal::Device,
    out_dtype: DType,
) -> PyResult<(MetalSeries, MetalSeries)> {
    let offsets_buf = device.new_buffer(8, MTLResourceOptions::StorageModeShared);
    unsafe { *(offsets_buf.contents() as *mut i64) = 0; }
    let chars_buf = device.new_buffer(1, MTLResourceOptions::StorageModeShared);
    let val_buf = device.new_buffer(
        out_dtype.size_in_bytes().max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    Ok((
        MetalSeries {
            data: SeriesData::Str {
                offsets: SharedBuffer::from_metal_buffer(offsets_buf, 1, DType::Int64),
                chars: SharedBuffer::from_metal_buffer(chars_buf, 0, DType::Uint8),
            },
            len: 0,
            dtype: DType::Utf8,
        },
        MetalSeries::from_numeric(SharedBuffer::from_metal_buffer(val_buf, 0, out_dtype)),
    ))
}

/// Gather unique key strings from `out_key_indices` into a new MetalSeries.
fn gather_key_strings(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    library: &metal::Library,
    key_offsets: &SharedBuffer,
    key_chars: &SharedBuffer,
    out_key_indices: &metal::Buffer,
    num_groups: usize,
) -> PyResult<MetalSeries> {
    let ng = num_groups as u64;
    let sizes_buf = device.new_buffer(ng * 8, MTLResourceOptions::StorageModeShared);
    let ng_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe { *(ng_buf.contents() as *mut u32) = num_groups as u32; }

    let gather_tg = (ng + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;

    // Pass 1: compute per-string sizes
    {
        let pl = get_pipeline_state(device, library, "string_gather_sizes")
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pl);
        enc.set_buffer(0, Some(out_key_indices), 0);
        enc.set_buffer(1, Some(key_offsets.metal_buffer()), 0);
        enc.set_buffer(2, Some(&sizes_buf), 0);
        enc.set_buffer(3, Some(&ng_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(gather_tg, 1, 1),
            MTLSize::new(THREADGROUP_SIZE, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        if cb.status() == metal::MTLCommandBufferStatus::Error {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "string_gather_sizes failed: Metal command buffer error"
            ));
        }
    }

    // CPU prefix-sum to build new offsets
    let new_offsets_buf = device.new_buffer(
        (num_groups + 1) as u64 * 8,
        MTLResourceOptions::StorageModeShared,
    );
    let mut total_chars: i64 = 0;
    unsafe {
        let sp = sizes_buf.contents() as *const i64;
        let op = new_offsets_buf.contents() as *mut i64;
        for i in 0..num_groups {
            *op.add(i) = total_chars;
            total_chars += *sp.add(i);
        }
        *op.add(num_groups) = total_chars;
    }

    // Pass 2: write gathered chars
    let new_chars_buf = device.new_buffer(
        total_chars.max(1) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    {
        let pl = get_pipeline_state(device, library, "string_gather_write")
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pl);
        enc.set_buffer(0, Some(out_key_indices), 0);
        enc.set_buffer(1, Some(key_offsets.metal_buffer()), 0);
        enc.set_buffer(2, Some(key_chars.metal_buffer()), 0);
        enc.set_buffer(3, Some(&new_offsets_buf), 0);
        enc.set_buffer(4, Some(&new_chars_buf), 0);
        enc.set_buffer(5, Some(&ng_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(gather_tg, 1, 1),
            MTLSize::new(THREADGROUP_SIZE, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        if cb.status() == metal::MTLCommandBufferStatus::Error {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "string_gather_write failed: Metal command buffer error"
            ));
        }
    }

    let new_offsets = SharedBuffer::from_metal_buffer(new_offsets_buf, num_groups + 1, DType::Int64);
    let new_chars = SharedBuffer::from_metal_buffer(new_chars_buf, total_chars as usize, DType::Uint8);

    Ok(MetalSeries {
        data: SeriesData::Str { offsets: new_offsets, chars: new_chars },
        len: num_groups,
        dtype: DType::Utf8,
    })
}

/// Initialize a hash accumulator buffer with the correct identity value.
fn init_string_accum(buf: &metal::Buffer, n: u64, val_dtype: DType, agg: &str) {
    unsafe {
        let len = n as usize;
        match (val_dtype, agg) {
            (_, "count")              => { let p = buf.contents() as *mut u32; for i in 0..len { *p.add(i) = 0; } }
            (DType::Float32, "sum")   => { let p = buf.contents() as *mut u32; for i in 0..len { *p.add(i) = 0; } }
            (DType::Int32,   "sum")   => { let p = buf.contents() as *mut i32; for i in 0..len { *p.add(i) = 0; } }
            (DType::Float32, "min")   => { let p = buf.contents() as *mut u32; for i in 0..len { *p.add(i) = f32::INFINITY.to_bits(); } }
            (DType::Int32,   "min")   => { let p = buf.contents() as *mut i32; for i in 0..len { *p.add(i) = i32::MAX; } }
            (DType::Float32, "max")   => { let p = buf.contents() as *mut u32; for i in 0..len { *p.add(i) = f32::NEG_INFINITY.to_bits(); } }
            (DType::Int32,   "max")   => { let p = buf.contents() as *mut i32; for i in 0..len { *p.add(i) = i32::MIN; } }
            _ => unreachable!("string groupby: unsupported dtype/agg combo"),
        }
    }
}

/// Kernel name for accumulate pass.
fn string_accum_kernel(agg: &str, val_suffix: &str) -> String {
    match agg {
        "sum" => format!("string_groupby_hash_sum_{}", val_suffix),
        "min" => format!("string_groupby_hash_min_{}", val_suffix),
        "max" => format!("string_groupby_hash_max_{}", val_suffix),
        "count" => "string_groupby_hash_count".to_string(),
        _ => unreachable!(),
    }
}

/// Kernel name for compact pass.
fn string_compact_kernel(agg: &str, val_suffix: &str) -> String {
    match agg {
        "sum" => format!("string_groupby_hash_compact_sum_{}", val_suffix),
        "min" | "max" => format!("string_groupby_hash_compact_minmax_{}", val_suffix),
        "count" => "string_groupby_hash_compact_count".to_string(),
        _ => unreachable!(),
    }
}

// ============================================================================
// Hash-based path (<=500K rows)
// ============================================================================

/// Hash-based string groupby for sum/min/max/count.
fn metal_string_groupby_hash(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    key_offsets: &SharedBuffer,
    key_chars: &SharedBuffer,
    values_buf: &SharedBuffer,
    n: usize,
    val_dtype: DType,
    agg: &str,
) -> PyResult<(MetalSeries, MetalSeries)> {
    let library = load_strings_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let n64 = n as u64;
    let val_suffix = val_dtype.kernel_suffix();
    let out_dtype = if agg == "count" { DType::Float32 } else { val_dtype };

    // Hash table sizing: 2x data size, power of two, clamped to [1024, 4M].
    let table_size = (1024u64)
        .max((n64 * 2).next_power_of_two())
        .min(4_194_304);

    let table_hashes = device.new_buffer(table_size * 4, MTLResourceOptions::StorageModeShared);
    let table_gids = device.new_buffer(table_size * 4, MTLResourceOptions::StorageModeShared);
    let table_key_indices = device.new_buffer(table_size * 4, MTLResourceOptions::StorageModeShared);
    let group_counter = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

    unsafe {
        let hp = table_hashes.contents() as *mut u32;
        let kp = table_key_indices.contents() as *mut u32;
        for i in 0..table_size as usize {
            *hp.add(i) = 0xFFFF_FFFFu32;
            *kp.add(i) = 0xFFFF_FFFFu32;
        }
        *(group_counter.contents() as *mut u32) = 0;
    }

    let accum = device.new_buffer(n64.max(1) * 4, MTLResourceOptions::StorageModeShared);
    init_string_accum(&accum, n64.max(1), val_dtype, agg);

    let out_key_indices = device.new_buffer(n64.max(1) * 4, MTLResourceOptions::StorageModeShared);
    let out_values = device.new_buffer(
        n64.max(1) * out_dtype.size_in_bytes() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let len_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let ts_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe {
        *(len_buf.contents() as *mut u32) = n as u32;
        *(ts_buf.contents() as *mut u32) = table_size as u32;
    }

    let cb = queue.new_command_buffer();
    let num_tg = (n64 + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    let table_tg = (table_size + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;

    // Pass 1: Build hash table
    {
        let pl = get_pipeline_state(device, &library, "string_groupby_hash_build")
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pl);
        enc.set_buffer(0, Some(key_offsets.metal_buffer()), 0);
        enc.set_buffer(1, Some(key_chars.metal_buffer()), 0);
        enc.set_buffer(2, Some(&table_hashes), 0);
        enc.set_buffer(3, Some(&table_gids), 0);
        enc.set_buffer(4, Some(&table_key_indices), 0);
        enc.set_buffer(5, Some(&group_counter), 0);
        enc.set_buffer(6, Some(&len_buf), 0);
        enc.set_buffer(7, Some(&ts_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(num_tg, 1, 1),
            MTLSize::new(THREADGROUP_SIZE, 1, 1),
        );
        enc.end_encoding();
    }

    // Pass 2: Accumulate
    {
        let kernel_name = string_accum_kernel(agg, val_suffix);
        let pl = get_pipeline_state(device, &library, &kernel_name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pl);

        if agg == "count" {
            enc.set_buffer(0, Some(key_offsets.metal_buffer()), 0);
            enc.set_buffer(1, Some(key_chars.metal_buffer()), 0);
            enc.set_buffer(2, Some(&table_hashes), 0);
            enc.set_buffer(3, Some(&table_gids), 0);
            enc.set_buffer(4, Some(&table_key_indices), 0);
            enc.set_buffer(5, Some(&accum), 0);
            enc.set_buffer(6, Some(&len_buf), 0);
            enc.set_buffer(7, Some(&ts_buf), 0);
        } else {
            enc.set_buffer(0, Some(key_offsets.metal_buffer()), 0);
            enc.set_buffer(1, Some(key_chars.metal_buffer()), 0);
            enc.set_buffer(2, Some(values_buf.metal_buffer()), 0);
            enc.set_buffer(3, Some(&table_hashes), 0);
            enc.set_buffer(4, Some(&table_gids), 0);
            enc.set_buffer(5, Some(&table_key_indices), 0);
            enc.set_buffer(6, Some(&accum), 0);
            enc.set_buffer(7, Some(&len_buf), 0);
            enc.set_buffer(8, Some(&ts_buf), 0);
        }

        enc.dispatch_thread_groups(
            MTLSize::new(num_tg, 1, 1),
            MTLSize::new(THREADGROUP_SIZE, 1, 1),
        );
        enc.end_encoding();
    }

    // Pass 3: Compact
    {
        let kernel_name = string_compact_kernel(agg, val_suffix);
        let pl = get_pipeline_state(device, &library, &kernel_name)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pl);
        enc.set_buffer(0, Some(&table_hashes), 0);
        enc.set_buffer(1, Some(&table_gids), 0);
        enc.set_buffer(2, Some(&table_key_indices), 0);
        enc.set_buffer(3, Some(&accum), 0);
        enc.set_buffer(4, Some(&out_key_indices), 0);
        enc.set_buffer(5, Some(&out_values), 0);
        enc.set_buffer(6, Some(&ts_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(table_tg, 1, 1),
            MTLSize::new(THREADGROUP_SIZE, 1, 1),
        );
        enc.end_encoding();
    }

    cb.commit();
    cb.wait_until_completed();
    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "String groupby (hash) failed: Metal command buffer error"
        ));
    }

    let num_groups = unsafe { *(group_counter.contents() as *const u32) as usize };

    if num_groups == 0 {
        return empty_string_groupby(device, out_dtype);
    }

    let key_series = gather_key_strings(
        device, queue, &library, key_offsets, key_chars, &out_key_indices, num_groups,
    )?;

    let value_series = MetalSeries::from_numeric(
        SharedBuffer::from_metal_buffer(out_values, num_groups, out_dtype)
    );

    Ok((key_series, value_series))
}

/// Hash-based mean: one hash build, sum + count accumulate against the same
/// frozen table, compact sum, then divide on CPU.
fn metal_string_groupby_mean_hash(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    key_offsets: &SharedBuffer,
    key_chars: &SharedBuffer,
    values_buf: &SharedBuffer,
    n: usize,
    val_dtype: DType,
) -> PyResult<(MetalSeries, MetalSeries)> {
    let library = load_strings_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let n64 = n as u64;
    let val_suffix = val_dtype.kernel_suffix();

    let table_size = (1024u64)
        .max((n64 * 2).next_power_of_two())
        .min(4_194_304);

    let table_hashes = device.new_buffer(table_size * 4, MTLResourceOptions::StorageModeShared);
    let table_gids = device.new_buffer(table_size * 4, MTLResourceOptions::StorageModeShared);
    let table_key_indices = device.new_buffer(table_size * 4, MTLResourceOptions::StorageModeShared);
    let group_counter = device.new_buffer(4, MTLResourceOptions::StorageModeShared);

    unsafe {
        let hp = table_hashes.contents() as *mut u32;
        let kp = table_key_indices.contents() as *mut u32;
        for i in 0..table_size as usize {
            *hp.add(i) = 0xFFFF_FFFFu32;
            *kp.add(i) = 0xFFFF_FFFFu32;
        }
        *(group_counter.contents() as *mut u32) = 0;
    }

    let sum_accum = device.new_buffer(n64.max(1) * 4, MTLResourceOptions::StorageModeShared);
    init_string_accum(&sum_accum, n64.max(1), val_dtype, "sum");
    let count_accum = device.new_buffer(n64.max(1) * 4, MTLResourceOptions::StorageModeShared);
    init_string_accum(&count_accum, n64.max(1), val_dtype, "count");

    let out_key_indices = device.new_buffer(n64.max(1) * 4, MTLResourceOptions::StorageModeShared);
    let out_sum_values = device.new_buffer(
        n64.max(1) * val_dtype.size_in_bytes() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let len_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let ts_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe {
        *(len_buf.contents() as *mut u32) = n as u32;
        *(ts_buf.contents() as *mut u32) = table_size as u32;
    }

    let cb = queue.new_command_buffer();
    let num_tg = (n64 + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    let table_tg = (table_size + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;

    // Build
    {
        let pl = get_pipeline_state(device, &library, "string_groupby_hash_build")
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pl);
        enc.set_buffer(0, Some(key_offsets.metal_buffer()), 0);
        enc.set_buffer(1, Some(key_chars.metal_buffer()), 0);
        enc.set_buffer(2, Some(&table_hashes), 0);
        enc.set_buffer(3, Some(&table_gids), 0);
        enc.set_buffer(4, Some(&table_key_indices), 0);
        enc.set_buffer(5, Some(&group_counter), 0);
        enc.set_buffer(6, Some(&len_buf), 0);
        enc.set_buffer(7, Some(&ts_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(num_tg, 1, 1),
            MTLSize::new(THREADGROUP_SIZE, 1, 1),
        );
        enc.end_encoding();
    }

    // Sum accumulate
    {
        let pl = get_pipeline_state(device, &library, &format!("string_groupby_hash_sum_{}", val_suffix))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pl);
        enc.set_buffer(0, Some(key_offsets.metal_buffer()), 0);
        enc.set_buffer(1, Some(key_chars.metal_buffer()), 0);
        enc.set_buffer(2, Some(values_buf.metal_buffer()), 0);
        enc.set_buffer(3, Some(&table_hashes), 0);
        enc.set_buffer(4, Some(&table_gids), 0);
        enc.set_buffer(5, Some(&table_key_indices), 0);
        enc.set_buffer(6, Some(&sum_accum), 0);
        enc.set_buffer(7, Some(&len_buf), 0);
        enc.set_buffer(8, Some(&ts_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(num_tg, 1, 1),
            MTLSize::new(THREADGROUP_SIZE, 1, 1),
        );
        enc.end_encoding();
    }

    // Count accumulate (same table)
    {
        let pl = get_pipeline_state(device, &library, "string_groupby_hash_count")
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pl);
        enc.set_buffer(0, Some(key_offsets.metal_buffer()), 0);
        enc.set_buffer(1, Some(key_chars.metal_buffer()), 0);
        enc.set_buffer(2, Some(&table_hashes), 0);
        enc.set_buffer(3, Some(&table_gids), 0);
        enc.set_buffer(4, Some(&table_key_indices), 0);
        enc.set_buffer(5, Some(&count_accum), 0);
        enc.set_buffer(6, Some(&len_buf), 0);
        enc.set_buffer(7, Some(&ts_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(num_tg, 1, 1),
            MTLSize::new(THREADGROUP_SIZE, 1, 1),
        );
        enc.end_encoding();
    }

    // Compact sum + keys
    {
        let pl = get_pipeline_state(device, &library, &format!("string_groupby_hash_compact_sum_{}", val_suffix))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pl);
        enc.set_buffer(0, Some(&table_hashes), 0);
        enc.set_buffer(1, Some(&table_gids), 0);
        enc.set_buffer(2, Some(&table_key_indices), 0);
        enc.set_buffer(3, Some(&sum_accum), 0);
        enc.set_buffer(4, Some(&out_key_indices), 0);
        enc.set_buffer(5, Some(&out_sum_values), 0);
        enc.set_buffer(6, Some(&ts_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize::new(table_tg, 1, 1),
            MTLSize::new(THREADGROUP_SIZE, 1, 1),
        );
        enc.end_encoding();
    }

    cb.commit();
    cb.wait_until_completed();
    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "String groupby mean (hash) failed: Metal command buffer error"
        ));
    }

    let num_groups = unsafe { *(group_counter.contents() as *const u32) as usize };

    if num_groups == 0 {
        return empty_string_groupby(device, DType::Float32);
    }

    // Divide sum/count on CPU to get mean (hash path: count is u32)
    let (mean_buf, mean_dtype) = string_compute_mean(
        device, &out_sum_values, &count_accum, num_groups, val_dtype, false,
    );

    let key_series = gather_key_strings(
        device, queue, &library, key_offsets, key_chars, &out_key_indices, num_groups,
    )?;

    let value_series = MetalSeries::from_numeric(
        SharedBuffer::from_metal_buffer(mean_buf, num_groups, mean_dtype)
    );

    Ok((key_series, value_series))
}

// ============================================================================
// Sort-based path (>500K rows)
// ============================================================================

/// Run bitonic sort on string keys, returning the sorted index array (uint32).
/// Does NOT gather the strings.
fn string_bitonic_sort_indices(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    offsets: &SharedBuffer,
    chars: &SharedBuffer,
    n: usize,
) -> PyResult<metal::Buffer> {
    let library = load_strings_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let padded_n = n.next_power_of_two() as u64;

    let indices_buf = device.new_buffer(padded_n * 4, MTLResourceOptions::StorageModeShared);
    unsafe {
        let ptr = indices_buf.contents() as *mut u32;
        for i in 0..padded_n as usize {
            *ptr.add(i) = if i < n { i as u32 } else { n as u32 };
        }
    }

    run_string_bitonic_passes(device, queue, &library, &indices_buf, offsets, chars, padded_n, n)?;

    Ok(indices_buf)
}

/// Gather numeric values by sorted indices, producing a new buffer.
fn string_gather_values(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    values_buf: &SharedBuffer,
    sorted_indices: &metal::Buffer,
    n: usize,
    val_dtype: DType,
) -> PyResult<metal::Buffer> {
    let sort_lib = load_sort_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let elem_size = val_dtype.size_in_bytes() as u64;
    let n64 = n as u64;

    // Copy values into a working buffer
    let work_values = device.new_buffer(n64 * elem_size, MTLResourceOptions::StorageModeShared);
    unsafe {
        std::ptr::copy_nonoverlapping(
            values_buf.metal_buffer().contents(),
            work_values.contents(),
            n * elem_size as usize,
        );
    }

    let sorted_values = device.new_buffer(n64 * elem_size, MTLResourceOptions::StorageModeShared);

    let gather_pl = get_pipeline_state(device, &sort_lib, &format!("gather_{}", val_dtype.kernel_suffix()))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let len_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe { *(len_buf.contents() as *mut u32) = n as u32; }

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&gather_pl);
    enc.set_buffer(0, Some(&work_values), 0);
    enc.set_buffer(1, Some(sorted_indices), 0);
    enc.set_buffer(2, Some(&sorted_values), 0);
    enc.set_buffer(3, Some(&len_buf), 0);
    let num_groups = (n64 + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    enc.dispatch_thread_groups(
        MTLSize::new(num_groups, 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "gather values failed: Metal command buffer error"
        ));
    }

    Ok(sorted_values)
}

/// Sort-based string groupby for sum/min/max/count.
fn metal_string_groupby_sort(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    key_offsets: &SharedBuffer,
    key_chars: &SharedBuffer,
    values_buf: &SharedBuffer,
    n: usize,
    val_dtype: DType,
    agg: &str,
) -> PyResult<(MetalSeries, MetalSeries)> {
    let library = load_strings_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let n64 = n as u64;
    let val_suffix = val_dtype.kernel_suffix();
    let out_dtype = if agg == "count" { DType::Float32 } else { val_dtype };

    // Step 1: Sort string keys to get sorted index array
    let sorted_indices = string_bitonic_sort_indices(device, queue, key_offsets, key_chars, n)?;

    // Step 2: Gather numeric values by sorted indices
    let sorted_values = string_gather_values(device, queue, values_buf, &sorted_indices, n, val_dtype)?;

    // Step 3: Leader-scan kernel
    let out_key_indices = device.new_buffer(n64.max(1) * 4, MTLResourceOptions::StorageModeShared);
    let out_values = device.new_buffer(
        n64.max(1) * out_dtype.size_in_bytes() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let group_counter = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe { *(group_counter.contents() as *mut u32) = 0; }

    let len_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe { *(len_buf.contents() as *mut u32) = n as u32; }

    let kernel_name = match agg {
        "sum" => format!("string_groupby_sum_direct_{}", val_suffix),
        "min" => format!("string_groupby_min_direct_{}", val_suffix),
        "max" => format!("string_groupby_max_direct_{}", val_suffix),
        "count" => "string_groupby_count_direct".to_string(),
        _ => unreachable!(),
    };

    let pipeline = get_pipeline_state(device, &library, &kernel_name)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);

    if agg == "count" {
        // count kernel: no sorted_values buffer
        enc.set_buffer(0, Some(&sorted_indices), 0);
        enc.set_buffer(1, Some(key_offsets.metal_buffer()), 0);
        enc.set_buffer(2, Some(key_chars.metal_buffer()), 0);
        enc.set_buffer(3, Some(&out_key_indices), 0);
        enc.set_buffer(4, Some(&out_values), 0);
        enc.set_buffer(5, Some(&group_counter), 0);
        enc.set_buffer(6, Some(&len_buf), 0);
    } else {
        enc.set_buffer(0, Some(&sorted_indices), 0);
        enc.set_buffer(1, Some(key_offsets.metal_buffer()), 0);
        enc.set_buffer(2, Some(key_chars.metal_buffer()), 0);
        enc.set_buffer(3, Some(&sorted_values), 0);
        enc.set_buffer(4, Some(&out_key_indices), 0);
        enc.set_buffer(5, Some(&out_values), 0);
        enc.set_buffer(6, Some(&group_counter), 0);
        enc.set_buffer(7, Some(&len_buf), 0);
    }

    let num_tg = (n64 + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    enc.dispatch_thread_groups(
        MTLSize::new(num_tg, 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "String groupby (sort) failed: Metal command buffer error"
        ));
    }

    let num_groups = unsafe { *(group_counter.contents() as *const u32) as usize };

    if num_groups == 0 {
        return empty_string_groupby(device, out_dtype);
    }

    let key_series = gather_key_strings(
        device, queue, &library, key_offsets, key_chars, &out_key_indices, num_groups,
    )?;

    let value_series = MetalSeries::from_numeric(
        SharedBuffer::from_metal_buffer(out_values, num_groups, out_dtype)
    );

    Ok((key_series, value_series))
}

/// Sort-based mean: sort once, fused sum+count leader-scan, then divide.
fn metal_string_groupby_mean_sort(
    device: &metal::Device,
    queue: &metal::CommandQueue,
    key_offsets: &SharedBuffer,
    key_chars: &SharedBuffer,
    values_buf: &SharedBuffer,
    n: usize,
    val_dtype: DType,
) -> PyResult<(MetalSeries, MetalSeries)> {
    let library = load_strings_library(device)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let n64 = n as u64;
    let val_suffix = val_dtype.kernel_suffix();

    let sorted_indices = string_bitonic_sort_indices(device, queue, key_offsets, key_chars, n)?;
    let sorted_values = string_gather_values(device, queue, values_buf, &sorted_indices, n, val_dtype)?;

    let out_key_indices = device.new_buffer(n64.max(1) * 4, MTLResourceOptions::StorageModeShared);
    let out_sums = device.new_buffer(
        n64.max(1) * val_dtype.size_in_bytes() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let out_counts = device.new_buffer(n64.max(1) * 4, MTLResourceOptions::StorageModeShared);
    let group_counter = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe { *(group_counter.contents() as *mut u32) = 0; }

    let len_buf = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe { *(len_buf.contents() as *mut u32) = n as u32; }

    let pipeline = get_pipeline_state(
        device, &library, &format!("string_groupby_sum_count_direct_{}", val_suffix),
    ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&sorted_indices), 0);
    enc.set_buffer(1, Some(key_offsets.metal_buffer()), 0);
    enc.set_buffer(2, Some(key_chars.metal_buffer()), 0);
    enc.set_buffer(3, Some(&sorted_values), 0);
    enc.set_buffer(4, Some(&out_key_indices), 0);
    enc.set_buffer(5, Some(&out_sums), 0);
    enc.set_buffer(6, Some(&out_counts), 0);
    enc.set_buffer(7, Some(&group_counter), 0);
    enc.set_buffer(8, Some(&len_buf), 0);
    let num_tg = (n64 + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    enc.dispatch_thread_groups(
        MTLSize::new(num_tg, 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "String groupby mean (sort) failed: Metal command buffer error"
        ));
    }

    let num_groups = unsafe { *(group_counter.contents() as *const u32) as usize };

    if num_groups == 0 {
        return empty_string_groupby(device, DType::Float32);
    }

    // Sort path: count is stored as float
    let (mean_buf, mean_dtype) = string_compute_mean(
        device, &out_sums, &out_counts, num_groups, val_dtype, true,
    );

    let key_series = gather_key_strings(
        device, queue, &library, key_offsets, key_chars, &out_key_indices, num_groups,
    )?;

    let value_series = MetalSeries::from_numeric(
        SharedBuffer::from_metal_buffer(mean_buf, num_groups, mean_dtype)
    );

    Ok((key_series, value_series))
}

/// Divide per-group sum by per-group count, returning (buffer, dtype).
/// `count_is_float`: true for sort path (counts stored as f32),
///                    false for hash path (counts stored as u32).
fn string_compute_mean(
    device: &metal::Device,
    sum_buf: &metal::Buffer,
    count_buf: &metal::Buffer,
    num_groups: usize,
    val_dtype: DType,
    count_is_float: bool,
) -> (metal::Buffer, DType) {
    unsafe {
        let read_count = |g: usize| -> f64 {
            if count_is_float {
                *(count_buf.contents() as *const f32).add(g) as f64
            } else {
                *(count_buf.contents() as *const u32).add(g) as f64
            }
        };
        match val_dtype {
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
            _ => unreachable!("string groupby mean: unsupported dtype"),
        }
    }
}

// ============================================================================
// Public entry point
// ============================================================================

#[pyfunction]
pub fn metal_string_groupby(
    keys: &MetalSeries,
    values: &MetalSeries,
    agg: &str,
) -> PyResult<(MetalSeries, MetalSeries)> {
    let (key_offsets, key_chars) = keys.as_str_checked()?;
    let values_buf = values.as_numeric_checked()?;

    let val_dtype = values_buf.dtype;
    match val_dtype {
        DType::Float32 | DType::Int32 => {}
        _ => return Err(pyo3::exceptions::PyTypeError::new_err(
            format!("String groupby only supports float32/int32 values, got {:?}", val_dtype)
        )),
    }

    match agg {
        "sum" | "count" | "min" | "max" | "mean" => {}
        _ => return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Unsupported aggregation '{}', expected sum/count/min/max/mean", agg)
        )),
    }

    let n = keys.len;
    if n != values_buf.len {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "keys and values must have same length"
        ));
    }

    let (device, queue) = MetalBackend::device_and_queue()?;

    if n == 0 {
        let out_dtype = match agg {
            "count" => DType::Float32,
            "mean" => if val_dtype == DType::Int32 { DType::Float64 } else { DType::Float32 },
            _ => val_dtype,
        };
        return empty_string_groupby(device, out_dtype);
    }

    if n <= STRING_HASH_MAX_LEN {
        if agg == "mean" {
            metal_string_groupby_mean_hash(device, queue, key_offsets, key_chars, values_buf, n, val_dtype)
        } else {
            metal_string_groupby_hash(device, queue, key_offsets, key_chars, values_buf, n, val_dtype, agg)
        }
    } else {
        if agg == "mean" {
            metal_string_groupby_mean_sort(device, queue, key_offsets, key_chars, values_buf, n, val_dtype)
        } else {
            metal_string_groupby_sort(device, queue, key_offsets, key_chars, values_buf, n, val_dtype, agg)
        }
    }
}
