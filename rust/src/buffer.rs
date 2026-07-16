// SharedBuffer — zero-copy buffer shared between numpy and Metal GPU.
//
// Uses MTLStorageModeShared so CPU and GPU access the same physical memory
// on Apple Silicon unified memory architecture.

use std::sync::Arc;

use metal::{Buffer, MTLResourceOptions};
use numpy::{ndarray::ArrayView1, PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use crate::backend::MetalBackend;

/// Simple dtype enum for buffer type tracking.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DType {
    Float32,
    Float64,
    Int32,
    Int64,
    Int8,
    Int16,
    Uint8,
    Uint16,
    /// Boolean data column (comparison results, logical ops). Stored as one
    /// `uint8_t` per element (0 = false, 1 = true) — NOT packed bits. This is
    /// distinct from `NullMask`, which IS a packed validity bitmask; `Bool`
    /// is an ordinary data dtype that happens to share `Uint8`'s storage
    /// width.
    Bool,
    /// Unsigned 32-bit data column (one `uint32_t` per element). Used
    /// alongside `Int32` by the prefix-sum/scan kernel (Task 3.1) — the two
    /// share identical two's-complement addition semantics, but a distinct
    /// `Uint32` dtype keeps unsigned-count buffers (e.g. future boolean-mask
    /// scan output consumed by Phase 4 filtering) self-describing.
    Uint32,
    Uint64,
    /// Datetime column (nanoseconds since 1970-01-01 UTC). Stored as int64.
    /// Uses `kernel_suffix = "int64"` so all int64 kernels work automatically.
    Datetime,
    /// Timedelta column (duration in nanoseconds). Stored as int64.
    /// Uses `kernel_suffix = "int64"` so all int64 kernels work automatically.
    Timedelta,
    Utf8,
}

impl DType {
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::Float64 => 8,
            DType::Int8 => 1,
            DType::Int16 => 2,
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::Uint8 => 1,
            DType::Uint16 => 2,
            DType::Bool => 1,
            DType::Uint32 => 4,
            DType::Uint64 => 8,
            DType::Datetime => 8,
            DType::Timedelta => 8,
            DType::Utf8 => panic!("Utf8 is a series-level dtype, not a buffer-level dtype"),
        }
    }

    pub fn kernel_suffix(&self) -> &'static str {
        match self {
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Int8 => "int8",
            DType::Int16 => "int16",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::Uint8 => "uint8",
            DType::Uint16 => "uint16",
            DType::Bool => "bool",
            DType::Uint32 => "uint32",
            DType::Uint64 => "uint64",
            DType::Datetime => "int64",
            DType::Timedelta => "int64",
            DType::Utf8 => panic!("Utf8 has no kernel suffix"),
        }
    }

    pub fn radix_passes(&self) -> u32 {
        match self {
            DType::Int8 | DType::Uint8 | DType::Bool => 1,
            DType::Int16 | DType::Uint16 => 2,
            DType::Float32 | DType::Int32 => 4,
            DType::Uint32 => 4,
            DType::Float64 | DType::Int64 => 8,
            DType::Uint64 => 8,
            DType::Datetime | DType::Timedelta => 8,
            DType::Utf8 => panic!("radix_passes not supported for {:?}", self),
        }
    }

    /// Fill `buf[start..end]` with the maximum value for this dtype.
    /// Used to pad sort buffers so padding sorts last.
    pub unsafe fn fill_max(&self, ptr: *mut u8, start: usize, end: usize) {
        match self {
            DType::Float32 => {
                let p = ptr as *mut f32;
                for i in start..end { *p.add(i) = f32::INFINITY; }
            }
            DType::Float64 => {
                let p = ptr as *mut f64;
                for i in start..end { *p.add(i) = f64::INFINITY; }
            }
            DType::Int8 => {
                let p = ptr as *mut i8;
                for i in start..end { *p.add(i) = i8::MAX; }
            }
            DType::Int16 => {
                let p = ptr as *mut i16;
                for i in start..end { *p.add(i) = i16::MAX; }
            }
            DType::Int32 => {
                let p = ptr as *mut i32;
                for i in start..end { *p.add(i) = i32::MAX; }
            }
            DType::Int64 => {
                let p = ptr as *mut i64;
                for i in start..end { *p.add(i) = i64::MAX; }
            }
            DType::Uint8 | DType::Bool => {
                let p = ptr as *mut u8;
                for i in start..end { *p.add(i) = u8::MAX; }
            }
            DType::Uint16 => {
                let p = ptr as *mut u16;
                for i in start..end { *p.add(i) = u16::MAX; }
            }
            DType::Uint32 => {
                let p = ptr as *mut u32;
                for i in start..end { *p.add(i) = u32::MAX; }
            }
            DType::Uint64 => {
                let p = ptr as *mut u64;
                for i in start..end { *p.add(i) = u64::MAX; }
            }
            DType::Datetime | DType::Timedelta => {
                let p = ptr as *mut i64;
                for i in start..end { *p.add(i) = i64::MAX; }
            }
            DType::Utf8 => panic!("fill_max not supported for {:?}", self),
        }
    }
}

/// SharedBuffer wraps an MTLBuffer that shares memory with numpy.
///
/// No longer a `#[pyclass]` — Python only ever sees `MetalSeries` (see
/// `crate::series`), which wraps this internal type.
#[derive(Clone)]
pub struct SharedBuffer {
    buffer: Arc<Buffer>,
    pub len: usize,
    pub dtype: DType,
}

fn from_numpy_inner(ptr: *const u8, byte_len: usize, len: usize, dtype: DType) -> PyResult<SharedBuffer> {
    let device = MetalBackend::device()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal device available"))?;
    if byte_len == 0 {
        let buffer = device.new_buffer(1, MTLResourceOptions::StorageModeShared);
        return Ok(SharedBuffer { buffer: Arc::new(buffer), len: 0, dtype });
    }
    let buffer = device.new_buffer_with_bytes_no_copy(
        ptr as *const _, byte_len as u64, MTLResourceOptions::StorageModeShared, None,
    );
    Ok(SharedBuffer { buffer: Arc::new(buffer), len, dtype })
}

// `SharedBuffer` is no longer a `#[pyclass]` (see module docs above), so we
// can't use `Py::new` to build the numpy-array "owner" object anymore. A
// `PyCapsule` wraps an arbitrary `'static + Send` Rust value as a plain
// Python object instead, which is all `borrow_from_array_bound` needs: it
// just keeps `container` (and therefore the cloned `Arc<Buffer>` inside it)
// alive for as long as the returned numpy array is alive.
macro_rules! impl_to_numpy_arm {
    ($self:expr, $py:expr, $rust_type:ty) => {{
        let view = unsafe {
            ArrayView1::from_shape_ptr($self.len, $self.buffer.contents() as *const $rust_type)
        };
        let container = pyo3::types::PyCapsule::new_bound($py, $self.clone(), None)?.into_any();
        unsafe {
            Ok(PyArray1::borrow_from_array_bound(&view, container).into_any())
        }
    }};
}

// Internal Rust-only methods (not exposed to Python directly — MetalSeries
// is the pyclass and forwards into these).
impl SharedBuffer {
    pub fn from_numpy(data: &Bound<PyArray1<f32>>) -> PyResult<SharedBuffer> {
        let r = data.readonly();
        let s = r.as_slice().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("numpy: {e}")))?;
        from_numpy_inner(s.as_ptr() as *const u8, s.len() * 4, s.len(), DType::Float32)
    }

    pub fn from_numpy_f64(data: &Bound<PyArray1<f64>>) -> PyResult<SharedBuffer> {
        let r = data.readonly();
        let s = r.as_slice().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("numpy: {e}")))?;
        from_numpy_inner(s.as_ptr() as *const u8, s.len() * 8, s.len(), DType::Float64)
    }

    pub fn from_numpy_i32(data: &Bound<PyArray1<i32>>) -> PyResult<SharedBuffer> {
        let r = data.readonly();
        let s = r.as_slice().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("numpy: {e}")))?;
        from_numpy_inner(s.as_ptr() as *const u8, s.len() * 4, s.len(), DType::Int32)
    }

    pub fn from_numpy_i64(data: &Bound<PyArray1<i64>>) -> PyResult<SharedBuffer> {
        let r = data.readonly();
        let s = r.as_slice().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("numpy: {e}")))?;
        from_numpy_inner(s.as_ptr() as *const u8, s.len() * 8, s.len(), DType::Int64)
    }

    /// Build a `Uint32`-dtype buffer from a numpy `uint32` array (see
    /// `DType::Uint32` docs — used by the prefix-sum/scan kernel).
    pub fn from_numpy_u32(data: &Bound<PyArray1<u32>>) -> PyResult<SharedBuffer> {
        let r = data.readonly();
        let s = r.as_slice().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("numpy: {e}")))?;
        from_numpy_inner(s.as_ptr() as *const u8, s.len() * 4, s.len(), DType::Uint32)
    }

    /// Build a `Bool`-dtype buffer from a numpy `uint8` array (values
    /// expected to be 0 or 1). Bool storage is one byte per element, same
    /// width as `Uint8`, but tracked as its own dtype so kernel dispatch and
    /// `dtype()` reporting distinguish "boolean data" from "raw bytes".
    pub fn from_numpy_bool(data: &Bound<PyArray1<u8>>) -> PyResult<SharedBuffer> {
        let r = data.readonly();
        let s = r.as_slice().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("numpy: {e}")))?;
        from_numpy_inner(s.as_ptr() as *const u8, s.len(), s.len(), DType::Bool)
    }

    pub fn from_numpy_i8(data: &Bound<PyArray1<i8>>) -> PyResult<SharedBuffer> {
        let r = data.readonly();
        let s = r.as_slice().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("numpy: {e}")))?;
        from_numpy_inner(s.as_ptr() as *const u8, s.len(), s.len(), DType::Int8)
    }

    pub fn from_numpy_i16(data: &Bound<PyArray1<i16>>) -> PyResult<SharedBuffer> {
        let r = data.readonly();
        let s = r.as_slice().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("numpy: {e}")))?;
        from_numpy_inner(s.as_ptr() as *const u8, s.len() * 2, s.len(), DType::Int16)
    }

    pub fn from_numpy_u8(data: &Bound<PyArray1<u8>>) -> PyResult<SharedBuffer> {
        let r = data.readonly();
        let s = r.as_slice().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("numpy: {e}")))?;
        from_numpy_inner(s.as_ptr() as *const u8, s.len(), s.len(), DType::Uint8)
    }

    pub fn from_numpy_u16(data: &Bound<PyArray1<u16>>) -> PyResult<SharedBuffer> {
        let r = data.readonly();
        let s = r.as_slice().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("numpy: {e}")))?;
        from_numpy_inner(s.as_ptr() as *const u8, s.len() * 2, s.len(), DType::Uint16)
    }

    pub fn from_numpy_u64(data: &Bound<PyArray1<u64>>) -> PyResult<SharedBuffer> {
        let r = data.readonly();
        let s = r.as_slice().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("numpy: {e}")))?;
        from_numpy_inner(s.as_ptr() as *const u8, s.len() * 8, s.len(), DType::Uint64)
    }

    pub fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.dtype {
            DType::Float32 => impl_to_numpy_arm!(self, py, f32),
            DType::Float64 => impl_to_numpy_arm!(self, py, f64),
            DType::Int8    => impl_to_numpy_arm!(self, py, i8),
            DType::Int16   => impl_to_numpy_arm!(self, py, i16),
            DType::Int32   => impl_to_numpy_arm!(self, py, i32),
            DType::Int64   => impl_to_numpy_arm!(self, py, i64),
            DType::Uint8   => impl_to_numpy_arm!(self, py, u8),
            DType::Uint16  => impl_to_numpy_arm!(self, py, u16),
            DType::Bool    => impl_to_numpy_arm!(self, py, u8),
            DType::Uint32  => impl_to_numpy_arm!(self, py, u32),
            DType::Uint64  => impl_to_numpy_arm!(self, py, u64),
            DType::Datetime  => impl_to_numpy_arm!(self, py, i64),
            DType::Timedelta => impl_to_numpy_arm!(self, py, i64),
            DType::Utf8 => Err(pyo3::exceptions::PyTypeError::new_err(
                format!("to_numpy not supported for {:?}", self.dtype)
            )),
        }
    }

    /// Return the number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return the dtype of the buffer.
    pub fn dtype(&self) -> String {
        format!("{:?}", self.dtype)
    }

    /// Return a reference to the underlying Metal buffer.
    pub fn metal_buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Create from an existing Metal buffer (used internally by kernels).
    pub fn from_metal_buffer(buffer: Buffer, len: usize, dtype: DType) -> Self {
        SharedBuffer {
            buffer: Arc::new(buffer),
            len,
            dtype,
        }
    }

    /// Create a SharedBuffer directly from raw bytes (used for string offset/
    /// char buffers, which don't come from a Python numpy array).
    pub fn from_raw_bytes(device: &metal::Device, data: &[u8]) -> Self {
        let buffer = device.new_buffer_with_data(
            data.as_ptr() as *const _,
            data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        SharedBuffer { buffer: Arc::new(buffer), len: data.len(), dtype: DType::Uint8 }
    }

    /// Consume self and return the underlying Metal buffer.
    /// Used when transferring ownership into a `MetalColumn`.
    pub fn into_metal_buffer(self) -> Buffer {
        Arc::try_unwrap(self.buffer).unwrap_or_else(|arc| (*arc).clone())
    }
}

/// NullMask wraps a packed validity bitmask buffer (1 bit per element; bit
/// set = valid, bit clear = null) stored in `MTLStorageModeShared` memory so
/// both the CPU and GPU can read/write it directly. See
/// `rust/metal/common/04_null_mask.h` for the matching MSL-side helpers
/// (`is_valid`/`set_valid`/`set_invalid`) that kernels use against the same
/// bit layout.
pub struct NullMask {
    buffer: Arc<Buffer>,
    len: usize,
}

impl NullMask {
    /// Allocate a mask with every bit set (all `len` elements valid).
    pub fn new_all_valid(device: &metal::Device, len: usize) -> Self {
        let byte_len = (len + 7) / 8;
        let buf = device.new_buffer(
            byte_len.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            std::ptr::write_bytes(buf.contents() as *mut u8, 0xFF, byte_len);
        }
        NullMask { buffer: Arc::new(buf), len }
    }

    /// Scan a float32 slice for NaNs, building a validity bitmask (NaN =
    /// null) and a cleaned copy of the data with NaNs replaced by `0.0`
    /// (GPU kernels never need to special-case NaN payloads — they just
    /// consult the mask). Returns `None` for the mask when no NaNs were
    /// found, since an all-valid column doesn't need one tracked at all.
    pub fn from_numpy_nans(device: &metal::Device, arr: &[f32]) -> (Vec<f32>, Option<NullMask>) {
        let len = arr.len();
        let byte_len = (len + 7) / 8;
        let mut mask_bytes = vec![0xFFu8; byte_len.max(1)];
        let mut cleaned = Vec::with_capacity(len);
        let mut has_null = false;

        for (i, &v) in arr.iter().enumerate() {
            if v.is_nan() {
                has_null = true;
                mask_bytes[i / 8] &= !(1u8 << (i % 8));
                cleaned.push(0.0);
            } else {
                cleaned.push(v);
            }
        }

        if !has_null {
            return (cleaned, None);
        }

        let buf = device.new_buffer_with_data(
            mask_bytes.as_ptr() as *const _,
            byte_len.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        (cleaned, Some(NullMask { buffer: Arc::new(buf), len }))
    }

    /// Create from an existing Metal buffer (used internally by kernels).
    pub fn from_metal_buffer(buffer: Buffer, len: usize) -> Self {
        NullMask { buffer: Arc::new(buffer), len }
    }

    pub fn metal_buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// A cloned `Arc` handle to the same underlying Metal buffer, for
    /// storing alongside a `MetalColumn`'s `null_mask: Option<Arc<Buffer>>`.
    pub fn buffer_arc(&self) -> Arc<Buffer> {
        self.buffer.clone()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_valid(&self, idx: usize) -> bool {
        let ptr = self.buffer.contents() as *const u8;
        unsafe { (*ptr.add(idx / 8) & (1u8 << (idx % 8))) != 0 }
    }

    pub fn count_valid(&self) -> usize {
        let ptr = self.buffer.contents() as *const u8;
        let byte_len = (self.len + 7) / 8;
        let bytes = unsafe { std::slice::from_raw_parts(ptr, byte_len) };
        let remainder = self.len % 8;
        if remainder == 0 || byte_len == 0 {
            bytes.iter().map(|b| b.count_ones()).sum::<u32>() as usize
        } else {
            let full: u32 = bytes[..byte_len - 1].iter().map(|b| b.count_ones()).sum();
            let last_mask = (1u8 << remainder) - 1;
            let last = (bytes[byte_len - 1] & last_mask).count_ones();
            (full + last) as usize
        }
    }
}

impl Clone for NullMask {
    fn clone(&self) -> Self {
        NullMask { buffer: self.buffer.clone(), len: self.len }
    }
}
