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
    Uint8,
    Utf8,
}

impl DType {
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::Float32 => 4,
            DType::Float64 => 8,
            DType::Int32 => 4,
            DType::Int64 => 8,
            DType::Uint8 => 1,
            DType::Utf8 => panic!("Utf8 is a series-level dtype, not a buffer-level dtype"),
        }
    }

    pub fn kernel_suffix(&self) -> &'static str {
        match self {
            DType::Float32 => "float32",
            DType::Float64 => "float64",
            DType::Int32 => "int32",
            DType::Int64 => "int64",
            DType::Uint8 => "uint8",
            DType::Utf8 => panic!("Utf8 has no kernel suffix"),
        }
    }

    pub fn radix_passes(&self) -> u32 {
        match self {
            DType::Float32 | DType::Int32 => 4,
            DType::Float64 | DType::Int64 => 8,
            _ => panic!("radix_passes not supported for {:?}", self),
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
            DType::Int32 => {
                let p = ptr as *mut i32;
                for i in start..end { *p.add(i) = i32::MAX; }
            }
            DType::Int64 => {
                let p = ptr as *mut i64;
                for i in start..end { *p.add(i) = i64::MAX; }
            }
            _ => panic!("fill_max not supported for {:?}", self),
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

    pub fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match self.dtype {
            DType::Float32 => impl_to_numpy_arm!(self, py, f32),
            DType::Float64 => impl_to_numpy_arm!(self, py, f64),
            DType::Int32   => impl_to_numpy_arm!(self, py, i32),
            DType::Int64   => impl_to_numpy_arm!(self, py, i64),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
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
}
