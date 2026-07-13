// MetalSeries — the public pyclass wrapping SharedBuffer (numeric dtypes)
// and string data. Python only ever touches MetalSeries; SharedBuffer is an
// internal implementation detail.
//
// Internally, MetalSeries now also owns a canonical `MetalColumn` (GPU-
// resident storage with data/null_mask/children — see `crate::column`),
// exposed via `column()` and `view()` for the Phase 1 storage-layer
// refactor. The pre-existing `SeriesData` representation (a `SharedBuffer`,
// or an offsets/chars pair of `SharedBuffer`s) is kept alongside it purely
// for backward compatibility: `as_numeric()`/`as_numeric_checked()`/
// `as_str()`/`as_str_checked()` must keep returning `&SharedBuffer` (by
// reference, tied to `&self`) because that's exactly what
// reductions.rs/sort.rs/groupby.rs/strings.rs already depend on throughout
// (`.metal_buffer()`/`.len`/`.dtype` access, and dozens of helper functions
// typed to take `&SharedBuffer` parameters) — changing that return type
// would ripple through all of those call sites. The `SharedBuffer`(s) in
// `data` share the exact same underlying Metal buffer as `column`'s
// data/children (a cheap `metal::Buffer` retain via `.clone()`, not a new
// GPU allocation), so there's no data duplication on the GPU — just two
// owning Rust-side handles to the same bytes.

use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::PyArray1;
use metal::MTLResourceOptions;
use std::sync::Arc;

use crate::backend::MetalBackend;
use crate::buffer::{SharedBuffer, DType};
use crate::column::MetalColumn;
use crate::column_view::MetalColumnView;

pub(crate) enum SeriesData {
    Numeric(SharedBuffer),
    Str {
        offsets: SharedBuffer,
        chars: SharedBuffer,
    },
}

#[pyclass]
pub struct MetalSeries {
    pub(crate) column: MetalColumn,
    pub(crate) data: SeriesData,
    pub(crate) len: usize,
    pub(crate) dtype: DType,
}

#[pymethods]
impl MetalSeries {
    #[staticmethod]
    pub fn from_numpy(data: &Bound<PyArray1<f32>>) -> PyResult<MetalSeries> {
        let buf = SharedBuffer::from_numpy(data)?;
        let len = buf.len;
        let column = MetalColumn::from_buffer(buf.metal_buffer().clone(), len, DType::Float32);
        Ok(MetalSeries { column, data: SeriesData::Numeric(buf), len, dtype: DType::Float32 })
    }

    #[staticmethod]
    pub fn from_numpy_f64(data: &Bound<PyArray1<f64>>) -> PyResult<MetalSeries> {
        let buf = SharedBuffer::from_numpy_f64(data)?;
        let len = buf.len;
        let column = MetalColumn::from_buffer(buf.metal_buffer().clone(), len, DType::Float64);
        Ok(MetalSeries { column, data: SeriesData::Numeric(buf), len, dtype: DType::Float64 })
    }

    #[staticmethod]
    pub fn from_numpy_i32(data: &Bound<PyArray1<i32>>) -> PyResult<MetalSeries> {
        let buf = SharedBuffer::from_numpy_i32(data)?;
        let len = buf.len;
        let column = MetalColumn::from_buffer(buf.metal_buffer().clone(), len, DType::Int32);
        Ok(MetalSeries { column, data: SeriesData::Numeric(buf), len, dtype: DType::Int32 })
    }

    #[staticmethod]
    pub fn from_numpy_i64(data: &Bound<PyArray1<i64>>) -> PyResult<MetalSeries> {
        let buf = SharedBuffer::from_numpy_i64(data)?;
        let len = buf.len;
        let column = MetalColumn::from_buffer(buf.metal_buffer().clone(), len, DType::Int64);
        Ok(MetalSeries { column, data: SeriesData::Numeric(buf), len, dtype: DType::Int64 })
    }

    pub fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.as_numeric_checked()?.to_numpy(py)
    }

    #[getter]
    pub fn len(&self) -> usize {
        self.len
    }

    #[getter]
    pub fn dtype(&self) -> String {
        format!("{:?}", self.dtype)
    }

    #[staticmethod]
    pub fn from_strings(data: &Bound<PyList>) -> PyResult<MetalSeries> {
        let device = MetalBackend::device()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal device"))?;

        let strings: Vec<String> = data.extract()?;
        let n = strings.len();

        let mut offsets_vec: Vec<i64> = Vec::with_capacity(n + 1);
        let mut chars_vec: Vec<u8> = Vec::new();

        let mut offset: i64 = 0;
        for s in &strings {
            offsets_vec.push(offset);
            chars_vec.extend_from_slice(s.as_bytes());
            offset += s.len() as i64;
        }
        offsets_vec.push(offset);

        let offsets_byte_len = offsets_vec.len() * std::mem::size_of::<i64>();
        let offsets_buf = device.new_buffer(
            offsets_byte_len.max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                offsets_vec.as_ptr() as *const u8,
                offsets_buf.contents() as *mut u8,
                offsets_byte_len,
            );
        }
        let offsets = SharedBuffer::from_metal_buffer(offsets_buf, offsets_vec.len(), DType::Int64);

        let chars_buf = device.new_buffer(
            chars_vec.len().max(1) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        if !chars_vec.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    chars_vec.as_ptr(),
                    chars_buf.contents() as *mut u8,
                    chars_vec.len(),
                );
            }
        }
        let chars = SharedBuffer::from_metal_buffer(chars_buf, chars_vec.len(), DType::Uint8);

        Ok(MetalSeries::from_str_parts(device, offsets, chars, n))
    }

    pub fn to_strings<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let (offsets, chars) = match &self.data {
            SeriesData::Str { offsets, chars } => (offsets, chars),
            SeriesData::Numeric(_) => return Err(pyo3::exceptions::PyTypeError::new_err(
                "Cannot convert numeric series to strings"
            )),
        };

        let offsets_ptr = offsets.metal_buffer().contents() as *const i64;
        let chars_ptr = chars.metal_buffer().contents() as *const u8;

        let mut result: Vec<String> = Vec::with_capacity(self.len);
        for i in 0..self.len {
            let start = unsafe { *offsets_ptr.add(i) } as usize;
            let end = unsafe { *offsets_ptr.add(i + 1) } as usize;
            let bytes = unsafe { std::slice::from_raw_parts(chars_ptr.add(start), end - start) };
            let s = String::from_utf8(bytes.to_vec()).map_err(|e|
                pyo3::exceptions::PyValueError::new_err(format!("Invalid UTF-8: {e}"))
            )?;
            result.push(s);
        }

        Ok(PyList::new_bound(py, &result))
    }
}

impl MetalSeries {
    /// Reference to the canonical, GPU-resident `MetalColumn` backing this
    /// series (Phase 1 storage-layer refactor).
    pub fn column(&self) -> &MetalColumn {
        &self.column
    }

    /// A lightweight, non-owning view over this series' column data, for
    /// zero-copy kernel dispatch.
    pub fn view(&self) -> MetalColumnView<'_> {
        self.column.view()
    }

    /// Build a `MetalSeries` from a `MetalColumn` (numeric or Utf8).
    pub fn from_column(col: MetalColumn) -> Self {
        let len = col.size();
        let dtype = col.dtype();
        let data = if dtype == DType::Utf8 {
            let off_col = col.child(0);
            let char_col = col.child(1);
            SeriesData::Str {
                offsets: SharedBuffer::from_metal_buffer(off_col.data().clone(), off_col.size(), off_col.dtype()),
                chars: SharedBuffer::from_metal_buffer(char_col.data().clone(), char_col.size(), char_col.dtype()),
            }
        } else {
            SeriesData::Numeric(SharedBuffer::from_metal_buffer(col.data().clone(), len, dtype))
        };
        MetalSeries { column: col, data, len, dtype }
    }

    /// Build a string series from offsets/chars `SharedBuffer`s, deriving a
    /// canonical Utf8 `MetalColumn` (a 1-byte dummy data buffer — Utf8's
    /// real payload lives entirely in its children — plus `[offsets_col,
    /// chars_col]` children) alongside the back-compat `SeriesData::Str`
    /// used by `as_str()`/`as_str_checked()`.
    ///
    /// `len` is the number of strings (NOT the offsets buffer's element
    /// count, which is `len + 1`).
    pub(crate) fn from_str_parts(
        device: &metal::Device,
        offsets: SharedBuffer,
        chars: SharedBuffer,
        len: usize,
    ) -> Self {
        let off_col = MetalColumn::from_buffer(offsets.metal_buffer().clone(), offsets.len, DType::Int64);
        let char_col = MetalColumn::from_buffer(chars.metal_buffer().clone(), chars.len, DType::Uint8);
        let dummy = device.new_buffer(1, MTLResourceOptions::StorageModeShared);
        let column = MetalColumn::new(Arc::new(dummy), None, DType::Utf8, len, 0, vec![off_col, char_col]);
        MetalSeries { column, data: SeriesData::Str { offsets, chars }, len, dtype: DType::Utf8 }
    }

    pub fn as_numeric(&self) -> &SharedBuffer {
        match &self.data {
            SeriesData::Numeric(buf) => buf,
            SeriesData::Str { .. } => panic!("Expected numeric series, got string"),
        }
    }

    pub fn as_numeric_checked(&self) -> PyResult<&SharedBuffer> {
        match &self.data {
            SeriesData::Numeric(buf) => Ok(buf),
            SeriesData::Str { .. } => Err(pyo3::exceptions::PyTypeError::new_err(
                "Cannot convert string series to numpy"
            )),
        }
    }

    pub fn as_str(&self) -> (&SharedBuffer, &SharedBuffer) {
        match &self.data {
            SeriesData::Str { offsets, chars } => (offsets, chars),
            SeriesData::Numeric(_) => panic!("Expected string series, got numeric"),
        }
    }

    pub fn as_str_checked(&self) -> PyResult<(&SharedBuffer, &SharedBuffer)> {
        match &self.data {
            SeriesData::Str { offsets, chars } => Ok((offsets, chars)),
            SeriesData::Numeric(_) => Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected string series, got numeric"
            )),
        }
    }

    pub fn from_numeric(buf: SharedBuffer) -> Self {
        let len = buf.len;
        let dtype = buf.dtype;
        let column = MetalColumn::from_buffer(buf.metal_buffer().clone(), len, dtype);
        MetalSeries { column, data: SeriesData::Numeric(buf), len, dtype }
    }
}
