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
use numpy::{PyArray1, PyArrayMethods};
use metal::MTLResourceOptions;
use std::sync::Arc;

use crate::backend::MetalBackend;
use crate::buffer::{SharedBuffer, DType, NullMask};
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
    pub(crate) null_mask: Option<NullMask>,
}

#[pymethods]
impl MetalSeries {
    #[staticmethod]
    pub fn from_numpy(data: &Bound<PyArray1<f32>>) -> PyResult<MetalSeries> {
        let buf = SharedBuffer::from_numpy(data)?;
        let len = buf.len;
        let column = MetalColumn::from_buffer(buf.metal_buffer().clone(), len, DType::Float32);
        Ok(MetalSeries { column, data: SeriesData::Numeric(buf), len, dtype: DType::Float32, null_mask: None })
    }

    #[staticmethod]
    pub fn from_numpy_f64(data: &Bound<PyArray1<f64>>) -> PyResult<MetalSeries> {
        let buf = SharedBuffer::from_numpy_f64(data)?;
        let len = buf.len;
        let column = MetalColumn::from_buffer(buf.metal_buffer().clone(), len, DType::Float64);
        Ok(MetalSeries { column, data: SeriesData::Numeric(buf), len, dtype: DType::Float64, null_mask: None })
    }

    #[staticmethod]
    pub fn from_numpy_i32(data: &Bound<PyArray1<i32>>) -> PyResult<MetalSeries> {
        let buf = SharedBuffer::from_numpy_i32(data)?;
        let len = buf.len;
        let column = MetalColumn::from_buffer(buf.metal_buffer().clone(), len, DType::Int32);
        Ok(MetalSeries { column, data: SeriesData::Numeric(buf), len, dtype: DType::Int32, null_mask: None })
    }

    #[staticmethod]
    pub fn from_numpy_i64(data: &Bound<PyArray1<i64>>) -> PyResult<MetalSeries> {
        let buf = SharedBuffer::from_numpy_i64(data)?;
        let len = buf.len;
        let column = MetalColumn::from_buffer(buf.metal_buffer().clone(), len, DType::Int64);
        Ok(MetalSeries { column, data: SeriesData::Numeric(buf), len, dtype: DType::Int64, null_mask: None })
    }

    /// Build a `Datetime`-dtype `MetalSeries` from a numpy `int64` array (the
    /// caller is expected to have already viewed a `datetime64[ns]` array as
    /// `int64` — see `DType::Datetime` docs). Storage/kernels are identical
    /// to plain `Int64`; only the dtype tag differs.
    #[staticmethod]
    pub fn from_numpy_datetime(data: &Bound<PyArray1<i64>>) -> PyResult<MetalSeries> {
        let buf = SharedBuffer::from_numpy_i64(data)?;
        let len = buf.len;
        let column = MetalColumn::from_buffer(buf.metal_buffer().clone(), len, DType::Datetime);
        Ok(MetalSeries { column, data: SeriesData::Numeric(buf), len, dtype: DType::Datetime, null_mask: None })
    }

    /// Build a `Timedelta`-dtype `MetalSeries` from a numpy `int64` array
    /// (caller has viewed a `timedelta64[ns]` array as `int64`). See
    /// `DType::Timedelta` docs.
    #[staticmethod]
    pub fn from_numpy_timedelta(data: &Bound<PyArray1<i64>>) -> PyResult<MetalSeries> {
        let buf = SharedBuffer::from_numpy_i64(data)?;
        let len = buf.len;
        let column = MetalColumn::from_buffer(buf.metal_buffer().clone(), len, DType::Timedelta);
        Ok(MetalSeries { column, data: SeriesData::Numeric(buf), len, dtype: DType::Timedelta, null_mask: None })
    }

    /// Build a `Uint32`-dtype `MetalSeries` from a numpy `uint32` array. See
    /// `DType::Uint32` docs — used by the prefix-sum/scan kernel (Task 3.1).
    #[staticmethod]
    pub fn from_numpy_u32(data: &Bound<PyArray1<u32>>) -> PyResult<MetalSeries> {
        let buf = SharedBuffer::from_numpy_u32(data)?;
        let len = buf.len;
        let column = MetalColumn::from_buffer(buf.metal_buffer().clone(), len, DType::Uint32);
        Ok(MetalSeries { column, data: SeriesData::Numeric(buf), len, dtype: DType::Uint32, null_mask: None })
    }

    /// Build a `Bool`-dtype `MetalSeries` from a numpy `uint8` array (values
    /// expected to be 0 or 1) — the storage type for comparison results and
    /// logical-op output (see `rust/metal/elementwise/logical.metal`).
    #[staticmethod]
    pub fn from_numpy_bool(data: &Bound<PyArray1<u8>>) -> PyResult<MetalSeries> {
        let buf = SharedBuffer::from_numpy_bool(data)?;
        let len = buf.len;
        let column = MetalColumn::from_buffer(buf.metal_buffer().clone(), len, DType::Bool);
        Ok(MetalSeries { column, data: SeriesData::Numeric(buf), len, dtype: DType::Bool, null_mask: None })
    }

    /// Build a `MetalSeries` from a float32 numpy array, treating NaN values
    /// as nulls: a validity bitmask (`NullMask`) is built marking each NaN
    /// position invalid, and the underlying GPU buffer stores `0.0` in place
    /// of each NaN (kernels must consult the mask, not the payload, to
    /// determine validity). If the array has no NaNs, no mask is allocated
    /// and `.null_mask` reads back as `None`.
    #[staticmethod]
    pub fn from_numpy_with_nulls(data: &Bound<PyArray1<f32>>) -> PyResult<MetalSeries> {
        let device = MetalBackend::device()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal device available"))?;
        let r = data.readonly();
        let s = r.as_slice().map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("numpy: {e}")))?;

        let (cleaned, null_mask) = NullMask::from_numpy_nans(device, s);
        let len = cleaned.len();

        let byte_len = (len * std::mem::size_of::<f32>()) as u64;
        let buf = device.new_buffer(byte_len.max(1), MTLResourceOptions::StorageModeShared);
        if len > 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    cleaned.as_ptr() as *const u8,
                    buf.contents() as *mut u8,
                    byte_len as usize,
                );
            }
        }
        let shared = SharedBuffer::from_metal_buffer(buf, len, DType::Float32);

        let column = MetalColumn::from_buffer(shared.metal_buffer().clone(), len, DType::Float32)
            .with_null_mask(null_mask.as_ref().map(|m| m.buffer_arc()));

        Ok(MetalSeries {
            column,
            data: SeriesData::Numeric(shared),
            len,
            dtype: DType::Float32,
            null_mask,
        })
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

    /// Validity bitmask as a numpy bool array (`True` = valid, `False` =
    /// null), or `None` if this series has no tracked nulls.
    #[getter]
    pub fn null_mask<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        match &self.null_mask {
            None => Ok(py.None()),
            Some(mask) => {
                let mut valid = Vec::with_capacity(self.len);
                for i in 0..self.len {
                    valid.push(mask.is_valid(i));
                }
                let arr = PyArray1::from_vec_bound(py, valid);
                Ok(arr.into_py(py))
            }
        }
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
        // Preserve the column's validity bitmask, if any, so nulls survive
        // the column -> series round-trip (e.g. kernels in tasks 1.2+ that
        // produce a `MetalColumn` with a populated `null_mask`).
        let null_mask = col.null_mask().map(|buf| NullMask::from_metal_buffer(buf.clone(), len));
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
        MetalSeries { column: col, data, len, dtype, null_mask }
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
        MetalSeries { column, data: SeriesData::Str { offsets, chars }, len, dtype: DType::Utf8, null_mask: None }
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
        MetalSeries { column, data: SeriesData::Numeric(buf), len, dtype, null_mask: None }
    }

    /// Build a `MetalSeries` from a numeric buffer plus a `NullMask` computed
    /// by a null-aware kernel (e.g. the `_masked` elementwise kernel variants
    /// in `kernels/elementwise.rs`). Attaches the mask both to the returned
    /// series' `.null_mask` (so the Python `null_mask` getter surfaces it)
    /// and to the canonical `MetalColumn`, mirroring `from_numpy_with_nulls`.
    pub fn from_numeric_with_mask(buf: SharedBuffer, mask: NullMask) -> Self {
        let len = buf.len;
        let dtype = buf.dtype;
        let column = MetalColumn::from_buffer(buf.metal_buffer().clone(), len, dtype)
            .with_null_mask(Some(mask.buffer_arc()));
        MetalSeries { column, data: SeriesData::Numeric(buf), len, dtype, null_mask: Some(mask) }
    }
}
