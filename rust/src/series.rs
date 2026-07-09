// MetalSeries — the public pyclass wrapping SharedBuffer (numeric dtypes)
// and, in a later task, string data. Python only ever touches MetalSeries;
// SharedBuffer is an internal implementation detail.

use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::PyArray1;
use metal::MTLResourceOptions;

use crate::backend::MetalBackend;
use crate::buffer::{SharedBuffer, DType};

pub(crate) enum SeriesData {
    Numeric(SharedBuffer),
    Str {
        offsets: SharedBuffer,
        chars: SharedBuffer,
    },
}

#[pyclass]
pub struct MetalSeries {
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
        Ok(MetalSeries { data: SeriesData::Numeric(buf), len, dtype: DType::Float32 })
    }

    #[staticmethod]
    pub fn from_numpy_f64(data: &Bound<PyArray1<f64>>) -> PyResult<MetalSeries> {
        let buf = SharedBuffer::from_numpy_f64(data)?;
        let len = buf.len;
        Ok(MetalSeries { data: SeriesData::Numeric(buf), len, dtype: DType::Float64 })
    }

    #[staticmethod]
    pub fn from_numpy_i32(data: &Bound<PyArray1<i32>>) -> PyResult<MetalSeries> {
        let buf = SharedBuffer::from_numpy_i32(data)?;
        let len = buf.len;
        Ok(MetalSeries { data: SeriesData::Numeric(buf), len, dtype: DType::Int32 })
    }

    #[staticmethod]
    pub fn from_numpy_i64(data: &Bound<PyArray1<i64>>) -> PyResult<MetalSeries> {
        let buf = SharedBuffer::from_numpy_i64(data)?;
        let len = buf.len;
        Ok(MetalSeries { data: SeriesData::Numeric(buf), len, dtype: DType::Int64 })
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

        Ok(MetalSeries {
            data: SeriesData::Str { offsets, chars },
            len: n,
            dtype: DType::Utf8,
        })
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
        MetalSeries { data: SeriesData::Numeric(buf), len, dtype }
    }
}
