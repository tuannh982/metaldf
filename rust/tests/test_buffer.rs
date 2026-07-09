use numpy::{PyArray1, PyArrayMethods};
use pyo3::types::PyAnyMethods;
use pyo3::{Bound, Python};

use metaldf_engine::series::MetalSeries;

#[test]
fn test_metal_series_from_numpy_roundtrip() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let arr = PyArray1::<f32>::from_vec_bound(py, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let series = MetalSeries::from_numpy(&arr).expect("Failed to create MetalSeries");
        assert_eq!(series.len(), 5);

        let out = series.to_numpy(py).expect("Failed to convert back to numpy");
        let out_arr: &Bound<PyArray1<f32>> = out.downcast().unwrap();
        let out_slice: &[f32] = unsafe { out_arr.as_slice().unwrap() };
        assert_eq!(out_slice, &[1.0, 2.0, 3.0, 4.0, 5.0]);
    });
}

#[test]
fn test_metal_series_zero_copy() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let data = vec![1.0_f32, 2.0, 3.0];
        let arr = PyArray1::<f32>::from_vec_bound(py, data.clone());
        let ptr_before = arr.data();

        let series = MetalSeries::from_numpy(&arr).expect("Failed to create MetalSeries");

        let out = series.to_numpy(py).expect("Failed to convert back to numpy");
        let out_arr: &Bound<PyArray1<f32>> = out.downcast().unwrap();
        let ptr_after = out_arr.data();

        // With StorageModeShared, the pointer should be the same
        assert_eq!(ptr_before, ptr_after, "Series should be zero-copy");
    });
}

#[test]
fn test_metal_series_i32_roundtrip() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let data = vec![1i32, -2, 3, -4, 5];
        let arr = PyArray1::<i32>::from_vec_bound(py, data.clone());
        let series = MetalSeries::from_numpy_i32(&arr).unwrap();
        assert_eq!(series.len(), 5);
        assert_eq!(series.dtype(), "Int32");
        let result = series.to_numpy(py).unwrap();
        let result_arr: &Bound<PyArray1<i32>> = result.downcast().unwrap();
        let result_slice = unsafe { result_arr.as_slice().unwrap() };
        assert_eq!(result_slice, &[1, -2, 3, -4, 5]);
    });
}

#[test]
fn test_metal_series_i64_roundtrip() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let data = vec![1i64, -2, 3, -4, 5];
        let arr = PyArray1::<i64>::from_vec_bound(py, data.clone());
        let series = MetalSeries::from_numpy_i64(&arr).unwrap();
        assert_eq!(series.len(), 5);
        assert_eq!(series.dtype(), "Int64");
        let result = series.to_numpy(py).unwrap();
        let result_arr: &Bound<PyArray1<i64>> = result.downcast().unwrap();
        let result_slice = unsafe { result_arr.as_slice().unwrap() };
        assert_eq!(result_slice, &[1, -2, 3, -4, 5]);
    });
}

#[test]
fn test_metal_series_f64_roundtrip() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let data = vec![1.5f64, -2.5, 3.5];
        let arr = PyArray1::<f64>::from_vec_bound(py, data.clone());
        let series = MetalSeries::from_numpy_f64(&arr).unwrap();
        assert_eq!(series.len(), 3);
        assert_eq!(series.dtype(), "Float64");
        let result = series.to_numpy(py).unwrap();
        let result_arr: &Bound<PyArray1<f64>> = result.downcast().unwrap();
        let result_slice = unsafe { result_arr.as_slice().unwrap() };
        assert_eq!(result_slice, &[1.5, -2.5, 3.5]);
    });
}
