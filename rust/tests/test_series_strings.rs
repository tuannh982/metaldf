use pyo3::prelude::*;
use pyo3::types::PyList;
use metaldf_engine::series::MetalSeries;

#[test]
fn test_string_roundtrip_basic() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let strings: Vec<String> = vec!["hello".into(), "world".into(), "!".into()];
        let py_list = PyList::new_bound(py, &strings);
        let series = MetalSeries::from_strings(&py_list).expect("from_strings failed");
        assert_eq!(series.len(), 3);
        assert_eq!(series.dtype(), "Utf8");

        let result = series.to_strings(py).expect("to_strings failed");
        let result_vec: Vec<String> = result.extract().unwrap();
        assert_eq!(result_vec, strings);
    });
}

#[test]
fn test_string_roundtrip_empty_strings() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let strings: Vec<String> = vec!["".into(), "".into(), "".into()];
        let py_list = PyList::new_bound(py, &strings);
        let series = MetalSeries::from_strings(&py_list).expect("from_strings failed");
        assert_eq!(series.len(), 3);

        let result = series.to_strings(py).expect("to_strings failed");
        let result_vec: Vec<String> = result.extract().unwrap();
        assert_eq!(result_vec, strings);
    });
}

#[test]
fn test_string_roundtrip_mixed_lengths() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let strings: Vec<String> = vec!["a".into(), "hello world".into(), "".into()];
        let py_list = PyList::new_bound(py, &strings);
        let series = MetalSeries::from_strings(&py_list).expect("from_strings failed");
        assert_eq!(series.len(), 3);

        let result = series.to_strings(py).expect("to_strings failed");
        let result_vec: Vec<String> = result.extract().unwrap();
        assert_eq!(result_vec, strings);
    });
}

#[test]
fn test_string_offsets_layout() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let strings: Vec<String> = vec!["hello".into(), "world".into(), "!".into()];
        let py_list = PyList::new_bound(py, &strings);
        let series = MetalSeries::from_strings(&py_list).expect("from_strings failed");

        let (offsets, chars) = series.as_str();
        // offsets: [0, 5, 10, 11] — 4 i64 values for 3 strings
        assert_eq!(offsets.len, 4);
        // chars: "helloworld!" — 11 bytes
        assert_eq!(chars.len, 11);

        let offsets_ptr = offsets.metal_buffer().contents() as *const i64;
        unsafe {
            assert_eq!(*offsets_ptr.add(0), 0);
            assert_eq!(*offsets_ptr.add(1), 5);
            assert_eq!(*offsets_ptr.add(2), 10);
            assert_eq!(*offsets_ptr.add(3), 11);
        }
    });
}
