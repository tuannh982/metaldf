use metaldf_engine::column::MetalColumn;
use metaldf_engine::column_view::MetalColumnView;
use metaldf_engine::buffer::DType;
use metaldf_engine::series::MetalSeries;
use metal::MTLResourceOptions;
use numpy::PyArray1;
use pyo3::Python;

#[test]
fn test_column_from_buffer() {
    let device = metal::Device::system_default().unwrap();
    let data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let buffer = device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let col = MetalColumn::from_buffer(buffer, 3, DType::Float32);
    assert_eq!(col.size(), 3);
    assert_eq!(col.dtype(), DType::Float32);
    assert_eq!(col.offset(), 0);
    assert!(col.null_mask().is_none());
    assert_eq!(col.num_children(), 0);
}

#[test]
fn test_column_with_offset() {
    let device = metal::Device::system_default().unwrap();
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let buffer = device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let col = MetalColumn::new(
        std::sync::Arc::new(buffer),
        None, DType::Float32, 3, 2, vec![],
    );
    assert_eq!(col.size(), 3);
    assert_eq!(col.offset(), 2);
}

#[test]
fn test_column_with_children() {
    let device = metal::Device::system_default().unwrap();
    let offsets: Vec<i64> = vec![0, 5, 11];
    let chars: Vec<u8> = b"helloworld!".to_vec();
    let off_buf = device.new_buffer_with_data(
        offsets.as_ptr() as *const _, (offsets.len() * 8) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let char_buf = device.new_buffer_with_data(
        chars.as_ptr() as *const _, chars.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let off_col = MetalColumn::from_buffer(off_buf, offsets.len(), DType::Int64);
    let char_col = MetalColumn::from_buffer(char_buf, chars.len(), DType::Uint8);
    let str_col = MetalColumn::new(
        std::sync::Arc::new(device.new_buffer(1, MTLResourceOptions::StorageModeShared)),
        None, DType::Utf8, 2, 0, vec![off_col, char_col],
    );
    assert_eq!(str_col.num_children(), 2);
    assert_eq!(str_col.child(0).dtype(), DType::Int64);
    assert_eq!(str_col.child(1).dtype(), DType::Uint8);
}

#[test]
fn test_column_view_from_column() {
    let device = metal::Device::system_default().unwrap();
    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let buffer = device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let col = MetalColumn::new(
        std::sync::Arc::new(buffer), None, DType::Float32, 3, 2, vec![],
    );
    let view = col.view();
    assert_eq!(view.size(), 3);
    assert_eq!(view.offset(), 2);
    assert_eq!(view.dtype(), DType::Float32);
    assert_eq!(view.data_ptr_offset(), 8); // 2 elements * 4 bytes
    assert!(view.null_mask().is_none());
}

#[test]
fn test_column_view_from_sharedbuffer() {
    let device = metal::Device::system_default().unwrap();
    let data: Vec<f32> = vec![10.0, 20.0, 30.0];
    let buffer = device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let view = MetalColumnView::new(&buffer, None, DType::Float32, 3, 0);
    assert_eq!(view.size(), 3);
    assert_eq!(view.data_ptr_offset(), 0);
}

#[test]
fn test_series_has_column_and_view() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let arr = PyArray1::<f32>::from_vec_bound(py, vec![1.0, 2.0, 3.0]);
        let series = MetalSeries::from_numpy(&arr).unwrap();
        let col = series.column();
        assert_eq!(col.size(), 3);
        assert_eq!(col.dtype(), DType::Float32);
        let view = series.view();
        assert_eq!(view.size(), 3);
    });
}

#[test]
fn test_series_from_column() {
    let device = metal::Device::system_default().unwrap();
    let data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let buffer = device.new_buffer_with_data(
        data.as_ptr() as *const _,
        (data.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let col = MetalColumn::from_buffer(buffer, 3, DType::Float32);
    let series = MetalSeries::from_column(col);
    assert_eq!(series.len(), 3);
}
