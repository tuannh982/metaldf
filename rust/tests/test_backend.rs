use metaldf_engine::backend::MetalBackend;

#[test]
fn test_device_is_available() {
    let device = MetalBackend::device();
    assert!(device.is_some(), "Metal device should be available on Apple Silicon");
}

#[test]
fn test_queue_is_created() {
    let queue = MetalBackend::queue();
    assert!(queue.is_some(), "Command queue should be created");
}
