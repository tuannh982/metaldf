use metaldf_engine::backend::{MetalBackend, BatchContext};

#[test]
fn test_batch_context_create_and_commit() {
    let batch = MetalBackend::begin_batch().unwrap();
    // Empty batch should commit successfully
    batch.commit_and_wait().unwrap();
}
