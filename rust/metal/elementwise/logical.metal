// Logical AND/OR/NOT kernels on `Bool`-dtype data (one uint8_t per element,
// 0 = false / 1 = true — see `DType::Bool` in rust/src/buffer.rs). Unlike
// the arithmetic/comparison kernels generated from BINARY_KERNEL/CMP_KERNEL
// in 01_types.h, these operate on a single fixed type (uint8_t), so they're
// written directly rather than through a macro.

kernel void logical_and_bool(device const uint8_t* a [[buffer(0)]],
                              device const uint8_t* b [[buffer(1)]],
                              device uint8_t* out     [[buffer(2)]],
                              uint idx [[thread_position_in_grid]]) {
    out[idx] = (a[idx] && b[idx]) ? 1 : 0;
}

kernel void logical_or_bool(device const uint8_t* a [[buffer(0)]],
                             device const uint8_t* b [[buffer(1)]],
                             device uint8_t* out     [[buffer(2)]],
                             uint idx [[thread_position_in_grid]]) {
    out[idx] = (a[idx] || b[idx]) ? 1 : 0;
}

kernel void logical_not_bool(device const uint8_t* a [[buffer(0)]],
                              device uint8_t* out     [[buffer(1)]],
                              uint idx [[thread_position_in_grid]]) {
    out[idx] = a[idx] ? 0 : 1;
}
