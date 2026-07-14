// Kernel generation macros for elementwise operations.
// Each macro produces a named kernel function from a type, suffix, and expression.

#define BINARY_KERNEL(name, T, suffix, expr)                         \
kernel void name##_##suffix(device const T* a [[buffer(0)]],         \
                            device const T* b [[buffer(1)]],         \
                            device T* out      [[buffer(2)]],        \
                            uint idx [[thread_position_in_grid]]) {  \
    out[idx] = (expr);                                               \
}

#define UNARY_KERNEL(name, T, suffix, expr)                          \
kernel void name##_##suffix(device const T* a [[buffer(0)]],         \
                            device T* out      [[buffer(1)]],        \
                            uint idx [[thread_position_in_grid]]) {  \
    out[idx] = (expr);                                               \
}

#define CMP_KERNEL(name, T, suffix, op)                              \
kernel void name##_##suffix(device const T* a [[buffer(0)]],         \
                            device const T* b [[buffer(1)]],         \
                            device int* out    [[buffer(2)]],        \
                            uint idx [[thread_position_in_grid]]) {  \
    out[idx] = (a[idx] op b[idx]) ? 1 : 0;                          \
}

// Null-aware variants. `mask_a`/`mask_b`/`mask_in` are packed validity
// bitmasks consulted via `is_valid()` (see `common/04_null_mask.h`); a
// `nullptr` mask means "no nulls, everything valid" so a fast path with no
// mask on either operand never needs to bind these kernels at all (see
// `dispatch_binary_inner`/`dispatch_unary_inner` in
// `rust/src/kernels/elementwise.rs`).
//
// `mask_out`/`valid_out` is intentionally NOT a packed bitmask: it's one
// `uint8_t` per element (0 = null, 1 = valid). Elementwise kernels run one
// thread per element with no threadgroup coordination, so if this were a
// packed bitmask, threads whose elements share a byte (8 elements/byte)
// would race on a non-atomic `mask[i/8] |= ...` read-modify-write. Writing
// one full byte per element instead means every thread only ever touches
// its own byte — no races, no atomics needed. The Rust dispatch side packs
// this per-element buffer down into a proper bit-packed `NullMask` after
// the kernel completes (see `pack_validity_to_mask`), which is cheap CPU
// work compared to the kernel launch itself.
#define BINARY_KERNEL_MASKED(name, T, suffix, expr)                          \
kernel void name##_##suffix##_masked(                                        \
    device const T* a            [[buffer(0)]],                              \
    device const T* b            [[buffer(1)]],                              \
    device T* out                [[buffer(2)]],                              \
    device const uint8_t* mask_a [[buffer(3)]],                              \
    device const uint8_t* mask_b [[buffer(4)]],                              \
    device uint8_t* valid_out    [[buffer(5)]],                              \
    uint idx [[thread_position_in_grid]]) {                                  \
    bool va = is_valid(mask_a, idx);                                         \
    bool vb = is_valid(mask_b, idx);                                         \
    if (va && vb) {                                                          \
        out[idx] = (expr);                                                   \
        valid_out[idx] = 1;                                                  \
    } else {                                                                 \
        out[idx] = T(0);                                                     \
        valid_out[idx] = 0;                                                  \
    }                                                                         \
}

#define UNARY_KERNEL_MASKED(name, T, suffix, expr)                           \
kernel void name##_##suffix##_masked(                                        \
    device const T* a             [[buffer(0)]],                             \
    device T* out                 [[buffer(1)]],                             \
    device const uint8_t* mask_in [[buffer(2)]],                             \
    device uint8_t* valid_out     [[buffer(3)]],                             \
    uint idx [[thread_position_in_grid]]) {                                  \
    if (is_valid(mask_in, idx)) {                                            \
        out[idx] = (expr);                                                   \
        valid_out[idx] = 1;                                                  \
    } else {                                                                 \
        out[idx] = T(0);                                                     \
        valid_out[idx] = 0;                                                  \
    }                                                                         \
}
