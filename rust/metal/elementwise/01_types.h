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
