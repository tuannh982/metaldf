// float32
BINARY_KERNEL(binary_add, float, f32, a[idx] + b[idx])
BINARY_KERNEL(binary_sub, float, f32, a[idx] - b[idx])
BINARY_KERNEL(binary_mul, float, f32, a[idx] * b[idx])
BINARY_KERNEL(binary_div, float, f32, a[idx] / b[idx])
BINARY_KERNEL(binary_mod, float, f32, mod_op(a[idx], b[idx]))

// int32
BINARY_KERNEL(binary_add, int, i32, a[idx] + b[idx])
BINARY_KERNEL(binary_sub, int, i32, a[idx] - b[idx])
BINARY_KERNEL(binary_mul, int, i32, a[idx] * b[idx])
BINARY_KERNEL(binary_div, int, i32, a[idx] / b[idx])
BINARY_KERNEL(binary_mod, int, i32, mod_op(a[idx], b[idx]))

// int64
BINARY_KERNEL(binary_add, long, i64, a[idx] + b[idx])
BINARY_KERNEL(binary_sub, long, i64, a[idx] - b[idx])
BINARY_KERNEL(binary_mul, long, i64, a[idx] * b[idx])
BINARY_KERNEL(binary_div, long, i64, a[idx] / b[idx])
BINARY_KERNEL(binary_mod, long, i64, mod_op(a[idx], b[idx]))
