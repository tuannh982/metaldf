// float32
UNARY_KERNEL(unary_abs,   float, f32, abs(a[idx]))
UNARY_KERNEL(unary_neg,   float, f32, -a[idx])
UNARY_KERNEL(unary_sqrt,  float, f32, sqrt(a[idx]))
UNARY_KERNEL(unary_exp,   float, f32, exp(a[idx]))
UNARY_KERNEL(unary_log,   float, f32, log(a[idx]))
UNARY_KERNEL(unary_ceil,  float, f32, ceil(a[idx]))
UNARY_KERNEL(unary_floor, float, f32, floor(a[idx]))

// int32
UNARY_KERNEL(unary_abs, int, i32, abs(a[idx]))
UNARY_KERNEL(unary_neg, int, i32, -a[idx])
