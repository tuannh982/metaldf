// float32
UNARY_KERNEL(unary_abs,   float, f32, abs(a[idx]))
UNARY_KERNEL(unary_neg,   float, f32, -a[idx])
UNARY_KERNEL(unary_sqrt,  float, f32, sqrt(a[idx]))
UNARY_KERNEL(unary_exp,   float, f32, exp(a[idx]))
UNARY_KERNEL(unary_log,   float, f32, log(a[idx]))
UNARY_KERNEL(unary_ceil,  float, f32, ceil(a[idx]))
UNARY_KERNEL(unary_floor, float, f32, floor(a[idx]))

// float32 trig/hyperbolic (Phase 5 -- Metal has no `double`, so these are
// f32-only, unlike the pandas/numpy ops they mirror which also support f64)
UNARY_KERNEL(unary_sin,   float, f32, sin(a[idx]))
UNARY_KERNEL(unary_cos,   float, f32, cos(a[idx]))
UNARY_KERNEL(unary_tan,   float, f32, tan(a[idx]))
UNARY_KERNEL(unary_asin,  float, f32, asin(a[idx]))
UNARY_KERNEL(unary_acos,  float, f32, acos(a[idx]))
UNARY_KERNEL(unary_atan,  float, f32, atan(a[idx]))
UNARY_KERNEL(unary_sinh,  float, f32, sinh(a[idx]))
UNARY_KERNEL(unary_cosh,  float, f32, cosh(a[idx]))
UNARY_KERNEL(unary_tanh,  float, f32, tanh(a[idx]))

// float32 log
UNARY_KERNEL(unary_log2,  float, f32, log2(a[idx]))
UNARY_KERNEL(unary_log10, float, f32, log10(a[idx]))

// float32 rounding -- `rint` (not `round`) to match numpy's round-half-to-even
UNARY_KERNEL(unary_round, float, f32, rint(a[idx]))
UNARY_KERNEL(unary_trunc, float, f32, trunc(a[idx]))

// float32 power -- MSL has no `cbrt` builtin; copysign(pow(abs(x), 1/3), x)
// handles negative inputs the same way numpy's cbrt does (real-valued root
// with the sign of the input), whereas pow(x, 1/3) alone would return NaN
// for negative x.
UNARY_KERNEL(unary_cbrt,  float, f32, copysign(pow(abs(a[idx]), 1.0f/3.0f), a[idx]))

// int32
UNARY_KERNEL(unary_abs, int, i32, abs(a[idx]))
UNARY_KERNEL(unary_neg, int, i32, -a[idx])

// int64
UNARY_KERNEL(unary_abs, long, i64, abs(a[idx]))
UNARY_KERNEL(unary_neg, long, i64, -a[idx])

// int8
UNARY_KERNEL(unary_abs, char, i8, abs(a[idx]))
UNARY_KERNEL(unary_neg, char, i8, -a[idx])

// int16
UNARY_KERNEL(unary_abs, short, i16, abs(a[idx]))
UNARY_KERNEL(unary_neg, short, i16, -a[idx])

// uint8
UNARY_KERNEL(unary_abs, uchar, u8, a[idx])
UNARY_KERNEL(unary_neg, uchar, u8, -a[idx])

// uint16
UNARY_KERNEL(unary_abs, ushort, u16, a[idx])
UNARY_KERNEL(unary_neg, ushort, u16, -a[idx])

// uint32
UNARY_KERNEL(unary_abs, uint, u32, a[idx])
UNARY_KERNEL(unary_neg, uint, u32, -a[idx])

// uint64
UNARY_KERNEL(unary_abs, ulong, u64, a[idx])
UNARY_KERNEL(unary_neg, ulong, u64, -a[idx])

// float32 masked
UNARY_KERNEL_MASKED(unary_abs,   float, f32, abs(a[idx]))
UNARY_KERNEL_MASKED(unary_neg,   float, f32, -a[idx])
UNARY_KERNEL_MASKED(unary_sqrt,  float, f32, sqrt(a[idx]))
UNARY_KERNEL_MASKED(unary_exp,   float, f32, exp(a[idx]))
UNARY_KERNEL_MASKED(unary_log,   float, f32, log(a[idx]))
UNARY_KERNEL_MASKED(unary_ceil,  float, f32, ceil(a[idx]))
UNARY_KERNEL_MASKED(unary_floor, float, f32, floor(a[idx]))

// int32 masked
UNARY_KERNEL_MASKED(unary_abs, int, i32, abs(a[idx]))
UNARY_KERNEL_MASKED(unary_neg, int, i32, -a[idx])

// int64 masked
UNARY_KERNEL_MASKED(unary_abs, long, i64, abs(a[idx]))
UNARY_KERNEL_MASKED(unary_neg, long, i64, -a[idx])

// int8 masked
UNARY_KERNEL_MASKED(unary_abs, char, i8, abs(a[idx]))
UNARY_KERNEL_MASKED(unary_neg, char, i8, -a[idx])

// int16 masked
UNARY_KERNEL_MASKED(unary_abs, short, i16, abs(a[idx]))
UNARY_KERNEL_MASKED(unary_neg, short, i16, -a[idx])

// uint8 masked
UNARY_KERNEL_MASKED(unary_abs, uchar, u8, a[idx])
UNARY_KERNEL_MASKED(unary_neg, uchar, u8, -a[idx])

// uint16 masked
UNARY_KERNEL_MASKED(unary_abs, ushort, u16, a[idx])
UNARY_KERNEL_MASKED(unary_neg, ushort, u16, -a[idx])

// uint32 masked
UNARY_KERNEL_MASKED(unary_abs, uint, u32, a[idx])
UNARY_KERNEL_MASKED(unary_neg, uint, u32, -a[idx])

// uint64 masked
UNARY_KERNEL_MASKED(unary_abs, ulong, u64, a[idx])
UNARY_KERNEL_MASKED(unary_neg, ulong, u64, -a[idx])
