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

// int8
BINARY_KERNEL(binary_add, char, i8, a[idx] + b[idx])
BINARY_KERNEL(binary_sub, char, i8, a[idx] - b[idx])
BINARY_KERNEL(binary_mul, char, i8, a[idx] * b[idx])
BINARY_KERNEL(binary_div, char, i8, a[idx] / b[idx])
BINARY_KERNEL(binary_mod, char, i8, mod_op(a[idx], b[idx]))

// int16
BINARY_KERNEL(binary_add, short, i16, a[idx] + b[idx])
BINARY_KERNEL(binary_sub, short, i16, a[idx] - b[idx])
BINARY_KERNEL(binary_mul, short, i16, a[idx] * b[idx])
BINARY_KERNEL(binary_div, short, i16, a[idx] / b[idx])
BINARY_KERNEL(binary_mod, short, i16, mod_op(a[idx], b[idx]))

// uint8
BINARY_KERNEL(binary_add, uchar, u8, a[idx] + b[idx])
BINARY_KERNEL(binary_sub, uchar, u8, a[idx] - b[idx])
BINARY_KERNEL(binary_mul, uchar, u8, a[idx] * b[idx])
BINARY_KERNEL(binary_div, uchar, u8, a[idx] / b[idx])
BINARY_KERNEL(binary_mod, uchar, u8, mod_op(a[idx], b[idx]))

// uint16
BINARY_KERNEL(binary_add, ushort, u16, a[idx] + b[idx])
BINARY_KERNEL(binary_sub, ushort, u16, a[idx] - b[idx])
BINARY_KERNEL(binary_mul, ushort, u16, a[idx] * b[idx])
BINARY_KERNEL(binary_div, ushort, u16, a[idx] / b[idx])
BINARY_KERNEL(binary_mod, ushort, u16, mod_op(a[idx], b[idx]))

// uint32
BINARY_KERNEL(binary_add, uint, u32, a[idx] + b[idx])
BINARY_KERNEL(binary_sub, uint, u32, a[idx] - b[idx])
BINARY_KERNEL(binary_mul, uint, u32, a[idx] * b[idx])
BINARY_KERNEL(binary_div, uint, u32, a[idx] / b[idx])
BINARY_KERNEL(binary_mod, uint, u32, mod_op(a[idx], b[idx]))

// uint64
BINARY_KERNEL(binary_add, ulong, u64, a[idx] + b[idx])
BINARY_KERNEL(binary_sub, ulong, u64, a[idx] - b[idx])
BINARY_KERNEL(binary_mul, ulong, u64, a[idx] * b[idx])
BINARY_KERNEL(binary_div, ulong, u64, a[idx] / b[idx])
BINARY_KERNEL(binary_mod, ulong, u64, mod_op(a[idx], b[idx]))

// float32 masked
BINARY_KERNEL_MASKED(binary_add, float, f32, a[idx] + b[idx])
BINARY_KERNEL_MASKED(binary_sub, float, f32, a[idx] - b[idx])
BINARY_KERNEL_MASKED(binary_mul, float, f32, a[idx] * b[idx])
BINARY_KERNEL_MASKED(binary_div, float, f32, a[idx] / b[idx])
BINARY_KERNEL_MASKED(binary_mod, float, f32, mod_op(a[idx], b[idx]))

// int32 masked
BINARY_KERNEL_MASKED(binary_add, int, i32, a[idx] + b[idx])
BINARY_KERNEL_MASKED(binary_sub, int, i32, a[idx] - b[idx])
BINARY_KERNEL_MASKED(binary_mul, int, i32, a[idx] * b[idx])
BINARY_KERNEL_MASKED(binary_div, int, i32, a[idx] / b[idx])
BINARY_KERNEL_MASKED(binary_mod, int, i32, mod_op(a[idx], b[idx]))

// int64 masked
BINARY_KERNEL_MASKED(binary_add, long, i64, a[idx] + b[idx])
BINARY_KERNEL_MASKED(binary_sub, long, i64, a[idx] - b[idx])
BINARY_KERNEL_MASKED(binary_mul, long, i64, a[idx] * b[idx])
BINARY_KERNEL_MASKED(binary_div, long, i64, a[idx] / b[idx])
BINARY_KERNEL_MASKED(binary_mod, long, i64, mod_op(a[idx], b[idx]))

// int8 masked
BINARY_KERNEL_MASKED(binary_add, char, i8, a[idx] + b[idx])
BINARY_KERNEL_MASKED(binary_sub, char, i8, a[idx] - b[idx])
BINARY_KERNEL_MASKED(binary_mul, char, i8, a[idx] * b[idx])
BINARY_KERNEL_MASKED(binary_div, char, i8, a[idx] / b[idx])
BINARY_KERNEL_MASKED(binary_mod, char, i8, mod_op(a[idx], b[idx]))

// int16 masked
BINARY_KERNEL_MASKED(binary_add, short, i16, a[idx] + b[idx])
BINARY_KERNEL_MASKED(binary_sub, short, i16, a[idx] - b[idx])
BINARY_KERNEL_MASKED(binary_mul, short, i16, a[idx] * b[idx])
BINARY_KERNEL_MASKED(binary_div, short, i16, a[idx] / b[idx])
BINARY_KERNEL_MASKED(binary_mod, short, i16, mod_op(a[idx], b[idx]))

// uint8 masked
BINARY_KERNEL_MASKED(binary_add, uchar, u8, a[idx] + b[idx])
BINARY_KERNEL_MASKED(binary_sub, uchar, u8, a[idx] - b[idx])
BINARY_KERNEL_MASKED(binary_mul, uchar, u8, a[idx] * b[idx])
BINARY_KERNEL_MASKED(binary_div, uchar, u8, a[idx] / b[idx])
BINARY_KERNEL_MASKED(binary_mod, uchar, u8, mod_op(a[idx], b[idx]))

// uint16 masked
BINARY_KERNEL_MASKED(binary_add, ushort, u16, a[idx] + b[idx])
BINARY_KERNEL_MASKED(binary_sub, ushort, u16, a[idx] - b[idx])
BINARY_KERNEL_MASKED(binary_mul, ushort, u16, a[idx] * b[idx])
BINARY_KERNEL_MASKED(binary_div, ushort, u16, a[idx] / b[idx])
BINARY_KERNEL_MASKED(binary_mod, ushort, u16, mod_op(a[idx], b[idx]))

// uint32 masked
BINARY_KERNEL_MASKED(binary_add, uint, u32, a[idx] + b[idx])
BINARY_KERNEL_MASKED(binary_sub, uint, u32, a[idx] - b[idx])
BINARY_KERNEL_MASKED(binary_mul, uint, u32, a[idx] * b[idx])
BINARY_KERNEL_MASKED(binary_div, uint, u32, a[idx] / b[idx])
BINARY_KERNEL_MASKED(binary_mod, uint, u32, mod_op(a[idx], b[idx]))

// uint64 masked
BINARY_KERNEL_MASKED(binary_add, ulong, u64, a[idx] + b[idx])
BINARY_KERNEL_MASKED(binary_sub, ulong, u64, a[idx] - b[idx])
BINARY_KERNEL_MASKED(binary_mul, ulong, u64, a[idx] * b[idx])
BINARY_KERNEL_MASKED(binary_div, ulong, u64, a[idx] / b[idx])
BINARY_KERNEL_MASKED(binary_mod, ulong, u64, mod_op(a[idx], b[idx]))
