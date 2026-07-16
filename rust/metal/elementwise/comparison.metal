// float32
CMP_KERNEL(cmp_eq, float, f32, ==)
CMP_KERNEL(cmp_ne, float, f32, !=)
CMP_KERNEL(cmp_lt, float, f32, <)
CMP_KERNEL(cmp_le, float, f32, <=)
CMP_KERNEL(cmp_gt, float, f32, >)
CMP_KERNEL(cmp_ge, float, f32, >=)

// int32
CMP_KERNEL(cmp_eq, int, i32, ==)
CMP_KERNEL(cmp_ne, int, i32, !=)
CMP_KERNEL(cmp_lt, int, i32, <)
CMP_KERNEL(cmp_le, int, i32, <=)
CMP_KERNEL(cmp_gt, int, i32, >)
CMP_KERNEL(cmp_ge, int, i32, >=)

// int64 (also backs Datetime/Timedelta — both stored as int64 nanoseconds)
CMP_KERNEL(cmp_eq, long, i64, ==)
CMP_KERNEL(cmp_ne, long, i64, !=)
CMP_KERNEL(cmp_lt, long, i64, <)
CMP_KERNEL(cmp_le, long, i64, <=)
CMP_KERNEL(cmp_gt, long, i64, >)
CMP_KERNEL(cmp_ge, long, i64, >=)

// int8
CMP_KERNEL(cmp_eq, char, i8, ==)
CMP_KERNEL(cmp_ne, char, i8, !=)
CMP_KERNEL(cmp_lt, char, i8, <)
CMP_KERNEL(cmp_le, char, i8, <=)
CMP_KERNEL(cmp_gt, char, i8, >)
CMP_KERNEL(cmp_ge, char, i8, >=)

// int16
CMP_KERNEL(cmp_eq, short, i16, ==)
CMP_KERNEL(cmp_ne, short, i16, !=)
CMP_KERNEL(cmp_lt, short, i16, <)
CMP_KERNEL(cmp_le, short, i16, <=)
CMP_KERNEL(cmp_gt, short, i16, >)
CMP_KERNEL(cmp_ge, short, i16, >=)

// uint8
CMP_KERNEL(cmp_eq, uchar, u8, ==)
CMP_KERNEL(cmp_ne, uchar, u8, !=)
CMP_KERNEL(cmp_lt, uchar, u8, <)
CMP_KERNEL(cmp_le, uchar, u8, <=)
CMP_KERNEL(cmp_gt, uchar, u8, >)
CMP_KERNEL(cmp_ge, uchar, u8, >=)

// uint16
CMP_KERNEL(cmp_eq, ushort, u16, ==)
CMP_KERNEL(cmp_ne, ushort, u16, !=)
CMP_KERNEL(cmp_lt, ushort, u16, <)
CMP_KERNEL(cmp_le, ushort, u16, <=)
CMP_KERNEL(cmp_gt, ushort, u16, >)
CMP_KERNEL(cmp_ge, ushort, u16, >=)

// uint32
CMP_KERNEL(cmp_eq, uint, u32, ==)
CMP_KERNEL(cmp_ne, uint, u32, !=)
CMP_KERNEL(cmp_lt, uint, u32, <)
CMP_KERNEL(cmp_le, uint, u32, <=)
CMP_KERNEL(cmp_gt, uint, u32, >)
CMP_KERNEL(cmp_ge, uint, u32, >=)

// uint64
CMP_KERNEL(cmp_eq, ulong, u64, ==)
CMP_KERNEL(cmp_ne, ulong, u64, !=)
CMP_KERNEL(cmp_lt, ulong, u64, <)
CMP_KERNEL(cmp_le, ulong, u64, <=)
CMP_KERNEL(cmp_gt, ulong, u64, >)
CMP_KERNEL(cmp_ge, ulong, u64, >=)
