// fillna: replace NaN with a scalar fill value.

[[kernel]] void fillna_f32(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    device const float* fill   [[buffer(2)]],
    device const uint* len_ptr [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= *len_ptr) return;
    output[idx] = isnan(input[idx]) ? *fill : input[idx];
}

// Mask-based fillna for integer types (no NaN sentinel).
// If mask bit is clear (null), replace with fill value; otherwise keep input.

#define FILLNA_MASK_KERNEL(T, suffix) \
kernel void fillna_mask_##suffix( \
    device const T* input       [[buffer(0)]], \
    device T* output            [[buffer(1)]], \
    device const T* fill        [[buffer(2)]], \
    device const uint8_t* mask  [[buffer(3)]], \
    device const uint* len_ptr  [[buffer(4)]], \
    uint idx [[thread_position_in_grid]] \
) { \
    if (idx >= *len_ptr) return; \
    output[idx] = is_valid(mask, idx) ? input[idx] : *fill; \
}

FILLNA_MASK_KERNEL(int, i32)
FILLNA_MASK_KERNEL(long, i64)
FILLNA_MASK_KERNEL(char, i8)
FILLNA_MASK_KERNEL(short, i16)
FILLNA_MASK_KERNEL(uchar, u8)
FILLNA_MASK_KERNEL(ushort, u16)
FILLNA_MASK_KERNEL(uint, u32)
FILLNA_MASK_KERNEL(ulong, u64)
