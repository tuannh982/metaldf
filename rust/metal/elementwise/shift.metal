// Shift kernel: copy with offset, fill out-of-bounds with NaN (float) or 0 (int).
//
// NOTE: `fill_value` is passed as a regular function argument (computed once
// per-thread from the macro's `fill` expression) rather than a non-type
// template parameter -- Metal Shading Language (C++14/17-based) only allows
// integral/enum non-type template parameters, so `template <typename T, T
// fill_value>` fails to compile for `T = float` (the `as_type<float>(...)`
// NaN bit-pattern case below).

template <typename T>
void shift_impl(
    device const T* input,
    device T* output,
    device const int* periods_ptr,
    device const uint* len_ptr,
    T fill_value,
    uint idx
) {
    uint len = *len_ptr;
    if (idx >= len) return;
    int p = *periods_ptr;
    int src = int(idx) - p;
    if (src >= 0 && src < int(len)) {
        output[idx] = input[src];
    } else {
        output[idx] = fill_value;
    }
}

#define SHIFT_KERNEL(suffix, metal_type, fill) \
    [[kernel]] void shift_##suffix( \
        device const metal_type* input [[buffer(0)]], \
        device metal_type* output      [[buffer(1)]], \
        device const int* periods_ptr  [[buffer(2)]], \
        device const uint* len_ptr     [[buffer(3)]], \
        uint idx [[thread_position_in_grid]] \
    ) { shift_impl<metal_type>(input, output, periods_ptr, len_ptr, (metal_type)(fill), idx); }

SHIFT_KERNEL(float32, float, as_type<float>(0x7FC00000u))
SHIFT_KERNEL(int32, int, 0)
SHIFT_KERNEL(int64, long, 0L)
