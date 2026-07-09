#include <metal_stdlib>
using namespace metal;

// String comparison kernels — each thread processes one element

kernel void string_eq(
    device const int64_t* offsets_a [[buffer(0)]],
    device const uchar*   chars_a  [[buffer(1)]],
    device const int64_t* offsets_b [[buffer(2)]],
    device const uchar*   chars_b  [[buffer(3)]],
    device int32_t*       output   [[buffer(4)]],
    device const uint*    len_ptr  [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    StringRef a = get_string(offsets_a, chars_a, tid);
    StringRef b = get_string(offsets_b, chars_b, tid);
    output[tid] = string_equals(a, b) ? 1 : 0;
}

kernel void string_ne(
    device const int64_t* offsets_a [[buffer(0)]],
    device const uchar*   chars_a  [[buffer(1)]],
    device const int64_t* offsets_b [[buffer(2)]],
    device const uchar*   chars_b  [[buffer(3)]],
    device int32_t*       output   [[buffer(4)]],
    device const uint*    len_ptr  [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    StringRef a = get_string(offsets_a, chars_a, tid);
    StringRef b = get_string(offsets_b, chars_b, tid);
    output[tid] = string_equals(a, b) ? 0 : 1;
}

kernel void string_lt(
    device const int64_t* offsets_a [[buffer(0)]],
    device const uchar*   chars_a  [[buffer(1)]],
    device const int64_t* offsets_b [[buffer(2)]],
    device const uchar*   chars_b  [[buffer(3)]],
    device int32_t*       output   [[buffer(4)]],
    device const uint*    len_ptr  [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    StringRef a = get_string(offsets_a, chars_a, tid);
    StringRef b = get_string(offsets_b, chars_b, tid);
    output[tid] = (string_compare(a, b) < 0) ? 1 : 0;
}

kernel void string_gt(
    device const int64_t* offsets_a [[buffer(0)]],
    device const uchar*   chars_a  [[buffer(1)]],
    device const int64_t* offsets_b [[buffer(2)]],
    device const uchar*   chars_b  [[buffer(3)]],
    device int32_t*       output   [[buffer(4)]],
    device const uint*    len_ptr  [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    StringRef a = get_string(offsets_a, chars_a, tid);
    StringRef b = get_string(offsets_b, chars_b, tid);
    output[tid] = (string_compare(a, b) > 0) ? 1 : 0;
}

kernel void string_le(
    device const int64_t* offsets_a [[buffer(0)]],
    device const uchar*   chars_a  [[buffer(1)]],
    device const int64_t* offsets_b [[buffer(2)]],
    device const uchar*   chars_b  [[buffer(3)]],
    device int32_t*       output   [[buffer(4)]],
    device const uint*    len_ptr  [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    StringRef a = get_string(offsets_a, chars_a, tid);
    StringRef b = get_string(offsets_b, chars_b, tid);
    output[tid] = (string_compare(a, b) <= 0) ? 1 : 0;
}

kernel void string_ge(
    device const int64_t* offsets_a [[buffer(0)]],
    device const uchar*   chars_a  [[buffer(1)]],
    device const int64_t* offsets_b [[buffer(2)]],
    device const uchar*   chars_b  [[buffer(3)]],
    device int32_t*       output   [[buffer(4)]],
    device const uint*    len_ptr  [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    StringRef a = get_string(offsets_a, chars_a, tid);
    StringRef b = get_string(offsets_b, chars_b, tid);
    output[tid] = (string_compare(a, b) >= 0) ? 1 : 0;
}

// Scalar comparison: compare each string against a single pattern
kernel void string_eq_scalar(
    device const int64_t* offsets   [[buffer(0)]],
    device const uchar*   chars    [[buffer(1)]],
    device const int64_t* pat_offsets [[buffer(2)]],
    device const uchar*   pat_chars  [[buffer(3)]],
    device int32_t*       output   [[buffer(4)]],
    device const uint*    len_ptr  [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    StringRef s = get_string(offsets, chars, tid);
    StringRef p = get_string(pat_offsets, pat_chars, 0);
    output[tid] = string_equals(s, p) ? 1 : 0;
}
