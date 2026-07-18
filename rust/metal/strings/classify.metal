#include <metal_stdlib>
using namespace metal;

inline bool is_ascii_alpha(uchar c) {
    return (c >= 0x41 && c <= 0x5A) || (c >= 0x61 && c <= 0x7A);
}

inline bool is_ascii_digit(uchar c) {
    return c >= 0x30 && c <= 0x39;
}

inline bool is_ascii_space(uchar c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0B || c == 0x0C;
}

inline bool is_ascii_upper(uchar c) {
    return c >= 0x41 && c <= 0x5A;
}

inline bool is_ascii_lower(uchar c) {
    return c >= 0x61 && c <= 0x7A;
}

kernel void string_isalpha(
    device const int64_t* offsets [[buffer(0)]],
    device const uchar*   chars  [[buffer(1)]],
    device int32_t*       output [[buffer(2)]],
    device const uint*    len_ptr [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    StringRef s = get_string(offsets, chars, tid);
    if (s.len == 0) { output[tid] = 0; return; }
    for (uint i = 0; i < s.len; i++) {
        if (!is_ascii_alpha(s.data[i])) { output[tid] = 0; return; }
    }
    output[tid] = 1;
}

kernel void string_isdigit(
    device const int64_t* offsets [[buffer(0)]],
    device const uchar*   chars  [[buffer(1)]],
    device int32_t*       output [[buffer(2)]],
    device const uint*    len_ptr [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    StringRef s = get_string(offsets, chars, tid);
    if (s.len == 0) { output[tid] = 0; return; }
    for (uint i = 0; i < s.len; i++) {
        if (!is_ascii_digit(s.data[i])) { output[tid] = 0; return; }
    }
    output[tid] = 1;
}

kernel void string_isspace(
    device const int64_t* offsets [[buffer(0)]],
    device const uchar*   chars  [[buffer(1)]],
    device int32_t*       output [[buffer(2)]],
    device const uint*    len_ptr [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    StringRef s = get_string(offsets, chars, tid);
    if (s.len == 0) { output[tid] = 0; return; }
    for (uint i = 0; i < s.len; i++) {
        if (!is_ascii_space(s.data[i])) { output[tid] = 0; return; }
    }
    output[tid] = 1;
}

kernel void string_isalnum(
    device const int64_t* offsets [[buffer(0)]],
    device const uchar*   chars  [[buffer(1)]],
    device int32_t*       output [[buffer(2)]],
    device const uint*    len_ptr [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    StringRef s = get_string(offsets, chars, tid);
    if (s.len == 0) { output[tid] = 0; return; }
    for (uint i = 0; i < s.len; i++) {
        if (!is_ascii_alpha(s.data[i]) && !is_ascii_digit(s.data[i])) { output[tid] = 0; return; }
    }
    output[tid] = 1;
}

kernel void string_isupper(
    device const int64_t* offsets [[buffer(0)]],
    device const uchar*   chars  [[buffer(1)]],
    device int32_t*       output [[buffer(2)]],
    device const uint*    len_ptr [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    StringRef s = get_string(offsets, chars, tid);
    bool has_cased = false;
    for (uint i = 0; i < s.len; i++) {
        if (is_ascii_lower(s.data[i])) { output[tid] = 0; return; }
        if (is_ascii_upper(s.data[i])) has_cased = true;
    }
    output[tid] = has_cased ? 1 : 0;
}

kernel void string_islower(
    device const int64_t* offsets [[buffer(0)]],
    device const uchar*   chars  [[buffer(1)]],
    device int32_t*       output [[buffer(2)]],
    device const uint*    len_ptr [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    StringRef s = get_string(offsets, chars, tid);
    bool has_cased = false;
    for (uint i = 0; i < s.len; i++) {
        if (is_ascii_upper(s.data[i])) { output[tid] = 0; return; }
        if (is_ascii_lower(s.data[i])) has_cased = true;
    }
    output[tid] = has_cased ? 1 : 0;
}

kernel void string_istitle(
    device const int64_t* offsets [[buffer(0)]],
    device const uchar*   chars  [[buffer(1)]],
    device int32_t*       output [[buffer(2)]],
    device const uint*    len_ptr [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    StringRef s = get_string(offsets, chars, tid);
    if (s.len == 0) { output[tid] = 0; return; }
    bool has_cased = false;
    bool after_boundary = true;
    for (uint i = 0; i < s.len; i++) {
        uchar c = s.data[i];
        if (is_ascii_upper(c)) {
            if (!after_boundary) { output[tid] = 0; return; }
            has_cased = true;
            after_boundary = false;
        } else if (is_ascii_lower(c)) {
            if (after_boundary) { output[tid] = 0; return; }
            has_cased = true;
            after_boundary = false;
        } else {
            after_boundary = true;
        }
    }
    output[tid] = has_cased ? 1 : 0;
}
