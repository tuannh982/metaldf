#include <metal_stdlib>
using namespace metal;

kernel void string_contains(
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
    if (p.len == 0) { output[tid] = 1; return; }
    if (s.len < p.len) { output[tid] = 0; return; }
    output[tid] = 0;
    for (uint i = 0; i <= s.len - p.len; i++) {
        bool match = true;
        for (uint j = 0; j < p.len; j++) {
            if (s.data[i + j] != p.data[j]) { match = false; break; }
        }
        if (match) { output[tid] = 1; return; }
    }
}

kernel void string_startswith(
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
    if (p.len == 0) { output[tid] = 1; return; }
    if (s.len < p.len) { output[tid] = 0; return; }
    output[tid] = 1;
    for (uint i = 0; i < p.len; i++) {
        if (s.data[i] != p.data[i]) { output[tid] = 0; return; }
    }
}

kernel void string_endswith(
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
    if (p.len == 0) { output[tid] = 1; return; }
    if (s.len < p.len) { output[tid] = 0; return; }
    uint off = s.len - p.len;
    output[tid] = 1;
    for (uint i = 0; i < p.len; i++) {
        if (s.data[off + i] != p.data[i]) { output[tid] = 0; return; }
    }
}

kernel void string_find(
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
    if (p.len == 0) { output[tid] = 0; return; }
    if (s.len < p.len) { output[tid] = -1; return; }
    for (uint i = 0; i <= s.len - p.len; i++) {
        bool match = true;
        for (uint j = 0; j < p.len; j++) {
            if (s.data[i + j] != p.data[j]) { match = false; break; }
        }
        if (match) { output[tid] = int32_t(i); return; }
    }
    output[tid] = -1;
}
