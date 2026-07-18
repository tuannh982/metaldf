#include <metal_stdlib>
using namespace metal;

kernel void string_len(
    device const int64_t* offsets [[buffer(0)]],
    device int64_t*       output [[buffer(1)]],
    device const uint*    len_ptr [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    output[tid] = offsets[tid + 1] - offsets[tid];
}

kernel void string_count(
    device const int64_t* offsets   [[buffer(0)]],
    device const uchar*   chars    [[buffer(1)]],
    device const int64_t* pat_offsets [[buffer(2)]],
    device const uchar*   pat_chars  [[buffer(3)]],
    device int64_t*       output   [[buffer(4)]],
    device const uint*    len_ptr  [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    StringRef s = get_string(offsets, chars, tid);
    StringRef p = get_string(pat_offsets, pat_chars, 0);
    if (p.len == 0) {
        output[tid] = int64_t(s.len + 1);
        return;
    }
    if (s.len < p.len) {
        output[tid] = 0;
        return;
    }
    int64_t count = 0;
    uint i = 0;
    while (i + p.len <= s.len) {
        bool match = true;
        for (uint j = 0; j < p.len; j++) {
            if (s.data[i + j] != p.data[j]) { match = false; break; }
        }
        if (match) {
            count++;
            i += p.len;
        } else {
            i++;
        }
    }
    output[tid] = count;
}
