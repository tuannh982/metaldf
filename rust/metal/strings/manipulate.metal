#include <metal_stdlib>
using namespace metal;

// --- slice: two-pass ---

kernel void string_slice_sizes(
    device const int64_t* offsets_in [[buffer(0)]],
    device int64_t*       sizes     [[buffer(1)]],
    device const uint*    len_ptr   [[buffer(2)]],
    device const int*     params    [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    int slen = int(offsets_in[tid + 1] - offsets_in[tid]);
    int s = params[0];
    int e = params[1];
    if (s < 0) s = max(0, slen + s);
    if (e < 0) e = max(0, slen + e);
    s = min(s, slen);
    e = min(e, slen);
    sizes[tid] = int64_t(max(0, e - s));
}

kernel void string_slice_write(
    device const int64_t* offsets_in  [[buffer(0)]],
    device const uchar*   chars_in   [[buffer(1)]],
    device const int64_t* offsets_out [[buffer(2)]],
    device uchar*         chars_out  [[buffer(3)]],
    device const uint*    len_ptr    [[buffer(4)]],
    device const int*     params     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    int64_t in_start = offsets_in[tid];
    int slen = int(offsets_in[tid + 1] - in_start);
    int s = params[0];
    int e = params[1];
    if (s < 0) s = max(0, slen + s);
    if (e < 0) e = max(0, slen + e);
    s = min(s, slen);
    e = min(e, slen);
    int64_t out_start = offsets_out[tid];
    for (int i = s; i < e; i++) {
        chars_out[out_start + (i - s)] = chars_in[in_start + i];
    }
}

// --- get: two-pass ---

kernel void string_get_sizes(
    device const int64_t* offsets_in [[buffer(0)]],
    device int64_t*       sizes     [[buffer(1)]],
    device const uint*    len_ptr   [[buffer(2)]],
    device const int*     idx_ptr   [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    int slen = int(offsets_in[tid + 1] - offsets_in[tid]);
    int idx = *idx_ptr;
    if (idx < 0) idx = slen + idx;
    sizes[tid] = (idx >= 0 && idx < slen) ? 1 : 0;
}

kernel void string_get_write(
    device const int64_t* offsets_in  [[buffer(0)]],
    device const uchar*   chars_in   [[buffer(1)]],
    device const int64_t* offsets_out [[buffer(2)]],
    device uchar*         chars_out  [[buffer(3)]],
    device const uint*    len_ptr    [[buffer(4)]],
    device const int*     idx_ptr    [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    int64_t in_start = offsets_in[tid];
    int slen = int(offsets_in[tid + 1] - in_start);
    int idx = *idx_ptr;
    if (idx < 0) idx = slen + idx;
    if (idx >= 0 && idx < slen) {
        chars_out[offsets_out[tid]] = chars_in[in_start + idx];
    }
}

// --- repeat: two-pass ---

kernel void string_repeat_sizes(
    device const int64_t* offsets_in [[buffer(0)]],
    device int64_t*       sizes     [[buffer(1)]],
    device const uint*    len_ptr   [[buffer(2)]],
    device const uint*    repeat_n  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    int64_t slen = offsets_in[tid + 1] - offsets_in[tid];
    sizes[tid] = slen * int64_t(*repeat_n);
}

kernel void string_repeat_write(
    device const int64_t* offsets_in  [[buffer(0)]],
    device const uchar*   chars_in   [[buffer(1)]],
    device const int64_t* offsets_out [[buffer(2)]],
    device uchar*         chars_out  [[buffer(3)]],
    device const uint*    len_ptr    [[buffer(4)]],
    device const uint*    repeat_n   [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    int64_t in_start = offsets_in[tid];
    int64_t slen = offsets_in[tid + 1] - in_start;
    int64_t out_pos = offsets_out[tid];
    uint rn = *repeat_n;
    for (uint r = 0; r < rn; r++) {
        for (int64_t i = 0; i < slen; i++) {
            chars_out[out_pos++] = chars_in[in_start + i];
        }
    }
}

// --- pad: two-pass, side: 0=left, 1=right, 2=both ---

kernel void string_pad_sizes(
    device const int64_t* offsets_in [[buffer(0)]],
    device int64_t*       sizes     [[buffer(1)]],
    device const uint*    len_ptr   [[buffer(2)]],
    device const int*     params    [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    int slen = int(offsets_in[tid + 1] - offsets_in[tid]);
    int width = params[0];
    sizes[tid] = int64_t(max(slen, width));
}

kernel void string_pad_write(
    device const int64_t* offsets_in  [[buffer(0)]],
    device const uchar*   chars_in   [[buffer(1)]],
    device const int64_t* offsets_out [[buffer(2)]],
    device uchar*         chars_out  [[buffer(3)]],
    device const uint*    len_ptr    [[buffer(4)]],
    device const int*     params     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    int64_t in_start = offsets_in[tid];
    int slen = int(offsets_in[tid + 1] - in_start);
    int width = params[0];
    int side = params[1];
    uchar fillchar = uchar(params[2]);
    int64_t out_start = offsets_out[tid];
    int pad_total = max(0, width - slen);
    int left_pad = 0;
    if (side == 0) left_pad = pad_total;
    else if (side == 2) {
        // Matches CPython's str.center (and thus pandas' str.center):
        // marg = pad_total; left = marg/2, plus one extra when *both*
        // marg and width are odd -- a plain floor(pad_total/2) (as this
        // used to compute) puts the extra padding char on the wrong side
        // whenever width is odd (e.g. slen=20, width=25 -> pad_total=5,
        // correct left=3/right=2, not left=2/right=3).
        left_pad = pad_total / 2 + (((pad_total & 1) != 0 && (width & 1) != 0) ? 1 : 0);
    }
    int right_pad = pad_total - left_pad;
    int64_t pos = out_start;
    for (int i = 0; i < left_pad; i++) chars_out[pos++] = fillchar;
    for (int i = 0; i < slen; i++) chars_out[pos++] = chars_in[in_start + i];
    for (int i = 0; i < right_pad; i++) chars_out[pos++] = fillchar;
}

// --- zfill: pad with '0' on left, sign-aware ---

kernel void string_zfill_sizes(
    device const int64_t* offsets_in [[buffer(0)]],
    device int64_t*       sizes     [[buffer(1)]],
    device const uint*    len_ptr   [[buffer(2)]],
    device const int*     width_ptr [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    int slen = int(offsets_in[tid + 1] - offsets_in[tid]);
    sizes[tid] = int64_t(max(slen, *width_ptr));
}

kernel void string_zfill_write(
    device const int64_t* offsets_in  [[buffer(0)]],
    device const uchar*   chars_in   [[buffer(1)]],
    device const int64_t* offsets_out [[buffer(2)]],
    device uchar*         chars_out  [[buffer(3)]],
    device const uint*    len_ptr    [[buffer(4)]],
    device const int*     width_ptr  [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    int64_t in_start = offsets_in[tid];
    int slen = int(offsets_in[tid + 1] - in_start);
    int width = *width_ptr;
    int64_t out_start = offsets_out[tid];
    int pad = max(0, width - slen);

    bool has_sign = (slen > 0) && (chars_in[in_start] == '+' || chars_in[in_start] == '-');
    int64_t pos = out_start;
    if (has_sign) {
        chars_out[pos++] = chars_in[in_start];
        for (int i = 0; i < pad; i++) chars_out[pos++] = '0';
        for (int i = 1; i < slen; i++) chars_out[pos++] = chars_in[in_start + i];
    } else {
        for (int i = 0; i < pad; i++) chars_out[pos++] = '0';
        for (int i = 0; i < slen; i++) chars_out[pos++] = chars_in[in_start + i];
    }
}

// --- removeprefix / removesuffix: two-pass, direction: 0=prefix, 1=suffix ---

kernel void string_remove_affix_sizes(
    device const int64_t* offsets_in  [[buffer(0)]],
    device const uchar*   chars_in   [[buffer(1)]],
    device const int64_t* pat_offsets [[buffer(2)]],
    device const uchar*   pat_chars  [[buffer(3)]],
    device int64_t*       sizes      [[buffer(4)]],
    device const uint*    len_ptr    [[buffer(5)]],
    device const uint*    dir_ptr    [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    StringRef s = get_string(offsets_in, chars_in, tid);
    StringRef p = get_string(pat_offsets, pat_chars, 0);
    uint dir = *dir_ptr;
    if (p.len == 0 || s.len < p.len) {
        sizes[tid] = int64_t(s.len);
        return;
    }
    bool match = true;
    if (dir == 0) {
        for (uint i = 0; i < p.len; i++) {
            if (s.data[i] != p.data[i]) { match = false; break; }
        }
    } else {
        uint off = s.len - p.len;
        for (uint i = 0; i < p.len; i++) {
            if (s.data[off + i] != p.data[i]) { match = false; break; }
        }
    }
    sizes[tid] = match ? int64_t(s.len - p.len) : int64_t(s.len);
}

kernel void string_remove_affix_write(
    device const int64_t* offsets_in  [[buffer(0)]],
    device const uchar*   chars_in   [[buffer(1)]],
    device const int64_t* pat_offsets [[buffer(2)]],
    device const uchar*   pat_chars  [[buffer(3)]],
    device const int64_t* offsets_out [[buffer(4)]],
    device uchar*         chars_out  [[buffer(5)]],
    device const uint*    len_ptr    [[buffer(6)]],
    device const uint*    dir_ptr    [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    StringRef s = get_string(offsets_in, chars_in, tid);
    StringRef p = get_string(pat_offsets, pat_chars, 0);
    uint dir = *dir_ptr;
    int64_t out_start = offsets_out[tid];
    int64_t out_len = offsets_out[tid + 1] - out_start;

    if (out_len == int64_t(s.len)) {
        for (uint i = 0; i < s.len; i++) chars_out[out_start + i] = s.data[i];
    } else if (dir == 0) {
        for (int64_t i = 0; i < out_len; i++) chars_out[out_start + i] = s.data[p.len + i];
    } else {
        for (int64_t i = 0; i < out_len; i++) chars_out[out_start + i] = s.data[i];
    }
}
