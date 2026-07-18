#include <metal_stdlib>
using namespace metal;

// --- lower/upper: same-length output (1:1 byte mapping for ASCII) ---

kernel void string_lower(
    device const int64_t* offsets_in  [[buffer(0)]],
    device const uchar*   chars_in   [[buffer(1)]],
    device uchar*         chars_out  [[buffer(2)]],
    device const uint*    len_ptr    [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    int64_t start = offsets_in[tid];
    int64_t end = offsets_in[tid + 1];
    for (int64_t i = start; i < end; i++) {
        uchar c = chars_in[i];
        chars_out[i] = (c >= 0x41 && c <= 0x5A) ? (c + 32) : c;
    }
}

kernel void string_upper(
    device const int64_t* offsets_in  [[buffer(0)]],
    device const uchar*   chars_in   [[buffer(1)]],
    device uchar*         chars_out  [[buffer(2)]],
    device const uint*    len_ptr    [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    int64_t start = offsets_in[tid];
    int64_t end = offsets_in[tid + 1];
    for (int64_t i = start; i < end; i++) {
        uchar c = chars_in[i];
        chars_out[i] = (c >= 0x61 && c <= 0x7A) ? (c - 32) : c;
    }
}

// --- strip: variable-length output (two-pass) ---

// Pass 1: compute output size for each string
kernel void string_strip_sizes(
    device const int64_t* offsets_in  [[buffer(0)]],
    device const uchar*   chars_in   [[buffer(1)]],
    device int64_t*       sizes      [[buffer(2)]],
    device const uint*    len_ptr    [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    int64_t start = offsets_in[tid];
    int64_t end = offsets_in[tid + 1];
    int64_t s = start;
    int64_t e = end;
    while (s < e && (chars_in[s] == ' ' || chars_in[s] == '\t' || chars_in[s] == '\n' || chars_in[s] == '\r' || chars_in[s] == 0x0B || chars_in[s] == 0x0C)) s++;
    while (e > s && (chars_in[e-1] == ' ' || chars_in[e-1] == '\t' || chars_in[e-1] == '\n' || chars_in[e-1] == '\r' || chars_in[e-1] == 0x0B || chars_in[e-1] == 0x0C)) e--;
    sizes[tid] = e - s;
}

// Pass 2: write stripped chars to output
kernel void string_strip_write(
    device const int64_t* offsets_in  [[buffer(0)]],
    device const uchar*   chars_in   [[buffer(1)]],
    device const int64_t* offsets_out [[buffer(2)]],
    device uchar*         chars_out  [[buffer(3)]],
    device const uint*    len_ptr    [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    int64_t start = offsets_in[tid];
    int64_t end = offsets_in[tid + 1];
    int64_t s = start;
    int64_t e = end;
    while (s < e && (chars_in[s] == ' ' || chars_in[s] == '\t' || chars_in[s] == '\n' || chars_in[s] == '\r' || chars_in[s] == 0x0B || chars_in[s] == 0x0C)) s++;
    while (e > s && (chars_in[e-1] == ' ' || chars_in[e-1] == '\t' || chars_in[e-1] == '\n' || chars_in[e-1] == '\r' || chars_in[e-1] == 0x0B || chars_in[e-1] == 0x0C)) e--;
    int64_t out_start = offsets_out[tid];
    for (int64_t i = s; i < e; i++) {
        chars_out[out_start + (i - s)] = chars_in[i];
    }
}

// --- replace: variable-length output (two-pass) ---

// Pass 1: compute output size for each string after replacing all occurrences of pattern
kernel void string_replace_sizes(
    device const int64_t* offsets_in  [[buffer(0)]],
    device const uchar*   chars_in   [[buffer(1)]],
    device const int64_t* pat_offsets [[buffer(2)]],
    device const uchar*   pat_chars  [[buffer(3)]],
    device const int64_t* repl_offsets [[buffer(4)]],
    device const uchar*   repl_chars  [[buffer(5)]],
    device int64_t*       sizes      [[buffer(6)]],
    device const uint*    len_ptr    [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    StringRef s = get_string(offsets_in, chars_in, tid);
    StringRef p = get_string(pat_offsets, pat_chars, 0);
    StringRef r = get_string(repl_offsets, repl_chars, 0);
    if (p.len == 0) { sizes[tid] = int64_t(s.len); return; }
    int64_t out_len = 0;
    uint i = 0;
    // Use (i + p.len <= s.len) rather than (i <= s.len - p.len) to avoid unsigned
    // underflow when p.len > s.len (which would wrap s.len - p.len to a huge value
    // and cause an effectively infinite loop).
    while (i + p.len <= s.len) {
        bool match = true;
        for (uint j = 0; j < p.len; j++) {
            if (s.data[i + j] != p.data[j]) { match = false; break; }
        }
        if (match) {
            out_len += int64_t(r.len);
            i += p.len;
        } else {
            out_len++;
            i++;
        }
    }
    while (i < s.len) { out_len++; i++; }
    sizes[tid] = out_len;
}

// Pass 2: write replaced chars
kernel void string_replace_write(
    device const int64_t* offsets_in  [[buffer(0)]],
    device const uchar*   chars_in   [[buffer(1)]],
    device const int64_t* pat_offsets [[buffer(2)]],
    device const uchar*   pat_chars  [[buffer(3)]],
    device const int64_t* repl_offsets [[buffer(4)]],
    device const uchar*   repl_chars  [[buffer(5)]],
    device const int64_t* offsets_out [[buffer(6)]],
    device uchar*         chars_out  [[buffer(7)]],
    device const uint*    len_ptr    [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    StringRef s = get_string(offsets_in, chars_in, tid);
    StringRef p = get_string(pat_offsets, pat_chars, 0);
    StringRef r = get_string(repl_offsets, repl_chars, 0);
    int64_t out_pos = offsets_out[tid];
    if (p.len == 0) {
        for (uint i = 0; i < s.len; i++) chars_out[out_pos + i] = s.data[i];
        return;
    }
    uint i = 0;
    // Use (i + p.len <= s.len) rather than (i <= s.len - p.len) to avoid unsigned
    // underflow when p.len > s.len (which would wrap s.len - p.len to a huge value
    // and cause an effectively infinite loop).
    while (i + p.len <= s.len) {
        bool match = true;
        for (uint j = 0; j < p.len; j++) {
            if (s.data[i + j] != p.data[j]) { match = false; break; }
        }
        if (match) {
            for (uint j = 0; j < r.len; j++) chars_out[out_pos++] = r.data[j];
            i += p.len;
        } else {
            chars_out[out_pos++] = s.data[i++];
        }
    }
    while (i < s.len) chars_out[out_pos++] = s.data[i++];
}

// --- swapcase: same-length output (1:1 byte mapping for ASCII) ---

kernel void string_swapcase(
    device const int64_t* offsets_in  [[buffer(0)]],
    device const uchar*   chars_in   [[buffer(1)]],
    device uchar*         chars_out  [[buffer(2)]],
    device const uint*    len_ptr    [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    int64_t start = offsets_in[tid];
    int64_t end = offsets_in[tid + 1];
    for (int64_t i = start; i < end; i++) {
        uchar c = chars_in[i];
        if (c >= 0x41 && c <= 0x5A)      chars_out[i] = c + 32;
        else if (c >= 0x61 && c <= 0x7A) chars_out[i] = c - 32;
        else                              chars_out[i] = c;
    }
}

// --- title: same-length output, uppercase after word boundary ---

kernel void string_title(
    device const int64_t* offsets_in  [[buffer(0)]],
    device const uchar*   chars_in   [[buffer(1)]],
    device uchar*         chars_out  [[buffer(2)]],
    device const uint*    len_ptr    [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    int64_t start = offsets_in[tid];
    int64_t end = offsets_in[tid + 1];
    bool after_boundary = true;
    for (int64_t i = start; i < end; i++) {
        uchar c = chars_in[i];
        bool is_alpha = (c >= 0x41 && c <= 0x5A) || (c >= 0x61 && c <= 0x7A);
        if (is_alpha && after_boundary)
            chars_out[i] = (c >= 0x61) ? c - 32 : c;
        else if (is_alpha)
            chars_out[i] = (c <= 0x5A) ? c + 32 : c;
        else
            chars_out[i] = c;
        after_boundary = !is_alpha;
    }
}

// --- capitalize: uppercase first char, lowercase rest ---

kernel void string_capitalize(
    device const int64_t* offsets_in  [[buffer(0)]],
    device const uchar*   chars_in   [[buffer(1)]],
    device uchar*         chars_out  [[buffer(2)]],
    device const uint*    len_ptr    [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    int64_t start = offsets_in[tid];
    int64_t end = offsets_in[tid + 1];
    for (int64_t i = start; i < end; i++) {
        uchar c = chars_in[i];
        if (i == start) {
            chars_out[i] = (c >= 0x61 && c <= 0x7A) ? (c - 32) : c;
        } else {
            chars_out[i] = (c >= 0x41 && c <= 0x5A) ? (c + 32) : c;
        }
    }
}

// --- lstrip: variable-length output (two-pass), trim leading whitespace only ---

kernel void string_lstrip_sizes(
    device const int64_t* offsets_in  [[buffer(0)]],
    device const uchar*   chars_in   [[buffer(1)]],
    device int64_t*       sizes      [[buffer(2)]],
    device const uint*    len_ptr    [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    int64_t start = offsets_in[tid];
    int64_t end = offsets_in[tid + 1];
    int64_t s = start;
    while (s < end && (chars_in[s] == ' ' || chars_in[s] == '\t' || chars_in[s] == '\n' || chars_in[s] == '\r' || chars_in[s] == 0x0B || chars_in[s] == 0x0C)) s++;
    sizes[tid] = end - s;
}

kernel void string_lstrip_write(
    device const int64_t* offsets_in  [[buffer(0)]],
    device const uchar*   chars_in   [[buffer(1)]],
    device const int64_t* offsets_out [[buffer(2)]],
    device uchar*         chars_out  [[buffer(3)]],
    device const uint*    len_ptr    [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    int64_t start = offsets_in[tid];
    int64_t end = offsets_in[tid + 1];
    int64_t s = start;
    while (s < end && (chars_in[s] == ' ' || chars_in[s] == '\t' || chars_in[s] == '\n' || chars_in[s] == '\r' || chars_in[s] == 0x0B || chars_in[s] == 0x0C)) s++;
    int64_t out_start = offsets_out[tid];
    for (int64_t i = s; i < end; i++) {
        chars_out[out_start + (i - s)] = chars_in[i];
    }
}

// --- rstrip: variable-length output (two-pass), trim trailing whitespace only ---

kernel void string_rstrip_sizes(
    device const int64_t* offsets_in  [[buffer(0)]],
    device const uchar*   chars_in   [[buffer(1)]],
    device int64_t*       sizes      [[buffer(2)]],
    device const uint*    len_ptr    [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    int64_t start = offsets_in[tid];
    int64_t end = offsets_in[tid + 1];
    int64_t e = end;
    while (e > start && (chars_in[e-1] == ' ' || chars_in[e-1] == '\t' || chars_in[e-1] == '\n' || chars_in[e-1] == '\r' || chars_in[e-1] == 0x0B || chars_in[e-1] == 0x0C)) e--;
    sizes[tid] = e - start;
}

kernel void string_rstrip_write(
    device const int64_t* offsets_in  [[buffer(0)]],
    device const uchar*   chars_in   [[buffer(1)]],
    device const int64_t* offsets_out [[buffer(2)]],
    device uchar*         chars_out  [[buffer(3)]],
    device const uint*    len_ptr    [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    int64_t start = offsets_in[tid];
    int64_t end = offsets_in[tid + 1];
    int64_t e = end;
    while (e > start && (chars_in[e-1] == ' ' || chars_in[e-1] == '\t' || chars_in[e-1] == '\n' || chars_in[e-1] == '\r' || chars_in[e-1] == 0x0B || chars_in[e-1] == 0x0C)) e--;
    int64_t out_start = offsets_out[tid];
    for (int64_t i = start; i < e; i++) {
        chars_out[out_start + (i - start)] = chars_in[i];
    }
}
