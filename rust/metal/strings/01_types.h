#pragma once
#include <metal_stdlib>
using namespace metal;

struct StringRef {
    device const uchar* data;
    uint len;
};

inline StringRef get_string(device const int64_t* offsets,
                            device const uchar* chars,
                            uint idx) {
    int64_t start = offsets[idx];
    return { chars + start, uint(offsets[idx + 1] - start) };
}

inline int string_compare(StringRef a, StringRef b) {
    uint min_len = min(a.len, b.len);
    for (uint i = 0; i < min_len; i++) {
        if (a.data[i] != b.data[i])
            return (a.data[i] < b.data[i]) ? -1 : 1;
    }
    if (a.len != b.len) return (a.len < b.len) ? -1 : 1;
    return 0;
}

inline bool string_equals(StringRef a, StringRef b) {
    if (a.len != b.len) return false;
    for (uint i = 0; i < a.len; i++) {
        if (a.data[i] != b.data[i]) return false;
    }
    return true;
}

inline uint string_hash_fnv1a(StringRef s) {
    uint hash = 2166136261u;
    for (uint i = 0; i < s.len; i++) {
        hash ^= uint(s.data[i]);
        hash *= 16777619u;
    }
    return hash;
}
