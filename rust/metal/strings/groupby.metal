#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// String-key hash groupby — three-pass approach:
//   1. Build: insert string keys into hash table, assign group IDs
//   2. Accumulate: look up each key's group ID, atomically aggregate value
//   3. Compact: scan table, extract unique key indices + accumulated values
//
// Uses FNV-1a hash + string_equals for probing (linear probing).
//
// Sort-based fallback (>500K rows): bitonic sort on string indices, then a
// parallel leader-scan that walks sorted keys and accumulates per-group
// (same pattern as the numeric groupby_*_direct kernels in sort_agg.metal).
// ---------------------------------------------------------------------------

// Atomic add for float via CAS loop (Metal has no native atomic float add).
inline void string_groupby_atomic_add_float(device atomic_uint* addr, float val) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        float sum = as_type<float>(expected) + val;
        if (atomic_compare_exchange_weak_explicit(addr, &expected, as_type<uint>(sum),
            memory_order_relaxed, memory_order_relaxed)) break;
    }
}

// Atomic min for float via CAS loop.
inline void string_groupby_atomic_min_float(device atomic_uint* addr, float val) {
    uint val_bits = as_type<uint>(val);
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        float current = as_type<float>(expected);
        if (current <= val) break;
        if (atomic_compare_exchange_weak_explicit(addr, &expected, val_bits,
            memory_order_relaxed, memory_order_relaxed)) break;
    }
}

// Atomic max for float via CAS loop.
inline void string_groupby_atomic_max_float(device atomic_uint* addr, float val) {
    uint val_bits = as_type<uint>(val);
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        float current = as_type<float>(expected);
        if (current >= val) break;
        if (atomic_compare_exchange_weak_explicit(addr, &expected, val_bits,
            memory_order_relaxed, memory_order_relaxed)) break;
    }
}

// Hash probe helper: look up key at `tid` in the hash table, return group ID
// or 0xFFFFFFFF if tid is out of bounds.
inline uint string_hash_probe_gid(
    device const int64_t* key_offsets,
    device const uchar*   key_chars,
    device const uint*    table_hashes,
    device const uint*    table_gids,
    device const uint*    table_key_indices,
    uint tid, uint n, uint table_size
) {
    if (tid >= n) return 0xFFFFFFFF;
    StringRef key = get_string(key_offsets, key_chars, tid);
    uint hash = string_hash_fnv1a(key);
    if (hash == 0xFFFFFFFF) hash = 0xFFFFFFFE;
    uint mask = table_size - 1;
    uint slot = hash & mask;
    while (true) {
        uint h = table_hashes[slot];
        if (h == hash) {
            uint idx = table_key_indices[slot];
            StringRef existing = get_string(key_offsets, key_chars, idx);
            if (string_equals(key, existing)) {
                return table_gids[slot];
            }
        }
        slot = (slot + 1) & mask;
    }
}

// ===== Pass 1: Build hash table =====

kernel void string_groupby_hash_build(
    device const int64_t* key_offsets       [[buffer(0)]],
    device const uchar*   key_chars         [[buffer(1)]],
    device atomic_uint*   table_hashes      [[buffer(2)]],
    device atomic_uint*   table_gids        [[buffer(3)]],
    device atomic_uint*   table_key_indices [[buffer(4)]],
    device atomic_uint*   group_counter     [[buffer(5)]],
    device const uint*    len_ptr           [[buffer(6)]],
    device const uint*    ts_ptr            [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    uint table_size = *ts_ptr;
    uint mask = table_size - 1;
    if (tid >= n) return;

    StringRef key = get_string(key_offsets, key_chars, tid);
    uint hash = string_hash_fnv1a(key);
    if (hash == 0xFFFFFFFF) hash = 0xFFFFFFFE;
    uint slot = hash & mask;
    uint sentinel = 0xFFFFFFFF;

    while (true) {
        uint expected = sentinel;
        bool won = atomic_compare_exchange_weak_explicit(
            &table_hashes[slot], &expected, hash,
            memory_order_relaxed, memory_order_relaxed);
        if (won) {
            uint gid = atomic_fetch_add_explicit(group_counter, 1, memory_order_relaxed);
            atomic_store_explicit(&table_gids[slot], gid, memory_order_relaxed);
            atomic_store_explicit(&table_key_indices[slot], tid, memory_order_relaxed);
            return;
        }
        if (expected == hash) {
            uint existing_idx;
            do {
                existing_idx = atomic_load_explicit(&table_key_indices[slot], memory_order_relaxed);
            } while (existing_idx == sentinel);
            StringRef existing = get_string(key_offsets, key_chars, existing_idx);
            if (string_equals(key, existing)) return;
        }
        slot = (slot + 1) & mask;
    }
}

// ===== Pass 2: Accumulate — sum =====

kernel void string_groupby_hash_sum_float32(
    device const int64_t* key_offsets       [[buffer(0)]],
    device const uchar*   key_chars         [[buffer(1)]],
    device const float*   values            [[buffer(2)]],
    device const uint*    table_hashes      [[buffer(3)]],
    device const uint*    table_gids        [[buffer(4)]],
    device const uint*    table_key_indices [[buffer(5)]],
    device atomic_uint*   accum             [[buffer(6)]],
    device const uint*    len_ptr           [[buffer(7)]],
    device const uint*    ts_ptr            [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    uint gid = string_hash_probe_gid(key_offsets, key_chars, table_hashes,
        table_gids, table_key_indices, tid, *len_ptr, *ts_ptr);
    if (gid == 0xFFFFFFFF) return;
    string_groupby_atomic_add_float(&accum[gid], values[tid]);
}

kernel void string_groupby_hash_sum_int32(
    device const int64_t* key_offsets       [[buffer(0)]],
    device const uchar*   key_chars         [[buffer(1)]],
    device const int*     values            [[buffer(2)]],
    device const uint*    table_hashes      [[buffer(3)]],
    device const uint*    table_gids        [[buffer(4)]],
    device const uint*    table_key_indices [[buffer(5)]],
    device atomic_int*    accum             [[buffer(6)]],
    device const uint*    len_ptr           [[buffer(7)]],
    device const uint*    ts_ptr            [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    uint gid = string_hash_probe_gid(key_offsets, key_chars, table_hashes,
        table_gids, table_key_indices, tid, *len_ptr, *ts_ptr);
    if (gid == 0xFFFFFFFF) return;
    atomic_fetch_add_explicit(&accum[gid], values[tid], memory_order_relaxed);
}

// ===== Pass 2: Accumulate — min =====

kernel void string_groupby_hash_min_float32(
    device const int64_t* key_offsets       [[buffer(0)]],
    device const uchar*   key_chars         [[buffer(1)]],
    device const float*   values            [[buffer(2)]],
    device const uint*    table_hashes      [[buffer(3)]],
    device const uint*    table_gids        [[buffer(4)]],
    device const uint*    table_key_indices [[buffer(5)]],
    device atomic_uint*   accum             [[buffer(6)]],
    device const uint*    len_ptr           [[buffer(7)]],
    device const uint*    ts_ptr            [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    uint gid = string_hash_probe_gid(key_offsets, key_chars, table_hashes,
        table_gids, table_key_indices, tid, *len_ptr, *ts_ptr);
    if (gid == 0xFFFFFFFF) return;
    string_groupby_atomic_min_float(&accum[gid], values[tid]);
}

kernel void string_groupby_hash_min_int32(
    device const int64_t* key_offsets       [[buffer(0)]],
    device const uchar*   key_chars         [[buffer(1)]],
    device const int*     values            [[buffer(2)]],
    device const uint*    table_hashes      [[buffer(3)]],
    device const uint*    table_gids        [[buffer(4)]],
    device const uint*    table_key_indices [[buffer(5)]],
    device atomic_int*    accum             [[buffer(6)]],
    device const uint*    len_ptr           [[buffer(7)]],
    device const uint*    ts_ptr            [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    uint gid = string_hash_probe_gid(key_offsets, key_chars, table_hashes,
        table_gids, table_key_indices, tid, *len_ptr, *ts_ptr);
    if (gid == 0xFFFFFFFF) return;
    atomic_fetch_min_explicit(&accum[gid], values[tid], memory_order_relaxed);
}

// ===== Pass 2: Accumulate — max =====

kernel void string_groupby_hash_max_float32(
    device const int64_t* key_offsets       [[buffer(0)]],
    device const uchar*   key_chars         [[buffer(1)]],
    device const float*   values            [[buffer(2)]],
    device const uint*    table_hashes      [[buffer(3)]],
    device const uint*    table_gids        [[buffer(4)]],
    device const uint*    table_key_indices [[buffer(5)]],
    device atomic_uint*   accum             [[buffer(6)]],
    device const uint*    len_ptr           [[buffer(7)]],
    device const uint*    ts_ptr            [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    uint gid = string_hash_probe_gid(key_offsets, key_chars, table_hashes,
        table_gids, table_key_indices, tid, *len_ptr, *ts_ptr);
    if (gid == 0xFFFFFFFF) return;
    string_groupby_atomic_max_float(&accum[gid], values[tid]);
}

kernel void string_groupby_hash_max_int32(
    device const int64_t* key_offsets       [[buffer(0)]],
    device const uchar*   key_chars         [[buffer(1)]],
    device const int*     values            [[buffer(2)]],
    device const uint*    table_hashes      [[buffer(3)]],
    device const uint*    table_gids        [[buffer(4)]],
    device const uint*    table_key_indices [[buffer(5)]],
    device atomic_int*    accum             [[buffer(6)]],
    device const uint*    len_ptr           [[buffer(7)]],
    device const uint*    ts_ptr            [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    uint gid = string_hash_probe_gid(key_offsets, key_chars, table_hashes,
        table_gids, table_key_indices, tid, *len_ptr, *ts_ptr);
    if (gid == 0xFFFFFFFF) return;
    atomic_fetch_max_explicit(&accum[gid], values[tid], memory_order_relaxed);
}

// ===== Pass 2: Accumulate — count =====

kernel void string_groupby_hash_count(
    device const int64_t* key_offsets       [[buffer(0)]],
    device const uchar*   key_chars         [[buffer(1)]],
    device const uint*    table_hashes      [[buffer(2)]],
    device const uint*    table_gids        [[buffer(3)]],
    device const uint*    table_key_indices [[buffer(4)]],
    device atomic_uint*   accum             [[buffer(5)]],
    device const uint*    len_ptr           [[buffer(6)]],
    device const uint*    ts_ptr            [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    uint gid = string_hash_probe_gid(key_offsets, key_chars, table_hashes,
        table_gids, table_key_indices, tid, *len_ptr, *ts_ptr);
    if (gid == 0xFFFFFFFF) return;
    atomic_fetch_add_explicit(&accum[gid], 1u, memory_order_relaxed);
}

// ===== Pass 3: Compact — sum =====

kernel void string_groupby_hash_compact_sum_float32(
    device const uint*  table_hashes       [[buffer(0)]],
    device const uint*  table_gids         [[buffer(1)]],
    device const uint*  table_key_indices  [[buffer(2)]],
    device const uint*  accum              [[buffer(3)]],
    device uint*        out_key_indices    [[buffer(4)]],
    device float*       out_values         [[buffer(5)]],
    device const uint*  ts_ptr             [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint table_size = *ts_ptr;
    if (tid >= table_size) return;
    if (table_hashes[tid] == 0xFFFFFFFF) return;
    uint gid = table_gids[tid];
    out_key_indices[gid] = table_key_indices[tid];
    out_values[gid] = as_type<float>(accum[gid]);
}

kernel void string_groupby_hash_compact_sum_int32(
    device const uint*  table_hashes       [[buffer(0)]],
    device const uint*  table_gids         [[buffer(1)]],
    device const uint*  table_key_indices  [[buffer(2)]],
    device const int*   accum              [[buffer(3)]],
    device uint*        out_key_indices    [[buffer(4)]],
    device int*         out_values         [[buffer(5)]],
    device const uint*  ts_ptr             [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint table_size = *ts_ptr;
    if (tid >= table_size) return;
    if (table_hashes[tid] == 0xFFFFFFFF) return;
    uint gid = table_gids[tid];
    out_key_indices[gid] = table_key_indices[tid];
    out_values[gid] = accum[gid];
}

// ===== Pass 3: Compact — minmax =====
// Float: accum stores float bits as uint (via CAS), reinterpret on read.
// Int: accum is plain int, copy directly.

kernel void string_groupby_hash_compact_minmax_float32(
    device const uint*  table_hashes       [[buffer(0)]],
    device const uint*  table_gids         [[buffer(1)]],
    device const uint*  table_key_indices  [[buffer(2)]],
    device const uint*  accum              [[buffer(3)]],
    device uint*        out_key_indices    [[buffer(4)]],
    device float*       out_values         [[buffer(5)]],
    device const uint*  ts_ptr             [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint table_size = *ts_ptr;
    if (tid >= table_size) return;
    if (table_hashes[tid] == 0xFFFFFFFF) return;
    uint gid = table_gids[tid];
    out_key_indices[gid] = table_key_indices[tid];
    out_values[gid] = as_type<float>(accum[gid]);
}

kernel void string_groupby_hash_compact_minmax_int32(
    device const uint*  table_hashes       [[buffer(0)]],
    device const uint*  table_gids         [[buffer(1)]],
    device const uint*  table_key_indices  [[buffer(2)]],
    device const int*   accum              [[buffer(3)]],
    device uint*        out_key_indices    [[buffer(4)]],
    device int*         out_values         [[buffer(5)]],
    device const uint*  ts_ptr             [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint table_size = *ts_ptr;
    if (tid >= table_size) return;
    if (table_hashes[tid] == 0xFFFFFFFF) return;
    uint gid = table_gids[tid];
    out_key_indices[gid] = table_key_indices[tid];
    out_values[gid] = accum[gid];
}

// ===== Pass 3: Compact — count (uint accum -> float32 output) =====

kernel void string_groupby_hash_compact_count(
    device const uint*  table_hashes       [[buffer(0)]],
    device const uint*  table_gids         [[buffer(1)]],
    device const uint*  table_key_indices  [[buffer(2)]],
    device const uint*  accum              [[buffer(3)]],
    device uint*        out_key_indices    [[buffer(4)]],
    device float*       out_values         [[buffer(5)]],
    device const uint*  ts_ptr             [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint table_size = *ts_ptr;
    if (tid >= table_size) return;
    if (table_hashes[tid] == 0xFFFFFFFF) return;
    uint gid = table_gids[tid];
    out_key_indices[gid] = table_key_indices[tid];
    out_values[gid] = float(accum[gid]);
}

// ===================================================================
// Sort-based direct reduction — parallel leader-scan on sorted string
// keys (via sorted index array). Each thread checks if it starts a new
// group, then scans forward and reduces the segment. Same pattern as
// the numeric groupby_*_direct kernels in sort_agg.metal.
// ===================================================================

// --- Helper: check if gid is a group leader (new string key) ---
inline bool string_is_leader(
    device const uint* sorted_indices,
    device const int64_t* key_offsets,
    device const uchar* key_chars,
    uint gid, uint n
) {
    if (gid >= n) return false;
    if (gid == 0) return true;
    StringRef prev = get_string(key_offsets, key_chars, sorted_indices[gid - 1]);
    StringRef curr = get_string(key_offsets, key_chars, sorted_indices[gid]);
    return !string_equals(prev, curr);
}

// --- Helper: find the end of the current group ---
inline uint string_group_end(
    device const uint* sorted_indices,
    device const int64_t* key_offsets,
    device const uchar* key_chars,
    uint gid, uint n
) {
    StringRef key = get_string(key_offsets, key_chars, sorted_indices[gid]);
    uint end = gid + 1;
    while (end < n) {
        StringRef next = get_string(key_offsets, key_chars, sorted_indices[end]);
        if (!string_equals(key, next)) break;
        end++;
    }
    return end;
}

// --- Aggregation direct (sum/min/max) ---
template <typename T, typename Op>
void string_groupby_agg_direct_impl(
    device const uint*    sorted_indices,
    device const int64_t* key_offsets,
    device const uchar*   key_chars,
    device const T*       sorted_values,
    device uint*          out_key_indices,
    device T*             out_values,
    device atomic_uint*   group_counter,
    uint gid, uint n
) {
    if (!string_is_leader(sorted_indices, key_offsets, key_chars, gid, n)) return;

    uint end = string_group_end(sorted_indices, key_offsets, key_chars, gid, n);

    T acc = Op::identity;
    for (uint i = gid; i < end; i++) acc = Op::apply(acc, sorted_values[i]);

    uint g = atomic_fetch_add_explicit(group_counter, 1u, memory_order_relaxed);
    out_key_indices[g] = sorted_indices[gid];
    out_values[g] = acc;
}

#define INSTANTIATE_STRING_GROUPBY_AGG_DIRECT(T, suffix, Op, opname) \
[[kernel]] void string_groupby_##opname##_direct_##suffix( \
    device const uint*    sorted_indices [[buffer(0)]], \
    device const int64_t* key_offsets    [[buffer(1)]], \
    device const uchar*   key_chars     [[buffer(2)]], \
    device const T*       sorted_values [[buffer(3)]], \
    device uint*          out_key_indices [[buffer(4)]], \
    device T*             out_values    [[buffer(5)]], \
    device atomic_uint*   group_counter [[buffer(6)]], \
    device const uint*    len_ptr       [[buffer(7)]], \
    uint gid [[thread_position_in_grid]] \
) { string_groupby_agg_direct_impl<T, Op<T>>(sorted_indices, key_offsets, key_chars, \
    sorted_values, out_key_indices, out_values, group_counter, gid, *len_ptr); }

INSTANTIATE_STRING_GROUPBY_AGG_DIRECT(float, float32, SumOp, sum)
INSTANTIATE_STRING_GROUPBY_AGG_DIRECT(int,   int32,   SumOp, sum)
INSTANTIATE_STRING_GROUPBY_AGG_DIRECT(float, float32, MinOp, min)
INSTANTIATE_STRING_GROUPBY_AGG_DIRECT(int,   int32,   MinOp, min)
INSTANTIATE_STRING_GROUPBY_AGG_DIRECT(float, float32, MaxOp, max)
INSTANTIATE_STRING_GROUPBY_AGG_DIRECT(int,   int32,   MaxOp, max)

// --- Count direct ---
void string_groupby_count_direct_impl(
    device const uint*    sorted_indices,
    device const int64_t* key_offsets,
    device const uchar*   key_chars,
    device uint*          out_key_indices,
    device float*         out_values,
    device atomic_uint*   group_counter,
    uint gid, uint n
) {
    if (!string_is_leader(sorted_indices, key_offsets, key_chars, gid, n)) return;

    uint end = string_group_end(sorted_indices, key_offsets, key_chars, gid, n);

    uint g = atomic_fetch_add_explicit(group_counter, 1u, memory_order_relaxed);
    out_key_indices[g] = sorted_indices[gid];
    out_values[g] = float(end - gid);
}

[[kernel]] void string_groupby_count_direct(
    device const uint*    sorted_indices [[buffer(0)]],
    device const int64_t* key_offsets    [[buffer(1)]],
    device const uchar*   key_chars     [[buffer(2)]],
    device uint*          out_key_indices [[buffer(3)]],
    device float*         out_values    [[buffer(4)]],
    device atomic_uint*   group_counter [[buffer(5)]],
    device const uint*    len_ptr       [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) { string_groupby_count_direct_impl(sorted_indices, key_offsets, key_chars,
    out_key_indices, out_values, group_counter, gid, *len_ptr); }

// --- Fused sum+count direct (for mean) ---
template <typename T>
void string_groupby_sum_count_direct_impl(
    device const uint*    sorted_indices,
    device const int64_t* key_offsets,
    device const uchar*   key_chars,
    device const T*       sorted_values,
    device uint*          out_key_indices,
    device T*             out_sums,
    device float*         out_counts,
    device atomic_uint*   group_counter,
    uint gid, uint n
) {
    if (!string_is_leader(sorted_indices, key_offsets, key_chars, gid, n)) return;

    uint end = string_group_end(sorted_indices, key_offsets, key_chars, gid, n);

    T sum = SumOp<T>::identity;
    for (uint i = gid; i < end; i++) sum = SumOp<T>::apply(sum, sorted_values[i]);

    uint g = atomic_fetch_add_explicit(group_counter, 1u, memory_order_relaxed);
    out_key_indices[g] = sorted_indices[gid];
    out_sums[g] = sum;
    out_counts[g] = float(end - gid);
}

#define INSTANTIATE_STRING_GROUPBY_SUM_COUNT_DIRECT(T, suffix) \
[[kernel]] void string_groupby_sum_count_direct_##suffix( \
    device const uint*    sorted_indices [[buffer(0)]], \
    device const int64_t* key_offsets    [[buffer(1)]], \
    device const uchar*   key_chars     [[buffer(2)]], \
    device const T*       sorted_values [[buffer(3)]], \
    device uint*          out_key_indices [[buffer(4)]], \
    device T*             out_sums      [[buffer(5)]], \
    device float*         out_counts    [[buffer(6)]], \
    device atomic_uint*   group_counter [[buffer(7)]], \
    device const uint*    len_ptr       [[buffer(8)]], \
    uint gid [[thread_position_in_grid]] \
) { string_groupby_sum_count_direct_impl<T>(sorted_indices, key_offsets, key_chars, \
    sorted_values, out_key_indices, out_sums, out_counts, group_counter, gid, *len_ptr); }

INSTANTIATE_STRING_GROUPBY_SUM_COUNT_DIRECT(float, float32)
INSTANTIATE_STRING_GROUPBY_SUM_COUNT_DIRECT(int,   int32)
