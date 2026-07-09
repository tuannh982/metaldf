// Hash-based GroupBy — linear probing hash table with atomic CAS.
// 3 passes batched into a single command buffer:
//   1. Build: insert keys, assign group_ids
//   2. Accumulate: sum/min/max/count per group
//   3. Compact: extract dense (key, value) pairs
// Only float32/int32 supported (32-bit atomics).

// --- Hash probe: find the slot for a key in the hash table ---
template <typename T>
uint hash_probe(
    device const uint* table_keys,
    uint array_len, uint table_size,
    device const T* keys, uint gid
) {
    if (gid >= array_len) return table_size; // sentinel: out of bounds
    using K = typename KeyBits<T>::type;
    K key_bits = RadixTraits<T>::to_key(keys[gid]);
    uint hash = uint(key_bits) & (table_size - 1);
    for (uint probe = 0; probe < table_size; probe++) {
        uint slot = (hash + probe) & (table_size - 1);
        if (table_keys[slot] == key_bits) return slot;
    }
    return table_size;
}

// --- Build ---
template <typename T>
void groupby_hash_build_impl(
    device const T* keys,
    device atomic_uint* table_keys,
    device uint* table_gids,
    device atomic_uint* group_counter,
    uint gid, uint array_len, uint table_size
) {
    if (gid >= array_len) return;
    using K = typename KeyBits<T>::type;
    K key_bits = RadixTraits<T>::to_key(keys[gid]);
    K sentinel = HashSentinel<K>::value;
    uint hash = uint(key_bits) & (table_size - 1);

    for (uint probe = 0; probe < table_size; probe++) {
        uint slot = (hash + probe) & (table_size - 1);
        K existing = atomic_load_explicit(&table_keys[slot], memory_order_relaxed);

        if (existing == key_bits) return;
        if (existing == sentinel) {
            K expected = sentinel;
            bool claimed = atomic_compare_exchange_weak_explicit(
                &table_keys[slot], &expected, key_bits,
                memory_order_relaxed, memory_order_relaxed);
            if (claimed) {
                uint g = atomic_fetch_add_explicit(group_counter, 1u, memory_order_relaxed);
                table_gids[slot] = g;
                return;
            }
            existing = atomic_load_explicit(&table_keys[slot], memory_order_relaxed);
            if (existing == key_bits) return;
        }
    }
}

// --- Accumulate (all ops use hash_probe to find the slot) ---
template <typename T>
void groupby_hash_sum_impl(
    device const T* keys, device const T* values,
    device const uint* table_keys, device const uint* table_gids,
    device typename SumAccum<T>::AtomicT* out,
    uint gid, uint array_len, uint table_size
) {
    uint slot = hash_probe(table_keys, array_len, table_size, keys, gid);
    if (slot < table_size) SumAccum<T>::accumulate(&out[table_gids[slot]], values[gid]);
}

template <typename T>
void groupby_hash_min_impl(
    device const T* keys, device const T* values,
    device const uint* table_keys, device const uint* table_gids,
    device typename MinAccum<T>::AtomicT* out,
    uint gid, uint array_len, uint table_size
) {
    uint slot = hash_probe(table_keys, array_len, table_size, keys, gid);
    if (slot < table_size) MinAccum<T>::accumulate(&out[table_gids[slot]], values[gid]);
}

template <typename T>
void groupby_hash_max_impl(
    device const T* keys, device const T* values,
    device const uint* table_keys, device const uint* table_gids,
    device typename MaxAccum<T>::AtomicT* out,
    uint gid, uint array_len, uint table_size
) {
    uint slot = hash_probe(table_keys, array_len, table_size, keys, gid);
    if (slot < table_size) MaxAccum<T>::accumulate(&out[table_gids[slot]], values[gid]);
}

template <typename T>
void groupby_hash_count_impl(
    device const T* keys,
    device const uint* table_keys, device const uint* table_gids,
    device atomic_uint* out,
    uint gid, uint array_len, uint table_size
) {
    uint slot = hash_probe(table_keys, array_len, table_size, keys, gid);
    if (slot < table_size) atomic_fetch_add_explicit(&out[table_gids[slot]], 1u, memory_order_relaxed);
}

// --- Compact: extract dense output from sparse hash table ---
template <typename T>
void groupby_hash_compact_impl(
    device const uint* table_keys, device const uint* table_gids,
    device T* out_keys,
    uint gid, uint table_size
) {
    if (gid >= table_size) return;
    uint key_bits = table_keys[gid];
    if (key_bits == HashSentinel<uint>::value) return;
    out_keys[table_gids[gid]] = decode_hash_key<T>(key_bits);
}

template <typename T, typename RawT>
void groupby_hash_compact_vals_impl(
    device const uint* table_keys, device const uint* table_gids,
    device const RawT* raw_vals, device T* out_keys, device T* out_vals,
    uint gid, uint table_size
) {
    if (gid >= table_size) return;
    uint key_bits = table_keys[gid];
    if (key_bits == HashSentinel<uint>::value) return;
    uint g = table_gids[gid];
    out_keys[g] = decode_hash_key<T>(key_bits);
    out_vals[g] = raw_vals[g];
}

template <typename T, typename RawT>
void groupby_hash_compact_minmax_impl(
    device const uint* table_keys, device const uint* table_gids,
    device const RawT* raw_vals, device T* out_keys, device T* out_vals,
    uint gid, uint table_size
) {
    if (gid >= table_size) return;
    uint key_bits = table_keys[gid];
    if (key_bits == HashSentinel<uint>::value) return;
    uint g = table_gids[gid];
    out_keys[g] = decode_hash_key<T>(key_bits);
    out_vals[g] = MinMaxStorage<T>::load(raw_vals[g]);
}

template <typename T>
void groupby_hash_compact_count_impl(
    device const uint* table_keys, device const uint* table_gids,
    device const uint* raw_vals, device T* out_keys, device long* out_vals,
    uint gid, uint table_size
) {
    if (gid >= table_size) return;
    uint key_bits = table_keys[gid];
    if (key_bits == HashSentinel<uint>::value) return;
    uint g = table_gids[gid];
    out_keys[g] = decode_hash_key<T>(key_bits);
    out_vals[g] = long(raw_vals[g]);
}

// --- Kernel instantiation ---

#define INSTANTIATE_HASH_BUILD(T, suffix) \
    [[kernel]] void groupby_hash_build_##suffix( \
        device const T* keys [[buffer(0)]], device atomic_uint* table_keys [[buffer(1)]], \
        device uint* table_gids [[buffer(2)]], device atomic_uint* group_counter [[buffer(3)]], \
        uint gid [[thread_position_in_grid]], \
        device const uint* len_ptr [[buffer(4)]], device const uint* ts_ptr [[buffer(5)]] \
    ) { groupby_hash_build_impl<T>(keys, table_keys, table_gids, group_counter, gid, *len_ptr, *ts_ptr); }

#define INSTANTIATE_HASH_ACCUM(T, suffix, SumAtomicT, MinMaxAtomicT) \
    [[kernel]] void groupby_hash_sum_##suffix( \
        device const T* keys [[buffer(0)]], device const T* values [[buffer(1)]], \
        device const uint* tk [[buffer(2)]], device const uint* tg [[buffer(3)]], \
        device SumAtomicT* out [[buffer(4)]], uint gid [[thread_position_in_grid]], \
        device const uint* len_ptr [[buffer(5)]], device const uint* ts_ptr [[buffer(6)]] \
    ) { groupby_hash_sum_impl<T>(keys, values, tk, tg, out, gid, *len_ptr, *ts_ptr); } \
    [[kernel]] void groupby_hash_min_##suffix( \
        device const T* keys [[buffer(0)]], device const T* values [[buffer(1)]], \
        device const uint* tk [[buffer(2)]], device const uint* tg [[buffer(3)]], \
        device MinMaxAtomicT* out [[buffer(4)]], uint gid [[thread_position_in_grid]], \
        device const uint* len_ptr [[buffer(5)]], device const uint* ts_ptr [[buffer(6)]] \
    ) { groupby_hash_min_impl<T>(keys, values, tk, tg, out, gid, *len_ptr, *ts_ptr); } \
    [[kernel]] void groupby_hash_max_##suffix( \
        device const T* keys [[buffer(0)]], device const T* values [[buffer(1)]], \
        device const uint* tk [[buffer(2)]], device const uint* tg [[buffer(3)]], \
        device MinMaxAtomicT* out [[buffer(4)]], uint gid [[thread_position_in_grid]], \
        device const uint* len_ptr [[buffer(5)]], device const uint* ts_ptr [[buffer(6)]] \
    ) { groupby_hash_max_impl<T>(keys, values, tk, tg, out, gid, *len_ptr, *ts_ptr); } \
    [[kernel]] void groupby_hash_count_##suffix( \
        device const T* keys [[buffer(0)]], device const uint* tk [[buffer(1)]], \
        device const uint* tg [[buffer(2)]], device atomic_uint* out [[buffer(3)]], \
        uint gid [[thread_position_in_grid]], \
        device const uint* len_ptr [[buffer(4)]], device const uint* ts_ptr [[buffer(5)]] \
    ) { groupby_hash_count_impl<T>(keys, tk, tg, out, gid, *len_ptr, *ts_ptr); }

#define INSTANTIATE_HASH_COMPACT(T, suffix, MinMaxRawT) \
    [[kernel]] void groupby_hash_compact_sum_##suffix( \
        device const uint* tk [[buffer(0)]], device const uint* tg [[buffer(1)]], \
        device const T* raw [[buffer(2)]], device T* ok [[buffer(3)]], device T* ov [[buffer(4)]], \
        uint gid [[thread_position_in_grid]], device const uint* ts_ptr [[buffer(5)]] \
    ) { groupby_hash_compact_vals_impl<T, T>(tk, tg, raw, ok, ov, gid, *ts_ptr); } \
    [[kernel]] void groupby_hash_compact_minmax_##suffix( \
        device const uint* tk [[buffer(0)]], device const uint* tg [[buffer(1)]], \
        device const MinMaxRawT* raw [[buffer(2)]], device T* ok [[buffer(3)]], device T* ov [[buffer(4)]], \
        uint gid [[thread_position_in_grid]], device const uint* ts_ptr [[buffer(5)]] \
    ) { groupby_hash_compact_minmax_impl<T, MinMaxRawT>(tk, tg, raw, ok, ov, gid, *ts_ptr); } \
    [[kernel]] void groupby_hash_compact_count_##suffix( \
        device const uint* tk [[buffer(0)]], device const uint* tg [[buffer(1)]], \
        device const uint* raw [[buffer(2)]], device T* ok [[buffer(3)]], device long* ov [[buffer(4)]], \
        uint gid [[thread_position_in_grid]], device const uint* ts_ptr [[buffer(5)]] \
    ) { groupby_hash_compact_count_impl<T>(tk, tg, raw, ok, ov, gid, *ts_ptr); }

INSTANTIATE_HASH_BUILD(float, float32)
INSTANTIATE_HASH_BUILD(int,   int32)

INSTANTIATE_HASH_ACCUM(float, float32, atomic_float, atomic_uint)
INSTANTIATE_HASH_ACCUM(int,   int32,   atomic_int,   atomic_int)

INSTANTIATE_HASH_COMPACT(float, float32, uint)
INSTANTIATE_HASH_COMPACT(int,   int32,   int)
