// Hash join kernels — GPU equi-join via linear-probing hash table.
//
// Three kernel passes:
//   1. join_build_{suffix}:       Insert build-table keys into hash table
//   2. join_probe_count_{suffix}: Count matches per probe row
//   3. join_probe_write_{suffix}: Write (left_idx, right_idx) pairs
//
// Between passes 2 and 3, the Rust dispatch runs a prefix-sum on the
// count buffer to compute write offsets and total output size.
//
// Only float32/int32 supported (32-bit atomics). Uses RadixTraits::to_key
// for key encoding (same bijection as groupby) so the sentinel value
// 0xFFFFFFFF can never collide with a real key.
//
// Unlike groupby's hash build, join build does NOT deduplicate: each build
// row occupies its own slot, so duplicate keys get multiple slots. This
// allows many-to-many joins.

// ---------------------------------------------------------------------------
// Build: insert build-table keys into hash table via atomic CAS.
// Each build row gets its own slot (no deduplication).
// ---------------------------------------------------------------------------

template <typename T>
void join_build_impl(
    device const T* keys,
    device atomic_uint* table_keys,   // table_size slots, init to SENTINEL
    device uint* table_rows,          // table_size slots (row index storage)
    uint gid, uint build_len, uint table_size
) {
    if (gid >= build_len) return;

    using KeyT = typename RadixTraits<T>::KeyT;
    KeyT key_bits = RadixTraits<T>::to_key(keys[gid]);
    uint hash = uint(key_bits) & (table_size - 1);

    for (uint p = 0; p < table_size; p++) {
        uint slot = (hash + p) & (table_size - 1);
        uint existing = atomic_load_explicit(&table_keys[slot], memory_order_relaxed);

        if (existing == JOIN_EMPTY_SENTINEL) {
            uint expected = JOIN_EMPTY_SENTINEL;
            bool claimed = atomic_compare_exchange_weak_explicit(
                &table_keys[slot], &expected, uint(key_bits),
                memory_order_relaxed, memory_order_relaxed);
            if (claimed) {
                table_rows[slot] = gid;
                return;
            }
            // CAS failed — another thread claimed this slot. Continue probing.
        }
        // Slot occupied (by this key or another). Continue linear probing.
    }
}

// ---------------------------------------------------------------------------
// Probe — pass 1: count matches per probe row
// ---------------------------------------------------------------------------

template <typename T>
void join_probe_count_impl(
    device const T* probe_keys,
    device const uint* table_keys,    // non-atomic read (build is done)
    device const uint* table_rows,
    device uint* counts,              // one uint32 per probe row
    uint gid, uint probe_len, uint table_size
) {
    if (gid >= probe_len) return;

    using KeyT = typename RadixTraits<T>::KeyT;
    KeyT key_bits = RadixTraits<T>::to_key(probe_keys[gid]);
    uint hash = uint(key_bits) & (table_size - 1);
    uint count = 0;

    for (uint p = 0; p < table_size; p++) {
        uint slot = (hash + p) & (table_size - 1);
        uint slot_key = table_keys[slot];
        if (slot_key == JOIN_EMPTY_SENTINEL) break;
        if (slot_key == uint(key_bits)) {
            count++;
        }
    }

    counts[gid] = count;
}

// ---------------------------------------------------------------------------
// Probe — pass 2: write (left_idx, right_idx) pairs at pre-computed offsets
// ---------------------------------------------------------------------------

template <typename T>
void join_probe_write_impl(
    device const T* probe_keys,
    device const uint* table_keys,
    device const uint* table_rows,
    device const uint* offsets,        // exclusive prefix (offset to write at)
    device uint* left_indices,         // output: build-table row indices
    device uint* right_indices,        // output: probe-table row indices
    uint gid, uint probe_len, uint table_size
) {
    if (gid >= probe_len) return;

    using KeyT = typename RadixTraits<T>::KeyT;
    KeyT key_bits = RadixTraits<T>::to_key(probe_keys[gid]);
    uint hash = uint(key_bits) & (table_size - 1);
    uint write_pos = offsets[gid];

    for (uint p = 0; p < table_size; p++) {
        uint slot = (hash + p) & (table_size - 1);
        uint slot_key = table_keys[slot];
        if (slot_key == JOIN_EMPTY_SENTINEL) break;
        if (slot_key == uint(key_bits)) {
            left_indices[write_pos] = table_rows[slot];
            right_indices[write_pos] = gid;
            write_pos++;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel instantiation
// ---------------------------------------------------------------------------

#define INSTANTIATE_JOIN(T, suffix) \
    [[kernel]] void join_build_##suffix( \
        device const T* keys [[buffer(0)]], \
        device atomic_uint* table_keys [[buffer(1)]], \
        device uint* table_rows [[buffer(2)]], \
        uint gid [[thread_position_in_grid]], \
        device const uint* build_len_ptr [[buffer(3)]], \
        device const uint* table_size_ptr [[buffer(4)]] \
    ) { join_build_impl<T>(keys, table_keys, table_rows, gid, *build_len_ptr, *table_size_ptr); } \
    \
    [[kernel]] void join_probe_count_##suffix( \
        device const T* probe_keys [[buffer(0)]], \
        device const uint* table_keys [[buffer(1)]], \
        device const uint* table_rows [[buffer(2)]], \
        device uint* counts [[buffer(3)]], \
        uint gid [[thread_position_in_grid]], \
        device const uint* probe_len_ptr [[buffer(4)]], \
        device const uint* table_size_ptr [[buffer(5)]] \
    ) { join_probe_count_impl<T>(probe_keys, table_keys, table_rows, counts, gid, *probe_len_ptr, *table_size_ptr); } \
    \
    [[kernel]] void join_probe_write_##suffix( \
        device const T* probe_keys [[buffer(0)]], \
        device const uint* table_keys [[buffer(1)]], \
        device const uint* table_rows [[buffer(2)]], \
        device const uint* offsets [[buffer(3)]], \
        device uint* left_indices [[buffer(4)]], \
        device uint* right_indices [[buffer(5)]], \
        uint gid [[thread_position_in_grid]], \
        device const uint* probe_len_ptr [[buffer(6)]], \
        device const uint* table_size_ptr [[buffer(7)]] \
    ) { join_probe_write_impl<T>(probe_keys, table_keys, table_rows, offsets, left_indices, right_indices, gid, *probe_len_ptr, *table_size_ptr); }

INSTANTIATE_JOIN(float, float32)
INSTANTIATE_JOIN(int,   int32)
