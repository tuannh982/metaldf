// Sort-based GroupBy — serial leader-scan fallback for high cardinality.
// Each thread starting a new key run scans forward and reduces its segment.

template <typename T, typename Op>
void groupby_agg_direct_impl(
    device const T* sorted_keys, device const T* sorted_values,
    device T* out_keys, device T* out_values,
    device atomic_uint* group_counter,
    uint gid, uint array_len
) {
    if (gid >= array_len) return;
    bool is_leader = (gid == 0) || (sorted_keys[gid] != sorted_keys[gid - 1]);
    if (!is_leader) return;

    T key = sorted_keys[gid];
    uint end = gid + 1;
    while (end < array_len && sorted_keys[end] == key) end++;

    T acc = Op::identity;
    for (uint i = gid; i < end; i++) acc = Op::apply(acc, sorted_values[i]);

    uint g = atomic_fetch_add_explicit(group_counter, 1u, memory_order_relaxed);
    out_keys[g] = key;
    out_values[g] = acc;
}

template <typename T>
void groupby_count_direct_impl(
    device const T* sorted_keys,
    device T* out_keys, device long* out_values,
    device atomic_uint* group_counter,
    uint gid, uint array_len
) {
    if (gid >= array_len) return;
    bool is_leader = (gid == 0) || (sorted_keys[gid] != sorted_keys[gid - 1]);
    if (!is_leader) return;

    T key = sorted_keys[gid];
    uint end = gid + 1;
    while (end < array_len && sorted_keys[end] == key) end++;

    uint g = atomic_fetch_add_explicit(group_counter, 1u, memory_order_relaxed);
    out_keys[g] = key;
    out_values[g] = long(end - gid);
}

// Fused sum+count for mean — single group_counter claim per leader
// guarantees sum[g] and count[g] refer to the same group.
template <typename T>
void groupby_sum_count_direct_impl(
    device const T* sorted_keys, device const T* sorted_values,
    device T* out_keys, device T* out_sums, device long* out_counts,
    device atomic_uint* group_counter,
    uint gid, uint array_len
) {
    if (gid >= array_len) return;
    bool is_leader = (gid == 0) || (sorted_keys[gid] != sorted_keys[gid - 1]);
    if (!is_leader) return;

    T key = sorted_keys[gid];
    uint end = gid + 1;
    while (end < array_len && sorted_keys[end] == key) end++;

    T sum = SumOp<T>::identity;
    for (uint i = gid; i < end; i++) sum = SumOp<T>::apply(sum, sorted_values[i]);

    uint g = atomic_fetch_add_explicit(group_counter, 1u, memory_order_relaxed);
    out_keys[g] = key;
    out_sums[g] = sum;
    out_counts[g] = long(end - gid);
}

#define INSTANTIATE_GROUPBY_AGG_DIRECT(T, suffix, Op, opname) \
    [[kernel]] void groupby_##opname##_direct_##suffix( \
        device const T* sorted_keys [[buffer(0)]], device const T* sorted_values [[buffer(1)]], \
        device T* out_keys [[buffer(2)]], device T* out_values [[buffer(3)]], \
        device atomic_uint* group_counter [[buffer(4)]], \
        uint gid [[thread_position_in_grid]], device const uint* array_len_ptr [[buffer(5)]] \
    ) { groupby_agg_direct_impl<T, Op<T>>(sorted_keys, sorted_values, out_keys, out_values, group_counter, gid, *array_len_ptr); }

#define INSTANTIATE_GROUPBY_COUNT_DIRECT(T, suffix) \
    [[kernel]] void groupby_count_direct_##suffix( \
        device const T* sorted_keys [[buffer(0)]], device T* out_keys [[buffer(1)]], \
        device long* out_values [[buffer(2)]], device atomic_uint* group_counter [[buffer(3)]], \
        uint gid [[thread_position_in_grid]], device const uint* array_len_ptr [[buffer(4)]] \
    ) { groupby_count_direct_impl<T>(sorted_keys, out_keys, out_values, group_counter, gid, *array_len_ptr); }

#define INSTANTIATE_GROUPBY_SUM_COUNT_DIRECT(T, suffix) \
    [[kernel]] void groupby_sum_count_direct_##suffix( \
        device const T* sorted_keys [[buffer(0)]], device const T* sorted_values [[buffer(1)]], \
        device T* out_keys [[buffer(2)]], device T* out_sums [[buffer(3)]], \
        device long* out_counts [[buffer(4)]], device atomic_uint* group_counter [[buffer(5)]], \
        uint gid [[thread_position_in_grid]], device const uint* array_len_ptr [[buffer(6)]] \
    ) { groupby_sum_count_direct_impl<T>(sorted_keys, sorted_values, out_keys, out_sums, out_counts, group_counter, gid, *array_len_ptr); }

#define INSTANTIATE_GROUPBY_SORT(T, suffix) \
    INSTANTIATE_GROUPBY_AGG_DIRECT(T, suffix, SumOp, sum) \
    INSTANTIATE_GROUPBY_AGG_DIRECT(T, suffix, MinOp, min) \
    INSTANTIATE_GROUPBY_AGG_DIRECT(T, suffix, MaxOp, max) \
    INSTANTIATE_GROUPBY_COUNT_DIRECT(T, suffix) \
    INSTANTIATE_GROUPBY_SUM_COUNT_DIRECT(T, suffix)

INSTANTIATE_GROUPBY_SORT(float, float32)
INSTANTIATE_GROUPBY_SORT(int,   int32)
