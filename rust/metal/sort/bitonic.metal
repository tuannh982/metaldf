// Bitonic sort — used for small N (< 100K) where GPU dispatch overhead
// of multi-pass radix sort isn't worth it.
//
// Three kernel types, dispatched by the Rust host:
//   1. bitonic_sort_local:  sorts each 1024-element block entirely in
//      threadgroup memory (stages 0..9, 55 barriers, zero global traffic)
//   2. bitonic_sort_global: one (stage, step) comparison-swap in global
//      memory, for steps where the comparison distance >= 1024
//   3. bitonic_merge_local: after global steps finish for a stage, does
//      the remaining local steps (9..0) in threadgroup memory
//
// Apple GPU Family 9: max 1024 threads/threadgroup, 32 KB threadgroup mem.
// 1024 keys (8B) + 1024 indices (4B) = 12 KB — well under limit.

// Tuned by Rust host via #define (per GPU family).
#ifndef LOCAL_SORT_SIZE
#define LOCAL_SORT_SIZE 1024
#endif
#ifndef LOCAL_SORT_STAGES
#define LOCAL_SORT_STAGES 10
#endif

// --- Global step (unchanged, used for cross-threadgroup comparisons) ---

template <typename T>
void bitonic_sort_impl(
    device T* data,
    device uint* indices,
    uint gid,
    uint n,
    uint stage,
    uint step
) {
    if (gid >= n) return;
    uint i = gid;
    uint j = i ^ (1u << step);
    if (j > i && j < n) {
        T a = data[i];
        T b = data[j];
        uint ia = indices[i];
        uint ib = indices[j];
        bool ascending = ((i >> (stage + 1)) & 1u) == 0u;
        bool swap = ascending ? (a > b) : (a < b);
        if (swap) {
            data[i] = b;
            data[j] = a;
            indices[i] = ib;
            indices[j] = ia;
        }
    }
}

// --- Local sort: full bitonic sort of each 1024-element block in shared memory ---

template <typename T>
void bitonic_sort_local_impl(
    device T* data,
    device uint* indices,
    threadgroup T* lk,
    threadgroup uint* li,
    uint tid,
    uint group_id,
    uint n
) {
    uint gid = group_id * LOCAL_SORT_SIZE + tid;
    lk[tid] = (gid < n) ? data[gid] : Limits<T>::max_val;
    li[tid] = (gid < n) ? indices[gid] : gid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stage = 0; stage < LOCAL_SORT_STAGES; stage++) {
        for (int step = int(stage); step >= 0; step--) {
            uint partner = tid ^ (1u << uint(step));
            if (partner > tid) {
                bool asc = ((gid >> (stage + 1)) & 1u) == 0u;
                T a = lk[tid], b = lk[partner];
                if (asc ? (a > b) : (a < b)) {
                    lk[tid] = b; lk[partner] = a;
                    uint tmp = li[tid]; li[tid] = li[partner]; li[partner] = tmp;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (gid < n) { data[gid] = lk[tid]; indices[gid] = li[tid]; }
}

// --- Local merge: after global steps of a stage, do remaining local steps in shared memory ---

template <typename T>
void bitonic_merge_local_impl(
    device T* data,
    device uint* indices,
    threadgroup T* lk,
    threadgroup uint* li,
    uint tid,
    uint group_id,
    uint n,
    uint stage
) {
    uint gid = group_id * LOCAL_SORT_SIZE + tid;
    lk[tid] = (gid < n) ? data[gid] : Limits<T>::max_val;
    li[tid] = (gid < n) ? indices[gid] : gid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool asc = ((gid >> (stage + 1)) & 1u) == 0u;

    for (int step = int(LOCAL_SORT_STAGES) - 1; step >= 0; step--) {
        uint partner = tid ^ (1u << uint(step));
        if (partner > tid) {
            T a = lk[tid], b = lk[partner];
            if (asc ? (a > b) : (a < b)) {
                lk[tid] = b; lk[partner] = a;
                uint tmp = li[tid]; li[tid] = li[partner]; li[partner] = tmp;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (gid < n) { data[gid] = lk[tid]; indices[gid] = li[tid]; }
}

// --- Instantiation ---

#define INSTANTIATE_BITONIC(T, suffix) \
    [[kernel]] void bitonic_sort_##suffix##_ascending( \
        device T* data              [[buffer(0)]], \
        device uint* indices        [[buffer(1)]], \
        uint gid                    [[thread_position_in_grid]], \
        device const uint* array_len [[buffer(2)]], \
        device const uint* stage    [[buffer(3)]], \
        device const uint* step     [[buffer(4)]] \
    ) { bitonic_sort_impl<T>(data, indices, gid, *array_len, *stage, *step); }

#define INSTANTIATE_BITONIC_LOCAL(T, suffix) \
    [[kernel]] void bitonic_sort_local_##suffix( \
        device T* data              [[buffer(0)]], \
        device uint* indices        [[buffer(1)]], \
        threadgroup T* lk           [[threadgroup(0)]], \
        threadgroup uint* li        [[threadgroup(1)]], \
        uint tid                    [[thread_position_in_threadgroup]], \
        uint group_id               [[threadgroup_position_in_grid]], \
        device const uint* n_ptr    [[buffer(2)]] \
    ) { bitonic_sort_local_impl<T>(data, indices, lk, li, tid, group_id, *n_ptr); } \
    [[kernel]] void bitonic_merge_local_##suffix( \
        device T* data              [[buffer(0)]], \
        device uint* indices        [[buffer(1)]], \
        threadgroup T* lk           [[threadgroup(0)]], \
        threadgroup uint* li        [[threadgroup(1)]], \
        uint tid                    [[thread_position_in_threadgroup]], \
        uint group_id               [[threadgroup_position_in_grid]], \
        device const uint* n_ptr    [[buffer(2)]], \
        device const uint* stage_ptr [[buffer(3)]] \
    ) { bitonic_merge_local_impl<T>(data, indices, lk, li, tid, group_id, *n_ptr, *stage_ptr); }

INSTANTIATE_BITONIC(float, float32)
INSTANTIATE_BITONIC(int,   int32)
INSTANTIATE_BITONIC(long,  int64)

INSTANTIATE_BITONIC_LOCAL(float, float32)
INSTANTIATE_BITONIC_LOCAL(int,   int32)
INSTANTIATE_BITONIC_LOCAL(long,  int64)
