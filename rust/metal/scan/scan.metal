// GPU inclusive prefix-sum (scan) kernels — two-pass algorithm.
//
// Pass 1 (scan_inclusive_*): each threadgroup loads up to
// SCAN_THREADGROUP_SIZE elements into shared memory (one element per
// thread), runs a Hillis-Steele inclusive scan there, writes the scanned
// values back to `output`, and has its last thread publish the group's
// total (the group's inclusive-scan value at its final lane) to
// `partials[group_id]`.
//
// Between pass 1 and pass 2, the Rust dispatcher (`rust/src/kernels/scan.rs`)
// recursively scans `partials` in place (via a fresh call to
// `prefix_sum_inclusive`), turning per-group totals into per-group exclusive
// prefixes-of-all-prior-groups.
//
// Pass 2 (scan_propagate_*): adds `partials[group_id - 1]` (the running
// total contributed by every earlier group) to each element of group
// `group_id`. Group 0 has no predecessor and is left untouched.
//
// Only `uint32_t` and `int32_t` are instantiated (see Task 3.1 brief:
// float64 isn't supported on Metal at all — discovered in Task 2.1 — and
// int64/float32 scan support is deferred to a later task; Phase 4's
// boolean-mask-counting use case only needs 32-bit integer counts).

#ifndef SCAN_THREADGROUP_SIZE
#define SCAN_THREADGROUP_SIZE 256
#endif

template <typename T>
void scan_inclusive_impl(
    device const T* input,
    device T* output,
    device T* partials,
    threadgroup T* shared,
    uint tid,
    uint group_id,
    uint group_size,
    device const uint* len_ptr
) {
    uint len = *len_ptr;
    uint base = group_id * group_size + tid;

    // Load into shared memory. Out-of-bounds lanes (last, partial
    // threadgroup) contribute the additive identity so they don't perturb
    // the scan of the in-bounds lanes that share this threadgroup.
    shared[tid] = (base < len) ? input[base] : T(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hillis-Steele inclusive scan: log2(group_size) passes. Each pass reads
    // the whole shared array into a local `val` (guarded by the trailing
    // barrier from the *previous* iteration), then a barrier ensures every
    // thread has finished reading before any thread overwrites `shared` —
    // required because this scan is done in place.
    for (uint offset = 1; offset < group_size; offset *= 2) {
        T val = shared[tid];
        if (tid >= offset) {
            val = shared[tid - offset] + shared[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[tid] = val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (base < len) {
        output[base] = shared[tid];
    }

    // Last thread in the group publishes this group's total as the partial
    // sum consumed by pass 2 (and by the recursive scan of `partials`).
    if (tid == group_size - 1) {
        partials[group_id] = shared[tid];
    }
}

template <typename T>
void scan_propagate_impl(
    device T* data,
    device const T* partials,
    uint tid,
    uint group_id,
    uint group_size,
    device const uint* len_ptr
) {
    if (group_id == 0) return;
    uint len = *len_ptr;
    uint base = group_id * group_size + tid;
    if (base < len) {
        data[base] += partials[group_id - 1];
    }
}

#define INSTANTIATE_SCAN(T, suffix) \
    [[kernel]] void scan_inclusive_##suffix( \
        device const T* input       [[buffer(0)]], \
        device T* output            [[buffer(1)]], \
        device T* partials          [[buffer(2)]], \
        device const uint* len_ptr  [[buffer(3)]], \
        threadgroup T* shared       [[threadgroup(0)]], \
        uint tid                    [[thread_position_in_threadgroup]], \
        uint group_id               [[threadgroup_position_in_grid]], \
        uint group_size             [[threads_per_threadgroup]] \
    ) { scan_inclusive_impl<T>(input, output, partials, shared, tid, group_id, group_size, len_ptr); } \
    [[kernel]] void scan_propagate_##suffix( \
        device T* data              [[buffer(0)]], \
        device const T* partials    [[buffer(1)]], \
        device const uint* len_ptr  [[buffer(2)]], \
        uint tid                    [[thread_position_in_threadgroup]], \
        uint group_id               [[threadgroup_position_in_grid]], \
        uint group_size             [[threads_per_threadgroup]] \
    ) { scan_propagate_impl<T>(data, partials, tid, group_id, group_size, len_ptr); }

INSTANTIATE_SCAN(uint, uint32)
INSTANTIATE_SCAN(int, int32)
