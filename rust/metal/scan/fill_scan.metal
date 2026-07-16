// Forward-fill and backward-fill via parallel scan.
// ffill propagates the last valid (non-NaN) value forward.
// bfill propagates the next valid value backward.

#ifndef SCAN_THREADGROUP_SIZE
#define SCAN_THREADGROUP_SIZE 256
#endif

// --- Forward fill (ffill) ---

template <typename T>
void ffill_scan_impl(
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

    shared[tid] = (base < len) ? input[base] : NAN;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = 1; offset < group_size; offset *= 2) {
        T val = shared[tid];
        if (tid >= offset) {
            val = isnan(shared[tid]) ? shared[tid - offset] : shared[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[tid] = val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (base < len) {
        output[base] = shared[tid];
    }

    if (tid == group_size - 1) {
        partials[group_id] = shared[tid];
    }
}

template <typename T>
void ffill_propagate_impl(
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
    if (base < len && isnan(data[base])) {
        T prev = partials[group_id - 1];
        if (!isnan(prev)) {
            data[base] = prev;
        }
    }
}

#define INSTANTIATE_FFILL(T, suffix) \
    [[kernel]] void ffill_scan_##suffix( \
        device const T* input       [[buffer(0)]], \
        device T* output            [[buffer(1)]], \
        device T* partials          [[buffer(2)]], \
        device const uint* len_ptr  [[buffer(3)]], \
        threadgroup T* shared       [[threadgroup(0)]], \
        uint tid                    [[thread_position_in_threadgroup]], \
        uint group_id               [[threadgroup_position_in_grid]], \
        uint group_size             [[threads_per_threadgroup]] \
    ) { ffill_scan_impl<T>(input, output, partials, shared, tid, group_id, group_size, len_ptr); } \
    [[kernel]] void ffill_propagate_##suffix( \
        device T* data              [[buffer(0)]], \
        device const T* partials    [[buffer(1)]], \
        device const uint* len_ptr  [[buffer(2)]], \
        uint tid                    [[thread_position_in_threadgroup]], \
        uint group_id               [[threadgroup_position_in_grid]], \
        uint group_size             [[threads_per_threadgroup]] \
    ) { ffill_propagate_impl<T>(data, partials, tid, group_id, group_size, len_ptr); }

INSTANTIATE_FFILL(float, float32)

// --- Backward fill (bfill) ---

template <typename T>
void bfill_scan_impl(
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
    uint ridx = (base < len) ? (len - 1 - base) : base;

    shared[tid] = (base < len) ? input[ridx] : NAN;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = 1; offset < group_size; offset *= 2) {
        T val = shared[tid];
        if (tid >= offset) {
            val = isnan(shared[tid]) ? shared[tid - offset] : shared[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        shared[tid] = val;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (base < len) {
        output[ridx] = shared[tid];
    }

    if (tid == group_size - 1) {
        partials[group_id] = shared[tid];
    }
}

template <typename T>
void bfill_propagate_impl(
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
        uint ridx = len - 1 - base;
        if (isnan(data[ridx])) {
            T prev = partials[group_id - 1];
            if (!isnan(prev)) {
                data[ridx] = prev;
            }
        }
    }
}

#define INSTANTIATE_BFILL(T, suffix) \
    [[kernel]] void bfill_scan_##suffix( \
        device const T* input       [[buffer(0)]], \
        device T* output            [[buffer(1)]], \
        device T* partials          [[buffer(2)]], \
        device const uint* len_ptr  [[buffer(3)]], \
        threadgroup T* shared       [[threadgroup(0)]], \
        uint tid                    [[thread_position_in_threadgroup]], \
        uint group_id               [[threadgroup_position_in_grid]], \
        uint group_size             [[threads_per_threadgroup]] \
    ) { bfill_scan_impl<T>(input, output, partials, shared, tid, group_id, group_size, len_ptr); } \
    [[kernel]] void bfill_propagate_##suffix( \
        device T* data              [[buffer(0)]], \
        device const T* partials    [[buffer(1)]], \
        device const uint* len_ptr  [[buffer(2)]], \
        uint tid                    [[thread_position_in_threadgroup]], \
        uint group_id               [[threadgroup_position_in_grid]], \
        uint group_size             [[threads_per_threadgroup]] \
    ) { bfill_propagate_impl<T>(data, partials, tid, group_id, group_size, len_ptr); }

INSTANTIATE_BFILL(float, float32)
