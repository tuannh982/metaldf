// GPU inclusive prefix-scan kernels — op-generic two-pass algorithm.
//
// Pass 1 (scan_inclusive_*): each threadgroup loads up to
// SCAN_THREADGROUP_SIZE elements, runs a Hillis-Steele inclusive scan
// in shared memory using Op::apply, writes scanned values to output,
// and publishes the group's total to partials[group_id].
//
// Between passes, the Rust dispatcher recursively scans partials.
//
// Pass 2 (scan_propagate_*): applies Op::apply(partials[group_id-1], elem)
// to each element of group group_id. Group 0 is left untouched.

#ifndef SCAN_THREADGROUP_SIZE
#define SCAN_THREADGROUP_SIZE 256
#endif

template <typename T, typename Op>
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

    shared[tid] = (base < len) ? input[base] : Op::identity;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = 1; offset < group_size; offset *= 2) {
        T val = shared[tid];
        if (tid >= offset) {
            val = Op::apply(shared[tid - offset], shared[tid]);
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

template <typename T, typename Op>
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
        data[base] = Op::apply(partials[group_id - 1], data[base]);
    }
}

#define INSTANTIATE_SCAN_OP(T, Op, suffix) \
    [[kernel]] void scan_inclusive_##suffix( \
        device const T* input       [[buffer(0)]], \
        device T* output            [[buffer(1)]], \
        device T* partials          [[buffer(2)]], \
        device const uint* len_ptr  [[buffer(3)]], \
        threadgroup T* shared       [[threadgroup(0)]], \
        uint tid                    [[thread_position_in_threadgroup]], \
        uint group_id               [[threadgroup_position_in_grid]], \
        uint group_size             [[threads_per_threadgroup]] \
    ) { scan_inclusive_impl<T, Op<T>>(input, output, partials, shared, tid, group_id, group_size, len_ptr); } \
    [[kernel]] void scan_propagate_##suffix( \
        device T* data              [[buffer(0)]], \
        device const T* partials    [[buffer(1)]], \
        device const uint* len_ptr  [[buffer(2)]], \
        uint tid                    [[thread_position_in_threadgroup]], \
        uint group_id               [[threadgroup_position_in_grid]], \
        uint group_size             [[threads_per_threadgroup]] \
    ) { scan_propagate_impl<T, Op<T>>(data, partials, tid, group_id, group_size, len_ptr); }

// cumsum
INSTANTIATE_SCAN_OP(float, SumOp, sum_float32)
INSTANTIATE_SCAN_OP(int,   SumOp, sum_int32)
INSTANTIATE_SCAN_OP(long,  SumOp, sum_int64)
INSTANTIATE_SCAN_OP(uint,  SumOp, sum_uint32)

// cummin
INSTANTIATE_SCAN_OP(float, MinOp, min_float32)
INSTANTIATE_SCAN_OP(int,   MinOp, min_int32)
INSTANTIATE_SCAN_OP(long,  MinOp, min_int64)

// cummax
INSTANTIATE_SCAN_OP(float, MaxOp, max_float32)
INSTANTIATE_SCAN_OP(int,   MaxOp, max_int32)
INSTANTIATE_SCAN_OP(long,  MaxOp, max_int64)
