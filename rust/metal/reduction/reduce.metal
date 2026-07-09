// Templated, SIMD-optimized reduction kernels.
//
// metal_stdlib, `using namespace metal;`, Limits<T>, and SumOp/MinOp/MaxOp
// are provided by the preamble (debug.metal + types.h + reduce_ops.h)
// prepended by `with_preamble()` in kernels/mod.rs — do not re-include them.
//
// Each thread reads N_READS elements before doing a SIMD-group reduction
// (simd_sum/simd_min/simd_max), then a small threadgroup-level reduction
// across SIMD-group partials.
//
// NOTE: Apple GPU SIMD-group intrinsics (simd_sum/simd_min/simd_max, and the
// simd_shuffle_* primitives they would otherwise be built from) only accept
// operands up to 32 bits wide — `__is_valid_simdgroup_type<long>` is false,
// so `long`/`ulong` cannot go through Op::simd_reduce. The int64 kernels
// therefore fall back to a plain shared-memory tree reduction across the
// whole threadgroup instead of the SIMD-group fast path used by float32/int32.

// Tuned by Rust host via #define REDUCE_N_READS (per GPU family).
#ifndef REDUCE_N_READS
#define REDUCE_N_READS 4
#endif
constant constexpr uint N_READS = REDUCE_N_READS;

template <typename T, typename Op>
void reduce_impl_simd(
    device const T* input,
    device T* output,
    threadgroup T* shared,
    uint tid,
    uint group_id,
    uint group_size,
    device const uint* len_ptr
) {
    uint len = *len_ptr;

    T total = Op::identity;
    uint base = group_id * group_size * N_READS + tid;
    for (uint i = 0; i < N_READS; i++) {
        uint idx = base + i * group_size;
        total = Op::apply(total, (idx < len) ? input[idx] : Op::identity);
    }

    total = Op::simd_reduce(total);

    uint simd_gid = tid / 32;
    uint simd_lid = tid % 32;
    if (simd_lid == 0) shared[simd_gid] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simds = group_size / 32;
    if (tid < 32) {
        T val = (tid < num_simds) ? shared[tid] : Op::identity;
        val = Op::simd_reduce(val);
        if (tid == 0) output[group_id] = val;
    }
}

// Fallback for types without hardware SIMD-group reduction support (64-bit
// integers on Apple GPUs). Same per-thread N_READS unroll, but the
// cross-lane combine is a classic power-of-two shared-memory tree instead
// of Op::simd_reduce. `group_size` (THREADGROUP_SIZE = 256) must be a power
// of two.
template <typename T, typename Op>
void reduce_impl_tree(
    device const T* input,
    device T* output,
    threadgroup T* shared,
    uint tid,
    uint group_id,
    uint group_size,
    device const uint* len_ptr
) {
    uint len = *len_ptr;

    T total = Op::identity;
    uint base = group_id * group_size * N_READS + tid;
    for (uint i = 0; i < N_READS; i++) {
        uint idx = base + i * group_size;
        total = Op::apply(total, (idx < len) ? input[idx] : Op::identity);
    }

    shared[tid] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = Op::apply(shared[tid], shared[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        output[group_id] = shared[0];
    }
}

#define INSTANTIATE_REDUCE(T, suffix, Op) \
    [[kernel]] void reduce_##suffix( \
        device const T* input       [[buffer(0)]], \
        device T* output            [[buffer(1)]], \
        threadgroup T* shared       [[threadgroup(0)]], \
        uint tid                    [[thread_position_in_threadgroup]], \
        uint group_id               [[threadgroup_position_in_grid]], \
        uint group_size             [[threads_per_threadgroup]], \
        device const uint* len_ptr  [[buffer(2)]] \
    ) { reduce_impl_simd<T, Op<T>>(input, output, shared, tid, group_id, group_size, len_ptr); }

#define INSTANTIATE_REDUCE_TREE(T, suffix, Op) \
    [[kernel]] void reduce_##suffix( \
        device const T* input       [[buffer(0)]], \
        device T* output            [[buffer(1)]], \
        threadgroup T* shared       [[threadgroup(0)]], \
        uint tid                    [[thread_position_in_threadgroup]], \
        uint group_id               [[threadgroup_position_in_grid]], \
        uint group_size             [[threads_per_threadgroup]], \
        device const uint* len_ptr  [[buffer(2)]] \
    ) { reduce_impl_tree<T, Op<T>>(input, output, shared, tid, group_id, group_size, len_ptr); }

INSTANTIATE_REDUCE(float, float32_sum, SumOp)
INSTANTIATE_REDUCE(float, float32_min, MinOp)
INSTANTIATE_REDUCE(float, float32_max, MaxOp)

INSTANTIATE_REDUCE(int, int32_sum, SumOp)
INSTANTIATE_REDUCE(int, int32_min, MinOp)
INSTANTIATE_REDUCE(int, int32_max, MaxOp)

INSTANTIATE_REDUCE_TREE(long, int64_sum, SumOp)
INSTANTIATE_REDUCE_TREE(long, int64_min, MinOp)
INSTANTIATE_REDUCE_TREE(long, int64_max, MaxOp)

// Widening sum: reads narrow input T, accumulates as int64 to avoid overflow.
// Used by metal_mean for integer types. First pass widens, subsequent passes
// use reduce_int64_sum above.
template <typename T>
void reduce_widen_sum_impl(
    device const T* input,
    device long* output,
    threadgroup long* shared,
    uint tid, uint group_id, uint group_size,
    device const uint* len_ptr
) {
    uint len = *len_ptr;
    long total = 0;
    uint base = group_id * group_size * N_READS + tid;
    for (uint i = 0; i < N_READS; i++) {
        uint idx = base + i * group_size;
        total += (idx < len) ? long(input[idx]) : 0L;
    }

    shared[tid] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[group_id] = shared[0];
}

#define INSTANTIATE_REDUCE_WIDEN_SUM(T, suffix) \
    [[kernel]] void reduce_widen_sum_##suffix( \
        device const T* input       [[buffer(0)]], \
        device long* output         [[buffer(1)]], \
        threadgroup long* shared    [[threadgroup(0)]], \
        uint tid                    [[thread_position_in_threadgroup]], \
        uint group_id               [[threadgroup_position_in_grid]], \
        uint group_size             [[threads_per_threadgroup]], \
        device const uint* len_ptr  [[buffer(2)]] \
    ) { reduce_widen_sum_impl<T>(input, output, shared, tid, group_id, group_size, len_ptr); }

INSTANTIATE_REDUCE_WIDEN_SUM(int, int32)
