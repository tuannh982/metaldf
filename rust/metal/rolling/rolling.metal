// Rolling window kernels — GPU building blocks for
// `series.rolling(window).sum()/.mean()/.min()/.max()/.count()` (Phase 7).
//
// Naive parallel strategy (used for all window sizes today — see Task 7.1
// brief: the prefix-sum-based strategy for large windows is deferred to a
// follow-up task): each thread `idx` computes exactly one output element by
// iterating over its own window, `[max(0, idx - window + 1), idx]`
// (inclusive), directly from `data`. No cross-thread synchronization or
// threadgroup memory needed — simple, but O(window) work per thread, so this
// strategy is only efficient for small-to-moderate windows.
//
// Windows that aren't yet fully "filled" (i.e. `idx + 1 < window`, near the
// start of the series) are NOT specially masked here: each kernel just
// operates over however many elements are actually available
// (`min(idx + 1, window)`), matching pandas' default `min_periods=1`
// behavior directly. Callers wanting a stricter `min_periods` (including
// pandas' `rolling(window)` default of `min_periods=window`, which pandas
// itself overrides to `1` for size/count-style reductions) mask the affected
// leading positions to NaN in Python — this kernel does not decide that.
//
// `rolling_count_f32` counts how many elements are in-window at each
// position (`min(idx + 1, window)`), NOT how many are non-NaN — null-aware
// counting is a separate concern (this repo's rolling ops don't yet consult
// a `NullMask`; see `rust/src/kernels/rolling.rs`).
//
// f32 only for now (see Task 7.1 brief: i32 rolling variants are deferred).
// Float64 isn't instantiated: Metal has no `double` type (Task 2.1).
//
// Dispatched with `dispatch_thread_groups` (grid padded to a threadgroup
// multiple), hence the explicit `idx >= len` bounds guard on every kernel
// (see `rust/src/kernels/rolling.rs` for the dispatch side and the
// project-wide bounds-guard convention).

kernel void rolling_sum_f32(
    device const float* data   [[buffer(0)]],
    device float* output       [[buffer(1)]],
    device const uint* params  [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    uint len = params[0];
    uint window = params[1];
    if (idx >= len) return;

    uint start = (idx + 1 >= window) ? (idx + 1 - window) : 0;
    float sum = 0.0f;
    for (uint i = start; i <= idx; i++) {
        sum += data[i];
    }
    output[idx] = sum;
}

kernel void rolling_min_f32(
    device const float* data   [[buffer(0)]],
    device float* output       [[buffer(1)]],
    device const uint* params  [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    uint len = params[0];
    uint window = params[1];
    if (idx >= len) return;

    uint start = (idx + 1 >= window) ? (idx + 1 - window) : 0;
    float val = data[start];
    for (uint i = start + 1; i <= idx; i++) {
        val = min(val, data[i]);
    }
    output[idx] = val;
}

kernel void rolling_max_f32(
    device const float* data   [[buffer(0)]],
    device float* output       [[buffer(1)]],
    device const uint* params  [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    uint len = params[0];
    uint window = params[1];
    if (idx >= len) return;

    uint start = (idx + 1 >= window) ? (idx + 1 - window) : 0;
    float val = data[start];
    for (uint i = start + 1; i <= idx; i++) {
        val = max(val, data[i]);
    }
    output[idx] = val;
}

kernel void rolling_count_f32(
    device const float* data   [[buffer(0)]],
    device float* output       [[buffer(1)]],
    device const uint* params  [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    uint len = params[0];
    uint window = params[1];
    if (idx >= len) return;

    uint start = (idx + 1 >= window) ? (idx + 1 - window) : 0;
    output[idx] = float(idx - start + 1);
}
