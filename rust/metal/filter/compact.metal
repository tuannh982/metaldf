// Stream compaction (compact) and gather-by-index (take) kernels — the GPU
// building blocks for `df[mask]` boolean indexing (Phase 4).
//
// compact_{suffix}: keeps only elements where mask[i] == 1, writing each
// kept element contiguously into `output` at `prefix_sum[i] - 1`.
// `prefix_sum` is the GPU inclusive prefix-sum (see
// `rust/src/kernels/scan.rs::prefix_sum_inclusive`) of the mask cast to
// uint32 — i.e. `prefix_sum[i]` is the 1-based rank of element `i` among
// all kept elements up to and including it, so `prefix_sum[i] - 1` is the
// right 0-based output slot whenever `mask[i]` is set. Dispatched with
// `dispatch_thread_groups` (grid padded up to a threadgroup multiple), so
// (unlike the elementwise kernels, which use `dispatch_threads` and skip
// this) it needs the explicit `idx >= len` bounds guard.
//
// take_{suffix}: `output[i] = data[indices[i]]` — a plain gather by index,
// unrelated to the mask/prefix-sum machinery above (`indices` need not come
// from a mask at all). Also dispatched with `dispatch_thread_groups`, hence
// the same bounds guard.
//
// Suffixes here follow the elementwise convention (`f32`/`i32`/`i64`/`u8`),
// not `DType::kernel_suffix()`'s (`float32`/`int32`/`int64`/`uint8`) — see
// `metal_suffix()`'s docs in `rust/src/kernels/elementwise.rs` for why the
// two naming schemes coexist. Float64 isn't instantiated: Metal has no
// `double` type (discovered in Task 2.1; see `rust/src/kernels/scan.rs`).

#define COMPACT_KERNEL(T, suffix) \
kernel void compact_##suffix( \
    device const T* data           [[buffer(0)]], \
    device const uint8_t* mask     [[buffer(1)]], \
    device const uint* prefix_sum  [[buffer(2)]], \
    device T* output               [[buffer(3)]], \
    device const uint* len_ptr     [[buffer(4)]], \
    uint idx [[thread_position_in_grid]] \
) { \
    uint len = *len_ptr; \
    if (idx >= len) return; \
    if (mask[idx]) { \
        output[prefix_sum[idx] - 1] = data[idx]; \
    } \
}

COMPACT_KERNEL(float, f32)
COMPACT_KERNEL(int, i32)
COMPACT_KERNEL(long, i64)
COMPACT_KERNEL(uchar, u8)
COMPACT_KERNEL(char, i8)
COMPACT_KERNEL(short, i16)
COMPACT_KERNEL(uint, u32)
COMPACT_KERNEL(ushort, u16)
COMPACT_KERNEL(ulong, u64)

#define TAKE_KERNEL(T, suffix) \
kernel void take_##suffix( \
    device const T* data           [[buffer(0)]], \
    device const uint* indices     [[buffer(1)]], \
    device T* output               [[buffer(2)]], \
    device const uint* len_ptr     [[buffer(3)]], \
    uint idx [[thread_position_in_grid]] \
) { \
    uint len = *len_ptr; \
    if (idx >= len) return; \
    output[idx] = data[indices[idx]]; \
}

TAKE_KERNEL(float, f32)
TAKE_KERNEL(int, i32)
TAKE_KERNEL(long, i64)
TAKE_KERNEL(uchar, u8)
TAKE_KERNEL(char, i8)
TAKE_KERNEL(short, i16)
TAKE_KERNEL(uint, u32)
TAKE_KERNEL(ushort, u16)
TAKE_KERNEL(ulong, u64)
