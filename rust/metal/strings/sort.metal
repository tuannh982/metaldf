#include <metal_stdlib>
using namespace metal;

// Bitonic sort on an index array, comparing the strings each index points to
// via `string_compare`. One dispatch = one (stage, step) pass of the
// classic bitonic sort network; each thread owns exactly one comparison
// pair, so `padded_n / 2` threads are dispatched per pass.
//
// `indices` has `padded_n` (next power of two >= n) slots. Positions
// `[n, padded_n)` are pre-filled with the sentinel value `n` (an
// out-of-bounds string index) by the caller. Because `n` can never be a
// valid string index, any index value `>= real_n` is recognized here as
// padding and is treated as "greater than every real string" — mirroring
// how the numeric bitonic sort pads with +infinity — so padding always
// migrates to the tail of the array by the time the network finishes.
kernel void string_bitonic_sort(
    device uint*          indices      [[buffer(0)]],
    device const int64_t* offsets      [[buffer(1)]],
    device const uchar*   chars        [[buffer(2)]],
    device const uint*    padded_n_ptr [[buffer(3)]],
    device const uint*    stage_ptr    [[buffer(4)]],
    device const uint*    step_ptr     [[buffer(5)]],
    device const uint*    real_n_ptr   [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    uint padded_n = *padded_n_ptr;
    uint stage = *stage_ptr;
    uint step = *step_ptr;
    uint real_n = *real_n_ptr;

    // `outer_block_size` (k = 2^(stage+1)) is the size of the bitonic
    // sequence being merged this stage — it alone decides each element's
    // ascending/descending direction. `pair_distance` (j = 2^step) is the
    // comparison distance for *this* pass within that merge, and must also
    // drive the sub-block decomposition used to pick each thread's (a, b)
    // pair: sub-blocks of size 2j, split into two halves of size j each
    // compared j apart. Using the (stage-derived) outer block size for the
    // pairing math instead of the (step-derived) 2j sub-block — as opposed
    // to just the direction — would make every pass after the first one
    // for a given stage assign some elements to two overlapping pairs.
    uint pair_distance = 1u << step;             // j
    uint outer_block_size = 1u << (stage + 1);   // k
    uint half_block = pair_distance;             // j
    uint block_size = pair_distance << 1;        // 2j

    uint total_pairs = padded_n >> 1;
    if (tid >= total_pairs) return;

    uint block_id = tid / half_block;
    uint pos_in_block = tid % half_block;

    uint a_idx = block_id * block_size + pos_in_block;
    uint b_idx = a_idx + pair_distance;

    if (b_idx >= padded_n) return;

    uint idx_a = indices[a_idx];
    uint idx_b = indices[b_idx];

    bool a_valid = idx_a < real_n;
    bool b_valid = idx_b < real_n;

    int cmp;
    if (a_valid && b_valid) {
        StringRef sa = get_string(offsets, chars, idx_a);
        StringRef sb = get_string(offsets, chars, idx_b);
        cmp = string_compare(sa, sb);
    } else if (!a_valid && !b_valid) {
        cmp = 0;
    } else if (!a_valid) {
        cmp = 1;   // a is padding -> sorts after any real string
    } else {
        cmp = -1;  // b is padding -> a (real) sorts before it
    }

    bool ascending = (a_idx & outer_block_size) == 0;
    bool should_swap = ascending ? (cmp > 0) : (cmp < 0);

    if (should_swap) {
        indices[a_idx] = idx_b;
        indices[b_idx] = idx_a;
    }
}

// Gather pass 1: compute the byte length of the string each sorted index
// points to, so the caller can prefix-sum on the CPU to build new offsets.
kernel void string_gather_sizes(
    device const uint*    indices   [[buffer(0)]],
    device const int64_t* offsets   [[buffer(1)]],
    device int64_t*       sizes_out [[buffer(2)]],
    device const uint*    len_ptr   [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    uint idx = indices[tid];
    sizes_out[tid] = offsets[idx + 1] - offsets[idx];
}

// Gather pass 2: copy each source string's bytes into its new location,
// given the offsets produced from the CPU prefix-sum over pass 1's sizes.
kernel void string_gather_write(
    device const uint*    indices     [[buffer(0)]],
    device const int64_t* offsets_in  [[buffer(1)]],
    device const uchar*   chars_in    [[buffer(2)]],
    device const int64_t* offsets_out [[buffer(3)]],
    device uchar*         chars_out   [[buffer(4)]],
    device const uint*    len_ptr     [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    uint n = *len_ptr;
    if (tid >= n) return;
    uint idx = indices[tid];
    int64_t src_start = offsets_in[idx];
    int64_t src_len = offsets_in[idx + 1] - src_start;
    int64_t dst_start = offsets_out[tid];
    for (int64_t i = 0; i < src_len; i++) {
        chars_out[dst_start + i] = chars_in[src_start + i];
    }
}
