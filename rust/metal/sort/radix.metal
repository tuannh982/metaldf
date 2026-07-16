// LSD 8-bit radix sort scatter kernel.
// CPU computes histogram + per-element local offsets; this kernel writes
// each element to its final sorted position for one radix pass.

template <typename T>
void radix_scatter_impl(
    device const T* keys_in,
    device const uint* indices_in,
    device T* keys_out,
    device uint* indices_out,
    device const uint* bucket_offsets,
    device const uint* local_offsets,
    uint gid,
    uint array_len,
    uint pass
) {
    if (gid >= array_len) return;
    using Traits = RadixTraits<T>;
    T key_val = keys_in[gid];
    typename Traits::KeyT key = Traits::to_key(key_val);
    uint digit = (key >> (pass * 8)) & 0xFFu;
    uint idx = indices_in[gid];
    uint pos = bucket_offsets[digit] + local_offsets[gid];
    keys_out[pos] = key_val;
    indices_out[pos] = idx;
}

#define INSTANTIATE_RADIX_SCATTER(T, suffix) \
    [[kernel]] void radix_scatter_##suffix( \
        device const T* keys_in     [[buffer(0)]], \
        device const uint* idx_in   [[buffer(1)]], \
        device T* keys_out          [[buffer(2)]], \
        device uint* idx_out        [[buffer(3)]], \
        device const uint* bucket_offsets [[buffer(4)]], \
        device const uint* local_offsets  [[buffer(5)]], \
        uint gid                    [[thread_position_in_grid]], \
        device const uint* len_ptr  [[buffer(6)]], \
        device const uint* pass_ptr [[buffer(7)]] \
    ) { \
        uint len = *len_ptr; uint pass = *pass_ptr; \
        radix_scatter_impl<T>(keys_in, idx_in, keys_out, idx_out, \
            bucket_offsets, local_offsets, gid, len, pass); \
    }

INSTANTIATE_RADIX_SCATTER(float, float32)
INSTANTIATE_RADIX_SCATTER(int,   int32)
INSTANTIATE_RADIX_SCATTER(long,  int64)
INSTANTIATE_RADIX_SCATTER(char,   int8)
INSTANTIATE_RADIX_SCATTER(short,  int16)
INSTANTIATE_RADIX_SCATTER(uchar,  uint8)
INSTANTIATE_RADIX_SCATTER(ushort, uint16)
INSTANTIATE_RADIX_SCATTER(uint,   uint32)
INSTANTIATE_RADIX_SCATTER(ulong,  uint64)
