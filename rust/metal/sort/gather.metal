// Gather kernel — reorders values by an index array.
// Used after radix sort to reorder associated value columns.

template <typename T>
void gather_impl(
    device const T* values,
    device const uint* indices,
    device T* out,
    uint gid,
    uint array_len
) {
    if (gid >= array_len) return;
    out[gid] = values[indices[gid]];
}

#define INSTANTIATE_GATHER(T, suffix) \
    [[kernel]] void gather_##suffix( \
        device const T* values      [[buffer(0)]], \
        device const uint* indices  [[buffer(1)]], \
        device T* out               [[buffer(2)]], \
        uint gid                    [[thread_position_in_grid]], \
        device const uint* len_ptr  [[buffer(3)]] \
    ) { gather_impl<T>(values, indices, out, gid, *len_ptr); }

INSTANTIATE_GATHER(float, float32)
INSTANTIATE_GATHER(int,   int32)
INSTANTIATE_GATHER(long,  int64)
