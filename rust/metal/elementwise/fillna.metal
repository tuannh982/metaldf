// fillna: replace NaN with a scalar fill value.

[[kernel]] void fillna_f32(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    device const float* fill   [[buffer(2)]],
    device const uint* len_ptr [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= *len_ptr) return;
    output[idx] = isnan(input[idx]) ? *fill : input[idx];
}
