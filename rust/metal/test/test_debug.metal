// test_debug.metal
// Minimal kernel to verify debug logging works end-to-end.
// debug.metal is prepended by the Rust build system.

#include <metal_stdlib>
using namespace metal;

[[kernel]] void test_debug_printf(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    uint gid                  [[thread_position_in_grid]]
) {
    METAL_LOG_IF(gid, 8, "input[%u] = %f", gid, input[gid]);

    float result = input[gid] + 1.0f;
    output[gid] = result;

    METAL_LOG_IF(gid, 4, "output[%u] = %f", gid, result);
}
