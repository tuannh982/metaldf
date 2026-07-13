#include <metal_stdlib>
using namespace metal;

// Opcodes are defined in 01_opcodes.h, concatenated before this file (same
// constants used by eval.metal).

// Evaluate bytecode program for element at position `gid`, returning
// the resulting float. Shared with the pure-eval kernel but inlined here
// to avoid cross-file dependencies.
inline float eval_program(
    device const float* cols[8],
    constant uint8_t* program,
    uint prog_len,
    uint gid
) {
    float stack[8];
    int sp = 0;

    for (uint pc = 0; pc < prog_len; ) {
        uint8_t op = program[pc++];

        if (op < 8) {
            stack[sp++] = cols[op][gid];
            continue;
        }

        if (op == OP_LOAD_SCALAR) {
            stack[sp++] = as_type<float>(
                uint(program[pc]) | (uint(program[pc+1]) << 8) |
                (uint(program[pc+2]) << 16) | (uint(program[pc+3]) << 24)
            );
            pc += 4;
            continue;
        }

        if (op >= 16 && op < 32) {
            float b = stack[--sp];
            float a = stack[--sp];
            switch (op) {
                case OP_ADD: stack[sp++] = a + b; break;
                case OP_SUB: stack[sp++] = a - b; break;
                case OP_MUL: stack[sp++] = a * b; break;
                case OP_DIV: stack[sp++] = a / b; break;
                case OP_MOD: stack[sp++] = fmod(a, b); break;
                case OP_EQ:  stack[sp++] = (a == b) ? 1.0f : 0.0f; break;
                case OP_NE:  stack[sp++] = (a != b) ? 1.0f : 0.0f; break;
                case OP_LT:  stack[sp++] = (a < b)  ? 1.0f : 0.0f; break;
                case OP_LE:  stack[sp++] = (a <= b) ? 1.0f : 0.0f; break;
                case OP_GT:  stack[sp++] = (a > b)  ? 1.0f : 0.0f; break;
                case OP_GE:  stack[sp++] = (a >= b) ? 1.0f : 0.0f; break;
            }
            continue;
        }

        if (op >= 32) {
            float a = stack[sp - 1];
            switch (op) {
                case OP_ABS:   stack[sp-1] = abs(a);   break;
                case OP_NEG:   stack[sp-1] = -a;       break;
                case OP_SQRT:  stack[sp-1] = sqrt(a);  break;
                case OP_EXP:   stack[sp-1] = exp(a);   break;
                case OP_LOG:   stack[sp-1] = log(a);   break;
                case OP_CEIL:  stack[sp-1] = ceil(a);  break;
                case OP_FLOOR: stack[sp-1] = floor(a); break;
            }
        }
    }

    return stack[0];
}

// Macro to generate a fused expression-evaluate + reduce kernel.
// Each thread evaluates the expression for its element, then the
// threadgroup cooperatively reduces to a single partial result.
#define EVAL_REDUCE_KERNEL(name, reduce_op, identity)                           \
kernel void name(                                                               \
    device const float* col0 [[buffer(0)]],                                     \
    device const float* col1 [[buffer(1)]],                                     \
    device const float* col2 [[buffer(2)]],                                     \
    device const float* col3 [[buffer(3)]],                                     \
    device const float* col4 [[buffer(4)]],                                     \
    device const float* col5 [[buffer(5)]],                                     \
    device const float* col6 [[buffer(6)]],                                     \
    device const float* col7 [[buffer(7)]],                                     \
    device float* partials     [[buffer(8)]],                                   \
    constant uint8_t* program  [[buffer(9)]],                                   \
    constant uint& prog_len    [[buffer(10)]],                                  \
    constant uint& data_len    [[buffer(11)]],                                  \
    threadgroup float* shared  [[threadgroup(0)]],                              \
    uint tid       [[thread_position_in_threadgroup]],                          \
    uint gid       [[thread_position_in_grid]],                                 \
    uint group_id  [[threadgroup_position_in_grid]],                            \
    uint group_size [[threads_per_threadgroup]]                                 \
) {                                                                             \
    device const float* cols[8] = {col0, col1, col2, col3, col4, col5, col6, col7}; \
                                                                                \
    float val = identity;                                                       \
    if (gid < data_len) {                                                       \
        val = eval_program(cols, program, prog_len, gid);                       \
    }                                                                           \
                                                                                \
    shared[tid] = val;                                                          \
    threadgroup_barrier(mem_flags::mem_threadgroup);                             \
                                                                                \
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {             \
        if (tid < stride) {                                                     \
            shared[tid] = reduce_op(shared[tid], shared[tid + stride]);         \
        }                                                                       \
        threadgroup_barrier(mem_flags::mem_threadgroup);                         \
    }                                                                           \
                                                                                \
    if (tid == 0) {                                                             \
        partials[group_id] = shared[0];                                         \
    }                                                                           \
}

// Reduce op lambdas
inline float reduce_sum(float a, float b) { return a + b; }
inline float reduce_min(float a, float b) { return a < b ? a : b; }
inline float reduce_max(float a, float b) { return a > b ? a : b; }

EVAL_REDUCE_KERNEL(eval_reduce_sum_f32, reduce_sum,  0.0f)
EVAL_REDUCE_KERNEL(eval_reduce_min_f32, reduce_min,  INFINITY)
EVAL_REDUCE_KERNEL(eval_reduce_max_f32, reduce_max, -INFINITY)
