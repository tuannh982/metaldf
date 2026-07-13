#include <metal_stdlib>
using namespace metal;

// Opcodes are defined in 01_opcodes.h, concatenated before this file.

kernel void eval_expression_f32(
    device const float* col0 [[buffer(0)]],
    device const float* col1 [[buffer(1)]],
    device const float* col2 [[buffer(2)]],
    device const float* col3 [[buffer(3)]],
    device const float* col4 [[buffer(4)]],
    device const float* col5 [[buffer(5)]],
    device const float* col6 [[buffer(6)]],
    device const float* col7 [[buffer(7)]],
    device float* output     [[buffer(8)]],
    constant uint8_t* program [[buffer(9)]],
    constant uint& prog_len  [[buffer(10)]],
    uint idx [[thread_position_in_grid]]
) {
    device const float* cols[8] = {col0, col1, col2, col3, col4, col5, col6, col7};

    float stack[8];
    int sp = 0;

    for (uint pc = 0; pc < prog_len; ) {
        uint8_t op = program[pc++];

        if (op < 8) {
            stack[sp++] = cols[op][idx];
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

    output[idx] = stack[0];
}
