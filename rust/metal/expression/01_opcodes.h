// Bytecode opcodes shared by every expression-evaluating kernel in this
// directory (eval.metal, eval_reduce.metal, ...). Lives in a numeric-prefix
// header so build.rs's alphabetical concatenation places it before any
// sibling .metal file -- see rust/build.rs's module comment.
constant uint8_t OP_LOAD_SCALAR = 8;
constant uint8_t OP_ADD = 16;
constant uint8_t OP_SUB = 17;
constant uint8_t OP_MUL = 18;
constant uint8_t OP_DIV = 19;
constant uint8_t OP_MOD = 20;
constant uint8_t OP_EQ  = 24;
constant uint8_t OP_NE  = 25;
constant uint8_t OP_LT  = 26;
constant uint8_t OP_LE  = 27;
constant uint8_t OP_GT  = 28;
constant uint8_t OP_GE  = 29;
constant uint8_t OP_ABS   = 32;
constant uint8_t OP_NEG   = 33;
constant uint8_t OP_SQRT  = 34;
constant uint8_t OP_EXP   = 35;
constant uint8_t OP_LOG   = 36;
constant uint8_t OP_CEIL  = 37;
constant uint8_t OP_FLOOR = 38;
