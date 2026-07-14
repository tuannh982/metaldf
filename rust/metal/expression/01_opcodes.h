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
constant uint8_t OP_AND = 40;
constant uint8_t OP_OR  = 41;
constant uint8_t OP_NOT = 42;
constant uint8_t OP_SIN   = 43;
constant uint8_t OP_COS   = 44;
constant uint8_t OP_TAN   = 45;
constant uint8_t OP_ASIN  = 46;
constant uint8_t OP_ACOS  = 47;
constant uint8_t OP_ATAN  = 48;
constant uint8_t OP_SINH  = 49;
constant uint8_t OP_COSH  = 50;
constant uint8_t OP_TANH  = 51;
constant uint8_t OP_LOG2  = 52;
constant uint8_t OP_LOG10 = 53;
constant uint8_t OP_ROUND = 54;
constant uint8_t OP_TRUNC = 55;
constant uint8_t OP_CBRT  = 56;
