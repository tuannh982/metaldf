// Per-type operation helpers where the expression differs by type.

#include <metal_stdlib>
using namespace metal;

template <typename T> inline T mod_op(T a, T b);
template <> inline float mod_op<float>(float a, float b) { return fmod(a, b); }
template <> inline int   mod_op<int>(int a, int b)       { return a % b; }
template <> inline long  mod_op<long>(long a, long b)    { return a % b; }
