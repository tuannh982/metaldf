// Per-type operation helpers where the expression differs by type.

#include <metal_stdlib>
using namespace metal;

template <typename T> inline T mod_op(T a, T b);
template <> inline float mod_op<float>(float a, float b) { return fmod(a, b); }
template <> inline int   mod_op<int>(int a, int b)       { return a % b; }
template <> inline long   mod_op<long>(long a, long b)     { return a % b; }
template <> inline char   mod_op<char>(char a, char b)     { return a % b; }
template <> inline short  mod_op<short>(short a, short b)  { return a % b; }
template <> inline uchar  mod_op<uchar>(uchar a, uchar b) { return a % b; }
template <> inline ushort mod_op<ushort>(ushort a, ushort b) { return a % b; }
template <> inline uint   mod_op<uint>(uint a, uint b)     { return a % b; }
template <> inline ulong  mod_op<ulong>(ulong a, ulong b) { return a % b; }
