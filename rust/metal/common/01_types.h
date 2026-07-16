#include <metal_stdlib>
using namespace metal;

// --- Numeric Limits ---
template <typename T> struct Limits;

template <> struct Limits<float> {
    static constant constexpr float max_val =  __builtin_inff();
    static constant constexpr float min_val = -__builtin_inff();
    static constant constexpr float zero = 0.0f;
};
template <> struct Limits<int> {
    static constant constexpr int max_val =  2147483647;
    static constant constexpr int min_val = -2147483648;
    static constant constexpr int zero = 0;
};
template <> struct Limits<uint> {
    static constant constexpr uint max_val = 0xFFFFFFFFu;
    static constant constexpr uint min_val = 0;
    static constant constexpr uint zero = 0;
};
template <> struct Limits<long> {
    static constant constexpr long max_val =  9223372036854775807L;
    static constant constexpr long min_val = -9223372036854775807L - 1;
    static constant constexpr long zero = 0L;
};
template <> struct Limits<char> {
    static constant constexpr char max_val = 127;
    static constant constexpr char min_val = -128;
    static constant constexpr char zero = 0;
};
template <> struct Limits<short> {
    static constant constexpr short max_val = 32767;
    static constant constexpr short min_val = -32768;
    static constant constexpr short zero = 0;
};
template <> struct Limits<uchar> {
    static constant constexpr uchar max_val = 255;
    static constant constexpr uchar min_val = 0;
    static constant constexpr uchar zero = 0;
};
template <> struct Limits<ushort> {
    static constant constexpr ushort max_val = 65535;
    static constant constexpr ushort min_val = 0;
    static constant constexpr ushort zero = 0;
};
template <> struct Limits<ulong> {
    static constant constexpr ulong max_val = 0xFFFFFFFFFFFFFFFFuL;
    static constant constexpr ulong min_val = 0;
    static constant constexpr ulong zero = 0uL;
};

// --- Radix Sort Key Conversion ---
template <typename T> struct RadixTraits;

template <> struct RadixTraits<float> {
    using KeyT = uint;
    static constant constexpr uint num_passes = 4;
    static inline KeyT to_key(float f) {
        uint u = as_type<uint>(f);
        return (u & 0x80000000u) ? (~u) : (u ^ 0x80000000u);
    }
};
template <> struct RadixTraits<int> {
    using KeyT = uint;
    static constant constexpr uint num_passes = 4;
    static inline KeyT to_key(int i) {
        return as_type<uint>(i) ^ 0x80000000u;
    }
};
template <> struct RadixTraits<long> {
    using KeyT = ulong;
    static constant constexpr uint num_passes = 8;
    static inline KeyT to_key(long i) {
        return as_type<ulong>(i) ^ 0x8000000000000000uL;
    }
};
template <> struct RadixTraits<char> {
    using KeyT = uchar;
    static constant constexpr uint num_passes = 1;
    static inline KeyT to_key(char i) {
        return as_type<uchar>(i) ^ 0x80u;
    }
};
template <> struct RadixTraits<short> {
    using KeyT = ushort;
    static constant constexpr uint num_passes = 2;
    static inline KeyT to_key(short i) {
        return as_type<ushort>(i) ^ 0x8000u;
    }
};
template <> struct RadixTraits<uchar> {
    using KeyT = uchar;
    static constant constexpr uint num_passes = 1;
    static inline KeyT to_key(uchar i) { return i; }
};
template <> struct RadixTraits<ushort> {
    using KeyT = ushort;
    static constant constexpr uint num_passes = 2;
    static inline KeyT to_key(ushort i) { return i; }
};
template <> struct RadixTraits<uint> {
    using KeyT = uint;
    static constant constexpr uint num_passes = 4;
    static inline KeyT to_key(uint i) { return i; }
};
template <> struct RadixTraits<ulong> {
    using KeyT = ulong;
    static constant constexpr uint num_passes = 8;
    static inline KeyT to_key(ulong i) { return i; }
};

