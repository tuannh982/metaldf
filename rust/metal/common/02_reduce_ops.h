// Reduction operation structs with SIMD intrinsics.
// Each op provides: identity value, binary apply, and hardware simd_reduce.

template <typename T>
struct SumOp {
    static constant constexpr T identity = Limits<T>::zero;
    static inline T apply(T a, T b) { return a + b; }
    static inline T simd_reduce(T val) { return simd_sum(val); }
};

template <typename T>
struct MinOp {
    static constant constexpr T identity = Limits<T>::max_val;
    static inline T apply(T a, T b) { return a < b ? a : b; }
    static inline T simd_reduce(T val) { return simd_min(val); }
};

template <typename T>
struct MaxOp {
    static constant constexpr T identity = Limits<T>::min_val;
    static inline T apply(T a, T b) { return a > b ? a : b; }
    static inline T simd_reduce(T val) { return simd_max(val); }
};
