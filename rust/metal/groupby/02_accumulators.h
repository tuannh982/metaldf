// Accumulator storage traits for groupby aggregation.
// Maps (T, Op) to the atomic type backing per-group accumulators.

template <typename T> struct SumAccum;
template <> struct SumAccum<float> {
    using AtomicT = atomic_float;
    static inline void accumulate(device atomic_float* addr, float val) {
        atomic_fetch_add_explicit(addr, val, memory_order_relaxed);
    }
};
template <> struct SumAccum<int> {
    using AtomicT = atomic_int;
    static inline void accumulate(device atomic_int* addr, int val) {
        atomic_fetch_add_explicit(addr, val, memory_order_relaxed);
    }
};

template <typename T> struct MinMaxStorage;
template <> struct MinMaxStorage<float> {
    using AtomicT = atomic_uint;
    using RawT = uint;
    static inline float load(uint bits) { return as_type<float>(bits); }
};
template <> struct MinMaxStorage<int> {
    using AtomicT = atomic_int;
    using RawT = int;
    static inline int load(int v) { return v; }
};

template <typename T> struct MinAccum;
template <> struct MinAccum<float> {
    using AtomicT = MinMaxStorage<float>::AtomicT;
    static inline void accumulate(device atomic_uint* addr, float val) {
        uint val_bits = as_type<uint>(val);
        uint expected = atomic_load_explicit(addr, memory_order_relaxed);
        while (true) {
            float current = as_type<float>(expected);
            if (current <= val) break;
            if (atomic_compare_exchange_weak_explicit(
                addr, &expected, val_bits,
                memory_order_relaxed, memory_order_relaxed)) break;
        }
    }
};
template <> struct MinAccum<int> {
    using AtomicT = MinMaxStorage<int>::AtomicT;
    static inline void accumulate(device atomic_int* addr, int val) {
        atomic_fetch_min_explicit(addr, val, memory_order_relaxed);
    }
};

template <typename T> struct MaxAccum;
template <> struct MaxAccum<float> {
    using AtomicT = MinMaxStorage<float>::AtomicT;
    static inline void accumulate(device atomic_uint* addr, float val) {
        uint val_bits = as_type<uint>(val);
        uint expected = atomic_load_explicit(addr, memory_order_relaxed);
        while (true) {
            float current = as_type<float>(expected);
            if (current >= val) break;
            if (atomic_compare_exchange_weak_explicit(
                addr, &expected, val_bits,
                memory_order_relaxed, memory_order_relaxed)) break;
        }
    }
};
template <> struct MaxAccum<int> {
    using AtomicT = MinMaxStorage<int>::AtomicT;
    static inline void accumulate(device atomic_int* addr, int val) {
        atomic_fetch_max_explicit(addr, val, memory_order_relaxed);
    }
};

// Hash key encoding via RadixTraits bijection (avoids -1 sentinel collision).
template <typename T> inline T decode_hash_key(typename KeyBits<T>::type k);
template <> inline float decode_hash_key<float>(uint k) {
    uint u = (k & 0x80000000u) ? (k ^ 0x80000000u) : (~k);
    return as_type<float>(u);
}
template <> inline int decode_hash_key<int>(uint k) {
    return as_type<int>(k ^ 0x80000000u);
}
