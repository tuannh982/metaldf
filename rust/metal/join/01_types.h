// Hash join types — shared between build/probe kernels.

// Each slot in the join hash table stores the key bits (for comparison)
// and the row index in the build table. Empty slots use SENTINEL key bits.
struct JoinSlot {
    uint key_bits;
    uint row_index;
};

constant constexpr uint JOIN_EMPTY_SENTINEL = 0xFFFFFFFFu;
