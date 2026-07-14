// Null mask helpers — shared by every kernel family that needs to check or
// mutate per-element validity bits. A null mask is a packed bitmask buffer
// (1 bit per element, LSB-first within each byte): bit set = valid, bit
// clear = null. A `nullptr` mask means "no nulls, everything valid" so
// kernels operating on non-nullable columns don't need a branch.

inline bool is_valid(device const uint8_t* mask, uint idx) {
    return mask == nullptr || (mask[idx / 8] & (1u << (idx % 8))) != 0;
}

inline void set_valid(device uint8_t* mask, uint idx) {
    mask[idx / 8] |= (1u << (idx % 8));
}

inline void set_invalid(device uint8_t* mask, uint idx) {
    mask[idx / 8] &= ~(1u << (idx % 8));
}

inline uint null_mask_bytes(uint len) {
    return (len + 7) / 8;
}
