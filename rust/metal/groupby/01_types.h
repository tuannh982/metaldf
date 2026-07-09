// Groupby-specific type traits.

// Hash table empty-slot sentinels.
template <typename T> struct HashSentinel;
template <> struct HashSentinel<uint>  { static constant constexpr uint  value = 0xFFFFFFFFu; };
template <> struct HashSentinel<ulong> { static constant constexpr ulong value = 0xFFFFFFFFFFFFFFFFuL; };

// Maps value type to the unsigned type used to store key bits in hash tables.
template <typename T> struct KeyBits;
template <> struct KeyBits<float> { using type = uint; };
template <> struct KeyBits<int>   { using type = uint; };
template <> struct KeyBits<long>  { using type = ulong; };
