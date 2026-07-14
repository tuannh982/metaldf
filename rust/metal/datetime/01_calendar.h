// Calendar helpers for GPU-side datetime extraction (Task 4).
//
// `MetalSeries` datetime columns are (or will be — see the TODO in
// `rust/src/kernels/datetime.rs`) stored as raw int64 nanoseconds since the
// Unix epoch (1970-01-01T00:00:00 UTC). Extracting calendar fields
// (year/month/day/...) on the GPU therefore comes down to pure integer
// arithmetic on that int64 value: no floating point, no timezone handling
// (everything here is UTC).
//
// Two subtleties this header exists to handle correctly:
//
// 1. Floor vs. truncating division. MSL's `/` and `%` on signed integers
//    truncate toward zero (like C), not toward negative infinity. For
//    pre-1970 timestamps (negative nanosecond counts — e.g. dates before
//    the epoch), truncating division gives the wrong day/component for any
//    timestamp that isn't exactly midnight UTC. `floor_div`/`floor_mod`
//    below implement floor semantics (matching Python's `//`/`%` and
//    pandas' own datetime component extraction) so negative timestamps
//    resolve to the calendar date/time an observer would actually read off
//    a clock, not one that's off by a day/component.
//
// 2. Civil (Gregorian) calendar reconstruction from a day count. Turning a
//    "days since epoch" integer into a (year, month, day) triple correctly
//    — including leap years, the Julian/Gregorian leap-year exception every
//    100 (but not 400) years, and negative day counts for pre-1970 dates —
//    is a solved problem: this is Howard Hinnant's `civil_from_days`
//    algorithm (http://howardhinnant.github.io/date_algorithms.html),
//    which is branch-free, uses only integer arithmetic, and is valid for
//    the entire range representable by a 64-bit day count.

constant long NS_PER_SEC  = 1000000000L;
constant long NS_PER_MIN  = 60L * NS_PER_SEC;
constant long NS_PER_HOUR = 3600L * NS_PER_SEC;
constant long NS_PER_DAY  = 86400L * NS_PER_SEC;

struct CivilDate { int year; int month; int day; };

// Floor division: rounds toward negative infinity (unlike MSL's `/`, which
// truncates toward zero). Matches Python's `//` and pandas' datetime
// component semantics for negative `a` (pre-epoch nanosecond counts).
inline long floor_div(long a, long b) {
    return (a >= 0) ? a / b : (a - b + 1) / b;
}

// Floor modulo: result always has the same sign as `b` (here, always
// non-negative for positive `b`), unlike MSL's `%`, which can return a
// negative remainder for negative `a`.
inline long floor_mod(long a, long b) {
    long r = a % b;
    return (r < 0) ? r + b : r;
}

// Howard Hinnant's `civil_from_days`: converts `z`, a signed day count
// relative to the Unix epoch (1970-01-01 = day 0), into a proleptic
// Gregorian (year, month, day) triple. Valid for the entire int64 range of
// `z`. See http://howardhinnant.github.io/date_algorithms.html for the
// derivation.
inline CivilDate civil_from_days(long z) {
    z += 719468;
    long era = (z >= 0 ? z : z - 146096) / 146097;
    uint doe = uint(z - era * 146097);
    uint yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;
    long y = long(yoe) + era * 400;
    uint doy = doe - (365*yoe + yoe/4 - yoe/100);
    uint mp = (5*doy + 2) / 153;
    uint d = doy - (153*mp + 2) / 5 + 1;
    uint m = mp + (mp < 10 ? 3 : -9);
    y += (m <= 2);
    return CivilDate{int(y), int(m), int(d)};
}
