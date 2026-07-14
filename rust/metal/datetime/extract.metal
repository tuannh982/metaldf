// Calendar-component extraction kernels — the GPU backing for the `.dt`
// accessor (`.dt.year`/`.month`/`.day`/`.hour`/`.minute`/`.second`/
// `.dayofweek`). Each kernel reads one int64 nanosecond-since-epoch value
// per thread and writes a single int32 component.
//
// All seven kernels are dispatched with `dispatch_thread_groups` (grid
// padded up to a threadgroup multiple), so — unlike the elementwise
// kernels, which use `dispatch_threads` and skip this — each one needs the
// explicit `idx >= len` bounds guard below.
//
// See `01_calendar.h` for `floor_div`/`floor_mod` (why plain `/`/`%` would
// be wrong for pre-epoch/negative timestamps) and `civil_from_days` (the
// day-count -> (year, month, day) conversion used by year/month/day).

kernel void dt_year_i64(device const long* ns [[buffer(0)]],
                        device int* out        [[buffer(1)]],
                        device const uint* len_ptr [[buffer(2)]],
                        uint idx [[thread_position_in_grid]]) {
    if (idx >= *len_ptr) return;
    long days = floor_div(ns[idx], NS_PER_DAY);
    out[idx] = civil_from_days(days).year;
}

kernel void dt_month_i64(device const long* ns [[buffer(0)]],
                         device int* out        [[buffer(1)]],
                         device const uint* len_ptr [[buffer(2)]],
                         uint idx [[thread_position_in_grid]]) {
    if (idx >= *len_ptr) return;
    long days = floor_div(ns[idx], NS_PER_DAY);
    out[idx] = civil_from_days(days).month;
}

kernel void dt_day_i64(device const long* ns [[buffer(0)]],
                       device int* out        [[buffer(1)]],
                       device const uint* len_ptr [[buffer(2)]],
                       uint idx [[thread_position_in_grid]]) {
    if (idx >= *len_ptr) return;
    long days = floor_div(ns[idx], NS_PER_DAY);
    out[idx] = civil_from_days(days).day;
}

kernel void dt_hour_i64(device const long* ns [[buffer(0)]],
                        device int* out        [[buffer(1)]],
                        device const uint* len_ptr [[buffer(2)]],
                        uint idx [[thread_position_in_grid]]) {
    if (idx >= *len_ptr) return;
    out[idx] = int(floor_mod(floor_div(ns[idx], NS_PER_HOUR), 24L));
}

kernel void dt_minute_i64(device const long* ns [[buffer(0)]],
                          device int* out        [[buffer(1)]],
                          device const uint* len_ptr [[buffer(2)]],
                          uint idx [[thread_position_in_grid]]) {
    if (idx >= *len_ptr) return;
    out[idx] = int(floor_mod(floor_div(ns[idx], NS_PER_MIN), 60L));
}

kernel void dt_second_i64(device const long* ns [[buffer(0)]],
                          device int* out        [[buffer(1)]],
                          device const uint* len_ptr [[buffer(2)]],
                          uint idx [[thread_position_in_grid]]) {
    if (idx >= *len_ptr) return;
    out[idx] = int(floor_mod(floor_div(ns[idx], NS_PER_SEC), 60L));
}

// 1970-01-01 (day 0) was a Thursday. Pandas' `.dt.dayofweek` uses
// Monday=0 .. Sunday=6, so shifting day 0 (Thursday, would be weekday 3 in
// a Monday=0 scheme) forward by 3 and taking floor_mod 7 lines Monday up
// with 0: `floor_mod(days + 3, 7)`.
kernel void dt_dayofweek_i64(device const long* ns [[buffer(0)]],
                             device int* out        [[buffer(1)]],
                             device const uint* len_ptr [[buffer(2)]],
                             uint idx [[thread_position_in_grid]]) {
    if (idx >= *len_ptr) return;
    long days = floor_div(ns[idx], NS_PER_DAY);
    out[idx] = int(floor_mod(days + 3, 7L));
}
