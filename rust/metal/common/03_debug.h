// debug.metal
// Compile-time on/off shader logging for Metal kernels.
// Uses Metal 3.2 os_log API. When METALDF_DEBUG is not defined, macros compile to no-ops.

#ifdef METALDF_DEBUG

#define METAL_LOG(gid, fmt, ...) \
    metal::os_log_default.log_info(fmt, __VA_ARGS__)

#define METAL_LOG_IF(gid, max_threads, fmt, ...) \
    if ((gid) < (max_threads)) { \
        metal::os_log_default.log_info(fmt, __VA_ARGS__); \
    }

#else

#define METAL_LOG(gid, fmt, ...)       ((void)0)
#define METAL_LOG_IF(gid, max, fmt, ...) ((void)0)

#endif
