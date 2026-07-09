use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use metal::{Device, ComputePipelineState, Library};
use metal::foreign_types::ForeignType;
use metal::objc::{msg_send, sel, sel_impl};

use crate::backend::MetalBackend;

static DEBUG_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn is_debug_enabled() -> bool {
    DEBUG_ENABLED.load(Ordering::SeqCst)
}

// ---------------------------------------------------------------------------
// GPU family detection + per-family tuning
// ---------------------------------------------------------------------------

pub(crate) fn detect_gpu_family(device: &Device) -> &'static str {
    let families: &[(isize, &str)] = &[
        (1009, "apple9"), (1008, "apple8"), (1007, "apple7"),
        (1006, "apple6"), (1005, "apple5"), (1004, "apple4"),
        (1003, "apple3"),
    ];
    for &(value, name) in families {
        let supported: bool = unsafe {
            msg_send![device.as_ptr() as *mut metal::objc::runtime::Object, supportsFamily: value]
        };
        if supported { return name; }
    }
    "unknown"
}

/// Per-GPU-family tuning constants. Used both as #defines in Metal source
/// and as dispatch parameters in Rust.
pub struct GpuTuning {
    /// Threads per threadgroup for reduction kernels.
    pub reduce_threadgroup_size: u64,
    /// Elements each thread loads in a reduction kernel.
    pub reduce_n_reads: u64,
    /// Threads (= elements) in a bitonic local sort block.
    pub local_sort_size: u64,
    /// log2(local_sort_size) — number of stages done in threadgroup memory.
    pub local_sort_stages: u32,
}

impl GpuTuning {
    pub fn elements_per_reduce_group(&self) -> u64 {
        self.reduce_threadgroup_size * self.reduce_n_reads
    }

    fn for_family(family: &str) -> Self {
        match family {
            // Apple4+ all support 1024 threads/threadgroup, 32KB shared memory
            "apple9" | "apple8" | "apple7" | "apple6" | "apple5" | "apple4" => Self {
                reduce_threadgroup_size: 1024,
                reduce_n_reads: 8,
                local_sort_size: 1024,
                local_sort_stages: 10,
            },
            // Older or unknown GPUs — conservative defaults
            _ => Self {
                reduce_threadgroup_size: 256,
                reduce_n_reads: 4,
                local_sort_size: 256,
                local_sort_stages: 8,
            },
        }
    }

    fn metal_defines(&self) -> String {
        format!(
            "#define REDUCE_THREADGROUP_SIZE {}\n\
             #define REDUCE_N_READS {}\n\
             #define LOCAL_SORT_SIZE {}\n\
             #define LOCAL_SORT_STAGES {}\n",
            self.reduce_threadgroup_size,
            self.reduce_n_reads,
            self.local_sort_size,
            self.local_sort_stages,
        )
    }
}

lazy_static::lazy_static! {
    static ref GPU_TUNING: GpuTuning = {
        let device = MetalBackend::device()
            .expect("Metal device required for GPU tuning");
        GpuTuning::for_family(detect_gpu_family(device))
    };
}

pub fn tuning() -> &'static GpuTuning {
    &GPU_TUNING
}

pub mod reductions;
pub mod sort;
pub mod groupby;
pub mod strings;

include!(concat!(env!("OUT_DIR"), "/common_preamble_src.rs"));
include!(concat!(env!("OUT_DIR"), "/reduction_metal_src.rs"));
include!(concat!(env!("OUT_DIR"), "/sort_metal_src.rs"));
include!(concat!(env!("OUT_DIR"), "/groupby_metal_src.rs"));
include!(concat!(env!("OUT_DIR"), "/strings_metal_src.rs"));
include!(concat!(env!("OUT_DIR"), "/test_debug_metal_src.rs"));

lazy_static::lazy_static! {
    static ref PIPELINE_CACHE: Mutex<HashMap<String, ComputePipelineState>> =
        Mutex::new(HashMap::new());
}

pub fn set_debug_enabled(enabled: bool) {
    DEBUG_ENABLED.store(enabled, Ordering::SeqCst);
    let mut cache = PIPELINE_CACHE.lock().unwrap();
    cache.clear();
}

fn with_preamble(source: &str) -> String {
    let mut result = String::new();
    if is_debug_enabled() {
        result.push_str("#define METALDF_DEBUG\n");
    }
    result.push_str(&tuning().metal_defines());
    result.push_str(COMMON_PREAMBLE_SRC);
    result.push('\n');
    result.push_str(source);
    result
}

fn create_compile_options() -> metal::CompileOptions {
    let options = metal::CompileOptions::new();
    if is_debug_enabled() {
        options.set_fast_math_enabled(false);
        unsafe {
            let ptr = options.as_ptr() as *mut metal::objc::runtime::Object;
            // Metal 3.2 (0x30002) + enableLogging required for os_log in shaders
            let _: () = metal::objc::msg_send![ptr, setLanguageVersion: 0x30002u64];
            let _: () = metal::objc::msg_send![ptr, setEnableLogging: metal::objc::runtime::YES];
        }
    }
    options
}

fn load_library(device: &Device, name: &str, source: &str) -> Result<Library, String> {
    let options = create_compile_options();
    let full_source = with_preamble(source);
    device.new_library_with_source(&full_source, &options)
        .map_err(|e| format!("Failed to compile {name} MSL: {:?}", e))
}

pub fn load_reductions_library(device: &Device) -> Result<Library, String> {
    load_library(device, "reductions", REDUCTION_METAL_SRC)
}

pub fn load_sort_library(device: &Device) -> Result<Library, String> {
    load_library(device, "sort", SORT_METAL_SRC)
}

pub fn load_groupby_library(device: &Device) -> Result<Library, String> {
    load_library(device, "groupby", GROUPBY_METAL_SRC)
}

pub fn load_strings_library(device: &Device) -> Result<Library, String> {
    load_library(device, "strings", STRINGS_METAL_SRC)
}

pub fn get_pipeline_state(
    device: &Device,
    library: &Library,
    kernel_name: &str,
) -> Result<ComputePipelineState, String> {
    let mut cache = PIPELINE_CACHE.lock().unwrap();
    if let Some(state) = cache.get(kernel_name) {
        return Ok(state.clone());
    }

    let function = library.get_function(kernel_name, None)
        .map_err(|e| format!("Kernel function '{}' not found in library: {}", kernel_name, e))?;

    let pipeline = device.new_compute_pipeline_state_with_function(&function)
        .map_err(|e| format!("Failed to create pipeline state for '{}': {:?}", kernel_name, e))?;

    cache.insert(kernel_name.to_string(), pipeline.clone());
    Ok(pipeline)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::MetalBackend;

    #[test]
    fn test_debug_printf_compiles_and_runs() {
        let device = MetalBackend::device().expect("No Metal device");
        let queue = MetalBackend::queue().expect("No Metal queue");

        let debug_source = format!("#define METALDF_DEBUG\n{}\n{}", COMMON_PREAMBLE_SRC, TEST_DEBUG_METAL_SRC);
        let library = device.new_library_with_source(&debug_source, &metal::CompileOptions::new())
            .expect("Failed to compile test_debug.metal with debug enabled");

        let func = library.get_function("test_debug_printf", None)
            .expect("test_debug_printf kernel not found");
        let pipeline = device.new_compute_pipeline_state_with_function(&func)
            .expect("Failed to create pipeline");

        let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let in_buf = device.new_buffer(
            input.len() as u64 * 4,
            metal::MTLResourceOptions::StorageModeShared,
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                in_buf.contents() as *mut f32,
                input.len(),
            );
        }
        let out_buf = device.new_buffer(
            input.len() as u64 * 4,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&in_buf), 0);
        enc.set_buffer(1, Some(&out_buf), 0);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(1, 1, 1),
            metal::MTLSize::new(8, 1, 1),
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        let output: &[f32] = unsafe {
            std::slice::from_raw_parts(out_buf.contents() as *const f32, 8)
        };
        for i in 0..8 {
            assert!(
                (output[i] - (i as f32 + 1.0)).abs() < 1e-6,
                "output[{}] = {}, expected {}",
                i, output[i], i as f32 + 1.0
            );
        }
    }
}
