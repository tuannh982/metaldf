// Embeds Metal shader source files into the Rust binary as string constants,
// and (when possible) pre-compiles the standard kernel families into
// `.metallib` binaries at build time for faster cold starts.
//
// Metal kernels are normally compiled at RUNTIME by Apple's GPU driver via
// device.new_library_with_source(). That's ~1-5ms per pipeline the first
// time it's needed. When `xcrun metal` is available (requires a full Xcode
// install, not just Command Line Tools), we additionally compile each
// kernel family's MSL source -> .air -> .metallib ahead of time and embed
// the resulting binary via include_bytes!. At runtime, kernels/mod.rs tries
// `device.new_library_with_data()` first (near-instant) and falls back to
// `new_library_with_source()` if that fails for any reason (e.g. GPU family
// mismatch, missing metallib).
//
// Directory layout:
//   metal/common/   → COMMON_PREAMBLE_SRC  (prepended to every library)
//   metal/sort/     → SORT_METAL_SRC (+ SORT_METALLIB when pre-compiled)
//   metal/reduction/→ REDUCTION_METAL_SRC (+ REDUCTION_METALLIB)
//   metal/groupby/  → GROUPBY_METAL_SRC (+ GROUPBY_METALLIB)
//   metal/test/     → TEST_DEBUG_METAL_SRC  (never pre-compiled — debug only)
//
// Adding a new .metal or .h file to any directory is automatically picked up.
// Adding a new directory generates a new {DIR}_METAL_SRC constant (and an
// accompanying {DIR}_METALLIB pre-compilation attempt).

use std::fs;
use std::path::Path;
use std::process::Command;

/// Conservative default GPU tuning #defines used only for build-time metallib
/// pre-compilation. These mirror the fallback branch of `GpuTuning::for_family`
/// in `src/kernels/mod.rs`. The actual runtime GPU family isn't known at build
/// time, so we bake in the safe/conservative values here; if the GPU present
/// at runtime would benefit from better tuning, `load_library` falls back to
/// runtime source compilation (with the correct defines for that GPU) anyway.
const DEFAULT_TUNING_DEFINES: &str = "\
#define REDUCE_THREADGROUP_SIZE 256\n\
#define REDUCE_N_READS 4\n\
#define LOCAL_SORT_SIZE 256\n\
#define LOCAL_SORT_STAGES 8\n";

/// Read all .metal and .h files under `dir` (recursively), return their
/// contents concatenated in alphabetical order. Use numeric prefixes
/// (01_, 02_) on header files to enforce dependency ordering.
fn read_all_sources(dir: &Path) -> String {
    let mut files = Vec::new();
    find_source_files(dir, &mut files);
    files.sort();

    let mut combined = String::new();
    for path in &files {
        combined.push_str(&fs::read_to_string(path).unwrap_or_else(|_| panic!("Failed to read {}", path.display())));
        combined.push('\n');
        println!("cargo:rerun-if-changed={}", path.display());
    }
    combined
}

fn find_source_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let mut entries: Vec<_> = fs::read_dir(dir)
        .unwrap_or_else(|_| panic!("Failed to read {}", dir.display()))
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            find_source_files(&path, out);
        } else if let Some(ext) = path.extension() {
            if ext == "metal" || ext == "h" {
                out.push(path);
            }
        }
    }
}

/// Write a Rust source file that defines `pub const {name}: &str = "..."`.
fn write_string_constant(out_dir: &str, name: &str, source: &str) {
    let escaped = source.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
    let code = format!(r#"pub const {name}: &str = "{escaped}";"#);
    let path = Path::new(out_dir).join(format!("{}.rs", name.to_lowercase()));
    fs::write(&path, code).unwrap_or_else(|_| panic!("Failed to write {}", path.display()));
}

/// Write `pub const {NAME}_METALLIB: Option<&[u8]> = None;` — used whenever
/// pre-compilation isn't possible or fails, so callers always have a
/// `{name}_metallib.rs` file to include regardless of build-machine capability.
fn write_none_metallib(out_dir: &str, name: &str) {
    let code = format!("pub const {}_METALLIB: Option<&[u8]> = None;", name.to_uppercase());
    let rs_path = Path::new(out_dir).join(format!("{}_metallib.rs", name.to_lowercase()));
    fs::write(&rs_path, code).unwrap_or_else(|_| panic!("Failed to write {}", rs_path.display()));
}

/// Attempt to compile `source` (already including the preamble/defines) into
/// a `.metallib` and embed it as `{NAME}_METALLIB: Option<&[u8]>`. Returns
/// true on success. On any failure (missing Xcode, compile error, link
/// error), writes a `None` constant instead so the runtime falls back to
/// source compilation with no behavior change.
fn try_precompile_metallib(out_dir: &str, name: &str, source: &str) -> bool {
    // Check if xcrun metal is available (requires full Xcode, not just
    // Command Line Tools).
    let metal_check = Command::new("xcrun").args(["--find", "metal"]).output();

    if metal_check.is_err() || !metal_check.unwrap().status.success() {
        println!("cargo:warning=xcrun metal not found, skipping pre-compilation for {name}");
        write_none_metallib(out_dir, name);
        return false;
    }

    let source_path = Path::new(out_dir).join(format!("{name}.metal"));
    let air_path = Path::new(out_dir).join(format!("{name}.air"));
    let lib_path = Path::new(out_dir).join(format!("{name}.metallib"));

    // Write the full source (with preamble + conservative tuning defines) to
    // a temp file.
    fs::write(&source_path, source).unwrap_or_else(|_| panic!("Failed to write {}", source_path.display()));

    // Compile .metal → .air
    let compile = Command::new("xcrun")
        .args(["metal", "-c", "-std=metal3.0"])
        .arg(&source_path)
        .arg("-o")
        .arg(&air_path)
        .output();

    let compile = match compile {
        Ok(output) if output.status.success() => output,
        Ok(output) => {
            println!(
                "cargo:warning=Failed to compile {name}.metal to .air, using runtime compilation: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            write_none_metallib(out_dir, name);
            return false;
        }
        Err(e) => {
            println!("cargo:warning=Failed to invoke xcrun metal for {name}: {e}, using runtime compilation");
            write_none_metallib(out_dir, name);
            return false;
        }
    };
    let _ = compile;

    // Link .air → .metallib
    let link = Command::new("xcrun")
        .args(["metallib"])
        .arg(&air_path)
        .arg("-o")
        .arg(&lib_path)
        .output();

    match link {
        Ok(output) if output.status.success() => {}
        Ok(output) => {
            println!(
                "cargo:warning=Failed to link {name}.metallib, using runtime compilation: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            write_none_metallib(out_dir, name);
            return false;
        }
        Err(e) => {
            println!("cargo:warning=Failed to invoke xcrun metallib for {name}: {e}, using runtime compilation");
            write_none_metallib(out_dir, name);
            return false;
        }
    }

    // Generate Rust code to embed the metallib.
    let code = format!(
        "pub const {}_METALLIB: Option<&[u8]> = Some(include_bytes!(concat!(env!(\"OUT_DIR\"), \"/{name}.metallib\")));",
        name.to_uppercase()
    );
    let rs_path = Path::new(out_dir).join(format!("{}_metallib.rs", name.to_lowercase()));
    fs::write(&rs_path, code).unwrap_or_else(|_| panic!("Failed to write {}", rs_path.display()));

    println!("cargo:warning=Pre-compiled {name}.metallib for faster cold start");
    true
}

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let metal_dir = Path::new("metal");

    // common/ is the shared preamble — prepended to every kernel library at runtime
    let common_preamble = read_all_sources(&metal_dir.join("common"));
    write_string_constant(&out_dir, "COMMON_PREAMBLE_SRC", &common_preamble);

    // Each subdirectory (except common/ and test/) becomes a kernel library
    let mut dirs: Vec<_> = fs::read_dir(metal_dir)
        .expect("Failed to read metal/")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir() && e.file_name() != "common" && e.file_name() != "test")
        .collect();
    dirs.sort_by_key(|e| e.file_name());

    for dir in &dirs {
        let name_lower = dir.file_name().to_string_lossy().to_lowercase();
        let name_upper = name_lower.to_uppercase();
        let kernel_source = read_all_sources(&dir.path());
        write_string_constant(&out_dir, &format!("{name_upper}_METAL_SRC"), &kernel_source);

        // Best-effort build-time pre-compilation for faster cold starts.
        // Uses conservative default tuning constants (DEFAULT_TUNING_DEFINES)
        // since the actual runtime GPU family isn't known at build time, and
        // is compiled WITHOUT METALDF_DEBUG (debug mode always uses runtime
        // source compilation, see kernels/mod.rs). If `xcrun metal` isn't
        // available (e.g. only Command Line Tools installed, no full Xcode),
        // this is skipped and {NAME}_METALLIB becomes `None` — the runtime
        // falls back to the existing source-compilation path, so there is no
        // behavior change, only a possible loss of the cold-start speedup.
        let full_source = format!("{DEFAULT_TUNING_DEFINES}{common_preamble}\n{kernel_source}");
        try_precompile_metallib(&out_dir, &name_lower, &full_source);
    }

    // test/ kernels — always compiled at runtime with debug options, never pre-compiled
    let test_dir = metal_dir.join("test");
    if test_dir.exists() {
        write_string_constant(&out_dir, "TEST_DEBUG_METAL_SRC", &read_all_sources(&test_dir));
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=metal");
}
