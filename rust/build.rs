// Embeds Metal shader source files into the Rust binary as string constants.
//
// Metal kernels are compiled at RUNTIME by Apple's GPU driver via
// device.new_library_with_source(). This script just reads the .metal/.h
// files and packs their text into Rust constants so we don't need to
// ship loose files.
//
// Directory layout:
//   metal/common/   → COMMON_PREAMBLE_SRC  (prepended to every library)
//   metal/sort/     → SORT_METAL_SRC
//   metal/reduction/→ REDUCTION_METAL_SRC
//   metal/groupby/  → GROUPBY_METAL_SRC
//   metal/test/     → TEST_DEBUG_METAL_SRC
//
// Adding a new .metal or .h file to any directory is automatically picked up.
// Adding a new directory generates a new {DIR}_METAL_SRC constant.

use std::fs;
use std::path::Path;

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

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let metal_dir = Path::new("metal");

    // common/ is the shared preamble — prepended to every kernel library at runtime
    write_string_constant(&out_dir, "COMMON_PREAMBLE_SRC", &read_all_sources(&metal_dir.join("common")));

    // Each subdirectory (except common/ and test/) becomes a kernel library
    let mut dirs: Vec<_> = fs::read_dir(metal_dir)
        .expect("Failed to read metal/")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir() && e.file_name() != "common" && e.file_name() != "test")
        .collect();
    dirs.sort_by_key(|e| e.file_name());

    for dir in &dirs {
        let name = dir.file_name().to_string_lossy().to_uppercase();
        write_string_constant(&out_dir, &format!("{name}_METAL_SRC"), &read_all_sources(&dir.path()));
    }

    // test/ kernels
    let test_dir = metal_dir.join("test");
    if test_dir.exists() {
        write_string_constant(&out_dir, "TEST_DEBUG_METAL_SRC", &read_all_sources(&test_dir));
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=metal");
}
