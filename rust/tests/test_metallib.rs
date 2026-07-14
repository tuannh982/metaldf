// Task 8.2: Pre-compiled metallib for faster cold starts.
//
// build.rs best-effort compiles each kernel family's MSL source into a
// `.metallib` at build time (requires `xcrun metal`, i.e. full Xcode, not
// just Command Line Tools) and embeds it as `Option<&[u8]>`. At runtime,
// `kernels::load_library` tries `Device::new_library_with_data` first and
// falls back to `new_library_with_source` on any failure — including the
// common case where pre-compilation was skipped entirely (`None`).
//
// These tests verify that invariant: kernel libraries must load correctly
// regardless of which machine built the crate, and any embedded metallib
// that IS present must actually be valid.

use metal::{Device, Library};
use metaldf_engine::backend::MetalBackend;
use metaldf_engine::kernels::{
    load_elementwise_library, load_expression_library, load_filter_library, load_groupby_library,
    load_join_library, load_reductions_library, load_rolling_library, load_scan_library,
    load_sort_library, load_strings_library, EXPRESSION_METALLIB, FILTER_METALLIB,
    GROUPBY_METALLIB, JOIN_METALLIB, ELEMENTWISE_METALLIB, REDUCTION_METALLIB, ROLLING_METALLIB,
    SCAN_METALLIB, SORT_METALLIB, STRINGS_METALLIB,
};

const LOADERS: &[(&str, fn(&Device) -> Result<Library, String>)] = &[
    ("elementwise", load_elementwise_library),
    ("reductions", load_reductions_library),
    ("sort", load_sort_library),
    ("groupby", load_groupby_library),
    ("strings", load_strings_library),
    ("expression", load_expression_library),
    ("scan", load_scan_library),
    ("filter", load_filter_library),
    ("join", load_join_library),
    ("rolling", load_rolling_library),
];

/// Every kernel family must load successfully whether or not a pre-compiled
/// `.metallib` was embedded at build time. This is the core invariant of
/// Task 8.2: pre-compilation is a pure cold-start optimization with a
/// transparent fallback, never a behavior change.
#[test]
fn test_all_kernel_libraries_load() {
    let device = MetalBackend::device().expect("Metal device required for this test");

    for (name, loader) in LOADERS {
        let result = loader(device);
        assert!(result.is_ok(), "{name} library failed to load: {:?}", result.err());
    }
}

/// Any embedded pre-compiled metallib must actually be loadable via
/// `new_library_with_data` — otherwise it's dead weight in the binary that
/// always falls through to source compilation anyway.
#[test]
fn test_precompiled_metallibs_are_valid_when_present() {
    let device = MetalBackend::device().expect("Metal device required for this test");

    let metallibs: &[(&str, Option<&[u8]>)] = &[
        ("elementwise", ELEMENTWISE_METALLIB),
        ("reduction", REDUCTION_METALLIB),
        ("sort", SORT_METALLIB),
        ("groupby", GROUPBY_METALLIB),
        ("strings", STRINGS_METALLIB),
        ("expression", EXPRESSION_METALLIB),
        ("scan", SCAN_METALLIB),
        ("filter", FILTER_METALLIB),
        ("join", JOIN_METALLIB),
        ("rolling", ROLLING_METALLIB),
    ];

    let mut precompiled_count = 0usize;
    for (name, data) in metallibs {
        if let Some(bytes) = data {
            precompiled_count += 1;
            let lib = device.new_library_with_data(bytes);
            assert!(lib.is_ok(), "pre-compiled {name}.metallib failed to load: {:?}", lib.err());
        }
    }

    // Informational only: 0 is expected on machines without full Xcode
    // (only Command Line Tools installed — `xcrun metal` unavailable), and
    // up to 10 when pre-compilation succeeds for every kernel family. Both
    // are valid, supported states; this test asserts they're internally
    // consistent, not which one applies on the current machine.
    println!("Pre-compiled metallibs available: {precompiled_count}/{}", metallibs.len());
}
