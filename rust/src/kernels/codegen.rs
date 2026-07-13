// eval_expression_codegen -- an alternative to `eval_expression` (Task 14)
// that "decompiles" the same stack-based bytecode program (see
// rust/metal/expression/eval.metal for the opcode set / semantics) into a
// single MSL arithmetic expression, wraps it in a small generated compute
// kernel, and compiles that kernel *at runtime* via
// `Device::new_library_with_source`. Compiled pipelines are cached by a
// hash of (program bytes, column count) so repeated calls with the same
// program reuse the cached `ComputePipelineState` instead of recompiling.
//
// Rationale: a fused, single-expression kernel avoids the interpreter's
// per-opcode dispatch/stack overhead at the cost of a one-time runtime
// compile. If codegen or the generated kernel fails for any reason, this
// module falls back to the Task 14 interpreter (`eval_expression`) so
// callers always get a correct result.
//
// Dispatch uses `dispatch_threads` (exact grid size = element count) rather
// than `dispatch_thread_groups`, matching the interpreter kernel (Task 14)
// and the elementwise kernels: the generated kernel has no `idx >= len`
// bounds guard, so a threadgroup-padded grid would read/write out of
// bounds for lengths that aren't a multiple of the threadgroup size.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use metal::{ComputePipelineState, Device, CompileOptions, MTLSize, MTLResourceOptions};
use pyo3::prelude::*;

use crate::backend::MetalBackend;
use crate::buffer::{SharedBuffer, DType};
use crate::series::MetalSeries;
use crate::kernels::expression::eval_expression;

const THREADGROUP_SIZE: u64 = 256;

lazy_static::lazy_static! {
    static ref CODEGEN_CACHE: Mutex<HashMap<u64, ComputePipelineState>> =
        Mutex::new(HashMap::new());
}

/// Hash both the program bytes and the column count: the same bytecode
/// could in principle be dispatched with a different number of bound
/// input columns (the generated kernel signature has one `device const
/// float*` parameter per column), so the column count must be part of the
/// cache key to avoid reusing a pipeline compiled for the wrong arity.
fn hash_program(program: &[u8], num_cols: usize) -> u64 {
    let mut hasher = DefaultHasher::new();
    program.hash(&mut hasher);
    num_cols.hash(&mut hasher);
    hasher.finish()
}

/// Walk the bytecode `program` and build a single MSL expression string,
/// using a stack of string fragments (mirrors the interpreter's float
/// stack in rust/metal/expression/eval.metal, but at the source-text
/// level instead of at runtime).
fn decompile_to_expr(program: &[u8]) -> String {
    let mut stack: Vec<String> = Vec::new();
    let mut pc = 0;

    while pc < program.len() {
        let op = program[pc];
        pc += 1;

        // Opcodes 0-7: LOAD_COL_N.
        if op < 8 {
            stack.push(format!("c{}[i]", op));
            continue;
        }

        // Opcode 8: LOAD_SCALAR (little-endian f32 immediate follows).
        if op == 8 {
            let bytes = &program[pc..pc + 4];
            let val = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            stack.push(format!("{:.8}f", val));
            pc += 4;
            continue;
        }

        // Opcodes 16-29: binary ops.
        if op >= 16 && op < 32 {
            let b = stack.pop().unwrap_or_default();
            let a = stack.pop().unwrap_or_default();
            let expr = match op {
                16 => format!("({} + {})", a, b),
                17 => format!("({} - {})", a, b),
                18 => format!("({} * {})", a, b),
                19 => format!("({} / {})", a, b),
                20 => format!("fmod({}, {})", a, b),
                24 => format!("(({} == {}) ? 1.0f : 0.0f)", a, b),
                25 => format!("(({} != {}) ? 1.0f : 0.0f)", a, b),
                26 => format!("(({} < {}) ? 1.0f : 0.0f)", a, b),
                27 => format!("(({} <= {}) ? 1.0f : 0.0f)", a, b),
                28 => format!("(({} > {}) ? 1.0f : 0.0f)", a, b),
                29 => format!("(({} >= {}) ? 1.0f : 0.0f)", a, b),
                _ => format!("/* unknown binary op {} */", op),
            };
            stack.push(expr);
            continue;
        }

        // Opcodes 32-38: unary ops.
        if op >= 32 {
            let a = stack.pop().unwrap_or_default();
            let expr = match op {
                32 => format!("abs({})", a),
                33 => format!("(-{})", a),
                34 => format!("sqrt({})", a),
                35 => format!("exp({})", a),
                36 => format!("log({})", a),
                37 => format!("ceil({})", a),
                38 => format!("floor({})", a),
                _ => format!("/* unknown unary op {} */", op),
            };
            stack.push(expr);
        }
    }

    stack.pop().unwrap_or_else(|| "0.0f".to_string())
}

/// Generate a complete, self-contained MSL source string for a kernel
/// named `fused_{hash}` that evaluates `program` over `num_cols` input
/// columns and writes the result to `out`.
fn codegen_msl(program: &[u8], num_cols: usize, func_name: &str) -> String {
    let mut params = String::new();
    for i in 0..num_cols {
        params.push_str(&format!(
            "    device const float* c{} [[buffer({})]],\n", i, i
        ));
    }
    params.push_str(&format!(
        "    device float* out [[buffer({})]],\n", num_cols
    ));
    params.push_str("    uint i [[thread_position_in_grid]]");

    let expr = decompile_to_expr(program);

    format!(
        "#pragma clang fp contract(off)\n\
         #include <metal_stdlib>\nusing namespace metal;\n\n\
         kernel void {}(\n{}\n) {{\n    out[i] = {};\n}}\n",
        func_name, params, expr
    )
}

/// Compile (or fetch from cache) the fused pipeline for `program` /
/// `num_cols`. Returns the pipeline plus the generated kernel's function
/// name.
fn compile_and_cache(
    device: &Device,
    program: &[u8],
    num_cols: usize,
) -> Result<ComputePipelineState, String> {
    let prog_hash = hash_program(program, num_cols);

    {
        let cache = CODEGEN_CACHE.lock().unwrap();
        if let Some(pipeline) = cache.get(&prog_hash) {
            return Ok(pipeline.clone());
        }
    }

    let func_name = format!("fused_{:016x}", prog_hash);
    let source = codegen_msl(program, num_cols, &func_name);

    let library = device.new_library_with_source(&source, &CompileOptions::new())
        .map_err(|e| format!("Codegen compile failed: {:?}", e))?;
    let function = library.get_function(&func_name, None)
        .map_err(|e| format!("Function '{}' not found: {:?}", func_name, e))?;
    let pipeline = device.new_compute_pipeline_state_with_function(&function)
        .map_err(|e| format!("Pipeline creation failed: {:?}", e))?;

    let mut cache = CODEGEN_CACHE.lock().unwrap();
    cache.insert(prog_hash, pipeline.clone());

    Ok(pipeline)
}

/// Evaluate a bytecode `program` against up to 8 input `columns` by
/// generating and compiling a fused MSL kernel at runtime, falling back to
/// the Task 14 interpreter (`eval_expression`) if codegen, compilation, or
/// GPU execution fails.
#[pyfunction]
pub fn eval_expression_codegen(
    program: Vec<u8>,
    columns: Vec<PyRef<MetalSeries>>,
    size: usize,
) -> PyResult<MetalSeries> {
    if columns.is_empty() || size == 0 || columns.len() > 8 {
        return eval_expression(program, columns, size);
    }

    let device = match MetalBackend::device() {
        Some(device) => device,
        None => return eval_expression(program, columns, size),
    };

    let pipeline = match compile_and_cache(device, &program, columns.len()) {
        Ok(pipeline) => pipeline,
        Err(_) => return eval_expression(program, columns, size),
    };

    let queue = match MetalBackend::queue() {
        Some(queue) => queue,
        None => return eval_expression(program, columns, size),
    };

    let out_buf = device.new_buffer(
        (size * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);

    for (i, col) in columns.iter().enumerate() {
        let buf = match col.as_numeric_checked() {
            Ok(buf) => buf,
            Err(_) => {
                enc.end_encoding();
                return eval_expression(program, columns, size);
            }
        };
        enc.set_buffer(i as u64, Some(buf.metal_buffer()), 0);
    }
    enc.set_buffer(columns.len() as u64, Some(&out_buf), 0);

    let tg_size = THREADGROUP_SIZE.min(size as u64);
    enc.dispatch_threads(
        MTLSize::new(size as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    if cb.status() == metal::MTLCommandBufferStatus::Error {
        return eval_expression(program, columns, size);
    }

    let result_buf = SharedBuffer::from_metal_buffer(out_buf, size, DType::Float32);
    Ok(MetalSeries::from_numeric(result_buf))
}
