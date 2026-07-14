use pyo3::prelude::*;

pub mod backend;
pub mod buffer;
pub mod column;
pub mod column_view;
pub mod kernels;
pub mod series;

use backend::{BatchContext, MetalBackend};
use series::MetalSeries;
use kernels::{is_debug_enabled, set_debug_enabled, detect_gpu_family, tuning};
use kernels::comparison::metal_compare_op;
use kernels::elementwise::{
    metal_binary_op, metal_binary_op_batched, metal_unary_op,
    metal_logical_and, metal_logical_or, metal_logical_not,
};
use kernels::reductions::{metal_sum, metal_min, metal_max, metal_mean};
use kernels::sort::{metal_sort, metal_argsort};
use kernels::groupby::{metal_groupby_sum, metal_groupby_mean, metal_groupby_min, metal_groupby_max, metal_groupby_count};
use kernels::strings::{
    metal_string_eq, metal_string_ne, metal_string_lt, metal_string_gt,
    metal_string_le, metal_string_ge, metal_string_eq_scalar,
    metal_string_contains, metal_string_startswith, metal_string_endswith, metal_string_find,
    metal_string_lower, metal_string_upper, metal_string_strip, metal_string_replace,
    metal_string_sort, metal_string_groupby,
};
use kernels::expression::{eval_expression, eval_expression_reduce};
use kernels::codegen::{eval_expression_codegen, eval_multi_expression_codegen};
use kernels::scan::metal_prefix_sum;
use kernels::filter::{metal_compact, metal_take};
use kernels::join::metal_hash_join;
use kernels::rolling::{metal_rolling_sum, metal_rolling_min, metal_rolling_max, metal_rolling_count, metal_rolling_mean};
use kernels::datetime::{
    metal_dt_year, metal_dt_month, metal_dt_day,
    metal_dt_hour, metal_dt_minute, metal_dt_second, metal_dt_dayofweek,
};

#[pyfunction]
fn metal_gpu_info(py: Python) -> PyResult<PyObject> {
    let device = MetalBackend::device()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal device"))?;
    let t = tuning();

    let dict = pyo3::types::PyDict::new_bound(py);
    dict.set_item("name", device.name().to_string())?;
    dict.set_item("gpu_family", detect_gpu_family(device))?;

    let max_tpg = device.max_threads_per_threadgroup();
    dict.set_item("max_threads_per_threadgroup", max_tpg.width)?;
    dict.set_item("max_threadgroup_memory_bytes", device.max_threadgroup_memory_length())?;
    dict.set_item("max_buffer_length_bytes", device.max_buffer_length())?;

    dict.set_item("tuning_reduce_threadgroup_size", t.reduce_threadgroup_size)?;
    dict.set_item("tuning_reduce_n_reads", t.reduce_n_reads)?;
    dict.set_item("tuning_local_sort_size", t.local_sort_size)?;
    dict.set_item("tuning_local_sort_stages", t.local_sort_stages)?;

    Ok(dict.into())
}

/// Start a new batch: allocates a fresh command buffer that subsequent
/// `metal_*_batched` calls encode dispatches into, without committing.
/// Call `batch_commit` once all desired ops have been encoded.
#[pyfunction]
fn begin_batch() -> PyResult<BatchContext> {
    MetalBackend::begin_batch()
}

/// Commit a batch's command buffer and block until the GPU finishes
/// executing every dispatch encoded into it (via `metal_*_batched` calls).
#[pyfunction]
fn batch_commit(batch: &BatchContext) -> PyResult<()> {
    batch.commit_and_wait()
}

#[pymodule]
fn metaldf_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MetalSeries>()?;
    m.add_class::<BatchContext>()?;
    m.add_wrapped(wrap_pyfunction!(metal_binary_op))?;
    m.add_wrapped(wrap_pyfunction!(metal_binary_op_batched))?;
    m.add_wrapped(wrap_pyfunction!(begin_batch))?;
    m.add_wrapped(wrap_pyfunction!(batch_commit))?;
    m.add_wrapped(wrap_pyfunction!(metal_unary_op))?;
    m.add_wrapped(wrap_pyfunction!(metal_logical_and))?;
    m.add_wrapped(wrap_pyfunction!(metal_logical_or))?;
    m.add_wrapped(wrap_pyfunction!(metal_logical_not))?;
    m.add_wrapped(wrap_pyfunction!(metal_compare_op))?;
    m.add_wrapped(wrap_pyfunction!(metal_sum))?;
    m.add_wrapped(wrap_pyfunction!(metal_min))?;
    m.add_wrapped(wrap_pyfunction!(metal_max))?;
    m.add_wrapped(wrap_pyfunction!(metal_mean))?;
    m.add_wrapped(wrap_pyfunction!(metal_sort))?;
    m.add_wrapped(wrap_pyfunction!(metal_argsort))?;
    m.add_wrapped(wrap_pyfunction!(metal_groupby_sum))?;
    m.add_wrapped(wrap_pyfunction!(metal_groupby_mean))?;
    m.add_wrapped(wrap_pyfunction!(metal_groupby_min))?;
    m.add_wrapped(wrap_pyfunction!(metal_groupby_max))?;
    m.add_wrapped(wrap_pyfunction!(metal_groupby_count))?;
    m.add_wrapped(wrap_pyfunction!(metal_string_eq))?;
    m.add_wrapped(wrap_pyfunction!(metal_string_ne))?;
    m.add_wrapped(wrap_pyfunction!(metal_string_lt))?;
    m.add_wrapped(wrap_pyfunction!(metal_string_gt))?;
    m.add_wrapped(wrap_pyfunction!(metal_string_le))?;
    m.add_wrapped(wrap_pyfunction!(metal_string_ge))?;
    m.add_wrapped(wrap_pyfunction!(metal_string_eq_scalar))?;
    m.add_wrapped(wrap_pyfunction!(metal_string_contains))?;
    m.add_wrapped(wrap_pyfunction!(metal_string_startswith))?;
    m.add_wrapped(wrap_pyfunction!(metal_string_endswith))?;
    m.add_wrapped(wrap_pyfunction!(metal_string_find))?;
    m.add_wrapped(wrap_pyfunction!(metal_string_lower))?;
    m.add_wrapped(wrap_pyfunction!(metal_string_upper))?;
    m.add_wrapped(wrap_pyfunction!(metal_string_strip))?;
    m.add_wrapped(wrap_pyfunction!(metal_string_replace))?;
    m.add_wrapped(wrap_pyfunction!(metal_string_sort))?;
    m.add_wrapped(wrap_pyfunction!(metal_string_groupby))?;
    m.add_wrapped(wrap_pyfunction!(eval_expression))?;
    m.add_wrapped(wrap_pyfunction!(eval_expression_reduce))?;
    m.add_wrapped(wrap_pyfunction!(eval_expression_codegen))?;
    m.add_wrapped(wrap_pyfunction!(eval_multi_expression_codegen))?;
    m.add_wrapped(wrap_pyfunction!(metal_gpu_info))?;
    m.add_wrapped(wrap_pyfunction!(metal_prefix_sum))?;
    m.add_wrapped(wrap_pyfunction!(metal_compact))?;
    m.add_wrapped(wrap_pyfunction!(metal_take))?;
    m.add_wrapped(wrap_pyfunction!(metal_hash_join))?;
    m.add_wrapped(wrap_pyfunction!(metal_rolling_sum))?;
    m.add_wrapped(wrap_pyfunction!(metal_rolling_min))?;
    m.add_wrapped(wrap_pyfunction!(metal_rolling_max))?;
    m.add_wrapped(wrap_pyfunction!(metal_rolling_count))?;
    m.add_wrapped(wrap_pyfunction!(metal_rolling_mean))?;
    m.add_wrapped(wrap_pyfunction!(metal_dt_year))?;
    m.add_wrapped(wrap_pyfunction!(metal_dt_month))?;
    m.add_wrapped(wrap_pyfunction!(metal_dt_day))?;
    m.add_wrapped(wrap_pyfunction!(metal_dt_hour))?;
    m.add_wrapped(wrap_pyfunction!(metal_dt_minute))?;
    m.add_wrapped(wrap_pyfunction!(metal_dt_second))?;
    m.add_wrapped(wrap_pyfunction!(metal_dt_dayofweek))?;

    m.add_wrapped(wrap_pyfunction!(py_set_debug_enabled))?;
    m.add_wrapped(wrap_pyfunction!(py_is_debug_enabled))?;

    Ok(())
}

#[pyfunction(name = "set_debug_enabled")]
fn py_set_debug_enabled(enabled: bool) {
    set_debug_enabled(enabled);
}

#[pyfunction(name = "is_debug_enabled")]
fn py_is_debug_enabled() -> bool {
    is_debug_enabled()
}
