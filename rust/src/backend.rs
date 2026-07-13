// MetalBackend — wraps Metal device and command queue.
//
// Provides singleton access to the system default Metal device and
// a command queue for submitting GPU work.
//
// Also provides `BatchContext`, a thin wrapper around a single
// `metal::CommandBuffer` that lets callers encode several compute
// dispatches (via repeated `encode()` calls) before committing and
// waiting on the batch exactly once (`commit_and_wait()`), instead of
// paying a commit/wait round-trip per kernel dispatch.

use std::sync::OnceLock;

use pyo3::prelude::*;
use metal::{CommandBuffer, CommandQueue, ComputePipelineState, Device, MTLSize};

static DEVICE: OnceLock<Device> = OnceLock::new();
static QUEUE: OnceLock<CommandQueue> = OnceLock::new();

pub struct MetalBackend;

impl MetalBackend {
    pub fn device() -> Option<&'static Device> {
        DEVICE.get_or_init(|| {
            Device::system_default()
                .expect("No Metal device found — requires Apple Silicon or Metal-capable GPU")
        });
        DEVICE.get()
    }

    pub fn queue() -> Option<&'static CommandQueue> {
        QUEUE.get_or_init(|| {
            let device = Self::device()
                .expect("Metal device must be initialized before creating command queue");
            device.new_command_queue()
        });
        QUEUE.get()
    }

    pub fn device_and_queue() -> pyo3::PyResult<(&'static Device, &'static CommandQueue)> {
        let device = Self::device()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal device"))?;
        let queue = Self::queue()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal queue"))?;
        Ok((device, queue))
    }

    /// Start a new batch: allocates a fresh `CommandBuffer` from the shared
    /// queue that callers can encode multiple compute dispatches into
    /// (via `BatchContext::encode`) before committing once
    /// (`BatchContext::commit_and_wait`).
    pub fn begin_batch() -> pyo3::PyResult<BatchContext> {
        let queue = Self::queue()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("No Metal queue"))?;
        Ok(BatchContext {
            command_buffer: queue.new_command_buffer().to_owned(),
        })
    }
}

/// A single `CommandBuffer` shared across multiple compute dispatches.
///
/// Each call to `encode()` opens its own compute command encoder, sets up
/// the pipeline/buffers/dispatch, and ends encoding — but all encoders share
/// `command_buffer`, so the GPU work is only committed and waited on once,
/// via `commit_and_wait()`, regardless of how many kernels were encoded.
///
/// Marked `#[pyclass]` so a `BatchContext` can be handed to and returned
/// from Python code (e.g. held across several Rust-side dispatch calls
/// driven from a Python loop). `metal::CommandBuffer` is `Send + Sync`
/// (see the `metal` crate's `foreign_obj_type!` macro), so no
/// `unsendable` annotation is required.
#[pyclass]
pub struct BatchContext {
    command_buffer: CommandBuffer,
}

impl BatchContext {
    /// Encode one compute dispatch into this batch's command buffer:
    /// binds `pipeline`, binds each `(buffer, offset)` pair to consecutive
    /// buffer indices starting at 0, optionally sets threadgroup memory
    /// length, dispatches `grid_size` threadgroups of `threadgroup_size`
    /// threads each, and ends encoding. Does not commit or wait — callers
    /// may call `encode()` any number of times before a single
    /// `commit_and_wait()`.
    pub fn encode(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[(&metal::Buffer, u64)],
        grid_size: MTLSize,
        threadgroup_size: MTLSize,
        threadgroup_memory: Option<(u64, u64)>,
    ) {
        let enc = self.command_buffer.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        for (i, (buf, offset)) in buffers.iter().enumerate() {
            enc.set_buffer(i as u64, Some(buf), *offset);
        }
        if let Some((index, length)) = threadgroup_memory {
            enc.set_threadgroup_memory_length(index, length);
        }
        enc.dispatch_thread_groups(grid_size, threadgroup_size);
        enc.end_encoding();
    }

    /// Encode one compute dispatch into this batch's command buffer using
    /// `dispatch_threads` (an exact thread count, no threadgroup-multiple
    /// padding) rather than `encode()`'s `dispatch_thread_groups`. Mirrors
    /// the unbatched elementwise dispatch in
    /// `crate::kernels::elementwise::dispatch_elementwise`: those kernels
    /// have no `idx >= len` bounds guard, so a padded grid (as
    /// `dispatch_thread_groups` requires) would read/write out of bounds
    /// for lengths that aren't a multiple of the threadgroup size.
    pub fn encode_threads(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[(&metal::Buffer, u64)],
        thread_count: MTLSize,
        threadgroup_size: MTLSize,
    ) {
        let enc = self.command_buffer.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pipeline);
        for (i, (buf, offset)) in buffers.iter().enumerate() {
            enc.set_buffer(i as u64, Some(buf), *offset);
        }
        enc.dispatch_threads(thread_count, threadgroup_size);
        enc.end_encoding();
    }

    /// Commit the batch's command buffer and block until the GPU finishes
    /// executing every kernel encoded via `encode()`, surfacing a Metal
    /// command-buffer error (if any) as a `PyRuntimeError`.
    pub fn commit_and_wait(&self) -> pyo3::PyResult<()> {
        self.command_buffer.commit();
        self.command_buffer.wait_until_completed();
        if self.command_buffer.status() == metal::MTLCommandBufferStatus::Error {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Batched command buffer failed"
            ));
        }
        Ok(())
    }
}
