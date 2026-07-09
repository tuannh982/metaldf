// MetalBackend — wraps Metal device and command queue.
//
// Provides singleton access to the system default Metal device and
// a command queue for submitting GPU work.

use std::sync::OnceLock;

use metal::{CommandQueue, Device};

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
}
