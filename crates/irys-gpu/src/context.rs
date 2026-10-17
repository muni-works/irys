//! The device: one wgpu adapter, device and queue per process.
//!
//! Ported from vcad-kernel-gpu, which now re-exports this.

use std::sync::OnceLock;
use wgpu::{Device, Instance, Queue};

// Wrapper to make GpuContext Send+Sync for WASM. Safe only while the
// target is single-threaded WASM; if wasm32 threads (SharedArrayBuffer)
// are ever enabled, `target_feature = "atomics"` becomes active and this
// cfg_attr trips a compile error so the bad state can't silently ship.
#[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
struct SendSyncWrapper(GpuContext);

#[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
unsafe impl Send for SendSyncWrapper {}
#[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
unsafe impl Sync for SendSyncWrapper {}

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
compile_error!(
    "kosm-render SendSyncWrapper relies on single-threaded WASM; \
     revisit GPU_CONTEXT locking before enabling wasm32 threads"
);

#[cfg(all(target_arch = "wasm32", not(target_feature = "atomics")))]
static GPU_CONTEXT: OnceLock<SendSyncWrapper> = OnceLock::new();

#[cfg(not(target_arch = "wasm32"))]
static GPU_CONTEXT: OnceLock<GpuContext> = OnceLock::new();

// Guard to prevent concurrent initialization attempts
#[cfg(target_arch = "wasm32")]
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(target_arch = "wasm32")]
static INIT_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

/// Errors that can occur during GPU operations.
#[derive(Debug)]
pub enum GpuError {
    /// No compatible GPU adapter found.
    NoAdapter,
    /// GPU context was already initialized.
    AlreadyInitialized,
    /// Failed to request GPU device.
    DeviceRequest(wgpu::RequestDeviceError),
    /// Buffer mapping failed.
    BufferMapping,
    /// GPU context not initialized.
    NotInitialized,
    /// Invalid caller-provided input (dimensions, alignment, lengths).
    InvalidInput(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::NoAdapter => write!(f, "No compatible GPU adapter found"),
            GpuError::AlreadyInitialized => write!(f, "GPU context already initialized"),
            GpuError::DeviceRequest(e) => write!(f, "Failed to request GPU device: {e}"),
            GpuError::BufferMapping => write!(f, "Buffer mapping failed"),
            GpuError::NotInitialized => {
                write!(
                    f,
                    "GPU context not initialized - call GpuContext::init() first"
                )
            }
            GpuError::InvalidInput(s) => write!(f, "Invalid input: {s}"),
        }
    }
}

impl std::error::Error for GpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GpuError::DeviceRequest(e) => Some(e),
            _ => None,
        }
    }
}

impl From<wgpu::RequestDeviceError> for GpuError {
    fn from(e: wgpu::RequestDeviceError) -> Self {
        GpuError::DeviceRequest(e)
    }
}

/// Global GPU context holding device and queue.
pub struct GpuContext {
    /// The wgpu device for creating resources and pipelines.
    pub device: Device,
    /// The command queue for submitting work.
    pub queue: Queue,
}

impl GpuContext {
    /// Create a new GPU context from device and queue.
    async fn create() -> Result<Self, GpuError> {
        // PRIMARY = Vulkan | Metal | DX12 | BrowserWebGPU — the same web
        // backends as before, plus native GPUs (Metal on macOS) so the
        // desktop app gets the compute pipelines too. GL stays as fallback.
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY | wgpu::Backends::GL,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        // wgpu 30 reports *why* no adapter came back; this crate has one
        // failure for all of them.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .map_err(|_| GpuError::NoAdapter)?;

        // The raytrace bind group layout needs 10 storage buffers per
        // compute stage; default wgpu limits only allow 8. Inherit the
        // adapter's full limits so this and similar pipelines validate.
        let required_limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("kosm-render GPU device"),
                required_features: wgpu::Features::empty(),
                required_limits,
                memory_hints: wgpu::MemoryHints::default(),
                ..Default::default()
            })
            .await?;

        // Surface WebGPU validation / out-of-memory errors that don't sit
        // inside an explicit error scope. Without this, a bad shader or
        // bind-group layout leaves the pipeline silently invalid and the
        // output texture stays at its initial zero — exactly the failure
        // mode that hid the raytrace WGSL bug for too long. Route them to
        // a place a human will see (browser console on wasm, eprintln
        // otherwise).
        device.on_uncaptured_error(std::sync::Arc::new(|err| {
            eprintln!("WebGPU uncaptured error: {err}");
        }));

        Ok(GpuContext { device, queue })
    }

    /// Initialize the GPU context asynchronously.
    ///
    /// This should be called once at application startup. Subsequent calls
    /// will return the existing context.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn init() -> Result<&'static Self, GpuError> {
        if let Some(ctx) = GPU_CONTEXT.get() {
            return Ok(ctx);
        }

        let ctx = Self::create().await?;

        // If another thread won the race, use its context; ours is dropped.
        let _ = GPU_CONTEXT.set(ctx);

        Ok(GPU_CONTEXT.get().unwrap())
    }

    /// Initialize the GPU context asynchronously (WASM version).
    #[cfg(target_arch = "wasm32")]
    pub async fn init() -> Result<&'static Self, GpuError> {
        // Already initialized - return existing context
        if let Some(wrapper) = GPU_CONTEXT.get() {
            return Ok(&wrapper.0);
        }

        // Check if another init is in progress (prevents race condition)
        if INIT_IN_PROGRESS.swap(true, Ordering::SeqCst) {
            // Another init is running, wait for it by polling
            // In single-threaded WASM, this means init was called recursively or re-entrantly
            // Just return an error to avoid the FnOnce panic
            return Err(GpuError::AlreadyInitialized);
        }

        let result = Self::create().await;

        match result {
            Ok(ctx) => {
                let _ = GPU_CONTEXT.set(SendSyncWrapper(ctx));
                INIT_IN_PROGRESS.store(false, Ordering::SeqCst);
                Ok(&GPU_CONTEXT.get().unwrap().0)
            }
            Err(e) => {
                INIT_IN_PROGRESS.store(false, Ordering::SeqCst);
                Err(e)
            }
        }
    }

    /// Get the GPU context if it has been initialized.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn get() -> Option<&'static Self> {
        GPU_CONTEXT.get()
    }

    /// Get the GPU context if it has been initialized (WASM version).
    #[cfg(target_arch = "wasm32")]
    pub fn get() -> Option<&'static Self> {
        GPU_CONTEXT.get().map(|w| &w.0)
    }

    /// Get the GPU context, returning an error if not initialized.
    pub fn require() -> Result<&'static Self, GpuError> {
        Self::get().ok_or(GpuError::NotInitialized)
    }

    /// Initialize the GPU context synchronously (native only).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn init_blocking() -> Result<&'static Self, GpuError> {
        pollster::block_on(Self::init())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires GPU"]
    fn test_gpu_init() {
        let ctx = GpuContext::init_blocking();
        assert!(ctx.is_ok() || matches!(ctx, Err(GpuError::NoAdapter)));
    }
}
