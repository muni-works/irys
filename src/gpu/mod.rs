//! The GPU tier: the same integrator, on a wgpu compute pipeline.
//!
//! Behind the `gpu` feature. What is here is *renderer*: the device, the
//! BSDF, the environment, the camera and render state, the accumulator, the
//! per-pixel history and the denoiser. What is not here is geometry — a
//! client supplies that as WGSL plus packed buffers. [`geometry`] is the whole
//! of that seam, and documents the binding split it implies.

pub mod analytic;
mod budget;
mod buffers;
mod context;
pub mod geometry;
mod history;
pub mod neural;
mod pipeline;
mod resident;
mod scene;
pub mod shaders;

/// The `wgpu` this crate was built against.
///
/// Re-exported so a client — or one of this crate's own tests — can name a
/// texture format or a buffer usage without pinning the same version itself
/// and risking two `wgpu`s in the graph.
pub use wgpu;

pub use analytic::{AnalyticGeometry, AnalyticPrim};
pub use budget::{Budget, SampleBudget};
pub use buffers::{
    DEFAULT_ENV_INTENSITY, DEFAULT_FIREFLY_CLAMP, DEFAULT_MAX_DEPTH, DEFAULT_RR_START,
    FLAG_BUDGET_GUIDES, FLAG_BUDGET_MASK, FLAG_CAMERA_VISIBLE_LIGHTS, FLAG_RAW_SAMPLE,
    GpuAreaLight, GpuCamera, GpuMaterial, GpuRenderState, depth_for_frame, pack_light_power_table,
};
pub use context::{GpuContext, GpuError};
pub use geometry::{GeometryModule, GeometrySlab, GpuGeometry, storage_entry};
pub use history::{
    GpuDenoiseParams, Guides, History, HistoryBuffers, HistoryPipeline, InstanceMotion,
    MAX_DENOISE_ITERS, atrous_iters_for,
};
pub use neural::{NeuralDenoiser, NeuralPipeline};
pub use pipeline::RayTracePipeline;
pub use resident::ResidentScene;
pub use scene::{NoGeometry, SceneRef};
