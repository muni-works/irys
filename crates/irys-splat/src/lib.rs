//! irys-splat: fit a 3D Gaussian splat map to posed photos with irys's own
//! GPU tier (wgpu 30, Metal here) — no CUDA, no gsplat.
//!
//! - [`project`]: EWA projection into an OpenCV pinhole and its adjoint (f64).
//! - [`raster_cpu`]: the f64 reference rasteriser and its adjoint.
//! - [`gpu`]: the tile rasteriser on the device (`raster.wgsl`): forward,
//!   adjoint, and a per-splat reduction in a fixed order (no float atomics).
//! - [`loss`]: L1 + D-SSIM (+ depth L1) and its gradient.
//! - [`train`]: Adam per parameter group, densification, opacity reset,
//!   pruning, SH warm-up.
//! - [`data`]: posed image sets (`poses.jsonl` + `camera_info.json`).
//!
//! docs/design.md, "Splat training", has the method and the numbers.

pub mod data;
pub mod gpu;
pub mod knn;
pub mod loss;
pub mod model;
pub mod project;
pub mod raster_cpu;
pub mod rng;
pub mod sh;
pub mod tiles;
pub mod train;
