//! Phase 4, "the look": OpenPBR Surface, participating media, hair.
//! design.md, "Phase 4", is the why and the numbers.
//!
//! - [`openpbr`]: the spec's parameters, Fresnel, F82, the exact Airy film,
//!   the albedo tables, a material at one wavelength.
//! - [`graph`]: the albedo-scaled OpenPBR stack as one tang IR graph.
//! - [`swatch`]: the CPU f64 swatch integrator with path replay.
//! - `scene`, `scene_gpu` (phase 5b, features `assets` + `gpu`): full scenes
//!   with OpenPBR materials on the device.

#[cfg(feature = "gpu")]
pub mod gpu;
pub mod graph;
pub mod openpbr;
pub mod sheen;
pub mod swatch;
pub mod volume;
pub mod hair;
#[cfg(feature = "gpu")]
pub mod hair_gpu;
#[cfg(feature = "assets")]
pub mod scene;
#[cfg(all(feature = "gpu", feature = "assets"))]
pub mod scene_gpu;
// M1 (docs/belief-codec.md): splats in the scene kernel, the sensor kernel
// (AOVs, batched cameras, pinholes, TLAS-only rebuilds).
#[cfg(all(feature = "gpu", feature = "assets"))]
pub mod batch;
#[cfg(all(feature = "gpu", feature = "assets"))]
pub mod instances;
#[cfg(feature = "assets")]
pub mod pinhole;
#[cfg(all(feature = "gpu", feature = "assets"))]
pub mod present;
#[cfg(all(feature = "gpu", feature = "assets"))]
pub mod residual;
#[cfg(all(feature = "gpu", feature = "assets"))]
pub mod sensor;
#[cfg(all(feature = "gpu", feature = "assets"))]
pub mod splat_gpu;
