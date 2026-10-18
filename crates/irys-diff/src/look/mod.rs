//! Phase 4, "the look": OpenPBR Surface, participating media, hair.
//! design.md, "Phase 4", is the why and the numbers.
//!
//! - [`openpbr`]: the spec's parameters, Fresnel, F82, the exact Airy film,
//!   the albedo tables, a material at one wavelength.
//! - [`graph`]: the albedo-scaled OpenPBR stack as one tang IR graph.
//! - [`swatch`]: the CPU f64 swatch integrator with path replay.

#[cfg(feature = "gpu")]
pub mod gpu;
pub mod graph;
pub mod openpbr;
pub mod swatch;
pub mod volume;
pub mod hair;
