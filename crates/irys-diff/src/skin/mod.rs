//! Photoreal skin: a textured head, a spectral random-walk subsurface
//! medium under a dual-lobe specular layer, portrait lighting, and path
//! replay backpropagation into the medium's chromophores and the layer's
//! roughness. design.md, "Skin", is the why and the numbers.
//!
//! - [`mesh`]: textured meshes, normal mapping, the glTF loader (`assets`).
//! - [`optics`]: skin's spectra, the exact albedo of the walk (Chandrasekhar's
//!   `H`), the albedo-map → chromophore inversion.
//! - [`graph`]: the two differentiable factors as tang IR graphs.
//! - [`render`]: the CPU f64 integrator, forward and replay.
//! - [`accel`]: nearest-triangle distance, the walk's safe sphere.
//! - [`head`]: the Lee Perry-Smith portrait scene.
//! - `gpu` (feature `gpu`): the same integrator in WGSL, forward only.

pub mod accel;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod graph;
pub mod head;
pub mod mesh;
pub mod optics;
pub mod render;
