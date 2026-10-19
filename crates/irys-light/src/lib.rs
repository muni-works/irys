#![warn(missing_docs)]

//! The light: spectra, optics, environments, the CPU path tracer, caustics,
//! splats and the CPU side of the neural denoiser.
//!
//! One crate because the pieces are one loop: the integrator composites
//! splats and gathers caustics, and both of those shade through the
//! integrator's own surface model. `spectrum` and `optics` call each other.

// The modules were written as one crate; these keep their `crate::` paths.
use irys_core::{bvh, geometry, math, ray, tlas};

pub mod caustics;
pub mod cpu;
pub mod env;
pub mod neural;
pub mod optics;
pub mod pathtrace;
pub mod ply;
pub mod spectrum;
pub mod splats;
mod tables;
pub use tables::spec_albedo;

pub use env::{BuiltinEnv, generate as generate_env, parse_hdr};
pub use optics::{fresnel, index, reflect, refract, sellmeier};
pub use pathtrace::{
    AreaLight, Camera, EnvMap, Environment, Film, GradientEnv, Ground, Object, PathTraceOptions,
    Pbr, PixelFilter, Projection, Scene, Sun, denoise, render, render_into, studio_rig,
};
pub use spectrum::{BK7_SELLMEIER, cauchy_index, cie_xyz, hero_weight, sellmeier_index};
pub use ply::PlyCloud;
pub use splats::{Splats, unpack_payload};
