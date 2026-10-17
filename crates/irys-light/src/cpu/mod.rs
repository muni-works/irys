//! The CPU tier: the reference path tracer.
//!
//! Physically-based unidirectional path tracing on `rayon` (or a plain
//! iterator on wasm). It mirrors [`crate::gpu`] file for file — the surface
//! model, the emitters, the scene and its acceleration structures, the
//! camera, the integrator, the film and the denoiser — so the two tiers can
//! be read side by side.
//!
//! Where a rasteriser evaluates a hand-tuned lighting rig at the primary hit
//! and stops, this solves the rendering equation by Monte Carlo integration:
//! multiple bounces, importance-sampled microfacet lobes, multiple importance
//! sampling against explicit area lights, and a physical camera with a real
//! aperture.
//!
//! It knows nothing about what it is tracing. Geometry arrives as a
//! [`Geometry`] implementation behind a [`Bvh`], so a B-rep's analytic faces
//! are traced exactly — curved silhouettes and specular highlights on fillets
//! correct at any resolution, no tessellation anywhere — and a triangle soup
//! or a splat cloud goes through the same integrator unchanged.
//!
//! # Design
//!
//! - **Lights** are intersectable rectangles ("softboxes"). Because they can
//!   be hit by a BSDF ray *and* sampled directly, both strategies combine
//!   under MIS with the power heuristic. That is what puts crisp, correctly
//!   shaped highlights on metal.
//! - **The environment** is, by default, a smooth analytic studio gradient.
//!   It is low-frequency by construction, so BSDF sampling alone converges
//!   quickly and no environment CDF is needed. An opt-in lat-long HDR image
//!   ([`EnvMap`]) is also supported; because a real HDRI carries windows and
//!   sun discs, that variant builds a `sin(theta)`-weighted 2D CDF and joins
//!   the MIS mix as a third sampling strategy.
//! - **The BSDF** is a layered metallic-roughness model: Lambert diffuse,
//!   a GGX specular lobe with VNDF sampling, and a GGX clearcoat lobe.
//!   Clearcoat is what sells anodised aluminium and moulded plastic.

pub(crate) use std::sync::Arc;

pub(crate) use crate::bvh::Bvh;
pub(crate) use crate::caustics::CausticMap;
pub(crate) use crate::geometry::Geometry;
pub(crate) use crate::math::{Aabb, Point3, Transform, Vec3};
pub(crate) use crate::ray::Ray;
pub(crate) use crate::splats::{SplatSegment, Splats};
pub(crate) use crate::tlas::{Instance, InstanceHit, Tlas};

/// Rows of the film, in parallel where there are threads to do it with.
///
/// The browser has no `rayon`, and a renderer that cannot run where the
/// picture is looked at is half a renderer — so the row split is a macro and
/// the two spellings sit behind one `cfg`. Every adaptor used downstream
/// (`zip`, `enumerate`, `skip`, `take`, `for_each`) exists on both.
#[cfg(not(target_arch = "wasm32"))]
macro_rules! film_rows {
    ($buf:expr, $n:expr) => {
        $buf.par_chunks_mut($n)
    };
}

#[cfg(target_arch = "wasm32")]
macro_rules! film_rows {
    ($buf:expr, $n:expr) => {
        $buf.chunks_mut($n)
    };
}

pub mod atrous;
pub mod camera;
pub mod film;
pub mod integrator;
pub mod light;
pub mod material;
pub mod rng;
pub mod scene;
#[cfg(test)]
mod testing;

pub use atrous::*;
pub use camera::*;
pub use film::*;
pub use integrator::*;
pub use light::*;
pub use material::*;
pub(crate) use rng::*;
pub use scene::*;
