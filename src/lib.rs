#![warn(missing_docs)]

//! The light.
//!
//! Rays, acceleration structures and (eventually) an integrator, over
//! anything that can be bounded and hit. The engine owns the renderer; the
//! renderer owns no geometry.
//!
//! # The seam
//!
//! [`Geometry`] is the entire boundary. It says: how many primitives you
//! have, where each one is, and what a ray finds when it meets one. Implement
//! it and you get a [`Bvh`] over your primitives, a [`Tlas`] over placed
//! instances of them, and everything above.
//!
//! vcad implements it with the trimmed analytic faces of a BRep solid; a
//! phyz collider set or a splat cloud would implement the same three
//! methods. Nothing in here knows the difference, which is the point.
//!
//! # The dependency rule
//!
//! `kosm-render` depends on [`tang`] (one scalar, one set of math types),
//! `rayon`, and later `wgpu`/`bytemuck`. **Never** on `vcad-*`, `phyz-*`, or
//! any other Kosm crate. It has to build for `wasm32-unknown-unknown`, so
//! anything native-only is feature-gated or off the main path.

#[cfg(feature = "gpu")]
pub mod gpu;

pub mod bvh;
pub mod env;
pub mod geometry;
pub mod heightfield;
pub mod math;
pub mod optics;
pub mod pathtrace;
pub mod splats;
mod ray;
pub mod spectrum;
mod sah;
mod tables;
pub mod tlas;

pub use bvh::{Bvh, BvhNode, FlatBvhNode};
pub use env::{BuiltinEnv, generate as generate_env, parse_hdr};
pub use geometry::{Geometry, TriMesh, TriangleHit, intersect_triangle};
pub use heightfield::HeightField;
pub use math::{Aabb, Dir3, Point2, Point3, Transform, Vec2, Vec3, transform_from_column_major};
pub use optics::{fresnel, index, reflect, refract, sellmeier};
pub use spectrum::{cauchy_index, cie_xyz, hero_weight, sellmeier_index, BK7_SELLMEIER};
pub use pathtrace::{
    AreaLight, Camera, EnvMap, Environment, Film, GradientEnv, Ground, Object, PathTraceOptions,
    Pbr, PixelFilter, Scene, Sun, denoise, render, render_into, studio_rig,
};
pub use ray::{Hit, Ray};
pub use splats::{Splats, unpack_payload};
pub use tlas::{FlatTlasNode, Instance, InstanceHit, Tlas};
