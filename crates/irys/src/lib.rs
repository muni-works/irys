#![warn(missing_docs)]

//! The light.
//!
//! A facade over `irys-core`, `irys-light`, `irys-prims` and (behind `gpu`)
//! `irys-gpu`, with the paths `kosm-render` 0.2 had.
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
//! `irys` depends on [`tang`] (one scalar, one set of math types),
//! `rayon`, and later `wgpu`/`bytemuck`. **Never** on `vcad-*`, `phyz-*`, or
//! any other Kosm crate. It has to build for `wasm32-unknown-unknown`, so
//! anything native-only is feature-gated or off the main path.

#[cfg(feature = "gpu")]
pub use irys_gpu as gpu;

pub use irys_core::{bvh, geometry, math, sampler, tlas};
pub use irys_light::{caustics, cpu, env, neural, optics, pathtrace, spectrum, splats};
pub use irys_prims::{analytic, heightfield};

pub use irys_core::{
    Aabb, Bvh, BvhNode, Dir3, FlatBvhNode, FlatTlasNode, Geometry, Hit, Instance, InstanceHit,
    Point2, Point3, Ray, Tlas, Transform, TriMesh, TriangleHit, Vec2, Vec3, intersect_triangle,
    transform_from_column_major,
};
pub use irys_light::{
    AreaLight, BK7_SELLMEIER, BuiltinEnv, Camera, EnvMap, Environment, Film, GradientEnv, Ground,
    Object, PathTraceOptions, Pbr, PixelFilter, Projection, Scene, Splats, Sun, cauchy_index,
    cie_xyz, denoise, fresnel, generate_env, hero_weight, index, parse_hdr, reflect, refract,
    render, render_into, sellmeier, sellmeier_index, spec_albedo, studio_rig, unpack_payload,
};
pub use irys_prims::{Analytic, Frame, HeightField, Prim};
