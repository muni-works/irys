#![warn(missing_docs)]

//! The seam and the acceleration structures.
//!
//! [`Geometry`] is the entire boundary between irys and whatever it draws:
//! how many primitives there are, where each one is, and what a ray finds
//! when it meets one. Implement it and you get a [`Bvh`] over your
//! primitives and a [`Tlas`] over placed instances of them. The sampler's
//! blue-noise mask lives here too, because both tiers read it.
//!
//! Depends on [`tang`] and nothing else.

pub mod bvh;
pub mod geometry;
pub mod math;
pub mod ray;
mod sah;
pub mod sampler;
pub mod tlas;

pub use bvh::{Bvh, BvhNode, FlatBvhNode};
pub use geometry::{Geometry, TriMesh, TriangleHit, intersect_triangle};
pub use math::{Aabb, Dir3, Point2, Point3, Transform, Vec2, Vec3, transform_from_column_major};
pub use ray::{Hit, Ray};
pub use tlas::{FlatTlasNode, Instance, InstanceHit, Tlas};
