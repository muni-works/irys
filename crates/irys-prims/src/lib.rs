#![warn(missing_docs)]

//! Stock [`Geometry`](irys_core::Geometry) implementations: analytic
//! primitives and heightfields. Splats sit in `irys-light`, because the CPU
//! integrator composites them itself.

// The modules were written as one crate; these keep their `crate::` paths.
use irys_core::{geometry, math, ray};
#[cfg(test)]
use irys_core::bvh;

pub mod analytic;
pub mod heightfield;

pub use analytic::{Analytic, Frame, Prim};
pub use heightfield::HeightField;
