//! Image to CAD, v0 (sim-to-sim): recover a vcad program and its
//! parameters from irys renders. See `docs/image-to-cad.md`, "v0 results".
//!
//! - [`dsl`]: the program family (a typed vcad DSL), text round-trip.
//! - [`generate`]: seeded random valid programs.
//! - [`canon`]: the normal form; topology and dimension errors.
//! - [`geom`]: tessellation and d(vertices)/d(θ).
//! - [`render`], [`model`]: irys renders of a program in known views.
//! - [`jac`]: the image Jacobian (irys's primary edge term on vertex velocities).
//! - [`fit`]: Levenberg–Marquardt, blur homotopy, silhouette distance term.
//! - [`mdl`]: the score and the judge (Fisher information).
//! - [`search`]: the outer loop and the `Proposer` seam.

pub mod canon;
pub mod dsl;
pub mod fit;
pub mod generate;
pub mod geom;
pub mod jac;
pub mod mdl;
pub mod model;
pub mod render;
pub mod search;
pub mod tag;
