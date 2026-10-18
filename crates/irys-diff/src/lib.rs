#![warn(missing_docs)]

//! Phase 1 of irys: the kernel contract, end to end, on the CPU.
//!
//! A BSDF is written once as a tang tensor-IR graph ([`ggx`]) and lowered
//! twice — eval (`f`, `pdf`) and adjoint (`δf → δθ`, from
//! `TensorGraph::backward`) — while its sampling is plain, detached f64.
//! [`prb`] is path replay backpropagation over a wavefront of those
//! vertices: a forward pass that keeps `L` per path, and a replay pass that
//! regenerates each path from its seed and pushes the loss adjoint into `θ`.
//!
//! A crate of its own rather than a module of irys-light: tang-compile is a
//! git dependency until it publishes, and irys-light has to stay
//! publishable from crates.io. When the IR ships, this folds into
//! irys-light.

pub mod ggx;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod interior;
pub mod ir;
pub mod prb;
pub mod qmc;
pub mod secondary;
pub mod vis;
