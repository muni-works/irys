//! The CPU path tracer, re-exported.
//!
//! The tier now lives in [`crate::cpu`], one file per concern, so that it
//! mirrors [`crate::gpu`]. This module is the old path kept intact: every
//! name that was `kosm_render::pathtrace::X` still is.

pub use crate::cpu::*;
