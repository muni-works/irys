//! kosm-render is now [irys](https://github.com/ecto/irys).
//!
//! This crate re-exports the `irys` facade so that every path that was
//! `kosm_render::X` — `kosm_render::Bvh`, `kosm_render::pathtrace::Scene`,
//! `kosm_render::gpu::ResidentScene` — still is. New code should depend on
//! `irys` directly.

pub use irys::*;
