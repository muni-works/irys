//! Image to CAD, v2: a learned scene proposer.
//!
//! See `docs/image-to-cad.md`, "v2: neural".
//!
//! - [`scene`]: the multi-object scene language and its sampler.
//! - [`render`]: irys renders of a scene — the photo (GPU, OpenPBR), the
//!   labels and the linear-appearance model image (CPU, irys TLAS).
//! - [`data`]: the seeded data engine and its on-disk shards.
//! - [`net`]: the network, a tang tensor-IR graph.
//! - [`train`]: running it (compiled C, f32), supervised and as a policy.

pub mod data;
pub mod net;
pub mod render;
pub mod scene;
pub mod train;
