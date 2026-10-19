//! Image to CAD, v2 and v3: a learned scene proposer and the MDL search
//! over a faithful render.
//!
//! See `docs/image-to-cad.md`, "v2: neural" and "v3".
//!
//! - [`scene`]: the multi-object scene language and its sampler.
//! - [`render`]: irys renders of a scene — the photo (GPU, OpenPBR), the
//!   labels and the linear-appearance model image (CPU, irys TLAS).
//! - [`data`]: the seeded data engine and its on-disk shards.
//! - [`net`]: the network, a tang tensor-IR graph.
//! - [`train`]: running it (compiled C, f32), supervised and as a policy.
//! - [`infer`]: decoding heads into proposals.
//! - [`search`]: proposals into the MDL search and LM fit (v2).
//! - [`faithful`] (v3): the compare image as irys's OpenPBR scene kernel
//!   renders it, with the appearance fitted to the photo.

pub mod data;
pub mod eval;
#[cfg(feature = "gpu")]
pub mod faithful;
pub mod infer;
pub mod net;
pub mod render;
pub mod scene;
pub mod search;
pub mod train;
