//! What the renderer needs to know about a scene, borrowed.
//!
//! Geometry is opaque — a handful of packed slabs and the WGSL that reads
//! them. Everything else (materials, the light rig, the environment) is the
//! renderer's own.

use super::buffers::{GpuAreaLight, GpuMaterial};
use super::geometry::GpuGeometry;
use crate::pathtrace::GpuEnvPack;

/// A scene, borrowed for the length of one call.
///
/// Clients build one of these from whatever their own scene type is; the
/// idiomatic move is `impl<'a> From<&'a MyScene> for SceneRef<'a>`, which lets
/// `&my_scene` be passed straight to every method here.
#[derive(Clone, Copy)]
pub struct SceneRef<'a> {
    /// The client's packed primitives.
    pub geometry: &'a dyn GpuGeometry,
    /// Materials, indexed by the geometry module's `hit_material_index`.
    pub materials: &'a [GpuMaterial],
    /// Area lights (softboxes). Intersectable, so BSDF sampling and NEE both
    /// find them and combine under MIS.
    pub lights: &'a [GpuAreaLight],
    /// Optional lat-long HDR environment. `None` uses the analytic gradient in
    /// the render state, matching `pathtrace::Environment::default()`.
    pub environment: Option<&'a GpuEnvPack>,
}

/// The empty geometry: no slabs. Only useful with a module that declares none.
pub struct NoGeometry;

impl GpuGeometry for NoGeometry {
    fn slabs(&self) -> Vec<super::geometry::GeometrySlab<'_>> {
        Vec::new()
    }
}
