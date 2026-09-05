//! A built-in geometry module: spheres and planes.
//!
//! Not a production tier — a real client packs its own primitives. This is
//! what the renderer's own tests trace, so they need no client at all, and it
//! is the smallest complete worked example of the [`super::geometry`]
//! contract: one storage buffer at binding 1, six functions in WGSL.

use bytemuck::{Pod, Zeroable};

use super::geometry::{GeometryModule, GeometrySlab, GpuGeometry, storage_entry};

/// One analytic primitive. Layout must match `AnalyticPrim` in `analytic.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct AnalyticPrim {
    /// 0 = sphere, 1 = plane.
    pub kind: u32,
    /// Index into the scene's materials.
    pub material_idx: u32,
    /// 0 = forward, 1 = reversed (the normal is flipped).
    pub orientation: u32,
    /// Padding to a 16-byte boundary.
    pub _pad: u32,
    /// Sphere: centre in `.xyz`, radius in `.w`. Plane: a point on it.
    pub a: [f32; 4],
    /// Sphere: unused. Plane: the unit normal.
    pub b: [f32; 4],
}

impl AnalyticPrim {
    /// A sphere.
    pub fn sphere(center: [f32; 3], radius: f32, material_idx: u32) -> Self {
        Self {
            kind: 0,
            material_idx,
            orientation: 0,
            _pad: 0,
            a: [center[0], center[1], center[2], radius],
            b: [0.0; 4],
        }
    }

    /// An unbounded plane through `point` with unit normal `normal`.
    pub fn plane(point: [f32; 3], normal: [f32; 3], material_idx: u32) -> Self {
        Self {
            kind: 1,
            material_idx,
            orientation: 0,
            _pad: 0,
            a: [point[0], point[1], point[2], 0.0],
            b: [normal[0], normal[1], normal[2], 0.0],
        }
    }
}

/// A scene's worth of analytic primitives.
#[derive(Clone, Debug, Default)]
pub struct AnalyticGeometry {
    /// The primitives, traced linearly. `trace_scene` returns the index into
    /// this list as the hit's `face_idx`.
    pub prims: Vec<AnalyticPrim>,
}

impl AnalyticGeometry {
    /// The WGSL and the single binding this module needs.
    pub fn module() -> GeometryModule {
        GeometryModule {
            wgsl: super::shaders::ANALYTIC_SHADER.to_string(),
            layout: vec![storage_entry(1)],
        }
    }
}

/// A zero-length storage buffer is invalid, so an empty scene binds one
/// zeroed primitive; `arrayLength` is what the shader loops over, and a
/// zeroed sphere of radius 0 is never hit.
static ZEROS: [u8; std::mem::size_of::<AnalyticPrim>()] = [0; std::mem::size_of::<AnalyticPrim>()];

impl GpuGeometry for AnalyticGeometry {
    fn slabs(&self) -> Vec<GeometrySlab<'_>> {
        vec![GeometrySlab {
            label: "Analytic Prims",
            bytes: if self.prims.is_empty() {
                &ZEROS
            } else {
                bytemuck::cast_slice(&self.prims)
            },
        }]
    }
}
