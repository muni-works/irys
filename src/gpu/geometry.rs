//! The geometry seam of the GPU tier.
//!
//! The integrator owns light: camera rays, the path, the BSDF, the lights,
//! the environment, the accumulator and the denoiser. It owns no primitives.
//! A client supplies those, in two halves.
//!
//! # The WGSL half
//!
//! A [`GeometryModule`] carries a WGSL source string that is composed between
//! the renderer's prelude and its integrator. It must define, using whatever
//! representation it likes:
//!
//! ```text
//! fn trace_scene(origin: vec3<f32>, dir: vec3<f32>) -> RayHit
//! fn hit_normal(hit: RayHit) -> vec3<f32>
//! fn hit_tangent(hit: RayHit) -> vec3<f32>
//! fn hit_material_index(hit: RayHit) -> u32
//! fn hit_orientation(hit: RayHit) -> u32
//! ```
//!
//! `RayHit`, `MAX_T`, `EPSILON`, `FACE_IDX_MISS`, `FACE_IDX_GROUND`,
//! `ray_eps`, `offset_origin`, `intersect_aabb` and `shading_frame` come from
//! the prelude; `PI`, `GpuMaterial` and `onb` from the BSDF. `trace_scene`
//! must return `FACE_IDX_MISS` on a miss and must never return
//! `FACE_IDX_GROUND`, which the integrator reserves for its implicit ground.
//! `hit_tangent` may return the zero vector where the parameterisation is
//! degenerate. `hit_material_index` indexes the renderer's own `materials`
//! binding.
//!
//! # The binding half
//!
//! Browsers guarantee only ten storage buffers per compute stage, and the
//! renderer's half of bind group 0 uses five of them. The split is fixed:
//!
//! | binding | owner      | what                                        |
//! |---------|------------|---------------------------------------------|
//! | 0       | renderer   | `camera` uniform                            |
//! | 1..=5   | **client** | geometry slabs, read-only storage           |
//! | 6       | renderer   | `output` storage texture                    |
//! | 7       | renderer   | `render_state` uniform                      |
//! | 8       | renderer   | accumulation buffer (rw storage)            |
//! | 9       | renderer   | `materials` (storage)                       |
//! | 10      | renderer   | depth/normal + guide planes (rw storage)    |
//! | 11      | renderer   | `lights` (storage)                          |
//! | 12      | renderer   | feature-id buffer (rw storage)              |
//! | 13, 14  | renderer   | environment textures (not storage buffers)  |
//! | 15      | renderer   | `caustics` uniform                          |
//! | 16, 17  | renderer   | photon-map textures (not storage buffers)   |
//!
//! Five client storage buffers plus five renderer ones is exactly ten. That
//! ceiling is why the environment is a pair of textures rather than buffers,
//! and why the photon map is too: bindings 15..=17 add a uniform and two
//! sampled textures and no storage buffer. It is pinned by
//! `render_shader_fits_the_browser_storage_buffer_budget`.
//! A client that needs fewer than five slabs simply declares fewer; it may not
//! declare more.

/// The most storage-buffer bindings a geometry module may declare.
pub const MAX_GEOMETRY_BINDINGS: usize = 5;

/// One packed geometry buffer, ready to upload.
pub struct GeometrySlab<'a> {
    /// Debug label for the wgpu buffer.
    pub label: &'a str,
    /// The packed bytes. Must be non-empty: a zero-sized storage buffer is
    /// invalid, so pack at least one zeroed element.
    pub bytes: &'a [u8],
}

/// The static half of the seam: the client's WGSL and the layout of the
/// bindings it declares.
pub struct GeometryModule {
    /// WGSL implementing the contract above.
    pub wgsl: String,
    /// Layout entries for bindings 1..=5, in ascending binding order. At most
    /// five, all `Storage { read_only: true }` in practice.
    pub layout: Vec<wgpu::BindGroupLayoutEntry>,
}

/// The per-scene half: the packed buffers for one scene's geometry, in the
/// same order as [`GeometryModule::layout`].
pub trait GpuGeometry {
    /// The slabs to bind, one per entry of the module's layout.
    fn slabs(&self) -> Vec<GeometrySlab<'_>>;
}

/// A read-only storage-buffer layout entry for a compute stage, the shape
/// every geometry slab uses.
pub fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
