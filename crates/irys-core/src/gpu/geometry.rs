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

/// Deepest root-to-leaf path a client's WGSL traversal is expected to walk.
///
/// A stack-based BVH/TLAS walk in WGSL holds its stack in a fixed
/// `array<u32, N>` and *silently drops* a push that would overflow it —
/// geometry simply disappears from the render, with no diagnostic at all.
/// Sixty-four, not the thirty-two such traversals started at: a merged offline
/// scene (one BLAS per object folded into a single tree) is deeper than any
/// viewport scene.
///
/// The renderer owns no primitives, so it cannot enforce this in a shader of
/// its own; [`validate_tree_depth`] is the gate a geometry module calls before
/// upload, so the bound is checked rather than hoped for.
pub const MAX_TRAVERSAL_DEPTH: usize = 64;

/// Measure a flattened tree and refuse one deeper than [`MAX_TRAVERSAL_DEPTH`].
///
/// `nodes` is the array [`crate::bvh::Bvh::flatten`] produces, in the same
/// order: node 0 is the root, an internal node's `left_or_first` and
/// `right_or_count` are child indices, and a leaf's are a range into the
/// primitive-index list. An empty tree is depth 0 and passes.
///
/// Cycles and out-of-range indices cannot arise from `flatten`, but a client
/// may pack its own array, so the walk visits each node at most once rather
/// than trusting the shape.
pub fn validate_tree_depth(nodes: &[crate::bvh::FlatBvhNode]) -> Result<(), super::GpuError> {
    let depth = tree_depth(nodes);
    if depth > MAX_TRAVERSAL_DEPTH {
        return Err(super::GpuError::InvalidInput(format!(
            "packed tree is {depth} levels deep (max {MAX_TRAVERSAL_DEPTH}) -- a WGSL \
             traversal stack cannot hold it and would silently drop geometry. Build \
             shallower (fewer primitives per tree, or a two-level TLAS), or trace on \
             the CPU"
        )));
    }
    Ok(())
}

/// The deepest root-to-leaf path in a flattened tree, counting the root as 1.
///
/// Iterative: a 64-deep tree is fine on the stack, but a malformed array that
/// chains a million nodes should return a number, not blow the host's.
pub fn tree_depth(nodes: &[crate::bvh::FlatBvhNode]) -> usize {
    if nodes.is_empty() {
        return 0;
    }
    let mut best = 0usize;
    let mut visited = vec![false; nodes.len()];
    let mut stack = vec![(0u32, 1usize)];
    while let Some((idx, depth)) = stack.pop() {
        let Some(&(_, is_leaf, left, right)) = nodes.get(idx as usize) else {
            continue;
        };
        if std::mem::replace(&mut visited[idx as usize], true) {
            continue;
        }
        best = best.max(depth);
        if !is_leaf {
            stack.push((left, depth + 1));
            stack.push((right, depth + 1));
        }
    }
    best
}
