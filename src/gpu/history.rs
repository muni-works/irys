//! Per-pixel history and denoising, on the device.
//!
//! [`RayTracePipeline::render_resident_linear`] hands a host one raw linear
//! sample per pass and leaves it to keep the history itself. That works, and
//! for a viewport it is the wrong place for the work: folding the sample in is
//! cheap, but running [`crate::pathtrace::denoise`] over the whole frame every
//! pass is not — measured from Kosm's viewer at 512x288, 420 ms of CPU filter
//! against 25 ms of tracing. The filter, not the path tracer, was the frame.
//!
//! [`HistoryBuffers`] moves all of it onto the GPU. The running mean, the
//! per-pixel sample count and the moments that feed the variance estimate live
//! in buffers beside the resident scene; a small compute pass folds each raw
//! sample in, an à-trous pass filters the mean guided by the resident depth,
//! normal and albedo planes, and a resolve pass tonemaps into a storage
//! texture the caller owns. Nothing is read back, so a viewport can blit the
//! texture straight to its surface.
//!
//! # What the caller still owns
//!
//! **The keep mask.** The device side takes one byte per pixel, 1 to keep
//! accumulating and 0 to restart that pixel: everything the device cannot
//! know, from a re-pose to a region the caller wants re-converged. An empty
//! slice is all-1.
//!
//! # One denoise a frame, not one a box
//!
//! [`RayTracePipeline::accumulate_and_denoise_resident`] is the whole of a
//! pass in one call, and for a host with one dirty rectangle a frame that is
//! the right shape. A host with several is a different matter: the trace and
//! the fold scissor down to a box, but the denoise chain cannot — the filter
//! reaches 32 pixels off a box's edge and the resolve has to leave the target
//! texture whole — so `k` boxes through the fused call is `k` full-frame
//! filters to show one frame.
//!
//! [`RayTracePipeline::accumulate_resident`] and
//! [`RayTracePipeline::denoise_and_resolve_resident`] are the two halves on
//! their own: call the first once per box and the second once, and the chain
//! runs once. The fused calls are thin wrappers over the pair, so a host with
//! one box needs to know none of this.
//!
//! **Reprojection** used to be on that list, and a host without one had only
//! the blunt instrument: upload an all-zero mask on a camera move and watch
//! every pixel of an orbit restart from a single sample, most of them the
//! same surface seen from a hair to the left.
//! [`RayTracePipeline::accumulate_and_denoise_resident_reprojected`] does it
//! on the device instead — hand it the previous pass's camera and it carries
//! each pixel's mean and count across the move, restarting only what the move
//! actually disoccluded. See [`HistoryBuffers`]' `prev_guides`.
//!
//! # What moved, per pixel rather than per rectangle
//!
//! A camera-only reprojection still leaves the host with the *world's* motion
//! to express, and the only instrument for that was the keep mask: a rectangle
//! around each moved object's old and new pose, every pixel inside it
//! restarted from one sample. That rectangle is visible. It is a box of grain
//! travelling with the ball, and it is grain over pixels — the ball itself,
//! most of all — whose shading did not change at all.
//!
//! [`InstanceMotion`] replaces it. The host says which instance each of the
//! geometry module's primitives belongs to and where each instance was last
//! frame; the reprojection carries a pixel's world point back through its own
//! instance's transform before projecting it, and validates on the hit's
//! **identity** as well as its depth and normal. So the ball keeps its own
//! shading while it flies, and the floor uncovered behind it — a different id
//! at a different depth — is the only thing that restarts. There is no
//! rectangle, so there is nothing to see the edge of.
//!
//! Two things finish it, because reprojection alone is not enough:
//!
//! * [`GpuDenoiseParams::history_cap`] bounds the fold. An unbounded running
//!   mean over four hundred frames cannot be moved by what the pixel is
//!   seeing now; an exponential moving average with a 1/64 floor converges
//!   just as far and then keeps up.
//! * [`GpuDenoiseParams::clamp_k`] catches what no geometric test can see —
//!   lighting that went stale under a surface that did not move, which is
//!   exactly the shadow a ball leaves behind on the floor. A pixel whose
//!   history disagrees with this pass's neighbourhood by more than the error
//!   bar on that neighbourhood has its history *shortened* to
//!   [`GpuDenoiseParams::clamp_reset`], and the next few samples carry it the
//!   rest of the way. Shortened, not overwritten: snapping the colour is the
//!   usual TAA move and it is biased, and a hundred still passes of a biased
//!   nudge is a tint.
//!
//! And [`GpuDenoiseParams::spatial_variance`] is what keeps the pixels that
//! *do* restart from showing it: a pixel with fewer than four samples is
//! given SVGF's 7x7 spatial variance instead of its own two temporal moments,
//! so the à-trous filter is wide and correct on the frame the pixel appears
//! rather than one convergence later.
//!
//! # Parity with the CPU filter
//!
//! [`HistoryBuffers::denoise_params`] defaults to
//! [`crate::pathtrace::PathTraceOptions`]'s filter constants, and the shader
//! is a port of [`crate::pathtrace::denoise`] weight for weight: same
//! B3-spline taps, same normal/depth/luminance edge stops, same albedo
//! demodulation with the same floor, same variance prefilter. A one-sample
//! history therefore denoises to what the CPU would have produced from the
//! same [`crate::pathtrace::Film`], which is what `tests/gpu_history.rs`
//! checks.
//!
//! The one deliberate difference is the fade: the filter's strength falls
//! linearly with the pixel's history length and reaches zero at
//! [`GpuDenoiseParams::count_cutoff`] samples, at which point the à-trous pass
//! skips the pixel entirely. A converged pixel needs no filter and paying for
//! one only softens it.

use super::context::{GpuContext, GpuError};
use bytemuck::{Pod, Zeroable};

use super::buffers::{GpuCamera, GpuRenderState};
use super::pipeline::{RayTracePipeline, read_back_f32};
use super::resident::ResidentScene;

/// The most à-trous iterations one call may dispatch.
///
/// Each iteration needs its own slot in the parameter buffer, which is sized
/// once. Five is [`crate::pathtrace::PathTraceOptions`]'s default and doubles
/// the tap stride each time, so the widest footprint is already 32 pixels.
pub const MAX_DENOISE_ITERS: u32 = 8;

/// How many à-trous iterations a pixel with `count` samples of history still
/// gets: all `iters` of them on its first sample, falling linearly to none at
/// `count_cutoff`.
///
/// The mirror of `atrous_iters_for` in `history.wgsl`, exposed so a caller can
/// budget the pass and a test can pin the two together. Note what it gives at
/// `count == 1`: the full count, so a one-sample history is filtered exactly
/// as [`crate::pathtrace::denoise`] filters a `Film`.
pub fn atrous_iters_for(count: f32, iters: u32, count_cutoff: u32) -> u32 {
    let cutoff = count_cutoff.max(1) as f32;
    let t = ((cutoff - count) / (cutoff - 1.0).max(1e-6)).clamp(0.0, 1.0);
    (iters as f32 * t).ceil() as u32
}

/// Uniform slots: one per à-trous iteration, one shared by the accumulate,
/// demodulate and resolve passes, and one for the reprojection.
///
/// The reprojection needs a slot of its own because it is the one pass that
/// always covers the whole frame: an accumulate scissored to a box drives
/// slot 0 with the box's dispatch origin, and the reprojection cannot inherit
/// that origin.
const PARAM_SLOTS: u32 = MAX_DENOISE_ITERS + 2;

/// The parameter slot [`RayTracePipeline::accumulate_resident`] drives the
/// reprojection from; see [`PARAM_SLOTS`].
const REPROJECT_SLOT: u32 = MAX_DENOISE_ITERS + 1;

/// Uniform buffer offsets must be a multiple of this on every backend we
/// target, so each parameter slot is padded out to it.
const PARAM_STRIDE: u64 = 256;

/// Filter settings for [`RayTracePipeline::accumulate_and_denoise_resident`].
///
/// The `sigma_*` fields and `iters` mean exactly what the identically-named
/// fields of [`crate::pathtrace::PathTraceOptions`] mean, and default to the
/// same values, so the two tiers filter the same way.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuDenoiseParams {
    /// Number of à-trous iterations, clamped to [`MAX_DENOISE_ITERS`]. Zero
    /// turns the filter off and tonemaps the bare running mean.
    pub iters: u32,
    /// Tolerance on the normal guide.
    pub sigma_normal: f32,
    /// Tolerance on the depth guide, relative to the centre pixel's depth.
    pub sigma_depth: f32,
    /// Tolerance on illumination, in units of the pixel's own error bar.
    pub sigma_lum: f32,
    /// History length at which the filter has fully faded out. A pixel with at
    /// least this many samples is passed through untouched — the temporal mean
    /// is already cleaner than anything a spatial filter would leave.
    pub count_cutoff: u32,
    /// Linear exposure applied before the tonemap curve.
    pub exposure: f32,
    /// The longest history a pixel may hold, in samples.
    ///
    /// Past it the fold stops being a true mean and becomes an exponential
    /// moving average with a fixed `1/history_cap` weight. A still picture
    /// converges exactly as far — the estimator's floor is
    /// `1/history_cap` of the noise, well under a display bit at 64 — and a
    /// live one keeps up, because a pixel whose lighting changed is no longer
    /// outvoted four hundred to one by frames that saw the old lighting.
    ///
    /// Zero is read as one. Set it very large for a `--shot`-style render
    /// that is only ever going to converge.
    pub history_cap: u32,
    /// Neighbourhood colour clamping, in standard deviations of this pass's
    /// raw 3x3 neighbourhood. Zero turns it off.
    ///
    /// This is the only thing that catches lighting that went stale *without*
    /// the geometry moving under the pixel — a ball's shadow left behind on
    /// the floor. The depth, normal and id gates all say the floor is still
    /// the floor; the raw sample says it is not that colour any more.
    pub clamp_k: f32,
    /// The history length a clamped pixel drops to.
    ///
    /// Not one: a clamped pixel has been seen before and only needs its
    /// brightness re-earned, and restarting it from a single sample puts the
    /// grain back. Small enough that the à-trous filter widens there.
    pub clamp_reset: u32,
    /// Estimate a short-history pixel's variance from a 7x7 spatial
    /// neighbourhood rather than from its own two temporal moments — SVGF's
    /// spatiotemporal estimator.
    ///
    /// On (the default) a pixel with fewer than four samples is filtered as
    /// wide as its neighbours say it needs on the frame it appears. Off
    /// reproduces [`crate::pathtrace::denoise`] exactly.
    pub spatial_variance: bool,
}

impl Default for GpuDenoiseParams {
    fn default() -> Self {
        let d = crate::pathtrace::PathTraceOptions::default();
        Self {
            iters: d.denoise_iters,
            sigma_normal: d.sigma_normal,
            sigma_depth: d.sigma_depth,
            sigma_lum: d.sigma_lum,
            count_cutoff: 32,
            exposure: 1.0,
            history_cap: 64,
            clamp_k: 4.0,
            clamp_reset: 2,
            spatial_variance: true,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HistoryParams {
    width: u32,
    height: u32,
    count_cutoff: u32,
    iters: u32,
    sigma_lum: f32,
    sigma_depth: f32,
    sigma_normal: f32,
    exposure: f32,
    stride: u32,
    src_is_b: u32,
    scissor_xy: u32,
    scissor_wh: u32,
    // `reproject` only; zero elsewhere. Both views as the shader's ray
    // generator builds them, then (tan(fov/2), aspect) for each.
    cur_eye: [f32; 4],
    cur_right: [f32; 4],
    cur_up: [f32; 4],
    cur_forward: [f32; 4],
    prev_eye: [f32; 4],
    prev_right: [f32; 4],
    prev_up: [f32; 4],
    prev_forward: [f32; 4],
    view_params: [f32; 4],
    reprojected: u32,
    iter_index: u32,
    // The frame-space pixel the dispatch's (0, 0) invocation stands on.
    // `accumulate` dispatches over its scissor box's workgroups rather than
    // the frame's, so its invocation ids have to be shifted onto the box's
    // corner; every other pass covers the frame and leaves these zero.
    origin_x: u32,
    origin_y: u32,
    history_cap: u32,
    clamp_k: f32,
    clamp_reset: u32,
    motion_instances: u32,
    motion_ids: u32,
    spatial_variance: u32,
    _pad0: u32,
    _pad1: u32,
}

/// The camera basis the shader's ray generator derives from a [`GpuCamera`],
/// reproduced here so the reprojection pass can be told about a view it is
/// not currently rendering.
fn view_basis(cam: &GpuCamera) -> ([f32; 4], [f32; 4], [f32; 4], [f32; 4], f32, f32) {
    let norm = |v: [f32; 3]| {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / l, v[1] / l, v[2] / l]
    };
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let eye = [cam.position[0], cam.position[1], cam.position[2]];
    let fwd = norm([
        cam.target[0] - eye[0],
        cam.target[1] - eye[1],
        cam.target[2] - eye[2],
    ]);
    let right = norm(cross(fwd, [cam.up[0], cam.up[1], cam.up[2]]));
    let up = cross(right, fwd);
    let v4 = |v: [f32; 3]| [v[0], v[1], v[2], 0.0];
    (
        [eye[0], eye[1], eye[2], 1.0],
        v4(right),
        v4(up),
        v4(fwd),
        (cam.fov * 0.5).tan(),
        cam.width as f32 / cam.height as f32,
    )
}

/// Per-instance object motion for one frame, packed for the reprojection
/// pass.
///
/// The device-side reprojection can carry a pixel's history across a *camera*
/// move on its own — it has this frame's depth and both views. It cannot
/// carry it across an *object* move, because nothing on the device knows that
/// the ball is somewhere else than it was. The host does: it placed the
/// instance both times.
///
/// So the host hands over two things a frame:
///
/// * `instance_of_id` — the instance each of the geometry module's primitive
///   ids belongs to, indexed by that id. [`InstanceMotion::STATIC`] for a
///   primitive that did not move, which is most of a scene and costs nothing.
/// * `transforms` — one 3x4 row-major matrix per instance, `prev_T · cur_T⁻¹`:
///   given a world point where this frame put it, where the previous frame
///   had it. The identity for an instance that did not move.
///
/// The reprojection then reads the hit id out of the guide planes, looks up
/// the instance, moves the world point back, and projects *that* through the
/// previous camera — so a ball in flight keeps its own shading and the floor
/// revealed behind it restarts.
#[derive(Debug, Clone, Default)]
pub struct InstanceMotion {
    packed: Vec<[f32; 4]>,
    ids: u32,
    instances: u32,
}

impl InstanceMotion {
    /// The instance slot for a primitive with no motion of its own.
    pub const STATIC: u32 = u32::MAX;

    /// The 3x4 identity, for an instance that did not move this frame.
    pub const IDENTITY: [f32; 12] = [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0,
    ];

    /// Pack one frame's motion.
    ///
    /// `instance_of_id[id]` is the instance the geometry module's primitive
    /// `id` belongs to, or [`InstanceMotion::STATIC`]. `transforms[k]` is
    /// instance `k`'s `prev_T · cur_T⁻¹`, row-major 3x4.
    pub fn new(instance_of_id: &[u32], transforms: &[[f32; 12]]) -> Self {
        let table = instance_of_id.len().div_ceil(4);
        let mut packed = Vec::with_capacity(table + transforms.len() * 3);
        for chunk in instance_of_id.chunks(4) {
            let mut v = [f32::from_bits(Self::STATIC); 4];
            for (k, &id) in chunk.iter().enumerate() {
                v[k] = f32::from_bits(id);
            }
            packed.push(v);
        }
        for m in transforms {
            packed.push([m[0], m[1], m[2], m[3]]);
            packed.push([m[4], m[5], m[6], m[7]]);
            packed.push([m[8], m[9], m[10], m[11]]);
        }
        Self {
            packed,
            ids: instance_of_id.len() as u32,
            instances: transforms.len() as u32,
        }
    }

    fn bytes(&self) -> &[u8] {
        bytemuck::cast_slice(self.packed.as_slice())
    }
}

/// The running mean and sample count read back off the device.
///
/// For tests and for a host that wants to checkpoint a converged frame; the
/// render path never needs it.
#[derive(Debug, Clone)]
pub struct History {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Running mean of the linear radiance, 3 floats per pixel.
    pub rgb: Vec<f32>,
    /// Running mean of the path tracer's coverage, one float per pixel.
    pub alpha: Vec<f32>,
    /// How many samples each pixel's mean is over.
    pub count: Vec<u32>,
    /// Variance of each pixel's mean luminance, in
    /// [`crate::pathtrace::Film::variance`]'s convention.
    pub variance: Vec<f32>,
}

/// The device-side history for one resident scene, at one frame size.
///
/// Built on first use by [`RayTracePipeline::accumulate_and_denoise_resident`]
/// and thrown away whenever the scene is resized.
pub struct HistoryBuffers {
    width: u32,
    height: u32,
    /// (linear radiance, coverage), the running mean.
    ///
    /// Visible to the `gpu` module because the neural denoiser in
    /// [`super::neural`] reads it, the stats and the à-trous scratch directly
    /// — it stands exactly where a wavelet iteration stands.
    pub(super) mean: wgpu::Buffer,
    /// (count, luminance sum, luminance-squared sum, variance of the mean).
    pub(super) stats: wgpu::Buffer,
    /// The caller's keep mask, widened to one `u32` per pixel.
    keep: wgpu::Buffer,
    /// The *previous* pass's guide plane 1 — (normal, distance from that
    /// pass's eye) — one vec4 per pixel. Copied out of the resident scene's
    /// depth/normal buffer at the end of every pass, so the next pass's
    /// reprojection has something to test against. Zeroed until the first
    /// pass has run, which reads as "restart everything".
    ///
    /// Two planes, not one: plane 0 is (normal, depth) and plane 1 is
    /// (albedo, biased hit id), because the reprojection validates on the id
    /// as well as on the geometry.
    prev_guides: wgpu::Buffer,
    /// One frame's packed [`InstanceMotion`], or a stub when nothing moved.
    motion: wgpu::Buffer,
    /// (illumination, variance) ping-pong for the wavelet iterations.
    pub(super) scratch_a: wgpu::Buffer,
    pub(super) scratch_b: wgpu::Buffer,
    /// One uniform slot per pass; see [`PARAM_SLOTS`].
    params: wgpu::Buffer,
    /// Staging for [`RayTracePipeline::read_history`], allocated on first use.
    readback: Option<wgpu::Buffer>,
    /// Scratch for widening the caller's `u8` mask, kept so a per-frame
    /// upload allocates nothing.
    keep_staging: Vec<u32>,
    /// A 1x1 storage texture standing in for the resolve target on the passes
    /// that have no target and never write one. The bind group layout demands
    /// a texture at binding 8; `accumulate_resident` has no business asking
    /// its caller for one.
    placeholder: wgpu::TextureView,
}

impl HistoryBuffers {
    pub(super) fn new(ctx: &GpuContext, width: u32, height: u32) -> Self {
        let n = (width as u64) * (height as u64);
        let mk = |label: &str, size: u64| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        Self {
            width,
            height,
            mean: mk("History Mean", n * 16),
            stats: mk("History Stats", n * 16),
            keep: mk("History Keep Mask", n * 4),
            prev_guides: mk("History Previous Guides", n * 32),
            motion: mk("History Instance Motion", 64),
            scratch_a: mk("History Scratch A", n * 16),
            scratch_b: mk("History Scratch B", n * 16),
            params: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("History Params"),
                size: PARAM_STRIDE * PARAM_SLOTS as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            readback: None,
            keep_staging: Vec::new(),
            placeholder: ctx
                .device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("History Placeholder Target"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::STORAGE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default()),
        }
    }

    /// The stand-in bound where the resolve target goes on a pass that has
    /// none; see `placeholder`.
    fn placeholder_view(&self) -> &wgpu::TextureView {
        &self.placeholder
    }

    /// The frame size this history is allocated for.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Forget every pixel's history.
    ///
    /// Equivalent to passing an all-zero keep mask to the next pass, and
    /// cheaper: it is a buffer clear rather than an upload.
    pub fn clear(&self, ctx: &GpuContext) {
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("History Clear"),
            });
        enc.clear_buffer(&self.mean, 0, None);
        enc.clear_buffer(&self.stats, 0, None);
        // A reprojection into a history that is gone would carry zeros, which
        // is harmless, but forgetting the previous view too keeps "cleared"
        // meaning exactly one thing.
        enc.clear_buffer(&self.prev_guides, 0, None);
        ctx.queue.submit(Some(enc.finish()));
    }
}

/// The three compute pipelines the history passes run, plus their shared
/// layout.
///
/// Built once and reused; hand it to
/// [`RayTracePipeline::accumulate_and_denoise_resident`] every pass.
pub struct HistoryPipeline {
    reproject: wgpu::ComputePipeline,
    accumulate: wgpu::ComputePipeline,
    demodulate: wgpu::ComputePipeline,
    atrous: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl HistoryPipeline {
    /// Compile the history and denoise passes.
    pub fn new(ctx: &GpuContext) -> Result<Self, GpuError> {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("History Shader"),
                source: wgpu::ShaderSource::Wgsl(super::shaders::HISTORY_SHADER.into()),
            });

        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("History Bind Group Layout"),
                entries: &[
                    // Params, one slot per pass, selected by dynamic offset.
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: true,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<HistoryParams>() as u64,
                            ),
                        },
                        count: None,
                    },
                    storage(1, true),  // raw sample
                    storage(2, true),  // guide planes
                    storage(3, false), // mean
                    storage(4, false), // stats
                    storage(5, true),  // keep mask
                    storage(6, false), // scratch src
                    storage(7, false), // scratch dst
                    wgpu::BindGroupLayoutEntry {
                        binding: 8,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    storage(9, true),  // the previous pass's guide planes
                    storage(10, true), // per-instance object motion
                ],
            });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("History Pipeline Layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });

        let mk = |entry: &str| {
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };

        Ok(Self {
            reproject: mk("reproject"),
            accumulate: mk("accumulate"),
            demodulate: mk("demodulate"),
            atrous: mk("atrous"),
            resolve: mk("resolve"),
            layout,
        })
    }
}

/// One history compute pass: bind the parameter slot and dispatch `groups`
/// workgroups.
fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    group: &wgpu::BindGroup,
    slot: u32,
    groups: (u32, u32),
    label: &str,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, group, &[(PARAM_STRIDE * slot as u64) as u32]);
    pass.dispatch_workgroups(groups.0.max(1), groups.1.max(1), 1);
}

/// The history passes' one bind group, over a chosen scratch source and
/// destination so the wavelet iterations can ping-pong under a fixed layout.
///
/// `target` is the resolve pass's output texture. The accumulate half of a
/// frame never touches it and has none to hand over, so it passes `None` and
/// gets the scene's own output view bound in its place — a binding the passes
/// it dispatches do not write.
#[allow(clippy::too_many_arguments)]
fn history_bind_group(
    ctx: &GpuContext,
    history_pipeline: &HistoryPipeline,
    hist: &HistoryBuffers,
    raw: &wgpu::Buffer,
    guides: &wgpu::Buffer,
    src: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    target: Option<&wgpu::TextureView>,
    label: &str,
) -> wgpu::BindGroup {
    let placeholder;
    let view = match target {
        Some(v) => v,
        None => {
            placeholder = hist.placeholder_view();
            placeholder
        }
    };
    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &history_pipeline.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &hist.params,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<HistoryParams>() as u64),
                }),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: raw.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: guides.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: hist.mean.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: hist.stats.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: hist.keep.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: src.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: dst.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: hist.prev_guides.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: hist.motion.as_entire_binding(),
            },
        ],
    })
}

impl RayTracePipeline {
    /// Trace one pass into the device-side history, scissored to one box.
    ///
    /// The first half of [`RayTracePipeline::accumulate_and_denoise_resident`],
    /// split out so a caller with several dirty rectangles pays for the
    /// denoise chain once rather than once per rectangle. Call it once per
    /// box, then [`RayTracePipeline::denoise_and_resolve_resident`] once, and
    /// the result is what the fused call would have produced from the same
    /// boxes — byte for byte, when the boxes tile the frame.
    ///
    /// Everything here is **scissored to the box** that `state` carries (see
    /// `GpuRenderState::set_scissor`): the trace dispatches only over the
    /// rectangle, the accumulate dispatch covers only the rectangle's
    /// workgroups rather than the frame's, and the keep mask upload is only
    /// the rows the rectangle touches. Every pixel outside keeps the mean, the
    /// sample count and the variance it already had. With no scissor the box
    /// is the whole frame, which is the fused call's behaviour exactly.
    ///
    /// `keep` is still indexed over the **whole frame** — one byte per pixel,
    /// row-major, 1 keeps that pixel's history and 0 restarts it — so the same
    /// mask can be handed to every box of a pass. Only the entries the box
    /// covers are read, and only the rows it touches are uploaded. An empty
    /// slice is read as all-1 and still costs the (small) upload, because the
    /// resident mask may hold a previous pass's zeros.
    ///
    /// `prev_view` runs the device-side reprojection, and is a **once per
    /// frame** thing: the pass gathers over the whole frame into the scratch
    /// pair and every later box in the same frame reads that gather, so pass
    /// the previous camera on the first box only and `None` on the rest.
    /// Passing it on a later box would re-gather from a history that has
    /// already been folded into. See
    /// [`RayTracePipeline::accumulate_and_denoise_resident_reprojected`] for
    /// what the reprojection tests.
    ///
    /// `state` is forced into raw-sample mode with its refinement pass off,
    /// exactly as [`RayTracePipeline::render_resident_linear`] forces them.
    ///
    /// The work is submitted and this returns immediately.
    #[allow(clippy::too_many_arguments)]
    pub fn accumulate_resident(
        &self,
        ctx: &GpuContext,
        history_pipeline: &HistoryPipeline,
        res: &mut ResidentScene,
        camera: &GpuCamera,
        state: GpuRenderState,
        keep: &[u8],
        prev_view: Option<&GpuCamera>,
    ) -> Result<(), GpuError> {
        self.accumulate_resident_temporal(
            ctx,
            history_pipeline,
            res,
            camera,
            state,
            keep,
            prev_view,
            &GpuDenoiseParams::default(),
            None,
        )
    }

    /// [`RayTracePipeline::accumulate_resident`], told what moved.
    ///
    /// Two things the plain call cannot express:
    ///
    /// * `motion` — one frame's [`InstanceMotion`]. With it the reprojection
    ///   follows a *moving object*: a pixel's world point is carried back
    ///   through its own instance's transform before being projected into the
    ///   previous view, so a ball in flight keeps its own shading rather than
    ///   sampling whatever the floor behind it looked like. `None` is every
    ///   surface static, which is exactly the camera-only reprojection.
    ///
    /// * `denoise` — `accumulate` reads the temporal knobs off it:
    ///   [`GpuDenoiseParams::history_cap`], [`GpuDenoiseParams::clamp_k`] and
    ///   [`GpuDenoiseParams::clamp_reset`]. The filter fields are the
    ///   denoise call's business and are ignored here. Pass the same struct to
    ///   both halves of the frame.
    ///
    /// `motion` is a **once per frame** thing for the same reason `prev_view`
    /// is: the reprojection gathers over the whole frame on the first box.
    #[allow(clippy::too_many_arguments)]
    pub fn accumulate_resident_temporal(
        &self,
        ctx: &GpuContext,
        history_pipeline: &HistoryPipeline,
        res: &mut ResidentScene,
        camera: &GpuCamera,
        state: GpuRenderState,
        keep: &[u8],
        prev_view: Option<&GpuCamera>,
        denoise: &GpuDenoiseParams,
        motion: Option<&InstanceMotion>,
    ) -> Result<(), GpuError> {
        let (w, h) = res.size();
        let n = (w as usize) * (h as usize);
        if !keep.is_empty() && keep.len() != n {
            return Err(GpuError::InvalidInput(format!(
                "keep mask has {} entries, expected {w}x{h} = {n}",
                keep.len(),
            )));
        }

        res.ensure_history(ctx, w, h);

        // The box, clamped to the frame. A zero-size scissor means the whole
        // frame, which is what an unscissored state carries.
        let (bx, by, bw, bh) = match state.scissor() {
            Some([x, y, sw, sh]) => {
                let x = x.min(w);
                let y = y.min(h);
                (x, y, sw.min(w - x), sh.min(h - y))
            }
            None => (0, 0, w, h),
        };
        if bw == 0 || bh == 0 {
            return Ok(());
        }

        // The two views the reprojection pass works between. With no previous
        // view the pass is not dispatched and these are inert.
        let cur = view_basis(camera);
        let prev = view_basis(prev_view.unwrap_or(camera));
        // The reprojection is worth a dispatch when *either* view moved or
        // something in the scene did. A still camera over a moving ball is
        // the second case: the pixel is where it was and the surface under it
        // is not.
        let reproject =
            prev_view.is_some() || motion.map(|m| m.instances > 0).unwrap_or(false);

        // The motion table. Grown rather than reallocated per frame: a scene's
        // instance count barely moves, so after the first frame this is a
        // write into a buffer that already fits.
        let (motion_ids, motion_instances) = match motion {
            Some(m) if m.instances > 0 => {
                let bytes = m.bytes();
                let hist = res.history_mut().expect("history was just ensured");
                if (bytes.len() as u64) > hist.motion.size() {
                    hist.motion = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("History Instance Motion"),
                        size: (bytes.len() as u64).next_power_of_two(),
                        usage: wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_DST
                            | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    });
                }
                ctx.queue.write_buffer(&hist.motion, 0, bytes);
                (m.ids, m.instances)
            }
            _ => (0, 0),
        };

        {
            let hist = res.history_mut().expect("history was just ensured");

            // Widen the caller's mask, for the rows this box touches only.
            // WGSL has no 8-bit storage type, and one word per pixel is a
            // 590 KB upload at 512x288 for a whole frame — a box worth a
            // tenth of it pays about a tenth of that. Whole rows rather than
            // the exact rectangle: a buffer is row-major, so a sub-rectangle
            // is `bh` separate writes where a row range is one, and the extra
            // words are never read.
            let row_lo = (by as usize) * (w as usize);
            let row_hi = ((by + bh) as usize) * (w as usize);
            hist.keep_staging.clear();
            hist.keep_staging.reserve(row_hi - row_lo);
            if keep.is_empty() {
                hist.keep_staging.resize(row_hi - row_lo, 1);
            } else {
                hist.keep_staging
                    .extend(keep[row_lo..row_hi].iter().map(|&b| u32::from(b)));
            }
            ctx.queue.write_buffer(
                &hist.keep,
                (row_lo * 4) as u64,
                bytemuck::cast_slice(hist.keep_staging.as_slice()),
            );

            // Slot 0 drives reproject and accumulate here; the denoise call
            // rewrites it for demodulate and resolve. Only the fields those
            // two passes read matter, and the denoise parameters are not
            // among them.
            let base = HistoryParams {
                width: w,
                height: h,
                // Placeholders: `resolve` is the only pass that reads these
                // and it runs out of the denoise call's slot 0.
                count_cutoff: 1,
                iters: 0,
                sigma_lum: 0.0,
                sigma_depth: 0.0,
                sigma_normal: 0.0,
                exposure: 1.0,
                stride: 1,
                src_is_b: 0,
                // Straight from the trace pass's own state, so the two can
                // never disagree about which pixels this pass refreshed.
                scissor_xy: state.scissor_xy,
                scissor_wh: state.scissor_wh,
                cur_eye: cur.0,
                cur_right: cur.1,
                cur_up: cur.2,
                cur_forward: cur.3,
                prev_eye: prev.0,
                prev_right: prev.1,
                prev_up: prev.2,
                prev_forward: prev.3,
                view_params: [cur.4, cur.5, prev.4, prev.5],
                reprojected: u32::from(reproject),
                iter_index: 0,
                // `accumulate` runs over the box's workgroups only, so its
                // invocation ids start at the box's corner rather than the
                // frame's.
                origin_x: bx,
                origin_y: by,
                history_cap: denoise.history_cap.max(1),
                clamp_k: denoise.clamp_k.max(0.0),
                clamp_reset: denoise.clamp_reset.max(1),
                motion_instances,
                motion_ids,
                spatial_variance: u32::from(denoise.spatial_variance),
                _pad0: 0,
                _pad1: 0,
            };
            ctx.queue
                .write_buffer(&hist.params, 0, bytemuck::bytes_of(&base));
            // The reprojection gathers over the whole frame — a pixel inside
            // the box may have been outside it last frame — so it gets a slot
            // of its own with no origin. `REPROJECT_SLOT` is otherwise an
            // à-trous slot the denoise call rewrites, and the two calls never
            // read it at the same time.
            if reproject {
                let full = HistoryParams {
                    origin_x: 0,
                    origin_y: 0,
                    ..base
                };
                ctx.queue.write_buffer(
                    &hist.params,
                    PARAM_STRIDE * REPROJECT_SLOT as u64,
                    bytemuck::bytes_of(&full),
                );
            }
        }

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("History Accumulate Encoder"),
            });

        // One raw sample into the resident scene's own accumulation buffer,
        // with the guide planes filled.
        let mut state = state;
        state.set_raw_sample(true);
        state.refine_sample_count = 0;
        self.encode_raw_sample_into(ctx, res, camera, state, &mut encoder);

        let (raw, guides) = res.raw_and_guide_buffers();
        let hist = res.history().expect("history was just ensured");
        let ab = history_bind_group(
            ctx,
            history_pipeline,
            hist,
            raw,
            guides,
            &hist.scratch_a,
            &hist.scratch_b,
            None,
            "History Bind Group A->B",
        );

        // Before anything is folded in: carry what the previous view already
        // knew about each pixel onto this view's pixel grid.
        // `accumulate` picks the gather up out of the scratch pair.
        if reproject {
            dispatch(
                &mut encoder,
                &history_pipeline.reproject,
                &ab,
                REPROJECT_SLOT,
                (w.div_ceil(8), h.div_ceil(8)),
                "History Reproject",
            );
        }
        dispatch(
            &mut encoder,
            &history_pipeline.accumulate,
            &ab,
            0,
            (bw.div_ceil(8), bh.div_ceil(8)),
            "History Accumulate",
        );

        // Keep this pass's depth/normal plane for the next one to reproject
        // against, for the rows the trace just refreshed — outside them the
        // copy would put back what is already there. Guide plane 1 starts one
        // plane into the depth/normal buffer; plane 0 is the shader's own
        // edge-detection copy.
        let plane = (w as u64) * (h as u64) * 16;
        let row = (w as u64) * 16;
        for k in 0..2u64 {
            encoder.copy_buffer_to_buffer(
                guides,
                plane * (k + 1) + row * by as u64,
                &hist.prev_guides,
                plane * k + row * by as u64,
                row * bh as u64,
            );
        }

        ctx.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Denoise the running mean and tonemap it into `target` — the second
    /// half of [`RayTracePipeline::accumulate_and_denoise_resident`], run
    /// **once per frame** however many boxes were accumulated into it.
    ///
    /// Demodulate, `denoise.iters` à-trous iterations and resolve, all over
    /// the whole frame: the filter reaches up to 32 pixels off a box's edge
    /// and the resolve has to leave the target texture whole, so neither can
    /// be scissored to a box the way the accumulate can. What the split buys
    /// is that this chain runs once for `k` boxes rather than `k` times.
    ///
    /// The à-trous pass is not, though, the same cost every frame: a pixel's
    /// iteration budget falls with its history length (see
    /// [`atrous_iters_for`]) and reaches zero at `denoise.count_cutoff`, at
    /// which point the pass writes the pixel through and does none of its 25
    /// taps. A converged frame with one small moving box therefore still
    /// dispatches over the frame — every workgroup runs, and every invocation
    /// does its one read and one write — but only the box and its
    /// neighbourhood pay for the taps.
    ///
    /// `target` must be a view of an `Rgba8Unorm` texture with
    /// `STORAGE_BINDING` usage, at least the resident scene's size.
    pub fn denoise_and_resolve_resident(
        &self,
        ctx: &GpuContext,
        history_pipeline: &HistoryPipeline,
        res: &mut ResidentScene,
        denoise: &GpuDenoiseParams,
        target: &wgpu::TextureView,
    ) -> Result<(), GpuError> {
        let (w, h) = res.size();
        res.ensure_history(ctx, w, h);
        let iters = denoise.iters.min(MAX_DENOISE_ITERS);

        {
            let hist = res.history().expect("history was just ensured");
            // Slot 0 is shared by demodulate and resolve; slots 1..=iters
            // carry each wavelet iteration's tap stride. The accumulate calls
            // wrote slot 0 for their own passes; this overwrites it, and the
            // fields they cared about — the scissor, the reprojection flag,
            // the dispatch origin — are ones neither pass here reads.
            let base = HistoryParams {
                width: w,
                height: h,
                count_cutoff: denoise.count_cutoff.max(1),
                iters,
                sigma_lum: denoise.sigma_lum,
                sigma_depth: denoise.sigma_depth,
                sigma_normal: denoise.sigma_normal,
                exposure: denoise.exposure,
                stride: 1,
                // The final iteration lands in scratch_b when the count is
                // odd, since iteration 0 reads A and writes B.
                src_is_b: u32::from(iters % 2 == 1),
                scissor_xy: 0,
                scissor_wh: 0,
                cur_eye: [0.0; 4],
                cur_right: [0.0; 4],
                cur_up: [0.0; 4],
                cur_forward: [0.0; 4],
                prev_eye: [0.0; 4],
                prev_right: [0.0; 4],
                prev_up: [0.0; 4],
                prev_forward: [0.0; 4],
                view_params: [0.0; 4],
                reprojected: 0,
                iter_index: 0,
                origin_x: 0,
                origin_y: 0,
                history_cap: denoise.history_cap.max(1),
                clamp_k: denoise.clamp_k.max(0.0),
                clamp_reset: denoise.clamp_reset.max(1),
                motion_instances: 0,
                motion_ids: 0,
                spatial_variance: u32::from(denoise.spatial_variance),
                _pad0: 0,
                _pad1: 0,
            };
            ctx.queue
                .write_buffer(&hist.params, 0, bytemuck::bytes_of(&base));
            for it in 0..iters {
                let p = HistoryParams {
                    stride: 1u32 << it,
                    iter_index: it,
                    ..base
                };
                ctx.queue.write_buffer(
                    &hist.params,
                    PARAM_STRIDE * (1 + it) as u64,
                    bytemuck::bytes_of(&p),
                );
            }
        }

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("History Denoise Encoder"),
            });

        let (raw, guides) = res.raw_and_guide_buffers();
        let hist = res.history().expect("history was just ensured");

        // Two bind groups differing only in which scratch buffer is the
        // source, so the wavelet iterations can ping-pong under a fixed
        // layout.
        let ab = history_bind_group(
            ctx,
            history_pipeline,
            hist,
            raw,
            guides,
            &hist.scratch_a,
            &hist.scratch_b,
            Some(target),
            "History Bind Group A->B",
        );
        let ba = history_bind_group(
            ctx,
            history_pipeline,
            hist,
            raw,
            guides,
            &hist.scratch_b,
            &hist.scratch_a,
            Some(target),
            "History Bind Group B->A",
        );

        let groups = (w.div_ceil(8), h.div_ceil(8));
        if iters > 0 {
            dispatch(
                &mut encoder,
                &history_pipeline.demodulate,
                &ab,
                0,
                groups,
                "History Demodulate",
            );
            for it in 0..iters {
                // Iteration 0 reads A and writes B, so even iterations use the
                // A->B group and odd ones B->A.
                let group = if it % 2 == 0 { &ab } else { &ba };
                dispatch(
                    &mut encoder,
                    &history_pipeline.atrous,
                    group,
                    1 + it,
                    groups,
                    "History A-Trous",
                );
            }
        }
        dispatch(
            &mut encoder,
            &history_pipeline.resolve,
            &ab,
            0,
            groups,
            "History Resolve",
        );

        ctx.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// [`RayTracePipeline::denoise_and_resolve_resident`] with the learned
    /// filter in place of the à-trous chain.
    ///
    /// The two are interchangeable by construction. The network reads the
    /// same running mean, the same per-pixel statistics and the same guide
    /// planes a wavelet iteration reads, and writes the same
    /// `(illumination, variance)` scratch buffer; `resolve` then remodulates
    /// by the albedo, fades the filter out as the history grows and tonemaps,
    /// with no idea which of the two produced what it is reading. So a host
    /// switches denoisers by calling a different method, and nothing else
    /// about its frame changes.
    ///
    /// `denoise` is still honoured for everything that is not the filter
    /// itself — the exposure and the count cutoff — while its `iters`,
    /// `sigma_*` and variance settings are simply not consulted, because the
    /// network has no analogue of them. The cutoff is copied onto the
    /// denoiser, so one field governs both tiers.
    ///
    /// `neural` is resized here if the scene has been; see
    /// [`super::neural::NeuralDenoiser::ensure`].
    ///
    /// `target` must be a view of an `Rgba8Unorm` texture with
    /// `STORAGE_BINDING` usage, at least the resident scene's size.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_and_resolve_resident_neural(
        &self,
        ctx: &GpuContext,
        history_pipeline: &HistoryPipeline,
        neural_pipeline: &super::neural::NeuralPipeline,
        neural: &mut super::neural::NeuralDenoiser,
        res: &mut ResidentScene,
        denoise: &GpuDenoiseParams,
        target: &wgpu::TextureView,
    ) -> Result<(), GpuError> {
        let (w, h) = res.size();
        res.ensure_history(ctx, w, h);
        neural.ensure(ctx, w, h);
        neural.set_count_cutoff(denoise.count_cutoff.max(1));

        {
            let hist = res.history().expect("history was just ensured");
            // Slot 0 is the resolve's. `iters` is 1 rather than the caller's
            // count: to `resolve` it is not a wavelet iteration count, it is
            // the flag for "a filtered image is waiting in the scratch".
            // `src_is_b` is 0 because the network writes A.
            let base = HistoryParams {
                width: w,
                height: h,
                count_cutoff: denoise.count_cutoff.max(1),
                iters: 1,
                sigma_lum: denoise.sigma_lum,
                sigma_depth: denoise.sigma_depth,
                sigma_normal: denoise.sigma_normal,
                exposure: denoise.exposure,
                stride: 1,
                src_is_b: 0,
                scissor_xy: 0,
                scissor_wh: 0,
                cur_eye: [0.0; 4],
                cur_right: [0.0; 4],
                cur_up: [0.0; 4],
                cur_forward: [0.0; 4],
                prev_eye: [0.0; 4],
                prev_right: [0.0; 4],
                prev_up: [0.0; 4],
                prev_forward: [0.0; 4],
                view_params: [0.0; 4],
                reprojected: 0,
                iter_index: 0,
                origin_x: 0,
                origin_y: 0,
                history_cap: denoise.history_cap.max(1),
                clamp_k: denoise.clamp_k.max(0.0),
                clamp_reset: denoise.clamp_reset.max(1),
                motion_instances: 0,
                motion_ids: 0,
                spatial_variance: u32::from(denoise.spatial_variance),
                _pad0: 0,
                _pad1: 0,
            };
            ctx.queue
                .write_buffer(&hist.params, 0, bytemuck::bytes_of(&base));
        }

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Neural Denoise Encoder"),
            });

        let (raw, guides) = res.raw_and_guide_buffers();
        let hist = res.history().expect("history was just ensured");

        neural.record(
            ctx,
            neural_pipeline,
            &mut encoder,
            guides,
            &hist.mean,
            &hist.stats,
            &hist.scratch_a,
        );

        let ab = history_bind_group(
            ctx,
            history_pipeline,
            hist,
            raw,
            guides,
            &hist.scratch_a,
            &hist.scratch_b,
            Some(target),
            "History Bind Group A->B",
        );
        dispatch(
            &mut encoder,
            &history_pipeline.resolve,
            &ab,
            0,
            (w.div_ceil(8), h.div_ceil(8)),
            "History Resolve",
        );

        ctx.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Trace one pass, fold it into the device-side history, denoise the
    /// running mean, and tonemap the result into `target` — with no readback
    /// at all.
    ///
    /// This is the whole of a progressive viewport's per-frame work, in one
    /// call. It is exactly [`RayTracePipeline::accumulate_resident`] over one
    /// box followed by [`RayTracePipeline::denoise_and_resolve_resident`]; a
    /// caller with several dirty rectangles per frame wants those two
    /// directly, so the denoise chain runs once rather than once per
    /// rectangle.
    ///
    /// Call it once per pass with an increasing `state.frame_index`; the
    /// shader's jitter and RNG are driven by that index, so successive passes
    /// are independent samples of the same picture and the running mean
    /// converges.
    ///
    /// `keep` is one byte per pixel in row-major order, the same length as the
    /// frame: **1 keeps that pixel's history, 0 restarts it** at this pass's
    /// sample. It is how the caller expresses everything the device cannot
    /// know — a camera move (upload zeros), a reprojection that found a valid
    /// history for some pixels and not others, a region the caller wants to
    /// re-converge. Pass an all-1 mask for a still frame. An empty slice is
    /// read as all-1, which is the common case.
    ///
    /// A **scissor** set on `state` (see `GpuRenderState::set_scissor`) is
    /// honoured through the trace and the fold: the trace pass dispatches only
    /// over the rectangle, and the accumulate pass folds a sample in only for
    /// the pixels inside it. Every pixel outside keeps the mean, the sample
    /// count and the variance it already had — nothing stale is counted as
    /// fresh. The resolve pass still covers the frame, so the target texture
    /// stays whole; it simply re-resolves the untouched pixels from their
    /// unchanged history. That is what lets a viewer trace only the part of
    /// the frame that moved and keep the rest.
    ///
    /// `target` must be a view of an `Rgba8Unorm` texture with
    /// `STORAGE_BINDING` usage, at least the resident scene's size.
    ///
    /// `state` is forced into raw-sample mode and its refinement pass turned
    /// off, exactly as [`RayTracePipeline::render_resident_linear`] forces
    /// them: what the history folds in has to be one unweighted sample, not a
    /// step of the shader's own running average.
    ///
    /// The work is submitted and this returns immediately; the caller
    /// sequences it against whatever presents the texture.
    #[allow(clippy::too_many_arguments)]
    pub fn accumulate_and_denoise_resident(
        &self,
        ctx: &GpuContext,
        history_pipeline: &HistoryPipeline,
        res: &mut ResidentScene,
        camera: &GpuCamera,
        state: GpuRenderState,
        keep: &[u8],
        denoise: &GpuDenoiseParams,
        target: &wgpu::TextureView,
    ) -> Result<(), GpuError> {
        self.accumulate_and_denoise_resident_reprojected(
            ctx,
            history_pipeline,
            res,
            camera,
            state,
            keep,
            denoise,
            target,
            None,
        )
    }

    /// [`RayTracePipeline::accumulate_and_denoise_resident`], plus the
    /// device-side reprojection.
    ///
    /// Pass the camera the *previous* call to this method rendered from as
    /// `prev_view` and the history is carried across the move: each pixel is
    /// unprojected through this pass's depth, projected into the previous
    /// view, and takes the nearest previous pixel's mean and sample count
    /// where that pixel's depth agrees within 2% and its normal within a dot
    /// product of 0.9. Everything else — disocclusions, pixels that were off
    /// the previous film, background — restarts at this pass's sample.
    ///
    /// `None` skips the pass entirely and is exactly
    /// [`RayTracePipeline::accumulate_and_denoise_resident`]. Pass `None` on
    /// a still frame too: reprojecting a view onto itself is a no-op that
    /// still costs a dispatch, and passing the *same* camera is harmless but
    /// pointless.
    ///
    /// The keep mask still applies, and applies *after* the reprojection: a
    /// caller that reprojects on the device wants an empty (all-keep) mask
    /// and lets this pass decide what survives. Zeroing a pixel's mask entry
    /// still restarts it, whatever the reprojection found.
    ///
    /// Depth is [`crate::pathtrace::Film::depth`]'s convention throughout —
    /// distance from the eye along the primary ray, zero for background — and
    /// the reprojection reads the previous pass's copy of that plane, which
    /// this method keeps for itself. The first pass after a resize therefore
    /// restarts every pixel, since there is no previous plane to test yet.
    #[allow(clippy::too_many_arguments)]
    pub fn accumulate_and_denoise_resident_reprojected(
        &self,
        ctx: &GpuContext,
        history_pipeline: &HistoryPipeline,
        res: &mut ResidentScene,
        camera: &GpuCamera,
        state: GpuRenderState,
        keep: &[u8],
        denoise: &GpuDenoiseParams,
        target: &wgpu::TextureView,
        prev_view: Option<&GpuCamera>,
    ) -> Result<(), GpuError> {
        self.accumulate_and_denoise_resident_moving(
            ctx,
            history_pipeline,
            res,
            camera,
            state,
            keep,
            denoise,
            target,
            prev_view,
            None,
        )
    }

    /// [`RayTracePipeline::accumulate_and_denoise_resident_reprojected`], told
    /// what moved.
    ///
    /// The whole of a live frame in one call: trace, carry each pixel's
    /// history across both the camera's move and its own instance's move,
    /// clamp what has gone stale, filter what is short, tonemap. See
    /// [`InstanceMotion`] for what `motion` is and
    /// [`RayTracePipeline::accumulate_resident_temporal`] for the two halves.
    #[allow(clippy::too_many_arguments)]
    pub fn accumulate_and_denoise_resident_moving(
        &self,
        ctx: &GpuContext,
        history_pipeline: &HistoryPipeline,
        res: &mut ResidentScene,
        camera: &GpuCamera,
        state: GpuRenderState,
        keep: &[u8],
        denoise: &GpuDenoiseParams,
        target: &wgpu::TextureView,
        prev_view: Option<&GpuCamera>,
        motion: Option<&InstanceMotion>,
    ) -> Result<(), GpuError> {
        self.accumulate_resident_temporal(
            ctx,
            history_pipeline,
            res,
            camera,
            state,
            keep,
            prev_view,
            denoise,
            motion,
        )?;
        self.denoise_and_resolve_resident(ctx, history_pipeline, res, denoise, target)
    }

    /// Read the device-side history back.
    ///
    /// For tests, and for a host that wants to save the converged frame. The
    /// render path never needs it — that is the point of
    /// [`RayTracePipeline::accumulate_and_denoise_resident`].
    ///
    /// Returns `None` if no pass has built a history for this scene yet.
    pub async fn read_history(
        &self,
        ctx: &GpuContext,
        res: &mut ResidentScene,
    ) -> Result<Option<History>, GpuError> {
        let Some((w, h)) = res.history().map(|hi| (hi.width, hi.height)) else {
            return Ok(None);
        };
        let n = (w as u64) * (h as u64);
        let plane = n * 16;

        {
            let hist = res.history_mut().expect("just checked");
            if hist.readback.is_none() {
                hist.readback = Some(ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("History Readback"),
                    size: plane * 2,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }));
            }
        }

        let hist = res.history().expect("just checked");
        let staging = hist.readback.as_ref().expect("just allocated");
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("History Readback Encoder"),
            });
        encoder.copy_buffer_to_buffer(&hist.mean, 0, staging, 0, plane);
        encoder.copy_buffer_to_buffer(&hist.stats, 0, staging, plane, plane);
        ctx.queue.submit(Some(encoder.finish()));

        let raw = read_back_f32(ctx, staging).await?;
        let n = n as usize;
        let mut out = History {
            width: w,
            height: h,
            rgb: vec![0.0; n * 3],
            alpha: vec![0.0; n],
            count: vec![0; n],
            variance: vec![0.0; n],
        };
        for i in 0..n {
            let m = &raw[i * 4..i * 4 + 4];
            let s = &raw[(n + i) * 4..(n + i) * 4 + 4];
            out.rgb[i * 3..i * 3 + 3].copy_from_slice(&m[..3]);
            out.alpha[i] = m[3];
            out.count[i] = s[0] as u32;
            out.variance[i] = s[3];
        }
        Ok(Some(out))
    }
}
