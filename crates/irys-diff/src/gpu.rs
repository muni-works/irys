//! Phase 2a: [`prb`](crate::prb) as a GPU wavefront, with bit-exact replay.
//!
//! Each bounce is five dispatches, in one compute pass per frame:
//!
//! 1. `bounce` (hand WGSL, `gpu/wavefront.wgsl`): intersect (hardware ray
//!    queries where the device has them, `gpu/rt.rs`), sample, and stage a
//!    conductor vertex's lane inputs; then lane compaction
//!    (`gpu/compact.wgsl`): a deterministic scan of which paths staged
//!    lanes, and `scatter`, which writes the graph's seven input columns at
//!    the compact lanes only;
//! 2. the **eval lowering of [`ggx::Graph`], as tang-compile emits it**
//!    ([`TensorGraph::to_wgsl`]) — `f` and `pdf` per lane, dispatched
//!    indirectly over the compact lanes alone;
//! 3. `update`: throughput `*= f cos / p`, and in replay `δf` per lane;
//! 4. replay only: the **adjoint lowering**, the same way — `Σ_lanes δf ∂f/∂θ`
//!    as the IR's own sum, scheduled `split = 256` (one workgroup per
//!    parameter, threads stride the lanes and tree-combine);
//! 5. replay only: `accumulate`, `grad += δθ`, one thread per parameter.
//!
//! The forward pass keeps `L` per lane in a device buffer; the replay reads
//! it there. The host sees `L` only to score the image.
//!
//! **Bit-exact replay.** Forward and replay dispatch the *same pipelines over
//! the same bind groups*: one `wgpu::ShaderModule` per kernel, built once, so
//! `f` (and every instruction that feeds a path) is the same compiled code on
//! the same inputs in both. The replay differs only in a uniform that gates
//! the `δf` store, which feeds nothing on the path. The adjoint program's own
//! recomputation of `f` is never used for throughput (design note 5).
//!
//! **The reduction.** No float atomics: WGSL has none, and a CAS loop on
//! `atomic<u32>` would make the gradient depend on scheduling. The IR's
//! reduction (workgroup tree per parameter) gives four numbers per bounce
//! and `accumulate` adds them in bounce order, so the gradient is
//! deterministic run to run.
//!
//! **The RNG** is [`prb::Rng`] bit for bit (SplitMix64 on `u32` pairs); a
//! uniform keeps the top 24 of the CPU's 53 bits.

use std::time::{Duration, Instant};

use irys_gpu::wgpu::{self, util::DeviceExt};
use irys_gpu::{GpuContext, GpuError};
use tang_compile::tensor::{Op, TensorGraph, TensorId, numel};

use crate::ggx::{COLUMNS, DF, Graph, PARAMS, THETA};
use crate::prb::{self, Loss, Scene};

const WAVEFRONT: &str = include_str!("gpu/wavefront.wgsl");
const EDGES: &str = include_str!("gpu/edges.wgsl");
const RESTIR: &str = include_str!("gpu/restir.wgsl");
const GUIDE: &str = include_str!("gpu/guide.wgsl");
const RT: &str = include_str!("gpu/rt.wgsl");
const COMPACT: &str = include_str!("gpu/compact.wgsl");

mod rt;

/// The wavefront's WGSL: with the hardware walk compiled in when the device
/// has ray queries (`rt.wgsl`, chosen per pass by `Params::rt`), else the
/// software walk alone.
fn wavefront_src(hw: bool, rest: &str) -> String {
    if hw {
        format!("enable wgpu_ray_query;\n{WAVEFRONT}\n{RT}\n{rest}")
    } else {
        format!(
            "{WAVEFRONT}\nfn intersect(o: vec3<f32>, d: vec3<f32>, tmax: f32, any: bool) -> Hit {{ return intersect_sw(o, d, tmax, any); }}\n{rest}"
        )
    }
}

/// `qmc::byte_tables` (dimensions 1–3, four byte tables each) and
/// `qmc::value_keys` as WGSL module constants, so the device's direction
/// numbers and scramble keys are the CPU's by construction.
fn sobol_const() -> String {
    let words: Vec<String> = crate::qmc::byte_tables().iter().map(|w| format!("{w}u")).collect();
    let keys: Vec<String> = crate::qmc::value_keys().iter().map(|w| format!("{w}u")).collect();
    format!(
        "const SOBOL_T = array<u32, {}>({});\nconst QKEY = array<u32, 4>({});\n",
        words.len(),
        words.join(", "),
        keys.join(", ")
    )
}
const PATH_BYTES: u64 = 128;
const WG: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    seed: [u32; 2],
    size: [u32; 2],
    spp: u32,
    nw: u32,
    paths: u32,
    replay: u32,
    eye: [f32; 4],
    fwd: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    tan_half: f32,
    albedo: f32,
    alpha: f32,
    eps: f32,
    resp: [f32; 4],
    pixels: u32,
    inv_n: f32,
    checker: f32,
    splat_root: u32,
    splat_base: u32,
    inst_base: u32,
    nee: u32,
    cap: u32,
    base: u32,
    total: u32,
    points: u32,
    rt: u32,
    sun: [f32; 4],
    env: [f32; 4],
    light: [f32; 4],
    restir: [u32; 4],
    guide: [f32; 4],
    gmin: [f32; 4],
    gres: [u32; 4],
    gres2: [u32; 4],
    qmc: [u32; 4],
    rr: [f32; 4],
}

/// The splat compositor's segment start, `EPS / 10` (phase 2b's, unchanged).
/// Opaque hits no longer use it: origins are pushed off by `offset_ray`,
/// scaled to the scene ([`prb::offset`]).
pub const EPS: f32 = 1e-4;

/// One tang-emitted program: a pipeline, bind group and grid per kernel.
struct Ir {
    kernels: Vec<(wgpu::ComputePipeline, wgpu::BindGroup, u32)>,
    buffers: Vec<(TensorId, wgpu::Buffer)>,
}

impl Ir {
    fn new(
        dev: &wgpu::Device,
        g: &TensorGraph,
        outputs: &[TensorId],
        inputs: &[&wgpu::Buffer],
        label: &str,
    ) -> Self {
        let prog = g.to_wgsl(outputs);
        let mut buffers: Vec<(TensorId, wgpu::Buffer)> = Vec::new();
        let mut own = Vec::new();
        for b in &prog.buffers {
            assert_eq!(
                b.format,
                tang_compile::tensor::Format::F32,
                "{label}: f32 only"
            );
            match b.input {
                Some(_) => {}
                None => own.push((
                    b.node,
                    dev.create_buffer(&wgpu::BufferDescriptor {
                        label: Some(label),
                        size: (b.len.max(1) * 4) as u64,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    }),
                )),
            }
        }
        buffers.extend(own);
        let input_of = |t: TensorId| {
            prog.buffers
                .iter()
                .find(|b| b.node == t)
                .and_then(|b| b.input)
        };
        let lookup = |t: TensorId| -> &wgpu::Buffer {
            let t = prog.storage(t);
            match input_of(t) {
                Some(k) => inputs[k],
                None => &buffers.iter().find(|(n, _)| *n == t).expect("buffer").1,
            }
        };
        let kernels = prog
            .kernels
            .iter()
            .map(|k| {
                let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(k.source.as_str().into()),
                });
                let pipe = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
                let uni = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::cast_slice(&[k.points, 0, 0, 0]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let mut entries: Vec<wgpu::BindGroupEntry> = k
                    .reads
                    .iter()
                    .enumerate()
                    .map(|(i, &t)| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: lookup(t).as_entire_binding(),
                    })
                    .collect();
                entries.push(wgpu::BindGroupEntry {
                    binding: k.reads.len() as u32,
                    resource: lookup(k.node).as_entire_binding(),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: k.reads.len() as u32 + 1,
                    resource: uni.as_entire_binding(),
                });
                let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(label),
                    layout: &pipe.get_bind_group_layout(0),
                    entries: &entries,
                });
                let groups = k.count.div_ceil(k.workgroup_size);
                assert!(groups <= 65535, "{label}: {groups} workgroups");
                (pipe, bg, groups)
            })
            .collect();
        Self { kernels, buffers }
    }

    fn buffer(&self, t: TensorId) -> &wgpu::Buffer {
        &self
            .buffers
            .iter()
            .find(|(n, _)| *n == t)
            .expect("output")
            .1
    }

    fn dispatch(&self, pass: &mut wgpu::ComputePass) {
        for (pipe, bg, groups) in &self.kernels {
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(*groups, 1, 1);
        }
    }

    /// Every kernel over the workgroups `args` holds (lane compaction):
    /// the kernels are pointwise over lanes, so a shorter grid evaluates
    /// the first lanes only.
    fn dispatch_indirect(&self, pass: &mut wgpu::ComputePass, args: &wgpu::Buffer) {
        for (pipe, bg, _) in &self.kernels {
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups_indirect(args, 0);
        }
    }
}

/// What runs on the device after a forward pass.
#[derive(Clone, Copy, PartialEq)]
enum Score {
    None,
    /// The image, the loss against the target, and `∂loss/∂image`.
    Loss,
    /// δL per lane from the last `∂loss/∂image`.
    LaneAdjoint,
    /// Both, for one seed.
    Both,
    /// The image only, added to the device's running sum.
    Accumulate,
}

/// What the last pass cost on the device (submit to readback).
#[derive(Debug, Clone, Copy, Default)]
pub struct Timing {
    /// Forward pass, including reading `L` back.
    pub forward: Duration,
    /// Replay pass, including reading `δθ` and the replayed `L` back.
    pub replay: Duration,
}

/// The scene's wavefront on the GPU, sized for one [`Scene`].
pub struct GpuPrb {
    ctx: &'static GpuContext,
    /// Paths per chunk: what the path buffer and the graph batch hold.
    paths: u32,
    chunks: u32,
    lanes: usize,
    width: usize,
    params: wgpu::Buffer,
    theta: wgpu::Buffer,
    dl: wgpu::Buffer,
    l: wgpu::Buffer,
    l2: wgpu::Buffer,
    grad: wgpu::Buffer,
    target: wgpu::Buffer,
    img: wgpu::Buffer,
    loss: wgpu::Buffer,
    nodes: wgpu::Buffer,
    ids: wgpu::Buffer,
    geo: wgpu::Buffer,
    points: wgpu::Buffer,
    /// Phase 3: emitter triangles (`pack_lights`).
    lights: wgpu::Buffer,
    /// ReSTIR DI: this frame's G-buffer, the previous one's, this frame's
    /// final reservoirs, the previous one's.
    restir: [wgpu::Buffer; 4],
    /// Path guiding: histograms, CDFs, trained flags, sample counts.
    guide: [wgpu::Buffer; 4],
    grid: Option<Grid>,
    /// The next forward passes record guiding samples.
    training: bool,
    eval: Ir,
    adjoint: Ir,
    f: TensorId,
    pipes: [wgpu::ComputePipeline; 18],
    // gen, bounce, update, finish→l, finish→l2, accumulate, score,
    // loss_sum, lane_adjoint
    groups: [wgpu::BindGroup; 19],
    /// The running sum of [`accumulate`](Self::accumulate)d frames.
    acc: wgpu::Buffer,
    /// Frames in `acc`.
    acc_frames: usize,
    /// Lane compaction's indirect dispatch: (workgroups, 1, 1, lanes).
    ind: wgpu::Buffer,
    /// Graph lanes per chunk.
    chunk_width: usize,
    layout: Layout,
    /// Timings of the last forward and replay.
    pub timing: Timing,
    /// The next pass's points are on the device already ([`GpuEdges`]).
    device_points: std::cell::Cell<bool>,
    /// Speed piece 1: the hardware acceleration structures, where the
    /// device has ray queries.
    accel: Option<rt::Accel>,
    /// Trace through `accel` (else the WGSL walk).
    hw: bool,
}

/// A bind group of `e`'s buffers, plus the acceleration structure and its
/// instance map (bindings 57, 58) when `accel` is given.
fn bind_with(
    dev: &wgpu::Device,
    p: &wgpu::ComputePipeline,
    e: &[(u32, &wgpu::Buffer)],
    accel: Option<&rt::Accel>,
) -> wgpu::BindGroup {
    let mut entries: Vec<wgpu::BindGroupEntry> = e
        .iter()
        .map(|&(binding, b)| wgpu::BindGroupEntry {
            binding,
            resource: b.as_entire_binding(),
        })
        .collect();
    if let Some(a) = accel {
        entries.push(wgpu::BindGroupEntry {
            binding: 57,
            resource: a.tlas.as_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 58,
            resource: a.map.as_entire_binding(),
        });
    }
    dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &p.get_bind_group_layout(0),
        entries: &entries,
    })
}

/// Where the instance and splat records start in the packed scene.
#[derive(Clone, Copy)]
struct Layout {
    splat_root: u32,
    splat_base: u32,
    inst_base: u32,
}

const NONE: u32 = u32::MAX;
const LEAF: u32 = 1 << 31;

/// A [`Scene`] packed for the wavefront: one node array (the top-level tree
/// rooted at 0, then every bottom-level tree, then the splat tree), one
/// leaf-index array, one array of `vec4` records (opaque prims, then
/// instances, then splats). Refuses what the WGSL cannot trace — an analytic
/// prim other than a sphere or disc, view-dependent splat colour, a tree
/// deeper than its traversal stack — rather than dropping it.
fn pack(scene: &Scene) -> (Vec<[f32; 4]>, Vec<u32>, Vec<[f32; 4]>, Layout) {
    use irys_core::Geometry;
    use irys_prims::analytic::Prim;
    let (mut nodes, mut ids, mut geo) = (Vec::new(), Vec::new(), Vec::new());
    let mut tree = |flat: &[irys_core::FlatBvhNode],
                    prims: &[u32],
                    nodes: &mut Vec<[f32; 4]>,
                    stack: usize| {
        let depth = irys_gpu::geometry::tree_depth(flat);
        assert!(
            depth + 4 <= stack,
            "tree {depth} deep: the WGSL stack holds {stack}"
        );
        let (base, first) = ((nodes.len() / 2) as u32, ids.len() as u32);
        for &(b, leaf, a, c) in flat {
            let (a, c) = if leaf {
                (a + first, c | LEAF)
            } else {
                (a + base, c + base)
            };
            let (lo, hi) = (b.min, b.max);
            nodes.push([lo.x as f32, lo.y as f32, lo.z as f32, f32::from_bits(a)]);
            nodes.push([hi.x as f32, hi.y as f32, hi.z as f32, f32::from_bits(c)]);
        }
        ids.extend_from_slice(prims);
        base
    };
    let (flat, inst) = scene.tlas.flatten();
    assert!(!flat.is_empty(), "a scene with no opaque geometry");
    tree(&flat, &inst, &mut nodes, 32);
    let v = |p: irys_core::Point3, w: f32| [p.x as f32, p.y as f32, p.z as f32, w];
    // (root node, first prim record) per bottom-level tree.
    let mut roots = Vec::new();
    for blas in &scene.blas {
        let (flat, prims) = blas.flatten();
        let root = tree(&flat, &prims, &mut nodes, 64);
        let prim_base = (geo.len() / 3) as u32;
        roots.push((root, prim_base));
        let o = blas.geometry();
        let mesh = o.mesh();
        let surf = |i: usize| match o.surface(i) {
            prb::Surface::Conductor => 0.0,
            prb::Surface::Diffuse => 1.0,
            prb::Surface::Emitter => 2.0,
        };
        for (i, t) in mesh.triangles().iter().enumerate() {
            let p = |k: usize| mesh.positions()[t[k] as usize];
            geo.extend([v(p(0), 0.0), v(p(1), surf(i)), v(p(2), 0.0)]);
        }
        for (k, prim) in o.analytic().prims().iter().enumerate() {
            let s = surf(mesh.len() + k);
            match *prim {
                Prim::Sphere { center, radius } => geo.extend([
                    v(center, 1.0),
                    [0.0, 0.0, 0.0, s],
                    [0.0, 0.0, 0.0, radius as f32],
                ]),
                Prim::Disc {
                    center,
                    normal,
                    radius,
                } => geo.extend([
                    v(center, 2.0),
                    [normal.x as f32, normal.y as f32, normal.z as f32, s],
                    [0.0, 0.0, 0.0, radius as f32],
                ]),
                other => panic!("irys-diff gpu: {other:?} has no WGSL intersector yet"),
            }
        }
    }
    let mut layout = Layout {
        splat_root: NONE,
        splat_base: 0,
        inst_base: geo.len() as u32,
    };
    // Instance records in `Tlas::instances` order, which is what the
    // top-level leaves index.
    for i in scene.tlas.instances() {
        let place = &scene.placements[i.payload()];
        let m = i
            .to_world()
            .inverse()
            .expect("placed, so invertible")
            .matrix;
        let row = |r: usize| {
            [
                m.get(r, 0) as f32,
                m.get(r, 1) as f32,
                m.get(r, 2) as f32,
                m.get(r, 3) as f32,
            ]
        };
        let (root, prim_base) = roots[place.blas];
        geo.extend([
            row(0),
            row(1),
            row(2),
            [
                f32::from_bits(root),
                f32::from_bits(prim_base),
                place.albedo as f32,
                if place.checker { 1.0 } else { 0.0 },
            ],
        ]);
    }
    layout.splat_base = geo.len() as u32;
    if let Some(cloud) = &scene.splats
        && !cloud.geometry().is_empty()
    {
        let (flat, prims) = cloud.flatten();
        layout.splat_root = tree(&flat, &prims, &mut nodes, 64);
        let sp = cloud.geometry();
        assert_eq!(
            sp.degree(),
            0,
            "irys-diff gpu: splat colour is degree 0 only"
        );
        let up = irys_core::Dir3::new_normalize(irys_core::Vec3::new(0.0, 0.0, 1.0));
        for i in 0..sp.len() {
            let a = sp.inv_cov(i);
            let c = sp.colour(i, up);
            geo.extend([
                v(sp.center(i), sp.opacity(i)),
                [a[0] as f32, a[1] as f32, a[2] as f32, a[3] as f32],
                [a[4] as f32, a[5] as f32, 0.0, 0.0],
                [c[0], c[1], c[2], 0.0],
            ]);
        }
    }
    (nodes, ids, geo, layout)
}

/// Path guiding's spatial grid: cubic cells over the scene's bounds, `res`
/// along the longest side, ×6 normal classes.
#[derive(Debug, Clone)]
struct Grid {
    /// Origin and 1 / cell size.
    min: [f32; 4],
    res: [u32; 3],
}

impl Grid {
    fn new(scene: &Scene, g: prb::Guide) -> Self {
        let b = scene.tlas().bounds().expect("a scene with geometry");
        let ext = b.max - b.min;
        let long = ext.x.max(ext.y).max(ext.z) * 1.0001;
        let cell = long / g.res.max(1) as f64;
        let n = |e: f64| ((e * 1.0001 / cell).ceil() as u32).max(1);
        Self {
            min: [b.min.x as f32, b.min.y as f32, b.min.z as f32, (1.0 / cell) as f32],
            res: [n(ext.x), n(ext.y), n(ext.z)],
        }
    }

    fn cells(&self) -> u32 {
        self.res[0] * self.res[1] * self.res[2] * 6
    }
}

/// The sampler's tables for the device: Sobol's four direction matrices,
/// then the wavelength CDF as `f32` bits ([`crate::qmc`]).
fn sampler_tables() -> Vec<u32> {
    let mut t: Vec<u32> = crate::qmc::sobol_matrices().iter().flatten().copied().collect();
    t.extend(crate::qmc::Spectral::get().cdf.iter().map(|c| c.to_bits()));
    t
}

/// The scene's emitter triangles for `sample_light`: three `vec4`s each,
/// `(v0, cumulative share)`, `(v1, 0)`, `(v2, 0)`. One harmless record when
/// there are none, so the binding is never empty.
fn pack_lights(scene: &Scene) -> Vec<[f32; 4]> {
    let l = scene.lights();
    let v = |p: irys_core::Point3, w: f64| [p.x as f32, p.y as f32, p.z as f32, w as f32];
    let mut out: Vec<[f32; 4]> = l
        .tris
        .iter()
        .zip(&l.cdf)
        .flat_map(|(t, &c)| [v(t[0], c), v(t[1], 0.0), v(t[2], 0.0)])
        .collect();
    if out.is_empty() {
        out = vec![[0.0; 4]; 3];
    }
    out
}

/// The device, or why there is none. Tests skip on `Err`.
pub fn context() -> Result<&'static GpuContext, GpuError> {
    GpuContext::init_blocking()
}

impl GpuPrb {
    /// Build every pipeline and buffer for `scene`'s lane count. A frame
    /// of any size is one call: the wavefront runs it in chunks of as many
    /// paths as one dispatch of the IR's one-dimensional grid holds.
    pub fn new(ctx: &'static GpuContext, scene: &Scene) -> Self {
        Self::chunked(ctx, scene, usize::MAX)
    }

    /// [`new`](Self::new) with chunks of at most `max_paths` paths, so tests
    /// can split a small frame.
    pub fn chunked(ctx: &'static GpuContext, scene: &Scene, max_paths: usize) -> Self {
        let dev = &ctx.device;
        let scope = dev.push_error_scope(wgpu::ErrorFilter::Validation);
        let lanes = scene.lanes();
        let per_path = scene.graph_width() / scene.paths().max(1);
        // `to_wgsl` grids are 1-D: 65535 workgroups of (at most) 64.
        let fit = 65535 * WG as usize / per_path.max(1);
        let paths = scene.paths().min(fit).min(max_paths).max(1) as u32;
        let chunks = scene.paths().div_ceil(paths as usize) as u32;
        assert!(
            scene.restir.is_none() || chunks == 1,
            "ReSTIR DI reuses across the frame: it must fit one chunk ({} paths > {paths})",
            scene.paths()
        );
        let storage = |label: &str, bytes: u64| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(16),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let lane_bytes = lanes as u64 * 4;
        let width = paths as usize * per_path;
        let graph_bytes = width as u64 * 4;
        let cols: Vec<wgpu::Buffer> = (0..COLUMNS)
            .map(|_| storage("column", graph_bytes))
            .collect();
        let theta = storage("theta", PARAMS as u64 * 4);
        let df = storage("df", graph_bytes);
        let dl = storage("dl", lane_bytes);
        let l = storage("l", lane_bytes);
        let l2 = storage("l replayed", lane_bytes);
        let grad = storage("grad", PARAMS as u64 * 4);
        let pixels = (scene.size.0 * scene.size.1) as u64;
        let target = storage("target", pixels * 12);
        let img = storage("image", pixels * 12);
        let d_img = storage("d image", pixels * 12);
        let acc = storage("image sum", pixels * 12);
        let sq = storage("squared error", pixels * 4);
        let loss = storage("loss", 4);
        let path_buf = storage("paths", paths as u64 * PATH_BYTES);
        let (nodes, ids, geo, layout) = pack(scene);
        let accel = rt::Accel::new(ctx, scene);
        // Room for the top-level tree to change shape when instances move
        // (`update_geometry`): at most two nodes (four vec4s) per instance.
        let room = 8 * (scene.placements().len() as u64 + 1) * 16;
        let init = |label: &str, bytes: &[u8], extra: u64| {
            let b = dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (bytes.len() as u64 + extra).max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            b.slice(..bytes.len() as u64)
                .get_mapped_range_mut()
                .expect("mapped")
                .copy_from_slice(bytes);
            b.unmap();
            b
        };
        let nodes = init("nodes", bytemuck::cast_slice(&nodes), room);
        let ids = init("prim ids", bytemuck::cast_slice(&ids), room / 4);
        let geo = init("prims", bytemuck::cast_slice(&geo), 0);
        let points = storage("points", scene.points().map_or(0, |p| p.len() as u64 * 8));
        let lights = init("lights", bytemuck::cast_slice(&pack_lights(scene)), 0);
        let qtab = init("sampler tables", bytemuck::cast_slice(&sampler_tables()), 0);
        // ReSTIR DI: G-buffers (three vec4 per slot) and reservoirs (two),
        // this frame's and the previous one's.
        let slots = if scene.restir.is_some() { paths as u64 } else { 0 };
        let g_cur = storage("restir g", slots * 48);
        let g_prev = storage("restir g prev", slots * 48);
        let r_a = storage("restir a", slots * 32);
        let r_b = storage("restir b", slots * 32);
        let r_prev = storage("restir prev", slots * 32);
        // Path guiding: the grid, its histograms and CDFs, the records.
        let grid = scene.guide.map(|gd| Grid::new(scene, gd));
        let cells = grid.as_ref().map_or(0, |g| g.cells() as u64);
        let recs = if grid.is_some() { paths as u64 * scene.depth as u64 } else { 0 };
        let g_hist = storage("guide hist", cells * 256 * 8);
        let g_cdf = storage("guide cdf", cells * 256 * 4);
        let g_ok = storage("guide ok", cells * 4);
        let g_rec = storage("guide records", recs * 16);
        let g_nrec = storage("guide record counts", if grid.is_some() { paths as u64 * 4 } else { 0 });
        let g_cnt = storage("guide counts", cells * 4);
        // Lane compaction (`compact.wgsl`).
        let lane_bits = storage("lane bits", paths as u64 * 4);
        let stage = storage("lane stage", paths as u64 * 32);
        let lane_off = storage("lane offsets", paths as u64 * 4);
        let lane_top = storage("lane block offsets", (paths as u64).div_ceil(256) * 4);
        let ind = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("eval indirect"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        // Lanes nothing has written are harmless (`f = 0`, finite
        // derivatives), as `bounce` used to write them every bounce.
        for c in &cols[..4] {
            ctx.queue.write_buffer(c, 0, bytemuck::cast_slice(&vec![1.0f32; width]));
        }
        let params = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        assert!(
            !(scene.nee && scene.splats().is_some()),
            "NEE with splats: shadow rays do not composite them"
        );
        let mut g = Graph::new(width);
        // The adjoint's reductions (δθ: four points, each over every lane)
        // split across a workgroup; the C and interpreter ignore this.
        let reductions: Vec<TensorId> =
            g.g.nodes()
                .iter()
                .enumerate()
                .filter(|(_, n)| {
                    matches!(&n.op, Op::Comp(c) if c.monoid.is_some()) && numel(&n.shape) <= PARAMS
                })
                .map(|(i, _)| TensorId::from_index(i))
                .collect();
        for t in reductions {
            g.g.set_schedule(t, 256, (0, 0, 0), (0, 0));
        }
        let mut inputs: Vec<&wgpu::Buffer> = cols.iter().collect();
        assert_eq!(inputs.len(), THETA);
        inputs.push(&theta);
        assert_eq!(inputs.len(), DF);
        inputs.push(&df);
        let eval = Ir::new(dev, &g.g, &[g.f, g.pdf], &inputs, "ggx eval");
        let adjoint = Ir::new(dev, &g.g, &[g.dtheta], &inputs, "ggx adjoint");

        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prb wavefront"),
            source: wgpu::ShaderSource::Wgsl(
                wavefront_src(accel.is_some(), &format!("{RESTIR}\n{GUIDE}\n{COMPACT}\n{}", sobol_const())).into(),
            ),
        });
        let pipe = |entry: &str| {
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipes = [
            "gen",
            "bounce",
            "update",
            "finish",
            "accumulate",
            "score",
            "loss_sum",
            "lane_adjoint",
            "tail",
            "ris_init",
            "ris_temporal",
            "ris_spatial",
            "guide_build",
            "cscan_block",
            "cscan_tops",
            "scatter",
            "clear",
            "accum",
        ]
        .map(pipe);
        let bind = |p: &wgpu::ComputePipeline, e: &[(u32, &wgpu::Buffer)]| {
            bind_with(dev, p, e, None)
        };
        // The kernels that trace: with the acceleration structures as well.
        let bind_rt = |p: &wgpu::ComputePipeline, e: &[(u32, &wgpu::Buffer)]| {
            bind_with(dev, p, e, accel.as_ref())
        };
        let f_buf = eval.buffer(g.f);
        let bounce: Vec<(u32, &wgpu::Buffer)> = vec![
            (0, &params),
            (1, &path_buf),
            (23, &nodes),
            (24, &ids),
            (25, &geo),
            (40, &lights),
            (43, &r_a),
            (51, &g_cdf),
            (52, &g_ok),
            (53, &g_rec),
            (54, &g_nrec),
            (59, &lane_bits),
            (60, &stage),
        ];
        let mut scatter: Vec<(u32, &wgpu::Buffer)> = vec![
            (0, &params),
            (1, &path_buf),
            (59, &lane_bits),
            (60, &stage),
            (61, &lane_off),
            (62, &lane_top),
        ];
        scatter.extend(cols.iter().enumerate().map(|(c, b)| (2 + c as u32, b)));
        let groups = [
            bind(
                &pipes[0],
                &[
                    (0, &params),
                    (1, &path_buf),
                    (26, &points),
                    (54, &g_nrec),
                    (56, &qtab),
                ],
            ),
            bind_rt(&pipes[1], &bounce),
            bind(
                &pipes[2],
                &[
                    (0, &params),
                    (1, &path_buf),
                    (9, f_buf),
                    (10, &l),
                    (11, &dl),
                    (12, &df),
                    (59, &lane_bits),
                    (61, &lane_off),
                    (62, &lane_top),
                ],
            ),
            bind(
                &pipes[3],
                &[
                    (0, &params),
                    (1, &path_buf),
                    (13, &l),
                    (50, &g_hist),
                    (53, &g_rec),
                    (54, &g_nrec),
                    (55, &g_cnt),
                ],
            ),
            bind(
                &pipes[3],
                &[
                    (0, &params),
                    (1, &path_buf),
                    (13, &l2),
                    (50, &g_hist),
                    (53, &g_rec),
                    (54, &g_nrec),
                    (55, &g_cnt),
                ],
            ),
            bind(&pipes[4], &[(14, &grad), (15, adjoint.buffer(g.dtheta))]),
            bind(
                &pipes[5],
                &[
                    (0, &params),
                    (56, &qtab),
                    (10, &l),
                    (17, &img),
                    (18, &target),
                    (19, &d_img),
                    (20, &sq),
                ],
            ),
            bind(&pipes[6], &[(0, &params), (20, &sq), (21, &loss)]),
            bind(
                &pipes[7],
                &[(0, &params), (19, &d_img), (22, &dl), (56, &qtab)],
            ),
            bind_rt(&pipes[8], &bounce),
            bind_rt(
                &pipes[9],
                &[
                    (0, &params),
                    (1, &path_buf),
                    (23, &nodes),
                    (24, &ids),
                    (25, &geo),
                    (40, &lights),
                    (41, &g_cur),
                    (43, &r_a),
                ],
            ),
            bind(
                &pipes[10],
                &[
                    (0, &params),
                    (41, &g_cur),
                    (42, &g_prev),
                    (43, &r_a),
                    (44, &r_b),
                    (45, &r_prev),
                ],
            ),
            bind(
                &pipes[11],
                &[(0, &params), (41, &g_cur), (43, &r_a), (44, &r_b)],
            ),
            bind(
                &pipes[12],
                &[(0, &params), (50, &g_hist), (51, &g_cdf), (52, &g_ok), (55, &g_cnt)],
            ),
            bind(&pipes[13], &[(0, &params), (59, &lane_bits), (61, &lane_off), (62, &lane_top)]),
            bind(&pipes[14], &[(0, &params), (62, &lane_top), (63, &ind)]),
            bind(&pipes[15], &scatter),
            bind(&pipes[16], &[(12, &df), (63, &ind)]),
            bind(&pipes[17], &[(0, &params), (17, &img), (64, &acc)]),
        ];
        let f = g.f;
        if let Some(e) = pollster::block_on(scope.pop()) {
            panic!("irys-diff gpu: {e}");
        }
        Self {
            ctx,
            paths,
            chunks,
            lanes,
            width: scene.graph_width(),
            params,
            theta,
            dl,
            l,
            l2,
            grad,
            target,
            img,
            loss,
            nodes,
            ids,
            geo,
            points,
            lights,
            restir: [g_cur, g_prev, r_a, r_prev],
            guide: [g_hist, g_cdf, g_ok, g_cnt],
            grid,
            training: false,
            layout,
            eval,
            adjoint,
            f,
            pipes,
            groups,
            ind,
            chunk_width: width,
            acc,
            acc_frames: 0,
            timing: Timing::default(),
            device_points: std::cell::Cell::new(false),
            accel,
            hw: std::env::var("IRYS_RT").map_or(true, |v| v != "sw"),
        }
    }

    /// Whether passes trace through the hardware's ray queries (the device
    /// has them and [`set_hardware_rt`](Self::set_hardware_rt) left them on).
    pub fn hardware_rt(&self) -> bool {
        self.accel.is_some() && self.hw
    }

    /// Trace through the hardware (`true`, where the device has ray
    /// queries) or the WGSL walk, the reference. Takes effect next pass.
    pub fn set_hardware_rt(&mut self, on: bool) {
        self.hw = on;
    }

    /// Lanes this wavefront was built for.
    pub fn lanes(&self) -> usize {
        self.lanes
    }

    fn params(
        &self,
        scene: &Scene,
        theta: &[f64; PARAMS],
        seed: u64,
        replay: bool,
        chunk: u32,
    ) -> Params {
        let base = chunk * self.paths;
        let v = |v: irys_core::Vec3| [v.x as f32, v.y as f32, v.z as f32, 0.0];
        Params {
            seed: [seed as u32, (seed >> 32) as u32],
            size: [scene.size.0 as u32, scene.size.1 as u32],
            spp: scene.spp as u32,
            nw: scene.wavelengths as u32,
            paths: self.paths.min(scene.paths() as u32 - base),
            replay: replay as u32,
            eye: v(scene.eye - irys_core::Point3::origin()),
            fwd: v(scene.forward),
            right: v(scene.right),
            up: v(scene.up),
            tan_half: scene.tan_half as f32,
            albedo: scene.floor_albedo as f32,
            alpha: scene.frozen_alpha.unwrap_or(theta[0]) as f32,
            eps: EPS,
            resp: {
                let k = (irys_light::spectrum::LAMBDA_MAX_NM - irys_light::spectrum::LAMBDA_MIN_NM)
                    / (scene.spp * scene.wavelengths) as f64;
                let r = irys_light::spectrum::RESPONSE_NORM;
                [(r[0] * k) as f32, (r[1] * k) as f32, (r[2] * k) as f32, 0.0]
            },
            pixels: (scene.size.0 * scene.size.1) as u32,
            inv_n: 1.0 / (scene.size.0 * scene.size.1 * 3) as f32,
            checker: scene.floor_checker.unwrap_or(0.0) as f32,
            splat_root: self.layout.splat_root,
            splat_base: self.layout.splat_base,
            inst_base: self.layout.inst_base,
            nee: scene.nee as u32,
            cap: self.paths,
            base,
            total: scene.paths() as u32,
            points: scene.points().is_some() as u32,
            rt: (self.accel.is_some() && self.hw) as u32,
            sun: {
                let e = &scene.env;
                [
                    e.sun.x as f32,
                    e.sun.y as f32,
                    e.sun.z as f32,
                    e.kappa as f32,
                ]
            },
            env: {
                let e = &scene.env;
                let k = e.kappa;
                let vmf = k / (2.0 * std::f64::consts::PI * (1.0 - (-2.0 * k).exp()));
                [
                    e.sun_radiance as f32,
                    e.sky as f32,
                    prb::SUN_PICK as f32,
                    vmf as f32,
                ]
            },
            light: [
                scene.lights().tris.len() as f32,
                scene.env_pick() as f32,
                scene.lights().area as f32,
                scene.emission as f32,
            ],
            guide: [scene.guide.map_or(0.0, |g| g.mix as f32), 0.0, 0.0, 0.0],
            gmin: self.grid.as_ref().map_or([0.0; 4], |g| g.min),
            gres: {
                let r = self.grid.as_ref().map_or([0; 3], |g| g.res);
                let flags = self.grid.is_some() as u32
                    | ((self.grid.is_some() && self.training && !replay) as u32) << 1;
                [r[0], r[1], r[2], flags]
            },
            gres2: [
                scene.depth as u32,
                self.grid.as_ref().map_or(0, |g| g.cells()),
                scene.guide.map_or(0, |g| g.min_samples),
                0,
            ],
            qmc: [scene.qmc as u32, scene.spectral_is as u32, 0, 0],
            rr: match scene.roulette {
                Some(r) => [1.0, r.start as f32, r.floor as f32, 0.0],
                None => [0.0; 4],
            },
            restir: match scene.restir {
                Some(r) => [
                    (r.neighbours.min(9) << 8) | r.radius.min(255),
                    r.candidates.max(1),
                    r.temporal as u32,
                    r.cap,
                ],
                None => [0; 4],
            },
        }
    }

    /// One chunk's paths, start to finish: `gen`, then per bounce `bounce`
    /// → eval IR → `update` [→ adjoint IR → `accumulate`], then `finish`.
    fn encode_trace(&self, scene: &Scene, replay: bool) -> wgpu::CommandBuffer {
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            let groups = self.paths.div_ceil(WG);
            self.dispatch(&mut pass, 0, 0, groups);
            if scene.restir.is_some() {
                self.dispatch(&mut pass, 9, 10, groups);
                self.dispatch(&mut pass, 10, 11, groups);
                self.dispatch(&mut pass, 11, 12, groups);
            }
            for _ in 0..scene.depth {
                self.dispatch(&mut pass, 1, 1, groups);
                // Lane compaction: scan the lane bits, scatter the columns,
                // evaluate only the compact lanes.
                self.dispatch(&mut pass, 13, 14, self.paths.div_ceil(256));
                self.dispatch(&mut pass, 14, 15, 1);
                self.dispatch(&mut pass, 15, 16, groups);
                if replay {
                    self.dispatch(&mut pass, 16, 17, (self.chunk_width as u32).div_ceil(WG));
                }
                self.eval.dispatch_indirect(&mut pass, &self.ind);
                self.dispatch(&mut pass, 2, 2, groups);
                if replay {
                    self.adjoint.dispatch(&mut pass);
                    self.dispatch(&mut pass, 4, 5, 1);
                }
            }
            if scene.complete {
                self.dispatch(&mut pass, 8, 9, groups);
            }
            self.dispatch(&mut pass, 3, if replay { 4 } else { 3 }, groups);
        }
        enc.finish()
    }

    /// Scoring over the whole frame, after every chunk is traced.
    fn encode_score(&self, scene: &Scene, score: Score) -> wgpu::CommandBuffer {
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            let px = (scene.size.0 * scene.size.1) as u32;
            let all = (scene.paths() as u32).div_ceil(WG);
            if matches!(score, Score::Loss | Score::Both) {
                self.dispatch(&mut pass, 5, 6, px.div_ceil(WG));
                self.dispatch(&mut pass, 6, 7, 1);
            }
            if matches!(score, Score::LaneAdjoint | Score::Both) {
                self.dispatch(&mut pass, 7, 8, all);
            }
            if score == Score::Accumulate {
                self.dispatch(&mut pass, 5, 6, px.div_ceil(WG));
                self.dispatch(&mut pass, 17, 18, px.div_ceil(WG));
            }
        }
        enc.finish()
    }

    /// Trace the frame chunk by chunk, each its own submit after its own
    /// uniforms (queued writes land at the start of the next submit), then
    /// score it.
    fn run(&self, scene: &Scene, theta: &[f64; PARAMS], seed: u64, replay: bool, score: Score) {
        let q = &self.ctx.queue;
        q.write_buffer(
            &self.theta,
            0,
            bytemuck::cast_slice(&theta.map(|x| x as f32)),
        );
        if let Some(p) = scene.points()
            && !self.device_points.replace(false)
        {
            let p32: Vec<[f32; 2]> = p.iter().map(|q| [q[0] as f32, q[1] as f32]).collect();
            assert!(
                p32.len() as u64 * 8 <= self.points.size(),
                "points frame size"
            );
            q.write_buffer(&self.points, 0, bytemuck::cast_slice(&p32));
        }
        for c in 0..self.chunks {
            q.write_buffer(
                &self.params,
                0,
                bytemuck::bytes_of(&self.params(scene, theta, seed, replay, c)),
            );
            q.submit([self.encode_trace(scene, replay)]);
        }
        if score != Score::None {
            q.submit([self.encode_score(scene, score)]);
        }
    }

    /// `groups` workgroups as a grid WGSL's 65535-per-dimension limit takes;
    /// kernels flatten it back (`flat` in the WGSL).
    fn dispatch(&self, pass: &mut wgpu::ComputePass, p: usize, b: usize, groups: u32) {
        pass.set_pipeline(&self.pipes[p]);
        pass.set_bind_group(0, &self.groups[b], &[]);
        let x = groups.min(65535);
        pass.dispatch_workgroups(x, groups.div_ceil(x.max(1)), 1);
    }

    /// Re-upload the scene's geometry after [`Scene::place`] moved
    /// instances: the top-level tree and the instance records. The trees
    /// must be the ones this wavefront was built with.
    pub fn update_geometry(&mut self, scene: &Scene) {
        let (nodes, ids, geo, layout) = pack(scene);
        let fits = |b: &wgpu::Buffer, n: usize| (n as u64) <= b.size();
        assert!(
            fits(&self.nodes, nodes.len() * 16)
                && fits(&self.ids, ids.len() * 4)
                && fits(&self.geo, geo.len() * 16),
            "update_geometry: not the scene this wavefront was built for"
        );
        let q = &self.ctx.queue;
        q.write_buffer(&self.nodes, 0, bytemuck::cast_slice(&nodes));
        q.write_buffer(&self.ids, 0, bytemuck::cast_slice(&ids));
        q.write_buffer(&self.geo, 0, bytemuck::cast_slice(&geo));
        q.write_buffer(&self.lights, 0, bytemuck::cast_slice(&pack_lights(scene)));
        self.layout = layout;
        if let Some(a) = &mut self.accel {
            a.update(self.ctx, scene);
        }
    }

    /// ReSTIR DI: make the last frame's reservoirs and G-buffer the previous
    /// frame's, for the next frame's temporal reuse. Explicit, so a replay
    /// at the same seed sees exactly what its forward pass saw.
    pub fn advance(&mut self) {
        let [g_cur, g_prev, r_a, r_prev] = &self.restir;
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(g_cur, 0, g_prev, 0, g_cur.size());
        enc.copy_buffer_to_buffer(r_a, 0, r_prev, 0, r_a.size());
        self.ctx.queue.submit([enc.finish()]);
    }

    /// Path guiding: whether the next forward passes record training
    /// samples into the histograms (replays never do).
    pub fn set_guide_training(&mut self, on: bool) {
        assert!(self.grid.is_some(), "no guide: Scene::guide is None");
        self.training = on;
    }

    /// Path guiding: rebuild every cell's CDF from its histogram. Until the
    /// next build, the guide is a constant.
    pub fn guide_build(&mut self, scene: &Scene) {
        let cells = self.grid.as_ref().expect("no guide").cells();
        let q = &self.ctx.queue;
        q.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&self.params(scene, &[0.0; PARAMS], 0, true, 0)),
        );
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipes[12]);
            pass.set_bind_group(0, &self.groups[13], &[]);
            let x = cells.min(65535);
            pass.dispatch_workgroups(x, cells.div_ceil(x), 1);
        }
        q.submit([enc.finish()]);
    }

    /// Forget all history: ReSTIR's previous frame and everything the
    /// guide learned. A new render starts here.
    pub fn reset_history(&mut self) {
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for b in self.restir.iter().chain(&self.guide) {
            enc.clear_buffer(b, 0, None);
        }
        self.ctx.queue.submit([enc.finish()]);
    }

    /// Path guiding: forget everything learned.
    pub fn guide_reset(&mut self) {
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for b in &self.guide {
            enc.clear_buffer(b, 0, None);
        }
        self.ctx.queue.submit([enc.finish()]);
    }

    /// Chunks a frame is traced in.
    pub fn chunks(&self) -> u32 {
        self.chunks
    }

    fn check(&self, scene: &Scene) {
        assert_eq!(
            scene.lanes(),
            self.lanes,
            "scene does not match the wavefront"
        );
        assert_eq!(
            scene.graph_width(),
            self.width,
            "NEE on one and off the other"
        );
        assert!(scene.wavelengths <= prb::HERO);
    }

    /// The forward pass: `L` per lane, kept on the device for the replay and
    /// returned to score the image.
    pub fn forward(&mut self, scene: &Scene, theta: &[f64; PARAMS], seed: u64) -> Vec<f32> {
        self.check(scene);
        let t = Instant::now();
        self.run(scene, theta, seed, false, Score::None);
        let l = self.read(&self.l, self.lanes);
        self.timing.forward = t.elapsed();
        l
    }

    /// The replay against the last [`forward`](Self::forward) at the same
    /// seed: `(δθ, replayed L)`. `dl` is the loss adjoint per lane.
    pub fn replay(
        &mut self,
        scene: &Scene,
        theta: &[f64; PARAMS],
        seed: u64,
        dl: &[f64],
    ) -> ([f64; PARAMS], Vec<f32>) {
        self.check(scene);
        assert_eq!(dl.len(), self.lanes);
        let t = Instant::now();
        let q = &self.ctx.queue;
        let dl32: Vec<f32> = dl.iter().map(|&x| x as f32).collect();
        q.write_buffer(&self.dl, 0, bytemuck::cast_slice(&dl32));
        q.write_buffer(&self.grad, 0, bytemuck::cast_slice(&[0.0f32; PARAMS]));
        self.run(scene, theta, seed, true, Score::None);
        let g = self.read(&self.grad, PARAMS);
        let l2 = self.read(&self.l2, self.lanes);
        self.timing.replay = t.elapsed();
        (std::array::from_fn(|p| g[p] as f64), l2)
    }

    /// The image [`gradient_l2`] and [`loss_l2`] score against, linear sRGB
    /// `w·h·3` as [`prb::image`] lays it out. Until set, it is black.
    pub fn set_target(&mut self, target: &[f64]) {
        assert_eq!(target.len() as u64 * 4, self.target.size(), "target size");
        let t: Vec<f32> = target.iter().map(|&x| x as f32).collect();
        self.ctx
            .queue
            .write_buffer(&self.target, 0, bytemuck::cast_slice(&t));
    }

    /// Piece 1's gradient, scored on the device: forward the primal frame,
    /// splat its image, take the L2 loss and `∂loss/∂image`; forward the
    /// adjoint frame, turn that into δL per lane; replay. The host writes
    /// the uniforms and reads back five floats, `loss` and `δθ`.
    pub fn gradient_l2(
        &mut self,
        scene: &Scene,
        theta: &[f64; PARAMS],
        seeds: (u64, u64),
    ) -> (f64, [f64; PARAMS]) {
        self.check(scene);
        let (primal, adjoint) = seeds;
        let t = Instant::now();
        if adjoint == primal {
            self.run(scene, theta, primal, false, Score::Both);
        } else {
            self.run(scene, theta, primal, false, Score::Loss);
            self.run(scene, theta, adjoint, false, Score::LaneAdjoint);
        }
        self.timing.forward = t.elapsed();
        let t = Instant::now();
        self.ctx
            .queue
            .write_buffer(&self.grad, 0, bytemuck::cast_slice(&[0.0f32; PARAMS]));
        self.run(scene, theta, adjoint, true, Score::None);
        let out = self.read_many(&[(&self.loss, 1), (&self.grad, PARAMS)]);
        self.timing.replay = t.elapsed();
        (out[0] as f64, std::array::from_fn(|p| out[1 + p] as f64))
    }

    /// The L2 loss against the target, scored on the device. Reads back one
    /// float.
    pub fn loss_l2(&mut self, scene: &Scene, theta: &[f64; PARAMS], seed: u64) -> f64 {
        self.check(scene);
        self.run(scene, theta, seed, false, Score::Loss);
        self.read(&self.loss, 1)[0] as f64
    }

    /// Render one frame and splat it on the device: linear sRGB `w·h·3`.
    pub fn image(&mut self, scene: &Scene, theta: &[f64; PARAMS], seed: u64) -> Vec<f32> {
        self.check(scene);
        self.run(scene, theta, seed, false, Score::Loss);
        self.last_image()
    }

    /// Render one frame and add its image to the device's running sum,
    /// reading nothing back: a render of many frames costs one readback
    /// ([`accumulated`](Self::accumulated)) instead of one per frame. The
    /// frame's image is [`image`](Self::image)'s, bit for bit.
    pub fn accumulate(&mut self, scene: &Scene, theta: &[f64; PARAMS], seed: u64) {
        self.check(scene);
        self.run(scene, theta, seed, false, Score::Accumulate);
        self.acc_frames += 1;
    }

    /// The mean of the frames [`accumulate`](Self::accumulate)d since the
    /// last [`clear_accumulated`](Self::clear_accumulated) (summed on the
    /// device in f32, in frame order, so the same frames give the same
    /// bits), and the number of frames.
    pub fn accumulated(&self) -> (Vec<f32>, usize) {
        let sum = self.read(&self.acc, (self.acc.size() / 4) as usize);
        let n = self.acc_frames.max(1) as f32;
        (sum.iter().map(|v| v / n).collect(), self.acc_frames)
    }

    /// Start a new running sum.
    pub fn clear_accumulated(&mut self) {
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.clear_buffer(&self.acc, 0, None);
        self.ctx.queue.submit([enc.finish()]);
        self.acc_frames = 0;
    }

    /// The image the last scoring pass splatted.
    pub fn last_image(&self) -> Vec<f32> {
        self.read(&self.img, (self.img.size() / 4) as usize)
    }

    /// δL per lane as the last [`gradient_l2`](Self::gradient_l2) computed it.
    pub fn last_lane_adjoint(&self) -> Vec<f32> {
        self.read(&self.dl, self.lanes)
    }

    /// `f` per lane from the last bounce, for debugging.
    pub fn last_f(&self) -> Vec<f32> {
        self.read(self.eval.buffer(self.f), self.lanes)
    }

    fn read(&self, src: &wgpu::Buffer, n: usize) -> Vec<f32> {
        self.read_many(&[(src, n)])
    }

    /// Several buffers' heads into one staging buffer, one map.
    fn read_many(&self, srcs: &[(&wgpu::Buffer, usize)]) -> Vec<f32> {
        let dev = &self.ctx.device;
        let bytes = srcs.iter().map(|(_, n)| *n as u64 * 4).sum::<u64>();
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut at = 0;
        for (src, n) in srcs {
            enc.copy_buffer_to_buffer(src, 0, &staging, at, *n as u64 * 4);
            at += *n as u64 * 4;
        }
        self.ctx.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        loop {
            dev.poll(wgpu::PollType::wait_indefinitely())
                .expect("device lost");
            match rx.try_recv() {
                Ok(r) => break r.expect("map failed"),
                Err(std::sync::mpsc::TryRecvError::Empty) => std::thread::yield_now(),
                Err(e) => panic!("map: {e}"),
            }
        }
        let out =
            bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range().expect("mapped")).to_vec();
        staging.unmap();
        out
    }
}

/// [`prb::gradient`] on the GPU: `(loss, ∂loss/∂θ)`, scored on the host in
/// f64 from the device's `L` — phase 2a's path, kept as the reference the
/// device scoring ([`GpuPrb::gradient_l2`]) is tested against, and for losses
/// other than L2.
pub fn gradient(
    gpu: &mut GpuPrb,
    scene: &Scene,
    theta: &[f64; PARAMS],
    seeds: (u64, u64),
    loss: &impl Loss,
) -> (f64, [f64; PARAMS]) {
    let (primal, adjoint) = seeds;
    let l = widen(&gpu.forward(scene, theta, primal));
    let (value, d_img) = loss.eval(&prb::image(scene, primal, &l));
    if adjoint != primal {
        gpu.forward(scene, theta, adjoint);
    }
    let dl = prb::lane_adjoint(scene, adjoint, &d_img);
    (value, gpu.replay(scene, theta, adjoint, &dl).0)
}

/// The loss alone, for central differences on the device.
pub fn loss(
    gpu: &mut GpuPrb,
    scene: &Scene,
    theta: &[f64; PARAMS],
    seed: u64,
    loss: &impl Loss,
) -> f64 {
    let l = widen(&gpu.forward(scene, theta, seed));
    loss.eval(&prb::image(scene, seed, &l)).0
}

/// f32 lanes as f64, for the host-side image and loss.
pub fn widen(l: &[f32]) -> Vec<f64> {
    l.iter().map(|&x| x as f64).collect()
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EdgeParams {
    seed: [u32; 2],
    n: u32,
    slots: u32,
    moving: u32,
    np: u32,
    policy: u32,
    blocks: u32,
    eps_px: f32,
    _p: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Mover {
    rec: u32,
    prim_base: u32,
    tris: u32,
    slot_base: u32,
    edge_base: u32,
    _p: [u32; 3],
}

const SCAN_BLOCK: u32 = 1024;

/// What the last [`GpuEdges::sample`] cost.
#[derive(Debug, Clone, Copy, Default)]
pub struct EdgeTiming {
    /// Host: the rig's placements and uniforms.
    pub host: Duration,
    /// Device: candidates, scan, samples, and reading the weights back.
    pub device: Duration,
}

/// Phase 2e: [`vis::EdgeSamples`] drawn on the device (`edges.wgsl`), for a
/// [`GpuPrb`] built for `2 n` points, writing them straight into its point
/// buffer. The host keeps the rig (placements at `φ ± RIG_H`, a few
/// matrices) and the final weighted sum; projection, importance, the prefix
/// sum, sampling and the ownership rays are the device's, deterministic
/// (no atomics; the scan runs in a fixed order).
pub struct GpuEdges {
    pipes: [wgpu::ComputePipeline; 5],
    groups: [wgpu::BindGroup; 5],
    ep: wgpu::Buffer,
    params: wgpu::Buffer,
    xf: wgpu::Buffer,
    pts: wgpu::Buffer,
    smp: wgpu::Buffer,
    slots: u32,
    blocks: u32,
    n: u32,
    np: u32,
    policy: u32,
    moving: Vec<usize>,
    /// What the last draw cost.
    pub timing: EdgeTiming,
}

impl GpuEdges {
    /// Buffers and pipelines for `rig`'s moving instances of `scene`, `n`
    /// samples, `np` parameters; `front_only` is phase 2e's silhouette
    /// importance (2d's otherwise).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gp: &GpuPrb,
        scene: &Scene,
        edges: &crate::vis::Edges,
        moving: &[usize],
        n: usize,
        np: usize,
        front_only: bool,
    ) -> Self {
        use irys_core::Geometry;
        let dev = &gp.ctx.device;
        let scope = dev.push_error_scope(wgpu::ErrorFilter::Validation);
        assert!(np <= 8, "at most 8 parameters");
        assert!(
            gp.points.size() >= 2 * n as u64 * 8,
            "the wavefront holds 2n points"
        );
        let mut base = vec![0u32; scene.blas().len()];
        let mut acc = 0u32;
        for (b, t) in scene.blas().iter().enumerate() {
            base[b] = acc;
            acc += t.geometry().len() as u32;
        }
        let (mut movers, mut nbr, mut dih) = (Vec::new(), Vec::<u32>::new(), Vec::<f32>::new());
        let mut slot = 0u32;
        for &k in moving {
            let b = scene.placements()[k].blas;
            let rec = scene
                .tlas()
                .instances()
                .iter()
                .position(|i| i.payload() == k)
                .expect("placed") as u32;
            let tris = edges.nbr[b].len() as u32;
            movers.push(Mover {
                rec,
                prim_base: base[b],
                tris,
                slot_base: slot,
                edge_base: nbr.len() as u32,
                _p: [0; 3],
            });
            nbr.extend(edges.nbr[b].iter().flatten());
            dih.extend(edges.cos[b].iter().flatten());
            slot += 3 * tris;
        }
        let slots = slot.max(1);
        let blocks = slots.div_ceil(SCAN_BLOCK);
        let init = |label: &str, bytes: &[u8], usage: wgpu::BufferUsages| {
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: if bytes.is_empty() { &[0; 16] } else { bytes },
                usage,
            })
        };
        let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let rw = |label: &str, bytes: u64| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(16),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let movers_b = init("movers", bytemuck::cast_slice(&movers), st);
        let nbr_b = init("nbr", bytemuck::cast_slice(&nbr), st);
        let dih_b = init("dihedral", bytemuck::cast_slice(&dih), st);
        let xf = rw("xf", (moving.len() * 3 * (1 + np) * 16) as u64);
        let wt = rw("edge weights", slots as u64 * 4);
        let rt = rw("edge ranges", slots as u64 * 16);
        let btot = rw("block totals", (blocks as u64 + 1) * 4);
        let pts = rw("edge points", 2 * n as u64 * 8);
        let smp = rw("edge samples", n as u64 * (2 + np as u64) * 4);
        let uni = |label: &str, bytes: u64| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let ep = uni("edge params", std::mem::size_of::<EdgeParams>() as u64);
        let params = uni("edge scene params", std::mem::size_of::<Params>() as u64);
        // The edge sampler's visibility stays on the WGSL walk: one eye ray
        // per edge point, not a frame's worth of bounces.
        let src = wavefront_src(false, &format!("{RESTIR}\n{GUIDE}\n{EDGES}\n{}", sobol_const()));
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("edges"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let pipe = |entry: &str| {
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipes = [
            "edge_cand",
            "scan_block",
            "scan_tops",
            "scan_fill",
            "edge_sample",
        ]
        .map(pipe);
        let bind = |p: &wgpu::ComputePipeline, e: &[(u32, &wgpu::Buffer)]| {
            dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &p.get_bind_group_layout(0),
                entries: &e
                    .iter()
                    .map(|&(binding, b)| wgpu::BindGroupEntry {
                        binding,
                        resource: b.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            })
        };
        let groups = [
            bind(
                &pipes[0],
                &[
                    (0, &params),
                    (25, &gp.geo),
                    (27, &ep),
                    (28, &movers_b),
                    (29, &xf),
                    (30, &nbr_b),
                    (31, &dih_b),
                    (32, &wt),
                    (33, &rt),
                ],
            ),
            bind(&pipes[1], &[(27, &ep), (32, &wt), (34, &btot)]),
            bind(&pipes[2], &[(27, &ep), (34, &btot)]),
            bind(&pipes[3], &[(27, &ep), (32, &wt), (34, &btot)]),
            bind(
                &pipes[4],
                &[
                    (0, &params),
                    (23, &gp.nodes),
                    (24, &gp.ids),
                    (25, &gp.geo),
                    (27, &ep),
                    (28, &movers_b),
                    (29, &xf),
                    (32, &wt),
                    (33, &rt),
                    (34, &btot),
                    (35, &pts),
                    (36, &smp),
                ],
            ),
        ];
        if let Some(e) = pollster::block_on(scope.pop()) {
            panic!("irys-diff gpu edges: {e}");
        }
        Self {
            pipes,
            groups,
            ep,
            params,
            xf,
            pts,
            smp,
            slots,
            blocks,
            n: n as u32,
            np: np as u32,
            policy: front_only as u32,
            moving: moving.to_vec(),
            timing: EdgeTiming::default(),
        }
    }

    /// Draw the edge samples of `scene` posed at `phi` (and `gp` updated to
    /// it) with generator `seed`, into `gp`'s point buffer: the next
    /// [`GpuPrb::forward`] traces them (the scene must be in point mode with
    /// `2 n` points; their values are ignored). Returns the weights.
    pub fn sample(
        &mut self,
        gp: &GpuPrb,
        scene: &Scene,
        rig: &crate::vis::Rig,
        phi: &[f64],
        seed: u64,
    ) -> crate::vis::EdgeSamples {
        use crate::vis::RIG_H;
        assert_eq!(
            rig.moving, self.moving,
            "the rig this sampler was built for"
        );
        assert_eq!(phi.len(), self.np as usize);
        let t0 = Instant::now();
        let np = phi.len();
        let shifted: Vec<[Vec<irys_core::Transform>; 2]> = (0..np)
            .map(|j| {
                let mut p = phi.to_vec();
                p[j] += RIG_H;
                let a = (rig.place)(&p);
                p[j] -= 2.0 * RIG_H;
                [a, (rig.place)(&p)]
            })
            .collect();
        let rows = |m: &irys_core::Transform| -> [[f64; 4]; 3] {
            std::array::from_fn(|r| std::array::from_fn(|c| m.matrix.get(r, c)))
        };
        let mut xf: Vec<[f32; 4]> = Vec::new();
        for (mi, &k) in self.moving.iter().enumerate() {
            for r in rows(&scene.placements()[k].to_world) {
                xf.push(r.map(|x| x as f32));
            }
            for [a, b] in &shifted {
                let (ra, rb) = (rows(&a[mi]), rows(&b[mi]));
                for r in 0..3 {
                    xf.push(std::array::from_fn(|c| {
                        ((ra[r][c] - rb[r][c]) / (2.0 * RIG_H)) as f32
                    }));
                }
            }
        }
        let q = &gp.ctx.queue;
        q.write_buffer(&self.xf, 0, bytemuck::cast_slice(&xf));
        q.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&gp.params(scene, &[0.0; PARAMS], seed, false, 0)),
        );
        q.write_buffer(
            &self.ep,
            0,
            bytemuck::bytes_of(&EdgeParams {
                seed: [seed as u32, (seed >> 32) as u32],
                n: self.n,
                slots: self.slots,
                moving: self.moving.len() as u32,
                np: self.np,
                policy: self.policy,
                blocks: self.blocks,
                eps_px: crate::vis::EPS_PX as f32,
                _p: [0; 3],
            }),
        );
        let host = t0.elapsed();
        let t1 = Instant::now();
        let dev = &gp.ctx.device;
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            let go = |pass: &mut wgpu::ComputePass, p: usize, groups: u32| {
                pass.set_pipeline(&self.pipes[p]);
                pass.set_bind_group(0, &self.groups[p], &[]);
                let x = groups.clamp(1, 65535);
                pass.dispatch_workgroups(x, groups.div_ceil(x), 1);
            };
            go(&mut pass, 0, self.slots.div_ceil(WG));
            go(&mut pass, 1, self.blocks.div_ceil(WG));
            go(&mut pass, 2, 1);
            go(&mut pass, 3, self.blocks.div_ceil(WG));
            go(&mut pass, 4, self.n.div_ceil(WG));
        }
        enc.copy_buffer_to_buffer(&self.pts, 0, &gp.points, 0, 2 * self.n as u64 * 8);
        q.submit([enc.finish()]);
        gp.device_points.set(true);
        let stride = 2 + np;
        let out = gp.read(&self.smp, self.n as usize * stride);
        let (mut pixel, mut weight, mut visible) = (Vec::new(), Vec::new(), 0);
        for s in out.chunks_exact(stride) {
            pixel.push(s[0] as usize);
            visible += (s[1] > 0.0) as usize;
            weight.push(s[2..].iter().map(|&x| x as f64).collect());
        }
        self.timing = EdgeTiming {
            host,
            device: t1.elapsed(),
        };
        crate::vis::EdgeSamples::from_parts(pixel, weight, visible)
    }

    /// The device's points, as camera-plane pairs (for tests).
    pub fn points(&self, gp: &GpuPrb) -> Vec<[f64; 2]> {
        gp.read(&self.pts, 2 * self.n as usize * 2)
            .chunks_exact(2)
            .map(|p| [p[0] as f64, p[1] as f64])
            .collect()
    }
}
