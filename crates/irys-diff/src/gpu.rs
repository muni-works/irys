//! Phase 2a: [`prb`](crate::prb) as a GPU wavefront, with bit-exact replay.
//!
//! Each bounce is five dispatches, in one compute pass per frame:
//!
//! 1. `bounce` (hand WGSL, `gpu/wavefront.wgsl`): intersect, sample, and
//!    write the graph's seven input columns per lane;
//! 2. the **eval lowering of [`ggx::Graph`], as tang-compile emits it**
//!    ([`TensorGraph::to_wgsl`]) — `f` and `pdf` per lane;
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
    _pad: [u32; 4],
    sun: [f32; 4],
    env: [f32; 4],
}

/// Offset off a surface for the next ray. f32 cannot hold the CPU's 1e-6 at
/// unit scale; hits need `t > eps / 10`.
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
    paths: u32,
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
    eval: Ir,
    adjoint: Ir,
    f: TensorId,
    pipes: [wgpu::ComputePipeline; 8],
    // gen, bounce, update, finish→l, finish→l2, accumulate, score,
    // loss_sum, lane_adjoint
    groups: [wgpu::BindGroup; 9],
    layout: Layout,
    /// Timings of the last forward and replay.
    pub timing: Timing,
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
                0.0,
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

/// The device, or why there is none. Tests skip on `Err`.
pub fn context() -> Result<&'static GpuContext, GpuError> {
    GpuContext::init_blocking()
}

impl GpuPrb {
    /// Build every pipeline and buffer for `scene`'s lane count.
    pub fn new(ctx: &'static GpuContext, scene: &Scene) -> Self {
        let dev = &ctx.device;
        let scope = dev.push_error_scope(wgpu::ErrorFilter::Validation);
        let lanes = scene.lanes();
        let paths = scene.paths() as u32;
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
        let graph_bytes = scene.graph_width() as u64 * 4;
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
        let sq = storage("squared error", pixels * 4);
        let loss = storage("loss", 4);
        let path_buf = storage("paths", paths as u64 * PATH_BYTES);
        let (nodes, ids, geo, layout) = pack(scene);
        let init = |label: &str, bytes: &[u8]| {
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        let nodes = init("nodes", bytemuck::cast_slice(&nodes));
        let ids = init("prim ids", bytemuck::cast_slice(&ids));
        let geo = init("prims", bytemuck::cast_slice(&geo));
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
        let width = scene.graph_width();
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
            source: wgpu::ShaderSource::Wgsl(WAVEFRONT.into()),
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
        let f_buf = eval.buffer(g.f);
        let mut bounce: Vec<(u32, &wgpu::Buffer)> = vec![
            (0, &params),
            (1, &path_buf),
            (23, &nodes),
            (24, &ids),
            (25, &geo),
        ];
        bounce.extend(cols.iter().enumerate().map(|(c, b)| (2 + c as u32, b)));
        let groups = [
            bind(&pipes[0], &[(0, &params), (1, &path_buf)]),
            bind(&pipes[1], &bounce),
            bind(
                &pipes[2],
                &[
                    (0, &params),
                    (1, &path_buf),
                    (9, f_buf),
                    (10, &l),
                    (11, &dl),
                    (12, &df),
                ],
            ),
            bind(&pipes[3], &[(0, &params), (1, &path_buf), (13, &l)]),
            bind(&pipes[3], &[(0, &params), (1, &path_buf), (13, &l2)]),
            bind(&pipes[4], &[(14, &grad), (15, adjoint.buffer(g.dtheta))]),
            bind(
                &pipes[5],
                &[
                    (0, &params),
                    (1, &path_buf),
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
                &[(0, &params), (1, &path_buf), (19, &d_img), (22, &dl)],
            ),
        ];
        let f = g.f;
        if let Some(e) = pollster::block_on(scope.pop()) {
            panic!("irys-diff gpu: {e}");
        }
        Self {
            ctx,
            paths,
            lanes,
            width,
            params,
            theta,
            dl,
            l,
            l2,
            grad,
            target,
            img,
            loss,
            layout,
            eval,
            adjoint,
            f,
            pipes,
            groups,
            timing: Timing::default(),
        }
    }

    /// Lanes this wavefront was built for.
    pub fn lanes(&self) -> usize {
        self.lanes
    }

    fn params(&self, scene: &Scene, theta: &[f64; PARAMS], seed: u64, replay: bool) -> Params {
        let v = |v: irys_core::Vec3| [v.x as f32, v.y as f32, v.z as f32, 0.0];
        Params {
            seed: [seed as u32, (seed >> 32) as u32],
            size: [scene.size.0 as u32, scene.size.1 as u32],
            spp: scene.spp as u32,
            nw: scene.wavelengths as u32,
            paths: self.paths,
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
            _pad: [0; 4],
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
        }
    }

    fn encode(&self, scene: &Scene, replay: bool) -> wgpu::CommandBuffer {
        self.encode_with(scene, replay, Score::None)
    }

    fn encode_with(&self, scene: &Scene, replay: bool, score: Score) -> wgpu::CommandBuffer {
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            let groups = self.paths.div_ceil(WG);
            self.dispatch(&mut pass, 0, 0, groups);
            for _ in 0..scene.depth {
                self.dispatch(&mut pass, 1, 1, groups);
                self.eval.dispatch(&mut pass);
                self.dispatch(&mut pass, 2, 2, groups);
                if replay {
                    self.adjoint.dispatch(&mut pass);
                    self.dispatch(&mut pass, 4, 5, 1);
                }
            }
            self.dispatch(&mut pass, 3, if replay { 4 } else { 3 }, groups);
            match score {
                Score::None => {}
                Score::Loss => {
                    let px = (scene.size.0 * scene.size.1) as u32;
                    self.dispatch(&mut pass, 5, 6, px.div_ceil(WG));
                    self.dispatch(&mut pass, 6, 7, 1);
                }
                Score::LaneAdjoint => self.dispatch(&mut pass, 7, 8, groups),
                Score::Both => {
                    let px = (scene.size.0 * scene.size.1) as u32;
                    self.dispatch(&mut pass, 5, 6, px.div_ceil(WG));
                    self.dispatch(&mut pass, 6, 7, 1);
                    self.dispatch(&mut pass, 7, 8, groups);
                }
            }
        }
        enc.finish()
    }

    fn dispatch(&self, pass: &mut wgpu::ComputePass, p: usize, b: usize, groups: u32) {
        pass.set_pipeline(&self.pipes[p]);
        pass.set_bind_group(0, &self.groups[b], &[]);
        pass.dispatch_workgroups(groups, 1, 1);
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
        let q = &self.ctx.queue;
        q.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&self.params(scene, theta, seed, false)),
        );
        q.write_buffer(
            &self.theta,
            0,
            bytemuck::cast_slice(&theta.map(|x| x as f32)),
        );
        q.submit([self.encode(scene, false)]);
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
        q.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&self.params(scene, theta, seed, true)),
        );
        q.write_buffer(
            &self.theta,
            0,
            bytemuck::cast_slice(&theta.map(|x| x as f32)),
        );
        let dl32: Vec<f32> = dl.iter().map(|&x| x as f32).collect();
        q.write_buffer(&self.dl, 0, bytemuck::cast_slice(&dl32));
        q.write_buffer(&self.grad, 0, bytemuck::cast_slice(&[0.0f32; PARAMS]));
        q.submit([self.encode(scene, true)]);
        let g = self.read(&self.grad, PARAMS);
        let l2 = self.read(&self.l2, self.lanes);
        self.timing.replay = t.elapsed();
        (std::array::from_fn(|p| g[p] as f64), l2)
    }

    fn upload(&self, scene: &Scene, theta: &[f64; PARAMS], seed: u64, replay: bool) {
        let q = &self.ctx.queue;
        q.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&self.params(scene, theta, seed, replay)),
        );
        q.write_buffer(
            &self.theta,
            0,
            bytemuck::cast_slice(&theta.map(|x| x as f32)),
        );
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
        let ctx = self.ctx;
        let q = &ctx.queue;
        self.upload(scene, theta, primal, false);
        if adjoint == primal {
            q.submit([self.encode_with(scene, false, Score::Both)]);
        } else {
            q.submit([self.encode_with(scene, false, Score::Loss)]);
            self.upload(scene, theta, adjoint, false);
            q.submit([self.encode_with(scene, false, Score::LaneAdjoint)]);
        }
        self.timing.forward = t.elapsed();
        let t = Instant::now();
        self.upload(scene, theta, adjoint, true);
        q.write_buffer(&self.grad, 0, bytemuck::cast_slice(&[0.0f32; PARAMS]));
        q.submit([self.encode(scene, true)]);
        let out = self.read_many(&[(&self.loss, 1), (&self.grad, PARAMS)]);
        self.timing.replay = t.elapsed();
        (out[0] as f64, std::array::from_fn(|p| out[1 + p] as f64))
    }

    /// The L2 loss against the target, scored on the device. Reads back one
    /// float.
    pub fn loss_l2(&mut self, scene: &Scene, theta: &[f64; PARAMS], seed: u64) -> f64 {
        self.check(scene);
        self.upload(scene, theta, seed, false);
        self.ctx
            .queue
            .submit([self.encode_with(scene, false, Score::Loss)]);
        self.read(&self.loss, 1)[0] as f64
    }

    /// Render one frame and splat it on the device: linear sRGB `w·h·3`.
    pub fn image(&mut self, scene: &Scene, theta: &[f64; PARAMS], seed: u64) -> Vec<f32> {
        self.check(scene);
        self.upload(scene, theta, seed, false);
        self.ctx
            .queue
            .submit([self.encode_with(scene, false, Score::Loss)]);
        self.last_image()
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
