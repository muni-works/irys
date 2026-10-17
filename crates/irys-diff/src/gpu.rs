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
const PATH_BYTES: u64 = 112;
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
    params: wgpu::Buffer,
    theta: wgpu::Buffer,
    dl: wgpu::Buffer,
    l: wgpu::Buffer,
    l2: wgpu::Buffer,
    grad: wgpu::Buffer,
    eval: Ir,
    adjoint: Ir,
    f: TensorId,
    pipes: [wgpu::ComputePipeline; 5],
    // gen, bounce, update, finish→l, finish→l2, accumulate
    groups: [wgpu::BindGroup; 6],
    /// Timings of the last forward and replay.
    pub timing: Timing,
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
        let cols: Vec<wgpu::Buffer> = (0..COLUMNS)
            .map(|_| storage("column", lane_bytes))
            .collect();
        let theta = storage("theta", PARAMS as u64 * 4);
        let df = storage("df", lane_bytes);
        let dl = storage("dl", lane_bytes);
        let l = storage("l", lane_bytes);
        let l2 = storage("l replayed", lane_bytes);
        let grad = storage("grad", PARAMS as u64 * 4);
        let path_buf = storage("paths", paths as u64 * PATH_BYTES);
        let params = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut g = Graph::new(lanes);
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
        let pipes = ["gen", "bounce", "update", "finish", "accumulate"].map(pipe);
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
        let mut bounce: Vec<(u32, &wgpu::Buffer)> = vec![(0, &params), (1, &path_buf)];
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
        ];
        let f = g.f;
        if let Some(e) = pollster::block_on(scope.pop()) {
            panic!("irys-diff gpu: {e}");
        }
        Self {
            ctx,
            paths,
            lanes,
            params,
            theta,
            dl,
            l,
            l2,
            grad,
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
            tan_half: (20.0f64).to_radians().tan() as f32,
            albedo: scene.floor_albedo as f32,
            alpha: scene.frozen_alpha.unwrap_or(theta[0]) as f32,
            eps: EPS,
        }
    }

    fn encode(&self, scene: &Scene, replay: bool) -> wgpu::CommandBuffer {
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

    /// `f` per lane from the last bounce, for debugging.
    pub fn last_f(&self) -> Vec<f32> {
        self.read(self.eval.buffer(self.f), self.lanes)
    }

    fn read(&self, src: &wgpu::Buffer, n: usize) -> Vec<f32> {
        let dev = &self.ctx.device;
        let bytes = n as u64 * 4;
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
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
/// f64 from the device's `L`.
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
