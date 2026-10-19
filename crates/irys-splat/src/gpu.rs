//! The trainer on irys's wgpu 30 device: `raster.wgsl` (tile forward,
//! adjoint, per-splat reduction) and `proj.wgsl` (projection, adjoint +
//! Adam, loss + gradient) in one module and one bind group.
//!
//! Per step the host only bins: it reads back one `(u, v, radius, z)` per
//! splat, builds the tile lists (`tiles::bin`) and uploads them; the rest —
//! composite, loss, its gradient, the adjoint, the parameter update — is one
//! submit with no readback. Parameters and Adam moments stay resident;
//! densification reads them back every `refine_every` steps.

use bytemuck::{Pod, Zeroable};
use irys_diff::look::pinhole::Pinhole;
use irys_gpu::GpuContext;
use irys_gpu::wgpu;

use crate::model::Gaussians;
use crate::project::{Grad2, Grad3, NEAR, Proj};
use crate::tiles::Bins;

const SHADER: [&str; 2] = [include_str!("raster.wgsl"), include_str!("proj.wgsl")];

/// f32 per splat in the device's parameter layout.
pub const NP: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Uniform {
    size: [u32; 4],
    bg: [f32; 4],
    rot: [[f32; 4]; 3],
    intr: [f32; 4],
    lim: [f32; 4],
    misc: [f32; 4],
    lr0: [f32; 4],
    lr1: [f32; 4],
    win: [f32; 12],
}

struct Buf {
    buf: wgpu::Buffer,
    cap: u64,
}

const NB: usize = 19; // bindings 1..=19

/// The device's parameter packing of `g` (and of its Adam moments).
pub fn pack_params(g: &Gaussians) -> Vec<f32> {
    let mut v = vec![0f32; NP * g.len()];
    for i in 0..g.len() {
        let o = &mut v[NP * i..NP * (i + 1)];
        for a in 0..3 {
            o[a] = g.means[i][a] as f32;
            o[3 + a] = g.log_scales[i][a] as f32;
        }
        for a in 0..4 {
            o[6 + a] = g.quats[i][a] as f32;
        }
        o[10] = g.opacity[i] as f32;
        for k in 0..16 {
            for ch in 0..3 {
                o[12 + 3 * k + ch] = g.sh[i][k][ch] as f32;
            }
        }
    }
    v
}

pub fn unpack_params(v: &[f32]) -> Gaussians {
    let n = v.len() / NP;
    let mut g = Gaussians::zeros(n);
    for i in 0..n {
        let o = &v[NP * i..NP * (i + 1)];
        for a in 0..3 {
            g.means[i][a] = o[a] as f64;
            g.log_scales[i][a] = o[3 + a] as f64;
        }
        for a in 0..4 {
            g.quats[i][a] = o[6 + a] as f64;
        }
        g.opacity[i] = o[10] as f64;
        for k in 0..16 {
            for ch in 0..3 {
                g.sh[i][k][ch] = o[12 + 3 * k + ch] as f64;
            }
        }
    }
    g
}

/// The raw gradient of the device's layout as a [`Grad3`].
pub fn grad3_of(o: &[f32]) -> Grad3 {
    let mut g = Grad3::default();
    for a in 0..3 {
        g.mean[a] = o[a] as f64;
        g.log_scale[a] = o[3 + a] as f64;
    }
    for a in 0..4 {
        g.quat[a] = o[6 + a] as f64;
    }
    g.opacity = o[10] as f64;
    for k in 0..16 {
        for ch in 0..3 {
            g.sh[k][ch] = o[12 + 3 * k + ch] as f64;
        }
    }
    g
}

/// The device's packing of host-projected splats: 3 × vec4.
pub fn pack(p: &[Option<Proj>]) -> Vec<[f32; 4]> {
    let mut v = Vec::with_capacity(3 * p.len());
    for q in p {
        let q = q.unwrap_or_default();
        let f = |x: f64| x as f32;
        v.push([f(q.uv[0]), f(q.uv[1]), f(q.conic[0]), f(q.conic[1])]);
        v.push([f(q.conic[2]), f(q.opacity), f(q.colour[0]), f(q.colour[1])]);
        v.push([f(q.colour[2]), f(q.colour[3]), 0.0, 0.0]);
    }
    v
}

/// Per-step learning rates and Adam bias corrections.
#[derive(Debug, Clone, Copy, Default)]
pub struct Lr {
    pub means: f64,
    pub scales: f64,
    pub quats: f64,
    pub opacity: f64,
    pub sh0: f64,
    pub shn: f64,
    pub bc1: f64,
    pub bc2: f64,
}

/// Loss settings for a step.
#[derive(Debug, Clone, Copy)]
pub struct LossCfg {
    pub ssim: f64,
    /// depth weight / valid-pixel count (0: off)
    pub depth_per_px: f64,
}

fn storage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
}

pub struct Device {
    ctx: &'static GpuContext,
    pipes: Vec<(String, wgpu::ComputePipeline)>,
    layout: wgpu::BindGroupLayout,
    uni: wgpu::Buffer,
    bufs: Vec<Buf>,
    group: Option<wgpu::BindGroup>,
    u: Uniform,
    tiles: (usize, usize),
    splats: usize,
    /// Wall ms of the last submits: [project + key readback, step].
    pub last_ms: [f64; 2],
}

/// The old name: the raster half of [`Device`].
pub type Raster = Device;

impl Device {
    pub fn new(ctx: &'static GpuContext) -> Self {
        let dev = &ctx.device;
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("irys-splat"),
            source: wgpu::ShaderSource::Wgsl(SHADER.join("\n").into()),
        });
        let layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("irys-splat"),
            entries: &(0..=NB as u32)
                .map(|b| wgpu::BindGroupLayoutEntry {
                    binding: b,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if b == 0 { wgpu::BufferBindingType::Uniform } else { wgpu::BufferBindingType::Storage { read_only: false } },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("irys-splat"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipes = ["forward", "backward", "reduce", "project", "adjoint", "ssim_h1", "ssim_v1", "ssim_g", "ssim_h2", "ssim_v2"]
            .iter()
            .map(|e| {
                let p = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(e),
                    layout: Some(&pl),
                    module: &module,
                    entry_point: Some(e),
                    compilation_options: Default::default(),
                    cache: None,
                });
                (e.to_string(), p)
            })
            .collect();
        let uni = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("irys-splat uniform"),
            size: std::mem::size_of::<Uniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bufs = (1..=NB)
            .map(|b| Buf {
                buf: dev.create_buffer(&wgpu::BufferDescriptor { label: Some(&format!("irys-splat {b}")), size: 256, usage: storage(), mapped_at_creation: false }),
                cap: 256,
            })
            .collect();
        let mut u = Uniform::default();
        let k: Vec<f64> = (0..11).map(|i| (-((i as f64 - 5.0).powi(2)) / 4.5).exp()).collect();
        let s: f64 = k.iter().sum();
        for i in 0..11 {
            u.win[i] = (k[i] / s) as f32;
        }
        Self { ctx, pipes, layout, uni, bufs, group: None, u, tiles: (0, 0), splats: 0, last_ms: [0.0; 2] }
    }

    fn pipe(&self, name: &str) -> &wgpu::ComputePipeline {
        &self.pipes.iter().find(|p| p.0 == name).unwrap().1
    }

    fn ensure(&mut self, binding: usize, bytes: u64) {
        let b = &mut self.bufs[binding - 1];
        if b.cap >= bytes {
            return;
        }
        let cap = (bytes + bytes / 2).max(256).next_multiple_of(256);
        let nb = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("irys-splat {binding}")),
            size: cap,
            usage: storage(),
            mapped_at_creation: false,
        });
        b.buf = nb;
        b.cap = cap;
        self.group = None;
    }

    fn write<T: Pod>(&mut self, binding: usize, data: &[T]) {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        self.ensure(binding, bytes.len() as u64);
        if !bytes.is_empty() {
            self.ctx.queue.write_buffer(&self.bufs[binding - 1].buf, 0, bytes);
        }
    }

    fn group(&mut self) -> wgpu::BindGroup {
        if let Some(g) = &self.group {
            return g.clone();
        }
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: self.uni.as_entire_binding() }];
        for (k, b) in self.bufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry { binding: k as u32 + 1, resource: b.buf.as_entire_binding() });
        }
        let g = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("irys-splat"), layout: &self.layout, entries: &entries });
        self.group = Some(g.clone());
        g
    }

    fn read<T: Pod>(&self, binding: usize, count: usize) -> Vec<T> {
        let bytes = (count * std::mem::size_of::<T>()) as u64;
        if bytes == 0 {
            return Vec::new();
        }
        let dev = &self.ctx.device;
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("irys-splat readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&self.bufs[binding - 1].buf, 0, &staging, 0, bytes);
        self.ctx.queue.submit([enc.finish()]);
        let sl = staging.slice(..);
        sl.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        dev.poll(wgpu::PollType::wait_indefinitely()).expect("device lost");
        let v = bytemuck::cast_slice(&sl.get_mapped_range().expect("mapped")).to_vec();
        staging.unmap();
        v
    }

    fn set_camera(&mut self, cam: &Pinhole, degree: u32) {
        let (wd, ht) = (cam.width as f64, cam.height as f64);
        let tanx = wd / (2.0 * cam.fx);
        let tany = ht / (2.0 * cam.fy);
        self.u.size[0] = cam.width as u32;
        self.u.size[1] = cam.height as u32;
        for r in 0..3 {
            self.u.rot[r] = [cam.rot[r][0] as f32, cam.rot[r][1] as f32, cam.rot[r][2] as f32, cam.pos[r] as f32];
        }
        self.u.intr = [cam.fx as f32, cam.fy as f32, cam.cx as f32, cam.cy as f32];
        self.u.lim = [
            (-(cam.cx / cam.fx + 0.3 * tanx)) as f32,
            ((wd - cam.cx) / cam.fx + 0.3 * tanx) as f32,
            (-(cam.cy / cam.fy + 0.3 * tany)) as f32,
            ((ht - cam.cy) / cam.fy + 0.3 * tany) as f32,
        ];
        self.u.misc[0] = NEAR as f32;
        self.u.misc[1] = degree as f32;
        self.u.misc[3] = (1.0 / (3.0 * wd * ht)) as f32;
    }

    fn flush_uniform(&self) {
        self.ctx.queue.write_buffer(&self.uni, 0, bytemuck::bytes_of(&self.u));
    }

    fn dispatch_splats(&self, p: &mut wgpu::ComputePass, name: &str) {
        let groups = self.splats.div_ceil(256).max(1);
        let (gx, gy) = if groups > 65535 { (65535, groups.div_ceil(65535)) } else { (groups, 1) };
        p.set_pipeline(self.pipe(name));
        p.dispatch_workgroups(gx as u32, gy as u32, 1);
    }

    // ---- resident training state -------------------------------------

    /// Upload parameters and moments (`m`, `v`: same layout; `None` zeros).
    pub fn set_state(&mut self, g: &Gaussians, m: Option<&Gaussians>, v: Option<&Gaussians>) {
        let n = g.len();
        self.splats = n;
        self.u.size[3] = n as u32;
        self.write(11, &pack_params(g));
        let z = vec![0f32; NP * n];
        match m {
            Some(m) => self.write(12, &pack_params(m)),
            None => self.write(12, &z),
        }
        match v {
            Some(v) => self.write(13, &pack_params(v)),
            None => self.write(13, &z),
        }
        self.write(15, &vec![[0f32; 2]; n]);
        self.ensure(14, (n.max(1) * 16) as u64);
        self.ensure(19, (n.max(1) * NP * 4) as u64);
        self.ensure(1, (n.max(1) * 48) as u64);
        self.ensure(10, (n.max(1) * 48) as u64);
    }

    /// Parameters, moments and densification stats `(Σ grad, views)`.
    pub fn state(&self) -> (Gaussians, Gaussians, Gaussians, Vec<[f32; 2]>) {
        let n = self.splats;
        (
            unpack_params(&self.read::<f32>(11, NP * n)),
            unpack_params(&self.read::<f32>(12, NP * n)),
            unpack_params(&self.read::<f32>(13, NP * n)),
            self.read::<[f32; 2]>(15, n),
        )
    }

    /// The raw gradient of the last step, per splat (`gdbg`).
    pub fn last_grads(&self) -> Vec<Grad3> {
        let v = self.read::<f32>(19, NP * self.splats);
        v.chunks_exact(NP).map(grad3_of).collect()
    }

    /// Project the resident cloud at `cam`: `(u, v, radius, z)` per splat
    /// (radius 0: culled); the device keeps the packed splats.
    pub fn project(&mut self, cam: &Pinhole, degree: u32) -> Vec<[f32; 4]> {
        let t0 = std::time::Instant::now();
        self.set_camera(cam, degree);
        self.flush_uniform();
        let g = self.group();
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("project"), timestamp_writes: None });
            p.set_bind_group(0, &g, &[]);
            self.dispatch_splats(&mut p, "project");
        }
        self.ctx.queue.submit([enc.finish()]);
        let k = self.read::<[f32; 4]>(14, self.splats);
        self.last_ms[0] = t0.elapsed().as_secs_f64() * 1e3;
        k
    }

    fn upload_bins(&mut self, bins: &Bins, size: (usize, usize)) {
        let (w, h) = size;
        self.tiles = (bins.tiles_x, bins.tiles_y);
        self.u.size[2] = bins.tiles_x as u32;
        self.write(2, &bins.ranges);
        self.write(3, &bins.ids);
        self.ensure(4, (w * h * 16) as u64);
        self.ensure(5, (w * h * 8) as u64);
        self.ensure(6, (w * h * 16) as u64);
        self.ensure(7, (bins.ids.len().max(1) * 48) as u64);
        self.write(8, &bins.csr_off);
        self.write(9, &bins.csr_slots);
        self.ensure(16, (w * h * 16) as u64);
        self.ensure(17, (w * h * 60) as u64);
        self.ensure(18, (w * h * 60) as u64);
    }

    /// One training step after [`Self::project`] at the same camera: the
    /// composite, the loss gradient against `gt` (sRGB + depth), the adjoint
    /// and Adam, in one submit. Returns nothing; nothing is read back
    /// unless `want_image` (then the rendered frame).
    pub fn step(&mut self, bins: &Bins, gt: &[[f32; 4]], bg: [f32; 3], loss: LossCfg, lr: Lr, want_image: bool) -> Option<Vec<[f32; 4]>> {
        let t0 = std::time::Instant::now();
        let (w, h) = (self.u.size[0] as usize, self.u.size[1] as usize);
        self.upload_bins(bins, (w, h));
        self.write(16, gt);
        self.u.bg = [bg[0], bg[1], bg[2], loss.ssim as f32];
        self.u.misc[2] = loss.depth_per_px as f32;
        let f = |x: f64| x as f32;
        self.u.lr0 = [f(lr.means), f(lr.scales), f(lr.quats), f(lr.opacity)];
        self.u.lr1 = [f(lr.sh0), f(lr.shn), f(lr.bc1), f(lr.bc2)];
        self.flush_uniform();
        let g = self.group();
        let (tx, ty) = (self.tiles.0 as u32, self.tiles.1 as u32);
        let (px, py) = (w.div_ceil(16) as u32, h.div_ceil(16) as u32);
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("step"), timestamp_writes: None });
            p.set_bind_group(0, &g, &[]);
            p.set_pipeline(self.pipe("forward"));
            p.dispatch_workgroups(tx, ty, 1);
            for e in ["ssim_h1", "ssim_v1", "ssim_g", "ssim_h2", "ssim_v2"] {
                p.set_pipeline(self.pipe(e));
                p.dispatch_workgroups(px, py, 1);
            }
            p.set_pipeline(self.pipe("backward"));
            p.dispatch_workgroups(tx, ty, 1);
            self.dispatch_splats(&mut p, "reduce");
            self.dispatch_splats(&mut p, "adjoint");
        }
        self.ctx.queue.submit([enc.finish()]);
        let img = if want_image {
            Some(self.read::<[f32; 4]>(4, w * h))
        } else {
            self.ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("device lost");
            None
        };
        self.last_ms[1] = t0.elapsed().as_secs_f64() * 1e3;
        img
    }

    /// Render the resident cloud (projection already done at this camera).
    pub fn composite(&mut self, bins: &Bins, bg: [f32; 3]) -> Vec<[f32; 4]> {
        let (w, h) = (self.u.size[0] as usize, self.u.size[1] as usize);
        self.upload_bins(bins, (w, h));
        self.u.bg = [bg[0], bg[1], bg[2], self.u.bg[3]];
        self.flush_uniform();
        let g = self.group();
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("composite"), timestamp_writes: None });
            p.set_bind_group(0, &g, &[]);
            p.set_pipeline(self.pipe("forward"));
            p.dispatch_workgroups(self.tiles.0 as u32, self.tiles.1 as u32, 1);
        }
        self.ctx.queue.submit([enc.finish()]);
        self.read::<[f32; 4]>(4, w * h)
    }

    /// The device's per-pixel loss gradient of the last step.
    pub fn last_dl(&self) -> Vec<[f32; 4]> {
        self.read::<[f32; 4]>(6, (self.u.size[0] * self.u.size[1]) as usize)
    }

    // ---- the raster alone, on host-projected splats (gradient checks) ----

    /// Composite host-packed `proj` binned by `bins`.
    pub fn forward(&mut self, proj: &[[f32; 4]], bins: &Bins, size: (usize, usize), bg: [f32; 3]) -> (Vec<[f32; 4]>, Vec<[u32; 2]>) {
        let n = proj.len() / 3;
        self.splats = n;
        self.u.size = [size.0 as u32, size.1 as u32, bins.tiles_x as u32, n as u32];
        self.write(1, proj);
        self.ensure(10, (n.max(1) * 48) as u64);
        let c = self.composite(bins, bg);
        let t = self.read::<[u32; 2]>(5, size.0 * size.1);
        (c, t)
    }

    /// The adjoint of the last [`Self::forward`]: per splat `Grad2`.
    pub fn backward(&mut self, dl: &[[f32; 4]]) -> Vec<Grad2> {
        self.write(6, dl);
        let g = self.group();
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("backward"), timestamp_writes: None });
            p.set_bind_group(0, &g, &[]);
            p.set_pipeline(self.pipe("backward"));
            p.dispatch_workgroups(self.tiles.0 as u32, self.tiles.1 as u32, 1);
            self.dispatch_splats(&mut p, "reduce");
        }
        self.ctx.queue.submit([enc.finish()]);
        self.screen_grads()
    }

    /// The reduced screen-space gradient per splat (binding 10).
    pub fn screen_grads(&self) -> Vec<Grad2> {
        let raw = self.read::<[f32; 4]>(10, 3 * self.splats);
        (0..self.splats)
            .map(|i| {
                let (a, b, c) = (raw[3 * i], raw[3 * i + 1], raw[3 * i + 2]);
                let f = |x: f32| x as f64;
                Grad2 { uv: [f(a[0]), f(a[1])], conic: [f(a[2]), f(a[3]), f(b[0])], opacity: f(b[1]), colour: [f(b[2]), f(b[3]), f(c[0]), f(c[1])] }
            })
            .collect()
    }
}
