//! M1: the residual pass on the device (docs/belief-codec.md, M1 item 6).
//!
//! Given a real colour + depth frame and the predicted AOVs, [`Residual`]
//! computes the explained mask under a joint colour / depth noise model
//! ([`Noise`]) and the unexplained connected components as a compact list
//! ([`Blob`]: root pixel, pixel count, bbox, mean depth). Deterministic: the
//! components are labelled by their smallest pixel index (a fixed point that
//! does not depend on thread order), reduced with integer atomics only, and
//! the list is sorted by root on the host. `residual.wgsl` has the model.
//!
//! [`residual_cpu`] is the host reference (same model in f64, union–find).

use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use irys_gpu::GpuContext;
use irys_gpu::wgpu;

use super::pinhole::Pinhole;
use super::sensor::{Sensor, aov, map_read};

const SHADER: &str = include_str!("residual.wgsl");

/// The noise model and the component rules.
#[derive(Debug, Clone, Copy)]
pub struct Noise {
    /// Colour σ floor, sRGB units (0–1).
    pub colour_floor: f32,
    /// Shot-like colour term: `σ_c² = floor² + shot · c_real`.
    pub shot: f32,
    /// Depth σ = `z_a + z_b z²`, metres (stereo: error grows with z²).
    pub z_a: f32,
    /// See `z_a`, 1/m.
    pub z_b: f32,
    /// Explained when `e < tau` (`e` = mean colour χ² + depth χ²; 9.21 is
    /// the 99 % point of χ² with two degrees of freedom).
    pub tau: f32,
    /// Neighbours join a component when their real depths differ by at most
    /// `jump_abs + jump_rel · z` (either depth missing: they join).
    pub jump_abs: f32,
    /// See `jump_abs`.
    pub jump_rel: f32,
    /// Components smaller than this are not listed (they stay in the mask).
    pub min_pixels: u32,
}

impl Default for Noise {
    fn default() -> Self {
        Self { colour_floor: 0.04, shot: 0.002, z_a: 0.01, z_b: 0.01, tau: 9.21, jump_abs: 0.05, jump_rel: 0.05, min_pixels: 16 }
    }
}

/// One unexplained component.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Blob {
    /// Its smallest pixel index (`y · w + x`): its label.
    pub root: u32,
    /// Pixels.
    pub pixels: u32,
    /// `[x0, y0, x1, y1]`, inclusive.
    pub bbox: [u32; 4],
    /// Mean real optical depth over its pixels that have one, metres
    /// (rounded per pixel to the millimetre).
    pub mean_z: Option<f64>,
}

/// What a residual pass returns.
#[derive(Debug, Clone, Default)]
pub struct Output {
    /// Pixels explained.
    pub explained: usize,
    /// Components of at least `min_pixels`, by root.
    pub blobs: Vec<Blob>,
    /// More components than the list holds (the rest were dropped).
    pub overflow: bool,
    /// Label-propagation submits until the fixed point.
    pub rounds: u32,
    /// Wall time, seconds.
    pub seconds: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    size: [u32; 4],
    cam: [f32; 4],
    noise: [f32; 4],
    misc: [f32; 4],
    lim: [u32; 4],
}

/// Propagation steps (min + jump) per submit between convergence checks.
const STEPS: usize = 32;
/// Components the list holds.
pub const LIST: usize = 4096;

/// The residual pass for frames of one size.
pub struct Residual {
    ctx: &'static GpuContext,
    pipes: [wgpu::ComputePipeline; 5],
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    real_rgb: wgpu::Buffer,
    real_z: wgpu::Buffer,
    host_pred: [wgpu::Buffer; 3],
    mask: wgpu::Buffer,
    label: wgpu::Buffer,
    stats: wgpu::Buffer,
    list: wgpu::Buffer,
    flag: wgpu::Buffer,
    err: wgpu::Buffer,
    size: (usize, usize),
}

impl Residual {
    /// Pipelines and buffers for `size` frames.
    pub fn new(ctx: &'static GpuContext, size: (usize, usize)) -> Self {
        let dev = &ctx.device;
        let n = (size.0 * size.1) as u64;
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("residual"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("residual"),
            entries: &(0..12u32)
                .map(|b| wgpu::BindGroupLayoutEntry {
                    binding: b,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if b == 0 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage { read_only: (1..=5).contains(&b) }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("residual"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mk = |e: &str| {
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("residual"),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(e),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipes = ["classify", "cc_min", "cc_jump", "cc_stats", "cc_compact"].map(mk);
        let buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            dev.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes.max(16), usage, mapped_at_creation: false })
        };
        let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        Self {
            ctx,
            pipes,
            layout,
            params: buf("residual params", 80, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            real_rgb: buf("residual real rgb", n * 4, st),
            real_z: buf("residual real z", n * 4, st),
            host_pred: [buf("residual pred rgb", n * 16, st), buf("residual pred dist", n * 4, st), buf("residual pred sigma", n * 4, st)],
            mask: buf("residual mask", n * 4, st),
            label: buf("residual label", n * 4, st),
            stats: buf("residual stats", n * 28, st),
            list: buf("residual list", (1 + 8 * LIST as u64) * 4, st),
            flag: buf("residual flag", 4, st),
            err: buf("residual err", n * 4, st),
            size,
        }
    }

    /// Run on predicted AOVs from the host (linear rgb, ray distance, σ).
    #[allow(clippy::too_many_arguments)]
    pub fn run_host(
        &mut self,
        real_rgb: &[[u8; 3]],
        real_z: &[f32],
        pred_rgb: &[[f32; 4]],
        pred_dist: &[f32],
        sigma: Option<&[f32]>,
        k: &Pinhole,
        noise: &Noise,
    ) -> Output {
        let q = &self.ctx.queue;
        q.write_buffer(&self.host_pred[0], 0, bytemuck::cast_slice(pred_rgb));
        q.write_buffer(&self.host_pred[1], 0, bytemuck::cast_slice(pred_dist));
        if let Some(s) = sigma {
            q.write_buffer(&self.host_pred[2], 0, bytemuck::cast_slice(s));
        }
        let [a, b, c] = &self.host_pred;
        let (a, b, c) = (a.clone(), b.clone(), c.clone());
        self.run(real_rgb, real_z, (&a, &b, &c), 0, sigma.is_some(), k, noise)
    }

    /// Run on camera `camera` of the last [`Sensor::trace`], which must have
    /// traced [`aov::RGB`] and [`aov::DIST`] (and [`aov::SIGMA`] if `sigma`):
    /// the prediction never leaves the device.
    #[allow(clippy::too_many_arguments)]
    pub fn run_sensor(
        &mut self,
        real_rgb: &[[u8; 3]],
        real_z: &[f32],
        sensor: &Sensor,
        camera: usize,
        sigma: bool,
        k: &Pinhole,
        noise: &Noise,
    ) -> Output {
        assert_eq!(sensor.size(), self.size);
        let first = camera * self.size.0 * self.size.1;
        let (a, b, c) = (sensor.buffer(aov::RGB).clone(), sensor.buffer(aov::DIST).clone(), sensor.buffer(aov::SIGMA).clone());
        self.run(real_rgb, real_z, (&a, &b, &c), first as u32, sigma, k, noise)
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        real_rgb: &[[u8; 3]],
        real_z: &[f32],
        pred: (&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer),
        first: u32,
        sigma: bool,
        k: &Pinhole,
        noise: &Noise,
    ) -> Output {
        let t0 = Instant::now();
        let (w, h) = self.size;
        let n = w * h;
        assert_eq!(real_rgb.len(), n);
        assert_eq!(real_z.len(), n);
        let (dev, q) = (&self.ctx.device, &self.ctx.queue);
        let packed: Vec<u32> = real_rgb.iter().map(|c| c[0] as u32 | (c[1] as u32) << 8 | (c[2] as u32) << 16).collect();
        q.write_buffer(&self.real_rgb, 0, bytemuck::cast_slice(&packed));
        q.write_buffer(&self.real_z, 0, bytemuck::cast_slice(real_z));
        let p = Params {
            size: [w as u32, h as u32, first, sigma as u32],
            cam: [k.fx as f32, k.fy as f32, k.cx as f32, k.cy as f32],
            noise: [noise.colour_floor, noise.shot, noise.z_a, noise.z_b],
            misc: [noise.tau, noise.jump_abs, noise.jump_rel, 0.0],
            lim: [noise.min_pixels.max(1), LIST as u32, 0, 0],
        };
        q.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
        let bufs: [&wgpu::Buffer; 12] = [
            &self.params,
            &self.real_rgb,
            &self.real_z,
            pred.0,
            pred.1,
            pred.2,
            &self.mask,
            &self.label,
            &self.stats,
            &self.list,
            &self.flag,
            &self.err,
        ];
        let group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("residual"),
            layout: &self.layout,
            entries: &bufs
                .iter()
                .enumerate()
                .map(|(b, r)| wgpu::BindGroupEntry { binding: b as u32, resource: r.as_entire_binding() })
                .collect::<Vec<_>>(),
        });
        let groups = n.div_ceil(64) as u32;
        let pass = |enc: &mut wgpu::CommandEncoder, k: usize| {
            let mut ps = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            ps.set_pipeline(&self.pipes[k]);
            ps.set_bind_group(0, &group, &[]);
            ps.dispatch_workgroups(groups, 1, 1);
        };
        let flag_read = || {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("residual flag read"),
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.clear_buffer(&self.flag, 0, None);
        pass(&mut enc, 0);
        let mut rounds = 0;
        loop {
            rounds += 1;
            enc.clear_buffer(&self.flag, 0, Some(4));
            for _ in 0..STEPS {
                pass(&mut enc, 1);
                pass(&mut enc, 2);
            }
            let fr = flag_read();
            enc.copy_buffer_to_buffer(&self.flag, 0, &fr, 0, 4);
            q.submit([enc.finish()]);
            let f = map_read(self.ctx, &fr);
            enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            if f[..4] == [0, 0, 0, 0] {
                break;
            }
        }
        enc.clear_buffer(&self.stats, 0, None);
        enc.clear_buffer(&self.list, 0, None);
        pass(&mut enc, 3);
        pass(&mut enc, 4);
        // the list and the explained count come back in one readback
        let lr = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("residual list read"),
            size: (2 + 8 * LIST as u64) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_buffer_to_buffer(&self.list, 0, &lr, 0, (1 + 8 * LIST as u64) * 4);
        enc.copy_buffer_to_buffer(&self.flag, 4, &lr, (1 + 8 * LIST as u64) * 4, 4);
        q.submit([enc.finish()]);
        let raw = map_read(self.ctx, &lr);
        let l: &[u32] = bytemuck::cast_slice(&raw);
        let count = l[0] as usize;
        let mut blobs: Vec<Blob> = (0..count.min(LIST))
            .map(|i| {
                let e = &l[1 + 8 * i..9 + 8 * i];
                Blob {
                    root: e[0],
                    pixels: e[1],
                    bbox: [e[2], e[3], e[4], e[5]],
                    mean_z: (e[7] > 0).then(|| e[6] as f64 / 1000.0 / e[7] as f64),
                }
            })
            .collect();
        blobs.sort_by_key(|b| b.root);
        let explained = l[1 + 8 * LIST] as usize;
        Output {
            explained,
            blobs,
            overflow: count > LIST,
            rounds,
            seconds: t0.elapsed().as_secs_f64(),
        }
    }

    fn read_u32(&self, b: &wgpu::Buffer) -> Vec<u32> {
        let n = self.size.0 * self.size.1;
        let dev = &self.ctx.device;
        let s = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("residual read"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(b, 0, &s, 0, (n * 4) as u64);
        self.ctx.queue.submit([enc.finish()]);
        bytemuck::cast_slice(&map_read(self.ctx, &s)).to_vec()
    }

    /// The explained mask of the last run (1: explained).
    pub fn mask(&self) -> Vec<u8> {
        self.read_u32(&self.mask).into_iter().map(|m| m as u8).collect()
    }

    /// The component label per pixel of the last run (`u32::MAX`: explained).
    pub fn labels(&self) -> Vec<u32> {
        self.read_u32(&self.label)
    }

    /// The per-pixel error `e` of the last run.
    pub fn errors(&self) -> Vec<f32> {
        bytemuck::cast_slice(&self.read_u32(&self.err)).to_vec()
    }
}

/// The host reference: the same model in f64 and a union–find over the
/// same joins. Returns `(mask, blobs)`; blobs by root, `min_pixels` applied.
pub fn residual_cpu(
    real_rgb: &[[u8; 3]],
    real_z: &[f32],
    pred_rgb: &[[f32; 4]],
    pred_dist: &[f32],
    sigma: Option<&[f32]>,
    k: &Pinhole,
    size: (usize, usize),
    noise: &Noise,
) -> (Vec<u8>, Vec<Blob>) {
    let (w, h) = size;
    let n = w * h;
    let enc = |c: f64| super::splat_gpu::srgb_enc(c);
    let mut mask = vec![0u8; n];
    for p in 0..n {
        let (u, v) = ((p % w) as f64, (p / w) as f64);
        let mut e = 0.0;
        for ch in 0..3 {
            let r = real_rgb[p][ch] as f64 / 255.0;
            let s2 = (noise.colour_floor as f64).powi(2) + noise.shot as f64 * r;
            e += (r - enc(pred_rgb[p][ch] as f64)).powi(2) / s2;
        }
        e /= 3.0;
        let zr = real_z[p] as f64;
        let zp = pred_dist[p] as f64 * k.z_per_dist(u, v);
        if zr > 0.0 {
            if zp > 0.0 && zp.is_finite() {
                let sz = noise.z_a as f64 + noise.z_b as f64 * zr * zr;
                let s2 = sz * sz + sigma.map_or(0.0, |s| (s[p] as f64).powi(2));
                e += (zr - zp).powi(2) / s2;
            } else {
                e = f64::INFINITY;
            }
        }
        mask[p] = (e < noise.tau as f64) as u8;
    }
    let blobs = components(&mask, real_z, size, noise);
    (mask, blobs)
}

/// Union–find components of the unexplained pixels of `mask` under the
/// residual pass's joins.
pub fn components(mask: &[u8], real_z: &[f32], size: (usize, usize), noise: &Noise) -> Vec<Blob> {
    let (w, h) = size;
    let n = w * h;
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    let joined = |a: usize, b: usize| {
        if mask[b] != 0 {
            return false;
        }
        let (za, zb) = (real_z[a], real_z[b]);
        if za > 0.0 && zb > 0.0 {
            (za - zb).abs() <= noise.jump_abs + noise.jump_rel * za.min(zb)
        } else {
            true
        }
    };
    for p in 0..n {
        if mask[p] != 0 {
            continue;
        }
        let x = p % w;
        for q in [(x + 1 < w).then_some(p + 1), (p + w < n).then_some(p + w)].into_iter().flatten() {
            if joined(p, q) && joined(q, p) {
                let (a, b) = (find(&mut parent, p), find(&mut parent, q));
                let (lo, hi) = (a.min(b), a.max(b));
                parent[hi] = lo;
            }
        }
    }
    let mut acc: std::collections::BTreeMap<usize, (u32, [u32; 4], u64, u32)> = Default::default();
    for p in 0..n {
        if mask[p] != 0 {
            continue;
        }
        let r = find(&mut parent, p);
        let (x, y) = ((p % w) as u32, (p / w) as u32);
        let e = acc.entry(r).or_insert((0, [u32::MAX, u32::MAX, 0, 0], 0, 0));
        e.0 += 1;
        e.1 = [e.1[0].min(x), e.1[1].min(y), e.1[2].max(x), e.1[3].max(y)];
        if real_z[p] > 0.0 {
            e.2 += (real_z[p] * 1000.0).round() as u64;
            e.3 += 1;
        }
    }
    let _ = h;
    acc.into_iter()
        .filter(|(_, e)| e.0 >= noise.min_pixels.max(1))
        .map(|(r, e)| Blob { root: r as u32, pixels: e.0, bbox: e.1, mean_z: (e.3 > 0).then(|| e.2 as f64 / 1000.0 / e.3 as f64) })
        .collect()
}
