//! The tile rasteriser on irys's wgpu 30 device (`raster.wgsl`).
//!
//! The host projects and bins (`project`, `tiles`); the device composites
//! every pixel of every tile (the O(pixels × depth) part), runs the adjoint,
//! and reduces each splat's per-tile records in a fixed order. Buffers grow
//! on demand and are reused across iterations.

use bytemuck::{Pod, Zeroable};
use irys_gpu::GpuContext;
use irys_gpu::wgpu;

use crate::project::{Grad2, Proj};
use crate::tiles::Bins;

const SHADER: &str = include_str!("raster.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniform {
    size: [u32; 4],
    bg: [f32; 4],
}

struct Buf {
    buf: wgpu::Buffer,
    cap: u64,
}

pub struct Raster {
    ctx: &'static GpuContext,
    pipes: [wgpu::ComputePipeline; 3],
    layout: wgpu::BindGroupLayout,
    uni: wgpu::Buffer,
    bufs: Vec<Buf>, // bindings 1..=10
    group: Option<wgpu::BindGroup>,
    size: (usize, usize),
    tiles: (usize, usize),
    splats: usize,
    /// Seconds of device time (submit → readback) of the last call, per pass.
    pub last_ms: [f64; 2],
}

/// The device's packing of one projected splat: 3 × vec4.
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

fn storage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
}

impl Raster {
    pub fn new(ctx: &'static GpuContext) -> Self {
        let dev = &ctx.device;
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splat raster"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let ro = [1u32, 2, 3, 6, 8, 9];
        let layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splat raster"),
            entries: &(0..11u32)
                .map(|b| wgpu::BindGroupLayoutEntry {
                    binding: b,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if b == 0 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage { read_only: ro.contains(&b) }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat raster"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mk = |e: &str| {
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(e),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(e),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipes = ["forward", "backward", "reduce"].map(mk);
        let uni = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splat raster uniform"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bufs = (1..=10)
            .map(|b| Buf {
                buf: dev.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("splat raster {b}")),
                    size: 256,
                    usage: storage(),
                    mapped_at_creation: false,
                }),
                cap: 256,
            })
            .collect();
        Self { ctx, pipes, layout, uni, bufs, group: None, size: (0, 0), tiles: (0, 0), splats: 0, last_ms: [0.0; 2] }
    }

    fn ensure(&mut self, binding: usize, bytes: u64) {
        let b = &mut self.bufs[binding - 1];
        if b.cap >= bytes {
            return;
        }
        let cap = (bytes + bytes / 2).max(256).next_multiple_of(256);
        b.buf = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("splat raster {binding}")),
            size: cap,
            usage: storage(),
            mapped_at_creation: false,
        });
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
        let g = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("splat raster"), layout: &self.layout, entries: &entries });
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
            label: Some("splat raster readback"),
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

    /// Composite `proj` (from [`pack`]) binned by `bins` at `size`: per
    /// pixel rgb + depth, and `(T_N, last)`.
    pub fn forward(&mut self, proj: &[[f32; 4]], bins: &Bins, size: (usize, usize), bg: [f32; 3]) -> (Vec<[f32; 4]>, Vec<[u32; 2]>) {
        let t0 = std::time::Instant::now();
        let (w, h) = size;
        let n = proj.len() / 3;
        self.size = size;
        self.tiles = (bins.tiles_x, bins.tiles_y);
        self.splats = n;
        let u = Uniform { size: [w as u32, h as u32, bins.tiles_x as u32, n as u32], bg: [bg[0], bg[1], bg[2], 0.0] };
        self.ctx.queue.write_buffer(&self.uni, 0, bytemuck::bytes_of(&u));
        self.write(1, proj);
        self.write(2, &bins.ranges);
        self.write(3, &bins.ids);
        self.ensure(4, (w * h * 16) as u64);
        self.ensure(5, (w * h * 8) as u64);
        self.ensure(6, (w * h * 16) as u64);
        self.ensure(7, (bins.ids.len().max(1) * 48) as u64);
        self.write(8, &bins.csr_off);
        self.write(9, &bins.csr_slots);
        self.ensure(10, (n.max(1) * 48) as u64);
        let g = self.group();
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("splat forward"), timestamp_writes: None });
            p.set_pipeline(&self.pipes[0]);
            p.set_bind_group(0, &g, &[]);
            p.dispatch_workgroups(bins.tiles_x as u32, bins.tiles_y as u32, 1);
        }
        self.ctx.queue.submit([enc.finish()]);
        let c = self.read::<[f32; 4]>(4, w * h);
        let t = self.read::<[u32; 2]>(5, w * h);
        self.last_ms[0] = t0.elapsed().as_secs_f64() * 1e3;
        (c, t)
    }

    /// The adjoint of the last [`Self::forward`]: per splat `Grad2` from
    /// `∂L/∂(rgb, depth)` per pixel.
    pub fn backward(&mut self, dl: &[[f32; 4]]) -> Vec<Grad2> {
        let t0 = std::time::Instant::now();
        self.write(6, dl);
        let g = self.group();
        let n = self.splats;
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut p = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("splat backward"), timestamp_writes: None });
            p.set_pipeline(&self.pipes[1]);
            p.set_bind_group(0, &g, &[]);
            p.dispatch_workgroups(self.tiles.0 as u32, self.tiles.1 as u32, 1);
            p.set_pipeline(&self.pipes[2]);
            let groups = n.div_ceil(256).max(1);
            let (gx, gy) = if groups > 65535 { (65535, groups.div_ceil(65535)) } else { (groups, 1) };
            p.dispatch_workgroups(gx as u32, gy as u32, 1);
        }
        self.ctx.queue.submit([enc.finish()]);
        let raw = self.read::<[f32; 4]>(10, 3 * n);
        self.last_ms[1] = t0.elapsed().as_secs_f64() * 1e3;
        (0..n)
            .map(|i| {
                let (a, b, c) = (raw[3 * i], raw[3 * i + 1], raw[3 * i + 2]);
                let f = |x: f32| x as f64;
                Grad2 {
                    uv: [f(a[0]), f(a[1])],
                    conic: [f(a[2]), f(a[3]), f(b[0])],
                    opacity: f(b[1]),
                    colour: [f(b[2]), f(b[3]), f(c[0]), f(c[1])],
                }
            })
            .collect()
    }
}
