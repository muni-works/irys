//! The skin integrator on the device (`skin.wgsl`): forward only, one
//! thread per pixel per sample, the image accumulated in place (each thread
//! owns its pixel; samples are separate dispatches, added in sample order,
//! so the frame is bit-deterministic).

use bytemuck::{Pod, Zeroable};
use irys_gpu::GpuContext;
use irys_gpu::wgpu;

use super::graph::PARAMS;
use super::render::{Chromophores, MAX_LIGHTS, NW, Scene};

const SHADER: &str = include_str!("skin.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuLight {
    c: [f32; 4],
    u: [f32; 4],
    v: [f32; 4],
    n: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    size: [u32; 4],
    crop: [u32; 4],
    seed: [u32; 4],
    flags: [u32; 4],
    eye: [f32; 4],
    fwd: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    th0: [f32; 4],
    th1: [f32; 4],
    sm0: [f32; 4],
    sm1: [f32; 4],
    tex: [u32; 4],
    resp: [f32; 4],
    hair0: [f32; 4],
    hair1: [f32; 4],
    lights: [GpuLight; MAX_LIGHTS],
}

/// Fibres in a skin frame ([`GpuSkin::with_strands`]): the device strands,
/// the fibre's shape, and the melanin `θ` (each segment's own scale on top).
pub struct SkinHair<'a> {
    /// The fibres.
    pub strands: &'a crate::look::hair_gpu::GpuStrands,
    /// Shape.
    pub fiber: crate::look::hair::Fiber,
    /// Eumelanin, pheomelanin.
    pub melanin: [f64; 2],
}

/// Pixels per dispatch: keeps one command buffer well under the OS's GPU
/// watchdog on a 1024² SSS frame.
const CHUNK: u32 = 1 << 16;

/// The device-side frame for one scene's geometry and maps.
pub struct GpuSkin {
    ctx: &'static GpuContext,
    pipe: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    params: wgpu::Buffer,
    accum: wgpu::Buffer,
    lanes: wgpu::Buffer,
    pixels: usize,
    paths_cap: usize,
    hair: Option<([f32; 4], [f32; 4])>,
}

fn v4(p: irys_core::Point3, w: f32) -> [f32; 4] {
    [p.x as f32, p.y as f32, p.z as f32, w]
}

fn d4(p: irys_core::Vec3, w: f32) -> [f32; 4] {
    [p.x as f32, p.y as f32, p.z as f32, w]
}

impl GpuSkin {
    /// Upload `scene`'s mesh, tree, maps and the medium table. `lanes` makes
    /// room to read back every path's `L` (small frames, for tests).
    pub fn new(ctx: &'static GpuContext, scene: &Scene, lanes: bool) -> Self {
        Self::with_strands(ctx, scene, lanes, None)
    }

    /// [`Self::new`] with fibres traced alongside the skin: nearest of skin
    /// and fibre per ray, Chiang's BCSDF at a fibre, fibres in every shadow
    /// test (so hair shadows the skin, the SSS exits included); the walks
    /// see only the skin.
    pub fn with_strands(
        ctx: &'static GpuContext,
        scene: &Scene,
        lanes: bool,
        hair: Option<SkinHair<'_>>,
    ) -> Self {
        let dev = &ctx.device;
        let scope = dev.push_error_scope(wgpu::ErrorFilter::Validation);
        let bvh = scene.bvh();
        let (flat, ids) = bvh.flatten();
        let depth = irys_gpu::geometry::tree_depth(&flat);
        assert!(
            depth + 4 <= 64,
            "tree {depth} deep: the WGSL stack holds 64"
        );
        const LEAF: u32 = 0x8000_0000;
        let mut nodes: Vec<[f32; 4]> = Vec::with_capacity(flat.len() * 2);
        for &(b, leaf, a, c) in &flat {
            let c = if leaf { c | LEAF } else { c };
            nodes.push([
                b.min.x as f32,
                b.min.y as f32,
                b.min.z as f32,
                f32::from_bits(a),
            ]);
            nodes.push([
                b.max.x as f32,
                b.max.y as f32,
                b.max.z as f32,
                f32::from_bits(c),
            ]);
        }
        let m = &scene.mesh;
        let mut tris: Vec<[f32; 4]> = Vec::new();
        for t in bvh.geometry().triangles() {
            let [a, b, c] = t.map(|i| i as usize);
            let sign = m.tangents[a][3] as f32;
            tris.extend([
                v4(m.positions[a], sign),
                v4(m.positions[b], 0.0),
                v4(m.positions[c], 0.0),
            ]);
            for i in [a, b, c] {
                let tg = m.tangents[i];
                tris.push(d4(m.normals[i], m.uvs[i][0] as f32));
                tris.push([tg[0] as f32, tg[1] as f32, tg[2] as f32, m.uvs[i][1] as f32]);
            }
        }
        assert_eq!(scene.albedo.size, scene.spec.size, "maps share a size");
        assert_eq!(scene.albedo.size, scene.normal.size, "maps share a size");
        let tex_a: Vec<[f32; 4]> = scene
            .albedo
            .texels
            .iter()
            .zip(&scene.spec.texels)
            .map(|(a, s)| [a[0], a[1], a[2], s[0]])
            .collect();
        let tex_b: Vec<[f32; 4]> = scene
            .normal
            .texels
            .iter()
            .map(|n| [n[0], n[1], n[2], 0.0])
            .collect();
        let chromo: Vec<[f32; 2]> = match &scene.chromo {
            Chromophores::Map(c) => c.iter().map(|t| [t[0], t[1]]).collect(),
            Chromophores::Uniform => vec![[1.0, 1.0]],
        };
        let table: Vec<[f32; 4]> = super::optics::table()
            .iter()
            .map(|r| r.map(|x| x as f32))
            .collect();
        let init = |label: &str, bytes: &[u8]| {
            let b = dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (bytes.len() as u64).max(16).next_multiple_of(4),
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
        let nodes = init("skin nodes", bytemuck::cast_slice(&nodes));
        let ids = init("skin prim ids", bytemuck::cast_slice(&ids));
        let tris = init("skin tris", bytemuck::cast_slice(&tris));
        let tex_a = init("skin tex a", bytemuck::cast_slice(&tex_a));
        let tex_b = init("skin tex b", bytemuck::cast_slice(&tex_b));
        let chromo = init("skin chromo", bytemuck::cast_slice(&chromo));
        let table = init("skin table", bytemuck::cast_slice(&table));
        let pixels = scene.size.0 * scene.size.1;
        let paths_cap = if lanes { scene.paths() } else { 1 };
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
        let accum = storage("skin accum", pixels as u64 * 16);
        let lanes = storage("skin lanes", paths_cap as u64 * 16);
        let params = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skin params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let src = match &hair {
            Some(h) => format!("{}{SHADER}", crate::look::hair_gpu::shader_prelude(h.strands.prim)),
            None => format!(
                "{SHADER}\n{}\n{}",
                include_str!("../look/strands_none.wgsl"),
                include_str!("../look/bcsdf.wgsl")
            ),
        };
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("skin"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let pipe = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("skin trace"),
            layout: None,
            module: &module,
            entry_point: Some("trace"),
            compilation_options: Default::default(),
            cache: None,
        });
        let entries: Vec<(u32, &wgpu::Buffer)> = vec![
            (0, &params),
            (1, &nodes),
            (2, &ids),
            (3, &tris),
            (4, &tex_a),
            (5, &tex_b),
            (6, &chromo),
            (7, &table),
            (8, &accum),
            (9, &lanes),
        ];
        let mut bind: Vec<wgpu::BindGroupEntry> = entries
            .iter()
            .map(|&(binding, b)| wgpu::BindGroupEntry {
                binding,
                resource: b.as_entire_binding(),
            })
            .collect();
        if let Some(h) = &hair {
            bind.push(wgpu::BindGroupEntry {
                binding: 10,
                resource: wgpu::BindingResource::AccelerationStructure(&h.strands.tlas),
            });
            bind.push(wgpu::BindGroupEntry {
                binding: 11,
                resource: h.strands.segs.as_entire_binding(),
            });
        }
        let group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skin"),
            layout: &pipe.get_bind_group_layout(0),
            entries: &bind,
        });
        let hair = hair.map(|h| {
            let f = h.fiber;
            (
                [f.eta as f32, f.beta_m as f32, f.beta_n as f32, f.alpha as f32],
                [h.melanin[0] as f32, h.melanin[1] as f32, 1.0, 0.0],
            )
        });
        if let Some(e) = pollster::block_on(scope.pop()) {
            panic!("skin gpu: {e}");
        }
        Self {
            ctx,
            pipe,
            group,
            params,
            accum,
            lanes,
            pixels,
            paths_cap,
            hair,
        }
    }

    fn params(
        &self,
        scene: &Scene,
        theta: &[f64; PARAMS],
        seed: u64,
        sample: u32,
        first: u32,
        lanes: bool,
    ) -> Params {
        let eff = |t: &[f64; PARAMS]| {
            let mut t = *t;
            if !scene.mode.dual {
                t[4] = t[3];
            }
            t.map(|x| x as f32)
        };
        let th = eff(theta);
        let sm = eff(scene.frozen.as_ref().unwrap_or(theta));
        let c = &scene.camera;
        let band =
            (irys_light::spectrum::LAMBDA_MAX_NM - irys_light::spectrum::LAMBDA_MIN_NM) as f32;
        let norm = irys_light::spectrum::RESPONSE_NORM;
        let per = band / (scene.spp * NW) as f32;
        let mut lights = [GpuLight::zeroed(); MAX_LIGHTS];
        for (g, l) in lights.iter_mut().zip(&scene.lights) {
            let n = l.normal();
            *g = GpuLight {
                c: v4(l.center, l.radiance as f32),
                u: d4(l.u, l.tint[0] as f32),
                v: d4(l.v, l.tint[1] as f32),
                n: d4(n, l.tint[2] as f32),
            };
        }
        Params {
            size: [
                scene.size.0 as u32,
                scene.size.1 as u32,
                scene.full.0 as u32,
                scene.full.1 as u32,
            ],
            crop: [
                scene.crop.0 as u32,
                scene.crop.1 as u32,
                scene.spp as u32,
                scene.depth as u32,
            ],
            seed: [seed as u32, (seed >> 32) as u32, sample, first],
            flags: [
                scene.mode.sss as u32,
                scene.lights.len() as u32,
                scene.max_events as u32,
                matches!(scene.chromo, Chromophores::Uniform) as u32,
            ],
            eye: v4(c.eye, c.tan_half as f32),
            fwd: d4(c.forward, c.aperture as f32),
            right: d4(c.right, c.focus as f32),
            up: d4(c.up, scene.mm_per_unit as f32),
            th0: [th[0], th[1], th[2], th[3]],
            th1: [
                th[4],
                scene.env as f32,
                scene.bump as f32,
                scene.spec_gain as f32,
            ],
            sm0: [sm[0], sm[1], sm[2], sm[3]],
            sm1: [sm[4], 0.0, 0.0, 0.0],
            tex: [
                scene.albedo.size.0 as u32,
                scene.albedo.size.1 as u32,
                self.pixels as u32,
                lanes as u32,
            ],
            resp: [
                norm[0] as f32 * per,
                norm[1] as f32 * per,
                norm[2] as f32 * per,
                0.0,
            ],
            hair0: self.hair.map_or([0.0; 4], |h| h.0),
            hair1: self.hair.map_or([0.0; 4], |h| h.1),
            lights,
        }
    }

    /// Render `scene` at `θ` with `seed`: the linear-sRGB image (`w·h·3`)
    /// and, if built with `lanes`, every path's `L` (`paths · 4`).
    pub fn render(
        &self,
        scene: &Scene,
        theta: &[f64; PARAMS],
        seed: u64,
        lanes: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(
            scene.size.0 * scene.size.1,
            self.pixels,
            "frame size changed"
        );
        assert!(
            !lanes || self.paths_cap >= scene.paths(),
            "built without lanes"
        );
        let (dev, queue) = (&self.ctx.device, &self.ctx.queue);
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.clear_buffer(&self.accum, 0, None);
        queue.submit([enc.finish()]);
        for s in 0..scene.spp as u32 {
            for first in (0..self.pixels as u32).step_by(CHUNK as usize) {
                let n = (self.pixels as u32 - first).min(CHUNK);
                let p = self.params(scene, theta, seed, s, first, lanes);
                queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
                let mut enc =
                    dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&self.pipe);
                    pass.set_bind_group(0, &self.group, &[]);
                    pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
                }
                queue.submit([enc.finish()]);
            }
            // One sample at a time in flight: bounded queue, and the OS sees
            // short command buffers.
            dev.poll(wgpu::PollType::wait_indefinitely())
                .expect("device lost");
        }
        let acc = self.read(&self.accum, self.pixels * 4);
        let img = acc
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect();
        let l = if lanes {
            self.read(&self.lanes, scene.paths() * 4)
        } else {
            Vec::new()
        };
        (img, l)
    }

    fn read(&self, src: &wgpu::Buffer, n: usize) -> Vec<f32> {
        let dev = &self.ctx.device;
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skin readback"),
            size: n as u64 * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, n as u64 * 4);
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
