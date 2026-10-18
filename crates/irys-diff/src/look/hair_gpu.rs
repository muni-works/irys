//! Hair on the device: fibres in a hardware acceleration structure (wgpu 30
//! ray queries, Metal's on Apple silicon) and [`HairScene`] traced by
//! `hair.wgsl`, forward only.
//!
//! Metal's ray queries have no curve primitive, so a fibre is one of two
//! [`StrandPrim`]s (design.md, Phase 4b, measures both):
//!
//! - [`StrandPrim::Cylinders`]: one AABB per segment, intersected in the
//!   query loop as the exact round cylinder [`Strands`] is on the CPU
//!   (`rayQueryGenerateIntersection`, as the speed work does for spheres);
//! - [`StrandPrim::Tubes`]: an `n`-sided prism per segment in a triangle
//!   BLAS (radius `r π / (n sin(π/n))`, the circle's mean width), committed
//!   by the hardware; the fibre is the triangle's segment, its tangent and
//!   offset taken from that segment's axis.
//!
//! `strands.wgsl` is the shared intersection; `bcsdf.wgsl` the BCSDF.

use bytemuck::{Pod, Zeroable};
use irys_gpu::GpuContext;
use irys_gpu::wgpu::{self, util::DeviceExt};

use super::hair::{HairScene, PARAMS, Strands};

const STRANDS: &str = include_str!("strands.wgsl");
const BCSDF: &str = include_str!("bcsdf.wgsl");
const HAIR: &str = include_str!("hair.wgsl");

/// How fibres enter the hardware structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrandPrim {
    /// AABBs with the exact cylinder in the shader.
    Cylinders,
    /// `n`-sided triangle tubes.
    Tubes(u32),
}

impl StrandPrim {
    fn sides(self) -> u32 {
        match self {
            Self::Cylinders => 0,
            Self::Tubes(n) => n.max(3),
        }
    }
}

/// The WGSL prelude a module that traces strands starts with.
pub fn shader_prelude(prim: StrandPrim) -> String {
    format!(
        "enable wgpu_ray_query;\nconst STRAND_SIDES: u32 = {}u;\n{STRANDS}\n{BCSDF}\n",
        prim.sides()
    )
}

/// Fibres on the device: the acceleration structure (bindings 10) and the
/// segment records (binding 11), two vec4 per segment.
pub struct GpuStrands {
    /// The kind.
    pub prim: StrandPrim,
    pub(crate) tlas: wgpu::Tlas,
    pub(crate) segs: wgpu::Buffer,
    _blas: wgpu::Blas,
    _inputs: Vec<wgpu::Buffer>,
    /// Triangles or boxes in the BLAS.
    pub primitives: usize,
}

fn wait(ctx: &GpuContext) {
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device lost");
}

/// Whether this device can trace fibres (ray queries).
pub fn available(ctx: &GpuContext) -> bool {
    ctx.device
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
}

impl GpuStrands {
    /// Upload `strands` as `prim`; `melanin[i]` scales segment `i`'s
    /// melanin (`None`: 1). `None` without ray queries.
    pub fn new(ctx: &GpuContext, strands: &Strands, melanin: Option<&[f32]>, prim: StrandPrim) -> Option<Self> {
        if !available(ctx) {
            return None;
        }
        let dev = &ctx.device;
        let n = strands.segs.len().max(1);
        let mut recs: Vec<[f32; 4]> = Vec::with_capacity(2 * n);
        for (i, &(a, b, r)) in strands.segs.iter().enumerate() {
            let m = melanin.map_or(1.0, |m| m[i]);
            recs.push([a.x as f32, a.y as f32, a.z as f32, r as f32]);
            recs.push([b.x as f32, b.y as f32, b.z as f32, m]);
        }
        if recs.is_empty() {
            recs = vec![[0.0; 4]; 2];
        }
        let segs = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("strand segs"),
            contents: bytemuck::cast_slice(&recs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let flags = wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE;
        let mut inputs = Vec::new();
        let (blas, primitives) = match prim {
            StrandPrim::Cylinders => {
                let boxes: Vec<[f32; 6]> = strands
                    .segs
                    .iter()
                    .map(|&(a, b, r)| {
                        let pad = |x: f64| r * 1e-3 + 1e-6 * (x.abs() + r);
                        let lo = |p: f64, q: f64| (p.min(q) - r - pad(p)) as f32;
                        let hi = |p: f64, q: f64| (p.max(q) + r + pad(p)) as f32;
                        [lo(a.x, b.x), lo(a.y, b.y), lo(a.z, b.z), hi(a.x, b.x), hi(a.y, b.y), hi(a.z, b.z)]
                    })
                    .collect();
                let ab = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("strand aabbs"),
                    contents: bytemuck::cast_slice(&boxes),
                    usage: wgpu::BufferUsages::BLAS_INPUT,
                });
                let size = wgpu::BlasAABBGeometrySizeDescriptor {
                    primitive_count: boxes.len() as u32,
                    flags: wgpu::AccelerationStructureGeometryFlags::empty(),
                };
                let bl = dev.create_blas(
                    &wgpu::CreateBlasDescriptor {
                        label: Some("strand aabbs"),
                        flags,
                        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                    },
                    wgpu::BlasGeometrySizeDescriptors::AABBs {
                        descriptors: vec![size.clone()],
                    },
                );
                let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                enc.build_acceleration_structures(
                    std::iter::once(&wgpu::BlasBuildEntry {
                        blas: &bl,
                        geometry: wgpu::BlasGeometries::AabbGeometries(vec![wgpu::BlasAabbGeometry {
                            size: &size,
                            stride: 24,
                            aabb_buffer: &ab,
                            primitive_offset: 0,
                        }]),
                    }),
                    std::iter::empty(),
                );
                ctx.queue.submit([enc.finish()]);
                inputs.push(ab);
                (bl, boxes.len())
            }
            StrandPrim::Tubes(_) => {
                let k = prim.sides() as usize;
                let scale = std::f64::consts::PI / (k as f64 * (std::f64::consts::PI / k as f64).sin());
                let mut v: Vec<[f32; 3]> = Vec::with_capacity(strands.segs.len() * 2 * k);
                let mut idx: Vec<u32> = Vec::with_capacity(strands.segs.len() * 6 * k);
                for &(a, b, r) in &strands.segs {
                    let t = (b - a).normalize();
                    let (u, w) = crate::skin::mesh::basis(t);
                    let base = v.len() as u32;
                    for end in [a, b] {
                        for j in 0..k {
                            let ph = 2.0 * std::f64::consts::PI * j as f64 / k as f64;
                            let p = end + (u * ph.cos() + w * ph.sin()) * (r * scale);
                            v.push([p.x as f32, p.y as f32, p.z as f32]);
                        }
                    }
                    for j in 0..k as u32 {
                        let j1 = (j + 1) % k as u32;
                        let (a0, a1, b0, b1) = (base + j, base + j1, base + k as u32 + j, base + k as u32 + j1);
                        idx.extend([a0, a1, b1, a0, b1, b0]);
                    }
                }
                if v.is_empty() {
                    v = vec![[0.0; 3]; 3];
                    idx = vec![0, 1, 2];
                }
                let vb = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("strand tube vertices"),
                    contents: bytemuck::cast_slice(&v),
                    usage: wgpu::BufferUsages::BLAS_INPUT,
                });
                let ib = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("strand tube indices"),
                    contents: bytemuck::cast_slice(&idx),
                    usage: wgpu::BufferUsages::BLAS_INPUT,
                });
                let size = wgpu::BlasTriangleGeometrySizeDescriptor {
                    vertex_format: wgpu::VertexFormat::Float32x3,
                    vertex_count: v.len() as u32,
                    index_format: Some(wgpu::IndexFormat::Uint32),
                    index_count: Some(idx.len() as u32),
                    flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
                };
                let bl = dev.create_blas(
                    &wgpu::CreateBlasDescriptor {
                        label: Some("strand tubes"),
                        flags,
                        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                    },
                    wgpu::BlasGeometrySizeDescriptors::Triangles {
                        descriptors: vec![size.clone()],
                    },
                );
                let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                enc.build_acceleration_structures(
                    std::iter::once(&wgpu::BlasBuildEntry {
                        blas: &bl,
                        geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                            size: &size,
                            vertex_buffer: &vb,
                            first_vertex: 0,
                            vertex_stride: 12,
                            index_buffer: Some(&ib),
                            first_index: Some(0),
                            transform_buffer: None,
                            transform_buffer_offset: None,
                        }]),
                    }),
                    std::iter::empty(),
                );
                ctx.queue.submit([enc.finish()]);
                inputs.extend([vb, ib]);
                (bl, idx.len() / 3)
            }
        };
        // The BLAS build must finish before the TLAS build reads it
        // (gpu/rt.rs: wgpu 30's Metal backend orders neither).
        wait(ctx);
        let mut tlas = dev.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("strand tlas"),
            max_instances: 1,
            flags,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });
        let id = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        tlas[0] = Some(wgpu::TlasInstance::new(&blas, id, 0, 0xff));
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.build_acceleration_structures(std::iter::empty(), [&tlas]);
        ctx.queue.submit([enc.finish()]);
        wait(ctx);
        Some(Self {
            prim,
            tlas,
            segs,
            _blas: blas,
            _inputs: inputs,
            primitives,
        })
    }
}

/// A buffer read back to the host.
pub(crate) fn read_back(ctx: &GpuContext, src: &wgpu::Buffer, bytes: u64) -> Vec<u8> {
    let dev = &ctx.device;
    let staging = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
    ctx.queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    loop {
        wait(ctx);
        match rx.try_recv() {
            Ok(r) => break r.expect("map failed"),
            Err(std::sync::mpsc::TryRecvError::Empty) => std::thread::yield_now(),
            Err(e) => panic!("map: {e}"),
        }
    }
    let out = slice.get_mapped_range().expect("mapped").to_vec();
    staging.unmap();
    out
}

const PROBE: &str = r#"
@group(0) @binding(0) var<storage, read> rays: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> hits: array<vec4<f32>>;

@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = gid.x + gid.y * nwg.x * 64u;
    if (2u * i + 1u >= arrayLength(&rays)) { return; }
    let o = rays[2u * i].xyz;
    let d = rays[2u * i + 1u].xyz;
    let h = strand_hit(o, d, 1e-7, 3.4e38, false);
    if (h.seg == 0xffffffffu) {
        hits[i] = vec4<f32>(-1.0, 0.0, 0.0, 0.0);
        return;
    }
    let tx = strand_tangent(h.seg);
    let z = normalize(-(d - tx * dot(d, tx)));
    let y = cross(z, tx);
    hits[i] = vec4<f32>(h.t, bitcast<f32>(h.seg), dot(h.n, y), 0.0);
}
"#;

/// Closest strand hits for a batch of rays: `(t, segment, h)` per ray (`t <
/// 0`: a miss), and the device time of the dispatch in seconds.
pub fn probe(ctx: &GpuContext, s: &GpuStrands, rays: &[([f32; 3], [f32; 3])]) -> (Vec<(f32, u32, f32)>, f64) {
    let dev = &ctx.device;
    let src = format!("{}{PROBE}", shader_prelude(s.prim));
    let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("strand probe"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let pipe = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("strand probe"),
        layout: None,
        module: &module,
        entry_point: Some("probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let data: Vec<[f32; 4]> = rays
        .iter()
        .flat_map(|(o, d)| [[o[0], o[1], o[2], 0.0], [d[0], d[1], d[2], 0.0]])
        .collect();
    let rb = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("probe rays"),
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let hb = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe hits"),
        size: (rays.len() * 16).max(16) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipe.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: rb.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: hb.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::AccelerationStructure(&s.tlas) },
            wgpu::BindGroupEntry { binding: 11, resource: s.segs.as_entire_binding() },
        ],
    });
    let groups = (rays.len() as u32).div_ceil(64);
    let (gx, gy) = (groups.min(65535), groups.div_ceil(65535));
    let t0 = std::time::Instant::now();
    let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
    }
    ctx.queue.submit([enc.finish()]);
    wait(ctx);
    let secs = t0.elapsed().as_secs_f64();
    let raw = read_back(ctx, &hb, (rays.len() * 16) as u64);
    let f: &[[f32; 4]] = bytemuck::cast_slice(&raw);
    (f.iter().map(|h| (h[0], h[1].to_bits(), h[2])).collect(), secs)
}

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
struct Hp {
    size: [u32; 4],
    seed: [u32; 4],
    eye: [f32; 4],
    fwd: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    scalp: [f32; 4],
    rho: [f32; 4],
    fiber: [f32; 4],
    theta: [f32; 4],
    env: [f32; 4],
    lights: [GpuLight; 4],
}

/// [`HairScene`] on the device.
pub struct GpuHair {
    ctx: &'static GpuContext,
    pipe: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    params: wgpu::Buffer,
    lanes: wgpu::Buffer,
    paths: usize,
    /// The fibres.
    pub strands: GpuStrands,
}

/// Paths per dispatch.
const CHUNK: usize = 1 << 18;

impl GpuHair {
    /// Upload `scene`'s fibres as `prim`. `None` without ray queries.
    pub fn new(ctx: &'static GpuContext, scene: &HairScene, prim: StrandPrim) -> Option<Self> {
        let strands = GpuStrands::new(ctx, scene.bvh.geometry(), None, prim)?;
        let dev = &ctx.device;
        let scope = dev.push_error_scope(wgpu::ErrorFilter::Validation);
        let src = format!("{}{HAIR}", shader_prelude(prim));
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hair"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let pipe = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("hair trace"),
            layout: None,
            module: &module,
            entry_point: Some("trace"),
            compilation_options: Default::default(),
            cache: None,
        });
        let paths = scene.paths();
        let lanes = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hair lanes"),
            size: (paths * 8).max(16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hair params"),
            size: std::mem::size_of::<Hp>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: lanes.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::AccelerationStructure(&strands.tlas),
                },
                wgpu::BindGroupEntry { binding: 11, resource: strands.segs.as_entire_binding() },
            ],
        });
        if let Some(e) = pollster::block_on(scope.pop()) {
            panic!("hair gpu: {e}");
        }
        Some(Self {
            ctx,
            pipe,
            group,
            params,
            lanes,
            paths,
            strands,
        })
    }

    /// `(L, λ)` per path, as [`HairScene::forward`] returns them.
    pub fn forward(&self, scene: &HairScene, theta: &[f64; PARAMS], seed: u64) -> Vec<(f64, f64)> {
        assert_eq!(scene.paths(), self.paths, "frame changed");
        assert!(scene.lights.len() <= 4, "four lights");
        let (dev, queue) = (&self.ctx.device, &self.ctx.queue);
        let ts = scene.sampler.unwrap_or(*theta);
        let fwd = (scene.target - scene.eye).normalize();
        let right = fwd.cross(&irys_core::Vec3::new(0.0, 0.0, 1.0)).normalize();
        let up = right.cross(&fwd);
        let v4 = |x: f64, y: f64, z: f64, w: f64| [x as f32, y as f32, z as f32, w as f32];
        let mut lights = [GpuLight::zeroed(); 4];
        for (g, l) in lights.iter_mut().zip(&scene.lights) {
            let n = l.normal();
            *g = GpuLight {
                c: v4(l.center.x, l.center.y, l.center.z, l.radiance),
                u: v4(l.u.x, l.u.y, l.u.z, l.tint[0]),
                v: v4(l.v.x, l.v.y, l.v.z, l.tint[1]),
                n: v4(n.x, n.y, n.z, l.tint[2]),
            };
        }
        let (sc, rho) = match scene.scalp {
            Some((c, r, a)) => {
                let h = super::openpbr::hats(a);
                (v4(c.x, c.y, c.z, r), v4(h[0], h[1], h[2], 0.0))
            }
            None => ([0.0; 4], [0.0; 4]),
        };
        let (w, h) = scene.size;
        let f = &scene.fiber;
        for first in (0..self.paths).step_by(CHUNK) {
            let n = (self.paths - first).min(CHUNK);
            let p = Hp {
                size: [w as u32, h as u32, scene.spp as u32, scene.depth as u32],
                seed: [seed as u32, (seed >> 32) as u32, first as u32, scene.lights.len() as u32],
                eye: v4(scene.eye.x, scene.eye.y, scene.eye.z, (0.5 * scene.vfov).to_radians().tan()),
                fwd: v4(fwd.x, fwd.y, fwd.z, w as f64 / h as f64),
                right: v4(right.x, right.y, right.z, 0.0),
                up: v4(up.x, up.y, up.z, 0.0),
                scalp: sc,
                rho,
                fiber: v4(f.eta, f.beta_m, f.beta_n, f.alpha),
                theta: v4(theta[0], theta[1], ts[0], ts[1]),
                env: [scene.env as f32, n as f32, 0.0, 0.0],
                lights,
            };
            queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
            let groups = (n as u32).div_ceil(64);
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(&self.pipe);
                pass.set_bind_group(0, &self.group, &[]);
                pass.dispatch_workgroups(groups.min(65535), groups.div_ceil(65535), 1);
            }
            queue.submit([enc.finish()]);
            wait(self.ctx);
        }
        let raw = read_back(self.ctx, &self.lanes, (self.paths * 8) as u64);
        let v: &[[f32; 2]] = bytemuck::cast_slice(&raw);
        v.iter().map(|x| (x[0] as f64, x[1] as f64)).collect()
    }
}
