//! Phase 5b: [`LookScene`] on the device (`scene.wgsl`): per-material
//! parameter buffers, textures, the albedo tables, per-emitter radiance,
//! hardware ray queries with instance masks and non-opaque triangles.
//!
//! Forward only. A frame is bit-deterministic: one dispatch per sample (in
//! chunks of pixels), each thread owning its pixel's accumulator and each
//! workgroup its own slots of the mean spectrum.

use bytemuck::{Pod, Zeroable};
use irys_gpu::GpuContext;
use irys_gpu::wgpu::{self, util::DeviceExt};

use super::openpbr::{NE, OpenPbr, TN, Tables, fresnel_average, hats};
use super::scene::{Camera, LookScene, Material, hat_inverse};

const SHADER: &str = include_str!("scene.wgsl");
/// Spectrum bins across 380–780 nm.
pub const BINS: usize = 40;
const MS: usize = 20;
const NONE: u32 = u32::MAX;
/// Pixels per dispatch.
const CHUNK: usize = 1 << 16;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    size: [u32; 4],
    seed: [u32; 4],
    flags: [u32; 4],
    eye: [f32; 4],
    fwd: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    resp: [f32; 4],
    sun_dir: [f32; 4],
    sun_l: [f32; 4],
    picks: [f32; 4],
    env: [u32; 4],
    minv0: [f32; 4],
    minv1: [f32; 4],
    minv2: [f32; 4],
    emit_off: [u32; 4],
}

/// How to render a frame.
#[derive(Debug, Clone, Copy)]
pub struct Opts {
    /// Samples per pixel.
    pub spp: u32,
    /// Cycles' `max_bounces`: scattering events per path.
    pub max_bounces: u32,
    /// Russian roulette from this bounce on (`u32::MAX`: off).
    pub rr_from: u32,
    /// Per-sample contribution clamp after the first bounce (0: off),
    /// Cycles' `sample_clamp_indirect`.
    pub clamp: f32,
    /// Seed.
    pub seed: u64,
    /// Debug view (0: off; 1 material hash, 2 shading normal, 3 tw / front / t).
    pub debug: u32,
    /// Evaluate every material per wavelength (the dispersive path) even
    /// where the four-lane path applies: the test of one against the other.
    pub per_lane: bool,
}

impl Default for Opts {
    fn default() -> Self {
        Self { spp: 16, max_bounces: 7, rr_from: u32::MAX, clamp: 0.0, seed: 1, debug: 0, per_lane: false }
    }
}

/// A rendered frame.
pub struct Frame {
    /// Linear Rec.709, `w · h · 3`, the mean over samples.
    pub rgb: Vec<f32>,
    /// The frame's mean spectral radiance in [`BINS`] bins over 380–780 nm
    /// (per-wavelength radiance averaged over pixels and samples).
    pub spectrum: Vec<f64>,
    /// Wall time of the samples (device), seconds.
    pub seconds: f64,
}

/// What the device holds for one scene.
pub struct GpuLook {
    ctx: &'static GpuContext,
    pipe: wgpu::ComputePipeline,
    probe_pipe: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    params: wgpu::Buffer,
    accum: wgpu::Buffer,
    spec: wgpu::Buffer,
    probe_io: wgpu::Buffer,
    pixels: usize,
    size: (usize, usize),
    flags: [u32; 4],
    sun: ([f32; 4], [f32; 4]),
    picks: [f32; 4],
    env: [u32; 4],
    emit_off: [u32; 4],
    _keep: Vec<wgpu::Buffer>,
    _blas: Vec<wgpu::Blas>,
    _tlas: wgpu::Tlas,
    /// Placed emitter triangles (lights and emissive meshes), portal triangles.
    pub emitters: (usize, usize),
    /// Light-group shares: sun, world, emitters, portals.
    pub shares: [f32; 4],
}

fn lum(c: [f64; 3]) -> f64 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

fn h4(c: [f64; 3], w: f64) -> [f32; 4] {
    let h = hats(c);
    [h[0] as f32, h[1] as f32, h[2] as f32, w as f32]
}

/// One material's [`MS`] vec4 (scene.wgsl's `load_mat`).
fn pack_material(m: &Material, portal_off: Option<u32>) -> [[f32; 4]; MS] {
    let p: &OpenPbr = &m.pbr;
    let mut flags = 0u32;
    if m.thin_walled {
        flags |= 1;
    }
    if m.cutout {
        flags |= 2;
    }
    if m.non_opaque() {
        flags |= 4;
    }
    if m.emit_back {
        flags |= 8;
    }
    if m.portal_sky.is_some() {
        flags |= 16;
    }
    if m.spot.is_some() {
        flags |= 32;
    }
    if m.emit_both {
        flags |= 64;
    }
    if m.emits() {
        flags |= 128;
    }
    if m.metal_color.is_some() {
        flags |= 256;
    }
    let abbe = if p.transmission_dispersion_scale > 0.0 {
        p.transmission_dispersion_abbe_number / p.transmission_dispersion_scale
    } else {
        0.0
    };
    let bits = |u: Option<usize>| f32::from_bits(u.map_or(NONE, |x| x as u32));
    let mut o = [[0f32; 4]; MS];
    o[0] = h4(p.base_color, p.base_weight);
    o[1] = h4(m.metal_color.unwrap_or(p.base_color), p.base_metalness);
    o[2] = h4(p.specular_color, p.specular_weight);
    o[3] = [p.specular_roughness as f32, p.specular_ior as f32, p.transmission_weight as f32, p.transmission_depth as f32];
    o[4] = h4(p.transmission_color, abbe);
    o[5] = h4(p.subsurface_color, p.subsurface_weight);
    o[6] = h4(p.coat_color, p.coat_weight);
    o[7] = [p.coat_roughness as f32, p.coat_ior as f32, p.coat_darkening as f32, p.fuzz_weight as f32];
    o[8] = h4(p.fuzz_color, p.fuzz_roughness);
    o[9] = [
        p.thin_film_weight as f32,
        (p.thin_film_thickness * 1e3) as f32,
        p.thin_film_ior as f32,
        fresnel_average(p.coat_ior) as f32,
    ];
    let e = hats(m.emission);
    o[10] = [e[0] as f32, e[1] as f32, e[2] as f32, f32::from_bits(flags)];
    o[11] = h4(m.translucent_color, m.translucency);
    o[12] = [bits(m.base_tex), bits(m.metal_tex), bits(m.alpha_tex), m.opacity as f32];
    o[13] = m.uv_xf;
    let (c, a, cos, sm) = m.spot.map_or(([0.0; 3], [0.0, 0.0, 1.0], -1.0, 1.0), |s| (s.center, s.axis, s.cos_half, s.smooth));
    o[14] = [f32::from_bits(portal_off.unwrap_or(0)), c[0] as f32, c[1] as f32, c[2] as f32];
    o[15] = [a[0] as f32, a[1] as f32, a[2] as f32, cos as f32];
    o[16] = [sm as f32, 0.0, 0.0, 0.0];
    o
}

fn apply(xf: &[[f32; 4]; 3], p: [f32; 3]) -> [f64; 3] {
    std::array::from_fn(|r| {
        (xf[r][0] * p[0] + xf[r][1] * p[1] + xf[r][2] * p[2] + xf[r][3]) as f64
    })
}

/// Normalised inclusive CDF of `w` (all zero: uniform).
fn cdf(w: &[f64]) -> Vec<f32> {
    let s: f64 = w.iter().sum();
    let mut acc = 0.0;
    let n = w.len().max(1) as f64;
    w.iter()
        .enumerate()
        .map(|(i, x)| {
            acc += if s > 0.0 { x / s } else { 1.0 / n };
            if i + 1 == w.len() { 1.0 } else { acc as f32 }
        })
        .collect()
}

impl GpuLook {
    /// Upload `scene` and build its acceleration structures. `None` when the
    /// device has no ray queries.
    pub fn new(ctx: &'static GpuContext, scene: &LookScene) -> Option<Self> {
        let dev = &ctx.device;
        if !dev.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
            return None;
        }
        let scope = dev.push_error_scope(wgpu::ErrorFilter::Validation);
        // Triangles, mesh after mesh.
        let mut tris: Vec<[f32; 4]> = Vec::new();
        let mut first_tri = Vec::with_capacity(scene.meshes.len());
        for m in &scene.meshes {
            first_tri.push((tris.len() / 7) as u32);
            for t in m.indices.chunks_exact(3) {
                let [a, b, c] = [t[0] as usize, t[1] as usize, t[2] as usize];
                for &i in &[a, b, c] {
                    let p = m.positions[i];
                    tris.push([p[0], p[1], p[2], 0.0]);
                }
                let pa = m.positions[a];
                let (pb, pc) = (m.positions[b], m.positions[c]);
                let e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                let e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
                let fnrm = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
                let uv = |i: usize| m.uvs.get(i).copied().unwrap_or([0.0, 0.0]);
                for &i in &[a, b, c] {
                    let n = m.normals.get(i).copied().unwrap_or(fnrm);
                    tris.push([n[0], n[1], n[2], uv(i)[0]]);
                }
                tris.push([uv(a)[1], uv(b)[1], uv(c)[1], 0.0]);
            }
        }
        // Materials, portal skies after the world in one buffer.
        let mut envt: Vec<[f32; 4]> = Vec::new();
        let mut env = [0u32; 4];
        if let Some(e) = &scene.env {
            let (w, h) = (e.size.0 as usize, e.size.1 as usize);
            let mut rows = Vec::with_capacity(h);
            for j in 0..h {
                let th = std::f64::consts::PI * (j as f64 + 0.5) / h as f64;
                let row: Vec<f64> = (0..w)
                    .map(|i| {
                        let c = e.rgb[j * w + i];
                        lum([c[0] as f64, c[1] as f64, c[2] as f64]).max(0.0) * th.sin()
                    })
                    .collect();
                let c = cdf(&row);
                for i in 0..w {
                    let t = e.rgb[j * w + i];
                    envt.push([t[0], t[1], t[2], c[i]]);
                }
                rows.push(row.iter().sum::<f64>());
            }
            let mc = cdf(&rows);
            let tot: f64 = rows.iter().sum();
            for j in 0..h {
                envt.push([mc[j], (rows[j] / tot.max(1e-30)) as f32, 0.0, 0.0]);
            }
            env = [w as u32, h as u32, 1, (w * h) as u32];
        }
        let mut sky_off = Vec::new();
        for q in &scene.portal_skies {
            sky_off.push(envt.len() as u32);
            envt.push([f32::from_bits(q.size.0), f32::from_bits(q.size.1), 0.0, 0.0]);
            envt.extend(q.rgb.iter().map(|t| [t[0], t[1], t[2], 0.0]));
        }
        if envt.is_empty() {
            envt.push([0.0; 4]);
        }
        let mats: Vec<[f32; 4]> = scene
            .materials
            .iter()
            .flat_map(|m| pack_material(m, m.portal_sky.map(|i| sky_off[i])))
            .collect();
        // Emitters: placed triangles of emitting materials, then portals.
        let mut emit: Vec<[f32; 4]> = Vec::new();
        let mut inst: Vec<[u32; 4]> = Vec::new();
        let mut groups: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        let mut power = [0.0f64; 2];
        let mut order: Vec<(usize, usize)> = Vec::new(); // (group, instance)
        for (k, i) in scene.instances.iter().enumerate() {
            let m = &scene.materials[scene.meshes[i.mesh].material];
            if m.emits() {
                order.push((if m.portal_sky.is_some() { 1 } else { 0 }, k));
            }
        }
        order.sort_by_key(|&(g, k)| (g, k));
        let mut emit_base = vec![NONE; scene.instances.len()];
        let mut portal_first = 0u32;
        let mut seen_portal = false;
        for &(g, k) in &order {
            if g == 1 && !seen_portal {
                portal_first = (emit.len() / 3) as u32;
                seen_portal = true;
            }
            let i = &scene.instances[k];
            let mesh = &scene.meshes[i.mesh];
            let mi = mesh.material;
            let m = &scene.materials[mi];
            let l = if let Some(s) = m.portal_sky {
                let q = &scene.portal_skies[s];
                q.rgb.iter().map(|t| lum([t[0] as f64, t[1] as f64, t[2] as f64])).sum::<f64>() / q.rgb.len() as f64
            } else {
                lum(m.emission)
            };
            emit_base[k] = (emit.len() / 3) as u32;
            for t in mesh.indices.chunks_exact(3) {
                let p = [t[0], t[1], t[2]].map(|v| apply(&i.xf, mesh.positions[v as usize]));
                let e1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
                let e2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
                let c = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
                let area = 0.5 * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
                emit.push([p[0][0] as f32, p[0][1] as f32, p[0][2] as f32, f32::from_bits(mi as u32)]);
                emit.push([p[1][0] as f32, p[1][1] as f32, p[1][2] as f32, area.max(1e-12) as f32]);
                emit.push([p[2][0] as f32, p[2][1] as f32, p[2][2] as f32, 0.0]);
                groups[g].push(area * l);
                power[g] += area * l;
            }
        }
        let n_emit = groups[0].len();
        let n_portal = groups[1].len();
        if !seen_portal {
            portal_first = n_emit as u32;
        }
        let c0 = cdf(&groups[0]);
        let c1 = cdf(&groups[1]);
        for (j, c) in c0.iter().chain(c1.iter()).enumerate() {
            emit[3 * j + 2][3] = *c;
        }
        if emit.is_empty() {
            emit.extend([[0.0; 4]; 3]);
        }
        for (k, i) in scene.instances.iter().enumerate() {
            inst.push([first_tri[i.mesh], scene.meshes[i.mesh].material as u32, emit_base[k], 0]);
        }
        // Light-group shares: irradiance at a reference distance.
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for i in &scene.instances {
            for p in &scene.meshes[i.mesh].positions {
                let w = apply(&i.xf, *p);
                for a in 0..3 {
                    lo[a] = lo[a].min(w[a]);
                    hi[a] = hi[a].max(w[a]);
                }
            }
        }
        let diag = ((0..3).map(|a| (hi[a] - lo[a]).powi(2)).sum::<f64>()).sqrt();
        let d = (0.1 * diag).clamp(1.0, 10.0);
        let sun_e = scene.sun.map_or(0.0, |s| {
            std::f64::consts::PI * s.radius.sin().powi(2) * lum(s.radiance)
        });
        let env_e = scene.env.as_ref().map_or(0.0, |e| {
            let (w, h) = (e.size.0 as usize, e.size.1 as usize);
            let mut s = 0.0;
            let mut ws = 0.0;
            for j in 0..h {
                let st = (std::f64::consts::PI * (j as f64 + 0.5) / h as f64).sin();
                for i in 0..w {
                    let c = e.rgb[j * w + i];
                    s += lum([c[0] as f64, c[1] as f64, c[2] as f64]) * st;
                    ws += st;
                }
            }
            std::f64::consts::PI * s / ws
        });
        let raw = [sun_e, env_e, power[0] / (d * d), power[1] / (d * d)];
        let present = [scene.sun.is_some(), env[2] != 0, n_emit > 0, n_portal > 0];
        let tot: f64 = raw.iter().sum::<f64>().max(1e-30);
        let mut sh = [0.0f64; 4];
        for k in 0..4 {
            if present[k] {
                sh[k] = (raw[k] / tot).max(0.05);
            }
        }
        let st: f64 = sh.iter().sum::<f64>().max(1e-30);
        let shares = sh.map(|x| (x / st) as f32);
        let picks = [
            shares[0],
            shares[0] + shares[1],
            shares[0] + shares[1] + shares[2],
            1.0,
        ];
        let sun = scene.sun.map_or(([0.0; 4], [0.0; 4]), |s| {
            let omega = std::f64::consts::TAU * (1.0 - s.radius.cos());
            ([s.dir[0] as f32, s.dir[1] as f32, s.dir[2] as f32, s.radius.cos() as f32], h4(s.radiance, omega))
        });
        // Textures.
        let mut texels: Vec<u32> = Vec::new();
        let mut textab: Vec<[u32; 4]> = Vec::new();
        for t in &scene.textures {
            textab.push([texels.len() as u32, t.size.0, t.size.1, 0]);
            texels.extend_from_slice(&t.texels);
        }
        if texels.is_empty() {
            texels.push(0);
            textab.push([0, 1, 1, 0]);
        }
        // Tables.
        let tb = Tables::get();
        let mut tabs: Vec<f32> = tb.diel.iter().map(|&x| x as f32).collect();
        assert_eq!(tabs.len(), TN * TN * NE);
        for k in 0..3 {
            tabs.extend(tb.metal[k].iter().map(|&x| x as f32));
        }
        tabs.extend_from_slice(super::sheen::table());

        // Acceleration structures: one BLAS per mesh.
        let mut blas: Vec<Option<wgpu::Blas>> = Vec::new();
        let mut keep = Vec::new();
        let flags = wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE;
        let mut entries = Vec::new();
        for m in &scene.meshes {
            if m.indices.len() < 3 {
                blas.push(None);
                entries.push(None);
                continue;
            }
            let vb = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("look vertices"),
                contents: bytemuck::cast_slice(&m.positions),
                usage: wgpu::BufferUsages::BLAS_INPUT,
            });
            let ib = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("look indices"),
                contents: bytemuck::cast_slice(&m.indices),
                usage: wgpu::BufferUsages::BLAS_INPUT,
            });
            let opaque = !scene.materials[m.material].non_opaque();
            let size = wgpu::BlasTriangleGeometrySizeDescriptor {
                vertex_format: wgpu::VertexFormat::Float32x3,
                vertex_count: m.positions.len() as u32,
                index_format: Some(wgpu::IndexFormat::Uint32),
                index_count: Some(m.indices.len() as u32),
                flags: if opaque {
                    wgpu::AccelerationStructureGeometryFlags::OPAQUE
                } else {
                    wgpu::AccelerationStructureGeometryFlags::empty()
                },
            };
            let bl = dev.create_blas(
                &wgpu::CreateBlasDescriptor {
                    label: Some("look blas"),
                    flags,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                },
                wgpu::BlasGeometrySizeDescriptors::Triangles { descriptors: vec![size.clone()] },
            );
            blas.push(Some(bl));
            entries.push(Some((vb, ib, size)));
        }
        for (b, e) in blas.iter().zip(&entries) {
            let (Some(b), Some((vb, ib, size))) = (b, e) else { continue };
            let entry = wgpu::BlasBuildEntry {
                blas: b,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                    size,
                    vertex_buffer: vb,
                    first_vertex: 0,
                    vertex_stride: 12,
                    index_buffer: Some(ib),
                    first_index: Some(0),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                }]),
            };
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.build_acceleration_structures(std::iter::once(&entry), std::iter::empty());
            ctx.queue.submit([enc.finish()]);
        }
        dev.poll(wgpu::PollType::wait_indefinitely()).expect("device lost");
        for e in entries.into_iter().flatten() {
            keep.push(e.0);
            keep.push(e.1);
        }
        let n_inst = scene.instances.len().max(1) as u32;
        assert!(n_inst < (1 << 24), "instance custom data is 24 bits");
        let mut tlas = dev.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("look tlas"),
            max_instances: n_inst,
            flags,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });
        for (k, i) in scene.instances.iter().enumerate() {
            if let Some(b) = &blas[i.mesh] {
                let x = &i.xf;
                let xf = [
                    x[0][0], x[0][1], x[0][2], x[0][3], x[1][0], x[1][1], x[1][2], x[1][3], x[2][0], x[2][1], x[2][2],
                    x[2][3],
                ];
                tlas[k] = Some(wgpu::TlasInstance::new(b, xf, k as u32, i.mask as u8));
            }
        }
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.build_acceleration_structures(std::iter::empty(), [&tlas]);
        ctx.queue.submit([enc.finish()]);
        dev.poll(wgpu::PollType::wait_indefinitely()).expect("device lost");

        let init = |label: &str, bytes: &[u8]| {
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: if bytes.is_empty() { &[0u8; 16] } else { bytes },
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
        };
        if tris.is_empty() {
            tris.extend([[0.0; 4]; 7]);
        }
        if inst.is_empty() {
            inst.push([0; 4]);
        }
        let b_tris = init("look tris", bytemuck::cast_slice(&tris));
        let b_inst = init("look inst", bytemuck::cast_slice(&inst));
        let b_mats = init("look mats", bytemuck::cast_slice(&mats));
        let b_texels = init("look texels", bytemuck::cast_slice(&texels));
        let b_textab = init("look textab", bytemuck::cast_slice(&textab));
        let b_tabs = init("look tables", bytemuck::cast_slice(&tabs));
        let b_emit = init("look emitters", bytemuck::cast_slice(&emit));
        let b_env = init("look env", bytemuck::cast_slice(&envt));
        let (w, h) = scene.size;
        let pixels = w * h;
        let storage = |label: &str, bytes: u64| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let accum = storage("look accum", pixels as u64 * 16);
        let groups_n = pixels.div_ceil(64);
        let spec = storage("look spectrum", (groups_n * BINS) as u64 * 4);
        let probe_io = storage("look probe", 1 << 20);
        let params = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("look params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("look scene"),
            source: wgpu::ShaderSource::Wgsl(format!("enable wgpu_ray_query;\n{SHADER}").into()),
        });
        let layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("look"),
            entries: &(0..13u32)
                .map(|b| wgpu::BindGroupLayoutEntry {
                    binding: b,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: match b {
                        0 => wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        11 => wgpu::BindingType::AccelerationStructure { vertex_return: false },
                        9 | 10 | 12 => wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        _ => wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("look"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mk = |entry: &str| {
            dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("look"),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipe = mk("trace");
        let probe_pipe = mk("probe");
        let bufs: [(u32, &wgpu::Buffer); 12] = [
            (0, &params),
            (1, &b_tris),
            (2, &b_inst),
            (3, &b_mats),
            (4, &b_texels),
            (5, &b_textab),
            (6, &b_tabs),
            (7, &b_emit),
            (8, &b_env),
            (9, &accum),
            (10, &spec),
            (12, &probe_io),
        ];
        let mut ents: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .map(|&(binding, b)| wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() })
            .collect();
        ents.push(wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::AccelerationStructure(&tlas) });
        let group = dev.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("look"), layout: &layout, entries: &ents });
        if let Some(e) = pollster::block_on(scope.pop()) {
            panic!("look gpu: {e}");
        }
        Some(Self {
            ctx,
            pipe,
            probe_pipe,
            group,
            params,
            accum,
            spec,
            probe_io,
            pixels,
            size: scene.size,
            flags: [pixels as u32, n_emit as u32, n_portal as u32, 0],
            sun,
            picks,
            env,
            emit_off: [0, portal_first, 0, 0],
            _keep: [b_tris, b_inst, b_mats, b_texels, b_textab, b_tabs, b_emit, b_env]
                .into_iter()
                .chain(keep)
                .collect(),
            _blas: blas.into_iter().flatten().collect(),
            _tlas: tlas,
            emitters: (n_emit, n_portal),
            shares,
        })
    }

    fn params(&self, cam: &Camera, o: &Opts, sample: u32, first: u32) -> Params {
        let band = (irys_light::spectrum::LAMBDA_MAX_NM - irys_light::spectrum::LAMBDA_MIN_NM) as f32;
        let norm = irys_light::spectrum::RESPONSE_NORM;
        let per = band / 4.0;
        let mi = hat_inverse();
        let v = |a: [f64; 3], w: f32| [a[0] as f32, a[1] as f32, a[2] as f32, w];
        let mut flags = self.flags;
        flags[3] = o.rr_from;
        Params {
            size: [self.size.0 as u32, self.size.1 as u32, o.debug, o.max_bounces],
            seed: [o.seed as u32, (o.seed >> 32) as u32, sample, first],
            flags,
            eye: v(cam.eye, cam.tan_half as f32),
            fwd: v(cam.forward, o.clamp),
            right: v(cam.right, 0.0),
            up: v(cam.up, if o.per_lane { 1.0 } else { 0.0 }),
            resp: [norm[0] as f32 * per, norm[1] as f32 * per, norm[2] as f32 * per, 0.0],
            sun_dir: self.sun.0,
            sun_l: self.sun.1,
            picks: self.picks,
            env: self.env,
            minv0: [mi[0][0], mi[0][1], mi[0][2], 0.0],
            minv1: [mi[1][0], mi[1][1], mi[1][2], 0.0],
            minv2: [mi[2][0], mi[2][1], mi[2][2], 0.0],
            emit_off: self.emit_off,
        }
    }

    /// Render one frame from `cam`.
    pub fn render(&self, cam: &Camera, o: &Opts) -> Frame {
        let (dev, queue) = (&self.ctx.device, &self.ctx.queue);
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.clear_buffer(&self.accum, 0, None);
        enc.clear_buffer(&self.spec, 0, None);
        queue.submit([enc.finish()]);
        dev.poll(wgpu::PollType::wait_indefinitely()).expect("device lost");
        let t0 = std::time::Instant::now();
        for s in 0..o.spp {
            for first in (0..self.pixels).step_by(CHUNK) {
                let n = (self.pixels - first).min(CHUNK);
                let p = self.params(cam, o, s, first as u32);
                queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
                let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                    pass.set_pipeline(&self.pipe);
                    pass.set_bind_group(0, &self.group, &[]);
                    pass.dispatch_workgroups(n.div_ceil(64) as u32, 1, 1);
                }
                queue.submit([enc.finish()]);
            }
            dev.poll(wgpu::PollType::wait_indefinitely()).expect("device lost");
        }
        let seconds = t0.elapsed().as_secs_f64();
        let acc = self.read(&self.accum, self.pixels * 4);
        let inv = 1.0 / o.spp.max(1) as f32;
        let rgb = acc.chunks_exact(4).flat_map(|p| [p[0] * inv, p[1] * inv, p[2] * inv]).collect();
        let groups = self.pixels.div_ceil(64);
        let sp = self.read(&self.spec, groups * BINS);
        let mut spectrum = vec![0.0f64; BINS];
        for g in 0..groups {
            for b in 0..BINS {
                spectrum[b] += sp[g * BINS + b] as f64;
            }
        }
        // Σ L over the lanes in a bin, over (lanes expected in a bin).
        let per_bin = (self.pixels as f64 * o.spp as f64 * 4.0) / BINS as f64;
        for s in &mut spectrum {
            *s /= per_bin;
        }
        Frame { rgb, spectrum, seconds }
    }

    /// Evaluate material lanes on the device: per lane `(material, λ,
    /// inside, ω_o, ω_i, u)` → `([f, pdf, w0, w1], [w2, w3, w4, thin pass],
    /// [sampled ω_i, lobe])`.
    pub fn probe(&self, lanes: &[(u32, f32, bool, [f32; 3], [f32; 3], [f32; 3])]) -> Vec<[[f32; 4]; 3]> {
        let (dev, queue) = (&self.ctx.device, &self.ctx.queue);
        assert!(1 + lanes.len() * 7 <= (1 << 20) / 16, "too many probe lanes");
        let mut io: Vec<[f32; 4]> = vec![[f32::from_bits(lanes.len() as u32), 0.0, 0.0, 0.0]];
        for l in lanes {
            io.push([f32::from_bits(l.0), l.1, if l.2 { 1.0 } else { 0.0 }, 0.0]);
            io.push([l.3[0], l.3[1], l.3[2], 0.0]);
            io.push([l.4[0], l.4[1], l.4[2], 0.0]);
            io.push([l.5[0], l.5[1], l.5[2], 0.0]);
            io.extend([[0.0; 4]; 3]);
        }
        queue.write_buffer(&self.probe_io, 0, bytemuck::cast_slice(&io));
        let cam = Camera { eye: [0.0; 3], forward: [0.0, 1.0, 0.0], right: [1.0, 0.0, 0.0], up: [0.0, 0.0, 1.0], tan_half: 1.0 };
        let p = self.params(&cam, &Opts::default(), 0, 0);
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.probe_pipe);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(lanes.len().div_ceil(64).max(1) as u32, 1, 1);
        }
        queue.submit([enc.finish()]);
        let out = self.read(&self.probe_io, io.len() * 4);
        (0..lanes.len())
            .map(|i| {
                let b = (1 + i * 7 + 4) * 4;
                std::array::from_fn(|k| std::array::from_fn(|c| out[b + k * 4 + c]))
            })
            .collect()
    }

    fn read(&self, src: &wgpu::Buffer, n: usize) -> Vec<f32> {
        let dev = &self.ctx.device;
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("look readback"),
            size: (n as u64 * 4).max(16),
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
            dev.poll(wgpu::PollType::wait_indefinitely()).expect("device lost");
            match rx.try_recv() {
                Ok(r) => break r.expect("map failed"),
                Err(std::sync::mpsc::TryRecvError::Empty) => std::thread::yield_now(),
                Err(e) => panic!("map: {e}"),
            }
        }
        let out = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range().expect("mapped"))[..n].to_vec();
        staging.unmap();
        out
    }
}
