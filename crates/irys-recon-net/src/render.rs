//! Rendering a scene three ways, all through irys:
//!
//! - [`photo`] (feature `gpu`): the synthetic photograph, path traced by
//!   irys's OpenPBR scene kernel on the device, then a camera (defocus,
//!   exposure, Poisson–Gaussian noise, tone, JPEG);
//! - [`Caster`]: primary rays through an irys TLAS of the posed meshes —
//!   instance ids, metric depth and normals per pixel, for the dense labels;
//! - [`Compare`]: the same rays with a linear appearance (per object and
//!   channel, `c₀ + c·n`; the table a constant), the model image the MDL
//!   residual and the renderer-as-teacher loss are measured on.

use std::sync::Arc;

use irys_core::{Bvh, Instance, Point3, Ray, Tlas, TriMesh, Vec3};

use crate::scene::{Cam, Object, Scene, Tris, mesh};

/// Per-pixel labels.
#[derive(Debug, Clone)]
pub struct Labels {
    pub size: (usize, usize),
    /// Object index, or `u16::MAX` for the table.
    pub inst: Vec<u16>,
    /// Metric depth along the camera axis, mm.
    pub depth: Vec<f32>,
    /// Shading normal (world), facing the camera.
    pub normal: Vec<[f32; 3]>,
}

pub const TABLE: u16 = u16::MAX;
/// What a camera ray sees past the plane's horizon (the model image's
/// backdrop; labels fold it into [`TABLE`]).
pub const BACKDROP: u16 = u16::MAX - 1;

/// Primary rays through an irys TLAS.
pub struct Caster {
    tlas: Tlas<TriMesh>,
}

/// The BLAS of a world mesh (`None` when empty).
pub fn blas(m: &Tris) -> Option<Arc<Bvh<TriMesh>>> {
    (!m.t.is_empty()).then(|| Arc::new(Bvh::build(trimesh(m))))
}

pub fn trimesh(m: &Tris) -> TriMesh {
    let p: Vec<Point3> = m.p.iter().map(|q| Point3::new(q[0], q[1], q[2])).collect();
    let idx: Vec<u32> = m.t.iter().flatten().copied().collect();
    TriMesh::new(p, Vec::new(), &idx)
}

impl Caster {
    /// From world-space meshes; payload `i` is mesh `i`.
    pub fn new(meshes: &[Tris]) -> Self {
        let inst = meshes
            .iter()
            .enumerate()
            .filter(|(_, m)| !m.t.is_empty())
            .filter_map(|(i, m)| Instance::identity(Arc::new(Bvh::build(trimesh(m))), i))
            .collect();
        Self { tlas: Tlas::build(inst) }
    }

    /// From prebuilt BLASes (`None`: absent); payload `i` is entry `i`.
    pub fn from_blas(blas: &[Option<Arc<Bvh<TriMesh>>>]) -> Self {
        let inst = blas.iter().enumerate().filter_map(|(i, b)| b.as_ref().and_then(|b| Instance::identity(b.clone(), i))).collect();
        Self { tlas: Tlas::build(inst) }
    }

    /// `(object or TABLE, depth mm, normal)` along the ray through `(x, y)`.
    pub fn cast(&self, cam: &Cam, x: f64, y: f64) -> (u16, f64, [f64; 3]) {
        let d = cam.dir(x, y);
        let (f, ..) = cam.frame();
        let along = d[0] * f[0] + d[1] * f[1] + d[2] * f[2];
        let ray = Ray::new(Point3::new(cam.eye[0], cam.eye[1], cam.eye[2]), Vec3::new(d[0], d[1], d[2]));
        let t_plane = if d[2] < -1e-9 { -cam.eye[2] / d[2] } else { f64::INFINITY };
        if let Some(h) = self.tlas.trace_closest(&ray) {
            if h.hit.t < t_plane {
                let n = h.hit.normal;
                let mut n = [n.x, n.y, n.z];
                if n[0] * d[0] + n[1] * d[1] + n[2] * d[2] > 0.0 {
                    n = [-n[0], -n[1], -n[2]];
                }
                return (h.payload as u16, h.hit.t * along, n);
            }
        }
        if t_plane.is_finite() { (TABLE, t_plane * along, [0.0, 0.0, 1.0]) } else { (BACKDROP, 1e5, [0.0, 0.0, 1.0]) }
    }

    pub fn labels(&self, cam: &Cam) -> Labels {
        use rayon::prelude::*;
        let (w, h) = cam.size;
        let px: Vec<(u16, f64, [f64; 3])> = (0..w * h).into_par_iter().map(|i| self.cast(cam, (i % w) as f64 + 0.5, (i / w) as f64 + 0.5)).collect();
        Labels {
            size: (w, h),
            inst: px.iter().map(|p| if p.0 == BACKDROP { TABLE } else { p.0 }).collect(),
            depth: px.iter().map(|p| p.1 as f32).collect(),
            normal: px.iter().map(|p| [p.2[0] as f32, p.2[1] as f32, p.2[2] as f32]).collect(),
        }
    }
}

/// Every object's world mesh (a plate that does not build is empty).
pub fn world_meshes(objects: &[Object]) -> Vec<Tris> {
    use rayon::prelude::*;
    objects.par_iter().map(|o| mesh(o).map(|m| m.posed(o.pose)).unwrap_or_default()).collect()
}

/// The model image: linear appearance on the cast geometry.
pub struct Compare {
    pub size: (usize, usize),
    /// Supersampling per axis.
    pub ss: usize,
}

/// One frame of the model: per sub-sample id and normal.
pub struct Cast {
    pub size: (usize, usize),
    pub ss: usize,
    pub id: Vec<u16>,
    pub n: Vec<[f64; 3]>,
}

impl Compare {
    pub fn cast(&self, caster: &Caster, cam: &Cam, window: Option<(usize, usize, usize, usize)>) -> Cast {
        use rayon::prelude::*;
        let (w, h) = self.size;
        let s = self.ss;
        let cam = cam.resized(self.size);
        let (x0, y0, x1, y1) = window.unwrap_or((0, 0, w, h));
        let n = (x1 - x0) * (y1 - y0) * s * s;
        let cw = (x1 - x0) * s;
        let out: Vec<(u16, [f64; 3])> = (0..n)
            .into_par_iter()
            .map(|k| {
                let (sx, sy) = (k % cw, k / cw);
                let x = x0 as f64 + (sx as f64 + 0.5) / s as f64;
                let y = y0 as f64 + (sy as f64 + 0.5) / s as f64;
                let (id, _, nn) = caster.cast(&cam, x, y);
                // The background's appearance is linear in the image
                // coordinates (vignetting, light falloff, a wall's shading).
                if id == TABLE || id == BACKDROP {
                    return (id, [2.0 * x / w as f64 - 1.0, 2.0 * y / h as f64 - 1.0, 0.0]);
                }
                (id, nn)
            })
            .collect();
        Cast { size: (x1 - x0, y1 - y0), ss: s, id: out.iter().map(|o| o.0).collect(), n: out.iter().map(|o| o.1).collect() }
    }
}

/// Fit the linear appearance of `cast` to `target` (linear RGB, the cast's
/// window) and return the model image and the per-pixel residual.
///
/// Every object gets `c₀ + c·n` per channel (four unknowns, least squares
/// over its sub-samples), the table and the backdrop `c₀ + c·(x, y)` in the
/// image: the shading
/// that any albedo under any low-frequency light makes, so what is left in
/// the residual is geometry.
pub fn appearance(cast: &Cast, target: &[[f64; 3]]) -> Vec<[f64; 3]> {
    use std::collections::HashMap;
    let (w, h) = cast.size;
    let s = cast.ss;
    let ss = s * s;
    // Normal equations per id.
    let mut acc: HashMap<u16, ([[f64; 4]; 4], [[f64; 4]; 3])> = HashMap::new();
    for y in 0..h * s {
        for x in 0..w * s {
            let k = y * w * s + x;
            let id = cast.id[k];
            let t = target[(y / s) * w + x / s];
            let n = cast.n[k];
            let b = [1.0, n[0], n[1], n[2]];
            let e = acc.entry(id).or_insert(([[0.0; 4]; 4], [[0.0; 4]; 3]));
            for i in 0..4 {
                for j in 0..4 {
                    e.0[i][j] += b[i] * b[j];
                }
                for c in 0..3 {
                    e.1[c][i] += b[i] * t[c];
                }
            }
        }
    }
    let mut coef: HashMap<u16, [[f64; 4]; 3]> = HashMap::new();
    for (id, (mut a, rhs)) in acc {
        for i in 0..4 {
            a[i][i] += 1e-3 * (a[0][0] + 1.0) / 4.0 + 1e-9;
        }
        let mut out = [[0.0; 4]; 3];
        for c in 0..3 {
            out[c] = solve4(a, rhs[c]);
        }
        coef.insert(id, out);
    }
    let mut img = vec![[0.0; 3]; w * h];
    for y in 0..h * s {
        for x in 0..w * s {
            let k = y * w * s + x;
            let id = cast.id[k];
            let n = cast.n[k];
            let b = [1.0, n[0], n[1], n[2]];
            let c = coef[&id];
            let p = &mut img[(y / s) * w + x / s];
            for ch in 0..3 {
                p[ch] += (c[ch][0] * b[0] + c[ch][1] * b[1] + c[ch][2] * b[2] + c[ch][3] * b[3]) / ss as f64;
            }
        }
    }
    img
}

fn solve4(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> [f64; 4] {
    for i in 0..4 {
        let mut piv = i;
        for r in i + 1..4 {
            if a[r][i].abs() > a[piv][i].abs() {
                piv = r;
            }
        }
        a.swap(i, piv);
        b.swap(i, piv);
        let d = a[i][i];
        if d.abs() < 1e-12 {
            continue;
        }
        for r in 0..4 {
            if r != i {
                let f = a[r][i] / d;
                for c in i..4 {
                    a[r][c] -= f * a[i][c];
                }
                b[r] -= f * b[i];
            }
        }
    }
    let mut x = [0.0; 4];
    for i in 0..4 {
        x[i] = if a[i][i].abs() < 1e-12 { 0.0 } else { b[i] / a[i][i] };
    }
    x
}

/// sRGB 8-bit to linear.
pub fn lin(c: u8) -> f64 {
    irys_recon::look::srgb_to_linear(c as f64 / 255.0)
}

/// Box-downsample a linear RGB image by an integer factor.
pub fn downsample(img: &[[f64; 3]], size: (usize, usize), k: usize) -> Vec<[f64; 3]> {
    let (w, h) = (size.0 / k, size.1 / k);
    let mut out = vec![[0.0; 3]; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut s = [0.0; 3];
            for j in 0..k {
                for i in 0..k {
                    let p = img[(y * k + j) * size.0 + x * k + i];
                    for c in 0..3 {
                        s[c] += p[c];
                    }
                }
            }
            out[y * w + x] = [s[0] / (k * k) as f64, s[1] / (k * k) as f64, s[2] / (k * k) as f64];
        }
    }
    out
}

/// Resample an 8-bit RGB image to `size` (area average), returning linear RGB.
pub fn resample_rgb8(src: &image::RgbImage, size: (usize, usize)) -> Vec<[f64; 3]> {
    let (sw, sh) = (src.width() as usize, src.height() as usize);
    let (w, h) = size;
    let mut out = vec![[0.0; 3]; w * h];
    for y in 0..h {
        let ya = y * sh / h;
        let yb = ((y + 1) * sh / h).max(ya + 1);
        for x in 0..w {
            let xa = x * sw / w;
            let xb = ((x + 1) * sw / w).max(xa + 1);
            let mut s = [0.0; 3];
            let mut n = 0.0;
            for yy in ya..yb {
                for xx in xa..xb {
                    let p = src.get_pixel(xx as u32, yy as u32);
                    for c in 0..3 {
                        s[c] += lin(p[c]);
                    }
                    n += 1.0;
                }
            }
            out[y * w + x] = [s[0] / n, s[1] / n, s[2] / n];
        }
    }
    out
}

#[cfg(feature = "gpu")]
pub use gpu::*;

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use irys_diff::look::openpbr::OpenPbr;
    use irys_diff::look::scene::{Camera as LCam, Equirect, Instance as LInst, LookScene, Material, Mesh, Sun, mask};
    use irys_diff::look::scene_gpu::{GpuLook, Opts};
    use irys_recon::jac::Mix;

    use crate::scene::{Lighting, Mat};

    fn env(l: &Lighting) -> Equirect {
        let (w, h) = (96u32, 48u32);
        let mut rgb = Vec::with_capacity((w * h) as usize);
        for j in 0..h {
            for i in 0..w {
                let theta = (j as f64 + 0.5) / h as f64 * std::f64::consts::PI;
                let phi = (i as f64 + 0.5) / w as f64 * std::f64::consts::TAU;
                let el = 90.0 - theta.to_degrees();
                // atan2(−x, −y) = φ (Blender's equirect).
                let az = (phi.to_degrees() + 360.0) % 360.0;
                let mut c = if el < 0.0 {
                    [0.25 * l.horizon[0], 0.25 * l.horizon[1], 0.25 * l.horizon[2]]
                } else {
                    let s = (el.to_radians()).sin();
                    [l.horizon[0] + (l.zenith[0] - l.horizon[0]) * s, l.horizon[1] + (l.zenith[1] - l.horizon[1]) * s, l.horizon[2] + (l.zenith[2] - l.horizon[2]) * s]
                };
                let daz = ((az - l.window[0] + 540.0) % 360.0 - 180.0).abs();
                if daz < l.window[2] && (el - l.window[1]).abs() < l.window[2] * 0.7 {
                    for v in &mut c {
                        *v += l.window[3];
                    }
                }
                rgb.push([c[0] as f32, c[1] as f32, c[2] as f32]);
            }
        }
        Equirect { size: (w, h), rgb }
    }

    fn material(m: &Mat, name: &str) -> Material {
        Material::new(
            name,
            OpenPbr {
                base_color: m.rgb,
                base_metalness: if m.metal { 1.0 } else { 0.0 },
                specular_roughness: m.rough,
                specular_ior: 1.5,
                ..OpenPbr::default()
            },
        )
    }

    /// The scene as irys's OpenPBR kernel wants it, at `size`.
    pub fn look_scene(s: &Scene, meshes: &[Tris], size: (usize, usize)) -> LookScene {
        let mut lm = Vec::new();
        let mut inst = Vec::new();
        let mut mats = Vec::new();
        let id: [[f32; 4]; 3] = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
        for (i, (o, m)) in s.objects.iter().zip(meshes).enumerate() {
            if m.t.is_empty() {
                continue;
            }
            mats.push(material(&o.mat, &format!("obj{i}")));
            lm.push(Mesh {
                positions: m.p.iter().map(|q| [q[0] as f32, q[1] as f32, q[2] as f32]).collect(),
                normals: Vec::new(),
                uvs: Vec::new(),
                indices: m.t.iter().flatten().copied().collect(),
                material: mats.len() - 1,
            });
            inst.push(LInst { mesh: lm.len() - 1, xf: id, mask: mask::GEOMETRY });
        }
        mats.push(material(&s.table, "table"));
        let half = 20000.0f32;
        lm.push(Mesh {
            positions: vec![[-half, -half, 0.0], [half, -half, 0.0], [half, half, 0.0], [-half, half, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 4],
            uvs: Vec::new(),
            indices: vec![0, 1, 2, 0, 2, 3],
            material: mats.len() - 1,
        });
        inst.push(LInst { mesh: lm.len() - 1, xf: id, mask: mask::GEOMETRY });
        let e = env(&s.light);
        let mean_env: f64 = e.rgb.iter().map(|c| (c[0] + c[1] + c[2]) as f64 / 3.0).sum::<f64>() / e.rgb.len() as f64;
        let (az, el) = (s.light.key[0].to_radians(), s.light.key[1].to_radians());
        let kr = s.light.key_radius.to_radians();
        let ke = mean_env * std::f64::consts::PI * s.light.key_radiance;
        let krad = ke / (std::f64::consts::PI * kr.sin().powi(2));
        let cam = s.cam.resized(size);
        let (f, r, u, ty) = cam.frame();
        LookScene {
            meshes: lm,
            instances: inst,
            materials: mats,
            textures: Vec::new(),
            env: Some(e),
            sun: Some(Sun { dir: [el.cos() * az.cos(), el.cos() * az.sin(), el.sin()], radius: kr, radiance: [krad, krad * 0.97, krad * 0.93] }),
            portal_skies: Vec::new(),
            camera: LCam { eye: cam.eye, forward: f, right: r, up: u, tan_half: ty },
            size,
            frames: Vec::new(),
            fps: 24.0,
            dropped: Vec::new(),
        }
    }

    /// A clean linear render (no camera post) at `size`.
    pub fn render_linear(ctx: &'static irys_gpu::GpuContext, s: &Scene, meshes: &[Tris], size: (usize, usize), spp: u32) -> Result<Vec<[f64; 3]>, String> {
        let ls = look_scene(s, meshes, size);
        let g = GpuLook::new(ctx, &ls).ok_or("no ray queries on this device")?;
        let fr = g.render(&ls.camera, &Opts { spp, max_bounces: 3, clamp: 6.0, seed: s.seed + 1, ..Opts::default() });
        Ok(fr.rgb.chunks_exact(3).map(|c| [c[0] as f64, c[1] as f64, c[2] as f64]).map(|c| c.map(|v| if v.is_finite() { v } else { 0.0 })).collect())
    }

    /// The synthetic photograph: path traced at `2 × out` and `spp`, then the
    /// camera; 8-bit sRGB at `out`, as a JPEG would decode.
    pub fn photo(ctx: &'static irys_gpu::GpuContext, s: &Scene, meshes: &[Tris], out: (usize, usize), spp: u32) -> Result<image::RgbImage, String> {
        let big = (out.0 * 2, out.1 * 2);
        let ls = look_scene(s, meshes, big);
        let t0 = std::time::Instant::now();
        let g = GpuLook::new(ctx, &ls).ok_or("no ray queries on this device")?;
        let t1 = std::time::Instant::now();
        let fr = g.render(&ls.camera, &Opts { spp, max_bounces: 3, clamp: 6.0, seed: s.seed + 1, ..Opts::default() });
        if std::env::var("NEURAL_TIME").is_ok() {
            eprintln!("gpulook new {:.3}s render {:.3}s", (t1 - t0).as_secs_f64(), t1.elapsed().as_secs_f64());
        }
        let mut lin: Vec<f64> = fr.rgb.iter().map(|&x| if x.is_finite() { x as f64 } else { 0.0 }).collect();
        // Defocus is a property of the lens: the same angular blur at any size.
        lin = irys_recon::fit::blur(&lin, big, 3, s.post[0] * 2.0 * out.0 as f64 / 128.0);
        let px: Vec<[f64; 3]> = lin.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        let small = downsample(&px, big, 2);
        let mut lum: Vec<f64> = small.iter().map(|c| (c[0] + c[1] + c[2]) / 3.0).collect();
        lum.sort_by(f64::total_cmp);
        let expo = 0.85 / lum[lum.len() * 98 / 100].max(1e-6);
        let mut noise = Mix(s.seed ^ 0xa11ce);
        let mut gauss = move || {
            let (u1, u2) = (noise.uniform().max(1e-12), noise.uniform());
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        };
        let mut bytes = Vec::with_capacity(out.0 * out.1 * 3);
        for p in &small {
            for c in 0..3 {
                let v = p[c] * expo;
                let v = v + s.post[3] * (0.0006 * v.max(0.0) + 4e-6).sqrt() * gauss();
                let sv = irys_recon::render::srgb8(v) as f64 / 255.0;
                bytes.push((sv.powf(s.post[1]) * 255.0 + 0.5).clamp(0.0, 255.0) as u8);
            }
        }
        let img = image::RgbImage::from_raw(out.0 as u32, out.1 as u32, bytes).unwrap();
        let mut buf = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, s.post[2] as u8).encode_image(&img).map_err(|e| e.to_string())?;
        let dec = image::load_from_memory(&buf).map_err(|e| e.to_string())?.to_rgb8();
        Ok(dec)
    }
}
