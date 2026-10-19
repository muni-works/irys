//! M1: a 3DGS cloud ([`PlyCloud`]) on the device, for the scene kernel's
//! TLAS (`splat.wgsl`): per-splat records, half-float SH, and an AABB BLAS
//! of the 3σ boxes (3DGRT's proxies).
//!
//! [`composite_cpu`] is the f64 twin of `splat_segment`: the same response,
//! cut-offs, (t, index) order and early-out, by brute force over the cloud.

use irys_gpu::GpuContext;
use irys_gpu::wgpu::{self, util::DeviceExt};
pub use irys_light::PlyCloud;

/// Composite rounds per segment (K = 16 splats each): at most 2048 splats
/// along one ray before the walk gives up with what it has.
pub const ROUNDS: u32 = 128;
/// The kernel's buffer depth.
pub const K: usize = 16;

/// Rotation matrix of a unit `[w, x, y, z]`.
fn rot(q: [f32; 4]) -> [[f64; 3]; 3] {
    let [w, x, y, z] = q.map(|v| v as f64);
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

fn scale_ok(s: f32) -> f64 {
    if s.is_finite() && s > 0.0 { s as f64 } else { 1e-7 }
}

/// `M = S⁻¹ Rᵀ` (rows) and the half-extents of splat `i`'s proxy box: the
/// ellipsoid where its response reaches 1/255, `k σ` with
/// `k = √(2 ln 255 o)` (3DGRT's opacity-aware bound; zero for a splat that
/// can never reach 1/255). The rasteriser cuts at the same 1/255, not at
/// 3σ: a hard 3σ cut shows as facets on large opaque splats.
pub fn whiten(c: &PlyCloud, i: usize) -> ([[f64; 3]; 3], [f64; 3]) {
    let r = rot(c.quats[i]);
    let s = c.scales[i].map(scale_ok);
    let m = std::array::from_fn(|k| std::array::from_fn(|a| r[a][k] / s[k]));
    let o = c.opacities[i] as f64;
    let k = if o * 255.0 > 1.0 { (2.0 * (255.0 * o).ln()).sqrt() } else { 0.0 };
    let ext = std::array::from_fn(|a| k * (0..3).map(|j| (r[a][j] * s[j]).powi(2)).sum::<f64>().sqrt());
    (m, ext)
}

/// World margin (m) added to every proxy box so the trainer's screen-space
/// dilation fits inside it: 0.3 px² at 200 px focal length reaches
/// √(2 ln 255) · √0.3 · t / 200 ≈ 2 cm at 7 m. Beyond that a tiny splat's
/// dilated footprint is clipped to its box (deterministically, on CPU and
/// device alike).
pub const DILATE_PAD: f64 = 0.02;

/// `A = Σ⁻¹` and `Σ` of splat `i`, `[00, 01, 02, 11, 12, 22]` each.
pub fn covariances(c: &PlyCloud, i: usize) -> ([f64; 6], [f64; 6]) {
    let (m, _) = whiten(c, i);
    let s = c.scales[i].map(scale_ok);
    // A = MᵀM; Σ = R S² Rᵀ with R[a][k] = m[k][a] s_k
    let a = |p: usize, q: usize| (0..3).map(|k| m[k][p] * m[k][q]).sum::<f64>();
    let sg = |p: usize, q: usize| (0..3).map(|k| m[k][p] * s[k] * s[k] * m[k][q] * s[k] * s[k]).sum::<f64>();
    let idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];
    (idx.map(|(p, q)| a(p, q)), idx.map(|(p, q)| sg(p, q)))
}

/// The proxy box half-extents the device uses (f32, padded).
pub fn box_extent(c: &PlyCloud, i: usize) -> [f32; 3] {
    let (_, e) = whiten(c, i);
    let p = c.positions[i];
    std::array::from_fn(|a| (e[a] * (1.0 + 1e-3) + 1e-6 * p[a].abs() as f64 + DILATE_PAD) as f32)
}

/// The device's form of a cloud.
pub struct Packed {
    /// 5 vec4 per splat: `(μ, opacity)`, `A` and `Σ` (12 floats), the box.
    pub rec: Vec<[f32; 4]>,
    /// SH as f16 pairs, `words` u32 per splat.
    pub sh: Vec<u32>,
    /// `[lo, hi]` per splat.
    pub boxes: Vec<[f32; 6]>,
    /// u32 words of SH per splat.
    pub words: u32,
}

/// Pack `c` for `splat.wgsl`.
pub fn pack(c: &PlyCloud) -> Packed {
    let n = c.len();
    let stride = c.stride();
    let words = (3 * stride).div_ceil(2);
    let mut rec = Vec::with_capacity(5 * n);
    let mut boxes = Vec::with_capacity(n);
    let mut sh = vec![0u32; n * words];
    for i in 0..n {
        let (a, sg) = covariances(c, i);
        let e = box_extent(c, i);
        let p = c.positions[i];
        let f = |x: f64| x as f32;
        rec.push([p[0], p[1], p[2], c.opacities[i]]);
        rec.push([f(a[0]), f(a[1]), f(a[2]), f(a[3])]);
        rec.push([f(a[4]), f(a[5]), f(sg[0]), f(sg[1])]);
        rec.push([f(sg[2]), f(sg[3]), f(sg[4]), f(sg[5])]);
        rec.push([e[0], e[1], e[2], 0.0]);
        boxes.push(std::array::from_fn(|k| {
            let a = k % 3;
            if k < 3 { p[a] - e[a] } else { p[a] + e[a] }
        }));
        for k in 0..stride {
            for ch in 0..3 {
                let x = 3 * k + ch;
                let h = f16_bits(c.sh[i * stride + k][ch]) as u32;
                sh[i * words + x / 2] |= h << (16 * (x & 1));
            }
        }
    }
    Packed { rec, sh, boxes, words: words as u32 }
}

/// f32 → IEEE half bits, round to nearest even (WGSL's `unpack2x16float` reads them).
pub fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 0xff {
        return sign | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let e2 = e - 127 + 15;
    if e2 >= 0x1f {
        return sign | 0x7c00;
    }
    if e2 <= 0 {
        if e2 < -10 {
            return sign;
        }
        let m = m | 0x80_0000;
        let shift = (14 - e2) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let r = if rem > mid || (rem == mid && half & 1 == 1) { half + 1 } else { half };
        return sign | r as u16;
    }
    let half = ((e2 as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    let r = if rem > 0x1000 || (rem == 0x1000 && half & 1 == 1) { half + 1 } else { half };
    sign | r as u16
}

/// Half bits → f32.
pub fn f16_value(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    match e {
        0 => s * m * 2f32.powi(-24),
        0x1f => if m == 0.0 { s * f32::INFINITY } else { f32::NAN },
        _ => s * (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

/// The device's buffers and BLAS for one cloud.
pub struct GpuSplats {
    /// Records (binding 60).
    pub rec: wgpu::Buffer,
    /// SH (binding 61).
    pub sh: wgpu::Buffer,
    /// `SplatParams` (binding 62).
    pub params: wgpu::Buffer,
    /// The AABB BLAS, one primitive per splat (absent for an empty cloud).
    pub blas: Option<wgpu::Blas>,
    _boxes: Option<wgpu::Buffer>,
    /// Splats.
    pub count: usize,
}

impl GpuSplats {
    /// Upload `c` and build its BLAS (waited on: Metal does not order an
    /// acceleration-structure build before the TLAS build that reads it).
    /// `None` is the empty cloud: dummy buffers, no BLAS.
    pub fn new(ctx: &GpuContext, c: Option<&PlyCloud>) -> Self {
        let dev = &ctx.device;
        let empty = PlyCloud::default();
        let c = c.unwrap_or(&empty);
        let p = pack(c);
        let init = |label: &str, bytes: &[u8], usage| {
            dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: if bytes.is_empty() { &[0u8; 16] } else { bytes },
                usage,
            })
        };
        let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let rec = init("splat records", bytemuck::cast_slice(&p.rec), st);
        let sh = init("splat sh", bytemuck::cast_slice(&p.sh), st);
        let info = [c.len() as u32, c.degree, p.words, ROUNDS];
        let params = init("splat params", bytemuck::cast_slice(&info), wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let (blas, boxes) = if c.is_empty() {
            (None, None)
        } else {
            let ab = init("splat aabbs", bytemuck::cast_slice(&p.boxes), wgpu::BufferUsages::BLAS_INPUT);
            let size = wgpu::BlasAABBGeometrySizeDescriptor {
                primitive_count: p.boxes.len() as u32,
                flags: wgpu::AccelerationStructureGeometryFlags::empty(),
            };
            let bl = dev.create_blas(
                &wgpu::CreateBlasDescriptor {
                    label: Some("splat blas"),
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                },
                wgpu::BlasGeometrySizeDescriptors::AABBs { descriptors: vec![size.clone()] },
            );
            let entry = wgpu::BlasBuildEntry {
                blas: &bl,
                geometry: wgpu::BlasGeometries::AabbGeometries(vec![wgpu::BlasAabbGeometry {
                    size: &size,
                    stride: 24,
                    aabb_buffer: &ab,
                    primitive_offset: 0,
                }]),
            };
            let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            enc.build_acceleration_structures(std::iter::once(&entry), std::iter::empty());
            ctx.queue.submit([enc.finish()]);
            dev.poll(wgpu::PollType::wait_indefinitely()).expect("device lost");
            (Some(bl), Some(ab))
        };
        Self { rec, sh, params, blas, _boxes: boxes, count: c.len() }
    }
}

/// What a composite returns (`SplatSeg`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment {
    /// `Σ T α c`, sRGB-valued.
    pub rgb: [f64; 3],
    /// Transmittance after the segment.
    pub trans: f64,
    /// `t` where `T` first fell to ≤ ½ (`None`: never).
    pub t_half: Option<f64>,
}

/// `(t*, α)` of splat `i` along `o + t d` (`d` unit; α = 0: no response),
/// f64: `splat_eval`, dilation (`pix`, radians per pixel) and box included.
pub fn eval_cpu(c: &PlyCloud, i: usize, o: [f64; 3], d: [f64; 3], pix: f64) -> (f64, f64) {
    let (a, sg) = covariances(c, i);
    let ext = box_extent(c, i);
    let p = c.positions[i];
    let del = [o[0] - p[0] as f64, o[1] - p[1] as f64, o[2] - p[2] as f64];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mul = |m: [f64; 6], v: [f64; 3]| {
        [m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[1] * v[0] + m[3] * v[1] + m[4] * v[2], m[2] * v[0] + m[4] * v[1] + m[5] * v[2]]
    };
    let ad = mul(a, d);
    let dad = dot(d, ad);
    if dad <= 0.0 {
        return (0.0, 0.0);
    }
    let t = -dot(ad, del) / dad;
    let x: [f64; 3] = std::array::from_fn(|k| del[k] + t * d[k]);
    if (0..3).any(|k| x[k].abs() > ext[k] as f64) {
        return (t, 0.0);
    }
    let (e1, e2) = basis(d);
    let (s1, s2) = (mul(sg, e1), mul(sg, e2));
    let dil = 0.3 * (pix * t.max(0.0)).powi(2);
    let (c00, c01, c11) = (dot(e1, s1) + dil, dot(e1, s2), dot(e2, s2) + dil);
    let det = c00 * c11 - c01 * c01;
    if det <= 0.0 {
        return (t, 0.0);
    }
    let (u, v) = (dot(e1, x), dot(e2, x));
    let q = (c11 * u * u - 2.0 * c01 * u * v + c00 * v * v) / det;
    let al = (c.opacities[i] as f64 * (-0.5 * q).exp()).min(0.99);
    if al < 1.0 / 255.0 { (t, 0.0) } else { (t, al) }
}

/// scene.wgsl's `basis_t`, `basis_b` (Duff et al.).
fn basis(n: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let s = if n[2] >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (s + n[2]);
    let b = n[0] * n[1] * a;
    ([1.0 + s * n[0] * n[0] * a, s * b, -s * n[0]], [b, s + n[1] * n[1] * a, -n[1]])
}

/// The 3DGS colour of splat `i` toward `d` (+ ½, clamped), f64, from the
/// half-float SH the device reads.
pub fn colour_cpu(c: &PlyCloud, i: usize, d: [f64; 3]) -> [f64; 3] {
    let s = c.stride();
    let at = |k: usize| c.sh[i * s + k].map(|v| f16_value(f16_bits(v)) as f64);
    let mut col = [0.5f64; 3];
    let mut add = |w: f64, v: [f64; 3]| {
        for ch in 0..3 {
            col[ch] += w * v[ch];
        }
    };
    add(0.28209479177387814, at(0));
    let [x, y, z] = d;
    if c.degree >= 1 {
        let c1 = 0.4886025119029199;
        add(-c1 * y, at(1));
        add(c1 * z, at(2));
        add(-c1 * x, at(3));
        if c.degree >= 2 {
            let (xx, yy, zz) = (x * x, y * y, z * z);
            add(1.0925484305920792 * x * y, at(4));
            add(-1.0925484305920792 * y * z, at(5));
            add(0.31539156525252005 * (2.0 * zz - xx - yy), at(6));
            add(-1.0925484305920792 * x * z, at(7));
            add(0.5462742152960396 * (xx - yy), at(8));
            if c.degree >= 3 {
                add(-0.5900435899266435 * y * (3.0 * xx - yy), at(9));
                add(2.890611442640554 * x * y * z, at(10));
                add(-0.4570457994644658 * y * (4.0 * zz - xx - yy), at(11));
                add(0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy), at(12));
                add(-0.4570457994644658 * x * (4.0 * zz - xx - yy), at(13));
                add(1.445305721320277 * z * (xx - yy), at(14));
                add(-0.5900435899266435 * x * (xx - 3.0 * yy), at(15));
            }
        }
    }
    col.map(|v| v.max(0.0))
}

/// `splat_segment` in f64 by brute force: every responding splat in
/// `(tmin, tmax)`, in `(t, index)` order, composited front to back.
pub fn composite_cpu(c: &PlyCloud, o: [f64; 3], d: [f64; 3], pix: f64, tmin: f64, tmax: f64) -> Segment {
    let mut hits: Vec<(f64, usize, f64)> = (0..c.len())
        .filter_map(|i| {
            let (t, a) = eval_cpu(c, i, o, d, pix);
            (a > 0.0 && t > tmin && t < tmax).then_some((t, i, a))
        })
        .collect();
    hits.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut seg = Segment { rgb: [0.0; 3], trans: 1.0, t_half: None };
    for (t, i, a) in hits.into_iter().take(ROUNDS as usize * K) {
        let col = colour_cpu(c, i, d);
        for ch in 0..3 {
            seg.rgb[ch] += seg.trans * a * col[ch];
        }
        seg.trans *= 1.0 - a;
        if seg.t_half.is_none() && seg.trans <= 0.5 {
            seg.t_half = Some(t);
        }
        if seg.trans <= 1e-4 {
            seg.trans = 0.0;
            break;
        }
    }
    seg
}

/// sRGB decode of one channel.
pub fn srgb_dec(c: f64) -> f64 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// sRGB encode of one channel, clamped to `[0, 1]`.
pub fn srgb_enc(c: f64) -> f64 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 { 12.92 * c } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

/// A segment's linear radiance (`splat_linear`).
pub fn linear(s: &Segment) -> [f64; 3] {
    let a = 1.0 - s.trans;
    if a <= 0.0 {
        return [0.0; 3];
    }
    s.rgb.map(|v| srgb_dec(v / a) * a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_round_trip() {
        for x in [0.0f32, 1.0, -1.0, 0.5, 1.5e-3, 65504.0, 3.1415927, -2.7e-5, 1e-8] {
            let y = f16_value(f16_bits(x));
            assert!((y - x).abs() <= x.abs() * 1e-3 + 6e-8, "{x} → {y}");
        }
    }

    #[test]
    fn whitening_is_the_inverse_covariance() {
        let c = PlyCloud {
            positions: vec![[0.0; 3]],
            scales: vec![[0.1, 0.2, 0.3]],
            quats: vec![[0.9238795, 0.0, 0.3826834, 0.0]],
            opacities: vec![0.5],
            sh: vec![[0.0; 3]],
            degree: 0,
        };
        let (m, _) = whiten(&c, 0);
        let s = c.to_splats();
        let a = s.inv_cov(0);
        // MᵀM = A
        let mtm = |i: usize, j: usize| (0..3).map(|k| m[k][i] * m[k][j]).sum::<f64>();
        let want = [a[0], a[1], a[2], a[3], a[4], a[5]];
        let got = [mtm(0, 0), mtm(0, 1), mtm(0, 2), mtm(1, 1), mtm(1, 2), mtm(2, 2)];
        for k in 0..6 {
            assert!((want[k] - got[k]).abs() < 1e-6 * want[k].abs().max(1.0), "{k}: {want:?} {got:?}");
        }
    }
}
