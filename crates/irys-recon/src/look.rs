//! v1: appearance — what a print on a table under room light looks like.
//!
//! The model is deterministic (no Monte Carlo): every pixel's radiance is
//! a closed form of a few G-buffer quantities and a [`Look`]:
//!
//! - **Material.** OpenPBR's `base_color`, `specular_roughness` and
//!   `specular_ior` for an opaque dielectric (PLA): a Lambert base weighted
//!   by `1 − F(n·v)` under a GGX dielectric specular (height-correlated
//!   Smith, Schlick at `specular_ior`), the spec's glossy-diffuse slab at
//!   single scatter. The fit works in grey; the colour is a per-channel
//!   ratio fitted at the end.
//! - **Environment.** Irradiance as a second-order spherical harmonic (nine
//!   coefficients, Ramamoorthi–Hanrahan) times a per-point ambient
//!   occlusion by the part itself, plus one **key light**: a disc of
//!   angular radius `key_radius` about `key_dir`, drawn as `KEY` fixed
//!   directions with a shadow ray each (so its shadows are soft and move
//!   with the geometry; [`crate::photo`] differentiates them).
//! - **Table.** The tag plane, Lambertian, albedo 1 — every radiance is in
//!   units of the table's; the part's albedo is relative to it.
//! - **Camera response.** The photo is linearised with the sRGB curve and
//!   then a fitted power `gamma` (phones tone-map; a power is the first
//!   term of that), and the exposure is absorbed in the lighting.
//!
//! For fixed albedo, roughness and key direction the image is linear in
//! the SH coefficients and the key's radiance, so [`fit`] alternates a
//! robust (Huber, IRLS) linear solve for those with 1-D solves for the
//! albedo and searches over roughness, the key's direction and gamma.

use irys_core::Vec3;

/// Key-light directions (fixed pattern on its disc).
pub const KEY: usize = 8;
/// Ambient-occlusion directions (fixed cosine-weighted pattern).
pub const AO: usize = 16;

/// The appearance parameters.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Look {
    /// SH irradiance coefficients (table-albedo units).
    pub sh: [f64; 9],
    /// Key radiance (a white Lambertian facing it reads `key`).
    pub key: f64,
    /// Unit direction towards the key light.
    pub key_dir: [f64; 3],
    /// Angular radius of the key, radians.
    pub key_radius: f64,
    /// Part albedo relative to the table (grey).
    pub albedo: f64,
    /// OpenPBR `specular_roughness`.
    pub rough: f64,
    /// OpenPBR `specular_ior`.
    pub ior: f64,
    /// Display power the photo was encoded with on top of sRGB.
    pub gamma: f64,
    /// Part `base_color`, linear sRGB, relative to the table's colour.
    pub rgb: [f64; 3],
}

impl Default for Look {
    fn default() -> Self {
        Self {
            sh: [0.6, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            key: 0.6,
            key_dir: [-0.4, -0.5, 0.77],
            key_radius: 0.12,
            albedo: 0.8,
            rough: 0.45,
            ior: 1.5,
            gamma: 1.0,
            rgb: [0.8; 3],
        }
    }
}

/// The real SH basis to second order (unnormalised polynomials: the
/// coefficients are fitted, so the constants do not matter).
pub fn sh_basis(n: Vec3) -> [f64; 9] {
    let (x, y, z) = (n.x, n.y, n.z);
    [1.0, y, z, x, x * y, y * z, 3.0 * z * z - 1.0, x * z, x * x - y * y]
}

/// What the shading needs at one surface sample.
#[derive(Debug, Clone, Copy)]
pub struct Surf {
    /// 0: nothing (excluded), 1: the part, 2: the table.
    pub kind: u8,
    /// Unit normal.
    pub n: Vec3,
    /// Unit direction to the eye.
    pub v: Vec3,
    /// Unoccluded fraction of the cosine-weighted hemisphere.
    pub ao: f64,
    /// Bit `k`: key direction `k` reaches the point.
    pub vis: u32,
}

impl Surf {
    /// Nothing.
    pub fn none() -> Surf {
        Surf { kind: 0, n: Vec3::new(0.0, 0.0, 1.0), v: Vec3::new(0.0, 0.0, 1.0), ao: 0.0, vis: 0 }
    }
}

/// The key's `KEY` directions: a centre and a ring of seven at 0.7 of the
/// radius (equal weights).
pub fn key_dirs(l: &Look) -> [Vec3; KEY] {
    let c = Vec3::new(l.key_dir[0], l.key_dir[1], l.key_dir[2]).normalize();
    let a = if c.z.abs() < 0.9 { Vec3::new(0.0, 0.0, 1.0) } else { Vec3::new(1.0, 0.0, 0.0) };
    let t = c.cross(&a).normalize();
    let b = c.cross(&t);
    let mut out = [c; KEY];
    for (k, o) in out.iter_mut().enumerate().skip(1) {
        let phi = std::f64::consts::TAU * (k - 1) as f64 / (KEY - 1) as f64;
        let r = 0.7 * l.key_radius;
        *o = (c + (t * phi.cos() + b * phi.sin()) * r.tan()).normalize();
    }
    out
}

/// `AO` fixed cosine-weighted directions about `+z` (a spiral).
pub fn ao_dirs() -> [Vec3; AO] {
    std::array::from_fn(|i| {
        let u = (i as f64 + 0.5) / AO as f64;
        let r = u.sqrt();
        let phi = i as f64 * 2.399_963_229_728_653;
        Vec3::new(r * phi.cos(), r * phi.sin(), (1.0 - u).sqrt())
    })
}

/// A frame about `n`: `(t, b)`.
pub fn frame(n: Vec3) -> (Vec3, Vec3) {
    let a = if n.z.abs() < 0.9 { Vec3::new(0.0, 0.0, 1.0) } else { Vec3::new(1.0, 0.0, 0.0) };
    let t = n.cross(&a).normalize();
    (t, n.cross(&t))
}

fn schlick(c: f64, ior: f64) -> f64 {
    let f0 = ((ior - 1.0) / (ior + 1.0)).powi(2);
    f0 + (1.0 - f0) * (1.0 - c.clamp(0.0, 1.0)).powi(5)
}

/// GGX dielectric specular `f · cos θ_l · π` for unit `n`, `v`, `l`.
fn ggx(n: Vec3, v: Vec3, l: Vec3, rough: f64, ior: f64) -> f64 {
    let (nl, nv) = (n.dot(&l), n.dot(&v));
    if nl <= 0.0 || nv <= 0.0 {
        return 0.0;
    }
    let h = (v + l).normalize();
    let nh = n.dot(&h).max(0.0);
    let a = (rough * rough).max(1e-3);
    let a2 = a * a;
    let d = a2 / (std::f64::consts::PI * (nh * nh * (a2 - 1.0) + 1.0).powi(2));
    let lam = |c: f64| 0.5 * (-1.0 + (1.0 + a2 * (1.0 - c * c) / (c * c)).sqrt());
    let g = 1.0 / (1.0 + lam(nl) + lam(nv));
    let f = schlick(v.dot(&h), ior);
    std::f64::consts::PI * d * g * f / (4.0 * nv)
}

/// The pieces of one sample's radiance that the linear solve needs:
/// `(coefficients of sh[0..9] and key, the albedo-free remainder split
/// the same way)`: `L = albedo · Σ d_i p_i + Σ s_i p_i` with `p =
/// (sh, key)`.
pub fn basis(s: &Surf, l: &Look, keys: &[Vec3; KEY]) -> ([f64; 10], [f64; 10]) {
    let mut d = [0.0; 10];
    let mut sp = [0.0; 10];
    match s.kind {
        1 => {
            let nv = s.n.dot(&s.v).max(0.0);
            let kd = 1.0 - schlick(nv, l.ior);
            let y = sh_basis(s.n);
            for i in 0..9 {
                d[i] = kd * y[i] * s.ao;
            }
            // Environment specular: the irradiance SH read in the mirror
            // direction stands in for the blurred radiance there.
            let r = s.n * (2.0 * nv) - s.v;
            let yr = sh_basis(r);
            let fs = schlick(nv, l.ior) * (1.0 - 0.6 * l.rough);
            for i in 0..9 {
                sp[i] = fs * yr[i] * s.ao;
            }
            let (mut kdif, mut kspec) = (0.0, 0.0);
            for (k, dir) in keys.iter().enumerate() {
                if s.vis >> k & 1 == 1 {
                    kdif += s.n.dot(dir).max(0.0);
                    kspec += ggx(s.n, s.v, *dir, l.rough, l.ior);
                }
            }
            d[9] = kd * kdif / KEY as f64;
            sp[9] = kspec / KEY as f64;
        }
        2 => {
            let y = sh_basis(s.n);
            for i in 0..9 {
                sp[i] = y[i] * s.ao;
            }
            let mut k = 0.0;
            for (kk, dir) in keys.iter().enumerate() {
                if s.vis >> kk & 1 == 1 {
                    k += s.n.dot(dir).max(0.0);
                }
            }
            sp[9] = k / KEY as f64;
        }
        _ => {}
    }
    (d, sp)
}

/// One sample's radiance under `l`.
pub fn shade(s: &Surf, l: &Look, keys: &[Vec3; KEY]) -> f64 {
    if s.kind == 0 {
        return 0.0;
    }
    let (d, sp) = basis(s, l, keys);
    let mut out = 0.0;
    for i in 0..10 {
        let p = if i < 9 { l.sh[i] } else { l.key };
        out += (l.albedo * d[i] + sp[i]) * p;
    }
    out
}

/// sRGB-encoded `[0, 1]` to linear.
pub fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// Huber's weight for a residual in units of σ, threshold `k`.
pub fn huber_w(r: f64, k: f64) -> f64 {
    if r.abs() <= k { 1.0 } else { k / r.abs() }
}

/// Huber's loss.
pub fn huber_rho(r: f64, k: f64) -> f64 {
    if r.abs() <= k { 0.5 * r * r } else { k * r.abs() - 0.5 * k * k }
}

/// Per-pixel bases of an image: for each pixel, the mean over its samples
/// of [`basis`] (`d`, `s`), with the pixel's weight (0: excluded).
pub struct Bases {
    /// Albedo-scaled part, per pixel.
    pub d: Vec<[f64; 10]>,
    /// The rest, per pixel.
    pub s: Vec<[f64; 10]>,
}

/// Solve `(sh, key)` then `albedo` against `target` (linear grey) with
/// Huber weights (`sigma` the residual scale, threshold 1.5σ), `iters`
/// alternations; returns the weighted Huber cost.
pub fn solve_linear(b: &Bases, target: &[f64], weight: &[f64], l: &mut Look, sigma: f64, iters: usize) -> f64 {
    let n = target.len();
    let k = 1.5;
    let mut wr = vec![1.0; n];
    let mut cost = 0.0;
    for _ in 0..iters {
        // (sh, key) given albedo.
        let mut a = vec![0.0; 100];
        let mut g = vec![0.0; 10];
        for p in 0..n {
            let w = weight[p] * wr[p];
            if w == 0.0 {
                continue;
            }
            let row: [f64; 10] = std::array::from_fn(|i| l.albedo * b.d[p][i] + b.s[p][i]);
            for i in 0..10 {
                g[i] += w * row[i] * target[p];
                for j in 0..10 {
                    a[i * 10 + j] += w * row[i] * row[j];
                }
            }
        }
        // A ridge on the SH beyond the constant term: the table sees one
        // normal and the part a few, so the higher orders are poorly
        // determined and would otherwise trade against the albedo.
        let tr: f64 = (0..10).map(|i| a[i * 10 + i]).sum::<f64>() / 10.0;
        for i in 0..10 {
            a[i * 10 + i] += if (1..9).contains(&i) { 1e-2 } else { 1e-6 } * tr.max(1e-12);
        }
        if let Some(x) = crate::fit::solve(&a, &g, 10) {
            l.sh.copy_from_slice(&x[..9]);
            l.key = x[9].max(0.0);
        }
        // Albedo given (sh, key): target − S = albedo · D on the part.
        let (mut num, mut den) = (0.0, 0.0);
        for p in 0..n {
            let w = weight[p] * wr[p];
            if w == 0.0 {
                continue;
            }
            let dd: f64 = (0..10).map(|i| b.d[p][i] * if i < 9 { l.sh[i] } else { l.key }).sum();
            let ss: f64 = (0..10).map(|i| b.s[p][i] * if i < 9 { l.sh[i] } else { l.key }).sum();
            num += w * dd * (target[p] - ss);
            den += w * dd * dd;
        }
        if den > 1e-12 {
            l.albedo = (num / den).clamp(0.02, 5.0);
        }
        // Huber weights from the new residual.
        cost = 0.0;
        for p in 0..n {
            if weight[p] == 0.0 {
                continue;
            }
            let m: f64 = (0..10).map(|i| (l.albedo * b.d[p][i] + b.s[p][i]) * if i < 9 { l.sh[i] } else { l.key }).sum();
            let r = (m - target[p]) / sigma;
            wr[p] = huber_w(r, k);
            cost += weight[p] * huber_rho(r, k);
        }
    }
    cost
}

/// The part's colour: per channel, the grey solution's lighting scaled by
/// `(α_env, α_key)` (the light's colour, from the table) and the part's
/// albedo `a` (`L = a (α_e E_d + α_k K_d) + α_e E_s + α_k K_s`), by
/// alternating least squares; returns `a` for `target`.
pub fn solve_colour(b: &Bases, target: &[f64], weight: &[f64], l: &Look) -> f64 {
    let n = target.len();
    let p = |i: usize| if i < 9 { l.sh[i] } else { l.key };
    let parts: Vec<[f64; 4]> = (0..n)
        .map(|q| {
            let ed: f64 = (0..9).map(|i| b.d[q][i] * p(i)).sum();
            let es: f64 = (0..9).map(|i| b.s[q][i] * p(i)).sum();
            [ed, b.d[q][9] * l.key, es, b.s[q][9] * l.key]
        })
        .collect();
    let (mut ae, mut ak, mut a) = (1.0, 1.0, l.albedo);
    for _ in 0..12 {
        // (α_e, α_k) given a.
        let (mut m, mut g) = ([0.0; 4], [0.0; 2]);
        for q in 0..n {
            if weight[q] == 0.0 {
                continue;
            }
            let x = [a * parts[q][0] + parts[q][2], a * parts[q][1] + parts[q][3]];
            m[0] += x[0] * x[0];
            m[1] += x[0] * x[1];
            m[3] += x[1] * x[1];
            g[0] += x[0] * target[q];
            g[1] += x[1] * target[q];
        }
        m[2] = m[1];
        let det = m[0] * m[3] - m[1] * m[1];
        if det.abs() > 1e-18 {
            ae = (m[3] * g[0] - m[1] * g[1]) / det;
            ak = (m[0] * g[1] - m[1] * g[0]) / det;
        }
        // a given (α_e, α_k).
        let (mut num, mut den) = (0.0, 0.0);
        for q in 0..n {
            if weight[q] == 0.0 {
                continue;
            }
            let dd = ae * parts[q][0] + ak * parts[q][1];
            let ss = ae * parts[q][2] + ak * parts[q][3];
            num += dd * (target[q] - ss);
            den += dd * dd;
        }
        if den > 1e-18 {
            a = (num / den).clamp(0.0, 5.0);
        }
    }
    a
}
