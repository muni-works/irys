//! EWA projection of one splat into an OpenCV pinhole (`look::pinhole`),
//! and its adjoint, f64. The forward is 3DGS's / gsplat's: `Σ = R S Sᵀ Rᵀ`,
//! `Σ₂ = J W Σ Wᵀ Jᵀ + 0.3 I` (the same 0.3 px² dilation irys's ray tracer
//! applies), the camera-space mean clamped to 1.3× the frustum before the
//! Jacobian, the opacity-aware radius `√(2 ln(255 o) λ_max)` (the 1/255
//! cut makes it exact), SH colour toward the splat, and the optical depth
//! as a fourth channel. [`backward`] is the reverse of [`project`] line by
//! line; `tests/gradcheck.rs` checks it against central differences.

use irys_diff::look::pinhole::Pinhole;

use crate::model::{Gaussians, sigmoid};
use crate::sh::{self, COEFS};

/// The screen-space splat.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Proj {
    /// Mean, pixels (pixel `(i, j)`'s centre is `(i, j)`).
    pub uv: [f64; 2],
    /// `Σ₂⁻¹` as `(A, B, C)`: `q = A dx² + 2 B dx dy + C dy²`.
    pub conic: [f64; 3],
    /// Activated opacity.
    pub opacity: f64,
    /// sRGB colour (SH + ½, clamped at 0) and the optical depth `z`.
    pub colour: [f64; 4],
    /// Pixel radius of the 1/255 contour (0: culled).
    pub radius: f64,
}

/// The screen-space gradient of one splat, summed over pixels.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Grad2 {
    pub uv: [f64; 2],
    pub conic: [f64; 3],
    pub opacity: f64,
    pub colour: [f64; 4],
}

/// One splat's gradient in raw parameters.
#[derive(Debug, Clone, Copy)]
pub struct Grad3 {
    pub mean: [f64; 3],
    pub log_scale: [f64; 3],
    pub quat: [f64; 4],
    pub opacity: f64,
    pub sh: [[f64; 3]; COEFS],
}

impl Default for Grad3 {
    fn default() -> Self {
        Self { mean: [0.0; 3], log_scale: [0.0; 3], quat: [0.0; 4], opacity: 0.0, sh: [[0.0; 3]; COEFS] }
    }
}

/// Nearest optical depth a splat is drawn at (metres).
pub const NEAR: f64 = 0.02;
/// The screen-space dilation, px² (3DGS, gsplat, and irys's ray tracer).
pub const DILATE: f64 = 0.3;

type M3 = [[f64; 3]; 3];

fn quat_rot(q: [f64; 4]) -> M3 {
    let [w, x, y, z] = q;
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

/// Everything the adjoint needs, recomputed from the parameters.
struct Fwd {
    pc: [f64; 3],
    clamp: [f64; 2], // 0 free, else the ratio the component was clamped to
    t: [f64; 2],
    j: [[f64; 3]; 2],
    w: M3,         // world → camera
    qn: [f64; 4],  // normalised quaternion
    qlen: f64,
    r: M3,
    s: [f64; 3],
    m: M3,         // R S
    sig: M3,       // Σ
    tm: [[f64; 3]; 2], // J W
    s2: [f64; 3],  // Σ₂ (a, b, c), dilated
    det: f64,
    o: f64,
    dir: [f64; 3],
    vlen: f64,
    raw: [f64; 3], // colour before the clamp
    basis: [f64; COEFS],
    dbasis: [[f64; 3]; COEFS],
}

fn forward(g: &Gaussians, i: usize, cam: &Pinhole, degree: u32) -> Option<(Proj, Fwd)> {
    let mu = g.means[i];
    let v: [f64; 3] = std::array::from_fn(|a| mu[a] - cam.pos[a]);
    let w: M3 = std::array::from_fn(|k| std::array::from_fn(|a| cam.rot[a][k]));
    let pc: [f64; 3] = std::array::from_fn(|k| (0..3).map(|a| w[k][a] * v[a]).sum());
    let z = pc[2];
    if !(z > NEAR) {
        return None;
    }
    let (wd, ht) = (cam.width as f64, cam.height as f64);
    let tanx = wd / (2.0 * cam.fx);
    let tany = ht / (2.0 * cam.fy);
    let lim = [
        (-(cam.cx / cam.fx + 0.3 * tanx), (wd - cam.cx) / cam.fx + 0.3 * tanx),
        (-(cam.cy / cam.fy + 0.3 * tany), (ht - cam.cy) / cam.fy + 0.3 * tany),
    ];
    let mut clamp = [0.0; 2];
    let mut t = [0.0; 2];
    for a in 0..2 {
        let r = pc[a] / z;
        let rc = r.clamp(lim[a].0, lim[a].1);
        if rc != r {
            clamp[a] = rc;
        }
        t[a] = rc * z;
    }
    let (fx, fy) = (cam.fx, cam.fy);
    let j = [[fx / z, 0.0, -fx * t[0] / (z * z)], [0.0, fy / z, -fy * t[1] / (z * z)]];
    let q = g.quats[i];
    let qlen = q.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
    let qn = q.map(|v| v / qlen);
    let r = quat_rot(qn);
    let s = g.log_scales[i].map(f64::exp);
    let m: M3 = std::array::from_fn(|a| std::array::from_fn(|k| r[a][k] * s[k]));
    let sig: M3 = std::array::from_fn(|a| std::array::from_fn(|b| (0..3).map(|k| m[a][k] * m[b][k]).sum()));
    let tm: [[f64; 3]; 2] = std::array::from_fn(|rr| std::array::from_fn(|a| (0..3).map(|k| j[rr][k] * w[k][a]).sum()));
    let ts = |p: usize, qq: usize| -> f64 {
        (0..3).map(|a| (0..3).map(|b| tm[p][a] * sig[a][b] * tm[qq][b]).sum::<f64>()).sum()
    };
    let s2 = [ts(0, 0) + DILATE, ts(0, 1), ts(1, 1) + DILATE];
    let det = s2[0] * s2[2] - s2[1] * s2[1];
    if !(det > 0.0) {
        return None;
    }
    let conic = [s2[2] / det, -s2[1] / det, s2[0] / det];
    let o = sigmoid(g.opacity[i]);
    if o * 255.0 <= 1.0 {
        return None;
    }
    let mid = 0.5 * (s2[0] + s2[2]);
    let lmax = mid + (mid * mid - det).max(0.1).sqrt();
    let radius = (2.0 * (255.0 * o).ln() * lmax).sqrt();
    let uv = [fx * pc[0] / z + cam.cx, fy * pc[1] / z + cam.cy];
    // off screen entirely
    if uv[0] + radius < -0.5 || uv[0] - radius > wd - 0.5 || uv[1] + radius < -0.5 || uv[1] - radius > ht - 0.5 {
        return None;
    }
    let vlen = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
    let dir = v.map(|x| x / vlen);
    let (basis, dbasis) = sh::basis(dir, degree);
    let nc = sh::count(degree);
    let raw: [f64; 3] = std::array::from_fn(|ch| 0.5 + (0..nc).map(|k| basis[k] * g.sh[i][k][ch]).sum::<f64>());
    let p = Proj {
        uv,
        conic,
        opacity: o,
        colour: [raw[0].max(0.0), raw[1].max(0.0), raw[2].max(0.0), z],
        radius,
    };
    Some((p, Fwd { pc, clamp, t, j, w, qn, qlen, r, s, m, sig, tm, s2, det, o, dir, vlen, raw, basis, dbasis }))
}

/// Project splat `i` (`None`: culled — behind `NEAR`, degenerate, below
/// 1/255 opacity, or off screen).
pub fn project(g: &Gaussians, i: usize, cam: &Pinhole, degree: u32) -> Option<Proj> {
    forward(g, i, cam, degree).map(|(p, _)| p)
}

/// The adjoint of [`project`]: raw-parameter gradient from the screen-space
/// one. Zero for a culled splat.
pub fn backward(g: &Gaussians, i: usize, cam: &Pinhole, degree: u32, d: &Grad2) -> Grad3 {
    let mut out = Grad3::default();
    let Some((_, f)) = forward(g, i, cam, degree) else { return out };
    let (fx, fy) = (cam.fx, cam.fy);
    let z = f.pc[2];
    let mut dpc = [0.0; 3];
    // colour: SH coefficients and the view direction
    let nc = sh::count(degree);
    let mut ddir = [0.0; 3];
    for ch in 0..3 {
        if f.raw[ch] <= 0.0 {
            continue;
        }
        let dc = d.colour[ch];
        for k in 0..nc {
            out.sh[k][ch] = f.basis[k] * dc;
            let c = g.sh[i][k][ch];
            for a in 0..3 {
                ddir[a] += f.dbasis[k][a] * c * dc;
            }
        }
    }
    // dir = v / |v|
    let dd = ddir[0] * f.dir[0] + ddir[1] * f.dir[1] + ddir[2] * f.dir[2];
    let mut dmu: [f64; 3] = std::array::from_fn(|a| (ddir[a] - f.dir[a] * dd) / f.vlen);
    // depth channel
    dpc[2] += d.colour[3];
    // opacity
    out.opacity = d.opacity * f.o * (1.0 - f.o);
    // uv
    dpc[0] += d.uv[0] * fx / z;
    dpc[1] += d.uv[1] * fy / z;
    dpc[2] -= d.uv[0] * fx * f.pc[0] / (z * z) + d.uv[1] * fy * f.pc[1] / (z * z);
    // conic = Σ₂⁻¹: G_Σ₂ = −K G_K K, with G_K the symmetric matrix gradient
    let k = [[f.s2[2] / f.det, -f.s2[1] / f.det], [-f.s2[1] / f.det, f.s2[0] / f.det]];
    let gk = [[d.conic[0], 0.5 * d.conic[1]], [0.5 * d.conic[1], d.conic[2]]];
    let mut gs2 = [[0.0; 2]; 2];
    for a in 0..2 {
        for b in 0..2 {
            gs2[a][b] = -(0..2).map(|p| (0..2).map(|q| k[a][p] * gk[p][q] * k[q][b]).sum::<f64>()).sum::<f64>();
        }
    }
    // Σ₂ = T Σ Tᵀ: dΣ = Tᵀ G T, dT = 2 G T Σ
    let tm = f.tm;
    let dsig: M3 = std::array::from_fn(|a| {
        std::array::from_fn(|b| (0..2).map(|p| (0..2).map(|q| tm[p][a] * gs2[p][q] * tm[q][b]).sum::<f64>()).sum())
    });
    let ts: [[f64; 3]; 2] = std::array::from_fn(|p| std::array::from_fn(|b| (0..3).map(|a| tm[p][a] * f.sig[a][b]).sum()));
    let dt: [[f64; 3]; 2] = std::array::from_fn(|p| std::array::from_fn(|b| 2.0 * (0..2).map(|q| gs2[p][q] * ts[q][b]).sum::<f64>()));
    // T = J W: dJ = dT Wᵀ
    let dj: [[f64; 3]; 2] = std::array::from_fn(|p| std::array::from_fn(|kk| (0..3).map(|a| dt[p][a] * f.w[kk][a]).sum()));
    let _ = f.j;
    dpc[2] += dj[0][0] * (-fx / (z * z)) + dj[0][2] * (2.0 * fx * f.t[0] / (z * z * z)) + dj[1][1] * (-fy / (z * z))
        + dj[1][2] * (2.0 * fy * f.t[1] / (z * z * z));
    let dtv = [dj[0][2] * (-fx / (z * z)), dj[1][2] * (-fy / (z * z))];
    for a in 0..2 {
        if f.clamp[a] == 0.0 {
            dpc[a] += dtv[a];
        } else {
            dpc[2] += dtv[a] * f.clamp[a];
        }
    }
    // pc = W (μ − pos)
    for a in 0..3 {
        dmu[a] += (0..3).map(|kk| f.w[kk][a] * dpc[kk]).sum::<f64>();
    }
    out.mean = dmu;
    // Σ = M Mᵀ: dM = 2 dΣ M (dΣ symmetric)
    let dm: M3 = std::array::from_fn(|a| std::array::from_fn(|kk| 2.0 * (0..3).map(|b| dsig[a][b] * f.m[b][kk]).sum::<f64>()));
    // M = R S
    let mut dr = [[0.0; 3]; 3];
    for a in 0..3 {
        for kk in 0..3 {
            dr[a][kk] = dm[a][kk] * f.s[kk];
        }
    }
    for kk in 0..3 {
        let ds: f64 = (0..3).map(|a| dm[a][kk] * f.r[a][kk]).sum();
        out.log_scale[kk] = ds * f.s[kk];
    }
    let [w, x, y, zq] = f.qn;
    let dqn = [
        2.0 * (-zq * dr[0][1] + y * dr[0][2] + zq * dr[1][0] - x * dr[1][2] - y * dr[2][0] + x * dr[2][1]),
        2.0 * (y * dr[0][1] + zq * dr[0][2] + y * dr[1][0] - 2.0 * x * dr[1][1] - w * dr[1][2] + zq * dr[2][0] + w * dr[2][1]
            - 2.0 * x * dr[2][2]),
        2.0 * (-2.0 * y * dr[0][0] + x * dr[0][1] + w * dr[0][2] + x * dr[1][0] + zq * dr[1][2] - w * dr[2][0] + zq * dr[2][1]
            - 2.0 * y * dr[2][2]),
        2.0 * (-2.0 * zq * dr[0][0] - w * dr[0][1] + x * dr[0][2] + w * dr[1][0] - 2.0 * zq * dr[1][1] + y * dr[1][2]
            + x * dr[2][0]
            + y * dr[2][1]),
    ];
    let dot: f64 = (0..4).map(|a| dqn[a] * f.qn[a]).sum();
    out.quat = std::array::from_fn(|a| (dqn[a] - f.qn[a] * dot) / f.qlen);
    out
}
