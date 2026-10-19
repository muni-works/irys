//! The image Jacobian of the v3 compare image by edge sampling (v3, piece 2):
//! irys-recon's v0 estimator (`jac.rs`, phase 2d's primary-visibility edge
//! term) carried to the scene families, with
//!
//! - per-vertex velocities from [`crate::velocity`] (analytic, forward-mode
//!   through the mesher) instead of v0's carrier surfaces;
//! - `f_in − f_out` read from the faithful render itself (bilinear, one
//!   pixel either side of the edge) instead of two traced rays, so the
//!   difference carries the OpenPBR shading, the shadows and the world;
//! - 2f's secondary edges for the key light's shadow on the support plane:
//!   every light-silhouette edge is projected onto `z = 0` along the key
//!   direction, `Ṗ = Ẋ − (Ẋ_z / L_z) L`, and sampled the same way where the
//!   plane is what the camera sees on both sides.
//!
//! Visibility is irys's: the two sides of each sample are cast through the
//! TLAS of the current scene ([`Caster`]); a primary edge counts where one
//! side is the object at the edge's own depth, a shadow edge where both
//! sides are the plane at the shadow point's depth. The interior term
//! (shading change of a face moving along itself) is left out, as in v0.

use crate::render::{Caster, TABLE};
use crate::scene::{Cam, Object, cross, dot, normalize, sub};
use crate::velocity::{NV, velocities};

/// Dense `∂I/∂θ`: `[pixel·3 + channel][slot]` over a window.
pub struct Jac {
    pub win: (usize, usize, usize, usize),
    pub cols: Vec<Vec<f64>>,
    /// Samples that passed visibility: primary, shadow.
    pub hits: (usize, usize),
}

fn bilinear(img: &[[f64; 3]], size: (usize, usize), x: f64, y: f64) -> [f64; 3] {
    let (w, h) = size;
    let fx = (x - 0.5).clamp(0.0, w as f64 - 1.0);
    let fy = (y - 0.5).clamp(0.0, h as f64 - 1.0);
    let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
    let p = |xx: usize, yy: usize| img[yy * w + xx];
    let mut o = [0.0; 3];
    for c in 0..3 {
        o[c] = (1.0 - ty) * ((1.0 - tx) * p(x0, y0)[c] + tx * p(x1, y0)[c]) + ty * ((1.0 - tx) * p(x0, y1)[c] + tx * p(x1, y1)[c]);
    }
    o
}

/// Screen position and its derivative along a 3D velocity.
fn project_d(cam: &Cam, x: [f64; 3], v: &[[f64; 3]; NV]) -> Option<([f64; 2], [[f64; 2]; NV])> {
    let (f, r, u, ty) = cam.frame();
    let d = sub(x, cam.eye);
    let z = dot(d, f);
    if z <= 1e-6 {
        return None;
    }
    let (w, h) = (cam.size.0 as f64, cam.size.1 as f64);
    let ax = ty * w / h;
    let (dr, du) = (dot(d, r), dot(d, u));
    let s = [(dr / z / ax + 1.0) * w / 2.0, (1.0 - du / z / ty) * h / 2.0];
    let ds = std::array::from_fn(|k| {
        let vz = dot(v[k], f);
        let dsx = (dot(v[k], r) * z - dr * vz) / (z * z) / ax;
        let dsy = (dot(v[k], u) * z - du * vz) / (z * z) / ty;
        [dsx * w / 2.0, -dsy * h / 2.0]
    });
    Some((s, ds))
}

fn depth(cam: &Cam, x: [f64; 3]) -> f64 {
    let (f, ..) = cam.frame();
    dot(sub(x, cam.eye), f)
}

/// Edges of a mesh: `(a, b, faces)` with vertices merged by position.
fn edges(p: &[[f64; 3]], t: &[[u32; 3]]) -> (Vec<usize>, Vec<((usize, usize), Vec<usize>)>, Vec<[f64; 3]>) {
    use std::collections::HashMap;
    let key = |q: [f64; 3]| ((q[0] * 1e4).round() as i64, (q[1] * 1e4).round() as i64, (q[2] * 1e4).round() as i64);
    let mut canon: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let rep: Vec<usize> = (0..p.len()).map(|i| *canon.entry(key(p[i])).or_insert(i)).collect();
    let mut fn_: Vec<[f64; 3]> = Vec::with_capacity(t.len());
    let mut map: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (fi, tri) in t.iter().enumerate() {
        let [a, b, c] = tri.map(|v| rep[v as usize]);
        let n = cross(sub(p[b], p[a]), sub(p[c], p[a]));
        let area = dot(n, n).sqrt();
        fn_.push(if area > 1e-9 { [n[0] / area, n[1] / area, n[2] / area] } else { [0.0; 3] });
        if area <= 1e-9 || a == b || b == c || a == c {
            continue;
        }
        for (x, y) in [(a, b), (b, c), (c, a)] {
            map.entry((x.min(y), x.max(y))).or_default().push(fi);
        }
    }
    let mut v: Vec<_> = map.into_iter().collect();
    v.sort_by_key(|e| e.0);
    (rep, v, fn_)
}

/// Options.
#[derive(Debug, Clone, Copy)]
pub struct EdgeOpts {
    /// Samples per pixel of projected edge length.
    pub density: f64,
    /// How far either side of the edge `f` is read (pixels).
    pub read: f64,
    /// Crease angle (degrees) above which an interior edge is sampled.
    pub crease: f64,
    /// Include the key light's shadow edges.
    pub shadows: bool,
}

impl Default for EdgeOpts {
    fn default() -> Self {
        EdgeOpts { density: 2.0, read: 1.0, crease: 25.0, shadows: true }
    }
}

/// `∂(model)/∂θ` of object `i` (all [`NV`] slots) over `win`, with `model`
/// the current render at `cam.size`, `caster` the current scene (object
/// `i` is payload `i`), `key` the direction toward the key light.
pub fn jacobian(o: &Object, i: usize, cam: &Cam, model: &[[f64; 3]], caster: &Caster, key: [f64; 3], win: (usize, usize, usize, usize), opt: &EdgeOpts) -> Jac {
    let size = cam.size;
    let (x0, y0, x1, y1) = win;
    let ww = x1 - x0;
    let npx = ww * (y1 - y0);
    let mut cols = vec![vec![0.0; npx * 3]; NV];
    let (p, vel, t) = velocities(o);
    let (rep, es, fnrm) = edges(&p, &t);
    let _ = rep;
    let cosc = opt.crease.to_radians().cos();
    let eye = cam.eye;
    let mut hits = (0, 0);
    let mut deposit = |s: [f64; 2], df: [f64; 3], vn: [f64; NV], wgt: f64| {
        let (px, py) = (s[0].floor() as i64, s[1].floor() as i64);
        if px < x0 as i64 || py < y0 as i64 || px >= x1 as i64 || py >= y1 as i64 {
            return;
        }
        let q = (py as usize - y0) * ww + (px as usize - x0);
        for k in 0..NV {
            if vn[k] == 0.0 {
                continue;
            }
            for c in 0..3 {
                cols[k][q * 3 + c] += df[c] * vn[k] * wgt;
            }
        }
    };
    let l = normalize(key);
    for ((a, b), faces) in &es {
        let (a, b) = (*a, *b);
        // Classify.
        let facing = |f: usize| dot(fnrm[f], sub(p[a], eye)) < 0.0;
        let lit = |f: usize| dot(fnrm[f], l) > 0.0;
        let primary = faces.len() == 1
            || faces.len() > 2
            || (faces.len() == 2 && (facing(faces[0]) != facing(faces[1]) || (facing(faces[0]) && dot(fnrm[faces[0]], fnrm[faces[1]]) < cosc)));
        let shadow = opt.shadows && l[2] > 0.05 && (faces.len() == 1 || (faces.len() == 2 && lit(faces[0]) != lit(faces[1])));
        if !primary && !shadow {
            continue;
        }
        // Primary edge.
        if primary {
            if let (Some((sa, da)), Some((sb, db))) = (project_d(cam, p[a], &vel[a]), project_d(cam, p[b], &vel[b])) {
                let e = [sb[0] - sa[0], sb[1] - sa[1]];
                let len = e[0].hypot(e[1]);
                if len > 1e-6 {
                    let n = [-e[1] / len, e[0] / len];
                    let ns = (len * opt.density).ceil().max(1.0) as usize;
                    let wgt = len / ns as f64;
                    for j in 0..ns {
                        let tt = (j as f64 + 0.5) / ns as f64;
                        let s = [sa[0] + e[0] * tt, sa[1] + e[1] * tt];
                        if s[0] < 0.0 || s[1] < 0.0 || s[0] >= size.0 as f64 || s[1] >= size.1 as f64 {
                            continue;
                        }
                        let x3 = [p[a][0] + (p[b][0] - p[a][0]) * tt, p[a][1] + (p[b][1] - p[a][1]) * tt, p[a][2] + (p[b][2] - p[a][2]) * tt];
                        let z = depth(cam, x3);
                        let tol = (0.01 * z).max(2.0);
                        let eps = 0.35;
                        let sm = caster.cast(cam, s[0] - eps * n[0], s[1] - eps * n[1]);
                        let sp = caster.cast(cam, s[0] + eps * n[0], s[1] + eps * n[1]);
                        let on = |h: &(u16, f64, [f64; 3])| h.0 as usize == i && (h.1 - z).abs() < tol;
                        if !(on(&sm) || on(&sp)) {
                            continue;
                        }
                        // A boundary in the image: the sides differ in what,
                        // how far, or which way they face (else the edge lies
                        // inside a face of a union and moves nothing).
                        if sm.0 == sp.0 && (sm.1 - sp.1).abs() < tol && dot(sm.2, sp.2) > cosc {
                            continue;
                        }
                        hits.0 += 1;
                        let fa = bilinear(model, size, s[0] - opt.read * n[0], s[1] - opt.read * n[1]);
                        let fb = bilinear(model, size, s[0] + opt.read * n[0], s[1] + opt.read * n[1]);
                        let df = [fa[0] - fb[0], fa[1] - fb[1], fa[2] - fb[2]];
                        let vn: [f64; NV] = std::array::from_fn(|k| {
                            let v = [da[k][0] + (db[k][0] - da[k][0]) * tt, da[k][1] + (db[k][1] - da[k][1]) * tt];
                            v[0] * n[0] + v[1] * n[1]
                        });
                        deposit(s, df, vn, wgt);
                    }
                }
            }
        }
        // Shadow edge on the plane.
        if shadow {
            let sh = |x: [f64; 3], v: &[[f64; 3]; NV]| -> ([f64; 3], [[f64; 3]; NV]) {
                let s = x[2] / l[2];
                let pp = [x[0] - s * l[0], x[1] - s * l[1], 0.0];
                let pv = std::array::from_fn(|k| {
                    let sv = v[k][2] / l[2];
                    [v[k][0] - sv * l[0], v[k][1] - sv * l[1], 0.0]
                });
                (pp, pv)
            };
            if p[a][2] < 1e-6 && p[b][2] < 1e-6 {
                continue; // on the plane already: no shadow edge of its own
            }
            let (pa, va) = sh(p[a], &vel[a]);
            let (pb, vb) = sh(p[b], &vel[b]);
            if let (Some((sa, da)), Some((sb, db))) = (project_d(cam, pa, &va), project_d(cam, pb, &vb)) {
                let e = [sb[0] - sa[0], sb[1] - sa[1]];
                let len = e[0].hypot(e[1]);
                if len > 1e-6 && len < 4.0 * (size.0 + size.1) as f64 {
                    let n = [-e[1] / len, e[0] / len];
                    let ns = (len * opt.density).ceil().max(1.0) as usize;
                    let wgt = len / ns as f64;
                    for j in 0..ns {
                        let tt = (j as f64 + 0.5) / ns as f64;
                        let s = [sa[0] + e[0] * tt, sa[1] + e[1] * tt];
                        if s[0] < 0.0 || s[1] < 0.0 || s[0] >= size.0 as f64 || s[1] >= size.1 as f64 {
                            continue;
                        }
                        let x3 = [pa[0] + (pb[0] - pa[0]) * tt, pa[1] + (pb[1] - pa[1]) * tt, 0.0];
                        let z = depth(cam, x3);
                        let tol = (0.01 * z).max(2.0);
                        let eps = 0.35;
                        let sm = caster.cast(cam, s[0] - eps * n[0], s[1] - eps * n[1]);
                        let sp = caster.cast(cam, s[0] + eps * n[0], s[1] + eps * n[1]);
                        let on = |h: &(u16, f64, [f64; 3])| h.0 == TABLE && (h.1 - z).abs() < tol;
                        if !(on(&sm) && on(&sp)) {
                            continue;
                        }
                        hits.1 += 1;
                        let fa = bilinear(model, size, s[0] - opt.read * n[0], s[1] - opt.read * n[1]);
                        let fb = bilinear(model, size, s[0] + opt.read * n[0], s[1] + opt.read * n[1]);
                        let df = [fa[0] - fb[0], fa[1] - fb[1], fa[2] - fb[2]];
                        let vn: [f64; NV] = std::array::from_fn(|k| {
                            let v = [da[k][0] + (db[k][0] - da[k][0]) * tt, da[k][1] + (db[k][1] - da[k][1]) * tt];
                            v[0] * n[0] + v[1] * n[1]
                        });
                        deposit(s, df, vn, wgt);
                    }
                }
            }
        }
    }
    Jac { win, cols, hits }
}
