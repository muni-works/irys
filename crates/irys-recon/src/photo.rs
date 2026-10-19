//! v1: the photo model — a part standing on the tag plane, seen by the
//! camera the tag gives, under a fitted [`Look`].
//!
//! The world is the tag frame (millimetres, `z` up, the table at `z = 0`).
//! The part is the program's solid placed by its pose `(x, y, yaw)`
//! ([`Program::pose`]); the camera is a pinhole with the tag's pose and
//! focal length and the principal point it was solved with, cropped and
//! resampled to the fit's film ([`Camera::crop`]).
//!
//! The renderer is deterministic: `ss × ss` stratified primary rays per
//! pixel, each shaded in closed form ([`crate::look`]) from a G-buffer
//! sample — the surface, its normal, its ambient occlusion from
//! [`look::AO`] fixed rays and its visibility to the key's [`look::KEY`]
//! directions. With no Monte Carlo noise, the model is a smooth function
//! of θ away from visibility events, and those are what the Jacobian's
//! boundary terms carry:
//!
//! - **primary edges** — the same estimator as v0's ([`crate::jac`]),
//!   with the two sides of each sample shaded by this model instead of
//!   traced paths;
//! - **secondary edges** (phase 2f's idea on vertex velocities): the key's
//!   shadow boundaries. For each key direction `l`, the part's
//!   light-silhouette edges are sampled in 3D; a sample `X` is pushed along
//!   `−l` to its receiver `S` (the table, or the part's own face, normal
//!   `m`), whose motion is `dS = v − (m·v / m·l) l`; projected, the
//!   shadow edge's normal speed times the shading jump across it is the
//!   pixel's derivative;
//! - **interior** for the pose: a yaw turns every face normal under a
//!   fixed pixel, `∂L/∂n · (ẑ × n)`, by a difference of the closed form
//!   (translations on the plane change no normal).
//!
//! What is left out is the derivative of the ambient occlusion with the
//! geometry (smooth and small; the forward-difference columns of v0, which
//! carried it, are kept behind [`Rig::fd_occluded`]).

use std::sync::Arc;

use irys_core::{Bvh, Point3, Ray, TriMesh, Vec3};
use rayon::prelude::*;

use crate::dsl::Program;
use crate::geom;
use crate::jac::{self, Mix, ViewJac};
use crate::look::{self, Look, Surf};
use crate::model::Eval;
use crate::tag::{Intrinsics, Pose};

/// A pinhole: square pixels, focal `f` and principal point in film pixels,
/// world → camera rotation rows (`x` right, `y` down, `z` forward), eye.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Camera {
    /// Focal length, film pixels.
    pub f: f64,
    /// Principal point, film pixels.
    pub cx: f64,
    /// Principal point, film pixels.
    pub cy: f64,
    /// World → camera rotation, rows.
    pub r: [[f64; 3]; 3],
    /// Eye, world mm.
    pub eye: [f64; 3],
    /// Film size.
    pub size: (usize, usize),
}

impl Camera {
    /// The tag's camera on the full image, cropped to `(x0, y0, w0, h0)`
    /// (full-image pixels) and resampled to `size`.
    pub fn crop(k: &Intrinsics, pose: &Pose, crop: (f64, f64, f64, f64), size: (usize, usize)) -> Self {
        let s = size.0 as f64 / crop.2;
        Self { f: k.f * s, cx: (k.cx - crop.0) * s, cy: (k.cy - crop.1) * s, r: pose.r, eye: pose.eye(), size }
    }

    /// The same camera on a film `scale` times the size.
    pub fn scaled(&self, scale: f64) -> Self {
        let size = ((self.size.0 as f64 * scale).round() as usize, (self.size.1 as f64 * scale).round() as usize);
        Self { f: self.f * scale, cx: self.cx * scale, cy: self.cy * scale, size, ..*self }
    }

    /// Eye as a point.
    pub fn eye(&self) -> Point3 {
        Point3::new(self.eye[0], self.eye[1], self.eye[2])
    }

    /// The world direction through film point `(x, y)` (unit).
    pub fn dir(&self, x: f64, y: f64) -> Vec3 {
        let (u, v) = ((x - self.cx) / self.f, (y - self.cy) / self.f);
        let r = &self.r;
        Vec3::new(
            r[0][0] * u + r[1][0] * v + r[2][0],
            r[0][1] * u + r[1][1] * v + r[2][1],
            r[0][2] * u + r[1][2] * v + r[2][2],
        )
        .normalize()
    }

    /// Film position of `p` and the 2 × 3 Jacobian of that map; `None`
    /// behind the eye.
    pub fn project(&self, p: Point3) -> Option<([f64; 2], [Vec3; 2])> {
        let d = p - self.eye();
        let row = |i: usize| Vec3::new(self.r[i][0], self.r[i][1], self.r[i][2]);
        let (r0, r1, r2) = (row(0), row(1), row(2));
        let (a, b, z) = (r0.dot(&d), r1.dot(&d), r2.dot(&d));
        if z <= 1e-6 {
            return None;
        }
        let x = self.f * a / z + self.cx;
        let y = self.f * b / z + self.cy;
        let jx = (r0 - r2 * (a / z)) * (self.f / z);
        let jy = (r1 - r2 * (b / z)) * (self.f / z);
        Some(([x, y], [jx, jy]))
    }

    /// Where the film point's ray meets the table `z = 0`.
    pub fn on_table(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let d = self.dir(x, y);
        let e = self.eye();
        (d.z < -1e-9).then(|| {
            let t = -e.z / d.z;
            (e.x + t * d.x, e.y + t * d.y)
        })
    }
}

/// One photo as the model sees it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct View {
    /// Its camera on the fit's film.
    pub cam: Camera,
    /// Its appearance.
    pub look: Look,
}

/// Every photo of one part, and the renderer's options.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Rig {
    /// Views (one per photo).
    pub views: Vec<View>,
    /// Primary rays per pixel per axis.
    pub ss: usize,
    /// Forward-difference columns for [`Program::occluded_params`] (v0's
    /// treatment), on top of the boundary terms.
    pub fd_occluded: bool,
    /// The key's shadow edges ([`secondary`]).
    pub secondary: bool,
    /// Secondary edge samples per key direction.
    pub shadow_samples: usize,
}

impl Rig {
    /// Every camera scaled by `s`.
    pub fn scaled(&self, s: f64) -> Rig {
        let mut r = self.clone();
        for v in &mut r.views {
            v.cam = v.cam.scaled(s);
        }
        r
    }
}

/// `prog`'s solid placed by its pose: world positions and triangles, and
/// the part-frame positions.
pub struct Posed {
    /// World positions.
    pub p: Vec<Point3>,
    /// Part-frame positions.
    pub local: Vec<Point3>,
    /// Triangles.
    pub t: Vec<[u32; 3]>,
    /// The tree.
    pub bvh: Bvh<TriMesh>,
    /// `[x, y, yaw°]`.
    pub pose: [f64; 3],
}

/// Part frame → world for `pose`.
pub fn place(pose: [f64; 3], q: Point3) -> Point3 {
    let (s, c) = pose[2].to_radians().sin_cos();
    Point3::new(c * q.x - s * q.y + pose[0], s * q.x + c * q.y + pose[1], q.z)
}

/// World → part frame for `pose`.
pub fn unplace(pose: [f64; 3], p: Point3) -> Point3 {
    let (s, c) = pose[2].to_radians().sin_cos();
    let (x, y) = (p.x - pose[0], p.y - pose[1]);
    Point3::new(c * x + s * y, -s * x + c * y, p.z)
}

/// Turn a part-frame vector into the world.
pub fn turn(pose: [f64; 3], v: Vec3) -> Vec3 {
    let (s, c) = pose[2].to_radians().sin_cos();
    Vec3::new(c * v.x - s * v.y, s * v.x + c * v.y, v.z)
}

/// Build and place `prog`.
pub fn posed(prog: &Program) -> Result<Posed, String> {
    let solid = prog.build()?;
    let (local, t) = geom::mesh(&solid);
    let pose = prog.pose.unwrap_or([0.0; 3]);
    let p: Vec<Point3> = local.iter().map(|&q| place(pose, q)).collect();
    let bvh = Bvh::build(geom::trimesh(&p, &t));
    Ok(Posed { p, local, t, bvh, pose })
}

/// The G-buffer sample of the ray through film point `(x, y)`; also the
/// hit point and whether it is the part.
pub fn surf(m: &Posed, cam: &Camera, keys: &[Vec3; look::KEY], ao: &[Vec3; look::AO], x: f64, y: f64) -> (Surf, Point3) {
    let d = cam.dir(x, y);
    let e = cam.eye();
    let ray = Ray::new(e, d);
    let t_table = if d.z < -1e-9 { -e.z / d.z } else { f64::INFINITY };
    let hit = m.bvh.trace_closest(&ray);
    let (kind, n, pt) = match hit {
        Some(h) if h.t < t_table => {
            let mut n = *h.normal.as_ref();
            if n.dot(&d) > 0.0 {
                n = -n;
            }
            (1u8, n, h.point)
        }
        _ if t_table.is_finite() => (2u8, Vec3::new(0.0, 0.0, 1.0), e + d * t_table),
        _ => return (Surf::none(), e),
    };
    let o = pt + n * 1e-3;
    let (tt, bb) = look::frame(n);
    let mut free = 0usize;
    for a in ao {
        let w = tt * a.x + bb * a.y + n * a.z;
        if !m.bvh.occluded(&Ray::new(o, w), f64::INFINITY) {
            free += 1;
        }
    }
    let mut vis = 0u32;
    for (k, l) in keys.iter().enumerate() {
        if n.dot(l) > 0.0 && !m.bvh.occluded(&Ray::new(o, *l), f64::INFINITY) {
            vis |= 1 << k;
        }
    }
    (Surf { kind, n, v: -d, ao: free as f64 / look::AO as f64, vis }, pt)
}

/// Radiance at film point `(x, y)`.
fn shade_at(m: &Posed, cam: &Camera, l: &Look, keys: &[Vec3; look::KEY], ao: &[Vec3; look::AO], x: f64, y: f64) -> (f64, bool) {
    let (s, _) = surf(m, cam, keys, ao, x, y);
    (look::shade(&s, l, keys), s.kind == 1)
}

/// The sub-pixel offsets of `ss × ss` stratified samples.
fn offsets(ss: usize) -> Vec<(f64, f64)> {
    (0..ss * ss).map(|i| (((i % ss) as f64 + 0.5) / ss as f64, ((i / ss) as f64 + 0.5) / ss as f64)).collect()
}

/// Per pixel: radiance, coverage, and the yaw interior derivative (per
/// degree) when `yaw` is asked for.
fn frame(m: &Posed, v: &View, ss: usize, yaw: bool) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (w, h) = v.cam.size;
    let keys = look::key_dirs(&v.look);
    let ao = look::ao_dirs();
    let offs = offsets(ss);
    let rows: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..h)
        .into_par_iter()
        .map(|y| {
            let (mut img, mut cov, mut dyaw) = (vec![0.0; w], vec![0.0; w], vec![0.0; if yaw { w } else { 0 }]);
            for x in 0..w {
                let (mut a, mut c, mut dy) = (0.0, 0.0, 0.0);
                for &(ox, oy) in &offs {
                    let (s, _) = surf(m, &v.cam, &keys, &ao, x as f64 + ox, y as f64 + oy);
                    a += look::shade(&s, &v.look, &keys);
                    if s.kind == 1 {
                        c += 1.0;
                        if yaw {
                            let eps = 1e-3;
                            let n2 = (s.n + Vec3::new(-s.n.y, s.n.x, 0.0) * eps).normalize();
                            let s2 = Surf { n: n2, ..s };
                            dy += (look::shade(&s2, &v.look, &keys) - look::shade(&s, &v.look, &keys)) / eps;
                        }
                    }
                }
                let k = offs.len() as f64;
                img[x] = a / k;
                cov[x] = c / k;
                if yaw {
                    dyaw[x] = dy / k * std::f64::consts::PI / 180.0;
                }
            }
            (img, cov, dyaw)
        })
        .collect();
    let mut out = (Vec::with_capacity(w * h), Vec::with_capacity(w * h), Vec::new());
    for (a, b, c) in rows {
        out.0.extend(a);
        out.1.extend(b);
        out.2.extend(c);
    }
    out
}

/// Per-pixel linear bases of the image under `v.look`'s key direction
/// (for [`look::solve_linear`]).
pub fn bases(m: &Posed, v: &View, ss: usize) -> look::Bases {
    let (w, h) = v.cam.size;
    let keys = look::key_dirs(&v.look);
    let ao = look::ao_dirs();
    let offs = offsets(ss);
    let px: Vec<([f64; 10], [f64; 10])> = (0..w * h)
        .into_par_iter()
        .map(|p| {
            let (x, y) = ((p % w) as f64, (p / w) as f64);
            let (mut d, mut s) = ([0.0; 10], [0.0; 10]);
            for &(ox, oy) in &offs {
                let (sf, _) = surf(m, &v.cam, &keys, &ao, x + ox, y + oy);
                let (a, b) = look::basis(&sf, &v.look, &keys);
                for i in 0..10 {
                    d[i] += a[i] / offs.len() as f64;
                    s[i] += b[i] / offs.len() as f64;
                }
            }
            (d, s)
        })
        .collect();
    look::Bases { d: px.iter().map(|x| x.0).collect(), s: px.iter().map(|x| x.1).collect() }
}

/// World velocities of every vertex for every parameter: the shape's
/// ([`geom::velocities`], turned by the yaw) and the pose's.
fn world_velocities(prog: &Program, m: &Posed) -> Vec<Vec<Vec3>> {
    let mut vel = geom::velocities(prog, &m.local);
    let pi = prog.pose_index();
    for (i, v) in vel.iter_mut().enumerate() {
        for x in v.iter_mut() {
            *x = turn(m.pose, *x);
        }
        if let Some(j) = pi {
            let p = m.p[i];
            v[j] = Vec3::new(1.0, 0.0, 0.0);
            v[j + 1] = Vec3::new(0.0, 1.0, 0.0);
            v[j + 2] = Vec3::new(-(p.y - m.pose[1]), p.x - m.pose[0], 0.0) * (std::f64::consts::PI / 180.0);
        }
    }
    vel
}

/// v0's primary edge estimator on this model.
fn primary(m: &Posed, v: &View, vel: &[Vec<Vec3>], np: usize, n: usize, seed: u64, out: &mut ViewJac) {
    let cam = &v.cam;
    let (w, h) = cam.size;
    let (p, t) = (&m.p, &m.t);
    let keys = look::key_dirs(&v.look);
    let ao = look::ao_dirs();
    let eye = cam.eye();
    let cands = jac::candidates_from(eye, p, t);
    struct Piece {
        ci: usize,
        sa: [f64; 2],
        sb: [f64; 2],
        za: f64,
        zb: f64,
        t0: f64,
        t1: f64,
        we: f64,
    }
    let fwd = Vec3::new(cam.r[2][0], cam.r[2][1], cam.r[2][2]);
    let mut pieces = Vec::new();
    for (ci, c) in cands.iter().enumerate() {
        let (pa, pb) = (p[c.a], p[c.b]);
        let (Some((sa, _)), Some((sb, _))) = (cam.project(pa), cam.project(pb)) else { continue };
        let Some((t0, t1)) = jac::clip(sa, sb, w as f64, h as f64) else { continue };
        let len = (sb[0] - sa[0]).hypot(sb[1] - sa[1]) * (t1 - t0);
        if len < 1e-9 {
            continue;
        }
        let (za, zb) = ((pa - eye).dot(&fwd), (pb - eye).dot(&fwd));
        pieces.push(Piece { ci, sa, sb, za, zb, t0, t1, we: len * c.weight });
    }
    if pieces.is_empty() || n == 0 {
        return;
    }
    let total: f64 = pieces.iter().map(|q| q.we).sum();
    let mut cdf = Vec::with_capacity(pieces.len());
    let mut acc = 0.0;
    for q in &pieces {
        acc += q.we;
        cdf.push(acc / total);
    }
    let mut rng = Mix(seed ^ 0x5eed_ed9e);
    let draws: Vec<(f64, f64)> = (0..n).map(|i| ((i as f64 + rng.uniform()) / n as f64, rng.uniform())).collect();
    let results: Vec<Option<(usize, Vec<f64>, f64, f64)>> = draws
        .par_iter()
        .map(|&(u, u2)| {
            let pi = cdf.partition_point(|&c| c < u).min(pieces.len() - 1);
            let q = &pieces[pi];
            let c = &cands[q.ci];
            let us = q.t0 + (q.t1 - q.t0) * u2;
            let s = [q.sa[0] + us * (q.sb[0] - q.sa[0]), q.sa[1] + us * (q.sb[1] - q.sa[1])];
            let t3 = us * q.za / (us * q.za + (1.0 - us) * q.zb);
            let x = p[c.a] + (p[c.b] - p[c.a]) * t3;
            let d = [q.sb[0] - q.sa[0], q.sb[1] - q.sa[1]];
            let dl = d[0].hypot(d[1]);
            let mut nrm = [-d[1] / dl, d[0] / dl];
            let (sc, _) = cam.project(p[c.c])?;
            if (sc[0] - s[0]) * nrm[0] + (sc[1] - s[1]) * nrm[1] > 0.0 {
                nrm = [-nrm[0], -nrm[1]];
            }
            let (_, jp) = cam.project(x)?;
            let scale = total / (c.weight * n as f64);
            let wn: Vec<f64> = (0..np)
                .map(|j| {
                    let vv = vel[c.a][j] * (1.0 - t3) + vel[c.b][j] * t3;
                    scale * (nrm[0] * jp[0].dot(&vv) + nrm[1] * jp[1].dot(&vv))
                })
                .collect();
            let e = jac::EPS_PX;
            let inner = [s[0] - e * nrm[0], s[1] - e * nrm[1]];
            let outer = [s[0] + e * nrm[0], s[1] + e * nrm[1]];
            // Ownership: the inner ray must land on the owner's plane.
            let fnrm = jac::face_normal(p, &t[c.tri]).normalize();
            let plane_c = fnrm.dot(&p[c.a].coords());
            let hi = m.bvh.trace_closest(&Ray::new(eye, cam.dir(inner[0], inner[1])))?;
            if (fnrm.dot(&hi.point.coords()) - plane_c).abs() > 1e-2 {
                return None;
            }
            if let Some(ho) = m.bvh.trace_closest(&Ray::new(eye, cam.dir(outer[0], outer[1]))) {
                if (ho.point - x).norm() < 2e-2 {
                    let view = (eye - x).normalize();
                    let fo = ho.normal.as_ref().dot(&view).abs();
                    let fi = fnrm.dot(&view).abs();
                    let tie = (fo - fi).abs() < 1e-9;
                    if fo > fi + 1e-9 || (tie && jac::key(ho.normal.as_ref()) > jac::key(&fnrm)) {
                        return None;
                    }
                }
            }
            let (fi, _) = shade_at(m, cam, &v.look, &keys, &ao, inner[0], inner[1]);
            let (fo, out_part) = shade_at(m, cam, &v.look, &keys, &ao, outer[0], outer[1]);
            let px = (s[1].floor().clamp(0.0, h as f64 - 1.0) as usize) * w + s[0].floor().clamp(0.0, w as f64 - 1.0) as usize;
            Some((px, wn, fi - fo, 1.0 - out_part as u8 as f64))
        })
        .collect();
    for (px, wn, df, dm) in results.into_iter().flatten() {
        for j in 0..np {
            out.img[px * np + j] += wn[j] * df;
            out.mask[px * np + j] += wn[j] * dm;
        }
    }
}

/// The key's shadow edges (see the module docs).
fn secondary(m: &Posed, v: &View, vel: &[Vec<Vec3>], np: usize, per_dir: usize, seed: u64, out: &mut ViewJac) {
    use std::collections::HashMap;
    let cam = &v.cam;
    let (w, h) = cam.size;
    let (p, t) = (&m.p, &m.t);
    let keys = look::key_dirs(&v.look);
    let ao = look::ao_dirs();
    let eye = cam.eye();
    // Welded edges and their faces.
    let q = |v: Point3| ((v.x * 1e4).round() as i64, (v.y * 1e4).round() as i64, (v.z * 1e4).round() as i64);
    let mut weld: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let id: Vec<usize> = p
        .iter()
        .map(|&v| {
            let n = weld.len();
            *weld.entry(q(v)).or_insert(n)
        })
        .collect();
    let normals: Vec<Vec3> = t.iter().map(|tri| jac::face_normal(p, tri)).collect();
    let mut edges: HashMap<(usize, usize), Vec<(usize, usize, usize)>> = HashMap::new();
    for (ti, tri) in t.iter().enumerate() {
        if normals[ti].norm() < 1e-6 {
            continue;
        }
        for k in 0..3 {
            let (a, b) = (tri[k] as usize, tri[(k + 1) % 3] as usize);
            let (ia, ib) = (id[a], id[b]);
            if ia == ib {
                continue;
            }
            edges.entry((ia.min(ib), ia.max(ib))).or_default().push((ti, a, b));
        }
    }
    let mut list: Vec<_> = edges.into_iter().collect();
    list.sort_by_key(|(k, _)| *k);
    let mut rng = Mix(seed ^ 0x5ad0_77ed);
    for (kd, l) in keys.iter().enumerate() {
        if l.z <= 0.05 {
            continue;
        }
        // Light-silhouette edges: one adjacent face towards the light, one
        // away (or an unmatched edge of a lit face).
        let sil: Vec<(usize, usize, f64)> = list
            .iter()
            .filter_map(|(_, fs)| {
                let lit: Vec<bool> = fs.iter().map(|&(ti, ..)| normals[ti].dot(l) > 0.0).collect();
                let (ti, a, b) = fs[0];
                let ok = match fs.len() {
                    2 => lit[0] != lit[1] && normals[fs[0].0].normalize().dot(&normals[fs[1].0].normalize()) < 1.0 - 1e-7,
                    _ => lit[0],
                };
                let _ = ti;
                ok.then(|| (a, b, (p[b] - p[a]).norm()))
            })
            .collect();
        if sil.is_empty() {
            continue;
        }
        let total: f64 = sil.iter().map(|e| e.2).sum();
        let mut cdf = Vec::with_capacity(sil.len());
        let mut acc = 0.0;
        for e in &sil {
            acc += e.2;
            cdf.push(acc / total);
        }
        let n = per_dir;
        let draws: Vec<(f64, f64)> = (0..n).map(|i| ((i as f64 + rng.uniform()) / n as f64, rng.uniform())).collect();
        let res: Vec<Option<(usize, Vec<f64>)>> = draws
            .par_iter()
            .map(|&(u, u2)| {
                let ei = cdf.partition_point(|&c| c < u).min(sil.len() - 1);
                let (a, b, len) = sil[ei];
                let x = p[a] + (p[b] - p[a]) * u2;
                // The receiver: away from the light from X.
                let back = -*l;
                let ray = Ray::new(x + back * 1e-3, back);
                let t_table = if back.z < -1e-9 { -(x.z) / back.z } else { f64::INFINITY };
                let (s_pt, mnorm) = match m.bvh.trace_closest(&ray) {
                    Some(hh) if hh.t < t_table => (hh.point, *hh.normal.as_ref()),
                    _ if t_table.is_finite() => (x + back * t_table, Vec3::new(0.0, 0.0, 1.0)),
                    _ => return None,
                };
                let ml = mnorm.dot(l);
                if ml.abs() < 1e-3 {
                    return None;
                }
                // Seen from the camera?
                let (s, jp) = cam.project(s_pt)?;
                if s[0] < 0.0 || s[1] < 0.0 || s[0] >= w as f64 || s[1] >= h as f64 {
                    return None;
                }
                let d = cam.dir(s[0], s[1]);
                let tt = (s_pt - eye).norm();
                if m.bvh.occluded(&Ray::new(eye, d), tt - 2e-2) {
                    return None;
                }
                let e3 = (p[b] - p[a]).normalize();
                let es = e3 - *l * (mnorm.dot(&e3) / ml);
                let tau = [jp[0].dot(&es), jp[1].dot(&es)];
                let tl = tau[0].hypot(tau[1]);
                if tl < 1e-9 {
                    return None;
                }
                let ns = [-tau[1] / tl, tau[0] / tl];
                let eps = jac::EPS_PX * 2.0;
                let (fa, _) = shade_at(m, cam, &v.look, &keys, &ao, s[0] - eps * ns[0], s[1] - eps * ns[1]);
                let (fb, _) = shade_at(m, cam, &v.look, &keys, &ao, s[0] + eps * ns[0], s[1] + eps * ns[1]);
                let df = fa - fb;
                if df == 0.0 {
                    return None;
                }
                let scale = total / n as f64 * tl;
                let wn: Vec<f64> = (0..np)
                    .map(|j| {
                        let vv = vel[a][j] * (1.0 - u2) + vel[b][j] * u2;
                        let ds = vv - *l * (mnorm.dot(&vv) / ml);
                        scale * df * (ns[0] * jp[0].dot(&ds) + ns[1] * jp[1].dot(&ds))
                    })
                    .collect();
                let _ = (len, kd);
                let px = (s[1].floor() as usize) * w + s[0].floor() as usize;
                Some((px, wn))
            })
            .collect();
        for (px, wn) in res.into_iter().flatten() {
            for j in 0..np {
                out.img[px * np + j] += wn[j];
            }
        }
    }
}

/// Render `prog` in every view of `rig`; with `jac`, its Jacobian.
pub fn eval(prog: &Program, rig: &Rig, edge_samples: usize, seed: u64, jac: bool) -> Result<Eval, String> {
    let m = posed(prog)?;
    let np = prog.params().len();
    let vel = if jac { world_velocities(prog, &m) } else { Vec::new() };
    let yaw_col = prog.pose_index().map(|j| j + 2);
    let mut e = Eval { img: Vec::new(), cov: Vec::new(), jac: jac.then(Vec::new), tris: m.t.len() };
    for (vi, v) in rig.views.iter().enumerate() {
        let (img, cov, dyaw) = frame(&m, v, rig.ss, jac && yaw_col.is_some());
        if jac {
            let npx = img.len();
            let mut j = ViewJac { img: vec![0.0; npx * np], mask: vec![0.0; npx * np], np };
            let s = seed + 1000 * vi as u64;
            primary(&m, v, &vel, np, edge_samples, s + 17, &mut j);
            if rig.secondary {
                secondary(&m, v, &vel, np, rig.shadow_samples, s + 29, &mut j);
            }
            if let Some(c) = yaw_col {
                for px in 0..npx {
                    j.img[px * np + c] += dyaw[px];
                }
            }
            e.jac.as_mut().unwrap().push(j);
        }
        e.img.push(img);
        e.cov.push(cov);
    }
    if jac && rig.fd_occluded {
        for j in prog.occluded_params() {
            let mut th = prog.params();
            let hstep = crate::model::FD_STEP;
            th[j] += hstep;
            let mut q = prog.clone();
            q.set_params(&th);
            let Ok(ep) = eval(&q, rig, 0, seed, false) else { continue };
            let jv = e.jac.as_mut().unwrap();
            for (vi, view) in jv.iter_mut().enumerate() {
                for px in 0..e.img[vi].len() {
                    view.img[px * np + j] = (ep.img[vi][px] - e.img[vi][px]) / hstep;
                }
            }
        }
    }
    Ok(e)
}

/// The rig as a shareable handle.
pub type Shared = Arc<Rig>;

/// v0's residual blobs on the top face, for the photo model: the top-face
/// hits of strongly mis-explained pixels, in the part frame.
pub fn blobs(fit: &crate::fit::Fit, rig: &Rig) -> Vec<crate::search::Blob> {
    let Ok(m) = posed(&fit.prog) else { return Vec::new() };
    let h = fit.prog.h;
    let size = rig.views[0].cam.size;
    let (w, ht) = size;
    let mut tops = Vec::new();
    let mut eyes = Vec::new();
    for (vi, v) in rig.views.iter().enumerate() {
        let cam = &v.cam;
        eyes.push(unplace(m.pose, cam.eye()));
        let rb = crate::fit::blur(&fit.residual[vi], size, 1, 1.0);
        let top: Vec<Option<(f64, f64)>> = (0..w * ht)
            .into_par_iter()
            .map(|p| {
                if rb[p].abs() < 2.0 {
                    return None;
                }
                let d = cam.dir((p % w) as f64 + 0.5, (p / w) as f64 + 0.5);
                let hit = m.bvh.trace_closest(&Ray::new(cam.eye(), d))?;
                let n = hit.normal.as_ref();
                (n.z > 0.95 && (hit.point.z - h).abs() < 0.05).then(|| {
                    let q = unplace(m.pose, hit.point);
                    (q.x, q.y)
                })
            })
            .collect();
        tops.push(top);
    }
    crate::search::blobs_from(&tops, size, &eyes, h)
}

/// The base box and its pose from the masks: the table points under the
/// mask give the footprint's orientation (minimum-area rectangle) and
/// centre; then a grid over `(w, d, h)` and a shift towards the camera
/// (the mask overhangs the footprint away from it) scored by mask IoU
/// with ray–box slabs, no render.
pub fn init_base(target: &crate::fit::Target, rig: &Rig) -> Program {
    init_bases(target, rig).swap_remove(0)
}

/// [`init_base`]'s best box in each band of heights (`h` ≤ 9, 9–15, ≥ 15
/// mm), best first. One silhouette trades the plate's depth against its
/// height (`d sin e + h cos e`), and a boss on the plate reads as a taller
/// plate, so the silhouette alone cannot pick; the shading can
/// ([`crate::v1`] fits each).
pub fn init_bases(target: &crate::fit::Target, rig: &Rig) -> Vec<Program> {
    let (w, h) = target.size;
    let mut pts = Vec::new();
    for (vi, v) in rig.views.iter().enumerate() {
        for p in 0..w * h {
            if target.mask[vi][p] > 0.5 && target.w(vi, p) > 0.0 {
                if let Some(q) = v.cam.on_table((p % w) as f64 + 0.5, (p / w) as f64 + 0.5) {
                    pts.push(q);
                }
            }
        }
    }
    if pts.len() < 10 {
        return vec![Program { w: 50.0, d: 40.0, h: 8.0, corner: crate::dsl::Corner::Sharp, features: Vec::new(), pose: Some([0.0, 60.0, 0.0]) }];
    }
    let pct = |mut v: Vec<f64>, q: f64| {
        v.sort_by(f64::total_cmp);
        v[((v.len() - 1) as f64 * q) as usize]
    };
    let mut best = (f64::INFINITY, 0.0, [0.0; 4]);
    for k in 0..180 {
        let yaw = (k as f64 * 0.5).to_radians();
        let (s, c) = yaw.sin_cos();
        let us: Vec<f64> = pts.iter().map(|p| c * p.0 + s * p.1).collect();
        let vs: Vec<f64> = pts.iter().map(|p| -s * p.0 + c * p.1).collect();
        let (u0, u1, v0, v1) = (pct(us.clone(), 0.02), pct(us, 0.98), pct(vs.clone(), 0.02), pct(vs, 0.98));
        let area = (u1 - u0) * (v1 - v0);
        if area < best.0 {
            best = (area, yaw, [u0, u1, v0, v1]);
        }
    }
    let (yaw, [u0, u1, v0, v1]) = (best.1, best.2);
    let (s, c) = yaw.sin_cos();
    let (uc, vc) = ((u0 + u1) / 2.0, (v0 + v1) / 2.0);
    let centre = (c * uc - s * vc, s * uc + c * vc);
    let (ew, ed) = (u1 - u0, v1 - v0);
    // The horizontal direction from the first camera to the part.
    let e0 = rig.views[0].cam.eye();
    let away = Vec3::new(centre.0 - e0.x, centre.1 - e0.y, 0.0).normalize();
    // Rays in the world, per view, at the target's resolution.
    let rays: Vec<Vec<(Point3, Vec3, bool, bool)>> = rig
        .views
        .iter()
        .enumerate()
        .map(|(vi, v)| {
            (0..w * h)
                .map(|p| {
                    let d = v.cam.dir((p % w) as f64 + 0.5, (p / w) as f64 + 0.5);
                    (v.cam.eye(), d, target.mask[vi][p] > 0.5, target.w(vi, p) > 0.0)
                })
                .collect()
        })
        .collect();
    let deg = yaw.to_degrees();
    let mut cands = Vec::new();
    for wi in 0..13 {
        for di in 0..13 {
            for hi in 0..14 {
                for si in 0..6 {
                    cands.push((ew * (0.55 + 0.05 * wi as f64), ed * (0.55 + 0.05 * di as f64), 2.0 + 2.0 * hi as f64, -2.5 * si as f64));
                }
            }
        }
    }
    // Agreement of a box with the mask: the box is the base plate, and
    // the features (a boss above it) add to the mask, so the mask outside
    // the box costs less than the box outside the mask.
    let iou = |bw: f64, bd: f64, bh: f64, pose: [f64; 3]| {
        let (mut inter, mut over, mut under, mut n) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for rs in &rays {
            for &(o, d, m, ok) in rs {
                if !ok {
                    continue;
                }
                let a = slab(unplace(pose, o), turn([0.0, 0.0, -pose[2]], d), bw / 2.0, bd / 2.0, bh);
                inter += (a && m) as u8 as f64;
                over += (a && !m) as u8 as f64;
                under += (m && !a) as u8 as f64;
                n += m as u8 as f64;
            }
        }
        (inter - 1.5 * over - 0.5 * under) / n.max(1.0)
    };
    let scored: Vec<(f64, (f64, f64, f64, f64))> = cands
        .par_iter()
        .map(|&(bw, bd, bh, sh)| {
            let pose = [centre.0 + away.x * sh, centre.1 + away.y * sh, deg];
            (iou(bw, bd, bh, pose), (bw, bd, bh, sh))
        })
        .collect();
    if let Ok(v) = std::env::var("RECON_IOU") {
        let x: Vec<f64> = v.split(',').map(|t| t.parse().unwrap()).collect();
        eprintln!("iou of {x:?}: {:.4}; extents {ew:.1} {ed:.1} centre {centre:?} yaw {deg:.1}", iou(x[0], x[1], x[2], [x[3], x[4], x[5]]));
        let img = image::RgbImage::from_fn(w as u32, h as u32, |px, py| {
            let (o, d, m, ok) = rays[0][(py as usize) * w + px as usize];
            let pose = [x[3], x[4], x[5]];
            let a = slab(unplace(pose, o), turn([0.0, 0.0, -pose[2]], d), x[0] / 2.0, x[1] / 2.0, x[2]);
            image::Rgb([if m { 255 } else { 0 }, if a { 255 } else { 0 }, if ok { 80 } else { 0 }])
        });
        img.save("/tmp/recon_iou.png").ok();
    }
    let bands = [(1.5, 9.0), (9.0, 15.0), (15.0, 40.0)];
    let mut out: Vec<(f64, Program)> = Vec::new();
    for (lo, hi) in bands {
        let Some((_, (bw, bd, bh, sh))) = scored.iter().filter(|(_, c)| c.2 >= lo && c.2 <= hi).fold(None, |a: Option<(f64, (f64, f64, f64, f64))>, b| if a.is_none_or(|a| b.0 > a.0) { Some(*b) } else { a }) else { continue };
        // Pattern search on (w, d, h, x, y, yaw) from the band's best.
        let mut x = [bw, bd, bh, centre.0 + away.x * sh, centre.1 + away.y * sh, deg];
        let f = |x: &[f64; 6]| if x[0] > 4.0 && x[1] > 4.0 && x[2] >= lo && x[2] <= hi { iou(x[0], x[1], x[2], [x[3], x[4], x[5]]) } else { f64::NEG_INFINITY };
        let mut fx = f(&x);
        let mut step = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0];
        for _ in 0..6 {
            loop {
                let mut moved = false;
                for k in 0..6 {
                    for sgn in [1.0, -1.0] {
                        let mut y = x;
                        y[k] += sgn * step[k];
                        let fy = f(&y);
                        if fy > fx {
                            (x, fx, moved) = (y, fy, true);
                        }
                    }
                }
                if !moved {
                    break;
                }
            }
            for s in step.iter_mut() {
                *s *= 0.5;
            }
        }
        if std::env::var("RECON_IOU").is_ok() {
            eprintln!("band {lo}-{hi}: agreement {fx:.4} at {x:?}");
        }
        out.push((fx, Program { w: x[0], d: x[1], h: x[2], corner: crate::dsl::Corner::Sharp, features: Vec::new(), pose: Some([x[3], x[4], x[5]]) }));
    }
    out.sort_by(|a, b| b.0.total_cmp(&a.0));
    out.into_iter().map(|(_, p)| p).collect()
}

/// Whether the ray `o + t d` meets the box `[−a, a] × [−b, b] × [0, c]`.
fn slab(o: Point3, d: Vec3, a: f64, b: f64, c: f64) -> bool {
    let (mut t0, mut t1) = (0.0f64, f64::INFINITY);
    for (oo, dd, lo, hi) in [(o.x, d.x, -a, a), (o.y, d.y, -b, b), (o.z, d.z, 0.0, c)] {
        if dd.abs() < 1e-12 {
            if oo < lo || oo > hi {
                return false;
            }
            continue;
        }
        let (ta, tb) = ((lo - oo) / dd, (hi - oo) / dd);
        t0 = t0.max(ta.min(tb));
        t1 = t1.min(ta.max(tb));
    }
    t0 <= t1
}

/// What [`fit_look`] searches.
#[derive(Debug, Clone, Copy)]
pub struct LookSearch {
    /// Search the key's direction on a grid first.
    pub key_grid: bool,
    /// Search roughness, key radius and gamma.
    pub nonlinear: bool,
}

/// Robust σ of `r` over the weighted pixels (1.4826 · MAD).
pub fn mad_sigma(r: &[f64], weight: &[f64]) -> f64 {
    let mut v: Vec<f64> = r.iter().zip(weight).filter(|(_, w)| **w > 0.0).map(|(x, _)| x.abs()).collect();
    if v.is_empty() {
        return 1.0;
    }
    v.sort_by(f64::total_cmp);
    (1.4826 * v[v.len() / 2]).max(1e-6)
}

/// Fit view `vi`'s look to `target` (linear grey, before the view's
/// gamma) with the geometry of `prog` held: the linear solve always, and
/// with `search` the key direction (grid, then local), roughness, key
/// radius and gamma. Returns the robust σ of the final residual.
pub fn fit_look(prog: &Program, rig: &mut Rig, vi: usize, target_lin: &[f64], weight: &[f64], search: LookSearch) -> f64 {
    let Ok(m) = posed(prog) else { return 1.0 };
    let ss = rig.ss.min(2);
    let eval = |look: &Look, cam: &Camera, gamma: f64, sigma: f64| -> (f64, Look) {
        let mut l = *look;
        let v = View { cam: *cam, look: l };
        let b = bases(&m, &v, ss);
        let t: Vec<f64> = target_lin.iter().map(|x| x.max(0.0).powf(1.0 / gamma)).collect();
        let c = look::solve_linear(&b, &t, weight, &mut l, sigma, 4);
        l.gamma = gamma;
        (c, l)
    };
    let cam = rig.views[vi].cam;
    let mut cur = rig.views[vi].look;
    // A first σ from a plain fit.
    let (_, l0) = eval(&cur, &cam, cur.gamma, 1e9);
    cur = l0;
    let resid = |l: &Look| -> Vec<f64> {
        let v = View { cam, look: *l };
        let (img, _, _) = frame(&m, &v, ss, false);
        img.iter().zip(target_lin).map(|(a, b)| a - b.max(0.0).powf(1.0 / l.gamma)).collect()
    };
    let mut sigma = mad_sigma(&resid(&cur), weight);
    if search.key_grid {
        let mut best = (f64::INFINITY, cur);
        for el in [20.0f64, 32.0, 44.0, 56.0, 68.0, 80.0] {
            for az in 0..18 {
                let a = (az as f64 * 20.0).to_radians();
                let e = el.to_radians();
                let mut l = cur;
                l.key_dir = [e.cos() * a.cos(), e.cos() * a.sin(), e.sin()];
                let (c, l2) = eval(&l, &cam, cur.gamma, sigma);
                if c < best.0 {
                    best = (c, l2);
                }
            }
        }
        cur = best.1;
    }
    if search.nonlinear {
        // Coordinate descent: key direction (local), roughness, key
        // radius, gamma.
        let mut best_c = eval(&cur, &cam, cur.gamma, sigma).0;
        for step in [8.0f64, 3.0, 1.0] {
            for _ in 0..3 {
                let mut moved = false;
                let kd = Vec3::new(cur.key_dir[0], cur.key_dir[1], cur.key_dir[2]).normalize();
                let (el, az) = (kd.z.asin(), kd.y.atan2(kd.x));
                for (de, da) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
                    let (e, a) = (el + (de * step).to_radians(), az + (da * step / el.cos().max(0.2)).to_radians());
                    if e > 1.5 || e < 0.1 {
                        continue;
                    }
                    let mut l = cur;
                    l.key_dir = [e.cos() * a.cos(), e.cos() * a.sin(), e.sin()];
                    let (c, l2) = eval(&l, &cam, cur.gamma, sigma);
                    if c < best_c {
                        best_c = c;
                        cur = l2;
                        moved = true;
                    }
                }
                if !moved {
                    break;
                }
            }
        }
        for r in [0.25, 0.35, 0.45, 0.6, 0.8] {
            let mut l = cur;
            l.rough = r;
            let (c, l2) = eval(&l, &cam, cur.gamma, sigma);
            if c < best_c {
                best_c = c;
                cur = l2;
            }
        }
        for kr in [0.04, 0.08, 0.12, 0.2, 0.3] {
            let mut l = cur;
            l.key_radius = kr;
            let (c, l2) = eval(&l, &cam, cur.gamma, sigma);
            if c < best_c {
                best_c = c;
                cur = l2;
            }
        }
        // Gamma changes the target's units: compare the costs in the
        // photo's linear units (the residual's derivative scales by
        // dT'/dT, so judge each by its own σ, as a likelihood would:
        // Σρ + n log σ).
        let n_w = weight.iter().filter(|w| **w > 0.0).count() as f64;
        let mut best_g = (f64::INFINITY, cur, sigma);
        for g in [0.8, 0.9, 1.0, 1.1, 1.25, 1.4] {
            let mut l = cur;
            l.gamma = g;
            let s0 = mad_sigma(&resid(&eval(&l, &cam, g, 1e9).1), weight);
            let (c, l2) = eval(&l, &cam, g, s0);
            // The Jacobian of T' = T^{1/g}: mean over the pixels of log |dT'/dT|.
            let jl: f64 = target_lin.iter().zip(weight).filter(|(_, w)| **w > 0.0).map(|(t, _)| ((1.0 / g) * t.max(1e-4).powf(1.0 / g - 1.0)).ln()).sum();
            let nll = c + n_w * s0.ln() - jl;
            if nll < best_g.0 {
                best_g = (nll, l2, s0);
            }
        }
        cur = best_g.1;
        sigma = best_g.2;
    }
    let (_, lf) = eval(&cur, &cam, cur.gamma, sigma);
    rig.views[vi].look = lf;
    mad_sigma(&resid(&lf), weight)
}

/// Refit view `vi`'s linear appearance (SH, key, albedo) on `prog`
/// against `target` (already in the model's units), robustly.
pub fn refit_linear(prog: &Program, rig: &mut Rig, vi: usize, target: &[f64], weight: &[f64]) {
    let Ok(m) = posed(prog) else { return };
    let v = rig.views[vi].clone();
    let b = bases(&m, &v, rig.ss.min(2));
    let (img, _, _) = frame(&m, &v, rig.ss.min(2), false);
    let r: Vec<f64> = img.iter().zip(target).map(|(a, b)| a - b).collect();
    let sigma = mad_sigma(&r, weight);
    let mut l = v.look;
    look::solve_linear(&b, target, weight, &mut l, sigma, 3);
    rig.views[vi].look = l;
}
