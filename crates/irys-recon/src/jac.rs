//! The image Jacobian `∂I_p/∂θ` for shape parameters: irys's
//! primary-visibility edge term (phase 2d, Li et al. 2018's edge sampling),
//! driven by per-vertex velocities instead of a rigid rig.
//!
//! `vis::EdgeSamples` takes a [`irys_diff::vis::Rig`] — placements that move
//! rigidly — and returns `∂ℒ/∂φ` for one loss. Shape parameters move
//! vertices non-rigidly (a hole's radius scales its wall; a width moves two
//! faces apart), and Levenberg–Marquardt wants the Jacobian itself, per
//! pixel, not one contraction of it. So the estimator is re-stated here
//! over the same pieces: every crease and silhouette edge of the part is
//! projected, clipped and drawn with density ∝ screen length × importance;
//! each sample traces the two sides `s ∓ εn` through irys's points mode
//! ([`irys_diff::prb::Scene::set_points`], one generator per pair: common
//! random numbers), checks the inner ray lands on the edge's own face (else
//! the edge is hidden there and the sample is 0), and adds
//! `(Σℓw / (w_e N)) · (f_in − f_out) · v_n,j` to its pixel's row, with
//! `v_n,j = n · J_π(X) · ∂X/∂θ_j` from [`crate::geom::velocities`].
//!
//! The same samples give `∂coverage/∂θ` (`f` the part's mask: 1 inside, the
//! outer ray's hit outside), which the silhouette (distance-transform) term
//! needs.
//!
//! # The interior term
//!
//! irys's interior term ([`irys_diff::interior`]) carries rigid placements
//! only (its dual tangents are a placement's `R(φ)`), so it does not apply
//! to shape parameters as written. For this material (grey Lambertian, flat
//! shading) a face translating along its own normal leaves the shading at a
//! fixed pixel unchanged except through occlusion and the second bounce, so
//! the interior part is small; it is left out, and the test against central
//! differences ([`tests`] in `tests/jac.rs`) measures what that costs.

use irys_core::{Point3, Vec3};
use irys_diff::prb::Scene;
use irys_light::spectrum::hero_weight;

use crate::render::{self, KernelCache};

/// Half-width of an edge sample, pixels.
pub const EPS_PX: f64 = 1e-3;

/// One view's Jacobians, dense, row-major `[pixel][param]`.
pub struct ViewJac {
    /// `∂grey/∂θ`.
    pub img: Vec<f64>,
    /// `∂coverage/∂θ`.
    pub mask: Vec<f64>,
    /// Parameters.
    pub np: usize,
}

struct Cand {
    /// Owner triangle.
    tri: usize,
    a: usize,
    b: usize,
    /// Third vertex of the owner, for the outward side.
    c: usize,
    weight: f64,
}

fn face_normal(p: &[Point3], t: &[u32; 3]) -> Vec3 {
    let (a, b, c) = (p[t[0] as usize], p[t[1] as usize], p[t[2] as usize]);
    (b - a).cross(&(c - a))
}

/// Candidate edges: every triangle edge that is not shared with exactly one
/// coplanar triangle, owned once (creases between two front faces by the
/// one facing the eye more squarely).
fn candidates(scene: &Scene, p: &[Point3], t: &[[u32; 3]]) -> Vec<Cand> {
    use std::collections::HashMap;
    let q = |v: Point3| ((v.x * 1e4).round() as i64, (v.y * 1e4).round() as i64, (v.z * 1e4).round() as i64);
    let mut weld: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let id: Vec<usize> = p
        .iter()
        .map(|&v| {
            let n = weld.len();
            *weld.entry(q(v)).or_insert(n)
        })
        .collect();
    let mut edges: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
    // Slivers (vcad's repair leaves some along booleans' seams) have no
    // usable normal: they are not neighbours, so the crease beside one is
    // still a candidate (the per-sample test sorts out what it really is).
    let area: Vec<f64> = t.iter().map(|tri| face_normal(p, tri).norm()).collect();
    for (ti, tri) in t.iter().enumerate() {
        if area[ti] < 1e-6 {
            continue;
        }
        for k in 0..3 {
            let (a, b) = (id[tri[k] as usize], id[tri[(k + 1) % 3] as usize]);
            if a == b {
                continue;
            }
            edges.entry((a.min(b), a.max(b))).or_default().push((ti, k));
        }
    }
    let (eye, ..) = scene.camera();
    let normals: Vec<Vec3> = t.iter().map(|tri| face_normal(p, tri)).collect();
    let facing = |ti: usize| {
        let n = normals[ti];
        let c = p[t[ti][0] as usize];
        let nn = n.norm();
        if nn == 0.0 { 0.0 } else { n.dot(&(eye - c)) / (nn * (eye - c).norm()) }
    };
    let mut out = Vec::new();
    for list in edges.values() {
        for (i, &(ti, k)) in list.iter().enumerate() {
            let tri = t[ti];
            let (a, b, c) = (tri[k] as usize, tri[(k + 1) % 3] as usize, tri[(k + 2) % 3] as usize);
            let w = if list.len() == 2 {
                let (tj, _) = list[1 - i];
                let (ni, nj) = (normals[ti], normals[tj]);
                let (li, lj) = (ni.norm(), nj.norm());
                if li == 0.0 || lj == 0.0 {
                    continue;
                }
                let cos = ni.dot(&nj) / (li * lj);
                if cos > 1.0 - 1e-7 {
                    continue; // coplanar: f is continuous across it
                }
                let (fi, fj) = (facing(ti), facing(tj));
                if fi <= 0.0 {
                    continue; // a back face never owns a visible side
                }
                if fj > 0.0 {
                    // A crease between two front faces: the squarer one owns it.
                    if fi < fj || (fi == fj && ti > tj) {
                        continue;
                    }
                    (4.0 * (1.0 - cos)).clamp(0.02, 1.0)
                } else {
                    1.0
                }
            } else {
                if facing(ti) <= 0.0 {
                    continue;
                }
                1.0
            };
            out.push(Cand { tri: ti, a, b, c, weight: w });
        }
    }
    out.sort_by_key(|c| (c.tri, c.a, c.b));
    out
}

/// Clip the screen segment `a → b` to `[0, w] × [0, h]`: the parameter range.
fn clip(a: [f64; 2], b: [f64; 2], w: f64, h: f64) -> Option<(f64, f64)> {
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    let d = [b[0] - a[0], b[1] - a[1]];
    for (p, q) in [(-d[0], a[0]), (d[0], w - a[0]), (-d[1], a[1]), (d[1], h - a[1])] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
        }
    }
    (t0 < t1).then_some((t0, t1))
}

fn ray_hit(scene: &Scene, s: [f64; 2]) -> Option<irys_core::Hit> {
    let (eye, f, r, u, _) = scene.camera();
    scene.tlas().trace_closest(&irys_core::Ray::new(eye, f + r * s[0] + u * s[1])).map(|h| h.hit)
}

fn key(n: &Vec3) -> (i64, i64, i64) {
    ((n.x * 1e6) as i64, (n.y * 1e6) as i64, (n.z * 1e6) as i64)
}

/// SplitMix64, for drawing samples deterministically.
pub struct Mix(pub u64);
impl Mix {
    /// Next uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The edge-term Jacobian of `scene` (posed with the part `p`, `t`) for
/// vertex velocities `vel[vertex][param]`, from `n` edge samples drawn with
/// `seed` (the pairs traced with `seed` too).
#[allow(clippy::too_many_arguments)]
pub fn edge_jacobian(
    scene: &mut Scene,
    p: &[Point3],
    t: &[[u32; 3]],
    vel: &[Vec<Vec3>],
    np: usize,
    n: usize,
    seed: u64,
    k: &mut KernelCache,
) -> ViewJac {
    let (w, h) = scene.size;
    let npx = w * h;
    let mut out = ViewJac { img: vec![0.0; npx * np], mask: vec![0.0; npx * np], np };
    let cands = candidates(scene, p, t);
    // Screen pieces and weights.
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
    let (eye, fwd, ..) = scene.camera();
    let eye0 = eye;
    let mut pieces = Vec::new();
    for (ci, c) in cands.iter().enumerate() {
        let (pa, pb) = (p[c.a], p[c.b]);
        let (Some((sa, _)), Some((sb, _))) = (scene.project(pa), scene.project(pb)) else {
            continue;
        };
        let Some((t0, t1)) = clip(sa, sb, w as f64, h as f64) else {
            continue;
        };
        let len = (sb[0] - sa[0]).hypot(sb[1] - sa[1]) * (t1 - t0);
        if len < 1e-9 {
            continue;
        }
        let (za, zb) = ((pa - eye).dot(&fwd), (pb - eye).dot(&fwd));
        pieces.push(Piece { ci, sa, sb, za, zb, t0, t1, we: len * c.weight });
    }
    if pieces.is_empty() || n == 0 {
        return out;
    }
    let total: f64 = pieces.iter().map(|q| q.we).sum();
    let mut cdf = Vec::with_capacity(pieces.len());
    let mut acc = 0.0;
    for q in &pieces {
        acc += q.we;
        cdf.push(acc / total);
    }
    // Draw: stratified over the CDF.
    struct Sample {
        px: usize,
        wn: Vec<f64>,
    }
    let mut rng = Mix(seed ^ 0x5eed_ed9e);
    let mut samples = Vec::with_capacity(n);
    let mut points = Vec::with_capacity(2 * n);
    for i in 0..n {
        let u = (i as f64 + rng.uniform()) / n as f64;
        let pi = cdf.partition_point(|&c| c < u).min(pieces.len() - 1);
        let q = &pieces[pi];
        let c = &cands[q.ci];
        let us = q.t0 + (q.t1 - q.t0) * rng.uniform();
        let s = [q.sa[0] + us * (q.sb[0] - q.sa[0]), q.sa[1] + us * (q.sb[1] - q.sa[1])];
        // Perspective-correct parameter along the 3D edge.
        let t3 = us * q.za / (us * q.za + (1.0 - us) * q.zb);
        let x = p[c.a] + (p[c.b] - p[c.a]) * t3;
        // Screen normal, away from the owner's third vertex.
        let d = [q.sb[0] - q.sa[0], q.sb[1] - q.sa[1]];
        let dl = d[0].hypot(d[1]);
        let mut nrm = [-d[1] / dl, d[0] / dl];
        let Some((sc, _)) = scene.project(p[c.c]) else { continue };
        if (sc[0] - s[0]) * nrm[0] + (sc[1] - s[1]) * nrm[1] > 0.0 {
            nrm = [-nrm[0], -nrm[1]];
        }
        let Some((_, jp)) = scene.project(x) else { continue };
        // Density of s is importance/Σw (w_e = ℓ·importance), so the
        // weight of ∫ g dℓ is Σw / (importance · n).
        let scale = total / (c.weight * n as f64);
        let wn: Vec<f64> = (0..np)
            .map(|j| {
                let v = vel[c.a][j] * (1.0 - t3) + vel[c.b][j] * t3;
                let sv = [jp[0].dot(&v), jp[1].dot(&v)];
                scale * (nrm[0] * sv[0] + nrm[1] * sv[1])
            })
            .collect();
        let inner = scene.screen(s[0] - EPS_PX * nrm[0], s[1] - EPS_PX * nrm[1]);
        let outer = scene.screen(s[0] + EPS_PX * nrm[0], s[1] + EPS_PX * nrm[1]);
        // Ownership: the inner ray must land on the owner's plane.
        let fnrm = face_normal(p, &t[c.tri]).normalize();
        let plane = (fnrm, fnrm.dot(&p[c.a].coords()));
        if render::hits_plane(scene, inner, plane, 1e-2) != Some(true) {
            continue;
        }
        // Per-sample ownership of a crease: if the outer ray lands on a face
        // adjacent at X (not something farther behind a silhouette) that
        // meets the eye more squarely, the sample is that face's to draw.
        // This is what keeps a crease counted once when the tessellation
        // splits one side of it with T-junctions (unmatched edges).
        if let Some(hit) = ray_hit(scene, outer) {
            if (hit.point - x).norm() < 2e-2 {
                let view = (eye0 - x).normalize();
                let fo = hit.normal.as_ref().dot(&view).abs();
                let fi = fnrm.dot(&view).abs();
                let tie = (fo - fi).abs() < 1e-9;
                if fo > fi + 1e-9 || (tie && key(hit.normal.as_ref()) > key(&fnrm)) {
                    continue;
                }
            }
        }
        let px = (s[1].floor().clamp(0.0, h as f64 - 1.0) as usize) * w + s[0].floor().clamp(0.0, w as f64 - 1.0) as usize;
        samples.push(Sample { px, wn });
        points.push(inner);
        points.push(outer);
    }
    if samples.is_empty() {
        return out;
    }
    // Pad to a fixed 2n paths: the kernels are compiled per graph width, so
    // a width that changed with every call would compile C every call.
    let real = points.len();
    while points.len() < 2 * n {
        points.push(points[real - 2]);
        points.push(points[real - 1]);
    }
    scene.set_points(Some(points.clone()));
    let l = render::trace_points(scene, seed, k);
    scene.set_points(None);
    let (eye, f, r, u, _) = scene.camera();
    for (i, smp) in samples.iter().enumerate() {
        let nw = scene.wavelengths;
        let lam = scene.wavelengths_of(scene.path_rng(seed, 2 * i).uniform());
        let mut df = 0.0;
        for kk in 0..nw {
            let hw = hero_weight(lam[kk]);
            let g = (hw[0] + hw[1] + hw[2]) as f64 / 3.0 / nw as f64;
            df += (l[2 * i * nw + kk] - l[(2 * i + 1) * nw + kk]) * g;
        }
        let o = points[2 * i + 1];
        let out_hit = scene
            .tlas()
            .trace_closest(&irys_core::Ray::new(eye, f + r * o[0] + u * o[1]))
            .is_some();
        let dm = 1.0 - out_hit as u8 as f64;
        for j in 0..np {
            out.img[smp.px * np + j] += smp.wn[j] * df;
            out.mask[smp.px * np + j] += smp.wn[j] * dm;
        }
    }
    out
}
