//! Hair: Chiang, Bitterli, Tappan & Burley 2016 ("A Practical and
//! Controllable Hair and Fur Model for Production Path Tracing") on round
//! fibres, with melanin absorption — the d'Eon/Marschner lobes R, TT, TRT
//! and a residual, a Gaussian-like longitudinal term `M_p` (d'Eon 2011's
//! with the Bessel `I₀`), the trimmed-logistic azimuthal `N_p`, cuticle
//! tilt `α`, and the exact attenuation `A_p` through the fibre.
//!
//! The geometry is a set of polylines swept as round cylinders
//! ([`Strands`], an [`irys_core::Geometry`] under irys's own BVH). Metal's
//! hardware ray queries (wgpu 30's `ray_query`) intersect triangles and
//! AABB-less instances only — no curve primitives — so fibres are traced
//! in software here (design.md, Phase 4).
//!
//! Differentiable in the melanin concentrations `θ = (eumelanin,
//! pheomelanin)`: `σ_a(λ) = θ₀ E(λ) + θ₁ P(λ)` (Chiang's RGB absorption
//! spectra on the three hats), and the [`Hair`] graph is
//! `f = Σ_p (M_p N_p) A_p(F, T)`, `T = e^{−σ_a l}`, with the θ-free
//! `M_p N_p`, the Fresnel `F` and the path length `l` as columns.
//! Sampling (lobe by `A_p`, then `M_p`, then `N_p`) is detached.

use std::f64::consts::PI;

use rayon::prelude::*;
use tang_compile::tensor::{Arg, Form, Monoid, TensorGraph, TensorId, View};
use tang_expr::ExprId;

use irys_core::{Aabb, Bvh, Geometry, Hit, Point2, Point3, Ray, Vec3};
use irys_light::spectrum::{hero_weight, sample_lambda_nm};

use super::openpbr::fresnel;
use crate::ggx::band_weights;
use crate::ir::Program;
use crate::prb::{Rng, mis};
use crate::skin::mesh::basis;
use crate::skin::render::Light;

/// `θ`: eumelanin, pheomelanin concentrations.
pub const PARAMS: usize = 2;
/// [`Hair`] columns: `mn₀ … mn₃, F, l, E(λ), P(λ)`.
pub const COLUMNS: usize = 8;

/// Eumelanin and pheomelanin absorption per unit concentration (Chiang
/// 2016, linear RGB, per fibre radius).
pub const EUMELANIN: [f64; 3] = [0.419, 0.697, 1.37];
#[allow(missing_docs)]
pub const PHEOMELANIN: [f64; 3] = [0.187, 0.4, 1.05];

/// An RGB absorption on the hats (650 nm red, 550 green, 450 blue).
pub fn absorb(rgb: [f64; 3], lambda: f64) -> f64 {
    let w = band_weights(lambda);
    w[0] * rgb[2] + w[1] * rgb[1] + w[2] * rgb[0]
}

/// The fibre's θ-free shape parameters.
#[derive(Debug, Clone, Copy)]
pub struct Fiber {
    /// Index of the cortex.
    pub eta: f64,
    /// Longitudinal and azimuthal roughness in `[0, 1]`.
    pub beta_m: f64,
    #[allow(missing_docs)]
    pub beta_n: f64,
    /// Cuticle scale tilt, degrees.
    pub alpha: f64,
}

impl Default for Fiber {
    fn default() -> Self {
        Self {
            eta: 1.55,
            beta_m: 0.3,
            beta_n: 0.3,
            alpha: 2.0,
        }
    }
}

fn i0(x: f64) -> f64 {
    let (mut v, mut x2i, mut ifact, mut i4) = (0.0, 1.0, 1.0, 1.0);
    for i in 0..10 {
        if i > 1 {
            ifact *= i as f64;
        }
        v += x2i / (i4 * ifact * ifact);
        x2i *= x * x;
        i4 *= 4.0;
    }
    v
}

fn log_i0(x: f64) -> f64 {
    if x > 12.0 {
        x + 0.5 * (-(2.0 * PI).ln() + (1.0 / x).ln() + 1.0 / (8.0 * x))
    } else {
        i0(x).ln()
    }
}

/// d'Eon's longitudinal scattering.
pub fn mp(cti: f64, cto: f64, sti: f64, sto: f64, v: f64) -> f64 {
    let a = cti * cto / v;
    let b = sti * sto / v;
    if v <= 0.1 {
        (log_i0(a) - b - 1.0 / v + 0.6931 + (1.0 / (2.0 * v)).ln()).exp()
    } else {
        (-b).exp() * i0(a) / ((1.0 / v).sinh() * 2.0 * v)
    }
}

fn logistic(x: f64, s: f64) -> f64 {
    let x = x.abs();
    let e = (-x / s).exp();
    e / (s * (1.0 + e) * (1.0 + e))
}

fn logistic_cdf(x: f64, s: f64) -> f64 {
    1.0 / (1.0 + (-x / s).exp())
}

fn trimmed_logistic(x: f64, s: f64) -> f64 {
    logistic(x, s) / (logistic_cdf(PI, s) - logistic_cdf(-PI, s))
}

fn sample_trimmed_logistic(u: f64, s: f64) -> f64 {
    let k = logistic_cdf(PI, s) - logistic_cdf(-PI, s);
    let x = -s * (1.0 / (u * k + logistic_cdf(-PI, s)) - 1.0).ln();
    x.clamp(-PI, PI)
}

fn phi(p: usize, gamma_o: f64, gamma_t: f64) -> f64 {
    2.0 * p as f64 * gamma_t - 2.0 * gamma_o + p as f64 * PI
}

fn np(dphi: f64, p: usize, s: f64, go: f64, gt: f64) -> f64 {
    let mut d = dphi - phi(p, go, gt);
    while d > PI {
        d -= 2.0 * PI;
    }
    while d < -PI {
        d += 2.0 * PI;
    }
    trimmed_logistic(d, s)
}

/// The fibre at one hit: everything θ-free, for `ω_o` in the hair frame
/// (`x` along the fibre, `z` toward the viewer, `h ∈ [−1, 1]` across).
#[derive(Debug, Clone, Copy)]
pub struct Bcsdf {
    v: [f64; 4],
    s: f64,
    sin2k: [f64; 3],
    cos2k: [f64; 3],
    h: f64,
    gamma_o: f64,
    eta: f64,
}

impl Bcsdf {
    /// The fibre's lobes at offset `h`.
    pub fn new(f: &Fiber, h: f64) -> Self {
        let bm = f.beta_m;
        let v0 = (0.726 * bm + 0.812 * bm * bm + 3.7 * bm.powi(20)).powi(2);
        let bn = f.beta_n;
        let s = (PI / 8.0).sqrt() * (0.265 * bn + 1.194 * bn * bn + 5.372 * bn.powi(22));
        let mut sin2k = [0.0; 3];
        let mut cos2k = [0.0; 3];
        sin2k[0] = f.alpha.to_radians().sin();
        cos2k[0] = (1.0 - sin2k[0] * sin2k[0]).max(0.0).sqrt();
        for i in 1..3 {
            sin2k[i] = 2.0 * cos2k[i - 1] * sin2k[i - 1];
            cos2k[i] = cos2k[i - 1] * cos2k[i - 1] - sin2k[i - 1] * sin2k[i - 1];
        }
        let h = h.clamp(-0.999_999, 0.999_999);
        Self {
            v: [v0, 0.25 * v0, 4.0 * v0, 4.0 * v0],
            s,
            sin2k,
            cos2k,
            h,
            gamma_o: h.asin(),
            eta: f.eta,
        }
    }

    fn tilt(&self, p: usize, sto: f64, cto: f64) -> (f64, f64) {
        let (s, c) = match p {
            0 => (
                sto * self.cos2k[1] - cto * self.sin2k[1],
                cto * self.cos2k[1] + sto * self.sin2k[1],
            ),
            1 => (
                sto * self.cos2k[0] + cto * self.sin2k[0],
                cto * self.cos2k[0] - sto * self.sin2k[0],
            ),
            2 => (
                sto * self.cos2k[2] + cto * self.sin2k[2],
                cto * self.cos2k[2] - sto * self.sin2k[2],
            ),
            _ => (sto, cto),
        };
        (s, c.abs())
    }

    fn geometry(&self, wo: [f64; 3]) -> (f64, f64, f64, f64, f64, f64) {
        let sto = wo[0];
        let cto = (1.0 - sto * sto).max(0.0).sqrt();
        let phio = wo[2].atan2(wo[1]);
        let stt = sto / self.eta;
        let ctt = (1.0 - stt * stt).max(0.0).sqrt();
        let etap = (self.eta * self.eta - sto * sto).sqrt() / cto.max(1e-12);
        let sgt = self.h / etap;
        let cgt = (1.0 - sgt * sgt).max(0.0).sqrt();
        let gt = sgt.asin();
        let f = fresnel(cto * self.gamma_o.cos(), self.eta);
        let l = 2.0 * cgt / ctt.max(1e-12);
        (sto, cto, phio, gt, f, l)
    }

    /// A lane's columns for `(ω_o, ω_i)` at `lambda`.
    pub fn columns(&self, wo: [f64; 3], wi: [f64; 3], lambda: f64) -> [f64; COLUMNS] {
        let (sto, cto, phio, gt, f, l) = self.geometry(wo);
        let sti = wi[0];
        let cti = (1.0 - sti * sti).max(0.0).sqrt();
        let phii = wi[2].atan2(wi[1]);
        let dphi = phii - phio;
        let mut c = [0.0; COLUMNS];
        for p in 0..3 {
            let (s, cc) = self.tilt(p, sto, cto);
            c[p] = mp(cti, cc, sti, s, self.v[p]) * np(dphi, p, self.s, self.gamma_o, gt);
        }
        c[3] = mp(cti, cto, sti, sto, self.v[3]) / (2.0 * PI);
        c[4] = f;
        c[5] = l;
        c[6] = absorb(EUMELANIN, lambda);
        c[7] = absorb(PHEOMELANIN, lambda);
        c
    }

    /// `A_p` at `θ` (f64) — what the sampler picks a lobe by.
    pub fn ap(&self, wo: [f64; 3], lambda: f64, theta: &[f64; PARAMS]) -> [f64; 4] {
        let (_, _, _, _, f, l) = self.geometry(wo);
        let sa = theta[0] * absorb(EUMELANIN, lambda) + theta[1] * absorb(PHEOMELANIN, lambda);
        let t = (-sa * l).exp();
        let a1 = (1.0 - f) * (1.0 - f) * t;
        let a2 = a1 * f * t;
        [f, a1, a2, a2 * f * t / (1.0 - t * f)]
    }

    /// The pdf of `ω_i` (detached; `ap` at the sampler's `θ`).
    pub fn pdf(&self, wo: [f64; 3], wi: [f64; 3], ap: &[f64; 4]) -> f64 {
        let (sto, cto, phio, gt, _, _) = self.geometry(wo);
        let sum: f64 = ap.iter().sum();
        let sti = wi[0];
        let cti = (1.0 - sti * sti).max(0.0).sqrt();
        let dphi = wi[2].atan2(wi[1]) - phio;
        let mut pdf = 0.0;
        for p in 0..3 {
            let (s, c) = self.tilt(p, sto, cto);
            pdf += mp(cti, c, sti, s, self.v[p]) * ap[p] / sum * np(dphi, p, self.s, self.gamma_o, gt);
        }
        pdf + mp(cti, cto, sti, sto, self.v[3]) * ap[3] / sum / (2.0 * PI)
    }

    /// Sample `ω_i` (detached).
    pub fn sample(&self, wo: [f64; 3], ap: &[f64; 4], u: [f64; 4]) -> [f64; 3] {
        let (sto, cto, phio, gt, _, _) = self.geometry(wo);
        let sum: f64 = ap.iter().sum();
        let mut p = 3;
        let mut acc = 0.0;
        for (k, a) in ap.iter().enumerate() {
            acc += a / sum;
            if u[0] < acc {
                p = k;
                break;
            }
        }
        let (s, c) = self.tilt(p, sto, cto);
        let v = self.v[p];
        let u1 = u[1].max(1e-5);
        let ct = 1.0 + v * (u1 + (1.0 - u1) * (-2.0 / v).exp()).ln();
        let st = (1.0 - ct * ct).max(0.0).sqrt();
        let cphi = (2.0 * PI * u[2]).cos();
        let sti = (-ct * s + st * cphi * c).clamp(-1.0, 1.0);
        let cti = (1.0 - sti * sti).max(0.0).sqrt();
        let dphi = if p < 3 {
            phi(p, self.gamma_o, gt) + sample_trimmed_logistic(u[3], self.s)
        } else {
            2.0 * PI * u[3]
        };
        let phii = phio + dphi;
        [sti, cti * phii.cos(), cti * phii.sin()]
    }
}

/// `f` of a lane in f64 (the graph's twin).
pub fn hair_f(c: &[f64; COLUMNS], t: &[f64; PARAMS]) -> f64 {
    let [m0, m1, m2, m3, f, l, e, p] = *c;
    let tt = (-(t[0] * e + t[1] * p) * l).exp();
    let a1 = (1.0 - f) * (1.0 - f) * tt;
    let a2 = a1 * f * tt;
    m0 * f + m1 * a1 + m2 * a2 + m3 * a2 * f * tt / (1.0 - tt * f)
}

/// The BCSDF graph over `n` lanes: inputs the eight columns, `θ`, `δf`;
/// outputs `f` and `δθ`.
pub struct Hair {
    /// The graph.
    pub g: TensorGraph,
    /// `f`.
    pub f: TensorId,
    /// `δθ`.
    pub dtheta: TensorId,
}

impl Hair {
    /// Build it.
    pub fn new(n: usize) -> Self {
        let mut g = TensorGraph::new();
        let cols: Vec<TensorId> = (0..COLUMNS).map(|_| g.input(&[n])).collect();
        let theta = g.input(&[PARAMS]);
        let df = g.input(&[n]);
        let mut args: Vec<Arg> = cols
            .iter()
            .map(|&c| Arg {
                src: c,
                view: View::identity(1, 1),
            })
            .collect();
        args.extend((0..PARAMS as i64).map(|k| Arg {
            src: theta,
            view: View {
                guard: false,
                forms: vec![Form {
                    coefs: vec![0],
                    offset: k,
                }],
            },
        }));
        let f = g
            .comp(&[n], &[] as &[usize], args, None, |x: &[ExprId]| {
                let one = <ExprId as tang::Scalar>::from_f64(1.0);
                let (m0, m1, m2, m3, fr, l, e, p) = (x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
                let tt = <ExprId as tang::Scalar>::exp(-((x[8] * e + x[9] * p) * l));
                let a1 = (one - fr) * (one - fr) * tt;
                let a2 = a1 * fr * tt;
                vec![m0 * fr + m1 * a1 + m2 * a2 + m3 * a2 * fr * tt / (one - tt * fr)]
            })
            .unwrap();
        let pick = |t: TensorId| Arg {
            src: t,
            view: View::pick(2, &[Some(1)]),
        };
        let s = g
            .comp(
                &[1usize],
                &[n],
                vec![pick(df), pick(f)],
                Some(Monoid::sum()),
                |x| vec![x[0] * x[1]],
            )
            .unwrap();
        let dtheta = g.backward(s, &[theta]).unwrap()[0];
        Self { g, f, dtheta }
    }

    /// The eval lowering.
    pub fn eval(&self, native: bool) -> Program {
        let o = [self.f];
        if native { Program::compile(&self.g, &o) } else { Program::interpreted(&self.g, &o) }
    }

    /// The adjoint lowering.
    pub fn adjoint(&self, native: bool) -> Program {
        let o = [self.dtheta];
        if native { Program::compile(&self.g, &o) } else { Program::interpreted(&self.g, &o) }
    }
}

/// Reduce `(columns, δf)` lanes through the adjoint, in order.
pub fn reduce(prog: &mut Program, width: usize, lanes: &[([f64; COLUMNS], f64)], theta: &[f64; PARAMS]) -> [f64; PARAMS] {
    let mut out = [0.0; PARAMS];
    prog.write(COLUMNS, theta);
    for chunk in lanes.chunks(width) {
        for k in 0..COLUMNS {
            let mut v = vec![0.0; width];
            for (i, e) in chunk.iter().enumerate() {
                v[i] = e.0[k];
            }
            prog.write(k, &v);
        }
        let mut d = vec![0.0; width];
        for (i, e) in chunk.iter().enumerate() {
            d[i] = e.1;
        }
        prog.write(COLUMNS + 1, &d);
        prog.run();
        let g = prog.read(0);
        for k in 0..PARAMS {
            out[k] += g[k];
        }
    }
    out
}

// ─── geometry ────────────────────────────────────────────────────────────

/// Fibres as round cylinders between consecutive polyline points.
#[derive(Debug, Clone, Default)]
pub struct Strands {
    /// Segment endpoints and radius.
    pub segs: Vec<(Point3, Point3, f64)>,
}

impl Strands {
    /// Add a polyline of constant radius.
    pub fn push(&mut self, pts: &[Point3], r: f64) {
        for w in pts.windows(2) {
            self.segs.push((w[0], w[1], r));
        }
    }

    /// The unit tangent of segment `i`.
    pub fn tangent(&self, i: usize) -> Vec3 {
        let (a, b, _) = self.segs[i];
        (b - a).normalize()
    }
}

impl Geometry for Strands {
    fn len(&self) -> usize {
        self.segs.len()
    }

    fn bounds(&self, i: usize) -> Aabb {
        let (a, b, r) = self.segs[i];
        let lo = Point3::new(a.x.min(b.x) - r, a.y.min(b.y) - r, a.z.min(b.z) - r);
        let hi = Point3::new(a.x.max(b.x) + r, a.y.max(b.y) + r, a.z.max(b.z) + r);
        Aabb::new(lo, hi)
    }

    fn intersect(&self, ray: &Ray, i: usize, t_min: f64, t_max: f64) -> Option<Hit> {
        let (a, b, r) = self.segs[i];
        let ax = b - a;
        let len = ax.norm();
        let t_ax = ax / len;
        let d = *ray.direction;
        let oc = ray.origin - a;
        let dp = d - t_ax * d.dot(&t_ax);
        let op = oc - t_ax * oc.dot(&t_ax);
        let qa = dp.norm_squared();
        if qa < 1e-18 {
            return None;
        }
        let qb = 2.0 * dp.dot(&op);
        let qc = op.norm_squared() - r * r;
        let disc = qb * qb - 4.0 * qa * qc;
        if disc < 0.0 {
            return None;
        }
        let sq = disc.sqrt();
        for t in [(-qb - sq) / (2.0 * qa), (-qb + sq) / (2.0 * qa)] {
            if t <= t_min || t >= t_max {
                continue;
            }
            let p = ray.origin + d * t;
            let s = (p - a).dot(&t_ax);
            if s < 0.0 || s > len {
                continue;
            }
            let n = (p - (a + t_ax * s)).normalize();
            return Some(Hit::new(t, p, irys_core::Dir3::new_normalize(n), Point2::new(s / len, 0.0), i as u32));
        }
        None
    }
}

/// A hair scene: a diffuse sphere (the scalp) with strands, quads, a
/// constant environment, a pinhole camera.
pub struct HairScene {
    /// The fibres and their tree.
    pub bvh: Bvh<Strands>,
    /// Fibre shape.
    pub fiber: Fiber,
    /// Scalp: centre, radius, linear-sRGB albedo (`None`: no scalp).
    pub scalp: Option<(Point3, f64, [f64; 3])>,
    /// Emitters, environment.
    pub lights: Vec<Light>,
    #[allow(missing_docs)]
    pub env: f64,
    /// Camera.
    pub eye: Point3,
    #[allow(missing_docs)]
    pub target: Point3,
    #[allow(missing_docs)]
    pub vfov: f64,
    /// Pixels, spp, depth.
    pub size: (usize, usize),
    #[allow(missing_docs)]
    pub spp: usize,
    #[allow(missing_docs)]
    pub depth: usize,
    /// Frozen sampler `θ_s`.
    pub sampler: Option<[f64; PARAMS]>,
}

enum Surf {
    Hair(Hit),
    Scalp(Point3, Vec3),
    Light(usize),
}

impl HairScene {
    /// Paths per frame.
    pub fn paths(&self) -> usize {
        self.size.0 * self.size.1 * self.spp
    }

    fn scalp_t(&self, o: Point3, d: Vec3) -> Option<f64> {
        let (c, r, _) = self.scalp?;
        let oc = o - c;
        let b = oc.dot(&d);
        let q = oc.norm_squared() - r * r;
        let disc = b * b - q;
        if disc < 0.0 {
            return None;
        }
        let s = disc.sqrt();
        [-b - s, -b + s].into_iter().find(|&t| t > 1e-7)
    }

    fn closest(&self, o: Point3, d: Vec3) -> Option<(f64, Surf)> {
        let mut best: Option<(f64, Surf)> = None;
        if let Some(h) = self.bvh.trace_closest_range(&Ray::new(o, d), 1e-7, f64::INFINITY) {
            best = Some((h.t, Surf::Hair(h)));
        }
        if let Some(t) = self.scalp_t(o, d) {
            if best.as_ref().is_none_or(|b| t < b.0) {
                let x = o + d * t;
                let n = (x - self.scalp.unwrap().0).normalize();
                best = Some((t, Surf::Scalp(x, n)));
            }
        }
        let tb = best.as_ref().map_or(f64::INFINITY, |b| b.0);
        for (i, q) in self.lights.iter().enumerate() {
            if let Some(t) = q.hit(o, d, tb) {
                if best.as_ref().is_none_or(|b| t < b.0) {
                    best = Some((t, Surf::Light(i)));
                }
            }
        }
        best
    }

    fn visible(&self, o: Point3, d: Vec3, dist: f64) -> bool {
        let r = Ray::new(o, d);
        if self.bvh.occluded_range(&r, 1e-7, dist) {
            return false;
        }
        self.scalp_t(o, d).is_none_or(|t| t >= dist)
    }

    fn sample_light(&self, x: Point3, u: [f64; 3]) -> Option<(Vec3, f64, f64, usize)> {
        let n = self.lights.len();
        let i = ((u[0] * n as f64) as usize).min(n - 1);
        let q = &self.lights[i];
        let p = q.center + q.u * (2.0 * u[1] - 1.0) + q.v * (2.0 * u[2] - 1.0);
        let dv = p - x;
        let d2 = dv.norm_squared();
        let d = dv / d2.sqrt();
        let cl = -d.dot(&q.normal());
        (cl > 0.0).then(|| (d, d2.sqrt(), d2 / (cl * q.area() * n as f64), i))
    }

    fn light_pdf(&self, x: Point3, d: Vec3, i: usize, t: f64) -> f64 {
        let q = &self.lights[i];
        let _ = x;
        t * t / (-d.dot(&q.normal()) * q.area() * self.lights.len() as f64)
    }

    /// One path; `rec` gets `(columns, δf/δL)` per hair lane.
    pub fn trace(
        &self,
        theta: &[f64; PARAMS],
        seed: u64,
        path: usize,
        rec: Option<&mut Vec<([f64; COLUMNS], f64)>>,
    ) -> (f64, f64) {
        let mut rng = Rng::new(seed, path as u64);
        let lambda = sample_lambda_nm(((path % self.spp) as f64 + rng.uniform()) / self.spp as f64);
        let ts = self.sampler.unwrap_or(*theta);
        let (w, h) = self.size;
        let px = path / self.spp;
        let (ix, iy) = ((px % w) as f64 + rng.uniform(), (px / w) as f64 + rng.uniform());
        let fwd = (self.target - self.eye).normalize();
        let right = fwd.cross(&Vec3::new(0.0, 0.0, 1.0)).normalize();
        let up = right.cross(&fwd);
        let th = (0.5 * self.vfov).to_radians().tan();
        let asp = w as f64 / h as f64;
        let mut d = (fwd + right * ((2.0 * ix / w as f64 - 1.0) * th * asp)
            + up * ((1.0 - 2.0 * iy / h as f64) * th))
            .normalize();
        let mut o = self.eye;
        let mut beta = 1.0;
        let mut l = 0.0;
        let mut last_pdf: Option<f64> = None;
        // (columns, f, L before, nee contribution or NaN)
        let mut lanes: Vec<([f64; COLUMNS], f64, f64, f64)> = Vec::new();
        let recording = rec.is_some();
        for _ in 0..self.depth {
            let Some((t, s)) = self.closest(o, d) else {
                l += beta * self.env;
                break;
            };
            match s {
                Surf::Light(i) => {
                    let pl = self.light_pdf(o, d, i, t);
                    l += beta * self.lights[i].le(lambda) * last_pdf.map_or(1.0, |pb| mis(pb, pl));
                    break;
                }
                Surf::Scalp(x, n) => {
                    let rho = super::openpbr::lift(self.scalp.unwrap().2, lambda);
                    let un = [rng.uniform(), rng.uniform(), rng.uniform()];
                    if let Some((dl, dist, pl, i)) = self.sample_light(x, un) {
                        let c = dl.dot(&n);
                        if c > 0.0 && self.visible(x + n * 1e-6, dl, dist) {
                            l += beta * rho / PI * c * self.lights[i].le(lambda) / pl * mis(pl, c / PI);
                        }
                    }
                    let (tu, tv) = basis(n);
                    let cd = super::openpbr::cosine([rng.uniform(), rng.uniform()]);
                    d = tu * cd[0] + tv * cd[1] + n * cd[2];
                    o = x + n * 1e-6;
                    beta *= rho;
                    last_pdf = Some(cd[2] / PI);
                }
                Surf::Hair(hit) => {
                    let x = hit.point;
                    let tx = self.bvh.geometry().tangent(hit.prim as usize);
                    let mut z = -(d - tx * d.dot(&tx));
                    if z.norm() < 1e-9 {
                        break;
                    }
                    z = z.normalize();
                    let y = z.cross(&tx);
                    let r = self.bvh.geometry().segs[hit.prim as usize].2;
                    // The offset across the fibre as seen from the ray.
                    let hh = (*hit.normal).dot(&y).clamp(-1.0, 1.0);
                    let loc = |v: Vec3| [v.dot(&tx), v.dot(&y), v.dot(&z)];
                    let wo = loc(-d);
                    let bc = Bcsdf::new(&self.fiber, hh);
                    let ap_s = bc.ap(wo, lambda, &ts);
                    let un = [rng.uniform(), rng.uniform(), rng.uniform()];
                    let off = *hit.normal * (2.0 * r);
                    if let Some((dl, dist, pl, i)) = self.sample_light(x, un) {
                        let side = if dl.dot(&hit.normal) > 0.0 { off } else { -off };
                        if self.visible(x + side, dl, dist) {
                            let wi = loc(dl);
                            let cols = bc.columns(wo, wi, lambda);
                            let f = hair_f(&cols, theta);
                            let pb = bc.pdf(wo, wi, &ap_s);
                            let c = beta * f * self.lights[i].le(lambda) / pl * mis(pl, pb);
                            l += c;
                            if recording {
                                lanes.push((cols, f, 0.0, c));
                            }
                        }
                    }
                    let us = [rng.uniform(), rng.uniform(), rng.uniform(), rng.uniform()];
                    let wi = bc.sample(wo, &ap_s, us);
                    let pdf = bc.pdf(wo, wi, &ap_s);
                    if !(pdf > 0.0) {
                        break;
                    }
                    let cols = bc.columns(wo, wi, lambda);
                    let f = hair_f(&cols, theta);
                    if recording {
                        lanes.push((cols, f, l, f64::NAN));
                    }
                    beta *= f / pdf;
                    d = (tx * wi[0] + y * wi[1] + z * wi[2]).normalize();
                    o = x + if d.dot(&hit.normal) > 0.0 { off } else { -off };
                    last_pdf = Some(pdf);
                }
            }
            if !(beta > 0.0) || !beta.is_finite() {
                break;
            }
        }
        if let Some(r) = rec {
            for (c, f, before, nee) in lanes {
                let wgt = if nee.is_nan() { l - before } else { nee };
                r.push((c, if f != 0.0 { wgt / f } else { 0.0 }));
            }
        }
        (l, lambda)
    }

    /// Forward.
    pub fn forward(&self, theta: &[f64; PARAMS], seed: u64) -> Vec<(f64, f64)> {
        (0..self.paths())
            .into_par_iter()
            .map(|i| self.trace(theta, seed, i, None))
            .collect()
    }

    /// Linear sRGB.
    pub fn image(&self, l: &[(f64, f64)]) -> Vec<f64> {
        let mut img = vec![0.0; self.size.0 * self.size.1 * 3];
        for (j, &(lj, lambda)) in l.iter().enumerate() {
            let w = hero_weight(lambda);
            let px = j / self.spp;
            for c in 0..3 {
                img[px * 3 + c] += lj * w[c] as f64 / self.spp as f64;
            }
        }
        img
    }

    /// `(loss, δθ, L of the replay)` by PRB.
    pub fn gradient(
        &self,
        theta: &[f64; PARAMS],
        seed: u64,
        loss: &impl crate::prb::Loss,
        native: bool,
    ) -> (f64, [f64; PARAMS], Vec<f64>) {
        let fwd = self.forward(theta, seed);
        let (v, d_img) = loss.eval(&self.image(&fwd));
        let per: Vec<(f64, Vec<([f64; COLUMNS], f64)>)> = (0..self.paths())
            .into_par_iter()
            .map(|i| {
                let mut r = Vec::new();
                let (l, _) = self.trace(theta, seed, i, Some(&mut r));
                (l, r)
            })
            .collect();
        let mut lanes = Vec::new();
        let mut ls = Vec::new();
        for (j, (l, r)) in per.into_iter().enumerate() {
            let w = hero_weight(fwd[j].1);
            let px = j / self.spp;
            let dl: f64 = (0..3).map(|c| d_img[px * 3 + c] * w[c] as f64 / self.spp as f64).sum();
            ls.push(l);
            lanes.extend(r.into_iter().map(|(c, x)| (c, dl * x)));
        }
        let width = 4096;
        let mut prog = Hair::new(width).adjoint(native);
        (v, reduce(&mut prog, width, &lanes, theta), ls)
    }
}
