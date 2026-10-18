//! Participating media: homogeneous and gridded density in a box, spectral
//! `σ`, Henyey–Greenstein, delta tracking for free flight, ratio tracking
//! for the transmittance of a light sample, an emission grid, and path
//! replay into `θ = (density scale, albedo on the three hats)`.
//!
//! # The estimator
//!
//! `σ_t(x, λ) = s · d(x) · c(λ)`, `σ_s = α(λ) σ_t` with `α = Σ_k b_k(λ) θ_{1+k}`.
//! Distances are drawn against a majorant `μ̄` fixed by the *sampler's*
//! medium (detached), and a tentative collision at `x` is real with
//! probability `σ_t^s(x)/μ̄`. The weights are the frozen-sampler ratios
//!
//! ```text
//! real:  α σ_t(x) / σ_t^s(x)        (a scatter; absorption is the α)
//! null:  (μ̄ − σ_t(x)) / (μ̄ − σ_t^s(x))
//! ```
//!
//! so at `θ = θ_s` they are `α` and `1` — analog delta tracking with
//! absorption folded into the weight. A light sample's transmittance is
//! ratio tracking, `Π (μ̄ − σ_t)/μ̄` over the tentative collisions. Every
//! θ-dependent factor is one [`VolEvent`] lane, `ℓ = real · ln(α s d c) +
//! null · ln(μ̄ − s d c)`; PRB owes each `δL · W`, with `W = L − L_j` for a
//! path factor and the contribution `C` for a ratio-tracking factor.
//! Emission (`e(x) L_e`, its own grid, θ-free) is collected at every
//! tentative collision as `β e(x) L_e / μ̄`, which is unbiased for
//! `∫ T e L_e dt`.

use rayon::prelude::*;

use tang_compile::tensor::{Arg, Form, Monoid, TensorGraph, TensorId, View};
use tang_expr::ExprId;

use irys_core::{Point3, Vec3};
use irys_light::spectrum::{hero_weight, sample_lambda_nm};

use super::openpbr::lift;
use crate::ggx::band_weights;
use crate::ir::Program;
use crate::prb::{Rng, mis};
use crate::skin::mesh::basis;
use crate::skin::render::{Light, hg_sample};

const PI: f64 = std::f64::consts::PI;

/// Parameters: density scale, albedo at 450, 550, 650 nm.
pub const PARAMS: usize = 4;
/// [`VolEvent`] columns: `real, null, d, c, μ̄, b₀, b₁, b₂`.
pub const COLUMNS: usize = 8;

/// A density field over a box.
#[derive(Debug, Clone)]
pub struct Grid {
    /// Box corners.
    pub lo: Point3,
    #[allow(missing_docs)]
    pub hi: Point3,
    /// Resolution; `[1, 1, 1]` is homogeneous.
    pub n: [usize; 3],
    /// `n₀ n₁ n₂` values at cell centres, `x` fastest.
    pub data: Vec<f64>,
    /// [`Self::max`], computed once (a path's [`Track`] asks for it; at
    /// 96³ the scan cost more than the path).
    vmax: std::sync::OnceLock<f64>,
}

impl Grid {
    /// A constant field.
    pub fn constant(lo: Point3, hi: Point3, v: f64) -> Self {
        Self {
            lo,
            hi,
            n: [1, 1, 1],
            data: vec![v],
            vmax: Default::default(),
        }
    }

    /// A field sampled from `f` over the unit cube of the box.
    pub fn from_fn(lo: Point3, hi: Point3, n: [usize; 3], f: impl Fn([f64; 3]) -> f64) -> Self {
        let mut data = Vec::with_capacity(n[0] * n[1] * n[2]);
        for k in 0..n[2] {
            for j in 0..n[1] {
                for i in 0..n[0] {
                    data.push(f([
                        (i as f64 + 0.5) / n[0] as f64,
                        (j as f64 + 0.5) / n[1] as f64,
                        (k as f64 + 0.5) / n[2] as f64,
                    ]));
                }
            }
        }
        Self { lo, hi, n, data, vmax: Default::default() }
    }

    /// Trilinear value at `x` (0 outside the box).
    pub fn at(&self, x: Point3) -> f64 {
        let e = self.hi - self.lo;
        let u = [(x.x - self.lo.x) / e.x, (x.y - self.lo.y) / e.y, (x.z - self.lo.z) / e.z];
        if !(u[0] >= 0.0 && u[0] <= 1.0 && u[1] >= 0.0 && u[1] <= 1.0 && u[2] >= 0.0 && u[2] <= 1.0) {
            return 0.0;
        }
        if self.data.len() == 1 {
            return self.data[0];
        }
        let mut idx = [0usize; 3];
        let mut t = [0.0; 3];
        for a in 0..3 {
            let s = (u[a] * self.n[a] as f64 - 0.5).clamp(0.0, (self.n[a] - 1) as f64);
            let i = (s.floor() as usize).min(self.n[a].saturating_sub(2));
            idx[a] = i;
            t[a] = if self.n[a] > 1 { s - i as f64 } else { 0.0 };
        }
        let at = |i: usize, j: usize, k: usize| {
            let (i, j, k) = (i.min(self.n[0] - 1), j.min(self.n[1] - 1), k.min(self.n[2] - 1));
            self.data[(k * self.n[1] + j) * self.n[0] + i]
        };
        let mut v = 0.0;
        for dk in 0..2 {
            for dj in 0..2 {
                for di in 0..2 {
                    let w = (if di == 1 { t[0] } else { 1.0 - t[0] })
                        * (if dj == 1 { t[1] } else { 1.0 - t[1] })
                        * (if dk == 1 { t[2] } else { 1.0 - t[2] });
                    v += w * at(idx[0] + di, idx[1] + dj, idx[2] + dk);
                }
            }
        }
        v
    }

    /// The largest value (cached on first use: do not edit `data` after).
    pub fn max(&self) -> f64 {
        *self.vmax.get_or_init(|| self.data.iter().cloned().fold(0.0, f64::max))
    }

    /// Entry and exit parameters of a ray through the box.
    pub fn span(&self, o: Point3, d: Vec3) -> Option<(f64, f64)> {
        let (mut t0, mut t1) = (0.0f64, f64::INFINITY);
        for a in 0..3 {
            let g3p = |p: Point3| [p.x, p.y, p.z][a];
            let (oa, da, lo, hi) = (g3p(o), [d.x, d.y, d.z][a], g3p(self.lo), g3p(self.hi));
            if da.abs() < 1e-300 {
                if oa < lo || oa > hi {
                    return None;
                }
                continue;
            }
            let (mut ta, mut tb) = ((lo - oa) / da, (hi - oa) / da);
            if ta > tb {
                std::mem::swap(&mut ta, &mut tb);
            }
            t0 = t0.max(ta);
            t1 = t1.min(tb);
        }
        (t1 > t0).then_some((t0, t1))
    }
}

/// A medium in a box (index-matched boundary).
#[derive(Debug, Clone)]
pub struct Medium {
    /// Density.
    pub density: Grid,
    /// `c(λ)`: the extinction's spectral shape (linear sRGB on the hats),
    /// per scene unit at density 1.
    pub sigma: [f64; 3],
    /// Henyey–Greenstein asymmetry.
    pub g: f64,
    /// Emission density (same box) and its linear-sRGB radiance.
    pub emission: Option<(Grid, [f64; 3])>,
    /// Per-supervoxel majorants ([`Medium::with_majorant_grid`]); `None`
    /// tracks against one global majorant.
    pub majorant: Option<MajorantGrid>,
}

/// A coarse grid of the density's maximum over each supervoxel (and
/// whether any emission is there), for delta and ratio tracking against a
/// piecewise-constant majorant (Novák, Georgiev, Hanika & Jarosz 2018,
/// §3.3; Kutz+ 2017). θ-free: built once from the density.
#[derive(Debug, Clone)]
pub struct MajorantGrid {
    /// Supervoxels per axis.
    pub n: [usize; 3],
    /// The largest density any trilinear lookup inside the supervoxel can
    /// return (the max over every data cell its interpolation touches).
    pub dmax: Vec<f64>,
    /// Some emission in the supervoxel.
    pub emissive: Vec<bool>,
}

impl MajorantGrid {
    /// Build at `r` supervoxels per axis (clamped to the data resolution).
    pub fn new(density: &Grid, emission: Option<&Grid>, r: usize) -> Self {
        let n = [0, 1, 2].map(|a| r.clamp(1, density.n[a].max(1)));
        let range = |g: &Grid, a: usize, i: usize| -> (usize, usize) {
            let m = g.n[a];
            if m == 1 {
                return (0, 0);
            }
            let lo = (i as f64 / n[a] as f64 * m as f64 - 0.5).floor().max(0.0) as usize;
            let hi = (((i + 1) as f64 / n[a] as f64 * m as f64 - 0.5).floor() + 1.0).max(0.0) as usize;
            (lo.min(m - 1), hi.min(m - 1))
        };
        let gmax = |g: &Grid, c: [usize; 3]| -> f64 {
            if g.data.len() == 1 {
                return g.data[0];
            }
            let (r0, r1, r2) = (range(g, 0, c[0]), range(g, 1, c[1]), range(g, 2, c[2]));
            let mut v: f64 = 0.0;
            for k in r2.0..=r2.1 {
                for j in r1.0..=r1.1 {
                    for i in r0.0..=r0.1 {
                        v = v.max(g.data[(k * g.n[1] + j) * g.n[0] + i]);
                    }
                }
            }
            v
        };
        let mut dmax = Vec::with_capacity(n[0] * n[1] * n[2]);
        let mut emissive = Vec::with_capacity(n[0] * n[1] * n[2]);
        for k in 0..n[2] {
            for j in 0..n[1] {
                for i in 0..n[0] {
                    dmax.push(gmax(density, [i, j, k]));
                    // The emission grid may have another resolution: map the
                    // supervoxel's box onto it.
                    emissive.push(emission.is_some_and(|e| {
                        let sub = |a: usize, i: usize| -> (usize, usize) {
                            let m = e.n[a];
                            if m == 1 {
                                return (0, 0);
                            }
                            let lo = (i as f64 / n[a] as f64 * m as f64 - 0.5).floor().max(0.0) as usize;
                            let hi = (((i + 1) as f64 / n[a] as f64 * m as f64 - 0.5).floor() + 1.0) as usize;
                            (lo.min(m - 1), hi.min(m - 1))
                        };
                        if e.data.len() == 1 {
                            return e.data[0] > 0.0;
                        }
                        let (r0, r1, r2) = (sub(0, i), sub(1, j), sub(2, k));
                        (r2.0..=r2.1).any(|kk| {
                            (r1.0..=r1.1).any(|jj| {
                                (r0.0..=r0.1).any(|ii| e.data[(kk * e.n[1] + jj) * e.n[0] + ii] > 0.0)
                            })
                        })
                    }));
                }
            }
        }
        Self { n, dmax, emissive }
    }
}

impl Medium {
    /// Track against a majorant grid of `r³` supervoxels.
    pub fn with_majorant_grid(mut self, r: usize) -> Self {
        self.majorant = Some(MajorantGrid::new(&self.density, self.emission.as_ref().map(|e| &e.0), r));
        self
    }

    /// `c(λ)`.
    pub fn c(&self, lambda: f64) -> f64 {
        lift(self.sigma, lambda)
    }
}

/// One θ-dependent event of a walk or a ratio track.
#[derive(Debug, Clone, Copy)]
pub struct Event {
    /// The [`VolEvent`] columns.
    pub cols: [f64; COLUMNS],
    /// `δℓ / δL`: `L − L_j` (path) or `C` (ratio track).
    pub w: f64,
    /// Whether `w` is final (ratio track) or still `−L_j` awaiting `L`.
    pub fixed: bool,
}

/// `ℓ` of an event in f64 (the graph's twin).
pub fn ell(c: &[f64; COLUMNS], t: &[f64; PARAMS]) -> f64 {
    let [real, null, d, cc, mu, b0, b1, b2] = *c;
    let st = t[0] * d * cc;
    let a = b0 * t[1] + b1 * t[2] + b2 * t[3];
    let mut v = 0.0;
    if real != 0.0 {
        v += real * (a * st).ln();
    }
    if null != 0.0 {
        v += null * (mu - st).ln();
    }
    v
}

/// The medium-event graph over `n` lanes: inputs the eight columns, `θ`,
/// `δℓ`; outputs `ℓ`, and `δθ = Σ δℓ ∂ℓ/∂θ`.
pub struct VolEvent {
    /// The graph.
    pub g: TensorGraph,
    /// `ℓ`, `[n]`.
    pub ell: TensorId,
    /// `δθ`.
    pub dtheta: TensorId,
}

impl VolEvent {
    /// Build it.
    pub fn new(n: usize) -> Self {
        let mut g = TensorGraph::new();
        let cols: Vec<TensorId> = (0..COLUMNS).map(|_| g.input(&[n])).collect();
        let theta = g.input(&[PARAMS]);
        let dl = g.input(&[n]);
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
        let l = g
            .comp(&[n], &[] as &[usize], args, None, |x: &[ExprId]| {
                let ln = <ExprId as tang::Scalar>::ln;
                let one = <ExprId as tang::Scalar>::from_f64(1.0);
                let (real, null, d, c, mu) = (x[0], x[1], x[2], x[3], x[4]);
                let st = x[COLUMNS] * d * c;
                let a = x[5] * x[COLUMNS + 1] + x[6] * x[COLUMNS + 2] + x[7] * x[COLUMNS + 3];
                // Lanes of the other kind carry a harmless 1 inside the log.
                let lr = ln(real * a * st + (one - real));
                let lnull = ln(null * (mu - st) + (one - null));
                vec![lr + lnull]
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
                vec![pick(dl), pick(l)],
                Some(Monoid::sum()),
                |x| vec![x[0] * x[1]],
            )
            .unwrap();
        let dtheta = g.backward(s, &[theta]).unwrap()[0];
        Self { g, ell: l, dtheta }
    }

    /// The eval lowering, `[ℓ]`.
    pub fn eval(&self, native: bool) -> Program {
        let o = [self.ell];
        if native { Program::compile(&self.g, &o) } else { Program::interpreted(&self.g, &o) }
    }

    /// The adjoint lowering, `[δθ]`.
    pub fn adjoint(&self, native: bool) -> Program {
        let o = [self.dtheta];
        if native { Program::compile(&self.g, &o) } else { Program::interpreted(&self.g, &o) }
    }
}

/// Reduce events through the adjoint in chunks of `width`, in order.
pub fn reduce(prog: &mut Program, width: usize, ev: &[([f64; COLUMNS], f64)], theta: &[f64; PARAMS]) -> [f64; PARAMS] {
    let mut out = [0.0; PARAMS];
    prog.write(COLUMNS, theta);
    for chunk in ev.chunks(width) {
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

/// The tracker's state for one medium at one wavelength.
#[derive(Debug, Clone, Copy)]
pub struct Track<'a> {
    /// The medium.
    pub m: &'a Medium,
    /// `θ` and the sampler's `θ_s`.
    pub theta: [f64; PARAMS],
    #[allow(missing_docs)]
    pub theta_s: [f64; PARAMS],
    /// `c(λ)`, hats, majorant.
    pub c: f64,
    #[allow(missing_docs)]
    pub b: [f64; 3],
    #[allow(missing_docs)]
    pub mu: f64,
}

impl<'a> Track<'a> {
    /// At `lambda`; the majorant is the sampler's with 25 % headroom (detached:
    /// `θ₀` may move up to 1.25 `θ_s₀` before a null weight goes negative).
    pub fn new(m: &'a Medium, theta: [f64; PARAMS], theta_s: [f64; PARAMS], lambda: f64) -> Self {
        let c = m.c(lambda);
        let mu = 1.25 * theta_s[0] * m.density.max() * c + 1e-12;
        Self {
            m,
            theta,
            theta_s,
            c,
            b: band_weights(lambda),
            mu,
        }
    }

    fn cols(&self, real: bool, d: f64, mu: f64) -> [f64; COLUMNS] {
        let r = if real { 1.0 } else { 0.0 };
        [r, 1.0 - r, d, self.c, mu, self.b[0], self.b[1], self.b[2]]
    }

    /// The majorant segments of the ray `o + t d`, `t ∈ [0, tmax]`: calls
    /// `seg(t_in, t_out, μ̄, emissive)` in order until it returns `false`. One segment
    /// at the global `μ̄` without a grid; otherwise one per supervoxel
    /// crossed (`μ̄ = 1.25 θ_s₀ c · dmax`, floored at 5 % of the global
    /// majorant where there is emission, so the emission estimator
    /// `β e L_e / μ̄` sees every emitting point).
    fn segments(&self, o: Point3, d: Vec3, tmax: f64, mut seg: impl FnMut(f64, f64, f64, bool) -> bool) {
        let Some(mg) = &self.m.majorant else {
            seg(0.0, tmax, self.mu, self.m.emission.is_some());
            return;
        };
        let g = &self.m.density;
        let e = g.hi - g.lo;
        let size = [e.x / mg.n[0] as f64, e.y / mg.n[1] as f64, e.z / mg.n[2] as f64];
        let (lo, oo, dd) = ([g.lo.x, g.lo.y, g.lo.z], [o.x, o.y, o.z], [d.x, d.y, d.z]);
        let k = 1.25 * self.theta_s[0] * self.c;
        let floor = 0.05 * self.mu;
        let mut t = 0.0;
        while t < tmax {
            // The supervoxel at the segment's start, nudged forward; its exit
            // from its own bounds (robust: no incremental DDA state).
            let probe = t + 1e-9 * (1.0 + tmax);
            let mut cell = [0usize; 3];
            let mut t_out = tmax;
            for a in 0..3 {
                let p = oo[a] + dd[a] * probe;
                let i = (((p - lo[a]) / size[a]).floor().max(0.0) as usize).min(mg.n[a] - 1);
                cell[a] = i;
                if dd[a].abs() > 1e-300 {
                    let b = lo[a] + size[a] * (if dd[a] > 0.0 { i + 1 } else { i }) as f64;
                    let tb = (b - oo[a]) / dd[a];
                    if tb > t {
                        t_out = t_out.min(tb);
                    }
                }
            }
            let t_out = t_out.max(probe).min(tmax);
            let ci = (cell[2] * mg.n[1] + cell[1]) * mg.n[0] + cell[0];
            let mut mu = k * mg.dmax[ci];
            if mg.emissive[ci] {
                mu = mu.max(floor);
            }
            if !seg(t, t_out, mu, mg.emissive[ci]) {
                return;
            }
            t = t_out;
        }
    }

    fn albedo(&self, t: &[f64; PARAMS]) -> f64 {
        self.b[0] * t[1] + self.b[1] * t[2] + self.b[2] * t[3]
    }

    /// Free flight from `o` along `d` up to `tmax`: `Some((t, weight))` at
    /// a real collision, `None` if it leaves; events pushed to `ev` (with
    /// `fixed = false`, `w = −L_j` filled by the caller), weights folded
    /// into `beta`, emission added to `l`.
    #[allow(clippy::too_many_arguments)]
    pub fn flight(
        &self,
        o: Point3,
        d: Vec3,
        tmax: f64,
        rng: &mut Rng,
        beta: &mut f64,
        l: &mut f64,
        lambda: f64,
        ev: &mut Option<&mut Vec<Event>>,
    ) -> Option<f64> {
        // One optical depth `τ ~ Exp(1)` is spent across the segments
        // (`μ̄ Δt` each); a tentative collision where it runs out, then a
        // fresh `τ`.
        let mut out = None;
        let mut tau = -(1.0 - rng.uniform()).ln();
        self.segments(o, d, tmax, |t_in, t_out, mu, emissive| {
            let mut t = t_in;
            loop {
                let run = mu * (t_out - t);
                if !(tau < run) {
                    tau -= run;
                    return true;
                }
                t += tau / mu;
                tau = -(1.0 - rng.uniform()).ln();
                let x = o + d * t;
                let dens = self.m.density.at(x);
                if emissive {
                    if let Some((eg, le)) = &self.m.emission {
                        *l += *beta * eg.at(x) * lift(*le, lambda) / mu;
                    }
                }
                let st = self.theta[0] * dens * self.c;
                let sts = self.theta_s[0] * dens * self.c;
                let u = rng.uniform();
                let real = u < sts / mu;
                if let Some(e) = ev.as_deref_mut() {
                    e.push(Event {
                        cols: self.cols(real, dens, mu),
                        w: -*l,
                        fixed: false,
                    });
                }
                if real {
                    *beta *= self.albedo(&self.theta) * st / sts;
                    out = Some(t);
                    return false;
                }
                *beta *= (mu - st) / (mu - sts);
            }
        });
        out
    }

    /// Ratio-tracking transmittance from `o` along `d` over `[0, tmax]`;
    /// its events (with `w` to be set to `C` by the caller) go to `ev`.
    pub fn transmittance(
        &self,
        o: Point3,
        d: Vec3,
        tmax: f64,
        rng: &mut Rng,
        ev: &mut Vec<[f64; COLUMNS]>,
    ) -> f64 {
        let mut tr = 1.0;
        let mut tau = -(1.0 - rng.uniform()).ln();
        self.segments(o, d, tmax, |t_in, t_out, mu, _| {
            let mut t = t_in;
            loop {
                let run = mu * (t_out - t);
                if !(tau < run) {
                    tau -= run;
                    return true;
                }
                t += tau / mu;
                tau = -(1.0 - rng.uniform()).ln();
                let dens = self.m.density.at(o + d * t);
                tr *= 1.0 - self.theta[0] * dens * self.c / mu;
                ev.push(self.cols(false, dens, mu));
            }
        });
        tr
    }
}

/// A volume scene: one medium, a Lambertian floor at `z = 0`, quads, a
/// constant environment, a pinhole camera.
pub struct VolumeScene {
    /// The medium.
    pub medium: Medium,
    /// Floor albedo (linear sRGB); `None`: no floor.
    pub floor: Option<[f64; 3]>,
    /// Emitters.
    pub lights: Vec<Light>,
    /// Environment radiance.
    pub env: f64,
    /// Camera: eye, target, vertical fov (degrees).
    pub eye: Point3,
    #[allow(missing_docs)]
    pub target: Point3,
    #[allow(missing_docs)]
    pub vfov: f64,
    /// Pixels, samples per pixel, path depth.
    pub size: (usize, usize),
    #[allow(missing_docs)]
    pub spp: usize,
    #[allow(missing_docs)]
    pub depth: usize,
    /// The frozen sampler's `θ_s`; `None`: the sampler is `θ`.
    pub sampler: Option<[f64; PARAMS]>,
}

impl VolumeScene {
    /// Paths per frame.
    pub fn paths(&self) -> usize {
        self.size.0 * self.size.1 * self.spp
    }

    fn occluded_floor(&self, o: Point3, d: Vec3, tmax: f64) -> bool {
        self.floor.is_some() && d.z < 0.0 && -o.z / d.z < tmax && o.z > 0.0
    }

    /// One path; `rec` collects `(columns, δℓ/δL)` once `L` is known.
    pub fn trace(
        &self,
        theta: &[f64; PARAMS],
        seed: u64,
        path: usize,
        rec: Option<&mut Vec<([f64; COLUMNS], f64)>>,
    ) -> (f64, f64) {
        let mut rng = Rng::new(seed, path as u64);
        // Wavelengths stratified over a pixel's samples: the chroma noise of
        // one wavelength per path falls as 1/spp, not 1/√spp.
        let lambda = sample_lambda_nm(((path % self.spp) as f64 + rng.uniform()) / self.spp as f64);
        let ts = self.sampler.unwrap_or(*theta);
        let tr = Track::new(&self.medium, *theta, ts, lambda);
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
        let mut evs: Vec<Event> = Vec::new();
        let recording = rec.is_some();
        let mut last_pdf: Option<f64> = None;
        for _ in 0..self.depth {
            // Nearest surface: floor or a light.
            let mut t_surf = f64::INFINITY;
            let mut hit_light: Option<usize> = None;
            if self.floor.is_some() && d.z < 0.0 && o.z > 0.0 {
                t_surf = -o.z / d.z;
            }
            for (i, q) in self.lights.iter().enumerate() {
                if let Some(t) = q.hit(o, d, t_surf) {
                    t_surf = t;
                    hit_light = Some(i);
                }
            }
            // The medium along the segment.
            let mut scattered = None;
            if let Some((t0, t1)) = self.medium.density.span(o, d) {
                let t1 = t1.min(t_surf);
                if t1 > t0 {
                    let mut ev = if recording { Some(&mut evs) } else { None };
                    if let Some(t) =
                        tr.flight(o + d * t0, d, t1 - t0, &mut rng, &mut beta, &mut l, lambda, &mut ev)
                    {
                        scattered = Some(t0 + t);
                    }
                }
            }
            if let Some(t) = scattered {
                let x = o + d * t;
                // NEE with the phase function.
                let u = [rng.uniform(), rng.uniform(), rng.uniform()];
                if let Some((dl, dist, pl, i)) = sample_light(&self.lights, x, u) {
                    if !self.occluded_floor(x, dl, dist) {
                        let mut tev = Vec::new();
                        let tt = match self.medium.density.span(x, dl) {
                            Some((a, b)) => {
                                let b = b.min(dist);
                                if b > a {
                                    tr.transmittance(x + dl * a, dl, b - a, &mut rng, &mut tev)
                                } else {
                                    1.0
                                }
                            }
                            None => 1.0,
                        };
                        let ph = hg(d.dot(&dl), self.medium.g);
                        let c = beta * ph * tt * self.lights[i].le(lambda) / pl * mis(pl, ph);
                        l += c;
                        if recording {
                            for cols in tev {
                                evs.push(Event { cols, w: c, fixed: true });
                            }
                        }
                    }
                }
                let nd = hg_sample(d, self.medium.g, rng.uniform(), rng.uniform());
                last_pdf = Some(hg(d.dot(&nd), self.medium.g));
                d = nd;
                o = x;
                continue;
            }
            if let Some(i) = hit_light {
                let (_, pl) = light_pdf(&self.lights, o, d);
                let wgt = last_pdf.map_or(1.0, |pb| mis(pb, pl));
                l += beta * self.lights[i].le(lambda) * wgt;
                break;
            }
            if t_surf.is_finite() {
                // The floor: Lambertian, NEE through the medium.
                let x = o + d * t_surf;
                let n = Vec3::new(0.0, 0.0, 1.0);
                let rho = lift(self.floor.unwrap(), lambda);
                let u = [rng.uniform(), rng.uniform(), rng.uniform()];
                if let Some((dl, dist, pl, i)) = sample_light(&self.lights, x, u) {
                    let c0 = dl.dot(&n);
                    if c0 > 0.0 {
                        let mut tev = Vec::new();
                        let tt = match self.medium.density.span(x, dl) {
                            Some((a, b)) => {
                                let b = b.min(dist);
                                if b > a {
                                    tr.transmittance(x + dl * a, dl, b - a, &mut rng, &mut tev)
                                } else {
                                    1.0
                                }
                            }
                            None => 1.0,
                        };
                        let pb = c0 / PI;
                        let c = beta * rho / PI * c0 * tt * self.lights[i].le(lambda) / pl
                            * mis(pl, pb);
                        l += c;
                        if recording {
                            for cols in tev {
                                evs.push(Event { cols, w: c, fixed: true });
                            }
                        }
                    }
                }
                let (tu, tv) = basis(n);
                let cd = super::openpbr::cosine([rng.uniform(), rng.uniform()]);
                d = tu * cd[0] + tv * cd[1] + n * cd[2];
                o = x + n * 1e-7;
                beta *= rho;
                last_pdf = Some(cd[2] / PI);
                continue;
            }
            l += beta * self.env;
            break;
        }
        if let Some(r) = rec {
            for e in evs {
                r.push((e.cols, if e.fixed { e.w } else { l + e.w }));
            }
        }
        (l, lambda)
    }

    /// Forward: `(L, λ)` per path.
    pub fn forward(&self, theta: &[f64; PARAMS], seed: u64) -> Vec<(f64, f64)> {
        (0..self.paths())
            .into_par_iter()
            .map(|i| self.trace(theta, seed, i, None))
            .collect()
    }

    /// Linear sRGB image.
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

    /// `(loss, δθ)` by PRB with `loss` over the image, and the per-path `L`
    /// of the replay.
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
        let mut ev = Vec::new();
        let mut ls = Vec::new();
        for (j, (l, r)) in per.into_iter().enumerate() {
            let w = hero_weight(fwd[j].1);
            let px = j / self.spp;
            let dl: f64 = (0..3)
                .map(|c| d_img[px * 3 + c] * w[c] as f64 / self.spp as f64)
                .sum();
            ls.push(l);
            for (c, wl) in r {
                ev.push((c, dl * wl));
            }
        }
        let width = 4096;
        let mut prog = VolEvent::new(width).adjoint(native);
        (v, reduce(&mut prog, width, &ev, theta), ls)
    }
}

/// Henyey–Greenstein phase value at cosine `c`.
pub fn hg(c: f64, g: f64) -> f64 {
    let d = 1.0 + g * g - 2.0 * g * c;
    (1.0 - g * g) / (4.0 * PI * d * d.sqrt())
}

fn sample_light(lights: &[Light], x: Point3, u: [f64; 3]) -> Option<(Vec3, f64, f64, usize)> {
    let n = lights.len();
    if n == 0 {
        return None;
    }
    let i = ((u[0] * n as f64) as usize).min(n - 1);
    let q = &lights[i];
    let p = q.center + q.u * (2.0 * u[1] - 1.0) + q.v * (2.0 * u[2] - 1.0);
    let dv = p - x;
    let d2 = dv.norm_squared();
    let d = dv / d2.sqrt();
    let cl = -d.dot(&q.normal());
    if cl <= 0.0 {
        return None;
    }
    Some((d, d2.sqrt(), d2 / (cl * q.area() * n as f64), i))
}

fn light_pdf(lights: &[Light], o: Point3, d: Vec3) -> (f64, f64) {
    let n = lights.len() as f64;
    let mut best = (f64::INFINITY, 0.0);
    for q in lights {
        if let Some(t) = q.hit(o, d, best.0) {
            let cl = -d.dot(&q.normal());
            best = (t, t * t / (cl * q.area() * n));
        }
    }
    best
}
