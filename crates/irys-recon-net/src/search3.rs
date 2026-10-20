//! v3's MDL search: v2's (`search.rs`) with the faithful compare image
//! ([`crate::faithful`]) and the edge-sampled Jacobian ([`crate::edgejac`]).
//!
//! ```text
//! MDL = Σ ρ(r/σ)/(κτ)  +  ½ log det(Σ_p F/(κτ) + I)  +  ½ Σ ((θ − θ₀)/σ_p)²  +  program length     (nats)
//! ```
//!
//! `r` is the band-passed difference between the OpenPBR render of the scene
//! (fitted lights and substances) and the photo, `κ` the number of pixels a
//! blur of σ px makes one independent measurement of (`1 + 4πσ²`, so the
//! residual is not counted once per correlated pixel), `F = JᵀWJ` with `J`
//! from the edge term on analytic vertex velocities, `τ` the residual's
//! dispersion ([`dispersion`]) and `θ₀, σ_p` the network's prior on the
//! parameters (the structure is coded by its `−log p / T`). Per object (largest
//! first) every candidate structure is fitted by Levenberg–Marquardt over a
//! blur homotopy and the shortest description kept; dropping the object is
//! always a candidate.
//!
//! The pose is fitted first on the coarsest level through the TLAS alone
//! ([`Compare3::set_poses`]: M1's `set_instances`, no geometry upload); a
//! shape step rebuilds the scene on the device.

use rayon::prelude::*;

use crate::edgejac::{self, EdgeOpts};
use crate::faithful::{Appearance, Compare3, local_meshes};
use crate::scene::{self, Cam, Object};
use crate::search::{Fitted, State, Window, crop, decode, encode, window};
use crate::velocity::NV;

/// What one search sees.
pub struct Ctx3<'a> {
    pub gpu: &'static irys_gpu::GpuContext,
    /// The photo linearised with `app.gamma`, at `size`.
    pub target: &'a [[f64; 3]],
    pub size: (usize, usize),
    pub cam: Cam,
    pub app: Appearance,
    pub spp: u32,
    pub seed: u64,
    /// The photo's noise σ (linear units, unblurred).
    pub sigma: f64,
    /// The residual's σ at each level (blurred photo and render noise,
    /// [`noise_levels`]); empty: `sigma` everywhere.
    pub sig_levels: Vec<f64>,
    /// Blur levels (px), coarse to fine; the MDL is read on the last.
    pub levels: Vec<f64>,
    /// Temperature on the structure prior: `−log p / T`.
    pub prior_temp: f64,
    /// Count blur-correlated pixels once (`κ`); off is v2's sum.
    pub decorrelate: bool,
    pub edge: EdgeOpts,
    /// Give every candidate its own colour (the closed-form scale of the
    /// object's pixels), so geometry is not asked to absorb a colour misfit.
    pub recolour: bool,
    /// … and a linear-in-normal correction of its shading.
    pub shade: bool,
    /// Forward differences for the [`weak`] columns.
    pub fd_weak: bool,
    /// The prior σ (mm) of a detection's position, with `param_sd`.
    pub pose_sd: f64,
    /// Band-pass the residual: `G(s) − G(k·s)` at each level (≤ 1: off).
    pub highpass: f64,
    /// The residual's dispersion: how many times its noise-only value the
    /// fitted model's residual is, per independent measurement (≥ 1; a
    /// quasi-likelihood temperature, [`dispersion`]).
    pub tau: f64,
    /// Per object, the prior σ of each parameter slot in log units (the
    /// network's Gaussian head; empty: no parameter prior).
    pub param_sd: Vec<Vec<f64>>,
    /// The oracle's exact world, key and exposure (else the fitted lobes).
    pub exact: Option<(irys_diff::look::scene::Equirect, irys_diff::look::scene::Sun, f64)>,
}

impl Ctx3<'_> {
    /// The model's blur at level `s`: the level's and the lens's.
    pub fn model_blur(&self, s: f64) -> f64 {
        let d = self.app.defocus * self.size.0 as f64;
        (s * s + d * d).sqrt()
    }
    /// The residual's σ at blur `s`.
    pub fn sig(&self, s: f64) -> f64 {
        self.levels.iter().position(|&l| l == s).and_then(|k| self.sig_levels.get(k).copied()).unwrap_or(self.sigma)
    }
    pub fn kappa(&self, s: f64) -> f64 {
        if self.decorrelate { 1.0 + 4.0 * std::f64::consts::PI * s * s } else { 1.0 }
    }
}

const HUBER: f64 = 2.0;

pub fn rho_sum(r: &[f64]) -> f64 {
    r.iter().map(|&x| irys_recon::look::huber_rho(x, HUBER)).sum()
}

fn blur_win(img: &[f64], sz: (usize, usize), s: f64) -> Vec<f64> {
    if s <= 0.0 { img.to_vec() } else { irys_recon::fit::blur(img, sz, 3, s) }
}

/// The level's filter: a Gaussian of σ `s`, or with `k > 1` the band
/// `G(s) − G(k·s)` (a difference of Gaussians: shading that varies slowly
/// across a face — what a misfit of light or substance leaves — is
/// removed, edges and their contrast are kept).
fn band(img: &[f64], sz: (usize, usize), s: f64, k: f64) -> Vec<f64> {
    let a = blur_win(img, sz, s);
    if k <= 1.0 {
        return a;
    }
    let b = blur_win(img, sz, s * k);
    a.iter().zip(&b).map(|(x, y)| x - y).collect()
}

fn flat(v: &[[f64; 3]]) -> Vec<f64> {
    v.iter().flat_map(|p| p.iter().copied()).collect()
}

/// The device scene of `st` (absent objects dropped).
pub fn compare(st: &State, ctx: &Ctx3) -> Compare3 {
    let meshes: Vec<scene::Tris> = local_meshes(&st.objs).into_iter().enumerate().map(|(i, m)| if st.present(i) { m } else { Default::default() }).collect();
    let mut c = Compare3::new(ctx.gpu, &st.objs, &meshes, &ctx.app, &ctx.cam, ctx.size).expect("device");
    c.set_appearance(&ctx.app);
    if let Some((e, sun, k)) = &ctx.exact {
        c.set_exact(e, sun, *k);
    }
    c
}

/// The render of `st`.
pub fn render(st: &State, ctx: &Ctx3) -> Vec<[f64; 3]> {
    compare(st, ctx).render(&ctx.cam, ctx.spp, ctx.seed)
}

/// Object `i`'s appearance refitted on its own pixels in `win`: per
/// channel the render is multiplied by `a + b·n` (`n` the surface normal
/// the pixel sees), in closed form with a ridge on `b` (`b = 0`: a colour
/// scale only, `ctx.shade` off). The render with that colour and that
/// correction of the shading's direction, up to interreflection: the
/// nuisance freedom a candidate gets so its geometry is not asked to absorb
/// an appearance misfit.
pub fn recolour(model: &[[f64; 3]], st: &State, i: usize, ctx: &Ctx3, win: Window) -> Vec<[f64; 3]> {
    if !ctx.recolour || !st.present(i) {
        return model.to_vec();
    }
    let caster = st.caster();
    let cam = ctx.cam.resized(ctx.size);
    let (x0, y0, x1, y1) = win;
    let w = ctx.size.0;
    let px: Vec<(usize, [f64; 3])> = (y0..y1)
        .into_par_iter()
        .flat_map_iter(|y| {
            let caster = &caster;
            (x0..x1).filter_map(move |x| {
                let h = caster.cast(&cam, x as f64 + 0.5, y as f64 + 0.5);
                (h.0 as usize == i).then_some((y * w + x, h.2))
            })
        })
        .collect();
    let nb = if ctx.shade { 4 } else { 1 };
    let mut out = model.to_vec();
    for c in 0..3 {
        let mut a = [[0.0f64; 4]; 4];
        let mut g = [0.0f64; 4];
        for &(p, n) in &px {
            let t = ctx.target[p];
            if t[c] >= crate::faithful::SAT && model[p][c] >= crate::faithful::SAT {
                continue;
            }
            let m = model[p][c];
            let f = [m, m * n[0], m * n[1], m * n[2]];
            for u in 0..nb {
                g[u] += f[u] * t[c];
                for v in 0..nb {
                    a[u][v] += f[u] * f[v];
                }
            }
        }
        if a[0][0] <= 1e-12 {
            continue;
        }
        // Ridge: b pulled toward 0 with a tenth of the colour's weight;
        // a toward 1.
        let lam = 0.1 * a[0][0];
        for u in 1..nb {
            a[u][u] += lam;
        }
        a[0][0] += 1e-3 * a[0][0];
        g[0] += 1e-3 * a[0][0];
        let mut aa: Vec<Vec<f64>> = (0..nb).map(|u| a[u][..nb].to_vec()).collect();
        let _ = &mut aa;
        let Some(x) = solve(&aa, &g[..nb]) else { continue };
        let k0 = x[0].clamp(0.5, 2.0);
        for &(p, n) in &px {
            let mut f = k0;
            for u in 1..nb {
                f += x[u] * n[u - 1];
            }
            out[p][c] = model[p][c] * f.clamp(0.2, 3.0);
        }
    }
    out
}

/// Residuals `(model − target)/σ`, blurred by `s`, over `win`.
pub fn residual_of(model: &[[f64; 3]], ctx: &Ctx3, win: Window, s: f64) -> Vec<f64> {
    let sz = (win.2 - win.0, win.3 - win.1);
    // The lens blurs, then the sensor clips, then the level's blur (both).
    let d = ctx.app.defocus * ctx.size.0 as f64;
    let m: Vec<f64> = blur_win(&flat(&crop(model, ctx.size, win)), sz, d).iter().map(|v| v.min(1.0)).collect();
    let m = band(&m, sz, s, ctx.highpass);
    let t = band(&flat(&crop(ctx.target, ctx.size, win)), sz, s, ctx.highpass);
    let sg = ctx.sig(s);
    m.iter().zip(&t).map(|(a, b)| (a - b) / sg).collect()
}

/// Encoded variable `j` of `o` ([`encode`]'s order) → its tangent slot and
/// `∂slot/∂v` (positions per 50 mm, yaw per 20°, log parameters).
pub fn slots(o: &Object) -> Vec<(usize, f64)> {
    let mut v = vec![(scene::NP, 50.0), (scene::NP + 1, 50.0)];
    if o.family.has_yaw() {
        v.push((scene::NP + 2, 20.0));
    }
    for k in 0..o.params.len() {
        if scene::param_used(o.family, &o.tokens, k) && !matches!((o.family, k), (scene::Family::Grid, 3)) {
            v.push((k, o.params[k]));
        }
    }
    v
}

/// Columns of `∂r/∂v` (blurred by `s`, in σ units) for object `i` of `st`
/// with `model` its render.
pub fn jacobian(st: &State, i: usize, model: &[[f64; 3]], ctx: &Ctx3, win: Window, s: f64, pose_only: bool) -> Vec<Vec<f64>> {
    let o = &st.objs[i];
    let cam = ctx.cam.resized(ctx.size);
    let j = edgejac::jacobian(o, i, &cam, model, &st.caster(), ctx.app.key.dir, win, &ctx.edge);
    let sz = (win.2 - win.0, win.3 - win.1);
    let sl = slots(o);
    let d = ctx.app.defocus * ctx.size.0 as f64;
    let unclipped: Vec<f64> = blur_win(&flat(&crop(model, ctx.size, win)), sz, d).iter().map(|&v| if v < 1.0 { 1.0 } else { 0.0 }).collect();
    let n = if pose_only { if o.family.has_yaw() { 3 } else { 2 } } else { sl.len() };
    let r0 = if ctx.fd_weak && !pose_only { Some(residual_of(model, ctx, win, s)) } else { None };
    sl[..n]
        .iter()
        .map(|&(slot, scale)| {
            debug_assert!(slot < NV);
            if let (Some(r0), true) = (&r0, weak(o, slot)) {
                // Forward difference: one shape rebuild and render.
                let dv = 0.02f64;
                let mut q = o.clone();
                q.params[slot] *= dv.exp();
                let sn = st.with(i, Some(&q));
                let m = recolour(&render(&sn, ctx), &sn, i, ctx, win);
                let r1 = residual_of(&m, ctx, win, s);
                return r1.iter().zip(r0).map(|(a, b)| (a - b) / dv).collect();
            }
            let sg = ctx.sig(s);
            let c: Vec<f64> = blur_win(&j.cols[slot], sz, d).iter().zip(&unclipped).map(|(a, b)| a * b).collect();
            band(&c, sz, s, ctx.highpass).iter().map(|x| x * scale / sg).collect()
        })
        .collect()
}

/// Columns whose edge term misses most of the derivative
/// (`neural_v3_jaccheck`: cosine under 0.8 against central differences):
/// the lathe's inner profile radii (shading of the bulge, an interior
/// term), the keyboard's base and key-cap heights, the honeycomb cell, and
/// a wall's thickness and hole pitch (desk scenes: 0.02–0.03).
pub fn weak(o: &Object, slot: usize) -> bool {
    use scene::Family::*;
    matches!((o.family, slot), (Lathe, 2) | (Lathe, 3) | (Grid, 2) | (Grid, 4) | (Wall, 2) | (Wall, 3)) || (o.family == Shell && o.tokens[1] == 1 && slot == 4)
}

fn solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a.iter().zip(b).map(|(r, &y)| r.iter().copied().chain([y]).collect()).collect();
    for i in 0..n {
        let piv = (i..n).max_by(|&x, &y| m[x][i].abs().total_cmp(&m[y][i].abs()))?;
        m.swap(i, piv);
        if m[i][i].abs() < 1e-14 {
            return None;
        }
        for r in 0..n {
            if r != i {
                let f = m[r][i] / m[i][i];
                for c in i..=n {
                    m[r][c] -= f * m[i][c];
                }
            }
        }
    }
    Some((0..n).map(|i| m[i][n] / m[i][i]).collect())
}

fn logdet_plus_i(f: &[Vec<f64>]) -> f64 {
    let n = f.len();
    let mut l = vec![vec![0.0; n]; n];
    let mut ld = 0.0;
    for i in 0..n {
        for j in 0..=i {
            let mut s = f[i][j] + if i == j { 1.0 } else { 0.0 };
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                let d = s.max(1e-12).sqrt();
                l[i][i] = d;
                ld += 2.0 * d.ln();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    ld
}

/// Counters.
#[derive(Debug, Default, Clone, Copy)]
pub struct Counts {
    pub builds: usize,
    pub moves: usize,
    pub renders: usize,
    pub jacobians: usize,
}

/// Levenberg–Marquardt on object `i` of `st` (others fixed) over `win`.
pub fn fit(st: &State, i: usize, init: &Object, prior: f64, ctx: &Ctx3, win: Window, iters: usize, n: &mut Counts) -> Fitted {
    let mut v = encode(init);
    let mut cur = st.with(i, Some(init));
    // 1. Pose on the coarsest level through the TLAS only.
    let s0 = ctx.levels[0];
    let mut cmp = compare(&cur, ctx);
    n.builds += 1;
    let mut model = recolour(&cmp.render(&ctx.cam, ctx.spp, ctx.seed), &cur, i, ctx, win);
    n.renders += 1;
    let present: Vec<bool> = (0..cur.objs.len()).map(|k| cur.present(k)).collect();
    let mut cost = rho_sum(&residual_of(&model, ctx, win, s0));
    let mut lambda = 1e-2;
    let npose = if init.family.has_yaw() { 3 } else { 2 };
    for _ in 0..iters.min(4) {
        let r = residual_of(&model, ctx, win, s0);
        let cols = jacobian(&cur, i, &model, ctx, win, s0, true);
        n.jacobians += 1;
        let (a, g) = normal(&cols, &r);
        let mut improved = false;
        for _ in 0..3 {
            let Some(d) = damped(&a, &g, lambda) else { break };
            let mut vn = v.clone();
            for k in 0..npose {
                vn[k] += d[k];
            }
            let on = decode(init, &vn);
            let mut objs = cur.objs.clone();
            objs[i] = on.clone();
            cmp.set_poses(&objs, &present);
            n.moves += 1;
            let mut probe = cur.clone();
            probe.objs[i] = on.clone();
            let probe = probe.with(i, Some(&on));
            let mn = recolour(&cmp.render(&ctx.cam, ctx.spp, ctx.seed), &probe, i, ctx, win);
            n.renders += 1;
            let cn = rho_sum(&residual_of(&mn, ctx, win, s0));
            if cn < cost {
                let rel = (cost - cn) / cost.max(1e-9);
                v = vn;
                cost = cn;
                model = mn;
                cur.objs[i] = on; // same shape: the BLAS is unchanged, only the pose
                cur = cur.with(i, Some(&cur.objs[i].clone()));
                lambda = (lambda / 3.0).max(1e-6);
                improved = rel > 1e-3;
                break;
            }
            lambda *= 4.0;
            cmp.set_poses(&cur.objs, &present);
        }
        if !improved {
            break;
        }
    }
    // 2. Everything, over the homotopy, with the parameter prior.
    let v0 = encode(init);
    let sd: Vec<f64> = {
        let ps = ctx.param_sd.get(i).cloned().unwrap_or_default();
        let sl = slots(init);
        // Positions: the detection's own place, ± `pose_sd` mm (encoded per
        // 50 mm); yaw free.
        let psd = if ps.is_empty() { f64::INFINITY } else { ctx.pose_sd / 50.0 };
        sl.iter().map(|&(slot, _)| if slot < scene::NP { ps.get(slot).copied().unwrap_or(f64::INFINITY) } else if slot < scene::NP + 2 { psd } else { f64::INFINITY }).collect()
    };
    let prior_of = |v: &[f64]| -> f64 { v.iter().zip(&v0).zip(&sd).map(|((a, b), s)| if s.is_finite() { 0.5 * ((a - b) / s).powi(2) } else { 0.0 }).sum() };
    let mut jtwj: Vec<Vec<f64>> = Vec::new();
    let mut r_fin = Vec::new();
    for (li, &s) in ctx.levels.iter().enumerate() {
        let q = 1.0 / (ctx.kappa(s) * ctx.tau);
        let mut r = residual_of(&model, ctx, win, s);
        cost = q * rho_sum(&r) + prior_of(&v);
        lambda = 1e-2;
        let per = if li + 1 == ctx.levels.len() { iters } else { iters.div_ceil(2) };
        for it in 0..=per {
            let cols = jacobian(&cur, i, &model, ctx, win, s, false);
            n.jacobians += 1;
            let w: Vec<f64> = r.iter().map(|&x| irys_recon::look::huber_w(x, HUBER)).collect();
            let wc: Vec<Vec<f64>> = cols.iter().map(|c| c.iter().zip(&w).map(|(a, b)| a * b.sqrt()).collect()).collect();
            let wr: Vec<f64> = r.iter().zip(&w).map(|(a, b)| a * b.sqrt()).collect();
            let (a, g) = normal(&wc, &wr);
            jtwj = a.clone();
            let mut a: Vec<Vec<f64>> = a.iter().map(|row| row.iter().map(|x| x * q).collect()).collect();
            let mut g: Vec<f64> = g.iter().map(|x| x * q).collect();
            for k in 0..g.len() {
                if sd[k].is_finite() {
                    a[k][k] += 1.0 / (sd[k] * sd[k]);
                    g[k] += (v[k] - v0[k]) / (sd[k] * sd[k]);
                }
            }
            if it == per {
                break;
            }
            let mut improved = false;
            for _ in 0..3 {
                let Some(d) = damped(&a, &g, lambda) else { break };
                let vn: Vec<f64> = v.iter().zip(&d).map(|(a, b)| a + b).collect();
                let on = decode(init, &vn);
                let sn = cur.with(i, Some(&on));
                let mn = recolour(&render(&sn, ctx), &sn, i, ctx, win);
                n.builds += 1;
                n.renders += 1;
                let rn = residual_of(&mn, ctx, win, s);
                let cn = q * rho_sum(&rn) + prior_of(&vn);
                if cn < cost {
                    let rel = (cost - cn) / cost.max(1e-9);
                    v = vn;
                    cur = sn;
                    model = mn;
                    r = rn;
                    cost = cn;
                    lambda = (lambda / 3.0).max(1e-6);
                    improved = rel > 1e-4;
                    break;
                }
                lambda *= 4.0;
            }
            if !improved {
                break;
            }
        }
        r_fin = r;
    }
    let s_fin = *ctx.levels.last().unwrap();
    let kappa = ctx.kappa(s_fin) * ctx.tau;
    let residual = rho_sum(&r_fin) / kappa;
    // The parameters' code: ½ log det(I + Σ_prior F) (F alone where there
    // is no prior), plus the prior's penalty at the estimate.
    let fk: Vec<Vec<f64>> = jtwj
        .iter()
        .enumerate()
        .map(|(a, row)| row.iter().enumerate().map(|(b, x)| x / kappa * if sd[a].is_finite() && sd[b].is_finite() { sd[a] * sd[b] } else { 1.0 }).collect())
        .collect();
    let param_cost = 0.5 * logdet_plus_i(&fk) + prior_of(&v);
    let obj = cur.objs[i].clone();
    let length = scene::program_length_with(&obj, prior / ctx.prior_temp);
    let sigma = (0..fk.len()).map(|k| 1.0 / (fk[k][k] + 1e-9).sqrt()).collect();
    Fitted { mdl: residual + param_cost + length, residual, param_cost, length, sigma, object: Some(obj), evals: 0 }
}

fn normal(cols: &[Vec<f64>], r: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = cols.len();
    let mut a = vec![vec![0.0; n]; n];
    let mut g = vec![0.0; n];
    for x in 0..n {
        for y in 0..=x {
            let s: f64 = cols[x].iter().zip(&cols[y]).map(|(p, q)| p * q).sum();
            a[x][y] = s;
            a[y][x] = s;
        }
        g[x] = cols[x].iter().zip(r).map(|(p, q)| p * q).sum();
    }
    (a, g)
}

fn damped(a: &[Vec<f64>], g: &[f64], lambda: f64) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    for k in 0..g.len() {
        m[k][k] += lambda * (a[k][k] + 1e-3);
    }
    let d = solve(&m, &g.iter().map(|x| -x).collect::<Vec<_>>())?;
    Some(d.iter().map(|x| x.clamp(-0.5, 0.5)).collect())
}

/// The window's cost with object `i` dropped.
pub fn dropped(st: &State, i: usize, ctx: &Ctx3, win: Window, n: &mut Counts) -> Fitted {
    let m = render(&st.with(i, None), ctx);
    n.builds += 1;
    n.renders += 1;
    let s = *ctx.levels.last().unwrap();
    let c = rho_sum(&residual_of(&m, ctx, win, s)) / (ctx.kappa(s) * ctx.tau);
    Fitted { object: None, residual: c, param_cost: 0.0, length: 0.0, mdl: c, sigma: Vec::new(), evals: 1 }
}

/// For every object (largest first), fit each candidate structure and keep
/// the shortest description. Returns the state, each object's winning fit
/// and every candidate's fit (for the MDL-vs-error study).
pub fn search(objs: Vec<(Object, f64)>, cands: &[Vec<(Object, f64)>], ctx: &Ctx3, iters: usize, sweeps: usize) -> (State, Vec<Fitted>, Vec<Vec<Fitted>>, Counts) {
    let mut n = Counts::default();
    let mut priors: Vec<f64> = objs.iter().map(|o| o.1).collect();
    let mut st = State::new(objs.into_iter().map(|o| o.0).collect());
    let mut order: Vec<usize> = (0..st.objs.len()).collect();
    order.sort_by(|&a, &b| st.objs[b].radius().total_cmp(&st.objs[a].radius()));
    let mut best_fit: Vec<Option<Fitted>> = vec![None; st.objs.len()];
    let mut all: Vec<Vec<Fitted>> = vec![Vec::new(); st.objs.len()];
    let cam = ctx.cam.resized(ctx.size);
    for _ in 0..sweeps {
        for &i in &order {
            let cur = (st.objs[i].clone(), priors[i]);
            let mut cs: Vec<&(Object, f64)> = cands[i].iter().collect();
            if st.present(i) {
                cs.insert(0, &cur);
            }
            let ws: Vec<&Object> = cs.iter().map(|c| &c.0).collect();
            let Some(win) = window(&ws, &cam, ctx.size, 10.0) else {
                continue;
            };
            let mut best = dropped(&st, i, ctx, win, &mut n);
            let mut best_prior = priors[i];
            all[i].clear();
            all[i].push(best.clone());
            for c in cs {
                let f = fit(&st, i, &c.0, c.1, ctx, win, iters, &mut n);
                all[i].push(f.clone());
                if f.mdl < best.mdl {
                    best = f;
                    best_prior = c.1;
                }
            }
            if std::env::var("V3_DEBUG").is_ok() {
                let line: Vec<String> = all[i].iter().map(|f| format!("{}:{:.0}(r{:.0} p{:.0} l{:.0})", f.object.as_ref().map(|o| o.class.name()).unwrap_or("drop"), f.mdl, f.residual, f.param_cost, f.length)).collect();
                eprintln!("obj {i}: {}", line.join("  "));
            }
            priors[i] = best_prior;
            st = st.with(i, best.object.as_ref());
            best_fit[i] = Some(best);
        }
    }
    let fits = best_fit.into_iter().map(|f| f.unwrap_or(Fitted { object: None, residual: 0.0, param_cost: 0.0, length: 0.0, mdl: 0.0, sigma: Vec::new(), evals: 0 })).collect();
    (st, fits, all, n)
}

/// The MDL of a whole scene on the full frame at the finest level
/// (residual + every kept object's program length).
pub fn scene_mdl(objs: &[Object], ctx: &Ctx3) -> f64 {
    let st = State::new(objs.to_vec());
    let m = render(&st, ctx);
    let s = *ctx.levels.last().unwrap();
    rho_sum(&residual_of(&m, ctx, (0, 0, ctx.size.0, ctx.size.1), s)) / (ctx.kappa(s) * ctx.tau) + objs.iter().map(scene::program_length).sum::<f64>()
}

/// The residual's σ at every level: the render's own noise (two seeds,
/// blurred as the model is) and the photo's (white noise of `ctx.sigma`,
/// blurred as the target is), in quadrature.
pub fn noise_levels(st: &State, ctx: &Ctx3) -> Vec<f64> {
    let c = compare(st, ctx);
    let (a, b) = (c.render(&ctx.cam, ctx.spp, ctx.seed), c.render(&ctx.cam, ctx.spp, ctx.seed + 1));
    let d: Vec<f64> = a.iter().zip(&b).flat_map(|(x, y)| (0..3).map(move |k| (x[k] - y[k]) / 2f64.sqrt())).collect();
    let mut rng = irys_recon::jac::Mix(11);
    let white: Vec<f64> = (0..d.len())
        .map(|_| {
            let (u1, u2) = (rng.uniform().max(1e-12), rng.uniform());
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        })
        .collect();
    let mad = |v: &[f64]| {
        let mut x: Vec<f64> = v.iter().map(|q| q.abs()).collect();
        x.sort_by(f64::total_cmp);
        1.4826 * x[x.len() / 2]
    };
    ctx.levels
        .iter()
        .map(|&s| {
            let d0 = ctx.app.defocus * ctx.size.0 as f64;
            let sr = mad(&band(&blur_win(&d, ctx.size, d0), ctx.size, s, ctx.highpass));
            let sp = ctx.sigma * mad(&band(&white, ctx.size, s, ctx.highpass));
            (sr * sr + sp * sp).sqrt().max(1e-5)
        })
        .collect()
}

/// The dispersion of the residual for `st` at the finest level: its Huber
/// sum per independent measurement over the frame, over the noise-only
/// value (½ per measurement), at least 1.
pub fn dispersion(st: &State, ctx: &Ctx3) -> f64 {
    let m = render(st, ctx);
    let s = *ctx.levels.last().unwrap();
    let r = residual_of(&m, ctx, (0, 0, ctx.size.0, ctx.size.1), s);
    let n_eff = r.len() as f64 / ctx.kappa(s);
    (rho_sum(&r) / ctx.kappa(s) / (0.5 * n_eff)).max(1.0)
}

/// The camera's pitch over the support plane, refined on the whole frame
/// (the network's plane head is the largest single error of the net-camera
/// mode). Every object keeps its place in the image: its base centre is
/// re-cast through the new camera, and the move goes through the TLAS only
/// ([`Compare3::set_poses`]). The height is left alone: with everything
/// scaled together the image does not change, so only a gauge can set it.
/// Returns the camera and the re-posed objects.
pub fn refine_pitch(objs: &[Object], ctx: &Ctx3, span_deg: f64) -> (Cam, Vec<Object>) {
    let cam0 = ctx.cam;
    let [lh, p0] = cam0.plane();
    let repose = |cam: &Cam| -> Vec<Object> {
        objs.iter()
            .map(|o| {
                let mut o = o.clone();
                if let Some(px) = cam0.project([o.pose[0], o.pose[1], 0.0]) {
                    if let Some(xy) = cam.on_plane(px[0], px[1]) {
                        o.pose[0] = xy[0];
                        o.pose[1] = xy[1];
                    }
                }
                o
            })
            .collect()
    };
    let st = State::new(objs.to_vec());
    let mut cmp = compare(&st, ctx);
    let present: Vec<bool> = vec![true; objs.len()];
    let s = ctx.levels[ctx.levels.len() / 2];
    let full = (0, 0, ctx.size.0, ctx.size.1);
    let mut eval = |d: f64| -> f64 {
        let cam = Cam { size: cam0.size, ..Cam::from_plane([lh, p0 + d.to_radians()], cam0.hfov, cam0.size) };
        let os = repose(&cam);
        cmp.set_poses(&os, &present);
        let m = cmp.render(&cam, ctx.spp, ctx.seed);
        rho_sum(&residual_of(&m, ctx, full, s))
    };
    let mut best = (eval(0.0), 0.0);
    let n = 6;
    for k in -n..=n {
        let d = span_deg * k as f64 / n as f64;
        if k != 0 {
            let c = eval(d);
            if c < best.0 {
                best = (c, d);
            }
        }
    }
    let step = span_deg / n as f64;
    for d in [best.1 - step / 2.0, best.1 + step / 2.0] {
        let c = eval(d);
        if c < best.0 {
            best = (c, d);
        }
    }
    let cam = Cam { size: cam0.size, ..Cam::from_plane([lh, p0 + best.1.to_radians()], cam0.hfov, cam0.size) };
    (cam, repose(&cam))
}

/// Seeds for an object behind an unexplained image region `bbox`
/// (`[x0, y0, x1, y1]` at `size`): its bottom edge is where it meets the
/// support plane, its top edge gives its height on the vertical through
/// that point, and each family's default is sized to that.
pub fn birth_candidates(bbox: [u32; 4], cam: &Cam, size: (usize, usize)) -> Vec<Object> {
    let cam = cam.resized(size);
    let (x0, y0, x1, y1) = (bbox[0] as f64, bbox[1] as f64, bbox[2] as f64 + 1.0, bbox[3] as f64 + 1.0);
    let xm = 0.5 * (x0 + x1);
    let (Some(p), Some(pl), Some(pr)) = (cam.on_plane(xm, y1 - 0.5), cam.on_plane(x0, y1 - 0.5), cam.on_plane(x1, y1 - 0.5)) else {
        return Vec::new();
    };
    let w = (pr[0] - pl[0]).abs().hypot(pr[1] - pl[1]).clamp(15.0, 900.0);
    // Height: the top ray's closest approach to the vertical through p.
    let d = cam.dir(xm, y0 + 0.5);
    let e = cam.eye;
    let (ox, oy) = (e[0] - p[0], e[1] - p[1]);
    let dh = (d[0] * d[0] + d[1] * d[1]).max(1e-9);
    let u = -(ox * d[0] + oy * d[1]) / dh;
    let h = (e[2] + u * d[2]).clamp(3.0, 600.0);
    let mat = scene::Mat { rgb: [0.5; 3], rough: 0.5, metal: false };
    use scene::Class::*;
    let mk = |class: scene::Class, tokens: [usize; 4], params: Vec<f64>, depth: f64| {
        let mut o = Object { class, family: class.family(), tokens, params, pose: [p[0], p[1] + 0.5 * depth, 0.0], mat };
        crate::infer::repair(&mut o);
        o
    };
    let r = 0.5 * w;
    vec![
        mk(Can, [1, 0, 0, 0], vec![h, r, r, r, 0.85 * r], w),
        mk(Mouse, [2, 1, 0, 0], vec![w, 1.6 * w, h, 0.3 * h.min(w), 50.0], 1.6 * w),
        mk(Speaker, [2, 0, 0, 0], vec![w, 0.8 * w, h, 0.05 * w, 50.0], 0.8 * w),
        mk(Keyboard, [3, 0, 5, 0], vec![w, (0.33 * w).max(5.0 * 19.05 + 12.0), (0.6 * h).max(8.0), 19.05, (0.4 * h).max(5.0)], 0.33 * w),
        mk(Tray, [4, 0, 0, 0], vec![w, 0.7 * w, h, 2.5, 10.0], 0.7 * w),
        mk(Bracket, [0, 0, 2, 1], vec![w, 0.6 * w, h.min(8.0), 3.0, 0.06 * w, 0.4 * w], 0.6 * w),
    ]
}

/// Births: regions the fitted scene leaves unexplained become new
/// objects when an object there shortens the description. The residual at
/// the middle level is thresholded (`k` σ, mean over channels), split into
/// components (M1's residual pass's union–find), and each component of at
/// least `min_px` pixels whose bottom edge is on the support plane is
/// offered every family's seed; the shortest description, or none, wins.
pub fn births(st: State, ctx: &Ctx3, iters: usize, k: f64, min_px: u32, max_new: usize) -> (State, Vec<(usize, Fitted)>) {
    use irys_diff::look::residual::{Noise, components};
    let mut st = st;
    let s = ctx.levels[ctx.levels.len() / 2];
    let full = (0, 0, ctx.size.0, ctx.size.1);
    let m = render(&st, ctx);
    let r = residual_of(&m, ctx, full, s);
    let (w, h) = ctx.size;
    let e: Vec<f64> = (0..w * h).map(|p| (r[3 * p].abs() + r[3 * p + 1].abs() + r[3 * p + 2].abs()) / 3.0).collect();
    // The threshold: `k` σ, or `k` times the frame's median residual where the
    // model is poor everywhere (a real photo).
    let mut sorted = e.clone();
    sorted.sort_by(f64::total_cmp);
    let thr = k.max(k * sorted[sorted.len() / 2]);
    // Only the support plane's near part (within `reach` mm of the camera's
    // foot): births stand on it.
    let cs = ctx.cam.resized(ctx.size);
    let reach = 1500.0;
    let mask: Vec<u8> = (0..w * h)
        .map(|p| {
            let near = cs.on_plane((p % w) as f64 + 0.5, (p / w) as f64 + 0.5).is_some_and(|q| (q[0] - cs.eye[0]).hypot(q[1] - cs.eye[1]) < reach);
            (!(near && e[p] > thr)) as u8
        })
        .collect();
    let z = vec![0.0f32; w * h];
    let mut blobs = components(&mask, &z, ctx.size, &Noise { min_pixels: min_px, ..Noise::default() });
    blobs.sort_by(|a, b| b.pixels.cmp(&a.pixels));
    if std::env::var("V3_DEBUG").is_ok() {
        eprintln!("births: {} components ≥ {min_px} px; unexplained {:.1} %", blobs.len(), 100.0 * mask.iter().filter(|&&m| m == 0).count() as f64 / mask.len() as f64);
    }
    let cam = ctx.cam.resized(ctx.size);
    let mut born = Vec::new();
    let mut n = Counts::default();
    for b in blobs.iter().take(4 * max_new) {
        if born.len() >= max_new {
            break;
        }
        let bb = b.bbox;
        // Standing on the plane: its bottom row must see the plane.
        if cam.on_plane(0.5 * (bb[0] + bb[2]) as f64, bb[3] as f64 + 0.5).is_none_or(|q| (q[0] - cam.eye[0]).hypot(q[1] - cam.eye[1]) > 4000.0) {
            continue;
        }
        let cands = birth_candidates(bb, &ctx.cam, ctx.size);
        if cands.is_empty() {
            continue;
        }
        let i = st.objs.len();
        let mut objs = st.objs.clone();
        objs.push(cands[0].clone());
        let mut grown = State::new(objs);
        let mut app = ctx.app.clone();
        app.mats.resize(i + 1, scene::Mat { rgb: [0.5; 3], rough: 0.5, metal: false });
        let mut sd = ctx.param_sd.clone();
        sd.resize(i + 1, Vec::new());
        let c2 = Ctx3 { app, param_sd: sd, target: ctx.target, sig_levels: ctx.sig_levels.clone(), levels: ctx.levels.clone(), exact: ctx.exact.clone(), ..*ctx };
        let ws: Vec<&Object> = cands.iter().collect();
        let Some(win) = window(&ws, &cam, ctx.size, 10.0) else { continue };
        let none = dropped(&grown, i, &c2, win, &mut n);
        let mut best = none.clone();
        for c in &cands {
            let f = fit(&grown, i, c, scene::structure_nats(c), &c2, win, iters, &mut n);
            if std::env::var("V3_DEBUG").is_ok() {
                eprintln!("    {} mdl {:.0} (r {:.0} p {:.0} l {:.0}) {}", c.class.name(), f.mdl, f.residual, f.param_cost, f.length, f.object.as_ref().map(|o| o.to_text()).unwrap_or_default());
            }
            if f.mdl < best.mdl {
                best = f;
            }
        }
        if std::env::var("V3_DEBUG").is_ok() {
            eprintln!("birth bbox {:?} px {}: none {:.0} best {:.0} {}", bb, b.pixels, none.mdl, best.mdl, best.object.as_ref().map(|o| o.to_text()).unwrap_or_default());
        }
        if let Some(o) = best.object.clone() {
            grown = grown.with(i, Some(&o));
            st = grown;
            born.push((i, best));
        }
    }
    (st, born)
}

/// The scene's metric scale where nothing in it is a gauge: the residual
/// does not change when the camera and every object scale together, so the
/// scale is the priors' to set. Each kept object that is still coded with
/// the structure it started with contributes its parameters' log ratios to
/// the network's means, weighted by the head's precision; the result is the
/// factor `s` that, applied with [`crate::infer::rescale`], minimises the
/// prior term. `init[i]` and `sd[i]` are object `i`'s start and σ (log units).
pub fn prior_scale(st: &State, init: &[Object], sd: &[Vec<f64>]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..st.objs.len().min(init.len()) {
        if !st.present(i) {
            continue;
        }
        let (o, o0) = (&st.objs[i], &init[i]);
        if o.family != o0.family || o.tokens != o0.tokens {
            continue;
        }
        for k in 0..o.params.len() {
            if !scene::param_used(o.family, &o.tokens, k) || matches!((o.family, k), (scene::Family::Grid, 3)) {
                continue;
            }
            let s = sd.get(i).and_then(|v| v.get(k)).copied().unwrap_or(0.3).max(0.02);
            let w = 1.0 / (s * s);
            num += w * (o0.params[k].ln() - o.params[k].ln());
            den += w;
        }
    }
    if den > 0.0 { (num / den).exp().clamp(0.5, 2.0) } else { 1.0 }
}
