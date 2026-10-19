//! The MDL search over scenes: every object's structure is chosen among
//! candidates (the network's top-k, or every family for the baseline), its
//! continuous parameters fitted by Levenberg–Marquardt on the irys-cast
//! model image, and the candidate with the shortest description kept:
//!
//!     MDL = Σ ρ(r/σ)  +  ½ log det(F + I)  +  program length     (nats)
//!
//! over the object's window, with `ρ` Huber, `F = JᵀWJ` the Fisher
//! information of its normalised parameters, and the program length of
//! [`scene::program_length`]. Dropping the object is always a candidate, so
//! a false detection costs its residual and nothing else.
//!
//! The Jacobian is by forward differences of the cast image (3 × 3
//! supersampled, blurred): the scene families are not v0's carrier
//! surfaces, so irys-recon's edge-term Jacobian does not apply here.

use std::sync::Arc;

use irys_core::{Bvh, TriMesh};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::render::{Caster, Compare, TABLE, appearance, blas};
use crate::scene::{self, Cam, Family, Object, mesh_fast as mesh, param_used};

/// What one search sees.
pub struct Ctx<'a> {
    pub target: &'a [[f64; 3]],
    pub size: (usize, usize),
    pub cam: Cam,
    pub ss: usize,
    /// Noise σ of the residual (linear units).
    pub sigma: f64,
    /// Blur σ (px) of model and target.
    pub blur: f64,
}

/// A scene estimate with its BLASes.
#[derive(Clone)]
pub struct State {
    pub objs: Vec<Object>,
    blas: Vec<Option<Arc<Bvh<TriMesh>>>>,
}

pub fn world(o: &Object) -> scene::Tris {
    mesh(o).map(|m| m.posed(o.pose)).unwrap_or_default()
}

impl State {
    pub fn new(objs: Vec<Object>) -> State {
        let blas = objs.par_iter().map(|o| blas(&world(o))).collect();
        State { objs, blas }
    }
    pub fn with(&self, i: usize, o: Option<&Object>) -> State {
        let mut s = self.clone();
        match o {
            Some(o) => {
                s.objs[i] = o.clone();
                s.blas[i] = blas(&world(o));
            }
            None => s.blas[i] = None,
        }
        s
    }
    pub fn caster(&self) -> Caster {
        Caster::from_blas(&self.blas)
    }
    pub fn present(&self, i: usize) -> bool {
        self.blas[i].is_some()
    }
    /// Objects that are present.
    pub fn kept(&self) -> Vec<Object> {
        self.objs.iter().zip(&self.blas).filter(|(_, b)| b.is_some()).map(|(o, _)| o.clone()).collect()
    }
}

pub type Window = (usize, usize, usize, usize);

/// The model image over `win` (appearance fitted to the target there).
pub fn model(st: &State, ctx: &Ctx, win: Window) -> Vec<[f64; 3]> {
    let cmp = Compare { size: ctx.size, ss: ctx.ss };
    let cast = cmp.cast(&st.caster(), &ctx.cam, Some(win));
    let t = crop(ctx.target, ctx.size, win);
    appearance(&cast, &t)
}

pub fn crop(img: &[[f64; 3]], size: (usize, usize), (x0, y0, x1, y1): Window) -> Vec<[f64; 3]> {
    let mut out = Vec::with_capacity((x1 - x0) * (y1 - y0));
    for y in y0..y1 {
        for x in x0..x1 {
            out.push(img[y * size.0 + x]);
        }
    }
    out
}

fn blur3(img: &[[f64; 3]], size: (usize, usize), s: f64) -> Vec<f64> {
    let flat: Vec<f64> = img.iter().flat_map(|p| p.iter().copied()).collect();
    if s <= 0.0 { flat } else { irys_recon::fit::blur(&flat, size, 3, s) }
}

/// Residuals `(model − target)/σ` (blurred) over the window.
pub fn residual(st: &State, ctx: &Ctx, win: Window) -> Vec<f64> {
    let m = model(st, ctx, win);
    let t = crop(ctx.target, ctx.size, win);
    let sz = (win.2 - win.0, win.3 - win.1);
    let (mb, tb) = (blur3(&m, sz, ctx.blur), blur3(&t, sz, ctx.blur));
    mb.iter().zip(&tb).map(|(a, b)| (a - b) / ctx.sigma).collect()
}

const HUBER: f64 = 2.0;

pub fn rho_sum(r: &[f64]) -> f64 {
    r.iter().map(|&x| irys_recon::look::huber_rho(x, HUBER)).sum()
}

/// The window an object covers (projected bounding box plus a margin).
pub fn window(objs: &[&Object], cam: &Cam, size: (usize, usize), margin: f64) -> Option<Window> {
    let cam = cam.resized(size);
    let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for o in objs {
        let r = o.radius() * 1.05;
        let h = o.height();
        for (dx, dy) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            for z in [0.0, h] {
                if let Some(p) = cam.project([o.pose[0] + dx * r, o.pose[1] + dy * r, z]) {
                    x0 = x0.min(p[0]);
                    y0 = y0.min(p[1]);
                    x1 = x1.max(p[0]);
                    y1 = y1.max(p[1]);
                }
            }
        }
    }
    let (w, h) = (size.0 as f64, size.1 as f64);
    let x0 = (x0 - margin).clamp(0.0, w);
    let y0 = (y0 - margin).clamp(0.0, h);
    let x1 = (x1 + margin).clamp(0.0, w);
    let y1 = (y1 + margin).clamp(0.0, h);
    if x1 - x0 < 3.0 || y1 - y0 < 3.0 {
        return None;
    }
    Some((x0 as usize, y0 as usize, x1.ceil() as usize, y1.ceil() as usize))
}

/// The fitted variables of an object: position (per 50 mm), yaw (per 20°)
/// when observable, and the log of every used parameter.
pub fn encode(o: &Object) -> Vec<f64> {
    let mut v = vec![o.pose[0] / 50.0, o.pose[1] / 50.0];
    if o.family.has_yaw() {
        v.push(o.pose[2] / 20.0);
    }
    for k in 0..o.params.len() {
        if param_used(o.family, &o.tokens, k) && !fixed(o, k) {
            v.push(o.params[k].ln());
        }
    }
    v
}

/// Parameters the search leaves alone: the keyboard's pitch (a gauge) and
/// the plate's hole pattern pitch when there is one hole.
fn fixed(o: &Object, k: usize) -> bool {
    matches!((o.family, k), (Family::Grid, 3))
}

pub fn decode(o: &Object, v: &[f64]) -> Object {
    let mut o = o.clone();
    o.pose[0] = v[0] * 50.0;
    o.pose[1] = v[1] * 50.0;
    let mut i = 2;
    if o.family.has_yaw() {
        o.pose[2] = v[2] * 20.0;
        i = 3;
    }
    for k in 0..o.params.len() {
        if param_used(o.family, &o.tokens, k) && !fixed(&o, k) {
            o.params[k] = v[i].clamp(-2.0, 8.0).exp();
            i += 1;
        }
    }
    crate::infer::repair(&mut o);
    o
}

/// A fitted candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fitted {
    pub object: Option<Object>,
    pub residual: f64,
    pub param_cost: f64,
    pub length: f64,
    pub mdl: f64,
    /// σ of every fitted variable (normalised units), from `F⁻¹`.
    pub sigma: Vec<f64>,
    pub evals: usize,
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
    // Cholesky of F + I.
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

fn inv_diag(f: &[Vec<f64>]) -> Vec<f64> {
    let n = f.len();
    (0..n)
        .map(|i| {
            let mut e = vec![0.0; n];
            e[i] = 1.0;
            let mut a = f.to_vec();
            for (k, r) in a.iter_mut().enumerate() {
                r[k] += 1e-9;
            }
            solve(&a, &e).map(|x| x[i].max(0.0).sqrt()).unwrap_or(f64::INFINITY)
        })
        .collect()
}

/// Levenberg–Marquardt on object `i` of `st` (others fixed) over `win`.
pub fn fit(st: &State, i: usize, init: &Object, prior: f64, ctx: &Ctx, win: Window, iters: usize) -> Fitted {
    let mut v = encode(init);
    let n = v.len();
    let eval = |v: &[f64]| -> (Object, Vec<f64>) {
        let o = decode(init, v);
        let r = residual(&st.with(i, Some(&o)), ctx, win);
        (o, r)
    };
    let (mut obj, mut r) = eval(&v);
    let mut cost = rho_sum(&r);
    let mut lambda = 1e-2;
    let mut evals = 1;
    let mut jtwj = vec![vec![0.0; n]; n];
    for _ in 0..iters {
        let h = 0.04;
        let cols: Vec<Vec<f64>> = (0..n)
            .into_par_iter()
            .map(|k| {
                let mut vp = v.clone();
                vp[k] += h;
                let (_, rp) = eval(&vp);
                rp.iter().zip(&r).map(|(a, b)| (a - b) / h).collect()
            })
            .collect();
        evals += n;
        let w: Vec<f64> = r.iter().map(|&x| irys_recon::look::huber_w(x, HUBER)).collect();
        let mut g = vec![0.0; n];
        for a in 0..n {
            for b in 0..=a {
                let s: f64 = (0..r.len()).map(|p| cols[a][p] * w[p] * cols[b][p]).sum();
                jtwj[a][b] = s;
                jtwj[b][a] = s;
            }
            g[a] = (0..r.len()).map(|p| cols[a][p] * w[p] * r[p]).sum();
        }
        let mut improved = false;
        for _ in 0..4 {
            let mut a = jtwj.clone();
            for k in 0..n {
                a[k][k] += lambda * (jtwj[k][k] + 1e-3);
            }
            let Some(d) = solve(&a, &g.iter().map(|x| -x).collect::<Vec<_>>()) else {
                lambda *= 10.0;
                continue;
            };
            let d: Vec<f64> = d.iter().map(|x| x.clamp(-0.5, 0.5)).collect();
            let vn: Vec<f64> = v.iter().zip(&d).map(|(a, b)| a + b).collect();
            let (on, rn) = eval(&vn);
            evals += 1;
            let cn = rho_sum(&rn);
            if cn < cost {
                v = vn;
                obj = on;
                r = rn;
                let rel = (cost - cn) / cost.max(1e-9);
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
    let param_cost = 0.5 * logdet_plus_i(&jtwj);
    let length = scene::program_length_with(&obj, prior);
    Fitted { mdl: cost + param_cost + length, residual: cost, param_cost, length, sigma: inv_diag(&jtwj), object: Some(obj), evals }
}

/// Cost of the window with object `i` dropped.
pub fn dropped(st: &State, i: usize, ctx: &Ctx, win: Window) -> Fitted {
    let r = residual(&st.with(i, None), ctx, win);
    let c = rho_sum(&r);
    Fitted { object: None, residual: c, param_cost: 0.0, length: 0.0, mdl: c, sigma: Vec::new(), evals: 1 }
}

/// For every object in turn (largest first), fit each candidate structure
/// and keep the shortest description. `cands[i]` are object `i`'s
/// candidates, each with its structure's prior in nats (`−log p`; the
/// network's, or [`scene::structure_nats`] for the grammar alone).
pub fn search(objs: Vec<(Object, f64)>, cands: &[Vec<(Object, f64)>], ctx: &Ctx, iters: usize, sweeps: usize) -> (State, Vec<Fitted>, usize) {
    let mut priors: Vec<f64> = objs.iter().map(|o| o.1).collect();
    let mut st = State::new(objs.into_iter().map(|o| o.0).collect());
    let mut order: Vec<usize> = (0..st.objs.len()).collect();
    order.sort_by(|&a, &b| st.objs[b].radius().total_cmp(&st.objs[a].radius()));
    let mut best_fit: Vec<Option<Fitted>> = vec![None; st.objs.len()];
    let mut evals = 0;
    for _ in 0..sweeps {
        for &i in &order {
            let cur = (st.objs[i].clone(), priors[i]);
            let mut cs: Vec<&(Object, f64)> = cands[i].iter().collect();
            if st.present(i) {
                cs.insert(0, &cur);
            }
            let ws: Vec<&Object> = cs.iter().map(|c| &c.0).collect();
            let Some(win) = window(&ws, &ctx.cam, ctx.size, 4.0) else {
                continue;
            };
            let mut best = dropped(&st, i, ctx, win);
            let mut best_prior = priors[i];
            for c in cs {
                let f = fit(&st, i, &c.0, c.1, ctx, win, iters);
                evals += f.evals;
                if f.mdl < best.mdl {
                    best = f;
                    best_prior = c.1;
                }
            }
            priors[i] = best_prior;
            st = st.with(i, best.object.as_ref());
            best_fit[i] = Some(best);
        }
    }
    let fits = best_fit.into_iter().map(|f| f.unwrap_or(Fitted { object: None, residual: 0.0, param_cost: 0.0, length: 0.0, mdl: 0.0, sigma: Vec::new(), evals: 0 })).collect();
    (st, fits, evals)
}

/// The MDL of a whole scene estimate on the full frame (residual + every
/// kept object's program length; no Fisher term): the score used to
/// compare estimates without labels.
pub fn scene_mdl(objs: &[Object], ctx: &Ctx) -> f64 {
    let st = State::new(objs.to_vec());
    let r = residual(&st, ctx, (0, 0, ctx.size.0, ctx.size.1));
    rho_sum(&r) + objs.iter().map(scene::program_length).sum::<f64>()
}

/// A robust noise σ: the MAD of the table-only residual's pixel differences.
pub fn noise_sigma(target: &[[f64; 3]], size: (usize, usize)) -> f64 {
    let mut d: Vec<f64> = Vec::new();
    for y in 0..size.1 {
        for x in 1..size.0 {
            let (a, b) = (target[y * size.0 + x], target[y * size.0 + x - 1]);
            for c in 0..3 {
                d.push((a[c] - b[c]).abs());
            }
        }
    }
    d.sort_by(f64::total_cmp);
    (1.4826 * d[d.len() / 2] / 2f64.sqrt()).max(0.004)
}

// ---------------------------------------------------------------------------
// The no-network baseline: segment, then try every family.
// ---------------------------------------------------------------------------

/// Blobs that differ from the table colour: `(pixels)` per component.
pub fn segment(target: &[[f64; 3]], size: (usize, usize), min_px: usize) -> Vec<Vec<usize>> {
    let (w, h) = size;
    let mut border: Vec<[f64; 3]> = Vec::new();
    for x in 0..w {
        border.push(target[(h - 1) * w + x]);
        border.push(target[x + w * (h * 2 / 3)]);
    }
    let med = |c: usize| {
        let mut v: Vec<f64> = border.iter().map(|p| p[c]).collect();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let bg = [med(0), med(1), med(2)];
    let lum = (bg[0] + bg[1] + bg[2]) / 3.0;
    let fg: Vec<bool> = target
        .iter()
        .map(|p| {
            let d = ((p[0] - bg[0]).powi(2) + (p[1] - bg[1]).powi(2) + (p[2] - bg[2]).powi(2)).sqrt();
            d > 0.12 * lum.max(0.05) + 0.02
        })
        .collect();
    let mut seen = vec![false; w * h];
    let mut out = Vec::new();
    for s in 0..w * h {
        if !fg[s] || seen[s] {
            continue;
        }
        let mut stack = vec![s];
        seen[s] = true;
        let mut comp = Vec::new();
        while let Some(k) = stack.pop() {
            comp.push(k);
            let (x, y) = (k % w, k / w);
            let nb = [(x.wrapping_sub(1), y), (x + 1, y), (x, y.wrapping_sub(1)), (x, y + 1)];
            for (nx, ny) in nb {
                if nx < w && ny < h {
                    let q = ny * w + nx;
                    if fg[q] && !seen[q] {
                        seen[q] = true;
                        stack.push(q);
                    }
                }
            }
        }
        if comp.len() >= min_px {
            out.push(comp);
        }
    }
    out
}

/// Every family's default guess for a blob seen through `cam`.
pub fn baseline_candidates(blob: &[usize], size: (usize, usize), cam: &Cam) -> Vec<Object> {
    let cam = cam.resized(size);
    let w = size.0;
    let mut pts: Vec<[f64; 2]> = Vec::new();
    let mut ys: Vec<usize> = blob.iter().map(|k| k / w).collect();
    ys.sort();
    let y_lo = ys[ys.len() * 3 / 4];
    for &k in blob {
        let (x, y) = (k % w, k / w);
        if y >= y_lo {
            if let Some(p) = cam.on_plane(x as f64 + 0.5, y as f64 + 0.5) {
                pts.push(p);
            }
        }
    }
    let all: Vec<[f64; 2]> = blob.iter().filter_map(|&k| cam.on_plane((k % w) as f64 + 0.5, (k / w) as f64 + 0.5)).collect();
    if pts.is_empty() || all.is_empty() {
        return Vec::new();
    }
    let cx = pts.iter().map(|p| p[0]).sum::<f64>() / pts.len() as f64;
    let cy = pts.iter().map(|p| p[1]).sum::<f64>() / pts.len() as f64;
    // Width from the blob's lower quarter (on or near the plane): the rest
    // projects past the object.
    let xs: Vec<f64> = pts.iter().map(|p| p[0]).collect();
    let _ = &all;
    let span = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - xs.iter().cloned().fold(f64::INFINITY, f64::min);
    let s = span.clamp(10.0, 800.0);
    let cyc = cy + 0.25 * s;
    let mat = scene::Mat { rgb: [0.5; 3], rough: 0.5, metal: false };
    use scene::Class::*;
    let mk = |class: scene::Class, tokens: [usize; 4], params: Vec<f64>| {
        let mut o = Object { class, family: class.family(), tokens, params, pose: [cx, cyc, 0.0], mat };
        crate::infer::repair(&mut o);
        o
    };
    vec![
        mk(Bracket, [0, 0, 2, 1], vec![s, 0.6 * s, 5.0, 3.0, 0.06 * s, 0.4 * s]),
        mk(Can, [1, 0, 0, 0], vec![0.8 * s, 0.5 * s, 0.5 * s, 0.5 * s, 0.45 * s]),
        mk(Speaker, [2, 0, 0, 0], vec![s, 0.7 * s, 0.6 * s, 0.08 * s, 50.0]),
        mk(Mouse, [2, 1, 0, 0], vec![s, 0.7 * s, 0.4 * s, 0.15 * s, 50.0]),
        mk(Keyboard, [3, 0, 5, 0], vec![s, 0.35 * s, 15.0, 19.05, 8.0]),
        mk(Tray, [4, 0, 0, 0], vec![s, 0.7 * s, 0.3 * s, 2.5, 10.0]),
        mk(HoneycombCase, [4, 1, 0, 0], vec![s, 0.7 * s, 0.3 * s, 2.5, 10.0]),
    ]
}

/// Inventory overlay colour per id.
pub fn colour(i: u16) -> [u8; 3] {
    if i >= crate::render::BACKDROP {
        return [0, 0, 0];
    }
    let h = (i as u32 + 1).wrapping_mul(2654435761);
    [(h >> 8) as u8 | 80, (h >> 16) as u8 | 80, (h >> 24) as u8 | 80]
}
