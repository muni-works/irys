//! The inner loop: Levenberg–Marquardt over a program's continuous
//! parameters, coarse to fine.
//!
//! The residual stacks two terms per view and pixel:
//!
//! - the image term `(G_σ * (I(θ) − T)) / σ_n`, the blurred grey difference
//!   in units of the render noise; its Jacobian is `G_σ * J` (blurring is
//!   linear), `J` from [`crate::jac`];
//! - the silhouette term `λ · D(p) · (c_θ(p) − m(p))`, `c_θ` the part's
//!   coverage, `m` the target's mask (background subtraction against the
//!   empty studio) and `D` the distance in pixels to `m`'s boundary — a
//!   distance-transform loss whose gradient points the silhouette at the
//!   target from far outside the image term's basin.
//!
//! The schedule is a blur homotopy: `σ` from a few pixels to none, `λ`
//! annealed to 0, so the final level — the one the score is read at — is the
//! image term alone. A step is clamped to the parameter bounds; one that
//! leaves the valid program set ([`Program::valid`], or a vcad build error)
//! is rejected like one that does not descend.

use crate::dsl::Program;
use crate::jac::ViewJac;
use crate::model::{Eval, Renderer, Setup};

/// A target as the solver sees it: grey images, a silhouette and its
/// distance transform per view.
#[derive(Clone)]
pub struct Target {
    /// Grey image per view.
    pub img: Vec<Vec<f64>>,
    /// Background-subtracted mask per view.
    pub mask: Vec<Vec<f64>>,
    /// Distance (pixels) to the mask's boundary, plus one.
    pub dist: Vec<Vec<f64>>,
    /// Image size.
    pub size: (usize, usize),
}

impl Target {
    /// From grey images and the empty studio's.
    pub fn new(img: Vec<Vec<f64>>, background: &[Vec<f64>], size: (usize, usize)) -> Self {
        let mask: Vec<Vec<f64>> = img
            .iter()
            .zip(background)
            .map(|(t, b)| {
                let m: Vec<f64> = t.iter().zip(b).map(|(a, b)| ((a - b).abs() > 0.03) as u8 as f64).collect();
                open_close(&m, size)
            })
            .collect();
        let dist = mask.iter().map(|m| boundary_distance(m, size)).collect();
        Self { img, mask, dist, size }
    }

    /// Box-downsampled by 2 (a 2 × 2 box is the pixel of the half-size
    /// camera).
    pub fn half(&self, background: &[Vec<f64>]) -> Self {
        let (w, h) = self.size;
        let (w2, h2) = (w / 2, h / 2);
        let down = |v: &Vec<f64>| -> Vec<f64> {
            (0..w2 * h2)
                .map(|i| {
                    let (x, y) = (2 * (i % w2), 2 * (i / w2));
                    (v[y * w + x] + v[y * w + x + 1] + v[(y + 1) * w + x] + v[(y + 1) * w + x + 1]) / 4.0
                })
                .collect()
        };
        let img = self.img.iter().map(down).collect();
        let bg: Vec<Vec<f64>> = background.iter().map(down).collect();
        Self::new(img, &bg, (w2, h2))
    }
}

/// A 3 × 3 majority filter: drops speckle from noise in the subtraction.
fn open_close(m: &[f64], (w, h): (usize, usize)) -> Vec<f64> {
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as i64, (i / w) as i64);
            let mut s = 0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (xx, yy) = ((x + dx).clamp(0, w as i64 - 1), (y + dy).clamp(0, h as i64 - 1));
                    s += m[yy as usize * w + xx as usize] as i32;
                }
            }
            (s >= 5) as u8 as f64
        })
        .collect()
}

/// Two-pass chamfer distance (3-4) to the nearest pixel of the other class,
/// in pixels, plus one.
pub fn boundary_distance(m: &[f64], (w, h): (usize, usize)) -> Vec<f64> {
    let inf = 1e9;
    let mut d: Vec<f64> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let me = m[i];
            let edge = [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)].iter().any(|&(dx, dy)| {
                let (xx, yy) = (x as i64 + dx, y as i64 + dy);
                xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h && m[yy as usize * w + xx as usize] != me
            });
            if edge { 0.0 } else { inf }
        })
        .collect();
    let at = |d: &Vec<f64>, x: i64, y: i64| -> f64 {
        if x < 0 || y < 0 || x as usize >= w || y as usize >= h { inf } else { d[y as usize * w + x as usize] }
    };
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            let v = d[y as usize * w + x as usize]
                .min(at(&d, x - 1, y) + 1.0)
                .min(at(&d, x, y - 1) + 1.0)
                .min(at(&d, x - 1, y - 1) + 1.4142)
                .min(at(&d, x + 1, y - 1) + 1.4142);
            d[y as usize * w + x as usize] = v;
        }
    }
    for y in (0..h as i64).rev() {
        for x in (0..w as i64).rev() {
            let v = d[y as usize * w + x as usize]
                .min(at(&d, x + 1, y) + 1.0)
                .min(at(&d, x, y + 1) + 1.0)
                .min(at(&d, x + 1, y + 1) + 1.4142)
                .min(at(&d, x - 1, y + 1) + 1.4142);
            d[y as usize * w + x as usize] = v;
        }
    }
    d.iter().map(|v| v + 1.0).collect()
}

/// Separable Gaussian blur of `cols` interleaved channels, clamped edges.
pub fn blur(v: &[f64], (w, h): (usize, usize), cols: usize, sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 {
        return v.to_vec();
    }
    let r = (3.0 * sigma).ceil() as i64;
    let k: Vec<f64> = (-r..=r).map(|i| (-(i * i) as f64 / (2.0 * sigma * sigma)).exp()).collect();
    let ks: f64 = k.iter().sum();
    let k: Vec<f64> = k.iter().map(|x| x / ks).collect();
    let mut tmp = vec![0.0; v.len()];
    for y in 0..h {
        for x in 0..w {
            for c in 0..cols {
                let mut s = 0.0;
                for (i, kv) in k.iter().enumerate() {
                    let xx = (x as i64 + i as i64 - r).clamp(0, w as i64 - 1) as usize;
                    s += kv * v[(y * w + xx) * cols + c];
                }
                tmp[(y * w + x) * cols + c] = s;
            }
        }
    }
    let mut out = vec![0.0; v.len()];
    for y in 0..h {
        for x in 0..w {
            for c in 0..cols {
                let mut s = 0.0;
                for (i, kv) in k.iter().enumerate() {
                    let yy = (y as i64 + i as i64 - r).clamp(0, h as i64 - 1) as usize;
                    s += kv * tmp[(yy * w + x) * cols + c];
                }
                out[(y * w + x) * cols + c] = s;
            }
        }
    }
    out
}

/// One level of the homotopy.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Level {
    /// Blur, pixels.
    pub sigma: f64,
    /// Silhouette weight.
    pub lambda: f64,
    /// Most LM iterations.
    pub iters: usize,
}

/// The default schedule.
pub fn schedule() -> Vec<Level> {
    vec![
        Level { sigma: 3.0, lambda: 0.6, iters: 5 },
        Level { sigma: 1.5, lambda: 0.15, iters: 5 },
        Level { sigma: 0.0, lambda: 0.0, iters: 8 },
    ]
}

/// A short schedule for candidates whose program is a warm start.
pub fn short() -> Vec<Level> {
    vec![
        Level { sigma: 2.0, lambda: 0.2, iters: 4 },
        Level { sigma: 0.0, lambda: 0.0, iters: 5 },
    ]
}

/// What a fit returns.
#[derive(Clone)]
pub struct Fit {
    /// The program at the fitted parameters.
    pub prog: Program,
    /// `½ Σ r²` of the image term at the finest level (nats, with the
    /// noise model's constant dropped).
    pub nll: f64,
    /// `JᵀJ` of the image term (unblurred, noise-scaled): the Gauss–Newton
    /// Fisher information.
    pub fisher: Vec<f64>,
    /// Residual image per view (`(I − T)/σ_n`, unblurred).
    pub residual: Vec<Vec<f64>>,
    /// Model image per view at the fit.
    pub img: Vec<Vec<f64>>,
    /// LM iterations taken (accepted or not).
    pub iters: usize,
    /// Renders (evaluations).
    pub evals: usize,
}

/// The fitter: a renderer, a setup, a target and the noise level.
pub struct Fitter {
    /// Setup (film must match the target's size).
    pub setup: Setup,
    /// Target.
    pub target: Target,
    /// Per-pixel noise variance of `I − T`.
    pub sigma2: f64,
    renderer: Renderer,
}

struct Lin {
    cost: f64,
    nll: f64,
    r: Vec<f64>,
    j: Vec<f64>,
}

impl Fitter {
    /// Build one; estimates the noise from two seeds of `probe`'s render.
    pub fn new(setup: Setup, target: Target, target_spp: usize, probe: &Program) -> Self {
        let mut renderer = Renderer::new(setup.views.len());
        let a = renderer.eval(probe, &setup, false).expect("probe builds");
        let mut s2 = setup.clone();
        s2.seed += 1;
        let b = renderer.eval(probe, &s2, false).expect("probe builds");
        let mut acc = 0.0;
        let mut n = 0.0;
        for (x, y) in a.img.iter().zip(&b.img) {
            for (p, q) in x.iter().zip(y) {
                acc += (p - q) * (p - q);
                n += 1.0;
            }
        }
        let model = acc / n / 2.0;
        let sigma2 = model * (1.0 + setup.film.spp as f64 / target_spp as f64);
        Self { setup, target, sigma2, renderer }
    }

    /// With a known noise level (no probe renders).
    pub fn with_sigma2(setup: Setup, target: Target, sigma2: f64) -> Self {
        let renderer = Renderer::new(setup.views.len());
        Self { setup, target, sigma2, renderer }
    }

    /// Render `prog` (with its Jacobian).
    pub fn eval(&mut self, prog: &Program, jac: bool) -> Result<Eval, String> {
        self.renderer.eval(prog, &self.setup, jac)
    }

    fn linearise(&self, e: &Eval, level: Level, np: usize) -> Lin {
        let size = self.target.size;
        let sn = self.sigma2.sqrt();
        let mut r = Vec::new();
        let mut j = Vec::new();
        let mut nll = 0.0;
        let jac = e.jac.as_ref();
        for v in 0..e.img.len() {
            let d: Vec<f64> = e.img[v].iter().zip(&self.target.img[v]).map(|(a, b)| (a - b) / sn).collect();
            let db = blur(&d, size, 1, level.sigma);
            nll += 0.5 * d.iter().map(|x| x * x).sum::<f64>();
            r.extend_from_slice(&db);
            if let Some(jv) = jac {
                let ji: Vec<f64> = jv[v].img.iter().map(|x| x / sn).collect();
                j.extend(blur(&ji, size, np, level.sigma));
            }
            if level.lambda > 0.0 {
                let (dist, m, c) = (&self.target.dist[v], &self.target.mask[v], &e.cov[v]);
                for p in 0..c.len() {
                    r.push(level.lambda * dist[p] * (c[p] - m[p]));
                }
                if let Some(jv) = jac {
                    let ViewJac { mask, .. } = &jv[v];
                    for p in 0..c.len() {
                        for k in 0..np {
                            j.push(level.lambda * dist[p] * mask[p * np + k]);
                        }
                    }
                }
            }
        }
        let cost = 0.5 * r.iter().map(|x| x * x).sum::<f64>();
        Lin { cost, nll, r, j }
    }

    /// Fit `prog`'s continuous parameters over `levels`.
    pub fn fit(&mut self, prog: &Program, levels: &[Level]) -> Result<Fit, String> {
        let info = prog.param_info();
        let np = info.len();
        let mut cur = prog.clone();
        let mut e = self.eval(&cur, true)?;
        let (mut iters, mut evals) = (0, 1);
        for (li, &level) in levels.iter().enumerate() {
            let mut lin = self.linearise(&e, level, np);
            let mut mu = 1e-3;
            let mut it = 0;
            while it < level.iters {
                it += 1;
                iters += 1;
                let (a, g) = normal_eq(&lin.j, &lin.r, np);
                let mut solved = None;
                // Retry the damping within one iteration until a step descends.
                for _ in 0..6 {
                    let mut m = a.clone();
                    for k in 0..np {
                        m[k * np + k] += mu * a[k * np + k].max(1e-9) + 1e-9;
                    }
                    let Some(delta) = solve(&m, &g.iter().map(|x| -x).collect::<Vec<_>>(), np) else {
                        mu *= 4.0;
                        continue;
                    };
                    let th: Vec<f64> = cur
                        .params()
                        .iter()
                        .zip(&delta)
                        .zip(&info)
                        .map(|((t, d), i)| (t + d).clamp(i.lo, i.hi))
                        .collect();
                    let mut cand = cur.clone();
                    cand.set_params(&th);
                    if !cand.valid() {
                        mu *= 4.0;
                        continue;
                    }
                    // The image alone decides; the Jacobian is only built
                    // where a step is taken.
                    let Ok(ce) = self.eval(&cand, false) else {
                        mu *= 4.0;
                        continue;
                    };
                    evals += 1;
                    let cl = self.linearise(&ce, level, np);
                    if cl.cost < lin.cost {
                        let Ok(ce) = self.eval(&cand, true) else {
                            mu *= 4.0;
                            continue;
                        };
                        evals += 1;
                        let cl = self.linearise(&ce, level, np);
                        let step = delta.iter().fold(0.0f64, |m, d| m.max(d.abs()));
                        solved = Some((cand, ce, cl, step));
                        mu = (mu / 3.0).max(1e-7);
                        break;
                    }
                    mu *= 4.0;
                }
                let Some((cand, ce, cl, step)) = solved else { break };
                let rel = (lin.cost - cl.cost) / lin.cost.max(1e-12);
                cur = cand;
                e = ce;
                lin = cl;
                if rel < 1e-4 || step < 2e-3 {
                    break;
                }
            }
            let _ = li;
        }
        // The score's pieces at the finest, unblurred image term.
        let fine = Level { sigma: 0.0, lambda: 0.0, iters: 0 };
        let lin = self.linearise(&e, fine, np);
        let (fisher, _) = normal_eq(&lin.j, &lin.r, np);
        let sn = self.sigma2.sqrt();
        let residual = e
            .img
            .iter()
            .zip(&self.target.img)
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y) / sn).collect())
            .collect();
        Ok(Fit { prog: cur, nll: lin.nll, fisher, residual, img: e.img, iters, evals })
    }
}

/// `(JᵀJ, Jᵀr)`, `J` row-major with `np` columns.
pub fn normal_eq(j: &[f64], r: &[f64], np: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = vec![0.0; np * np];
    let mut g = vec![0.0; np];
    for (row, &ri) in j.chunks_exact(np).zip(r) {
        if row.iter().all(|&x| x == 0.0) {
            continue;
        }
        for p in 0..np {
            g[p] += row[p] * ri;
            for q in p..np {
                a[p * np + q] += row[p] * row[q];
            }
        }
    }
    for p in 0..np {
        for q in 0..p {
            a[p * np + q] = a[q * np + p];
        }
    }
    (a, g)
}

/// Cholesky solve of SPD `a x = b`; `None` if not positive definite.
pub fn solve(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in i + 1..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    Some(x)
}

/// Symmetric eigen-decomposition by cyclic Jacobi: `(values, vectors)` with
/// vectors as columns of a row-major `n × n`.
pub fn eigh(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut m = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..100 {
        let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| m[i * n + j].powi(2)).sum();
        if off < 1e-22 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = m[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let th = 0.5 * (2.0 * apq).atan2(m[q * n + q] - m[p * n + p]);
                let (s, c) = th.sin_cos();
                for k in 0..n {
                    let (x, y) = (m[k * n + p], m[k * n + q]);
                    m[k * n + p] = c * x - s * y;
                    m[k * n + q] = s * x + c * y;
                }
                for k in 0..n {
                    let (x, y) = (m[p * n + k], m[q * n + k]);
                    m[p * n + k] = c * x - s * y;
                    m[q * n + k] = s * x + c * y;
                }
                for k in 0..n {
                    let (x, y) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * x - s * y;
                    v[k * n + q] = s * x + c * y;
                }
            }
        }
    }
    ((0..n).map(|i| m[i * n + i]).collect(), v)
}
