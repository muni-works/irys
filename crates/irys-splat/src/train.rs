//! The optimiser and its schedule, gsplat's `simple_trainer` defaults:
//! Adam per parameter group (means 1.6e-4 × scene scale, decayed ×0.01 over
//! the run; log scales 5e-3; quaternions 1e-3; opacity logits 5e-2; SH DC
//! 2.5e-3, SH rest 2.5e-3 / 20), one view per step, SH degree raised every
//! `sh_every` steps, and `DefaultStrategy`'s refinement: every 100 steps in
//! `[500, 15000)` clone (small) or split (large) the splats whose mean
//! screen-space gradient exceeds 2e-4 (NDC units), prune opacity < 0.005
//! (and, after the first reset, world size > 0.1 × scene scale), reset
//! opacity to ≤ 0.01 every 3000 steps.

use irys_diff::look::pinhole::Pinhole;
use irys_gpu::GpuContext;

use crate::data::View;
use crate::gpu::{Device, LossCfg, Lr};
use crate::loss::{self, Weights};
use crate::model::{Gaussians, logit, sigmoid};
use crate::rng::Rng;
use crate::sh::COEFS;
use crate::tiles;

#[derive(Debug, Clone)]
pub struct Config {
    pub iters: usize,
    pub sh_degree: u32,
    pub sh_every: usize,
    pub lr_means: f64,
    pub lr_means_final: f64,
    pub lr_scales: f64,
    pub lr_quats: f64,
    pub lr_opacity: f64,
    pub lr_sh0: f64,
    pub lr_shn: f64,
    pub weights: Weights,
    pub refine_start: usize,
    pub refine_stop: usize,
    pub refine_every: usize,
    pub reset_every: usize,
    pub grow_grad2d: f64,
    pub grow_scale3d: f64,
    pub prune_opa: f64,
    pub prune_scale3d: f64,
    pub max_splats: usize,
    pub bg: [f64; 3],
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            iters: 30_000,
            sh_degree: 3,
            sh_every: 1000,
            lr_means: 1.6e-4,
            lr_means_final: 0.01,
            lr_scales: 5e-3,
            lr_quats: 1e-3,
            lr_opacity: 5e-2,
            lr_sh0: 2.5e-3,
            lr_shn: 2.5e-3 / 20.0,
            weights: Weights::default(),
            refine_start: 500,
            refine_stop: 15_000,
            refine_every: 100,
            reset_every: 3000,
            grow_grad2d: 2e-4,
            grow_scale3d: 0.01,
            prune_opa: 0.005,
            prune_scale3d: 0.1,
            max_splats: 1_500_000,
            bg: [0.0; 3],
            seed: 0,
        }
    }
}

impl Config {
    /// Scale the refinement schedule to a shorter (or longer) run, keeping
    /// gsplat's proportions of a 30k-step run.
    pub fn scaled(iters: usize) -> Self {
        let d = Self::default();
        let f = iters as f64 / 30_000.0;
        let s = |x: usize| ((x as f64 * f).round() as usize).max(1);
        Self {
            iters,
            sh_every: s(d.sh_every),
            refine_start: s(d.refine_start),
            refine_stop: s(d.refine_stop),
            reset_every: s(d.reset_every),
            refine_every: s(d.refine_every).max(25),
            ..d
        }
    }
}

/// Scene scale (gsplat): 1.1 × the largest camera distance from their centroid.
pub fn scene_scale(cams: &[Pinhole]) -> f64 {
    let n = cams.len() as f64;
    let c: [f64; 3] = std::array::from_fn(|a| cams.iter().map(|k| k.pos[a]).sum::<f64>() / n);
    let r = cams.iter().map(|k| (0..3).map(|a| (k.pos[a] - c[a]).powi(2)).sum::<f64>().sqrt()).fold(0.0, f64::max);
    1.1 * r.max(0.1)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StepStats {
    pub loss: f64,
    pub l1: f64,
    pub ssim: f64,
    pub splats: usize,
    pub isects: usize,
    /// device project + key readback, host binning, device step, refinement (ms)
    pub ms: [f64; 5],
}

pub struct Trainer {
    /// The host copy of the cloud: current after [`Self::sync`].
    pub g: Gaussians,
    pub cfg: Config,
    pub scale: f64,
    pub it: usize,
    m: Gaussians,
    v: Gaussians,
    t: i32,
    accum: Vec<f64>,
    count: Vec<u32>,
    rng: Rng,
    pub dev: Device,
    synced: bool,
    n_dev: usize,
}

impl Trainer {
    pub fn new(ctx: &'static GpuContext, g: Gaussians, cfg: Config, scale: f64) -> Self {
        let n = g.len();
        let rng = Rng::new(cfg.seed ^ 0x5eed);
        let mut dev = Device::new(ctx);
        dev.set_state(&g, None, None);
        Self {
            m: Gaussians::zeros(n),
            v: Gaussians::zeros(n),
            accum: vec![0.0; n],
            count: vec![0; n],
            g,
            cfg,
            scale,
            it: 0,
            t: 0,
            rng,
            dev,
            synced: true,
            n_dev: n,
        }
    }

    pub fn degree(&self) -> u32 {
        self.cfg.sh_degree.min((self.it / self.cfg.sh_every) as u32)
    }

    /// Splats on the device.
    pub fn len(&self) -> usize {
        self.n_dev
    }

    /// Pull parameters, moments and densification stats off the device.
    pub fn sync(&mut self) {
        if self.synced {
            return;
        }
        let (g, m, v, st) = self.dev.state();
        self.g = g;
        self.m = m;
        self.v = v;
        self.accum = st.iter().map(|s| s[0] as f64).collect();
        self.count = st.iter().map(|s| s[1] as u32).collect();
        self.synced = true;
    }

    fn push(&mut self) {
        self.dev.set_state(&self.g, Some(&self.m), Some(&self.v));
        self.n_dev = self.g.len();
    }

    /// Render the resident cloud at `cam` (rgb + depth per pixel).
    pub fn render(&mut self, cam: &Pinhole, degree: u32) -> Vec<[f64; 4]> {
        let k = self.dev.project(cam, degree);
        let bins = tiles::bin_keys(&k, cam.width, cam.height);
        let c = self.dev.composite(&bins, self.cfg.bg.map(|v| v as f32));
        c.iter().map(|v| v.map(|x| x as f64)).collect()
    }

    /// One optimisation step on `view`. With `want_loss` the frame is read
    /// back and the loss evaluated on the host (for logging).
    pub fn step(&mut self, view: &View, want_loss: bool) -> StepStats {
        let mut st = StepStats::default();
        let deg = self.degree();
        let cam = &view.cam;
        let (w, h) = (cam.width, cam.height);
        let k = self.dev.project(cam, deg);
        st.ms[0] = self.dev.last_ms[0];
        let t1 = std::time::Instant::now();
        let bins = tiles::bin_keys(&k, w, h);
        st.ms[1] = t1.elapsed().as_secs_f64() * 1e3;
        let gt: Vec<[f32; 4]> = (0..w * h)
            .map(|p| {
                let c = view.rgb[p];
                [c[0], c[1], c[2], view.depth.as_ref().map_or(0.0, |d| d[p])]
            })
            .collect();
        let depth_per_px = match (&view.depth, self.cfg.weights.depth > 0.0) {
            (Some(d), true) => {
                let n = d.iter().filter(|&&z| z > 0.0).count();
                if n > 0 { self.cfg.weights.depth / n as f64 } else { 0.0 }
            }
            _ => 0.0,
        };
        self.t += 1;
        let c = &self.cfg;
        let frac = self.it as f64 / c.iters.max(1) as f64;
        let lr = Lr {
            means: c.lr_means * self.scale * c.lr_means_final.powf(frac),
            scales: c.lr_scales,
            quats: c.lr_quats,
            opacity: c.lr_opacity,
            sh0: c.lr_sh0,
            shn: c.lr_shn,
            bc1: 1.0 - 0.9f64.powi(self.t),
            bc2: 1.0 - 0.999f64.powi(self.t),
        };
        let img = self.dev.step(&bins, &gt, c.bg.map(|v| v as f32), LossCfg { ssim: c.weights.ssim, depth_per_px }, lr, want_loss);
        st.ms[2] = self.dev.last_ms[1];
        self.synced = false;
        if let Some(img) = img {
            let pred: Vec<[f64; 4]> = img.iter().map(|v| v.map(|x| x as f64)).collect();
            let ev = loss::eval(&pred, &view.rgb, view.depth.as_deref(), w, h, self.cfg.weights);
            st.loss = ev.loss;
            st.l1 = ev.l1;
            st.ssim = ev.ssim;
        }
        st.isects = bins.ids.len();
        self.it += 1;
        let t3 = std::time::Instant::now();
        self.refine();
        st.ms[3] = t3.elapsed().as_secs_f64() * 1e3;
        st.splats = self.n_dev;
        st
    }

    fn refine(&mut self) {
        let it = self.it;
        let c = self.cfg.clone();
        if it > c.refine_stop {
            return;
        }
        let grow = it > c.refine_start && it % c.refine_every == 0;
        let reset = it % c.reset_every == 0 && it > 0;
        if !grow && !reset {
            return;
        }
        self.sync();
        if grow {
            self.grow();
            self.prune(it > c.reset_every);
        }
        if reset {
            let cap = logit(2.0 * c.prune_opa);
            for (o, (m, v)) in self.g.opacity.iter_mut().zip(self.m.opacity.iter_mut().zip(self.v.opacity.iter_mut())) {
                *o = o.min(cap);
                *m = 0.0;
                *v = 0.0;
            }
        }
        self.push();
        self.accum = vec![0.0; self.g.len()];
        self.count = vec![0; self.g.len()];
    }

    fn grow(&mut self) {
        let n = self.g.len();
        let c = &self.cfg;
        let mut dup = Vec::new();
        let mut split = Vec::new();
        for i in 0..n {
            if self.count[i] == 0 {
                continue;
            }
            let avg = self.accum[i] / self.count[i] as f64;
            if avg <= c.grow_grad2d {
                continue;
            }
            let smax = self.g.log_scales[i].iter().cloned().fold(f64::MIN, f64::max).exp();
            if smax <= c.grow_scale3d * self.scale { dup.push(i) } else { split.push(i) }
        }
        let room = c.max_splats.saturating_sub(n);
        dup.truncate(room);
        split.truncate(room.saturating_sub(dup.len()));
        let mut keep = vec![true; n];
        for &i in &split {
            keep[i] = false;
        }
        let mut g = Gaussians::default();
        let mut m = Gaussians::default();
        let mut v = Gaussians::default();
        for i in (0..n).filter(|&i| keep[i]) {
            g.push_from(&self.g, i);
            m.push_from(&self.m, i);
            v.push_from(&self.v, i);
        }
        let z = Gaussians::zeros(1);
        for &i in &dup {
            g.push_from(&self.g, i);
            m.push_from(&z, 0);
            v.push_from(&z, 0);
        }
        for &i in &split {
            let s = self.g.log_scales[i].map(f64::exp);
            let q = self.g.quats[i];
            let ql = q.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
            let [w, x, y, zq] = q.map(|x| x / ql);
            let r = [
                [1.0 - 2.0 * (y * y + zq * zq), 2.0 * (x * y - w * zq), 2.0 * (x * zq + w * y)],
                [2.0 * (x * y + w * zq), 1.0 - 2.0 * (x * x + zq * zq), 2.0 * (y * zq - w * x)],
                [2.0 * (x * zq - w * y), 2.0 * (y * zq + w * x), 1.0 - 2.0 * (x * x + y * y)],
            ];
            for _ in 0..2 {
                let xi: [f64; 3] = std::array::from_fn(|k| s[k] * self.rng.normal());
                let k = g.len();
                g.push_from(&self.g, i);
                for a in 0..3 {
                    g.means[k][a] += (0..3).map(|b| r[a][b] * xi[b]).sum::<f64>();
                }
                g.log_scales[k] = s.map(|v| (v / 1.6).ln());
                m.push_from(&z, 0);
                v.push_from(&z, 0);
            }
        }
        self.g = g;
        self.m = m;
        self.v = v;
        self.accum = vec![0.0; self.g.len()];
        self.count = vec![0; self.g.len()];
    }

    fn prune(&mut self, big: bool) {
        let c = &self.cfg;
        let lim = c.prune_scale3d * self.scale;
        let keep: Vec<bool> = (0..self.g.len())
            .map(|i| {
                let o = sigmoid(self.g.opacity[i]);
                let smax = self.g.log_scales[i].iter().cloned().fold(f64::MIN, f64::max).exp();
                o >= c.prune_opa && !(big && smax > lim)
            })
            .collect();
        if keep.iter().all(|&k| k) {
            return;
        }
        self.g.retain(&keep);
        self.m.retain(&keep);
        self.v.retain(&keep);
        let mut k = 0;
        self.accum.retain(|_| {
            k += 1;
            keep[k - 1]
        });
        let mut k = 0;
        self.count.retain(|_| {
            k += 1;
            keep[k - 1]
        });
    }
}

/// Initial clouds.
pub mod init {
    use super::*;

    /// `n` splats uniform in the box `[lo, hi]`, grey-ish random colour.
    pub fn random_box(n: usize, lo: [f64; 3], hi: [f64; 3], seed: u64) -> Gaussians {
        let mut r = Rng::new(seed);
        let pts: Vec<[f32; 3]> = (0..n).map(|_| std::array::from_fn(|a| r.range(lo[a], hi[a]) as f32)).collect();
        let col: Vec<[f32; 3]> = (0..n).map(|_| std::array::from_fn(|_| r.uniform() as f32)).collect();
        Gaussians::from_points(&pts, &col)
    }
}
