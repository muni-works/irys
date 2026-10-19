//! Running the graph: one compiled C program per worker (f32), gradients
//! from `TensorGraph::backward` summed over workers, Adam on the host.

use irys_diff::ir::Program;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::data::Targets;
use crate::net::{self, Arch, Graph, LOSS_NAMES, P};

/// One worker's inputs beyond the parameters, in graph input order.
pub struct Batch {
    pub x: Vec<f64>,
    pub t: Targets,
    pub g_par: Vec<f64>,
    pub g_yaw: Vec<f64>,
    pub g_plane: Vec<f64>,
}

impl Batch {
    /// Concatenate per-image pieces.
    pub fn new(xs: &[Vec<f64>], ts: &[Targets]) -> Batch {
        let cat = |f: &dyn Fn(&Targets) -> &Vec<f64>| ts.iter().flat_map(|t| f(t).iter().copied()).collect::<Vec<f64>>();
        let b = xs.len();
        Batch {
            x: xs.iter().flatten().copied().collect(),
            t: Targets {
                t_pix: cat(&|t| &t.t_pix),
                w_pix: cat(&|t| &t.w_pix),
                t_ctr: cat(&|t| &t.t_ctr),
                w_ctr: cat(&|t| &t.w_ctr),
                t_cls: cat(&|t| &t.t_cls),
                w_cls: cat(&|t| &t.w_cls),
                t_tok: cat(&|t| &t.t_tok),
                w_tok: cat(&|t| &t.w_tok),
                t_par: cat(&|t| &t.t_par),
                w_par: cat(&|t| &t.w_par),
                t_yaw: cat(&|t| &t.t_yaw),
                w_yaw: cat(&|t| &t.w_yaw),
                t_plane: cat(&|t| &t.t_plane),
                w_plane: cat(&|t| &t.w_plane),
            },
            g_par: vec![0.0; b * P * 2 * crate::scene::NP],
            g_yaw: vec![0.0; b * P * 2],
            g_plane: vec![0.0; b * 2],
        }
    }
}

/// Saved weights.
#[derive(Serialize, Deserialize, Clone)]
pub struct Weights {
    pub arch: Arch,
    pub names: Vec<String>,
    pub shapes: Vec<Vec<usize>>,
    pub values: Vec<Vec<f32>>,
}

impl Weights {
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::write(path, serde_json::to_vec(self).unwrap())
    }
    pub fn load(path: &std::path::Path) -> std::io::Result<Weights> {
        serde_json::from_slice(&std::fs::read(path)?).map_err(std::io::Error::other)
    }
    pub fn f64s(&self) -> Vec<Vec<f64>> {
        self.values.iter().map(|v| v.iter().map(|&x| x as f64).collect()).collect()
    }
}

/// Workers, each a compiled training graph for `per` images.
pub struct Trainer {
    pub arch: Arch,
    pub per: usize,
    progs: Vec<Program>,
    n_params: usize,
    pub params: Vec<Vec<f64>>,
    names: Vec<String>,
    shapes: Vec<Vec<usize>>,
    m: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
    t: i32,
    pub native: bool,
}

/// Loss terms of one step, summed over workers.
pub type Losses = [f64; 11];

impl Trainer {
    pub fn new(arch: Arch, per: usize, workers: usize, params: Option<Vec<Vec<f64>>>, seed: u64) -> Trainer {
        let gr: Graph = net::build(arch, per, true);
        let mut outs = gr.losses.clone();
        outs.extend(&gr.grads);
        let (g, outs) = net::optimize(&gr.g, &outs);
        let progs: Vec<Program> = (0..workers).into_par_iter().map(|_| Program::compile(&g, &outs)).collect();
        let native = progs[0].is_native();
        let params = params.unwrap_or_else(|| net::init(&gr.params, seed));
        let m = params.iter().map(|p| vec![0.0; p.len()]).collect();
        let v = params.iter().map(|p| vec![0.0; p.len()]).collect();
        Trainer {
            arch,
            per,
            progs,
            n_params: gr.params.ids.len(),
            params,
            names: gr.params.names.clone(),
            shapes: gr.params.shapes.clone(),
            m,
            v,
            t: 0,
            native,
        }
    }

    pub fn workers(&self) -> usize {
        self.progs.len()
    }

    pub fn parameter_count(&self) -> usize {
        self.params.iter().map(|p| p.len()).sum()
    }

    /// Losses and summed gradients for one batch per worker.
    pub fn grads(&mut self, batches: &[Batch]) -> (Losses, Vec<Vec<f64>>) {
        assert_eq!(batches.len(), self.progs.len());
        let np = self.n_params;
        let params = &self.params;
        let res: Vec<(Vec<f64>, Vec<Vec<f64>>)> = self
            .progs
            .par_iter_mut()
            .zip(batches)
            .map(|(pr, b)| {
                for (k, p) in params.iter().enumerate() {
                    pr.write(k, p);
                }
                let t = &b.t;
                let ins: [&Vec<f64>; 18] = [
                    &b.x, &t.t_pix, &t.w_pix, &t.t_ctr, &t.w_ctr, &t.t_cls, &t.w_cls, &t.t_tok, &t.w_tok, &t.t_par, &t.w_par, &t.t_yaw, &t.w_yaw, &t.t_plane,
                    &t.w_plane, &b.g_par, &b.g_yaw, &b.g_plane,
                ];
                for (k, v) in ins.iter().enumerate() {
                    pr.write(np + k, v);
                }
                pr.run();
                let losses: Vec<f64> = (0..LOSS_NAMES.len()).map(|i| pr.read(i)[0]).collect();
                let grads = (0..np).map(|i| pr.read(LOSS_NAMES.len() + i)).collect();
                (losses, grads)
            })
            .collect();
        let mut l = [0.0; 11];
        let mut g: Vec<Vec<f64>> = self.params.iter().map(|p| vec![0.0; p.len()]).collect();
        for (ls, gs) in res {
            for (a, b) in l.iter_mut().zip(&ls) {
                *a += b;
            }
            for (a, b) in g.iter_mut().zip(&gs) {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
            }
        }
        (l, g)
    }

    /// Adam with global-norm clipping and decoupled weight decay.
    pub fn adam(&mut self, g: &[Vec<f64>], lr: f64, clip: f64, wd: f64) -> f64 {
        let norm = g.iter().flatten().map(|x| x * x).sum::<f64>().sqrt();
        let s = if norm > clip { clip / norm } else { 1.0 };
        self.t += 1;
        let (b1, b2, eps): (f64, f64, f64) = (0.9, 0.999, 1e-8);
        let bc1 = 1.0 - b1.powi(self.t);
        let bc2 = 1.0 - b2.powi(self.t);
        for (k, p) in self.params.iter_mut().enumerate() {
            let decay = !(self.names[k].ends_with(".b") || self.names[k].ends_with(".g"));
            for i in 0..p.len() {
                let gi = g[k][i] * s;
                if !gi.is_finite() {
                    continue;
                }
                self.m[k][i] = b1 * self.m[k][i] + (1.0 - b1) * gi;
                self.v[k][i] = b2 * self.v[k][i] + (1.0 - b2) * gi * gi;
                let upd = (self.m[k][i] / bc1) / ((self.v[k][i] / bc2).sqrt() + eps);
                p[i] -= lr * (upd + if decay { wd * p[i] } else { 0.0 });
            }
        }
        norm
    }

    pub fn weights(&self) -> Weights {
        Weights {
            arch: self.arch,
            names: self.names.clone(),
            shapes: self.shapes.clone(),
            values: self.params.iter().map(|p| p.iter().map(|&x| x as f32).collect()).collect(),
        }
    }
}

/// The forward graph alone, compiled, for inference.
pub struct Infer {
    prog: Program,
    pub per: usize,
    n_params: usize,
}

/// Head outputs for one image.
#[derive(Debug, Clone)]
pub struct Out {
    pub pix: Vec<f64>,
    pub ctr: Vec<f64>,
    pub cls: Vec<f64>,
    pub tok: Vec<f64>,
    pub par: Vec<f64>,
    pub yaw: Vec<f64>,
    pub plane: [f64; 2],
}

impl Infer {
    pub fn new(w: &Weights, per: usize) -> Infer {
        let gr = net::build(w.arch, per, false);
        let outs = gr.heads.all().to_vec();
        let (g, outs) = net::optimize(&gr.g, &outs);
        let mut prog = Program::compile(&g, &outs);
        for (k, p) in w.f64s().iter().enumerate() {
            prog.write(k, p);
        }
        Infer { prog, per, n_params: gr.params.ids.len() }
    }

    pub fn set_weights(&mut self, params: &[Vec<f64>]) {
        for (k, p) in params.iter().enumerate() {
            self.prog.write(k, p);
        }
    }

    /// Up to `per` patchified images.
    pub fn run(&mut self, xs: &[Vec<f64>]) -> Vec<Out> {
        let n = xs.len();
        assert!(n <= self.per);
        let mut x: Vec<f64> = xs.iter().flatten().copied().collect();
        x.resize(self.per * P * net::DIN, 0.0);
        self.prog.write(self.n_params, &x);
        self.prog.run();
        let r: Vec<Vec<f64>> = (0..7).map(|i| self.prog.read(i)).collect();
        let per = self.per;
        let split = |v: &Vec<f64>, i: usize| {
            let k = v.len() / per;
            v[i * k..(i + 1) * k].to_vec()
        };
        (0..n)
            .map(|i| Out {
                pix: split(&r[0], i),
                ctr: split(&r[1], i),
                cls: split(&r[2], i),
                tok: split(&r[3], i),
                par: split(&r[4], i),
                yaw: split(&r[5], i),
                plane: [r[6][i * 2], r[6][i * 2 + 1]],
            })
            .collect()
    }
}
