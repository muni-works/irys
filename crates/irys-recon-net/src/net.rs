//! The proposer: a small MLP-Mixer (a ViT without attention) over 8 × 8
//! patches, with dense heads, written once as a tang tensor-IR graph.
//!
//! Every op is a comprehension (`TensorGraph::comp`) read through pick
//! views; the gradient is `TensorGraph::backward`'s and the graph runs as C
//! (f32) through [`irys_diff::ir::Program`]. There is no hand-written
//! backward pass anywhere in this crate.
//!
//! Heads, per patch `p` of image `b`:
//!
//! - `pix [B,P,4,64]`: per pixel of the patch, foreground logit, offset to
//!   the instance centre (x, y, in patches), log metric depth;
//! - `ctr [B,P]`: centre heat logit (CenterNet-style instances);
//! - `cls [B,P,C]`: object class logits;
//! - `tok [B,P,T,V]`: program-structure tokens over the scene grammar;
//! - `par [B,P,2,NP]`: continuous parameters, mean and log σ (normalised log mm);
//! - `yaw [B,P,2]`: `(cos 2ψ, sin 2ψ)` on the support plane;
//! - `plane [B,2]`: camera height (log m) and pitch (rad): the metric scale.

use tang_compile::tensor::{Arg, Monoid, TensorGraph, TensorId, View};
use tang_expr::ExprId;

use crate::scene::{CLASSES, NP, NTOK, VOCAB};

pub use crate::data::{DIN, GH, GW, H, P, PIX, PS, W};

/// Widths.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Arch {
    pub d: usize,
    pub tok_hidden: usize,
    pub ch_hidden: usize,
    pub blocks: usize,
}

impl Default for Arch {
    fn default() -> Self {
        Self { d: 96, tok_hidden: 192, ch_hidden: 192, blocks: 4 }
    }
}

fn lit(v: f64) -> ExprId {
    <ExprId as tang::Scalar>::from_f64(v)
}

fn pk(x: TensorId, space: usize, picks: &[Option<usize>]) -> Arg {
    Arg { src: x, view: View::pick(space, picks) }
}

fn ids(n: usize) -> Vec<Option<usize>> {
    (0..n).map(Some).collect()
}

fn gelu(a: ExprId) -> ExprId {
    let c = lit(0.797_884_560_8);
    lit(0.5) * a * (lit(1.0) + <ExprId as tang::Scalar>::tanh(c * (a + lit(0.044715) * a * a * a)))
}

fn softplus(z: ExprId) -> ExprId {
    use tang::Scalar;
    z.max(lit(0.0)) + (lit(1.0) + (lit(0.0) - z.abs()).exp()).ln()
}

/// Logits bounded to ±8 so `exp` cannot overflow in a log-sum-exp.
fn bound(z: ExprId) -> ExprId {
    lit(8.0) * <ExprId as tang::Scalar>::tanh(z * lit(0.125))
}

fn dims(g: &TensorGraph, x: TensorId) -> Vec<usize> {
    g.shape(x).iter().map(|d| d.n()).collect()
}

/// The parameter tensors, in input order (they are the graph's first inputs).
#[derive(Debug, Clone)]
pub struct Params {
    pub ids: Vec<TensorId>,
    pub shapes: Vec<Vec<usize>>,
    pub names: Vec<String>,
}

impl Params {
    fn add(&mut self, g: &mut TensorGraph, name: &str, shape: &[usize]) -> TensorId {
        let t = g.input(shape);
        self.ids.push(t);
        self.shapes.push(shape.to_vec());
        self.names.push(name.into());
        t
    }
    pub fn count(&self) -> usize {
        self.shapes.iter().map(|s| s.iter().product::<usize>()).sum()
    }
}

/// `y[b,p,o…] = Σ_i x[b,p,i] W[i,o…] + bias[o…]`.
fn dense(g: &mut TensorGraph, x: TensorId, w: TensorId, bias: TensorId, act: bool) -> TensorId {
    let xd = dims(g, x);
    let wd = dims(g, w);
    let lead = xd.len() - 1;
    let outs = &wd[1..];
    let mut shape = xd[..lead].to_vec();
    shape.extend_from_slice(outs);
    let r = shape.len();
    let mut xp = ids(lead);
    xp.push(Some(r));
    let mut wp = vec![Some(r)];
    wp.extend((lead..r).map(Some));
    let z = g
        .comp(&shape, &[xd[lead]], vec![pk(x, r + 1, &xp), pk(w, r + 1, &wp)], Some(Monoid::sum()), |v| {
            vec![v[0] * v[1]]
        })
        .expect("dense");
    let bp: Vec<Option<usize>> = (lead..r).map(Some).collect();
    g.comp(&shape, &[] as &[usize], vec![pk(z, r, &ids(r)), pk(bias, r, &bp)], None, |v| {
        let a = v[0] + v[1];
        vec![if act { gelu(a) } else { a }]
    })
    .expect("bias")
}

/// Token mixing: `y[b,q,d] = Σ_p W[q,p] x[b,p,d] + bias[q]`.
fn tmix(g: &mut TensorGraph, x: TensorId, w: TensorId, bias: TensorId, act: bool) -> TensorId {
    let [b, p, d] = dims(g, x)[..] else { panic!() };
    let q = dims(g, w)[0];
    let z = g
        .comp(&[b, q, d], &[p], vec![pk(w, 4, &[Some(1), Some(3)]), pk(x, 4, &[Some(0), Some(3), Some(2)])], Some(Monoid::sum()), |v| {
            vec![v[0] * v[1]]
        })
        .expect("tmix");
    g.comp(&[b, q, d], &[] as &[usize], vec![pk(z, 3, &ids(3)), pk(bias, 3, &[Some(1)])], None, |v| {
        let a = v[0] + v[1];
        vec![if act { gelu(a) } else { a }]
    })
    .expect("tmix bias")
}

/// Layer norm over the last axis of `[b,p,d]`.
fn norm(g: &mut TensorGraph, x: TensorId, gain: TensorId, beta: TensorId) -> TensorId {
    let [b, p, d] = dims(g, x)[..] else { panic!() };
    let inv = 1.0 / d as f64;
    let mean = g
        .comp(&[b, p], &[d], vec![pk(x, 3, &ids(3))], Some(Monoid::sum()), |v| vec![v[0] * lit(inv)])
        .expect("mean");
    let var = g
        .comp(&[b, p], &[d], vec![pk(x, 3, &ids(3)), pk(mean, 3, &[Some(0), Some(1)])], Some(Monoid::sum()), |v| {
            let c = v[0] - v[1];
            vec![c * c * lit(inv)]
        })
        .expect("var");
    g.comp(
        &[b, p, d],
        &[] as &[usize],
        vec![
            pk(x, 3, &ids(3)),
            pk(mean, 3, &[Some(0), Some(1)]),
            pk(var, 3, &[Some(0), Some(1)]),
            pk(gain, 3, &[Some(2)]),
            pk(beta, 3, &[Some(2)]),
        ],
        None,
        |v| vec![(v[0] - v[1]) / <ExprId as tang::Scalar>::sqrt(v[2] + lit(1e-5)) * v[3] + v[4]],
    )
    .expect("norm")
}

fn add(g: &mut TensorGraph, a: TensorId, b: TensorId) -> TensorId {
    g.map(&[a, b], |v| v[0] + v[1]).expect("add")
}

/// The heads' tensors.
#[derive(Debug, Clone, Copy)]
pub struct Heads {
    pub pix: TensorId,
    pub ctr: TensorId,
    pub cls: TensorId,
    pub tok: TensorId,
    pub par: TensorId,
    pub yaw: TensorId,
    pub plane: TensorId,
}

impl Heads {
    pub fn all(&self) -> [TensorId; 7] {
        [self.pix, self.ctr, self.cls, self.tok, self.par, self.yaw, self.plane]
    }
}

/// Target and weight inputs of the loss (host-normalised weights: each
/// loss is `Σ w ⊙ ℓ`).
#[derive(Debug, Clone, Copy)]
pub struct LossInputs {
    pub t_pix: TensorId,
    pub w_pix: TensorId,
    pub t_ctr: TensorId,
    pub w_ctr: TensorId,
    pub t_cls: TensorId,
    pub w_cls: TensorId,
    pub t_tok: TensorId,
    pub w_tok: TensorId,
    pub t_par: TensorId,
    pub w_par: TensorId,
    pub t_yaw: TensorId,
    pub w_yaw: TensorId,
    pub t_plane: TensorId,
    pub w_plane: TensorId,
    /// Upstream gradients for the continuous heads (renderer as teacher):
    /// the loss gains `Σ G ⊙ head`, so `backward` carries `G` into the weights.
    pub g_par: TensorId,
    pub g_yaw: TensorId,
    pub g_plane: TensorId,
}

/// The whole graph: parameters, image input, heads, losses and gradients.
pub struct Graph {
    pub g: TensorGraph,
    pub params: Params,
    pub x: TensorId,
    pub heads: Heads,
    pub li: Option<LossInputs>,
    /// `[total, fg, off, depth, ctr, cls, tok, par, yaw, plane, teacher]`.
    pub losses: Vec<TensorId>,
    pub grads: Vec<TensorId>,
    pub batch: usize,
    pub arch: Arch,
}

pub const LOSS_NAMES: [&str; 11] = ["total", "fg", "off", "depth", "ctr", "cls", "tok", "par", "yaw", "plane", "teacher"];

/// Build the graph for a batch of `batch` images.
pub fn build(arch: Arch, batch: usize, with_loss: bool) -> Graph {
    let mut g = TensorGraph::new();
    let mut ps = Params { ids: Vec::new(), shapes: Vec::new(), names: Vec::new() };
    let d = arch.d;
    let we = ps.add(&mut g, "embed.w", &[DIN, d]);
    let be = ps.add(&mut g, "embed.b", &[d]);
    let pos = ps.add(&mut g, "pos", &[P, d]);
    let mut blocks = Vec::new();
    for k in 0..arch.blocks {
        let n = |s: &str| format!("block{k}.{s}");
        blocks.push([
            ps.add(&mut g, &n("ln1.g"), &[d]),
            ps.add(&mut g, &n("ln1.b"), &[d]),
            ps.add(&mut g, &n("tok1.w"), &[arch.tok_hidden, P]),
            ps.add(&mut g, &n("tok1.b"), &[arch.tok_hidden]),
            ps.add(&mut g, &n("tok2.w"), &[P, arch.tok_hidden]),
            ps.add(&mut g, &n("tok2.b"), &[P]),
            ps.add(&mut g, &n("ln2.g"), &[d]),
            ps.add(&mut g, &n("ln2.b"), &[d]),
            ps.add(&mut g, &n("ch1.w"), &[d, arch.ch_hidden]),
            ps.add(&mut g, &n("ch1.b"), &[arch.ch_hidden]),
            ps.add(&mut g, &n("ch2.w"), &[arch.ch_hidden, d]),
            ps.add(&mut g, &n("ch2.b"), &[d]),
        ]);
    }
    let lnf_g = ps.add(&mut g, "lnf.g", &[d]);
    let lnf_b = ps.add(&mut g, "lnf.b", &[d]);
    let hp = [
        ps.add(&mut g, "pix.w", &[d, 4, PIX]),
        ps.add(&mut g, "pix.b", &[4, PIX]),
        ps.add(&mut g, "ctr.w", &[d, 1]),
        ps.add(&mut g, "ctr.b", &[1]),
        ps.add(&mut g, "cls.w", &[d, CLASSES]),
        ps.add(&mut g, "cls.b", &[CLASSES]),
        ps.add(&mut g, "tok.w", &[d, NTOK, VOCAB]),
        ps.add(&mut g, "tok.b", &[NTOK, VOCAB]),
        ps.add(&mut g, "par.w", &[d, 2, NP]),
        ps.add(&mut g, "par.b", &[2, NP]),
        ps.add(&mut g, "yaw.w", &[d, 2]),
        ps.add(&mut g, "yaw.b", &[2]),
        ps.add(&mut g, "plane.w", &[d, 2]),
        ps.add(&mut g, "plane.b", &[2]),
    ];

    let b = batch;
    let x = g.input(&[b, P, DIN]);
    let e = dense(&mut g, x, we, be, false);
    let mut h = g
        .comp(&[b, P, d], &[] as &[usize], vec![pk(e, 3, &ids(3)), pk(pos, 3, &[Some(1), Some(2)])], None, |v| {
            vec![v[0] + v[1]]
        })
        .unwrap();
    for bl in &blocks {
        let n1 = norm(&mut g, h, bl[0], bl[1]);
        let t1 = tmix(&mut g, n1, bl[2], bl[3], true);
        let t2 = tmix(&mut g, t1, bl[4], bl[5], false);
        h = add(&mut g, h, t2);
        let n2 = norm(&mut g, h, bl[6], bl[7]);
        let c1 = dense(&mut g, n2, bl[8], bl[9], true);
        let c2 = dense(&mut g, c1, bl[10], bl[11], false);
        h = add(&mut g, h, c2);
    }
    let f = norm(&mut g, h, lnf_g, lnf_b);
    let pix = dense(&mut g, f, hp[0], hp[1], false);
    let ctr1 = dense(&mut g, f, hp[2], hp[3], false);
    let ctr = g
        .comp(&[b, P], &[1usize], vec![pk(ctr1, 3, &[Some(0), Some(1), Some(2)])], Some(Monoid::sum()), |v| vec![v[0]])
        .unwrap();
    let cls = dense(&mut g, f, hp[4], hp[5], false);
    let tok = dense(&mut g, f, hp[6], hp[7], false);
    let par = dense(&mut g, f, hp[8], hp[9], false);
    let yaw = dense(&mut g, f, hp[10], hp[11], false);
    let pooled = g
        .comp(&[b, d], &[P], vec![pk(f, 3, &[Some(0), Some(2), Some(1)])], Some(Monoid::sum()), |v| vec![v[0] * lit(1.0 / P as f64)])
        .unwrap();
    let plane = {
        let z = g
            .comp(&[b, 2], &[d], vec![pk(pooled, 3, &[Some(0), Some(2)]), pk(hp[12], 3, &[Some(2), Some(1)])], Some(Monoid::sum()), |v| {
                vec![v[0] * v[1]]
            })
            .unwrap();
        g.comp(&[b, 2], &[] as &[usize], vec![pk(z, 2, &ids(2)), pk(hp[13], 2, &[Some(1)])], None, |v| vec![v[0] + v[1]]).unwrap()
    };
    let heads = Heads { pix, ctr, cls, tok, par, yaw, plane };

    let li = with_loss.then(|| LossInputs {
        t_pix: g.input(&[b, P, 4, PIX]),
        w_pix: g.input(&[b, P, 4, PIX]),
        t_ctr: g.input(&[b, P]),
        w_ctr: g.input(&[b, P]),
        t_cls: g.input(&[b, P, CLASSES]),
        w_cls: g.input(&[b, P]),
        t_tok: g.input(&[b, P, NTOK, VOCAB]),
        w_tok: g.input(&[b, P, NTOK]),
        t_par: g.input(&[b, P, NP]),
        w_par: g.input(&[b, P, NP]),
        t_yaw: g.input(&[b, P, 2]),
        w_yaw: g.input(&[b, P, 2]),
        t_plane: g.input(&[b, 2]),
        w_plane: g.input(&[b, 2]),
        g_par: g.input(&[b, P, 2, NP]),
        g_yaw: g.input(&[b, P, 2]),
        g_plane: g.input(&[b, 2]),
    });
    let mut losses = Vec::new();
    let mut grads = Vec::new();
    if let Some(li) = li {
        let sum_all = |g: &mut TensorGraph, args: Vec<Arg>, red: &[usize], f: &dyn Fn(&[ExprId]) -> ExprId| -> TensorId {
            g.comp(&[1usize], red, args, Some(Monoid::sum()), |v| vec![f(v)]).unwrap()
        };
        // pixel channel k of pix/t_pix/w_pix: space [1, b, P, PIX]
        let pixk = |x: TensorId, k: i64| {
            let mut v = View::pick(4, &[Some(1), Some(2), None, Some(3)]);
            v.forms[2].offset = k;
            Arg { src: x, view: v }
        };
        let red4 = [b, P, PIX];
        let l_fg = sum_all(&mut g, vec![pixk(pix, 0), pixk(li.t_pix, 0), pixk(li.w_pix, 0)], &red4, &|v| {
            v[2] * (softplus(v[0]) - v[1] * v[0])
        });
        let off = |g: &mut TensorGraph, k: i64| {
            sum_all(g, vec![pixk(pix, k), pixk(li.t_pix, k), pixk(li.w_pix, k)], &red4, &|v| {
                v[2] * <ExprId as tang::Scalar>::abs(v[0] - v[1])
            })
        };
        let l_ox = off(&mut g, 1);
        let l_oy = off(&mut g, 2);
        let l_off = add(&mut g, l_ox, l_oy);
        let l_dep = off(&mut g, 3);
        let s3 = |x: TensorId| pk(x, 4, &[Some(1), Some(2), Some(3)]);
        let s2 = |x: TensorId| pk(x, 3, &[Some(1), Some(2)]);
        let l_ctr = sum_all(&mut g, vec![s2(ctr), s2(li.t_ctr), s2(li.w_ctr)], &[b, P], &|v| v[2] * (softplus(v[0]) - v[1] * v[0]));
        // Cross-entropy with a log-sum-exp over the last axis.
        let lse_cls = g
            .comp(&[b, P], &[CLASSES], vec![pk(cls, 3, &ids(3))], Some(Monoid::sum()), |v| {
                vec![<ExprId as tang::Scalar>::exp(bound(v[0]))]
            })
            .unwrap();
        let l_cls = {
            let pick_t = g
                .comp(&[b, P], &[CLASSES], vec![pk(cls, 3, &ids(3)), pk(li.t_cls, 3, &ids(3))], Some(Monoid::sum()), |v| vec![bound(v[0]) * v[1]])
                .unwrap();
            sum_all(&mut g, vec![s2(lse_cls), s2(pick_t), s2(li.w_cls)], &[b, P], &|v| {
                v[2] * (<ExprId as tang::Scalar>::ln(v[0]) - v[1])
            })
        };
        let l_tok = {
            let lse = g
                .comp(&[b, P, NTOK], &[VOCAB], vec![pk(tok, 4, &ids(4))], Some(Monoid::sum()), |v| {
                    vec![<ExprId as tang::Scalar>::exp(bound(v[0]))]
                })
                .unwrap();
            let pick_t = g
                .comp(&[b, P, NTOK], &[VOCAB], vec![pk(tok, 4, &ids(4)), pk(li.t_tok, 4, &ids(4))], Some(Monoid::sum()), |v| {
                    vec![bound(v[0]) * v[1]]
                })
                .unwrap();
            sum_all(&mut g, vec![s3(lse), s3(pick_t), s3(li.w_tok)], &[b, P, NTOK], &|v| {
                v[2] * (<ExprId as tang::Scalar>::ln(v[0]) - v[1])
            })
        };
        // Gaussian NLL: log σ + ½ (μ − y)² / σ², log σ bounded to ±4.
        let park = |k: i64| {
            let mut v = View::pick(4, &[Some(1), Some(2), None, Some(3)]);
            v.forms[2].offset = k;
            Arg { src: par, view: v }
        };
        let l_par = sum_all(&mut g, vec![park(0), park(1), s3(li.t_par), s3(li.w_par)], &[b, P, NP], &|v| {
            let ls = lit(4.0) * <ExprId as tang::Scalar>::tanh(v[1] * lit(0.25));
            let r = v[0] - v[2];
            v[3] * (ls + lit(0.5) * r * r * <ExprId as tang::Scalar>::exp(lit(-2.0) * ls))
        });
        let l_yaw = sum_all(&mut g, vec![s3(yaw), s3(li.t_yaw), s3(li.w_yaw)], &[b, P, 2], &|v| {
            let r = v[0] - v[1];
            v[2] * r * r
        });
        let l_plane = sum_all(&mut g, vec![s2(plane), s2(li.t_plane), s2(li.w_plane)], &[b, 2], &|v| {
            let r = v[0] - v[1];
            v[2] * r * r
        });
        let t_par = sum_all(&mut g, vec![pk(par, 5, &[Some(1), Some(2), Some(3), Some(4)]), pk(li.g_par, 5, &[Some(1), Some(2), Some(3), Some(4)])], &[b, P, 2, NP], &|v| {
            v[0] * v[1]
        });
        let t_yaw = sum_all(&mut g, vec![s3(yaw), s3(li.g_yaw)], &[b, P, 2], &|v| v[0] * v[1]);
        let t_plane = sum_all(&mut g, vec![s2(plane), s2(li.g_plane)], &[b, 2], &|v| v[0] * v[1]);
        let teach = add(&mut g, t_par, t_yaw);
        let teach = add(&mut g, teach, t_plane);
        let parts = [l_fg, l_off, l_dep, l_ctr, l_cls, l_tok, l_par, l_yaw, l_plane, teach];
        let mut total = parts[0];
        for &p in &parts[1..] {
            total = add(&mut g, total, p);
        }
        losses.push(total);
        losses.extend(parts);
        grads = g.backward(total, &ps.ids).expect("backward");
    }
    Graph { g, params: ps, x, heads, li, losses, grads, batch, arch }
}

/// The rewrites kosm runs before lowering (none changes the answer).
pub fn optimize(g: &TensorGraph, outputs: &[TensorId]) -> (TensorGraph, Vec<TensorId>) {
    let passes = std::env::var("NEURAL_PASSES").unwrap_or_else(|_| "subst,cse,hoist,siblings".into());
    let mut cur = (g.clone(), outputs.to_vec());
    for p in passes.split(',') {
        cur = match p {
            "subst" => cur.0.substitute_maps(&cur.1),
            "cse" => cur.0.eliminate_common(&cur.1),
            "hoist" => cur.0.hoist_invariant_factors(&cur.1),
            "siblings" => cur.0.merge_siblings(&cur.1),
            "mergered" => cur.0.merge_reduce_dims(&cur.1),
            "splitred" => cur.0.split_reduce_dims(&cur.1),
            _ => cur,
        };
    }
    cur.0.eliminate_dead(&cur.1)
}

/// A deterministic initialisation: LN gains 1, biases 0, weights
/// `N(0, 1/fan_in)` (heads scaled down).
pub fn init(params: &Params, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = irys_recon::jac::Mix(seed ^ 0x5eed);
    let mut gauss = move || {
        let (u1, u2) = (rng.uniform().max(1e-12), rng.uniform());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    };
    params
        .names
        .iter()
        .zip(&params.shapes)
        .map(|(n, s)| {
            let len: usize = s.iter().product();
            if n.ends_with(".g") {
                vec![1.0; len]
            } else if n.ends_with(".b") {
                vec![0.0; len]
            } else {
                let fan_in = s[if n.starts_with("block") && n.contains("tok") { 1 } else { 0 }] as f64;
                let fan_in = if n == "pos" { 50.0 } else { fan_in };
                let head = ["pix", "ctr", "cls", "tok", "par", "yaw", "plane"].iter().any(|h| n.starts_with(h));
                let k = if head { 0.3 } else { 1.0 };
                (0..len).map(|_| gauss() * k / fan_in.sqrt()).collect()
            }
        })
        .collect()
}
