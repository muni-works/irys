//! A rough conductor, written once as a tang IR graph.
//!
//! The graph is batched over a wavefront: every input is one column of `n`
//! path vertices, and the parameters `θ` are broadcast to all of them. From
//! that one graph come both lowerings the integrator wants:
//!
//! - **eval**: `f` and `pdf` per vertex;
//! - **adjoint**: `δθ = Σ_v δf_v · ∂f_v/∂θ`, which is
//!   [`TensorGraph::backward`] of `s = Σ_v δf_v · f_v`. The sum over the
//!   batch is the reduction inside the graph — the CPU stand-in for the GPU's
//!   atomic add.
//!
//! The model is isotropic GGX (Trowbridge–Reitz) with the height-correlated
//! Smith term and Schlick's Fresnel:
//!
//! ```text
//! D  = α² / (π (cos²θh (α² − 1) + 1)²)
//! Λ(c) = (√(1 + α² (1 − c²) / c²) − 1) / 2
//! f  = F(o·h) · D · 1/(1 + Λ(i) + Λ(o)) / (4 cosθi cosθo)
//! pdf = D · 1/(1 + Λ(o)) / (4 cosθo)            (visible normals, Heitz 2018)
//! ```
//!
//! `θ = (α, F0₀, F0₁, F0₂)`. `F0` is a spectrum: three hat functions over the
//! band, and a vertex carries its hero wavelength's three hat weights, so
//! `F0(λ) = Σ_k b_k(λ) θ_{1+k}` is linear in the parameters.
//!
//! Sampling ([`sample`]) is plain f64 on the CPU and **detached**: it is never
//! differentiated, and the integrator treats the pdf it returns as a number.

use tang_compile::tensor::{Arg, Form, Monoid, TensorGraph, TensorId, View};
use tang_expr::ExprId;

use crate::ir::Program;

/// Number of parameters: roughness and three F0 bands.
pub const PARAMS: usize = 4;

/// Wavelengths (nm) the three F0 hats peak at.
pub const BANDS_NM: [f64; 3] = [450.0, 550.0, 650.0];

/// The three hat weights of `lambda_nm`: piecewise linear between the band
/// centres, flat beyond them, and summing to one.
pub fn band_weights(lambda_nm: f64) -> [f64; 3] {
    let [a, b, c] = BANDS_NM;
    if lambda_nm <= a {
        [1.0, 0.0, 0.0]
    } else if lambda_nm <= b {
        let t = (lambda_nm - a) / (b - a);
        [1.0 - t, t, 0.0]
    } else if lambda_nm <= c {
        let t = (lambda_nm - b) / (c - b);
        [0.0, 1.0 - t, t]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Graph input columns, in order. Every one is `[n]`.
pub const COLUMNS: usize = 7;

/// The conductor's graph over a wavefront of `n` vertices.
///
/// Inputs, in order: `cos_i, cos_o, cos_h, o·h, b₀, b₁, b₂` (each `[n]`),
/// `θ` (`[PARAMS]`), `δf` (`[n]`). Outputs: `f`, `pdf`, `δθ`.
pub struct Graph {
    /// The graph itself.
    pub g: TensorGraph,
    /// `f` per vertex, `[n]`.
    pub f: TensorId,
    /// `pdf` per vertex, `[n]`, with respect to solid angle at `i`.
    pub pdf: TensorId,
    /// `Σ_v δf_v ∂f_v/∂θ`, `[PARAMS]`.
    pub dtheta: TensorId,
}

fn lit(v: f64) -> ExprId {
    <ExprId as tang::Scalar>::from_f64(v)
}

fn sqrt(x: ExprId) -> ExprId {
    <ExprId as tang::Scalar>::sqrt(x)
}

/// `Λ(c)` of the Smith term, for roughness `a2 = α²`.
fn lambda(a2: ExprId, c: ExprId) -> ExprId {
    let c2 = c * c;
    (sqrt(lit(1.0) + a2 * (lit(1.0) - c2) / c2) - lit(1.0)) * lit(0.5)
}

impl Graph {
    /// Build it for a wavefront of `n`.
    pub fn new(n: usize) -> Self {
        let mut g = TensorGraph::new();
        let cols: Vec<TensorId> = (0..COLUMNS).map(|_| g.input(&[n])).collect();
        let theta = g.input(&[PARAMS]);
        let df = g.input(&[n]);
        let col = |t: TensorId| Arg {
            src: t,
            view: View::identity(1, 1),
        };
        let param = |k: i64| Arg {
            src: theta,
            view: View {
                guard: false,
                forms: vec![Form {
                    coefs: vec![0],
                    offset: k,
                }],
            },
        };
        let mut args: Vec<Arg> = cols.iter().map(|&c| col(c)).collect();
        args.extend((0..PARAMS as i64).map(param));

        let f = g
            .comp(&[n], &[] as &[usize], args.clone(), None, |x| {
                let (ci, co, ch, oh) = (x[0], x[1], x[2], x[3]);
                let a2 = x[7] * x[7];
                let f0 = x[4] * x[8] + x[5] * x[9] + x[6] * x[10];
                let m = lit(1.0) - oh;
                let m2 = m * m;
                let fresnel = f0 + (lit(1.0) - f0) * m2 * m2 * m;
                let k = ch * ch * (a2 - lit(1.0)) + lit(1.0);
                let d = a2 / (lit(std::f64::consts::PI) * k * k);
                let g2 = lit(1.0) / (lit(1.0) + lambda(a2, ci) + lambda(a2, co));
                vec![fresnel * d * g2 / (lit(4.0) * ci * co)]
            })
            .unwrap();
        let pdf = g
            .comp(&[n], &[] as &[usize], args, None, |x| {
                let (co, ch) = (x[1], x[2]);
                let a2 = x[7] * x[7];
                let k = ch * ch * (a2 - lit(1.0)) + lit(1.0);
                let d = a2 / (lit(std::f64::consts::PI) * k * k);
                let g1 = lit(1.0) / (lit(1.0) + lambda(a2, co));
                vec![d * g1 / (lit(4.0) * co)]
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
        Self { g, f, pdf, dtheta }
    }

    /// The eval lowering: outputs `[f, pdf]`.
    pub fn eval(&self, native: bool) -> Program {
        self.program(&[self.f, self.pdf], native)
    }

    /// The adjoint lowering: output `[δθ]`.
    pub fn adjoint(&self, native: bool) -> Program {
        self.program(&[self.dtheta], native)
    }

    fn program(&self, outputs: &[TensorId], native: bool) -> Program {
        if native {
            Program::compile(&self.g, outputs)
        } else {
            Program::interpreted(&self.g, outputs)
        }
    }
}

/// The adjoint per lane, for the device's compacted replay (speed round 2).
///
/// [`Graph`]'s `f`, with each `θ_k` read as `θ_k + z_k[v]` through four
/// extra `[n]` inputs that are always zero, so `f` is the same numbers; the
/// gradient of `Σ_v δf_v f_v` with respect to `z_k` is then
/// `δf_v ∂f_v/∂θ_k` per lane — `TensorGraph::backward`'s, pointwise, with
/// no reduction over lanes. The device sums those over the compact lanes
/// only (a fixed-shape tree, `compact.wgsl`), where [`Graph::dtheta`]'s
/// reduction runs over every lane of the wavefront.
///
/// Inputs, in order: `cos_i, cos_o, cos_h, o·h, b₀, b₁, b₂` (each `[n]`),
/// `θ` (`[PARAMS]`), `δf` (`[n]`), `z₀ … z₃` (each `[n]`).
pub struct LaneAdjoint {
    /// The graph itself.
    pub g: TensorGraph,
    /// `δf_v ∂f_v/∂θ_k`, four `[n]`.
    pub dz: [TensorId; PARAMS],
}

impl LaneAdjoint {
    /// Build it for a wavefront of `n`.
    pub fn new(n: usize) -> Self {
        let mut g = TensorGraph::new();
        let cols: Vec<TensorId> = (0..COLUMNS).map(|_| g.input(&[n])).collect();
        let theta = g.input(&[PARAMS]);
        let df = g.input(&[n]);
        let z: Vec<TensorId> = (0..PARAMS).map(|_| g.input(&[n])).collect();
        let col = |t: TensorId| Arg {
            src: t,
            view: View::identity(1, 1),
        };
        let param = |k: i64| Arg {
            src: theta,
            view: View {
                guard: false,
                forms: vec![Form {
                    coefs: vec![0],
                    offset: k,
                }],
            },
        };
        let mut args: Vec<Arg> = cols.iter().map(|&c| col(c)).collect();
        args.extend((0..PARAMS as i64).map(param));
        args.extend(z.iter().map(|&t| col(t)));
        let f = g
            .comp(&[n], &[] as &[usize], args, None, |x| {
                let (ci, co, ch, oh) = (x[0], x[1], x[2], x[3]);
                let t = |k: usize| x[7 + k] + x[7 + PARAMS + k];
                let a2 = t(0) * t(0);
                let f0 = x[4] * t(1) + x[5] * t(2) + x[6] * t(3);
                let m = lit(1.0) - oh;
                let m2 = m * m;
                let fresnel = f0 + (lit(1.0) - f0) * m2 * m2 * m;
                let k = ch * ch * (a2 - lit(1.0)) + lit(1.0);
                let d = a2 / (lit(std::f64::consts::PI) * k * k);
                let g2 = lit(1.0) / (lit(1.0) + lambda(a2, ci) + lambda(a2, co));
                vec![fresnel * d * g2 / (lit(4.0) * ci * co)]
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
        let dz = g.backward(s, &z).unwrap();
        Self {
            g,
            dz: std::array::from_fn(|k| dz[k]),
        }
    }
}

/// Graph input index of `θ`.
pub const THETA: usize = COLUMNS;
/// Graph input index of `δf`.
pub const DF: usize = COLUMNS + 1;

/// `f` and `pdf` in plain f64, for tests that want a third opinion.
pub fn eval_f64(
    theta: &[f64; PARAMS],
    b: [f64; 3],
    ci: f64,
    co: f64,
    ch: f64,
    oh: f64,
) -> (f64, f64) {
    let a2 = theta[0] * theta[0];
    let lam = |c: f64| ((1.0 + a2 * (1.0 - c * c) / (c * c)).sqrt() - 1.0) * 0.5;
    let f0 = b[0] * theta[1] + b[1] * theta[2] + b[2] * theta[3];
    let fr = f0 + (1.0 - f0) * (1.0 - oh).powi(5);
    let k = ch * ch * (a2 - 1.0) + 1.0;
    let d = a2 / (std::f64::consts::PI * k * k);
    (
        fr * d / (1.0 + lam(ci) + lam(co)) / (4.0 * ci * co),
        d / (1.0 + lam(co)) / (4.0 * co),
    )
}

/// Sample an incident direction for outgoing `wo`, both in the local frame
/// (`z` the normal), from GGX's visible normals at roughness `alpha` (Heitz,
/// "Sampling the GGX Distribution of Visible Normals", JCGT 2018). Returns
/// `(wi, pdf)`; `wi.z ≤ 0` means the sample went under the surface.
///
/// Detached: `alpha` here is a number, and nothing differentiates this.
pub fn sample(alpha: f64, wo: [f64; 3], u: [f64; 2]) -> ([f64; 3], f64) {
    let vh = normalize([alpha * wo[0], alpha * wo[1], wo[2]]);
    let len2 = vh[0] * vh[0] + vh[1] * vh[1];
    let t1 = if len2 > 0.0 {
        scale([-vh[1], vh[0], 0.0], 1.0 / len2.sqrt())
    } else {
        [1.0, 0.0, 0.0]
    };
    let t2 = cross(vh, t1);
    let r = u[0].sqrt();
    let phi = 2.0 * std::f64::consts::PI * u[1];
    let p1 = r * phi.cos();
    let s = 0.5 * (1.0 + vh[2]);
    let p2 = (1.0 - s) * (1.0 - p1 * p1).max(0.0).sqrt() + s * r * phi.sin();
    let p3 = (1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt();
    let nh = [
        p1 * t1[0] + p2 * t2[0] + p3 * vh[0],
        p1 * t1[1] + p2 * t2[1] + p3 * vh[1],
        p1 * t1[2] + p2 * t2[2] + p3 * vh[2],
    ];
    let h = normalize([alpha * nh[0], alpha * nh[1], nh[2].max(1e-12)]);
    let oh = dot(wo, h);
    let wi = [
        2.0 * oh * h[0] - wo[0],
        2.0 * oh * h[1] - wo[1],
        2.0 * oh * h[2] - wo[2],
    ];
    (wi, pdf(alpha, wo[2], h[2]))
}

/// Visible-normal pdf of `wi` (solid angle) given `cos θo` and `cos θh`.
pub fn pdf(alpha: f64, co: f64, ch: f64) -> f64 {
    eval_f64(&[alpha, 0.0, 0.0, 0.0], [0.0; 3], 1.0, co, ch, 1.0).1
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// `a` scaled to unit length.
pub fn normalize(a: [f64; 3]) -> [f64; 3] {
    scale(a, 1.0 / dot(a, a).sqrt())
}
