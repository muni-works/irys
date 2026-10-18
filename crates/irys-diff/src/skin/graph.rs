//! Skin's two differentiable factors, each written once as a tang IR graph
//! batched over a column of vertices, like [`crate::ggx`]:
//!
//! - [`Spec`]: the oily layer, a dual-lobe GGX dielectric (Schlick at
//!   `η = 1.4`, height-correlated Smith), lobes mixed 0.85 / 0.15 at
//!   perceptual roughness `θ₃`, `θ₄` (`α = r²`), scaled per vertex by the
//!   specular map. The two-lobe split is the one d'Eon & Luebke measured
//!   for skin ("Advanced Techniques for Realistic Real-Time Skin
//!   Rendering", GPU Gems 3 ch. 14), the roughnesses Unreal's / Lobe's.
//! - [`Medium`]: one random-walk event in the skin medium,
//!   `f = σs^s e^{−σt t}` (`s = 1` a scatter at `t`, `s = 0` the walk
//!   leaving through the boundary at `t`), with
//!   `σa = θ₂ (θ₀ m Eₘ + θ₁ h Eₕ + E_b)`, `σs = θ₂ S` per lane
//!   ([`super::optics`]). The graph is written in the log domain,
//!   `ℓ = s ln σs − t σt`, `f = e^ℓ`; its adjoint is `backward(Σ δℓ · ℓ)`.
//!   `ℓ` is linear in `(s, t)` for fixed coefficients, so a whole walk's
//!   events reduce exactly to one lane whose `s` is `Σ δℓ_e s_e` and `t`
//!   is `Σ δℓ_e t_e` (with `δℓ = 1`): the replay runs the adjoint once per
//!   walk, not once per event.
//!
//! `θ = (melanin scale, blood scale, density scale, r₁, r₂)`.

use tang_compile::tensor::{Arg, Form, Monoid, TensorGraph, TensorId, View};
use tang_expr::ExprId;

use crate::ir::Program;

/// Parameters: melanin, blood, density scale, two roughnesses.
pub const PARAMS: usize = 5;

/// Lobe weights of the two specular lobes.
pub const MIX: [f64; 2] = [0.85, 0.15];

/// Input columns of [`Spec`]: `cos_i, cos_o, cos_h, o·h, k` (map scale).
pub const SPEC_COLUMNS: usize = 5;

/// Input columns of [`Medium`]: `s, t, m, h, Eₘ, Eₕ, E_b, S`.
pub const MEDIUM_COLUMNS: usize = 8;

fn lit(v: f64) -> ExprId {
    <ExprId as tang::Scalar>::from_f64(v)
}

fn sqrt(x: ExprId) -> ExprId {
    <ExprId as tang::Scalar>::sqrt(x)
}

fn lambda(a2: ExprId, c: ExprId) -> ExprId {
    let c2 = c * c;
    (sqrt(lit(1.0) + a2 * (lit(1.0) - c2) / c2) - lit(1.0)) * lit(0.5)
}

/// A graph's outputs: the factor per lane and the adjoint.
pub struct Built {
    /// The graph.
    pub g: TensorGraph,
    /// `f` per lane, `[n]`.
    pub f: TensorId,
    /// `δθ`, `[PARAMS]`.
    pub dtheta: TensorId,
}

impl Built {
    /// The eval lowering, `[f]`.
    pub fn eval(&self, native: bool) -> Program {
        self.program(&[self.f], native)
    }

    /// The adjoint lowering, `[δθ]`.
    pub fn adjoint(&self, native: bool) -> Program {
        self.program(&[self.dtheta], native)
    }

    fn program(&self, out: &[TensorId], native: bool) -> Program {
        if native {
            Program::compile(&self.g, out)
        } else {
            Program::interpreted(&self.g, out)
        }
    }
}

fn args(g: &mut TensorGraph, n: usize, cols: usize) -> (Vec<Arg>, TensorId, TensorId) {
    let cs: Vec<TensorId> = (0..cols).map(|_| g.input(&[n])).collect();
    let theta = g.input(&[PARAMS]);
    let d = g.input(&[n]);
    let mut a: Vec<Arg> = cs
        .iter()
        .map(|&c| Arg {
            src: c,
            view: View::identity(1, 1),
        })
        .collect();
    a.extend((0..PARAMS as i64).map(|k| Arg {
        src: theta,
        view: View {
            guard: false,
            forms: vec![Form {
                coefs: vec![0],
                offset: k,
            }],
        },
    }));
    (a, theta, d)
}

fn reduce(g: &mut TensorGraph, n: usize, d: TensorId, x: TensorId) -> TensorId {
    let pick = |t: TensorId| Arg {
        src: t,
        view: View::pick(2, &[Some(1)]),
    };
    g.comp(
        &[1usize],
        &[n],
        vec![pick(d), pick(x)],
        Some(Monoid::sum()),
        |x| vec![x[0] * x[1]],
    )
    .unwrap()
}

/// The dual-lobe specular graph over `n` lanes. Inputs: the five columns,
/// `θ`, `δf`.
pub struct Spec;

impl Spec {
    /// Build it.
    pub fn build(n: usize) -> Built {
        let mut g = TensorGraph::new();
        let (a, theta, df) = args(&mut g, n, SPEC_COLUMNS);
        let f = g
            .comp(&[n], &[] as &[usize], a, None, |x| {
                let (ci, co, ch, oh, k) = (x[0], x[1], x[2], x[3], x[4]);
                let m = lit(1.0) - oh;
                let m2 = m * m;
                let fr = lit(super::optics::F0) + lit(1.0 - super::optics::F0) * m2 * m2 * m;
                let lobe = |r: ExprId| {
                    let a2 = r * r * r * r;
                    let q = ch * ch * (a2 - lit(1.0)) + lit(1.0);
                    let d = a2 / (lit(std::f64::consts::PI) * q * q);
                    d / (lit(1.0) + lambda(a2, ci) + lambda(a2, co))
                };
                let s = lit(MIX[0]) * lobe(x[SPEC_COLUMNS + 3])
                    + lit(MIX[1]) * lobe(x[SPEC_COLUMNS + 4]);
                vec![k * fr * s / (lit(4.0) * ci * co)]
            })
            .unwrap();
        let s = reduce(&mut g, n, df, f);
        let dtheta = g.backward(s, &[theta]).unwrap()[0];
        Built { g, f, dtheta }
    }
}

/// The medium-event graph over `n` lanes. Inputs: the eight columns, `θ`,
/// `δℓ`. Its `f` is `e^ℓ`; its adjoint differentiates `Σ δℓ · ℓ`.
pub struct Medium;

impl Medium {
    /// Build it.
    pub fn build(n: usize) -> Built {
        let mut g = TensorGraph::new();
        let (a, theta, dl) = args(&mut g, n, MEDIUM_COLUMNS);
        let ell = |x: &[ExprId]| {
            let (s, t, m, h, em, eh, eb, ss) = (x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
            let c = MEDIUM_COLUMNS;
            let scale = x[c + 2];
            let sa = scale * (x[c] * m * em + x[c + 1] * h * eh + eb);
            let sigma_s = scale * ss;
            s * <ExprId as tang::Scalar>::ln(sigma_s) - t * (sa + sigma_s)
        };
        let l = g
            .comp(&[n], &[] as &[usize], a.clone(), None, |x| vec![ell(x)])
            .unwrap();
        let f = g
            .comp(&[n], &[] as &[usize], a, None, |x| {
                vec![<ExprId as tang::Scalar>::exp(ell(x))]
            })
            .unwrap();
        let s = reduce(&mut g, n, dl, l);
        let dtheta = g.backward(s, &[theta]).unwrap()[0];
        Built { g, f, dtheta }
    }
}

/// [`Spec`]'s `f` in plain f64: the integrator's, and the graph's oracle.
pub fn spec_f64(theta: &[f64; PARAMS], ci: f64, co: f64, ch: f64, oh: f64, k: f64) -> f64 {
    let fr = super::optics::F0 + (1.0 - super::optics::F0) * (1.0 - oh).powi(5);
    let lobe = |r: f64| {
        let a2 = r * r * r * r;
        let lam = |c: f64| ((1.0 + a2 * (1.0 - c * c) / (c * c)).sqrt() - 1.0) * 0.5;
        let q = ch * ch * (a2 - 1.0) + 1.0;
        a2 / (std::f64::consts::PI * q * q) / (1.0 + lam(ci) + lam(co))
    };
    k * fr * (MIX[0] * lobe(theta[3]) + MIX[1] * lobe(theta[4])) / (4.0 * ci * co)
}

/// The visible-normal pdf of the lobe mixture (detached): what the
/// integrator divides a reflected sample by.
pub fn spec_pdf(theta: &[f64; PARAMS], co: f64, ch: f64) -> f64 {
    MIX[0] * crate::ggx::pdf(theta[3] * theta[3], co, ch)
        + MIX[1] * crate::ggx::pdf(theta[4] * theta[4], co, ch)
}

/// [`Medium`]'s `ℓ` in plain f64 for one event.
pub fn medium_ell_f64(theta: &[f64; PARAMS], cols: &[f64; MEDIUM_COLUMNS]) -> f64 {
    let [s, t, m, h, em, eh, eb, ss] = *cols;
    let sa = theta[2] * (theta[0] * m * em + theta[1] * h * eh + eb);
    let sigma_s = theta[2] * ss;
    s * sigma_s.ln() - t * (sa + sigma_s)
}

/// `∂ℓ/∂θ` for one event in plain f64 (tests).
pub fn medium_dell_f64(theta: &[f64; PARAMS], cols: &[f64; MEDIUM_COLUMNS]) -> [f64; PARAMS] {
    let [s, t, m, h, em, eh, eb, ss] = *cols;
    let sc = theta[2];
    [
        -t * sc * m * em,
        -t * sc * h * eh,
        s / sc - t * (theta[0] * m * em + theta[1] * h * eh + eb + ss),
        0.0,
        0.0,
    ]
}
