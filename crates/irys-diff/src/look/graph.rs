//! The OpenPBR stack as one tang IR graph.
//!
//! [`stack`] is written once, generic over [`tang::Scalar`]: instantiated at
//! `ExprId` it is the body of [`Stack`]'s comprehension (eval and, through
//! `TensorGraph::backward`, the adjoint); at `f64` it is the integrator's
//! `f`. One source, so the graph and its twin cannot drift.
//!
//! A lane is one of three kinds (one-hot columns): a reflection
//! (`f_r`, every reflective lobe), a refraction (`f_t`, the rough
//! dielectric BTDF under the coat and fuzz) or a subsurface entry (`f_s`,
//! the dimensionless weight into the walk). Everything that depends on `θ`
//! is computed in the graph; every θ-free number (Fresnel factors at fixed
//! indices, the coat and fuzz lobes, the film) is a column
//! ([`super::openpbr::At::columns`]). The tabulated albedos that depend on
//! the roughness `θ₄` enter linearised at the current roughness,
//! `E = E(r₀) + ∂E/∂r · (r − r₀)`: the value is the table's and the
//! derivative is the table's, so the adjoint is the interpolant's exact
//! gradient.

use tang::Scalar;
use tang_compile::tensor::{Arg, Form, Monoid, TensorGraph, TensorId, View};
use tang_expr::ExprId;

use super::openpbr::{A_MIN, COLUMNS, MU_BAR, PARAMS, col::*};
use crate::ir::Program;

/// The stack's `f` for one lane: `c` its [`COLUMNS`] columns, `t` the
/// [`PARAMS`] parameters.
pub fn stack<S: Scalar>(c: &[S], t: &[S]) -> S {
    let l = S::from_f64;
    let one = l(1.0);
    let cc = |k: usize| c[k];
    let base = cc(B0) * t[0] + cc(B0 + 1) * t[1] + cc(B0 + 2) * t[2];
    let (met, r, cw, fw, wb) = (t[3], t[4], t[5], t[6], t[7]);
    let (ci, co, ch, oh) = (cc(CI), cc(CO), cc(CH), cc(OH));
    let a = l(A_MIN) + l(1.0 - A_MIN) * r * r;
    let a2 = a * a;
    let k = ch * ch * (a2 - one) + one;
    let d = a2 / (l(std::f64::consts::PI) * k * k);
    let lam = |x: S| ((one + a2 * (one - x * x) / (x * x)).sqrt() - one) * l(0.5);
    let g2 = one / (one + lam(ci) + lam(co));
    let dg = d * g2 / (l(4.0) * ci * co);
    let dr = r - cc(R0);
    let sc = cc(SC);
    let f0 = wb * base;
    let p5 = |x: S| {
        let m = one - x;
        let m2 = m * m;
        m2 * m2 * m
    };
    let fs = |x: S| f0 + (one - f0) * p5(x);
    let fs_bar = fs(l(MU_BAR));
    let kk = (one - sc) * fs_bar;
    let w82 = {
        let m = one - oh;
        let m3 = m * m * m;
        oh * m3 * m3 / l(MU_BAR * (1.0 - MU_BAR).powi(6))
    };
    let f82 = fs(oh) - w82 * kk;
    let wfm = cc(W_FILM_M);
    let f_metal = cc(XI_S) * ((one - wfm) * f82 + wfm * cc(FILM_M));
    let (tw, sw) = (cc(T), cc(S));
    let pi = l(std::f64::consts::PI);
    let lin = |k0: usize| cc(k0) + cc(k0 + 1) * dr;
    let (e1, e5, ew) = (lin(EM1), lin(EM1 + 2), lin(EM1 + 4));
    let em1 = cc(XI_S) * (f0 * (e1 - e5) + e5 - kk * ew);
    let etap = cc(ETAP);
    let btdf = d * g2 * cc(IH) * oh / (ci * co * cc(DENOM2) * etap * etap);
    // The base substrate with one set of dielectric columns (`k`, the
    // uncoated or the coated index): its reflection, refraction and
    // subsurface-entry parts, and its normal-incidence albedo.
    let branch = |k: [usize; 6]| {
        let (fd, es, es1) = (
            cc(k[0]),
            (cc(k[1]) + cc(k[2]) * dr) * cc(k[5]),
            cc(k[3]) + cc(k[4]) * dr,
        );
        let diffuse = (one - tw) * (one - sw) * (one - sc * es) * wb * base / pi;
        let refl = met * f_metal * dg + (one - met) * (sc * fd * dg + diffuse);
        let trans = (one - met) * tw * cc(TC) * (one - fd) * btdf;
        let sss = (one - met) * (one - tw) * sw * (one - sc * es);
        let eb = met * em1
            + (one - met)
                * (sc * es1
                    + (one - sc * es1) * (one - tw) * ((one - sw) * wb * base + sw * cc(SSC)));
        (refl, trans, sss, eb)
    };
    let (ru, tu, su, _) = branch(UNCOATED);
    let (rc, tc, sc_, eb) = branch(COATED);
    // Darkening of the coated base: its normal-incidence albedo against the
    // coat's internal reflection.
    let rd = one + (r - one) * cc(XS_FS);
    let rb = rd + (r - rd) * met;
    let kr = cc(K_S) + (cc(K_R) - cc(K_S)) * rb;
    let delta = (one - kr) / (one - eb * kr);
    let dark = one + (delta - one) * cc(DELTA);
    let under = cc(T_COAT) * (one - cc(E_COAT)) * dark;
    // layer(base, coat, C) = (1 − C) base + C (f_coat + T (1 − E_c) B Δ base_c)
    let coat = |u: S, c: S| (one - cw) * u + cw * under * c;
    let fuzzmul = one - fw * cc(E_FUZZ);
    let f_r = fw * cc(F_FUZZ) + fuzzmul * (cw * cc(F_COAT) + coat(ru, rc));
    let f_t = fuzzmul * coat(tu, tc);
    let f_s = fuzzmul * coat(su, sc_);
    cc(KR) * f_r + cc(KT) * f_t + cc(KS) * f_s
}

/// The stack's graph over `n` lanes. Inputs, in order: the lanes'
/// columns as one `[n, COLUMNS]` row-major input (one storage buffer on the
/// device, which allows 29 per stage), `θ` (`[PARAMS]`), `δf` (`[n]`). Outputs `f` and
/// `δθ = Σ_v δf_v ∂f_v/∂θ`.
pub struct Stack {
    /// The graph.
    pub g: TensorGraph,
    /// `f`, `[n]`.
    pub f: TensorId,
    /// `δθ`, `[PARAMS]`.
    pub dtheta: TensorId,
}

/// Graph input index of `θ`.
pub const THETA: usize = 1;
/// Graph input index of `δf`.
pub const DF: usize = 2;

impl Stack {
    /// Build it for `n` lanes.
    pub fn new(n: usize) -> Self {
        let mut g = TensorGraph::new();
        let cols = g.input(&[n, COLUMNS]);
        let theta = g.input(&[PARAMS]);
        let df = g.input(&[n]);
        let mut args: Vec<Arg> = (0..COLUMNS as i64)
            .map(|k| Arg {
                src: cols,
                view: View {
                    guard: false,
                    forms: vec![
                        Form {
                            coefs: vec![1],
                            offset: 0,
                        },
                        Form {
                            coefs: vec![0],
                            offset: k,
                        },
                    ],
                },
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
                vec![stack(&x[..COLUMNS], &x[COLUMNS..])]
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

/// Run the adjoint over lanes `rows` (each [`COLUMNS`] long) with seeds
/// `df`, in chunks of `width`, reducing in order: `Σ δf ∂f/∂θ`.
pub fn reduce(
    prog: &mut Program,
    width: usize,
    rows: &[[f64; COLUMNS]],
    df: &[f64],
    theta: &[f64; PARAMS],
) -> [f64; PARAMS] {
    let mut out = [0.0; PARAMS];
    prog.write(THETA, theta);
    for (chunk, dchunk) in rows.chunks(width).zip(df.chunks(width)) {
        let mut v = Vec::with_capacity(width * COLUMNS);
        for r in chunk {
            v.extend_from_slice(r);
        }
        // Pad lanes: a valid reflection lane with δf = 0.
        for _ in chunk.len()..width {
            v.extend((0..COLUMNS).map(pad));
        }
        prog.write(0, &v);
        let mut d = vec![0.0; width];
        d[..dchunk.len()].copy_from_slice(dchunk);
        prog.write(DF, &d);
        prog.run();
        let g = prog.read(0);
        for k in 0..PARAMS {
            out[k] += g[k];
        }
    }
    out
}

/// A harmless column value for padding lanes.
fn pad(k: usize) -> f64 {
    match k {
        KR | CI | CO | CH | OH | ETAP | DENOM2 | FILM_RATIO => 1.0,
        _ => 0.0,
    }
}
