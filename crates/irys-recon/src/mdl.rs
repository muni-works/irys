//! The score and the judge.
//!
//! ```text
//! MDL(program, θ̂) = ½ Σ r²                         residual, nats (σ_n known; constant dropped)
//!                 + Σ_eig max(0, ½ log(λ(R F R) / 2π))   parameter cost (Laplace, uniform prior, ranges R)
//!                 − log p(program) + k               grammar prior, one nat per parameter
//! ```
//!
//! `F = JᵀJ / σ_n²` is the Gauss–Newton Fisher information of the image
//! term at the fit. The grammar prior: the corner kind uniform over three, a
//! geometric number of features (`p(k) = 2^-(k+1)`), each feature's kind
//! uniform over six, a slot's axis over two, a linear pattern's count over
//! `2..=6`, a circular pattern's over `3..=8`.

use crate::dsl::{Feature, Program};
use crate::fit::{Fit, eigh};

/// `−log p(program)` under the grammar, nats.
pub fn program_length(p: &Program) -> f64 {
    let ln = f64::ln;
    let mut l = ln(3.0) + (p.features.len() + 1) as f64 * ln(2.0);
    for f in &p.features {
        l += ln(6.0);
        l += match f {
            Feature::Slot { .. } => ln(2.0),
            Feature::LinearHoles { .. } => ln(5.0),
            Feature::CircularHoles { .. } => ln(6.0),
            _ => 0.0,
        };
    }
    l
}

/// The score's three terms.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Score {
    /// Residual, nats.
    pub residual: f64,
    /// Parameter cost, nats.
    pub params: f64,
    /// Program length, nats.
    pub program: f64,
    /// Their sum.
    pub total: f64,
}

/// Each continuous parameter's floor in the program length: writing a
/// number down costs something even when the photo pins it for free, and
/// without it an extra parameter the photo cannot see is a free lunch.
pub const NATS_PER_PARAM: f64 = 1.0;

/// Score a fit.
///
/// The parameter cost is the Laplace term whitened by the prior ranges
/// `R`: per eigen-direction `λ` of `R F R`, `max(0, ½ log(λ / 2π))` — the
/// nats it takes to state that direction to the precision the photo gives,
/// and nothing for a direction the photo does not determine (the posterior
/// there is the prior). The bare `½ log det F` would instead *reward* an
/// unidentifiable parameter with `−∞`.
pub fn score(fit: &Fit) -> Score {
    let info = fit.prog.param_info();
    let k = info.len();
    let mut w = fit.fisher.clone();
    for i in 0..k {
        for j in 0..k {
            w[i * k + j] *= info[i].range * info[j].range;
        }
    }
    let (ev, _) = eigh(&w, k);
    let params: f64 = ev.iter().map(|&l| (0.5 * (l / std::f64::consts::TAU).max(1e-300).ln()).max(0.0)).sum();
    let program = program_length(&fit.prog) + NATS_PER_PARAM * k as f64;
    Score { residual: fit.nll, params, program, total: fit.nll + params + program }
}

/// The judge's verdict on one fit.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Judge {
    /// Per parameter: name and ±σ from `F⁻¹`.
    pub sigma: Vec<(String, f64)>,
    /// Near-null directions of `F`: `(σ along it, the parameters it mixes
    /// with their weights)`, for every eigen-direction whose σ exceeds
    /// [`UNIDENTIFIABLE_MM`].
    pub weak: Vec<(f64, Vec<(String, f64)>)>,
    /// `F⁻¹`, row-major.
    pub covariance: Vec<f64>,
}

/// A direction is flagged when the photo pins it no better than this.
pub const UNIDENTIFIABLE_MM: f64 = 0.5;

/// Covariances and unidentifiable directions of a fit.
pub fn judge(fit: &Fit) -> Judge {
    let info = fit.prog.param_info();
    let k = info.len();
    let (ev, vecs) = eigh(&fit.fisher, k);
    // F⁻¹ by eigen-decomposition (pseudo-inverse past 1e-12).
    let mut cov = vec![0.0; k * k];
    for (e, &lam) in ev.iter().enumerate() {
        let inv = if lam > 1e-12 { 1.0 / lam } else { 1e12 };
        for i in 0..k {
            for j in 0..k {
                cov[i * k + j] += vecs[i * k + e] * vecs[j * k + e] * inv;
            }
        }
    }
    let sigma = info.iter().enumerate().map(|(i, p)| (p.name.clone(), cov[i * k + i].sqrt())).collect();
    let mut weak = Vec::new();
    for (e, &lam) in ev.iter().enumerate() {
        let s = 1.0 / lam.max(1e-24).sqrt();
        if s > UNIDENTIFIABLE_MM {
            let mut mix: Vec<(String, f64)> =
                (0..k).map(|i| (info[i].name.clone(), vecs[i * k + e])).filter(|(_, w)| w.abs() > 0.2).collect();
            mix.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
            weak.push((s, mix));
        }
    }
    weak.sort_by(|a, b| b.0.total_cmp(&a.0));
    Judge { sigma, weak, covariance: cov }
}
