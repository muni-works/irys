//! The score and the judge.
//!
//! ```text
//! MDL(program, θ̂) = ½ Σ r²                         residual, nats (σ_n known; constant dropped)
//!                 + Σ log range_i + ½ log det F − (k/2) log 2π    parameter cost (Laplace, uniform prior)
//!                 − log p(program)                  grammar prior
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
#[derive(Debug, Clone, Copy, serde::Serialize)]
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

/// `log det` of an SPD matrix via its eigenvalues (floored).
fn logdet(a: &[f64], n: usize) -> f64 {
    let (ev, _) = eigh(a, n);
    ev.iter().map(|&e| e.max(1e-12).ln()).sum()
}

/// Score a fit.
pub fn score(fit: &Fit) -> Score {
    let info = fit.prog.param_info();
    let k = info.len();
    let ranges: f64 = info.iter().map(|i| i.range.ln()).sum();
    let params = ranges + 0.5 * logdet(&fit.fisher, k) - 0.5 * k as f64 * std::f64::consts::TAU.ln();
    let program = program_length(&fit.prog);
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
