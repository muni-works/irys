//! Risers from the real fit-loss curves.
//!
//! The best-so-far loss (running minimum) is normalised to a progress
//! `p ∈ [0,1]`: 0 at the first iteration, 1 at the final best. The fit has
//! *converged* at the first iteration where `p ≥ 0.98`. The riser's span is
//! stretched so that iteration lands exactly on the section's end — the drop.
//! Progress drives a band-passed noise sweep (300 Hz → 13 kHz) and a pitch
//! riser (two octaves); a plateau in the loss is a plateau in the riser.

use crate::dsp::{Rng, Svf, pan};
use serde::Serialize;
use std::f64::consts::PI;
use std::path::Path;

pub struct Curve {
    /// Progress per iteration.
    pub p: Vec<f64>,
    pub converged_at: usize,
    pub first: f64,
    pub best: f64,
}

#[derive(Serialize)]
pub struct RiserOut {
    pub csv: String,
    pub start: f64,
    pub drop: f64,
    pub iterations: usize,
    pub converged_iteration: usize,
    pub loss_first: f64,
    pub loss_best: f64,
}

impl Curve {
    pub fn load(path: &Path, column: &str, run: Option<&str>) -> Result<Curve, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut lines = text.lines();
        let head: Vec<&str> = lines.next().ok_or("empty csv")?.split(',').collect();
        let col = head.iter().position(|h| h.trim() == column).ok_or_else(|| format!("no column {column}"))?;
        let run_col = head.iter().position(|h| h.trim() == "run");
        // A quoted or ';'-bearing run name has no commas, so a plain split is safe.
        let loss: Vec<f64> = lines
            .filter(|l| match (run, run_col) {
                (Some(r), Some(c)) => l.split(',').nth(c).map(str::trim) == Some(r),
                _ => true,
            })
            .filter_map(|l| l.split(',').nth(col)?.trim().parse().ok())
            .collect();
        if loss.is_empty() {
            return Err(format!("{}: no rows for run {run:?}", path.display()).into());
        }
        let mut best = f64::INFINITY;
        let run: Vec<f64> = loss.iter().map(|&l| { best = best.min(l); best }).collect();
        let (first, last) = (run[0], *run.last().unwrap());
        let span = (first - last).max(1e-12);
        let p: Vec<f64> = run.iter().map(|&m| (first - m) / span).collect();
        let converged_at = p.iter().position(|&v| v >= 0.98).unwrap_or(p.len() - 1);
        Ok(Curve { p, converged_at, first, best: last })
    }

    /// Progress at fraction `u ∈ [0,1]` of the way to convergence.
    fn at(&self, u: f64) -> f64 {
        let x = u.clamp(0.0, 1.0) * self.converged_at as f64;
        let i = x.floor() as usize;
        let j = (i + 1).min(self.p.len() - 1);
        let f = x - i as f64;
        self.p[i] * (1.0 - f) + self.p[j] * f
    }

    pub fn render(&self, out: &mut [[f64; 2]], t0: f64, t1: f64, level: f64, sr: f64, rng: &mut Rng, name: &str) -> RiserOut {
        let (a, b) = ((t0 * sr) as usize, ((t1 * sr) as usize).min(out.len()));
        let mut f = [Svf::default(), Svf::default()];
        let mut ph = 0.0;
        // The last sixteenth before the drop is silence: the inhale.
        let gap = (b - a).min((0.06 * sr) as usize);
        for i in a..b - gap {
            let u = (i - a) as f64 / (b - a - gap) as f64;
            let p = self.at(u);
            let hz = 300.0 * 2f64.powf(5.4 * p);
            let env = (0.15 + 0.85 * p).powf(1.5) * u.sqrt();
            let tone_hz = 110.0 * 2f64.powf(2.0 * p);
            ph = (ph + tone_hz / sr) % 1.0;
            let tone = ((2.0 * PI * ph).sin() + 0.5 * (4.0 * PI * ph).sin()) * 0.05 * p * p;
            for c in 0..2 {
                let n = f[c].tick(rng.bipolar(), hz, 2.0 + 5.0 * p, sr).band;
                let pn = pan(if c == 0 { -0.6 } else { 0.6 });
                out[i][c] += (n * 0.35 * pn[c] * 1.4 + tone) * env * level;
            }
        }
        RiserOut {
            csv: name.to_string(),
            start: t0,
            drop: t1,
            iterations: self.p.len(),
            converged_iteration: self.converged_at + 1,
            loss_first: self.first,
            loss_best: self.best,
        }
    }
}
