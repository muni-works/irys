//! Struck objects: modal synthesis from material constants and a free-free bar.
//!
//! The constants are kosm's (`kosm::material::named(..).modal()` gives rho, E,
//! nu and the loss factor η; `crates/kosm/src/material/library.rs`), copied
//! here rather than depended on so this tool does not pull kosm's graph (phyz,
//! vcad, a second tang) into irys. The physics follows `kosm::audio`:
//! Euler–Bernoulli free-free bending modes, `f_n = (β_n L)² / (2π L²) ·
//! √(E I / ρ A)` with `I/A = t²/12`, and the decay `π f / Q` with `Q = 1/η`
//! (`kosm::audio::bar_modes`). Strike weights are the free-free mode shapes at
//! the strike point.

use std::f64::consts::PI;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub struct Substance {
    pub name: &'static str,
    /// kg/m³
    pub rho: f64,
    /// Young's modulus, Pa
    pub e: f64,
    pub nu: f64,
    /// Structural loss factor η.
    pub loss: f64,
    pub source: &'static str,
}

pub const BRASS: Substance = Substance { name: "brass", rho: 8500.0, e: 100.0e9, nu: 0.34, loss: 1.0e-3, source: "kosm library" };
pub const GOLD: Substance = Substance { name: "gold leaf", rho: 19300.0, e: 79.0e9, nu: 0.44, loss: 3.0e-4, source: "kosm library" };
pub const GLASS: Substance = Substance { name: "glass", rho: 2500.0, e: 70.0e9, nu: 0.23, loss: 1.0e-3, source: "kosm library" };
pub const SKIN: Substance = Substance { name: "skin", rho: 1050.0, e: 0.5e6, nu: 0.45, loss: 0.30, source: "kosm library" };
pub const OAK: Substance = Substance { name: "oak", rho: 750.0, e: 11.0e9, nu: 0.35, loss: 0.010, source: "kosm library" };
pub const LIMESTONE: Substance = Substance { name: "limestone", rho: 2600.0, e: 50.0e9, nu: 0.25, loss: 0.01, source: "kosm library" };
/// kosm has no aluminium; CRC 6061-T6 constants, η doubled for the anodic layer (estimated).
pub const ANODISED_AL: Substance = Substance { name: "anodised aluminium", rho: 2700.0, e: 69.0e9, nu: 0.33, loss: 2.0e-3, source: "CRC 6061, η estimated" };

/// Free-free βL roots; beyond the fourth, (2n+1)π/2 is exact to 1e-6.
fn beta_l(n: usize) -> f64 {
    const R: [f64; 4] = [4.730_040_7, 7.853_204_6, 10.995_607_8, 14.137_165_5];
    if n < 4 { R[n] } else { (2 * n + 3) as f64 * PI / 2.0 }
}

/// |φ| of the free-free mode with eigenvalue `b` at `x ∈ [0,1]`, normalised so
/// the free end is 2.
fn mode_shape(b: f64, x: f64) -> f64 {
    let s = (b.cosh() - b.cos()) / (b.sinh() - b.sin());
    let bx = b * x;
    (bx.cosh() + bx.cos() - s * (bx.sinh() + bx.sin())).abs()
}

#[derive(Clone, Debug)]
pub struct Mode {
    pub hz: f64,
    /// Amplitude decay, 1/s.
    pub decay: f64,
    pub amp: f64,
}

/// A bar of `len` × `thk` (m), struck at fraction `at` of its length.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Voice {
    pub label: &'static str,
    pub sub: Substance,
    pub len: f64,
    pub thk: f64,
    pub at: f64,
    pub modes: Vec<Mode>,
    /// Hertz contact time scale (s): modes above ~1/τ are excited less.
    pub contact: f64,
    /// Pitch drop at the onset (ratio, 1 = none) and its time constant.
    pub chirp: (f64, f64),
    /// A body resonance under the struck object (Hz, decay 1/s, level), 0 = none.
    pub body: (f64, f64, f64),
    pub level: f64,
}

impl Voice {
    pub fn bar(label: &'static str, sub: Substance, len: f64, thk: f64, at: f64, count: usize) -> Voice {
        let c = (sub.e / sub.rho).sqrt();
        let k = c * thk / 12f64.sqrt();
        let q = 1.0 / sub.loss;
        let modes = (0..count)
            .map(|n| {
                let b = beta_l(n);
                let hz = b * b / (2.0 * PI * len * len) * k;
                Mode { hz, decay: PI * hz / q, amp: mode_shape(b, at) / 2.0 / (1.0 + n as f64 * 0.6) }
            })
            .filter(|m| m.hz < 20_000.0)
            .collect();
        Voice { label, sub, len, thk, at, modes, contact: 0.0, chirp: (1.0, 0.0), body: (0.0, 0.0, 0.0), level: 1.0 }
    }

    pub fn fundamental(&self) -> f64 {
        self.modes.first().map_or(0.0, |m| m.hz)
    }

    /// Render one strike at `vel ∈ (0,1]` into `out`, starting at sample `at`.
    pub fn strike(&self, out: &mut [[f64; 2]], at: usize, vel: f64, pan: [f64; 2], sr: f64, rng: &mut crate::dsp::Rng) {
        // A harder strike is a shorter contact: brighter.
        let tau = self.contact * (1.3 - 0.5 * vel);
        let longest = self.modes.iter().map(|m| m.decay).fold(f64::INFINITY, f64::min).max(0.7);
        let len = ((7.0 / longest).min(4.0) * sr) as usize;
        let end = (at + len).min(out.len());
        let phases: Vec<f64> = self.modes.iter().map(|_| rng.uniform() * 0.3).collect();
        let weights: Vec<f64> = self
            .modes
            .iter()
            .map(|m| m.amp / (1.0 + (m.hz * tau * 2.0).powi(2)))
            .collect();
        let mut ph = phases.clone();
        let mut click = crate::dsp::Svf::default();
        let tail = len as f64 / sr;
        for (i, s) in out[at..end].iter_mut().enumerate() {
            let t = i as f64 / sr;
            let bend = 1.0 + (self.chirp.0 - 1.0) * (-t / self.chirp.1.max(1e-6)).exp();
            let mut y = 0.0;
            for ((m, w), p) in self.modes.iter().zip(&weights).zip(ph.iter_mut()) {
                *p += 2.0 * PI * m.hz * bend / sr;
                y += w * p.sin() * (-m.decay * t).exp();
            }
            if self.body.2 > 0.0 {
                y += self.body.2 * (2.0 * PI * self.body.0 * t * (1.0 + (bend - 1.0) * 0.5)).sin() * (-self.body.1 * t).exp();
            }
            // Contact noise: a millisecond of filtered noise, the hit itself.
            let n = click.tick(rng.bipolar(), 1.0 / tau.max(2e-5), 0.7, sr).band;
            y += 0.5 * n * (-t / 0.0015).exp();
            // Fade the last 5% so a truncated ring does not click.
            let fade = ((tail - t) / (0.05 * tail)).clamp(0.0, 1.0);
            let v = y * vel * self.level * fade;
            s[0] += v * pan[0];
            s[1] += v * pan[1];
        }
    }
}

/// The kit. Geometry is chosen per role; the frequencies follow from it.
pub struct Kit {
    pub kick: Voice,
    pub stone: Voice,
    pub wood: Voice,
    pub glass: Voice,
    pub alu: Voice,
    pub brass: Voice,
    pub gold: Voice,
}

impl Kit {
    pub fn new() -> Kit {
        // Skin: a 70 × 20 mm pad of dermis. Its own loss (η = 0.3) kills it in
        // a cycle, which is the thud; the body under it is the kick's cavity,
        // tuned to the pad's fundamental, and the onset chirp is the pad
        // stiffening under the strike (a musical liberty, not kosm's).
        let mut kick = Voice::bar("kick: skin pad 70x20 mm", SKIN, 0.070, 0.020, 0.5, 3);
        kick.contact = 4e-4;
        kick.chirp = (2.6, 0.03);
        kick.body = (kick.fundamental() * 0.62, 9.0, 1.4);
        kick.level = 1.1;
        // Limestone: a 1.2 m × 80 mm slab, struck near the end — a low knock.
        let mut stone = Voice::bar("stone: limestone slab 1200x80 mm", LIMESTONE, 1.2, 0.080, 0.1, 6);
        stone.contact = 2e-4;
        stone.level = 0.55;
        // Oak: a 250 × 20 mm plank, the classroom's woodblock clap.
        let mut wood = Voice::bar("clap: oak plank 250x20 mm", OAK, 0.25, 0.020, 0.15, 6);
        wood.contact = 6e-5;
        wood.level = 0.75;
        // Glass: a 50 × 4 mm rod, the closed hat.
        let mut glass = Voice::bar("hat: glass rod 50x4 mm", GLASS, 0.050, 0.004, 0.08, 4);
        glass.contact = 1.5e-5;
        glass.level = 0.32;
        // Anodised aluminium: 120 × 3 mm strip, the off-beat ping.
        let mut alu = Voice::bar("ping: anodised Al strip 120x3 mm", ANODISED_AL, 0.12, 0.003, 0.1, 6);
        alu.contact = 3e-5;
        alu.level = 0.22;
        // Brass: a 300 × 10 mm bar, the crash-bell on downbeats of a drop.
        let mut brass = Voice::bar("bell: brass bar 300x10 mm", BRASS, 0.30, 0.010, 0.05, 7);
        brass.contact = 5e-5;
        brass.level = 0.35;
        // Gold: a 200 × 8 mm bar of gold, long-ringing, for the break and the end.
        let mut gold = Voice::bar("bell: gold bar 200x8 mm", GOLD, 0.20, 0.008, 0.05, 7);
        gold.contact = 8e-5;
        gold.level = 0.3;
        Kit { kick, stone, wood, glass, alu, brass, gold }
    }

    pub fn voices(&self) -> [&Voice; 7] {
        [&self.kick, &self.stone, &self.wood, &self.glass, &self.alu, &self.brass, &self.gold]
    }
}
