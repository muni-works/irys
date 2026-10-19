//! Small, deterministic DSP: a seeded noise source, filters, a reverb, a limiter.

use std::f64::consts::PI;

/// xorshift64* — the only randomness in the soundtrack, seeded from the arrangement.
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in [-1, 1).
    pub fn bipolar(&mut self) -> f64 {
        2.0 * self.uniform() - 1.0
    }
}

/// Chamberlin state-variable filter (resonant, cheap, stable below ~sr/6).
#[derive(Clone, Default)]
pub struct Svf {
    low: f64,
    band: f64,
}

#[allow(dead_code)]
pub struct SvfOut {
    pub low: f64,
    pub band: f64,
    pub high: f64,
}

impl Svf {
    pub fn tick(&mut self, x: f64, hz: f64, q: f64, sr: f64) -> SvfOut {
        let f = 2.0 * (PI * hz.min(sr / 6.5) / sr).sin();
        let damp = 1.0 / q.max(0.5);
        let high = x - self.low - damp * self.band;
        self.band += f * high;
        self.low += f * self.band;
        SvfOut { low: self.low, band: self.band, high }
    }
}

/// One-pole low-pass.
#[derive(Clone, Default)]
pub struct OnePole(f64);

impl OnePole {
    pub fn tick(&mut self, x: f64, hz: f64, sr: f64) -> f64 {
        let a = 1.0 - (-2.0 * PI * hz / sr).exp();
        self.0 += a * (x - self.0);
        self.0
    }
}

struct Comb {
    buf: Vec<f64>,
    i: usize,
    store: f64,
}

impl Comb {
    fn tick(&mut self, x: f64, fb: f64, damp: f64) -> f64 {
        let y = self.buf[self.i];
        self.store = y * (1.0 - damp) + self.store * damp;
        self.buf[self.i] = x + self.store * fb;
        self.i = (self.i + 1) % self.buf.len();
        y
    }
}

struct Allpass {
    buf: Vec<f64>,
    i: usize,
}

impl Allpass {
    fn tick(&mut self, x: f64) -> f64 {
        let b = self.buf[self.i];
        let y = b - x;
        self.buf[self.i] = x + b * 0.5;
        self.i = (self.i + 1) % self.buf.len();
        y
    }
}

/// Freeverb-shaped stereo reverb: eight combs and four allpasses per side, the
/// right side's delays spread by 23 samples so the tail is decorrelated.
pub struct Reverb {
    combs: [Vec<Comb>; 2],
    aps: [Vec<Allpass>; 2],
    pub feedback: f64,
    pub damp: f64,
}

impl Reverb {
    pub fn new(sr: f64) -> Self {
        let scale = sr / 44_100.0;
        let comb_len = [1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617];
        let ap_len = [556, 441, 341, 225];
        let mk = |spread: usize| {
            let c = comb_len
                .iter()
                .map(|&n| Comb { buf: vec![0.0; ((n + spread) as f64 * scale) as usize], i: 0, store: 0.0 })
                .collect();
            let a = ap_len
                .iter()
                .map(|&n| Allpass { buf: vec![0.0; ((n + spread) as f64 * scale) as usize], i: 0 })
                .collect();
            (c, a)
        };
        let (c0, a0) = mk(0);
        let (c1, a1) = mk(23);
        Reverb { combs: [c0, c1], aps: [a0, a1], feedback: 0.86, damp: 0.3 }
    }

    pub fn tick(&mut self, x: f64) -> [f64; 2] {
        let mut out = [0.0; 2];
        for ch in 0..2 {
            let mut s = 0.0;
            for c in &mut self.combs[ch] {
                s += c.tick(x * 0.015, self.feedback, self.damp);
            }
            for a in &mut self.aps[ch] {
                s = a.tick(s);
            }
            out[ch] = s;
        }
        out
    }
}

/// Equal-power pan, `p` in [-1, 1].
pub fn pan(p: f64) -> [f64; 2] {
    let a = (p.clamp(-1.0, 1.0) + 1.0) * PI / 4.0;
    [a.cos(), a.sin()]
}

/// MIDI note to Hz (A4 = 69 = 440 Hz).
pub fn midi_hz(m: f64) -> f64 {
    440.0 * 2f64.powf((m - 69.0) / 12.0)
}

/// Peak-normalise to `ceiling` after a gentle tanh soft clip.
pub fn master(buf: &mut [[f64; 2]], drive: f64, ceiling: f64) {
    for s in buf.iter_mut() {
        for c in s.iter_mut() {
            *c = (*c * drive).tanh();
        }
    }
    let peak = buf.iter().flat_map(|s| s.iter()).fold(0.0f64, |m, &v| m.max(v.abs()));
    if peak > 0.0 {
        let g = ceiling / peak;
        for s in buf.iter_mut() {
            s[0] *= g;
            s[1] *= g;
        }
    }
}
