//! The deterministic PRNG and the low-discrepancy camera sequence.

#[allow(unused_imports)]
use super::*;

// ─── rng ──────────────────────────────────────────────────────────────────

/// Small, fast, deterministic PRNG (PCG-XSH-RR style).
#[derive(Clone, Copy)]
pub(crate) struct Rng(u64);

impl Rng {
    #[inline]
    pub(crate) fn new(seed: u64) -> Self {
        // Mix so neighbouring pixel seeds decorrelate immediately.
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        s ^= s >> 29;
        s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        s ^= s >> 32;
        Rng(s | 1)
    }

    #[inline]
    pub(crate) fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = (((self.0 >> 18) ^ self.0) >> 27) as u32;
        let rot = (self.0 >> 59) as u32;
        x.rotate_right(rot)
    }

    /// Uniform in [0, 1).
    #[inline]
    pub(crate) fn f64(&mut self) -> f64 {
        (self.next_u32() as f64) * (1.0 / 4294967296.0)
    }
}

// ─── low-discrepancy sampling ─────────────────────────────────────────────

/// Van der Corput radical inverse of `i` in `BASE`.
///
/// Reflects `i`'s digits in `BASE` about the radix point, which spreads
/// consecutive indices as far apart as the base allows. Successive prime
/// bases give the Halton sequence, which is what the camera dimensions use.
#[inline]
pub(crate) fn radical_inverse<const BASE: u32>(mut i: u64) -> f64 {
    let inv_base = 1.0 / BASE as f64;
    let mut inv_bn = 1.0;
    let mut acc = 0u64;
    // Accumulate the reversed digits as an integer, then scale once: doing
    // the division per digit accumulates rounding error over ~50 digits.
    while i > 0 {
        let digit = i % BASE as u64;
        acc = acc * BASE as u64 + digit;
        i /= BASE as u64;
        inv_bn *= inv_base;
    }
    (acc as f64 * inv_bn).min(1.0 - f64::EPSILON)
}

/// Cranley-Patterson rotation: shift `x` by `offset` on the unit torus.
///
/// Preserves the point set's discrepancy while randomising its absolute
/// placement, which is what lets every pixel share one low-discrepancy set
/// without the shared structure showing up as a visible pattern.
#[inline]
pub(crate) fn cp_rotate(x: f64, offset: f64) -> f64 {
    let v = x + offset;
    if v >= 1.0 { v - 1.0 } else { v }
}

/// Samples traced between convergence checks.
///
/// The check needs a sample variance to be worth anything, so it cannot run
/// after every sample; 16 gives a usable estimate and is fine enough that a
/// converged pixel wastes at most 15 samples past the line.
pub(crate) const ADAPTIVE_BATCH: u32 = 16;

/// Minimum samples every pixel gets, whatever the variance estimate says.
///
/// A pixel that happens to draw several near-equal samples early reports a
/// tiny variance and would quit while genuinely unconverged — the classic
/// adaptive-sampling failure, and it shows up as blotching in exactly the
/// smooth regions adaptivity was meant to speed up.
pub(crate) const ADAPTIVE_FLOOR: u32 = 32;

/// Relative tolerance on the 95% confidence half-width of pixel luminance.
pub(crate) const ADAPTIVE_TOL: f32 = 0.10;

/// Absolute luminance added to the mean before applying [`ADAPTIVE_TOL`].
///
/// Pure relative error never converges in shadow, where the mean approaches
/// zero; pure absolute error over-samples highlights. Adding the two is the
/// usual compromise.
pub(crate) const ADAPTIVE_LUM_FLOOR: f32 = 0.02;
