//! Phase 3, piece 3: where the samples go before any light transport —
//! a quasi-random sampler and importance-sampled wavelengths.
//!
//! **Sobol, shuffled and Owen-scrambled** (Burley, "Practical Hash-based
//! Owen Scrambling", JCGT 2020 — Cycles' Sobol–Burley pattern). A path's
//! draws come in tuples of up to four dimensions (the camera tuple is
//! `(λ, x, y)`; each vertex has a light tuple and a BSDF tuple), and tuple
//! `t` of sample `i` in pixel `p` is
//!
//! ```text
//! seed = hash(p, t);  j = owen(i, seed);  x_d = owen(sobol_d(j), hash(seed, d + 1))
//! ```
//!
//! Scrambling the *index* with a nested uniform scramble (bit-reversed
//! Laine–Karras) maps every aligned power-of-two block of indices onto
//! another such block, so each tuple's first 2^k samples are still a
//! (0, k, 2)-net in its first two dimensions, while different tuples get
//! independent blocks — Kollig & Keller's padding, without a table.
//! Scrambling the *values* randomises each net. Everything is integer
//! arithmetic, the same on the CPU ([`PathRng`]) and in WGSL, and a draw
//! is a pure function of (pixel, sample, dimension), so replay is exact
//! by construction. A uniform keeps the top 24 bits, as the random stream's
//! does.
//!
//! **Wavelengths by importance** ([`Spectral`]): the hero wavelength's
//! uniform goes through the inverse CDF of `p(λ) ∝ |r̄| + |ḡ| + |b̄|` (the
//! linear-sRGB matching functions irys's film integrates against), and the
//! other three rotate by `k/4` in CDF space, so each is marginally `p` and
//! the set stays stratified. The film divides by `p`. Measured on a white
//! spectrum: per-channel variance of four hero lanes 0.152 → 0.054.

use irys_light::spectrum::{LAMBDA_MAX_NM, LAMBDA_MIN_NM, hero_weight};

/// Bits in a Sobol direction matrix.
pub const BITS: usize = 32;

/// Sobol dimensions a tuple draws from.
pub const DIMS: usize = 4;

/// The direction matrices of Sobol's first four dimensions (Joe & Kuo's
/// primitive polynomials for dimensions 2–4; the first is van der Corput),
/// column `k` the image of index bit `k`.
pub fn sobol_matrices() -> [[u32; BITS]; DIMS] {
    // (degree s, coefficients a, initial m), Joe–Kuo new-joe-kuo-6.21201
    let jk: [(usize, u32, &[u32]); 3] = [(1, 0, &[1]), (2, 1, &[1, 3]), (3, 1, &[1, 3, 1])];
    let mut out = [[0u32; BITS]; DIMS];
    for (k, v) in out[0].iter_mut().enumerate() {
        *v = 1u32 << (31 - k);
    }
    for (d, &(s, a, m)) in jk.iter().enumerate() {
        let v = &mut out[d + 1];
        for k in 0..BITS {
            v[k] = if k < s {
                m[k] << (31 - k)
            } else {
                let mut x = v[k - s] ^ (v[k - s] >> s);
                for j in 1..s {
                    if (a >> (s - 1 - j)) & 1 == 1 {
                        x ^= v[k - j];
                    }
                }
                x
            };
        }
    }
    out
}

static MATRICES: std::sync::LazyLock<[[u32; BITS]; DIMS]> =
    std::sync::LazyLock::new(sobol_matrices);

/// Bytes of an index, and so byte tables per dimension.
pub const BYTES: usize = BITS / 8;

/// The direction matrices of dimensions 1–3 cut into byte tables (Burley
/// 2020, §4): `tables()[((d − 1) · BYTES + k) · 256 + v]` is the XOR of
/// dimension `d`'s columns `8k … 8k + 7` selected by the bits of `v`, so
/// `sobol(i, d)` is four lookups XORed. XOR is exact, so the tables give
/// the column walk's integers bit for bit, here and on the device (which
/// gets this same array as a module constant). Dimension 0 is van der
/// Corput, the index's bits reversed, and needs no table.
pub fn byte_tables() -> Vec<u32> {
    let m = sobol_matrices();
    let mut t = Vec::with_capacity((DIMS - 1) * BYTES * 256);
    for col in &m[1..] {
        for k in 0..BYTES {
            for v in 0..256usize {
                t.push((0..8).filter(|b| (v >> b) & 1 == 1).fold(0u32, |x, b| x ^ col[8 * k + b]));
            }
        }
    }
    t
}

static TABLES: std::sync::LazyLock<Vec<u32>> = std::sync::LazyLock::new(byte_tables);

/// Sobol dimension `d` of index `i`, by the column walk (the reference the
/// byte tables are tested against).
pub fn sobol_walk(i: u32, d: usize) -> u32 {
    let m = &MATRICES[d];
    let (mut x, mut b, mut k) = (0u32, i, 0usize);
    while b != 0 {
        if b & 1 == 1 {
            x ^= m[k];
        }
        b >>= 1;
        k += 1;
    }
    x
}

/// Sobol dimension `d` of index `i`: van der Corput for `d = 0`, four
/// byte-table lookups otherwise.
pub fn sobol(i: u32, d: usize) -> u32 {
    if d == 0 {
        return i.reverse_bits();
    }
    let t = &TABLES[(d - 1) * BYTES * 256..];
    t[(i & 255) as usize]
        ^ t[256 + ((i >> 8) & 255) as usize]
        ^ t[512 + ((i >> 16) & 255) as usize]
        ^ t[768 + (i >> 24) as usize]
}

/// Chris Wellons' `lowbias32`.
pub fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

/// Two words into one.
pub fn hash2(a: u32, b: u32) -> u32 {
    hash(a ^ hash(b).wrapping_add(0x9e37_79b9))
}

fn laine_karras(mut x: u32, seed: u32) -> u32 {
    x = x.wrapping_add(seed);
    x ^= x.wrapping_mul(0x6c50_b47c);
    x ^= x.wrapping_mul(0xb82f_1e52);
    x ^= x.wrapping_mul(0xc7af_e638);
    x ^= x.wrapping_mul(0x8d22_f6e6);
    x
}

/// Burley's nested uniform scramble: an Owen scramble, each bit flipped by
/// a hash of the bits above it.
pub fn owen(x: u32, seed: u32) -> u32 {
    laine_karras(x.reverse_bits(), seed).reverse_bits()
}

/// Dimension `d` (< [`DIMS`]) of tuple `t` of sample `i` in the pixel
/// keyed `pixel`, as 32 bits.
pub fn draw(i: u32, t: u32, d: u32, pixel: u32) -> u32 {
    let seed = hash2(pixel, t);
    owen(sobol(owen(i, seed), d as usize), hash2(seed, d + 1))
}

/// `hash(d + 1) + 0x9e3779b9` for each dimension: the half of
/// `hash2(seed, d + 1)` that does not depend on the seed, so a value
/// scramble costs one hash (`hash(seed ^ VALUE_KEYS[d])`).
pub fn value_keys() -> [u32; DIMS] {
    std::array::from_fn(|d| hash(d as u32 + 1).wrapping_add(0x9e37_79b9))
}

/// A path's generator: SplitMix64 ([`crate::prb::Rng`]) or the scrambled
/// Sobol sequence of its pixel. Tuples start at [`PathRng::align`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PathRng {
    /// The phase 1–2 stream, bit for bit.
    Random(crate::prb::Rng),
    /// Sample `index` of the pixel keyed `pixel`; `counter` is
    /// `4 × tuple + dimension` of the next draw.
    Sobol {
        /// The sample's index in its pixel's sequence.
        index: u32,
        /// The pixel's scramble key.
        pixel: u32,
        /// The next draw.
        counter: u32,
    },
}

impl PathRng {
    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        match self {
            PathRng::Random(r) => r.uniform(),
            PathRng::Sobol {
                index,
                pixel,
                counter,
            } => {
                let c = *counter;
                *counter += 1;
                let x = draw(*index, c >> 2, c & 3, *pixel);
                (x >> 8) as f64 * (1.0 / 16_777_216.0)
            }
        }
    }

    /// Start a new tuple (Sobol); nothing for the random stream.
    pub fn align(&mut self) {
        if let PathRng::Sobol { counter, .. } = self {
            *counter = (*counter + 3) & !3;
        }
    }
}

/// Bins of the wavelength CDF.
pub const SPECTRAL_BINS: usize = 256;

/// The wavelength distribution `p(λ) ∝ |r̄| + |ḡ| + |b̄|`, piecewise
/// constant over [`SPECTRAL_BINS`] bins of the band, as a CDF (`f32`
/// values, so the device's copy is the same numbers).
#[derive(Debug, Clone)]
pub struct Spectral {
    /// `cdf[0] = 0, …, cdf[BINS] = 1`.
    pub cdf: Vec<f32>,
}

impl Spectral {
    /// The table, built once.
    pub fn get() -> &'static Spectral {
        static S: std::sync::LazyLock<Spectral> = std::sync::LazyLock::new(Spectral::build);
        &S
    }

    fn build() -> Self {
        let sub = 64;
        let band = LAMBDA_MAX_NM - LAMBDA_MIN_NM;
        let mass: Vec<f64> = (0..SPECTRAL_BINS)
            .map(|b| {
                (0..sub)
                    .map(|k| {
                        let l = LAMBDA_MIN_NM
                            + band * (b as f64 + (k as f64 + 0.5) / sub as f64)
                                / SPECTRAL_BINS as f64;
                        hero_weight(l).iter().map(|w| (*w as f64).abs()).sum::<f64>()
                    })
                    .sum::<f64>()
                    // a floor, so every wavelength keeps a density
                    .max(1e-3)
            })
            .collect();
        let total: f64 = mass.iter().sum();
        let mut acc = 0.0;
        let mut cdf = vec![0.0f32];
        for m in &mass {
            acc += m;
            cdf.push((acc / total) as f32);
        }
        *cdf.last_mut().unwrap() = 1.0;
        Self { cdf }
    }

    /// The wavelength `u` maps to, nm.
    pub fn lambda(&self, u: f64) -> f64 {
        let c = &self.cdf;
        let b = c[1..SPECTRAL_BINS]
            .partition_point(|&x| (x as f64) <= u)
            .min(SPECTRAL_BINS - 1);
        let (c0, c1) = (c[b] as f64, c[b + 1] as f64);
        let t = ((u - c0) / (c1 - c0)).clamp(0.0, 1.0);
        LAMBDA_MIN_NM + (LAMBDA_MAX_NM - LAMBDA_MIN_NM) * (b as f64 + t) / SPECTRAL_BINS as f64
    }

    /// The density at `lambda_nm`, relative to uniform over the band.
    pub fn pdf(&self, lambda_nm: f64) -> f64 {
        let x = (lambda_nm - LAMBDA_MIN_NM) / (LAMBDA_MAX_NM - LAMBDA_MIN_NM);
        let b = ((x * SPECTRAL_BINS as f64) as usize).min(SPECTRAL_BINS - 1);
        (self.cdf[b + 1] - self.cdf[b]) as f64 * SPECTRAL_BINS as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first 2^k points of every tuple are a (0, k, 2)-net in the
    /// tuple's first two dimensions, for every pixel key and tuple.
    #[test]
    fn shuffled_prefixes_are_nets() {
        for pixel in [0u32, 7, 123_456] {
            for t in [0u32, 1, 9] {
                for k in [4u32, 6] {
                    let n = 1u32 << k;
                    // every elementary interval of the (2^a × 2^(k−a)) grids holds one point
                    for a in 0..=k {
                        let mut seen = std::collections::HashSet::new();
                        for i in 0..n {
                            let top = |v: u32, bits: u32| if bits == 0 { 0 } else { v >> (32 - bits) };
                            let x = top(draw(i, t, 0, pixel), a);
                            let y = top(draw(i, t, 1, pixel), k - a);
                            assert!(seen.insert((x, y)), "pixel {pixel} tuple {t} k {k} a {a}");
                        }
                    }
                }
            }
        }
    }

    /// The byte tables are the column walk, bit for bit, and the hoisted
    /// value key is `hash2`'s.
    #[test]
    fn byte_tables_are_the_walk() {
        let mut x = 0x1234_5678u32;
        let mut idx: Vec<u32> = (0..4096).collect();
        idx.extend([u32::MAX, 1 << 31, 0xdead_beef]);
        for _ in 0..200_000 {
            x = hash(x);
            idx.push(x);
        }
        for &i in &idx {
            for d in 0..DIMS {
                assert_eq!(sobol(i, d), sobol_walk(i, d), "i {i} d {d}");
            }
        }
        let k = value_keys();
        for seed in [0u32, 1, 0x9e37_79b9, 0xffff_ffff, 123_456] {
            for d in 0..DIMS as u32 {
                assert_eq!(hash(seed ^ k[d as usize]), hash2(seed, d + 1));
            }
        }
    }

    /// White stays white: `E[w(λ) / p(λ)] = 1` per channel under `p`.
    #[test]
    fn spectral_table_keeps_white() {
        let s = Spectral::get();
        let n = 200_000;
        let mut m = [0.0f64; 3];
        for i in 0..n {
            let l = s.lambda((i as f64 + 0.5) / n as f64);
            let w = hero_weight(l);
            for c in 0..3 {
                m[c] += w[c] as f64 / s.pdf(l) / n as f64;
            }
        }
        for c in m {
            assert!((c - 1.0).abs() < 2e-3, "white maps to {m:?}");
        }
    }
}
