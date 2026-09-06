//! The sample pattern: where a pixel's random numbers come from.
//!
//! The path tracer draws every random number through one function, keyed by
//! the pixel, the frame and a *dimension* — which draw along the path it is
//! (the first bounce's light pick, the second bounce's lobe choice, ...).
//! [`SamplePattern`] is what that function does with the key.
//!
//! * [`SamplePattern::White`] is a PCG hash of all three: white noise in
//!   space and in time. Every pixel is independent of its neighbours and of
//!   itself last frame, so the error a temporal filter has to hide is
//!   uniform across all frequencies — and the low-frequency part of it, the
//!   blotches a few pixels wide, is exactly what an à-trous filter cannot
//!   tell from signal and what a short temporal history shows as grain
//!   crawling frame to frame.
//!
//! * [`SamplePattern::BlueNoise`] is Heitz et al. 2019, *A Low-Discrepancy
//!   Sampler that Distributes Monte Carlo Errors as a Blue Noise in Screen
//!   Space*, in the form game renderers ship it: one Owen-scrambled Sobol
//!   sequence shared by every pixel, indexed by the frame, and shifted per
//!   pixel by a tileable blue-noise mask. Within a frame two neighbouring
//!   pixels evaluate the integrand at points that differ by the *difference*
//!   of their mask values, and the mask is blue, so the error between
//!   neighbours is high-frequency: the same total error as white noise,
//!   moved to the frequencies a filter removes. Across frames the sequence
//!   advances, so a pixel's samples over time are a low-discrepancy
//!   sequence — better than independent draws for the running average — and
//!   consecutive frames are decorrelated by the scramble.
//!
//! # The construction
//!
//! Sobol dimensions 0 and 1 only — the van der Corput sequence and its
//! partner, whose direction numbers are known exactly — *padded* over
//! dimension pairs as Burley 2020 (*Practical Hash-based Owen Scrambling*)
//! and PBRT-v4's `PaddedSobolSampler` do: pair *k* of the path draws the
//! (0, 2)-sequence at an index that has been Owen-shuffled by a hash of *k*,
//! and its two values are Owen-scrambled by a hash of *k* and the component.
//! Every pair is then a proper (0, 2)-sequence in its own right, and the
//! pairs are decorrelated from one another. There is no direction-number
//! table to get wrong, and no cap on the number of dimensions.
//!
//! The mask is `blue_noise_128.bin`: a 128×128 tile of 2D shift vectors,
//! annealed by `examples/blue_noise_mask.rs` so that the *vector* field is
//! blue. Every pair at a pixel is shifted by that pixel's one vector.
//!
//! Heitz's paper stores a per-pixel *ranking* key as well and permutes the
//! sample index with it; that key is optimised against the mask, and the
//! optimiser is the bulk of the paper. The Cranley-Patterson shift here is
//! the earlier, simpler Georgiev & Fajardo 2016 form of the same idea, which
//! needs no optimisation and measures blue on the furnace test in
//! `tests/sampler_spectrum.rs`. It is also why the frame does *not* rotate
//! the mask: a per-frame golden-ratio shift on top of the advancing index
//! would decorrelate consecutive frames a second time, at the cost of the
//! stratification of the pixel's samples over the history window. The
//! scramble already decorrelates them.
//!
//! The GPU runs the same construction in `gpu/shaders/sampler.wgsl`, bit for
//! bit; `tests/gpu_sampler.rs` holds them to it.

/// Which sample pattern the path tracer draws its random numbers from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SamplePattern {
    /// A hash of pixel, frame and dimension: white noise.
    White,
    /// Owen-scrambled Sobol, shifted per pixel by a blue-noise mask.
    ///
    /// The default, on the court's numbers (`kosm-view --denoise-eval
    /// --width 640 --at 0.4`): the à-trous history moved 1.73 codes a frame
    /// on the back wall against 1.78 under white noise, and the RMSE against
    /// a 256-pass reference came down in every region — the whole frame
    /// from 12.65 to 12.12 codes, the hoop and backboard from 18.69 to
    /// 17.62. `kosm-view --sampler white` is the old pattern.
    #[default]
    BlueNoise,
}

impl SamplePattern {
    /// Parse a command-line spelling: `white` or `blue` (`blue-noise`,
    /// `bluenoise`, `blue_noise` are accepted too).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "white" => Some(Self::White),
            "blue" | "blue-noise" | "bluenoise" | "blue_noise" => Some(Self::BlueNoise),
            _ => None,
        }
    }

    /// One uniform number in `[0, 1)` for this pixel, frame and dimension.
    pub fn sample(self, pixel: [u32; 2], frame: u32, dim: u32) -> f32 {
        match self {
            Self::White => white(pixel, frame, dim),
            Self::BlueNoise => blue_noise(pixel, frame, dim),
        }
    }
}

/// The mask's side, in texels.
pub const MASK_SIZE: u32 = 128;

/// The blue-noise mask: `MASK_SIZE`² texels, row-major, two bytes each — the
/// shift vector's `u` and `v` in 256ths — baked by
/// `examples/blue_noise_mask.rs`.
pub const MASK: &[u8; (MASK_SIZE * MASK_SIZE * 2) as usize] = include_bytes!("blue_noise_128.bin");

/// The mask's shift vector at `(x, y)`, on the torus, in 256ths.
pub fn mask_at(x: u32, y: u32) -> [u8; 2] {
    let i = (((y % MASK_SIZE) * MASK_SIZE + (x % MASK_SIZE)) * 2) as usize;
    [MASK[i], MASK[i + 1]]
}

/// The PCG hash the shader has always used, as `rand_uniform` there computes
/// it: `dim` is the shader's `sample_idx` salt.
pub fn white(pixel: [u32; 2], frame: u32, dim: u32) -> f32 {
    let mut state = pixel[0]
        .wrapping_mul(1973)
        .wrapping_add(pixel[1].wrapping_mul(9277))
        .wrapping_add(dim.wrapping_mul(26699))
        .wrapping_add(frame.wrapping_mul(12345))
        .wrapping_add(1);
    state = state.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277803737);
    let r = (word >> 22) ^ word;
    r as f32 / 4294967296.0
}

/// The blue-noise pattern at `(pixel, frame, dim)`.
///
/// The frame is the sample index. Frame indices start at 1 in the renderer
/// (0 is "no frame yet"), which is harmless: an Owen-shuffled index is a
/// bijection, and any run of consecutive indices is as well-stratified as
/// any other of the same length.
pub fn blue_noise(pixel: [u32; 2], frame: u32, dim: u32) -> f32 {
    let pair = dim >> 1;
    let comp = dim & 1;
    // The pair's own shuffle of the index: what makes pair 3 a different
    // (0, 2)-sequence from pair 2 rather than a copy of it.
    let index = nested_uniform_scramble(frame, hash(pair.wrapping_add(0x9E37_79B9)));
    let raw = if comp == 0 {
        sobol_dim0(index)
    } else {
        sobol_dim1(index)
    };
    let scrambled = nested_uniform_scramble(raw, hash(dim.wrapping_add(0x7F4A_7C15)));
    // Top 24 bits, so the value is exact in f32 and the two tiers agree
    // bit for bit.
    let value = (scrambled >> 8) as f32 * (1.0 / 16777216.0);
    // The blue-noise shift: the same vector for every pair at this pixel,
    // `u` for the pair's first component and `v` for its second. One vector
    // and not one per pair, so the path's error as a whole is a function of
    // one blue field; see the mask's bake for why the two components must be
    // optimised together.
    let shift = mask_at(pixel[0], pixel[1])[comp as usize] as f32 * (1.0 / 256.0);
    fract(value + shift)
}

fn fract(x: f32) -> f32 {
    x - x.floor()
}

/// Sobol dimension 0: the van der Corput sequence, the index bit-reversed.
pub fn sobol_dim0(index: u32) -> u32 {
    index.reverse_bits()
}

/// Sobol dimension 1: the primitive polynomial `x + 1`, whose direction
/// numbers are `m_k = m_{k-1} ^ (m_{k-1} << 1)` from `m_1 = 1`.
pub fn sobol_dim1(index: u32) -> u32 {
    let mut x = 0u32;
    let mut i = index;
    let mut k = 0;
    while i != 0 {
        if i & 1 != 0 {
            x ^= SOBOL_DIM1[k];
        }
        i >>= 1;
        k += 1;
    }
    x
}

/// Dimension 1's direction numbers, `m_k << (32 - k)`.
pub const SOBOL_DIM1: [u32; 32] = {
    let mut v = [0u32; 32];
    let mut m: u32 = 1;
    let mut k = 0;
    while k < 32 {
        v[k] = m << (31 - k);
        m ^= m << 1;
        k += 1;
    }
    v
};

/// A hash-based Owen scramble of the base-2 digits of `x`, most significant
/// digit first, keyed by `seed`: Burley 2020's nested uniform scramble in the
/// form PBRT-v4 ships as `FastOwenScrambler`.
///
/// Reversed into a domain where the first digit is the lowest bit, every
/// step — an XOR with an even multiple of `x`, an add, a multiply by an odd
/// constant — sets each bit from the bits below it and is invertible, which
/// is exactly what an Owen scramble is: a digit flipped or not according to
/// the digits before it.
pub fn nested_uniform_scramble(x: u32, seed: u32) -> u32 {
    let mut v = x.reverse_bits();
    v ^= v.wrapping_mul(0x3d20adea);
    v = v.wrapping_add(seed);
    v = v.wrapping_mul((seed >> 16) | 1);
    v ^= v.wrapping_mul(0x05526c56);
    v ^= v.wrapping_mul(0x53a22864);
    v.reverse_bits()
}

/// A 32-bit integer hash (lowbias32), for the per-pair and per-dimension
/// seeds.
pub fn hash(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dim1_direction_numbers_are_the_classic_ones() {
        // m_k = 1, 3, 5, 15, 17, 51, 85, 255, ...
        let m: Vec<u32> = (0..8).map(|k| SOBOL_DIM1[k] >> (31 - k)).collect();
        assert_eq!(m, [1, 3, 5, 15, 17, 51, 85, 255]);
    }

    /// The first 2^m points of (dim0, dim1), scrambled or not, put exactly one
    /// point in every elementary interval of area 2^-m.
    #[test]
    fn scrambled_pair_is_a_zero_two_sequence() {
        for seed in [0u32, 1, 1234, 0xDEAD_BEEF] {
            for m in 1..=8u32 {
                let n = 1u32 << m;
                for a in 0..=m {
                    let b = m - a;
                    let mut cells = vec![0u32; n as usize];
                    for i in 0..n {
                        let idx = if seed == 0 {
                            i
                        } else {
                            nested_uniform_scramble(i, seed)
                        };
                        let mut x = sobol_dim0(idx);
                        let mut y = sobol_dim1(idx);
                        if seed != 0 {
                            x = nested_uniform_scramble(x, hash(seed));
                            y = nested_uniform_scramble(y, hash(seed ^ 1));
                        }
                        let cx = if a == 0 { 0 } else { x >> (32 - a) };
                        let cy = if b == 0 { 0 } else { y >> (32 - b) };
                        cells[((cx << b) | cy) as usize] += 1;
                    }
                    assert!(
                        cells.iter().all(|&c| c == 1),
                        "seed {seed}, m {m}, split ({a}, {b}) is not a net"
                    );
                }
            }
        }
    }

    #[test]
    fn white_matches_the_shader_hash_for_a_known_input() {
        // The shader's own formula, worked once by hand for (0, 0, 0, 0):
        // state = 1 → 1*747796405 + 2891336453 = 3639132858 (mod 2^32).
        let mut state: u32 = 1u32.wrapping_mul(747796405).wrapping_add(2891336453);
        let word = ((state >> ((state >> 28).wrapping_add(4))) ^ state).wrapping_mul(277803737);
        state = (word >> 22) ^ word;
        assert_eq!(white([0, 0], 0, 0), state as f32 / 4294967296.0);
    }

    #[test]
    fn blue_noise_is_in_the_unit_interval_and_neighbours_differ() {
        let a = blue_noise([10, 10], 1, 0);
        let b = blue_noise([11, 10], 1, 0);
        assert!((0.0..1.0).contains(&a) && (0.0..1.0).contains(&b));
        assert_ne!(a, b);
    }

    #[test]
    fn mask_values_are_stratified() {
        // 16384 vectors from a (0, 2)-sequence: 64 per 16×16 cell.
        let mut cells = [0u32; 256];
        for t in MASK.chunks_exact(2) {
            cells[((t[0] >> 4) as usize) * 16 + (t[1] >> 4) as usize] += 1;
        }
        assert!(cells.iter().all(|&c| c == 64), "{cells:?}");
    }
}
