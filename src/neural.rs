//! A learned denoiser: the network, its weights, and the reference forward.
//!
//! [`crate::pathtrace::denoise`] and its device twin in
//! [`crate::gpu::history`] are one filter with five hand-chosen constants,
//! and they have to be right for every scene anyone points the renderer at.
//! This is the other option: a network small enough to run as one compute
//! pass, fitted offline to reference renders of the *particular* scene it
//! will be shown. Nothing here trains anything — the fit lives in the game,
//! where the level and its ground truth are — and nothing here knows what
//! scene it was fitted to. What it knows is the shape of the network and how
//! to evaluate it.
//!
//! # The network
//!
//! Kernel-predicting, in the sense of Bako et al.: three 3x3 convolutions
//! over per-pixel features, a softmax over the 25 outputs, and those 25
//! numbers used as the weights of a 5x5 average over the frame's own
//! demodulated illumination. It predicts a *filter*, not a picture, so every
//! pixel it produces is a convex combination of light the path tracer
//! actually measured. That is the same promise the à-trous filter makes; the
//! difference is where the weights come from.
//!
//! Deliberately not a U-Net. A U-Net's downsamples are what let it invent
//! plausible structure, which is exactly the failure a real-time renderer
//! cannot tolerate — a hallucinated shadow that flickers as the network
//! changes its mind is worse than the noise it replaced. And three
//! convolutions with no resampling are three dispatches with no mip chain.
//!
//! # Layout
//!
//! [`Weights`] is a flat `f32` blob in one fixed order — `W1 b1 W2 b2 W3 b3`,
//! each `W` as `[out][in][3][3]` — so the same bytes are what
//! [`Weights::forward`] indexes and what the GPU pass uploads into a storage
//! buffer without rearrangement.

use std::io::Read;
use std::path::Path;

/// Input feature planes per pixel; see [`Weights::features_at`].
pub const C_IN: usize = 10;
/// The predicted filter's footprint.
pub const TAPS: usize = 5;
/// Predicted weights per pixel.
pub const K: usize = TAPS * TAPS;
/// Convolution kernel size in each layer.
pub const KS: usize = 3;

/// The albedo floor the demodulation divides by — the same constant
/// [`crate::pathtrace::denoise`] uses, because the two filters have to mean
/// the same thing by "illumination".
pub const DEMOD_FLOOR: f32 = 0.01;

/// Soft normalisation for the depth feature, in the scene's own units.
pub const DEPTH_SCALE: f32 = 3000.0;

const MAGIC: &[u8; 8] = b"KOSMKPN1";

/// A trained network.
#[derive(Debug, Clone)]
pub struct Weights {
    /// Hidden channels in both interior layers.
    pub hidden: usize,
    /// `W1 b1 W2 b2 W3 b3`, flat.
    pub data: Vec<f32>,
}

/// Why a weight file would not load.
#[derive(Debug)]
pub enum WeightsError {
    /// The file could not be read.
    Io(std::io::Error),
    /// The file is not a KPN blob, or is a shape this build cannot evaluate.
    Shape(String),
}

impl std::fmt::Display for WeightsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Shape(s) => write!(f, "{s}"),
        }
    }
}

impl std::error::Error for WeightsError {}

impl From<std::io::Error> for WeightsError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl Weights {
    /// Parse the blob a trainer wrote: an eight-byte magic, `C_IN`, hidden,
    /// `K` and `KS` as `u32`, then every `f32`.
    ///
    /// The four shape words are checked rather than trusted. A file with the
    /// wrong feature count would otherwise run — every index would be in
    /// bounds — and produce a picture that is subtly wrong everywhere, which
    /// is the hardest kind of wrong to find.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, WeightsError> {
        if bytes.len() < 24 || &bytes[0..8] != MAGIC {
            return Err(WeightsError::Shape("not a KPN weight file".into()));
        }
        let word = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
        let (c_in, hidden, k, ks) = (word(8), word(12), word(16), word(20));
        if c_in != C_IN || k != K || ks != KS {
            return Err(WeightsError::Shape(format!(
                "weight file is {c_in}x{k}x{ks}, this build evaluates {C_IN}x{K}x{KS}"
            )));
        }
        if hidden == 0 {
            return Err(WeightsError::Shape("zero hidden channels".into()));
        }
        let data: Vec<f32> = bytes[24..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let me = Self { hidden, data };
        let want = me.expected_len();
        if me.data.len() != want {
            return Err(WeightsError::Shape(format!(
                "weight file holds {} floats, {hidden} hidden channels needs {want}",
                me.data.len()
            )));
        }
        Ok(me)
    }

    /// [`Weights::from_bytes`] on the contents of a file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, WeightsError> {
        let mut b = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut b)?;
        Self::from_bytes(&b)
    }

    fn expected_len(&self) -> usize {
        let h = self.hidden;
        (h * C_IN * KS * KS + h) + (h * h * KS * KS + h) + (K * h * KS * KS + K)
    }

    /// Offsets into [`Weights::data`], in the order the blob stores them:
    /// `[W1, b1, W2, b2, W3, b3]`. The GPU pass wants exactly these six
    /// numbers in its uniform, so they are computed in one place.
    pub fn offsets(&self) -> [u32; 6] {
        let h = self.hidden;
        let mut at = 0u32;
        let push = |n: usize, at: &mut u32| {
            let o = *at;
            *at += n as u32;
            o
        };
        [
            push(h * C_IN * KS * KS, &mut at),
            push(h, &mut at),
            push(h * h * KS * KS, &mut at),
            push(h, &mut at),
            push(K * h * KS * KS, &mut at),
            push(K, &mut at),
        ]
    }

    /// Trainable scalars.
    pub fn parameters(&self) -> usize {
        self.data.len()
    }

    /// One pixel's input features, in the order the shader reads them.
    ///
    /// | plane | what |
    /// |-------|------|
    /// | 0..3  | `ln(1 + mean/albedo)`, the demodulated running mean |
    /// | 3     | `1/sqrt(count)` |
    /// | 4     | `ln(1 + sqrt(variance)/luminance(albedo))` |
    /// | 5..8  | the world normal |
    /// | 8     | `depth / (depth + DEPTH_SCALE)`, zero on background |
    /// | 9     | `luminance(albedo)` |
    ///
    /// The two radiance-like planes are log-compressed so a light panel does
    /// not drown a floor; the count plane is what lets one network serve
    /// every history length, since a pixel on its first sample and one on its
    /// sixteenth want different kernels.
    pub fn features_at(
        mean: [f32; 3],
        variance: f32,
        normal: [f32; 3],
        depth: f32,
        albedo: [f32; 3],
        count: f32,
    ) -> [f32; C_IN] {
        let a = [
            albedo[0].max(DEMOD_FLOOR),
            albedo[1].max(DEMOD_FLOOR),
            albedo[2].max(DEMOD_FLOOR),
        ];
        let la = luminance(a).max(DEMOD_FLOOR);
        [
            (1.0 + mean[0] / a[0]).max(1e-8).ln(),
            (1.0 + mean[1] / a[1]).max(1e-8).ln(),
            (1.0 + mean[2] / a[2]).max(1e-8).ln(),
            1.0 / count.max(1.0).sqrt(),
            (1.0 + (variance.max(0.0) / (la * la)).sqrt()).ln(),
            normal[0],
            normal[1],
            normal[2],
            if depth > 0.0 {
                depth / (depth + DEPTH_SCALE)
            } else {
                0.0
            },
            la,
        ]
    }

    /// Evaluate the network over a whole frame — the reference the WGSL pass
    /// is checked against.
    ///
    /// Every buffer is interleaved and row-major: `mean`, `normal` and
    /// `albedo` are three floats a pixel, `variance` and `depth` one. The
    /// result is filtered *radiance*, remodulated, three floats a pixel.
    ///
    /// This is a plain triple loop and it is not fast; it exists so the
    /// shader has something to be wrong against.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        width: usize,
        height: usize,
        mean: &[f32],
        variance: &[f32],
        normal: &[f32],
        depth: &[f32],
        albedo: &[f32],
        count: f32,
    ) -> Vec<f32> {
        let n = width * height;
        let h = self.hidden;
        let off = self.offsets();

        let mut feat = vec![0.0f32; C_IN * n];
        let mut illum = vec![0.0f32; 3 * n];
        for p in 0..n {
            let m = [mean[p * 3], mean[p * 3 + 1], mean[p * 3 + 2]];
            let al = [albedo[p * 3], albedo[p * 3 + 1], albedo[p * 3 + 2]];
            let f = Self::features_at(
                m,
                variance[p],
                [normal[p * 3], normal[p * 3 + 1], normal[p * 3 + 2]],
                depth[p],
                al,
                count,
            );
            for (c, v) in f.iter().enumerate() {
                feat[c * n + p] = *v;
            }
            for c in 0..3 {
                illum[c * n + p] = m[c] / al[c].max(DEMOD_FLOOR);
            }
        }

        let conv = |src: &[f32], c_in: usize, c_out: usize, w: u32, b: u32, relu: bool| {
            let mut dst = vec![0.0f32; c_out * n];
            for o in 0..c_out {
                for y in 0..height {
                    for x in 0..width {
                        let mut acc = self.data[b as usize + o];
                        for i in 0..c_in {
                            let base = w as usize + (o * c_in + i) * KS * KS;
                            for dy in 0..KS {
                                let sy = clamp(y as i32 + dy as i32 - 1, height);
                                for dx in 0..KS {
                                    let sx = clamp(x as i32 + dx as i32 - 1, width);
                                    acc += self.data[base + dy * KS + dx]
                                        * src[i * n + sy * width + sx];
                                }
                            }
                        }
                        dst[o * n + y * width + x] = if relu { acc.max(0.0) } else { acc };
                    }
                }
            }
            dst
        };

        let a1 = conv(&feat, C_IN, h, off[0], off[1], true);
        let a2 = conv(&a1, h, h, off[2], off[3], true);
        let z3 = conv(&a2, h, K, off[4], off[5], false);

        let mut out = vec![0.0f32; 3 * n];
        let r = (TAPS / 2) as i32;
        for y in 0..height {
            for x in 0..width {
                let p = y * width + x;
                let mut mx = f32::NEG_INFINITY;
                for k in 0..K {
                    mx = mx.max(z3[k * n + p]);
                }
                let mut w = [0.0f32; K];
                let mut sum = 0.0;
                for k in 0..K {
                    w[k] = (z3[k * n + p] - mx).exp();
                    sum += w[k];
                }
                let inv = 1.0 / sum;
                let mut acc = [0.0f32; 3];
                for ky in 0..TAPS {
                    let sy = clamp(y as i32 + ky as i32 - r, height);
                    for kx in 0..TAPS {
                        let sx = clamp(x as i32 + kx as i32 - r, width);
                        let q = sy * width + sx;
                        let wk = w[ky * TAPS + kx] * inv;
                        for c in 0..3 {
                            acc[c] += wk * illum[c * n + q];
                        }
                    }
                }
                for c in 0..3 {
                    out[p * 3 + c] = acc[c] * albedo[p * 3 + c].max(DEMOD_FLOOR);
                }
            }
        }
        out
    }
}

fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

#[inline]
fn clamp(v: i32, hi: usize) -> usize {
    v.clamp(0, hi as i32 - 1) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A blob with the shape header a trainer writes and arbitrary weights.
    pub(crate) fn synthetic(hidden: usize, seed: u64) -> Weights {
        let w = Weights {
            hidden,
            data: Vec::new(),
        };
        let n = w.expected_len();
        let mut s = seed | 1;
        let data = (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / 8388608.0) - 0.5
            })
            .collect();
        Weights { hidden, data }
    }

    fn blob(w: &Weights) -> Vec<u8> {
        let mut b = MAGIC.to_vec();
        for v in [C_IN as u32, w.hidden as u32, K as u32, KS as u32] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        for f in &w.data {
            b.extend_from_slice(&f.to_le_bytes());
        }
        b
    }

    #[test]
    fn a_blob_round_trips_and_a_wrong_shape_is_refused() {
        let w = synthetic(8, 3);
        let back = Weights::from_bytes(&blob(&w)).unwrap();
        assert_eq!(back.hidden, 8);
        assert_eq!(back.data, w.data);

        let mut short = blob(&w);
        short.truncate(short.len() - 8);
        assert!(matches!(
            Weights::from_bytes(&short),
            Err(WeightsError::Shape(_))
        ));

        let mut wrong = blob(&w);
        wrong[8] = 99; // a C_IN this build cannot evaluate
        assert!(matches!(
            Weights::from_bytes(&wrong),
            Err(WeightsError::Shape(_))
        ));
    }

    #[test]
    fn the_output_is_a_convex_combination_of_the_input() {
        // The whole point of predicting a kernel rather than a colour: the
        // filtered illumination cannot leave the neighbourhood's range, so
        // the network cannot invent light.
        let (w, h) = (9usize, 7usize);
        let n = w * h;
        let net = synthetic(6, 11);
        let mut mean = vec![0.0f32; n * 3];
        let mut albedo = vec![0.0f32; n * 3];
        let mut normal = vec![0.0f32; n * 3];
        let variance = vec![0.05f32; n];
        let depth = vec![1500.0f32; n];
        for p in 0..n {
            for c in 0..3 {
                mean[p * 3 + c] = 0.2 + ((p * 7 + c) % 13) as f32 * 0.05;
                albedo[p * 3 + c] = 0.5;
                normal[p * 3 + c] = if c == 2 { 1.0 } else { 0.0 };
            }
        }
        let out = net.forward(w, h, &mean, &variance, &normal, &depth, &albedo, 4.0);
        let lo = mean.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = mean.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for v in out {
            assert!(v >= lo - 1e-4 && v <= hi + 1e-4, "{v} left [{lo}, {hi}]");
        }
    }
}
