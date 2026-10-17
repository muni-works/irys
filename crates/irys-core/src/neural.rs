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
//! demodulated illumination, after a firefly [`Veto`] has gated and clamped
//! the taps that are outliers against their own neighbourhood. It predicts a
//! *filter*, not a picture, so every pixel it produces stays inside the range
//! of light the path tracer actually measured nearby. That is the same promise the à-trous filter makes; the
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
//! [`Weights`] is a flat `f32` blob in one fixed order — `W1 b1 W2 b2 W3 b3`
//! then the three [`Veto`] scalars, each `W` as `[out][in][3][3]` — so the
//! same bytes are what [`Weights::forward`] indexes and what the GPU pass
//! uploads into a storage buffer without rearrangement.

use std::io::Read;
use std::path::Path;

/// Input feature planes per pixel; see [`Weights::features_at`].
pub const C_IN: usize = 11;
/// The predicted filter's footprint.
pub const TAPS: usize = 5;
/// Predicted weights per pixel.
pub const K: usize = TAPS * TAPS;
/// Convolution kernel size in each layer.
pub const KS: usize = 3;
/// Trailing scalars after the three layers: the firefly veto's shape.
///
/// `[0]` is the gate's steepness, `[1]` its threshold and `[2]` the clamp's
/// headroom, all *pre-activation* — see [`Veto::from_raw`].
pub const VETO: usize = 3;

/// The albedo floor the demodulation divides by — the same constant
/// [`crate::pathtrace::denoise`] uses, because the two filters have to mean
/// the same thing by "illumination".
pub const DEMOD_FLOOR: f32 = 0.01;

/// Soft normalisation for the depth feature, in the scene's own units.
pub const DEPTH_SCALE: f32 = 3000.0;

/// Fold a biased hit id into a feature the convolutions can find edges in.
///
/// The id is a *label*, not a quantity: id 7 is not between id 6 and id 8 in
/// any sense the network should be allowed to interpolate over. What the
/// network actually needs from it is one question — "is my neighbour the same
/// surface as me?" — and a hash answers exactly that: identical ids give
/// identical values, different ids give values that differ, and a 3x3
/// convolution reads the difference as an edge. Feeding the raw id instead
/// would invite the net to learn that high-numbered materials are shiny.
///
/// Zero — the background sentinel — is kept at zero rather than hashed, so
/// "nothing here" is a value and not an arbitrary point in the range.
pub fn id_feature(id: f32) -> f32 {
    if id <= 0.0 {
        return 0.0;
    }
    let mut h = (id as u32).wrapping_mul(0x9E37_79B9);
    h ^= h >> 15;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    // [0, 1), and never exactly 0 for a real id, so background stays distinct
    (h >> 8) as f32 / 16_777_216.0
}

const MAGIC: &[u8; 8] = b"KOSMKPN2";

/// The gate's floor: how much of a vetoed tap survives.
///
/// Not zero, and that is the point. If every tap in a neighbourhood is a
/// bright outlier against the other twenty-four — a small, genuinely bright
/// thing filling the whole 5x5 — then every gate closes at once, and a hard
/// zero would leave the softmax with nothing to normalise. At a floor the
/// gates all collapse to the same small number, renormalise back to the plain
/// softmax, and the filter degrades into the one it was before the veto.
pub const VETO_FLOOR: f32 = 1e-3;

/// The luminance offset that keeps the log and the ratio finite in the dark.
pub const VETO_EPS: f32 = 1e-4;

/// The firefly veto's three scalars, after their activations.
///
/// # What the veto is for
///
/// Demodulation divides the running mean by the albedo, and on the
/// backboard's glass the albedo sits at [`DEMOD_FLOOR`] — so a single stray
/// path that landed a hundred times the neighbourhood's radiance comes out of
/// that divide a hundred times brighter still. A softmax over 25 taps cannot
/// throw such a tap away: its weights are strictly positive, so the best it
/// can do is make the firefly a small fraction of a very large number. The
/// à-trous filter has no such trouble, because its luminance edge-stop is an
/// exponential that reaches zero — and that is most of why the hand-tuned
/// filter still beat the network on the glass and on the net.
///
/// So the kernel gets a veto the softmax cannot undo, in two parts:
///
/// - a **soft gate**, `g = FLOOR + (1 - FLOOR)·σ(-scale·(ln ratio - thresh))`,
///   multiplied into each tap's softmax weight before renormalisation. The
///   ratio is the tap's luminance over a *leave-one-out* mean of the other
///   twenty-four, so a tap is judged against a neighbourhood it is not itself
///   inflating. Multiplying a softmax weight by a gate is adding `ln g` to the
///   logit, which is exactly the unbounded-below term the softmax lacked.
/// - a **hard clamp**, scaling a tap's whole RGB so its luminance is at most
///   `cap` times that same leave-one-out mean. The gate is soft, and a network
///   that wanted to could learn to hold it open; the clamp is a `min` and it
///   cannot. Nothing is un-clamped afterwards, because a firefly is not energy
///   the picture is missing — it is a sampling artefact, and the reference
///   render does not have it either.
///
/// `cap >= 1` is enforced by the activation, and that is what keeps the
/// network's promise intact: a clamped tap is pulled down to at least its own
/// neighbourhood's mean and never below it, so the filtered pixel still lies
/// inside the range of light the path tracer measured nearby.
#[derive(Debug, Clone, Copy)]
pub struct Veto {
    /// Steepness of the gate, in log-luminance-ratio.
    pub scale: f32,
    /// Where the gate is half closed, in nats of ratio.
    pub thresh: f32,
    /// The clamp's ceiling, as a multiple of the leave-one-out mean.
    pub cap: f32,
}

/// `ln(1 + e^x)`, without overflowing for large `x`.
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

impl Veto {
    /// The three trained scalars, read through the activations that keep them
    /// in the range the veto means anything in.
    ///
    /// `scale` and the clamp's headroom go through a softplus because a
    /// negative steepness would gate the *dim* taps and a cap below one would
    /// pull every tap under its own neighbourhood; `thresh` is free, because
    /// every real number is a sensible place to put the knee.
    pub fn from_raw(raw: [f32; VETO]) -> Self {
        Self {
            scale: softplus(raw[0]),
            thresh: raw[1],
            cap: 1.0 + softplus(raw[2]),
        }
    }

    /// The raw scalars a fresh network starts from: a gate half closed at
    /// about 1.4x the neighbourhood, and a clamp at 3x.
    pub fn initial_raw() -> [f32; VETO] {
        [3.9819, 0.35, 1.8546]
    }

    /// One tap's gate and clamp, from its luminance and the leave-one-out
    /// mean of the rest of its neighbourhood.
    ///
    /// Returns `(gate, clamp)`: the first multiplies the tap's softmax weight,
    /// the second scales the tap's radiance.
    pub fn tap(&self, l: f32, mu: f32) -> (f32, f32) {
        let t = ((l + VETO_EPS) / (mu + VETO_EPS)).ln();
        let s = 1.0 / (1.0 + (self.scale * (t - self.thresh)).exp());
        let gate = VETO_FLOOR + (1.0 - VETO_FLOOR) * s;
        let clamp = ((self.cap * mu + VETO_EPS) / (l + VETO_EPS)).min(1.0);
        (gate, clamp)
    }
}

/// The leave-one-out means of a neighbourhood's luminances.
///
/// `mu[j]` is the mean of every luminance but `l[j]`. Excluding the tap is
/// what makes the estimate robust *to* the tap: a firefly weighed against a
/// mean it is itself a twenty-fifth of would talk its own threshold up, and
/// two fireflies in one neighbourhood would cover for each other.
pub fn leave_one_out(l: &[f32; K]) -> [f32; K] {
    let sum: f32 = l.iter().sum();
    let inv = 1.0 / (K - 1) as f32;
    let mut mu = [0.0f32; K];
    for j in 0..K {
        mu[j] = ((sum - l[j]) * inv).max(0.0);
    }
    mu
}

/// A trained network.
#[derive(Debug, Clone)]
pub struct Weights {
    /// Hidden channels in both interior layers.
    pub hidden: usize,
    /// `W1 b1 W2 b2 W3 b3 veto`, flat.
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
    /// The magic is `KOSMKPN2` and not `KOSMKPN1` because the tail grew: a
    /// v1 blob has no veto scalars, and reading one as if it did would take
    /// three of the last layer's biases for a gate.
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
        (h * C_IN * KS * KS + h) + (h * h * KS * KS + h) + (K * h * KS * KS + K) + VETO
    }

    /// Offsets into [`Weights::data`], in the order the blob stores them:
    /// `[W1, b1, W2, b2, W3, b3, veto]`. The GPU pass wants exactly these
    /// seven numbers in its uniform, so they are computed in one place.
    pub fn offsets(&self) -> [u32; 7] {
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
            push(VETO, &mut at),
        ]
    }

    /// The firefly veto this fit learned; see [`Veto`].
    pub fn veto(&self) -> Veto {
        let o = self.offsets()[6] as usize;
        Veto::from_raw([self.data[o], self.data[o + 1], self.data[o + 2]])
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
    /// | 10    | [`id_feature`] of the biased hit id |
    ///
    /// The two radiance-like planes are log-compressed so a light panel does
    /// not drown a floor; the count plane is what lets one network serve
    /// every history length, since a pixel on its first sample and one on its
    /// sixteenth want different kernels. The id plane is what keeps a
    /// backboard's glass out of the wall behind it: normal and depth agree
    /// across that silhouette often enough, and the id never does.
    #[allow(clippy::too_many_arguments)]
    pub fn features_at(
        mean: [f32; 3],
        variance: f32,
        normal: [f32; 3],
        depth: f32,
        albedo: [f32; 3],
        count: f32,
        id: f32,
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
            id_feature(id),
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
        id: &[f32],
        count: &[f32],
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
                count[p],
                id[p],
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
        let veto = self.veto();
        let r = (TAPS / 2) as i32;
        for y in 0..height {
            for x in 0..width {
                let p = y * width + x;

                // the 5x5 neighbourhood, once: its pixels and their luminances
                let mut tap = [0usize; K];
                let mut lum = [0.0f32; K];
                for ky in 0..TAPS {
                    let sy = clamp(y as i32 + ky as i32 - r, height);
                    for kx in 0..TAPS {
                        let sx = clamp(x as i32 + kx as i32 - r, width);
                        let q = sy * width + sx;
                        let j = ky * TAPS + kx;
                        tap[j] = q;
                        lum[j] = luminance([illum[q], illum[n + q], illum[2 * n + q]]).max(0.0);
                    }
                }
                let mu = leave_one_out(&lum);

                let mut mx = f32::NEG_INFINITY;
                for k in 0..K {
                    mx = mx.max(z3[k * n + p]);
                }
                // softmax, gated: the exponential is the network's opinion and
                // the gate is the veto's, and they multiply before the sum
                // that normalises them — so a vetoed tap does not merely lose
                // weight, it hands that weight to the taps that survived.
                let mut w = [0.0f32; K];
                let mut clip = [0.0f32; K];
                let mut sum = 0.0;
                for k in 0..K {
                    let (gate, clamp_k) = veto.tap(lum[k], mu[k]);
                    clip[k] = clamp_k;
                    w[k] = (z3[k * n + p] - mx).exp() * gate;
                    sum += w[k];
                }
                let inv = 1.0 / sum.max(1e-20);
                let mut acc = [0.0f32; 3];
                for k in 0..K {
                    let q = tap[k];
                    let wk = w[k] * inv * clip[k];
                    for c in 0..3 {
                        acc[c] += wk * illum[c * n + q];
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
        let mut data: Vec<f32> = (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / 8388608.0) - 0.5
            })
            .collect();
        // The veto's three scalars are not weights and random values there
        // mean a random gate; a synthetic net gets the fresh one instead.
        let tail = data.len() - VETO;
        data[tail..].copy_from_slice(&Veto::initial_raw());
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
        let id = vec![3.0f32; n];
        let count = vec![4.0f32; n];
        let out = net.forward(
            w, h, &mean, &variance, &normal, &depth, &albedo, &id, &count,
        );
        let lo = mean.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = mean.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for v in out {
            assert!(v >= lo - 1e-4 && v <= hi + 1e-4, "{v} left [{lo}, {hi}]");
        }
    }
}
