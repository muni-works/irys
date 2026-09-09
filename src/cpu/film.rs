//! The rendered frame and its tonemapping.

#[allow(unused_imports)]
use super::*;

/// A rendered frame in linear space.
pub struct Film {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Linear RGB radiance, 3 floats per pixel, row-major top-to-bottom.
    pub rgb: Vec<f32>,
    /// Coverage in 0..1, one float per pixel.
    pub alpha: Vec<f32>,
    /// World normal at each pixel's first hit, 3 floats per pixel. Zero for
    /// background pixels. Guide buffer for [`denoise`].
    pub normal: Vec<f32>,
    /// Distance from the camera to each pixel's first hit, one float per
    /// pixel. **Zero means the primary ray escaped** — the background
    /// sentinel. Guide buffer for [`denoise`].
    pub depth: Vec<f32>,
    /// Surface colour at each pixel's first hit, 3 floats per pixel. Divided
    /// out before filtering and multiplied back after, so [`denoise`] only
    /// ever blurs illumination.
    pub albedo: Vec<f32>,
    /// Estimated variance of each pixel's mean radiance luminance, one float
    /// per pixel — the Monte Carlo estimator's own error bar.
    ///
    /// [`denoise`] scales its luminance edge-stopping tolerance by this, which
    /// is what lets the filter tell "this neighbour is genuinely a different
    /// brightness" from "this pixel is a noise spike". Without it a firefly
    /// rejects every neighbour and survives the filter untouched.
    pub variance: Vec<f32>,
}

/// One pixel's worth of the integrator, writing into the row slices the film
/// keeps for that scanline.
///
/// Factored out of [`render`] so [`render_into`] can drive exactly the same
/// code on a subset of pixels. The RNG seed is a pure function of the pixel
/// coordinates and `opts.seed`, which is what makes a masked pass reproduce
/// the full render's pixels bit for bit — and what makes either of them
/// independent of how rayon happens to schedule the rows.
pub(crate) struct PixelOut<'a> {
    pub(crate) rgb: &'a mut [f32],
    pub(crate) alpha: &'a mut [f32],
    pub(crate) normal: &'a mut [f32],
    pub(crate) depth: &'a mut [f32],
    pub(crate) albedo: &'a mut [f32],
    pub(crate) variance: &'a mut [f32],
}

impl Film {
    /// A black film of `width` x `height`, with every guide buffer zeroed.
    ///
    /// [`render`] allocates its own; this is for the caller who holds one
    /// frame and keeps patching it with [`render_into`].
    pub fn new(width: u32, height: u32) -> Self {
        let n = (width as usize) * (height as usize);
        Self {
            width,
            height,
            rgb: vec![0.0; n * 3],
            alpha: vec![0.0; n],
            normal: vec![0.0; n * 3],
            depth: vec![0.0; n],
            albedo: vec![0.0; n * 3],
            variance: vec![0.0; n],
        }
    }
}

// ─── tonemapping ──────────────────────────────────────────────────────────

/// ACES filmic tonemap (Narkowicz fit).
#[inline]
pub fn tonemap_aces(x: f32) -> f32 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
}

/// Linear to sRGB transfer.
#[inline]
pub fn linear_to_srgb(x: f32) -> f32 {
    if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

impl Film {
    /// Convert to 8-bit sRGB RGBA with ACES tonemapping.
    ///
    /// `exposure` scales linear radiance before the tonemap curve.
    pub fn to_srgb8(&self, exposure: f32, transparent: bool) -> Vec<u8> {
        let n = (self.width * self.height) as usize;
        let mut out = vec![0u8; n * 4];
        for i in 0..n {
            for c in 0..3 {
                let v = tonemap_aces(self.rgb[i * 3 + c] * exposure);
                out[i * 4 + c] = (linear_to_srgb(v) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
            out[i * 4 + 3] = if transparent {
                (self.alpha[i] * 255.0 + 0.5).clamp(0.0, 255.0) as u8
            } else {
                255
            };
        }
        out
    }
}
