//! The photometric loss and its gradient: `(1 − λ) L1 + λ (1 − SSIM)`,
//! λ = 0.2 (3DGS), SSIM over an 11 × 11 Gaussian window (σ 1.5) with zero
//! padding, as `ipse/scripts/splat_train.py`'s `ssim`; plus an optional
//! L1 on the rendered optical depth where a depth frame is valid.

use rayon::prelude::*;

const R: usize = 5;

fn window() -> [f64; 2 * R + 1] {
    let mut w = [0.0; 2 * R + 1];
    for (i, v) in w.iter_mut().enumerate() {
        let x = i as f64 - R as f64;
        *v = (-(x * x) / 4.5).exp();
    }
    let s: f64 = w.iter().sum();
    w.map(|v| v / s)
}

/// Separable zero-padded Gaussian blur of a `w × h` plane (its own adjoint).
pub fn blur(src: &[f64], w: usize, h: usize) -> Vec<f64> {
    let k = window();
    let mut tmp = vec![0.0; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut s = 0.0;
            for (j, kv) in k.iter().enumerate() {
                let xx = x as isize + j as isize - R as isize;
                if xx >= 0 && (xx as usize) < w {
                    s += kv * src[y * w + xx as usize];
                }
            }
            row[x] = s;
        }
    });
    let mut out = vec![0.0; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut s = 0.0;
            for (j, kv) in k.iter().enumerate() {
                let yy = y as isize + j as isize - R as isize;
                if yy >= 0 && (yy as usize) < h {
                    s += kv * tmp[yy as usize * w + x];
                }
            }
            row[x] = s;
        }
    });
    out
}

const C1: f64 = 0.01 * 0.01;
const C2: f64 = 0.03 * 0.03;

/// Mean SSIM of one channel plane and `scale · ∂(mean SSIM)/∂a`.
pub fn ssim_plane(a: &[f64], b: &[f64], w: usize, h: usize, scale: f64) -> (f64, Vec<f64>) {
    let n = w * h;
    let scale = scale / n as f64;
    let mu1 = blur(a, w, h);
    let mu2 = blur(b, w, h);
    let aa: Vec<f64> = a.iter().map(|v| v * v).collect();
    let bb: Vec<f64> = b.iter().map(|v| v * v).collect();
    let ab: Vec<f64> = a.iter().zip(b).map(|(x, y)| x * y).collect();
    let (saa, sbb, sab) = (blur(&aa, w, h), blur(&bb, w, h), blur(&ab, w, h));
    let mut gmu = vec![0.0; n];
    let mut g11 = vec![0.0; n];
    let mut g12 = vec![0.0; n];
    let mut total = 0.0;
    for i in 0..n {
        let (m1, m2) = (mu1[i], mu2[i]);
        let s11 = saa[i] - m1 * m1;
        let s22 = sbb[i] - m2 * m2;
        let s12 = sab[i] - m1 * m2;
        let (n1, n2) = (2.0 * m1 * m2 + C1, 2.0 * s12 + C2);
        let (d1, d2) = (m1 * m1 + m2 * m2 + C1, s11 + s22 + C2);
        let s = n1 * n2 / (d1 * d2);
        total += s;
        let ds_dm1 = 2.0 * m2 * n2 / (d1 * d2) - s * 2.0 * m1 / d1;
        let ds_ds12 = 2.0 * n1 / (d1 * d2);
        let ds_ds11 = -s / d2;
        g11[i] = scale * ds_ds11;
        g12[i] = scale * ds_ds12;
        gmu[i] = scale * ds_dm1 - 2.0 * m1 * g11[i] - m2 * g12[i];
    }
    let (bmu, b11, b12) = (blur(&gmu, w, h), blur(&g11, w, h), blur(&g12, w, h));
    let grad = (0..n).map(|i| bmu[i] + 2.0 * a[i] * b11[i] + b[i] * b12[i]).collect();
    (total / n as f64, grad)
}

/// Loss weights.
#[derive(Debug, Clone, Copy)]
pub struct Weights {
    /// D-SSIM share (3DGS: 0.2).
    pub ssim: f64,
    /// Depth L1 weight (0: off).
    pub depth: f64,
}

impl Default for Weights {
    fn default() -> Self {
        Self { ssim: 0.2, depth: 0.0 }
    }
}

/// What a loss evaluation returns.
pub struct Eval {
    pub loss: f64,
    pub l1: f64,
    pub ssim: f64,
    /// `∂loss/∂(r, g, b, depth)` per pixel.
    pub grad: Vec<[f64; 4]>,
}

/// `pred` rgb+depth per pixel, `gt` sRGB in 0..1, optional `gt_depth`
/// (optical z, metres; ≤ 0 or non-finite: invalid).
pub fn eval(pred: &[[f64; 4]], gt: &[[f32; 3]], gt_depth: Option<&[f32]>, w: usize, h: usize, wt: Weights) -> Eval {
    let n = w * h;
    let m = (3 * n) as f64;
    let mut grad = vec![[0.0; 4]; n];
    let mut l1 = 0.0;
    for i in 0..n {
        for ch in 0..3 {
            let e = pred[i][ch] - gt[i][ch] as f64;
            l1 += e.abs();
            grad[i][ch] = (1.0 - wt.ssim) * e.signum() / m;
        }
    }
    l1 /= m;
    let mut ssim = 0.0;
    if wt.ssim > 0.0 {
        let planes: Vec<(f64, Vec<f64>)> = (0..3)
            .into_par_iter()
            .map(|ch| {
                let a: Vec<f64> = pred.iter().map(|p| p[ch]).collect();
                let b: Vec<f64> = gt.iter().map(|p| p[ch] as f64).collect();
                ssim_plane(&a, &b, w, h, -wt.ssim / 3.0)
            })
            .collect();
        for (ch, (s, g)) in planes.into_iter().enumerate() {
            ssim += s / 3.0;
            for i in 0..n {
                grad[i][ch] += g[i];
            }
        }
    }
    let mut loss = (1.0 - wt.ssim) * l1 + wt.ssim * (1.0 - ssim);
    if let (Some(d), true) = (gt_depth, wt.depth > 0.0) {
        let valid: Vec<usize> = (0..n).filter(|&i| d[i].is_finite() && d[i] > 0.0).collect();
        if !valid.is_empty() {
            let k = valid.len() as f64;
            let mut dl = 0.0;
            for &i in &valid {
                let e = pred[i][3] - d[i] as f64;
                dl += e.abs();
                grad[i][3] = wt.depth * e.signum() / k;
            }
            loss += wt.depth * dl / k;
        }
    }
    Eval { loss, l1, ssim, grad }
}

/// PSNR (dB) of sRGB images in 0..1, the prediction clamped.
pub fn psnr(pred: &[[f64; 4]], gt: &[[f32; 3]]) -> f64 {
    let mut se = 0.0;
    for (p, g) in pred.iter().zip(gt) {
        for ch in 0..3 {
            se += (p[ch].clamp(0.0, 1.0) - g[ch] as f64).powi(2);
        }
    }
    let mse = se / (3 * gt.len()) as f64;
    -10.0 * mse.max(1e-12).log10()
}

/// Mean SSIM of sRGB images, the prediction clamped.
pub fn ssim(pred: &[[f64; 4]], gt: &[[f32; 3]], w: usize, h: usize) -> f64 {
    (0..3)
        .map(|ch| {
            let a: Vec<f64> = pred.iter().map(|p| p[ch].clamp(0.0, 1.0)).collect();
            let b: Vec<f64> = gt.iter().map(|p| p[ch] as f64).collect();
            ssim_plane(&a, &b, w, h, 0.0).0
        })
        .sum::<f64>()
        / 3.0
}
