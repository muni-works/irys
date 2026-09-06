//! The claim behind `SamplePattern::BlueNoise` is about *where* the error
//! is, not how much of it there is: one sample a pixel of the same integrand
//! carries the same variance under either pattern, and the blue-noise
//! pattern moves it out of the low spatial frequencies. This measures that.
//!
//! A furnace: every pixel of a 128×128 frame estimates the same integral
//! from its own one sample, so the frame *is* the error image. Its 2D FFT,
//! averaged over frames, is the error's power spectrum; the share of that
//! power inside the lowest eighth of the frequency range is what an à-trous
//! filter would pass through as blotches. White noise is flat, so its share
//! is the area of that disc; blue noise has to come in well under it.

use kosm_render::sampler::SamplePattern;

const N: usize = 128;

/// In-place radix-2 FFT of one complex row, `n` a power of two.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = -2.0 * std::f64::consts::PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        for start in (0..n).step_by(len) {
            let (mut cr, mut ci) = (1.0, 0.0);
            for k in 0..len / 2 {
                let (ar, ai) = (re[start + k], im[start + k]);
                let (br, bi) = (re[start + k + len / 2], im[start + k + len / 2]);
                let (tr, ti) = (br * cr - bi * ci, br * ci + bi * cr);
                re[start + k] = ar + tr;
                im[start + k] = ai + ti;
                re[start + k + len / 2] = ar - tr;
                im[start + k + len / 2] = ai - ti;
                let ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
        }
        len <<= 1;
    }
}

/// Power spectrum of an N×N real image.
fn power_spectrum(img: &[f64]) -> Vec<f64> {
    let mut re = img.to_vec();
    let mut im = vec![0.0; N * N];
    for y in 0..N {
        fft(&mut re[y * N..(y + 1) * N], &mut im[y * N..(y + 1) * N]);
    }
    let (mut cr, mut ci) = (vec![0.0; N], vec![0.0; N]);
    for x in 0..N {
        for y in 0..N {
            cr[y] = re[y * N + x];
            ci[y] = im[y * N + x];
        }
        fft(&mut cr, &mut ci);
        for y in 0..N {
            re[y * N + x] = cr[y];
            im[y * N + x] = ci[y];
        }
    }
    re.iter().zip(&im).map(|(r, i)| r * r + i * i).collect()
}

/// The share of a spectrum's power (DC excluded) within `radius` of DC.
fn low_frequency_share(power: &[f64], radius: f64) -> f64 {
    let (mut low, mut total) = (0.0, 0.0);
    for y in 0..N {
        for x in 0..N {
            if x == 0 && y == 0 {
                continue;
            }
            let fx = if x <= N / 2 { x } else { N - x } as f64;
            let fy = if y <= N / 2 { y } else { N - y } as f64;
            let p = power[y * N + x];
            total += p;
            if (fx * fx + fy * fy).sqrt() <= radius {
                low += p;
            }
        }
    }
    low / total
}

/// One frame of the furnace under `pattern`: each pixel's one-sample
/// estimate of the integral, minus the integral.
///
/// The integrand is a visibility-shaped one, a quarter disc of radius 1 in
/// the unit square (mean π/4), drawn from dimensions 2 and 3 — the first
/// bounce's first pair — and a smooth one, a cosine bump on the pixel's own
/// jitter (mean 1/2), from dimensions 0 and 1. Two very different integrands
/// so the claim is not about one shape.
fn error_frames(pattern: SamplePattern, dims: [u32; 2], frames: u32) -> Vec<Vec<f64>> {
    (1..=frames)
        .map(|frame| {
            (0..N * N)
                .map(|i| {
                    let p = [(i % N) as u32, (i / N) as u32];
                    let u = pattern.sample(p, frame, dims[0]) as f64;
                    let v = pattern.sample(p, frame, dims[1]) as f64;
                    if dims[0] == 0 {
                        0.5 + 0.5
                            * (2.0 * std::f64::consts::PI * u).cos()
                            * (2.0 * std::f64::consts::PI * v).sin()
                            - 0.5
                    } else {
                        (if u * u + v * v < 1.0 { 1.0 } else { 0.0 }) - std::f64::consts::FRAC_PI_4
                    }
                })
                .collect()
        })
        .collect()
}

fn mean_low_share(pattern: SamplePattern, dims: [u32; 2]) -> (f64, f64) {
    let frames = error_frames(pattern, dims, 16);
    let mut acc = vec![0.0; N * N];
    let mut var = 0.0;
    for f in &frames {
        for (a, p) in acc.iter_mut().zip(power_spectrum(f)) {
            *a += p;
        }
        var += f.iter().map(|e| e * e).sum::<f64>() / (N * N) as f64;
    }
    (
        low_frequency_share(&acc, N as f64 / 8.0),
        var / frames.len() as f64,
    )
}

#[test]
fn blue_noise_moves_the_error_out_of_the_low_frequencies() {
    // The disc |f| <= N/8 holds (π/64) ≈ 4.9% of the spectrum's area, which
    // is the share a flat (white) spectrum puts there.
    let flat = std::f64::consts::PI / 64.0;
    for dims in [[0u32, 1], [2, 3]] {
        let (white, white_var) = mean_low_share(SamplePattern::White, dims);
        let (blue, blue_var) = mean_low_share(SamplePattern::BlueNoise, dims);
        eprintln!(
            "dims {dims:?}: low-frequency share white {white:.4} blue {blue:.4} (flat {flat:.4}); \
             per-frame variance white {white_var:.4} blue {blue_var:.4}"
        );
        assert!(
            (white - flat).abs() < 0.4 * flat,
            "white noise is not white: {white}"
        );
        // A smooth integrand's error is a smooth function of the shift and
        // inherits the mask's spectrum whole: an order of magnitude. A step's
        // error is which side of it the shift lands on, and only the step
        // family in the mask's energy pulls that below white — measured at
        // 0.70 of white, against 1.6 for a mask annealed without it.
        let bound = if dims[0] == 0 { 0.25 } else { 0.85 };
        assert!(
            blue < bound * white,
            "blue noise is not blue: {blue} against {white}"
        );
        // Same integrand, one sample: the same variance, give or take.
        assert!(
            (blue_var - white_var).abs() < 0.25 * white_var,
            "{blue_var} vs {white_var}"
        );
    }
}

#[test]
fn blue_noise_converges_faster_than_white_over_a_history() {
    // Over 16 frames a pixel's samples are a (0, 2)-sequence prefix under
    // blue noise and 16 independent draws under white: the mean of the disc
    // estimate should come in with visibly less error.
    let n_frames = 16;
    let err = |pattern: SamplePattern| {
        let frames = error_frames(pattern, [2, 3], n_frames);
        let mut rmse = 0.0;
        for i in 0..N * N {
            let m: f64 = frames.iter().map(|f| f[i]).sum::<f64>() / n_frames as f64;
            rmse += m * m;
        }
        (rmse / (N * N) as f64).sqrt()
    };
    let (w, b) = (err(SamplePattern::White), err(SamplePattern::BlueNoise));
    eprintln!("16-frame mean RMSE: white {w:.4} blue {b:.4}");
    assert!(b < 0.8 * w, "blue {b} should beat white {w}");
}
