//! The edge-avoiding a-trous denoiser.

use super::*;

// ─── denoising ────────────────────────────────────────────────────────────

/// 5×5 separable B3-spline (cubic) kernel, `[1 4 6 4 1] / 16`.
pub(crate) const B3_SPLINE: [f32; 5] = [1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0];

/// Floor on the demodulation divisor.
///
/// Dividing by a near-black albedo would turn a dark surface's illumination
/// into enormous numbers, and any filtering error there comes back multiplied.
/// Clamping trades a little residual colour-blurring on very dark materials
/// for numerical sanity.
pub(crate) const DEMOD_FLOOR: f32 = 0.05;

/// Edge-aware à-trous wavelet denoiser (Dammertz et al., EGSR 2010).
///
/// Filters the film's linear radiance in place, guided by the normal, depth,
/// and albedo buffers that [`render`] records from each pixel's primary ray.
///
/// The algorithm is a sequence of 5×5 B3-spline convolutions whose taps are
/// spread by a doubling stride ("holes" — *à trous*), each tap weighted by how
/// well the neighbour matches the centre pixel's normal, depth, and
/// illumination. That reaches a wide footprint in a few passes while refusing
/// to average across geometric or shading discontinuities.
///
/// Two properties are worth naming, because the tests pin them:
///
/// - **Illumination only.** Radiance is divided by albedo before filtering and
///   multiplied back afterwards, so a part's colour is never blurred into its
///   neighbour's — only the Monte Carlo noise in the lighting is smoothed.
/// - **Background is inviolable.** A pixel whose primary ray escaped
///   (`depth == 0`) is passed through untouched, and no surface pixel ever
///   accepts a tap from one. Silhouettes against the backdrop stay exactly as
///   sharp as the path tracer drew them.
///
/// This is a post-process: it consumes no random numbers and never touches the
/// integrator, so a reference render is exactly the un-denoised film.
///
/// Only the `denoise_iters` and `sigma_*` fields of `opts` are read. Calling
/// this *is* the request to filter, so [`PathTraceOptions::denoise`] is the
/// caller's gate — as [`render`] uses it — and is deliberately ignored here.
pub fn denoise(film: &mut Film, opts: &PathTraceOptions) {
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(not(target_arch = "wasm32"))]
    use rayon::prelude::*;

    let w = film.width as usize;
    let h = film.height as usize;
    let n = w * h;
    if n == 0 || opts.denoise_iters == 0 {
        return;
    }

    // Demodulate: work on illumination = radiance / albedo.
    let mut illum = vec![0.0f32; n * 3];
    let mut var = vec![0.0f32; n];
    for i in 0..n {
        for c in 0..3 {
            let a = film.albedo[i * 3 + c].max(DEMOD_FLOOR);
            illum[i * 3 + c] = film.rgb[i * 3 + c] / a;
        }
        // Variance was measured on radiance; demodulation scales it by the
        // square of the (scalar) albedo it divided through.
        let la = luminance([
            film.albedo[i * 3].max(DEMOD_FLOOR),
            film.albedo[i * 3 + 1].max(DEMOD_FLOOR),
            film.albedo[i * 3 + 2].max(DEMOD_FLOOR),
        ])
        .max(DEMOD_FLOOR);
        var[i] = film.variance[i] / (la * la);
    }

    // Prefilter the variance estimate with a 3×3 box. The per-pixel estimate
    // is itself noisy at low sample counts, and a noisy error bar makes the
    // luminance weight jitter between "trust" and "reject" from pixel to
    // pixel.
    {
        let mut smooth = var.clone();
        for y in 0..h {
            for x in 0..w {
                let mut s = 0.0f32;
                let mut k = 0.0f32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let (qx, qy) = (x as i32 + dx, y as i32 + dy);
                        if qx < 0 || qy < 0 || qx >= w as i32 || qy >= h as i32 {
                            continue;
                        }
                        let q = qy as usize * w + qx as usize;
                        if film.depth[q] <= 0.0 {
                            continue;
                        }
                        s += var[q];
                        k += 1.0;
                    }
                }
                if k > 0.0 {
                    smooth[y * w + x] = s / k;
                }
            }
        }
        var = smooth;
    }

    let sigma_n2 = (opts.sigma_normal.max(1e-4)).powi(2);
    let mut scratch = illum.clone();
    let mut var_scratch = var.clone();
    let g_depth = &film.depth;
    let g_normal = &film.normal;

    for it in 0..opts.denoise_iters {
        let stride = 1usize << it;
        // Dammertz shrinks a *fixed* illumination tolerance as the footprint
        // grows. Here the tolerance is already scaled by the filtered variance,
        // which shrinks on its own as the estimate gets cleaner, so shrinking
        // sigma too would penalise the wide passes twice and they would reject
        // every tap. Measured: with the extra 2^-i, iterations past the first
        // bought nothing at all.
        let sigma_l = opts.sigma_lum.max(1e-6);
        let sigma_z = opts.sigma_depth.max(1e-6) * stride as f32;

        film_rows!(scratch, w * 3)
            .zip(film_rows!(var_scratch, w))
            .enumerate()
            .for_each(|(y, (row, vrow))| {
                for x in 0..w {
                    let p = y * w + x;
                    let z_p = g_depth[p];
                    if z_p <= 0.0 {
                        // Background: analytic and noise-free. Pass through.
                        row[x * 3] = illum[p * 3];
                        row[x * 3 + 1] = illum[p * 3 + 1];
                        row[x * 3 + 2] = illum[p * 3 + 2];
                        vrow[x] = var[p];
                        continue;
                    }
                    let n_p = [g_normal[p * 3], g_normal[p * 3 + 1], g_normal[p * 3 + 2]];
                    let c_p = [illum[p * 3], illum[p * 3 + 1], illum[p * 3 + 2]];
                    let l_p = luminance(c_p);
                    // The estimator's own error bar sets how much luminance
                    // disagreement counts as signal rather than noise. A
                    // firefly has an enormous error bar, so it stops
                    // protecting itself and gets filtered.
                    let l_tol = sigma_l * var[p].max(0.0).sqrt() + 1e-4;

                    let mut sum = [0.0f32; 3];
                    let mut vsum = 0.0f32;
                    let mut wsum = 0.0f32;

                    for (ky, dy) in (-2i32..=2).enumerate() {
                        let qy = y as i32 + dy * stride as i32;
                        if qy < 0 || qy >= h as i32 {
                            continue;
                        }
                        for (kx, dx) in (-2i32..=2).enumerate() {
                            let qx = x as i32 + dx * stride as i32;
                            if qx < 0 || qx >= w as i32 {
                                continue;
                            }
                            let q = qy as usize * w + qx as usize;
                            let z_q = g_depth[q];
                            if z_q <= 0.0 {
                                // Never let the backdrop bleed onto a surface.
                                continue;
                            }

                            // Normal: squared distance between unit normals.
                            let dn = [
                                n_p[0] - g_normal[q * 3],
                                n_p[1] - g_normal[q * 3 + 1],
                                n_p[2] - g_normal[q * 3 + 2],
                            ];
                            let dn2 = dn[0] * dn[0] + dn[1] * dn[1] + dn[2] * dn[2];
                            let w_n = (-dn2 / sigma_n2).exp();

                            // Depth: relative, so the tolerance scales with
                            // scene size instead of being tuned per model.
                            let w_z = (-(z_p - z_q).abs() / (sigma_z * z_p)).exp();

                            // Illumination: rejects the far side of a shadow
                            // edge or a specular highlight.
                            let c_q = [illum[q * 3], illum[q * 3 + 1], illum[q * 3 + 2]];
                            let w_l = (-(l_p - luminance(c_q)).abs() / l_tol).exp();

                            let weight = B3_SPLINE[kx] * B3_SPLINE[ky] * w_n * w_z * w_l;
                            if weight <= 0.0 {
                                continue;
                            }
                            sum = add3(sum, scale3(c_q, weight));
                            // Variance of a weighted mean of independent
                            // estimates carries the *squared* weights.
                            vsum += weight * weight * var[q];
                            wsum += weight;
                        }
                    }

                    let (out, vout) = if wsum > 0.0 {
                        (scale3(sum, 1.0 / wsum), vsum / (wsum * wsum))
                    } else {
                        (c_p, var[p])
                    };
                    row[x * 3] = out[0];
                    row[x * 3 + 1] = out[1];
                    row[x * 3 + 2] = out[2];
                    vrow[x] = vout;
                }
            });

        std::mem::swap(&mut illum, &mut scratch);
        std::mem::swap(&mut var, &mut var_scratch);
    }

    // Re-modulate back into radiance. Background pixels are left exactly as
    // the tracer wrote them — a divide-then-multiply round trip is not
    // bit-exact in f32, and the backdrop has no noise to remove anyway.
    for i in 0..n {
        if film.depth[i] <= 0.0 {
            continue;
        }
        for c in 0..3 {
            let a = film.albedo[i * 3 + c].max(DEMOD_FLOOR);
            film.rgb[i * 3 + c] = illum[i * 3 + c] * a;
        }
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::cpu::testing::*;
    #[allow(unused_imports)]
    use crate::geometry::TriMesh;

    fn rmse(a: &Film, b: &Film) -> f32 {
        assert_eq!(a.rgb.len(), b.rgb.len());
        let s: f32 = a
            .rgb
            .iter()
            .zip(&b.rgb)
            .map(|(x, y)| {
                let d = tonemap_aces(*x) - tonemap_aces(*y);
                d * d
            })
            .sum();
        (s / a.rgb.len() as f32).sqrt()
    }

    /// The property that matters: denoising a noisy render must move it
    /// *closer to the truth*, not merely change it. A blur that smeared
    /// everything would also "change the output" while making the image
    /// worse, and this is the test that tells the two apart.
    #[test]
    fn denoise_moves_low_spp_toward_high_spp_reference() {
        let scene = test_scene();
        let cam = test_camera();
        // Big enough that the doubling stride is meaningful: at 28px a
        // 5-iteration à-trous reaches past the image edge and the later passes
        // can only over-blur, which made an earlier version of this test
        // report a 7% win where the real figure is ~60%.
        let (w, h) = (96, 96);

        let reference = render(
            &scene,
            &cam,
            w,
            h,
            &PathTraceOptions {
                spp: 1024,
                denoise: false,
                ..Default::default()
            },
        );
        let noisy = render(
            &scene,
            &cam,
            w,
            h,
            &PathTraceOptions {
                spp: 4,
                denoise: false,
                ..Default::default()
            },
        );
        let denoised = render(
            &scene,
            &cam,
            w,
            h,
            &PathTraceOptions {
                spp: 4,
                denoise: true,
                ..Default::default()
            },
        );

        // Denoising is a post-process, so the two 4-spp films must have come
        // from the very same samples.
        assert_eq!(
            noisy.alpha, denoised.alpha,
            "denoising perturbed the sampling"
        );

        let before = rmse(&noisy, &reference);
        let after = rmse(&denoised, &reference);
        eprintln!("RMSE vs 1024spp: noisy {before:.5} -> denoised {after:.5}");
        // Measured ~60% reduction; assert a conservative fraction of it so the
        // test pins real quality rather than just "something happened", without
        // being brittle to sampling changes upstream.
        assert!(
            after < before * 0.75,
            "denoising did not meaningfully improve the estimate: \
             RMSE {before} -> {after}"
        );
    }

    /// The denoiser must not blur across a silhouette. Background pixels are
    /// analytic and noise-free, so they must come through untouched, and no
    /// surface pixel may pick up any backdrop.
    #[test]
    fn denoise_preserves_silhouette_edge() {
        let scene = test_scene();
        let cam = test_camera();
        let (w, h) = (48, 48);
        let opts = PathTraceOptions {
            spp: 4,
            denoise: false,
            ..Default::default()
        };
        let raw = render(&scene, &cam, w, h, &opts);
        let mut filtered = render(&scene, &cam, w, h, &opts);
        denoise(
            &mut filtered,
            &PathTraceOptions {
                denoise: true,
                ..opts
            },
        );

        let n = (w * h) as usize;
        let bg: Vec<usize> = (0..n).filter(|&i| raw.depth[i] <= 0.0).collect();
        let fg: Vec<usize> = (0..n).filter(|&i| raw.depth[i] > 0.0).collect();
        assert!(
            !bg.is_empty() && !fg.is_empty(),
            "test framing must contain both subject and backdrop"
        );

        // Backdrop is bit-identical.
        for &i in &bg {
            for c in 0..3 {
                assert_eq!(
                    raw.rgb[i * 3 + c],
                    filtered.rgb[i * 3 + c],
                    "backdrop pixel {i} was modified by the denoiser"
                );
            }
        }

        // Silhouette contrast is retained. A filter that leaked across the
        // edge would pull the two sides toward each other.
        let mean_lum = |f: &Film, idx: &[usize]| -> f32 {
            let s: f32 = idx
                .iter()
                .map(|&i| luminance([f.rgb[i * 3], f.rgb[i * 3 + 1], f.rgb[i * 3 + 2]]))
                .sum();
            s / idx.len() as f32
        };
        // Only the surface pixels that actually touch the backdrop.
        let rim: Vec<usize> = fg
            .iter()
            .copied()
            .filter(|&i| {
                let (x, y) = ((i % w as usize) as i32, (i / w as usize) as i32);
                [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)]
                    .iter()
                    .any(|(dx, dy)| {
                        let (qx, qy) = (x + dx, y + dy);
                        qx >= 0
                            && qy >= 0
                            && qx < w as i32
                            && qy < h as i32
                            && raw.depth[qy as usize * w as usize + qx as usize] <= 0.0
                    })
            })
            .collect();
        assert!(!rim.is_empty(), "expected a silhouette rim");

        let before = (mean_lum(&raw, &rim) - mean_lum(&raw, &bg)).abs();
        let after = (mean_lum(&filtered, &rim) - mean_lum(&filtered, &bg)).abs();
        assert!(
            after >= before * 0.95,
            "silhouette contrast collapsed: {before} -> {after}"
        );
    }
}
