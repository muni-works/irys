//! The path integrator: options, `radiance`, and the render entry points.

use super::*;

/// Integrator settings.
#[derive(Debug, Clone, Copy)]
pub struct PathTraceOptions {
    /// Samples per pixel.
    pub spp: u32,
    /// Maximum path length (1 = direct lighting only).
    pub max_depth: u32,
    /// Depth at which Russian roulette begins.
    pub rr_start: u32,
    /// Clamp on indirect radiance, to kill fireflies. `None` disables.
    ///
    /// An *absolute* cap, in radiance units: any direct-lighting estimate at
    /// depth > 0 is truncated to it. Cheap and effective, and biased in a way
    /// that does not matter for a studio render — but it is a fixed number
    /// against a quantity whose scale is the scene's, so a bright scene has
    /// its highlights shaved and a dim one keeps its fireflies.
    ///
    /// A caustic is exactly the case where that bias is *not* acceptable: a
    /// focused spot is legitimately many times the surrounding radiance, and
    /// an absolute cap is indistinguishable from throwing the caustic away.
    /// Contributions read out of a [`crate::caustics::CausticMap`] are
    /// therefore never clamped — they are a density estimate, not a Monte
    /// Carlo spike, and they have no long tail to cut.
    pub firefly_clamp: Option<f32>,
    /// Clamp on indirect radiance *relative to what the pixel has already
    /// measured*, replacing [`Self::firefly_clamp`] when set.
    ///
    /// The number is a multiple of the pixel's running mean luminance: `8.0`
    /// lets any sample through that is within eight times the brightness the
    /// pixel has settled on so far, and cuts the ones past it. Scale-free, so
    /// the same value works on a sunlit court and a dim pool, and it adapts to
    /// the pixel rather than to the scene — a pixel inside a caustic has a
    /// high running mean and keeps its energy, a pixel in shadow does not.
    ///
    /// The first few samples have no mean to speak of, so the clamp does not
    /// engage until [`Self::firefly_clamp_warmup`] samples have landed.
    ///
    /// `None` — the default — leaves the absolute clamp in charge and every
    /// render that predates this field bit-identical.
    pub firefly_clamp_relative: Option<f32>,
    /// Samples a pixel must take before [`Self::firefly_clamp_relative`]
    /// engages.
    pub firefly_clamp_warmup: u32,
    /// Render the environment behind the subject rather than leaving it clear.
    pub show_background: bool,
    /// Random seed.
    pub seed: u64,
    /// Stop sampling a pixel early once its own variance estimate says the
    /// remaining budget cannot move it visibly.
    ///
    /// [`spp`](Self::spp) becomes a *ceiling* rather than a fixed count. Every
    /// pixel still gets at least a floor of samples, and the decision is made
    /// from the pixel's own running sums, so the film stays deterministic and
    /// independent of how the frame was tiled — a pixel that stops early keeps
    /// the unbiased mean of the samples it did take.
    ///
    /// Set `false` for a reference render, where a uniform sample count is
    /// the point.
    pub adaptive: bool,
    /// Reconstruction filter for primary-ray placement within the pixel.
    ///
    /// [`PixelFilter::Box`] — uniform jitter — is the default and reproduces
    /// every earlier render bit for bit.
    pub filter: PixelFilter,
    /// Run the edge-aware à-trous denoiser over the film before returning.
    ///
    /// This is a pure post-process on the accumulated radiance — it consumes
    /// no random numbers and cannot change the integrator's estimate.
    pub denoise: bool,
    /// À-trous iterations. Each doubles the tap stride, so `n` iterations
    /// reach a footprint of roughly `2^(n+1)` pixels.
    pub denoise_iters: u32,
    /// Edge-stopping tolerance on the world normal, as `‖n_p − n_q‖`.
    /// Smaller keeps creases sharper and denoises less.
    pub sigma_normal: f32,
    /// Edge-stopping tolerance on hit distance, *relative* to the centre
    /// pixel's depth and scaled by the tap stride (so grazing surfaces still
    /// filter).
    pub sigma_depth: f32,
    /// Edge-stopping tolerance on demodulated illumination luminance. Halved
    /// each iteration, per Dammertz, so late wide passes cannot flatten
    /// detail the early passes already resolved.
    pub sigma_lum: f32,
}

impl Default for PathTraceOptions {
    fn default() -> Self {
        Self {
            spp: 128,
            max_depth: 6,
            rr_start: 3,
            firefly_clamp: Some(12.0),
            firefly_clamp_relative: None,
            firefly_clamp_warmup: 16,
            show_background: true,
            seed: 0x5eed_1234,
            adaptive: true,
            filter: PixelFilter::Box,
            denoise: true,
            denoise_iters: 5,
            sigma_normal: 0.35,
            sigma_depth: 0.02,
            sigma_lum: 4.0,
        }
    }
}

// ─── integrator ───────────────────────────────────────────────────────────

/// What the primary ray of a path found at depth 0.
///
/// Recorded for the denoiser's guide buffers. `depth == 0.0` means the primary
/// ray escaped the scene — the sentinel for "background", which the filter
/// refuses to mix with any surface.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Primary {
    /// Whether the primary ray hit geometry or an emitter (drives alpha).
    pub(crate) hit: bool,
    /// Face-forwarded world normal at the first hit.
    pub(crate) normal: [f32; 3],
    /// Distance from the camera to the first hit; 0 for a miss.
    pub(crate) depth: f32,
    /// Surface colour at the first hit, for albedo demodulation.
    pub(crate) albedo: [f32; 3],
}

/// Trace one path and return its radiance estimate, plus what its primary ray
/// landed on (for alpha and for the denoiser's guide buffers).
#[allow(clippy::too_many_arguments)]
pub(crate) fn radiance<G: Geometry>(
    scene: &Scene<G>,
    accel: &SceneAccel<G>,
    opts: &PathTraceOptions,
    caustics: Option<&CausticMap>,
    clamp_scale: Option<f32>,
    ray: Ray,
    rng: &mut Rng,
) -> ([f32; 3], Primary) {
    let origin = ray.origin;
    let mut primary = Primary::default();
    let mut l = [0.0f32; 3];
    let mut throughput = [1.0f32; 3];
    let mut ray = ray;
    // The previous bounce was sampled from a lobe with this PDF; used to MIS
    // against light sampling when the new ray lands on an emitter.
    let mut prev_bsdf_pdf = 0.0f32;
    let mut specular_chain = true;
    // The path's hero wavelength, in nanometres. `None` until the path meets
    // a material whose index actually depends on it — an RGB path stays RGB,
    // draws no extra random number, and renders bit-identically to what it
    // did before dispersion existed.
    let mut lambda_nm: Option<f64> = None;
    // The medium the path is currently inside, for Beer–Lambert absorption.
    // One slot, not a stack: this tracks a ray inside *a* solid, which is
    // every glass in these scenes. Nested dielectrics (a bubble in glass, ice
    // in a drink) would need a stack and would get the outer medium wrong
    // here; that is the documented limit.
    let mut medium: Option<Pbr> = None;

    for depth in 0..opts.max_depth {
        let landing = scene.intersect(accel, &ray);
        // The splat cloud along this segment, composited front to back and
        // stopped at whatever the segment ran into. A captured cloud is a
        // radiance field with its lighting already baked in, so it is added
        // as emission and its accumulated opacity veils everything past it:
        // `L += throughput · C`, then `throughput *= T`. Doing it here — for
        // the camera ray and for every bounce ray alike — is what makes the
        // cloud an *environment with depth*: a bounce ray that finds no
        // analytic surface comes back with the room's own colour, so the
        // marble is lit by the garage it is standing in.
        if scene.splats.is_some()
            && !(depth == 0 && !opts.show_background && matches!(landing, Landing::Miss))
        {
            let t_hit = match &landing {
                Landing::Surface { point, .. } => (*point - ray.origin).norm(),
                Landing::Light { distance, .. } => *distance,
                Landing::Miss => f64::INFINITY,
            };
            let seg = scene.splat_segment(&ray, 1e-6, t_hit);
            if max3(seg.radiance) > 0.0 {
                l = add3(l, mul3(throughput, seg.radiance));
            }
            if depth == 0 && seg.transmittance < 0.5 {
                // The cloud, not the background, is what this pixel shows.
                primary.hit = true;
                primary.albedo = seg.radiance;
            }
            throughput = scale3(throughput, seg.transmittance);
            if max3(throughput) <= 1e-5 {
                break;
            }
        }
        // Absorb along the segment just travelled, if it was inside glass.
        if let Some(med) = &medium {
            let sigma = med.extinction();
            if max3(sigma) > 0.0 {
                let d = match &landing {
                    Landing::Surface { point, .. } => (*point - ray.origin).norm() as f32,
                    Landing::Light { distance, .. } => *distance as f32,
                    Landing::Miss => 0.0,
                };
                if d > 0.0 {
                    throughput = mul3(
                        throughput,
                        [
                            (-sigma[0] * d).exp(),
                            (-sigma[1] * d).exp(),
                            (-sigma[2] * d).exp(),
                        ],
                    );
                }
            }
        }
        match landing {
            Landing::Miss => {
                let dir = ray.direction.into_inner();
                let env = scene.env.radiance(dir);
                if depth == 0 && !opts.show_background {
                    // Leave the backdrop clear; still no contribution.
                    break;
                }
                // MIS against environment NEE, which could also have found
                // this direction. A specular chain (including the primary
                // ray) had no other strategy, so it takes full weight.
                let w = if specular_chain || !scene.env.is_importance_sampled() {
                    1.0
                } else {
                    power_heuristic(prev_bsdf_pdf, scene.env.pdf(dir))
                };
                l = add3(l, scale3(mul3(throughput, env), w));
                // The sun disc, if this ray happened to land in it. NEE
                // samples the same cone, so the two strategies share the
                // direction under the balance heuristic; a specular chain
                // (the primary ray included) had no other way to find it.
                if let Some(sun) = &scene.sun {
                    let li = sun.radiance_in(dir);
                    if max3(li) > 0.0 {
                        let ws = if specular_chain {
                            1.0
                        } else {
                            power_heuristic(prev_bsdf_pdf, sun.pdf(dir))
                        };
                        l = add3(l, scale3(mul3(throughput, li), ws));
                    }
                }
                break;
            }
            Landing::Light {
                emission,
                light_index,
                distance,
                point,
            } => {
                if depth == 0 {
                    // An emitter seen directly. It is noise-free by
                    // construction, but it still needs a guide entry so the
                    // filter treats it as its own surface rather than as
                    // background.
                    primary.hit = true;
                    primary.depth = distance as f32;
                    primary.normal = vec_to_f32(scene.lights[light_index].normal());
                    primary.albedo = [1.0; 3];
                }
                let w = if specular_chain {
                    1.0
                } else {
                    // MIS against the NEE strategy that could also have found
                    // this light.
                    let light = &scene.lights[light_index];
                    let ln = light.normal();
                    let cos_light = (-ray.direction.into_inner().dot(ln)).max(1e-9);
                    let light_pdf = accel.light_pick_pdf(light_index)
                        * (distance * distance / (cos_light * light.area())) as f32;
                    let _ = point;
                    power_heuristic(prev_bsdf_pdf, light_pdf)
                };
                l = add3(l, scale3(mul3(throughput, emission), w));
                break;
            }
            Landing::Surface {
                point,
                normal,
                tangent,
                material,
            } => {
                let wo_world = -ray.direction.into_inner();
                // Which side of the *geometric* normal the ray arrived on is
                // the whole of the inside/outside bookkeeping: a front face is
                // an entry, a back face an exit. Read before the face-forward
                // that follows destroys the distinction.
                let entering = normal.dot(wo_world) >= 0.0;
                // Face-forward: interior faces (bore walls) must shade right.
                let n = if normal.dot(wo_world) < 0.0 {
                    -normal
                } else {
                    normal
                };
                // A dispersive material turns the path monochromatic, once.
                // The draw is inside the `if` so a scene without dispersion
                // consumes the RNG stream exactly as it always has.
                if lambda_nm.is_none() && material.is_dispersive() {
                    let nm = crate::spectrum::sample_lambda_nm(rng.f64());
                    lambda_nm = Some(nm);
                    throughput = mul3(throughput, crate::spectrum::hero_weight(nm));
                }
                // `eta` is n_transmitted / n_incident for this crossing.
                let eta = if material.transmission > 0.0 {
                    let n_glass = material.index_at(lambda_nm).max(1e-3);
                    if material.thin_walled || entering {
                        n_glass
                    } else {
                        1.0 / n_glass
                    }
                } else {
                    1.0
                };
                if depth == 0 {
                    primary.hit = true;
                    primary.depth = (point - origin).norm() as f32;
                    primary.normal = vec_to_f32(n);
                    primary.albedo = material.denoise_albedo();
                }
                // The hero wavelength as the BSDF wants it: `0` for an RGB
                // path, which is the sentinel every lobe reads as "achromatic".
                let hero = lambda_nm.unwrap_or(0.0) as f32;
                let frame = shading_frame(n, tangent);
                let wo_local = to_local(frame.t, frame.b, n, wo_world);
                if wo_local.z <= 0.0 {
                    break;
                }

                l = add3(l, mul3(throughput, material.emissive));

                // Next-event estimation: explicit lights, plus the
                // environment when it is importance-sampled.
                let direct = add3(
                    add3(
                        scene.sample_lights(
                            accel, point, &frame, wo_local, &material, eta, hero, rng,
                        ),
                        scene.sample_environment(
                            accel, point, &frame, wo_local, &material, eta, hero, rng,
                        ),
                    ),
                    scene.sample_sun(accel, point, &frame, wo_local, &material, eta, hero, rng),
                );
                let direct = if depth > 0 {
                    // The relative clamp, when armed, takes over from the
                    // absolute one; otherwise nothing about this changed.
                    match (clamp_scale, opts.firefly_clamp) {
                        (Some(c), _) | (None, Some(c)) => {
                            [direct[0].min(c), direct[1].min(c), direct[2].min(c)]
                        }
                        (None, None) => direct,
                    }
                } else {
                    direct
                };
                l = add3(l, mul3(throughput, direct));

                // The caustic map's share: light that arrived here by
                // refraction through a solid, which next-event estimation
                // could not have found and which the shadow rays above
                // therefore did not count. Added *outside* the firefly clamp
                // — it is a density estimate with no long tail, and clamping
                // it is indistinguishable from deleting the caustic.
                if let Some(map) = caustics.filter(|_| material.transmission <= 0.0) {
                    let rho = material.diffuse_albedo();
                    if max3(rho) > 0.0 {
                        let e = map.irradiance(point, n);
                        if max3(e) > 0.0 {
                            let k = 1.0 / std::f32::consts::PI;
                            l = add3(l, mul3(throughput, scale3(mul3(rho, e), k)));
                        }
                    }
                }

                // Continue the path.
                let Some(sampled) = bsdf_sample(&material, wo_local, eta, hero, rng) else {
                    break;
                };
                let (wi_local, f, pdf) = match sampled {
                    Sampled::Surface(wi, f, pdf) => (wi, f, pdf),
                    Sampled::Subsurface(entry_weight) => {
                        // The path leaves the surface entirely: it goes into
                        // the object, walks, and comes back out somewhere
                        // else. Everything after this is about the *exit*.
                        throughput = mul3(throughput, entry_weight);
                        let Some(exit) = subsurface_walk(&material, point, n, rng, |p, d| {
                            match scene.intersect(accel, &Ray::new(p, d)) {
                                Landing::Surface { point, normal, .. } => {
                                    Some(((point - p).norm(), normal))
                                }
                                _ => None,
                            }
                        }) else {
                            break;
                        };
                        throughput = mul3(throughput, exit.weight);
                        // Out through the boundary, cosine-distributed: the
                        // index-matched exit the inversion was fitted with,
                        // whose f/pdf is exactly 1.
                        let (t_ax, b_ax) = onb(exit.normal);
                        let c = cosine_hemisphere(rng.f64(), rng.f64());
                        let wi_world = to_world(t_ax, b_ax, exit.normal, c);
                        ray = Ray::new(exit.point + exit.normal * 1e-5, wi_world);
                        // No NEE strategy found this direction — there was no
                        // surface event at the exit to sample lights from — so
                        // an emitter downstream takes full MIS weight, which
                        // is what a specular chain means here.
                        specular_chain = true;
                        prev_bsdf_pdf = 0.0;
                        if depth >= opts.rr_start {
                            let q = max3(throughput).clamp(0.0, 0.95);
                            if (rng.f64() as f32) > q {
                                break;
                            }
                            throughput = scale3(throughput, 1.0 / q);
                        }
                        if max3(throughput) <= 1e-5 {
                            break;
                        }
                        continue;
                    }
                };
                throughput = mul3(throughput, scale3(f, 1.0 / pdf));
                prev_bsdf_pdf = pdf;
                specular_chain = false;

                // A transmitted ray leaves on the far side, so it is offset
                // the other way — and, for a solid, it changes which medium
                // the path is in.
                let transmitted = wi_local.z < 0.0;
                if transmitted && !material.thin_walled {
                    medium = if entering { Some(material) } else { None };
                }
                let wi_world = to_world(frame.t, frame.b, n, wi_local);
                let offset = if transmitted { -n } else { n };
                ray = Ray::new(point + offset * 1e-5, wi_world);

                // Russian roulette.
                if depth >= opts.rr_start {
                    let q = max3(throughput).clamp(0.0, 0.95);
                    if (rng.f64() as f32) > q {
                        break;
                    }
                    throughput = scale3(throughput, 1.0 / q);
                }
                if max3(throughput) <= 1e-5 {
                    break;
                }
            }
        }
    }

    (l, primary)
}

#[inline]
pub(crate) fn vec_to_f32(v: Vec3) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn trace_pixel<G: Geometry>(
    scene: &Scene<G>,
    accel: &SceneAccel<G>,
    caustics: Option<&CausticMap>,
    cam: &Camera,
    opts: &PathTraceOptions,
    width: u32,
    height: u32,
    px: usize,
    py: usize,
    out: &mut PixelOut<'_>,
) -> u32 {
    let aspect = width as f64 / height as f64;
    let spp = opts.spp.max(1);
    let mut rng =
        Rng::new(opts.seed ^ ((py as u64) << 32) ^ (px as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut acc = [0.0f32; 3];
    let mut cov = 0.0f32;
    // Running sums for the estimator's own variance.
    let mut lsum = 0.0f32;
    let mut lsum2 = 0.0f32;
    // Cranley-Patterson rotations for the four camera dimensions, drawn once
    // per pixel. The low-discrepancy point set below is the *same* for every
    // pixel; rotating it by a per-pixel random offset keeps each pixel's
    // stratification intact while decorrelating neighbours, so the residual
    // error looks like noise rather than a repeating pattern locked to the
    // pixel grid. Drawing them from the existing PCG is what keeps seed
    // determinism: no global state, no thread-dependent order.
    let rot = [rng.f64(), rng.f64(), rng.f64(), rng.f64()];

    // Sample in batches so the estimator can be asked, between batches,
    // whether it has already resolved this pixel. `traced` is the count
    // actually spent, which is <= spp under adaptive sampling.
    let mut traced = 0u32;
    'batches: while traced < spp {
        let batch = ADAPTIVE_BATCH.min(spp - traced);
        for k in 0..batch {
            let s = traced + k;
            // Pixel jitter and lens position come from a 4D Halton set rotated
            // into this pixel's frame, not from four fresh uniforms. Four
            // independent uniforms can clump — at low sample counts a purely
            // random jitter leaves visibly uneven coverage of the pixel
            // footprint, and that shows up as extra aliasing on every
            // silhouette. A low-discrepancy set covers the square evenly by
            // construction.
            //
            // Halton rather than Hammersley: Hammersley's first dimension is
            // `s / N`, which needs the final sample count up front. Adaptive
            // sampling does not know it, and a set that changes shape when the
            // loop stops early is worse than a slightly weaker set that is
            // correct at every prefix.
            //
            // The uniforms still go through the reconstruction filter's warp, so
            // the plain mean below is the filtered estimate exactly as before.
            let jx = 0.5
                + opts
                    .filter
                    .warp(cp_rotate(radical_inverse::<2>(s as u64), rot[0]));
            let jy = 0.5
                + opts
                    .filter
                    .warp(cp_rotate(radical_inverse::<3>(s as u64), rot[1]));
            let sx = 2.0 * ((px as f64 + jx) / width as f64) - 1.0;
            let sy = 1.0 - 2.0 * ((py as f64 + jy) / height as f64);
            let (lu, lv) = concentric_disc(
                cp_rotate(radical_inverse::<5>(s as u64), rot[2]),
                cp_rotate(radical_inverse::<7>(s as u64), rot[3]),
            );

            let ray = cam.ray(sx, sy, aspect, lu, lv);
            // The relative clamp's threshold, from what this pixel has measured
            // so far. It is deliberately a *running* mean and not a two-pass
            // estimate: a pixel is its own scale, and one pass is what keeps the
            // integrator streaming.
            let clamp_scale = opts.firefly_clamp_relative.and_then(|k| {
                if s >= opts.firefly_clamp_warmup && s > 0 {
                    Some((k * lsum / s as f32).max(1e-6))
                } else {
                    None
                }
            });
            let (l, primary) = radiance(scene, accel, opts, caustics, clamp_scale, ray, &mut rng);
            acc = add3(acc, l);
            let ls = luminance(l);
            lsum += ls;
            lsum2 += ls * ls;
            if primary.hit {
                cov += 1.0;
            }
            if s == 0 {
                // Guide buffers come from one primary ray, not an average:
                // averaging normals and depths across samples would soften
                // exactly the silhouettes the edge-stopping weights exist to
                // protect.
                out.normal[px * 3] = primary.normal[0];
                out.normal[px * 3 + 1] = primary.normal[1];
                out.normal[px * 3 + 2] = primary.normal[2];
                out.depth[px] = primary.depth;
                out.albedo[px * 3] = primary.albedo[0];
                out.albedo[px * 3 + 1] = primary.albedo[1];
                out.albedo[px * 3 + 2] = primary.albedo[2];
            }
        }
        traced += batch;

        // Stop once the estimator's own error bar says the remaining samples
        // cannot move this pixel by anything a viewer could see. The mean kept
        // below is still the unbiased mean of the samples actually taken, so
        // stopping early costs precision, never accuracy. The floor is
        // non-negotiable: a pixel that happened to draw several near-equal
        // samples early would otherwise report a tiny variance and quit while
        // genuinely unconverged.
        if opts.adaptive && traced >= ADAPTIVE_FLOOR.min(spp) && traced < spp {
            let n = traced as f32;
            let mean = lsum / n;
            // The clamp is load-bearing, not defensive: once the samples agree
            // closely, `lsum2 / n` and `mean * mean` cancel to within f32
            // rounding and can land just below zero, which would put a NaN
            // through the sqrt below — and a NaN compares false, so the pixel
            // would never converge.
            let sample_var = (lsum2 / n - mean * mean).max(0.0) * n / (n - 1.0);
            // Half-width of the 95% confidence interval on the mean.
            let ci = 1.96 * (sample_var / n).sqrt();
            if ci <= ADAPTIVE_TOL * (mean + ADAPTIVE_LUM_FLOOR) {
                break 'batches;
            }
        }
    }

    let inv = 1.0 / traced as f32;
    out.rgb[px * 3] = acc[0] * inv;
    out.rgb[px * 3 + 1] = acc[1] * inv;
    out.rgb[px * 3 + 2] = acc[2] * inv;
    out.alpha[px] = cov * inv;
    // Variance of the *mean*: sample variance / n. A single sample carries
    // no information about its own spread, so fall back to the estimate
    // itself as a scale.
    out.variance[px] = if traced > 1 {
        let n = traced as f32;
        let mean = lsum * inv;
        let sample_var = (lsum2 * inv - mean * mean).max(0.0) * n / (n - 1.0);
        sample_var / n
    } else {
        let mean = lsum;
        mean * mean
    };
    traced
}

/// Render `scene` from `cam` into a linear-space [`Film`].
///
/// Scanlines are traced in parallel. Each pixel's RNG is seeded from its
/// coordinates and the option seed, so output is deterministic and
/// independent of thread scheduling.
///
/// When [`PathTraceOptions::denoise`] is set (the default), the film is run
/// through [`denoise`] before returning. Pass `denoise: false` for a
/// reference render.
pub fn render<G: Geometry + Send + Sync>(
    scene: &Scene<G>,
    cam: &Camera,
    width: u32,
    height: u32,
    opts: &PathTraceOptions,
) -> Film {
    render_with_caustics(scene, cam, width, height, opts, None)
}

/// [`render`], plus a caustic map read as direct light at every diffuse hit.
///
/// The map is built once by [`crate::caustics::trace`] and handed in here.
/// It is a separate argument rather than a field on [`Scene`] or
/// [`PathTraceOptions`] because it is neither: the scene does not own it (it
/// is derived from the scene) and the options are `Copy`.
///
/// Passing `None` is exactly [`render`], to the bit.
pub fn render_with_caustics<G: Geometry + Send + Sync>(
    scene: &Scene<G>,
    cam: &Camera,
    width: u32,
    height: u32,
    opts: &PathTraceOptions,
    caustics: Option<&CausticMap>,
) -> Film {
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(not(target_arch = "wasm32"))]
    use rayon::prelude::*;

    // One TLAS for the whole frame: every ray, primary and shadow, traverses
    // it instead of scanning `scene.objects` linearly.
    let accel = SceneAccel::build(scene);

    let mut rgb = vec![0.0f32; (width * height * 3) as usize];
    let mut alpha = vec![0.0f32; (width * height) as usize];
    let mut normal = vec![0.0f32; (width * height * 3) as usize];
    let mut depth = vec![0.0f32; (width * height) as usize];
    let mut albedo = vec![0.0f32; (width * height * 3) as usize];
    let mut variance = vec![0.0f32; (width * height) as usize];

    let w3 = width as usize * 3;
    let w1 = width as usize;
    film_rows!(rgb, w3)
        .zip(film_rows!(alpha, w1))
        .zip(film_rows!(normal, w3))
        .zip(film_rows!(depth, w1))
        .zip(film_rows!(albedo, w3))
        .zip(film_rows!(variance, w1))
        .enumerate()
        .for_each(|(py, (((((row, arow), nrow), drow), brow), vrow))| {
            let mut out = PixelOut {
                rgb: row,
                alpha: arow,
                normal: nrow,
                depth: drow,
                albedo: brow,
                variance: vrow,
            };
            for px in 0..width as usize {
                let _ = trace_pixel(
                    scene, &accel, caustics, cam, opts, width, height, px, py, &mut out,
                );
            }
        });

    let mut film = Film {
        width,
        height,
        rgb,
        alpha,
        normal,
        depth,
        albedo,
        variance,
    };
    if opts.denoise {
        denoise(&mut film, opts);
    }
    film
}

/// Re-render only the pixels inside `rects`, leaving the rest of `film`
/// exactly as it was.
///
/// Each rect is `[x, y, w, h]` in pixels, top-left origin, and is clipped to
/// the film. A pixel inside the union is traced with the same seed, the same
/// sample sequence and the same integrator [`render`] would have given it, so
/// the result is *bit-identical* to the corresponding pixels of a full render
/// with the same options — which is the whole point: a caller can re-trace the
/// region under a moving widget, or a tile the user is zoomed into, and drop
/// the result straight into the frame it already has without a seam.
///
/// Rows within each rect are traced in parallel. Overlapping rects simply
/// trace their shared pixels more than once, to the same values.
///
/// Unlike [`render`] this never denoises. The à-trous filter reads a
/// neighbourhood well outside any rect, so filtering a masked pass would blend
/// fresh radiance into stale and put a visible seam at the rect's edge; a
/// caller that wants a filtered frame runs [`denoise`] over the whole film
/// once the patches are in. `opts.denoise` is therefore ignored here, and
/// comparing against a reference means comparing against a `denoise: false`
/// render.
///
/// The film must already be the size the camera is being sampled at —
/// `film.width` and `film.height` are the resolution, not `rects`.
pub fn render_into<G: Geometry + Send + Sync>(
    scene: &Scene<G>,
    cam: &Camera,
    film: &mut Film,
    opts: &PathTraceOptions,
    rects: &[[u32; 4]],
) {
    render_into_with_caustics(scene, cam, film, opts, rects, None)
}

/// [`render_into`] with a caustic map, the patch-render counterpart of
/// [`render_with_caustics`].
#[allow(clippy::too_many_arguments)]
pub fn render_into_with_caustics<G: Geometry + Send + Sync>(
    scene: &Scene<G>,
    cam: &Camera,
    film: &mut Film,
    opts: &PathTraceOptions,
    rects: &[[u32; 4]],
    caustics: Option<&CausticMap>,
) {
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(not(target_arch = "wasm32"))]
    use rayon::prelude::*;

    let (width, height) = (film.width, film.height);
    if width == 0 || height == 0 {
        return;
    }
    let accel = SceneAccel::build(scene);
    let w3 = width as usize * 3;
    let w1 = width as usize;

    for r in rects {
        // Clip to the film. A rect that starts past the edge, or is empty,
        // contributes nothing rather than panicking on a caller's arithmetic.
        let x0 = r[0].min(width) as usize;
        let y0 = r[1].min(height) as usize;
        let x1 = r[0].saturating_add(r[2]).min(width) as usize;
        let y1 = r[1].saturating_add(r[3]).min(height) as usize;
        if x0 >= x1 || y0 >= y1 {
            continue;
        }

        // Row-chunked so every parallel task owns a disjoint slice of each
        // buffer; the columns outside the rect are simply never written.
        film_rows!(film.rgb, w3)
            .zip(film_rows!(film.alpha, w1))
            .zip(film_rows!(film.normal, w3))
            .zip(film_rows!(film.depth, w1))
            .zip(film_rows!(film.albedo, w3))
            .zip(film_rows!(film.variance, w1))
            .enumerate()
            .skip(y0)
            .take(y1 - y0)
            .for_each(|(py, (((((row, arow), nrow), drow), brow), vrow))| {
                let mut out = PixelOut {
                    rgb: row,
                    alpha: arow,
                    normal: nrow,
                    depth: drow,
                    albedo: brow,
                    variance: vrow,
                };
                for px in x0..x1 {
                    let _ = trace_pixel(
                        scene, &accel, caustics, cam, opts, width, height, px, py, &mut out,
                    );
                }
            });
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

    #[test]
    fn an_opaque_scene_occludes_exactly_as_the_any_hit_test_did() {
        let mut scene = open_scene(vec![panel(Point3::new(0.0, 0.0, 6.0), [10.0; 3], 1.0)]);
        scene.objects.push(Object::new(
            Arc::new(Bvh::build(cube_mesh())),
            Pbr::plastic([0.8, 0.3, 0.2], 0.35, 0.0),
        ));
        scene.objects.push(Object::new(
            Arc::new(Bvh::build(pane_mesh(3.0, 2.0))),
            // Transmissive but *not* thin-walled: a solid, which still
            // blocks. The caustic pass is what carries light through those.
            Pbr::glass(1.5, 0.0),
        ));
        let accel = SceneAccel::build(&scene);
        let mut rng = Rng::new(12345);
        let mut checked = 0;
        for _ in 0..4000 {
            let o = Point3::new(
                20.0 * rng.f64() - 5.0,
                20.0 * rng.f64() - 5.0,
                20.0 * rng.f64() - 5.0,
            );
            let d = Vec3::new(
                2.0 * rng.f64() - 1.0,
                2.0 * rng.f64() - 1.0,
                2.0 * rng.f64() - 1.0,
            );
            if d.norm() < 1e-6 {
                continue;
            }
            let d = d.normalize();
            let dist = 30.0 * rng.f64();
            let old = accel
                .tlas
                .occluded_range(&Ray::new(o, d), 1e-6, dist - 1e-6);
            let new = scene.shadow_transmittance(&accel, o, d, dist).is_none();
            assert_eq!(old, new, "from {o:?} along {d:?} for {dist}");
            checked += 1;
        }
        assert!(checked > 3000);
    }

    /// A shadow ray must see *through* a pane of glass, dimmed by exactly the
    /// factor the thin-walled BSDF applies to a refracted path.
    ///
    /// The old material-blind any-hit test returned black here, which is why
    /// a room could not be lit through a window at any sample count.
    #[test]
    fn next_event_passes_through_a_thin_pane() {
        let light = panel(Point3::new(0.0, 0.0, 6.0), [10.0; 3], 0.35);
        let open = open_scene(vec![light]);
        let mut glazed = open_scene(vec![light]);
        glazed.objects.push(Object::new(
            Arc::new(Bvh::build(pane_mesh(3.0, 4.0))),
            window_glass(),
        ));

        let n = 200_000;
        let bare = nee_mean(&open, n);
        let through = nee_mean(&glazed, n);
        assert!(bare > 0.0, "the open scene must be lit at all");

        // The light is small and nearly overhead, so every shadow ray meets
        // the pane within a few degrees of normal incidence.
        let f = fresnel_dielectric(1.0, 1.5);
        let expected = (1.0 - f) as f64;
        let ratio = through / bare;
        assert!(
            (ratio - expected).abs() < 0.02 * expected,
            "pane transmittance {ratio} is not within 2% of {expected}"
        );
    }

    /// A stack of panes deeper than the cap is an honest blocker, so the
    /// traversal cannot run away.
    #[test]
    fn a_shadow_ray_gives_up_past_the_sheet_cap() {
        let light = panel(Point3::new(0.0, 0.0, 6.0), [10.0; 3], 0.35);
        let mut stacked = open_scene(vec![light]);
        for i in 0..(MAX_SHADOW_SHEETS + 1) {
            stacked.objects.push(Object::new(
                Arc::new(Bvh::build(pane_mesh(1.0 + i as f64 * 0.5, 4.0))),
                window_glass(),
            ));
        }
        assert_eq!(nee_mean(&stacked, 4_000), 0.0);
    }

    /// A frosted pane still blocks: the straight-line shadow ray is only the
    /// right answer in the smooth limit, so a wide lobe tapers it away.
    #[test]
    fn a_frosted_pane_still_blocks_the_shadow_ray() {
        let light = panel(Point3::new(0.0, 0.0, 6.0), [10.0; 3], 0.35);
        let mut frosted = open_scene(vec![light]);
        frosted.objects.push(Object::new(
            Arc::new(Bvh::build(pane_mesh(3.0, 4.0))),
            Pbr {
                roughness: 1.0,
                ..window_glass()
            },
        ));
        assert_eq!(nee_mean(&frosted, 4_000), 0.0);
    }

    /// The *whole* estimator — NEE and BSDF sampling combined under MIS —
    /// must land on the same `(1 − F)` factor the single strategy does. If
    /// the two disagreed, MIS would double count the refracted path in one
    /// direction and lose it in the other; agreement is the check.
    #[test]
    fn a_converged_render_through_a_pane_matches_the_single_strategy() {
        let floor = || {
            Object::new(
                Arc::new(Bvh::build(pane_mesh(0.0, 6.0))),
                Pbr {
                    base_color: [1.0; 3],
                    metallic: 0.0,
                    roughness: 1.0,
                    specular: 0.0,
                    ..Pbr::default()
                },
            )
        };
        let light = panel(Point3::new(0.0, 0.0, 6.0), [10.0; 3], 0.35);
        let camera = Camera::look_at(
            Point3::new(0.0, -0.01, 2.0),
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            30.0,
        );
        let opts = PathTraceOptions {
            spp: 400,
            max_depth: 4,
            firefly_clamp: None,
            denoise: false,
            show_background: false,
            ..PathTraceOptions::default()
        };

        let mean = |with_pane: bool| -> f64 {
            let mut scene = open_scene(vec![light]);
            scene.objects.push(floor());
            if with_pane {
                scene.objects.push(Object::new(
                    Arc::new(Bvh::build(pane_mesh(3.0, 5.0))),
                    window_glass(),
                ));
            }
            let film = render(&scene, &camera, 24, 24, &opts);
            let mut sum = 0.0f64;
            for i in 0..(24 * 24) {
                sum +=
                    luminance([film.rgb[i * 3], film.rgb[i * 3 + 1], film.rgb[i * 3 + 2]]) as f64;
            }
            sum / (24.0 * 24.0)
        };

        let bare = mean(false);
        let glazed = mean(true);
        assert!(bare > 0.0);
        let expected = (1.0 - fresnel_dielectric(1.0, 1.5)) as f64;
        let ratio = glazed / bare;
        assert!(
            (ratio - expected).abs() < 0.03 * expected,
            "converged ratio {ratio} is not within 3% of {expected}"
        );
    }

    /// One light per bounce, drawn from the power table and divided by its
    /// pick probability, must integrate to the same direct lighting as
    /// shadow-raying every light. Two lights of very different power, so a
    /// uniform pick would not have been enough.
    #[test]
    fn one_light_per_bounce_matches_all_lights_in_expectation() {
        let scene = open_scene(vec![
            panel(Point3::new(-2.0, 0.0, 4.0), [12.0, 11.0, 10.0], 1.5),
            panel(Point3::new(3.0, 1.0, 5.0), [0.6, 0.7, 1.4], 0.7),
        ]);
        let accel = SceneAccel::build(&scene);
        let n = 400_000;
        let a = nee_unweighted_mean(&scene, &accel, true, n);
        let b = nee_unweighted_mean(&scene, &accel, false, n);
        for c in 0..3 {
            let rel = (a[c] - b[c]).abs() / b[c].abs().max(1e-6);
            assert!(
                rel < 0.02,
                "channel {c}: one-light mean {} vs all-lights mean {} (rel {rel})",
                a[c],
                b[c]
            );
        }
    }

    /// The power table must actually be power-weighted: the bright panel is
    /// picked far more often than the dim one, and the probabilities sum to 1.
    #[test]
    fn light_table_is_power_weighted() {
        let scene = open_scene(vec![
            panel(Point3::new(-2.0, 0.0, 4.0), [12.0, 11.0, 10.0], 1.5),
            panel(Point3::new(3.0, 1.0, 5.0), [0.6, 0.7, 1.4], 0.7),
        ]);
        let accel = SceneAccel::build(&scene);
        let p0 = accel.light_pick_pdf(0);
        let p1 = accel.light_pick_pdf(1);
        assert!((p0 + p1 - 1.0).abs() < 1e-5, "pick pdf must sum to 1");
        assert!(p0 > 0.9, "the bright, large panel should dominate: {p0}");
        // Drawing follows the table.
        let mut rng = Rng::new(7);
        let mut hits = [0u32; 2];
        for _ in 0..20_000 {
            let (i, _) = accel.pick_light(rng.f64() as f32).unwrap();
            hits[i] += 1;
        }
        let frac0 = hits[0] as f32 / 20_000.0;
        assert!((frac0 - p0).abs() < 0.02, "draw {frac0} vs table {p0}");
    }

    /// A full render of a multi-light scene must still land on the same image
    /// the all-lights estimator gives, within Monte Carlo noise.
    #[test]
    fn multi_light_render_matches_reference_mean() {
        // Exercise the reference path so it cannot rot.
        let scene = open_scene(vec![
            panel(Point3::new(-2.0, 0.0, 4.0), [8.0, 8.0, 8.0], 1.2),
            panel(Point3::new(3.0, 1.0, 5.0), [2.0, 2.0, 2.0], 1.0),
        ]);
        let accel = SceneAccel::build(&scene);
        let nrm = Vec3::new(0.0, 0.0, 1.0);
        let frame = shading_frame(nrm, None);
        let m = Pbr::default();
        let wo_local = to_local(frame.t, frame.b, nrm, nrm);
        let mut rng = Rng::new(3);
        let mut r = [0.0f64; 3];
        let mut o = [0.0f64; 3];
        let n = 200_000;
        for _ in 0..n {
            let a = scene.sample_lights(
                &accel,
                Point3::new(0.0, 0.0, 0.0),
                &frame,
                wo_local,
                &m,
                1.0,
                0.0,
                &mut rng,
            );
            let b = sample_all_lights_reference(
                &scene,
                &accel,
                Point3::new(0.0, 0.0, 0.0),
                &frame,
                wo_local,
                &m,
                1.0,
                &mut rng,
            );
            for c in 0..3 {
                r[c] += a[c] as f64;
                o[c] += b[c] as f64;
            }
        }
        // MIS weights differ slightly between the two (the light pdf carries
        // the pick probability), so this is a loose sanity band, not equality.
        for c in 0..3 {
            let rel = (r[c] - o[c]).abs() / (o[c] / n as f64).abs().max(1e-9) / n as f64;
            assert!(
                rel < 0.06,
                "channel {c}: {} vs {} (rel {rel})",
                r[c] / n as f64,
                o[c] / n as f64
            );
        }
    }

    /// A masked pass must reproduce the full render exactly — not "within
    /// noise", *bit for bit*. That is only true if the per-pixel seed depends
    /// on nothing but the pixel, which is the property that lets a caller drop
    /// a re-traced patch into a frame it already has without a seam.
    #[test]
    fn render_into_is_bit_identical_to_the_full_render() {
        let scene = test_scene();
        let cam = test_camera();
        let opts = PathTraceOptions {
            spp: 3,
            max_depth: 3,
            denoise: false,
            ..Default::default()
        };
        let (w, h) = (40u32, 32u32);
        let full = render(&scene, &cam, w, h, &opts);

        // Rects that clip, touch the edges, and overlap each other.
        let rects = [
            [3, 4, 10, 9],
            [9, 6, 12, 20],
            [0, 0, 5, 5],
            [35, 28, 20, 20],
        ];
        let mut patched = Film::new(w, h);
        render_into(&scene, &cam, &mut patched, &opts, &rects);

        let inside = |px: u32, py: u32| {
            rects
                .iter()
                .any(|r| px >= r[0] && py >= r[1] && px < r[0] + r[2] && py < r[1] + r[3])
        };
        let mut covered = 0usize;
        for py in 0..h {
            for px in 0..w {
                let i = (py * w + px) as usize;
                if inside(px, py) {
                    covered += 1;
                    for c in 0..3 {
                        assert_eq!(
                            patched.rgb[i * 3 + c].to_bits(),
                            full.rgb[i * 3 + c].to_bits(),
                            "pixel ({px}, {py}) channel {c}: masked render gave {} \
                             where the full render gave {}. The per-pixel seed \
                             must not depend on anything but the pixel.",
                            patched.rgb[i * 3 + c],
                            full.rgb[i * 3 + c],
                        );
                        assert_eq!(patched.albedo[i * 3 + c], full.albedo[i * 3 + c]);
                        assert_eq!(patched.normal[i * 3 + c], full.normal[i * 3 + c]);
                    }
                    assert_eq!(patched.alpha[i], full.alpha[i]);
                    assert_eq!(patched.depth[i], full.depth[i]);
                    assert_eq!(patched.variance[i].to_bits(), full.variance[i].to_bits());
                } else {
                    // Outside the union nothing was touched at all.
                    assert_eq!(
                        (patched.rgb[i * 3], patched.alpha[i], patched.depth[i]),
                        (0.0, 0.0, 0.0),
                        "pixel ({px}, {py}) is outside every rect and was written anyway",
                    );
                }
            }
        }
        assert!(covered > 300, "the rects covered only {covered} px");
        assert!(
            covered < (w * h) as usize,
            "the rects covered the whole film"
        );
    }

    /// A patched frame is the frame: re-tracing every rect of a partition of
    /// the film reconstructs the full render exactly.
    #[test]
    fn tiling_the_film_with_rects_reconstructs_the_whole_render() {
        let scene = test_scene();
        let cam = test_camera();
        let opts = PathTraceOptions {
            spp: 2,
            max_depth: 2,
            denoise: false,
            ..Default::default()
        };
        let (w, h) = (24u32, 24u32);
        let full = render(&scene, &cam, w, h, &opts);
        let rects: Vec<[u32; 4]> = (0..3)
            .flat_map(|i| (0..3).map(move |j| [i * 8, j * 8, 8, 8]))
            .collect();
        let mut patched = Film::new(w, h);
        render_into(&scene, &cam, &mut patched, &opts, &rects);
        assert_eq!(patched.rgb, full.rgb);
        assert_eq!(patched.depth, full.depth);
    }

    /// Degenerate and out-of-bounds rects are clipped, not panics.
    #[test]
    fn render_into_clips_rects_to_the_film() {
        let scene = test_scene();
        let cam = test_camera();
        let opts = PathTraceOptions {
            spp: 1,
            max_depth: 1,
            denoise: false,
            ..Default::default()
        };
        let mut film = Film::new(16, 16);
        render_into(
            &scene,
            &cam,
            &mut film,
            &opts,
            &[
                [0, 0, 0, 0],
                [20, 20, 4, 4],
                [14, 14, u32::MAX, u32::MAX],
                [0, 0, 16, 16],
            ],
        );
        assert_eq!(film.rgb.len(), 16 * 16 * 3);
    }

    #[test]
    fn renders_non_empty() {
        let scene = test_scene();
        let cam = test_camera();
        let film = render(
            &scene,
            &cam,
            24,
            24,
            &PathTraceOptions {
                spp: 4,
                ..Default::default()
            },
        );
        assert_eq!(film.rgb.len(), 24 * 24 * 3);
        let lit = film.rgb.iter().filter(|v| **v > 0.0).count();
        assert!(lit > 0, "path tracer produced an entirely black frame");
    }

    /// `Object::transform` must actually place the BLAS. This is what an
    /// animated render leans on: the same BVH, re-posed per frame.
    #[test]
    fn object_transform_moves_the_subject() {
        let cam = test_camera();
        let coverage = |t: Transform| {
            let scene = Scene {
                objects: vec![Object::placed(
                    Arc::new(Bvh::build(cube_mesh())),
                    Pbr::plastic([0.8, 0.3, 0.2], 0.35, 0.0),
                    t,
                )],
                // No area lights: they are hittable geometry, and a rig that
                // stays put while the cube moves would muddy the coverage
                // signal this test reads.
                lights: Vec::new(),
                env: Environment::default(),
                sun: None,
                ground: None,
                splats: None,
            };
            let film = render(
                &scene,
                &cam,
                32,
                32,
                &PathTraceOptions {
                    spp: 2,
                    ..Default::default()
                },
            );
            film.alpha.iter().map(|a| *a > 0.5).collect::<Vec<_>>()
        };
        let here = coverage(Transform::identity());
        // Far enough out of frame that nothing overlaps.
        let there = coverage(Transform::translation(400.0, 0.0, 0.0));
        assert!(here.iter().any(|c| *c), "identity placement lost the cube");
        assert!(
            !there.iter().any(|c| *c),
            "translated placement was ignored — the cube stayed put"
        );
    }

    #[test]
    fn subject_is_covered() {
        let scene = test_scene();
        let cam = test_camera();
        let film = render(
            &scene,
            &cam,
            32,
            32,
            &PathTraceOptions {
                spp: 4,
                ..Default::default()
            },
        );
        let covered = film.alpha.iter().filter(|a| **a > 0.5).count();
        assert!(
            covered > 40,
            "expected the cube to cover a chunk of frame, got {covered}"
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let scene = test_scene();
        let cam = test_camera();
        let o = PathTraceOptions {
            spp: 2,
            ..Default::default()
        };
        let a = render(&scene, &cam, 16, 16, &o);
        let b = render(&scene, &cam, 16, 16, &o);
        assert_eq!(a.rgb, b.rgb, "render must be seed-deterministic");
    }

    /// The BSDF sampling PDF must match the analytic PDF used by MIS, or
    /// light sampling and BSDF sampling silently disagree and the image is
    /// energy-wrong in a way that is hard to see by eye.

    // ─── the splat volume in the integrator ───────────────────────────────
    //
    // A splat cloud is composited, not shaded, so what these check is the
    // arithmetic of the walk — that `C += T·α·c; T *= (1 − α)` happens in
    // front of the analytic scene, in the right order, and on shadow rays.

    mod splat_volume {
        use super::*;
        use crate::splats::Splats;

        /// The degree-0 SH coefficient that makes a splat render as `c`.
        fn dc(c: [f32; 3]) -> [f32; 3] {
            const SH_C0: f32 = 0.282_094_79;
            [
                (c[0] - 0.5) / SH_C0,
                (c[1] - 0.5) / SH_C0,
                (c[2] - 0.5) / SH_C0,
            ]
        }

        /// Isotropic splats on the z axis, `(z, opacity, colour)` each.
        fn cloud(items: &[(f32, f32, [f32; 3])]) -> Arc<Bvh<Splats>> {
            let positions: Vec<[f32; 3]> = items.iter().map(|it| [0.0, 0.0, it.0]).collect();
            let scales = vec![[0.2f32; 3]; items.len()];
            let quats = vec![[1.0f32, 0.0, 0.0, 0.0]; items.len()];
            let opacities: Vec<f32> = items.iter().map(|it| it.1).collect();
            let sh: Vec<[f32; 3]> = items.iter().map(|it| dc(it.2)).collect();
            Arc::new(Bvh::build(Splats::from_parts(
                &positions, &scales, &quats, &opacities, &sh,
            )))
        }

        /// An emissive floor at z = 0 under a black sky: a "plane" whose
        /// radiance is exactly 1, so anything the camera reads that is not 1
        /// came from the cloud.
        fn scene(splats: Option<Arc<Bvh<Splats>>>) -> Scene<TriMesh> {
            Scene {
                objects: Vec::new(),
                lights: Vec::new(),
                env: Environment::constant([0.0; 3]),
                sun: None,
                ground: Some(Ground {
                    z: 0.0,
                    material: Pbr {
                        base_color: [0.0; 3],
                        roughness: 1.0,
                        emissive: [1.0; 3],
                        ..Default::default()
                    },
                    shadow_catcher: false,
                }),
                splats,
            }
        }

        /// The radiance of one ray straight down the z axis at the floor.
        fn down(scene: &Scene<TriMesh>) -> [f32; 3] {
            let accel = SceneAccel::build(scene);
            let opts = PathTraceOptions::default();
            let ray = Ray::new(Point3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
            let mut rng = Rng::new(7);
            radiance(scene, &accel, &opts, None, None, ray, &mut rng).0
        }

        #[test]
        fn an_opaque_splat_hides_the_plane() {
            let s = scene(Some(cloud(&[(2.5, 1.0, [0.25, 0.5, 0.75])])));
            let l = down(&s);
            assert!((l[0] - 0.25).abs() < 1e-4, "{l:?}");
            assert!((l[1] - 0.5).abs() < 1e-4, "{l:?}");
            assert!((l[2] - 0.75).abs() < 1e-4, "the floor's 1.0 is gone: {l:?}");
        }

        #[test]
        fn a_half_transparent_splat_composites_fifty_fifty() {
            let s = scene(Some(cloud(&[(2.5, 0.5, [0.0, 0.0, 0.0])])));
            let l = down(&s);
            // Black cloud at α = ½ over an emissive floor at 1: half the
            // floor survives, and none of the cloud's own colour shows.
            for ch in 0..3 {
                assert!((l[ch] - 0.5).abs() < 1e-4, "{l:?}");
            }
            // And with a white cloud instead, the two halves add back to one.
            let s = scene(Some(cloud(&[(2.5, 0.5, [1.0, 1.0, 1.0])])));
            let l = down(&s);
            for ch in 0..3 {
                assert!((l[ch] - 1.0).abs() < 1e-4, "{l:?}");
            }
        }

        #[test]
        fn the_nearer_splat_dominates() {
            // Red in front at z = 3, blue behind at z = 1, both α = ½.
            let s = scene(Some(cloud(&[
                (3.0, 0.5, [1.0, 0.0, 0.0]),
                (1.0, 0.5, [0.0, 0.0, 1.0]),
            ])));
            let l = down(&s);
            // Front to back: ½·red, then ½·½·blue, then ¼ of the floor —
            // and the floor is white, so it adds ¼ to every channel.
            assert!((l[0] - (0.5 + 0.25)).abs() < 1e-4, "red at full T: {l:?}");
            assert!((l[2] - (0.25 + 0.25)).abs() < 1e-4, "blue at half T: {l:?}");
            assert!(l[0] > l[2], "the nearer colour weighs more: {l:?}");
            // Green sees only the floor's quarter.
            assert!((l[1] - 0.25).abs() < 1e-4, "{l:?}");
            // Swapping the depths swaps the weights, which is the whole test.
            let s = scene(Some(cloud(&[
                (3.0, 0.5, [0.0, 0.0, 1.0]),
                (1.0, 0.5, [1.0, 0.0, 0.0]),
            ])));
            let l2 = down(&s);
            assert!((l2[2] - 0.75).abs() < 1e-4, "{l2:?}");
            assert!((l2[0] - 0.5).abs() < 1e-4, "{l2:?}");
            assert!(l2[2] > l2[0], "swapping the depths swaps the weights");
        }

        #[test]
        fn a_shadow_ray_is_attenuated_by_the_cloud() {
            let s = scene(Some(cloud(&[(2.5, 0.5, [0.0; 3])])));
            let accel = SceneAccel::build(&s);
            // Upward, from just under the cloud past it — the ground plane is
            // below the origin, so it does not block.
            let tr = s
                .shadow_transmittance(
                    &accel,
                    Point3::new(0.0, 0.0, 1.0),
                    Vec3::new(0.0, 0.0, 1.0),
                    10.0,
                )
                .expect("a half-transparent splat is not a blocker");
            for ch in 0..3 {
                assert!((tr[ch] - 0.5).abs() < 1e-4, "{tr:?}");
            }
            // Two of them multiply.
            let s = scene(Some(cloud(&[(2.5, 0.5, [0.0; 3]), (3.5, 0.5, [0.0; 3])])));
            let accel = SceneAccel::build(&s);
            let tr = s
                .shadow_transmittance(
                    &accel,
                    Point3::new(0.0, 0.0, 1.0),
                    Vec3::new(0.0, 0.0, 1.0),
                    10.0,
                )
                .expect("still not a blocker");
            assert!((tr[0] - 0.25).abs() < 1e-4, "{tr:?}");
            // An opaque one is.
            let s = scene(Some(cloud(&[(2.5, 1.0, [0.0; 3])])));
            let accel = SceneAccel::build(&s);
            assert!(
                s.shadow_transmittance(
                    &accel,
                    Point3::new(0.0, 0.0, 1.0),
                    Vec3::new(0.0, 0.0, 1.0),
                    10.0,
                )
                .is_none(),
                "an opaque splat stops the light"
            );
        }

        #[test]
        fn a_bounce_ray_that_misses_everything_returns_the_cloud() {
            // The environment-with-depth claim: no analytic geometry at all,
            // so the only thing a ray can find is the captured field.
            let mut s = scene(Some(cloud(&[(2.5, 1.0, [0.3, 0.4, 0.5])])));
            s.ground = None;
            let l = down(&s);
            assert!((l[0] - 0.3).abs() < 1e-4, "{l:?}");
            assert!((l[2] - 0.5).abs() < 1e-4, "{l:?}");
        }
    }

    // ─── low-discrepancy camera sampling and adaptive sampling ────────────

    #[test]
    fn radical_inverse_matches_hand_computed_values() {
        // Base 2: 1 -> 0.1b = 1/2, 2 -> 0.01b = 1/4, 3 -> 0.11b = 3/4.
        assert_eq!(radical_inverse::<2>(0), 0.0);
        assert!((radical_inverse::<2>(1) - 0.5).abs() < 1e-12);
        assert!((radical_inverse::<2>(2) - 0.25).abs() < 1e-12);
        assert!((radical_inverse::<2>(3) - 0.75).abs() < 1e-12);
        // Base 3: 1 -> 1/3, 2 -> 2/3, 4 = 11_3 -> 0.11_3 = 4/9.
        assert!((radical_inverse::<3>(1) - 1.0 / 3.0).abs() < 1e-12);
        assert!((radical_inverse::<3>(2) - 2.0 / 3.0).abs() < 1e-12);
        assert!((radical_inverse::<3>(4) - 4.0 / 9.0).abs() < 1e-12);
        // The rotation stays on the torus whatever the offset.
        for &x in &[0.0, 0.25, 0.99] {
            for &o in &[0.0, 0.5, 0.999] {
                let v = cp_rotate(x, o);
                assert!((0.0..1.0).contains(&v), "cp_rotate({x}, {o}) = {v}");
            }
        }
    }

    /// The whole point of the point set: no gaps and no clumps. A purely
    /// random 2D sample would routinely leave a stratum empty at these
    /// counts, which is the aliasing this replaced.
    #[test]
    fn camera_point_set_covers_every_stratum() {
        let n = 64u32;
        let sample = |s: u32, ox: f64, oy: f64| {
            (
                cp_rotate(radical_inverse::<2>(s as u64), ox),
                cp_rotate(radical_inverse::<3>(s as u64), oy),
            )
        };
        for &(ox, oy) in &[(0.0, 0.0), (0.317, 0.61), (0.94, 0.02)] {
            let mut hits = [[0u32; 8]; 8];
            for s in 0..n {
                let (x, y) = sample(s, ox, oy);
                assert!((0.0..1.0).contains(&x) && (0.0..1.0).contains(&y));
                hits[(y * 8.0) as usize][(x * 8.0) as usize] += 1;
            }
            let worst = hits.iter().flatten().copied().max().unwrap();
            assert!(
                worst <= 3,
                "rotated set clumped {worst} samples in a stratum"
            );
        }
    }

    /// The Halton camera set beats four fresh uniforms at the job the camera
    /// dimensions actually do: estimating how much of a pixel's footprint a
    /// silhouette covers, on a scene that is flat either side of the edge.
    ///
    /// Measured per pixel and pooled, because the per-pixel Cranley-Patterson
    /// rotation makes a *single* sample of the Halton set exactly as random
    /// as a uniform draw — the two sets only separate once a pixel takes more
    /// than one, which is the smallest count at which the comparison means
    /// anything. Each "pixel" here is one draw of the rotation.
    #[test]
    fn the_halton_camera_set_beats_random_jitter_on_a_flat_edge() {
        // Coverage of the unit pixel square by the half-plane x + y < c, for
        // a random edge offset c per pixel — a silhouette crossing the pixel
        // footprint anywhere. The exact area is known, and the estimator is
        // the fraction of samples that land under the edge.
        let n = 16u32;
        let pixels = 8192u32;
        let mut halton_err = 0.0f64;
        let mut random_err = 0.0f64;
        for pixel in 0..pixels {
            let mut rng = Rng::new(0xA17E_u64 ^ pixel as u64);
            let rot = [rng.f64(), rng.f64()];
            let c = 2.0 * rng.f64();
            let exact = if c <= 1.0 {
                0.5 * c * c
            } else {
                1.0 - 0.5 * (2.0 - c) * (2.0 - c)
            };
            let mut h = 0.0f64;
            let mut r = 0.0f64;
            for s in 0..n {
                let hx = cp_rotate(radical_inverse::<2>(s as u64), rot[0]);
                let hy = cp_rotate(radical_inverse::<3>(s as u64), rot[1]);
                if hx + hy < c {
                    h += 1.0;
                }
                if rng.f64() + rng.f64() < c {
                    r += 1.0;
                }
            }
            halton_err += (h / n as f64 - exact).powi(2);
            random_err += (r / n as f64 - exact).powi(2);
        }
        let halton_rmse = (halton_err / pixels as f64).sqrt();
        let random_rmse = (random_err / pixels as f64).sqrt();
        eprintln!(
            "edge coverage RMSE at {n} spp: halton {halton_rmse:.5}, random {random_rmse:.5}"
        );
        assert!(
            halton_rmse < random_rmse,
            "the low-discrepancy set is no better than random: {halton_rmse} vs {random_rmse}"
        );
    }

    /// Below the floor, adaptive sampling must be a no-op — not "almost" a
    /// no-op. A low-spp render is exactly where an early stop would do the
    /// most damage, so the option must not touch it at all.
    #[test]
    fn adaptive_is_inert_below_the_sample_floor() {
        let scene = test_scene();
        let cam = test_camera();
        let base = PathTraceOptions {
            spp: ADAPTIVE_FLOOR,
            max_depth: 3,
            denoise: false,
            seed: 11,
            ..Default::default()
        };
        let fixed = render(
            &scene,
            &cam,
            16,
            16,
            &PathTraceOptions {
                adaptive: false,
                ..base
            },
        );
        let adaptive = render(
            &scene,
            &cam,
            16,
            16,
            &PathTraceOptions {
                adaptive: true,
                ..base
            },
        );
        assert_eq!(
            fixed.rgb, adaptive.rgb,
            "adaptive sampling fired at or below the floor"
        );
    }

    /// Total samples spent, tracing every pixel the way [`render`] does.
    fn spend(scene: &Scene<TriMesh>, cam: &Camera, w: u32, h: u32, opts: &PathTraceOptions) -> u64 {
        let accel = SceneAccel::build(scene);
        let mut total = 0u64;
        for py in 0..h as usize {
            let mut rgb = vec![0.0f32; w as usize * 3];
            let mut alpha = vec![0.0f32; w as usize];
            let mut normal = vec![0.0f32; w as usize * 3];
            let mut depth = vec![0.0f32; w as usize];
            let mut albedo = vec![0.0f32; w as usize * 3];
            let mut variance = vec![0.0f32; w as usize];
            let mut out = PixelOut {
                rgb: &mut rgb,
                alpha: &mut alpha,
                normal: &mut normal,
                depth: &mut depth,
                albedo: &mut albedo,
                variance: &mut variance,
            };
            for px in 0..w as usize {
                total += trace_pixel(scene, &accel, None, cam, opts, w, h, px, py, &mut out) as u64;
            }
        }
        total
    }

    /// A converged pixel keeps the unbiased mean of the samples it took, so
    /// the adaptive film must land on the fixed-count film's estimate — it
    /// just gets there for less.
    #[test]
    fn adaptive_matches_the_fixed_count_mean_for_fewer_samples() {
        // A flat, softly lit scene: nearly every pixel resolves early, which
        // is exactly the case adaptivity exists for.
        let scene = Scene::<TriMesh> {
            objects: Vec::new(),
            lights: studio_rig(Point3::new(5.0, 5.0, 5.0), 9.0),
            env: Environment::default(),
            sun: None,
            ground: None,
            splats: None,
        };
        let cam = test_camera();
        let (w, h) = (24u32, 24u32);
        let base = PathTraceOptions {
            spp: 256,
            max_depth: 3,
            denoise: false,
            seed: 5,
            ..Default::default()
        };
        let fixed_opts = PathTraceOptions {
            adaptive: false,
            ..base
        };
        let adaptive_opts = PathTraceOptions {
            adaptive: true,
            ..base
        };

        let fixed = render(&scene, &cam, w, h, &fixed_opts);
        let adaptive = render(&scene, &cam, w, h, &adaptive_opts);

        let mean = |f: &Film| {
            f.rgb
                .chunks_exact(3)
                .map(|p| luminance([p[0], p[1], p[2]]) as f64)
                .sum::<f64>()
                / (f.rgb.len() / 3) as f64
        };
        let (mf, ma) = (mean(&fixed), mean(&adaptive));
        let n_fixed = spend(&scene, &cam, w, h, &fixed_opts);
        let n_adaptive = spend(&scene, &cam, w, h, &adaptive_opts);
        eprintln!(
            "mean luminance: fixed {mf:.6} ({n_fixed} samples), adaptive {ma:.6} ({n_adaptive} samples)"
        );
        assert!(
            (ma - mf).abs() <= 0.02 * mf.abs().max(1e-3),
            "adaptive biased the estimate: {mf} -> {ma}"
        );
        assert!(
            n_adaptive < n_fixed / 2,
            "adaptive spent {n_adaptive} of {n_fixed} samples — no real saving"
        );
    }
}
