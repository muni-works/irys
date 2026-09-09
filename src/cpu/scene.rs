//! The scene graph and its acceleration structures.

use super::*;

// ─── scene & camera ───────────────────────────────────────────────────────

/// A traceable object: one BVH over a BRep solid plus its material.
pub struct Object<G> {
    /// Acceleration structure over the solid's analytic faces.
    pub bvh: Arc<Bvh<G>>,
    /// Surface description.
    pub material: Pbr,
    /// Object → world placement of the BVH, which is otherwise traced
    /// wherever it was built.
    ///
    /// Most callers bake placement into the geometry and leave this at the
    /// identity. An animation instead holds the geometry (and its BVH) still
    /// and moves this, so a jointed assembly re-poses with no re-evaluation
    /// and no BLAS rebuild — only the top-level structure is rebuilt.
    pub transform: Transform,
}

impl<G> Object<G> {
    /// A traceable object placed where its BVH was built.
    pub fn new(bvh: Arc<Bvh<G>>, material: Pbr) -> Self {
        Self {
            bvh,
            material,
            transform: Transform::identity(),
        }
    }

    /// A traceable object placed by an object→world transform.
    pub fn placed(bvh: Arc<Bvh<G>>, material: Pbr, transform: Transform) -> Self {
        Self {
            bvh,
            material,
            transform,
        }
    }
}

/// Everything the integrator needs to render a frame.
pub struct Scene<G> {
    /// Traceable BRep objects.
    pub objects: Vec<Object<G>>,
    /// Explicit area lights.
    pub lights: Vec<AreaLight>,
    /// Analytic sky, or a lat-long HDR environment map.
    pub env: Environment,
    /// An optional directional light of finite angular size — daylight.
    ///
    /// `None` is the historical behaviour: the environment and the area
    /// lights are the whole of the illumination.
    pub sun: Option<Sun>,
    /// Optional studio floor.
    pub ground: Option<Ground>,
    /// An optional captured Gaussian splat cloud, composited *additively*
    /// over every ray segment.
    ///
    /// This is a radiance field, not geometry: the colours in it already
    /// include the lighting of the room they were captured in. So the
    /// integrator treats it as **emissive and absorbing** — along every
    /// segment it emits `Σ T·α·c` and attenuates what lies beyond by
    /// `Π (1 − α)` (see [`crate::splats::composite`]). It lights nothing by
    /// next-event estimation, receives nothing, and spawns no rays; it does
    /// veil analytic surfaces in front of it and attenuate shadow rays that
    /// cross it.
    ///
    /// The honest way to say it: **a splat backdrop is an environment with
    /// depth.** Like a lat-long [`EnvMap`] it supplies the radiance for rays
    /// that hit no analytic surface — so the marble in a captured garage
    /// picks up reflections and diffuse bounce from the real room — and
    /// unlike one it also occupies space, so it can stand in front of
    /// something as well as behind it.
    ///
    /// The limitation that comes with that: the splat field is **not
    /// importance sampled.** There is no `Environment::sample` for it and no
    /// MIS strategy aimed at its bright spots, so indirect light from the
    /// cloud arrives only on BSDF-sampled bounce rays. A mirror or a smooth
    /// glass marble is therefore clean at low sample counts and a rough
    /// diffuse surface under a small bright window is noisy — exactly the
    /// behaviour of the analytic [`GradientEnv`], for the same reason.
    pub splats: Option<Arc<Bvh<Splats>>>,
}

// ─── intersection ─────────────────────────────────────────────────────────

/// What a ray landed on.
pub(crate) enum Landing {
    Surface {
        point: Point3,
        normal: Vec3,
        /// Surface tangent dP/du, when the parameterisation has one.
        tangent: Option<Vec3>,
        material: Pbr,
    },
    Light {
        emission: [f32; 3],
        light_index: usize,
        distance: f64,
        point: Point3,
    },
    Miss,
}

/// The scene's geometry gathered into a TLAS, built once per render.
///
/// `Scene` keeps `objects` as its authoring surface — a plain list is the
/// right thing to *write* — while the integrator traces against this. Kept
/// separate rather than added as a `Scene` field so the public struct-literal
/// construction in `Scene { objects, lights, env, ground }` keeps working.
pub(crate) struct SceneAccel<G> {
    pub(crate) tlas: Tlas<G>,
    /// Cumulative distribution over `scene.lights`, weighted by emitted power
    /// (emission luminance × area). One entry per light, ending at 1.0.
    ///
    /// Built once per render so next-event estimation can draw *one* light per
    /// bounce instead of shadow-raying all of them: the cost per bounce stops
    /// scaling with the number of softboxes, and the estimator stays unbiased
    /// because each contribution is divided by its own pick probability.
    pub(crate) light_cdf: Vec<f32>,
    /// Probability of picking each light, i.e. the CDF's per-entry mass. Kept
    /// alongside so the MIS weight for a BSDF ray that lands on an emitter can
    /// use the same pick probability the NEE strategy would have used.
    pub(crate) light_pick_pdf: Vec<f32>,
}

impl<G> SceneAccel<G> {
    /// Probability that [`SceneAccel::pick_light`] would choose `index`.
    #[inline]
    pub(crate) fn light_pick_pdf(&self, index: usize) -> f32 {
        self.light_pick_pdf.get(index).copied().unwrap_or(0.0)
    }

    /// Draw one light from the power-weighted table. Returns its index and the
    /// probability with which it was drawn.
    #[inline]
    pub(crate) fn pick_light(&self, u: f32) -> Option<(usize, f32)> {
        if self.light_cdf.is_empty() {
            return None;
        }
        let i = match self
            .light_cdf
            .binary_search_by(|c| c.partial_cmp(&u).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(i) | Err(i) => i.min(self.light_cdf.len() - 1),
        };
        let pdf = self.light_pick_pdf[i];
        if pdf > 0.0 { Some((i, pdf)) } else { None }
    }
}

/// Power-weighted selection table over a light list: per-light pick
/// probabilities and their running sum.
///
/// Shared by the CPU integrator and the GPU scene upload so both sample the
/// same distribution — a parity test that compared two different tables would
/// be testing nothing.
pub fn light_power_table(lights: &[AreaLight]) -> (Vec<f32>, Vec<f32>) {
    let powers: Vec<f32> = lights
        .iter()
        .map(|l| (luminance(l.emission) as f64 * l.area()).max(0.0) as f32)
        .collect();
    power_table_from_weights(&powers)
}

/// The weight → (CDF, per-entry probability) half of [`light_power_table`],
/// split out so the GPU scene upload can build the identical table from its
/// own packed lights.
pub fn power_table_from_weights(powers: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let total: f32 = powers.iter().sum();
    let n = powers.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    // A scene whose lights all carry zero power still needs a valid
    // distribution; uniform costs nothing and keeps the estimator finite.
    let pick: Vec<f32> = if total > 0.0 && total.is_finite() {
        powers.iter().map(|p| p / total).collect()
    } else {
        vec![1.0 / n as f32; n]
    };
    let mut cdf = Vec::with_capacity(n);
    let mut run = 0.0f32;
    for p in &pick {
        run += *p;
        cdf.push(run);
    }
    // Guard against float drift leaving the last entry just under 1.
    if let Some(last) = cdf.last_mut() {
        *last = 1.0;
    }
    (cdf, pick)
}

impl<G: Geometry> SceneAccel<G> {
    /// Place every object by its own transform (the identity for the usual
    /// case of geometry that arrives already world-placed) and gather them
    /// under one TLAS. Objects already hold `Arc<Bvh>`, so repeated parts
    /// share a BLAS without any copying — and a re-posed frame rebuilds only
    /// this structure.
    pub(crate) fn build(scene: &Scene<G>) -> Self {
        let instances = scene
            .objects
            .iter()
            .enumerate()
            .filter_map(|(i, obj)| Instance::new(Arc::clone(&obj.bvh), obj.transform.clone(), i))
            .collect();
        let (light_cdf, light_pick_pdf) = light_power_table(&scene.lights);
        Self {
            tlas: Tlas::build(instances),
            light_cdf,
            light_pick_pdf,
        }
    }
}

impl<G: Geometry> Scene<G> {
    /// What the splat cloud, if there is one, adds along `(t_min, t_max)` of
    /// `ray` — radiance emitted and transmittance surviving.
    ///
    /// The identity segment (no radiance, full transmittance) when the scene
    /// carries no cloud, so every caller can add it unconditionally.
    pub(crate) fn splat_segment(&self, ray: &Ray, t_min: f64, t_max: f64) -> SplatSegment {
        match &self.splats {
            Some(bvh) => crate::splats::composite(bvh, ray, t_min, t_max),
            None => SplatSegment::default(),
        }
    }

    /// Closest intersection against objects, ground, and lights.
    pub(crate) fn intersect(&self, accel: &SceneAccel<G>, ray: &Ray) -> Landing {
        let mut best_t = f64::INFINITY;
        let mut landing = Landing::Miss;

        // `1e-7` as the interval floor rather than a post-filter: pushed into
        // the traversal, a surface just behind the one the ray left is still
        // found instead of the whole query being discarded.
        if let Some(found) = self.tlas_hit(accel, ray, 1e-7) {
            best_t = found.hit.t;
            landing = Landing::Surface {
                point: found.hit.point,
                normal: found.hit.normal.into_inner(),
                tangent: found.hit.dpdu,
                material: self.objects[found.payload].material,
            };
        }

        if let Some(g) = &self.ground {
            let d = ray.direction.into_inner();
            if d.z.abs() > 1e-12 {
                let t = (g.z - ray.origin.z) / d.z;
                if t > 1e-6 && t < best_t {
                    best_t = t;
                    landing = Landing::Surface {
                        point: ray.at(t),
                        normal: Vec3::new(0.0, 0.0, 1.0),
                        // The studio sweep is a backdrop, not a machined
                        // face; it has no grain to align to.
                        tangent: None,
                        material: g.material,
                    };
                }
            }
        }

        for (i, l) in self.lights.iter().enumerate() {
            if let Some(t) = l.intersect(ray) {
                if t < best_t {
                    best_t = t;
                    landing = Landing::Light {
                        emission: l.emission,
                        light_index: i,
                        distance: t,
                        point: ray.at(t),
                    };
                }
            }
        }

        landing
    }

    /// Closest geometry hit past `t_min`, in world space.
    pub(crate) fn tlas_hit(
        &self,
        accel: &SceneAccel<G>,
        ray: &Ray,
        t_min: f64,
    ) -> Option<InstanceHit> {
        accel.tlas.trace_closest_range(ray, t_min, f64::INFINITY)
    }

    /// Any-hit occlusion test against geometry only (lights do not occlude).
    ///
    /// A true any-hit traversal: it returns at the first blocker rather than
    /// finding the nearest one and then comparing distance, which is strictly
    /// more work than a shadow ray needs.
    ///
    /// Kept as the fast path for the common case; light sampling goes through
    /// [`Scene::shadow_transmittance`], which can see *through* a pane.
    #[allow(dead_code)]
    pub(crate) fn occluded(
        &self,
        accel: &SceneAccel<G>,
        origin: Point3,
        dir: Vec3,
        max_dist: f64,
    ) -> bool {
        self.shadow_transmittance(accel, origin, dir, max_dist)
            .is_none()
    }

    /// How much of a light's radiance survives the trip from `origin` along
    /// `dir` to `max_dist` — `None` when the ray is blocked outright.
    ///
    /// The material-blind any-hit test this replaces made a window pane an
    /// opaque wall: NEE found a blocker and returned black, so a room lit
    /// through glass could only be lit by paths that *happened* to refract
    /// into the sun, which at any sane spp is never. That is why the court's
    /// clerestory had to be cut open.
    ///
    /// A thin-walled transmissive sheet is not a blocker, it is a filter. It
    /// has no interior for a ray to travel through and no lateral offset
    /// (see [`Pbr::thin_walled`]), so the shadow ray carries straight on with
    /// its throughput multiplied by the sheet's transmittance. The factor is
    /// exactly the one [`dielectric_eval`]'s thin-walled branch applies to a
    /// BSDF-sampled path — `transmission · (1 − F(cos θ))` — so the two
    /// strategies estimate the same integral and MIS stays consistent. (It is
    /// *not* `(1 − F)²`: this renderer's sheet is a single Fresnel interface
    /// with `R + T = 1`, and a shadow ray that disagreed with the BSDF by a
    /// second factor of `(1 − F)` would double-count under MIS in one
    /// direction and lose energy in the other.)
    ///
    /// Frosted glass is *not* handled: a rough sheet scatters, and pretending
    /// the light arrives along the straight line is only right in the smooth
    /// limit. The transmittance is therefore weighted by the sheet's
    /// specular lobe narrowness — a fully rough pane blocks as before.
    ///
    /// At most [`MAX_SHADOW_SHEETS`] panes are crossed; a shadow ray that
    /// finds more is treated as blocked, which bounds the traversal cost and
    /// keeps a stack of panes from turning into an unbounded loop.
    pub(crate) fn shadow_transmittance(
        &self,
        accel: &SceneAccel<G>,
        origin: Point3,
        dir: Vec3,
        max_dist: f64,
    ) -> Option<[f32; 3]> {
        let limit = max_dist - 1e-6;
        if let Some(g) = &self.ground {
            let d = dir;
            if d.z.abs() > 1e-12 {
                let t = (g.z - origin.z) / d.z;
                if t > 1e-6 && t < limit {
                    return None;
                }
            }
        }

        let ray = Ray::new(origin, dir);
        // The splat cloud shadows by its accumulated opacity: a captured wall
        // is opaque enough to stop the light, a captured net or a wisp of
        // reconstruction dust is not. Grey, because the alphas are grey —
        // a Gaussian's colour is emission, not a filter.
        let splat_tr = self.splat_segment(&ray, 1e-6, limit).transmittance;
        if splat_tr <= 1e-4 {
            return None;
        }
        let mut tr = [splat_tr; 3];
        let mut t0 = 1e-6;
        let mut crossed = 0usize;
        loop {
            let Some(found) = accel.tlas.trace_closest_range(&ray, t0, limit) else {
                return Some(tr);
            };
            if crossed == MAX_SHADOW_SHEETS {
                // More sheets than the cap allows: fall back to opaque.
                return None;
            }
            crossed += 1;
            let m = &self.objects[found.payload].material;
            let Some(sheet) = sheet_transmittance(m, found.hit.normal.into_inner().dot(dir)) else {
                return None;
            };
            tr = mul3(tr, sheet);
            if max3(tr) <= 1e-6 {
                return None;
            }
            t0 = found.hit.t + 1e-6;
            if t0 >= limit {
                return Some(tr);
            }
        }
    }

    /// Next-event estimation: sample *one* area light, drawn from the
    /// accel's power-weighted table, MIS-weighted against the BSDF sampling
    /// strategy.
    ///
    /// One shadow ray per bounce regardless of how many softboxes the rig
    /// has. Dividing the contribution by the pick probability leaves the
    /// estimator unbiased — the mean over many samples matches the old
    /// sample-every-light estimator exactly — and picking by power means the
    /// lights that matter are the ones usually chosen.
    // The shading frame (p, t, b, n) and the outgoing direction are the
    // integrator's hot-loop state; bundling them into a struct just to
    // satisfy the lint would add a copy per light sample.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sample_lights(
        &self,
        accel: &SceneAccel<G>,
        p: Point3,
        frame: &Frame,
        wo_local: Vec3,
        m: &Pbr,
        eta: f32,
        lambda_nm: f32,
        rng: &mut Rng,
    ) -> [f32; 3] {
        let Frame { t, b, n } = *frame;
        // The pick draw comes first so the light choice is independent of the
        // position draw on the chosen rectangle.
        let Some((index, pick_pdf)) = accel.pick_light(rng.f64() as f32) else {
            return [0.0; 3];
        };
        let light = &self.lights[index];
        let lp = light.sample(rng.f64(), rng.f64());
        let to_light = lp - p;
        let dist = to_light.norm();
        if dist < 1e-9 {
            return [0.0; 3];
        }
        let wi_world = to_light / dist;
        let ln = light.normal();
        let cos_light = -wi_world.dot(ln);
        if cos_light <= 1e-9 {
            return [0.0; 3];
        }
        let wi_local = to_local(t, b, n, wi_world);
        if wi_local.z <= 0.0 {
            return [0.0; 3];
        }

        let (f, bsdf_pdf) = bsdf_eval(m, wo_local, wi_local, eta, lambda_nm);
        if max3(f) <= 0.0 {
            return [0.0; 3];
        }

        // Solid-angle PDF of the *full* NEE strategy: pick this light, then
        // pick a point on it. The BSDF-hits-a-light branch in `radiance`
        // reconstructs the same product, so MIS stays consistent.
        let light_pdf = pick_pdf * (dist * dist / (cos_light * light.area())) as f32;
        if !light_pdf.is_finite() || light_pdf <= 0.0 {
            return [0.0; 3];
        }

        let Some(tr) = self.shadow_transmittance(accel, p + n * 1e-5, wi_world, dist) else {
            return [0.0; 3];
        };

        let w = power_heuristic(light_pdf, bsdf_pdf);
        scale3(mul3(mul3(f, light.emission), tr), w / light_pdf)
    }

    /// Next-event estimation against the environment, MIS-weighted against
    /// BSDF sampling.
    ///
    /// Only runs for an importance-sampled environment ([`EnvMap`]). The
    /// analytic gradient stays BSDF-only, exactly as before — it is
    /// low-frequency enough that a second strategy buys nothing.
    pub(crate) fn sample_environment(
        &self,
        accel: &SceneAccel<G>,
        p: Point3,
        frame: &Frame,
        wo_local: Vec3,
        m: &Pbr,
        eta: f32,
        lambda_nm: f32,
        rng: &mut Rng,
    ) -> [f32; 3] {
        let Frame { t, b, n } = *frame;
        let Some((wi_world, li, env_pdf)) = self.env.sample(rng.f64(), rng.f64()) else {
            return [0.0; 3];
        };
        if !env_pdf.is_finite() || env_pdf <= 0.0 || max3(li) <= 0.0 {
            return [0.0; 3];
        }
        let wi_local = to_local(t, b, n, wi_world);
        if wi_local.z <= 0.0 {
            return [0.0; 3];
        }
        let (f, bsdf_pdf) = bsdf_eval(m, wo_local, wi_local, eta, lambda_nm);
        if max3(f) <= 0.0 {
            return [0.0; 3];
        }
        // The environment is at infinity: nothing between here and the sky
        // may block, so the shadow ray is unbounded.
        let Some(tr) = self.shadow_transmittance(accel, p + n * 1e-5, wi_world, f64::INFINITY)
        else {
            return [0.0; 3];
        };
        let w = power_heuristic(env_pdf, bsdf_pdf);
        scale3(mul3(mul3(f, li), tr), w / env_pdf)
    }

    /// Next-event estimation against the sun disc, MIS-weighted against BSDF
    /// sampling — the same three-line shape as the area lights, over a cone
    /// at infinity instead of a rectangle at a distance.
    pub(crate) fn sample_sun(
        &self,
        accel: &SceneAccel<G>,
        p: Point3,
        frame: &Frame,
        wo_local: Vec3,
        m: &Pbr,
        eta: f32,
        lambda_nm: f32,
        rng: &mut Rng,
    ) -> [f32; 3] {
        let Some(sun) = &self.sun else {
            return [0.0; 3];
        };
        let Frame { t, b, n } = *frame;
        let (wi_world, li, sun_pdf) = sun.sample(rng.f64(), rng.f64());
        if !sun_pdf.is_finite() || sun_pdf <= 0.0 || max3(li) <= 0.0 {
            return [0.0; 3];
        }
        let wi_local = to_local(t, b, n, wi_world);
        if wi_local.z <= 0.0 {
            return [0.0; 3];
        }
        let (f, bsdf_pdf) = bsdf_eval(m, wo_local, wi_local, eta, lambda_nm);
        if max3(f) <= 0.0 {
            return [0.0; 3];
        }
        // The sun is at infinity, so the shadow ray is unbounded.
        let Some(tr) = self.shadow_transmittance(accel, p + n * 1e-5, wi_world, f64::INFINITY)
        else {
            return [0.0; 3];
        };
        let w = power_heuristic(sun_pdf, bsdf_pdf);
        scale3(mul3(mul3(f, li), tr), w / sun_pdf)
    }
}

// ─── the caustic pass's half of the integrator ────────────────────────────

/// The acceleration structure the caustic pass traces against, built once.
///
/// A thin wrapper so [`crate::caustics`] can hold the TLAS without the
/// integrator's internals leaking out of this module.
pub(crate) struct CausticContext<G> {
    pub(crate) accel: SceneAccel<G>,
}

impl<G: Geometry> CausticContext<G> {
    pub(crate) fn new(scene: &Scene<G>) -> Self {
        Self {
            accel: SceneAccel::build(scene),
        }
    }
}

/// Centre and bounding radius of the scene's *refracting* geometry, or `None`
/// when there is none.
///
/// This is what the caustic pass aims at. Aiming is importance sampling and
/// not a cheat — the emitted power carries the solid angle of the cone — but
/// it is the difference between a caustic in seconds and a caustic never.
pub(crate) fn caustic_bounds<G: Geometry>(scene: &Scene<G>) -> Option<(Point3, f64)> {
    let mut bounds: Option<Aabb> = None;
    for obj in &scene.objects {
        if !crate::caustics::is_caustic_refractor(&obj.material) {
            continue;
        }
        let Some(inst) = Instance::new(Arc::clone(&obj.bvh), obj.transform.clone(), 0) else {
            continue;
        };
        let b = inst.world_aabb();
        match &mut bounds {
            Some(acc) => acc.include(&b),
            None => bounds = Some(b),
        }
    }
    let b = bounds?;
    let c = b.center();
    let r: f64 = 0.5 * (b.max - b.min).norm();
    if !r.is_finite() || r <= 0.0 {
        return None;
    }
    Some((c, r))
}

/// Follow one photon from a light until it lands on a diffuse surface.
///
/// Returns the landing point, the surface normal there and the power the
/// photon still carries — or `None` if it was absorbed, escaped, or reached a
/// diffuse surface without ever having been refracted by a *solid*.
///
/// That last condition is the whole of the double-counting rule: light that
/// only ever passed through a thin pane already belongs to next-event
/// estimation (see `sheet_transmittance`), so a photon carrying it is dropped
/// here rather than added twice.
///
/// The BSDF is the camera path's BSDF, unmodified. It can be, because
/// [`dielectric_eval`]'s transmission branch already cancels Walter's `η_t²`
/// against the `1/η²` radiance compression — so what it returns is the
/// symmetric quantity a photon wants, and importance transport needs no
/// correction factor here.
pub(crate) fn trace_photon<G: Geometry>(
    scene: &Scene<G>,
    ctx: &CausticContext<G>,
    origin: Point3,
    dir: Vec3,
    power: [f32; 3],
    max_bounces: u32,
    rng: &mut Rng,
) -> Option<(Point3, Vec3, [f32; 3])> {
    let accel = &ctx.accel;
    let mut ray = Ray::new(origin, dir);
    let mut power = power;
    let mut lambda_nm: Option<f64> = None;
    let mut medium: Option<Pbr> = None;
    let mut refracted_by_a_solid = false;

    for _ in 0..max_bounces {
        let landing = scene.intersect(accel, &ray);
        // Absorb along the segment just travelled, if it was inside glass.
        if let Some(med) = &medium {
            let sigma = med.extinction();
            if let Landing::Surface { point, .. } = &landing
                && max3(sigma) > 0.0
            {
                {
                    let d = (*point - ray.origin).norm() as f32;
                    power = mul3(
                        power,
                        [
                            (-sigma[0] * d).exp(),
                            (-sigma[1] * d).exp(),
                            (-sigma[2] * d).exp(),
                        ],
                    );
                }
            }
        }
        let Landing::Surface {
            point,
            normal,
            tangent,
            material,
        } = landing
        else {
            // Off into the sky, or onto a light's back: no deposit.
            return None;
        };

        let wo_world = -ray.direction.into_inner();
        let entering = normal.dot(wo_world) >= 0.0;
        let n = if normal.dot(wo_world) < 0.0 {
            -normal
        } else {
            normal
        };

        if material.transmission <= 0.0 {
            // A diffuse receiver. Deposit only if the light got here the way
            // the path tracer cannot follow.
            return if refracted_by_a_solid && max3(power) > 0.0 {
                Some((point, n, power))
            } else {
                None
            };
        }

        if lambda_nm.is_none() && material.is_dispersive() {
            let nm = crate::spectrum::sample_lambda_nm(rng.f64());
            lambda_nm = Some(nm);
            power = mul3(power, crate::spectrum::hero_weight(nm));
        }
        let n_glass = material.index_at(lambda_nm).max(1e-3);
        let eta = if material.thin_walled || entering {
            n_glass
        } else {
            1.0 / n_glass
        };
        let hero = lambda_nm.unwrap_or(0.0) as f32;

        let frame = shading_frame(n, tangent);
        let wo_local = to_local(frame.t, frame.b, n, wo_world);
        if wo_local.z <= 0.0 {
            return None;
        }
        let Some(Sampled::Surface(wi_local, f, pdf)) =
            bsdf_sample(&material, wo_local, eta, hero, rng)
        else {
            return None;
        };
        power = mul3(power, scale3(f, 1.0 / pdf));
        if max3(power) <= 1e-12 {
            return None;
        }

        let transmitted = wi_local.z < 0.0;
        if transmitted && !material.thin_walled {
            refracted_by_a_solid = true;
            medium = if entering { Some(material) } else { None };
        }
        let wi_world = to_world(frame.t, frame.b, n, wi_local);
        let offset = if transmitted { -n } else { n };
        ray = Ray::new(point + offset * 1e-5, wi_world);
    }
    None
}
