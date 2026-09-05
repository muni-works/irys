//! Photon-mapped caustics: the light a path tracer cannot find.
//!
//! # Why a second pass at all
//!
//! Next-event estimation is what makes a path tracer converge, and it works
//! by drawing a straight line from a shading point to a light. That line is
//! only the right answer when nothing bends it. Put a glass sphere or a
//! sheet of water between the two and the connection is *invalid*: the light
//! that actually arrives came in along a refracted path, and no amount of
//! shadow-ray sampling will find it. What is left is BSDF sampling, which
//! has to land a diffuse bounce on the sphere, refract twice and hit a
//! one-degree sun disc by chance — roughly never. Hence the pool's verdict:
//! caustics unreachable at any spp.
//!
//! [`Scene::shadow_transmittance`](crate::pathtrace) fixed the easy half of
//! this — a *thin* pane does not bend the line measurably, so the shadow ray
//! is allowed through it, dimmed. This module is the hard half: light through
//! a refracting *solid*, or through water, where the bending is the whole
//! effect.
//!
//! # What this does
//!
//! Forward light transport, the direction the photons actually travel. Emit
//! photons from the lights *at the refractive geometry only*, follow them
//! through refraction and reflection with the same BSDF sampling the camera
//! paths use, and deposit each one's power at the first **diffuse** surface
//! it reaches. The deposits go into a world-space hash grid; at shading time
//! the integrator asks the grid for the irradiance in a disc around the
//! shading point and adds it as direct light.
//!
//! This is Jensen's photon map with the caustic half kept and the global half
//! thrown away — the global half is exactly what the path tracer already
//! integrates well, and mixing the two would double count.
//!
//! # The division of labour, and why it does not double count
//!
//! Every unit of light arriving at a diffuse surface is claimed by exactly
//! one estimator:
//!
//! - **unobstructed light** — next-event estimation, as always;
//! - **light through a thin sheet** — next-event estimation, attenuated by
//!   the sheet (see `sheet_transmittance`);
//! - **light through a refracting solid or a water surface** — this map, and
//!   *only* this map. A shadow ray still treats such geometry as opaque, so
//!   NEE contributes nothing along those directions.
//!
//! The rule is enforced at deposit time: a photon is only written into the
//! map if its history included a transmissive event on geometry that is *not*
//! thin-walled. A photon that merely passed through a pane is dropped,
//! because NEE already has that light.
//!
//! # Density estimation
//!
//! A world-space hash grid of splats with a fixed kernel radius, rather than
//! a per-object texel grid. The grid does not care what shape the receiver
//! is, needs no parameterisation and no projection onto a dominant plane, and
//! a pool floor with a drain and a step in it is exactly the case where a
//! projected grid goes wrong. The cost is the usual one: the radius is a
//! blur, and a caustic sharper than the radius is smoothed.
//!
//! The kernel is constant over the disc — `Φ/(π r²)` — not a cone or a
//! Gaussian. A constant kernel conserves energy exactly, which is what makes
//! the energy check in the tests a real check rather than a check of the
//! kernel's normalisation.

use std::collections::HashMap;

use crate::geometry::Geometry;
use crate::math::{Point3, Vec3};
use crate::pathtrace::{
    AreaLight, Pbr, Rng, Scene, Sun, caustic_bounds, cosine_hemisphere, luminance, onb,
    trace_photon, CausticContext,
};

/// How the caustic pass is shot.
#[derive(Debug, Clone, Copy)]
pub struct CausticOptions {
    /// Photons emitted, in total, split between the lights by the power each
    /// one aims at the refractive geometry.
    ///
    /// The map's noise falls as `1/sqrt(photons)` inside the gather radius,
    /// so this is the knob that trades time for a clean caustic. Tens of
    /// thousands read as a caustic; a million is a photograph.
    pub photons: usize,
    /// Gather radius in scene units. `None` derives one from the scene's
    /// size — a two-hundredth of the refractive geometry's bounding radius,
    /// which is fine detail on the object that made the caustic.
    pub radius: Option<f64>,
    /// How many surface events one photon may take before it is dropped.
    pub max_bounces: u32,
    /// Random seed.
    pub seed: u64,
}

impl Default for CausticOptions {
    fn default() -> Self {
        Self {
            photons: 200_000,
            radius: None,
            max_bounces: 12,
            seed: 0xca05_71c5,
        }
    }
}

/// One deposited photon.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Photon {
    /// Where it landed.
    pub(crate) point: Point3,
    /// The surface normal there, so a gather does not pull photons off the
    /// far side of a thin floor or around a corner.
    pub(crate) normal: Vec3,
    /// The power it carried, in the same units as the lights' emission times
    /// area — watts, if the lights are watts.
    pub(crate) power: [f32; 3],
}

/// A built caustic map: photons in a world-space hash grid, ready to be
/// asked for irradiance.
#[derive(Debug, Clone)]
pub struct CausticMap {
    radius: f64,
    inv_cell: f64,
    photons: Vec<Photon>,
    cells: HashMap<[i64; 3], Vec<u32>>,
    emitted: [f32; 3],
    deposited: [f32; 3],
}

impl CausticMap {
    /// An empty map — the identity for the integrator, and what a scene with
    /// no refractive geometry produces.
    pub fn empty() -> Self {
        Self {
            radius: 1.0,
            inv_cell: 1.0,
            photons: Vec::new(),
            cells: HashMap::new(),
            emitted: [0.0; 3],
            deposited: [0.0; 3],
        }
    }

    /// Number of photons actually deposited.
    pub fn len(&self) -> usize {
        self.photons.len()
    }

    /// Whether the map has any photons at all.
    pub fn is_empty(&self) -> bool {
        self.photons.is_empty()
    }

    /// The gather radius the map was built with.
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Total power emitted toward the refractive geometry, summed over every
    /// photon the pass shot — including the ones that never landed.
    ///
    /// The denominator of the pass's energy budget, exposed so a test can ask
    /// what fraction of the light that went in came back out.
    pub fn emitted_power(&self) -> [f32; 3] {
        self.emitted
    }

    /// Total power deposited into the map.
    pub fn deposited_power(&self) -> [f32; 3] {
        self.deposited
    }

    /// Irradiance at `p` on a surface with normal `n`, by density estimation
    /// over the gather disc.
    ///
    /// Constant kernel: the sum of the powers within the radius over the
    /// disc's area. Photons on a surface facing more than about 25° away are
    /// rejected, which is what keeps a caustic on the pool floor off the
    /// underside of the step next to it.
    pub fn irradiance(&self, p: Point3, n: Vec3) -> [f32; 3] {
        if self.photons.is_empty() {
            return [0.0; 3];
        }
        let r2 = self.radius * self.radius;
        let c = self.cell(p);
        let mut sum = [0.0f32; 3];
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let key = [c[0] + dx, c[1] + dy, c[2] + dz];
                    let Some(list) = self.cells.get(&key) else {
                        continue;
                    };
                    for &i in list {
                        let ph = &self.photons[i as usize];
                        if (ph.point - p).norm_squared() > r2 {
                            continue;
                        }
                        if ph.normal.dot(n) < 0.9 {
                            continue;
                        }
                        sum[0] += ph.power[0];
                        sum[1] += ph.power[1];
                        sum[2] += ph.power[2];
                    }
                }
            }
        }
        let area = (std::f64::consts::PI * r2) as f32;
        [sum[0] / area, sum[1] / area, sum[2] / area]
    }

    #[inline]
    fn cell(&self, p: Point3) -> [i64; 3] {
        [
            (p.x * self.inv_cell).floor() as i64,
            (p.y * self.inv_cell).floor() as i64,
            (p.z * self.inv_cell).floor() as i64,
        ]
    }
}

/// Shoot the caustic pass over `scene`.
///
/// Returns an empty map when the scene has no transmissive geometry, which
/// makes calling this unconditionally free for the scenes that do not need
/// it.
pub fn trace<G: Geometry + Send + Sync>(
    scene: &Scene<G>,
    opts: &CausticOptions,
) -> CausticMap {
    let Some((center, extent)) = caustic_bounds(scene) else {
        return CausticMap::empty();
    };
    let radius = opts.radius.unwrap_or((extent / 200.0).max(1e-6));

    // Split the photon budget between the emitters by the power each one
    // aims at the refractive geometry — a rim light behind the glass matters
    // more to a caustic than a big soft fill that misses it.
    let mut emitters: Vec<Emitter> = Vec::new();
    for (i, l) in scene.lights.iter().enumerate() {
        emitters.push(Emitter::Area {
            index: i,
            power: area_light_aimed_power(l, center, extent),
        });
    }
    if let Some(sun) = &scene.sun {
        let p = luminance(sun.irradiance) * (std::f64::consts::PI * extent * extent) as f32;
        emitters.push(Emitter::Sun { power: p });
    }
    let total: f32 = emitters.iter().map(|e| e.power()).sum();
    if !total.is_finite() || total <= 0.0 || opts.photons == 0 {
        return CausticMap::empty();
    }

    let ctx = CausticContext::new(scene);
    let mut photons: Vec<Photon> = Vec::new();
    let mut emitted = [0.0f32; 3];
    let mut deposited = [0.0f32; 3];

    for (ei, em) in emitters.iter().enumerate() {
        let share = em.power() / total;
        let n = ((opts.photons as f64) * share as f64).round() as usize;
        if n == 0 {
            continue;
        }
        let seed = opts.seed ^ ((ei as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        for k in 0..n {
            let mut rng = Rng::new(seed.wrapping_add(k as u64).wrapping_mul(0x2545_f491_4f6c_dd1d));
            let Some((origin, dir, power)) = (match em {
                Emitter::Area { index, .. } => emit_from_area(
                    &scene.lights[*index],
                    center,
                    extent,
                    n,
                    &mut rng,
                ),
                Emitter::Sun { .. } => {
                    emit_from_sun(scene.sun.as_ref().unwrap(), center, extent, n, &mut rng)
                }
            }) else {
                continue;
            };
            emitted[0] += power[0];
            emitted[1] += power[1];
            emitted[2] += power[2];
            if let Some(hit) =
                trace_photon(scene, &ctx, origin, dir, power, opts.max_bounces, &mut rng)
            {
                deposited[0] += hit.2[0];
                deposited[1] += hit.2[1];
                deposited[2] += hit.2[2];
                photons.push(Photon {
                    point: hit.0,
                    normal: hit.1,
                    power: hit.2,
                });
            }
        }
    }

    let inv_cell = 1.0 / radius;
    let mut cells: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
    for (i, ph) in photons.iter().enumerate() {
        let key = [
            (ph.point.x * inv_cell).floor() as i64,
            (ph.point.y * inv_cell).floor() as i64,
            (ph.point.z * inv_cell).floor() as i64,
        ];
        cells.entry(key).or_default().push(i as u32);
    }

    CausticMap {
        radius,
        inv_cell,
        photons,
        cells,
        emitted,
        deposited,
    }
}

enum Emitter {
    Area { index: usize, power: f32 },
    Sun { power: f32 },
}

impl Emitter {
    fn power(&self) -> f32 {
        match self {
            Emitter::Area { power, .. } | Emitter::Sun { power } => *power,
        }
    }
}

/// Luminous power an area light aims into the cone subtending the refractive
/// geometry — the budget split's weight, not a physical quantity in itself.
fn area_light_aimed_power(l: &AreaLight, center: Point3, extent: f64) -> f32 {
    let d = (center - l.center).norm();
    if d <= 1e-9 {
        return 0.0;
    }
    let sin_max = (extent / d).min(1.0);
    let cos_max = (1.0 - sin_max * sin_max).max(0.0).sqrt();
    let omega = 2.0 * std::f64::consts::PI * (1.0 - cos_max);
    let lum = luminance(l.emission);
    (lum as f64 * l.area() * omega).max(0.0) as f32
}

/// One photon off an area light, aimed into the cone that contains the
/// refractive geometry.
///
/// Aiming is not an approximation, it is importance sampling: the photon's
/// power carries the cone's solid angle, so the estimator is the same one a
/// full-hemisphere emission would give and simply spends none of its budget
/// on photons that were never going to reach the glass.
fn emit_from_area(
    l: &AreaLight,
    center: Point3,
    extent: f64,
    n: usize,
    rng: &mut Rng,
) -> Option<(Point3, Vec3, [f32; 3])> {
    let p = l.sample(rng.f64(), rng.f64());
    let ln = l.normal();
    let to = center - p;
    let d = to.norm();
    if d <= extent {
        // Inside the target's bounds: no cone to aim into, so fall back to
        // the whole hemisphere.
        let (t, b) = onb(ln);
        let c = cosine_hemisphere(rng.f64(), rng.f64());
        let dir = t * c.x + b * c.y + ln * c.z;
        let scale = (std::f64::consts::PI * l.area() / n as f64) as f32;
        return Some((p, dir, [
            l.emission[0] * scale,
            l.emission[1] * scale,
            l.emission[2] * scale,
        ]));
    }
    let w = to / d;
    let sin_max = extent / d;
    let cos_max = (1.0 - sin_max * sin_max).max(0.0).sqrt();
    let cos_theta = 1.0 - rng.f64() * (1.0 - cos_max);
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = 2.0 * std::f64::consts::PI * rng.f64();
    let (t, b) = onb(w);
    let dir = t * (sin_theta * phi.cos()) + b * (sin_theta * phi.sin()) + w * cos_theta;
    let cos_l = dir.dot(ln);
    if cos_l <= 0.0 {
        return None;
    }
    let omega = 2.0 * std::f64::consts::PI * (1.0 - cos_max);
    // Radiance × area × cos × dω, divided by the photon count: exactly the
    // power this photon represents.
    let scale = (cos_l * l.area() * omega / n as f64) as f32;
    Some((
        p,
        dir,
        [
            l.emission[0] * scale,
            l.emission[1] * scale,
            l.emission[2] * scale,
        ],
    ))
}

/// One photon off the sun: a parallel ray from a disc that covers the
/// refractive geometry's bounds.
fn emit_from_sun(
    sun: &Sun,
    center: Point3,
    extent: f64,
    n: usize,
    rng: &mut Rng,
) -> Option<(Point3, Vec3, [f32; 3])> {
    let w = sun.direction;
    let (t, b) = onb(w);
    // Uniform in the disc, by the usual sqrt warp.
    let r = extent * rng.f64().sqrt();
    let phi = 2.0 * std::f64::consts::PI * rng.f64();
    let origin = center + w * (extent * 2.0) + t * (r * phi.cos()) + b * (r * phi.sin());
    let area = std::f64::consts::PI * extent * extent;
    let scale = (area / n as f64) as f32;
    let e = sun.irradiance;
    Some((
        origin,
        -w,
        [e[0] * scale, e[1] * scale, e[2] * scale],
    ))
}

/// Whether a material is the kind of refractor the caustic pass exists for:
/// a solid or a water surface, not a pane.
pub(crate) fn is_caustic_refractor(m: &Pbr) -> bool {
    m.transmission > 0.0 && !m.thin_walled
}
