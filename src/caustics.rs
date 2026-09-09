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
    AreaLight, CausticContext, Pbr, Rng, Scene, Sun, caustic_bounds, cosine_hemisphere, luminance,
    onb, trace_photon,
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

/// How nearly a photon's surface must face the way the gather does before it
/// counts: about 25°. It is what keeps a caustic on the pool floor off the
/// underside of the step beside it, and every reader of the map — the
/// integrator's irradiance, a score's power — has to apply the same rule or
/// they are not looking at the same caustic.
const ONE_SIDED: f64 = 0.9;

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
                        if ph.normal.dot(n) < ONE_SIDED {
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

    /// The power deposited within `r` of `centre` on a surface facing
    /// `normal` — watts, not watts per square metre.
    ///
    /// [`CausticMap::irradiance`] divides by the disc's area because it is
    /// answering the integrator's question. A score does not want that: the
    /// rune asks how much of the sun is going through the keyhole, and the
    /// numerator of that fraction is power. Same photons, same one-sided
    /// rejection, one fewer division.
    ///
    /// The grid is walked when the disc spans only a few cells, which is the
    /// case the grid is for; past that the walk touches more cells than there
    /// are photons and a straight scan is both simpler and faster.
    pub fn power_within(&self, centre: Point3, normal: Vec3, r: f64) -> [f32; 3] {
        if self.photons.is_empty() || !r.is_finite() || r <= 0.0 {
            return [0.0; 3];
        }
        let r2 = r * r;
        let mut sum = [0.0f32; 3];
        let mut add = |ph: &Photon| {
            if (ph.point - centre).norm_squared() <= r2 && ph.normal.dot(normal) >= ONE_SIDED {
                sum[0] += ph.power[0];
                sum[1] += ph.power[1];
                sum[2] += ph.power[2];
            }
        };
        let k = (r * self.inv_cell).ceil() as i64;
        if k > 3 {
            self.photons.iter().for_each(add);
            return sum;
        }
        let c = self.cell(centre);
        for dz in -k..=k {
            for dy in -k..=k {
                for dx in -k..=k {
                    let Some(list) = self.cells.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) else {
                        continue;
                    };
                    for &i in list {
                        add(&self.photons[i as usize]);
                    }
                }
            }
        }
        sum
    }

    /// A map built straight from a photon list, so a test can know the
    /// answer before it asks. The grid is laid out exactly as [`trace`] lays
    /// it out; that is the point of building it here rather than by hand.
    #[cfg(test)]
    pub(crate) fn from_photons(radius: f64, photons: Vec<Photon>) -> Self {
        let inv_cell = 1.0 / radius;
        let mut cells: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
        let mut deposited = [0.0f32; 3];
        for (i, ph) in photons.iter().enumerate() {
            let key = [
                (ph.point.x * inv_cell).floor() as i64,
                (ph.point.y * inv_cell).floor() as i64,
                (ph.point.z * inv_cell).floor() as i64,
            ];
            cells.entry(key).or_default().push(i as u32);
            for k in 0..3 {
                deposited[k] += ph.power[k];
            }
        }
        Self {
            radius,
            inv_cell,
            photons,
            cells,
            emitted: deposited,
            deposited,
        }
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
pub fn trace<G: Geometry + Send + Sync>(scene: &Scene<G>, opts: &CausticOptions) -> CausticMap {
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
        // One photon per index, each with its own seeded stream, so the pass
        // is deterministic however rayon schedules it — the same property the
        // pixel loop has, for the same reason.
        let shoot = |k: usize| -> ([f32; 3], Option<Photon>) {
            let mut rng = Rng::new(
                seed.wrapping_add(k as u64)
                    .wrapping_mul(0x2545_f491_4f6c_dd1d),
            );
            let Some((origin, dir, power)) = (match em {
                Emitter::Area { index, .. } => {
                    emit_from_area(&scene.lights[*index], center, extent, n, &mut rng)
                }
                Emitter::Sun { .. } => {
                    emit_from_sun(scene.sun.as_ref().unwrap(), center, extent, n, &mut rng)
                }
            }) else {
                return ([0.0; 3], None);
            };
            let landed = trace_photon(scene, &ctx, origin, dir, power, opts.max_bounces, &mut rng)
                .map(|(point, normal, power)| Photon {
                    point,
                    normal,
                    power,
                });
            (power, landed)
        };

        #[cfg(not(target_arch = "wasm32"))]
        let landed: Vec<([f32; 3], Option<Photon>)> = {
            use rayon::prelude::*;
            (0..n).into_par_iter().map(shoot).collect()
        };
        #[cfg(target_arch = "wasm32")]
        let landed: Vec<([f32; 3], Option<Photon>)> = (0..n).map(shoot).collect();

        for (power, ph) in landed {
            emitted[0] += power[0];
            emitted[1] += power[1];
            emitted[2] += power[2];
            if let Some(ph) = ph {
                deposited[0] += ph.power[0];
                deposited[1] += ph.power[1];
                deposited[2] += ph.power[2];
                photons.push(ph);
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
        return Some((
            p,
            dir,
            [
                l.emission[0] * scale,
                l.emission[1] * scale,
                l.emission[2] * scale,
            ],
        ));
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
    Some((origin, -w, [e[0] * scale, e[1] * scale, e[2] * scale]))
}

/// Whether a material is the kind of refractor the caustic pass exists for:
/// a solid or a water surface, not a pane.
pub(crate) fn is_caustic_refractor(m: &Pbr) -> bool {
    m.transmission > 0.0 && !m.thin_walled
}

// ─── the device layout ─────────────────────────────────────────────────────

/// Photons per texture row in [`CausticPack`]. Every WebGPU device allows a
/// 2D texture at least 8192 wide; 4096 leaves the same headroom in height.
pub const PACK_WIDTH: u32 = 4096;

/// Texels one packed photon occupies: position, normal, power.
pub const PACK_TEXELS_PER_PHOTON: u32 = 3;

/// The map, re-laid for a shader: photons sorted by the hash of their cell,
/// and a bucket table of `(start, count)` into that order.
///
/// The CPU map keys a `HashMap` by the exact cell. A device has no hash map,
/// so the cell's key is hashed into a power-of-two table and the photons are
/// sorted by that bucket. Two cells may share a bucket; that costs the
/// gather a few extra distance tests and changes nothing about its answer,
/// because every photon is still tested against the radius before it counts
/// — exactly as in [`CausticMap::irradiance`]. The one hazard, two of the
/// twenty-seven cells around a shading point sharing a bucket and so being
/// walked twice, is the shader's to avoid; it remembers the buckets it has
/// visited.
///
/// The layout is what the textures the GPU tier binds want: the bucket table
/// as one `(start, count)` pair per texel of an `Rg32Uint` image, the photons
/// as three `Rgba32Float` texels each. Both images are [`PACK_WIDTH`] texels
/// wide and as tall as they need to be.
#[derive(Debug, Clone)]
pub struct CausticPack {
    /// The gather radius, in scene units.
    pub radius: f32,
    /// `1 / radius`: the cell size is the radius, as on the CPU.
    pub inv_cell: f32,
    /// Bucket count, a power of two.
    pub table_size: u32,
    /// `(start, count)` per bucket, `table_size` of them.
    pub buckets: Vec<[u32; 2]>,
    /// Photons in bucket order: `position.xyz, 0`, `normal.xyz, 0`,
    /// `power.rgb, 0` — twelve floats each.
    pub photons: Vec<f32>,
    /// How many photons `photons` holds.
    pub photon_count: u32,
    /// The photons' bounding box, grown by the radius on every side: a
    /// point outside it gathers nothing, and the shader says so without
    /// walking a bucket.
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
}

/// The bucket a cell hashes to. Mirrors `caustic_bucket` in
/// `integrator.wgsl`: the three primes are Teschner's, and the arithmetic is
/// wrapping `u32`, which is what WGSL's is.
pub fn cell_bucket(cell: [i32; 3], table_size: u32) -> u32 {
    let h = (cell[0] as u32).wrapping_mul(73_856_093)
        ^ (cell[1] as u32).wrapping_mul(19_349_663)
        ^ (cell[2] as u32).wrapping_mul(83_492_791);
    h & (table_size - 1)
}

impl CausticPack {
    /// Lay `map` out for the device.
    ///
    /// The table has at least twice as many buckets as occupied cells (and
    /// never fewer than sixteen), so a bucket is usually one cell.
    pub fn new(map: &CausticMap) -> Self {
        let radius = map.radius as f32;
        let inv_cell = 1.0 / radius;
        let occupied = map.cells.len().max(1);
        let table_size = (occupied * 2).next_power_of_two().max(16) as u32;

        // Bucket each photon by the cell the CPU map put it in — the same
        // `floor(p / r)` in f64 — so the two tiers agree on which photons
        // live where, whatever f32 makes of a point on a cell boundary.
        let mut keyed: Vec<(u32, u32)> = Vec::with_capacity(map.photons.len());
        for (key, list) in &map.cells {
            let cell = [key[0] as i32, key[1] as i32, key[2] as i32];
            let b = cell_bucket(cell, table_size);
            keyed.extend(list.iter().map(|&i| (b, i)));
        }
        keyed.sort_unstable();

        let mut buckets = vec![[0u32; 2]; table_size as usize];
        let mut photons = Vec::with_capacity(keyed.len() * 12);
        for (slot, &(b, i)) in keyed.iter().enumerate() {
            let entry = &mut buckets[b as usize];
            if entry[1] == 0 {
                entry[0] = slot as u32;
            }
            entry[1] += 1;
            let ph = &map.photons[i as usize];
            photons.extend_from_slice(&[
                ph.point.x as f32,
                ph.point.y as f32,
                ph.point.z as f32,
                0.0,
                ph.normal.x as f32,
                ph.normal.y as f32,
                ph.normal.z as f32,
                0.0,
                ph.power[0],
                ph.power[1],
                ph.power[2],
                0.0,
            ]);
        }
        let mut bounds_min = [f32::INFINITY; 3];
        let mut bounds_max = [f32::NEG_INFINITY; 3];
        for ph in photons.chunks_exact(12) {
            for k in 0..3 {
                bounds_min[k] = bounds_min[k].min(ph[k] - radius);
                bounds_max[k] = bounds_max[k].max(ph[k] + radius);
            }
        }
        Self {
            radius,
            inv_cell,
            table_size,
            buckets,
            photons,
            photon_count: keyed.len() as u32,
            bounds_min,
            bounds_max,
        }
    }

    /// Whether there is anything to gather.
    pub fn is_empty(&self) -> bool {
        self.photon_count == 0
    }

    /// The bucket image's size in texels.
    pub fn bucket_image_size(&self) -> (u32, u32) {
        let w = self.table_size.min(PACK_WIDTH);
        (w, self.table_size.div_ceil(w))
    }

    /// The photon image's size in texels.
    pub fn photon_image_size(&self) -> (u32, u32) {
        let texels = (self.photon_count * PACK_TEXELS_PER_PHOTON).max(1);
        (PACK_WIDTH, texels.div_ceil(PACK_WIDTH))
    }

    /// Irradiance at `p`, gathered exactly the way the shader does it — over
    /// the twenty-seven buckets around the point's cell, each visited once,
    /// every photon tested against the radius and the normal. The CPU map's
    /// [`CausticMap::irradiance`] and this must agree; a test holds them to it.
    pub fn irradiance(&self, p: [f32; 3], n: [f32; 3]) -> [f32; 3] {
        if self.is_empty() {
            return [0.0; 3];
        }
        if (0..3).any(|k| p[k] < self.bounds_min[k] || p[k] > self.bounds_max[k]) {
            return [0.0; 3];
        }
        let r2 = self.radius * self.radius;
        let c = [
            (p[0] * self.inv_cell).floor() as i32,
            (p[1] * self.inv_cell).floor() as i32,
            (p[2] * self.inv_cell).floor() as i32,
        ];
        let mut seen: Vec<u32> = Vec::with_capacity(27);
        let mut sum = [0.0f32; 3];
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let b = cell_bucket([c[0] + dx, c[1] + dy, c[2] + dz], self.table_size);
                    if seen.contains(&b) {
                        continue;
                    }
                    seen.push(b);
                    let [start, count] = self.buckets[b as usize];
                    for i in start..start + count {
                        let base = (i * 12) as usize;
                        let ph = &self.photons[base..base + 12];
                        let d = [ph[0] - p[0], ph[1] - p[1], ph[2] - p[2]];
                        if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > r2 {
                            continue;
                        }
                        if ph[4] * n[0] + ph[5] * n[1] + ph[6] * n[2] < 0.9 {
                            continue;
                        }
                        sum[0] += ph[8];
                        sum[1] += ph[9];
                        sum[2] += ph[10];
                    }
                }
            }
        }
        let area = std::f32::consts::PI * r2;
        [sum[0] / area, sum[1] / area, sum[2] / area]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn photon(p: [f64; 3], n: [f64; 3], w: f32) -> Photon {
        Photon {
            point: Point3::new(p[0], p[1], p[2]),
            normal: Vec3::new(n[0], n[1], n[2]),
            power: [w, 2.0 * w, 3.0 * w],
        }
    }

    /// A map whose photons are all on the floor `z = 0` at known radii from
    /// the origin, plus one on the wall next to them.
    fn floor_map(radius: f64) -> CausticMap {
        let up = [0.0, 0.0, 1.0];
        CausticMap::from_photons(
            radius,
            vec![
                photon([0.0, 0.0, 0.0], up, 1.0),
                photon([0.03, 0.0, 0.0], up, 2.0),
                photon([0.0, -0.04, 0.0], up, 4.0),
                photon([0.09, 0.0, 0.0], up, 8.0),
                photon([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 16.0),
                photon([0.4, 0.4, 0.0], up, 32.0),
            ],
        )
    }

    #[test]
    fn power_within_sums_the_photons_in_the_disc() {
        let map = floor_map(0.05);
        let (o, up) = (Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0));
        // nothing but the photon at the centre; the wall photon is edge-on
        // and rejected, as the irradiance gather rejects it
        assert_eq!(map.power_within(o, up, 0.01), [1.0, 2.0, 3.0]);
        // out to 35 mm picks up the second, to 45 mm the third
        assert_eq!(map.power_within(o, up, 0.035), [3.0, 6.0, 9.0]);
        assert_eq!(map.power_within(o, up, 0.045), [7.0, 14.0, 21.0]);
        // the boundary is inclusive, at exactly the photon's distance
        assert_eq!(map.power_within(o, up, 0.03), [3.0, 6.0, 9.0]);
        // and the wall photon is the only thing a wall-facing gather sees
        assert_eq!(map.power_within(o, Vec3::new(1.0, 0.0, 0.0), 0.01), [16.0, 32.0, 48.0]);
    }

    #[test]
    fn power_within_is_zero_where_no_photon_landed() {
        let map = floor_map(0.05);
        let up = Vec3::new(0.0, 0.0, 1.0);
        assert_eq!(map.power_within(Point3::new(1.0, 1.0, 0.0), up, 0.05), [0.0; 3]);
        assert_eq!(map.power_within(Point3::new(0.0, 0.0, 1.0), up, 0.05), [0.0; 3]);
        assert_eq!(map.power_within(Point3::new(0.0, 0.0, 0.0), up, 0.0), [0.0; 3]);
        assert_eq!(CausticMap::empty().power_within(Point3::new(0.0, 0.0, 0.0), up, 1.0), [0.0; 3]);
    }

    /// The grid walk and the scan are two ways of asking one question, and a
    /// radius of a few cells is where the code switches between them.
    #[test]
    fn the_grid_walk_and_the_scan_agree() {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let o = Point3::new(0.0, 0.0, 0.0);
        for r in [0.005, 0.02, 0.031, 0.05, 0.12, 0.5, 1.0] {
            // the same photons in a coarse map (grid walk) and a fine one
            // (scan, because the disc spans more than three cells)
            let coarse = floor_map(1.0).power_within(o, up, r);
            let fine = floor_map(0.001).power_within(o, up, r);
            assert_eq!(coarse, fine, "at r = {r}");
        }
    }

    #[test]
    fn the_disc_is_the_irradiance_gather_without_the_area() {
        let map = floor_map(0.05);
        let (o, up) = (Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0));
        let area = (std::f64::consts::PI * map.radius() * map.radius()) as f32;
        let want = map.power_within(o, up, map.radius());
        let got = map.irradiance(o, up);
        for k in 0..3 {
            assert!((got[k] - want[k] / area).abs() < 1e-5, "{got:?} vs {want:?} / {area}");
        }
    }
}
