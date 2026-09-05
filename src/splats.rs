//! Gaussian splats as ray-traceable primitives.
//!
//! A 3D Gaussian splat is not a surface. It is a blob of density — an
//! anisotropic Gaussian with a colour that varies with view direction — and
//! a scene made of a few hundred thousand of them is rendered by *sorting
//! and compositing*, not by finding where a ray stops. That is what a
//! rasteriser like `tang-3dgs` does: project every Gaussian to the screen,
//! sort by depth, accumulate `C += T·α·c; T *= (1 − α)` front to back.
//!
//! This file puts the same cloud behind the [`Geometry`] seam so a ray
//! tracer can ask the same questions of it — with one honest compromise
//! spelled out below.
//!
//! # What "the hit" means for a blob
//!
//! There is no surface to intersect, so [`Splats::intersect`] reports the
//! **maximum-response point**: the `t` at which the Gaussian's density along
//! the ray peaks. For a Gaussian with inverse covariance `A`, the exponent
//! along `p(t) = o + t·d` is a quadratic in `t`, so the peak is closed form:
//!
//! ```text
//! Δ = o − μ,   t* = −(dᵀAΔ) / (dᵀAd),   d² = q(t*) = (Δ + t*d)ᵀ A (Δ + t*d)
//! ```
//!
//! and the response there is `α · exp(−½·d²)` — exactly the per-splat alpha
//! the rasteriser computes, only evaluated in 3D against the ray instead of
//! in 2D against a pixel. `t*` is the depth to sort by and `d²` is the
//! squared Mahalanobis distance of closest approach.
//!
//! Splats farther than 3σ (`d² > 9`) are reported as misses, which keeps the
//! intersector consistent with [`Splats::bounds`] — the 3σ box — so the tree
//! never culls a hit the intersector would have accepted, or vice versa.
//!
//! ## The payload
//!
//! `Hit::payload` packs two things into its 64 bits:
//!
//! ```text
//! bits  0..32   splat index (u32), the same value as `Hit::prim`
//! bits 32..64   peak opacity α·exp(−½·d²), as `f32::to_bits`
//! ```
//!
//! Unpack with [`unpack_payload`]. The alpha rides in the payload rather
//! than being recomputed because the compositor needs it for every splat
//! along the ray and the intersector has already paid for it.
//!
//! `Hit::normal` is the ellipsoid's gradient direction at the peak point,
//! `normalize(A(p − μ))`, offered for shading a splat as if it had a
//! surface. Note it is always perpendicular to the ray: the peak point is by
//! definition where the ray is tangent to a level set. `Hit::uv` is unused
//! and zero — a Gaussian has no parameterisation.
//!
//! # The compositing path
//!
//! [`composite`] does the walk, and the integrator calls it once per ray
//! segment:
//!
//! 1. Trace the splat [`Bvh`] with [`Geometry::intersect_all`] (or
//!    [`Bvh::trace`], which sorts) to get every splat along the ray in
//!    increasing `t`.
//! 2. Walk them front to back with `T = 1`, `C = 0`, unpacking `α` from each
//!    payload and the colour from [`Splats::colour`] with the ray direction:
//!    `C += T·α·c; T *= (1 − α)`. Break when `T` falls under ~1e-4.
//! 3. Treat the resulting `(C, 1 − T)` as an **emissive backdrop**, not a
//!    BSDF: a scanned splat cloud already has its lighting baked in, so
//!    shading it again double-counts. It contributes radiance directly and
//!    spawns no secondary rays.
//!
//! Mixing with analytic geometry is the same walk with one extra bound:
//! find the nearest opaque analytic hit `t_geo` first, composite splats only
//! for `t < t_geo`, then add `T · L_analytic` for whatever the analytic
//! surface shades to. Splats in front of a wall veil it; splats behind it
//! are never reached. Shadow rays use the same accumulation for
//! transmittance ([`transmittance`]), at the cost of the sort.
//!
//! `Scene::splats` in [`crate::pathtrace`] is where that lands: because the
//! walk runs on *bounce* rays too, a cloud is an environment with depth — it
//! supplies the radiance for rays that find no analytic surface, so an
//! analytic object dropped inside a capture is lit by the captured room.
//!
//! [`Bvh`]: crate::Bvh
//! [`Bvh::trace`]: crate::Bvh::trace

use crate::bvh::Bvh;
use crate::geometry::Geometry;
use crate::math::{Aabb, Dir3, Point2, Point3, Vec3};
use crate::ray::{Hit, Ray};

/// How many standard deviations the bounds — and so the intersector — reach.
///
/// Three is the rasteriser's convention: `exp(−4.5) ≈ 1.1 %` of the peak,
/// below the 1/255 alpha cut-off for anything but a fully opaque splat.
const CUTOFF_SIGMA: f64 = 3.0;

/// Squared Mahalanobis distance at the cut-off.
const CUTOFF_SQ: f64 = CUTOFF_SIGMA * CUTOFF_SIGMA;

/// A cloud of anisotropic Gaussians.
///
/// Stored ready to intersect: each splat keeps the six unique entries of its
/// inverse covariance and the half-extents of its 3σ box, both computed once
/// at construction. The scale and rotation that produced them are not kept —
/// nothing downstream of a ray needs them.
#[derive(Debug, Clone, Default)]
pub struct Splats {
    centers: Vec<Point3>,
    /// Inverse covariance, `[a00, a01, a02, a11, a12, a22]`.
    inv_cov: Vec<[f64; 6]>,
    /// Half-extents of the 3σ axis-aligned box.
    extents: Vec<Vec3>,
    opacities: Vec<f32>,
    /// Spherical harmonic coefficients, `sh_stride` per splat, RGB each.
    sh: Vec<[f32; 3]>,
    sh_stride: usize,
    degree: u32,
}

impl Splats {
    /// A cloud from the arrays a 3DGS trainer or `.ply` loader produces.
    ///
    /// * `positions` — Gaussian means, in world space.
    /// * `scales` — per-axis standard deviations, **linear**, not the log
    ///   scales a `.ply` stores. Exponentiate at the loader; this crate does
    ///   not know that convention and should not guess it.
    /// * `quats` — orientation as `[w, x, y, z]`, matching `tang-3dgs`.
    ///   Normalised here, because a trainer's are only approximately unit.
    /// * `opacities` — the base `α`, already through its sigmoid, in `0..=1`.
    /// * `sh` — coefficients laid out coefficient-major per splat:
    ///   `sh[i * stride + k]` is coefficient `k` of splat `i`, RGB. `stride`
    ///   is `(degree + 1)²` and the degree is inferred from the length; this
    ///   is exactly `GaussianCloud::sh_coeffs` reinterpreted as `[f32; 3]`
    ///   triples, so no repacking is needed at the seam.
    ///
    /// A splat with a non-positive or non-finite scale on any axis has that
    /// axis clamped to a tiny positive value: a degenerate covariance is not
    /// invertible, and a trainer that collapses a Gaussian should not be
    /// able to hand this crate a NaN.
    ///
    /// # Panics
    ///
    /// If the arrays disagree in length, or if `sh` is not a whole number of
    /// square-count coefficients per splat. Both are loader bugs that would
    /// otherwise show up as scrambled colour.
    pub fn from_parts(
        positions: &[[f32; 3]],
        scales: &[[f32; 3]],
        quats: &[[f32; 4]],
        opacities: &[f32],
        sh: &[[f32; 3]],
    ) -> Self {
        let n = positions.len();
        assert_eq!(scales.len(), n, "one scale per splat");
        assert_eq!(quats.len(), n, "one rotation per splat");
        assert_eq!(opacities.len(), n, "one opacity per splat");

        let sh_stride = if n == 0 { 0 } else { sh.len() / n };
        assert_eq!(sh_stride * n, sh.len(), "sh must divide evenly per splat");
        let degree = match sh_stride {
            0 => 0,
            s => {
                let d = (s as f64).sqrt().round() as usize;
                assert_eq!(d * d, s, "sh stride must be (degree + 1)^2, got {s}");
                (d - 1) as u32
            }
        };

        let mut centers = Vec::with_capacity(n);
        let mut inv_cov = Vec::with_capacity(n);
        let mut extents = Vec::with_capacity(n);

        for i in 0..n {
            let p = positions[i];
            centers.push(Point3::new(p[0] as f64, p[1] as f64, p[2] as f64));

            let r = quat_to_mat3(quats[i]);
            let s = [
                sanitize_scale(scales[i][0]),
                sanitize_scale(scales[i][1]),
                sanitize_scale(scales[i][2]),
            ];

            // A = R diag(1/s²) Rᵀ, symmetric by construction.
            let mut a = [0.0f64; 6];
            for k in 0..3 {
                let w = 1.0 / (s[k] * s[k]);
                let (c0, c1, c2) = (r[0][k], r[1][k], r[2][k]);
                a[0] += w * c0 * c0;
                a[1] += w * c0 * c1;
                a[2] += w * c0 * c2;
                a[3] += w * c1 * c1;
                a[4] += w * c1 * c2;
                a[5] += w * c2 * c2;
            }
            inv_cov.push(a);

            // Σ_ii = Σ_k (R_ik s_k)²; the box reaches 3σ along each world
            // axis, which is the tightest axis-aligned box that contains the
            // 3σ ellipsoid.
            let diag = |row: usize| -> f64 {
                (0..3)
                    .map(|k| {
                        let v = r[row][k] * s[k];
                        v * v
                    })
                    .sum::<f64>()
            };
            extents.push(Vec3::new(
                CUTOFF_SIGMA * diag(0).sqrt(),
                CUTOFF_SIGMA * diag(1).sqrt(),
                CUTOFF_SIGMA * diag(2).sqrt(),
            ));
        }

        Self {
            centers,
            inv_cov,
            extents,
            opacities: opacities.to_vec(),
            sh: sh.to_vec(),
            sh_stride,
            degree,
        }
    }

    /// How many splats.
    pub fn count(&self) -> usize {
        self.centers.len()
    }

    /// The spherical-harmonic degree the cloud carries: 0 through 3.
    pub fn degree(&self) -> u32 {
        self.degree
    }

    /// The mean of splat `i`.
    pub fn center(&self, i: usize) -> Point3 {
        self.centers[i]
    }

    /// The base opacity of splat `i`, before the Gaussian falloff.
    pub fn opacity(&self, i: usize) -> f32 {
        self.opacities[i]
    }

    /// The colour of splat `i` seen from direction `dir`.
    ///
    /// `dir` points **from the camera toward the splat** — the ray's own
    /// direction, which is what the caller has. Evaluates the stored
    /// harmonics up to degree 3, adds the ½ offset the 3DGS convention bakes
    /// into the DC term, and clamps at zero. Coefficients above the cloud's
    /// degree are simply absent and contribute nothing, so a degree-0 cloud
    /// returns a constant colour for every direction.
    ///
    /// Colour lookup lives here rather than on the [`Hit`] because it needs
    /// the view direction, which a hit does not carry, and because a
    /// compositor wants it only for the splats that survive the alpha test.
    pub fn colour(&self, i: usize, dir: Dir3) -> [f32; 3] {
        let mut c = [0.5f32; 3];
        if self.sh_stride == 0 {
            return c;
        }
        let base = i * self.sh_stride;
        let at = |k: usize| self.sh[base + k];

        let add = |c: &mut [f32; 3], w: f32, v: [f32; 3]| {
            for ch in 0..3 {
                c[ch] += w * v[ch];
            }
        };

        add(&mut c, SH_C0, at(0));

        if self.degree >= 1 && self.sh_stride >= 4 {
            let (x, y, z) = (dir.x as f32, dir.y as f32, dir.z as f32);
            add(&mut c, -SH_C1 * y, at(1));
            add(&mut c, SH_C1 * z, at(2));
            add(&mut c, -SH_C1 * x, at(3));

            if self.degree >= 2 && self.sh_stride >= 9 {
                let (xx, yy, zz) = (x * x, y * y, z * z);
                let (xy, yz, xz) = (x * y, y * z, x * z);
                add(&mut c, SH_C2[0] * xy, at(4));
                add(&mut c, SH_C2[1] * yz, at(5));
                add(&mut c, SH_C2[2] * (2.0 * zz - xx - yy), at(6));
                add(&mut c, SH_C2[3] * xz, at(7));
                add(&mut c, SH_C2[4] * (xx - yy), at(8));

                if self.degree >= 3 && self.sh_stride >= 16 {
                    add(&mut c, SH_C3[0] * y * (3.0 * xx - yy), at(9));
                    add(&mut c, SH_C3[1] * xy * z, at(10));
                    add(&mut c, SH_C3[2] * y * (4.0 * zz - xx - yy), at(11));
                    add(&mut c, SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy), at(12));
                    add(&mut c, SH_C3[4] * x * (4.0 * zz - xx - yy), at(13));
                    add(&mut c, SH_C3[5] * z * (xx - yy), at(14));
                    add(&mut c, SH_C3[6] * x * (xx - 3.0 * yy), at(15));
                }
            }
        }

        [c[0].max(0.0), c[1].max(0.0), c[2].max(0.0)]
    }

    /// `A · v` for splat `i`, with `A` the inverse covariance.
    #[inline]
    fn mul_a(&self, i: usize, v: Vec3) -> Vec3 {
        let a = &self.inv_cov[i];
        Vec3::new(
            a[0] * v.x + a[1] * v.y + a[2] * v.z,
            a[1] * v.x + a[3] * v.y + a[4] * v.z,
            a[2] * v.x + a[4] * v.y + a[5] * v.z,
        )
    }
}

/// Split a splat [`Hit::payload`] into `(splat index, peak alpha)`.
///
/// The inverse of the packing documented at the top of this module.
#[inline]
pub fn unpack_payload(payload: u64) -> (u32, f32) {
    (
        payload as u32,
        f32::from_bits((payload >> 32) as u32),
    )
}

/// Pack a splat index and its peak alpha into a payload word.
#[inline]
fn pack_payload(index: u32, alpha: f32) -> u64 {
    ((alpha.to_bits() as u64) << 32) | index as u64
}

/// Keep a scale strictly positive and finite so the covariance inverts.
#[inline]
fn sanitize_scale(s: f32) -> f64 {
    let s = s as f64;
    if s.is_finite() && s > 1e-9 { s } else { 1e-9 }
}

/// Rotation matrix from a `[w, x, y, z]` quaternion, normalised first.
fn quat_to_mat3(q: [f32; 4]) -> [[f64; 3]; 3] {
    let (w, x, y, z) = (q[0] as f64, q[1] as f64, q[2] as f64, q[3] as f64);
    let n = (w * w + x * x + y * y + z * z).sqrt();
    if !(n > 1e-12) {
        // A zero quaternion is no rotation, not a NaN.
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let (w, x, y, z) = (w / n, x / n, y / n, z / n);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// Real spherical harmonic normalisation constants, degrees 0 through 3, in
/// the order the 3DGS reference implementation stores its coefficients.
const SH_C0: f32 = 0.282_094_79;
const SH_C1: f32 = 0.488_602_5;
const SH_C2: [f32; 5] = [
    1.092_548_4,
    -1.092_548_4,
    0.315_391_57,
    -1.092_548_4,
    0.546_274_2,
];
const SH_C3: [f32; 7] = [
    -0.590_043_6,
    2.890_611_4,
    -0.457_045_8,
    0.373_176_33,
    -0.457_045_8,
    1.445_305_7,
    -0.590_043_6,
];

impl Geometry for Splats {
    fn len(&self) -> usize {
        self.centers.len()
    }

    fn bounds(&self, i: usize) -> Aabb {
        let c = self.centers[i];
        let e = self.extents[i];
        Aabb::new(
            Point3::new(c.x - e.x, c.y - e.y, c.z - e.z),
            Point3::new(c.x + e.x, c.y + e.y, c.z + e.z),
        )
    }

    fn intersect(&self, ray: &Ray, i: usize, t_min: f64, t_max: f64) -> Option<Hit> {
        let d = *ray.direction.as_ref();
        let delta = ray.origin - self.centers[i];

        let a_d = self.mul_a(i, d);
        let denom = d.dot(a_d);
        if !(denom > 0.0) {
            // Only possible for a degenerate covariance; nothing to report.
            return None;
        }

        // The exponent along the ray is q(t) = t²·denom + 2t·(dᵀAΔ) + ΔᵀAΔ,
        // a convex parabola, so its minimum is where the derivative vanishes.
        let a_delta = self.mul_a(i, delta);
        let t = -d.dot(a_delta) / denom;
        if !(t > t_min && t < t_max) {
            return None;
        }

        let closest = delta + t * d;
        let d2 = closest.dot(self.mul_a(i, closest)).max(0.0);
        if d2 > CUTOFF_SQ {
            return None;
        }

        let alpha = self.opacities[i] * (-0.5 * d2).exp() as f32;
        let grad = self.mul_a(i, closest);
        let normal = if grad.norm() > 1e-30 {
            Dir3::new_normalize(grad)
        } else {
            // Dead centre: the gradient vanishes, so face the ray.
            Dir3::new_normalize(-d)
        };

        Some(
            Hit::new(t, ray.at(t), normal, Point2::new(0.0, 0.0), i as u32)
                .with_payload(pack_payload(i as u32, alpha)),
        )
    }

    fn intersect_all(&self, ray: &Ray, i: usize, out: &mut Vec<Hit>) {
        // A Gaussian has exactly one peak along a ray. The *sorting* a
        // compositor needs happens one level up, in `Bvh::trace`, which
        // gathers these and orders them by `t`.
        if let Some(hit) = self.intersect(ray, i, 0.0, f64::INFINITY) {
            out.push(hit);
        }
    }

    fn occludes(&self, _ray: &Ray, _i: usize, _t_min: f64, _t_max: f64) -> bool {
        // A splat is never opaque enough to stop a shadow ray on its own:
        // occlusion through a cloud is an accumulated transmittance, not a
        // yes/no. Saying "no" here keeps a naive any-hit query from turning
        // a wisp of density into a hard shadow.
        false
    }
}

// ─── the compositing path ─────────────────────────────────────────────────

/// What one ray segment picked up crossing a splat cloud.
///
/// The pair a front-to-back walk produces: the radiance the Gaussians added
/// along the segment, and how much of whatever lies *beyond* the segment
/// still gets through.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplatSegment {
    /// Accumulated radiance, `Σ T·α·c`.
    pub radiance: [f32; 3],
    /// Remaining transmittance, `Π (1 − α)`, in `0..=1`.
    pub transmittance: f32,
}

impl Default for SplatSegment {
    fn default() -> Self {
        Self {
            radiance: [0.0; 3],
            transmittance: 1.0,
        }
    }
}

/// Below this the walk stops: the rest of the cloud cannot change the pixel.
const T_CUTOFF: f32 = 1e-4;

/// Composite the splats a ray meets in `(t_min, t_max)`, front to back.
///
/// The walk of the module header, done: gather every splat along the segment
/// with [`Bvh::trace`] (which sorts by `t`), then accumulate
/// `C += T·α·c(dir); T *= (1 − α)` and stop once `T` is negligible.
///
/// # The model
///
/// A splat cloud is a **captured radiance field**: its colours already
/// include the room's lighting, baked in at capture time. So the cloud is
/// treated as *emissive and absorbing* — it emits [`SplatSegment::radiance`]
/// and it attenuates by [`SplatSegment::transmittance`]. It is never shaded,
/// never spawns a secondary ray, and never receives light from the analytic
/// scene. What it *is*, for the integrator, is an environment with depth:
/// like a lat-long map it supplies radiance for directions that hit nothing,
/// and unlike one it also sits in front of things, veils them, and casts its
/// accumulated opacity across shadow rays.
pub fn composite(bvh: &Bvh<Splats>, ray: &Ray, t_min: f64, t_max: f64) -> SplatSegment {
    let mut seg = SplatSegment::default();
    if bvh.geometry().is_empty() {
        return seg;
    }
    let dir = ray.direction;
    for hit in bvh.trace(ray) {
        if hit.t <= t_min {
            continue;
        }
        if hit.t >= t_max {
            break;
        }
        let (i, alpha) = unpack_payload(hit.payload);
        let alpha = alpha.clamp(0.0, 1.0);
        if alpha <= 0.0 {
            continue;
        }
        let c = bvh.geometry().colour(i as usize, dir);
        let w = seg.transmittance * alpha;
        for ch in 0..3 {
            seg.radiance[ch] += w * c[ch];
        }
        seg.transmittance *= 1.0 - alpha;
        if seg.transmittance <= T_CUTOFF {
            seg.transmittance = 0.0;
            break;
        }
    }
    seg
}

/// How much of a light's radiance survives the crossing — the transmittance
/// half of [`composite`], for a shadow ray, which does not want the colour.
///
/// Still pays for the sort, because the alphas multiply in any order but the
/// early-out needs them front to back. Cheap enough at these densities.
pub fn transmittance(bvh: &Bvh<Splats>, ray: &Ray, t_min: f64, t_max: f64) -> f32 {
    composite(bvh, ray, t_min, t_max).transmittance
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bvh::Bvh;

    /// One isotropic splat of radius `s` at `c` with opacity `a`, degree 0.
    fn single(c: [f32; 3], s: f32, a: f32, colour: [f32; 3]) -> Splats {
        Splats::from_parts(&[c], &[[s, s, s]], &[[1.0, 0.0, 0.0, 0.0]], &[a], &[colour])
    }

    #[test]
    fn an_isotropic_splat_peaks_at_its_centre() {
        let splats = single([0.0, 0.0, 0.0], 0.5, 0.8, [0.0, 0.0, 0.0]);
        let ray = Ray::new(Point3::new(-4.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = splats
            .intersect(&ray, 0, 0.0, f64::INFINITY)
            .expect("straight through the middle");
        assert!((hit.t - 4.0).abs() < 1e-12, "t = {}", hit.t);
        let (idx, alpha) = unpack_payload(hit.payload);
        assert_eq!(idx, 0);
        assert!((alpha - 0.8).abs() < 1e-6, "alpha = {alpha}");
    }

    #[test]
    fn an_offset_ray_gets_the_gaussian_falloff() {
        let s = 0.5f64;
        let splats = single([0.0, 0.0, 0.0], s as f32, 1.0, [0.0, 0.0, 0.0]);
        // One sigma off-axis: the peak response is exp(-1/2).
        let ray = Ray::new(Point3::new(-4.0, s, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hit = splats.intersect(&ray, 0, 0.0, f64::INFINITY).expect("within 3σ");
        let (_, alpha) = unpack_payload(hit.payload);
        assert!(
            (alpha as f64 - (-0.5f64).exp()).abs() < 1e-6,
            "alpha = {alpha}"
        );
        // The normal at the peak is perpendicular to the ray, and points out
        // along the offset.
        assert!(hit.normal.x.abs() < 1e-9);
        assert!(hit.normal.y > 0.9);
    }

    #[test]
    fn a_ray_past_three_sigma_misses() {
        let splats = single([0.0, 0.0, 0.0], 0.5, 1.0, [0.0, 0.0, 0.0]);
        let ray = Ray::new(Point3::new(-4.0, 1.6, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(splats.intersect(&ray, 0, 0.0, f64::INFINITY).is_none());
    }

    #[test]
    fn an_anisotropic_splat_is_wide_where_its_scale_is() {
        // A cigar along x: 1.0 by 0.05 by 0.05, unrotated.
        let splats = Splats::from_parts(
            &[[0.0, 0.0, 0.0]],
            &[[1.0, 0.05, 0.05]],
            &[[1.0, 0.0, 0.0, 0.0]],
            &[1.0],
            &[[0.0, 0.0, 0.0]],
        );
        let b = splats.bounds(0);
        assert!((b.max.x - 3.0).abs() < 1e-5, "3σ along x");
        assert!((b.max.y - 0.15).abs() < 1e-5, "3σ along y");
        // Down the short axis, half a sigma off along x is nothing.
        let ray = Ray::new(Point3::new(0.5, -3.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let hit = splats.intersect(&ray, 0, 0.0, f64::INFINITY).expect("hits");
        let (_, alpha) = unpack_payload(hit.payload);
        assert!((alpha as f64 - (-0.125f64).exp()).abs() < 1e-6, "{alpha}");
    }

    #[test]
    fn two_splats_come_back_sorted_by_t() {
        let splats = Splats::from_parts(
            &[[5.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            &[[0.3; 3], [0.3; 3]],
            &[[1.0, 0.0, 0.0, 0.0]; 2],
            &[0.5, 0.25],
            &[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        );
        let bvh = Bvh::build(splats);
        let ray = Ray::new(Point3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let hits = bvh.trace(&ray);
        assert_eq!(hits.len(), 2);
        assert!((hits[0].t - 2.0).abs() < 1e-9, "the near one first");
        assert!((hits[1].t - 6.0).abs() < 1e-9);
        assert_eq!(unpack_payload(hits[0].payload).0, 1);
        assert_eq!(unpack_payload(hits[1].payload).0, 0);
    }

    #[test]
    fn degree_zero_colour_round_trips() {
        // The DC term that encodes a target colour: (c - 0.5) / C0.
        let want = [0.2f32, 0.7, 0.45];
        let dc = [
            (want[0] - 0.5) / SH_C0,
            (want[1] - 0.5) / SH_C0,
            (want[2] - 0.5) / SH_C0,
        ];
        let splats = single([0.0, 0.0, 0.0], 0.5, 1.0, dc);
        for dir in [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.3),
            Vec3::new(-0.4, 0.2, -0.9),
        ] {
            let got = splats.colour(0, Dir3::new_normalize(dir));
            for ch in 0..3 {
                assert!(
                    (got[ch] - want[ch]).abs() < 1e-6,
                    "channel {ch}: {got:?} vs {want:?}"
                );
            }
        }
    }

    #[test]
    fn higher_degrees_vary_with_direction_and_stay_non_negative() {
        let stride = 16;
        let mut sh = vec![[0.0f32; 3]; stride];
        sh[0] = [0.5, 0.5, 0.5];
        sh[2] = [0.9, -0.4, 0.2]; // a degree-1 z lobe
        sh[12] = [0.3, 0.3, -0.6]; // a degree-3 z lobe
        let splats = Splats::from_parts(
            &[[0.0, 0.0, 0.0]],
            &[[0.3; 3]],
            &[[1.0, 0.0, 0.0, 0.0]],
            &[1.0],
            &sh,
        );
        assert_eq!(splats.degree(), 3);
        let up = splats.colour(0, Dir3::new_normalize(Vec3::new(0.0, 0.0, 1.0)));
        let down = splats.colour(0, Dir3::new_normalize(Vec3::new(0.0, 0.0, -1.0)));
        assert!(up != down, "view dependence should show");
        for c in up.iter().chain(down.iter()) {
            assert!(*c >= 0.0, "colour clamps at zero");
        }
    }

    #[test]
    fn a_shadow_ray_is_never_stopped_by_a_splat() {
        let splats = single([0.0, 0.0, 0.0], 0.5, 1.0, [0.0, 0.0, 0.0]);
        let bvh = Bvh::build(splats);
        let ray = Ray::new(Point3::new(-4.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(bvh.trace_closest(&ray).is_some(), "but it is still hit");
        assert!(!bvh.occluded(&ray, f64::INFINITY));
    }

    #[test]
    fn a_hundred_thousand_splats_build_and_trace() {
        use std::time::Instant;

        let n = 100_000;
        let mut positions = Vec::with_capacity(n);
        let mut scales = Vec::with_capacity(n);
        let mut quats = Vec::with_capacity(n);
        let mut opacities = Vec::with_capacity(n);
        let mut sh = Vec::with_capacity(n);

        // A deterministic hash, so the timing is repeatable.
        let mut state = 0x243f_6a88_85a3_08d3u64;
        let mut rand = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f32 / (1u64 << 53) as f32
        };

        for _ in 0..n {
            positions.push([
                rand() * 20.0 - 10.0,
                rand() * 20.0 - 10.0,
                rand() * 20.0 - 10.0,
            ]);
            let s = 0.01 + rand() * 0.04;
            scales.push([s, s * (0.5 + rand()), s * (0.5 + rand())]);
            quats.push([rand() - 0.5, rand() - 0.5, rand() - 0.5, rand() - 0.5]);
            opacities.push(rand());
            sh.push([rand(), rand(), rand()]);
        }

        let t0 = Instant::now();
        let bvh = Bvh::build(Splats::from_parts(
            &positions, &scales, &quats, &opacities, &sh,
        ));
        let build = t0.elapsed();

        let rays: Vec<Ray> = (0..64)
            .map(|k| {
                let a = k as f64 * 0.19;
                Ray::new(
                    Point3::new(-30.0, a.sin() * 3.0, a.cos() * 3.0),
                    Vec3::new(1.0, a.sin() * 0.05, a.cos() * 0.05),
                )
            })
            .collect();

        // Warm up, then time the steady state.
        let mut warm = 0usize;
        for r in &rays {
            warm += bvh.trace(r).len();
        }
        assert!(warm > 0, "a ray through the middle should find splats");

        let reps = 8;
        let t1 = Instant::now();
        let mut found = 0usize;
        for _ in 0..reps {
            for r in &rays {
                found += bvh.trace(r).len();
            }
        }
        let per_ray = t1.elapsed().as_secs_f64() / (reps * rays.len()) as f64;

        println!(
            "100k splats: build {:.1} ms, trace {:.1} µs/ray ({:.1} splats/ray)",
            build.as_secs_f64() * 1e3,
            per_ray * 1e6,
            found as f64 / (reps * rays.len()) as f64
        );
        assert!(
            per_ray < 1e-3,
            "a ray should take well under a millisecond, took {:.1} µs",
            per_ray * 1e6
        );
    }

    #[test]
    fn an_empty_cloud_is_empty() {
        let splats = Splats::from_parts(&[], &[], &[], &[], &[]);
        assert!(splats.is_empty());
        assert_eq!(splats.degree(), 0);
    }
}
