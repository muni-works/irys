//! Analytic primitives: the shapes a ray can be solved against exactly.
//!
//! Every client of this crate has, so far, written this file itself. The
//! pool wrote an ellipsoid; the marble level wrote a box, a sphere and a
//! capped cylinder over its phyz colliders; the court will want a torus for
//! a basketball's seam and a rim. They are all the same three questions
//! [`Geometry`] asks, answered with the same quadratic, so they belong here
//! once.
//!
//! What you get for a shape solved rather than tessellated: a silhouette
//! that is a true circle at any zoom, an *exact* normal (the gradient of the
//! implicit, not an interpolated vertex attribute), and a real
//! parameterisation to hang anisotropy and texture on.
//!
//! # Precision, and why every solve re-origins
//!
//! The court is in millimetres. A basketball's seam ten metres from the eye
//! is a torus of `R = 74`, `r = 2` sitting at a world coordinate of `1e4`.
//! Solved from the eye, the quartic's coefficients are built out of `|o| ~
//! 1e4` terms — `o·d`, `o·o`, and a depressed cubic term that cancels two of
//! those against each other — while the answer they have to resolve is the
//! 2 mm tube.
//!
//! So each solve first slides the ray's origin down to its closest approach
//! to the primitive ([`closest_approach`]), forms its polynomial there, and
//! adds the shift back onto the root. `t` is invariant under that slide — it
//! is a rigid translation of the ray along itself — and every coefficient
//! becomes the size of the *primitive* instead of the size of the scene.
//! This is the same move vcad's WGSL BRep tracer makes, for the same reason;
//! it matters less in `f64` than in `f32`, but "less" is not "not at all",
//! and the tests here pin it at `1e4`.
//!
//! # There is no GPU tier here
//!
//! `super::gpu::analytic` exists, and is *not* this. That one is a
//! sphere/plane/box toy so the GPU integrator's own tests have something to
//! trace without a client; it is deliberately not a `GpuGeometry` path for
//! these primitives, and this module does not implement one. A client that
//! wants these shapes on the GPU packs its own slab — the CPU tier is the
//! shared thing, because the CPU tier is where exactness is cheap.

use core::f64::consts::{PI, TAU};

use crate::geometry::Geometry;
use crate::math::{Aabb, Dir3, Point2, Point3, Vec3};
use crate::ray::{Hit, Ray};

/// An orthonormal frame: the primitive's own axes, written in world
/// coordinates.
///
/// Stored as the shape → world rotation (the columns of that matrix are
/// `x`, `y`, `z`), because that is the direction a normal travels. A caller
/// holding a world → shape rotation — phyz's `GeomInstance::origin.rot`, for
/// one — hands over its *rows*.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    /// The shape's local X axis, in world coordinates.
    pub x: Vec3,
    /// The shape's local Y axis, in world coordinates.
    pub y: Vec3,
    /// The shape's local Z axis, in world coordinates.
    pub z: Vec3,
}

impl Default for Frame {
    fn default() -> Self {
        Self::identity()
    }
}

impl Frame {
    /// The world axes: local coordinates *are* world coordinates.
    pub fn identity() -> Self {
        Self {
            x: Vec3::new(1.0, 0.0, 0.0),
            y: Vec3::new(0.0, 1.0, 0.0),
            z: Vec3::new(0.0, 0.0, 1.0),
        }
    }

    /// A frame from three axes, taken as given. Orthonormality is the
    /// caller's promise; nothing here re-orthogonalises, because a caller
    /// that has a rotation matrix already has one.
    pub fn from_axes(x: Vec3, y: Vec3, z: Vec3) -> Self {
        Self { x, y, z }
    }

    /// A frame whose local Z is `axis`, with X and Y chosen arbitrarily but
    /// *deterministically*: the same axis always yields the same reference
    /// direction, so a surface's `u` does not drift between frames.
    ///
    /// Duff et al.'s branchless orthonormal basis — stable near the poles,
    /// where the naive `cross` with a fixed up-vector degenerates.
    pub fn from_z(axis: Dir3) -> Self {
        let z = *axis.as_ref();
        let sign = if z.z >= 0.0 { 1.0 } else { -1.0 };
        let a = -1.0 / (sign + z.z);
        let b = z.x * z.y * a;
        Self {
            x: Vec3::new(1.0 + sign * z.x * z.x * a, sign * b, -sign * z.x),
            y: Vec3::new(b, sign + z.y * z.y * a, -z.y),
            z,
        }
    }

    /// World vector → local coordinates.
    #[inline]
    pub fn to_local(&self, v: Vec3) -> Vec3 {
        Vec3::new(v.dot(self.x), v.dot(self.y), v.dot(self.z))
    }

    /// Local coordinates → world vector.
    #[inline]
    pub fn to_world(&self, v: Vec3) -> Vec3 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }
}

/// One analytic shape, in world coordinates.
///
/// Every rotated variant carries a [`Frame`] rather than an axis alone where
/// the shape has three distinguishable directions to pin — a box's faces are
/// not interchangeable — and an axis alone where it does not, in which case
/// the reference direction for `u` comes from [`Frame::from_z`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Prim {
    /// A ball.
    Sphere {
        /// Centre.
        center: Point3,
        /// Radius.
        radius: f64,
    },
    /// An oriented box.
    Box {
        /// Centre.
        center: Point3,
        /// Half-extent along each local axis.
        half: Vec3,
        /// Local axes in world coordinates.
        rot: Frame,
    },
    /// A cylinder along its local Z, closed by a disc at each end.
    Cylinder {
        /// Centre of the axis segment.
        center: Point3,
        /// Axis direction; local Z.
        axis: Dir3,
        /// Radius.
        radius: f64,
        /// Half the height along the axis.
        half_height: f64,
    },
    /// A capped conical frustum along its local Z. `radius_top = 0` is a
    /// plain cone, whose apex is the one point where the normal is not
    /// defined and where this reports no hit.
    Cone {
        /// Centre of the axis segment.
        center: Point3,
        /// Axis direction; local Z.
        axis: Dir3,
        /// Radius at `-half_height`.
        radius_base: f64,
        /// Radius at `+half_height`.
        radius_top: f64,
        /// Half the height along the axis.
        half_height: f64,
    },
    /// A torus: a tube of radius `minor` swept around a circle of radius
    /// `major` in the plane normal to `axis`.
    Torus {
        /// Centre of the swept circle.
        center: Point3,
        /// Axis of revolution; local Z.
        axis: Dir3,
        /// Major radius: centre to tube centre.
        major: f64,
        /// Minor radius: the tube.
        minor: f64,
    },
    /// An unbounded plane. Its bounds are enormous by necessity, which makes
    /// it a poor citizen of a [`Bvh`](crate::Bvh) — a ground plane is better
    /// off tested outside the tree, or replaced by a large [`Prim::Disc`].
    Plane {
        /// A point on the plane; the origin of its `uv`.
        point: Point3,
        /// Unit normal.
        normal: Dir3,
    },
    /// A flat disc: a plane clipped to a radius.
    Disc {
        /// Centre.
        center: Point3,
        /// Unit normal.
        normal: Dir3,
        /// Radius.
        radius: f64,
    },
    /// An ellipsoid: the unit sphere in a scaled, rotated frame. Solved as
    /// exactly that — the ray is carried into the frame where the shape is a
    /// unit sphere, and `t` survives the change unchanged because both
    /// origin and direction are carried with it.
    Ellipsoid {
        /// Centre.
        center: Point3,
        /// Local axes in world coordinates.
        rot: Frame,
        /// Semi-axis length along each local axis.
        semi: Vec3,
    },
}

/// The ray parameter at the ray's closest approach to `oc` (the origin's
/// offset from some centre), clamped so the shift never runs backwards past
/// the ray's own start.
///
/// See the module docs: this is what keeps a millimetre primitive's solve
/// out of the scene's coordinate range.
#[inline]
pub fn closest_approach(oc: Vec3, dir: Vec3) -> f64 {
    (-oc.dot(dir) / dir.dot(dir)).max(0.0)
}

/// The smallest `t` strictly inside `(t_min, t_max)`.
fn nearest(hits: &[Hit], t_min: f64, t_max: f64) -> Option<Hit> {
    hits.iter()
        .filter(|h| h.t > t_min && h.t < t_max)
        .fold(None, |best: Option<&Hit>, h| {
            Some(match best {
                Some(b) if b.t <= h.t => b,
                _ => h,
            })
        })
        .copied()
}

/// Both roots of `t² + 2bt + c`, in the numerically stable order — the
/// larger-magnitude root from the quadratic formula, the other by Vieta —
/// so the small root is not the difference of two nearly equal numbers.
fn quadratic_roots(b: f64, c: f64) -> Option<(f64, f64)> {
    quadratic_roots_a(1.0, b, c)
}

/// Same, for `a t² + 2bt + c`.
fn quadratic_roots_a(a: f64, b: f64, c: f64) -> Option<(f64, f64)> {
    if a == 0.0 {
        return None;
    }
    let disc = b * b - a * c;
    if disc < 0.0 {
        return None;
    }
    let root = disc.sqrt();
    // `q` is the root formed without cancellation; the other follows from
    // `t0 · t1 = c / a`.
    let q = if b >= 0.0 { -b - root } else { -b + root };
    let (t0, t1) = if q != 0.0 { (q / a, c / q) } else { (0.0, 0.0) };
    Some(if t0 <= t1 { (t0, t1) } else { (t1, t0) })
}

impl Prim {
    /// A world-space box containing the primitive.
    ///
    /// Tight, not corner-derived, wherever the shape has a closed form for
    /// its own extent: an oriented cylinder's box from the eight corners of a
    /// bounding cuboid is noticeably looser than `|h·a| + r·√(1 − a²)`, and
    /// a loose box is a slower tree.
    pub fn bounds(&self) -> Aabb {
        match *self {
            Prim::Sphere { center, radius } => centered(center, Vec3::new(radius, radius, radius)),
            Prim::Box { center, half, rot } => centered(center, oriented_extent(&rot, half)),
            Prim::Cylinder {
                center,
                axis,
                radius,
                half_height,
            } => centered(center, tube_extent(*axis.as_ref(), half_height, radius)),
            Prim::Cone {
                center,
                axis,
                radius_base,
                radius_top,
                half_height,
            } => centered(
                center,
                tube_extent(*axis.as_ref(), half_height, radius_base.max(radius_top)),
            ),
            Prim::Torus {
                center,
                axis,
                major,
                minor,
            } => {
                let a = *axis.as_ref();
                let e = |ai: f64| major * (1.0 - ai * ai).max(0.0).sqrt() + minor;
                centered(center, Vec3::new(e(a.x), e(a.y), e(a.z)))
            }
            // Big, but finite: an infinity here would poison the SAH's
            // surface areas. A plane in a tree is a bad idea regardless.
            Prim::Plane { point, .. } => centered(point, Vec3::new(1e15, 1e15, 1e15)),
            Prim::Disc {
                center,
                normal,
                radius,
            } => centered(center, tube_extent(*normal.as_ref(), 0.0, radius)),
            Prim::Ellipsoid { center, rot, semi } => {
                // The support function of an ellipsoid along world axis e:
                // √Σⱼ (semiⱼ · (axisⱼ · e))².
                let e = |pick: fn(Vec3) -> f64| {
                    let (a, b, c) = (
                        semi.x * pick(rot.x),
                        semi.y * pick(rot.y),
                        semi.z * pick(rot.z),
                    );
                    (a * a + b * b + c * c).sqrt()
                };
                centered(center, Vec3::new(e(|v| v.x), e(|v| v.y), e(|v| v.z)))
            }
        }
    }

    /// Every hit on this primitive inside `(t_min, t_max)`, appended to
    /// `out`, in no particular order.
    ///
    /// Not [`Prim::intersect`] plus a filter: a torus is not convex and a
    /// cylinder is entered and left by the same ray, so this is the
    /// primitive operation and `intersect` is the selection over it.
    pub fn hits(&self, ray: &Ray, prim: u32, t_min: f64, t_max: f64, out: &mut Vec<Hit>) {
        let d = *ray.direction.as_ref();
        let mut keep = |t: f64, n_local: Vec3, uv: Point2, dpdu_local: Vec3, f: &Frame| {
            if !(t > t_min && t < t_max) {
                return;
            }
            let n = f.to_world(n_local);
            let dpdu = f.to_world(dpdu_local);
            out.push(
                Hit::new(t, ray.at(t), Dir3::new_normalize(n), uv, prim)
                    .with_tangent((dpdu.norm() > 0.0).then_some(dpdu))
                    .with_payload(prim as u64),
            );
        };

        match *self {
            Prim::Sphere { center, radius } => {
                let frame = Frame::identity();
                let oc0 = ray.origin - center;
                let t0 = closest_approach(oc0, d);
                let oc = oc0 + t0 * d;
                let Some((r0, r1)) = quadratic_roots(oc.dot(d), oc.dot(oc) - radius * radius)
                else {
                    return;
                };
                for tl in [r0, r1] {
                    // The normal off the *shifted* offset, never off
                    // `ray.at(t) − center`: that difference is two 1e4
                    // coordinates cancelling into a 1e2 radius.
                    let p = oc + tl * d;
                    let n = p / radius;
                    keep(t0 + tl, n, sphere_uv(n), circumferential(p), &frame);
                }
            }

            Prim::Box { center, half, rot } => {
                let o = rot.to_local(ray.origin - center);
                let dl = rot.to_local(d);
                let (oa, da, h) = (o.as_array(), dl.as_array(), half.as_array());
                let (mut t0, mut t1) = (f64::NEG_INFINITY, f64::INFINITY);
                let (mut a0, mut a1) = (0usize, 0usize);
                let (mut s0, mut s1) = (-1.0f64, 1.0f64);
                for k in 0..3 {
                    let inv = 1.0 / da[k];
                    let (mut lo, mut hi) = ((-h[k] - oa[k]) * inv, (h[k] - oa[k]) * inv);
                    let (mut slo, mut shi) = (-1.0, 1.0);
                    if lo > hi {
                        core::mem::swap(&mut lo, &mut hi);
                        core::mem::swap(&mut slo, &mut shi);
                    }
                    if lo > t0 {
                        t0 = lo;
                        a0 = k;
                        s0 = slo;
                    }
                    if hi < t1 {
                        t1 = hi;
                        a1 = k;
                        s1 = shi;
                    }
                    if t1 < t0 {
                        return;
                    }
                }
                for (t, axis, sign) in [(t0, a0, s0), (t1, a1, s1)] {
                    if !t.is_finite() {
                        continue;
                    }
                    let p = (o + t * dl).as_array();
                    let mut n = [0.0; 3];
                    n[axis] = sign;
                    let (uu, vv) = ((axis + 1) % 3, (axis + 2) % 3);
                    let face = 2 * axis + usize::from(sign > 0.0);
                    let s = unit_lerp(p[uu], h[uu]);
                    let mut du = [0.0; 3];
                    du[uu] = 2.0 * h[uu];
                    keep(
                        t,
                        Vec3::new(n[0], n[1], n[2]),
                        // `u` packs the face index with the in-face
                        // coordinate: `face = floor(6u)`, `s = 6u − face`.
                        // One chart over the whole box, and the face is
                        // still recoverable — which a bare per-face `[0,1]²`
                        // is not.
                        Point2::new((face as f64 + s) / 6.0, unit_lerp(p[vv], h[vv])),
                        Vec3::new(du[0], du[1], du[2]),
                        &rot,
                    );
                }
            }

            Prim::Cylinder {
                center,
                axis,
                radius,
                half_height,
            } => {
                let frame = Frame::from_z(axis);
                let o0 = frame.to_local(ray.origin - center);
                let dl = frame.to_local(d);
                let t0 = closest_approach(o0, dl);
                let o = o0 + t0 * dl;
                let a = dl.x * dl.x + dl.y * dl.y;
                if a > 0.0
                    && let Some((r0, r1)) = quadratic_roots_a(
                        a,
                        o.x * dl.x + o.y * dl.y,
                        o.x * o.x + o.y * o.y - radius * radius,
                    )
                {
                    for tl in [r0, r1] {
                        let p = o + tl * dl;
                        if p.z.abs() <= half_height {
                            keep(
                                t0 + tl,
                                Vec3::new(p.x, p.y, 0.0) / radius,
                                Point2::new(azimuth(p.x, p.y), unit_lerp(p.z, half_height)),
                                circumferential(p),
                                &frame,
                            );
                        }
                    }
                }
                if dl.z != 0.0 {
                    for sign in [-1.0f64, 1.0] {
                        let tl = (sign * half_height - o.z) / dl.z;
                        let p = o + tl * dl;
                        let rr = p.x * p.x + p.y * p.y;
                        if rr <= radius * radius {
                            keep(
                                t0 + tl,
                                Vec3::new(0.0, 0.0, sign),
                                // A cap shares the side's `u`; its `v` is the
                                // radial fraction, and the normal is what
                                // tells a cap hit from a side hit.
                                Point2::new(azimuth(p.x, p.y), rr.sqrt() / radius),
                                circumferential(p),
                                &frame,
                            );
                        }
                    }
                }
            }

            Prim::Cone {
                center,
                axis,
                radius_base,
                radius_top,
                half_height,
            } => {
                let frame = Frame::from_z(axis);
                let o0 = frame.to_local(ray.origin - center);
                let dl = frame.to_local(d);
                let t0 = closest_approach(o0, dl);
                let o = o0 + t0 * dl;
                // r(z) = m·z + k, so the side is x² + y² − (m·z + k)² = 0.
                let m = (radius_top - radius_base) / (2.0 * half_height);
                let k = 0.5 * (radius_top + radius_base);
                let (rz_o, rz_d) = (m * o.z + k, m * dl.z);
                let a = dl.x * dl.x + dl.y * dl.y - rz_d * rz_d;
                let b = o.x * dl.x + o.y * dl.y - rz_o * rz_d;
                let c = o.x * o.x + o.y * o.y - rz_o * rz_o;
                let roots: [Option<f64>; 2] = if a != 0.0 {
                    match quadratic_roots_a(a, b, c) {
                        Some((r0, r1)) => [Some(r0), Some(r1)],
                        None => [None, None],
                    }
                } else if b != 0.0 {
                    // Degenerate: the ray runs parallel to a ruling.
                    [Some(-0.5 * c / b), None]
                } else {
                    [None, None]
                };
                for tl in roots.into_iter().flatten() {
                    let p = o + tl * dl;
                    let r_here = m * p.z + k;
                    if p.z.abs() <= half_height && r_here > 0.0 {
                        keep(
                            t0 + tl,
                            // ∇(x² + y² − r(z)²) = (2x, 2y, −2·r·m).
                            Vec3::new(p.x, p.y, -r_here * m),
                            Point2::new(azimuth(p.x, p.y), unit_lerp(p.z, half_height)),
                            circumferential(p),
                            &frame,
                        );
                    }
                }
                if dl.z != 0.0 {
                    for (sign, rad) in [(-1.0f64, radius_base), (1.0, radius_top)] {
                        if rad <= 0.0 {
                            continue;
                        }
                        let tl = (sign * half_height - o.z) / dl.z;
                        let p = o + tl * dl;
                        let rr = p.x * p.x + p.y * p.y;
                        if rr <= rad * rad {
                            keep(
                                t0 + tl,
                                Vec3::new(0.0, 0.0, sign),
                                Point2::new(azimuth(p.x, p.y), rr.sqrt() / rad),
                                circumferential(p),
                                &frame,
                            );
                        }
                    }
                }
            }

            Prim::Torus {
                center,
                axis,
                major,
                minor,
            } => {
                let frame = Frame::from_z(axis);
                let o0 = frame.to_local(ray.origin - center);
                let dl = frame.to_local(d);
                let t0 = closest_approach(o0, dl);
                let o = o0 + t0 * dl;

                let (r2, a2) = (major * major, minor * minor);
                let od = o.dot(dl);
                let oo = o.dot(o);
                let dd = dl.dot(dl);
                let (oa, da) = (o.z, dl.z);
                let k = oo - (r2 + a2);

                let c4 = dd * dd;
                let c3 = 4.0 * dd * od;
                let c2 = 2.0 * dd * k + 4.0 * od * od + 4.0 * r2 * da * da;
                let c1 = 4.0 * k * od + 8.0 * r2 * oa * da;
                let c0 = k * k - 4.0 * r2 * (a2 - oa * oa);

                let mut roots = solve_quartic(c4, c3, c2, c1, c0);
                roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
                roots.dedup_by(|a, b| (*a - *b).abs() <= 1e-9 * (1.0 + a.abs()));

                for tl in roots {
                    // Ferrari's reconstruction cannot return digits the
                    // depressed coefficients never had; two Newton steps on
                    // the torus's own implicit put them back, and also throw
                    // out the plausible-looking numbers a missing ray's
                    // resolvent produces.
                    let Some(tl) = polish_torus(o, dl, major, minor, tl) else {
                        continue;
                    };
                    let p = o + tl * dl;
                    let s = (p.x * p.x + p.y * p.y).sqrt();
                    if s <= 0.0 {
                        continue;
                    }
                    // ∇((√(x²+y²) − R)² + z² − r²).
                    let f = (s - major) / s;
                    keep(
                        t0 + tl,
                        Vec3::new(p.x * f, p.y * f, p.z),
                        Point2::new(azimuth(p.x, p.y), wrap_unit(p.z.atan2(s - major))),
                        circumferential(p),
                        &frame,
                    );
                }
            }

            Prim::Plane { point, normal } => {
                let frame = Frame::from_z(normal);
                let n = *normal.as_ref();
                let denom = d.dot(n);
                if denom == 0.0 {
                    return;
                }
                let q = ray.origin - point;
                let t = -q.dot(n) / denom;
                // `q + t·d`, not `ray.at(t) − point`: differencing two 1e4
                // world coordinates *after* the march costs a millimetre in
                // the uv, which on a court floor is the whole texture.
                let p = q + t * d;
                keep(
                    t,
                    n,
                    Point2::new(p.dot(frame.x), p.dot(frame.y)),
                    frame.x,
                    &Frame::identity(),
                );
            }

            Prim::Disc {
                center,
                normal,
                radius,
            } => {
                let frame = Frame::from_z(normal);
                let n = *normal.as_ref();
                let denom = d.dot(n);
                if denom == 0.0 {
                    return;
                }
                let q = ray.origin - center;
                let t = -q.dot(n) / denom;
                let p = q + t * d;
                let (x, y) = (p.dot(frame.x), p.dot(frame.y));
                if x * x + y * y <= radius * radius {
                    keep(
                        t,
                        n,
                        Point2::new(azimuth(x, y), (x * x + y * y).sqrt() / radius),
                        frame.x,
                        &Frame::identity(),
                    );
                }
            }

            Prim::Ellipsoid { center, rot, semi } => {
                // Into the frame where the shape is the unit sphere. The
                // scaling divides origin *and* direction by the same
                // semi-axes, so `t` means the same thing on both sides and
                // comes back needing no correction.
                let o0 = componentwise_div(rot.to_local(ray.origin - center), semi);
                let dl = componentwise_div(rot.to_local(d), semi);
                let t0 = closest_approach(o0, dl);
                let o = o0 + t0 * dl;
                let Some((r0, r1)) = quadratic_roots_a(dl.dot(dl), o.dot(dl), o.dot(o) - 1.0)
                else {
                    return;
                };
                for tl in [r0, r1] {
                    let p = o + tl * dl;
                    keep(
                        t0 + tl,
                        // ∇Σ(xᵢ/aᵢ)² = 2·xᵢ/aᵢ²: the unit-sphere point
                        // divided once more by the semi-axes.
                        componentwise_div(p, semi),
                        sphere_uv(p),
                        // The unit sphere's dP/du, carried back out through
                        // the scale.
                        Vec3::new(-p.y * semi.x, p.x * semi.y, 0.0) * TAU,
                        &rot,
                    );
                }
            }
        }
    }

    /// The nearest hit strictly inside `(t_min, t_max)`.
    pub fn intersect(&self, ray: &Ray, prim: u32, t_min: f64, t_max: f64) -> Option<Hit> {
        let mut hits = Vec::with_capacity(4);
        self.hits(ray, prim, t_min, t_max, &mut hits);
        nearest(&hits, t_min, t_max)
    }
}

fn centered(c: Point3, e: Vec3) -> Aabb {
    Aabb::new(
        Point3::new(c.x - e.x, c.y - e.y, c.z - e.z),
        Point3::new(c.x + e.x, c.y + e.y, c.z + e.z),
    )
}

/// Half-extent of an oriented box: `Σ hᵢ·|axisᵢ|`, per world axis.
fn oriented_extent(f: &Frame, half: Vec3) -> Vec3 {
    let (a, b, c) = (f.x * half.x, f.y * half.y, f.z * half.z);
    Vec3::new(
        a.x.abs() + b.x.abs() + c.x.abs(),
        a.y.abs() + b.y.abs() + c.y.abs(),
        a.z.abs() + b.z.abs() + c.z.abs(),
    )
}

/// Half-extent of a capped tube of half-height `h` and radius `r` about a
/// unit `axis`: `|h·aᵢ| + r·√(1 − aᵢ²)`. Exact, and tighter than the eight
/// corners of a bounding cuboid.
fn tube_extent(axis: Vec3, h: f64, r: f64) -> Vec3 {
    let e = |a: f64| (h * a).abs() + r * (1.0 - a * a).max(0.0).sqrt();
    Vec3::new(e(axis.x), e(axis.y), e(axis.z))
}

fn componentwise_div(v: Vec3, by: Vec3) -> Vec3 {
    Vec3::new(v.x / by.x, v.y / by.y, v.z / by.z)
}

/// `x` in `[-h, h]` mapped to `[0, 1]`.
fn unit_lerp(x: f64, h: f64) -> f64 {
    if h > 0.0 { 0.5 * (x / h + 1.0) } else { 0.5 }
}

/// The azimuth of `(x, y)` as a fraction of a turn, in `[0, 1)`.
fn azimuth(x: f64, y: f64) -> f64 {
    wrap_unit(y.atan2(x))
}

fn wrap_unit(angle: f64) -> f64 {
    let u = angle / TAU;
    let u = u - u.floor();
    if u >= 1.0 { 0.0 } else { u }
}

/// Lat-long on the unit sphere, with the pole on local Z: `u` the azimuth,
/// `v` from the south pole to the north.
fn sphere_uv(n: Vec3) -> Point2 {
    Point2::new(
        azimuth(n.x, n.y),
        (n.z.clamp(-1.0, 1.0).acos() / PI).clamp(0.0, 1.0),
    )
}

/// `dP/du` for every `u` that is an azimuth: the circumferential direction,
/// which is the grain a lathe leaves and a wound ball is wound along.
fn circumferential(p: Vec3) -> Vec3 {
    Vec3::new(-p.y, p.x, 0.0) * TAU
}

/// Newton on `(√(x²+y²) − R)² + z² − r²` along the ray, and a residual test
/// that rejects a root the quartic invented.
fn polish_torus(o: Vec3, d: Vec3, major: f64, minor: f64, t: f64) -> Option<f64> {
    let mut t = t;
    for _ in 0..3 {
        let p = o + t * d;
        let s = (p.x * p.x + p.y * p.y).sqrt();
        if s <= 0.0 {
            return None;
        }
        let k = s - major;
        let f = k * k + p.z * p.z - minor * minor;
        let g = Vec3::new(2.0 * k * p.x / s, 2.0 * k * p.y / s, 2.0 * p.z);
        let fp = g.dot(d);
        if fp == 0.0 {
            break;
        }
        t -= f / fp;
        if !t.is_finite() {
            return None;
        }
    }
    let p = o + t * d;
    let s = (p.x * p.x + p.y * p.y).sqrt();
    let k = s - major;
    let resid = k * k + p.z * p.z - minor * minor;
    // Relative to the tube: a 2 mm seam and a 400 mm rim want the same test.
    (resid.abs() <= 1e-9 * minor * minor).then_some(t)
}

/// A bag of [`Prim`]s: the [`Geometry`] every analytic client needs.
///
/// The hit's `prim` is the index into [`Analytic::prims`], and its `payload`
/// is that same index — so a caller that stacks this behind a
/// [`Tlas`](crate::Tlas), where `prim` is rewritten per instance, still
/// recovers which shape it was.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Analytic {
    prims: Vec<Prim>,
}

impl Analytic {
    /// An empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// A set from primitives.
    pub fn from_prims(prims: Vec<Prim>) -> Self {
        Self { prims }
    }

    /// The primitives, in the order the tracer indexes them.
    pub fn prims(&self) -> &[Prim] {
        &self.prims
    }

    /// Add one; the index it takes is its `Hit::prim`.
    pub fn push(&mut self, prim: Prim) -> &mut Self {
        self.prims.push(prim);
        self
    }

    /// A single ball — the one-liner every test wants.
    pub fn sphere(center: Point3, radius: f64) -> Self {
        Self::from_prims(vec![Prim::Sphere { center, radius }])
    }
}

impl FromIterator<Prim> for Analytic {
    fn from_iter<I: IntoIterator<Item = Prim>>(iter: I) -> Self {
        Self::from_prims(iter.into_iter().collect())
    }
}

impl Geometry for Analytic {
    fn len(&self) -> usize {
        self.prims.len()
    }

    fn bounds(&self, i: usize) -> Aabb {
        self.prims[i].bounds()
    }

    fn intersect(&self, ray: &Ray, i: usize, t_min: f64, t_max: f64) -> Option<Hit> {
        self.prims[i].intersect(ray, i as u32, t_min, t_max)
    }

    fn intersect_all(&self, ray: &Ray, i: usize, out: &mut Vec<Hit>) {
        // Not the default: a torus is not convex, and a cylinder is entered
        // and left by the same ray.
        self.prims[i].hits(ray, i as u32, 0.0, f64::INFINITY, out);
    }
}

// ---------------------------------------------------------------------------
// Polynomials
// ---------------------------------------------------------------------------

/// Real roots of `a·x⁴ + b·x³ + c·x² + d·x + e`, by Ferrari's method.
fn solve_quartic(a: f64, b: f64, c: f64, d: f64, e: f64) -> Vec<f64> {
    if a == 0.0 {
        return solve_cubic(b, c, d, e);
    }
    let (p, q, r, s) = (b / a, c / a, d / a, e / a);

    let p2 = p * p;
    let a2 = q - 3.0 * p2 / 8.0;
    let a1 = r - p * q / 2.0 + p2 * p / 8.0;
    let a0 = s - p * r / 4.0 + p2 * q / 16.0 - 3.0 * p2 * p2 / 256.0;

    // Resolvent cubic 8u³ + 8·a2·u² + (2·a2² − 8·a0)·u − a1² = 0. The
    // reconstruction below reads `u` on *this* scale; solved on any other it
    // hands back four numbers that satisfy nothing.
    let u = solve_cubic(8.0, 8.0 * a2, 2.0 * a2 * a2 - 8.0 * a0, -a1 * a1)
        .into_iter()
        .filter(|u| *u > 0.0)
        .fold(None, |best: Option<f64>, u| {
            Some(best.map_or(u, |b: f64| b.max(u)))
        })
        .unwrap_or(0.0);

    let sqrt_2u = (2.0 * u).max(0.0).sqrt();
    let shift = p / 4.0;
    let mut roots = Vec::with_capacity(4);

    if sqrt_2u > 0.0 {
        // y⁴ + a2·y² + a1·y + a0 = (y² + a2/2 + u)² − 2u·(y − a1/(4u))², so
        // the quartic splits into the two quadratics below. Their signs are
        // *not* interchangeable: the `+beta` constant belongs to the
        // `−√(2u)·y` factor. Pair them the other way and a renderer draws a
        // torus half again as big as the real one — invisible in a
        // through-the-axis test, where a1 vanishes and both pairings agree.
        let alpha = a2 + 2.0 * u;
        let beta = a1 / sqrt_2u;
        for sign in [1.0f64, -1.0] {
            let disc = 2.0 * u - 2.0 * (alpha + sign * beta);
            if disc >= 0.0 {
                let sq = disc.sqrt();
                roots.push((sign * sqrt_2u + sq) / 2.0 - shift);
                roots.push((sign * sqrt_2u - sq) / 2.0 - shift);
            }
        }
    } else {
        // Biquadratic: y⁴ + a2·y² + a0 = 0.
        let disc = a2 * a2 - 4.0 * a0;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            for y2 in [(-a2 + sq) / 2.0, (-a2 - sq) / 2.0] {
                if y2 >= 0.0 {
                    let y = y2.sqrt();
                    roots.push(y - shift);
                    roots.push(-y - shift);
                }
            }
        }
    }
    roots
}

/// Real roots of `a·x³ + b·x² + c·x + d`, by Cardano / Vieta.
fn solve_cubic(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
    if a == 0.0 {
        return solve_quadratic(b, c, d);
    }
    let (p, q, r) = (b / a, c / a, d / a);
    let p2 = p * p;
    let aa = q - p2 / 3.0;
    let bb = r - p * q / 3.0 + 2.0 * p2 * p / 27.0;
    let delta = bb * bb / 4.0 + aa * aa * aa / 27.0;
    let shift = p / 3.0;

    if delta > 0.0 {
        let sq = delta.sqrt();
        vec![cbrt(-bb / 2.0 + sq) + cbrt(-bb / 2.0 - sq) - shift]
    } else if delta == 0.0 {
        if aa == 0.0 && bb == 0.0 {
            vec![-shift]
        } else {
            let u = cbrt(-bb / 2.0);
            vec![2.0 * u - shift, -u - shift]
        }
    } else {
        // Three real roots: the trigonometric form, which needs no complex
        // cube root and does not lose the two Cardano would drop.
        let m = 2.0 * (-aa / 3.0).sqrt();
        let theta = (3.0 * bb / (aa * m)).clamp(-1.0, 1.0).acos() / 3.0;
        vec![
            m * theta.cos() - shift,
            m * (theta - 2.0 * PI / 3.0).cos() - shift,
            m * (theta + 2.0 * PI / 3.0).cos() - shift,
        ]
    }
}

fn solve_quadratic(a: f64, b: f64, c: f64) -> Vec<f64> {
    if a == 0.0 {
        return if b != 0.0 { vec![-c / b] } else { Vec::new() };
    }
    match quadratic_roots_a(a, b / 2.0, c) {
        Some((t0, t1)) => vec![t0, t1],
        None => Vec::new(),
    }
}

fn cbrt(x: f64) -> f64 {
    if x >= 0.0 {
        x.powf(1.0 / 3.0)
    } else {
        -(-x).powf(1.0 / 3.0)
    }
}

#[cfg(test)]
mod tests;
