//! A ray, and what it found.

use crate::math::{Aabb, Dir3, Point2, Point3, Vec3};

/// A ray in 3D space: an origin, a unit direction, and the reciprocals the
/// slab test wants precomputed.
#[derive(Debug, Clone, Copy)]
pub struct Ray {
    /// Origin point of the ray.
    pub origin: Point3,
    /// Unit direction of the ray.
    pub direction: Dir3,
    /// Precomputed reciprocal of the direction, for fast box tests.
    inv_direction: Vec3,
    /// Sign of each direction component (0 positive, 1 negative).
    sign: [usize; 3],
}

impl Ray {
    /// A ray from an origin and a direction. The direction is normalised.
    pub fn new(origin: Point3, direction: Vec3) -> Self {
        let dir = Dir3::new_normalize(direction);
        let inv = Vec3::new(1.0 / dir.x, 1.0 / dir.y, 1.0 / dir.z);
        let sign = [
            if inv.x < 0.0 { 1 } else { 0 },
            if inv.y < 0.0 { 1 } else { 0 },
            if inv.z < 0.0 { 1 } else { 0 },
        ];
        Self { origin, direction: dir, inv_direction: inv, sign }
    }

    /// The point at parameter `t`: `origin + t * direction`.
    #[inline]
    pub fn at(&self, t: f64) -> Point3 {
        self.origin + t * self.direction.as_ref()
    }

    /// Slab test against a box. `Some((enter, exit))` when the ray crosses
    /// it, `None` when it does not. Infinities are handled, so an
    /// axis-aligned ray is not a special case.
    #[inline]
    pub fn intersect_aabb(&self, aabb: &Aabb) -> Option<(f64, f64)> {
        let bounds = [aabb.min, aabb.max];

        let tx1 = (bounds[self.sign[0]].x - self.origin.x) * self.inv_direction.x;
        let tx2 = (bounds[1 - self.sign[0]].x - self.origin.x) * self.inv_direction.x;

        let mut t_min = tx1;
        let mut t_max = tx2;

        let ty1 = (bounds[self.sign[1]].y - self.origin.y) * self.inv_direction.y;
        let ty2 = (bounds[1 - self.sign[1]].y - self.origin.y) * self.inv_direction.y;

        t_min = t_min.max(ty1);
        t_max = t_max.min(ty2);

        let tz1 = (bounds[self.sign[2]].z - self.origin.z) * self.inv_direction.z;
        let tz2 = (bounds[1 - self.sign[2]].z - self.origin.z) * self.inv_direction.z;

        t_min = t_min.max(tz1);
        t_max = t_max.min(tz2);

        if t_max >= t_min && t_max >= 0.0 {
            Some((t_min.max(0.0), t_max))
        } else {
            None
        }
    }
}

/// What a ray found on a surface.
///
/// Deliberately says nothing about *whose* surface. The primitive index and
/// the payload are the two handles back to the geometry that produced it: a
/// BRep implementation looks its face up by `prim`, a splat cloud its
/// gaussian, a collider its shape.
#[derive(Debug, Clone, Copy)]
pub struct Hit {
    /// Parameter along the ray where the intersection occurs.
    pub t: f64,
    /// The intersection point.
    pub point: Point3,
    /// Surface normal there, pointing outward.
    pub normal: Dir3,
    /// Surface parameters `(u, v)`. Barycentric weights for a triangle.
    pub uv: Point2,
    /// Index of the primitive that was hit, into the [`Geometry`] that owns
    /// it. This is how a face id, a triangle index or a splat id is
    /// recovered: ask the geometry.
    ///
    /// [`Geometry`]: crate::Geometry
    pub prim: u32,
    /// A word the geometry may attach to the hit. Zero when it has nothing
    /// to say.
    pub payload: u64,
    /// Surface tangent `dP/du`, when the parameterisation carries a
    /// meaningful direction.
    ///
    /// This is the *grain* of the surface: on a cylinder it is the
    /// circumferential direction, which is exactly the axis a lathe leaves
    /// its marks along. Anisotropic shading orients its specular lobe by it.
    /// `None` where the parameterisation is arbitrary — fitted splines,
    /// triangles, the poles of a sphere.
    ///
    /// Not normalised, and not guaranteed orthogonal to [`Self::normal`]:
    /// orthogonalise before shading with it.
    pub dpdu: Option<Vec3>,
}

impl Hit {
    /// A hit with no tangent.
    pub fn new(t: f64, point: Point3, normal: Dir3, uv: Point2, prim: u32) -> Self {
        Self { t, point, normal, uv, prim, payload: 0, dpdu: None }
    }

    /// Attach a surface tangent.
    pub fn with_tangent(mut self, dpdu: Option<Vec3>) -> Self {
        self.dpdu = dpdu;
        self
    }

    /// Attach a payload word.
    pub fn with_payload(mut self, payload: u64) -> Self {
        self.payload = payload;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_at_walks_the_direction() {
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        let p = ray.at(5.0);
        assert!((p.x - 5.0).abs() < 1e-12);
        assert!(p.y.abs() < 1e-12);
        assert!(p.z.abs() < 1e-12);
    }

    #[test]
    fn slab_test_finds_entry_and_exit() {
        let ray = Ray::new(Point3::new(-5.0, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0));
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        let (t_min, t_max) = ray.intersect_aabb(&aabb).expect("crosses the box");
        assert!((t_min - 5.0).abs() < 1e-10);
        assert!((t_max - 6.0).abs() < 1e-10);
    }

    #[test]
    fn a_ray_beside_the_box_misses() {
        let ray = Ray::new(Point3::new(-5.0, 5.0, 5.0), Vec3::new(1.0, 0.0, 0.0));
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        assert!(ray.intersect_aabb(&aabb).is_none());
    }

    #[test]
    fn an_origin_inside_enters_at_zero() {
        let ray = Ray::new(Point3::new(0.5, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0));
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        let (t_min, t_max) = ray.intersect_aabb(&aabb).expect("starts inside");
        assert!(t_min >= 0.0);
        assert!((t_max - 0.5).abs() < 1e-10);
    }

    #[test]
    fn a_ray_pointing_away_misses() {
        let ray = Ray::new(Point3::new(-5.0, 0.5, 0.5), Vec3::new(-1.0, 0.0, 0.0));
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        assert!(ray.intersect_aabb(&aabb).is_none());
    }
}
