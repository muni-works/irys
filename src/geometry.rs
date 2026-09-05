//! What the tracer needs to know about a thing in order to hit it.

use crate::math::{Aabb, Dir3, Point2, Point3, Vec3};
use crate::ray::{Hit, Ray};

/// A bag of primitives a ray can be tested against.
///
/// This is the whole boundary between this crate and any geometry. A BRep
/// solid implements it with its trimmed analytic faces; a triangle mesh with
/// its triangles ([`TriMesh`]); a splat cloud with its gaussians; a physics
/// world with its colliders. Nothing above this line knows which.
///
/// Primitives are addressed by a flat index in `0..len()`. That index is
/// handed back on every [`Hit`], which is how a caller recovers whatever the
/// geometry actually calls the thing — a face id, a triangle, a body.
pub trait Geometry {
    /// How many primitives there are.
    fn len(&self) -> usize;

    /// Whether there are none.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bounds of one primitive, in the geometry's own space. Used at build
    /// time only, so it may be as loose as it needs to be — but a tight box
    /// is a faster tree.
    fn bounds(&self, i: usize) -> Aabb;

    /// The nearest hit on primitive `i` strictly inside `(t_min, t_max)`.
    ///
    /// The interval is open at both ends: `t_min` is how a caller skips the
    /// surface a ray just left without nudging its origin.
    fn intersect(&self, ray: &Ray, i: usize, t_min: f64, t_max: f64) -> Option<Hit>;

    /// Every hit on primitive `i`, appended to `out`, in any order.
    ///
    /// A primitive need not be convex — a trimmed cylinder is entered and
    /// left by the same ray — so this is not simply [`Self::intersect`]. The
    /// default assumes convexity and reports the one hit; override it where
    /// that is a lie.
    fn intersect_all(&self, ray: &Ray, i: usize, out: &mut Vec<Hit>) {
        if let Some(hit) = self.intersect(ray, i, 0.0, f64::INFINITY) {
            out.push(hit);
        }
    }

    /// Does primitive `i` block the ray inside `(t_min, t_max)`?
    ///
    /// Separate from [`Self::intersect`] because a shadow ray does not care
    /// *which* hit or *where* — it can stop at the first one, and an
    /// implementation that shades its hits should skip that work here.
    fn occludes(&self, ray: &Ray, i: usize, t_min: f64, t_max: f64) -> bool {
        self.intersect(ray, i, t_min, t_max).is_some()
    }
}

/// Barycentric result of a ray-triangle test.
#[derive(Debug, Clone, Copy)]
pub struct TriangleHit {
    /// Parameter along the ray.
    pub t: f64,
    /// Barycentric weight of the second vertex.
    pub u: f64,
    /// Barycentric weight of the third vertex.
    pub v: f64,
}

impl TriangleHit {
    /// Barycentric weight of the first vertex: `1 - u - v`.
    #[inline]
    pub fn w(&self) -> f64 {
        1.0 - self.u - self.v
    }
}

/// Relative epsilon for the determinant test, scaled by the triangle's edge
/// magnitudes so the test is size-independent: an absolute epsilon rejects
/// legitimate hits on millimetre triangles and accepts degenerate ones on
/// metre-scale parts.
const DET_EPS: f64 = 1e-12;

/// Möller–Trumbore, double-sided.
///
/// Double-sided because a mesh has no reliable winding — a decimated or
/// imported one may be inconsistent — and the renderer face-forwards its
/// shading normal anyway, so culling here would only punch holes in parts.
///
/// `None` when the ray misses, runs parallel to the plane, the triangle is
/// degenerate, or the hit lies behind the origin.
pub fn intersect_triangle(ray: &Ray, v0: Point3, v1: Point3, v2: Point3) -> Option<TriangleHit> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let d = ray.direction.as_ref();

    let pvec = d.cross(e2);
    let det = e1.dot(pvec);

    let scale = e1.norm() * e2.norm();
    if det.abs() <= DET_EPS * scale.max(1.0) {
        return None;
    }

    let inv_det = 1.0 / det;
    let tvec = ray.origin - v0;
    let u = tvec.dot(pvec) * inv_det;
    if !(-1e-12..=1.0 + 1e-12).contains(&u) {
        return None;
    }

    let qvec = tvec.cross(e1);
    let v = d.dot(qvec) * inv_det;
    if v < -1e-12 || u + v > 1.0 + 1e-12 {
        return None;
    }

    let t = e2.dot(qvec) * inv_det;
    if t <= 0.0 {
        return None;
    }

    Some(TriangleHit { t, u, v })
}

/// A triangle soup: the [`Geometry`] every renderer needs at least once.
///
/// Positions and normals are `f64` because the tracer is; widening happens
/// once here rather than per intersection test. Degenerate triangles are
/// dropped at construction — one that can never be hit would only inflate
/// the tree, and worse, would make a fully degenerate mesh look traceable.
#[derive(Debug, Clone, Default)]
pub struct TriMesh {
    positions: Vec<Point3>,
    normals: Vec<Vec3>,
    tris: Vec<[u32; 3]>,
}

impl TriMesh {
    /// A mesh from positions, optional per-vertex normals, and corner
    /// indices. Normals are used only when there is exactly one per vertex;
    /// a partial array cannot be indexed and is ignored, and hits then
    /// report the geometric face normal.
    pub fn new(positions: Vec<Point3>, normals: Vec<Vec3>, indices: &[u32]) -> Self {
        let vertex_count = positions.len();
        let normals = if normals.len() == vertex_count {
            normals
        } else {
            Vec::new()
        };

        let tris: Vec<[u32; 3]> = (0..indices.len() / 3)
            .map(|i| [indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]])
            .filter(|t| {
                if !t.iter().all(|&i| (i as usize) < vertex_count) {
                    return false;
                }
                if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
                    return false;
                }
                // Zero-area corners too, by the same size-relative test the
                // intersector uses, so build and trace agree on what counts
                // as degenerate.
                let (a, b, c) = (
                    positions[t[0] as usize],
                    positions[t[1] as usize],
                    positions[t[2] as usize],
                );
                let (e1, e2) = (b - a, c - a);
                e1.cross(e2).norm() > 1e-12 * (e1.norm() * e2.norm()).max(1.0)
            })
            .collect();

        Self {
            positions,
            normals,
            tris,
        }
    }

    /// The vertex positions.
    pub fn positions(&self) -> &[Point3] {
        &self.positions
    }

    /// The surviving triangles, as corner indices.
    pub fn triangles(&self) -> &[[u32; 3]] {
        &self.tris
    }

    /// Intersect one triangle, shading normal included.
    fn test(&self, ray: &Ray, tri: u32) -> Option<Hit> {
        let [i0, i1, i2] = self.tris[tri as usize];
        let (v0, v1, v2) = (
            self.positions[i0 as usize],
            self.positions[i1 as usize],
            self.positions[i2 as usize],
        );

        let hit = intersect_triangle(ray, v0, v1, v2)?;

        // Geometric normal. Non-zero by construction: degenerate triangles
        // never made it into `tris`.
        let geometric = (v1 - v0).cross(v2 - v0);

        // Smooth shading, so a mesh part does not read as faceted next to an
        // analytic one. Falls back where the blend cancels.
        let smooth = if self.normals.is_empty() {
            None
        } else {
            let n = self.normals[i0 as usize] * hit.w()
                + self.normals[i1 as usize] * hit.u
                + self.normals[i2 as usize] * hit.v;
            (n.norm() > 1e-12).then_some(n)
        };

        Some(Hit::new(
            hit.t,
            ray.at(hit.t),
            Dir3::new_normalize(smooth.unwrap_or(geometric)),
            Point2::new(hit.u, hit.v),
            tri,
        ))
    }
}

impl Geometry for TriMesh {
    fn len(&self) -> usize {
        self.tris.len()
    }

    fn bounds(&self, i: usize) -> Aabb {
        let mut aabb = Aabb::empty();
        for &vi in &self.tris[i] {
            aabb.include_point(&self.positions[vi as usize]);
        }
        aabb
    }

    fn intersect(&self, ray: &Ray, i: usize, t_min: f64, t_max: f64) -> Option<Hit> {
        self.test(ray, i as u32)
            .filter(|h| h.t > t_min && h.t < t_max)
    }

    fn intersect_all(&self, ray: &Ray, i: usize, out: &mut Vec<Hit>) {
        // A triangle is convex: at most one hit, and nothing to sort.
        if let Some(hit) = self.test(ray, i as u32) {
            out.push(hit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri() -> (Point3, Point3, Point3) {
        (
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        )
    }

    #[test]
    fn hits_an_interior_point() {
        let (v0, v1, v2) = tri();
        let ray = Ray::new(Point3::new(0.25, 0.25, -3.0), Vec3::new(0.0, 0.0, 1.0));
        let hit = intersect_triangle(&ray, v0, v1, v2).expect("should hit");
        assert!((hit.t - 3.0).abs() < 1e-12, "t = {}", hit.t);
        assert!((hit.u - 0.25).abs() < 1e-12);
        assert!((hit.v - 0.25).abs() < 1e-12);
        assert!((hit.w() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn misses_outside_the_triangle() {
        let (v0, v1, v2) = tri();
        let ray = Ray::new(Point3::new(0.9, 0.9, -3.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(intersect_triangle(&ray, v0, v1, v2).is_none());
    }

    #[test]
    fn hits_from_the_back_side_too() {
        let (v0, v1, v2) = tri();
        let ray = Ray::new(Point3::new(0.25, 0.25, 3.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = intersect_triangle(&ray, v0, v1, v2).expect("double-sided");
        assert!((hit.t - 3.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_hits_behind_the_origin() {
        let (v0, v1, v2) = tri();
        let ray = Ray::new(Point3::new(0.25, 0.25, 3.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(intersect_triangle(&ray, v0, v1, v2).is_none());
    }

    #[test]
    fn a_parallel_ray_misses() {
        let (v0, v1, v2) = tri();
        let ray = Ray::new(Point3::new(0.25, 0.25, 1.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(intersect_triangle(&ray, v0, v1, v2).is_none());
    }

    #[test]
    fn a_degenerate_triangle_misses() {
        let v0 = Point3::new(0.0, 0.0, 0.0);
        let ray = Ray::new(Point3::new(0.0, 0.0, -3.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(intersect_triangle(&ray, v0, v0, v0).is_none());
    }

    #[test]
    fn a_micron_triangle_is_not_degenerate() {
        let s = 1e-3;
        let v0 = Point3::new(0.0, 0.0, 0.0);
        let v1 = Point3::new(s, 0.0, 0.0);
        let v2 = Point3::new(0.0, s, 0.0);
        let ray = Ray::new(
            Point3::new(s / 4.0, s / 4.0, -1.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        assert!(intersect_triangle(&ray, v0, v1, v2).is_some());
    }

    #[test]
    fn a_trimesh_drops_its_degenerate_triangles() {
        let mesh = TriMesh::new(
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            Vec::new(),
            &[0, 1, 2, 0, 0, 1, 0, 1, 9],
        );
        assert_eq!(mesh.len(), 1);
    }
}
