//! A regular grid of heights: water, terrain, anything single-valued in z.
//!
//! The primitive is a **cell**, and a cell is a bilinear patch — not two
//! triangles. The patch is worth the algebra: the ray–patch equation is a
//! plain quadratic in `t` (both the surface and the ray are linear in `x`
//! and `y`, so their difference is degree two), which lands the hit point on
//! the interpolated surface to machine precision instead of to the chord
//! error of a triangulation, and it makes the surface C0 across cell edges
//! with an analytic gradient inside them. Two triangles would also have
//! disagreed with the physics: [`kosm-spike`]'s pool marches
//! `p.z - surface.height(p.x, p.y)` against a *bilinear* sample of the same
//! grid, so a triangulated renderer and the simulator would have been
//! looking at two different water surfaces.
//!
//! # The frame
//!
//! Heights are `z` over an `x`/`y` lattice: sample `(i, j)` sits at
//! `origin + (i * cell.0, j * cell.1)` and carries `heights[j * nx + i]`.
//! That is the row-major layout `HeightGrid` already uses, so a frame of
//! simulated water is a slice copy.
//!
//! # A grid that changes every frame
//!
//! Water moves. Rebuilding a hierarchy over 130k cells every frame is not
//! affordable, and it is not necessary: the *topology* of the tree — which
//! cell sits in which leaf — depends only on the lattice, which never
//! changes. Only the boxes move, and only in `z`. So the loop is
//! [`HeightField::update_heights`] followed by [`Bvh::refit`], which walks
//! the existing tree bottom-up recomputing bounds. It is O(n) with a tiny
//! constant and it never allocates. See the crate benchmark in this file's
//! tests for the measured ratio.
//!
//! A refit tree is a slightly worse tree than a rebuilt one — the SAH split
//! planes were chosen for last frame's boxes. For a height field that
//! barely matters, because the split planes are essentially `x`/`y` planes
//! and the lattice they partition is fixed.
//!
//! # The fast path
//!
//! A height field has structure a BVH throws away: the cells tile the plane
//! regularly, so the cells a ray crosses can be *enumerated* rather than
//! searched for. [`HeightField::intersect_march`] does a 2D DDA over the
//! lattice, testing patches in increasing `t` and stopping at the first hit.
//! For a camera or shadow ray coming down at the water it beats the tree,
//! and it needs no tree at all — which is one less thing to refit.
//!
//! # What the integrator needs
//!
//! Nothing new. A [`HeightField`] is an ordinary opaque [`Geometry`]: it
//! reports a point, a normal and a `(u, v)`, and the existing path tracer
//! shades it with whatever material the object carries. Water wants a
//! transmissive material, but that is the material system's problem, not
//! this file's. The only integration note is the per-frame one above: a
//! client that animates the grid must call `update_heights` then `refit`,
//! and must not hold a hit from the previous frame across that call.
//!
//! [`kosm-spike`]: https://docs.rs/kosm-spike
//! [`Bvh::refit`]: crate::Bvh::refit

use crate::geometry::Geometry;
use crate::math::{Aabb, Dir3, Point2, Point3};
use crate::ray::{Hit, Ray};

/// How far outside `[0, 1]` a patch coordinate may land and still count.
///
/// A ray crossing a cell edge is, in exact arithmetic, on both patches. The
/// slack lets either one claim it rather than letting rounding drop the hit
/// into the crack between them.
const UV_EPS: f64 = 1e-9;

/// A regular lattice of heights over the `xy` plane.
///
/// `nx` by `ny` *samples* — so `(nx - 1) * (ny - 1)` cells, which are the
/// primitives. A grid narrower than two samples in either direction has no
/// cells and is empty rather than an error: a degenerate frame of water
/// should render as nothing, not panic.
#[derive(Debug, Clone, Default)]
pub struct HeightField {
    nx: usize,
    ny: usize,
    origin: Point3,
    dx: f64,
    dy: f64,
    heights: Vec<f64>,
}

impl HeightField {
    /// A field from a lattice and its samples.
    ///
    /// `origin` is the world position of sample `(0, 0)`; its `z` is added
    /// to every height, so a grid of zeroes is a flat plane at `origin.z`.
    /// `heights` is row-major, `nx` samples per row, `ny` rows.
    ///
    /// # Panics
    ///
    /// If `heights.len() != nx * ny`, or if either cell size is not
    /// positive. Both are programmer errors at the seam — a mismatched
    /// slice would silently render a garbled surface.
    pub fn new(
        nx: usize,
        ny: usize,
        origin: Point3,
        cell: (f64, f64),
        heights: Vec<f64>,
    ) -> Self {
        assert_eq!(heights.len(), nx * ny, "heights must be nx * ny, row-major");
        assert!(
            cell.0 > 0.0 && cell.1 > 0.0,
            "cell size must be positive in both directions"
        );
        Self {
            nx,
            ny,
            origin,
            dx: cell.0,
            dy: cell.1,
            heights,
        }
    }

    /// Replace every height, keeping the lattice.
    ///
    /// This is the per-frame call. It touches no index and no bound, so a
    /// [`Bvh`] built over this field stays structurally valid and only needs
    /// a [`refit`]; rebuilding after it is correct but wasteful.
    ///
    /// # Panics
    ///
    /// If the slice is not the same length as the existing one. Changing the
    /// lattice is a new field, not an update.
    ///
    /// [`Bvh`]: crate::Bvh
    /// [`refit`]: crate::Bvh::refit
    pub fn update_heights(&mut self, heights: &[f64]) {
        assert_eq!(
            heights.len(),
            self.heights.len(),
            "update_heights keeps the lattice; build a new field to change it"
        );
        self.heights.copy_from_slice(heights);
    }

    /// Samples across.
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Samples down.
    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Cells across.
    pub fn cells_x(&self) -> usize {
        self.nx.saturating_sub(1)
    }

    /// Cells down.
    pub fn cells_y(&self) -> usize {
        self.ny.saturating_sub(1)
    }

    /// The raw samples, row-major.
    pub fn heights(&self) -> &[f64] {
        &self.heights
    }

    /// The height at sample `(i, j)`, in world `z`.
    #[inline]
    pub fn sample(&self, i: usize, j: usize) -> f64 {
        self.origin.z + self.heights[j * self.nx + i]
    }

    /// The interpolated surface at a world `(x, y)`, clamped at the border.
    ///
    /// This is the function the patch intersector solves against, so a hit
    /// point's `z` and `height_at` of its `x`, `y` agree to rounding. Points
    /// outside the lattice clamp to the edge rather than extrapolating,
    /// which keeps the far field finite.
    pub fn height_at(&self, x: f64, y: f64) -> f64 {
        if self.cells_x() == 0 || self.cells_y() == 0 {
            return self.origin.z;
        }
        let fx = ((x - self.origin.x) / self.dx).clamp(0.0, self.cells_x() as f64);
        let fy = ((y - self.origin.y) / self.dy).clamp(0.0, self.cells_y() as f64);
        let i = (fx.floor() as usize).min(self.cells_x() - 1);
        let j = (fy.floor() as usize).min(self.cells_y() - 1);
        let (u, v) = (fx - i as f64, fy - j as f64);
        let c = self.corners(i, j);
        c.0 + (c.1 - c.0) * u + (c.2 - c.0) * v + (c.0 - c.1 - c.2 + c.3) * u * v
    }

    /// The analytic surface gradient `(dz/dx, dz/dy)` at a world `(x, y)`.
    ///
    /// Discontinuous across cell edges — the patch is C0, not C1 — which is
    /// what a facetted-but-interpolated surface is. Outside the lattice it
    /// is zero, matching the clamped height.
    pub fn gradient_at(&self, x: f64, y: f64) -> (f64, f64) {
        if self.cells_x() == 0 || self.cells_y() == 0 {
            return (0.0, 0.0);
        }
        let fx = (x - self.origin.x) / self.dx;
        let fy = (y - self.origin.y) / self.dy;
        if fx < 0.0 || fy < 0.0 || fx > self.cells_x() as f64 || fy > self.cells_y() as f64 {
            return (0.0, 0.0);
        }
        let i = (fx.floor() as usize).min(self.cells_x() - 1);
        let j = (fy.floor() as usize).min(self.cells_y() - 1);
        let (u, v) = (fx - i as f64, fy - j as f64);
        self.patch_gradient(i, j, u, v)
    }

    /// The four corner heights of cell `(i, j)`, as `(z00, z10, z01, z11)`.
    #[inline]
    fn corners(&self, i: usize, j: usize) -> (f64, f64, f64, f64) {
        (
            self.sample(i, j),
            self.sample(i + 1, j),
            self.sample(i, j + 1),
            self.sample(i + 1, j + 1),
        )
    }

    /// `(dz/dx, dz/dy)` inside cell `(i, j)` at local `(u, v)`.
    #[inline]
    fn patch_gradient(&self, i: usize, j: usize, u: f64, v: f64) -> (f64, f64) {
        let (z00, z10, z01, z11) = self.corners(i, j);
        let b = z10 - z00;
        let c = z01 - z00;
        let e = z00 - z10 - z01 + z11;
        ((b + e * v) / self.dx, (c + e * u) / self.dy)
    }

    /// Cell index from lattice coordinates.
    #[inline]
    fn cell_index(&self, i: usize, j: usize) -> usize {
        j * self.cells_x() + i
    }

    /// Lattice coordinates from a cell index.
    #[inline]
    fn cell_coords(&self, cell: usize) -> (usize, usize) {
        (cell % self.cells_x(), cell / self.cells_x())
    }

    /// Ray against one cell's bilinear patch.
    ///
    /// Both `u` and `v` run linearly in `t`, so the residual
    /// `z(t) - h(u(t), v(t))` is a quadratic whose only nonlinear term is
    /// the patch's twist `e * u * v`. A flat or ruled cell degenerates to a
    /// linear equation, which is handled rather than divided by.
    fn intersect_cell(&self, ray: &Ray, i: usize, j: usize, t_min: f64, t_max: f64) -> Option<Hit> {
        let (z00, z10, z01, z11) = self.corners(i, j);
        let b = z10 - z00;
        let c = z01 - z00;
        let e = z00 - z10 - z01 + z11;

        let x0 = self.origin.x + i as f64 * self.dx;
        let y0 = self.origin.y + j as f64 * self.dy;
        let d = ray.direction.as_ref();

        let u0 = (ray.origin.x - x0) / self.dx;
        let v0 = (ray.origin.y - y0) / self.dy;
        let ut = d.x / self.dx;
        let vt = d.y / self.dy;

        let a2 = -e * ut * vt;
        let a1 = d.z - b * ut - c * vt - e * (u0 * vt + ut * v0);
        let a0 = ray.origin.z - z00 - b * u0 - c * v0 - e * u0 * v0;

        let mut roots = [f64::NAN; 2];
        let count = solve_quadratic(a2, a1, a0, &mut roots);

        for k in 0..count {
            let t = roots[k];
            if !(t > t_min && t < t_max) {
                continue;
            }
            let u = u0 + ut * t;
            let v = v0 + vt * t;
            if !(-UV_EPS..=1.0 + UV_EPS).contains(&u) || !(-UV_EPS..=1.0 + UV_EPS).contains(&v) {
                continue;
            }
            let (u, v) = (u.clamp(0.0, 1.0), v.clamp(0.0, 1.0));
            let (gx, gy) = self.patch_gradient(i, j, u, v);
            let cell = self.cell_index(i, j);
            return Some(
                Hit::new(
                    t,
                    ray.at(t),
                    // Up-facing by construction: the field is single-valued,
                    // so the outward side is +z and the tracer face-forwards
                    // anything it needs to.
                    Dir3::new_normalize(crate::math::Vec3::new(-gx, -gy, 1.0)),
                    // `uv` is the position on the lattice, in cells: a
                    // caller can read a texture with it, or floor it to get
                    // the sample it sits between.
                    Point2::new(i as f64 + u, j as f64 + v),
                    cell as u32,
                )
                .with_payload(cell as u64),
            );
        }
        None
    }

    /// The nearest hit anywhere on the field, by marching the lattice.
    ///
    /// A 2D DDA in `x`/`y`: clip the ray to the field's footprint, then walk
    /// cell to cell in increasing `t`, testing each patch and returning the
    /// first hit. Because cells are visited in `t` order the first hit is
    /// the nearest one, so this is exact for any ray — it is simply *fastest*
    /// for the near-vertical rays a camera above the water sends down, where
    /// it touches a handful of cells and no tree.
    ///
    /// A ray with no `x`/`y` motion at all skips the march and tests the one
    /// cell it stands over.
    pub fn intersect_march(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        if self.is_empty() {
            return None;
        }
        let (cx, cy) = (self.cells_x(), self.cells_y());
        let d = ray.direction.as_ref();

        // Clip to the footprint in x and y. `t_enter`/`t_exit` bracket the
        // span of the ray that is over the lattice at all.
        let (mut t_enter, mut t_exit) = (t_min.max(0.0), t_max);
        for axis in 0..2 {
            let (o, dir, lo, hi) = if axis == 0 {
                (
                    ray.origin.x,
                    d.x,
                    self.origin.x,
                    self.origin.x + cx as f64 * self.dx,
                )
            } else {
                (
                    ray.origin.y,
                    d.y,
                    self.origin.y,
                    self.origin.y + cy as f64 * self.dy,
                )
            };
            if dir.abs() < f64::MIN_POSITIVE {
                if o < lo || o > hi {
                    return None;
                }
                continue;
            }
            let (mut a, mut b) = ((lo - o) / dir, (hi - o) / dir);
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            t_enter = t_enter.max(a);
            t_exit = t_exit.min(b);
        }
        if t_enter > t_exit {
            return None;
        }

        // The cell standing under the entry point.
        let entry = ray.at(t_enter);
        let mut i = (((entry.x - self.origin.x) / self.dx).floor() as isize)
            .clamp(0, cx as isize - 1);
        let mut j = (((entry.y - self.origin.y) / self.dy).floor() as isize)
            .clamp(0, cy as isize - 1);

        // Per-axis stepping: which way, how far to the next boundary, and
        // how far between boundaries. A zero component never steps.
        let setup = |o: f64, dir: f64, base: f64, cell: f64, idx: isize| {
            if dir.abs() < f64::MIN_POSITIVE {
                return (0isize, f64::INFINITY, f64::INFINITY);
            }
            let step = if dir > 0.0 { 1isize } else { -1isize };
            let next = base + (idx + if dir > 0.0 { 1 } else { 0 }) as f64 * cell;
            (step, (next - o) / dir, (cell / dir).abs())
        };
        let (step_i, mut next_i, delta_i) =
            setup(ray.origin.x, d.x, self.origin.x, self.dx, i);
        let (step_j, mut next_j, delta_j) =
            setup(ray.origin.y, d.y, self.origin.y, self.dy, j);

        loop {
            if let Some(hit) =
                self.intersect_cell(ray, i as usize, j as usize, t_min, t_max.min(t_exit))
            {
                return Some(hit);
            }
            // Leave through whichever boundary comes first.
            if next_i.min(next_j) > t_exit {
                return None;
            }
            if next_i < next_j {
                i += step_i;
                next_i += delta_i;
            } else {
                j += step_j;
                next_j += delta_j;
            }
            if i < 0 || j < 0 || i >= cx as isize || j >= cy as isize {
                return None;
            }
        }
    }
}

/// Real roots of `a t² + b t + c`, ascending, written into `out`.
///
/// Returns how many there are. Degenerates gracefully: a vanishing leading
/// coefficient is solved as a line, which is the common case here (a flat or
/// ruled cell has no twist term). The stable form avoids the catastrophic
/// cancellation of the schoolbook one when `b² >> 4ac`, which is exactly the
/// grazing-ray case a height field sees constantly.
fn solve_quadratic(a: f64, b: f64, c: f64, out: &mut [f64; 2]) -> usize {
    // Scale-relative: "a is zero" has to mean small next to the other terms,
    // or a millimetre-scale grid looks degenerate and a kilometre one never
    // does.
    let scale = b.abs().max(c.abs()).max(1e-300);
    if a.abs() <= 1e-14 * scale {
        if b.abs() <= 1e-300 {
            return 0;
        }
        out[0] = -c / b;
        return 1;
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return 0;
    }
    let sq = disc.sqrt();
    let q = -0.5 * (b + b.signum() * sq);
    let (mut r0, mut r1) = if q.abs() > 1e-300 {
        (q / a, c / q)
    } else {
        (0.0, 0.0)
    };
    if r0 > r1 {
        std::mem::swap(&mut r0, &mut r1);
    }
    out[0] = r0;
    out[1] = r1;
    2
}

impl Geometry for HeightField {
    fn len(&self) -> usize {
        self.cells_x() * self.cells_y()
    }

    fn bounds(&self, i: usize) -> Aabb {
        let (ci, cj) = self.cell_coords(i);
        let (z00, z10, z01, z11) = self.corners(ci, cj);
        // The bilinear patch is contained in the box of its corners: every
        // interior value is a convex combination of the four.
        let zmin = z00.min(z10).min(z01).min(z11);
        let zmax = z00.max(z10).max(z01).max(z11);
        let x0 = self.origin.x + ci as f64 * self.dx;
        let y0 = self.origin.y + cj as f64 * self.dy;
        Aabb::new(
            Point3::new(x0, y0, zmin),
            Point3::new(x0 + self.dx, y0 + self.dy, zmax),
        )
    }

    fn intersect(&self, ray: &Ray, i: usize, t_min: f64, t_max: f64) -> Option<Hit> {
        let (ci, cj) = self.cell_coords(i);
        self.intersect_cell(ray, ci, cj, t_min, t_max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bvh::Bvh;
    use crate::math::Vec3;

    fn flat(nx: usize, ny: usize, h: f64) -> HeightField {
        HeightField::new(
            nx,
            ny,
            Point3::new(0.0, 0.0, 0.0),
            (1.0, 1.0),
            vec![h; nx * ny],
        )
    }

    fn sinusoid(nx: usize, ny: usize) -> HeightField {
        let (dx, dy) = (4.0 / (nx - 1) as f64, 4.0 / (ny - 1) as f64);
        let mut h = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                let (x, y) = (i as f64 * dx, j as f64 * dy);
                h.push(0.3 * (1.7 * x).sin() * (1.1 * y).cos());
            }
        }
        HeightField::new(nx, ny, Point3::new(0.0, 0.0, 0.0), (dx, dy), h)
    }

    #[test]
    fn a_flat_field_is_hit_at_its_height() {
        let field = flat(8, 8, 2.5);
        let bvh = Bvh::build(field);
        let ray = Ray::new(Point3::new(3.3, 4.7, 10.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = bvh.trace_closest(&ray).expect("straight down at the slab");
        assert!((hit.t - 7.5).abs() < 1e-12, "t = {}", hit.t);
        assert!((hit.point.z - 2.5).abs() < 1e-12);
        assert!(hit.normal.z > 0.0);
        assert!(hit.normal.x.abs() < 1e-12 && hit.normal.y.abs() < 1e-12);
    }

    #[test]
    fn a_slanted_ray_still_lands_on_a_flat_field() {
        let field = flat(8, 8, 1.0);
        let bvh = Bvh::build(field);
        let ray = Ray::new(Point3::new(0.5, 0.5, 5.0), Vec3::new(0.4, 0.2, -1.0));
        let hit = bvh.trace_closest(&ray).expect("hits the slab");
        assert!((hit.point.z - 1.0).abs() < 1e-12);
    }

    #[test]
    fn hit_points_lie_on_the_interpolated_surface() {
        let field = sinusoid(65, 65);
        let bvh = Bvh::build(field);
        let mut tested = 0;
        for a in 0..12 {
            for b in 0..12 {
                let (x, y) = (0.31 + a as f64 * 0.29, 0.17 + b as f64 * 0.31);
                let ray = Ray::new(
                    Point3::new(x, y, 4.0),
                    Vec3::new(0.05 * (a as f64 - 6.0), 0.04 * (b as f64 - 6.0), -3.0),
                );
                let Some(hit) = bvh.trace_closest(&ray) else {
                    continue;
                };
                let h = bvh.geometry().height_at(hit.point.x, hit.point.y);
                assert!(
                    (hit.point.z - h).abs() < 1e-9,
                    "off the surface by {}",
                    (hit.point.z - h).abs()
                );
                tested += 1;
            }
        }
        assert!(tested > 100, "only {tested} rays landed");
    }

    #[test]
    fn normals_match_the_analytic_gradient() {
        // Fine enough that the patch gradient tracks the sinusoid's own.
        let field = sinusoid(257, 257);
        let bvh = Bvh::build(field);
        let mut worst: f64 = 0.0;
        for a in 0..10 {
            for b in 0..10 {
                let (x, y) = (0.4 + a as f64 * 0.3, 0.4 + b as f64 * 0.3);
                let ray = Ray::new(Point3::new(x, y, 4.0), Vec3::new(0.0, 0.0, -1.0));
                let hit = bvh.trace_closest(&ray).expect("straight down");
                let (px, py) = (hit.point.x, hit.point.y);
                let (gx, gy) = (
                    0.3 * 1.7 * (1.7 * px).cos() * (1.1 * py).cos(),
                    -0.3 * 1.1 * (1.7 * px).sin() * (1.1 * py).sin(),
                );
                let want = Dir3::new_normalize(Vec3::new(-gx, -gy, 1.0));
                let cos = hit.normal.as_ref().dot(want.as_ref()).clamp(-1.0, 1.0);
                worst = worst.max(cos.acos().to_degrees());
            }
        }
        assert!(worst < 1.0, "worst normal error {worst}°");
    }

    #[test]
    fn the_march_agrees_with_the_tree() {
        let field = sinusoid(33, 33);
        let bvh = Bvh::build(field.clone());
        for a in 0..15 {
            for b in 0..15 {
                let ray = Ray::new(
                    Point3::new(0.2 + a as f64 * 0.25, 0.2 + b as f64 * 0.25, 3.0),
                    Vec3::new(0.2 * (a as f64 - 7.0), 0.2 * (b as f64 - 7.0), -4.0),
                );
                let tree = bvh.trace_closest(&ray);
                let march = field.intersect_march(&ray, 0.0, f64::INFINITY);
                match (tree, march) {
                    (Some(t), Some(m)) => assert!(
                        (t.t - m.t).abs() < 1e-9,
                        "tree {} vs march {}",
                        t.t,
                        m.t
                    ),
                    (None, None) => {}
                    (t, m) => panic!("disagree: {:?} vs {:?}", t.map(|h| h.t), m.map(|h| h.t)),
                }
            }
        }
    }

    #[test]
    fn the_march_misses_beside_the_field() {
        let field = flat(8, 8, 0.0);
        let ray = Ray::new(Point3::new(-5.0, -5.0, 1.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(field.intersect_march(&ray, 0.0, f64::INFINITY).is_none());
    }

    #[test]
    fn payload_and_uv_name_the_cell() {
        let field = flat(8, 8, 0.0);
        let ray = Ray::new(Point3::new(3.25, 5.75, 1.0), Vec3::new(0.0, 0.0, -1.0));
        let hit = field
            .intersect_march(&ray, 0.0, f64::INFINITY)
            .expect("over cell (3, 5)");
        assert_eq!(hit.payload, hit.prim as u64);
        assert_eq!(hit.prim, (5 * 7 + 3) as u32);
        assert!((hit.uv.x - 3.25).abs() < 1e-12);
        assert!((hit.uv.y - 5.75).abs() < 1e-12);
    }

    #[test]
    fn refit_after_an_update_matches_a_rebuild() {
        let mut field = sinusoid(33, 33);
        let mut bvh = Bvh::build(field.clone());

        // A different frame of "water".
        let (nx, ny) = (field.nx(), field.ny());
        let (dx, dy) = (4.0 / (nx - 1) as f64, 4.0 / (ny - 1) as f64);
        let mut next = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                let (x, y) = (i as f64 * dx, j as f64 * dy);
                next.push(0.45 * (2.3 * x + 1.0).sin() * (0.9 * y - 0.4).cos());
            }
        }

        field.update_heights(&next);
        bvh.geometry_mut().update_heights(&next);
        bvh.refit();
        let rebuilt = Bvh::build(field);

        for a in 0..15 {
            for b in 0..15 {
                let ray = Ray::new(
                    Point3::new(0.2 + a as f64 * 0.25, 0.2 + b as f64 * 0.25, 3.0),
                    Vec3::new(0.15 * (a as f64 - 7.0), 0.15 * (b as f64 - 7.0), -4.0),
                );
                match (bvh.trace_closest(&ray), rebuilt.trace_closest(&ray)) {
                    (Some(r), Some(w)) => assert!((r.t - w.t).abs() < 1e-12),
                    (None, None) => {}
                    (r, w) => {
                        panic!("refit {:?} vs rebuild {:?}", r.map(|h| h.t), w.map(|h| h.t))
                    }
                }
            }
        }
    }

    #[test]
    fn refit_is_much_cheaper_than_a_rebuild() {
        use std::time::Instant;

        let n = 256;
        let field = sinusoid(n, n);
        let mut bvh = Bvh::build(field.clone());

        let mut next: Vec<f64> = field.heights().to_vec();
        for (k, h) in next.iter_mut().enumerate() {
            *h += 0.01 * (k as f64 * 0.37).sin();
        }

        // Warm up both paths so neither pays for a cold allocator.
        bvh.geometry_mut().update_heights(&next);
        bvh.refit();
        let _ = Bvh::build(field.clone());

        let reps = 5;
        let t0 = Instant::now();
        for _ in 0..reps {
            bvh.geometry_mut().update_heights(&next);
            bvh.refit();
        }
        let refit = t0.elapsed().as_secs_f64() / reps as f64;

        let t1 = Instant::now();
        for _ in 0..reps {
            let _ = Bvh::build(field.clone());
        }
        let rebuild = t1.elapsed().as_secs_f64() / reps as f64;

        println!(
            "256x256 ({} cells): refit {:.3} ms, rebuild {:.3} ms, {:.1}x",
            bvh.geometry().len(),
            refit * 1e3,
            rebuild * 1e3,
            rebuild / refit
        );
        assert!(
            refit < rebuild,
            "refit {refit:.6}s should beat rebuild {rebuild:.6}s"
        );
    }

    #[test]
    fn a_lattice_too_small_to_have_cells_is_empty() {
        let field = HeightField::new(1, 5, Point3::new(0.0, 0.0, 0.0), (1.0, 1.0), vec![0.0; 5]);
        assert!(field.is_empty());
        let ray = Ray::new(Point3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(field.intersect_march(&ray, 0.0, f64::INFINITY).is_none());
    }
}
