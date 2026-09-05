//! The math this crate speaks: tang's types, monomorphised to `f64`.
//!
//! Nothing here is new arithmetic. It is a naming layer, so that a geometry
//! crate handing us a point does not have to convert one: `Point3` *is*
//! `tang::Point3<f64>`, which is what vcad's kernel math and phyz's poses are
//! already made of.
//!
//! [`Transform`] is the one type with a body of its own, and it is a single
//! `tang::Mat4<f64>` field — full affine, because an instance may be scaled or
//! mirrored and a rigid rotation+translation pair cannot say that.

/// A point in 3D space.
pub type Point3 = tang::Point3<f64>;

/// A vector in 3D space.
pub type Vec3 = tang::Vec3<f64>;

/// A unit direction in 3D space.
pub type Dir3 = tang::Dir3<f64>;

/// A point in 2D parameter space.
pub type Point2 = tang::Point2<f64>;

/// A vector in 2D space.
pub type Vec2 = tang::Vec2<f64>;

/// A 4×4 affine placement: object → world.
///
/// Affine rather than rigid on purpose. An instance of a part may be scaled
/// non-uniformly or mirrored, and the tracer has to handle both: the ray is
/// mapped into local space by the inverse, and the normal comes back out by
/// the inverse transpose, which is what makes a mirrored instance's outward
/// normals still point outward.
#[derive(Debug, Clone, PartialEq)]
pub struct Transform {
    /// The underlying 4×4 matrix, column vectors on the right.
    pub matrix: tang::Mat4<f64>,
}

impl Transform {
    /// The identity placement.
    pub fn identity() -> Self {
        Self { matrix: tang::Mat4::identity() }
    }

    /// Translation by `(dx, dy, dz)`.
    pub fn translation(dx: f64, dy: f64, dz: f64) -> Self {
        Self { matrix: tang::Mat4::translation(dx, dy, dz) }
    }

    /// Non-uniform scale by `(sx, sy, sz)`.
    pub fn scale(sx: f64, sy: f64, sz: f64) -> Self {
        Self { matrix: tang::Mat4::scale(sx, sy, sz) }
    }

    /// Rotation about the X axis, radians.
    pub fn rotation_x(angle: f64) -> Self {
        Self { matrix: tang::Mat4::rotation_x(angle) }
    }

    /// Rotation about the Y axis, radians.
    pub fn rotation_y(angle: f64) -> Self {
        Self { matrix: tang::Mat4::rotation_y(angle) }
    }

    /// Rotation about the Z axis, radians.
    pub fn rotation_z(angle: f64) -> Self {
        Self { matrix: tang::Mat4::rotation_z(angle) }
    }

    /// Wrap a raw matrix.
    pub fn from_matrix(matrix: tang::Mat4<f64>) -> Self {
        Self { matrix }
    }

    /// Compose: `self` then `other` (`self * other`).
    pub fn then(&self, other: &Transform) -> Self {
        Self { matrix: self.matrix * other.matrix }
    }

    /// Transform a point.
    #[inline]
    pub fn apply_point(&self, p: &Point3) -> Point3 {
        self.matrix.transform_point(*p)
    }

    /// Transform a direction: rotation and scale, no translation.
    #[inline]
    pub fn apply_vec(&self, v: &Vec3) -> Vec3 {
        self.matrix.transform_vec(*v)
    }

    /// Transform a normal by the inverse transpose of the upper-left 3×3.
    #[inline]
    pub fn apply_normal(&self, n: &Vec3) -> Vec3 {
        self.matrix.transform_normal(*n)
    }

    /// The inverse placement, if the matrix is invertible.
    pub fn inverse(&self) -> Option<Self> {
        self.matrix.try_inverse().map(|matrix| Self { matrix })
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::identity()
    }
}

impl From<tang::Mat4<f64>> for Transform {
    fn from(matrix: tang::Mat4<f64>) -> Self {
        Self { matrix }
    }
}

/// Build a [`Transform`] from a **column-major** 4×4 laid out as 16
/// contiguous `f64`s — the wire format glTF and Three.js produce, where the
/// translation lives at indices 12, 13, 14 rather than 3, 7, 11.
pub fn transform_from_column_major(m: &[f64]) -> Option<Transform> {
    if m.len() < 16 {
        return None;
    }
    // `tang::Mat4::new` takes its arguments row-major, so feeding
    // `m[0], m[4], m[8], m[12]` as the first row is what reads the input as
    // column-major.
    Some(Transform {
        matrix: tang::Mat4::new(
            m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6], m[10], m[14], m[3], m[7],
            m[11], m[15],
        ),
    })
}

/// An axis-aligned box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    /// Minimum corner.
    pub min: Point3,
    /// Maximum corner.
    pub max: Point3,
}

impl Aabb {
    /// A box from its two corners.
    pub fn new(min: Point3, max: Point3) -> Self {
        Self { min, max }
    }

    /// The inverted box: contains nothing, and absorbs the first point it is
    /// given. The right starting value for a fold.
    pub fn empty() -> Self {
        Self {
            min: Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY),
            max: Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
        }
    }

    /// Grow to include a point.
    #[inline]
    pub fn include_point(&mut self, p: &Point3) {
        self.min.x = self.min.x.min(p.x);
        self.min.y = self.min.y.min(p.y);
        self.min.z = self.min.z.min(p.z);
        self.max.x = self.max.x.max(p.x);
        self.max.y = self.max.y.max(p.y);
        self.max.z = self.max.z.max(p.z);
    }

    /// Grow to include another box.
    #[inline]
    pub fn include(&mut self, other: &Aabb) {
        self.include_point(&other.min);
        self.include_point(&other.max);
    }

    /// Centre of the box.
    #[inline]
    pub fn center(&self) -> Point3 {
        Point3::new(
            (self.min.x + self.max.x) / 2.0,
            (self.min.y + self.max.y) / 2.0,
            (self.min.z + self.max.z) / 2.0,
        )
    }

    /// Total surface area — the S in SAH.
    #[inline]
    pub fn surface_area(&self) -> f64 {
        let d = Vec3::new(
            self.max.x - self.min.x,
            self.max.y - self.min.y,
            self.max.z - self.min.z,
        );
        2.0 * (d.x * d.y + d.y * d.z + d.z * d.x)
    }
}

/// World box of a local box under a placement: transform all eight corners
/// and re-bound. Tighter than transforming min/max alone, which is only
/// correct for an axis-aligned scale.
pub fn transform_aabb(local: &Aabb, to_world: &Transform) -> Aabb {
    let mut out = Aabb::empty();
    for i in 0..8 {
        let corner = Point3::new(
            if i & 1 == 0 { local.min.x } else { local.max.x },
            if i & 2 == 0 { local.min.y } else { local.max.y },
            if i & 4 == 0 { local.min.z } else { local.max.z },
        );
        out.include_point(&to_world.apply_point(&corner));
    }
    out
}
