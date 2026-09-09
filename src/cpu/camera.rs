//! The physical camera and the pixel reconstruction filter.

use super::*;

/// A physical camera. Perspective with a real aperture, or orthographic for
/// drafting-style framing.
///
/// The screen basis is stored explicitly rather than derived from an
/// up-hint. Callers that already have a projection basis (a CAD view matrix,
/// say) can hand it over verbatim with [`Camera::from_basis`] and get pixel
/// alignment with their existing renderer — including bases that are
/// mirrored, which a `look_at` construction cannot reproduce.
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    /// Eye position.
    pub eye: Point3,
    /// Unit direction the camera looks along.
    pub forward: Vec3,
    /// Unit world direction mapping to screen +x.
    pub right: Vec3,
    /// Unit world direction mapping to screen +y (up).
    pub up: Vec3,
    /// Vertical field of view in degrees (perspective only).
    pub fov_deg: f64,
    /// Aperture *radius* in world units. Zero gives a pinhole.
    pub aperture: f64,
    /// Distance to the plane of exact focus.
    pub focus_dist: f64,
    /// When set, render orthographically with this half-height instead.
    pub ortho_half_height: Option<f64>,
}

impl Camera {
    /// Conventional right-handed camera aimed at `target`.
    pub fn look_at(eye: Point3, target: Point3, up_hint: Vec3, fov_deg: f64) -> Self {
        let forward = (target - eye).normalize();
        let right = forward.cross(up_hint).normalize();
        let up = right.cross(forward).normalize();
        Self {
            eye,
            forward,
            right,
            up,
            fov_deg,
            aperture: 0.0,
            focus_dist: (target - eye).norm(),
            ortho_half_height: None,
        }
    }

    /// Build from an explicit screen basis. Vectors are normalised but
    /// otherwise used as given, so a mirrored basis stays mirrored.
    pub fn from_basis(
        eye: Point3,
        forward: Vec3,
        right: Vec3,
        up: Vec3,
        fov_deg: f64,
        focus_dist: f64,
    ) -> Self {
        Self {
            eye,
            forward: forward.normalize(),
            right: right.normalize(),
            up: up.normalize(),
            fov_deg,
            aperture: 0.0,
            focus_dist,
            ortho_half_height: None,
        }
    }

    /// Generate a primary ray through normalised screen coords in [-1, 1],
    /// with `(lu, lv)` a uniform sample on the unit disc for lens defocus.
    pub(crate) fn ray(&self, sx: f64, sy: f64, aspect: f64, lu: f64, lv: f64) -> Ray {
        let (fwd, right, up) = (self.forward, self.right, self.up);

        if let Some(hh) = self.ortho_half_height {
            let hw = hh * aspect;
            let origin = self.eye + right * (sx * hw) + up * (sy * hh);
            return Ray::new(origin, fwd);
        }

        let half_h = (self.fov_deg.to_radians() * 0.5).tan();
        let half_w = half_h * aspect;

        // Point on the focal plane this pixel maps to.
        let dir = fwd + right * (sx * half_w) + up * (sy * half_h);
        let focal_point = self.eye + dir * self.focus_dist;

        if self.aperture <= 0.0 {
            return Ray::new(self.eye, focal_point - self.eye);
        }
        let offset = right * (lu * self.aperture) + up * (lv * self.aperture);
        let origin = self.eye + offset;
        Ray::new(origin, focal_point - origin)
    }
}

// ─── the pixel filter ─────────────────────────────────────────────────────

/// Gaussian pixel filter standard deviation, in pixels.
///
/// 0.4 is the usual choice: narrow enough that the image is not visibly soft,
/// wide enough that the filter actually does something at the edges a box
/// filter aliases.
pub const GAUSSIAN_SIGMA: f64 = 0.4;

/// Blackman-Harris coefficients, the standard 4-term minimum-sidelobe set.
pub(crate) const BH: [f64; 4] = [0.35875, -0.48829, 0.14128, -0.01168];

/// The reconstruction filter a pixel's samples are drawn against.
///
/// Primary rays have always been jittered *uniformly* inside the pixel, which
/// is a box filter — the worst reconstruction filter there is, and the reason
/// a thin bright feature against a dark background (a rim, a net cord) crawls
/// and stairsteps however many samples it gets. A better filter would
/// normally mean carrying a per-pixel weight sum, which is a second buffer and
/// a different accumulation rule on both tiers.
///
/// It does not have to. Draw the sample *position* from the filter itself —
/// importance-sample the kernel — and the plain mean of the samples already
/// is the filtered estimate. The accumulation rule, the running mean in the
/// device history, and the variance estimator all stay exactly as they were;
/// the only thing that changes is where in the pixel a ray is aimed.
///
/// [`PixelFilter::Box`] is the default, and it is the old behaviour to the
/// bit: its warp is `u - 0.5`, and the sample position was `u`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PixelFilter {
    /// Uniform inside the pixel. The historical behaviour, and the default.
    #[default]
    Box,
    /// Gaussian of standard deviation [`GAUSSIAN_SIGMA`], truncated at the
    /// filter radius.
    Gaussian,
    /// Blackman-Harris — a narrower main lobe than the Gaussian and far lower
    /// sidelobes, so it rings less on a high-contrast edge.
    BlackmanHarris,
}

impl PixelFilter {
    /// Half-width of the filter's support, in pixels.
    ///
    /// The box filter stops at the pixel edge; the other two reach into their
    /// neighbours, which is what lets them reconstruct an edge at all. 1.5 is
    /// a hair under four standard deviations of the Gaussian, so the truncated
    /// tail is a part in ten thousand.
    pub fn radius(self) -> f64 {
        match self {
            PixelFilter::Box => 0.5,
            PixelFilter::Gaussian | PixelFilter::BlackmanHarris => 1.5,
        }
    }

    /// The filter kernel at `x` pixels from the pixel centre, unnormalised.
    /// Zero outside [`PixelFilter::radius`].
    pub fn weight(self, x: f64) -> f64 {
        let r = self.radius();
        if x.abs() > r {
            return 0.0;
        }
        match self {
            PixelFilter::Box => 1.0,
            PixelFilter::Gaussian => (-(x * x) / (2.0 * GAUSSIAN_SIGMA * GAUSSIAN_SIGMA)).exp(),
            PixelFilter::BlackmanHarris => {
                let t = (x + r) / (2.0 * r);
                let tau = core::f64::consts::TAU;
                BH[0]
                    + BH[1] * (tau * t).cos()
                    + BH[2] * (2.0 * tau * t).cos()
                    + BH[3] * (3.0 * tau * t).cos()
            }
        }
    }

    /// Unnormalised CDF of the kernel from `-radius` to `x`.
    ///
    /// Both non-box filters integrate in closed form — the Gaussian through
    /// `erf`, Blackman-Harris because a sum of cosines integrates to a sum of
    /// sines — so there is no table to build and nothing to keep in sync
    /// between the two tiers.
    pub(crate) fn cdf(self, x: f64) -> f64 {
        let r = self.radius();
        let x = x.clamp(-r, r);
        match self {
            PixelFilter::Box => x + r,
            PixelFilter::Gaussian => {
                let k = 1.0 / (GAUSSIAN_SIGMA * core::f64::consts::SQRT_2);
                erf(x * k) - erf(-r * k)
            }
            PixelFilter::BlackmanHarris => {
                let t = (x + r) / (2.0 * r);
                let tau = core::f64::consts::TAU;
                BH[0] * t
                    + BH[1] * (tau * t).sin() / tau
                    + BH[2] * (2.0 * tau * t).sin() / (2.0 * tau)
                    + BH[3] * (3.0 * tau * t).sin() / (3.0 * tau)
            }
        }
    }

    /// Map a uniform `u` in [0, 1) to a sample offset from the pixel centre,
    /// distributed as the filter.
    ///
    /// Inverted by bisection on [`PixelFilter::cdf`], which is monotone
    /// wherever the kernel is non-negative — as all three of these are. Forty
    /// halvings over a three-pixel span reaches the last bit of an `f32`
    /// several times over, and being a deterministic function of `u` alone is
    /// what makes the CPU and the GPU aim at the same point: the device's
    /// jitter is warped by *this* function on the host before it is uploaded.
    pub fn warp(self, u: f64) -> f64 {
        let r = self.radius();
        if self == PixelFilter::Box {
            // Exact, and bit-identical to the un-filtered jitter it replaces.
            return u - 0.5;
        }
        let total = self.cdf(r);
        if !(total > 0.0) {
            return 0.0;
        }
        let target = u.clamp(0.0, 1.0) * total;
        let (mut lo, mut hi) = (-r, r);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if self.cdf(mid) < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }
}

/// Abramowitz & Stegun 7.1.26, good to 1.5e-7 — three orders finer than the
/// bisection that consumes it can resolve, and it avoids a libm `erf` that
/// wasm would have to bring its own copy of.
pub(crate) fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::cpu::testing::*;
    #[allow(unused_imports)]
    use crate::geometry::TriMesh;

    /// `from_basis` must preserve a mirrored (left-handed) screen basis;
    /// `look_at` cannot express one. vcad's isometric view is exactly such a
    /// basis, so this is the property the render path depends on.
    #[test]
    fn from_basis_preserves_mirrored_basis() {
        let c30 = 30f64.to_radians().cos();
        let s30 = 30f64.to_radians().sin();
        let cam = Vec3::new(1.0, 1.0, 1.0).normalize();
        let right = Vec3::new(c30, -c30, 0.0);
        let up = -Vec3::new(s30, s30, -1.0);
        let camera = Camera::from_basis(
            Point3::new(0.0, 0.0, 0.0) + cam * 100.0,
            -cam,
            right,
            up,
            34.0,
            100.0,
        );
        assert!(
            (camera.right - right.normalize()).norm() < 1e-12,
            "right vector was silently re-derived"
        );
        // A right-handed reconstruction would have flipped it.
        let rhs = camera.forward.cross(camera.up).normalize();
        assert!(
            (rhs - camera.right).norm() > 1.0,
            "expected this basis to be mirrored"
        );
    }
}
