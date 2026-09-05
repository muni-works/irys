//! The laws at an interface, generic over the scalar.
//!
//! Snell, Fresnel and Sellmeier are physics, not shading: they belong to the
//! light and not to whatever the light is passing through. They live here so
//! the marble's glass, the pool's water surface and (later) this crate's own
//! integrator all read the same three functions rather than three copies that
//! drift.
//!
//! Everything is generic over [`tang::Scalar`]. On `f64` these are the
//! ordinary formulae; on `Dual<f64>` the same code is its own derivative,
//! which is what makes `∂caustic/∂n_d` fall out of a forward pass. That
//! genericity is the reason these are not written against this crate's `f64`
//! [`Vec3`](crate::Vec3) alias.

use tang::{Scalar, Vec3};

/// Unpolarized Fresnel reflectance at an interface between indices `n1` and
/// `n2`, given the cosines on either side.
///
/// The average of the s- and p-polarised reflectances, which is what an
/// unpolarised source and an unpolarised sensor see.
pub fn fresnel<S: Scalar>(n1: S, n2: S, cos_i: S, cos_t: S) -> S {
    let rs = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t);
    let rp = (n1 * cos_t - n2 * cos_i) / (n1 * cos_t + n2 * cos_i);
    S::HALF * (rs * rs + rp * rp)
}

/// Snell refraction of a unit direction `d` at a unit normal `n` facing
/// against `d`, going from index `n1` into `n2`.
///
/// Returns the refracted direction and the two cosines (which
/// [`fresnel`] wants next), or `None` on total internal reflection.
pub fn refract<S: Scalar>(d: Vec3<S>, n: Vec3<S>, n1: S, n2: S) -> Option<(Vec3<S>, S, S)> {
    let eta = n1 / n2;
    let cos_i = -d.dot(&n);
    let k = S::ONE - eta * eta * (S::ONE - cos_i * cos_i);
    if k < S::ZERO {
        return None;
    }
    let cos_t = k.sqrt();
    Some((d * eta + n * (eta * cos_i - cos_t), cos_i, cos_t))
}

/// Mirror `d` about a unit normal `n`.
pub fn reflect<S: Scalar>(d: Vec3<S>, n: Vec3<S>) -> Vec3<S> {
    d - n * (S::TWO * d.dot(&n))
}

/// N-BK7 Sellmeier coefficients (Schott datasheet): the dispersion *shape*.
const BK7_B: [f64; 3] = [1.039_612_12, 0.231_792_344, 1.010_469_45];
const BK7_C: [f64; 3] = [0.006_000_698_67, 0.020_017_914_4, 103.560_653];

/// The helium d line, 587.6 nm, in microns: where "the index of refraction"
/// of a glass is quoted.
pub const D_LINE_UM: f64 = 0.5876;

/// N-BK7's Sellmeier index at `lambda_um`.
///
/// The wavelength is an `f64` because it is a property of the band being
/// traced and never a differentiable knob; the *result* is generic, so it
/// composes with a dual-valued index.
pub fn sellmeier<S: Scalar>(lambda_um: f64) -> S {
    let l2 = lambda_um * lambda_um;
    let mut n2 = 1.0;
    for k in 0..3 {
        n2 += BK7_B[k] * l2 / (l2 - BK7_C[k]);
    }
    S::from_f64(n2.sqrt())
}

/// A glass's index at `lambda_um`: N-BK7's dispersion shape, shifted so the
/// d-line index is `nd`.
///
/// `nd` is the knob. Seed it with a `Dual` and every wavelength's index
/// carries the derivative, so a spectral caustic differentiates with respect
/// to the material and not with respect to a curve fit.
pub fn index<S: Scalar>(nd: S, lambda_um: f64) -> S {
    nd + sellmeier::<S>(lambda_um) - sellmeier::<S>(D_LINE_UM)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_incidence_matches_the_schlick_form() {
        let (n1, n2) = (1.0, 1.5);
        let r = fresnel(n1, n2, 1.0, 1.0);
        let r0 = ((n1 - n2) / (n1 + n2)).powi(2);
        assert!((r - r0).abs() < 1e-12, "{r} vs {r0}");
    }

    #[test]
    fn refraction_bends_toward_the_normal_entering_glass() {
        let d = Vec3::new(1.0f64, 0.0, -1.0).normalize();
        let n = Vec3::new(0.0, 0.0, 1.0);
        let (t, cos_i, cos_t) = refract(d, n, 1.0, 1.5).expect("no TIR entering a denser medium");
        assert!(cos_t > cos_i, "the ray bends toward the normal");
        assert!((t.norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_grazing_ray_leaving_glass_totally_reflects() {
        let d = Vec3::new(1.0f64, 0.0, -0.05).normalize();
        let n = Vec3::new(0.0, 0.0, 1.0);
        assert!(refract(d, n, 1.5, 1.0).is_none());
    }

    #[test]
    fn reflection_flips_only_the_normal_component() {
        let d = Vec3::new(1.0f64, 2.0, -3.0);
        let n = Vec3::new(0.0, 0.0, 1.0);
        let r = reflect(d, n);
        assert!((r.x - d.x).abs() < 1e-12 && (r.y - d.y).abs() < 1e-12);
        assert!((r.z + d.z).abs() < 1e-12);
    }

    #[test]
    fn bk7_disperses_blue_more_than_red() {
        let blue: f64 = sellmeier(0.45);
        let red: f64 = sellmeier(0.65);
        assert!(blue > red, "{blue} vs {red}");
        // the datasheet's n_d, to five places
        let nd: f64 = sellmeier(D_LINE_UM);
        assert!((nd - 1.516_80).abs() < 1e-4, "n_d = {nd}");
    }

    #[test]
    fn the_index_knob_is_the_d_line() {
        let n: f64 = index(1.9, D_LINE_UM);
        assert!((n - 1.9).abs() < 1e-12);
    }
}
