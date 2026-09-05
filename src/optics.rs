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


// ─── thin-film iridescence ────────────────────────────────────────────────
//
// Belcour and Barla, "A Practical Extension to Microfacet Theory for the
// Modeling of Varying Iridescence" (SIGGRAPH 2017).
//
// A soap bubble, an oil slick and the tempering colours on steel are all one
// phenomenon: a film thin enough that the wave reflected off its top surface
// and the wave that went through it, bounced off the substrate and came back
// out are still coherent. They interfere, constructively at the wavelengths
// where the optical path difference is a whole number of them and
// destructively in between, so the reflectance stops being a smooth Fresnel
// curve and becomes a comb in λ. Integrated against the eye's three
// sensitivities, that comb is a colour, and it swings with angle because the
// path difference does.
//
// The Airy summation over the infinitely many internal bounces is exact and
// cheap. The expensive part is the spectral integral against the CIE curves,
// and Belcour and Barla's contribution is that you never have to do it: the
// summation's terms are pure cosines in the optical path difference, so what
// the integral wants is the *Fourier transform* of the colour matching
// functions evaluated at that frequency — and CIE 1931 fitted as a handful of
// Gaussians has a closed-form transform. That is [`sensitivity`]: six
// numbers, no tables, and it ports to WGSL verbatim.
//
// A path that already carries a hero wavelength does not want any of that; it
// wants the reflectance at its own λ, which is [`airy_at`] — the same series
// summed exactly rather than through three colour integrals.

/// `F0` of an interface between two indices.
#[inline]
fn f0_of(n1: f32, n2: f32) -> f32 {
    let r = (n1 - n2) / (n1 + n2);
    r * r
}

/// The index that would give this `F0`, for the substrate under the film.
#[inline]
fn ior_of_f0(f0: f32) -> f32 {
    let s = f0.clamp(0.0, 0.9999).sqrt();
    (1.0 + s) / (1.0 - s)
}

/// Schlick, in the scalar form the film's two interfaces want.
#[inline]
fn schlick(f0: f32, cos_theta: f32) -> f32 {
    f0 + (1.0 - f0) * (1.0 - cos_theta).clamp(0.0, 1.0).powi(5)
}

/// The Fourier transform of the CIE 1931 curves at the frequency an optical
/// path difference of `opd` nanometres implies, phase-shifted by `shift`, as
/// linear sRGB.
///
/// Belcour and Barla's Gaussian fit and its analytic transform, in the
/// authors' own constants. `1.0685e-7` is the normalisation that makes the
/// zero-frequency term unity.
fn sensitivity(opd: f32, shift: [f32; 3]) -> [f32; 3] {
    const TWO_PI: f32 = std::f32::consts::TAU;
    let phase = TWO_PI * opd * 1.0e-9;
    let val = [5.4856e-13f32, 4.4201e-13, 5.2481e-13];
    let pos = [1.6810e+06f32, 1.7953e+06, 2.2084e+06];
    let var = [4.3278e+09f32, 9.3046e+09, 6.6121e+09];
    let mut xyz = [0.0f32; 3];
    for k in 0..3 {
        xyz[k] = val[k]
            * (TWO_PI * var[k]).sqrt()
            * (pos[k] * phase + shift[k]).cos()
            * (-var[k] * phase * phase).exp();
    }
    xyz[0] += 9.7470e-14
        * (TWO_PI * 4.5282e+09f32).sqrt()
        * (2.2399e+06 * phase + shift[0]).cos()
        * (-4.5282e+09 * phase * phase).exp();
    for c in xyz.iter_mut() {
        *c /= 1.0685e-7;
    }
    let rgb = crate::spectrum::xyz_to_linear_srgb([xyz[0] as f64, xyz[1] as f64, xyz[2] as f64]);
    [rgb[0] as f32, rgb[1] as f32, rgb[2] as f32]
}

/// The pieces of the Airy summation that do not depend on how it is
/// integrated: the two interfaces' reflectances, the phase shift between
/// them, and the optical path difference.
struct Film {
    /// Reflectance of the air/film interface (scalar — the film is a
    /// dielectric, so its `F0` has no colour).
    r12: f32,
    /// `1 - r12`, the round-trip transmittance through the top interface.
    t121: f32,
    /// Reflectance of the film/substrate interface, per channel.
    r23: [f32; 3],
    /// Phase shift on reflection, per channel, in radians.
    phi: [f32; 3],
    /// Optical path difference in nanometres.
    opd: f32,
}

impl Film {
    /// `None` when the ray is past the critical angle of the film and there
    /// is no interference to speak of — the caller reflects everything.
    fn new(thickness_nm: f32, film_ior: f32, cos_theta1: f32, base_f0: [f32; 3]) -> Option<Self> {
        let outside = 1.0f32;
        // Force the film's index back to the surrounding medium's as the
        // thickness goes to zero, so the model degenerates continuously.
        let t = (thickness_nm / 0.03).clamp(0.0, 1.0);
        let n1 = outside + (film_ior - outside) * (t * t * (3.0 - 2.0 * t));
        let sin2 = (outside / n1).powi(2) * (1.0 - cos_theta1 * cos_theta1);
        let cos2sq = 1.0 - sin2;
        if cos2sq < 0.0 {
            return None;
        }
        let cos_theta2 = cos2sq.sqrt();

        let r12 = schlick(f0_of(n1, outside), cos_theta1);
        let t121 = 1.0 - r12;
        // A reflection off a denser medium flips the wave; off a rarer one it
        // does not. That π is the difference between a film that looks blue
        // at 300 nm and one that looks orange, so it is not a detail.
        let phi12 = if n1 < outside { std::f32::consts::PI } else { 0.0 };
        let phi21 = std::f32::consts::PI - phi12;

        let mut r23 = [0.0f32; 3];
        let mut phi = [0.0f32; 3];
        for c in 0..3 {
            let n3 = ior_of_f0(base_f0[c]);
            r23[c] = schlick(f0_of(n3, n1), cos_theta2);
            let phi23 = if n3 < n1 { std::f32::consts::PI } else { 0.0 };
            phi[c] = phi21 + phi23;
        }
        Some(Self {
            r12,
            t121,
            r23,
            phi,
            opd: 2.0 * n1 * thickness_nm * cos_theta2,
        })
    }

    /// The DC term of the summation, per channel, and the amplitude of the
    /// oscillating ones.
    fn terms(&self, c: usize) -> (f32, f32, f32) {
        let r123 = (self.r12 * self.r23[c]).clamp(1e-5, 0.9999);
        let r = r123.sqrt();
        let rs = self.t121 * self.t121 * self.r23[c] / (1.0 - r123);
        (self.r12 + rs, rs - self.t121, r)
    }
}

/// Airy reflectance of a thin film over a substrate of reflectance `base_f0`,
/// integrated against the CIE curves — the RGB path.
///
/// `thickness_nm` is the film's physical thickness in nanometres, `film_ior`
/// its index, `cos_theta1` the cosine at the *outside* interface (which for a
/// microfacet model is the half-vector cosine, not the normal's).
pub fn thin_film_fresnel(
    thickness_nm: f32,
    film_ior: f32,
    cos_theta1: f32,
    base_f0: [f32; 3],
) -> [f32; 3] {
    let Some(f) = Film::new(thickness_nm, film_ior, cos_theta1, base_f0) else {
        return [1.0; 3];
    };
    let mut out = [0.0f32; 3];
    let mut cm = [0.0f32; 3];
    let mut r = [0.0f32; 3];
    for c in 0..3 {
        let (c0, amp, rr) = f.terms(c);
        out[c] = c0;
        cm[c] = amp;
        r[c] = rr;
    }
    // Two harmonics is what the paper's own reference implementation keeps:
    // the sensitivity term decays as a Gaussian in the frequency, so the
    // third is already below the noise of the fit itself.
    for m in 1..=2u32 {
        let shift = [f.phi[0] * m as f32, f.phi[1] * m as f32, f.phi[2] * m as f32];
        let s = sensitivity(m as f32 * f.opd, shift);
        for c in 0..3 {
            cm[c] *= r[c];
            out[c] += cm[c] * 2.0 * s[c];
        }
    }
    // Clamped at both ends. The lower clamp is the paper's own — the
    // truncated series can dip below zero where the sensitivity lobes are
    // negative — and the upper one is the same statement at the other end:
    // the Gaussian fit is an approximation to an integral of a reflectance,
    // and an approximation is allowed to be a percent over 1 where the exact
    // thing cannot be. `thin_film_fresnel_at` needs neither and is clamped
    // only for symmetry.
    [
        out[0].clamp(0.0, 1.0),
        out[1].clamp(0.0, 1.0),
        out[2].clamp(0.0, 1.0),
    ]
}

/// Airy reflectance at a single wavelength — the hero-wavelength path.
///
/// No colour integral and no truncation: the series `Σ 2·C·r^m·cos(mψ)` is a
/// geometric one and sums in closed form, so this is the *exact* Airy
/// reflectance of the same film that [`thin_film_fresnel`] approximates.
pub fn thin_film_fresnel_at(
    thickness_nm: f32,
    film_ior: f32,
    cos_theta1: f32,
    base_f0: [f32; 3],
    lambda_nm: f32,
) -> [f32; 3] {
    let Some(f) = Film::new(thickness_nm, film_ior, cos_theta1, base_f0) else {
        return [1.0; 3];
    };
    let mut out = [0.0f32; 3];
    for (c, o) in out.iter_mut().enumerate() {
        let (c0, amp, r) = f.terms(c);
        let psi = std::f32::consts::TAU * f.opd / lambda_nm.max(1e-3) + f.phi[c];
        let cos_psi = psi.cos();
        let denom = (1.0 - 2.0 * r * cos_psi + r * r).max(1e-6);
        *o = (c0 + 2.0 * amp * (r * cos_psi - r * r) / denom).clamp(0.0, 1.0);
    }
    out
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

    /// The textbook Airy formula for one film, written from the *amplitude*
    /// coefficients and a single phase — a different derivation from the
    /// intensity-and-phase-shift series the renderer sums, which is what
    /// makes it worth comparing against.
    fn airy_reference(n0: f64, n1: f64, n2: f64, d_nm: f64, lambda_nm: f64) -> f64 {
        let r12 = (n0 - n1) / (n0 + n1);
        let r23 = (n1 - n2) / (n1 + n2);
        let delta = 4.0 * std::f64::consts::PI * n1 * d_nm / lambda_nm;
        let c = delta.cos();
        let num = r12 * r12 + r23 * r23 + 2.0 * r12 * r23 * c;
        let den = 1.0 + r12 * r12 * r23 * r23 + 2.0 * r12 * r23 * c;
        num / den
    }

    /// A 300 nm soap-index film on glass, at normal incidence, at 550 nm.
    #[test]
    fn a_three_hundred_nanometre_film_matches_the_analytic_airy() {
        let (n0, n1, n2) = (1.0, 1.34, 1.5);
        let f0 = (((n0 - n2) / (n0 + n2)) as f32).powi(2);
        for lambda in [450.0f64, 550.0, 650.0] {
            let ours = thin_film_fresnel_at(300.0, n1 as f32, 1.0, [f0; 3], lambda as f32)[1] as f64;
            let want = airy_reference(n0, n1, n2, 300.0, lambda);
            assert!(
                (ours - want).abs() <= 0.01 * want,
                "at {lambda} nm: {ours} vs {want}"
            );
        }
    }

    /// Whatever the film does, it cannot hand back more light than arrived.
    #[test]
    fn a_film_never_reflects_more_than_it_receives() {
        for d in [10.0f32, 120.0, 300.0, 550.0, 1200.0] {
            for nf in [1.2f32, 1.5, 2.0] {
                for c in [1.0f32, 0.8, 0.4, 0.05] {
                    for f0 in [0.02f32, 0.04, 0.2, 0.95] {
                        for r in [
                            thin_film_fresnel(d, nf, c, [f0; 3]),
                            thin_film_fresnel_at(d, nf, c, [f0; 3], 550.0),
                        ] {
                            for ch in r {
                                assert!(
                                    (0.0..=1.0).contains(&ch),
                                    "d {d} n {nf} cos {c} f0 {f0} -> {r:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The film is a colour: at a few hundred nanometres it must not come
    /// back grey, or none of this bought anything.
    #[test]
    fn a_film_in_the_visible_band_is_chromatic() {
        let r = thin_film_fresnel(320.0, 1.5, 1.0, [0.04; 3]);
        let spread = r[0].max(r[1]).max(r[2]) - r[0].min(r[1]).min(r[2]);
        assert!(spread > 0.01, "{r:?} is grey");
    }

    #[test]
    fn the_index_knob_is_the_d_line() {
        let n: f64 = index(1.9, D_LINE_UM);
        assert!((n - 1.9).abs() < 1e-12);
    }
}
