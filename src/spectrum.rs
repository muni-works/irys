//! One wavelength at a time: the colour half of dispersion.
//!
//! RGB rendering asks "how much red, green and blue came back". That question
//! has no answer at a prism, because the prism's whole behaviour is that the
//! index — and therefore the geometry of the path — depends on the wavelength,
//! not on a band. A path through glass has to *be* a wavelength.
//!
//! So a path that meets a dispersive material stops being RGB and becomes
//! monochromatic: it draws one **hero wavelength** `λ`, uniform over the
//! visible band, and from there carries a scalar spectral throughput. What
//! comes back is a radiance at `λ`, and this module is the piece that turns
//! that back into the linear-sRGB triple the film stores.
//!
//! # The estimator
//!
//! At the moment of the switch, the path's RGB throughput is multiplied by
//! [`hero_weight`]:
//!
//! ```text
//! w(λ) = r̄(λ) / p(λ)      p(λ) = 1 / (λ_max − λ_min)
//! ```
//!
//! where `r̄` is the linear-sRGB colour-matching response normalised so that
//! `∫ r̄(λ) dλ = (1, 1, 1)` over the band. That normalisation is the whole
//! trick: for a material whose behaviour does *not* actually vary with `λ`,
//! `E[w(λ)] = (1, 1, 1)` exactly, so the spectral estimator has the same mean
//! as the RGB one it replaced. Dispersion only shows up when the geometry
//! downstream of the switch genuinely depends on `λ` — which is exactly when
//! it should. [`hero_weight_integrates_to_white`] pins it.
//!
//! Because the sRGB primaries do not contain the spectral locus, `r̄` has
//! negative lobes (red goes negative through the cyans). A single hero sample
//! can therefore land negative in a channel; the mean is still right, and the
//! film clamps at tone-map time. That is the price of one wavelength per path
//! rather than four, and it is paid in noise, not in bias.
//!
//! # The colour matching functions
//!
//! Wyman, Sloan and Shirley's (2013) multi-lobe Gaussian fit to CIE 1931 —
//! seven skew-Gaussians, no tables, max error well under a percent of peak.
//! A table would be more accurate and would also have to be uploaded to the
//! GPU and interpolated there; the fit is twenty lines that port to WGSL
//! verbatim, which is what keeps the two tiers honest.

/// Shortest wavelength a path may draw, in nanometres.
pub const LAMBDA_MIN_NM: f64 = 380.0;
/// Longest wavelength a path may draw, in nanometres.
pub const LAMBDA_MAX_NM: f64 = 780.0;

/// Skew-Gaussian: one `sigma` below the peak, another above.
#[inline]
fn gauss(x: f64, mu: f64, s1: f64, s2: f64) -> f64 {
    let s = if x < mu { s1 } else { s2 };
    let t = (x - mu) / s;
    (-0.5 * t * t).exp()
}

/// CIE 1931 `(x̄, ȳ, z̄)` at `lambda_nm`, from the Wyman/Sloan/Shirley fit.
pub fn cie_xyz(lambda_nm: f64) -> [f64; 3] {
    let x = 1.056 * gauss(lambda_nm, 599.8, 37.9, 31.0)
        + 0.362 * gauss(lambda_nm, 442.0, 16.0, 26.7)
        - 0.065 * gauss(lambda_nm, 501.1, 20.4, 26.2);
    let y = 0.821 * gauss(lambda_nm, 568.8, 46.9, 40.5)
        + 0.286 * gauss(lambda_nm, 530.9, 16.3, 31.1);
    let z = 1.217 * gauss(lambda_nm, 437.0, 11.8, 36.0)
        + 0.681 * gauss(lambda_nm, 459.0, 26.0, 13.8);
    [x, y, z]
}

/// CIE XYZ (D65) to linear sRGB.
#[inline]
pub fn xyz_to_linear_srgb(c: [f64; 3]) -> [f64; 3] {
    [
        3.240_454_2 * c[0] - 1.537_138_5 * c[1] - 0.498_531_4 * c[2],
        -0.969_266_0 * c[0] + 1.876_010_8 * c[1] + 0.041_556_0 * c[2],
        0.055_643_4 * c[0] - 0.204_025_9 * c[1] + 1.057_225_2 * c[2],
    ]
}

/// Reciprocals of `∫ rgb(λ) dλ` over the band, per channel — the constants
/// that make the response integrate to white.
///
/// Quadrature over the fit above at 1 pm; `hero_weight_integrates_to_white`
/// re-derives them at test time, so they cannot silently rot.
const RESPONSE_NORM: [f64; 3] = [
    1.0 / 128.361_021_397_897_6,
    1.0 / 101.538_081_160_242_4,
    1.0 / 97.064_804_112_803_7,
];

/// The weight a path picks up when it becomes monochromatic at `lambda_nm`:
/// the normalised linear-sRGB response divided by the wavelength PDF.
///
/// Multiply the RGB throughput by this once, at the switch, and nowhere else.
pub fn hero_weight(lambda_nm: f64) -> [f32; 3] {
    let rgb = xyz_to_linear_srgb(cie_xyz(lambda_nm));
    let band = LAMBDA_MAX_NM - LAMBDA_MIN_NM;
    [
        (rgb[0] * RESPONSE_NORM[0] * band) as f32,
        (rgb[1] * RESPONSE_NORM[1] * band) as f32,
        (rgb[2] * RESPONSE_NORM[2] * band) as f32,
    ]
}

/// Map a uniform `u` in `[0, 1)` onto the visible band.
#[inline]
pub fn sample_lambda_nm(u: f64) -> f64 {
    LAMBDA_MIN_NM + (LAMBDA_MAX_NM - LAMBDA_MIN_NM) * u
}

/// Sellmeier's index from the three `B` and three `C` coefficients, with the
/// wavelength in microns.
///
/// The general form of [`crate::optics::sellmeier`], which hard-codes N-BK7:
/// a material carries its own glass, so it carries its own coefficients.
#[inline]
pub fn sellmeier_index(b: [f64; 3], c: [f64; 3], lambda_um: f64) -> f64 {
    let l2 = lambda_um * lambda_um;
    let mut n2 = 1.0;
    for k in 0..3 {
        let d = l2 - c[k];
        if d.abs() > 1e-12 {
            n2 += b[k] * l2 / d;
        }
    }
    n2.max(1.0).sqrt()
}

/// N-BK7's Sellmeier coefficients — the pair `light.rs` traces its caustic
/// with, exported so a material can be given a real glass rather than an
/// Abbe number.
pub const BK7_SELLMEIER: ([f64; 3], [f64; 3]) = (
    [1.039_612_12, 0.231_792_344, 1.010_469_45],
    [0.006_000_698_67, 0.020_017_914_4, 103.560_653],
);

/// The Fraunhofer lines an Abbe number is defined against, in microns.
pub const LINE_F_UM: f64 = 0.486_13;
/// The helium d line, where `ior` is quoted.
pub const LINE_D_UM: f64 = 0.587_56;
/// The hydrogen C line.
pub const LINE_C_UM: f64 = 0.656_27;

/// Cauchy dispersion reconstructed from an index and an Abbe number —
/// OpenPBR's `specular_ior` / `specular_ior_dispersion` pair.
///
/// The Abbe number `V = (n_d − 1) / (n_F − n_C)` fixes exactly one degree of
/// freedom, so the one-term Cauchy `n(λ) = A + B/λ²` is the model it
/// determines: solve `B` from the F-to-C spread and `A` from the d line.
/// `V = 0` means "no dispersion" and returns `nd` unchanged.
#[inline]
pub fn cauchy_index(nd: f64, abbe: f64, lambda_um: f64) -> f64 {
    if abbe <= 0.0 {
        return nd;
    }
    let inv2 = |l: f64| 1.0 / (l * l);
    let b = (nd - 1.0) / (abbe * (inv2(LINE_F_UM) - inv2(LINE_C_UM)));
    let a = nd - b * inv2(LINE_D_UM);
    a + b * inv2(lambda_um)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The normalisation constants are the whole reason a spectral path has
    /// the same mean as the RGB path it replaced.
    #[test]
    fn hero_weight_integrates_to_white() {
        let n = 40_000;
        let mut acc = [0.0f64; 3];
        for i in 0..n {
            let u = (i as f64 + 0.5) / n as f64;
            let w = hero_weight(sample_lambda_nm(u));
            for c in 0..3 {
                acc[c] += w[c] as f64 / n as f64;
            }
        }
        for c in 0..3 {
            assert!((acc[c] - 1.0).abs() < 2e-3, "channel {c} averaged {}", acc[c]);
        }
    }

    #[test]
    fn the_general_sellmeier_reproduces_bk7() {
        let (b, c) = BK7_SELLMEIER;
        for l in [0.45, 0.5876, 0.65] {
            let general = sellmeier_index(b, c, l);
            let hardcoded: f64 = crate::optics::sellmeier(l);
            assert!((general - hardcoded).abs() < 1e-12, "{general} vs {hardcoded}");
        }
    }

    /// N-BK7 is n_d = 1.5168, V_d = 64.17. Cauchy from those two numbers
    /// should land within a few units in the fourth decimal of Sellmeier
    /// across the visible band — which is what makes `abbe` a usable stand-in
    /// for a datasheet.
    #[test]
    fn cauchy_tracks_bk7_sellmeier() {
        let (b, c) = BK7_SELLMEIER;
        for l in [LINE_F_UM, 0.55, LINE_D_UM, LINE_C_UM] {
            let cauchy = cauchy_index(1.5168, 64.17, l);
            let exact = sellmeier_index(b, c, l);
            assert!((cauchy - exact).abs() < 1e-3, "at {l}um: {cauchy} vs {exact}");
        }
    }

    #[test]
    fn zero_abbe_is_no_dispersion() {
        for l in [0.4, 0.6, 0.7] {
            assert_eq!(cauchy_index(1.52, 0.0, l), 1.52);
        }
    }

    #[test]
    fn blue_bends_more_than_red() {
        let (b, c) = BK7_SELLMEIER;
        assert!(sellmeier_index(b, c, 0.45) > sellmeier_index(b, c, 0.65));
        assert!(cauchy_index(1.5168, 64.17, 0.45) > cauchy_index(1.5168, 64.17, 0.65));
    }
}
