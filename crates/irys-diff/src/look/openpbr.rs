//! OpenPBR Surface 1.x, spectral, as the spec defines it.
//!
//! The ASWF specification (academysoftwarefoundation.github.io/OpenPBR)
//! builds the surface from slabs combined by two operators,
//!
//! ```text
//! M_surface         = layer(M_coated_base, S_fuzz, fuzz_weight)
//! M_coated_base     = layer(M_base_substrate, S_coat, coat_weight)
//! M_base_substrate  = mix(M_dielectric_base, S_metal, base_metalness)
//! M_dielectric_base = mix(M_opaque_base, S_translucent, transmission_weight)
//! M_opaque_base     = mix(M_glossy_diffuse, S_subsurface, subsurface_weight)
//! M_glossy_diffuse  = layer(S_diffuse, S_gloss)
//! ```
//!
//! and its reference evaluation of `layer` is **albedo scaling**:
//! `f = w f_top + lerp(1, T_top (1 − E_top(ω_o)), w) f_sub`. That is what
//! this module implements (design.md, Phase 4, says why it is not the
//! position-free walk the design first proposed). The albedos `E` are
//! tabulated here by quadrature of the very lobes the integrator evaluates,
//! so a white furnace closes to the table's accuracy.
//!
//! Everything is evaluated at one wavelength: colours are lifted onto the
//! three hats of [`crate::ggx::band_weights`], the dielectric index follows
//! Cauchy from the Abbe number, and the thin film is the exact two-beam
//! Airy sum at that wavelength (s and p averaged), not Belcour–Barla's RGB
//! fit.
//!
//! This module is the plain f64 side: parameters, Fresnel, the film, the
//! albedo tables, and the per-vertex [`At`] that builds a lane's columns for
//! [`super::graph::Stack`] and samples directions (detached).

use std::f64::consts::PI;
use std::sync::LazyLock;

use crate::ggx::band_weights;

/// The smallest GGX `α`: `α = A_MIN + (1 − A_MIN) r²` keeps `r = 0` finite
/// and smooth in `r`.
pub const A_MIN: f64 = 1e-3;

/// `μ̄` of the F82 tint, `1/7` (cos 82°, near enough; the spec's own value).
pub const MU_BAR: f64 = 1.0 / 7.0;

/// GGX `α` of a perceptual roughness.
pub fn alpha(r: f64) -> f64 {
    A_MIN + (1.0 - A_MIN) * r * r
}

/// OpenPBR Surface's parameters, the spec's names and defaults. Colours are
/// linear sRGB. Anisotropy, `base_diffuse_roughness`, `transmission_scatter`
/// and thin-walled mode are not implemented (design.md, Phase 4 deviations).
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(missing_docs)]
pub struct OpenPbr {
    pub base_weight: f64,
    pub base_color: [f64; 3],
    pub base_metalness: f64,
    pub specular_weight: f64,
    pub specular_color: [f64; 3],
    pub specular_roughness: f64,
    pub specular_ior: f64,
    pub transmission_weight: f64,
    pub transmission_color: [f64; 3],
    pub transmission_depth: f64,
    pub transmission_dispersion_scale: f64,
    pub transmission_dispersion_abbe_number: f64,
    pub subsurface_weight: f64,
    pub subsurface_color: [f64; 3],
    pub subsurface_radius: f64,
    pub subsurface_radius_scale: [f64; 3],
    pub subsurface_scatter_anisotropy: f64,
    pub coat_weight: f64,
    pub coat_color: [f64; 3],
    pub coat_roughness: f64,
    pub coat_ior: f64,
    pub coat_darkening: f64,
    pub fuzz_weight: f64,
    pub fuzz_color: [f64; 3],
    pub fuzz_roughness: f64,
    pub thin_film_weight: f64,
    /// Micrometres.
    pub thin_film_thickness: f64,
    pub thin_film_ior: f64,
    /// cd/m² in the spec; here a radiance scale.
    pub emission_luminance: f64,
    pub emission_color: [f64; 3],
}

impl Default for OpenPbr {
    fn default() -> Self {
        Self {
            base_weight: 1.0,
            base_color: [0.8; 3],
            base_metalness: 0.0,
            specular_weight: 1.0,
            specular_color: [1.0; 3],
            specular_roughness: 0.3,
            specular_ior: 1.5,
            transmission_weight: 0.0,
            transmission_color: [1.0; 3],
            transmission_depth: 0.0,
            transmission_dispersion_scale: 0.0,
            transmission_dispersion_abbe_number: 20.0,
            subsurface_weight: 0.0,
            subsurface_color: [0.8; 3],
            subsurface_radius: 1.0,
            subsurface_radius_scale: [1.0, 0.5, 0.25],
            subsurface_scatter_anisotropy: 0.0,
            coat_weight: 0.0,
            coat_color: [1.0; 3],
            coat_roughness: 0.0,
            coat_ior: 1.6,
            coat_darkening: 1.0,
            fuzz_weight: 0.0,
            fuzz_color: [1.0; 3],
            fuzz_roughness: 0.5,
            thin_film_weight: 0.0,
            thin_film_thickness: 0.5,
            thin_film_ior: 1.4,
            emission_luminance: 0.0,
            emission_color: [1.0; 3],
        }
    }
}

/// The differentiable parameters, in [`super::graph::Stack`]'s order.
pub const PARAMS: usize = 8;

/// Names of `θ`.
pub const PARAM_NAMES: [&str; PARAMS] = [
    "base_color@450",
    "base_color@550",
    "base_color@650",
    "base_metalness",
    "specular_roughness",
    "coat_weight",
    "fuzz_weight",
    "base_weight",
];

impl OpenPbr {
    /// `θ`: base colour on the three hats (450/550/650 nm, [`hats`]),
    /// metalness, specular roughness, coat weight, fuzz weight, base weight.
    pub fn theta(&self) -> [f64; PARAMS] {
        let c = hats(self.base_color);
        [
            c[0],
            c[1],
            c[2],
            self.base_metalness,
            self.specular_roughness,
            self.coat_weight,
            self.fuzz_weight,
            self.base_weight,
        ]
    }

    /// The material with `θ` written back.
    pub fn with_theta(mut self, t: &[f64; PARAMS]) -> Self {
        self.base_color = rgb_of_hats([t[0], t[1], t[2]]);
        self.base_metalness = t[3];
        self.specular_roughness = t[4];
        self.coat_weight = t[5];
        self.fuzz_weight = t[6];
        self.base_weight = t[7];
        self
    }

    /// The dielectric's index at `lambda` (nm): Cauchy through
    /// `specular_ior` with `V_d = abbe / scale` (spec, "Dispersion").
    pub fn ior_at(&self, lambda: f64) -> f64 {
        let s = self.transmission_dispersion_scale;
        if s <= 0.0 {
            return self.specular_ior;
        }
        let v = self.transmission_dispersion_abbe_number / s;
        irys_light::spectrum::cauchy_index(self.specular_ior, v, lambda * 1e-3)
    }
}

/// `M[c][k]`: linear-sRGB channel `c` of a frame whose reflectance is hat
/// `k` (450, 550, 650 nm) under a flat illuminant, as the hero-wavelength
/// integrator renders it (its `hero_weight` averaged over the band).
pub fn hat_matrix() -> &'static [[f64; 3]; 3] {
    static M: LazyLock<[[f64; 3]; 3]> = LazyLock::new(|| {
        let n = 8000;
        let mut m = [[0.0; 3]; 3];
        for i in 0..n {
            let u = (i as f64 + 0.5) / n as f64;
            let lambda = irys_light::spectrum::sample_lambda_nm(u);
            let w = irys_light::spectrum::hero_weight(lambda);
            let b = band_weights(lambda);
            for c in 0..3 {
                for k in 0..3 {
                    m[c][k] += w[c] as f64 * b[k] / n as f64;
                }
            }
        }
        m
    });
    &M
}

fn inv3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let d = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let c = |i: usize, j: usize| {
        let (a, b) = ((i + 1) % 3, (i + 2) % 3);
        let (p, q) = ((j + 1) % 3, (j + 2) % 3);
        m[a][p] * m[b][q] - m[a][q] * m[b][p]
    };
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[j][i] = c(i, j) / d;
        }
    }
    r
}

/// The hat coefficients whose spectrum renders as linear-sRGB `rgb`
/// (`M⁻¹ rgb`): the spectral uplift every colour parameter goes through,
/// so a swatch's colour round-trips through the spectral integrator.
pub fn hats(rgb: [f64; 3]) -> [f64; 3] {
    static MI: LazyLock<[[f64; 3]; 3]> = LazyLock::new(|| inv3(hat_matrix()));
    std::array::from_fn(|k| (0..3).map(|c| MI[k][c] * rgb[c]).sum())
}

/// `M h`: the colour hats `h` render as.
pub fn rgb_of_hats(h: [f64; 3]) -> [f64; 3] {
    let m = hat_matrix();
    std::array::from_fn(|c| (0..3).map(|k| m[c][k] * h[k]).sum())
}

/// Linear-sRGB `rgb` at `lambda`: [`hats`] on the three hats.
pub fn lift(rgb: [f64; 3], lambda: f64) -> f64 {
    let w = band_weights(lambda);
    let h = hats(rgb);
    w[0] * h[0] + w[1] * h[1] + w[2] * h[2]
}

// ─── Fresnel and the film ────────────────────────────────────────────────

/// Exact unpolarised dielectric reflectance at cosine `c` (≥ 0) for the
/// relative index `eta = n_t / n_i`; 1 under total internal reflection.
pub fn fresnel(c: f64, eta: f64) -> f64 {
    let c = c.clamp(0.0, 1.0);
    let g2 = eta * eta - 1.0 + c * c;
    if g2 <= 0.0 {
        return 1.0;
    }
    let g = g2.sqrt();
    let a = (g - c) / (g + c);
    let b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    0.5 * a * a * (1.0 + b * b)
}

/// The specular-weight-modulated index (spec, "Specular"): `F_s` scaled by
/// `ξ_s`, inverted back to an index.
pub fn modulated_ior(eta: f64, xi: f64) -> f64 {
    let fs = ((1.0 - eta) / (1.0 + eta)).powi(2);
    let xi = xi.clamp(0.0, if fs > 0.0 { 1.0 / fs } else { 1.0 });
    let e = (eta - 1.0).signum() * (xi * fs).sqrt();
    (1.0 + e) / (1.0 - e).max(1e-9)
}

/// Exact Airy reflectance of a film of index `nf` and thickness `d_nm`
/// between air and a substrate of real index `ns`, at external cosine `c`
/// and wavelength `lambda` (nm): `|r|²` of the two-interface sum, s and p
/// averaged. No series, no colour fit: irys is spectral.
pub fn airy(c: f64, nf: f64, d_nm: f64, ns: f64, lambda: f64) -> f64 {
    let c1 = c.clamp(1e-6, 1.0);
    let s1 = (1.0 - c1 * c1).max(0.0).sqrt();
    let s2 = s1 / nf;
    let c2 = (1.0 - s2 * s2).max(0.0).sqrt();
    let s3 = s1 / ns;
    let c3 = (1.0 - s3 * s3).max(0.0).sqrt();
    let delta = 4.0 * PI * nf * d_nm * c2 / lambda;
    let one = |r12: f64, r23: f64| {
        let k = 2.0 * r12 * r23 * delta.cos();
        (r12 * r12 + r23 * r23 + k) / (1.0 + r12 * r12 * r23 * r23 + k)
    };
    let rs12 = (c1 - nf * c2) / (c1 + nf * c2);
    let rs23 = (nf * c2 - ns * c3) / (nf * c2 + ns * c3);
    let rp12 = (nf * c1 - c2) / (nf * c1 + c2);
    let rp23 = (ns * c2 - nf * c3) / (ns * c2 + nf * c3);
    0.5 * (one(rs12, rs23) + one(rp12, rp23))
}

/// Hemispherical average of the Fresnel factor, `2 ∫ F(μ) μ dμ`.
pub fn fresnel_average(eta: f64) -> f64 {
    let n = 128;
    (0..n)
        .map(|k| {
            let mu = (k as f64 + 0.5) / n as f64;
            2.0 * fresnel(mu, eta) * mu / n as f64
        })
        .sum()
}

// ─── GGX ─────────────────────────────────────────────────────────────────

/// GGX `D(cos θh)`.
pub fn ggx_d(a: f64, ch: f64) -> f64 {
    let a2 = a * a;
    let k = ch * ch * (a2 - 1.0) + 1.0;
    a2 / (PI * k * k)
}

/// Smith `Λ(c)`.
pub fn ggx_lambda(a: f64, c: f64) -> f64 {
    let c2 = (c * c).max(1e-12);
    ((1.0 + a * a * (1.0 - c2) / c2).sqrt() - 1.0) * 0.5
}

/// Visible-normal sample of a half vector for `wo` (local, `wo.z > 0`).
pub fn vndf(a: f64, wo: [f64; 3], u: [f64; 2]) -> [f64; 3] {
    let vh = norm([a * wo[0], a * wo[1], wo[2]]);
    let len2 = vh[0] * vh[0] + vh[1] * vh[1];
    let t1 = if len2 > 0.0 {
        [-vh[1] / len2.sqrt(), vh[0] / len2.sqrt(), 0.0]
    } else {
        [1.0, 0.0, 0.0]
    };
    let t2 = cross(vh, t1);
    let r = u[0].sqrt();
    let phi = 2.0 * PI * u[1];
    let p1 = r * phi.cos();
    let s = 0.5 * (1.0 + vh[2]);
    let p2 = (1.0 - s) * (1.0 - p1 * p1).max(0.0).sqrt() + s * r * phi.sin();
    let p3 = (1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt();
    let nh = [
        p1 * t1[0] + p2 * t2[0] + p3 * vh[0],
        p1 * t1[1] + p2 * t2[1] + p3 * vh[1],
        p1 * t1[2] + p2 * t2[2] + p3 * vh[2],
    ];
    norm([a * nh[0], a * nh[1], nh[2].max(1e-12)])
}

/// The visible-normal density of `h` for `wo`: `G1(o) |o·h| D(h) / cos o`.
pub fn vndf_pdf(a: f64, co: f64, ch: f64, oh: f64) -> f64 {
    ggx_d(a, ch) * oh.abs() / ((1.0 + ggx_lambda(a, co)) * co)
}

// ─── small vector helpers (local frame, z the normal) ────────────────────

/// Dot product.
pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Cross product.
pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Unit length.
pub fn norm(a: [f64; 3]) -> [f64; 3] {
    let l = dot(a, a).sqrt();
    [a[0] / l, a[1] / l, a[2] / l]
}

/// Mirror `wo` about `h`.
pub fn reflect(wo: [f64; 3], h: [f64; 3]) -> [f64; 3] {
    let k = 2.0 * dot(wo, h);
    [k * h[0] - wo[0], k * h[1] - wo[1], k * h[2] - wo[2]]
}

/// Refract `wo` through microfacet `h` (`wo·h > 0`) for `etap = n_t/n_i`;
/// `None` under total internal reflection.
pub fn refract(wo: [f64; 3], h: [f64; 3], etap: f64) -> Option<[f64; 3]> {
    let c = dot(wo, h);
    let s2 = (1.0 - c * c).max(0.0) / (etap * etap);
    if s2 >= 1.0 {
        return None;
    }
    let ct = (1.0 - s2).sqrt();
    Some([
        -wo[0] / etap + (c / etap - ct) * h[0],
        -wo[1] / etap + (c / etap - ct) * h[1],
        -wo[2] / etap + (c / etap - ct) * h[2],
    ])
}

/// A cosine direction about `+z`.
pub fn cosine(u: [f64; 2]) -> [f64; 3] {
    let r = u[0].sqrt();
    let phi = 2.0 * PI * u[1];
    [r * phi.cos(), r * phi.sin(), (1.0 - u[0]).max(0.0).sqrt()]
}

// ─── the albedo tables ───────────────────────────────────────────────────

/// Nodes per axis.
pub const TN: usize = 32;
const QN: usize = 24;

fn mu_node(k: usize) -> f64 {
    (k as f64 / (TN - 1) as f64).max(1e-3)
}
fn r_node(k: usize) -> f64 {
    k as f64 / (TN - 1) as f64
}
const LN_ETA: f64 = 1.098_612_288_668_109_7; // ln 3
/// Index nodes: an odd count, so `η = 1` (where total internal reflection
/// switches on below it) is a node and never interpolated across.
pub const NE: usize = 33;
fn eta_node(k: usize) -> f64 {
    (LN_ETA * (2.0 * k as f64 / (NE - 1) as f64 - 1.0)).exp()
}

/// `(i, t)` of `x` on a uniform grid over `[0, 1]`.
fn cell(x: f64) -> (usize, f64) {
    let s = x.clamp(0.0, 1.0) * (TN - 1) as f64;
    let i = (s.floor() as usize).min(TN - 2);
    (i, s - i as f64)
}

/// Reflection albedo of a smooth-Fresnel GGX lobe by quadrature of the
/// visible normals: `∫ F(o·h) G2/G1 dP(h)` (`fres` is the Fresnel).
fn albedo_ggx(mu: f64, a: f64, fres: impl Fn(f64) -> f64) -> f64 {
    let wo = [(1.0 - mu * mu).max(0.0).sqrt(), 0.0, mu];
    let mut s = 0.0;
    for i in 0..QN {
        for j in 0..QN {
            let u = [(i as f64 + 0.5) / QN as f64, (j as f64 + 0.5) / QN as f64];
            let h = vndf(a, wo, u);
            let wi = reflect(wo, h);
            if wi[2] <= 0.0 {
                continue;
            }
            let g1 = 1.0 / (1.0 + ggx_lambda(a, mu));
            let g2 = 1.0 / (1.0 + ggx_lambda(a, mu) + ggx_lambda(a, wi[2]));
            s += fres(dot(wo, h)) * g2 / g1;
        }
    }
    s / (QN * QN) as f64
}

/// The tables, built once (≈ 1 s).
pub struct Tables {
    /// `E_d(μ, r, η)`: rough dielectric reflection, `[μ][r][η]`.
    pub diel: Vec<f64>,
    /// Metal: `∫ G2/G1`, `∫ (1 − o·h)⁵ G2/G1`, `∫ w(o·h) G2/G1`, `[μ][r]`.
    pub metal: [Vec<f64>; 3],
    /// The fuzz lobe's own albedo, `[μ][r_f]`.
    pub fuzz: Vec<f64>,
}

/// The F82 weight `μ (1 − μ)⁶ / (μ̄ (1 − μ̄)⁶)`.
pub fn f82_w(mu: f64) -> f64 {
    mu * (1.0 - mu).powi(6) / (MU_BAR * (1.0 - MU_BAR).powi(6))
}

impl Tables {
    /// The process-wide tables.
    pub fn get() -> &'static Tables {
        static T: LazyLock<Tables> = LazyLock::new(Tables::build);
        &T
    }

    fn build() -> Self {
        // Plain threads, not rayon: `get` is reached from inside rayon
        // workers, and a pool blocked on this lazy init cannot run it.
        fn par(n: usize, f: impl Fn(usize) -> f64 + Sync) -> Vec<f64> {
            let threads = std::thread::available_parallelism().map_or(4, |x| x.get());
            let mut out = vec![0.0; n];
            let chunk = n.div_ceil(threads);
            std::thread::scope(|s| {
                for (c, part) in out.chunks_mut(chunk).enumerate() {
                    let f = &f;
                    s.spawn(move || {
                        for (i, v) in part.iter_mut().enumerate() {
                            *v = f(c * chunk + i);
                        }
                    });
                }
            });
            out
        }
        let diel: Vec<f64> = par(TN * TN * NE, |k| {
                let (m, r, e) = (k / (TN * NE), (k / NE) % TN, k % NE);
                let eta = eta_node(e);
                albedo_ggx(mu_node(m), alpha(r_node(r)), |c| fresnel(c, eta))
            });
        let metal_k = |w: usize| -> Vec<f64> {
            par(TN * TN, |k| {
                let (m, r) = (k / TN, k % TN);
                albedo_ggx(mu_node(m), alpha(r_node(r)), |c| match w {
                    0 => 1.0,
                    1 => (1.0 - c).powi(5),
                    _ => f82_w(c),
                })
            })
        };
        let metal = [metal_k(0), metal_k(1), metal_k(2)];
        let fuzz = par(TN * TN, |k| {
                let (m, r) = (k / TN, k % TN);
                let mu = mu_node(m);
                let wo = [(1.0 - mu * mu).max(0.0).sqrt(), 0.0, mu];
                let n = 48;
                let mut s = 0.0;
                for i in 0..n {
                    for j in 0..n {
                        let wi = cosine([(i as f64 + 0.5) / n as f64, (j as f64 + 0.5) / n as f64]);
                        s += charlie(r_node(r), wo, wi) * PI;
                    }
                }
                s / (n * n) as f64
            });
        Self { diel, metal, fuzz }
    }

    fn lerp2(t: &[f64], mu: f64, r: f64) -> (f64, f64) {
        let (i, a) = cell(mu);
        let (j, b) = cell(r);
        let v = |i: usize, j: usize| t[i * TN + j];
        let lo = v(i, j) * (1.0 - a) + v(i + 1, j) * a;
        let hi = v(i, j + 1) * (1.0 - a) + v(i + 1, j + 1) * a;
        (lo * (1.0 - b) + hi * b, (hi - lo) * (TN - 1) as f64)
    }

    /// `E_d(μ, r, η)` and `∂E_d/∂r`: the reflection albedo of the rough
    /// dielectric interface (untinted).
    pub fn dielectric(&self, mu: f64, r: f64, eta: f64) -> (f64, f64) {
        let s = ((eta.ln() / LN_ETA + 1.0) * 0.5).clamp(0.0, 1.0) * (NE - 1) as f64;
        let k = (s.floor() as usize).min(NE - 2);
        let c = s - k as f64;
        let slab = |k: usize| -> (f64, f64) {
            let (i, a) = cell(mu);
            let (j, b) = cell(r);
            let v = |i: usize, j: usize| self.diel[(i * TN + j) * NE + k];
            let lo = v(i, j) * (1.0 - a) + v(i + 1, j) * a;
            let hi = v(i, j + 1) * (1.0 - a) + v(i + 1, j + 1) * a;
            (lo * (1.0 - b) + hi * b, (hi - lo) * (TN - 1) as f64)
        };
        let (e0, d0) = slab(k);
        let (e1, d1) = slab(k + 1);
        (e0 * (1.0 - c) + e1 * c, d0 * (1.0 - c) + d1 * c)
    }

    /// The metal tables at `(μ, r)` and their `r`-derivatives.
    pub fn metal(&self, mu: f64, r: f64) -> [(f64, f64); 3] {
        std::array::from_fn(|k| Self::lerp2(&self.metal[k], mu, r))
    }

    /// The fuzz lobe's albedo, normalised so that its largest value over
    /// `μ` is at most 1 (Charlie with Neubelt–Pettineo's visibility
    /// reflects up to 1.4× what it receives at grazing view; the lobe is
    /// scaled down by [`Self::fuzz_norm`] wherever it does).
    pub fn fuzz(&self, mu: f64, rf: f64) -> f64 {
        Self::lerp2(&self.fuzz, mu, rf).0 / self.fuzz_norm(rf)
    }

    /// `max(1, max_μ E(μ, r_f))`: the fuzz lobe's normaliser.
    pub fn fuzz_norm(&self, rf: f64) -> f64 {
        (0..TN)
            .map(|k| Self::lerp2(&self.fuzz, k as f64 / (TN - 1) as f64, rf).0)
            .fold(1.0, f64::max)
    }
}

/// The fuzz lobe: Estevez & Kulla's "Charlie" sheen distribution with
/// Neubelt & Pettineo's visibility, `D(h) V / 1` (a BRDF, colour 1). The
/// spec recommends Zeltner+ 2022's LTC fit of an SGGX fibre layer; this is
/// the analytic stand-in (design.md, Phase 4 deviations).
pub fn charlie(rf: f64, wo: [f64; 3], wi: [f64; 3]) -> f64 {
    let (ci, co) = (wi[2], wo[2]);
    if ci <= 0.0 || co <= 0.0 {
        return 0.0;
    }
    let a = (rf * rf).max(0.07);
    let h = norm([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
    let s2 = (1.0 - h[2] * h[2]).max(0.0);
    let inv = 1.0 / a;
    let d = (2.0 + inv) * s2.powf(0.5 * inv) / (2.0 * PI);
    d / (4.0 * (ci + co - ci * co))
}

// ─── a material at one wavelength ────────────────────────────────────────

/// What a lane is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Reflection: every reflective lobe at `ω_i` above the surface.
    Reflect,
    /// Refraction through the rough dielectric (transmission lobe).
    Transmit,
    /// Entry into the subsurface walk: a dimensionless weight.
    Subsurface,
}

/// The material at one wavelength and on one side: every θ-free number a
/// lane needs, and the current `θ` as numbers for sampling.
#[derive(Debug, Clone, Copy)]
pub struct At {
    /// The material.
    pub m: OpenPbr,
    /// Wavelength, nm.
    pub lambda: f64,
    /// The three hat weights at `lambda`.
    pub b: [f64; 3],
    /// The vertex is on the inside of the surface (`ω_o` inside).
    pub inside: bool,
    /// `n_t / n_i` of the dielectric seen from `ω_o`'s side (Fresnel).
    pub eta_f: f64,
    /// The same under a full coat.
    pub eta_c: f64,
    /// The dielectric's index at `lambda`.
    pub eta_raw: f64,
    /// `n_t / n_i` for refraction (the true index, no coat or weight).
    pub etap: f64,
    /// Specular colour, transmission tint, subsurface colour, coat colour,
    /// fuzz colour at `lambda`.
    pub sc: f64,
    #[allow(missing_docs)]
    pub tc: f64,
    #[allow(missing_docs)]
    pub ssc: f64,
    #[allow(missing_docs)]
    pub cc: f64,
    #[allow(missing_docs)]
    pub fc: f64,
    /// `E_F(η_c)`.
    pub ef_coat: f64,
}

/// Number of [`super::graph::Stack`] input columns.
pub const COLUMNS: usize = 49;

/// Column indices.
#[allow(missing_docs)]
pub mod col {
    pub const KR: usize = 0;
    pub const KT: usize = 1;
    pub const KS: usize = 2;
    pub const CI: usize = 3;
    pub const CO: usize = 4;
    pub const CH: usize = 5;
    pub const OH: usize = 6;
    pub const IH: usize = 7;
    pub const B0: usize = 8;
    pub const SC: usize = 11;
    pub const FD: usize = 12;
    pub const FILM_M: usize = 13;
    pub const ES: usize = 15;
    pub const DES: usize = 16;
    pub const ES1: usize = 17;
    pub const DES1: usize = 18;
    pub const EM1: usize = 19; // 19..25: (E1, dE1, E5, dE5, Ew, dEw) at μ = 1
    pub const R0: usize = 25;
    pub const T: usize = 26;
    pub const TC: usize = 27;
    pub const S: usize = 28;
    pub const SSC: usize = 29;
    pub const F_COAT: usize = 30;
    pub const E_COAT: usize = 31;
    pub const T_COAT: usize = 32;
    pub const K_S: usize = 33;
    pub const K_R: usize = 34;
    pub const DELTA: usize = 35;
    pub const XS_FS: usize = 36;
    pub const F_FUZZ: usize = 37;
    pub const E_FUZZ: usize = 38;
    pub const ETAP: usize = 39;
    pub const DENOM2: usize = 40;
    pub const XI_S: usize = 41;
    pub const FILM_RATIO: usize = 42;
    pub const W_FILM_M: usize = 14;
    /// The coated-base copies of `FD`, `ES`, `DES`, `ES1`, `DES1`,
    /// `FILM_RATIO` (the base seen under a full coat).
    pub const FD_C: usize = 43;
    pub const ES_C: usize = 44;
    pub const DES_C: usize = 45;
    pub const ES1_C: usize = 46;
    pub const DES1_C: usize = 47;
    pub const FILM_RATIO_C: usize = 48;
    /// The uncoated set, in the same order.
    pub const UNCOATED: [usize; 6] = [FD, ES, DES, ES1, DES1, FILM_RATIO];
    /// The coated set.
    pub const COATED: [usize; 6] = [FD_C, ES_C, DES_C, ES1_C, DES1_C, FILM_RATIO_C];
}

impl At {
    /// `m` at `lambda`, seen from outside (`inside = false`) or inside.
    pub fn new(m: &OpenPbr, lambda: f64, inside: bool) -> Self {
        let eta = m.ior_at(lambda);
        // Under the coat the base sees `η / η_c` (folded to > 1: albedo
        // scaling has no refraction into the coat); the specular weight
        // then modulates the contrast. The stack evaluates the base both
        // ways and mixes by coverage, as the spec's weighted layer does.
        let rel = (eta / m.coat_ior).max(m.coat_ior / eta);
        let eu = modulated_ior(eta, m.specular_weight);
        let ec = modulated_ior(rel, m.specular_weight);
        let (eta_f, eta_c, etap) = if inside {
            (1.0 / eu, 1.0 / ec, 1.0 / eta)
        } else {
            (eu, ec, eta)
        };
        Self {
            m: *m,
            lambda,
            b: band_weights(lambda),
            inside,
            eta_f,
            eta_c,
            eta_raw: eta,
            etap,
            sc: lift(m.specular_color, lambda),
            tc: if m.transmission_depth > 0.0 {
                1.0
            } else {
                lift(m.transmission_color, lambda)
            },
            ssc: lift(m.subsurface_color, lambda),
            cc: lift(m.coat_color, lambda),
            fc: lift(m.fuzz_color, lambda),
            ef_coat: fresnel_average(m.coat_ior),
        }
    }

    /// Base colour at `lambda`.
    pub fn base(&self) -> f64 {
        lift(self.m.base_color, self.lambda)
    }

    /// The dielectric Fresnel at `c`, the film folded in (outside only).
    pub fn fd(&self, c: f64) -> f64 {
        self.fd_at(c, self.eta_f)
    }

    /// [`Self::fd`] at index `eta`.
    pub fn fd_at(&self, c: f64, eta: f64) -> f64 {
        let f = fresnel(c, eta);
        let w = self.m.thin_film_weight;
        if w <= 0.0 || self.inside {
            return f;
        }
        let film = airy(
            c,
            self.m.thin_film_ior,
            self.m.thin_film_thickness * 1e3,
            eta,
            self.lambda,
        );
        f + (film - f) * w
    }

    fn film_metal(&self, c: f64) -> f64 {
        let f0 = (self.m.base_weight * self.base()).clamp(0.0, 0.98);
        let n = (1.0 + f0.sqrt()) / (1.0 - f0.sqrt());
        airy(
            c,
            self.m.thin_film_ior,
            self.m.thin_film_thickness * 1e3,
            n,
            self.lambda,
        )
    }

    /// The lane's columns for `kind` at `(ω_o, ω_i)` (local, `ω_o.z > 0`).
    /// For [`Kind::Subsurface`] `ω_i` is ignored.
    pub fn columns(&self, kind: Kind, wo: [f64; 3], wi: [f64; 3]) -> [f64; COLUMNS] {
        use col::*;
        let t = Tables::get();
        let m = &self.m;
        let mut c = [0.0; COLUMNS];
        let co = wo[2].max(1e-9);
        let r0 = m.specular_roughness;
        let (h, ci) = match kind {
            Kind::Reflect => (norm([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]), wi[2]),
            Kind::Transmit => {
                // Generalised half vector, flipped to the `ω_o` side.
                let e = self.etap;
                let mut h = norm([
                    wo[0] + wi[0] * e,
                    wo[1] + wi[1] * e,
                    wo[2] + wi[2] * e,
                ]);
                if h[2] < 0.0 {
                    h = [-h[0], -h[1], -h[2]];
                }
                (h, -wi[2])
            }
            Kind::Subsurface => ([0.0, 0.0, 1.0], 1.0),
        };
        let oh = dot(wo, h).abs().max(1e-9);
        let ih = dot(wi, h).abs();
        c[match kind {
            Kind::Reflect => KR,
            Kind::Transmit => KT,
            Kind::Subsurface => KS,
        }] = 1.0;
        c[CI] = ci.abs().max(1e-9);
        c[CO] = co;
        c[CH] = h[2].clamp(1e-9, 1.0);
        c[OH] = oh;
        c[IH] = ih;
        c[B0..B0 + 3].copy_from_slice(&self.b);
        c[SC] = self.sc;
        let film_on = m.thin_film_weight > 0.0 && !self.inside;
        c[FILM_M] = if film_on { self.film_metal(oh) } else { 0.0 };
        c[W_FILM_M] = if film_on { m.thin_film_weight } else { 0.0 };
        // The film's change to the albedo, by its ratio at the view cosine.
        let ratio = |mu: f64, eta: f64| {
            if film_on {
                self.fd_at(mu, eta) / fresnel(mu, eta).max(1e-9)
            } else {
                1.0
            }
        };
        for (eta, k) in [(self.eta_f, UNCOATED), (self.eta_c, COATED)] {
            let (es, des) = t.dielectric(co, r0, eta);
            let (es1, des1) = t.dielectric(1.0, r0, eta);
            let v = [self.fd_at(oh, eta), es, des, es1, des1, ratio(co, eta)];
            for i in 0..6 {
                c[k[i]] = v[i];
            }
        }
        let em = t.metal(1.0, r0);
        for k in 0..3 {
            c[EM1 + 2 * k] = em[k].0;
            c[EM1 + 2 * k + 1] = em[k].1;
        }
        c[R0] = r0;
        c[T] = m.transmission_weight;
        c[TC] = self.tc;
        c[S] = m.subsurface_weight;
        c[SSC] = self.ssc;
        // The coat.
        let (ac, etac) = (alpha(m.coat_roughness), m.coat_ior);
        let etac_side = if self.inside { 1.0 / etac } else { etac };
        c[F_COAT] = if kind == Kind::Reflect {
            fresnel(oh, etac_side) * ggx_d(ac, c[CH])
                / (1.0 + ggx_lambda(ac, c[CI]) + ggx_lambda(ac, co))
                / (4.0 * c[CI] * co)
        } else {
            0.0
        };
        c[E_COAT] = t.dielectric(co, m.coat_roughness, etac_side).0;
        let mut_ = |mu: f64| (1.0 - (1.0 - mu * mu) / (etac * etac)).max(1e-6).sqrt();
        let path = if kind == Kind::Subsurface {
            2.0 / mut_(co)
        } else {
            1.0 / mut_(c[CI]) + 1.0 / mut_(co)
        };
        c[T_COAT] = self.cc.max(1e-12).powf(0.5 * path);
        c[K_S] = fresnel(co, etac_side);
        c[K_R] = 1.0 - (1.0 - self.ef_coat) / (etac * etac);
        c[DELTA] = m.coat_darkening;
        let fs = ((1.0 - self.eta_raw) / (1.0 + self.eta_raw)).powi(2);
        c[XS_FS] = (m.specular_weight * fs).clamp(0.0, 1.0);
        c[XI_S] = m.specular_weight;
        // The fuzz.
        c[F_FUZZ] = if kind == Kind::Reflect {
            self.fc * charlie(m.fuzz_roughness, wo, wi) / t.fuzz_norm(m.fuzz_roughness)
        } else {
            0.0
        };
        c[E_FUZZ] = self.fc * t.fuzz(co, m.fuzz_roughness);
        // Refraction.
        c[ETAP] = self.etap;
        let den = dot(wi, h) + dot(wo, h) / self.etap;
        c[DENOM2] = (den * den).max(1e-12);
        c
    }

    /// `f` of a lane: [`super::graph::stack`] in f64 at `θ`.
    pub fn f(&self, kind: Kind, wo: [f64; 3], wi: [f64; 3], theta: &[f64; PARAMS]) -> f64 {
        super::graph::stack(&self.columns(kind, wo, wi), theta)
    }

    /// The lobe-selection weights at `ω_o` (detached): `[spec GGX, coat
    /// GGX, cosine (diffuse + fuzz), transmission, subsurface]`.
    pub fn lobe_weights(&self, co: f64) -> [f64; 5] {
        let t = Tables::get();
        let m = &self.m;
        let r = m.specular_roughness;
        let es = t.dielectric(co, r, self.eta_f).0;
        let etac = if self.inside { 1.0 / m.coat_ior } else { m.coat_ior };
        let ec = t.dielectric(co, m.coat_roughness, etac).0;
        let ef = self.fc * t.fuzz(co, m.fuzz_roughness);
        let (fw, cw) = (m.fuzz_weight, m.coat_weight);
        let above = (1.0 - fw * ef).max(0.0) * (1.0 - cw * ec).max(0.0);
        let mt = t.metal(co, r);
        let f0 = m.base_weight * self.base();
        let em = (f0 * (mt[0].0 - mt[1].0) + mt[1].0).max(0.0);
        let met = m.base_metalness;
        let diel = 1.0 - met;
        let (tw, sw) = (m.transmission_weight, m.subsurface_weight);
        let spec = above * (met * em + diel * self.sc * es);
        let diff = above * diel * (1.0 - tw) * (1.0 - sw) * (1.0 - self.sc * es).max(0.0)
            * m.base_weight * self.base();
        let trans = above * diel * tw * (1.0 - es) * self.tc;
        let sss = above * diel * (1.0 - tw) * sw * (1.0 - self.sc * es).max(0.0);
        let mut w = [
            spec + 1e-3,
            cw * ec,
            diff + fw * ef,
            trans,
            if self.inside { 0.0 } else { sss },
        ];
        if self.inside {
            // From inside only the dielectric interface matters for sampling.
            w = [w[0] + w[2], 0.0, 0.0, w[3], 0.0];
        }
        let s: f64 = w.iter().sum();
        w.map(|x| x / s)
    }

    /// The directional pdf of `ω_i` under the lobe mixture (solid angle),
    /// with the subsurface mass not counted (it is a discrete event).
    pub fn pdf(&self, w: &[f64; 5], wo: [f64; 3], wi: [f64; 3]) -> f64 {
        let m = &self.m;
        let co = wo[2];
        if wi[2] > 0.0 {
            let h = norm([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
            let oh = dot(wo, h);
            let jac = 1.0 / (4.0 * oh.abs().max(1e-12));
            let ps = vndf_pdf(alpha(m.specular_roughness), co, h[2], oh) * jac;
            let pc = vndf_pdf(alpha(m.coat_roughness), co, h[2], oh) * jac;
            w[0] * ps + w[1] * pc + w[2] * wi[2] / PI
        } else {
            let e = self.etap;
            let mut h = norm([wo[0] + wi[0] * e, wo[1] + wi[1] * e, wo[2] + wi[2] * e]);
            if h[2] < 0.0 {
                h = [-h[0], -h[1], -h[2]];
            }
            let (oh, ih) = (dot(wo, h), dot(wi, h));
            if oh <= 0.0 || ih >= 0.0 {
                return 0.0;
            }
            let den = ih + oh / e;
            w[3] * vndf_pdf(alpha(m.specular_roughness), co, h[2], oh) * ih.abs() / (den * den)
        }
    }

    /// Sample a lobe and a direction (detached). Returns `None` for the
    /// subsurface event (the caller enters the walk), else `Some(ω_i)`
    /// (possibly invalid, `ω_i = 0`, on TIR).
    pub fn sample(&self, w: &[f64; 5], wo: [f64; 3], u: [f64; 3]) -> Option<[f64; 3]> {
        let m = &self.m;
        let mut acc = 0.0;
        let mut k = 0;
        for (i, &wi) in w.iter().enumerate() {
            acc += wi;
            k = i;
            if u[0] < acc {
                break;
            }
        }
        let uv = [u[1], u[2]];
        match k {
            0 => Some(reflect(wo, vndf(alpha(m.specular_roughness), wo, uv))),
            1 => Some(reflect(wo, vndf(alpha(m.coat_roughness), wo, uv))),
            2 => Some(cosine(uv)),
            3 => {
                let h = vndf(alpha(m.specular_roughness), wo, uv);
                Some(refract(wo, h, self.etap).unwrap_or([0.0; 3]))
            }
            _ => None,
        }
    }
}

/// OpenPBR subsurface: the medium's single-scattering albedo from the
/// subsurface colour (van de Hulst's inversion, spec "Subsurface"), and
/// `μ_t = 1 / (radius · radius_scale)`.
pub fn subsurface_medium(m: &OpenPbr, lambda: f64) -> (f64, f64) {
    let c = lift(m.subsurface_color, lambda).clamp(1e-4, 0.999);
    let g = m.subsurface_scatter_anisotropy;
    let s = 4.09712 + 4.20863 * c - (9.59217 + 41.6808 * c + 17.7126 * c * c).sqrt();
    let s2 = s * s;
    let albedo = ((1.0 - s2) / (1.0 - g * s2)).clamp(0.0, 0.999_999);
    let r = (m.subsurface_radius * lift(m.subsurface_radius_scale, lambda)).max(1e-6);
    let mt = 1.0 / r;
    (mt * (1.0 - albedo), mt * albedo)
}
