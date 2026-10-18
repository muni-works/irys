//! Skin's optical constants per wavelength, the albedo of a semi-infinite
//! random-walk medium, and the inversion from an albedo map to chromophores.
//!
//! # The medium
//!
//! One homogeneous layer per entry point, coefficients in mm⁻¹:
//!
//! ```text
//! σa(λ) = s · (m · μa_mel(λ) + h · μa_blood(λ) + μa_base(λ))
//! σs(λ) = s · μs'(λ) / (1 − g)
//! ```
//!
//! - `μa_mel = 51.9 (λ/500)^−3.5` mm⁻¹, eumelanin in a melanosome (Jacques,
//!   "Optical properties of biological tissues: a review", PMB 2013, eq. 6);
//! - `μa_blood = ln 10 · 150 g/L / 64 500 g/mol · (S ε_HbO₂ + (1 − S) ε_Hb)`
//!   with `S = 0.75` and Prahl's molar extinction table (omlc.org, compiled
//!   from Gratzer and Kollias), whole blood at 150 g/L;
//! - `μa_base = 0.0244 + 8.53 e^{−(λ − 154)/66.2}` mm⁻¹, bloodless tissue
//!   (Jacques 1998);
//! - `μs' = 4.60 (f_R (λ/500)^−4 + (1 − f_R)(λ/500)^−0.562)` mm⁻¹ with
//!   `f_R = 0.409`, the average-skin fit of Jacques 2013 (table 2), and
//!   Henyey–Greenstein `g = 0.8`, so `σs = μs'/0.2`.
//!
//! `m` and `h` are bulk volume fractions of melanosomes and blood, `s` a
//! density scale (1 is the measured tissue). This is a one-layer
//! simplification of Donner & Jensen's two-layer epidermis/dermis model:
//! melanin's thin layer is folded into the bulk, which is what the inversion
//! can identify from a colour anyway (see design.md, "Skin").
//!
//! # The albedo of the walk, exactly
//!
//! For isotropic scattering and an index-matched boundary, the reflectance
//! of a semi-infinite medium of single-scattering albedo `ϖ` under light
//! arriving at `μ₀ = cos θ` is `R(μ₀) = 1 − H(μ₀) √(1 − ϖ)`, with
//! Chandrasekhar's `H` function (Radiative Transfer, 1960, ch. V). The walk
//! enters cosine-distributed, so its total albedo is the hemispherical
//! average `A(ϖ) = 1 − 2 √(1 − ϖ) ∫ H(μ) μ dμ`. [`h_function`] solves the
//! integral equation; tests hold the random walk to it. With `g ≠ 0` the
//! similarity relation `ϖ* = ϖ (1 − g)/(1 − ϖ g)` maps to the isotropic
//! albedo; that is an approximation, and the test says how good.

use std::sync::OnceLock;

/// Henyey–Greenstein asymmetry of the skin medium.
pub const G: f64 = 0.8;

/// Oxygen saturation of the blood.
pub const SATURATION: f64 = 0.75;

/// Skin's refractive index, and the Schlick `F0` it implies.
pub const ETA: f64 = 1.4;

/// `((η − 1)/(η + 1))²`.
pub const F0: f64 = (ETA - 1.0) * (ETA - 1.0) / ((ETA + 1.0) * (ETA + 1.0));

/// Schlick's Fresnel at cosine `c` for [`F0`].
pub fn schlick(c: f64) -> f64 {
    let m = (1.0 - c.clamp(0.0, 1.0)).powi(5);
    F0 + (1.0 - F0) * m
}

/// The cosine-weighted hemispherical mean of [`schlick`]:
/// `F0 + (1 − F0)/21`.
pub const F_MEAN: f64 = F0 + (1.0 - F0) / 21.0;

/// Prahl's molar extinction coefficients (cm⁻¹ M⁻¹): `(λ nm, HbO₂, Hb)`.
const PRAHL: [(f64, f64, f64); 35] = [
    (380.0, 125_000.0, 102_000.0),
    (390.0, 166_000.0, 135_000.0),
    (400.0, 266_232.0, 223_296.0),
    (410.0, 466_816.0, 303_956.0),
    (420.0, 480_360.0, 407_560.0),
    (430.0, 386_240.0, 528_600.0),
    (440.0, 102_580.0, 413_280.0),
    (450.0, 62_816.0, 103_292.0),
    (460.0, 44_480.0, 23_388.8),
    (470.0, 33_209.2, 16_156.4),
    (480.0, 26_629.2, 14_550.0),
    (490.0, 23_684.4, 16_684.0),
    (500.0, 20_932.8, 20_035.2),
    (510.0, 20_035.2, 25_773.6),
    (520.0, 24_202.4, 31_589.6),
    (530.0, 39_956.8, 39_036.4),
    (540.0, 53_236.0, 46_592.0),
    (550.0, 43_016.0, 53_412.0),
    (560.0, 32_613.2, 53_788.0),
    (570.0, 44_496.0, 45_072.0),
    (580.0, 50_104.0, 37_020.0),
    (590.0, 14_400.8, 28_324.4),
    (600.0, 3_200.0, 14_677.2),
    (610.0, 1_506.0, 9_443.6),
    (620.0, 942.0, 6_509.6),
    (630.0, 610.0, 5_148.8),
    (640.0, 442.0, 4_345.2),
    (650.0, 368.0, 3_750.12),
    (660.0, 319.6, 3_226.56),
    (670.0, 294.0, 2_795.12),
    (680.0, 277.6, 2_407.92),
    (690.0, 276.0, 2_051.96),
    (700.0, 290.0, 1_794.28),
    (720.0, 348.0, 1_325.88),
    (780.0, 710.0, 760.0),
];

/// Melanosome absorption, mm⁻¹.
pub fn mu_a_melanin(lambda: f64) -> f64 {
    51.9 * (lambda / 500.0).powf(-3.5)
}

/// Whole-blood absorption at [`SATURATION`], mm⁻¹.
pub fn mu_a_blood(lambda: f64) -> f64 {
    let i = PRAHL
        .windows(2)
        .position(|w| lambda <= w[1].0)
        .unwrap_or(PRAHL.len() - 2);
    let (a, b) = (PRAHL[i], PRAHL[i + 1]);
    let t = ((lambda - a.0) / (b.0 - a.0)).clamp(0.0, 1.0);
    let oxy = a.1 + (b.1 - a.1) * t;
    let de = a.2 + (b.2 - a.2) * t;
    let eps = SATURATION * oxy + (1.0 - SATURATION) * de;
    // cm⁻¹ → mm⁻¹.
    std::f64::consts::LN_10 * eps * 150.0 / 64_500.0 / 10.0
}

/// Bloodless-tissue absorption, mm⁻¹.
pub fn mu_a_base(lambda: f64) -> f64 {
    0.0244 + 8.53 * (-(lambda - 154.0) / 66.2).exp()
}

/// Reduced scattering, mm⁻¹.
pub fn mu_s_reduced(lambda: f64) -> f64 {
    let x = lambda / 500.0;
    4.60 * (0.409 * x.powi(-4) + 0.591 * x.powf(-0.562))
}

/// Shortest wavelength in [`table`].
pub const TABLE_MIN: f64 = 380.0;
/// Entries in [`table`], one per nm.
pub const TABLE_LEN: usize = 401;

/// Per-nm coefficients `[μa_mel, μa_blood, μa_base, σs]` (mm⁻¹, `σs` already
/// divided by `1 − g`), 380 … 780 nm. CPU and GPU both interpolate this one
/// table, so they agree on the medium to f32 rounding.
pub fn table() -> &'static [[f64; 4]] {
    static T: OnceLock<Vec<[f64; 4]>> = OnceLock::new();
    T.get_or_init(|| {
        (0..TABLE_LEN)
            .map(|i| {
                let l = TABLE_MIN + i as f64;
                [
                    mu_a_melanin(l),
                    mu_a_blood(l),
                    mu_a_base(l),
                    mu_s_reduced(l) / (1.0 - G),
                ]
            })
            .collect()
    })
}

/// [`table`] at `lambda`, linearly interpolated.
pub fn coefficients(lambda: f64) -> [f64; 4] {
    let t = table();
    let x = (lambda - TABLE_MIN).clamp(0.0, (TABLE_LEN - 1) as f64);
    let i = (x.floor() as usize).min(TABLE_LEN - 2);
    let f = x - i as f64;
    std::array::from_fn(|k| t[i][k] * (1.0 - f) + t[i + 1][k] * f)
}

/// `(σa, σs)` in mm⁻¹ at `lambda` for melanin `m`, blood `h`, scale `s`.
pub fn sigma(lambda: f64, m: f64, h: f64, s: f64) -> (f64, f64) {
    let c = coefficients(lambda);
    (s * (m * c[0] + h * c[1] + c[2]), s * c[3])
}

/// Gauss–Legendre nodes and weights on `[0, 1]`.
fn gauss(n: usize) -> Vec<(f64, f64)> {
    // Newton on P_n, Golub–Welsch would be overkill.
    (0..n)
        .map(|i| {
            let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
            let mut dp = 0.0;
            for _ in 0..100 {
                let (mut p0, mut p1) = (1.0, x);
                for k in 2..=n {
                    let p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                    p0 = p1;
                    p1 = p2;
                }
                dp = n as f64 * (x * p1 - p0) / (x * x - 1.0);
                let dx = p1 / dp;
                x -= dx;
                if dx.abs() < 1e-15 {
                    break;
                }
            }
            let w = 2.0 / ((1.0 - x * x) * dp * dp);
            (0.5 * (x + 1.0), 0.5 * w)
        })
        .collect()
}

/// Chandrasekhar's `H(μ)` for isotropic scattering at albedo `w < 1`, at the
/// nodes of an `n`-point Gauss–Legendre rule on `[0, 1]`: `(μ_i, w_i, H_i)`.
/// Solved by iterating `1/H(μ) = √(1 − ϖ) + (ϖ/2) ∫ μ' H(μ')/(μ + μ') dμ'`.
pub fn h_function(w: f64, n: usize) -> Vec<(f64, f64, f64)> {
    let q = gauss(n);
    let mut h = vec![1.0; n];
    let root = (1.0 - w).max(0.0).sqrt();
    for _ in 0..2000 {
        let next: Vec<f64> = q
            .iter()
            .map(|&(mu, _)| {
                let s: f64 = q
                    .iter()
                    .zip(&h)
                    .map(|(&(m2, w2), &h2)| w2 * m2 * h2 / (mu + m2))
                    .sum();
                1.0 / (root + 0.5 * w * s)
            })
            .collect();
        let d = next
            .iter()
            .zip(&h)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        h = next;
        if d < 1e-14 {
            break;
        }
    }
    q.iter().zip(h).map(|(&(m, w), h)| (m, w, h)).collect()
}

/// `H(μ)` at any `μ`, from the converged nodes (one more application of the
/// integral equation — Nyström interpolation).
pub fn h_at(w: f64, nodes: &[(f64, f64, f64)], mu: f64) -> f64 {
    let s: f64 = nodes.iter().map(|&(m, wq, h)| wq * m * h / (mu + m)).sum();
    1.0 / ((1.0 - w).max(0.0).sqrt() + 0.5 * w * s)
}

/// Semi-infinite, index-matched, isotropic: reflectance under a collimated
/// beam at `μ₀`, `1 − H(μ₀) √(1 − ϖ)`.
pub fn albedo_collimated(w: f64, mu0: f64) -> f64 {
    let nodes = h_function(w, 48);
    1.0 - h_at(w, &nodes, mu0) * (1.0 - w).sqrt()
}

/// Same medium under cosine-distributed light: `1 − 2 √(1 − ϖ) ∫ H μ dμ`.
pub fn albedo_diffuse(w: f64) -> f64 {
    let nodes = h_function(w, 48);
    let h1: f64 = nodes.iter().map(|&(m, wq, h)| wq * m * h).sum();
    1.0 - 2.0 * (1.0 - w).sqrt() * h1
}

/// The similarity relation: the isotropic albedo equivalent to `w` at
/// asymmetry `g`.
pub fn similar(w: f64, g: f64) -> f64 {
    w * (1.0 - g) / (1.0 - w * g)
}

/// [`albedo_diffuse`] of the similarity-mapped albedo, tabulated on a
/// grid fine in `1 − ϖ*` and interpolated, so the inversion can call it a
/// million times.
pub fn walk_albedo(w_star: f64) -> f64 {
    static T: OnceLock<Vec<(f64, f64)>> = OnceLock::new();
    let t = T.get_or_init(|| {
        // x = √(1 − ϖ*) ∈ [0, 1], where A is smooth.
        (0..=256)
            .map(|i| {
                let x = i as f64 / 256.0;
                let w = 1.0 - x * x;
                (x, if i == 0 { 1.0 } else { albedo_diffuse(w) })
            })
            .collect()
    });
    let x = (1.0 - w_star).clamp(0.0, 1.0).sqrt() * 256.0;
    let i = (x.floor() as usize).min(255);
    let f = x - i as f64;
    t[i].1 * (1.0 - f) + t[i + 1].1 * f
}

/// The diffuse colour (linear sRGB) a semi-infinite slab of chromophores
/// `(m, h)` shows under the walk's boundary model: `(1 − F̄) A(ϖ*(λ))`
/// integrated against the sRGB response.
pub fn slab_rgb(m: f64, h: f64) -> [f64; 3] {
    const N: usize = 80;
    let mut c = [0.0; 3];
    for i in 0..N {
        let l = TABLE_MIN + (i as f64 + 0.5) * 400.0 / N as f64;
        let (sa, ss) = sigma(l, m, h, 1.0);
        let a = (1.0 - F_MEAN) * walk_albedo(similar(ss / (sa + ss), G));
        let wgt = irys_light::spectrum::hero_weight(l);
        for k in 0..3 {
            c[k] += a * wgt[k] as f64 / N as f64;
        }
    }
    c
}

/// Melanin fractions the inversion searches.
pub const MEL_MAX: f64 = 0.25;
/// Blood fractions the inversion searches.
pub const HB_MAX: f64 = 0.12;

/// Chromophores for an albedo map: per texel the `(m, h)` whose
/// [`slab_rgb`] is nearest (in log colour) to the texel's. Solved once per
/// cell of a 32³ grid over linear RGB and looked up nearest, so the cost is
/// the grid's. Returns the map (`[m, h, err]`, `err` the RGB distance) and
/// the mean error.
pub fn invert(albedo: &crate::skin::mesh::Texture) -> (Vec<[f32; 3]>, f64) {
    use rayon::prelude::*;
    const C: usize = 48;
    const Q: usize = 32;
    // Candidates on a grid quadratic in each axis (fine near zero).
    let cands: Vec<(f64, f64, [f64; 3])> = (0..C * C)
        .into_par_iter()
        .map(|i| {
            let (a, b) = (
                (i % C) as f64 / (C - 1) as f64,
                (i / C) as f64 / (C - 1) as f64,
            );
            let (m, h) = (MEL_MAX * a * a, HB_MAX * b * b);
            (m, h, slab_rgb(m, h))
        })
        .collect();
    let lg = |c: [f64; 3]| c.map(|x| (x.max(1e-4)).ln());
    let best = |rgb: [f64; 3]| {
        let t = lg(rgb);
        cands
            .iter()
            .map(|&(m, h, c)| {
                let l = lg(c);
                let d = (0..3).map(|k| (l[k] - t[k]).powi(2)).sum::<f64>();
                (d, m, h, c)
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .unwrap()
    };
    let cell = |x: f64| ((x.clamp(0.0, 1.0).sqrt() * (Q - 1) as f64).round() as usize).min(Q - 1);
    let grid: Vec<(f64, f64, [f64; 3])> = (0..Q * Q * Q)
        .into_par_iter()
        .map(|i| {
            let f = |k: usize| {
                let v = k as f64 / (Q - 1) as f64;
                v * v
            };
            let rgb = [f(i % Q), f((i / Q) % Q), f(i / (Q * Q))];
            let (_, m, h, c) = best(rgb);
            (m, h, c)
        })
        .collect();
    let out: Vec<[f32; 3]> = albedo
        .texels
        .par_iter()
        .map(|t| {
            let rgb = t.map(|x| x as f64);
            let (m, h, c) = grid[cell(rgb[0]) + Q * cell(rgb[1]) + Q * Q * cell(rgb[2])];
            let err = (0..3).map(|k| (c[k] - rgb[k]).powi(2)).sum::<f64>().sqrt();
            [m as f32, h as f32, err as f32]
        })
        .collect();
    let mean = out.iter().map(|t| t[2] as f64).sum::<f64>() / out.len() as f64;
    (out, mean)
}
