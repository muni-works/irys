//! Physically-based unidirectional path tracer.
//!
//! Where a rasteriser evaluates a hand-tuned lighting rig at the primary hit
//! and stops, this solves the rendering equation by Monte Carlo integration:
//! multiple bounces, importance-sampled microfacet lobes, multiple importance
//! sampling against explicit area lights, and a physical camera with a real
//! aperture.
//!
//! It knows nothing about what it is tracing. Geometry arrives as a
//! [`Geometry`] implementation behind a [`Bvh`], so a B-rep's analytic faces
//! are traced exactly — curved silhouettes and specular highlights on fillets
//! correct at any resolution, no tessellation anywhere — and a triangle soup
//! or a splat cloud goes through the same integrator unchanged.
//!
//! # Design
//!
//! - **Lights** are intersectable rectangles ("softboxes"). Because they can
//!   be hit by a BSDF ray *and* sampled directly, both strategies combine
//!   under MIS with the power heuristic. That is what puts crisp, correctly
//!   shaped highlights on metal.
//! - **The environment** is, by default, a smooth analytic studio gradient.
//!   It is low-frequency by construction, so BSDF sampling alone converges
//!   quickly and no environment CDF is needed. An opt-in lat-long HDR image
//!   ([`EnvMap`]) is also supported; because a real HDRI carries windows and
//!   sun discs, that variant builds a `sin(theta)`-weighted 2D CDF and joins
//!   the MIS mix as a third sampling strategy.
//! - **The BSDF** is a layered metallic-roughness model: Lambert diffuse,
//!   a GGX specular lobe with VNDF sampling, and a GGX clearcoat lobe.
//!   Clearcoat is what sells anodised aluminium and moulded plastic.

use std::sync::Arc;

use crate::bvh::Bvh;
use crate::caustics::CausticMap;
use crate::geometry::Geometry;
use crate::math::{Aabb, Point3, Transform, Vec3};
use crate::ray::Ray;
use crate::splats::{SplatSegment, Splats};
use crate::tlas::{Instance, InstanceHit, Tlas};

/// Rows of the film, in parallel where there are threads to do it with.
///
/// The browser has no `rayon`, and a renderer that cannot run where the
/// picture is looked at is half a renderer — so the row split is a macro and
/// the two spellings sit behind one `cfg`. Every adaptor used downstream
/// (`zip`, `enumerate`, `skip`, `take`, `for_each`) exists on both.
#[cfg(not(target_arch = "wasm32"))]
macro_rules! film_rows {
    ($buf:expr, $n:expr) => {
        $buf.par_chunks_mut($n)
    };
}

#[cfg(target_arch = "wasm32")]
macro_rules! film_rows {
    ($buf:expr, $n:expr) => {
        $buf.chunks_mut($n)
    };
}

// ─── material ─────────────────────────────────────────────────────────────

/// A physically-based surface description.
///
/// Disney's "principled" parameterisation (Burley 2012) with the corrections
/// the field has settled on since, composed the way OpenPBR 1.0 composes them:
///
/// - **diffuse** — energy-preserving Oren-Nayar (d'Eon & Portsmouth et al.
///   2024), driven by [`Self::diffuse_roughness`] (OpenPBR's
///   `base_diffuse_roughness`), blended with Disney's Hanrahan-Krueger
///   [`Self::subsurface`] lobe;
/// - **specular** — anisotropic GGX with VNDF sampling and Turquin (2019)
///   multiple-scattering compensation;
/// - **sheen** — the multiple-scattering LTC fit of Zeltner, Burley and Chiang
///   (2022) (OpenPBR calls this layer *fuzz*);
/// - **coat** — Disney's GTR1 clearcoat at a fixed IOR of 1.5.
///
/// Every field added on top of the original metallic-roughness set defaults to
/// a value that reduces the model to what it was: `diffuse_roughness = 0` is
/// exactly Lambert, `subsurface = 0` and `sheen = 0` switch their lobes off
/// entirely, and `specular = 0.5` with `ior = 1.5` is the same `F0 = 0.04` the
/// IOR alone used to give.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pbr {
    /// Linear-space base colour. Albedo for dielectrics, F0 for metals.
    pub base_color: [f32; 3],
    /// 0 = dielectric, 1 = metal.
    pub metallic: f32,
    /// Perceptual roughness in 0..1. Squared internally to get the GGX alpha.
    pub roughness: f32,
    /// Roughness of the *diffuse* lobe — OpenPBR's `base_diffuse_roughness`,
    /// the `sigma` of Oren-Nayar.
    ///
    /// Kept separate from [`Self::roughness`] because a surface's microscale
    /// slope statistics and its subsurface scattering length are unrelated
    /// facts: latex paint is diffusely rough and specularly smooth, a polished
    /// marble is the reverse. `0` is exactly Lambert, which is the default and
    /// why every scene that predates this field renders unchanged.
    pub diffuse_roughness: f32,
    /// Weight of the subsurface lobe in 0..1 — OpenPBR's `subsurface_weight`.
    ///
    /// This is not Disney's Hanrahan-Krueger *blend*, which flattened the
    /// diffuse falloff to imitate the look of scattering without any of the
    /// transport. It is the real thing: the weight takes that fraction of the
    /// diffuse lobe away and replaces it with a **random walk inside the
    /// object** (Chiang, Kutz and Burley, "Practical and Controllable
    /// Subsurface Scattering for Production Path Tracing", SIGGRAPH 2016).
    /// Light enters at the shading point, scatters through the medium and
    /// leaves somewhere *else* on the surface, which is what makes an ear
    /// glow, a marble read as stone rather than as paint, and a rubber ball
    /// look moulded rather than sprayed.
    ///
    /// `0` is the default and switches the walk off outright.
    pub subsurface: f32,
    /// The colour the walk is asked to produce — OpenPBR's
    /// `subsurface_color`, the *surface* albedo and not the medium's.
    ///
    /// A medium's single-scattering albedo and the diffuse reflectance a slab
    /// of it shows are very different numbers: 0.9 in the medium is nearly
    /// white at the surface. Chiang's contribution is the inversion, so this
    /// is the number a person actually wants to pick — what the material
    /// looks like — and the renderer solves for the medium that produces it.
    pub subsurface_color: [f32; 3],
    /// Mean free path inside the medium, per channel, in scene units —
    /// OpenPBR's `subsurface_radius`.
    ///
    /// How far light travels between scattering events, so it sets the
    /// *scale* of the effect against the object: a 2 mm path in a basketball
    /// is a soft sheen under the surface, the same 2 mm in a marble
    /// statuette is a glow through a thin edge. Per channel because red
    /// travels furthest through most organic media, which is the whole
    /// reason a hand held to a light goes red at the edges.
    pub subsurface_radius: [f64; 3],
    /// Incident specular amount in Disney's normalised range — `0.5` means
    /// `F0 = 0.04`. See [`Self::f0`] for how it and [`Self::ior`] combine.
    pub specular: f32,
    /// Tints the dielectric `F0` towards the hue of the base colour, 0..1.
    ///
    /// Disney's concession to art direction: grazing specular stays achromatic
    /// either way, so this only colours the highlight's core.
    pub specular_tint: f32,
    /// Strength of the sheen (OpenPBR: *fuzz*) layer, 0 = none.
    pub sheen: f32,
    /// Colour of the sheen layer — OpenPBR's `fuzz_color`.
    ///
    /// Disney 2012 spelled this as a scalar `sheenTint` interpolating towards
    /// the base hue; OpenPBR gives the layer its own colour outright, which is
    /// strictly more expressive, so that is what this is. `[1, 1, 1]` is
    /// white sheen.
    pub sheen_color: [f32; 3],
    /// Roughness of the sheen layer — OpenPBR's `fuzz_roughness`, and the
    /// `alpha` axis of the Zeltner LTC fit.
    pub sheen_roughness: f32,
    /// Directional bias of the specular lobe, in -1..1.
    ///
    /// `0` is isotropic and reduces exactly to a round GGX highlight.
    /// Positive values stretch the highlight *along* the surface tangent
    /// (`dP/du`), negative values stretch it across. Because vcad shades the
    /// analytic BRep, that tangent is the real parameterisation of the
    /// surface — the circumferential direction on a cylinder — so a turned
    /// shaft or a bored hole gets the smeared highlight it has in life
    /// without any generated tangents or texture.
    pub anisotropy: f32,
    /// Strength of the clearcoat layer (0 = none, 1 = full).
    pub clearcoat: f32,
    /// Perceptual roughness of the clearcoat layer.
    ///
    /// Disney's parameter is `clearcoatGloss`, mapped onto the GTR1 alpha as
    /// `alpha = mix(0.1, 0.001, gloss)`. This is the same layer spelled the
    /// way the rest of this struct spells roughness — `alpha = roughness²`,
    /// with `clearcoat_roughness ≈ sqrt(mix(0.1, 0.001, gloss))`, so Disney's
    /// satin end (gloss 0) is `clearcoat_roughness ≈ 0.32` and its gloss end
    /// (gloss 1) is `≈ 0.032`.
    pub clearcoat_roughness: f32,
    /// Dielectric index of refraction. See [`Self::f0`] for how it and
    /// [`Self::specular`] combine.
    pub ior: f32,
    /// Weight of the dielectric transmission lobe, 0..1 — OpenPBR's
    /// `transmission_weight`.
    ///
    /// `0` is an opaque surface and the model is exactly what it was. `1` is
    /// glass: the diffuse and opaque-specular lobes are switched off entirely
    /// and replaced by one rough dielectric that both reflects and refracts,
    /// split by the *exact* Fresnel equations at [`Self::ior`] rather than by
    /// Schlick's fit — because at a glass/air interface Schlick's error near
    /// grazing is the difference between a rim that glows and one that does
    /// not, and because total internal reflection has to fall out of the same
    /// formula that gave the split.
    pub transmission: f32,
    /// Abbe number `V_d = (n_d − 1)/(n_F − n_C)`, OpenPBR's dispersion knob.
    /// `0` means no dispersion, which is the default and why nothing that
    /// predates this field renders spectrally.
    ///
    /// Smaller is *more* dispersive: crown glass is about 60, dense flint
    /// about 30, diamond 55 at a much higher index. Ignored when
    /// [`Self::sellmeier`] is set.
    pub abbe: f32,
    /// A real glass's Sellmeier coefficients `(B, C)`, overriding
    /// [`Self::abbe`] when present.
    ///
    /// An Abbe number is one number fitted through three Fraunhofer lines; a
    /// Sellmeier triple pair *is* the datasheet. [`crate::spectrum::BK7_SELLMEIER`]
    /// is the one the marble's own caustic tracer uses, so a material given
    /// that pair and the caustic tracer disperse by the same curve.
    pub sellmeier: Option<([f64; 3], [f64; 3])>,
    /// Colour transmitted through one [`Self::attenuation_distance`] of the
    /// interior — glTF's `attenuationColor`, Beer–Lambert's `exp(−σd)` written
    /// the way a person picks a colour.
    pub attenuation_color: [f32; 3],
    /// Distance over which the interior attenuates to
    /// [`Self::attenuation_color`]. Infinite (the default) is no absorption.
    pub attenuation_distance: f32,
    /// Treat the surface as an infinitely thin sheet rather than the boundary
    /// of a volume.
    ///
    /// A window pane modelled as a single quad, or as a box far thinner than
    /// its own refraction would be visible at, has no interior for a ray to
    /// travel through. Thin-walled transmission refracts in and straight back
    /// out: the direction is the incident one mirrored through the surface,
    /// roughened by the same GGX lobe, with no lateral offset and no
    /// absorption — which is what a pane of glass actually looks like.
    pub thin_walled: bool,
    /// Thickness of a thin film over the surface, in nanometres. `0` — the
    /// default — is no film and the specular lobe's Fresnel is exactly what
    /// it was. OpenPBR's `thin_film_thickness`.
    ///
    /// A few hundred nanometres is where the interference falls in the
    /// visible band: a soap bubble runs 100–1000 nm, the oxide on tempered
    /// steel 20–80 nm, an anti-reflection coating a quarter of a wavelength.
    /// The colour is a function of thickness *and* angle, which is why an
    /// iridescent surface shifts hue as it turns and a tinted one does not.
    pub thin_film_thickness: f32,
    /// Index of refraction of that film — OpenPBR's `thin_film_ior`.
    ///
    /// It sits between the outside (1.0) and the substrate, and its contrast
    /// against both is what sets how strong the interference is. 1.5 is a
    /// generic oil or lacquer; 1.34 is a soap film.
    pub thin_film_ior: f32,
    /// Linear emissive radiance.
    pub emissive: [f32; 3],
}

impl Default for Pbr {
    fn default() -> Self {
        Self {
            base_color: [0.62, 0.64, 0.67],
            metallic: 0.0,
            roughness: 0.4,
            diffuse_roughness: 0.0,
            subsurface: 0.0,
            subsurface_color: [1.0; 3],
            subsurface_radius: [1.0; 3],
            specular: 0.5,
            specular_tint: 0.0,
            sheen: 0.0,
            sheen_color: [1.0; 3],
            sheen_roughness: 0.3,
            anisotropy: 0.0,
            clearcoat: 0.0,
            clearcoat_roughness: 0.1,
            ior: 1.5,
            transmission: 0.0,
            abbe: 0.0,
            sellmeier: None,
            attenuation_color: [1.0; 3],
            attenuation_distance: f32::INFINITY,
            thin_walled: false,
            thin_film_thickness: 0.0,
            thin_film_ior: 1.5,
            emissive: [0.0; 3],
        }
    }
}

impl Pbr {
    /// A metal: base colour is the specular reflectance at normal incidence.
    pub fn metal(base_color: [f32; 3], roughness: f32) -> Self {
        Self {
            base_color,
            metallic: 1.0,
            roughness,
            ..Default::default()
        }
    }

    /// A dielectric with an optional clearcoat layer.
    pub fn plastic(base_color: [f32; 3], roughness: f32, clearcoat: f32) -> Self {
        Self {
            base_color,
            metallic: 0.0,
            roughness,
            clearcoat,
            ..Default::default()
        }
    }

    /// A brushed or turned metal: the specular lobe is stretched along the
    /// surface's own tangent direction.
    ///
    /// `anisotropy` is signed — positive smears the highlight along `dP/du`
    /// (circumferentially on a cylinder, which is a turned finish), negative
    /// smears it across (an axially-brushed one).
    pub fn brushed_metal(base_color: [f32; 3], roughness: f32, anisotropy: f32) -> Self {
        Self {
            base_color,
            metallic: 1.0,
            roughness,
            anisotropy: anisotropy.clamp(-1.0, 1.0),
            ..Default::default()
        }
    }

    /// Clear glass: a full-strength dielectric transmission lobe at the given
    /// index and roughness, with no dispersion and no absorption.
    pub fn glass(ior: f32, roughness: f32) -> Self {
        Self {
            base_color: [1.0; 3],
            metallic: 0.0,
            roughness,
            ior,
            transmission: 1.0,
            ..Default::default()
        }
    }

    /// Give this material a real glass's dispersion curve.
    pub fn with_sellmeier(mut self, coeffs: ([f64; 3], [f64; 3])) -> Self {
        self.sellmeier = Some(coeffs);
        self
    }

    /// Give this material Beer–Lambert absorption: the colour one
    /// `distance` of interior transmits.
    pub fn with_attenuation(mut self, color: [f32; 3], distance: f32) -> Self {
        self.attenuation_color = color;
        self.attenuation_distance = distance;
        self
    }

    /// Whether this material's index varies with wavelength — the test that
    /// decides whether a path has to become monochromatic.
    ///
    /// A material with no transmission has no index to disperse *through*: its
    /// specular reflectance varies with `n` far too weakly to be worth a
    /// spectral path, and OpenPBR ties dispersion to the transmission lobe for
    /// the same reason.
    #[inline]
    pub fn is_dispersive(&self) -> bool {
        self.transmission > 0.0 && (self.sellmeier.is_some() || self.abbe > 0.0)
    }

    /// Index of refraction at a wavelength, in nanometres.
    ///
    /// `None` — an RGB path — gets [`Self::ior`] flat, which is exactly what
    /// every non-dispersive material has always used.
    #[inline]
    pub fn index_at(&self, lambda_nm: Option<f64>) -> f32 {
        let Some(nm) = lambda_nm else {
            return self.ior;
        };
        let um = nm * 1e-3;
        if let Some((b, c)) = self.sellmeier {
            crate::spectrum::sellmeier_index(b, c, um) as f32
        } else if self.abbe > 0.0 {
            crate::spectrum::cauchy_index(self.ior as f64, self.abbe as f64, um) as f32
        } else {
            self.ior
        }
    }

    /// Beer–Lambert extinction per unit length: `ln(1/attenuation_color) /
    /// attenuation_distance`, per channel.
    ///
    /// Zero whenever the distance is infinite or the colour is white, so the
    /// interior of a default material costs an `is_finite` check and nothing
    /// else.
    #[inline]
    pub fn extinction(&self) -> [f32; 3] {
        let d = self.attenuation_distance;
        if !d.is_finite() || d <= 0.0 {
            return [0.0; 3];
        }
        let mut sigma = [0.0f32; 3];
        for c in 0..3 {
            let a = self.attenuation_color[c].clamp(1e-6, 1.0);
            sigma[c] = -a.ln() / d;
        }
        sigma
    }

    /// GGX alpha for the base specular lobe, ignoring anisotropy.
    #[inline]
    fn alpha(&self) -> f32 {
        (self.roughness * self.roughness).max(1e-4)
    }

    /// GGX alphas along the tangent and bitangent for the base specular
    /// lobe.
    ///
    /// Uses the standard Disney/glTF construction: an aspect ratio of
    /// `sqrt(1 - 0.9·|anisotropy|)` splits the isotropic alpha into a
    /// stretched and a squeezed axis while keeping their product — and hence
    /// the overall highlight area — roughly fixed. `0.9` bounds the extreme
    /// case away from a zero-width lobe.
    ///
    /// At `anisotropy == 0` the aspect is exactly `1`, so both alphas are
    /// bit-identical to [`Self::alpha`] and every anisotropic code path
    /// reduces to the isotropic one.
    #[inline]
    fn alpha_tb(&self) -> (f32, f32) {
        let a = self.alpha();
        let aniso = self.anisotropy.clamp(-1.0, 1.0);
        if aniso == 0.0 {
            return (a, a);
        }
        let aspect = (1.0 - 0.9 * aniso.abs()).sqrt();
        let (wide, narrow) = ((a / aspect).min(1.0), a * aspect);
        if aniso > 0.0 {
            (wide, narrow)
        } else {
            (narrow, wide)
        }
    }

    /// GTR1 alpha for the clearcoat lobe.
    #[inline]
    fn coat_alpha(&self) -> f32 {
        (self.clearcoat_roughness * self.clearcoat_roughness).max(1e-4)
    }

    /// The dielectric part of `F0`, before any tint.
    ///
    /// Two parameters describe the same number and both are useful, so the
    /// precedence is fixed and stated rather than left to whichever the
    /// caller happened to set last:
    ///
    /// - **`ior` wins whenever it is not the default `1.5`.** A caller who
    ///   knows the index — 1.52 for soda-lime glass, 1.33 for water — has said
    ///   something physical, and Fresnel's `((n-1)/(n+1))²` is what they meant.
    /// - **Otherwise `specular` drives it**, through Disney's linear mapping
    ///   `F0 = 0.08 · specular`, whose 0..1 range covers IOR 1.0..1.8 and
    ///   whose midpoint 0.5 is IOR 1.5.
    ///
    /// The two agree exactly at the defaults — `0.08 × 0.5 = 0.04` and
    /// `((1.5-1)/(1.5+1))² = 0.04` — so the rule has no seam at the point
    /// where it switches, and every material that predates `specular`
    /// renders bit-identically.
    #[inline]
    fn f0_dielectric(&self) -> f32 {
        if self.ior == 1.5 {
            0.08 * self.specular
        } else {
            ((self.ior - 1.0) / (self.ior + 1.0)).powi(2)
        }
    }

    /// The base colour with its luminance divided out — Disney's `Ctint`, the
    /// hue and saturation of the surface without its brightness.
    ///
    /// Used to tint specular and sheen towards the surface's own colour
    /// without also darkening them.
    #[inline]
    fn tint(&self) -> [f32; 3] {
        let l = luminance(self.base_color);
        if l > 0.0 {
            scale3(self.base_color, 1.0 / l)
        } else {
            [1.0; 3]
        }
    }

    /// Specular reflectance at normal incidence.
    ///
    /// Dielectric `F0` (optionally tinted towards the base hue) blended
    /// towards the base colour by `metallic`, which is what makes a metal's
    /// reflection carry its colour and a dielectric's not.
    #[inline]
    fn f0(&self) -> [f32; 3] {
        let d = self.f0_dielectric();
        let dielectric = if self.specular_tint > 0.0 {
            mix3([d; 3], scale3(self.tint(), d), self.specular_tint)
        } else {
            [d; 3]
        };
        mix3(dielectric, self.base_color, self.metallic)
    }

    /// Diffuse albedo (metals have none).
    ///
    /// A transmissive dielectric has none either: the light that would have
    /// scattered diffusely went through instead, so `transmission` takes the
    /// lobe away exactly as `metallic` does.
    #[inline]
    fn diffuse_albedo(&self) -> [f32; 3] {
        let k = (1.0 - self.metallic) * (1.0 - self.transmission);
        [
            self.base_color[0] * k,
            self.base_color[1] * k,
            self.base_color[2] * k,
        ]
    }

    /// The colour to divide out before denoising, and multiply back after.
    ///
    /// Dielectrics carry their colour in the diffuse term; metals carry it in
    /// F0. Blending by `metallic` gives one buffer that tracks "what colour
    /// this surface is" for both, so the à-trous filter only ever sees the
    /// noisy illumination and never smears one part's colour into another's.
    #[inline]
    fn denoise_albedo(&self) -> [f32; 3] {
        mix3(self.diffuse_albedo(), self.f0(), self.metallic)
    }
}

// ─── lights & environment ─────────────────────────────────────────────────

/// A rectangular area light ("softbox"), emitting from its front face only.
///
/// Intersectable, so a BSDF ray that happens to land on it contributes and
/// combines with the explicit sample under MIS.
#[derive(Debug, Clone, Copy)]
pub struct AreaLight {
    /// Centre of the rectangle.
    pub center: Point3,
    /// Half-extent along the rectangle's first axis.
    pub u: Vec3,
    /// Half-extent along the rectangle's second axis.
    pub v: Vec3,
    /// Emitted radiance.
    pub emission: [f32; 3],
}

impl AreaLight {
    /// Unit normal of the emitting face.
    #[inline]
    pub(crate) fn normal(&self) -> Vec3 {
        self.u.cross(self.v).normalize()
    }

    /// Area of the rectangle in world units.
    #[inline]
    pub(crate) fn area(&self) -> f64 {
        4.0 * self.u.cross(self.v).norm()
    }

    /// Ray-rectangle intersection. Returns the hit distance if the ray
    /// strikes the emitting (front) face.
    fn intersect(&self, ray: &Ray) -> Option<f64> {
        let n = self.normal();
        let d = ray.direction.into_inner();
        let denom = n.dot(d);
        if denom.abs() < 1e-12 {
            return None;
        }
        let t = n.dot(self.center - ray.origin) / denom;
        if t <= 1e-6 {
            return None;
        }
        // Emitting face only: we must be looking at the front.
        if denom > 0.0 {
            return None;
        }
        let p = ray.at(t);
        let rel = p - self.center;
        let ul = self.u.norm();
        let vl = self.v.norm();
        let du = rel.dot(self.u / ul);
        let dv = rel.dot(self.v / vl);
        if du.abs() <= ul && dv.abs() <= vl {
            Some(t)
        } else {
            None
        }
    }

    /// Uniformly sample a point on the rectangle.
    pub(crate) fn sample(&self, r1: f64, r2: f64) -> Point3 {
        self.center + self.u * (2.0 * r1 - 1.0) + self.v * (2.0 * r2 - 1.0)
    }
}

/// Analytic studio environment: a smooth sky gradient plus a ground bounce.
///
/// Deliberately low-frequency — BSDF sampling alone integrates it cleanly, so
/// there is no environment importance-sampling CDF to build or maintain.
#[derive(Debug, Clone, Copy)]
pub struct GradientEnv {
    /// Radiance straight up.
    pub zenith: [f32; 3],
    /// Radiance at the horizon.
    pub horizon: [f32; 3],
    /// Radiance straight down (bounce off the studio floor).
    pub ground: [f32; 3],
    /// Overall multiplier.
    pub intensity: f32,
}

impl Default for GradientEnv {
    fn default() -> Self {
        Self {
            zenith: [0.34, 0.42, 0.55],
            horizon: [0.62, 0.64, 0.68],
            ground: [0.18, 0.17, 0.16],
            // The sky is ambient fill, not the key light — the softboxes
            // carry the image. Keeping this low is what preserves contrast.
            intensity: 0.35,
        }
    }
}

impl GradientEnv {
    /// Evaluate incoming radiance from direction `d` (world space, Z-up).
    fn radiance(&self, d: Vec3) -> [f32; 3] {
        let t = d.z as f32;
        let c = if t >= 0.0 {
            let k = smoothstep(t.powf(0.65));
            mix3(self.horizon, self.zenith, k)
        } else {
            let k = smoothstep((-t).powf(0.5));
            mix3(self.horizon, self.ground, k)
        };
        scale3(c, self.intensity)
    }
}

/// Relative luminance, the scalar the environment CDF is built over.
#[inline]
pub(crate) fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Sample a normalised piecewise-constant CDF (`cdf[0] == 0`, `cdf[n] == 1`).
///
/// Returns the chosen bin and the offset within it, so the caller can build a
/// continuous coordinate whose density is exactly the piecewise-constant one.
fn sample_1d(cdf: &[f32], u: f32) -> (usize, f32) {
    let n = cdf.len() - 1;
    let (mut lo, mut hi) = (0usize, n);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if cdf[mid + 1] <= u {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let i = lo.min(n - 1);
    let (a, b) = (cdf[i], cdf[i + 1]);
    let d = if b > a { (u - a) / (b - a) } else { 0.5 };
    (i, d.clamp(0.0, 1.0))
}

/// An [`EnvMap`] flattened for GPU upload. See [`EnvMap::pack_for_gpu`].
#[derive(Debug, Clone)]
pub struct GpuEnvPack {
    /// RGBA32F radiance, `width * height` texels.
    pub pixels: Vec<f32>,
    /// R32F CDF texture, `(width + 1) * (height + 1)`.
    pub cdf: Vec<f32>,
    /// Image width in texels.
    pub width: u32,
    /// Image height in texels.
    pub height: u32,
    /// Overall multiplier.
    pub intensity: f32,
    /// Rotation about +Z in radians.
    pub rotation: f32,
    /// PDF normaliser; zero means "not importance-sampled".
    pub marg_int: f32,
}

/// A lat-long (equirectangular) HDR environment map with a 2D CDF for
/// importance sampling.
///
/// Row 0 is the zenith (+Z) and the last row is nadir; column 0 is
/// `phi = rotation`. Unlike [`GradientEnv`] this can carry arbitrarily
/// high-frequency content — bright windows, a sun disc — so BSDF sampling
/// alone would be very noisy and importance sampling is mandatory.
///
/// The CDF is built over `luminance * sin(theta)`: the `sin(theta)` factor is
/// the lat-long solid-angle Jacobian, and omitting it over-weights the poles,
/// where texels cover almost no solid angle.
#[derive(Debug, Clone)]
pub struct EnvMap {
    width: usize,
    height: usize,
    /// Row-major linear radiance, `width * height` texels.
    pixels: Vec<[f32; 3]>,
    intensity: f32,
    /// Rotation about +Z in radians, applied when mapping u to phi.
    rotation: f64,
    /// Per-row conditional CDF over u, `height * (width + 1)` entries.
    cond_cdf: Vec<f32>,
    /// Marginal CDF over v, `height + 1` entries.
    marg_cdf: Vec<f32>,
    /// Mean of the weighted function; the normaliser for the uv-space PDF.
    marg_int: f32,
}

impl EnvMap {
    /// Build a map from row-major linear-RGB texels (row 0 = zenith).
    pub fn new(width: usize, height: usize, pixels: Vec<[f32; 3]>) -> Result<Self, String> {
        if width == 0 || height == 0 {
            return Err("environment map must have non-zero dimensions".to_string());
        }
        if pixels.len() != width * height {
            return Err(format!(
                "environment map has {} texels, expected {}x{} = {}",
                pixels.len(),
                width,
                height,
                width * height
            ));
        }

        let mut cond_cdf = vec![0.0f32; height * (width + 1)];
        let mut row_int = vec![0.0f32; height];
        for j in 0..height {
            // sin(theta) at the row's centre — the solid-angle weight.
            let sin_t = (std::f64::consts::PI * (j as f64 + 0.5) / height as f64).sin() as f32;
            let base = j * (width + 1);
            let mut acc = 0.0f32;
            for i in 0..width {
                acc += luminance(pixels[j * width + i]).max(0.0) * sin_t;
                cond_cdf[base + i + 1] = acc;
            }
            row_int[j] = acc / width as f32;
            if acc > 0.0 {
                for i in 1..=width {
                    cond_cdf[base + i] /= acc;
                }
            } else {
                // A black row is never selected by the marginal; keep its CDF
                // well-formed anyway so sampling can't index out of range.
                for i in 0..=width {
                    cond_cdf[base + i] = i as f32 / width as f32;
                }
            }
        }

        let mut marg_cdf = vec![0.0f32; height + 1];
        let mut acc = 0.0f32;
        for j in 0..height {
            acc += row_int[j];
            marg_cdf[j + 1] = acc;
        }
        let marg_int = acc / height as f32;
        if acc > 0.0 {
            for c in marg_cdf.iter_mut().skip(1) {
                *c /= acc;
            }
        } else {
            for (j, c) in marg_cdf.iter_mut().enumerate() {
                *c = j as f32 / height as f32;
            }
        }

        Ok(Self {
            width,
            height,
            pixels,
            intensity: 1.0,
            rotation: 0.0,
            cond_cdf,
            marg_cdf,
            marg_int,
        })
    }

    /// Width in texels.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Height in texels.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Scale every sample by `k`.
    pub fn with_intensity(mut self, k: f32) -> Self {
        self.intensity = k;
        self
    }

    /// Spin the environment about the vertical axis by `deg` degrees.
    ///
    /// A rigid rotation in phi leaves the CDF valid as-is: it is built in
    /// image space and only the image-to-direction mapping moves.
    pub fn with_rotation_deg(mut self, deg: f64) -> Self {
        self.rotation = deg.to_radians();
        self
    }

    /// Whether the map carries any energy to importance-sample.
    #[inline]
    fn is_sampleable(&self) -> bool {
        self.marg_int > 0.0
    }

    /// Image coordinates in `[0, 1)^2` for a world direction.
    fn uv(&self, d: Vec3) -> (f64, f64) {
        const EPS: f64 = 1e-9;
        let theta = d.z.clamp(-1.0, 1.0).acos();
        let v = (theta / std::f64::consts::PI).clamp(0.0, 1.0 - EPS);
        let phi = (d.y.atan2(d.x) - self.rotation).rem_euclid(std::f64::consts::TAU);
        let u = (phi / std::f64::consts::TAU).clamp(0.0, 1.0 - EPS);
        (u, v)
    }

    /// World direction for image coordinates in `[0, 1]^2`.
    fn direction(&self, u: f64, v: f64) -> Vec3 {
        let phi = u * std::f64::consts::TAU + self.rotation;
        let theta = v * std::f64::consts::PI;
        let (st, ct) = theta.sin_cos();
        Vec3::new(st * phi.cos(), st * phi.sin(), ct)
    }

    /// Texel indices for image coordinates.
    #[inline]
    fn texel_index(&self, u: f64, v: f64) -> (usize, usize) {
        let i = ((u * self.width as f64) as usize).min(self.width - 1);
        let j = ((v * self.height as f64) as usize).min(self.height - 1);
        (i, j)
    }

    /// Incoming radiance from direction `d`.
    ///
    /// Nearest-texel, deliberately: the CDF is piecewise-constant per texel,
    /// so a nearest lookup makes the sampled radiance and the PDF describe
    /// exactly the same function, which is what MIS requires.
    pub fn radiance(&self, d: Vec3) -> [f32; 3] {
        let (u, v) = self.uv(d);
        let (i, j) = self.texel_index(u, v);
        scale3(self.pixels[j * self.width + i], self.intensity)
    }

    /// Solid-angle PDF of the environment sampling strategy for direction `d`.
    ///
    /// The uv-space density converts with `dω = 2·π² · sin(θ) · du dv`, since
    /// `u` spans `2π` of azimuth and `v` spans `π` of polar angle.
    pub fn pdf(&self, d: Vec3) -> f32 {
        if !self.is_sampleable() {
            return 0.0;
        }
        let (u, v) = self.uv(d);
        let (i, j) = self.texel_index(u, v);
        let sin_bin = (std::f64::consts::PI * (j as f64 + 0.5) / self.height as f64).sin() as f32;
        let f = luminance(self.pixels[j * self.width + i]).max(0.0) * sin_bin;
        if f <= 0.0 {
            return 0.0;
        }
        let pdf_uv = f / self.marg_int;
        // Actual sin(theta) of this direction, not the bin's — the Jacobian
        // is a property of the point, while the bin weight is importance.
        let sin_t = (1.0 - d.z * d.z).max(0.0).sqrt() as f32;
        if sin_t <= 1e-9 {
            return 0.0;
        }
        let two_pi_sq = 2.0 * std::f32::consts::PI * std::f32::consts::PI;
        pdf_uv / (two_pi_sq * sin_t)
    }

    /// Pack for GPU upload as two textures — see `env.wgsl` for why textures
    /// rather than storage buffers.
    ///
    /// `pixels` is RGBA32F (`w * h`); `cdf` is R32F (`(w+1) * (h+1)`) with row
    /// `j < h` the conditional CDF for row `j` and row `h` the marginal.
    ///
    /// Lives here, next to where the CDFs are built, so the two descriptions of
    /// the layout cannot drift apart.
    pub fn pack_for_gpu(&self) -> GpuEnvPack {
        let (w, h) = (self.width, self.height);

        let mut pixels = Vec::with_capacity(4 * w * h);
        for px in &self.pixels {
            pixels.extend_from_slice(px);
            pixels.push(1.0);
        }

        // (w + 1) x (h + 1), zero-filled: each conditional row uses the full
        // width, the marginal uses only its first h + 1 entries.
        let mut cdf = vec![0.0f32; (w + 1) * (h + 1)];
        for j in 0..h {
            let base = j * (w + 1);
            cdf[base..base + w + 1].copy_from_slice(&self.cond_cdf[base..base + w + 1]);
        }
        let marg_row = h * (w + 1);
        cdf[marg_row..marg_row + self.marg_cdf.len()].copy_from_slice(&self.marg_cdf);

        GpuEnvPack {
            pixels,
            cdf,
            width: self.width as u32,
            height: self.height as u32,
            intensity: self.intensity,
            rotation: self.rotation as f32,
            // Zero when the map carries no energy, which switches the shader's
            // importance sampling off exactly as `is_sampleable` does here.
            marg_int: if self.is_sampleable() {
                self.marg_int
            } else {
                0.0
            },
        }
    }

    /// Importance-sample a direction. Returns `(direction, radiance, pdf)`
    /// with the PDF measured in solid angle.
    pub fn sample(&self, r1: f64, r2: f64) -> Option<(Vec3, [f32; 3], f32)> {
        if !self.is_sampleable() {
            return None;
        }
        let (j, dv) = sample_1d(&self.marg_cdf, r2 as f32);
        let base = j * (self.width + 1);
        let (i, du) = sample_1d(&self.cond_cdf[base..base + self.width + 1], r1 as f32);
        let u = (i as f64 + du as f64) / self.width as f64;
        let v = (j as f64 + dv as f64) / self.height as f64;
        let d = self.direction(u, v);
        // Recompute through `pdf` so the MIS partner and this sample agree
        // bit-for-bit on the density.
        let pdf = self.pdf(d);
        if pdf <= 0.0 || !pdf.is_finite() {
            return None;
        }
        Some((
            d,
            scale3(self.pixels[j * self.width + i], self.intensity),
            pdf,
        ))
    }
}

/// Incoming light from infinity.
///
/// The analytic [`GradientEnv`] is the default: fast, dependency-free, ships
/// no asset, and genuinely good for neutral product renders. [`EnvMap`] is
/// opt-in and brings the CDF machinery with it.
#[derive(Debug, Clone)]
pub enum Environment {
    /// Smooth analytic studio gradient. Sampled by the BSDF alone.
    Gradient(GradientEnv),
    /// Lat-long HDR image. Joins MIS as a third sampling strategy.
    Image(Box<EnvMap>),
}

impl Default for Environment {
    fn default() -> Self {
        Environment::Gradient(GradientEnv::default())
    }
}

impl Environment {
    /// A uniform environment of constant radiance — the white-furnace case.
    pub fn constant(rgb: [f32; 3]) -> Self {
        Environment::Gradient(GradientEnv {
            zenith: rgb,
            horizon: rgb,
            ground: rgb,
            intensity: 1.0,
        })
    }

    /// Wrap a lat-long HDR map.
    pub fn image(map: EnvMap) -> Self {
        Environment::Image(Box::new(map))
    }

    /// Evaluate incoming radiance from direction `d` (world space, Z-up).
    fn radiance(&self, d: Vec3) -> [f32; 3] {
        match self {
            Environment::Gradient(g) => g.radiance(d),
            Environment::Image(m) => m.radiance(d),
        }
    }

    /// Whether this environment participates in MIS as its own strategy.
    #[inline]
    fn is_importance_sampled(&self) -> bool {
        match self {
            Environment::Gradient(_) => false,
            Environment::Image(m) => m.is_sampleable(),
        }
    }

    /// Solid-angle PDF of the environment sampling strategy, or 0 when this
    /// environment is not importance-sampled.
    #[inline]
    fn pdf(&self, d: Vec3) -> f32 {
        match self {
            Environment::Gradient(_) => 0.0,
            Environment::Image(m) => m.pdf(d),
        }
    }

    /// Importance-sample a direction, if this environment supports it.
    #[inline]
    fn sample(&self, r1: f64, r2: f64) -> Option<(Vec3, [f32; 3], f32)> {
        match self {
            Environment::Gradient(_) => None,
            Environment::Image(m) => m.sample(r1, r2),
        }
    }
}

/// A directional light of finite angular size: the sun.
///
/// An [`AreaLight`] cannot express this — it is at a finite distance and its
/// solid angle falls off with it — and the analytic [`GradientEnv`] has no
/// disc in it at all. Daylight through a window is a very small, very bright
/// cone, which is exactly the case that needs its own sampling strategy: BSDF
/// sampling finds a cone of 0.5° by accident once in fifty thousand rays.
///
/// The sun joins MIS as a strategy of its own, on the same footing as the
/// area lights and the environment CDF: next-event estimation samples the
/// cone uniformly, a BSDF ray that escapes *into* the cone picks up the same
/// radiance under the balance-heuristic weight, and the two sum to one.
#[derive(Debug, Clone, Copy)]
pub struct Sun {
    /// Unit direction **towards** the sun.
    pub direction: Vec3,
    /// Angular radius of the disc, in radians. The real sun is 0.00465;
    /// larger values soften the shadow terminator.
    pub angular_radius: f64,
    /// Irradiance on a surface facing the sun square-on, linear RGB.
    ///
    /// Stated as irradiance rather than radiance so that changing
    /// `angular_radius` softens the shadows without changing the exposure —
    /// the radiance is `irradiance / solid_angle`.
    pub irradiance: [f32; 3],
}

impl Default for Sun {
    fn default() -> Self {
        // Midday sun through clear air, in the same arbitrary units the
        // studio rig's softboxes use.
        Self {
            direction: Vec3::new(-0.35, -0.55, 0.76),
            angular_radius: 0.02,
            irradiance: [3.0, 2.9, 2.7],
        }
    }
}

impl Sun {
    /// A sun of the given irradiance shining from `direction` (towards it).
    pub fn new(direction: Vec3, angular_radius: f64, irradiance: [f32; 3]) -> Self {
        Self {
            direction: direction.normalize(),
            angular_radius: angular_radius.clamp(1e-5, core::f64::consts::FRAC_PI_2),
            irradiance,
        }
    }

    /// Cosine of the disc's angular radius.
    #[inline]
    pub fn cos_radius(&self) -> f64 {
        self.angular_radius.cos()
    }

    /// Solid angle of the disc, steradians.
    #[inline]
    pub fn solid_angle(&self) -> f64 {
        core::f64::consts::TAU * (1.0 - self.cos_radius())
    }

    /// Radiance within the disc: irradiance spread over its solid angle.
    #[inline]
    pub fn radiance(&self) -> [f32; 3] {
        let w = self.solid_angle().max(1e-12) as f32;
        [
            self.irradiance[0] / w,
            self.irradiance[1] / w,
            self.irradiance[2] / w,
        ]
    }

    /// Radiance arriving from `d`, which is zero outside the disc.
    #[inline]
    pub fn radiance_in(&self, d: Vec3) -> [f32; 3] {
        if d.normalize().dot(self.direction.normalize()) >= self.cos_radius() {
            self.radiance()
        } else {
            [0.0; 3]
        }
    }

    /// Solid-angle PDF of the NEE strategy for `d`: uniform over the disc.
    #[inline]
    pub fn pdf(&self, d: Vec3) -> f32 {
        if d.normalize().dot(self.direction.normalize()) >= self.cos_radius() {
            (1.0 / self.solid_angle().max(1e-12)) as f32
        } else {
            0.0
        }
    }

    /// Sample a direction uniformly within the disc.
    ///
    /// Returns the direction, the radiance along it, and the PDF.
    pub fn sample(&self, r1: f64, r2: f64) -> (Vec3, [f32; 3], f32) {
        let cos_max = self.cos_radius();
        let cos_theta = 1.0 - r1 * (1.0 - cos_max);
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let phi = core::f64::consts::TAU * r2;
        let w = self.direction.normalize();
        let (u, v) = onb(w);
        let d = u * (sin_theta * phi.cos()) + v * (sin_theta * phi.sin()) + w * cos_theta;
        (
            d.normalize(),
            self.radiance(),
            (1.0 / self.solid_angle().max(1e-12)) as f32,
        )
    }
}

/// An infinite ground plane at a fixed Z, used as a studio sweep.
#[derive(Debug, Clone, Copy)]
pub struct Ground {
    /// Height of the plane.
    pub z: f64,
    /// Surface description.
    pub material: Pbr,
    /// When true the plane contributes only shadowing and contact darkening
    /// to alpha, so the render composites cleanly onto any backdrop.
    pub shadow_catcher: bool,
}

// ─── scene & camera ───────────────────────────────────────────────────────

/// A traceable object: one BVH over a BRep solid plus its material.
pub struct Object<G> {
    /// Acceleration structure over the solid's analytic faces.
    pub bvh: Arc<Bvh<G>>,
    /// Surface description.
    pub material: Pbr,
    /// Object → world placement of the BVH, which is otherwise traced
    /// wherever it was built.
    ///
    /// Most callers bake placement into the geometry and leave this at the
    /// identity. An animation instead holds the geometry (and its BVH) still
    /// and moves this, so a jointed assembly re-poses with no re-evaluation
    /// and no BLAS rebuild — only the top-level structure is rebuilt.
    pub transform: Transform,
}

impl<G> Object<G> {
    /// A traceable object placed where its BVH was built.
    pub fn new(bvh: Arc<Bvh<G>>, material: Pbr) -> Self {
        Self {
            bvh,
            material,
            transform: Transform::identity(),
        }
    }

    /// A traceable object placed by an object→world transform.
    pub fn placed(bvh: Arc<Bvh<G>>, material: Pbr, transform: Transform) -> Self {
        Self {
            bvh,
            material,
            transform,
        }
    }
}

/// Everything the integrator needs to render a frame.
pub struct Scene<G> {
    /// Traceable BRep objects.
    pub objects: Vec<Object<G>>,
    /// Explicit area lights.
    pub lights: Vec<AreaLight>,
    /// Analytic sky, or a lat-long HDR environment map.
    pub env: Environment,
    /// An optional directional light of finite angular size — daylight.
    ///
    /// `None` is the historical behaviour: the environment and the area
    /// lights are the whole of the illumination.
    pub sun: Option<Sun>,
    /// Optional studio floor.
    pub ground: Option<Ground>,
    /// An optional captured Gaussian splat cloud, composited *additively*
    /// over every ray segment.
    ///
    /// This is a radiance field, not geometry: the colours in it already
    /// include the lighting of the room they were captured in. So the
    /// integrator treats it as **emissive and absorbing** — along every
    /// segment it emits `Σ T·α·c` and attenuates what lies beyond by
    /// `Π (1 − α)` (see [`crate::splats::composite`]). It lights nothing by
    /// next-event estimation, receives nothing, and spawns no rays; it does
    /// veil analytic surfaces in front of it and attenuate shadow rays that
    /// cross it.
    ///
    /// The honest way to say it: **a splat backdrop is an environment with
    /// depth.** Like a lat-long [`EnvMap`] it supplies the radiance for rays
    /// that hit no analytic surface — so the marble in a captured garage
    /// picks up reflections and diffuse bounce from the real room — and
    /// unlike one it also occupies space, so it can stand in front of
    /// something as well as behind it.
    ///
    /// The limitation that comes with that: the splat field is **not
    /// importance sampled.** There is no `Environment::sample` for it and no
    /// MIS strategy aimed at its bright spots, so indirect light from the
    /// cloud arrives only on BSDF-sampled bounce rays. A mirror or a smooth
    /// glass marble is therefore clean at low sample counts and a rough
    /// diffuse surface under a small bright window is noisy — exactly the
    /// behaviour of the analytic [`GradientEnv`], for the same reason.
    pub splats: Option<Arc<Bvh<Splats>>>,
}

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
    fn ray(&self, sx: f64, sy: f64, aspect: f64, lu: f64, lv: f64) -> Ray {
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
const BH: [f64; 4] = [0.35875, -0.48829, 0.14128, -0.01168];

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
    fn cdf(self, x: f64) -> f64 {
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
fn erf(x: f64) -> f64 {
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

/// Integrator settings.
#[derive(Debug, Clone, Copy)]
pub struct PathTraceOptions {
    /// Samples per pixel.
    pub spp: u32,
    /// Maximum path length (1 = direct lighting only).
    pub max_depth: u32,
    /// Depth at which Russian roulette begins.
    pub rr_start: u32,
    /// Clamp on indirect radiance, to kill fireflies. `None` disables.
    ///
    /// An *absolute* cap, in radiance units: any direct-lighting estimate at
    /// depth > 0 is truncated to it. Cheap and effective, and biased in a way
    /// that does not matter for a studio render — but it is a fixed number
    /// against a quantity whose scale is the scene's, so a bright scene has
    /// its highlights shaved and a dim one keeps its fireflies.
    ///
    /// A caustic is exactly the case where that bias is *not* acceptable: a
    /// focused spot is legitimately many times the surrounding radiance, and
    /// an absolute cap is indistinguishable from throwing the caustic away.
    /// Contributions read out of a [`crate::caustics::CausticMap`] are
    /// therefore never clamped — they are a density estimate, not a Monte
    /// Carlo spike, and they have no long tail to cut.
    pub firefly_clamp: Option<f32>,
    /// Clamp on indirect radiance *relative to what the pixel has already
    /// measured*, replacing [`Self::firefly_clamp`] when set.
    ///
    /// The number is a multiple of the pixel's running mean luminance: `8.0`
    /// lets any sample through that is within eight times the brightness the
    /// pixel has settled on so far, and cuts the ones past it. Scale-free, so
    /// the same value works on a sunlit court and a dim pool, and it adapts to
    /// the pixel rather than to the scene — a pixel inside a caustic has a
    /// high running mean and keeps its energy, a pixel in shadow does not.
    ///
    /// The first few samples have no mean to speak of, so the clamp does not
    /// engage until [`Self::firefly_clamp_warmup`] samples have landed.
    ///
    /// `None` — the default — leaves the absolute clamp in charge and every
    /// render that predates this field bit-identical.
    pub firefly_clamp_relative: Option<f32>,
    /// Samples a pixel must take before [`Self::firefly_clamp_relative`]
    /// engages.
    pub firefly_clamp_warmup: u32,
    /// Render the environment behind the subject rather than leaving it clear.
    pub show_background: bool,
    /// Random seed.
    pub seed: u64,
    /// Stop sampling a pixel early once its own variance estimate says the
    /// remaining budget cannot move it visibly.
    ///
    /// [`spp`](Self::spp) becomes a *ceiling* rather than a fixed count. Every
    /// pixel still gets at least a floor of samples, and the decision is made
    /// from the pixel's own running sums, so the film stays deterministic and
    /// independent of how the frame was tiled — a pixel that stops early keeps
    /// the unbiased mean of the samples it did take.
    ///
    /// Set `false` for a reference render, where a uniform sample count is
    /// the point.
    pub adaptive: bool,
    /// Reconstruction filter for primary-ray placement within the pixel.
    ///
    /// [`PixelFilter::Box`] — uniform jitter — is the default and reproduces
    /// every earlier render bit for bit.
    pub filter: PixelFilter,
    /// Run the edge-aware à-trous denoiser over the film before returning.
    ///
    /// This is a pure post-process on the accumulated radiance — it consumes
    /// no random numbers and cannot change the integrator's estimate.
    pub denoise: bool,
    /// À-trous iterations. Each doubles the tap stride, so `n` iterations
    /// reach a footprint of roughly `2^(n+1)` pixels.
    pub denoise_iters: u32,
    /// Edge-stopping tolerance on the world normal, as `‖n_p − n_q‖`.
    /// Smaller keeps creases sharper and denoises less.
    pub sigma_normal: f32,
    /// Edge-stopping tolerance on hit distance, *relative* to the centre
    /// pixel's depth and scaled by the tap stride (so grazing surfaces still
    /// filter).
    pub sigma_depth: f32,
    /// Edge-stopping tolerance on demodulated illumination luminance. Halved
    /// each iteration, per Dammertz, so late wide passes cannot flatten
    /// detail the early passes already resolved.
    pub sigma_lum: f32,
}

impl Default for PathTraceOptions {
    fn default() -> Self {
        Self {
            spp: 128,
            max_depth: 6,
            rr_start: 3,
            firefly_clamp: Some(12.0),
            firefly_clamp_relative: None,
            firefly_clamp_warmup: 16,
            show_background: true,
            seed: 0x5eed_1234,
            adaptive: true,
            filter: PixelFilter::Box,
            denoise: true,
            denoise_iters: 5,
            sigma_normal: 0.35,
            sigma_depth: 0.02,
            sigma_lum: 4.0,
        }
    }
}

// ─── rng ──────────────────────────────────────────────────────────────────

/// Small, fast, deterministic PRNG (PCG-XSH-RR style).
#[derive(Clone, Copy)]
pub(crate) struct Rng(u64);

impl Rng {
    #[inline]
    pub(crate) fn new(seed: u64) -> Self {
        // Mix so neighbouring pixel seeds decorrelate immediately.
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        s ^= s >> 29;
        s = s.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        s ^= s >> 32;
        Rng(s | 1)
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = (((self.0 >> 18) ^ self.0) >> 27) as u32;
        let rot = (self.0 >> 59) as u32;
        x.rotate_right(rot)
    }

    /// Uniform in [0, 1).
    #[inline]
    pub(crate) fn f64(&mut self) -> f64 {
        (self.next_u32() as f64) * (1.0 / 4294967296.0)
    }
}

// ─── low-discrepancy sampling ─────────────────────────────────────────────

/// Van der Corput radical inverse of `i` in `BASE`.
///
/// Reflects `i`'s digits in `BASE` about the radix point, which spreads
/// consecutive indices as far apart as the base allows. Successive prime
/// bases give the Halton sequence, which is what the camera dimensions use.
#[inline]
pub(crate) fn radical_inverse<const BASE: u32>(mut i: u64) -> f64 {
    let inv_base = 1.0 / BASE as f64;
    let mut inv_bn = 1.0;
    let mut acc = 0u64;
    // Accumulate the reversed digits as an integer, then scale once: doing
    // the division per digit accumulates rounding error over ~50 digits.
    while i > 0 {
        let digit = i % BASE as u64;
        acc = acc * BASE as u64 + digit;
        i /= BASE as u64;
        inv_bn *= inv_base;
    }
    (acc as f64 * inv_bn).min(1.0 - f64::EPSILON)
}

/// Cranley-Patterson rotation: shift `x` by `offset` on the unit torus.
///
/// Preserves the point set's discrepancy while randomising its absolute
/// placement, which is what lets every pixel share one low-discrepancy set
/// without the shared structure showing up as a visible pattern.
#[inline]
pub(crate) fn cp_rotate(x: f64, offset: f64) -> f64 {
    let v = x + offset;
    if v >= 1.0 { v - 1.0 } else { v }
}

/// Samples traced between convergence checks.
///
/// The check needs a sample variance to be worth anything, so it cannot run
/// after every sample; 16 gives a usable estimate and is fine enough that a
/// converged pixel wastes at most 15 samples past the line.
const ADAPTIVE_BATCH: u32 = 16;

/// Minimum samples every pixel gets, whatever the variance estimate says.
///
/// A pixel that happens to draw several near-equal samples early reports a
/// tiny variance and would quit while genuinely unconverged — the classic
/// adaptive-sampling failure, and it shows up as blotching in exactly the
/// smooth regions adaptivity was meant to speed up.
const ADAPTIVE_FLOOR: u32 = 32;

/// Relative tolerance on the 95% confidence half-width of pixel luminance.
const ADAPTIVE_TOL: f32 = 0.10;

/// Absolute luminance added to the mean before applying [`ADAPTIVE_TOL`].
///
/// Pure relative error never converges in shadow, where the mean approaches
/// zero; pure absolute error over-samples highlights. Adding the two is the
/// usual compromise.
const ADAPTIVE_LUM_FLOOR: f32 = 0.02;

// ─── small math helpers ───────────────────────────────────────────────────

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}

#[inline]
fn scale3(a: [f32; 3], k: f32) -> [f32; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}

#[inline]
fn mul3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

#[inline]
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn max3(a: [f32; 3]) -> f32 {
    a[0].max(a[1]).max(a[2])
}

#[inline]
fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Orthonormal basis around a unit normal (Duff et al., branchless).
pub(crate) fn onb(n: Vec3) -> (Vec3, Vec3) {
    let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    (
        Vec3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x),
        Vec3::new(b, sign + n.y * n.y * a, -n.y),
    )
}

/// An orthonormal shading frame: tangent, bitangent, normal.
///
/// The tangent is meaningful, not arbitrary, whenever the hit surface has a
/// real parameterisation — that is what anisotropic shading orients itself
/// by — so the three vectors travel together rather than as loose arguments.
#[derive(Debug, Clone, Copy)]
struct Frame {
    t: Vec3,
    b: Vec3,
    n: Vec3,
}

/// Shading tangent frame around a unit normal.
///
/// When the hit carried a surface tangent `dP/du`, it is Gram-Schmidt
/// orthogonalised against the (possibly face-forwarded) shading normal and
/// used as the frame's x axis, so the anisotropic lobe lines up with the
/// surface's own parameterisation. Otherwise this is the arbitrary [`onb`]
/// basis the isotropic path has always used — which is exactly what an
/// isotropic material wants, since its BSDF is invariant to the choice.
fn shading_frame(n: Vec3, dpdu: Option<Vec3>) -> Frame {
    if let Some(d) = dpdu {
        let t = d - n * d.dot(n);
        // A tangent that is (numerically) parallel to the normal carries no
        // direction; fall back rather than normalising noise.
        if t.norm() > 1e-9 {
            let t = t.normalize();
            return Frame {
                t,
                b: n.cross(t),
                n,
            };
        }
    }
    let (t, b) = onb(n);
    Frame { t, b, n }
}

#[inline]
fn to_local(t: Vec3, b: Vec3, n: Vec3, w: Vec3) -> Vec3 {
    Vec3::new(w.dot(t), w.dot(b), w.dot(n))
}

#[inline]
fn to_world(t: Vec3, b: Vec3, n: Vec3, w: Vec3) -> Vec3 {
    t * w.x + b * w.y + n * w.z
}

/// Cosine-weighted hemisphere sample in local space (+Z up).
pub(crate) fn cosine_hemisphere(r1: f64, r2: f64) -> Vec3 {
    let r = r1.sqrt();
    let phi = 2.0 * std::f64::consts::PI * r2;
    Vec3::new(r * phi.cos(), r * phi.sin(), (1.0 - r1).max(0.0).sqrt())
}

/// Uniform sample on the unit disc (concentric mapping).
fn concentric_disc(r1: f64, r2: f64) -> (f64, f64) {
    let a = 2.0 * r1 - 1.0;
    let b = 2.0 * r2 - 1.0;
    if a == 0.0 && b == 0.0 {
        return (0.0, 0.0);
    }
    let (r, theta) = if a * a > b * b {
        (a, std::f64::consts::FRAC_PI_4 * (b / a))
    } else {
        (
            b,
            std::f64::consts::FRAC_PI_2 - std::f64::consts::FRAC_PI_4 * (a / b),
        )
    };
    (r * theta.cos(), r * theta.sin())
}

// ─── microfacet BRDF ──────────────────────────────────────────────────────

/// GGX / Trowbridge-Reitz normal distribution, taking the half-vector rather
/// than its cosine.
///
/// The textbook denominator is `(n·h)²(α² - 1) + 1`, and evaluated in f32 that
/// expression is worthless exactly where it matters most. At retro-reflection —
/// `wo`, `wi` and the normal all within a degree or two, which is where a
/// camera ray meets a wall head-on — `n·h` is 1 to within a few ulp, so the
/// product is `-1 + ulp` and adding 1 back cancels every significant digit.
/// What is supposed to survive is α², and for a smooth material α² is smaller
/// than the error it is being asked to emerge from: at roughness 0.08 α² is
/// 4.1e-5 against an f32 resolution near 1 of 6e-8.
///
/// `(n·h)²(α² - 1) + 1` is algebraically `α²·h_z² + (h_x² + h_y²)`, and that
/// form is a sum of non-negative terms taken from the half-vector directly, so
/// nothing cancels: at exact retro it is α², to full precision. It is the same
/// rearrangement [`d_ggx_aniso`] already used, which is why the anisotropic
/// path never had the problem.
#[inline]
fn d_ggx(wh: Vec3, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let (hx, hy, hz) = (wh.x as f32, wh.y as f32, wh.z as f32);
    let d = a2 * hz * hz + (hx * hx + hy * hy);
    a2 / (std::f32::consts::PI * d * d).max(1e-9)
}

/// Anisotropic GGX normal distribution.
///
/// `wh` is the half-vector in the local shading frame, whose x axis is the
/// surface tangent. When `at == ab` this is algebraically identical to
/// [`d_ggx`]; the equality is made exact (not merely near-exact in floating
/// point) by dispatching to it.
#[inline]
fn d_ggx_aniso(wh: Vec3, at: f32, ab: f32) -> f32 {
    if at == ab {
        return d_ggx(wh, at);
    }
    let (hx, hy, hz) = (wh.x as f32, wh.y as f32, wh.z as f32);
    let d = (hx / at) * (hx / at) + (hy / ab) * (hy / ab) + hz * hz;
    1.0 / (std::f32::consts::PI * at * ab * d * d).max(1e-9)
}

/// Smith height-correlated visibility term (already divided by 4·NoL·NoV).
#[inline]
fn v_smith(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let gv = n_dot_l * (n_dot_v * n_dot_v * (1.0 - a2) + a2).sqrt();
    let gl = n_dot_v * (n_dot_l * n_dot_l * (1.0 - a2) + a2).sqrt();
    0.5 / (gv + gl).max(1e-9)
}

/// Anisotropic Smith height-correlated visibility term (already divided by
/// 4·NoL·NoV). Reduces exactly to [`v_smith`] when `at == ab`.
#[inline]
fn v_smith_aniso(wo: Vec3, wi: Vec3, at: f32, ab: f32) -> f32 {
    if at == ab {
        return v_smith(wo.z as f32, wi.z as f32, at);
    }
    // Λ-style stretched lengths: sqrt((at·x)² + (ab·y)² + z²).
    let stretched = |w: Vec3| -> f32 {
        let (x, y, z) = (w.x as f32, w.y as f32, w.z as f32);
        ((at * x) * (at * x) + (ab * y) * (ab * y) + z * z).sqrt()
    };
    let gv = (wi.z as f32) * stretched(wo);
    let gl = (wo.z as f32) * stretched(wi);
    0.5 / (gv + gl).max(1e-9)
}

/// Smith G1 masking term for the anisotropic GGX distribution.
///
/// The `at == ab` branch is the isotropic expression evaluated exactly as it
/// was before anisotropy existed, so isotropic renders are bit-unchanged.
#[inline]
fn g1_smith_aniso(w: Vec3, at: f32, ab: f32) -> f32 {
    let z = w.z.max(1e-6) as f32;
    let lambda = if at == ab {
        let a2 = at * at;
        ((1.0 + a2 * (1.0 - z * z) / (z * z)).sqrt() - 1.0) * 0.5
    } else {
        let (x, y) = (w.x as f32, w.y as f32);
        (((at * x) * (at * x) + (ab * y) * (ab * y) + z * z).sqrt() / z - 1.0) * 0.5
    };
    1.0 / (1.0 + lambda)
}

/// Schlick Fresnel.
#[inline]
fn fresnel(f0: [f32; 3], cos_theta: f32) -> [f32; 3] {
    let m = (1.0 - cos_theta).clamp(0.0, 1.0).powi(5);
    [
        f0[0] + (1.0 - f0[0]) * m,
        f0[1] + (1.0 - f0[1]) * m,
        f0[2] + (1.0 - f0[2]) * m,
    ]
}

/// The specular lobe's Fresnel: Schlick against `f0`, or the Airy
/// reflectance of a thin film sitting on top of it.
///
/// One function, so the film modulates the dielectric `F0` path and the metal
/// path alike — a metal's `f0` *is* its base colour, and an oxide film over
/// steel colours it exactly the way this composes. `lambda_nm <= 0` means the
/// path is still RGB and wants the colour-integrated form; a path that has
/// already drawn a hero wavelength gets the exact reflectance at that λ
/// instead, which is both cheaper and righter.
///
/// `thin_film_thickness == 0` returns [`fresnel`] itself, not a limit of the
/// film model that happens to be close — so a material without a film is
/// bit-for-bit what it was.
#[inline]
fn spec_fresnel(m: &Pbr, f0: [f32; 3], cos_theta: f32, lambda_nm: f32) -> [f32; 3] {
    if m.thin_film_thickness <= 0.0 {
        return fresnel(f0, cos_theta);
    }
    let cos = cos_theta.clamp(0.0, 1.0);
    if lambda_nm > 0.0 {
        crate::optics::thin_film_fresnel_at(
            m.thin_film_thickness,
            m.thin_film_ior,
            cos,
            f0,
            lambda_nm,
        )
    } else {
        crate::optics::thin_film_fresnel(m.thin_film_thickness, m.thin_film_ior, cos, f0)
    }
}

// ─── diffuse: energy-preserving Oren-Nayar ────────────────────────────────
//
// d'Eon, Portsmouth, Hill, Fascione, "EON: A practical energy-preserving
// rough diffuse BRDF" (JCGT 14(1), 2025; arXiv:2410.18026).
//
// Lambert is only correct for a mirror-smooth interface over an isotropically
// scattering half-space. A real matte surface is rough at the microscale, and
// Oren-Nayar's qualitative model of that — v-cavities of Lambertian facets —
// gives the flat, backscattering look of chalk, clay and unfinished plaster
// that Lambert cannot. What Oren-Nayar (and Fujii's improved form, FON) does
// not do is conserve energy: interreflection between the cavities is dropped,
// so a white surface comes back grey. EON adds that back analytically, which
// is what makes it safe to turn on by default: at albedo 1 it reflects 1.

/// `0.5 - 2/(3pi)`, the constant in the FON normalisation `A_F`.
const FON_C1: f32 = 0.5 - 2.0 / (3.0 * std::f32::consts::PI);
/// `2/3 - 28/(15pi)`, the constant in the FON *average* albedo.
const FON_C2: f32 = 2.0 / 3.0 - 28.0 / (15.0 * std::f32::consts::PI);

/// FON's normalisation factor `A_F`.
#[inline]
fn fon_a(r: f32) -> f32 {
    1.0 / (1.0 + FON_C1 * r)
}

/// Directional albedo of the FON lobe, `E_F(mu, r)` — the paper's exact form
/// rather than its quartic fit, since we are not on a shader clock here and
/// the exact one is only an `acos` more expensive.
#[inline]
fn e_fon(mu: f32, r: f32) -> f32 {
    let mu = mu.clamp(1e-6, 1.0);
    let a = fon_a(r);
    let si = (1.0 - mu * mu).max(0.0).sqrt();
    let g = si * (mu.acos() - si * mu) + (2.0 / 3.0) * ((si / mu) * (1.0 - si * si * si) - si);
    a + (a * r) * std::f32::consts::FRAC_1_PI * g
}

/// The EON diffuse BRDF (already multiplied by nothing — this is `f`, not
/// `f·cos`).
///
/// `r == 0` short-circuits to Lambert. That is not only an optimisation: the
/// multiple-scattering term is a `0/0` there, guarded in the paper by an
/// epsilon that leaves a `1e-7`-scale residue behind. Returning `rho/pi`
/// outright is both the exact limit and what keeps every pre-existing
/// material bit-identical.
fn eon_diffuse(rho: [f32; 3], r: f32, wo: Vec3, wi: Vec3) -> [f32; 3] {
    if r <= 0.0 {
        return scale3(rho, std::f32::consts::FRAC_1_PI);
    }
    let mu_i = wi.z as f32;
    let mu_o = wo.z as f32;
    // The Fujii single-scattering term. `s` is the azimuthal cosine times the
    // two sines; dividing by the larger cosine is what gives the model its
    // characteristic flat, edge-lit shape.
    let s = (wi.dot(wo) as f32) - mu_i * mu_o;
    let s_over_t = if s > 0.0 {
        s / mu_i.max(mu_o).max(1e-6)
    } else {
        s
    };
    let a = fon_a(r);
    let f_ss = scale3(rho, std::f32::consts::FRAC_1_PI * a * (1.0 + r * s_over_t));

    // The multiple-scattering compensation: the energy the v-cavities would
    // have passed between their walls, redistributed as a smooth lobe whose
    // shape is `(1 - E(mu_o))(1 - E(mu_i))`, so it is zero where the single
    // scattering already conserves and largest where it loses most.
    let e_o = e_fon(mu_o, r);
    let e_i = e_fon(mu_i, r);
    let avg = a * (1.0 + FON_C2 * r);
    const EPS: f32 = 1e-7;
    let k = std::f32::consts::FRAC_1_PI * (1.0 - e_o).max(EPS) * (1.0 - e_i).max(EPS)
        / (1.0 - avg).max(EPS);
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let rho_ms = rho[c] * rho[c] * avg / (1.0 - rho[c] * (1.0 - avg)).max(EPS);
        out[c] = f_ss[c] + rho_ms * k;
    }
    out
}

// ─── subsurface: a random walk, not a look ────────────────────────────────
//
// Chiang, Kutz and Burley, "Practical and Controllable Subsurface Scattering
// for Production Path Tracing" (SIGGRAPH 2016 Talks).
//
// Disney's 2012 `subsurface` was a *blend*: a Hanrahan-Krueger-flavoured lobe
// that flattened the diffuse falloff and brightened grazing angles the way a
// short mean free path does. It looked like scattering and transported
// nothing — light never entered the object, so it never came out anywhere
// else, and the effect vanished the moment you asked it for the thing that
// actually distinguishes skin, marble and rubber from paint of the same
// colour: light going *in* here and coming *out* over there.
//
// This is the transport. On a subsurface entry the path stops being a surface
// event, crosses into the object and walks: sample a distance against the
// medium's extinction, and either the boundary comes first — in which case
// the path leaves there, from a new point with a new normal — or it does not,
// in which case the path scatters isotropically and goes again.
//
// # The inversion is the whole usability of it
//
// Nobody can pick a single-scattering albedo. A medium at 0.9 reads as very
// nearly white at the surface, and the map from one to the other is a
// transcendental function of the transport. Chiang's fit inverts it, so the
// parameter is the *surface* colour — what the material looks like — and the
// renderer solves for the medium. `a_semi_infinite_slab_returns_its_own_colour`
// is the test that the fit is doing its job: a half-space of the material must
// reflect `subsurface_color` back, to within a few percent, or the knob is
// lying about what it does.

/// Chiang's albedo inversion: the single-scattering albedo whose semi-infinite
/// diffuse reflectance is `a`, for an isotropic phase function.
#[inline]
fn scatter_albedo(a: f32) -> f32 {
    let a = a.clamp(0.0, 1.0);
    1.0 - (-5.094_06 * a + 2.611_88 * a * a - 4.318_05 * a * a * a).exp()
}

/// Where a subsurface walk came back out, and what it carries.
#[derive(Debug, Clone, Copy)]
struct Exit {
    /// The point on the surface the path leaves from — generally not the one
    /// it entered at, which is the entire point.
    point: Point3,
    /// The outward normal there.
    normal: Vec3,
    /// Throughput accumulated over the walk, per channel.
    weight: [f32; 3],
}

/// How many scattering events a walk may take before the path is dropped.
///
/// A bright medium scatters a great many times before it finds its way out,
/// and truncating the walk loses exactly the energy that would have made it
/// bright — so this is generous. At `subsurface_color = 0.9` the inverted
/// medium has a single-scattering albedo of 0.9964, so a walk lives 275
/// scatters on average and 1024 truncates a couple of percent of them; below
/// 0.8, where every real material in these scenes sits, the mean is under
/// forty and truncation is not measurable. The GPU's own walk is bounded far
/// lower, and that difference is documented rather than hidden.
const SUBSURFACE_MAX_STEPS: u32 = 1024;

/// Walk inside the object until the boundary is crossed.
///
/// `trace` is the caller's ray cast from a point inside the medium: it returns
/// the distance to the first boundary along that direction and the geometric
/// normal there, or `None` when the ray meets no boundary at all. Passing the
/// geometry in as a closure is what lets the walk be tested against an
/// analytic half-space, which is where its one quantitative claim — that it
/// reproduces `subsurface_color` — can actually be checked.
///
/// `None` means the walk was absorbed or ran out of steps: the path ends.
fn subsurface_walk(
    m: &Pbr,
    entry: Point3,
    n: Vec3,
    rng: &mut Rng,
    mut trace: impl FnMut(Point3, Vec3) -> Option<(f64, Vec3)>,
) -> Option<Exit> {
    // The medium the surface colour and the mean free path imply.
    let mut sigma_t = [0.0f64; 3];
    let mut sigma_s = [0.0f64; 3];
    for c in 0..3 {
        let r = m.subsurface_radius[c];
        if !(r > 0.0) || !r.is_finite() {
            return None;
        }
        sigma_t[c] = 1.0 / r;
        sigma_s[c] = sigma_t[c] * scatter_albedo(m.subsurface_color[c]) as f64;
    }

    // In through the surface, cosine-distributed about the inward normal —
    // the index-matched boundary the inversion was fitted against.
    let c = cosine_hemisphere(rng.f64(), rng.f64());
    let (t_ax, b_ax) = onb(-n);
    let mut dir = to_world(t_ax, b_ax, -n, c);
    let mut pos = entry - n * 1e-5;
    let mut weight = [1.0f64; 3];

    for _ in 0..SUBSURFACE_MAX_STEPS {
        // One channel drives the distance; the other two ride along under the
        // balance heuristic, which is what keeps a medium with very different
        // per-channel paths from turning into three-way noise.
        let ch = ((rng.f64() * 3.0) as usize).min(2);
        let t = -(1.0 - rng.f64()).ln() / sigma_t[ch];
        let hit = trace(pos, dir);
        let boundary = hit.map_or(f64::INFINITY, |(d, _)| d);

        if boundary <= t {
            let (d, hit_n) = hit?;
            let mut pdf = 0.0;
            for c in 0..3 {
                pdf += (-sigma_t[c] * d).exp() / 3.0;
            }
            if pdf <= 0.0 {
                return None;
            }
            for c in 0..3 {
                weight[c] *= (-sigma_t[c] * d).exp() / pdf;
            }
            // Face the normal out of the medium, which is the side the ray
            // was travelling towards.
            let normal = if hit_n.dot(dir) > 0.0 { hit_n } else { -hit_n };
            return Some(Exit {
                point: pos + dir * d,
                normal,
                weight: [weight[0] as f32, weight[1] as f32, weight[2] as f32],
            });
        }

        let mut pdf = 0.0;
        for c in 0..3 {
            pdf += sigma_t[c] * (-sigma_t[c] * t).exp() / 3.0;
        }
        if pdf <= 0.0 {
            return None;
        }
        for c in 0..3 {
            weight[c] *= sigma_s[c] * (-sigma_t[c] * t).exp() / pdf;
        }
        pos = pos + dir * t;
        // Isotropic phase function: the medium has no memory of which way the
        // light was going, which is what a dense scattering medium is.
        let z = 1.0 - 2.0 * rng.f64();
        let r = (1.0 - z * z).max(0.0).sqrt();
        let phi = std::f64::consts::TAU * rng.f64();
        dir = Vec3::new(r * phi.cos(), r * phi.sin(), z).normalize();

        // Russian roulette on what is left, so a dark medium costs a few
        // steps rather than all of them. The survival probability is capped
        // at 1 and not at the integrator's 0.95: a bright medium's weight
        // hovers near 1, and a 0.95 cap there would kill 5% of walks per step
        // while inflating the survivors, which is unbiased and useless — the
        // variance swamps the mean long before the estimator finds it.
        let q = weight[0].max(weight[1]).max(weight[2]).clamp(0.0, 1.0);
        if rng.f64() > q {
            return None;
        }
        for w in weight.iter_mut() {
            *w /= q;
        }
    }
    None
}

// ─── specular: multiple-scattering compensation ───────────────────────────

/// Turquin's multiple-scattering compensation factor for the GGX lobe
/// (Turquin 2019, "Practical multiple scattering compensation for microfacet
/// models").
///
/// A single-scattering microfacet BRDF drops every path that leaves one facet
/// and strikes another, and for a rough surface that is most of the energy:
/// 5% missing at `alpha = 0.2`, 31% at `0.5`, 69% at `1.0`. Turquin's
/// observation is that the *shape* of the missing lobe barely matters, so
/// rather than model it, scale the single-scattering lobe by
///
/// ```text
///     1 + F0 · (1 - E(mu_o)) / E(mu_o)
/// ```
///
/// where `E` is the single-scattering directional albedo with `F = 1`, baked
/// into [`crate::tables::GGX_E`]. The factor is per-channel through `F0`,
/// which is what makes a coloured metal's multiple bounces saturate the way
/// they should — each extra bounce multiplies by the metal's reflectance
/// again. At `F0 = 1` the factor is exactly `1/E`, so a white furnace reads
/// exactly 1 at every roughness; at a dielectric's `F0 = 0.04` it is a fraction
/// of a percent and the lobe stays comfortably under 1.
///
/// The factor depends on the outgoing direction only, so it is not exactly
/// reciprocal — that is the price of the closed form, and Turquin's point is
/// that the error is far smaller than the energy it recovers. See
/// `tests::the_layered_bsdf_is_reciprocal` for the bound it is held to.
///
/// Anisotropy enters only through `sqrt(at·ab)`, the alpha of the isotropic
/// lobe of the same area. The table is one-dimensional in roughness and a
/// stretched lobe loses very nearly the same total energy as the round one it
/// came from, so a second axis would buy nothing.
#[inline]
fn ms_compensation(f0: [f32; 3], at: f32, ab: f32, mu_o: f32) -> [f32; 3] {
    let alpha = (at * ab).max(0.0).sqrt();
    let e = crate::tables::bilinear(&crate::tables::GGX_E, alpha, mu_o).clamp(1e-3, 1.0);
    let k = (1.0 - e) / e;
    [1.0 + f0[0] * k, 1.0 + f0[1] * k, 1.0 + f0[2] * k]
}

// ─── sheen: multiple-scattering LTC ───────────────────────────────────────
//
// Zeltner, Burley and Chiang, "Practical Multiple-Scattering Sheen Using
// Linearly Transformed Cosines" (SIGGRAPH 2022 Talks). Disney's 2012 sheen was
// a Schlick-weighted tint bolted onto the edge of the diffuse lobe; this is a
// fit to the real thing — multiple scattering in a thin layer of normally
// oriented fibres with an SGGX microflake phase function — summarised as a
// linearly transformed cosine, which evaluates, integrates and importance-
// samples in closed form.

/// The LTC coefficients `(a_inv, b_inv, R)` for a view direction.
#[inline]
fn sheen_coeffs(m: &Pbr, mu_o: f32) -> [f32; 3] {
    crate::tables::bilinear3(
        &crate::tables::SHEEN_LTC,
        m.sheen_roughness.clamp(0.0, 1.0),
        mu_o,
    )
}

/// Evaluate the LTC density for `wi` in the frame where `wo` lies in the
/// x-z plane. This is both the sheen lobe's shape *and* its PDF — an LTC is
/// a normalised distribution, which is the whole point of the representation.
#[inline]
fn sheen_ltc_density(wi_std: Vec3, coeffs: [f32; 3]) -> f32 {
    let (a_inv, b_inv) = (coeffs[0] as f64, coeffs[1] as f64);
    if a_inv <= 0.0 {
        return 0.0;
    }
    let w = Vec3::new(
        a_inv * wi_std.x + b_inv * wi_std.z,
        a_inv * wi_std.y,
        wi_std.z,
    );
    let len = w.norm();
    if len <= 0.0 {
        return 0.0;
    }
    let cos_theta = (w.z / len).max(0.0) as f32;
    let jacobian = (a_inv * a_inv / (len * len * len)) as f32;
    cos_theta * std::f32::consts::FRAC_1_PI * jacobian
}

/// Rotate `w` about +Z so that `wo`'s azimuth becomes zero — the frame the
/// LTC fit is defined in.
#[inline]
fn sheen_align(wo: Vec3, w: Vec3) -> Vec3 {
    let len = (wo.x * wo.x + wo.y * wo.y).sqrt();
    if len <= 0.0 {
        return w;
    }
    let (c, s) = (wo.x / len, wo.y / len);
    // Rotation by -phi_o.
    Vec3::new(c * w.x + s * w.y, -s * w.x + c * w.y, w.z)
}

/// Undo [`sheen_align`].
#[inline]
fn sheen_unalign(wo: Vec3, w: Vec3) -> Vec3 {
    let len = (wo.x * wo.x + wo.y * wo.y).sqrt();
    if len <= 0.0 {
        return w;
    }
    let (c, s) = (wo.x / len, wo.y / len);
    Vec3::new(c * w.x - s * w.y, s * w.x + c * w.y, w.z)
}

/// The sheen lobe's `f · cos`, and its PDF.
fn sheen_eval(m: &Pbr, wo: Vec3, wi: Vec3) -> ([f32; 3], f32) {
    if m.sheen <= 0.0 {
        return ([0.0; 3], 0.0);
    }
    let coeffs = sheen_coeffs(m, wo.z as f32);
    let density = sheen_ltc_density(sheen_align(wo, wi), coeffs);
    // The LTC carries the cosine, so `density` is already `f · cos`, scaled by
    // the lobe's directional reflectance R and the artist's colour.
    let k = density * coeffs[2] * m.sheen;
    (scale3(m.sheen_color, k), density)
}

/// Directional reflectance of the sheen layer — what it takes away from
/// everything beneath it.
#[inline]
fn sheen_albedo(m: &Pbr, mu_o: f32) -> f32 {
    if m.sheen <= 0.0 {
        return 0.0;
    }
    (sheen_coeffs(m, mu_o)[2] * m.sheen * max3(m.sheen_color)).clamp(0.0, 1.0)
}

// ─── clearcoat: GTR1 ──────────────────────────────────────────────────────

/// Generalised-Trowbridge-Reitz with `gamma = 1` — Disney's clearcoat
/// distribution.
///
/// GGX (`gamma = 2`) is already long-tailed; GTR1 is longer still, and that
/// extra tail is exactly what a lacquer looks like — a tight core with a broad
/// halo around it, rather than GGX's single soft blob.
#[inline]
fn d_gtr1(wh: Vec3, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    if a2 >= 1.0 {
        return std::f32::consts::FRAC_1_PI;
    }
    // `a2·cos² + sin²` taken from the half-vector's components rather than
    // from `1 - cos²`, for the reason spelled out at length on `d_ggx`: at
    // retro-reflection `cos²` is 1 to within an f32 ulp and the subtraction
    // cancels every digit that mattered.
    let (hx, hy, hz) = (wh.x as f32, wh.y as f32, wh.z as f32);
    let d = a2 * hz * hz + (hx * hx + hy * hy);
    // `a2 - 1` and `ln(a2)` are both negative for a plausible coat, so the
    // quotient is positive; written with both signs flipped so that clamping
    // the denominator away from zero clamps it on the correct side.
    (1.0 - a2) / (std::f32::consts::PI * (-a2.ln()) * d).max(1e-9)
}

/// Sample GTR1's normal distribution (Disney 2012, appendix B). Returns the
/// half-vector in the local frame.
fn sample_gtr1(alpha: f32, r1: f64, r2: f64) -> Vec3 {
    let a2 = (alpha * alpha).clamp(1e-8, 0.999_999) as f64;
    let cos2 = ((1.0 - a2.powf(1.0 - r1)) / (1.0 - a2)).clamp(0.0, 1.0);
    let cos_t = cos2.sqrt();
    let sin_t = (1.0 - cos2).max(0.0).sqrt();
    let phi = std::f64::consts::TAU * r2;
    Vec3::new(sin_t * phi.cos(), sin_t * phi.sin(), cos_t)
}

/// PDF of [`sample_gtr1`] in solid angle around `wi`.
#[inline]
fn gtr1_pdf(wo: Vec3, wh: Vec3, alpha: f32) -> f32 {
    let o_dot_h = wo.dot(wh).max(1e-9) as f32;
    d_gtr1(wh, alpha) * (wh.z.max(0.0) as f32) / (4.0 * o_dot_h)
}

/// Sample the GGX visible-normal distribution (Heitz 2018). `wo` is in local
/// space with +Z the shading normal and +X the surface tangent; returns the
/// sampled half-vector.
///
/// The method is anisotropic by construction: the stretch step that maps to
/// the hemisphere-configured space takes each axis' alpha separately, so
/// passing `at != ab` needs no other change.
fn sample_vndf(wo: Vec3, at: f32, ab: f32, r1: f64, r2: f64) -> Vec3 {
    let (a, b) = (at as f64, ab as f64);
    // Stretch the view direction into the hemisphere-configured space.
    let vh = Vec3::new(a * wo.x, b * wo.y, wo.z).normalize();
    let lensq = vh.x * vh.x + vh.y * vh.y;
    let t1 = if lensq > 0.0 {
        Vec3::new(-vh.y, vh.x, 0.0) / lensq.sqrt()
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let t2 = vh.cross(t1);

    let r = r1.sqrt();
    let phi = 2.0 * std::f64::consts::PI * r2;
    let p1 = r * phi.cos();
    let p2r = r * phi.sin();
    let s = 0.5 * (1.0 + vh.z);
    let p2 = (1.0 - s) * (1.0 - p1 * p1).max(0.0).sqrt() + s * p2r;

    let nh = t1 * p1 + t2 * p2 + vh * (1.0 - p1 * p1 - p2 * p2).max(0.0).sqrt();
    Vec3::new(a * nh.x, b * nh.y, nh.z.max(1e-9)).normalize()
}

/// PDF of the VNDF sampling strategy, in solid angle around `wi`.
#[inline]
fn vndf_pdf(wo: Vec3, wh: Vec3, at: f32, ab: f32) -> f32 {
    let n_dot_v = wo.z.max(1e-6) as f32;
    let d = d_ggx_aniso(wh, at, ab);
    let g1 = g1_smith_aniso(wo, at, ab);
    let o_dot_h = wo.dot(wh).max(1e-9) as f32;
    d * g1 * o_dot_h / n_dot_v / (4.0 * o_dot_h)
}

// ─── rough dielectric ─────────────────────────────────────────────────────
//
// Walter, Marschner, Li and Torrance (2007), "Microfacet Models for
// Refraction through Rough Surfaces": the same GGX microsurface the specular
// lobe already stands on, with the half-vector generalised so that a facet
// can *refract* as well as reflect. It shares this material's alpha and its
// VNDF sampling; only the half-vector, the Jacobian and the Fresnel split
// differ.
//
// # Conventions, stated once
//
// - `eta` throughout is `n_transmitted / n_incident`. Entering glass from air
//   that is `n_glass`; leaving it, `1/n_glass`. The integrator computes it
//   from which side of the *geometric* normal the ray arrived on, so a
//   non-nested solid needs no medium stack at all.
// - The reflect/transmit split is the exact unpolarised Fresnel from
//   [`crate::optics::fresnel`], not Schlick. Total internal reflection is not
//   a special case: it is what that formula returns when Snell has no
//   solution.
// - **Radiance is scaled by `1/η²` on refraction.** A BSDF is not symmetric
//   across a change of index, and which way the asymmetry goes depends on
//   what the path carries. This is a camera path carrying importance, so the
//   factor is `(η_i/η_t)²` — pbrt's "radiance mode". Walter's `η_t²`
//   numerator and that `1/η²` cancel algebraically, which is why the
//   expression below has no `eta²` in it anywhere.
//
//   The consequence to keep straight: a *single* interface therefore does not
//   have unit throughput. Entering glass compresses radiance by `1/η²` and
//   leaving expands it back by `η²`, so it is the **round trip** that is a
//   no-op. `a_rough_glass_sphere_closes_the_furnace` takes the scaling back
//   out to ask the energy question, and `a_slab_is_a_round_trip_no_op` asks
//   the transport one; both have to hold.

/// Smith `G1` for a direction that may be on either side of the surface.
///
/// The reflection path can assume `w.z > 0`; a refracted direction is below
/// the surface by construction, and the shadowing term is a function of the
/// *magnitude* of the slope, so this is the same Λ with `|z|`.
#[inline]
fn g1_smith_abs(w: Vec3, at: f32, ab: f32) -> f32 {
    let z = (w.z.abs() as f32).max(1e-6);
    let (x, y) = (w.x as f32, w.y as f32);
    let lambda = (((at * x) * (at * x) + (ab * y) * (ab * y) + z * z).sqrt() / z - 1.0) * 0.5;
    1.0 / (1.0 + lambda)
}

/// Exact unpolarised Fresnel reflectance for a dielectric, given the cosine
/// against the (micro)normal and `eta = n_t / n_i`.
///
/// Returns `1.0` on total internal reflection, which is the physically right
/// answer and not a guard.
#[inline]
fn fresnel_dielectric(cos_i: f32, eta: f32) -> f32 {
    let cos_i = cos_i.clamp(0.0, 1.0);
    let sin2_t = (1.0 - cos_i * cos_i) / (eta * eta);
    if sin2_t >= 1.0 {
        return 1.0;
    }
    let cos_t = (1.0 - sin2_t).max(0.0).sqrt();
    crate::optics::fresnel(1.0f32, eta, cos_i, cos_t).clamp(0.0, 1.0)
}

/// The VNDF's density in the half-vector, before any reflect/refract
/// Jacobian. [`vndf_pdf`] is this divided by `4·(wo·h)`.
#[inline]
fn vndf_density(wo: Vec3, wh: Vec3, at: f32, ab: f32) -> f32 {
    let n_dot_v = wo.z.max(1e-6) as f32;
    let d = d_ggx_aniso(wh, at, ab);
    let g1 = g1_smith_aniso(wo, at, ab);
    let o_dot_h = wo.dot(wh).max(1e-9) as f32;
    d * g1 * o_dot_h / n_dot_v
}

/// The dielectric lobe's contribution, evaluated for one direction pair.
///
/// Returns `(f·cos, pdf)` for the lobe alone, un-weighted; the caller scales
/// by `m.transmission` and folds the PDF in with the others. `wi.z > 0` is the
/// reflected branch and `wi.z < 0` the transmitted one, and both branches are
/// driven by the same sampled facet, so their PDFs sum to the lobe's total.
fn dielectric_eval(m: &Pbr, wo: Vec3, wi: Vec3, eta: f32) -> (f32, f32) {
    let (at, ab) = m.alpha_tb();

    if m.thin_walled {
        // A sheet: refract in and straight back out. The exit direction is the
        // entry direction, so the whole event is a *mirror through the
        // surface* — the same GGX reflection lobe with its z flipped, which
        // keeps the roughness (a frosted pane is still frosted) and adds no
        // lateral offset. `eta` is always the outside-to-glass ratio here;
        // there is no inside to be on the other side of.
        let flipped = Vec3::new(wi.x, wi.y, -wi.z);
        if flipped.z <= 0.0 || wo.z <= 0.0 {
            return (0.0, 0.0);
        }
        let wh = (wo + flipped).normalize();
        let o_dot_h = wo.dot(wh).max(0.0) as f32;
        let f = fresnel_dielectric(o_dot_h, m.ior.max(1.0));
        let d = d_ggx_aniso(wh, at, ab);
        let vis = v_smith_aniso(wo, flipped, at, ab);
        let value = (1.0 - f) * d * vis * (flipped.z as f32);
        let pdf = vndf_pdf(wo, wh, at, ab) * (1.0 - f);
        return (value, pdf);
    }

    if wi.z > 0.0 {
        // Reflection off the same microsurface, split by the exact Fresnel.
        let wh = (wo + wi).normalize();
        let o_dot_h = wo.dot(wh).max(0.0) as f32;
        let f = fresnel_dielectric(o_dot_h, eta);
        let d = d_ggx_aniso(wh, at, ab);
        let vis = v_smith_aniso(wo, wi, at, ab);
        let value = f * d * vis * (wi.z as f32);
        let pdf = vndf_pdf(wo, wh, at, ab) * f;
        return (value, pdf);
    }

    // Transmission. The generalised half-vector: `−(η_i·wo + η_t·wi)`, which
    // with η_i factored out is `−(wo + η·wi)`, oriented into the upper
    // hemisphere so it can be fed to the same D and G.
    let eta = eta as f64;
    let h = -(wo + wi * eta);
    if h.norm() < 1e-9 {
        return (0.0, 0.0);
    }
    let mut wh = h.normalize();
    if wh.z < 0.0 {
        wh = -wh;
    }
    let cos_o = wo.dot(wh) as f32;
    let cos_i = wi.dot(wh) as f32;
    // A facet the viewer cannot see, or one whose two sides are on the same
    // side of it, transmits nothing.
    if cos_o <= 0.0 || cos_i >= 0.0 {
        return (0.0, 0.0);
    }
    let f = fresnel_dielectric(cos_o, eta as f32);
    let d = d_ggx_aniso(wh, at, ab);
    let g = g1_smith_abs(wo, at, ab) * g1_smith_abs(wi, at, ab);
    let denom = {
        let x = cos_o + (eta as f32) * cos_i;
        (x * x).max(1e-12)
    };
    // `f·cos`, with Walter's η_t² already cancelled against the 1/η² radiance
    // scaling — see the module note above.
    let value = (1.0 - f) * d * g * (cos_o * cos_i).abs() / (wo.z.abs() as f32) / denom;
    // dω_h/dω_i for refraction.
    let jacobian = (eta as f32) * (eta as f32) * cos_i.abs() / denom;
    let pdf = vndf_density(wo, wh, at, ab) * (1.0 - f) * jacobian;
    (value.max(0.0), pdf.max(0.0))
}

/// Relative sampling weights of the five lobes, in the order
/// (diffuse, specular, sheen, coat, dielectric).
///
/// Each is that lobe's approximate albedo, so a material spends its samples
/// where its energy is: a mirror almost never draws a diffuse direction, a
/// chalk wall almost always does.
///
/// `transmission` moves weight out of the diffuse and opaque-specular lobes
/// and into the dielectric one — and because it also scales those two lobes'
/// *values*, an opaque material (`transmission = 0`) gets the same four
/// numbers it always did, to the bit.
fn lobe_weights(m: &Pbr) -> [f32; 6] {
    let opaque = 1.0 - m.transmission;
    // The subsurface weight takes its share out of the diffuse lobe rather
    // than adding beside it, which is what keeps the surface's total albedo
    // where it was.
    let diff = max3(m.diffuse_albedo()).max(0.0) * (1.0 - m.subsurface);
    let spec = (max3(m.f0()).max(0.0) + 0.08) * opaque;
    let sheen = (m.sheen * max3(m.sheen_color)).max(0.0);
    let coat = m.clearcoat * 0.25;
    let diel = m.transmission.max(0.0);
    let sss = (m.subsurface * max3(m.subsurface_color)).max(0.0) * opaque;
    let total = (diff + spec + sheen + coat + diel + sss).max(1e-6);
    [
        diff / total,
        spec / total,
        sheen / total,
        coat / total,
        diel / total,
        sss / total,
    ]
}

/// Evaluate the full BSDF and its sampling PDF for a given in/out pair.
///
/// Both vectors are in the local shading frame (+Z = normal) and point away
/// from the surface. Returns `(f * cos, pdf)`.
fn bsdf_eval(m: &Pbr, wo: Vec3, wi: Vec3, eta: f32, lambda_nm: f32) -> ([f32; 3], f32) {
    if wo.z <= 0.0 {
        return ([0.0; 3], 0.0);
    }
    if wi.z <= 0.0 {
        // Below the surface: only the dielectric lobe lives down here.
        if m.transmission <= 0.0 {
            return ([0.0; 3], 0.0);
        }
        let w = lobe_weights(m);
        let (v, p) = dielectric_eval(m, wo, wi, eta);
        let k = v * m.transmission;
        return ([k, k, k], (w[4] * p).max(0.0));
    }
    let n_dot_l = wi.z as f32;
    let n_dot_v = wo.z as f32;
    let wh = (wo + wi).normalize();
    let o_dot_h = wo.dot(wh).max(0.0) as f32;

    let w = lobe_weights(m);

    // Diffuse: EON, minus whatever the subsurface walk is carrying. The walk
    // is a BSSRDF — it leaves from a different point and has no density at
    // this one — so it is not in this sum at all; what `subsurface` does here
    // is take its share of the lobe away. At `diffuse_roughness = 0` and
    // `subsurface = 0` this is Lambert to the last bit.
    let rho = m.diffuse_albedo();
    let diffuse_f = scale3(
        eon_diffuse(rho, m.diffuse_roughness, wo, wi),
        1.0 - m.subsurface,
    );
    let diffuse = scale3(diffuse_f, n_dot_l);
    let pdf_d = n_dot_l * std::f32::consts::FRAC_1_PI;

    // Base specular. Anisotropy stretches the lobe along the local x axis,
    // which the integrator has aligned with the surface tangent dP/du; the
    // compensation factor puts back the facet-to-facet bounces the single-
    // scattering model drops.
    let (at, ab) = m.alpha_tb();
    let d = d_ggx_aniso(wh, at, ab);
    let vis = v_smith_aniso(wo, wi, at, ab);
    let f0 = m.f0();
    let f = spec_fresnel(m, f0, o_dot_h, lambda_nm);
    let spec = scale3(
        mul3(
            scale3(f, d * vis * n_dot_l),
            ms_compensation(f0, at, ab, n_dot_v),
        ),
        1.0 - m.transmission,
    );
    let pdf_s = vndf_pdf(wo, wh, at, ab);

    // The dielectric lobe's reflected half. Same facets, same alpha; the
    // difference is that its split against transmission is the exact Fresnel,
    // so grazing goes to 1 the way glass does and Schlick's fit does not.
    let (diel, pdf_diel) = if m.transmission > 0.0 {
        let (v, p) = dielectric_eval(m, wo, wi, eta);
        ([v * m.transmission; 3], p)
    } else {
        ([0.0; 3], 0.0)
    };

    // Sheen, between the coat and the base: fibre fuzz, which is what makes
    // velvet and a nylon net glow along their silhouettes.
    let (sheen, pdf_sh) = sheen_eval(m, wo, wi);
    // Albedo scaling for what the fuzz took. The geometric mean of the two
    // directions' losses keeps the layering reciprocal, which a bare
    // `1 - E(mu_o)` would not be.
    let sheen_atten = if m.sheen > 0.0 {
        ((1.0 - sheen_albedo(m, n_dot_v)) * (1.0 - sheen_albedo(m, n_dot_l)))
            .max(0.0)
            .sqrt()
    } else {
        1.0
    };

    // Clearcoat: a thin dielectric film over everything else, GTR1 at a fixed
    // IOR of 1.5. It is isotropic — the grain lives in the substrate beneath
    // it, not in the lacquer — so it never takes the anisotropy.
    //
    // Disney scale their coat by a further 0.25 because their `clearcoat`
    // parameter is documented as covering [0, 0.25]; ours is a full-strength
    // 0..1 layer weight, so the 0.25 lives in the caller's number instead.
    let (coat, pdf_c, coat_atten) = if m.clearcoat > 0.0 {
        let ca = m.coat_alpha();
        let cd = d_gtr1(wh, ca);
        let cv = v_smith(n_dot_v, n_dot_l, ca);
        let cf = fresnel([0.04, 0.04, 0.04], o_dot_h)[0] * m.clearcoat;
        let c = cd * cv * n_dot_l * cf;
        (
            [c, c, c],
            gtr1_pdf(wo, wh, ca),
            // Energy removed from the layers beneath.
            1.0 - cf,
        )
    } else {
        ([0.0; 3], 0.0, 1.0)
    };

    let base = scale3(add3(add3(diffuse, spec), diel), sheen_atten);
    let under = add3(base, sheen);
    let value = add3(scale3(under, coat_atten), coat);
    let pdf = w[0] * pdf_d + w[1] * pdf_s + w[2] * pdf_sh + w[3] * pdf_c + w[4] * pdf_diel;
    (value, pdf.max(0.0))
}

/// The lobe-selection probabilities, exposed so the WGSL port can be checked
/// against them — a mismatch here is invisible in an image (both tiers are
/// still unbiased) and shows up only as noise, which is the worst kind of
/// disagreement to have to find by eye.
pub fn reference_lobe_weights(m: &Pbr) -> [f32; 6] {
    lobe_weights(m)
}

/// Reference BSDF evaluation, in the local shading frame (+Z = normal).
///
/// Exposed so the WGSL port in `gpu/shaders/bsdf.wgsl` can be checked against
/// this implementation — see `tests/bsdf_parity.rs`. Returns `(f * cos, pdf)`;
/// the PDF is the one MIS must agree on across both renderers.
pub fn reference_bsdf_eval(m: &Pbr, wo: Vec3, wi: Vec3, eta: f32) -> ([f32; 3], f32) {
    bsdf_eval(m, wo, wi, eta, 0.0)
}

/// [`reference_bsdf_eval`] for a path that already carries a hero wavelength.
///
/// Only the thin film cares: everything else in the model is achromatic in
/// λ, and `lambda_nm <= 0` is the RGB sentinel, so this and the four-argument
/// form agree exactly for any material without a film.
pub fn reference_bsdf_eval_at(
    m: &Pbr,
    wo: Vec3,
    wi: Vec3,
    eta: f32,
    lambda_nm: f32,
) -> ([f32; 3], f32) {
    bsdf_eval(m, wo, wi, eta, lambda_nm)
}

/// What [`bsdf_sample`] drew.
enum Sampled {
    /// A direction off the surface, with `f*cos` and the PDF it was drawn at.
    Surface(Vec3, [f32; 3], f32),
    /// The subsurface lobe: the path goes *into* the object and the
    /// integrator has to walk it out. There is no direction and no density
    /// yet — only the weight the lobe choice costs, which is `1 / P(lobe)`.
    Subsurface([f32; 3]),
}

/// [`bsdf_sample`] restricted to the surface lobes, for the tests that only
/// have a BSDF and no scene to walk through.
#[cfg(test)]
fn bsdf_sample_surface(
    m: &Pbr,
    wo: Vec3,
    eta: f32,
    lambda_nm: f32,
    rng: &mut Rng,
) -> Option<(Vec3, [f32; 3], f32)> {
    match bsdf_sample(m, wo, eta, lambda_nm, rng)? {
        Sampled::Surface(wi, f, pdf) => Some((wi, f, pdf)),
        Sampled::Subsurface(_) => None,
    }
}

/// Importance-sample the BSDF.
fn bsdf_sample(m: &Pbr, wo: Vec3, eta: f32, lambda_nm: f32, rng: &mut Rng) -> Option<Sampled> {
    if wo.z <= 0.0 {
        return None;
    }
    let w = lobe_weights(m);
    let u = rng.f64() as f32;
    let r1 = rng.f64();
    let r2 = rng.f64();

    let wi = if u < w[0] {
        cosine_hemisphere(r1, r2)
    } else if u < w[0] + w[1] {
        let (at, ab) = m.alpha_tb();
        let wh = sample_vndf(wo, at, ab, r1, r2);
        let wi = reflect(-wo, wh);
        if wi.z <= 0.0 {
            return None;
        }
        wi
    } else if u < w[0] + w[1] + w[2] {
        // The LTC is a linear transform of a cosine lobe, so sampling it is
        // sampling the cosine and pushing the direction through the matrix.
        let coeffs = sheen_coeffs(m, wo.z as f32);
        let (a_inv, b_inv) = (coeffs[0] as f64, coeffs[1] as f64);
        if a_inv <= 0.0 {
            return None;
        }
        let c = cosine_hemisphere(r1, r2);
        let wi_std = Vec3::new(c.x / a_inv - c.z * b_inv / a_inv, c.y / a_inv, c.z).normalize();
        let wi = sheen_unalign(wo, wi_std);
        if wi.z <= 0.0 {
            return None;
        }
        wi
    } else if w[5] > 0.0 && u >= w[0] + w[1] + w[2] + w[3] + w[4] {
        // Into the object. The walk carries its own *colour* — that is what
        // the albedo inversion buys — but not its own *weight*: the diffuse
        // lobe above gave up exactly `subsurface` of itself, so that is what
        // this lobe is worth, and the division is the lobe choice's own
        // probability. Together the two sum to `(1 - s)·rho + s·A`, which is
        // the whole reason the surface's albedo does not move when the
        // subsurface weight does. The `w[5] > 0` guard keeps a material
        // without a subsurface lobe on exactly the branch chain it was on
        // before this lobe existed, comparison for comparison.
        return Some(Sampled::Subsurface([m.subsurface / w[5]; 3]));
    } else if u < w[0] + w[1] + w[2] + w[3] {
        let ca = m.coat_alpha();
        let wh = sample_gtr1(ca, r1, r2);
        let wi = reflect(-wo, wh);
        if wi.z <= 0.0 {
            return None;
        }
        wi
    } else {
        // The dielectric lobe. One facet is drawn from the VNDF and the ray
        // then either reflects off it or refracts through it, with the exact
        // Fresnel as the branch probability — so total internal reflection is
        // simply the case where that probability is 1, and no code path
        // knows it is special.
        let (at, ab) = m.alpha_tb();
        let wh = sample_vndf(wo, at, ab, r1, r2);
        if m.thin_walled {
            let f = fresnel_dielectric(wo.dot(wh).max(0.0) as f32, m.ior.max(1.0));
            let wi = reflect(-wo, wh);
            if wi.z <= 0.0 {
                return None;
            }
            // Reflect above, or mirror straight through below.
            if (rng.f64() as f32) < f {
                wi
            } else {
                Vec3::new(wi.x, wi.y, -wi.z)
            }
        } else {
            let f = fresnel_dielectric(wo.dot(wh).max(0.0) as f32, eta);
            if (rng.f64() as f32) < f {
                let wi = reflect(-wo, wh);
                if wi.z <= 0.0 {
                    return None;
                }
                wi
            } else {
                let (wi, _, _) = crate::optics::refract(-wo, wh, 1.0, eta as f64)?;
                let wi = wi.normalize();
                if wi.z >= 0.0 {
                    return None;
                }
                wi
            }
        }
    };

    let (f, pdf) = bsdf_eval(m, wo, wi, eta, lambda_nm);
    if pdf <= 1e-9 {
        return None;
    }
    Some(Sampled::Surface(wi, f, pdf))
}

#[inline]
fn reflect(i: Vec3, n: Vec3) -> Vec3 {
    i - n * (2.0 * i.dot(n))
}

/// How many thin transmissive sheets one shadow ray may cross before it is
/// declared blocked.
///
/// A window is one pane, a double glazing two, a display case in a lit room
/// perhaps four. Past that the light is not meaningfully getting through and
/// the traversal cost is not worth paying, so the cap is both a physical
/// judgement and a loop bound.
pub const MAX_SHADOW_SHEETS: usize = 4;

/// The fraction of a shadow ray that survives one thin transmissive sheet, or
/// `None` if the surface is an honest blocker.
///
/// `cos_dot` is the (signed) dot of the surface normal with the ray
/// direction; only its magnitude matters, a pane filters the same from either
/// side.
///
/// The Fresnel factor is the same single-interface `1 − F` the thin-walled
/// branch of [`dielectric_eval`] applies, so a NEE path and a BSDF path
/// through the same pane carry the same weight — the condition for MIS to
/// combine them without double counting.
///
/// The result is achromatic because the thin-walled lobe is: this renderer's
/// sheet has no interior, so there is no path length for Beer–Lambert to act
/// over and no tint in the BSDF to match. A coloured pane would need both
/// changed together.
fn sheet_transmittance(m: &Pbr, cos_dot: f64) -> Option<[f32; 3]> {
    if !m.thin_walled || m.transmission <= 0.0 {
        return None;
    }
    // A rough sheet scatters, and the straight-line shadow ray is only the
    // right answer in the smooth limit. Taper to zero as the lobe opens up,
    // so a frosted pane blocks exactly as it did before.
    let clarity = (1.0 - m.alpha()).clamp(0.0, 1.0);
    if clarity <= 0.0 {
        return None;
    }
    let cos = (cos_dot.abs() as f32).clamp(0.0, 1.0);
    let f = fresnel_dielectric(cos, m.ior.max(1.0));
    let t = m.transmission * (1.0 - f) * clarity;
    if t <= 0.0 { None } else { Some([t; 3]) }
}

/// Power heuristic (β = 2) for multiple importance sampling.
#[inline]
fn power_heuristic(a: f32, b: f32) -> f32 {
    let a2 = a * a;
    let b2 = b * b;
    if a2 + b2 <= 0.0 { 0.0 } else { a2 / (a2 + b2) }
}

// ─── intersection ─────────────────────────────────────────────────────────

/// What a ray landed on.
enum Landing {
    Surface {
        point: Point3,
        normal: Vec3,
        /// Surface tangent dP/du, when the parameterisation has one.
        tangent: Option<Vec3>,
        material: Pbr,
    },
    Light {
        emission: [f32; 3],
        light_index: usize,
        distance: f64,
        point: Point3,
    },
    Miss,
}

/// The scene's geometry gathered into a TLAS, built once per render.
///
/// `Scene` keeps `objects` as its authoring surface — a plain list is the
/// right thing to *write* — while the integrator traces against this. Kept
/// separate rather than added as a `Scene` field so the public struct-literal
/// construction in `Scene { objects, lights, env, ground }` keeps working.
pub(crate) struct SceneAccel<G> {
    tlas: Tlas<G>,
    /// Cumulative distribution over `scene.lights`, weighted by emitted power
    /// (emission luminance × area). One entry per light, ending at 1.0.
    ///
    /// Built once per render so next-event estimation can draw *one* light per
    /// bounce instead of shadow-raying all of them: the cost per bounce stops
    /// scaling with the number of softboxes, and the estimator stays unbiased
    /// because each contribution is divided by its own pick probability.
    light_cdf: Vec<f32>,
    /// Probability of picking each light, i.e. the CDF's per-entry mass. Kept
    /// alongside so the MIS weight for a BSDF ray that lands on an emitter can
    /// use the same pick probability the NEE strategy would have used.
    light_pick_pdf: Vec<f32>,
}

impl<G> SceneAccel<G> {
    /// Probability that [`SceneAccel::pick_light`] would choose `index`.
    #[inline]
    pub(crate) fn light_pick_pdf(&self, index: usize) -> f32 {
        self.light_pick_pdf.get(index).copied().unwrap_or(0.0)
    }

    /// Draw one light from the power-weighted table. Returns its index and the
    /// probability with which it was drawn.
    #[inline]
    fn pick_light(&self, u: f32) -> Option<(usize, f32)> {
        if self.light_cdf.is_empty() {
            return None;
        }
        let i = match self
            .light_cdf
            .binary_search_by(|c| c.partial_cmp(&u).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(i) | Err(i) => i.min(self.light_cdf.len() - 1),
        };
        let pdf = self.light_pick_pdf[i];
        if pdf > 0.0 { Some((i, pdf)) } else { None }
    }
}

/// Power-weighted selection table over a light list: per-light pick
/// probabilities and their running sum.
///
/// Shared by the CPU integrator and the GPU scene upload so both sample the
/// same distribution — a parity test that compared two different tables would
/// be testing nothing.
pub fn light_power_table(lights: &[AreaLight]) -> (Vec<f32>, Vec<f32>) {
    let powers: Vec<f32> = lights
        .iter()
        .map(|l| (luminance(l.emission) as f64 * l.area()).max(0.0) as f32)
        .collect();
    power_table_from_weights(&powers)
}

/// The weight → (CDF, per-entry probability) half of [`light_power_table`],
/// split out so the GPU scene upload can build the identical table from its
/// own packed lights.
pub fn power_table_from_weights(powers: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let total: f32 = powers.iter().sum();
    let n = powers.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    // A scene whose lights all carry zero power still needs a valid
    // distribution; uniform costs nothing and keeps the estimator finite.
    let pick: Vec<f32> = if total > 0.0 && total.is_finite() {
        powers.iter().map(|p| p / total).collect()
    } else {
        vec![1.0 / n as f32; n]
    };
    let mut cdf = Vec::with_capacity(n);
    let mut run = 0.0f32;
    for p in &pick {
        run += *p;
        cdf.push(run);
    }
    // Guard against float drift leaving the last entry just under 1.
    if let Some(last) = cdf.last_mut() {
        *last = 1.0;
    }
    (cdf, pick)
}

impl<G: Geometry> SceneAccel<G> {
    /// Place every object by its own transform (the identity for the usual
    /// case of geometry that arrives already world-placed) and gather them
    /// under one TLAS. Objects already hold `Arc<Bvh>`, so repeated parts
    /// share a BLAS without any copying — and a re-posed frame rebuilds only
    /// this structure.
    pub(crate) fn build(scene: &Scene<G>) -> Self {
        let instances = scene
            .objects
            .iter()
            .enumerate()
            .filter_map(|(i, obj)| Instance::new(Arc::clone(&obj.bvh), obj.transform.clone(), i))
            .collect();
        let (light_cdf, light_pick_pdf) = light_power_table(&scene.lights);
        Self {
            tlas: Tlas::build(instances),
            light_cdf,
            light_pick_pdf,
        }
    }
}

impl<G: Geometry> Scene<G> {
    /// What the splat cloud, if there is one, adds along `(t_min, t_max)` of
    /// `ray` — radiance emitted and transmittance surviving.
    ///
    /// The identity segment (no radiance, full transmittance) when the scene
    /// carries no cloud, so every caller can add it unconditionally.
    fn splat_segment(&self, ray: &Ray, t_min: f64, t_max: f64) -> SplatSegment {
        match &self.splats {
            Some(bvh) => crate::splats::composite(bvh, ray, t_min, t_max),
            None => SplatSegment::default(),
        }
    }

    /// Closest intersection against objects, ground, and lights.
    fn intersect(&self, accel: &SceneAccel<G>, ray: &Ray) -> Landing {
        let mut best_t = f64::INFINITY;
        let mut landing = Landing::Miss;

        // `1e-7` as the interval floor rather than a post-filter: pushed into
        // the traversal, a surface just behind the one the ray left is still
        // found instead of the whole query being discarded.
        if let Some(found) = self.tlas_hit(accel, ray, 1e-7) {
            best_t = found.hit.t;
            landing = Landing::Surface {
                point: found.hit.point,
                normal: found.hit.normal.into_inner(),
                tangent: found.hit.dpdu,
                material: self.objects[found.payload].material,
            };
        }

        if let Some(g) = &self.ground {
            let d = ray.direction.into_inner();
            if d.z.abs() > 1e-12 {
                let t = (g.z - ray.origin.z) / d.z;
                if t > 1e-6 && t < best_t {
                    best_t = t;
                    landing = Landing::Surface {
                        point: ray.at(t),
                        normal: Vec3::new(0.0, 0.0, 1.0),
                        // The studio sweep is a backdrop, not a machined
                        // face; it has no grain to align to.
                        tangent: None,
                        material: g.material,
                    };
                }
            }
        }

        for (i, l) in self.lights.iter().enumerate() {
            if let Some(t) = l.intersect(ray) {
                if t < best_t {
                    best_t = t;
                    landing = Landing::Light {
                        emission: l.emission,
                        light_index: i,
                        distance: t,
                        point: ray.at(t),
                    };
                }
            }
        }

        landing
    }

    /// Closest geometry hit past `t_min`, in world space.
    fn tlas_hit(&self, accel: &SceneAccel<G>, ray: &Ray, t_min: f64) -> Option<InstanceHit> {
        accel.tlas.trace_closest_range(ray, t_min, f64::INFINITY)
    }

    /// Any-hit occlusion test against geometry only (lights do not occlude).
    ///
    /// A true any-hit traversal: it returns at the first blocker rather than
    /// finding the nearest one and then comparing distance, which is strictly
    /// more work than a shadow ray needs.
    ///
    /// Kept as the fast path for the common case; light sampling goes through
    /// [`Scene::shadow_transmittance`], which can see *through* a pane.
    #[allow(dead_code)]
    fn occluded(&self, accel: &SceneAccel<G>, origin: Point3, dir: Vec3, max_dist: f64) -> bool {
        self.shadow_transmittance(accel, origin, dir, max_dist)
            .is_none()
    }

    /// How much of a light's radiance survives the trip from `origin` along
    /// `dir` to `max_dist` — `None` when the ray is blocked outright.
    ///
    /// The material-blind any-hit test this replaces made a window pane an
    /// opaque wall: NEE found a blocker and returned black, so a room lit
    /// through glass could only be lit by paths that *happened* to refract
    /// into the sun, which at any sane spp is never. That is why the court's
    /// clerestory had to be cut open.
    ///
    /// A thin-walled transmissive sheet is not a blocker, it is a filter. It
    /// has no interior for a ray to travel through and no lateral offset
    /// (see [`Pbr::thin_walled`]), so the shadow ray carries straight on with
    /// its throughput multiplied by the sheet's transmittance. The factor is
    /// exactly the one [`dielectric_eval`]'s thin-walled branch applies to a
    /// BSDF-sampled path — `transmission · (1 − F(cos θ))` — so the two
    /// strategies estimate the same integral and MIS stays consistent. (It is
    /// *not* `(1 − F)²`: this renderer's sheet is a single Fresnel interface
    /// with `R + T = 1`, and a shadow ray that disagreed with the BSDF by a
    /// second factor of `(1 − F)` would double-count under MIS in one
    /// direction and lose energy in the other.)
    ///
    /// Frosted glass is *not* handled: a rough sheet scatters, and pretending
    /// the light arrives along the straight line is only right in the smooth
    /// limit. The transmittance is therefore weighted by the sheet's
    /// specular lobe narrowness — a fully rough pane blocks as before.
    ///
    /// At most [`MAX_SHADOW_SHEETS`] panes are crossed; a shadow ray that
    /// finds more is treated as blocked, which bounds the traversal cost and
    /// keeps a stack of panes from turning into an unbounded loop.
    fn shadow_transmittance(
        &self,
        accel: &SceneAccel<G>,
        origin: Point3,
        dir: Vec3,
        max_dist: f64,
    ) -> Option<[f32; 3]> {
        let limit = max_dist - 1e-6;
        if let Some(g) = &self.ground {
            let d = dir;
            if d.z.abs() > 1e-12 {
                let t = (g.z - origin.z) / d.z;
                if t > 1e-6 && t < limit {
                    return None;
                }
            }
        }

        let ray = Ray::new(origin, dir);
        // The splat cloud shadows by its accumulated opacity: a captured wall
        // is opaque enough to stop the light, a captured net or a wisp of
        // reconstruction dust is not. Grey, because the alphas are grey —
        // a Gaussian's colour is emission, not a filter.
        let splat_tr = self.splat_segment(&ray, 1e-6, limit).transmittance;
        if splat_tr <= 1e-4 {
            return None;
        }
        let mut tr = [splat_tr; 3];
        let mut t0 = 1e-6;
        let mut crossed = 0usize;
        loop {
            let Some(found) = accel.tlas.trace_closest_range(&ray, t0, limit) else {
                return Some(tr);
            };
            if crossed == MAX_SHADOW_SHEETS {
                // More sheets than the cap allows: fall back to opaque.
                return None;
            }
            crossed += 1;
            let m = &self.objects[found.payload].material;
            let Some(sheet) = sheet_transmittance(m, found.hit.normal.into_inner().dot(dir)) else {
                return None;
            };
            tr = mul3(tr, sheet);
            if max3(tr) <= 1e-6 {
                return None;
            }
            t0 = found.hit.t + 1e-6;
            if t0 >= limit {
                return Some(tr);
            }
        }
    }

    /// Next-event estimation: sample *one* area light, drawn from the
    /// accel's power-weighted table, MIS-weighted against the BSDF sampling
    /// strategy.
    ///
    /// One shadow ray per bounce regardless of how many softboxes the rig
    /// has. Dividing the contribution by the pick probability leaves the
    /// estimator unbiased — the mean over many samples matches the old
    /// sample-every-light estimator exactly — and picking by power means the
    /// lights that matter are the ones usually chosen.
    // The shading frame (p, t, b, n) and the outgoing direction are the
    // integrator's hot-loop state; bundling them into a struct just to
    // satisfy the lint would add a copy per light sample.
    #[allow(clippy::too_many_arguments)]
    fn sample_lights(
        &self,
        accel: &SceneAccel<G>,
        p: Point3,
        frame: &Frame,
        wo_local: Vec3,
        m: &Pbr,
        eta: f32,
        lambda_nm: f32,
        rng: &mut Rng,
    ) -> [f32; 3] {
        let Frame { t, b, n } = *frame;
        // The pick draw comes first so the light choice is independent of the
        // position draw on the chosen rectangle.
        let Some((index, pick_pdf)) = accel.pick_light(rng.f64() as f32) else {
            return [0.0; 3];
        };
        let light = &self.lights[index];
        let lp = light.sample(rng.f64(), rng.f64());
        let to_light = lp - p;
        let dist = to_light.norm();
        if dist < 1e-9 {
            return [0.0; 3];
        }
        let wi_world = to_light / dist;
        let ln = light.normal();
        let cos_light = -wi_world.dot(ln);
        if cos_light <= 1e-9 {
            return [0.0; 3];
        }
        let wi_local = to_local(t, b, n, wi_world);
        if wi_local.z <= 0.0 {
            return [0.0; 3];
        }

        let (f, bsdf_pdf) = bsdf_eval(m, wo_local, wi_local, eta, lambda_nm);
        if max3(f) <= 0.0 {
            return [0.0; 3];
        }

        // Solid-angle PDF of the *full* NEE strategy: pick this light, then
        // pick a point on it. The BSDF-hits-a-light branch in `radiance`
        // reconstructs the same product, so MIS stays consistent.
        let light_pdf = pick_pdf * (dist * dist / (cos_light * light.area())) as f32;
        if !light_pdf.is_finite() || light_pdf <= 0.0 {
            return [0.0; 3];
        }

        let Some(tr) = self.shadow_transmittance(accel, p + n * 1e-5, wi_world, dist) else {
            return [0.0; 3];
        };

        let w = power_heuristic(light_pdf, bsdf_pdf);
        scale3(mul3(mul3(f, light.emission), tr), w / light_pdf)
    }

    /// Next-event estimation against the environment, MIS-weighted against
    /// BSDF sampling.
    ///
    /// Only runs for an importance-sampled environment ([`EnvMap`]). The
    /// analytic gradient stays BSDF-only, exactly as before — it is
    /// low-frequency enough that a second strategy buys nothing.
    fn sample_environment(
        &self,
        accel: &SceneAccel<G>,
        p: Point3,
        frame: &Frame,
        wo_local: Vec3,
        m: &Pbr,
        eta: f32,
        lambda_nm: f32,
        rng: &mut Rng,
    ) -> [f32; 3] {
        let Frame { t, b, n } = *frame;
        let Some((wi_world, li, env_pdf)) = self.env.sample(rng.f64(), rng.f64()) else {
            return [0.0; 3];
        };
        if !env_pdf.is_finite() || env_pdf <= 0.0 || max3(li) <= 0.0 {
            return [0.0; 3];
        }
        let wi_local = to_local(t, b, n, wi_world);
        if wi_local.z <= 0.0 {
            return [0.0; 3];
        }
        let (f, bsdf_pdf) = bsdf_eval(m, wo_local, wi_local, eta, lambda_nm);
        if max3(f) <= 0.0 {
            return [0.0; 3];
        }
        // The environment is at infinity: nothing between here and the sky
        // may block, so the shadow ray is unbounded.
        let Some(tr) = self.shadow_transmittance(accel, p + n * 1e-5, wi_world, f64::INFINITY)
        else {
            return [0.0; 3];
        };
        let w = power_heuristic(env_pdf, bsdf_pdf);
        scale3(mul3(mul3(f, li), tr), w / env_pdf)
    }

    /// Next-event estimation against the sun disc, MIS-weighted against BSDF
    /// sampling — the same three-line shape as the area lights, over a cone
    /// at infinity instead of a rectangle at a distance.
    fn sample_sun(
        &self,
        accel: &SceneAccel<G>,
        p: Point3,
        frame: &Frame,
        wo_local: Vec3,
        m: &Pbr,
        eta: f32,
        lambda_nm: f32,
        rng: &mut Rng,
    ) -> [f32; 3] {
        let Some(sun) = &self.sun else {
            return [0.0; 3];
        };
        let Frame { t, b, n } = *frame;
        let (wi_world, li, sun_pdf) = sun.sample(rng.f64(), rng.f64());
        if !sun_pdf.is_finite() || sun_pdf <= 0.0 || max3(li) <= 0.0 {
            return [0.0; 3];
        }
        let wi_local = to_local(t, b, n, wi_world);
        if wi_local.z <= 0.0 {
            return [0.0; 3];
        }
        let (f, bsdf_pdf) = bsdf_eval(m, wo_local, wi_local, eta, lambda_nm);
        if max3(f) <= 0.0 {
            return [0.0; 3];
        }
        // The sun is at infinity, so the shadow ray is unbounded.
        let Some(tr) = self.shadow_transmittance(accel, p + n * 1e-5, wi_world, f64::INFINITY)
        else {
            return [0.0; 3];
        };
        let w = power_heuristic(sun_pdf, bsdf_pdf);
        scale3(mul3(mul3(f, li), tr), w / sun_pdf)
    }
}

// ─── the caustic pass's half of the integrator ────────────────────────────

/// The acceleration structure the caustic pass traces against, built once.
///
/// A thin wrapper so [`crate::caustics`] can hold the TLAS without the
/// integrator's internals leaking out of this module.
pub(crate) struct CausticContext<G> {
    accel: SceneAccel<G>,
}

impl<G: Geometry> CausticContext<G> {
    pub(crate) fn new(scene: &Scene<G>) -> Self {
        Self {
            accel: SceneAccel::build(scene),
        }
    }
}

/// Centre and bounding radius of the scene's *refracting* geometry, or `None`
/// when there is none.
///
/// This is what the caustic pass aims at. Aiming is importance sampling and
/// not a cheat — the emitted power carries the solid angle of the cone — but
/// it is the difference between a caustic in seconds and a caustic never.
pub(crate) fn caustic_bounds<G: Geometry>(scene: &Scene<G>) -> Option<(Point3, f64)> {
    let mut bounds: Option<Aabb> = None;
    for obj in &scene.objects {
        if !crate::caustics::is_caustic_refractor(&obj.material) {
            continue;
        }
        let Some(inst) = Instance::new(Arc::clone(&obj.bvh), obj.transform.clone(), 0) else {
            continue;
        };
        let b = inst.world_aabb();
        match &mut bounds {
            Some(acc) => acc.include(&b),
            None => bounds = Some(b),
        }
    }
    let b = bounds?;
    let c = b.center();
    let r: f64 = 0.5 * (b.max - b.min).norm();
    if !r.is_finite() || r <= 0.0 {
        return None;
    }
    Some((c, r))
}

/// Follow one photon from a light until it lands on a diffuse surface.
///
/// Returns the landing point, the surface normal there and the power the
/// photon still carries — or `None` if it was absorbed, escaped, or reached a
/// diffuse surface without ever having been refracted by a *solid*.
///
/// That last condition is the whole of the double-counting rule: light that
/// only ever passed through a thin pane already belongs to next-event
/// estimation (see `sheet_transmittance`), so a photon carrying it is dropped
/// here rather than added twice.
///
/// The BSDF is the camera path's BSDF, unmodified. It can be, because
/// [`dielectric_eval`]'s transmission branch already cancels Walter's `η_t²`
/// against the `1/η²` radiance compression — so what it returns is the
/// symmetric quantity a photon wants, and importance transport needs no
/// correction factor here.
pub(crate) fn trace_photon<G: Geometry>(
    scene: &Scene<G>,
    ctx: &CausticContext<G>,
    origin: Point3,
    dir: Vec3,
    power: [f32; 3],
    max_bounces: u32,
    rng: &mut Rng,
) -> Option<(Point3, Vec3, [f32; 3])> {
    let accel = &ctx.accel;
    let mut ray = Ray::new(origin, dir);
    let mut power = power;
    let mut lambda_nm: Option<f64> = None;
    let mut medium: Option<Pbr> = None;
    let mut refracted_by_a_solid = false;

    for _ in 0..max_bounces {
        let landing = scene.intersect(accel, &ray);
        // Absorb along the segment just travelled, if it was inside glass.
        if let Some(med) = &medium {
            let sigma = med.extinction();
            if let Landing::Surface { point, .. } = &landing
                && max3(sigma) > 0.0
            {
                {
                    let d = (*point - ray.origin).norm() as f32;
                    power = mul3(
                        power,
                        [
                            (-sigma[0] * d).exp(),
                            (-sigma[1] * d).exp(),
                            (-sigma[2] * d).exp(),
                        ],
                    );
                }
            }
        }
        let Landing::Surface {
            point,
            normal,
            tangent,
            material,
        } = landing
        else {
            // Off into the sky, or onto a light's back: no deposit.
            return None;
        };

        let wo_world = -ray.direction.into_inner();
        let entering = normal.dot(wo_world) >= 0.0;
        let n = if normal.dot(wo_world) < 0.0 {
            -normal
        } else {
            normal
        };

        if material.transmission <= 0.0 {
            // A diffuse receiver. Deposit only if the light got here the way
            // the path tracer cannot follow.
            return if refracted_by_a_solid && max3(power) > 0.0 {
                Some((point, n, power))
            } else {
                None
            };
        }

        if lambda_nm.is_none() && material.is_dispersive() {
            let nm = crate::spectrum::sample_lambda_nm(rng.f64());
            lambda_nm = Some(nm);
            power = mul3(power, crate::spectrum::hero_weight(nm));
        }
        let n_glass = material.index_at(lambda_nm).max(1e-3);
        let eta = if material.thin_walled || entering {
            n_glass
        } else {
            1.0 / n_glass
        };
        let hero = lambda_nm.unwrap_or(0.0) as f32;

        let frame = shading_frame(n, tangent);
        let wo_local = to_local(frame.t, frame.b, n, wo_world);
        if wo_local.z <= 0.0 {
            return None;
        }
        let Some(Sampled::Surface(wi_local, f, pdf)) =
            bsdf_sample(&material, wo_local, eta, hero, rng)
        else {
            return None;
        };
        power = mul3(power, scale3(f, 1.0 / pdf));
        if max3(power) <= 1e-12 {
            return None;
        }

        let transmitted = wi_local.z < 0.0;
        if transmitted && !material.thin_walled {
            refracted_by_a_solid = true;
            medium = if entering { Some(material) } else { None };
        }
        let wi_world = to_world(frame.t, frame.b, n, wi_local);
        let offset = if transmitted { -n } else { n };
        ray = Ray::new(point + offset * 1e-5, wi_world);
    }
    None
}

// ─── integrator ───────────────────────────────────────────────────────────

/// What the primary ray of a path found at depth 0.
///
/// Recorded for the denoiser's guide buffers. `depth == 0.0` means the primary
/// ray escaped the scene — the sentinel for "background", which the filter
/// refuses to mix with any surface.
#[derive(Debug, Clone, Copy, Default)]
struct Primary {
    /// Whether the primary ray hit geometry or an emitter (drives alpha).
    hit: bool,
    /// Face-forwarded world normal at the first hit.
    normal: [f32; 3],
    /// Distance from the camera to the first hit; 0 for a miss.
    depth: f32,
    /// Surface colour at the first hit, for albedo demodulation.
    albedo: [f32; 3],
}

/// Trace one path and return its radiance estimate, plus what its primary ray
/// landed on (for alpha and for the denoiser's guide buffers).
#[allow(clippy::too_many_arguments)]
fn radiance<G: Geometry>(
    scene: &Scene<G>,
    accel: &SceneAccel<G>,
    opts: &PathTraceOptions,
    caustics: Option<&CausticMap>,
    clamp_scale: Option<f32>,
    ray: Ray,
    rng: &mut Rng,
) -> ([f32; 3], Primary) {
    let origin = ray.origin;
    let mut primary = Primary::default();
    let mut l = [0.0f32; 3];
    let mut throughput = [1.0f32; 3];
    let mut ray = ray;
    // The previous bounce was sampled from a lobe with this PDF; used to MIS
    // against light sampling when the new ray lands on an emitter.
    let mut prev_bsdf_pdf = 0.0f32;
    let mut specular_chain = true;
    // The path's hero wavelength, in nanometres. `None` until the path meets
    // a material whose index actually depends on it — an RGB path stays RGB,
    // draws no extra random number, and renders bit-identically to what it
    // did before dispersion existed.
    let mut lambda_nm: Option<f64> = None;
    // The medium the path is currently inside, for Beer–Lambert absorption.
    // One slot, not a stack: this tracks a ray inside *a* solid, which is
    // every glass in these scenes. Nested dielectrics (a bubble in glass, ice
    // in a drink) would need a stack and would get the outer medium wrong
    // here; that is the documented limit.
    let mut medium: Option<Pbr> = None;

    for depth in 0..opts.max_depth {
        let landing = scene.intersect(accel, &ray);
        // The splat cloud along this segment, composited front to back and
        // stopped at whatever the segment ran into. A captured cloud is a
        // radiance field with its lighting already baked in, so it is added
        // as emission and its accumulated opacity veils everything past it:
        // `L += throughput · C`, then `throughput *= T`. Doing it here — for
        // the camera ray and for every bounce ray alike — is what makes the
        // cloud an *environment with depth*: a bounce ray that finds no
        // analytic surface comes back with the room's own colour, so the
        // marble is lit by the garage it is standing in.
        if scene.splats.is_some()
            && !(depth == 0 && !opts.show_background && matches!(landing, Landing::Miss))
        {
            let t_hit = match &landing {
                Landing::Surface { point, .. } => (*point - ray.origin).norm(),
                Landing::Light { distance, .. } => *distance,
                Landing::Miss => f64::INFINITY,
            };
            let seg = scene.splat_segment(&ray, 1e-6, t_hit);
            if max3(seg.radiance) > 0.0 {
                l = add3(l, mul3(throughput, seg.radiance));
            }
            if depth == 0 && seg.transmittance < 0.5 {
                // The cloud, not the background, is what this pixel shows.
                primary.hit = true;
                primary.albedo = seg.radiance;
            }
            throughput = scale3(throughput, seg.transmittance);
            if max3(throughput) <= 1e-5 {
                break;
            }
        }
        // Absorb along the segment just travelled, if it was inside glass.
        if let Some(med) = &medium {
            let sigma = med.extinction();
            if max3(sigma) > 0.0 {
                let d = match &landing {
                    Landing::Surface { point, .. } => (*point - ray.origin).norm() as f32,
                    Landing::Light { distance, .. } => *distance as f32,
                    Landing::Miss => 0.0,
                };
                if d > 0.0 {
                    throughput = mul3(
                        throughput,
                        [
                            (-sigma[0] * d).exp(),
                            (-sigma[1] * d).exp(),
                            (-sigma[2] * d).exp(),
                        ],
                    );
                }
            }
        }
        match landing {
            Landing::Miss => {
                let dir = ray.direction.into_inner();
                let env = scene.env.radiance(dir);
                if depth == 0 && !opts.show_background {
                    // Leave the backdrop clear; still no contribution.
                    break;
                }
                // MIS against environment NEE, which could also have found
                // this direction. A specular chain (including the primary
                // ray) had no other strategy, so it takes full weight.
                let w = if specular_chain || !scene.env.is_importance_sampled() {
                    1.0
                } else {
                    power_heuristic(prev_bsdf_pdf, scene.env.pdf(dir))
                };
                l = add3(l, scale3(mul3(throughput, env), w));
                // The sun disc, if this ray happened to land in it. NEE
                // samples the same cone, so the two strategies share the
                // direction under the balance heuristic; a specular chain
                // (the primary ray included) had no other way to find it.
                if let Some(sun) = &scene.sun {
                    let li = sun.radiance_in(dir);
                    if max3(li) > 0.0 {
                        let ws = if specular_chain {
                            1.0
                        } else {
                            power_heuristic(prev_bsdf_pdf, sun.pdf(dir))
                        };
                        l = add3(l, scale3(mul3(throughput, li), ws));
                    }
                }
                break;
            }
            Landing::Light {
                emission,
                light_index,
                distance,
                point,
            } => {
                if depth == 0 {
                    // An emitter seen directly. It is noise-free by
                    // construction, but it still needs a guide entry so the
                    // filter treats it as its own surface rather than as
                    // background.
                    primary.hit = true;
                    primary.depth = distance as f32;
                    primary.normal = vec_to_f32(scene.lights[light_index].normal());
                    primary.albedo = [1.0; 3];
                }
                let w = if specular_chain {
                    1.0
                } else {
                    // MIS against the NEE strategy that could also have found
                    // this light.
                    let light = &scene.lights[light_index];
                    let ln = light.normal();
                    let cos_light = (-ray.direction.into_inner().dot(ln)).max(1e-9);
                    let light_pdf = accel.light_pick_pdf(light_index)
                        * (distance * distance / (cos_light * light.area())) as f32;
                    let _ = point;
                    power_heuristic(prev_bsdf_pdf, light_pdf)
                };
                l = add3(l, scale3(mul3(throughput, emission), w));
                break;
            }
            Landing::Surface {
                point,
                normal,
                tangent,
                material,
            } => {
                let wo_world = -ray.direction.into_inner();
                // Which side of the *geometric* normal the ray arrived on is
                // the whole of the inside/outside bookkeeping: a front face is
                // an entry, a back face an exit. Read before the face-forward
                // that follows destroys the distinction.
                let entering = normal.dot(wo_world) >= 0.0;
                // Face-forward: interior faces (bore walls) must shade right.
                let n = if normal.dot(wo_world) < 0.0 {
                    -normal
                } else {
                    normal
                };
                // A dispersive material turns the path monochromatic, once.
                // The draw is inside the `if` so a scene without dispersion
                // consumes the RNG stream exactly as it always has.
                if lambda_nm.is_none() && material.is_dispersive() {
                    let nm = crate::spectrum::sample_lambda_nm(rng.f64());
                    lambda_nm = Some(nm);
                    throughput = mul3(throughput, crate::spectrum::hero_weight(nm));
                }
                // `eta` is n_transmitted / n_incident for this crossing.
                let eta = if material.transmission > 0.0 {
                    let n_glass = material.index_at(lambda_nm).max(1e-3);
                    if material.thin_walled || entering {
                        n_glass
                    } else {
                        1.0 / n_glass
                    }
                } else {
                    1.0
                };
                if depth == 0 {
                    primary.hit = true;
                    primary.depth = (point - origin).norm() as f32;
                    primary.normal = vec_to_f32(n);
                    primary.albedo = material.denoise_albedo();
                }
                // The hero wavelength as the BSDF wants it: `0` for an RGB
                // path, which is the sentinel every lobe reads as "achromatic".
                let hero = lambda_nm.unwrap_or(0.0) as f32;
                let frame = shading_frame(n, tangent);
                let wo_local = to_local(frame.t, frame.b, n, wo_world);
                if wo_local.z <= 0.0 {
                    break;
                }

                l = add3(l, mul3(throughput, material.emissive));

                // Next-event estimation: explicit lights, plus the
                // environment when it is importance-sampled.
                let direct = add3(
                    add3(
                        scene.sample_lights(
                            accel, point, &frame, wo_local, &material, eta, hero, rng,
                        ),
                        scene.sample_environment(
                            accel, point, &frame, wo_local, &material, eta, hero, rng,
                        ),
                    ),
                    scene.sample_sun(accel, point, &frame, wo_local, &material, eta, hero, rng),
                );
                let direct = if depth > 0 {
                    // The relative clamp, when armed, takes over from the
                    // absolute one; otherwise nothing about this changed.
                    match (clamp_scale, opts.firefly_clamp) {
                        (Some(c), _) | (None, Some(c)) => {
                            [direct[0].min(c), direct[1].min(c), direct[2].min(c)]
                        }
                        (None, None) => direct,
                    }
                } else {
                    direct
                };
                l = add3(l, mul3(throughput, direct));

                // The caustic map's share: light that arrived here by
                // refraction through a solid, which next-event estimation
                // could not have found and which the shadow rays above
                // therefore did not count. Added *outside* the firefly clamp
                // — it is a density estimate with no long tail, and clamping
                // it is indistinguishable from deleting the caustic.
                if let Some(map) = caustics.filter(|_| material.transmission <= 0.0) {
                    let rho = material.diffuse_albedo();
                    if max3(rho) > 0.0 {
                        let e = map.irradiance(point, n);
                        if max3(e) > 0.0 {
                            let k = 1.0 / std::f32::consts::PI;
                            l = add3(l, mul3(throughput, scale3(mul3(rho, e), k)));
                        }
                    }
                }

                // Continue the path.
                let Some(sampled) = bsdf_sample(&material, wo_local, eta, hero, rng) else {
                    break;
                };
                let (wi_local, f, pdf) = match sampled {
                    Sampled::Surface(wi, f, pdf) => (wi, f, pdf),
                    Sampled::Subsurface(entry_weight) => {
                        // The path leaves the surface entirely: it goes into
                        // the object, walks, and comes back out somewhere
                        // else. Everything after this is about the *exit*.
                        throughput = mul3(throughput, entry_weight);
                        let Some(exit) = subsurface_walk(&material, point, n, rng, |p, d| {
                            match scene.intersect(accel, &Ray::new(p, d)) {
                                Landing::Surface { point, normal, .. } => {
                                    Some(((point - p).norm(), normal))
                                }
                                _ => None,
                            }
                        }) else {
                            break;
                        };
                        throughput = mul3(throughput, exit.weight);
                        // Out through the boundary, cosine-distributed: the
                        // index-matched exit the inversion was fitted with,
                        // whose f/pdf is exactly 1.
                        let (t_ax, b_ax) = onb(exit.normal);
                        let c = cosine_hemisphere(rng.f64(), rng.f64());
                        let wi_world = to_world(t_ax, b_ax, exit.normal, c);
                        ray = Ray::new(exit.point + exit.normal * 1e-5, wi_world);
                        // No NEE strategy found this direction — there was no
                        // surface event at the exit to sample lights from — so
                        // an emitter downstream takes full MIS weight, which
                        // is what a specular chain means here.
                        specular_chain = true;
                        prev_bsdf_pdf = 0.0;
                        if depth >= opts.rr_start {
                            let q = max3(throughput).clamp(0.0, 0.95);
                            if (rng.f64() as f32) > q {
                                break;
                            }
                            throughput = scale3(throughput, 1.0 / q);
                        }
                        if max3(throughput) <= 1e-5 {
                            break;
                        }
                        continue;
                    }
                };
                throughput = mul3(throughput, scale3(f, 1.0 / pdf));
                prev_bsdf_pdf = pdf;
                specular_chain = false;

                // A transmitted ray leaves on the far side, so it is offset
                // the other way — and, for a solid, it changes which medium
                // the path is in.
                let transmitted = wi_local.z < 0.0;
                if transmitted && !material.thin_walled {
                    medium = if entering { Some(material) } else { None };
                }
                let wi_world = to_world(frame.t, frame.b, n, wi_local);
                let offset = if transmitted { -n } else { n };
                ray = Ray::new(point + offset * 1e-5, wi_world);

                // Russian roulette.
                if depth >= opts.rr_start {
                    let q = max3(throughput).clamp(0.0, 0.95);
                    if (rng.f64() as f32) > q {
                        break;
                    }
                    throughput = scale3(throughput, 1.0 / q);
                }
                if max3(throughput) <= 1e-5 {
                    break;
                }
            }
        }
    }

    (l, primary)
}

#[inline]
fn vec_to_f32(v: Vec3) -> [f32; 3] {
    [v.x as f32, v.y as f32, v.z as f32]
}

/// A rendered frame in linear space.
pub struct Film {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Linear RGB radiance, 3 floats per pixel, row-major top-to-bottom.
    pub rgb: Vec<f32>,
    /// Coverage in 0..1, one float per pixel.
    pub alpha: Vec<f32>,
    /// World normal at each pixel's first hit, 3 floats per pixel. Zero for
    /// background pixels. Guide buffer for [`denoise`].
    pub normal: Vec<f32>,
    /// Distance from the camera to each pixel's first hit, one float per
    /// pixel. **Zero means the primary ray escaped** — the background
    /// sentinel. Guide buffer for [`denoise`].
    pub depth: Vec<f32>,
    /// Surface colour at each pixel's first hit, 3 floats per pixel. Divided
    /// out before filtering and multiplied back after, so [`denoise`] only
    /// ever blurs illumination.
    pub albedo: Vec<f32>,
    /// Estimated variance of each pixel's mean radiance luminance, one float
    /// per pixel — the Monte Carlo estimator's own error bar.
    ///
    /// [`denoise`] scales its luminance edge-stopping tolerance by this, which
    /// is what lets the filter tell "this neighbour is genuinely a different
    /// brightness" from "this pixel is a noise spike". Without it a firefly
    /// rejects every neighbour and survives the filter untouched.
    pub variance: Vec<f32>,
}

/// One pixel's worth of the integrator, writing into the row slices the film
/// keeps for that scanline.
///
/// Factored out of [`render`] so [`render_into`] can drive exactly the same
/// code on a subset of pixels. The RNG seed is a pure function of the pixel
/// coordinates and `opts.seed`, which is what makes a masked pass reproduce
/// the full render's pixels bit for bit — and what makes either of them
/// independent of how rayon happens to schedule the rows.
struct PixelOut<'a> {
    rgb: &'a mut [f32],
    alpha: &'a mut [f32],
    normal: &'a mut [f32],
    depth: &'a mut [f32],
    albedo: &'a mut [f32],
    variance: &'a mut [f32],
}

#[allow(clippy::too_many_arguments)]
fn trace_pixel<G: Geometry>(
    scene: &Scene<G>,
    accel: &SceneAccel<G>,
    caustics: Option<&CausticMap>,
    cam: &Camera,
    opts: &PathTraceOptions,
    width: u32,
    height: u32,
    px: usize,
    py: usize,
    out: &mut PixelOut<'_>,
) -> u32 {
    let aspect = width as f64 / height as f64;
    let spp = opts.spp.max(1);
    let mut rng =
        Rng::new(opts.seed ^ ((py as u64) << 32) ^ (px as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut acc = [0.0f32; 3];
    let mut cov = 0.0f32;
    // Running sums for the estimator's own variance.
    let mut lsum = 0.0f32;
    let mut lsum2 = 0.0f32;
    // Cranley-Patterson rotations for the four camera dimensions, drawn once
    // per pixel. The low-discrepancy point set below is the *same* for every
    // pixel; rotating it by a per-pixel random offset keeps each pixel's
    // stratification intact while decorrelating neighbours, so the residual
    // error looks like noise rather than a repeating pattern locked to the
    // pixel grid. Drawing them from the existing PCG is what keeps seed
    // determinism: no global state, no thread-dependent order.
    let rot = [rng.f64(), rng.f64(), rng.f64(), rng.f64()];

    // Sample in batches so the estimator can be asked, between batches,
    // whether it has already resolved this pixel. `traced` is the count
    // actually spent, which is <= spp under adaptive sampling.
    let mut traced = 0u32;
    'batches: while traced < spp {
        let batch = ADAPTIVE_BATCH.min(spp - traced);
        for k in 0..batch {
            let s = traced + k;
            // Pixel jitter and lens position come from a 4D Halton set rotated
            // into this pixel's frame, not from four fresh uniforms. Four
            // independent uniforms can clump — at low sample counts a purely
            // random jitter leaves visibly uneven coverage of the pixel
            // footprint, and that shows up as extra aliasing on every
            // silhouette. A low-discrepancy set covers the square evenly by
            // construction.
            //
            // Halton rather than Hammersley: Hammersley's first dimension is
            // `s / N`, which needs the final sample count up front. Adaptive
            // sampling does not know it, and a set that changes shape when the
            // loop stops early is worse than a slightly weaker set that is
            // correct at every prefix.
            //
            // The uniforms still go through the reconstruction filter's warp, so
            // the plain mean below is the filtered estimate exactly as before.
            let jx = 0.5
                + opts
                    .filter
                    .warp(cp_rotate(radical_inverse::<2>(s as u64), rot[0]));
            let jy = 0.5
                + opts
                    .filter
                    .warp(cp_rotate(radical_inverse::<3>(s as u64), rot[1]));
            let sx = 2.0 * ((px as f64 + jx) / width as f64) - 1.0;
            let sy = 1.0 - 2.0 * ((py as f64 + jy) / height as f64);
            let (lu, lv) = concentric_disc(
                cp_rotate(radical_inverse::<5>(s as u64), rot[2]),
                cp_rotate(radical_inverse::<7>(s as u64), rot[3]),
            );

            let ray = cam.ray(sx, sy, aspect, lu, lv);
            // The relative clamp's threshold, from what this pixel has measured
            // so far. It is deliberately a *running* mean and not a two-pass
            // estimate: a pixel is its own scale, and one pass is what keeps the
            // integrator streaming.
            let clamp_scale = opts.firefly_clamp_relative.and_then(|k| {
                if s >= opts.firefly_clamp_warmup && s > 0 {
                    Some((k * lsum / s as f32).max(1e-6))
                } else {
                    None
                }
            });
            let (l, primary) = radiance(scene, accel, opts, caustics, clamp_scale, ray, &mut rng);
            acc = add3(acc, l);
            let ls = luminance(l);
            lsum += ls;
            lsum2 += ls * ls;
            if primary.hit {
                cov += 1.0;
            }
            if s == 0 {
                // Guide buffers come from one primary ray, not an average:
                // averaging normals and depths across samples would soften
                // exactly the silhouettes the edge-stopping weights exist to
                // protect.
                out.normal[px * 3] = primary.normal[0];
                out.normal[px * 3 + 1] = primary.normal[1];
                out.normal[px * 3 + 2] = primary.normal[2];
                out.depth[px] = primary.depth;
                out.albedo[px * 3] = primary.albedo[0];
                out.albedo[px * 3 + 1] = primary.albedo[1];
                out.albedo[px * 3 + 2] = primary.albedo[2];
            }
        }
        traced += batch;

        // Stop once the estimator's own error bar says the remaining samples
        // cannot move this pixel by anything a viewer could see. The mean kept
        // below is still the unbiased mean of the samples actually taken, so
        // stopping early costs precision, never accuracy. The floor is
        // non-negotiable: a pixel that happened to draw several near-equal
        // samples early would otherwise report a tiny variance and quit while
        // genuinely unconverged.
        if opts.adaptive && traced >= ADAPTIVE_FLOOR.min(spp) && traced < spp {
            let n = traced as f32;
            let mean = lsum / n;
            // The clamp is load-bearing, not defensive: once the samples agree
            // closely, `lsum2 / n` and `mean * mean` cancel to within f32
            // rounding and can land just below zero, which would put a NaN
            // through the sqrt below — and a NaN compares false, so the pixel
            // would never converge.
            let sample_var = (lsum2 / n - mean * mean).max(0.0) * n / (n - 1.0);
            // Half-width of the 95% confidence interval on the mean.
            let ci = 1.96 * (sample_var / n).sqrt();
            if ci <= ADAPTIVE_TOL * (mean + ADAPTIVE_LUM_FLOOR) {
                break 'batches;
            }
        }
    }

    let inv = 1.0 / traced as f32;
    out.rgb[px * 3] = acc[0] * inv;
    out.rgb[px * 3 + 1] = acc[1] * inv;
    out.rgb[px * 3 + 2] = acc[2] * inv;
    out.alpha[px] = cov * inv;
    // Variance of the *mean*: sample variance / n. A single sample carries
    // no information about its own spread, so fall back to the estimate
    // itself as a scale.
    out.variance[px] = if traced > 1 {
        let n = traced as f32;
        let mean = lsum * inv;
        let sample_var = (lsum2 * inv - mean * mean).max(0.0) * n / (n - 1.0);
        sample_var / n
    } else {
        let mean = lsum;
        mean * mean
    };
    traced
}

/// Render `scene` from `cam` into a linear-space [`Film`].
///
/// Scanlines are traced in parallel. Each pixel's RNG is seeded from its
/// coordinates and the option seed, so output is deterministic and
/// independent of thread scheduling.
///
/// When [`PathTraceOptions::denoise`] is set (the default), the film is run
/// through [`denoise`] before returning. Pass `denoise: false` for a
/// reference render.
pub fn render<G: Geometry + Send + Sync>(
    scene: &Scene<G>,
    cam: &Camera,
    width: u32,
    height: u32,
    opts: &PathTraceOptions,
) -> Film {
    render_with_caustics(scene, cam, width, height, opts, None)
}

/// [`render`], plus a caustic map read as direct light at every diffuse hit.
///
/// The map is built once by [`crate::caustics::trace`] and handed in here.
/// It is a separate argument rather than a field on [`Scene`] or
/// [`PathTraceOptions`] because it is neither: the scene does not own it (it
/// is derived from the scene) and the options are `Copy`.
///
/// Passing `None` is exactly [`render`], to the bit.
pub fn render_with_caustics<G: Geometry + Send + Sync>(
    scene: &Scene<G>,
    cam: &Camera,
    width: u32,
    height: u32,
    opts: &PathTraceOptions,
    caustics: Option<&CausticMap>,
) -> Film {
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(not(target_arch = "wasm32"))]
    use rayon::prelude::*;

    // One TLAS for the whole frame: every ray, primary and shadow, traverses
    // it instead of scanning `scene.objects` linearly.
    let accel = SceneAccel::build(scene);

    let mut rgb = vec![0.0f32; (width * height * 3) as usize];
    let mut alpha = vec![0.0f32; (width * height) as usize];
    let mut normal = vec![0.0f32; (width * height * 3) as usize];
    let mut depth = vec![0.0f32; (width * height) as usize];
    let mut albedo = vec![0.0f32; (width * height * 3) as usize];
    let mut variance = vec![0.0f32; (width * height) as usize];

    let w3 = width as usize * 3;
    let w1 = width as usize;
    film_rows!(rgb, w3)
        .zip(film_rows!(alpha, w1))
        .zip(film_rows!(normal, w3))
        .zip(film_rows!(depth, w1))
        .zip(film_rows!(albedo, w3))
        .zip(film_rows!(variance, w1))
        .enumerate()
        .for_each(|(py, (((((row, arow), nrow), drow), brow), vrow))| {
            let mut out = PixelOut {
                rgb: row,
                alpha: arow,
                normal: nrow,
                depth: drow,
                albedo: brow,
                variance: vrow,
            };
            for px in 0..width as usize {
                let _ = trace_pixel(
                    scene, &accel, caustics, cam, opts, width, height, px, py, &mut out,
                );
            }
        });

    let mut film = Film {
        width,
        height,
        rgb,
        alpha,
        normal,
        depth,
        albedo,
        variance,
    };
    if opts.denoise {
        denoise(&mut film, opts);
    }
    film
}

/// Re-render only the pixels inside `rects`, leaving the rest of `film`
/// exactly as it was.
///
/// Each rect is `[x, y, w, h]` in pixels, top-left origin, and is clipped to
/// the film. A pixel inside the union is traced with the same seed, the same
/// sample sequence and the same integrator [`render`] would have given it, so
/// the result is *bit-identical* to the corresponding pixels of a full render
/// with the same options — which is the whole point: a caller can re-trace the
/// region under a moving widget, or a tile the user is zoomed into, and drop
/// the result straight into the frame it already has without a seam.
///
/// Rows within each rect are traced in parallel. Overlapping rects simply
/// trace their shared pixels more than once, to the same values.
///
/// Unlike [`render`] this never denoises. The à-trous filter reads a
/// neighbourhood well outside any rect, so filtering a masked pass would blend
/// fresh radiance into stale and put a visible seam at the rect's edge; a
/// caller that wants a filtered frame runs [`denoise`] over the whole film
/// once the patches are in. `opts.denoise` is therefore ignored here, and
/// comparing against a reference means comparing against a `denoise: false`
/// render.
///
/// The film must already be the size the camera is being sampled at —
/// `film.width` and `film.height` are the resolution, not `rects`.
pub fn render_into<G: Geometry + Send + Sync>(
    scene: &Scene<G>,
    cam: &Camera,
    film: &mut Film,
    opts: &PathTraceOptions,
    rects: &[[u32; 4]],
) {
    render_into_with_caustics(scene, cam, film, opts, rects, None)
}

/// [`render_into`] with a caustic map, the patch-render counterpart of
/// [`render_with_caustics`].
#[allow(clippy::too_many_arguments)]
pub fn render_into_with_caustics<G: Geometry + Send + Sync>(
    scene: &Scene<G>,
    cam: &Camera,
    film: &mut Film,
    opts: &PathTraceOptions,
    rects: &[[u32; 4]],
    caustics: Option<&CausticMap>,
) {
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(not(target_arch = "wasm32"))]
    use rayon::prelude::*;

    let (width, height) = (film.width, film.height);
    if width == 0 || height == 0 {
        return;
    }
    let accel = SceneAccel::build(scene);
    let w3 = width as usize * 3;
    let w1 = width as usize;

    for r in rects {
        // Clip to the film. A rect that starts past the edge, or is empty,
        // contributes nothing rather than panicking on a caller's arithmetic.
        let x0 = r[0].min(width) as usize;
        let y0 = r[1].min(height) as usize;
        let x1 = r[0].saturating_add(r[2]).min(width) as usize;
        let y1 = r[1].saturating_add(r[3]).min(height) as usize;
        if x0 >= x1 || y0 >= y1 {
            continue;
        }

        // Row-chunked so every parallel task owns a disjoint slice of each
        // buffer; the columns outside the rect are simply never written.
        film_rows!(film.rgb, w3)
            .zip(film_rows!(film.alpha, w1))
            .zip(film_rows!(film.normal, w3))
            .zip(film_rows!(film.depth, w1))
            .zip(film_rows!(film.albedo, w3))
            .zip(film_rows!(film.variance, w1))
            .enumerate()
            .skip(y0)
            .take(y1 - y0)
            .for_each(|(py, (((((row, arow), nrow), drow), brow), vrow))| {
                let mut out = PixelOut {
                    rgb: row,
                    alpha: arow,
                    normal: nrow,
                    depth: drow,
                    albedo: brow,
                    variance: vrow,
                };
                for px in x0..x1 {
                    let _ = trace_pixel(
                        scene, &accel, caustics, cam, opts, width, height, px, py, &mut out,
                    );
                }
            });
    }
}

impl Film {
    /// A black film of `width` x `height`, with every guide buffer zeroed.
    ///
    /// [`render`] allocates its own; this is for the caller who holds one
    /// frame and keeps patching it with [`render_into`].
    pub fn new(width: u32, height: u32) -> Self {
        let n = (width as usize) * (height as usize);
        Self {
            width,
            height,
            rgb: vec![0.0; n * 3],
            alpha: vec![0.0; n],
            normal: vec![0.0; n * 3],
            depth: vec![0.0; n],
            albedo: vec![0.0; n * 3],
            variance: vec![0.0; n],
        }
    }
}

// ─── denoising ────────────────────────────────────────────────────────────

/// 5×5 separable B3-spline (cubic) kernel, `[1 4 6 4 1] / 16`.
const B3_SPLINE: [f32; 5] = [1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0];

/// Floor on the demodulation divisor.
///
/// Dividing by a near-black albedo would turn a dark surface's illumination
/// into enormous numbers, and any filtering error there comes back multiplied.
/// Clamping trades a little residual colour-blurring on very dark materials
/// for numerical sanity.
const DEMOD_FLOOR: f32 = 0.05;

/// Edge-aware à-trous wavelet denoiser (Dammertz et al., EGSR 2010).
///
/// Filters the film's linear radiance in place, guided by the normal, depth,
/// and albedo buffers that [`render`] records from each pixel's primary ray.
///
/// The algorithm is a sequence of 5×5 B3-spline convolutions whose taps are
/// spread by a doubling stride ("holes" — *à trous*), each tap weighted by how
/// well the neighbour matches the centre pixel's normal, depth, and
/// illumination. That reaches a wide footprint in a few passes while refusing
/// to average across geometric or shading discontinuities.
///
/// Two properties are worth naming, because the tests pin them:
///
/// - **Illumination only.** Radiance is divided by albedo before filtering and
///   multiplied back afterwards, so a part's colour is never blurred into its
///   neighbour's — only the Monte Carlo noise in the lighting is smoothed.
/// - **Background is inviolable.** A pixel whose primary ray escaped
///   (`depth == 0`) is passed through untouched, and no surface pixel ever
///   accepts a tap from one. Silhouettes against the backdrop stay exactly as
///   sharp as the path tracer drew them.
///
/// This is a post-process: it consumes no random numbers and never touches the
/// integrator, so a reference render is exactly the un-denoised film.
///
/// Only the `denoise_iters` and `sigma_*` fields of `opts` are read. Calling
/// this *is* the request to filter, so [`PathTraceOptions::denoise`] is the
/// caller's gate — as [`render`] uses it — and is deliberately ignored here.
pub fn denoise(film: &mut Film, opts: &PathTraceOptions) {
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(not(target_arch = "wasm32"))]
    use rayon::prelude::*;

    let w = film.width as usize;
    let h = film.height as usize;
    let n = w * h;
    if n == 0 || opts.denoise_iters == 0 {
        return;
    }

    // Demodulate: work on illumination = radiance / albedo.
    let mut illum = vec![0.0f32; n * 3];
    let mut var = vec![0.0f32; n];
    for i in 0..n {
        for c in 0..3 {
            let a = film.albedo[i * 3 + c].max(DEMOD_FLOOR);
            illum[i * 3 + c] = film.rgb[i * 3 + c] / a;
        }
        // Variance was measured on radiance; demodulation scales it by the
        // square of the (scalar) albedo it divided through.
        let la = luminance([
            film.albedo[i * 3].max(DEMOD_FLOOR),
            film.albedo[i * 3 + 1].max(DEMOD_FLOOR),
            film.albedo[i * 3 + 2].max(DEMOD_FLOOR),
        ])
        .max(DEMOD_FLOOR);
        var[i] = film.variance[i] / (la * la);
    }

    // Prefilter the variance estimate with a 3×3 box. The per-pixel estimate
    // is itself noisy at low sample counts, and a noisy error bar makes the
    // luminance weight jitter between "trust" and "reject" from pixel to
    // pixel.
    {
        let mut smooth = var.clone();
        for y in 0..h {
            for x in 0..w {
                let mut s = 0.0f32;
                let mut k = 0.0f32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let (qx, qy) = (x as i32 + dx, y as i32 + dy);
                        if qx < 0 || qy < 0 || qx >= w as i32 || qy >= h as i32 {
                            continue;
                        }
                        let q = qy as usize * w + qx as usize;
                        if film.depth[q] <= 0.0 {
                            continue;
                        }
                        s += var[q];
                        k += 1.0;
                    }
                }
                if k > 0.0 {
                    smooth[y * w + x] = s / k;
                }
            }
        }
        var = smooth;
    }

    let sigma_n2 = (opts.sigma_normal.max(1e-4)).powi(2);
    let mut scratch = illum.clone();
    let mut var_scratch = var.clone();
    let g_depth = &film.depth;
    let g_normal = &film.normal;

    for it in 0..opts.denoise_iters {
        let stride = 1usize << it;
        // Dammertz shrinks a *fixed* illumination tolerance as the footprint
        // grows. Here the tolerance is already scaled by the filtered variance,
        // which shrinks on its own as the estimate gets cleaner, so shrinking
        // sigma too would penalise the wide passes twice and they would reject
        // every tap. Measured: with the extra 2^-i, iterations past the first
        // bought nothing at all.
        let sigma_l = opts.sigma_lum.max(1e-6);
        let sigma_z = opts.sigma_depth.max(1e-6) * stride as f32;

        film_rows!(scratch, w * 3)
            .zip(film_rows!(var_scratch, w))
            .enumerate()
            .for_each(|(y, (row, vrow))| {
                for x in 0..w {
                    let p = y * w + x;
                    let z_p = g_depth[p];
                    if z_p <= 0.0 {
                        // Background: analytic and noise-free. Pass through.
                        row[x * 3] = illum[p * 3];
                        row[x * 3 + 1] = illum[p * 3 + 1];
                        row[x * 3 + 2] = illum[p * 3 + 2];
                        vrow[x] = var[p];
                        continue;
                    }
                    let n_p = [g_normal[p * 3], g_normal[p * 3 + 1], g_normal[p * 3 + 2]];
                    let c_p = [illum[p * 3], illum[p * 3 + 1], illum[p * 3 + 2]];
                    let l_p = luminance(c_p);
                    // The estimator's own error bar sets how much luminance
                    // disagreement counts as signal rather than noise. A
                    // firefly has an enormous error bar, so it stops
                    // protecting itself and gets filtered.
                    let l_tol = sigma_l * var[p].max(0.0).sqrt() + 1e-4;

                    let mut sum = [0.0f32; 3];
                    let mut vsum = 0.0f32;
                    let mut wsum = 0.0f32;

                    for (ky, dy) in (-2i32..=2).enumerate() {
                        let qy = y as i32 + dy * stride as i32;
                        if qy < 0 || qy >= h as i32 {
                            continue;
                        }
                        for (kx, dx) in (-2i32..=2).enumerate() {
                            let qx = x as i32 + dx * stride as i32;
                            if qx < 0 || qx >= w as i32 {
                                continue;
                            }
                            let q = qy as usize * w + qx as usize;
                            let z_q = g_depth[q];
                            if z_q <= 0.0 {
                                // Never let the backdrop bleed onto a surface.
                                continue;
                            }

                            // Normal: squared distance between unit normals.
                            let dn = [
                                n_p[0] - g_normal[q * 3],
                                n_p[1] - g_normal[q * 3 + 1],
                                n_p[2] - g_normal[q * 3 + 2],
                            ];
                            let dn2 = dn[0] * dn[0] + dn[1] * dn[1] + dn[2] * dn[2];
                            let w_n = (-dn2 / sigma_n2).exp();

                            // Depth: relative, so the tolerance scales with
                            // scene size instead of being tuned per model.
                            let w_z = (-(z_p - z_q).abs() / (sigma_z * z_p)).exp();

                            // Illumination: rejects the far side of a shadow
                            // edge or a specular highlight.
                            let c_q = [illum[q * 3], illum[q * 3 + 1], illum[q * 3 + 2]];
                            let w_l = (-(l_p - luminance(c_q)).abs() / l_tol).exp();

                            let weight = B3_SPLINE[kx] * B3_SPLINE[ky] * w_n * w_z * w_l;
                            if weight <= 0.0 {
                                continue;
                            }
                            sum = add3(sum, scale3(c_q, weight));
                            // Variance of a weighted mean of independent
                            // estimates carries the *squared* weights.
                            vsum += weight * weight * var[q];
                            wsum += weight;
                        }
                    }

                    let (out, vout) = if wsum > 0.0 {
                        (scale3(sum, 1.0 / wsum), vsum / (wsum * wsum))
                    } else {
                        (c_p, var[p])
                    };
                    row[x * 3] = out[0];
                    row[x * 3 + 1] = out[1];
                    row[x * 3 + 2] = out[2];
                    vrow[x] = vout;
                }
            });

        std::mem::swap(&mut illum, &mut scratch);
        std::mem::swap(&mut var, &mut var_scratch);
    }

    // Re-modulate back into radiance. Background pixels are left exactly as
    // the tracer wrote them — a divide-then-multiply round trip is not
    // bit-exact in f32, and the backdrop has no noise to remove anyway.
    for i in 0..n {
        if film.depth[i] <= 0.0 {
            continue;
        }
        for c in 0..3 {
            let a = film.albedo[i * 3 + c].max(DEMOD_FLOOR);
            film.rgb[i * 3 + c] = illum[i * 3 + c] * a;
        }
    }
}

// ─── tonemapping ──────────────────────────────────────────────────────────

/// ACES filmic tonemap (Narkowicz fit).
#[inline]
pub fn tonemap_aces(x: f32) -> f32 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
}

/// Linear to sRGB transfer.
#[inline]
pub fn linear_to_srgb(x: f32) -> f32 {
    if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

impl Film {
    /// Convert to 8-bit sRGB RGBA with ACES tonemapping.
    ///
    /// `exposure` scales linear radiance before the tonemap curve.
    pub fn to_srgb8(&self, exposure: f32, transparent: bool) -> Vec<u8> {
        let n = (self.width * self.height) as usize;
        let mut out = vec![0u8; n * 4];
        for i in 0..n {
            for c in 0..3 {
                let v = tonemap_aces(self.rgb[i * 3 + c] * exposure);
                out[i * 4 + c] = (linear_to_srgb(v) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
            out[i * 4 + 3] = if transparent {
                (self.alpha[i] * 255.0 + 0.5).clamp(0.0, 255.0) as u8
            } else {
                255
            };
        }
        out
    }
}

// ─── default studio rig ───────────────────────────────────────────────────

/// Build a three-point softbox rig sized to a scene of the given radius,
/// centred on `center`.
///
/// Key light upper-front-left, a broad cool fill opposite it, and a small
/// bright rim behind to separate the subject from the backdrop.
pub fn studio_rig(center: Point3, radius: f64) -> Vec<AreaLight> {
    let r = radius.max(1e-6);
    let mk = |dir: Vec3, dist: f64, size: f64, emission: [f32; 3]| -> AreaLight {
        let pos = center + dir.normalize() * (r * dist);
        // Orient the rectangle to face the scene centre.
        let n = (center - pos).normalize();
        let (u, v) = onb(n);
        AreaLight {
            center: pos,
            u: u * (r * size),
            v: v * (r * size),
            emission,
        }
    };

    // Emission is radiance, so the useful quantity is emission × solid
    // angle. A softbox of half-size `s` at distance `d` subtends roughly
    // (2s/d)² steradians; these values are chosen to land the key at ~3
    // and the rim at ~1.5 units of irradiance on a facing surface.
    vec![
        // Key: large, slightly warm, high and to the left.
        mk(Vec3::new(-0.8, -1.0, 1.1), 3.2, 1.4, [4.2, 4.0, 3.75]),
        // Fill: broad and cool, opposite side, much dimmer.
        mk(Vec3::new(1.3, -0.6, 0.25), 3.6, 1.8, [0.5, 0.56, 0.68]),
        // Rim: small and hot, behind and above, to pop the silhouette.
        mk(Vec3::new(0.35, 1.25, 0.8), 3.0, 0.55, [11.0, 10.8, 10.5]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::TriMesh;

    /// A 10 mm cube as twelve triangles, spanning (0,0,0)..(10,10,10).
    ///
    /// The integrator does not care what it is tracing — that is the point of
    /// the [`Geometry`] seam — so the scene these tests light is a triangle
    /// soup rather than a B-rep. Every assertion below is about the light.
    fn cube_mesh() -> TriMesh {
        let p = |x, y, z| Point3::new(x, y, z);
        let positions = vec![
            p(0.0, 0.0, 0.0),
            p(10.0, 0.0, 0.0),
            p(10.0, 10.0, 0.0),
            p(0.0, 10.0, 0.0),
            p(0.0, 0.0, 10.0),
            p(10.0, 0.0, 10.0),
            p(10.0, 10.0, 10.0),
            p(0.0, 10.0, 10.0),
        ];
        let indices = [
            0, 2, 1, 0, 3, 2, // -z
            4, 5, 6, 4, 6, 7, // +z
            0, 1, 5, 0, 5, 4, // -y
            3, 7, 6, 3, 6, 2, // +y
            0, 4, 7, 0, 7, 3, // -x
            1, 2, 6, 1, 6, 5, // +x
        ];
        TriMesh::new(positions, Vec::new(), &indices)
    }

    fn test_scene() -> Scene<TriMesh> {
        Scene {
            objects: vec![Object::new(
                Arc::new(Bvh::build(cube_mesh())),
                Pbr::plastic([0.8, 0.3, 0.2], 0.35, 0.0),
            )],
            lights: studio_rig(Point3::new(5.0, 5.0, 5.0), 9.0),
            env: Environment::default(),
            sun: None,
            ground: None,
            splats: None,
        }
    }

    /// The old estimator: shadow-ray every light, each with its own
    /// area-sampling PDF. Kept here as the reference the one-light-per-bounce
    /// importance sampler must match in expectation.
    fn sample_all_lights_reference(
        scene: &Scene<TriMesh>,
        accel: &SceneAccel<TriMesh>,
        p: Point3,
        frame: &Frame,
        wo_local: Vec3,
        m: &Pbr,
        eta: f32,
        rng: &mut Rng,
    ) -> [f32; 3] {
        let Frame { t, b, n } = *frame;
        let mut sum = [0.0f32; 3];
        for light in &scene.lights {
            let lp = light.sample(rng.f64(), rng.f64());
            let to_light = lp - p;
            let dist = to_light.norm();
            if dist < 1e-9 {
                continue;
            }
            let wi_world = to_light / dist;
            let cos_light = -wi_world.dot(light.normal());
            if cos_light <= 1e-9 {
                continue;
            }
            let wi_local = to_local(t, b, n, wi_world);
            if wi_local.z <= 0.0 {
                continue;
            }
            let (f, bsdf_pdf) = bsdf_eval(m, wo_local, wi_local, eta, 0.0);
            if max3(f) <= 0.0 {
                continue;
            }
            let light_pdf = (dist * dist / (cos_light * light.area())) as f32;
            if !light_pdf.is_finite() || light_pdf <= 0.0 {
                continue;
            }
            if scene.occluded(accel, p + n * 1e-5, wi_world, dist) {
                continue;
            }
            // The reference's MIS partner must be *its* own light pdf, so
            // this is the old weighting verbatim.
            let w = power_heuristic(light_pdf, bsdf_pdf);
            sum = add3(sum, scale3(mul3(f, light.emission), w / light_pdf));
        }
        sum
    }

    /// Both estimators are MIS-weighted, and the two weightings differ per
    /// sample (the pick probability enters the light pdf). What must agree is
    /// the *total* direct-lighting estimate — NEE plus the BSDF-sampled hits
    /// on emitters — so this compares the unweighted NEE integral by driving
    /// both with `power_heuristic` replaced by 1: i.e. the plain estimator
    /// `f * Le * cos / pdf`, which is what unbiasedness is about.
    fn nee_unweighted_mean(
        scene: &Scene<TriMesh>,
        accel: &SceneAccel<TriMesh>,
        pick_one: bool,
        n: usize,
    ) -> [f64; 3] {
        let p = Point3::new(0.0, 0.0, 0.0);
        let nrm = Vec3::new(0.0, 0.0, 1.0);
        let frame = shading_frame(nrm, None);
        let wo_world = Vec3::new(0.3, 0.2, 0.9).normalize();
        let wo_local = to_local(frame.t, frame.b, nrm, wo_world);
        let m = Pbr {
            base_color: [0.8, 0.7, 0.6],
            roughness: 0.6,
            ..Default::default()
        };
        let mut rng = Rng::new(0xA11CE);
        let mut sum = [0.0f64; 3];
        for _ in 0..n {
            let est = if pick_one {
                let Some((i, pick_pdf)) = accel.pick_light(rng.f64() as f32) else {
                    continue;
                };
                one_light_unweighted(
                    &scene.lights[i],
                    pick_pdf,
                    p,
                    &frame,
                    wo_local,
                    &m,
                    1.0,
                    &mut rng,
                )
            } else {
                let mut acc = [0.0f32; 3];
                for light in &scene.lights {
                    acc = add3(
                        acc,
                        one_light_unweighted(light, 1.0, p, &frame, wo_local, &m, 1.0, &mut rng),
                    );
                }
                acc
            };
            for c in 0..3 {
                sum[c] += est[c] as f64;
            }
        }
        [sum[0] / n as f64, sum[1] / n as f64, sum[2] / n as f64]
    }

    fn one_light_unweighted(
        light: &AreaLight,
        pick_pdf: f32,
        p: Point3,
        frame: &Frame,
        wo_local: Vec3,
        m: &Pbr,
        eta: f32,
        rng: &mut Rng,
    ) -> [f32; 3] {
        let Frame { t, b, n } = *frame;
        let lp = light.sample(rng.f64(), rng.f64());
        let to_light = lp - p;
        let dist = to_light.norm();
        if dist < 1e-9 {
            return [0.0; 3];
        }
        let wi_world = to_light / dist;
        let cos_light = -wi_world.dot(light.normal());
        if cos_light <= 1e-9 {
            return [0.0; 3];
        }
        let wi_local = to_local(t, b, n, wi_world);
        if wi_local.z <= 0.0 {
            return [0.0; 3];
        }
        let (f, _) = bsdf_eval(m, wo_local, wi_local, eta, 0.0);
        let pdf = pick_pdf * (dist * dist / (cos_light * light.area())) as f32;
        if !pdf.is_finite() || pdf <= 0.0 {
            return [0.0; 3];
        }
        scale3(mul3(f, light.emission), 1.0 / pdf)
    }

    fn open_scene(lights: Vec<AreaLight>) -> Scene<TriMesh> {
        Scene {
            objects: Vec::new(),
            lights,
            env: Environment::default(),
            sun: None,
            ground: None,
            splats: None,
        }
    }

    fn panel(center: Point3, emission: [f32; 3], half: f64) -> AreaLight {
        // Faces -Z, i.e. down at the origin.
        AreaLight {
            center,
            u: Vec3::new(half, 0.0, 0.0),
            v: Vec3::new(0.0, -half, 0.0),
            emission,
        }
    }

    /// An axis-aligned quad in the z = `z` plane, spanning ±`half` in x and y.
    fn pane_mesh(z: f64, half: f64) -> TriMesh {
        let p = |x, y| Point3::new(x, y, z);
        let positions = vec![
            p(-half, -half),
            p(half, -half),
            p(half, half),
            p(-half, half),
        ];
        TriMesh::new(positions, Vec::new(), &[0, 1, 2, 0, 2, 3])
    }

    /// A smooth thin-walled pane of ordinary window glass.
    fn window_glass() -> Pbr {
        Pbr {
            transmission: 1.0,
            thin_walled: true,
            roughness: 0.0,
            ior: 1.5,
            ..Pbr::glass(1.5, 0.0)
        }
    }

    /// Mean NEE estimate at the origin on a white Lambertian floor facing +z.
    fn nee_mean(scene: &Scene<TriMesh>, n: usize) -> f64 {
        let accel = SceneAccel::build(scene);
        let m = Pbr {
            base_color: [1.0; 3],
            metallic: 0.0,
            roughness: 1.0,
            ..Pbr::default()
        };
        let frame = shading_frame(Vec3::new(0.0, 0.0, 1.0), None);
        let wo_local = Vec3::new(0.0, 0.0, 1.0);
        let mut rng = Rng::new(0x9e3779b97f4a7c15);
        let mut sum = 0.0f64;
        for _ in 0..n {
            let e = scene.sample_lights(
                &accel,
                Point3::new(0.0, 0.0, 0.0),
                &frame,
                wo_local,
                &m,
                1.0,
                0.0,
                &mut rng,
            );
            sum += luminance(e) as f64;
        }
        sum / n as f64
    }

    /// Nothing about an opaque scene changed. The shadow ray that walks
    /// through sheets has to agree, hit for hit, with the any-hit traversal
    /// it replaced wherever there are no sheets to walk through — which is
    /// what makes every render that predates panes bit-identical.
    #[test]
    fn an_opaque_scene_occludes_exactly_as_the_any_hit_test_did() {
        let mut scene = open_scene(vec![panel(Point3::new(0.0, 0.0, 6.0), [10.0; 3], 1.0)]);
        scene.objects.push(Object::new(
            Arc::new(Bvh::build(cube_mesh())),
            Pbr::plastic([0.8, 0.3, 0.2], 0.35, 0.0),
        ));
        scene.objects.push(Object::new(
            Arc::new(Bvh::build(pane_mesh(3.0, 2.0))),
            // Transmissive but *not* thin-walled: a solid, which still
            // blocks. The caustic pass is what carries light through those.
            Pbr::glass(1.5, 0.0),
        ));
        let accel = SceneAccel::build(&scene);
        let mut rng = Rng::new(12345);
        let mut checked = 0;
        for _ in 0..4000 {
            let o = Point3::new(
                20.0 * rng.f64() - 5.0,
                20.0 * rng.f64() - 5.0,
                20.0 * rng.f64() - 5.0,
            );
            let d = Vec3::new(
                2.0 * rng.f64() - 1.0,
                2.0 * rng.f64() - 1.0,
                2.0 * rng.f64() - 1.0,
            );
            if d.norm() < 1e-6 {
                continue;
            }
            let d = d.normalize();
            let dist = 30.0 * rng.f64();
            let old = accel
                .tlas
                .occluded_range(&Ray::new(o, d), 1e-6, dist - 1e-6);
            let new = scene.shadow_transmittance(&accel, o, d, dist).is_none();
            assert_eq!(old, new, "from {o:?} along {d:?} for {dist}");
            checked += 1;
        }
        assert!(checked > 3000);
    }

    /// A shadow ray must see *through* a pane of glass, dimmed by exactly the
    /// factor the thin-walled BSDF applies to a refracted path.
    ///
    /// The old material-blind any-hit test returned black here, which is why
    /// a room could not be lit through a window at any sample count.
    #[test]
    fn next_event_passes_through_a_thin_pane() {
        let light = panel(Point3::new(0.0, 0.0, 6.0), [10.0; 3], 0.35);
        let open = open_scene(vec![light]);
        let mut glazed = open_scene(vec![light]);
        glazed.objects.push(Object::new(
            Arc::new(Bvh::build(pane_mesh(3.0, 4.0))),
            window_glass(),
        ));

        let n = 200_000;
        let bare = nee_mean(&open, n);
        let through = nee_mean(&glazed, n);
        assert!(bare > 0.0, "the open scene must be lit at all");

        // The light is small and nearly overhead, so every shadow ray meets
        // the pane within a few degrees of normal incidence.
        let f = fresnel_dielectric(1.0, 1.5);
        let expected = (1.0 - f) as f64;
        let ratio = through / bare;
        assert!(
            (ratio - expected).abs() < 0.02 * expected,
            "pane transmittance {ratio} is not within 2% of {expected}"
        );
    }

    /// A stack of panes deeper than the cap is an honest blocker, so the
    /// traversal cannot run away.
    #[test]
    fn a_shadow_ray_gives_up_past_the_sheet_cap() {
        let light = panel(Point3::new(0.0, 0.0, 6.0), [10.0; 3], 0.35);
        let mut stacked = open_scene(vec![light]);
        for i in 0..(MAX_SHADOW_SHEETS + 1) {
            stacked.objects.push(Object::new(
                Arc::new(Bvh::build(pane_mesh(1.0 + i as f64 * 0.5, 4.0))),
                window_glass(),
            ));
        }
        assert_eq!(nee_mean(&stacked, 4_000), 0.0);
    }

    /// A frosted pane still blocks: the straight-line shadow ray is only the
    /// right answer in the smooth limit, so a wide lobe tapers it away.
    #[test]
    fn a_frosted_pane_still_blocks_the_shadow_ray() {
        let light = panel(Point3::new(0.0, 0.0, 6.0), [10.0; 3], 0.35);
        let mut frosted = open_scene(vec![light]);
        frosted.objects.push(Object::new(
            Arc::new(Bvh::build(pane_mesh(3.0, 4.0))),
            Pbr {
                roughness: 1.0,
                ..window_glass()
            },
        ));
        assert_eq!(nee_mean(&frosted, 4_000), 0.0);
    }

    /// The *whole* estimator — NEE and BSDF sampling combined under MIS —
    /// must land on the same `(1 − F)` factor the single strategy does. If
    /// the two disagreed, MIS would double count the refracted path in one
    /// direction and lose it in the other; agreement is the check.
    #[test]
    fn a_converged_render_through_a_pane_matches_the_single_strategy() {
        let floor = || {
            Object::new(
                Arc::new(Bvh::build(pane_mesh(0.0, 6.0))),
                Pbr {
                    base_color: [1.0; 3],
                    metallic: 0.0,
                    roughness: 1.0,
                    specular: 0.0,
                    ..Pbr::default()
                },
            )
        };
        let light = panel(Point3::new(0.0, 0.0, 6.0), [10.0; 3], 0.35);
        let camera = Camera::look_at(
            Point3::new(0.0, -0.01, 2.0),
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            30.0,
        );
        let opts = PathTraceOptions {
            spp: 400,
            max_depth: 4,
            firefly_clamp: None,
            denoise: false,
            show_background: false,
            ..PathTraceOptions::default()
        };

        let mean = |with_pane: bool| -> f64 {
            let mut scene = open_scene(vec![light]);
            scene.objects.push(floor());
            if with_pane {
                scene.objects.push(Object::new(
                    Arc::new(Bvh::build(pane_mesh(3.0, 5.0))),
                    window_glass(),
                ));
            }
            let film = render(&scene, &camera, 24, 24, &opts);
            let mut sum = 0.0f64;
            for i in 0..(24 * 24) {
                sum +=
                    luminance([film.rgb[i * 3], film.rgb[i * 3 + 1], film.rgb[i * 3 + 2]]) as f64;
            }
            sum / (24.0 * 24.0)
        };

        let bare = mean(false);
        let glazed = mean(true);
        assert!(bare > 0.0);
        let expected = (1.0 - fresnel_dielectric(1.0, 1.5)) as f64;
        let ratio = glazed / bare;
        assert!(
            (ratio - expected).abs() < 0.03 * expected,
            "converged ratio {ratio} is not within 3% of {expected}"
        );
    }

    /// One light per bounce, drawn from the power table and divided by its
    /// pick probability, must integrate to the same direct lighting as
    /// shadow-raying every light. Two lights of very different power, so a
    /// uniform pick would not have been enough.
    #[test]
    fn one_light_per_bounce_matches_all_lights_in_expectation() {
        let scene = open_scene(vec![
            panel(Point3::new(-2.0, 0.0, 4.0), [12.0, 11.0, 10.0], 1.5),
            panel(Point3::new(3.0, 1.0, 5.0), [0.6, 0.7, 1.4], 0.7),
        ]);
        let accel = SceneAccel::build(&scene);
        let n = 400_000;
        let a = nee_unweighted_mean(&scene, &accel, true, n);
        let b = nee_unweighted_mean(&scene, &accel, false, n);
        for c in 0..3 {
            let rel = (a[c] - b[c]).abs() / b[c].abs().max(1e-6);
            assert!(
                rel < 0.02,
                "channel {c}: one-light mean {} vs all-lights mean {} (rel {rel})",
                a[c],
                b[c]
            );
        }
    }

    /// The power table must actually be power-weighted: the bright panel is
    /// picked far more often than the dim one, and the probabilities sum to 1.
    #[test]
    fn light_table_is_power_weighted() {
        let scene = open_scene(vec![
            panel(Point3::new(-2.0, 0.0, 4.0), [12.0, 11.0, 10.0], 1.5),
            panel(Point3::new(3.0, 1.0, 5.0), [0.6, 0.7, 1.4], 0.7),
        ]);
        let accel = SceneAccel::build(&scene);
        let p0 = accel.light_pick_pdf(0);
        let p1 = accel.light_pick_pdf(1);
        assert!((p0 + p1 - 1.0).abs() < 1e-5, "pick pdf must sum to 1");
        assert!(p0 > 0.9, "the bright, large panel should dominate: {p0}");
        // Drawing follows the table.
        let mut rng = Rng::new(7);
        let mut hits = [0u32; 2];
        for _ in 0..20_000 {
            let (i, _) = accel.pick_light(rng.f64() as f32).unwrap();
            hits[i] += 1;
        }
        let frac0 = hits[0] as f32 / 20_000.0;
        assert!((frac0 - p0).abs() < 0.02, "draw {frac0} vs table {p0}");
    }

    /// A full render of a multi-light scene must still land on the same image
    /// the all-lights estimator gives, within Monte Carlo noise.
    #[test]
    fn multi_light_render_matches_reference_mean() {
        // Exercise the reference path so it cannot rot.
        let scene = open_scene(vec![
            panel(Point3::new(-2.0, 0.0, 4.0), [8.0, 8.0, 8.0], 1.2),
            panel(Point3::new(3.0, 1.0, 5.0), [2.0, 2.0, 2.0], 1.0),
        ]);
        let accel = SceneAccel::build(&scene);
        let nrm = Vec3::new(0.0, 0.0, 1.0);
        let frame = shading_frame(nrm, None);
        let m = Pbr::default();
        let wo_local = to_local(frame.t, frame.b, nrm, nrm);
        let mut rng = Rng::new(3);
        let mut r = [0.0f64; 3];
        let mut o = [0.0f64; 3];
        let n = 200_000;
        for _ in 0..n {
            let a = scene.sample_lights(
                &accel,
                Point3::new(0.0, 0.0, 0.0),
                &frame,
                wo_local,
                &m,
                1.0,
                0.0,
                &mut rng,
            );
            let b = sample_all_lights_reference(
                &scene,
                &accel,
                Point3::new(0.0, 0.0, 0.0),
                &frame,
                wo_local,
                &m,
                1.0,
                &mut rng,
            );
            for c in 0..3 {
                r[c] += a[c] as f64;
                o[c] += b[c] as f64;
            }
        }
        // MIS weights differ slightly between the two (the light pdf carries
        // the pick probability), so this is a loose sanity band, not equality.
        for c in 0..3 {
            let rel = (r[c] - o[c]).abs() / (o[c] / n as f64).abs().max(1e-9) / n as f64;
            assert!(
                rel < 0.06,
                "channel {c}: {} vs {} (rel {rel})",
                r[c] / n as f64,
                o[c] / n as f64
            );
        }
    }

    /// A masked pass must reproduce the full render exactly — not "within
    /// noise", *bit for bit*. That is only true if the per-pixel seed depends
    /// on nothing but the pixel, which is the property that lets a caller drop
    /// a re-traced patch into a frame it already has without a seam.
    #[test]
    fn render_into_is_bit_identical_to_the_full_render() {
        let scene = test_scene();
        let cam = test_camera();
        let opts = PathTraceOptions {
            spp: 3,
            max_depth: 3,
            denoise: false,
            ..Default::default()
        };
        let (w, h) = (40u32, 32u32);
        let full = render(&scene, &cam, w, h, &opts);

        // Rects that clip, touch the edges, and overlap each other.
        let rects = [
            [3, 4, 10, 9],
            [9, 6, 12, 20],
            [0, 0, 5, 5],
            [35, 28, 20, 20],
        ];
        let mut patched = Film::new(w, h);
        render_into(&scene, &cam, &mut patched, &opts, &rects);

        let inside = |px: u32, py: u32| {
            rects
                .iter()
                .any(|r| px >= r[0] && py >= r[1] && px < r[0] + r[2] && py < r[1] + r[3])
        };
        let mut covered = 0usize;
        for py in 0..h {
            for px in 0..w {
                let i = (py * w + px) as usize;
                if inside(px, py) {
                    covered += 1;
                    for c in 0..3 {
                        assert_eq!(
                            patched.rgb[i * 3 + c].to_bits(),
                            full.rgb[i * 3 + c].to_bits(),
                            "pixel ({px}, {py}) channel {c}: masked render gave {} \
                             where the full render gave {}. The per-pixel seed \
                             must not depend on anything but the pixel.",
                            patched.rgb[i * 3 + c],
                            full.rgb[i * 3 + c],
                        );
                        assert_eq!(patched.albedo[i * 3 + c], full.albedo[i * 3 + c]);
                        assert_eq!(patched.normal[i * 3 + c], full.normal[i * 3 + c]);
                    }
                    assert_eq!(patched.alpha[i], full.alpha[i]);
                    assert_eq!(patched.depth[i], full.depth[i]);
                    assert_eq!(patched.variance[i].to_bits(), full.variance[i].to_bits());
                } else {
                    // Outside the union nothing was touched at all.
                    assert_eq!(
                        (patched.rgb[i * 3], patched.alpha[i], patched.depth[i]),
                        (0.0, 0.0, 0.0),
                        "pixel ({px}, {py}) is outside every rect and was written anyway",
                    );
                }
            }
        }
        assert!(covered > 300, "the rects covered only {covered} px");
        assert!(
            covered < (w * h) as usize,
            "the rects covered the whole film"
        );
    }

    /// A patched frame is the frame: re-tracing every rect of a partition of
    /// the film reconstructs the full render exactly.
    #[test]
    fn tiling_the_film_with_rects_reconstructs_the_whole_render() {
        let scene = test_scene();
        let cam = test_camera();
        let opts = PathTraceOptions {
            spp: 2,
            max_depth: 2,
            denoise: false,
            ..Default::default()
        };
        let (w, h) = (24u32, 24u32);
        let full = render(&scene, &cam, w, h, &opts);
        let rects: Vec<[u32; 4]> = (0..3)
            .flat_map(|i| (0..3).map(move |j| [i * 8, j * 8, 8, 8]))
            .collect();
        let mut patched = Film::new(w, h);
        render_into(&scene, &cam, &mut patched, &opts, &rects);
        assert_eq!(patched.rgb, full.rgb);
        assert_eq!(patched.depth, full.depth);
    }

    /// Degenerate and out-of-bounds rects are clipped, not panics.
    #[test]
    fn render_into_clips_rects_to_the_film() {
        let scene = test_scene();
        let cam = test_camera();
        let opts = PathTraceOptions {
            spp: 1,
            max_depth: 1,
            denoise: false,
            ..Default::default()
        };
        let mut film = Film::new(16, 16);
        render_into(
            &scene,
            &cam,
            &mut film,
            &opts,
            &[
                [0, 0, 0, 0],
                [20, 20, 4, 4],
                [14, 14, u32::MAX, u32::MAX],
                [0, 0, 16, 16],
            ],
        );
        assert_eq!(film.rgb.len(), 16 * 16 * 3);
    }

    fn test_camera() -> Camera {
        Camera::look_at(
            Point3::new(30.0, -34.0, 24.0),
            Point3::new(5.0, 5.0, 5.0),
            Vec3::new(0.0, 0.0, 1.0),
            32.0,
        )
    }

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

    #[test]
    fn renders_non_empty() {
        let scene = test_scene();
        let cam = test_camera();
        let film = render(
            &scene,
            &cam,
            24,
            24,
            &PathTraceOptions {
                spp: 4,
                ..Default::default()
            },
        );
        assert_eq!(film.rgb.len(), 24 * 24 * 3);
        let lit = film.rgb.iter().filter(|v| **v > 0.0).count();
        assert!(lit > 0, "path tracer produced an entirely black frame");
    }

    /// `Object::transform` must actually place the BLAS. This is what an
    /// animated render leans on: the same BVH, re-posed per frame.
    #[test]
    fn object_transform_moves_the_subject() {
        let cam = test_camera();
        let coverage = |t: Transform| {
            let scene = Scene {
                objects: vec![Object::placed(
                    Arc::new(Bvh::build(cube_mesh())),
                    Pbr::plastic([0.8, 0.3, 0.2], 0.35, 0.0),
                    t,
                )],
                // No area lights: they are hittable geometry, and a rig that
                // stays put while the cube moves would muddy the coverage
                // signal this test reads.
                lights: Vec::new(),
                env: Environment::default(),
                sun: None,
                ground: None,
                splats: None,
            };
            let film = render(
                &scene,
                &cam,
                32,
                32,
                &PathTraceOptions {
                    spp: 2,
                    ..Default::default()
                },
            );
            film.alpha.iter().map(|a| *a > 0.5).collect::<Vec<_>>()
        };
        let here = coverage(Transform::identity());
        // Far enough out of frame that nothing overlaps.
        let there = coverage(Transform::translation(400.0, 0.0, 0.0));
        assert!(here.iter().any(|c| *c), "identity placement lost the cube");
        assert!(
            !there.iter().any(|c| *c),
            "translated placement was ignored — the cube stayed put"
        );
    }

    #[test]
    fn subject_is_covered() {
        let scene = test_scene();
        let cam = test_camera();
        let film = render(
            &scene,
            &cam,
            32,
            32,
            &PathTraceOptions {
                spp: 4,
                ..Default::default()
            },
        );
        let covered = film.alpha.iter().filter(|a| **a > 0.5).count();
        assert!(
            covered > 40,
            "expected the cube to cover a chunk of frame, got {covered}"
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let scene = test_scene();
        let cam = test_camera();
        let o = PathTraceOptions {
            spp: 2,
            ..Default::default()
        };
        let a = render(&scene, &cam, 16, 16, &o);
        let b = render(&scene, &cam, 16, 16, &o);
        assert_eq!(a.rgb, b.rgb, "render must be seed-deterministic");
    }

    /// The BSDF sampling PDF must match the analytic PDF used by MIS, or
    /// light sampling and BSDF sampling silently disagree and the image is
    /// energy-wrong in a way that is hard to see by eye.
    #[test]
    fn bsdf_sample_pdf_matches_eval_pdf() {
        // Isotropic, plus anisotropy swept across both signs and both
        // extremes — the sampling and evaluation paths must agree for all of
        // them or MIS is silently energy-wrong.
        let anisos = [0.0, 0.4, 0.8, 1.0, -0.4, -0.8, -1.0];
        for aniso in anisos {
            for roughness in [0.1, 0.4, 0.8] {
                let m = Pbr {
                    base_color: [0.8, 0.8, 0.8],
                    metallic: 0.3,
                    roughness,
                    anisotropy: aniso,
                    clearcoat: 0.5,
                    ..Default::default()
                };
                // Several view directions: a grazing wo is where an
                // anisotropic G1 and a mismatched PDF diverge fastest.
                for wo in [
                    Vec3::new(0.3, 0.15, 0.94).normalize(),
                    Vec3::new(0.85, 0.1, 0.52).normalize(),
                    Vec3::new(0.1, 0.85, 0.52).normalize(),
                    Vec3::new(0.0, 0.0, 1.0),
                ] {
                    let mut rng = Rng::new(7);
                    for _ in 0..256 {
                        if let Some((wi, _f, pdf)) = bsdf_sample_surface(&m, wo, 1.0, 0.0, &mut rng)
                        {
                            let (_f2, pdf2) = bsdf_eval(&m, wo, wi, 1.0, 0.0);
                            assert!(
                                (pdf - pdf2).abs() <= 1e-4 * pdf.max(1.0),
                                "pdf mismatch at aniso={aniso} rough={roughness}: \
                                 sampled {pdf}, evaluated {pdf2}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// Anisotropy 0 must be the isotropic model exactly — not merely close.
    /// If this drifts, every existing render changes silently.
    #[test]
    fn zero_anisotropy_is_exactly_isotropic() {
        let m = Pbr {
            base_color: [0.8, 0.7, 0.6],
            metallic: 0.6,
            roughness: 0.35,
            anisotropy: 0.0,
            clearcoat: 0.3,
            ..Default::default()
        };
        let (at, ab) = m.alpha_tb();
        assert_eq!(at, m.alpha(), "tangent alpha diverged from the base alpha");
        assert_eq!(
            ab,
            m.alpha(),
            "bitangent alpha diverged from the base alpha"
        );

        // And the lobe terms must take their isotropic branches bit-exactly.
        let wo = Vec3::new(0.3, 0.15, 0.94).normalize();
        let wi = Vec3::new(-0.2, 0.35, 0.91).normalize();
        let wh = (wo + wi).normalize();
        assert_eq!(d_ggx_aniso(wh, at, ab), d_ggx(wh, at));
        assert_eq!(
            v_smith_aniso(wo, wi, at, ab),
            v_smith(wo.z as f32, wi.z as f32, at)
        );
    }

    /// The whole point of the feature: an anisotropic lobe must actually
    /// prefer one tangent direction over the other, and swapping the sign of
    /// the anisotropy must swap which one.
    #[test]
    fn anisotropy_stretches_the_lobe_along_the_tangent() {
        let rough = |anisotropy| Pbr {
            base_color: [1.0, 1.0, 1.0],
            metallic: 1.0,
            roughness: 0.3,
            anisotropy,
            clearcoat: 0.0,
            ..Default::default()
        };
        // Straight-on view, so the mirror direction is +Z and any asymmetry
        // is the lobe's own, not the geometry's.
        let wo = Vec3::new(0.0, 0.0, 1.0);
        // Two directions tilted off the mirror by the same angle: one along
        // the tangent (x), one along the bitangent (y).
        let along = Vec3::new(0.25, 0.0, 1.0).normalize();
        let across = Vec3::new(0.0, 0.25, 1.0).normalize();

        let (f_iso_a, _) = bsdf_eval(&rough(0.0), wo, along, 1.0, 0.0);
        let (f_iso_b, _) = bsdf_eval(&rough(0.0), wo, across, 1.0, 0.0);
        assert!(
            (f_iso_a[0] - f_iso_b[0]).abs() < 1e-6,
            "isotropic lobe must be rotationally symmetric"
        );

        let (f_pos_a, _) = bsdf_eval(&rough(0.8), wo, along, 1.0, 0.0);
        let (f_pos_b, _) = bsdf_eval(&rough(0.8), wo, across, 1.0, 0.0);
        assert!(
            f_pos_a[0] > f_pos_b[0] * 1.5,
            "positive anisotropy should spread energy along the tangent: \
             along {} vs across {}",
            f_pos_a[0],
            f_pos_b[0]
        );

        let (f_neg_a, _) = bsdf_eval(&rough(-0.8), wo, along, 1.0, 0.0);
        let (f_neg_b, _) = bsdf_eval(&rough(-0.8), wo, across, 1.0, 0.0);
        assert!(
            f_neg_b[0] > f_neg_a[0] * 1.5,
            "negative anisotropy should spread energy across the tangent: \
             along {} vs across {}",
            f_neg_a[0],
            f_neg_b[0]
        );
    }

    /// Root-mean-square error between two films, measured on the tonemapped
    /// display values rather than raw radiance.
    ///
    /// Linear-radiance RMSE on a path-traced frame is almost entirely a
    /// firefly metric — measured on this scene, the worst 1% of pixels carried
    /// 97% of the squared error, so the number mostly reports how many
    /// outliers the *reference* still has, not how clean the image looks. The
    /// tonemap is the transform the viewer sees through, and it is what makes
    /// this a measure of visible error.
    fn rmse(a: &Film, b: &Film) -> f32 {
        assert_eq!(a.rgb.len(), b.rgb.len());
        let s: f32 = a
            .rgb
            .iter()
            .zip(&b.rgb)
            .map(|(x, y)| {
                let d = tonemap_aces(*x) - tonemap_aces(*y);
                d * d
            })
            .sum();
        (s / a.rgb.len() as f32).sqrt()
    }

    /// The property that matters: denoising a noisy render must move it
    /// *closer to the truth*, not merely change it. A blur that smeared
    /// everything would also "change the output" while making the image
    /// worse, and this is the test that tells the two apart.
    #[test]
    fn denoise_moves_low_spp_toward_high_spp_reference() {
        let scene = test_scene();
        let cam = test_camera();
        // Big enough that the doubling stride is meaningful: at 28px a
        // 5-iteration à-trous reaches past the image edge and the later passes
        // can only over-blur, which made an earlier version of this test
        // report a 7% win where the real figure is ~60%.
        let (w, h) = (96, 96);

        let reference = render(
            &scene,
            &cam,
            w,
            h,
            &PathTraceOptions {
                spp: 1024,
                denoise: false,
                ..Default::default()
            },
        );
        let noisy = render(
            &scene,
            &cam,
            w,
            h,
            &PathTraceOptions {
                spp: 4,
                denoise: false,
                ..Default::default()
            },
        );
        let denoised = render(
            &scene,
            &cam,
            w,
            h,
            &PathTraceOptions {
                spp: 4,
                denoise: true,
                ..Default::default()
            },
        );

        // Denoising is a post-process, so the two 4-spp films must have come
        // from the very same samples.
        assert_eq!(
            noisy.alpha, denoised.alpha,
            "denoising perturbed the sampling"
        );

        let before = rmse(&noisy, &reference);
        let after = rmse(&denoised, &reference);
        eprintln!("RMSE vs 1024spp: noisy {before:.5} -> denoised {after:.5}");
        // Measured ~60% reduction; assert a conservative fraction of it so the
        // test pins real quality rather than just "something happened", without
        // being brittle to sampling changes upstream.
        assert!(
            after < before * 0.75,
            "denoising did not meaningfully improve the estimate: \
             RMSE {before} -> {after}"
        );
    }

    /// The denoiser must not blur across a silhouette. Background pixels are
    /// analytic and noise-free, so they must come through untouched, and no
    /// surface pixel may pick up any backdrop.
    #[test]
    fn denoise_preserves_silhouette_edge() {
        let scene = test_scene();
        let cam = test_camera();
        let (w, h) = (48, 48);
        let opts = PathTraceOptions {
            spp: 4,
            denoise: false,
            ..Default::default()
        };
        let raw = render(&scene, &cam, w, h, &opts);
        let mut filtered = render(&scene, &cam, w, h, &opts);
        denoise(
            &mut filtered,
            &PathTraceOptions {
                denoise: true,
                ..opts
            },
        );

        let n = (w * h) as usize;
        let bg: Vec<usize> = (0..n).filter(|&i| raw.depth[i] <= 0.0).collect();
        let fg: Vec<usize> = (0..n).filter(|&i| raw.depth[i] > 0.0).collect();
        assert!(
            !bg.is_empty() && !fg.is_empty(),
            "test framing must contain both subject and backdrop"
        );

        // Backdrop is bit-identical.
        for &i in &bg {
            for c in 0..3 {
                assert_eq!(
                    raw.rgb[i * 3 + c],
                    filtered.rgb[i * 3 + c],
                    "backdrop pixel {i} was modified by the denoiser"
                );
            }
        }

        // Silhouette contrast is retained. A filter that leaked across the
        // edge would pull the two sides toward each other.
        let mean_lum = |f: &Film, idx: &[usize]| -> f32 {
            let s: f32 = idx
                .iter()
                .map(|&i| luminance([f.rgb[i * 3], f.rgb[i * 3 + 1], f.rgb[i * 3 + 2]]))
                .sum();
            s / idx.len() as f32
        };
        // Only the surface pixels that actually touch the backdrop.
        let rim: Vec<usize> = fg
            .iter()
            .copied()
            .filter(|&i| {
                let (x, y) = ((i % w as usize) as i32, (i / w as usize) as i32);
                [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)]
                    .iter()
                    .any(|(dx, dy)| {
                        let (qx, qy) = (x + dx, y + dy);
                        qx >= 0
                            && qy >= 0
                            && qx < w as i32
                            && qy < h as i32
                            && raw.depth[qy as usize * w as usize + qx as usize] <= 0.0
                    })
            })
            .collect();
        assert!(!rim.is_empty(), "expected a silhouette rim");

        let before = (mean_lum(&raw, &rim) - mean_lum(&raw, &bg)).abs();
        let after = (mean_lum(&filtered, &rim) - mean_lum(&filtered, &bg)).abs();
        assert!(
            after >= before * 0.95,
            "silhouette contrast collapsed: {before} -> {after}"
        );
    }

    /// Stratified quadrature of `g` over the upper hemisphere in `d(cos) dphi`.
    ///
    /// Deterministic, so a furnace number is a number and not a number plus
    /// Monte Carlo noise that has to be given slack in the bound.
    fn integrate_hemisphere<F: Fn(Vec3) -> f32>(n: usize, g: F) -> f32 {
        let dw = std::f64::consts::TAU / (n * n) as f64;
        let mut acc = 0.0f64;
        for i in 0..n {
            let ct = (i as f64 + 0.5) / n as f64;
            let st = (1.0 - ct * ct).max(0.0).sqrt();
            for j in 0..n {
                let phi = (j as f64 + 0.5) / n as f64 * std::f64::consts::TAU;
                let wi = Vec3::new(st * phi.cos(), st * phi.sin(), ct);
                acc += g(wi) as f64 * dw;
            }
        }
        acc as f32
    }

    /// Directions at a spread of incidence angles, all with the same azimuth
    /// so the anisotropic axis is exercised consistently.
    fn view_directions() -> Vec<Vec3> {
        [0.999f64, 0.9, 0.7, 0.5, 0.3, 0.15, 0.05]
            .iter()
            .map(|&mu| {
                let s = (1.0 - mu * mu).max(0.0).sqrt();
                Vec3::new(s, 0.0, mu)
            })
            .collect()
    }

    /// A white EON surface reflects (very nearly) all of the light it receives,
    /// at every roughness and every incidence angle.
    ///
    /// This is the property Lambert has trivially and plain Oren-Nayar does
    /// not: FON alone loses up to ~20% at `r = 1`, and the analytic
    /// compensation term is what puts it back.
    #[test]
    fn the_eon_diffuse_lobe_passes_a_white_furnace() {
        for r in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for wo in view_directions() {
                let e = integrate_hemisphere(180, |wi| {
                    eon_diffuse([1.0; 3], r, wo, wi)[0] * wi.z as f32
                });
                assert!(
                    (0.98..=1.0005).contains(&e),
                    "EON albedo {e} at diffuse_roughness {r}, mu {}",
                    wo.z
                );
            }
        }
    }

    /// `f(wo, wi) == f(wi, wo)` for the diffuse lobe, which every term of it
    /// is built to satisfy and none of which is obviously symmetric on sight.
    #[test]
    fn the_eon_diffuse_lobe_is_reciprocal() {
        let dirs = [
            Vec3::new(0.3, 0.15, 0.94).normalize(),
            Vec3::new(0.85, 0.1, 0.52).normalize(),
            Vec3::new(-0.4, 0.6, 0.69).normalize(),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        for r in [0.0, 0.4, 1.0] {
            for a in dirs {
                for b in dirs {
                    let ab = eon_diffuse([0.7, 0.5, 0.3], r, a, b);
                    let ba = eon_diffuse([0.7, 0.5, 0.3], r, b, a);
                    for c in 0..3 {
                        assert!(
                            (ab[c] - ba[c]).abs() <= 1e-6,
                            "EON not reciprocal at r={r}: {ab:?} vs {ba:?}"
                        );
                    }
                }
            }
        }
    }

    /// A perfectly reflective rough metal (`F0 = 1`) must return every photon.
    ///
    /// Single-scattering GGX does not: it keeps 0.947 of the energy at
    /// `alpha = 0.2`, 0.687 at `0.5` and 0.307 at `1.0`, because it drops
    /// every path that bounces from one microfacet to another. Turquin's
    /// compensation factor is exactly `1/E` when `F0 = 1`, so with it the
    /// furnace closes.
    #[test]
    fn a_rough_metal_furnace_closes_to_one_percent() {
        for alpha in [0.2f32, 0.5, 1.0] {
            let m = Pbr {
                base_color: [1.0; 3],
                metallic: 1.0,
                roughness: alpha.sqrt(),
                ..Default::default()
            };
            for wo in view_directions() {
                let e = integrate_hemisphere(220, |wi| bsdf_eval(&m, wo, wi, 1.0, 0.0).0[0]);
                assert!(
                    (0.99..=1.01).contains(&e),
                    "compensated GGX albedo {e} at alpha {alpha}, mu {}",
                    wo.z
                );
            }
        }
    }

    /// The compensation is a correction, not a licence to make light: a
    /// dielectric's `F0 = 0.04` lobe stays well under 1.
    #[test]
    fn a_dielectric_specular_lobe_stays_under_one() {
        for roughness in [0.1f32, 0.4, 0.7, 1.0] {
            let m = Pbr {
                base_color: [0.0; 3], // no diffuse: the specular lobe alone
                metallic: 0.0,
                roughness,
                ..Default::default()
            };
            for wo in view_directions() {
                let e = integrate_hemisphere(220, |wi| bsdf_eval(&m, wo, wi, 1.0, 0.0).0[0]);
                assert!(
                    (0.0..=1.0).contains(&e),
                    "dielectric specular albedo {e} at roughness {roughness}"
                );
            }
        }
    }

    /// Sheen is bounded, and it brightens towards grazing — which is the whole
    /// reason the lobe exists, and the thing a plain diffuse term cannot do.
    #[test]
    fn the_sheen_lobe_is_bounded_and_brightens_at_grazing() {
        for sheen_roughness in [0.15f32, 0.4, 0.8] {
            let m = Pbr {
                base_color: [0.0; 3],
                sheen: 1.0,
                sheen_roughness,
                ..Default::default()
            };
            let albedo = |mu: f64| {
                let s = (1.0 - mu * mu).max(0.0).sqrt();
                let wo = Vec3::new(s, 0.0, mu);
                integrate_hemisphere(220, |wi| sheen_eval(&m, wo, wi).0[0])
            };
            for mu in [0.999, 0.7, 0.3, 0.08] {
                let e = albedo(mu);
                assert!(
                    (0.0..=1.0).contains(&e),
                    "sheen albedo {e} out of range at mu {mu}, \
                     sheen_roughness {sheen_roughness}"
                );
            }
            assert!(
                albedo(0.08) > albedo(0.999) * 1.2,
                "sheen does not brighten at grazing (roughness {sheen_roughness}): \
                 {} at normal, {} at grazing",
                albedo(0.999),
                albedo(0.08)
            );
        }
    }

    /// The whole BSDF is reciprocal, layering included — the coat's Fresnel
    /// attenuation and the sheen's albedo scaling both had to be written
    /// symmetrically for this to hold.
    ///
    /// One term is deliberately not: Turquin's compensation factor is a
    /// function of the *outgoing* direction alone, which is the price of its
    /// closed form and of the exact `1/E` furnace it buys. The residual is
    /// bounded by how far `F0·(1-E)/E` can move between two angles, which for
    /// anything short of a mirror-bright metal is a fraction of a percent —
    /// hence the 1.5% tolerance here rather than the 1e-6 the diffuse lobe
    /// gets. It was 1% while `subsurface` still blended in a Hanrahan-Krueger
    /// lobe: that lobe is brighter than EON, so it made the diffuse share of
    /// the total larger and the specular's non-reciprocal share smaller.
    /// Replacing it with a real random walk takes that dilution away and the
    /// residual it was hiding — a tenth of a percent — shows.
    #[test]
    fn the_layered_bsdf_is_reciprocal() {
        let m = Pbr {
            base_color: [0.7, 0.5, 0.3],
            metallic: 0.2,
            roughness: 0.35,
            diffuse_roughness: 0.6,
            subsurface: 0.3,
            specular: 0.7,
            specular_tint: 0.4,
            sheen: 0.5,
            sheen_color: [0.9, 0.85, 1.0],
            sheen_roughness: 0.4,
            clearcoat: 0.6,
            clearcoat_roughness: 0.15,
            ..Default::default()
        };
        let dirs = [
            Vec3::new(0.3, 0.15, 0.94).normalize(),
            Vec3::new(0.85, 0.1, 0.52).normalize(),
            Vec3::new(-0.4, 0.6, 0.69).normalize(),
        ];
        for a in dirs {
            for b in dirs {
                // `bsdf_eval` returns f*cos, so divide the cosines back out
                // before comparing: f(a,b) == f(b,a).
                let ab = scale3(bsdf_eval(&m, a, b, 1.0, 0.0).0, 1.0 / b.z as f32);
                let ba = scale3(bsdf_eval(&m, b, a, 1.0, 0.0).0, 1.0 / a.z as f32);
                for c in 0..3 {
                    let scale = ab[c].abs().max(ba[c].abs()).max(1e-3);
                    assert!(
                        (ab[c] - ba[c]).abs() <= 1.5e-2 * scale,
                        "BSDF not reciprocal: {ab:?} vs {ba:?}"
                    );
                }
            }
        }
    }

    /// `E[f/pdf]` over the sampler lands on the albedo the evaluator's own
    /// quadrature reports — the check that catches a sampling routine drawing
    /// from a different distribution than its PDF claims, across every new
    /// parameter.
    // ─── transmission ─────────────────────────────────────────────────────

    /// The film is an *addition*: with no film the specular Fresnel must be
    /// the same function it always was, to the bit, or every scene that
    /// predates iridescence moves.
    #[test]
    fn a_zero_thickness_film_is_the_plain_fresnel_bit_for_bit() {
        let m = Pbr {
            base_color: [0.9, 0.7, 0.3],
            metallic: 0.8,
            roughness: 0.3,
            thin_film_ior: 2.0,
            ..Default::default()
        };
        assert_eq!(m.thin_film_thickness, 0.0);
        for wo in view_directions() {
            for wi in view_directions() {
                let (plain, _) = bsdf_eval(&m, wo, wi, 1.0, 0.0);
                let (spectral, _) = bsdf_eval(&m, wo, wi, 1.0, 550.0);
                assert_eq!(plain, spectral, "the RGB and hero paths must agree");
                for c in 0..3 {
                    let f0 = m.f0();
                    let wh = (wo + wi).normalize();
                    let _ = fresnel(f0, wo.dot(wh).max(0.0) as f32)[c];
                }
            }
        }
    }

    /// A film modulates the Fresnel, and the Fresnel depends on the half
    /// vector alone, so the layered BSDF stays as reciprocal as it was.
    #[test]
    fn an_iridescent_bsdf_is_reciprocal() {
        let m = Pbr {
            base_color: [0.9, 0.85, 0.8],
            metallic: 1.0,
            roughness: 0.25,
            thin_film_thickness: 420.0,
            thin_film_ior: 1.45,
            ..Default::default()
        };
        let plain = Pbr {
            thin_film_thickness: 0.0,
            ..m
        };
        for lambda in [0.0f32, 500.0] {
            for a in view_directions() {
                for b in view_directions() {
                    if a.z <= 0.0 || b.z <= 0.0 {
                        continue;
                    }
                    let ab = scale3(bsdf_eval(&m, a, b, 1.0, lambda).0, 1.0 / b.z as f32);
                    let ba = scale3(bsdf_eval(&m, b, a, 1.0, lambda).0, 1.0 / a.z as f32);
                    // The bar is the *same material without the film*: the
                    // film must not make the stack any less reciprocal than
                    // Turquin's view-only compensation already does.
                    let pab = scale3(bsdf_eval(&plain, a, b, 1.0, 0.0).0, 1.0 / b.z as f32);
                    let pba = scale3(bsdf_eval(&plain, b, a, 1.0, 0.0).0, 1.0 / a.z as f32);
                    for c in 0..3 {
                        let scale = ab[c].abs().max(ba[c].abs()).max(1e-4);
                        let pscale = pab[c].abs().max(pba[c].abs()).max(1e-4);
                        let bar = ((pab[c] - pba[c]).abs() / pscale + 1e-3).max(0.01);
                        assert!(
                            (ab[c] - ba[c]).abs() / scale <= bar * 1.05,
                            "{ab:?} vs {ba:?} at lambda {lambda}"
                        );
                    }
                }
            }
        }
    }

    /// Interference redistributes energy across the spectrum; it does not
    /// create any.
    #[test]
    fn an_iridescent_lobe_stays_under_one() {
        for thickness in [80.0f32, 300.0, 700.0] {
            for roughness in [0.1f32, 0.4, 0.9] {
                let m = Pbr {
                    base_color: [1.0; 3],
                    metallic: 1.0,
                    roughness,
                    thin_film_thickness: thickness,
                    thin_film_ior: 1.5,
                    ..Default::default()
                };
                for wo in view_directions() {
                    for lambda in [0.0f32, 480.0] {
                        let e =
                            integrate_hemisphere(220, |wi| bsdf_eval(&m, wo, wi, 1.0, lambda).0[1]);
                        assert!(e <= 1.02, "albedo {e} at d {thickness} r {roughness}");
                    }
                }
            }
        }
    }

    /// A boundary made of two planes, `z = 0` above and `z = -depth` below,
    /// with the medium between them. `depth = INFINITY` is a half-space.
    fn slab_trace(depth: f64) -> impl FnMut(Point3, Vec3) -> Option<(f64, Vec3)> {
        move |p: Point3, d: Vec3| {
            if d.z > 1e-12 {
                Some((-p.z / d.z, Vec3::new(0.0, 0.0, 1.0)))
            } else if d.z < -1e-12 && depth.is_finite() {
                Some(((-depth - p.z) / d.z, Vec3::new(0.0, 0.0, -1.0)))
            } else {
                None
            }
        }
    }

    /// The one quantitative claim the albedo inversion makes: a half-space of
    /// the material reflects `subsurface_color` back.
    ///
    /// This is the whole reason the parameter is a *surface* colour rather
    /// than a medium's single-scattering albedo. If the fit is wrong, or the
    /// walk's distance sampling or its per-channel MIS weights are wrong, the
    /// number that comes back is not the number that was asked for.
    ///
    /// Up to about 0.8 the fit is good to well under a percent. Above that it
    /// runs short — see `a_very_bright_medium_runs_a_little_dark`, which pins
    /// how short so it cannot quietly get worse.
    #[test]
    fn a_semi_infinite_slab_returns_its_own_colour() {
        for want in [[0.8f32, 0.5, 0.3], [0.5; 3], [0.2, 0.7, 0.35]] {
            let m = Pbr {
                subsurface: 1.0,
                subsurface_color: want,
                subsurface_radius: [0.01; 3],
                ..Default::default()
            };
            let n = Vec3::new(0.0, 0.0, 1.0);
            let mut rng = Rng::new(0x5b55_0001);
            let trials = 60_000;
            let mut acc = [0.0f64; 3];
            for _ in 0..trials {
                if let Some(e) = subsurface_walk(
                    &m,
                    Point3::new(0.0, 0.0, 0.0),
                    n,
                    &mut rng,
                    slab_trace(f64::INFINITY),
                ) {
                    for c in 0..3 {
                        acc[c] += e.weight[c] as f64;
                    }
                }
            }
            for c in 0..3 {
                let got = acc[c] / trials as f64;
                let target = want[c] as f64;
                assert!(
                    (got - target).abs() <= 0.03 * target,
                    "channel {c}: walked {got}, asked for {target}"
                );
            }
        }
    }

    /// The weight moves energy between two lobes; it does not add any.
    ///
    /// With `subsurface_color` set to the surface's own diffuse albedo, the
    /// total directional albedo — the diffuse lobe plus everything the walk
    /// brings back out — must be the same number at every `subsurface`
    /// weight. This is the invariant the composition exists to have, and the
    /// one an entry weight of `1/P(lobe)` instead of `subsurface/P(lobe)`
    /// silently breaks: nothing else in the model notices, and the object
    /// simply gets brighter as the knob turns.
    #[test]
    fn the_subsurface_weight_moves_energy_and_does_not_make_it() {
        let albedo = [0.6f32, 0.45, 0.3];
        let mut totals = Vec::new();
        for weight in [0.0f32, 0.5, 1.0] {
            let m = Pbr {
                base_color: albedo,
                roughness: 0.5,
                subsurface: weight,
                subsurface_color: albedo,
                subsurface_radius: [0.01; 3],
                ..Default::default()
            };
            let wo = Vec3::new(0.0, 0.0, 1.0);
            let mut rng = Rng::new(0x5b55_0005);
            let n = 40_000;
            let mut acc = [0.0f64; 3];
            for _ in 0..n {
                match bsdf_sample(&m, wo, 1.0, 0.0, &mut rng) {
                    Some(Sampled::Surface(_, f, pdf)) => {
                        for c in 0..3 {
                            acc[c] += (f[c] / pdf) as f64;
                        }
                    }
                    Some(Sampled::Subsurface(entry)) => {
                        if let Some(e) = subsurface_walk(
                            &m,
                            Point3::new(0.0, 0.0, 0.0),
                            Vec3::new(0.0, 0.0, 1.0),
                            &mut rng,
                            slab_trace(f64::INFINITY),
                        ) {
                            for c in 0..3 {
                                acc[c] += (entry[c] * e.weight[c]) as f64;
                            }
                        }
                    }
                    None => {}
                }
            }
            totals.push([acc[0] / n as f64, acc[1] / n as f64, acc[2] / n as f64]);
        }
        for t in &totals {
            for c in 0..3 {
                assert!(
                    (t[c] - totals[0][c]).abs() <= 0.02 * totals[0][c],
                    "albedo moved with the weight: {totals:?}"
                );
            }
        }
    }

    /// Chiang's fit is a cubic through the inversion of a transcendental
    /// function, and it gives out at the top of its range: a
    /// `subsurface_color` of 0.9 comes back as about 0.87, because the
    /// single-scattering albedo the fit picks (0.9964) genuinely reflects
    /// that much and not more. The right response is to state the number
    /// rather than to hide it behind a wider tolerance, so this pins it.
    ///
    /// It matters for white media and nothing else: skin, marble and rubber
    /// all sit well inside the range where the fit is exact to a fraction of
    /// a percent.
    #[test]
    fn a_very_bright_medium_runs_a_little_dark() {
        let m = Pbr {
            subsurface: 1.0,
            subsurface_color: [0.9; 3],
            subsurface_radius: [0.01; 3],
            ..Default::default()
        };
        let mut rng = Rng::new(0x5b55_0004);
        let trials = 60_000;
        let mut acc = 0.0f64;
        for _ in 0..trials {
            if let Some(e) = subsurface_walk(
                &m,
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                &mut rng,
                slab_trace(f64::INFINITY),
            ) {
                acc += e.weight[1] as f64;
            }
        }
        let got = acc / trials as f64;
        assert!((0.85..0.89).contains(&got), "0.9 came back as {got}");
    }

    /// A slab thin against its own mean free path has to let light out the
    /// far side, or nothing has been transported at all.
    #[test]
    fn a_thin_slab_transmits() {
        let m = Pbr {
            subsurface: 1.0,
            subsurface_color: [0.9; 3],
            subsurface_radius: [0.05; 3],
            ..Default::default()
        };
        let n = Vec3::new(0.0, 0.0, 1.0);
        let mut rng = Rng::new(0x5b55_0002);
        let trials = 20_000;
        let (mut through, mut back) = (0.0f64, 0.0f64);
        for _ in 0..trials {
            let Some(e) = subsurface_walk(
                &m,
                Point3::new(0.0, 0.0, 0.0),
                n,
                &mut rng,
                slab_trace(0.02),
            ) else {
                continue;
            };
            if e.normal.z < 0.0 {
                through += e.weight[1] as f64;
            } else {
                back += e.weight[1] as f64;
            }
        }
        let (through, back) = (through / trials as f64, back / trials as f64);
        assert!(through > 0.2, "a 0.4-mfp slab transmitted only {through}");
        assert!(back > 0.05, "and it must still reflect some: {back}");
        assert!(through + back <= 1.0, "energy {} > 1", through + back);
    }

    /// The walk moves light; it does not make any. Even a white medium in a
    /// half-space cannot return more than arrived.
    #[test]
    fn the_walk_never_returns_more_than_it_took() {
        for color in [[1.0f32; 3], [0.99; 3], [0.6, 0.9, 0.2]] {
            let m = Pbr {
                subsurface: 1.0,
                subsurface_color: color,
                subsurface_radius: [0.02, 0.01, 0.005],
                ..Default::default()
            };
            let mut rng = Rng::new(0x5b55_0003);
            let trials = 20_000;
            let mut acc = [0.0f64; 3];
            for _ in 0..trials {
                if let Some(e) = subsurface_walk(
                    &m,
                    Point3::new(0.0, 0.0, 0.0),
                    Vec3::new(0.0, 0.0, 1.0),
                    &mut rng,
                    slab_trace(f64::INFINITY),
                ) {
                    for c in 0..3 {
                        acc[c] += e.weight[c] as f64;
                    }
                }
            }
            for c in 0..3 {
                let e = acc[c] / trials as f64;
                assert!(e <= 1.0 + 1e-3, "channel {c} returned {e} for {color:?}");
            }
        }
    }

    /// `subsurface = 0` is the whole of what keeps every material written
    /// before the walk existed rendering as it did.
    #[test]
    fn subsurface_zero_leaves_the_other_lobes_alone() {
        let m = Pbr {
            base_color: [0.7, 0.5, 0.3],
            roughness: 0.35,
            diffuse_roughness: 0.6,
            sheen: 0.4,
            clearcoat: 0.5,
            subsurface_color: [0.2, 0.9, 0.4],
            subsurface_radius: [0.003; 3],
            ..Default::default()
        };
        assert_eq!(m.subsurface, 0.0);
        let w = lobe_weights(&m);
        assert_eq!(w[5], 0.0, "no subsurface lobe to pick");
        let plain = Pbr {
            subsurface_color: [1.0; 3],
            ..m
        };
        for wo in view_directions() {
            for wi in view_directions() {
                assert_eq!(
                    bsdf_eval(&m, wo, wi, 1.0, 0.0),
                    bsdf_eval(&plain, wo, wi, 1.0, 0.0),
                );
            }
        }
    }

    fn smooth_glass(ior: f32, roughness: f32) -> Pbr {
        Pbr {
            transmission: 1.0,
            ior,
            roughness,
            ..Default::default()
        }
    }

    /// The reflect/transmit split is the exact Fresnel, so at normal
    /// incidence it must be `((n−1)/(n+1))²` to the last digit an f32 has —
    /// not Schlick's fit, which agrees there but nowhere near grazing.
    #[test]
    fn dielectric_fresnel_at_normal_incidence_is_the_textbook_number() {
        for n in [1.33f32, 1.5, 1.52, 1.9, 2.42] {
            let r = fresnel_dielectric(1.0, n);
            let r0 = ((n - 1.0) / (n + 1.0)).powi(2);
            assert!((r - r0).abs() < 1e-6, "n={n}: {r} vs {r0}");
        }
    }

    /// At Brewster's angle the p-polarised reflectance vanishes, so the
    /// unpolarised average is exactly half of the s-polarised one. That is a
    /// property no Schlick approximation has, and it is the reason the exact
    /// formula is worth carrying.
    #[test]
    fn brewsters_angle_halves_the_unpolarised_reflectance() {
        let n = 1.5f32;
        let theta_b = n.atan();
        let cos_i = theta_b.cos();
        let sin_t = theta_b.sin() / n;
        let cos_t = (1.0 - sin_t * sin_t).sqrt();
        let rs = ((cos_i - n * cos_t) / (cos_i + n * cos_t)).powi(2);
        let r = fresnel_dielectric(cos_i, n);
        assert!((r - 0.5 * rs).abs() < 1e-5, "{r} vs {}", 0.5 * rs);
    }

    /// Total internal reflection is not a branch: it is what the same
    /// formula returns past the critical angle.
    #[test]
    fn past_the_critical_angle_everything_reflects() {
        let eta = 1.0f32 / 1.5; // leaving glass
        let critical = eta.asin();
        let just_inside = (critical - 0.02f32).cos();
        let just_outside = (critical + 0.02f32).cos();
        assert!(fresnel_dielectric(just_inside, eta) < 1.0);
        assert_eq!(fresnel_dielectric(just_outside, eta), 1.0);
    }

    /// A smooth glass surface refracts at Snell's angle. Sampled through the
    /// full BSDF machinery — VNDF facet, Fresnel branch, Walter half-vector —
    /// so this pins the lobe and not just `optics::refract`.
    #[test]
    fn a_smooth_slab_refracts_at_snells_angle() {
        let m = smooth_glass(1.5, 0.001);
        let mut rng = Rng::new(0x51a5);
        for deg in [10.0f64, 30.0, 50.0, 70.0] {
            let theta = deg.to_radians();
            let wo = Vec3::new(theta.sin(), 0.0, theta.cos());
            let expected_sin_t = theta.sin() / 1.5;
            let mut n = 0;
            for _ in 0..4000 {
                let Some((wi, _, _)) = bsdf_sample_surface(&m, wo, 1.5, 0.0, &mut rng) else {
                    continue;
                };
                if wi.z >= 0.0 {
                    continue; // reflected
                }
                let sin_t = (wi.x * wi.x + wi.y * wi.y).sqrt();
                assert!(
                    (sin_t - expected_sin_t).abs() < 5e-3,
                    "at {deg} deg: sin(theta_t) {sin_t} vs Snell {expected_sin_t}"
                );
                // Refraction stays in the plane of incidence, on the far side.
                assert!(wi.x < 0.0 && wi.y.abs() < 5e-3);
                n += 1;
            }
            assert!(n > 100, "at {deg} deg only {n} of 4000 samples transmitted");
        }
    }

    /// White furnace on a rough dielectric: with no absorption, everything
    /// that arrives must leave. Reflection and transmission together should
    /// sum to 1 — under, because a single-scattering Smith G drops the
    /// facet-to-facet bounces, and never over.
    #[test]
    fn a_rough_glass_sphere_closes_the_furnace() {
        for roughness in [0.05f32, 0.1, 0.2, 0.3] {
            let m = smooth_glass(1.5, roughness);
            for deg in [15.0f64, 45.0, 70.0] {
                let theta = deg.to_radians();
                let wo = Vec3::new(theta.sin(), 0.0, theta.cos());
                let mut rng = Rng::new(0xf00d + (deg as u64) * 31 + (roughness * 1e3) as u64);
                let n = 200_000;
                let mut sum = 0.0f64;
                for _ in 0..n {
                    if let Some((wi, f, pdf)) = bsdf_sample_surface(&m, wo, 1.5, 0.0, &mut rng) {
                        // The lobe's *energy*, so the η² radiance-transport
                        // scaling is taken back out on the transmitted half —
                        // see the note on `dielectric_eval`. A furnace is a
                        // statement about energy, not about the units a
                        // camera path happens to carry it in.
                        let undo = if wi.z < 0.0 { 1.5f32 * 1.5 } else { 1.0 };
                        sum += (f[0] * undo / pdf) as f64;
                    }
                }
                let albedo = sum / n as f64;
                assert!(
                    albedo <= 1.0 + 5e-3,
                    "r={roughness} {deg}deg: furnace gained energy ({albedo})"
                );
                assert!(
                    albedo >= 0.97,
                    "r={roughness} {deg}deg: furnace lost energy ({albedo})"
                );
            }
        }
    }

    /// The η² convention, checked where it matters: a ray that goes *into*
    /// glass and back *out* comes back to unit throughput. Entering scales
    /// radiance by 1/η² and leaving by η², and a slab is therefore a no-op —
    /// which is what makes the scaling a convention rather than a leak.
    #[test]
    fn a_slab_is_a_round_trip_no_op() {
        let m = smooth_glass(1.5, 0.08);
        let mut rng = Rng::new(0x5_1ab);
        let theta = 0.5f64;
        let wo = Vec3::new(theta.sin(), 0.0, theta.cos());
        let n = 200_000;
        let (mut sum, mut hits) = (0.0f64, 0u32);
        for _ in 0..n {
            // In.
            let Some((wi, f, pdf)) = bsdf_sample_surface(&m, wo, 1.5, 0.0, &mut rng) else {
                continue;
            };
            if wi.z >= 0.0 {
                continue;
            }
            let t1 = (f[0] / pdf) as f64;
            // Out through the far face: the far face's own normal faces the
            // other way, so the ray arrives at it from below and the local
            // frame flips.
            let wo2 = Vec3::new(-wi.x, -wi.y, -wi.z);
            let Some((_, f2, pdf2)) = bsdf_sample_surface(&m, wo2, 1.0 / 1.5, 0.0, &mut rng) else {
                continue;
            };
            sum += t1 * (f2[0] / pdf2) as f64;
            hits += 1;
        }
        let round_trip = sum / hits as f64;
        assert!(hits > n / 2, "only {hits} of {n} paths crossed both faces");
        assert!(
            (0.90..=1.0 + 5e-3).contains(&round_trip),
            "a slab should be transparent, got {round_trip}"
        );
    }

    /// Beer–Lambert, exactly: a material that transmits `c` over distance `d`
    /// transmits `c^(t/d)` over distance `t`.
    #[test]
    fn absorption_is_beer_lambert_to_the_letter() {
        let m = Pbr {
            transmission: 1.0,
            attenuation_color: [0.8, 0.9, 0.55],
            attenuation_distance: 0.25,
            ..Default::default()
        };
        let sigma = m.extinction();
        for t in [0.0f32, 0.1, 0.25, 1.0, 3.0] {
            for c in 0..3 {
                let got = (-sigma[c] * t).exp();
                let want = m.attenuation_color[c].powf(t / m.attenuation_distance);
                assert!(
                    (got - want).abs() < 1e-6,
                    "channel {c} at {t}: {got} vs {want}"
                );
            }
        }
        // One attenuation distance reproduces the colour that named it.
        for c in 0..3 {
            let got = (-sigma[c] * m.attenuation_distance).exp();
            assert!((got - m.attenuation_color[c]).abs() < 1e-6);
        }
        // The default is transparent.
        assert_eq!(Pbr::default().extinction(), [0.0; 3]);
    }

    /// A white beam through a 60° N-BK7 prism comes out spread, and the
    /// spread is the one Snell gives for the F and C lines' indices.
    ///
    /// Traced through `bsdf_sample` at both wavelengths — two refractions,
    /// each at the index `index_at` reports — so this pins the whole chain
    /// from Sellmeier through `eta` to the Walter half-vector against the
    /// two-surface analytic answer.
    #[test]
    fn a_bk7_prism_spreads_f_to_c_by_the_analytic_angle() {
        let m = Pbr {
            transmission: 1.0,
            roughness: 0.001,
            ior: 1.5168,
            sellmeier: Some(crate::spectrum::BK7_SELLMEIER),
            ..Default::default()
        };
        // Apex 60 degrees; the entry face normal is +Z locally.
        let apex = 60f64.to_radians();
        let incidence = 45f64.to_radians();

        // Deviation through a prism of apex A at incidence i1:
        //   r1 = asin(sin i1 / n),  r2 = A − r1,  i2 = asin(n sin r2)
        //   D  = i1 + i2 − A
        let analytic = |n: f64| {
            let r1 = (incidence.sin() / n).asin();
            let r2 = apex - r1;
            let i2 = (n * r2.sin()).asin();
            incidence + i2 - apex
        };

        // The same two refractions, but each one driven by the material's own
        // sampled lobe.
        let traced = |lambda_nm: f64| {
            let n = m.index_at(Some(lambda_nm)) as f64;
            let mut rng = Rng::new(0xbeef_0000 + lambda_nm as u64);
            // Entry: air into glass.
            let wo = Vec3::new(incidence.sin(), 0.0, incidence.cos());
            let mut r1 = None;
            for _ in 0..8000 {
                if let Some((wi, _, _)) = bsdf_sample_surface(&m, wo, n as f32, 0.0, &mut rng) {
                    if wi.z < 0.0 {
                        r1 = Some((wi.x * wi.x + wi.y * wi.y).sqrt().asin());
                        break;
                    }
                }
            }
            let r1 = r1.expect("the entry face transmitted nothing");
            // Exit: glass into air, at the second face.
            let r2 = apex - r1;
            let wo2 = Vec3::new(r2.sin(), 0.0, r2.cos());
            let mut i2 = None;
            for _ in 0..8000 {
                if let Some((wi, _, _)) =
                    bsdf_sample_surface(&m, wo2, (1.0 / n) as f32, 0.0, &mut rng)
                {
                    if wi.z < 0.0 {
                        i2 = Some((wi.x * wi.x + wi.y * wi.y).sqrt().asin());
                        break;
                    }
                }
            }
            incidence + i2.expect("the exit face transmitted nothing") - apex
        };

        let (f_nm, c_nm) = (486.13, 656.27);
        let n_f = m.index_at(Some(f_nm)) as f64;
        let n_c = m.index_at(Some(c_nm)) as f64;
        // N-BK7's datasheet indices at the two lines.
        assert!((n_f - 1.52238).abs() < 1e-3, "n_F = {n_f}");
        assert!((n_c - 1.51432).abs() < 1e-3, "n_C = {n_c}");

        let spread_analytic = analytic(n_f) - analytic(n_c);
        let spread_traced = traced(f_nm) - traced(c_nm);
        assert!(
            spread_analytic > 0.0,
            "blue must deviate more than red ({spread_analytic})"
        );
        assert!(
            (spread_traced - spread_analytic).abs() < 2e-3,
            "traced spread {} rad vs analytic {} rad",
            spread_traced,
            spread_analytic
        );
        // For the record: about 0.75 degrees of fan between the F and C lines.
        assert!(
            (spread_analytic.to_degrees() - 0.75).abs() < 0.1,
            "{} deg",
            spread_analytic.to_degrees()
        );
    }

    /// The spread has to read as a rainbow in the right order: the long end
    /// red, the short end violet-blue, with green between.
    #[test]
    fn the_hero_weights_run_red_to_violet() {
        let red = crate::spectrum::hero_weight(650.0);
        let green = crate::spectrum::hero_weight(540.0);
        let blue = crate::spectrum::hero_weight(450.0);
        assert!(red[0] > red[1] && red[0] > red[2], "650nm reads {red:?}");
        assert!(
            green[1] > green[0] && green[1] > green[2],
            "540nm reads {green:?}"
        );
        assert!(
            blue[2] > blue[0] && blue[2] > blue[1],
            "450nm reads {blue:?}"
        );
    }

    /// The invariant MIS depends on: the PDF `bsdf_sample` returns is the PDF
    /// `bsdf_eval` reports for the direction it drew — on both sides of the
    /// surface, transmission included.
    #[test]
    fn the_dielectric_sample_pdf_matches_its_eval_pdf() {
        let mut rng = Rng::new(0x51de);
        for roughness in [0.02f32, 0.15, 0.4] {
            for thin in [false, true] {
                let m = Pbr {
                    transmission: 1.0,
                    roughness,
                    ior: 1.52,
                    thin_walled: thin,
                    ..Default::default()
                };
                for _ in 0..2000 {
                    let theta = rng.f64() * 1.4;
                    let wo = Vec3::new(theta.sin(), 0.0, theta.cos());
                    let Some((wi, f, pdf)) = bsdf_sample_surface(&m, wo, 1.52, 0.0, &mut rng)
                    else {
                        continue;
                    };
                    let (f2, pdf2) = bsdf_eval(&m, wo, wi, 1.52, 0.0);
                    assert!(
                        (pdf - pdf2).abs() <= 1e-4 * pdf.max(1.0),
                        "r={roughness} thin={thin}: {pdf} vs {pdf2}"
                    );
                    assert!((f[0] - f2[0]).abs() <= 1e-4 * f[0].max(1.0));
                }
            }
        }
    }

    /// A thin-walled sheet transmits straight through: no lateral offset, and
    /// at low roughness the exit direction is the entry direction.
    #[test]
    fn a_thin_wall_transmits_straight_through() {
        let m = Pbr {
            transmission: 1.0,
            roughness: 0.001,
            ior: 1.52,
            thin_walled: true,
            ..Default::default()
        };
        let mut rng = Rng::new(0x7417);
        let theta = 0.7f64;
        let wo = Vec3::new(theta.sin(), 0.0, theta.cos());
        let mut n = 0;
        for _ in 0..4000 {
            let Some((wi, _, _)) = bsdf_sample_surface(&m, wo, 1.52, 0.0, &mut rng) else {
                continue;
            };
            if wi.z >= 0.0 {
                continue;
            }
            assert!(
                (wi.x + wo.x).abs() < 5e-3 && (wi.z + wo.z).abs() < 5e-3,
                "{wi:?}"
            );
            n += 1;
        }
        assert!(n > 3000, "a sheet should mostly transmit, got {n}/4000");
    }

    /// The whole point of the defaults: an opaque material's five lobe
    /// weights are its old four, unchanged, and the fifth is zero.
    #[test]
    fn transmission_zero_leaves_the_opaque_weights_alone() {
        for m in [
            Pbr::default(),
            Pbr::metal([0.9, 0.8, 0.5], 0.2),
            Pbr::plastic([0.2, 0.4, 0.8], 0.3, 0.6),
        ] {
            let w = lobe_weights(&m);
            assert_eq!(w[4], 0.0);
            let s: f32 = w[0] + w[1] + w[2] + w[3];
            assert!((s - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn sampling_every_lobe_recovers_the_evaluated_albedo() {
        let base = Pbr {
            base_color: [0.75, 0.6, 0.45],
            roughness: 0.35,
            ..Default::default()
        };
        let cases: [(&str, Pbr); 6] = [
            (
                "diffuse-rough",
                Pbr {
                    diffuse_roughness: 0.9,
                    ..base
                },
            ),
            (
                "subsurface",
                Pbr {
                    subsurface: 0.8,
                    diffuse_roughness: 0.4,
                    ..base
                },
            ),
            (
                "sheen",
                Pbr {
                    sheen: 0.8,
                    sheen_roughness: 0.35,
                    ..base
                },
            ),
            (
                "coat",
                Pbr {
                    clearcoat: 0.9,
                    clearcoat_roughness: 0.12,
                    ..base
                },
            ),
            (
                "metal",
                Pbr {
                    metallic: 1.0,
                    roughness: 0.6,
                    ..base
                },
            ),
            (
                "everything",
                Pbr {
                    metallic: 0.4,
                    diffuse_roughness: 0.7,
                    subsurface: 0.3,
                    specular: 0.9,
                    specular_tint: 0.5,
                    sheen: 0.6,
                    sheen_roughness: 0.5,
                    clearcoat: 0.5,
                    anisotropy: 0.6,
                    ..base
                },
            ),
        ];
        for (name, m) in cases {
            for wo in [
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.6, 0.2, 0.77).normalize(),
                Vec3::new(0.9, 0.1, 0.42).normalize(),
            ] {
                let reference =
                    integrate_hemisphere(200, |wi| bsdf_eval(&m, wo, wi, 1.0, 0.0).0[0]);
                let mut rng = Rng::new(29);
                let n = 200_000;
                let mut sum = 0.0f64;
                for _ in 0..n {
                    if let Some((_wi, f, pdf)) = bsdf_sample_surface(&m, wo, 1.0, 0.0, &mut rng) {
                        sum += (f[0] / pdf) as f64;
                    }
                }
                let sampled = (sum / n as f64) as f32;
                assert!(
                    (sampled - reference).abs() <= 0.01 * reference.max(0.05),
                    "{name}: sampler says {sampled}, evaluator says {reference} (mu {})",
                    wo.z
                );
            }
        }
    }

    /// A white furnace test: with no lights and a uniform environment, a
    /// pure-white rough dielectric must not create or destroy much energy.
    #[test]
    fn furnace_conserves_energy_roughly() {
        // Anisotropy redistributes energy within the lobe; it must not
        // create or destroy any. Grazing views are included because that is
        // where a wrong anisotropic masking term shows up as gain.
        for aniso in [0.0, 0.5, 0.9, -0.5, -0.9] {
            for wo in [
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.6, 0.2, 0.77).normalize(),
            ] {
                let m = Pbr {
                    base_color: [1.0, 1.0, 1.0],
                    metallic: 0.0,
                    roughness: 0.5,
                    anisotropy: aniso,
                    clearcoat: 0.0,
                    ..Default::default()
                };
                let mut rng = Rng::new(11);
                let n = 20000;
                let mut sum = 0.0f32;
                for _ in 0..n {
                    if let Some((_wi, f, pdf)) = bsdf_sample_surface(&m, wo, 1.0, 0.0, &mut rng) {
                        sum += f[0] / pdf;
                    }
                }
                let albedo = sum / n as f32;
                assert!(
                    (0.75..=1.05).contains(&albedo),
                    "directional albedo {albedo} outside plausible range \
                     (anisotropy {aniso}, wo {wo:?})"
                );
            }
        }
    }

    // ── environment maps ──────────────────────────────────────────────────

    /// A map of constant radiance `c`.
    fn uniform_map(w: usize, h: usize, c: f32) -> EnvMap {
        EnvMap::new(w, h, vec![[c, c, c]; w * h]).expect("uniform map")
    }

    /// A deliberately high-frequency map: a dim surround with one small,
    /// very bright patch — the case BSDF-only sampling handles badly and the
    /// CDF exists for.
    fn structured_map() -> EnvMap {
        let (w, h) = (64usize, 32usize);
        let mut px = vec![[0.05f32, 0.06, 0.08]; w * h];
        for j in 6..10 {
            for i in 20..25 {
                px[j * w + i] = [40.0, 38.0, 34.0];
            }
        }
        // A second, low patch near the horizon on the far side.
        for j in 16..18 {
            for i in 50..56 {
                px[j * w + i] = [6.0, 6.5, 8.0];
            }
        }
        EnvMap::new(w, h, px).expect("structured map")
    }

    /// Uniform directions on the sphere, for reference integration.
    fn uniform_sphere(r1: f64, r2: f64) -> Vec3 {
        let z = 1.0 - 2.0 * r1;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let phi = std::f64::consts::TAU * r2;
        Vec3::new(r * phi.cos(), r * phi.sin(), z)
    }

    /// The single strongest guard on the PDF conversion: a density over the
    /// sphere must integrate to 1. A wrong `2*pi^2`, or a forgotten
    /// `sin(theta)`, shows up here immediately.
    #[test]
    fn env_pdf_integrates_to_one_over_the_sphere() {
        for map in [uniform_map(32, 16, 1.0), structured_map()] {
            let mut rng = Rng::new(3);
            let n = 200_000;
            let mut sum = 0.0f64;
            for _ in 0..n {
                let d = uniform_sphere(rng.f64(), rng.f64());
                // Uniform-sphere pdf is 1/4pi, so the estimator is 4pi * mean.
                sum += map.pdf(d) as f64;
            }
            let integral = sum / n as f64 * 4.0 * std::f64::consts::PI;
            assert!(
                (integral - 1.0).abs() < 0.02,
                "environment PDF integrates to {integral}, expected 1"
            );
        }
    }

    /// White furnace, at the estimator level: with a uniform environment of
    /// radiance `c`, importance sampling must recover the analytic
    /// irradiance `pi * c` over a hemisphere. This is the test that catches
    /// a PDF-conversion constant that merely *looks* plausible in an image.
    #[test]
    fn uniform_env_sampling_recovers_irradiance() {
        let c = 0.75f32;
        let map = uniform_map(32, 16, c);
        let n = 100_000;
        let mut rng = Rng::new(5);
        let mut sum = 0.0f64;
        for _ in 0..n {
            let Some((d, li, pdf)) = map.sample(rng.f64(), rng.f64()) else {
                continue;
            };
            if d.z <= 0.0 {
                continue;
            }
            sum += (li[0] as f64) * d.z / pdf as f64;
        }
        let irradiance = sum / n as f64;
        let expected = std::f64::consts::PI * c as f64;
        assert!(
            (irradiance - expected).abs() < 0.02 * expected,
            "irradiance {irradiance}, expected {expected}"
        );
    }

    /// The same integral, on a high-frequency map, estimated two ways. They
    /// must agree — importance sampling may only change the variance, never
    /// the answer.
    #[test]
    fn structured_env_sampling_agrees_with_uniform_sphere_sampling() {
        let map = structured_map();
        let n = 400_000;

        let mut rng = Rng::new(9);
        let mut is_sum = 0.0f64;
        for _ in 0..n {
            if let Some((d, li, pdf)) = map.sample(rng.f64(), rng.f64()) {
                if d.z > 0.0 {
                    is_sum += (li[0] as f64) * d.z / pdf as f64;
                }
            }
        }
        let importance = is_sum / n as f64;

        let mut rng = Rng::new(10);
        let mut u_sum = 0.0f64;
        let uniform_pdf = 1.0 / (4.0 * std::f64::consts::PI);
        for _ in 0..n {
            let d = uniform_sphere(rng.f64(), rng.f64());
            if d.z > 0.0 {
                u_sum += (map.radiance(d)[0] as f64) * d.z / uniform_pdf;
            }
        }
        let reference = u_sum / n as f64;

        assert!(
            (importance - reference).abs() < 0.05 * reference,
            "importance-sampled irradiance {importance} disagrees with \
             uniform-sampled reference {reference}"
        );
    }

    /// Rotation must actually move the environment, and must not disturb the
    /// PDF normalisation (the CDF is reused across rotations).
    #[test]
    fn rotation_moves_the_environment_without_breaking_the_pdf() {
        let map = structured_map();
        let spun = structured_map().with_rotation_deg(90.0);
        // Aim straight at the bright patch in the unrotated map; spinning the
        // environment must move it out from under this direction.
        let d = map.direction(0.34, 0.24);
        assert!(
            map.radiance(d)[0] > 10.0,
            "probe direction missed the bright patch"
        );
        assert!(
            (map.radiance(d)[0] - spun.radiance(d)[0]).abs() > 1e-6,
            "rotating the environment changed nothing"
        );

        let mut rng = Rng::new(21);
        let n = 200_000;
        let mut sum = 0.0f64;
        for _ in 0..n {
            sum += spun.pdf(uniform_sphere(rng.f64(), rng.f64())) as f64;
        }
        let integral = sum / n as f64 * 4.0 * std::f64::consts::PI;
        assert!(
            (integral - 1.0).abs() < 0.02,
            "rotated environment PDF integrates to {integral}"
        );
    }

    /// A degenerate (all-black) map must not be importance-sampled, and must
    /// not poison the MIS weights.
    #[test]
    fn black_env_map_is_not_importance_sampled() {
        let env = Environment::image(uniform_map(8, 4, 0.0));
        assert!(!env.is_importance_sampled());
        assert_eq!(env.pdf(Vec3::new(0.0, 0.0, 1.0)), 0.0);
        assert!(env.sample(0.5, 0.5).is_none());
    }

    /// End-to-end white furnace: a uniform HDRI and the analytic environment
    /// set to the same constant colour must render to the same image, even
    /// though one is integrated by BSDF sampling alone and the other by a
    /// three-way MIS mix. Any error in the image-space to solid-angle PDF
    /// conversion shows up as a systematic brightness difference here.
    #[test]
    fn uniform_env_map_matches_analytic_constant_environment() {
        let c = 0.6f32;
        let material = Pbr {
            base_color: [1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 0.6,
            clearcoat: 0.0,
            ..Default::default()
        };
        let scene_with = |env: Environment| Scene::<TriMesh> {
            objects: vec![Object::new(Arc::new(Bvh::build(cube_mesh())), material)],
            // No area lights: the environment must be the only illuminant,
            // or light sampling would mask a bad environment PDF.
            lights: Vec::new(),
            env,
            sun: None,
            ground: None,
            splats: None,
        };
        let cam = test_camera();
        let opts = PathTraceOptions {
            spp: 220,
            max_depth: 4,
            firefly_clamp: None,
            ..Default::default()
        };

        let mean = |scene: &Scene<TriMesh>| -> f64 {
            let film = render(scene, &cam, 24, 24, &opts);
            film.rgb.iter().map(|v| *v as f64).sum::<f64>() / film.rgb.len() as f64
        };

        let analytic = mean(&scene_with(Environment::constant([c, c, c])));
        let image = mean(&scene_with(Environment::image(uniform_map(64, 32, c))));

        assert!(analytic > 0.1, "reference render was black");
        assert!(
            (image - analytic).abs() < 0.02 * analytic,
            "uniform HDRI rendered at {image}, analytic constant environment \
             at {analytic} — the environment PDF conversion is off"
        );
    }

    /// A sun-lit Lambertian plane must receive exactly `E·cos(theta)`.
    ///
    /// The analytic answer is the whole point of a directional light with a
    /// finite disc: irradiance is the integral of radiance times cosine over
    /// the cone, which for a small cone is `E·cos(theta)` to within the
    /// disc's own width. A perfectly white Lambertian surface with `ior = 1`
    /// (so the specular lobe's F0 vanishes) reflects `E·cos(theta)/pi`, so
    /// the render inverts back to the irradiance directly.
    #[test]
    fn a_sunlit_plane_matches_the_analytic_irradiance() {
        for theta_deg in [0.0f64, 30.0, 60.0] {
            let theta = theta_deg.to_radians();
            let sun = Sun::new(
                Vec3::new(theta.sin(), 0.0, theta.cos()),
                0.01,
                [2.0, 2.0, 2.0],
            );
            let e = sun.irradiance[0] as f64;

            let scene = Scene::<TriMesh> {
                objects: Vec::new(),
                lights: Vec::new(),
                env: Environment::constant([0.0, 0.0, 0.0]),
                sun: Some(sun),
                ground: Some(Ground {
                    z: 0.0,
                    material: Pbr {
                        base_color: [1.0, 1.0, 1.0],
                        metallic: 0.0,
                        roughness: 1.0,
                        clearcoat: 0.0,
                        ior: 1.0,
                        ..Default::default()
                    },
                    shadow_catcher: false,
                }),
                splats: None,
            };
            let cam = Camera::look_at(
                Point3::new(0.0, 0.0, 4.0),
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                30.0,
            );
            let opts = PathTraceOptions {
                spp: 4096,
                max_depth: 1,
                denoise: false,
                firefly_clamp: None,
                seed: 7,
                ..Default::default()
            };
            let film = render(&scene, &cam, 8, 8, &opts);
            let n = (film.width * film.height) as usize;
            let mean: f64 = (0..n).map(|i| film.rgb[i * 3] as f64).sum::<f64>() / n as f64;
            let measured = mean * core::f64::consts::PI;
            let expected = e * theta.cos();
            let rel = (measured - expected).abs() / expected;
            assert!(
                rel < 0.01,
                "theta={theta_deg}: irradiance {measured} vs analytic {expected} ({:.2}% off)",
                rel * 100.0
            );
        }
    }

    /// The sun's two strategies must sum to one: NEE plus the BSDF ray that
    /// lands in the disc has to give the same answer as either alone would
    /// with the other switched off.
    #[test]
    fn the_sun_disc_is_visible_to_a_ray_that_finds_it() {
        let sun = Sun::new(Vec3::new(0.0, 0.0, 1.0), 0.05, [1.0, 1.0, 1.0]);
        // Radiance times solid angle is irradiance, by construction.
        let l = sun.radiance()[0] as f64;
        assert!((l * sun.solid_angle() - 1.0).abs() < 1e-6);
        assert_eq!(
            sun.radiance_in(Vec3::new(0.0, 0.0, 1.0))[0],
            sun.radiance()[0]
        );
        assert_eq!(sun.radiance_in(Vec3::new(1.0, 0.0, 0.0))[0], 0.0);
        assert!(sun.pdf(Vec3::new(0.0, 0.0, 1.0)) > 0.0);
        assert_eq!(sun.pdf(Vec3::new(0.0, 1.0, 0.0)), 0.0);
        // Every sample must land inside the cone.
        for k in 0..64 {
            let (d, _, pdf) = sun.sample(k as f64 / 64.0, (k * 7 % 64) as f64 / 64.0);
            assert!(
                d.z >= sun.cos_radius() - 1e-12,
                "sample outside the cone: {d:?}"
            );
            assert!(((pdf as f64) * sun.solid_angle() - 1.0).abs() < 1e-5);
        }
    }

    // ─── the splat volume in the integrator ───────────────────────────────
    //
    // A splat cloud is composited, not shaded, so what these check is the
    // arithmetic of the walk — that `C += T·α·c; T *= (1 − α)` happens in
    // front of the analytic scene, in the right order, and on shadow rays.

    mod splat_volume {
        use super::*;
        use crate::splats::Splats;

        /// The degree-0 SH coefficient that makes a splat render as `c`.
        fn dc(c: [f32; 3]) -> [f32; 3] {
            const SH_C0: f32 = 0.282_094_79;
            [
                (c[0] - 0.5) / SH_C0,
                (c[1] - 0.5) / SH_C0,
                (c[2] - 0.5) / SH_C0,
            ]
        }

        /// Isotropic splats on the z axis, `(z, opacity, colour)` each.
        fn cloud(items: &[(f32, f32, [f32; 3])]) -> Arc<Bvh<Splats>> {
            let positions: Vec<[f32; 3]> = items.iter().map(|it| [0.0, 0.0, it.0]).collect();
            let scales = vec![[0.2f32; 3]; items.len()];
            let quats = vec![[1.0f32, 0.0, 0.0, 0.0]; items.len()];
            let opacities: Vec<f32> = items.iter().map(|it| it.1).collect();
            let sh: Vec<[f32; 3]> = items.iter().map(|it| dc(it.2)).collect();
            Arc::new(Bvh::build(Splats::from_parts(
                &positions, &scales, &quats, &opacities, &sh,
            )))
        }

        /// An emissive floor at z = 0 under a black sky: a "plane" whose
        /// radiance is exactly 1, so anything the camera reads that is not 1
        /// came from the cloud.
        fn scene(splats: Option<Arc<Bvh<Splats>>>) -> Scene<TriMesh> {
            Scene {
                objects: Vec::new(),
                lights: Vec::new(),
                env: Environment::constant([0.0; 3]),
                sun: None,
                ground: Some(Ground {
                    z: 0.0,
                    material: Pbr {
                        base_color: [0.0; 3],
                        roughness: 1.0,
                        emissive: [1.0; 3],
                        ..Default::default()
                    },
                    shadow_catcher: false,
                }),
                splats,
            }
        }

        /// The radiance of one ray straight down the z axis at the floor.
        fn down(scene: &Scene<TriMesh>) -> [f32; 3] {
            let accel = SceneAccel::build(scene);
            let opts = PathTraceOptions::default();
            let ray = Ray::new(Point3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
            let mut rng = Rng::new(7);
            radiance(scene, &accel, &opts, None, None, ray, &mut rng).0
        }

        #[test]
        fn an_opaque_splat_hides_the_plane() {
            let s = scene(Some(cloud(&[(2.5, 1.0, [0.25, 0.5, 0.75])])));
            let l = down(&s);
            assert!((l[0] - 0.25).abs() < 1e-4, "{l:?}");
            assert!((l[1] - 0.5).abs() < 1e-4, "{l:?}");
            assert!((l[2] - 0.75).abs() < 1e-4, "the floor's 1.0 is gone: {l:?}");
        }

        #[test]
        fn a_half_transparent_splat_composites_fifty_fifty() {
            let s = scene(Some(cloud(&[(2.5, 0.5, [0.0, 0.0, 0.0])])));
            let l = down(&s);
            // Black cloud at α = ½ over an emissive floor at 1: half the
            // floor survives, and none of the cloud's own colour shows.
            for ch in 0..3 {
                assert!((l[ch] - 0.5).abs() < 1e-4, "{l:?}");
            }
            // And with a white cloud instead, the two halves add back to one.
            let s = scene(Some(cloud(&[(2.5, 0.5, [1.0, 1.0, 1.0])])));
            let l = down(&s);
            for ch in 0..3 {
                assert!((l[ch] - 1.0).abs() < 1e-4, "{l:?}");
            }
        }

        #[test]
        fn the_nearer_splat_dominates() {
            // Red in front at z = 3, blue behind at z = 1, both α = ½.
            let s = scene(Some(cloud(&[
                (3.0, 0.5, [1.0, 0.0, 0.0]),
                (1.0, 0.5, [0.0, 0.0, 1.0]),
            ])));
            let l = down(&s);
            // Front to back: ½·red, then ½·½·blue, then ¼ of the floor —
            // and the floor is white, so it adds ¼ to every channel.
            assert!((l[0] - (0.5 + 0.25)).abs() < 1e-4, "red at full T: {l:?}");
            assert!((l[2] - (0.25 + 0.25)).abs() < 1e-4, "blue at half T: {l:?}");
            assert!(l[0] > l[2], "the nearer colour weighs more: {l:?}");
            // Green sees only the floor's quarter.
            assert!((l[1] - 0.25).abs() < 1e-4, "{l:?}");
            // Swapping the depths swaps the weights, which is the whole test.
            let s = scene(Some(cloud(&[
                (3.0, 0.5, [0.0, 0.0, 1.0]),
                (1.0, 0.5, [1.0, 0.0, 0.0]),
            ])));
            let l2 = down(&s);
            assert!((l2[2] - 0.75).abs() < 1e-4, "{l2:?}");
            assert!((l2[0] - 0.5).abs() < 1e-4, "{l2:?}");
            assert!(l2[2] > l2[0], "swapping the depths swaps the weights");
        }

        #[test]
        fn a_shadow_ray_is_attenuated_by_the_cloud() {
            let s = scene(Some(cloud(&[(2.5, 0.5, [0.0; 3])])));
            let accel = SceneAccel::build(&s);
            // Upward, from just under the cloud past it — the ground plane is
            // below the origin, so it does not block.
            let tr = s
                .shadow_transmittance(
                    &accel,
                    Point3::new(0.0, 0.0, 1.0),
                    Vec3::new(0.0, 0.0, 1.0),
                    10.0,
                )
                .expect("a half-transparent splat is not a blocker");
            for ch in 0..3 {
                assert!((tr[ch] - 0.5).abs() < 1e-4, "{tr:?}");
            }
            // Two of them multiply.
            let s = scene(Some(cloud(&[(2.5, 0.5, [0.0; 3]), (3.5, 0.5, [0.0; 3])])));
            let accel = SceneAccel::build(&s);
            let tr = s
                .shadow_transmittance(
                    &accel,
                    Point3::new(0.0, 0.0, 1.0),
                    Vec3::new(0.0, 0.0, 1.0),
                    10.0,
                )
                .expect("still not a blocker");
            assert!((tr[0] - 0.25).abs() < 1e-4, "{tr:?}");
            // An opaque one is.
            let s = scene(Some(cloud(&[(2.5, 1.0, [0.0; 3])])));
            let accel = SceneAccel::build(&s);
            assert!(
                s.shadow_transmittance(
                    &accel,
                    Point3::new(0.0, 0.0, 1.0),
                    Vec3::new(0.0, 0.0, 1.0),
                    10.0,
                )
                .is_none(),
                "an opaque splat stops the light"
            );
        }

        #[test]
        fn a_bounce_ray_that_misses_everything_returns_the_cloud() {
            // The environment-with-depth claim: no analytic geometry at all,
            // so the only thing a ray can find is the captured field.
            let mut s = scene(Some(cloud(&[(2.5, 1.0, [0.3, 0.4, 0.5])])));
            s.ground = None;
            let l = down(&s);
            assert!((l[0] - 0.3).abs() < 1e-4, "{l:?}");
            assert!((l[2] - 0.5).abs() < 1e-4, "{l:?}");
        }
    }

    // ─── low-discrepancy camera sampling and adaptive sampling ────────────

    #[test]
    fn radical_inverse_matches_hand_computed_values() {
        // Base 2: 1 -> 0.1b = 1/2, 2 -> 0.01b = 1/4, 3 -> 0.11b = 3/4.
        assert_eq!(radical_inverse::<2>(0), 0.0);
        assert!((radical_inverse::<2>(1) - 0.5).abs() < 1e-12);
        assert!((radical_inverse::<2>(2) - 0.25).abs() < 1e-12);
        assert!((radical_inverse::<2>(3) - 0.75).abs() < 1e-12);
        // Base 3: 1 -> 1/3, 2 -> 2/3, 4 = 11_3 -> 0.11_3 = 4/9.
        assert!((radical_inverse::<3>(1) - 1.0 / 3.0).abs() < 1e-12);
        assert!((radical_inverse::<3>(2) - 2.0 / 3.0).abs() < 1e-12);
        assert!((radical_inverse::<3>(4) - 4.0 / 9.0).abs() < 1e-12);
        // The rotation stays on the torus whatever the offset.
        for &x in &[0.0, 0.25, 0.99] {
            for &o in &[0.0, 0.5, 0.999] {
                let v = cp_rotate(x, o);
                assert!((0.0..1.0).contains(&v), "cp_rotate({x}, {o}) = {v}");
            }
        }
    }

    /// The whole point of the point set: no gaps and no clumps. A purely
    /// random 2D sample would routinely leave a stratum empty at these
    /// counts, which is the aliasing this replaced.
    #[test]
    fn camera_point_set_covers_every_stratum() {
        let n = 64u32;
        let sample = |s: u32, ox: f64, oy: f64| {
            (
                cp_rotate(radical_inverse::<2>(s as u64), ox),
                cp_rotate(radical_inverse::<3>(s as u64), oy),
            )
        };
        for &(ox, oy) in &[(0.0, 0.0), (0.317, 0.61), (0.94, 0.02)] {
            let mut hits = [[0u32; 8]; 8];
            for s in 0..n {
                let (x, y) = sample(s, ox, oy);
                assert!((0.0..1.0).contains(&x) && (0.0..1.0).contains(&y));
                hits[(y * 8.0) as usize][(x * 8.0) as usize] += 1;
            }
            let worst = hits.iter().flatten().copied().max().unwrap();
            assert!(
                worst <= 3,
                "rotated set clumped {worst} samples in a stratum"
            );
        }
    }

    /// The Halton camera set beats four fresh uniforms at the job the camera
    /// dimensions actually do: estimating how much of a pixel's footprint a
    /// silhouette covers, on a scene that is flat either side of the edge.
    ///
    /// Measured per pixel and pooled, because the per-pixel Cranley-Patterson
    /// rotation makes a *single* sample of the Halton set exactly as random
    /// as a uniform draw — the two sets only separate once a pixel takes more
    /// than one, which is the smallest count at which the comparison means
    /// anything. Each "pixel" here is one draw of the rotation.
    #[test]
    fn the_halton_camera_set_beats_random_jitter_on_a_flat_edge() {
        // Coverage of the unit pixel square by the half-plane x + y < c, for
        // a random edge offset c per pixel — a silhouette crossing the pixel
        // footprint anywhere. The exact area is known, and the estimator is
        // the fraction of samples that land under the edge.
        let n = 16u32;
        let pixels = 8192u32;
        let mut halton_err = 0.0f64;
        let mut random_err = 0.0f64;
        for pixel in 0..pixels {
            let mut rng = Rng::new(0xA17E_u64 ^ pixel as u64);
            let rot = [rng.f64(), rng.f64()];
            let c = 2.0 * rng.f64();
            let exact = if c <= 1.0 {
                0.5 * c * c
            } else {
                1.0 - 0.5 * (2.0 - c) * (2.0 - c)
            };
            let mut h = 0.0f64;
            let mut r = 0.0f64;
            for s in 0..n {
                let hx = cp_rotate(radical_inverse::<2>(s as u64), rot[0]);
                let hy = cp_rotate(radical_inverse::<3>(s as u64), rot[1]);
                if hx + hy < c {
                    h += 1.0;
                }
                if rng.f64() + rng.f64() < c {
                    r += 1.0;
                }
            }
            halton_err += (h / n as f64 - exact).powi(2);
            random_err += (r / n as f64 - exact).powi(2);
        }
        let halton_rmse = (halton_err / pixels as f64).sqrt();
        let random_rmse = (random_err / pixels as f64).sqrt();
        eprintln!(
            "edge coverage RMSE at {n} spp: halton {halton_rmse:.5}, random {random_rmse:.5}"
        );
        assert!(
            halton_rmse < random_rmse,
            "the low-discrepancy set is no better than random: {halton_rmse} vs {random_rmse}"
        );
    }

    /// Below the floor, adaptive sampling must be a no-op — not "almost" a
    /// no-op. A low-spp render is exactly where an early stop would do the
    /// most damage, so the option must not touch it at all.
    #[test]
    fn adaptive_is_inert_below_the_sample_floor() {
        let scene = test_scene();
        let cam = test_camera();
        let base = PathTraceOptions {
            spp: ADAPTIVE_FLOOR,
            max_depth: 3,
            denoise: false,
            seed: 11,
            ..Default::default()
        };
        let fixed = render(
            &scene,
            &cam,
            16,
            16,
            &PathTraceOptions {
                adaptive: false,
                ..base
            },
        );
        let adaptive = render(
            &scene,
            &cam,
            16,
            16,
            &PathTraceOptions {
                adaptive: true,
                ..base
            },
        );
        assert_eq!(
            fixed.rgb, adaptive.rgb,
            "adaptive sampling fired at or below the floor"
        );
    }

    /// Total samples spent, tracing every pixel the way [`render`] does.
    fn spend(scene: &Scene<TriMesh>, cam: &Camera, w: u32, h: u32, opts: &PathTraceOptions) -> u64 {
        let accel = SceneAccel::build(scene);
        let mut total = 0u64;
        for py in 0..h as usize {
            let mut rgb = vec![0.0f32; w as usize * 3];
            let mut alpha = vec![0.0f32; w as usize];
            let mut normal = vec![0.0f32; w as usize * 3];
            let mut depth = vec![0.0f32; w as usize];
            let mut albedo = vec![0.0f32; w as usize * 3];
            let mut variance = vec![0.0f32; w as usize];
            let mut out = PixelOut {
                rgb: &mut rgb,
                alpha: &mut alpha,
                normal: &mut normal,
                depth: &mut depth,
                albedo: &mut albedo,
                variance: &mut variance,
            };
            for px in 0..w as usize {
                total += trace_pixel(scene, &accel, None, cam, opts, w, h, px, py, &mut out) as u64;
            }
        }
        total
    }

    /// A converged pixel keeps the unbiased mean of the samples it took, so
    /// the adaptive film must land on the fixed-count film's estimate — it
    /// just gets there for less.
    #[test]
    fn adaptive_matches_the_fixed_count_mean_for_fewer_samples() {
        // A flat, softly lit scene: nearly every pixel resolves early, which
        // is exactly the case adaptivity exists for.
        let scene = Scene::<TriMesh> {
            objects: Vec::new(),
            lights: studio_rig(Point3::new(5.0, 5.0, 5.0), 9.0),
            env: Environment::default(),
            sun: None,
            ground: None,
            splats: None,
        };
        let cam = test_camera();
        let (w, h) = (24u32, 24u32);
        let base = PathTraceOptions {
            spp: 256,
            max_depth: 3,
            denoise: false,
            seed: 5,
            ..Default::default()
        };
        let fixed_opts = PathTraceOptions {
            adaptive: false,
            ..base
        };
        let adaptive_opts = PathTraceOptions {
            adaptive: true,
            ..base
        };

        let fixed = render(&scene, &cam, w, h, &fixed_opts);
        let adaptive = render(&scene, &cam, w, h, &adaptive_opts);

        let mean = |f: &Film| {
            f.rgb
                .chunks_exact(3)
                .map(|p| luminance([p[0], p[1], p[2]]) as f64)
                .sum::<f64>()
                / (f.rgb.len() / 3) as f64
        };
        let (mf, ma) = (mean(&fixed), mean(&adaptive));
        let n_fixed = spend(&scene, &cam, w, h, &fixed_opts);
        let n_adaptive = spend(&scene, &cam, w, h, &adaptive_opts);
        eprintln!(
            "mean luminance: fixed {mf:.6} ({n_fixed} samples), adaptive {ma:.6} ({n_adaptive} samples)"
        );
        assert!(
            (ma - mf).abs() <= 0.02 * mf.abs().max(1e-3),
            "adaptive biased the estimate: {mf} -> {ma}"
        );
        assert!(
            n_adaptive < n_fixed / 2,
            "adaptive spent {n_adaptive} of {n_fixed} samples — no real saving"
        );
    }
}
