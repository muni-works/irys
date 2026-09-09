//! The surface model: `Pbr` and every BSDF lobe that evaluates it.

use super::*;

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
    /// Henyey–Greenstein asymmetry `g` of the medium's phase function, in
    /// `-1..1` — OpenPBR's `subsurface_scatter_anisotropy`.
    ///
    /// `0` — the default — is the isotropic scattering the walk has always
    /// done, and reduces to it exactly. Positive `g` is forward scattering:
    /// each event deflects the ray only a little, so light drives *deeper*
    /// before it turns around and a thin part reads as more translucent than
    /// its mean free path alone would say. Skin runs about `0.8`, marble and
    /// most minerals near `0`, and a few pigmented media are mildly backward.
    ///
    /// The albedo inversion behind [`Self::subsurface_color`] was fitted for
    /// an isotropic medium, so it is applied under **similarity theory**:
    /// [`Self::subsurface_radius`] is read as the *reduced* (transport) mean
    /// free path and the fit's answer as the reduced single-scattering
    /// albedo, and the true `(sigma_s, sigma_t)` are derived back out through
    /// `sigma_s' = sigma_s (1 - g)`. That is what keeps the colour the artist
    /// wrote when the anisotropy knob turns — see
    /// `anisotropy_keeps_the_surface_colour`.
    ///
    /// The GPU tier ignores this, along with every other subsurface field;
    /// its walk is a separate and much shorter one.
    pub subsurface_anisotropy: f32,
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
            subsurface_anisotropy: 0.0,
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
    pub(crate) fn alpha(&self) -> f32 {
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
    pub(crate) fn alpha_tb(&self) -> (f32, f32) {
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
    pub(crate) fn coat_alpha(&self) -> f32 {
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
    pub(crate) fn f0_dielectric(&self) -> f32 {
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
    pub(crate) fn tint(&self) -> [f32; 3] {
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
    pub(crate) fn f0(&self) -> [f32; 3] {
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
    pub(crate) fn diffuse_albedo(&self) -> [f32; 3] {
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
    pub(crate) fn denoise_albedo(&self) -> [f32; 3] {
        mix3(self.diffuse_albedo(), self.f0(), self.metallic)
    }
}

// ─── small math helpers ───────────────────────────────────────────────────

#[inline]
pub(crate) fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
pub(crate) fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
    ]
}

#[inline]
pub(crate) fn scale3(a: [f32; 3], k: f32) -> [f32; 3] {
    [a[0] * k, a[1] * k, a[2] * k]
}

#[inline]
pub(crate) fn mul3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] * b[0], a[1] * b[1], a[2] * b[2]]
}

#[inline]
pub(crate) fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
pub(crate) fn max3(a: [f32; 3]) -> f32 {
    a[0].max(a[1]).max(a[2])
}

#[inline]
pub(crate) fn smoothstep(t: f32) -> f32 {
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
pub(crate) struct Frame {
    pub(crate) t: Vec3,
    pub(crate) b: Vec3,
    pub(crate) n: Vec3,
}

/// Shading tangent frame around a unit normal.
///
/// When the hit carried a surface tangent `dP/du`, it is Gram-Schmidt
/// orthogonalised against the (possibly face-forwarded) shading normal and
/// used as the frame's x axis, so the anisotropic lobe lines up with the
/// surface's own parameterisation. Otherwise this is the arbitrary [`onb`]
/// basis the isotropic path has always used — which is exactly what an
/// isotropic material wants, since its BSDF is invariant to the choice.
pub(crate) fn shading_frame(n: Vec3, dpdu: Option<Vec3>) -> Frame {
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
pub(crate) fn to_local(t: Vec3, b: Vec3, n: Vec3, w: Vec3) -> Vec3 {
    Vec3::new(w.dot(t), w.dot(b), w.dot(n))
}

#[inline]
pub(crate) fn to_world(t: Vec3, b: Vec3, n: Vec3, w: Vec3) -> Vec3 {
    t * w.x + b * w.y + n * w.z
}

/// Cosine-weighted hemisphere sample in local space (+Z up).
pub(crate) fn cosine_hemisphere(r1: f64, r2: f64) -> Vec3 {
    let r = r1.sqrt();
    let phi = 2.0 * std::f64::consts::PI * r2;
    Vec3::new(r * phi.cos(), r * phi.sin(), (1.0 - r1).max(0.0).sqrt())
}

/// Uniform sample on the unit disc (concentric mapping).
pub(crate) fn concentric_disc(r1: f64, r2: f64) -> (f64, f64) {
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
pub(crate) fn d_ggx(wh: Vec3, alpha: f32) -> f32 {
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
pub(crate) fn d_ggx_aniso(wh: Vec3, at: f32, ab: f32) -> f32 {
    if at == ab {
        return d_ggx(wh, at);
    }
    let (hx, hy, hz) = (wh.x as f32, wh.y as f32, wh.z as f32);
    let d = (hx / at) * (hx / at) + (hy / ab) * (hy / ab) + hz * hz;
    1.0 / (std::f32::consts::PI * at * ab * d * d).max(1e-9)
}

/// Smith height-correlated visibility term (already divided by 4·NoL·NoV).
#[inline]
pub(crate) fn v_smith(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let gv = n_dot_l * (n_dot_v * n_dot_v * (1.0 - a2) + a2).sqrt();
    let gl = n_dot_v * (n_dot_l * n_dot_l * (1.0 - a2) + a2).sqrt();
    0.5 / (gv + gl).max(1e-9)
}

/// Anisotropic Smith height-correlated visibility term (already divided by
/// 4·NoL·NoV). Reduces exactly to [`v_smith`] when `at == ab`.
#[inline]
pub(crate) fn v_smith_aniso(wo: Vec3, wi: Vec3, at: f32, ab: f32) -> f32 {
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
pub(crate) fn g1_smith_aniso(w: Vec3, at: f32, ab: f32) -> f32 {
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
pub(crate) fn fresnel(f0: [f32; 3], cos_theta: f32) -> [f32; 3] {
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
pub(crate) fn spec_fresnel(m: &Pbr, f0: [f32; 3], cos_theta: f32, lambda_nm: f32) -> [f32; 3] {
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
pub(crate) const FON_C1: f32 = 0.5 - 2.0 / (3.0 * std::f32::consts::PI);
/// `2/3 - 28/(15pi)`, the constant in the FON *average* albedo.
pub(crate) const FON_C2: f32 = 2.0 / 3.0 - 28.0 / (15.0 * std::f32::consts::PI);

/// FON's normalisation factor `A_F`.
#[inline]
pub(crate) fn fon_a(r: f32) -> f32 {
    1.0 / (1.0 + FON_C1 * r)
}

/// Directional albedo of the FON lobe, `E_F(mu, r)` — the paper's exact form
/// rather than its quartic fit, since we are not on a shader clock here and
/// the exact one is only an `acos` more expensive.
#[inline]
pub(crate) fn e_fon(mu: f32, r: f32) -> f32 {
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
pub(crate) fn eon_diffuse(rho: [f32; 3], r: f32, wo: Vec3, wi: Vec3) -> [f32; 3] {
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
pub(crate) fn scatter_albedo(a: f32) -> f32 {
    let a = a.clamp(0.0, 1.0);
    1.0 - (-5.094_06 * a + 2.611_88 * a * a - 4.318_05 * a * a * a).exp()
}

/// Draw a scattered direction from the Henyey–Greenstein phase function
/// about `w`, the direction the ray was already travelling.
///
/// HG is one lobe with one parameter and a closed-form inverse CDF, which is
/// why it has outlived every more faithful phase function in production
/// renderers: the draw is exact, so the estimator's weight for the event is
/// `p / pdf = 1` and the phase function adds no variance at all.
///
/// `g` is the mean cosine of the deflection. `g = 0` collapses the expression
/// to `cos = 1 - 2u`, uniform on the sphere; the branch is taken on a
/// tolerance rather than on equality because the closed form divides by `g`.
#[inline]
pub(crate) fn henyey_greenstein(w: Vec3, g: f64, u1: f64, u2: f64) -> Vec3 {
    let cos_theta = if g.abs() < 1e-3 {
        1.0 - 2.0 * u1
    } else {
        let s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1);
        ((1.0 + g * g - s * s) / (2.0 * g)).clamp(-1.0, 1.0)
    };
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = std::f64::consts::TAU * u2;
    let (t_ax, b_ax) = onb(w);
    (t_ax * (sin_theta * phi.cos()) + b_ax * (sin_theta * phi.sin()) + w * cos_theta).normalize()
}

/// Where a subsurface walk came back out, and what it carries.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Exit {
    /// The point on the surface the path leaves from — generally not the one
    /// it entered at, which is the entire point.
    pub(crate) point: Point3,
    /// The outward normal there.
    pub(crate) normal: Vec3,
    /// Throughput accumulated over the walk, per channel.
    pub(crate) weight: [f32; 3],
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
pub(crate) const SUBSURFACE_MAX_STEPS: u32 = 1024;

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
pub(crate) fn subsurface_walk(
    m: &Pbr,
    entry: Point3,
    n: Vec3,
    rng: &mut Rng,
    mut trace: impl FnMut(Point3, Vec3) -> Option<(f64, Vec3)>,
) -> Option<Exit> {
    // The medium the surface colour and the mean free path imply.
    //
    // The inversion is fitted for isotropic scattering, so what it returns is
    // read as the *reduced* albedo of the *reduced* medium whose transport
    // mean free path is `subsurface_radius`. Similarity theory then gives the
    // real one back: absorption is invariant under the reduction, and
    // `sigma_s' = sigma_s (1 - g)` undoes to `sigma_s = sigma_s' / (1 - g)`.
    // At `g = 0` every line below is the identity and the medium is bit for
    // bit the one the walk used before the knob existed.
    let g = (m.subsurface_anisotropy as f64).clamp(-0.95, 0.95);
    let mut sigma_t = [0.0f64; 3];
    let mut sigma_s = [0.0f64; 3];
    for c in 0..3 {
        let r = m.subsurface_radius[c];
        if !(r > 0.0) || !r.is_finite() {
            return None;
        }
        let sigma_t_reduced = 1.0 / r;
        let albedo_reduced = scatter_albedo(m.subsurface_color[c]) as f64;
        // sigma_a is the same medium either way.
        let sigma_a = (1.0 - albedo_reduced) * sigma_t_reduced;
        sigma_s[c] = albedo_reduced * sigma_t_reduced / (1.0 - g);
        sigma_t[c] = sigma_a + sigma_s[c];
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
        // Henyey–Greenstein, sampled exactly, so `f / pdf` is 1 and the phase
        // function costs the walk nothing but a direction. At `g = 0` the
        // cosine drawn is `1 - 2u`, uniform on the sphere — the same
        // *distribution* the isotropic draw this replaced had, from the same
        // two numbers in the same order. It is not the same direction for a
        // given pair, because it is now built around the incoming ray rather
        // than around the world axes, so a `g = 0` walk matches the old one
        // statistically and not sample for sample.
        dir = henyey_greenstein(dir, g, rng.f64(), rng.f64());

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
pub(crate) fn ms_compensation(f0: [f32; 3], at: f32, ab: f32, mu_o: f32) -> [f32; 3] {
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
pub(crate) fn sheen_coeffs(m: &Pbr, mu_o: f32) -> [f32; 3] {
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
pub(crate) fn sheen_ltc_density(wi_std: Vec3, coeffs: [f32; 3]) -> f32 {
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
pub(crate) fn sheen_align(wo: Vec3, w: Vec3) -> Vec3 {
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
pub(crate) fn sheen_unalign(wo: Vec3, w: Vec3) -> Vec3 {
    let len = (wo.x * wo.x + wo.y * wo.y).sqrt();
    if len <= 0.0 {
        return w;
    }
    let (c, s) = (wo.x / len, wo.y / len);
    Vec3::new(c * w.x - s * w.y, s * w.x + c * w.y, w.z)
}

/// The sheen lobe's `f · cos`, and its PDF.
pub(crate) fn sheen_eval(m: &Pbr, wo: Vec3, wi: Vec3) -> ([f32; 3], f32) {
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
pub(crate) fn sheen_albedo(m: &Pbr, mu_o: f32) -> f32 {
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
pub(crate) fn d_gtr1(wh: Vec3, alpha: f32) -> f32 {
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
pub(crate) fn sample_gtr1(alpha: f32, r1: f64, r2: f64) -> Vec3 {
    let a2 = (alpha * alpha).clamp(1e-8, 0.999_999) as f64;
    let cos2 = ((1.0 - a2.powf(1.0 - r1)) / (1.0 - a2)).clamp(0.0, 1.0);
    let cos_t = cos2.sqrt();
    let sin_t = (1.0 - cos2).max(0.0).sqrt();
    let phi = std::f64::consts::TAU * r2;
    Vec3::new(sin_t * phi.cos(), sin_t * phi.sin(), cos_t)
}

/// PDF of [`sample_gtr1`] in solid angle around `wi`.
#[inline]
pub(crate) fn gtr1_pdf(wo: Vec3, wh: Vec3, alpha: f32) -> f32 {
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
pub(crate) fn sample_vndf(wo: Vec3, at: f32, ab: f32, r1: f64, r2: f64) -> Vec3 {
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
pub(crate) fn vndf_pdf(wo: Vec3, wh: Vec3, at: f32, ab: f32) -> f32 {
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
pub(crate) fn g1_smith_abs(w: Vec3, at: f32, ab: f32) -> f32 {
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
pub(crate) fn fresnel_dielectric(cos_i: f32, eta: f32) -> f32 {
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
pub(crate) fn vndf_density(wo: Vec3, wh: Vec3, at: f32, ab: f32) -> f32 {
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
pub(crate) fn dielectric_eval(m: &Pbr, wo: Vec3, wi: Vec3, eta: f32) -> (f32, f32) {
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
pub(crate) fn lobe_weights(m: &Pbr) -> [f32; 6] {
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
pub(crate) fn bsdf_eval(m: &Pbr, wo: Vec3, wi: Vec3, eta: f32, lambda_nm: f32) -> ([f32; 3], f32) {
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
pub(crate) enum Sampled {
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
pub(crate) fn bsdf_sample_surface(
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
pub(crate) fn bsdf_sample(
    m: &Pbr,
    wo: Vec3,
    eta: f32,
    lambda_nm: f32,
    rng: &mut Rng,
) -> Option<Sampled> {
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
pub(crate) fn reflect(i: Vec3, n: Vec3) -> Vec3 {
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
pub(crate) fn sheet_transmittance(m: &Pbr, cos_dot: f64) -> Option<[f32; 3]> {
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
pub(crate) fn power_heuristic(a: f32, b: f32) -> f32 {
    let a2 = a * a;
    let b2 = b * b;
    if a2 + b2 <= 0.0 { 0.0 } else { a2 / (a2 + b2) }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::cpu::testing::*;
    #[allow(unused_imports)]
    use crate::geometry::TriMesh;

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

    /// A slab many mean free paths thick has to stop the light, or the walk
    /// is not attenuating at all. The counterpart to `a_thin_slab_transmits`:
    /// between them they pin that the transmission is a function of the ratio
    /// of thickness to mean free path and not a constant — 0.4 mfp passes
    /// about 28% and 40 mfp passes 0.33%, a factor of ~85 for a factor of 100
    /// in thickness.
    ///
    /// It is not zero, and it should not be: at `subsurface_color = 0.9` the
    /// medium's single-scattering albedo is 0.9964, absorption is nearly nil
    /// and the light gets across by diffusion rather than by any straight
    /// path. That last third of a percent is the physics, so it is stated
    /// here rather than tightened away.
    #[test]
    fn a_thick_slab_does_not_transmit() {
        let m = Pbr {
            subsurface: 1.0,
            subsurface_color: [0.9; 3],
            subsurface_radius: [0.05; 3],
            ..Default::default()
        };
        let n = Vec3::new(0.0, 0.0, 1.0);
        let mut rng = Rng::new(0x5b55_0006);
        let trials = 20_000;
        // 40 mean free paths, against the thin slab's 0.4.
        let (mut through, mut back) = (0.0f64, 0.0f64);
        for _ in 0..trials {
            let Some(e) = subsurface_walk(
                &m,
                Point3::new(0.0, 0.0, 0.0),
                n,
                &mut rng,
                slab_trace(2.0),
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
        assert!(through < 0.01, "a 40-mfp slab leaked {through}");
        assert!(back > 0.8, "and it must reflect nearly all of it: {back}");
    }

    /// The anisotropy knob changes how the light gets around inside; it must
    /// not change what colour comes back out.
    ///
    /// That is the whole job of the similarity-theory correction in
    /// [`subsurface_walk`]: without it, a forward-scattering medium built
    /// from the isotropic fit's numbers travels further per event and comes
    /// back visibly brighter than the colour that was asked for.
    #[test]
    fn anisotropy_keeps_the_surface_colour() {
        let want = [0.65f32, 0.5, 0.4];
        for g in [-0.5f32, 0.0, 0.4, 0.8] {
            let m = Pbr {
                subsurface: 1.0,
                subsurface_color: want,
                subsurface_radius: [0.01; 3],
                subsurface_anisotropy: g,
                ..Default::default()
            };
            let mut rng = Rng::new(0x5b55_0007);
            let trials = 60_000;
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
                let got = acc[c] / trials as f64;
                let target = want[c] as f64;
                assert!(
                    (got - target).abs() <= 0.05 * target,
                    "g={g}, channel {c}: walked {got}, asked for {target}"
                );
            }
        }
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
}
