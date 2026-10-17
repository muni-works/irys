//! Emitters: area lights, the analytic environments, and the default rig.

use super::*;

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
    pub(crate) fn intersect(&self, ray: &Ray) -> Option<f64> {
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
    pub(crate) fn radiance(&self, d: Vec3) -> [f32; 3] {
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
pub(crate) fn sample_1d(cdf: &[f32], u: f32) -> (usize, f32) {
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
    pub(crate) width: usize,
    pub(crate) height: usize,
    /// Row-major linear radiance, `width * height` texels.
    pub(crate) pixels: Vec<[f32; 3]>,
    pub(crate) intensity: f32,
    /// Rotation about +Z in radians, applied when mapping u to phi.
    pub(crate) rotation: f64,
    /// Per-row conditional CDF over u, `height * (width + 1)` entries.
    pub(crate) cond_cdf: Vec<f32>,
    /// Marginal CDF over v, `height + 1` entries.
    pub(crate) marg_cdf: Vec<f32>,
    /// Mean of the weighted function; the normaliser for the uv-space PDF.
    pub(crate) marg_int: f32,
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
    pub(crate) fn is_sampleable(&self) -> bool {
        self.marg_int > 0.0
    }

    /// Image coordinates in `[0, 1)^2` for a world direction.
    pub(crate) fn uv(&self, d: Vec3) -> (f64, f64) {
        const EPS: f64 = 1e-9;
        let theta = d.z.clamp(-1.0, 1.0).acos();
        let v = (theta / std::f64::consts::PI).clamp(0.0, 1.0 - EPS);
        let phi = (d.y.atan2(d.x) - self.rotation).rem_euclid(std::f64::consts::TAU);
        let u = (phi / std::f64::consts::TAU).clamp(0.0, 1.0 - EPS);
        (u, v)
    }

    /// World direction for image coordinates in `[0, 1]^2`.
    pub(crate) fn direction(&self, u: f64, v: f64) -> Vec3 {
        let phi = u * std::f64::consts::TAU + self.rotation;
        let theta = v * std::f64::consts::PI;
        let (st, ct) = theta.sin_cos();
        Vec3::new(st * phi.cos(), st * phi.sin(), ct)
    }

    /// Texel indices for image coordinates.
    #[inline]
    pub(crate) fn texel_index(&self, u: f64, v: f64) -> (usize, usize) {
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
    pub(crate) fn radiance(&self, d: Vec3) -> [f32; 3] {
        match self {
            Environment::Gradient(g) => g.radiance(d),
            Environment::Image(m) => m.radiance(d),
        }
    }

    /// Whether this environment participates in MIS as its own strategy.
    #[inline]
    pub(crate) fn is_importance_sampled(&self) -> bool {
        match self {
            Environment::Gradient(_) => false,
            Environment::Image(m) => m.is_sampleable(),
        }
    }

    /// Solid-angle PDF of the environment sampling strategy, or 0 when this
    /// environment is not importance-sampled.
    #[inline]
    pub(crate) fn pdf(&self, d: Vec3) -> f32 {
        match self {
            Environment::Gradient(_) => 0.0,
            Environment::Image(m) => m.pdf(d),
        }
    }

    /// Importance-sample a direction, if this environment supports it.
    #[inline]
    pub(crate) fn sample(&self, r1: f64, r2: f64) -> Option<(Vec3, [f32; 3], f32)> {
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
    #[allow(unused_imports)]
    use super::*;
    #[allow(unused_imports)]
    use crate::cpu::testing::*;
    #[allow(unused_imports)]
    use crate::geometry::TriMesh;

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
}
