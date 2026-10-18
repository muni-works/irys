//! The skin integrator: a spectral path tracer over a textured head, with a
//! random-walk subsurface medium under a dual-lobe specular layer, area
//! lights with NEE + MIS, a thin-lens camera, and path replay
//! backpropagation into [`PARAMS`](super::graph::PARAMS).
//!
//! # A path
//!
//! Camera ray (thin lens) → closest hit among the mesh and the lights. At a
//! skin vertex (outside the medium):
//!
//! 1. **NEE**: one light sample (pick by power × the two cosines / d²,
//!    uniform on the quad); the specular layer's `f` (the
//!    [`Spec`](super::graph::Spec) graph) times `cos · Le / p_L`, weighted
//!    against the reflection pdf by the power heuristic. With [`Mode::sss`]
//!    off, the Lambertian base adds `ρ (1 − F(cos_o))/π`.
//! 2. **Reflect or enter**: with `P_r = clamp(4 F(cos_o) k, 0.1, 0.6)`
//!    (detached), reflect off one GGX lobe (VNDF at the lobe picked by its
//!    mix weight; the pdf is the mixture's); otherwise go through, weight
//!    `(1 − F(cos_o))/(1 − P_r)`. Through means, in SSS mode, a cosine
//!    direction into the medium; without SSS, a cosine bounce off the
//!    Lambertian base of albedo `ρ`.
//! 3. **The walks** (SSS): coefficients from the entry texel's chromophores,
//!    one analog walk ([`walk_g`]) per wavelength: distances drawn at the
//!    sampler's `σt`, `f = σs e^{−σt t}` per scatter and `e^{−σt d}` at the
//!    boundary — the [`Medium`](super::graph::Medium) graph's events —
//!    Henyey–Greenstein turns, no Fresnel at the inner boundary (Cycles'
//!    random walk does the same). (Four wavelengths on one walk with
//!    spectral MIS compounds the per-event weight variance over hundreds of
//!    events; it was tried, and was chroma noise.)
//! 4. **Exits**: each walk's exit is a Lambertian vertex (`1/π`, θ-free,
//!    about the smooth normal) with NEE for its own wavelength; the path
//!    goes on from one exit, picked uniformly, that wavelength alone
//!    (weight 4), with a cosine direction out.
//!
//! # What the replay owes the parameters
//!
//! Every factor of `L` is a [`Spec`](super::graph::Spec) `f` (a reflected
//! direction or a light sample) or a [`Medium`](super::graph::Medium) event;
//! the rest are θ-free numbers (pdfs, `P_r`, `1 − F`, `q`, albedos, the exit
//! lobe). As in [`crate::prb`], `∂L/∂θ = Σ_j ∂θ log f_j · (L − L_j⁺)`, plus
//! `∂θ log f_L · C_j` for light samples. Inside a walk nothing is collected,
//! so every event of one walk shares `W = δL (L − L_entry⁺)` and the walk's
//! events reduce to one [`Medium`](super::graph::Medium) lane with
//! `s = W · #scatters`, `t = W · Σ t` ([`super::graph`]).
//!
//! # Determinism
//!
//! One SplitMix64 stream per path ([`crate::prb::Rng`]), consumed in a fixed
//! order; the replay re-walks it. Paths run in parallel and are collected in
//! order, adjoint records are reduced in path order, chunk by chunk: the
//! gradient is bit-identical run to run.

use rayon::prelude::*;

use irys_core::{Bvh, Point3, Ray, TriMesh, Vec3};
use irys_light::spectrum::{hero_weight, sample_lambda_nm};

use super::accel::Accel;
use super::graph::{self, MEDIUM_COLUMNS, Medium, PARAMS, SPEC_COLUMNS, Spec, spec_f64, spec_pdf};
use super::mesh::{self, Mesh, Texture};
use super::optics::{self, G};
use crate::ggx::{band_weights, normalize};
use crate::prb::{Rng, mis, offset};

/// Wavelengths per path.
pub const NW: usize = 4;

const PI: f64 = std::f64::consts::PI;

/// A one-sided rectangular emitter: `center ± u ± v`, emitting on the
/// `u × v` side.
#[derive(Debug, Clone, Copy)]
pub struct Light {
    /// Centre.
    pub center: Point3,
    /// Half-edge.
    pub u: Vec3,
    /// Half-edge.
    pub v: Vec3,
    /// Radiance scale.
    pub radiance: f64,
    /// Linear-sRGB tint, lifted onto the three hats.
    pub tint: [f64; 3],
}

impl Light {
    /// A quad of `w × h` at `center` facing `toward`, its `u` edge kept
    /// horizontal (`⊥ z`).
    pub fn facing(
        center: Point3,
        toward: Point3,
        w: f64,
        h: f64,
        radiance: f64,
        tint: [f64; 3],
    ) -> Self {
        let n = (toward - center).normalize();
        let mut u = n.cross(&Vec3::new(0.0, 0.0, 1.0));
        if u.norm() < 1e-6 {
            u = Vec3::new(1.0, 0.0, 0.0);
        }
        let u = u.normalize();
        let v = u.cross(&n).normalize();
        // u × v must be n.
        let (u, v) = if u.cross(&v).dot(&n) > 0.0 {
            (u, v)
        } else {
            (v, u)
        };
        Self {
            center,
            u: u * (0.5 * w),
            v: v * (0.5 * h),
            radiance,
            tint,
        }
    }

    /// Unit normal (the emitting side).
    pub fn normal(&self) -> Vec3 {
        self.u.cross(&self.v).normalize()
    }

    /// Area.
    pub fn area(&self) -> f64 {
        4.0 * self.u.cross(&self.v).norm()
    }

    /// Radiance at `lambda`.
    pub fn le(&self, lambda: f64) -> f64 {
        self.radiance * lift(self.tint, lambda)
    }

    /// Ray parameter of a hit on the emitting side, if any, in `(0, tmax)`.
    pub fn hit(&self, o: Point3, d: Vec3, tmax: f64) -> Option<f64> {
        let n = self.normal();
        let den = d.dot(&n);
        if den >= 0.0 {
            return None;
        }
        let t = (self.center - o).dot(&n) / den;
        if !(t > 0.0 && t < tmax) {
            return None;
        }
        let p = o + d * t - self.center;
        let (a, b) = (
            p.dot(&self.u) / self.u.norm_squared(),
            p.dot(&self.v) / self.v.norm_squared(),
        );
        (a.abs() <= 1.0 && b.abs() <= 1.0).then_some(t)
    }
}

/// Linear-sRGB `rgb` at `lambda`, by the conductor's three hats (450, 550,
/// 650 nm: blue, green, red).
pub fn lift(rgb: [f64; 3], lambda: f64) -> f64 {
    let w = band_weights(lambda);
    w[0] * rgb[2] + w[1] * rgb[1] + w[2] * rgb[0]
}

/// Which model a frame renders.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mode {
    /// Random-walk subsurface under the specular layer; off, a Lambertian
    /// base of the albedo map.
    pub sss: bool,
    /// Two specular lobes; off, one (at `θ₃`).
    pub dual: bool,
}

/// Where the chromophores come from.
#[derive(Debug, Clone)]
pub enum Chromophores {
    /// Per texel `[m, h, err]` ([`optics::invert`] of the albedo map),
    /// scaled by `θ₀`, `θ₁`.
    Map(Vec<[f32; 3]>),
    /// Uniform: the fractions are `θ₀`, `θ₁` themselves.
    Uniform,
}

/// The camera: a thin lens.
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    /// Lens centre.
    pub eye: Point3,
    /// Unit view direction.
    pub forward: Vec3,
    /// Unit right.
    pub right: Vec3,
    /// Unit up.
    pub up: Vec3,
    /// `tan` of half the vertical field of view.
    pub tan_half: f64,
    /// Lens radius (scene units); 0 is a pinhole.
    pub aperture: f64,
    /// Distance to the plane in focus, along `forward`.
    pub focus: f64,
}

impl Camera {
    /// At `eye` looking at `target`, `+z` up, focused on `target`.
    pub fn look_at(eye: Point3, target: Point3, vfov_deg: f64, aperture: f64) -> Self {
        let forward = (target - eye).normalize();
        let right = forward.cross(&Vec3::new(0.0, 0.0, 1.0)).normalize();
        let up = right.cross(&forward);
        Self {
            eye,
            forward,
            right,
            up,
            tan_half: (0.5 * vfov_deg).to_radians().tan(),
            aperture,
            focus: (target - eye).norm(),
        }
    }
}

/// Everything a skin frame needs.
pub struct Scene {
    /// The mesh (scene units, `+z` up).
    pub mesh: Mesh,
    pub(crate) bvh: Bvh<TriMesh>,
    pub(crate) accel: Accel,
    /// Base colour (linear).
    pub albedo: Texture,
    /// Specular intensity.
    pub spec: Texture,
    /// Tangent-space normal map.
    pub normal: Texture,
    /// Chromophores.
    pub chromo: Chromophores,
    /// Millimetres per scene unit: the medium's coefficients are per mm.
    pub mm_per_unit: f64,
    /// Emitters.
    pub lights: Vec<Light>,
    /// Environment radiance: `env · (0.55 + 0.45 d_z)`, grey.
    pub env: f64,
    /// The lens.
    pub camera: Camera,
    /// Pixels.
    pub size: (usize, usize),
    /// Samples per pixel.
    pub spp: usize,
    /// Surface vertices per path.
    pub depth: usize,
    /// The model.
    pub mode: Mode,
    /// Normal-map strength (0 off).
    pub bump: f64,
    /// The specular map's multiplier: `k = gain · map`.
    pub spec_gain: f64,
    /// Most events in one walk; a walk that runs out is absorbed.
    pub max_events: usize,
    /// Sample at these parameters instead of the current ones: paths then
    /// do not move with `θ`, so central differences with common random
    /// numbers see the estimator PRB differentiates.
    pub frozen: Option<[f64; PARAMS]>,
    /// Pixel window `(x0, y0)` of a crop: the frame renders `size` pixels
    /// starting there, of a full image `full`.
    pub crop: (usize, usize),
    /// The full image the crop is of (for the camera).
    pub full: (usize, usize),
}

/// Most lights a scene holds (the GPU's fixed array).
pub const MAX_LIGHTS: usize = 4;

impl Scene {
    /// A scene over `mesh` and its maps; lights, camera and the rest are
    /// set by the caller (the fields are public) and [`Scene::lights_changed`].
    pub fn new(
        mesh: Mesh,
        albedo: Texture,
        spec: Texture,
        normal: Texture,
        chromo: Chromophores,
        mm_per_unit: f64,
    ) -> Self {
        let bvh = Bvh::build(mesh.trimesh());
        let accel = Accel::new(&bvh);
        Self {
            mesh,
            bvh,
            accel,
            albedo,
            spec,
            normal,
            chromo,
            mm_per_unit,
            lights: Vec::new(),
            env: 0.0,
            camera: Camera::look_at(Point3::new(0.0, -10.0, 0.0), Point3::origin(), 30.0, 0.0),
            size: (64, 64),
            spp: 4,
            depth: 4,
            mode: Mode {
                sss: true,
                dual: true,
            },
            bump: 1.0,
            spec_gain: 1.0,
            max_events: 4096,
            frozen: None,
            crop: (0, 0),
            full: (64, 64),
        }
    }

    /// Check the lights (at most [`MAX_LIGHTS`]) after editing them.
    pub fn lights_changed(&mut self) {
        assert!(
            self.lights.len() <= MAX_LIGHTS,
            "at most {MAX_LIGHTS} lights"
        );
    }

    /// Set the image size (uncropped).
    pub fn set_size(&mut self, w: usize, h: usize) {
        self.size = (w, h);
        self.full = (w, h);
        self.crop = (0, 0);
    }

    /// The BVH over the mesh.
    pub fn bvh(&self) -> &Bvh<TriMesh> {
        &self.bvh
    }

    /// Paths in a frame.
    pub fn paths(&self) -> usize {
        self.size.0 * self.size.1 * self.spp
    }

    /// Lanes in a frame (path × wavelength).
    pub fn lanes(&self) -> usize {
        self.paths() * NW
    }

    /// The probabilities NEE picks each light with at receiver `x`,
    /// normal `n`: power × the two cosines (floored at 0.02, so no light
    /// that could contribute is ever impossible) / distance². Detached.
    pub fn light_probs(&self, x: Point3, n: Vec3) -> [f64; MAX_LIGHTS] {
        let mut p = [0.0; MAX_LIGHTS];
        let mut sum = 0.0;
        for (i, l) in self.lights.iter().enumerate() {
            let to = l.center - x;
            let d2 = to.norm_squared().max(1e-12);
            let d = to / d2.sqrt();
            let cr = n.dot(&d).max(0.02);
            let cl = (-d.dot(&l.normal())).max(0.02);
            p[i] = l.radiance * l.area() * cr * cl / d2;
            sum += p[i];
        }
        for v in &mut p {
            *v /= sum;
        }
        p
    }

    /// Environment radiance towards `d`.
    pub fn env_radiance(&self, d: Vec3) -> f64 {
        self.env * (0.55 + 0.45 * d.z)
    }

    /// Medium coefficients at texel `uv`: `(m, h)` before `θ` scaling.
    pub fn chromophores(&self, uv: [f64; 2]) -> (f64, f64) {
        match &self.chromo {
            Chromophores::Uniform => (1.0, 1.0),
            Chromophores::Map(map) => {
                // Nearest texel: chromophores are not linear in colour, and
                // the walk takes the entry point's.
                let (w, h) = self.albedo.size;
                let x = (((uv[0] - uv[0].floor()) * w as f64) as usize).min(w - 1);
                let y = (((uv[1] - uv[1].floor()) * h as f64) as usize).min(h - 1);
                let t = map[y * w + x];
                (t[0] as f64, t[1] as f64)
            }
        }
    }

    fn camera_ray(&self, path: usize, rng: &mut Rng) -> Ray {
        let px = path / self.spp;
        let (w, _) = self.size;
        let (fw, fh) = (self.full.0 as f64, self.full.1 as f64);
        let x = (self.crop.0 + px % w) as f64 + rng.uniform();
        let y = (self.crop.1 + px / w) as f64 + rng.uniform();
        let c = &self.camera;
        let sx = (2.0 * x / fw - 1.0) * c.tan_half * fw / fh;
        let sy = (1.0 - 2.0 * y / fh) * c.tan_half;
        let d = c.forward + c.right * sx + c.up * sy;
        let (u1, u2) = (rng.uniform(), rng.uniform());
        if c.aperture <= 0.0 {
            return Ray::new(c.eye, d);
        }
        let (dx, dy) = concentric(u1, u2);
        let lens = c.eye + c.right * (dx * c.aperture) + c.up * (dy * c.aperture);
        let focus = c.eye + d * c.focus;
        Ray::new(lens, focus - lens)
    }

    fn occluded(&self, o: Point3, d: Vec3, dist: f64) -> bool {
        self.bvh
            .occluded_range(&Ray::new(o, d), 0.0, dist * (1.0 - 1e-4))
    }

    /// Closest light hit before `tmax`: `(light, t)`.
    fn light_hit(&self, o: Point3, d: Vec3, tmax: f64) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64)> = None;
        for (i, l) in self.lights.iter().enumerate() {
            if let Some(t) = l.hit(o, d, best.map_or(tmax, |b| b.1)) {
                best = Some((i, t));
            }
        }
        best
    }
}

/// Shirley–Chiu concentric map of the unit square onto the unit disc.
pub fn concentric(u1: f64, u2: f64) -> (f64, f64) {
    let (a, b) = (2.0 * u1 - 1.0, 2.0 * u2 - 1.0);
    if a == 0.0 && b == 0.0 {
        return (0.0, 0.0);
    }
    let (r, phi) = if a.abs() > b.abs() {
        (a, 0.25 * PI * (b / a))
    } else {
        (b, 0.5 * PI - 0.25 * PI * (a / b))
    };
    (r * phi.cos(), r * phi.sin())
}

/// Henyey–Greenstein: a direction scattered from `w` at asymmetry `g`.
pub fn hg_sample(w: Vec3, g: f64, u1: f64, u2: f64) -> Vec3 {
    let cos = if g.abs() < 1e-3 {
        1.0 - 2.0 * u1
    } else {
        let s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1);
        ((1.0 + g * g - s * s) / (2.0 * g)).clamp(-1.0, 1.0)
    };
    let sin = (1.0 - cos * cos).max(0.0).sqrt();
    let phi = 2.0 * PI * u2;
    let (t, b) = mesh::basis(w);
    t * (sin * phi.cos()) + b * (sin * phi.sin()) + w * cos
}

/// A cosine-distributed direction about `n`.
fn cosine(n: Vec3, u1: f64, u2: f64) -> (Vec3, f64) {
    let (r, phi) = (u1.sqrt(), 2.0 * PI * u2);
    let cz = (1.0 - u1).max(0.0).sqrt();
    let (t, b) = mesh::basis(n);
    (t * (r * phi.cos()) + b * (r * phi.sin()) + n * cz, cz)
}

/// The reflect-or-enter probability (detached).
pub fn p_reflect(co: f64, k: f64) -> f64 {
    (4.0 * optics::schlick(co) * k).clamp(0.1, 0.6)
}

/// One lane of the [`Spec`] graph: its columns and the `δf` the replay
/// seeds it with.
#[derive(Debug, Clone, Copy)]
pub struct SpecRecord {
    /// `cos_i, cos_o, cos_h, o·h, k`.
    pub cols: [f64; SPEC_COLUMNS],
    /// `δf`.
    pub df: f64,
}

/// One lane of the [`Medium`] graph: a whole walk at one wavelength,
/// reduced (`s`, `t` already multiplied by `W`, `δℓ = 1`).
#[derive(Debug, Clone, Copy)]
pub struct MediumRecord {
    /// `s, t, m, h, Eₘ, Eₕ, E_b, S`.
    pub cols: [f64; MEDIUM_COLUMNS],
}

/// What a replayed path owes `θ`.
#[derive(Debug, Clone, Default)]
pub struct Records {
    /// Specular lanes.
    pub spec: Vec<SpecRecord>,
    /// Medium lanes.
    pub medium: Vec<MediumRecord>,
}

/// Counters for one pass.
#[derive(Debug, Clone, Copy, Default)]
pub struct Counts {
    /// Medium events (scatter or exit), all paths.
    pub events: u64,
    /// Walks.
    pub walks: u64,
    /// Walks that ran out of events.
    pub capped: u64,
}

/// The parameters the model evaluates at: in single-lobe mode the second
/// lobe is the first.
fn effective(mode: Mode, theta: &[f64; PARAMS]) -> [f64; PARAMS] {
    let mut t = *theta;
    if !mode.dual {
        t[4] = t[3];
    }
    t
}

/// The hit's shading, mapped: `(ng, n, uv, k)`.
fn surface(scene: &Scene, h: &irys_core::Hit, d: Vec3) -> (Vec3, Vec3, [f64; 2], f64) {
    let s = mesh::shade(&scene.mesh, &scene.bvh, h);
    let flip = if s.ng.dot(&d) > 0.0 { -1.0 } else { 1.0 };
    let ng = s.ng * flip;
    let mut s2 = s;
    s2.ns = s.ns * flip;
    let mut n = if scene.bump > 0.0 {
        mesh::bump(&s2, scene.normal.sample(s.uv[0], s.uv[1]), scene.bump)
    } else {
        s2.ns
    };
    // A shading normal facing away from the viewer is replaced by the
    // geometric one.
    if n.dot(&(-d)) <= 1e-4 {
        n = ng;
    }
    let k = scene.spec_gain * scene.spec.sample(s.uv[0], s.uv[1])[0];
    (ng, n, s.uv, k)
}

/// A light sample from `x` (normal `n`): `(light, ω, distance, solid-angle
/// pdf)`, or `None` when the point on the light faces away or `ω` is under
/// `n`. Consumes nothing: the caller passes its three uniforms.
fn sample_light(scene: &Scene, x: Point3, n: Vec3, u: [f64; 3]) -> Option<(usize, Vec3, f64, f64)> {
    if scene.lights.is_empty() {
        return None;
    }
    let p = scene.light_probs(x, n);
    let mut i = scene.lights.len() - 1;
    let mut acc = 0.0;
    for (k, pk) in p.iter().enumerate().take(scene.lights.len()) {
        acc += pk;
        if u[0] < acc {
            i = k;
            break;
        }
    }
    let l = &scene.lights[i];
    let y = l.center + l.u * (2.0 * u[1] - 1.0) + l.v * (2.0 * u[2] - 1.0);
    let to = y - x;
    let dist = to.norm();
    let wi = to / dist;
    let cl = -wi.dot(&l.normal());
    if cl <= 0.0 || wi.dot(&n) <= 0.0 {
        return None;
    }
    Some((i, wi, dist, p[i] * dist * dist / (cl * l.area())))
}

/// The NEE pdf of direction `d` from `x` (normal `n`) towards light `i` hit
/// at distance `t`.
fn light_pdf(scene: &Scene, x: Point3, n: Vec3, i: usize, d: Vec3, t: f64) -> f64 {
    let l = &scene.lights[i];
    let cos = -d.dot(&l.normal());
    if cos <= 0.0 {
        return 0.0;
    }
    scene.light_probs(x, n)[i] * t * t / (cos * l.area())
}

/// One path, forward or replayed: `L` per wavelength and, in replay, its
/// records. `replay = (L, δL)` of the forward pass, per wavelength.
pub fn trace_path(
    scene: &Scene,
    theta: &[f64; PARAMS],
    seed: u64,
    index: usize,
    replay: Option<([f64; NW], [f64; NW])>,
    counts: &mut Counts,
) -> ([f64; NW], Records) {
    let th = effective(scene.mode, theta);
    let smp = effective(scene.mode, scene.frozen.as_ref().unwrap_or(theta));
    let mut rec = Records::default();
    let mut rng = Rng::new(seed, index as u64);
    let u0 = rng.uniform();
    let lam: [f64; NW] = std::array::from_fn(|k| {
        let v = u0 + k as f64 / NW as f64;
        sample_lambda_nm(v - v.floor())
    });
    let mut ray = scene.camera_ray(index, &mut rng);
    let mut tp = [1.0f64; NW];
    let mut rad = [0.0f64; NW];
    // The vertex the current ray left (point, normal) and the pdf its own
    // sampling drew the ray with; pdf 0 for the camera ray.
    let mut prev = (ray.origin, Vec3::new(0.0, 0.0, 1.0), 0.0f64);
    let spec_cols = |li: [f64; 3], lo: [f64; 3], k: f64| {
        let hv = normalize([li[0] + lo[0], li[1] + lo[1], li[2] + lo[2]]);
        [li[2], lo[2], hv[2], dot3(lo, hv), k]
    };
    for _ in 0..scene.depth {
        let d: Vec3 = *ray.direction.as_ref();
        let hit = scene.bvh.trace_closest_range(&ray, 0.0, f64::INFINITY);
        let tmax = hit.as_ref().map_or(f64::INFINITY, |h| h.t);
        if let Some((li, t)) = scene.light_hit(ray.origin, d, tmax) {
            let w = if prev.2 > 0.0 {
                mis(prev.2, light_pdf(scene, prev.0, prev.1, li, d, t))
            } else {
                1.0
            };
            for k in 0..NW {
                rad[k] += tp[k] * scene.lights[li].le(lam[k]) * w;
            }
            break;
        }
        let Some(h) = hit else {
            for k in 0..NW {
                rad[k] += tp[k] * scene.env_radiance(d);
            }
            break;
        };
        let (ng, n, uv, kspec) = surface(scene, &h, d);
        let wo = -d;
        let origin = offset(h.point, ng, ray.origin, h.t);
        let (t, b) = mesh::basis(n);
        let local = |v: Vec3| [v.dot(&t), v.dot(&b), v.dot(&n)];
        let lo = local(wo);
        let co = lo[2].max(1e-6);
        let pr = p_reflect(co, kspec);
        let fr_o = optics::schlick(co);
        // Lambertian base (SSS off): ρ(1 − F(cos_o))/π per lane.
        let fd: [f64; NW] = if scene.mode.sss {
            [0.0; NW]
        } else {
            let a = scene.albedo.sample(uv[0], uv[1]);
            std::array::from_fn(|k| lift(a, lam[k]) * (1.0 - fr_o) / PI)
        };
        // This vertex's sampling density of reflected direction `li`.
        let bsdf_pdf = |li: [f64; 3]| {
            if li[2] <= 0.0 {
                return 0.0;
            }
            let c = spec_cols(li, lo, 0.0);
            let mut p = pr * spec_pdf(&smp, c[1], c[2]);
            if !scene.mode.sss {
                p += (1.0 - pr) * li[2] / PI;
            }
            p
        };
        // 1. NEE.
        let u3 = [rng.uniform(), rng.uniform(), rng.uniform()];
        if let Some((i, wi, dist, pl)) = sample_light(scene, origin, n, u3)
            && wi.dot(&ng) > 0.0
            && !scene.occluded(origin, wi, dist)
        {
            let li = local(wi);
            let w = mis(pl, bsdf_pdf(li));
            let cols = spec_cols(li, lo, kspec);
            let fs = spec_f64(&th, cols[0], cols[1], cols[2], cols[3], cols[4]);
            let s = li[2] * w / pl;
            let mut df = 0.0;
            for k in 0..NW {
                let g = tp[k] * s * scene.lights[i].le(lam[k]);
                rad[k] += g * (fs + fd[k]);
                if let Some((_, dl)) = replay {
                    df += dl[k] * g;
                }
            }
            if df != 0.0 {
                rec.spec.push(SpecRecord { cols, df });
            }
        }
        // 2. Reflect, or go through.
        let (uc, um, u1, u2) = (rng.uniform(), rng.uniform(), rng.uniform(), rng.uniform());
        if uc < pr || !scene.mode.sss {
            // A reflected direction: a GGX lobe with probability P_r, else
            // (Lambert mode) the cosine-weighted base. Both strategies share
            // the direction, so the weight is the mixture's.
            let li = if uc < pr {
                let r = if um < graph::MIX[0] { smp[3] } else { smp[4] };
                crate::ggx::sample(r * r, lo, [u1, u2]).0
            } else {
                local(cosine(n, u1, u2).0)
            };
            let pdf = bsdf_pdf(li);
            if li[2] <= 0.0 || !(pdf > 0.0 && pdf.is_finite()) {
                break;
            }
            let wi = t * li[0] + b * li[1] + n * li[2];
            if wi.dot(&ng) <= 0.0 {
                break;
            }
            let cols = spec_cols(li, lo, kspec);
            let fs = spec_f64(&th, cols[0], cols[1], cols[2], cols[3], cols[4]);
            if let Some((l, dl)) = replay {
                // f_k = fs + fd_k, ∂ log f_k = ∂fs / f_k.
                let df: f64 = (0..NW)
                    .map(|k| {
                        let f = fs + fd[k];
                        if f > 0.0 {
                            dl[k] * (l[k] - rad[k]) / f
                        } else {
                            0.0
                        }
                    })
                    .sum();
                if df != 0.0 {
                    rec.spec.push(SpecRecord { cols, df });
                }
            }
            for k in 0..NW {
                tp[k] *= (fs + fd[k]) * li[2] / pdf;
            }
            prev = (origin, n, pdf);
            ray = Ray::new(origin, wi);
            continue;
        }
        // 3. Into the medium: four walks, one per wavelength.
        let through = (1.0 - fr_o) / (1.0 - pr);
        for v in &mut tp {
            *v *= through;
        }
        let (m, hb) = scene.chromophores(uv);
        let mm = scene.mm_per_unit;
        let coef: [[f64; 4]; NW] = std::array::from_fn(|k| optics::coefficients(lam[k]));
        let sig = |p: &[f64; PARAMS], k: usize| {
            let c = coef[k];
            (
                p[2] * (p[0] * m * c[0] + p[1] * hb * c[1] + c[2]) * mm,
                p[2] * c[3] * mm,
            )
        };
        let entry = rad;
        let x0 = offset(h.point, -ng, ray.origin, h.t);
        let w0 = cosine(-ng, u1, u2).0;
        let c = ((rng.uniform() * NW as f64) as usize).min(NW - 1);
        let mut cont = None;
        for k in 0..NW {
            let wk = walk(
                &scene.bvh,
                Some(&scene.accel),
                x0,
                w0,
                sig(&th, k),
                sig(&smp, k),
                scene.max_events,
                &mut rng,
            );
            counts.walks += 1;
            counts.events += wk.events;
            if wk.exit.is_none() {
                counts.capped += 1;
            }
            let mut exit_at = None;
            if let Some((eh, w, x)) = &wk.exit {
                let mut ne: Vec3 = *eh.normal.as_ref();
                if ne.dot(w) < 0.0 {
                    ne = -ne;
                }
                let xo = offset(eh.point, ne, *x, eh.t);
                // The exit lobe shades with the smooth normal (the faceted
                // geometric one shows the mesh); it must agree in side.
                let sm = mesh::shade(&scene.mesh, &scene.bvh, eh);
                let mut ns = if sm.ng.dot(&ne) < 0.0 { -sm.ns } else { sm.ns };
                if ns.dot(&ne) <= 0.0 {
                    ns = ne;
                }
                exit_at = Some((xo, ne, ns));
                // 4. The exit's NEE, this wavelength only.
                let u3 = [rng.uniform(), rng.uniform(), rng.uniform()];
                if let Some((i, wi, dist, pl)) = sample_light(scene, xo, ns, u3)
                    && wi.dot(&ne) > 0.0
                    && !scene.occluded(xo, wi, dist)
                {
                    let ci = wi.dot(&ns);
                    let w = mis(pl, ci / PI);
                    rad[k] += tp[k] * wk.weight / PI * ci * scene.lights[i].le(lam[k]) * w / pl;
                }
            }
            if let Some((l, dl)) = replay {
                let wgt = dl[k] * (l[k] - entry[k]);
                if wgt != 0.0 {
                    let cf = coef[k];
                    // σ·t in mm: (σ per mm)(t in mm).
                    rec.medium.push(MediumRecord {
                        cols: [
                            wgt * wk.scatters,
                            wgt * wk.length * mm,
                            m,
                            hb,
                            cf[0],
                            cf[1],
                            cf[2],
                            cf[3],
                        ],
                    });
                }
            }
            if k == c {
                cont = exit_at.map(|e| (e, wk.weight));
            }
        }
        // The path goes on from walk `c`'s exit, that wavelength alone.
        let Some(((xo, ne, ns), wgt)) = cont else {
            break;
        };
        for (k, v) in tp.iter_mut().enumerate() {
            *v = if k == c { *v * wgt * NW as f64 } else { 0.0 };
        }
        let (u1, u2) = (rng.uniform(), rng.uniform());
        let (wi, cz) = cosine(ns, u1, u2);
        if wi.dot(&ne) <= 0.0 {
            break;
        }
        prev = (xo, ns, cz / PI);
        ray = Ray::new(xo, wi);
    }
    (rad, rec)
}

/// What one random walk did.
#[derive(Debug, Clone)]
pub struct Walk {
    /// The boundary hit it left through, the direction it left along and
    /// the last point inside; `None` if it ran out of events (a walk that
    /// leaves an open mesh through a hole wanders until it does: absorbed).
    pub exit: Option<(irys_core::Hit, Vec3, Point3)>,
    /// `Π f / p` over its events.
    pub weight: f64,
    /// Scatter events.
    pub scatters: f64,
    /// Path length inside, scene units.
    pub length: f64,
    /// Events (scatters + the exit).
    pub events: u64,
}

/// A random walk from `x` along `w` inside the closed surface `bvh`, in a
/// medium of `(σa, σs)` per scene unit (`model`), distances drawn at
/// `samp` (the sampler's, detached), Henyey–Greenstein at [`G`]. Consumes
/// two uniforms per scatter and one per distance.
#[allow(clippy::too_many_arguments)]
pub fn walk(
    bvh: &Bvh<TriMesh>,
    accel: Option<&Accel>,
    x: Point3,
    w: Vec3,
    model: (f64, f64),
    samp: (f64, f64),
    max_events: usize,
    rng: &mut Rng,
) -> Walk {
    walk_g(bvh, accel, x, w, model, samp, max_events, rng, G)
}

/// [`walk`] at asymmetry `g`.
#[allow(clippy::too_many_arguments)]
pub fn walk_g(
    bvh: &Bvh<TriMesh>,
    accel: Option<&Accel>,
    mut x: Point3,
    mut w: Vec3,
    (sa_m, ss_m): (f64, f64),
    (sa_s, ss_s): (f64, f64),
    max_events: usize,
    rng: &mut Rng,
    g: f64,
) -> Walk {
    let (st_m, st_s) = (sa_m + ss_m, sa_s + ss_s);
    let mut safe: Option<(Point3, f64)> = None;
    let mut out = Walk {
        exit: None,
        weight: 1.0,
        scatters: 0.0,
        length: 0.0,
        events: 0,
    };
    for _ in 0..max_events {
        let tt = -(1.0 - rng.uniform()).ln() / st_s;
        out.events += 1;
        let end = x + w * tt;
        let inside =
            |s: &Option<(Point3, f64)>| s.is_some_and(|(c, r)| (end - c).norm_squared() < r * r);
        // A step ending inside the safe sphere cannot reach the boundary
        // ([`Accel`]); otherwise refresh the sphere here and, if the step
        // still leaves it, trace it — to `t` alone, since only a boundary
        // nearer than `t` matters.
        let mut hit = None;
        if !inside(&safe) {
            if let Some(a) = accel {
                safe = Some((x, a.distance(x) * (1.0 - 1e-6)));
            }
            if !inside(&safe) {
                hit = bvh.trace_closest_range(&Ray::new(x, w), 0.0, tt);
            }
        }
        if let Some(bh) = hit {
            // Leaves: f = e^{−σt d}, probability e^{−σ̄t d}.
            out.weight *= (-st_m * bh.t).exp() / (-st_s * bh.t).exp();
            out.length += bh.t;
            out.exit = Some((bh, w, x));
            return out;
        }
        // Scatters: f = σs e^{−σt t}, pdf σ̄t e^{−σ̄t t}.
        out.weight *= ss_m * (-st_m * tt).exp() / (st_s * (-st_s * tt).exp());
        out.scatters += 1.0;
        out.length += tt;
        x = end;
        let (ua, ub) = (rng.uniform(), rng.uniform());
        w = hg_sample(w, g, ua, ub);
    }
    out.weight = 0.0;
    out
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The forward pass: `L` per lane (`path · NW + k`).
pub fn forward(scene: &Scene, theta: &[f64; PARAMS], seed: u64) -> (Vec<f64>, Counts) {
    let out: Vec<([f64; NW], Counts)> = (0..scene.paths())
        .into_par_iter()
        .map(|i| {
            let mut c = Counts::default();
            let (l, _) = trace_path(scene, theta, seed, i, None, &mut c);
            (l, c)
        })
        .collect();
    let mut counts = Counts::default();
    let mut l = Vec::with_capacity(out.len() * NW);
    for (li, c) in out {
        l.extend_from_slice(&li);
        counts.events += c.events;
        counts.walks += c.walks;
        counts.capped += c.capped;
    }
    (l, counts)
}

/// The IR programs the replay reduces its records through.
pub struct Adjoints {
    spec: crate::ir::Program,
    medium: crate::ir::Program,
    width: usize,
}

impl Adjoints {
    /// Build both adjoint lowerings for batches of `width` records:
    /// compiled C when `native`, the f64 interpreter otherwise.
    pub fn new(width: usize, native: bool) -> Self {
        Self {
            spec: Spec::build(width).adjoint(native),
            medium: Medium::build(width).adjoint(native),
            width,
        }
    }

    /// Whether the programs are compiled C.
    pub fn is_native(&self) -> bool {
        self.spec.is_native()
    }

    /// `δθ` of all `records`, in order, `width` at a time.
    pub fn reduce(
        &mut self,
        theta: &[f64; PARAMS],
        mode: Mode,
        records: &[Records],
    ) -> [f64; PARAMS] {
        let th = effective(mode, theta);
        let n = self.width;
        let mut grad = [0.0; PARAMS];
        let spec: Vec<&SpecRecord> = records.iter().flat_map(|r| r.spec.iter()).collect();
        let med: Vec<&MediumRecord> = records.iter().flat_map(|r| r.medium.iter()).collect();
        self.spec.write(SPEC_COLUMNS, &th);
        for chunk in spec.chunks(n) {
            let mut cols = vec![vec![1.0; n]; SPEC_COLUMNS];
            let mut df = vec![0.0; n];
            for c in &mut cols[4] {
                *c = 0.0;
            }
            for (j, r) in chunk.iter().enumerate() {
                for (k, col) in cols.iter_mut().enumerate() {
                    col[j] = r.cols[k];
                }
                df[j] = r.df;
            }
            for (k, col) in cols.iter().enumerate() {
                self.spec.write(k, col);
            }
            self.spec.write(SPEC_COLUMNS + 1, &df);
            self.spec.run();
            for (g, d) in grad.iter_mut().zip(self.spec.read(0)) {
                *g += d;
            }
        }
        self.medium.write(MEDIUM_COLUMNS, &th);
        for chunk in med.chunks(n) {
            let mut cols = vec![vec![0.0; n]; MEDIUM_COLUMNS];
            let mut dl = vec![0.0; n];
            for c in &mut cols[7] {
                *c = 1.0;
            }
            for (j, r) in chunk.iter().enumerate() {
                for (k, col) in cols.iter_mut().enumerate() {
                    col[j] = r.cols[k];
                }
                dl[j] = 1.0;
            }
            for (k, col) in cols.iter().enumerate() {
                self.medium.write(k, col);
            }
            self.medium.write(MEDIUM_COLUMNS + 1, &dl);
            self.medium.run();
            for (g, d) in grad.iter_mut().zip(self.medium.read(0)) {
                *g += d;
            }
        }
        if !mode.dual {
            // The second lobe was the first.
            grad[3] += grad[4];
            grad[4] = 0.0;
        }
        grad
    }
}

/// The replay pass: re-walk every path against the forward `L` and the loss
/// adjoint `dl` per lane; returns `δθ`, the replayed `L` (bit-identical to
/// `l`), and the record count.
pub fn replay(
    scene: &Scene,
    theta: &[f64; PARAMS],
    seed: u64,
    l: &[f64],
    dl: &[f64],
    adj: &mut Adjoints,
) -> ([f64; PARAMS], Vec<f64>, usize) {
    let out: Vec<([f64; NW], Records)> = (0..scene.paths())
        .into_par_iter()
        .map(|i| {
            let li: [f64; NW] = std::array::from_fn(|k| l[i * NW + k]);
            let di: [f64; NW] = std::array::from_fn(|k| dl[i * NW + k]);
            let mut c = Counts::default();
            trace_path(scene, theta, seed, i, Some((li, di)), &mut c)
        })
        .collect();
    let l2: Vec<f64> = out.iter().flat_map(|(l, _)| l.iter().copied()).collect();
    let recs: Vec<Records> = out.into_iter().map(|(_, r)| r).collect();
    let count = recs.iter().map(|r| r.spec.len() + r.medium.len()).sum();
    (adj.reduce(theta, scene.mode, &recs), l2, count)
}

/// What lane `lane`'s `L` is worth to each channel of its pixel.
pub fn lane_weight(scene: &Scene, seed: u64, lane: usize) -> [f64; 3] {
    let u = Rng::new(seed, (lane / NW) as u64).uniform();
    let v = u + (lane % NW) as f64 / NW as f64;
    let lambda = sample_lambda_nm(v - v.floor());
    hero_weight(lambda).map(|w| w as f64 / (scene.spp * NW) as f64)
}

/// Linear-sRGB image (`w·h·3`) from per-lane `L`.
pub fn image(scene: &Scene, seed: u64, l: &[f64]) -> Vec<f64> {
    let mut img = vec![0.0; scene.size.0 * scene.size.1 * 3];
    for (j, &lj) in l.iter().enumerate() {
        let w = lane_weight(scene, seed, j);
        let px = j / NW / scene.spp;
        for c in 0..3 {
            img[px * 3 + c] += lj * w[c];
        }
    }
    img
}

/// The loss adjoint per lane from the adjoint per pixel channel.
pub fn lane_adjoint(scene: &Scene, seed: u64, d_img: &[f64]) -> Vec<f64> {
    (0..scene.lanes())
        .map(|j| {
            let w = lane_weight(scene, seed, j);
            let px = j / NW / scene.spp;
            (0..3).map(|c| d_img[px * 3 + c] * w[c]).sum()
        })
        .collect()
}

/// Render, score, replay: `(loss, δθ)`, primal and adjoint seeds as in
/// [`crate::prb::gradient`].
pub fn gradient(
    scene: &Scene,
    theta: &[f64; PARAMS],
    seeds: (u64, u64),
    loss: &impl crate::prb::Loss,
    adj: &mut Adjoints,
) -> (f64, [f64; PARAMS]) {
    let (primal, adjoint) = seeds;
    let (l, _) = forward(scene, theta, primal);
    let (value, d_img) = loss.eval(&image(scene, primal, &l));
    let l = if adjoint == primal {
        l
    } else {
        forward(scene, theta, adjoint).0
    };
    let dl = lane_adjoint(scene, adjoint, &d_img);
    let (g, _, _) = replay(scene, theta, adjoint, &l, &dl, adj);
    (value, g)
}

/// The loss alone.
pub fn loss(scene: &Scene, theta: &[f64; PARAMS], seed: u64, loss: &impl crate::prb::Loss) -> f64 {
    let (l, _) = forward(scene, theta, seed);
    loss.eval(&image(scene, seed, &l)).0
}

/// The default parameters: the inversion's chromophores as they are, the
/// measured density, roughness 0.35 / 0.6.
pub const THETA: [f64; PARAMS] = [1.0, 1.0, 1.0, 0.35, 0.6];

/// The ACES filmic curve (Narkowicz's fit of the RRT + ODT), applied to
/// exposure-scaled linear sRGB, then the sRGB transfer, to 8 bits. Display
/// only: no loss ever sees it.
pub fn tonemap(img: &[f64], exposure: f64) -> Vec<u8> {
    img.iter()
        .map(|&v| {
            let x = (v * exposure).max(0.0);
            let y = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
            let y = y.clamp(0.0, 1.0);
            let s = if y <= 0.003_130_8 {
                12.92 * y
            } else {
                1.055 * y.powf(1.0 / 2.4) - 0.055
            };
            (s * 255.0 + 0.5) as u8
        })
        .collect()
}
