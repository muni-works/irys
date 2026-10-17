//! Path replay backpropagation (Vicini, Speierer & Jakob, SIGGRAPH 2021) on
//! a wavefront, over one scene: a rough-conductor ball on a grey floor under
//! a sky with a soft sun.
//!
//! # The estimator, and its derivative
//!
//! A path that leaves the camera, scatters at vertices `1 … k` and escapes
//! to the sky collects, after each escape or emitter hit, a term
//!
//! ```text
//! L = Σ_k  T_k · Le_k,      T_k = Π_{j<k} f_j cos_j / p_j
//! ```
//!
//! where `p_j` is the pdf the direction was *actually drawn from*. Sampling
//! is detached: `p_j` is the number the sampler returned, not a function of
//! `θ`. That is what keeps the gradient unbiased — the expectation of
//! `f/p · g` over `ω ~ p` is `∫ f g dω` whatever `p` is, so `∂θ` passes
//! through the integral onto `f` alone (no boundary terms: `θ` moves no
//! geometry, so no discontinuity in `ω` moves with it). Then
//!
//! ```text
//! ∂L/∂θ = Σ_k T_k Le_k Σ_{j<k} ∂θ log f_j
//!       = Σ_j ∂θ log f_j · Σ_{k>j} T_k Le_k
//!       = Σ_j ∂θ log f_j · (L − L_j)
//! ```
//!
//! with `L_j` what the path had collected before it scattered at `j`. Given
//! the loss's adjoint `δL` for the path, each vertex contributes
//!
//! ```text
//! δθ += δL · (L − L_j) · ∂θ f_j / f_j
//! ```
//!
//! which is the IR's adjoint lowering with `δf_j = δL (L − L_j) / f_j`. (The
//! design's "`(L − L_so_far) · ∂f/∂θ`" is this with the `1/f_j` folded in:
//! the remaining radiance already carries `f_j` once.)
//!
//! # The two passes
//!
//! The forward pass keeps one number per path, `L`. The replay pass reseeds
//! every path's generator, walks the identical path again, and at each
//! vertex knows `L` (from the forward pass) and `L_j` (its own running sum),
//! which is everything the formula wants. Nothing is stored per vertex: the
//! wavefront's state is [`Path`] × paths plus one column buffer of the same
//! width, reused at every bounce. [`Stats::state_words`] counts it, and a
//! test holds it fixed while the depth changes.
//!
//! # Hero wavelengths
//!
//! A path carries [`Scene::wavelengths`] (default [`HERO`] = 4) stratified
//! wavelengths: the hero `λ₀` from the path's first uniform, the rest
//! rotated by `k/4` of the band (Wilkie et al. 2014). Directions are drawn
//! by the hero alone; every vertex is evaluated per (vertex, wavelength) —
//! a *lane* — so the graph's batch is `paths × 4` wide. GGX's sampler does
//! not depend on λ, so all four share the hero's pdf and the hero MIS
//! weight `p_k / Σ p` is `1/4` for each: the image averages the four lanes.
//! Each lane is a path in the PRB sense, with its own `L`, `L_j`, `δL`.

use irys_core::{Aabb, Bvh, Geometry, Hit, Point3, Ray, TriMesh, Vec3};
use irys_light::spectrum::{hero_weight, sample_lambda_nm};
use irys_light::splats::{self, Splats};
use irys_prims::analytic::{Analytic, Prim};

use crate::ggx::{self, COLUMNS, DF, Graph, PARAMS, THETA, band_weights, normalize};

/// Most wavelengths a path carries (Wilkie et al., "Hero Wavelength Spectral
/// Sampling", EGSR 2014).
pub const HERO: usize = 4;

/// The `w` wavelengths of a path whose hero draw is `u`: the hero, then
/// `w − 1` more rotated by `k/w` of the band, so the set is stratified.
pub fn wavelengths(u: f64, w: usize) -> [f64; HERO] {
    std::array::from_fn(|k| {
        let v = u + k as f64 / w as f64;
        sample_lambda_nm(v - v.floor())
    })
}
use crate::ir::Program;

/// SplitMix64, keyed per path. Replay determinism is this generator being
/// reseeded to the same key and consumed in the same order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rng(u64);

impl Rng {
    /// The generator for path `path` of the frame drawn with `seed`.
    pub fn new(seed: u64, path: u64) -> Self {
        let mut r = Self(seed ^ path.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        r.next();
        r
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// What an opaque primitive is made of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    /// The rough conductor: `θ`'s GGX graph.
    Conductor,
    /// The grey Lambertian floor, θ-free ([`Scene::floor_albedo`]).
    Diffuse,
}

/// Every opaque primitive of a scene behind one [`Geometry`], so one tree
/// holds them: index `i < mesh.len()` is triangle `i`, past that analytic
/// primitive `i − mesh.len()`. Hits report this combined index.
pub struct Opaque {
    mesh: TriMesh,
    analytic: Analytic,
    surfaces: Vec<Surface>,
}

impl Opaque {
    /// A mesh, all of `mesh_surface`, and analytic prims, one surface each.
    pub fn new(
        mesh: TriMesh,
        mesh_surface: Surface,
        analytic: Analytic,
        analytic_surfaces: &[Surface],
    ) -> Self {
        assert_eq!(
            analytic.len(),
            analytic_surfaces.len(),
            "one surface per prim"
        );
        let mut surfaces = vec![mesh_surface; mesh.len()];
        surfaces.extend_from_slice(analytic_surfaces);
        Self {
            mesh,
            analytic,
            surfaces,
        }
    }

    /// The triangles.
    pub fn mesh(&self) -> &TriMesh {
        &self.mesh
    }

    /// The analytic primitives.
    pub fn analytic(&self) -> &Analytic {
        &self.analytic
    }

    /// What primitive `i` (combined index) is made of.
    pub fn surface(&self, i: usize) -> Surface {
        self.surfaces[i]
    }
}

impl Geometry for Opaque {
    fn len(&self) -> usize {
        self.mesh.len() + self.analytic.len()
    }

    fn bounds(&self, i: usize) -> Aabb {
        let nt = self.mesh.len();
        if i < nt {
            self.mesh.bounds(i)
        } else {
            self.analytic.bounds(i - nt)
        }
    }

    fn intersect(&self, ray: &Ray, i: usize, t_min: f64, t_max: f64) -> Option<Hit> {
        let nt = self.mesh.len();
        let hit = if i < nt {
            self.mesh.intersect(ray, i, t_min, t_max)
        } else {
            self.analytic.intersect(ray, i - nt, t_min, t_max)
        };
        hit.map(|mut h| {
            h.prim = i as u32;
            h
        })
    }
}

/// A splat's baked radiance at `lambda_nm`: its linear-sRGB colour lifted
/// onto the conductor's three-hat basis (450, 550, 650 nm — blue, green,
/// red). Linear, so it commutes with the compositor's sums. The GPU does the
/// same in `wavefront.wgsl`.
pub fn splat_radiance(rgb: [f32; 3], lambda_nm: f64) -> f64 {
    let w = band_weights(lambda_nm);
    w[0] * rgb[2] as f64 + w[1] * rgb[1] as f64 + w[2] * rgb[0] as f64
}

/// The ball, the floor, the camera and the sky; optionally a splat cloud.
pub struct Scene {
    pub(crate) opaque: Bvh<Opaque>,
    /// Gaussian splats, composited front to back along every segment
    /// ([`splats::composite`]): emissive and absorbing, θ-free.
    pub(crate) splats: Option<Bvh<Splats>>,
    /// Image width and height in pixels.
    pub size: (usize, usize),
    /// Paths per pixel.
    pub spp: usize,
    /// Longest path, in surface vertices.
    pub depth: usize,
    /// Wavelengths per path, `1 ..= HERO`. One is plain hero sampling.
    pub wavelengths: usize,
    /// The floor's (grey, θ-free) albedo.
    pub floor_albedo: f64,
    /// When set, diffuse surfaces are a checker of this square size, the
    /// odd squares at [`CHECKER_DARK`] of the albedo.
    pub floor_checker: Option<f64>,
    /// When set, the conductor samples at this roughness instead of the
    /// current one. Paths then do not move with `θ`, so central differences
    /// with common random numbers see exactly the estimator PRB
    /// differentiates — the tight test.
    pub frozen_alpha: Option<f64>,
    pub(crate) tan_half: f64,
    pub(crate) eye: Point3,
    pub(crate) forward: Vec3,
    pub(crate) right: Vec3,
    pub(crate) up: Vec3,
}

/// The odd checker squares' share of the floor albedo.
pub const CHECKER_DARK: f64 = 0.55;

impl Scene {
    /// The marble-like test scene at `w × h`, `spp` paths per pixel: the
    /// ball and a disc floor, two analytic prims.
    pub fn new(w: usize, h: usize, spp: usize) -> Self {
        let prims = vec![
            Prim::Sphere {
                center: Point3::new(0.0, 0.0, 1.0),
                radius: 1.0,
            },
            Prim::Disc {
                center: Point3::new(0.0, 0.0, 0.0),
                normal: irys_core::Dir3::new_normalize(Vec3::new(0.0, 0.0, 1.0)),
                radius: 30.0,
            },
        ];
        let opaque = Opaque::new(
            TriMesh::default(),
            Surface::Diffuse,
            Analytic::from_prims(prims),
            &[Surface::Conductor, Surface::Diffuse],
        );
        Self::with(opaque, None, (w, h), spp, Point3::new(0.0, -4.5, 2.2), 20.0)
    }

    /// Phase 2b's scene: the ball on a triangle-mesh floor (a 16 × 16 grid,
    /// 512 triangles, checkered), in front of a bush of [`BUSH_SPLATS`]
    /// Gaussian splats.
    pub fn garden(w: usize, h: usize, spp: usize) -> Self {
        let opaque = Opaque::new(
            grid_floor(16, 12.0),
            Surface::Diffuse,
            Analytic::from_prims(vec![Prim::Sphere {
                center: Point3::new(0.0, 0.0, 1.0),
                radius: 1.0,
            }]),
            &[Surface::Conductor],
        );
        let mut s = Self::with(
            opaque,
            Some(Bvh::build(bush(BUSH_SPLATS, 17))),
            (w, h),
            spp,
            Point3::new(0.0, -4.5, 2.2),
            24.0,
        );
        s.floor_checker = Some(1.0);
        s
    }

    fn with(
        opaque: Opaque,
        splats: Option<Bvh<Splats>>,
        size: (usize, usize),
        spp: usize,
        eye: Point3,
        half_fov_deg: f64,
    ) -> Self {
        let forward = (Point3::new(0.0, 0.0, 0.8) - eye).normalize();
        let right = forward.cross(&Vec3::new(0.0, 0.0, 1.0)).normalize();
        let up = right.cross(&forward);
        Self {
            opaque: Bvh::build(opaque),
            splats,
            size,
            spp,
            depth: 4,
            wavelengths: HERO,
            floor_albedo: 0.5,
            floor_checker: None,
            frozen_alpha: None,
            tan_half: half_fov_deg.to_radians().tan(),
            eye,
            forward,
            right,
            up,
        }
    }

    /// The opaque tree.
    pub fn opaque(&self) -> &Bvh<Opaque> {
        &self.opaque
    }

    /// Replace (or with `None`, remove) the splat cloud.
    pub fn set_splats(&mut self, splats: Option<Splats>) {
        self.splats = splats.map(Bvh::build);
    }

    /// The splat tree, if the scene has one.
    pub fn splats(&self) -> Option<&Bvh<Splats>> {
        self.splats.as_ref()
    }

    /// The diffuse albedo at `p`.
    pub fn albedo(&self, p: Point3) -> f64 {
        match self.floor_checker {
            Some(s) if ((p.x / s).floor() + (p.y / s).floor()).rem_euclid(2.0) == 1.0 => {
                self.floor_albedo * CHECKER_DARK
            }
            _ => self.floor_albedo,
        }
    }

    /// Paths in one frame.
    pub fn paths(&self) -> usize {
        self.size.0 * self.size.1 * self.spp
    }

    /// Lanes in one frame: a lane is one (path, wavelength), and is what
    /// the graph is batched over and what the forward pass keeps an `L` for.
    /// Lane `i · wavelengths + k` is path `i`'s `k`-th wavelength.
    pub fn lanes(&self) -> usize {
        self.paths() * self.wavelengths
    }

    fn camera_ray(&self, path: usize, rng: &mut Rng) -> Ray {
        let (w, h) = self.size;
        let px = path / self.spp;
        let (x, y) = (
            (px % w) as f64 + rng.uniform(),
            (px / w) as f64 + rng.uniform(),
        );
        let tan = self.tan_half;
        let sx = (2.0 * x / w as f64 - 1.0) * tan * w as f64 / h as f64;
        let sy = (1.0 - 2.0 * y / h as f64) * tan;
        Ray::new(self.eye, self.forward + self.right * sx + self.up * sy)
    }
}

/// A flat `n × n` grid of quads over `[−half, half]²` at `z = 0`, two
/// triangles each, wound so the face normal is `+z`.
pub fn grid_floor(n: usize, half: f64) -> TriMesh {
    let at = |i: usize| -half + 2.0 * half * i as f64 / n as f64;
    let positions = (0..=n)
        .flat_map(|j| (0..=n).map(move |i| (i, j)))
        .map(|(i, j)| Point3::new(at(i), at(j), 0.0))
        .collect();
    let v = |i: usize, j: usize| (j * (n + 1) + i) as u32;
    let indices: Vec<u32> = (0..n)
        .flat_map(|j| (0..n).map(move |i| (i, j)))
        .flat_map(|(i, j)| {
            [
                v(i, j),
                v(i + 1, j),
                v(i + 1, j + 1),
                v(i, j),
                v(i + 1, j + 1),
                v(i, j + 1),
            ]
        })
        .collect();
    TriMesh::new(positions, Vec::new(), &indices)
}

/// Splats in [`Scene::garden`]'s bush.
pub const BUSH_SPLATS: usize = 900;

/// A procedural bush: `n` anisotropic Gaussians filling an ellipsoid that
/// sits on the floor behind and left of the ball, greens darkening toward
/// the ground. Degree-0 colour, so the radiance is view-independent.
pub fn bush(n: usize, seed: u64) -> Splats {
    const SH_C0: f32 = 0.282_094_8;
    let centre = [-1.6, 2.5, 0.9];
    let radii = [1.0, 0.8, 0.9];
    let mut rng = Rng::new(seed, 0);
    let mut u = move || rng.uniform() as f32;
    let (mut pos, mut scale, mut quat, mut opac, mut sh) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    while pos.len() < n {
        let q = [2.0 * u() - 1.0, 2.0 * u() - 1.0, 2.0 * u() - 1.0];
        if q[0] * q[0] + q[1] * q[1] + q[2] * q[2] > 1.0 {
            continue;
        }
        let p: [f32; 3] = std::array::from_fn(|k| centre[k] + radii[k] * q[k]);
        pos.push(p);
        scale.push([0.035 + 0.07 * u(), 0.035 + 0.07 * u(), 0.025 + 0.045 * u()]);
        quat.push([
            2.0 * u() - 1.0,
            2.0 * u() - 1.0,
            2.0 * u() - 1.0,
            2.0 * u() - 1.0,
        ]);
        opac.push(0.5 + 0.4 * u());
        // Lighter and yellower toward the top, a few berries.
        let h = (p[2] / (2.0 * radii[2])).clamp(0.0, 1.0);
        let k = 0.6 + 0.6 * u();
        let c = if u() < 0.02 {
            [0.45, 0.05, 0.04]
        } else {
            [
                (0.03 + 0.08 * h) * k,
                (0.10 + 0.18 * h) * k,
                (0.02 + 0.03 * h) * k,
            ]
        };
        sh.push(c.map(|c| (c - 0.5) / SH_C0));
    }
    Splats::from_parts(&pos, &scale, &quat, &opac, &sh)
}

/// Sky radiance towards `d` at `lambda_nm`: a dim gradient and a soft sun.
/// Smooth on purpose — the gradient's variance is the sun's edge.
pub fn sky(d: Vec3, lambda_nm: f64) -> f64 {
    let sun = Vec3::new(0.5, -0.35, 0.8).normalize();
    let tint = 1.0 + 0.4 * (550.0 - lambda_nm) / 200.0;
    tint * (0.15 + 0.25 * d.z.max(0.0)) + 6.0 * (40.0 * (d.dot(&sun) - 1.0)).exp()
}

/// One path's whole state on the wavefront. Fixed size: this is the O(1).
#[derive(Debug, Clone, Copy)]
struct Path {
    ray: Ray,
    throughput: [f64; HERO],
    /// `L_j`: radiance collected so far, per wavelength.
    radiance: [f64; HERO],
    lambda: [f64; HERO],
    rng: Rng,
    alive: bool,
}

/// What a pass measured.
#[derive(Debug, Clone, Default)]
pub struct Stats {
    /// f64 words of wavefront state the pass held, all told. Independent of
    /// depth by construction; tests check it.
    pub state_words: usize,
    /// Conductor vertices evaluated (per path, not per wavelength).
    pub vertices: usize,
}

/// The conductor's two programs, built for one wavefront width.
pub struct Kernels {
    eval: Program,
    adjoint: Program,
}

impl Kernels {
    /// Compiled C when `native` (and a compiler exists), the f64
    /// interpreter otherwise. `n` is lanes ([`Scene::lanes`]).
    pub fn new(n: usize, native: bool) -> Self {
        let g = Graph::new(n);
        Self {
            eval: g.eval(native),
            adjoint: g.adjoint(native),
        }
    }

    /// Whether the eval program is compiled C.
    pub fn is_native(&self) -> bool {
        self.eval.is_native()
    }
}

/// A local frame around `n` (Duff et al. 2017).
fn basis(n: Vec3) -> (Vec3, Vec3) {
    let s = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    (
        Vec3::new(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
        Vec3::new(b, s + n.y * n.y * a, -n.y),
    )
}

/// Trace every path once. `replay = None` is the forward pass and returns
/// each path's `L`; `Some((l, dl))` replays against the forward pass's `L`
/// and the loss adjoint per path, accumulating into `grad`.
fn trace(
    scene: &Scene,
    theta: &[f64; PARAMS],
    seed: u64,
    k: &mut Kernels,
    replay: Option<(&[f64], &[f64])>,
    grad: &mut [f64; PARAMS],
) -> (Vec<f64>, Stats) {
    let n = scene.paths();
    let nw = scene.wavelengths;
    assert!((1..=HERO).contains(&nw), "wavelengths {nw}");
    let lanes = scene.lanes();
    let mut paths: Vec<Path> = (0..n)
        .map(|i| {
            let mut rng = Rng::new(seed, i as u64);
            let lambda = wavelengths(rng.uniform(), nw);
            let ray = scene.camera_ray(i, &mut rng);
            Path {
                ray,
                throughput: [1.0; HERO],
                radiance: [0.0; HERO],
                lambda,
                rng,
                alive: true,
            }
        })
        .collect();
    // The column buffer: one row of COLUMNS + δf per lane, reused.
    let mut cols = vec![vec![0.0; lanes]; COLUMNS];
    let mut df = vec![0.0; lanes];
    let mut sampled = vec![(Vec3::zeros(), 0.0f64, 0.0f64); n]; // (wi, cos_i, pdf)
    let mut ball = vec![false; n];
    let mut stats = Stats {
        state_words: n * (std::mem::size_of::<Path>() + std::mem::size_of_val(&sampled[0]) + 1) / 8
            + (COLUMNS + 1) * lanes,
        vertices: 0,
    };
    k.eval.write(THETA, theta);
    k.adjoint.write(THETA, theta);

    for _ in 0..scene.depth {
        let mut any = false;
        for (i, p) in paths.iter_mut().enumerate() {
            // Inactive slots get a harmless vertex and no adjoint.
            for j in i * nw..(i + 1) * nw {
                for (c, v) in cols.iter_mut().zip([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) {
                    c[j] = v;
                }
                df[j] = 0.0;
            }
            ball[i] = false;
            if !p.alive {
                continue;
            }
            let hit = scene
                .opaque
                .trace_closest_range(&p.ray, 1e-6, f64::INFINITY);
            // Splats in front of the surface (or of the sky): emissive and
            // absorbing, composited front to back, θ-free.
            if let Some(cloud) = &scene.splats {
                let seg = splats::composite(
                    cloud,
                    &p.ray,
                    1e-6,
                    hit.as_ref().map_or(f64::INFINITY, |h| h.t),
                );
                for w in 0..nw {
                    p.radiance[w] += p.throughput[w] * splat_radiance(seg.radiance, p.lambda[w]);
                    p.throughput[w] *= seg.transmittance as f64;
                }
                if seg.transmittance == 0.0 {
                    p.alive = false;
                    continue;
                }
            }
            let Some(hit) = hit else {
                for w in 0..nw {
                    p.radiance[w] += p.throughput[w] * sky(*p.ray.direction.as_ref(), p.lambda[w]);
                }
                p.alive = false;
                continue;
            };
            let nrm: Vec3 = *hit.normal.as_ref();
            let wo = -*p.ray.direction.as_ref();
            let (t, b) = basis(nrm);
            let origin = hit.point + nrm * 1e-6;
            if scene.opaque.geometry().surface(hit.prim as usize) == Surface::Diffuse {
                // Lambertian floor: cosine sampling, weight = albedo. θ-free.
                let albedo = scene.albedo(hit.point);
                let (u1, u2) = (p.rng.uniform(), p.rng.uniform());
                let (r, phi) = (u1.sqrt(), 2.0 * std::f64::consts::PI * u2);
                let wi =
                    t * (r * phi.cos()) + b * (r * phi.sin()) + nrm * (1.0 - u1).max(0.0).sqrt();
                for tp in &mut p.throughput[..nw] {
                    *tp *= albedo;
                }
                p.ray = Ray::new(origin, wi);
                continue;
            }
            // The hero alone picks the direction. GGX's sampling does not
            // depend on λ, so every wavelength's pdf is the hero's and the
            // hero-wavelength MIS weight reduces to `f_k cos / p`.
            let lo = [wo.dot(&t), wo.dot(&b), wo.dot(&nrm)];
            let u = [p.rng.uniform(), p.rng.uniform()];
            let alpha = scene.frozen_alpha.unwrap_or(theta[0]);
            let (li, pdf) = ggx::sample(alpha, lo, u);
            if li[2] <= 0.0 || lo[2] <= 0.0 || !pdf.is_finite() || pdf <= 0.0 {
                p.alive = false;
                continue;
            }
            let h = normalize([li[0] + lo[0], li[1] + lo[1], li[2] + lo[2]]);
            let oh = lo[0] * h[0] + lo[1] * h[1] + lo[2] * h[2];
            for w in 0..nw {
                let bw = band_weights(p.lambda[w]);
                for (c, v) in cols
                    .iter_mut()
                    .zip([li[2], lo[2], h[2], oh, bw[0], bw[1], bw[2]])
                {
                    c[i * nw + w] = v;
                }
            }
            sampled[i] = (t * li[0] + b * li[1] + nrm * li[2], li[2], pdf);
            p.ray = Ray::new(origin, sampled[i].0);
            ball[i] = true;
            any = true;
        }
        if !any {
            continue;
        }
        for (c, col) in cols.iter().enumerate() {
            k.eval.write(c, col);
        }
        k.eval.run();
        let f = k.eval.read(0);
        let mut need_adjoint = false;
        for (i, p) in paths.iter_mut().enumerate() {
            if !ball[i] {
                continue;
            }
            stats.vertices += 1;
            let (_, cos_i, pdf) = sampled[i];
            for w in 0..nw {
                let j = i * nw + w;
                if let Some((l, dl)) = replay
                    && f[j] > 0.0
                {
                    df[j] = dl[j] * (l[j] - p.radiance[w]) / f[j];
                    need_adjoint |= df[j] != 0.0;
                }
                p.throughput[w] *= f[j] * cos_i / pdf;
            }
            if !p.throughput[..nw].iter().any(|&t| t > 0.0) {
                p.alive = false;
            }
        }
        if need_adjoint {
            for (c, col) in cols.iter().enumerate() {
                k.adjoint.write(c, col);
            }
            k.adjoint.write(DF, &df);
            k.adjoint.run();
            for (g, d) in grad.iter_mut().zip(k.adjoint.read(0)) {
                *g += d;
            }
        }
    }
    (
        paths
            .iter()
            .flat_map(|p| p.radiance[..nw].iter().copied())
            .collect(),
        stats,
    )
}

/// The forward pass: `L` per lane ([`Scene::lanes`]), and what it cost.
pub fn forward(
    scene: &Scene,
    theta: &[f64; PARAMS],
    seed: u64,
    k: &mut Kernels,
) -> (Vec<f64>, Stats) {
    trace(scene, theta, seed, k, None, &mut [0.0; PARAMS])
}

/// The replay pass alone: `δθ` given the forward pass's `l` and the loss
/// adjoint `dl` per path. Also returns the replayed `L`s, which must equal
/// `l` bit for bit.
pub fn replay(
    scene: &Scene,
    theta: &[f64; PARAMS],
    seed: u64,
    k: &mut Kernels,
    l: &[f64],
    dl: &[f64],
) -> ([f64; PARAMS], Vec<f64>, Stats) {
    let mut grad = [0.0; PARAMS];
    let (l2, stats) = trace(scene, theta, seed, k, Some((l, dl)), &mut grad);
    (grad, l2, stats)
}

/// Linear-sRGB image (`w·h·3`, row-major) from per-lane `L`.
pub fn image(scene: &Scene, seed: u64, l: &[f64]) -> Vec<f64> {
    let mut img = vec![0.0; scene.size.0 * scene.size.1 * 3];
    for (j, &lj) in l.iter().enumerate() {
        let w = lane_weight(scene, seed, j);
        let px = j / scene.wavelengths / scene.spp;
        for c in 0..3 {
            img[px * 3 + c] += lj * w[c];
        }
    }
    img
}

/// What one lane's `L` is worth to each channel of its pixel: its
/// wavelength's sRGB response over `spp × wavelengths`. Each wavelength is
/// a uniform draw over the band on its own (the rotation keeps that), so
/// the hero estimator is the plain average of the `w`. Re-derived from the
/// seed, because the forward pass keeps nothing per path but `L`.
pub fn lane_weight(scene: &Scene, seed: u64, lane: usize) -> [f64; 3] {
    let nw = scene.wavelengths;
    let u = Rng::new(seed, (lane / nw) as u64).uniform();
    let lambda = wavelengths(u, nw)[lane % nw];
    hero_weight(lambda).map(|w| w as f64 / (scene.spp * nw) as f64)
}

/// A loss over the image: its value and its adjoint per pixel channel.
pub trait Loss {
    /// `(loss, ∂loss/∂image)`.
    fn eval(&self, img: &[f64]) -> (f64, Vec<f64>);
}

/// Mean squared error to a target image.
pub struct L2<'a>(pub &'a [f64]);

impl Loss for L2<'_> {
    fn eval(&self, img: &[f64]) -> (f64, Vec<f64>) {
        let n = img.len() as f64;
        let loss = img
            .iter()
            .zip(self.0)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / n;
        (
            loss,
            img.iter()
                .zip(self.0)
                .map(|(a, b)| 2.0 * (a - b) / n)
                .collect(),
        )
    }
}

/// The mean pixel value. Linear, so a gradient of it at one seed is an
/// unbiased estimate of the expected gradient — the loss for checking PRB
/// against central differences when paths move with `θ`.
pub struct Mean;

impl Loss for Mean {
    fn eval(&self, img: &[f64]) -> (f64, Vec<f64>) {
        let n = img.len() as f64;
        (img.iter().sum::<f64>() / n, vec![1.0 / n; img.len()])
    }
}

/// Render, score, replay: `(loss, ∂loss/∂θ)` at `θ`.
///
/// `seeds = (primal, adjoint)`. The primal frame is rendered and scored;
/// the adjoint frame's paths are walked forward and replayed against that
/// score. With one seed for both, the result is the exact derivative of
/// that one frame's loss — what central differences with common random
/// numbers measure. For a nonlinear loss that derivative is biased (the
/// L2 gradient `2 (I − T) ∂I` correlates the noise in `I` with the noise in
/// `∂I`, so descent also minimises variance); two seeds decorrelate them,
/// and that is what an optimiser should use.
pub fn gradient(
    scene: &Scene,
    theta: &[f64; PARAMS],
    seeds: (u64, u64),
    k: &mut Kernels,
    loss: &impl Loss,
) -> (f64, [f64; PARAMS]) {
    let (primal, adjoint) = seeds;
    let (l, _) = forward(scene, theta, primal, k);
    let (value, d_img) = loss.eval(&image(scene, primal, &l));
    let l = if adjoint == primal {
        l
    } else {
        forward(scene, theta, adjoint, k).0
    };
    let dl = lane_adjoint(scene, adjoint, &d_img);
    let (grad, _, _) = replay(scene, theta, adjoint, k, &l, &dl);
    (value, grad)
}

/// The loss adjoint per lane of the frame drawn with `seed`, from the
/// adjoint per pixel channel.
pub fn lane_adjoint(scene: &Scene, seed: u64, d_img: &[f64]) -> Vec<f64> {
    (0..scene.lanes())
        .map(|j| {
            let w = lane_weight(scene, seed, j);
            let px = j / scene.wavelengths / scene.spp;
            (0..3).map(|c| d_img[px * 3 + c] * w[c]).sum()
        })
        .collect()
}

/// The loss alone, for central differences.
pub fn loss(
    scene: &Scene,
    theta: &[f64; PARAMS],
    seed: u64,
    k: &mut Kernels,
    loss: &impl Loss,
) -> f64 {
    let (l, _) = forward(scene, theta, seed, k);
    loss.eval(&image(scene, seed, &l)).0
}
