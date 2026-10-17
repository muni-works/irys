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

use std::sync::Arc;

use irys_core::{
    Aabb, Bvh, Geometry, Hit, Instance, InstanceHit, Point3, Ray, Tlas, Transform, TriMesh, Vec3,
};
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

/// Where one instance of a bottom-level tree sits, and what it is tinted.
#[derive(Debug, Clone)]
pub struct Placement {
    /// Index into the scene's bottom-level trees.
    pub blas: usize,
    /// Object → world.
    pub to_world: Transform,
    /// Multiplies the diffuse albedo of this instance's diffuse prims.
    pub albedo: f64,
    /// Whether [`Scene::floor_checker`] patterns this instance.
    pub checker: bool,
}

impl Placement {
    /// Tree `blas` placed by `to_world`, untinted.
    pub fn new(blas: usize, to_world: Transform) -> Self {
        Self {
            blas,
            to_world,
            albedo: 1.0,
            checker: true,
        }
    }

    /// The same placement with diffuse albedo scaled by `albedo`, and no
    /// checker.
    pub fn tinted(mut self, albedo: f64) -> Self {
        self.albedo = albedo;
        self.checker = false;
        self
    }
}

/// Share of light samples [`Env::sample`] sends at the sun lobe; the rest
/// are uniform over the sphere for the sky gradient.
pub const SUN_PICK: f64 = 0.5;

/// The environment: a dim sky gradient and a sun lobe,
/// `L(d) = sky · tint(λ) (0.15 + 0.25 max(d_z, 0)) + sun_radiance · e^{κ (d·sun − 1)}`.
/// The lobe is a von Mises–Fisher density up to scale, so it can be sampled
/// exactly; [`Env::default`] is phases 1–2b's sky.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Env {
    /// Unit direction to the sun.
    pub sun: Vec3,
    /// Lobe concentration: angular radius ≈ 1/√κ.
    pub kappa: f64,
    /// Peak sun radiance.
    pub sun_radiance: f64,
    /// Scale of the gradient.
    pub sky: f64,
}

impl Default for Env {
    fn default() -> Self {
        Self {
            sun: Vec3::new(0.5, -0.35, 0.8).normalize(),
            kappa: 40.0,
            sun_radiance: 6.0,
            sky: 1.0,
        }
    }
}

impl Env {
    /// Radiance towards `d` at `lambda_nm`.
    pub fn radiance(&self, d: Vec3, lambda_nm: f64) -> f64 {
        let tint = 1.0 + 0.4 * (550.0 - lambda_nm) / 200.0;
        tint * self.sky * (0.15 + 0.25 * d.z.max(0.0))
            + self.sun_radiance * (self.kappa * (d.dot(&self.sun) - 1.0)).exp()
    }

    /// A light-sampling direction from three uniforms: the first picks the
    /// sun lobe (vMF, exactly) or the uniform sphere. Detached, like every
    /// sampler here.
    pub fn sample(&self, u: [f64; 3]) -> Vec3 {
        let phi = 2.0 * std::f64::consts::PI * u[2];
        if u[0] < SUN_PICK {
            let k = self.kappa;
            let c = (1.0 + (u[1] + (1.0 - u[1]) * (-2.0 * k).exp()).ln() / k).clamp(-1.0, 1.0);
            let r = (1.0 - c * c).max(0.0).sqrt();
            let (t, b) = basis(self.sun);
            t * (r * phi.cos()) + b * (r * phi.sin()) + self.sun * c
        } else {
            let z = 1.0 - 2.0 * u[1];
            let r = (1.0 - z * z).max(0.0).sqrt();
            Vec3::new(r * phi.cos(), r * phi.sin(), z)
        }
    }

    /// The density [`Env::sample`] draws from, per steradian.
    pub fn pdf(&self, d: Vec3) -> f64 {
        let k = self.kappa;
        let vmf = k / (2.0 * std::f64::consts::PI * (1.0 - (-2.0 * k).exp()))
            * (k * (d.dot(&self.sun) - 1.0)).exp();
        SUN_PICK * vmf + (1.0 - SUN_PICK) / (4.0 * std::f64::consts::PI)
    }
}

/// The power heuristic (β = 2): the weight of the strategy with pdf `a`
/// against the one with pdf `b`.
pub fn mis(a: f64, b: f64) -> f64 {
    let (a2, b2) = (a * a, b * b);
    if a2 + b2 > 0.0 { a2 / (a2 + b2) } else { 0.0 }
}

/// The scene: bottom-level trees of opaque prims placed by a [`Tlas`], the
/// camera and the environment; optionally a splat cloud.
pub struct Scene {
    pub(crate) blas: Vec<Arc<Bvh<Opaque>>>,
    pub(crate) tlas: Tlas<Opaque>,
    /// By instance payload (the index it was placed with).
    pub(crate) placements: Vec<Placement>,
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
    /// Next-event estimation: a shadow ray to [`Env::sample`] at every
    /// vertex, combined with BSDF sampling by the power heuristic. Off, the
    /// estimator is phases 1–2b's, bit for bit. Not with splats (shadow rays
    /// would have to composite them).
    pub nee: bool,
    /// The sky and sun.
    pub env: Env,
    /// When set, the frame is these primary rays instead of the image's:
    /// path `i` leaves the camera through camera-plane point `points[i]`
    /// (`(sx, sy)`, [`Scene::screen`]) and is keyed `i / 2`, so the two rays
    /// of a pair — the two sides of an edge sample ([`crate::vis`]) — draw
    /// the same wavelengths and the same uniforms after the camera.
    pub(crate) points: Option<Vec<[f64; 2]>>,
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
        Self::one(opaque, None, (w, h), spp, 20.0)
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
        let mut s = Self::one(
            opaque,
            Some(Bvh::build(bush(BUSH_SPLATS, 17))),
            (w, h),
            spp,
            24.0,
        );
        s.floor_checker = Some(1.0);
        s
    }

    /// Phase 2c's scene: one triangle-mesh ball ([`uv_sphere`], conductor)
    /// placed twice — once uniformly scaled, once squashed and turned, so the
    /// normal takes the inverse transpose — on the checkered mesh floor (a
    /// third instance, of a second tree). NEE on. Everything, camera
    /// included, is scaled by `scale`: 1000 is the same scene in millimetres.
    pub fn pair(w: usize, h: usize, spp: usize, scale: f64) -> Self {
        let ball = Opaque::new(
            uv_sphere(16, 32),
            Surface::Conductor,
            Analytic::from_prims(Vec::new()),
            &[],
        );
        let floor = Opaque::new(
            grid_floor(16, 12.0),
            Surface::Diffuse,
            Analytic::from_prims(Vec::new()),
            &[],
        );
        let s = |t: Transform| Transform::scale(scale, scale, scale).then(&t);
        let placements = vec![
            Placement::new(
                0,
                s(Transform::translation(-1.15, 0.0, 0.9).then(&Transform::scale(0.9, 0.9, 0.9))),
            ),
            Placement::new(
                0,
                s(Transform::translation(1.1, 0.6, 0.65)
                    .then(&Transform::rotation_z(0.6))
                    .then(&Transform::rotation_x(0.4))
                    .then(&Transform::scale(0.95, 0.7, 0.65))),
            ),
            Placement::new(1, s(Transform::identity())),
        ];
        let mut sc = Self::instanced(
            vec![ball, floor],
            placements,
            None,
            (w, h),
            spp,
            (
                Point3::new(0.0, -4.5 * scale, 2.2 * scale),
                Point3::new(0.0, 0.0, 0.8 * scale),
            ),
            24.0,
        );
        sc.floor_checker = Some(scale);
        sc.nee = true;
        sc
    }

    fn one(
        opaque: Opaque,
        splats: Option<Bvh<Splats>>,
        size: (usize, usize),
        spp: usize,
        half_fov_deg: f64,
    ) -> Self {
        Self::instanced(
            vec![opaque],
            vec![Placement::new(0, Transform::identity())],
            splats,
            size,
            spp,
            (Point3::new(0.0, -4.5, 2.2), Point3::new(0.0, 0.0, 0.8)),
            half_fov_deg,
        )
    }

    /// A scene of bottom-level trees placed by `placements` (one instance
    /// each, its payload its index), a pinhole camera at `eye` looking at
    /// `target` (`+z` up), NEE off and the default [`Env`].
    pub fn instanced(
        blas: Vec<Opaque>,
        placements: Vec<Placement>,
        splats: Option<Bvh<Splats>>,
        size: (usize, usize),
        spp: usize,
        (eye, target): (Point3, Point3),
        half_fov_deg: f64,
    ) -> Self {
        let blas: Vec<Arc<Bvh<Opaque>>> =
            blas.into_iter().map(|o| Arc::new(Bvh::build(o))).collect();
        let instances = placements
            .iter()
            .enumerate()
            .map(|(k, p)| {
                Instance::new(blas[p.blas].clone(), p.to_world.clone(), k)
                    .unwrap_or_else(|| panic!("placement {k}: empty tree or singular transform"))
            })
            .collect();
        let forward = (target - eye).normalize();
        let right = forward.cross(&Vec3::new(0.0, 0.0, 1.0)).normalize();
        let up = right.cross(&forward);
        Self {
            blas,
            tlas: Tlas::build(instances),
            placements,
            splats,
            size,
            spp,
            depth: 4,
            wavelengths: HERO,
            floor_albedo: 0.5,
            floor_checker: None,
            frozen_alpha: None,
            nee: false,
            env: Env::default(),
            points: None,
            tan_half: half_fov_deg.to_radians().tan(),
            eye,
            forward,
            right,
            up,
        }
    }

    /// The top-level tree.
    pub fn tlas(&self) -> &Tlas<Opaque> {
        &self.tlas
    }

    /// The bottom-level trees.
    pub fn blas(&self) -> &[Arc<Bvh<Opaque>>] {
        &self.blas
    }

    /// The placements, by instance payload.
    pub fn placements(&self) -> &[Placement] {
        &self.placements
    }

    /// Replace (or with `None`, remove) the splat cloud.
    pub fn set_splats(&mut self, splats: Option<Splats>) {
        self.splats = splats.map(Bvh::build);
    }

    /// The splat tree, if the scene has one.
    pub fn splats(&self) -> Option<&Bvh<Splats>> {
        self.splats.as_ref()
    }

    /// What the prim an instance hit is made of.
    pub fn surface(&self, h: &InstanceHit) -> Surface {
        self.blas[self.placements[h.payload].blas]
            .geometry()
            .surface(h.hit.prim as usize)
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

    /// Paths in one frame: the image's, or one per [`Scene::set_points`]
    /// point.
    pub fn paths(&self) -> usize {
        match &self.points {
            Some(p) => p.len(),
            None => self.size.0 * self.size.1 * self.spp,
        }
    }

    /// The generator key of path `i`: `i`, or in point mode the pair `i / 2`.
    pub fn path_key(&self, i: usize) -> u64 {
        match self.points {
            Some(_) => (i / 2) as u64,
            None => i as u64,
        }
    }

    /// Trace these camera-plane points ([`Scene::screen`]) instead of the
    /// image, two per pair; `None` goes back to the image.
    pub fn set_points(&mut self, points: Option<Vec<[f64; 2]>>) {
        if let Some(p) = &points {
            assert!(p.len() % 2 == 0, "points come in pairs");
        }
        self.points = points;
    }

    /// The points being traced, if any.
    pub fn points(&self) -> Option<&[[f64; 2]]> {
        self.points.as_deref()
    }

    /// Move placements: `(payload, object → world)` each, then rebuild the
    /// top-level tree. The bottom-level trees are untouched.
    pub fn place(&mut self, moves: &[(usize, Transform)]) {
        for (k, t) in moves {
            self.placements[*k].to_world = t.clone();
        }
        let instances = self
            .placements
            .iter()
            .enumerate()
            .map(|(k, p)| {
                Instance::new(self.blas[p.blas].clone(), p.to_world.clone(), k)
                    .unwrap_or_else(|| panic!("placement {k}: singular transform"))
            })
            .collect();
        self.tlas = Tlas::build(instances);
    }

    /// Pixel coordinates (`x` right, `y` down, `[0, w) × [0, h)`) to the
    /// camera plane: the ray through them is `forward + right sx + up sy`.
    pub fn screen(&self, x: f64, y: f64) -> [f64; 2] {
        let (w, h) = (self.size.0 as f64, self.size.1 as f64);
        [
            (2.0 * x / w - 1.0) * self.tan_half * w / h,
            (1.0 - 2.0 * y / h) * self.tan_half,
        ]
    }

    /// The pinhole: eye, forward, right, up, `tan` of the half field of view.
    pub fn camera(&self) -> (Point3, Vec3, Vec3, Vec3, f64) {
        (self.eye, self.forward, self.right, self.up, self.tan_half)
    }

    /// Where world point `p` lands in pixel coordinates, and the 2 × 3
    /// Jacobian of that map; `None` behind the eye.
    pub fn project(&self, p: Point3) -> Option<([f64; 2], [Vec3; 2])> {
        let d = p - self.eye;
        let z = d.dot(&self.forward);
        if z <= 0.0 {
            return None;
        }
        let (a, b) = (d.dot(&self.right), d.dot(&self.up));
        let (w, h) = (self.size.0 as f64, self.size.1 as f64);
        let kx = w / (2.0 * self.tan_half * w / h);
        let ky = h / (2.0 * self.tan_half);
        let x = (a / z) * kx + w / 2.0;
        let y = h / 2.0 - (b / z) * ky;
        let jx = (self.right * z - self.forward * a) * (kx / (z * z));
        let jy = (self.up * z - self.forward * b) * (-ky / (z * z));
        Some(([x, y], [jx, jy]))
    }

    /// Lanes in one frame: a lane is one (path, wavelength), and is what
    /// the graph is batched over and what the forward pass keeps an `L` for.
    /// Lane `i · wavelengths + k` is path `i`'s `k`-th wavelength.
    pub fn lanes(&self) -> usize {
        self.paths() * self.wavelengths
    }

    /// The conductor graph's batch: one vertex per lane, and with NEE a
    /// second, light-sampled one per lane after them (`lanes + j`).
    pub fn graph_width(&self) -> usize {
        self.lanes() * if self.nee { 2 } else { 1 }
    }

    fn camera_ray(&self, path: usize, rng: &mut Rng) -> Ray {
        let (w, h) = self.size;
        let px = path / self.spp;
        let (x, y) = (
            (px % w) as f64 + rng.uniform(),
            (px / w) as f64 + rng.uniform(),
        );
        if let Some(p) = &self.points {
            let [sx, sy] = p[path];
            return Ray::new(self.eye, self.forward + self.right * sx + self.up * sy);
        }
        let tan = self.tan_half;
        let sx = (2.0 * x / w as f64 - 1.0) * tan * w as f64 / h as f64;
        let sy = (1.0 - 2.0 * y / h as f64) * tan;
        Ray::new(self.eye, self.forward + self.right * sx + self.up * sy)
    }
}

/// A UV sphere of unit radius at the origin, `lat × lon` quads (the caps
/// fans), wound outward, face normals only.
pub fn uv_sphere(lat: usize, lon: usize) -> TriMesh {
    let mut positions = vec![Point3::new(0.0, 0.0, 1.0)];
    for i in 1..lat {
        let th = std::f64::consts::PI * i as f64 / lat as f64;
        for j in 0..lon {
            let ph = 2.0 * std::f64::consts::PI * j as f64 / lon as f64;
            positions.push(Point3::new(
                th.sin() * ph.cos(),
                th.sin() * ph.sin(),
                th.cos(),
            ));
        }
    }
    positions.push(Point3::new(0.0, 0.0, -1.0));
    let south = positions.len() as u32 - 1;
    let v = |i: usize, j: usize| (1 + (i - 1) * lon + j % lon) as u32;
    let mut idx = Vec::new();
    for j in 0..lon {
        idx.extend([0, v(1, j), v(1, j + 1)]);
        idx.extend([south, v(lat - 1, j + 1), v(lat - 1, j)]);
    }
    for i in 1..lat - 1 {
        for j in 0..lon {
            let (a, b, c, d) = (v(i, j), v(i + 1, j), v(i + 1, j + 1), v(i, j + 1));
            idx.extend([a, b, c, a, c, d]);
        }
    }
    TriMesh::new(positions, Vec::new(), &idx)
}

/// An axis-aligned box of half-extents `(hx, hy, hz)` about the origin,
/// twelve triangles wound outward.
pub fn cuboid(hx: f64, hy: f64, hz: f64) -> TriMesh {
    let positions = (0..8)
        .map(|i| {
            Point3::new(
                if i & 1 == 0 { -hx } else { hx },
                if i & 2 == 0 { -hy } else { hy },
                if i & 4 == 0 { -hz } else { hz },
            )
        })
        .collect();
    #[rustfmt::skip]
    let idx = [
        0, 2, 3, 0, 3, 1, // -z
        4, 5, 7, 4, 7, 6, // +z
        0, 1, 5, 0, 5, 4, // -y
        2, 6, 7, 2, 7, 3, // +y
        0, 4, 6, 0, 6, 2, // -x
        1, 3, 7, 1, 7, 5, // +x
    ];
    TriMesh::new(positions, Vec::new(), &idx)
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
    Env::default().radiance(d, lambda_nm)
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
    /// The BSDF pdf the current ray was drawn with; 0 for the camera ray.
    prev_pdf: f64,
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
    width: usize,
}

impl Kernels {
    /// Compiled C when `native` (and a compiler exists), the f64
    /// interpreter otherwise. `n` is the graph's batch,
    /// [`Scene::graph_width`] (which is [`Scene::lanes`] without NEE).
    pub fn new(n: usize, native: bool) -> Self {
        let g = Graph::new(n);
        Self {
            eval: g.eval(native),
            adjoint: g.adjoint(native),
            width: n,
        }
    }

    /// Sized for `scene`.
    pub fn for_scene(scene: &Scene, native: bool) -> Self {
        Self::new(scene.graph_width(), native)
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

/// Off-surface origin for a hit at `x = o + t d`: along the normal by
/// `ε = max(2⁻¹⁵ |x|∞, 2⁻¹⁷ (|o|∞ + t))`, the device's `offset_ray` in f64
/// (same constants, so both tiers push off by the same relative amount at
/// any scale). The first term is Wächter & Binder's 256 ulps of the position
/// ("A Fast and Robust Method for Avoiding Self-Intersection", Ray Tracing
/// Gems ch. 6); their per-coordinate integer form assumes a hit point
/// interpolated from the vertices, and `o + t d` has rounding that grows with
/// `|o| + t` (a floor at `z = 0` seen from far off has a `z` error of
/// `ulp(t)`, not `ulp(0)`), hence the second. Hits then need only `t > 0`.
pub fn offset(x: Point3, n: Vec3, o: Point3, t: f64) -> Point3 {
    let inf = |p: Point3| p.x.abs().max(p.y.abs()).max(p.z.abs());
    let e = (inf(x) / 32768.0).max((inf(o) + t) / 131072.0);
    x + n * e
}

/// The light-sampled half of a conductor vertex: direction and the
/// θ-free factor `s = cos_L · w_L / p_L` (visibility folded in).
#[derive(Debug, Clone, Copy)]
struct LightSample {
    dir: Vec3,
    s: f64,
}

/// Trace every path once. `replay = None` is the forward pass and returns
/// each path's `L`; `Some((l, dl))` replays against the forward pass's `L`
/// and the loss adjoint per path, accumulating into `grad`.
///
/// # NEE, MIS, and what the replay owes each vertex
///
/// With [`Scene::nee`], every surface vertex `j` (throughput `T_j` on
/// arrival) also draws `ω_L ~ p_L` ([`Env::sample`]), traces a shadow ray,
/// and collects
///
/// ```text
/// C_j = T_j · f_j(ω_L) cos_L · Le(ω_L) · V · w_L / p_L,   w_L = p_L² / (p_L² + p_B(ω_L)²)
/// ```
///
/// *before* the BSDF sample `ω_B ~ p_B` moves the path on; an escape after
/// a BSDF bounce collects `T Le(ω_B) w_B`, `w_B = p_B² / (p_B² + p_L(ω_B)²)`
/// (the camera ray's escape has no rival strategy: `w = 1`). The weights are
/// functions of the two detached pdfs only, so they are numbers to `∂θ`
/// exactly like `p` is (with a live sampler `p_B` moves with `α`, and the
/// weights still sum to one for every `ω`, so treating them as constants
/// leaves the estimator of `∫ ∂f` unbiased).
///
/// Every term of `L` is a product of BSDF values: the BSDF-sampled
/// `f_i(ω_B,i)` of every vertex before it and, for `C_j`, the light-sampled
/// `f_j(ω_L,j)` once. So `∂θ log` of a term is a sum over the `f` it
/// contains, and regrouping by `f` as in the plain case,
///
/// ```text
/// ∂L/∂θ = Σ_j [ ∂θ log f_j(ω_B,j) · (L − L_j⁺)  +  ∂θ log f_j(ω_L,j) · C_j ]
/// ```
///
/// where `L_j⁺` is what the path had collected *including* `C_j` — the
/// only terms carrying `f_j(ω_B,j)` are those collected after the BSDF
/// sample, and `C_j` is not one of them. The replay therefore, at a
/// conductor vertex and in this order: adds `C_j` to its running `L_j`,
/// seeds the light lane with `δf_L = δL · C_j / f_j(ω_L) = δL · T_j s Le`
/// (no division: `f_L` may be 0), then seeds the BSDF lane with
/// `δf_B = δL (L − L_j⁺) / f_j(ω_B)`. Both lanes are columns of the same
/// graph batch (light lanes after the BSDF lanes), so one adjoint
/// reduction takes both. Diffuse vertices are θ-free: their `C_j` lands in
/// `L_j` and nothing else.
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
    assert!(
        !(scene.nee && scene.splats.is_some()),
        "NEE with splats: shadow rays do not composite them"
    );
    let lanes = scene.lanes();
    let width = scene.graph_width();
    assert_eq!(k.width, width, "kernels built for another graph width");
    let env = scene.env;
    let mut paths: Vec<Path> = (0..n)
        .map(|i| {
            let mut rng = Rng::new(seed, scene.path_key(i));
            let lambda = wavelengths(rng.uniform(), nw);
            let ray = scene.camera_ray(i, &mut rng);
            Path {
                ray,
                throughput: [1.0; HERO],
                radiance: [0.0; HERO],
                lambda,
                rng,
                prev_pdf: 0.0,
                alive: true,
            }
        })
        .collect();
    // The column buffer: one row of COLUMNS + δf per graph lane, reused.
    let mut cols = vec![vec![0.0; width]; COLUMNS];
    let mut df = vec![0.0; width];
    let mut sampled = vec![(Vec3::zeros(), 0.0f64, 0.0f64); n]; // (wi, cos_i, pdf)
    let mut light: Vec<Option<LightSample>> = vec![None; n];
    let mut ball = vec![false; n];
    let mut stats = Stats {
        state_words: n
            * (std::mem::size_of::<Path>()
                + std::mem::size_of_val(&sampled[0])
                + std::mem::size_of::<Option<LightSample>>()
                + 1)
            / 8
            + (COLUMNS + 1) * width,
        vertices: 0,
    };
    k.eval.write(THETA, theta);
    k.adjoint.write(THETA, theta);
    let harmless = |cols: &mut [Vec<f64>], df: &mut [f64], j: usize| {
        for (c, v) in cols.iter_mut().zip([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) {
            c[j] = v;
        }
        df[j] = 0.0;
    };
    let lane = |cols: &mut [Vec<f64>], j: usize, li: [f64; 3], lo: [f64; 3], lam: f64| {
        let h = normalize([li[0] + lo[0], li[1] + lo[1], li[2] + lo[2]]);
        let oh = lo[0] * h[0] + lo[1] * h[1] + lo[2] * h[2];
        let bw = band_weights(lam);
        for (c, v) in cols
            .iter_mut()
            .zip([li[2], lo[2], h[2], oh, bw[0], bw[1], bw[2]])
        {
            c[j] = v;
        }
        h
    };

    for _ in 0..scene.depth {
        let mut any = false;
        for (i, p) in paths.iter_mut().enumerate() {
            // Inactive slots get a harmless vertex and no adjoint.
            for j in i * nw..(i + 1) * nw {
                harmless(&mut cols, &mut df, j);
                if scene.nee {
                    harmless(&mut cols, &mut df, lanes + j);
                }
            }
            ball[i] = false;
            light[i] = None;
            if !p.alive {
                continue;
            }
            let hit = scene.tlas.trace_closest_range(&p.ray, 0.0, f64::INFINITY);
            // Splats in front of the surface (or of the sky): emissive and
            // absorbing, composited front to back, θ-free.
            if let Some(cloud) = &scene.splats {
                let seg = splats::composite(
                    cloud,
                    &p.ray,
                    1e-6,
                    hit.as_ref().map_or(f64::INFINITY, |h| h.hit.t),
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
            let d: Vec3 = *p.ray.direction.as_ref();
            let Some(ih) = hit else {
                let w = if scene.nee && p.prev_pdf > 0.0 {
                    mis(p.prev_pdf, env.pdf(d))
                } else {
                    1.0
                };
                for k in 0..nw {
                    p.radiance[k] += p.throughput[k] * env.radiance(d, p.lambda[k]) * w;
                }
                p.alive = false;
                continue;
            };
            let mut nrm: Vec3 = *ih.hit.normal.as_ref();
            if nrm.dot(&d) > 0.0 {
                nrm = -nrm; // two-sided: shade the side the ray came from
            }
            let wo = -d;
            let (t, b) = basis(nrm);
            let origin = offset(ih.hit.point, nrm, p.ray.origin, ih.hit.t);
            // The light sample: drawn before the BSDF's, three uniforms
            // whether or not it lands, so the stream does not depend on it.
            let nee = scene.nee.then(|| {
                let dir = env.sample([p.rng.uniform(), p.rng.uniform(), p.rng.uniform()]);
                let cos = dir.dot(&nrm);
                let lit = cos > 0.0
                    && !scene
                        .tlas
                        .occluded_range(&Ray::new(origin, dir), 0.0, f64::INFINITY);
                (dir, cos, env.pdf(dir), lit)
            });
            if scene.surface(&ih) == Surface::Diffuse {
                // Lambertian: cosine sampling, weight = albedo. θ-free.
                let place = &scene.placements[ih.payload];
                let albedo = place.albedo
                    * if place.checker {
                        scene.albedo(ih.hit.point)
                    } else {
                        scene.floor_albedo
                    };
                if let Some((dir, cos, pl, true)) = nee {
                    let w = mis(pl, cos / std::f64::consts::PI);
                    for k in 0..nw {
                        p.radiance[k] += p.throughput[k] * albedo / std::f64::consts::PI
                            * cos
                            * env.radiance(dir, p.lambda[k])
                            * w
                            / pl;
                    }
                }
                let (u1, u2) = (p.rng.uniform(), p.rng.uniform());
                let (r, phi) = (u1.sqrt(), 2.0 * std::f64::consts::PI * u2);
                let cz = (1.0 - u1).max(0.0).sqrt();
                let wi = t * (r * phi.cos()) + b * (r * phi.sin()) + nrm * cz;
                for tp in &mut p.throughput[..nw] {
                    *tp *= albedo;
                }
                p.prev_pdf = cz / std::f64::consts::PI;
                p.ray = Ray::new(origin, wi);
                continue;
            }
            // The hero alone picks the direction. GGX's sampling does not
            // depend on λ, so every wavelength's pdf is the hero's and the
            // hero-wavelength MIS weight reduces to `f_k cos / p`.
            let lo = [wo.dot(&t), wo.dot(&b), wo.dot(&nrm)];
            let alpha = scene.frozen_alpha.unwrap_or(theta[0]);
            if let Some((dir, cos, pl, true)) = nee
                && lo[2] > 0.0
            {
                let li = [dir.dot(&t), dir.dot(&b), cos];
                let mut h = [0.0; 3];
                for w in 0..nw {
                    h = lane(&mut cols, lanes + i * nw + w, li, lo, p.lambda[w]);
                }
                let pb = ggx::pdf(alpha, lo[2], h[2]);
                light[i] = Some(LightSample {
                    dir,
                    s: cos * mis(pl, pb) / pl,
                });
                ball[i] = true;
                any = true;
            }
            let u = [p.rng.uniform(), p.rng.uniform()];
            let (li, pdf) = ggx::sample(alpha, lo, u);
            if li[2] <= 0.0 || lo[2] <= 0.0 || !pdf.is_finite() || pdf <= 0.0 {
                // Dead after this vertex; its light sample still counts.
                sampled[i] = (Vec3::zeros(), 0.0, 0.0);
                p.alive = false;
                continue;
            }
            for w in 0..nw {
                lane(&mut cols, i * nw + w, li, lo, p.lambda[w]);
            }
            sampled[i] = (t * li[0] + b * li[1] + nrm * li[2], li[2], pdf);
            p.prev_pdf = pdf;
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
                if let Some(ls) = light[i] {
                    let le = env.radiance(ls.dir, p.lambda[w]);
                    let g = p.throughput[w] * ls.s * le;
                    if let Some((_, dl)) = replay {
                        df[lanes + j] = dl[j] * g;
                        need_adjoint |= df[lanes + j] != 0.0;
                    }
                    p.radiance[w] += g * f[lanes + j];
                }
                if pdf <= 0.0 {
                    continue;
                }
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
