//! Phase 2d: primary-visibility gradients with respect to the pose of
//! instances, by sampling the edges that move.
//!
//! # What is differentiated
//!
//! A pose `φ` (joint angles, say) places some instances ([`Rig`]); `θ`
//! ([`crate::prb`]) is the conductor's finish. A pixel is the box integral
//! `I_p = ∫_p f(x) dx` of the radiance `f` arriving through camera-plane
//! point `x`. Moving an instance moves the curves across which `f` jumps —
//! its silhouette against what is behind it, and its own creases (flat
//! shaded: every edge between faces that are not coplanar) — and Reynolds'
//! transport theorem gives the part of `∂I_p/∂φ` that PRB cannot see:
//!
//! ```text
//! ∂I_p/∂φ ⊇ ∫_{E ∩ p} (f(s − εn) − f(s + εn)) · v_n(s) dℓ(s)
//! ```
//!
//! over the screen-space edges `E` of the moving triangles, `n` the edge's
//! outward normal (away from its own triangle) and `v_n = n · ∂π(X(s))/∂φ`
//! the normal speed of the edge point `X(s)` (a material point of the mesh,
//! carried by its instance; `π` the pinhole). A loss `ℒ(I)` takes
//! `Σ_p ∂ℒ/∂I_p` of it.
//!
//! # The estimator (Li et al. 2018's edge sampling, primary rays)
//!
//! Every triangle edge of every moving instance is projected and clipped to
//! the frame; an edge sample picks a point with density ∝ screen length ×
//! importance (stratified, `n` per call) — importance 1 on silhouettes and
//! boundaries, a crease between front faces by its dihedral angle
//! ([`crease_weight`]), [`BACK_WEIGHT`] behind — and weighs
//! `(Σ ℓ w / (w_e n)) · v_n` times the difference of two full paths traced through
//! `s ∓ εn` ([`Scene::set_points`]; the pair shares its generator, so
//! the difference is common random numbers). Each triangle owns its edges:
//! the inner ray must hit *that* triangle or one coplanar with it on the
//! same face — CAD tessellations are full of slivers thinner than `ε` —
//! (otherwise the edge is hidden at `s` and the sample is 0). A crease
//! between two front faces is sampled from one side only, the face that
//! meets the eye more squarely (phase 2e: 2d sampled it from both and
//! halved where the outer ray landed next to `X`, which lost half the jump
//! whenever the grazing side's ownership test failed — invisible in a mask,
//! O(1) when shaded). Edges shared by two coplanar
//! triangles are never candidates: `f` is continuous across them. No mesh
//! adjacency is assumed beyond that filter, so unwelded and T-junctioned
//! tessellations stay correct.
//!
//! What it leaves out, by design for this phase: the interior derivative
//! of `f` at fixed `x` (shading at the hit moving with the pose) and every
//! secondary discontinuity (shadows and reflections moving). Against the
//! sky at depth 1 — where `f` is the instance's mask — the estimator is the
//! whole gradient, and that is the test.

use irys_core::{Point3, Ray, Transform, Vec3};
use irys_light::spectrum::hero_weight;

use crate::prb::{HERO, Rng, Scene, wavelengths};

/// The half-width of an edge sample, in pixels: the two rays leave the
/// camera this far either side of the edge. Small against a pixel, large
/// against the device's f32 camera plane.
pub const EPS_PX: f64 = 1e-3;

/// A pose: which placements it moves, and where it puts them.
pub struct Rig<'a> {
    /// Payloads of the moving placements.
    pub moving: Vec<usize>,
    /// `φ` → object → world of each of `moving`, in order.
    pub place: Box<dyn Fn(&[f64]) -> Vec<Transform> + 'a>,
}

/// The step `Rig::place` is differenced at for the edges' velocities. The
/// placement is a closed form in `φ`, so central differences of it are
/// exact to ~1e-10 — this is a derivative of the rig, not of the image.
pub const RIG_H: f64 = 1e-6;

impl Rig<'_> {
    /// Put `scene`'s moving instances at `phi`.
    pub fn pose(&self, scene: &mut Scene, phi: &[f64]) {
        let t = (self.place)(phi);
        assert_eq!(
            t.len(),
            self.moving.len(),
            "one transform per moving placement"
        );
        let moves: Vec<(usize, Transform)> = self.moving.iter().copied().zip(t).collect();
        scene.place(&moves);
    }
}

/// A pinhole: eye, forward, right, up, `tan` of the half field of view.
#[derive(Debug, Clone, Copy)]
pub struct Pinhole {
    /// Eye.
    pub eye: Point3,
    /// Unit forward.
    pub forward: Vec3,
    /// Unit right.
    pub right: Vec3,
    /// Unit up.
    pub up: Vec3,
    /// `tan` of the half field of view.
    pub tan_half: f64,
}

impl Pinhole {
    /// The scene's.
    pub fn of(scene: &Scene) -> Self {
        let (eye, forward, right, up, tan_half) = scene.camera();
        Self {
            eye,
            forward,
            right,
            up,
            tan_half,
        }
    }

    /// [`Scene::project`] through this pinhole, without the Jacobian.
    pub fn project(&self, size: (usize, usize), p: Point3) -> Option<[f64; 2]> {
        let d = p - self.eye;
        let z = d.dot(&self.forward);
        if z <= 0.0 {
            return None;
        }
        let (w, h) = (size.0 as f64, size.1 as f64);
        let kx = w / (2.0 * self.tan_half * w / h);
        let ky = h / (2.0 * self.tan_half);
        Some([
            d.dot(&self.right) / z * kx + w / 2.0,
            h / 2.0 - d.dot(&self.up) / z * ky,
        ])
    }
}

/// The camera as a parameter (phase 2e): `c = (t, ω, δ)` — the eye moved
/// by `t` (world), the frame turned about the eye by `R_z(ω_z) R_y(ω_y)
/// R_x(ω_x)` (world axes), the half field of view widened by `δ` radians —
/// all about a base pinhole, so `c = 0` is the base.
#[derive(Debug, Clone, Copy)]
pub struct CameraRig {
    /// The pinhole at `c = 0`.
    pub base: Pinhole,
}

/// Camera parameters: three of translation, three of rotation, one of fov.
pub const CAMERA_PARAMS: usize = 7;

impl CameraRig {
    /// Rigged about `scene`'s current camera.
    pub fn new(scene: &Scene) -> Self {
        Self {
            base: Pinhole::of(scene),
        }
    }

    /// The pinhole at `c`.
    pub fn pinhole(&self, c: &[f64]) -> Pinhole {
        assert_eq!(c.len(), CAMERA_PARAMS);
        let o = Point3::origin();
        let r = about(o, Vec3::new(0.0, 0.0, 1.0), c[5])
            .then(&about(o, Vec3::new(0.0, 1.0, 0.0), c[4]))
            .then(&about(o, Vec3::new(1.0, 0.0, 0.0), c[3]));
        let b = &self.base;
        Pinhole {
            eye: b.eye + Vec3::new(c[0], c[1], c[2]),
            forward: r.apply_vec(&b.forward),
            right: r.apply_vec(&b.right),
            up: r.apply_vec(&b.up),
            tan_half: (b.tan_half.atan() + c[6]).tan(),
        }
    }

    /// Put `scene`'s camera at `c`.
    pub fn pose(&self, scene: &mut Scene, c: &[f64]) {
        let p = self.pinhole(c);
        scene.set_camera(p.eye, p.forward, p.right, p.up, p.tan_half);
    }
}

/// No neighbour across an edge (a boundary, or more than two triangles
/// share it after welding).
pub(crate) const NONE: u32 = u32::MAX;
/// Neighbour marker for an edge whose neighbour is coplanar: never a
/// candidate, `f` is continuous across it.
pub(crate) const FLAT: u32 = u32::MAX - 1;

/// Importance of a crease between two front faces whose normals meet at
/// `cos`: ∝ how much the shading can jump, from 0.02 (a fine tessellation
/// of a cylinder) to 1 (a right angle) — silhouettes and boundaries are 1.
pub(crate) fn crease_weight(cos: f64) -> f64 {
    (4.0 * (1.0 - cos)).clamp(0.02, 1.0)
}

/// What faces the camera is a hint, not a promise (two-sided shading, a
/// mesh wound inconsistently): back-only edges keep this much weight, so
/// the estimator stays unbiased whatever the winding.
pub(crate) const BACK_WEIGHT: f64 = 0.02;

/// Per bottom-level tree, every triangle edge's neighbour across it (by
/// welded position) and their dihedral cosine: which edges can never be a
/// discontinuity (shared with one coplanar triangle), and how to weigh the
/// rest when sampling.
pub struct Edges {
    pub(crate) nbr: Vec<Vec<[u32; 3]>>,
    pub(crate) cos: Vec<Vec<[f32; 3]>>,
    /// Object-space face normals (unnormalised: the winding's).
    pub(crate) normal: Vec<Vec<Vec3>>,
}

impl Edges {
    /// Weld every tree of `scene` and find each edge's neighbour.
    pub fn new(scene: &Scene) -> Self {
        let (mut nbrs, mut coss, mut normals) = (Vec::new(), Vec::new(), Vec::new());
        for b in scene.blas() {
            let m = b.geometry().mesh();
            let (pos, tris) = (m.positions(), m.triangles());
            let mut bb = irys_core::Aabb::empty();
            for p in pos {
                bb.include_point(p);
            }
            let tol = if pos.is_empty() {
                1.0
            } else {
                ((bb.max - bb.min).norm() * 1e-7).max(1e-12)
            };
            let key = |i: u32| {
                let p = pos[i as usize];
                [p.x, p.y, p.z].map(|c| (c / tol).round() as i64)
            };
            let normal: Vec<Vec3> = tris
                .iter()
                .map(|t| {
                    let [a, b, c] = t.map(|i| pos[i as usize]);
                    (b - a).cross(&(c - a))
                })
                .collect();
            let mut map: std::collections::HashMap<_, Vec<(usize, usize)>> =
                std::collections::HashMap::new();
            for (ti, t) in tris.iter().enumerate() {
                for e in 0..3 {
                    let (a, b) = (key(t[e]), key(t[(e + 1) % 3]));
                    map.entry(if a < b { (a, b) } else { (b, a) })
                        .or_default()
                        .push((ti, e));
                }
            }
            let mut nbr = vec![[NONE; 3]; tris.len()];
            let mut cos = vec![[0.0f32; 3]; tris.len()];
            for v in map.values() {
                if let [(t0, e0), (t1, e1)] = v[..] {
                    let (n0, n1) = (normal[t0], normal[t1]);
                    let (l0, l1) = (n0.norm(), n1.norm());
                    if l0 > 0.0 && l1 > 0.0 {
                        let c = n0.dot(&n1) / (l0 * l1);
                        let (a, b) = if c > 1.0 - 1e-10 {
                            (FLAT, FLAT)
                        } else {
                            (t1 as u32, t0 as u32)
                        };
                        nbr[t0][e0] = a;
                        nbr[t1][e1] = b;
                        cos[t0][e0] = c as f32;
                        cos[t1][e1] = c as f32;
                    }
                }
            }
            nbrs.push(nbr);
            coss.push(cos);
            normals.push(normal);
        }
        Self {
            nbr: nbrs,
            cos: coss,
            normal: normals,
        }
    }

    /// Edges that are candidates, over every tree.
    pub fn candidates(&self) -> usize {
        self.nbr
            .iter()
            .flatten()
            .map(|s| s.iter().filter(|x| **x != FLAT).count())
            .sum()
    }
}

/// A hidden edge's share of its importance under [`Visibility`].
pub const HIDDEN_WEIGHT: f64 = 0.01;

/// What an ID buffer says about an edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seen {
    /// Behind the visible surface at every probe.
    Hidden,
    /// Visible somewhere, next to the sky or another instance.
    Border,
    /// Visible, surrounded by its own instance.
    Inside,
}

/// The primary hit of every pixel centre: which instance (or the sky) and
/// how far along the view axis. One ray a pixel; what [`EdgeSamples`]
/// consults to stop spending samples on edges nobody can see.
pub struct Visibility {
    size: (usize, usize),
    id: Vec<u32>,
    depth: Vec<f64>,
}

/// The ID buffer's word for the sky.
pub const SKY: u32 = u32::MAX;

impl Visibility {
    /// Trace one ray through every pixel centre of `scene`.
    pub fn new(scene: &Scene) -> Self {
        let (w, h) = scene.size;
        let (eye, fwd, right, up, _) = scene.camera();
        let mut id = vec![SKY; w * h];
        let mut depth = vec![f64::INFINITY; w * h];
        for y in 0..h {
            for x in 0..w {
                let [sx, sy] = scene.screen(x as f64 + 0.5, y as f64 + 0.5);
                let r = Ray::new(eye, fwd + right * sx + up * sy);
                if let Some(hit) = scene.tlas().trace_closest_range(&r, 0.0, f64::INFINITY) {
                    id[y * w + x] = hit.payload as u32;
                    depth[y * w + x] = (hit.hit.point - eye).dot(&fwd);
                }
            }
        }
        Self::from_parts(scene.size, id, depth)
    }

    /// The instance (or [`SKY`]) at every pixel centre, row-major.
    pub fn ids(&self) -> &[u32] {
        &self.id
    }

    /// A buffer from its columns (the device's, say).
    pub fn from_parts(size: (usize, usize), id: Vec<u32>, depth: Vec<f64>) -> Self {
        assert_eq!(id.len(), size.0 * size.1);
        assert_eq!(depth.len(), id.len());
        Self { size, id, depth }
    }

    /// [`Visibility::classify`] per probe interval of `[t0, t1]`, merged
    /// into runs of one class: `(t0, t1, class)` covering it exactly.
    #[allow(clippy::too_many_arguments)]
    pub fn runs(
        &self,
        k: usize,
        pa: [f64; 2],
        pb: [f64; 2],
        za: f64,
        zb: f64,
        t0: f64,
        t1: f64,
    ) -> Vec<(f64, f64, Seen)> {
        let len = (pb[0] - pa[0]).hypot(pb[1] - pa[1]) * (t1 - t0);
        let m = ((len).ceil() as usize).clamp(1, 64);
        let mut out: Vec<(f64, f64, Seen)> = Vec::new();
        for i in 0..m {
            let (a, b) = (
                t0 + (t1 - t0) * i as f64 / m as f64,
                if i + 1 == m {
                    t1
                } else {
                    t0 + (t1 - t0) * (i + 1) as f64 / m as f64
                },
            );
            let c = self.classify(k, pa, pb, za, zb, a, b);
            match out.last_mut() {
                Some(l) if l.2 == c => l.1 = b,
                _ => out.push((a, b, c)),
            }
        }
        out
    }

    /// Classify the screen segment `pa → pb` (view depths `za`, `zb` at its
    /// ends) of an edge of instance `k`, over its clipped part `[t0, t1]`:
    /// probes every half pixel; at each, the edge is visible if it is no
    /// deeper than the farthest surface in the surrounding 3 × 3 pixels
    /// (conservative: a silhouette always has sky or a farther surface
    /// beside it).
    #[allow(clippy::too_many_arguments)]
    pub fn classify(
        &self,
        k: usize,
        pa: [f64; 2],
        pb: [f64; 2],
        za: f64,
        zb: f64,
        t0: f64,
        t1: f64,
    ) -> Seen {
        let (w, h) = (self.size.0 as i64, self.size.1 as i64);
        let len = (pb[0] - pa[0]).hypot(pb[1] - pa[1]) * (t1 - t0);
        let m = ((2.0 * len).ceil() as usize).clamp(1, 64);
        let mut seen = Seen::Hidden;
        for i in 0..m {
            let t = t0 + (t1 - t0) * (i as f64 + 0.5) / m as f64;
            let p = [pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1])];
            // 1/z is linear in screen space.
            let z = 1.0 / ((1.0 - t) / za + t / zb);
            let (cx, cy) = (p[0].floor() as i64, p[1].floor() as i64);
            let (mut far, mut border) = (0.0f64, false);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (x, y) = (cx + dx, cy + dy);
                    if x < 0 || y < 0 || x >= w || y >= h {
                        far = f64::INFINITY;
                        border = true;
                        continue;
                    }
                    let j = (y * w + x) as usize;
                    far = far.max(self.depth[j]);
                    border |= self.id[j] != k as u32;
                }
            }
            if z <= far * (1.0 + 2e-3) {
                if border {
                    return Seen::Border;
                }
                seen = Seen::Inside;
            }
        }
        seen
    }
}

/// One frame's edge samples: the ray pairs to trace and what each pair is
/// worth per pose parameter.
pub struct EdgeSamples {
    points: Vec<[f64; 2]>,
    pixel: Vec<usize>,
    /// Per sample, per parameter: `v_n · Σℓw / (w_e n)`.
    weight: Vec<Vec<f64>>,
    /// Screen length of every candidate edge in the frame, pixels.
    pub total_len: f64,
    /// Of it, the length of silhouettes and boundaries (importance 1).
    pub silhouette_len: f64,
    /// Candidate edge pieces inside the frame.
    pub segments: usize,
    /// Samples whose edge was visible at the sample (inner ray hit its own
    /// triangle).
    pub visible: usize,
    /// Of those, samples whose outer ray lands on the same instance next to
    /// the edge point (creases; a diagnostic).
    pub halved: usize,
}

struct Seg {
    mi: usize,
    k: usize,
    tri: usize,
    e: usize,
    pa: [f64; 2],
    pb: [f64; 2],
    pc: [f64; 2],
    t0: f64,
    t1: f64,
    /// Sampling importance.
    w: f64,
}

/// The part of the segment `a → b` inside `[0, w] × [0, h]`, as parameters.
fn clip(a: [f64; 2], b: [f64; 2], w: f64, h: f64) -> Option<(f64, f64)> {
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    let d = [b[0] - a[0], b[1] - a[1]];
    for (p, q) in [
        (-d[0], a[0]),
        (d[0], w - a[0]),
        (-d[1], a[1]),
        (d[1], h - a[1]),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
        }
    }
    (t0 < t1).then_some((t0, t1))
}

impl EdgeSamples {
    /// Samples drawn elsewhere (the device): pixel and weights per sample,
    /// the points left where the device wrote them.
    pub(crate) fn from_parts(pixel: Vec<usize>, weight: Vec<Vec<f64>>, visible: usize) -> Self {
        Self {
            points: Vec::new(),
            pixel,
            weight,
            total_len: 0.0,
            silhouette_len: 0.0,
            segments: 0,
            visible,
            halved: 0,
        }
    }

    /// Draw `n` edge samples of `scene` as posed at `phi` (the caller has
    /// [`Rig::pose`]d it) with generator `seed`.
    pub fn new(scene: &Scene, edges: &Edges, rig: &Rig, phi: &[f64], n: usize, seed: u64) -> Self {
        Self::with_visibility(scene, edges, rig, phi, n, seed, None)
    }

    /// [`EdgeSamples::new`] (`vis = None`: 2d's importance), or phase 2e's
    /// importance from an ID buffer ([`Visibility`]): a silhouette is
    /// sampled from its front face only (the back face's side scores 0),
    /// and an edge the buffer shows hidden everywhere along it keeps
    /// [`HIDDEN_WEIGHT`] of its importance — never 0, so a wrong guess
    /// costs variance, not bias.
    pub fn with_visibility(
        scene: &Scene,
        edges: &Edges,
        rig: &Rig,
        phi: &[f64],
        n: usize,
        seed: u64,
        vis: Option<&Visibility>,
    ) -> Self {
        Self::for_rig(scene, edges, rig, phi, n, seed, vis.is_some(), vis)
    }

    /// 2e's front-only silhouettes without an ID buffer (what the device
    /// draws, [`crate::gpu::GpuEdges`]), or 2d's importance.
    pub fn with_policy(
        scene: &Scene,
        edges: &Edges,
        rig: &Rig,
        phi: &[f64],
        n: usize,
        seed: u64,
        front_only: bool,
    ) -> Self {
        Self::for_rig(scene, edges, rig, phi, n, seed, front_only, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn for_rig(
        scene: &Scene,
        edges: &Edges,
        rig: &Rig,
        phi: &[f64],
        n: usize,
        seed: u64,
        front_only: bool,
        vis: Option<&Visibility>,
    ) -> Self {
        let np = phi.len();
        // ± RIG_H placements per parameter, for the edges' velocities.
        let shifted: Vec<[Vec<Transform>; 2]> = (0..np)
            .map(|j| {
                let mut p = phi.to_vec();
                p[j] += RIG_H;
                let plus = (rig.place)(&p);
                p[j] -= 2.0 * RIG_H;
                [plus, (rig.place)(&p)]
            })
            .collect();
        // A material point carried by its instance: `J_π · ∂X/∂φ`.
        let vel = |mi: usize, xl: Point3, _x: Point3, jac: [Vec3; 2]| {
            shifted
                .iter()
                .map(|[p, q]| {
                    let dx = (p[mi].apply_point(&xl) - q[mi].apply_point(&xl)) / (2.0 * RIG_H);
                    [jac[0].dot(&dx), jac[1].dot(&dx)]
                })
                .collect::<Vec<_>>()
        };
        Self::build(
            scene,
            edges,
            &rig.moving,
            &vel,
            np,
            n,
            seed,
            front_only,
            vis,
        )
    }

    /// Edge samples for the camera's parameters ([`CameraRig`] at `c`; the
    /// scene is set there): every instance's edges are candidates, and a
    /// world point's screen velocity is the pinhole's derivative.
    pub fn for_camera(
        scene: &Scene,
        edges: &Edges,
        cam: &CameraRig,
        c: &[f64],
        n: usize,
        seed: u64,
    ) -> Self {
        let shifted: Vec<[Pinhole; 2]> = (0..c.len())
            .map(|j| {
                let mut p = c.to_vec();
                p[j] += RIG_H;
                let a = cam.pinhole(&p);
                p[j] -= 2.0 * RIG_H;
                [a, cam.pinhole(&p)]
            })
            .collect();
        let size = scene.size;
        let vel = |_mi: usize, _xl: Point3, x: Point3, _jac: [Vec3; 2]| {
            shifted
                .iter()
                .map(|[a, b]| match (a.project(size, x), b.project(size, x)) {
                    (Some(p), Some(q)) => {
                        [(p[0] - q[0]) / (2.0 * RIG_H), (p[1] - q[1]) / (2.0 * RIG_H)]
                    }
                    _ => [0.0, 0.0],
                })
                .collect::<Vec<_>>()
        };
        let all: Vec<usize> = (0..scene.placements().len()).collect();
        Self::build(scene, edges, &all, &vel, c.len(), n, seed, true, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn build(
        scene: &Scene,
        edges: &Edges,
        moving: &[usize],
        vel: &dyn Fn(usize, Point3, Point3, [Vec3; 2]) -> Vec<[f64; 2]>,
        np: usize,
        n: usize,
        seed: u64,
        front_only: bool,
        vis: Option<&Visibility>,
    ) -> Self {
        let (w, h) = (scene.size.0 as f64, scene.size.1 as f64);
        let mut segs = Vec::new();
        let mut cum = Vec::new();
        let (mut total, mut wtotal, mut sil) = (0.0, 0.0, 0.0);
        let (eye, fwd, right, up, _) = scene.camera();
        for (mi, &k) in moving.iter().enumerate() {
            let place = &scene.placements()[k];
            let m = scene.blas()[place.blas].geometry().mesh();
            let pos = m.positions();
            let (nbr, cos) = (&edges.nbr[place.blas], &edges.cos[place.blas]);
            let normal = &edges.normal[place.blas];
            let tris = m.triangles();
            // How squarely triangle `tri` faces the eye: > 0 is front.
            let facing = |tri: usize| {
                let x = place.to_world.apply_point(&pos[tris[tri][0] as usize]);
                let n = place.to_world.apply_normal(&normal[tri]);
                n.dot(&(eye - x)) / (n.norm() * (eye - x).norm()).max(1e-300)
            };
            let front = |tri: usize| facing(tri) > 0.0;
            for (tri, t) in tris.iter().enumerate() {
                let wp: Vec<Point3> = t
                    .iter()
                    .map(|&i| place.to_world.apply_point(&pos[i as usize]))
                    .collect();
                let pz: Vec<Option<([f64; 2], f64)>> = wp
                    .iter()
                    .map(|&x| scene.project(x).map(|p| (p.0, (x - eye).dot(&fwd))))
                    .collect();
                if pz.iter().all(Option::is_none) {
                    continue;
                }
                let f = front(tri);
                for e in 0..3 {
                    let imp = match nbr[tri][e] {
                        FLAT => continue,
                        NONE if f => 1.0,
                        NONE => BACK_WEIGHT,
                        n => match (f, front(n as usize)) {
                            // A crease between front faces has one owner:
                            // the squarer face, so its inner ray is the
                            // robust one, and the jump is counted once
                            // whatever the grazing side's ray lands on.
                            (true, true) => {
                                let (a, b) = (facing(tri), facing(n as usize));
                                if a < b || (a == b && tri > n as usize) {
                                    continue;
                                }
                                crease_weight(cos[tri][e] as f64)
                            }
                            (false, false) => BACK_WEIGHT,
                            // A silhouette is the front face's: the back
                            // face's inner ray lands on the front one and
                            // scores 0 (2d gives it weight 1 — half of
                            // every silhouette sample wasted).
                            (true, false) => 1.0,
                            (false, true) if front_only => BACK_WEIGHT,
                            (false, true) => 1.0,
                        },
                    };
                    // The edge's ends must be in front of the eye; the
                    // third corner only says which side is the triangle's,
                    // so behind the eye a point just inside the triangle
                    // next to the edge says it instead (2d dropped every
                    // triangle with a corner behind the eye).
                    let (Some((pa, za)), Some((pb, zb))) = (pz[e], pz[(e + 1) % 3]) else {
                        continue;
                    };
                    let pc = match pz[(e + 2) % 3] {
                        Some((p, _)) => p,
                        None => {
                            let m = wp[e] + (wp[(e + 1) % 3] - wp[e]) * 0.5;
                            match scene.project(m + (wp[(e + 2) % 3] - m) * 1e-3) {
                                Some((p, _)) => p,
                                None => continue,
                            }
                        }
                    };
                    let Some((t0, t1)) = clip(pa, pb, w, h) else {
                        continue;
                    };
                    let len = (pb[0] - pa[0]).hypot(pb[1] - pa[1]) * (t1 - t0);
                    if len <= 0.0 {
                        continue;
                    }
                    let runs = match vis {
                        None => vec![(t0, t1, imp)],
                        Some(v) => {
                            v.runs(k, pa, pb, za, zb, t0, t1)
                                .into_iter()
                                .map(|(a, b, seen)| {
                                    // Visible creases stay: shaded, they
                                    // jump too, inside the instance's own
                                    // image.
                                    let w = match seen {
                                        Seen::Hidden => imp * HIDDEN_WEIGHT,
                                        Seen::Border | Seen::Inside => imp,
                                    };
                                    (a, b, w)
                                })
                                .collect()
                        }
                    };
                    let full = (pb[0] - pa[0]).hypot(pb[1] - pa[1]);
                    for (t0, t1, imp) in runs {
                        let len = full * (t1 - t0);
                        if len <= 0.0 {
                            continue;
                        }
                        total += len;
                        if imp == 1.0 {
                            sil += len;
                        }
                        wtotal += len * imp;
                        cum.push(wtotal);
                        segs.push(Seg {
                            mi,
                            k,
                            tri,
                            e,
                            pa,
                            pb,
                            pc,
                            t0,
                            t1,
                            w: imp,
                        });
                    }
                }
            }
        }
        let mut out = Self {
            points: Vec::with_capacity(2 * n),
            pixel: Vec::with_capacity(n),
            weight: Vec::with_capacity(n),
            total_len: total,
            silhouette_len: sil,
            segments: segs.len(),
            visible: 0,
            halved: 0,
        };
        for i in 0..n {
            let put = |o: &mut Self, s: [f64; 2], nrm: [f64; 2], wt: Vec<f64>| {
                let at = |d: f64| scene.screen(s[0] + d * nrm[0], s[1] + d * nrm[1]);
                o.points.push(at(-EPS_PX));
                o.points.push(at(EPS_PX));
                let x = (s[0].floor().clamp(0.0, w - 1.0)) as usize;
                let y = (s[1].floor().clamp(0.0, h - 1.0)) as usize;
                o.pixel.push(y * scene.size.0 + x);
                o.weight.push(wt);
            };
            // One generator per sample (phase 2e: the device draws the
            // same `u` in parallel).
            let u =
                (i as f64 + Rng::new(seed ^ 0x6564_6765, i as u64).uniform()) / n as f64 * wtotal;
            if segs.is_empty() {
                put(&mut out, [0.0, 0.0], [1.0, 0.0], vec![0.0; np]);
                continue;
            }
            let si = cum.partition_point(|&c| c <= u).min(segs.len() - 1);
            let seg = &segs[si];
            let start = if si == 0 { 0.0 } else { cum[si - 1] };
            let frac = ((u - start) / (cum[si] - start)).clamp(0.0, 1.0);
            let t = seg.t0 + frac * (seg.t1 - seg.t0);
            let d = [seg.pb[0] - seg.pa[0], seg.pb[1] - seg.pa[1]];
            let s = [seg.pa[0] + t * d[0], seg.pa[1] + t * d[1]];
            let dl = d[0].hypot(d[1]);
            let mut nrm = [d[1] / dl, -d[0] / dl];
            if nrm[0] * (seg.pc[0] - seg.pa[0]) + nrm[1] * (seg.pc[1] - seg.pa[1]) > 0.0 {
                nrm = [-nrm[0], -nrm[1]];
            }
            // The material point under `s`: the edge line's closest approach
            // to the camera ray through `s`.
            let place = &scene.placements()[seg.k];
            let m = scene.blas()[place.blas].geometry().mesh();
            let tri = m.triangles()[seg.tri];
            let (la, lb) = (
                m.positions()[tri[seg.e] as usize],
                m.positions()[tri[(seg.e + 1) % 3] as usize],
            );
            let (a, b) = (
                place.to_world.apply_point(&la),
                place.to_world.apply_point(&lb),
            );
            let [sx, sy] = scene.screen(s[0], s[1]);
            let r = fwd + right * sx + up * sy;
            let e = b - a;
            let w0 = eye - a;
            let (aa, bb, cc, dd, ee) = (r.dot(&r), r.dot(&e), e.dot(&e), r.dot(&w0), e.dot(&w0));
            let den = aa * cc - bb * bb;
            let v = if den.abs() > 0.0 {
                ((aa * ee - bb * dd) / den).clamp(0.0, 1.0)
            } else {
                t
            };
            let x = a + e * v;
            let xl = la + (lb - la) * v;
            let jac = scene.project(x).map(|p| p.1);
            // Visibility at `s`: the inner ray must land on this triangle.
            let hit = |q: [f64; 2]| {
                let r = fwd + right * q[0] + up * q[1];
                scene
                    .tlas()
                    .trace_closest_range(&Ray::new(eye, r), 0.0, f64::INFINITY)
            };
            let inner = scene.screen(s[0] - EPS_PX * nrm[0], s[1] - EPS_PX * nrm[1]);
            let outer = scene.screen(s[0] + EPS_PX * nrm[0], s[1] + EPS_PX * nrm[1]);
            // A sliver thinner than ε puts the inner point past its own
            // triangle onto a coplanar neighbour of the same face: that is
            // still this side of the edge. A triangle folded behind (the
            // back half of a silhouette) is not coplanar, and is refused.
            let nt = edges.normal[place.blas][seg.tri];
            let own = hit(inner).is_some_and(|ih| {
                if ih.payload != seg.k {
                    return false;
                }
                let p = ih.hit.prim as usize;
                if p == seg.tri {
                    return true;
                }
                let Some(nh) = edges.normal[place.blas].get(p) else {
                    return false;
                };
                let (ln, lh) = (nt.norm(), nh.norm());
                let local = place
                    .to_world
                    .inverse()
                    .map(|t| t.apply_point(&ih.hit.point));
                ln > 0.0
                    && lh > 0.0
                    && nt.dot(nh) / (ln * lh) > 1.0 - 1e-6
                    && local.is_some_and(|q| {
                        (nt / ln).dot(&(q - la)).abs() < 1e-4 * (lb - la).norm().max(1e-9)
                    })
            });
            let (Some(jac), true) = (jac, own) else {
                put(&mut out, s, nrm, vec![0.0; np]);
                continue;
            };
            out.visible += 1;
            let crease = hit(outer).is_some_and(|oh| {
                oh.payload == seg.k && (oh.hit.point - x).norm() < 1e-4 * (x - eye).norm()
            });
            if crease {
                out.halved += 1;
            }
            let wt = vel(seg.mi, xl, x, jac)
                .into_iter()
                .map(|v| (v[0] * nrm[0] + v[1] * nrm[1]) * wtotal / (seg.w * n as f64))
                .collect();
            put(&mut out, s, nrm, wt);
        }
        out
    }

    /// Per sample, per parameter, what its pair's difference is worth.
    pub fn weights(&self) -> &[Vec<f64>] {
        &self.weight
    }

    /// The ray pairs, inner then outer, as camera-plane points.
    pub fn points(&self) -> &[[f64; 2]] {
        &self.points
    }

    /// Samples drawn.
    pub fn len(&self) -> usize {
        self.pixel.len()
    }

    /// Whether none were drawn.
    pub fn is_empty(&self) -> bool {
        self.pixel.is_empty()
    }

    /// `∂ℒ/∂φ`'s edge term from the pairs' traced `L` (per lane, as the
    /// points frame lays them out, generator `seed`) and the loss adjoint
    /// per pixel channel.
    pub fn gradient(&self, scene: &Scene, seed: u64, l: &[f64], d_img: &[f64]) -> Vec<f64> {
        let nw = scene.wavelengths;
        assert_eq!(l.len(), 2 * self.len() * nw, "L of the points frame");
        let np = self.weight.first().map_or(0, Vec::len);
        let mut g = vec![0.0; np];
        for (i, (&px, wt)) in self.pixel.iter().zip(&self.weight).enumerate() {
            if wt.iter().all(|&x| x == 0.0) {
                continue;
            }
            let lam: [f64; HERO] = wavelengths(Rng::new(seed, i as u64).uniform(), nw);
            let mut s = 0.0;
            for k in 0..nw {
                let d = l[2 * i * nw + k] - l[(2 * i + 1) * nw + k];
                let hw = hero_weight(lam[k]);
                for c in 0..3 {
                    s += d * hw[c] as f64 / nw as f64 * d_img[px * 3 + c];
                }
            }
            for (gj, wj) in g.iter_mut().zip(wt) {
                *gj += wj * s;
            }
        }
        g
    }
}

/// The edge term in one call: sample, trace the pairs with `trace` (which
/// gets `scene` in point mode and returns `L` per lane — [`crate::prb::forward`]
/// or the GPU's), weigh. `scene` is left as it came.
pub fn boundary(
    scene: &mut Scene,
    edges: &Edges,
    rig: &Rig,
    phi: &[f64],
    d_img: &[f64],
    n: usize,
    seed: u64,
    trace: impl FnOnce(&Scene) -> Vec<f64>,
) -> (Vec<f64>, EdgeSamples) {
    let es = EdgeSamples::new(scene, edges, rig, phi, n, seed);
    scene.set_points(Some(es.points.clone()));
    let l = trace(scene);
    scene.set_points(None);
    (es.gradient(scene, seed, &l, d_img), es)
}

/// [`boundary`] with candidate importance from `vis` (built for the
/// scene as posed).
#[allow(clippy::too_many_arguments)]
pub fn boundary_with(
    scene: &mut Scene,
    edges: &Edges,
    rig: &Rig,
    phi: &[f64],
    d_img: &[f64],
    n: usize,
    seed: u64,
    vis: Option<&Visibility>,
    trace: impl FnOnce(&Scene) -> Vec<f64>,
) -> (Vec<f64>, EdgeSamples) {
    let es = EdgeSamples::with_visibility(scene, edges, rig, phi, n, seed, vis);
    scene.set_points(Some(es.points.clone()));
    let l = trace(scene);
    scene.set_points(None);
    (es.gradient(scene, seed, &l, d_img), es)
}

/// Rotation by `angle` about the line through `c` along unit `axis`.
pub fn about(c: Point3, axis: Vec3, angle: f64) -> Transform {
    let (s, co) = angle.sin_cos();
    let (x, y, z) = (axis.x, axis.y, axis.z);
    let t = 1.0 - co;
    let r = [
        [t * x * x + co, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + co, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + co],
    ];
    let cv = c.coords();
    let rc = Vec3::new(
        r[0][0] * cv.x + r[0][1] * cv.y + r[0][2] * cv.z,
        r[1][0] * cv.x + r[1][1] * cv.y + r[1][2] * cv.z,
        r[2][0] * cv.x + r[2][1] * cv.y + r[2][2] * cv.z,
    );
    let tr = cv - rc;
    let m = tang::Mat4::new(
        r[0][0], r[0][1], r[0][2], tr.x, r[1][0], r[1][1], r[1][2], tr.y, r[2][0], r[2][1],
        r[2][2], tr.z, 0.0, 0.0, 0.0, 1.0,
    );
    Transform::from_matrix(m)
}

/// Phase 2f: one piece of an intersection curve — where triangle `ta` of
/// placement `ka` passes through triangle `tb` of placement `kb` (world
/// ends `a`, `b`, at the current pose).
#[derive(Debug, Clone, Copy)]
pub struct CurveSeg {
    /// First placement and its triangle.
    pub ka: usize,
    /// Triangle of `ka`.
    pub ta: usize,
    /// Second placement and its triangle.
    pub kb: usize,
    /// Triangle of `kb`.
    pub tb: usize,
    /// World ends.
    pub a: Point3,
    /// World ends.
    pub b: Point3,
}

fn tri_world(scene: &Scene, k: usize, t: usize, place: Option<&Transform>) -> [Point3; 3] {
    let p = &scene.placements()[k];
    let m = scene.blas()[p.blas].geometry().mesh();
    let tr = place.unwrap_or(&p.to_world);
    m.triangles()[t].map(|i| tr.apply_point(&m.positions()[i as usize]))
}

/// The segment where two triangles cross, if they do (Möller 1997's
/// interval overlap on the planes' common line).
pub fn tri_tri(ta: [Point3; 3], tb: [Point3; 3]) -> Option<(Point3, Point3)> {
    let plane = |t: &[Point3; 3]| {
        let n = (t[1] - t[0]).cross(&(t[2] - t[0]));
        let l = n.norm();
        (l > 0.0).then(|| (n / l, n.dot(&t[0].coords()) / l))
    };
    let (na, da) = plane(&ta)?;
    let (nb, db) = plane(&tb)?;
    let dir = na.cross(&nb);
    if dir.norm() < 1e-9 {
        return None;
    }
    let dir = dir.normalize();
    // The part of `t` on the other plane, as an interval along `dir`.
    let clip = |t: &[Point3; 3], n: Vec3, d: f64| -> Option<(f64, f64, Point3, Point3)> {
        let s: Vec<f64> = t.iter().map(|p| n.dot(&p.coords()) - d).collect();
        let mut pts = Vec::new();
        for i in 0..3 {
            let j = (i + 1) % 3;
            if (s[i] > 0.0) != (s[j] > 0.0) {
                let u = s[i] / (s[i] - s[j]);
                pts.push(t[i] + (t[j] - t[i]) * u);
            }
        }
        if pts.len() < 2 {
            return None;
        }
        let (p, q) = (pts[0], pts[1]);
        let (u, v) = (dir.dot(&p.coords()), dir.dot(&q.coords()));
        Some(if u <= v { (u, v, p, q) } else { (v, u, q, p) })
    };
    let (a0, a1, pa0, pa1) = clip(&ta, nb, db)?;
    let (b0, b1, pb0, pb1) = clip(&tb, na, da)?;
    let lo = if a0 >= b0 { pa0 } else { pb0 };
    let hi = if a1 <= b1 { pa1 } else { pb1 };
    (a0.max(b0) < a1.min(b1)).then_some((lo, hi))
}

/// Every intersection-curve piece between a moving placement and anything
/// else (or another moving one), at the scene's pose: from
/// [`crate::secondary::pierces`]' crossings, each piercing edge's two
/// triangles against the pierced triangle.
pub fn curves(scene: &Scene, edges: &Edges, moving: &[usize]) -> Vec<CurveSeg> {
    let mut pairs = std::collections::BTreeSet::new();
    let mut add = |ka: usize, ta: usize, kb: usize, tb: usize| {
        let key = if (ka, ta) < (kb, tb) { (ka, ta, kb, tb) } else { (kb, tb, ka, ta) };
        pairs.insert(key);
    };
    let mut from: Vec<usize> = moving.to_vec();
    // Static instances pierced by a moving one may pierce back.
    let mut extra = std::collections::BTreeSet::new();
    for &k in moving {
        for p in crate::secondary::pierces(scene, edges, k) {
            if !moving.contains(&p.into) {
                extra.insert(p.into);
            }
        }
    }
    from.extend(extra);
    for &k in &from {
        let blas = scene.placements()[k].blas;
        for p in crate::secondary::pierces(scene, edges, k) {
            if !moving.contains(&k) && !moving.contains(&p.into) {
                continue;
            }
            add(k, p.tri, p.into, p.prim);
            let nb = edges.nbr[blas][p.tri][p.e];
            if nb != NONE && nb != FLAT {
                add(k, nb as usize, p.into, p.prim);
            } else if nb == FLAT {
                // The coplanar neighbour across a flat edge: find it by
                // the shared corners.
                let m = scene.blas()[blas].geometry().mesh();
                let t = m.triangles()[p.tri];
                let (i0, i1) = (t[p.e], t[(p.e + 1) % 3]);
                for (q, tq) in m.triangles().iter().enumerate() {
                    if q != p.tri && tq.contains(&i0) && tq.contains(&i1) {
                        add(k, q, p.into, p.prim);
                    }
                }
            }
        }
    }
    pairs
        .into_iter()
        .filter_map(|(ka, ta, kb, tb)| {
            let (a, b) = tri_tri(tri_world(scene, ka, ta, None), tri_world(scene, kb, tb, None))?;
            ((b - a).norm() > 0.0).then_some(CurveSeg { ka, ta, kb, tb, a, b })
        })
        .collect()
}

impl EdgeSamples {
    /// Phase 2f: samples on the intersection curves `segs` (from
    /// [`curves`] at the scene's pose `phi`), `n` of them by screen length.
    /// Each side of the curve shows a different instance; the jump is
    /// `f(s − εn) − f(s + εn)` like any edge's, and the curve's normal
    /// speed comes from re-intersecting the two triangles' planes at
    /// `φ ± RIG_H`. A sample counts only where the two rays land on the
    /// two instances at the curve's depth.
    pub fn for_curves(scene: &Scene, rig: &Rig, phi: &[f64], segs: &[CurveSeg], n: usize, seed: u64) -> Self {
        let np = phi.len();
        let (w, h) = (scene.size.0 as f64, scene.size.1 as f64);
        let (eye, fwd, right, up, _) = scene.camera();
        let shifted: Vec<[Vec<Transform>; 2]> = (0..np)
            .map(|j| {
                let mut p = phi.to_vec();
                p[j] += RIG_H;
                let plus = (rig.place)(&p);
                p[j] -= 2.0 * RIG_H;
                [plus, (rig.place)(&p)]
            })
            .collect();
        let placement = |k: usize, j: usize, side: usize| -> Option<&Transform> {
            rig.moving.iter().position(|&m| m == k).map(|mi| &shifted[j][side][mi])
        };
        let mut proj = Vec::new();
        let mut cum = Vec::new();
        let mut total = 0.0;
        for (si, sg) in segs.iter().enumerate() {
            let (Some((pa, _)), Some((pb, _))) = (scene.project(sg.a), scene.project(sg.b)) else {
                continue;
            };
            let Some((t0, t1)) = clip(pa, pb, w, h) else {
                continue;
            };
            let len = (pb[0] - pa[0]).hypot(pb[1] - pa[1]) * (t1 - t0);
            if len <= 0.0 {
                continue;
            }
            total += len;
            cum.push(total);
            proj.push((si, pa, pb, t0, t1));
        }
        let mut out = Self {
            points: Vec::with_capacity(2 * n),
            pixel: Vec::with_capacity(n),
            weight: Vec::with_capacity(n),
            total_len: total,
            silhouette_len: 0.0,
            segments: proj.len(),
            visible: 0,
            halved: 0,
        };
        for i in 0..n {
            let put = |o: &mut Self, s: [f64; 2], nrm: [f64; 2], wt: Vec<f64>| {
                let at = |d: f64| scene.screen(s[0] + d * nrm[0], s[1] + d * nrm[1]);
                o.points.push(at(-EPS_PX));
                o.points.push(at(EPS_PX));
                let x = (s[0].floor().clamp(0.0, w - 1.0)) as usize;
                let y = (s[1].floor().clamp(0.0, h - 1.0)) as usize;
                o.pixel.push(y * scene.size.0 + x);
                o.weight.push(wt);
            };
            if proj.is_empty() {
                put(&mut out, [0.0, 0.0], [1.0, 0.0], vec![0.0; np]);
                continue;
            }
            let u = (i as f64 + Rng::new(seed ^ 0x6375_7276, i as u64).uniform()) / n as f64 * total;
            let pi = cum.partition_point(|&c| c <= u).min(proj.len() - 1);
            let (si, pa, pb, t0, t1) = proj[pi];
            let start = if pi == 0 { 0.0 } else { cum[pi - 1] };
            let frac = ((u - start) / (cum[pi] - start)).clamp(0.0, 1.0);
            let t = t0 + frac * (t1 - t0);
            let d = [pb[0] - pa[0], pb[1] - pa[1]];
            let s = [pa[0] + t * d[0], pa[1] + t * d[1]];
            let dl = d[0].hypot(d[1]);
            let nrm = [d[1] / dl, -d[0] / dl];
            let sg = segs[si];
            // The 3D point under `s`: the segment line's closest approach
            // to the camera ray through `s`.
            let [sx, sy] = scene.screen(s[0], s[1]);
            let r = fwd + right * sx + up * sy;
            let e = sg.b - sg.a;
            let w0 = eye - sg.a;
            let (aa, bb, cc, dd, ee) = (r.dot(&r), r.dot(&e), e.dot(&e), r.dot(&w0), e.dot(&w0));
            let den = aa * cc - bb * bb;
            let v = if den.abs() > 0.0 { ((aa * ee - bb * dd) / den).clamp(0.0, 1.0) } else { t };
            let x = sg.a + e * v;
            let depth = (x - eye).norm();
            let hit = |q: [f64; 2]| {
                let r = fwd + right * q[0] + up * q[1];
                scene.tlas().trace_closest_range(&Ray::new(eye, r), 0.0, f64::INFINITY)
            };
            let near = |ih: &Option<irys_core::InstanceHit>| -> Option<usize> {
                let ih = ih.as_ref()?;
                (((ih.hit.point - eye).norm() - depth).abs() < 1e-4 * depth.max(1e-9)
                    && (ih.payload == sg.ka || ih.payload == sg.kb))
                    .then_some(ih.payload)
            };
            let hi = hit(scene.screen(s[0] - EPS_PX * nrm[0], s[1] - EPS_PX * nrm[1]));
            let ho = hit(scene.screen(s[0] + EPS_PX * nrm[0], s[1] + EPS_PX * nrm[1]));
            let (Some(a), Some(b), Some(jac)) = (near(&hi), near(&ho), scene.project(x).map(|p| p.1)) else {
                put(&mut out, s, nrm, vec![0.0; np]);
                continue;
            };
            if a == b {
                put(&mut out, s, nrm, vec![0.0; np]);
                continue;
            }
            out.visible += 1;
            // The curve's motion: re-intersect the two planes at φ ± h and
            // take the point on the new line nearest `x`.
            let line_at = |j: usize, side: usize| -> Option<Point3> {
                let ta = tri_world(scene, sg.ka, sg.ta, placement(sg.ka, j, side));
                let tb = tri_world(scene, sg.kb, sg.tb, placement(sg.kb, j, side));
                let n1 = (ta[1] - ta[0]).cross(&(ta[2] - ta[0])).normalize();
                let n2 = (tb[1] - tb[0]).cross(&(tb[2] - tb[0])).normalize();
                let dir = n1.cross(&n2);
                let l2 = dir.norm_squared();
                if l2 < 1e-18 {
                    return None;
                }
                // Point on both planes nearest x: x minus its plane
                // offsets, solved in the planes' normal space.
                let (d1, d2) = (n1.dot(&(x - ta[0])), n2.dot(&(x - tb[0])));
                let c = n1.dot(&n2);
                let det = 1.0 - c * c;
                let (g1, g2) = ((d1 - c * d2) / det, (d2 - c * d1) / det);
                Some(x - n1 * g1 - n2 * g2)
            };
            let wt: Vec<f64> = (0..np)
                .map(|j| match (line_at(j, 0), line_at(j, 1)) {
                    (Some(p), Some(q)) => {
                        let dx = (p - q) / (2.0 * RIG_H);
                        let vel = [jac[0].dot(&dx), jac[1].dot(&dx)];
                        (vel[0] * nrm[0] + vel[1] * nrm[1]) * total / n as f64
                    }
                    _ => 0.0,
                })
                .collect();
            put(&mut out, s, nrm, wt);
        }
        out
    }
}
