//! Phase 2e: the interior term of the pose (and camera) gradient.
//!
//! [`crate::vis`] gives the part of `∂I_p/∂φ` that lives on the curves where
//! the integrand jumps. This is the rest: the derivative of the path
//! estimator *at fixed random numbers* (primary sample space), where the
//! integrand is smooth. As an instance moves, a path through a fixed pixel
//! point hits it at a moving point, with a turning normal; the next
//! direction, drawn in the hit's local frame, turns too; every BSDF value,
//! cosine, pdf and MIS weight downstream moves with it.
//!
//! # Hand-derived forward mode, not the IR
//!
//! The derivative is carried by forward-mode dual numbers ([`D`], one
//! tangent per parameter, at most [`MAXP`]) through a re-trace of the
//! frame: intersection (Möller–Trumbore against the triangle the primal
//! trace hit, with its corners' world positions as duals), the shading
//! frame, GGX visible-normal sampling, GGX eval, NEE, the environment and
//! the MIS weights. Every rule is the textbook one for `+ − × ÷ √ exp`,
//! written out here — no graph. The IR's [`crate::ggx::Graph`] is batched
//! over vertices with `θ` broadcast and differentiates *to `θ`*; making it
//! take `∂cos/∂φ` lanes would need a second graph whose backward is to the
//! columns, for a gain only on the GGX factor. The duals cover every
//! factor at once, and the test against central differences is the same.
//!
//! # Attached frames
//!
//! The renderer draws a vertex's direction in `basis(n)` (Duff et al.),
//! which flips sign where `n_z` crosses 0 — a discontinuity in `φ` at fixed
//! random numbers that no edge term would see. The derivative here is
//! instead that of the estimator whose frame at a vertex on a moving
//! instance is `R(φ) R(φ₀)ᵀ basis(n(φ₀))`: rigidly attached. At `φ₀` it is
//! the rendered estimator exactly, and for every `φ` it is an equally valid
//! estimator of the same `I(φ)` (any orthonormal frame around the normal
//! is), so its derivative at `φ₀` is an unbiased estimate of `∂I/∂φ` from
//! the very paths the forward pass drew. Rigid motion is assumed for the
//! moving placements (pose rigs are); the frame's angular velocity is the
//! skew part of `L̇ L⁻¹` of the placement's linear part.
//!
//! # What it leaves out
//!
//! Discontinuities in primary sample space other than the primary edges:
//! a shadow ray's occluder moving across it (shadow edges), and secondary
//! rays' visibility changing. Control flow (which triangle, lit or not,
//! alive or not) is the primal trace's. Analytic prims are refused on the
//! path (triangles only), splats are refused, and the checker albedo is
//! piecewise constant (derivative 0, its edges ignored).

use irys_core::{Point3, Ray, Transform, Vec3};

use crate::ggx::PARAMS;
use crate::prb::{Rng, Scene, Surface, lane_adjoint, offset, wavelengths};
use crate::vis::{RIG_H, Rig};

/// Most parameters a tangent carries.
pub const MAXP: usize = 8;

/// A dual number: a value and its derivative to each of up to [`MAXP`]
/// parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct D {
    /// The value.
    pub v: f64,
    /// `∂v/∂φ_j`.
    pub d: [f64; MAXP],
}

impl D {
    /// A constant.
    pub const fn c(v: f64) -> Self {
        Self { v, d: [0.0; MAXP] }
    }

    fn map(self, v: f64, dv: f64) -> Self {
        Self {
            v,
            d: self.d.map(|x| x * dv),
        }
    }

    /// `√x`.
    pub fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        self.map(s, if s > 0.0 { 0.5 / s } else { 0.0 })
    }

    /// `eˣ`.
    pub fn exp(self) -> Self {
        let e = self.v.exp();
        self.map(e, e)
    }

    /// `max(x, 0)`.
    pub fn max0(self) -> Self {
        if self.v > 0.0 { self } else { D::c(0.0) }
    }

    /// `xⁿ`.
    pub fn powi(self, n: i32) -> Self {
        self.map(self.v.powi(n), n as f64 * self.v.powi(n - 1))
    }
}

macro_rules! binop {
    ($tr:ident, $f:ident, $body:expr) => {
        impl std::ops::$tr for D {
            type Output = D;
            fn $f(self, o: D) -> D {
                let g: fn(D, D) -> D = $body;
                g(self, o)
            }
        }
        impl std::ops::$tr<f64> for D {
            type Output = D;
            fn $f(self, o: f64) -> D {
                std::ops::$tr::$f(self, D::c(o))
            }
        }
        impl std::ops::$tr<D> for f64 {
            type Output = D;
            fn $f(self, o: D) -> D {
                std::ops::$tr::$f(D::c(self), o)
            }
        }
    };
}

fn zip(a: D, b: D, v: f64, f: impl Fn(f64, f64) -> f64) -> D {
    D {
        v,
        d: std::array::from_fn(|j| f(a.d[j], b.d[j])),
    }
}

binop!(Add, add, |a, b| zip(a, b, a.v + b.v, |x, y| x + y));
binop!(Sub, sub, |a, b| zip(a, b, a.v - b.v, |x, y| x - y));
binop!(Mul, mul, |a, b| zip(a, b, a.v * b.v, |x, y| x * b.v
    + a.v * y));
binop!(Div, div, |a, b| {
    let q = a.v / b.v;
    zip(a, b, q, |x, y| (x - q * y) / b.v)
});

impl std::ops::Neg for D {
    type Output = D;
    fn neg(self) -> D {
        self.map(-self.v, -1.0)
    }
}

/// A 3-vector of duals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct V3 {
    /// Components.
    pub x: D,
    /// Components.
    pub y: D,
    /// Components.
    pub z: D,
}

impl V3 {
    /// A constant vector.
    pub fn c(v: Vec3) -> Self {
        Self {
            x: D::c(v.x),
            y: D::c(v.y),
            z: D::c(v.z),
        }
    }

    /// A constant point.
    pub fn p(p: Point3) -> Self {
        Self::c(p.coords())
    }

    /// Value `v`, derivative `d[j]` to parameter `j`.
    pub fn with(v: Vec3, d: &[Vec3]) -> Self {
        let mut o = Self::c(v);
        for (j, dj) in d.iter().enumerate() {
            o.x.d[j] = dj.x;
            o.y.d[j] = dj.y;
            o.z.d[j] = dj.z;
        }
        o
    }

    /// The value.
    pub fn v(&self) -> Vec3 {
        Vec3::new(self.x.v, self.y.v, self.z.v)
    }

    /// Tangent `j`.
    pub fn d(&self, j: usize) -> Vec3 {
        Vec3::new(self.x.d[j], self.y.d[j], self.z.d[j])
    }

    /// `a · b`.
    pub fn dot(&self, o: &V3) -> D {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    /// `a × b`.
    pub fn cross(&self, o: &V3) -> V3 {
        V3 {
            x: self.y * o.z - self.z * o.y,
            y: self.z * o.x - self.x * o.z,
            z: self.x * o.y - self.y * o.x,
        }
    }

    /// Scaled by `s`.
    pub fn s(&self, s: D) -> V3 {
        V3 {
            x: self.x * s,
            y: self.y * s,
            z: self.z * s,
        }
    }

    /// Unit length.
    pub fn normalize(&self) -> V3 {
        self.s(1.0 / self.dot(self).sqrt())
    }
}

impl std::ops::Add for V3 {
    type Output = V3;
    fn add(self, o: V3) -> V3 {
        V3 {
            x: self.x + o.x,
            y: self.y + o.y,
            z: self.z + o.z,
        }
    }
}

impl std::ops::Sub for V3 {
    type Output = V3;
    fn sub(self, o: V3) -> V3 {
        V3 {
            x: self.x - o.x,
            y: self.y - o.y,
            z: self.z - o.z,
        }
    }
}

impl std::ops::Neg for V3 {
    type Output = V3;
    fn neg(self) -> V3 {
        V3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// The pinhole as duals: what the camera parameters move.
#[derive(Debug, Clone, Copy)]
pub struct Cam {
    /// Eye.
    pub eye: V3,
    /// Forward.
    pub forward: V3,
    /// Right.
    pub right: V3,
    /// Up.
    pub up: V3,
    /// `tan` of the half field of view.
    pub tan_half: D,
}

impl Cam {
    /// The scene's camera, held fixed.
    pub fn fixed(scene: &Scene) -> Self {
        let (eye, f, r, u, t) = scene.camera();
        Self {
            eye: V3::p(eye),
            forward: V3::c(f),
            right: V3::c(r),
            up: V3::c(u),
            tan_half: D::c(t),
        }
    }
}

/// How each placement and the camera move with the parameters.
pub struct Tangents {
    /// Parameters.
    pub np: usize,
    /// Per placement payload, for a moving one: per parameter, its
    /// placement at `φ ± RIG_H`.
    shifted: Vec<Option<Vec<[Transform; 2]>>>,
    /// Per placement payload, for a moving one: per parameter, the angular
    /// velocity (skew 3 × 3, rows) of its frame.
    omega: Vec<Option<Vec<[[f64; 3]; 3]>>>,
    /// The camera.
    pub cam: Cam,
}

fn linear(t: &Transform) -> [[f64; 3]; 3] {
    let c = [
        t.apply_vec(&Vec3::new(1.0, 0.0, 0.0)),
        t.apply_vec(&Vec3::new(0.0, 1.0, 0.0)),
        t.apply_vec(&Vec3::new(0.0, 0.0, 1.0)),
    ];
    std::array::from_fn(|r| std::array::from_fn(|k| [c[k].x, c[k].y, c[k].z][r]))
}

fn mat_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn mat_vec(a: &[[f64; 3]; 3], v: Vec3) -> Vec3 {
    let v = [v.x, v.y, v.z];
    let r: [f64; 3] = std::array::from_fn(|i| (0..3).map(|k| a[i][k] * v[k]).sum());
    Vec3::new(r[0], r[1], r[2])
}

fn inverse3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let c = |i: usize, j: usize| {
        let (a, b) = ((i + 1) % 3, (i + 2) % 3);
        let (d, e) = ((j + 1) % 3, (j + 2) % 3);
        m[a][d] * m[b][e] - m[a][e] * m[b][d]
    };
    let det: f64 = (0..3).map(|j| m[0][j] * c(0, j)).sum();
    std::array::from_fn(|i| std::array::from_fn(|j| c(j, i) / det))
}

impl Tangents {
    /// Nothing moves: every tangent is zero (`np` parameters).
    pub fn none(scene: &Scene, np: usize) -> Self {
        assert!(np <= MAXP, "{np} parameters, at most {MAXP}");
        Self {
            np,
            shifted: vec![None; scene.placements().len()],
            omega: vec![None; scene.placements().len()],
            cam: Cam::fixed(scene),
        }
    }

    /// The pose `rig` at `phi` (the scene is posed there).
    pub fn rig(scene: &Scene, rig: &Rig, phi: &[f64]) -> Self {
        let mut t = Self::none(scene, phi.len());
        let at: Vec<[Vec<Transform>; 2]> = (0..phi.len())
            .map(|j| {
                let mut p = phi.to_vec();
                p[j] += RIG_H;
                let a = (rig.place)(&p);
                p[j] -= 2.0 * RIG_H;
                [a, (rig.place)(&p)]
            })
            .collect();
        for (mi, &k) in rig.moving.iter().enumerate() {
            let l0inv = inverse3(&linear(&scene.placements()[k].to_world));
            let sh: Vec<[Transform; 2]> = at
                .iter()
                .map(|[a, b]| [a[mi].clone(), b[mi].clone()])
                .collect();
            let om = sh
                .iter()
                .map(|[a, b]| {
                    let (la, lb) = (linear(a), linear(b));
                    let dl: [[f64; 3]; 3] = std::array::from_fn(|i| {
                        std::array::from_fn(|j| (la[i][j] - lb[i][j]) / (2.0 * RIG_H))
                    });
                    let w = mat_mul(&dl, &l0inv);
                    std::array::from_fn(|i| std::array::from_fn(|j| 0.5 * (w[i][j] - w[j][i])))
                })
                .collect();
            t.shifted[k] = Some(sh);
            t.omega[k] = Some(om);
        }
        t
    }

    /// The camera rig at `c` (the scene's camera is there; nothing else
    /// moves).
    pub fn camera(scene: &Scene, cam: &crate::vis::CameraRig, c: &[f64]) -> Self {
        let mut t = Self::none(scene, c.len());
        let at: Vec<[crate::vis::Pinhole; 2]> = (0..c.len())
            .map(|j| {
                let mut p = c.to_vec();
                p[j] += RIG_H;
                let a = cam.pinhole(&p);
                p[j] -= 2.0 * RIG_H;
                [a, cam.pinhole(&p)]
            })
            .collect();
        let p0 = cam.pinhole(c);
        let dv = |f: &dyn Fn(&crate::vis::Pinhole) -> Vec3| -> Vec<Vec3> {
            at.iter()
                .map(|[a, b]| (f(a) - f(b)) / (2.0 * RIG_H))
                .collect()
        };
        t.cam = Cam {
            eye: V3::with(p0.eye.coords(), &dv(&|p| p.eye.coords())),
            forward: V3::with(p0.forward, &dv(&|p| p.forward)),
            right: V3::with(p0.right, &dv(&|p| p.right)),
            up: V3::with(p0.up, &dv(&|p| p.up)),
            tan_half: {
                let mut d = D::c(p0.tan_half);
                for (j, [a, b]) in at.iter().enumerate() {
                    d.d[j] = (a.tan_half - b.tan_half) / (2.0 * RIG_H);
                }
                d
            },
        };
        t
    }

    /// A world point carried by placement `k`, from its object-space `p`.
    fn point(&self, scene: &Scene, k: usize, p: Point3) -> V3 {
        let x = scene.placements()[k].to_world.apply_point(&p).coords();
        match &self.shifted[k] {
            None => V3::c(x),
            Some(sh) => {
                let d: Vec<Vec3> = sh
                    .iter()
                    .map(|[a, b]| (a.apply_point(&p) - b.apply_point(&p)) / (2.0 * RIG_H))
                    .collect();
                V3::with(x, &d)
            }
        }
    }

    /// A unit world vector `v` rigidly attached to placement `k`.
    fn attached(&self, k: usize, v: Vec3) -> V3 {
        match &self.omega[k] {
            None => V3::c(v),
            Some(om) => {
                let d: Vec<Vec3> = om.iter().map(|w| mat_vec(w, v)).collect();
                V3::with(v, &d)
            }
        }
    }
}

fn pt(v: Vec3) -> Point3 {
    Point3::new(v.x, v.y, v.z)
}

/// A local frame around `n` (Duff et al. 2017), the renderer's.
fn basis(n: Vec3) -> (Vec3, Vec3) {
    let s = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    (
        Vec3::new(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
        Vec3::new(b, s + n.y * n.y * a, -n.y),
    )
}

/// `f` and the visible-normal pdf, [`crate::ggx::eval_f64`] in duals.
fn ggx_eval(
    theta: &[f64; PARAMS],
    alpha_pdf: f64,
    b: [f64; 3],
    ci: D,
    co: D,
    ch: D,
    oh: D,
) -> (D, D) {
    let lam = |a2: f64, c: D| (((1.0 + a2 * (1.0 - c * c) / (c * c)).sqrt()) - 1.0) * 0.5;
    let dd = |a2: f64| {
        let k = ch * ch * (a2 - 1.0) + 1.0;
        a2 / (std::f64::consts::PI * k * k)
    };
    let a2 = theta[0] * theta[0];
    let f0 = b[0] * theta[1] + b[1] * theta[2] + b[2] * theta[3];
    let fr = f0 + (1.0 - f0) * (1.0 - oh).powi(5);
    let f = fr * dd(a2) / (1.0 + lam(a2, ci) + lam(a2, co)) / (4.0 * ci * co);
    let p2 = alpha_pdf * alpha_pdf;
    let pdf = dd(p2) / (1.0 + lam(p2, co)) / (4.0 * co);
    (f, pdf)
}

/// [`crate::ggx::sample`] in duals: `(wi, h)` in the local frame.
fn ggx_sample(alpha: f64, wo: [D; 3], u: [f64; 2]) -> ([D; 3], [D; 3]) {
    let norm3 = |a: [D; 3]| {
        let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        [a[0] / l, a[1] / l, a[2] / l]
    };
    let vh = norm3([alpha * wo[0], alpha * wo[1], wo[2]]);
    let len2 = vh[0] * vh[0] + vh[1] * vh[1];
    let t1 = if len2.v > 0.0 {
        let s = 1.0 / len2.sqrt();
        [-vh[1] * s, vh[0] * s, D::c(0.0)]
    } else {
        [D::c(1.0), D::c(0.0), D::c(0.0)]
    };
    let t2 = [
        vh[1] * t1[2] - vh[2] * t1[1],
        vh[2] * t1[0] - vh[0] * t1[2],
        vh[0] * t1[1] - vh[1] * t1[0],
    ];
    let r = u[0].sqrt();
    let phi = 2.0 * std::f64::consts::PI * u[1];
    let p1 = r * phi.cos();
    let s = 0.5 * (1.0 + vh[2]);
    let p2 = (1.0 - s) * (1.0 - p1 * p1).max(0.0).sqrt() + s * (r * phi.sin());
    let p3 = (D::c(1.0 - p1 * p1) - p2 * p2).max0().sqrt();
    let nh: [D; 3] = std::array::from_fn(|k| p1 * t1[k] + p2 * t2[k] + p3 * vh[k]);
    let nz = if nh[2].v > 1e-12 { nh[2] } else { D::c(1e-12) };
    let h = norm3([alpha * nh[0], alpha * nh[1], nz]);
    let oh = wo[0] * h[0] + wo[1] * h[1] + wo[2] * h[2];
    let wi = std::array::from_fn(|k| 2.0 * oh * h[k] - wo[k]);
    (wi, h)
}

fn env_radiance(scene: &Scene, d: &V3, lambda_nm: f64) -> D {
    let e = scene.env;
    let tint = 1.0 + 0.4 * (550.0 - lambda_nm) / 200.0;
    tint * e.sky * (0.15 + 0.25 * d.z.max0())
        + e.sun_radiance * (e.kappa * (d.dot(&V3::c(e.sun)) - 1.0)).exp()
}

fn env_pdf(scene: &Scene, d: &V3) -> D {
    let e = scene.env;
    let k = e.kappa;
    let vmf = k / (2.0 * std::f64::consts::PI * (1.0 - (-2.0 * k).exp()))
        * (k * (d.dot(&V3::c(e.sun)) - 1.0)).exp();
    crate::prb::SUN_PICK * vmf + (1.0 - crate::prb::SUN_PICK) / (4.0 * std::f64::consts::PI)
}

fn mis_d(a: D, b: D) -> D {
    let (a2, b2) = (a * a, b * b);
    if a2.v + b2.v > 0.0 {
        a2 / (a2 + b2)
    } else {
        D::c(0.0)
    }
}

/// One path's `L` per wavelength, as duals.
fn path(scene: &Scene, tan: &Tangents, theta: &[f64; PARAMS], seed: u64, i: usize) -> [D; 4] {
    let nw = scene.wavelengths;
    let env = scene.env;
    let mut rng = Rng::new(seed, scene.path_key(i));
    let lambda = wavelengths(rng.uniform(), nw);
    // The camera ray, as `Scene::camera_ray` draws it.
    let (w, h) = scene.size;
    let px = i / scene.spp;
    let (x, y) = (
        (px % w) as f64 + rng.uniform(),
        (px / w) as f64 + rng.uniform(),
    );
    let c = &tan.cam;
    let (wf, hf) = (w as f64, h as f64);
    let sx = (2.0 * x / wf - 1.0) * c.tan_half * (wf / hf);
    let sy = (1.0 - 2.0 * y / hf) * c.tan_half;
    let mut o = c.eye;
    let mut o_trace = pt(c.eye.v());
    let mut d = (c.forward + c.right.s(sx) + c.up.s(sy)).normalize();
    let mut tp = [D::c(1.0); 4];
    let mut rad = [D::c(0.0); 4];
    let mut prev_pdf = D::c(0.0);
    for _ in 0..scene.depth {
        let ray = Ray::new(o_trace, d.v());
        let Some(ih) = scene.tlas().trace_closest_range(&ray, 0.0, f64::INFINITY) else {
            let wm = if scene.nee && prev_pdf.v > 0.0 {
                mis_d(prev_pdf, env_pdf(scene, &d))
            } else {
                D::c(1.0)
            };
            for k in 0..nw {
                rad[k] = rad[k] + tp[k] * env_radiance(scene, &d, lambda[k]) * wm;
            }
            break;
        };
        // The hit, re-intersected in duals against the triangle it found.
        let k = ih.payload;
        let place = &scene.placements()[k];
        let geo = scene.blas()[place.blas].geometry();
        let mesh = geo.mesh();
        let prim = ih.hit.prim as usize;
        assert!(
            prim < mesh.triangles().len(),
            "interior term: triangles only on the path"
        );
        let tri = mesh.triangles()[prim];
        let [a, b, cc] = tri.map(|vi| tan.point(scene, k, mesh.positions()[vi as usize]));
        let (e1, e2) = (b - a, cc - a);
        let pv = d.cross(&e2);
        let det = e1.dot(&pv);
        let tv = o - a;
        let bu = tv.dot(&pv) / det;
        let qv = tv.cross(&e1);
        let bv = d.dot(&qv) / det;
        let t = e2.dot(&qv) / det;
        let xh = o + d.s(t);
        let mut n = if mesh.normals().is_empty() {
            e1.cross(&e2).normalize()
        } else {
            let wn = |vi: u32| {
                let nl = mesh.normals()[vi as usize];
                tan.attached(k, place.to_world.apply_normal(&nl).normalize())
            };
            let [n0, n1, n2] = tri.map(wn);
            (n0.s(1.0 - bu - bv) + n1.s(bu) + n2.s(bv)).normalize()
        };
        if n.dot(&d).v > 0.0 {
            n = -n;
        }
        let nv = n.v();
        let (t0, b0) = basis(nv);
        let (tt, bb) = (tan.attached(k, t0), tan.attached(k, b0));
        let wo = -d;
        let eps = {
            let inf = |p: Vec3| p.x.abs().max(p.y.abs()).max(p.z.abs());
            (inf(xh.v()) / 32768.0).max((inf(o.v()) + ih.hit.t) / 131072.0)
        };
        let origin_v = offset(ih.hit.point, nv, o_trace, ih.hit.t);
        let origin = xh + n.s(D::c(eps));
        let nee = scene.nee.then(|| {
            let dir = env.sample([rng.uniform(), rng.uniform(), rng.uniform()]);
            let cos = dir.dot(&nv);
            let lit = cos > 0.0
                && !scene
                    .tlas()
                    .occluded_range(&Ray::new(origin_v, dir), 0.0, f64::INFINITY);
            (dir, env.pdf(dir), lit)
        });
        if scene.surface(&ih) == Surface::Diffuse {
            let albedo = place.albedo
                * if place.checker {
                    scene.albedo(ih.hit.point)
                } else {
                    scene.floor_albedo
                };
            if let Some((dir, pl, true)) = nee {
                let cos = n.dot(&V3::c(dir));
                let wl = mis_d(D::c(pl), cos / std::f64::consts::PI);
                for kk in 0..nw {
                    rad[kk] = rad[kk]
                        + tp[kk]
                            * (albedo / std::f64::consts::PI)
                            * cos
                            * env.radiance(dir, lambda[kk])
                            * wl
                            / pl;
                }
            }
            let (u1, u2) = (rng.uniform(), rng.uniform());
            let (r, ph) = (u1.sqrt(), 2.0 * std::f64::consts::PI * u2);
            let cz = (1.0 - u1).max(0.0).sqrt();
            d = tt.s(D::c(r * ph.cos())) + bb.s(D::c(r * ph.sin())) + n.s(D::c(cz));
            for x in &mut tp[..nw] {
                *x = *x * albedo;
            }
            prev_pdf = D::c(cz / std::f64::consts::PI);
            o = origin;
            o_trace = origin_v;
            continue;
        }
        let lo = [wo.dot(&tt), wo.dot(&bb), wo.dot(&n)];
        let alpha = scene.frozen_alpha.unwrap_or(theta[0]);
        if let Some((dir, pl, true)) = nee
            && lo[2].v > 0.0
        {
            let dv = V3::c(dir);
            let li = [dv.dot(&tt), dv.dot(&bb), dv.dot(&n)];
            let hh = {
                let s = [li[0] + lo[0], li[1] + lo[1], li[2] + lo[2]];
                let l = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
                [s[0] / l, s[1] / l, s[2] / l]
            };
            let oh = lo[0] * hh[0] + lo[1] * hh[1] + lo[2] * hh[2];
            for kk in 0..nw {
                let bw = crate::ggx::band_weights(lambda[kk]);
                let (f, pb) = ggx_eval(theta, alpha, bw, li[2], lo[2], hh[2], oh);
                let s = li[2] * mis_d(D::c(pl), pb) / pl;
                rad[kk] = rad[kk] + tp[kk] * s * env.radiance(dir, lambda[kk]) * f;
            }
        }
        let u = [rng.uniform(), rng.uniform()];
        let (li, hh) = ggx_sample(alpha, lo, u);
        let oh = lo[0] * hh[0] + lo[1] * hh[1] + lo[2] * hh[2];
        let (_, pdf) = ggx_eval(theta, alpha, [0.0; 3], li[2], lo[2], hh[2], oh);
        if li[2].v <= 0.0 || lo[2].v <= 0.0 || !pdf.v.is_finite() || pdf.v <= 0.0 {
            break;
        }
        // The light-sampled half above used the light's `h`; the BSDF half
        // uses the sampled one, as the forward pass's `lane` recomputes it
        // from `li + lo` (equal to the sampled `h` up to rounding).
        let hs = {
            let s = [li[0] + lo[0], li[1] + lo[1], li[2] + lo[2]];
            let l = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
            [s[0] / l, s[1] / l, s[2] / l]
        };
        let ohs = lo[0] * hs[0] + lo[1] * hs[1] + lo[2] * hs[2];
        let mut alive = false;
        for kk in 0..nw {
            let bw = crate::ggx::band_weights(lambda[kk]);
            let (f, _) = ggx_eval(theta, alpha, bw, li[2], lo[2], hs[2], ohs);
            tp[kk] = tp[kk] * f * li[2] / pdf;
            alive |= tp[kk].v > 0.0;
        }
        d = tt.s(li[0]) + bb.s(li[1]) + n.s(li[2]);
        // The forward pass normalises the new ray's direction.
        d = d.normalize();
        prev_pdf = pdf;
        o = origin;
        o_trace = origin_v;
        if !alive {
            break;
        }
    }
    rad
}

/// The interior term of `∂ℒ/∂φ` for the image frame drawn with `seed`,
/// given the loss adjoint per pixel channel: `Σ_lanes δL · ∂L/∂φ` at fixed
/// random numbers. CPU, f64, threads over fixed chunks (so the sum order,
/// and the result, does not depend on the machine).
pub fn interior(
    scene: &Scene,
    tan: &Tangents,
    theta: &[f64; PARAMS],
    seed: u64,
    d_img: &[f64],
) -> Vec<f64> {
    assert!(scene.points().is_none(), "interior term: image mode only");
    assert!(scene.splats().is_none(), "interior term: no splats");
    let nw = scene.wavelengths;
    let dl = lane_adjoint(scene, seed, d_img);
    let n = scene.paths();
    const CHUNKS: usize = 64;
    let per = n.div_ceil(CHUNKS).max(1);
    let parts: Vec<[f64; MAXP]> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..n.div_ceil(per))
            .map(|c| {
                let dl = &dl;
                s.spawn(move || {
                    let mut g = [0.0; MAXP];
                    for i in c * per..((c + 1) * per).min(n) {
                        let l = path(scene, tan, theta, seed, i);
                        for k in 0..nw {
                            let a = dl[i * nw + k];
                            for (gj, dj) in g.iter_mut().zip(l[k].d) {
                                *gj += a * dj;
                            }
                        }
                    }
                    g
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut g = vec![0.0; tan.np];
    for p in parts {
        for (gj, pj) in g.iter_mut().zip(p) {
            *gj += pj;
        }
    }
    g
}

/// `L` per lane from the dual trace's values — equal to
/// [`crate::prb::forward`]'s up to rounding (the forward pass's GGX is the
/// IR's, f32 when native). For tests.
pub fn values(scene: &Scene, tan: &Tangents, theta: &[f64; PARAMS], seed: u64) -> Vec<f64> {
    let nw = scene.wavelengths;
    (0..scene.paths())
        .flat_map(|i| {
            let l = path(scene, tan, theta, seed, i);
            (0..nw).map(move |k| l[k].v)
        })
        .collect()
}

/// Every lane's `∂L/∂φ_j` from the dual trace, for tests.
pub fn lane_tangents(
    scene: &Scene,
    tan: &Tangents,
    theta: &[f64; PARAMS],
    seed: u64,
    j: usize,
) -> Vec<f64> {
    let nw = scene.wavelengths;
    (0..scene.paths())
        .flat_map(|i| {
            let l = path(scene, tan, theta, seed, i);
            (0..nw).map(move |k| l[k].d[j])
        })
        .collect()
}
