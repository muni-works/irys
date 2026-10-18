//! The OpenPBR swatch integrator: a unit sphere of one [`OpenPbr`] under
//! studio quads and a constant environment, orthographic camera, CPU f64,
//! one wavelength per path, NEE + MIS, and path replay backpropagation into
//! [`PARAMS`] through [`Stack`]'s adjoint.
//!
//! # A path
//!
//! At a sphere vertex the [`At`] of the *sampler's* material picks a lobe
//! (spec GGX, coat GGX, cosine, refraction, or the subsurface event) and a
//! direction (detached; one-sample MIS over the directional lobes); the
//! throughput takes `f cos / pdf` with `f` the [`stack`] at `θ`. Light
//! samples (outside only) use the reflection lane. Inside a transmissive
//! sphere with `transmission_depth > 0` the path is attenuated by
//! `exp(−μ_t d)`, `μ_t = −ln(transmission_color)/depth`. The subsurface
//! event multiplies the entry weight `f_s / p` and runs the skin random walk
//! ([`crate::skin::render::walk_g`]) on a tessellated copy of the sphere,
//! with van de Hulst's inversion for the albedo; the exit is a Lambertian
//! vertex with its own light sample.
//!
//! # The replay
//!
//! Each throughput factor `f_j` owes `δL (L − L_j)/f_j` to its lane, where
//! `L_j` is what had been collected when it was applied; each light sample
//! owes `δL C/f_L`. θ-free factors (walk weights, absorption, pdfs) owe
//! nothing. Paths run in parallel and collect in order; lanes reduce in
//! path order: the gradient is bit-identical run to run.

use rayon::prelude::*;

use irys_core::{Bvh, Point3, TriMesh, Vec3};
use irys_light::spectrum::{hero_weight, sample_lambda_nm};

use super::graph::{self, Stack};
use super::openpbr::{At, COLUMNS, Kind, OpenPbr, PARAMS, subsurface_medium};
use crate::prb::{Rng, mis};
use crate::skin::mesh::basis;
use crate::skin::render::{Light, walk_g};

const PI: f64 = std::f64::consts::PI;

/// The swatch scene.
pub struct Swatch {
    /// The material at `θ` (the sampler's unless [`Self::sampler`] is set).
    pub material: OpenPbr,
    /// A frozen sampler material (central-difference tests); `None` samples
    /// with `material` itself.
    pub sampler: Option<OpenPbr>,
    /// Studio quads.
    pub lights: Vec<Light>,
    /// Constant environment radiance.
    pub env: f64,
    /// Pixels.
    pub size: (usize, usize),
    /// Samples per pixel.
    pub spp: usize,
    /// Orthographic film width (scene units); the camera looks along `+y`.
    pub width: f64,
    /// Maximum path vertices.
    pub depth: usize,
    walk_mesh: Option<Bvh<TriMesh>>,
}

/// The default studio: a warm key, a cool fill, a rim, all quads.
pub fn studio() -> Vec<Light> {
    vec![
        Light::facing(
            Point3::new(-2.6, -3.0, 2.6),
            Point3::origin(),
            2.0,
            2.0,
            5.0,
            [1.0, 0.96, 0.9],
        ),
        Light::facing(
            Point3::new(3.5, -2.5, 0.5),
            Point3::origin(),
            3.0,
            3.0,
            1.0,
            [0.9, 0.95, 1.0],
        ),
        Light::facing(
            Point3::new(1.0, 3.5, 2.5),
            Point3::origin(),
            4.0,
            1.0,
            3.0,
            [1.0, 1.0, 1.0],
        ),
    ]
}

/// A subdivided icosahedron of radius 1 (the walk's boundary).
pub fn icosphere(level: usize) -> TriMesh {
    let t = (1.0 + 5f64.sqrt()) / 2.0;
    let mut v: Vec<[f64; 3]> = vec![
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ];
    let mut f: Vec<[u32; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    let unit = |p: [f64; 3]| super::openpbr::norm(p);
    for p in v.iter_mut() {
        *p = unit(*p);
    }
    for _ in 0..level {
        let mut cache = std::collections::HashMap::new();
        let mut mid = |a: u32, b: u32, v: &mut Vec<[f64; 3]>| -> u32 {
            let key = (a.min(b), a.max(b));
            *cache.entry(key).or_insert_with(|| {
                let (pa, pb) = (v[a as usize], v[b as usize]);
                v.push(unit([pa[0] + pb[0], pa[1] + pb[1], pa[2] + pb[2]]));
                (v.len() - 1) as u32
            })
        };
        let mut nf = Vec::with_capacity(f.len() * 4);
        for [a, b, c] in f {
            let ab = mid(a, b, &mut v);
            let bc = mid(b, c, &mut v);
            let ca = mid(c, a, &mut v);
            nf.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        f = nf;
    }
    let pos: Vec<Point3> = v.iter().map(|p| Point3::new(p[0], p[1], p[2])).collect();
    let nrm: Vec<Vec3> = v.iter().map(|p| Vec3::new(p[0], p[1], p[2])).collect();
    let idx: Vec<u32> = f.iter().flatten().copied().collect();
    TriMesh::new(pos, nrm, &idx)
}

/// One lane the replay owes: its columns, and `δf / δL` for it.
#[derive(Debug, Clone, Copy)]
pub struct Lane {
    /// The [`Stack`] columns.
    pub cols: [f64; COLUMNS],
    /// `f` as the forward pass evaluated it.
    pub f: f64,
    /// For a throughput factor, `L` collected before it applied; for a
    /// light sample, `−C` (so `L_total − w` is `C`... see [`Lane::weight`]).
    pub before: f64,
    /// A light sample (its contribution is `C = −before`).
    pub nee: bool,
}

impl Lane {
    /// `δf` per unit `δL`, given the path's total `L`.
    pub fn weight(&self, l: f64) -> f64 {
        let w = if self.nee { -self.before } else { l - self.before };
        if self.f != 0.0 { w / self.f } else { 0.0 }
    }
}

impl Swatch {
    /// A swatch of `material` in the studio.
    pub fn new(material: OpenPbr, size: (usize, usize), spp: usize) -> Self {
        let walk_mesh = (material.subsurface_weight > 0.0).then(|| Bvh::build(icosphere(5)));
        Self {
            material,
            sampler: None,
            lights: studio(),
            env: 0.08,
            size,
            spp,
            width: 2.3,
            depth: 12,
            walk_mesh,
        }
    }

    /// Paths in a frame.
    pub fn paths(&self) -> usize {
        self.size.0 * self.size.1 * self.spp
    }

    fn sampler_material(&self, theta: &[f64; PARAMS]) -> OpenPbr {
        self.sampler
            .unwrap_or_else(|| self.material.with_theta(theta))
    }

    fn light_sample(&self, x: Point3, u: [f64; 3]) -> Option<(Vec3, f64, f64, f64)> {
        let n = self.lights.len();
        let i = ((u[0] * n as f64) as usize).min(n - 1);
        let l = &self.lights[i];
        let p = l.center + l.u * (2.0 * u[1] - 1.0) + l.v * (2.0 * u[2] - 1.0);
        let dv = p - x;
        let d2 = dv.norm_squared();
        let d = dv / d2.sqrt();
        let cl = -d.dot(&l.normal());
        if cl <= 0.0 {
            return None;
        }
        let pdf = d2 / (cl * l.area() * n as f64);
        Some((d, d2.sqrt(), pdf, i as f64))
    }

    fn light_pdf(&self, x: Point3, d: Vec3) -> (f64, f64) {
        // Nearest light hit along `d` and its pdf; `(t, pdf)`.
        let n = self.lights.len() as f64;
        let mut best = (f64::INFINITY, 0.0);
        for l in &self.lights {
            if let Some(t) = l.hit(x, d, best.0) {
                let cl = -d.dot(&l.normal());
                best = (t, t * t / (cl * l.area() * n));
            }
        }
        best
    }

    /// One path: `L` and, when `rec` is given, its lanes.
    pub fn trace(
        &self,
        theta: &[f64; PARAMS],
        seed: u64,
        path: usize,
        mut rec: Option<&mut Vec<Lane>>,
    ) -> (f64, f64) {
        let mut rng = Rng::new(seed, path as u64);
        // Wavelengths stratified over a pixel's samples: the chroma noise of
        // one wavelength per path falls as 1/spp, not 1/√spp.
        let lambda = sample_lambda_nm(((path % self.spp) as f64 + rng.uniform()) / self.spp as f64);
        let px = path / self.spp;
        let (w, h) = self.size;
        let (ix, iy) = ((px % w) as f64, (px / w) as f64);
        let (jx, jy) = (rng.uniform(), rng.uniform());
        let fw = self.width;
        let fh = fw * h as f64 / w as f64;
        let mut o = Point3::new(
            -0.5 * fw + (ix + jx) / w as f64 * fw,
            -5.0,
            0.5 * fh - (iy + jy) / h as f64 * fh,
        );
        let mut d = Vec3::new(0.0, 1.0, 0.0);
        let mat = self.material.with_theta(theta);
        let smat = self.sampler_material(theta);
        let mut beta = 1.0;
        let mut l = 0.0;
        let mut last_pdf: Option<f64> = None; // None: the camera
        for _ in 0..self.depth {
            // Sphere, then lights.
            let b = o.coords().dot(&d);
            let c = o.coords().norm_squared() - 1.0;
            let disc = b * b - c;
            let mut ts = f64::INFINITY;
            if disc > 0.0 {
                let s = disc.sqrt();
                for t in [-b - s, -b + s] {
                    if t > 1e-7 {
                        ts = t;
                        break;
                    }
                }
            }
            let (tl, pl) = self.light_pdf(o, d);
            if tl < ts {
                let i = self
                    .lights
                    .iter()
                    .position(|q| q.hit(o, d, tl * (1.0 + 1e-9)).is_some())
                    .unwrap();
                let le = self.lights[i].le(lambda);
                let wgt = match last_pdf {
                    None => 1.0,
                    Some(pb) => mis(pb, pl),
                };
                l += beta * le * wgt;
                break;
            }
            if ts == f64::INFINITY {
                l += beta * self.env;
                break;
            }
            let x = o + d * ts;
            let n_out = x.coords().normalize();
            let inside = d.dot(&n_out) > 0.0;
            if inside && mat.transmission_depth > 0.0 {
                let tc = super::openpbr::lift(mat.transmission_color, lambda).max(1e-6);
                beta *= (tc.ln() / mat.transmission_depth * ts).exp();
            }
            let n = if inside { -n_out } else { n_out };
            let (tu, tv) = basis(n);
            let local = |v: Vec3| [v.dot(&tu), v.dot(&tv), v.dot(&n)];
            let world = |v: [f64; 3]| tu * v[0] + tv * v[1] + n * v[2];
            let wo = local(-d);
            let at = At::new(&mat, lambda, inside);
            let sat = At::new(&smat, lambda, inside);
            let lw = sat.lobe_weights(wo[2]);
            if !inside && mat.emission_luminance > 0.0 {
                l += beta
                    * mat.emission_luminance
                    * super::openpbr::lift(mat.emission_color, lambda);
            }
            // Light sample.
            let un = [rng.uniform(), rng.uniform(), rng.uniform()];
            if !inside {
                if let Some((dl, _, pl, li)) = self.light_sample(x, un) {
                    let wi = local(dl);
                    if wi[2] > 0.0 {
                        let cols = at.columns(Kind::Reflect, wo, wi);
                        let f = graph::stack(&cols, theta);
                        let pb = sat.pdf(&lw, wo, wi);
                        let cl = beta * f * wi[2] * self.lights[li as usize].le(lambda) / pl
                            * mis(pl, pb);
                        if let Some(r) = rec.as_deref_mut() {
                            r.push(Lane { cols, f, before: -cl, nee: true });
                        }
                        l += cl;
                    }
                }
            }
            let us = [rng.uniform(), rng.uniform(), rng.uniform()];
            match sat.sample(&lw, wo, us) {
                None => {
                    // The subsurface event.
                    let cols = at.columns(Kind::Subsurface, wo, wo);
                    let f = graph::stack(&cols, theta);
                    if let Some(r) = rec.as_deref_mut() {
                        r.push(Lane { cols, f, before: l, nee: false });
                    }
                    beta *= f / lw[4];
                    let bvh = self.walk_mesh.as_ref().expect("walk mesh");
                    let wd = world(super::openpbr::cosine([rng.uniform(), rng.uniform()]));
                    let model = subsurface_medium(&mat, lambda);
                    let samp = subsurface_medium(&smat, lambda);
                    // Inside the tessellated boundary: the analytic hit is
                    // outside the inscribed faces by up to their sagitta
                    // (≈ 2e-4 at level 5), and a walk started there would
                    // leave at once.
                    let start = Point3::origin() + x.coords() * (1.0 - 5e-4);
                    let walk = walk_g(
                        bvh,
                        None,
                        start,
                        -wd,
                        model,
                        samp,
                        1024,
                        &mut rng,
                        mat.subsurface_scatter_anisotropy,
                    );
                    let Some((hit, _, _)) = walk.exit else {
                        break;
                    };
                    beta *= walk.weight;
                    let xe = Point3::origin() + hit.point.coords().normalize() * (1.0 + 1e-7);
                    let ne = xe.coords().normalize();
                    let un = [rng.uniform(), rng.uniform(), rng.uniform()];
                    if let Some((dl, _, pl, li)) = self.light_sample(xe, un) {
                        let c = dl.dot(&ne);
                        if c > 0.0 {
                            let pb = c / PI;
                            l += beta * c / PI * self.lights[li as usize].le(lambda) / pl
                                * mis(pl, pb);
                        }
                    }
                    let (eu, ev) = basis(ne);
                    let cd = super::openpbr::cosine([rng.uniform(), rng.uniform()]);
                    d = eu * cd[0] + ev * cd[1] + ne * cd[2];
                    o = xe;
                    last_pdf = Some(cd[2] / PI);
                    continue;
                }
                Some(wi) => {
                    if wi == [0.0; 3] {
                        break;
                    }
                    let pdf = sat.pdf(&lw, wo, wi);
                    if !(pdf > 0.0) {
                        break;
                    }
                    let kind = if wi[2] > 0.0 { Kind::Reflect } else { Kind::Transmit };
                    if kind == Kind::Transmit && at.m.transmission_weight <= 0.0 {
                        break;
                    }
                    let cols = at.columns(kind, wo, wi);
                    let f = graph::stack(&cols, theta);
                    if let Some(r) = rec.as_deref_mut() {
                        r.push(Lane { cols, f, before: l, nee: false });
                    }
                    beta *= f * wi[2].abs() / pdf;
                    d = world(wi).normalize();
                    o = x + (if wi[2] > 0.0 { n } else { -n }) * 1e-7;
                    // Light hits after a refraction or from inside are not
                    // light-sampled: full weight.
                    last_pdf = if inside || kind == Kind::Transmit {
                        None
                    } else {
                        Some(pdf)
                    };
                }
            }
            if !(beta > 0.0) || !beta.is_finite() {
                break;
            }
        }
        (l, lambda)
    }

    /// The forward pass: `(L, λ)` per path.
    pub fn forward(&self, theta: &[f64; PARAMS], seed: u64) -> Vec<(f64, f64)> {
        (0..self.paths())
            .into_par_iter()
            .map(|i| self.trace(theta, seed, i, None))
            .collect()
    }

    /// Linear sRGB `w·h·3` from the forward pass.
    pub fn image(&self, l: &[(f64, f64)]) -> Vec<f64> {
        let mut img = vec![0.0; self.size.0 * self.size.1 * 3];
        for (j, &(lj, lambda)) in l.iter().enumerate() {
            let w = hero_weight(lambda);
            let px = j / self.spp;
            for c in 0..3 {
                img[px * 3 + c] += lj * w[c] as f64 / self.spp as f64;
            }
        }
        img
    }

    /// Render: forward, then the image.
    pub fn render(&self, theta: &[f64; PARAMS], seed: u64) -> Vec<f64> {
        self.image(&self.forward(theta, seed))
    }

    /// The replay: re-trace every path with records, and reduce
    /// `δL_j · weight` per lane through [`Stack`]'s adjoint (`prog`, its
    /// width). `dl` is `δloss/δL` per path. Returns `(δθ, L per path)`.
    pub fn replay(
        &self,
        theta: &[f64; PARAMS],
        seed: u64,
        dl: &[f64],
        prog: &mut crate::ir::Program,
        width: usize,
    ) -> ([f64; PARAMS], Vec<f64>) {
        let per: Vec<(f64, Vec<Lane>)> = (0..self.paths())
            .into_par_iter()
            .map(|i| {
                let mut r = Vec::new();
                let (l, _) = self.trace(theta, seed, i, Some(&mut r));
                (l, r)
            })
            .collect();
        let mut rows = Vec::new();
        let mut df = Vec::new();
        let mut ls = Vec::with_capacity(per.len());
        for (i, (l, lanes)) in per.into_iter().enumerate() {
            ls.push(l);
            for ln in lanes {
                rows.push(ln.cols);
                df.push(dl[i] * ln.weight(l));
            }
        }
        (graph::reduce(prog, width, &rows, &df, theta), ls)
    }

    /// `δloss/δL` per path from `δloss/δimage`.
    pub fn path_adjoint(&self, l: &[(f64, f64)], d_img: &[f64]) -> Vec<f64> {
        l.iter()
            .enumerate()
            .map(|(j, &(_, lambda))| {
                let w = hero_weight(lambda);
                let px = j / self.spp;
                (0..3)
                    .map(|c| d_img[px * 3 + c] * w[c] as f64 / self.spp as f64)
                    .sum()
            })
            .collect()
    }

    /// `(loss, ∂loss/∂θ)` by PRB.
    pub fn gradient(
        &self,
        theta: &[f64; PARAMS],
        seed: u64,
        loss: &impl crate::prb::Loss,
        native: bool,
    ) -> (f64, [f64; PARAMS]) {
        let fwd = self.forward(theta, seed);
        let (v, d_img) = loss.eval(&self.image(&fwd));
        let dl = self.path_adjoint(&fwd, &d_img);
        let width = 4096;
        let mut prog = Stack::new(width).adjoint(native);
        (v, self.replay(theta, seed, &dl, &mut prog, width).0)
    }
}
