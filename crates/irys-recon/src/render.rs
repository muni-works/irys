//! Targets and model images from irys: the part in a studio, grey diffuse,
//! seen by a few known pinholes.
//!
//! The renderer is irys-diff's CPU path tracer ([`prb::forward`]): the
//! default sky gradient and sun lobe with NEE, two bounces, one hero
//! wavelength per path, no floor (the background is the sky, so the
//! silhouette and the view through every through-hole carry signal). A
//! frame is reduced to grey (the mean of its linear sRGB channels).

use std::collections::HashMap;

use irys_core::{Point3, Ray, Transform, Vec3};
use irys_diff::ggx::PARAMS;
use irys_diff::prb::{self, Kernels, Opaque, Placement, Scene, Surface};
use irys_prims::analytic::Analytic;

use crate::geom;

/// The part's diffuse albedo.
pub const ALBEDO: f64 = 0.6;
/// The conductor's θ: unused (the part is diffuse), but the graph wants it.
pub const THETA: [f64; PARAMS] = [0.5, 0.5, 0.5, 0.5];

/// A known camera.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct View {
    /// Eye, millimetres.
    pub eye: [f64; 3],
    /// Look-at point.
    pub target: [f64; 3],
    /// Half vertical field of view, degrees.
    pub half_fov: f64,
}

impl View {
    /// On a sphere of radius `dist` about `target`, azimuth and elevation in
    /// degrees.
    pub fn orbit(az: f64, el: f64, dist: f64, half_fov: f64) -> Self {
        let (a, e) = (az.to_radians(), el.to_radians());
        let target = [0.0, 0.0, 6.0];
        Self {
            eye: [
                target[0] + dist * e.cos() * a.cos(),
                target[1] + dist * e.cos() * a.sin(),
                target[2] + dist * e.sin(),
            ],
            target,
            half_fov,
        }
    }
}

/// The v0 rig: three views. View 0 alone is the single-view setting.
pub fn views() -> Vec<View> {
    vec![
        View::orbit(-70.0, 58.0, 215.0, 15.0),
        View::orbit(20.0, 28.0, 215.0, 15.0),
        View::orbit(145.0, 72.0, 215.0, 15.0),
    ]
}

/// Image size and paths per pixel.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Film {
    /// Width, height.
    pub size: (usize, usize),
    /// Paths per pixel.
    pub spp: usize,
}

/// A scene of `mesh` under `view`.
pub fn scene(mesh: Option<&(Vec<Point3>, Vec<[u32; 3]>)>, view: &View, film: Film) -> Scene {
    // An empty part still needs a tree: a speck far below the floor.
    let tm = match mesh {
        Some((p, t)) if !t.is_empty() => geom::trimesh(p, t),
        _ => geom::trimesh(
            &[Point3::new(0.0, 0.0, -1e5), Point3::new(1e-3, 0.0, -1e5), Point3::new(0.0, 1e-3, -1e5)],
            &[[0, 1, 2]],
        ),
    };
    let o = Opaque::new(tm, Surface::Diffuse, Analytic::from_prims(Vec::new()), &[]);
    let p = Placement::new(0, Transform::identity()).tinted(ALBEDO);
    let e = view.eye;
    let t = view.target;
    let mut s = Scene::instanced(
        vec![o],
        vec![p],
        None,
        film.size,
        film.spp,
        (Point3::new(e[0], e[1], e[2]), Point3::new(t[0], t[1], t[2])),
        view.half_fov,
    );
    s.nee = true;
    s.depth = 2;
    s.wavelengths = irys_diff::prb::HERO;
    s.floor_albedo = ALBEDO;
    s.env = studio();
    s
}

/// The studio: the sky gradient and one broad key light (angular radius
/// ≈ 1/√κ ≈ 24°), so faces are told apart by shading while the shadows
/// stay soft. A sharp sun would put hard shadow edges on the part, which
/// move with θ and which the primary edge term does not see (2f's
/// secondary edges would).
pub fn studio() -> irys_diff::prb::Env {
    irys_diff::prb::Env {
        sun: Vec3::new(-0.45, -0.55, 0.7).normalize(),
        kappa: 6.0,
        sun_radiance: 4.0,
        sky: 0.8,
        pick: None,
    }
}

/// Compiled kernels by graph width: building one compiles C, so they are
/// kept.
#[derive(Default)]
pub struct KernelCache(HashMap<usize, Kernels>);

impl KernelCache {
    /// Kernels for `scene`.
    pub fn get(&mut self, scene: &Scene) -> &mut Kernels {
        let w = scene.graph_width();
        self.0.entry(w).or_insert_with(|| Kernels::new(w, true))
    }
}

/// The grey image (`w·h`) of `scene` drawn with `seed`.
pub fn grey(scene: &Scene, seed: u64, k: &mut KernelCache) -> Vec<f64> {
    let (l, _) = prb::forward(scene, &THETA, seed, k.get(scene));
    let rgb = prb::image(scene, seed, &l);
    rgb.chunks_exact(3).map(|c| (c[0] + c[1] + c[2]) / 3.0).collect()
}

/// `L` per path for `scene` in points mode (one wavelength per path).
pub fn trace_points(scene: &Scene, seed: u64, k: &mut KernelCache) -> Vec<f64> {
    prb::forward(scene, &THETA, seed, k.get(scene)).0
}

/// Coverage of the part per pixel from `sub × sub` pinhole rays (mask, for
/// the silhouette term and for views of the target's background).
pub fn coverage(scene: &Scene, sub: usize) -> Vec<f64> {
    let (w, h) = scene.size;
    let (eye, f, r, u, _) = scene.camera();
    let mut out = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut hit = 0;
            for j in 0..sub {
                for i in 0..sub {
                    let [sx, sy] = scene.screen(
                        x as f64 + (i as f64 + 0.5) / sub as f64,
                        y as f64 + (j as f64 + 0.5) / sub as f64,
                    );
                    let ray = Ray::new(eye, f + r * sx + u * sy);
                    if scene.tlas().trace_closest(&ray).is_some() {
                        hit += 1;
                    }
                }
            }
            out[y * w + x] = hit as f64 / (sub * sub) as f64;
        }
    }
    out
}

/// Whether the camera ray through camera-plane point `s` hits triangle
/// `tri` or one coplanar with it (same plane within `tol`).
pub fn hits_plane(scene: &Scene, s: [f64; 2], plane: (Vec3, f64), tol: f64) -> Option<bool> {
    let (eye, f, r, u, _) = scene.camera();
    let ray = Ray::new(eye, f + r * s[0] + u * s[1]);
    let h = scene.tlas().trace_closest(&ray)?;
    let p = h.hit.point;
    Some((plane.0.dot(&p.coords()) - plane.1).abs() < tol)
}

/// Linear grey to 8-bit sRGB.
pub fn srgb8(x: f64) -> u8 {
    let x = x.clamp(0.0, 1.0);
    let s = if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}
