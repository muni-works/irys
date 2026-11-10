//! The landing page's live demos: irys-diff compiled to wasm, on the CPU in
//! f64, one gradient step per call. Only diffuse surfaces and splats for
//! now: their adjoints are hand-written, while the conductor's material
//! graph runs in tang's interpreter, which spawns a thread the browser
//! cannot give it.
//!
//! Each demo renders, scores against a target the page chose, replays for
//! the gradient and takes an Adam step. The scored frame and the replayed
//! frame use different seeds: with one seed the L2 gradient also descends
//! the image's variance, which pulls albedos low.

use irys_core::{Bvh, Geometry, Point3, Transform};
use irys_light::Splats;
use irys_diff::prb::{self, Env, Kernels, L2, Loss, Opaque, Placement, Scene, Surface};
use irys_prims::analytic::Analytic;
use wasm_bindgen::prelude::*;

/// Rust panics reach the browser console as messages, not `unreachable`.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Linear sRGB `w·h·3` to display RGBA bytes.
fn rgba(img: &[f64]) -> Vec<u8> {
    let enc = |x: f64| {
        let x = x.clamp(0.0, 1.0);
        let s = if x <= 0.003_130_8 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
        (s * 255.0 + 0.5) as u8
    };
    img.chunks_exact(3)
        .flat_map(|p| [enc(p[0]), enc(p[1]), enc(p[2]), 255])
        .collect()
}

struct Adam<const N: usize> {
    m: [f64; N],
    v: [f64; N],
    t: i32,
    lr: f64,
}

impl<const N: usize> Adam<N> {
    fn new(lr: f64) -> Self {
        Self { m: [0.0; N], v: [0.0; N], t: 0, lr }
    }
    fn step(&mut self, x: &mut [f64; N], g: &[f64; N], lo: f64, hi: f64) {
        let (b1, b2) = (0.9, 0.99);
        self.t += 1;
        for i in 0..N {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g[i];
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * g[i] * g[i];
            let mh = self.m[i] / (1.0 - b1.powi(self.t));
            let vh = self.v[i] / (1.0 - b2.powi(self.t));
            x[i] = (x[i] - self.lr * mh / (vh.sqrt() + 1e-8)).clamp(lo, hi);
        }
    }
}

/// The adjoint frame's seed for primal seed `seed`.
fn adjoint_seed(seed: u64) -> u64 {
    seed ^ 0x5bd1_e995_0000_0000
}

/// One step's frames: render and score at `seed`, replay at its adjoint
/// seed. `(loss, image, δθ, extra)`.
fn frame(scene: &Scene, theta: &[f64; 4], seed: u64, k: &mut Kernels, target: &[f64]) -> (f64, Vec<f64>, [f64; 4], prb::Extra) {
    let (l, _) = prb::forward(scene, theta, seed, k);
    let img = prb::image(scene, seed, &l);
    let (loss, d_img) = L2(target).eval(&img);
    let a = adjoint_seed(seed);
    let la = prb::forward(scene, theta, a, k).0;
    let dl = prb::lane_adjoint(scene, a, &d_img);
    let (g, extra, _, _) = prb::replay_extra(scene, theta, a, k, &la, &dl);
    (loss, img, g, extra)
}

/// A closed room of diffuse slabs and two boxes under a quad light: no
/// conductor, so a gradient runs only the hand-written diffuse adjoint.
fn room(size: usize, spp: usize) -> Scene {
    let mesh = |m, s| Opaque::new(m, s, Analytic::from_prims(Vec::new()), &[]);
    let slab = |lo: [f64; 3], hi: [f64; 3], tint: f64| {
        let c = |k: usize| 0.5 * (lo[k] + hi[k]);
        let s = |k: usize| hi[k] - lo[k];
        Placement::new(0, Transform::translation(c(0), c(1), c(2)).then(&Transform::scale(s(0), s(1), s(2)))).tinted(tint)
    };
    let t = 0.05;
    let placements = vec![
        slab([-1.0 - t, -1.0, -t], [1.0 + t, 1.0 + t, 0.0], 1.0),
        slab([-1.0 - t, -1.0, 2.0], [1.0 + t, 1.0 + t, 2.0 + t], 1.0),
        slab([-1.0 - t, 1.0, 0.0], [1.0 + t, 1.0 + t, 2.0], 1.0),
        slab([-1.0 - t, -1.0, 0.0], [-1.0, 1.0, 2.0], 0.7),
        slab([1.0, -1.0, 0.0], [1.0 + t, 1.0, 2.0], 0.7),
        Placement::new(0, Transform::translation(-0.35, 0.3, 0.6).then(&Transform::rotation_z(0.3)).then(&Transform::scale(0.6, 0.6, 1.2))).tinted(0.9),
        Placement::new(0, Transform::translation(0.4, -0.3, 0.3).then(&Transform::rotation_z(-0.3)).then(&Transform::scale(0.6, 0.6, 0.6))).tinted(0.9),
        Placement::new(1, Transform::translation(0.0, 0.1, 1.995).then(&Transform::rotation_x(std::f64::consts::PI)).then(&Transform::scale(0.6, 0.5, 1.0))),
    ];
    let mut s = Scene::instanced(
        vec![mesh(prb::cuboid(0.5, 0.5, 0.5), Surface::Diffuse), mesh(prb::grid_floor(1, 0.5), Surface::Emitter)],
        placements,
        None,
        (size, size),
        spp,
        (Point3::new(0.0, -3.4, 1.0), Point3::new(0.0, 0.0, 1.0)),
        19.5,
    );
    s.floor_albedo = 0.75;
    s.nee = true;
    s.complete = true;
    s.depth = 4;
    s.emission = 12.0;
    s.env = Env { sky: 0.0, sun_radiance: 0.0, ..Env::default() };
    s
}

/// Recover the room's paint, a three-band spectral albedo, from a target.
#[wasm_bindgen]
pub struct PaintFit {
    scene: Scene,
    k: Kernels,
    target: Vec<f64>,
    d: [f64; 3],
    adam: Adam<3>,
    seed: u64,
    image: Vec<f64>,
}

#[wasm_bindgen]
impl PaintFit {
    /// `truth = [blue, green, red]` band albedos.
    #[wasm_bindgen(constructor)]
    pub fn new(size: usize, spp: usize, target_spp: usize, truth: &[f64]) -> PaintFit {
        let mut reference = room(size, target_spp);
        reference.diffuse_bands = Some([truth[0], truth[1], truth[2]]);
        let mut kr = Kernels::for_scene(&reference, false);
        let t0 = [0.3, 0.9, 0.7, 0.4];
        let target = prb::image(&reference, 1000, &prb::forward(&reference, &t0, 1000, &mut kr).0);
        let mut scene = room(size, spp);
        scene.diffuse_bands = Some([0.5; 3]);
        let k = Kernels::for_scene(&scene, false);
        PaintFit { scene, k, target, d: [0.5; 3], adam: Adam::new(0.07), seed: 1, image: Vec::new() }
    }

    pub fn step(&mut self) -> f64 {
        self.scene.diffuse_bands = Some(self.d);
        let (loss, img, _, ex) = frame(&self.scene, &[0.3, 0.9, 0.7, 0.4], self.seed, &mut self.k, &self.target);
        self.adam.step(&mut self.d, &ex.diffuse, 0.02, 0.98);
        self.seed += 1;
        self.image = img;
        loss
    }

    /// `[blue, green, red]`.
    pub fn bands(&self) -> Vec<f64> {
        self.d.to_vec()
    }

    pub fn image_rgba(&self) -> Vec<u8> {
        rgba(&self.image)
    }

    pub fn target_rgba(&self) -> Vec<u8> {
        rgba(&self.target)
    }
}

/// A bush of Gaussian splats on a checker floor under the sky.
fn bush_scene(size: usize, spp: usize, cloud: Splats) -> Scene {
    let floor = Opaque::new(prb::grid_floor(8, 6.0), Surface::Diffuse, Analytic::from_prims(Vec::new()), &[]);
    let mut p = Placement::new(0, Transform::identity());
    p.checker = true;
    let mut s = Scene::instanced(
        vec![floor],
        vec![p],
        Some(Bvh::build(cloud)),
        (size, size),
        spp,
        (Point3::new(-1.6, -0.9, 1.5), Point3::new(-1.6, 2.5, 0.85)),
        24.0,
    );
    s.floor_checker = Some(1.0);
    s.depth = 2;
    s
}

const SH_C0: f32 = 0.282_094_79;

/// The target: every leaf turned from green to autumn orange, and the bush
/// a little thinner.
fn autumn(base: &Splats) -> Splats {
    let mut t = base.clone();
    for i in 0..t.len() {
        let c = base.dc(i).map(|d| 0.5 + SH_C0 * d);
        let g = c[1];
        let a = [g * 2.4 + 0.02, g * 0.95, c[2] * 0.6];
        t.set_dc(i, a.map(|x| (x - 0.5) / SH_C0));
        t.set_opacity(i, base.opacity(i) * 0.8);
    }
    t
}

/// Recover every splat's colour and opacity in a bush from a target render.
#[wasm_bindgen]
pub struct SplatFit {
    scene: Scene,
    k: Kernels,
    target: Vec<f64>,
    cloud: Splats,
    m: Vec<f64>,
    v: Vec<f64>,
    t: i32,
    seed: u64,
    image: Vec<f64>,
}

#[wasm_bindgen]
impl SplatFit {
    #[wasm_bindgen(constructor)]
    pub fn new(size: usize, spp: usize, target_spp: usize) -> SplatFit {
        let base = prb::bush(prb::BUSH_SPLATS, 17);
        let reference = bush_scene(size, target_spp, autumn(&base));
        let mut kr = Kernels::for_scene(&reference, false);
        let th = [0.3, 0.9, 0.7, 0.4];
        let target = prb::image(&reference, 1000, &prb::forward(&reference, &th, 1000, &mut kr).0);
        let mut scene = bush_scene(size, spp, base.clone());
        scene.splat_grad = true;
        let k = Kernels::for_scene(&scene, false);
        let n = base.len() * 4;
        SplatFit { scene, k, target, cloud: base, m: vec![0.0; n], v: vec![0.0; n], t: 0, seed: 1, image: Vec::new() }
    }

    /// One Adam step over all splats' DC colours and opacities.
    pub fn step(&mut self) -> f64 {
        let th = [0.3, 0.9, 0.7, 0.4];
        let (loss, img, _, extra) = frame(&self.scene, &th, self.seed, &mut self.k, &self.target);
        let (b1, b2, lr) = (0.9, 0.99, 0.08);
        self.t += 1;
        for (i, g) in extra.splats.iter().enumerate() {
            let mut dc = self.cloud.dc(i);
            let mut op = self.cloud.opacity(i);
            for p in 0..4 {
                let j = 4 * i + p;
                self.m[j] = b1 * self.m[j] + (1.0 - b1) * g[p];
                self.v[j] = b2 * self.v[j] + (1.0 - b2) * g[p] * g[p];
                let mh = self.m[j] / (1.0 - b1.powi(self.t));
                let vh = self.v[j] / (1.0 - b2.powi(self.t));
                let d = (lr * mh / (vh.sqrt() + 1e-8)) as f32;
                if p < 3 {
                    dc[p] -= d * 3.0;
                } else {
                    op = (op - d * 0.5).clamp(0.02, 0.99);
                }
            }
            self.cloud.set_dc(i, dc);
            self.cloud.set_opacity(i, op);
        }
        self.scene.set_splats(Some(self.cloud.clone()));
        self.seed += 1;
        self.image = img;
        loss
    }

    pub fn image_rgba(&self) -> Vec<u8> {
        rgba(&self.image)
    }

    pub fn target_rgba(&self) -> Vec<u8> {
        rgba(&self.target)
    }
}

/// The paint room, progressively: one frame per call, the running mean
/// shown. All diffuse, so it needs no material graph.
#[wasm_bindgen]
pub struct RoomRender {
    scene: Scene,
    k: Kernels,
    sum: Vec<f64>,
    frames: u32,
}

#[wasm_bindgen]
impl RoomRender {
    #[wasm_bindgen(constructor)]
    pub fn new(size: usize, spp: usize) -> RoomRender {
        let mut scene = room(size, spp);
        scene.depth = 6;
        scene.diffuse_bands = Some([0.6, 0.55, 0.15]);
        let k = Kernels::for_scene(&scene, false);
        RoomRender { sum: vec![0.0; size * size * 3], scene, k, frames: 0 }
    }

    /// Repaint the room, `[blue, green, red]`, and start over.
    pub fn set_paint(&mut self, b: f64, g: f64, r: f64) {
        self.scene.diffuse_bands = Some([b, g, r]);
        self.sum.iter_mut().for_each(|x| *x = 0.0);
        self.frames = 0;
    }

    pub fn frame(&mut self) -> u32 {
        self.frames += 1;
        let seed = self.frames as u64;
        let (l, _) = prb::forward(&self.scene, &[0.3, 0.9, 0.7, 0.4], seed, &mut self.k);
        for (s, v) in self.sum.iter_mut().zip(prb::image(&self.scene, seed, &l)) {
            *s += v;
        }
        self.frames
    }

    pub fn image_rgba(&self) -> Vec<u8> {
        let n = self.frames.max(1) as f64;
        rgba(&self.sum.iter().map(|x| x / n).collect::<Vec<_>>())
    }
}
