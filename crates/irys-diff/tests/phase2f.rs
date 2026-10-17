//! Phase 2f: the secondary edge term ([`irys_diff::secondary`],
//! [`irys_diff::interior::interior_with`]) — shadows and reflections moving —
//! with 2e's edge and interior terms, against central differences of the
//! fully shaded image. The scenes are 2e's (`tests/phase2e.rs`), copied.

use irys_core::{Point3, Transform, Vec3};
use irys_diff::ggx::PARAMS;
use irys_diff::interior::{self, Tangents};
use irys_diff::prb::{self, Kernels, Loss, Opaque, Placement, Scene, Surface};
use irys_diff::secondary::Secondary;
use irys_diff::vis::{self, Edges, Rig};
use irys_prims::analytic::Analytic;

const THETA: [f64; PARAMS] = [0.3, 0.9, 0.8, 0.5];

/// `Σ g · I` with a smooth `g`, different per channel.
struct Weighted(Vec<f64>);

impl Weighted {
    fn new(w: usize, h: usize) -> Self {
        let tau = std::f64::consts::TAU;
        Self(
            (0..w * h * 3)
                .map(|i| {
                    let (px, c) = (i / 3, i % 3);
                    let (x, y) = ((px % w) as f64 / w as f64, (px / w) as f64 / h as f64);
                    1.0 + 0.6 * (tau * x + c as f64).sin() * (3.0 * y + 0.5 * c as f64).cos()
                })
                .collect(),
        )
    }
}

impl Loss for Weighted {
    fn eval(&self, img: &[f64]) -> (f64, Vec<f64>) {
        (
            img.iter().zip(&self.0).map(|(a, b)| a * b).sum(),
            self.0.clone(),
        )
    }
}

fn opaque(m: irys_core::TriMesh, s: Surface) -> Opaque {
    Opaque::new(m, s, Analytic::from_prims(Vec::new()), &[])
}

const SHOULDER: [f64; 3] = [-0.9, 0.3, 1.25];
const UPPER: f64 = 0.55;
const FORE: f64 = 0.45;
const GAP: f64 = 0.15;

/// An arm on a floor: a static post, a conductor upper link and a diffuse
/// forearm (boxes, each pre-rolled about its own axis so no face is
/// axis-aligned), NEE on, depth `depth`.
fn arm_scene(w: usize, h: usize, spp: usize, depth: usize) -> Scene {
    arm_scene_with(w, h, spp, depth, true)
}

fn arm_scene_with(w: usize, h: usize, spp: usize, depth: usize, floor: bool) -> Scene {
    let blas = vec![
        opaque(
            if floor {
                prb::grid_floor(8, 6.0)
            } else {
                // Far below and small: out of every path's way.
                prb::cuboid(0.01, 0.01, 0.01)
            },
            Surface::Diffuse,
        ),
        opaque(prb::cuboid(0.12, 0.12, 0.62), Surface::Diffuse),
        opaque(prb::cuboid(UPPER, 0.1, 0.1), Surface::Conductor),
        opaque(prb::cuboid(FORE, 0.085, 0.085), Surface::Diffuse),
    ];
    let placements = vec![
        Placement::new(
            0,
            Transform::translation(0.0, 0.0, if floor { 0.0 } else { -50.0 }),
        )
        .tinted(1.0),
        Placement::new(
            1,
            Transform::translation(SHOULDER[0], SHOULDER[1] + 0.3, 0.62),
        )
        .tinted(0.6),
        Placement::new(2, Transform::identity()),
        Placement::new(3, Transform::identity()).tinted(0.8),
    ];
    let mut s = Scene::instanced(
        blas,
        placements,
        None,
        (w, h),
        spp,
        (Point3::new(0.0, -4.5, 2.2), Point3::new(0.0, 0.0, 0.8)),
        20.0,
    );
    s.nee = true;
    s.depth = depth;
    s
}

/// Shoulder pitch (about `y`), shoulder yaw (about `z`), elbow (about the
/// upper link's `y`).
fn arm_rig() -> Rig<'static> {
    arm_rig_gap(GAP)
}

fn arm_rig_gap(gap: f64) -> Rig<'static> {
    Rig {
        moving: vec![2, 3],
        place: Box::new(move |p: &[f64]| {
            let s = Point3::new(SHOULDER[0], SHOULDER[1], SHOULDER[2]);
            let sh = vis::about(s, Vec3::new(0.0, 0.0, 1.0), p[1]).then(&vis::about(
                s,
                Vec3::new(0.0, 1.0, 0.0),
                p[0],
            ));
            // Gaps at both joints: no two boxes interpenetrate at any pose
            // near PHI (an intersection curve is a discontinuity that is
            // no triangle's edge, and the edge term cannot see it).
            let e = Point3::new(
                SHOULDER[0] + 2.0 * UPPER + 2.0 * gap,
                SHOULDER[1],
                SHOULDER[2],
            );
            let roll = |a: f64| Transform::rotation_x(a);
            let upper = sh
                .then(&Transform::translation(
                    SHOULDER[0] + UPPER + gap,
                    SHOULDER[1],
                    SHOULDER[2],
                ))
                .then(&roll(0.35));
            let fore = sh
                .then(&vis::about(e, Vec3::new(0.0, 1.0, 0.0), p[2]))
                .then(&Transform::translation(e.x + FORE + gap, e.y, e.z))
                .then(&roll(-0.5));
            vec![upper, fore]
        }),
    }
}

const PHI: [f64; 3] = [0.35, 0.45, -0.7];


#[derive(Debug)]
struct Estimate {
    mean: Vec<f64>,
    stderr: Vec<f64>,
}

fn stats(runs: &[Vec<f64>]) -> Estimate {
    let n = runs.len() as f64;
    let np = runs[0].len();
    let mean: Vec<f64> = (0..np)
        .map(|j| runs.iter().map(|r| r[j]).sum::<f64>() / n)
        .collect();
    let stderr = (0..np)
        .map(|j| {
            (runs.iter().map(|r| (r[j] - mean[j]).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt()
        })
        .collect();
    Estimate { mean, stderr }
}

/// The gate: on the shaded arm (depth 4, NEE, a conductor and two diffuse
/// links over a floor), edge + interior against central differences of the
/// expected image, each averaged over independent seeds.

/// `Σ g I` with every pixel integrated over `k²` R2 points.
fn grid_loss(scene: &mut Scene, k: usize, g: &Weighted, seed: u64, kern: &mut Kernels) -> f64 {
    let (w, h) = scene.size;
    let mut pts = Vec::with_capacity(w * h * k * k);
    for py in 0..h {
        for px in 0..w {
            for j in 0..k * k {
                let (a1, a2) = (0.754_877_666_246_692_8, 0.569_840_290_998_053_2);
                // Cranley–Patterson shifted per seed (2f): the points'
                // discreteness is then variance across seeds, not a bias
                // every seed shares.
                let mut r = prb::Rng::new(seed, 0x6772_6964);
                let (s0, s1) = (r.uniform(), r.uniform());
                let u = (s0 + a1 * j as f64).fract();
                let v = (s1 + a2 * j as f64).fract();
                pts.push(scene.screen(px as f64 + u, py as f64 + v));
            }
        }
    }
    scene.set_points(Some(pts));
    let (l, _) = prb::forward(scene, &THETA, seed, kern);
    scene.set_points(None);
    let nw = scene.wavelengths;
    let mut loss = 0.0;
    for (lane, lj) in l.iter().enumerate() {
        let i = lane / nw;
        let lam = prb::wavelengths(prb::Rng::new(seed, (i / 2) as u64).uniform(), nw)[lane % nw];
        let hw = irys_light::spectrum::hero_weight(lam);
        let px = i / (k * k);
        for c in 0..3 {
            loss += lj * hw[c] as f64 / (nw * k * k) as f64 * g.0[px * 3 + c];
        }
    }
    loss
}

/// The three terms on one scene, `seeds` independent estimates.
fn terms(
    s: &mut Scene,
    rig: &Rig,
    g: &Weighted,
    sec: &Secondary,
    seeds: u64,
) -> (Estimate, Estimate, Estimate, Estimate, std::time::Duration) {
    let edges = Edges::new(s);
    let tan = Tangents::rig(s, rig, &PHI);
    let n_edge = 8192;
    let mut ke = Kernels::new(2 * 2 * n_edge * s.wavelengths, true);
    let (mut er, mut ir, mut sr, mut tot) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut t_sec = std::time::Duration::ZERO;
    for sd in 0..seeds {
        let (ge, _) = vis::boundary(s, &edges, rig, &PHI, &g.0, n_edge, 500 + sd, |sc| {
            prb::forward(sc, &THETA, 500 + sd, &mut ke).0
        });
        let t0 = std::time::Instant::now();
        let (gi, gs) = interior::interior_with(s, &tan, &THETA, 900 + sd, &g.0, Some(sec));
        t_sec += t0.elapsed();
        tot.push((0..3).map(|j| ge[j] + gi[j] + gs[j]).collect::<Vec<_>>());
        er.push(ge);
        ir.push(gi);
        sr.push(gs);
    }
    (stats(&er), stats(&ir), stats(&sr), stats(&tot), t_sec)
}

fn cd_rig(s: &mut Scene, rig: &Rig, g: &Weighted) -> Estimate {
    let (w, h) = s.size;
    let k = 16;
    let mut kf = Kernels::new(w * h * k * k * s.wavelengths * 2, true);
    let hh = 0.01;
    let mut cdr = Vec::new();
    let cds: u64 = std::env::var("CD_SEEDS").map_or(8, |v| v.parse().unwrap());
    for sd in 0..cds {
        let mut r = Vec::new();
        for j in 0..3 {
            let mut side = |d: f64| {
                let mut p = PHI.to_vec();
                p[j] += d;
                rig.pose(s, &p);
                grid_loss(s, k, g, 7000 + sd, &mut kf)
            };
            let (a, b) = (side(hh), side(-hh));
            r.push((a - b) / (2.0 * hh));
        }
        cdr.push(r);
    }
    rig.pose(s, &PHI);
    stats(&cdr)
}

fn report(name: &str, e: &Estimate, i: &Estimate, sc: &Estimate, t: &Estimate, cd: &Estimate) -> Vec<f64> {
    (0..t.mean.len())
        .map(|j| {
            let z = (t.mean[j] - cd.mean[j]) / t.stderr[j].hypot(cd.stderr[j]);
            let z2 = (t.mean[j] - sc.mean[j] - cd.mean[j])
                / (t.stderr[j].powi(2) - sc.stderr[j].powi(2)).max(0.0).sqrt().hypot(cd.stderr[j]);
            eprintln!(
                "{name} φ{j}: edge {:+.3} ± {:.3}, interior {:+.3} ± {:.3}, secondary {:+.3} ± {:.3}, sum {:+.3} ± {:.3}; CD {:+.3} ± {:.3}; z {z:+.2} (2e alone z {z2:+.2})",
                e.mean[j], e.stderr[j], i.mean[j], i.stderr[j], sc.mean[j], sc.stderr[j],
                t.mean[j], t.stderr[j], cd.mean[j], cd.stderr[j]
            );
            z
        })
        .collect()
}

/// 2e's ignored gate, now gated: the arm over a floor it shadows (depth 4,
/// NEE). Edge + interior + secondary against central differences.
#[test]
fn shaded_arm_on_a_floor_with_secondary_edges() {
    let (w, h) = (48, 36);
    let rig = arm_rig();
    let g = Weighted::new(w, h);
    let mut s = arm_scene_with(w, h, 4, 4, true);
    rig.pose(&mut s, &PHI);
    let sec = Secondary::new(&s);
    let seeds = std::env::var("SEEDS").map_or(24, |v| v.parse().unwrap());
    let (e, i, sc, t, ts) = terms(&mut s, &rig, &g, &sec, seeds);
    let cd = cd_rig(&mut s, &rig, &g);
    eprintln!("interior + secondary: {:.1?} per seed", ts / seeds as u32);
    let zs = report("floor", &e, &i, &sc, &t, &cd);
    for (j, z) in zs.iter().enumerate() {
        assert!(z.abs() < 3.0, "φ{j}: z {z}");
    }
}

/// No floor (2e's gated scene): the secondary term must not break it.
#[test]
fn shaded_arm_with_secondary_edges() {
    let (w, h) = (48, 36);
    let rig = arm_rig();
    let g = Weighted::new(w, h);
    let mut s = arm_scene_with(w, h, 4, 4, false);
    rig.pose(&mut s, &PHI);
    let sec = Secondary::new(&s);
    let seeds = std::env::var("SEEDS").map_or(24, |v| v.parse().unwrap());
    let (e, i, sc, t, _) = terms(&mut s, &rig, &g, &sec, seeds);
    let cd = cd_rig(&mut s, &rig, &g);
    let zs = report("sky", &e, &i, &sc, &t, &cd);
    for (j, z) in zs.iter().enumerate() {
        assert!(z.abs() < 3.0, "φ{j}: z {z}");
    }
}

/// 2e's ignored camera gate on the shaded arm (its own shadows move on
/// screen): all seven camera parameters.
#[test]
fn camera_gradient_of_the_shaded_arm_with_secondary_edges() {
    let rig = arm_rig();
    let depth = std::env::var("DEPTH").map_or(4, |v| v.parse().unwrap());
    let mut s = arm_scene_with(48, 36, 4, depth, false);
    rig.pose(&mut s, &PHI);
    let (w, h) = s.size;
    let g = Weighted::new(w, h);
    let cam = vis::CameraRig::new(&s);
    let c = [0.0; 7];
    let edges = Edges::new(&s);
    let sec = Secondary::new(&s);
    let tan = Tangents::camera(&s, &cam, &c);
    let n = 8192;
    let mut ke = Kernels::new(4 * n * s.wavelengths, true);
    let (mut er, mut ir, mut sr, mut tot) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let seeds = std::env::var("SEEDS").map_or(24, |v| v.parse().unwrap());
    for sd in 0..seeds {
        let es = vis::EdgeSamples::for_camera(&s, &edges, &cam, &c, n, 600 + sd);
        s.set_points(Some(es.points().to_vec()));
        let l = prb::forward(&s, &THETA, 600 + sd, &mut ke).0;
        s.set_points(None);
        let ge = es.gradient(&s, 600 + sd, &l, &g.0);
        let (gi, gs) = interior::interior_with(&s, &tan, &THETA, 1600 + sd, &g.0, Some(&sec));
        tot.push((0..7).map(|j| ge[j] + gi[j] + gs[j]).collect::<Vec<_>>());
        er.push(ge);
        ir.push(gi);
        sr.push(gs);
    }
    let (e, i, sc, t) = (stats(&er), stats(&ir), stats(&sr), stats(&tot));
    let k = 16;
    let mut kf = Kernels::new(w * h * k * k * s.wavelengths * 2, true);
    let rs: f64 = std::env::var("ROT").map_or(0.004, |v| v.parse().unwrap());
    let steps = [0.05, 0.05, 0.05, rs, rs, rs, 0.01];
    let mut cdr = Vec::new();
    let cds: u64 = std::env::var("CD_SEEDS").map_or(8, |v| v.parse().unwrap());
    for sd in 0..cds {
        let mut r = Vec::new();
        for j in 0..7 {
            let mut side = |d: f64| {
                let mut p = c.to_vec();
                p[j] += d;
                cam.pose(&mut s, &p);
                grid_loss(&mut s, k, &g, 8000 + sd, &mut kf)
            };
            let (a, b) = (side(steps[j]), side(-steps[j]));
            r.push((a - b) / (2.0 * steps[j]));
        }
        cdr.push(r);
    }
    cam.pose(&mut s, &c);
    let cd = stats(&cdr);
    let zs = report("camera", &e, &i, &sc, &t, &cd);
    for (j, z) in zs.iter().enumerate() {
        assert!(z.abs() < 3.0, "c{j}: z {z}");
    }
}

/// The method choice, measured: on the floor arm, the light sample's share
/// of the secondary boundary term by edge sampling and by warped-area
/// reparameterisation (several auxiliary-ray budgets), same paths, same
/// seeds. Both estimate the same thing; the table is mean ± stderr and
/// time. Measured, not gated (`--ignored`).
#[test]
#[ignore]
fn secondary_estimators_compared() {
    use irys_diff::interior::SecMode;
    let (w, h) = (48, 36);
    let rig = arm_rig();
    let g = Weighted::new(w, h);
    let mut s = arm_scene_with(w, h, 4, 4, true);
    rig.pose(&mut s, &PHI);
    let sec = Secondary::new(&s);
    let tan = Tangents::rig(&s, &rig, &PHI);
    let seeds: u64 = std::env::var("SEEDS").map_or(24, |v| v.parse().unwrap());
    let modes = [
        ("interior alone", None),
        ("edges, light share", Some(SecMode::EdgesNee)),
        ("edges, both", Some(SecMode::Edges)),
        ("warp K=4 κ=400", Some(SecMode::WarpNee { k: 4, kappa: 400.0 })),
        ("warp K=8 κ=400", Some(SecMode::WarpNee { k: 8, kappa: 400.0 })),
        ("warp K=8 κ=100", Some(SecMode::WarpNee { k: 8, kappa: 100.0 })),
        ("warp K=8 κ=1600", Some(SecMode::WarpNee { k: 8, kappa: 1600.0 })),
        ("warp K=32 κ=400", Some(SecMode::WarpNee { k: 32, kappa: 400.0 })),
    ];
    for (name, mode) in modes {
        let t0 = std::time::Instant::now();
        let runs: Vec<Vec<f64>> = (0..seeds)
            .map(|sd| {
                let (gi, gs) = match mode {
                    None => (interior::interior(&s, &tan, &THETA, 900 + sd, &g.0), vec![0.0; 3]),
                    Some(m) => interior::interior_mode(&s, &tan, &THETA, 900 + sd, &g.0, Some(&sec), m),
                };
                (0..3).map(|j| gi[j] + gs[j]).collect()
            })
            .collect();
        let dt = t0.elapsed() / seeds as u32;
        let e = stats(&runs);
        eprintln!(
            "{name:20} {:+.3} ± {:.3}  {:+.3} ± {:.3}  {:+.3} ± {:.3}   {dt:.1?} per estimate",
            e.mean[0], e.stderr[0], e.mean[1], e.stderr[1], e.mean[2], e.stderr[2]
        );
    }
}
