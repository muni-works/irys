//! Phase 2f: the secondary edge term ([`irys_diff::secondary`],
//! [`irys_diff::interior::interior_with`]) — shadows and reflections moving —
//! with 2e's edge and interior terms, against central differences of the
//! fully shaded image. The scenes are 2e's (`tests/phase2e.rs`), copied.

mod common;

use common::paper;
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
            if name == "floor" {
                paper(&format!("sec.floor_z{}", j + 1), z);
                paper(&format!("sec.no_sec_z{}", j + 1), z2);
            }
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
    let mut se: Vec<(&str, Vec<f64>)> = Vec::new();
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
        se.push((name, e.stderr.clone()));
    }
    // The warp's best setting (K=8, κ=1600) against edge sampling's light
    // share: the smallest per-φ variance ratio, the paper's lower bound.
    let find = |n: &str| &se.iter().find(|(m, _)| *m == n).unwrap().1;
    let (warp, edge) = (find("warp K=8 κ=1600"), find("edges, light share"));
    paper(
        "sec.var_vs_warp",
        (0..3).map(|j| (warp[j] / edge[j]).powi(2)).fold(f64::INFINITY, f64::min),
    );
}

/// Piece 2: links sunk into each other (negative gaps at both joints) —
/// the intersection curves are discontinuities no triangle owns. Edge +
/// interior + secondary + curve samples against central differences.
#[test]
fn interpenetrating_arm_with_curve_samples() {
    let (w, h) = (48, 36);
    let rig = arm_rig_gap(-0.06);
    let g = Weighted::new(w, h);
    let mut s = arm_scene_with(w, h, 4, 4, false);
    rig.pose(&mut s, &PHI);
    let edges = Edges::new(&s);
    let segs = vis::curves(&s, &edges, &rig.moving);
    assert!(!segs.is_empty(), "the links should interpenetrate");
    let sec = Secondary::new(&s);
    let seeds: u64 = std::env::var("SEEDS").map_or(24, |v| v.parse().unwrap());
    let (e, i, sc, t, _) = terms(&mut s, &rig, &g, &sec, seeds);
    let n = 8192;
    let mut kc = Kernels::new(2 * 2 * n * s.wavelengths, true);
    let cr: Vec<Vec<f64>> = (0..seeds)
        .map(|sd| {
            let cs = vis::EdgeSamples::for_curves(&s, &rig, &PHI, &segs, n, 300 + sd);
            s.set_points(Some(cs.points().to_vec()));
            let l = prb::forward(&s, &THETA, 300 + sd, &mut kc).0;
            s.set_points(None);
            cs.gradient(&s, 300 + sd, &l, &g.0)
        })
        .collect();
    let c = stats(&cr);
    let cd = cd_rig(&mut s, &rig, &g);
    let without = report("sunk, no curves", &e, &i, &sc, &t, &cd);
    let tot: Vec<f64> = (0..3).map(|j| t.mean[j] + c.mean[j]).collect();
    for j in 0..3 {
        let sig = t.stderr[j].hypot(c.stderr[j]).hypot(cd.stderr[j]);
        let z = (tot[j] - cd.mean[j]) / sig;
        eprintln!(
            "sunk φ{j}: curves {:+.3} ± {:.3}, all {:+.3} vs CD {:+.3}: z {z:+.2} (without curves z {:+.2}; {} pieces)",
            c.mean[j], c.stderr[j], tot[j], cd.mean[j], without[j], segs.len()
        );
        assert!(z.abs() < 3.0, "φ{j}: z {z}");
    }
}

/// A closed cylinder along `x`, radius `r`, half-length `hl`, `n` side
/// facets (long, thin and, at the silhouette, edge-on: rose's tubes).
fn tube(r: f64, hl: f64, n: usize) -> irys_core::TriMesh {
    let tau = std::f64::consts::TAU;
    let mut p = Vec::new();
    for x in [-hl, hl] {
        for j in 0..n {
            let a = tau * j as f64 / n as f64;
            p.push(Point3::new(x, r * a.cos(), r * a.sin()));
        }
    }
    p.push(Point3::new(-hl, 0.0, 0.0));
    p.push(Point3::new(hl, 0.0, 0.0));
    let (c0, c1) = (2 * n as u32, 2 * n as u32 + 1);
    let mut idx = Vec::new();
    for j in 0..n as u32 {
        let k = (j + 1) % n as u32;
        let (a, b, c, d) = (j, k, n as u32 + k, n as u32 + j);
        idx.extend([a, b, c, a, c, d]);
        idx.extend([c0, k, j]);
        idx.extend([c1, n as u32 + j, n as u32 + k]);
    }
    irys_core::TriMesh::new(p, Vec::new(), &idx)
}

const TUBE_N: usize = 192;

/// A box of half-extents `h` tessellated as a CAD kernel does a planar
/// face: each face's outline carries its own number of points per side
/// (`k`, `k + 7`, …: T-junctions along every edge of the box, which only
/// meet in exact arithmetic) and is fanned from one corner — slivers, and zero-area triangles along the two sides through
/// the fan's corner (three collinear points), which are silhouette
/// candidates like any other edge.
fn cad_box(h: [f64; 3], k: usize) -> irys_core::TriMesh {
    let mut pos: Vec<Point3> = Vec::new();
    let mut key = std::collections::HashMap::new();
    let mut id = |p: Point3, pos: &mut Vec<Point3>| -> u32 {
        let q = [p.x, p.y, p.z].map(|c| (c * 1e6).round() as i64);
        *key.entry(q).or_insert_with(|| {
            pos.push(p);
            pos.len() as u32 - 1
        })
    };
    let mut idx = Vec::new();
    // Each face: normal axis a, sign sg; u, v the other two axes.
    for a in 0..3 {
        for sg in [-1.0, 1.0] {
            let (u, v) = ((a + 1) % 3, (a + 2) % 3);
            let corner = |cu: f64, cv: f64| {
                let mut c = [0.0; 3];
                c[a] = sg * h[a];
                c[u] = cu * h[u];
                c[v] = cv * h[v];
                Point3::new(c[0], c[1], c[2])
            };
            // The outline, counter-clockwise in (u, v), k points per side.
            let mut ring = Vec::new();
            let sides = [((-1.0, -1.0), (1.0, -1.0)), ((1.0, -1.0), (1.0, 1.0)), ((1.0, 1.0), (-1.0, 1.0)), ((-1.0, 1.0), (-1.0, -1.0))];
            let kf = k + 7 * (2 * a + (sg > 0.0) as usize);
            for ((u0, v0), (u1, v1)) in sides {
                for i in 0..kf {
                    let f = i as f64 / kf as f64;
                    ring.push(id(corner(u0 + f * (u1 - u0), v0 + f * (v1 - v0)), &mut pos));
                }
            }
            for i in 1..ring.len() - 1 {
                let (p, q) = (ring[i], ring[i + 1]);
                // Outward: (u, v, a) is right-handed, so ccw faces +a.
                if sg > 0.0 { idx.extend([ring[0], p, q]); } else { idx.extend([ring[0], q, p]); }
            }
        }
    }
    irys_core::TriMesh::new(pos, Vec::new(), &idx)
}

/// Rose's scale: millimetres, a metre and a half from the origin, seen
/// from 1.7 m through an 11° lens. Two tubes on a shoulder over a post.
fn mm_tube_arm(w: usize, h: usize) -> (Scene, Rig<'static>) {
    const SH: [f64; 3] = [120.0, 240.0, 1400.0];
    let blas = vec![
        opaque(prb::cuboid(40.0, 40.0, 200.0), Surface::Diffuse),
        opaque(cad_box([70.0, 22.0, 16.0], 24), Surface::Conductor),
        opaque(tube(17.0, 60.0, TUBE_N), Surface::Diffuse),
    ];
    let placements = vec![
        Placement::new(0, Transform::translation(SH[0] - 60.0, SH[1] + 70.0, SH[2] - 200.0)).tinted(0.6),
        Placement::new(1, Transform::identity()),
        Placement::new(2, Transform::identity()).tinted(0.8),
    ];
    let mut s = Scene::instanced(
        blas,
        placements,
        None,
        (w, h),
        1,
        (Point3::new(1074.0, 1526.0, 1021.0), Point3::new(170.0, 235.0, 1390.0)),
        11.0,
    );
    s.depth = 1;
    let rig = Rig {
        moving: vec![1, 2],
        place: Box::new(|p: &[f64]| {
            let s = Point3::new(SH[0], SH[1], SH[2]);
            let sh = vis::about(s, Vec3::new(0.0, 0.0, 1.0), p[1]).then(&vis::about(s, Vec3::new(0.0, 1.0, 0.0), p[0]));
            let e = Point3::new(SH[0] + 150.0, SH[1], SH[2]);
            let upper = sh.then(&Transform::translation(SH[0] + 72.0, SH[1], SH[2])).then(&Transform::rotation_x(0.35));
            let fore = sh
                .then(&vis::about(e, Vec3::new(0.0, 1.0, 0.0), p[2]))
                .then(&Transform::translation(e.x + 65.0, e.y, e.z))
                .then(&Transform::rotation_x(-0.5));
            vec![upper, fore]
        }),
    };
    (s, rig)
}

#[cfg(feature = "gpu")]
mod gpu_tier {
    use super::*;
    use irys_diff::gpu::{self, GpuEdges, GpuPrb};

    /// 2f residual (a), fixed: on the device's own draws, its f32
    /// ownership test agrees with the host's f64 one at rose's scale (mm,
    /// 1.5 m from the origin) on CAD-style tessellation. Before the fix the
    /// device's inner ray, at a sliver, a zero-area filler triangle or a
    /// T-junction, missed the surface the host lands on and refused the
    /// sample: here 0.17 % of the host's weight (0.00 % after), on rose
    /// the samples with the largest jumps — 30 % of the shoulder pitch's
    /// arm gradient.
    #[test]
    fn device_ownership_matches_the_host_at_cad_scale() {
        let ctx = match gpu::context() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("no GPU ({e}): skipping");
                return;
            }
        };
        let (w, h) = (192, 144);
        let (mut s, rig) = mm_tube_arm(w, h);
        let phi = [0.25, 0.4, -0.9];
        rig.pose(&mut s, &phi);
        let edges = Edges::new(&s);
        let n = 16384;
        s.set_points(Some(vec![[0.0, 0.0]; 2 * n]));
        let gp = GpuPrb::new(ctx, &s);
        let mut ge = GpuEdges::new(&gp, &s, &edges, &rig.moving, n, 3, true);
        let (mut lost, mut gained, mut both, mut total) = (0.0, 0.0, 0usize, 0.0);
        for seed in 0..4 {
            let dev = ge.sample(&gp, &s, &rig, &phi, 77 + seed);
            ge.debug = true;
            let _ = ge.sample(&gp, &s, &rig, &phi, 77 + seed);
            let draws: Vec<_> = ge.draws(&gp).into_iter().map(|d| d.0).collect();
            ge.debug = false;
            let host = vis::EdgeSamples::at_draws(&s, &edges, &rig, &phi, &draws, true);
            for i in 0..n {
                let (hw, dw) = (&host.weights()[i], &dev.weights()[i]);
                let size = |w: &[f64]| w.iter().map(|x| x.abs()).sum::<f64>();
                let (hv, dv) = (size(hw) > 0.0, size(dw) > 0.0);
                total += size(hw);
                match (hv, dv) {
                    (true, false) => lost += size(hw),
                    (false, true) => gained += size(dw),
                    (true, true) => {
                        both += 1;
                        assert!(
                            (size(hw) - size(dw)).abs() <= 1e-2 * size(hw) + 1e-9,
                            "sample {i}: host {hw:?} device {dw:?}"
                        );
                    }
                    _ => {}
                }
            }
        }
        eprintln!(
            "mm CAD arm: {both} samples both visible; host-only {:.3} %, device-only {:.3} % of the host's Σ|w|",
            100.0 * lost / total,
            100.0 * gained / total
        );
        assert!(both > 4 * 2000, "the arm should be in view ({both})");
        assert!(lost < 5e-4 * total, "the device refuses {:.3} % of the host's weight", 100.0 * lost / total);
        assert!(gained < 0.01 * total, "the device adds {:.2} % to the host's weight", 100.0 * gained / total);
    }
}
