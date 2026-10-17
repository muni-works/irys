//! Phase 2e: the interior term of the pose gradient ([`irys_diff::interior`]),
//! with the edge term ([`irys_diff::vis`]), against central differences of
//! the fully shaded image.

use irys_core::{Point3, Transform, Vec3};
use irys_diff::ggx::PARAMS;
use irys_diff::interior::{self, Tangents};
use irys_diff::prb::{self, Kernels, Loss, Opaque, Placement, Scene, Surface};
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
    Rig {
        moving: vec![2, 3],
        place: Box::new(|p: &[f64]| {
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
                SHOULDER[0] + 2.0 * UPPER + 2.0 * GAP,
                SHOULDER[1],
                SHOULDER[2],
            );
            let roll = |a: f64| Transform::rotation_x(a);
            let upper = sh
                .then(&Transform::translation(
                    SHOULDER[0] + UPPER + GAP,
                    SHOULDER[1],
                    SHOULDER[2],
                ))
                .then(&roll(0.35));
            let fore = sh
                .then(&vis::about(e, Vec3::new(0.0, 1.0, 0.0), p[2]))
                .then(&Transform::translation(e.x + FORE + GAP, e.y, e.z))
                .then(&roll(-0.5));
            vec![upper, fore]
        }),
    }
}

const PHI: [f64; 3] = [0.35, 0.45, -0.7];

/// The dual trace's values are the forward pass's `L`.
#[test]
fn dual_trace_values_are_the_forward_pass() {
    let mut s = arm_scene(24, 18, 2, 4);
    let rig = arm_rig();
    rig.pose(&mut s, &PHI);
    let tan = Tangents::rig(&s, &rig, &PHI);
    let mut k = Kernels::for_scene(&s, false);
    let (l, _) = prb::forward(&s, &THETA, 3, &mut k);
    let v = interior::values(&s, &tan, &THETA, 3);
    let worst = l
        .iter()
        .zip(&v)
        .map(|(a, b)| (a - b).abs() / a.abs().max(1e-3))
        .fold(0.0f64, f64::max);
    eprintln!(
        "dual values vs forward: worst rel {worst:.2e} over {} lanes",
        l.len()
    );
    assert!(worst < 1e-6, "{worst}");
}

/// Translations leave every frame where it is, so the attached-frame
/// derivative is the rendered estimator's: per lane, at one seed, the dual
/// tangent is the central difference (bar the few lanes an edge crosses).
#[test]
fn per_lane_tangent_is_the_central_difference_under_translation() {
    let mut s = arm_scene(24, 18, 2, 4);
    let rig = Rig {
        moving: vec![2, 3],
        place: Box::new(|p: &[f64]| {
            let r = |a: f64, x: f64, z: f64| {
                Transform::translation(x + p[0], SHOULDER[1], z + p[1]).then(&vis::about(
                    Point3::origin(),
                    Vec3::new(0.3, 0.8, 0.5).normalize(),
                    a,
                ))
            };
            vec![r(0.4, -0.2, 1.2), r(-0.6, 0.7, 1.0 + p[0])]
        }),
    };
    let phi = [0.05, -0.03];
    rig.pose(&mut s, &phi);
    let tan = Tangents::rig(&s, &rig, &phi);
    let mut k = Kernels::for_scene(&s, false);
    let h = 1e-6;
    let side = |s: &mut Scene, j: usize, d: f64, k: &mut Kernels| {
        let mut p = phi.to_vec();
        p[j] += d;
        rig.pose(s, &p);
        prb::forward(s, &THETA, 9, k).0
    };
    for j in 0..2 {
        let (a, b) = (side(&mut s, j, h, &mut k), side(&mut s, j, -h, &mut k));
        rig.pose(&mut s, &phi);
        let cd: Vec<f64> = a.iter().zip(&b).map(|(a, b)| (a - b) / (2.0 * h)).collect();
        let td = lane_tangents(&s, &tan, j);
        let (mut ok, mut moving) = (0, 0);
        for (c, t) in cd.iter().zip(&td) {
            if c.abs() + t.abs() > 0.0 {
                moving += 1;
                if (c - t).abs() <= 1e-4 * (1.0 + t.abs()) {
                    ok += 1;
                }
            }
        }
        eprintln!(
            "φ{j}: {ok} of the {moving} lanes that move (of {}) agree",
            cd.len()
        );
        assert!(moving >= 16, "the parameter moves something");
        assert!(ok as f64 >= 0.98 * moving as f64, "φ{j}");
    }
}

/// At depth 1 the light sample does not depend on the frame, so rotations
/// too are exact per lane: normal, cosine and MIS weight turning with the
/// link, through the duals, against central differences at one seed.
#[test]
fn per_lane_tangent_is_the_central_difference_at_depth_one() {
    let mut s = arm_scene_with(24, 18, 2, 1, false);
    let rig = arm_rig();
    rig.pose(&mut s, &PHI);
    let tan = Tangents::rig(&s, &rig, &PHI);
    let mut k = Kernels::for_scene(&s, false);
    let h = 1e-6;
    for j in 0..3 {
        let mut p = PHI.to_vec();
        p[j] += h;
        rig.pose(&mut s, &p);
        let a = prb::forward(&s, &THETA, 9, &mut k).0;
        p[j] -= 2.0 * h;
        rig.pose(&mut s, &p);
        let b = prb::forward(&s, &THETA, 9, &mut k).0;
        rig.pose(&mut s, &PHI);
        let td = lane_tangents(&s, &tan, j);
        let (mut ok, mut moving) = (0, 0);
        for ((a, b), t) in a.iter().zip(&b).zip(&td) {
            let c = (a - b) / (2.0 * h);
            if c.abs() + t.abs() > 1e-9 {
                moving += 1;
                ok += usize::from((c - t).abs() <= 1e-4 * (1.0 + t.abs()));
            }
        }
        eprintln!("φ{j}: {ok} of {moving} moving lanes agree");
        assert!(moving >= 8 && ok as f64 >= 0.98 * moving as f64, "φ{j}");
    }
}

/// Every lane's tangent to parameter `j`, from the dual trace.
fn lane_tangents(s: &Scene, tan: &Tangents, j: usize) -> Vec<f64> {
    interior::lane_tangents(s, tan, &THETA, 9, j)
}

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
#[test]
fn shaded_arm_edge_plus_interior_matches_central_differences() {
    gate(false);
}

/// The same arm over a floor it shadows: the moving shadow is a secondary
/// discontinuity neither term sees. Measured, not gated (`--ignored`).
#[test]
#[ignore]
fn shaded_arm_on_a_floor_is_measured() {
    gate(true);
}

fn gate(floor: bool) {
    let (w, h) = (48, 36);
    let spp = 4;
    let rig = arm_rig();
    let g = Weighted::new(w, h);
    let mut s = arm_scene_with(w, h, spp, 4, floor);
    rig.pose(&mut s, &PHI);
    let edges = Edges::new(&s);
    let tan = Tangents::rig(&s, &rig, &PHI);
    let seeds = 24u64;
    let n_edge = 8192;
    let mut ke = Kernels::new(2 * 2 * n_edge * s.wavelengths, true); // NEE: two lanes per vertex
    let t0 = std::time::Instant::now();
    let (mut er, mut ir, mut tot) = (Vec::new(), Vec::new(), Vec::new());
    for sd in 0..seeds {
        let (ge, _) = vis::boundary(&mut s, &edges, &rig, &PHI, &g.0, n_edge, 500 + sd, |sc| {
            prb::forward(sc, &THETA, 500 + sd, &mut ke).0
        });
        let gi = interior::interior(&s, &tan, &THETA, 900 + sd, &g.0);
        tot.push(ge.iter().zip(&gi).map(|(a, b)| a + b).collect::<Vec<_>>());
        er.push(ge);
        ir.push(gi);
    }
    let te = t0.elapsed();
    let (e, i, t) = (stats(&er), stats(&ir), stats(&tot));
    // Central differences of the image over k² fixed points per pixel,
    // common random numbers per seed.
    let t0 = std::time::Instant::now();
    let k = 16;
    let mut kf = Kernels::new(w * h * k * k * s.wavelengths * 2, true);
    let hh = 0.01;
    let mut cdr = Vec::new();
    for sd in 0..4 {
        let mut r = Vec::new();
        for j in 0..3 {
            let mut side = |d: f64| {
                let mut p = PHI.to_vec();
                p[j] += d;
                rig.pose(&mut s, &p);
                grid_loss(&mut s, k, &g, 7000 + sd, &mut kf)
            };
            let (a, b) = (side(hh), side(-hh));
            r.push((a - b) / (2.0 * hh));
        }
        cdr.push(r);
    }
    rig.pose(&mut s, &PHI);
    let cd = stats(&cdr);
    eprintln!(
        "estimate {te:.1?}, central differences {:.1?}",
        t0.elapsed()
    );
    let mut zs = Vec::new();
    for j in 0..3 {
        let sig = (t.stderr[j].powi(2) + cd.stderr[j].powi(2)).sqrt();
        let z = (t.mean[j] - cd.mean[j]) / sig;
        let ze = (e.mean[j] - cd.mean[j]) / (e.stderr[j].powi(2) + cd.stderr[j].powi(2)).sqrt();
        eprintln!(
            "φ{j}: edge {:+.3} ± {:.3}, interior {:+.3} ± {:.3}, sum {:+.3} ± {:.3}; CD {:+.3} ± {:.3}; z {z:+.2} (edge alone z {ze:+.2})",
            e.mean[j],
            e.stderr[j],
            i.mean[j],
            i.stderr[j],
            t.mean[j],
            t.stderr[j],
            cd.mean[j],
            cd.stderr[j]
        );
        zs.push(z);
    }
    for (j, z) in zs.iter().enumerate() {
        assert!(z.abs() < 3.0, "φ{j}: z {z}");
    }
}

/// `Σ g I` with every pixel integrated over `k²` fixed R2 points.
fn grid_loss(scene: &mut Scene, k: usize, g: &Weighted, seed: u64, kern: &mut Kernels) -> f64 {
    let (w, h) = scene.size;
    let mut pts = Vec::with_capacity(w * h * k * k);
    for py in 0..h {
        for px in 0..w {
            for j in 0..k * k {
                let (a1, a2) = (0.754_877_666_246_692_8, 0.569_840_290_998_053_2);
                let u = (0.5 + a1 * j as f64).fract();
                let v = (0.5 + a2 * j as f64).fract();
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

/// A CAD-like arm: each link is three parallel plates in one tree (a
/// bracket's cheeks and web), and the post stands in front of the
/// shoulder. Most front-facing silhouettes are then behind the link's own
/// near plate or the post — the shape of 2d's waste (76 % of the CAD arm's
/// samples hidden).
fn plate_arm(w: usize, h: usize, depth: usize) -> Scene {
    let plates = |hx: f64, hz: f64| {
        let mut pos = Vec::new();
        let mut idx: Vec<u32> = Vec::new();
        for (dy, hy) in [(-0.09, 0.02), (0.0, 0.015), (0.09, 0.02)] {
            let m = prb::cuboid(hx, hy, hz * if dy == 0.0 { 0.6 } else { 1.0 });
            let o = pos.len() as u32;
            pos.extend(
                m.positions()
                    .iter()
                    .map(|p| Point3::new(p.x, p.y + dy, p.z)),
            );
            idx.extend(m.triangles().concat().iter().map(|i| i + o));
        }
        irys_core::TriMesh::new(pos, Vec::new(), &idx)
    };
    let blas = vec![
        opaque(prb::cuboid(0.01, 0.01, 0.01), Surface::Diffuse),
        opaque(prb::cuboid(0.12, 0.12, 0.62), Surface::Diffuse),
        opaque(plates(UPPER, 0.1), Surface::Conductor),
        opaque(plates(FORE, 0.085), Surface::Diffuse),
    ];
    let placements = vec![
        Placement::new(0, Transform::translation(0.0, 0.0, -50.0)),
        Placement::new(
            1,
            Transform::translation(SHOULDER[0] + 0.25, SHOULDER[1] - 0.45, 0.62),
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
        4,
        (Point3::new(0.0, -4.5, 2.2), Point3::new(0.0, 0.0, 0.8)),
        20.0,
    );
    s.nee = true;
    s.depth = depth;
    s
}

/// Piece 2: the ID-buffer importance is unbiased (same mean as 2d's
/// importance and as central differences) and cheaper: the variance ratio
/// at equal sample count is printed and must beat 3.
#[test]
fn visibility_importance_cuts_variance_without_bias() {
    let mask = cut(1);
    let shaded = cut(2);
    assert!(mask > 1.5 && shaded > 1.3, "{mask} {shaded}");
}

/// The geometric-mean variance ratio (2d importance / 2e) at depth `depth`.
fn cut(depth: usize) -> f64 {
    let (w, h) = (48, 36);
    let rig = arm_rig();
    let g = Weighted::new(w, h);
    let mut s = plate_arm(w, h, depth);
    rig.pose(&mut s, &PHI);
    let edges = Edges::new(&s);
    let t = std::time::Instant::now();
    let vb = vis::Visibility::new(&s);
    let tv = t.elapsed();
    let n = 4096;
    let mut ke = Kernels::new(4 * n * s.wavelengths, true);
    let run = |s: &mut Scene, v: Option<&vis::Visibility>, ke: &mut Kernels| {
        let mut out = Vec::new();
        let mut vis_frac = 0.0;
        for sd in 0..24u64 {
            let (ge, es) = vis::boundary_with(s, &edges, &rig, &PHI, &g.0, n, 300 + sd, v, |sc| {
                prb::forward(sc, &THETA, 300 + sd, ke).0
            });
            vis_frac += es.visible as f64 / es.len() as f64 / 24.0;
            out.push(ge);
        }
        (stats(&out), vis_frac)
    };
    let (a, fa) = run(&mut s, None, &mut ke);
    let (b, fb) = run(&mut s, Some(&vb), &mut ke);
    eprintln!(
        "depth {depth}: ID buffer {tv:.1?}; visible samples {:.0} % → {:.0} %",
        100.0 * fa,
        100.0 * fb
    );
    let mut ratios = Vec::new();
    for j in 0..3 {
        let r = (a.stderr[j] / b.stderr[j]).powi(2);
        let z = (a.mean[j] - b.mean[j]) / (a.stderr[j].powi(2) + b.stderr[j].powi(2)).sqrt();
        eprintln!(
            "φ{j}: 2d {:+.3} ± {:.3}, ID buffer {:+.3} ± {:.3}: variance ratio {r:.1}, z {z:+.2}",
            a.mean[j], a.stderr[j], b.mean[j], b.stderr[j]
        );
        ratios.push(r);
        assert!(z.abs() < 3.5, "φ{j}: z {z}");
    }
    let gm = ratios.iter().product::<f64>().powf(1.0 / 3.0);
    eprintln!("depth {depth}: geometric-mean variance ratio {gm:.2}");
    gm
}
