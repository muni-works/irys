//! Phase 2d: primary-visibility gradients by edge sampling ([`irys_diff::vis`]),
//! against central differences of the image with common random numbers.
//!
//! The scenes are against the sky at depth 1, where `f` is the moving
//! instance's mask (sky outside, black inside) and the edge term is the
//! whole derivative. The loss is linear, `Σ g · I` with fixed random `g`,
//! so a one-seed estimate of its gradient is unbiased for the expected one
//! and the two can be compared at any spp.

use irys_core::{Point3, Transform, Vec3};
use irys_diff::ggx::PARAMS;
use irys_diff::prb::{self, Kernels, Loss, Opaque, Placement, Scene, Surface};
use irys_diff::vis::{self, Edges, Rig};
use irys_prims::analytic::Analytic;

const THETA: [f64; PARAMS] = [0.3, 0.9, 0.8, 0.5];

/// `Σ g · I` with a smooth `g` over the frame, different per channel —
/// smooth so that the derivative is not a sum of cancelling pixels, which
/// would bury it in the central differences' noise.
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

fn opaque(m: irys_core::TriMesh) -> Opaque {
    Opaque::new(m, Surface::Diffuse, Analytic::from_prims(Vec::new()), &[])
}

/// A box against the sky, posed by `(x, yaw, pitch)`.
fn box_scene(w: usize, h: usize, spp: usize) -> Scene {
    let mut s = Scene::instanced(
        vec![opaque(prb::cuboid(0.7, 0.45, 0.55))],
        vec![Placement::new(0, Transform::identity())],
        None,
        (w, h),
        spp,
        (Point3::new(0.0, -4.5, 2.2), Point3::new(0.0, 0.0, 0.8)),
        20.0,
    );
    s.depth = 1;
    s
}

fn box_rig() -> Rig<'static> {
    Rig {
        moving: vec![0],
        place: Box::new(|p: &[f64]| {
            vec![
                Transform::translation(p[0], 0.0, 0.8)
                    .then(&vis::about(
                        Point3::origin(),
                        Vec3::new(0.0, 0.0, 1.0),
                        p[1],
                    ))
                    .then(&vis::about(
                        Point3::origin(),
                        Vec3::new(1.0, 0.0, 0.0),
                        p[2],
                    )),
            ]
        }),
    }
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.abs().max(b.abs()).max(1e-12)
}

#[test]
fn about_rotates_about_the_line() {
    let t = vis::about(
        Point3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        std::f64::consts::FRAC_PI_2,
    );
    let p = t.apply_point(&Point3::new(2.0, 0.0, 5.0));
    assert!((p - Point3::new(1.0, 1.0, 5.0)).norm() < 1e-12, "{p:?}");
}

/// The pinhole's projection is the inverse of its camera ray, and its
/// Jacobian is its derivative.
#[test]
fn projection_inverts_the_camera() {
    let s = box_scene(40, 30, 1);
    let (eye, f, r, u, _) = s.camera();
    let [sx, sy] = s.screen(13.25, 7.5);
    let x = eye + (f + r * sx + u * sy) * 3.7;
    let (px, jac) = s.project(x).unwrap();
    assert!(
        (px[0] - 13.25).abs() < 1e-9 && (px[1] - 7.5).abs() < 1e-9,
        "{px:?}"
    );
    let d = Vec3::new(0.3, -0.2, 0.5);
    let h = 1e-6;
    let (a, _) = s.project(x + d * h).unwrap();
    let (b, _) = s.project(x - d * h).unwrap();
    for c in 0..2 {
        let fd = (a[c] - b[c]) / (2.0 * h);
        assert!(
            (fd - jac[c].dot(&d)).abs() < 1e-5,
            "{fd} vs {}",
            jac[c].dot(&d)
        );
    }
}

/// Point mode traces the given rays, pairs keyed together.
#[test]
fn point_mode_pairs_share_a_generator() {
    let mut s = box_scene(16, 16, 1);
    s.depth = 3;
    s.set_points(Some(vec![[0.01, 0.2], [0.01, 0.2], [0.3, 0.1], [0.3, 0.1]]));
    let mut k = Kernels::for_scene(&s, false);
    let (l, _) = prb::forward(&s, &THETA, 5, &mut k);
    let nw = s.wavelengths;
    assert_eq!(l.len(), 4 * nw);
    assert_eq!(
        &l[..nw],
        &l[nw..2 * nw],
        "a pair through one point is one path"
    );
    assert_eq!(&l[2 * nw..3 * nw], &l[3 * nw..]);
}

#[derive(Debug)]
struct Estimate {
    mean: Vec<f64>,
    stderr: Vec<f64>,
}

/// `seeds` independent edge estimates of `∂(Σ g I)/∂φ`, `n` samples each.
fn edge_estimate(
    scene: &mut Scene,
    rig: &Rig,
    phi: &[f64],
    g: &Weighted,
    n: usize,
    seeds: u64,
) -> Estimate {
    let edges = Edges::new(scene);
    let mut k = Kernels::new(2 * n * scene.wavelengths, true);
    let runs: Vec<Vec<f64>> = (0..seeds)
        .map(|sd| {
            let (g, es) = vis::boundary(scene, &edges, rig, phi, &g.0, n, 100 + sd, |s| {
                prb::forward(s, &THETA, 100 + sd, &mut k).0
            });
            if sd == 0 {
                eprintln!(
                    "{} segments, Σℓ {:.1} px; {} of {} samples visible, {} halved",
                    es.segments,
                    es.total_len,
                    es.visible,
                    es.len(),
                    es.halved
                );
            }
            g
        })
        .collect();
    stats(&runs)
}

/// `Σ g I` with every pixel integrated over `k²` fixed quasi-random points
/// (the mask term is then a deterministic quadrature: the only randomness
/// left is the wavelengths, common to both sides of a difference).
fn grid_loss(scene: &mut Scene, k: usize, g: &Weighted, seed: u64, kern: &mut Kernels) -> f64 {
    let (w, h) = scene.size;
    let mut pts = Vec::with_capacity(w * h * k * k);
    for py in 0..h {
        for px in 0..w {
            // Roberts' R2 sequence: low discrepancy, and no row of points
            // lines up with an edge (a grid flips whole rows at once).
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

/// Central differences of [`grid_loss`].
fn grid_central(
    scene: &mut Scene,
    rig: &Rig,
    phi: &[f64],
    g: &Weighted,
    k: usize,
    h: f64,
) -> Vec<f64> {
    let (w, hh) = scene.size;
    let mut kern = Kernels::new(w * hh * k * k * scene.wavelengths, true);
    (0..phi.len())
        .map(|j| {
            let mut side = |d: f64| {
                let mut p = phi.to_vec();
                p[j] += d;
                rig.pose(scene, &p);
                grid_loss(scene, k, g, 1, &mut kern)
            };
            let (a, b) = (side(h), side(-h));
            rig.pose(scene, phi);
            (a - b) / (2.0 * h)
        })
        .collect()
}

/// Mean and standard error of `runs`, per parameter.
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

/// The edge term against central differences of the expected image: both
/// are averaged over seeds (a one-seed CD with CRN is a difference of step
/// functions, unbiased but noisy), and must agree within their errors.
#[test]
fn box_silhouette_gradient_matches_central_differences() {
    let (w, h) = (40, 30);
    let phi = [0.15, 0.5, 0.2];
    let rig = box_rig();
    let g = Weighted::new(w, h);
    let mut s = box_scene(w, h, 1);
    rig.pose(&mut s, &phi);
    let t = std::time::Instant::now();
    let est = edge_estimate(&mut s, &rig, &phi, &g, 8192, 24);
    let te = t.elapsed();
    let t = std::time::Instant::now();
    // 2304 fixed points per pixel and h = 0.05 (edges move ~½ px): ~10⁴
    // flipped samples per side, so the reference is good to ~1 %.
    let cd = grid_central(&mut s, &rig, &phi, &g, 48, 0.05);
    eprintln!(
        "edge 24 × 8192 samples {te:.1?}; central differences over 48² points per pixel {:.1?}",
        t.elapsed()
    );
    for j in 0..3 {
        let z = (est.mean[j] - cd[j]) / est.stderr[j];
        eprintln!(
            "φ{j}: edge {:+.4} ± {:.4} (one run of 8192: rel σ {:.3}), central {:+.4}, rel {:.2e}, z {z:+.2}",
            est.mean[j],
            est.stderr[j],
            est.stderr[j] * 24f64.sqrt() / est.mean[j].abs().max(1e-9),
            cd[j],
            rel(est.mean[j], cd[j]),
        );
        assert!(
            (est.mean[j] - cd[j]).abs() < 4.0 * est.stderr[j] + 0.03 * cd[j].abs(),
            "φ{j}: z {z}"
        );
    }
}

/// vcad's tessellation of a rounded 60 × 60 × 3 mm plate (two boxes and
/// four corner cylinders, unioned — rose's `plate-z`): thousands of
/// triangles, slivers thinner than the edge sample's ε along the arcs. The
/// inner ray then lands on a coplanar neighbour, which must still count as
/// the edge's own side (it did not, and the edge term came out 60 % high).
#[test]
fn vcad_plate_silhouette_matches_central_differences() {
    use vcad::Part;
    let (r, t) = (20.0, 3.0);
    let corner = |x: f64, y: f64| Part::cylinder("c", r, t, 24).translate(x, y, 0.0);
    let plate = Part::cube("core", 60.0 - 2.0 * r, 60.0, t)
        .translate(r, 0.0, 0.0)
        .union(&Part::cube("cross", 60.0, 60.0 - 2.0 * r, t).translate(0.0, r, 0.0))
        .union(&corner(r, r))
        .union(&corner(r, 60.0 - r))
        .union(&corner(60.0 - r, r))
        .union(&corner(60.0 - r, 60.0 - r));
    let m = plate.to_mesh();
    let v = m.vertices();
    let pos = v
        .chunks_exact(3)
        .map(|p| {
            Point3::new(
                p[0] as f64 / 40.0 - 0.75,
                p[1] as f64 / 40.0 - 0.75,
                p[2] as f64 / 40.0,
            )
        })
        .collect();
    let mesh = irys_core::TriMesh::new(pos, Vec::new(), &m.indices());
    eprintln!("vcad plate: {} triangles", mesh.triangles().len());
    let (w, h) = (40, 30);
    let mut s = Scene::instanced(
        vec![opaque(mesh)],
        vec![Placement::new(0, Transform::identity())],
        None,
        (w, h),
        1,
        (Point3::new(0.0, -4.5, 2.2), Point3::new(0.0, 0.0, 0.8)),
        20.0,
    );
    s.depth = 1;
    let rig = box_rig();
    let phi = [0.15, 0.5, 0.2];
    rig.pose(&mut s, &phi);
    let g = Weighted::new(w, h);
    let est = edge_estimate(&mut s, &rig, &phi, &g, 8192, 16);
    let cd = grid_central(&mut s, &rig, &phi, &g, 32, 0.05);
    for j in 0..3 {
        let z = (est.mean[j] - cd[j]) / est.stderr[j];
        eprintln!(
            "φ{j}: edge {:+.4} ± {:.4}, central {:+.4}, rel {:.2e}, z {z:+.2}",
            est.mean[j],
            est.stderr[j],
            cd[j],
            rel(est.mean[j], cd[j]),
        );
        assert!(
            (est.mean[j] - cd[j]).abs() < 4.0 * est.stderr[j] + 0.03 * cd[j].abs(),
            "φ{j}: z {z}"
        );
    }
}

#[cfg(feature = "gpu")]
mod gpu_tier {
    use super::*;
    use irys_diff::gpu::{self, GpuPrb};

    macro_rules! device {
        () => {
            match gpu::context() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("no GPU ({e}): skipping");
                    return;
                }
            }
        };
    }

    /// The same edge samples, their pairs traced on the device (f32) and on
    /// the CPU (f64): the edge term agrees, and the device's point-mode
    /// replay retraces its forward pass bit for bit.
    #[test]
    fn gpu_edge_term_matches_cpu() {
        let ctx = device!();
        let (w, h) = (40, 30);
        let phi = [0.15, 0.5, 0.2];
        let rig = box_rig();
        let g = Weighted::new(w, h);
        let mut s = box_scene(w, h, 1);
        s.depth = 3; // shaded sides too, not just the mask
        rig.pose(&mut s, &phi);
        let edges = Edges::new(&s);
        let n = 8192;
        let es = vis::EdgeSamples::new(&s, &edges, &rig, &phi, n, 77);
        s.set_points(Some(es.points().to_vec()));
        let mut k = Kernels::for_scene(&s, true);
        let (lc, _) = prb::forward(&s, &THETA, 77, &mut k);
        let mut gp = GpuPrb::new(ctx, &s);
        let lg = gpu::widen(&gp.forward(&s, &THETA, 77));
        let (_, l2) = gp.replay(&s, &THETA, 77, &vec![1.0; lg.len()]);
        let same = lg
            .iter()
            .zip(&l2)
            .filter(|(a, b)| (**a as f32).to_bits() == b.to_bits())
            .count();
        eprintln!("point-mode replay: {same}/{} lanes bit-identical", lg.len());
        assert_eq!(same, lg.len());
        s.set_points(None);
        let (gc, gg) = (
            es.gradient(&s, 77, &lc, &g.0),
            es.gradient(&s, 77, &lg, &g.0),
        );
        for j in 0..3 {
            eprintln!(
                "φ{j}: cpu {:+.6} gpu {:+.6} rel {:.2e}",
                gc[j],
                gg[j],
                rel(gc[j], gg[j])
            );
            assert!(rel(gc[j], gg[j]) < 2e-3, "φ{j}");
        }
    }

    /// Moving instances and re-uploading the geometry is the same frame as
    /// building the wavefront at the new pose.
    #[test]
    fn update_geometry_is_a_rebuild() {
        let ctx = device!();
        let rig = box_rig();
        let mut s = box_scene(32, 24, 2);
        s.depth = 3;
        rig.pose(&mut s, &[0.0, 0.0, 0.0]);
        let mut a = GpuPrb::new(ctx, &s);
        let _ = a.image(&s, &THETA, 3);
        rig.pose(&mut s, &[0.2, 0.7, -0.3]);
        a.update_geometry(&s);
        let ia = a.image(&s, &THETA, 3);
        let ib = GpuPrb::new(ctx, &s).image(&s, &THETA, 3);
        assert_eq!(
            ia.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            ib.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
}
