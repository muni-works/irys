//! Phase 2d: rose full — the whole robot, read from the rose repo's
//! assembly `full/cad/rose-full.loon` and tessellated by vcad 0.10's kernel
//! ([`arm`]), its left arm posed — as irys's hero frame, and the check that
//! the pose gradient holds on it.
//!
//! `cargo run --release -p irys-diff --features gpu --example rose -- out/rose`
//! writes `beauty.ppm` (768 × 768, 256 spp, depth 6, sun and sky with NEE)
//! at [`arm::TRUE_PHI`] and [`arm::TRUE_THETA`]. With `check` after the
//! directory it also runs the arm's silhouette gradient (depth 1, against
//! the sky: the mask) and the shaded one against central differences over
//! fixed quasi-random points per pixel.

#[path = "rose/arm.rs"]
mod arm;

use std::{path::Path, time::Instant};

use irys_diff::gpu::{self, GpuPrb};
use irys_diff::prb::{self, Kernels, Scene, wavelengths};
use irys_diff::vis::{self, Edges, Rig};
use irys_light::spectrum::hero_weight;

/// `g` per pixel channel: smooth over the frame, different per channel.
fn weights(w: usize, h: usize) -> Vec<f64> {
    let tau = std::f64::consts::TAU;
    (0..w * h * 3)
        .map(|i| {
            let (px, c) = (i / 3, i % 3);
            let (x, y) = ((px % w) as f64 / w as f64, (px / w) as f64 / h as f64);
            1.0 + 0.6 * (tau * x + c as f64).sin() * (3.0 * y + 0.5 * c as f64).cos()
        })
        .collect()
}

/// `Σ g I` with each pixel integrated over `k²` fixed R2 points.
fn grid_loss(scene: &mut Scene, k: usize, g: &[f64], seed: u64, gpu: &mut GpuPrb) -> f64 {
    grid_loss_with(scene, k, g, seed, |s| {
        gpu.update_geometry(s);
        gpu::widen(&gpu.forward(s, &arm::TRUE_THETA, seed))
    })
}

fn grid_loss_with(
    scene: &mut Scene,
    k: usize,
    g: &[f64],
    seed: u64,
    trace: impl FnOnce(&Scene) -> Vec<f64>,
) -> f64 {
    let (w, h) = scene.size;
    let mut pts = Vec::with_capacity(w * h * k * k);
    for py in 0..h {
        for px in 0..w {
            for j in 0..k * k {
                let u = (0.5 + 0.754_877_666_246_692_8 * j as f64).fract();
                let v = (0.5 + 0.569_840_290_998_053_2 * j as f64).fract();
                pts.push(scene.screen(px as f64 + u, py as f64 + v));
            }
        }
    }
    scene.set_points(Some(pts));
    let l = trace(scene);
    scene.set_points(None);
    let nw = scene.wavelengths;
    let mut loss = 0.0;
    for (lane, &lj) in l.iter().enumerate() {
        let i = lane / nw;
        let lam = wavelengths(prb::Rng::new(seed, (i / 2) as u64).uniform(), nw)[lane % nw];
        let hw = hero_weight(lam);
        let px = i / (k * k);
        for c in 0..3 {
            loss += lj * hw[c] as f64 / (nw * k * k) as f64 * g[px * 3 + c];
        }
    }
    loss
}

fn check(arm: &arm::Arm, depth: usize, label: &str) {
    let ctx = gpu::context().expect("check needs a GPU adapter");
    let (w, h) = (96, 96);
    let phi = arm::TRUE_PHI;
    let rig: Rig = arm.rig();
    let g = weights(w, h);
    let mut s = arm::scene(arm, (w, h), 1, &phi);
    s.depth = depth;
    if depth == 1 {
        // The mask: no NEE, and the ground shrunk to nothing far below, so
        // every pixel is the sky or the robot (the body is static).
        s.nee = false;
        s.place(&[(
            arm.meshes.len(),
            irys_core::Transform::translation(-5e6, -5e6, -1e6)
                .then(&irys_core::Transform::scale(1e-3, 1e-3, 1e-3)),
        )]);
    }
    // Edge estimate, CPU f64 and GPU f32 traces of the same samples.
    let edges = Edges::new(&s);
    let (n, runs) = (16384, 64);
    let t = Instant::now();
    let mut kc = Kernels::new(2 * n * s.wavelengths * if s.nee { 2 } else { 1 }, true);
    let mut cpu = Vec::new();
    let mut dev = Vec::new();
    let mut gp: Option<GpuPrb> = None;
    for r in 0..runs {
        let es = vis::EdgeSamples::new(&s, &edges, &rig, &phi, n, 500 + r);
        s.set_points(Some(es.points().to_vec()));
        if r < 2 {
            let (l, _) = prb::forward(&s, &arm::TRUE_THETA, 500 + r, &mut kc);
            cpu.push(es.gradient(&s, 500 + r, &l, &g));
        }
        let gpu = gp.get_or_insert_with(|| GpuPrb::new(ctx, &s));
        let l = gpu::widen(&gpu.forward(&s, &arm::TRUE_THETA, 500 + r));
        s.set_points(None);
        dev.push(es.gradient(&s, 500 + r, &l, &g));
        if r == 0 {
            eprintln!(
                "{label}: {} candidate edges, {} in frame, Σℓ {:.0} px ({:.0} silhouette or boundary); {} of {n} samples visible, {} halved",
                edges.candidates(),
                es.segments,
                es.total_len,
                es.silhouette_len,
                es.visible,
                es.halved
            );
        }
    }
    let te = t.elapsed();
    for (c, d) in cpu.iter().zip(&dev) {
        let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let diff: Vec<f64> = c.iter().zip(d).map(|(a, b)| a - b).collect();
        eprintln!(
            "{label}: same samples, GPU f32 vs CPU f64 edge term |Δ|/|g| {:.2e} (cpu {:.4?})",
            norm(&diff) / norm(c),
            c
        );
    }
    let nr = runs as f64;
    let mean: Vec<f64> = (0..3)
        .map(|j| dev.iter().map(|r| r[j]).sum::<f64>() / nr)
        .collect();
    let se: Vec<f64> = (0..3)
        .map(|j| {
            (dev.iter().map(|r| (r[j] - mean[j]).powi(2)).sum::<f64>() / (nr - 1.0) / nr).sqrt()
        })
        .collect();
    // Central differences over 32² fixed points per pixel, on the device.
    let env = |k: &str, d: f64| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let k = env("CD_K", 32.0) as usize;
    let t = Instant::now();
    s.set_points(Some(vec![[0.0; 2]; w * h * k * k]));
    let mut gg = GpuPrb::new(ctx, &s);
    s.set_points(None);
    let hh = env("CD_H", 0.01);
    let cd: Vec<f64> = (0..3)
        .map(|j| {
            let mut side = |d: f64| {
                let mut p = phi.to_vec();
                p[j] += d;
                rig.pose(&mut s, &p);
                grid_loss(&mut s, k, &g, 1, &mut gg)
            };
            let (a, b) = (side(hh), side(-hh));
            rig.pose(&mut s, &phi);
            (a - b) / (2.0 * hh)
        })
        .collect();
    if std::env::var("CPU_CD").is_ok() {
        let k = 12;
        let mut kc = Kernels::new(
            w * h * k * k * s.wavelengths * if s.nee { 2 } else { 1 },
            true,
        );
        let cd: Vec<f64> = (0..3)
            .map(|j| {
                let mut side = |d: f64| {
                    let mut p = phi.to_vec();
                    p[j] += d;
                    rig.pose(&mut s, &p);
                    grid_loss_with(&mut s, k, &g, 1, |s| {
                        prb::forward(s, &arm::TRUE_THETA, 1, &mut kc).0
                    })
                };
                let (a, b) = (side(0.02), side(-0.02));
                rig.pose(&mut s, &phi);
                (a - b) / 0.04
            })
            .collect();
        eprintln!("{label}: CPU central differences ({k}² points, h 0.02): {cd:.4?}");
    }
    eprintln!(
        "{label}: edge {runs} × {n} samples {te:.1?}; central differences ({k}² points per pixel, h {hh}) {:.1?}",
        t.elapsed()
    );
    for j in 0..3 {
        eprintln!(
            "{label} φ{j}: edge {:+.4e} ± {:.1e} (one run: rel σ {:.3}), central {:+.4e}, rel {:.2e}, z {:+.2}",
            mean[j],
            se[j],
            se[j] * nr.sqrt() / mean[j].abs(),
            cd[j],
            (mean[j] - cd[j]).abs() / cd[j].abs(),
            (mean[j] - cd[j]) / se[j]
        );
    }
}

/// One tree alone against the sky, posed by `(x, yaw, pitch)` about its
/// centre: the edge term against CPU central differences.
fn mini_check(mesh: &irys_core::TriMesh, label: &str) {
    use irys_core::{Point3, Transform, Vec3};
    let mut bb = irys_core::Aabb::empty();
    for p in mesh.positions() {
        bb.include_point(p);
    }
    let c = bb.center();
    let r = (bb.max - bb.min).norm() / 2.0;
    let (w, h) = (64, 64);
    let mut s = Scene::instanced(
        vec![prb::Opaque::new(
            mesh.clone(),
            prb::Surface::Diffuse,
            irys_prims::analytic::Analytic::from_prims(Vec::new()),
            &[],
        )],
        vec![prb::Placement::new(0, Transform::identity())],
        None,
        (w, h),
        1,
        (c + Vec3::new(1.2 * r, -3.0 * r, 1.5 * r), c),
        22.0,
    );
    s.depth = 1;
    let rig = Rig {
        moving: vec![0],
        place: Box::new(move |p: &[f64]| {
            vec![
                Transform::translation(p[0] * r, 0.0, 0.0)
                    .then(&vis::about(c, Vec3::new(0.0, 0.0, 1.0), p[1]))
                    .then(&vis::about(c, Vec3::new(1.0, 0.0, 0.0), p[2])),
            ]
        }),
    };
    let phi = [0.05, 0.3, 0.2];
    rig.pose(&mut s, &phi);
    let g = weights(w, h);
    let edges = Edges::new(&s);
    let n = 16384;
    let mut kc = Kernels::new(2 * n * s.wavelengths, true);
    let runs: Vec<Vec<f64>> = (0..8)
        .map(|r| {
            let (gr, es) = vis::boundary(&mut s, &edges, &rig, &phi, &g, n, 40 + r, |s| {
                prb::forward(s, &arm::TRUE_THETA, 40 + r, &mut kc).0
            });
            if r == 0 {
                eprintln!(
                    "{label}: Σℓ {:.0} ({:.0} silhouette), {} visible, {} halved",
                    es.total_len, es.silhouette_len, es.visible, es.halved
                );
            }
            gr
        })
        .collect();
    let mean: Vec<f64> = (0..3)
        .map(|j| runs.iter().map(|r| r[j]).sum::<f64>() / 8.0)
        .collect();
    let k = 24;
    let mut kc = Kernels::new(w * h * k * k * s.wavelengths, true);
    let cd: Vec<f64> = (0..3)
        .map(|j| {
            let mut side = |d: f64| {
                let mut p = phi.to_vec();
                p[j] += d;
                rig.pose(&mut s, &p);
                grid_loss_with(&mut s, k, &g, 1, |s| {
                    prb::forward(s, &arm::TRUE_THETA, 1, &mut kc).0
                })
            };
            let (a, b) = (side(0.01), side(-0.01));
            rig.pose(&mut s, &phi);
            (a - b) / 0.02
        })
        .collect();
    let _ = Point3::origin();
    eprintln!("{label}: edge {mean:.3?} central {cd:.3?}");
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next().unwrap_or_else(|| "out/rose".into());
    let out = Path::new(&out);
    std::fs::create_dir_all(out).unwrap();
    let arm = arm::load_arm(&arm::rose_dir());
    for (name, m) in arm.names.iter().zip(&arm.meshes) {
        eprintln!("  {name:14} {:6} triangles", m.triangles().len());
    }
    let mode = args.next();
    if mode.as_deref() == Some("prims") {
        for (name, m) in arm::test_parts() {
            eprintln!(
                "{name}: (triangles, edges of valence 1, 2, >2) {:?}",
                arm::valence(&m)
            );
            if std::env::var("SKIP").is_err() {
                mini_check(&m, name);
            }
        }
        return;
    }
    if mode.as_deref() == Some("mini") {
        for (name, m) in arm.names.iter().zip(&arm.meshes).take(4) {
            mini_check(m, name);
        }
        return;
    }
    if mode.as_deref() == Some("check") {
        check(&arm, 1, "mask");
        check(&arm, 5, "shaded");
        return;
    }
    let ctx = gpu::context().expect("rose needs a GPU adapter");
    let t = Instant::now();
    let mut beauty = arm::scene(&arm, (768, 768), 16, &arm::TRUE_PHI);
    beauty.depth = 6;
    let mut g = GpuPrb::new(ctx, &beauty);
    let built = t.elapsed();
    let t = Instant::now();
    let frames = 16;
    let mut acc = vec![0.0f64; 768 * 768 * 3];
    for f in 0..frames {
        for (a, x) in acc
            .iter_mut()
            .zip(g.image(&beauty, &arm::TRUE_THETA, 9000 + f))
        {
            *a += x as f64 / frames as f64;
        }
    }
    let img: Vec<f32> = acc.iter().map(|&x| x as f32).collect();
    arm::ppm(&out.join("beauty.ppm"), beauty.size, &img, 0.6);
    eprintln!(
        "beauty 768×768×{} (depth 6, NEE, {} chunks per call): build {built:.0?}, render {:.1?}",
        16 * frames,
        g.chunks(),
        t.elapsed()
    );
}
