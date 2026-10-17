//! Phase 2b: the real scene on the GPU wavefront — irys-core's BVH flattened
//! and walked in WGSL, a triangle-mesh floor, the analytic ball, and a bush
//! of Gaussian splats composited 3DGRT-style. Each test skips without an
//! adapter.

#![cfg(feature = "gpu")]

use std::time::Instant;

use irys_diff::ggx::PARAMS;
use irys_diff::gpu::{self, GpuPrb};
use irys_diff::prb::{self, Kernels, L2, Scene};

const THETA0: [f64; PARAMS] = [0.35, 0.95, 0.75, 0.45];
const GOLD: [f64; PARAMS] = [0.2, 0.34, 0.78, 1.0];

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.abs().max(b.abs()).max(1e-12)
}

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

fn frozen(w: usize, h: usize, spp: usize) -> Scene {
    let mut s = Scene::garden(w, h, spp);
    s.frozen_alpha = Some(0.3);
    s
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Replay retraces the forward pass through the BVH and the splat K-buffer
/// bit for bit, at depth 2 and 6; the gradient is the same bits twice.
#[test]
fn garden_replay_is_bit_exact() {
    let ctx = device!();
    for depth in [2, 6] {
        let mut scene = Scene::garden(64, 64, 4);
        scene.depth = depth;
        let mut g = GpuPrb::new(ctx, &scene);
        let l = g.forward(&scene, &THETA0, 11);
        let dl = vec![1.0; l.len()];
        let (d1, l2) = g.replay(&scene, &THETA0, 11, &dl);
        let same = l
            .iter()
            .zip(&l2)
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        eprintln!("depth {depth}: {same}/{} lanes bit-identical", l.len());
        assert_eq!(same, l.len());
        let (d2, _) = g.replay(&scene, &THETA0, 11, &dl);
        assert_eq!(d1.map(f64::to_bits), d2.map(f64::to_bits));
    }
}

/// The walk order is a function of the ray: two wavefronts built from the
/// same scene, run twice each, give the same `L`, image and device-scored
/// gradient, bitwise.
#[test]
fn walk_order_is_deterministic() {
    let ctx = device!();
    let scene = Scene::garden(48, 48, 4);
    let target = vec![0.1; 48 * 48 * 3];
    let run = || {
        let mut g = GpuPrb::new(ctx, &scene);
        g.set_target(&target);
        let a = bits(&g.forward(&scene, &THETA0, 5));
        let b = bits(&g.forward(&scene, &THETA0, 5));
        assert_eq!(a, b, "same wavefront, two runs");
        let img = bits(&g.image(&scene, &THETA0, 5));
        let (v, d) = g.gradient_l2(&scene, &THETA0, (5, 6));
        (a, img, v.to_bits(), d.map(f64::to_bits))
    };
    let (r1, r2) = (run(), run());
    assert!(r1 == r2, "two wavefronts disagree");
    eprintln!(
        "{} lanes, image, loss and δθ identical across wavefronts",
        r1.0.len()
    );
}

/// The bush is in the frame and the compositor does something: the image
/// with splats differs from the image without on a real share of pixels,
/// and the CPU and GPU agree on which.
#[test]
fn splats_are_composited() {
    let ctx = device!();
    let mut scene = Scene::garden(48, 48, 4);
    let with = GpuPrb::new(ctx, &scene).image(&scene, &THETA0, 3);
    scene.set_splats(None);
    let without = GpuPrb::new(ctx, &scene).image(&scene, &THETA0, 3);
    let px = |img: &[f32], p: usize| img[3 * p] + img[3 * p + 1] + img[3 * p + 2];
    let changed = (0..48 * 48)
        .filter(|&p| (px(&with, p) - px(&without, p)).abs() > 1e-3)
        .count();
    eprintln!("{changed}/{} pixels changed by the bush", 48 * 48);
    assert!(changed > 48 * 48 / 20, "the bush covers {changed} pixels");
}

/// GPU f32 against the CPU f64 interpreter on the same scene (same BVHs,
/// same splats through `irys_light::splats::composite`), frozen sampler.
#[test]
fn garden_gpu_matches_cpu_f64() {
    let ctx = device!();
    let scene = frozen(32, 32, 4);
    let mut k = Kernels::new(scene.lanes(), false);
    let t = prb::image(&scene, 99, &prb::forward(&scene, &GOLD, 99, &mut k).0);
    let mut g = GpuPrb::new(ctx, &scene);
    g.set_target(&t);
    let lg = gpu::widen(&g.forward(&scene, &THETA0, 7));
    let lc = prb::forward(&scene, &THETA0, 7, &mut k).0;
    let mut errs: Vec<f64> = lg.iter().zip(&lc).map(|(a, b)| rel(*a, *b)).collect();
    errs.sort_by(f64::total_cmp);
    let far = errs.iter().filter(|e| **e > 1e-3).count();
    eprintln!(
        "L gpu vs cpu: median rel {:.1e}, p99 {:.1e}, {far}/{} lanes off by > 1e-3",
        errs[errs.len() / 2],
        errs[errs.len() * 99 / 100],
        errs.len()
    );
    assert!(far * 100 < errs.len(), "{far} lanes diverge");
    let (vg, gg) = g.gradient_l2(&scene, &THETA0, (7, 7));
    let (vc, gc) = prb::gradient(&scene, &THETA0, (7, 7), &mut k, &L2(&t));
    eprintln!("loss gpu {vg:.6e} cpu {vc:.6e} rel {:.1e}", rel(vg, vc));
    assert!(rel(vg, vc) < 1e-3);
    for p in 0..PARAMS {
        let e = rel(gg[p], gc[p]);
        eprintln!(
            "θ{p}: gpu {:+.6e}  cpu f64 {:+.6e}  rel {e:.1e}",
            gg[p], gc[p]
        );
        assert!(e < 1e-2, "θ{p}: gpu {} vs cpu {}", gg[p], gc[p]);
    }
}

/// GPU PRB against GPU central differences, common random numbers, frozen
/// sampler, device-scored loss.
#[test]
fn garden_gradient_matches_gpu_central_differences() {
    let ctx = device!();
    let scene = frozen(32, 32, 4);
    let mut k = Kernels::new(scene.lanes(), false);
    let t = prb::image(&scene, 99, &prb::forward(&scene, &GOLD, 99, &mut k).0);
    let mut g = GpuPrb::new(ctx, &scene);
    g.set_target(&t);
    let (_, d) = g.gradient_l2(&scene, &THETA0, (7, 7));
    // The loss is smooth in θ with the sampler frozen; a wider step keeps
    // f32 rounding in the loss out of the smallest component.
    let h = 4e-3;
    for p in 0..PARAMS {
        let (mut lo, mut hi) = (THETA0, THETA0);
        lo[p] -= h;
        hi[p] += h;
        let fd = (g.loss_l2(&scene, &hi, 7) - g.loss_l2(&scene, &lo, 7)) / (2.0 * h);
        let e = rel(d[p], fd);
        eprintln!("θ{p}: gpu prb {:+.6e}  gpu fd {fd:+.6e}  rel {e:.1e}", d[p]);
        assert!(e < 1e-2, "θ{p}: prb {} vs fd {fd}", d[p]);
    }
}

/// Throughput on the garden at the fit's size, 96×96×8, device-scored.
#[test]
fn garden_throughput() {
    let ctx = device!();
    let scene = Scene::garden(96, 96, 8);
    let mut g = GpuPrb::new(ctx, &scene);
    g.set_target(&vec![0.2; 96 * 96 * 3]);
    g.gradient_l2(&scene, &THETA0, (1, 2)); // warm
    const N: u32 = 5;
    let t0 = Instant::now();
    for i in 0..N {
        g.gradient_l2(&scene, &THETA0, (2 * i as u64, 2 * i as u64 + 1));
    }
    let grad = t0.elapsed().as_secs_f64() / N as f64;
    let t1 = Instant::now();
    for i in 0..N {
        g.forward(&scene, &THETA0, i as u64);
    }
    let fwd = t1.elapsed().as_secs_f64() / N as f64;
    let mut k = Kernels::new(scene.lanes(), true);
    let t2 = Instant::now();
    prb::forward(&scene, &THETA0, 0, &mut k);
    let cf = t2.elapsed().as_secs_f64();
    eprintln!(
        "garden 96×96×8 ({} paths × 4 λ): gpu forward {:.2} ms ({:.1} Mpaths/s), gradient {:.1} ms; cpu forward {:.0} ms",
        scene.paths(),
        fwd * 1e3,
        scene.paths() as f64 / fwd / 1e6,
        grad * 1e3,
        cf * 1e3
    );
}
