//! Phase 2a: the PRB wavefront on the GPU. Each test skips (passes, with a
//! note) when there is no adapter, so a headless CI stays green.

#![cfg(feature = "gpu")]

use std::time::Instant;

use irys_diff::ggx::PARAMS;
use irys_diff::gpu::{self, GpuPrb};
use irys_diff::prb::{self, Kernels, L2, Loss, Scene};

const THETA0: [f64; PARAMS] = [0.35, 0.95, 0.75, 0.45];

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
    let mut s = Scene::new(w, h, spp);
    s.frozen_alpha = Some(0.3);
    s
}

fn target(scene: &Scene) -> Vec<f64> {
    let mut k = Kernels::new(scene.lanes(), false);
    prb::image(
        scene,
        99,
        &prb::forward(scene, &[0.2, 0.9, 0.9, 0.9], 99, &mut k).0,
    )
}

/// (1) The replay regenerates every path from (seed, path index) and runs
/// the same `f` pipelines: its `L` equals the forward pass's bit for bit,
/// on every lane, and the gradient is the same twice over.
#[test]
fn gpu_replay_is_bit_exact() {
    let ctx = device!();
    for depth in [2, 8] {
        let mut scene = Scene::new(64, 64, 4);
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
        let lit = l.iter().filter(|x| **x > 0.0).count();
        eprintln!(
            "depth {depth}: {same}/{} lanes bit-identical ({lit} nonzero)",
            l.len()
        );
        assert_eq!(same, l.len());
        assert!(
            lit > l.len() / 2,
            "the frame is mostly black: {lit}/{}",
            l.len()
        );
        let (d2, _) = g.replay(&scene, &THETA0, 11, &dl);
        assert_eq!(
            d1.map(f64::to_bits),
            d2.map(f64::to_bits),
            "gradient not deterministic"
        );
    }
}

/// (2) GPU f32 against the CPU f64 interpreter, frozen sampler, one seed:
/// the same random numbers (24 of 53 bits), the same paths, f32 rounding.
#[test]
fn gpu_gradient_matches_cpu_f64() {
    let ctx = device!();
    let scene = frozen(32, 32, 4);
    let t = target(&scene);
    let mut g = GpuPrb::new(ctx, &scene);
    let mut k = Kernels::new(scene.lanes(), false);
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
    let (vg, gg) = gpu::gradient(&mut g, &scene, &THETA0, (7, 7), &L2(&t));
    let (vc, gc) = prb::gradient(&scene, &THETA0, (7, 7), &mut k, &L2(&t));
    eprintln!("loss gpu {vg:.6e} cpu {vc:.6e} rel {:.1e}", rel(vg, vc));
    for p in 0..PARAMS {
        let e = rel(gg[p], gc[p]);
        eprintln!(
            "θ{p}: gpu {:+.6e}  cpu f64 {:+.6e}  rel {e:.1e}",
            gg[p], gc[p]
        );
        assert!(e < 1e-3, "θ{p}: gpu {} vs cpu {}", gg[p], gc[p]);
    }
    assert!(rel(vg, vc) < 1e-3);
}

/// (3) GPU gradient against central differences of the GPU's own loss,
/// common random numbers, frozen sampler.
#[test]
fn gpu_gradient_matches_gpu_central_differences() {
    let ctx = device!();
    let scene = frozen(32, 32, 4);
    let t = target(&scene);
    let mut g = GpuPrb::new(ctx, &scene);
    let (_, d) = gpu::gradient(&mut g, &scene, &THETA0, (7, 7), &L2(&t));
    let h = 1e-3;
    for p in 0..PARAMS {
        let (mut lo, mut hi) = (THETA0, THETA0);
        lo[p] -= h;
        hi[p] += h;
        let fd = (gpu::loss(&mut g, &scene, &hi, 7, &L2(&t))
            - gpu::loss(&mut g, &scene, &lo, 7, &L2(&t)))
            / (2.0 * h);
        let e = rel(d[p], fd);
        eprintln!("θ{p}: gpu prb {:+.6e}  gpu fd {fd:+.6e}  rel {e:.1e}", d[p]);
        assert!(e < 5e-3, "θ{p}: prb {} vs fd {fd}", d[p]);
    }
}

/// Piece 1: the image, the L2 loss and δL per lane on the device agree with
/// the host's f64 scoring of the same `L`, to f32.
#[test]
fn device_scoring_matches_the_host() {
    let ctx = device!();
    let scene = frozen(32, 32, 4);
    let t = target(&scene);
    let mut g = GpuPrb::new(ctx, &scene);
    g.set_target(&t);
    let (vd, gd) = g.gradient_l2(&scene, &THETA0, (7, 7));
    let dl_dev = g.last_lane_adjoint();
    let img_dev = g.last_image();

    let l = gpu::widen(&g.forward(&scene, &THETA0, 7));
    let img = prb::image(&scene, 7, &l);
    let (vh, d_img) = L2(&t).eval(&img);
    let dl = prb::lane_adjoint(&scene, 7, &d_img);
    let worst = |a: &[f32], b: &[f64]| {
        let scale = b.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        a.iter()
            .zip(b)
            .fold(0.0f64, |m, (x, y)| m.max((*x as f64 - y).abs()))
            / scale
    };
    let (ei, ed) = (worst(&img_dev, &img), worst(&dl_dev, &dl));
    eprintln!(
        "image max err / max {ei:.1e}; δL max err / max {ed:.1e}; loss device {vd:.7e} host {vh:.7e} rel {:.1e}",
        rel(vd, vh)
    );
    assert!(ei < 1e-5 && ed < 1e-5);
    assert!(rel(vd, vh) < 1e-5);
    let (_, gh) = gpu::gradient(&mut g, &scene, &THETA0, (7, 7), &L2(&t));
    for p in 0..PARAMS {
        let e = rel(gd[p], gh[p]);
        eprintln!(
            "θ{p}: device-scored {:+.6e}  host-scored {:+.6e}  rel {e:.1e}",
            gd[p], gh[p]
        );
        assert!(e < 1e-4, "θ{p}");
    }
}

/// Piece 1: the device-scored gradient, two seeds, is the same bits twice.
#[test]
fn device_scored_gradient_is_deterministic() {
    let ctx = device!();
    let scene = Scene::new(48, 48, 4);
    let t = target(&scene);
    let mut g = GpuPrb::new(ctx, &scene);
    g.set_target(&t);
    let a = g.gradient_l2(&scene, &THETA0, (3, 4));
    let b = g.gradient_l2(&scene, &THETA0, (3, 4));
    assert_eq!(a.0.to_bits(), b.0.to_bits(), "loss");
    assert_eq!(a.1.map(f64::to_bits), b.1.map(f64::to_bits), "δθ");
}

/// Throughput at the fit's size, 96×96×8, against the CPU compiled path.
#[test]
fn gpu_throughput() {
    let ctx = device!();
    let scene = Scene::new(96, 96, 8);
    let t = target(&Scene::new(96, 96, 8));
    let mut g = GpuPrb::new(ctx, &scene);
    gpu::gradient(&mut g, &scene, &THETA0, (1, 2), &L2(&t)); // warm
    const N: u32 = 5;
    let t0 = Instant::now();
    let (mut fwd, mut rep) = (0.0, 0.0);
    for i in 0..N {
        gpu::gradient(
            &mut g,
            &scene,
            &THETA0,
            (2 * i as u64, 2 * i as u64 + 1),
            &L2(&t),
        );
        fwd += g.timing.forward.as_secs_f64();
        rep += g.timing.replay.as_secs_f64();
    }
    let grad = t0.elapsed().as_secs_f64() / N as f64;
    g.set_target(&t);
    g.gradient_l2(&scene, &THETA0, (1, 2)); // warm
    let t3 = Instant::now();
    for i in 0..N {
        g.gradient_l2(&scene, &THETA0, (2 * i as u64, 2 * i as u64 + 1));
    }
    let dev = t3.elapsed().as_secs_f64() / N as f64;
    eprintln!(
        "gpu gradient, scored on the host {:.1} ms, scored on the device {:.1} ms",
        grad * 1e3,
        dev * 1e3
    );
    let (fwd, rep) = (fwd / (2 * N) as f64, rep / N as f64);
    let paths = scene.paths() as f64;
    eprintln!(
        "gpu, {} paths × {} λ: forward {:.2} ms ({:.1} Mpaths/s), replay {:.2} ms, gradient (2 forward + replay + host loss) {:.1} ms",
        scene.paths(),
        scene.wavelengths,
        fwd * 1e3,
        paths / fwd / 1e6,
        rep * 1e3,
        grad * 1e3
    );
    let mut k = Kernels::new(scene.lanes(), true);
    let t1 = Instant::now();
    prb::forward(&scene, &THETA0, 0, &mut k);
    let cf = t1.elapsed().as_secs_f64();
    let t2 = Instant::now();
    prb::gradient(&scene, &THETA0, (0, 1), &mut k, &L2(&t));
    let cg = t2.elapsed().as_secs_f64();
    eprintln!(
        "cpu (compiled C f32 kernels, native {}, this profile): forward {:.0} ms ({:.2} Mpaths/s), gradient {:.0} ms; gpu {:.0}× on the gradient (device-scored)",
        k.is_native(),
        cf * 1e3,
        paths / cf / 1e6,
        cg * 1e3,
        cg / dev
    );
}
