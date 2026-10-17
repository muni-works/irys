//! Phase 2c: instances through a TLAS, next-event estimation and MIS, on the
//! CPU reference and the GPU wavefront. The scene is [`Scene::pair`]: one
//! mesh ball placed twice (one squashed and turned) on a mesh floor, NEE on.
//! GPU tests skip without an adapter.

#![cfg(feature = "gpu")]

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
    let mut s = Scene::pair(w, h, spp, 1.0);
    s.frozen_alpha = Some(0.3);
    s
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The scene is what it says: three instances of two trees, and the two
/// ball instances share one.
#[test]
fn pair_is_instanced() {
    let s = Scene::pair(8, 8, 1, 1.0);
    assert_eq!(s.blas().len(), 2);
    assert_eq!(s.tlas().len(), 3);
    assert_eq!(
        s.placements().iter().filter(|p| p.blas == 0).count(),
        2,
        "the ball is placed twice"
    );
}

/// Replay retraces the forward pass — TLAS walk, shadow rays, light lanes —
/// bit for bit at depth 2 and 6; the gradient is the same bits twice.
#[test]
fn nee_replay_is_bit_exact() {
    let ctx = device!();
    for depth in [2, 6] {
        let mut scene = Scene::pair(64, 64, 4, 1.0);
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
        assert!(d1.iter().all(|x| x.is_finite() && *x != 0.0), "δθ {d1:?}");
    }
}

/// Two wavefronts from the same scene, run twice each: `L`, image, loss and
/// δθ bitwise identical.
#[test]
fn nee_is_deterministic() {
    let ctx = device!();
    let scene = Scene::pair(48, 48, 4, 1.0);
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
    assert!(run() == run(), "two wavefronts disagree");
}

/// The NEE replay formula on the CPU f64 interpreter: PRB against central
/// differences with common random numbers, frozen sampler. Every term of
/// the derivation (light lanes, `L_j⁺`, the MIS weights as constants) is in
/// this number.
#[test]
fn nee_cpu_gradient_matches_central_differences() {
    let scene = frozen(16, 16, 2);
    let mut k = Kernels::for_scene(&scene, false);
    let t = prb::image(&scene, 99, &prb::forward(&scene, &GOLD, 99, &mut k).0);
    let (_, d) = prb::gradient(&scene, &THETA0, (7, 7), &mut k, &L2(&t));
    let h = 1e-5;
    for p in 0..PARAMS {
        let (mut lo, mut hi) = (THETA0, THETA0);
        lo[p] -= h;
        hi[p] += h;
        let fd = (prb::loss(&scene, &hi, 7, &mut k, &L2(&t))
            - prb::loss(&scene, &lo, 7, &mut k, &L2(&t)))
            / (2.0 * h);
        let e = rel(d[p], fd);
        eprintln!("θ{p}: cpu prb {:+.9e}  fd {fd:+.9e}  rel {e:.1e}", d[p]);
        assert!(e < 1e-5, "θ{p}: prb {} vs fd {fd}", d[p]);
    }
}

/// GPU f32 against the CPU f64 interpreter, frozen sampler: `L` per lane,
/// the device loss and δθ.
#[test]
fn nee_gpu_matches_cpu_f64() {
    let ctx = device!();
    let scene = frozen(32, 32, 4);
    let mut k = Kernels::for_scene(&scene, false);
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
    assert!(far * 50 < errs.len(), "{far} lanes diverge");
    let (vg, gg) = g.gradient_l2(&scene, &THETA0, (7, 7));
    let (vc, gc) = prb::gradient(&scene, &THETA0, (7, 7), &mut k, &L2(&t));
    eprintln!("loss gpu {vg:.6e} cpu {vc:.6e} rel {:.1e}", rel(vg, vc));
    assert!(rel(vg, vc) < 1e-2);
    for p in 0..PARAMS {
        let e = rel(gg[p], gc[p]);
        eprintln!(
            "θ{p}: gpu {:+.6e}  cpu f64 {:+.6e}  rel {e:.1e}",
            gg[p], gc[p]
        );
        assert!(e < 2e-2, "θ{p}: gpu {} vs cpu {}", gg[p], gc[p]);
    }
}

/// GPU PRB against GPU central differences, device-scored, frozen sampler.
#[test]
fn nee_gpu_gradient_matches_gpu_central_differences() {
    let ctx = device!();
    let scene = frozen(32, 32, 4);
    let mut k = Kernels::for_scene(&scene, false);
    let t = prb::image(&scene, 99, &prb::forward(&scene, &GOLD, 99, &mut k).0);
    let mut g = GpuPrb::new(ctx, &scene);
    g.set_target(&t);
    let (_, d) = g.gradient_l2(&scene, &THETA0, (7, 7));
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

/// NEE + MIS against BSDF sampling alone at equal spp: the same mean
/// (both unbiased for the same integral — tested per 8×8 block as a z-score
/// over frames), less variance.
#[test]
fn nee_cuts_variance_at_equal_spp() {
    let ctx = device!();
    let (w, h, spp, frames) = (64, 64, 4, 48);
    let stats = |nee: bool| {
        let mut scene = Scene::pair(w, h, spp, 1.0);
        scene.nee = nee;
        let mut g = GpuPrb::new(ctx, &scene);
        let imgs: Vec<Vec<f64>> = (0..frames)
            .map(|f| gpu::widen(&g.image(&scene, &GOLD, 100 + f as u64)))
            .collect();
        let n = imgs[0].len();
        let mean = |v: &dyn Fn(&Vec<f64>) -> f64| imgs.iter().map(v).sum::<f64>() / frames as f64;
        let px_var = (0..n)
            .map(|i| {
                let m = mean(&|im| im[i]);
                imgs.iter().map(|im| (im[i] - m).powi(2)).sum::<f64>() / (frames - 1) as f64
            })
            .sum::<f64>()
            / n as f64;
        // (mean, variance of the mean) per 8×8 block of the image.
        let block = |im: &Vec<f64>, b: usize| {
            (0..h * w)
                .filter(|p| (p / w) * 8 / h * 8 + (p % w) * 8 / w == b)
                .map(|p| im[3 * p] + im[3 * p + 1] + im[3 * p + 2])
                .sum::<f64>()
        };
        let blocks: Vec<(f64, f64)> = (0..64)
            .map(|b| {
                let m = mean(&|im| block(im, b));
                let v = imgs
                    .iter()
                    .map(|im| (block(im, b) - m).powi(2))
                    .sum::<f64>()
                    / ((frames - 1) * frames) as f64;
                (m, v)
            })
            .collect();
        (px_var, blocks)
    };
    let (v_bsdf, b_bsdf) = stats(false);
    let (v_nee, b_nee) = stats(true);
    let total = |b: &[(f64, f64)]| b.iter().map(|x| x.0).sum::<f64>();
    let bias = rel(total(&b_bsdf), total(&b_nee));
    let z = b_bsdf
        .iter()
        .zip(&b_nee)
        .map(|(a, b)| (a.0 - b.0).abs() / (a.1 + b.1).sqrt())
        .fold(0.0, f64::max);
    eprintln!(
        "per-pixel variance at {spp} spp: bsdf {v_bsdf:.3e}, nee+mis {v_nee:.3e}, ratio {:.1}×; image means rel {bias:.1e}, worst 8×8 block z {z:.2}",
        v_bsdf / v_nee
    );
    assert!(bias < 1e-2, "means differ by {bias}");
    assert!(z < 4.5, "a block's means differ by {z} σ");
    assert!(v_nee * 2.0 < v_bsdf, "NEE does not cut variance");
}

/// Throughput on the pair at the fit's size, with and without NEE.
#[test]
fn nee_throughput() {
    let ctx = device!();
    for nee in [false, true] {
        let mut scene = Scene::pair(96, 96, 8, 1.0);
        scene.nee = nee;
        let mut g = GpuPrb::new(ctx, &scene);
        g.set_target(&vec![0.2; 96 * 96 * 3]);
        g.gradient_l2(&scene, &THETA0, (1, 2));
        const N: u32 = 5;
        let t0 = std::time::Instant::now();
        for i in 0..N {
            g.gradient_l2(&scene, &THETA0, (2 * i as u64, 2 * i as u64 + 1));
        }
        let grad = t0.elapsed().as_secs_f64() / N as f64;
        let t1 = std::time::Instant::now();
        for i in 0..N {
            g.forward(&scene, &THETA0, i as u64);
        }
        let fwd = t1.elapsed().as_secs_f64() / N as f64;
        eprintln!(
            "pair 96×96×8, nee {nee}: forward {:.2} ms ({:.1} Mpaths/s), gradient {:.1} ms",
            fwd * 1e3,
            scene.paths() as f64 / fwd / 1e6,
            grad * 1e3
        );
    }
}

/// Piece 2: the same scene in millimetres. Light transport is scale-free,
/// so the ×1000 frame is the unit frame up to rounding; self-intersection
/// ("acne") shows as pixels that went dark. Both images are normalised by
/// their mean before comparing.
#[test]
fn cad_scale_renders_without_acne() {
    let ctx = device!();
    let (w, h) = (96, 96);
    let render = |scale: f64| {
        let scene = Scene::pair(w, h, 8, scale);
        let mut g = GpuPrb::new(ctx, &scene);
        let img = gpu::widen(&g.image(&scene, &GOLD, 21));
        let m = img.iter().sum::<f64>() / img.len() as f64;
        img.iter().map(|x| x / m).collect::<Vec<f64>>()
    };
    let unit = render(1.0);
    for scale in [1e3, 1e5] {
        let big = render(scale);
        let px = |im: &[f64], p: usize| im[3 * p] + im[3 * p + 1] + im[3 * p + 2];
        let mad = (0..w * h)
            .map(|p| (px(&big, p) - px(&unit, p)).abs())
            .sum::<f64>()
            / (0..w * h).map(|p| px(&unit, p)).sum::<f64>();
        let dark = (0..w * h)
            .filter(|&p| px(&big, p) < 0.8 * px(&unit, p))
            .count();
        eprintln!(
            "×{scale:e}: mean |Δ| / mean {mad:.2e}, {dark}/{} pixels > 20% darker",
            w * h
        );
        assert!(mad < 2e-2, "×{scale}: images differ by {mad}");
        assert!(dark * 200 < w * h, "×{scale}: {dark} acne pixels");
    }
}

/// Chunking is invisible: a frame traced in ten chunks (the last one
/// short) has the same `L`, image and loss bits as in one dispatch, the same
/// gradient up to the reduction's grouping, and replays bit-exactly.
#[test]
fn chunks_are_invisible() {
    let ctx = device!();
    let scene = Scene::pair(48, 48, 4, 1.0);
    let target = vec![0.1; 48 * 48 * 3];
    let mut one = GpuPrb::new(ctx, &scene);
    let mut ten = GpuPrb::chunked(ctx, &scene, 1000);
    assert_eq!((one.chunks(), ten.chunks()), (1, 10));
    one.set_target(&target);
    ten.set_target(&target);
    let l1 = one.forward(&scene, &THETA0, 3);
    let l10 = ten.forward(&scene, &THETA0, 3);
    assert_eq!(bits(&l1), bits(&l10), "L");
    assert_eq!(
        bits(&one.image(&scene, &THETA0, 3)),
        bits(&ten.image(&scene, &THETA0, 3)),
        "image"
    );
    let (v1, d1) = one.gradient_l2(&scene, &THETA0, (3, 4));
    let (v10, d10) = ten.gradient_l2(&scene, &THETA0, (3, 4));
    assert_eq!(v1.to_bits(), v10.to_bits(), "loss");
    for p in 0..PARAMS {
        eprintln!(
            "θ{p}: one {:+.7e} ten {:+.7e} rel {:.1e}",
            d1[p],
            d10[p],
            rel(d1[p], d10[p])
        );
        assert!(rel(d1[p], d10[p]) < 1e-4);
    }
    let l = ten.forward(&scene, &THETA0, 9);
    let (_, l2) = ten.replay(&scene, &THETA0, 9, &vec![1.0; l.len()]);
    assert_eq!(bits(&l), bits(&l2), "chunked replay");
}

/// A 1024 × 1024 frame (4 spp, NEE: 4.2 M paths, 16.8 M lanes, past the old
/// 65535 × 64-lane cap) in one call: rendered, replayed bit-exactly, and
/// the same picture as a small frame.
#[test]
fn megapixel_in_one_call() {
    let ctx = device!();
    let scene = Scene::pair(1024, 1024, 4, 1.0);
    let t = std::time::Instant::now();
    let mut g = GpuPrb::new(ctx, &scene);
    let built = t.elapsed();
    let t = std::time::Instant::now();
    let img = gpu::widen(&g.image(&scene, &GOLD, 1));
    let render = t.elapsed();
    let l = g.forward(&scene, &GOLD, 2);
    let t = std::time::Instant::now();
    let (d, l2) = g.replay(&scene, &GOLD, 2, &vec![1e-6; l.len()]);
    let replay = t.elapsed();
    let same = l
        .iter()
        .zip(&l2)
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    eprintln!(
        "1024² × 4 spp in {} chunks: build {built:.0?}, image {render:.0?}, replay {replay:.0?}; {same}/{} lanes bit-identical; δθ {d:.3?}",
        g.chunks(),
        l.len()
    );
    assert!(g.chunks() > 1);
    assert_eq!(same, l.len());
    assert!(img.iter().all(|x| x.is_finite()));
    let small = Scene::pair(128, 128, 64, 1.0);
    let s = gpu::widen(&GpuPrb::new(ctx, &small).image(&small, &GOLD, 1));
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    eprintln!(
        "mean 1024² {:.5}, 128² × 64 spp {:.5}",
        mean(&img),
        mean(&s)
    );
    assert!(rel(mean(&img), mean(&s)) < 1e-2);
}
