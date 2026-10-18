//! Phase 3: sampling. Mesh emitters with next-event estimation
//! ([`Scene::cornell`]); every invariant of phase 2 — replay bit-exact,
//! determinism, GPU against the CPU f64 reference, PRB against central
//! differences — and unbiasedness against BSDF sampling alone. GPU tests skip
//! without an adapter.

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

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn frozen(w: usize, h: usize, spp: usize) -> Scene {
    let mut s = Scene::cornell(w, h, spp);
    s.frozen_alpha = Some(0.3);
    s.depth = 4;
    s
}

/// The box's light is two triangles of 0.3 m², sampled with all of NEE
/// (the sky is black).
#[test]
fn cornell_has_one_quad_light() {
    let s = Scene::cornell(8, 8, 1);
    assert_eq!(s.lights().tris.len(), 2);
    assert!((s.lights().area - 0.3).abs() < 1e-12);
    assert_eq!(s.env_pick(), 0.0);
    assert_eq!(Scene::pair(8, 8, 1, 1.0).env_pick(), 1.0);
}

/// Emitter hits and emitter shadow rays replay bit for bit.
#[test]
fn emitter_replay_is_bit_exact() {
    let ctx = device!();
    let mut scene = Scene::cornell(64, 64, 4);
    scene.depth = 6;
    let mut g = GpuPrb::new(ctx, &scene);
    let l = g.forward(&scene, &THETA0, 11);
    let dl = vec![1.0; l.len()];
    let (d1, l2) = g.replay(&scene, &THETA0, 11, &dl);
    assert_eq!(bits(&l), bits(&l2));
    let (d2, _) = g.replay(&scene, &THETA0, 11, &dl);
    assert_eq!(d1.map(f64::to_bits), d2.map(f64::to_bits));
    assert!(d1.iter().all(|x| x.is_finite() && *x != 0.0), "δθ {d1:?}");
}

/// CPU f64 PRB against central differences with emitters: the emitter's
/// light lanes and hits are θ-free factors, so the NEE replay holds as is.
#[test]
fn emitter_cpu_gradient_matches_central_differences() {
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

/// GPU f32 against the CPU f64 reference with emitters, frozen sampler.
#[test]
fn emitter_gpu_matches_cpu_f64() {
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
        eprintln!("θ{p}: gpu {:+.6e}  cpu f64 {:+.6e}  rel {e:.1e}", gg[p], gc[p]);
        assert!(e < 2e-2, "θ{p}: gpu {} vs cpu {}", gg[p], gc[p]);
    }
}

/// Image mean and per-pixel variance over `frames` frames.
fn frame_stats(
    ctx: &'static irys_gpu::GpuContext,
    scene: &Scene,
    theta: &[f64; PARAMS],
    frames: usize,
) -> (Vec<f64>, f64) {
    let mut g = GpuPrb::new(ctx, scene);
    let imgs: Vec<Vec<f64>> = (0..frames)
        .map(|f| gpu::widen(&g.image(scene, theta, 100 + f as u64)))
        .collect();
    let n = imgs[0].len();
    let mean: Vec<f64> = (0..n)
        .map(|i| imgs.iter().map(|im| im[i]).sum::<f64>() / frames as f64)
        .collect();
    let var = (0..n)
        .map(|i| imgs.iter().map(|im| (im[i] - mean[i]).powi(2)).sum::<f64>() / (frames - 1) as f64)
        .sum::<f64>()
        / n as f64;
    (mean, var)
}

/// NEE on emitters is unbiased: the same image as BSDF sampling alone (which
/// finds the light only by hitting it), at much lower variance.
#[test]
fn emitter_nee_is_unbiased_and_cuts_variance() {
    let ctx = device!();
    let mut on = Scene::cornell(48, 48, 16);
    on.depth = 5;
    let mut off = Scene::cornell(48, 48, 16);
    off.depth = 5;
    off.nee = false;
    let (m_on, v_on) = frame_stats(ctx, &on, &GOLD, 64);
    let (m_off, v_off) = frame_stats(ctx, &off, &GOLD, 64);
    let s = |m: &[f64]| m.iter().sum::<f64>();
    let bias = rel(s(&m_on), s(&m_off));
    eprintln!(
        "cornell: mean rel {bias:.1e}; per-pixel variance nee {v_on:.3e}, bsdf {v_off:.3e} ({:.0}×)",
        v_off / v_on
    );
    // the BSDF-only image is noisy: its total is good to ~1 %
    assert!(bias < 2e-2, "means differ by {bias}");
    assert!(v_on * 2.0 < v_off, "NEE does not cut variance");
}

// ---- ReSTIR DI ----

use irys_diff::prb::Restir;

fn with_restir(mut s: Scene) -> Scene {
    s.restir = Some(Restir::default());
    s
}

/// Mean image, per-pixel variance, and per-8×8-block (mean, variance of the
/// mean) over `frames` frames, advancing the reservoirs between frames.
fn restir_stats(
    ctx: &'static irys_gpu::GpuContext,
    scene: &Scene,
    frames: usize,
) -> (f64, f64, Vec<(f64, f64)>) {
    let (w, h) = scene.size;
    let mut g = GpuPrb::new(ctx, scene);
    let imgs: Vec<Vec<f64>> = (0..frames)
        .map(|f| {
            let im = gpu::widen(&g.image(scene, &GOLD, 300 + f as u64));
            if scene.restir.is_some() {
                g.advance();
            }
            im
        })
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
    let block = |im: &Vec<f64>, b: usize| {
        (0..h * w)
            .filter(|p| (p / w) * 8 / h * 8 + (p % w) * 8 / w == b)
            .map(|p| im[3 * p] + im[3 * p + 1] + im[3 * p + 2])
            .sum::<f64>()
    };
    // Frames are correlated through temporal reuse: the variance of the
    // mean is estimated from batches of 8 frames.
    let batches = frames / 8;
    let blocks = (0..64)
        .map(|b| {
            let bm: Vec<f64> = (0..batches)
                .map(|k| (0..8).map(|f| block(&imgs[8 * k + f], b)).sum::<f64>() / 8.0)
                .collect();
            let m = bm.iter().sum::<f64>() / batches as f64;
            let v = bm.iter().map(|x| (x - m).powi(2)).sum::<f64>()
                / ((batches - 1) * batches) as f64;
            (m, v)
        })
        .collect();
    let total = mean(&|im| im.iter().sum::<f64>());
    (total, px_var, blocks)
}

/// ReSTIR DI is unbiased: the same image as plain NEE, on the sky-and-sun
/// pair and on the emitter box, block by block; and what it does to the
/// per-pixel variance at equal spp.
#[test]
fn restir_is_unbiased() {
    let ctx = device!();
    for (name, base) in [
        ("pair", {
            let mut s = Scene::pair(48, 48, 4, 1.0);
            s.complete = true;
            s
        }),
        ("cornell", {
            let mut s = Scene::cornell(48, 48, 4);
            s.depth = 4;
            s
        }),
    ] {
        let frames = 128;
        let (t0, v0, b0) = restir_stats(ctx, &base, frames);
        let (t1, v1, b1) = restir_stats(ctx, &with_restir(base), frames);
        let bias = rel(t0, t1);
        let z = b0
            .iter()
            .zip(&b1)
            .map(|(a, b)| (a.0 - b.0).abs() / (a.1 + b.1).sqrt())
            .fold(0.0, f64::max);
        eprintln!(
            "{name}: means rel {bias:.1e}, worst 8×8 block z {z:.2}; per-pixel variance nee {v0:.3e}, restir {v1:.3e} ({:.2}×)",
            v0 / v1
        );
        assert!(bias < 1e-2, "{name}: means differ by {bias}");
        assert!(z < 4.5, "{name}: a block differs by {z} σ");
    }
}

/// The reservoirs are a function of the seed and the previous frame's
/// buffers: replay at the same seed (no `advance` between) is bit-exact,
/// and two wavefronts walking the same frames agree bit for bit.
#[test]
fn restir_replay_is_bit_exact_and_deterministic() {
    let ctx = device!();
    let scene = with_restir(Scene::cornell(64, 64, 2));
    let run = || {
        let mut g = GpuPrb::new(ctx, &scene);
        let mut out = Vec::new();
        for f in 0..3u64 {
            let l = g.forward(&scene, &THETA0, 40 + f);
            let dl = vec![1.0; l.len()];
            let (d, l2) = g.replay(&scene, &THETA0, 40 + f, &dl);
            assert_eq!(bits(&l), bits(&l2), "frame {f}: replay");
            assert!(d.iter().all(|x| x.is_finite()));
            out.push((bits(&l), d.map(f64::to_bits)));
            g.advance();
        }
        out
    };
    assert!(run() == run(), "two wavefronts disagree");
}

/// PRB through ReSTIR's light lanes against central differences on the
/// device (frozen sampler, so the reservoirs do not move with θ).
#[test]
fn restir_gpu_gradient_matches_gpu_central_differences() {
    let ctx = device!();
    let mut scene = with_restir(Scene::pair(32, 32, 4, 1.0));
    scene.frozen_alpha = Some(0.3);
    let mut k = Kernels::for_scene(&Scene::pair(32, 32, 4, 1.0), false);
    let t = prb::image(
        &Scene::pair(32, 32, 4, 1.0),
        99,
        &prb::forward(&Scene::pair(32, 32, 4, 1.0), &GOLD, 99, &mut k).0,
    );
    let mut g = GpuPrb::new(ctx, &scene);
    g.set_target(&t);
    // one frame of history, then hold it
    g.image(&scene, &THETA0, 6);
    g.advance();
    let (_, d) = g.gradient_l2(&scene, &THETA0, (7, 7));
    let h = 4e-3;
    for p in 0..PARAMS {
        let (mut lo, mut hi) = (THETA0, THETA0);
        lo[p] -= h;
        hi[p] += h;
        let fd = (g.loss_l2(&scene, &hi, 7) - g.loss_l2(&scene, &lo, 7)) / (2.0 * h);
        let e = rel(d[p], fd);
        eprintln!("θ{p}: restir prb {:+.6e}  fd {fd:+.6e}  rel {e:.1e}", d[p]);
        assert!(e < 1e-2, "θ{p}: prb {} vs fd {fd}", d[p]);
    }
}
