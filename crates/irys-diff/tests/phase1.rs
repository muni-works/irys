//! The phase 1 exit gate: gradients against central differences, compiled
//! against interpreted, replay against forward, and one inverse problem.

use std::time::Instant;

use irys_diff::ggx::{self, COLUMNS, DF, Graph, PARAMS, THETA};
use irys_diff::prb::{self, Kernels, L2, Mean, Scene};

const THETA0: [f64; PARAMS] = [0.35, 0.95, 0.75, 0.45];

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.abs().max(b.abs()).max(1e-12)
}

/// Central differences with common random numbers, per parameter.
fn central(
    scene: &Scene,
    theta: &[f64; PARAMS],
    seed: u64,
    k: &mut Kernels,
    loss: &impl prb::Loss,
    h: f64,
) -> [f64; PARAMS] {
    std::array::from_fn(|p| {
        let (mut lo, mut hi) = (*theta, *theta);
        lo[p] -= h;
        hi[p] += h;
        (prb::loss(scene, &hi, seed, k, loss) - prb::loss(scene, &lo, seed, k, loss)) / (2.0 * h)
    })
}

/// (a) With the sampler frozen, paths do not move with θ, so CRN central
/// differences see exactly the estimator PRB differentiates: f64, tight.
#[test]
fn gradient_matches_central_differences_frozen_sampler() {
    let mut scene = Scene::new(16, 16, 2);
    scene.frozen_alpha = Some(0.3);
    let mut k = Kernels::new(scene.lanes(), false);
    let target = prb::image(
        &scene,
        99,
        &prb::forward(&scene, &[0.2, 0.9, 0.9, 0.9], 99, &mut k).0,
    );
    let t = Instant::now();
    let (_, g) = prb::gradient(&scene, &THETA0, (7, 7), &mut k, &L2(&target));
    let t_grad = t.elapsed();
    let fd = central(&scene, &THETA0, 7, &mut k, &L2(&target), 1e-5);
    for p in 0..PARAMS {
        let e = rel(g[p], fd[p]);
        eprintln!("θ{p}: prb {:+.6e}  fd {:+.6e}  rel {e:.1e}", g[p], fd[p]);
        assert!(e < 1e-5, "θ{p}: prb {} vs fd {}", g[p], fd[p]);
    }
    eprintln!(
        "gradient (forward + replay, interpreter, {} paths): {t_grad:?}",
        scene.paths()
    );
}

/// (a) Sampling at the current roughness: paths move with θ, so the two only
/// agree in expectation. The mean pixel value is linear, so each seed's
/// gradient is unbiased; average over seeds and compare within three
/// standard errors. Central differences are the noisy side here — a sample
/// that crosses the ball's silhouette as α moves is a jump of size 1/h.
#[test]
fn gradient_matches_central_differences_live_sampler() {
    const SEEDS: u64 = 64;
    let scene = Scene::new(16, 16, 4);
    let mut k = Kernels::new(scene.lanes(), true);
    let (mut g, mut fd) = (vec![[0.0; PARAMS]; 0], vec![[0.0; PARAMS]; 0]);
    for s in 0..SEEDS {
        g.push(prb::gradient(&scene, &THETA0, (s, s), &mut k, &Mean).1);
        fd.push(central(&scene, &THETA0, s, &mut k, &Mean, 2e-2));
    }
    let stats = |v: &[[f64; PARAMS]], p: usize| {
        let n = v.len() as f64;
        let m = v.iter().map(|x| x[p]).sum::<f64>() / n;
        (
            m,
            (v.iter().map(|x| (x[p] - m).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt(),
        )
    };
    for p in 0..PARAMS {
        let ((gm, gs), (fm, fs)) = (stats(&g, p), stats(&fd, p));
        let sigma = (gs * gs + fs * fs).sqrt();
        eprintln!(
            "θ{p}: prb {gm:+.4e} ± {gs:.1e}  fd {fm:+.4e} ± {fs:.1e}  ({:.1}σ, rel {:.1e})",
            (gm - fm).abs() / sigma,
            rel(gm, fm)
        );
        assert!(
            (gm - fm).abs() < 3.0 * sigma,
            "θ{p}: prb {gm} ± {gs} vs fd {fm} ± {fs}"
        );
    }
}

/// (b) The same graph, compiled to C (f32) and interpreted (f64): f, pdf
/// and δθ agree, and f and pdf match a hand-written f64 GGX.
#[test]
fn compiled_matches_interpreter() {
    let n = 256;
    let gr = Graph::new(n);
    let mut native = (gr.eval(true), gr.adjoint(true));
    if !native.0.is_native() {
        eprintln!("no C compiler (or IRYS_IR=interp): skipping the compiled side");
        return;
    }
    let mut interp = (gr.eval(false), gr.adjoint(false));
    let mut rng = prb::Rng::new(3, 0);
    let mut cols = vec![vec![0.0; n]; COLUMNS];
    let mut df = vec![0.0; n];
    for i in 0..n {
        let alpha = THETA0[0];
        let wo = ggx::normalize([
            rng.uniform() - 0.5,
            rng.uniform() - 0.5,
            0.2 + rng.uniform(),
        ]);
        let (wi, _) = ggx::sample(alpha, wo, [rng.uniform(), rng.uniform()]);
        let wi = if wi[2] > 0.05 { wi } else { [0.0, 0.0, 1.0] };
        let h = ggx::normalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
        let b = ggx::band_weights(380.0 + 400.0 * rng.uniform());
        let row = [
            wi[2],
            wo[2],
            h[2],
            wo[0] * h[0] + wo[1] * h[1] + wo[2] * h[2],
            b[0],
            b[1],
            b[2],
        ];
        for (c, v) in cols.iter_mut().zip(row) {
            c[i] = v;
        }
        df[i] = rng.uniform() - 0.5;
    }
    let mut out = Vec::new();
    for p in [&mut native, &mut interp] {
        for (c, col) in cols.iter().enumerate() {
            p.0.write(c, col);
            p.1.write(c, col);
        }
        p.0.write(THETA, &THETA0);
        p.1.write(THETA, &THETA0);
        p.1.write(DF, &df);
        p.0.run();
        p.1.run();
        out.push((p.0.read(0), p.0.read(1), p.1.read(0)));
    }
    let (worst_f, worst_d) = (0..n).fold((0.0f64, 0.0f64), |(wf, wp), i| {
        let (f, pdf) = ggx::eval_f64(
            &THETA0,
            [cols[4][i], cols[5][i], cols[6][i]],
            cols[0][i],
            cols[1][i],
            cols[2][i],
            cols[3][i],
        );
        assert!(
            rel(out[1].0[i], f) < 1e-12 && rel(out[1].1[i], pdf) < 1e-12,
            "interpreter vs f64 at {i}"
        );
        (wf.max(rel(out[0].0[i], f)), wp.max(rel(out[0].1[i], pdf)))
    });
    let worst_t = (0..PARAMS)
        .map(|p| rel(out[0].2[p], out[1].2[p]))
        .fold(0.0, f64::max);
    eprintln!(
        "compiled f32 vs interpreter f64: f {worst_f:.1e}, pdf {worst_d:.1e}, δθ {worst_t:.1e}"
    );
    assert!(worst_f < 1e-4 && worst_d < 1e-4 && worst_t < 1e-4);
}

/// The graph's pdf is the sampler's: the two lowerings of one BSDF agree.
#[test]
fn graph_pdf_is_the_samplers() {
    let mut rng = prb::Rng::new(5, 0);
    for _ in 0..64 {
        let wo = ggx::normalize([
            rng.uniform() - 0.5,
            rng.uniform() - 0.5,
            0.1 + rng.uniform(),
        ]);
        let (wi, pdf) = ggx::sample(0.4, wo, [rng.uniform(), rng.uniform()]);
        let h = ggx::normalize([wi[0] + wo[0], wi[1] + wo[1], wi[2] + wo[2]]);
        let (_, p) = ggx::eval_f64(
            &[0.4, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            wi[2].max(1e-3),
            wo[2],
            h[2],
            1.0,
        );
        assert!(rel(p, pdf) < 1e-9, "{p} vs {pdf}");
    }
}

/// (c) Replay regenerates every path: its L equals the forward pass's bit
/// for bit, on both backends. And the wavefront's state does not grow with
/// depth: no per-vertex storage.
#[test]
fn replay_is_bit_exact_and_constant_memory() {
    for native in [false, true] {
        let mut scene = Scene::new(16, 16, 2);
        let mut k = Kernels::new(scene.lanes(), native);
        let mut words = Vec::new();
        for depth in [2, 8] {
            scene.depth = depth;
            let (l, fs) = prb::forward(&scene, &THETA0, 11, &mut k);
            let dl = vec![1.0; l.len()];
            let (_, l2, rs) = prb::replay(&scene, &THETA0, 11, &mut k, &l, &dl);
            let same = l
                .iter()
                .zip(&l2)
                .filter(|(a, b)| a.to_bits() == b.to_bits())
                .count();
            assert_eq!(
                same,
                l.len(),
                "native {native} depth {depth}: {same}/{} paths replayed exactly",
                l.len()
            );
            assert_eq!(fs.vertices, rs.vertices);
            words.push((fs.state_words, rs.state_words));
        }
        assert!(
            words.iter().all(|&w| w == words[0]),
            "state grew with depth: {words:?}"
        );
        eprintln!(
            "native {}: {} paths, {} state words ({:.1} per path) at depth 2 and 8",
            k.is_native(),
            scene.paths(),
            words[0].0,
            words[0].0 as f64 / scene.paths() as f64
        );
    }
}

/// (d) Recover roughness from a target image by gradient descent.
#[test]
fn recovers_roughness() {
    let scene = Scene::new(32, 32, 4);
    let mut k = Kernels::new(scene.lanes(), true);
    let truth = [0.25, THETA0[1], THETA0[2], THETA0[3]];
    // The target is the same view at 64 spp: its noise only shifts the loss.
    let reference = Scene::new(32, 32, 64);
    let mut kr = Kernels::new(reference.lanes(), true);
    let target = prb::image(
        &reference,
        1000,
        &prb::forward(&reference, &truth, 1000, &mut kr).0,
    );
    let mut theta = [0.6, THETA0[1], THETA0[2], THETA0[3]];
    // Adam on α alone, fresh (and decorrelated) seeds per iteration.
    const ITERS: i32 = 40;
    let (lr, b1, b2) = (0.05, 0.9, 0.99);
    let (mut m, mut v) = (0.0, 0.0);
    let t = Instant::now();
    let (mut first, mut tail) = (None, Vec::new());
    for it in 1..=ITERS {
        let (loss, g) = prb::gradient(
            &scene,
            &theta,
            (2 * it as u64, 2 * it as u64 + 1),
            &mut k,
            &L2(&target),
        );
        m = b1 * m + (1.0 - b1) * g[0];
        v = b2 * v + (1.0 - b2) * g[0] * g[0];
        let step = lr / (1.0 + it as f64 / 15.0) * (m / (1.0 - b1.powi(it)))
            / ((v / (1.0 - b2.powi(it))).sqrt() + 1e-12);
        theta[0] = (theta[0] - step).clamp(0.02, 1.0);
        if it % 5 == 0 {
            eprintln!("it {it:3}  α {:.4}  loss {loss:.4e}", theta[0]);
        }
        if first.is_none() && (theta[0] - truth[0]).abs() < 0.02 {
            first = Some(it);
        }
        if it > ITERS - 10 {
            tail.push(theta[0]);
        }
    }
    let settled = tail.iter().sum::<f64>() / tail.len() as f64;
    eprintln!(
        "α within 0.02 of {} at iteration {first:?}; last-10 mean {settled:.4}; {ITERS} iterations in {:?}, native {}",
        truth[0],
        t.elapsed(),
        k.is_native()
    );
    assert!(
        first.is_some_and(|i| i <= 30),
        "α never got within 0.02 in 30 iterations"
    );
    assert!((settled - truth[0]).abs() < 0.03, "α settled at {settled}");
}

/// Hero wavelengths: at the same spp, four rotated wavelengths per path cut
/// the per-pixel variance — most of it chroma noise from one-λ paths —
/// against one wavelength per path.
#[test]
fn hero_wavelengths_cut_colour_noise() {
    const SEEDS: u64 = 16;
    let variance = |wavelengths: usize| {
        let mut scene = Scene::new(24, 24, 2);
        scene.wavelengths = wavelengths;
        let mut k = Kernels::new(scene.lanes(), true);
        let imgs: Vec<Vec<f64>> = (0..SEEDS)
            .map(|s| prb::image(&scene, s, &prb::forward(&scene, &THETA0, s, &mut k).0))
            .collect();
        let n = imgs[0].len();
        let (mut total, mut chroma) = (0.0, 0.0);
        for px in 0..n / 3 {
            let mean = |f: &dyn Fn(&[f64]) -> f64| {
                imgs.iter().map(|i| f(&i[px * 3..px * 3 + 3])).sum::<f64>() / SEEDS as f64
            };
            let var = |f: &dyn Fn(&[f64]) -> f64| {
                let m = mean(f);
                imgs.iter()
                    .map(|i| (f(&i[px * 3..px * 3 + 3]) - m).powi(2))
                    .sum::<f64>()
                    / (SEEDS - 1) as f64
            };
            total += (0..3).map(|c| var(&|p: &[f64]| p[c])).sum::<f64>();
            // Chroma: each channel less the pixel's grey.
            chroma += (0..3)
                .map(|c| var(&|p: &[f64]| p[c] - (p[0] + p[1] + p[2]) / 3.0))
                .sum::<f64>();
        }
        (total, chroma)
    };
    let (t1, c1) = variance(1);
    let (t4, c4) = variance(4);
    eprintln!(
        "per-pixel variance at 2 spp, 1 λ vs 4 λ: total {:.2}×, chroma {:.2}× lower",
        t1 / t4,
        c1 / c4
    );
    assert!(c1 / c4 > 3.0, "chroma variance only {:.2}× lower", c1 / c4);
    assert!(t1 / t4 > 1.5, "total variance only {:.2}× lower", t1 / t4);
}
