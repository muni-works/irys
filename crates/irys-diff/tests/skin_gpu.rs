//! The skin megakernel against the CPU f64 integrator, and its determinism.
//! Skips without an adapter.
#![cfg(feature = "gpu")]

mod common;

use common::{THETA, ball};
use irys_diff::skin::gpu::GpuSkin;
use irys_diff::skin::render::{self, Mode, NW};

fn ctx() -> Option<&'static irys_gpu::GpuContext> {
    match irys_diff::gpu::context() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("no GPU ({e:?}): skipped");
            None
        }
    }
}

fn rel_stats(cpu: &[f64], gpu: &[f32]) -> (f64, f64, usize) {
    let mut r: Vec<f64> = cpu
        .iter()
        .zip(gpu)
        .map(|(&c, &g)| (c - g as f64).abs() / c.abs().max(1e-3))
        .collect();
    let big = r.iter().filter(|&&x| x > 1e-3).count();
    r.sort_by(f64::total_cmp);
    (r[r.len() / 2], r[r.len() * 99 / 100], big)
}

#[test]
fn lambert_lanes_match_the_cpu() {
    let Some(ctx) = ctx() else { return };
    let s = ball(
        32,
        2,
        Mode {
            sss: false,
            dual: true,
        },
    );
    let (l, _) = render::forward(&s, &THETA, 5);
    let g = GpuSkin::new(ctx, &s, true);
    let (img, lg) = g.render(&s, &THETA, 5, true);
    let (med, p99, big) = rel_stats(&l, &lg);
    println!(
        "lambert: {} lanes, L median rel {med:.1e}, p99 {p99:.1e}, {big} > 1e-3",
        l.len()
    );
    assert!(med < 1e-5 && p99 < 1e-2, "median {med} p99 {p99}");
    assert!((big as f64) < 0.01 * l.len() as f64);
    let ci = render::image(&s, 5, &l);
    let (sc, sg) = (
        ci.iter().sum::<f64>(),
        img.iter().map(|&x| x as f64).sum::<f64>(),
    );
    println!(
        "lambert image sum: cpu {sc:.6} gpu {sg:.6} rel {:.1e}",
        (sc - sg).abs() / sc
    );
    assert!((sc - sg).abs() / sc < 1e-4);
}

#[test]
fn sss_matches_the_cpu() {
    let Some(ctx) = ctx() else { return };
    // Per lane until f32 rounding moves a walk's branch, then statistically.
    let s = ball(
        24,
        32,
        Mode {
            sss: true,
            dual: true,
        },
    );
    let (l, _) = render::forward(&s, &THETA, 9);
    let g = GpuSkin::new(ctx, &s, true);
    let (img, lg) = g.render(&s, &THETA, 9, true);
    let (med, p99, big) = rel_stats(&l, &lg);
    println!(
        "sss: {} lanes, L median rel {med:.1e}, p99 {p99:.1e}, {big} > 1e-3",
        l.len()
    );
    assert!(med < 1e-4, "median {med}");
    let ci = render::image(&s, 9, &l);
    // Frame mean, with the standard error from per-path sums.
    let per_path: Vec<f64> = l.chunks(NW).map(|c| c.iter().sum()).collect();
    let n = per_path.len() as f64;
    let m = per_path.iter().sum::<f64>() / n;
    let se = (per_path.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt();
    let mg = lg
        .chunks(NW)
        .map(|c| c.iter().map(|&x| x as f64).sum::<f64>())
        .sum::<f64>()
        / n;
    println!(
        "sss mean L per path: cpu {m:.5} gpu {mg:.5} (se {se:.1e}), {:+.2}σ",
        (mg - m) / se
    );
    assert!(
        ((mg - m) / se).abs() < 1.0,
        "the two tiers walk the same paths; the means must agree well inside noise"
    );
    let (sc, sg) = (
        ci.iter().sum::<f64>(),
        img.iter().map(|&x| x as f64).sum::<f64>(),
    );
    println!(
        "sss image sum: cpu {sc:.5} gpu {sg:.5} rel {:.1e}",
        (sc - sg).abs() / sc
    );
}

#[test]
fn gpu_frames_are_deterministic() {
    let Some(ctx) = ctx() else { return };
    let s = ball(
        24,
        4,
        Mode {
            sss: true,
            dual: true,
        },
    );
    let g = GpuSkin::new(ctx, &s, false);
    let (a, _) = g.render(&s, &THETA, 3, false);
    let (b, _) = g.render(&s, &THETA, 3, false);
    assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
}
