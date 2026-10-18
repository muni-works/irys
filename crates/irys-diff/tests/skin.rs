//! Skin: the graphs against their f64 twins, replay bit-exactness through
//! random walks, and PRB gradients against central differences.
//!
//! The scene is synthetic (a closed UV sphere with constant maps) so the
//! gates need no asset; `skin_head.rs` runs the head.

mod common;

use common::{THETA, ball};
use irys_diff::prb::{L2, Mean, Rng};
use irys_diff::skin::graph::{self, MEDIUM_COLUMNS, Medium, PARAMS, SPEC_COLUMNS, Spec};
use irys_diff::skin::render::{self, Adjoints, Mode, NW};

const SSS: Mode = Mode {
    sss: true,
    dual: true,
};
const LAMBERT: Mode = Mode {
    sss: false,
    dual: false,
};

fn columns(n: usize, seed: u64, f: impl Fn(&mut Rng) -> Vec<f64>) -> Vec<Vec<f64>> {
    let mut rng = Rng::new(seed, 0);
    let rows: Vec<Vec<f64>> = (0..n).map(|_| f(&mut rng)).collect();
    (0..rows[0].len())
        .map(|c| rows.iter().map(|r| r[c]).collect())
        .collect()
}

fn spec_rows(n: usize) -> Vec<Vec<f64>> {
    columns(n, 3, |r| {
        let ci = 0.05 + 0.95 * r.uniform();
        let co = 0.05 + 0.95 * r.uniform();
        let ch = 0.2 + 0.8 * r.uniform();
        let oh = 0.05 + 0.95 * r.uniform();
        vec![ci, co, ch, oh, 0.2 + r.uniform()]
    })
}

fn medium_rows(n: usize) -> Vec<Vec<f64>> {
    columns(n, 5, |r| {
        let lam = 380.0 + 400.0 * r.uniform();
        let c = irys_diff::skin::optics::coefficients(lam);
        let s = if r.uniform() < 0.7 { 1.0 } else { 0.0 };
        vec![
            s,
            0.5 * r.uniform(),
            r.uniform(),
            r.uniform(),
            c[0],
            c[1],
            c[2],
            c[3],
        ]
    })
}

#[test]
fn spec_graph_is_its_f64_twin() {
    let n = 64;
    let cols = spec_rows(n);
    let b = Spec::build(n);
    let mut p = b.eval(false);
    for (k, c) in cols.iter().enumerate() {
        p.write(k, c);
    }
    p.write(SPEC_COLUMNS, &THETA);
    p.run();
    let f = p.read(0);
    for j in 0..n {
        let want = graph::spec_f64(
            &THETA, cols[0][j], cols[1][j], cols[2][j], cols[3][j], cols[4][j],
        );
        assert!(
            (f[j] - want).abs() <= 1e-12 * want.abs().max(1.0),
            "{j}: {} vs {want}",
            f[j]
        );
    }
}

#[test]
fn medium_graph_is_its_f64_twin_and_its_adjoint_is_exact() {
    let n = 64;
    let cols = medium_rows(n);
    let b = Medium::build(n);
    let mut p = b.eval(false);
    let mut a = b.adjoint(false);
    let mut dl = vec![0.0; n];
    let mut want = [0.0; PARAMS];
    let mut rng = Rng::new(9, 0);
    for j in 0..n {
        let row: [f64; MEDIUM_COLUMNS] = std::array::from_fn(|k| cols[k][j]);
        dl[j] = rng.uniform() - 0.5;
        let d = graph::medium_dell_f64(&THETA, &row);
        for k in 0..PARAMS {
            want[k] += dl[j] * d[k];
        }
    }
    for (k, c) in cols.iter().enumerate() {
        p.write(k, c);
        a.write(k, c);
    }
    p.write(MEDIUM_COLUMNS, &THETA);
    a.write(MEDIUM_COLUMNS, &THETA);
    a.write(MEDIUM_COLUMNS + 1, &dl);
    p.run();
    a.run();
    let f = p.read(0);
    for j in 0..n {
        let row: [f64; MEDIUM_COLUMNS] = std::array::from_fn(|k| cols[k][j]);
        let e = graph::medium_ell_f64(&THETA, &row).exp();
        assert!((f[j] - e).abs() <= 1e-12 * e.max(1e-300), "{j}");
    }
    let got = a.read(0);
    for k in 0..PARAMS {
        assert!(
            (got[k] - want[k]).abs() <= 1e-10 * want[k].abs().max(1e-12),
            "θ{k}: {} vs {}",
            got[k],
            want[k]
        );
    }
}

#[test]
fn spec_adjoint_matches_central_differences() {
    let n = 32;
    let cols = spec_rows(n);
    let b = Spec::build(n);
    let mut a = b.adjoint(false);
    let df: Vec<f64> = (0..n)
        .map(|j| ((j * 7919) % 13) as f64 / 13.0 - 0.4)
        .collect();
    for (k, c) in cols.iter().enumerate() {
        a.write(k, c);
    }
    a.write(SPEC_COLUMNS, &THETA);
    a.write(SPEC_COLUMNS + 1, &df);
    a.run();
    let got = a.read(0);
    let s = |th: &[f64; PARAMS]| -> f64 {
        (0..n)
            .map(|j| {
                df[j]
                    * graph::spec_f64(
                        th, cols[0][j], cols[1][j], cols[2][j], cols[3][j], cols[4][j],
                    )
            })
            .sum()
    };
    for k in 3..5 {
        let h = 1e-6;
        let (mut lo, mut hi) = (THETA, THETA);
        lo[k] -= h;
        hi[k] += h;
        let cd = (s(&hi) - s(&lo)) / (2.0 * h);
        assert!(
            (got[k] - cd).abs() < 1e-6 * cd.abs().max(1e-3),
            "θ{k}: {} vs {cd}",
            got[k]
        );
    }
    assert_eq!(&got[..3], &[0.0; 3]);
}

#[test]
fn compiled_adjoints_agree_with_the_interpreter() {
    let n = 256;
    let native = Spec::build(n).adjoint(true);
    if !native.is_native() {
        eprintln!("no C compiler: skipped");
        return;
    }
    let s = ball(10, 2, SSS);
    let (l, _) = render::forward(&s, &THETA, 4);
    let dl: Vec<f64> = (0..l.len()).map(|j| 1.0 / (1.0 + j as f64)).collect();
    let (gi, _, _) = render::replay(&s, &THETA, 4, &l, &dl, &mut Adjoints::new(n, false));
    let (gc, _, _) = render::replay(&s, &THETA, 4, &l, &dl, &mut Adjoints::new(n, true));
    for k in 0..PARAMS {
        let rel = (gi[k] - gc[k]).abs() / gi[k].abs().max(1e-12);
        println!(
            "θ{k}: interp {:.6e} compiled {:.6e} rel {rel:.1e}",
            gi[k], gc[k]
        );
        assert!(rel < 1e-4, "θ{k}");
    }
}

#[test]
fn replay_is_bit_exact_through_the_walks() {
    for mode in [SSS, LAMBERT] {
        let s = ball(16, 2, mode);
        let (l, c) = render::forward(&s, &THETA, 21);
        if mode.sss {
            assert!(c.walks >= 500 && c.events > 50_000, "{c:?}");
        }
        let dl: Vec<f64> = (0..l.len()).map(|j| (j % 7) as f64 - 3.0).collect();
        let mut adj = Adjoints::new(1024, false);
        let (g1, l2, recs) = render::replay(&s, &THETA, 21, &l, &dl, &mut adj);
        let same = l
            .iter()
            .zip(&l2)
            .filter(|(a, b)| a.to_bits() == b.to_bits())
            .count();
        println!(
            "{mode:?}: {same}/{} lanes bit-identical, {recs} records, {c:?}",
            l.len()
        );
        assert_eq!(same, l.len());
        let (g2, _, _) = render::replay(&s, &THETA, 21, &l, &dl, &mut adj);
        assert_eq!(
            g1.map(f64::to_bits),
            g2.map(f64::to_bits),
            "δθ not deterministic"
        );
        // And a second forward is the first.
        let (l3, _) = render::forward(&s, &THETA, 21);
        assert!(l.iter().zip(&l3).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
}

/// Central differences with common random numbers and frozen sampling: the
/// estimator PRB differentiates, so the two agree to rounding.
fn check_cd(mode: Mode, params: &[usize], tol: f64) {
    let mut s = ball(12, 2, mode);
    s.frozen = Some(THETA);
    let seed = 33;
    let mut target_theta = THETA;
    target_theta[0] *= 1.3;
    target_theta[3] = 0.42;
    let (lt, _) = render::forward(&s, &target_theta, 77);
    let target = render::image(&s, 77, &lt);
    let mut adj = Adjoints::new(4096, false);
    for (name, loss) in [("mean", 0usize), ("l2", 1)] {
        let (_, g) = if loss == 0 {
            render::gradient(&s, &THETA, (seed, seed), &Mean, &mut adj)
        } else {
            render::gradient(&s, &THETA, (seed, seed), &L2(&target), &mut adj)
        };
        for &k in params {
            let h = 1e-5 * THETA[k].abs().max(1e-2);
            let (mut lo, mut hi) = (THETA, THETA);
            lo[k] -= h;
            hi[k] += h;
            let f = |th: &[f64; PARAMS]| {
                if loss == 0 {
                    render::loss(&s, th, seed, &Mean)
                } else {
                    render::loss(&s, th, seed, &L2(&target))
                }
            };
            let cd = (f(&hi) - f(&lo)) / (2.0 * h);
            let rel = (g[k] - cd).abs() / cd.abs().max(1e-14);
            println!(
                "{mode:?} {name} θ{k}: prb {:+.9e} cd {cd:+.9e} rel {rel:.1e}",
                g[k]
            );
            assert!(rel < tol, "{mode:?} {name} θ{k}: {} vs {cd}", g[k]);
        }
    }
}

#[test]
fn sss_gradient_matches_central_differences() {
    check_cd(SSS, &[0, 1, 2, 3, 4], 1e-5);
}

#[test]
fn lambert_gradient_matches_central_differences() {
    check_cd(LAMBERT, &[3], 1e-5);
}

#[test]
fn live_sampler_gradient_is_unbiased() {
    // Paths move with θ (distances drawn at the current density): average
    // PRB over seeds against central differences over the same seeds.
    let s = ball(8, 2, SSS);
    let mut adj = Adjoints::new(4096, false);
    let k = 0;
    let n = 24;
    let (mut gs, mut cds) = (Vec::new(), Vec::new());
    for seed in 0..n {
        let (_, g) = render::gradient(&s, &THETA, (seed, seed), &Mean, &mut adj);
        gs.push(g[k]);
        let h = 2e-3;
        let (mut lo, mut hi) = (THETA, THETA);
        lo[k] -= h;
        hi[k] += h;
        cds.push(
            (render::loss(&s, &hi, seed, &Mean) - render::loss(&s, &lo, seed, &Mean)) / (2.0 * h),
        );
    }
    let stat = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
        (m, (var / v.len() as f64).sqrt())
    };
    let ((gm, gse), (cm, cse)) = (stat(&gs), stat(&cds));
    let z = (gm - cm) / (gse * gse + cse * cse).sqrt();
    println!(
        "melanin, live sampler, {n} seeds: prb {gm:.4e} ± {gse:.1e}, cd {cm:.4e} ± {cse:.1e}, {z:+.1}σ"
    );
    assert!(z.abs() < 3.5);
    assert!(gm < 0.0, "more melanin darkens");
}

#[test]
fn records_are_per_walk_not_per_event() {
    let s = ball(8, 1, SSS);
    let (l, c) = render::forward(&s, &THETA, 2);
    let dl = vec![1.0; l.len()];
    let (_, _, recs) = render::replay(&s, &THETA, 2, &l, &dl, &mut Adjoints::new(512, false));
    println!("{} events, {} walks, {recs} records", c.events, c.walks);
    assert!((recs as u64) < c.walks * 2 + (s.paths() * NW * 4) as u64);
    assert!(c.events > 20 * c.walks);
}
