//! The random walk against exact transport: semi-infinite slabs.
//!
//! `cargo test -p irys-diff --release --test skin_sss -- --nocapture`
//! prints the table design.md's "Skin" section quotes.

use rayon::prelude::*;

use irys_core::{Bvh, Point3, Vec3};
use irys_diff::prb::{Rng, cuboid};
use irys_diff::skin::accel::Accel;
use irys_diff::skin::optics;
use irys_diff::skin::render::{hg_sample, walk_g};

/// A slab much thicker than any walk reaches: σt = 1 per unit, 400 deep,
/// 4000 wide. Entry at the top centre.
fn slab() -> Bvh<irys_core::TriMesh> {
    Bvh::build(cuboid(2000.0, 2000.0, 200.0))
}

/// Mean weight of `n` walks entering at the top (cosine-distributed if
/// `diffuse`, straight down otherwise) at albedo `w`, asymmetry `g`; with
/// its standard error.
fn albedo(
    bvh: &Bvh<irys_core::TriMesh>,
    w: f64,
    g: f64,
    diffuse: bool,
    n: usize,
    seed: u64,
) -> (f64, f64) {
    let accel = Accel::new(bvh);
    let r: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut rng = Rng::new(seed, i as u64);
            let dir = if diffuse {
                let (u1, u2) = (rng.uniform(), rng.uniform());
                let (r, phi) = (u1.sqrt(), 2.0 * std::f64::consts::PI * u2);
                Vec3::new(r * phi.cos(), r * phi.sin(), -(1.0 - u1).max(0.0).sqrt())
            } else {
                Vec3::new(0.0, 0.0, -1.0)
            };
            let _ = hg_sample;
            let m = (1.0 - w, w);
            let wk = walk_g(
                bvh,
                Some(&accel),
                Point3::new(0.0, 0.0, 200.0 - 1e-9),
                dir,
                m,
                m,
                1 << 20,
                &mut rng,
                g,
            );
            match wk.exit {
                Some((h, _, _)) if h.point.z > 199.0 => wk.weight,
                _ => 0.0,
            }
        })
        .collect();
    let mean = r.iter().sum::<f64>() / n as f64;
    let var = r.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    (mean, (var / n as f64).sqrt())
}

#[test]
fn h_function_identities() {
    // ∫ H dμ = (2/ϖ)(1 − √(1 − ϖ)) (Chandrasekhar's zeroth moment).
    for w in [0.2, 0.5, 0.9, 0.99, 0.9999] {
        let nodes = optics::h_function(w, 48);
        let h0: f64 = nodes.iter().map(|&(_, q, h)| q * h).sum();
        let exact = 2.0 / w * (1.0 - (1.0 - w).sqrt());
        assert!((h0 - exact).abs() < 1e-10, "ϖ {w}: H0 {h0} vs {exact}");
    }
    // Conservative limit: no absorption reflects everything.
    assert!((optics::albedo_diffuse(1.0 - 1e-12) - 1.0).abs() < 1e-5);
}

#[test]
fn walk_matches_chandrasekhar() {
    let bvh = slab();
    println!("| ϖ | incidence | walk (± 1σ) | exact | rel | σ |");
    println!("|---|---|---|---|---|---|");
    for (w, n) in [
        (0.3, 400_000),
        (0.6, 400_000),
        (0.9, 200_000),
        (0.99, 100_000),
        (0.999, 50_000),
    ] {
        for diffuse in [true, false] {
            let (a, se) = albedo(&bvh, w, 0.0, diffuse, n, 7);
            let exact = if diffuse {
                optics::albedo_diffuse(w)
            } else {
                optics::albedo_collimated(w, 1.0)
            };
            let z = (a - exact) / se;
            println!(
                "| {w} | {} | {a:.5} ± {se:.5} | {exact:.5} | {:+.2e} | {z:+.1} |",
                if diffuse { "diffuse" } else { "normal" },
                (a - exact) / exact
            );
            assert!(
                z.abs() < 4.0,
                "ϖ {w} diffuse {diffuse}: {a} ± {se} vs {exact}"
            );
        }
    }
}

#[test]
fn similarity_for_skin_anisotropy() {
    // g = 0.8 has no closed form; the similarity relation is the inversion's
    // model of it. Report how far it is, and hold it to 3% absolute.
    let bvh = slab();
    println!("| ϖ (g = 0.8) | walk | similarity ϖ* | A(ϖ*) | abs |");
    println!("|---|---|---|---|---|");
    for (w, n) in [
        (0.9, 100_000),
        (0.98, 50_000),
        (0.995, 40_000),
        (0.999, 20_000),
    ] {
        let (a, se) = albedo(&bvh, w, 0.8, true, n, 11);
        let ws = optics::similar(w, 0.8);
        let pred = optics::walk_albedo(ws);
        println!(
            "| {w} | {a:.4} ± {se:.4} | {ws:.4} | {pred:.4} | {:+.4} |",
            a - pred
        );
        assert!(
            (a - pred).abs() < 0.03 + 4.0 * se,
            "ϖ {w}: walk {a} vs similarity {pred}"
        );
    }
}

/// Cycles' subsurface mode (`look::openpbr::SubsurfaceMode::Cycles`) maps a
/// colour to a single-scattering albedo with Chiang+ 2016's fit; the walk
/// it drives on a half-space, entered diffusely, returns about that colour.
#[test]
fn chiang_fit_inverts_the_walk() {
    use irys_diff::look::openpbr::chiang_alpha;
    let bvh = slab();
    assert_eq!(chiang_alpha(0.0, 0.0), 0.0);
    let mut last = 0.0;
    for k in 1..=20 {
        let a = chiang_alpha(k as f64 / 20.0, 0.0);
        assert!(a >= last && a < 1.0, "monotone in (0, 1): {k} {a} {last}");
        last = a;
    }
    for c in [0.2, 0.5, 0.8, 0.95] {
        let a = chiang_alpha(c, 0.0);
        let (e, s) = albedo(&bvh, a, 0.0, false, 100_000, 11);
        let (ed, _) = albedo(&bvh, a, 0.0, true, 100_000, 11);
        println!("colour {c:.2}: α {a:.5} → walk {e:.4} ± {s:.4} diffuse {ed:.4}");
        // The fit is for diffuse incidence; at 0.95 it saturates (α
        // clamped to 1 − 1e-6, as in Cycles).
        let tol = if c > 0.9 { 0.05 } else { 0.015 };
        assert!((ed - c).abs() < tol, "colour {c}: walk {ed}");
    }
}
