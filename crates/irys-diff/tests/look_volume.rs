//! Phase 4, volumes: transmittance against the analytic Beer–Lambert, a
//! semi-infinite slab against Chandrasekhar's `H`, the event graph against
//! its twin, replay bit-exactness, and PRB against central differences.

use irys_core::{Point3, Vec3};
use irys_diff::look::volume::{
    self, COLUMNS, Grid, Medium, PARAMS, Track, VolEvent, VolumeScene,
};
use irys_diff::prb::{Mean, Rng};
use irys_diff::skin::optics;
use irys_diff::skin::render::{Light, hg_sample};

fn homogeneous(lo: Point3, hi: Point3, sigma: f64) -> Medium {
    Medium {
        density: Grid::constant(lo, hi, 1.0),
        sigma: [sigma; 3],
        g: 0.0,
        emission: None,
    }
}

#[test]
fn ratio_tracking_is_beer_lambert() {
    let m = homogeneous(Point3::new(0.0, -1.0, -1.0), Point3::new(10.0, 1.0, 1.0), 1.3);
    let th = [1.0, 0.8, 0.8, 0.8];
    let tr = Track::new(&m, th, th, 550.0);
    let mut rng = Rng::new(1, 0);
    for d in [0.2, 1.0, 3.0] {
        let n = 200_000;
        let (mut s, mut s2) = (0.0, 0.0);
        let mut esc = 0.0;
        for _ in 0..n {
            let mut ev = Vec::new();
            let t = tr.transmittance(Point3::origin(), Vec3::new(1.0, 0.0, 0.0), d, &mut rng, &mut ev);
            s += t;
            s2 += t * t;
            // Delta tracking: the free-flight escape probability.
            let (mut beta, mut l) = (1.0, 0.0);
            if tr
                .flight(Point3::origin(), Vec3::new(1.0, 0.0, 0.0), d, &mut rng, &mut beta, &mut l, 550.0, &mut None)
                .is_none()
            {
                esc += 1.0;
            }
        }
        let mean = s / n as f64;
        let sd = ((s2 / n as f64 - mean * mean) / n as f64).sqrt();
        let exact = (-1.3 * d).exp();
        let e = esc / n as f64;
        let esd = (exact * (1.0 - exact) / n as f64).sqrt();
        println!("d {d}: ratio {mean:.5} ± {sd:.5}, escape {e:.5} ± {esd:.5}, exact {exact:.5}");
        assert!((mean - exact).abs() < 4.0 * sd.max(1e-9));
        assert!((e - exact).abs() < 4.0 * esd);
    }
}

#[test]
fn heterogeneous_transmittance_matches_the_integral() {
    let (lo, hi) = (Point3::new(0.0, -1.0, -1.0), Point3::new(2.0, 1.0, 1.0));
    let m = Medium {
        density: Grid::from_fn(lo, hi, [32, 8, 8], |u| {
            0.2 + 1.8 * u[0] * u[0] + 0.5 * (6.0 * u[1]).sin().abs()
        }),
        sigma: [1.0, 0.7, 0.4],
        g: 0.0,
        emission: None,
    };
    let th = [1.5, 0.9, 0.9, 0.9];
    let lambda = 500.0;
    let tr = Track::new(&m, th, th, lambda);
    let (o, d) = (Point3::new(0.0, 0.13, -0.2), Vec3::new(1.0, 0.0, 0.0));
    let q = 200_000;
    let tau: f64 = (0..q)
        .map(|i| {
            let x = (i as f64 + 0.5) / q as f64 * 2.0;
            th[0] * m.density.at(o + d * x) * m.c(lambda) * 2.0 / q as f64
        })
        .sum();
    let exact = (-tau).exp();
    let mut rng = Rng::new(2, 0);
    let n = 200_000;
    let (mut s, mut s2) = (0.0, 0.0);
    for _ in 0..n {
        let mut ev = Vec::new();
        let t = tr.transmittance(o, d, 2.0, &mut rng, &mut ev);
        s += t;
        s2 += t * t;
    }
    let mean = s / n as f64;
    let sd = ((s2 / n as f64 - mean * mean) / n as f64).sqrt();
    println!("τ {tau:.4}: ratio {mean:.5} ± {sd:.5} exact {exact:.5}");
    assert!((mean - exact).abs() < 4.0 * sd);
}

/// Diffuse reflectance of a half-space by delta tracking.
fn half_space(w: f64, n: usize) -> (f64, f64) {
    let big = 1e4;
    let m = homogeneous(Point3::new(-big, -big, -big), Point3::new(big, big, 0.0), 1.0);
    let th = [1.0, w, w, w];
    let tr = Track::new(&m, th, th, 550.0);
    let mut rng = Rng::new(3, 0);
    let (mut s, mut s2) = (0.0, 0.0);
    for _ in 0..n {
        let c = irys_diff::look::openpbr::cosine([rng.uniform(), rng.uniform()]);
        let mut d = Vec3::new(c[0], c[1], -c[2]);
        let mut x = Point3::origin();
        let mut beta = 1.0;
        let mut l = 0.0;
        let mut out = 0.0;
        for _ in 0..100_000 {
            let tmax = if d.z > 0.0 { -x.z / d.z } else { f64::INFINITY };
            match tr.flight(x, d, tmax.min(1e6), &mut rng, &mut beta, &mut l, 550.0, &mut None) {
                None => {
                    out = beta;
                    break;
                }
                Some(t) => {
                    x += d * t;
                    d = hg_sample(d, 0.0, rng.uniform(), rng.uniform());
                }
            }
        }
        s += out;
        s2 += out * out;
    }
    let mean = s / n as f64;
    (mean, ((s2 / n as f64 - mean * mean) / n as f64).sqrt())
}

#[test]
fn half_space_matches_chandrasekhar() {
    for w in [0.3, 0.8, 0.95, 0.99] {
        let (r, sd) = half_space(w, 100_000);
        let exact = optics::albedo_diffuse(w);
        println!("ϖ {w}: {r:.5} ± {sd:.5} vs H {exact:.5} ({:.1} σ)", (r - exact) / sd);
        assert!((r - exact).abs() < 4.0 * sd);
    }
}

fn event_rows(n: usize) -> Vec<[f64; COLUMNS]> {
    let mut rng = Rng::new(4, 0);
    (0..n)
        .map(|_| {
            let real = (rng.uniform() < 0.4) as u8 as f64;
            let b = irys_diff::ggx::band_weights(380.0 + 400.0 * rng.uniform());
            [real, 1.0 - real, 0.2 + rng.uniform(), 0.3 + rng.uniform(), 6.0, b[0], b[1], b[2]]
        })
        .collect()
}

#[test]
fn event_graph_is_its_twin_and_its_adjoint_is_exact() {
    let n = 256;
    let rows = event_rows(n);
    let th = [1.2, 0.7, 0.8, 0.9];
    let g = VolEvent::new(n);
    let mut p = g.eval(false);
    for k in 0..COLUMNS {
        p.write(k, &rows.iter().map(|r| r[k]).collect::<Vec<_>>());
    }
    p.write(COLUMNS, &th);
    p.run();
    let l = p.read(0);
    let worst = rows
        .iter()
        .zip(&l)
        .map(|(r, &v)| (v - volume::ell(r, &th)).abs())
        .fold(0.0, f64::max);
    assert!(worst < 1e-12, "{worst}");
    let mut rng = Rng::new(5, 0);
    let ev: Vec<([f64; COLUMNS], f64)> = rows.iter().map(|r| (*r, rng.uniform() - 0.5)).collect();
    let mut a = g.adjoint(false);
    let gr = volume::reduce(&mut a, n, &ev, &th);
    for k in 0..PARAMS {
        let s = |h: f64| {
            let mut t = th;
            t[k] += h;
            ev.iter().map(|(r, w)| w * volume::ell(r, &t)).sum::<f64>()
        };
        let cd = (s(1e-6) - s(-1e-6)) / 2e-6;
        assert!((gr[k] - cd).abs() < 1e-6 * cd.abs().max(1e-3), "θ{k}: {} vs {cd}", gr[k]);
    }
}

pub fn smoke(size: usize, spp: usize, theta_s: Option<[f64; PARAMS]>) -> VolumeScene {
    let (lo, hi) = (Point3::new(-0.5, -0.5, 0.0), Point3::new(0.5, 0.5, 1.0));
    VolumeScene {
        medium: Medium {
            density: Grid::from_fn(lo, hi, [16, 16, 16], |u| {
                let r = ((u[0] - 0.5).powi(2) + (u[1] - 0.5).powi(2) + (u[2] - 0.4).powi(2)).sqrt();
                (1.0 - 2.0 * r).max(0.0) * (1.0 + 0.5 * (9.0 * u[2]).sin())
            }),
            sigma: [3.0, 3.5, 4.0],
            g: 0.4,
            emission: Some((Grid::from_fn(lo, hi, [8, 8, 8], |u| (0.3 - u[2]).max(0.0)), [3.0, 1.2, 0.3])),
        },
        floor: Some([0.5, 0.5, 0.5]),
        lights: vec![Light::facing(Point3::new(1.0, -1.0, 2.5), Point3::new(0.0, 0.0, 0.4), 1.0, 1.0, 8.0, [1.0; 3])],
        env: 0.05,
        eye: Point3::new(0.0, -3.0, 1.0),
        target: Point3::new(0.0, 0.0, 0.4),
        vfov: 35.0,
        size: (size, size),
        spp,
        depth: 6,
        sampler: theta_s,
    }
}

#[test]
fn replay_is_bit_exact_and_prb_matches_central_differences() {
    let th = [1.0, 0.8, 0.85, 0.9];
    let s = smoke(16, 4, Some(th));
    let fwd = s.forward(&th, 3);
    let (_, g, ls) = s.gradient(&th, 3, &Mean, false);
    assert!(fwd.iter().zip(&ls).all(|(a, b)| a.0.to_bits() == b.to_bits()), "replay re-walks");
    let (_, g2, _) = s.gradient(&th, 3, &Mean, false);
    assert_eq!(g.map(f64::to_bits), g2.map(f64::to_bits), "δθ bit-identical");
    let mut worst: f64 = 0.0;
    for k in 0..PARAMS {
        let h = 1e-5;
        let loss = |d: f64| {
            let mut t = th;
            t[k] += d;
            let img = s.image(&s.forward(&t, 3));
            img.iter().sum::<f64>() / img.len() as f64
        };
        let cd = (loss(h) - loss(-h)) / (2.0 * h);
        let rel = (g[k] - cd).abs() / cd.abs().max(1e-6);
        println!("θ{k}: prb {:+.8e} cd {:+.8e} rel {rel:.1e}", g[k], cd);
        worst = worst.max(rel);
    }
    assert!(worst < 1e-5, "{worst}");
}
