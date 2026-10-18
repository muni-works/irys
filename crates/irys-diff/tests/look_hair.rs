//! Phase 4, hair: the Chiang BCSDF's white furnace and pdf normalisation,
//! importance sampling against uniform sampling, the graph against its
//! twin, replay bit-exactness and PRB against central differences.

use irys_core::{Bvh, Point3, Vec3};
use irys_diff::look::hair::{self, Bcsdf, COLUMNS, Fiber, Hair, HairScene, PARAMS, Strands};
use irys_diff::prb::{Mean, Rng};
use irys_diff::skin::render::Light;

fn uniform_sphere(u: f64, v: f64) -> [f64; 3] {
    let z = 1.0 - 2.0 * u;
    let r = (1.0 - z * z).max(0.0).sqrt();
    let p = 2.0 * std::f64::consts::PI * v;
    [z, r * p.cos(), r * p.sin()]
}

fn wo_at(theta_o: f64) -> [f64; 3] {
    [theta_o.sin(), 0.0, theta_o.cos()]
}

#[test]
fn white_furnace_and_pdf_normalisation() {
    let fiber = Fiber::default();
    let mut rng = Rng::new(1, 0);
    let n = 400_000;
    for (h, to) in [(0.0, 0.0), (0.5, 0.4), (-0.8, -0.9), (0.95, 1.2)] {
        let bc = Bcsdf::new(&fiber, h);
        let wo = wo_at(to);
        let ap = bc.ap(wo, 550.0, &[0.0, 0.0]);
        let (mut e, mut p, mut is) = (0.0, 0.0, 0.0);
        for _ in 0..n {
            let wi = uniform_sphere(rng.uniform(), rng.uniform());
            let c = bc.columns(wo, wi, 550.0);
            e += hair::hair_f(&c, &[0.0, 0.0]) * 4.0 * std::f64::consts::PI;
            p += bc.pdf(wo, wi, &ap) * 4.0 * std::f64::consts::PI;
            let ws = bc.sample(wo, &ap, [rng.uniform(), rng.uniform(), rng.uniform(), rng.uniform()]);
            let c = bc.columns(wo, ws, 550.0);
            is += hair::hair_f(&c, &[0.0, 0.0]) / bc.pdf(wo, ws, &ap);
        }
        let (e, p, is) = (e / n as f64, p / n as f64, is / n as f64);
        println!("h {h:+.2} θo {to:+.2}: ∫f {e:.4} ∫pdf {p:.4} E[f/pdf] {is:.4}");
        assert!((e - 1.0).abs() < 0.02, "furnace {e}");
        assert!((p - 1.0).abs() < 0.02, "pdf {p}");
        assert!((is - e).abs() < 0.02, "sampling {is} vs {e}");
    }
}

#[test]
fn melanin_darkens_and_reddens() {
    let bc = Bcsdf::new(&Fiber::default(), 0.3);
    let wo = wo_at(0.2);
    let mut rng = Rng::new(2, 0);
    let albedo = |th: [f64; 2], lambda: f64, rng: &mut Rng| {
        let n = 100_000;
        (0..n)
            .map(|_| {
                let wi = uniform_sphere(rng.uniform(), rng.uniform());
                hair::hair_f(&bc.columns(wo, wi, lambda), &th) * 4.0 * std::f64::consts::PI
            })
            .sum::<f64>()
            / n as f64
    };
    let blonde = albedo([0.3, 0.0], 650.0, &mut rng) / albedo([0.3, 0.0], 450.0, &mut rng);
    let black = albedo([8.0, 0.0], 550.0, &mut rng);
    let red = albedo([0.3, 2.0], 650.0, &mut rng) / albedo([0.3, 2.0], 450.0, &mut rng);
    println!("blonde red/blue {blonde:.3}, black albedo {black:.4}, red-head red/blue {red:.3}");
    assert!(blonde > 1.2 && red > blonde && black < 0.1);
}

#[test]
fn hair_graph_is_its_twin_and_its_adjoint_is_exact() {
    let n = 200;
    let bc = Bcsdf::new(&Fiber::default(), 0.4);
    let mut rng = Rng::new(3, 0);
    let rows: Vec<[f64; COLUMNS]> = (0..n)
        .map(|_| {
            let wo = uniform_sphere(rng.uniform(), rng.uniform());
            let wi = uniform_sphere(rng.uniform(), rng.uniform());
            bc.columns(wo, wi, 380.0 + 400.0 * rng.uniform())
        })
        .collect();
    let th = [1.3, 0.4];
    let g = Hair::new(n);
    for native in [false, true] {
        let mut p = g.eval(native);
        for k in 0..COLUMNS {
            p.write(k, &rows.iter().map(|r| r[k]).collect::<Vec<_>>());
        }
        p.write(COLUMNS, &th);
        p.run();
        let f = p.read(0);
        let tol = if p.is_native() { 1e-4 } else { 1e-12 };
        let worst = rows
            .iter()
            .zip(&f)
            .map(|(r, &v)| (v - hair::hair_f(r, &th)).abs() / hair::hair_f(r, &th).abs().max(1e-3))
            .fold(0.0, f64::max);
        println!("native {}: worst {worst:.2e}", p.is_native());
        assert!(worst < tol);
    }
    let lanes: Vec<_> = rows.iter().map(|r| (*r, rng.uniform() - 0.5)).collect();
    let mut a = g.adjoint(false);
    let gr = hair::reduce(&mut a, n, &lanes, &th);
    for k in 0..PARAMS {
        let s = |h: f64| {
            let mut t = th;
            t[k] += h;
            lanes.iter().map(|(r, w)| w * hair::hair_f(r, &t)).sum::<f64>()
        };
        let cd = (s(1e-6) - s(-1e-6)) / 2e-6;
        assert!((gr[k] - cd).abs() < 1e-6 * cd.abs().max(1e-3), "θ{k}: {} vs {cd}", gr[k]);
    }
}

/// A clump of wavy strands hanging in front of a light.
pub fn clump(size: usize, spp: usize, sampler: Option<[f64; PARAMS]>) -> HairScene {
    let mut s = Strands::default();
    let mut rng = Rng::new(9, 0);
    for _ in 0..400 {
        let (x0, y0) = (0.3 * (rng.uniform() - 0.5), 0.3 * (rng.uniform() - 0.5));
        let ph = 6.28 * rng.uniform();
        let pts: Vec<Point3> = (0..=24)
            .map(|k| {
                let t = k as f64 / 24.0;
                Point3::new(x0 + 0.04 * (8.0 * t + ph).sin(), y0 + 0.04 * (7.0 * t + ph).cos(), 1.0 - t)
            })
            .collect();
        s.push(&pts, 0.004);
    }
    HairScene {
        bvh: Bvh::build(s),
        fiber: Fiber::default(),
        scalp: None,
        lights: vec![Light::facing(Point3::new(-1.0, -1.5, 1.5), Point3::new(0.0, 0.0, 0.5), 1.0, 1.0, 6.0, [1.0; 3])],
        env: 0.1,
        eye: Point3::new(0.0, -2.0, 0.5),
        target: Point3::new(0.0, 0.0, 0.5),
        vfov: 30.0,
        size: (size, size),
        spp,
        depth: 12,
        sampler,
    }
}

#[test]
fn replay_is_bit_exact_and_prb_matches_central_differences() {
    let th = [0.8, 0.3];
    let s = clump(24, 4, Some(th));
    let fwd = s.forward(&th, 4);
    let (_, g, ls) = s.gradient(&th, 4, &Mean, false);
    assert!(fwd.iter().zip(&ls).all(|(a, b)| a.0.to_bits() == b.to_bits()));
    let (_, g2, _) = s.gradient(&th, 4, &Mean, false);
    assert_eq!(g.map(f64::to_bits), g2.map(f64::to_bits));
    for k in 0..PARAMS {
        let h = 1e-5;
        let loss = |d: f64| {
            let mut t = th;
            t[k] += d;
            let img = s.image(&s.forward(&t, 4));
            img.iter().sum::<f64>() / img.len() as f64
        };
        let cd = (loss(h) - loss(-h)) / (2.0 * h);
        let rel = (g[k] - cd).abs() / cd.abs().max(1e-7);
        println!("θ{k}: prb {:+.8e} cd {:+.8e} rel {rel:.1e}", g[k], cd);
        assert!(rel < 1e-5);
    }
    let _ = Vec3::zeros();
}
