//! Phase 4, OpenPBR: the stack graph against its twin, white furnaces and
//! energy per lobe, reciprocity, the film and the dispersion, replay
//! bit-exactness and PRB against central differences.

use irys_diff::look::graph::{self, Stack, THETA};
use irys_diff::look::openpbr::{
    self, At, COLUMNS, Kind, OpenPbr, PARAMS, Tables, airy, fresnel, subsurface_medium,
};
use irys_diff::look::swatch::Swatch;
use irys_diff::prb::{Mean, Rng};

fn dir(rng: &mut Rng, lo: f64) -> [f64; 3] {
    let z = lo + (1.0 - lo) * rng.uniform();
    let p = 2.0 * std::f64::consts::PI * rng.uniform();
    let s = (1.0 - z * z).sqrt();
    [s * p.cos(), s * p.sin(), z]
}

fn busy() -> OpenPbr {
    OpenPbr {
        base_color: [0.7, 0.45, 0.25],
        base_metalness: 0.35,
        specular_roughness: 0.4,
        coat_weight: 0.4,
        coat_roughness: 0.15,
        coat_color: [0.9, 0.8, 0.7],
        fuzz_weight: 0.3,
        fuzz_color: [0.8, 0.8, 0.9],
        thin_film_weight: 0.5,
        thin_film_thickness: 0.4,
        transmission_weight: 0.2,
        subsurface_weight: 0.2,
        ..Default::default()
    }
}

/// Lanes of every kind for `m`, at random directions and wavelengths.
fn lanes(m: &OpenPbr, n: usize, seed: u64) -> Vec<[f64; COLUMNS]> {
    let mut rng = Rng::new(seed, 0);
    (0..n)
        .map(|k| {
            let lambda = 380.0 + 400.0 * rng.uniform();
            let at = At::new(m, lambda, false);
            let wo = dir(&mut rng, 0.05);
            let mut wi = dir(&mut rng, 0.05);
            let kind = match k % 3 {
                0 => Kind::Reflect,
                1 => {
                    wi[2] = -wi[2];
                    Kind::Transmit
                }
                _ => Kind::Subsurface,
            };
            at.columns(kind, wo, wi)
        })
        .collect()
}

fn write_cols(p: &mut irys_diff::ir::Program, rows: &[[f64; COLUMNS]]) {
    let v: Vec<f64> = rows.iter().flatten().copied().collect();
    p.write(0, &v);
}

#[test]
fn stack_graph_is_its_f64_twin() {
    let m = busy();
    let theta = m.theta();
    let n = 300;
    let rows = lanes(&m, n, 1);
    let g = Stack::new(n);
    for native in [false, true] {
        let mut p = g.eval(native);
        write_cols(&mut p, &rows);
        p.write(THETA, &theta);
        p.run();
        let f = p.read(0);
        let tol = if p.is_native() { 2e-4 } else { 1e-12 };
        let mut worst: f64 = 0.0;
        for (r, &fg) in rows.iter().zip(&f) {
            let ft = graph::stack(r, &theta);
            worst = worst.max((fg - ft).abs() / ft.abs().max(1e-3));
        }
        println!("native {}: worst rel {worst:.2e}", p.is_native());
        assert!(worst < tol, "graph vs twin {worst}");
    }
}

#[test]
fn stack_adjoint_matches_central_differences() {
    let m = busy();
    let theta = m.theta();
    let n = 90;
    let rows = lanes(&m, n, 2);
    let mut rng = Rng::new(9, 0);
    let df: Vec<f64> = (0..n).map(|_| rng.uniform() - 0.5).collect();
    let mut p = Stack::new(n).adjoint(false);
    let g = graph::reduce(&mut p, n, &rows, &df, &theta);
    for k in 0..PARAMS {
        let h = 1e-6;
        let s = |d: f64| {
            let mut t = theta;
            t[k] += d;
            rows.iter()
                .zip(&df)
                .map(|(r, w)| w * graph::stack(r, &t))
                .sum::<f64>()
        };
        let cd = (s(h) - s(-h)) / (2.0 * h);
        println!("{:>20}: ad {:+.9e} cd {:+.9e}", openpbr::PARAM_NAMES[k], g[k], cd);
        assert!((g[k] - cd).abs() <= 1e-6 * cd.abs().max(1e-3), "θ{k}");
    }
}

/// `∫ f |cos| dω` over the sphere at `ω_o` (cosine `mu`) by importance
/// sampling the material's own lobes, plus the subsurface weight (a walk
/// that loses nothing returns it all).
fn albedo(m: &OpenPbr, mu: f64, lambda: f64, n: usize) -> f64 {
    let at = At::new(m, lambda, false);
    let theta = m.theta();
    let wo = [(1.0 - mu * mu).sqrt(), 0.0, mu];
    let w = at.lobe_weights(mu);
    let mut rng = Rng::new(7, (mu * 1e6) as u64);
    let mut s = 0.0;
    for _ in 0..n {
        let u = [rng.uniform(), rng.uniform(), rng.uniform()];
        match at.sample(&w, wo, u) {
            None => s += at.f(Kind::Subsurface, wo, wo, &theta) / w[4],
            Some(wi) if wi != [0.0; 3] => {
                let p = at.pdf(&w, wo, wi);
                if p > 0.0 {
                    let kind = if wi[2] > 0.0 { Kind::Reflect } else { Kind::Transmit };
                    s += at.f(kind, wo, wi, &theta) * wi[2].abs() / p;
                }
            }
            _ => {}
        }
    }
    s / n as f64
}

#[test]
fn white_furnaces() {
    let white = OpenPbr {
        base_color: [1.0; 3],
        ..Default::default()
    };
    let n = 400_000;
    let cases: Vec<(&str, OpenPbr, f64)> = vec![
        (
            "lambert",
            OpenPbr {
                specular_weight: 0.0,
                ..white
            },
            2e-3,
        ),
        (
            "glossy-diffuse r0.1",
            OpenPbr {
                specular_roughness: 0.1,
                ..white
            },
            6e-3,
        ),
        ("glossy-diffuse r0.5", OpenPbr { specular_roughness: 0.5, ..white }, 6e-3),
        ("glossy-diffuse r1.0", OpenPbr { specular_roughness: 1.0, ..white }, 6e-3),
        (
            "coat over white (δ = 0)",
            OpenPbr {
                coat_weight: 1.0,
                coat_roughness: 0.2,
                coat_darkening: 0.0,
                ..white
            },
            8e-3,
        ),
        (
            "fuzz over white",
            OpenPbr {
                fuzz_weight: 1.0,
                ..white
            },
            8e-3,
        ),
        (
            "subsurface white",
            OpenPbr {
                subsurface_weight: 1.0,
                ..white
            },
            6e-3,
        ),
    ];
    for (name, m, tol) in &cases {
        for mu in [0.15, 0.5, 0.95] {
            let e = albedo(m, mu, 550.0, n);
            println!("{name:>26} μ {mu:.2}: E = {e:.5}");
            assert!((e - 1.0).abs() < *tol, "{name} at μ {mu}: {e}");
        }
    }
}

#[test]
fn energy_per_lobe_never_exceeds_one() {
    let t = Tables::get();
    let mut rng = Rng::new(3, 0);
    let mut worst: f64 = 0.0;
    for k in 0..40 {
        let m = OpenPbr {
            base_color: [rng.uniform(), rng.uniform(), rng.uniform()],
            base_metalness: rng.uniform(),
            specular_roughness: rng.uniform(),
            specular_ior: 1.1 + 1.5 * rng.uniform(),
            coat_weight: rng.uniform(),
            coat_roughness: rng.uniform(),
            fuzz_weight: rng.uniform(),
            transmission_weight: if k % 2 == 0 { rng.uniform() } else { 0.0 },
            thin_film_weight: rng.uniform(),
            thin_film_thickness: rng.uniform(),
            ..Default::default()
        };
        let mu = 0.05 + 0.95 * rng.uniform();
        let e = albedo(&m, mu, 400.0 + 300.0 * rng.uniform(), 40_000);
        worst = worst.max(e);
        assert!(e < 1.0 + 1e-2, "material {k}: E = {e}");
    }
    // The tables themselves: every entry in [0, 1].
    let mx = |v: &[f64]| v.iter().cloned().fold(0.0, f64::max);
    println!("table maxima: diel {:.4} fuzz {:.4} metal {:.4}", mx(&t.diel), mx(&t.fuzz), mx(&t.metal[0]));
    assert!(t.diel.iter().all(|&e| (0.0..=1.0 + 1e-9).contains(&e)));
    for k in 0..=20 {
        for j in 0..=20 {
            let e = t.fuzz(k as f64 / 20.0, j as f64 / 20.0);
            assert!((0.0..=1.0 + 1e-9).contains(&e), "normalised fuzz {e}");
        }
    }
    for tab in &t.metal[..2] {
        assert!(tab.iter().all(|&e| (0.0..=1.0 + 1e-9).contains(&e)));
    }
    println!("worst random-material albedo {worst:.4}");
}

#[test]
fn metal_furnace_is_the_single_scatter_table() {
    // F = 1 (F0 = 1, no tint): E is the GGX single-scatter albedo, which
    // the spec does not compensate. Hold it against the table.
    let m = OpenPbr {
        base_color: [1.0; 3],
        base_metalness: 1.0,
        specular_roughness: 0.6,
        ..Default::default()
    };
    for mu in [0.2, 0.6, 1.0] {
        let e = albedo(&m, mu, 550.0, 200_000);
        let tab = Tables::get().metal(mu, 0.6)[0].0;
        println!("metal μ {mu}: MC {e:.5} table {tab:.5}");
        assert!((e - tab).abs() < 4e-3);
    }
}

#[test]
fn lobes_are_reciprocal() {
    let mut rng = Rng::new(11, 0);
    // Metal, dielectric specular, coat: each alone.
    let alone = [
        OpenPbr {
            base_metalness: 1.0,
            base_color: [0.9, 0.6, 0.3],
            specular_color: [0.8, 0.9, 1.0],
            ..Default::default()
        },
        OpenPbr {
            base_weight: 0.0,
            ..Default::default()
        },
        OpenPbr {
            base_weight: 0.0,
            specular_weight: 0.0,
            coat_weight: 1.0,
            coat_roughness: 0.3,
            coat_darkening: 0.0,
            ..Default::default()
        },
    ];
    for (k, m) in alone.iter().enumerate() {
        let at = At::new(m, 550.0, false);
        let th = m.theta();
        for _ in 0..200 {
            let (a, b) = (dir(&mut rng, 0.05), dir(&mut rng, 0.05));
            let (f1, f2) = (at.f(Kind::Reflect, a, b, &th), at.f(Kind::Reflect, b, a, &th));
            assert!((f1 - f2).abs() <= 1e-9 * f1.abs().max(1e-6), "lobe {k}: {f1} vs {f2}");
        }
    }
    // The rough dielectric BTDF: f(o→i)/η_i² = f(i→o)/η_o² (radiance).
    let m = OpenPbr {
        base_weight: 0.0,
        transmission_weight: 1.0,
        specular_roughness: 0.4,
        ..Default::default()
    };
    let th = m.theta();
    let (out, inn) = (At::new(&m, 550.0, false), At::new(&m, 550.0, true));
    let mut worst: f64 = 0.0;
    for _ in 0..500 {
        let a = dir(&mut rng, 0.1);
        let mut b = dir(&mut rng, 0.1);
        b[2] = -b[2];
        let f1 = out.f(Kind::Transmit, a, b, &th);
        // The same pair from inside: the frame flips (z → −z).
        let f2 = inn.f(Kind::Transmit, [b[0], b[1], -b[2]], [a[0], a[1], -a[2]], &th);
        if f1 > 1e-6 {
            let eta = m.specular_ior;
            worst = worst.max((f1 * eta * eta - f2).abs() / f2.max(1e-9));
        }
    }
    println!("BTDF reciprocity: worst rel {worst:.2e}");
    assert!(worst < 1e-9);
    // The albedo-scaled stack is not reciprocal (the spec says so): measure.
    let m = OpenPbr {
        specular_roughness: 0.3,
        ..Default::default()
    };
    let at = At::new(&m, 550.0, false);
    let th = m.theta();
    let mut asym: f64 = 0.0;
    for _ in 0..500 {
        let (a, b) = (dir(&mut rng, 0.05), dir(&mut rng, 0.05));
        let (f1, f2) = (at.f(Kind::Reflect, a, b, &th), at.f(Kind::Reflect, b, a, &th));
        asym = asym.max((f1 - f2).abs() / f1.max(f2));
    }
    println!("glossy-diffuse stack: worst |f(o,i) − f(i,o)| / f = {asym:.3}");
}

#[test]
fn the_film_is_exact_airy() {
    for c in [0.1, 0.5, 1.0] {
        for ns in [1.5, 2.5] {
            // No film, or a film of the ambient index: plain Fresnel.
            assert!((airy(c, 1.4, 0.0, ns, 550.0) - fresnel(c, ns)).abs() < 1e-12);
            assert!((airy(c, 1.0, 300.0, ns, 550.0) - fresnel(c, ns)).abs() < 1e-12);
            assert!((airy(c, ns, 300.0, ns, 550.0) - fresnel(c, ns)).abs() < 1e-12);
        }
    }
    // A quarter-wave anti-reflection film at normal incidence: n_f = √n_s,
    // d = λ / (4 n_f) — reflectance zero.
    let ns: f64 = 1.5;
    let nf = ns.sqrt();
    let r = airy(1.0, nf, 550.0 / (4.0 * nf), ns, 550.0);
    assert!(r < 1e-12, "{r}");
    // Half-wave: the film is absent.
    let r = airy(1.0, 1.38, 550.0 / (2.0 * 1.38), ns, 550.0);
    assert!((r - fresnel(1.0, ns)).abs() < 1e-12);
}

#[test]
fn dispersion_follows_the_abbe_number() {
    let m = OpenPbr {
        specular_ior: 1.5168,
        transmission_dispersion_scale: 1.0,
        transmission_dispersion_abbe_number: 64.17,
        ..Default::default()
    };
    let nf = m.ior_at(486.13);
    let nc = m.ior_at(656.27);
    let nd = m.ior_at(587.56);
    let v = (nd - 1.0) / (nf - nc);
    println!("n_d {nd:.6} V {v:.4}");
    assert!((nd - 1.5168).abs() < 1e-9 && (v - 64.17).abs() < 1e-6);
    assert_eq!(OpenPbr::default().ior_at(450.0), 1.5);
}

#[test]
fn van_de_hulst_round_trip() {
    for c in [0.1, 0.4, 0.8, 0.95] {
        let m = OpenPbr {
            subsurface_color: [c; 3],
            ..Default::default()
        };
        let (sa, ss) = subsurface_medium(&m, 550.0);
        let a = ss / (sa + ss);
        let s = ((1.0 - a) / (1.0 - 0.0 * a)).sqrt();
        let back = (1.0 - s) * (1.0 - 0.139 * s) / (1.0 + 1.17 * s);
        assert!((back - c).abs() < 2e-3, "{c} → {back}");
    }
}

fn small(m: OpenPbr) -> Swatch {
    let mut s = Swatch::new(m, (20, 20), 4);
    s.sampler = Some(m);
    s
}

#[test]
fn replay_is_bit_exact_and_deterministic() {
    let s = small(busy());
    let th = s.material.theta();
    let fwd = s.forward(&th, 5);
    let dl = vec![1.0; fwd.len()];
    let mut p = Stack::new(4096).adjoint(true);
    let (g1, l1) = s.replay(&th, 5, &dl, &mut p, 4096);
    let (g2, _) = s.replay(&th, 5, &dl, &mut p, 4096);
    let same = fwd.iter().zip(&l1).filter(|(a, b)| a.0.to_bits() == b.to_bits()).count();
    assert_eq!(same, fwd.len(), "replay re-walks every path");
    assert_eq!(g1.map(f64::to_bits), g2.map(f64::to_bits), "δθ bit-identical");
}

#[test]
fn prb_matches_central_differences() {
    // The film on a metal is detached in the base colour (its substrate
    // index is read from F0 as a number), so the gate runs without it.
    let m = OpenPbr {
        thin_film_weight: 0.0,
        ..busy()
    };
    let s = small(m);
    let th = m.theta();
    let (_, g) = s.gradient(&th, 13, &Mean, false);
    let mut worst: f64 = 0.0;
    for k in 0..PARAMS {
        let h = 1e-5;
        let loss = |d: f64| {
            let mut t = th;
            t[k] += d;
            let img = s.render(&t, 13);
            img.iter().sum::<f64>() / img.len() as f64
        };
        let cd = (loss(h) - loss(-h)) / (2.0 * h);
        let rel = (g[k] - cd).abs() / cd.abs().max(1e-4);
        println!("{:>20}: prb {:+.8e} cd {:+.8e} rel {rel:.1e}", openpbr::PARAM_NAMES[k], g[k], cd);
        worst = worst.max(rel);
    }
    assert!(worst < 1e-5, "worst {worst}");
}
