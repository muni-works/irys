//! v0's tests. Rendering ones are slow in debug: `cargo test --release -p irys-recon`.

use irys_recon::dsl::{Axis, Corner, Feature, Program};
use irys_recon::fit::{self, Target};
use irys_recon::model::{self, Renderer, Setup};
use irys_recon::{canon, generate, geom, render};

fn sample() -> Program {
    Program {
        w: 70.0,
        d: 50.0,
        h: 12.0,
        corner: Corner::Fillet(6.0),
        features: vec![
            Feature::Hole { x: -20.0, y: 10.0, r: 4.0 },
            Feature::Counterbore { x: 15.0, y: 8.0, r: 3.0, cr: 6.0, cd: 4.0 },
            Feature::Slot { x: -15.0, y: -12.0, len: 10.0, r: 2.5, axis: Axis::X },
            Feature::Boss { x: 14.0, y: -10.0, r: 5.0, bh: 6.0 },
        ],
        pose: None,
    }
}

#[test]
fn dsl_round_trips() {
    let mut progs = vec![sample()];
    progs.extend((0..30).map(generate::program));
    for p in progs {
        let q = Program::from_text(&p.to_text()).expect("parses");
        assert_eq!(p.signature(), q.signature());
        for (a, b) in p.params().iter().zip(q.params()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
        // Parameters round-trip through the flat vector too.
        let mut r = p.clone();
        r.set_params(&p.params());
        assert_eq!(r, p);
        // The normal form is a fixed point.
        let c = canon::canonical(&p);
        assert_eq!(canon::canonical(&c).to_text(), c.to_text());
    }
}

#[test]
fn canonical_merges_equivalent_holes() {
    // Three loose holes on a circle at 120° are a circular pattern, and two
    // equal holes a linear one; listing order does not matter.
    let mut p = Program { w: 80.0, d: 60.0, h: 10.0, corner: Corner::Sharp, features: Vec::new(), pose: None };
    for k in [2, 0, 1] {
        let a = (30.0f64 + 120.0 * k as f64).to_radians();
        p.features.push(Feature::Hole { x: 10.0 + 12.0 * a.cos(), y: 5.0 + 12.0 * a.sin(), r: 2.0 });
    }
    p.features.push(Feature::Hole { x: -30.0, y: -20.0, r: 3.5 });
    p.features.push(Feature::Hole { x: -18.0, y: -20.0, r: 3.5 });
    let c = canon::canonical(&p);
    assert_eq!(c.signature(), "sharp[linear_holes×2,circular_holes×3]");
    let pattern = Program {
        features: vec![
            Feature::CircularHoles { x: 10.0, y: 5.0, rad: 12.0, phase: 150.0, r: 2.0, n: 3 },
            Feature::LinearHoles { x: -18.0, y: -20.0, dx: -12.0, dy: 0.0, r: 3.5, n: 2 },
        ],
        ..p.clone()
    };
    assert!(canon::same_topology(&p, &pattern));
    // A vertical run fitted with a hair of dx either way is one pattern.
    let up = Feature::LinearHoles { x: 0.0, y: -10.0, dx: 0.013, dy: 10.0, r: 2.0, n: 3 };
    let down = Feature::LinearHoles { x: 0.026, y: 10.0, dx: -0.013, dy: -10.0, r: 2.0, n: 3 };
    let a = Program { features: vec![up], ..p.clone() };
    let b = Program { features: vec![down], ..p.clone() };
    assert!(canon::errors(&a, &b).unwrap().iter().all(|(_, e)| *e < 0.05));
    let e = canon::errors(&pattern, &p).unwrap();
    assert!(e.iter().all(|(_, v)| *v < 1e-6), "{e:?}");
}

#[test]
fn generator_gives_valid_breps() {
    for seed in 0..40 {
        let p = generate::program(seed);
        assert!(p.valid(), "seed {seed}");
        let s = p.build().expect("builds");
        assert!(s.as_brep().is_some(), "seed {seed}: not a BRep");
        let (pos, tri) = geom::mesh(&s);
        assert!(!tri.is_empty());
        // The tessellation's volume matches the program's closed form, and
        // every vertex but the bottom face's lies on a carrier (so has a
        // velocity).
        let v = s.volume();
        assert!((v - p.volume()).abs() < 2e-3 * p.volume(), "seed {seed}: {v} vs {}", p.volume());
        assert_eq!(geom::orphans(&p, &pos), 0, "seed {seed}");
        // Deterministic.
        assert_eq!(generate::program(seed), p);
    }
}

/// The edge-term Jacobian against central differences of the rendered
/// mask (exact: coverage has no path noise) and of the image L2 loss
/// (vcad re-tessellates at every step).
#[test]
fn inner_fit_gradient_matches_central_differences() {
    let p = sample();
    let mut setup = Setup::new(render::views());
    setup.film = render::Film { size: (128, 96), spp: 32 };
    setup.coverage_sub = 8;
    let mut r = Renderer::new(setup.views.len());
    let e = r.eval(&p, &setup, true).unwrap();
    let np = p.params().len();
    let info = p.param_info();
    // A target with every parameter 0.5 mm off, for a realistic residual.
    let mut q = p.clone();
    q.set_params(&p.params().iter().map(|x| x + 0.5).collect::<Vec<_>>());
    let tgt = model::target(&q, &setup, 64, 5).unwrap();
    let loss = |e: &model::Eval| -> f64 {
        e.img.iter().zip(&tgt).map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>()).sum()
    };
    let (w, _) = setup.film.size;
    let weight = |px: usize| ((px % w) as f64 / 9.0).sin() + ((px / w) as f64 / 7.0).cos();
    let jac = e.jac.as_ref().unwrap();
    let mut worst_mask: f64 = 0.0;
    let mut img_rel = Vec::new();
    for j in 0..np {
        let h = 0.15;
        let mut th = p.params();
        th[j] += h;
        let mut qp = p.clone();
        qp.set_params(&th);
        let ep = r.eval(&qp, &setup, false).unwrap();
        th[j] -= 2.0 * h;
        let mut qm = p.clone();
        qm.set_params(&th);
        let em = r.eval(&qm, &setup, false).unwrap();
        let mut gm = 0.0;
        let mut cm = 0.0;
        let mut gi = 0.0;
        for v in 0..3 {
            for px in 0..e.img[v].len() {
                gm += weight(px) * jac[v].mask[px * np + j];
                cm += weight(px) * (ep.cov[v][px] - em.cov[v][px]) / (2.0 * h);
                gi += 2.0 * (e.img[v][px] - tgt[v][px]) * jac[v].img[px * np + j];
            }
        }
        let ci = (loss(&ep) - loss(&em)) / (2.0 * h);
        let scale_m = cm.abs().max(5.0);
        worst_mask = worst_mask.max((gm - cm).abs() / scale_m);
        eprintln!("{:12} mask {gm:9.3} cd {cm:9.3} | L2 {gi:9.4} cd {ci:9.4}", info[j].name);
        img_rel.push(((gi - ci).abs(), ci.abs()));
    }
    assert!(worst_mask < 0.1, "mask Jacobian off by {worst_mask}");
    // The image term: the base's large columns within 25 % (the interior
    // term and soft shadows are not in the edge estimator).
    for (j, (d, c)) in img_rel.iter().enumerate().take(4) {
        assert!(d / c < 0.25, "{}: {d} of {c}", info[j].name);
    }
}

/// One feature, unknown parameters: LM from 1.5 mm off recovers every
/// dimension within 0.2 mm.
#[test]
fn recovers_a_single_feature_part() {
    let truth = Program::from_text("base 64 46 10\nhole -12 6 4").unwrap();
    let setup = Setup::new(render::views());
    let t = model::target(&truth, &setup, 64, 21).unwrap();
    let bg = model::background(&setup, 16);
    let tg = Target::new(t, &bg, setup.film.size);
    let mut start = truth.clone();
    start.set_params(&truth.params().iter().enumerate().map(|(i, x)| x + if i % 2 == 0 { 1.5 } else { -1.5 }).collect::<Vec<_>>());
    let mut f = fit::Fitter::new(setup, tg, 64, &start);
    let r = f.fit(&start, &fit::schedule()).unwrap();
    let e = canon::errors(&truth, &r.prog).unwrap();
    for (n, v) in &e {
        assert!(*v < 0.2, "{n}: {v}");
    }
}
