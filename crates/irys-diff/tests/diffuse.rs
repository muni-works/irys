//! Differentiable diffuse albedo ([`Scene::diffuse_bands`]): every diffuse
//! surface's albedo as a three-hat spectrum, and PRB's derivative of the
//! image with respect to it. The same invariants as the conductor: the CPU
//! f64 gradient against central differences with common random numbers,
//! the GPU against the CPU f64 reference, and bit-exact replay. GPU tests
//! skip without an adapter.

use irys_diff::ggx::PARAMS;
use irys_diff::prb::{self, Kernels, L2, Scene};

const THETA0: [f64; PARAMS] = [0.35, 0.95, 0.75, 0.45];
const GOLD: [f64; PARAMS] = [0.2, 0.34, 0.78, 1.0];
const D0: [f64; 3] = [0.55, 0.7, 0.4];
const DGOLD: [f64; 3] = [0.8, 0.5, 0.65];

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.abs().max(b.abs()).max(1e-12)
}

/// The Cornell box: diffuse walls and boxes, a quad emitter, NEE. The
/// conductor samples at a fixed roughness, so paths do not move with `θ`.
fn frozen(w: usize, h: usize, spp: usize) -> Scene {
    let mut s = Scene::cornell(w, h, spp);
    s.frozen_alpha = Some(0.3);
    s.depth = 4;
    s
}

fn target(s: &mut Scene, k: &mut Kernels) -> Vec<f64> {
    s.diffuse_bands = Some(DGOLD);
    let t = prb::image(s, 99, &prb::forward(s, &GOLD, 99, k).0);
    s.diffuse_bands = None;
    t
}

/// Grey bands reproduce the grey albedo: the hats sum to one.
#[test]
fn grey_bands_are_the_grey_albedo() {
    let mut s = frozen(16, 16, 2);
    let mut k = Kernels::for_scene(&s, false);
    let grey = prb::forward(&s, &THETA0, 5, &mut k).0;
    s.diffuse_bands = Some([s.floor_albedo; 3]);
    let banded = prb::forward(&s, &THETA0, 5, &mut k).0;
    let worst = grey.iter().zip(&banded).map(|(a, b)| rel(*a, *b)).fold(0.0, f64::max);
    eprintln!("grey vs banded: worst rel {worst:.1e}");
    assert!(worst < 1e-12);
}

/// CPU f64 PRB against central differences in `d`, with NEE and an
/// emitter: the derivative of one frame's L2 loss at fixed random numbers.
#[test]
fn diffuse_cpu_gradient_matches_central_differences() {
    let mut scene = frozen(16, 16, 2);
    let mut k = Kernels::for_scene(&scene, false);
    let t = target(&mut scene, &mut k);
    scene.diffuse_bands = Some(D0);
    let (_, dt, dd) = prb::gradient_diffuse(&scene, &THETA0, (7, 7), &mut k, &L2(&t));
    let h = 1e-5;
    for p in 0..3 {
        let (mut lo, mut hi) = (D0, D0);
        lo[p] -= h;
        hi[p] += h;
        scene.diffuse_bands = Some(hi);
        let fhi = prb::loss(&scene, &THETA0, 7, &mut k, &L2(&t));
        scene.diffuse_bands = Some(lo);
        let flo = prb::loss(&scene, &THETA0, 7, &mut k, &L2(&t));
        let fd = (fhi - flo) / (2.0 * h);
        let e = rel(dd[p], fd);
        eprintln!("d{p}: cpu prb {:+.9e}  fd {fd:+.9e}  rel {e:.1e}", dd[p]);
        assert!(e < 1e-5, "d{p}: prb {} vs fd {fd}", dd[p]);
    }
    // the conductor's gradient is unchanged in form
    assert!(dt.iter().all(|x| x.is_finite()));
}

/// The conductor's θ gradient still matches central differences when the
/// diffuse surfaces are spectral (their albedo enters every later vertex).
#[test]
fn conductor_gradient_holds_with_diffuse_bands() {
    let mut scene = frozen(16, 16, 2);
    let mut k = Kernels::for_scene(&scene, false);
    let t = target(&mut scene, &mut k);
    scene.diffuse_bands = Some(D0);
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
