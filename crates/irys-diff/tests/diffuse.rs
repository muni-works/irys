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

#[cfg(feature = "gpu")]
mod gpu_tier {
    use super::*;
    use irys_diff::gpu::{self, GpuPrb};

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

    /// GPU f32 against the CPU f64 reference: the loss, θ and `d`.
    #[test]
    fn diffuse_gpu_matches_cpu_f64() {
        let ctx = device!();
        let mut scene = frozen(32, 32, 4);
        let mut k = Kernels::for_scene(&scene, false);
        let t = target(&mut scene, &mut k);
        scene.diffuse_bands = Some(D0);
        let mut g = GpuPrb::new(ctx, &scene);
        g.set_target(&t);
        let (vg, tg) = g.gradient_l2(&scene, &THETA0, (7, 7));
        let dg = g.diffuse_grad(&scene);
        let (vc, tc, dc) = prb::gradient_diffuse(&scene, &THETA0, (7, 7), &mut k, &L2(&t));
        eprintln!("loss gpu {vg:.6e} cpu {vc:.6e} rel {:.1e}", rel(vg, vc));
        assert!(rel(vg, vc) < 1e-2);
        for p in 0..3 {
            let e = rel(dg[p], dc[p]);
            eprintln!("d{p}: gpu {:+.6e}  cpu f64 {:+.6e}  rel {e:.1e}", dg[p], dc[p]);
            assert!(e < 2e-2, "d{p}: gpu {} vs cpu {}", dg[p], dc[p]);
        }
        for p in 0..PARAMS {
            let e = rel(tg[p], tc[p]);
            eprintln!("θ{p}: gpu {:+.6e}  cpu f64 {:+.6e}  rel {e:.1e}", tg[p], tc[p]);
            assert!(e < 2e-2, "θ{p}: gpu {} vs cpu {}", tg[p], tc[p]);
        }
    }

    /// The replay reproduces the forward radiance bit for bit with spectral
    /// diffuse surfaces, and two replays give bitwise-identical `δd`.
    #[test]
    fn diffuse_replay_is_bit_exact() {
        let ctx = device!();
        let mut scene = Scene::cornell(64, 64, 4);
        scene.depth = 6;
        scene.diffuse_bands = Some(D0);
        let mut g = GpuPrb::new(ctx, &scene);
        let l = g.forward(&scene, &THETA0, 11);
        let dl = vec![1.0; l.len()];
        let (t1, l2) = g.replay(&scene, &THETA0, 11, &dl);
        let d1 = g.diffuse_grad(&scene);
        assert_eq!(bits(&l), bits(&l2), "replay L differs from forward");
        let (t2, _) = g.replay(&scene, &THETA0, 11, &dl);
        let d2 = g.diffuse_grad(&scene);
        assert_eq!(t1.map(f64::to_bits), t2.map(f64::to_bits));
        assert_eq!(d1.map(f64::to_bits), d2.map(f64::to_bits));
        eprintln!("{} of {} lanes bit-identical; δd {d1:?}", l.len(), l.len());
        assert!(d1.iter().all(|x| x.is_finite() && *x != 0.0), "δd {d1:?}");
    }
}
