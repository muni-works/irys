//! Differentiable splats ([`Scene::splat_grad`]): each splat's DC colour and
//! base opacity, through the path tracer's front-to-back compositor. The
//! colour is linear in the image; the opacity also dims every splat behind
//! it and the light from beyond the segment, so its derivative carries
//! PRB's remaining-radiance term. Splat parameters are stored as f32, so
//! central differences here are taken at `h = 3e-3` and held to 1e-3.

use irys_diff::ggx::PARAMS;
use irys_diff::prb::{self, Kernels, L2, Scene};
use irys_core::Geometry;
use irys_light::Splats;

const THETA0: [f64; PARAMS] = [0.35, 0.95, 0.75, 0.45];

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.abs().max(b.abs()).max(1e-12)
}

fn garden() -> Scene {
    let mut s = Scene::garden(24, 24, 2);
    s.frozen_alpha = Some(0.3);
    s.depth = 3;
    s
}

fn cloud(s: &Scene) -> Splats {
    s.splats().expect("garden has splats").geometry().clone()
}

/// The target: the same bush with its splats recoloured and thinned.
fn target(s: &mut Scene, k: &mut Kernels) -> Vec<f64> {
    let base = cloud(s);
    let mut gold = base.clone();
    for i in 0..gold.len() {
        let dc = gold.dc(i);
        gold.set_dc(i, [dc[0] * 0.8 + 0.1, dc[1] * 1.1, dc[2] * 0.9 - 0.05]);
        gold.set_opacity(i, gold.opacity(i) * 0.85);
    }
    s.set_splats(Some(gold));
    let t = prb::image(s, 99, &prb::forward(s, &THETA0, 99, k).0);
    s.set_splats(Some(base));
    t
}

/// The splat walk with gradients on renders the same image as without.
#[test]
fn splat_grad_does_not_change_the_image() {
    let mut s = garden();
    let mut k = Kernels::for_scene(&s, false);
    let a = prb::forward(&s, &THETA0, 5, &mut k).0;
    s.splat_grad = true;
    let (l, _) = prb::forward(&s, &THETA0, 5, &mut k);
    let dl = vec![1.0; l.len()];
    let (_, _, l2, _) = prb::replay_extra(&s, &THETA0, 5, &mut k, &l, &dl);
    assert_eq!(a, l);
    assert_eq!(l, l2, "replay L differs");
}

/// CPU f64 PRB against central differences in the splats' colours and
/// opacities, at the splats that matter most to the loss.
#[test]
fn splat_cpu_gradient_matches_central_differences() {
    let mut scene = garden();
    let mut k = Kernels::for_scene(&scene, false);
    let t = target(&mut scene, &mut k);
    scene.splat_grad = true;
    let (_, _, ex) = prb::gradient_splats(&scene, &THETA0, (7, 7), &mut k, &L2(&t));
    let base = cloud(&scene);
    let mut order: Vec<usize> = (0..ex.splats.len()).collect();
    order.sort_by(|&a, &b| {
        let m = |i: usize| ex.splats[i].iter().map(|x| x.abs()).fold(0.0, f64::max);
        m(b).total_cmp(&m(a))
    });
    let h = 3e-3f32;
    let mut worst = 0.0f64;
    for &i in order.iter().take(4) {
        for p in 0..4 {
            let (mut lo, mut hi) = (base.clone(), base.clone());
            let (dlo, dhi) = if p < 3 {
                let (mut a, mut b) = (base.dc(i), base.dc(i));
                a[p] -= h;
                b[p] += h;
                lo.set_dc(i, a);
                hi.set_dc(i, b);
                (a[p] as f64, b[p] as f64)
            } else {
                let o = base.opacity(i);
                lo.set_opacity(i, o - h);
                hi.set_opacity(i, o + h);
                ((o - h) as f64, (o + h) as f64)
            };
            scene.set_splats(Some(hi));
            let fhi = prb::loss(&scene, &THETA0, 7, &mut k, &L2(&t));
            scene.set_splats(Some(lo));
            let flo = prb::loss(&scene, &THETA0, 7, &mut k, &L2(&t));
            let fd = (fhi - flo) / (dhi - dlo);
            let e = rel(ex.splats[i][p], fd);
            worst = worst.max(e);
            eprintln!("splat {i} p{p}: prb {:+.6e}  fd {fd:+.6e}  rel {e:.1e}", ex.splats[i][p]);
        }
    }
    scene.set_splats(Some(base));
    eprintln!("worst rel {worst:.1e}");
    assert!(worst < 1e-3, "worst rel {worst}");
}
