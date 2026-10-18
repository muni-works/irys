//! Speed: hardware ray queries (`gpu/rt.rs`) and lane compaction
//! (`gpu/compact.wgsl`). The hardware walk against the WGSL walk (the
//! reference), replay bit-exactness and run-to-run equality under both,
//! compaction against chunking, and PRB against central differences through
//! the hardware walk. GPU tests skip without an adapter; the hardware tests
//! skip without ray queries.

#![cfg(feature = "gpu")]

use irys_core::Transform;
use irys_diff::ggx::PARAMS;
use irys_diff::gpu::{self, GpuPrb};
use irys_diff::prb::{self, Kernels, Scene};

const THETA0: [f64; PARAMS] = [0.35, 0.95, 0.75, 0.45];
const GOLD: [f64; PARAMS] = [0.2, 0.34, 0.78, 1.0];

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.abs().max(b.abs()).max(1e-12)
}

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

/// The scenes: instanced meshes (pair: a squashed, turned ball on a
/// floor), the emitter box, analytic prims alone (marble: a sphere and a
/// disc), and triangles with an analytic sphere behind splats (garden).
fn scenes() -> Vec<(&'static str, Scene)> {
    let mut pair = Scene::pair(64, 48, 4, 1.0);
    pair.depth = 6;
    pair.complete = true;
    let mut cornell = Scene::cornell(64, 48, 4);
    cornell.qmc = true;
    cornell.spectral_is = true;
    let mut marble = Scene::new(64, 48, 4);
    marble.depth = 4;
    // triangles and an analytic sphere in one tree, behind splats
    let garden = Scene::garden(64, 48, 2);
    vec![("pair", pair), ("cornell", cornell), ("marble", marble), ("garden", garden)]
}

fn mean(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64
}

/// The hardware walk renders the software walk's image: the same mean to
/// 1e-4, and nearly every pixel bit-identical (a hardware triangle test is
/// watertight where Möller–Trumbore leaves cracks, so an edge ray can
/// differ, and its path with it).
#[test]
fn hardware_walk_matches_software_walk() {
    let ctx = device!();
    for (name, scene) in scenes() {
        let mut g = GpuPrb::new(ctx, &scene);
        if !g.hardware_rt() {
            eprintln!("no ray queries: skipping");
            return;
        }
        let hw = g.image(&scene, &GOLD, 5);
        g.set_hardware_rt(false);
        let sw = g.image(&scene, &GOLD, 5);
        let same = hw.iter().zip(&sw).filter(|(a, b)| a == b).count();
        let (mh, ms) = (mean(&hw), mean(&sw));
        eprintln!(
            "{name}: mean hw {mh:.6} sw {ms:.6}, {same} of {} values identical",
            hw.len()
        );
        assert!(rel(mh, ms) < 1e-3, "{name}: mean {mh} vs {ms}");
        let err: f64 = hw
            .iter()
            .zip(&sw)
            .map(|(&a, &b)| ((a - b) as f64).powi(2) / ((b as f64).powi(2) + 1e-2))
            .sum::<f64>()
            / hw.len() as f64;
        assert!(err < 1e-3, "{name}: relMSE hw vs sw {err}");
    }
}

/// Forward and replay take the same hardware path (bit-exact `L`), and two
/// wavefronts agree bit for bit on `L` and `δθ`, under both walks.
#[test]
fn replay_is_bit_exact_and_deterministic_under_both_walks() {
    let ctx = device!();
    for (name, scene) in scenes() {
        for hw in [true, false] {
            let run = || {
                let mut g = GpuPrb::new(ctx, &scene);
                g.set_hardware_rt(hw);
                let l = g.forward(&scene, &THETA0, 3);
                let (d, l2) = g.replay(&scene, &THETA0, 3, &vec![1.0; l.len()]);
                assert_eq!(bits(&l), bits(&l2), "{name} hw {hw}: replay");
                assert!(d.iter().all(|x| x.is_finite()), "{name}: δθ {d:?}");
                (bits(&l), d.map(f64::to_bits), g.hardware_rt())
            };
            let (a, b) = (run(), run());
            assert!(a == b, "{name} hw {hw}: two wavefronts disagree");
            eprintln!("{name}: hardware {} replay and rerun bit-exact", a.2);
        }
    }
}

/// Compaction is a function of the frame alone: the frame in chunks of 1000
/// paths (each its own scan) renders bit for bit what one chunk does.
#[test]
fn compaction_is_independent_of_chunking() {
    let ctx = device!();
    for (name, scene) in scenes() {
        let a = GpuPrb::new(ctx, &scene).image(&scene, &GOLD, 9);
        let mut c = GpuPrb::chunked(ctx, &scene, 1000);
        assert!(c.chunks() > 1);
        let b = c.image(&scene, &GOLD, 9);
        assert_eq!(bits(&a), bits(&b), "{name}");
    }
}

/// Speed round 2: frames summed on the device are the frames `image`
/// returns, summed in f32 in frame order, bit for bit; chunked and whole
/// wavefronts agree on the sum; and nothing is read back until asked.
#[test]
fn device_accumulation_is_the_frames_summed() {
    let ctx = device!();
    for (name, scene) in scenes() {
        let mut g = GpuPrb::new(ctx, &scene);
        let seeds = [11u64, 12, 13];
        let frames: Vec<Vec<f32>> = seeds.iter().map(|&s| g.image(&scene, &GOLD, s)).collect();
        let mut want = vec![0.0f32; frames[0].len()];
        for f in &frames {
            for (w, v) in want.iter_mut().zip(f) {
                *w += v;
            }
        }
        let want: Vec<f32> = want.iter().map(|v| v / seeds.len() as f32).collect();
        let mut run = |g: &mut GpuPrb| {
            g.clear_accumulated();
            for &s in &seeds {
                g.accumulate(&scene, &GOLD, s);
            }
            let (img, n) = g.accumulated();
            assert_eq!(n, seeds.len());
            bits(&img)
        };
        let a = run(&mut g);
        assert_eq!(a, bits(&want), "{name}: the device sum is not the frames'");
        assert_eq!(a, run(&mut g), "{name}: rerun");
        let mut c = GpuPrb::chunked(ctx, &scene, 1000);
        assert!(c.chunks() > 1);
        assert_eq!(a, run(&mut c), "{name}: chunked");
    }
}

/// Moving an instance rebuilds the TLAS: the hardware walk still renders
/// the software walk's image.
#[test]
fn moved_instances_rebuild_the_tlas() {
    let ctx = device!();
    let mut scene = Scene::pair(64, 48, 2, 1.0);
    let mut g = GpuPrb::new(ctx, &scene);
    if !g.hardware_rt() {
        eprintln!("no ray queries: skipping");
        return;
    }
    let before = g.image(&scene, &GOLD, 2);
    let k = 0;
    let to = Transform::translation(0.05, 0.03, 0.0).then(&scene.placements()[k].to_world);
    scene.place(&[(k, to)]);
    g.update_geometry(&scene);
    let hw = g.image(&scene, &GOLD, 2);
    g.set_hardware_rt(false);
    let sw = g.image(&scene, &GOLD, 2);
    assert!(bits(&hw) != bits(&before), "the move changed nothing");
    assert!(rel(mean(&hw), mean(&sw)) < 1e-3, "{} vs {}", mean(&hw), mean(&sw));
}

/// PRB through the hardware walk and compacted lanes against central
/// differences of the device's own loss.
#[test]
fn hardware_gradient_matches_central_differences() {
    let ctx = device!();
    let mut scene = Scene::pair(32, 32, 4, 1.0);
    scene.frozen_alpha = Some(0.3);
    let base = Scene::pair(32, 32, 4, 1.0);
    let mut k = Kernels::for_scene(&base, false);
    let t = prb::image(&base, 99, &prb::forward(&base, &GOLD, 99, &mut k).0);
    let mut g = GpuPrb::new(ctx, &scene);
    g.set_target(&t);
    let (_, d) = g.gradient_l2(&scene, &THETA0, (7, 7));
    let h = 4e-3;
    for p in 0..PARAMS {
        let (mut lo, mut hi) = (THETA0, THETA0);
        lo[p] -= h;
        hi[p] += h;
        let fd = (g.loss_l2(&scene, &hi, 7) - g.loss_l2(&scene, &lo, 7)) / (2.0 * h);
        let e = rel(d[p], fd);
        eprintln!("θ{p}: prb {:+.6e}  fd {fd:+.6e}  rel {e:.1e}  (hw {})", d[p], g.hardware_rt());
        assert!(e < 1e-2, "θ{p}: prb {} vs fd {fd}", d[p]);
    }
}

/// Speed round 3: NEE's sun share by peak radiance (`Env::pick_by_peak`)
/// renders the default share's image (unbiased: the mean over seeds agrees
/// to well under the noise of one), the CPU agrees with the device under
/// it, and PRB's gradient still matches central differences.
#[test]
fn sun_pick_by_peak_is_unbiased() {
    let ctx = device!();
    let pair = || {
        let mut s = Scene::pair(64, 48, 64, 1.0);
        s.depth = 6;
        s.complete = true;
        s
    };
    let a = pair();
    let mut b = pair();
    b.env = b.env.pick_by_peak();
    assert!(b.env.pick > 0.9 && b.env.pick < 1.0, "{}", b.env.pick);
    let mut ga = GpuPrb::new(ctx, &a);
    let mut gb = GpuPrb::new(ctx, &b);
    let (mut ma, mut mb) = (0.0, 0.0);
    for seed in 0..4 {
        ma += mean(&ga.image(&a, &GOLD, 100 + seed));
        mb += mean(&gb.image(&b, &GOLD, 200 + seed));
    }
    eprintln!("mean: pick 0.5 {:.6}, by peak {:.6}", ma / 4.0, mb / 4.0);
    assert!(rel(ma, mb) < 2e-3, "{ma} {mb}");

    let mut small = Scene::pair(32, 24, 4, 1.0);
    small.env = small.env.pick_by_peak();
    let mut k = Kernels::for_scene(&small, false);
    let cpu = prb::image(&small, 9, &prb::forward(&small, &GOLD, 9, &mut k).0);
    let gpu = GpuPrb::new(ctx, &small).image(&small, &GOLD, 9);
    let mc = cpu.iter().sum::<f64>() / cpu.len() as f64;
    eprintln!("cpu {mc:.6} gpu {:.6}", mean(&gpu));
    assert!(rel(mc, mean(&gpu)) < 1e-3);

    let mut base = Scene::pair(32, 32, 4, 1.0);
    base.env = base.env.pick_by_peak();
    let mut scene = Scene::pair(32, 32, 4, 1.0);
    scene.env = base.env;
    scene.frozen_alpha = Some(0.3);
    let mut k = Kernels::for_scene(&base, false);
    let t = prb::image(&base, 99, &prb::forward(&base, &GOLD, 99, &mut k).0);
    let mut g = GpuPrb::new(ctx, &scene);
    g.set_target(&t);
    let (_, d) = g.gradient_l2(&scene, &THETA0, (7, 7));
    let h = 4e-3;
    for p in 0..PARAMS {
        let (mut lo, mut hi) = (THETA0, THETA0);
        lo[p] -= h;
        hi[p] += h;
        let fd = (g.loss_l2(&scene, &hi, 7) - g.loss_l2(&scene, &lo, 7)) / (2.0 * h);
        let e = rel(d[p], fd);
        eprintln!("θ{p}: prb {:+.6e}  fd {fd:+.6e}  rel {e:.1e}", d[p]);
        assert!(e < 1e-2, "θ{p}: prb {} vs fd {fd}", d[p]);
    }
}
