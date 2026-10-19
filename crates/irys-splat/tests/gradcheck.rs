//! Gradient checks, per parameter type.
//!
//! 1. host adjoint (project + raster_cpu, f64) vs central differences of the
//!    same f64 forward, for means, log scales, quaternions, opacity logits,
//!    SH DC and SH rest (degree 3), on a smooth linear image functional;
//! 2. the loss (L1 + D-SSIM, depth L1) vs central differences;
//! 3. the device (raster.wgsl, f32) vs the host f64 twin: forward image and
//!    per-splat screen-space gradient, and the device run twice bitwise.
//!
//! `cargo test --release -p irys-splat --test gradcheck -- --nocapture`
//! prints the tables design.md quotes.

use irys_diff::look::pinhole::Pinhole;
use irys_splat::model::{Gaussians, logit};
use irys_splat::project::{self, Grad2, Grad3, Proj};
use irys_splat::raster_cpu;
use irys_splat::rng::Rng;
use irys_splat::sh::COEFS;

fn camera(w: usize, h: usize) -> Pinhole {
    // a tilted camera at (0.1, -0.2, 0.05) looking along +y-ish
    let (a, b) = (0.3f64, -0.2f64);
    let rz = [[a.cos(), -a.sin(), 0.0], [a.sin(), a.cos(), 0.0], [0.0, 0.0, 1.0]];
    let rx = [[1.0, 0.0, 0.0], [0.0, b.cos(), -b.sin()], [0.0, b.sin(), b.cos()]];
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = (0..3).map(|k| rz[i][k] * rx[k][j]).sum();
        }
    }
    Pinhole { width: w, height: h, fx: 0.75 * w as f64, fy: 0.75 * w as f64, cx: 0.48 * w as f64, cy: 0.53 * h as f64, rot: r, pos: [0.1, -0.2, 0.05] }
}

fn scene(n: usize, cam: &Pinhole, seed: u64) -> Gaussians {
    let mut rng = Rng::new(seed);
    let mut g = Gaussians::zeros(n);
    for i in 0..n {
        // in camera space, then to the world
        let z = rng.range(1.0, 3.0);
        let x = rng.range(-0.5, 0.5) * z;
        let y = rng.range(-0.4, 0.4) * z;
        let pc = [x, y, z];
        g.means[i] = std::array::from_fn(|a| cam.pos[a] + (0..3).map(|k| cam.rot[a][k] * pc[k]).sum::<f64>());
        g.log_scales[i] = std::array::from_fn(|_| rng.range(0.03, 0.15).ln());
        g.quats[i] = std::array::from_fn(|_| rng.normal());
        g.opacity[i] = logit(rng.range(0.2, 0.85));
        for k in 0..COEFS {
            let amp = if k == 0 { 1.0 } else { 0.2 };
            g.sh[i][k] = std::array::from_fn(|_| amp * rng.range(-1.0, 1.0));
        }
    }
    g
}

/// A smooth functional of the frame: Σ wt · (rgb, depth).
fn functional(g: &Gaussians, cam: &Pinhole, wt: &[[f64; 4]]) -> f64 {
    let p: Vec<Option<Proj>> = (0..g.len()).map(|i| project::project(g, i, cam, 3)).collect();
    let ord = raster_cpu::order(&p);
    let f = raster_cpu::forward(&p, &ord, cam.width, cam.height, [0.1, 0.2, 0.3]);
    f.colour.iter().zip(wt).map(|(c, w)| (0..4).map(|k| c[k] * w[k]).sum::<f64>()).sum()
}

fn analytic(g: &Gaussians, cam: &Pinhole, wt: &[[f64; 4]]) -> Vec<Grad3> {
    let p: Vec<Option<Proj>> = (0..g.len()).map(|i| project::project(g, i, cam, 3)).collect();
    let ord = raster_cpu::order(&p);
    let f = raster_cpu::forward(&p, &ord, cam.width, cam.height, [0.1, 0.2, 0.3]);
    let d2 = raster_cpu::backward(&p, &ord, &f, wt, cam.width, [0.1, 0.2, 0.3]);
    (0..g.len()).map(|i| project::backward(g, i, cam, 3, &d2[i])).collect()
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Mean(usize),
    LogScale(usize),
    Quat(usize),
    Opacity,
    ShDc(usize),
    ShRest(usize, usize),
}

fn param(g: &mut Gaussians, i: usize, k: Kind) -> &mut f64 {
    match k {
        Kind::Mean(a) => &mut g.means[i][a],
        Kind::LogScale(a) => &mut g.log_scales[i][a],
        Kind::Quat(a) => &mut g.quats[i][a],
        Kind::Opacity => &mut g.opacity[i],
        Kind::ShDc(c) => &mut g.sh[i][0][c],
        Kind::ShRest(k, c) => &mut g.sh[i][k][c],
    }
}

fn grad_of(gr: &Grad3, k: Kind) -> f64 {
    match k {
        Kind::Mean(a) => gr.mean[a],
        Kind::LogScale(a) => gr.log_scale[a],
        Kind::Quat(a) => gr.quat[a],
        Kind::Opacity => gr.opacity,
        Kind::ShDc(c) => gr.sh[0][c],
        Kind::ShRest(k, c) => gr.sh[k][c],
    }
}

#[test]
fn host_adjoint_vs_central_differences() {
    let (w, h) = (48, 36);
    let cam = camera(w, h);
    let g = scene(24, &cam, 7);
    let mut rng = Rng::new(3);
    let wt: Vec<[f64; 4]> = (0..w * h).map(|_| [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), 0.1 * rng.range(-1.0, 1.0)]).collect();
    let an = analytic(&g, &cam, &wt);
    let kinds: Vec<(&str, Vec<Kind>)> = vec![
        ("mean", (0..3).map(Kind::Mean).collect()),
        ("log scale", (0..3).map(Kind::LogScale).collect()),
        ("quaternion", (0..4).map(Kind::Quat).collect()),
        ("opacity logit", vec![Kind::Opacity]),
        ("SH DC", (0..3).map(Kind::ShDc).collect()),
        ("SH rest (deg 1–3)", (1..COEFS).flat_map(|k| (0..3).map(move |c| Kind::ShRest(k, c))).collect()),
    ];
    let h_step = 1e-6;
    println!("| parameter | entries | max rel err | median rel err |");
    for (name, ks) in kinds {
        let mut errs = Vec::new();
        for i in 0..g.len() {
            for &k in &ks {
                let mut gp = g.clone();
                *param(&mut gp, i, k) += h_step;
                let mut gm = g.clone();
                *param(&mut gm, i, k) -= h_step;
                let cd = (functional(&gp, &cam, &wt) - functional(&gm, &cam, &wt)) / (2.0 * h_step);
                let a = grad_of(&an[i], k);
                let scale = cd.abs().max(a.abs());
                if scale < 1e-6 {
                    continue;
                }
                errs.push((a - cd).abs() / scale);
            }
        }
        errs.sort_by(f64::total_cmp);
        let max = *errs.last().unwrap();
        let med = errs[errs.len() / 2];
        println!("| {name} | {} | {max:.1e} | {med:.1e} |", errs.len());
        // a contour (the 1/255 cut) crossing inside ±h is the only way to
        // be far off; allow it in at most 1 % of entries
        let bad = errs.iter().filter(|&&e| e > 1e-4).count();
        assert!(bad * 100 <= errs.len(), "{name}: {bad} of {} entries > 1e-4 (max {max:.2e})", errs.len());
        assert!(med < 1e-6, "{name}: median {med:.2e}");
    }
}

#[test]
fn loss_vs_central_differences() {
    use irys_splat::loss::{Weights, eval};
    let (w, h) = (23, 17);
    let mut rng = Rng::new(11);
    let pred: Vec<[f64; 4]> = (0..w * h).map(|_| [rng.uniform(), rng.uniform(), rng.uniform(), rng.range(1.0, 3.0)]).collect();
    let gt: Vec<[f32; 3]> = (0..w * h).map(|_| [rng.uniform() as f32, rng.uniform() as f32, rng.uniform() as f32]).collect();
    let gd: Vec<f32> = (0..w * h).map(|i| if i % 7 == 0 { 0.0 } else { rng.range(1.0, 3.0) as f32 }).collect();
    let wt = Weights { ssim: std::env::var("S").map_or(0.2, |s| s.parse().unwrap()), depth: std::env::var("D").map_or(0.5, |s| s.parse().unwrap()) };
    let e = eval(&pred, &gt, Some(&gd), w, h, wt);
    let gmax = e.grad.iter().flat_map(|g| g.iter()).fold(0.0f64, |m, v| m.max(v.abs()));
    let mut worst: f64 = 0.0;
    for i in (0..w * h).step_by(5) {
        for ch in 0..4 {
            let hs = 1e-7;
            let mut p = pred.clone();
            p[i][ch] += hs;
            let lp = eval(&p, &gt, Some(&gd), w, h, wt).loss;
            p[i][ch] -= 2.0 * hs;
            let lm = eval(&p, &gt, Some(&gd), w, h, wt).loss;
            let cd = (lp - lm) / (2.0 * hs);
            let a = e.grad[i][ch];
            worst = worst.max((a - cd).abs() / gmax);
        }
    }
    println!("loss (L1 + D-SSIM + depth L1) vs CD: max err / max |grad| {worst:.1e}");
    assert!(worst < 1e-5, "{worst}");
}

#[test]
fn device_vs_host() {
    let ctx = irys_gpu::GpuContext::init_blocking().expect("gpu");
    let (w, h) = (100, 70); // partial edge tiles
    let cam = camera(w, h);
    let g = scene(300, &cam, 5);
    let p: Vec<Option<Proj>> = (0..g.len()).map(|i| project::project(&g, i, &cam, 3)).collect();
    // the device sees f32: compare against the host fed the same f32 values
    let pf: Vec<Option<Proj>> = p
        .iter()
        .map(|q| {
            q.map(|q| {
                let r = |x: f64| x as f32 as f64;
                Proj { uv: q.uv.map(r), conic: q.conic.map(r), opacity: r(q.opacity), colour: q.colour.map(r), radius: q.radius }
            })
        })
        .collect();
    let bg = [0.1, 0.2, 0.3];
    let ord = raster_cpu::order(&pf);
    let fh = raster_cpu::forward(&pf, &ord, w, h, bg);
    let bins = irys_splat::tiles::bin(&pf, w, h);
    let mut ras = irys_splat::gpu::Raster::new(ctx);
    let packed = irys_splat::gpu::pack(&pf);
    let (c, t) = ras.forward(&packed, &bins, (w, h), [0.1, 0.2, 0.3]);
    let mut maxd: f64 = 0.0;
    for i in 0..w * h {
        for k in 0..3 {
            maxd = maxd.max((c[i][k] as f64 - fh.colour[i][k]).abs());
        }
        maxd = maxd.max((f32::from_bits(t[i][0]) as f64 - fh.trans[i]).abs());
    }
    let mut rng = Rng::new(9);
    let dl: Vec<[f64; 4]> = (0..w * h).map(|_| [rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), 0.1 * rng.range(-1.0, 1.0)]).collect();
    let dlf: Vec<[f32; 4]> = dl.iter().map(|v| v.map(|x| x as f32)).collect();
    let dl32: Vec<[f64; 4]> = dlf.iter().map(|v| v.map(|x| x as f64)).collect();
    let gh = raster_cpu::backward(&pf, &ord, &fh, &dl32, w, bg);
    let gd = ras.backward(&dlf);
    let flat = |g: &Grad2| [g.uv[0], g.uv[1], g.conic[0], g.conic[1], g.conic[2], g.opacity, g.colour[0], g.colour[1], g.colour[2], g.colour[3]];
    let names = ["u", "v", "A", "B", "C", "opacity", "r", "g", "b", "depth"];
    println!("| screen-space gradient | max rel err (vs per-type max) | median rel err |");
    for k in 0..10 {
        let scale = gh.iter().map(|g| flat(g)[k].abs()).fold(0.0, f64::max).max(1e-12);
        let mut e: Vec<f64> = (0..g.len()).map(|i| (flat(&gd[i])[k] - flat(&gh[i])[k]).abs() / scale).collect();
        e.sort_by(f64::total_cmp);
        println!("| {} | {:.1e} | {:.1e} |", names[k], e.last().unwrap(), e[e.len() / 2]);
        assert!(*e.last().unwrap() < 2e-3, "{}: {:?}", names[k], e.last());
    }
    println!("forward max |Δ| (rgb, T) device vs host: {maxd:.1e}");
    assert!(maxd < 1e-4, "{maxd}");
    // determinism
    let (c2, t2) = ras.forward(&packed, &bins, (w, h), [0.1, 0.2, 0.3]);
    let gd2 = ras.backward(&dlf);
    assert!(c == c2 && t == t2, "forward not bitwise repeatable");
    assert!(gd == gd2, "backward not bitwise repeatable");
    println!("device forward + adjoint run twice: bit-identical");
}

#[test]
fn device_projection_adjoint_and_loss() {
    use irys_splat::gpu::{Device, LossCfg, Lr};
    use irys_splat::loss::{Weights, eval};
    let ctx = irys_gpu::GpuContext::init_blocking().expect("gpu");
    let (w, h) = (100, 70);
    let cam = camera(w, h);
    let g = scene(300, &cam, 5);
    let mut dev = Device::new(ctx);
    dev.set_state(&g, None, None);
    let keys = dev.project(&cam, 3);
    // the device's projection vs the host's
    let hp: Vec<Option<Proj>> = (0..g.len()).map(|i| project::project(&g, i, &cam, 3)).collect();
    let mut worst: f64 = 0.0;
    for i in 0..g.len() {
        match hp[i] {
            Some(p) => {
                let k = keys[i];
                worst = worst.max((k[0] as f64 - p.uv[0]).abs()).max((k[1] as f64 - p.uv[1]).abs());
                worst = worst.max((k[3] as f64 - p.colour[3]).abs());
            }
            None => assert_eq!(keys[i][2], 0.0, "splat {i} culled on the host only"),
        }
    }
    println!("device projection: max |Δ| (uv px, z m) {worst:.1e}");
    assert!(worst < 1e-3);
    let bins = irys_splat::tiles::bin_keys(&keys, w, h);
    let mut rng = Rng::new(21);
    let gt: Vec<[f32; 4]> = (0..w * h).map(|i| [rng.uniform() as f32, rng.uniform() as f32, rng.uniform() as f32, if i % 5 == 0 { 0.0 } else { rng.range(1.0, 3.0) as f32 }]).collect();
    let valid = gt.iter().filter(|v| v[3] > 0.0).count();
    let lr = Lr { bc1: 0.1, bc2: 0.001, ..Lr::default() }; // lr 0: parameters stay
    let img = dev.step(&bins, &gt, [0.1, 0.2, 0.3], LossCfg { ssim: 0.2, depth_per_px: 0.5 / valid as f64 }, lr, true).unwrap();
    // loss gradient: device vs host on the device's own frame
    let pred: Vec<[f64; 4]> = img.iter().map(|v| v.map(|x| x as f64)).collect();
    let rgb: Vec<[f32; 3]> = gt.iter().map(|v| [v[0], v[1], v[2]]).collect();
    let d: Vec<f32> = gt.iter().map(|v| v[3]).collect();
    let ev = eval(&pred, &rgb, Some(&d), w, h, Weights { ssim: 0.2, depth: 0.5 });
    let dl = dev.last_dl();
    let gmax = ev.grad.iter().flat_map(|v| v.iter()).fold(0.0f64, |m, v| m.max(v.abs()));
    let lw = (0..w * h).flat_map(|i| (0..4).map(move |k| (i, k))).map(|(i, k)| (dl[i][k] as f64 - ev.grad[i][k]).abs()).fold(0.0, f64::max) / gmax;
    println!("device loss gradient vs host: max |Δ| / max |grad| {lw:.1e}");
    assert!(lw < 1e-3, "{lw}");
    // projection adjoint: device vs host fed the device's screen-space gradient
    let g2 = dev.screen_grads();
    let gd = dev.last_grads();
    let gh: Vec<Grad3> = (0..g.len()).map(|i| project::backward(&g, i, &cam, 3, &g2[i])).collect();
    let flat = |q: &Grad3| -> Vec<f64> {
        let mut v = q.mean.to_vec();
        v.extend(q.log_scale);
        v.extend(q.quat);
        v.push(q.opacity);
        v.extend(q.sh.iter().flatten());
        v
    };
    let groups = [("mean", 0..3), ("log scale", 3..6), ("quaternion", 6..10), ("opacity logit", 10..11), ("SH DC", 11..14), ("SH rest", 14..59)];
    println!("| device adjoint vs host adjoint | max |Δ| / per-type max | median |");
    for (name, r) in groups {
        let scale = gh.iter().flat_map(|q| flat(q)[r.clone()].to_vec()).fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
        let mut e: Vec<f64> = (0..g.len()).flat_map(|i| {
            let (a, b) = (flat(&gd[i]), flat(&gh[i]));
            r.clone().map(move |k| (a[k] - b[k]).abs() / scale).collect::<Vec<_>>()
        }).collect();
        e.sort_by(f64::total_cmp);
        println!("| {name} | {:.1e} | {:.1e} |", e.last().unwrap(), e[e.len() / 2]);
        assert!(*e.last().unwrap() < 1e-3, "{name}");
    }
}
