//! Recover rose full's left arm pose (shoulder pitch, shoulder roll, elbow)
//! *and* its finish (GGX `α`, F0 at three bands) from a target render, by
//! path replay backprop for `θ` on the GPU wavefront and, for `φ`, phase
//! 2e's gradient: the edge term drawn on the device ([`gpu::GpuEdges`],
//! front-only silhouettes) plus the interior term ([`interior`], CPU f64
//! duals).
//!
//! `cargo run --release -p irys-diff --features gpu --example rose_fit -- out/rose_fit [flags] [target.ppm]`
//!
//! - default: phase 2e "in context": the whole robot from rose-full.loon on
//!   the checker floor, from 2d's fit camera ([`CONTEXT_CAM`]), where the
//!   forearm now lies across the torso. `--cam 1200,1500,1900,150,230,1250,10`
//!   looks down so the checker floor is behind the forearm too.
//! - `--control`: phase 2d's scene, the arm alone over a plain floor with
//!   the forearm against the sky ([`arm::FIT_CAM`]).
//! - `--edge-only`: 2d's pose gradient (the edge term alone).
//! - `--vis`: edge samples drawn on the host with the ID-buffer importance
//!   ([`vis::Visibility`]) and traced on the device, instead of
//!   [`gpu::GpuEdges`] (which has no ID buffer).
//! - `--interior-spp N` (default 1): paths per pixel of the interior
//!   estimate (CPU), independent of the 8 the image uses.
//! - `--iters N`, `--pose-only N`, `--cam ex,ey,ez,tx,ty,tz,fov`,
//!   `--target-only` (write `target.ppm` and `start.ppm`, then stop).
//! - `--blur R`: coarse to fine, the pose-only phase's pose loss on R-px
//!   box-blurred images.
//! - diagnostics: `--check` (the pose gradient at the start, or `--at
//!   p,r,e` degrees, against central differences of the 256-spp MSE; `SEEDS`
//!   and `DUMP` in the environment), `SCAN=1` (the MSE along the elbow),
//!   `--depth N`, `--no-nee`, `--no-sun`, `--sun x,y,z`.
//!
//! Writes `target.ppm`, `iter_NNN.ppm` previews, `loss.csv` (every
//! iteration's 8-spp loss, `φ`, `θ`, both pose terms, milliseconds per
//! part) and `filmstrip.csv`. A positional target is a binary PPM at the
//! fit's size (display sRGB, exposure 0.6 assumed) used instead of an irys
//! render.

#[path = "rose/arm.rs"]
mod arm;

use std::{
    fs,
    io::Write,
    path::Path,
    time::{Duration, Instant},
};

use irys_diff::gpu::{self, GpuEdges, GpuPrb};
use irys_diff::interior::{self, Tangents};
use irys_diff::prb::Scene;
use irys_diff::vis::{self, Edges};

const SIZE: (usize, usize) = (192, 192);
const EXPOSURE: f32 = 0.6;

/// The context fit's camera: 2d's ([`arm::FIT_CAM`]), so the only change
/// from the control is the body (and checker) behind the forearm.
const CONTEXT_CAM: ([f64; 3], [f64; 3], f64) =
    arm::FIT_CAM;

fn accumulate(g: &mut GpuPrb, s: &Scene, theta: &[f64; 4], seed: u64, frames: u64) -> Vec<f32> {
    let mut acc = vec![0.0f64; s.size.0 * s.size.1 * 3];
    for f in 0..frames {
        for (a, x) in acc.iter_mut().zip(g.image(s, theta, seed + f)) {
            *a += x as f64 / frames as f64;
        }
    }
    acc.iter().map(|&x| x as f32).collect()
}

/// A binary PPM in display sRGB back to linear, undoing `ppm`'s exposure.
fn read_ppm(path: &Path) -> ((usize, usize), Vec<f32>) {
    let b = fs::read(path).expect("target ppm");
    let mut fields = Vec::new();
    let mut i = 0;
    while fields.len() < 4 {
        while b[i].is_ascii_whitespace() {
            i += 1;
        }
        if b[i] == b'#' {
            while b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        let s = i;
        while !b[i].is_ascii_whitespace() {
            i += 1;
        }
        fields.push(String::from_utf8_lossy(&b[s..i]).to_string());
    }
    i += 1;
    let (w, h): (usize, usize) = (fields[1].parse().unwrap(), fields[2].parse().unwrap());
    let lin = |c: u8| {
        let s = c as f32 / 255.0;
        let l = if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        };
        l / EXPOSURE
    };
    (
        (w, h),
        b[i..i + w * h * 3].iter().map(|&c| lin(c)).collect(),
    )
}

/// A separable box blur of radius `r` px (clamped at the border), per
/// channel: the low-pass image a coarse-to-fine pose loss compares. It is
/// symmetric, so it is its own adjoint.
fn blur(img: &[f32], r: usize) -> Vec<f64> {
    let (w, h) = SIZE;
    let mut a: Vec<f64> = img.iter().map(|&x| x as f64).collect();
    let mut b = vec![0.0; a.len()];
    for pass in 0..2 {
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let mut s = 0.0;
                    for k in -(r as isize)..=(r as isize) {
                        let (xx, yy) = if pass == 0 {
                            ((x as isize + k).clamp(0, w as isize - 1) as usize, y)
                        } else {
                            (x, (y as isize + k).clamp(0, h as isize - 1) as usize)
                        };
                        s += a[(yy * w + xx) * 3 + c];
                    }
                    b[(y * w + x) * 3 + c] = s / (2 * r + 1) as f64;
                }
            }
        }
        std::mem::swap(&mut a, &mut b);
    }
    a
}

fn npx_of() -> f64 {
    (SIZE.0 * SIZE.1 * 3) as f64
}

fn main() {
    let (mut out, mut photo, mut cam) = (None, None, None);
    let (mut control, mut edge_only, mut use_vis, mut target_only) = (false, false, false, false);
    let (mut interior_spp, mut iters, mut pose_only) = (1usize, 300usize, 100usize);
    let (mut check, mut depth, mut no_nee, mut no_sun, mut at) = (false, None, false, false, None);
    let mut blur_r = 0usize;
    let mut sun: Option<[f64; 3]> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().expect("flag value");
        match a.as_str() {
            "--control" => control = true,
            "--edge-only" => edge_only = true,
            "--vis" => use_vis = true,
            "--target-only" => target_only = true,
            "--check" => check = true,
            "--blur" => blur_r = val().parse().unwrap(),
            "--sun" => {
                let v: Vec<f64> = val().split(',').map(|x| x.parse().unwrap()).collect();
                sun = Some([v[0], v[1], v[2]]);
            }
            "--depth" => depth = Some(val().parse::<usize>().unwrap()),
            "--no-nee" => no_nee = true,
            "--no-sun" => no_sun = true,
            "--at" => {
                let v: Vec<f64> = val().split(',').map(|x| x.parse::<f64>().unwrap().to_radians()).collect();
                at = Some([v[0], v[1], v[2]]);
            }
            "--interior-spp" => interior_spp = val().parse().unwrap(),
            "--iters" => iters = val().parse().unwrap(),
            "--pose-only" => pose_only = val().parse().unwrap(),
            "--cam" => {
                let v: Vec<f64> = val().split(',').map(|x| x.parse().unwrap()).collect();
                cam = Some(([v[0], v[1], v[2]], [v[3], v[4], v[5]], v[6]));
            }
            _ if out.is_none() => out = Some(a),
            _ => photo = Some(a),
        }
    }
    let out = out.unwrap_or_else(|| "out/rose_fit".into());
    let out = Path::new(&out);
    fs::create_dir_all(out).unwrap();
    let ctx = gpu::context().expect("rose_fit needs a GPU adapter");
    let arm = arm::load_arm(&arm::rose_dir());
    let rig = arm.rig();
    let cam = cam.unwrap_or(if control { arm::FIT_CAM } else { CONTEXT_CAM });
    let make = |spp: usize, phi: &[f64]| {
        let mut s = if control {
            arm::arm_scene_from(&arm, SIZE, spp, phi, cam)
        } else {
            arm::scene_from(&arm, SIZE, spp, phi, cam)
        };
        if let Some(d) = depth {
            s.depth = d;
        }
        if no_nee {
            s.nee = false;
        }
        if let Some(d) = sun {
            s.env.sun = irys_core::Vec3::new(d[0], d[1], d[2]).normalize();
        }
        if no_sun {
            s.env.sun_radiance = 0.0;
        }
        s
    };
    eprintln!(
        "scene: {}, camera {cam:?}; pose gradient: {}{}",
        if control {
            "arm alone (control)"
        } else {
            "whole robot + checker"
        },
        if use_vis {
            "host edges + Visibility"
        } else {
            "GpuEdges"
        },
        if edge_only {
            " alone".to_string()
        } else {
            format!(" + interior ({interior_spp} spp)")
        }
    );

    let truth_phi = arm::TRUE_PHI;
    let truth_theta = arm::TRUE_THETA;
    let mut phi = [
        truth_phi[0] + 0.30,
        truth_phi[1] - 0.22,
        truth_phi[2] + 0.40,
    ];
    let mut theta = [0.6, 0.55, 0.62, 0.70];
    if let Some(a) = at {
        phi = a;
    }
    let (phi0, theta0) = (phi, theta);

    let mut reference = make(32, &truth_phi);
    let mut gr = GpuPrb::new(ctx, &reference);
    let target = match &photo {
        Some(p) => {
            let (size, img) = read_ppm(Path::new(p));
            assert_eq!(size, SIZE, "target must be {SIZE:?}");
            img
        }
        None => accumulate(&mut gr, &reference, &truth_theta, 1000, 8),
    };
    arm::ppm(&out.join("target.ppm"), SIZE, &target, EXPOSURE);
    if std::env::var("SCAN").is_ok() {
        // The loss along the elbow alone (pitch and roll at truth, θ at
        // truth): is there a minimum short of the truth?
        for k in 0..=14 {
            let e = (-80.0 + 5.0 * k as f64).to_radians();
            let p = [truth_phi[0], truth_phi[1], e];
            rig.pose(&mut reference, &p);
            gr.update_geometry(&reference);
            let img = accumulate(&mut gr, &reference, &truth_theta, 3000, 4);
            let l = img.iter().zip(&target).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / npx_of();
            let r: Vec<String> = [4usize, 8, 16]
                .iter()
                .map(|&rad| {
                    let (bi, bt) = (blur(&img, rad), blur(&target, rad));
                    let lb = bi.iter().zip(&bt).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / npx_of();
                    format!("blur{rad} {lb:.4e}")
                })
                .collect();
            eprintln!("scan elbow {:+6.1}°  mse {l:.4e}  {}", e.to_degrees(), r.join("  "));
        }
        return;
    }
    if target_only {
        rig.pose(&mut reference, &phi0);
        gr.update_geometry(&reference);
        let img = accumulate(&mut gr, &reference, &theta0, 1000, 4);
        arm::ppm(&out.join("start.ppm"), SIZE, &img, EXPOSURE);
        return;
    }
    let target64: Vec<f64> = target.iter().map(|&x| x as f64).collect();

    let mut scene = make(8, &phi);
    let mut g = GpuPrb::new(ctx, &scene);
    g.set_target(&target64);
    let edges = Edges::new(&scene);
    let n_edge = 16384;
    // The edge wavefront is built for 2n points; GpuEdges overwrites them.
    let es0 = vis::EdgeSamples::new(&scene, &edges, &rig, &phi, n_edge, 1);
    let points0 = es0.points().to_vec();
    scene.set_points(Some(points0.clone()));
    let mut ge = GpuPrb::new(ctx, &scene);
    let mut gedges = GpuEdges::new(&ge, &scene, &edges, &rig.moving, n_edge, 3, true);
    scene.set_points(None);

    if check {
        // The pose gradient at phi0 (θ held at truth, so only the pose
        // differs from the target) against central differences of the
        // 256-spp image's MSE, common random numbers per seed.
        let th = truth_theta;
        let seeds: u64 = std::env::var("SEEDS").map_or(16, |s| s.parse().unwrap());
        rig.pose(&mut reference, &phi0);
        gr.update_geometry(&reference);
        let base = accumulate(&mut gr, &reference, &th, 100, 8);
        let d_img: Vec<f64> = base
            .iter()
            .zip(&target)
            .map(|(a, b)| 2.0 * (a - b) as f64 / npx_of())
            .collect();
        rig.pose(&mut scene, &phi0);
        ge.update_geometry(&scene);
        let (mut er, mut ir, mut cr) = (Vec::new(), Vec::new(), Vec::new());
        for sd in 0..seeds {
            let es = 300 + sd;
            if use_vis {
                let vb = vis::Visibility::new(&scene);
                er.push(
                    vis::boundary_with(&mut scene, &edges, &rig, &phi0, &d_img, n_edge, es, Some(&vb), |s| {
                        gpu::widen(&ge.forward(s, &th, es))
                    })
                    .0,
                );
            } else {
                scene.set_points(Some(points0.clone()));
                let smp = gedges.sample(&ge, &scene, &rig, &phi0, es);
                let l = gpu::widen(&ge.forward(&scene, &th, es));
                er.push(smp.gradient(&scene, es, &l, &d_img));
                scene.set_points(None);
            }
            let spp = scene.spp;
            scene.spp = 4;
            let tan = Tangents::rig(&scene, &rig, &phi0);
            ir.push(interior::interior(&scene, &tan, &th, 600 + sd, &d_img));
            scene.spp = spp;
        }
        let hh = 0.01;
        for sd in 0..8u64 {
            let mut r = Vec::new();
            for j in 0..3 {
                let mut side = |d: f64| {
                    let mut p = phi0;
                    p[j] += d;
                    rig.pose(&mut reference, &p);
                    gr.update_geometry(&reference);
                    let img = accumulate(&mut gr, &reference, &th, 900 + 8 * sd, 8);
                    img.iter()
                        .zip(&target)
                        .map(|(a, b)| ((a - b) as f64).powi(2))
                        .sum::<f64>()
                        / npx_of()
                };
                r.push((side(hh) - side(-hh)) / (2.0 * hh));
            }
            cr.push(r);
        }
        let stat = |v: &[Vec<f64>]| {
            let n = v.len() as f64;
            (0..3)
                .map(|j| {
                    let m = v.iter().map(|x| x[j]).sum::<f64>() / n;
                    let s2 = v.iter().map(|x| (x[j] - m).powi(2)).sum::<f64>() / (n - 1.0);
                    (m, (s2 / n).sqrt())
                })
                .collect::<Vec<_>>()
        };
        if std::env::var("DUMP").is_ok() {
            for (k, x) in er.iter().enumerate() {
                eprintln!("  edge seed {k}: {:+.3e} {:+.3e} {:+.3e}", x[0], x[1], x[2]);
            }
        }
        let (e, i, c) = (stat(&er), stat(&ir), stat(&cr));
        let tot: Vec<Vec<f64>> = er.iter().zip(&ir).map(|(a, b)| (0..3).map(|j| a[j] + b[j]).collect()).collect();
        let t = stat(&tot);
        let f = |v: &[(f64, f64)]| v.iter().map(|(m, s)| format!("{m:+.3e} ± {s:.1e}")).collect::<Vec<_>>().join(", ");
        eprintln!("check at φ° {:+.2?} (depth {}, nee {}, sun {})", phi0.map(f64::to_degrees), scene.depth, scene.nee, scene.env.sun_radiance);
        eprintln!("  edge      {}", f(&e));
        eprintln!("  interior  {}", f(&i));
        eprintln!("  sum       {}", f(&t));
        eprintln!("  CD        {}", f(&c));
        eprintln!(
            "  z(sum-CD) {:?}",
            (0..3).map(|j| ((t[j].0 - c[j].0) / (t[j].1.hypot(c[j].1))).round()).collect::<Vec<_>>()
        );
        return;
    }

    let tail = 40;
    // Pose first: until the silhouettes overlap, the finish would only
    // learn to brighten what is in the wrong place.
    let (b1, b2) = (0.9, 0.99);
    let (lr_phi, lr_theta) = (0.03, 0.04);
    let (mut m, mut v, mut avg) = ([0.0; 7], [0.0; 7], [0.0; 7]);
    let mut csv = fs::File::create(out.join("loss.csv")).unwrap();
    writeln!(
        csv,
        "iter,loss,sho_pitch,sho_roll,elbow,alpha,f0_450,f0_550,f0_650,\
         g_edge_pitch,g_edge_roll,g_edge_elbow,g_int_pitch,g_int_roll,g_int_elbow,\
         ms_theta,ms_edge,ms_interior"
    )
    .unwrap();
    let mut strip = fs::File::create(out.join("filmstrip.csv")).unwrap();
    writeln!(
        strip,
        "iter,mse,sho_pitch,sho_roll,elbow,alpha,f0_450,f0_550,f0_650"
    )
    .unwrap();
    let mse = |img: &[f32]| {
        img.iter()
            .zip(&target)
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            / img.len() as f64
    };
    let mut render = |phi: &[f64; 3], theta: &[f64; 4]| {
        rig.pose(&mut reference, phi);
        gr.update_geometry(&reference);
        accumulate(&mut gr, &reference, theta, 2000, 4)
    };
    let mut preview = |it: usize, phi: &[f64; 3], theta: &[f64; 4], strip: &mut fs::File| {
        let img = render(phi, theta);
        arm::ppm(&out.join(format!("iter_{it:03}.ppm")), SIZE, &img, EXPOSURE);
        let e = mse(&img);
        writeln!(
            strip,
            "{it},{e:e},{},{},{},{},{},{},{}",
            phi[0], phi[1], phi[2], theta[0], theta[1], theta[2], theta[3]
        )
        .unwrap();
        e
    };
    preview(0, &phi, &theta, &mut strip);
    let t = Instant::now();
    let (mut t_theta, mut t_edge, mut t_int) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let npx = (SIZE.0 * SIZE.1 * 3) as f64;
    let target_blur = if blur_r > 0 {
        blur(&target, blur_r)
    } else {
        Vec::new()
    };
    for it in 1..=iters {
        rig.pose(&mut scene, &phi);
        g.update_geometry(&scene);
        let ti = Instant::now();
        let seeds = (2 * it as u64, 2 * it as u64 + 1);
        let (loss, gtheta) = g.gradient_l2(&scene, &theta, seeds);
        let img = g.last_image();
        let d_theta = ti.elapsed();

        let ti = Instant::now();
        // Coarse to fine (`--blur R`): while the pose fits alone, its loss
        // compares box-blurred images, `|B I − B T|²`, whose adjoint is
        // `Bᵀ 2 (B I − B T)` (B is symmetric up to the clamped border).
        let d_img: Vec<f64> = if blur_r > 0 && it <= pose_only {
            let r: Vec<f32> = blur(&img, blur_r)
                .iter()
                .zip(&target_blur)
                .map(|(a, b)| (2.0 * (a - b) / npx) as f32)
                .collect();
            blur(&r, blur_r)
        } else {
            img.iter()
                .zip(&target)
                .map(|(a, b)| 2.0 * (a - b) as f64 / npx)
                .collect()
        };
        ge.update_geometry(&scene);
        let es = 7000 + it as u64;
        let g_edge = if use_vis {
            let vb = vis::Visibility::new(&scene);
            vis::boundary_with(
                &mut scene,
                &edges,
                &rig,
                &phi,
                &d_img,
                n_edge,
                es,
                Some(&vb),
                |s| gpu::widen(&ge.forward(s, &theta, es)),
            )
            .0
        } else {
            scene.set_points(Some(points0.clone()));
            let smp = gedges.sample(&ge, &scene, &rig, &phi, es);
            let l = gpu::widen(&ge.forward(&scene, &theta, es));
            let bad_w = smp.weights().iter().filter(|w| w.iter().any(|x| !x.is_finite())).count();
            let bad_l = l.iter().filter(|x| !x.is_finite()).count();
            if bad_w + bad_l > 0 {
                eprintln!("it {it}: {bad_w} non-finite weights, {bad_l} non-finite L of {}", l.len());
            }
            let gr = smp.gradient(&scene, es, &l, &d_img);
            scene.set_points(None);
            gr
        };
        let d_edge = ti.elapsed();

        // The interior term, CPU f64, at its own (lower) path count: the
        // loss adjoint is the 8-spp image's, the derivative an independent
        // estimate, so the product stays unbiased.
        let ti = Instant::now();
        let g_int = if edge_only {
            vec![0.0; 3]
        } else {
            let spp = scene.spp;
            scene.spp = interior_spp;
            let tan = Tangents::rig(&scene, &rig, &phi);
            let gi = interior::interior(&scene, &tan, &theta, 50_000 + it as u64, &d_img);
            scene.spp = spp;
            gi
        };
        let d_int = ti.elapsed();
        t_theta += d_theta;
        t_edge += d_edge;
        t_int += d_int;
        writeln!(
            csv,
            "{it},{loss:e},{},{},{},{},{},{},{},{:e},{:e},{:e},{:e},{:e},{:e},{:.2},{:.2},{:.2}",
            phi[0],
            phi[1],
            phi[2],
            theta[0],
            theta[1],
            theta[2],
            theta[3],
            g_edge[0],
            g_edge[1],
            g_edge[2],
            g_int[0],
            g_int[1],
            g_int[2],
            ms(d_theta),
            ms(d_edge),
            ms(d_int)
        )
        .unwrap();
        let grad: Vec<f64> = (0..3)
            .map(|j| g_edge[j] + g_int[j])
            .chain(gtheta.iter().copied())
            .collect();
        for p in 0..7 {
            if p >= 3 && it <= pose_only {
                continue;
            }
            m[p] = b1 * m[p] + (1.0 - b1) * grad[p];
            v[p] = b2 * v[p] + (1.0 - b2) * grad[p] * grad[p];
            let lr = if p < 3 { lr_phi } else { lr_theta };
            // θ's Adam starts when θ does, so its bias correction counts from there.
            let n = if p < 3 {
                it
            } else {
                it.saturating_sub(pose_only).max(1)
            } as i32;
            let step = lr / (1.0 + it as f64 / 150.0) * (m[p] / (1.0 - b1.powi(n)))
                / ((v[p] / (1.0 - b2.powi(n))).sqrt() + 1e-12);
            if p < 3 {
                phi[p] -= step;
            } else {
                theta[p - 3] = (theta[p - 3] - step).clamp(0.02, 1.0);
            }
        }
        if it > iters - tail {
            for p in 0..7 {
                avg[p] += if p < 3 { phi[p] } else { theta[p - 3] } / tail as f64;
            }
        }
        if [25, 75, 150, 250].contains(&it) {
            preview(it, &phi, &theta, &mut strip);
        }
        if it % 25 == 0 || it == 1 {
            eprintln!(
                "it {it:3}  loss {loss:.3e}  φ° {:+.2?}  θ {theta:.3?}  ({:.0} θ / {:.0} edge / {:.0} interior ms)",
                phi.map(f64::to_degrees),
                ms(d_theta),
                ms(d_edge),
                ms(d_int)
            );
        }
    }
    let phi_hat = [avg[0], avg[1], avg[2]];
    let theta_hat = [avg[3], avg[4], avg[5], avg[6]];
    let e_hat = preview(iters, &phi_hat, &theta_hat, &mut strip);
    let e_truth = mse(&render(&truth_phi, &truth_theta));
    let deg = |x: f64| (x.to_degrees() * 100.0).round() / 100.0;
    let per = |d: Duration| ms(d) / iters as f64;
    eprintln!(
        "{iters} iterations in {:.1?} (per iteration: θ gradient {:.1} ms, edge term {:.1} ms, interior {:.1} ms; {n_edge} pairs)",
        t.elapsed(),
        per(t_theta),
        per(t_edge),
        per(t_int)
    );
    eprintln!(
        "φ (mean of the last {tail}) start {:?}° → {:?}° vs truth {:?}° (error {:?}°)",
        phi0.map(deg),
        phi_hat.map(deg),
        truth_phi.map(deg),
        [0, 1, 2].map(|j| deg(phi_hat[j] - truth_phi[j])),
    );
    eprintln!("θ start {theta0:.3?} → {theta_hat:.3?} vs truth {truth_theta:?}");
    eprintln!("image MSE {e_hat:.3e} (truth re-rendered: {e_truth:.3e})");
}
