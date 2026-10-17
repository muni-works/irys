//! Phase 2d: recover rose's arm pose (shoulder pitch, shoulder roll, elbow)
//! *and* its finish (GGX `α`, F0 at three bands) from a target render, by
//! path replay backprop for `θ` and edge-sampled visibility gradients for
//! `φ` ([`irys_diff::vis`]), both on the GPU wavefront.
//!
//! `cargo run --release -p irys-diff --features gpu --example rose_fit -- out/rose_fit [target.ppm]`
//! writes `target.ppm`, `iter_NNN.ppm` previews, `loss.csv` (every
//! iteration's 8-spp loss, `φ`, `θ`) and `filmstrip.csv`. With a second
//! argument the target is that image instead of an irys render (a binary
//! PPM at the fit's size, display sRGB, exposure 0.6 assumed), which is the
//! "from a photo" half of the gate.

#[path = "rose/arm.rs"]
mod arm;

use std::{fs, io::Write, path::Path, time::Instant};

use irys_diff::gpu::{self, GpuPrb};
use irys_diff::prb::Scene;
use irys_diff::vis::{self, Edges};

const SIZE: (usize, usize) = (192, 192);
const EXPOSURE: f32 = 0.6;

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

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next().unwrap_or_else(|| "out/rose_fit".into());
    let photo = args.next();
    let out = Path::new(&out);
    fs::create_dir_all(out).unwrap();
    let ctx = gpu::context().expect("rose_fit needs a GPU adapter");
    let arm = arm::load_arm(&arm::rose_dir());
    let rig = arm::rig();

    let truth_phi = arm::TRUE_PHI;
    let truth_theta = arm::TRUE_THETA;
    let mut phi = [
        truth_phi[0] + 0.30,
        truth_phi[1] - 0.22,
        truth_phi[2] + 0.40,
    ];
    let mut theta = [0.6, 0.55, 0.62, 0.70];
    let (phi0, theta0) = (phi, theta);

    let mut reference = arm::scene_from(&arm, SIZE, 32, &truth_phi, arm::FIT_CAM);
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
    let target64: Vec<f64> = target.iter().map(|&x| x as f64).collect();

    let mut scene = arm::scene_from(&arm, SIZE, 8, &phi, arm::FIT_CAM);
    let mut g = GpuPrb::new(ctx, &scene);
    g.set_target(&target64);
    let edges = Edges::new(&scene);
    let n_edge = 16384;
    let es0 = vis::EdgeSamples::new(&scene, &edges, &rig, &phi, n_edge, 1);
    scene.set_points(Some(es0.points().to_vec()));
    let mut ge = GpuPrb::new(ctx, &scene);
    scene.set_points(None);

    let iters = 300;
    let tail = 40;
    // Pose first: until the silhouettes overlap, the finish would only
    // learn to brighten what is in the wrong place.
    let pose_only = 100;
    let (b1, b2) = (0.9, 0.99);
    let (lr_phi, lr_theta) = (0.03, 0.04);
    let (mut m, mut v, mut avg) = ([0.0; 7], [0.0; 7], [0.0; 7]);
    let mut csv = fs::File::create(out.join("loss.csv")).unwrap();
    writeln!(
        csv,
        "iter,loss,sho_pitch,sho_roll,elbow,alpha,f0_450,f0_550,f0_650"
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
    let (mut t_theta, mut t_edge) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
    let npx = (SIZE.0 * SIZE.1 * 3) as f64;
    for it in 1..=iters as usize {
        rig.pose(&mut scene, &phi);
        g.update_geometry(&scene);
        let ti = Instant::now();
        let seeds = (2 * it as u64, 2 * it as u64 + 1);
        let (loss, gtheta) = g.gradient_l2(&scene, &theta, seeds);
        let img = g.last_image();
        t_theta += ti.elapsed();
        let ti = Instant::now();
        let d_img: Vec<f64> = img
            .iter()
            .zip(&target)
            .map(|(a, b)| 2.0 * (a - b) as f64 / npx)
            .collect();
        ge.update_geometry(&scene);
        let es = 7000 + it as u64;
        let (gphi, _) = vis::boundary(&mut scene, &edges, &rig, &phi, &d_img, n_edge, es, |s| {
            gpu::widen(&ge.forward(s, &theta, es))
        });
        t_edge += ti.elapsed();
        writeln!(
            csv,
            "{it},{loss:e},{},{},{},{},{},{},{}",
            phi[0], phi[1], phi[2], theta[0], theta[1], theta[2], theta[3]
        )
        .unwrap();
        let grad: Vec<f64> = gphi.iter().chain(gtheta.iter()).copied().collect();
        for p in 0..7 {
            if p >= 3 && it <= pose_only {
                continue;
            }
            m[p] = b1 * m[p] + (1.0 - b1) * grad[p];
            v[p] = b2 * v[p] + (1.0 - b2) * grad[p] * grad[p];
            let lr = match p {
                0..3 => lr_phi,
                _ if it <= pose_only => 0.0,
                _ => lr_theta,
            };
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
            eprintln!("it {it:3}  loss {loss:.3e}  φ {phi:+.3?}  θ {theta:.3?}");
        }
    }
    let phi_hat = [avg[0], avg[1], avg[2]];
    let theta_hat = [avg[3], avg[4], avg[5], avg[6]];
    let e_hat = preview(iters, &phi_hat, &theta_hat, &mut strip);
    let e_truth = mse(&render(&truth_phi, &truth_theta));
    let deg = |x: f64| (x.to_degrees() * 100.0).round() / 100.0;
    eprintln!(
        "{iters} iterations in {:.1?} (θ gradient {:.1} ms, edge term {:.1} ms per iteration, {n_edge} pairs)",
        t.elapsed(),
        t_theta.as_secs_f64() * 1e3 / iters as f64,
        t_edge.as_secs_f64() * 1e3 / iters as f64
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
