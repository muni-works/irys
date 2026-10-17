//! Phase 2b: recover the gold ball's roughness and colour in the garden —
//! a triangle-mesh floor and a bush of Gaussian splats, traced through
//! irys-core's BVH on the GPU wavefront — then render the fitted scene.
//!
//! `cargo run -p irys-diff --features gpu --example fit_scene -- out/fit_scene`
//! writes `target.ppm`, `iter_NNN.ppm`, `loss.csv` and `beauty.ppm`
//! (384×384, 256 spp, depth 6, the fitted θ).

use irys_diff::gpu::{self, GpuPrb};
use irys_diff::prb::Scene;
use std::{fs, io::Write, path::Path, time::Instant};

fn ppm(path: &Path, (w, h): (usize, usize), img: &[f32]) {
    let mut f = fs::File::create(path).unwrap();
    write!(f, "P6\n{w} {h}\n255\n").unwrap();
    let srgb = |x: f32| {
        let x = x.clamp(0.0, 1.0);
        let s = if x <= 0.0031308 {
            12.92 * x
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0 + 0.5) as u8
    };
    f.write_all(&img.iter().map(|&x| srgb(x)).collect::<Vec<_>>())
        .unwrap();
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "out/fit_scene".into());
    let out = Path::new(&out);
    fs::create_dir_all(out).unwrap();
    let ctx = gpu::context().expect("fit_scene needs a GPU adapter");

    let truth = [0.2, 0.34, 0.78, 1.0];
    let mut theta = [0.7, 0.5, 0.5, 0.5];

    let reference = Scene::garden(96, 96, 64);
    let mut gr = GpuPrb::new(ctx, &reference);
    let target = gr.image(&reference, &truth, 1000);
    ppm(&out.join("target.ppm"), reference.size, &target);

    let scene = Scene::garden(96, 96, 8);
    let mut g = GpuPrb::new(ctx, &scene);
    g.set_target(&target.iter().map(|&x| x as f64).collect::<Vec<_>>());
    let (lr, b1, b2) = (0.04, 0.9, 0.99);
    let (mut m, mut v) = ([0.0; 4], [0.0; 4]);
    let mut csv = fs::File::create(out.join("loss.csv")).unwrap();
    writeln!(csv, "iter,loss,alpha,f0_450,f0_550,f0_650").unwrap();
    let t = Instant::now();
    for it in 1..=60 {
        let seeds = (2 * it as u64, 2 * it as u64 + 1);
        let (loss, grad) = g.gradient_l2(&scene, &theta, seeds);
        writeln!(
            csv,
            "{it},{loss:e},{},{},{},{}",
            theta[0], theta[1], theta[2], theta[3]
        )
        .unwrap();
        if it == 1 || it % 10 == 0 {
            ppm(
                &out.join(format!("iter_{it:03}.ppm")),
                reference.size,
                &gr.image(&reference, &theta, 1000),
            );
        }
        for p in 0..4 {
            m[p] = b1 * m[p] + (1.0 - b1) * grad[p];
            v[p] = b2 * v[p] + (1.0 - b2) * grad[p] * grad[p];
            let step = lr / (1.0 + it as f64 / 30.0) * (m[p] / (1.0 - b1.powi(it)))
                / ((v[p] / (1.0 - b2.powi(it))).sqrt() + 1e-12);
            theta[p] = (theta[p] - step).clamp(0.02, 1.0);
        }
        eprintln!("it {it:3}  loss {loss:.3e}  θ {theta:.3?}");
    }
    eprintln!("60 iterations in {:?}; truth {truth:?}", t.elapsed());

    // The beauty frame: 64 frames of 4 spp, averaged.
    let t = Instant::now();
    let mut beauty = Scene::garden(384, 384, 4);
    beauty.depth = 6;
    let mut gb = GpuPrb::new(ctx, &beauty);
    let frames = 64;
    let mut acc = vec![0.0f64; 384 * 384 * 3];
    for f in 0..frames {
        for (a, x) in acc.iter_mut().zip(gb.image(&beauty, &theta, 5000 + f)) {
            *a += x as f64 / frames as f64;
        }
    }
    let acc: Vec<f32> = acc.iter().map(|&x| x as f32).collect();
    ppm(&out.join("beauty.ppm"), beauty.size, &acc);
    eprintln!("beauty 384×384×{} in {:?}", 4 * frames, t.elapsed());
}
