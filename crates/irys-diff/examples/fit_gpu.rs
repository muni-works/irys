//! `fit` on the GPU: recover the gold ball's roughness and colour by path
//! replay backprop on the device wavefront ([`irys_diff::gpu`]).
//!
//! `cargo run -p irys-diff --features gpu --example fit_gpu -- out/fit_gpu`
//! writes `target.ppm`, `iter_NNN.ppm` and `loss.csv`, with the same
//! schedule, seeds and Adam as `fit`.

use irys_diff::gpu::{self, GpuPrb};
use irys_diff::prb::{self, L2, Scene};
use std::{fs, io::Write, path::Path, time::Instant};

fn ppm(path: &Path, (w, h): (usize, usize), img: &[f64]) {
    let mut f = fs::File::create(path).unwrap();
    write!(f, "P6\n{w} {h}\n255\n").unwrap();
    let srgb = |x: f64| {
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
        .unwrap_or_else(|| "out/fit_gpu".into());
    let out = Path::new(&out);
    fs::create_dir_all(out).unwrap();
    let ctx = gpu::context().expect("fit_gpu needs a GPU adapter");

    let truth = [0.2, 0.34, 0.78, 1.0];
    let mut theta = [0.7, 0.5, 0.5, 0.5];

    let reference = Scene::new(96, 96, 64);
    let mut gr = GpuPrb::new(ctx, &reference);
    let render = |g: &mut GpuPrb, th: &[f64; 4]| {
        prb::image(
            &reference,
            1000,
            &gpu::widen(&g.forward(&reference, th, 1000)),
        )
    };
    let target = render(&mut gr, &truth);
    ppm(&out.join("target.ppm"), reference.size, &target);

    let scene = Scene::new(96, 96, 8);
    let mut g = GpuPrb::new(ctx, &scene);
    let (lr, b1, b2) = (0.04, 0.9, 0.99);
    let (mut m, mut v) = ([0.0; 4], [0.0; 4]);
    let mut csv = fs::File::create(out.join("loss.csv")).unwrap();
    writeln!(csv, "iter,loss,alpha,f0_450,f0_550,f0_650").unwrap();
    let t = Instant::now();
    for it in 1..=60 {
        let seeds = (2 * it as u64, 2 * it as u64 + 1);
        let (loss, grad) = gpu::gradient(&mut g, &scene, &theta, seeds, &L2(&target));
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
                &render(&mut gr, &theta),
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
}
