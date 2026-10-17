//! Watch an inverse render converge: recover a ball's roughness and colour
//! from one target image by path replay backprop.
//!
//! `cargo run -p irys-diff --example fit -- out/fit` writes `target.ppm`,
//! `iter_NNN.ppm` and `loss.csv`.

use irys_diff::prb::{self, Kernels, L2, Scene};
use std::{fs, io::Write, path::Path};

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
    let out = std::env::args().nth(1).unwrap_or_else(|| "out/fit".into());
    let out = Path::new(&out);
    fs::create_dir_all(out).unwrap();

    // Gold, fairly glossy (bands are 450, 550, 650 nm: blue first). The fit starts from a rough grey ball.
    let truth = [0.2, 0.34, 0.78, 1.0];
    let mut theta = [0.7, 0.5, 0.5, 0.5];

    let reference = Scene::new(96, 96, 64);
    let mut kr = Kernels::new(reference.lanes(), true);
    let target = prb::image(
        &reference,
        1000,
        &prb::forward(&reference, &truth, 1000, &mut kr).0,
    );
    ppm(&out.join("target.ppm"), reference.size, &target);

    let scene = Scene::new(96, 96, 8);
    let mut k = Kernels::new(scene.lanes(), true);
    let (lr, b1, b2) = (0.04, 0.9, 0.99);
    let (mut m, mut v) = ([0.0; 4], [0.0; 4]);
    let mut csv = fs::File::create(out.join("loss.csv")).unwrap();
    writeln!(csv, "iter,loss,alpha,f0_450,f0_550,f0_650").unwrap();
    for it in 1..=60 {
        let seeds = (2 * it as u64, 2 * it as u64 + 1);
        let (loss, g) = prb::gradient(&scene, &theta, seeds, &mut k, &L2(&target));
        writeln!(
            csv,
            "{it},{loss:e},{},{},{},{}",
            theta[0], theta[1], theta[2], theta[3]
        )
        .unwrap();
        if it == 1 || it % 10 == 0 {
            // Previewed at the reference's spp, so the frame shows θ, not the noise.
            let l = prb::forward(&reference, &theta, 1000, &mut kr).0;
            ppm(
                &out.join(format!("iter_{it:03}.ppm")),
                reference.size,
                &prb::image(&reference, 1000, &l),
            );
        }
        for p in 0..4 {
            m[p] = b1 * m[p] + (1.0 - b1) * g[p];
            v[p] = b2 * v[p] + (1.0 - b2) * g[p] * g[p];
            let step = lr / (1.0 + it as f64 / 30.0) * (m[p] / (1.0 - b1.powi(it)))
                / ((v[p] / (1.0 - b2.powi(it))).sqrt() + 1e-12);
            theta[p] = (theta[p] - step).clamp(0.02, 1.0);
        }
        eprintln!("it {it:3}  loss {loss:.3e}  θ {theta:.3?}");
    }
}
