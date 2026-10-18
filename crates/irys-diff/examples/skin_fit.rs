//! Recover skin from a picture of it by gradients: melanin, blood and the
//! specular roughness of the Lee Perry-Smith head, from a rendered target,
//! by path replay backpropagation through the random walk (CPU, f64).
//!
//! ```text
//! cargo run --release -p irys-diff --features assets --example skin_fit -- docs
//! ```
//!
//! (96² × 16 spp per gradient, 300 Adam steps with cosine decay) writes
//! `skin-fit.png` (target, start, fit, and the three parameters over
//! the iterations, each over its true value) and `skin-fit.csv`.
//!
//! The head is "Lee Perry-Smith" by Infinite-Realities (ir-ltd.net),
//! licensed CC BY 3.0, read from `$IRYS_ASSETS/lee-perry-smith`.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use irys_diff::prb::L2;
use irys_diff::skin::graph::PARAMS;
use irys_diff::skin::render::{self, Adjoints, Chromophores, Mode, Scene, tonemap};
use irys_diff::skin::{head, mesh};

#[path = "common/font.rs"]
mod font;

/// The truth: uniform chromophores (bulk melanosome and blood fractions),
/// the measured density, roughness 0.30 / 0.6.
const TRUTH: [f64; PARAMS] = [0.03, 0.012, 1.0, 0.30, 0.6];
/// Where the fit starts: three times the melanin, a third of the blood, a
/// rougher layer.
const START: [f64; PARAMS] = [0.09, 0.004, 1.0, 0.50, 0.6];
/// Which parameters move: melanin, blood, r₁.
const FREE: [usize; 3] = [0, 1, 3];
const NAMES: [&str; 3] = ["MELANIN", "BLOOD", "ROUGHNESS"];

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next().unwrap_or_else(|| "docs".into());
    let iters: usize = args.next().map_or(300, |s| s.parse().expect("iterations"));
    let out = Path::new(&out);
    std::fs::create_dir_all(out).expect("out dir");
    let asset = mesh::load(&mesh::asset_dir()).expect("the Lee Perry-Smith asset");

    let n = 96;
    let mut s = head::scene(&asset, Chromophores::Uniform, head::View::Front);
    s.set_size(n, n);
    s.mode = Mode {
        sss: true,
        dual: true,
    };
    s.depth = 3;
    s.camera.aperture = 0.0;

    // The target: the truth at 32 spp.
    let t = Instant::now();
    s.spp = 32;
    let target = render::image(&s, 7, &render::forward(&s, &TRUTH, 7).0);
    println!("target {n}² × 32 spp: {:.1} s", t.elapsed().as_secs_f64());
    s.spp = 16;

    let mut adj = Adjoints::new(1 << 14, true);
    println!("adjoints compiled: {}", adj.is_native());
    let mut theta = START;
    // Adam on (ln m, ln h, r₁).
    let (b1, b2, lr) = (0.9, 0.999, [0.08, 0.08, 0.03]);
    let (mut m1, mut m2) = ([0.0; 3], [0.0; 3]);
    let mut csv = std::fs::File::create(out.join("skin-fit.csv")).expect("csv");
    writeln!(csv, "iteration,loss,melanin,blood,roughness,seconds").unwrap();
    let mut history = vec![[theta[0], theta[1], theta[3]]];
    let t0 = Instant::now();
    for it in 0..iters {
        let seeds = (1000 + 2 * it as u64, 1001 + 2 * it as u64);
        let (loss, g) = render::gradient(&s, &theta, seeds, &L2(&target), &mut adj);
        for (j, &k) in FREE.iter().enumerate() {
            // Into the optimiser's coordinates: ∂/∂ ln θ = θ ∂/∂θ.
            let gj = if k == 3 { g[k] } else { g[k] * theta[k] };
            m1[j] = b1 * m1[j] + (1.0 - b1) * gj;
            m2[j] = b2 * m2[j] + (1.0 - b2) * gj * gj;
            let mh = m1[j] / (1.0 - b1.powi(it as i32 + 1));
            let vh = m2[j] / (1.0 - b2.powi(it as i32 + 1));
            // Cosine decay to a tenth: the late iterations average noise.
            let decay =
                0.1 + 0.45 * (1.0 + (std::f64::consts::PI * it as f64 / iters as f64).cos());
            let step = decay * lr[j] * mh / (vh.sqrt() + 1e-12);
            if k == 3 {
                theta[k] = (theta[k] - step).clamp(0.08, 0.95);
            } else {
                theta[k] *= (-step).exp();
            }
        }
        history.push([theta[0], theta[1], theta[3]]);
        writeln!(
            csv,
            "{it},{loss:.6e},{:.6},{:.6},{:.6},{:.2}",
            theta[0],
            theta[1],
            theta[3],
            t0.elapsed().as_secs_f64()
        )
        .unwrap();
        if it % 10 == 0 || it + 1 == iters {
            println!(
                "{it:3}  loss {loss:.4e}  melanin {:.4}  blood {:.4}  r1 {:.3}  ({:.0} s)",
                theta[0],
                theta[1],
                theta[3],
                t0.elapsed().as_secs_f64()
            );
        }
    }
    let last = history.len().saturating_sub(10);
    let tail: Vec<f64> = (0..3)
        .map(|j| history[last..].iter().map(|h| h[j]).sum::<f64>() / (history.len() - last) as f64)
        .collect();
    println!(
        "last-10 mean: melanin {:.4} (truth {}), blood {:.4} ({}), r1 {:.3} ({})",
        tail[0], TRUTH[0], tail[1], TRUTH[1], tail[2], TRUTH[3]
    );

    // The figure: target | start | fit at 64 spp, 3× nearest, and the plot.
    s.spp = 64;
    let imgs = [TRUTH, START, theta].map(|th| render::image(&s, 3, &render::forward(&s, &th, 3).0));
    figure(out, &s, &imgs, &history);
}

fn figure(out: &Path, s: &Scene, imgs: &[Vec<f64>; 3], history: &[[f64; 3]]) {
    let (n, k) = (s.size.0, 3);
    let side = n * k;
    let (w, h) = (3 * side + side * 3 / 2, side);
    let mut px = vec![12u8; w * h * 3];
    let exposure = 1.8;
    for (i, img) in imgs.iter().enumerate() {
        let tm = tonemap(img, exposure);
        for y in 0..side {
            for x in 0..side {
                let src = ((y / k) * n + x / k) * 3;
                let dst = (y * w + i * side + x) * 3;
                px[dst..dst + 3].copy_from_slice(&tm[src..src + 3]);
            }
        }
        font::text(
            &mut px,
            w,
            i * side + 6,
            6,
            ["TARGET", "START", "FIT"][i],
            2,
        );
    }
    // Each parameter over its truth, iterations along x.
    let (x0, pw) = (3 * side + 20, side * 3 / 2 - 30);
    let (y0, ph) = (60, h - 90);
    let truth = [TRUTH[0], TRUTH[1], TRUTH[3]];
    let (lo, hi) = (0.0f64, 3.5f64);
    let ymap = |v: f64| y0 + ((hi - v.clamp(lo, hi)) / (hi - lo) * ph as f64) as usize;
    let colours = [[235, 120, 90], [120, 200, 120], [120, 160, 240]];
    for x in x0..x0 + pw {
        let y = ymap(1.0);
        px[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&[90, 90, 90]);
    }
    for (j, col) in colours.iter().enumerate() {
        let pts: Vec<(usize, usize)> = history
            .iter()
            .enumerate()
            .map(|(i, hv)| {
                (
                    x0 + i * (pw - 1) / (history.len() - 1).max(1),
                    ymap(hv[j] / truth[j]),
                )
            })
            .collect();
        for p in pts.windows(2) {
            line(&mut px, w, p[0], p[1], *col);
        }
        let ly = 6 + j * 16;
        font::text(&mut px, w, x0 + 16, ly, &format!("{} / TRUE", NAMES[j]), 1);
        for dy in 0..7 {
            for dx in 0..10 {
                let (x, y) = (x0 + dx, ly + 3 + dy);
                px[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(col);
            }
        }
    }
    font::text(&mut px, w, x0, h - 24, "ITERATIONS. GREY: TRUTH (1.0)", 1);
    let path = out.join("skin-fit.png");
    image::save_buffer(
        &path,
        &px,
        w as u32,
        h as u32,
        image::ExtendedColorType::Rgb8,
    )
    .expect("png");
    println!("wrote {}", path.display());
}

fn line(px: &mut [u8], w: usize, a: (usize, usize), b: (usize, usize), c: [u8; 3]) {
    let steps = (a.0.abs_diff(b.0)).max(a.1.abs_diff(b.1)).max(1);
    for i in 0..=steps {
        let t = i as f64 / steps as f64;
        let x = (a.0 as f64 + (b.0 as f64 - a.0 as f64) * t).round() as usize;
        let y = (a.1 as f64 + (b.1 as f64 - a.1 as f64) * t).round() as usize;
        for (dx, dy) in [(0, 0), (1, 0), (0, 1)] {
            let o = ((y + dy) * w + x + dx) * 3;
            if o + 3 <= px.len() {
                px[o..o + 3].copy_from_slice(&c);
            }
        }
    }
}
