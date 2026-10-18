//! Phase 4: `docs/look-volume.png` — a heterogeneous smoke cube with a
//! warm emissive core over a grey floor, one softbox, delta tracking with
//! ratio-tracked light samples, spectral σ (bluer light scatters more).
//!
//! `cargo run --release -p irys-diff --features assets --example look_volume -- [w] [h] [spp] [grid]`

use irys_core::Point3;
use irys_diff::look::volume::{Grid, Medium, VolumeScene};
use irys_diff::skin::render::{Light, tonemap};

fn hash(i: i64, j: i64, k: i64) -> f64 {
    let mut h = (i.wrapping_mul(73856093) ^ j.wrapping_mul(19349663) ^ k.wrapping_mul(83492791)) as u64;
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51afd7ed558ccd);
    h ^= h >> 33;
    (h >> 11) as f64 / (1u64 << 53) as f64
}

fn value_noise(p: [f64; 3]) -> f64 {
    let f = p.map(f64::floor);
    let t = [p[0] - f[0], p[1] - f[1], p[2] - f[2]].map(|x| x * x * (3.0 - 2.0 * x));
    let (i, j, k) = (f[0] as i64, f[1] as i64, f[2] as i64);
    let mut v = 0.0;
    for dk in 0..2 {
        for dj in 0..2 {
            for di in 0..2 {
                let w = (if di == 1 { t[0] } else { 1.0 - t[0] })
                    * (if dj == 1 { t[1] } else { 1.0 - t[1] })
                    * (if dk == 1 { t[2] } else { 1.0 - t[2] });
                v += w * hash(i + di, j + dj, k + dk);
            }
        }
    }
    v
}

fn fbm(p: [f64; 3]) -> f64 {
    (0..5)
        .map(|o| {
            let s = 2f64.powi(o);
            value_noise(p.map(|x| x * s)) / s
        })
        .sum::<f64>()
        / 1.9375
}

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let (w, h, spp) = (a.first().copied().unwrap_or(640), a.get(1).copied().unwrap_or(400), a.get(2).copied().unwrap_or(256));
    let (lo, hi) = (Point3::new(-0.6, -0.6, 0.0), Point3::new(0.6, 0.6, 1.2));
    let density = Grid::from_fn(lo, hi, [96, 96, 96], |u| {
        let n = fbm([u[0] * 4.0, u[1] * 4.0, u[2] * 4.0 + 1.3]);
        // A soft-edged cube of billowing smoke, thinning upward.
        let edge = u.iter().map(|&x| (x.min(1.0 - x) * 12.0).min(1.0)).product::<f64>();
        ((n - 0.38) * 3.0).max(0.0) * edge * (1.15 - 0.6 * u[2])
    });
    let emission = Grid::from_fn(lo, hi, [48, 48, 48], |u| {
        let r = ((u[0] - 0.5).powi(2) + (u[1] - 0.5).powi(2) + ((u[2] - 0.32) * 0.8).powi(2)).sqrt();
        (1.0 - r / 0.22).max(0.0).powi(2) * fbm([u[0] * 6.0, u[1] * 6.0, u[2] * 6.0])
    });
    let scene = VolumeScene {
        medium: Medium {
            density,
            sigma: [5.0, 6.0, 7.5],
            g: 0.35,
            emission: Some((emission, [30.0, 9.0, 1.5])),
            majorant: None,
        },
        floor: Some([0.45, 0.45, 0.45]),
        lights: vec![
            Light::facing(Point3::new(-1.6, -1.2, 2.6), Point3::new(0.0, 0.0, 0.5), 1.4, 1.4, 9.0, [1.0, 0.97, 0.92]),
            Light::facing(Point3::new(3.2, 0.5, 2.2), Point3::new(0.0, 0.0, 0.6), 2.0, 2.0, 1.5, [0.8, 0.9, 1.0]),
        ],
        env: 0.02,
        eye: Point3::new(0.4, -3.4, 1.15),
        target: Point3::new(0.0, 0.0, 0.5),
        vfov: 30.0,
        size: (w, h),
        spp,
        depth: 10,
        sampler: None,
    };
    // Majorant grid: `r³` supervoxels (0: one global majorant).
    let r = a.get(3).copied().unwrap_or(8);
    let mut scene = scene;
    if r > 0 {
        scene.medium = scene.medium.clone().with_majorant_grid(r);
        let mg = scene.medium.majorant.as_ref().unwrap();
        let gm = scene.medium.density.max();
        let mean = mg.dmax.iter().sum::<f64>() / mg.dmax.len() as f64 / gm;
        let mean_d = scene.medium.density.data.iter().sum::<f64>() / scene.medium.density.data.len() as f64 / gm;
        let empty = mg.dmax.iter().filter(|&&d| d == 0.0).count() as f64 / mg.dmax.len() as f64;
        eprintln!("majorant/global {mean:.3}, density/global {mean_d:.3}, empty {empty:.3}");
    }
    let theta = [1.0, 0.9, 0.92, 0.95];
    let t0 = std::time::Instant::now();
    let img = scene.image(&scene.forward(&theta, 0x70c));
    let secs = t0.elapsed().as_secs_f64();
    println!("grid {r}³: {w}×{h} × {spp} spp in {secs:.1} s ({:.2} Mpaths/s)", (w * h * spp) as f64 / secs / 1e6);
    let px = tonemap(&img, 1.4);
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/look-volume.png");
    image::RgbImage::from_raw(w as u32, h as u32, px).unwrap().save(&out).unwrap();
    println!("wrote {}", out.display());
}
