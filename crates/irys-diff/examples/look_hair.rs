//! Phase 4: `docs/look-hair.png` — three procedural haircuts on spheres
//! (black, brown, blonde by melanin alone: eumelanin 8, 1.3, 0.3 with a
//! little pheomelanin), Chiang 2016 fibres traced as round cylinders.
//!
//! `cargo run --release -p irys-diff --features assets --example look_hair -- [w] [h] [spp]`

use irys_core::{Bvh, Point3, Vec3};
use irys_diff::look::hair::{Fiber, HairScene, Strands};
use irys_diff::prb::Rng;
use irys_diff::skin::render::{Light, tonemap};

/// A haircut on the unit sphere at `c`: roots over the crown, each strand
/// leaving along the normal, combed back and pulled down by gravity.
pub fn haircut(s: &mut Strands, c: Point3, n: usize, seed: u64) {
    let mut rng = Rng::new(seed, 0);
    let mut placed = 0;
    while placed < n {
        let z = 2.0 * rng.uniform() - 1.0;
        let p = 2.0 * std::f64::consts::PI * rng.uniform();
        let r = (1.0 - z * z).sqrt();
        let nrm = Vec3::new(r * p.cos(), r * p.sin(), z);
        // Hairline: the crown and the back, not the face (−y).
        if z < 0.05 || (nrm.y < -0.35 && z < 0.55) {
            continue;
        }
        placed += 1;
        let len = 0.75 + 0.35 * rng.uniform();
        let seg = 14;
        let mut x = c + nrm * 1.0;
        let mut dir = (nrm + Vec3::new(0.0, 0.6, 0.0)).normalize();
        let mut pts = vec![x];
        let phase = 6.28 * rng.uniform();
        for k in 0..seg {
            let t = k as f64 / seg as f64;
            let gravity = Vec3::new(0.0, 0.0, -1.0) * (0.25 + 0.9 * t);
            let wave = Vec3::new((9.0 * t + phase).sin(), 0.0, (7.0 * t + phase).cos()) * 0.08;
            dir = (dir + gravity * 0.35 + wave).normalize();
            let mut nx = x + dir * (len / seg as f64);
            // Keep it off the scalp.
            let off = nx - c;
            if off.norm() < 1.01 {
                nx = c + off.normalize() * 1.01;
            }
            x = nx;
            pts.push(x);
        }
        s.push(&pts, 0.0035);
    }
}

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let (w, h, spp) = (a.first().copied().unwrap_or(840), a.get(1).copied().unwrap_or(360), a.get(2).copied().unwrap_or(64));
    let heads = [(-2.3, [8.0, 0.1]), (0.0, [1.3, 0.2]), (2.3, [0.3, 0.05])];
    let mut img = vec![0.0; w * h * 3];
    let mut total = 0.0;
    // One scene per head (its own melanin); each renders the full frame
    // with only its strands, and the frame keeps the pixels nearest it.
    let third = w / 3;
    for (k, (x, theta)) in heads.iter().enumerate() {
        let mut s = Strands::default();
        let c = Point3::new(*x, 0.0, 0.0);
        haircut(&mut s, c, 9000, 7 + k as u64);
        let segs = s.segs.len();
        let scene = HairScene {
            bvh: Bvh::build(s),
            fiber: Fiber::default(),
            scalp: Some((c, 0.995, [0.35, 0.22, 0.17])),
            lights: vec![
                Light::facing(Point3::new(x - 2.4, 3.6, 3.0), c, 2.0, 2.0, 7.0, [1.0, 0.96, 0.9]),
                Light::facing(Point3::new(x + 1.0, -3.0, 3.6), c, 3.0, 1.0, 6.0, [1.0; 3]),
                Light::facing(Point3::new(x + 3.5, 2.0, 0.0), c, 3.0, 3.0, 0.8, [0.9, 0.95, 1.0]),
            ],
            env: 0.015,
            eye: Point3::new(x + 2.6, 7.4, 1.6),
            target: Point3::new(*x, 0.0, -0.25),
            vfov: 30.0,
            size: (third, h),
            spp,
            depth: 24,
            sampler: None,
        };
        let t0 = std::time::Instant::now();
        let part = scene.image(&scene.forward(theta, 0x4a1 + k as u64));
        let secs = t0.elapsed().as_secs_f64();
        total += secs;
        println!("head {k}: {segs} segments, {third}×{h} × {spp} spp in {secs:.1} s");
        for y in 0..h {
            for xx in 0..third {
                for ch in 0..3 {
                    img[(y * w + k * third + xx) * 3 + ch] = part[(y * third + xx) * 3 + ch];
                }
            }
        }
    }
    println!("total {total:.1} s");
    let px = tonemap(&img, 2.0);
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/look-hair.png");
    image::RgbImage::from_raw(w as u32, h as u32, px).unwrap().save(&out).unwrap();
    println!("wrote {}", out.display());
}
