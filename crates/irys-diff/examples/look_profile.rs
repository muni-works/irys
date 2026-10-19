//! Speed round 4: where the OpenPBR scene kernel's time goes.
//!
//! ```text
//! cargo run --release -p irys-diff --features gpu,assets --example look_profile -- <export dir>
//!     [--width 480] [--spp 64] [--reps 5] [--bounces 7]
//! ```
//!
//! Prints the device counters (`prof::STATS`: rays, traversal candidates,
//! SIMD occupancy and material classes per iteration) and the time per
//! sample of ablations (fastest of `reps`, interleaved): full, bounces
//! capped, no NEE, NEE without shadow rays, Lambertian materials, no alpha
//! test. Ablations change the picture; they only apportion time.

use std::path::PathBuf;

use irys_diff::look::scene;
use irys_diff::look::scene_gpu::{GpuLook, Opts, prof};

fn opt<T: std::str::FromStr>(args: &[String], k: &str, d: T) -> T {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(d)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let export = PathBuf::from(&args[0]);
    let width: usize = opt(&args, "--width", 480);
    let spp: u32 = opt(&args, "--spp", 64);
    let reps: usize = opt(&args, "--reps", 5);
    let bounces: u32 = opt(&args, "--bounces", 7);
    let clamp: f32 = opt(&args, "--clamp", 0.0);
    let sc = scene::load(&export, Some(width)).unwrap();
    let (w, h) = sc.size;
    let ctx = irys_gpu::GpuContext::init_blocking().expect("gpu");
    let g = GpuLook::new(ctx, &sc).expect("ray queries");
    let base = Opts { spp, max_bounces: bounces, clamp, ..Opts::default() };
    let _ = g.render(&sc.camera, &Opts { spp: 2, ..base });
    println!("{} {w}x{h}, emitters {:?}, shares {:?}", export.display(), g.emitters, g.shares);

    let f = g.render(&sc.camera, &Opts { spp: 8, prof: prof::STATS, ..base });
    let s = f.stats.unwrap();
    let px = (w * h * 8) as f64;
    println!("per path (8 spp): camera {:.2} cont {:.2} shadow {:.2} rays; candidates closest {:.2} shadow {:.2}; vertices {:.2} (slow {:.3})",
        s.rays[0] as f64 / px, s.rays[1] as f64 / px, s.rays[2] as f64 / px,
        s.candidates[0] as f64 / px, s.candidates[1] as f64 / px, s.vertices as f64 / px, s.slow as f64 / px);
    println!("iter  alive%  simd-occupancy  classes/simd");
    for i in 0..8 {
        if s.simd_alive[i] == 0 { continue; }
        println!("{i:4}  {:6.1}  {:14.3}  {:12.2}", 100.0 * s.alive[i] as f64 / px,
            s.alive[i] as f64 / (32.0 * s.simd_alive[i] as f64), s.simd_classes[i] as f64 / s.simd_alive[i] as f64);
    }
    let names = ["T", "M", "C", "F"];
    let mut cls: Vec<(usize, u64)> = s.class.iter().copied().enumerate().filter(|c| c.1 > 0).collect();
    cls.sort_by_key(|c| std::cmp::Reverse(c.1));
    let tot: u64 = s.class.iter().sum();
    let line: Vec<String> = cls.iter().map(|&(c, n)| {
        let mut k: String = (0..4).filter(|b| c >> b & 1 == 1).map(|b| names[b]).collect();
        if k.is_empty() { k = "D".into(); }
        if c & 16 != 0 { k += "/slow"; }
        format!("{k} {:.1}%", 100.0 * n as f64 / tot as f64)
    }).collect();
    println!("classes: {}", line.join(", "));

    let variants: Vec<(&str, Opts)> = vec![
        ("full", base),
        ("bounces 0", Opts { max_bounces: 0, ..base }),
        ("bounces 1", Opts { max_bounces: 1, ..base }),
        ("bounces 3", Opts { max_bounces: 3, ..base }),
        ("no NEE", Opts { prof: prof::NO_NEE, ..base }),
        ("no shadow", Opts { prof: prof::NO_SHADOW, ..base }),
        ("lambert", Opts { prof: prof::LAMBERT, ..base }),
        ("no alpha test", Opts { prof: prof::OPAQUE, ..base }),
        ("stats on", Opts { prof: prof::STATS, ..base }),
    ];
    let mut best = vec![f64::MAX; variants.len()];
    for _ in 0..reps {
        for (i, (_, o)) in variants.iter().enumerate() {
            best[i] = best[i].min(g.render(&sc.camera, o).seconds);
        }
    }
    for (i, (n, _)) in variants.iter().enumerate() {
        println!("{n:14} {:7.3} ms/spp  ({:5.1} %)", 1e3 * best[i] / spp as f64, 100.0 * best[i] / best[0]);
    }
}
