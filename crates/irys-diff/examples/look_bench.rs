//! Phase 5b: the irys side of the true-material benchmark, on the OpenPBR
//! scene kernel (`look::scene_gpu`), in the files `irys-bench report` reads.
//!
//! ```text
//! cargo run --release -p irys-diff --features gpu,assets --example look_bench -- <export dir> <bench dir>
//!     [--width 480] [--bounces 7] [--ref-spp 32768] [--max-spp 1024] [--reps 2]
//!     [--variant openpbr] [--clamp C] [--samp bits] [--prof bits] [--regen K] [--guide mix]
//! ```
//!
//! Writes `irys_ref.f32` (+ `irys_ref.json`) once, then for each spp level
//! 1, 2, 4, … `irys_<variant>_<spp>.f32` and `irys_<variant>.json` (spp,
//! fastest-of-`reps` seconds, relMSE against irys's own reference): phase
//! 3's protocol. The reference is rendered in chunks of 1024 spp with
//! distinct seeds and averaged.

use std::path::{Path, PathBuf};

use irys_diff::look::scene;
use irys_diff::look::scene_gpu::{GpuLook, Opts};

fn opt<T: std::str::FromStr>(args: &[String], k: &str, d: T) -> T {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(d)
}

fn write_f32(p: &Path, img: &[f32]) {
    let b: Vec<u8> = img.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(p, b).unwrap();
}

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// `irys_bench::relmse`: mean of `(x − r)² / (r² + 1e-2)`.
fn relmse(x: &[f32], r: &[f32]) -> f64 {
    x.iter().zip(r).map(|(&a, &b)| {
        let (a, b) = (a as f64, b as f64);
        (a - b) * (a - b) / (b * b + 1e-2)
    }).sum::<f64>() / x.len() as f64
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let export = PathBuf::from(&args[0]);
    let dir = PathBuf::from(&args[1]);
    std::fs::create_dir_all(&dir).unwrap();
    let width: usize = opt(&args, "--width", 480);
    let bounces: u32 = opt(&args, "--bounces", 7);
    let ref_spp: u32 = opt(&args, "--ref-spp", 32768);
    let max_spp: u32 = opt(&args, "--max-spp", 1024);
    let reps: usize = opt(&args, "--reps", 2);
    let clamp: f32 = opt(&args, "--clamp", 0.0);
    let variant: String = opt(&args, "--variant", "openpbr".to_string());
    let samp: u32 = opt(&args, "--samp", irys_diff::look::scene_gpu::samp::ALL);
    let prof: u32 = opt(&args, "--prof", 0);
    let regen: u32 = opt(&args, "--regen", 16);
    let guide: f32 = opt(&args, "--guide", 0.0);
    let seed_base: u64 = opt(&args, "--seed-base", 1000);
    let sc = scene::load(&export, Some(width)).unwrap();
    let (w, h) = sc.size;
    let ctx = irys_gpu::GpuContext::init_blocking().expect("gpu");
    let g = GpuLook::new(ctx, &sc).expect("ray queries");
    let base = Opts { spp: 1, max_bounces: bounces, clamp, samp, prof, regen, guide, ..Opts::default() };
    let _ = g.render(&sc.camera, &base); // warm
    let ref_path = dir.join("irys_ref.f32");
    if !ref_path.exists() {
        let chunk = 1024.min(ref_spp);
        let n = ref_spp / chunk;
        let mut acc = vec![0.0f64; w * h * 3];
        let t = std::time::Instant::now();
        for k in 0..n {
            let f = g.render(&sc.camera, &Opts { spp: chunk, seed: (1 << 20) + k as u64, ..base });
            for (a, x) in acc.iter_mut().zip(&f.rgb) {
                *a += *x as f64 / n as f64;
            }
        }
        let img: Vec<f32> = acc.iter().map(|&x| x as f32).collect();
        write_f32(&ref_path, &img);
        let secs = t.elapsed().as_secs_f64();
        std::fs::write(
            dir.join("irys_ref.json"),
            serde_json::to_string(&serde_json::json!({"spp": ref_spp, "seconds": secs, "clamp": clamp})).unwrap(),
        )
        .unwrap();
        eprintln!("irys reference {ref_spp} spp in {secs:.1} s");
    }
    let reference = read_f32(&ref_path);
    let mut rows = Vec::new();
    let mut spp = 1;
    while spp <= max_spp {
        let mut best = f64::MAX;
        let mut img = Vec::new();
        for r in 0..reps {
            let f = g.render(&sc.camera, &Opts { spp, seed: seed_base + spp as u64 * 7919 + r as u64 * 0, ..base });
            best = best.min(f.seconds);
            img = f.rgb;
        }
        let e = relmse(&img, &reference);
        eprintln!("irys {variant}: {spp:5} spp {best:8.4} s relMSE {e:.5}");
        write_f32(&dir.join(format!("irys_{variant}_{spp}.f32")), &img);
        rows.push(serde_json::json!({"spp": spp, "seconds": best, "relmse": e}));
        spp *= 2;
    }
    std::fs::write(
        dir.join(format!("irys_{variant}.json")),
        serde_json::to_string_pretty(&serde_json::json!({"variant": variant, "size": [w, h], "bounces": bounces,
            "clamp": clamp, "rows": rows}))
        .unwrap(),
    )
    .unwrap();
}
