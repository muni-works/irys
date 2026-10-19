//! The data engine: `neural_data --out out/neural/data --prefix a --seed 0
//! --n 20000 --shard 500 [--domain b] [--spp 32] [--preview 8]`.
//! Renders seeded tabletop scenes with irys (OpenPBR on the device), labels
//! them through an irys TLAS, writes shards, and reports renders per second.

use std::path::PathBuf;

use irys_recon_net::data::{Sample, generate, write_shard};
use irys_recon_net::net::{H, W};
use irys_recon_net::render::TABLE;
use irys_recon_net::scene::Domain;

fn arg<T: std::str::FromStr>(a: &[String], k: &str, d: T) -> T {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let out = PathBuf::from(arg(&a, "--out", "out/neural/data".to_string()));
    let prefix: String = arg(&a, "--prefix", "a".to_string());
    let seed0: u64 = arg(&a, "--seed", 0);
    let n: usize = arg(&a, "--n", 1000);
    let shard: usize = arg(&a, "--shard", 500);
    let spp: u32 = arg(&a, "--spp", 32);
    let preview: usize = arg(&a, "--preview", 0);
    let domain = if arg(&a, "--domain", "a".to_string()) == "b" { Domain::B } else { Domain::A };
    // v3: also keep the photo at `hi ×` (the compare image's target).
    let hi: usize = arg(&a, "--hi", 0);
    std::fs::create_dir_all(&out).unwrap();
    let ctx = irys_diff::gpu::context().expect("gpu");
    let t0 = std::time::Instant::now();
    let mut done = 0;
    let mut objects = 0;
    let mut k = 0;
    while done < n {
        let m = shard.min(n - done);
        let s0 = seed0 + done as u64;
        let samples = if hi > 0 {
            let both = irys_recon_net::data::generate_hi(ctx, s0, m, domain, spp, hi);
            let dir = out.join(format!("{prefix}-hi"));
            std::fs::create_dir_all(&dir).unwrap();
            for (s, img) in &both {
                img.save(dir.join(format!("{}.png", s.scene.seed))).unwrap();
            }
            both.into_iter().map(|x| x.0).collect()
        } else {
            generate(ctx, s0, m, domain, spp)
        };
        objects += samples.iter().map(|s| s.scene.objects.len()).sum::<usize>();
        if k == 0 {
            for s in samples.iter().take(preview) {
                save_preview(&out, s);
            }
        }
        write_shard(&out.join(format!("{prefix}-{s0:08}")), &samples).unwrap();
        done += m;
        k += 1;
        let dt = t0.elapsed().as_secs_f64();
        println!("{done}/{n} scenes, {objects} objects, {:.1} s, {:.2} scenes/s", dt, done as f64 / dt);
    }
}

fn save_preview(dir: &std::path::Path, s: &Sample) {
    let img = image::RgbImage::from_raw(W as u32, H as u32, s.rgb.clone()).unwrap();
    img.save(dir.join(format!("preview-{}.png", s.scene.seed))).unwrap();
    let pal = |i: u16| -> [u8; 3] {
        if i == TABLE {
            return [20, 20, 20];
        }
        let h = (i as u32).wrapping_mul(2654435761);
        [(h >> 8) as u8 | 64, (h >> 16) as u8 | 64, (h >> 24) as u8 | 64]
    };
    let m: Vec<u8> = s.inst.iter().flat_map(|&i| pal(i)).collect();
    image::RgbImage::from_raw(W as u32, H as u32, m).unwrap().save(dir.join(format!("preview-{}-inst.png", s.scene.seed))).unwrap();
    std::fs::write(dir.join(format!("preview-{}.txt", s.scene.seed)), &s.program).unwrap();
}
