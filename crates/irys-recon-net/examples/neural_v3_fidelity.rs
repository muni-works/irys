//! v3, piece 1: how faithful is the compare image? Render held-out scenes'
//! TRUE programs (true geometry, true camera) through each compare path and
//! measure PSNR against the photo:
//!
//! - v2: the irys TLAS cast with the linear appearance (`c₀ + c·n`), the
//!   photo linearised with gamma 1;
//! - v3, true appearance: the OpenPBR scene kernel with the scene's own
//!   materials and lights, only exposure fitted (the renderer's ceiling);
//! - v3, fitted appearance: [`faithful::fit_appearance`] from grey (what
//!   the search and the teacher see).
//!
//! `neural_v3_fidelity --set v3-a --n 30 --size 256` writes
//! `out/neural/v3/fidelity-<size>.json` and a few strips.

use std::path::PathBuf;

use irys_recon_net::data::{hi_path, read_dir};
use irys_recon_net::faithful::{self, Appearance, Compare3, FitOpts, display, linearise, local_meshes, psnr};
use irys_recon_net::render::{Caster, Compare, appearance, resample_rgb8, world_meshes};

fn arg<T: std::str::FromStr>(a: &[String], k: &str, d: T) -> T {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn blur_px(img: &[[f64; 3]], size: (usize, usize), s: f64) -> Vec<[f64; 3]> {
    if s <= 0.0 {
        return img.to_vec();
    }
    let flat: Vec<f64> = img.iter().flat_map(|p| p.iter().copied()).collect();
    irys_recon::fit::blur(&flat, size, 3, s).chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

fn to8(img: &[[f64; 3]], g: f64) -> Vec<u8> {
    img.iter().flat_map(|p| p.map(|v| (display(v, g) * 255.0 + 0.5).clamp(0.0, 255.0) as u8)).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = PathBuf::from(arg(&a, "--data", "out/neural/data".to_string()));
    let set: String = arg(&a, "--set", "v3-a".to_string());
    let n: usize = arg(&a, "--n", 30);
    let sw: usize = arg(&a, "--size", 256);
    let spp: u32 = arg(&a, "--spp", 16);
    let out = PathBuf::from(arg(&a, "--out", "out/neural/v3".to_string()));
    std::fs::create_dir_all(&out).unwrap();
    let size = (sw, sw * 3 / 4);
    let ctx = irys_diff::gpu::context().expect("gpu");
    let samples: Vec<_> = read_dir(&data, &set).into_iter().take(n).collect();
    let mut rows = Vec::new();
    let (mut s2, mut s3t, mut s3f, mut s2hi) = (0.0, 0.0, 0.0, 0.0);
    for (si, s) in samples.iter().enumerate() {
        let hi = image::open(hi_path(&data, &set, s.scene.seed)).unwrap().to_rgb8();
        let target_lin = resample_rgb8(&hi, size);
        let rgb8: Vec<u8> = target_lin.iter().flat_map(|p| p.map(irys_recon::render::srgb8)).collect();
        let cam = s.scene.cam;
        let objs = &s.scene.objects;
        // v2's compare at the net's size and at this size.
        let v2 = |sz: (usize, usize), photo8: &[u8]| {
            let caster = Caster::new(&world_meshes(objs));
            let cast = Compare { size: sz, ss: 3 }.cast(&caster, &cam, None);
            let t = linearise(photo8, 1.0);
            let m = appearance(&cast, &t);
            psnr(&m, photo8, 1.0)
        };
        let p2 = v2((128, 96), &s.rgb);
        let p2hi = v2(size, &rgb8);
        // v3, true appearance.
        let t0 = std::time::Instant::now();
        let truth = irys_recon_net::render::render_linear(ctx, &s.scene, &world_meshes(objs), size, spp).unwrap();
        let t_true = t0.elapsed().as_secs_f64();
        let g = s.scene.post[1];
        let tl = linearise(&rgb8, g);
        // Exposure: one scale, on the channels the sensor did not clip.
        let sat = faithful::SAT;
        let k = tl.iter().zip(&truth).map(|(t, m)| (0..3).filter(|&c| t[c] < sat).map(|c| t[c] * m[c]).sum::<f64>()).sum::<f64>()
            / tl.iter().zip(&truth).map(|(t, m)| (0..3).filter(|&c| t[c] < sat).map(|c| m[c] * m[c]).sum::<f64>()).sum::<f64>();
        let truth: Vec<[f64; 3]> = truth.iter().map(|m| m.map(|v| v * k)).collect();
        let blurs = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.6];
        let (p3t, bt) = blurs.iter().map(|&b| (psnr(&blur_px(&truth, size, b), &rgb8, g), b)).fold((f64::MIN, 0.0), |x, y| if y.0 > x.0 { y } else { x });
        // v3, fitted appearance.
        let t1 = std::time::Instant::now();
        let mut app = Appearance::grey(objs.len());
        let meshes = local_meshes(objs);
        let mut cmp = Compare3::new(ctx, objs, &meshes, &app, &cam, size).unwrap();
        let t_build = t1.elapsed().as_secs_f64();
        let rep = faithful::fit_appearance(&mut cmp, &cam, &rgb8, &mut app, &FitOpts { spp: 8, ..FitOpts::default() });
        let t2 = std::time::Instant::now();
        let m = cmp.render(&cam, spp, 99);
        let t_render = t2.elapsed().as_secs_f64();
        let (p3f, bf) = blurs.iter().map(|&b| (psnr(&blur_px(&m, size, b), &rgb8, app.gamma), b)).fold((f64::MIN, 0.0), |x, y| if y.0 > x.0 { y } else { x });
        // A pose move through the TLAS only.
        let present = vec![true; objs.len()];
        let mut moved = objs.clone();
        for o in &mut moved {
            o.pose[0] += 1.0;
        }
        let t3 = std::time::Instant::now();
        cmp.set_poses(&moved, &present);
        let t_move = t3.elapsed().as_secs_f64();
        cmp.set_poses(objs, &present);
        s2 += p2;
        s2hi += p2hi;
        s3t += p3t;
        s3f += p3f;
        println!(
            "[{}/{}] seed {}  v2@128 {:.2} dB  v2@{} {:.2}  v3 true {:.2} (blur {bt})  v3 fitted {:.2} (blur {bf}, γ {:.2} vs {:.2}, {} renders {:.1}s)  build {:.3}s render {:.3}s move {:.4}s",
            si + 1,
            samples.len(),
            s.scene.seed,
            p2,
            sw,
            p2hi,
            p3t,
            p3f,
            app.gamma,
            g,
            rep.renders,
            rep.seconds,
            t_build,
            t_render,
            t_move
        );
        if si < 4 {
            let row: Vec<u8> = rgb8.clone();
            let im = |v: Vec<u8>| image::RgbImage::from_raw(size.0 as u32, size.1 as u32, v).unwrap();
            let mut strip = image::RgbImage::new(size.0 as u32 * 3, size.1 as u32);
            image::imageops::replace(&mut strip, &im(row), 0, 0);
            image::imageops::replace(&mut strip, &im(to8(&blur_px(&truth, size, bt), g)), size.0 as i64, 0);
            image::imageops::replace(&mut strip, &im(to8(&blur_px(&m, size, bf), app.gamma)), 2 * size.0 as i64, 0);
            strip.save(out.join(format!("fidelity-{sw}-{}.png", s.scene.seed))).unwrap();
        }
        rows.push(serde_json::json!({"seed": s.scene.seed, "v2_128": p2, "v2_hi": p2hi, "v3_true": p3t, "v3_fitted": p3f,
            "gamma_fit": app.gamma, "gamma_true": g, "fit_renders": rep.renders, "fit_seconds": rep.seconds,
            "build_s": t_build, "render_s": t_render, "move_s": t_move, "true_render_s": t_true}));
        let k = (si + 1) as f64;
        let summary = serde_json::json!({"size": size, "scenes": si + 1, "v2_128": s2 / k, "v2_hi": s2hi / k, "v3_true": s3t / k, "v3_fitted": s3f / k, "rows": rows});
        std::fs::write(out.join(format!("fidelity-{sw}.json")), serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    }
    let k = samples.len() as f64;
    println!("mean PSNR: v2@128 {:.2}  v2@{sw} {:.2}  v3 true {:.2}  v3 fitted {:.2}", s2 / k, s2hi / k, s3t / k, s3f / k);
}
