//! Phase 5b: a Blender export rendered with its OpenPBR materials on the
//! GPU (`look::scene_gpu`).
//!
//! ```text
//! cargo run --release -p irys-diff --features gpu,assets --example blend_look -- <export dir> [options]
//!   --width W        image width (height keeps the export's aspect)
//!   --spp N          samples per pixel (default 64)
//!   --bounces B      Cycles' max_bounces (default 7)
//!   --clamp C        indirect sample clamp (0: off)
//!   --seed S
//!   --reps K         render K times, report the fastest (shared machine)
//!   --png FILE       write the frame (Standard view: sRGB, clipped)
//!   --f32 FILE       write the frame as raw little-endian f32 RGB
//!   --levels a,b,..  bench: render each spp level (fastest of --reps) into
//!                    --bench DIR as irys_<spp>.f32 + irys.json
//!   --sequence CAMS  render a camera sequence (JSON list of {eye, forward,
//!                    right, up, tan_half}) into --outdir as JPEG q92 frames,
//!                    with frames.json: per-frame mean spectrum and time
//! ```

use std::io::Write;
use std::path::{Path, PathBuf};

use irys_diff::look::scene::{self, Camera};
use irys_diff::look::scene_gpu::{BINS, GpuLook, Opts};

fn srgb8(x: f32) -> u8 {
    let x = x.clamp(0.0, 1.0);
    let s = if x <= 0.003_130_8 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}

/// Blender's Standard view transform: sRGB OETF, clipped.
pub fn to_srgb8(rgb: &[f32]) -> Vec<u8> {
    rgb.iter().map(|&x| srgb8(x)).collect()
}

fn cam_of(v: &serde_json::Value) -> Camera {
    let a = |k: &str| -> [f64; 3] {
        let x = &v[k];
        [x[0].as_f64().unwrap(), x[1].as_f64().unwrap(), x[2].as_f64().unwrap()]
    };
    Camera { eye: a("eye"), forward: a("forward"), right: a("right"), up: a("up"), tan_half: v["tan_half"].as_f64().unwrap() }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("export dir"));
    let rest: Vec<String> = args.collect();
    let opt = |k: &str| rest.iter().position(|a| a == k).map(|i| rest[i + 1].clone());
    let width = opt("--width").map(|s| s.parse().expect("width"));
    let mut o = Opts {
        spp: opt("--spp").map_or(64, |s| s.parse().unwrap()),
        max_bounces: opt("--bounces").map_or(7, |s| s.parse().unwrap()),
        clamp: opt("--clamp").map_or(0.0, |s| s.parse().unwrap()),
        seed: opt("--seed").map_or(1, |s| s.parse().unwrap()),
        rr_from: opt("--rr").map_or(u32::MAX, |s| s.parse().unwrap()),
        debug: opt("--debug").map_or(0, |s| s.parse().unwrap()),
        per_lane: rest.iter().any(|a| a == "--per-lane"),
    };
    let reps: usize = opt("--reps").map_or(1, |s| s.parse().unwrap());
    let t0 = std::time::Instant::now();
    let sc = scene::load(&dir, width).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let (tu, tp) = sc.triangles();
    eprintln!(
        "loaded {} in {:.1} s: {} meshes, {} instances, {} / {} triangles, {} materials, {} textures, {}×{}",
        dir.display(),
        t0.elapsed().as_secs_f64(),
        sc.meshes.len(),
        sc.instances.len(),
        tu,
        tp,
        sc.materials.len(),
        sc.textures.len(),
        sc.size.0,
        sc.size.1
    );
    if rest.iter().any(|a| a == "--stats") {
        for (mi, m) in sc.materials.iter().enumerate() {
            let (mut n, mut t) = (0, 0);
            let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
            for i in &sc.instances {
                let me = &sc.meshes[i.mesh];
                if me.material != mi {
                    continue;
                }
                n += 1;
                t += me.indices.len() / 3;
                for p in me.positions.iter().step_by(7) {
                    for r in 0..3 {
                        let w = i.xf[r][0] * p[0] + i.xf[r][1] * p[1] + i.xf[r][2] * p[2] + i.xf[r][3];
                        lo[r] = lo[r].min(w);
                        hi[r] = hi[r].max(w);
                    }
                }
            }
            eprintln!("{mi:3} {:28} inst {n:6} tris {t:9} z {:.2}..{:.2} flags thin={} cut={} op={:.2} tw={:.2}", m.name, lo[2], hi[2], m.thin_walled, m.cutout, m.opacity, m.pbr.transmission_weight);
        }
    }
    let ctx = irys_gpu::GpuContext::init_blocking().expect("gpu");
    let t1 = std::time::Instant::now();
    let gpu = GpuLook::new(ctx, &sc).expect("ray queries");
    eprintln!(
        "uploaded in {:.1} s: {} emitter / {} portal triangles, light shares {:?}",
        t1.elapsed().as_secs_f64(),
        gpu.emitters.0,
        gpu.emitters.1,
        gpu.shares
    );
    let (w, h) = sc.size;
    if let Some(levels) = opt("--levels") {
        let bench = PathBuf::from(opt("--bench").expect("--bench DIR"));
        std::fs::create_dir_all(&bench).unwrap();
        let mut rows = Vec::new();
        // warm the pipeline
        let _ = gpu.render(&sc.camera, &Opts { spp: 1, ..o });
        for l in levels.split(',') {
            let spp: u32 = l.parse().unwrap();
            let mut best = f64::MAX;
            let mut img = Vec::new();
            for r in 0..reps {
                let f = gpu.render(&sc.camera, &Opts { spp, seed: o.seed + 1000 * spp as u64 + r as u64 * 0, ..o });
                if f.seconds < best {
                    best = f.seconds;
                }
                img = f.rgb;
            }
            let name = if l == levels.split(',').last().unwrap() && opt("--ref").is_some() {
                "irys_ref.f32".to_string()
            } else {
                format!("irys_{spp}.f32")
            };
            write_f32(&bench.join(&name), &img);
            eprintln!("irys {spp:6} spp {best:9.4} s");
            rows.push(serde_json::json!({"spp": spp, "seconds": best, "file": name}));
        }
        let meta = serde_json::json!({"size": [w, h], "rows": rows, "bounces": o.max_bounces, "clamp": o.clamp});
        let p = bench.join(opt("--meta").unwrap_or_else(|| "irys.json".into()));
        std::fs::write(&p, serde_json::to_string_pretty(&meta).unwrap()).unwrap();
        return;
    }
    if let Some(cams) = opt("--sequence") {
        let outdir = PathBuf::from(opt("--outdir").expect("--outdir"));
        std::fs::create_dir_all(&outdir).unwrap();
        let list: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cams).unwrap()).unwrap();
        let cams: Vec<Camera> = list.as_array().unwrap().iter().map(cam_of).collect();
        let mut frames = Vec::new();
        let start = opt("--start").map_or(0, |s| s.parse().unwrap());
        for (k, c) in cams.iter().enumerate().skip(start) {
            let f = gpu.render(c, &Opts { seed: o.seed + k as u64, ..o });
            let px = to_srgb8(&f.rgb);
            let path = outdir.join(format!("{k:04}.jpg"));
            let file = std::fs::File::create(&path).unwrap();
            let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::BufWriter::new(file), 92);
            enc.encode(&px, w as u32, h as u32, image::ExtendedColorType::Rgb8).unwrap();
            eprintln!("frame {k:4} {:.2} s", f.seconds);
            frames.push(serde_json::json!({"frame": k, "file": path.file_name().unwrap().to_str(), "seconds": f.seconds,
                "spectrum": f.spectrum}));
            // keep progress on disk
            let meta = serde_json::json!({"size": [w, h], "spp": o.spp, "bounces": o.max_bounces, "bins": BINS,
                "bins_nm": [380, 780], "frames": frames});
            std::fs::write(outdir.join("frames.json"), serde_json::to_string(&meta).unwrap()).unwrap();
        }
        return;
    }
    let mut best = None::<irys_diff::look::scene_gpu::Frame>;
    for r in 0..reps {
        o.seed += r as u64 * 0;
        let f = gpu.render(&sc.camera, &o);
        eprintln!("render {r}: {} spp in {:.2} s ({:.1} Mpaths/s)", o.spp, f.seconds, (w * h) as f64 * o.spp as f64 / f.seconds / 1e6);
        if best.as_ref().is_none_or(|b| f.seconds < b.seconds) {
            best = Some(f);
        }
    }
    let f = best.unwrap();
    let mean: f64 = f.rgb.iter().map(|&x| x as f64).sum::<f64>() / f.rgb.len() as f64;
    println!(
        "{{\"spp\": {}, \"seconds\": {:.3}, \"mean\": {:.6}, \"size\": [{w}, {h}], \"bounces\": {}}}",
        o.spp, f.seconds, mean, o.max_bounces
    );
    if let Some(p) = opt("--png") {
        let px = to_srgb8(&f.rgb);
        image::save_buffer(&p, &px, w as u32, h as u32, image::ColorType::Rgb8).unwrap();
    }
    if let Some(p) = opt("--f32") {
        write_f32(Path::new(&p), &f.rgb);
    }
    if let Some(p) = opt("--spectrum") {
        let mut s = std::fs::File::create(p).unwrap();
        writeln!(s, "{}", serde_json::to_string(&f.spectrum).unwrap()).unwrap();
    }
}

fn write_f32(p: &Path, img: &[f32]) {
    let b: Vec<u8> = img.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(p, b).unwrap();
}
