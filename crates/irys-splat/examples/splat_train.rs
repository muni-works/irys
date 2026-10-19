//! Fit a 3DGS map to posed photos on irys's GPU tier.
//!
//! ```text
//! cargo run --release -p irys-splat --example splat_train -- \
//!     --images DIR --poses poses.jsonl --camera camera_info.json --out map.ply \
//!     [--train <capture>.train/ (ipse CAPTURE.md export: sets --images/--poses/--camera)] [--with-depth]
//!     [--depth DIR] [--depth-scale 0.001] [--depth-min 0.1] [--depth-max 20] [--depth-weight 0.1]
//!     [--init points.ply | --seed-depth N | --random N] [--iters 30000] [--sh-degree 3]
//!     [--holdout 8] [--eval-every 1000] [--report DIR] [--irys-eval] [--seed S] [--max-splats N]
//!     [--scene-scale M] [--box M]
//! ```
//!
//! Every `--holdout`-th view (0: none) is held out and scored (PSNR / SSIM)
//! every `--eval-every` steps. `--init` takes a point PLY (`x y z`, optional
//! `red green blue` bytes) or a 3DGS PLY; `--seed-depth N` back-projects N
//! random valid depth pixels of the training views ("SfM-like" sparse
//! points); otherwise `--random N` (default 100 000) uniform in the box
//! around the cameras (half-width `--box`, default 2 × the scene scale).
//! `--scene-scale` overrides gsplat's camera-spread scale (which sets the
//! mean learning rate and the clone / split / prune sizes) — needed when
//! the cameras barely move, as in a turn-in-place sweep. `--report DIR` writes `log.csv`, `metrics.json` and
//! held-out `gt / render / error` PNGs; `--irys-eval` also scores the map as
//! irys's ray tracer draws it (`look::sensor`'s splat layer).

use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use irys_splat::data::{self, DepthSpec, View};
use irys_splat::loss;
use irys_splat::model::Gaussians;
use irys_splat::rng::Rng;
use irys_splat::train::{self, Config, Trainer};

fn read_points(path: &Path) -> Result<(Vec<[f32; 3]>, Vec<[f32; 3]>), String> {
    let f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut r = std::io::BufReader::new(f);
    let mut props = Vec::new();
    let (mut n, mut ascii) = (0usize, false);
    loop {
        let mut l = String::new();
        r.read_line(&mut l).map_err(|e| e.to_string())?;
        let t: Vec<&str> = l.split_whitespace().collect();
        match t.as_slice() {
            ["format", f, ..] => ascii = *f == "ascii",
            ["element", "vertex", c] => n = c.parse().map_err(|_| "bad count")?,
            ["property", ty, name] => props.push((ty.to_string(), name.to_string())),
            ["end_header"] => break,
            [] if l.is_empty() => return Err("no end_header".into()),
            _ => {}
        }
    }
    let size = |t: &str| match t {
        "char" | "uchar" | "int8" | "uint8" => 1,
        "short" | "ushort" | "int16" | "uint16" => 2,
        "double" | "float64" => 8,
        _ => 4,
    };
    let idx = |k: &str| props.iter().position(|p| p.1 == k);
    let (ix, iy, iz) = (idx("x").ok_or("no x")?, idx("y").ok_or("no y")?, idx("z").ok_or("no z")?);
    let rgb = [idx("red"), idx("green"), idx("blue")];
    let mut pts = Vec::with_capacity(n);
    let mut col = Vec::with_capacity(n);
    let mut text = String::new();
    if ascii {
        r.read_to_string(&mut text).map_err(|e| e.to_string())?;
    }
    let mut words = text.split_whitespace();
    for _ in 0..n {
        let mut vals = vec![0f64; props.len()];
        for (k, (ty, _)) in props.iter().enumerate() {
            vals[k] = if ascii {
                words.next().ok_or("short ascii")?.parse().map_err(|_| "bad number")?
            } else {
                let mut b = vec![0u8; size(ty)];
                r.read_exact(&mut b).map_err(|e| e.to_string())?;
                match ty.as_str() {
                    "float" | "float32" => f32::from_le_bytes(b[..4].try_into().unwrap()) as f64,
                    "double" | "float64" => f64::from_le_bytes(b[..8].try_into().unwrap()),
                    "uchar" | "uint8" => b[0] as f64,
                    "char" | "int8" => b[0] as i8 as f64,
                    "ushort" | "uint16" => u16::from_le_bytes([b[0], b[1]]) as f64,
                    "short" | "int16" => i16::from_le_bytes([b[0], b[1]]) as f64,
                    "uint" | "uint32" => u32::from_le_bytes(b[..4].try_into().unwrap()) as f64,
                    _ => i32::from_le_bytes(b[..4].try_into().unwrap()) as f64,
                }
            };
        }
        pts.push([vals[ix] as f32, vals[iy] as f32, vals[iz] as f32]);
        col.push(rgb.map(|c| c.map_or(0.5, |k| (vals[k] / 255.0) as f32)));
    }
    Ok((pts, col))
}

fn seed_from_depth(views: &[&View], n: usize, seed: u64) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let mut r = Rng::new(seed);
    let (mut p, mut c) = (Vec::new(), Vec::new());
    let mut tries = 0;
    while p.len() < n && tries < 50 * n {
        tries += 1;
        let v = views[r.below(views.len())];
        let Some(d) = &v.depth else { continue };
        let k = &v.cam;
        let px = r.below(k.width * k.height);
        let z = d[px] as f64;
        if !(z > 0.0) {
            continue;
        }
        let (u, vv) = ((px % k.width) as f64, (px / k.width) as f64);
        let pc = [(u - k.cx) / k.fx * z, (vv - k.cy) / k.fy * z, z];
        p.push(std::array::from_fn(|a| (k.pos[a] + (0..3).map(|j| k.rot[a][j] * pc[j]).sum::<f64>()) as f32));
        c.push(v.rgb[px]);
    }
    (p, c)
}

fn save_png(path: &Path, px: &[[f32; 3]], w: usize, h: usize) {
    let b: Vec<u8> = px.iter().flat_map(|c| c.map(|x| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)).collect();
    image::save_buffer(path, &b, w as u32, h as u32, image::ColorType::Rgb8).unwrap();
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let opt = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let flag = |k: &str| a.iter().any(|x| x == k);
    let need = |k: &str| PathBuf::from(opt(k).unwrap_or_else(|| panic!("{k} is required (see the header of examples/splat_train.rs)")));
    // `--train DIR` (an ipse `<capture>.train/`) supplies the three defaults
    let train_dir = opt("--train").map(PathBuf::from);
    let dflt = |k: &str, f: &str| opt(k).map(PathBuf::from).or_else(|| train_dir.as_ref().map(|d| d.join(f)));
    let images = dflt("--images", ".").unwrap_or_else(|| need("--images"));
    let poses = dflt("--poses", "poses.jsonl").unwrap_or_else(|| need("--poses"));
    let camera = dflt("--camera", "camera_info.json").unwrap_or_else(|| need("--camera"));
    let out = need("--out");
    // `--depth DIR`, or `--with-depth` for the files the pose lines name
    let depth = opt("--depth").map(PathBuf::from).or_else(|| flag("--with-depth").then(|| poses.parent().unwrap().to_path_buf()));
    let spec = DepthSpec {
        scale: opt("--depth-scale").map_or(1e-3, |s| s.parse().unwrap()),
        min: opt("--depth-min").map_or(0.1, |s| s.parse().unwrap()),
        max: opt("--depth-max").map_or(20.0, |s| s.parse().unwrap()),
    };
    let iters: usize = opt("--iters").map_or(30_000, |s| s.parse().unwrap());
    let holdout: usize = opt("--holdout").map_or(8, |s| s.parse().unwrap());
    let eval_every: usize = opt("--eval-every").map_or(1000, |s| s.parse().unwrap());
    let seed: u64 = opt("--seed").map_or(0, |s| s.parse().unwrap());
    let report = opt("--report").map(PathBuf::from);
    let t0 = Instant::now();
    let views = data::load(&images, &poses, &camera, depth.as_deref(), spec).unwrap_or_else(|e| panic!("{e}"));
    let (train_v, test_v): (Vec<&View>, Vec<&View>) = if holdout > 0 {
        let tr = views.iter().enumerate().filter(|(i, _)| i % holdout != 0).map(|x| x.1).collect();
        let te = views.iter().enumerate().filter(|(i, _)| i % holdout == 0).map(|x| x.1).collect();
        (tr, te)
    } else {
        (views.iter().collect(), Vec::new())
    };
    let cams: Vec<_> = train_v.iter().map(|v| v.cam).collect();
    let scale = opt("--scene-scale").map_or_else(|| train::scene_scale(&cams), |s| s.parse().unwrap());
    eprintln!("{} views ({} train, {} held out), {}×{}, scene scale {scale:.2} m, loaded in {:.1} s", views.len(), train_v.len(), test_v.len(), cams[0].width, cams[0].height, t0.elapsed().as_secs_f64());
    let g = if let Some(p) = opt("--init") {
        let c = irys_light::PlyCloud::load(Path::new(&p));
        match c {
            Ok(c) if !c.is_empty() => Gaussians::from_cloud(&c),
            _ => {
                let (p, c) = read_points(Path::new(&p)).unwrap_or_else(|e| panic!("{p}: {e}"));
                Gaussians::from_points(&p, &c)
            }
        }
    } else if let Some(n) = opt("--seed-depth") {
        let (p, c) = seed_from_depth(&train_v, n.parse().unwrap(), seed);
        assert!(!p.is_empty(), "--seed-depth needs --depth");
        Gaussians::from_points(&p, &c)
    } else {
        let n: usize = opt("--random").map_or(100_000, |s| s.parse().unwrap());
        let c: [f64; 3] = std::array::from_fn(|k| cams.iter().map(|q| q.pos[k]).sum::<f64>() / cams.len() as f64);
        let r = opt("--box").map_or(2.0 * scale, |s| s.parse().unwrap());
        train::init::random_box(n, c.map(|v| v - r), c.map(|v| v + r), seed)
    };
    eprintln!("init: {} splats", g.len());
    let mut cfg = Config::scaled(iters);
    cfg.seed = seed;
    cfg.sh_degree = opt("--sh-degree").map_or(3, |s| s.parse().unwrap());
    if let Some(m) = opt("--max-splats") {
        cfg.max_splats = m.parse().unwrap();
    }
    if depth.is_some() {
        cfg.weights.depth = opt("--depth-weight").map_or(0.0, |s| s.parse().unwrap());
    }
    let ctx = irys_diff::gpu::context().expect("gpu");
    let mut tr = Trainer::new(ctx, g, cfg.clone(), scale);
    let mut rng = Rng::new(seed ^ 77);
    let mut order: Vec<usize> = Vec::new();
    let mut log = String::from("iter,seconds,splats,train_loss,test_psnr,test_ssim\n");
    let mut eval = |tr: &mut Trainer| -> (f64, f64) {
        if test_v.is_empty() {
            return (f64::NAN, f64::NAN);
        }
        let deg = tr.degree();
        let mut s = (0.0, 0.0);
        for v in &test_v {
            let p = tr.render(&v.cam, deg);
            s.0 += loss::psnr(&p, &v.rgb);
            s.1 += loss::ssim(&p, &v.rgb, v.cam.width, v.cam.height);
        }
        (s.0 / test_v.len() as f64, s.1 / test_v.len() as f64)
    };
    let tt = Instant::now();
    let mut ms = [0.0f64; 5];
    let mut avg_loss = 0.0;
    for it in 0..iters {
        if order.is_empty() {
            order = (0..train_v.len()).collect();
            for i in (1..order.len()).rev() {
                order.swap(i, rng.below(i + 1));
            }
        }
        let v = train_v[order.pop().unwrap()];
        let st = tr.step(v, (it + 1) % eval_every == 0);
        for k in 0..5 {
            ms[k] += st.ms[k];
        }
        avg_loss = st.loss * eval_every as f64;
        if (it + 1) % eval_every == 0 || it + 1 == iters {
            let (p, s) = eval(&mut tr);
            let n = eval_every as f64;
            eprintln!(
                "it {:6}  {:7.1} s  splats {:8}  isects {:8}  loss {:.4}  held-out PSNR {p:.2} SSIM {s:.4}  ms/it project {:.1} bin {:.1} step {:.1} refine {:.1} - {:.0}",
                it + 1, tt.elapsed().as_secs_f64(), st.splats, st.isects, st.loss, ms[0] / n, ms[1] / n, ms[2] / n, ms[3] / n, ms[4] / n
            );
            log += &format!("{},{:.1},{},{:.5},{p:.3},{s:.4}\n", it + 1, tt.elapsed().as_secs_f64(), st.splats, avg_loss / n);
            ms = [0.0; 5];
            avg_loss = 0.0;
        }
    }
    tr.sync();
    let train_s = tt.elapsed().as_secs_f64();
    let deg = tr.degree();
    tr.g.write_ply(&out, deg).expect("write ply");
    eprintln!("wrote {} ({} splats, SH degree {deg}) after {train_s:.0} s", out.display(), tr.g.len());
    let Some(rep) = report else { return };
    std::fs::create_dir_all(&rep).unwrap();
    std::fs::write(rep.join("log.csv"), &log).unwrap();
    // render speed: the raster forward at a training camera, median of 20
    let cam = train_v[0].cam;
    let _ = tr.render(&cam, deg);
    let mut ts: Vec<f64> = (0..20)
        .map(|_| {
            let t = Instant::now();
            let _ = tr.render(&cam, deg);
            t.elapsed().as_secs_f64()
        })
        .collect();
    ts.sort_by(f64::total_cmp);
    let (mut ps, mut ss, mut per) = (0.0, 0.0, Vec::new());
    for (k, v) in test_v.iter().enumerate() {
        let p = tr.render(&v.cam, deg);
        let (q, s) = (loss::psnr(&p, &v.rgb), loss::ssim(&p, &v.rgb, v.cam.width, v.cam.height));
        ps += q;
        ss += s;
        per.push(serde_json::json!({"view": v.name, "psnr": q, "ssim": s}));
        let (w, h) = (v.cam.width, v.cam.height);
        let r: Vec<[f32; 3]> = p.iter().map(|c| [c[0] as f32, c[1] as f32, c[2] as f32]).collect();
        let e: Vec<[f32; 3]> = r.iter().zip(&v.rgb).map(|(a, b)| {
            let d = ((a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs()) / 3.0 * 4.0;
            [d, d, d]
        }).collect();
        save_png(&rep.join(format!("test{k:03}_gt.png")), &v.rgb, w, h);
        save_png(&rep.join(format!("test{k:03}_render.png")), &r, w, h);
        save_png(&rep.join(format!("test{k:03}_error.png")), &e, w, h);
    }
    let nt = test_v.len().max(1) as f64;
    let mut m = serde_json::json!({
        "views": views.len(), "train": train_v.len(), "test": test_v.len(),
        "iters": iters, "splats": tr.g.len(), "sh_degree": deg, "train_seconds": train_s,
        "test_psnr": ps / nt, "test_ssim": ss / nt, "per_view": per,
        "raster_render_ms_median": 1e3 * ts[10], "raster_fps": 1.0 / ts[10],
        "size": [cam.width, cam.height], "scene_scale": scale,
    });
    if flag("--irys-eval") && !test_v.is_empty() {
        use irys_diff::look::scene::LookScene;
        use irys_diff::look::scene_gpu::GpuLook;
        use irys_diff::look::sensor::{Sensor, aov};
        use irys_diff::look::splat_gpu;
        let cloud = tr.g.to_cloud(deg);
        let mut sc = LookScene::spheres(vec![], 0.0, (cam.width, cam.height));
        sc.env = None;
        let gl = GpuLook::with_splats(ctx, &sc, Some(&cloud)).expect("ray queries");
        let mut sen = Sensor::new(&gl, (cam.width, cam.height), 1);
        let (mut ip, mut is, mut ims) = (0.0, 0.0, Vec::new());
        for (k, v) in test_v.iter().enumerate() {
            let t = Instant::now();
            let a = sen.render(&[v.cam.camera()], aov::RGB);
            ims.push(t.elapsed().as_secs_f64());
            let p: Vec<[f64; 4]> = a.rgb.unwrap().iter().map(|q| [splat_gpu::srgb_enc(q[0] as f64), splat_gpu::srgb_enc(q[1] as f64), splat_gpu::srgb_enc(q[2] as f64), 0.0]).collect();
            ip += loss::psnr(&p, &v.rgb);
            is += loss::ssim(&p, &v.rgb, v.cam.width, v.cam.height);
            let r: Vec<[f32; 3]> = p.iter().map(|c| [c[0] as f32, c[1] as f32, c[2] as f32]).collect();
            save_png(&rep.join(format!("test{k:03}_irys.png")), &r, v.cam.width, v.cam.height);
        }
        ims.sort_by(f64::total_cmp);
        m["irys_ray_traced"] = serde_json::json!({"test_psnr": ip / nt, "test_ssim": is / nt, "ms_median": 1e3 * ims[ims.len() / 2]});
    }
    std::fs::write(rep.join("metrics.json"), serde_json::to_string_pretty(&m).unwrap()).unwrap();
    println!("{}", serde_json::to_string(&m).unwrap());
}
