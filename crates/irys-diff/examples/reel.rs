//! Phase 5b: frame sequences of irys's hero moments for a reel, the
//! labelled side-by-sides of the demo renders, and the reel's manifest.
//!
//! ```text
//! cargo run --release -p irys-diff --features gpu,assets --example reel -- <mode> <out dir> [options]
//!   fit    <out>   the gold ball's fit converging (phase 2b's garden), along
//!                  docs/phase2b-loss.csv's trajectory
//!   rose   <out>   rose's arm fit (phase 2f, in context), along
//!                  docs/phase2f-rose-loss.csv's 600-iteration run
//!   head   <out>   a slow orbit of the Lee Perry-Smith head with hair
//!   swatch <out>   OpenPBR swatches under the pavilion's sky, each animating
//!                  one parameter
//!   compose <out.png> <cols> <WxH> <label=path.png>...   a labelled grid of WxH cells
//!   manifest <reel dir>   reel/manifest.json from every frames.json under it
//!   --frames N --spp N --width W --height H
//! ```
//!
//! Frames are 1920×1080 JPEG q92 (`NNNN.jpg`) with `frames.json`: per frame
//! the mean spectrum, irys's per-wavelength radiance averaged over the frame
//! in 40 bins over 380–780 nm. For fit and rose it is the wavefront's own
//! per-lane `L` (the lanes' wavelengths from the scene's sampler); for the
//! head, the skin kernel's per-path lanes; for the swatches, look::scene_gpu's
//! per-workgroup spectrum. The low-resolution spectrum passes (fit, rose,
//! head) sample the same frame as the image.

#[path = "rose/arm.rs"]
#[allow(dead_code)]
mod arm;
#[path = "common/font.rs"]
mod font;

use std::path::{Path, PathBuf};
use std::time::Instant;

use irys_diff::gpu::{self, GpuPrb};
use irys_diff::prb::Scene;

const BINS: usize = 40;

fn opt<T: std::str::FromStr>(args: &[String], k: &str, d: T) -> T {
    args.iter()
        .position(|a| a == k)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn srgb8(x: f32) -> u8 {
    let x = x.clamp(0.0, 1.0);
    let s = if x <= 0.003_130_8 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}

fn jpeg(path: &Path, px: &[u8], w: usize, h: usize) {
    let file = std::fs::File::create(path).unwrap();
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::BufWriter::new(file), 92);
    enc.encode(px, w as u32, h as u32, image::ExtendedColorType::Rgb8).unwrap();
}

/// Mean `L` per wavelength bin from lanes `(λ, L)`.
fn bin(lanes: impl Iterator<Item = (f64, f64)>) -> Vec<f64> {
    let mut s = vec![0.0; BINS];
    let mut n = vec![0.0; BINS];
    for (lam, l) in lanes {
        let b = (((lam - 380.0) / 400.0 * BINS as f64) as usize).min(BINS - 1);
        if l.is_finite() {
            s[b] += l;
        }
        n[b] += 1.0;
    }
    s.iter().zip(&n).map(|(a, c)| if *c > 0.0 { a / c } else { 0.0 }).collect()
}

/// A prb scene's mean spectrum: the device's per-lane `L`, each lane's
/// wavelength from the scene's own sampler (the CPU twin draws the same).
fn prb_spectrum(g: &mut GpuPrb, s: &Scene, theta: &[f64; 4], seed: u64) -> Vec<f64> {
    let l = g.forward(s, theta, seed);
    let nw = s.wavelengths;
    bin((0..s.paths()).flat_map(|i| {
        let mut rng = s.path_rng(seed, i);
        let lam = s.wavelengths_of(rng.uniform());
        let l = &l;
        (0..nw).map(move |w| (lam[w], l[i * nw + w] as f64))
    }))
}

struct Seq {
    dir: PathBuf,
    frames: Vec<serde_json::Value>,
    meta: serde_json::Value,
}

impl Seq {
    fn new(dir: &Path, meta: serde_json::Value) -> Self {
        std::fs::create_dir_all(dir).unwrap();
        Self { dir: dir.into(), frames: Vec::new(), meta }
    }
    fn push(&mut self, k: usize, px: &[u8], w: usize, h: usize, spectrum: Vec<f64>, seconds: f64, extra: serde_json::Value) {
        let name = format!("{k:04}.jpg");
        jpeg(&self.dir.join(&name), px, w, h);
        self.frames.push(serde_json::json!({"frame": k, "file": name, "seconds": seconds, "spectrum": spectrum, "state": extra}));
        let mut m = self.meta.clone();
        m["size"] = serde_json::json!([w, h]);
        m["bins"] = serde_json::json!(BINS);
        m["bins_nm"] = serde_json::json!([380, 780]);
        m["frames"] = serde_json::Value::Array(self.frames.clone());
        std::fs::write(self.dir.join("frames.json"), serde_json::to_string(&m).unwrap()).unwrap();
        eprintln!("{} {k:4} {seconds:.2} s", self.dir.display());
    }
}

fn csv_rows(path: &Path, run: Option<&str>) -> (Vec<String>, Vec<Vec<f64>>) {
    let txt = std::fs::read_to_string(path).unwrap();
    let mut lines = txt.lines();
    let head: Vec<String> = lines.next().unwrap().split(',').map(str::to_string).collect();
    let rows = lines
        .filter(|l| run.is_none_or(|r| l.starts_with(r)))
        .map(|l| {
            let c: Vec<&str> = l.split(',').collect();
            let skip = usize::from(run.is_some());
            c[skip..].iter().map(|x| x.parse().unwrap_or(0.0)).collect()
        })
        .collect();
    let head = if run.is_some() { head[1..].to_vec() } else { head };
    (head, rows)
}

fn col(head: &[String], k: &str) -> usize {
    head.iter().position(|h| h == k).unwrap_or_else(|| panic!("no column {k}"))
}

/// The state at fraction `s` of a trajectory, linear between its rows.
fn along(rows: &[Vec<f64>], cols: &[usize], s: f64) -> Vec<f64> {
    let p = s.clamp(0.0, 1.0) * (rows.len() - 1) as f64;
    let i = (p.floor() as usize).min(rows.len() - 2);
    let t = p - i as f64;
    cols.iter().map(|&c| rows[i][c] * (1.0 - t) + rows[i + 1][c] * t).collect()
}

/// Ease: slow at the end, where a fit's changes are small and worth seeing.
fn ease(s: f64) -> f64 {
    1.0 - (1.0 - s).powf(1.7)
}

fn fit(out: &Path, args: &[String]) {
    let (w, h) = (opt(args, "--width", 1920usize), opt(args, "--height", 1080usize));
    let n: usize = opt(args, "--frames", 150);
    let spp: u64 = opt(args, "--spp", 32);
    let ctx = gpu::context().expect("gpu");
    let (head, rows) = csv_rows(&irys_diff_docs().join("phase2b-loss.csv"), None);
    let cols = ["alpha", "f0_450", "f0_550", "f0_650"].map(|k| col(&head, k));
    let mut s = Scene::garden(w, h, 1);
    s.depth = 6;
    let mut g = GpuPrb::new(ctx, &s);
    let mut low = Scene::garden(240, 135, 4);
    low.depth = 6;
    let mut gl = GpuPrb::new(ctx, &low);
    let mut seq = Seq::new(out, serde_json::json!({"name": "fit_scene", "fps": 30, "spp": spp,
        "caption": "The gold ball's roughness and colour recovered by path replay backprop, 60 Adam steps",
        "spectrum_source": "GpuPrb per-lane L, 240x135 x 4 spp per frame"}));
    for k in 0..n {
        let st = along(&rows, &cols, ease(k as f64 / (n - 1) as f64));
        let theta = [st[0], st[1], st[2], st[3]];
        let t = Instant::now();
        g.clear_accumulated();
        for f in 0..spp {
            g.accumulate(&s, &theta, 50_000 + k as u64 * 1000 + f);
        }
        let (img, _) = g.accumulated();
        let px: Vec<u8> = img.iter().map(|&x| srgb8(x)).collect();
        let sp = prb_spectrum(&mut gl, &low, &theta, 90_000 + k as u64);
        seq.push(k, &px, w, h, sp, t.elapsed().as_secs_f64(), serde_json::json!({"theta": theta}));
    }
}

fn irys_diff_docs() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs")
}

fn rose(out: &Path, args: &[String]) {
    let (w, h) = (opt(args, "--width", 1920usize), opt(args, "--height", 1080usize));
    let n: usize = opt(args, "--frames", 180);
    let spp: u64 = opt(args, "--spp", 32);
    let ctx = gpu::context().expect("gpu");
    let (head, rows) = csv_rows(&irys_diff_docs().join("phase2f-rose-loss.csv"), Some("context; 2f; 600 its"));
    let cols = ["sho_pitch", "sho_roll", "elbow", "alpha", "f0_450", "f0_550", "f0_650"].map(|k| col(&head, k));
    let a = arm::load_arm(&arm::rose_dir());
    let rig = a.rig();
    let st0 = along(&rows, &cols, 0.0);
    let mut s = arm::scene_from(&a, (w, h), 1, &st0[..3], arm::FIT_CAM);
    let mut g = GpuPrb::new(ctx, &s);
    let mut low = arm::scene_from(&a, (240, 135), 4, &st0[..3], arm::FIT_CAM);
    let mut gl = GpuPrb::new(ctx, &low);
    let exposure = 0.6f32;
    let mut seq = Seq::new(out, serde_json::json!({"name": "rose_fit", "fps": 30, "spp": spp,
        "caption": "rose's left arm: pose and finish recovered together (edge + interior + secondary terms, 600 steps)",
        "spectrum_source": "GpuPrb per-lane L, 240x135 x 4 spp per frame (before the 0.6 exposure)"}));
    for k in 0..n {
        let st = along(&rows, &cols, ease(k as f64 / (n - 1) as f64));
        let (phi, theta) = ([st[0], st[1], st[2]], [st[3], st[4], st[5], st[6]]);
        let t = Instant::now();
        rig.pose(&mut s, &phi);
        g.update_geometry(&s);
        g.clear_accumulated();
        for f in 0..spp {
            g.accumulate(&s, &theta, 70_000 + k as u64 * 1000 + f);
        }
        let (img, _) = g.accumulated();
        let px: Vec<u8> = img.iter().map(|&x| srgb8(x * exposure)).collect();
        rig.pose(&mut low, &phi);
        gl.update_geometry(&low);
        let sp = prb_spectrum(&mut gl, &low, &theta, 95_000 + k as u64);
        seq.push(k, &px, w, h, sp, t.elapsed().as_secs_f64(), serde_json::json!({"phi": phi, "theta": theta}));
    }
}

fn head(out: &Path, args: &[String]) {
    use irys_diff::look::hair::Fiber;
    use irys_diff::look::hair_gpu::{GpuStrands, StrandPrim};
    use irys_diff::skin::gpu::{GpuSkin, SkinHair};
    use irys_diff::skin::groom::{GroomSpec, groom};
    use irys_diff::skin::render::{Camera, Chromophores, Mode, THETA, tonemap};
    use irys_diff::skin::{head as hd, mesh, optics};
    let (w, h) = (opt(args, "--width", 1920usize), opt(args, "--height", 1080usize));
    let n: usize = opt(args, "--frames", 120);
    let spp: usize = opt(args, "--spp", 24);
    let degrees: f64 = opt(args, "--degrees", 50.0);
    let ctx = gpu::context().expect("gpu");
    let asset = mesh::load(&mesh::asset_dir()).expect("the Lee Perry-Smith asset");
    let (chromo, _) = optics::invert(&asset.albedo);
    let s0 = hd::scene(&asset, Chromophores::Map(chromo.clone()), hd::View::Front);
    let gr = groom(&s0.mesh, &asset.albedo, &GroomSpec::default());
    let strands = GpuStrands::new(ctx, &gr.strands, Some(&gr.melanin), StrandPrim::Cylinders).expect("ray queries");
    let hair = || SkinHair { strands: &strands, fiber: Fiber::default(), melanin: [5.0, 0.4] };
    let mut s = hd::scene(&asset, Chromophores::Map(chromo.clone()), hd::View::Front);
    s.set_size(w, h);
    s.spp = spp;
    s.mode = Mode { sss: true, dual: true };
    let mut low = hd::scene(&asset, Chromophores::Map(chromo.clone()), hd::View::Front);
    low.set_size(192, 108);
    low.spp = 4;
    low.mode = Mode { sss: true, dual: true };
    let gs = GpuSkin::with_strands(ctx, &s, false, Some(hair()));
    let gl = GpuSkin::with_strands(ctx, &low, true, Some(hair()));
    let c = irys_core::Point3::new(hd::HEAD[0], hd::HEAD[1], hd::HEAD[2]);
    let (eye0, target0) = (s.camera.eye, c + irys_core::Vec3::new(0.0, -2.0, -0.35));
    let tan_half = s.camera.tan_half;
    let mut exposure = 0.0;
    let mut seq = Seq::new(out, serde_json::json!({"name": "look2_head", "fps": 30, "spp": spp,
        "caption": "Lee Perry-Smith, spectral SSS skin with 243k GPU hair strands, slow orbit",
        "credit": "Lee Perry-Smith head by Infinite-Realities, CC BY 3.0",
        "spectrum_source": "GpuSkin per-path lanes, 192x108 x 4 spp per frame (before tone mapping)"}));
    for k in 0..n {
        let u = k as f64 / (n - 1) as f64;
        let e = u * u * (3.0 - 2.0 * u);
        let a = ((e - 0.5) * degrees).to_radians();
        let v = eye0 - c;
        let (ca, sa) = (a.cos(), a.sin());
        let eye = c + irys_core::Vec3::new(v.x * ca - v.y * sa, v.x * sa + v.y * ca, v.z);
        let cam = Camera::look_at(eye, target0, 2.0 * tan_half.atan().to_degrees(), 0.05);
        s.camera = cam;
        low.camera = cam;
        let t = Instant::now();
        let (img, _) = gs.render(&s, &THETA, 1 + k as u64, false);
        let img: Vec<f64> = img.iter().map(|&x| x as f64).collect();
        if exposure == 0.0 {
            exposure = auto_exposure(&img);
        }
        let px = tonemap(&img, exposure);
        let seed = 7_000 + k as u64;
        let (_, lanes) = gl.render(&low, &THETA, seed, true);
        let sp = bin((0..low.paths()).flat_map(|i| {
            let mut rng = irys_diff::prb::Rng::new(seed, i as u64);
            // skin.wgsl's rand(): the 24 high bits of the same draw
            let u0 = (rng.uniform() * 16_777_216.0).floor() / 16_777_216.0;
            let lanes = &lanes;
            (0..4).map(move |k| {
                let v = u0 + k as f64 / 4.0;
                (380.0 + 400.0 * (v - v.floor()), lanes[i * 4 + k] as f64)
            })
        }));
        seq.push(k, &px, w, h, sp, t.elapsed().as_secs_f64(), serde_json::json!({"orbit_deg": a.to_degrees()}));
    }
}

fn auto_exposure(img: &[f64]) -> f64 {
    let mut lum: Vec<f64> = img.chunks_exact(3).map(|c| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]).collect();
    lum.sort_by(f64::total_cmp);
    let top = lum[lum.len() * 999 / 1000];
    let lit: Vec<f64> = lum.into_iter().filter(|&l| l > 0.05 * top).collect();
    0.24 / lit[lit.len() / 2]
}

fn swatch(out: &Path, args: &[String]) {
    use irys_diff::look::openpbr::OpenPbr;
    use irys_diff::look::scene::{LookScene, Material};
    use irys_diff::look::scene_gpu::{GpuLook, Opts};
    let (w, h) = (opt(args, "--width", 1920usize), opt(args, "--height", 1080usize));
    let n: usize = opt(args, "--frames", 150);
    let spp: u32 = opt(args, "--spp", 256);
    let env_dir: String = opt(args, "--env", "../irys-assets/blender-demos/export5b/pavilion".to_string());
    let ctx = gpu::context().expect("gpu");
    // Five swatches, each sweeping one OpenPBR parameter 0 → 1 → 0.
    let gold = [1.0, 0.71, 0.29];
    let mk = |u: f64| -> Vec<Material> {
        let tri = 0.5 - 0.5 * (std::f64::consts::TAU * u).cos();
        vec![
            Material::new("roughness", OpenPbr { base_color: gold, base_metalness: 1.0, specular_roughness: 0.02 + 0.6 * tri, ..OpenPbr::default() }),
            Material::new("coat", OpenPbr { base_color: [0.6, 0.05, 0.05], specular_roughness: 0.5, coat_weight: tri, coat_roughness: 0.03, coat_darkening: 1.0, ..OpenPbr::default() }),
            Material::new("fuzz", OpenPbr { base_color: [0.08, 0.1, 0.35], specular_roughness: 0.6, fuzz_weight: tri, fuzz_color: [0.9, 0.9, 1.0], fuzz_roughness: 0.4, ..OpenPbr::default() }),
            Material::new("thin film", OpenPbr { base_color: [0.04; 3], specular_roughness: 0.08, thin_film_weight: 1.0, thin_film_thickness: 0.15 + 0.55 * tri, thin_film_ior: 1.45, ..OpenPbr::default() }),
            Material::new("dispersion", OpenPbr { transmission_weight: 1.0, specular_roughness: 0.0, specular_ior: 1.6, transmission_dispersion_scale: tri, transmission_dispersion_abbe_number: 12.0, ..OpenPbr::default() }),
        ]
    };
    let mut sc = LookScene::spheres(mk(0.0), 1.0, (w, h));
    let side: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(Path::new(&env_dir).join("irys.json")).unwrap()).unwrap();
    let e = &side["env"]["equirect"];
    let (ew, eh) = (e["size"][0].as_u64().unwrap() as usize, e["size"][1].as_u64().unwrap() as usize);
    let b = std::fs::read(Path::new(&env_dir).join("env.f32")).unwrap();
    let rgb: Vec<[f32; 3]> = b
        .chunks_exact(12)
        .map(|c| std::array::from_fn(|i| f32::from_le_bytes([c[4 * i], c[4 * i + 1], c[4 * i + 2], c[4 * i + 3]])))
        .collect();
    assert_eq!(rgb.len(), ew * eh);
    sc.env = Some(irys_diff::look::scene::Equirect { size: (ew as u32, eh as u32), rgb });
    // a camera that frames the row at 16:9
    sc.camera.tan_half = 0.36;
    let mut seq = Seq::new(out, serde_json::json!({"name": "openpbr_sweep", "fps": 30, "spp": spp,
        "caption": "OpenPBR on the GPU: metal roughness, coat, fuzz, thin film and dispersion, each swept 0 -> 1 -> 0",
        "spectrum_source": "look::scene_gpu per-workgroup spectrum, every pixel and sample of the frame",
        "materials": ["gold roughness 0.02-0.62", "coat 0-1 on red", "fuzz 0-1 on blue", "thin film 150-700 nm", "dispersion 0-1 (V=12)"]}));
    for k in 0..n {
        let u = k as f64 / n as f64;
        sc.materials = mk(u);
        let g = GpuLook::new(ctx, &sc).expect("ray queries");
        let t = Instant::now();
        let f = g.render(&sc.camera, &Opts { spp, max_bounces: 16, seed: 11 + k as u64, ..Opts::default() });
        let px: Vec<u8> = f.rgb.iter().map(|&x| srgb8(x)).collect();
        seq.push(k, &px, w, h, f.spectrum, t.elapsed().as_secs_f64(), serde_json::json!({"phase": u}));
    }
}

fn compose(args: &[String]) {
    let out = &args[0];
    let cols: usize = args[1].parse().unwrap();
    let (cw, ch): (usize, usize) = {
        let (a, b) = args[2].split_once('x').expect("cell WxH");
        (a.parse().unwrap(), b.parse().unwrap())
    };
    let cells: Vec<(String, image::RgbImage)> = args[3..]
        .iter()
        .map(|a| {
            let (l, p) = a.split_once('=').unwrap();
            (l.to_string(), image::open(p).unwrap_or_else(|e| panic!("{p}: {e}")).to_rgb8())
        })
        .collect();
    let rows = cells.len().div_ceil(cols);
    let (w, h) = (cw * cols, ch * rows);
    let mut px = vec![0u8; w * h * 3];
    for (i, (label, img)) in cells.iter().enumerate() {
        let (x0, y0) = ((i % cols) * cw, (i / cols) * ch);
        let img = if img.width() as usize != cw {
            image::imageops::resize(img, cw as u32, ch as u32, image::imageops::FilterType::Triangle)
        } else {
            img.clone()
        };
        for y in 0..ch {
            for x in 0..cw {
                let p = img.get_pixel(x as u32, y as u32).0;
                px[((y0 + y) * w + x0 + x) * 3..((y0 + y) * w + x0 + x) * 3 + 3].copy_from_slice(&p);
            }
        }
        let scale = (cw / 320).max(2);
        // draw into a temporary full-width buffer at the cell's origin
        font::text(&mut px, w, x0 + 4 * scale, y0 + 4 * scale, &label.to_uppercase(), scale);
    }
    image::save_buffer(out, &px, w as u32, h as u32, image::ColorType::Rgb8).unwrap();
}

fn manifest(dir: &Path) {
    let mut seqs = Vec::new();
    let mut total = 0u64;
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir.join("frames")).unwrap().map(|e| e.unwrap().path()).collect();
    entries.sort();
    for d in entries {
        let f = d.join("frames.json");
        let Ok(t) = std::fs::read_to_string(&f) else { continue };
        let m: serde_json::Value = serde_json::from_str(&t).unwrap();
        let frames = m["frames"].as_array().unwrap();
        let bytes: u64 = std::fs::read_dir(&d).unwrap().map(|e| e.unwrap().metadata().unwrap().len()).sum();
        total += bytes;
        let spectra: Vec<serde_json::Value> = frames.iter().map(|x| x["spectrum"].clone()).collect();
        let secs: f64 = frames.iter().map(|x| x["seconds"].as_f64().unwrap_or(0.0)).sum();
        seqs.push(serde_json::json!({
            "name": d.file_name().unwrap().to_str(),
            "path": d.strip_prefix(dir).unwrap().to_str(),
            "frames": frames.len(),
            "fps": m["fps"],
            "size": m["size"],
            "spp": m["spp"],
            "caption": m["caption"],
            "credit": m.get("credit"),
            "spectrum_source": m["spectrum_source"],
            "render_seconds": secs,
            "bytes": bytes,
            "spectrum": {"bins": BINS, "range_nm": [380, 780],
                "bin_centres_nm": (0..BINS).map(|b| 380.0 + 400.0 * (b as f64 + 0.5) / BINS as f64).collect::<Vec<_>>(),
                "unit": "mean spectral radiance (irys units, the renderer's own scale) per frame",
                "per_frame": spectra},
        }));
    }
    let m = serde_json::json!({"generator": "irys examples/reel.rs (phase 5b)", "total_bytes": total, "sequences": seqs});
    std::fs::write(dir.join("manifest.json"), serde_json::to_string_pretty(&m).unwrap()).unwrap();
    eprintln!("{} sequences, {:.1} MB", m["sequences"].as_array().unwrap().len(), total as f64 / 1e6);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args[0].as_str();
    let rest = &args[1..];
    match mode {
        "fit" => fit(Path::new(&rest[0]), rest),
        "rose" => rose(Path::new(&rest[0]), rest),
        "head" => head(Path::new(&rest[0]), rest),
        "swatch" => swatch(Path::new(&rest[0]), rest),
        "compose" => compose(rest),
        "manifest" => manifest(Path::new(&rest[0])),
        m => panic!("unknown mode {m}"),
    }
}
