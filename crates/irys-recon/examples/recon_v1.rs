//! The v1 gate on synthetic photos.
//!
//! `cargo run --release -p irys-recon --features gpu --example recon_v1 -- render [--seeds A..B] [--out DIR]`
//! renders, per seed, one phone-like photo of v0's gate part standing at
//! an unknown pose beside a printed tag36h11 on a paper backdrop: irys's
//! OpenPBR scene kernel on the device (`look::scene_gpu`), PLA (a
//! saturated `base_color`, `specular_roughness` 0.35–0.6, `specular_ior`
//! 1.5), a captured environment (the Blender demos' baked worlds, turned
//! at random) plus a key light, then a camera: defocus, exposure, shot and
//! read noise, the sRGB curve with a random extra power, JPEG at 85–92, and
//! an EXIF focal length. Writes `DIR/<seed>.jpg` and `DIR/<seed>.json`
//! (the truth).
//!
//! `... --example recon_v1 -- gate [--seeds A..B] [--dir DIR] [--out OUT]`
//! runs v1 on each photo alone and writes `OUT/report.{json,md}`: topology,
//! dimension errors after the plate's quarter-turn symmetry, pass at
//! 0.5 mm and at 1 mm, pose errors, σ, time.

use std::path::{Path, PathBuf};

use irys_diff::look::openpbr::OpenPbr;
use irys_diff::look::scene::{Camera as LCam, Equirect, Instance, LookScene, Material, Mesh, Sun, Texture, mask};
use irys_diff::look::scene_gpu::{GpuLook, Opts};
use irys_recon::dsl::Program;
use irys_recon::jac::Mix;
use irys_recon::{canon, generate, geom, tag, v1};

const GATE: std::ops::Range<u64> = 3000..3020;

#[derive(serde::Serialize, serde::Deserialize)]
struct Truth {
    seed: u64,
    program: String,
    pose: [f64; 3],
    pla: [f64; 3],
    roughness: f64,
    env: String,
    env_turn_deg: f64,
    key_dir: [f64; 3],
    key_radius_deg: f64,
    eye: [f64; 3],
    at: [f64; 3],
    f35: u16,
    size: (usize, usize),
    extra_gamma: f64,
    jpeg_quality: u8,
}

fn args() -> (String, Vec<String>) {
    let mut a: Vec<String> = std::env::args().skip(1).collect();
    let cmd = if a.is_empty() { "gate".into() } else { a.remove(0) };
    (cmd, a)
}

fn opt<T: std::str::FromStr>(a: &[String], k: &str, d: T) -> T {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn seeds(a: &[String]) -> std::ops::Range<u64> {
    let v: String = opt(a, "--seeds", String::new());
    match v.split_once("..") {
        Some((x, y)) => x.parse().unwrap()..y.parse().unwrap(),
        None => GATE,
    }
}

fn main() {
    let (cmd, a) = args();
    match cmd.as_str() {
        "render" => render(&a),
        "gate" => gate(&a),
        c => panic!("unknown command {c}: render | gate"),
    }
}

fn load_env(dir: &Path) -> Equirect {
    let side: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("irys.json")).unwrap()).unwrap();
    let e = &side["env"]["equirect"];
    let (ew, eh) = (e["size"][0].as_u64().unwrap() as usize, e["size"][1].as_u64().unwrap() as usize);
    let b = std::fs::read(dir.join("env.f32")).unwrap();
    let rgb: Vec<[f32; 3]> = b
        .chunks_exact(12)
        .map(|c| std::array::from_fn(|i| f32::from_le_bytes([c[4 * i], c[4 * i + 1], c[4 * i + 2], c[4 * i + 3]])))
        .collect();
    assert_eq!(rgb.len(), ew * eh);
    Equirect { size: (ew as u32, eh as u32), rgb }
}

/// An indoor world: warm walls, a darker floor, a bright ceiling panel and
/// a window (bluish, wide), as a phone on a desk would see it.
fn room_env() -> Equirect {
    let (w, h) = (512u32, 256u32);
    let mut rgb = Vec::with_capacity((w * h) as usize);
    for j in 0..h {
        for i in 0..w {
            let th = std::f64::consts::PI * (j as f64 + 0.5) / h as f64;
            let ph = std::f64::consts::TAU * (i as f64 + 0.5) / w as f64;
            let z = th.cos();
            let el = z.asin().to_degrees();
            let az = ph.to_degrees();
            let mut c = if z < 0.0 { [0.07, 0.06, 0.05] } else { [0.22 + 0.1 * z, 0.2 + 0.1 * z, 0.17 + 0.1 * z] };
            if (40.0..95.0).contains(&az) && (3.0..42.0).contains(&el) {
                c = [2.2, 2.5, 3.0];
            }
            if el > 62.0 && (200.0..250.0).contains(&az) {
                c = [4.0, 3.7, 3.2];
            }
            rgb.push(c.map(|x| x as f32));
        }
    }
    Equirect { size: (w, h), rgb }
}

fn srgb8(x: f64) -> u8 {
    let x = x.clamp(0.0, 1.0);
    let s = if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}

fn rgba(c: [f64; 3]) -> u32 {
    let [r, g, b] = c.map(srgb8);
    r as u32 | (g as u32) << 8 | (b as u32) << 16 | 255 << 24
}

/// Rotate about z by `deg`.
fn rz(deg: f64, v: [f64; 3]) -> [f64; 3] {
    let (s, c) = deg.to_radians().sin_cos();
    [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]]
}

fn render(a: &[String]) {
    let out = PathBuf::from(opt(a, "--out", "../irys-assets/recon-v1-synth".to_string()));
    std::fs::create_dir_all(&out).unwrap();
    let assets = std::env::var("IRYS_ASSETS").unwrap_or("/Users/cam/Developer/irys-assets".into());
    let envs = [("pavilion", load_env(&Path::new(&assets).join("blender-demos/export5b/pavilion"))), ("room", room_env())];
    let ctx = irys_diff::gpu::context().expect("gpu");
    let size = (opt(a, "--width", 2016usize), opt(a, "--height", 1512usize));
    let spp: u32 = opt(a, "--spp", 160);
    // The tag paper's texture: 90 mm square at 0.125 mm per texel, the
    // print centred, the rest white paper.
    let cells = tag::cells(tag::TAG_ID);
    let paper_mm = 90.0;
    let tn = 720u32;
    let paper = [0.86, 0.86, 0.85];
    let texels: Vec<u32> = (0..tn * tn)
        .map(|i| {
            let (u, v) = ((i % tn) as f64 + 0.5, (i / tn) as f64 + 0.5);
            let (x, y) = (-paper_mm / 2.0 + u * paper_mm / tn as f64, paper_mm / 2.0 - v * paper_mm / tn as f64);
            match tag::white_at(&cells, tag::TAG_MM, x, y) {
                Some(false) => rgba([0.03, 0.03, 0.03]),
                _ => rgba(paper),
            }
        })
        .collect();
    let pla = [[0.85, 0.28, 0.04], [0.05, 0.18, 0.65], [0.65, 0.05, 0.06], [0.08, 0.45, 0.15], [0.80, 0.62, 0.04], [0.35, 0.08, 0.45]];
    for seed in seeds(a) {
        let mut rng = Mix(seed * 7919 + 13);
        let truth = generate::program(seed);
        let solid = truth.build().expect("gate parts build");
        let (lp, lt) = geom::mesh(&solid);
        // Where the part stands: beside the tag, clear of it.
        let half_diag = 0.5 * truth.w.hypot(truth.d);
        let dir = rng.uniform() * 360.0;
        let dist = 31.25 + 12.0 + half_diag + 10.0 * rng.uniform();
        let pose = [dist * dir.to_radians().cos(), dist * dir.to_radians().sin(), rng.uniform() * 360.0 - 180.0];
        // The camera: looking at the pair from the side, 30–60° up.
        let mid = [pose[0] * 0.45, pose[1] * 0.45, 0.0];
        let az = dir + if rng.uniform() < 0.5 { 90.0 } else { -90.0 } + (rng.uniform() - 0.5) * 60.0;
        let el = 30.0 + 30.0 * rng.uniform();
        let range = 300.0 + 80.0 * rng.uniform();
        let eye = [
            mid[0] + range * el.to_radians().cos() * az.to_radians().cos(),
            mid[1] + range * el.to_radians().cos() * az.to_radians().sin(),
            range * el.to_radians().sin(),
        ];
        let at = [mid[0] + (rng.uniform() - 0.5) * 20.0, mid[1] + (rng.uniform() - 0.5) * 20.0, 0.0];
        let f35 = 26u16;
        let f_px = f35 as f64 / 36f64.hypot(24.0) * (size.0 as f64).hypot(size.1 as f64);
        let tan_half = size.1 as f64 / 2.0 / f_px;
        // The world the renderer sees is the tag frame turned by `turn`, so
        // the captured environment lands at a random azimuth.
        let turn = rng.uniform() * 360.0;
        let (env_name, env) = &envs[(seed % 2) as usize];
        let el_k = (35.0 + 40.0 * rng.uniform()).to_radians();
        let az_k = (rng.uniform() * 360.0).to_radians();
        let key_dir = [el_k.cos() * az_k.cos(), el_k.cos() * az_k.sin(), el_k.sin()];
        let key_r = (3.0 + 7.0 * rng.uniform()).to_radians();
        let mean_env: f64 = env.rgb.iter().map(|c| (c[0] + c[1] + c[2]) as f64 / 3.0).sum::<f64>() / env.rgb.len() as f64;
        let key_e = mean_env * std::f64::consts::PI * (1.0 + 1.5 * rng.uniform());
        let key_rad = key_e / (std::f64::consts::PI * key_r.sin().powi(2));
        let colour = pla[(rng.uniform() * pla.len() as f64) as usize % pla.len()];
        let rough = 0.35 + 0.25 * rng.uniform();
        // part frame → tag frame → render world: Rz(turn) · (Rz(yaw) p + t).
        let part_xf: [[f32; 4]; 3] = {
            let (s, c) = (pose[2] + turn).to_radians().sin_cos();
            let t = rz(turn, [pose[0], pose[1], 0.0]);
            [[c as f32, -s as f32, 0.0, t[0] as f32], [s as f32, c as f32, 0.0, t[1] as f32], [0.0, 0.0, 1.0, t[2] as f32]]
        };
        let part = Mesh {
            positions: lp.iter().map(|p| [p.x as f32, p.y as f32, p.z as f32]).collect(),
            normals: Vec::new(),
            uvs: Vec::new(),
            indices: lt.iter().flatten().copied().collect(),
            material: 0,
        };
        let quad = |half: f32, z: f32, mat: usize| Mesh {
            positions: vec![[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]],
            normals: vec![[0.0, 0.0, 1.0]; 4],
            uvs: vec![[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            indices: vec![0, 1, 2, 0, 2, 3],
            material: mat,
        };
        let (st, ct) = turn.to_radians().sin_cos();
        let turn_xf: [[f32; 4]; 3] = [[ct as f32, -st as f32, 0.0, 0.0], [st as f32, ct as f32, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
        let mut paper_mat = Material::new("tag paper", OpenPbr { base_color: [1.0; 3], specular_roughness: 0.85, specular_weight: 0.5, ..OpenPbr::default() });
        paper_mat.base_tex = Some(0);
        let backdrop = Material::new("backdrop", OpenPbr { base_color: [0.74, 0.74, 0.72], specular_roughness: 0.9, specular_weight: 0.4, ..OpenPbr::default() });
        let pla_mat = Material::new("PLA", OpenPbr { base_color: colour, specular_roughness: rough, specular_ior: 1.5, ..OpenPbr::default() });
        let e_r = rz(turn, eye);
        let a_r = rz(turn, at);
        let fwd = norm(sub(a_r, e_r));
        let right = norm(cross(fwd, [0.0, 0.0, 1.0]));
        let up = cross(right, fwd);
        let scene = LookScene {
            meshes: vec![part, quad(paper_mm as f32 / 2.0, 0.08, 1), quad(4000.0, 0.0, 2)],
            instances: vec![
                Instance { mesh: 0, xf: part_xf, mask: mask::GEOMETRY },
                Instance { mesh: 1, xf: turn_xf, mask: mask::GEOMETRY },
                Instance { mesh: 2, xf: turn_xf, mask: mask::GEOMETRY },
            ],
            materials: vec![pla_mat, paper_mat, backdrop],
            textures: vec![Texture { size: (tn, tn), texels: texels.clone() }],
            env: Some(env.clone()),
            sun: Some(Sun { dir: rz(turn, key_dir), radius: key_r, radiance: [key_rad, key_rad * 0.97, key_rad * 0.92] }),
            portal_skies: Vec::new(),
            camera: LCam { eye: e_r, forward: fwd, right, up, tan_half },
            size,
            frames: Vec::new(),
            fps: 24.0,
            dropped: Vec::new(),
        };
        let g = GpuLook::new(ctx, &scene).expect("ray queries");
        let t0 = std::time::Instant::now();
        // Indirect samples clamped (Cycles' `sample_clamp_indirect`): the
        // sun seen through a glossy bounce otherwise leaves fireflies on
        // the part, which a camera would not.
        let fr = g.render(&scene.camera, &Opts { spp, max_bounces: 4, clamp: 6.0, seed: seed + 1, ..Opts::default() });
        // The camera: defocus (σ 0.7 px), exposure to a 99th percentile of
        // 0.9, Poisson–Gaussian noise, sRGB with an extra power, JPEG.
        let (w, h) = size;
        let nan = fr.rgb.iter().filter(|x| !x.is_finite()).count();
        let mut lin: Vec<f64> = fr.rgb.iter().map(|&x| if x.is_finite() { x as f64 } else { 0.0 }).collect();
        if nan > 0 {
            eprintln!("seed {seed}: {nan} non-finite samples zeroed");
        }
        lin = irys_recon::fit::blur(&lin, size, 3, 0.7);
        let mut lum: Vec<f64> = lin.chunks_exact(3).map(|c| (c[0] + c[1] + c[2]) / 3.0).collect();
        lum.sort_by(f64::total_cmp);
        let expo = 0.9 / lum[lum.len() * 99 / 100].max(1e-6);
        let extra_gamma = 0.9 + 0.2 * rng.uniform();
        let q = 85 + (rng.uniform() * 8.0) as u8;
        let mut noise = Mix(seed ^ 0xa11ce);
        let mut gauss = move || {
            let (u1, u2) = (noise.uniform().max(1e-12), noise.uniform());
            (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
        };
        let px: Vec<u8> = lin
            .iter()
            .map(|&x| {
                let v = x * expo;
                let v = v + (0.0006 * v.max(0.0) + 4e-6).sqrt() * gauss();
                let s = srgb8(v) as f64 / 255.0;
                (s.powf(extra_gamma) * 255.0 + 0.5).clamp(0.0, 255.0) as u8
            })
            .collect();
        let img = image::RgbImage::from_raw(w as u32, h as u32, px).unwrap();
        let mut buf = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, q).encode_image(&img).unwrap();
        std::fs::write(out.join(format!("{seed}.jpg")), tag::with_exif(&buf, 5.7, f35)).unwrap();
        let t = Truth {
            seed,
            program: truth.to_text(),
            pose,
            pla: colour,
            roughness: rough,
            env: env_name.to_string(),
            env_turn_deg: turn,
            key_dir,
            key_radius_deg: key_r.to_degrees(),
            eye,
            at,
            f35,
            size,
            extra_gamma,
            jpeg_quality: q,
        };
        std::fs::write(out.join(format!("{seed}.json")), serde_json::to_string_pretty(&t).unwrap()).unwrap();
        println!("seed {seed}: {:.1}s, pose ({:.1}, {:.1}, {:.1}°), el {:.0}°, {env_name}", t0.elapsed().as_secs_f64(), pose[0], pose[1], pose[2], el);
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: [f64; 3]) -> [f64; 3] {
    let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    [a[0] / n, a[1] / n, a[2] / n]
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Part {
    seed: u64,
    truth: String,
    recovered: String,
    topology_ok: bool,
    max_error_mm: Option<f64>,
    errors: Vec<(String, f64)>,
    pass_05: bool,
    pass_1: bool,
    pose_err: Option<[f64; 3]>,
    focal_err_pct: f64,
    tag_rms_px: f64,
    sigma: Vec<(String, f64)>,
    gamma: f64,
    roughness: f64,
    rgb: [f64; 3],
    pla: [f64; 3],
    seconds: f64,
    error: Option<String>,
}

/// The recovered program turned by the quarter turn that brings its yaw
/// closest to the truth's, and the pose error after it.
fn aligned(tp: [f64; 3], rec: &Program) -> (Program, [f64; 3]) {
    let rp = rec.pose.unwrap_or([0.0; 3]);
    let wrap = |a: f64| (a + 540.0).rem_euclid(360.0) - 180.0;
    let k = (0..4).min_by(|&a, &b| wrap(rp[2] - 90.0 * a as f64 - tp[2]).abs().total_cmp(&wrap(rp[2] - 90.0 * b as f64 - tp[2]).abs())).unwrap();
    let q = rec.quarter_turned(k);
    let qp = q.pose.unwrap();
    (q, [qp[0] - tp[0], qp[1] - tp[1], wrap(qp[2] - tp[2])])
}

fn gate(a: &[String]) {
    let dir = PathBuf::from(opt(a, "--dir", "../irys-assets/recon-v1-synth".to_string()));
    let out = PathBuf::from(opt(a, "--out", "out/recon_v1".to_string()));
    std::fs::create_dir_all(&out).unwrap();
    let mut o = v1::V1Options::default();
    if a.iter().any(|x| x == "--no-fd") {
        o.fd_occluded = false;
    }
    if a.iter().any(|x| x == "--no-secondary") {
        o.secondary = false;
    }
    if a.iter().any(|x| x == "--v0-search") {
        o.search = irys_recon::search::Options { huber: Some(o.huber), ..irys_recon::search::Options::v0() };
    }
    let mut parts = Vec::new();
    for seed in seeds(a) {
        let t: Truth = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{seed}.json"))).unwrap()).unwrap();
        let truth = Program::from_text(&t.program).unwrap();
        let t0 = std::time::Instant::now();
        let photo = v1::load(&dir.join(format!("{seed}.jpg"))).unwrap();
        let f_true = t.f35 as f64 / 36f64.hypot(24.0) * (t.size.0 as f64).hypot(t.size.1 as f64);
        let r = v1::recover_photos(std::slice::from_ref(&photo), &o);
        let part = match r {
            Ok(v) => {
                let (q, pe) = aligned(t.pose, &v.rec.program);
                let topology_ok = canon::same_topology(&truth, &q);
                let errors = canon::errors(&truth, &q).unwrap_or_default();
                let max_error_mm = topology_ok.then(|| errors.iter().fold(0.0f64, |m, e| m.max(e.1)));
                let l = v.looks[0];
                v1::overlay(&v, 0).save(out.join(format!("{seed}_overlay.png"))).ok();
                Part {
                    seed,
                    truth: t.program.clone(),
                    recovered: q.to_text(),
                    topology_ok,
                    max_error_mm,
                    errors,
                    pass_05: max_error_mm.is_some_and(|e| e <= 0.5),
                    pass_1: max_error_mm.is_some_and(|e| e <= 1.0),
                    pose_err: Some(pe),
                    focal_err_pct: 100.0 * (v.calib[0].k.f / f_true - 1.0),
                    tag_rms_px: v.calib[0].rms,
                    sigma: v.rec.judge.sigma.clone(),
                    gamma: l.gamma,
                    roughness: l.rough,
                    rgb: l.rgb,
                    pla: t.pla,
                    seconds: t0.elapsed().as_secs_f64(),
                    error: None,
                }
            }
            Err(e) => Part {
                seed,
                truth: t.program.clone(),
                recovered: String::new(),
                topology_ok: false,
                max_error_mm: None,
                errors: Vec::new(),
                pass_05: false,
                pass_1: false,
                pose_err: None,
                focal_err_pct: f64::NAN,
                tag_rms_px: f64::NAN,
                sigma: Vec::new(),
                gamma: f64::NAN,
                roughness: f64::NAN,
                rgb: [f64::NAN; 3],
                pla: t.pla,
                seconds: t0.elapsed().as_secs_f64(),
                error: Some(e),
            },
        };
        println!(
            "seed {seed}: {} / {} topo {} max err {:?} pose err {:?} f {:+.1}% {:.0}s {}",
            if part.pass_05 { "PASS" } else { "fail" },
            if part.pass_1 { "PASS" } else { "fail" },
            part.topology_ok,
            part.max_error_mm.map(|e| (e * 1000.0).round() / 1000.0),
            part.pose_err.map(|p| p.map(|x| (x * 100.0).round() / 100.0)),
            part.focal_err_pct,
            part.seconds,
            part.error.clone().unwrap_or_default()
        );
        parts.push(part);
        write_report(&out, &parts);
    }
}

fn write_report(out: &Path, parts: &[Part]) {
    std::fs::write(out.join("report.json"), serde_json::to_string_pretty(parts).unwrap()).unwrap();
    let n = parts.len();
    let p05 = parts.iter().filter(|p| p.pass_05).count();
    let p1 = parts.iter().filter(|p| p.pass_1).count();
    let topo = parts.iter().filter(|p| p.topology_ok).count();
    let mut md = format!("# recon v1, synthetic photos\n\n{p05}/{n} pass at 0.5 mm, {p1}/{n} at 1 mm ({topo}/{n} topology).\n\n");
    md += "| seed | 0.5 mm | 1 mm | topology | max err mm | pose err (x, y mm, yaw °) | focal err % | tag rms px | γ | rough | rgb (truth) | s | truth | recovered |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n";
    for p in parts {
        md += &format!(
            "| {} | {} | {} | {} | {} | {} | {:+.2} | {:.2} | {:.2} | {:.2} | {:.2},{:.2},{:.2} ({:.2},{:.2},{:.2}) | {:.0} | `{}` | `{}` |\n",
            p.seed,
            if p.pass_05 { "✓" } else { "✗" },
            if p.pass_1 { "✓" } else { "✗" },
            if p.topology_ok { "✓" } else { "✗" },
            p.max_error_mm.map_or("–".into(), |e| format!("{e:.3}")),
            p.pose_err.map_or("–".into(), |e| format!("{:.2}, {:.2}, {:.2}", e[0], e[1], e[2])),
            p.focal_err_pct,
            p.tag_rms_px,
            p.gamma,
            p.roughness,
            p.rgb[0], p.rgb[1], p.rgb[2], p.pla[0], p.pla[1], p.pla[2],
            p.seconds,
            p.truth.replace('\n', "; "),
            if p.recovered.is_empty() { p.error.clone().unwrap_or_default() } else { p.recovered.replace('\n', "; ") },
        );
    }
    std::fs::write(out.join("report.md"), md).unwrap();
}
