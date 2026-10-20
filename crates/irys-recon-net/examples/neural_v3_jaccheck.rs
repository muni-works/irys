//! v3, piece 2: the edge-sampled Jacobian of the faithful compare image
//! against central differences of the same render, per family and slot.
//!
//! For objects from held-out scenes (true geometry, appearance fitted to the
//! photo), every used slot `k` (parameters and pose): `J_edge[:, k]` from
//! [`edgejac::jacobian`] (primary + shadow edges, analytic vertex
//! velocities), and `(I(θ + h e_k) − I(θ − h e_k)) / 2h` from two full
//! renders (common random numbers, high spp). Both are blurred by the
//! search's finest σ. Reported: cosine and relative L2 error per column.
//!
//! `neural_v3_jaccheck --set v3-a --scenes 8 --per 4`

use std::collections::BTreeMap;
use std::path::PathBuf;

use irys_recon_net::data::{hi_path, read_dir};
use irys_recon_net::edgejac::{self, EdgeOpts};
use irys_recon_net::faithful::{self, Appearance, Compare3, FitOpts, local_meshes};
use irys_recon_net::render::resample_rgb8;
use irys_recon_net::scene::{NP, param_used};
use irys_recon_net::search::{State, window};
use irys_recon_net::velocity::NV;

fn arg<T: std::str::FromStr>(a: &[String], k: &str, d: T) -> T {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn blur_flat(v: &[f64], size: (usize, usize), s: f64) -> Vec<f64> {
    irys_recon::fit::blur(v, size, 3, s)
}

fn crop(img: &[[f64; 3]], size: (usize, usize), (x0, y0, x1, y1): (usize, usize, usize, usize)) -> Vec<f64> {
    let mut o = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            o.extend_from_slice(&img[y * size.0 + x]);
        }
    }
    o
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = PathBuf::from(arg(&a, "--data", "out/neural/data".to_string()));
    let set: String = arg(&a, "--set", "v3-a".to_string());
    let scenes: usize = arg(&a, "--scenes", 8);
    let per: usize = arg(&a, "--per", 4);
    let spp: u32 = arg(&a, "--spp", 64);
    let sw: usize = arg(&a, "--size", 256);
    let sigma: f64 = arg(&a, "--blur", 1.0);
    let shadows = arg(&a, "--shadows", 1) == 1;
    let out = PathBuf::from(arg(&a, "--out", "out/neural/v3".to_string()));
    std::fs::create_dir_all(&out).unwrap();
    let size = (sw, sw * 3 / 4);
    let ctx = irys_diff::gpu::context().expect("gpu");
    let samples: Vec<_> = read_dir(&data, &set).into_iter().take(scenes).collect();
    // family → slot name → (cos, rel)
    let mut res: BTreeMap<String, BTreeMap<String, Vec<(f64, f64)>>> = BTreeMap::new();
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    let mut fdn: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut rows = Vec::new();
    for s in &samples {
        let hi = image::open(hi_path(&data, &set, s.scene.seed)).unwrap().to_rgb8();
        let rgb8: Vec<u8> = resample_rgb8(&hi, size).iter().flat_map(|p| p.map(irys_recon::render::srgb8)).collect();
        let cam = s.scene.cam.resized(size);
        let objs = s.scene.objects.clone();
        let mut app = Appearance::grey(objs.len());
        let meshes = local_meshes(&objs);
        let mut cmp = Compare3::new(ctx, &objs, &meshes, &app, &cam, size).unwrap();
        faithful::fit_appearance(&mut cmp, &cam, &rgb8, &mut app, &FitOpts { rounds: 1, materials: false, ..FitOpts::default() });
        let base = cmp.render(&cam, spp, 5);
        let st = State::new(objs.clone());
        let caster = st.caster();
        for (i, o) in objs.iter().enumerate() {
            let fam = o.family.name().to_string();
            if *seen.get(&fam).unwrap_or(&0) >= per {
                continue;
            }
            let Some(win) = window(&[o], &cam, size, 6.0) else { continue };
            *seen.entry(fam.clone()).or_default() += 1;
            let t0 = std::time::Instant::now();
            let j = edgejac::jacobian(o, i, &cam, &base, &caster, app.key.dir, win, &EdgeOpts { shadows, ..EdgeOpts::default() });
            let t_edge = t0.elapsed().as_secs_f64();
            let wsz = (win.2 - win.0, win.3 - win.1);
            for kk in 0..NV {
                let used = if kk < NP { kk < o.params.len() && param_used(o.family, &o.tokens, kk) } else { kk != NP + 2 || o.family.has_yaw() };
                if !used {
                    continue;
                }
                let name = if kk < NP { o.family.param_names()[kk].to_string() } else { ["x", "y", "yaw"][kk - NP].to_string() };
                let h = if kk < NP { (0.01 * o.params[kk]).max(0.15) } else if kk == NP + 2 { 1.0 } else { 0.5 };
                let render_at = |sgn: f64, seed: u64| {
                    let mut q = objs.clone();
                    if kk < NP {
                        q[i].params[kk] += sgn * h;
                    } else {
                        q[i].pose[kk - NP] += sgn * h;
                    }
                    let m = local_meshes(&q);
                    let mut c = Compare3::new(ctx, &q, &m, &app, &cam, size).unwrap();
                    c.set_appearance(&app);
                    c.render(&cam, spp, seed)
                };
                let fd_of = |seed: u64| {
                    let (ip, im) = (render_at(1.0, seed), render_at(-1.0, seed));
                    let fd: Vec<f64> = crop(&ip, size, win).iter().zip(crop(&im, size, win)).map(|(a, b)| (a - b) / (2.0 * h)).collect();
                    blur_flat(&fd, wsz, sigma)
                };
                let fd = fd_of(5);
                // The differences' own noise: a second seed.
                let fd2 = fd_of(6);
                let cos_fd = fd.iter().zip(&fd2).map(|(a, b)| a * b).sum::<f64>() / (fd.iter().map(|x| x * x).sum::<f64>().sqrt() * fd2.iter().map(|x| x * x).sum::<f64>().sqrt()).max(1e-30);
                let je = blur_flat(&j.cols[kk], wsz, sigma);
                let dot: f64 = fd.iter().zip(&je).map(|(a, b)| a * b).sum();
                let (nf, ne) = (fd.iter().map(|x| x * x).sum::<f64>().sqrt(), je.iter().map(|x| x * x).sum::<f64>().sqrt());
                if nf < 1e-9 {
                    continue;
                }
                let cos = dot / (nf * ne).max(1e-30);
                let ratio = dot / (nf * nf);
                let rel = fd.iter().zip(&je).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt() / nf;
                println!("{:>8} {:>6} {:>7}  fd-vs-fd cos {cos_fd:.3}  cos {:.3}  ratio {ratio:.2}  rel {:.3}  |fd| {:.3e} |edge| {:.3e}  ({} prim, {} shadow samples, {:.2}s)", fam, o.class.name(), name, cos, rel, nf, ne, j.hits.0, j.hits.1, t_edge);
                res.entry(fam.clone()).or_default().entry(name.clone()).or_default().push((cos, ratio));
                fdn.entry(fam.clone()).or_default().push(cos_fd);
                rows.push(serde_json::json!({"seed": s.scene.seed, "family": fam, "class": o.class.name(), "slot": name, "cos": cos, "ratio": ratio, "rel": rel, "fd_norm": nf, "fd_seed_cos": cos_fd, "edge_norm": ne}));
            }
        }
        std::fs::write(out.join(format!("jaccheck{}.json", if shadows { "" } else { "-noshadow" })), serde_json::to_string_pretty(&rows).unwrap()).unwrap();
    }
    println!("\nmedian cos / ratio per family and slot:");
    let med = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    for (f, m) in &res {
        let mut all_c = Vec::new();
        let mut all_r = Vec::new();
        let mut line = format!("{f:>8}:");
        for (k, v) in m {
            let c = med(v.iter().map(|x| x.0).collect());
            let r = med(v.iter().map(|x| x.1).collect());
            all_c.extend(v.iter().map(|x| x.0));
            all_r.extend(v.iter().map(|x| x.1));
            line += &format!("  {k} {c:.2}/{r:.2}");
        }
        println!("{line}   | all {:.2}/{:.2} (n {})  fd-vs-fd cos {:.2}", med(all_c.clone()), med(all_r), all_c.len(), med(fdn[f].clone()));
    }
}
