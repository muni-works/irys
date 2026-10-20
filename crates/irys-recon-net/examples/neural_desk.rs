//! The desk demo: one photo in, a parametric inventory out.
//! `neural_desk --photo desk.jpg --weights out/neural/ft-desk/weights.json --out out/neural/desk`
//! Writes `desk.png` (photo | irys re-render | residual | labelled
//! inventory), `inventory.md` and `inventory.json`.

use std::path::PathBuf;

use irys_recon_net::data::patches;
use irys_recon_net::infer::{self, gauge_scale, hypotheses, proposals, rescale};
use irys_recon_net::net::{H, W};
use irys_recon_net::render::{BACKDROP, Compare, TABLE, appearance, render_linear, resample_rgb8};
use irys_recon_net::scene::{Cam, Lighting, Mat, Object, Scene, param_used};
use irys_recon_net::search::{self, Ctx, State, noise_sigma, rho_sum};
use irys_recon_net::train::{Infer, Weights};

fn arg<T: std::str::FromStr>(a: &[String], k: &str, d: T) -> T {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(d)
}

const GREY: Mat = Mat { rgb: [0.5; 3], rough: 0.5, metal: false };

fn srgb(v: f64) -> u8 {
    irys_recon::render::srgb8(v)
}

/// 3 × 5 digits.
const DIGITS: [[u8; 5]; 10] = [
    [7, 5, 5, 5, 7],
    [2, 6, 2, 2, 7],
    [7, 1, 7, 4, 7],
    [7, 1, 7, 1, 7],
    [5, 5, 7, 1, 1],
    [7, 4, 7, 1, 7],
    [7, 4, 7, 5, 7],
    [7, 1, 1, 1, 1],
    [7, 5, 7, 5, 7],
    [7, 5, 7, 1, 7],
];

fn text(img: &mut image::RgbImage, x: i64, y: i64, s: &str, scale: i64, c: [u8; 3]) {
    let mut cx = x;
    for ch in s.chars() {
        if let Some(d) = ch.to_digit(10) {
            for (row, bits) in DIGITS[d as usize].iter().enumerate() {
                for col in 0..3 {
                    if bits >> (2 - col) & 1 == 1 {
                        for dy in 0..scale {
                            for dx in 0..scale {
                                let (px, py) = (cx + col * scale + dx, y + row as i64 * scale + dy);
                                if px >= 0 && py >= 0 && (px as u32) < img.width() && (py as u32) < img.height() {
                                    img.put_pixel(px as u32, py as u32, image::Rgb(c));
                                }
                            }
                        }
                    }
                }
            }
        }
        cx += 4 * scale;
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let photo = PathBuf::from(arg(&a, "--photo", "/Users/cam/Developer/irys-assets/photos/desk-IMG_0579.jpg".to_string()));
    let wpath = PathBuf::from(arg(&a, "--weights", "out/neural/run/weights.json".to_string()));
    let out = PathBuf::from(arg(&a, "--out", "out/neural/desk".to_string()));
    let iters: usize = arg(&a, "--iters", 10);
    std::fs::create_dir_all(&out).unwrap();
    let bytes = std::fs::read(&photo).unwrap();
    let exif = irys_recon::tag::read_exif(&bytes);
    let img = image::load_from_memory(&bytes).unwrap().to_rgb8();
    let (pw, ph) = (img.width() as usize, img.height() as usize);
    let f35 = exif.f35.unwrap_or(26.0);
    let f_px = f35 / 36f64.hypot(24.0) * (pw as f64).hypot(ph as f64);
    let hfov = 2.0 * (pw as f64 / 2.0 / f_px).atan().to_degrees();
    println!("{}: {pw}x{ph}, f35 {f35} mm → horizontal FOV {hfov:.1}°", photo.display());

    // The net at its own size.
    let small = image::imageops::resize(&img, W as u32, H as u32, image::imageops::FilterType::Triangle);
    let w = Weights::load(&wpath).unwrap();
    let mut net = Infer::new(&w, 1);
    let t0 = std::time::Instant::now();
    let o = net.run(&[patches(small.as_raw())]).pop().unwrap();
    let props = proposals(&o, 0.3);
    let mut cam = infer::camera(&o, hfov);
    let mut objs = Vec::new();
    let mut alts = Vec::new();
    let mut probs = Vec::new();
    let mut kept_props = Vec::new();
    let mut priors = Vec::new();
    for p in &props {
        let hs = hypotheses(p, 3);
        let cands: Vec<(Object, f64)> = hs.iter().filter_map(|h| infer::object(p, h, &cam, GREY).map(|o| (o, -h.logp))).collect();
        if let Some(f) = cands.first() {
            probs.push(p.cls[f.0.class as usize]);
            objs.push(f.0.clone());
            priors.push(f.1);
            alts.push(cands[1..].to_vec());
            kept_props.push(p.clone());
        }
    }
    let plane_net = o.plane;
    let (scale, gauges) = gauge_scale(&objs, &probs);
    if !gauges.is_empty() {
        rescale(&mut objs, &mut cam, scale);
        for al in alts.iter_mut() {
            for o in al.iter_mut() {
                rescale(std::slice::from_mut(&mut o.0), &mut cam.clone(), scale);
            }
        }
    }
    let t_net = t0.elapsed().as_secs_f64();
    println!("net: {} detections in {:.3}s; camera {:.0} mm up, pitch {:.1}°; gauge scale {:.3} from {:?}", objs.len(), t_net, cam.eye[2], plane_net[1].to_degrees(), scale, gauges.iter().map(|&i| objs[i].class.name()).collect::<Vec<_>>());

    // The search at twice the net's size.
    let size = (2 * W, 2 * H);
    let target = resample_rgb8(&img, size);
    let sigma = noise_sigma(&target, size);
    let ctx = Ctx { target: &target, size, cam, ss: 3, sigma, blur: 1.0 };
    let t1 = std::time::Instant::now();
    let (st, fits, evals) = search::search(objs.iter().cloned().zip(priors.iter().copied()).collect(), &alts, &ctx, iters, 1);
    let t_search = t1.elapsed().as_secs_f64();
    println!("search: {evals} renders in {t_search:.1}s");
    let full = (0, 0, size.0, size.1);
    let mdl_net = rho_sum(&search::residual(&State::new(objs.clone()), &ctx, full));
    let mdl_search = rho_sum(&search::residual(&st, &ctx, full));
    println!("residual nats: net {mdl_net:.0}, net+search {mdl_search:.0}");

    // The model image and residual.
    let cmp = Compare { size, ss: 2 };
    let cast = cmp.cast(&st.caster(), &cam, None);
    let model = appearance(&cast, &target);
    let ids: Vec<u16> = (0..size.0 * size.1).map(|k| {
        let (x, y) = (k % size.0, k / size.0);
        cast.id[(y * 2) * size.0 * 2 + x * 2]
    }).collect();

    // Colours from the photo for the re-render.
    let kept: Vec<usize> = (0..st.objs.len()).filter(|&i| st.present(i)).collect();
    let mut mean = vec![[0.0f64; 3]; st.objs.len()];
    let mut cnt = vec![0.0f64; st.objs.len()];
    let mut tmean = [0.0f64; 3];
    let mut tc: f64 = 0.0;
    for (k, &id) in ids.iter().enumerate() {
        if id == BACKDROP {
            continue;
        }
        if id == TABLE {
            for c in 0..3 {
                tmean[c] += target[k][c];
            }
            tc += 1.0;
        } else {
            for c in 0..3 {
                mean[id as usize][c] += target[k][c];
            }
            cnt[id as usize] += 1.0;
        }
    }
    let bright = mean.iter().zip(&cnt).filter(|(_, n)| **n > 0.0).map(|(m, n)| m.iter().cloned().fold(0.0, f64::max) / n).fold(tmean.iter().cloned().fold(0.0, f64::max) / tc.max(1.0), f64::max).max(1e-3);
    let k_alb = 0.85 / bright;
    let mut scene_objs = Vec::new();
    for &i in &kept {
        let mut ob = st.objs[i].clone();
        let n = cnt[i].max(1.0);
        ob.mat = Mat { rgb: [mean[i][0] / n * k_alb, mean[i][1] / n * k_alb, mean[i][2] / n * k_alb].map(|v| v.clamp(0.01, 0.9)), rough: 0.45, metal: false };
        scene_objs.push(ob);
    }
    let table = Mat { rgb: [tmean[0] / tc.max(1.0) * k_alb, tmean[1] / tc.max(1.0) * k_alb, tmean[2] / tc.max(1.0) * k_alb].map(|v| v.clamp(0.01, 0.9)), rough: 0.7, metal: false };
    let light = Lighting { zenith: [0.9, 0.9, 0.9], horizon: [0.5, 0.5, 0.5], window: [90.0, 30.0, 25.0, 3.0], key: [200.0, 60.0, 0.0], key_radius: 8.0, key_radiance: 1.2 };
    let rs = Scene { objects: scene_objs.clone(), table, cam, light, seed: 7, post: [0.0, 1.0, 95.0, 0.0], clutter: Vec::new() };
    let big = (512, 384);
    let ctx_gpu = irys_diff::gpu::context().expect("gpu");
    let meshes = irys_recon_net::render::world_meshes(&scene_objs);
    let rr = render_linear(ctx_gpu, &rs, &meshes, big, 64).unwrap_or_else(|e| {
        eprintln!("re-render: {e}");
        vec![[0.0; 3]; big.0 * big.1]
    });
    let mut l: Vec<f64> = rr.iter().map(|c| (c[0] + c[1] + c[2]) / 3.0).collect();
    l.sort_by(f64::total_cmp);
    let expo = 0.85 / l[l.len() * 98 / 100].max(1e-6);

    // The figure: 2 × 2 panels of 512 × 384.
    let (fw, fh) = big;
    let mut fig = image::RgbImage::new(2 * fw as u32, 2 * fh as u32);
    let photo_big = image::imageops::resize(&img, fw as u32, fh as u32, image::imageops::FilterType::Triangle);
    let up = |x: usize, y: usize| (x * size.0 / fw, y * size.1 / fh);
    for y in 0..fh {
        for x in 0..fw {
            let p = photo_big.get_pixel(x as u32, y as u32);
            fig.put_pixel(x as u32, y as u32, *p);
            let r = rr[y * fw + x];
            fig.put_pixel((x + fw) as u32, y as u32, image::Rgb([srgb(r[0] * expo), srgb(r[1] * expo), srgb(r[2] * expo)]));
            let (sx, sy) = up(x, y);
            let k = sy * size.0 + sx;
            let res = (0..3).map(|c| (target[k][c] - model[k][c]).abs()).sum::<f64>() / 3.0;
            let v = srgb(res * 1.5);
            fig.put_pixel(x as u32, (y + fh) as u32, image::Rgb([v, v, v]));
            let id = ids[k];
            let base = *p;
            let px = if id == TABLE || id == BACKDROP {
                image::Rgb([base[0] / 2, base[1] / 2, base[2] / 2])
            } else {
                let c = search::colour(id);
                image::Rgb([((base[0] as u16 + c[0] as u16 * 2) / 3) as u8, ((base[1] as u16 + c[1] as u16 * 2) / 3) as u8, ((base[2] as u16 + c[2] as u16 * 2) / 3) as u8])
            };
            fig.put_pixel((x + fw) as u32, (y + fh) as u32, px);
        }
    }
    // Labels at each object's base centre.
    let cam_fig = cam.resized(big);
    let mut inv = Vec::new();
    for (n, &i) in kept.iter().enumerate() {
        let ob = &st.objs[i];
        let num = n + 1;
        if let Some(p) = cam_fig.project([ob.pose[0], ob.pose[1], ob.height() * 0.5]) {
            let (x, y) = (p[0] as i64 + fw as i64, p[1] as i64 + fh as i64);
            text(&mut fig, x - 1, y - 1, &num.to_string(), 3, [0, 0, 0]);
            text(&mut fig, x, y, &num.to_string(), 3, [255, 255, 255]);
        }
        let f = &fits[i];
        let names = ob.family.param_names();
        let mut dims = Vec::new();
        // σ of each fitted log parameter → mm.
        let lead = if ob.family.has_yaw() { 3 } else { 2 };
        let mut j = lead;
        for k in 0..ob.params.len() {
            if param_used(ob.family, &ob.tokens, k) {
                let fixed = matches!((ob.family, k), (irys_recon_net::scene::Family::Grid, 3));
                let s = if fixed { 0.0 } else {
                    let v = f.sigma.get(j).copied().unwrap_or(f64::NAN) * ob.params[k];
                    j += 1;
                    v
                };
                dims.push(format!("{} {:.0} ± {:.0}", names[k], ob.params[k], s));
            }
        }
        let pi = i;
        let p = &kept_props[pi];
        inv.push(serde_json::json!({
            "n": num, "class": ob.class.name(), "class_p": p.cls[ob.class as usize], "family": ob.family.name(),
            "program": ob.to_text(), "pose_mm_deg": ob.pose, "dims": dims, "gauge": gauges.contains(&pi),
            "mdl": f.mdl, "residual": f.residual, "length": f.length,
        }));
    }
    fig.save(out.join("desk.png")).unwrap();

    // Unexplained regions: large residual where nothing is modelled.
    let thr = 0.15;
    let mut un = vec![false; size.0 * size.1];
    for k in 0..un.len() {
        let res = (0..3).map(|c| (target[k][c] - model[k][c]).abs()).sum::<f64>() / 3.0;
        un[k] = (ids[k] == TABLE || ids[k] == BACKDROP) && res > thr;
    }
    let frac = un.iter().filter(|&&b| b).count() as f64 / un.len() as f64;

    let mut md = String::new();
    md += "# Desk photo: neural recon inventory\n\n";
    md += &format!("Photo `{}` ({pw}×{ph}, f35 {f35} mm, horizontal FOV {hfov:.1}°). Net input {W}×{H}; search at {}×{}.\n\n", photo.file_name().unwrap().to_string_lossy(), size.0, size.1);
    md += &format!("- Net: {} detections in {:.3} s. Support plane from the plane head: camera {:.0} mm above it, pitch {:.1}°.\n", objs.len(), t_net, cam.eye[2], plane_net[1].to_degrees());
    md += &format!("- Scale: {}.\n", if gauges.is_empty() { "no known-size object was recognised with confidence; the scale is the plane head's alone".to_string() } else { format!("×{scale:.3} from {}", gauges.iter().map(|&i| format!("{} (#{})", objs[i].class.name(), i + 1)).collect::<Vec<_>>().join(", ")) });
    md += &format!("- MDL search: {evals} irys renders in {t_search:.1} s; kept {} of {} detections; full-frame residual {mdl_net:.0} → {mdl_search:.0} nats.\n", kept.len(), objs.len());
    md += &format!("- Unexplained: {:.1} % of the frame is table-labelled yet differs from the model by more than {thr} (linear).\n\n", 100.0 * frac);
    md += "| # | class (p) | family | program | x, y (mm), yaw (°) | dimensions mm ± σ | gauge |\n|---|---|---|---|---|---|---|\n";
    for e in &inv {
        md += &format!(
            "| {} | {} ({:.2}) | {} | `{}` | {:.0}, {:.0}, {:.0} | {} | {} |\n",
            e["n"], e["class"].as_str().unwrap(), e["class_p"].as_f64().unwrap(), e["family"].as_str().unwrap(), e["program"].as_str().unwrap(),
            e["pose_mm_deg"][0].as_f64().unwrap(), e["pose_mm_deg"][1].as_f64().unwrap(), e["pose_mm_deg"][2].as_f64().unwrap(),
            e["dims"].as_array().unwrap().iter().map(|d| d.as_str().unwrap().to_string()).collect::<Vec<_>>().join(", "),
            if e["gauge"].as_bool().unwrap() { "yes" } else { "" }
        );
    }
    std::fs::write(out.join("inventory.md"), md).unwrap();
    std::fs::write(out.join("inventory.json"), serde_json::to_string_pretty(&serde_json::json!({"objects": inv, "net_seconds": t_net, "search_seconds": t_search, "residual_net": mdl_net, "residual_search": mdl_search, "unexplained_frac": frac, "camera": cam})).unwrap()).unwrap();
    let _ = Cam::from_plane;
    println!("wrote {}", out.display());
}
