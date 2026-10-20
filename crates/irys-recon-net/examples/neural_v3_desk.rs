//! v3's desk demo: one photo in, a parametric inventory out, through the
//! faithful compare image.
//! `neural_v3_desk --photo desk.jpg --weights out/neural/v3/desk-run/weights.json --out out/neural/v3/desk`
//!
//! The net reads the photo at 128 × 96; the appearance (world, key, tone,
//! substances) is fitted at 256 × 192; the MDL search runs at 512 × 384 on
//! irys's OpenPBR render. Writes `desk.png` (photo | irys re-render with the
//! fitted appearance | residual | labelled inventory), `topview.png` (the
//! inventory from above, rendered in the same submit as the re-render through
//! M1's batched cameras), `inventory.md` and `inventory.json`.

use std::path::PathBuf;

use irys_recon_net::data::patches;
use irys_recon_net::edgejac::EdgeOpts;
use irys_recon_net::faithful::{self, Appearance, Compare3, FitOpts, display, linearise, local_meshes};
use irys_recon_net::infer::net_scene;
use irys_recon_net::net::{H, W};
use irys_recon_net::render::resample_rgb8;
use irys_recon_net::scene::{Cam, Family, Mat, Object, param_used};
use irys_recon_net::search::{State, colour, noise_sigma};
use irys_recon_net::search3::{self, Ctx3};
use irys_recon_net::train::{Infer, Weights};

fn arg<T: std::str::FromStr>(a: &[String], k: &str, d: T) -> T {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(d)
}

const GREY: Mat = Mat { rgb: [0.5; 3], rough: 0.5, metal: false };

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
    let wpath = PathBuf::from(arg(&a, "--weights", "out/neural/run2/weights.json".to_string()));
    let out = PathBuf::from(arg(&a, "--out", "out/neural/v3/desk".to_string()));
    let iters: usize = arg(&a, "--iters", 6);
    let thr: f64 = arg(&a, "--thr", 0.3);
    let births_on = arg(&a, "--births", 1) == 1;
    let temp: f64 = arg(&a, "--temp", 0.1);
    let highpass: f64 = arg(&a, "--highpass", 2.5);
    std::fs::create_dir_all(&out).unwrap();
    let gpu = irys_diff::gpu::context().expect("gpu");
    let bytes = std::fs::read(&photo).unwrap();
    let exif = irys_recon::tag::read_exif(&bytes);
    let img = image::load_from_memory(&bytes).unwrap().to_rgb8();
    let (pw, ph) = (img.width() as usize, img.height() as usize);
    let f35 = exif.f35.unwrap_or(26.0);
    let f_px = f35 / 36f64.hypot(24.0) * (pw as f64).hypot(ph as f64);
    let hfov = 2.0 * (pw as f64 / 2.0 / f_px).atan().to_degrees();
    println!("{}: {pw}x{ph}, f35 {f35} mm → horizontal FOV {hfov:.1}°", photo.display());

    // The net at its own size.
    let small: Vec<u8> = resample_rgb8(&img, (W, H)).iter().flat_map(|p| p.map(irys_recon::render::srgb8)).collect();
    let w = Weights::load(&wpath).unwrap();
    let mut net = Infer::new(&w, 1);
    let t0 = std::time::Instant::now();
    let o = net.run(&[patches(&small)]).pop().unwrap();
    let props = irys_recon_net::infer::proposals(&o, thr);
    let _ = props;
    let (objs, alts, cam, gauges, sds) = net_scene(&o, hfov, 3, GREY);
    let t_net = t0.elapsed().as_secs_f64();
    let plain: Vec<Object> = objs.iter().map(|x| x.0.clone()).collect();
    println!("net: {} detections in {:.3}s; camera {:.0} mm up, pitch {:.1}°; gauges {:?}", objs.len(), t_net, cam.eye[2], o.plane[1].to_degrees(), gauges.iter().map(|&i| plain[i].class.name()).collect::<Vec<_>>());
    for (i, ob) in plain.iter().enumerate() {
        println!("  {i:2} {}", ob.to_text());
    }

    // Appearance at 256 × 192.
    let fsz = (256, 192);
    let rgb_f: Vec<u8> = resample_rgb8(&img, fsz).iter().flat_map(|p| p.map(irys_recon::render::srgb8)).collect();
    let mut app = Appearance::grey(plain.len());
    let mut cmp = Compare3::new(gpu, &plain, &local_meshes(&plain), &app, &cam, fsz).unwrap();
    let rep = faithful::fit_appearance(&mut cmp, &cam, &rgb_f, &mut app, &FitOpts::default());
    println!("appearance: {} renders, {:.1}s, γ {:.2}, defocus {:.4} of the width", rep.renders, rep.seconds, app.gamma, app.defocus);

    // The search at 512 × 384.
    let size = (512, 384);
    let rgb8: Vec<u8> = resample_rgb8(&img, size).iter().flat_map(|p| p.map(irys_recon::render::srgb8)).collect();
    let target = linearise(&rgb8, app.gamma);
    let sigma = noise_sigma(&target, size);
    let ctx = Ctx3 {
        gpu,
        target: &target,
        size,
        cam,
        app: app.clone(),
        spp: 16,
        seed: 3,
        sigma,
        sig_levels: Vec::new(),
        levels: vec![4.0, 2.0, 1.0],
        prior_temp: temp,
        decorrelate: true,
        edge: EdgeOpts::default(),
        recolour: true,
        shade: true,
        fd_weak: true,
        tau: 1.0,
        param_sd: sds.clone(),
        pose_sd: 30.0,
        highpass,
        exact: None,
    };
    let st0 = State::new(plain.clone());
    let ctx = Ctx3 { sig_levels: search3::noise_levels(&st0, &ctx), ..ctx };
    let ctx = Ctx3 { tau: search3::dispersion(&st0, &ctx), ..ctx };
    let mdl_net = search3::scene_mdl(&plain, &ctx);
    let t1 = std::time::Instant::now();
    let (st, mut fits, _all, counts) = search3::search(objs.clone(), &alts, &ctx, iters, 1);
    let t_search = t1.elapsed().as_secs_f64();
    // Births: what the net missed, from the residual.
    let t2 = std::time::Instant::now();
    let n_before = st.objs.len();
    let (st, born) = if births_on { search3::births(st, &ctx, 4, 3.0, 150, 10) } else { (st, Vec::new()) };
    for (_, f) in &born {
        fits.push(f.clone());
    }
    let t_births = t2.elapsed().as_secs_f64();
    println!("births: {} new objects in {t_births:.0}s", born.len());
    for (i, f) in &born {
        println!("  + {i:2} {}", f.object.as_ref().unwrap().to_text());
    }
    let mut ctx = ctx;
    ctx.app.mats.resize(st.objs.len(), GREY);
    ctx.param_sd.resize(st.objs.len(), Vec::new());
    let ctx = ctx;
    // Gauges: a can is 122 mm tall; a keyboard's keys are 19.05 mm apart.
    let mut gauge_scales: Vec<(String, f64)> = Vec::new();
    for i in (0..st.objs.len()).filter(|&i| st.present(i)) {
        let ob = &st.objs[i];
        if ob.family == Family::Lathe {
            let rmax = ob.params[1..5].iter().cloned().fold(0.0, f64::max);
            let ratio = ob.params[0] / rmax.max(1e-6);
            if (3.0..=4.6).contains(&ratio) && ob.params[0] > 20.0 {
                gauge_scales.push((format!("can #{} (h {:.0} mm, h/r {:.1})", i, ob.params[0], ratio), 122.0 / ob.params[0]));
            }
        }
        if ob.family == Family::Grid {
            // Scale the whole scene (and the camera) by s with the key pitch
            // held at 19.05 mm: only the key pattern moves; the best s is the
            // gauge's reading.
            let cs = cam.resized(size);
            let Some(win) = irys_recon_net::search::window(&[ob], &cs, size, 8.0) else { continue };
            let mut best = (f64::INFINITY, 1.0);
            for k in 0..25 {
                let sc = 0.55 * (1.9f64 / 0.55).powf(k as f64 / 24.0);
                let mut os = st.objs.clone();
                let mut c2 = cam;
                irys_recon_net::infer::rescale(&mut os, &mut c2, sc);
                os[i].params[3] = 19.05;
                let mut s2 = st.clone();
                for (j, o) in os.iter().enumerate() {
                    if st.present(j) {
                        s2 = s2.with(j, Some(o));
                    }
                }
                let cx = Ctx3 { cam: c2, target: &target, app: ctx.app.clone(), sig_levels: ctx.sig_levels.clone(), levels: ctx.levels.clone(), param_sd: Vec::new(), exact: None, ..ctx };
                let m = search3::render(&s2, &cx);
                let e = search3::rho_sum(&search3::residual_of(&m, &cx, win, 1.0));
                if e < best.0 {
                    best = (e, sc);
                }
            }
            gauge_scales.push((format!("keyboard #{} (key pitch at scale {:.2})", i, best.1), best.1));
        }
    }
    let gauge = if gauge_scales.is_empty() {
        1.0
    } else {
        let mut v: Vec<f64> = gauge_scales.iter().map(|g| g.1).collect();
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    println!("gauges: {gauge_scales:?} → scale {gauge:.3}");
    let _ = n_before;
    let kept: Vec<usize> = (0..st.objs.len()).filter(|&i| st.present(i)).collect();
    let mdl_search = search3::scene_mdl(&st.kept(), &ctx);
    println!("search: {} builds, {} TLAS moves, {} renders, {} Jacobians in {t_search:.1}s; scene MDL {mdl_net:.0} → {mdl_search:.0}", counts.builds, counts.moves, counts.renders, counts.jacobians);

    // The re-render (photo's camera) and a view from above, one submit.
    let mut mats = Vec::new();
    {
        // Each kept object's colour from its own pixels in the photo.
        let caster = st.caster();
        let cs = cam.resized(size);
        let mut num = vec![[0.0f64; 3]; st.objs.len()];
        let mut cnt = vec![0.0f64; st.objs.len()];
        for y in 0..size.1 {
            for x in 0..size.0 {
                let h = caster.cast(&cs, x as f64 + 0.5, y as f64 + 0.5);
                if (h.0 as usize) < st.objs.len() {
                    for c in 0..3 {
                        num[h.0 as usize][c] += target[y * size.0 + x][c];
                    }
                    cnt[h.0 as usize] += 1.0;
                }
            }
        }
        let t_mean: f64 = target.iter().map(|p| (p[0] + p[1] + p[2]) / 3.0).sum::<f64>() / target.len() as f64;
        let tb = (app.table.rgb.iter().sum::<f64>() / 3.0).max(1e-3);
        for i in 0..st.objs.len() {
            let n = cnt[i].max(1.0);
            // Albedo ∝ the pixels' colour, scaled like the table's.
            let k = tb / t_mean.max(1e-3);
            mats.push(Mat { rgb: [num[i][0] / n * k, num[i][1] / n * k, num[i][2] / n * k].map(|v| v.clamp(0.01, 0.95)), rough: 0.5, metal: false });
        }
    }
    let mut app2 = app.clone();
    app2.mats = mats;
    let objs_k: Vec<Object> = st.objs.clone();
    let meshes: Vec<_> = local_meshes(&objs_k).into_iter().enumerate().map(|(i, m)| if st.present(i) { m } else { Default::default() }).collect();
    let big = (512, 384);
    let mut c2 = Compare3::new(gpu, &objs_k, &meshes, &app2, &cam, big).unwrap();
    c2.set_appearance(&app2);
    // Above: looking down at the inventory's centre from 1.6 m.
    let (cx, cy) = kept.iter().map(|&i| (st.objs[i].pose[0], st.objs[i].pose[1])).fold((0.0, 0.0), |a, b| (a.0 + b.0 / kept.len().max(1) as f64, a.1 + b.1 / kept.len().max(1) as f64));
    let top = Cam { eye: [cx, cy - 300.0, 1600.0], at: [cx, cy, 0.0], hfov: 60.0, size: big };
    let views = c2.render_batch(&[cam.resized(big), top], 64, 9);
    let rr = faithful::blur_px(&views[0], big, app.defocus * big.0 as f64);
    let to8 = |v: f64| (display(v, app.gamma) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
    let mut topimg = image::RgbImage::new(big.0 as u32, big.1 as u32);
    let mut l: Vec<f64> = views[1].iter().map(|c| (c[0] + c[1] + c[2]) / 3.0).collect();
    l.sort_by(f64::total_cmp);
    let expo = 0.8 / l[l.len() * 98 / 100].max(1e-6);
    for (k, p) in views[1].iter().enumerate() {
        topimg.put_pixel((k % big.0) as u32, (k / big.0) as u32, image::Rgb(p.map(|v| irys_recon::render::srgb8(v * expo))));
    }
    let tcam = top.resized(big);
    for (n, &i) in kept.iter().enumerate() {
        if let Some(p) = tcam.project([st.objs[i].pose[0], st.objs[i].pose[1], st.objs[i].height()]) {
            text(&mut topimg, p[0] as i64 - 1, p[1] as i64 - 1, &(n + 1).to_string(), 3, [0, 0, 0]);
            text(&mut topimg, p[0] as i64, p[1] as i64, &(n + 1).to_string(), 3, [255, 255, 255]);
        }
    }
    topimg.save(out.join("topview.png")).unwrap();

    // The figure: 2 × 2 panels.
    let (fw, fh) = big;
    let mut fig = image::RgbImage::new(2 * fw as u32, 2 * fh as u32);
    let photo_big: Vec<[f64; 3]> = resample_rgb8(&img, big);
    let caster = st.caster();
    let cb = cam.resized(big);
    for y in 0..fh {
        for x in 0..fw {
            let pp = photo_big[y * fw + x];
            let p8 = pp.map(irys_recon::render::srgb8);
            fig.put_pixel(x as u32, y as u32, image::Rgb(p8));
            let r = rr[y * fw + x];
            fig.put_pixel((x + fw) as u32, y as u32, image::Rgb(r.map(to8)));
            let tl = linearise(&p8, app.gamma)[0];
            let res = (0..3).map(|c| (tl[c] - r[c].min(1.0)).abs()).sum::<f64>() / 3.0;
            let v = irys_recon::render::srgb8(res * 1.5);
            fig.put_pixel(x as u32, (y + fh) as u32, image::Rgb([v, v, v]));
            let id = caster.cast(&cb, x as f64 + 0.5, y as f64 + 0.5).0;
            let px = if (id as usize) < st.objs.len() {
                let c = colour(id);
                image::Rgb([((p8[0] as u16 + c[0] as u16 * 2) / 3) as u8, ((p8[1] as u16 + c[1] as u16 * 2) / 3) as u8, ((p8[2] as u16 + c[2] as u16 * 2) / 3) as u8])
            } else {
                image::Rgb([p8[0] / 2, p8[1] / 2, p8[2] / 2])
            };
            fig.put_pixel((x + fw) as u32, (y + fh) as u32, px);
        }
    }
    let mut inv = Vec::new();
    for (n, &i) in kept.iter().enumerate() {
        let ob = &st.objs[i];
        let num = n + 1;
        if let Some(p) = cb.project([ob.pose[0], ob.pose[1], ob.height() * 0.5]) {
            let (x, y) = (p[0] as i64 + fw as i64, p[1] as i64 + fh as i64);
            text(&mut fig, x - 1, y - 1, &num.to_string(), 3, [0, 0, 0]);
            text(&mut fig, x, y, &num.to_string(), 3, [255, 255, 255]);
        }
        let f = &fits[i];
        let names = ob.family.param_names();
        let mut dims = Vec::new();
        let sl = search3::slots(ob);
        for k in 0..ob.params.len() {
            if param_used(ob.family, &ob.tokens, k) {
                let fixed = matches!((ob.family, k), (Family::Grid, 3));
                let s = if fixed { 0.0 } else { sl.iter().position(|x| x.0 == k).and_then(|j| f.sigma.get(j)).copied().unwrap_or(f64::NAN) * ob.params[k] };
                dims.push(format!("{} {:.0} ± {:.0}", names[k], ob.params[k] * gauge, s * gauge));
            }
        }
        inv.push(serde_json::json!({
            "n": num, "class": ob.class.name(), "family": ob.family.name(), "program": ob.to_text(), "pose_mm_deg": [ob.pose[0] * gauge, ob.pose[1] * gauge, ob.pose[2]], "born": i >= objs.len(),
            "dims": dims, "gauge": gauges.contains(&i), "mdl": f.mdl, "residual": f.residual, "length": f.length,
        }));
    }
    fig.save(out.join("desk.png")).unwrap();
    let mut md = String::new();
    md += "# Desk photo: v3 inventory\n\n";
    md += &format!("Photo `{}` ({pw}×{ph}, f35 {f35} mm, horizontal FOV {hfov:.1}°). Net input {W}×{H}; appearance fitted at 256×192; search at 512×384 on irys's OpenPBR render.\n\n", photo.file_name().unwrap().to_string_lossy());
    md += &format!("- Net (`{}`): {} detections in {:.3} s; support plane from the plane head: camera {:.0} mm above it, pitch {:.1}°.\n", wpath.display(), objs.len(), t_net, cam.eye[2], o.plane[1].to_degrees());
    md += &format!("- Net gauges: {}.\n", if gauges.is_empty() { "none recognised with confidence by the net".to_string() } else { format!("{}", gauges.iter().map(|&i| format!("{} (#{})", plain[i].class.name(), i + 1)).collect::<Vec<_>>().join(", ")) });
    md += &format!("- Births from the residual: {} objects in {t_births:.0} s.\n", born.len());
    md += &format!("- Scale after the search: ×{gauge:.3} from {}; every length below is scaled by it.\n", if gauge_scales.is_empty() { "nothing (no gauge found; the plane head's scale)".to_string() } else { gauge_scales.iter().map(|g| format!("{} → ×{:.3}", g.0, g.1)).collect::<Vec<_>>().join("; ") });
    md += &format!("- Appearance: {} renders in {:.1} s; tone γ {:.2}; defocus σ {:.2} px at 512.\n", rep.renders, rep.seconds, app.gamma, app.defocus * 512.0);
    md += &format!("- Search: {} shape rebuilds, {} TLAS-only moves, {} renders, {} edge Jacobians in {t_search:.0} s; kept {} of {}; scene MDL {mdl_net:.0} → {mdl_search:.0} nats (τ {:.1}).\n\n", counts.builds, counts.moves, counts.renders, counts.jacobians, kept.len(), objs.len(), ctx.tau);
    md += "| # | class | family | program (search's scale) | x, y (mm), yaw (°) | dimensions mm ± σ (gauge scale) | born |\n|---|---|---|---|---|---|---|\n";
    for e in &inv {
        md += &format!(
            "| {} | {} | {} | `{}` | {:.0}, {:.0}, {:.0} | {} | {} |\n",
            e["n"], e["class"].as_str().unwrap(), e["family"].as_str().unwrap(), e["program"].as_str().unwrap(),
            e["pose_mm_deg"][0].as_f64().unwrap(), e["pose_mm_deg"][1].as_f64().unwrap(), e["pose_mm_deg"][2].as_f64().unwrap(),
            e["dims"].as_array().unwrap().iter().map(|d| d.as_str().unwrap().to_string()).collect::<Vec<_>>().join(", "),
            if e["born"].as_bool().unwrap() { "yes" } else { "" }
        );
    }
    std::fs::write(out.join("inventory.md"), md).unwrap();
    std::fs::write(out.join("inventory.json"), serde_json::to_string_pretty(&serde_json::json!({"objects": inv, "net": plain.iter().map(|o| o.to_text()).collect::<Vec<_>>(), "net_seconds": t_net, "search_seconds": t_search, "mdl_net": mdl_net, "mdl_search": mdl_search, "camera": cam, "appearance": app})).unwrap()).unwrap();
    println!("wrote {}", out.display());
}
