//! v3, piece 3: is the shortest description the right part? An oracle
//! study that takes the network out of it. For held-out scenes, with the
//! true camera and every other object at its truth, each object is coded by
//!
//! - its true program (fitted from the truth, and from a 12 %-perturbed start),
//! - every other family's default sized to its footprint and height
//!   (the v0/v1-style alternatives), and its own family with each other
//!   valid structure token,
//!
//! each fitted by v3's LM, and the MDL terms recorded with the error. The
//! question is whether `argmin MDL` is the truth, and if not, which term
//! (residual, Fisher, length) prefers the wrong part.
//!
//! `neural_v3_oracle --set v3-a --scenes 10 [--temp 1] [--decorrelate 1]`

use std::path::PathBuf;

use irys_recon_net::data::{hi_path, read_dir};
use irys_recon_net::edgejac::EdgeOpts;
use irys_recon_net::eval::object_error;
use irys_recon_net::faithful::{self, Appearance, Compare3, FitOpts, linearise, local_meshes};
use irys_recon_net::infer::{repair, valid_token};
use irys_recon_net::render::resample_rgb8;
use irys_recon_net::scene::{CLASS_LIST, Object, VOCAB, structure_nats, token_mask};
use irys_recon_net::search::{State, noise_sigma, window};
use irys_recon_net::search3::{self, Counts, Ctx3};

fn arg<T: std::str::FromStr>(a: &[String], k: &str, d: T) -> T {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// Alternatives to `t`: other families sized to it, other tokens of its own.
fn alternatives(t: &Object) -> Vec<Object> {
    let mut out = Vec::new();
    let s = 2.0 * t.radius();
    let h = t.height();
    let mut r = irys_recon::jac::Mix(t.params.iter().map(|v| v.to_bits()).fold(7, |a, b| a ^ b));
    for c in CLASS_LIST {
        if c.family() == t.family {
            continue;
        }
        if out.iter().any(|o: &Object| o.family == c.family()) {
            continue;
        }
        let mut o = irys_recon_net::scene::sample_object(&mut r, c);
        // Size it to the truth's footprint and height.
        let (os, oh) = (2.0 * o.radius(), o.height());
        for (k, v) in o.params.iter_mut().enumerate() {
            let is_h = matches!((c.family(), k), (irys_recon_net::scene::Family::Lathe, 0) | (_, 2));
            *v *= if is_h { h / oh } else { s / os };
        }
        o.pose = t.pose;
        o.mat = t.mat;
        repair(&mut o);
        if o.valid() {
            out.push(o);
        }
    }
    for k in 1..4 {
        if !t.token_mask()[k] {
            continue;
        }
        for v in 0..VOCAB {
            if v == t.tokens[k] || !valid_token(t.family, k, v) {
                continue;
            }
            let mut o = t.clone();
            o.tokens[k] = v;
            if !token_mask(o.family, &o.tokens)[k] {
                continue;
            }
            repair(&mut o);
            if o.valid() && (v as i64 - t.tokens[k] as i64).abs() <= 1 {
                out.push(o);
            }
        }
    }
    out
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = PathBuf::from(arg(&a, "--data", "out/neural/data".to_string()));
    let set: String = arg(&a, "--set", "v3-a".to_string());
    let scenes: usize = arg(&a, "--scenes", 10);
    let skip: usize = arg(&a, "--skip", 0);
    let sw: usize = arg(&a, "--size", 512);
    let spp: u32 = arg(&a, "--spp", 4);
    let iters: usize = arg(&a, "--iters", 6);
    let temp: f64 = arg(&a, "--temp", 1.0);
    let decor = arg(&a, "--decorrelate", 1) == 1;
    let recolour = arg(&a, "--recolour", 1) == 1;
    let shade = arg(&a, "--shade", 1) == 1;
    let highpass: f64 = arg(&a, "--highpass", 2.5);
    let pose_sd: f64 = arg(&a, "--pose-sd", 30.0);
    let fd_weak = arg(&a, "--fd-weak", 1) == 1;
    let tag: String = arg(&a, "--tag", "base".to_string());
    let true_app = arg(&a, "--true-app", 0) == 1;
    let dump = arg(&a, "--dump", 0) == 1;
    let combos = arg(&a, "--combos", 0) == 1;
    let levels: Vec<f64> = arg(&a, "--levels", "4,2,1".to_string()).split(',').map(|x| x.parse().unwrap()).collect();
    let out = PathBuf::from(arg(&a, "--out", "out/neural/v3".to_string()));
    std::fs::create_dir_all(&out).unwrap();
    let size = (sw, sw * 3 / 4);
    let gpu = irys_diff::gpu::context().expect("gpu");
    let samples: Vec<_> = read_dir(&data, &set).into_iter().skip(skip).take(scenes).collect();
    let mut rows = Vec::new();
    let (mut right, mut total) = (0, 0);
    for s in &samples {
        let hi = image::open(hi_path(&data, &set, s.scene.seed)).unwrap().to_rgb8();
        let rgb8: Vec<u8> = resample_rgb8(&hi, size).iter().flat_map(|p| p.map(irys_recon::render::srgb8)).collect();
        let cam = s.scene.cam;
        let truth = s.scene.objects.clone();
        // Appearance fitted with the true geometry (at 256 px).
        let fsz = (256, 192);
        let rgb_f: Vec<u8> = resample_rgb8(&hi, fsz).iter().flat_map(|p| p.map(irys_recon::render::srgb8)).collect();
        let mut app = Appearance::grey(truth.len());
        let mut cmp = Compare3::new(gpu, &truth, &local_meshes(&truth), &app, &cam, fsz).unwrap();
        faithful::fit_appearance(&mut cmp, &cam, &rgb_f, &mut app, &FitOpts::default());
        let mut exact = None;
        if true_app {
            app.mats = truth.iter().map(|o| o.mat).collect();
            app.table = s.scene.table;
            app.gamma = s.scene.post[1];
            app.defocus = s.scene.post[0] * 8.0 / 1024.0;
            let (e, sun) = irys_recon_net::render::true_light(&s.scene);
            app.key.dir = sun.dir;
            exact = Some((e, sun, 1.0));
        }
        let target = linearise(&rgb8, app.gamma);
        let sigma = noise_sigma(&target, size);
        let ctx = Ctx3 { gpu, target: &target, size, cam, app, spp, seed: 3, sigma, sig_levels: Vec::new(), levels: levels.clone(), prior_temp: temp, decorrelate: decor, edge: EdgeOpts::default(), exact, recolour, shade, fd_weak, tau: 1.0, param_sd: Vec::new(), highpass, pose_sd };
        let ctx = if let Some((e, sun, _)) = &ctx.exact {
            // Exposure: one scale.
            let m = search3::render(&State::new(truth.clone()), &ctx);
            let k = target.iter().zip(&m).filter(|(t, _)| t.iter().all(|&v| v < faithful::SAT)).map(|(t, m)| t[0] * m[0] + t[1] * m[1] + t[2] * m[2]).sum::<f64>() / target.iter().zip(&m).filter(|(t, _)| t.iter().all(|&v| v < faithful::SAT)).map(|(_, m)| m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sum::<f64>();
            Ctx3 { exact: Some((e.clone(), *sun, k)), ..ctx }
        } else {
            ctx
        };
        let ctx = Ctx3 { sig_levels: search3::noise_levels(&State::new(truth.clone()), &ctx), ..ctx };
        let st = State::new(truth.clone());
        if combos {
            // The residual at the true geometry under each nuisance model:
            // which fitted part of the appearance costs the most.
            let fitted = ctx.app.clone();
            let (e, sun) = irys_recon_net::render::true_light(&s.scene);
            let mut tm = fitted.clone();
            tm.mats = truth.iter().map(|o| o.mat).collect();
            tm.table = s.scene.table;
            let mut tt = fitted.clone();
            tt.gamma = s.scene.post[1];
            tt.defocus = s.scene.post[0] * 8.0 / 1024.0;
            let mut all = tm.clone();
            all.gamma = tt.gamma;
            all.defocus = tt.defocus;
            let variants: Vec<(&str, Appearance, bool)> = vec![
                ("fitted", fitted.clone(), false),
                ("true light", fitted.clone(), true),
                ("true mats", tm.clone(), false),
                ("true tone", tt.clone(), false),
                ("true light+mats", tm.clone(), true),
                ("all true", all, true),
            ];
            let mut line = format!("seed {} residual at truth:", s.scene.seed);
            let mut rec = serde_json::json!({"seed": s.scene.seed});
            for (name, app2, ex) in variants {
                let t2 = linearise(&rgb8, app2.gamma);
                let mut c2 = Ctx3 { gpu, target: &t2, size, cam, app: app2, spp, seed: 3, sigma: ctx.sigma, sig_levels: ctx.sig_levels.clone(), levels: ctx.levels.clone(), prior_temp: temp, decorrelate: decor, edge: EdgeOpts::default(), exact: None, recolour, shade, fd_weak, tau: 1.0, param_sd: Vec::new(), highpass, pose_sd };
                if ex {
                    c2.app.key.dir = sun.dir;
                    c2.exact = Some((e.clone(), sun, 1.0));
                    let m = search3::render(&st, &c2);
                    let ok = |t: &[f64; 3]| t.iter().all(|&v| v < faithful::SAT);
                    let k = t2.iter().zip(&m).filter(|(t, _)| ok(t)).map(|(t, m)| t[0] * m[0] + t[1] * m[1] + t[2] * m[2]).sum::<f64>()
                        / t2.iter().zip(&m).filter(|(t, _)| ok(t)).map(|(_, m)| m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sum::<f64>();
                    c2.exact = Some((e.clone(), sun, k));
                }
                let r = search3::scene_mdl(&truth, &c2) - truth.iter().map(irys_recon_net::scene::program_length).sum::<f64>();
                line += &format!("  {name} {r:.0}");
                rec[name] = serde_json::json!(r);
            }
            println!("{line}");
            rows.push(rec);
            std::fs::write(out.join(format!("oracle-{tag}.json")), serde_json::to_string_pretty(&rows).unwrap()).unwrap();
            continue;
        }
        if dump {
            let m = search3::render(&st, &ctx);
            let mb = faithful::blur_px(&m, size, ctx.model_blur(0.0));
            let g = ctx.app.gamma;
            let to8 = |img: &[[f64; 3]]| -> Vec<u8> { img.iter().flat_map(|p| p.map(|v| (faithful::display(v, g) * 255.0 + 0.5).clamp(0.0, 255.0) as u8)).collect() };
            let d: Vec<u8> = mb.iter().zip(&target).flat_map(|(a, b)| (0..3).map(move |c| (128.0 + 600.0 * (a[c] - b[c])).clamp(0.0, 255.0) as u8)).collect();
            let im = |v: Vec<u8>| image::RgbImage::from_raw(size.0 as u32, size.1 as u32, v).unwrap();
            im(to8(&mb)).save(out.join(format!("oracle-{tag}-{}-model.png", s.scene.seed))).unwrap();
            im(rgb8.clone()).save(out.join(format!("oracle-{tag}-{}-photo.png", s.scene.seed))).unwrap();
            im(d).save(out.join(format!("oracle-{tag}-{}-resid.png", s.scene.seed))).unwrap();
            println!("sig levels {:?} sigma {}", ctx.sig_levels, ctx.sigma);
        }
        for (i, t) in truth.iter().enumerate() {
            let c = cam.resized(size);
            let Some(win) = window(&[t], &c, size, 10.0) else { continue };
            if (win.2 - win.0) * (win.3 - win.1) < 400 {
                continue;
            }
            let mut n = Counts::default();
            let mut cands: Vec<(String, Object)> = vec![("truth".into(), t.clone())];
            let mut pert = t.clone();
            let mut r = irys_recon::jac::Mix(i as u64 + 99);
            for v in pert.params.iter_mut() {
                *v *= 1.0 + 0.12 * (2.0 * r.uniform() - 1.0);
            }
            pert.pose[0] += 6.0 * (2.0 * r.uniform() - 1.0);
            pert.pose[1] += 6.0 * (2.0 * r.uniform() - 1.0);
            repair(&mut pert);
            cands.push(("truth+12%".into(), pert));
            for o in alternatives(t) {
                cands.push(("alt".into(), o));
            }
            let mut fits = Vec::new();
            let f0 = search3::fit(&st, i, t, structure_nats(t), &ctx, win, 0, &mut n);
            let unfit = serde_json::json!({"mdl": f0.mdl, "residual": f0.residual, "param_cost": f0.param_cost});
            for (kind, o) in &cands {
                let f = search3::fit(&st, i, o, structure_nats(o), &ctx, win, iters, &mut n);
                let (topo, err) = object_error(f.object.as_ref().unwrap(), t);
                fits.push(serde_json::json!({"kind": kind, "text": f.object.as_ref().unwrap().to_text(), "mdl": f.mdl, "residual": f.residual, "param_cost": f.param_cost, "length": f.length, "topo": topo, "err": err}));
            }
            let d = search3::dropped(&st, i, &ctx, win, &mut n);
            let best = fits.iter().min_by(|x, y| x["mdl"].as_f64().unwrap().total_cmp(&y["mdl"].as_f64().unwrap())).unwrap();
            let ok = best["topo"].as_bool().unwrap();
            total += 1;
            right += ok as usize;
            println!(
                "seed {} {:>14}: best {:>9} topo {} err {:.3} | unfit {:.0} | truth mdl {:.0} res {:.0} | best mdl {:.0} res {:.0} | drop {:.0}",
                s.scene.seed,
                t.class.name(),
                best["kind"].as_str().unwrap(),
                ok,
                best["err"].as_f64().unwrap(),
                f0.mdl,
                fits[0]["mdl"].as_f64().unwrap(),
                fits[0]["residual"].as_f64().unwrap(),
                best["mdl"].as_f64().unwrap(),
                best["residual"].as_f64().unwrap(),
                d.mdl
            );
            rows.push(serde_json::json!({"seed": s.scene.seed, "truth": t.to_text(), "drop": d.mdl, "unfit": unfit, "fits": fits}));
        }
        std::fs::write(out.join(format!("oracle-{tag}.json")), serde_json::to_string_pretty(&rows).unwrap()).unwrap();
    }
    println!("argmin MDL has the true structure: {right}/{total} ({:.0} %)", 100.0 * right as f64 / total.max(1) as f64);
}
