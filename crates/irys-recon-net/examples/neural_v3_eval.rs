//! v3: the v2 tables with the faithful compare image. On a held-out set
//! with high-resolution photos (`neural_data --hi 4`):
//! `neural_v3_eval --set v3-a --n 50 --modes net,net+search,net+search@truecam,search`.
//!
//! Besides v2's scores it writes, per scene, every candidate the search
//! fitted (its MDL terms and its error against the matched truth) for the
//! MDL-vs-error study, and a paired dimension error: the same detections
//! before and after the search.

use std::path::PathBuf;

use irys_recon_net::data::{Sample, hi_path, patches, read_dir};
use irys_recon_net::edgejac::EdgeOpts;
use irys_recon_net::eval::{Score, canonical_truth, match_one, median, object_error, row, score};
use irys_recon_net::faithful::{self, Appearance, Compare3, FitOpts, linearise, local_meshes};
use irys_recon_net::infer::net_scene;
use irys_recon_net::render::resample_rgb8;
use irys_recon_net::scene::{Cam, Mat, Object, structure_nats};
use irys_recon_net::search::{baseline_candidates, noise_sigma, segment};
use irys_recon_net::search3::{self, Ctx3};
use irys_recon_net::train::{Infer, Weights};

fn arg<T: std::str::FromStr>(a: &[String], k: &str, d: T) -> T {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(d)
}

const GREY: Mat = Mat { rgb: [0.5; 3], rough: 0.5, metal: false };

/// Fit the appearance for `objs` seen through `cam` at the fitting size.
pub fn fit_app(gpu: &'static irys_gpu::GpuContext, hi: &image::RgbImage, objs: &[Object], cam: &Cam, fit_w: usize) -> Appearance {
    let size = (fit_w, fit_w * 3 / 4);
    let rgb8: Vec<u8> = resample_rgb8(hi, size).iter().flat_map(|p| p.map(irys_recon::render::srgb8)).collect();
    let mut app = Appearance::grey(objs.len());
    let meshes = local_meshes(objs);
    let mut cmp = Compare3::new(gpu, objs, &meshes, &app, cam, size).unwrap();
    faithful::fit_appearance(&mut cmp, cam, &rgb8, &mut app, &FitOpts::default());
    app
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = PathBuf::from(arg(&a, "--data", "out/neural/data".to_string()));
    let wpath = PathBuf::from(arg(&a, "--weights", "out/neural/run2/weights.json".to_string()));
    let set: String = arg(&a, "--set", "v3-a".to_string());
    let n: usize = arg(&a, "--n", 50);
    let skip: usize = arg(&a, "--skip", 0);
    let out = PathBuf::from(arg(&a, "--out", format!("out/neural/v3/eval-{set}")));
    let modes: String = arg(&a, "--modes", "net,net+search,net+search@truecam,search".to_string());
    let iters: usize = arg(&a, "--iters", 6);
    let sw: usize = arg(&a, "--size", 512);
    let spp: u32 = arg(&a, "--spp", 8);
    let temp: f64 = arg(&a, "--temp", 0.1);
    let decor = arg(&a, "--decorrelate", 1) == 1;
    let recolour = arg(&a, "--recolour", 1) == 1;
    let shade = arg(&a, "--shade", 1) == 1;
    let highpass: f64 = arg(&a, "--highpass", 2.5);
    let pose_sd: f64 = arg(&a, "--pose-sd", 30.0);
    let fd_weak = arg(&a, "--fd-weak", 1) == 1;
    let use_tau = arg(&a, "--tau", 1) == 1;
    let use_prior = arg(&a, "--prior", 1) == 1;
    let oracle_app = arg(&a, "--oracle-app", 0) == 1;
    let parts: String = arg(&a, "--oracle-parts", String::new());
    let refine_cam = arg(&a, "--refine-cam", 1) == 1;
    let prior_scale_on = arg(&a, "--prior-scale", 1) == 1;
    let k_alt: usize = arg(&a, "--k", 3);
    let levels: Vec<f64> = arg(&a, "--levels", "4,2,1".to_string()).split(',').map(|x| x.parse().unwrap()).collect();
    std::fs::create_dir_all(&out).unwrap();
    let gpu = irys_diff::gpu::context().expect("gpu");
    let size = (sw, sw * 3 / 4);
    let samples: Vec<Sample> = read_dir(&data, &set).into_iter().skip(skip).take(n).collect();
    let w = Weights::load(&wpath).unwrap();
    let mut net = Infer::new(&w, 8);
    let mut scores: Vec<(String, Score, f64, f64)> = modes.split(',').map(|m| (m.to_string(), Score::default(), 0.0, 0.0)).collect();
    let mut paired: Vec<(String, Vec<f64>, Vec<f64>)> = Vec::new();
    let mut per_scene = Vec::new();
    let mut raw_sum = vec![0.0f64; scores.len()];
    for (si, s) in samples.iter().enumerate() {
        let hi = image::open(hi_path(&data, &set, s.scene.seed)).unwrap().to_rgb8();
        let rgb8: Vec<u8> = resample_rgb8(&hi, size).iter().flat_map(|p| p.map(irys_recon::render::srgb8)).collect();
        let (tcam, truth) = canonical_truth(s);
        let t0 = std::time::Instant::now();
        let o = net.run(&[patches(&s.rgb)]).pop().unwrap();
        let (objs, alts, cam, gauges, sds) = net_scene(&o, s.scene.cam.hfov, k_alt, GREY);
        let t_net = t0.elapsed().as_secs_f64();
        let mut rec = serde_json::json!({"seed": s.scene.seed, "truth": s.program, "gauges": gauges});
        for (mi, (mode, sc, mdl, secs)) in scores.iter_mut().enumerate() {
            let t1 = std::time::Instant::now();
            let (init, cands, c): (Vec<(Object, f64)>, Vec<Vec<(Object, f64)>>, Cam) = match mode.as_str() {
                "net" => (objs.clone(), Vec::new(), cam),
                "net+search" => (objs.clone(), alts.clone(), cam),
                "net+search@truecam" => {
                    // Same detections, re-posed through the true camera.
                    let repose = |o: &mut Object| {
                        if let Some(px) = cam.project([o.pose[0], o.pose[1], 0.0]) {
                            if let Some(xy) = tcam.on_plane(px[0], px[1]) {
                                o.pose[0] = xy[0];
                                o.pose[1] = xy[1];
                            }
                        }
                    };
                    let (mut o2, mut a2) = (objs.clone(), alts.clone());
                    o2.iter_mut().for_each(|o| repose(&mut o.0));
                    a2.iter_mut().for_each(|v| v.iter_mut().for_each(|o| repose(&mut o.0)));
                    (o2, a2, tcam)
                }
                "search" => {
                    let tl = s.linear();
                    let blobs = segment(&tl, (128, 96), 12);
                    let mut init = Vec::new();
                    let mut cands = Vec::new();
                    for b in &blobs {
                        let cs = baseline_candidates(b, (128, 96), &tcam);
                        let cs: Vec<(Object, f64)> = cs.into_iter().map(|o| { let n = structure_nats(&o); (o, n) }).collect();
                        if let Some(f) = cs.first() {
                            init.push(f.clone());
                            cands.push(cs[1..].to_vec());
                        }
                    }
                    (init, cands, tcam)
                }
                _ => panic!("mode {mode}"),
            };
            let mut plain: Vec<Object> = init.iter().map(|x| x.0.clone()).collect();
            let (mut init, mut cands, mut c) = (init, cands, c);
            if mode == "net+search" && refine_cam {
                // The net's camera: its pitch refined on the whole frame first,
                // under a quick appearance.
                let app0 = fit_app(gpu, &hi, &plain, &c, 256);
                let t0 = linearise(&rgb8, app0.gamma);
                let ctx0 = Ctx3 {
                    gpu, target: &t0, size, cam: c, app: app0, spp, seed: 3, sigma: noise_sigma(&t0, size), sig_levels: Vec::new(), levels: levels.clone(),
                    prior_temp: temp, decorrelate: decor, edge: EdgeOpts::default(), recolour, shade, fd_weak, tau: 1.0, param_sd: Vec::new(), highpass, pose_sd, exact: None,
                };
                let (c2, _) = search3::refine_pitch(&plain, &ctx0, 6.0);
                let repose = |o: &mut Object| {
                    if let Some(px) = c.project([o.pose[0], o.pose[1], 0.0]) {
                        if let Some(xy) = c2.on_plane(px[0], px[1]) {
                            o.pose[0] = xy[0];
                            o.pose[1] = xy[1];
                        }
                    }
                };
                init.iter_mut().for_each(|o| repose(&mut o.0));
                cands.iter_mut().for_each(|v| v.iter_mut().for_each(|o| repose(&mut o.0)));
                plain = init.iter().map(|x| x.0.clone()).collect();
                c = c2;
            }
            let mut app = fit_app(gpu, &hi, &plain, &c, 256);
            let mut exact = None;
            if oracle_app || parts.contains("mats") {
                // Upper bound: the scene's own substances and light.
                for (k, o) in plain.iter().enumerate() {
                    if let Some(j) = match_one(o, &c, &truth, &tcam, 30) {
                        app.mats[k] = truth[j].0.mat;
                    }
                }
                app.table = s.scene.table;
            }
            if oracle_app || parts.contains("tone") {
                app.gamma = s.scene.post[1];
                app.defocus = s.scene.post[0] * 8.0 / 1024.0;
            }
            if oracle_app || parts.contains("light") {
                let (e, sun) = irys_recon_net::render::true_light(&s.scene);
                app.key.dir = sun.dir;
                exact = Some((e, sun, 1.0));
            }
            let mat_rec: Vec<serde_json::Value> = plain
                .iter()
                .enumerate()
                .filter_map(|(k, o)| match_one(o, &c, &truth, &tcam, 30).map(|j| serde_json::json!({"fit": app.mats[k], "true": truth[j].0.mat, "class": truth[j].0.class.name()})))
                .collect();
            rec[format!("{mode}_mats")] = serde_json::json!(mat_rec);
            let target = linearise(&rgb8, app.gamma);
            let sigma = noise_sigma(&target, size);
            let ctx = Ctx3 {
                gpu,
                target: &target,
                size,
                cam: c,
                app,
                spp,
                seed: 3,
                sigma,
                sig_levels: Vec::new(),
                levels: levels.clone(),
                prior_temp: temp,
                decorrelate: decor,
                edge: EdgeOpts::default(),
                exact: exact.clone(),
                recolour,
                shade,
                fd_weak,
                tau: 1.0,
                param_sd: Vec::new(),
                highpass,
                pose_sd,
            };
            let st0 = irys_recon_net::search::State::new(plain.clone());
            let ctx = if let Some((e, sun, _)) = &exact {
                let m = search3::render(&st0, &ctx);
                let ok = |t: &[f64; 3]| t.iter().all(|&v| v < faithful::SAT);
                let k = target.iter().zip(&m).filter(|(t, _)| ok(t)).map(|(t, m)| t[0] * m[0] + t[1] * m[1] + t[2] * m[2]).sum::<f64>()
                    / target.iter().zip(&m).filter(|(t, _)| ok(t)).map(|(_, m)| m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sum::<f64>();
                Ctx3 { exact: Some((e.clone(), *sun, k)), ..ctx }
            } else {
                ctx
            };
            let ctx = Ctx3 { sig_levels: search3::noise_levels(&st0, &ctx), ..ctx };
            let tau = if use_tau { search3::dispersion(&st0, &ctx) } else { 1.0 };
            // The baseline's detections have no network prior.
            let param_sd = if use_prior && mode.starts_with("net") { sds.clone() } else { Vec::new() };
            let ctx = Ctx3 { tau, param_sd, ..ctx };
            let mut mdl_pre = None;
            let (pred, cand_rec, counts) = if mode == "net" {
                (plain.clone(), Vec::new(), search3::Counts::default())
            } else {
                let (st, _fits, all, counts) = search3::search(init.clone(), &cands, &ctx, iters, 1);
                // The score before any rescaling (the image is the same after).
                mdl_pre = Some(search3::scene_mdl(&st.kept(), &ctx));
                // Every candidate with its error against the matched truth.
                let mut cr = Vec::new();
                for (i, fs) in all.iter().enumerate() {
                    let Some(j) = match_one(&init[i].0, &c, &truth, &tcam, 30) else { continue };
                    let t = &truth[j].0;
                    let row: Vec<serde_json::Value> = fs
                        .iter()
                        .map(|f| match &f.object {
                            Some(o) => {
                                let (topo, err) = object_error(o, t);
                                serde_json::json!({"text": o.to_text(), "mdl": f.mdl, "residual": f.residual, "param_cost": f.param_cost, "length": f.length, "topo": topo, "err": err})
                            }
                            None => serde_json::json!({"text": "drop", "mdl": f.mdl, "residual": f.residual}),
                        })
                        .collect();
                    let (_, e0) = object_error(&init[i].0, t);
                    cr.push(serde_json::json!({"object": i, "truth": t.to_text(), "init_err": e0, "cands": row}));
                }
                // With the net's camera and no gauge, the metric scale is the
                // priors' (the image does not fix it): rescale the scene.
                let mut c_out = c;
                let mut kept = st.kept();
                if mode == "net+search" && gauges.is_empty() && prior_scale_on {
                    let sc = search3::prior_scale(&st, &plain, &sds);
                    irys_recon_net::infer::rescale(&mut kept, &mut c_out, sc);
                }
                c = c_out;
                (kept, cr, counts)
            };
            let dt = t1.elapsed().as_secs_f64() + if mode.starts_with("net") { t_net } else { 0.0 };
            let m = mdl_pre.unwrap_or_else(|| search3::scene_mdl(&pred, &ctx));
            // The same score untempered (τ = 1): comparable across modes and
            // weights, where the tempered one is normalised by each start's own τ.
            let m_raw = if mode == "net" { (m - pred.iter().map(irys_recon_net::scene::program_length).sum::<f64>()) * ctx.tau + pred.iter().map(irys_recon_net::scene::program_length).sum::<f64>() } else { f64::NAN };
            let one = score(&pred, &c, &truth, &tcam, 30);
            // Paired: each initial detection's error before and after (both
            // topology-correct), matched to the same truth.
            if mode != "net" {
                let entry = match paired.iter_mut().position(|p| p.0 == *mode) {
                    Some(k) => k,
                    None => {
                        paired.push((mode.clone(), Vec::new(), Vec::new()));
                        paired.len() - 1
                    }
                };
                for (o0, _) in init.iter() {
                    let Some(j) = match_one(o0, &c, &truth, &tcam, 30) else { continue };
                    let t = &truth[j].0;
                    // The search keeps object i's slot order in st.objs, but
                    // `kept()` drops absent ones: match the after-state by truth.
                    let after = pred.iter().find(|p| match_one(p, &c, &truth, &tcam, 30) == Some(j));
                    let (ok0, e0) = object_error(o0, t);
                    if let Some(p) = after {
                        let (ok1, e1) = object_error(p, t);
                        if ok0 && ok1 {
                            paired[entry].1.push(e0);
                            paired[entry].2.push(e1);
                        }
                    }
                }
            }
            sc.add(&one);
            *mdl += m;
            raw_sum[mi] += m_raw;
            *secs += dt;
            rec[mode.as_str()] = serde_json::json!({
                "program": pred.iter().map(|o| o.to_text()).collect::<Vec<_>>(),
                "mdl": m, "mdl_raw": m_raw, "seconds": dt, "matched": one.matched, "gt": one.gt, "pred": one.pred, "topo_ok": one.topo_ok,
                "tau": ctx.tau, "builds": counts.builds, "moves": counts.moves, "renders": counts.renders, "jacobians": counts.jacobians,
                "candidates": cand_rec, "dim_err_rel": one.dim_err_rel,
            });
        }
        per_scene.push(rec);
        let line: Vec<String> = scores
            .iter()
            .map(|(m, sc, _, secs)| format!("{m}: R {:.2} P {:.2} topo {:.2} dim {:.1}% {:.0}s", sc.matched as f64 / sc.gt.max(1) as f64, sc.matched as f64 / sc.pred.max(1) as f64, sc.topo_ok as f64 / sc.matched.max(1) as f64, 100.0 * median(&sc.dim_err_rel), secs / (si + 1) as f64))
            .collect();
        let pl: Vec<String> = paired.iter().map(|(m, b, a)| format!("{m}: paired {:.1}% → {:.1}% (n {})", 100.0 * median(b), 100.0 * median(a), b.len())).collect();
        println!("[{}/{}] {} || {}", si + 1, samples.len(), line.join(" | "), pl.join(" | "));
        let rows: Vec<_> = scores.iter().map(|(m, sc, mdl, secs)| row(m, si + 1, sc, mdl / (si + 1) as f64, secs / (si + 1) as f64)).collect();
        let pj: Vec<_> = paired.iter().map(|(m, b, a)| serde_json::json!({"mode": m, "n": b.len(), "before_median_rel": median(b), "after_median_rel": median(a),
            "before_mean_rel": b.iter().sum::<f64>() / b.len().max(1) as f64, "after_mean_rel": a.iter().sum::<f64>() / a.len().max(1) as f64,
            "improved_frac": b.iter().zip(a).filter(|(x, y)| y < x).count() as f64 / b.len().max(1) as f64})).collect();
        let raw: Vec<_> = scores.iter().zip(&raw_sum).map(|((m, ..), r)| serde_json::json!({"mode": m, "mean_mdl_raw": r / (si + 1) as f64})).collect();
        std::fs::write(out.join("summary.json"), serde_json::to_string_pretty(&serde_json::json!({"rows": rows, "paired": pj, "raw": raw})).unwrap()).unwrap();
        std::fs::write(out.join("scenes.json"), serde_json::to_string_pretty(&per_scene).unwrap()).unwrap();
    }
}
