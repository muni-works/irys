//! Amortised inference + search on held-out synthetic scenes:
//! `neural_eval --weights out/neural/run/weights.json --set test-a --n 60`.
//! Modes: the net alone; net + MDL search (net camera); net + search (true
//! camera, as a tag would give); search alone (true camera; segmentation and
//! every family, the v0/v1-style baseline).

use std::path::PathBuf;

use irys_recon_net::data::{Sample, patches, read_dir};
use irys_recon_net::eval::{Score, canonical_truth, row, score};
use irys_recon_net::infer::{self, gauge_scale, hypotheses, proposals, rescale};
use irys_recon_net::net::{H, W};
use irys_recon_net::scene::{Cam, Mat, Object};
use irys_recon_net::search::{self, Ctx, baseline_candidates, noise_sigma, scene_mdl, segment};
use irys_recon_net::train::{Infer, Weights};

fn arg<T: std::str::FromStr>(a: &[String], k: &str, d: T) -> T {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(d)
}

const GREY: Mat = Mat { rgb: [0.5; 3], rough: 0.5, metal: false };

/// The net's scene: objects (top-1 structure), alternatives, camera.
pub fn net_scene(out: &irys_recon_net::train::Out, hfov: f64, k: usize) -> (Vec<(Object, f64)>, Vec<Vec<(Object, f64)>>, Cam, Vec<usize>) {
    let props = proposals(out, 0.35);
    let mut cam = infer::camera(out, hfov);
    let mut objs = Vec::new();
    let mut alts = Vec::new();
    let mut probs = Vec::new();
    for p in &props {
        let hs = hypotheses(p, k);
        let cands: Vec<(Object, f64)> = hs.iter().filter_map(|h| infer::object(p, h, &cam, GREY).map(|o| (o, -h.logp))).collect();
        if let Some(first) = cands.first() {
            probs.push(p.cls[first.0.class as usize]);
            objs.push(first.clone());
            alts.push(cands[1..].to_vec());
        }
    }
    let plain: Vec<Object> = objs.iter().map(|o: &(Object, f64)| o.0.clone()).collect();
    let (s, who) = gauge_scale(&plain, &probs);
    if who.is_empty() {
        return (objs, alts, cam, who);
    }
    let mut c2 = cam;
    for o in objs.iter_mut() {
        rescale(std::slice::from_mut(&mut o.0), &mut c2.clone(), s);
    }
    for a in alts.iter_mut() {
        for o in a.iter_mut() {
            rescale(std::slice::from_mut(&mut o.0), &mut c2.clone(), s);
        }
    }
    rescale(&mut [], &mut c2, s);
    cam = c2;
    (objs, alts, cam, who)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = PathBuf::from(arg(&a, "--data", "out/neural/data".to_string()));
    let wpath = PathBuf::from(arg(&a, "--weights", "out/neural/run/weights.json".to_string()));
    let set: String = arg(&a, "--set", "test-a".to_string());
    let n: usize = arg(&a, "--n", 60);
    let out = PathBuf::from(arg(&a, "--out", format!("out/neural/eval-{set}")));
    let modes: String = arg(&a, "--modes", "net,net+search,net+search@truecam,search".to_string());
    let iters: usize = arg(&a, "--iters", 8);
    std::fs::create_dir_all(&out).unwrap();
    let skip: usize = arg(&a, "--skip", 0);
    let samples: Vec<Sample> = read_dir(&data, &set).into_iter().skip(skip).take(n).collect();
    let w = Weights::load(&wpath).unwrap();
    let mut net = Infer::new(&w, 8);
    let mut scores: Vec<(String, Score, f64, f64)> = modes.split(',').map(|m| (m.to_string(), Score::default(), 0.0, 0.0)).collect();
    let mut per_scene = Vec::new();
    for (si, s) in samples.iter().enumerate() {
        let (tcam, truth) = canonical_truth(s);
        let target = s.linear();
        let size = (W, H);
        let sigma = noise_sigma(&target, size);
        let t0 = std::time::Instant::now();
        let o = net.run(&[patches(&s.rgb)]).pop().unwrap();
        let (objs, alts, cam, gauges) = net_scene(&o, s.scene.cam.hfov, 3);
        let t_net = t0.elapsed().as_secs_f64();
        let mut rec = serde_json::json!({"seed": s.scene.seed, "truth": s.program, "gauges": gauges});
        for (mode, sc, mdl, secs) in scores.iter_mut() {
            let t1 = std::time::Instant::now();
            let (pred, used_cam): (Vec<Object>, Cam) = match mode.as_str() {
                "net" => (objs.iter().map(|o| o.0.clone()).collect(), cam),
                "net+search" | "net+search@truecam" => {
                    let c = if mode == "net+search" { cam } else { tcam };
                    let (mut o2, mut a2) = (objs.clone(), alts.clone());
                    if mode != "net+search" {
                        // Same detections, re-posed through the true camera.
                        let repose = |o: &mut Object| {
                            if let Some(px) = cam.project([o.pose[0], o.pose[1], 0.0]) {
                                if let Some(xy) = tcam.on_plane(px[0], px[1]) {
                                    o.pose[0] = xy[0];
                                    o.pose[1] = xy[1];
                                }
                            }
                        };
                        o2.iter_mut().for_each(|o| repose(&mut o.0));
                        a2.iter_mut().for_each(|v| v.iter_mut().for_each(|o| repose(&mut o.0)));
                    }
                    let ctx = Ctx { target: &target, size, cam: c, ss: 3, sigma, blur: 0.7 };
                    let (st, _, _) = search::search(o2, &a2, &ctx, iters, 1);
                    (st.kept(), c)
                }
                "search" => {
                    let blobs = segment(&target, size, 12);
                    let mut init = Vec::new();
                    let mut cands = Vec::new();
                    for b in &blobs {
                        let cs = baseline_candidates(b, size, &tcam);
                        let cs: Vec<(Object, f64)> = cs.into_iter().map(|o| { let n = irys_recon_net::scene::structure_nats(&o); (o, n) }).collect();
                        if let Some(f) = cs.first() {
                            init.push(f.clone());
                            cands.push(cs[1..].to_vec());
                        }
                    }
                    let ctx = Ctx { target: &target, size, cam: tcam, ss: 3, sigma, blur: 0.7 };
                    let (st, _, _) = search::search(init, &cands, &ctx, iters, 1);
                    (st.kept(), tcam)
                }
                _ => panic!("mode {mode}"),
            };
            let dt = t1.elapsed().as_secs_f64() + if mode.starts_with("net") { t_net } else { 0.0 };
            let ctx = Ctx { target: &target, size, cam: used_cam, ss: 2, sigma, blur: 0.7 };
            let m = scene_mdl(&pred, &ctx);
            let one = score(&pred, &used_cam, &truth, &tcam, 30);
            sc.add(&one);
            *mdl += m;
            *secs += dt;
            rec[mode.as_str()] = serde_json::json!({
                "program": pred.iter().map(|o| o.to_text()).collect::<Vec<_>>(),
                "mdl": m, "seconds": dt, "matched": one.matched, "gt": one.gt, "pred": one.pred, "topo_ok": one.topo_ok,
            });
        }
        per_scene.push(rec);
        let line: Vec<String> = scores.iter().map(|(m, sc, _, secs)| format!("{m}: R {:.2} P {:.2} topo {:.2} {:.1}s", sc.matched as f64 / sc.gt.max(1) as f64, sc.matched as f64 / sc.pred.max(1) as f64, sc.topo_ok as f64 / sc.matched.max(1) as f64, secs / (si + 1) as f64)).collect();
        println!("[{}/{}] {}", si + 1, samples.len(), line.join(" | "));
        let rows: Vec<_> = scores.iter().map(|(m, sc, mdl, secs)| row(m, si + 1, sc, mdl / (si + 1) as f64, secs / (si + 1) as f64)).collect();
        std::fs::write(out.join("summary.json"), serde_json::to_string_pretty(&rows).unwrap()).unwrap();
        std::fs::write(out.join("scenes.json"), serde_json::to_string_pretty(&per_scene).unwrap()).unwrap();
    }
}
