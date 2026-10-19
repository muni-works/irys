//! Renderer as teacher: fine-tune on unlabelled photos.
//! `neural_finetune --weights out/neural/run/weights.json --set ft-b --steps 200`
//!
//! No label of the photos is read. Each detection is rendered through irys
//! (the TLAS cast with the linear appearance) and compared with the photo:
//!
//! - continuous heads (parameter means, yaw, the plane) get `∂L/∂head` by
//!   central differences of the render-compare loss `L` (Huber nats per
//!   pixel over the object's window, the full frame for the plane), fed to
//!   the graph as upstream gradients, so `TensorGraph::backward` carries
//!   them into the weights;
//! - the program-token heads are a policy: K structures are sampled from
//!   them under the grammar, each scored by its MDL (residual + length),
//!   and REINFORCE with the mean as baseline pushes up the shorter
//!   descriptions (the graph's cross-entropy with advantage-weighted targets).

use std::path::PathBuf;

use irys_recon::jac::Mix;
use irys_recon_net::data::{Sample, Targets, patches, read_dir};
use irys_recon_net::infer::{self, Hypothesis, Proposal, proposals, valid_token};
use irys_recon_net::net::{H, P, W};
use irys_recon_net::scene::{CLASSES, FAMILIES, Family, Mat, NP, NTOK, Object, VOCAB, norm_mm, param_used, program_length, token_mask};
use irys_recon_net::search::{Ctx, State, noise_sigma, residual, rho_sum, window};
use irys_recon_net::train::{Batch, Infer, Out, Trainer, Weights};
use rayon::prelude::*;

fn arg<T: std::str::FromStr>(a: &[String], k: &str, d: T) -> T {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(d)
}

const GREY: Mat = Mat { rgb: [0.5; 3], rough: 0.5, metal: false };

fn zero_targets() -> Targets {
    Targets {
        t_pix: vec![0.0; P * 4 * 64],
        w_pix: vec![0.0; P * 4 * 64],
        t_ctr: vec![0.0; P],
        w_ctr: vec![0.0; P],
        t_cls: vec![0.0; P * CLASSES],
        w_cls: vec![0.0; P],
        t_tok: vec![0.0; P * NTOK * VOCAB],
        w_tok: vec![0.0; P * NTOK],
        t_par: vec![0.0; P * NP],
        w_par: vec![0.0; P * NP],
        t_yaw: vec![0.0; P * 2],
        w_yaw: vec![0.0; P * 2],
        t_plane: vec![0.0; 2],
        w_plane: vec![0.0; 2],
    }
}

fn top1(p: &Proposal) -> Hypothesis {
    infer::hypotheses(p, 1).remove(0)
}

/// Sample a structure from the token heads under the grammar.
fn sample(p: &Proposal, r: &mut Mix) -> Hypothesis {
    let draw = |probs: &[f64], ok: &dyn Fn(usize) -> bool, r: &mut Mix| -> usize {
        let tot: f64 = (0..VOCAB).filter(|&v| ok(v)).map(|v| probs[v]).sum();
        let mut u = r.uniform() * tot;
        for v in 0..VOCAB {
            if ok(v) {
                u -= probs[v];
                if u <= 0.0 {
                    return v;
                }
            }
        }
        (0..VOCAB).rev().find(|&v| ok(v)).unwrap_or(0)
    };
    let f = draw(&p.tok[0], &|v| v < FAMILIES.len(), r);
    let fam = Family::from_token(f).unwrap();
    let mut t = [f, 0, 0, 0];
    for slot in 1..NTOK {
        let mut probe = t;
        probe[slot] = 1;
        let any = (0..VOCAB).any(|v| {
            let mut q = t;
            q[slot] = v;
            valid_token(fam, slot, v) && token_mask(fam, &q)[slot]
        });
        if any {
            t[slot] = draw(&p.tok[slot], &|v| valid_token(fam, slot, v), r);
        }
        let _ = probe;
    }
    Hypothesis { family: fam, tokens: t, logp: 0.0 }
}

struct Teach {
    g_par: Vec<f64>,
    g_yaw: Vec<f64>,
    g_plane: Vec<f64>,
    t_tok: Vec<f64>,
    w_tok: Vec<f64>,
    loss: f64,
    mdl_top1: f64,
}

/// The teacher's signal for one image's net output.
fn teach(out: &Out, s: &Sample, k_samples: usize, seed: u64) -> Teach {
    let target = s.linear();
    let size = (W, H);
    let sigma = noise_sigma(&target, size);
    let hfov = s.scene.cam.hfov;
    let mut tc = Teach {
        g_par: vec![0.0; P * 2 * NP],
        g_yaw: vec![0.0; P * 2],
        g_plane: vec![0.0; 2],
        t_tok: vec![0.0; P * NTOK * VOCAB],
        w_tok: vec![0.0; P * NTOK],
        loss: 0.0,
        mdl_top1: 0.0,
    };
    let mut props = proposals(out, 0.35);
    props.truncate(12);
    let cam = infer::camera(out, hfov);
    let build = |props: &[Proposal], cam: &irys_recon_net::scene::Cam| -> Vec<Option<Object>> { props.iter().map(|p| infer::object(p, &top1(p), cam, GREY)).collect() };
    let objs = build(&props, &cam);
    let kept: Vec<(usize, Object)> = objs.iter().enumerate().filter_map(|(i, o)| o.clone().map(|o| (i, o))).collect();
    if kept.is_empty() {
        return tc;
    }
    let st = State::new(kept.iter().map(|k| k.1.clone()).collect());
    let ctx = Ctx { target: &target, size, cam, ss: 2, sigma, blur: 0.7 };
    let full = (0, 0, W, H);
    let npx = (W * H * 3) as f64;
    let l_full = rho_sum(&residual(&st, &ctx, full)) / npx;
    tc.loss = l_full;
    // The plane: every object re-posed through the perturbed camera.
    for k in 0..2 {
        let h = 0.02;
        let mut o2 = out.clone();
        o2.plane[k] += h;
        let cp = infer::camera(&o2, hfov);
        let objs_p: Vec<Object> = build(&props, &cp).into_iter().flatten().collect();
        let ctxp = Ctx { cam: cp, ..ctx };
        let mut o3 = out.clone();
        o3.plane[k] -= h;
        let cm = infer::camera(&o3, hfov);
        let objs_m: Vec<Object> = build(&props, &cm).into_iter().flatten().collect();
        let ctxm = Ctx { cam: cm, ..ctx };
        let lp = rho_sum(&residual(&State::new(objs_p), &ctxp, full)) / npx;
        let lm = rho_sum(&residual(&State::new(objs_m), &ctxm, full)) / npx;
        tc.g_plane[k] = (lp - lm) / (2.0 * h);
    }
    let per_obj: Vec<(usize, Vec<(usize, f64)>, [f64; 2], Vec<(Hypothesis, f64)>, f64)> = kept
        .par_iter()
        .enumerate()
        .filter_map(|(j, (pi, o))| {
            let p = &props[*pi];
            let win = window(&[o], &cam, size, 4.0)?;
            let wpx = ((win.2 - win.0) * (win.3 - win.1) * 3) as f64;
            let cost = |o: &Object| rho_sum(&residual(&st.with(j, Some(o)), &ctx, win)) / wpx;
            let hyp = top1(p);
            // Continuous: central differences in each used mean.
            let mut gpar = Vec::new();
            for k in 0..hyp.family.param_names().len() {
                if !param_used(hyp.family, &hyp.tokens, k) {
                    continue;
                }
                let h = 0.03;
                let mut q = p.clone();
                q.mu[k] += h;
                let op = infer::object(&q, &hyp, &cam, GREY)?;
                q.mu[k] -= 2.0 * h;
                let om = infer::object(&q, &hyp, &cam, GREY)?;
                gpar.push((k, (cost(&op) - cost(&om)) / (2.0 * h)));
            }
            let mut gyaw = [0.0; 2];
            if hyp.family.has_yaw() {
                for c in 0..2 {
                    let h = 0.05;
                    let mut q = p.clone();
                    q.yaw[c] += h;
                    let op = infer::object(&q, &hyp, &cam, GREY)?;
                    q.yaw[c] -= 2.0 * h;
                    let om = infer::object(&q, &hyp, &cam, GREY)?;
                    gyaw[c] = (cost(&op) - cost(&om)) / (2.0 * h);
                }
            }
            // Discrete: K structures, MDL reward.
            let mut r = Mix(seed ^ (*pi as u64 * 7919));
            let mut samples = Vec::new();
            for _ in 0..k_samples {
                let hs = sample(p, &mut r);
                if let Some(os) = infer::object(p, &hs, &cam, GREY) {
                    let mdl = cost(&os) * wpx + program_length(&os);
                    samples.push((hs, -mdl / wpx));
                }
            }
            let top = cost(o) * wpx + program_length(o);
            Some((p.patch, gpar, gyaw, samples, top))
        })
        .collect();
    for (patch, gpar, gyaw, samples, top) in per_obj {
        tc.mdl_top1 += top;
        for (k, g) in gpar {
            tc.g_par[(patch * 2) * NP + k] = g;
        }
        tc.g_yaw[patch * 2] = gyaw[0];
        tc.g_yaw[patch * 2 + 1] = gyaw[1];
        if samples.len() >= 2 {
            let mean = samples.iter().map(|s| s.1).sum::<f64>() / samples.len() as f64;
            let sd = (samples.iter().map(|s| (s.1 - mean).powi(2)).sum::<f64>() / samples.len() as f64).sqrt().max(1e-3);
            for (h, rew) in &samples {
                let adv = (rew - mean) / sd / samples.len() as f64;
                for slot in 0..NTOK {
                    if token_mask(h.family, &h.tokens)[slot] {
                        tc.t_tok[(patch * NTOK + slot) * VOCAB + h.tokens[slot]] += adv;
                        tc.w_tok[patch * NTOK + slot] = 1.0;
                    }
                }
            }
        }
    }
    tc
}

/// A real photo as an unlabelled sample: its pixels at the net's size and
/// the field of view its EXIF gives; nothing else.
fn from_photo(path: &std::path::Path) -> Sample {
    let bytes = std::fs::read(path).unwrap();
    let exif = irys_recon::tag::read_exif(&bytes);
    let img = image::load_from_memory(&bytes).unwrap().to_rgb8();
    let (pw, ph) = (img.width() as f64, img.height() as f64);
    let f_px = exif.f35.unwrap_or(26.0) / 36f64.hypot(24.0) * pw.hypot(ph);
    let hfov = 2.0 * (pw / 2.0 / f_px).atan().to_degrees();
    let small = image::imageops::resize(&img, W as u32, H as u32, image::imageops::FilterType::Triangle);
    let mut scene = irys_recon_net::scene::sample_scene(0, irys_recon_net::scene::Domain::A, (W, H));
    scene.objects.clear();
    scene.cam.hfov = hfov;
    Sample { scene, program: String::new(), gauges: Vec::new(), rgb: small.into_raw(), inst: vec![0; W * H], depth: vec![0.0; W * H] }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = PathBuf::from(arg(&a, "--data", "out/neural/data".to_string()));
    let wpath = PathBuf::from(arg(&a, "--weights", "out/neural/run/weights.json".to_string()));
    let set: String = arg(&a, "--set", "ft-b".to_string());
    let out = PathBuf::from(arg(&a, "--out", "out/neural/ft".to_string()));
    let steps: usize = arg(&a, "--steps", 200);
    let lr: f64 = arg(&a, "--lr", 5e-5);
    let workers: usize = arg(&a, "--workers", 12);
    let k_samples: usize = arg(&a, "--k", 6);
    let w_teach: f64 = arg(&a, "--teach", 1.0);
    let w_pg: f64 = arg(&a, "--pg", 0.05);
    // The plane head's teacher weight (0: leave the metric scale alone).
    let w_plane: f64 = arg(&a, "--plane", 1.0);
    std::fs::create_dir_all(&out).unwrap();
    // Unlabelled: only the photo and the camera's field of view (EXIF) are read.
    let photo: String = arg(&a, "--photo", String::new());
    let photos: Vec<Sample> = if photo.is_empty() { read_dir(&data, &set) } else { vec![from_photo(std::path::Path::new(&photo))] };
    // Optional anchor: supervised batches from the *training* domain (A),
    // whose labels are the synthetic stream's own, never the tuning set's.
    let anchor_n: usize = arg(&a, "--anchor", 0);
    let anchor_w: f64 = arg(&a, "--anchor-weight", 0.5);
    let anchor: Vec<Sample> = if anchor_n > 0 { read_dir(&data, "train-a-0000").into_iter().take(2000).collect() } else { Vec::new() };
    let n_teach = workers - anchor_n.min(workers - 1);
    let w = Weights::load(&wpath).unwrap();
    let mut tr = Trainer::new(w.arch, 1, workers, Some(w.f64s()), 0);
    let mut net = Infer::new(&w, workers);
    let mut rng = Mix(99);
    let mut curve = Vec::new();
    let t0 = std::time::Instant::now();
    for step in 1..=steps {
        let picks: Vec<&Sample> = (0..n_teach).map(|_| &photos[(rng.uniform() * photos.len() as f64) as usize % photos.len()]).collect();
        net.set_weights(&tr.params);
        let xs: Vec<Vec<f64>> = picks.iter().map(|s| patches(&s.rgb)).collect();
        let outs = net.run(&xs);
        let xs = &xs[..n_teach];
        let teach_v: Vec<Teach> = outs.par_iter().zip(&picks).enumerate().map(|(i, (o, s))| teach(o, s, k_samples, step as u64 * 1000 + i as u64)).collect();
        let n = n_teach as f64;
        let mut batches: Vec<Batch> = xs
            .iter()
            .zip(&teach_v)
            .map(|(x, t)| {
                let mut tg = zero_targets();
                tg.t_tok = t.t_tok.clone();
                tg.w_tok = t.w_tok.iter().map(|v| v * w_pg / n).collect();
                let mut b = Batch::new(std::slice::from_ref(x), &[tg]);
                b.g_par = t.g_par.iter().map(|v| v * w_teach / n).collect();
                b.g_yaw = t.g_yaw.iter().map(|v| v * w_teach / n).collect();
                b.g_plane = t.g_plane.iter().map(|v| v * w_teach * w_plane / n).collect();
                b
            })
            .collect();
        for _ in n_teach..workers {
            let s = &anchor[(rng.uniform() * anchor.len() as f64) as usize % anchor.len()];
            let mut t = irys_recon_net::data::targets(s, workers - n_teach);
            for v in [&mut t.w_pix, &mut t.w_ctr, &mut t.w_cls, &mut t.w_tok, &mut t.w_par, &mut t.w_yaw, &mut t.w_plane] {
                v.iter_mut().for_each(|x| *x *= anchor_w);
            }
            batches.push(Batch::new(&[patches(&s.rgb)], &[t]));
        }
        let (_l, g) = tr.grads(&batches);
        let gn = tr.adam(&g, lr, 1.0, 0.0);
        let loss = teach_v.iter().map(|t| t.loss).sum::<f64>() / n;
        let mdl = teach_v.iter().map(|t| t.mdl_top1).sum::<f64>() / n;
        curve.push(serde_json::json!({"step": step, "render_loss": loss, "mdl_top1": mdl, "grad_norm": gn, "seconds": t0.elapsed().as_secs_f64()}));
        if step % 10 == 0 || step == steps {
            println!("step {step:4}  {:.0}s  render loss {loss:.4} nats/px  top-1 MDL {mdl:.0}  |g| {gn:.3}", t0.elapsed().as_secs_f64());
            std::fs::write(out.join("curve.json"), serde_json::to_string_pretty(&curve).unwrap()).unwrap();
            tr.weights().save(&out.join("weights.json")).unwrap();
        }
    }
    let _ = norm_mm(1.0);
}
