//! v3 renderer as teacher: v2's fine-tuning (`neural_finetune`) with the
//! faithful compare image and analytic gradients.
//! `neural_v3_finetune --weights out/neural/run2/weights.json --set v3-ftb --steps 120`
//!
//! No label of the photos is read. Per photo, once: the appearance (world,
//! key, tone, table) is fitted to the photo with the starting network's
//! detections and cached. Per step and image:
//!
//! - the network's top-1 scene is rendered by irys's OpenPBR kernel with
//!   that appearance, each object recoloured on its own pixels;
//! - the continuous heads (parameter means, yaw) get `∂L/∂head` from the
//!   **edge-sampled Jacobian** (`edgejac`, analytic vertex velocities,
//!   primary and shadow edges) contracted with the residual, `JᵀWr`, with
//!   forward differences only for the columns `search3::weak` names — no
//!   central differences of the loss as in v2;
//! - the structure heads are a policy: K structures sampled under the
//!   grammar, each scored by v3's MDL (band-passed, tempered residual +
//!   the program's length), REINFORCE with a normalised advantage.
//!
//! The loss is v3's score, so the teacher pushes the network where the
//! search would go.

use std::path::PathBuf;

use irys_recon::jac::Mix;
use irys_recon_net::data::{Sample, Targets, hi_path, patches, read_dir};
use irys_recon_net::edgejac::EdgeOpts;
use irys_recon_net::faithful::{self, Appearance, Compare3, FitOpts, linearise, local_meshes};
use irys_recon_net::infer::{self, Hypothesis, Proposal, proposals, valid_token};
use irys_recon_net::net::P;
use irys_recon_net::render::resample_rgb8;
use irys_recon_net::scene::{CLASSES, FAMILIES, Family, Mat, NP, NTOK, Object, VOCAB, program_length, token_mask};
use irys_recon_net::search::{State, noise_sigma, window};
use irys_recon_net::search3::{self, Ctx3};
use irys_recon_net::train::{Batch, Infer, Out, Trainer, Weights};

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
        let any = (0..VOCAB).any(|v| {
            let mut q = t;
            q[slot] = v;
            valid_token(fam, slot, v) && token_mask(fam, &q)[slot]
        });
        if any {
            t[slot] = draw(&p.tok[slot], &|v| valid_token(fam, slot, v), r);
        }
    }
    Hypothesis { family: fam, tokens: t, logp: 0.0 }
}

/// What is cached per photo.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct Cached {
    app: Appearance,
    sig_levels: Vec<f64>,
    tau: f64,
    sigma: f64,
}

struct Teach {
    g_par: Vec<f64>,
    g_yaw: Vec<f64>,
    /// Gauss–Newton targets (`--gn`): the heads regressed one damped step
    /// toward the render's optimum, `t_par`/`t_yaw` with their weights.
    t_par: Vec<f64>,
    w_par: Vec<f64>,
    t_yaw: Vec<f64>,
    w_yaw: Vec<f64>,
    t_tok: Vec<f64>,
    w_tok: Vec<f64>,
    loss: f64,
    mdl_top1: f64,
}

struct Opt {
    size: (usize, usize),
    levels: Vec<f64>,
    highpass: f64,
    k_samples: usize,
    max_obj: usize,
    gn: bool,
    refine_cam: bool,
    scale_free: bool,
}

fn solve(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = b.len();
    let mut m: Vec<Vec<f64>> = a.iter().zip(b).map(|(r, &y)| r.iter().copied().chain([y]).collect()).collect();
    for i in 0..n {
        let piv = (i..n).max_by(|&x, &y| m[x][i].abs().total_cmp(&m[y][i].abs()))?;
        m.swap(i, piv);
        if m[i][i].abs() < 1e-14 {
            return None;
        }
        for r in 0..n {
            if r != i {
                let f = m[r][i] / m[i][i];
                for c in i..=n {
                    m[r][c] -= f * m[i][c];
                }
            }
        }
    }
    Some((0..n).map(|i| m[i][n] / m[i][i]).collect())
}

fn ctx_of<'a>(gpu: &'static irys_gpu::GpuContext, c: &Cached, target: &'a [[f64; 3]], cam: irys_recon_net::scene::Cam, o: &Opt) -> Ctx3<'a> {
    Ctx3 {
        gpu,
        target,
        size: o.size,
        cam,
        app: c.app.clone(),
        spp: 8,
        seed: 3,
        sigma: c.sigma,
        sig_levels: c.sig_levels.clone(),
        levels: o.levels.clone(),
        prior_temp: 1.0,
        decorrelate: true,
        edge: EdgeOpts::default(),
        recolour: true,
        shade: true,
        fd_weak: true,
        tau: c.tau,
        param_sd: Vec::new(),
        pose_sd: 1e9,
        highpass: o.highpass,
        exact: None,
    }
}

/// The objects of the net's top-1 scene, with each one's proposal index.
fn top_scene(out: &Out, hfov: f64, max_obj: usize) -> (Vec<Proposal>, irys_recon_net::scene::Cam, Vec<(usize, Object)>) {
    let mut props = proposals(out, 0.35);
    props.truncate(max_obj);
    let cam = infer::camera(out, hfov);
    let kept = props.iter().enumerate().filter_map(|(i, p)| infer::object(p, &top1(p), &cam, GREY).map(|o| (i, o))).collect();
    (props, cam, kept)
}

fn teach(gpu: &'static irys_gpu::GpuContext, out: &Out, s: &Sample, rgb8: &[u8], c: &Cached, o: &Opt, seed: u64) -> Teach {
    let mut tc = Teach { g_par: vec![0.0; P * 2 * NP], g_yaw: vec![0.0; P * 2], t_par: vec![0.0; P * NP], w_par: vec![0.0; P * NP], t_yaw: vec![0.0; P * 2], w_yaw: vec![0.0; P * 2], t_tok: vec![0.0; P * NTOK * VOCAB], w_tok: vec![0.0; P * NTOK], loss: 0.0, mdl_top1: 0.0 };
    let target = linearise(rgb8, c.app.gamma);
    let (props, cam, kept) = top_scene(out, s.scene.cam.hfov, o.max_obj);
    if kept.is_empty() {
        return tc;
    }
    let mut objs: Vec<Object> = kept.iter().map(|k| k.1.clone()).collect();
    let mut kept = kept;
    let mut app = c.app.clone();
    app.mats = vec![GREY; objs.len()];
    let mut ctx = Ctx3 { app, ..ctx_of(gpu, c, &target, cam, o) };
    let mut cam = cam;
    if o.refine_cam {
        // The plane head's pitch, refined on the frame (objects keep their
        // image places), as the search does.
        let (c2, os) = search3::refine_pitch(&objs, &ctx, 6.0);
        cam = c2;
        ctx.cam = c2;
        objs = os;
        for (k, ob) in kept.iter_mut().zip(&objs) {
            k.1 = ob.clone();
        }
    }
    let st = State::new(objs.clone());
    let base = search3::render(&st, &ctx);
    let s_fin = *ctx.levels.last().unwrap();
    let q = 1.0 / (ctx.kappa(s_fin) * ctx.tau);
    let full = (0, 0, o.size.0, o.size.1);
    let rf = search3::residual_of(&base, &ctx, full, s_fin);
    tc.loss = search3::rho_sum(&rf) / rf.len() as f64;
    let cam_s = cam.resized(o.size);
    // Gauss–Newton steps of every size parameter: (patch, slot, ln θ, Δ).
    let mut steps: Vec<(usize, usize, f64, f64)> = Vec::new();
    for (j, (pi, ob)) in kept.iter().enumerate() {
        let p = &props[*pi];
        let Some(win) = window(&[ob], &cam_s, o.size, 6.0) else { continue };
        let model = search3::recolour(&base, &st, j, &ctx, win);
        let r = search3::residual_of(&model, &ctx, win, s_fin);
        let npx = r.len() as f64;
        let w: Vec<f64> = r.iter().map(|&x| irys_recon::look::huber_w(x, 2.0)).collect();
        // ∂L/∂v for every encoded variable: J^T (w ∘ r) / npx.
        let cols = search3::jacobian(&st, j, &model, &ctx, win, s_fin, false);
        let sl = search3::slots(ob);
        if o.gn {
            // One damped Gauss–Newton step on q·ρ + the head's own prior.
            let nv = cols.len();
            let mut a = vec![vec![0.0; nv]; nv];
            let mut g = vec![0.0; nv];
            for x in 0..nv {
                for y in 0..=x {
                    let v: f64 = cols[x].iter().zip(&cols[y]).zip(&w).map(|((p, q2), c)| p * q2 * c).sum::<f64>() * q;
                    a[x][y] = v;
                    a[y][x] = v;
                }
                g[x] = cols[x].iter().zip(&r).zip(&w).map(|((p, q2), c)| p * q2 * c).sum::<f64>() * q;
            }
            for (x, &(slot, _)) in sl.iter().enumerate() {
                let sd = if slot < NP { (1.5 * p.sigma[slot]).clamp(0.02, 1.0) } else { 1.0 };
                a[x][x] += 1.0 / (sd * sd) + 0.3 * a[x][x];
            }
            if let Some(d) = solve(&a, &g.iter().map(|v| -v).collect::<Vec<_>>()) {
                for (x, &(slot, _)) in sl.iter().enumerate() {
                    let dx = d[x].clamp(-0.25, 0.25);
                    if slot < NP {
                        steps.push((p.patch, slot, ob.params[slot].ln(), dx));
                    } else if slot == NP + 2 {
                        let yaw = (ob.pose[2] + 20.0 * dx).to_radians() * 2.0;
                        tc.t_yaw[p.patch * 2] = yaw.cos();
                        tc.t_yaw[p.patch * 2 + 1] = yaw.sin();
                        tc.w_yaw[p.patch * 2] = 1.0;
                        tc.w_yaw[p.patch * 2 + 1] = 1.0;
                    }
                }
            }
        }
        for (col, &(slot, _)) in cols.iter().zip(&sl) {
            let g: f64 = col.iter().zip(&r).zip(&w).map(|((a, b), c)| a * b * c).sum::<f64>() / npx;
            if slot < NP {
                // v = ln θ = 1.5 μ + 3.5.
                tc.g_par[(p.patch * 2) * NP + slot] = 1.5 * g;
            } else if slot == NP + 2 {
                // v = yaw/20 (degrees); yaw = ½ atan2(s, c).
                let (cc, ss) = (p.yaw[0], p.yaw[1]);
                let n2 = (cc * cc + ss * ss).max(1e-9);
                let dyaw = g / 20.0 * 0.5 * 180.0 / std::f64::consts::PI;
                tc.g_yaw[p.patch * 2] = dyaw * (-ss / n2);
                tc.g_yaw[p.patch * 2 + 1] = dyaw * (cc / n2);
            }
        }
        let top = q * search3::rho_sum(&r) + program_length(ob);
        tc.mdl_top1 += top;
        // Structures: K samples, each rendered and scored.
        let mut rng = Mix(seed ^ (*pi as u64 * 7919));
        let mut samples = Vec::new();
        for _ in 0..o.k_samples {
            let hs = sample(p, &mut rng);
            let Some(os) = infer::object(p, &hs, &cam, GREY) else { continue };
            let sn = st.with(j, Some(&os));
            let m = search3::recolour(&search3::render(&sn, &ctx), &sn, j, &ctx, win);
            let rr = search3::residual_of(&m, &ctx, win, s_fin);
            let mdl = q * search3::rho_sum(&rr) + program_length(&os);
            samples.push((hs, -mdl));
        }
        if samples.len() >= 2 {
            let mean = samples.iter().map(|s| s.1).sum::<f64>() / samples.len() as f64;
            let sd = (samples.iter().map(|s| (s.1 - mean).powi(2)).sum::<f64>() / samples.len() as f64).sqrt().max(1e-3);
            for (h, rew) in &samples {
                let adv = (rew - mean) / sd / samples.len() as f64;
                for slot in 0..NTOK {
                    if token_mask(h.family, &h.tokens)[slot] {
                        tc.t_tok[(p.patch * NTOK + slot) * VOCAB + h.tokens[slot]] += adv;
                        tc.w_tok[p.patch * NTOK + slot] = 1.0;
                    }
                }
            }
        }
    }
    // The photo fixes shapes up to one scale, and the scale belongs to the
    // gauges and the plane head: without a gauge the common part of the
    // steps (the image's mean log-size change) is removed.
    let mean = if o.scale_free && !steps.is_empty() { steps.iter().map(|s| s.3).sum::<f64>() / steps.len() as f64 } else { 0.0 };
    for (patch, slot, ln, dx) in steps {
        tc.t_par[patch * NP + slot] = irys_recon_net::scene::norm_mm((ln + dx - mean).exp());
        tc.w_par[patch * NP + slot] = 1.0;
    }
    tc
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = PathBuf::from(arg(&a, "--data", "out/neural/data".to_string()));
    let wpath = PathBuf::from(arg(&a, "--weights", "out/neural/run2/weights.json".to_string()));
    let set: String = arg(&a, "--set", "v3-ftb".to_string());
    let out = PathBuf::from(arg(&a, "--out", "out/neural/v3/ft".to_string()));
    let steps: usize = arg(&a, "--steps", 120);
    let lr: f64 = arg(&a, "--lr", 5e-5);
    let per_step: usize = arg(&a, "--images", 8);
    let w_teach: f64 = arg(&a, "--teach", 1.0);
    let w_pg: f64 = arg(&a, "--pg", 0.05);
    let sw: usize = arg(&a, "--size", 256);
    let anchor_n: usize = arg(&a, "--anchor", 4);
    let anchor_w: f64 = arg(&a, "--anchor-weight", 0.5);
    // One trainer worker per image: the teacher's images and the anchors.
    let workers = per_step + anchor_n;
    let opt = Opt { size: (sw, sw * 3 / 4), levels: vec![1.0], highpass: arg(&a, "--highpass", 2.5), k_samples: arg(&a, "--k", 4), max_obj: 10, gn: arg(&a, "--gn", 1) == 1, refine_cam: arg(&a, "--refine-cam", 1) == 1, scale_free: arg(&a, "--scale-free", 1) == 1 };
    std::fs::create_dir_all(out.join("app")).unwrap();
    let gpu = irys_diff::gpu::context().expect("gpu");
    let photos: Vec<Sample> = read_dir(&data, &set);
    let his: Vec<Vec<u8>> = photos
        .iter()
        .map(|s| {
            let hi = image::open(hi_path(&data, &set, s.scene.seed)).unwrap().to_rgb8();
            resample_rgb8(&hi, opt.size).iter().flat_map(|p| p.map(irys_recon::render::srgb8)).collect()
        })
        .collect();
    let anchor: Vec<Sample> = if anchor_n > 0 { read_dir(&data, "train-a-0000").into_iter().take(2000).collect() } else { Vec::new() };
    let w = Weights::load(&wpath).unwrap();
    let mut net = Infer::new(&w, workers);
    // The appearance of every photo, once, with the starting network.
    let t0 = std::time::Instant::now();
    let mut cache: Vec<Cached> = Vec::with_capacity(photos.len());
    for (k, s) in photos.iter().enumerate() {
        let path = out.join("app").join(format!("{}.json", s.scene.seed));
        if let Some(c) = std::fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str::<Cached>(&t).ok()) {
            cache.push(c);
            continue;
        }
        let o = net.run(&[patches(&s.rgb)]).pop().unwrap();
        let (_, cam, kept) = top_scene(&o, s.scene.cam.hfov, opt.max_obj);
        let objs: Vec<Object> = kept.into_iter().map(|k| k.1).collect();
        let mut app = Appearance::grey(objs.len());
        let mut cmp = Compare3::new(gpu, &objs, &local_meshes(&objs), &app, &cam, opt.size).unwrap();
        faithful::fit_appearance(&mut cmp, &cam, &his[k], &mut app, &FitOpts::default());
        let target = linearise(&his[k], app.gamma);
        let sigma = noise_sigma(&target, opt.size);
        let pre = Cached { app: app.clone(), sig_levels: Vec::new(), tau: 1.0, sigma };
        let ctx = ctx_of(gpu, &pre, &target, cam, &opt);
        let st = State::new(objs.clone());
        let sig_levels = search3::noise_levels(&st, &ctx);
        let ctx = Ctx3 { sig_levels: sig_levels.clone(), ..ctx };
        let tau = search3::dispersion(&st, &ctx);
        let c = Cached { app, sig_levels, tau, sigma };
        std::fs::write(&path, serde_json::to_string(&c).unwrap()).unwrap();
        cache.push(c);
        if k % 50 == 0 {
            println!("appearance {k}/{} {:.0}s", photos.len(), t0.elapsed().as_secs_f64());
        }
    }
    let mut tr = Trainer::new(w.arch, 1, workers, Some(w.f64s()), 0);
    let mut rng = Mix(99);
    let mut curve = Vec::new();
    let t0 = std::time::Instant::now();
    for step in 1..=steps {
        let picks: Vec<usize> = (0..per_step).map(|_| (rng.uniform() * photos.len() as f64) as usize % photos.len()).collect();
        net.set_weights(&tr.params);
        let xs: Vec<Vec<f64>> = picks.iter().map(|&k| patches(&photos[k].rgb)).collect();
        let outs = net.run(&xs);
        // The device renders one image at a time; each image's teacher is sequential.
        let teach_v: Vec<Teach> = outs.iter().zip(&picks).enumerate().map(|(i, (o, &k))| teach(gpu, o, &photos[k], &his[k], &cache[k], &opt, step as u64 * 1000 + i as u64)).collect();
        let n = per_step as f64;
        let mut batches: Vec<Batch> = xs
            .iter()
            .zip(&teach_v)
            .map(|(x, t)| {
                let mut tg = zero_targets();
                tg.t_tok = t.t_tok.clone();
                tg.w_tok = t.w_tok.iter().map(|v| v * w_pg / n).collect();
                if opt.gn {
                    tg.t_par = t.t_par.clone();
                    tg.w_par = t.w_par.iter().map(|v| v * w_teach / n).collect();
                    tg.t_yaw = t.t_yaw.clone();
                    tg.w_yaw = t.w_yaw.iter().map(|v| v * w_teach / n).collect();
                }
                let mut b = Batch::new(std::slice::from_ref(x), &[tg]);
                if !opt.gn {
                    b.g_par = t.g_par.iter().map(|v| v * w_teach / n).collect();
                    b.g_yaw = t.g_yaw.iter().map(|v| v * w_teach / n).collect();
                }
                b
            })
            .collect();
        for _ in 0..anchor_n {
            let s = &anchor[(rng.uniform() * anchor.len() as f64) as usize % anchor.len()];
            let mut t = irys_recon_net::data::targets(s, anchor_n);
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
        if step % 5 == 0 || step == steps {
            println!("step {step:4}  {:.0}s  render loss {loss:.4}  top-1 MDL {mdl:.0}  |g| {gn:.3}", t0.elapsed().as_secs_f64());
            std::fs::write(out.join("curve.json"), serde_json::to_string_pretty(&curve).unwrap()).unwrap();
            tr.weights().save(&out.join("weights.json")).unwrap();
        }
    }
}
