//! Supervised training on the synthetic stream:
//! `neural_train --data out/neural/data --out out/neural/run --steps 12000`.
//! Writes `weights.json`, `curve.json` (train and held-out losses).

use std::path::PathBuf;

use irys_recon::jac::Mix;
use irys_recon_net::data::{Sample, patches, read_dir, targets};
use irys_recon_net::net::{Arch, H, LOSS_NAMES, W};
use irys_recon_net::train::{Batch, Trainer, Weights};
use rayon::prelude::*;

fn arg<T: std::str::FromStr>(a: &[String], k: &str, d: T) -> T {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// The mirror image about the camera's vertical plane (every family is
/// mirror-symmetric in its own frame, so only poses change).
pub fn flip(s: &Sample) -> Sample {
    let mut f = s.clone();
    let ex = s.scene.cam.eye[0];
    for o in &mut f.scene.objects {
        o.pose[0] = 2.0 * ex - o.pose[0];
        o.pose[2] = -o.pose[2];
    }
    for y in 0..H {
        for x in 0..W {
            let (a, b) = (y * W + x, y * W + (W - 1 - x));
            f.inst[a] = s.inst[b];
            f.depth[a] = s.depth[b];
            for c in 0..3 {
                f.rgb[a * 3 + c] = s.rgb[b * 3 + c];
            }
        }
    }
    f
}

/// Photometric jitter: a per-channel gain and a tone power, on the bytes.
pub fn jitter(s: &mut Sample, seed: u64) {
    let mut r = Mix(seed);
    let gamma = 0.8 + 0.45 * r.uniform();
    let gain = [0.85 + 0.3 * r.uniform(), 0.85 + 0.3 * r.uniform(), 0.85 + 0.3 * r.uniform()];
    for (k, b) in s.rgb.iter_mut().enumerate() {
        let v = (*b as f64 / 255.0).powf(gamma) * gain[k % 3];
        *b = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    }
}

fn batches(samples: &[&Sample], workers: usize, per: usize, flips: &[bool], seeds: Option<&[u64]>) -> Vec<Batch> {
    let total = workers * per;
    let prepared: Vec<(Vec<f64>, irys_recon_net::data::Targets)> = samples
        .par_iter()
        .zip(flips)
        .enumerate()
        .map(|(i, (s, &fl))| {
            let mut s2;
            let s = if fl || seeds.is_some() {
                s2 = if fl { flip(s) } else { (*s).clone() };
                if let Some(sd) = seeds {
                    jitter(&mut s2, sd[i]);
                }
                &s2
            } else {
                *s
            };
            (patches(&s.rgb), targets(s, total))
        })
        .collect();
    prepared
        .chunks(per)
        .map(|c| {
            let xs: Vec<Vec<f64>> = c.iter().map(|p| p.0.clone()).collect();
            let ts: Vec<_> = c.iter().map(|p| p.1.clone()).collect();
            Batch::new(&xs, &ts)
        })
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = PathBuf::from(arg(&a, "--data", "out/neural/data".to_string()));
    let out = PathBuf::from(arg(&a, "--out", "out/neural/run".to_string()));
    let steps: usize = arg(&a, "--steps", 12000);
    let workers: usize = arg(&a, "--workers", 12);
    let per: usize = arg(&a, "--per", 4);
    let lr_max: f64 = arg(&a, "--lr", 1e-3);
    let wd: f64 = arg(&a, "--wd", 1e-4);
    let arch = Arch { d: arg(&a, "--d", 128), tok_hidden: arg(&a, "--tok", 256), ch_hidden: arg(&a, "--ch", 384), blocks: arg(&a, "--blocks", 6) };
    std::fs::create_dir_all(&out).unwrap();
    let t0 = std::time::Instant::now();
    let train = read_dir(&data, "train-a");
    let val = read_dir(&data, "test-a");
    println!("{} train, {} held-out samples ({:.1}s)", train.len(), val.len(), t0.elapsed().as_secs_f64());
    let mut tr = Trainer::new(arch, per, workers, None, 1);
    println!("{} parameters, {} workers × {per}, native {}", tr.parameter_count(), workers, tr.native);
    let total = workers * per;
    let mut rng = Mix(42);
    let mut curve: Vec<serde_json::Value> = Vec::new();
    let mut run = [0.0f64; 11];
    let mut nrun = 0.0;
    let val_n = (val.len() / total).min(4) * total;
    let t_train = std::time::Instant::now();
    let mut best_val = f64::INFINITY;
    for step in 1..=steps {
        let warm = 300.0;
        let lr = if (step as f64) < warm {
            lr_max * step as f64 / warm
        } else {
            let p = (step as f64 - warm) / (steps as f64 - warm);
            1e-5 + 0.5 * (lr_max - 1e-5) * (1.0 + (std::f64::consts::PI * p).cos())
        };
        let picks: Vec<&Sample> = (0..total).map(|_| &train[(rng.uniform() * train.len() as f64) as usize % train.len()]).collect();
        let flips: Vec<bool> = (0..total).map(|_| rng.uniform() < 0.5).collect();
        let seeds: Vec<u64> = (0..total).map(|_| (rng.uniform() * 1e15) as u64).collect();
        let bs = batches(&picks, workers, per, &flips, Some(&seeds));
        let (l, g) = tr.grads(&bs);
        let gn = tr.adam(&g, lr, 5.0, wd);
        for (r, v) in run.iter_mut().zip(&l) {
            *r += v;
        }
        nrun += 1.0;
        if step % 100 == 0 || step == steps {
            let mut rec = serde_json::Map::new();
            rec.insert("step".into(), step.into());
            rec.insert("images".into(), (step * total).into());
            rec.insert("seconds".into(), t_train.elapsed().as_secs_f64().into());
            rec.insert("lr".into(), lr.into());
            rec.insert("grad_norm".into(), gn.into());
            for (k, n) in LOSS_NAMES.iter().enumerate() {
                rec.insert(format!("train_{n}"), (run[k] / nrun).into());
            }
            if step % 500 == 0 || step == steps {
                let mut vl = [0.0f64; 11];
                let mut nv = 0.0;
                for c in val[..val_n].chunks(total) {
                    let refs: Vec<&Sample> = c.iter().collect();
                    let bs = batches(&refs, workers, per, &vec![false; total], None);
                    let (l, _) = tr.grads(&bs);
                    for (r, v) in vl.iter_mut().zip(&l) {
                        *r += v;
                    }
                    nv += 1.0;
                }
                for (k, n) in LOSS_NAMES.iter().enumerate() {
                    rec.insert(format!("val_{n}"), (vl[k] / nv).into());
                }
                tr.weights().save(&out.join("weights.json")).unwrap();
                let v = vl[0] / nv;
                if v < best_val {
                    best_val = v;
                    tr.weights().save(&out.join("weights-best.json")).unwrap();
                    rec.insert("best".into(), true.into());
                }
            }
            println!(
                "step {step:6}  {:.0}s  lr {lr:.2e}  |g| {gn:7.3}  loss {:.4}  fg {:.4} off {:.4} dep {:.4} ctr {:.4} cls {:.4} tok {:.4} par {:.4} yaw {:.4} plane {:.4}{}",
                t_train.elapsed().as_secs_f64(),
                run[0] / nrun,
                run[1] / nrun,
                run[2] / nrun,
                run[3] / nrun,
                run[4] / nrun,
                run[5] / nrun,
                run[6] / nrun,
                run[7] / nrun,
                run[8] / nrun,
                run[9] / nrun,
                rec.get("val_total").map(|v| format!("  val {:.4}", v.as_f64().unwrap())).unwrap_or_default()
            );
            curve.push(serde_json::Value::Object(rec));
            std::fs::write(out.join("curve.json"), serde_json::to_string_pretty(&curve).unwrap()).unwrap();
            run = [0.0; 11];
            nrun = 0.0;
        }
    }
    let w: Weights = tr.weights();
    w.save(&out.join("weights.json")).unwrap();
    println!("done: {} steps, {:.1} min", steps, t_train.elapsed().as_secs_f64() / 60.0);
}
