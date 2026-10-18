//! `irys-bench`: phase 3's equal-time harness. See `scripts/bench.sh`.
//!
//! ```text
//! irys-bench export  --scene room --out out/bench            # scene.json for Cycles
//! irys-bench irys    --scene room --out out/bench [--variant base] [--ref-spp 8192] [--max-spp 256]
//! irys-bench report  --out out/bench [--scenes pair,room] [--variants base,...] [--docs docs]
//! ```
//!
//! Sizes default to 400 × 300. Everything a run writes lands in
//! `<out>/<scene>/`.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    time::Instant,
};

use irys_bench::{
    error_at, export, font, mean_rgb, read_f32, read_json, relmse, scenes, to_srgb8, write_f32,
    write_json, write_png,
};
use irys_diff::gpu::{self, GpuPrb};
use serde_json::{Value, json};

struct Args(HashMap<String, String>);

impl Args {
    fn get(&self, k: &str, d: &str) -> String {
        self.0.get(k).cloned().unwrap_or_else(|| d.to_string())
    }
    fn num(&self, k: &str, d: usize) -> usize {
        self.0.get(k).map_or(d, |v| v.parse().expect(k))
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cmd = argv.first().cloned().unwrap_or_default();
    let mut m = HashMap::new();
    let mut i = 1;
    while i < argv.len() {
        let k = argv[i].trim_start_matches("--").to_string();
        let v = argv.get(i + 1).cloned().unwrap_or_default();
        m.insert(k, v);
        i += 2;
    }
    let a = Args(m);
    let out = PathBuf::from(a.get("out", "out/bench"));
    let (w, h) = (a.num("w", 400), a.num("h", 300));
    match cmd.as_str() {
        "export" => {
            for name in a.get("scene", "pair,room").split(',') {
                let b = scenes::build(name, w, h, 1);
                let dir = out.join(name);
                std::fs::create_dir_all(&dir).unwrap();
                write_json(&dir.join("scene.json"), &export(&b.scene, &b.theta, name));
                println!("{name}: wrote {}", dir.join("scene.json").display());
            }
        }
        "irys" => {
            for name in a.get("scene", "pair,room").split(',') {
                irys(&a, &out, name, w, h);
            }
        }
        "report" => report(&a, &out),
        "speed" => {
            for name in a.get("scene", "pair,room").split(',') {
                speed(&a, &out, name, w, h);
            }
        }
        "rtcheck" => {
            let ctx = context();
            for name in a.get("scene", "pair,room").split(',') {
                let b = scenes::build(name, w, h, a.num("spp", 4));
                let mut g = GpuPrb::new(ctx, &b.scene);
                let mut imgs = Vec::new();
                for hw in [false, true, true, false, true] {
                    g.set_hardware_rt(hw);
                    imgs.push(g.image(&b.scene, &b.theta, 5));
                }
                let diff = |x: &[f32], y: &[f32]| x.iter().zip(y).filter(|(p, q)| p != q).count();
                let m = |x: &[f32]| x.iter().map(|&v| v as f64).sum::<f64>() / x.len() as f64;
                println!(
                    "{name}: means sw {:.5} hw {:.5} {:.5} sw {:.5} hw {:.5}; hw-hw diff {} {}, sw-sw {}, hw-sw {}",
                    m(&imgs[0]), m(&imgs[1]), m(&imgs[2]), m(&imgs[3]), m(&imgs[4]),
                    diff(&imgs[1], &imgs[2]), diff(&imgs[1], &imgs[4]), diff(&imgs[0], &imgs[3]), diff(&imgs[0], &imgs[1])
                );
            }
        }
        "retime" => {
            for name in a.get("scene", "pair,room").split(',') {
                retime(&a, &out, name, w, h);
            }
        }
        "png" => {
            let img = read_f32(Path::new(&a.get("in", "")));
            let m = mean_rgb(&img);
            let exposure: f64 = a
                .0
                .get("exposure")
                .map_or(0.8 / (m.iter().sum::<f64>() / 3.0).max(1e-6), |v| v.parse().unwrap());
            write_png(Path::new(&a.get("png", "out.png")), w, h, &to_srgb8(&img, exposure));
            println!("mean {m:?}, exposure {exposure:.3}");
        }
        _ => {
            eprintln!("usage: irys-bench export|irys|report [--scene s] [--out dir]");
            std::process::exit(2);
        }
    }
}

/// One irys configuration of one scene, ready to render frames: a warm
/// wavefront per calls-of-`call`-spp size.
struct Runner {
    ctx: &'static irys_gpu::GpuContext,
    name: String,
    w: usize,
    h: usize,
    features: Vec<String>,
    call_max: usize,
    wf: HashMap<usize, (scenes::Bench, GpuPrb)>,
}

impl Runner {
    fn new(ctx: &'static irys_gpu::GpuContext, name: &str, w: usize, h: usize, variant: &str, call: Option<usize>) -> Self {
        let features: Vec<String> = variant.split('+').map(str::to_string).collect();
        let has = |f: &str| features.iter().any(|x| x == f);
        // ReSTIR reuses across the frame, which must fit one chunk; the
        // guide learns between calls, so it wants them small.
        let call_max = call.unwrap_or(if has("restir") || has("guide") { 4 } else { 16 });
        Self { ctx, name: name.to_string(), w, h, features, call_max, wf: HashMap::new() }
    }

    fn has(&self, f: &str) -> bool {
        self.features.iter().any(|x| x == f)
    }

    /// One frame of `spp` paths per pixel: `spp / call` calls of `call` spp
    /// each, averaged. Returns the image and the seconds the calls took (the
    /// wavefronts are built and warm beforehand, the history reset).
    fn render(&mut self, spp: usize, seed0: u64) -> (Vec<f32>, f64) {
        let (restir, guide) = (self.has("restir"), self.has("guide"));
        let call = spp.min(self.call_max);
        assert!(spp % call == 0, "spp {spp} is not a multiple of {call}");
        let (name, w, h, ctx) = (self.name.clone(), self.w, self.h, self.ctx);
        let features: Vec<&str> = self.features.iter().map(String::as_str).collect();
        let (b, g) = self.wf.entry(call).or_insert_with(|| {
            let mut b = scenes::build(&name, w, h, call);
            scenes::apply(&mut b.scene, &features);
            let g = GpuPrb::new(ctx, &b.scene);
            (b, g)
        });
        // warm: pipelines compiled, buffers touched
        let _ = g.image(&b.scene, &b.theta, u64::MAX);
        g.reset_history();
        let n = spp / call;
        let t = Instant::now();
        let mut acc = vec![0.0f64; w * h * 3];
        for k in 0..n {
            // Guiding: learn during the first half of the calls (at least
            // one), rebuilding the guide after each; every call counts
            // towards the image.
            let train = guide && (k == 0 || 2 * (k + 1) <= n);
            if guide {
                g.set_guide_training(train);
            }
            // Sobol: the low word is the frame's place in the sequence
            // (consecutive blocks), the high word the scramble.
            let seed = if b.scene.qmc {
                ((seed0 & 0xffff_ffff) << 32) | k as u64
            } else {
                seed0 + k as u64
            };
            let img = g.image(&b.scene, &b.theta, seed);
            if restir {
                g.advance();
            }
            if train {
                g.guide_build(&b.scene);
            }
            for (a, v) in acc.iter_mut().zip(img) {
                *a += v as f64;
            }
        }
        let secs = t.elapsed().as_secs_f64();
        (acc.iter().map(|v| (v / n as f64) as f32).collect(), secs)
    }
}

fn context() -> &'static irys_gpu::GpuContext {
    match gpu::context() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}): irys-bench needs one");
            std::process::exit(1);
        }
    }
}

/// Re-time variants on a shared machine: `rounds` rounds, each rendering
/// `spp` with every variant in turn (so a slow spell hits them alike), the
/// fastest time per variant kept; each variant's sweep times are then
/// rescaled by (clean time / its sweep's time at `spp`). Images and relMSE
/// are untouched.
fn retime(a: &Args, out: &Path, name: &str, w: usize, h: usize) {
    let ctx = context();
    let variants: Vec<String> = a.get("variants", "base").split(',').map(str::to_string).collect();
    let spp = a.num("spp", 256);
    let mut runners: Vec<Runner> = variants.iter().map(|v| Runner::new(ctx, name, w, h, v, None)).collect();
    let mut best = vec![f64::INFINITY; variants.len()];
    for _ in 0..a.num("rounds", 3) {
        for (k, r) in runners.iter_mut().enumerate() {
            best[k] = best[k].min(r.render(spp, 77).1);
        }
    }
    let dir = out.join(name);
    for (v, t) in variants.iter().zip(&best) {
        let path = dir.join(format!("irys_{v}.json"));
        let mut j = read_json(&path);
        let rows = j["rows"].as_array_mut().unwrap();
        let at = rows.iter().find(|r| r["spp"].as_u64() == Some(spp as u64)).expect("spp level")["seconds"]
            .as_f64()
            .unwrap();
        // Fastest observation: a slower round never slows the sweep down.
        let t = t.min(at);
        let f = t / at;
        for r in rows.iter_mut() {
            let s = r["seconds"].as_f64().unwrap();
            r["seconds"] = json!(s * f);
        }
        j["retimed"] = json!({"spp": spp, "clean_seconds": t, "factor": f});
        write_json(&path, &j);
        println!("{name} {v}: {spp} spp clean {t:.4} s, sweep had {at:.4} s, x{f:.3}");
    }
}

/// Time per sample, fastest of `rounds` renders of `spp`, per variant, with
/// the relMSE against the cached irys reference (speed work; no files
/// written).
fn speed(a: &Args, out: &Path, name: &str, w: usize, h: usize) {
    let ctx = context();
    let reference = read_f32(&out.join(name).join("irys_ref.f32"));
    let spp = a.num("spp", 256);
    for v in a.get("variants", "base").split(',') {
        let mut r = Runner::new(ctx, name, w, h, v, None);
        let mut best = f64::INFINITY;
        let mut img = Vec::new();
        for _ in 0..a.num("rounds", 3) {
            let (i, t) = r.render(spp, 77);
            best = best.min(t);
            img = i;
        }
        let hw = r.wf.values().next().is_some_and(|(_, g)| g.hardware_rt());
        println!(
            "{name} {v}: {:.3} ms/spp ({spp} spp, {}), relMSE {:.3e}",
            best / spp as f64 * 1e3,
            if hw { "hw rt" } else { "sw walk" },
            relmse(&img, &reference)
        );
    }
}

/// Render `name` with irys: the reference (cached) and the sweep.
fn irys(a: &Args, out: &Path, name: &str, w: usize, h: usize) {
    let ctx = context();
    let dir = out.join(name);
    std::fs::create_dir_all(&dir).unwrap();
    let variant = a.get("variant", "base");
    let mut runner = Runner::new(ctx, name, w, h, &variant, a.0.get("call-spp").map(|v| v.parse().unwrap()));
    let mut render = |spp: usize, seed0: u64| runner.render(spp, seed0);
    let ref_path = dir.join("irys_ref.f32");
    let ref_spp = a.num("ref-spp", 8192);
    if !ref_path.exists() {
        let (r, secs) = render(ref_spp, 1 << 40);
        write_f32(&ref_path, &r);
        write_json(
            &dir.join("irys_ref.json"),
            &json!({"spp": ref_spp, "seconds": secs, "mean": mean_rgb(&r)}),
        );
        println!("{name}: irys reference {ref_spp} spp in {secs:.1} s");
    }
    let reference = read_f32(&ref_path);
    let max_spp = a.num("max-spp", 256);
    let mut rows = Vec::new();
    let mut spp = 1;
    while spp <= max_spp {
        // The machine is shared: time each level `reps` times (same seed,
        // same image) and keep the fastest.
        let (img, mut secs) = render(spp, 1000 + spp as u64 * 7919);
        for _ in 1..a.num("reps", 2) {
            secs = secs.min(render(spp, 1000 + spp as u64 * 7919).1);
        }
        let e = relmse(&img, &reference);
        println!("{name} irys {variant}: {spp:5} spp  {secs:8.4} s  relMSE {e:.5}");
        write_f32(&dir.join(format!("irys_{variant}_{spp}.f32")), &img);
        rows.push(json!({"spp": spp, "seconds": secs, "relmse": e}));
        spp *= 2;
    }
    write_json(
        &dir.join(format!("irys_{variant}.json")),
        &json!({"variant": variant, "size": [w, h], "rows": rows}),
    );
}

fn curve(v: &Value) -> Vec<(f64, f64, usize)> {
    v["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            (
                r["seconds"].as_f64().unwrap(),
                r["relmse"].as_f64().unwrap(),
                r["spp"].as_u64().unwrap() as usize,
            )
        })
        .collect()
}

/// The row of `c` whose time is nearest `t` (in log time).
fn nearest(c: &[(f64, f64, usize)], t: f64) -> (f64, f64, usize) {
    *c.iter()
        .min_by(|a, b| {
            (a.0.ln() - t.ln())
                .abs()
                .partial_cmp(&(b.0.ln() - t.ln()).abs())
                .unwrap()
        })
        .unwrap()
}

/// The equal-time table and the comparison images.
fn report(a: &Args, out: &Path) {
    let docs = PathBuf::from(a.get("docs", "docs"));
    let variants: Vec<String> = a
        .get("variants", "base")
        .split(',')
        .map(str::to_string)
        .collect();
    let mut table = String::new();
    table.push_str("| scene | budget (s) | Cycles relMSE |");
    for v in &variants {
        table.push_str(&format!(" irys {v} relMSE | ratio |"));
    }
    table.push('\n');
    table.push_str("|---|---|---|");
    for _ in &variants {
        table.push_str("---|---|");
    }
    table.push('\n');
    let mut summary = Vec::new();
    for name in a.get("scenes", "pair,room").split(',') {
        let dir = out.join(name);
        let cy = read_json(&dir.join("cycles.json"));
        let (w, h) = (
            cy["size"][0].as_u64().unwrap() as usize,
            cy["size"][1].as_u64().unwrap() as usize,
        );
        let cref = read_f32(&dir.join("cycles_ref.f32"));
        let iref = read_f32(&dir.join("irys_ref.f32"));
        // Cycles' sweep scored against its own reference.
        let ccurve: Vec<(f64, f64, usize)> = cy["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                let spp = r["spp"].as_u64().unwrap() as usize;
                let img = read_f32(&dir.join(format!("cycles_{spp}.f32")));
                (r["seconds"].as_f64().unwrap(), relmse(&img, &cref), spp)
            })
            .collect();
        let icurves: Vec<Vec<(f64, f64, usize)>> = variants
            .iter()
            .map(|v| curve(&read_json(&dir.join(format!("irys_{v}.json")))))
            .collect();
        for r in &ccurve {
            println!("{name} cycles: {:5} spp {:8.4} s relMSE {:.3e}", r.2, r.0, r.1);
        }
        let (mc, mi) = (mean_rgb(&cref), mean_rgb(&iref));
        let cross = relmse(&iref, &cref);
        println!(
            "{name}: reference means Cycles {:.4} {:.4} {:.4}, irys {:.4} {:.4} {:.4}; irys ref vs Cycles ref relMSE {cross:.4}",
            mc[0], mc[1], mc[2], mi[0], mi[1], mi[2]
        );
        // Budgets: Cycles' own times at a few of its spp levels.
        let budgets: Vec<f64> = ccurve
            .iter()
            .filter(|r| [64, 256, 1024].contains(&r.2))
            .map(|r| r.0)
            .collect();
        let mut rec = json!({"scene": name, "cycles_ref_mean": mc, "irys_ref_mean": mi, "ref_cross_relmse": cross, "budgets": []});
        for &t in &budgets {
            let ec = error_at(&strip(&ccurve), t);
            table.push_str(&format!("| {name} | {t:.3} | {ec:.4} |"));
            let mut b = json!({"seconds": t, "cycles": ec});
            for (v, c) in variants.iter().zip(&icurves) {
                let ei = error_at(&strip(c), t);
                table.push_str(&format!(" {ei:.4} | {:.2} |", ei / ec));
                b[v] = json!(ei);
            }
            table.push('\n');
            rec["budgets"].as_array_mut().unwrap().push(b);
        }
        summary.push(rec);
        // The image: Cycles at its 16-spp time, each irys variant at its
        // nearest level, both references.
        let t16 = ccurve.iter().find(|r| r.2 == 64).map_or(ccurve[0].0, |r| r.0);
        let exposure = 0.8 / (mc.iter().sum::<f64>() / 3.0).max(1e-6);
        let mut panels: Vec<(Vec<f32>, String)> = Vec::new();
        let cr = nearest(&ccurve, t16);
        panels.push((
            read_f32(&dir.join(format!("cycles_{}.f32", cr.2))),
            format!("CYCLES {} SPP {:.2}S RELMSE {:.4}", cr.2, cr.0, cr.1),
        ));
        let shown: Vec<String> = match a.0.get("image-variants") {
            Some(v) => v.split(',').map(str::to_string).collect(),
            None => vec![variants[0].clone(), variants[variants.len() - 1].clone()],
        };
        for (v, c) in variants.iter().zip(&icurves) {
            if !shown.contains(v) {
                continue;
            }
            let r = nearest(c, t16);
            panels.push((
                read_f32(&dir.join(format!("irys_{v}_{}.f32", r.2))),
                format!("IRYS {} {} SPP {:.2}S RELMSE {:.4}", v, r.2, r.0, r.1),
            ));
        }
        panels.push((cref.clone(), "CYCLES REFERENCE".into()));
        panels.push((iref.clone(), "IRYS REFERENCE".into()));
        let cols = 2;
        let rows = panels.len().div_ceil(cols);
        let (pw, ph) = (cols * w, rows * h);
        let mut canvas = vec![0u8; pw * ph * 3];
        for (k, (img, text)) in panels.iter().enumerate() {
            let (ox, oy) = ((k % cols) * w, (k / cols) * h);
            let px = to_srgb8(img, exposure);
            for y in 0..h {
                for x in 0..w {
                    let s = (y * w + x) * 3;
                    let d = ((oy + y) * pw + ox + x) * 3;
                    canvas[d..d + 3].copy_from_slice(&px[s..s + 3]);
                }
            }
            font::label(&mut canvas, pw, ph, ox + 4, oy + 4, 1, text);
        }
        std::fs::create_dir_all(&docs).unwrap();
        let p = docs.join(format!("{}-{name}.png", a.get("prefix", "phase3")));
        write_png(&p, pw, ph, &canvas);
        println!("wrote {}", p.display());
    }
    println!("\n{table}");
    write_json(&out.join("report.json"), &json!({"variants": variants, "scenes": summary}));
    std::fs::write(out.join("report.md"), &table).unwrap();
}

/// `(seconds, relMSE)` by spp, with time made non-decreasing in spp (a
/// renderer's fixed overhead makes the low levels' times jitter).
fn strip(c: &[(f64, f64, usize)]) -> Vec<(f64, f64)> {
    let mut v: Vec<(f64, f64, usize)> = c.to_vec();
    v.sort_by_key(|r| r.2);
    let mut t: f64 = 0.0;
    v.iter()
        .map(|r| {
            t = t.max(r.0) * (1.0 + 1e-9);
            (t, r.1)
        })
        .collect()
}
