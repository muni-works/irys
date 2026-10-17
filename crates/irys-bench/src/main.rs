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

/// Render `name` with irys: the reference (cached) and the sweep.
fn irys(a: &Args, out: &Path, name: &str, w: usize, h: usize) {
    let ctx = match gpu::context() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("no GPU ({e}): irys-bench needs one");
            std::process::exit(1);
        }
    };
    let dir = out.join(name);
    std::fs::create_dir_all(&dir).unwrap();
    let variant = a.get("variant", "base");
    let call_max = a.num("call-spp", 16);
    let mut wf: HashMap<usize, (scenes::Bench, GpuPrb)> = HashMap::new();
    // One frame of `spp` paths per pixel: `spp / call` calls of `call` spp
    // each, seeds `seed0 + k`, averaged. Returns the image and the seconds
    // the calls took (the wavefronts are built and warm beforehand).
    let mut render = |spp: usize, seed0: u64| -> (Vec<f32>, f64) {
        let call = spp.min(call_max);
        assert!(spp % call == 0, "spp {spp} is not a multiple of {call}");
        let (b, g) = wf.entry(call).or_insert_with(|| {
            let b = scenes::build(name, w, h, call);
            let g = GpuPrb::new(ctx, &b.scene);
            (b, g)
        });
        // warm: pipelines compiled, buffers touched
        let _ = g.image(&b.scene, &b.theta, u64::MAX);
        let n = spp / call;
        let t = Instant::now();
        let mut acc = vec![0.0f64; w * h * 3];
        for k in 0..n {
            let img = g.image(&b.scene, &b.theta, seed0 + k as u64);
            for (a, v) in acc.iter_mut().zip(img) {
                *a += v as f64;
            }
        }
        let secs = t.elapsed().as_secs_f64();
        (acc.iter().map(|v| (v / n as f64) as f32).collect(), secs)
    };
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
        let (img, secs) = render(spp, 1000 + spp as u64 * 7919);
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
            println!("{name} cycles: {:5} spp {:8.4} s relMSE {:.5}", r.2, r.0, r.1);
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
        for (v, c) in variants.iter().zip(&icurves) {
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
        let p = docs.join(format!("phase3-{name}.png"));
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
