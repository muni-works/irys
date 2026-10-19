//! The v0 gate: 20 random parts, recovered from irys renders.
//!
//! `cargo run --release -p irys-recon --example recon_v0 -- [--seeds A..B] [--views N] [--out DIR] [--single]`
//!
//! Writes `DIR/report.json` and `DIR/report.md` (per part: pass/fail,
//! topology, dimension errors, iterations, renders, time), and with
//! `--figures` the two docs figures.

use std::{fs, io::Write, path::PathBuf};

use irys_recon::{canon, dsl::Program, fit::Target, generate, model::*, render, search::*};

/// The gate set: seeds 3000..3020, run once, after the solver was frozen.
/// The dev set used while building it is 0..100, 1000..1003 and 2000..2002:
/// 1000.. and 2000.. were earlier gate sets whose first failures were looked
/// at and fixed, which made them dev sets.
const GATE: std::ops::Range<u64> = 3000..3020;
const TARGET_SPP: usize = 64;
const TOL_MM: f64 = 0.5;

#[derive(serde::Serialize, serde::Deserialize)]
struct Part {
    seed: u64,
    truth: String,
    recovered: String,
    truth_signature: String,
    recovered_signature: String,
    topology_ok: bool,
    max_error_mm: Option<f64>,
    errors: Vec<(String, f64)>,
    pass: bool,
    sigma: Vec<(String, f64)>,
    weak: Vec<(f64, Vec<(String, f64)>)>,
    mdl: irys_recon::mdl::Score,
    mdl_truth: irys_recon::mdl::Score,
    rounds: usize,
    candidates: usize,
    iters: usize,
    evals: usize,
    seconds: f64,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut seeds, mut views, mut out) = (GATE, 3usize, PathBuf::from("out/recon_v0"));
    let mut rescore = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--seeds" => {
                let v = args.next().unwrap();
                let (a, b) = v.split_once("..").unwrap();
                seeds = a.parse().unwrap()..b.parse().unwrap();
            }
            "--views" => views = args.next().unwrap().parse().unwrap(),
            "--out" => out = args.next().unwrap().into(),
            "--rescore" => rescore = true,
            _ => panic!("unknown argument {a}"),
        }
    }
    fs::create_dir_all(&out).unwrap();
    if rescore {
        // Re-judge a finished run's saved programs against the current normal
        // form (no rendering): topology, errors, pass.
        let mut parts: Vec<Part> = serde_json::from_str(&fs::read_to_string(out.join("report.json")).unwrap()).unwrap();
        for p in &mut parts {
            let (t, r) = (Program::from_text(&p.truth).unwrap(), Program::from_text(&p.recovered).unwrap());
            p.topology_ok = canon::same_topology(&t, &r);
            p.recovered_signature = canon::canonical(&r).signature();
            p.errors = canon::errors(&t, &r).unwrap_or_default();
            p.max_error_mm = p.topology_ok.then(|| p.errors.iter().fold(0.0f64, |m, e| m.max(e.1)));
            p.pass = p.max_error_mm.is_some_and(|e| e <= TOL_MM);
            println!("seed {}: {}", p.seed, if p.pass { "PASS" } else { "FAIL" });
        }
        write_report(&out, &parts, views);
        return;
    }
    let setup = Setup::new(render::views().into_iter().take(views).collect());
    let bg = background(&setup, TARGET_SPP);
    let mut parts = Vec::new();
    for seed in seeds.clone() {
        let truth = generate::program(seed);
        let t = target(&truth, &setup, TARGET_SPP, 1_000_003 * seed + 11).unwrap();
        let tg = Target::new(t.clone(), &bg, setup.film.size);
        let r = recover(&tg, &bg, &setup, TARGET_SPP, &mut Algorithmic, &Options::default());
        let topology_ok = canon::same_topology(&truth, &r.program);
        let errors = canon::errors(&truth, &r.program).unwrap_or_default();
        let max_error_mm = topology_ok.then(|| errors.iter().fold(0.0f64, |m, e| m.max(e.1)));
        let pass = max_error_mm.is_some_and(|e| e <= TOL_MM);
        // The truth's own score at full resolution, for the failure analysis.
        let mut f = irys_recon::fit::Fitter::new(setup.clone(), tg.clone(), TARGET_SPP, &truth);
        let ft = f.fit(&truth, &[irys_recon::fit::Level { sigma: 0.0, lambda: 0.0, iters: 0 }]).unwrap();
        let mdl_truth = irys_recon::mdl::score(&ft);
        let part = Part {
            seed,
            truth: truth.to_text(),
            recovered: r.program.to_text(),
            truth_signature: canon::canonical(&truth).signature(),
            recovered_signature: r.program.signature(),
            topology_ok,
            max_error_mm,
            errors,
            pass,
            sigma: r.judge.sigma.clone(),
            weak: r.judge.weak.clone(),
            mdl: r.score,
            mdl_truth,
            rounds: r.accepted.len(),
            candidates: r.trace.len(),
            iters: r.iters,
            evals: r.evals,
            seconds: r.seconds,
        };
        println!(
            "seed {seed}: {} topo {} max err {:?} mdl {:.0} (truth {:.0}) {:.0}s",
            if pass { "PASS" } else { "FAIL" },
            topology_ok,
            max_error_mm.map(|e| (e * 1000.0).round() / 1000.0),
            r.score.total,
            mdl_truth.total,
            r.seconds
        );
        fs::write(out.join(format!("trace_{seed}.json")), serde_json::to_string_pretty(&r.trace).unwrap()).unwrap();
        write_images(&out, seed, &setup, &t, &r);
        parts.push(part);
        write_report(&out, &parts, views);
    }
}

fn write_images(out: &std::path::Path, seed: u64, setup: &Setup, t: &[Vec<f64>], r: &Recovery) {
    // Raw float images for the figure script: target, model, residual (view 0..n).
    let blob = serde_json::json!({
        "size": setup.film.size,
        "target": t,
        "model": r.fit.img,
        "residual": r.fit.residual,
    });
    fs::write(out.join(format!("images_{seed}.json")), blob.to_string()).unwrap();
}

fn write_report(out: &std::path::Path, parts: &[Part], views: usize) {
    fs::write(out.join("report.json"), serde_json::to_string_pretty(parts).unwrap()).unwrap();
    let n = parts.len();
    let pass = parts.iter().filter(|p| p.pass).count();
    let topo = parts.iter().filter(|p| p.topology_ok).count();
    let secs: f64 = parts.iter().map(|p| p.seconds).sum();
    let mut md = fs::File::create(out.join("report.md")).unwrap();
    writeln!(md, "# recon v0 gate ({views} view{})\n", if views == 1 { "" } else { "s" }).unwrap();
    writeln!(md, "**{pass}/{n} pass** (topology correct and every dimension within {TOL_MM} mm); topology correct {topo}/{n}; {:.0} s total, {:.0} s per part.\n", secs, secs / n.max(1) as f64).unwrap();
    writeln!(md, "| seed | truth | recovered | topo | max err mm | worst param | MDL (truth) | iters | renders | s | verdict |").unwrap();
    writeln!(md, "|---|---|---|---|---|---|---|---|---|---|---|").unwrap();
    for p in parts {
        let worst = p.errors.iter().max_by(|a, b| a.1.total_cmp(&b.1)).map_or("-".to_string(), |e| e.0.clone());
        writeln!(
            md,
            "| {} | `{}` | `{}` | {} | {} | {} | {:.0} ({:.0}) | {} | {} | {:.0} | {} |",
            p.seed,
            p.truth_signature,
            p.recovered_signature,
            if p.topology_ok { "yes" } else { "no" },
            p.max_error_mm.map_or("-".into(), |e| format!("{e:.3}")),
            worst,
            p.mdl.total,
            p.mdl_truth.total,
            p.iters,
            p.evals,
            p.seconds,
            if p.pass { "PASS" } else { "FAIL" }
        )
        .unwrap();
    }
    let _ = Program::from_text;
}
