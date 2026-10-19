//! v1 on real photos: one or more JPEGs of one part beside the kit's tag.
//!
//! `cargo run --release -p irys-recon --example recon_photo -- photo.jpg [more.jpg ...]
//!     [--out DIR] [--tag-mm 50] [--sensor-mm W] [--film 256x192] [--truth truth.txt]`
//!
//! Writes, in `DIR` (default `out/recon_photo/<first photo's stem>`):
//! `program.txt` (the recovered program, part frame, with its pose on the
//! tag plane), `part.vcad` (vcad's document: the same program as an IR
//! DAG with the fitted PLA material), `part.stl`, `overlay_<i>.png` (the
//! crop with the model's outline, and the residual), `report.json` and
//! `report.md` (every dimension ±σ, the camera, the appearance, and the
//! errors against `--truth` when given).

use std::fmt::Write as _;
use std::path::PathBuf;

use irys_recon::dsl::Program;
use irys_recon::{canon, export, v1};

fn main() {
    let mut photos = Vec::new();
    let mut o = v1::V1Options::default();
    let mut out: Option<PathBuf> = None;
    let mut truth: Option<Program> = None;
    let mut a = std::env::args().skip(1);
    while let Some(x) = a.next() {
        match x.as_str() {
            "--out" => out = a.next().map(PathBuf::from),
            "--tag-mm" => o.tag_mm = a.next().unwrap().parse().unwrap(),
            "--sensor-mm" => o.sensor_w_mm = a.next().map(|v| v.parse().unwrap()),
            "--film" => {
                let v = a.next().unwrap();
                let (w, h) = v.split_once('x').expect("--film WxH");
                o.film = (w.parse().unwrap(), h.parse().unwrap());
            }
            "--truth" => truth = Some(Program::from_text(&std::fs::read_to_string(a.next().unwrap()).unwrap()).unwrap()),
            _ => photos.push(PathBuf::from(x)),
        }
    }
    assert!(!photos.is_empty(), "usage: recon_photo photo.jpg [more.jpg] [--out DIR] [--tag-mm 50] [--truth truth.txt]");
    let out = out.unwrap_or_else(|| PathBuf::from("out/recon_photo").join(photos[0].file_stem().unwrap()));
    std::fs::create_dir_all(&out).unwrap();
    let loaded: Vec<v1::Photo> = photos.iter().map(|p| v1::load(p).unwrap_or_else(|e| panic!("{e}"))).collect();
    let t0 = std::time::Instant::now();
    let v = v1::recover_photos(&loaded, &o).unwrap_or_else(|e| panic!("recovery failed: {e}"));
    let secs = t0.elapsed().as_secs_f64();
    let prog = &v.rec.program;
    let look = v.looks[0];
    std::fs::write(out.join("program.txt"), prog.to_text() + "\n").unwrap();
    let mut part = prog.clone();
    part.pose = None;
    std::fs::write(out.join("part.stl"), export::stl(&part).unwrap()).unwrap();
    std::fs::write(out.join("part.vcad"), export::vcad_doc(&part, look.rgb, look.rough).to_json().unwrap()).unwrap();
    for i in 0..loaded.len() {
        v1::overlay(&v, i).save(out.join(format!("overlay_{i}.png"))).unwrap();
    }
    // The report.
    let info = prog.param_info();
    let sigma: std::collections::HashMap<String, f64> = v.rec.judge.sigma.iter().cloned().collect();
    let errs: Option<Vec<(String, f64)>> = truth.as_ref().and_then(|t| {
        // The plate's quarter-turn symmetry: compare in the truth's frame
        // (the turn that lines the recovered part up best).
        (0..4).map(|k| part.quarter_turned(k)).filter(|q| canon::same_topology(t, q)).filter_map(|q| canon::errors(t, &q)).min_by(|a, b| {
            let m = |e: &Vec<(String, f64)>| e.iter().fold(0.0f64, |m, x| m.max(x.1));
            m(a).total_cmp(&m(b))
        })
    });
    let mut md = format!("# recon_photo: {}\n\n", photos.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", "));
    let _ = writeln!(md, "```\n{}\n```\n", prog.to_text());
    let _ = writeln!(md, "MDL {:.0} nats (residual {:.0}, parameters {:.1}, program {:.1}); {:.0} s, {} candidates.\n", v.rec.score.total, v.rec.score.residual, v.rec.score.params, v.rec.score.program, secs, v.rec.trace.len());
    md += "| parameter | value | ±σ |";
    if errs.is_some() {
        md += " error vs truth |";
    }
    md += "\n|---|---|---|";
    if errs.is_some() {
        md += "---|";
    }
    md += "\n";
    for (i, x) in info.iter().zip(prog.params()) {
        let s = sigma.get(&i.name).copied().unwrap_or(f64::NAN);
        let _ = write!(md, "| {} | {:.3} {} | {:.3} |", i.name, x, if i.unit == irys_recon::dsl::Unit::Deg { "°" } else { "mm" }, s);
        if errs.is_some() {
            md += " |";
        }
        md += "\n";
    }
    if let Some(e) = &errs {
        let worst = e.iter().fold(0.0f64, |m, x| m.max(x.1));
        let _ = writeln!(md, "\nAgainst the truth: topology matches; worst dimension error {worst:.3} mm ({}).\n", if worst <= 1.0 { "within the 1 mm gate" } else { "outside the 1 mm gate" });
        for (n, x) in e {
            let _ = writeln!(md, "- {n}: {x:.3} mm");
        }
    } else if truth.is_some() {
        md += "\nAgainst the truth: **topology differs** (no quarter turn matches).\n";
    }
    if !v.rec.judge.weak.is_empty() {
        md += "\nDirections the photo does not determine (σ > 0.5 mm):\n";
        for (s, dir) in &v.rec.judge.weak {
            let _ = writeln!(md, "- σ {s:.2} mm along {}", dir.iter().filter(|d| d.1.abs() > 0.2).map(|(n, w)| format!("{w:+.2} {n}")).collect::<Vec<_>>().join(", "));
        }
    }
    md += "\n## Camera and appearance\n\n";
    for (i, c) in v.calib.iter().enumerate() {
        let l = v.looks[i];
        let _ = writeln!(
            md,
            "- photo {i}: tag id {} (hamming {}), corner rms {:.2} px; f {:.1} px (EXIF {}); crop {:?}; gamma {:.2}, key from ({:.2}, {:.2}, {:.2}) radius {:.0}°; PLA base_color {:.2}, {:.2}, {:.2} (relative to the backdrop), roughness {:.2}",
            c.tag.id, c.tag.hamming, c.rms, c.k.f, c.f_exif.map_or("none".into(), |f| format!("{f:.1} px")), { let c = v.framed[i].crop; (c.0.round(), c.1.round(), c.2.round(), c.3.round()) }, l.gamma, l.key_dir[0], l.key_dir[1], l.key_dir[2], l.key_radius.to_degrees(), l.rgb[0], l.rgb[1], l.rgb[2], l.rough
        );
    }
    let _ = writeln!(md, "\nσ are the Fisher information's (Cramér–Rao, robust noise scale {:.4} at full resolution); v0 found them optimistic by a median 2.3×.", v.sigma.1);
    std::fs::write(out.join("report.md"), &md).unwrap();
    let json = serde_json::json!({
        "photos": photos,
        "program": prog.to_text(),
        "params": info.iter().zip(prog.params()).map(|(i, x)| serde_json::json!({"name": i.name, "value": x, "sigma": sigma.get(&i.name)})).collect::<Vec<_>>(),
        "mdl": v.rec.score,
        "weak": v.rec.judge.weak,
        "calib": v.calib,
        "looks": v.looks,
        "errors_vs_truth": errs,
        "seconds": secs,
    });
    std::fs::write(out.join("report.json"), serde_json::to_string_pretty(&json).unwrap()).unwrap();
    print!("{md}");
    println!("\nwritten to {}", out.display());
}
