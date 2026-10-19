//! The v0 figures from a `recon_v0` run directory.
//!
//! `cargo run --release -p irys-recon --example recon_figs -- DIR SEED`
//! writes `docs/recon-v0.png` (target / recovered / residual for eight
//! parts) and `docs/recon-v0-fit.png` (part `SEED`'s search trace: MDL of
//! every candidate, the accepted programs, per-parameter ±σ).

#[path = "common/font.rs"]
mod font;

use std::path::Path;

use serde_json::Value;

const BG: [u8; 3] = [24, 24, 28];
const FG: [u8; 3] = [230, 230, 230];
const DIM: [u8; 3] = [140, 140, 150];
const GOOD: [u8; 3] = [90, 200, 120];
const BAD: [u8; 3] = [235, 95, 85];
const ACCENT: [u8; 3] = [110, 160, 245];

struct Canvas {
    w: usize,
    h: usize,
    px: Vec<u8>,
}

impl Canvas {
    fn new(w: usize, h: usize) -> Self {
        let mut px = vec![0; w * h * 3];
        for p in px.chunks_exact_mut(3) {
            p.copy_from_slice(&BG);
        }
        Self { w, h, px }
    }
    fn put(&mut self, x: i64, y: i64, c: [u8; 3]) {
        if x >= 0 && y >= 0 && (x as usize) < self.w && (y as usize) < self.h {
            let i = (y as usize * self.w + x as usize) * 3;
            self.px[i..i + 3].copy_from_slice(&c);
        }
    }
    fn rect(&mut self, x0: i64, y0: i64, x1: i64, y1: i64, c: [u8; 3]) {
        for y in y0.min(y1)..=y0.max(y1) {
            for x in x0.min(x1)..=x0.max(x1) {
                self.put(x, y, c);
            }
        }
    }
    fn line(&mut self, a: (f64, f64), b: (f64, f64), c: [u8; 3]) {
        let n = ((b.0 - a.0).abs().max((b.1 - a.1).abs()) as usize).max(1);
        for i in 0..=n {
            let t = i as f64 / n as f64;
            self.put((a.0 + t * (b.0 - a.0)).round() as i64, (a.1 + t * (b.1 - a.1)).round() as i64, c);
        }
    }
    fn dot(&mut self, x: f64, y: f64, r: i64, c: [u8; 3]) {
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r {
                    self.put(x.round() as i64 + dx, y.round() as i64 + dy, c);
                }
            }
        }
    }
    fn text(&mut self, x: usize, y: usize, s: &str, scale: usize, c: [u8; 3]) {
        font::text(&mut self.px, self.w, x, y, s, scale, c);
    }
    fn save(&self, p: &Path) {
        image::RgbImage::from_raw(self.w as u32, self.h as u32, self.px.clone()).unwrap().save(p).unwrap();
    }
    fn blit_grey(&mut self, x0: usize, y0: usize, img: &[f64], (w, h): (usize, usize)) {
        for y in 0..h {
            for x in 0..w {
                let v = srgb8(img[y * w + x]);
                self.put((x0 + x) as i64, (y0 + y) as i64, [v, v, v]);
            }
        }
    }
    /// Signed residual in σ: blue negative, red positive, clamped at ±6σ.
    fn blit_signed(&mut self, x0: usize, y0: usize, img: &[f64], (w, h): (usize, usize)) {
        for y in 0..h {
            for x in 0..w {
                let t = (img[y * w + x] / 6.0).clamp(-1.0, 1.0);
                let c = if t >= 0.0 {
                    [(30.0 + 225.0 * t) as u8, (30.0 + 40.0 * t) as u8, 30]
                } else {
                    [30, (30.0 - 70.0 * t) as u8, (30.0 - 225.0 * t) as u8]
                };
                self.put((x0 + x) as i64, (y0 + y) as i64, c);
            }
        }
    }
}

fn srgb8(x: f64) -> u8 {
    let x = x.clamp(0.0, 1.0);
    let s = if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0 + 0.5) as u8
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn load(p: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap()
}

fn grid(dir: &Path, report: &Value, out: &Path) {
    let parts = report.as_array().unwrap();
    // Up to eight: the failures first (they are the interesting ones), then passes.
    let mut pick: Vec<&Value> = parts.iter().filter(|p| !p["pass"].as_bool().unwrap()).take(4).collect();
    pick.extend(parts.iter().filter(|p| p["pass"].as_bool().unwrap()).take(8 - pick.len()));
    pick.truncate(8);
    let first = load(&dir.join(format!("images_{}.json", pick[0]["seed"])));
    let (w, h) = (first["size"][0].as_u64().unwrap() as usize, first["size"][1].as_u64().unwrap() as usize);
    let (cols, rows) = (2usize, pick.len().div_ceil(2));
    let lab = 34;
    let gap = 12;
    let cell_w = 3 * w + 2 * 4;
    let head = 44;
    let mut c = Canvas::new(cols * cell_w + (cols + 1) * gap, head + rows * (h + lab + gap) + gap);
    let n = parts.len();
    let pass = parts.iter().filter(|p| p["pass"].as_bool().unwrap()).count();
    c.text(gap, 10, &format!("RECON V0: TARGET  |  RECOVERED  |  RESIDUAL (±6 SIGMA)    GATE {pass}/{n} PASS"), 2, FG);
    for (k, p) in pick.iter().enumerate() {
        let (col, row) = (k % cols, k / cols);
        let (x0, y0) = (gap + col * (cell_w + gap), head + row * (h + lab + gap));
        let im = load(&dir.join(format!("images_{}.json", p["seed"])));
        c.blit_grey(x0, y0 + lab, &floats(&im["target"][0]), (w, h));
        c.blit_grey(x0 + w + 4, y0 + lab, &floats(&im["model"][0]), (w, h));
        c.blit_signed(x0 + 2 * (w + 4), y0 + lab, &floats(&im["residual"][0]), (w, h));
        let ok = p["pass"].as_bool().unwrap();
        let topo = p["topology_ok"].as_bool().unwrap();
        let err = p["max_error_mm"].as_f64();
        let verdict = if ok { "PASS" } else { "FAIL" };
        let detail = match (topo, err) {
            (true, Some(e)) => format!("MAX ERR {e:.2} MM"),
            _ => "WRONG TOPOLOGY".to_string(),
        };
        c.text(x0, y0 + 2, &format!("SEED {}  {verdict}  {detail}", p["seed"]), 2, if ok { GOOD } else { BAD });
        let sig = format!("{}  →  {}", p["truth_signature"].as_str().unwrap(), p["recovered_signature"].as_str().unwrap());
        c.text(x0, y0 + 20, &sig.replace("linear_holes", "LIN").replace("circular_holes", "CIRC").replace("counterbore", "CBORE"), 1, DIM);
    }
    c.save(out);
}

fn trace_fig(dir: &Path, report: &Value, seed: u64, out: &Path) {
    let part = report.as_array().unwrap().iter().find(|p| p["seed"].as_u64() == Some(seed)).expect("seed in report");
    let trace = load(&dir.join(format!("trace_{seed}.json")));
    let steps = trace.as_array().unwrap();
    let (cw, ch) = (1500usize, 900usize);
    let mut c = Canvas::new(cw, ch);
    c.text(20, 14, &format!("SEED {seed}: SEARCH TRACE  ({} CANDIDATES, {} ACCEPTED)", steps.len(), steps.iter().filter(|s| s["accepted"].as_bool().unwrap()).count()), 2, FG);
    // Plot: MDL (nats, log of the excess over the final) per candidate, by round.
    let (px0, py0, pw, ph) = (90.0, 70.0, 860.0, 420.0);
    let tot: Vec<Option<f64>> = steps.iter().map(|s| s["score"]["total"].as_f64()).collect();
    let fin = tot.iter().flatten().cloned().fold(f64::INFINITY, f64::min);
    let y_of = |v: f64| {
        let e = (v - fin + 1.0).max(1.0).log10();
        py0 + ph - e / 5.0 * ph
    };
    c.rect(px0 as i64, py0 as i64, px0 as i64, (py0 + ph) as i64, DIM);
    c.rect(px0 as i64, (py0 + ph) as i64, (px0 + pw) as i64, (py0 + ph) as i64, DIM);
    for d in 0..=5 {
        let y = py0 + ph - d as f64 / 5.0 * ph;
        c.line((px0 - 5.0, y), (px0, y), DIM);
        c.text(10, y as usize - 3, &format!("1E{d}"), 1, DIM);
    }
    c.text(10, (py0 - 20.0) as usize, "MDL − FINAL + 1 (NATS)", 1, DIM);
    c.text(px0 as usize, (py0 + ph + 10.0) as usize, "CANDIDATE (IN ORDER TRIED; ROUNDS SEPARATED)", 1, DIM);
    let n = steps.len().max(2);
    let x_of = |i: usize| px0 + 10.0 + (pw - 20.0) * i as f64 / (n - 1) as f64;
    let mut last_round = 0;
    let mut prev_acc: Option<(f64, f64)> = None;
    for (i, s) in steps.iter().enumerate() {
        let r = s["round"].as_u64().unwrap();
        if r != last_round {
            c.line((x_of(i) - 3.0, py0), (x_of(i) - 3.0, py0 + ph), [50, 50, 60]);
            last_round = r;
        }
        if let Some(v) = tot[i] {
            let acc = s["accepted"].as_bool().unwrap();
            let (x, y) = (x_of(i), y_of(v));
            if acc {
                if let Some(p) = prev_acc {
                    c.line(p, (x, y), ACCENT);
                }
                prev_acc = Some((x, y));
                c.dot(x, y, 4, ACCENT);
            } else {
                c.dot(x, y, 2, DIM);
            }
        }
    }
    // Accepted steps, as text.
    let mut ty = 70;
    c.text(980, 50, "ACCEPTED EDITS (MDL NATS)", 2, FG);
    for s in steps.iter().filter(|s| s["accepted"].as_bool().unwrap()) {
        let line = format!("R{} {} : {:.0}", s["round"], s["why"].as_str().unwrap(), s["score"]["total"].as_f64().unwrap());
        c.text(980, ty, &line.chars().take(84).collect::<String>(), 1, ACCENT);
        ty += 14;
    }
    ty += 10;
    c.text(980, ty, "RECOVERED PROGRAM", 2, FG);
    ty += 22;
    for l in part["recovered"].as_str().unwrap().lines() {
        c.text(980, ty, l, 1, FG);
        ty += 12;
    }
    ty += 8;
    c.text(980, ty, "TRUTH", 2, FG);
    ty += 22;
    for l in part["truth"].as_str().unwrap().lines() {
        c.text(980, ty, l, 1, DIM);
        ty += 12;
    }
    // Per-parameter ±σ (log scale), with the actual error beside it.
    let sig = part["sigma"].as_array().unwrap();
    let errs: std::collections::HashMap<String, f64> = part["errors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| (e[0].as_str().unwrap().to_string(), e[1].as_f64().unwrap()))
        .collect();
    let (bx0, by0, bw) = (180.0, 560.0, 700.0);
    c.text(20, 530, "JUDGE: ±SIGMA FROM (J^T J)^-1 (BAR, LOG MM)  AND |ERROR| VS TRUTH (DOT)", 2, FG);
    let xs = |v: f64| bx0 + ((v.max(1e-4)).log10() + 4.0) / 5.0 * bw; // 1e-4 .. 10 mm
    for d in -4..=1 {
        let x = xs(10f64.powi(d));
        c.line((x, by0 - 6.0), (x, by0 + 16.0 * sig.len() as f64), [50, 50, 60]);
        c.text(x as usize - 10, (by0 + 16.0 * sig.len() as f64 + 4.0) as usize, &format!("1E{d}"), 1, DIM);
    }
    let gate = xs(0.5);
    c.line((gate, by0 - 6.0), (gate, by0 + 16.0 * sig.len() as f64), BAD);
    c.text(gate as usize + 3, (by0 - 16.0) as usize, "0.5 MM GATE", 1, BAD);
    for (k, s) in sig.iter().enumerate() {
        let name = s[0].as_str().unwrap();
        let v = s[1].as_f64().unwrap();
        let y = by0 + 16.0 * k as f64;
        c.text(20, y as usize, name, 1, FG);
        c.rect(bx0 as i64, y as i64, xs(v) as i64, y as i64 + 8, ACCENT);
        if let Some(e) = errs.get(name) {
            c.dot(xs(*e), y + 4.0, 3, if *e <= 0.5 { GOOD } else { BAD });
        }
    }
    c.save(out);
}

fn main() {
    let mut a = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(a.next().expect("run directory"));
    let seed: u64 = a.next().expect("seed for the trace figure").parse().unwrap();
    let report = load(&dir.join("report.json"));
    grid(&dir, &report, Path::new("docs/recon-v0.png"));
    trace_fig(&dir, &report, seed, Path::new("docs/recon-v0-fit.png"));
}
