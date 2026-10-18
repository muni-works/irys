//! irys-reel: the hype reel's soundtrack, made by the stack.
//!
//! ```text
//! irys-reel stub     [out/stub-manifest.json]
//! irys-reel render   --manifest M --arrangement A --docs D --out DIR [--preview]
//! ```
//!
//! Harmony is the frames' spectra folded onto pitch classes (`spectral`),
//! percussion is struck kosm materials (`modal`), and the risers are the fit
//! loss curves in `docs/*-loss.csv` (`riser`).

mod dsp;
mod modal;
mod riser;
mod spectral;
mod spectrogram;
mod wav;

use dsp::{OnePole, Rng, Svf, midi_hz, pan};
use serde::{Deserialize, Serialize};
use spectral::Manifest;
use std::f64::consts::PI;
use std::path::{Path, PathBuf};

#[derive(Deserialize, Clone)]
struct RiserSpec {
    csv: String,
    column: String,
    #[serde(default = "one")]
    level: f64,
    /// The riser starts this many bars into its section (default 0).
    #[serde(default)]
    from_bar: usize,
}

fn one() -> f64 {
    1.0
}

#[derive(Deserialize, Clone)]
struct Section {
    name: String,
    bars: usize,
    sequences: Vec<String>,
    drums: String,
    #[serde(default)]
    drop: bool,
    #[serde(default)]
    riser: Option<RiserSpec>,
}

#[derive(Deserialize)]
struct Arrangement {
    seed: u64,
    bpm: f64,
    beats_per_bar: usize,
    sample_rate: f64,
    chord_every_beats: usize,
    chroma_smoothing: f64,
    sections: Vec<Section>,
}

#[derive(Serialize)]
struct SectionOut {
    name: String,
    start: f64,
    end: f64,
    bars: usize,
    sequences: Vec<SeqSpan>,
}

#[derive(Serialize)]
struct SeqSpan {
    id: String,
    start: f64,
    end: f64,
}

#[derive(Serialize)]
struct ChordOut {
    t: f64,
    chord: String,
    sequence: String,
    frame: usize,
    chroma: Vec<f64>,
    notes: Vec<String>,
}

#[derive(Serialize)]
struct Beats {
    bpm: f64,
    key: String,
    beat_s: f64,
    duration: f64,
    stub_manifest: bool,
    beats: Vec<f64>,
    downbeats: Vec<f64>,
    drops: Vec<f64>,
    sections: Vec<SectionOut>,
    risers: Vec<riser::RiserOut>,
    chords: Vec<ChordOut>,
}

const NAMES: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

fn note_name(m: f64) -> String {
    let m = m.round() as i64;
    format!("{}{}", NAMES[m.rem_euclid(12) as usize], m.div_euclid(12) - 1)
}

/// Major-scale degrees (semitones above the tonic).
const MAJOR: [usize; 7] = [0, 2, 4, 5, 7, 9, 11];
/// Krumhansl–Kessler major profile, for choosing the key from the mean chroma.
const KK_MAJOR: [f64; 12] = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];

/// The tonal centre: the major key whose profile best correlates with the
/// piece's mean chroma.
fn choose_key(mean: &[f64; 12]) -> usize {
    let m = mean.iter().sum::<f64>() / 12.0;
    let pm = KK_MAJOR.iter().sum::<f64>() / 12.0;
    (0..12)
        .map(|k| (k, (0..12).map(|i| (mean[(i + k) % 12] - m) * (KK_MAJOR[i] - pm)).sum::<f64>()))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap().then(b.0.cmp(&a.0)))
        .unwrap()
        .0
}

/// A diatonic chord of the key: its roman numeral, triad, and the two colour
/// tones the spectrum may choose between (all inside the mode).
struct Diatonic {
    name: &'static str,
    /// Degree index of the root (0 = I).
    degree: usize,
    /// Prior: ii is used sparingly.
    prior: f64,
    /// Colour tones as scale-degree offsets from the root: (label, offset).
    colours: [(&'static str, usize); 2],
}

const CHORDS: [Diatonic; 5] = [
    Diatonic { name: "I", degree: 0, prior: 1.0, colours: [("add9", 1), ("maj7", 6)] },
    Diatonic { name: "IV", degree: 3, prior: 1.0, colours: [("add9", 1), ("maj7", 6)] },
    Diatonic { name: "V", degree: 4, prior: 0.97, colours: [("sus2", 1), ("add9", 1)] },
    Diatonic { name: "vi", degree: 5, prior: 0.9, colours: [("add9", 1), ("7", 6)] },
    Diatonic { name: "ii", degree: 1, prior: 0.72, colours: [("add9", 1), ("7", 6)] },
];

/// Pitch class of the scale degree `root_deg + off` in `key`.
fn deg_pc(key: usize, root_deg: usize, off: usize) -> usize {
    (key + MAJOR[(root_deg + off) % 7]) % 12
}

/// The diatonic chord a chroma votes for: mean chroma over the triad, times the
/// chord's prior.
fn score(c: &[f64; 12], key: usize, d: &Diatonic) -> f64 {
    let t: f64 = [0, 2, 4].iter().map(|&o| c[deg_pc(key, d.degree, o)]).sum();
    d.prior * (t / 3.0 + 0.15 * c[deg_pc(key, d.degree, 0)])
}

/// Voice one diatonic chord: triad plus the colour tone the spectrum prefers
/// (sus2 replaces the third), bass on whichever triad tone is brightest (root
/// favoured), so the spectrum picks the inversion.
fn voicing(c: &[f64; 12], key: usize, d: &Diatonic) -> (Vec<usize>, usize, String) {
    let (lab, off) = if c[deg_pc(key, d.degree, d.colours[0].1)] >= c[deg_pc(key, d.degree, d.colours[1].1)] { d.colours[0] } else { d.colours[1] };
    let col = deg_pc(key, d.degree, off);
    let root = deg_pc(key, d.degree, 0);
    let third = deg_pc(key, d.degree, 2);
    let fifth = deg_pc(key, d.degree, 4);
    let tones = if lab == "sus2" { vec![root, col, fifth, root] } else { vec![root, third, fifth, col] };
    let sus = lab == "sus2";
    let bass = [(root, c[root] + 0.25), (third, if sus { f64::MIN } else { c[third] }), (fifth, c[fifth])]
        .iter()
        .fold((root, f64::MIN), |b, &(p, w)| if w > b.1 { (p, w) } else { b })
        .0;
    let inv = if bass == root { "" } else if bass == third { "/3" } else { "/5" };
    (tones, bass, format!("{}{}{}", d.name, lab, inv))
}

/// Voice-lead `pcs` onto the previous chord. Every assignment of pitch classes
/// to voices and every octave in range is tried; the cost is total motion, plus
/// a penalty for unisons and for the chord's centre drifting from D4/E4, so it
/// glides by the smallest steps without collapsing or creeping up the keyboard.
fn voice_lead(prev: &[f64], pcs: &[usize]) -> Vec<f64> {
    let n = prev.len();
    let mut targets: Vec<usize> = pcs.to_vec();
    while targets.len() < n {
        targets.push(pcs[targets.len() % pcs.len()]);
    }
    let octaves = |pc: usize| -> Vec<f64> { (3..7).map(|o| (pc + 12 * o) as f64).filter(|&m| (55.0..=81.0).contains(&m)).collect() };
    let mut best = (f64::INFINITY, prev.to_vec());
    let mut idx: Vec<usize> = (0..n).collect();
    permute(&mut idx, 0, &mut |perm: &[usize]| {
        let opts: Vec<Vec<f64>> = perm.iter().map(|&j| octaves(targets[j])).collect();
        let mut pick = vec![0usize; n];
        loop {
            let v: Vec<f64> = (0..n).map(|i| opts[i][pick[i]]).collect();
            let motion: f64 = v.iter().zip(prev).map(|(a, b)| (a - b).abs()).sum();
            let mut unison = 0.0;
            for i in 0..n {
                for k in i + 1..n {
                    if (v[i] - v[k]).abs() < 0.5 {
                        unison += 1.0;
                    }
                }
            }
            let centre = (v.iter().sum::<f64>() / n as f64 - 67.0).abs();
            let cost = motion + 20.0 * unison + 1.5 * centre;
            if cost < best.0 {
                best = (cost, v);
            }
            let mut i = 0;
            while i < n {
                pick[i] += 1;
                if pick[i] < opts[i].len() {
                    break;
                }
                pick[i] = 0;
                i += 1;
            }
            if i == n {
                break;
            }
        }
    });
    let mut v = best.1;
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

fn permute(v: &mut Vec<usize>, k: usize, f: &mut dyn FnMut(&[usize])) {
    if k == v.len() {
        f(v);
        return;
    }
    for i in k..v.len() {
        v.swap(k, i);
        permute(v, k + 1, f);
        v.swap(k, i);
    }
}

/// One step of a 16th grid: (voice index into the kit, velocity, pan).
fn pattern(kind: &str, bar: usize, bars: usize, step: usize, rng: &mut Rng) -> Vec<(usize, f64, f64)> {
    const KICK: usize = 0;
    const STONE: usize = 1;
    const WOOD: usize = 2;
    const GLASS: usize = 3;
    const ALU: usize = 4;
    const BRASS: usize = 5;
    const GOLD: usize = 6;
    let mut h = Vec::new();
    let jitter = 0.9 + 0.1 * rng.uniform();
    let last = bar + 1 == bars;
    match kind {
        "sparse" => {
            if step == 0 && bar % 2 == 0 {
                h.push((STONE, 0.7, -0.2));
            }
            if step == 8 {
                h.push((GLASS, 0.35 * jitter, 0.4));
            }
            if step == 14 && bar % 2 == 1 {
                h.push((ALU, 0.5, -0.5));
            }
            if step == 0 && bar == 0 {
                h.push((GOLD, 0.5, 0.0));
            }
        }
        "build" => {
            let quarters = bar + 2 >= bars;
            if (step % 8 == 0) || (quarters && step % 4 == 0) {
                if !(last && step >= 12) {
                    h.push((KICK, 0.8, 0.0));
                }
            }
            if step % 4 == 2 {
                h.push((GLASS, 0.45 * jitter, 0.35));
            }
            let roll = if last { step % 1 == 0 && step < 14 } else if bar + 2 == bars { step % 2 == 0 } else { step == 4 || step == 12 };
            if roll {
                let ramp = (bar as f64 * 16.0 + step as f64) / (bars as f64 * 16.0);
                h.push((WOOD, (0.3 + 0.6 * ramp) * jitter, if step % 2 == 0 { -0.25 } else { 0.25 }));
            }
        }
        "full" | "full_plus" => {
            let plus = kind == "full_plus";
            if step % 4 == 0 {
                h.push((KICK, 1.0, 0.0));
            }
            if step == 4 || step == 12 {
                h.push((WOOD, 0.95 * jitter, 0.1));
            }
            if step % 4 == 2 {
                h.push((GLASS, 0.75 * jitter, 0.35));
            } else if step % 2 == 1 || plus {
                h.push((GLASS, 0.25 * jitter, -0.35));
            }
            if step == 14 || (plus && step == 10) {
                h.push((ALU, 0.6, -0.55));
            }
            if step == 0 && bar % 4 == 0 {
                h.push((BRASS, 0.9, 0.0));
            }
            if step == 0 && bar % 2 == 0 {
                h.push((STONE, 0.6, 0.0));
            }
            if plus && step == 0 && bar == 0 {
                h.push((GOLD, 0.8, 0.3));
            }
            if plus && step == 15 && bar % 2 == 1 {
                h.push((WOOD, 0.35, -0.3));
            }
            if last && step >= 12 && !plus {
                h.push((WOOD, 0.5 + 0.1 * (step - 12) as f64, 0.0));
            }
        }
        "outro" => {
            if step == 0 && bar == 0 {
                h.extend([(KICK, 1.0, 0.0), (STONE, 0.8, 0.0), (BRASS, 0.9, -0.2), (GOLD, 0.9, 0.2)]);
            }
            if bar == 0 && step == 8 {
                h.push((GLASS, 0.3, 0.4));
            }
        }
        "none" => {
            if step == 0 && bar % 2 == 0 {
                h.push((GOLD, 0.55, if bar == 0 { -0.3 } else { 0.3 }));
            }
            if step == 8 {
                h.push((GLASS, 0.18, 0.5));
            }
        }
        _ => {}
    }
    h
}

/// Per-section mix: pad level, pad brightness (Hz), bass, arp.
fn mix(s: &Section) -> (f64, f64, f64, f64) {
    match s.drums.as_str() {
        "sparse" => (0.8, 2000.0, 0.0, 0.0),
        "build" => (0.75, 2600.0, 0.25, 0.0),
        "full" => (0.6, 5000.0, 1.0, 0.45),
        "full_plus" => (0.6, 6000.0, 1.0, 0.7),
        "none" => (0.95, 1500.0, 0.0, 0.15),
        "outro" => (0.8, 3000.0, 0.4, 0.0),
        _ => (0.7, 2000.0, 0.0, 0.0),
    }
}

fn arg(args: &[String], k: &str, d: &str) -> String {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| d.to_string())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("stub") => {
            let out = args.get(2).cloned().unwrap_or_else(|| "stub-manifest.json".into());
            std::fs::write(&out, serde_json::to_string(&spectral::stub())?)?;
            println!("{out}");
            Ok(())
        }
        Some("render") => render(&args),
        _ => {
            eprintln!("usage: irys-reel stub [path] | render --manifest M --arrangement A --docs D --out DIR");
            std::process::exit(2)
        }
    }
}

fn render(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let man_path = arg(args, "--manifest", "stub-manifest.json");
    let arr: Arrangement = serde_json::from_str(&std::fs::read_to_string(arg(args, "--arrangement", "arrangement.json"))?)?;
    let docs = PathBuf::from(arg(args, "--docs", "../../docs"));
    let out_dir = PathBuf::from(arg(args, "--out", "out"));
    std::fs::create_dir_all(&out_dir)?;
    let man: Manifest = serde_json::from_str(&std::fs::read_to_string(&man_path)?)?;
    if man.stub {
        eprintln!("note: {man_path} is a STUB manifest");
    }

    let sr = arr.sample_rate;
    let beat = 60.0 / arr.bpm;
    let bar = beat * arr.beats_per_bar as f64;
    let total_bars: usize = arr.sections.iter().map(|s| s.bars).sum();
    let tail = 2.5; // the last bell's ring
    let dur = total_bars as f64 * bar + tail;
    let n = (dur * sr) as usize;
    let mut rng = Rng::new(arr.seed);

    // ── timeline ────────────────────────────────────────────────────────────
    let mut starts = Vec::new();
    let mut t = 0.0;
    for s in &arr.sections {
        starts.push(t);
        t += s.bars as f64 * bar;
    }
    let music_end = t;
    let section_at = |t: f64| -> usize { starts.iter().rposition(|&s| t >= s - 1e-9).unwrap_or(0) };
    // Which sequence and frame plays at time t.
    let seq_at = |t: f64| -> (String, usize) {
        let si = section_at(t.min(music_end - 1e-6));
        let s = &arr.sections[si];
        let len = s.bars as f64 * bar / s.sequences.len() as f64;
        let local = t - starts[si];
        let k = ((local / len) as usize).min(s.sequences.len() - 1);
        let id = &s.sequences[k];
        let seq = man.get(id).unwrap_or_else(|| panic!("sequence {id} not in manifest"));
        let frame = (((local - k as f64 * len) * seq.fps) as usize).min(seq.frames.max(1) * 1000) % seq.frames.max(1);
        (id.clone(), frame)
    };

    // ── harmony: one chord per step, from the mean spectrum under it ─────────
    let step_s = beat * arr.chord_every_beats as f64;
    let steps = (music_end / step_s).round() as usize;
    let mut smooth = [0.0f64; 12];
    let mut voices = vec![60.0, 64.0, 67.0, 71.0];
    let mut chords: Vec<(f64, Vec<f64>, f64)> = Vec::new(); // (t, voices, bass)
    let mut chord_log = Vec::new();
    let mut per_step: Vec<([f64; 12], String, usize)> = Vec::new();
    for k in 0..steps {
        let t0 = k as f64 * step_s;
        let mut acc = [0.0; 12];
        let samples = 6;
        let (mut sid, mut sframe) = (String::new(), 0);
        for j in 0..samples {
            let (id, f) = seq_at(t0 + step_s * (j as f64 + 0.5) / samples as f64);
            let seq = man.get(&id).unwrap();
            let c = spectral::chroma(man.bins(seq), &seq.spectra[f.min(seq.spectra.len() - 1)]);
            for i in 0..12 {
                acc[i] += c[i] / samples as f64;
            }
            if j == 0 {
                sid = id;
                sframe = f;
            }
        }
        let a = if k == 0 { 0.0 } else { arr.chroma_smoothing };
        for i in 0..12 {
            smooth[i] = a * smooth[i] + (1.0 - a) * acc[i];
        }
        per_step.push((smooth, sid, sframe));
    }
    let mut mean = [0.0; 12];
    for (c, _, _) in &per_step {
        for i in 0..12 {
            mean[i] += c[i] / per_step.len() as f64;
        }
    }
    let key = choose_key(&mean);
    eprintln!("key: {} major", NAMES[key]);
    // Steps that resolve: the downbeat of every drop and of the outro.
    let resolves: Vec<usize> = arr
        .sections
        .iter()
        .zip(&starts)
        .filter(|(s, _)| s.drop || s.drums == "outro")
        .map(|(_, &t)| (t / step_s).round() as usize)
        .collect();
    // Score each chord against its own piece-wide baseline: broad spectra all
    // lean on the tonic, so what the frame says is how it departs from the mean.
    let base: Vec<f64> = CHORDS.iter().map(|d| score(&mean, key, d)).collect();
    let rel = |c: &[f64; 12], i: usize| score(c, key, &CHORDS[i]) - base[i] + 0.08 * (CHORDS[i].prior - 1.0);
    let mut prev: Option<usize> = None;
    let mut ii_run = 0;
    let mut ii_cool = 0usize;
    for (k, (smooth, sid, sframe)) in per_step.into_iter().enumerate() {
        let t0 = k as f64 * step_s;
        // ii sparingly: two beats at most, then six without it.
        let pool: Vec<usize> = if resolves.contains(&k) {
            vec![0, 1]
        } else {
            (0..CHORDS.len()).filter(|&i| !(CHORDS[i].name == "ii" && (ii_run >= 2 || ii_cool > 0))).collect()
        };
        let mut pick = *pool
            .iter()
            .max_by(|&&a, &&b| rel(&smooth, a).partial_cmp(&rel(&smooth, b)).unwrap().then(b.cmp(&a)))
            .unwrap();
        // Hysteresis: hold the last chord unless the new one wins clearly.
        if let Some(p) = prev {
            if pool.contains(&p) && rel(&smooth, pick) < rel(&smooth, p) + 0.006 {
                pick = p;
            }
        }
        prev = Some(pick);
        if CHORDS[pick].name == "ii" {
            ii_run += 1;
        } else {
            if ii_run > 0 {
                ii_cool = 6;
            }
            ii_run = 0;
            ii_cool = ii_cool.saturating_sub(1);
        }
        let (pcs, bass_pc, label) = voicing(&smooth, key, &CHORDS[pick]);
        voices = voice_lead(&voices, &pcs);
        let root = bass_pc as f64;
        let bass = root + 12.0 * ((36.0 - root) / 12.0).ceil();
        let _ = &label;
        chord_log.push(ChordOut {
            t: t0,
            sequence: sid,
            frame: sframe,
            chroma: smooth.iter().map(|v| (v * 1000.0).round() / 1000.0).collect(),
            chord: label,
            notes: voices.iter().map(|&m| note_name(m)).chain([note_name(bass)]).collect(),
        });
        chords.push((t0, voices.clone(), bass));
    }

    // ── buses ────────────────────────────────────────────────────────────────
    let mut music = vec![[0.0f64; 2]; n]; // ducked by the kick
    let mut drums = vec![[0.0f64; 2]; n];
    let mut send = vec![0.0f64; n]; // mono reverb send
    let mut duck = vec![1.0f64; n];

    // Pad: four voices × two detuned additive saws, gliding between chords.
    {
        let mut freq: Vec<f64> = chords[0].1.iter().map(|&m| midi_hz(m)).collect();
        let mut ph = vec![[0.0f64; 2]; freq.len()];
        let mut lp = [OnePole::default(), OnePole::default()];
        let mut lp2 = [OnePole::default(), OnePole::default()];
        let mut hp = [OnePole::default(), OnePole::default()];
        let mut level = 0.0;
        let mut cutoff = 1000.0;
        let glide = 1.0 - (-1.0 / (0.09 * sr)).exp();
        let smooth_k = 1.0 - (-1.0 / (0.4 * sr)).exp();
        for i in 0..n {
            let t = i as f64 / sr;
            let ci = ((t / step_s) as usize).min(chords.len() - 1);
            let si = section_at(t.min(music_end - 1e-6));
            let (lvl, cut, _, _) = mix(&arr.sections[si]);
            let mut target_lvl = lvl;
            if t < 2.0 * bar {
                target_lvl *= t / (2.0 * bar);
            }
            if t > music_end {
                target_lvl = 0.0;
            }
            level += (target_lvl - level) * smooth_k;
            cutoff += (cut - cutoff) * smooth_k;
            // Breathing: a slow, bar-long swell.
            let breathe = 0.85 + 0.15 * (2.0 * PI * t / (2.0 * bar)).sin();
            let mut s = [0.0; 2];
            for (v, &m) in chords[ci].1.iter().enumerate() {
                let f = midi_hz(m);
                freq[v] += (f - freq[v]) * glide;
                for side in 0..2 {
                    let det = if side == 0 { 0.997 } else { 1.003 };
                    ph[v][side] = (ph[v][side] + freq[v] * det / sr) % 1.0;
                    let mut y = 0.0;
                    for h in 1..=14 {
                        if freq[v] * h as f64 > 12000.0 {
                            break;
                        }
                        y += (2.0 * PI * ph[v][side] * h as f64).sin() / (h as f64).powf(0.85);
                    }
                    s[side] += y * 0.08;
                }
            }
            for side in 0..2 {
                let y = lp2[side].tick(lp[side].tick(s[side], cutoff, sr), cutoff * 1.5, sr);
                // Thin the low mids: the bass owns that range.
                let y = (y - hp[side].tick(y, 220.0, sr)) * level * breathe;
                music[i][side] += y;
                send[i] += y * 0.35;
            }
        }
    }

    // Bass and arp, from the chords, in the sections that ask for them.
    {
        let mut bph = 0.0;
        let mut bf = midi_hz(chords[0].2);
        let mut blp = OnePole::default();
        let mut arp_env = 0.0;
        let mut arp_ph = 0.0;
        let mut arp_f = 440.0;
        let mut arp_lp = OnePole::default();
        let sixteenth = beat / 4.0;
        let mut last16 = usize::MAX;
        for i in 0..n {
            let t = i as f64 / sr;
            if t >= music_end {
                break;
            }
            let ci = ((t / step_s) as usize).min(chords.len() - 1);
            let si = section_at(t);
            let (_, _, bass_l, arp_l) = mix(&arr.sections[si]);
            let target = midi_hz(chords[ci].2);
            bf += (target - bf) * 0.002;
            bph = (bph + bf / sr) % 1.0;
            let b = (2.0 * PI * bph).sin() + 0.3 * (4.0 * PI * bph).sin();
            let b = blp.tick(b, 400.0, sr) * 0.5 * bass_l;
            music[i][0] += b;
            music[i][1] += b;
            // Arp: 16ths over the chord tones, an octave up, as plucks.
            let k16 = (t / sixteenth) as usize;
            if k16 != last16 {
                last16 = k16;
                if arp_l > 0.0 {
                    let tones = &chords[ci].1;
                    let pick = [0usize, 2, 1, 3, 2, 1][k16 % 6] % tones.len();
                    arp_f = midi_hz(tones[pick] + 12.0);
                    arp_env = if k16 % 4 == 0 { 1.0 } else { 0.65 };
                }
            }
            arp_env *= (-1.0 / (0.07 * sr)).exp();
            arp_ph = (arp_ph + arp_f / sr) % 1.0;
            let a = (2.0 * PI * arp_ph).sin() + 0.25 * (6.0 * PI * arp_ph).sin();
            let a = arp_lp.tick(a, 3500.0, sr) * arp_env * 0.14 * arp_l;
            let p = pan(0.45 * ((k16 % 2) as f64 * 2.0 - 1.0));
            music[i][0] += a * p[0];
            music[i][1] += a * p[1];
            send[i] += a * 0.6;
        }
    }

    // ── drums ────────────────────────────────────────────────────────────────
    let kit = modal::Kit::new(key, &MAJOR);
    let vs = kit.voices();
    let sixteenth = beat / 4.0;
    for (si, s) in arr.sections.iter().enumerate() {
        for b in 0..s.bars {
            for st in 0..16 {
                let t = starts[si] + b as f64 * bar + st as f64 * sixteenth;
                for (v, vel, p) in pattern(&s.drums, b, s.bars, st, &mut rng) {
                    let at = (t * sr) as usize;
                    vs[v].strike(&mut drums, at, vel, pan(p), sr, &mut rng);
                    if v == 0 {
                        // Sidechain: the kick ducks the music for ~a 16th.
                        for (j, d) in duck[at..(at + (0.22 * sr) as usize).min(n)].iter_mut().enumerate() {
                            let x = j as f64 / (0.22 * sr);
                            *d = d.min(1.0 - 0.6 * vel * (1.0 - x).powi(2));
                        }
                    }
                }
            }
        }
    }
    // Bells and wood into the reverb: send a filtered copy of the drum bus.
    {
        let mut hp = Svf::default();
        for i in 0..n {
            let m = 0.5 * (drums[i][0] + drums[i][1]);
            send[i] += hp.tick(m, 600.0, 0.7, sr).high * 0.25;
        }
    }

    // ── risers: the loss curves ──────────────────────────────────────────────
    let mut riser_log = Vec::new();
    let mut fx = vec![[0.0f64; 2]; n];
    for (si, s) in arr.sections.iter().enumerate() {
        let Some(r) = &s.riser else { continue };
        let curve = riser::Curve::load(&docs.join(Path::new(&r.csv).strip_prefix("docs").unwrap_or(Path::new(&r.csv))), &r.column)?;
        let t0 = starts[si] + r.from_bar as f64 * bar;
        let t1 = starts[si] + s.bars as f64 * bar;
        riser_log.push(curve.render(&mut fx, t0, t1, r.level, sr, &mut rng, &r.csv));
    }

    // ── mix ──────────────────────────────────────────────────────────────────
    let mut rev = Reverb::new(sr);
    let mut mixbuf = vec![[0.0f64; 2]; n];
    for i in 0..n {
        let r = rev.tick(send[i]);
        for c in 0..2 {
            mixbuf[i][c] = music[i][c] * duck[i] + drums[i][c] * 0.9 + fx[i][c] + r[c] * 0.9;
        }
    }
    dsp::master(&mut mixbuf, 1.4, 0.89);

    let wav_path = out_dir.join("soundtrack.wav");
    wav::write(&wav_path, &mixbuf, sr as u32)?;
    let preview = section_window(&arr, &starts, bar, "build", "drop1");
    let (p0, p1) = ((preview.0 * sr) as usize, ((preview.1 * sr) as usize).min(n));
    wav::write(&out_dir.join("preview.wav"), &mixbuf[p0..p1], sr as u32)?;
    let marks: Vec<f64> = starts.iter().map(|&s| s / dur).collect();
    spectrogram::write(&out_dir.join("spectrogram.png"), &mixbuf, sr, &marks)?;

    // ── beats.json ───────────────────────────────────────────────────────────
    let total_beats = total_bars * arr.beats_per_bar;
    let beats: Vec<f64> = (0..=total_beats).map(|k| r3(k as f64 * beat)).collect();
    let downbeats: Vec<f64> = (0..=total_bars).map(|k| r3(k as f64 * bar)).collect();
    let drops: Vec<f64> = arr.sections.iter().zip(&starts).filter(|(s, _)| s.drop).map(|(_, &t)| r3(t)).collect();
    let sections = arr
        .sections
        .iter()
        .zip(&starts)
        .map(|(s, &t)| {
            let len = s.bars as f64 * bar / s.sequences.len() as f64;
            SectionOut {
                name: s.name.clone(),
                start: r3(t),
                end: r3(t + s.bars as f64 * bar),
                bars: s.bars,
                sequences: s
                    .sequences
                    .iter()
                    .enumerate()
                    .map(|(k, id)| SeqSpan { id: id.clone(), start: r3(t + k as f64 * len), end: r3(t + (k + 1) as f64 * len) })
                    .collect(),
            }
        })
        .collect();
    let beats_out = Beats {
        bpm: arr.bpm,
        key: format!("{} major", NAMES[key]),
        beat_s: beat,
        duration: r3(dur),
        stub_manifest: man.stub,
        beats,
        downbeats,
        drops,
        sections,
        risers: riser_log,
        chords: chord_log,
    };
    std::fs::write(out_dir.join("beats.json"), serde_json::to_string_pretty(&beats_out)?)?;

    for v in kit.voices() {
        let f: Vec<String> = v.modes.iter().take(4).map(|m| format!("{:.0}", m.hz)).collect();
        let rt = v.retuned.map_or(String::new(), |(f0, l0)| format!("  (retuned from {f0:.0} Hz, L {:.0} -> {:.1} mm)", l0 * 1e3, v.len * 1e3));
        println!("{:<38} {:>18}  modes {} Hz{rt}", v.label, v.sub.name, f.join(", "));
    }
    println!("{} ({:.1} s), preview {:.1}–{:.1} s", wav_path.display(), dur, preview.0, preview.1);
    Ok(())
}

use dsp::Reverb;

fn r3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn section_window(arr: &Arrangement, starts: &[f64], bar: f64, a: &str, b: &str) -> (f64, f64) {
    let i = arr.sections.iter().position(|s| s.name == a).unwrap_or(0);
    let j = arr.sections.iter().position(|s| s.name == b).unwrap_or(i);
    (starts[i], starts[j] + arr.sections[j].bars as f64 * bar)
}
