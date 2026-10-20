//! From heads to proposals: centre peaks, offset votes into instance masks,
//! attributes read at the peak, the support plane from the plane head, the
//! metric scale corrected by any known-size object in view.

use serde::{Deserialize, Serialize};

use crate::net::{GH, GW, H, P, PIX, PS, W};
use crate::scene::{CLASSES, Cam, Class, FAMILIES, Family, NP, NTOK, Object, VOCAB, denorm_mm, param_used};
use crate::train::Out;

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn softmax(z: &[f64]) -> Vec<f64> {
    let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = z.iter().map(|v| (v - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|v| v / s).collect()
}

/// One detected instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    /// Centre (projection of the base centre), net pixels.
    pub centre: [f64; 2],
    /// Peak patch.
    pub patch: usize,
    pub score: f64,
    /// Pixels (net image) of the instance.
    pub mask: Vec<u32>,
    pub cls: Vec<f64>,
    pub tok: Vec<Vec<f64>>,
    /// Normalised log-mm mean and σ.
    pub mu: Vec<f64>,
    pub sigma: Vec<f64>,
    pub yaw: [f64; 2],
}

/// Decode `out` into proposals (`thr`: centre heat threshold).
pub fn proposals(out: &Out, thr: f64) -> Vec<Proposal> {
    let heat: Vec<f64> = out.ctr.iter().map(|&z| sigmoid(z)).collect();
    let mut peaks = Vec::new();
    for p in 0..P {
        let (x, y) = ((p % GW) as i64, (p / GW) as i64);
        if heat[p] < thr {
            continue;
        }
        let mut is_max = true;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (nx, ny) = (x + dx, y + dy);
                if (dx, dy) != (0, 0) && nx >= 0 && ny >= 0 && nx < GW as i64 && ny < GH as i64 {
                    let q = (ny as usize) * GW + nx as usize;
                    if heat[q] > heat[p] || (heat[q] == heat[p] && q < p) {
                        is_max = false;
                    }
                }
            }
        }
        if is_max {
            peaks.push(p);
        }
    }
    // Votes: every foreground pixel's centre.
    let mut votes = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let p = (y / PS) * GW + x / PS;
            let j = (y % PS) * PS + x % PS;
            let at = |c: usize| out.pix[(p * 4 + c) * PIX + j];
            if sigmoid(at(0)) > 0.5 {
                let cx = x as f64 + 0.5 + at(1) * PS as f64;
                let cy = y as f64 + 0.5 + at(2) * PS as f64;
                votes.push(((y * W + x) as u32, cx, cy));
            }
        }
    }
    let mut centres: Vec<[f64; 2]> = peaks.iter().map(|&p| [((p % GW) as f64 + 0.5) * PS as f64, ((p / GW) as f64 + 0.5) * PS as f64]).collect();
    let mut assign = vec![usize::MAX; votes.len()];
    for _round in 0..2 {
        for (k, v) in votes.iter().enumerate() {
            let mut best = (f64::INFINITY, usize::MAX);
            for (i, c) in centres.iter().enumerate() {
                let d = (v.1 - c[0]).hypot(v.2 - c[1]);
                if d < best.0 {
                    best = (d, i);
                }
            }
            assign[k] = if best.0 < 2.5 * PS as f64 { best.1 } else { usize::MAX };
        }
        for (i, c) in centres.iter_mut().enumerate() {
            let mine: Vec<&(u32, f64, f64)> = votes.iter().zip(&assign).filter(|(_, a)| **a == i).map(|(v, _)| v).collect();
            if mine.len() >= 4 {
                let mut xs: Vec<f64> = mine.iter().map(|v| v.1).collect();
                let mut ys: Vec<f64> = mine.iter().map(|v| v.2).collect();
                xs.sort_by(f64::total_cmp);
                ys.sort_by(f64::total_cmp);
                *c = [xs[xs.len() / 2], ys[ys.len() / 2]];
            }
        }
    }
    peaks
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let mask = votes.iter().zip(&assign).filter(|(_, a)| **a == i).map(|(v, _)| v.0).collect();
            let tok = (0..NTOK).map(|k| softmax(&out.tok[(p * NTOK + k) * VOCAB..(p * NTOK + k + 1) * VOCAB])).collect();
            Proposal {
                centre: centres[i],
                patch: p,
                score: heat[p],
                mask,
                cls: softmax(&out.cls[p * CLASSES..(p + 1) * CLASSES]),
                tok,
                mu: (0..NP).map(|k| out.par[(p * 2) * NP + k]).collect(),
                sigma: (0..NP).map(|k| (4.0 * (out.par[(p * 2 + 1) * NP + k] * 0.25).tanh()).exp()).collect(),
                yaw: [out.yaw[p * 2], out.yaw[p * 2 + 1]],
            }
        })
        .collect()
}

/// A structure hypothesis: family and tokens, with its log probability.
#[derive(Debug, Clone)]
pub struct Hypothesis {
    pub family: Family,
    pub tokens: [usize; NTOK],
    pub logp: f64,
}

/// Which token values the grammar allows in a slot.
pub fn valid_token(f: Family, k: usize, v: usize) -> bool {
    match (f, k) {
        (_, 0) => v == f as usize,
        (Family::Plate, 1) => v <= 1,
        (Family::Plate, 2) => (1..=7).contains(&v),
        (Family::Plate, 3) => (1..=3).contains(&v),
        (Family::Product, 1) => v <= 1,
        (Family::Product, 2) => v <= 2,
        (Family::Wall, 1) => v <= 1,
        (Family::Arm, 1) => v <= 2,
        (Family::Grid, 2) => (2..=7).contains(&v),
        (Family::Shell, 1) => v <= 1,
        (Family::Shell, 2) => v <= 4,
        _ => v == 0,
    }
}

/// The `k` most probable structures under the token heads (a beam over
/// slots, restricted to the grammar).
pub fn hypotheses(p: &Proposal, k: usize) -> Vec<Hypothesis> {
    let mut all: Vec<Hypothesis> = Vec::new();
    for f in FAMILIES {
        let mut bs = vec![Hypothesis { family: f, tokens: [f as usize, 0, 0, 0], logp: p.tok[0][f as usize].max(1e-9).ln() }];
        for slot in 1..NTOK {
            let mut next = Vec::new();
            for b in &bs {
                let mut any = false;
                for v in 0..VOCAB {
                    if !valid_token(f, slot, v) {
                        continue;
                    }
                    let mut t = b.tokens;
                    t[slot] = v;
                    if !crate::scene::token_mask(f, &t)[slot] {
                        continue;
                    }
                    any = true;
                    next.push(Hypothesis { family: f, tokens: t, logp: b.logp + p.tok[slot][v].max(1e-9).ln() });
                }
                if !any {
                    let mut t = b.tokens;
                    t[slot] = 0;
                    next.push(Hypothesis { family: f, tokens: t, logp: b.logp });
                }
            }
            next.sort_by(|a, b| b.logp.total_cmp(&a.logp));
            next.truncate(k.max(2) * 2);
            bs = next;
        }
        all.extend(bs);
    }
    all.sort_by(|a, b| b.logp.total_cmp(&a.logp));
    all.truncate(k);
    all
}

/// The canonical camera of the plane head: at `(0, −H/tan pitch, H)`.
pub fn camera(out: &Out, hfov: f64) -> Cam {
    Cam::from_plane(out.plane, hfov, (W, H))
}

/// Translation from a world camera's frame into its canonical frame.
pub fn canon_offset(cam: &Cam) -> [f64; 2] {
    let pl = cam.plane();
    let h = cam.eye[2];
    [cam.eye[0], cam.eye[1] + h / pl[1].clamp(0.05, 1.5).tan()]
}

/// The most probable class of family `f`.
pub fn class_of(p: &Proposal, f: Family) -> Class {
    let mut best = (f64::NEG_INFINITY, Class::Bracket);
    for (i, &q) in p.cls.iter().enumerate() {
        if let Some(c) = Class::from_index(i) {
            if c.family() == f && q > best.0 {
                best = (q, c);
            }
        }
    }
    best.1
}

/// An object from a proposal and a structure, posed through `cam`.
pub fn object(p: &Proposal, h: &Hypothesis, cam: &Cam, mat: crate::scene::Mat) -> Option<Object> {
    let xy = cam.on_plane(p.centre[0], p.centre[1])?;
    let n = h.family.param_names().len();
    let mut params: Vec<f64> = (0..n).map(|k| denorm_mm(p.mu[k])).collect();
    for (k, v) in params.iter_mut().enumerate() {
        if !param_used(h.family, &h.tokens, k) {
            *v = match (h.family, k) {
                (Family::Product, 4) => 50.0,
                (Family::Shell, 4) => 10.0,
                (Family::Wall, 3) => 25.4,
                (Family::Wall, 4) => 3.0,
                (Family::Arm, 4) => 40.0,
                (Family::Arm, 5) => 60.0,
                (Family::Arm, 6) => 40.0,
                _ => *v,
            };
        }
    }
    let yaw = if h.family.has_yaw() { 0.5 * p.yaw[1].atan2(p.yaw[0]).to_degrees() } else { 0.0 };
    let class = class_of(p, h.family);
    let mut o = Object { class, family: h.family, tokens: h.tokens, params, pose: [xy[0], xy[1], yaw], mat };
    repair(&mut o);
    Some(o)
}

/// Nudge an object into the language's validity region.
pub fn repair(o: &mut Object) {
    let fam = o.family;
    let toks = o.tokens;
    let p = &mut o.params;
    for v in p.iter_mut() {
        *v = v.clamp(0.3, 3000.0);
    }
    match fam {
        Family::Plate => {
            p[2] = p[2].max(1.5);
            p[4] = p[4].min(15.0).max(0.6);
            if toks[1] == 0 {
                let (n, m) = (toks[2].max(1) as f64, toks[3].max(1) as f64);
                p[5] = p[5].max(2.0 * p[4] + 2.0);
                p[0] = p[0].max((n - 1.0) * p[5] + 2.0 * p[4] + 4.0);
                p[1] = p[1].max((m - 1.0) * p[5] + 2.0 * p[4] + 4.0);
            } else {
                p[5] = p[5].max(1.5 * p[4] + 1.0);
                let s = 2.0 * (p[5] + p[4]) + 4.0;
                p[0] = p[0].max(s);
                p[1] = p[1].max(s);
            }
            p[3] = if p[3] < 0.6 { 0.0 } else { p[3].min(0.3 * p[0].min(p[1])) };
        }
        Family::Product => {
            let cap = 0.45 * p[0].min(p[1]).min(if toks[1] == 1 { p[2] } else { 1e9 });
            p[3] = p[3].min(cap);
        }
        Family::Grid => {
            p[3] = p[3].max(6.0);
            p[0] = p[0].max(2.2 * p[3]);
            p[1] = p[1].max(2.2 * p[3]);
        }
        Family::Shell => {
            p[3] = p[3].min(0.15 * p[0].min(p[1])).max(0.6);
            p[4] = p[4].min(0.35 * p[0].min(p[1])).max(2.2 * p[3]);
        }
        Family::Lathe => {
            p[0] = p[0].max(6.0);
            for r in &mut p[1..5] {
                *r = r.max(2.5);
            }
        }
        Family::Wall => {
            p[0] = p[0].max(60.0);
            p[1] = p[1].max(60.0);
            p[2] = p[2].max(3.0);
            p[3] = p[3].clamp(9.0, 120.0);
            p[4] = p[4].min(0.3 * p[3]).max(1.2);
        }
        Family::Arm => {
            p[0] = p[0].max(25.0);
            p[1] = p[1].max(25.0);
            p[2] = p[2].clamp(3.5, 40.0);
            p[7] = p[7].clamp(1.0, 79.0);
            let lo = if toks[1] == 0 { p[2] + 31.0 } else { (p[2] + 31.0).max(0.5 * p[5] + 11.0) };
            p[3] = p[3].max(lo);
        }
    }
}

/// Metric scale from known-size objects: `(factor, which)`. The factor
/// multiplies every length (and the camera height).
pub fn gauge_scale(objs: &[Object], probs: &[f64]) -> (f64, Vec<usize>) {
    let mut logs = Vec::new();
    let mut who = Vec::new();
    for (i, o) in objs.iter().enumerate() {
        if let Some((k, mm)) = o.class.gauge() {
            if o.class.family() == o.family && probs[i] > 0.6 {
                logs.push((mm / o.params[k]).ln());
                who.push(i);
            }
        }
    }
    if logs.is_empty() {
        return (1.0, who);
    }
    logs.sort_by(f64::total_cmp);
    (logs[logs.len() / 2].exp(), who)
}

/// Scale a scene estimate by `s` (lengths, positions and the camera).
pub fn rescale(objs: &mut [Object], cam: &mut Cam, s: f64) {
    for o in objs.iter_mut() {
        for v in o.params.iter_mut() {
            *v *= s;
        }
        o.pose[0] *= s;
        o.pose[1] *= s;
    }
    for c in 0..3 {
        cam.eye[c] *= s;
        cam.at[c] *= s;
    }
}

/// The net's scene: objects (top-1 structure, with `−log p` of it),
/// alternatives (the next `k − 1` structures), the camera, and which
/// detections set the scale as gauges.
pub fn net_scene(out: &crate::train::Out, hfov: f64, k: usize, mat: crate::scene::Mat) -> (Vec<(Object, f64)>, Vec<Vec<(Object, f64)>>, Cam, Vec<usize>, Vec<Vec<f64>>) {
    let props = proposals(out, 0.35);
    let mut cam = camera(out, hfov);
    let mut objs = Vec::new();
    let mut alts = Vec::new();
    let mut probs = Vec::new();
    // Each detection's parameter σ in log units (the head's normalised σ × 1.5).
    let mut sds = Vec::new();
    for p in &props {
        let hs = hypotheses(p, k);
        let cands: Vec<(Object, f64)> = hs.iter().filter_map(|h| object(p, h, &cam, mat).map(|o| (o, -h.logp))).collect();
        if let Some(first) = cands.first() {
            probs.push(p.cls[first.0.class as usize]);
            objs.push(first.clone());
            alts.push(cands[1..].to_vec());
            sds.push(p.sigma.iter().map(|s| (1.5 * s).clamp(0.02, 1.0)).collect());
        }
    }
    let plain: Vec<Object> = objs.iter().map(|o: &(Object, f64)| o.0.clone()).collect();
    let (s, who) = gauge_scale(&plain, &probs);
    if who.is_empty() {
        return (objs, alts, cam, who, sds);
    }
    for o in objs.iter_mut() {
        rescale(std::slice::from_mut(&mut o.0), &mut cam.clone(), s);
    }
    for a in alts.iter_mut() {
        for o in a.iter_mut() {
            rescale(std::slice::from_mut(&mut o.0), &mut cam.clone(), s);
        }
    }
    rescale(&mut [], &mut cam, s);
    (objs, alts, cam, who, sds)
}
