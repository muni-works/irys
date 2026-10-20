//! Scoring a scene estimate against the truth.

use serde::{Deserialize, Serialize};

use crate::data::Sample;
use crate::infer::canon_offset;
use crate::render::TABLE;
use crate::scene::{Cam, Object, param_used};

/// The truth in the camera's canonical frame, with visible pixel counts.
pub fn canonical_truth(s: &Sample) -> (Cam, Vec<(Object, usize)>) {
    let off = canon_offset(&s.scene.cam);
    let mut cam = s.scene.cam;
    cam.eye[0] -= off[0];
    cam.eye[1] -= off[1];
    cam.at[0] -= off[0];
    cam.at[1] -= off[1];
    let objs = s
        .scene
        .objects
        .iter()
        .enumerate()
        .map(|(i, o)| {
            let mut o = o.clone();
            o.pose[0] -= off[0];
            o.pose[1] -= off[1];
            (o, s.inst.iter().filter(|&&k| k != TABLE && k as usize == i).count())
        })
        .collect();
    (cam, objs)
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Score {
    pub gt: usize,
    pub pred: usize,
    pub matched: usize,
    pub class_ok: usize,
    pub topo_ok: usize,
    /// Per used parameter of topology-correct matches: |error| mm, and relative.
    pub dim_err_mm: Vec<f64>,
    pub dim_err_rel: Vec<f64>,
    /// Objects (topology correct) whose every dimension is within 0.5 / 1 mm.
    pub obj_05: usize,
    pub obj_1: usize,
    /// Objects within 5 % on every dimension.
    pub obj_5pct: usize,
    pub pos_err_mm: Vec<f64>,
    pub yaw_err_deg: Vec<f64>,
}

impl Score {
    pub fn add(&mut self, o: &Score) {
        self.gt += o.gt;
        self.pred += o.pred;
        self.matched += o.matched;
        self.class_ok += o.class_ok;
        self.topo_ok += o.topo_ok;
        self.dim_err_mm.extend(&o.dim_err_mm);
        self.dim_err_rel.extend(&o.dim_err_rel);
        self.obj_05 += o.obj_05;
        self.obj_1 += o.obj_1;
        self.obj_5pct += o.obj_5pct;
        self.pos_err_mm.extend(&o.pos_err_mm);
        self.yaw_err_deg.extend(&o.yaw_err_deg);
    }
}

pub fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

pub fn frac_below(v: &[f64], t: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().filter(|&&x| x <= t).count() as f64 / v.len() as f64
}

/// Greedy match on image distance of the base centres (each through its
/// own camera: a wrong scale must not hide a right detection); objects with
/// fewer than `min_px` visible pixels are not counted as truth (but a match
/// to them is not a false positive either). Position error is on the plane.
pub fn score(pred: &[Object], pcam: &Cam, truth: &[(Object, usize)], tcam: &Cam, min_px: usize) -> Score {
    let mut s = Score { gt: truth.iter().filter(|t| t.1 >= min_px).count(), pred: 0, ..Default::default() };
    let mut pairs = Vec::new();
    let pc = pcam.resized((crate::data::W, crate::data::H));
    let tc = tcam.resized((crate::data::W, crate::data::H));
    for (i, p) in pred.iter().enumerate() {
        let Some(pp) = pc.project([p.pose[0], p.pose[1], 0.0]) else { continue };
        for (j, (t, vis)) in truth.iter().enumerate() {
            let Some(tp) = tc.project([t.pose[0], t.pose[1], 0.0]) else { continue };
            let d = (pp[0] - tp[0]).hypot(pp[1] - tp[1]);
            let r_px = (*vis as f64 / std::f64::consts::PI).sqrt();
            if d < (0.6 * r_px).max(4.0) {
                pairs.push((d, i, j));
            }
        }
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut used_p = vec![false; pred.len()];
    let mut used_t = vec![false; truth.len()];
    for (d, i, j) in pairs {
        if used_p[i] || used_t[j] {
            continue;
        }
        used_p[i] = true;
        used_t[j] = true;
        let (t, vis) = &truth[j];
        if *vis < min_px {
            continue;
        }
        let p = &pred[i];
        s.matched += 1;
        let _ = d;
        s.pos_err_mm.push((p.pose[0] - t.pose[0]).hypot(p.pose[1] - t.pose[1]));
        if p.class == t.class {
            s.class_ok += 1;
        }
        let topo = p.family == t.family && (0..4).all(|k| !t.token_mask()[k] || p.tokens[k] == t.tokens[k]);
        if topo {
            s.topo_ok += 1;
            let (mut worst, mut worst_rel) = (0.0f64, 0.0f64);
            for k in 0..t.params.len() {
                if param_used(t.family, &t.tokens, k) {
                    let e = (p.params[k] - t.params[k]).abs();
                    s.dim_err_mm.push(e);
                    s.dim_err_rel.push(e / t.params[k]);
                    worst = worst.max(e);
                    worst_rel = worst_rel.max(e / t.params[k]);
                }
            }
            if worst <= 0.5 {
                s.obj_05 += 1;
            }
            if worst <= 1.0 {
                s.obj_1 += 1;
            }
            if worst_rel <= 0.05 {
                s.obj_5pct += 1;
            }
            if t.family.has_yaw() {
                let dy = ((p.pose[2] - t.pose[2]).rem_euclid(180.0) + 180.0) % 180.0;
                s.yaw_err_deg.push(dy.min(180.0 - dy));
            }
        }
    }
    // Predictions matched to sub-threshold truth are neither right nor wrong.
    let ignored = (0..pred.len()).filter(|&i| used_p[i]).count() - s.matched;
    s.pred = pred.len() - ignored;
    s
}

/// A summary row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    pub mode: String,
    pub scenes: usize,
    pub recall: f64,
    pub precision: f64,
    pub class_acc: f64,
    pub topology_acc: f64,
    pub dims_within_05mm: f64,
    pub dims_within_1mm: f64,
    pub objects_all_05mm: f64,
    pub objects_all_1mm: f64,
    pub objects_all_5pct: f64,
    pub median_dim_err_mm: f64,
    pub median_dim_err_rel: f64,
    pub median_pos_err_mm: f64,
    pub median_yaw_err_deg: f64,
    pub mean_mdl: f64,
    pub seconds_per_scene: f64,
}

pub fn row(mode: &str, scenes: usize, s: &Score, mdl: f64, secs: f64) -> Row {
    let m = s.matched.max(1) as f64;
    let t = s.topo_ok.max(1) as f64;
    Row {
        mode: mode.into(),
        scenes,
        recall: s.matched as f64 / s.gt.max(1) as f64,
        precision: s.matched as f64 / s.pred.max(1) as f64,
        class_acc: s.class_ok as f64 / m,
        topology_acc: s.topo_ok as f64 / m,
        dims_within_05mm: frac_below(&s.dim_err_mm, 0.5),
        dims_within_1mm: frac_below(&s.dim_err_mm, 1.0),
        objects_all_05mm: s.obj_05 as f64 / t,
        objects_all_1mm: s.obj_1 as f64 / t,
        objects_all_5pct: s.obj_5pct as f64 / t,
        median_dim_err_mm: median(&s.dim_err_mm),
        median_dim_err_rel: median(&s.dim_err_rel),
        median_pos_err_mm: median(&s.pos_err_mm),
        median_yaw_err_deg: median(&s.yaw_err_deg),
        mean_mdl: mdl,
        seconds_per_scene: secs,
    }
}

/// v3: the truth a single prediction matches (same rule as [`score`]).
pub fn match_one(p: &Object, pcam: &Cam, truth: &[(Object, usize)], tcam: &Cam, min_px: usize) -> Option<usize> {
    let pc = pcam.resized((crate::data::W, crate::data::H));
    let tc = tcam.resized((crate::data::W, crate::data::H));
    let pp = pc.project([p.pose[0], p.pose[1], 0.0])?;
    truth
        .iter()
        .enumerate()
        .filter(|(_, t)| t.1 >= min_px)
        .filter_map(|(j, (t, vis))| {
            let tp = tc.project([t.pose[0], t.pose[1], 0.0])?;
            let d = (pp[0] - tp[0]).hypot(pp[1] - tp[1]);
            let r_px = (*vis as f64 / std::f64::consts::PI).sqrt();
            (d < (0.6 * r_px).max(4.0)).then_some((d, j))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|x| x.1)
}

/// v3: one prediction's error against its truth, defined for any
/// structure: the mean relative error of the used dimensions when the
/// topology is right, else `1 +` the relative error of the bounding
/// footprint and height (a wrong structure is always worse than a right one).
pub fn object_error(p: &Object, t: &Object) -> (bool, f64) {
    let topo = p.family == t.family && (0..4).all(|k| !t.token_mask()[k] || p.tokens[k] == t.tokens[k]);
    if topo {
        let e: Vec<f64> = (0..t.params.len()).filter(|&k| param_used(t.family, &t.tokens, k)).map(|k| (p.params[k] - t.params[k]).abs() / t.params[k]).collect();
        return (true, e.iter().sum::<f64>() / e.len().max(1) as f64);
    }
    let rel = |a: f64, b: f64| (a - b).abs() / b.max(1e-6);
    (false, 1.0 + (rel(p.radius(), t.radius()) + rel(p.height(), t.height())) / 2.0)
}
