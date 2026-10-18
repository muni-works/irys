//! The outer loop: a discrete search over programs, scored by MDL.
//!
//! A [`Proposer`] looks at the current fit (its program, its residual
//! images, its score, the history) and returns candidate programs. The
//! driver ([`recover`]) fits each (a short coarse-to-fine LM, warm-started
//! from the current parameters), scores it ([`crate::mdl::score`]) and
//! accepts the best only if it shortens the description. It stops when no
//! proposal does.
//!
//! v0's proposer is [`Algorithmic`]: residual blobs on the model's top face,
//! back-projected to the part, become hole / slot / counterbore / boss
//! candidates; equal blobs become patterns; and a fixed set of structural
//! edits (remove, hole↔counterbore, hole↔slot, merge holes into a pattern,
//! a pattern's count ±1, the corner finish) is always on offer.
//!
//! # An LLM proposer
//!
//! The trait is the seam. An agent implementing [`Proposer`] would get the
//! program as DSL text ([`Program::to_text`]), the score's three terms, the
//! judge's weak directions and a description of the residual (the blob list
//! [`blobs`] computes, or the residual images themselves for a vision
//! model), and return edited programs as text ([`Program::from_text`]).
//! Everything after that — snapping to an exact BRep, the fit, the MDL
//! verdict — stays here, so a hallucinated feature costs one fit and is
//! refused unless the photo pays for it. The agent's prior over programs
//! would replace [`crate::mdl::program_length`]'s grammar (its −log p of the
//! text).

use std::time::Instant;

use irys_core::{Point3, Ray, Vec3};

use crate::canon;
use crate::dsl::{Axis, Corner, Feature, Program};
use crate::fit::{self, Fit, Fitter, Level, Target};
use crate::mdl::{self, Score};
use crate::model::Setup;
use crate::render;

/// What a proposer sees.
pub struct Context<'a> {
    /// The current fit.
    pub fit: &'a Fit,
    /// Its score.
    pub score: Score,
    /// The target (search resolution).
    pub target: &'a Target,
    /// The setup (search resolution).
    pub setup: &'a Setup,
    /// Every step so far.
    pub history: &'a [Step],
}

/// A candidate program and why it was proposed.
#[derive(Debug, Clone)]
pub struct Proposal {
    /// The program (its parameters are the fit's warm start).
    pub program: Program,
    /// One line.
    pub why: String,
    /// For a proposal that adds one feature for one piece of evidence
    /// (a residual blob): that evidence's id. The driver combines the best
    /// interpretation of each piece of evidence into one more candidate,
    /// so a greedy round cannot lock in a joint explanation that is wrong
    /// for one of them.
    pub group: Option<usize>,
}

/// Proposes program edits.
pub trait Proposer {
    /// Candidates from `ctx`.
    fn propose(&mut self, ctx: &Context) -> Vec<Proposal>;
}

/// One candidate tried.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Step {
    /// Search round.
    pub round: usize,
    /// Why it was proposed.
    pub why: String,
    /// Its program (canonical text) after the fit, or the error.
    pub program: String,
    /// Its score, when it built.
    pub score: Option<Score>,
    /// Whether it was accepted.
    pub accepted: bool,
}

/// A residual blob on the model's top face, in part coordinates.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Blob {
    /// Centre, mm.
    pub x: f64,
    /// Centre, mm.
    pub y: f64,
    /// Half-width across (minor axis), mm.
    pub r: f64,
    /// Half-length along (major axis), mm.
    pub half_len: f64,
    /// Major axis angle, radians.
    pub angle: f64,
    /// Pixels (summed over views).
    pub pixels: usize,
    /// A raised feature's centre and height from the blob's parallax
    /// across views, `(x, y, height)`, when two or more views saw it.
    pub raised: Option<(f64, f64, f64)>,
}

/// Where the model's top face disagrees with the target: per view, pixels
/// whose camera ray hits the model on its top face (`z = h`, upward
/// normal) and whose blurred residual exceeds 4σ, grouped into connected
/// components, back-projected to the top face and merged across views.
pub fn blobs(fit: &Fit, setup: &Setup) -> Vec<Blob> {
    let Ok(solid) = fit.prog.build() else { return Vec::new() };
    let mesh = crate::geom::mesh(&solid);
    let h = fit.prog.h;
    let (w, ht) = setup.film.size;
    let mut per_view: Vec<(Vec<(f64, f64)>, usize)> = Vec::new();
    let mut eyes = Vec::new();
    for (vi, v) in setup.views.iter().enumerate() {
        let sc = render::scene(Some(&mesh), v, setup.film);
        let (eye, f, rt, up, _) = sc.camera();
        eyes.push(eye);
        let rb = fit::blur(&fit.residual[vi], (w, ht), 1, 1.0);
        let mut top: Vec<Option<(f64, f64)>> = vec![None; w * ht];
        for p in 0..w * ht {
            // Blurred by σ = 1 px, the noise's std is ≈ 0.28 of the
            // unblurred one: 2 is ≈ 7 of its σ.
            if rb[p].abs() < 2.0 {
                continue;
            }
            let s = sc.screen((p % w) as f64 + 0.5, (p / w) as f64 + 0.5);
            if let Some(hit) = sc.tlas().trace_closest(&Ray::new(eye, f + rt * s[0] + up * s[1])) {
                let n = hit.hit.normal.as_ref();
                if n.z > 0.95 && (hit.hit.point.z - h).abs() < 0.05 {
                    top[p] = Some((hit.hit.point.x, hit.hit.point.y));
                }
            }
        }
        // Connected components (4-neighbour).
        let mut seen = vec![false; w * ht];
        for p in 0..w * ht {
            if seen[p] || top[p].is_none() {
                continue;
            }
            let mut stack = vec![p];
            seen[p] = true;
            let mut pts = Vec::new();
            while let Some(q) = stack.pop() {
                pts.push(top[q].unwrap());
                let (x, y) = (q % w, q / w);
                let mut nb = Vec::new();
                if x > 0 { nb.push(q - 1) }
                if x + 1 < w { nb.push(q + 1) }
                if y > 0 { nb.push(q - w) }
                if y + 1 < ht { nb.push(q + w) }
                for n in nb {
                    if !seen[n] && top[n].is_some() {
                        seen[n] = true;
                        stack.push(n);
                    }
                }
            }
            if pts.len() >= 4 {
                per_view.push((pts, vi));
            }
        }
    }
    // One blob per component, then merge across views by centre distance.
    let mut out: Vec<(Blob, Vec<(f64, f64)>, Vec<(usize, f64, f64)>)> = Vec::new();
    for (pts, vi) in per_view {
        let b = describe(&pts);
        let c = (vi, b.x, b.y);
        if let Some((m, mp, cs)) = out.iter_mut().find(|(m, ..)| (m.x - b.x).hypot(m.y - b.y) < (m.r.max(b.r) + 1.0)) {
            mp.extend(pts);
            cs.push(c);
            *m = describe(mp);
        } else {
            out.push((b, pts, vec![c]));
        }
    }
    let mut v: Vec<Blob> = out
        .into_iter()
        .map(|(mut b, _, cs)| {
            b.raised = parallax(&cs, &eyes, h);
            b
        })
        .collect();
    v.sort_by(|a, b| b.pixels.cmp(&a.pixels).then(a.x.total_cmp(&b.x)));
    v
}

/// A raised cylinder's footprint on `z = h`, seen from eye `e`, is the hull
/// of its base disc at `p` and its top disc pushed by `bh · t` (`t` the
/// view ray's horizontal run per unit drop), so its centroid is about
/// `p + (bh/2) t`. Least squares over the views for `(p, bh)`.
fn parallax(cs: &[(usize, f64, f64)], eyes: &[Point3], h: f64) -> Option<(f64, f64, f64)> {
    let mut seen: Vec<usize> = cs.iter().map(|c| c.0).collect();
    seen.dedup();
    if seen.len() < 2 {
        return None;
    }
    // Unknowns (px, py, β): c = p + β t.
    let mut a = [[0.0; 3]; 3];
    let mut b = [0.0; 3];
    for &(vi, cx, cy) in cs {
        let e = eyes[vi];
        let d = Vec3::new(cx - e.x, cy - e.y, h - e.z);
        if d.z >= -1e-6 {
            continue;
        }
        let t = [d.x / -d.z, d.y / -d.z];
        let rows = [([1.0, 0.0, t[0]], cx), ([0.0, 1.0, t[1]], cy)];
        for (r, y) in rows {
            for i in 0..3 {
                b[i] += r[i] * y;
                for j in 0..3 {
                    a[i][j] += r[i] * r[j];
                }
            }
        }
    }
    let m: Vec<f64> = a.iter().flatten().copied().collect();
    let x = fit::solve(&m, &b, 3)?;
    let bh = 2.0 * x[2];
    (bh > 1.0 && bh < 40.0).then_some((x[0], x[1], bh))
}

fn describe(pts: &[(f64, f64)]) -> Blob {
    let n = pts.len() as f64;
    let (mx, my) = (pts.iter().map(|p| p.0).sum::<f64>() / n, pts.iter().map(|p| p.1).sum::<f64>() / n);
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for p in pts {
        let (dx, dy) = (p.0 - mx, p.1 - my);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    let th = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let (ux, uy) = (th.cos(), th.sin());
    let extent = |ax: f64, ay: f64| {
        let mut s: Vec<f64> = pts.iter().map(|p| (p.0 - mx) * ax + (p.1 - my) * ay).collect();
        s.sort_by(f64::total_cmp);
        let k = s.len();
        let (lo, hi) = (s[k / 50], s[k - 1 - k / 50]);
        ((hi - lo) / 2.0, (hi + lo) / 2.0)
    };
    let (major, cmaj) = extent(ux, uy);
    let (minor, cmin) = extent(-uy, ux);
    Blob {
        x: mx + ux * cmaj - uy * cmin,
        y: my + uy * cmaj + ux * cmin,
        r: minor.max(0.5),
        half_len: major.max(0.5),
        angle: th,
        pixels: pts.len(),
        raised: None,
    }
}

/// v0's enumerative proposer.
#[derive(Default)]
pub struct Algorithmic;

fn near_existing(p: &Program, x: f64, y: f64) -> bool {
    p.features.iter().any(|f| f.footprint().iter().any(|&(fx, fy, fr)| (fx - x).hypot(fy - y) < fr + 2.0))
}

impl Algorithmic {
    fn adds(&self, ctx: &Context, all: &[Blob], out: &mut Vec<Proposal>) {
        let p = &ctx.fit.prog;
        // Blobs at the outline are the corner finish or a side misfit, not a
        // feature (features keep a margin from the edge).
        let inside = |b: &Blob| b.x.abs() < p.w / 2.0 - 2.5 && b.y.abs() < p.d / 2.0 - 2.5;
        let bl: Vec<Blob> = all.iter().filter(|b| !near_existing(p, b.x, b.y) && inside(b)).cloned().collect();
        if std::env::var("RECON_DEBUG").is_ok() {
            let mx: Vec<f64> = ctx.fit.residual.iter().map(|r| r.iter().fold(0.0f64, |m, x| m.max(x.abs()))).collect();
            eprintln!("blobs {:?} max|r| {mx:?}", bl);
        }
        let with = |f: Feature| {
            let mut q = p.clone();
            q.features.push(f);
            q
        };
        for (bi, b) in bl.iter().take(6).enumerate() {
            let (x, y, r) = (b.x, b.y, b.r.clamp(1.0, 15.0));
            out.push(Proposal { program: with(Feature::Hole { x, y, r }), why: format!("hole at blob ({x:.1}, {y:.1}) r {r:.1}"), group: Some(bi) });
            if b.half_len > 1.5 * b.r {
                let axis = if b.angle.cos().abs() > b.angle.sin().abs() { Axis::X } else { Axis::Y };
                let len = (2.0 * (b.half_len - b.r)).max(2.0);
                out.push(Proposal {
                    program: with(Feature::Slot { x, y, len, r, axis }),
                    why: format!("slot{axis:?} at elongated blob ({x:.1}, {y:.1}) len {len:.1}"),
                    group: Some(bi),
                });
            }
            out.push(Proposal {
                program: with(Feature::Counterbore { x, y, r: (0.55 * r).max(1.0), cr: r, cd: p.h / 3.0 }),
                why: format!("counterbore at blob ({x:.1}, {y:.1})"),
                group: Some(bi),
            });
            out.push(Proposal {
                program: with(Feature::Boss { x, y, r, bh: 4.0 }),
                why: format!("boss at blob ({x:.1}, {y:.1})"),
                group: Some(bi),
            });
            if let Some((bx, by, bh)) = b.raised {
                out.push(Proposal {
                    program: with(Feature::Boss { x: bx, y: by, r: (0.8 * r).max(1.0), bh }),
                    why: format!("boss by parallax ({bx:.1}, {by:.1}) height {bh:.1}"),
                    group: Some(bi),
                });
            }
        }
        // Several blobs of one size: holes all at once, or a pattern.
        let small: Vec<&Blob> = bl.iter().filter(|b| b.half_len < 1.5 * b.r).collect();
        if small.len() >= 2 {
            let r0 = small[0].r;
            let same: Vec<(f64, f64, f64)> =
                small.iter().filter(|b| (b.r - r0).abs() < 0.35 * r0 + 0.5).map(|b| (b.x, b.y, r0)).collect();
            if same.len() >= 3 {
                if let Some(c) = force_circular(&same) {
                    out.push(Proposal { program: with(c), why: format!("circular pattern through {} blobs", same.len()), group: None });
                }
                for c in circular_supersets(&same) {
                    let Feature::CircularHoles { n, .. } = c else { unreachable!() };
                    out.push(Proposal { program: with(c), why: format!("circular ×{n} through {} blobs", same.len()), group: None });
                }
            }
            if same.len() >= 2 {
                if let Some(l) = force_linear(&same) {
                    out.push(Proposal { program: with(l), why: format!("linear pattern through {} blobs", same.len()), group: None });
                }
                let mut q = p.clone();
                for &(x, y, r) in &same {
                    q.features.push(Feature::Hole { x, y, r });
                }
                out.push(Proposal { program: q, why: format!("{} holes at once", same.len()), group: None });
            }
        }
    }

    fn edits(&self, p: &Program, all: &[Blob], out: &mut Vec<Proposal>) {
        let h = p.h;
        let raised_near = |x: f64, y: f64| {
            all.iter().find(|b| (b.x - x).hypot(b.y - y) < b.r + 4.0).and_then(|b| b.raised).map(|(bx, by, bh)| {
                let r = all.iter().find(|b| (b.x - x).hypot(b.y - y) < b.r + 4.0).map_or(3.0, |b| (0.8 * b.r).max(1.0));
                Feature::Boss { x: bx, y: by, r, bh }
            })
        };
        // The corner.
        let corners = match p.corner {
            Corner::Sharp => vec![Corner::Fillet(4.0), Corner::Chamfer(4.0)],
            Corner::Fillet(r) => vec![Corner::Chamfer(r * 0.7), Corner::Sharp],
            Corner::Chamfer(c) => vec![Corner::Fillet(c / 0.7), Corner::Sharp],
        };
        for c in corners {
            out.push(Proposal { program: Program { corner: c, ..p.clone() }, why: format!("corner → {c:?}"), group: None });
        }
        for (i, f) in p.features.iter().enumerate() {
            let mut rm = p.clone();
            rm.features.remove(i);
            out.push(Proposal { program: rm, why: format!("remove f{i} ({})", f.kind()), group: None });
            let swap = |g: Feature| {
                let mut q = p.clone();
                q.features[i] = g;
                q
            };
            match *f {
                Feature::Hole { x, y, r } => {
                    if let Some(raised) = raised_near(x, y) {
                        out.push(Proposal { program: swap(raised), why: format!("f{i} hole → boss by parallax"), group: None });
                    }
                    out.push(Proposal {
                        program: swap(Feature::Counterbore { x, y, r, cr: r + 2.0, cd: h / 3.0 }),
                        why: format!("f{i} hole → counterbore (bore outside)"),
                        group: None,
                    });
                    if r > 2.2 {
                        out.push(Proposal {
                            program: swap(Feature::Counterbore { x, y, r: 0.55 * r, cr: r, cd: h / 2.0 }),
                            why: format!("f{i} hole → counterbore (hole inside)"),
                            group: None,
                        });
                    }
                    for axis in [Axis::X, Axis::Y] {
                        out.push(Proposal {
                            program: swap(Feature::Slot { x, y, len: 3.0, r, axis }),
                            why: format!("f{i} hole → slot{axis:?}"),
                            group: None,
                        });
                    }
                    out.push(Proposal { program: swap(Feature::Boss { x, y, r, bh: 3.0 }), why: format!("f{i} hole → boss"), group: None });
                }
                Feature::Counterbore { x, y, r, cr, .. } => {
                    out.push(Proposal { program: swap(Feature::Hole { x, y, r }), why: format!("f{i} counterbore → hole"), group: None });
                    out.push(Proposal { program: swap(Feature::Hole { x, y, r: cr }), why: format!("f{i} counterbore → hole (outer)"), group: None });
                }
                Feature::Slot { x, y, r, axis, .. } => {
                    out.push(Proposal { program: swap(Feature::Hole { x, y, r }), why: format!("f{i} slot → hole"), group: None });
                    let other = if axis == Axis::X { Axis::Y } else { Axis::X };
                    if let Feature::Slot { len, .. } = *f {
                        out.push(Proposal { program: swap(Feature::Slot { x, y, len, r, axis: other }), why: format!("f{i} slot axis"), group: None });
                    }
                }
                Feature::Boss { x, y, r, .. } => {
                    out.push(Proposal { program: swap(Feature::Hole { x, y, r }), why: format!("f{i} boss → hole"), group: None });
                }
                Feature::LinearHoles { x, y, dx, dy, r, n } => {
                    out.push(Proposal { program: swap(Feature::LinearHoles { x, y, dx, dy, r, n: n + 1 }), why: format!("f{i} linear n+1 (end)"), group: None });
                    out.push(Proposal {
                        program: swap(Feature::LinearHoles { x: x - dx, y: y - dy, dx, dy, r, n: n + 1 }),
                        why: format!("f{i} linear n+1 (start)"),
                        group: None,
                    });
                    if n > 2 {
                        out.push(Proposal { program: swap(Feature::LinearHoles { x, y, dx, dy, r, n: n - 1 }), why: format!("f{i} linear n−1 (end)"), group: None });
                        out.push(Proposal {
                            program: swap(Feature::LinearHoles { x: x + dx, y: y + dy, dx, dy, r, n: n - 1 }),
                            why: format!("f{i} linear n−1 (start)"),
                            group: None,
                        });
                    }
                }
                Feature::CircularHoles { x, y, rad, phase, r, n } => {
                    for m in [n + 1, n.saturating_sub(1)] {
                        if m >= 3 {
                            out.push(Proposal { program: swap(Feature::CircularHoles { x, y, rad, phase, r, n: m }), why: format!("f{i} circular n → {m}"), group: None });
                        }
                    }
                }
            }
        }
        // Merge plain holes of about one size into a pattern.
        let holes: Vec<(usize, (f64, f64, f64))> = p
            .features
            .iter()
            .enumerate()
            .filter_map(|(i, f)| if let Feature::Hole { x, y, r } = *f { Some((i, (x, y, r))) } else { None })
            .collect();
        if holes.len() >= 2 {
            let hs: Vec<(f64, f64, f64)> = holes.iter().map(|h| h.1).collect();
            let rest = |q: &mut Program| q.features.retain(|f| !matches!(f, Feature::Hole { .. }));
            if hs.len() >= 3 {
                for c in circular_supersets(&hs) {
                    let Feature::CircularHoles { n, .. } = c else { unreachable!() };
                    let mut q = p.clone();
                    rest(&mut q);
                    q.features.push(c);
                    out.push(Proposal { program: q, why: format!("merge {} holes → circular ×{n}", hs.len()), group: None });
                }
                if let Some(c) = force_circular(&hs) {
                    let mut q = p.clone();
                    rest(&mut q);
                    q.features.push(c);
                    out.push(Proposal { program: q, why: format!("merge {} holes → circular", hs.len()), group: None });
                }
            }
            if let Some(l) = force_linear(&hs) {
                let mut q = p.clone();
                rest(&mut q);
                q.features.push(l);
                out.push(Proposal { program: q, why: format!("merge {} holes → linear", hs.len()), group: None });
            }
        }
        // A hole next to a pattern of its size joins it (the pattern's n+1 covers
        // the ends; this covers a hole found before its pattern was).
        let canon = canon::canonical(p);
        if canon.to_text() != p.to_text() {
            out.push(Proposal { program: canon, why: "canonical form".into(), group: None });
        }
    }
}

/// Circular patterns of `n ≥ k` holes of which `holes` (`k ≥ 3`) are
/// members: the circle through them, every hole within 6° of a slot of
/// `360/n`. Finds a pattern whose other holes the blobs missed.
pub fn circular_supersets(holes: &[(f64, f64, f64)]) -> Vec<Feature> {
    let k = holes.len();
    if k < 3 {
        return Vec::new();
    }
    let pts: Vec<(f64, f64)> = holes.iter().map(|h| (h.0, h.1)).collect();
    let (cx, cy, rad, rms) = canon::circle(&pts);
    if !(rms < 1.5 && rad > 2.0) {
        return Vec::new();
    }
    let r = holes.iter().map(|h| h.2).sum::<f64>() / k as f64;
    let ang: Vec<f64> = pts.iter().map(|p| (p.1 - cy).atan2(p.0 - cx).to_degrees()).collect();
    let mut out = Vec::new();
    for n in k + 1..=(k + 3).min(8) {
        let step = 360.0 / n as f64;
        let phase = ang[0].rem_euclid(step);
        let ok = ang.iter().all(|a| {
            let d = (a - phase).rem_euclid(step);
            d.min(step - d) < 6.0
        });
        if ok {
            out.push(Feature::CircularHoles { x: cx, y: cy, rad, phase, r, n });
        }
    }
    out
}

/// A circular pattern through `holes` whatever the spacing error.
pub fn force_circular(holes: &[(f64, f64, f64)]) -> Option<Feature> {
    let n = holes.len();
    let pts: Vec<(f64, f64)> = holes.iter().map(|h| (h.0, h.1)).collect();
    let (cx, cy) = (pts.iter().map(|p| p.0).sum::<f64>() / n as f64, pts.iter().map(|p| p.1).sum::<f64>() / n as f64);
    let rad = pts.iter().map(|p| (p.0 - cx).hypot(p.1 - cy)).sum::<f64>() / n as f64;
    if rad < 2.0 {
        return None;
    }
    let step = 360.0 / n as f64;
    let (mut s, mut c) = (0.0, 0.0);
    for p in &pts {
        let a = (p.1 - cy).atan2(p.0 - cx).to_degrees().rem_euclid(step) / step * std::f64::consts::TAU;
        s += a.sin();
        c += a.cos();
    }
    let phase = (s.atan2(c) / std::f64::consts::TAU * step).rem_euclid(step);
    let r = holes.iter().map(|h| h.2).sum::<f64>() / n as f64;
    Some(Feature::CircularHoles { x: cx, y: cy, rad, phase, r, n })
}

/// A linear pattern through `holes` (sorted along their principal axis),
/// whatever the spacing error.
pub fn force_linear(holes: &[(f64, f64, f64)]) -> Option<Feature> {
    let n = holes.len();
    let (mx, my) = (holes.iter().map(|h| h.0).sum::<f64>() / n as f64, holes.iter().map(|h| h.1).sum::<f64>() / n as f64);
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for h in holes {
        sxx += (h.0 - mx).powi(2);
        sxy += (h.0 - mx) * (h.1 - my);
        syy += (h.1 - my).powi(2);
    }
    let th = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let (ux, uy) = (th.cos(), th.sin());
    let mut s: Vec<f64> = holes.iter().map(|h| (h.0 - mx) * ux + (h.1 - my) * uy).collect();
    s.sort_by(f64::total_cmp);
    let span = s[n - 1] - s[0];
    if span < 2.0 {
        return None;
    }
    let step = span / (n - 1) as f64;
    let (dx, dy) = (ux * step, uy * step);
    let (x0, y0) = (mx + ux * s[0] + (-uy) * 0.0, my + uy * s[0]);
    let r = holes.iter().map(|h| h.2).sum::<f64>() / n as f64;
    let mut f = Feature::LinearHoles { x: x0, y: y0, dx, dy, r, n };
    // Snap a near-axis direction onto the axis (the fit refines the rest).
    if let Feature::LinearHoles { dx, dy, .. } = &mut f {
        if dx.abs() < 0.15 * dy.abs() {
            *dx = 0.0;
        }
        if dy.abs() < 0.15 * dx.abs() {
            *dy = 0.0;
        }
    }
    let p = Program { w: 1.0, d: 1.0, h: 1.0, corner: Corner::Sharp, features: vec![f] };
    canon::canonical(&p).features.into_iter().next().or(Some(Feature::LinearHoles { x: x0, y: y0, dx, dy, r, n }))
}

impl Proposer for Algorithmic {
    fn propose(&mut self, ctx: &Context) -> Vec<Proposal> {
        let mut out = Vec::new();
        let all = blobs(ctx.fit, ctx.setup);
        self.adds(ctx, &all, &mut out);
        self.edits(&ctx.fit.prog, &all, &mut out);
        out
    }
}

/// How the recovery went.
pub struct Recovery {
    /// The recovered program (canonical), fitted at full resolution.
    pub program: Program,
    /// The full-resolution fit.
    pub fit: Fit,
    /// Its score.
    pub score: Score,
    /// The judge.
    pub judge: mdl::Judge,
    /// Every candidate tried.
    pub trace: Vec<Step>,
    /// Accepted scores per round (search resolution), from the base fit on.
    pub accepted: Vec<(String, Score)>,
    /// LM iterations, over everything.
    pub iters: usize,
    /// Renders, over everything.
    pub evals: usize,
    /// Wall time, seconds.
    pub seconds: f64,
}

/// Options.
#[derive(Debug, Clone)]
pub struct Options {
    /// Parallel candidate fits.
    pub workers: usize,
    /// Most rounds.
    pub rounds: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self { workers: 4, rounds: 8 }
    }
}

/// The base box from the silhouettes alone: a grid over `(w, d, h)` scored
/// by mask IoU (ray–box slabs, no render).
fn init_base(target: &Target, setup: &Setup) -> Program {
    let (w, h) = target.size;
    let rays: Vec<Vec<(Point3, Vec3)>> = setup
        .views
        .iter()
        .map(|v| {
            let sc = render::scene(None, v, setup.film);
            let (eye, f, r, u, _) = sc.camera();
            (0..w * h)
                .map(|p| {
                    let s = sc.screen((p % w) as f64 + 0.5, (p / w) as f64 + 0.5);
                    (eye, f + r * s[0] + u * s[1])
                })
                .collect()
        })
        .collect();
    let hit = |o: Point3, d: Vec3, a: f64, b: f64, c: f64| {
        let (mut t0, mut t1) = (0.0f64, f64::INFINITY);
        for (oo, dd, lo, hi) in [(o.x, d.x, -a, a), (o.y, d.y, -b, b), (o.z, d.z, 0.0, c)] {
            if dd.abs() < 1e-12 {
                if oo < lo || oo > hi {
                    return false;
                }
                continue;
            }
            let (ta, tb) = ((lo - oo) / dd, (hi - oo) / dd);
            t0 = t0.max(ta.min(tb));
            t1 = t1.min(ta.max(tb));
        }
        t0 <= t1
    };
    let mut best = (f64::NEG_INFINITY, 60.0, 50.0, 10.0);
    let mut grid = Vec::new();
    for wi in 0..=12 {
        for di in 0..=10 {
            for hi in 0..=6 {
                grid.push((40.0 + 5.0 * wi as f64, 30.0 + 5.0 * di as f64, 4.0 + 3.0 * hi as f64));
            }
        }
    }
    for (bw, bd, bh) in grid {
        let (mut inter, mut uni) = (0.0f64, 0.0f64);
        for (v, rs) in rays.iter().enumerate() {
            for (p, &(o, d)) in rs.iter().enumerate() {
                let a = hit(o, d, bw / 2.0, bd / 2.0, bh);
                let m = target.mask[v][p] > 0.5;
                if a && m {
                    inter += 1.0;
                }
                if a || m {
                    uni += 1.0;
                }
            }
        }
        let iou = inter / uni.max(1.0);
        if iou > best.0 {
            best = (iou, bw, bd, bh);
        }
    }
    Program { w: best.1, d: best.2, h: best.3, corner: Corner::Sharp, features: Vec::new() }
}

/// Recover a program from a full-resolution target. The search runs at
/// half resolution (`setup_full` halved, the target box-downsampled); the
/// winner is refitted and scored at full resolution.
pub fn recover(
    target_full: &Target,
    background_full: &[Vec<f64>],
    setup_full: &Setup,
    target_spp: usize,
    proposer: &mut dyn Proposer,
    opts: &Options,
) -> Recovery {
    let t0 = Instant::now();
    let target = target_full.half(background_full);
    let mut setup = setup_full.clone();
    setup.film.size = target.size;
    setup.edge_samples = setup_full.edge_samples / 3;
    setup.film.spp = (setup_full.film.spp / 2).max(4);
    let base = init_base(&target, &setup);
    let probe = Fitter::new(setup.clone(), target.clone(), target_spp, &base);
    let sigma2 = probe.sigma2;
    let mut fitters: Vec<Fitter> =
        (0..opts.workers).map(|_| Fitter::with_sigma2(setup.clone(), target.clone(), sigma2)).collect();
    let (mut iters, mut evals) = (0usize, 2usize);
    let mut cur = fitters[0].fit(&base, &fit::schedule()).expect("the base box fits");
    iters += cur.iters;
    evals += cur.evals;
    let mut cur_score = mdl::score(&cur);
    let mut trace = vec![Step { round: 0, why: "base box from silhouettes".into(), program: cur.prog.to_text(), score: Some(cur_score), accepted: true }];
    let mut accepted = vec![(cur.prog.to_text(), cur_score)];
    for round in 1..=opts.rounds {
        let mut tried: std::collections::HashSet<String> = std::collections::HashSet::new();
        let ctx = Context { fit: &cur, score: cur_score, target: &target, setup: &setup, history: &trace };
        let mut props = proposer.propose(&ctx);
        // Drop invalid and already-tried structures-at-parameters.
        props.retain(|p| p.program.valid() && p.program.signature() + &round_text(&p.program) != cur.prog.signature() + &round_text(&cur.prog));
        props.retain(|p| tried.insert(format!("{}|{}", p.program.signature(), round_text(&p.program))));
        if props.is_empty() {
            break;
        }
        let results: Vec<(usize, Result<Fit, String>)> = {
            let chunks: Vec<Vec<(usize, &Proposal)>> = (0..opts.workers)
                .map(|w| props.iter().enumerate().filter(|(i, _)| i % opts.workers == w).collect())
                .collect();
            std::thread::scope(|s| {
                let hs: Vec<_> = fitters
                    .iter_mut()
                    .zip(chunks)
                    .map(|(f, ch)| s.spawn(move || ch.into_iter().map(|(i, p)| (i, f.fit(&p.program, &fit::short()))).collect::<Vec<_>>()))
                    .collect();
                let mut all: Vec<(usize, Result<Fit, String>)> = hs.into_iter().flat_map(|h| h.join().unwrap()).collect();
                all.sort_by_key(|(i, _)| *i);
                all
            })
        };
        let mut best: Option<(usize, Fit, Score)> = None;
        let first = trace.len();
        for (i, r) in results {
            match r {
                Ok(f) if degenerate(&f.prog).is_some() => {
                    iters += f.iters;
                    evals += f.evals;
                    let why = degenerate(&f.prog).unwrap();
                    trace.push(Step { round, why: format!("{} (degenerate: {why})", props[i].why), program: f.prog.to_text(), score: Some(mdl::score(&f)), accepted: false });
                }
                Ok(f) => {
                    iters += f.iters;
                    evals += f.evals;
                    let s = mdl::score(&f);
                    trace.push(Step { round, why: props[i].why.clone(), program: f.prog.to_text(), score: Some(s), accepted: false });
                    if best.as_ref().is_none_or(|b| s.total < b.2.total) {
                        best = Some((trace.len() - 1, f, s));
                    }
                }
                Err(e) => trace.push(Step { round, why: props[i].why.clone(), program: format!("error: {e}"), score: None, accepted: false }),
            }
        }
        // Combine: the best fitted interpretation of each piece of evidence
        // that beats the current program on its own, all at once.
        let mut by_group: std::collections::BTreeMap<usize, (f64, Feature)> = std::collections::BTreeMap::new();
        for (k, step) in trace[first..].iter().enumerate() {
            let (Some(g), Some(sc)) = (props.get(k).and_then(|p| p.group), step.score) else { continue };
            if sc.total >= cur_score.total || step.why.contains("degenerate") {
                continue;
            }
            let Ok(prog) = Program::from_text(&step.program) else { continue };
            let Some(f) = prog.features.last().cloned() else { continue };
            if by_group.get(&g).is_none_or(|(t, _)| sc.total < *t) {
                by_group.insert(g, (sc.total, f));
            }
        }
        if by_group.len() >= 2 {
            let mut q = cur.prog.clone();
            q.features.extend(by_group.values().map(|(_, f)| f.clone()));
            let why = format!("combine the best of {} blobs", by_group.len());
            if q.valid() {
                match fitters[0].fit(&q, &fit::short()) {
                    Ok(f) if degenerate(&f.prog).is_none() => {
                        iters += f.iters;
                        evals += f.evals;
                        let s = mdl::score(&f);
                        trace.push(Step { round, why, program: f.prog.to_text(), score: Some(s), accepted: false });
                        if best.as_ref().is_none_or(|b| s.total < b.2.total) {
                            best = Some((trace.len() - 1, f, s));
                        }
                    }
                    _ => {}
                }
            }
        }
        match best {
            Some((ti, f, s)) if s.total < cur_score.total => {
                trace[ti].accepted = true;
                cur = f;
                cur_score = s;
                accepted.push((cur.prog.to_text(), cur_score));
            }
            _ => break,
        }
    }
    // Full resolution: canonicalise, refit, score, judge.
    let prog = canon::canonical(&cur.prog);
    let mut full = Fitter::new(setup_full.clone(), target_full.clone(), target_spp, &prog);
    let levels = [Level { sigma: 1.0, lambda: 0.0, iters: 3 }, Level { sigma: 0.0, lambda: 0.0, iters: 8 }];
    let fit = full.fit(&prog, &levels).unwrap_or_else(|_| full.fit(&cur.prog, &levels).expect("the search's winner refits"));
    iters += fit.iters;
    evals += fit.evals + 2;
    let score = mdl::score(&fit);
    let judge = mdl::judge(&fit);
    Recovery {
        program: canon::canonical(&fit.prog),
        fit,
        score,
        judge,
        trace,
        accepted,
        iters,
        evals,
        seconds: t0.elapsed().as_secs_f64(),
    }
}

/// A fitted program that collapsed onto a simpler one — a blend, slot,
/// boss or counterbore shrunk to its limit — is that simpler program with
/// an extra parameter the Laplace term undercharges (the optimum sits on a
/// bound, where the Gaussian approximation does not hold). Refused; the
/// simpler program is always on offer as an edit.
pub fn degenerate(p: &Program) -> Option<&'static str> {
    match p.corner {
        Corner::Fillet(r) | Corner::Chamfer(r) if r < 0.8 => return Some("corner blend under 0.8 mm"),
        _ => {}
    }
    for f in &p.features {
        match *f {
            Feature::Slot { len, .. } if len < 1.5 => return Some("slot shorter than 1.5 mm"),
            Feature::Boss { bh, .. } if bh < 0.8 => return Some("boss under 0.8 mm"),
            Feature::Counterbore { r, cr, cd, .. } if r < 0.8 || cr - r < 0.4 || cd < 0.8 || cd > p.h - 1.2 => {
                return Some("counterbore without a step");
            }
            _ => {}
        }
    }
    None
}

fn round_text(p: &Program) -> String {
    p.params().iter().map(|x| format!("{:.0}", x)).collect::<Vec<_>>().join(",")
}
