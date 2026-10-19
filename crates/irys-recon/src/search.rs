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
    if let Some(rig) = &setup.photo {
        return crate::photo::blobs(fit, rig);
    }
    let Ok(solid) = fit.prog.build() else { return Vec::new() };
    let mesh = crate::geom::mesh(&solid);
    let h = fit.prog.h;
    let (w, ht) = setup.film.size;
    let mut tops = Vec::new();
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
        tops.push(top);
    }
    blobs_from(&tops, (w, ht), &eyes, h)
}

/// Blobs from per-view maps of residual pixels on the top face (their
/// part-frame `(x, y)`), the eyes in the part frame, the top's height.
pub fn blobs_from(tops: &[Vec<Option<(f64, f64)>>], (w, ht): (usize, usize), eyes: &[Point3], h: f64) -> Vec<Blob> {
    let mut per_view: Vec<(Vec<(f64, f64)>, usize)> = Vec::new();
    for (vi, top) in tops.iter().enumerate() {
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
            b.raised = parallax(&cs, eyes, h);
            b
        })
        .collect();
    v.sort_by(|a, b| b.pixels.cmp(&a.pixels).then(a.x.total_cmp(&b.x)));
    v
}

/// The eyes in the part frame, per view.
pub fn eyes(fit: &Fit, setup: &Setup) -> Vec<Point3> {
    match &setup.photo {
        Some(rig) => {
            let pose = fit.prog.pose.unwrap_or([0.0; 3]);
            rig.views.iter().map(|v| crate::photo::unplace(pose, v.cam.eye())).collect()
        }
        None => setup.views.iter().map(|v| Point3::new(v.eye[0], v.eye[1], v.eye[2])).collect(),
    }
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
            } else if let Some(e) = eyes(ctx.fit, ctx.setup).first() {
                // One view (v1): no parallax, so a boss of each of a few
                // heights, its centre moved back along the view by half its
                // height (the blob is its base and its top disc, the top
                // pushed away from the eye by `bh · t`).
                let drop = e.z - p.h;
                if drop > 1.0 {
                    let t = ((x - e.x) / drop, (y - e.y) / drop);
                    for bh in [3.0, 6.0, 10.0] {
                        let (bx, by) = (x - 0.5 * bh * t.0, y - 0.5 * bh * t.1);
                        out.push(Proposal {
                            program: with(Feature::Boss { x: bx, y: by, r: (0.8 * r).max(1.0), bh }),
                            why: format!("boss of height {bh} at blob ({bx:.1}, {by:.1})"),
                            group: Some(bi),
                        });
                    }
                }
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
                    if let Some(raised) = raised_near(x, y) {
                        out.push(Proposal { program: swap(raised), why: format!("f{i} counterbore → boss by parallax"), group: None });
                    }
                    out.push(Proposal { program: swap(Feature::Boss { x, y, r: cr, bh: 3.0 }), why: format!("f{i} counterbore → boss"), group: None });
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
    let p = Program { w: 1.0, d: 1.0, h: 1.0, corner: Corner::Sharp, features: vec![f], pose: None };
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
    /// States kept per round (v1: 2; v0 was 1).
    pub beam: usize,
    /// Distinct programs refitted at full resolution before the verdict
    /// (v1: 4; v0 refitted the search's winner alone).
    pub finalists: usize,
    /// The noise variance at (search, full) resolution, when known (the
    /// photo model is deterministic, so it cannot be measured from two
    /// seeds of itself; [`crate::v1`] measures it from the residual).
    pub sigma2: Option<(f64, f64)>,
    /// Huber threshold in σ (v1), or Gaussian (`None`, v0).
    pub huber: Option<f64>,
    /// The photo model's linear appearance (SH, key, albedo) is refitted
    /// on the leading state after every round.
    pub refit_look: bool,
    /// Start from this program instead of the silhouettes' box.
    pub base: Option<Program>,
}

impl Default for Options {
    fn default() -> Self {
        Self { workers: 8, rounds: 8, beam: 2, finalists: 4, sigma2: None, huber: None, refit_look: true, base: None }
    }
}

impl Options {
    /// v0's search: one state, the winner alone refitted.
    pub fn v0() -> Self {
        Self { beam: 1, finalists: 1, ..Self::default() }
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
    Program { w: best.1, d: best.2, h: best.3, corner: Corner::Sharp, features: Vec::new(), pose: None }
}

/// Recover a program from a full-resolution target. The search runs at
/// half resolution (`setup_full` halved, the target box-downsampled) with
/// a beam of [`Options::beam`] states; the best [`Options::finalists`]
/// distinct programs it saw are refitted and scored at full resolution,
/// and the lowest description wins.
pub fn recover(
    target_full: &Target,
    background_full: &[Vec<f64>],
    setup_full: &Setup,
    target_spp: usize,
    proposer: &mut dyn Proposer,
    opts: &Options,
) -> Recovery {
    let t0 = Instant::now();
    let photo = setup_full.photo.is_some();
    let (target, mut setup) = if photo {
        let mut s = setup_full.scaled(0.5);
        s.edge_samples = setup_full.edge_samples / 3;
        (target_full.half_masks(), s)
    } else {
        let target = target_full.half(background_full);
        let mut setup = setup_full.clone();
        setup.film.size = target.size;
        setup.edge_samples = setup_full.edge_samples / 3;
        setup.film.spp = (setup_full.film.spp / 2).max(4);
        (target, setup)
    };
    let base = match (&opts.base, &setup.photo) {
        (Some(b), _) => b.clone(),
        (None, Some(rig)) => crate::photo::init_base(&target, rig),
        (None, None) => init_base(&target, &setup),
    };
    let (sigma2, sigma2_full) = match opts.sigma2 {
        Some(s) => s,
        None => (Fitter::new(setup.clone(), target.clone(), target_spp, &base).sigma2, f64::NAN),
    };
    let make = |setup: &Setup| -> Vec<Fitter> {
        (0..opts.workers).map(|_| Fitter::with_sigma2(setup.clone(), target.clone(), sigma2).robust(opts.huber)).collect()
    };
    let mut fitters = make(&setup);
    let (mut iters, mut evals) = (0usize, 2usize);
    let mut cur = fitters[0].fit(&base, &fit::schedule()).expect("the base box fits");
    iters += cur.iters;
    evals += cur.evals;
    if photo && opts.refit_look {
        setup = refit_looks(&cur.prog, &setup, &target);
        fitters = make(&setup);
        cur = fitters[0].fit(&cur.prog, &fit::short()).unwrap_or(cur);
    }
    let cur_score = mdl::score(&cur);
    let mut trace = vec![Step { round: 0, why: "base box from silhouettes".into(), program: cur.prog.to_text(), score: Some(cur_score), accepted: true }];
    let mut accepted = vec![(cur.prog.to_text(), cur_score)];
    let key = |p: &Program| format!("{}|{}", canon::canonical(p).signature(), round_text(&canon::canonical(p)));
    let mut beam: Vec<(Fit, Score)> = vec![(cur, cur_score)];
    // Every fitted candidate, for the finalists: (score, round, fit).
    let mut pool: Vec<(Score, usize, Fit)> = Vec::new();
    let mut tried: std::collections::HashSet<String> = std::collections::HashSet::new();
    for round in 1..=opts.rounds {
        let mut props: Vec<Proposal> = Vec::new();
        for (bf, bs) in &beam {
            let ctx = Context { fit: bf, score: *bs, target: &target, setup: &setup, history: &trace };
            let mut ps = proposer.propose(&ctx);
            ps.retain(|p| p.program.valid() && p.program.signature() + &round_text(&p.program) != bf.prog.signature() + &round_text(&bf.prog));
            props.extend(ps);
        }
        // Drop already-tried structures-at-parameters (this round and before).
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
        let first = trace.len();
        let mut fresh: Vec<(Fit, Score, usize)> = Vec::new();
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
                    fresh.push((f, s, trace.len() - 1));
                }
                Err(e) => trace.push(Step { round, why: props[i].why.clone(), program: format!("error: {e}"), score: None, accepted: false }),
            }
        }
        // Combine: the best fitted interpretation of each piece of evidence
        // that beats the leading state on its own, all at once.
        let lead = beam[0].1.total;
        let mut by_group: std::collections::BTreeMap<usize, (f64, Feature)> = std::collections::BTreeMap::new();
        for (k, step) in trace[first..].iter().enumerate() {
            let (Some(g), Some(sc)) = (props.get(k).and_then(|p| p.group), step.score) else { continue };
            if sc.total >= lead || step.why.contains("degenerate") {
                continue;
            }
            let Ok(prog) = Program::from_text(&step.program) else { continue };
            let Some(f) = prog.features.last().cloned() else { continue };
            if by_group.get(&g).is_none_or(|(t, _)| sc.total < *t) {
                by_group.insert(g, (sc.total, f));
            }
        }
        if by_group.len() >= 2 {
            let mut q = beam[0].0.prog.clone();
            q.features.extend(by_group.values().map(|(_, f)| f.clone()));
            let why = format!("combine the best of {} blobs", by_group.len());
            if q.valid() {
                if let Ok(f) = fitters[0].fit(&q, &fit::short()) {
                    if degenerate(&f.prog).is_none() {
                        iters += f.iters;
                        evals += f.evals;
                        let s = mdl::score(&f);
                        trace.push(Step { round, why, program: f.prog.to_text(), score: Some(s), accepted: false });
                        fresh.push((f, s, trace.len() - 1));
                    }
                }
            }
        }
        for (f, s, _) in &fresh {
            pool.push((*s, round, f.clone()));
        }
        // The new beam: the best distinct programs of the old beam and the
        // fresh fits.
        let old: Vec<String> = beam.iter().map(|(f, _)| key(&f.prog)).collect();
        let mut all: Vec<(Fit, Score, Option<usize>)> = beam.drain(..).map(|(f, s)| (f, s, None)).collect();
        all.extend(fresh.into_iter().map(|(f, s, ti)| (f, s, Some(ti))));
        all.sort_by(|a, b| a.1.total.total_cmp(&b.1.total));
        let mut seen = std::collections::HashSet::new();
        for (f, s, ti) in all {
            if beam.len() == opts.beam.max(1) {
                break;
            }
            if !seen.insert(key(&f.prog)) {
                continue;
            }
            if let Some(ti) = ti {
                if beam.is_empty() {
                    trace[ti].accepted = true;
                }
            }
            beam.push((f, s));
        }
        let new: Vec<String> = beam.iter().map(|(f, _)| key(&f.prog)).collect();
        if new[0] != old[0] {
            accepted.push((beam[0].0.prog.to_text(), beam[0].1));
            if photo && opts.refit_look {
                setup = refit_looks(&beam[0].0.prog, &setup, &target);
                fitters = make(&setup);
                for b in beam.iter_mut() {
                    if let Ok(f) = fitters[0].fit(&b.0.prog, &[Level { sigma: 0.0, lambda: 0.0, iters: 0 }]) {
                        evals += 1;
                        b.1 = mdl::score(&f);
                        b.0 = f;
                    }
                }
                beam.sort_by(|a, b| a.1.total.total_cmp(&b.1.total));
            }
        }
        if new == old {
            break;
        }
    }
    // Finalists: the beam, then the best of the pool, distinct after the
    // normal form.
    let mut finalists: Vec<Program> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut ranked: Vec<(f64, Program)> = beam.iter().map(|(f, s)| (s.total - 1e-9, f.prog.clone())).collect();
    pool.sort_by(|a, b| a.0.total.total_cmp(&b.0.total));
    ranked.extend(pool.iter().map(|(s, _, f)| (s.total, f.prog.clone())));
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
    for (_, p) in ranked {
        if finalists.len() == opts.finalists.max(1) {
            break;
        }
        let c = canon::canonical(&p);
        if seen.insert(c.signature()) {
            finalists.push(c);
        }
    }
    // Full resolution: refit and score each; the photo model takes the
    // search's appearance with it.
    let mut setup_full = setup_full.clone();
    if let (Some(full), Some(half)) = (&setup_full.photo, &setup.photo) {
        let mut r = (**full).clone();
        for (a, b) in r.views.iter_mut().zip(&half.views) {
            a.look = b.look;
        }
        setup_full.photo = Some(std::sync::Arc::new(r));
    }
    let levels = [Level { sigma: 1.0, lambda: 0.0, iters: 3 }, Level { sigma: 0.0, lambda: 0.0, iters: 8 }];
    let mut best: Option<(Fit, Score)> = None;
    let mut finals = Vec::new();
    for prog in &finalists {
        let mut full = if photo {
            Fitter::with_sigma2(setup_full.clone(), target_full.clone(), sigma2_full).robust(opts.huber)
        } else {
            Fitter::new(setup_full.clone(), target_full.clone(), target_spp, prog)
        };
        let Ok(f) = full.fit(prog, &levels) else { continue };
        iters += f.iters;
        evals += f.evals + 2;
        if degenerate(&f.prog).is_some() {
            continue;
        }
        let s = mdl::score(&f);
        finals.push((f.prog.to_text(), s));
        if best.as_ref().is_none_or(|b| s.total < b.1.total) {
            best = Some((f, s));
        }
    }
    trace.push(Step { round: usize::MAX, why: format!("finalists at full resolution: {:?}", finals.iter().map(|(p, s)| (p.lines().count(), s.total.round())).collect::<Vec<_>>()), program: String::new(), score: None, accepted: false });
    let (fit, score) = best.unwrap_or_else(|| {
        let mut full = Fitter::with_sigma2(setup_full.clone(), target_full.clone(), if photo { sigma2_full } else { sigma2 }).robust(opts.huber);
        let f = full.fit(&beam[0].0.prog, &levels).expect("the search's winner refits");
        let s = mdl::score(&f);
        (f, s)
    });
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

/// The photo model's looks refitted (the linear part) on `prog`.
fn refit_looks(prog: &Program, setup: &Setup, target: &Target) -> Setup {
    let Some(rig) = &setup.photo else { return setup.clone() };
    let mut r = (**rig).clone();
    for vi in 0..r.views.len() {
        let w: Vec<f64> = (0..target.img[vi].len()).map(|p| target.w(vi, p)).collect();
        crate::photo::refit_linear(prog, &mut r, vi, &target.img[vi], &w);
    }
    let mut s = setup.clone();
    s.photo = Some(std::sync::Arc::new(r));
    s
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
