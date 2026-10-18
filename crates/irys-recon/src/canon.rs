//! A normal form for the DSL, so "topology correct" means something.
//!
//! [`canonical`] (no e-graph: the family is small enough for rewrite rules
//! applied to a fixed point):
//!
//! 1. Patterns are normalised: a linear pattern runs in `+x` (or `+y` when
//!    vertical), a circular pattern's phase is reduced into `[0, 360/n)`.
//! 2. Equivalent holes are merged: plain holes and the members of hole
//!    patterns are grouped by radius (within [`R_TOL`]); a group of three or
//!    more on a circle at equal angles becomes one circular pattern, a group
//!    on a line at equal spacing one linear pattern (two holes always are),
//!    otherwise the holes stay single.
//! 3. Features are sorted by kind, then position.
//!
//! Two programs have the same topology when their canonical signatures
//! ([`Program::signature`]) agree.

use crate::dsl::{Feature, Program};

/// Radii closer than this are the same hole size.
pub const R_TOL: f64 = 0.35;
/// Positions closer than this are the same place.
pub const P_TOL: f64 = 0.6;

fn order(f: &Feature) -> usize {
    match f {
        Feature::Hole { .. } => 0,
        Feature::LinearHoles { .. } => 1,
        Feature::CircularHoles { .. } => 2,
        Feature::Counterbore { .. } => 3,
        Feature::Slot { .. } => 4,
        Feature::Boss { .. } => 5,
    }
}

fn normalise(f: &Feature) -> Feature {
    match *f {
        Feature::LinearHoles { x, y, dx, dy, r, n } if dx < -1e-9 || (dx.abs() <= 1e-9 && dy < 0.0) => {
            let k = (n - 1) as f64;
            Feature::LinearHoles { x: x + k * dx, y: y + k * dy, dx: -dx, dy: -dy, r, n }
        }
        Feature::CircularHoles { x, y, rad, phase, r, n } => {
            let step = 360.0 / n as f64;
            Feature::CircularHoles { x, y, rad, phase: phase.rem_euclid(step), r, n }
        }
        ref f => f.clone(),
    }
}

/// Least-squares circle through points: `(cx, cy, rad, rms)`.
fn circle(p: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    // Kåsa: x² + y² = 2ax + 2by + c.
    let n = p.len() as f64;
    let (mut sxx, mut sxy, mut syy, mut sx, mut sy, mut sxz, mut syz, mut sz) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for &(x, y) in p {
        let z = x * x + y * y;
        sxx += x * x;
        sxy += x * y;
        syy += y * y;
        sx += x;
        sy += y;
        sxz += x * z;
        syz += y * z;
        sz += z;
    }
    let a = [[2.0 * sxx, 2.0 * sxy, sx], [2.0 * sxy, 2.0 * syy, sy], [2.0 * sx, 2.0 * sy, n]];
    let b = [sxz, syz, sz];
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(a);
    if d.abs() < 1e-12 {
        return (0.0, 0.0, 0.0, f64::INFINITY);
    }
    let col = |k: usize| {
        let mut m = a;
        for i in 0..3 {
            m[i][k] = b[i];
        }
        det(m) / d
    };
    let (cx, cy, c) = (col(0), col(1), col(2));
    let rad = (c + cx * cx + cy * cy).max(0.0).sqrt();
    let rms = (p.iter().map(|&(x, y)| ((x - cx).hypot(y - cy) - rad).powi(2)).sum::<f64>() / n).sqrt();
    (cx, cy, rad, rms)
}

/// A circular pattern through `holes`, if they are one.
pub fn as_circular(holes: &[(f64, f64, f64)]) -> Option<Feature> {
    let n = holes.len();
    if n < 3 {
        return None;
    }
    let pts: Vec<(f64, f64)> = holes.iter().map(|h| (h.0, h.1)).collect();
    let (cx, cy, rad, rms) = circle(&pts);
    if !(rms < P_TOL && rad > 1.0) {
        return None;
    }
    let mut ang: Vec<f64> = pts.iter().map(|&(x, y)| (y - cy).atan2(x - cx).to_degrees().rem_euclid(360.0)).collect();
    ang.sort_by(f64::total_cmp);
    let step = 360.0 / n as f64;
    // Equal angles: every gap within 3° of 360/n.
    for k in 0..n {
        let gap = if k + 1 < n { ang[k + 1] - ang[k] } else { ang[0] + 360.0 - ang[k] };
        if (gap - step).abs() > 3.0 {
            return None;
        }
    }
    // Phase: the circular mean of the angles mod step.
    let (mut s, mut c) = (0.0, 0.0);
    for a in &ang {
        let t = (a.rem_euclid(step) / step) * std::f64::consts::TAU;
        s += t.sin();
        c += t.cos();
    }
    let phase = (s.atan2(c) / std::f64::consts::TAU * step).rem_euclid(step);
    let r = holes.iter().map(|h| h.2).sum::<f64>() / n as f64;
    Some(Feature::CircularHoles { x: cx, y: cy, rad, phase, r, n })
}

/// A linear pattern through `holes`, if they are one.
pub fn as_linear(holes: &[(f64, f64, f64)]) -> Option<Feature> {
    let n = holes.len();
    if n < 2 {
        return None;
    }
    // Principal direction, then sort along it.
    let (mx, my) = (holes.iter().map(|h| h.0).sum::<f64>() / n as f64, holes.iter().map(|h| h.1).sum::<f64>() / n as f64);
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for h in holes {
        let (dx, dy) = (h.0 - mx, h.1 - my);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    let th = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let (ux, uy) = (th.cos(), th.sin());
    let mut s: Vec<(f64, (f64, f64, f64))> = holes.iter().map(|h| ((h.0 - mx) * ux + (h.1 - my) * uy, *h)).collect();
    s.sort_by(|a, b| a.0.total_cmp(&b.0));
    let k = (n - 1) as f64;
    let (first, last) = (s[0].1, s[n - 1].1);
    let (dx, dy) = ((last.0 - first.0) / k, (last.1 - first.1) / k);
    for (i, (_, h)) in s.iter().enumerate() {
        let (ex, ey) = (first.0 + i as f64 * dx, first.1 + i as f64 * dy);
        if (h.0 - ex).hypot(h.1 - ey) > P_TOL {
            return None;
        }
    }
    // Least squares for the start and the step.
    let (mut a, mut b, mut c, mut d) = (0.0, 0.0, 0.0, 0.0);
    for (i, (_, h)) in s.iter().enumerate() {
        a += i as f64;
        b += (i * i) as f64;
        c += h.0;
        d += i as f64 * h.0;
    }
    let nn = n as f64;
    let det = nn * b - a * a;
    let (x0, sx) = ((b * c - a * d) / det, (nn * d - a * c) / det);
    let (mut c2, mut d2) = (0.0, 0.0);
    for (i, (_, h)) in s.iter().enumerate() {
        c2 += h.1;
        d2 += i as f64 * h.1;
    }
    let (y0, sy) = ((b * c2 - a * d2) / det, (nn * d2 - a * c2) / det);
    let r = holes.iter().map(|h| h.2).sum::<f64>() / nn;
    Some(normalise(&Feature::LinearHoles { x: x0, y: y0, dx: sx, dy: sy, r, n }))
}

/// The normal form of `p`.
pub fn canonical(p: &Program) -> Program {
    let mut keep: Vec<Feature> = Vec::new();
    let mut holes: Vec<(f64, f64, f64)> = Vec::new();
    for f in &p.features {
        match f {
            Feature::Hole { .. } | Feature::LinearHoles { .. } | Feature::CircularHoles { .. } => holes.extend(f.holes()),
            f => keep.push(normalise(f)),
        }
    }
    // Group by radius (single linkage within R_TOL).
    holes.sort_by(|a, b| a.2.total_cmp(&b.2));
    let mut groups: Vec<Vec<(f64, f64, f64)>> = Vec::new();
    for h in holes {
        match groups.last_mut() {
            Some(g) if (h.2 - g.last().unwrap().2).abs() < R_TOL => g.push(h),
            _ => groups.push(vec![h]),
        }
    }
    for g in groups {
        if let Some(c) = as_circular(&g) {
            keep.push(c);
        } else if let Some(l) = as_linear(&g) {
            keep.push(l);
        } else {
            keep.extend(g.into_iter().map(|(x, y, r)| Feature::Hole { x, y, r }));
        }
    }
    keep.sort_by(|a, b| {
        let (pa, pb) = (a.params(), b.params());
        order(a).cmp(&order(b)).then(pa[0].total_cmp(&pb[0])).then(pa[1].total_cmp(&pb[1]))
    });
    Program { features: keep, ..p.clone() }
}

/// Whether `a` and `b` have the same canonical structure.
pub fn same_topology(a: &Program, b: &Program) -> bool {
    canonical(a).signature() == canonical(b).signature()
}

/// Per-parameter absolute errors of `rec` against `truth`, in millimetres
/// (a circular pattern's phase as arc length at its radius), matching
/// features of the same signature by position; `None` when the topology
/// differs.
pub fn errors(truth: &Program, rec: &Program) -> Option<Vec<(String, f64)>> {
    let (t, r) = (canonical(truth), canonical(rec));
    if t.signature() != r.signature() {
        return None;
    }
    let mut out = Vec::new();
    for (k, (a, b)) in [t.w, t.d, t.h].iter().zip([r.w, r.d, r.h]).enumerate() {
        out.push((["base.w", "base.d", "base.h"][k].to_string(), (a - b).abs()));
    }
    use crate::dsl::Corner::*;
    match (t.corner, r.corner) {
        (Fillet(a), Fillet(b)) => out.push(("base.fillet".into(), (a - b).abs())),
        (Chamfer(a), Chamfer(b)) => out.push(("base.chamfer".into(), (a - b).abs())),
        _ => {}
    }
    // Greedy match by signature and position.
    let mut used = vec![false; r.features.len()];
    for (i, ft) in t.features.iter().enumerate() {
        let pt = ft.params();
        let best = r
            .features
            .iter()
            .enumerate()
            .filter(|(j, fr)| !used[*j] && fr.signature() == ft.signature())
            .min_by(|(_, a), (_, b)| {
                let (pa, pb) = (a.params(), b.params());
                (pa[0] - pt[0]).hypot(pa[1] - pt[1]).total_cmp(&(pb[0] - pt[0]).hypot(pb[1] - pt[1]))
            })?;
        used[best.0] = true;
        let pr = best.1.params();
        for (k, (name, _)) in ft.names().iter().enumerate() {
            let e = if *name == "phase" {
                let Feature::CircularHoles { n, rad, .. } = ft else { unreachable!() };
                let step = 360.0 / *n as f64;
                let mut dp = (pt[k] - pr[k]).rem_euclid(step);
                if dp > step / 2.0 {
                    dp -= step;
                }
                (dp.to_radians() * rad).abs()
            } else {
                (pt[k] - pr[k]).abs()
            };
            out.push((format!("f{i}.{name}"), e));
        }
    }
    Some(out)
}
