//! The v0 program family: a typed vcad DSL for mechanical brackets and
//! blocks.
//!
//! A [`Program`] is a base plate (a sketch extrusion, `w × d × h`, centred
//! on the origin in `x, y`, standing on `z = 0`) whose four vertical corner
//! edges are sharp, filleted or chamfered, then an ordered list of
//! [`Feature`]s on its top face: through-holes, counterbores, through-slots,
//! bosses (a second extrusion, unioned), and linear and circular patterns of
//! holes. Cuts are booleans (`difference`), bosses `union`. Everything is
//! millimetres and degrees, vcad's units.
//!
//! The discrete structure is the corner kind, the feature kinds, slot
//! orientation and pattern counts; the continuous parameters are what
//! [`Program::params`] flattens, each with a name, a prior range and a
//! [`Unit`].
//!
//! Which edges a blend touches is fixed for v0 (the four vertical corners):
//! vcad 0.10's `Solid::fillet`/`chamfer` act on every edge and refuse faces
//! with inner loops, and `edge_blend` by direction lofts the blend, which
//! later booleans do not survive reliably. Rounding the sketch is exact and
//! keeps the BRep analytic.

use serde::{Deserialize, Serialize};
use vcad_kernel::Solid;
use vcad_kernel::vcad_kernel_math as km;
use vcad_kernel::vcad_kernel_sketch::{SketchProfile, SketchSegment};

/// Segments per full circle for cylinders and arcs.
pub const SEGMENTS: u32 = 48;

/// How the base's vertical corner edges are finished.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Corner {
    /// Sharp.
    Sharp,
    /// Rounded with radius `r`.
    Fillet(f64),
    /// Cut at 45° with leg `c`.
    Chamfer(f64),
}

/// Slot direction on the top face.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Axis {
    /// Along `x`.
    X,
    /// Along `y`.
    Y,
}

/// One feature on the top face of the base.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Feature {
    /// A through-hole of radius `r` at `(x, y)`.
    Hole { x: f64, y: f64, r: f64 },
    /// A through-hole of radius `r` with a counterbore of radius `cr` and
    /// depth `cd` from the top.
    Counterbore { x: f64, y: f64, r: f64, cr: f64, cd: f64 },
    /// A through-slot: two semicircles of radius `r` whose centres are `len`
    /// apart along `axis`, centred at `(x, y)`.
    Slot { x: f64, y: f64, len: f64, r: f64, axis: Axis },
    /// A cylindrical boss of radius `r` standing `bh` above the top face.
    Boss { x: f64, y: f64, r: f64, bh: f64 },
    /// `n` through-holes of radius `r`, the first at `(x, y)`, each next one
    /// `(dx, dy)` further.
    LinearHoles { x: f64, y: f64, dx: f64, dy: f64, r: f64, n: usize },
    /// `n` through-holes of radius `r` on a circle of radius `rad` about
    /// `(x, y)`, the first at angle `phase` degrees.
    CircularHoles { x: f64, y: f64, rad: f64, phase: f64, r: f64, n: usize },
}

/// A program: the base and its features.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Program {
    /// Base width (`x`).
    pub w: f64,
    /// Base depth (`y`).
    pub d: f64,
    /// Base height (`z`).
    pub h: f64,
    /// Corner finish.
    pub corner: Corner,
    /// Features, applied in order.
    pub features: Vec<Feature>,
}

fn permute(v: &mut Vec<usize>, k: usize, out: &mut Vec<Vec<usize>>) {
    if k == v.len() {
        out.push(v.clone());
        return;
    }
    for i in k..v.len() {
        v.swap(k, i);
        permute(v, k + 1, out);
        v.swap(k, i);
    }
}

/// What a parameter measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Unit {
    /// Millimetres.
    Mm,
    /// Degrees.
    Deg,
}

/// One continuous parameter's description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamInfo {
    /// `feature.name`, e.g. `f1.r`, or `base.w`.
    pub name: String,
    /// Units.
    pub unit: Unit,
    /// Prior range width, for the MDL parameter cost.
    pub range: f64,
    /// Lower bound kept by the fit.
    pub lo: f64,
    /// Upper bound kept by the fit.
    pub hi: f64,
}

impl Feature {
    /// Kind name, for the grammar and the report.
    pub fn kind(&self) -> &'static str {
        match self {
            Feature::Hole { .. } => "hole",
            Feature::Counterbore { .. } => "counterbore",
            Feature::Slot { .. } => "slot",
            Feature::Boss { .. } => "boss",
            Feature::LinearHoles { .. } => "linear_holes",
            Feature::CircularHoles { .. } => "circular_holes",
        }
    }

    /// The discrete part of this feature: kind, slot axis, pattern count.
    pub fn signature(&self) -> String {
        match self {
            Feature::Slot { axis, .. } => format!("slot{axis:?}"),
            Feature::LinearHoles { n, .. } => format!("linear_holes×{n}"),
            Feature::CircularHoles { n, .. } => format!("circular_holes×{n}"),
            f => f.kind().into(),
        }
    }

    /// Continuous parameters, in order.
    pub fn params(&self) -> Vec<f64> {
        match *self {
            Feature::Hole { x, y, r } => vec![x, y, r],
            Feature::Counterbore { x, y, r, cr, cd } => vec![x, y, r, cr, cd],
            Feature::Slot { x, y, len, r, .. } => vec![x, y, len, r],
            Feature::Boss { x, y, r, bh } => vec![x, y, r, bh],
            Feature::LinearHoles { x, y, dx, dy, r, .. } => vec![x, y, dx, dy, r],
            Feature::CircularHoles { x, y, rad, phase, r, .. } => vec![x, y, rad, phase, r],
        }
    }

    /// Parameter names and units.
    pub fn names(&self) -> Vec<(&'static str, Unit)> {
        use Unit::*;
        match self {
            Feature::Hole { .. } => vec![("x", Mm), ("y", Mm), ("r", Mm)],
            Feature::Counterbore { .. } => {
                vec![("x", Mm), ("y", Mm), ("r", Mm), ("cr", Mm), ("cd", Mm)]
            }
            Feature::Slot { .. } => vec![("x", Mm), ("y", Mm), ("len", Mm), ("r", Mm)],
            Feature::Boss { .. } => vec![("x", Mm), ("y", Mm), ("r", Mm), ("bh", Mm)],
            Feature::LinearHoles { .. } => {
                vec![("x", Mm), ("y", Mm), ("dx", Mm), ("dy", Mm), ("r", Mm)]
            }
            Feature::CircularHoles { .. } => {
                vec![("x", Mm), ("y", Mm), ("rad", Mm), ("phase", Deg), ("r", Mm)]
            }
        }
    }

    /// Set the continuous parameters from `p` (as [`Feature::params`]).
    pub fn set_params(&mut self, p: &[f64]) {
        match self {
            Feature::Hole { x, y, r } => (*x, *y, *r) = (p[0], p[1], p[2]),
            Feature::Counterbore { x, y, r, cr, cd } => {
                (*x, *y, *r, *cr, *cd) = (p[0], p[1], p[2], p[3], p[4])
            }
            Feature::Slot { x, y, len, r, .. } => (*x, *y, *len, *r) = (p[0], p[1], p[2], p[3]),
            Feature::Boss { x, y, r, bh } => (*x, *y, *r, *bh) = (p[0], p[1], p[2], p[3]),
            Feature::LinearHoles { x, y, dx, dy, r, .. } => {
                (*x, *y, *dx, *dy, *r) = (p[0], p[1], p[2], p[3], p[4])
            }
            Feature::CircularHoles { x, y, rad, phase, r, .. } => {
                (*x, *y, *rad, *phase, *r) = (p[0], p[1], p[2], p[3], p[4])
            }
        }
    }

    /// The holes a hole-like feature expands to, `(x, y, r)`; empty for
    /// slots and bosses.
    pub fn holes(&self) -> Vec<(f64, f64, f64)> {
        match *self {
            Feature::Hole { x, y, r } | Feature::Counterbore { x, y, r, .. } => vec![(x, y, r)],
            Feature::LinearHoles { x, y, dx, dy, r, n } => (0..n)
                .map(|i| (x + i as f64 * dx, y + i as f64 * dy, r))
                .collect(),
            Feature::CircularHoles { x, y, rad, phase, r, n } => (0..n)
                .map(|i| {
                    let a = (phase + 360.0 * i as f64 / n as f64).to_radians();
                    (x + rad * a.cos(), y + rad * a.sin(), r)
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Footprint discs on the top face `(x, y, radius)` the feature occupies,
    /// for validity (no overlaps, inside the base).
    pub fn footprint(&self) -> Vec<(f64, f64, f64)> {
        match *self {
            Feature::Counterbore { x, y, cr, .. } => vec![(x, y, cr)],
            Feature::Slot { x, y, len, r, axis } => {
                let (ux, uy) = if axis == Axis::X { (1.0, 0.0) } else { (0.0, 1.0) };
                let k = (len / (2.0 * r)).ceil() as usize + 1;
                (0..=k)
                    .map(|i| {
                        let t = -len / 2.0 + len * i as f64 / k as f64;
                        (x + ux * t, y + uy * t, r)
                    })
                    .collect()
            }
            Feature::Boss { x, y, r, .. } => vec![(x, y, r)],
            _ => self.holes(),
        }
    }
}

impl Program {
    /// Every continuous parameter, base first.
    pub fn params(&self) -> Vec<f64> {
        let mut p = vec![self.w, self.d, self.h];
        match self.corner {
            Corner::Sharp => {}
            Corner::Fillet(r) | Corner::Chamfer(r) => p.push(r),
        }
        for f in &self.features {
            p.extend(f.params());
        }
        p
    }

    /// Set every continuous parameter from `p` (as [`Program::params`]).
    pub fn set_params(&mut self, p: &[f64]) {
        assert_eq!(p.len(), self.params().len(), "parameter count");
        (self.w, self.d, self.h) = (p[0], p[1], p[2]);
        let mut i = 3;
        match &mut self.corner {
            Corner::Sharp => {}
            Corner::Fillet(r) | Corner::Chamfer(r) => {
                *r = p[3];
                i = 4;
            }
        }
        for f in &mut self.features {
            let n = f.params().len();
            f.set_params(&p[i..i + n]);
            i += n;
        }
    }

    /// Parameters whose image derivative is mostly not on primary edges:
    /// a counterbore's depth and its inner hole's radius change what the
    /// bore's floor and the hole below it see of the sky (occlusion — the
    /// secondary visibility and interior terms), which the primary edge
    /// term leaves out; its column there has the wrong sign. A boss's
    /// shadow and sky occlusion on the plate move with all four of its
    /// parameters. Their columns are forward differences instead
    /// ([`crate::model::Renderer::eval`]).
    pub fn occluded_params(&self) -> Vec<usize> {
        let mut out = Vec::new();
        let mut j = 3 + matches!(self.corner, Corner::Fillet(_) | Corner::Chamfer(_)) as usize;
        for f in &self.features {
            match f {
                Feature::Counterbore { .. } => out.extend([j + 2, j + 4]),
                // A boss casts a shadow and occludes the sky of the plate
                // around it; moving it moves both, and that change is larger
                // than its own edges' (the edge column is half the size, and
                // LM steps the boss away from the target with it).
                Feature::Boss { .. } => out.extend([j, j + 1, j + 2, j + 3]),
                _ => {}
            }
            j += f.params().len();
        }
        out
    }

    /// Descriptions of [`Program::params`], with fit bounds that keep the
    /// program valid where that is a box constraint.
    pub fn param_info(&self) -> Vec<ParamInfo> {
        let mut v = Vec::new();
        let mm = |name: String, range: f64, lo: f64, hi: f64| ParamInfo {
            name,
            unit: Unit::Mm,
            range,
            lo,
            hi,
        };
        v.push(mm("base.w".into(), 80.0, 10.0, 200.0));
        v.push(mm("base.d".into(), 80.0, 10.0, 200.0));
        v.push(mm("base.h".into(), 30.0, 2.0, 60.0));
        match self.corner {
            Corner::Sharp => {}
            Corner::Fillet(_) => v.push(mm("base.fillet".into(), 15.0, 0.5, 30.0)),
            Corner::Chamfer(_) => v.push(mm("base.chamfer".into(), 15.0, 0.5, 30.0)),
        }
        for (k, f) in self.features.iter().enumerate() {
            for (name, unit) in f.names() {
                let (range, lo, hi) = match (f, name) {
                    (_, "x" | "y") => (80.0, -200.0, 200.0),
                    (_, "phase") => (360.0, -720.0, 720.0),
                    (_, "dx" | "dy") => (60.0, -100.0, 100.0),
                    (_, "rad") => (40.0, 2.0, 100.0),
                    (_, "cd") => (self.h, 0.3, 60.0),
                    (_, "bh") => (30.0, 0.5, 60.0),
                    (_, "len") => (40.0, 0.5, 120.0),
                    _ => (15.0, 0.4, 40.0),
                };
                v.push(ParamInfo {
                    name: format!("f{k}.{name}"),
                    unit,
                    range,
                    lo,
                    hi,
                });
            }
        }
        v
    }

    /// Whether the program describes a part the builder makes faithfully:
    /// positive sizes, blends that fit, counterbores wider than and
    /// shallower than their holes, features inside the base with a margin
    /// and not touching each other.
    pub fn valid(&self) -> bool {
        let (w, d, h) = (self.w, self.d, self.h);
        if !(w > 4.0 && d > 4.0 && h > 1.0) {
            return false;
        }
        let blend = match self.corner {
            Corner::Sharp => 0.0,
            Corner::Fillet(r) | Corner::Chamfer(r) => r,
        };
        if blend < 0.0 || (blend > 0.0 && (blend < 0.5 || blend > 0.45 * w.min(d))) {
            return false;
        }
        let mut discs = Vec::new();
        for f in &self.features {
            match *f {
                Feature::Counterbore { r, cr, cd, .. } => {
                    if !(cr > r + 0.3 && cd > 0.5 && cd < h - 1.0) {
                        return false;
                    }
                }
                Feature::Slot { len, r, .. } => {
                    if !(len > 1.0 && r > 0.5) {
                        return false;
                    }
                }
                Feature::Boss { r, bh, .. } => {
                    if !(r > 0.8 && bh > 0.5) {
                        return false;
                    }
                }
                Feature::LinearHoles { n, dx, dy, .. } => {
                    if n < 2 || dx.hypot(dy) < 1.0 {
                        return false;
                    }
                }
                Feature::CircularHoles { n, rad, .. } => {
                    if n < 3 || rad < 1.0 {
                        return false;
                    }
                }
                Feature::Hole { .. } => {}
            }
            let fp = f.footprint();
            if fp.iter().any(|&(_, _, r)| r < 0.5) {
                return false;
            }
            // Inside the plate, clear of the edges and the corner blend.
            let margin = 1.5;
            for &(x, y, r) in &fp {
                let (ex, ey) = (w / 2.0 - margin - r, d / 2.0 - margin - r);
                if x.abs() > ex || y.abs() > ey {
                    return false;
                }
                let (cx, cy) = (w / 2.0 - blend, d / 2.0 - blend);
                if x.abs() > cx && y.abs() > cy {
                    let dd = (x.abs() - cx).hypot(y.abs() - cy);
                    if dd + r + margin > blend.max(0.0) * 0.7 {
                        return false;
                    }
                }
            }
            // Pattern members apart from each other.
            for (i, a) in fp.iter().enumerate() {
                for b in &fp[i + 1..] {
                    let same_slot = matches!(f, Feature::Slot { .. });
                    if !same_slot && (a.0 - b.0).hypot(a.1 - b.1) < a.2 + b.2 + margin {
                        return false;
                    }
                }
            }
            discs.push(fp);
        }
        for i in 0..discs.len() {
            for j in i + 1..discs.len() {
                for a in &discs[i] {
                    for b in &discs[j] {
                        if (a.0 - b.0).hypot(a.1 - b.1) < a.2 + b.2 + 1.5 {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// The base outline, counter-clockwise, as sketch segments.
    fn outline(&self) -> Vec<SketchSegment> {
        let (a, b) = (self.w / 2.0, self.d / 2.0);
        let p = km::Point2::new;
        let line = |s: km::Point2, e: km::Point2| SketchSegment::Line { start: s, end: e };
        match self.corner {
            Corner::Sharp => vec![
                line(p(-a, -b), p(a, -b)),
                line(p(a, -b), p(a, b)),
                line(p(a, b), p(-a, b)),
                line(p(-a, b), p(-a, -b)),
            ],
            Corner::Chamfer(c) => vec![
                line(p(-a + c, -b), p(a - c, -b)),
                line(p(a - c, -b), p(a, -b + c)),
                line(p(a, -b + c), p(a, b - c)),
                line(p(a, b - c), p(a - c, b)),
                line(p(a - c, b), p(-a + c, b)),
                line(p(-a + c, b), p(-a, b - c)),
                line(p(-a, b - c), p(-a, -b + c)),
                line(p(-a, -b + c), p(-a + c, -b)),
            ],
            Corner::Fillet(r) => {
                let arc = |s: km::Point2, e: km::Point2, c: km::Point2| SketchSegment::Arc {
                    start: s,
                    end: e,
                    center: c,
                    ccw: true,
                };
                vec![
                    line(p(-a + r, -b), p(a - r, -b)),
                    arc(p(a - r, -b), p(a, -b + r), p(a - r, -b + r)),
                    line(p(a, -b + r), p(a, b - r)),
                    arc(p(a, b - r), p(a - r, b), p(a - r, b - r)),
                    line(p(a - r, b), p(-a + r, b)),
                    arc(p(-a + r, b), p(-a, b - r), p(-a + r, b - r)),
                    line(p(-a, b - r), p(-a, -b + r)),
                    arc(p(-a, -b + r), p(-a + r, -b), p(-a + r, -b + r)),
                ]
            }
        }
    }

    /// The vcad solid, checked against the program's closed-form volume.
    ///
    /// vcad 0.10's booleans occasionally drop a cutter's faces depending on
    /// the order the cuts are made in (a slot cut after three holes can vanish
    /// while the reverse order is exact). The features commute — every cut is
    /// through or from the top, the boss is disjoint — so the builder tries
    /// orders until the volume agrees to 0.3 %, and fails closed otherwise.
    pub fn build(&self) -> Result<Solid, String> {
        let n = self.features.len();
        let rank = |f: &Feature| match f {
            Feature::Slot { .. } => 0,
            Feature::Counterbore { .. } => 1,
            Feature::Boss { .. } => 3,
            _ => 2,
        };
        let mut first: Vec<usize> = (0..n).collect();
        first.sort_by_key(|&i| rank(&self.features[i]));
        let mut orders = vec![first.clone(), (0..n).collect(), first.iter().rev().copied().collect()];
        if n <= 4 {
            let mut all = Vec::new();
            permute(&mut (0..n).collect::<Vec<_>>(), 0, &mut all);
            orders.extend(all);
        }
        let want = self.volume();
        let mut last = String::new();
        for o in orders {
            match self.build_in(&o) {
                Ok(s) => {
                    let v = s.volume();
                    if (v - want).abs() <= 3e-3 * want {
                        return Ok(s);
                    }
                    last = format!("volume {v:.1} vs {want:.1}");
                }
                Err(e) => last = e,
            }
        }
        Err(format!("no cut order builds it exactly: {last}"))
    }

    fn build_in(&self, order: &[usize]) -> Result<Solid, String> {
        let profile = SketchProfile::new(
            km::Point3::new(0.0, 0.0, 0.0),
            km::Vec3::new(1.0, 0.0, 0.0),
            km::Vec3::new(0.0, 1.0, 0.0),
            self.outline(),
        )
        .map_err(|e| format!("outline: {e:?}"))?;
        let mut s = Solid::extrude(profile, km::Vec3::new(0.0, 0.0, self.h))
            .map_err(|e| format!("extrude: {e:?}"))?;
        let h = self.h;
        let through = |x: f64, y: f64, r: f64| Solid::cylinder(r, h + 2.0, SEGMENTS).translate(x, y, -1.0);
        for f in order.iter().map(|&i| &self.features[i]) {
            s = match *f {
                Feature::Counterbore { x, y, r, cr, cd } => {
                    let bore = Solid::cylinder(cr, cd + 1.0, SEGMENTS).translate(x, y, h - cd);
                    s.try_difference(&through(x, y, r))
                        .and_then(|s| s.try_difference(&bore))
                        .map_err(|e| format!("counterbore: {e:?}"))?
                }
                Feature::Slot { x, y, len, r, axis } => {
                    // Three plain through-cuts — the box between the centres,
                    // then each end's cylinder — like holes. (A stadium sketch
                    // extruded as one cutter, or the three unioned first, lose
                    // faces in vcad 0.10's booleans for some sizes.)
                    let (ux, uy) = if axis == Axis::X { (1.0, 0.0) } else { (0.0, 1.0) };
                    let (bx, by) = if axis == Axis::X { (len, 2.0 * r) } else { (2.0 * r, len) };
                    let bar = Solid::cube(bx, by, h + 2.0).translate(x - bx / 2.0, y - by / 2.0, -1.0);
                    s.try_difference(&bar)
                        .and_then(|s| s.try_difference(&through(x - ux * len / 2.0, y - uy * len / 2.0, r)))
                        .and_then(|s| s.try_difference(&through(x + ux * len / 2.0, y + uy * len / 2.0, r)))
                        .map_err(|e| format!("slot: {e:?}"))?
                }
                Feature::Boss { x, y, r, bh } => {
                    let boss = Solid::cylinder(r, bh + h / 2.0, SEGMENTS).translate(x, y, h / 2.0);
                    s.try_union(&boss).map_err(|e| format!("boss: {e:?}"))?
                }
                _ => {
                    let mut s = s;
                    for (x, y, r) in f.holes() {
                        s = s
                            .try_difference(&through(x, y, r))
                            .map_err(|e| format!("hole: {e:?}"))?;
                    }
                    s
                }
            };
        }
        if s.as_brep().is_none() {
            return Err("result is not a BRep".into());
        }
        Ok(s)
    }

    /// The part's volume from the program alone (cylinders as their
    /// `SEGMENTS`-gons, arcs likewise), for checking the builder.
    pub fn volume(&self) -> f64 {
        let poly = |r: f64| 0.5 * SEGMENTS as f64 * r * r * (std::f64::consts::TAU / SEGMENTS as f64).sin();
        let mut a = self.w * self.d;
        match self.corner {
            Corner::Sharp => {}
            Corner::Chamfer(c) => a -= 2.0 * c * c,
            // A quarter of the 48-gon per corner (the tessellation's chords).
            Corner::Fillet(r) => a -= 4.0 * r * r - poly(r),
        }
        let mut v = a * self.h;
        for f in &self.features {
            match *f {
                Feature::Counterbore { r, cr, cd, .. } => {
                    v -= poly(r) * (self.h - cd) + poly(cr) * cd;
                }
                Feature::Slot { len, r, .. } => v -= (poly(r) + 2.0 * r * len) * self.h,
                Feature::Boss { r, bh, .. } => v += poly(r) * bh,
                _ => {
                    for (_, _, r) in f.holes() {
                        v -= poly(r) * self.h;
                    }
                }
            }
        }
        v
    }

    /// The program as text: one line per statement, the DSL's surface form.
    pub fn to_text(&self) -> String {
        let mut s = format!("base {:.3} {:.3} {:.3}", self.w, self.d, self.h);
        match self.corner {
            Corner::Sharp => {}
            Corner::Fillet(r) => s += &format!(" fillet {r:.3}"),
            Corner::Chamfer(c) => s += &format!(" chamfer {c:.3}"),
        }
        for f in &self.features {
            s += "\n";
            s += &match *f {
                Feature::Hole { x, y, r } => format!("hole {x:.3} {y:.3} {r:.3}"),
                Feature::Counterbore { x, y, r, cr, cd } => {
                    format!("counterbore {x:.3} {y:.3} {r:.3} {cr:.3} {cd:.3}")
                }
                Feature::Slot { x, y, len, r, axis } => {
                    format!("slot{} {x:.3} {y:.3} {len:.3} {r:.3}", if axis == Axis::X { "x" } else { "y" })
                }
                Feature::Boss { x, y, r, bh } => format!("boss {x:.3} {y:.3} {r:.3} {bh:.3}"),
                Feature::LinearHoles { x, y, dx, dy, r, n } => {
                    format!("linear {n} {x:.3} {y:.3} {dx:.3} {dy:.3} {r:.3}")
                }
                Feature::CircularHoles { x, y, rad, phase, r, n } => {
                    format!("circular {n} {x:.3} {y:.3} {rad:.3} {phase:.3} {r:.3}")
                }
            };
        }
        s
    }

    /// Parse [`Program::to_text`]'s form.
    pub fn from_text(t: &str) -> Result<Self, String> {
        let mut lines = t.lines().filter(|l| !l.trim().is_empty());
        let head: Vec<&str> = lines.next().ok_or("empty")?.split_whitespace().collect();
        let num = |s: &str| s.parse::<f64>().map_err(|e| format!("{s}: {e}"));
        if head.first() != Some(&"base") || head.len() < 4 {
            return Err("first line: base w d h".into());
        }
        let corner = match head.get(4) {
            None => Corner::Sharp,
            Some(&"fillet") => Corner::Fillet(num(head.get(5).ok_or("fillet r")?)?),
            Some(&"chamfer") => Corner::Chamfer(num(head.get(5).ok_or("chamfer c")?)?),
            Some(k) => return Err(format!("corner {k}")),
        };
        let mut p = Program {
            w: num(head[1])?,
            d: num(head[2])?,
            h: num(head[3])?,
            corner,
            features: Vec::new(),
        };
        for l in lines {
            let a: Vec<&str> = l.split_whitespace().collect();
            let v = |i: usize| -> Result<f64, String> { num(a.get(i).ok_or(format!("{l}: arity"))?) };
            let n = |i: usize| -> Result<usize, String> {
                a.get(i).ok_or(format!("{l}: arity"))?.parse::<usize>().map_err(|e| e.to_string())
            };
            p.features.push(match a[0] {
                "hole" => Feature::Hole { x: v(1)?, y: v(2)?, r: v(3)? },
                "counterbore" => Feature::Counterbore { x: v(1)?, y: v(2)?, r: v(3)?, cr: v(4)?, cd: v(5)? },
                "slotx" | "sloty" => Feature::Slot {
                    x: v(1)?,
                    y: v(2)?,
                    len: v(3)?,
                    r: v(4)?,
                    axis: if a[0] == "slotx" { Axis::X } else { Axis::Y },
                },
                "boss" => Feature::Boss { x: v(1)?, y: v(2)?, r: v(3)?, bh: v(4)? },
                "linear" => Feature::LinearHoles { n: n(1)?, x: v(2)?, y: v(3)?, dx: v(4)?, dy: v(5)?, r: v(6)? },
                "circular" => Feature::CircularHoles { n: n(1)?, x: v(2)?, y: v(3)?, rad: v(4)?, phase: v(5)?, r: v(6)? },
                k => return Err(format!("unknown statement {k}")),
            });
        }
        Ok(p)
    }

    /// The discrete structure: corner kind and feature signatures in order.
    pub fn signature(&self) -> String {
        let c = match self.corner {
            Corner::Sharp => "sharp",
            Corner::Fillet(_) => "fillet",
            Corner::Chamfer(_) => "chamfer",
        };
        let f: Vec<String> = self.features.iter().map(Feature::signature).collect();
        format!("{c}[{}]", f.join(","))
    }
}
