//! Random valid programs from the v0 family, seeded.

use crate::dsl::{Axis, Corner, Feature, Program};
use crate::jac::Mix;

fn range(r: &mut Mix, a: f64, b: f64) -> f64 {
    a + (b - a) * r.uniform()
}

/// Round to 0.1 mm (truth on a grid makes the report readable; the solver
/// does not know it).
fn q(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn feature(r: &mut Mix, p: &Program) -> Feature {
    let (a, b) = (p.w / 2.0, p.d / 2.0);
    let (x, y) = (q(range(r, -a, a)), q(range(r, -b, b)));
    let u = r.uniform();
    if u < 0.2 {
        Feature::Hole { x, y, r: q(range(r, 2.0, 5.0)) }
    } else if u < 0.35 {
        let hr = q(range(r, 1.8, 3.5));
        Feature::Counterbore { x, y, r: hr, cr: q(hr + range(r, 1.5, 3.5)), cd: q(range(r, 2.0, (p.h - 3.0).max(2.1))) }
    } else if u < 0.5 {
        let axis = if r.uniform() < 0.5 { Axis::X } else { Axis::Y };
        Feature::Slot { x, y, len: q(range(r, 6.0, 18.0)), r: q(range(r, 2.0, 4.0)), axis }
    } else if u < 0.65 {
        Feature::Boss { x, y, r: q(range(r, 4.0, 9.0)), bh: q(range(r, 3.0, 12.0)) }
    } else if u < 0.82 {
        let n = 2 + (r.uniform() * 3.0) as usize;
        let s = q(range(r, 9.0, 16.0));
        let (dx, dy) = if r.uniform() < 0.5 { (s, 0.0) } else { (0.0, s) };
        let (x0, y0) = (q(x - dx * (n - 1) as f64 / 2.0), q(y - dy * (n - 1) as f64 / 2.0));
        Feature::LinearHoles { x: x0, y: y0, dx, dy, r: q(range(r, 1.8, 3.5)), n }
    } else {
        let n = 3 + (r.uniform() * 4.0) as usize;
        let phase = q(range(r, 0.0, 360.0 / n as f64));
        Feature::CircularHoles { x, y, rad: q(range(r, 8.0, 16.0)), phase, r: q(range(r, 1.8, 3.0)), n }
    }
}

/// Holes of two different features with radii this close would be one
/// pattern's to a canonicaliser; the generator keeps them apart.
const DISTINCT_R: f64 = 0.6;

fn distinct_radii(p: &Program) -> bool {
    let rs: Vec<(usize, f64)> = p
        .features
        .iter()
        .enumerate()
        .flat_map(|(i, f)| f.holes().into_iter().map(move |h| (i, h.2)))
        .filter(|_| true)
        .collect();
    // Only plain holes and patterns merge; counterbores keep their own kind.
    let mergeable = |i: usize| {
        matches!(
            p.features[i],
            Feature::Hole { .. } | Feature::LinearHoles { .. } | Feature::CircularHoles { .. }
        )
    };
    for (k, &(i, r)) in rs.iter().enumerate() {
        for &(j, s) in &rs[k + 1..] {
            if i != j && mergeable(i) && mergeable(j) && (r - s).abs() < DISTINCT_R {
                return false;
            }
        }
    }
    true
}

/// A random program from `seed`: valid ([`Program::valid`]), distinct hole
/// radii across features, and it builds as a vcad BRep.
pub fn program(seed: u64) -> Program {
    let mut r = Mix(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ 0xA5A5);
    loop {
        let w = q(range(&mut r, 50.0, 90.0));
        let d = q(range(&mut r, 38.0, 70.0));
        let h = q(range(&mut r, 6.0, 18.0));
        let u = r.uniform();
        let corner = if u < 0.5 {
            Corner::Sharp
        } else if u < 0.75 {
            Corner::Fillet(q(range(&mut r, 3.0, 9.0)))
        } else {
            Corner::Chamfer(q(range(&mut r, 3.0, 8.0)))
        };
        let mut p = Program { w, d, h, corner, features: Vec::new(), pose: None };
        let k = {
            let u = r.uniform();
            if u < 0.3 { 1 } else if u < 0.7 { 2 } else { 3 }
        };
        for _ in 0..k {
            for _ in 0..200 {
                let f = feature(&mut r, &p);
                let mut c = p.clone();
                c.features.push(f);
                if c.valid() && distinct_radii(&c) {
                    p = c;
                    break;
                }
            }
        }
        if p.features.is_empty() {
            continue;
        }
        let p = crate::canon::canonical(&p);
        if p.build().is_ok() {
            return p;
        }
    }
}
