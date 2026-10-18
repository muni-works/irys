//! Distance to the nearest triangle, over the same flattened tree the GPU
//! walks: the random walk's safe sphere.
//!
//! A walk takes hundreds of steps a few hundredths of a millimetre long, most
//! of them far from any surface. Tracing each step against the tree costs a
//! root-to-leaf descent regardless of how short it is; instead the walk asks
//! once how far the nearest triangle is, and every later step that ends
//! inside that sphere cannot cross the boundary and needs no test at all
//! (the sphere is convex and the walk starts inside it). This changes no
//! outcome — a step either could not hit anything, or is traced exactly as
//! before — only how often the tree is touched.

use irys_core::{Bvh, Point3, TriMesh, Vec3};

/// The flattened tree and its triangles' corners.
#[derive(Debug, Clone)]
pub struct Accel {
    nodes: Vec<(Point3, Point3, bool, u32, u32)>,
    ids: Vec<u32>,
    tris: Vec<[Point3; 3]>,
}

impl Accel {
    /// From a tree over a [`TriMesh`].
    pub fn new(bvh: &Bvh<TriMesh>) -> Self {
        let (flat, ids) = bvh.flatten();
        let m = bvh.geometry();
        Self {
            nodes: flat
                .iter()
                .map(|&(b, leaf, a, c)| (b.min, b.max, leaf, a, c))
                .collect(),
            ids,
            tris: m
                .triangles()
                .iter()
                .map(|t| t.map(|i| m.positions()[i as usize]))
                .collect(),
        }
    }

    /// Distance from `p` to the nearest triangle (∞ for an empty mesh).
    pub fn distance(&self, p: Point3) -> f64 {
        if self.nodes.is_empty() {
            return f64::INFINITY;
        }
        let mut best = f64::INFINITY;
        let mut stack = [0u32; 64];
        let mut sp = 1;
        while sp > 0 {
            sp -= 1;
            let n = stack[sp] as usize;
            let (lo, hi, leaf, a, c) = self.nodes[n];
            if box_d2(p, lo, hi) >= best {
                continue;
            }
            if leaf {
                for k in a..a + c {
                    let t = &self.tris[self.ids[k as usize] as usize];
                    best = best.min(tri_d2(p, t[0], t[1], t[2]));
                }
                continue;
            }
            let (da, db) = (
                box_d2(p, self.nodes[a as usize].0, self.nodes[a as usize].1),
                box_d2(p, self.nodes[c as usize].0, self.nodes[c as usize].1),
            );
            // Nearer child on top.
            if da <= db {
                stack[sp] = c;
                stack[sp + 1] = a;
            } else {
                stack[sp] = a;
                stack[sp + 1] = c;
            }
            sp += 2;
        }
        best.sqrt()
    }
}

fn box_d2(p: Point3, lo: Point3, hi: Point3) -> f64 {
    let d = |x: f64, a: f64, b: f64| (a - x).max(0.0).max(x - b);
    let (dx, dy, dz) = (d(p.x, lo.x, hi.x), d(p.y, lo.y, hi.y), d(p.z, lo.z, hi.z));
    dx * dx + dy * dy + dz * dz
}

/// Squared distance from `p` to triangle `abc` (Ericson, Real-Time Collision
/// Detection, 5.1.5).
pub fn tri_d2(p: Point3, a: Point3, b: Point3, c: Point3) -> f64 {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));
    let q = |x: Vec3| x.norm_squared();
    if d1 <= 0.0 && d2 <= 0.0 {
        return q(ap);
    }
    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));
    if d3 >= 0.0 && d4 <= d3 {
        return q(bp);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return q(ap - ab * v);
    }
    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));
    if d6 >= 0.0 && d5 <= d6 {
        return q(cp);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return q(ap - ac * w);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return q(bp - (c - b) * w);
    }
    let denom = 1.0 / (va + vb + vc);
    let (v, w) = (vb * denom, vc * denom);
    q(ap - ab * v - ac * w)
}
