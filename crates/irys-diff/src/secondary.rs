//! Phase 2f: the edges a *secondary* ray can see, and how to pick one from a
//! path vertex (Li et al. 2018's edge hierarchy, restricted to what irys
//! needs).
//!
//! [`crate::vis`] samples the discontinuities of the image: projected edges,
//! one list per frame. A path vertex `x` has its own: every triangle edge
//! that bounds what `x` sees in some direction — an occluder's silhouette
//! across a shadow ray, the arm's outline in the torso's reflection, a
//! crease's shading jump seen through a bounce. Its boundary integral over
//! directions is
//!
//! ```text
//! ∂/∂φ ∫ F(ω) dω ⊇ ∮ (F(ω⁻) − F(ω⁺)) · v_n dℓ_ω
//! ```
//!
//! with `ℓ_ω` the edge's arc on the unit sphere around `x`, `n` its normal
//! there (away from the owning triangle), `v_n` the edge's angular speed
//! *relative to how the sample was drawn* (`crate::interior` owns that
//! part). This module only draws `(edge, point)` with a known density.
//!
//! # The tree
//!
//! One bounding-volume tree per bottom-level tree, over every candidate
//! edge (an edge shared with a coplanar triangle never is — [`Edges`]), in
//! object space, so a pose never rebuilds it. A node keeps its box and its
//! edges' total length. From `x` (moved into the instance's space) a node
//! weighs `Σℓ / max(d², r²)` — `d` the distance to its box, `r` half its
//! diagonal: roughly the solid angle its edges subtend — and the descent
//! picks a child by that. A leaf (≤ [`LEAF`] edges) weighs each edge exactly:
//! the 2e owner rules seen from `x` (a silhouette is its front face's, a
//! crease between front faces the squarer face's, back-only edges keep
//! [`crate::vis::BACK_WEIGHT`]), zero when both ends are below `x`'s
//! tangent plane (no direction there carries any `F`), times
//! `ℓ / (d_mid² + ℓ²/4)`. The density of the point drawn is the product of
//! the choices over its length, so nothing that can contribute has weight 0
//! and the estimate is unbiased whatever the heuristic gets wrong.
//!
//! Instances are the top level: each candidate instance's root weighed the
//! same way. Which instances are candidates is the caller's: the moving
//! ones, and every one when `x` itself moves.

use irys_core::{Point3, Transform, Vec3};

use crate::prb::{Rng, Scene};
use crate::vis::{BACK_WEIGHT, Edges, FLAT, NONE, crease_weight};

/// Edges per leaf.
pub const LEAF: usize = 4;

#[derive(Debug, Clone)]
struct Node {
    lo: [f64; 3],
    hi: [f64; 3],
    len: f64,
    /// Leaf: `(start, count)` into the entries, `count > 0`. Inner:
    /// `(left, 0)`, right is `left + 1`.
    a: u32,
    n: u32,
}

/// One bottom-level tree's candidate edges, in a tree of their own.
#[derive(Debug, Clone, Default)]
pub struct EdgeTree {
    /// `(triangle, edge)`, the edge from corner `e` to `e + 1`.
    entries: Vec<(u32, u8)>,
    nodes: Vec<Node>,
}

fn dist2(p: [f64; 3], lo: &[f64; 3], hi: &[f64; 3]) -> f64 {
    (0..3)
        .map(|k| {
            let d = (lo[k] - p[k]).max(p[k] - hi[k]).max(0.0);
            d * d
        })
        .sum()
}

fn half_diag2(lo: &[f64; 3], hi: &[f64; 3]) -> f64 {
    0.25 * (0..3).map(|k| (hi[k] - lo[k]).powi(2)).sum::<f64>()
}

impl Node {
    fn weight(&self, p: [f64; 3]) -> f64 {
        self.len / dist2(p, &self.lo, &self.hi).max(half_diag2(&self.lo, &self.hi)).max(1e-300)
    }
}

impl EdgeTree {
    fn build(scene: &Scene, edges: &Edges, blas: usize) -> Self {
        let m = scene.blas()[blas].geometry().mesh();
        let (pos, tris) = (m.positions(), m.triangles());
        let nbr = &edges.nbr[blas];
        let mut entries = Vec::new();
        let mut info = Vec::new(); // (lo, hi, centre, len)
        for (ti, t) in tris.iter().enumerate() {
            for e in 0..3 {
                if nbr[ti][e] == FLAT {
                    continue;
                }
                let (a, b) = (pos[t[e] as usize], pos[t[(e + 1) % 3] as usize]);
                let len = (b - a).norm();
                if len <= 0.0 {
                    continue;
                }
                let lo = [a.x.min(b.x), a.y.min(b.y), a.z.min(b.z)];
                let hi = [a.x.max(b.x), a.y.max(b.y), a.z.max(b.z)];
                let c = [0.5 * (a.x + b.x), 0.5 * (a.y + b.y), 0.5 * (a.z + b.z)];
                entries.push((ti as u32, e as u8));
                info.push((lo, hi, c, len));
            }
        }
        let mut t = Self {
            entries: Vec::with_capacity(entries.len()),
            nodes: Vec::new(),
        };
        if entries.is_empty() {
            return t;
        }
        let mut idx: Vec<usize> = (0..entries.len()).collect();
        t.nodes.push(Node {
            lo: [0.0; 3],
            hi: [0.0; 3],
            len: 0.0,
            a: 0,
            n: 0,
        });
        // Iterative: (node slot, range in idx).
        let mut stack = vec![(0usize, 0usize, idx.len())];
        let mut order = Vec::with_capacity(idx.len());
        while let Some((slot, s, e)) = stack.pop() {
            let (mut lo, mut hi, mut clo, mut chi) = (
                [f64::INFINITY; 3],
                [f64::NEG_INFINITY; 3],
                [f64::INFINITY; 3],
                [f64::NEG_INFINITY; 3],
            );
            let mut len = 0.0;
            for &i in &idx[s..e] {
                let (l, h, c, ln) = &info[i];
                for k in 0..3 {
                    lo[k] = lo[k].min(l[k]);
                    hi[k] = hi[k].max(h[k]);
                    clo[k] = clo[k].min(c[k]);
                    chi[k] = chi[k].max(c[k]);
                }
                len += ln;
            }
            let node = &mut t.nodes[slot];
            node.lo = lo;
            node.hi = hi;
            node.len = len;
            if e - s <= LEAF {
                node.a = order.len() as u32;
                node.n = (e - s) as u32;
                order.extend_from_slice(&idx[s..e]);
                continue;
            }
            let ax = (0..3)
                .max_by(|&a, &b| (chi[a] - clo[a]).total_cmp(&(chi[b] - clo[b])))
                .unwrap();
            let mid = s + (e - s) / 2;
            idx[s..e].select_nth_unstable_by(mid - s, |&a, &b| info[a].2[ax].total_cmp(&info[b].2[ax]));
            let left = t.nodes.len();
            t.nodes[slot].a = left as u32;
            t.nodes[slot].n = 0;
            for _ in 0..2 {
                t.nodes.push(Node {
                    lo: [0.0; 3],
                    hi: [0.0; 3],
                    len: 0.0,
                    a: 0,
                    n: 0,
                });
            }
            stack.push((left, s, mid));
            stack.push((left + 1, mid, e));
        }
        t.entries = order.into_iter().map(|i| entries[i]).collect();
        t
    }

    /// Candidate edges in the tree.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether it has none.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Every bottom-level tree's [`EdgeTree`]: built once per scene (object
/// space, so poses do not touch it).
pub struct Secondary {
    trees: Vec<EdgeTree>,
    pub(crate) edges: Edges,
}

/// One drawn edge point.
#[derive(Debug, Clone, Copy)]
pub struct EdgePick {
    /// Placement payload.
    pub k: usize,
    /// Triangle (the owner) and its edge.
    pub tri: usize,
    /// Edge `e → e + 1`.
    pub e: usize,
    /// Object-space ends.
    pub la: Point3,
    /// Object-space ends.
    pub lb: Point3,
    /// World-space ends.
    pub a: Point3,
    /// World-space ends.
    pub b: Point3,
    /// World-space third corner of the owner.
    pub c: Point3,
    /// Fraction along the edge.
    pub t: f64,
    /// Density of the point, per unit world length.
    pub pdf: f64,
}

/// Per instance, what a vertex needs from the current pose.
pub struct Frame {
    pub(crate) inv: Vec<Option<Transform>>,
    scale: Vec<f64>,
}

impl Secondary {
    /// The trees of `scene`'s bottom-level trees.
    pub fn new(scene: &Scene) -> Self {
        let edges = Edges::new(scene);
        let trees = (0..scene.blas().len())
            .map(|b| EdgeTree::build(scene, &edges, b))
            .collect();
        Self { trees, edges }
    }

    /// Candidate edges, all trees.
    pub fn candidates(&self) -> usize {
        self.trees.iter().map(EdgeTree::len).sum()
    }

    /// The pose's per-instance inverses.
    pub fn frame(&self, scene: &Scene) -> Frame {
        let (inv, scale) = scene
            .placements()
            .iter()
            .map(|p| {
                let t = &p.to_world;
                let v = [
                    t.apply_vec(&Vec3::new(1.0, 0.0, 0.0)),
                    t.apply_vec(&Vec3::new(0.0, 1.0, 0.0)),
                    t.apply_vec(&Vec3::new(0.0, 0.0, 1.0)),
                ];
                let det = v[0].dot(&v[1].cross(&v[2])).abs();
                (t.inverse(), det.cbrt().max(1e-300))
            })
            .unzip();
        Frame { inv, scale }
    }

    /// Draw an edge point seen from world point `x` with normal `n` (the
    /// side the path is on), among instances `cands`, by one-sample MIS over
    /// [`Lobes`]' strategies. `None` when nothing can contribute (or the
    /// strategy drawn found nothing: a defective strategy is still
    /// unbiased in the mixture).
    #[allow(clippy::too_many_arguments)]
    pub fn pick(
        &self,
        scene: &Scene,
        frame: &Frame,
        cands: &[usize],
        x: Point3,
        n: Vec3,
        lobes: &Lobes,
        rng: &mut Rng,
    ) -> Option<EdgePick> {
        let c = lobes.mix();
        let s = {
            let u = rng.uniform();
            if u < c[0] {
                0
            } else if u < c[0] + c[1] {
                1
            } else {
                2
            }
        };
        // Per strategy: the probability of the choices made so far.
        let mut p = [1.0f64; 3];
        let choose = |w: &[[f64; 3]], p: &mut [f64; 3], u: f64| -> Option<usize> {
            let tot: [f64; 3] = std::array::from_fn(|i| w.iter().map(|x| x[i]).sum());
            if !(tot[s] > 0.0) {
                return None;
            }
            let target = u * tot[s];
            let mut acc = 0.0;
            let mut pick = None;
            for (i, wi) in w.iter().enumerate() {
                if wi[s] <= 0.0 {
                    continue;
                }
                acc += wi[s];
                pick = Some(i);
                if target < acc {
                    break;
                }
            }
            let i = pick?;
            for k in 0..3 {
                p[k] *= if tot[k] > 0.0 { w[i][k] / tot[k] } else { 0.0 };
            }
            Some(i)
        };
        // Top level: each candidate instance's root.
        let w: Vec<[f64; 3]> = cands
            .iter()
            .map(|&k| {
                let pl = &scene.placements()[k];
                match (&frame.inv[k], self.trees[pl.blas].nodes.first()) {
                    (Some(inv), Some(root)) => {
                        let xl = inv.apply_point(&x);
                        let g = root.weight([xl.x, xl.y, xl.z]) / frame.scale[k];
                        lobes.weigh(g, x, cone(&pl.to_world, x, xl, root))
                    }
                    _ => [0.0; 3],
                }
            })
            .collect();
        let ci = choose(&w, &mut p, rng.uniform())?;
        let k = cands[ci];
        let place = &scene.placements()[k];
        let tree = &self.trees[place.blas];
        let inv = frame.inv[k].as_ref()?;
        let xlp = inv.apply_point(&x);
        let xl = [xlp.x, xlp.y, xlp.z];
        let mut node = 0usize;
        loop {
            let nd = &tree.nodes[node];
            if nd.n > 0 {
                break;
            }
            let kids = [nd.a as usize, nd.a as usize + 1];
            let w: Vec<[f64; 3]> = kids
                .iter()
                .map(|&q| {
                    let ch = &tree.nodes[q];
                    lobes.weigh(ch.weight(xl), x, cone(&place.to_world, x, xlp, ch))
                })
                .collect();
            node = kids[choose(&w, &mut p, rng.uniform())?];
        }
        // The leaf: exact weights seen from `x`.
        let nd = &tree.nodes[node];
        let m = scene.blas()[place.blas].geometry().mesh();
        let (pos, tris) = (m.positions(), m.triangles());
        let nbr = &self.edges.nbr[place.blas];
        let cosv = &self.edges.cos[place.blas];
        let normal = &self.edges.normal[place.blas];
        let facing = |tri: usize| {
            let p0 = place.to_world.apply_point(&pos[tris[tri][0] as usize]);
            let nw = place.to_world.apply_normal(&normal[tri]);
            nw.dot(&(x - p0)) / (nw.norm() * (x - p0).norm()).max(1e-300)
        };
        let range = nd.a as usize..(nd.a + nd.n) as usize;
        let mut lw = vec![[0.0f64; 3]; range.len()];
        for (slot, &(tri, e)) in tree.entries[range.clone()].iter().enumerate() {
            let (tri, e) = (tri as usize, e as usize);
            let t = tris[tri];
            let a = place.to_world.apply_point(&pos[t[e] as usize]);
            let b = place.to_world.apply_point(&pos[t[(e + 1) % 3] as usize]);
            if n.dot(&(a - x)) <= 0.0 && n.dot(&(b - x)) <= 0.0 {
                continue;
            }
            let fa = facing(tri);
            let f = fa > 0.0;
            let imp = match nbr[tri][e] {
                FLAT => 0.0,
                NONE if f => 1.0,
                NONE => BACK_WEIGHT,
                nb => {
                    let fb = facing(nb as usize);
                    match (f, fb > 0.0) {
                        (true, true) => {
                            if fa < fb || (fa == fb && tri > nb as usize) {
                                0.0
                            } else {
                                crease_weight(cosv[tri][e] as f64)
                            }
                        }
                        (true, false) => 1.0,
                        _ => BACK_WEIGHT,
                    }
                }
            };
            if imp <= 0.0 {
                continue;
            }
            let len = (b - a).norm();
            let mid = a + (b - a) * 0.5;
            let d2 = (mid - x).norm_squared();
            let g = imp * len / (d2 + 0.25 * len * len).max(1e-300);
            let half = (0.5 * len / d2.sqrt().max(1e-300)).min(1.0).asin();
            lw[slot] = lobes.weigh(g, x, ((mid - x).normalize(), half));
        }
        let slot = choose(&lw, &mut p, rng.uniform())?;
        let (tri, e) = tree.entries[nd.a as usize + slot];
        let (tri, e) = (tri as usize, e as usize);
        let t3 = tris[tri];
        let (la, lb) = (pos[t3[e] as usize], pos[t3[(e + 1) % 3] as usize]);
        let (a, b) = (place.to_world.apply_point(&la), place.to_world.apply_point(&lb));
        let cpt = place.to_world.apply_point(&pos[t3[(e + 2) % 3] as usize]);
        let len = (b - a).norm();
        // The point along the edge: per strategy, piecewise constant over
        // SEGS pieces by the strategy's directional factor.
        let seg: Vec<[f64; 3]> = (0..SEGS)
            .map(|i| {
                let m = a + (b - a) * ((i as f64 + 0.5) / SEGS as f64);
                let dm = (m - x).norm().max(1e-300);
                let half = (0.5 * len / SEGS as f64 / dm).min(1.0).asin();
                lobes.weigh(1.0, x, ((m - x) / dm, half))
            })
            .collect();
        let si = choose(&seg, &mut p, rng.uniform())?;
        let t = (si as f64 + rng.uniform()) / SEGS as f64;
        // `p` now holds each strategy's probability of this segment; the
        // point is uniform in it.
        let pdf: f64 = (0..3).map(|i| c[i] * p[i]).sum::<f64>() * SEGS as f64 / len;
        Some(EdgePick {
            k,
            tri,
            e,
            la,
            lb,
            a,
            b,
            c: cpt,
            t,
            pdf,
        })
    }
}

/// Pieces an edge is cut into for the point's density along it.
pub const SEGS: usize = 16;

/// The cone from world point `x` around node `nd` of an instance placed by
/// `to_world` (`xl` is `x` in its space): world direction to the box's
/// centre and the half-angle its bounding sphere subtends.
fn cone(to_world: &Transform, x: Point3, xl: Point3, nd: &Node) -> (Vec3, f64) {
    let c = Point3::new(
        0.5 * (nd.lo[0] + nd.hi[0]),
        0.5 * (nd.lo[1] + nd.hi[1]),
        0.5 * (nd.lo[2] + nd.hi[2]),
    );
    let r = half_diag2(&nd.lo, &nd.hi).sqrt();
    let d = (c - xl).norm();
    if d <= r || d <= 0.0 {
        return (Vec3::new(0.0, 0.0, 1.0), std::f64::consts::PI);
    }
    let cw = to_world.apply_point(&c);
    ((cw - x).normalize(), (r / d).asin())
}

/// The directional factors the three edge strategies weigh by: 1
/// (geometry alone), the sun lobe, and — at a conductor — its reflection
/// lobe. Each is an upper-bound-like proxy over a cone of directions; only
/// positivity matters for unbiasedness.
#[derive(Debug, Clone, Copy)]
pub struct Lobes {
    /// Unit direction to the sun, and its vMF concentration.
    pub sun: Vec3,
    /// Concentration.
    pub kappa: f64,
    /// At a conductor: the mirror direction and the roughness.
    pub mirror: Option<(Vec3, f64)>,
}

fn angle(a: Vec3, b: Vec3) -> f64 {
    a.dot(&b).clamp(-1.0, 1.0).acos()
}

impl Lobes {
    /// The strategies' probabilities: geometry, sun, mirror.
    pub fn mix(&self) -> [f64; 3] {
        match self.mirror {
            Some(_) => [0.34, 0.33, 0.33],
            None => [0.5, 0.5, 0.0],
        }
    }

    fn weigh(&self, g: f64, _x: Point3, (dir, half): (Vec3, f64)) -> [f64; 3] {
        let sun = {
            let d = (angle(dir, self.sun) - half).max(0.0);
            (self.kappa * (d.cos() - 1.0)).exp()
        };
        let mirror = match self.mirror {
            Some((r, alpha)) => {
                let d = (angle(dir, r) - half).max(0.0);
                let q = d / (2.0 * alpha).max(1e-3);
                1.0 / (1.0 + q * q).powi(2)
            }
            None => 0.0,
        };
        [g, g * sun, g * mirror]
    }
}
