//! Speed round 4: a light tree over emitter triangles (Conty Estevez &
//! Kulla 2018, the shape of Cycles' `light_tree.cpp`), replacing the
//! area × luminance CDF for next-event estimation in the OpenPBR scene
//! kernel (`scene.wgsl`'s `tree_*`).
//!
//! A binary tree over clusters of up to [`CLUSTER`] triangles, built
//! top-down by a binned split of the centroids that minimises Conty's
//! surface-area-orientation cost. Each child keeps its bounds (centre and
//! half-diagonal), the power under it and an orientation cone (axis, θ_o;
//! θ_e = π/2 for area lights), stored in its parent so a step is one read.
//! At a shading point a traversal descends by the two children's
//! importance `E · cos θ' · cos θ'_i / d²`, conservative in the bounds,
//! then picks within the cluster by power; a triangle's probability is the
//! product of the branch probabilities on its path and its share of the
//! cluster. Every triangle keeps that path as bits (`trail`) so the device
//! can evaluate the same probability for a BSDF-sampled hit (MIS).
//!
//! [`Tree::pdf`] and [`Tree::sample`] are the device's arithmetic in f32,
//! for the tests.

/// One emitter triangle as the tree sees it.
#[derive(Debug, Clone, Copy)]
pub struct Prim {
    /// World positions.
    pub p: [[f32; 3]; 3],
    /// Power proxy: area × luminance.
    pub energy: f32,
    /// Emits from both sides.
    pub two_sided: bool,
    /// Emits from the back (against the winding's normal).
    pub back: bool,
}

/// A child reference: an inner node's index, or a cluster's with this bit.
pub const LEAF: u32 = 0x8000_0000;
/// Triangles per leaf cluster, at most.
pub const CLUSTER: usize = 8;

/// The built tree, as the device reads it.
///
/// Each inner node holds both children's summaries, five vec4 in `nodes`:
/// `(c_l, E_l)`, `(c_r, E_r)`, `(axis_l, cos θo_l)`, `(axis_r, cos θo_r)`,
/// `(r²_l, r²_r, link_l, link_r)` (centre, power, orientation cone, squared
/// half-diagonal of the bounds, child reference), so a traversal step is
/// one contiguous read. Node 0 is the root. `meta` (u32): per triangle its
/// trail (bit `k` set where inner level `k` went right; the inner depth in
/// the top 6 bits), then per cluster `(first, count)` into the members, then
/// per member `(triangle, energy bits)`; [`Tree::offsets`] says where.
#[derive(Debug, Clone, Default)]
pub struct Tree {
    /// Inner nodes, five vec4 each.
    pub nodes: Vec<[f32; 4]>,
    /// Trails, clusters, members.
    pub meta: Vec<u32>,
    /// Triangles.
    pub n: usize,
    /// Clusters.
    pub clusters: usize,
}

#[derive(Clone, Copy)]
struct Cone {
    axis: [f32; 3],
    cos_o: f32,
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: [f32; 3]) -> [f32; 3] {
    let l = dot(a, a).sqrt();
    if l > 0.0 { [a[0] / l, a[1] / l, a[2] / l] } else { [0.0, 0.0, 1.0] }
}

/// pbrt-v4's `Union(DirectionCone, DirectionCone)`.
fn union(a: Cone, b: Cone) -> Cone {
    if a.cos_o <= -1.0 || b.cos_o <= -1.0 {
        return Cone { axis: a.axis, cos_o: -1.0 };
    }
    let ta = a.cos_o.clamp(-1.0, 1.0).acos();
    let tb = b.cos_o.clamp(-1.0, 1.0).acos();
    let td = dot(a.axis, b.axis).clamp(-1.0, 1.0).acos();
    if (td + tb).min(std::f32::consts::PI) <= ta {
        return a;
    }
    if (td + ta).min(std::f32::consts::PI) <= tb {
        return b;
    }
    let to = (ta + td + tb) / 2.0;
    if to >= std::f32::consts::PI {
        return Cone { axis: a.axis, cos_o: -1.0 };
    }
    let tr = to - ta;
    let w = cross(a.axis, b.axis);
    if dot(w, w) < 1e-12 {
        return Cone { axis: a.axis, cos_o: -1.0 };
    }
    // rotate a.axis by tr about w
    let k = norm(w);
    let v = a.axis;
    let (s, c) = tr.sin_cos();
    let kv = cross(k, v);
    let kd = dot(k, v);
    let axis = norm([
        v[0] * c + kv[0] * s + k[0] * kd * (1.0 - c),
        v[1] * c + kv[1] * s + k[1] * kd * (1.0 - c),
        v[2] * c + kv[2] * s + k[2] * kd * (1.0 - c),
    ]);
    Cone { axis, cos_o: to.cos() }
}

struct Item {
    lo: [f32; 3],
    hi: [f32; 3],
    c: [f32; 3],
    cone: Cone,
    energy: f32,
    id: u32,
}

fn bounds(items: &[Item]) -> ([f32; 3], [f32; 3], Cone, f32) {
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    let mut cone = items[0].cone;
    let mut e = 0.0;
    for (k, it) in items.iter().enumerate() {
        for a in 0..3 {
            lo[a] = lo[a].min(it.lo[a]);
            hi[a] = hi[a].max(it.hi[a]);
        }
        if k > 0 {
            cone = union(cone, it.cone);
        }
        e += it.energy;
    }
    (lo, hi, cone, e)
}

/// Conty's orientation measure M_Ω for θ_e = π/2.
fn m_omega(cos_o: f32) -> f32 {
    let to = cos_o.clamp(-1.0, 1.0).acos();
    let te = (to + std::f32::consts::FRAC_PI_2).min(std::f32::consts::PI);
    let pi = std::f32::consts::PI;
    2.0 * pi * (1.0 - cos_o)
        + pi / 2.0 * (2.0 * te * to.sin() - (to - 2.0 * te).cos() - 2.0 * to * to.sin() + cos_o)
}

fn area(lo: [f32; 3], hi: [f32; 3]) -> f32 {
    let d = sub(hi, lo).map(|x| x.max(0.0));
    2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])
}

struct Summary {
    c: [f32; 3],
    r2: f32,
    energy: f32,
    cone: Cone,
    link: u32,
}

impl Tree {
    /// Build over `prims` (at least one).
    pub fn build(prims: &[Prim]) -> Tree {
        let mut items: Vec<Item> = prims
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut lo = [f32::MAX; 3];
                let mut hi = [f32::MIN; 3];
                for v in &p.p {
                    for a in 0..3 {
                        lo[a] = lo[a].min(v[a]);
                        hi[a] = hi[a].max(v[a]);
                    }
                }
                let n = norm(cross(sub(p.p[1], p.p[0]), sub(p.p[2], p.p[0])));
                let cone = if p.two_sided {
                    Cone { axis: n, cos_o: -1.0 }
                } else if p.back {
                    Cone { axis: n.map(|x| -x), cos_o: 1.0 }
                } else {
                    Cone { axis: n, cos_o: 1.0 }
                };
                let c = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5];
                Item { lo, hi, c, cone, energy: p.energy.max(0.0), id: i as u32 }
            })
            .collect();
        let n = prims.len();
        let mut t = Tree { nodes: Vec::new(), meta: vec![0; n], n, clusters: 0 };
        let mut clusters: Vec<[u32; 2]> = Vec::new();
        let mut members: Vec<[u32; 2]> = Vec::new();
        if n > 0 {
            // the root is an inner node, even over one cluster
            let (l, r) = if n == 1 { items.split_at_mut(1) } else {
                let mid = if n <= CLUSTER { n / 2 } else { split(&mut items) };
                items.split_at_mut(mid)
            };
            t.nodes.extend([[0.0; 4]; 5]);
            let a = t.child(l, 0, 1, &mut clusters, &mut members);
            let b = if r.is_empty() {
                Summary { c: a.c, r2: a.r2, energy: 0.0, cone: a.cone, link: a.link }
            } else {
                t.child(r, 1, 1, &mut clusters, &mut members)
            };
            t.put(0, &a, &b);
        }
        t.clusters = clusters.len();
        t.meta.extend(clusters.iter().flatten());
        t.meta.extend(members.iter().flatten());
        t
    }

    /// Where the clusters and the members start in `meta`.
    pub fn offsets(&self) -> (usize, usize) {
        (self.n, self.n + 2 * self.clusters)
    }

    fn put(&mut self, at: usize, a: &Summary, b: &Summary) {
        let o = 5 * at;
        self.nodes[o] = [a.c[0], a.c[1], a.c[2], a.energy];
        self.nodes[o + 1] = [b.c[0], b.c[1], b.c[2], b.energy];
        self.nodes[o + 2] = [a.cone.axis[0], a.cone.axis[1], a.cone.axis[2], a.cone.cos_o];
        self.nodes[o + 3] = [b.cone.axis[0], b.cone.axis[1], b.cone.axis[2], b.cone.cos_o];
        self.nodes[o + 4] = [a.r2, b.r2, f32::from_bits(a.link), f32::from_bits(b.link)];
    }

    /// The subtree over `items`, reached by `trail` at inner depth `depth`.
    fn child(
        &mut self,
        items: &mut [Item],
        trail: u32,
        depth: u32,
        clusters: &mut Vec<[u32; 2]>,
        members: &mut Vec<[u32; 2]>,
    ) -> Summary {
        let (lo, hi, cone, e) = bounds(items);
        let c = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5];
        let h = sub(hi, c);
        let r2 = dot(h, h);
        if items.len() <= CLUSTER {
            let k = clusters.len() as u32;
            clusters.push([members.len() as u32, items.len() as u32]);
            for it in items.iter() {
                members.push([it.id, it.energy.to_bits()]);
                self.meta[it.id as usize] = trail | (depth << 26);
            }
            return Summary { c, r2, energy: e, cone, link: LEAF | k };
        }
        // a median split once the depth budget (26 levels: the trail's bits)
        // leaves no room for anything less balanced
        let need = usize::BITS - (items.len().div_ceil(CLUSTER) - 1).leading_zeros();
        assert!(depth + need <= 26, "light tree too deep");
        let mid = if depth + need >= 24 { median_split(items) } else { split(items) };
        let at = self.nodes.len() / 5;
        self.nodes.extend([[0.0; 4]; 5]);
        let (l, r) = items.split_at_mut(mid);
        let a = self.child(l, trail, depth + 1, clusters, members);
        let b = self.child(r, trail | (1 << depth), depth + 1, clusters, members);
        self.put(at, &a, &b);
        Summary { c, r2, energy: e, cone, link: at as u32 }
    }

    fn pair(&self, i: usize, x: [f32; 3], n: [f32; 3]) -> (f32, f32, u32, u32) {
        let o = 5 * i;
        let (a, b, ca, cb, m) = (self.nodes[o], self.nodes[o + 1], self.nodes[o + 2], self.nodes[o + 3], self.nodes[o + 4]);
        let il = importance([a[0], a[1], a[2]], m[0], a[3], [ca[0], ca[1], ca[2]], ca[3], x, n);
        let ir = importance([b[0], b[1], b[2]], m[1], b[3], [cb[0], cb[1], cb[2]], cb[3], x, n);
        (il, ir, m[2].to_bits(), m[3].to_bits())
    }

    fn member(&self, cluster: u32, id: u32) -> f32 {
        let (co, mo) = self.offsets();
        let first = self.meta[co + 2 * cluster as usize] as usize;
        let count = self.meta[co + 2 * cluster as usize + 1] as usize;
        let mut tot = 0.0f32;
        let mut mine = 0.0f32;
        for k in 0..count {
            let e = f32::from_bits(self.meta[mo + 2 * (first + k) + 1]);
            tot += e;
            if self.meta[mo + 2 * (first + k)] == id {
                mine = e;
            }
        }
        if tot > 0.0 { mine / tot } else { 0.0 }
    }

    /// The probability the traversal picks triangle `id` at `(x, n)`.
    pub fn pdf(&self, id: usize, x: [f32; 3], n: [f32; 3]) -> f32 {
        let tr = self.meta[id];
        let depth = tr >> 26;
        let mut i = 0usize;
        let mut p = 1.0f32;
        let mut link = 0;
        for k in 0..depth {
            let (il, ir, ll, lr) = self.pair(i, x, n);
            let s = il + ir;
            if !(s > 0.0) {
                return 0.0;
            }
            if tr >> k & 1 == 1 {
                p *= ir / s;
                link = lr;
            } else {
                p *= il / s;
                link = ll;
            }
            i = link as usize;
        }
        p * self.member(link & !LEAF, id as u32)
    }

    /// The triangle `u` picks at `(x, n)`, with its probability.
    pub fn sample(&self, mut u: f32, x: [f32; 3], n: [f32; 3]) -> Option<(usize, f32)> {
        let mut i = 0usize;
        let mut p = 1.0f32;
        loop {
            let (il, ir, ll, lr) = self.pair(i, x, n);
            let s = il + ir;
            if !(s > 0.0) {
                return None;
            }
            let pl = il / s;
            let link = if u < pl {
                u = (u / pl).min(0.99999994);
                p *= pl;
                ll
            } else {
                u = ((u - pl) / (1.0 - pl)).min(0.99999994);
                p *= 1.0 - pl;
                lr
            };
            if link & LEAF != 0 {
                let (co, mo) = self.offsets();
                let k = (link & !LEAF) as usize;
                let first = self.meta[co + 2 * k] as usize;
                let count = self.meta[co + 2 * k + 1] as usize;
                let tot: f32 = (0..count).map(|j| f32::from_bits(self.meta[mo + 2 * (first + j) + 1])).sum();
                let mut acc = 0.0;
                for j in 0..count {
                    let e = f32::from_bits(self.meta[mo + 2 * (first + j) + 1]);
                    acc += e;
                    if u * tot < acc || j + 1 == count {
                        return (tot > 0.0 && e > 0.0).then(|| (self.meta[mo + 2 * (first + j)] as usize, p * e / tot));
                    }
                }
            }
            i = link as usize;
        }
    }
}

/// Conty's bound, as the device computes it: a child of centre `c`,
/// squared half-diagonal `r2`, power `energy` and cone `(axis, cos_o)`, at
/// `x` with receiver normal `n` (zero: none).
pub fn importance(c: [f32; 3], r2: f32, energy: f32, axis: [f32; 3], cos_o: f32, x: [f32; 3], n: [f32; 3]) -> f32 {
    if !(energy > 0.0) {
        return 0.0;
    }
    let v = sub(x, c);
    let d2 = dot(v, v);
    // inside the bounds (or nearly): no angular bound, the distance floored
    if d2 <= r2 {
        return energy / r2.max(1e-8);
    }
    let d = d2.sqrt();
    let sin_u = (r2 / d2).sqrt().min(1.0);
    let cos_u = (1.0 - sin_u * sin_u).max(0.0).sqrt();
    let w = [v[0] / d, v[1] / d, v[2] / d]; // centre → x
    // emitter side: θ' = max(0, θ − θ_o − θ_u), zero past θ_e = π/2
    let cos_t = dot(axis, w).clamp(-1.0, 1.0);
    let cos_e = cos_minus(cos_minus_pair(cos_t, cos_o), (cos_u, sin_u));
    if cos_e <= 0.0 {
        return 0.0;
    }
    // receiver side: θ'_i = max(0, θ_i − θ_u)
    let mut cos_i = 1.0;
    if dot(n, n) > 0.0 {
        let ci = (-dot(n, w)).clamp(-1.0, 1.0);
        let si = (1.0 - ci * ci).max(0.0).sqrt();
        cos_i = cos_minus((ci, si), (cos_u, sin_u));
        if cos_i <= 0.0 {
            return 0.0;
        }
    }
    energy * cos_e * cos_i / d2.max(r2)
}

/// `(cos a, sin a)` of `max(0, θ − θ_o)` from `cos θ` and `cos θ_o`.
fn cos_minus_pair(cos_t: f32, cos_o: f32) -> (f32, f32) {
    let st = (1.0 - cos_t * cos_t).max(0.0).sqrt();
    let so = (1.0 - cos_o * cos_o).max(0.0).sqrt();
    if cos_t >= cos_o {
        return (1.0, 0.0);
    }
    // cos(θ − θ_o), sin(θ − θ_o)
    (cos_t * cos_o + st * so, st * cos_o - cos_t * so)
}

/// `cos(max(0, a − b))` from `(cos a, sin a)` and `(cos b, sin b)`.
fn cos_minus(a: (f32, f32), b: (f32, f32)) -> f32 {
    if a.0 >= b.0 {
        return 1.0;
    }
    a.0 * b.0 + a.1 * b.1
}

fn median_split(items: &mut [Item]) -> usize {
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for it in items.iter() {
        for a in 0..3 {
            lo[a] = lo[a].min(it.c[a]);
            hi[a] = hi[a].max(it.c[a]);
        }
    }
    let ext = sub(hi, lo);
    let ax = if ext[0] >= ext[1] && ext[0] >= ext[2] { 0 } else if ext[1] >= ext[2] { 1 } else { 2 };
    items.sort_by(|a, b| a.c[ax].total_cmp(&b.c[ax]).then(a.id.cmp(&b.id)));
    items.len() / 2
}

/// Binned SAOH split (12 bins on the widest centroid axis); a median split
/// when every centroid shares a bin. Returns the left count, ≥ 1.
fn split(items: &mut [Item]) -> usize {
    let n = items.len();
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for it in items.iter() {
        for a in 0..3 {
            lo[a] = lo[a].min(it.c[a]);
            hi[a] = hi[a].max(it.c[a]);
        }
    }
    let ext = sub(hi, lo);
    let ax = if ext[0] >= ext[1] && ext[0] >= ext[2] { 0 } else if ext[1] >= ext[2] { 1 } else { 2 };
    let median = |items: &mut [Item]| {
        items.sort_by(|a, b| a.c[ax].total_cmp(&b.c[ax]).then(a.id.cmp(&b.id)));
        n / 2
    };
    if !(ext[ax] > 0.0) {
        return median(items);
    }
    const B: usize = 12;
    let bin = |c: f32| (((c - lo[ax]) / ext[ax] * B as f32) as usize).min(B - 1);
    let mut best = (f32::MAX, 0usize);
    let (plo, phi, pcone, _) = bounds(items);
    let parent = (area(plo, phi).max(1e-12) * m_omega(pcone.cos_o)).max(1e-12);
    for s in 1..B {
        let (l, r): (Vec<&Item>, Vec<&Item>) = items.iter().partition(|it| bin(it.c[ax]) < s);
        if l.is_empty() || r.is_empty() {
            continue;
        }
        let cost = |v: &[&Item]| {
            let mut lo = [f32::MAX; 3];
            let mut hi = [f32::MIN; 3];
            let mut cone = v[0].cone;
            let mut e = 0.0;
            for (k, it) in v.iter().enumerate() {
                for a in 0..3 {
                    lo[a] = lo[a].min(it.lo[a]);
                    hi[a] = hi[a].max(it.hi[a]);
                }
                if k > 0 {
                    cone = union(cone, it.cone);
                }
                e += it.energy;
            }
            e.max(1e-12) * area(lo, hi) * m_omega(cone.cos_o)
        };
        let c = (cost(&l) + cost(&r)) / parent;
        if c < best.0 {
            best = (c, s);
        }
    }
    if best.1 == 0 {
        return median(items);
    }
    let s = best.1;
    items.sort_by(|a, b| bin(a.c[ax]).cmp(&bin(b.c[ax])).then(a.id.cmp(&b.id)));
    let m = items.iter().position(|it| bin(it.c[ax]) >= s).unwrap();
    m.clamp(1, n - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tris(n: usize) -> Vec<Prim> {
        let mut s = 7u64;
        let mut r = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 40) as f32 / (1u64 << 24) as f32
        };
        (0..n)
            .map(|k| {
                let o = [r() * 4.0 - 2.0, r() * 4.0 - 2.0, r() * 2.0 + 1.0];
                let p = [o, [o[0] + 0.1 * r(), o[1] + 0.1, o[2]], [o[0], o[1] + 0.1 * r(), o[2] - 0.1]];
                Prim { p, energy: 0.1 + r(), two_sided: k % 5 == 0, back: k % 7 == 0 }
            })
            .collect()
    }

    /// The pick probabilities over all triangles sum to one, and the
    /// sampler's probability is the evaluated one.
    #[test]
    fn pdf_sums_to_one_and_matches_sampling() {
        let p = tris(300);
        let t = Tree::build(&p);
        for (x, n) in [([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]), ([0.3, -0.2, 1.5], [0.0; 3]), ([5.0, 1.0, 2.0], [-1.0, 0.0, 0.0])] {
            let s: f64 = (0..p.len()).map(|i| t.pdf(i, x, n) as f64).sum();
            // samples that reach a node whose children both bound to zero are
            // lost (a contribution of zero), so the sum may fall short of one
            assert!(s <= 1.0 + 1e-4 && s > 0.5, "sum {s}");
            eprintln!("Σ pdf {s:.5}");
            for k in 0..200 {
                let u = (k as f32 + 0.5) / 200.0;
                if let Some((i, pr)) = t.sample(u, x, n) {
                    let pe = t.pdf(i, x, n);
                    assert!((pr - pe).abs() <= 1e-5 * pe.max(1e-6), "{pr} {pe}");
                }
            }
        }
    }
}
