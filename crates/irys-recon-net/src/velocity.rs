//! Analytic vertex velocities `∂X/∂θ` for every family (v3, piece 2).
//!
//! The scene families' meshers ([`crate::scene::mesh_fast`]) are restated
//! here generically over a scalar [`Sc`], with every topological choice
//! (key-cap columns, honeycomb cells, a clamp's active side, ear clipping)
//! made on the value, and run once with forward-mode duals [`D`] whose
//! tangents are the eight parameter slots and the pose `(x, y, yaw)`. The
//! result is the exact derivative of the piecewise-smooth mesher, vertex by
//! vertex; [`tests`] check the f64 instance against `mesh_fast` bit for
//! bit and the duals against central differences.

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::scene::{Family, NP, Object};

/// Tangent slots: the parameters, then pose x, y (mm) and yaw (degrees).
pub const NV: usize = NP + 3;

pub trait Sc: Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self> {
    fn c(v: f64) -> Self;
    fn v(self) -> f64;
}

impl Sc for f64 {
    fn c(v: f64) -> f64 {
        v
    }
    fn v(self) -> f64 {
        self
    }
}

/// A forward-mode dual with [`NV`] tangents.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct D {
    pub v: f64,
    pub d: [f64; NV],
}

impl D {
    pub fn var(v: f64, k: usize) -> D {
        let mut d = [0.0; NV];
        d[k] = 1.0;
        D { v, d }
    }
}

impl Add for D {
    type Output = D;
    fn add(self, o: D) -> D {
        D { v: self.v + o.v, d: std::array::from_fn(|k| self.d[k] + o.d[k]) }
    }
}
impl Sub for D {
    type Output = D;
    fn sub(self, o: D) -> D {
        D { v: self.v - o.v, d: std::array::from_fn(|k| self.d[k] - o.d[k]) }
    }
}
impl Mul for D {
    type Output = D;
    fn mul(self, o: D) -> D {
        D { v: self.v * o.v, d: std::array::from_fn(|k| self.d[k] * o.v + self.v * o.d[k]) }
    }
}
impl Div for D {
    type Output = D;
    fn div(self, o: D) -> D {
        let q = self.v / o.v;
        D { v: q, d: std::array::from_fn(|k| (self.d[k] - q * o.d[k]) / o.v) }
    }
}
impl Neg for D {
    type Output = D;
    fn neg(self) -> D {
        D { v: -self.v, d: self.d.map(|x| -x) }
    }
}
impl Sc for D {
    fn c(v: f64) -> D {
        D { v, d: [0.0; NV] }
    }
    fn v(self) -> f64 {
        self.v
    }
}

fn mx<T: Sc>(a: T, b: T) -> T {
    if a.v() >= b.v() { a } else { b }
}
fn mn<T: Sc>(a: T, b: T) -> T {
    if a.v() <= b.v() { a } else { b }
}
fn cl<T: Sc>(a: T, lo: T, hi: T) -> T {
    // f64::clamp: lo if a < lo, hi if a > hi.
    if a.v() < lo.v() {
        lo
    } else if a.v() > hi.v() {
        hi
    } else {
        a
    }
}
fn k<T: Sc>(v: f64) -> T {
    T::c(v)
}

/// A mesh over `T`.
#[derive(Debug, Clone)]
pub struct TrisG<T> {
    pub p: Vec<[T; 3]>,
    pub t: Vec<[u32; 3]>,
}

impl<T: Sc> TrisG<T> {
    fn new() -> Self {
        TrisG { p: Vec::new(), t: Vec::new() }
    }
    fn append(&mut self, o: TrisG<T>) {
        let n = self.p.len() as u32;
        self.p.extend(o.p);
        self.t.extend(o.t.into_iter().map(|[a, b, c]| [a + n, b + n, c + n]));
    }
    /// Placed by `pose` (x, y mm; yaw degrees) on the plane.
    pub fn posed(&self, pose: [T; 3]) -> TrisG<T> {
        let a = pose[2] * k(std::f64::consts::PI / 180.0);
        let (s, c) = sin_cos(a);
        TrisG { p: self.p.iter().map(|q| [c * q[0] - s * q[1] + pose[0], s * q[0] + c * q[1] + pose[1], q[2]]).collect(), t: self.t.clone() }
    }
}

/// `sin`, `cos` of a scalar (the only transcendental the meshers take of
/// a variable: the yaw).
fn sin_cos<T: Sc>(a: T) -> (T, T) {
    let (s, c) = a.v().sin_cos();
    // d sin = cos da, d cos = −sin da: (s, c) + linear terms in (a − a.v).
    let da = a - k(a.v());
    (k::<T>(s) + k::<T>(c) * da, k::<T>(c) - k::<T>(s) * da)
}

fn prism<T: Sc>(poly: &[[T; 2]], z0: T, z1: T) -> TrisG<T> {
    let n = poly.len();
    let mut m = TrisG::new();
    for q in poly {
        m.p.push([q[0], q[1], z0]);
    }
    for q in poly {
        m.p.push([q[0], q[1], z1]);
    }
    let (cx, cy) = poly.iter().fold((k::<T>(0.0), k::<T>(0.0)), |a, q| (a.0 + q[0] / k(n as f64), a.1 + q[1] / k(n as f64)));
    let b = m.p.len() as u32;
    m.p.push([cx, cy, z0]);
    m.p.push([cx, cy, z1]);
    for i in 0..n {
        let j = (i + 1) % n;
        let (i0, j0, i1, j1) = (i as u32, j as u32, (i + n) as u32, (j + n) as u32);
        m.t.push([i0, j0, j1]);
        m.t.push([i0, j1, i1]);
        m.t.push([b, j0, i0]);
        m.t.push([b + 1, i1, j1]);
    }
    m
}

fn rounded_rect<T: Sc>(w: T, d: T, r: T, seg: usize) -> Vec<[T; 2]> {
    let r = cl(r, k(0.0), k::<T>(0.499) * mn(w, d));
    let (a, b) = (w / k(2.0) - r, d / k(2.0) - r);
    let mut v = Vec::new();
    for (kk, (cx, cy)) in [(a, -b), (a, b), (-a, b), (-a, -b)].into_iter().enumerate() {
        let start = -std::f64::consts::FRAC_PI_2 + kk as f64 * std::f64::consts::FRAC_PI_2;
        let n = if r.v() > 0.0 { seg } else { 0 };
        for i in 0..=n {
            let t = start + std::f64::consts::FRAC_PI_2 * i as f64 / n.max(1) as f64;
            v.push([cx + r * k(t.cos()), cy + r * k(t.sin())]);
            if n == 0 {
                break;
            }
        }
    }
    v
}

fn cuboid<T: Sc>(cx: T, cy: T, w: T, d: T, z0: T, z1: T) -> TrisG<T> {
    let (a, b) = (w / k(2.0), d / k(2.0));
    prism(&[[cx - a, cy - b], [cx + a, cy - b], [cx + a, cy + b], [cx - a, cy + b]], z0, z1)
}

fn rounded_box<T: Sc>(w: T, d: T, h: T, r: T, seg: usize) -> TrisG<T> {
    let r = cl(r, k(0.01), k::<T>(0.499) * mn(mn(w, d), h));
    let (a, b, c) = (w / k(2.0) - r, d / k(2.0) - r, h / k(2.0) - r);
    let mut m = TrisG::new();
    let mut lons = Vec::new();
    for q in 0..4 {
        for i in 0..=seg {
            lons.push((q, (q as f64 + i as f64 / seg as f64) * std::f64::consts::FRAC_PI_2));
        }
    }
    let mut lats = Vec::new();
    for hsign in [-1.0f64, 1.0] {
        for j in 0..=seg {
            let t = j as f64 / seg as f64 * std::f64::consts::FRAC_PI_2;
            lats.push((hsign, if hsign < 0.0 { -std::f64::consts::FRAC_PI_2 + t } else { t }));
        }
    }
    for &(hs, lat) in &lats {
        for &(q, lon) in &lons {
            let (x, y, z) = (lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin());
            let sx = if q == 0 || q == 3 { 1.0 } else { -1.0 };
            let sy = if q == 0 || q == 1 { 1.0 } else { -1.0 };
            m.p.push([k::<T>(x) * r + k::<T>(sx) * a, k::<T>(y) * r + k::<T>(sy) * b, k::<T>(z) * r + k::<T>(hs) * c + h / k(2.0)]);
        }
    }
    let cols = lons.len();
    let rows = lats.len();
    for j in 0..rows - 1 {
        for i in 0..cols {
            let i2 = (i + 1) % cols;
            let (p00, p01, p10, p11) = ((j * cols + i) as u32, (j * cols + i2) as u32, ((j + 1) * cols + i) as u32, ((j + 1) * cols + i2) as u32);
            m.t.push([p00, p01, p11]);
            m.t.push([p00, p11, p10]);
        }
    }
    m
}

fn lathe<T: Sc>(h: T, radii: &[T], seg: usize) -> TrisG<T> {
    let kk = radii.len();
    let rows = 16;
    let prof: Vec<(T, T)> = (0..=rows)
        .map(|i| {
            let s = i as f64 / rows as f64 * (kk - 1) as f64;
            let j = (s.floor() as usize).min(kk - 2);
            let t = s - j as f64;
            let p = |m: i64| radii[m.clamp(0, kk as i64 - 1) as usize];
            let (p0, p1, p2, p3) = (p(j as i64 - 1), p(j as i64), p(j as i64 + 1), p(j as i64 + 2));
            let r = k::<T>(0.5)
                * (k::<T>(2.0) * p1
                    + (-p0 + p2) * k(t)
                    + (k::<T>(2.0) * p0 - k::<T>(5.0) * p1 + k::<T>(4.0) * p2 - p3) * k(t * t)
                    + (-p0 + k::<T>(3.0) * p1 - k::<T>(3.0) * p2 + p3) * k(t * t * t));
            (mx(r, k(0.5)), h * k(i as f64 / rows as f64))
        })
        .collect();
    let mut m = TrisG::new();
    for &(r, z) in &prof {
        for i in 0..seg {
            let a = std::f64::consts::TAU * i as f64 / seg as f64;
            m.p.push([r * k(a.cos()), r * k(a.sin()), z]);
        }
    }
    for j in 0..rows {
        for i in 0..seg {
            let i2 = (i + 1) % seg;
            let (a, b, c, d) = ((j * seg + i) as u32, (j * seg + i2) as u32, ((j + 1) * seg + i2) as u32, ((j + 1) * seg + i) as u32);
            m.t.push([a, b, c]);
            m.t.push([a, c, d]);
        }
    }
    let bot = m.p.len() as u32;
    m.p.push([k(0.0), k(0.0), k(0.0)]);
    m.p.push([k(0.0), k(0.0), h]);
    let top_row = (rows * seg) as u32;
    for i in 0..seg {
        let i2 = (i + 1) % seg;
        m.t.push([bot, i2 as u32, i as u32]);
        m.t.push([bot + 1, top_row + i as u32, top_row + i2 as u32]);
    }
    m
}

fn hex<T: Sc>(cx: T, cy: T, r: T) -> Vec<[T; 2]> {
    (0..6)
        .map(|kk| {
            let a = std::f64::consts::FRAC_PI_3 * kk as f64 + std::f64::consts::FRAC_PI_6;
            [cx + r * k(a.cos()), cy + r * k(a.sin())]
        })
        .collect()
}

fn hex_ring<T: Sc>(cx: T, cy: T, cell: T, wall: T, z0: T, z1: T) -> TrisG<T> {
    let ro = cell / k(3f64.sqrt());
    let ri = mx(cell - k::<T>(2.0) * wall, k(0.2)) / k(3f64.sqrt());
    let o = hex(cx, cy, ro);
    let i = hex(cx, cy, ri);
    let mut m = TrisG::new();
    for kk in 0..6 {
        let j = (kk + 1) % 6;
        m.append(prism(&[i[kk], o[kk], o[j], i[j]], z0, z1));
    }
    m
}

/// The plate without booleans ([`crate::scene::plate_fast`]), generic: the
/// polygon's topology (bridges, ear clipping) is computed on the values by
/// `plate_fast` itself, and every vertex is re-expressed by its source.
fn plate<T: Sc>(o: &Object, p: &[T]) -> TrisG<T> {
    let prog = o.plate_program().unwrap();
    let fast = crate::scene::plate_fast(&prog);
    let (w, d, h, r, hr, pitch) = (p[0], p[1], p[2], p[3], p[4], p[5]);
    let fil = if o.params[3] >= 0.6 { mn(r, k::<T>(0.44) * mn(w, d)) } else { k(0.0) };
    let outer = rounded_rect(w, d, fil, 6);
    // Hole centres, as `Object::plate_program` and `Feature::holes` place them.
    let mut holes: Vec<Vec<[T; 2]>> = Vec::new();
    let circle = |cx: T, cy: T, rr: T| -> Vec<[T; 2]> {
        (0..16)
            .map(|kk| {
                let a = std::f64::consts::TAU * kk as f64 / 16.0;
                [cx + rr * k(a.cos()), cy + rr * k(a.sin())]
            })
            .collect()
    };
    if o.tokens[1] == 0 {
        let (n, m) = (o.tokens[2].max(1), o.tokens[3].max(1));
        for j in 0..m {
            let y = k::<T>(j as f64 - (m as f64 - 1.0) / 2.0) * pitch;
            let x0 = k::<T>(-(n as f64 - 1.0) / 2.0) * pitch;
            for i in 0..n {
                let x = if n >= 2 { x0 + k::<T>(i as f64) * pitch } else { x0 };
                holes.push(circle(x, y, hr));
            }
        }
    } else {
        let n = o.tokens[2].max(3);
        for i in 0..n {
            let a = (360.0 * i as f64 / n as f64).to_radians();
            holes.push(circle(pitch * k(a.cos()), pitch * k(a.sin()), hr));
        }
    }
    // Every 2D point plate_fast emits is a copy of an outline or hole point:
    // match by value.
    let mut src: Vec<([f64; 2], [T; 2])> = outer.iter().map(|q| ([q[0].v(), q[1].v()], *q)).collect();
    for hl in &holes {
        src.extend(hl.iter().map(|q| ([q[0].v(), q[1].v()], *q)));
    }
    let find = |x: f64, y: f64| -> [T; 2] {
        src.iter().min_by(|a, b| ((a.0[0] - x).powi(2) + (a.0[1] - y).powi(2)).total_cmp(&((b.0[0] - x).powi(2) + (b.0[1] - y).powi(2)))).unwrap().1
    };
    let pts: Vec<[T; 3]> = fast
        .p
        .iter()
        .map(|q| {
            let xy = find(q[0], q[1]);
            [xy[0], xy[1], if q[2] > 0.5 * prog.h { h } else { k(0.0) }]
        })
        .collect();
    TrisG { p: pts, t: fast.t }
}

/// The object's mesh, object frame, over `T` (`p`: the parameter slots).
pub fn mesh_g<T: Sc>(o: &Object, p: &[T]) -> TrisG<T> {
    let seg = 6;
    match o.family {
        Family::Plate => plate(o, p),
        Family::Lathe => lathe(p[0], &p[1..5], 32),
        Family::Product => {
            let (w, d, h, r) = (p[0], p[1], p[2], p[3]);
            let z0 = if o.tokens[2] == 1 { p[4] } else { k(0.0) };
            let mut m = if o.tokens[1] == 0 {
                prism(&rounded_rect(w, d, r, seg), z0, z0 + h)
            } else {
                let mut b = rounded_box(w, d, h, r, seg);
                for q in &mut b.p {
                    q[2] = q[2] + z0;
                }
                b
            };
            if o.tokens[2] == 1 {
                let neck = cl(k::<T>(0.12) * w, k(30.0), k(80.0));
                m.append(cuboid(k(0.0), d * k(0.5) + k(8.0), neck, k(14.0), k(4.0), z0 + k::<T>(0.4) * h));
                let foot = prism(&rounded_rect(cl(k::<T>(0.4) * w, k(120.0), k(260.0)), mn(k(160.0), k::<T>(0.6) * z0 + k(80.0)), k(20.0), seg), k(0.0), k(6.0));
                m.append(foot.posed([k(0.0), d * k(0.5) + k(8.0), k(0.0)]));
            }
            m
        }
        Family::Grid => {
            let (w, d, h, pitch, kh) = (p[0], p[1], p[2], p[3], p[4]);
            let mut m = prism(&rounded_rect(w, d, k(3.0), 3), k(0.0), h);
            let rows = o.tokens[2].max(1);
            let cols = ((w.v() - 10.0) / pitch.v()).floor().max(1.0) as usize;
            let cap = k::<T>(0.8) * pitch;
            for j in 0..rows {
                let y = k::<T>(j as f64 - (rows as f64 - 1.0) / 2.0) * pitch;
                for i in 0..cols {
                    let x = k::<T>(i as f64 - (cols as f64 - 1.0) / 2.0) * pitch;
                    m.append(cuboid(x, y, cap, cap, h - k(0.5), h + kh));
                }
            }
            m
        }
        Family::Shell => {
            let (w, d, h, t, cell) = (p[0], p[1], p[2], p[3], p[4]);
            let two = k::<T>(2.0);
            let mut m = cuboid(k(0.0), k(0.0), w, d, k(0.0), mn(t, h * k(0.3)));
            m.append(cuboid(k(0.0), -d / two + t / two, w, t, k(0.0), h));
            m.append(cuboid(k(0.0), d / two - t / two, w, t, k(0.0), h));
            m.append(cuboid(-w / two + t / two, k(0.0), t, d, k(0.0), h));
            m.append(cuboid(w / two - t / two, k(0.0), t, d, k(0.0), h));
            if o.tokens[1] == 0 {
                let ribs = o.tokens[2];
                for kk in 0..ribs {
                    let x = -w / two + w * k((kk as f64 + 1.0) / (ribs as f64 + 1.0));
                    m.append(cuboid(x, k(0.0), t, d - t, k(0.0), k::<T>(0.7) * h));
                }
            } else {
                let lid = mx(two * t, k(2.0));
                let dx = cell;
                let dy = cell * k(3f64.sqrt() / 2.0);
                let ro = cell / k(3f64.sqrt());
                let ny = ((d.v() - 2.0 * t.v()) / dy.v()).floor() as i64;
                let nx = ((w.v() - 2.0 * t.v()) / dx.v()).floor() as i64;
                let mut any = false;
                for j in -ny..=ny {
                    for i in -nx..=nx {
                        let x = k::<T>(i as f64) * dx + if j.rem_euclid(2) == 1 { dx / two } else { k(0.0) };
                        let y = k::<T>(j as f64) * dy;
                        if x.v().abs() + ro.v() < w.v() / 2.0 - t.v() && y.v().abs() + ro.v() < d.v() / 2.0 - t.v() {
                            m.append(hex_ring(x, y, cell, k::<T>(0.5) * mx(t, k(1.0)), h - lid, h));
                            any = true;
                        }
                    }
                }
                if !any {
                    m.append(cuboid(k(0.0), k(0.0), w, d, h - lid, h));
                }
            }
            m
        }
    }
}

/// World vertices and `∂X/∂θ` for every tangent slot ([`NV`]: parameters,
/// then pose x, y, yaw), with the triangles.
pub fn velocities(o: &Object) -> (Vec<[f64; 3]>, Vec<[[f64; 3]; NV]>, Vec<[u32; 3]>) {
    let p: Vec<D> = (0..o.params.len()).map(|i| D::var(o.params[i], i)).collect();
    let pose = [D::var(o.pose[0], NP), D::var(o.pose[1], NP + 1), D::var(o.pose[2], NP + 2)];
    let m = mesh_g(o, &p).posed(pose);
    let x = m.p.iter().map(|q| [q[0].v, q[1].v, q[2].v]).collect();
    let v = m.p.iter().map(|q| std::array::from_fn(|kk| [q[0].d[kk], q[1].d[kk], q[2].d[kk]])).collect();
    (x, v, m.t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{CLASS_LIST, mesh_fast, sample_object};
    use irys_recon::jac::Mix;

    #[test]
    fn the_generic_mesher_is_the_mesher() {
        let mut r = Mix(17);
        for _ in 0..6 {
            for c in CLASS_LIST {
                let o = sample_object(&mut r, c);
                let want = mesh_fast(&o).unwrap();
                let got = mesh_g::<f64>(&o, &o.params);
                assert_eq!(got.t, want.t, "{c:?} triangles");
                for (a, b) in got.p.iter().zip(&want.p) {
                    for i in 0..3 {
                        assert!((a[i] - b[i]).abs() <= 1e-9 * (1.0 + b[i].abs()), "{c:?}: {a:?} vs {b:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn velocities_match_central_differences() {
        let mut r = Mix(23);
        for _ in 0..4 {
            for c in CLASS_LIST {
                let mut o = sample_object(&mut r, c);
                o.pose = [12.0, -30.0, 25.0];
                let (x, v, t) = velocities(&o);
                for kk in 0..NV {
                    let used = kk >= NP || (kk < o.params.len() && crate::scene::param_used(o.family, &o.tokens, kk));
                    if !used {
                        continue;
                    }
                    let h = if kk >= NP { 1e-4 } else { 1e-5 * o.params[kk].max(1.0) };
                    let bump = |s: f64| {
                        let mut q = o.clone();
                        if kk < NP {
                            q.params[kk] += s * h;
                        } else {
                            q.pose[kk - NP] += s * h;
                        }
                        let m = mesh_g::<f64>(&q, &q.params).posed([q.pose[0], q.pose[1], q.pose[2]]);
                        (m.p, m.t)
                    };
                    let (pp, tp) = bump(1.0);
                    let (pm, tm) = bump(-1.0);
                    if tp != t || tm != t {
                        continue; // a topological change inside the step
                    }
                    for i in 0..x.len() {
                        for a in 0..3 {
                            let fd = (pp[i][a] - pm[i][a]) / (2.0 * h);
                            assert!((fd - v[i][kk][a]).abs() < 1e-5 * (1.0 + fd.abs()), "{c:?} slot {kk} vertex {i} axis {a}: dual {} fd {fd}", v[i][kk][a]);
                        }
                    }
                }
            }
        }
    }
}
