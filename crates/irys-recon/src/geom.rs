//! From a program to triangles, and how every vertex moves with every
//! continuous parameter.
//!
//! # d(vertices)/d(θ): analytic, by carrier surfaces
//!
//! vcad's tessellation does not keep its topology under a perturbation of
//! the program (a boolean re-splits its faces differently), so finite
//! differences of vertex positions are not available in general. Instead
//! every face the DSL can make lies on an analytic carrier — a plane or a
//! vertical cylinder — whose placement is a closed form in θ ([`carriers`]):
//! the box's sides and top, the fillet cylinders and chamfer planes, each
//! hole's, bore's and slot's walls, the counterbore floor, the boss. A
//! carrier moves along its normal at `s_j = ∂(offset)/∂θ_j` (a cylinder:
//! `n · ∂c/∂θ_j + ∂r/∂θ_j`). A vertex lies on one to three carriers; its
//! velocity is the least-norm `v` with `n_i · v = s_i` for all of them —
//! exactly the normal motion of every face through it, with no tangential
//! slide (which no image term can see). A cylinder's polygon vertices sit on
//! the true circle, so its chords move with it.
//!
//! [`crate::jac`] checks the edge term built on these velocities against
//! central differences of the rendered loss, where vcad re-tessellates at
//! every step.

use irys_core::{Point3, TriMesh, Vec3};
use vcad_kernel::Solid;

use crate::dsl::{Axis, Corner, Feature, Program};

/// Positions and triangles of `solid`, vcad's tessellation, millimetres.
pub fn mesh(solid: &Solid) -> (Vec<Point3>, Vec<[u32; 3]>) {
    let m = solid.to_mesh(crate::dsl::SEGMENTS);
    let p = m
        .vertices
        .chunks_exact(3)
        .map(|v| Point3::new(v[0] as f64, v[1] as f64, v[2] as f64))
        .collect();
    let t = m.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
    (p, t)
}

/// An irys triangle mesh (flat shaded: no normals).
pub fn trimesh(p: &[Point3], t: &[[u32; 3]]) -> TriMesh {
    let idx: Vec<u32> = t.iter().flatten().copied().collect();
    TriMesh::new(p.to_vec(), Vec::new(), &idx)
}

/// A carrier surface and its normal speed per parameter.
#[derive(Debug, Clone)]
pub enum Carrier {
    /// `n · p = o`, for points in the box `lo ..= hi` (and, when `ring` is
    /// set, `ρ_min ≤ |p_xy − c| ≤ ρ_max`).
    Plane {
        n: Vec3,
        o: f64,
        lo: [f64; 3],
        hi: [f64; 3],
        ring: Option<([f64; 2], f64, f64)>,
        speed: Vec<(usize, f64)>,
    },
    /// The vertical cylinder `|p_xy − c| = r`, `z ∈ [z0, z1]`, restricted to
    /// the quadrant `quad` when set.
    Cylinder {
        c: [f64; 2],
        r: f64,
        z0: f64,
        z1: f64,
        quad: Option<[f64; 2]>,
        /// Per parameter: `(index, ∂c, ∂r)`.
        speed: Vec<(usize, [f64; 2], f64)>,
    },
}

const TOL: f64 = 2e-3;

impl Carrier {
    fn on(&self, p: Point3) -> bool {
        match self {
            Carrier::Plane { n, o, lo, hi, ring, .. } => {
                if (n.dot(&p.coords()) - o).abs() > TOL {
                    return false;
                }
                let q = [p.x, p.y, p.z];
                if (0..3).any(|k| q[k] < lo[k] - TOL || q[k] > hi[k] + TOL) {
                    return false;
                }
                match ring {
                    None => true,
                    Some((c, a, b)) => {
                        let rho = (p.x - c[0]).hypot(p.y - c[1]);
                        rho > a - TOL && rho < b + TOL
                    }
                }
            }
            Carrier::Cylinder { c, r, z0, z1, quad, .. } => {
                let (dx, dy) = (p.x - c[0], p.y - c[1]);
                // Booleans leave points on the polygon's chords too: allow the sagitta.
                let sag = r * (1.0 - (std::f64::consts::PI / crate::dsl::SEGMENTS as f64).cos());
                let rho = dx.hypot(dy);
                if rho < r - sag - TOL || rho > r + TOL || p.z < z0 - TOL || p.z > z1 + TOL {
                    return false;
                }
                match quad {
                    None => true,
                    Some(q) => dx * q[0] >= -TOL && dy * q[1] >= -TOL,
                }
            }
        }
    }

    /// `(n, [(param, s)])` at `p`.
    fn normal_speed(&self, p: Point3) -> (Vec3, Vec<(usize, f64)>) {
        match self {
            Carrier::Plane { n, speed, .. } => (*n, speed.clone()),
            Carrier::Cylinder { c, speed, .. } => {
                let d = Vec3::new(p.x - c[0], p.y - c[1], 0.0);
                let n = d / d.norm().max(1e-12);
                (n, speed.iter().map(|&(j, dc, dr)| (j, n.x * dc[0] + n.y * dc[1] + dr)).collect())
            }
        }
    }
}

/// Every carrier `prog` makes, with its normal speeds.
pub fn carriers(prog: &Program) -> Vec<Carrier> {
    let (a, b, h) = (prog.w / 2.0, prog.d / 2.0, prog.h);
    let (iw, id, ih) = (0usize, 1usize, 2usize);
    let mut v = Vec::new();
    let blend = match prog.corner {
        Corner::Sharp => 0.0,
        Corner::Fillet(r) | Corner::Chamfer(r) => r,
    };
    let big = 1e4;
    // Sides: ±x moves with w/2, ±y with d/2.
    for s in [1.0, -1.0] {
        v.push(Carrier::Plane {
            n: Vec3::new(s, 0.0, 0.0),
            o: a,
            lo: [-big, -b + blend, 0.0],
            hi: [big, b - blend, h],
            ring: None,
            speed: vec![(iw, 0.5)],
        });
        v.push(Carrier::Plane {
            n: Vec3::new(0.0, s, 0.0),
            o: b,
            lo: [-a + blend, -big, 0.0],
            hi: [a - blend, big, h],
            ring: None,
            speed: vec![(id, 0.5)],
        });
    }
    v.push(Carrier::Plane {
        n: Vec3::new(0.0, 0.0, 1.0),
        o: h,
        lo: [-a, -b, -big],
        hi: [a, b, big],
        ring: None,
        speed: vec![(ih, 1.0)],
    });
    match prog.corner {
        Corner::Sharp => {}
        Corner::Fillet(r) => {
            for sx in [1.0, -1.0] {
                for sy in [1.0, -1.0] {
                    v.push(Carrier::Cylinder {
                        c: [sx * (a - r), sy * (b - r)],
                        r,
                        z0: 0.0,
                        z1: h,
                        quad: Some([sx, sy]),
                        speed: vec![(iw, [sx * 0.5, 0.0], 0.0), (id, [0.0, sy * 0.5], 0.0), (3, [-sx, -sy], 1.0)],
                    });
                }
            }
        }
        Corner::Chamfer(c) => {
            let k = std::f64::consts::FRAC_1_SQRT_2;
            for sx in [1.0, -1.0] {
                for sy in [1.0, -1.0] {
                    let (lo, hi) = (
                        [if sx > 0.0 { a - c } else { -a }, if sy > 0.0 { b - c } else { -b }, 0.0],
                        [if sx > 0.0 { a } else { -a + c }, if sy > 0.0 { b } else { -b + c }, h],
                    );
                    v.push(Carrier::Plane {
                        n: Vec3::new(sx * k, sy * k, 0.0),
                        o: (a + b - c) * k,
                        lo,
                        hi,
                        ring: None,
                        speed: vec![(iw, 0.5 * k), (id, 0.5 * k), (3, -k)],
                    });
                }
            }
        }
    }
    let mut j = if blend > 0.0 { 4 } else { 3 };
    for f in &prog.features {
        let np = f.params().len();
        let hole = |v: &mut Vec<Carrier>, c: [f64; 2], r: f64, z1: f64, dc: Vec<(usize, [f64; 2])>, ir: usize| {
            let mut speed: Vec<(usize, [f64; 2], f64)> = dc.into_iter().map(|(k, d)| (k, d, 0.0)).collect();
            speed.push((ir, [0.0, 0.0], 1.0));
            v.push(Carrier::Cylinder { c, r, z0: 0.0, z1, quad: None, speed });
        };
        match *f {
            Feature::Hole { x, y, r } => hole(&mut v, [x, y], r, h, vec![(j, [1.0, 0.0]), (j + 1, [0.0, 1.0])], j + 2),
            Feature::Counterbore { x, y, r, cr, cd } => {
                hole(&mut v, [x, y], r, h - cd, vec![(j, [1.0, 0.0]), (j + 1, [0.0, 1.0])], j + 2);
                v.push(Carrier::Cylinder {
                    c: [x, y],
                    r: cr,
                    z0: h - cd,
                    z1: h,
                    quad: None,
                    speed: vec![(j, [1.0, 0.0], 0.0), (j + 1, [0.0, 1.0], 0.0), (j + 3, [0.0, 0.0], 1.0)],
                });
                v.push(Carrier::Plane {
                    n: Vec3::new(0.0, 0.0, 1.0),
                    o: h - cd,
                    lo: [x - cr, y - cr, -big],
                    hi: [x + cr, y + cr, big],
                    ring: Some(([x, y], r, cr)),
                    speed: vec![(ih, 1.0), (j + 4, -1.0)],
                });
            }
            Feature::Slot { x, y, len, r, axis } => {
                let u = if axis == Axis::X { [1.0, 0.0] } else { [0.0, 1.0] };
                for s in [1.0, -1.0] {
                    let c = [x + s * u[0] * len / 2.0, y + s * u[1] * len / 2.0];
                    let half = [s * u[0] * 0.5, s * u[1] * 0.5];
                    v.push(Carrier::Cylinder {
                        c,
                        r,
                        z0: 0.0,
                        z1: h,
                        // The outer half only: the inner half is cut away.
                        quad: None,
                        speed: vec![(j, [1.0, 0.0], 0.0), (j + 1, [0.0, 1.0], 0.0), (j + 2, half, 0.0), (j + 3, [0.0, 0.0], 1.0)],
                    });
                    // Side walls: n = s·w (w ⟂ u), o = s·(w·c0) + r.
                    let w = [-u[1], u[0]];
                    let n = Vec3::new(s * w[0], s * w[1], 0.0);
                    let o = s * (w[0] * x + w[1] * y) + r;
                    let (lo, hi) = if axis == Axis::X {
                        ([x - len / 2.0, -big, 0.0], [x + len / 2.0, big, h])
                    } else {
                        ([-big, y - len / 2.0, 0.0], [big, y + len / 2.0, h])
                    };
                    v.push(Carrier::Plane {
                        n,
                        o,
                        lo,
                        hi,
                        ring: None,
                        speed: vec![(j, s * w[0]), (j + 1, s * w[1]), (j + 3, 1.0)],
                    });
                }
            }
            Feature::Boss { x, y, r, bh } => {
                v.push(Carrier::Cylinder {
                    c: [x, y],
                    r,
                    z0: h,
                    z1: h + bh,
                    quad: None,
                    speed: vec![(j, [1.0, 0.0], 0.0), (j + 1, [0.0, 1.0], 0.0), (j + 2, [0.0, 0.0], 1.0)],
                });
                v.push(Carrier::Plane {
                    n: Vec3::new(0.0, 0.0, 1.0),
                    o: h + bh,
                    lo: [x - r, y - r, -big],
                    hi: [x + r, y + r, big],
                    ring: Some(([x, y], 0.0, r)),
                    speed: vec![(ih, 1.0), (j + 3, 1.0)],
                });
            }
            Feature::LinearHoles { x, y, dx, dy, r, n } => {
                for i in 0..n {
                    let k = i as f64;
                    hole(
                        &mut v,
                        [x + k * dx, y + k * dy],
                        r,
                        h,
                        vec![(j, [1.0, 0.0]), (j + 1, [0.0, 1.0]), (j + 2, [k, 0.0]), (j + 3, [0.0, k])],
                        j + 4,
                    );
                }
            }
            Feature::CircularHoles { x, y, rad, phase, r, n } => {
                for i in 0..n {
                    let ang = (phase + 360.0 * i as f64 / n as f64).to_radians();
                    let (s, c) = ang.sin_cos();
                    let d2r = std::f64::consts::PI / 180.0;
                    hole(
                        &mut v,
                        [x + rad * c, y + rad * s],
                        r,
                        h,
                        vec![(j, [1.0, 0.0]), (j + 1, [0.0, 1.0]), (j + 2, [c, s]), (j + 3, [-rad * s * d2r, rad * c * d2r])],
                        j + 4,
                    );
                }
            }
        }
        j += np;
    }
    v
}

/// Symmetric 3 × 3 pseudo-inverse times `b` (Jacobi eigenvalues).
fn pinv_solve(m: [[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    let mut a = m;
    let mut q = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..30 {
        for (p, r) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][r].abs() < 1e-14 {
                continue;
            }
            let th = 0.5 * (2.0 * a[p][r]).atan2(a[r][r] - a[p][p]);
            let (s, c) = th.sin_cos();
            for k in 0..3 {
                let (x, y) = (a[k][p], a[k][r]);
                a[k][p] = c * x - s * y;
                a[k][r] = s * x + c * y;
            }
            for k in 0..3 {
                let (x, y) = (a[p][k], a[r][k]);
                a[p][k] = c * x - s * y;
                a[r][k] = s * x + c * y;
            }
            for row in q.iter_mut() {
                let (x, y) = (row[p], row[r]);
                row[p] = c * x - s * y;
                row[r] = s * x + c * y;
            }
        }
    }
    let mut out = [0.0; 3];
    for e in 0..3 {
        let lam = a[e][e];
        if lam.abs() < 1e-6 {
            continue;
        }
        let qb: f64 = (0..3).map(|k| q[k][e] * b[k]).sum();
        for k in 0..3 {
            out[k] += q[k][e] * qb / lam;
        }
    }
    out
}

/// `∂p/∂θ_j` for every vertex `p` of `prog`'s tessellation: `[vertex][j]`.
pub fn velocities(prog: &Program, positions: &[Point3]) -> Vec<Vec<Vec3>> {
    let cs = carriers(prog);
    let np = prog.params().len();
    positions
        .iter()
        .map(|&p| {
            let on: Vec<(Vec3, Vec<(usize, f64)>)> =
                cs.iter().filter(|c| c.on(p)).map(|c| c.normal_speed(p)).collect();
            let mut m = [[0.0; 3]; 3];
            for (n, _) in &on {
                let n = [n.x, n.y, n.z];
                for r in 0..3 {
                    for c in 0..3 {
                        m[r][c] += n[r] * n[c];
                    }
                }
            }
            (0..np)
                .map(|j| {
                    let mut b = [0.0; 3];
                    let mut any = false;
                    for (n, sp) in &on {
                        let s: f64 = sp.iter().filter(|(k, _)| *k == j).map(|(_, s)| s).sum();
                        if s != 0.0 {
                            any = true;
                        }
                        b[0] += n.x * s;
                        b[1] += n.y * s;
                        b[2] += n.z * s;
                    }
                    if !any {
                        return Vec3::zeros();
                    }
                    let v = pinv_solve(m, b);
                    Vec3::new(v[0], v[1], v[2])
                })
                .collect()
        })
        .collect()
}

/// Vertices on no carrier, for tests (every vertex of a DSL part should be
/// on at least one, the bottom face excepted).
pub fn orphans(prog: &Program, positions: &[Point3]) -> usize {
    let cs = carriers(prog);
    positions
        .iter()
        .filter(|p| p.z.abs() > TOL && !cs.iter().any(|c| c.on(**p)))
        .count()
}
