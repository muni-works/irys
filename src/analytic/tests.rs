//! What "exact" has to mean.
//!
//! The bar throughout is the *implicit*: a reported hit is a point, and that
//! point either satisfies the shape's own equation or it does not. Testing
//! `t` against a hand-computed number only checks the cases someone thought
//! of; testing the residual checks every ray the sweep fires.
//!
//! Distance-form implicits, not squared ones — `|p − c| − r`, not
//! `|p − c|² − r²`. At a world coordinate of `1e4` the squared form's own
//! evaluation loses eight digits to cancellation before the intersector is
//! implicated in anything, and the tolerance would be measuring `f64`, not
//! the solve.

use super::*;
use crate::bvh::Bvh;

fn ray(from: [f64; 3], dir: [f64; 3]) -> Ray {
    Ray::new(
        Point3::new(from[0], from[1], from[2]),
        Vec3::new(dir[0], dir[1], dir[2]),
    )
}

fn dir(v: [f64; 3]) -> Dir3 {
    Dir3::new_normalize(Vec3::new(v[0], v[1], v[2]))
}

fn p(v: [f64; 3]) -> Point3 {
    Point3::new(v[0], v[1], v[2])
}

/// The signed distance-ish implicit of each primitive, zero on the surface.
fn implicit(prim: &Prim, q: Point3) -> f64 {
    match *prim {
        Prim::Sphere { center, radius } => (q - center).norm() - radius,
        Prim::Box { center, half, rot } => {
            // The face the point is on: the largest of the three slab
            // residuals, which is zero exactly on the boundary.
            let l = rot.to_local(q - center).as_array();
            let h = half.as_array();
            (0..3)
                .map(|k| l[k].abs() - h[k])
                .fold(f64::NEG_INFINITY, f64::max)
        }
        Prim::Cylinder {
            center,
            axis,
            radius,
            half_height,
        } => {
            let l = Frame::from_z(axis).to_local(q - center);
            let radial = (l.x * l.x + l.y * l.y).sqrt() - radius;
            radial.max(l.z.abs() - half_height)
        }
        Prim::Cone {
            center,
            axis,
            radius_base,
            radius_top,
            half_height,
        } => {
            let l = Frame::from_z(axis).to_local(q - center);
            let m = (radius_top - radius_base) / (2.0 * half_height);
            let r_here = m * l.z + 0.5 * (radius_top + radius_base);
            let radial = (l.x * l.x + l.y * l.y).sqrt() - r_here;
            radial.max(l.z.abs() - half_height)
        }
        Prim::Torus {
            center,
            axis,
            major,
            minor,
        } => {
            let l = Frame::from_z(axis).to_local(q - center);
            let s = (l.x * l.x + l.y * l.y).sqrt() - major;
            (s * s + l.z * l.z).sqrt() - minor
        }
        Prim::Plane { point, normal } => (q - point).dot(*normal.as_ref()),
        Prim::Disc { center, normal, .. } => (q - center).dot(*normal.as_ref()),
        Prim::Ellipsoid { center, rot, semi } => {
            let l = componentwise_div(rot.to_local(q - center), semi);
            l.norm() - 1.0
        }
    }
}

/// The gradient of `implicit`, by central differences. The step is scaled to
/// the coordinate so it survives at `1e4`.
fn numeric_gradient(prim: &Prim, q: Point3) -> Vec3 {
    let scale = q.x.abs().max(q.y.abs()).max(q.z.abs()).max(1.0);
    let h = 1e-6 * scale;
    let axis = |i: usize| {
        let mut e = [0.0; 3];
        e[i] = h;
        let e = Vec3::new(e[0], e[1], e[2]);
        (implicit(prim, q + e) - implicit(prim, q - e)) / (2.0 * h)
    };
    Vec3::new(axis(0), axis(1), axis(2))
}

/// A spread of directions from `eye` towards `at`, fanned by `spread`.
fn fan(eye: Point3, at: Point3, spread: f64, n: usize) -> Vec<Ray> {
    let base = Dir3::new_normalize(at - eye);
    let f = Frame::from_z(base);
    let mut out = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let a = spread * (2.0 * (i as f64 + 0.5) / n as f64 - 1.0);
            let b = spread * (2.0 * (j as f64 + 0.5) / n as f64 - 1.0);
            out.push(Ray::new(eye, *base.as_ref() + f.x * a + f.y * b));
        }
    }
    out
}

/// Fire a fan at a primitive and assert every hit is really on it, that the
/// reported normal is the implicit's gradient, and that enough rays landed
/// for the assertion to have meant something.
fn sweep(prim: &Prim, eye: Point3, at: Point3, spread: f64, n: usize, tol: f64) -> usize {
    let mut hits = 0;
    for r in fan(eye, at, spread, n) {
        let mut all = Vec::new();
        prim.hits(&r, 0, 0.0, f64::INFINITY, &mut all);
        for h in &all {
            hits += 1;
            let resid = implicit(prim, h.point);
            assert!(
                resid.abs() < tol,
                "off the surface by {resid:e} at t = {} on {prim:?}",
                h.t
            );
            let g = numeric_gradient(prim, h.point);
            if g.norm() > 1e-6 {
                let g = Dir3::new_normalize(g);
                let dotp = h.normal.as_ref().dot(*g.as_ref());
                assert!(
                    dotp > 0.999,
                    "normal {:?} is not the gradient {:?} (cos = {dotp})",
                    h.normal.as_ref(),
                    g.as_ref()
                );
            }
        }
    }
    hits
}

// ---------------------------------------------------------------------------
// Each primitive, at unit scale and at a millimetre scene's 1e4
// ---------------------------------------------------------------------------

/// Offsets under test: the origin, and the far corner of a regulation court
/// measured in millimetres.
const OFFSETS: [[f64; 3]; 2] = [[0.0, 0.0, 0.0], [1e4, -1e4, 1e4]];

#[test]
fn a_sphere_is_exact_at_both_scales() {
    for o in OFFSETS {
        let c = p([o[0], o[1], o[2] + 0.0]);
        let prim = Prim::Sphere {
            center: c,
            radius: 74.0,
        };
        let eye = c + Vec3::new(-900.0, 400.0, 250.0);
        let n = sweep(&prim, eye, c, 0.09, 24, 1e-9);
        assert!(n > 100, "only {n} hits — the sweep missed");
    }
}

#[test]
fn a_box_is_exact_at_both_scales() {
    for o in OFFSETS {
        let c = p(o);
        let prim = Prim::Box {
            center: c,
            half: Vec3::new(60.0, 30.0, 12.0),
            rot: Frame::from_z(dir([0.3, -0.4, 0.866])),
        };
        let eye = c + Vec3::new(-700.0, 300.0, 500.0);
        let n = sweep(&prim, eye, c, 0.07, 24, 1e-9);
        assert!(n > 100, "only {n} hits");
    }
}

#[test]
fn a_capped_cylinder_is_exact_at_both_scales() {
    for o in OFFSETS {
        let c = p(o);
        let prim = Prim::Cylinder {
            center: c,
            axis: dir([0.2, 0.3, 1.0]),
            radius: 20.0,
            half_height: 90.0,
        };
        let eye = c + Vec3::new(-800.0, 250.0, 300.0);
        let n = sweep(&prim, eye, c, 0.12, 24, 1e-9);
        assert!(n > 100, "only {n} hits");
    }
}

#[test]
fn a_cone_is_exact_at_both_scales() {
    for o in OFFSETS {
        let c = p(o);
        let prim = Prim::Cone {
            center: c,
            axis: dir([0.0, 0.0, 1.0]),
            radius_base: 45.0,
            radius_top: 12.0,
            half_height: 60.0,
        };
        let eye = c + Vec3::new(-600.0, 220.0, 180.0);
        let n = sweep(&prim, eye, c, 0.13, 24, 1e-9);
        assert!(n > 100, "only {n} hits");
    }
}

#[test]
fn an_ellipsoid_is_exact_at_both_scales() {
    for o in OFFSETS {
        let c = p(o);
        let prim = Prim::Ellipsoid {
            center: c,
            rot: Frame::from_z(dir([0.4, 0.1, 0.9])),
            semi: Vec3::new(120.0, 60.0, 30.0),
        };
        let eye = c + Vec3::new(-900.0, 400.0, 260.0);
        let n = sweep(&prim, eye, c, 0.14, 24, 1e-9);
        assert!(n > 100, "only {n} hits");
    }
}

#[test]
fn a_disc_and_a_plane_are_exact_at_both_scales() {
    for o in OFFSETS {
        let c = p(o);
        let normal = dir([0.1, -0.2, 1.0]);
        for prim in [
            Prim::Disc {
                center: c,
                normal,
                radius: 225.0,
            },
            Prim::Plane { point: c, normal },
        ] {
            let eye = c + Vec3::new(-500.0, 300.0, 700.0);
            let n = sweep(&prim, eye, c, 0.24, 16, 1e-9);
            assert!(n > 50, "only {n} hits on {prim:?}");
        }
    }
}

/// The quartic is the one that cannot be held to `1e-9`: Ferrari's
/// reconstruction spends digits the Newton polish can only mostly recover.
/// `1e-6` on a 2 mm tube is a nanometre of surface, which is a great deal
/// finer than anything the renderer above it can see.
#[test]
fn a_torus_is_exact_at_both_scales() {
    for o in OFFSETS {
        let c = p(o);
        let prim = Prim::Torus {
            center: c,
            axis: dir([0.25, -0.15, 1.0]),
            major: 74.0,
            minor: 12.0,
        };
        let eye = c + Vec3::new(-1200.0, 500.0, 400.0);
        let n = sweep(&prim, eye, c, 0.09, 28, 1e-6);
        assert!(n > 100, "only {n} hits — the sweep missed the torus");
    }
}

// ---------------------------------------------------------------------------
// The specific claims
// ---------------------------------------------------------------------------

#[test]
fn a_sphere_hits_where_the_analytic_root_says() {
    let geom = Analytic::sphere(Point3::new(0.0, 0.0, 0.0), 0.5);
    let h = geom
        .intersect(
            &ray([0.0, 0.0, -3.0], [0.0, 0.0, 1.0]),
            0,
            0.0,
            f64::INFINITY,
        )
        .expect("hits the ball");
    assert!((h.t - 2.5).abs() < 1e-12, "t = {}", h.t);
    assert!((h.normal.z + 1.0).abs() < 1e-12, "outward normal");
    assert_eq!(h.payload, 0, "the payload is the prim index");
}

/// A ray starting inside must leave by the far surface — the case a shadow
/// ray from a point *on* the shape depends on, and the one an implementation
/// that only ever reports the near root gets wrong.
#[test]
fn rays_that_start_inside_exit_by_the_far_surface() {
    let cases: [(Prim, f64); 4] = [
        (
            Prim::Sphere {
                center: Point3::origin(),
                radius: 3.0,
            },
            3.0,
        ),
        (
            Prim::Box {
                center: Point3::origin(),
                half: Vec3::new(3.0, 5.0, 7.0),
                rot: Frame::identity(),
            },
            3.0,
        ),
        (
            Prim::Cylinder {
                center: Point3::origin(),
                axis: dir([0.0, 0.0, 1.0]),
                radius: 3.0,
                half_height: 9.0,
            },
            3.0,
        ),
        (
            Prim::Ellipsoid {
                center: Point3::origin(),
                rot: Frame::identity(),
                semi: Vec3::new(3.0, 6.0, 9.0),
            },
            3.0,
        ),
    ];
    for (prim, expect) in cases {
        let h = prim
            .intersect(
                &ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
                0,
                0.0,
                f64::INFINITY,
            )
            .unwrap_or_else(|| panic!("must exit {prim:?}"));
        assert!((h.t - expect).abs() < 1e-12, "t = {} on {prim:?}", h.t);
        // The normal points outward — out of the solid, not back at the ray.
        assert!(h.normal.x > 0.9, "outward on the way out of {prim:?}");
    }
}

/// From inside the ring, a torus is entered and left twice: the case
/// `intersect_all`'s convexity default gets wrong, and the reason this
/// module overrides it.
#[test]
fn a_ray_through_the_ring_finds_all_four_torus_crossings() {
    let geom = Analytic::from_prims(vec![Prim::Torus {
        center: Point3::origin(),
        axis: dir([0.0, 0.0, 1.0]),
        major: 100.0,
        minor: 30.0,
    }]);
    let mut hits = Vec::new();
    geom.intersect_all(&ray([-300.0, 0.0, 0.0], [1.0, 0.0, 0.0]), 0, &mut hits);
    hits.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap());
    let ts: Vec<f64> = hits.iter().map(|h| h.t).collect();
    assert_eq!(ts.len(), 4, "outer, inner, inner, outer — got {ts:?}");
    for (got, want) in ts.iter().zip([170.0, 230.0, 370.0, 430.0]) {
        assert!((got - want).abs() < 1e-8, "{ts:?}");
    }
}

/// The point of the re-origining, stated as a test: a torus at a millimetre
/// scene's coordinates traces to the same `t` as the same torus at the
/// origin, once the ray is translated with it.
#[test]
fn a_torus_at_1e4_traces_like_the_same_torus_at_the_origin() {
    let offset = Vec3::new(1e4, 1e4, 1e4);
    let axis = dir([0.2, -0.3, 1.0]);
    let (major, minor) = (74.0, 2.0);
    let near = Prim::Torus {
        center: Point3::origin(),
        axis,
        major,
        minor,
    };
    let far = Prim::Torus {
        center: Point3::origin() + offset,
        axis,
        major,
        minor,
    };

    let eye = Point3::new(-900.0, 350.0, 420.0);
    let mut compared = 0;
    for r in fan(eye, Point3::origin(), 0.09, 48) {
        let d = *r.direction.as_ref();
        let shifted = Ray::new(r.origin + offset, d);
        let a = near.intersect(&r, 0, 0.0, f64::INFINITY);
        let b = far.intersect(&shifted, 0, 0.0, f64::INFINITY);
        match (a, b) {
            (Some(a), Some(b)) => {
                compared += 1;
                let rel = (a.t - b.t).abs() / a.t.abs().max(1.0);
                assert!(
                    rel < 1e-6,
                    "t drifted from {} to {} moving the torus to 1e4 (rel {rel:e})",
                    a.t,
                    b.t
                );
            }
            // A ray that grazes may legitimately land on opposite sides of
            // the silhouette at the two offsets; anything more than a
            // handful of those would show up as too few comparisons below.
            (None, None) => {}
            _ => {}
        }
    }
    assert!(compared > 40, "only {compared} rays hit both");
}

#[test]
fn a_box_uv_packs_the_face_index() {
    let prim = Prim::Box {
        center: Point3::origin(),
        half: Vec3::new(1.0, 1.0, 1.0),
        rot: Frame::identity(),
    };
    // Down the +Z face, dead centre: face 5, in-face (0.5, 0.5).
    let h = prim
        .intersect(
            &ray([0.0, 0.0, 5.0], [0.0, 0.0, -1.0]),
            0,
            0.0,
            f64::INFINITY,
        )
        .expect("hits the lid");
    let face = (h.uv.x * 6.0).floor();
    assert_eq!(face, 5.0, "u = {}", h.uv.x);
    assert!((h.uv.x * 6.0 - face - 0.5).abs() < 1e-12);
    assert!((h.uv.y - 0.5).abs() < 1e-12);
}

#[test]
fn a_sphere_uv_is_lat_long_and_dpdu_runs_around_it() {
    let prim = Prim::Sphere {
        center: Point3::origin(),
        radius: 2.0,
    };
    // Straight down onto the north pole: v = 0.
    let top = prim
        .intersect(
            &ray([0.0, 0.0, 9.0], [0.0, 0.0, -1.0]),
            0,
            0.0,
            f64::INFINITY,
        )
        .expect("pole");
    assert!(top.uv.y.abs() < 1e-9, "v = {}", top.uv.y);
    // On the +X equator: u = 0, v = 1/2, and dP/du points along +Y.
    let side = prim
        .intersect(
            &ray([9.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
            0,
            0.0,
            f64::INFINITY,
        )
        .expect("equator");
    assert!(side.uv.x.abs() < 1e-9, "u = {}", side.uv.x);
    assert!((side.uv.y - 0.5).abs() < 1e-9, "v = {}", side.uv.y);
    let t = side.dpdu.expect("a sphere has a grain");
    assert!(t.normalize().y > 0.999, "dP/du = {t:?}");
    assert!(t.dot(*side.normal.as_ref()).abs() < 1e-9, "tangent");
}

#[test]
fn a_cylinders_dpdu_is_circumferential_in_its_own_frame() {
    let axis = dir([0.0, 1.0, 0.0]);
    let prim = Prim::Cylinder {
        center: Point3::origin(),
        axis,
        radius: 3.0,
        half_height: 10.0,
    };
    let h = prim
        .intersect(
            &ray([0.0, 0.0, 9.0], [0.0, 0.0, -1.0]),
            0,
            0.0,
            f64::INFINITY,
        )
        .expect("hits the side");
    let t = h.dpdu.expect("a lathe leaves a grain");
    // Around the axis: perpendicular to both the normal and the axis.
    assert!(t.dot(*h.normal.as_ref()).abs() < 1e-9);
    assert!(t.dot(*axis.as_ref()).abs() < 1e-9);
}

#[test]
fn bounds_contain_every_hit() {
    let prims = mixed_prims(400);
    for prim in &prims {
        let b = prim.bounds();
        let c = b.center();
        for r in fan(c + Vec3::new(-3000.0, 1700.0, 2100.0), c, 0.05, 6) {
            let mut hits = Vec::new();
            prim.hits(&r, 0, 0.0, f64::INFINITY, &mut hits);
            for h in hits {
                let slack = 1e-6;
                assert!(
                    h.point.x >= b.min.x - slack
                        && h.point.x <= b.max.x + slack
                        && h.point.y >= b.min.y - slack
                        && h.point.y <= b.max.y + slack
                        && h.point.z >= b.min.z - slack
                        && h.point.z <= b.max.z + slack,
                    "{:?} escapes {b:?} on {prim:?}",
                    h.point
                );
            }
        }
    }
}

/// A deterministic spread of every bounded primitive, laid out on a grid at
/// millimetre coordinates.
fn mixed_prims(n: usize) -> Vec<Prim> {
    // A cheap LCG: the same scene every run, no dev-dependency.
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    (0..n)
        .map(|i| {
            let c = Point3::new(
                (i % 25) as f64 * 400.0,
                ((i / 25) % 25) as f64 * 400.0,
                (i / 625) as f64 * 400.0,
            );
            let axis = dir([next() - 0.5, next() - 0.5, next() + 0.2]);
            match i % 6 {
                0 => Prim::Sphere {
                    center: c,
                    radius: 40.0 + 60.0 * next(),
                },
                1 => Prim::Box {
                    center: c,
                    half: Vec3::new(30.0 + 40.0 * next(), 30.0 + 40.0 * next(), 20.0),
                    rot: Frame::from_z(axis),
                },
                2 => Prim::Cylinder {
                    center: c,
                    axis,
                    radius: 20.0 + 40.0 * next(),
                    half_height: 40.0 + 60.0 * next(),
                },
                3 => Prim::Cone {
                    center: c,
                    axis,
                    radius_base: 40.0 + 40.0 * next(),
                    radius_top: 10.0 * next(),
                    half_height: 50.0,
                },
                4 => Prim::Torus {
                    center: c,
                    axis,
                    major: 60.0 + 40.0 * next(),
                    minor: 5.0 + 15.0 * next(),
                },
                _ => Prim::Ellipsoid {
                    center: c,
                    rot: Frame::from_z(axis),
                    semi: Vec3::new(
                        30.0 + 50.0 * next(),
                        30.0 + 50.0 * next(),
                        30.0 + 50.0 * next(),
                    ),
                },
            }
        })
        .collect()
}

/// The whole point of the exercise: ten thousand mixed primitives in a tree,
/// traced, and every hit still on the surface it claims.
#[test]
fn a_bvh_over_ten_thousand_mixed_prims_traces() {
    let prims = mixed_prims(10_000);
    let bvh = Bvh::build(Analytic::from_prims(prims));
    assert_eq!(bvh.geometry().len(), 10_000);

    let mut hits = 0;
    let mut kinds = [0usize; 6];
    for i in 0..40 {
        for j in 0..40 {
            let origin = Point3::new(
                -4000.0,
                (i as f64 / 40.0) * 9600.0,
                (j as f64 / 40.0) * 6000.0,
            );
            let Some(h) = bvh.trace_closest(&Ray::new(origin, Vec3::new(1.0, 0.05, 0.03))) else {
                continue;
            };
            hits += 1;
            let prim = &bvh.geometry().prims()[h.prim as usize];
            kinds[(h.prim as usize) % 6] += 1;
            let tol = if matches!(prim, Prim::Torus { .. }) {
                1e-6
            } else {
                1e-9
            };
            let resid = implicit(prim, h.point);
            assert!(resid.abs() < tol, "{resid:e} off {prim:?}");
            assert_eq!(h.payload, h.prim as u64);
        }
    }
    assert!(hits > 500, "only {hits} of 1600 rays hit anything");
    // And the tree is not quietly returning one shape for everything.
    assert!(
        kinds.iter().filter(|k| **k > 0).count() >= 5,
        "kinds hit: {kinds:?}"
    );
}

#[test]
fn an_empty_set_is_empty() {
    let geom = Analytic::new();
    assert!(geom.is_empty());
    assert_eq!(geom.len(), 0);
    assert!(geom.prims().is_empty());
}

#[test]
fn from_z_is_orthonormal_everywhere_including_the_poles() {
    for v in [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [1e-16, -1e-16, -1.0],
        [0.3, -0.7, 0.2],
    ] {
        let f = Frame::from_z(dir(v));
        for (a, b) in [(f.x, f.y), (f.y, f.z), (f.z, f.x)] {
            assert!(a.dot(b).abs() < 1e-12, "not orthogonal for {v:?}");
        }
        for a in [f.x, f.y, f.z] {
            assert!((a.norm() - 1.0).abs() < 1e-12, "not unit for {v:?}");
        }
        // And it round-trips.
        let q = Vec3::new(1.0, -2.0, 3.0);
        assert!((f.to_world(f.to_local(q)) - q).norm() < 1e-12);
    }
}
