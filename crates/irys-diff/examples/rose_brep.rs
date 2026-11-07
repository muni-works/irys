//! The shared rose-arm benchmark with vcad's differentiable BRep seam
//! ("Exact Gradients Through a BRep Kernel", vcad-kernel-diff at d10f279).
//!
//! The part is rose full's left shoulder-pitch output plate,
//! `L_shoA_shop_out` in `full/cad/links/shoA.loon`: a 3 mm rounded plate with
//! a 49 mm bearing bore and twelve 4.4 mm bolt holes. `θ` is the bearing
//! bore's diameter. The chain is
//!
//! ```text
//! θ ──vcad──▶ frozen-plan mesh nodes x_i ──irys──▶ image I ──▶ ℒ = Σ g · I
//! ```
//!
//! irys renders the plan's mesh exactly as vcad emits it (no re-tessellation)
//! against the sky at depth 1, where the edge term is the whole derivative,
//! and gives `∂ℒ/∂x_i` per node by edge sampling
//! ([`EdgeSamples::for_vertices`]). vcad pulls that back to `dℒ/dθ` in one
//! pass (`evaluate_with_pullback`). It is checked three ways:
//!
//! 1. vcad forward mode: `Σ ∂ℒ/∂x_i · ẋ_i` with `evaluate_with_sensitivity`
//!    (the seam's own reverse/forward agreement);
//! 2. vcad's FD oracle: `Σ ∂ℒ/∂x_i · ẋ_i^FD` with `fd_velocities`;
//! 3. end to end: central differences of the rendered loss with the plan
//!    re-evaluated at `θ ± h` (common random numbers, quadrature pixels).
//!
//! `cargo run --release -p irys-diff --example rose_brep` (needs ../rose or $ROSE)

#[path = "rose/arm.rs"]
#[allow(dead_code)]
mod arm;

use std::time::Instant;

use arm::{Expr, axis_of, head, nums, place};
use irys_core::{Point3, Transform, TriMesh};
use irys_diff::ggx::PARAMS;
use irys_diff::prb::{self, Kernels, Opaque, Placement, Scene, Surface};
use irys_diff::vis::{EdgeSamples, Edges};
use irys_prims::analytic::Analytic;
use vcad_kernel_diff::{
    evaluate_with_pullback, evaluate_with_sensitivity, fd_velocities, synthesize_seeding,
};
use vcad_kernel_next::Solid;
use vcad_kernel_next::vcad_kernel_math::Vec3 as KVec3;
use vcad_kernel_next::vcad_kernel_primitives::BRepSolid;
use vcad_kernel_next::vcad_kernel_tessellate::TessellationParams;
use vcad_kernel_next::vcad_kernel_tessellate::frozen::{FrozenMesh, FrozenPlan, capture_plan, evaluate_plan};

const ROOT: &str = "L_shoA_shop_out";
/// The bearing bore, as shoA.loon has it.
const BORE: f64 = 49.0;
const CORNER_SEG: u32 = 24;
const BORE_SEG: u32 = 32;
const THETA: [f64; PARAMS] = [0.3, 0.9, 0.8, 0.5];
const W: usize = 160;
const H: usize = 128;

// ---- the plate on vcad's current kernel, with the bore as θ ----

fn cube(axis: usize, at: [f64; 3], size: [f64; 3]) -> Solid {
    let s = place(axis, size);
    let a = place(axis, at);
    Solid::cube(s[0], s[1], s[2]).translate(a[0], a[1], a[2])
}

fn cyl(axis: usize, r: f64, h: f64, seg: u32) -> Solid {
    let c = Solid::cylinder(r, h, seg);
    match axis {
        0 => c.rotate(0.0, 90.0, 0.0),
        1 => c.rotate(-90.0, 0.0, 0.0),
        _ => c,
    }
}

fn plate(axis: usize, [n0, n1, u0, u1, v0, v1, r]: [f64; 7]) -> Solid {
    let t = n1 - n0;
    let core = cube(axis, [n0, u0 + r, v0], [t, u1 - u0 - 2.0 * r, v1 - v0]);
    let cross = cube(axis, [n0, u0, v0 + r], [t, u1 - u0, v1 - v0 - 2.0 * r]);
    let cn = |cu: f64, cv: f64| {
        let a = place(axis, [n0, cu, cv]);
        cyl(axis, r, t, CORNER_SEG).translate(a[0], a[1], a[2])
    };
    core.union(&cross)
        .union(&cn(u0 + r, v0 + r))
        .union(&cn(u0 + r, v1 - r))
        .union(&cn(u1 - r, v0 + r))
        .union(&cn(u1 - r, v1 - r))
}

fn bore(axis: usize, [d, cu, cv, n0, n1]: [f64; 5]) -> Solid {
    let a = place(axis, [n0 - 2.0, cu, cv]);
    cyl(axis, d / 2.0, n1 - n0 + 4.0, BORE_SEG).translate(a[0], a[1], a[2])
}

/// The loon subset the plate uses, with the bearing bore's diameter
/// replaced by `d`.
fn eval(e: &Expr, d: f64) -> Solid {
    let (h, a) = head(e).expect("a form");
    match h {
        "union" => eval(&a[0], d).union(&eval(&a[1], d)),
        "difference" => eval(&a[1], d).difference(&eval(&a[0], d)),
        h if h.starts_with("plate-") => plate(axis_of(h), nums(a)),
        h if h.starts_with("bore-") => {
            let mut p: [f64; 5] = nums(a);
            if p[0] == BORE {
                p[0] = d;
            }
            bore(axis_of(h), p)
        }
        h => panic!("{h} is not in the plate's subset"),
    }
}

fn mesh(f: &FrozenMesh) -> TriMesh {
    let pos = f
        .positions
        .iter()
        .map(|p| Point3::new(p.x, p.y, p.z))
        .collect();
    let idx: Vec<u32> = f.triangles.iter().flatten().copied().collect();
    TriMesh::new(pos, Vec::new(), &idx)
}

// ---- the render and the loss (phase 2d's) ----

fn scene(m: TriMesh, spp: usize) -> Scene {
    // Face on to the plate (it spans x −58..36, z 426..498 at y 258..261),
    // a little off axis so the bores' inner walls show.
    let target = Point3::new(-11.0, 259.5, 462.0);
    let eye = Point3::new(30.0, -160.0, 520.0);
    let mut s = Scene::instanced(
        vec![Opaque::new(m, Surface::Diffuse, Analytic::from_prims(Vec::new()), &[])],
        vec![Placement::new(0, Transform::identity())],
        None,
        (W, H),
        spp,
        (eye, target),
        8.5,
    );
    s.depth = 1;
    s
}

/// `Σ g · I`, `g` smooth over the frame and different per channel.
fn weights() -> Vec<f64> {
    let tau = std::f64::consts::TAU;
    (0..W * H * 3)
        .map(|i| {
            let (px, c) = (i / 3, i % 3);
            let (x, y) = ((px % W) as f64 / W as f64, (px / W) as f64 / H as f64);
            1.0 + 0.6 * (tau * x + c as f64).sin() * (3.0 * y + 0.5 * c as f64).cos()
        })
        .collect()
}

/// [`weights`]' loss with every pixel integrated over `k²` fixed R2 points.
fn grid_loss(s: &mut Scene, k: usize, g: &[f64], seed: u64, kern: &mut Kernels) -> f64 {
    let mut pts = Vec::with_capacity(W * H * k * k);
    for py in 0..H {
        for px in 0..W {
            for j in 0..k * k {
                let (a1, a2) = (0.754_877_666_246_692_8, 0.569_840_290_998_053_2);
                let u = (0.5 + a1 * j as f64).fract();
                let v = (0.5 + a2 * j as f64).fract();
                pts.push(s.screen(px as f64 + u, py as f64 + v));
            }
        }
    }
    s.set_points(Some(pts));
    let (l, _) = prb::forward(s, &THETA, seed, kern);
    s.set_points(None);
    let nw = s.wavelengths;
    let mut loss = 0.0;
    for (lane, lj) in l.iter().enumerate() {
        let i = lane / nw;
        let lam = prb::wavelengths(prb::Rng::new(seed, (i / 2) as u64).uniform(), nw)[lane % nw];
        let hw = irys_light::spectrum::hero_weight(lam);
        let px = i / (k * k);
        for c in 0..3 {
            loss += lj * hw[c] as f64 / (nw * k * k) as f64 * g[px * 3 + c];
        }
    }
    loss
}

fn dot(g: &[KVec3], v: &[KVec3]) -> f64 {
    g.iter().zip(v).map(|(a, b)| a.dot(b)).sum()
}

fn main() {
    let t0 = Instant::now();
    let rose = arm::rose_dir();
    let root = arm::assembly(&rose)
        .into_iter()
        .find(|r| r.name == ROOT)
        .expect("L_shoA_shop_out in rose-full.loon");
    let expr = root.expr.clone();
    let build_one = |d: f64| -> BRepSolid {
        eval(&expr, d).as_brep().expect("the plate stays a BRep").clone()
    };
    let base = build_one(BORE);
    let build = |t: &[f64]| -> BRepSolid {
        if t[0] == BORE { base.clone() } else { build_one(t[0]) }
    };
    let params = TessellationParams {
        circle_segments: BORE_SEG,
        ..Default::default()
    };
    let plan: FrozenPlan = capture_plan(&base, &params).expect("capture the plan");
    let frozen = evaluate_plan(&base, &plan).expect("evaluate the plan");
    eprintln!(
        "{ROOT}: {} nodes, {} triangles ({:.1?})",
        plan.nodes.len(),
        plan.triangles.len(),
        t0.elapsed()
    );

    // irys: ∂ℒ/∂x per plan node, averaged over independent seeds.
    let g = weights();
    let mut s = scene(mesh(&frozen), 1);
    let edges = Edges::new(&s);
    let (n, seeds) = (
        std::env::var("N").ok().and_then(|v| v.parse().ok()).unwrap_or(200_000usize),
        std::env::var("SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(16u64),
    );
    let nn = plan.nodes.len();
    let mut kern = Kernels::new(2 * n * s.wavelengths, true);
    let seeding = synthesize_seeding(&build, &[BORE], 0, 1e-6).expect("synthesize the seeding");
    let fwd = evaluate_with_sensitivity(&base, &plan, &seeding).expect("forward seam");
    let fdv = fd_velocities(build_one, BORE, 1e-5, &plan).expect("fd velocities");
    let (mut pull, mut fwd_d, mut fd_d) = (Vec::new(), Vec::new(), Vec::new());
    let mut mean_grad = vec![KVec3::new(0.0, 0.0, 0.0); nn];
    for sd in 0..seeds {
        let es = EdgeSamples::for_vertices(&s, &edges, &[0], n, 100 + sd, None);
        s.set_points(Some(es.points().to_vec()));
        let (l, _) = prb::forward(&s, &THETA, 100 + sd, &mut kern);
        s.set_points(None);
        let gv = es.vertex_gradient(&s, 100 + sd, &l, &g, 0, nn);
        let gk: Vec<KVec3> = gv.iter().map(|v| KVec3::new(v.x, v.y, v.z)).collect();
        if sd == 0 {
            eprintln!(
                "edges: {} segments, Σℓ {:.0} px; {} of {} samples visible",
                es.segments,
                es.total_len,
                es.visible,
                es.len()
            );
        }
        let cot = evaluate_with_pullback(&base, &plan, &gk).expect("pullback");
        pull.push(cot.contract(&seeding));
        fwd_d.push(dot(&gk, &fwd.velocities));
        fd_d.push(dot(&gk, &fdv));
        for (m, v) in mean_grad.iter_mut().zip(&gk) {
            *m += *v / seeds as f64;
        }
    }
    let stat = |v: &[f64]| {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() as f64 - 1.0);
        (m, (var / v.len() as f64).sqrt())
    };
    let (p, pe) = stat(&pull);
    let (f, _) = stat(&fwd_d);
    let (fd, _) = stat(&fd_d);
    // Seam agreement per seed (same ∂ℒ/∂x on both sides: no render noise).
    let seam_fwd = pull
        .iter()
        .zip(&fwd_d)
        .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(1e-300))
        .fold(0.0, f64::max);
    let seam_fd = pull
        .iter()
        .zip(&fd_d)
        .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(1e-300))
        .fold(0.0, f64::max);

    // End to end: rendered loss at θ ± h through the same plan.
    // h must stay inside the plan's topology class: the bore's rim count is
    // sag-adaptive in its radius, and 0.05 mm already changes it.
    let k = std::env::var("K").ok().and_then(|v| v.parse().ok()).unwrap_or(12usize);
    let h = std::env::var("H").ok().and_then(|v| v.parse().ok()).unwrap_or(2e-3f64);
    let mut gk = Kernels::new(W * H * k * k * s.wavelengths, true);
    let mut side = |d: f64| {
        // LIN=1: the nodes moved along vcad's FD velocities instead of
        // re-evaluated (tests irys alone, at steps past the plan's anchor
        // tolerance).
        let fm = if std::env::var("LIN").is_ok() {
            FrozenMesh {
                positions: frozen.positions.iter().zip(&fdv).map(|(p, v)| *p + *v * d).collect(),
                triangles: frozen.triangles.clone(),
            }
        } else {
            evaluate_plan(&build_one(BORE + d), &plan).expect("plan at θ ± h")
        };
        let mut sc = scene(mesh(&fm), 1);
        grid_loss(&mut sc, k, &g, 1, &mut gk)
    };
    let e2e = (side(h) - side(-h)) / (2.0 * h);

    println!("rose-arm BRep benchmark: {ROOT}, θ = bearing bore diameter {BORE} mm");
    println!("  nodes {nn}, edge samples {n} × {seeds} seeds, image {W}×{H}");
    println!("  dℒ/dθ, vcad pullback (irys ∂ℒ/∂x):  {p:+.6e} ± {pe:.2e}");
    println!("  dℒ/dθ, vcad forward seam:            {f:+.6e}   max rel vs pullback {seam_fwd:.2e}");
    println!("  dℒ/dθ, vcad fd_velocities oracle:     {fd:+.6e}   max rel vs pullback {seam_fd:.2e}");
    println!(
        "  dℒ/dθ, end-to-end render FD (h {h} mm, {k}² pts/px): {e2e:+.6e}   rel {:.2e} ({:.1} σ)",
        (p - e2e).abs() / e2e.abs(),
        (p - e2e).abs() / pe
    );
    println!("  total {:.1?}", t0.elapsed());
    let _ = mean_grad;
}
