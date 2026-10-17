//! Phase 2c: muni's BVR1 rover, built from vcad parts and tessellated by
//! vcad's own kernel, in a scene on the GPU wavefront — instanced extrusions
//! and wheels, an anodised-gold GGX body, a mesh ground, sun and sky with
//! NEE — then the body's roughness and colour recovered from a target render
//! by path replay backprop.
//!
//! The parts are `muni/bvr/cad`'s (`BVR1Frame::generate`, `Extrusion2020`,
//! `UUMotor::kn6104`, `BaseTray::generate_simple`), ported line for line onto
//! vcad 0.10's `Part` API: that crate is a path dependency of a private
//! checkout and cannot be a dependency here, but the CAD is the same calls on
//! the same kernel. Every repeated part is built and tessellated once and
//! placed as an instance: one 500 mm rail (8 placements), one 20 mm corner
//! block (8), one 200 mm post (4), one wheel (4), one dropout plate (4).
//! Authoring is vcad's — millimetres, Z up — and the scene stays in
//! millimetres: piece 2's offsets are what make that safe.
//!
//! `cargo run -p irys-diff --features gpu --example rover -- out/rover`
//! writes `beauty.ppm` (768×512, 256 spp, depth 6, the fitted θ),
//! `target.ppm`, `iter_NNN.ppm` (previews at the iterate, 256 spp),
//! `filmstrip.csv` (their θ and image MSE) and `loss.csv` (every
//! iteration's 8-spp loss and θ).

use std::{fs, io::Write, path::Path, time::Instant};

use irys_core::{Point3, Transform, TriMesh, Vec3};
use irys_diff::ggx::PARAMS;
use irys_diff::gpu::{self, GpuPrb};
use irys_diff::prb::{Env, Opaque, Placement, Scene, Surface, grid_floor};
use irys_prims::analytic::Analytic;
use vcad::{Part, centered_cube, centered_cylinder};

/// The body's anodised gold: `(α, F0 at 450, 550, 650 nm)`.
const TRUTH: [f64; PARAMS] = [0.18, 0.42, 0.78, 0.98];

// ---- muni/bvr/cad, BVR1 defaults ----

const PROFILE: f64 = 20.0;
const FRAME_W: f64 = 540.0;
const FRAME_L: f64 = 540.0;
const FRAME_H: f64 = 240.0;
/// `BVR1AssemblyConfig::default().ground_clearance`.
const GC: f64 = 295.0;

/// `frame::Extrusion2020::generate`: a 20 × 20 bar along Z with four T-slots
/// and a centre bore, centred at the origin.
fn extrusion(len: f64) -> Part {
    let slot = || {
        let opening = centered_cube("slot_opening", 6.0, 6.0, len + 2.0);
        let inner = centered_cube("slot_inner", 10.0, 4.0, len + 2.0).translate(0.0, -2.0, 0.0);
        opening.union(&inner)
    };
    let bar = centered_cube("bar", PROFILE, PROFILE, len);
    let slots = slot()
        .translate(0.0, PROFILE / 2.0, 0.0)
        .union(
            &slot()
                .rotate(0.0, 0.0, 90.0)
                .translate(PROFILE / 2.0, 0.0, 0.0),
        )
        .union(
            &slot()
                .rotate(0.0, 0.0, 180.0)
                .translate(0.0, -PROFILE / 2.0, 0.0),
        )
        .union(
            &slot()
                .rotate(0.0, 0.0, 270.0)
                .translate(-PROFILE / 2.0, 0.0, 0.0),
        );
    let bore = centered_cylinder("bore", 2.5, len + 2.0, 32);
    bar.difference(&slots).difference(&bore)
}

/// `uumotor::UUMotor::kn6104().generate()` (`kn6104_single_shaft`): tyre,
/// hub, two flanges, the mount-side axle and its nut; axle along Y.
fn wheel() -> Part {
    let (wheel_d, tire_w, hub_d, hub_w) = (270.0, 100.0, 180.0, 85.0);
    let (axle_d, axle_outer, flange_d, flange_t) = (16.0, 45.0, 160.0, 8.0);
    let seg = 48;
    let tire = centered_cylinder("tire_outer", wheel_d / 2.0, tire_w, seg)
        .rotate(90.0, 0.0, 0.0)
        .difference(
            &centered_cylinder("tire_inner", hub_d / 2.0 - 10.0, tire_w + 2.0, seg)
                .rotate(90.0, 0.0, 0.0),
        );
    let hub = centered_cylinder("hub", hub_d / 2.0, hub_w, seg).rotate(90.0, 0.0, 0.0);
    let flange = |y: f64| {
        centered_cylinder("flange", flange_d / 2.0, flange_t, seg)
            .rotate(90.0, 0.0, 0.0)
            .translate(0.0, y, 0.0)
    };
    let axle = centered_cylinder("axle_outer", axle_d / 2.0, axle_outer, seg)
        .rotate(90.0, 0.0, 0.0)
        .translate(0.0, hub_w / 2.0 + axle_outer / 2.0, 0.0);
    let nut = centered_cylinder("nut", axle_d * 0.9, 10.0, 6)
        .rotate(90.0, 0.0, 0.0)
        .translate(0.0, hub_w / 2.0 + axle_outer - 5.0, 0.0);
    tire.union(&hub)
        .union(&flange(-hub_w / 2.0 + flange_t / 2.0))
        .union(&flange(hub_w / 2.0 - flange_t / 2.0))
        .union(&axle)
        .union(&nut)
}

/// vcad's tessellation of a part, in its own millimetres.
fn mesh(part: &Part) -> TriMesh {
    let m = part.to_mesh();
    let v = m.vertices();
    let positions = v
        .chunks_exact(3)
        .map(|p| Point3::new(p[0] as f64, p[1] as f64, p[2] as f64))
        .collect();
    TriMesh::new(positions, Vec::new(), &m.indices())
}

fn opaque(mesh: TriMesh, s: Surface) -> Opaque {
    Opaque::new(mesh, s, Analytic::from_prims(Vec::new()), &[])
}

fn deg(d: f64) -> f64 {
    d.to_radians()
}

fn at(x: f64, y: f64, z: f64) -> Transform {
    Transform::translation(x, y, z)
}

/// The rover on the ground at `size`, `spp` paths per pixel.
fn rover(size: (usize, usize), spp: usize, blas: &[TriMesh]) -> Scene {
    // Trees: 0 rail (500), 1 corner block (20), 2 post (200), 3 wheel,
    // 4 dropout plate, 5 tray, 6 lid, 7 ground.
    let surfaces = [
        Surface::Conductor,
        Surface::Conductor,
        Surface::Conductor,
        Surface::Diffuse,
        Surface::Diffuse,
        Surface::Diffuse,
        Surface::Conductor,
        Surface::Diffuse,
    ];
    let trees = blas
        .iter()
        .zip(surfaces)
        .map(|(m, s)| opaque(m.clone(), s))
        .collect();
    let mut p = Vec::new();
    // `BVR1Frame::create_rectangular_frame` at the bottom and top levels.
    let (xe, ye) = (FRAME_W / 2.0 - PROFILE / 2.0, FRAME_L / 2.0 - PROFILE / 2.0);
    for z in [0.0, FRAME_H - PROFILE] {
        let zc = GC + z + PROFILE / 2.0;
        for y in [ye, -ye] {
            p.push(Placement::new(
                0,
                at(0.0, y, zc).then(&Transform::rotation_y(deg(90.0))),
            ));
        }
        for x in [-xe, xe] {
            p.push(Placement::new(
                0,
                at(x, 0.0, zc).then(&Transform::rotation_x(deg(90.0))),
            ));
        }
        for (x, y) in [(-xe, ye), (xe, ye), (-xe, -ye), (xe, -ye)] {
            p.push(Placement::new(1, at(x, y, zc)));
        }
    }
    // `create_vertical_posts`.
    let post = FRAME_H - 2.0 * PROFILE;
    for (x, y) in [(-xe, ye), (xe, ye), (-xe, -ye), (xe, -ye)] {
        p.push(Placement::new(2, at(x, y, GC + PROFILE + post / 2.0)));
    }
    // `BVR1Assembly::add_wheels`: axles along X, wheels on the ground, each
    // hanging from a dropout plate outboard of it.
    let (wx, wy, wz) = (220.0, FRAME_L / 2.0 - 60.0, 135.0);
    for y in [wy, -wy] {
        for (x, turn) in [(-wx, 90.0), (wx, -90.0)] {
            let rubber = Placement::new(3, at(x, y, wz).then(&Transform::rotation_z(deg(turn))))
                .tinted(0.12);
            p.push(rubber);
            let plate_x = x.signum() * (wx + 42.5 + 20.0);
            p.push(Placement::new(4, at(plate_x, y, (GC + wz) / 2.0 + 10.0)).tinted(0.25));
        }
    }
    // `BaseTray::generate_simple` at `gc + 20`, and the lid on the top rails.
    p.push(Placement::new(5, at(0.0, 0.0, GC + 20.0)).tinted(0.08));
    p.push(Placement::new(6, at(0.0, 0.0, GC + FRAME_H + 3.0)));
    p.push(Placement::new(7, Transform::identity()));
    let mut s = Scene::instanced(
        trees,
        p,
        None,
        size,
        spp,
        (
            Point3::new(1250.0, -1550.0, 820.0),
            Point3::new(0.0, 0.0, 250.0),
        ),
        17.0,
    );
    s.nee = true;
    s.depth = 5;
    s.floor_albedo = 0.55;
    s.floor_checker = Some(500.0);
    s.env = Env {
        sun: Vec3::new(0.55, -0.25, 0.8).normalize(),
        kappa: 900.0,
        sun_radiance: 5.0 * 900.0 / (2.0 * std::f64::consts::PI),
        sky: 0.8,
    };
    s
}

fn ppm(path: &Path, (w, h): (usize, usize), img: &[f32], exposure: f32) {
    let mut f = fs::File::create(path).unwrap();
    write!(f, "P6\n{w} {h}\n255\n").unwrap();
    let srgb = |x: f32| {
        let x = (x * exposure).clamp(0.0, 1.0);
        let s = if x <= 0.0031308 {
            12.92 * x
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0 + 0.5) as u8
    };
    f.write_all(&img.iter().map(|&x| srgb(x)).collect::<Vec<_>>())
        .unwrap();
}

/// `frames` calls averaged: a frame of `frames × scene.spp` paths per pixel.
fn accumulate(
    g: &mut GpuPrb,
    scene: &Scene,
    theta: &[f64; PARAMS],
    seed: u64,
    frames: u64,
) -> Vec<f32> {
    let mut acc = vec![0.0f64; scene.size.0 * scene.size.1 * 3];
    for f in 0..frames {
        for (a, x) in acc.iter_mut().zip(g.image(scene, theta, seed + f)) {
            *a += x as f64 / frames as f64;
        }
    }
    acc.iter().map(|&x| x as f32).collect()
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "out/rover".into());
    let out = Path::new(&out);
    fs::create_dir_all(out).unwrap();
    let ctx = gpu::context().expect("rover needs a GPU adapter");
    const EXPOSURE: f32 = 0.6;

    let t = Instant::now();
    let parts = [
        extrusion(FRAME_L - 2.0 * PROFILE),
        extrusion(PROFILE),
        extrusion(FRAME_H - 2.0 * PROFILE),
        wheel(),
        centered_cube("dropout", 8.0, 60.0, GC - 135.0 + 20.0),
        centered_cube("tray", 500.0, 500.0, 4.0),
        centered_cube("lid", FRAME_W, FRAME_L, 6.0),
    ];
    let mut blas: Vec<TriMesh> = parts.iter().map(mesh).collect();
    blas.push(grid_floor(16, 24000.0));
    let tris: Vec<usize> = blas.iter().map(|m| m.triangles().len()).collect();
    let scene = rover((192, 128), 8, &blas);
    let placed: usize = scene.placements().iter().map(|p| tris[p.blas]).sum();
    eprintln!(
        "vcad: {} parts tessellated in {:.0?}, {} triangles in {} trees; {} instances place {placed}",
        parts.len(),
        t.elapsed(),
        tris.iter().sum::<usize>(),
        blas.len(),
        scene.placements().len(),
    );
    for (name, (part, n)) in ["rail", "corner", "post", "wheel", "dropout", "tray", "lid"]
        .iter()
        .zip(parts.iter().zip(&tris))
    {
        eprintln!(
            "  {name:8} {n:6} triangles, volume {:.0} mm³",
            part.volume()
        );
    }

    // ---- the fit: recover the body's α and F0 from a target render ----
    let mut theta = [0.6, 0.6, 0.6, 0.6];
    let reference = rover((192, 128), 32, &blas);
    let mut gr = GpuPrb::new(ctx, &reference);
    let target = accumulate(&mut gr, &reference, &TRUTH, 1000, 8);
    ppm(&out.join("target.ppm"), reference.size, &target, EXPOSURE);
    let mut g = GpuPrb::new(ctx, &scene);
    g.set_target(&target.iter().map(|&x| x as f64).collect::<Vec<_>>());
    let (lr, b1, b2) = (0.04, 0.9, 0.99);
    let (mut m, mut v) = ([0.0; 4], [0.0; 4]);
    let mut csv = fs::File::create(out.join("loss.csv")).unwrap();
    writeln!(csv, "iter,loss,alpha,f0_450,f0_550,f0_650").unwrap();
    let iters = 80;
    // The estimate is the mean of the last `tail` iterates (Polyak): each
    // gradient is an 8-spp estimate, so the iterates jitter round the answer.
    let tail = 20;
    let mut avg = [0.0; PARAMS];
    let mut strip = fs::File::create(out.join("filmstrip.csv")).unwrap();
    writeln!(strip, "iter,mse,alpha,f0_450,f0_550,f0_650").unwrap();
    let mse = |img: &[f32]| {
        img.iter()
            .zip(&target)
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            / img.len() as f64
    };
    let t = Instant::now();
    let mut fit_time = std::time::Duration::ZERO;
    for it in 1..=iters {
        let ti = Instant::now();
        let seeds = (2 * it as u64, 2 * it as u64 + 1);
        let (loss, grad) = g.gradient_l2(&scene, &theta, seeds);
        fit_time += ti.elapsed();
        writeln!(
            csv,
            "{it},{loss:e},{},{},{},{}",
            theta[0], theta[1], theta[2], theta[3]
        )
        .unwrap();
        if [1, 5, 10, 20, 40].contains(&it) {
            let img = accumulate(&mut gr, &reference, &theta, 2000, 8);
            ppm(
                &out.join(format!("iter_{it:03}.ppm")),
                reference.size,
                &img,
                EXPOSURE,
            );
            writeln!(
                strip,
                "{it},{:e},{},{},{},{}",
                mse(&img),
                theta[0],
                theta[1],
                theta[2],
                theta[3]
            )
            .unwrap();
        }
        for p in 0..PARAMS {
            m[p] = b1 * m[p] + (1.0 - b1) * grad[p];
            v[p] = b2 * v[p] + (1.0 - b2) * grad[p] * grad[p];
            let step = lr / (1.0 + it as f64 / 40.0) * (m[p] / (1.0 - b1.powi(it)))
                / ((v[p] / (1.0 - b2.powi(it))).sqrt() + 1e-12);
            theta[p] = (theta[p] - step).clamp(0.02, 1.0);
            if it > iters - tail {
                avg[p] += theta[p] / tail as f64;
            }
        }
        if it % 10 == 0 || it == 1 {
            eprintln!("it {it:3}  loss {loss:.3e}  θ {theta:.3?}");
        }
    }
    let theta = avg;
    let img = accumulate(&mut gr, &reference, &theta, 2000, 8);
    ppm(
        &out.join(format!("iter_{iters:03}.ppm")),
        reference.size,
        &img,
        EXPOSURE,
    );
    writeln!(
        strip,
        "{iters},{:e},{},{},{},{}",
        mse(&img),
        theta[0],
        theta[1],
        theta[2],
        theta[3]
    )
    .unwrap();
    let truth_img = accumulate(&mut gr, &reference, &TRUTH, 2000, 8);
    eprintln!(
        "{iters} iterations in {:.1?} ({:.1?} of it gradients, {:.1} ms each); θ (mean of the last {tail}) {theta:.3?} vs truth {TRUTH:?}; image MSE {:.2e} (truth re-rendered: {:.2e})",
        t.elapsed(),
        fit_time,
        fit_time.as_secs_f64() * 1e3 / iters as f64,
        mse(&img),
        mse(&truth_img),
    );

    // ---- the beauty frame, at the fitted θ ----
    let t = Instant::now();
    let mut beauty = rover((768, 512), 16, &blas);
    beauty.depth = 6;
    let mut gb = GpuPrb::new(ctx, &beauty);
    let built = t.elapsed();
    let t = Instant::now();
    let frames = 16;
    let img = accumulate(&mut gb, &beauty, &theta, 5000, frames);
    ppm(&out.join("beauty.ppm"), beauty.size, &img, EXPOSURE);
    eprintln!(
        "beauty 768×512×{} (depth 6, NEE, {} chunks per call): build {built:.0?}, render {:.1?}",
        16 * frames,
        gb.chunks(),
        t.elapsed()
    );
}
