//! irys's half of the Mitsuba 3 baseline (`bench/mitsuba/`): export the
//! bench scenes for `bench/mitsuba/run_mitsuba.py`, and time and sample
//! irys's own gradients on them.
//!
//! `cargo run --release -p irys-diff --features gpu --example mitsuba_bench -- bench/mitsuba/out [material] [pose]`
//!
//! - `material`: [`Scene::pair`] (two GGX conductor balls on a grey floor,
//!   the sky and a sun lobe, NEE + MIS), the checker off, `complete` on
//!   (so `depth` vertices are Mitsuba's `max_depth = depth + 1`). The
//!   gradient of the L2 loss against a target at [`GOLD`], at [`THETA0`],
//!   by the GPU wavefront's path replay ([`GpuPrb::gradient_l2`]), per seed
//!   and per spp; and central differences for reference, of the loss of the
//!   expected image (`mean((I_a − T)(I_b − T))`, two independent renders).
//! - `pose`: phase 2e's two-box arm on a floor (depth 2 + NEE, direct and
//!   one bounce): the pose gradient at [`PHI0`] against a target at
//!   [`PHI_T`], by the device edge sampler plus the CPU interior term, and
//!   central differences as above.
//! - `cd-material`, `cd-pose`: the central differences alone, from the
//!   targets already written.
//!
//! Every scene is also written for Mitsuba: one ASCII PLY per placement in
//! world space, and a JSON file with the camera, the environment, the
//! materials and (pose) the rig. CSVs: `irys_<bench>.csv`, one row per
//! (spp or sample count, seed): the gradient and the milliseconds it took.

use std::{fmt::Write as _, fs, path::Path, time::Instant};

use irys_core::{Point3, Transform, Vec3};
use irys_diff::ggx::PARAMS;
use irys_diff::gpu::{self, GpuEdges, GpuPrb};
use irys_diff::interior::{self, Tangents};
use irys_diff::prb::{self, Opaque, Placement, Scene, Surface};
use irys_diff::vis::{self, Edges, Rig};
use irys_prims::analytic::Analytic;

/// Where the material gradient is taken, and the target's finish.
const THETA0: [f64; PARAMS] = [0.35, 0.95, 0.75, 0.45];
const GOLD: [f64; PARAMS] = [0.2, 0.34, 0.78, 1.0];
/// The pose bench's start and target poses (radians).
const PHI0: [f64; 3] = [0.30, 0.40, -0.60];
const PHI_T: [f64; 3] = [0.35, 0.45, -0.70];
const THETA_ARM: [f64; PARAMS] = [0.3, 0.9, 0.8, 0.5];

fn ply(path: &Path, scene: &Scene, k: usize) {
    let pl = &scene.placements()[k];
    let m = scene.blas()[pl.blas].geometry().mesh();
    let mut s = String::new();
    let _ = writeln!(s, "ply\nformat ascii 1.0\nelement vertex {}", m.positions().len());
    s += "property float x\nproperty float y\nproperty float z\n";
    let _ = writeln!(s, "element face {}", m.triangles().len());
    s += "property list uchar int vertex_indices\nend_header\n";
    for p in m.positions() {
        let w = pl.to_world.apply_point(p);
        let _ = writeln!(s, "{} {} {}", w.x, w.y, w.z);
    }
    for t in m.triangles() {
        let _ = writeln!(s, "3 {} {} {}", t[0], t[1], t[2]);
    }
    fs::write(path, s).unwrap();
}

/// Local-space PLY (the rig moves it in Mitsuba).
fn ply_local(path: &Path, m: &irys_core::TriMesh) {
    let mut s = String::new();
    let _ = writeln!(s, "ply\nformat ascii 1.0\nelement vertex {}", m.positions().len());
    s += "property float x\nproperty float y\nproperty float z\n";
    let _ = writeln!(s, "element face {}", m.triangles().len());
    s += "property list uchar int vertex_indices\nend_header\n";
    for p in m.positions() {
        let _ = writeln!(s, "{} {} {}", p.x, p.y, p.z);
    }
    for t in m.triangles() {
        let _ = writeln!(s, "3 {} {} {}", t[0], t[1], t[2]);
    }
    fs::write(path, s).unwrap();
}

fn v3(v: Vec3) -> String {
    format!("[{}, {}, {}]", v.x, v.y, v.z)
}

fn p3(v: Point3) -> String {
    format!("[{}, {}, {}]", v.x, v.y, v.z)
}

/// The scene description both runners share (camera, light, materials).
fn scene_json(scene: &Scene, shapes: &[String], extra: &str) -> String {
    let (eye, fwd, _right, up, tan_half) = scene.camera();
    let env = scene.env;
    format!(
        "{{\n  \"size\": [{}, {}],\n  \"eye\": {},\n  \"target\": {},\n  \"up\": {},\n  \"fov_y_deg\": {},\n  \"depth\": {},\n  \"complete\": {},\n  \"nee\": {},\n  \"wavelengths\": {},\n  \"floor_albedo\": {},\n  \"env\": {{\"sun\": {}, \"kappa\": {}, \"sun_radiance\": {}, \"sky\": {}}},\n  \"theta0\": {:?},\n  \"theta_target\": {:?},\n  \"shapes\": [\n    {}\n  ]{}\n}}\n",
        scene.size.0,
        scene.size.1,
        p3(eye),
        p3(eye + fwd),
        v3(up),
        2.0 * tan_half.atan().to_degrees(),
        scene.depth,
        scene.complete,
        scene.nee,
        scene.wavelengths,
        scene.floor_albedo,
        v3(env.sun),
        env.kappa,
        env.sun_radiance,
        env.sky,
        THETA0,
        GOLD,
        shapes.join(",\n    "),
        extra,
    )
}

fn surface_name(scene: &Scene, k: usize) -> &'static str {
    let b = &scene.blas()[scene.placements()[k].blas];
    match b.geometry().surface(0) {
        Surface::Conductor => "conductor",
        _ => "diffuse",
    }
}

/// A PFM (little-endian, bottom row first) of an RGB image, top row first.
fn pfm(path: &Path, img: &[f64], w: usize, h: usize) {
    let mut b = format!("PF\n{w} {h}\n-1.0\n").into_bytes();
    for y in (0..h).rev() {
        for x in 0..w {
            for c in 0..3 {
                b.extend_from_slice(&(img[(y * w + x) * 3 + c] as f32).to_le_bytes());
            }
        }
    }
    fs::write(path, b).unwrap();
}

fn mean_image(g: &mut GpuPrb, scene: &Scene, theta: &[f64; PARAMS], frames: u64, seed0: u64) -> Vec<f64> {
    let mut acc: Vec<f64> = Vec::new();
    for f in 0..frames {
        let im = gpu::widen(&g.image(scene, theta, seed0 + f));
        if acc.is_empty() {
            acc = im;
        } else {
            acc.iter_mut().zip(&im).for_each(|(a, b)| *a += b);
        }
    }
    acc.iter_mut().for_each(|a| *a /= frames as f64);
    acc
}

fn material(out: &Path) {
    let ctx = gpu::context().expect("a GPU adapter");
    let (w, h) = (128, 128);
    let mk = |spp: usize| {
        let mut s = Scene::pair(w, h, spp, 1.0);
        s.floor_checker = None;
        s.complete = true;
        s
    };
    let scene = mk(1);
    let shapes: Vec<String> = (0..scene.placements().len())
        .map(|k| {
            let f = format!("material_{k}.ply");
            ply(&out.join(&f), &scene, k);
            format!("{{\"ply\": \"{f}\", \"bsdf\": \"{}\", \"albedo\": {}}}", surface_name(&scene, k), scene.placements()[k].albedo)
        })
        .collect();
    fs::write(out.join("material.json"), scene_json(&scene, &shapes, "")).unwrap();
    if std::env::var("EXPORT_ONLY").is_ok() {
        return;
    }

    // The target: GOLD at 4096 spp (256 frames of 16).
    let ts = mk(16);
    let mut g = GpuPrb::new(ctx, &ts);
    let target = mean_image(&mut g, &ts, &GOLD, 256, 1 << 20);
    pfm(&out.join("irys_material_target.pfm"), &target, w, h);
    let start = mean_image(&mut g, &ts, &THETA0, 64, 1 << 21);
    pfm(&out.join("irys_material_start.pfm"), &start, w, h);

    let mut csv = String::from("system,bench,spp,seed,g0,g1,g2,g3,loss,ms\n");
    let seeds = 64u64;
    for spp in [1usize, 4, 16] {
        let s = mk(spp);
        let mut g = GpuPrb::new(ctx, &s);
        g.set_target(&target);
        for i in 0..3 {
            g.gradient_l2(&s, &THETA0, (900 + 2 * i, 901 + 2 * i));
        }
        for sd in 0..seeds {
            let t = Instant::now();
            let (loss, gr) = g.gradient_l2(&s, &THETA0, (2 * sd, 2 * sd + 1));
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let _ = writeln!(csv, "irys,material,{spp},{sd},{},{},{},{},{loss},{ms}", gr[0], gr[1], gr[2], gr[3]);
        }
        eprintln!("material spp {spp}: done");
    }
    fs::write(out.join("irys_material.csv"), csv).unwrap();
    material_cd(out, &target);
}

/// `mean((I_a − T)(I_b − T))` over two independent renders: unbiased for
/// the loss of the expected image `(E I − T)²`, which is what a gradient
/// with decorrelated primal and adjoint seeds estimates. (The MSE of one
/// noisy image adds `Var I`, which depends on the parameters too — its
/// central differences are not the gradient.)
fn product_loss(a: &[f64], b: &[f64], t: &[f64]) -> f64 {
    a.iter().zip(b).zip(t).map(|((a, b), t)| (a - t) * (b - t)).sum::<f64>() / t.len() as f64
}

fn read_pfm(path: &Path) -> (Vec<f64>, usize, usize) {
    let b = fs::read(path).unwrap();
    let mut it = b.split(|&c| c == b'\n');
    let _ = it.next();
    let dims = String::from_utf8(it.next().unwrap().to_vec()).unwrap();
    let mut d = dims.split_whitespace().map(|x| x.parse::<usize>().unwrap());
    let (w, h) = (d.next().unwrap(), d.next().unwrap());
    let _ = it.next();
    let data = &b[b.len() - w * h * 12..];
    let mut img = vec![0.0; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let o = ((h - 1 - y) * w + x) * 12 + c * 4;
                img[(y * w + x) * 3 + c] = f32::from_le_bytes(data[o..o + 4].try_into().unwrap()) as f64;
            }
        }
    }
    (img, w, h)
}

fn material_cd(out: &Path, target: &[f64]) {
    let ctx = gpu::context().expect("a GPU adapter");
    let mut s = Scene::pair(128, 128, 16, 1.0);
    s.floor_checker = None;
    s.complete = true;
    let mut g = GpuPrb::new(ctx, &s);
    let hh = 0.01;
    let mut cd = String::from("system,bench,param,h,seed,dloss\n");
    for p in 0..PARAMS {
        for sd in 0..64u64 {
            let (mut a, mut b) = (THETA0, THETA0);
            a[p] += hh;
            b[p] -= hh;
            let mut l = |th: &[f64; PARAMS]| {
                let ia = gpu::widen(&g.image(&s, th, 5000 + sd));
                let ib = gpu::widen(&g.image(&s, th, 50000 + sd));
                product_loss(&ia, &ib, target)
            };
            let d = (l(&a) - l(&b)) / (2.0 * hh);
            let _ = writeln!(cd, "irys,material,{p},{hh},{sd},{d}");
        }
    }
    fs::write(out.join("irys_material_cd.csv"), cd).unwrap();
}

fn opaque(m: irys_core::TriMesh, s: Surface) -> Opaque {
    Opaque::new(m, s, Analytic::from_prims(Vec::new()), &[])
}

const SHOULDER: [f64; 3] = [-0.9, 0.3, 1.25];
const UPPER: f64 = 0.55;
const FORE: f64 = 0.45;
const GAP: f64 = 0.15;

/// Phase 2e's arm on a floor: a static post, a conductor upper link, a
/// diffuse forearm (boxes, each pre-rolled about its own axis).
fn arm_scene(w: usize, h: usize, spp: usize) -> Scene {
    let blas = vec![
        opaque(prb::grid_floor(8, 6.0), Surface::Diffuse),
        opaque(prb::cuboid(0.12, 0.12, 0.62), Surface::Diffuse),
        opaque(prb::cuboid(UPPER, 0.1, 0.1), Surface::Conductor),
        opaque(prb::cuboid(FORE, 0.085, 0.085), Surface::Diffuse),
    ];
    let placements = vec![
        Placement::new(0, Transform::identity()).tinted(1.0),
        Placement::new(1, Transform::translation(SHOULDER[0], SHOULDER[1] + 0.3, 0.62)).tinted(0.6),
        Placement::new(2, Transform::identity()),
        Placement::new(3, Transform::identity()).tinted(0.8),
    ];
    let mut s = Scene::instanced(
        blas,
        placements,
        None,
        (w, h),
        spp,
        (Point3::new(0.0, -4.5, 2.2), Point3::new(0.0, 0.0, 0.8)),
        20.0,
    );
    s.nee = true;
    s.depth = 2;
    s.complete = true;
    s
}

/// Shoulder pitch (about `y`), shoulder yaw (about `z`), elbow (about the
/// upper link's `y`); `bench/mitsuba/run_mitsuba.py` has the same rig.
fn arm_rig() -> Rig<'static> {
    Rig {
        moving: vec![2, 3],
        place: Box::new(|p: &[f64]| {
            let s = Point3::new(SHOULDER[0], SHOULDER[1], SHOULDER[2]);
            let sh = vis::about(s, Vec3::new(0.0, 0.0, 1.0), p[1]).then(&vis::about(s, Vec3::new(0.0, 1.0, 0.0), p[0]));
            let e = Point3::new(SHOULDER[0] + 2.0 * UPPER + 2.0 * GAP, SHOULDER[1], SHOULDER[2]);
            let upper = sh
                .then(&Transform::translation(SHOULDER[0] + UPPER + GAP, SHOULDER[1], SHOULDER[2]))
                .then(&Transform::rotation_x(0.35));
            let fore = sh
                .then(&vis::about(e, Vec3::new(0.0, 1.0, 0.0), p[2]))
                .then(&Transform::translation(e.x + FORE + GAP, e.y, e.z))
                .then(&Transform::rotation_x(-0.5));
            vec![upper, fore]
        }),
    }
}

fn pose(out: &Path) {
    let ctx = gpu::context().expect("a GPU adapter");
    let (w, h) = (96, 72);
    let rig = arm_rig();
    let mut scene = arm_scene(w, h, 1);
    rig.pose(&mut scene, &PHI0);
    // Static placements in world space; moving ones in their own space,
    // with the rig in the JSON.
    let mut shapes = Vec::new();
    for k in 0..scene.placements().len() {
        let f = format!("pose_{k}.ply");
        let moving = rig.moving.contains(&k);
        if moving {
            ply_local(&out.join(&f), scene.blas()[scene.placements()[k].blas].geometry().mesh());
        } else {
            ply(&out.join(&f), &scene, k);
        }
        shapes.push(format!(
            "{{\"ply\": \"{f}\", \"bsdf\": \"{}\", \"albedo\": {}, \"moving\": {}}}",
            surface_name(&scene, k),
            scene.placements()[k].albedo,
            if moving { rig.moving.iter().position(|&m| m == k).unwrap() as i64 } else { -1 }
        ));
    }
    let extra = format!(
        ",\n  \"theta_arm\": {THETA_ARM:?},\n  \"phi0\": {PHI0:?},\n  \"phi_target\": {PHI_T:?},\n  \"rig\": {{\"shoulder\": {SHOULDER:?}, \"upper\": {UPPER}, \"fore\": {FORE}, \"gap\": {GAP}, \"roll\": [0.35, -0.5]}}"
    );
    fs::write(out.join("pose.json"), scene_json(&scene, &shapes, &extra)).unwrap();
    for &k in &rig.moving {
        ply(&out.join(format!("pose_{k}_world.ply")), &scene, k); // for checking the rig
    }
    if std::env::var("EXPORT_ONLY").is_ok() {
        return;
    }

    // The target at PHI_T, 4096 spp (256 frames of 16), on the device.
    let mut ts = arm_scene(w, h, 16);
    rig.pose(&mut ts, &PHI_T);
    let mut g = GpuPrb::new(ctx, &ts);
    let target = mean_image(&mut g, &ts, &THETA_ARM, 256, 1 << 22);
    pfm(&out.join("irys_pose_target.pfm"), &target, w, h);
    let npx = (w * h * 3) as f64;

    // The pose gradient at PHI0: image at 64 spp for d_img, then per seed
    // the edge term (device sampler, n samples, traced on the device) and
    // the interior term (CPU f64 duals, `spp_int` per pixel).
    let mut s = arm_scene(w, h, 16);
    rig.pose(&mut s, &PHI0);
    let mut gi = GpuPrb::new(ctx, &s);
    gi.update_geometry(&s);
    let base = mean_image(&mut gi, &s, &THETA_ARM, 4, 1 << 23);
    let d_img: Vec<f64> = base.iter().zip(&target).map(|(a, b)| 2.0 * (a - b) / npx).collect();
    let edges = Edges::new(&s);
    let mut csv = String::from("system,bench,n_edge,spp_int,seed,g0,g1,g2,edge0,edge1,edge2,ms_edge,ms_interior,ms\n");
    for &(n_edge, spp_int) in &[(4096usize, 1usize), (16384, 4), (65536, 16)] {
        let mut ps = arm_scene(w, h, 1);
        rig.pose(&mut ps, &PHI0);
        ps.set_points(Some(vec![[0.0, 0.0]; 2 * n_edge]));
        let mut ge = GpuPrb::new(ctx, &ps);
        ge.update_geometry(&ps);
        let mut gedges = GpuEdges::new(&ge, &ps, &edges, &rig.moving, n_edge, 3, true);
        let mut is = arm_scene(w, h, spp_int);
        rig.pose(&mut is, &PHI0);
        let tan = Tangents::rig(&is, &rig, &PHI0);
        for sd in 0..40u64 + 3 {
            let t0 = Instant::now();
            let smp = gedges.sample(&ge, &ps, &rig, &PHI0, 300 + sd);
            let l = gpu::widen(&ge.forward(&ps, &THETA_ARM, 300 + sd));
            let gedge = smp.gradient(&ps, 300 + sd, &l, &d_img);
            let t1 = Instant::now();
            let (gint, _) = interior::interior_with(&is, &tan, &THETA_ARM, 600 + sd, &d_img, None);
            let t2 = Instant::now();
            if sd < 3 {
                continue; // warm-up
            }
            let (me, mi) = ((t1 - t0).as_secs_f64() * 1e3, (t2 - t1).as_secs_f64() * 1e3);
            let _ = writeln!(
                csv,
                "irys,pose,{n_edge},{spp_int},{},{},{},{},{},{},{},{me},{mi},{}",
                sd - 3,
                gedge[0] + gint[0],
                gedge[1] + gint[1],
                gedge[2] + gint[2],
                gedge[0],
                gedge[1],
                gedge[2],
                me + mi
            );
        }
        eprintln!("pose n_edge {n_edge}, interior {spp_int} spp: done");
    }
    fs::write(out.join("irys_pose.csv"), csv).unwrap();

    pose_cd(out, &target);
}

fn pose_cd(out: &Path, target: &[f64]) {
    let ctx = gpu::context().expect("a GPU adapter");
    let rig = arm_rig();
    let mut cs = arm_scene(96, 72, 64);
    let mut gc = GpuPrb::new(ctx, &cs);
    let hh = 0.01;
    let mut cd = String::from("system,bench,param,h,seed,dloss\n");
    for p in 0..3 {
        for sd in 0..32u64 {
            let (mut a, mut b) = (PHI0, PHI0);
            a[p] += hh;
            b[p] -= hh;
            let l = |phi: &[f64; 3], cs: &mut Scene, gc: &mut GpuPrb| {
                rig.pose(cs, phi);
                gc.update_geometry(cs);
                let ia = gpu::widen(&gc.image(cs, &THETA_ARM, 7000 + sd));
                let ib = gpu::widen(&gc.image(cs, &THETA_ARM, 70000 + sd));
                product_loss(&ia, &ib, target)
            };
            let d = (l(&a, &mut cs, &mut gc) - l(&b, &mut cs, &mut gc)) / (2.0 * hh);
            let _ = writeln!(cd, "irys,pose,{p},{hh},{sd},{d}");
        }
    }
    fs::write(out.join("irys_pose_cd.csv"), cd).unwrap();
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next().unwrap_or_else(|| "bench/mitsuba/out".into());
    let out = Path::new(&out);
    fs::create_dir_all(out).unwrap();
    let which: Vec<String> = args.collect();
    let all = which.is_empty();
    if all || which.iter().any(|w| w == "material") {
        material(out);
    }
    if all || which.iter().any(|w| w == "pose") {
        pose(out);
    }
    // Central differences alone, from the targets already written.
    if which.iter().any(|w| w == "cd-material") {
        material_cd(out, &read_pfm(&out.join("irys_material_target.pfm")).0);
    }
    if which.iter().any(|w| w == "cd-pose") {
        pose_cd(out, &read_pfm(&out.join("irys_pose_target.pfm")).0);
    }
}
