//! M1 (docs/belief-codec.md): splats in the scene kernel, AOVs, batched
//! cameras, TLAS-only rebuilds, the OpenCV pinhole.
//!
//! - splats on the device against `splat_gpu::composite_cpu` (f64, brute
//!   force), with a mesh occluding part of the cloud;
//! - determinism: AOVs and frames bit-identical run to run, batch against
//!   one-by-one, `set_instances` against a scene built with the same
//!   placements;
//! - the pinhole against analytic ray–plane depths and OpenCV projection.
//!
//! Skips without a device or without ray queries.
#![cfg(all(feature = "gpu", feature = "assets"))]

use irys_diff::look::openpbr::OpenPbr;
use irys_diff::look::scene::{Camera, Instance, LookScene, Material, Mesh, mask};
use irys_diff::look::scene_gpu::{GpuLook, Opts};
use irys_diff::look::pinhole::Pinhole;
use irys_diff::look::sensor::{MISS, SPLAT_ID, Sensor, aov, pixel_angle};
use irys_diff::look::splat_gpu::{self, PlyCloud};
use irys_diff::prb::Rng;

fn gpu() -> Option<&'static irys_gpu::GpuContext> {
    let c = irys_diff::gpu::context().ok()?;
    c.device.features().contains(irys_gpu::wgpu::Features::EXPERIMENTAL_RAY_QUERY).then_some(c)
}

/// A unit quad in z = 0, centred at the origin.
fn quad() -> Mesh {
    Mesh {
        positions: vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]],
        normals: vec![[0.0, 0.0, 1.0]; 4],
        uvs: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        indices: vec![0, 1, 2, 0, 2, 3],
        material: 0,
    }
}

fn place(x: f32, y: f32, z: f32, s: f32) -> Instance {
    Instance { mesh: 0, xf: [[s, 0.0, 0.0, x], [0.0, s, 0.0, y], [0.0, 0.0, s, z]], mask: mask::GEOMETRY }
}

fn scene(instances: Vec<Instance>, size: (usize, usize)) -> LookScene {
    let mut s = LookScene::spheres(vec![Material::new("grey", OpenPbr::default())], 0.0, size);
    s.meshes = vec![quad()];
    s.instances = instances;
    s.env = None;
    s
}

/// A camera at (0, 0, 3) looking down −z (optical y = world −y).
fn down(size: (usize, usize), f: f64, cx: f64, cy: f64) -> Pinhole {
    Pinhole {
        width: size.0,
        height: size.1,
        fx: f,
        fy: f * 1.05,
        cx,
        cy,
        rot: [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        pos: [0.0, 0.0, 3.0],
    }
}

fn cloud(n: usize, degree: u32, seed: u64) -> PlyCloud {
    let mut r = Rng::new(seed, 0);
    let s = (degree as usize + 1).pow(2);
    let mut c = PlyCloud { degree, ..Default::default() };
    for _ in 0..n {
        let mut u = || r.uniform() as f32;
        c.positions.push([1.6 * u() - 0.8, 1.6 * u() - 0.8, 0.8 * u() - 0.4]);
        c.scales.push([0.02 + 0.08 * u(), 0.02 + 0.08 * u(), 0.01 + 0.05 * u()]);
        let q = [u() - 0.5, u() - 0.5, u() - 0.5, u() - 0.5];
        let l = (q.iter().map(|x| x * x).sum::<f32>()).sqrt();
        c.quats.push(q.map(|x| x / l));
        c.opacities.push(0.2 + 0.79 * u());
        for k in 0..s {
            let a = if k == 0 { 1.5 } else { 0.4 };
            c.sh.push([a * (u() - 0.5), a * (u() - 0.5), a * (u() - 0.5)]);
        }
    }
    c
}

#[test]
fn splats_match_the_f64_twin_with_a_mesh_in_the_cloud() {
    let Some(ctx) = gpu() else { return };
    let size = (48, 40);
    // the quad at z = 0 cuts the cloud in half: splats under it are hidden
    let sc = scene(vec![place(0.0, 0.0, 0.0, 0.9)], size);
    let c = cloud(400, 3, 7);
    let g = GpuLook::with_splats(ctx, &sc, Some(&c)).unwrap();
    let pin = down(size, 55.0, 23.3, 20.6);
    let mut sen = Sensor::new(&g, size, 1);
    let pix = pixel_angle(&pin.camera(), size.1);
    assert!((pix - 1.0 / pin.fx).abs() < 1e-3 / pin.fx);
    let a = sen.render(&[pin.camera()], aov::ALL);
    let (rgb, dist, inst) = (a.rgb.as_ref().unwrap(), a.dist.as_ref().unwrap(), a.inst.as_ref().unwrap());
    let mut worst = 0f64;
    let mut worst_t = 0f64;
    let mut over = 0;
    let (mut n_splat, mut n_quad, mut n_miss) = (0, 0, 0);
    for v in 0..size.1 {
        for u in 0..size.0 {
            let k = v * size.0 + u;
            let d = pin.ray(u as f64, v as f64);
            // the quad, analytically: z = 0, |x|, |y| ≤ 0.45
            let t_q = (0.0 - pin.pos[2]) / d[2];
            let x = [pin.pos[0] + t_q * d[0], pin.pos[1] + t_q * d[1]];
            let on_quad = x[0].abs() < 0.45 && x[1].abs() < 0.45;
            let tmax = if on_quad { t_q } else { f64::INFINITY };
            let seg = splat_gpu::composite_cpu(&c, pin.pos, d, pix, 0.0, tmax);
            let lin = splat_gpu::linear(&seg);
            let e = (0..3).map(|ch| (lin[ch] - rgb[k][ch] as f64).abs()).fold(0.0, f64::max);
            let ea = ((1.0 - seg.trans) - rgb[k][3] as f64).abs();
            worst = worst.max(e.max(ea));
            if e.max(ea) > 1e-3 {
                over += 1;
            }
            match seg.t_half {
                Some(t) => {
                    n_splat += 1;
                    if inst[k] == SPLAT_ID {
                        worst_t = worst_t.max((t - dist[k] as f64).abs());
                    }
                }
                None if on_quad => {
                    n_quad += 1;
                    if x[0].abs() < 0.44 && x[1].abs() < 0.44 {
                        assert_eq!(inst[k], 0, "pixel {u},{v}");
                        assert!((dist[k] as f64 - t_q).abs() < 1e-4 * t_q, "{} vs {t_q}", dist[k]);
                    }
                }
                None => {
                    n_miss += 1;
                    if !(x[0].abs() < 0.46 && x[1].abs() < 0.46) {
                        assert_eq!(inst[k], MISS);
                        assert_eq!(dist[k], f32::INFINITY);
                    }
                }
            }
        }
    }
    eprintln!("splat px {n_splat}, quad px {n_quad}, miss {n_miss}; worst |Δ| {worst:.2e}, {over} px > 1e-3; worst Δt {worst_t:.2e}");
    assert!(n_splat > 200 && n_quad > 50 && n_miss > 50);
    assert!(over <= size.0 * size.1 / 200, "{over} pixels off");
    assert!(worst_t < 1e-3);
}

fn five_cameras(ctx: &'static irys_gpu::GpuContext, size: (usize, usize)) -> (GpuLook, Vec<Camera>) {
    let sc = scene((0..6).map(|i| place(-1.0 + 0.4 * i as f32, 0.1 * i as f32, -0.2 * i as f32, 0.35)).collect(), size);
    let c = cloud(300, 1, 3);
    let g = GpuLook::with_splats(ctx, &sc, Some(&c)).unwrap();
    let cams = (0..5)
        .map(|i| {
            let f = 20.0 + 3.0 * i as f64;
            down(size, f, 19.5 + i as f64, 14.5 - i as f64)
                .at([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], [0.1 * i as f64, -0.05 * i as f64, 2.5 + 0.2 * i as f64])
                .camera()
        })
        .collect();
    (g, cams)
}

#[test]
fn aovs_are_deterministic_and_batches_match_singles() {
    let Some(ctx) = gpu() else { return };
    let size = (40, 30);
    let (g, cams) = five_cameras(ctx, size);
    let mut sen = Sensor::new(&g, size, 8);
    let b1 = sen.render(&cams, aov::ALL);
    let b2 = sen.render(&cams, aov::ALL);
    assert_eq!(b1.dist, b2.dist);
    assert_eq!(b1.rgb, b2.rgb);
    let n = size.0 * size.1;
    for (i, cam) in cams.iter().enumerate() {
        let s = sen.render(std::slice::from_ref(cam), aov::ALL);
        let r = i * n..(i + 1) * n;
        assert_eq!(s.dist.as_deref().unwrap(), &b1.dist.as_ref().unwrap()[r.clone()]);
        assert_eq!(s.inst.as_deref().unwrap(), &b1.inst.as_ref().unwrap()[r.clone()]);
        assert_eq!(s.mat.as_deref().unwrap(), &b1.mat.as_ref().unwrap()[r.clone()]);
        assert_eq!(s.normal.as_deref().unwrap(), &b1.normal.as_ref().unwrap()[r.clone()]);
        assert_eq!(s.rgb.as_deref().unwrap(), &b1.rgb.as_ref().unwrap()[r]);
    }
    // a depth-only read returns the same depths
    let d = sen.render(&cams, aov::DIST);
    assert_eq!(d.dist, b1.dist);
    assert!(d.rgb.is_none());
    // splats show in the path tracer's frame: 64 spp of one camera against
    // the sensor's (unjittered) splat layer, as means over the frame
    let f = g.render(&cams[0], &Opts { spp: 64, max_bounces: 0, ..Opts::default() });
    let mean_pt: f64 = f.rgb.iter().map(|&x| x as f64).sum::<f64>() / f.rgb.len() as f64;
    let lay = &b1.rgb.as_ref().unwrap()[..n];
    let mean_s: f64 = lay.iter().flat_map(|p| [p[0], p[1], p[2]]).map(|x| x as f64).sum::<f64>() / (3 * n) as f64;
    eprintln!("path tracer mean {mean_pt:.4}, splat layer mean {mean_s:.4}");
    assert!(mean_s > 0.01);
    assert!((mean_pt - mean_s).abs() < 0.08 * mean_s, "{mean_pt} vs {mean_s}");
}

#[test]
fn the_pinhole_is_opencv() {
    let Some(ctx) = gpu() else { return };
    let size = (64, 48);
    // a tilted plane filling the view: depth varies with u and v, so the
    // focal lengths and the principal point all show
    let mut sc = scene(vec![], size);
    let tilt = Instance {
        mesh: 0,
        xf: [[20.0, 0.0, 0.0, 0.0], [0.0, 17.0, -10.0, 0.0], [0.0, 10.0, 17.0, -0.5]],
        mask: mask::GEOMETRY,
    };
    sc.instances = vec![tilt];
    let g = GpuLook::new(ctx, &sc).unwrap();
    let info = r#"{"width": 64, "height": 48, "k": [41.2, 0.0, 35.7, 0.0, 40.1, 21.1, 0.0, 0.0, 1.0], "d": [0,0,0,0,0]}"#;
    let pin = Pinhole::from_camera_info(info).unwrap().at([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], [0.1, 0.2, 3.0]);
    assert_eq!((pin.cx, pin.cy, pin.fy), (35.7, 21.1, 40.1));
    let mut sen = Sensor::new(&g, size, 1);
    let a = sen.render(&[pin.camera()], aov::DIST | aov::NORMAL);
    let dist = a.dist.unwrap();
    // the plane: through (0, 0, −0.5), normal = third column of xf's rotation
    let nrm = { let v = [0.0f64, -10.0, 17.0]; let l = (v[1] * v[1] + v[2] * v[2]).sqrt(); [0.0, v[1] / l, v[2] / l] };
    let mut worst = 0f64;
    for v in 0..size.1 {
        for u in 0..size.0 {
            let d = pin.ray(u as f64, v as f64);
            let o = [pin.pos[0], pin.pos[1], pin.pos[2] + 0.5];
            let t = -(o[0] * nrm[0] + o[1] * nrm[1] + o[2] * nrm[2]) / (d[0] * nrm[0] + d[1] * nrm[1] + d[2] * nrm[2]);
            let got = dist[v * size.0 + u] as f64;
            worst = worst.max((got - t).abs() / t);
            // and back through OpenCV's projection
            let p = std::array::from_fn(|k| pin.pos[k] + t * d[k]);
            let q = pin.project(p).unwrap();
            assert!((q[0] - u as f64).abs() < 1e-9 && (q[1] - v as f64).abs() < 1e-9);
            assert!((q[2] - t * pin.z_per_dist(u as f64, v as f64)).abs() < 1e-9);
        }
    }
    eprintln!("pinhole: worst relative depth error {worst:.2e}");
    assert!(worst < 2e-5, "{worst}");
    let n = a.normal.unwrap()[size.0 * 10 + 10];
    assert!((n[1] as f64 - nrm[1]).abs() < 1e-3 && (n[2] as f64 - nrm[2]).abs() < 1e-3, "{n:?}");
    // the path tracer's camera agrees: its pixel centres are the same rays
    // (a mesh edge seen through the camera lands where OpenCV projects it)
    let cam = pin.camera();
    let d = |u: f64, v: f64| {
        let w = size.0 as f64;
        let h = size.1 as f64;
        let sx = (2.0 * (u + 0.5) / w - 1.0) * cam.tan_half * w / h;
        let sy = (1.0 - 2.0 * (v + 0.5) / h) * cam.tan_half;
        let r: [f64; 3] = std::array::from_fn(|k| cam.forward[k] + cam.right[k] * sx + cam.up[k] * sy);
        let l = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        r.map(|x| x / l)
    };
    for (u, v) in [(0.0, 0.0), (63.0, 0.0), (35.7, 21.1), (10.0, 40.0)] {
        let (a, b) = (d(u, v), pin.ray(u, v));
        assert!((0..3).all(|k| (a[k] - b[k]).abs() < 1e-12), "{u},{v}");
    }
}

#[test]
fn sigma_follows_what_the_pixel_sees() {
    let Some(ctx) = gpu() else { return };
    let size = (32, 32);
    let g = GpuLook::new(ctx, &scene(vec![place(-0.4, 0.0, 0.0, 0.5), place(0.4, 0.0, 0.0, 0.5)], size)).unwrap();
    let mut sen = Sensor::new(&g, size, 1);
    sen.set_sigma(&[0.25, 0.5], 9.0, 7.0);
    let a = sen.render(&[down(size, 40.0, 15.5, 15.5).camera()], aov::INST | aov::SIGMA);
    let (inst, sig) = (a.inst.unwrap(), a.sigma.unwrap());
    let mut seen = [0; 3];
    for (i, s) in inst.iter().zip(&sig) {
        let want = match *i {
            0 => 0.25,
            1 => 0.5,
            MISS => 7.0,
            o => panic!("id {o}"),
        };
        seen[(*i).min(2) as usize] += 1;
        assert_eq!(*s, want);
    }
    assert!(seen.iter().all(|&n| n > 20), "{seen:?}");
}

#[test]
fn render_batch_matches_render() {
    let Some(ctx) = gpu() else { return };
    let (g, cams) = five_cameras(ctx, (40, 30));
    // the path tracer: batch = singles, bit for bit; spectrum optional
    let o = Opts { spp: 3, max_bounces: 3, ..Opts::default() };
    let fb = g.render_batch(&cams, &o, true);
    let fb2 = g.render_batch(&cams, &o, false);
    for (i, cam) in cams.iter().enumerate() {
        let f = g.render(cam, &o);
        assert_eq!(f.rgb, fb[i].rgb, "camera {i}");
        assert_eq!(f.spectrum, fb[i].spectrum);
        assert_eq!(f.rgb, fb2[i].rgb);
        assert!(fb2[i].spectrum.is_empty());
    }
}
