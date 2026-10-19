//! M1 item 6: the residual pass against its host reference, determinism,
//! planted objects found, and the device-resident path (`run_sensor`) equal
//! to the host-upload path.
#![cfg(all(feature = "gpu", feature = "assets"))]

use irys_diff::look::openpbr::OpenPbr;
use irys_diff::look::pinhole::Pinhole;
use irys_diff::look::residual::{Noise, Residual, components, residual_cpu};
use irys_diff::look::scene::{Instance, LookScene, Material, Mesh, mask};
use irys_diff::look::scene_gpu::GpuLook;
use irys_diff::look::sensor::{Sensor, aov};
use irys_diff::prb::Rng;

fn gpu() -> Option<&'static irys_gpu::GpuContext> {
    let c = irys_diff::gpu::context().ok()?;
    c.device.features().contains(irys_gpu::wgpu::Features::EXPERIMENTAL_RAY_QUERY).then_some(c)
}

fn quad() -> Mesh {
    Mesh {
        positions: vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]],
        normals: vec![[0.0, 0.0, 1.0]; 4],
        uvs: vec![[0.0; 2]; 4],
        indices: vec![0, 1, 2, 0, 2, 3],
        material: 0,
    }
}

const W: usize = 96;
const H: usize = 72;

fn pin() -> Pinhole {
    Pinhole {
        width: W,
        height: H,
        fx: 70.0,
        fy: 70.0,
        cx: 47.3,
        cy: 35.8,
        rot: [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        pos: [0.0, 0.0, 2.0],
    }
}

/// The predicted AOVs of a floor at z = 0 seen from 2 m (a wide quad), and
/// a "real" frame: the prediction with noise, plus a near box (a new
/// object), a far-off colour patch (a poster), and a missing-geometry hole.
fn frames() -> (Vec<[u8; 3]>, Vec<f32>, Vec<[f32; 4]>, Vec<f32>, Vec<f32>) {
    let ctx = gpu().unwrap();
    let mut s = LookScene::spheres(vec![Material::new("grey", OpenPbr::default())], 0.0, (W, H));
    s.meshes = vec![quad()];
    s.instances = vec![Instance { mesh: 0, xf: [[8.0, 0.0, 0.0, 0.0], [0.0, 8.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], mask: mask::GEOMETRY }];
    s.env = None;
    let g = GpuLook::new(ctx, &s).unwrap();
    let mut sen = Sensor::new(&g, (W, H), 1);
    sen.set_sigma(&[0.02], 0.0, 0.0);
    let a = sen.render(&[pin().camera()], aov::DIST | aov::SIGMA);
    let dist = a.dist.unwrap();
    let sig = a.sigma.unwrap();
    let k = pin();
    let mut rng = Rng::new(9, 0);
    let pred_rgb: Vec<[f32; 4]> = (0..W * H).map(|p| { let v = 0.15 + 0.1 * ((p % W) as f32 / W as f32); [v, v * 0.9, v * 0.8, 1.0] }).collect();
    let mut rgb = Vec::new();
    let mut z = Vec::new();
    for p in 0..W * H {
        let (u, v) = (p % W, p / W);
        let zp = dist[p] as f64 * k.z_per_dist(u as f64, v as f64);
        let enc = |c: f32| irys_diff::look::splat_gpu::srgb_enc(c as f64);
        let mut c = [0, 1, 2].map(|ch| enc(pred_rgb[p][ch]) + 0.01 * (rng.uniform() - 0.5));
        let mut zz = zp + 0.005 * (rng.uniform() - 0.5);
        if (20..34).contains(&u) && (10..30).contains(&v) {
            // a box 0.6 m tall standing on the floor
            zz = zp - 0.6;
            c = [0.8, 0.2, 0.1];
        } else if (60..80).contains(&u) && (40..50).contains(&v) {
            // a poster: colour only
            c = [0.1, 0.7, 0.9];
        } else if (5..12).contains(&u) && (55..65).contains(&v) {
            // no range here (a dark hole the stereo did not see), odd colour
            zz = 0.0;
            c = [0.9, 0.9, 0.0];
        }
        rgb.push(c.map(|x| (x.clamp(0.0, 1.0) * 255.0).round() as u8));
        z.push(zz as f32);
    }
    // two adjacent objects split by depth: a second box right of the first,
    // touching it, 0.3 m further back
    for v in 10..30 {
        for u in 34..40 {
            let p = v * W + u;
            z[p] -= 0.3;
            rgb[p] = [200, 40, 30];
        }
    }
    (rgb, z, pred_rgb, dist, sig)
}

#[test]
fn residual_matches_the_host_and_finds_the_objects() {
    let Some(ctx) = gpu() else { return };
    let (rgb, z, pred_rgb, dist, sig) = frames();
    let noise = Noise::default();
    let mut r = Residual::new(ctx, (W, H));
    let out = r.run_host(&rgb, &z, &pred_rgb, &dist, Some(&sig), &pin(), &noise);
    let mask = r.mask();
    let (cmask, cblobs) = residual_cpu(&rgb, &z, &pred_rgb, &dist, Some(&sig), &pin(), (W, H), &noise);
    let diff = mask.iter().zip(&cmask).filter(|(a, b)| a != b).count();
    eprintln!("explained {}/{}, mask vs host: {diff} px differ; {} blobs, {} rounds, {:.2} ms", out.explained, W * H, out.blobs.len(), out.rounds, 1e3 * out.seconds);
    for b in &out.blobs {
        eprintln!("  {b:?}");
    }
    assert!(diff <= 2, "{diff}");
    assert_eq!(out.explained, mask.iter().filter(|&&m| m == 1).count());
    // the components of the device's own mask, exactly
    assert_eq!(out.blobs, components(&mask, &z, (W, H), &noise));
    if diff == 0 {
        assert_eq!(out.blobs, cblobs);
    }
    // the planted ones: box, second box (split by depth), poster, hole
    let find = |x: u32, y: u32| out.blobs.iter().find(|b| b.bbox[0] <= x && x <= b.bbox[2] && b.bbox[1] <= y && y <= b.bbox[3]).copied();
    let b1 = find(25, 20).expect("box");
    let b2 = find(37, 20).expect("second box");
    assert_ne!(b1.root, b2.root, "a depth step splits the two boxes");
    assert_eq!((b1.bbox, b1.pixels), ([20, 10, 33, 29], 14 * 20));
    assert_eq!((b2.bbox, b2.pixels), ([34, 10, 39, 29], 6 * 20));
    assert!((b1.mean_z.unwrap() - 1.4).abs() < 0.02, "{:?}", b1.mean_z);
    let poster = find(70, 45).expect("poster");
    assert_eq!(poster.pixels, 20 * 10);
    let hole = find(8, 60).expect("hole");
    assert_eq!(hole.mean_z, None);
    assert_eq!(out.blobs.len(), 4, "noise stayed explained");
    // deterministic
    let again = r.run_host(&rgb, &z, &pred_rgb, &dist, Some(&sig), &pin(), &noise);
    assert_eq!(again.blobs, out.blobs);
    assert_eq!(r.mask(), mask);
}

#[test]
fn residual_reads_the_sensor_on_the_device() {
    let Some(ctx) = gpu() else { return };
    let (rgb, z, _, dist, sig) = frames();
    // the sensor's own splat layer (zero here: no splats) is the prediction
    let mut s = LookScene::spheres(vec![Material::new("grey", OpenPbr::default())], 0.0, (W, H));
    s.meshes = vec![quad()];
    s.instances = vec![Instance { mesh: 0, xf: [[8.0, 0.0, 0.0, 0.0], [0.0, 8.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], mask: mask::GEOMETRY }];
    s.env = None;
    let g = GpuLook::new(ctx, &s).unwrap();
    let mut sen = Sensor::new(&g, (W, H), 2);
    sen.set_sigma(&[0.02], 0.0, 0.0);
    let cams = [pin().at(pin().rot, [0.3, 0.0, 2.5]).camera(), pin().camera()];
    let a = sen.render(&cams, aov::RGB | aov::DIST | aov::SIGMA);
    let n = W * H;
    let noise = Noise::default();
    let mut r = Residual::new(ctx, (W, H));
    let dev = r.run_sensor(&rgb, &z, &sen, 1, true, &pin(), &noise);
    let rgb_h = &a.rgb.as_ref().unwrap()[n..];
    let host = r.run_host(&rgb, &z, rgb_h, &a.dist.as_ref().unwrap()[n..], Some(&a.sigma.as_ref().unwrap()[n..]), &pin(), &noise);
    assert_eq!(&a.dist.as_ref().unwrap()[n..], &dist[..]);
    assert_eq!(&a.sigma.as_ref().unwrap()[n..], &sig[..]);
    assert_eq!(dev.blobs, host.blobs);
    assert_eq!(dev.explained, host.explained);
}
