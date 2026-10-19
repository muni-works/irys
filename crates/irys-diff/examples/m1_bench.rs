//! M1 timings: batched depth-only cameras (per-camera ms at batch 1/8/64,
//! 544 × 448), `set_instances` with 30 placements, the path tracer batched
//! against one by one, and the residual pass.
//!
//! ```text
//! cargo run --release -p irys-diff --features gpu,assets --example m1_bench
//! ```

use std::time::Instant;

use irys_diff::look::openpbr::OpenPbr;
use irys_diff::look::pinhole::Pinhole;
use irys_diff::look::residual::{Noise, Residual};
use irys_diff::look::scene::{Camera, Instance, LookScene, Material, Mesh, mask};
use irys_diff::look::scene_gpu::{GpuLook, Opts};
use irys_diff::look::sensor::{Sensor, aov};

/// A UV sphere (unit radius).
fn sphere(material: usize) -> Mesh {
    let (lat, lon) = (24u32, 48u32);
    let mut m = Mesh { material, ..Mesh::default() };
    for i in 0..=lat {
        let th = std::f64::consts::PI * i as f64 / lat as f64;
        for j in 0..=lon {
            let ph = std::f64::consts::TAU * j as f64 / lon as f64;
            let d = [th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()].map(|x| x as f32);
            m.positions.push(d);
            m.normals.push(d);
            m.uvs.push([0.0; 2]);
        }
    }
    for i in 0..lat {
        for j in 0..lon {
            let a = i * (lon + 1) + j;
            m.indices.extend([a, a + lon + 1, a + 1, a + 1, a + lon + 1, a + lon + 2]);
        }
    }
    m
}

fn quad(material: usize) -> Mesh {
    Mesh {
        positions: vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]],
        normals: vec![[0.0, 0.0, 1.0]; 4],
        uvs: vec![[0.0; 2]; 4],
        indices: vec![0, 1, 2, 0, 2, 3],
        material,
    }
}

fn placements(phase: f32) -> Vec<Instance> {
    let mut v = vec![Instance { mesh: 0, xf: [[40.0, 0.0, 0.0, 0.0], [0.0, 40.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], mask: mask::GEOMETRY }];
    for i in 0..29 {
        let a = i as f32 * 0.7 + phase;
        let r = 0.2 + 0.05 * (i % 5) as f32;
        v.push(Instance {
            mesh: 1,
            xf: [[r, 0.0, 0.0, 3.0 * a.cos() + 0.1 * i as f32], [0.0, r, 0.0, 3.0 * a.sin()], [0.0, 0.0, r, r]],
            mask: mask::GEOMETRY,
        });
    }
    v
}

fn main() {
    let ctx = irys_diff::gpu::context().expect("gpu");
    let (w, h) = (544usize, 448usize);
    let mut scene = LookScene::spheres(vec![Material::new("grey", OpenPbr::default())], 1.0, (w, h));
    scene.meshes = vec![quad(0), sphere(0)];
    scene.instances = placements(0.0);
    let mut g = GpuLook::new(ctx, &scene).unwrap();
    let k1 = Pinhole::from_camera_info(
        r#"{"width": 544, "height": 448, "k": [204.63924771337284, 0.0, 277.81055335998536, 0.0, 204.63924771337284, 221.24615388758042, 0.0, 0.0, 1.0], "d": [0,0,0,0,0]}"#,
    )
    .unwrap();
    let cams: Vec<Camera> = (0..64)
        .map(|i| {
            let a = i as f64 * 0.098;
            // looking outward from the centre, level, 0.87 m up
            let f = [a.cos(), a.sin(), 0.0];
            let x = [a.sin(), -a.cos(), 0.0];
            let rot = [[x[0], 0.0, f[0]], [x[1], 0.0, f[1]], [x[2], -1.0, f[2]]];
            k1.at(rot, [0.0, 0.0, 0.87]).camera()
        })
        .collect();
    println!("## batched depth-only cameras, 544 × 448, 30 instances (≈ {} placed triangles)", scene.triangles().1);
    println!("| batch | total ms | per camera ms |");
    println!("|---|---|---|");
    let mut sen = Sensor::new(&g, (w, h), 64);
    for &n in &[1usize, 8, 64] {
        sen.render(&cams[..n], aov::DIST);
        let mut t = Vec::new();
        for _ in 0..9 {
            let t0 = Instant::now();
            let a = sen.render(&cams[..n], aov::DIST);
            t.push(t0.elapsed().as_secs_f64());
            assert_eq!(a.dist.unwrap().len(), n * w * h);
        }
        t.sort_by(f64::total_cmp);
        println!("| {n} | {:.2} | {:.3} |", 1e3 * t[4], 1e3 * t[4] / n as f64);
    }
    // all AOVs, for the record
    let mut t = Vec::new();
    for _ in 0..5 {
        let t0 = Instant::now();
        sen.render(&cams[..8], aov::ALL);
        t.push(t0.elapsed().as_secs_f64());
    }
    t.sort_by(f64::total_cmp);
    println!("all six AOVs, batch 8: {:.3} ms per camera", 1e3 * t[2] / 8.0);

    println!("\n## set_instances, 30 placements");
    let mut t = Vec::new();
    for i in 0..41 {
        t.push(g.set_instances(&placements(0.01 * i as f32)).unwrap());
    }
    t.sort_by(f64::total_cmp);
    println!("median {:.3} ms, min {:.3} ms, max {:.3} ms", 1e3 * t[20], 1e3 * t[0], 1e3 * t[40]);

    println!("\n## path tracer, 4 spp, depth 2: batched vs one by one");
    let o = Opts { spp: 4, max_bounces: 2, ..Opts::default() };
    for &n in &[1usize, 8] {
        g.render_batch(&cams[..n], &o, false);
        let t0 = Instant::now();
        g.render_batch(&cams[..n], &o, false);
        let tb = t0.elapsed().as_secs_f64();
        let t0 = Instant::now();
        for c in &cams[..n] {
            g.render(c, &o);
        }
        let ts = t0.elapsed().as_secs_f64();
        println!("batch {n}: {:.2} ms per camera batched, {:.2} ms one by one", 1e3 * tb / n as f64, 1e3 * ts / n as f64);
    }

    println!("\n## residual pass, 544 × 448");
    let a = sen.render(&cams[..1], aov::RGB | aov::DIST);
    let dist = a.dist.unwrap();
    let real_z: Vec<f32> = (0..w * h)
        .map(|p| {
            let (u, v) = ((p % w) as f64, (p / w) as f64);
            let z = dist[p] as f64 * k1.z_per_dist(u, v);
            if z.is_finite() && (0.4..4.0).contains(&z) { (z * if (100..160).contains(&(p % w)) { 0.8 } else { 1.0 }) as f32 } else { 0.0 }
        })
        .collect();
    let real_rgb = vec![[0u8; 3]; w * h];
    let mut r = Residual::new(ctx, (w, h));
    let mut t = Vec::new();
    let mut out = None;
    for _ in 0..7 {
        let o = r.run_sensor(&real_rgb, &real_z, &sen, 0, false, &k1, &Noise::default());
        t.push(o.seconds);
        out = Some(o);
    }
    t.sort_by(f64::total_cmp);
    let o = out.unwrap();
    println!("{:.2} ms (median of 7), {} rounds, {} blobs, {} explained", 1e3 * t[3], o.rounds, o.blobs.len(), o.explained);
}
