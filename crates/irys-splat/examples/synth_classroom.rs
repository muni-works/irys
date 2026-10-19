//! A synthetic posed dataset rendered by irys itself: Blender's classroom
//! (the 5b export, true OpenPBR materials) from `--n` cameras along a
//! walk-through, at the K1's 544 × 448 and intrinsics, with exact depth
//! from the sensor AOV. Written in the trainer's input format
//! (`images/`, `depth/`, `poses.jsonl`, `camera_info.json`).
//!
//! ```text
//! cargo run --release -p irys-splat --example synth_classroom -- \
//!     <export5b/classroom> <out dir> [--n 150] [--spp 512] [--bounds]
//! ```

use std::path::PathBuf;

use irys_diff::look::pinhole::Pinhole;
use irys_diff::look::scene::{self};
use irys_diff::look::scene_gpu::{GpuLook, Opts};
use irys_diff::look::sensor::{Sensor, aov};
use irys_splat::data::{self, View};

fn srgb(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    if x <= 0.003_130_8 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
}

fn norm(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    v.map(|x| x / l)
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

/// OpenCV camera → world rotation looking from `eye` at `target`, z up.
fn look_at(eye: [f64; 3], target: [f64; 3]) -> [[f64; 3]; 3] {
    let f = norm([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
    let r = norm(cross(f, [0.0, 0.0, 1.0]));
    let d = cross(f, r); // image down
    [[r[0], d[0], f[0]], [r[1], d[1], f[1]], [r[2], d[2], f[2]]]
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let dir = PathBuf::from(&a[0]);
    let out = PathBuf::from(&a[1]);
    let opt = |k: &str| a.iter().position(|x| x == k).map(|i| a[i + 1].clone());
    let n: usize = opt("--n").map_or(150, |s| s.parse().unwrap());
    let spp: u32 = opt("--spp").map_or(512, |s| s.parse().unwrap());
    let (w, h) = (544usize, 448usize);
    let mut sc = scene::load(&dir, Some(w)).expect("scene");
    sc.size = (w, h);
    if a.iter().any(|x| x == "--bounds") {
        let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
        for i in &sc.instances {
            for p in sc.meshes[i.mesh].positions.iter().step_by(3) {
                for r in 0..3 {
                    let v = i.xf[r][0] * p[0] + i.xf[r][1] * p[1] + i.xf[r][2] * p[2] + i.xf[r][3];
                    lo[r] = lo[r].min(v);
                    hi[r] = hi[r].max(v);
                }
            }
        }
        println!("bounds {lo:?} {hi:?}; camera {:?}", sc.camera);
        return;
    }
    // the K1's colour camera
    let k = Pinhole {
        width: w,
        height: h,
        fx: 204.63924771337284,
        fy: 204.63924771337284,
        cx: 277.81055335998536,
        cy: 221.24615388758042,
        rot: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        pos: [0.0; 3],
    };
    // the walk: an ellipse inside the room at head height, the gaze swept
    // across the room's centre with a slow yaw and pitch wobble
    let c0 = [
        opt("--cx").map_or(0.6, |s| s.parse().unwrap()),
        opt("--cy").map_or(-0.8, |s| s.parse().unwrap()),
    ];
    let (rx, ry) = (opt("--rx").map_or(1.6, |s| s.parse().unwrap()), opt("--ry").map_or(2.4, |s| s.parse::<f64>().unwrap()));
    let cams: Vec<Pinhole> = (0..n)
        .map(|i| {
            let s = i as f64 / n as f64;
            let th = std::f64::consts::TAU * s;
            let eye = [c0[0] + rx * th.cos(), c0[1] + ry * th.sin(), 1.25 + 0.25 * (3.0 * th).sin()];
            // look across the room, swinging ±50° about the inward direction
            let inward = (c0[1] - eye[1]).atan2(c0[0] - eye[0]);
            let yaw = inward + 0.9 * (2.0 * th).sin();
            let pitch = -0.15 + 0.2 * (5.0 * th).cos();
            let f = [yaw.cos() * pitch.cos(), yaw.sin() * pitch.cos(), pitch.sin()];
            k.at(look_at(eye, [eye[0] + f[0], eye[1] + f[1], eye[2] + f[2]]), eye)
        })
        .collect();
    let ctx = irys_gpu::GpuContext::init_blocking().expect("gpu");
    let gpu = GpuLook::new(ctx, &sc).expect("ray queries");
    let mut sensor = Sensor::new(&gpu, (w, h), 1);
    let o = Opts {
        spp,
        max_bounces: 7,
        clamp: opt("--clamp").map_or(4.0, |s| s.parse().unwrap()),
        seed: 1,
        rr_from: u32::MAX,
        debug: 0,
        per_lane: false,
        prof: 0,
        samp: irys_diff::look::scene_gpu::samp::ALL,
        regen: 16,
        guide: 0.0,
    };
    let exposure: f32 = opt("--exposure").map_or(1.0, |s| s.parse().unwrap());
    let t0 = std::time::Instant::now();
    let mut views = Vec::new();
    for (i, cam) in cams.iter().enumerate() {
        let kc = cam.camera();
        let f = gpu.render(&kc, &Opts { seed: 1 + i as u64, ..o });
        let rgb: Vec<[f32; 3]> = f.rgb.chunks_exact(3).map(|c| [srgb(exposure * c[0]), srgb(exposure * c[1]), srgb(exposure * c[2])]).collect();
        let d = sensor.render(&[kc], aov::DIST).dist.unwrap();
        let depth: Vec<f32> = (0..w * h)
            .map(|p| {
                let z = d[p] as f64 * cam.z_per_dist((p % w) as f64, (p / w) as f64);
                if z.is_finite() { z as f32 } else { 0.0 }
            })
            .collect();
        eprintln!("view {i:3}/{n}: {:.2} s render", f.seconds);
        views.push(View { name: format!("{i:06}"), cam: *cam, rgb, depth: Some(depth) });
    }
    data::save(&out, &views).expect("save");
    eprintln!("{} views in {:.0} s → {}", n, t0.elapsed().as_secs_f64(), out.display());
}
