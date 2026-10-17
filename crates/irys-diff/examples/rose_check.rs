//! Phase 2d: does irys's rose full match rose's own render? At zero pose,
//! from vcad-render's `--view iso` direction (`(1, 1, 1)`, screen right
//! `(-1, 1, 0)`), this writes
//!
//! - `labels.ppm`: an exact orthographic raster of the trees — red the
//!   body, green the left arm's links, blue its actuators — to lay over
//!   `full/render/rose-full-iso.png`'s alpha, and
//! - `iso.ppm`: irys's own render of the same view, a pinhole 40 m out
//!   (0.1° from orthographic across the robot), so the overlay checks the
//!   scene irys traces, not just the meshes.
//!
//! `cargo run --release -p irys-diff --features gpu --example rose_check -- out/rose_check`

#[path = "rose/arm.rs"]
mod arm;

use std::{fs, io::Write, path::Path, time::Instant};

use irys_core::{Point3, Vec3};
use irys_diff::gpu::{self, GpuPrb};

const H: usize = 1400;

fn main() {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "out/rose_check".into());
    let out = Path::new(&out);
    fs::create_dir_all(out).unwrap();
    let arm = arm::load_arm(&arm::rose_dir());
    for (name, m) in arm.names.iter().zip(&arm.meshes) {
        eprintln!("  {name:14} {:7} triangles", m.triangles().len());
    }
    for (n, (p, a)) in ["sho pitch", "sho roll", "elbow", "wrist yaw"]
        .iter()
        .zip(&arm.joints)
    {
        eprintln!(
            "  joint {n:10} through ({:.1}, {:.1}, {:.1}) about {a:?}",
            p.x, p.y, p.z
        );
    }

    // World-space trees at zero pose.
    let c = arm.chain(&[0.0; 3]);
    let world: Vec<Vec<Point3>> = arm
        .meshes
        .iter()
        .enumerate()
        .map(|(k, m)| {
            let t = match arm::MOVING.iter().position(|&x| x == k) {
                Some(_) if k <= 3 => &c[[0, 1, 2, 2][k]],
                Some(_) => &c[[0, 0, 1, 2][k - 4]],
                None => &arm.base(),
            };
            m.positions().iter().map(|p| t.apply_point(p)).collect()
        })
        .collect();
    let right = Vec3::new(-1.0, 1.0, 0.0).normalize();
    let down = Vec3::new(1.0, 1.0, -2.0).normalize();
    let view = Vec3::new(1.0, 1.0, 1.0).normalize();
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for p in world.iter().flatten() {
        let v = Vec3::new(p.x, p.y, p.z);
        let s = [v.dot(&right), v.dot(&down)];
        for i in 0..2 {
            lo[i] = lo[i].min(s[i]);
            hi[i] = hi[i].max(s[i]);
        }
    }
    // vcad-render's framing: --fill 0.92 --auto-aspect on a 1400 px height.
    let k = 0.92 * H as f64 / (hi[1] - lo[1]);
    let w = ((hi[0] - lo[0]) * k / 0.92).round() as usize;
    let (ox, oy) = (
        w as f64 / 2.0 - (lo[0] + hi[0]) / 2.0 * k,
        H as f64 / 2.0 - (lo[1] + hi[1]) / 2.0 * k,
    );
    let mut depth = vec![f64::INFINITY; w * H];
    let mut label = vec![0u8; w * H];
    for (tree, (pts, m)) in world.iter().zip(&arm.meshes).enumerate() {
        let lab = match tree {
            0..=3 => 2,
            4..=7 => 3,
            _ => 1,
        };
        let s: Vec<[f64; 3]> = pts
            .iter()
            .map(|p| {
                let v = Vec3::new(p.x, p.y, p.z);
                [v.dot(&right) * k + ox, v.dot(&down) * k + oy, -v.dot(&view)]
            })
            .collect();
        for t in m.triangles() {
            let [a, b, c] = t.map(|i| s[i as usize]);
            let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            if area.abs() < 1e-12 {
                continue;
            }
            let x0 = a[0].min(b[0]).min(c[0]).floor().max(0.0) as usize;
            let x1 = (a[0].max(b[0]).max(c[0]).ceil() as usize).min(w - 1);
            let y0 = a[1].min(b[1]).min(c[1]).floor().max(0.0) as usize;
            let y1 = (a[1].max(b[1]).max(c[1]).ceil() as usize).min(H - 1);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                    let e = |p: [f64; 3], q: [f64; 3]| {
                        ((q[0] - p[0]) * (py - p[1]) - (q[1] - p[1]) * (px - p[0])) / area
                    };
                    let (u, v, wb) = (e(b, c), e(c, a), e(a, b));
                    if u < 0.0 || v < 0.0 || wb < 0.0 {
                        continue;
                    }
                    let z = u * a[2] + v * b[2] + wb * c[2];
                    let i = y * w + x;
                    if z < depth[i] {
                        depth[i] = z;
                        label[i] = lab;
                    }
                }
            }
        }
    }
    let mut f = fs::File::create(out.join("labels.ppm")).unwrap();
    write!(f, "P6\n{w} {H}\n255\n").unwrap();
    let rgb: Vec<u8> = label
        .iter()
        .flat_map(|&l| match l {
            1 => [200, 60, 60],
            2 => [60, 220, 60],
            3 => [60, 90, 230],
            _ => [0, 0, 0],
        })
        .collect();
    f.write_all(&rgb).unwrap();
    eprintln!("labels.ppm {w}×{H}, {k:.4} px/mm");

    // irys: a pinhole far out on the same axis, the same framing.
    let ctx = gpu::context().expect("rose_check needs a GPU adapter");
    let mid = (lo[1] + hi[1]) / 2.0;
    let cv = right * ((lo[0] + hi[0]) / 2.0) + down * mid;
    let ctr = Point3::new(cv.x, cv.y, cv.z);
    // The centre's depth along the view: put it through the robot's middle.
    let cz: f64 = world
        .iter()
        .flatten()
        .map(|p| Vec3::new(p.x, p.y, p.z).dot(&view))
        .sum::<f64>()
        / world.iter().map(|v| v.len()).sum::<usize>() as f64;
    let target = ctr + view * cz;
    let dist = 40_000.0;
    let eye = target + view * dist;
    let half = (H as f64 / 2.0 / k / dist).atan().to_degrees();
    let (sw, sh) = (w / 2, H / 2);
    let mut s = arm::scene_from(
        &arm,
        (sw, sh),
        16,
        &[0.0; 3],
        ([eye.x, eye.y, eye.z], [target.x, target.y, target.z], half),
    );
    s.depth = 4;
    let t = Instant::now();
    let mut g = GpuPrb::new(ctx, &s);
    let frames = 4;
    let mut acc = vec![0.0f32; sw * sh * 3];
    for fr in 0..frames {
        for (a, x) in acc.iter_mut().zip(g.image(&s, &arm::TRUE_THETA, 77 + fr)) {
            *a += x / frames as f32;
        }
    }
    arm::ppm(&out.join("iso.ppm"), (sw, sh), &acc, 0.6);
    eprintln!(
        "iso.ppm {sw}×{sh} (half fov {half:.4}°) in {:.1?}",
        t.elapsed()
    );
}
