//! Phase 4b: hair on the device, cylinders (AABB + exact intersection in
//! the query) against tessellated tubes (triangles), on `look_hair`'s brown
//! haircut: primary-ray throughput, path throughput, and each frame against
//! the CPU f64 reference at the same seed.
//!
//! `cargo run --release -p irys-diff --features gpu,assets --example look_hair_gpu -- [w] [h] [spp] [cpu_spp]`

use irys_core::{Bvh, Point3, Ray, Vec3};
use irys_diff::look::hair::{Fiber, HairScene, Strands};
use irys_diff::look::hair_gpu::{self, GpuHair, GpuStrands, StrandPrim};
use irys_diff::skin::render::Light;

#[path = "look_hair.rs"]
#[allow(dead_code)]
mod look_hair;

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    let (w, h, spp) = (a.first().copied().unwrap_or(280), a.get(1).copied().unwrap_or(360), a.get(2).copied().unwrap_or(64));
    let cpu_spp = a.get(3).copied().unwrap_or(16);
    let ctx = irys_diff::gpu::context().expect("a GPU");
    assert!(hair_gpu::available(ctx), "ray queries");
    let mut s = Strands::default();
    let c = Point3::new(0.0, 0.0, 0.0);
    look_hair::haircut(&mut s, c, 9000, 8);
    let segs = s.segs.len();
    let scene = |spp: usize, s: Strands| HairScene {
        bvh: Bvh::build(s),
        fiber: Fiber::default(),
        scalp: Some((c, 0.995, [0.35, 0.22, 0.17])),
        lights: vec![
            Light::facing(Point3::new(-2.4, 3.6, 3.0), c, 2.0, 2.0, 7.0, [1.0, 0.96, 0.9]),
            Light::facing(Point3::new(1.0, -3.0, 3.6), c, 3.0, 1.0, 6.0, [1.0; 3]),
            Light::facing(Point3::new(3.5, 2.0, 0.0), c, 3.0, 3.0, 0.8, [0.9, 0.95, 1.0]),
        ],
        env: 0.015,
        eye: Point3::new(2.6, 7.4, 1.6),
        target: Point3::new(0.0, 0.0, -0.25),
        vfov: 30.0,
        size: (w, h),
        spp,
        depth: 24,
        sampler: None,
    };
    let theta = [1.3, 0.2];
    let cpu_s = scene(cpu_spp, s.clone());
    let t0 = std::time::Instant::now();
    let cpu = cpu_s.forward(&theta, 0x4a1);
    let cpu_secs = t0.elapsed().as_secs_f64();
    let cpu_img = cpu_s.image(&cpu);
    println!(
        "{segs} segments; CPU f64 reference {w}×{h} × {cpu_spp} spp: {cpu_secs:.1} s ({:.3} Mpaths/s)",
        cpu.len() as f64 / cpu_secs / 1e6
    );
    // Primary rays: a dense grid of camera rays, timed on the device.
    let fwd = (cpu_s.target - cpu_s.eye).normalize();
    let right = fwd.cross(&Vec3::new(0.0, 0.0, 1.0)).normalize();
    let up = right.cross(&fwd);
    let th = (0.5 * cpu_s.vfov).to_radians().tan();
    let (pw, ph) = (1024usize, 1024usize);
    let rays: Vec<([f32; 3], [f32; 3])> = (0..pw * ph)
        .map(|i| {
            let (x, y) = ((i % pw) as f64 + 0.5, (i / pw) as f64 + 0.5);
            let d = (fwd + right * ((2.0 * x / pw as f64 - 1.0) * th * w as f64 / h as f64) + up * ((1.0 - 2.0 * y / ph as f64) * th)).normalize();
            let o = cpu_s.eye;
            ([o.x as f32, o.y as f32, o.z as f32], [d.x as f32, d.y as f32, d.z as f32])
        })
        .collect();
    let cpu_hits: Vec<Option<u32>> = rays
        .iter()
        .map(|(o, d)| {
            let o = Point3::new(o[0] as f64, o[1] as f64, o[2] as f64);
            let d = Vec3::new(d[0] as f64, d[1] as f64, d[2] as f64).normalize();
            cpu_s.bvh.trace_closest_range(&Ray::new(o, d), 1e-7, f64::INFINITY).map(|h| h.prim)
        })
        .collect();
    // Converged frames: cylinders (which walk the CPU's paths, above) at
    // 256 spp are the reference; every kind at another seed against it.
    let conv = scene(256, s.clone());
    let gref = GpuHair::new(ctx, &conv, StrandPrim::Cylinders).unwrap();
    let ref_img = conv.image(&gref.forward(&conv, &theta, 11));
    let converged = |prim: StrandPrim| {
        let g = GpuHair::new(ctx, &conv, prim).unwrap();
        let img = conv.image(&g.forward(&conv, &theta, 12));
        let (mut num, mut den, mut sa, mut sb) = (0.0, 0.0, 0.0, 0.0);
        for (x, y) in img.iter().zip(&ref_img) {
            num += (x - y).abs();
            den += y;
            sa += x;
            sb += y;
        }
        (100.0 * (sa - sb) / sb, 100.0 * num / den)
    };
    for prim in [StrandPrim::Cylinders, StrandPrim::Tubes(3), StrandPrim::Tubes(4), StrandPrim::Tubes(6), StrandPrim::Tubes(8)] {
        let t0 = std::time::Instant::now();
        let gs = GpuStrands::new(ctx, cpu_s.bvh.geometry(), None, prim).unwrap();
        let build = t0.elapsed().as_secs_f64();
        let _ = hair_gpu::probe(ctx, &gs, &rays[..1024]);
        let (hits, psecs) = hair_gpu::probe(ctx, &gs, &rays);
        let agree = hits
            .iter()
            .zip(&cpu_hits)
            .filter(|((t, seg, _), c)| match c {
                Some(p) => *t >= 0.0 && seg == p,
                None => *t < 0.0,
            })
            .count();
        // Paths: same seed as the CPU at cpu_spp (frame vs the reference),
        // then the timed frame at spp.
        let gs_ref = scene(cpu_spp, s.clone());
        let g = GpuHair::new(ctx, &gs_ref, prim).unwrap();
        let gl = g.forward(&gs_ref, &theta, 0x4a1);
        let img = gs_ref.image(&gl);
        let (mut num, mut den, mut sa, mut sb) = (0.0, 0.0, 0.0, 0.0);
        for (x, y) in img.iter().zip(&cpu_img) {
            num += (x - y).abs();
            den += y;
            sa += x;
            sb += y;
        }
        let same = gl.iter().zip(&cpu).filter(|(g, c)| (g.0 - c.0).abs() <= 1e-3 * c.0.abs().max(1e-3)).count();
        let big = scene(spp, s.clone());
        let gb = GpuHair::new(ctx, &big, prim).unwrap();
        let _ = gb.forward(&big, &theta, 1);
        let t0 = std::time::Instant::now();
        let _ = gb.forward(&big, &theta, 2);
        let secs = t0.elapsed().as_secs_f64();
        let (cm, cmae) = converged(prim);
        println!(
            "  converged (256 spp, vs cylinders at another seed): mean {cm:+.3}%, pixel MAE {cmae:.2}%"
        );
        println!(
            "{prim:?}: {} prims, build {build:.2} s; primary {:.0} Mrays/s, same fibre as CPU {:.4}; \
             frame vs CPU: mean {:+.3}%, pixel MAE {:.2}%, paths within 1e-3 {:.3}; {w}×{h} × {spp} spp in {secs:.2} s ({:.2} Mpaths/s)",
            gs.primitives,
            rays.len() as f64 / psecs / 1e6,
            agree as f64 / rays.len() as f64,
            100.0 * (sa - sb) / sb,
            100.0 * num / den,
            same as f64 / cpu.len() as f64,
            big.paths() as f64 / secs / 1e6
        );
    }
}
