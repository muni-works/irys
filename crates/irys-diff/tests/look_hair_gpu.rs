//! Hair on the device: strand intersection (exact cylinders through AABBs,
//! and tessellated tubes) against the CPU's round cylinders, the
//! `HairScene` kernel per path against the CPU f64 integrator, and its
//! determinism. Skips without an adapter or ray queries.
#![cfg(feature = "gpu")]

use irys_core::{Bvh, Point3, Ray, Vec3};
use irys_diff::look::hair::{Fiber, HairScene, Strands};
use irys_diff::look::hair_gpu::{self, GpuHair, GpuStrands, StrandPrim};
use irys_diff::prb::Rng;
use irys_diff::skin::render::Light;

fn ctx() -> Option<&'static irys_gpu::GpuContext> {
    match irys_diff::gpu::context() {
        Ok(c) if hair_gpu::available(c) => Some(c),
        Ok(_) => {
            eprintln!("no ray queries: skipped");
            None
        }
        Err(e) => {
            eprintln!("no GPU ({e:?}): skipped");
            None
        }
    }
}

/// A small haircut on the unit sphere: `n` strands of 10 segments.
fn haircut(n: usize) -> Strands {
    let mut s = Strands::default();
    let mut rng = Rng::new(17, 0);
    let mut placed = 0;
    while placed < n {
        let z = 2.0 * rng.uniform() - 1.0;
        let p = 2.0 * std::f64::consts::PI * rng.uniform();
        let r = (1.0 - z * z).sqrt();
        let nrm = Vec3::new(r * p.cos(), r * p.sin(), z);
        if z < 0.1 {
            continue;
        }
        placed += 1;
        let mut x = Point3::origin() + nrm;
        let mut dir = (nrm + Vec3::new(0.0, 0.5, 0.0)).normalize();
        let mut pts = vec![x];
        for k in 0..10 {
            let t = k as f64 / 10.0;
            dir = (dir + Vec3::new(0.0, 0.0, -1.0) * (0.1 + 0.3 * t)).normalize();
            x = x + dir * 0.07;
            if (x - Point3::origin()).norm() < 1.01 {
                x = Point3::origin() + (x - Point3::origin()).normalize() * 1.01;
            }
            pts.push(x);
        }
        s.push(&pts, 0.004);
    }
    s
}

fn scene(size: usize, spp: usize) -> HairScene {
    HairScene {
        bvh: Bvh::build(haircut(1500)),
        fiber: Fiber::default(),
        scalp: Some((Point3::origin(), 0.995, [0.35, 0.22, 0.17])),
        lights: vec![
            Light::facing(Point3::new(-2.4, 3.6, 3.0), Point3::origin(), 2.0, 2.0, 7.0, [1.0, 0.96, 0.9]),
            Light::facing(Point3::new(1.0, -3.0, 3.6), Point3::origin(), 3.0, 1.0, 6.0, [1.0; 3]),
        ],
        env: 0.05,
        eye: Point3::new(2.6, 7.4, 1.6),
        target: Point3::new(0.0, 0.0, 0.3),
        vfov: 22.0,
        size: (size, size),
        spp,
        depth: 10,
        sampler: None,
    }
}

/// Camera rays through the scene's pixel centres.
fn rays(s: &HairScene) -> Vec<([f32; 3], [f32; 3])> {
    let (w, h) = s.size;
    let fwd = (s.target - s.eye).normalize();
    let right = fwd.cross(&Vec3::new(0.0, 0.0, 1.0)).normalize();
    let up = right.cross(&fwd);
    let th = (0.5 * s.vfov).to_radians().tan();
    let mut out = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let (ix, iy) = (x as f64 + 0.5, y as f64 + 0.5);
            let d = (fwd + right * ((2.0 * ix / w as f64 - 1.0) * th) + up * ((1.0 - 2.0 * iy / h as f64) * th)).normalize();
            let o = s.eye;
            out.push(([o.x as f32, o.y as f32, o.z as f32], [d.x as f32, d.y as f32, d.z as f32]));
        }
    }
    out
}

#[test]
fn strand_hits_match_the_cpu_cylinders() {
    let Some(ctx) = ctx() else { return };
    let s = scene(256, 1);
    let rs = rays(&s);
    let cpu: Vec<Option<(f64, u32, f64)>> = rs
        .iter()
        .map(|(o, d)| {
            let o = Point3::new(o[0] as f64, o[1] as f64, o[2] as f64);
            let d = Vec3::new(d[0] as f64, d[1] as f64, d[2] as f64).normalize();
            s.bvh.trace_closest_range(&Ray::new(o, d), 1e-7, f64::INFINITY).map(|h| {
                let tx = s.bvh.geometry().tangent(h.prim as usize);
                let z = (-(d - tx * d.dot(&tx))).normalize();
                let y = z.cross(&tx);
                (h.t, h.prim, (*h.normal).dot(&y))
            })
        })
        .collect();
    let hits = cpu.iter().filter(|h| h.is_some()).count();
    for prim in [StrandPrim::Cylinders, StrandPrim::Tubes(4), StrandPrim::Tubes(6)] {
        let g = GpuStrands::new(ctx, s.bvh.geometry(), None, prim).unwrap();
        let (gpu, _) = hair_gpu::probe(ctx, &g, &rs);
        let (mut agree, mut same_seg, mut dh, mut dt) = (0usize, 0usize, 0.0f64, 0.0f64);
        for (c, &(t, seg, h)) in cpu.iter().zip(&gpu) {
            let ghit = t >= 0.0;
            if c.is_some() == ghit {
                agree += 1;
            }
            if let Some((ct, cs, ch)) = *c {
                if ghit && seg == cs {
                    same_seg += 1;
                    dh += (ch - h as f64).abs();
                    dt = dt.max((ct - t as f64).abs() / ct);
                }
            }
        }
        let n = rs.len();
        let mh = dh / same_seg.max(1) as f64;
        println!(
            "{prim:?}: {} primitives; hit/miss agree {:.4}, same fibre {:.4} of {hits} hits, mean |Δh| {mh:.2e}, max rel Δt {dt:.1e}",
            g.primitives,
            agree as f64 / n as f64,
            same_seg as f64 / hits as f64
        );
        if prim == StrandPrim::Cylinders {
            assert!(agree as f64 / n as f64 > 0.9995, "cylinders agree with the CPU");
            assert!(same_seg as f64 / hits as f64 > 0.999);
            assert!(mh < 2e-3, "h {mh}");
            assert!(dt < 1e-4, "t {dt}");
        }
    }
}

fn compare(depth: usize) -> (f64, f64, usize, usize, f64) {
    let ctx = ctx().unwrap();
    let mut s = scene(48, 4);
    s.depth = depth;
    let theta = [1.3, 0.2];
    let cpu = s.forward(&theta, 0x77);
    let g = GpuHair::new(ctx, &s, StrandPrim::Cylinders).unwrap();
    let a = g.forward(&s, &theta, 0x77);
    let b = g.forward(&s, &theta, 0x77);
    assert!(a.iter().zip(&b).all(|(x, y)| x.0.to_bits() == y.0.to_bits()), "bit-identical twice");
    let lam = cpu.iter().zip(&a).map(|(c, g)| (c.1 - g.1).abs()).fold(0.0, f64::max);
    assert!(lam < 1e-3, "wavelengths");
    let mut r: Vec<f64> = cpu.iter().zip(&a).map(|(c, g)| (c.0 - g.0).abs() / c.0.abs().max(1e-3)).collect();
    let big = r.iter().filter(|&&x| x > 1e-3).count();
    r.sort_by(f64::total_cmp);
    let (sc, sg): (f64, f64) = (cpu.iter().map(|x| x.0).sum(), a.iter().map(|x| x.0).sum());
    let out = (r[r.len() / 2], r[r.len() * 99 / 100], big, r.len(), (sg - sc).abs() / sc);
    println!(
        "depth {depth}: per path median rel {:.1e}, p99 {:.1e}, > 1e-3: {big} / {}; frame sum rel {:.1e}",
        out.0, out.1, out.3, out.4
    );
    out
}

/// The kernel walks the CPU's paths: at depth 2 (camera → fibre or scalp →
/// one more vertex) nearly every path agrees to f32; at depth 10 the
/// multiple scattering through the fibres lets f32 rounding move lobe
/// choices on more paths, and the frame still agrees.
#[test]
fn hair_paths_match_the_cpu_and_repeat() {
    if ctx().is_none() {
        return;
    }
    let (med, _, big, n, sum) = compare(2);
    assert!(med < 1e-4 && (big as f64) < 0.03 * n as f64 && sum < 1e-3);
    let (med, _, big, n, sum) = compare(10);
    assert!(med < 1e-4 && (big as f64) < 0.2 * n as f64 && sum < 2e-3);
}

/// Shadow rays leaving a fibre (2r off its surface, through it when the
/// light is behind) agree with the CPU's occlusion.
#[test]
fn shadow_rays_match_the_cpu() {
    let Some(ctx) = ctx() else { return };
    let s = scene(128, 1);
    let rs = rays(&s);
    let mut sh = Vec::new();
    let mut want = Vec::new();
    let mut rng = Rng::new(3, 0);
    for (o, d) in &rs {
        let o = Point3::new(o[0] as f64, o[1] as f64, o[2] as f64);
        let d = Vec3::new(d[0] as f64, d[1] as f64, d[2] as f64).normalize();
        if let Some(h) = s.bvh.trace_closest_range(&Ray::new(o, d), 1e-7, f64::INFINITY) {
            let q = &s.lights[0];
            let p = q.center + q.u * (2.0 * rng.uniform() - 1.0) + q.v * (2.0 * rng.uniform() - 1.0);
            let dl = (p - h.point).normalize();
            let dist = (p - h.point).norm();
            let r = s.bvh.geometry().segs[h.prim as usize].2;
            let off = *h.normal * (2.0 * r);
            let x = h.point + if dl.dot(&h.normal) > 0.0 { off } else { -off };
            let occ = s.bvh.occluded_range(&Ray::new(x, dl), 1e-7, dist);
            sh.push(([x.x as f32, x.y as f32, x.z as f32], [dl.x as f32, dl.y as f32, dl.z as f32]));
            want.push((occ, dist));
        }
    }
    let g = GpuStrands::new(ctx, s.bvh.geometry(), None, StrandPrim::Cylinders).unwrap();
    let (gpu, _) = hair_gpu::probe(ctx, &g, &sh);
    let mut bad = 0;
    for (k, ((occ, dist), (t, _, _))) in want.iter().zip(&gpu).enumerate() {
        let gocc = *t >= 0.0 && (*t as f64) < *dist;
        if gocc != *occ {
            bad += 1;
            if bad < 6 {
                println!("ray {k}: cpu occluded {occ}, gpu t {t} dist {dist}");
            }
        }
    }
    println!("{bad} / {} shadow rays disagree", want.len());
    assert!(bad * 1000 <= want.len(), "{bad}");
}

