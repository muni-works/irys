//! M1's splat gate: the garage splat rendered by irys at the map's own
//! held-out training cameras (ipse `splat_train.py`'s, rebuilt from
//! `perim.derived`), against a CPU 3DGS rasteriser written for the gate
//! (EWA splatting as gsplat renders: projection, 0.3 px dilation, depth
//! sort, tiles), plus a synthetic cloud where the two models should agree.
//! The real frames wait on M0's registration (docs/design.md, "M1").
//!
//! ```text
//! cargo run --release -p irys-diff --features gpu,assets --example m1_garage -- [views] [cloud...]
//! SYN=300 …   # synthetic cloud size
//! ```
//!
//! Reads (never writes) `~/Developer/ipse/maps/garage-perim/*.ply` and
//! `~/Downloads/ipse-room/perim.derived`; writes PNGs to `out/m1/`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use irys_diff::look::scene::LookScene;
use irys_diff::look::scene_gpu::{GpuLook, Opts};
use irys_diff::look::pinhole::Pinhole;
use irys_diff::look::sensor::{Sensor, aov};
use irys_diff::look::splat_gpu::{self, PlyCloud};
use rayon::prelude::*;

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap())
}

/// The trainer's renderer on the CPU: project, EWA covariance (+0.3 px),
/// depth sort, front-to-back blending per pixel; sRGB-valued output.
fn rasterise(c: &PlyCloud, pin: &Pinhole) -> Vec<[f32; 3]> {
    let (w, h) = (pin.width, pin.height);
    const TILE: usize = 16;
    let (tw, th) = (w.div_ceil(TILE), h.div_ceil(TILE));
    struct G {
        z: f64,
        m: [f64; 2],
        con: [f64; 3],
        op: f64,
        col: [f32; 3],
    }
    let gs: Vec<Option<(G, [usize; 4])>> = (0..c.len())
        .into_par_iter()
        .map(|i| {
            let mu = c.positions[i].map(|x| x as f64);
            let d = [mu[0] - pin.pos[0], mu[1] - pin.pos[1], mu[2] - pin.pos[2]];
            let pc: [f64; 3] = std::array::from_fn(|k| (0..3).map(|a| pin.rot[a][k] * d[a]).sum());
            if pc[2] < 0.01 {
                return None;
            }
            // Σ_world = Mᵀ⁻¹… from the whitening: Σ = (MᵀM)⁻¹ = R S² Rᵀ
            let (m, _) = splat_gpu::whiten(c, i);
            // R S: columns k = R[:, k] s_k = m[k] * s_k²
            let s = c.scales[i].map(|x| x as f64);
            let rs: [[f64; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|k| m[k][a] * s[k] * s[k]));
            let sw: [[f64; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|b| (0..3).map(|k| rs[a][k] * rs[b][k]).sum()));
            // camera frame: Rᵀ Σ R
            let r = &pin.rot;
            let sc: [[f64; 3]; 3] = std::array::from_fn(|p| {
                std::array::from_fn(|q| {
                    (0..3).map(|a| (0..3).map(|b| r[a][p] * sw[a][b] * r[b][q]).sum::<f64>()).sum()
                })
            });
            let (x, y, z) = (pc[0], pc[1], pc[2]);
            // the trainer clamps the Jacobian's x/z, y/z to 1.3 × the half-FOV
            let lx = 1.3 * (w as f64 / (2.0 * pin.fx));
            let ly = 1.3 * (h as f64 / (2.0 * pin.fy));
            let (xc, yc) = ((x / z).clamp(-lx, lx) * z, (y / z).clamp(-ly, ly) * z);
            let j = [[pin.fx / z, 0.0, -pin.fx * xc / (z * z)], [0.0, pin.fy / z, -pin.fy * yc / (z * z)]];
            let js: [[f64; 3]; 2] = std::array::from_fn(|a| std::array::from_fn(|q| (0..3).map(|p| j[a][p] * sc[p][q]).sum()));
            let cov = |a: usize, b: usize| (0..3).map(|q| js[a][q] * j[b][q]).sum::<f64>();
            let (a, b, cc) = (cov(0, 0) + 0.3, cov(0, 1), cov(1, 1) + 0.3);
            let det = a * cc - b * b;
            if det <= 0.0 {
                return None;
            }
            let con = [cc / det, -b / det, a / det];
            let mid = 0.5 * (a + cc);
            let l1 = mid + (mid * mid - det).max(0.1).sqrt();
            let rad = (3.0 * l1.sqrt()).ceil();
            let m2 = [pin.fx * x / z + pin.cx, pin.fy * y / z + pin.cy];
            let lo = [((m2[0] - rad) / TILE as f64).floor().max(0.0), ((m2[1] - rad) / TILE as f64).floor().max(0.0)];
            let hi = [((m2[0] + rad) / TILE as f64).floor().min(tw as f64 - 1.0), ((m2[1] + rad) / TILE as f64).floor().min(th as f64 - 1.0)];
            if hi[0] < lo[0] || hi[1] < lo[1] {
                return None;
            }
            let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let col = splat_gpu::colour_cpu(c, i, d.map(|v| v / n)).map(|v| v as f32);
            Some((G { z, m: m2, con, op: c.opacities[i] as f64, col }, [lo[0] as usize, lo[1] as usize, hi[0] as usize, hi[1] as usize]))
        })
        .collect();
    let mut order: Vec<usize> = (0..gs.len()).filter(|&i| gs[i].is_some()).collect();
    order.sort_by(|&a, &b| gs[a].as_ref().unwrap().0.z.total_cmp(&gs[b].as_ref().unwrap().0.z).then(a.cmp(&b)));
    let mut tiles: Vec<Vec<usize>> = vec![Vec::new(); tw * th];
    for &i in &order {
        let (_, bb) = gs[i].as_ref().unwrap();
        for ty in bb[1]..=bb[3] {
            for tx in bb[0]..=bb[2] {
                tiles[ty * tw + tx].push(i);
            }
        }
    }
    let mut img = vec![[0f32; 3]; w * h];
    let rows: Vec<Vec<(usize, [f32; 3])>> = (0..tw * th)
        .into_par_iter()
        .map(|t| {
            let (tx, ty) = (t % tw, t / tw);
            let mut out = Vec::new();
            for py in ty * TILE..((ty + 1) * TILE).min(h) {
                for px in tx * TILE..((tx + 1) * TILE).min(w) {
                    let (mut tr, mut col) = (1.0f64, [0.0f64; 3]);
                    for &i in &tiles[t] {
                        let g = &gs[i].as_ref().unwrap().0;
                        let dx = [g.m[0] - px as f64, g.m[1] - py as f64];
                        let pw = -0.5 * (g.con[0] * dx[0] * dx[0] + g.con[2] * dx[1] * dx[1]) - g.con[1] * dx[0] * dx[1];
                        if pw > 0.0 {
                            continue;
                        }
                        let a = (g.op * pw.exp()).min(0.99);
                        if a < 1.0 / 255.0 {
                            continue;
                        }
                        let nt = tr * (1.0 - a);
                        if nt < 1e-4 {
                            break;
                        }
                        for k in 0..3 {
                            col[k] += g.col[k] as f64 * a * tr;
                        }
                        tr = nt;
                    }
                    out.push((py * w + px, col.map(|v| v as f32)));
                }
            }
            out
        })
        .collect();
    for r in rows {
        for (k, c) in r {
            img[k] = c;
        }
    }
    img
}

fn psnr(a: &[[f32; 3]], b: &[[f32; 3]]) -> f64 {
    let mse: f64 = a.iter().zip(b).map(|(x, y)| (0..3).map(|k| ((x[k].clamp(0.0, 1.0) - y[k].clamp(0.0, 1.0)) as f64).powi(2)).sum::<f64>()).sum::<f64>()
        / (3 * a.len()) as f64;
    -10.0 * mse.max(1e-12).log10()
}

fn save(path: &Path, img: &[[f32; 3]], w: usize, h: usize) {
    let mut b = image::RgbImage::new(w as u32, h as u32);
    for (k, p) in img.iter().enumerate() {
        b.put_pixel((k % w) as u32, (k / w) as u32, image::Rgb(p.map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)));
    }
    b.save(path).unwrap();
}

/// `.npy` v1, little-endian f32, C order: the data after the header.
fn npy_f32(path: &Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap();
    assert_eq!(&b[..6], b"\x93NUMPY");
    let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
    let head = std::str::from_utf8(&b[10..10 + hl]).unwrap();
    assert!(head.contains("'<f4'") && head.contains("False"), "{head}");
    bytemuck::cast_slice::<u8, f32>(&b[10 + hl..]).to_vec()
}

/// The cameras the splat was trained on (ipse `splat_train.py`): VGGT's
/// world(unit) → camera extrinsics, re-expressed map → camera in metres
/// through roombake's floor frame, and VGGT's intrinsics (518 × 420).
fn training_views(derived: &Path) -> Vec<Pinhole> {
    let ex = npy_f32(&derived.join("extri.npy"));
    let k = npy_f32(&derived.join("intri.npy"));
    let fl = npy_f32(&derived.join("floor.npy")).iter().map(|&x| x as f64).collect::<Vec<_>>();
    let rep: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(derived.join("report.json")).unwrap()).unwrap();
    let scale = rep["scale_m_per_unit"].as_f64().unwrap();
    // floor_frame: rows x, y, n
    let nl = (fl[0] * fl[0] + fl[1] * fl[1] + fl[2] * fl[2]).sqrt();
    let n = [fl[0] / nl, fl[1] / nl, fl[2] / nl];
    let d = fl[3];
    let mut x = [-n[0] * n[2], -n[1] * n[2], 1.0 - n[2] * n[2]];
    let xl = (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]).sqrt();
    x = x.map(|v| v / xl);
    let y = [n[1] * x[2] - n[2] * x[1], n[2] * x[0] - n[0] * x[2], n[0] * x[1] - n[1] * x[0]];
    let rf = [x, y, n];
    (0..ex.len() / 12)
        .map(|i| {
            let e = &ex[12 * i..12 * i + 12];
            let r: [[f64; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|b| e[4 * a + b] as f64));
            let t: [f64; 3] = std::array::from_fn(|a| e[4 * a + 3] as f64);
            // R' = R Rfᵀ, t' = scale t − R' [0, 0, d]
            let rp: [[f64; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|b| (0..3).map(|j| r[a][j] * rf[b][j]).sum()));
            let tp: [f64; 3] = std::array::from_fn(|a| scale * t[a] - rp[a][2] * d);
            // camera → map: rot = R'ᵀ, pos = −R'ᵀ t'
            let rot = std::array::from_fn(|a| std::array::from_fn(|b| rp[b][a]));
            let pos = std::array::from_fn(|a| -(0..3).map(|j| rp[j][a] * tp[j]).sum::<f64>());
            let kk = &k[9 * i..9 * i + 9];
            Pinhole { width: 518, height: 420, fx: kk[0] as f64, fy: kk[4] as f64, cx: kk[2] as f64, cy: kk[5] as f64, rot, pos }
        })
        .collect()
}

/// Small splats far from the camera, where EWA's affine projection is
/// accurate: the two renderers' models should agree there.
fn synthetic() {
    let mut c = PlyCloud { degree: 0, ..Default::default() };
    let mut s = 12345u64;
    let mut u = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 40) as f32 / (1u64 << 24) as f32
    };
    let n_syn: usize = std::env::var("SYN").ok().and_then(|v| v.parse().ok()).unwrap_or(3000);
    for _ in 0..n_syn {
        c.positions.push([2.0 * u() - 1.0, 2.0 * u() - 1.0, 0.5 * u() - 0.25]);
        c.scales.push([0.01 + 0.03 * u(), 0.01 + 0.03 * u(), 0.005 + 0.02 * u()]);
        let q = [u() - 0.5, u() - 0.5, u() - 0.5, u() - 0.5];
        let l = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        c.quats.push(q.map(|x| x / l));
        c.opacities.push(0.1 + 0.85 * u());
        c.sh.push([2.0 * u() - 1.0, 2.0 * u() - 1.0, 2.0 * u() - 1.0]);
    }
    let pin = Pinhole { width: 320, height: 240, fx: 300.0, fy: 300.0, cx: 160.0, cy: 120.0, rot: [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], pos: [0.0, 0.0, 4.0] };
    let ctx = irys_diff::gpu::context().expect("gpu");
    let mut scene = LookScene::spheres(vec![], 0.0, (320, 240));
    scene.env = None;
    let g = GpuLook::with_splats(ctx, &scene, Some(&c)).unwrap();
    let a = Sensor::new(&g, (320, 240), 1).render(&[pin.camera()], aov::RGB);
    let rgb: Vec<[f32; 3]> = a.rgb.unwrap().iter().map(|p| [0, 1, 2].map(|k| splat_gpu::srgb_enc(p[k] as f64) as f32)).collect();
    let cpu = rasterise(&c, &pin);
    println!("synthetic, 3000 small splats at 4 m: PSNR irys vs CPU raster {:.2} dB", psnr(&rgb, &cpu));
    save(Path::new("out/m1/synthetic_irys.png"), &rgb, 320, 240);
    save(Path::new("out/m1/synthetic_cpu.png"), &cpu, 320, 240);
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::fs::create_dir_all("out/m1").unwrap();
    synthetic();
    let views: usize = args.first().and_then(|s| s.parse().ok()).unwrap_or(4);
    let clouds: Vec<String> = if args.len() > 1 { args[1..].to_vec() } else { vec!["splat7k.ply".into(), "splat30k.ply".into(), "splat.ply".into()] };
    let map = home().join("Developer/ipse/maps/garage-perim");
    let derived = home().join("Downloads/ipse-room/perim.derived");
    let out = PathBuf::from("out/m1");
    std::fs::create_dir_all(&out).unwrap();
    let all = training_views(&derived);
    // splat_train.py's holdout: (i / 2) % 8 == 0, left eyes
    let held: Vec<usize> = (0..all.len()).filter(|i| (i / 2) % 8 == 0 && i % 2 == 0).collect();
    let pick: Vec<usize> = (0..views.min(held.len())).map(|k| held[k * held.len() / views.min(held.len())]).collect();
    let (w, h) = (all[0].width, all[0].height);
    let ctx = irys_diff::gpu::context().expect("gpu");
    let mut scene = LookScene::spheres(vec![], 0.0, (w, h));
    scene.env = None;
    println!("| cloud | splats | view | PSNR irys vs CPU raster | sensor ms (splat layer) | scene kernel 1 spp ms |");
    println!("|---|---|---|---|---|---|");
    for name in &clouds {
        let t0 = Instant::now();
        let c = PlyCloud::load(&map.join(name)).unwrap();
        let load_s = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let g = GpuLook::with_splats(ctx, &scene, Some(&c)).unwrap();
        let build_s = t1.elapsed().as_secs_f64();
        eprintln!("{name}: {} splats, degree {}, load {:.2} s, upload + BLAS {:.2} s", c.len(), c.degree, load_s, build_s);
        let mut sen = Sensor::new(&g, (w, h), 1);
        for &k in &pick {
            let pin = all[k];
            let cam = pin.camera();
            sen.render(&[cam], aov::RGB);
            let mut ts = Vec::new();
            let mut a = None;
            for _ in 0..5 {
                let t = Instant::now();
                a = Some(sen.render(&[cam], aov::RGB));
                ts.push(t.elapsed().as_secs_f64());
            }
            ts.sort_by(f64::total_cmp);
            let rgb: Vec<[f32; 3]> = a.unwrap().rgb.unwrap().iter().map(|p| [0, 1, 2].map(|k| splat_gpu::srgb_enc(p[k] as f64) as f32)).collect();
            let o = Opts { spp: 1, max_bounces: 1, ..Opts::default() };
            g.render(&cam, &o);
            let mut tk = Vec::new();
            for _ in 0..3 {
                let t = Instant::now();
                g.render(&cam, &o);
                tk.push(t.elapsed().as_secs_f64());
            }
            tk.sort_by(f64::total_cmp);
            let cpu = rasterise(&c, &pin);
            let stem = name.trim_end_matches(".ply");
            save(&out.join(format!("{stem}_v{k}_irys.png")), &rgb, w, h);
            save(&out.join(format!("{stem}_v{k}_cpu.png")), &cpu, w, h);
            println!("| {name} | {} | {k} | {:.2} | {:.2} | {:.2} |", c.len(), psnr(&rgb, &cpu), 1e3 * ts[2], 1e3 * tk[1]);
        }
    }
}
