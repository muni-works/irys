//! Phase 5b: the OpenPBR scene kernel (`look::scene_gpu`, `scene.wgsl`).
//!
//! - the device material (`eval_dir`, `lobe_weights`, `lobe_pdf`) against
//!   look::openpbr's f64 `At` on random lanes and random materials;
//! - white furnaces: a closed object whose BSDF conserves energy, under a
//!   world of radiance 1, renders 1 (diffuse, a smooth mirror, smooth glass,
//!   thin glass, a cut-out);
//! - determinism: a frame is bit-identical run to run, and a seed moves it.
//!
//! Skips without a device or without ray queries.
#![cfg(all(feature = "gpu", feature = "assets"))]

use irys_diff::look::openpbr::{At, Kind, OpenPbr};
use irys_diff::look::scene::{LookScene, Material};
use irys_diff::look::scene_gpu::{GpuLook, Opts};
use irys_diff::prb::Rng;

fn gpu() -> Option<&'static irys_gpu::GpuContext> {
    irys_diff::gpu::context().ok()
}

fn dir(rng: &mut Rng) -> [f64; 3] {
    let z = 0.08 + 0.92 * rng.uniform();
    let p = 2.0 * std::f64::consts::PI * rng.uniform();
    let s = (1.0 - z * z).sqrt();
    [s * p.cos(), s * p.sin(), z]
}

fn random_material(rng: &mut Rng, k: usize) -> OpenPbr {
    let mut c = || [rng.uniform(), rng.uniform(), rng.uniform()];
    let base_color = c();
    let specular_color = c();
    let coat_color = c();
    let fuzz_color = c();
    let transmission_color = c();
    let mut u = || rng.uniform();
    OpenPbr {
        base_color,
        base_metalness: if k % 3 == 0 { u() } else { 0.0 },
        specular_weight: 0.5 + u(),
        specular_color,
        specular_roughness: 0.05 + 0.9 * u(),
        specular_ior: 1.2 + u(),
        transmission_weight: if k % 4 == 1 { u() } else { 0.0 },
        transmission_color,
        coat_weight: if k % 2 == 0 { u() } else { 0.0 },
        coat_color,
        coat_roughness: 0.05 + 0.5 * u(),
        coat_darkening: if k % 5 == 0 { 0.0 } else { 1.0 },
        fuzz_weight: if k % 3 == 2 { u() } else { 0.0 },
        fuzz_color,
        fuzz_roughness: 0.2 + 0.7 * u(),
        thin_film_weight: if k % 7 == 3 { u() } else { 0.0 },
        thin_film_thickness: 0.2 + 0.5 * u(),
        transmission_dispersion_scale: if k % 6 == 1 { 1.0 } else { 0.0 },
        ..OpenPbr::default()
    }
}

#[test]
fn device_material_is_the_f64_twin() {
    let Some(ctx) = gpu() else {
        eprintln!("no device: skipped");
        return;
    };
    let mut rng = Rng::new(51, 0);
    let mats: Vec<OpenPbr> = (0..24).map(|k| random_material(&mut rng, k)).collect();
    let scene = LookScene::spheres(mats.iter().enumerate().map(|(k, m)| Material::new(&format!("m{k}"), *m)).collect(), 1.0, (8, 8));
    let Some(g) = GpuLook::new(ctx, &scene) else {
        eprintln!("no ray queries: skipped");
        return;
    };
    let n = 4000;
    let mut lanes = Vec::new();
    let mut twins = Vec::new();
    for k in 0..n {
        let mi = k % mats.len();
        let lam = 380.0 + 400.0 * rng.uniform();
        let inside = mats[mi].transmission_weight > 0.0 && k % 5 == 0;
        let wo = dir(&mut rng);
        let mut wi = dir(&mut rng);
        if mats[mi].transmission_weight > 0.0 && k % 2 == 1 {
            wi[2] = -wi[2];
        }
        let u = [rng.uniform(), rng.uniform(), rng.uniform()];
        let f32v = |a: [f64; 3]| a.map(|x| x as f32);
        lanes.push((mi as u32, lam as f32, inside, f32v(wo), f32v(wi), f32v(u)));
        let at = At::new(&mats[mi], lam, inside);
        let kind = if wi[2] > 0.0 { Kind::Reflect } else { Kind::Transmit };
        let f = at.f(kind, wo, wi, &mats[mi].theta());
        let w = at.lobe_weights(wo[2]);
        let p = at.pdf(&w, wo, wi);
        twins.push((f, p, w));
    }
    let out = g.probe(&lanes);
    let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-3);
    let (mut wf, mut wp, mut ww) = (0.0f64, 0.0f64, 0.0f64);
    let mut bad = 0;
    for (o, (f, p, w)) in out.iter().zip(&twins) {
        let ef = rel(o[0][0] as f64, *f);
        let ep = rel(o[0][1] as f64, *p);
        let gw = [o[0][2], o[0][3], o[1][0], o[1][1], o[1][2]];
        let ew = (0..5).map(|i| (gw[i] as f64 - w[i]).abs()).fold(0.0, f64::max);
        wf = wf.max(ef);
        wp = wp.max(ep);
        ww = ww.max(ew);
        if ef > 2e-3 || ep > 2e-3 || ew > 1e-4 {
            bad += 1;
        }
    }
    eprintln!("device vs f64 over {n} lanes: f {wf:.2e}, pdf {wp:.2e}, lobe weights {ww:.2e} (abs); {bad} lanes over tolerance");
    assert!(bad <= n / 200, "{bad} of {n} lanes disagree");
}

/// Mean radiance over the pixels whose camera ray hits the sphere.
fn furnace(m: Material, spp: u32) -> (f64, f64) {
    let ctx = gpu().unwrap();
    let scene = LookScene::spheres(vec![m], 1.0, (64, 64));
    let g = GpuLook::new(ctx, &scene).unwrap();
    let f = g.render(&scene.camera, &Opts { spp, max_bounces: 64, ..Opts::default() });
    let (w, h) = scene.size;
    let mut s = 0.0;
    let mut s2 = 0.0;
    let mut n = 0.0;
    for y in 0..h {
        for x in 0..w {
            // inside the sphere's disc, away from its silhouette
            let (dx, dy) = ((x as f64 + 0.5) / w as f64 - 0.5, (y as f64 + 0.5) / h as f64 - 0.5);
            if (dx * dx + dy * dy).sqrt() > 0.3 {
                continue;
            }
            let p = &f.rgb[(y * w + x) * 3..(y * w + x) * 3 + 3];
            let l = (p[0] + p[1] + p[2]) as f64 / 3.0;
            s += l;
            s2 += l * l;
            n += 1.0;
        }
    }
    let m = s / n;
    (m, ((s2 / n - m * m) / n).sqrt())
}

#[test]
fn white_furnaces() {
    let Some(ctx) = gpu() else {
        eprintln!("no device: skipped");
        return;
    };
    if GpuLook::new(ctx, &LookScene::spheres(vec![Material::new("x", OpenPbr::default())], 1.0, (4, 4))).is_none() {
        eprintln!("no ray queries: skipped");
        return;
    }
    let white = OpenPbr { base_color: [1.0; 3], specular_weight: 0.0, ..OpenPbr::default() };
    let mirror = OpenPbr { base_color: [1.0; 3], base_metalness: 1.0, specular_roughness: 0.0, ..OpenPbr::default() };
    let glass = OpenPbr { transmission_weight: 1.0, specular_roughness: 0.0, specular_ior: 1.5, ..OpenPbr::default() };
    let dispersive = OpenPbr { transmission_dispersion_scale: 1.0, transmission_dispersion_abbe_number: 20.0, ..glass };
    let mut thin = Material::new("thin", OpenPbr { transmission_weight: 1.0, specular_roughness: 0.0, ..OpenPbr::default() });
    thin.thin_walled = true;
    let mut veil = Material::new("veil", white);
    veil.opacity = 0.5;
    let mut sep = Material::new("sep", OpenPbr { base_color: [1.0; 3], base_metalness: 0.4, specular_weight: 0.0, specular_roughness: 0.0, ..OpenPbr::default() });
    sep.metal_color = Some([1.0; 3]);
    let mut tl = Material::new("translucent", OpenPbr { base_color: [1.0; 3], specular_weight: 0.0, ..OpenPbr::default() });
    tl.translucency = 0.5;
    tl.translucent_color = [1.0; 3];
    let cases = [
        ("white diffuse", Material::new("white", white)),
        ("smooth white metal", Material::new("mirror", mirror)),
        ("smooth glass", Material::new("glass", glass)),
        ("dispersive glass", Material::new("disp", dispersive)),
        ("thin glass", thin),
        ("half-opacity white", veil),
        ("diffuse + separate metal colour", sep),
        ("diffuse + translucency", tl),
    ];
    for (name, m) in cases {
        let (mean, se) = furnace(m, 512);
        eprintln!("furnace {name:32} {mean:.4} ± {se:.4}");
        assert!((mean - 1.0).abs() < 0.01 + 4.0 * se, "{name}: {mean} ± {se}");
    }
}

#[test]
fn frames_are_deterministic() {
    let Some(ctx) = gpu() else {
        eprintln!("no device: skipped");
        return;
    };
    let mut rng = Rng::new(9, 0);
    let mats: Vec<Material> = (0..4).map(|k| Material::new("m", random_material(&mut rng, k))).collect();
    let scene = LookScene::spheres(mats, 0.7, (96, 48));
    let Some(g) = GpuLook::new(ctx, &scene) else {
        eprintln!("no ray queries: skipped");
        return;
    };
    let o = Opts { spp: 8, ..Opts::default() };
    let a = g.render(&scene.camera, &o);
    let b = g.render(&scene.camera, &o);
    let c = g.render(&scene.camera, &Opts { seed: 2, ..o });
    let bits = |f: &[f32]| f.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a.rgb), bits(&b.rgb), "two runs differ");
    assert_eq!(a.spectrum, b.spectrum, "spectra differ");
    assert_ne!(bits(&a.rgb), bits(&c.rgb), "the seed does not reach the frame");
    assert!(a.spectrum.iter().all(|s| s.is_finite() && *s >= 0.0));
}

/// The four-lane material path (no dispersion, no film) against the
/// per-wavelength path it replaces: the same paths (the random stream does
/// not depend on it), the same frame to f32 rounding.
#[test]
fn four_lanes_are_the_per_lane_path() {
    let Some(ctx) = gpu() else {
        eprintln!("no device: skipped");
        return;
    };
    let mut rng = Rng::new(77, 0);
    let mats: Vec<Material> = (0..6)
        .map(|k| {
            let mut m = random_material(&mut rng, k);
            m.thin_film_weight = 0.0;
            m.transmission_dispersion_scale = 0.0;
            let mut mm = Material::new("m", m);
            if k == 3 {
                mm.translucency = 0.4;
                mm.translucent_color = [0.3, 0.8, 0.2];
            }
            if k == 4 {
                mm.metal_color = Some([0.9, 0.5, 0.2]);
            }
            mm
        })
        .collect();
    let scene = LookScene::spheres(mats, 0.8, (120, 40));
    let Some(g) = GpuLook::new(ctx, &scene) else {
        eprintln!("no ray queries: skipped");
        return;
    };
    let o = Opts { spp: 16, max_bounces: 8, ..Opts::default() };
    let a = g.render(&scene.camera, &o);
    let b = g.render(&scene.camera, &Opts { per_lane: true, ..o });
    let (mut d, mut m) = (0.0f64, 0.0f64);
    let mut far = 0;
    for (x, y) in a.rgb.iter().zip(&b.rgb) {
        let e = (x - y).abs() as f64;
        d += e;
        m += y.abs() as f64;
        if e > 1e-3 * (1.0 + y.abs() as f64) {
            far += 1;
        }
    }
    eprintln!("four lanes vs per lane: mean |Δ| / mean {:.2e}, {far} of {} values > 1e-3", d / m, a.rgb.len());
    assert!(d / m < 1e-4, "four-lane path drifts: {}", d / m);
    assert!(far <= a.rgb.len() / 1000);
}

/// Random spheres under a coloured gradient world and a sun, with small
/// emissive spheres (lights the camera sees), many emitter triangles: the
/// scene the samplers, the light tree and guiding are checked on.
fn lit_scene(size: (usize, usize)) -> LookScene {
    use irys_diff::look::scene::{Equirect, Sun, mask};
    let mut rng = Rng::new(31, 0);
    let mut mats: Vec<Material> = (0..4)
        .map(|k| {
            let mut m = random_material(&mut rng, k);
            m.transmission_dispersion_scale = 0.0;
            Material::new("m", m)
        })
        .collect();
    let mut lamp = Material::new("lamp", OpenPbr::default());
    lamp.emission = [6.0, 4.0, 2.0];
    let mut lamp2 = Material::new("lamp2", OpenPbr::default());
    lamp2.emission = [0.5, 1.5, 4.0];
    mats.push(lamp);
    mats.push(lamp2);
    let mut s = LookScene::spheres(mats, 1.0, size);
    // the lamps: small, above the row, lights the camera sees
    for (k, (x, z, r)) in [(4usize, (-1.5f32, 1.6f32, 0.35f32)), (5, (2.0, 1.4, 0.25))] {
        s.instances[k].xf = [[r, 0.0, 0.0, x], [0.0, r, 0.0, -0.8], [0.0, 0.0, r, z]];
        s.instances[k].mask = mask::LIGHT;
    }
    let (w, h) = (32usize, 16usize);
    let rgb = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32 / w as f32, (i / w) as f32 / h as f32);
            [0.2 + 0.6 * (1.0 - y), 0.3 + 0.3 * x, 0.6 * (1.0 - y) + 0.1]
        })
        .collect();
    s.env = Some(Equirect { size: (w as u32, h as u32), rgb });
    let d = [0.4f64, -0.6, 0.7];
    let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    s.sun = Some(Sun { dir: d.map(|x| x / n), radius: 0.02, radiance: [2000.0, 1900.0, 1700.0] });
    s
}

/// A frame's mean and its standard error from `seeds` independent frames.
fn frame_mean(g: &GpuLook, s: &LookScene, o: Opts, seeds: u64) -> (f64, f64) {
    let ms: Vec<f64> = (0..seeds)
        .map(|k| {
            let f = g.render(&s.camera, &Opts { seed: 1000 + 17 * k, ..o });
            f.rgb.iter().map(|&x| x as f64).sum::<f64>() / f.rgb.len() as f64
        })
        .collect();
    let m = ms.iter().sum::<f64>() / seeds as f64;
    let v = ms.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (seeds - 1) as f64;
    (m, (v / seeds as f64).sqrt())
}

/// Speed round 4: Sobol, film-response wavelengths and the sun share by
/// peak keep the frame's mean (hash + uniform + share by irradiance).
#[test]
fn samplers_are_unbiased() {
    use irys_diff::look::scene_gpu::samp;
    let Some(ctx) = gpu() else {
        eprintln!("no device: skipped");
        return;
    };
    let s = lit_scene((96, 48));
    let Some(g) = GpuLook::new(ctx, &s) else {
        eprintln!("no ray queries: skipped");
        return;
    };
    let o = Opts { spp: 64, max_bounces: 6, ..Opts::default() };
    let (m0, e0) = frame_mean(&g, &s, Opts { samp: 0, ..o }, 24);
    for (name, bits) in [("sobol", samp::SOBOL), ("spectral", samp::SPECTRAL), ("sun share", samp::SUN_PEAK), ("all", samp::ALL)] {
        let (m, e) = frame_mean(&g, &s, Opts { samp: bits, ..o }, 24);
        let z = (m - m0) / (e * e + e0 * e0).sqrt();
        eprintln!("{name:10} {m:.5} ± {e:.5} vs {m0:.5} ± {e0:.5}: z {z:+.2}");
        assert!(z.abs() < 4.0, "{name}: z {z}");
    }
}
