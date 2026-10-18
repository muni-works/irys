//! Phase 3's equal-time benchmark: the same scenes in irys and in Blender's
//! Cycles, a high-spp reference from each, and relMSE against time.
//!
//! The two renderers do not share a material model exactly (irys is
//! spectral, hero wavelengths ×4, a Schlick GGX conductor whose F0 is three
//! spectral hats; Cycles is RGB, Principled metallic), so each renderer is
//! scored against **its own** reference: relMSE measures how far a renderer
//! is from converging, which is the thing equal time compares. The scenes
//! are matched as far as the models allow (`export` writes everything Cycles
//! needs, and `scripts/cycles_bench.py` builds the same geometry, camera,
//! materials and sky), and the report checks the two references agree to a
//! few percent, so neither renderer is scored on an easier picture.

pub mod font;
pub mod scenes;

use std::{fs, io::Write, path::Path};

use irys_diff::ggx::{PARAMS, band_weights};
use irys_diff::prb::{CHECKER_DARK, Scene};
use irys_light::spectrum::{LAMBDA_MAX_NM, LAMBDA_MIN_NM, hero_weight};
use serde_json::{Value, json};

/// The relMSE denominator floor: `(x − r)² / (r² + EPS)`, the convention of
/// the denoising literature (Rousselle+ 2011; Müller+ 2017 use 1e-2).
pub const REL_EPS: f64 = 1e-2;

/// Mean over pixels and channels of `(x − r)² / (r² + ε)`.
pub fn relmse(x: &[f32], r: &[f32]) -> f64 {
    assert_eq!(x.len(), r.len());
    let s: f64 = x
        .iter()
        .zip(r)
        .map(|(&a, &b)| {
            let (a, b) = (a as f64, b as f64);
            (a - b) * (a - b) / (b * b + REL_EPS)
        })
        .sum();
    s / x.len() as f64
}

/// Mean of each channel, for the cross-renderer sanity check.
pub fn mean_rgb(x: &[f32]) -> [f64; 3] {
    let mut m = [0.0; 3];
    for (i, &v) in x.iter().enumerate() {
        m[i % 3] += v as f64;
    }
    let n = (x.len() / 3) as f64;
    m.map(|v| v / n)
}

/// Average of `f(λ) · hero_weight(λ)` over the band: the linear-sRGB colour
/// a spectrum `f` renders as in irys (a flat spectrum of 1 is white).
pub fn spectrum_rgb(f: impl Fn(f64) -> f64) -> [f64; 3] {
    let n = 4000;
    let mut c = [0.0; 3];
    for i in 0..n {
        let l = LAMBDA_MIN_NM + (LAMBDA_MAX_NM - LAMBDA_MIN_NM) * (i as f64 + 0.5) / n as f64;
        let w = hero_weight(l);
        for k in 0..3 {
            c[k] += f(l) * w[k] as f64;
        }
    }
    c.map(|v| v / n as f64)
}

/// The conductor's `F0` as linear sRGB, for Cycles' metallic base colour.
/// Schlick's Fresnel is linear in `F0`, so this is exact at every angle up
/// to the RGB projection.
pub fn f0_rgb(theta: &[f64; PARAMS]) -> [f64; 3] {
    spectrum_rgb(|l| {
        let b = band_weights(l);
        b[0] * theta[1] + b[1] * theta[2] + b[2] * theta[3]
    })
}

/// Everything Cycles needs to build `scene`: world-space triangles per
/// placement with its material, the pinhole, the sky and sun, the depth.
pub fn export(scene: &Scene, theta: &[f64; PARAMS], name: &str) -> Value {
    let mut objects = Vec::new();
    for (k, p) in scene.placements().iter().enumerate() {
        let o = scene.blas()[p.blas].geometry();
        let mesh = o.mesh();
        assert!(
            o.analytic().prims().is_empty(),
            "export: analytic prims are not exported"
        );
        let verts: Vec<f64> = mesh
            .positions()
            .iter()
            .flat_map(|v| {
                let w = p.to_world.apply_point(v);
                [w.x, w.y, w.z]
            })
            .collect();
        let tris: Vec<u32> = mesh.triangles().iter().flatten().copied().collect();
        let surf = o.surface(0);
        let material = match surf {
            irys_diff::prb::Surface::Conductor => json!({
                "kind": "conductor",
                "f0": f0_rgb(theta),
                "roughness": theta[0].sqrt(),
            }),
            irys_diff::prb::Surface::Diffuse => {
                let checker = if p.checker { scene.floor_checker } else { None };
                json!({
                    "kind": "diffuse",
                    "albedo": scene.floor_albedo * p.albedo,
                    "checker": checker,
                    "checker_dark": CHECKER_DARK,
                })
            }
            irys_diff::prb::Surface::Emitter => {
                let e = scene.emission;
                json!({"kind": "emitter", "radiance": [e, e, e]})
            }
        };
        objects.push(json!({
            "name": format!("{name}_{k}"),
            "verts": verts,
            "tris": tris,
            "material": material,
        }));
    }
    let (eye, fwd, right, up, tan_half) = scene.camera();
    let e = scene.env;
    let tint = spectrum_rgb(|l| 1.0 + 0.4 * (550.0 - l) / 200.0);
    json!({
        "name": name,
        "size": [scene.size.0, scene.size.1],
        "depth": scene.depth,
        "camera": {
            "eye": [eye.x, eye.y, eye.z],
            "forward": [fwd.x, fwd.y, fwd.z],
            "right": [right.x, right.y, right.z],
            "up": [up.x, up.y, up.z],
            "tan_half": tan_half,
        },
        "env": {
            "sun": [e.sun.x, e.sun.y, e.sun.z],
            "kappa": e.kappa,
            "sun_radiance": e.sun_radiance,
            "sky": e.sky,
            "sky_tint_rgb": tint,
        },
        "objects": objects,
    })
}

/// [`export`] for a scene too large to flatten (phase 5a's imports): each
/// bottom-level tree's triangles once, in its own frame, to
/// `<dir>/meshes.f32` (f32 xyz per corner), and per placement its tree, its
/// object → world matrix (row-major) and its material, so Cycles instances
/// them as irys does. Everything else as [`export`].
pub fn export_instanced(scene: &Scene, theta: &[f64; PARAMS], name: &str, dir: &Path) -> Value {
    let mut bin: Vec<u8> = Vec::new();
    let mut meshes = Vec::new();
    for t in scene.blas() {
        let o = t.geometry();
        let m = o.mesh();
        let start = bin.len() / 12;
        for tri in m.triangles() {
            for &k in tri {
                let p = m.positions()[k as usize];
                for c in [p.x, p.y, p.z] {
                    bin.extend((c as f32).to_le_bytes());
                }
            }
        }
        meshes.push(json!({"offset": start, "corners": bin.len() / 12 - start}));
    }
    fs::write(dir.join("meshes.f32"), &bin).unwrap();
    let mut objects = Vec::new();
    for p in scene.placements() {
        let o = scene.blas()[p.blas].geometry();
        let material = material_of(scene, theta, p, o.surface(0));
        let t = &p.to_world;
        let z = t.apply_point(&irys_core::Point3::new(0.0, 0.0, 0.0));
        let cols: Vec<irys_core::Vec3> = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            .iter()
            .map(|e| t.apply_point(&irys_core::Point3::new(e[0], e[1], e[2])) - z)
            .collect();
        let m = [
            cols[0].x, cols[1].x, cols[2].x, z.x, cols[0].y, cols[1].y, cols[2].y, z.y, cols[0].z,
            cols[1].z, cols[2].z, z.z,
        ];
        objects.push(json!({"mesh": p.blas, "matrix": m, "material": material}));
    }
    let mut v = export_head(scene, name);
    // The sun as a Cycles sun lamp of the same irradiance and solid angle
    // (irys's vMF lobe has 2π/κ sr): Cycles samples a lamp exactly, but a
    // lobe a few texels wide in its world map poorly, and the benchmark is
    // of the renderers, not of that map.
    let e = scene.env;
    if e.sun_radiance > 0.0 {
        v["sun_lamp"] = json!({
            "irradiance": e.sun_radiance * 2.0 * std::f64::consts::PI / e.kappa,
            "radius": (2.0 / e.kappa).sqrt(),
        });
    }
    v["meshes"] = json!(meshes);
    v["objects"] = json!(objects);
    v
}

fn material_of(
    scene: &Scene,
    theta: &[f64; PARAMS],
    p: &irys_diff::prb::Placement,
    surf: irys_diff::prb::Surface,
) -> Value {
    match surf {
        irys_diff::prb::Surface::Conductor => json!({
            "kind": "conductor",
            "f0": f0_rgb(theta),
            "roughness": theta[0].sqrt(),
        }),
        irys_diff::prb::Surface::Diffuse => {
            let checker = if p.checker { scene.floor_checker } else { None };
            json!({
                "kind": "diffuse",
                "albedo": scene.floor_albedo * p.albedo,
                "checker": checker,
                "checker_dark": CHECKER_DARK,
            })
        }
        irys_diff::prb::Surface::Emitter => {
            let e = scene.emission;
            json!({"kind": "emitter", "radiance": [e, e, e]})
        }
    }
}

fn export_head(scene: &Scene, name: &str) -> Value {
    let (eye, fwd, right, up, tan_half) = scene.camera();
    let e = scene.env;
    let tint = spectrum_rgb(|l| 1.0 + 0.4 * (550.0 - l) / 200.0);
    json!({
        "name": name,
        "size": [scene.size.0, scene.size.1],
        "depth": scene.depth,
        "camera": {
            "eye": [eye.x, eye.y, eye.z],
            "forward": [fwd.x, fwd.y, fwd.z],
            "right": [right.x, right.y, right.z],
            "up": [up.x, up.y, up.z],
            "tan_half": tan_half,
        },
        "env": {
            "sun": [e.sun.x, e.sun.y, e.sun.z],
            "kappa": e.kappa,
            "sun_radiance": e.sun_radiance,
            "sky": e.sky,
            "sky_tint_rgb": tint,
        },
    })
}

/// Raw little-endian f32, `w·h·3`, rows top to bottom.
pub fn write_f32(path: &Path, img: &[f32]) {
    let bytes: Vec<u8> = img.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(path, bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
}

/// [`write_f32`]'s inverse.
pub fn read_f32(path: &Path) -> Vec<f32> {
    let b = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Linear sRGB to 8-bit sRGB, scaled by `exposure`.
pub fn to_srgb8(img: &[f32], exposure: f64) -> Vec<u8> {
    img.iter()
        .map(|&v| {
            let x = (v as f64 * exposure).clamp(0.0, 1.0);
            let s = if x <= 0.003_130_8 {
                12.92 * x
            } else {
                1.055 * x.powf(1.0 / 2.4) - 0.055
            };
            (s * 255.0 + 0.5) as u8
        })
        .collect()
}

/// An 8-bit RGB PNG.
pub fn write_png(path: &Path, w: usize, h: usize, rgb: &[u8]) {
    let f = fs::File::create(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut wr = enc.write_header().expect("png header");
    wr.write_image_data(rgb).expect("png data");
}

/// Write `v` as pretty JSON.
pub fn write_json(path: &Path, v: &Value) {
    let mut f = fs::File::create(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    writeln!(f, "{}", serde_json::to_string_pretty(v).unwrap()).unwrap();
}

/// Read JSON.
pub fn read_json(path: &Path) -> Value {
    let s = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&s).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// `log e` against `log t`, linearly interpolated (and extrapolated from the
/// end segments) at `t`: the error a renderer reaches in time `t`.
pub fn error_at(curve: &[(f64, f64)], t: f64) -> f64 {
    assert!(curve.len() >= 2, "a curve needs two points");
    let mut c: Vec<(f64, f64)> = curve.iter().map(|&(t, e)| (t.ln(), e.ln())).collect();
    c.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let x = t.ln();
    let i = c
        .windows(2)
        .position(|w| x <= w[1].0)
        .unwrap_or(c.len() - 2);
    let (a, b) = (c[i], c[i + 1]);
    let s = (x - a.0) / (b.0 - a.0).max(1e-12);
    (a.1 + s * (b.1 - a.1)).exp()
}
