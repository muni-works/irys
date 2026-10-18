//! Phase 5a: the Blender export loader (`irys_diff::blend`).
//!
//! A tiny export written here by hand (the same glTF + sidecar layout
//! `scripts/blend_export.py` writes): a floor quad placed twice, an emissive
//! panel, a sun, an area light and a point light. The loader keeps the
//! instancing, shares one emitter radiance by shrinking the dimmer emitters,
//! turns the sun into the lobe, and the scene it builds renders the same on
//! the CPU and the device, bit-identically twice. The real classroom, when
//! its export is on disk, is an ignored test.
#![cfg(feature = "assets")]

use std::path::{Path, PathBuf};

use irys_diff::blend::{self, Options};
use irys_diff::prb::{self, Kernels, Surface};
use serde_json::json;

/// Write a glb of `meshes` (each: positions of independent triangles, one
/// material index) and `nodes` (mesh index, column-major matrix).
fn write_glb(path: &Path, meshes: &[(Vec<[f32; 3]>, usize)], nodes: &[(usize, [f32; 16])], n_mats: usize) {
    let mut bin: Vec<u8> = Vec::new();
    let (mut views, mut accs, mut gm) = (Vec::new(), Vec::new(), Vec::new());
    for (pos, mat) in meshes {
        let o = bin.len();
        for p in pos {
            for c in p {
                bin.extend(c.to_le_bytes());
            }
        }
        views.push(json!({"buffer": 0, "byteOffset": o, "byteLength": pos.len() * 12}));
        let (lo, hi) = pos.iter().fold(([f32::MAX; 3], [f32::MIN; 3]), |(mut lo, mut hi), p| {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
            (lo, hi)
        });
        accs.push(json!({"bufferView": views.len() - 1, "componentType": 5126, "count": pos.len(), "type": "VEC3", "min": lo, "max": hi}));
        let o = bin.len();
        for i in 0..pos.len() as u32 {
            bin.extend(i.to_le_bytes());
        }
        views.push(json!({"buffer": 0, "byteOffset": o, "byteLength": pos.len() * 4}));
        accs.push(json!({"bufferView": views.len() - 1, "componentType": 5125, "count": pos.len(), "type": "SCALAR"}));
        gm.push(json!({"primitives": [{"attributes": {"POSITION": accs.len() - 2}, "indices": accs.len() - 1, "material": mat}]}));
    }
    let nodes: Vec<_> = nodes.iter().map(|(m, x)| json!({"mesh": m, "matrix": x})).collect();
    let mats: Vec<_> = (0..n_mats).map(|i| json!({"name": format!("m{i}")})).collect();
    let doc = json!({
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": (0..nodes.len()).collect::<Vec<_>>()}],
        "nodes": nodes, "meshes": gm, "materials": mats, "accessors": accs, "bufferViews": views,
        "buffers": [{"byteLength": bin.len()}],
    });
    let mut js = serde_json::to_vec(&doc).unwrap();
    while js.len() % 4 != 0 {
        js.push(b' ');
    }
    while bin.len() % 4 != 0 {
        bin.push(0);
    }
    let mut out = Vec::new();
    out.extend(0x4654_6C67u32.to_le_bytes());
    out.extend(2u32.to_le_bytes());
    out.extend(((12 + 8 + js.len() + 8 + bin.len()) as u32).to_le_bytes());
    out.extend((js.len() as u32).to_le_bytes());
    out.extend(0x4E4F_534Au32.to_le_bytes());
    out.extend(&js);
    out.extend((bin.len() as u32).to_le_bytes());
    out.extend(0x004E_4942u32.to_le_bytes());
    out.extend(&bin);
    std::fs::write(path, out).unwrap();
}

/// A quad in Blender's frame (z up), `[-1, 1]²` at height `z`, facing `+z`.
fn quad(z: f32) -> Vec<[f32; 3]> {
    vec![[-1., -1., z], [1., -1., z], [1., 1., z], [-1., -1., z], [1., 1., z], [-1., 1., z]]
}

/// glTF's node matrix for a Blender world placement: C · W, C = (x, z, −y).
fn node(tx: f32, ty: f32, tz: f32, s: f32) -> [f32; 16] {
    // W = translate · scale; C·W column-major.
    [s, 0., 0., 0., 0., 0., -s, 0., 0., s, 0., 0., tx, tz, -ty, 1.]
}

fn tiny() -> PathBuf {
    // One directory per call: tests run in parallel in one process, and two
    // writing the same files raced (one read a half-written glb).
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("irys-phase5a-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // Mesh 0: the floor (diffuse 0.6), placed twice; mesh 1: a ceiling
    // panel facing down (emitter, radiance 4), placed once; mesh 2: glass,
    // left out.
    let mut panel = quad(0.0);
    for t in panel.chunks_mut(3) {
        t.swap(1, 2);
    }
    write_glb(
        &dir.join("scene.glb"),
        &[(quad(0.0), 0), (panel, 1), (quad(0.5), 2)],
        &[(0, node(0., 0., 0., 2.)), (0, node(5., 0., 0., 1.)), (1, node(0., 0., 2.5, 0.25)), (2, node(0., 0., 0., 1.))],
        3,
    );
    let side = json!({
        "camera": {"eye": [0.0, -6.0, 2.0], "forward": [0.0, 0.94868, -0.31623], "right": [1.0, 0.0, 0.0],
                   "up": [0.0, 0.31623, 0.94868], "tan_half": 0.5, "size": [64, 48]},
        "materials": [
            {"name": "floor", "kind": "diffuse", "albedo": 0.6, "rgb": [0.6, 0.6, 0.6]},
            {"name": "panel", "kind": "emitter", "radiance": 4.0, "side": "front", "rgb": [4.0, 4.0, 4.0]},
            {"name": "glass", "kind": "hidden", "rgb": [1.0, 1.0, 1.0]},
        ],
        "lights": [
            {"name": "sun", "kind": "sun", "dir": [0.3, -0.2, 0.93], "irradiance": 2.0, "radius": 0.05},
            {"name": "area", "kind": "quad", "center": [2.0, 0.0, 2.0], "u": [0.5, 0.0, 0.0], "v": [0.0, 0.5, 0.0],
             "normal": [0.0, 0.0, -1.0], "radiance": 16.0},
            {"name": "bulb", "kind": "sphere", "center": [-2.0, 1.0, 1.0], "radius": 0.1, "radiance": 1.0},
        ],
        "env": {"sky": 0.5},
        "dropped": {"materials": {"texture averaged": 1}, "lights": [], "camera": []},
    });
    std::fs::write(dir.join("irys.json"), serde_json::to_string(&side).unwrap()).unwrap();
    dir
}

#[test]
fn the_loader_keeps_instancing_and_shares_one_emission() {
    let dir = tiny();
    let im = blend::load(&dir, &Options { width: Some(32), spp: 2, depth: 4, grey: None }).unwrap();
    let s = &im.scene;
    let r = &im.report;
    // floor ×2, panel, lights; glass left out
    assert_eq!(r.trees, 3, "{r:?}");
    assert_eq!(r.placements, 4);
    assert_eq!(s.size, (32, 24));
    assert_eq!(s.emission, 16.0);
    // panel radiance 4 of 16: shrunk by ½; the bulb by ¼
    assert!((r.min_shrink - 0.25).abs() < 1e-12, "{}", r.min_shrink);
    let a = s.lights().area;
    let want = 0.25 * 0.25 + 1.0 // panel: (0.5 m)², then × ¼; the area light: 1 m²
        + 4.0 * std::f64::consts::PI * 0.025f64.powi(2); // bulb, r × ¼ (a 16 × 8 sphere: a little less)
    assert!((a - want).abs() / want < 0.03, "emitter area {a} vs {want}");
    // the floor's placements are diffuse 0.6, at the right places
    let floors: Vec<_> = s.placements().iter().filter(|p| p.blas == 0).collect();
    assert_eq!(floors.len(), 2);
    assert!(floors.iter().all(|p| p.albedo == 0.6));
    let c = floors[1].to_world.apply_point(&irys_core::Point3::new(1.0, 1.0, 0.0));
    assert!((c.x - 6.0).abs() < 1e-6 && (c.y - 1.0).abs() < 1e-6 && c.z.abs() < 1e-6, "{c:?}");
    // emitter panels face down, at z = 2.5
    let lt = &s.lights().tris[0];
    let n = (lt[1] - lt[0]).cross(&(lt[2] - lt[0]));
    assert!(n.z < 0.0 && (lt[0].z - 2.5).abs() < 1e-6, "{lt:?}");
    // the sun: irradiance 2 from the lobe
    let e = s.env;
    let irr = e.sun_radiance * 2.0 * std::f64::consts::PI / e.kappa;
    assert!((irr - 2.0).abs() < 1e-9 && (e.kappa - 800.0).abs() < 1e-6, "{e:?}");
    assert_eq!(e.sky, 0.5);
    assert!(e.sun_pick() > 0.99);
    // grey: every non-emitter diffuse 0.8
    let g = blend::load(&dir, &Options { width: Some(32), spp: 2, depth: 4, grey: Some(0.8) }).unwrap();
    assert!(g.scene.placements().iter().filter(|p| p.blas == 0).all(|p| p.albedo == 0.8));
    assert_eq!(s.blas()[1].geometry().surface(0), Surface::Emitter);
}

#[cfg(feature = "gpu")]
mod gpu_tier {
    use super::*;
    use irys_diff::gpu::{self, GpuPrb};

    fn mean(v: &[f32]) -> f64 {
        v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64
    }

    /// An imported scene renders the same on the CPU and the device, and
    /// the device twice bit for bit.
    fn agree(im: &blend::Imported, tol: f64) {
        let ctx = match gpu::context() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("no GPU ({e}): skipping");
                return;
            }
        };
        let s = &im.scene;
        let mut k = Kernels::for_scene(s, false);
        let cpu = prb::image(s, 9, &prb::forward(s, &im.theta, 9, &mut k).0);
        let mut g = GpuPrb::new(ctx, s);
        let a = g.image(s, &im.theta, 9);
        let b = g.image(s, &im.theta, 9);
        assert!(a == b, "device rerun differs");
        let mc = cpu.iter().sum::<f64>() / cpu.len() as f64;
        let rel = (mc - mean(&a)).abs() / mc.abs().max(1e-12);
        eprintln!("cpu {mc:.6} gpu {:.6} rel {rel:.2e}", mean(&a));
        assert!(rel < tol, "cpu {mc} gpu {}", mean(&a));
    }

    #[test]
    fn an_imported_scene_renders_the_same_on_cpu_and_device() {
        let dir = tiny();
        let im = blend::load(&dir, &Options { width: Some(48), spp: 4, depth: 4, grey: None }).unwrap();
        agree(&im, 1e-3);
    }

    /// The classroom export, if `IRYS_SCENES` (default
    /// `../../../irys-assets/blender-demos/export` from this crate) holds
    /// it. Measured (`--ignored`): it is not in the repo.
    #[test]
    #[ignore]
    fn the_classroom_renders_the_same_on_cpu_and_device() {
        let root = std::env::var("IRYS_SCENES")
            .map(PathBuf::from)
            .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../irys-assets/blender-demos/export"));
        let im = blend::load(&root.join("classroom"), &Options { width: Some(96), spp: 2, depth: 4, grey: None }).unwrap();
        agree(&im, 1e-3);
    }
}
