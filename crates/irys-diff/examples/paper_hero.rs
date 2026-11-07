//! The irys paper's hero: rose full, read from its own CAD, standing in the
//! Barcelona Pavilion ("Barcelona Pavillion" by eMirage, CC-BY, from
//! blender.org's demo files), rendered by the OpenPBR scene kernel
//! (`look::scene_gpu`). Writes the frame and the annotation the paper draws
//! over it, all projected through the same camera: the arm's joint axes and
//! angles, the trees' names, and the shoulder plate's bearing bore (the
//! shared BRep benchmark's part, see `rose_brep`).
//!
//! ```text
//! cargo run --release -p irys-diff --features gpu,assets --example paper_hero -- <out dir>
//!   [--pavilion DIR] [--width 2400] [--spp 256] [--at x,y] [--yaw deg]
//!   [--eye x,y,z] [--look x,y,z] [--fov deg]   (vertical half-angle)
//! ```
//! needs ../rose (or $ROSE) and the pavilion export under $IRYS_ASSETS.


use std::fmt::Write as _;
use std::path::PathBuf;

use irys_core::{Point3, Transform};
use irys_diff::look::openpbr::OpenPbr;
use irys_diff::look::scene::{self, Camera, Instance, Material, Mesh, mask};
use irys_diff::look::scene_gpu::{GpuLook, Opts};

fn opt<T: std::str::FromStr>(a: &[String], k: &str, d: T) -> T {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).and_then(|s| s.parse().ok()).unwrap_or(d)
}
fn v3(a: &[String], k: &str, d: [f64; 3]) -> [f64; 3] {
    match a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)) {
        Some(s) => {
            let v: Vec<f64> = s.split(',').map(|x| x.parse().unwrap()).collect();
            [v[0], v[1], v.get(2).copied().unwrap_or(0.0)]
        }
        None => d,
    }
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: [f64; 3]) -> [f64; 3] {
    let n = dot(a, a).sqrt();
    [a[0] / n, a[1] / n, a[2] / n]
}

/// Robot millimetres (posed by `t`) to pavilion metres, standing at `at`, turned by `yaw`.
struct Place {
    at: [f64; 2],
    floor: f64,
    yaw: f64,
}
impl Place {
    fn world(&self, t: &Transform, p: Point3) -> [f64; 3] {
        let q = t.apply_point(&p);
        let (s, c) = self.yaw.to_radians().sin_cos();
        let (x, y, z) = (q.x * 1e-3, q.y * 1e-3, q.z * 1e-3);
        [self.at[0] + c * x - s * y, self.at[1] + s * x + c * y, self.floor + z]
    }
    fn xf(&self, t: &Transform) -> [[f32; 4]; 3] {
        let o = self.world(t, Point3::new(0.0, 0.0, 0.0));
        let col = |e: [f64; 3]| sub(self.world(t, Point3::new(e[0], e[1], e[2])), o);
        let (ex, ey, ez) = (col([1.0, 0.0, 0.0]), col([0.0, 1.0, 0.0]), col([0.0, 0.0, 1.0]));
        std::array::from_fn(|r| [ex[r] as f32, ey[r] as f32, ez[r] as f32, o[r] as f32])
    }
}

fn project(c: &Camera, size: (usize, usize), p: [f64; 3]) -> Option<[f64; 2]> {
    let d = sub(p, c.eye);
    let z = dot(d, c.forward);
    if z <= 0.0 {
        return None;
    }
    let k = size.1 as f64 / 2.0 / c.tan_half;
    Some([size.0 as f64 / 2.0 + dot(d, c.right) / z * k, size.1 as f64 / 2.0 - dot(d, c.up) / z * k])
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let out = PathBuf::from(a.get(1).cloned().unwrap_or("out/paper_hero".into()));
    std::fs::create_dir_all(&out).unwrap();
    let assets = std::env::var("IRYS_ASSETS").unwrap_or("/Users/cam/Developer/irys-assets".into());
    let pav = PathBuf::from(opt(&a, "--pavilion", format!("{assets}/blender-demos/export5b/pavilion")));
    let width: usize = opt(&a, "--width", 2400);
    let mut sc = scene::load(&pav, Some(width)).expect("pavilion export");
    let size = sc.size;

    // rose full, streamlined (vcad hardware/rose-full at rose-streamlined 61e722d4a): the
    // assembly exported to GLB at zero pose (mm, x fore, y left, z up, soles on z = 0), each
    // part's material from rose-full.joints.json.
    let rose_glb = PathBuf::from(opt(&a, "--rose-glb", "out/rose-full.glb".to_string()));
    let joints: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(opt(&a, "--rose-joints", "/Users/cam/Developer/vcad-rose-streamlined/hardware/rose-full/rose-full.joints.json".to_string())).unwrap()).unwrap();
    let mut part_mat = std::collections::HashMap::new();
    for (_, l) in joints["links"].as_object().unwrap() {
        for p in l["parts"].as_array().unwrap() {
            part_mat.insert(p["name"].as_str().unwrap().to_string(), p["material"].as_str().unwrap().to_string());
        }
    }
    // rose quad00/vcad_export.py's MATS: (base rgb, metalness, roughness).
    let pal: &[(&str, [f64; 3], f64, f64)] = &[
        ("satin_black", [0.090, 0.094, 0.106], 0.1, 0.55), ("municipal_orange", [1.0, 0.353, 0.039], 0.0, 0.45),
        ("gunmetal", [0.30, 0.32, 0.34], 0.8, 0.35), ("visor_glass", [0.02, 0.02, 0.025], 0.2, 0.05),
        ("sole_rubber", [0.17, 0.176, 0.19], 0.0, 0.9), ("aluminium", [0.75, 0.76, 0.78], 0.9, 0.3),
        ("motor_dark", [0.08, 0.085, 0.09], 0.6, 0.4), ("cable_black", [0.03, 0.03, 0.035], 0.0, 0.7),
        ("heatsink", [0.55, 0.57, 0.6], 0.7, 0.4), ("pcb_green", [0.05, 0.32, 0.18], 0.1, 0.45),
        ("cell_pack", [0.13, 0.22, 0.42], 0.1, 0.5), ("estop_red", [0.78, 0.04, 0.03], 0.0, 0.35),
        ("estop_yellow", [0.98, 0.78, 0.05], 0.0, 0.45),
    ];
    let mat0 = sc.materials.len();
    for (n, c, m, r) in pal {
        sc.materials.push(Material::new(n, OpenPbr { base_color: *c, base_metalness: *m, specular_roughness: *r, ..OpenPbr::default() }));
    }
    // The node head's own materials (hardware/rose-full/node.loon): dark glass over the screen,
    // the face's eyes lit, as vcad's photoreal render shows them.
    let head0 = mat0 + pal.len();
    let head_mats = [
        Material::new("node glass", OpenPbr { base_color: [0.01, 0.012, 0.014], base_metalness: 0.0, specular_roughness: 0.04, ..OpenPbr::default() }),
        Material::emitter("node face", [2.2, 3.0, 2.4]),
        Material::new("node lens", OpenPbr { base_color: [0.035, 0.04, 0.045], base_metalness: 0.3, specular_roughness: 0.08, ..OpenPbr::default() }),
    ];
    let mat_of = |part: &str| {
        let key = part.trim_end_matches("_i");
        match key {
            "full-glass" => return head0,
            "full-face" => return head0 + 1,
            "full-lenses" => return head0 + 2,
            _ => {}
        }
        let m = part_mat.get(key).map(String::as_str).unwrap_or("satin_black");
        mat0 + pal.iter().position(|p| p.0 == m).unwrap_or(0)
    };
    sc.materials.extend(head_mats);
    let place = Place { at: { let v = v3(&a, "--at", [-6.0, 0.4, 0.0]); [v[0], v[1]] }, floor: opt(&a, "--floor", 1.49), yaw: opt(&a, "--yaw", 200.0) };
    let id = Transform::identity();
    let glb_bytes = std::fs::read(&rose_glb).expect("rose-full.glb (vcad export hardware/rose-full/rose-full.loon)");
    let doc = gltf::Gltf::from_slice(&glb_bytes).expect("glb");
    let blob = doc.blob.clone().expect("glb binary chunk");
    // node names (the gltf crate's `names` feature is off here): from the JSON chunk
    let jlen = u32::from_le_bytes(glb_bytes[12..16].try_into().unwrap()) as usize;
    let gj: serde_json::Value = serde_json::from_slice(&glb_bytes[20..20 + jlen]).unwrap();
    let node_name = |i: usize| gj["nodes"][i]["name"].as_str().unwrap_or("").to_string();
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for node in doc.nodes() {
        let Some(m) = node.mesh() else { continue };
        let name = node_name(node.index());
        let t = node.transform().matrix();
        for prim in m.primitives() {
            let r = prim.reader(|_| Some(&blob[..]));
            let pos: Vec<[f32; 3]> = r
                .read_positions()
                .unwrap()
                .map(|p| {
                    let q = [0, 1, 2].map(|i| t[0][i] * p[0] + t[1][i] * p[1] + t[2][i] * p[2] + t[3][i]);
                    for i in 0..3 {
                        lo[i] = lo[i].min(q[i] as f64);
                        hi[i] = hi[i].max(q[i] as f64);
                    }
                    q
                })
                .collect();
            let idx: Vec<u32> = match r.read_indices() {
                Some(i) => i.into_u32().collect(),
                None => (0..pos.len() as u32).collect(),
            };
            let mi = sc.meshes.len();
            sc.meshes.push(Mesh { positions: pos, normals: Vec::new(), uvs: Vec::new(), indices: idx, material: mat_of(&name) });
            sc.instances.push(Instance { mesh: mi, xf: place.xf(&id), mask: mask::GEOMETRY });
        }
    }
    eprintln!("rose full: {} parts, bbox {:?} .. {:?} (expect mm, z up, soles at 0)", doc.nodes().count(), lo, hi);

    // Camera: low, three-quarter, the robot left of centre, the pavilion's
    // water and travertine behind.
    let mid = place.world(&id, Point3::new(0.0, 0.0, 900.0));
    let eye = v3(&a, "--eye", [mid[0] - 2.6, mid[1] - 1.4, 1.15]);
    let look = v3(&a, "--look", [mid[0] + 0.25, mid[1] + 0.35, mid[2] - 0.05]);
    let fwd = norm(sub(look, eye));
    let right = norm(cross(fwd, [0.0, 0.0, 1.0]));
    let up = cross(right, fwd);
    let fov: f64 = opt(&a, "--fov", 15.0);
    sc.camera = Camera { eye, forward: fwd, right, up, tan_half: fov.to_radians().tan() };
    let cam = sc.camera.clone();

    // Annotation, through the same camera.
    let mut ann = String::new();
    let p2 = |p: [f64; 3]| project(&cam, size, p).map(|q| format!("{:.1}\t{:.1}", q[0], q[1])).unwrap_or("nan\tnan".into());
    writeln!(ann, "size\t{}\t{}", size.0, size.1).unwrap();
    // every joint at zero pose: its name, the pivot, and ±80 mm along its axis
    for j in joints["joints"].as_array().unwrap() {
        let name = j["name"].as_str().unwrap();
        let v = |k: &str| j[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect::<Vec<_>>();
        let (p, ax) = (v("pivot_mm"), v("axis"));
        let at = |s: f64| place.world(&id, Point3::new(p[0] + ax[0] * s, p[1] + ax[1] * s, p[2] + ax[2] * s));
        writeln!(ann, "joint\t{name}\t{}\t{}\t{}", p2(at(0.0)), p2(at(-80.0)), p2(at(80.0))).unwrap();
    }
    for (name, l) in joints["links"].as_object().unwrap() {
        let Some(b) = l["aabb_mm"].as_array() else { continue }; // the ankle cross has no parts of its own
        let c: Vec<f64> = (0..3).map(|i| (b[0][i].as_f64().unwrap() + b[1][i].as_f64().unwrap()) / 2.0).collect();
        writeln!(ann, "link\t{name}\t{}", p2(place.world(&id, Point3::new(c[0], c[1], c[2])))).unwrap();
    }
    std::fs::write(out.join("annotation.tsv"), ann).unwrap();

    let spp: u32 = opt(&a, "--spp", 256);
    let ctx = irys_diff::gpu::context().expect("gpu");
    let g = GpuLook::new(ctx, &sc).expect("ray queries");
    let t0 = std::time::Instant::now();
    let fr = g.render(&cam, &Opts { spp, max_bounces: 7, clamp: 10.0, seed: 3, ..Opts::default() });
    eprintln!("{}×{} {spp} spp in {:.1?}", size.0, size.1, t0.elapsed());
    let expose: f32 = opt(&a, "--expose", 1.0);
    let px: Vec<u8> = fr
        .rgb
        .iter()
        .map(|&x| {
            let x = if x.is_finite() { (x * expose).max(0.0) as f64 } else { 0.0 };
            let t = x / (1.0 + x / 4.0) ; // soft shoulder
            let t = t.min(1.0);
            let s = if t <= 0.0031308 { 12.92 * t } else { 1.055 * t.powf(1.0 / 2.4) - 0.055 };
            (s * 255.0 + 0.5) as u8
        })
        .collect();
    image::RgbImage::from_raw(size.0 as u32, size.1 as u32, px).unwrap().save(out.join("beauty.png")).unwrap();
    println!("wrote {}", out.display());
}
