//! Phase 5a: a Blender scene, exported by `scripts/blend_export.py`, as a
//! [`Scene`].
//!
//! The export is a directory holding `scene.glb` (glTF 2.0: each unique
//! mesh once, one primitive per material, one node per instance, Y-up) and
//! `irys.json` (the sidecar: camera, lights, the environment fit, each glTF
//! material's irys BSDF, and what the exporter dropped). [`load`] turns it
//! into irys's current model:
//!
//! - one bottom-level tree per (mesh, primitive), placed once per node, so
//!   instancing survives;
//! - a diffuse primitive's placement carries its scalar albedo
//!   ([`Placement::albedo`], with [`Scene::floor_albedo`] = 1);
//! - conductors share one `θ`: the conductor covering the most placed
//!   triangles sets it (`α` = roughness², F0 per band from its colour);
//! - every emitter shares one radiance ([`Scene::emission`]), the largest;
//!   each emitter triangle is shrunk about its centroid by `√(L / L_max)`,
//!   the same power from the same place;
//! - area lights become emitter quads or discs, point and spot lights
//!   emitter spheres of their soft radius, the brightest sun the
//!   environment's lobe; the world is the sidecar's gradient fit;
//! - glass (`hidden`) is left out: irys has no dielectric.
//!
//! [`Options::grey`] replaces every non-emitter with a grey Lambertian of
//! that albedo: the equal-material benchmark scene, which Cycles renders
//! from irys's own export of it.

use std::path::Path;

use irys_core::{Point3, Transform, TriMesh, Vec3, transform_from_column_major};
use irys_prims::analytic::Analytic;
use serde_json::Value;

use crate::ggx::PARAMS;
use crate::prb::{Env, Opaque, Placement, Scene, Surface, uv_sphere};

/// The conductor when a scene places none: phase 2c's gold.
pub const GOLD: [f64; PARAMS] = [0.2, 0.34, 0.78, 1.0];

/// How to read an export.
#[derive(Debug, Clone)]
pub struct Options {
    /// Image width; the height keeps the export's aspect. `None` is the
    /// export's own size.
    pub width: Option<usize>,
    /// Paths per pixel per frame.
    pub spp: usize,
    /// Longest path, in surface vertices (Cycles' `max_bounces + 1`).
    pub depth: usize,
    /// Every non-emitter a grey Lambertian of this albedo.
    pub grey: Option<f64>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            width: None,
            spp: 1,
            depth: 8,
            grey: None,
        }
    }
}

/// An export as an irys scene.
pub struct Imported {
    /// The scene: NEE on, complete paths, `depth` vertices.
    pub scene: Scene,
    /// The conductor every conductor primitive renders with.
    pub theta: [f64; PARAMS],
    /// What the import kept and what it could not.
    pub report: Report,
}

/// Counts and losses of one import.
#[derive(Debug, Clone, Default)]
pub struct Report {
    /// Placements (instances × primitives, plus the lights' one).
    pub placements: usize,
    /// Bottom-level trees.
    pub trees: usize,
    /// Triangles in the trees.
    pub triangles_unique: usize,
    /// Triangles placed.
    pub triangles_placed: usize,
    /// Emitter triangles, world space.
    pub emitter_triangles: usize,
    /// The shared emitter radiance.
    pub emission: f64,
    /// The smallest emitter shrink `√(L / L_max)`.
    pub min_shrink: f64,
    /// The material that set `θ`, if any conductor is placed.
    pub conductor: Option<String>,
    /// Losses: the exporter's and the import's own.
    pub dropped: Vec<String>,
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}

fn v3(v: &Value) -> Vec3 {
    Vec3::new(f(&v[0]), f(&v[1]), f(&v[2]))
}

/// The material record's irys surface and the radiance an emitter has.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Diffuse(f64),
    Conductor,
    Emitter(f64, bool),
    Hidden,
}

fn kind(m: &Value, grey: Option<f64>) -> Kind {
    let k = m["kind"].as_str().unwrap_or("diffuse");
    match (k, grey) {
        ("emitter", _) => Kind::Emitter(f(&m["radiance"]), m["side"].as_str() == Some("back")),
        ("hidden", _) => Kind::Hidden,
        (_, Some(a)) => Kind::Diffuse(a),
        ("conductor", None) => Kind::Conductor,
        (_, None) => Kind::Diffuse(f(&m["albedo"]).clamp(0.0, 0.99)),
    }
}

/// Shrink a triangle about its centroid by `s` (area by `s²`).
fn shrink(t: [Point3; 3], s: f64) -> [Point3; 3] {
    let c = Point3::from((t[0].coords() + t[1].coords() + t[2].coords()) / 3.0);
    t.map(|p| c + (p - c) * s)
}

/// Read the export in `dir`.
pub fn load(dir: &Path, opts: &Options) -> Result<Imported, String> {
    let side: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("irys.json")).map_err(|e| format!("{}: {e}", dir.display()))?,
    )
    .map_err(|e| format!("irys.json: {e}"))?;
    let bytes = std::fs::read(dir.join("scene.glb")).map_err(|e| format!("scene.glb: {e}"))?;
    let glb = gltf::Gltf::from_slice(&bytes).map_err(|e| e.to_string())?;
    let blob = glb.blob.as_deref().ok_or("glb without a binary chunk")?;
    let mats = side["materials"].as_array().ok_or("sidecar without materials")?;
    let mut report = Report::default();
    for (k, v) in side["dropped"].as_object().into_iter().flatten() {
        match v {
            Value::Array(a) => report
                .dropped
                .extend(a.iter().map(|x| format!("{k}: {}", x.as_str().unwrap_or("?")))),
            Value::Object(o) => report
                .dropped
                .extend(o.iter().map(|(n, c)| format!("{k}: {n} ({c})"))),
            _ => {}
        }
    }

    // Every emitter's radiance, to pick the shared one.
    let kinds: Vec<Kind> = mats.iter().map(|m| kind(m, opts.grey)).collect();
    let lights = side["lights"].as_array().cloned().unwrap_or_default();
    let mut l_max: f64 = 0.0;
    for k in &kinds {
        if let Kind::Emitter(l, _) = k {
            l_max = l_max.max(*l);
        }
    }
    for l in &lights {
        if l["kind"].as_str() != Some("sun") {
            l_max = l_max.max(f(&l["radiance"]));
        }
    }
    let mut min_shrink: f64 = 1.0;
    let mut scale_for = |l: f64| {
        let s = (l / l_max).sqrt();
        min_shrink = min_shrink.min(s);
        s
    };

    // One tree per (mesh, primitive); `trees[mesh][prim]` its index.
    let mut blas: Vec<Opaque> = Vec::new();
    let mut tree_of: Vec<Vec<Option<(usize, Kind, usize)>>> = Vec::new();
    for mesh in glb.meshes() {
        let mut row = Vec::new();
        for prim in mesh.primitives() {
            let mi = prim.material().index().unwrap_or(0);
            let k = kinds.get(mi).copied().unwrap_or(Kind::Diffuse(0.8));
            if k == Kind::Hidden {
                row.push(None);
                continue;
            }
            let reader = prim.reader(|b| match b.source() {
                gltf::buffer::Source::Bin => Some(blob),
                gltf::buffer::Source::Uri(_) => None,
            });
            let pos: Vec<Point3> = reader
                .read_positions()
                .ok_or("primitive without positions")?
                .map(|p| Point3::new(p[0] as f64, p[1] as f64, p[2] as f64))
                .collect();
            let mut idx: Vec<u32> = reader
                .read_indices()
                .ok_or("primitive without indices")?
                .into_u32()
                .collect();
            let (pos, surface) = match k {
                Kind::Emitter(l, back) => {
                    if back {
                        for t in idx.chunks_exact_mut(3) {
                            t.swap(1, 2);
                        }
                    }
                    let s = scale_for(l);
                    let mut out = Vec::with_capacity(idx.len());
                    for t in idx.chunks_exact(3) {
                        let tri = shrink([pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]], s);
                        out.extend(tri);
                    }
                    idx = (0..out.len() as u32).collect();
                    (out, Surface::Emitter)
                }
                Kind::Conductor => (pos, Surface::Conductor),
                _ => (pos, Surface::Diffuse),
            };
            // Geometric normals only: the device shades with them, and the
            // CPU must agree with it.
            let tm = TriMesh::new(pos, Vec::new(), &idx);
            let n = tm.triangles().len();
            if n == 0 {
                row.push(None);
                continue;
            }
            report.triangles_unique += n;
            blas.push(Opaque::new(tm, surface, Analytic::from_prims(Vec::new()), &[]));
            row.push(Some((blas.len() - 1, k, n)));
        }
        tree_of.push(row);
    }

    // glTF is Y-up: node = C · world with C (x, y, z) → (x, z, −y).
    let c_inv = Transform::from_matrix(tang::Mat4::new(
        1.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, -1.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ));
    let mut placements = Vec::new();
    let mut conductor_tris: Vec<usize> = vec![0; mats.len()];
    for node in glb.nodes() {
        let Some(mesh) = node.mesh() else { continue };
        let m: Vec<f64> = node
            .transform()
            .matrix()
            .iter()
            .flatten()
            .map(|&x| x as f64)
            .collect();
        let to_world = c_inv.then(&transform_from_column_major(&m).ok_or("node matrix")?);
        for (pi, prim) in mesh.primitives().enumerate() {
            let Some((t, k, n)) = tree_of[mesh.index()][pi] else { continue };
            let p = Placement::new(t, to_world.clone());
            let p = match k {
                Kind::Diffuse(a) => p.tinted(a),
                _ => p.tinted(1.0),
            };
            if k == Kind::Conductor {
                conductor_tris[prim.material().index().unwrap_or(0)] += n;
            }
            report.triangles_placed += n;
            placements.push(p);
        }
    }

    // Lights: emitter geometry in world space, one tree, placed once.
    let mut env = Env {
        sky: 0.0,
        sun_radiance: 0.0,
        ..Env::default()
    };
    let mut lpos: Vec<Point3> = Vec::new();
    let mut best_sun = 0.0;
    for l in &lights {
        let name = l["name"].as_str().unwrap_or("?");
        match l["kind"].as_str().unwrap_or("") {
            "sun" => {
                let e = f(&l["irradiance"]);
                if e <= best_sun {
                    report.dropped.push(format!("lights: {name}: a second sun"));
                    continue;
                }
                best_sun = e;
                let r = f(&l["radius"]).max(1e-3);
                let kappa = 2.0 / (r * r);
                env.sun = v3(&l["dir"]).normalize();
                env.kappa = kappa;
                env.sun_radiance = e * kappa / (2.0 * std::f64::consts::PI);
            }
            k @ ("quad" | "disk") => {
                let (c, u, v, n) = (v3(&l["center"]), v3(&l["u"]), v3(&l["v"]), v3(&l["normal"]));
                let s = scale_for(f(&l["radiance"]));
                let c = Point3::from(c);
                let ring: Vec<Point3> = if k == "quad" {
                    vec![c - u - v, c + u - v, c + u + v, c - u + v]
                } else {
                    (0..32)
                        .map(|i| {
                            let a = std::f64::consts::TAU * i as f64 / 32.0;
                            c + u * a.cos() + v * a.sin()
                        })
                        .collect()
                };
                let ring: Vec<Point3> = ring.iter().map(|p| c + (p - c) * s).collect();
                for i in 1..ring.len() - 1 {
                    let mut t = [ring[0], ring[i], ring[i + 1]];
                    if (t[1] - t[0]).cross(&(t[2] - t[0])).dot(&n) < 0.0 {
                        t.swap(1, 2);
                    }
                    lpos.extend(t);
                }
            }
            "sphere" => {
                let c = Point3::from(v3(&l["center"]));
                let r = f(&l["radius"]) * scale_for(f(&l["radiance"]));
                let sph = uv_sphere(8, 16);
                for t in sph.triangles() {
                    lpos.extend(t.map(|i| c + sph.positions()[i as usize].coords() * r));
                }
            }
            other => report.dropped.push(format!("lights: {name}: kind {other}")),
        }
    }
    if !lpos.is_empty() {
        let idx: Vec<u32> = (0..lpos.len() as u32).collect();
        let tm = TriMesh::new(lpos, Vec::new(), &idx);
        report.triangles_unique += tm.triangles().len();
        report.triangles_placed += tm.triangles().len();
        blas.push(Opaque::new(tm, Surface::Emitter, Analytic::from_prims(Vec::new()), &[]));
        placements.push(Placement::new(blas.len() - 1, Transform::identity()).tinted(1.0));
    }
    env.sky = f(&side["env"]["sky"]);

    // The conductor: the one covering the most placed triangles.
    let mut theta = GOLD;
    if let Some((mi, _)) = conductor_tris.iter().enumerate().filter(|(_, n)| **n > 0).max_by_key(|(_, n)| **n) {
        let m = &mats[mi];
        let r = f(&m["roughness"]).clamp(0.05, 1.0);
        let rgb = &m["rgb"];
        theta = [r * r, f(&rgb[2]).clamp(0.02, 1.0), f(&rgb[1]).clamp(0.02, 1.0), f(&rgb[0]).clamp(0.02, 1.0)];
        report.conductor = m["name"].as_str().map(str::to_string);
        let others = conductor_tris.iter().filter(|n| **n > 0).count() - 1;
        if others > 0 {
            report
                .dropped
                .push(format!("import: {others} other conductors render as {}", m["name"]));
        }
    }

    // The camera.
    let cam = &side["camera"];
    let (w0, h0) = (f(&cam["size"][0]) as usize, f(&cam["size"][1]) as usize);
    let (w, h) = match opts.width {
        Some(w) => (w, ((w as f64) * h0 as f64 / w0 as f64).round() as usize),
        None => (w0, h0),
    };
    let eye = Point3::from(v3(&cam["eye"]));
    let fwd = v3(&cam["forward"]);
    let tan_half = f(&cam["tan_half"]);
    report.trees = blas.len();
    report.placements = placements.len();
    let mut scene = Scene::instanced(
        blas,
        placements,
        None,
        (w, h),
        opts.spp,
        (eye, eye + fwd),
        tan_half.atan().to_degrees(),
    );
    scene.set_camera(eye, fwd.normalize(), v3(&cam["right"]).normalize(), v3(&cam["up"]).normalize(), tan_half);
    scene.floor_albedo = 1.0;
    scene.floor_checker = None;
    scene.nee = true;
    scene.complete = true;
    scene.depth = opts.depth;
    scene.env = env;
    scene.emission = if l_max > 0.0 { l_max } else { 1.0 };
    report.emission = scene.emission;
    report.min_shrink = min_shrink;
    report.emitter_triangles = scene.lights().tris.len();
    Ok(Imported { scene, theta, report })
}
