//! Phase 5b: a full scene with OpenPBR materials, as the device kernel
//! (`look/scene.wgsl`, [`super::scene_gpu`]) renders it.
//!
//! [`LookScene`] is plain data: triangle meshes with normals and UVs, one
//! material each; instances placing them; OpenPBR materials (plus the three
//! things Blender's older graphs need that OpenPBR does not name: a separate
//! metal colour, thin diffuse translucency and an alpha cut-out); 8-bit sRGB
//! textures; an equirect world; a sun; emitters (emissive meshes, area,
//! point and spot lights as geometry, each with its own RGB radiance); and
//! light portals (windows onto their own sky, seen only by bounce rays and
//! passed by shadow rays, as Cycles' are).
//!
//! [`load`] reads `scripts/blend_export.py`'s directory (scene.glb, irys.json,
//! textures/, env.f32, portal_*.f32). The test scenes build one by hand.

use std::path::Path;

use super::openpbr::{OpenPbr, SubsurfaceMode, hats};

/// Instance visibility masks (TLAS instance masks).
pub mod mask {
    /// Ordinary geometry: every ray.
    pub const GEOMETRY: u32 = 0x01;
    /// A light portal: bounce rays only.
    pub const PORTAL: u32 = 0x02;
    /// A light's geometry that the camera does not see.
    pub const LIGHT_HIDDEN: u32 = 0x04;
    /// A light's geometry the camera sees.
    pub const LIGHT: u32 = 0x08;
}

/// A triangle mesh with one material.
#[derive(Debug, Clone, Default)]
pub struct Mesh {
    /// Object-space positions.
    pub positions: Vec<[f32; 3]>,
    /// Per-vertex normals (empty: flat).
    pub normals: Vec<[f32; 3]>,
    /// Per-vertex UVs, glTF's (v down; empty: zero).
    pub uvs: Vec<[f32; 2]>,
    /// Triangle corners.
    pub indices: Vec<u32>,
    /// Index into [`LookScene::materials`].
    pub material: usize,
}

/// One placement of a mesh.
#[derive(Debug, Clone)]
pub struct Instance {
    /// Index into [`LookScene::meshes`].
    pub mesh: usize,
    /// Object → world, row-major 3 × 4.
    pub xf: [[f32; 4]; 3],
    /// Visibility ([`mask`]).
    pub mask: u32,
}

/// A spot light's cone (Cycles' `spot_attenuation`).
#[derive(Debug, Clone, Copy)]
pub struct Spot {
    /// The light's centre.
    pub center: [f64; 3],
    /// Unit axis.
    pub axis: [f64; 3],
    /// `cos(spot_size / 2)`.
    pub cos_half: f64,
    /// `(1 − cos_half) · blend`.
    pub smooth: f64,
}

/// An OpenPBR material and what irys adds to it.
#[derive(Debug, Clone)]
pub struct Material {
    /// Name, for reports.
    pub name: String,
    /// The OpenPBR parameters.
    pub pbr: OpenPbr,
    /// The metal slab's F0 colour, when it is not the base colour (a Blender
    /// Mix of Diffuse and Glossy BSDFs). `None`: base colour, the spec's.
    pub metal_color: Option<[f64; 3]>,
    /// Thin-walled: transmission passes straight through and shadow rays
    /// pass the surface (architectural glass).
    pub thin_walled: bool,
    /// `geometry_opacity` where there is no cut-out texture: a constant
    /// stochastic pass.
    pub opacity: f64,
    /// Share of the diffuse lobe transmitted diffusely (thin translucency).
    pub translucency: f64,
    /// Its colour.
    pub translucent_color: [f64; 3],
    /// Emitted radiance, linear RGB (no BSDF under it).
    pub emission: [f64; 3],
    /// Emits from the back of its triangles.
    pub emit_back: bool,
    /// A Blender light, not a surface: rays collect its emission and go on
    /// (Cycles' lights block nothing).
    pub light: bool,
    /// Emits from both sides (spheres, point lights).
    pub emit_both: bool,
    /// Base-colour texture (alpha: the cut-out, when [`Self::cutout`]).
    pub base_tex: Option<usize>,
    /// Metal-colour texture.
    pub metal_tex: Option<usize>,
    /// Opacity texture when there is no base-colour texture.
    pub alpha_tex: Option<usize>,
    /// Alpha-tested at 0.5.
    pub cutout: bool,
    /// Texture transform `uv · s + o`: `(sx, sy, ox, oy)`.
    pub uv_xf: [f32; 4],
    /// A portal's sky: index into [`LookScene::portal_skies`], looked up at
    /// the reflected direction (Blender's Texture Coordinate › Reflection).
    pub portal_sky: Option<usize>,
    /// A spot light's cone.
    pub spot: Option<Spot>,
}

impl Material {
    /// A plain OpenPBR material.
    pub fn new(name: &str, pbr: OpenPbr) -> Self {
        Self {
            name: name.into(),
            pbr,
            metal_color: None,
            thin_walled: false,
            opacity: 1.0,
            translucency: 0.0,
            translucent_color: [0.8; 3],
            emission: [0.0; 3],
            emit_back: false,
            light: false,
            emit_both: false,
            base_tex: None,
            metal_tex: None,
            alpha_tex: None,
            cutout: false,
            uv_xf: [1.0, 1.0, 0.0, 0.0],
            portal_sky: None,
            spot: None,
        }
    }

    /// An emitter of radiance `rgb` and nothing else.
    pub fn emitter(name: &str, rgb: [f64; 3]) -> Self {
        let mut m = Self::new(name, OpenPbr::default());
        m.emission = rgb;
        m
    }

    /// Whether it emits.
    pub fn emits(&self) -> bool {
        self.portal_sky.is_some() || self.emission.iter().any(|&e| e > 0.0)
    }

    /// Whether shadow rays or camera rays may pass parts of it.
    pub fn non_opaque(&self) -> bool {
        self.cutout || self.thin_walled || self.opacity < 1.0
    }
}

/// An 8-bit sRGB RGBA texture, row 0 at the top (glTF's `v = 0`).
#[derive(Debug, Clone)]
pub struct Texture {
    /// Width, height.
    pub size: (u32, u32),
    /// Packed RGBA8 (little-endian: r in the low byte).
    pub texels: Vec<u32>,
}

/// A linear-RGB equirect: row 0 is `+z`, column `u` is at azimuth
/// `atan2(−x, −y) = 2π u` (Blender's equirect, as `bake_world` writes it).
#[derive(Debug, Clone)]
pub struct Equirect {
    /// Width, height.
    pub size: (u32, u32),
    /// Texels, row-major.
    pub rgb: Vec<[f32; 3]>,
}

impl Equirect {
    /// The texel a direction falls in.
    pub fn texel(&self, d: [f64; 3]) -> (usize, usize) {
        let (w, h) = (self.size.0 as f64, self.size.1 as f64);
        let mut phi = (-d[0]).atan2(-d[1]);
        if phi < 0.0 {
            phi += std::f64::consts::TAU;
        }
        let u = phi / std::f64::consts::TAU;
        let v = d[2].clamp(-1.0, 1.0).acos() / std::f64::consts::PI;
        (((u * w) as usize).min(w as usize - 1), ((v * h) as usize).min(h as usize - 1))
    }

    /// Radiance toward `d`.
    pub fn at(&self, d: [f64; 3]) -> [f32; 3] {
        let (i, j) = self.texel(d);
        self.rgb[j * self.size.0 as usize + i]
    }
}

/// The sun: a uniform disc of angular radius `radius` around `dir`.
#[derive(Debug, Clone, Copy)]
pub struct Sun {
    /// Unit direction toward the sun.
    pub dir: [f64; 3],
    /// Angular radius.
    pub radius: f64,
    /// Disc radiance, linear RGB (irradiance / (π sin² r)).
    pub radiance: [f64; 3],
}

/// A pinhole camera; `tan_half` is the vertical half-angle's tangent.
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    /// Position.
    pub eye: [f64; 3],
    /// Unit forward.
    pub forward: [f64; 3],
    /// Unit right.
    pub right: [f64; 3],
    /// Unit up.
    pub up: [f64; 3],
    /// Vertical `tan(fov / 2)`.
    pub tan_half: f64,
}

/// A full scene for the device's OpenPBR kernel.
#[derive(Debug, Clone)]
pub struct LookScene {
    /// Meshes.
    pub meshes: Vec<Mesh>,
    /// Their placements.
    pub instances: Vec<Instance>,
    /// Materials.
    pub materials: Vec<Material>,
    /// Textures.
    pub textures: Vec<Texture>,
    /// The world, if it is not black.
    pub env: Option<Equirect>,
    /// The sun.
    pub sun: Option<Sun>,
    /// Portal skies.
    pub portal_skies: Vec<Equirect>,
    /// The camera.
    pub camera: Camera,
    /// Image size.
    pub size: (usize, usize),
    /// The file's camera frame by frame (`--frames`), and its frame rate.
    pub frames: Vec<Camera>,
    /// Frames per second of [`Self::frames`].
    pub fps: f64,
    /// What the import could not carry, with counts.
    pub dropped: Vec<String>,
}

fn f(v: &serde_json::Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}

fn v3(v: &serde_json::Value) -> [f64; 3] {
    [f(&v[0]), f(&v[1]), f(&v[2])]
}

fn rgb_or(v: &serde_json::Value, d: [f64; 3]) -> [f64; 3] {
    if v.is_array() { v3(v) } else { d }
}

fn camera_of(c: &serde_json::Value) -> Camera {
    Camera {
        eye: v3(&c["eye"]),
        forward: norm(v3(&c["forward"])),
        right: norm(v3(&c["right"])),
        up: norm(v3(&c["up"])),
        tan_half: f(&c["tan_half"]),
    }
}

fn norm(a: [f64; 3]) -> [f64; 3] {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-30);
    [a[0] / l, a[1] / l, a[2] / l]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

const IDENTITY: [[f32; 4]; 3] = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];

/// The OpenPBR record of the sidecar.
fn pbr_of(o: &serde_json::Value) -> OpenPbr {
    let d = OpenPbr::default();
    let g = |k: &str, x: f64| o.get(k).and_then(|v| v.as_f64()).unwrap_or(x);
    let c = |k: &str, x: [f64; 3]| o.get(k).map_or(x, |v| rgb_or(v, x));
    OpenPbr {
        base_weight: g("base_weight", d.base_weight),
        base_color: c("base_color", d.base_color),
        base_metalness: g("base_metalness", d.base_metalness).clamp(0.0, 1.0),
        specular_weight: g("specular_weight", d.specular_weight).max(0.0),
        specular_color: c("specular_color", d.specular_color),
        specular_roughness: g("specular_roughness", d.specular_roughness).clamp(0.0, 1.0),
        specular_ior: g("specular_ior", d.specular_ior).max(1.0001),
        transmission_weight: g("transmission_weight", d.transmission_weight).clamp(0.0, 1.0),
        transmission_color: c("transmission_color", d.transmission_color),
        transmission_depth: g("transmission_depth", d.transmission_depth),
        transmission_dispersion_scale: g("transmission_dispersion_scale", d.transmission_dispersion_scale),
        transmission_dispersion_abbe_number: g(
            "transmission_dispersion_abbe_number",
            d.transmission_dispersion_abbe_number,
        ),
        subsurface_weight: g("subsurface_weight", d.subsurface_weight).clamp(0.0, 1.0),
        subsurface_color: c("subsurface_color", d.subsurface_color),
        subsurface_radius: g("subsurface_radius", d.subsurface_radius),
        subsurface_radius_scale: c("subsurface_radius_scale", d.subsurface_radius_scale),
        subsurface_scatter_anisotropy: g("subsurface_scatter_anisotropy", 0.0),
        subsurface_mode: if o.get("subsurface_mode").and_then(|v| v.as_str()) == Some("cycles") {
            SubsurfaceMode::Cycles
        } else {
            SubsurfaceMode::Physical
        },
        coat_weight: g("coat_weight", d.coat_weight).clamp(0.0, 1.0),
        coat_color: c("coat_color", d.coat_color),
        coat_roughness: g("coat_roughness", d.coat_roughness).clamp(0.0, 1.0),
        coat_ior: g("coat_ior", d.coat_ior).max(1.0001),
        coat_darkening: g("coat_darkening", d.coat_darkening),
        fuzz_weight: g("fuzz_weight", d.fuzz_weight).clamp(0.0, 1.0),
        fuzz_color: c("fuzz_color", d.fuzz_color),
        fuzz_roughness: g("fuzz_roughness", d.fuzz_roughness),
        thin_film_weight: g("thin_film_weight", d.thin_film_weight),
        thin_film_thickness: g("thin_film_thickness", d.thin_film_thickness),
        thin_film_ior: g("thin_film_ior", d.thin_film_ior),
        emission_luminance: 0.0,
        emission_color: [1.0; 3],
    }
}

fn read_f32(path: &Path, n: usize) -> Result<Vec<[f32; 3]>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if b.len() != n * 12 {
        return Err(format!("{}: {} bytes, expected {}", path.display(), b.len(), n * 12));
    }
    Ok(b.chunks_exact(12)
        .map(|c| {
            let g = |i: usize| f32::from_le_bytes([c[i], c[i + 1], c[i + 2], c[i + 3]]);
            [g(0), g(4), g(8)]
        })
        .collect())
}

fn equirect(dir: &Path, v: &serde_json::Value) -> Result<Equirect, String> {
    let (w, h) = (f(&v["size"][0]) as u32, f(&v["size"][1]) as u32);
    let rgb = read_f32(&dir.join(v["path"].as_str().unwrap_or("")), (w * h) as usize)?;
    Ok(Equirect { size: (w, h), rgb })
}

fn texture(path: &Path) -> Result<Texture, String> {
    let img = image::open(path).map_err(|e| format!("{}: {e}", path.display()))?.to_rgba8();
    let (w, h) = img.dimensions();
    let texels = img.pixels().map(|p| u32::from_le_bytes(p.0)).collect();
    Ok(Texture { size: (w, h), texels })
}

/// A light record as an emitter mesh (world space), its material, and
/// whether the camera sees it.
fn light_mesh(l: &serde_json::Value, name: &str) -> Option<(Mesh, Material)> {
    let rgb = rgb_or(&l["rgb"], [1.0; 3]);
    let kind = l["kind"].as_str()?;
    let rad = f(&l["radiance"]);
    let mut mat = Material::emitter(name, rgb.map(|c| c * rad));
    let mut mesh = Mesh::default();
    let push = |mesh: &mut Mesh, p: [f64; 3]| mesh.positions.push(p.map(|x| x as f32));
    match kind {
        "quad" | "disk" => {
            let (c, u, v, n) = (v3(&l["center"]), v3(&l["u"]), v3(&l["v"]), v3(&l["normal"]));
            let at = |a: f64, b: f64| [c[0] + u[0] * a + v[0] * b, c[1] + u[1] * a + v[1] * b, c[2] + u[2] * a + v[2] * b];
            let ring: Vec<[f64; 3]> = if kind == "quad" {
                vec![at(-1.0, -1.0), at(1.0, -1.0), at(1.0, 1.0), at(-1.0, 1.0)]
            } else {
                (0..48)
                    .map(|i| {
                        let a = std::f64::consts::TAU * i as f64 / 48.0;
                        at(a.cos(), a.sin())
                    })
                    .collect()
            };
            push(&mut mesh, c);
            for p in &ring {
                push(&mut mesh, *p);
            }
            let k = ring.len() as u32;
            for i in 0..k {
                let (a, b) = (1 + i, 1 + (i + 1) % k);
                // wind so the face normal is the light's
                let e1 = [ring[(b - 1) as usize][0] - c[0], ring[(b - 1) as usize][1] - c[1], ring[(b - 1) as usize][2] - c[2]];
                let e0 = [ring[(a - 1) as usize][0] - c[0], ring[(a - 1) as usize][1] - c[1], ring[(a - 1) as usize][2] - c[2]];
                let fnrm = cross(e0, e1);
                if fnrm[0] * n[0] + fnrm[1] * n[1] + fnrm[2] * n[2] >= 0.0 {
                    mesh.indices.extend([0, a, b]);
                } else {
                    mesh.indices.extend([0, b, a]);
                }
            }
            if kind == "disk" {
                // same power as Blender's ellipse of area π/4 · sx sy: the
                // record's u, v are the half-axes, which the ring spans.
            }
        }
        "sphere" => {
            let c = v3(&l["center"]);
            let r = f(&l["radius"]);
            let (lat, lon) = (12u32, 24u32);
            for i in 0..=lat {
                let th = std::f64::consts::PI * i as f64 / lat as f64;
                for j in 0..lon {
                    let ph = std::f64::consts::TAU * j as f64 / lon as f64;
                    let d = [th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()];
                    push(&mut mesh, [c[0] + r * d[0], c[1] + r * d[1], c[2] + r * d[2]]);
                    mesh.normals.push(d.map(|x| x as f32));
                }
            }
            for i in 0..lat {
                for j in 0..lon {
                    let a = i * lon + j;
                    let b = i * lon + (j + 1) % lon;
                    let (c2, d2) = (a + lon, b + lon);
                    // outward winding
                    mesh.indices.extend([a, c2, b, b, c2, d2]);
                }
            }
            if let Some(s) = l.get("spot").filter(|s| s.is_object()) {
                let size = f(&s["size"]);
                let cos_half = (size * 0.5).cos();
                mat.spot = Some(Spot {
                    center: c,
                    axis: norm(v3(&s["axis"])),
                    cos_half,
                    smooth: ((1.0 - cos_half) * f(&s["blend"])).max(1e-6),
                });
            }
        }
        _ => return None,
    }
    if mesh.normals.len() != mesh.positions.len() {
        mesh.normals.clear();
    }
    Some((mesh, mat))
}

/// Read `scripts/blend_export.py`'s directory. `width`: the image width
/// (the height keeps the export's aspect); `None` is the export's size.
pub fn load(dir: &Path, width: Option<usize>) -> Result<LookScene, String> {
    let side: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("irys.json")).map_err(|e| format!("{}: {e}", dir.display()))?,
    )
    .map_err(|e| format!("irys.json: {e}"))?;
    if side.get("openpbr_version").is_none() {
        return Err("export predates phase 5b (run scripts/blend_export.py again)".into());
    }
    let bytes = std::fs::read(dir.join("scene.glb")).map_err(|e| format!("scene.glb: {e}"))?;
    let glb = gltf::Gltf::from_slice(&bytes).map_err(|e| e.to_string())?;
    let blob = glb.blob.as_deref().ok_or("glb without a binary chunk")?;
    let recs = side["materials"].as_array().ok_or("sidecar without materials")?;
    let mut dropped = Vec::new();
    if let Some(o) = side["dropped"]["openpbr"].as_object() {
        let mut v: Vec<(String, u64)> = o.iter().map(|(k, c)| (k.clone(), c.as_u64().unwrap_or(0))).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        dropped.extend(v.into_iter().map(|(k, c)| format!("materials: {k} ({c})")));
    }
    for k in ["lights", "camera"] {
        for x in side["dropped"][k].as_array().into_iter().flatten() {
            dropped.push(format!("{k}: {}", x.as_str().unwrap_or("?")));
        }
    }

    // Materials and their textures.
    let mut textures = Vec::new();
    let mut portal_skies = Vec::new();
    let mut materials = Vec::new();
    for r in recs {
        let name = r["name"].as_str().unwrap_or("?");
        let o = &r["openpbr"];
        let mut m = Material::new(name, pbr_of(o));
        m.metal_color = o.get("metal_color").filter(|v| v.is_array()).map(v3);
        m.thin_walled = f(&o["geometry_thin_walled"]) > 0.5;
        m.opacity = o.get("geometry_opacity").and_then(|v| v.as_f64()).unwrap_or(1.0).clamp(0.0, 1.0);
        m.translucency = f(&o["translucency"]).clamp(0.0, 1.0);
        m.translucent_color = rgb_or(&o["translucent_color"], [0.8; 3]);
        if let Some(e) = r.get("emission").filter(|e| e.is_object()) {
            m.emission = rgb_or(&e["rgb"], [0.0; 3]);
            m.emit_back = e["side"].as_str() == Some("back");
            if let Some(q) = e.get("equirect").filter(|q| q.is_object()) {
                portal_skies.push(equirect(dir, q)?);
                m.portal_sky = Some(portal_skies.len() - 1);
            }
        }
        let t = &r["textures"];
        let mut tex = |k: &str| -> Result<Option<(usize, [f32; 4], bool)>, String> {
            let Some(e) = t.get(k).filter(|e| e.is_object()) else { return Ok(None) };
            textures.push(texture(&dir.join(e["path"].as_str().unwrap_or("")))?);
            let xf = e["xf"].as_array().map_or([1.0, 1.0, 0.0, 0.0], |a| {
                [f(&a[0]) as f32, f(&a[1]) as f32, f(&a[2]) as f32, f(&a[3]) as f32]
            });
            Ok(Some((textures.len() - 1, xf, e["alpha"].as_bool().unwrap_or(false))))
        };
        if let Some((i, xf, a)) = tex("base_color")? {
            m.base_tex = Some(i);
            m.uv_xf = xf;
            m.cutout |= a;
        }
        if let Some((i, xf, _)) = tex("metal_color")? {
            m.metal_tex = Some(i);
            m.uv_xf = xf;
        }
        if let Some((i, xf, _)) = tex("opacity")? {
            m.alpha_tex = Some(i);
            m.uv_xf = xf;
            m.cutout = true;
        }
        if m.cutout {
            m.opacity = 1.0;
        }
        materials.push(m);
    }

    // Debugging aid: IRYS_LOOK_SET=name:key=value;... overrides a few fields.
    if let Ok(v) = std::env::var("IRYS_LOOK_SET") {
        for item in v.split(';').filter(|x| !x.is_empty()) {
            let (name, kv) = item.split_once(':').unwrap_or((item, ""));
            let (k, x) = kv.split_once('=').unwrap_or((kv, "0"));
            let x: f64 = x.parse().unwrap_or(0.0);
            for m in materials.iter_mut().filter(|m| m.name == name) {
                match k {
                    "translucency" => m.translucency = x,
                    "metalness" => m.pbr.base_metalness = x,
                    "specular_weight" => m.pbr.specular_weight = x,
                    "cutout" => m.cutout = x > 0.5,
                    "base" => m.pbr.base_color = [x; 3],
                    "notex" => m.base_tex = None,
                    _ => eprintln!("IRYS_LOOK_SET: unknown {k}"),
                }
            }
        }
    }
    // Debugging aid: IRYS_LOOK_HIDE=name,name drops those materials' meshes.
    let hide: Vec<String> = std::env::var("IRYS_LOOK_HIDE")
        .map(|v| v.split(',').map(str::to_string).collect())
        .unwrap_or_default();

    // Meshes: one per (glTF mesh, primitive).
    let mut meshes = Vec::new();
    let mut of: Vec<Vec<usize>> = Vec::new();
    for mesh in glb.meshes() {
        let mut row = Vec::new();
        for prim in mesh.primitives() {
            let reader = prim.reader(|b| match b.source() {
                gltf::buffer::Source::Bin => Some(blob),
                gltf::buffer::Source::Uri(_) => None,
            });
            let positions: Vec<[f32; 3]> = reader.read_positions().ok_or("no positions")?.collect();
            let normals: Vec<[f32; 3]> = reader.read_normals().map(|n| n.collect()).unwrap_or_default();
            let uvs: Vec<[f32; 2]> = reader.read_tex_coords(0).map(|t| t.into_f32().collect()).unwrap_or_default();
            let indices: Vec<u32> = reader.read_indices().ok_or("no indices")?.into_u32().collect();
            row.push(meshes.len());
            meshes.push(Mesh {
                positions,
                normals,
                uvs,
                indices,
                material: prim.material().index().unwrap_or(0),
            });
        }
        of.push(row);
    }
    // glTF is Y-up: world = C⁻¹ · node with C⁻¹ (x, y, z) → (x, −z, y).
    let mut instances = Vec::new();
    for node in glb.nodes() {
        let Some(mesh) = node.mesh() else { continue };
        let m = node.transform().matrix(); // column-major [col][row]
        let mut xf = [[0f32; 4]; 3];
        for c in 0..4 {
            let (x, y, z) = (m[c][0], m[c][1], m[c][2]);
            xf[0][c] = x;
            xf[1][c] = -z;
            xf[2][c] = y;
        }
        for &mi in &of[mesh.index()] {
            if hide.contains(&materials[meshes[mi].material].name) {
                continue;
            }
            instances.push(Instance { mesh: mi, xf, mask: mask::GEOMETRY });
        }
    }
    // Portals: world-space triangles, bounce rays only.
    for p in side["portals"].as_array().into_iter().flatten() {
        let mut mesh = Mesh {
            material: f(&p["material"]) as usize,
            ..Mesh::default()
        };
        for t in p["triangles"].as_array().into_iter().flatten() {
            for c in t.as_array().into_iter().flatten() {
                mesh.indices.push(mesh.positions.len() as u32);
                mesh.positions.push(v3(c).map(|x| x as f32));
            }
        }
        meshes.push(mesh);
        instances.push(Instance { mesh: meshes.len() - 1, xf: IDENTITY, mask: mask::PORTAL });
    }
    // Lights.
    let mut sun = None;
    for l in side["lights"].as_array().into_iter().flatten() {
        let name = l["name"].as_str().unwrap_or("light");
        if l["kind"].as_str() == Some("sun") {
            let r = f(&l["radius"]).max(1e-3);
            let e = f(&l["irradiance"]);
            let rgb = rgb_or(&l["rgb"], [1.0; 3]);
            let lr = e / (std::f64::consts::PI * r.sin().powi(2));
            if sun.is_some() {
                dropped.push(format!("lights: {name}: a second sun"));
                continue;
            }
            sun = Some(Sun { dir: norm(v3(&l["dir"])), radius: r, radiance: rgb.map(|c| c * lr) });
            continue;
        }
        let Some((mesh, mut mat)) = light_mesh(l, name) else {
            dropped.push(format!("lights: {name}: kind {}", l["kind"]));
            continue;
        };
        // A sphere light emits outward only (its winding is outward). Both
        // sides made next-event estimation, whose shadow rays lights do not
        // block, see the far side's inner face through the near one: twice
        // the light a BSDF-sampled ray (which stops at the near side) sees,
        // and the MIS mix of the two came out 2-25 % bright in the classroom.
        mat.emit_both = false;
        mat.light = true;
        materials.push(mat);
        meshes.push(Mesh { material: materials.len() - 1, ..mesh });
        let vis = l["visible_camera"].as_bool().unwrap_or(true);
        instances.push(Instance {
            mesh: meshes.len() - 1,
            xf: IDENTITY,
            mask: if vis { mask::LIGHT } else { mask::LIGHT_HIDDEN },
        });
    }
    let env = match side["env"].get("equirect").filter(|e| e.is_object()) {
        Some(e) => {
            let q = equirect(dir, e)?;
            q.rgb.iter().any(|t| t.iter().any(|&c| c > 0.0)).then_some(q)
        }
        None => None,
    };
    let cam = &side["camera"];
    let (w0, h0) = (f(&cam["size"][0]) as usize, f(&cam["size"][1]) as usize);
    let size = match width {
        Some(w) => (w, ((w as f64) * h0 as f64 / w0 as f64).round() as usize),
        None => (w0, h0),
    };
    let (frames, fps) = match side.get("camera_frames").filter(|c| c.is_object()) {
        Some(c) => (
            c["frames"].as_array().into_iter().flatten().map(camera_of).collect(),
            f(&c["fps"]),
        ),
        None => (Vec::new(), 24.0),
    };
    Ok(LookScene {
        meshes,
        instances,
        materials,
        textures,
        env,
        sun,
        portal_skies,
        camera: camera_of(cam),
        size,
        frames,
        fps,
        dropped,
    })
}

impl LookScene {
    /// Triangles: (unique, placed).
    pub fn triangles(&self) -> (usize, usize) {
        let u = self.meshes.iter().map(|m| m.indices.len() / 3).sum();
        let p = self.instances.iter().map(|i| self.meshes[i.mesh].indices.len() / 3).sum();
        (u, p)
    }

    /// A unit-sphere-ish test scene: `mats` on spheres in a row under a
    /// constant world of radiance `env` (white furnace when `env` is flat).
    pub fn spheres(mats: Vec<Material>, env: f32, size: (usize, usize)) -> Self {
        let n = mats.len();
        let (lat, lon) = (48u32, 96u32);
        let mut meshes = Vec::new();
        let mut instances = Vec::new();
        for (k, _) in mats.iter().enumerate() {
            let mut m = Mesh { material: k, ..Mesh::default() };
            for i in 0..=lat {
                let th = std::f64::consts::PI * i as f64 / lat as f64;
                for j in 0..=lon {
                    let ph = std::f64::consts::TAU * j as f64 / lon as f64;
                    let d = [th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()];
                    m.positions.push(d.map(|x| x as f32));
                    m.normals.push(d.map(|x| x as f32));
                    m.uvs.push([j as f32 / lon as f32, i as f32 / lat as f32]);
                }
            }
            for i in 0..lat {
                for j in 0..lon {
                    let a = i * (lon + 1) + j;
                    let (b, c, d) = (a + 1, a + lon + 1, a + lon + 2);
                    m.indices.extend([a, c, b, b, c, d]);
                }
            }
            meshes.push(m);
            let x = 2.4 * (k as f32 - (n as f32 - 1.0) / 2.0);
            instances.push(Instance {
                mesh: k,
                xf: [[1.0, 0.0, 0.0, x], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                mask: mask::GEOMETRY,
            });
        }
        let fov = (1.3 * n as f64 + 0.2) / 10.0;
        Self {
            meshes,
            instances,
            materials: mats,
            textures: Vec::new(),
            env: Some(Equirect { size: (8, 4), rgb: vec![[env; 3]; 32] }),
            sun: None,
            portal_skies: Vec::new(),
            camera: Camera {
                eye: [0.0, -10.0, 0.0],
                forward: [0.0, 1.0, 0.0],
                right: [1.0, 0.0, 0.0],
                up: [0.0, 0.0, 1.0],
                tan_half: fov * size.1 as f64 / size.0 as f64 * 1.0,
            },
            size,
            frames: Vec::new(),
            fps: 24.0,
            dropped: Vec::new(),
        }
    }
}

/// `M⁻¹`, rows: linear-sRGB → the three hats ([`hats`]).
pub fn hat_inverse() -> [[f32; 3]; 3] {
    let cols = [hats([1.0, 0.0, 0.0]), hats([0.0, 1.0, 0.0]), hats([0.0, 0.0, 1.0])];
    std::array::from_fn(|k| std::array::from_fn(|c| cols[c][k] as f32))
}
