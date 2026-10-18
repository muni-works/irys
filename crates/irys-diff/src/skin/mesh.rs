//! A textured triangle mesh: positions, normals, UVs and tangents per
//! vertex, plus the maps a skin needs, and what a hit on it looks like
//! (interpolated attributes, the normal map applied).
//!
//! The glTF loader ([`load`]) is behind the `assets` feature; everything
//! else is plain data so tests can build meshes by hand.

use irys_core::{Bvh, Hit, Point3, TriMesh, Vec3};

/// An RGB texture in linear `f32`, row 0 at the top (`v = 0` is the top
/// row, glTF's convention), sampled bilinearly with wrap.
#[derive(Debug, Clone)]
pub struct Texture {
    /// Width and height in texels.
    pub size: (usize, usize),
    /// `w · h` texels, row-major.
    pub texels: Vec<[f32; 3]>,
}

/// sRGB transfer function, decoded.
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

impl Texture {
    /// A constant texture.
    pub fn constant(c: [f32; 3]) -> Self {
        Self {
            size: (1, 1),
            texels: vec![c],
        }
    }

    /// From 8-bit RGB, decoding sRGB when `srgb` (colour maps) and mapping
    /// to `[0, 1]` otherwise (data maps: specular, normals).
    pub fn from_rgb8(size: (usize, usize), rgb: &[u8], srgb: bool) -> Self {
        assert_eq!(rgb.len(), size.0 * size.1 * 3);
        let texels = rgb
            .chunks_exact(3)
            .map(|p| {
                let f = |b: u8| {
                    let x = b as f32 / 255.0;
                    if srgb { srgb_to_linear(x) } else { x }
                };
                [f(p[0]), f(p[1]), f(p[2])]
            })
            .collect();
        Self { size, texels }
    }

    /// Bilinear lookup at `(u, v)`, wrapped. Texel centres at half-integers,
    /// so the GPU's `sample` (same arithmetic in f32) agrees.
    pub fn sample(&self, u: f64, v: f64) -> [f64; 3] {
        let (w, h) = self.size;
        let x = (u - u.floor()) * w as f64 - 0.5;
        let y = (v - v.floor()) * h as f64 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let wrap = |i: f64, n: usize| (i as i64).rem_euclid(n as i64) as usize;
        let (xa, xb) = (wrap(x0, w), wrap(x0 + 1.0, w));
        let (ya, yb) = (wrap(y0, h), wrap(y0 + 1.0, h));
        let t = |x: usize, y: usize| self.texels[y * w + x];
        let (a, b, c, d) = (t(xa, ya), t(xb, ya), t(xa, yb), t(xb, yb));
        std::array::from_fn(|k| {
            let top = a[k] as f64 * (1.0 - fx) + b[k] as f64 * fx;
            let bot = c[k] as f64 * (1.0 - fx) + d[k] as f64 * fx;
            top * (1.0 - fy) + bot * fy
        })
    }

    /// The mean texel.
    pub fn mean(&self) -> [f64; 3] {
        let mut m = [0.0; 3];
        for t in &self.texels {
            for k in 0..3 {
                m[k] += t[k] as f64;
            }
        }
        m.map(|x| x / self.texels.len() as f64)
    }
}

/// A triangle mesh with per-vertex shading attributes.
#[derive(Debug, Clone)]
pub struct Mesh {
    /// Positions.
    pub positions: Vec<Point3>,
    /// Unit shading normals, one per position.
    pub normals: Vec<Vec3>,
    /// Texture coordinates, one per position.
    pub uvs: Vec<[f64; 2]>,
    /// Tangents (`xyz`) and bitangent sign (`w`), one per position:
    /// `B = w · N × T`, glTF's convention.
    pub tangents: Vec<[f64; 4]>,
    /// Corner indices.
    pub indices: Vec<u32>,
}

impl Mesh {
    /// Per-vertex tangents from positions and UVs (Lengyel's method:
    /// accumulate each triangle's `∂p/∂u`, `∂p/∂v`, Gram–Schmidt against the
    /// normal), the stand-in for MikkTSpace a normal map baked against
    /// smooth UVs needs to within its own blur.
    pub fn compute_tangents(&mut self) {
        let n = self.positions.len();
        let mut t = vec![Vec3::zeros(); n];
        let mut b = vec![Vec3::zeros(); n];
        for c in self.indices.chunks_exact(3) {
            let [i0, i1, i2] = [c[0] as usize, c[1] as usize, c[2] as usize];
            let (p0, p1, p2) = (self.positions[i0], self.positions[i1], self.positions[i2]);
            let (w0, w1, w2) = (self.uvs[i0], self.uvs[i1], self.uvs[i2]);
            let (e1, e2) = (p1 - p0, p2 - p0);
            let (du1, dv1) = (w1[0] - w0[0], w1[1] - w0[1]);
            let (du2, dv2) = (w2[0] - w0[0], w2[1] - w0[1]);
            let det = du1 * dv2 - du2 * dv1;
            if det.abs() < 1e-14 {
                continue;
            }
            let r = 1.0 / det;
            let sdir = (e1 * dv2 - e2 * dv1) * r;
            let tdir = (e2 * du1 - e1 * du2) * r;
            for i in [i0, i1, i2] {
                t[i] += sdir;
                b[i] += tdir;
            }
        }
        self.tangents = (0..n)
            .map(|i| {
                let nn = self.normals[i];
                let mut tt = t[i] - nn * nn.dot(&t[i]);
                if tt.norm() < 1e-12 {
                    // No UV gradient: any perpendicular will do.
                    tt = basis(nn).0;
                }
                let tt = tt.normalize();
                let w = if nn.cross(&tt).dot(&b[i]) < 0.0 {
                    -1.0
                } else {
                    1.0
                };
                [tt.x, tt.y, tt.z, w]
            })
            .collect();
    }

    /// Per-vertex normals as area-weighted face normals.
    pub fn compute_normals(&mut self) {
        let mut n = vec![Vec3::zeros(); self.positions.len()];
        for c in self.indices.chunks_exact(3) {
            let [i0, i1, i2] = [c[0] as usize, c[1] as usize, c[2] as usize];
            let f = (self.positions[i1] - self.positions[i0])
                .cross(&(self.positions[i2] - self.positions[i0]));
            for i in [i0, i1, i2] {
                n[i] += f;
            }
        }
        self.normals = n.into_iter().map(|v| v.normalize()).collect();
    }

    /// Append `other`, offsetting its indices.
    pub fn append(&mut self, other: &Mesh) {
        let base = self.positions.len() as u32;
        self.positions.extend_from_slice(&other.positions);
        self.normals.extend_from_slice(&other.normals);
        self.uvs.extend_from_slice(&other.uvs);
        self.tangents.extend_from_slice(&other.tangents);
        self.indices.extend(other.indices.iter().map(|i| i + base));
    }

    /// Axis-aligned bounds `(min, max)`.
    pub fn bounds(&self) -> (Point3, Point3) {
        let mut lo = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut hi = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in &self.positions {
            lo = Point3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
            hi = Point3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
        }
        (lo, hi)
    }

    /// Apply `f` to every position and the matching linear map `g` to
    /// normals and tangents (renormalised).
    pub fn transform(&mut self, f: impl Fn(Point3) -> Point3, g: impl Fn(Vec3) -> Vec3) {
        for p in &mut self.positions {
            *p = f(*p);
        }
        for n in &mut self.normals {
            *n = g(*n).normalize();
        }
        for t in &mut self.tangents {
            let v = g(Vec3::new(t[0], t[1], t[2])).normalize();
            *t = [v.x, v.y, v.z, t[3]];
        }
    }

    /// A UV sphere: `lat × lon` quads, `u` around, `v` from the north pole
    /// (`+z`) down, tangents along `+u`.
    pub fn uv_sphere(center: Point3, radius: f64, lat: usize, lon: usize) -> Mesh {
        let mut m = Mesh {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            tangents: Vec::new(),
            indices: Vec::new(),
        };
        for i in 0..=lat {
            let th = std::f64::consts::PI * i as f64 / lat as f64;
            for j in 0..=lon {
                let ph = 2.0 * std::f64::consts::PI * j as f64 / lon as f64;
                let n = Vec3::new(th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos());
                m.positions.push(center + n * radius);
                m.normals.push(n);
                m.uvs.push([j as f64 / lon as f64, i as f64 / lat as f64]);
                m.tangents.push([-ph.sin(), ph.cos(), 0.0, 1.0]);
            }
        }
        let v = |i: usize, j: usize| (i * (lon + 1) + j) as u32;
        for i in 0..lat {
            for j in 0..lon {
                let (a, b, c, d) = (v(i, j), v(i + 1, j), v(i + 1, j + 1), v(i, j + 1));
                m.indices.extend([a, b, c, a, c, d]);
            }
        }
        m
    }

    /// The traceable triangles, positions only. Degenerate triangles are
    /// dropped by [`TriMesh::new`]; hits index [`TriMesh::triangles`], whose
    /// corners index this mesh's attributes.
    pub fn trimesh(&self) -> TriMesh {
        TriMesh::new(self.positions.clone(), Vec::new(), &self.indices)
    }
}

/// A local frame around `n` (Duff et al. 2017), `(t, b)`.
pub fn basis(n: Vec3) -> (Vec3, Vec3) {
    let s = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    (
        Vec3::new(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
        Vec3::new(b, s + n.y * n.y * a, -n.y),
    )
}

/// What a hit on a [`Mesh`] looks like once its attributes are interpolated.
#[derive(Debug, Clone, Copy)]
pub struct Shading {
    /// Geometric (face) normal, unit, the mesh's winding.
    pub ng: Vec3,
    /// Interpolated shading normal, unit.
    pub ns: Vec3,
    /// Interpolated tangent (not yet orthogonalised) and sign.
    pub tangent: [f64; 4],
    /// Texture coordinates.
    pub uv: [f64; 2],
}

/// Interpolate `mesh`'s attributes at a hit on `bvh` (built over
/// [`Mesh::trimesh`]).
pub fn shade(mesh: &Mesh, bvh: &Bvh<TriMesh>, hit: &Hit) -> Shading {
    let tri = bvh.geometry().triangles()[hit.prim as usize];
    let (u, v) = (hit.uv.x, hit.uv.y);
    let w = [1.0 - u - v, u, v];
    let [i0, i1, i2] = tri.map(|i| i as usize);
    let ns = mesh.normals[i0] * w[0] + mesh.normals[i1] * w[1] + mesh.normals[i2] * w[2];
    let uv = std::array::from_fn(|k| {
        mesh.uvs[i0][k] * w[0] + mesh.uvs[i1][k] * w[1] + mesh.uvs[i2][k] * w[2]
    });
    let tangent = std::array::from_fn(|k| {
        if k == 3 {
            mesh.tangents[i0][3]
        } else {
            mesh.tangents[i0][k] * w[0] + mesh.tangents[i1][k] * w[1] + mesh.tangents[i2][k] * w[2]
        }
    });
    let ng = *hit.normal.as_ref();
    let ns = if ns.norm() > 1e-12 {
        ns.normalize()
    } else {
        ng
    };
    Shading {
        ng,
        ns,
        tangent,
        uv,
    }
}

/// The normal map applied: `ns` bent by tangent-space normal `rgb`
/// (`[0, 1]` per channel, OpenGL convention: green up in the image),
/// blended toward `ns` by `strength`.
pub fn bump(s: &Shading, rgb: [f64; 3], strength: f64) -> Vec3 {
    let n = s.ns;
    let t0 = Vec3::new(s.tangent[0], s.tangent[1], s.tangent[2]);
    let t = t0 - n * n.dot(&t0);
    if t.norm() < 1e-9 {
        return n;
    }
    let t = t.normalize();
    let b = n.cross(&t) * s.tangent[3];
    // Green is image-up (OpenGL), and `b` points along `+v`, which is
    // image-down in glTF's UVs: green enters negated. (The map's own
    // evidence: `(r, −g)` is 2.7× closer to curl-free than `(r, g)` as a
    // height gradient in image coordinates; `skin_head.rs` checks it.)
    let d = rgb.map(|c| 2.0 * c - 1.0);
    let m = [
        d[0] * strength,
        -d[1] * strength,
        1.0 - strength + d[2].max(0.0) * strength,
    ];
    let out = t * m[0] + b * m[1] + n * m[2];
    if out.norm() < 1e-9 {
        n
    } else {
        out.normalize()
    }
}

/// The head and its maps as loaded from disk.
#[derive(Debug, Clone)]
pub struct Asset {
    /// The mesh, in the file's units and axes.
    pub mesh: Mesh,
    /// Base colour (linear).
    pub albedo: Texture,
    /// Specular intensity (data, `[0, 1]`).
    pub spec: Texture,
    /// Tangent-space normal map (data, `[0, 1]`).
    pub normal: Texture,
}

/// Where the Lee Perry-Smith asset lives: `$IRYS_ASSETS/lee-perry-smith`,
/// default `/Users/cam/Developer/irys-assets/lee-perry-smith`. The asset is
/// CC BY 3.0 Infinite-Realities (ir-ltd.net) and is kept out of the repo.
pub fn asset_dir() -> std::path::PathBuf {
    let root = std::env::var_os("IRYS_ASSETS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "/Users/cam/Developer/irys-assets".into());
    root.join("lee-perry-smith")
}

/// Load a JPEG/PNG map.
#[cfg(feature = "assets")]
pub fn load_texture(path: &std::path::Path, srgb: bool) -> Result<Texture, String> {
    let img = image::open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .to_rgb8();
    let (w, h) = img.dimensions();
    Ok(Texture::from_rgb8(
        (w as usize, h as usize),
        img.as_raw(),
        srgb,
    ))
}

/// Load the first mesh primitive of a `.glb` (positions, normals,
/// `TEXCOORD_0`, tangents if present, else computed), node transforms
/// ignored.
#[cfg(feature = "assets")]
pub fn load_glb(path: &std::path::Path) -> Result<Mesh, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let glb = gltf::Gltf::from_slice(&bytes).map_err(|e| e.to_string())?;
    let blob = glb.blob.as_deref().ok_or("glb without a binary chunk")?;
    let mesh = glb.meshes().next().ok_or("no mesh")?;
    let prim = mesh.primitives().next().ok_or("no primitive")?;
    let reader = prim.reader(|b| match b.source() {
        gltf::buffer::Source::Bin => Some(blob),
        gltf::buffer::Source::Uri(_) => None,
    });
    let positions: Vec<Point3> = reader
        .read_positions()
        .ok_or("no positions")?
        .map(|p| Point3::new(p[0] as f64, p[1] as f64, p[2] as f64))
        .collect();
    let normals: Vec<Vec3> = reader
        .read_normals()
        .ok_or("no normals")?
        .map(|n| Vec3::new(n[0] as f64, n[1] as f64, n[2] as f64).normalize())
        .collect();
    let uvs: Vec<[f64; 2]> = reader
        .read_tex_coords(0)
        .ok_or("no TEXCOORD_0")?
        .into_f32()
        .map(|t| [t[0] as f64, t[1] as f64])
        .collect();
    let indices: Vec<u32> = reader
        .read_indices()
        .ok_or("no indices")?
        .into_u32()
        .collect();
    let tangents: Vec<[f64; 4]> = reader
        .read_tangents()
        .map(|it| it.map(|t| t.map(|x| x as f64)).collect())
        .unwrap_or_default();
    let mut m = Mesh {
        positions,
        normals,
        uvs,
        tangents,
        indices,
    };
    if m.tangents.len() != m.positions.len() {
        m.compute_tangents();
    }
    Ok(m)
}

/// Load the Lee Perry-Smith head and its three maps from `dir`
/// ([`asset_dir`]).
#[cfg(feature = "assets")]
pub fn load(dir: &std::path::Path) -> Result<Asset, String> {
    Ok(Asset {
        mesh: load_glb(&dir.join("LeePerrySmith.glb"))?,
        albedo: load_texture(&dir.join("Map-COL.jpg"), true)?,
        spec: load_texture(&dir.join("Map-SPEC.jpg"), false)?,
        normal: load_texture(&dir.join("Infinite-Level_02_Tangent_SmoothUV.jpg"), false)?,
    })
}
