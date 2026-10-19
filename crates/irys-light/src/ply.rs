//! A loader for the standard 3D Gaussian splatting `.ply` (Kerbl et al.
//! 2023, the layout gsplat, nerfstudio and the INRIA trainer all write).
//!
//! One `vertex` element per splat, little-endian binary (ASCII is accepted
//! too), with these properties, in any order:
//!
//! | name | meaning | stored as | here |
//! |---|---|---|---|
//! | `x y z` | mean | world units | as is |
//! | `scale_0..2` | per-axis σ | **log** | `exp` |
//! | `rot_0..3` | rotation `[w, x, y, z]` | unnormalised | normalised |
//! | `opacity` | base α | **logit** | sigmoid |
//! | `f_dc_0..2` | SH degree 0, RGB | coefficient | as is |
//! | `f_rest_0..` | SH degrees 1–3 | **channel-major**: all R, then G, then B | transposed to coefficient-major |
//!
//! `nx ny nz` and anything else are skipped. The degree comes from the
//! `f_rest` count: 0, 9, 24 or 45 properties for degree 0–3.
//!
//! Colour convention (unchanged from the trainer): `c = 0.5 + Σ SH(d)`,
//! clamped at zero, with `d` the direction from the camera toward the
//! splat. A 3DGS cloud is trained against display-referred sRGB images, so
//! `c` is an sRGB value, not linear radiance; the device kernels decode it.

use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::splats::Splats;

/// A splat cloud as the file holds it, the activations applied.
#[derive(Debug, Clone, Default)]
pub struct PlyCloud {
    /// Means.
    pub positions: Vec<[f32; 3]>,
    /// Per-axis standard deviations (linear: `exp` of the file's).
    pub scales: Vec<[f32; 3]>,
    /// Unit rotations `[w, x, y, z]`.
    pub quats: Vec<[f32; 4]>,
    /// Base opacities in `0..=1` (sigmoid of the file's).
    pub opacities: Vec<f32>,
    /// Spherical harmonics, coefficient-major per splat: `sh[i * stride + k]`
    /// is coefficient `k` of splat `i`, RGB; `stride = (degree + 1)²`.
    pub sh: Vec<[f32; 3]>,
    /// SH degree, 0–3.
    pub degree: u32,
}

#[derive(Clone, Copy, PartialEq)]
enum Ty {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Ty {
    fn parse(s: &str) -> Result<Self, String> {
        Ok(match s {
            "char" | "int8" => Ty::I8,
            "uchar" | "uint8" => Ty::U8,
            "short" | "int16" => Ty::I16,
            "ushort" | "uint16" => Ty::U16,
            "int" | "int32" => Ty::I32,
            "uint" | "uint32" => Ty::U32,
            "float" | "float32" => Ty::F32,
            "double" | "float64" => Ty::F64,
            o => return Err(format!("ply: unknown property type {o}")),
        })
    }
    fn size(self) -> usize {
        match self {
            Ty::I8 | Ty::U8 => 1,
            Ty::I16 | Ty::U16 => 2,
            Ty::I32 | Ty::U32 | Ty::F32 => 4,
            Ty::F64 => 8,
        }
    }
    fn read(self, b: &[u8]) -> f32 {
        match self {
            Ty::I8 => b[0] as i8 as f32,
            Ty::U8 => b[0] as f32,
            Ty::I16 => i16::from_le_bytes([b[0], b[1]]) as f32,
            Ty::U16 => u16::from_le_bytes([b[0], b[1]]) as f32,
            Ty::I32 => i32::from_le_bytes(b[..4].try_into().unwrap()) as f32,
            Ty::U32 => u32::from_le_bytes(b[..4].try_into().unwrap()) as f32,
            Ty::F32 => f32::from_le_bytes(b[..4].try_into().unwrap()),
            Ty::F64 => f64::from_le_bytes(b[..8].try_into().unwrap()) as f32,
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl PlyCloud {
    /// Read a 3DGS `.ply`.
    pub fn load(path: &Path) -> Result<Self, String> {
        let f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::read(BufReader::new(f))
    }

    /// Read a 3DGS `.ply` from a stream.
    pub fn read<R: BufRead>(mut r: R) -> Result<Self, String> {
        let mut line = String::new();
        let mut format = String::new();
        let mut count = 0usize;
        let mut in_vertex = false;
        let mut props: Vec<(String, Ty)> = Vec::new();
        let mut first = true;
        loop {
            line.clear();
            if r.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
                return Err("ply: no end_header".into());
            }
            let l = line.trim();
            if first {
                if l != "ply" {
                    return Err("ply: not a ply file".into());
                }
                first = false;
                continue;
            }
            let mut w = l.split_whitespace();
            match w.next() {
                Some("format") => format = w.next().unwrap_or("").to_string(),
                Some("element") => {
                    let name = w.next().unwrap_or("");
                    in_vertex = name == "vertex";
                    if in_vertex {
                        count = w.next().and_then(|c| c.parse().ok()).ok_or("ply: bad vertex count")?;
                    } else if count == 0 || !props.is_empty() {
                        // elements after the vertices are not read
                    }
                }
                Some("property") if in_vertex => {
                    let t = w.next().ok_or("ply: property without a type")?;
                    if t == "list" {
                        return Err("ply: list properties on vertices are not supported".into());
                    }
                    let name = w.next().ok_or("ply: property without a name")?;
                    props.push((name.to_string(), Ty::parse(t)?));
                }
                Some("end_header") => break,
                _ => {}
            }
        }
        let idx = |n: &str| props.iter().position(|p| p.0 == n);
        let need = |n: &str| idx(n).ok_or_else(|| format!("ply: no `{n}` property"));
        let pos = [need("x")?, need("y")?, need("z")?];
        let sc = [need("scale_0")?, need("scale_1")?, need("scale_2")?];
        let rot = [need("rot_0")?, need("rot_1")?, need("rot_2")?, need("rot_3")?];
        let op = need("opacity")?;
        let dc = [need("f_dc_0")?, need("f_dc_1")?, need("f_dc_2")?];
        let mut rest = Vec::new();
        while let Some(i) = idx(&format!("f_rest_{}", rest.len())) {
            rest.push(i);
        }
        let degree = match rest.len() {
            0 => 0,
            9 => 1,
            24 => 2,
            45 => 3,
            n => return Err(format!("ply: {n} f_rest properties is not a whole SH degree")),
        };
        let stride = (degree as usize + 1).pow(2);
        let per = rest.len() / 3; // coefficients past DC, per channel

        let mut c = PlyCloud { degree, ..Default::default() };
        c.positions.reserve(count);
        c.scales.reserve(count);
        c.quats.reserve(count);
        c.opacities.reserve(count);
        c.sh.reserve(count * stride);
        let mut row = vec![0f32; props.len()];
        let push = |row: &[f32], c: &mut PlyCloud| {
            c.positions.push(pos.map(|i| row[i]));
            c.scales.push(sc.map(|i| row[i].exp()));
            let q = rot.map(|i| row[i]);
            let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            c.quats.push(if n > 0.0 && n.is_finite() { q.map(|x| x / n) } else { [1.0, 0.0, 0.0, 0.0] });
            c.opacities.push(sigmoid(row[op]));
            c.sh.push(dc.map(|i| row[i]));
            for k in 0..per {
                c.sh.push([row[rest[k]], row[rest[per + k]], row[rest[2 * per + k]]]);
            }
        };
        match format.as_str() {
            "binary_little_endian" => {
                let offs: Vec<usize> = props
                    .iter()
                    .scan(0, |o, p| {
                        let a = *o;
                        *o += p.1.size();
                        Some(a)
                    })
                    .collect();
                let stride_b: usize = props.iter().map(|p| p.1.size()).sum();
                let all_f32 = props.iter().all(|p| p.1 == Ty::F32);
                let mut buf = vec![0u8; stride_b * 4096];
                let mut left = count;
                while left > 0 {
                    let n = left.min(4096);
                    let b = &mut buf[..n * stride_b];
                    r.read_exact(b).map_err(|e| format!("ply: truncated ({e})"))?;
                    for v in b.chunks_exact(stride_b) {
                        if all_f32 {
                            for (k, x) in v.chunks_exact(4).enumerate() {
                                row[k] = f32::from_le_bytes([x[0], x[1], x[2], x[3]]);
                            }
                        } else {
                            for (k, p) in props.iter().enumerate() {
                                row[k] = p.1.read(&v[offs[k]..]);
                            }
                        }
                        push(&row, &mut c);
                    }
                    left -= n;
                }
            }
            "ascii" => {
                let mut text = String::new();
                r.read_to_string(&mut text).map_err(|e| e.to_string())?;
                let mut it = text.split_whitespace();
                for _ in 0..count {
                    for x in row.iter_mut() {
                        *x = it.next().and_then(|s| s.parse().ok()).ok_or("ply: truncated ascii")?;
                    }
                    push(&row, &mut c);
                }
            }
            o => return Err(format!("ply: format {o} is not supported")),
        }
        Ok(c)
    }

    /// How many splats.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// No splats.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// `(degree + 1)²`.
    pub fn stride(&self) -> usize {
        (self.degree as usize + 1).pow(2)
    }

    /// The CPU intersector's cloud ([`Splats`]).
    pub fn to_splats(&self) -> Splats {
        Splats::from_parts(&self.positions, &self.scales, &self.quats, &self.opacities, &self.sh)
    }

    /// Write a little-endian 3DGS `.ply` (the inverse of [`PlyCloud::read`]).
    pub fn write<W: std::io::Write>(&self, mut w: W) -> std::io::Result<()> {
        let per = self.stride() - 1;
        let mut h = format!("ply\nformat binary_little_endian 1.0\nelement vertex {}\n", self.len());
        for n in ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] {
            h += &format!("property float {n}\n");
        }
        for k in 0..3 * per {
            h += &format!("property float f_rest_{k}\n");
        }
        for n in ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"] {
            h += &format!("property float {n}\n");
        }
        h += "end_header\n";
        w.write_all(h.as_bytes())?;
        let s = self.stride();
        for i in 0..self.len() {
            let mut v: Vec<f32> = self.positions[i].to_vec();
            v.extend(self.sh[i * s]);
            for ch in 0..3 {
                for k in 1..s {
                    v.push(self.sh[i * s + k][ch]);
                }
            }
            let a = self.opacities[i].clamp(1e-7, 1.0 - 1e-7);
            v.push((a / (1.0 - a)).ln());
            v.extend(self.scales[i].map(f32::ln));
            v.extend(self.quats[i]);
            for x in v {
                w.write_all(&x.to_le_bytes())?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud(degree: u32) -> PlyCloud {
        let s = (degree as usize + 1).pow(2);
        let n = 5;
        PlyCloud {
            positions: (0..n).map(|i| [i as f32, 0.5 * i as f32, -1.0]).collect(),
            scales: (0..n).map(|i| [0.01 * (i + 1) as f32, 0.02, 0.03]).collect(),
            quats: (0..n).map(|_| [0.5f32, 0.5, -0.5, 0.5]).collect(),
            opacities: (0..n).map(|i| 0.1 + 0.15 * i as f32).collect(),
            sh: (0..n * s).map(|k| [k as f32 * 0.01, -(k as f32) * 0.02, 0.3]).collect(),
            degree,
        }
    }

    #[test]
    fn round_trip_every_degree() {
        for d in 0..=3 {
            let c = cloud(d);
            let mut b = Vec::new();
            c.write(&mut b).unwrap();
            let r = PlyCloud::read(&b[..]).unwrap();
            assert_eq!(r.degree, d);
            assert_eq!(r.len(), c.len());
            for i in 0..c.len() {
                for a in 0..3 {
                    assert!((r.positions[i][a] - c.positions[i][a]).abs() < 1e-6);
                    assert!((r.scales[i][a] / c.scales[i][a] - 1.0).abs() < 1e-5);
                }
                assert!((r.opacities[i] - c.opacities[i]).abs() < 1e-5);
                for a in 0..4 {
                    assert!((r.quats[i][a] - c.quats[i][a]).abs() < 1e-6);
                }
            }
            // coefficient-major SH survives the channel-major file layout
            assert_eq!(r.sh, c.sh);
        }
    }

    #[test]
    fn channel_major_rest_is_transposed() {
        // f_rest_0..8 = R coefficients 1..3 of degree 1, 9..17 would be G...
        let mut h = String::from("ply\nformat ascii 1.0\nelement vertex 1\n");
        for n in ["x", "y", "z", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3", "opacity"] {
            h += &format!("property float {n}\n");
        }
        for k in 0..3 {
            h += &format!("property float f_dc_{k}\n");
        }
        for k in 0..9 {
            h += &format!("property float f_rest_{k}\n");
        }
        h += "end_header\n0 0 0 0 0 0 1 0 0 0 0 1 2 3 10 11 12 20 21 22 30 31 32\n";
        let c = PlyCloud::read(h.as_bytes()).unwrap();
        assert_eq!(c.degree, 1);
        assert_eq!(c.sh, vec![[1.0, 2.0, 3.0], [10.0, 20.0, 30.0], [11.0, 21.0, 31.0], [12.0, 22.0, 32.0]]);
        assert_eq!(c.opacities[0], 0.5);
        assert_eq!(c.scales[0], [1.0; 3]);
    }

    #[test]
    fn the_garage_header_reads() {
        let p = Path::new("/Users/cam/Developer/ipse/maps/garage-perim/splat.ply");
        if !p.exists() {
            return;
        }
        let c = PlyCloud::load(p).unwrap();
        assert_eq!(c.len(), 274_499);
        assert_eq!(c.degree, 0);
        assert!(c.opacities.iter().all(|a| (0.0..=1.0).contains(a)));
    }
}
