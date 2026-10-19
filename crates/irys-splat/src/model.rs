//! The trained parameters, as the optimiser sees them: means, log scales,
//! unnormalised `[w, x, y, z]` quaternions, opacity logits, and degree-3 SH
//! (coefficient-major, `sh[i][k][ch]`; only `count(degree)` are live).
//! f64 on the host (the device only sees the projected f32 splats).

use std::io::Write;
use std::path::Path;

use irys_light::PlyCloud;

use crate::sh::{self, COEFS};

/// A splat cloud in raw (pre-activation) parameters.
#[derive(Debug, Clone, Default)]
pub struct Gaussians {
    /// Means, metres.
    pub means: Vec<[f64; 3]>,
    /// `ln σ` per axis.
    pub log_scales: Vec<[f64; 3]>,
    /// Unnormalised rotations `[w, x, y, z]`.
    pub quats: Vec<[f64; 4]>,
    /// `logit(opacity)`.
    pub opacity: Vec<f64>,
    /// SH, 16 coefficients × RGB per splat.
    pub sh: Vec<[[f64; 3]; COEFS]>,
}

/// One splat's gradient (or Adam moment), the same layout.
pub type Grads = Gaussians;

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

impl Gaussians {
    pub fn len(&self) -> usize {
        self.means.len()
    }

    pub fn is_empty(&self) -> bool {
        self.means.is_empty()
    }

    /// Zeros of the same shape.
    pub fn zeros(n: usize) -> Self {
        Self {
            means: vec![[0.0; 3]; n],
            log_scales: vec![[0.0; 3]; n],
            quats: vec![[0.0; 4]; n],
            opacity: vec![0.0; n],
            sh: vec![[[0.0; 3]; COEFS]; n],
        }
    }

    /// Append splat `i` of `other`.
    pub fn push_from(&mut self, other: &Self, i: usize) {
        self.means.push(other.means[i]);
        self.log_scales.push(other.log_scales[i]);
        self.quats.push(other.quats[i]);
        self.opacity.push(other.opacity[i]);
        self.sh.push(other.sh[i]);
    }

    /// Keep the splats where `keep` is true, in order.
    pub fn retain(&mut self, keep: &[bool]) {
        fn r<T: Copy>(v: &mut Vec<T>, keep: &[bool]) {
            let mut k = 0;
            v.retain(|_| {
                k += 1;
                keep[k - 1]
            });
        }
        r(&mut self.means, keep);
        r(&mut self.log_scales, keep);
        r(&mut self.quats, keep);
        r(&mut self.opacity, keep);
        r(&mut self.sh, keep);
    }

    /// Seed from points and sRGB colours: scale = the mean distance to the
    /// 3 nearest neighbours (a grid search), opacity 0.1, identity rotation.
    pub fn from_points(points: &[[f32; 3]], colours: &[[f32; 3]]) -> Self {
        let n = points.len();
        let nn = crate::knn::mean_nn_dist(points, 3);
        let mut g = Self::zeros(n);
        for i in 0..n {
            g.means[i] = points[i].map(|v| v as f64);
            let s = (nn[i].max(1e-4) as f64).ln();
            g.log_scales[i] = [s; 3];
            g.quats[i] = [1.0, 0.0, 0.0, 0.0];
            g.opacity[i] = logit(0.1);
            g.sh[i][0] = colours[i].map(|c| sh::dc_of(c as f64));
        }
        g
    }

    /// The activated cloud irys renders (`PlyCloud`), at SH `degree`.
    pub fn to_cloud(&self, degree: u32) -> PlyCloud {
        let s = sh::count(degree);
        let mut sh = Vec::with_capacity(self.len() * s);
        for c in &self.sh {
            sh.extend(c[..s].iter().map(|v| v.map(|x| x as f32)));
        }
        PlyCloud {
            positions: self.means.iter().map(|m| m.map(|v| v as f32)).collect(),
            scales: self.log_scales.iter().map(|l| l.map(|v| v.exp() as f32)).collect(),
            quats: self
                .quats
                .iter()
                .map(|q| {
                    let n = q.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
                    q.map(|v| (v / n) as f32)
                })
                .collect(),
            opacities: self.opacity.iter().map(|&o| sigmoid(o) as f32).collect(),
            sh,
            degree,
        }
    }

    /// Raw parameters back from a cloud (the inverse of [`Self::to_cloud`]).
    pub fn from_cloud(c: &PlyCloud) -> Self {
        let s = c.stride();
        let mut g = Self::zeros(c.len());
        for i in 0..c.len() {
            g.means[i] = c.positions[i].map(|v| v as f64);
            g.log_scales[i] = c.scales[i].map(|v| (v.max(1e-8) as f64).ln());
            g.quats[i] = c.quats[i].map(|v| v as f64);
            g.opacity[i] = logit(c.opacities[i] as f64);
            for k in 0..s.min(COEFS) {
                g.sh[i][k] = c.sh[i * s + k].map(|v| v as f64);
            }
        }
        g
    }

    /// Write the INRIA 3DGS layout, the attribute names and order of
    /// `ipse/maps/garage-perim/splat.ply`: `x y z nx ny nz f_dc_0..2`,
    /// then `f_rest_*` (channel-major, only when `degree > 0`), `opacity`
    /// (logit), `scale_0..2` (log), `rot_0..3` (`w x y z`, normalised).
    pub fn write_ply(&self, path: &Path, degree: u32) -> std::io::Result<()> {
        let rest = sh::count(degree) - 1;
        let mut h = format!("ply\nformat binary_little_endian 1.0\nelement vertex {}\n", self.len());
        for n in ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] {
            h += &format!("property float {n}\n");
        }
        for k in 0..3 * rest {
            h += &format!("property float f_rest_{k}\n");
        }
        for n in ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"] {
            h += &format!("property float {n}\n");
        }
        h += "end_header\n";
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        w.write_all(h.as_bytes())?;
        let mut row: Vec<f32> = Vec::with_capacity(17 + 3 * rest);
        for i in 0..self.len() {
            row.clear();
            let f = |v: f64| v as f32;
            row.extend(self.means[i].map(f));
            row.extend([0.0; 3]);
            row.extend(self.sh[i][0].map(f));
            for ch in 0..3 {
                for k in 1..=rest {
                    row.push(f(self.sh[i][k][ch]));
                }
            }
            row.push(f(self.opacity[i]));
            row.extend(self.log_scales[i].map(f));
            let q = self.quats[i];
            let n = q.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
            row.extend(q.map(|v| f(v / n)));
            for x in &row {
                w.write_all(&x.to_le_bytes())?;
            }
        }
        w.flush()
    }
}
