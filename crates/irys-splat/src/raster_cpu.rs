//! The f64 reference of `raster.wgsl`: front-to-back alpha compositing of
//! depth-sorted screen-space splats at pixel centres, and its adjoint
//! (3DGS's backward: recover `T` by division, carry the colour behind).
//! Same response, cut-offs and termination as the device, so the device
//! is checked against it and it is checked against central differences.

use rayon::prelude::*;

use crate::project::{Grad2, Proj};

/// Splat response cut (3DGS): below it a splat is skipped.
pub const ALPHA_MIN: f64 = 1.0 / 255.0;
/// Opacity clamp.
pub const ALPHA_MAX: f64 = 0.99;
/// Transmittance at which a pixel stops.
pub const T_MIN: f64 = 1e-4;

/// The depth order the device uses: `(z, index)`.
pub fn order(p: &[Option<Proj>]) -> Vec<usize> {
    let mut o: Vec<usize> = (0..p.len()).filter(|&i| p[i].is_some()).collect();
    o.sort_by(|&a, &b| p[a].unwrap().colour[3].total_cmp(&p[b].unwrap().colour[3]).then(a.cmp(&b)));
    o
}

/// `(α, G, dx, dy)` of a splat at pixel `(px, py)`; `None` when it does not
/// respond. `G = exp(power)`.
#[inline]
pub fn response(p: &Proj, px: f64, py: f64) -> Option<(f64, f64, f64, f64)> {
    let dx = px - p.uv[0];
    let dy = py - p.uv[1];
    let [a, b, c] = p.conic;
    let power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy;
    if power > 0.0 {
        return None;
    }
    let g = power.exp();
    let al = (p.opacity * g).min(ALPHA_MAX);
    (al >= ALPHA_MIN).then_some((al, g, dx, dy))
}

/// One frame: per pixel rgb+depth, final transmittance, and how many of the
/// ordered list it consumed (`last`: the device's `n_contrib`).
pub struct Frame {
    pub colour: Vec<[f64; 4]>,
    pub trans: Vec<f64>,
    pub last: Vec<usize>,
}

pub fn forward(p: &[Option<Proj>], ord: &[usize], w: usize, h: usize, bg: [f64; 3]) -> Frame {
    let px: Vec<([f64; 4], f64, usize)> = (0..w * h)
        .into_par_iter()
        .map(|pix| {
            let (x, y) = ((pix % w) as f64, (pix / w) as f64);
            let mut t = 1.0;
            let mut c = [0.0; 4];
            let mut last = 0;
            for (k, &i) in ord.iter().enumerate() {
                let s = p[i].as_ref().unwrap();
                let Some((al, _, _, _)) = response(s, x, y) else { continue };
                let nt = t * (1.0 - al);
                if nt < T_MIN {
                    break;
                }
                for ch in 0..4 {
                    c[ch] += al * t * s.colour[ch];
                }
                t = nt;
                last = k + 1;
            }
            for ch in 0..3 {
                c[ch] += t * bg[ch];
            }
            (c, t, last)
        })
        .collect();
    Frame {
        colour: px.iter().map(|v| v.0).collect(),
        trans: px.iter().map(|v| v.1).collect(),
        last: px.iter().map(|v| v.2).collect(),
    }
}

/// The adjoint: `∂L/∂(uv, conic, opacity, colour)` per splat from
/// `∂L/∂colour` per pixel (`dl[pix]`, rgb + depth).
pub fn backward(p: &[Option<Proj>], ord: &[usize], f: &Frame, dl: &[[f64; 4]], w: usize, bg: [f64; 3]) -> Vec<Grad2> {
    let n = p.len();
    let parts: Vec<Vec<(usize, Grad2)>> = (0..f.colour.len())
        .into_par_iter()
        .map(|pix| {
            let (x, y) = ((pix % w) as f64, (pix / w) as f64);
            let mut t = f.trans[pix];
            let mut s = [bg[0], bg[1], bg[2], 0.0];
            let d = dl[pix];
            let mut out = Vec::new();
            for k in (0..f.last[pix]).rev() {
                let i = ord[k];
                let sp = p[i].as_ref().unwrap();
                let Some((al, g, dx, dy)) = response(sp, x, y) else { continue };
                let ti = t / (1.0 - al);
                let mut gr = Grad2::default();
                let mut dal = 0.0;
                for ch in 0..4 {
                    gr.colour[ch] = al * ti * d[ch];
                    dal += ti * (sp.colour[ch] - s[ch]) * d[ch];
                    s[ch] = al * sp.colour[ch] + (1.0 - al) * s[ch];
                }
                t = ti;
                if sp.opacity * g < ALPHA_MAX {
                    gr.opacity = g * dal;
                    let dpow = sp.opacity * g * dal;
                    let [a, b, c] = sp.conic;
                    gr.uv = [dpow * (a * dx + b * dy), dpow * (b * dx + c * dy)];
                    gr.conic = [-0.5 * dx * dx * dpow, -dx * dy * dpow, -0.5 * dy * dy * dpow];
                }
                out.push((i, gr));
            }
            out
        })
        .collect();
    let mut g = vec![Grad2::default(); n];
    for v in parts {
        for (i, d) in v {
            let o = &mut g[i];
            o.uv[0] += d.uv[0];
            o.uv[1] += d.uv[1];
            for c in 0..3 {
                o.conic[c] += d.conic[c];
            }
            o.opacity += d.opacity;
            for c in 0..4 {
                o.colour[c] += d.colour[c];
            }
        }
    }
    g
}
