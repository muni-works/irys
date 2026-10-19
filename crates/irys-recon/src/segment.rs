//! v1: segmentation on a plain backdrop, with a prior from the tag plane.
//!
//! v0 subtracted a rendered empty studio. A photo has no empty plate, but
//! it has the table: every pixel's ray meets the tag plane at a known
//! point, so the backdrop can be modelled where it must be.
//!
//! - **Where to look** ([`Region`]): table points within a radius of the
//!   tag, outside the tag's own square plus a margin (the print is not the
//!   backdrop), and below the horizon.
//! - **The backdrop's colour**: chromaticity `rgb / Σrgb` is nearly
//!   invariant to shading, so the backdrop is one robust chromaticity (the
//!   median over the region; the part is a minority there) with a robust
//!   spread. Luminance is not: shadows and light fall-off darken the
//!   table smoothly, so it is a robust quadratic in table coordinates
//!   (IRLS with Huber weights).
//! - **A pixel is the part** if its chromaticity is far from the
//!   backdrop's, or if it is brighter than the table model allows, or much
//!   darker than a shadow could make it. A dark key shadow on the table
//!   stays backdrop (same chromaticity, darker).
//! - **Edges**: a pixel whose luminance gradient is strong and which
//!   touches the part becomes part (closing a boundary the colour test
//!   leaves ragged), then a 3 × 3 majority filter; components smaller than
//!   a few pixels are dropped. Holes in the mask are kept: a through-hole
//!   shows the table, and that is signal.
//!
//! No dependency beyond the image itself. A part the colour of its
//! backdrop (white PLA on white paper) is the case this cannot do; the kit
//! asks for contrast.

use crate::photo::Camera;

/// Which table points count as backdrop.
#[derive(Debug, Clone, Copy)]
pub struct Region {
    /// Half-side of the excluded square about the tag's centre, mm: the
    /// kit's tag sheet is trimmed to 90 mm, so 50 covers it with a margin.
    pub tag_half: f64,
    /// Radius about the tag's centre, mm.
    pub radius: f64,
}

impl Default for Region {
    fn default() -> Self {
        Self { tag_half: 50.0, radius: 260.0 }
    }
}

/// A backdrop model.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Backdrop {
    /// Median chromaticity `(r, g)` (`b = 1 − r − g`).
    pub chroma: [f64; 2],
    /// Robust spread of the chromaticity distance.
    pub chroma_sigma: f64,
    /// Luminance quadratic in table mm: `c0 + c1 x + c2 y + c3 x² + c4 xy + c5 y²`.
    pub lum: [f64; 6],
    /// Robust relative spread of the luminance.
    pub lum_sigma: f64,
}

fn lum(c: [f32; 3]) -> f64 {
    (c[0] as f64 + c[1] as f64 + c[2] as f64) / 3.0
}

fn chroma(c: [f32; 3]) -> [f64; 2] {
    let s = (c[0] + c[1] + c[2]) as f64 + 1e-4;
    [c[0] as f64 / s, c[1] as f64 / s]
}

fn quad(x: f64, y: f64) -> [f64; 6] {
    let (x, y) = (x / 100.0, y / 100.0);
    [1.0, x, y, x * x, x * y, y * y]
}

/// Per pixel of `cam`'s film: the table point, or `None` (above the
/// horizon, or in the tag's square, or past the radius).
pub fn table_points(cam: &Camera, region: &Region) -> Vec<Option<(f64, f64)>> {
    let (w, h) = cam.size;
    (0..w * h)
        .map(|p| {
            let q = cam.on_table((p % w) as f64 + 0.5, (p / w) as f64 + 0.5)?;
            let inside_tag = q.0.abs() < region.tag_half && q.1.abs() < region.tag_half;
            (!inside_tag && q.0.hypot(q.1) < region.radius).then_some(q)
        })
        .collect()
}

/// Fit the backdrop from the pixels of `rgb` (linear, `cam`'s film) whose
/// table point is in `region`.
pub fn backdrop(rgb: &[[f32; 3]], cam: &Camera, region: &Region) -> Option<Backdrop> {
    let pts = table_points(cam, region);
    let idx: Vec<usize> = (0..rgb.len()).filter(|&p| pts[p].is_some() && lum(rgb[p]) > 1e-3).collect();
    if idx.len() < 50 {
        return None;
    }
    let med = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let cr = med(idx.iter().map(|&p| chroma(rgb[p])[0]).collect());
    let cg = med(idx.iter().map(|&p| chroma(rgb[p])[1]).collect());
    let dist: Vec<f64> = idx.iter().map(|&p| {
        let c = chroma(rgb[p]);
        (c[0] - cr).hypot(c[1] - cg)
    }).collect();
    let chroma_sigma = (1.4826 * med(dist.clone())).max(2e-3);
    // Luminance: IRLS on the pixels whose chromaticity is the backdrop's.
    let sel: Vec<usize> = idx.iter().zip(&dist).filter(|(_, d)| **d < 3.0 * chroma_sigma).map(|(p, _)| *p).collect();
    let mut wts = vec![1.0; sel.len()];
    let mut c = [0.0; 6];
    let mut s = 0.1;
    for _ in 0..8 {
        let mut a = vec![0.0; 36];
        let mut g = vec![0.0; 6];
        for (k, &p) in sel.iter().enumerate() {
            let (x, y) = pts[p].unwrap();
            let f = quad(x, y);
            let l = lum(rgb[p]);
            for i in 0..6 {
                g[i] += wts[k] * f[i] * l;
                for j in 0..6 {
                    a[i * 6 + j] += wts[k] * f[i] * f[j];
                }
            }
        }
        for i in 0..6 {
            a[i * 6 + i] += 1e-9;
        }
        let x = crate::fit::solve(&a, &g, 6)?;
        c.copy_from_slice(&x);
        let res: Vec<f64> = sel.iter().map(|&p| {
            let (x, y) = pts[p].unwrap();
            let m: f64 = quad(x, y).iter().zip(&c).map(|(a, b)| a * b).sum();
            (lum(rgb[p]) - m) / m.abs().max(1e-4)
        }).collect();
        s = (1.4826 * med(res.iter().map(|r| r.abs()).collect())).max(5e-3);
        for (k, r) in res.iter().enumerate() {
            wts[k] = crate::look::huber_w(r / s, 1.5);
        }
    }
    Some(Backdrop { chroma: [cr, cg], chroma_sigma, lum: c, lum_sigma: s })
}

/// The part's mask on `cam`'s film (1 = part).
pub fn mask(rgb: &[[f32; 3]], cam: &Camera, bd: &Backdrop) -> Vec<f64> {
    let (w, h) = cam.size;
    let mut m = vec![0.0; w * h];
    let mut l = vec![0.0; w * h];
    for p in 0..w * h {
        l[p] = lum(rgb[p]);
        let Some(q) = cam.on_table((p % w) as f64 + 0.5, (p / w) as f64 + 0.5) else {
            continue;
        };
        let c = chroma(rgb[p]);
        let dc = (c[0] - bd.chroma[0]).hypot(c[1] - bd.chroma[1]) / bd.chroma_sigma;
        let lt: f64 = quad(q.0, q.1).iter().zip(&bd.lum).map(|(a, b)| a * b).sum::<f64>().max(1e-4);
        let ratio = l[p] / lt;
        let bright = ratio > 1.0 + 5.0 * bd.lum_sigma;
        // A key shadow darkens the table by its share of the light; more
        // than ~85 % darker is not the table.
        let dark = ratio < 0.15;
        m[p] = (dc > 6.0 || bright || dark) as u8 as f64;
    }
    // Edges touching the part join it.
    let grad = |x: usize, y: usize| -> f64 {
        let at = |x: i64, y: i64| l[(y.clamp(0, h as i64 - 1) as usize) * w + x.clamp(0, w as i64 - 1) as usize];
        let (x, y) = (x as i64, y as i64);
        let gx = at(x + 1, y) - at(x - 1, y);
        let gy = at(x, y + 1) - at(x, y - 1);
        gx.hypot(gy) / (2.0 * at(x, y).max(0.02))
    };
    let mut m2 = m.clone();
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let p = y * w + x;
            if m[p] == 0.0 && grad(x, y) > 0.25 && (m[p - 1] + m[p + 1] + m[p - w] + m[p + w]) > 0.0 {
                m2[p] = 1.0;
            }
        }
    }
    // 3 × 3 majority.
    let mut m3 = m2.clone();
    for y in 0..h {
        for x in 0..w {
            let mut s = 0.0;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (xx, yy) = ((x as i64 + dx).clamp(0, w as i64 - 1), (y as i64 + dy).clamp(0, h as i64 - 1));
                    s += m2[yy as usize * w + xx as usize];
                }
            }
            m3[y * w + x] = (s >= 5.0) as u8 as f64;
        }
    }
    m3
}

/// Connected components of `m` (4-neighbour): `(pixels, bbox [x0, y0, x1, y1])`, largest first.
pub fn components(m: &[f64], (w, h): (usize, usize)) -> Vec<(Vec<usize>, [usize; 4])> {
    let mut seen = vec![false; w * h];
    let mut out = Vec::new();
    for p in 0..w * h {
        if seen[p] || m[p] < 0.5 {
            continue;
        }
        let mut stack = vec![p];
        seen[p] = true;
        let mut px = Vec::new();
        let mut bb = [w, h, 0, 0];
        while let Some(q) = stack.pop() {
            px.push(q);
            let (x, y) = (q % w, q / w);
            bb = [bb[0].min(x), bb[1].min(y), bb[2].max(x), bb[3].max(y)];
            let mut nb = Vec::new();
            if x > 0 { nb.push(q - 1) }
            if x + 1 < w { nb.push(q + 1) }
            if y > 0 { nb.push(q - w) }
            if y + 1 < h { nb.push(q + w) }
            for n in nb {
                if !seen[n] && m[n] >= 0.5 {
                    seen[n] = true;
                    stack.push(n);
                }
            }
        }
        out.push((px, bb));
    }
    out.sort_by_key(|c| std::cmp::Reverse(c.0.len()));
    out
}

/// Keep only the components of `m` with at least `min` pixels.
pub fn drop_specks(m: &mut [f64], size: (usize, usize), min: usize) {
    for (px, _) in components(m, size) {
        if px.len() < min {
            for p in px {
                m[p] = 0.0;
            }
        }
    }
}
