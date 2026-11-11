//! tag36h11 AprilTag detection, pure Rust.
//!
//! One detector for every consumer (irys-recon's photo import, the ipse
//! capture tools): `kornia-apriltag` finds candidate quads, and everything
//! after that is here.
//!
//! - **Both polarities.** Normal and negative prints (black and white
//!   swapped) are both searched, every time, and merged.
//! - **Every decimation.** kornia's quad finder works at one decimation per
//!   run; a tag can be too small for one level and too large (or too
//!   fragmented by its own cells) for another. All of [`DECIMATIONS`] run
//!   and their tags are merged.
//! - **Small tags.** When nothing is found, the image is searched again at
//!   2× (bilinear) and the corners are mapped back.
//! - **Own decode.** Each quad is decoded by sampling the 36 bits under all
//!   eight assignments of its corners to the tag's (four rotations, both
//!   windings), so no corner-order convention of the quad finder is trusted.
//! - **Sub-pixel corners**, refined on linear light ([`detect_refined`]).
//! - **Any id.** Every tag36h11 id is reported.
//!
//! Corners come back in [`CORNER_ORDER`]: top-left, top-right,
//! bottom-right, bottom-left in the tag's **own** frame (row 0 is the top),
//! whichever way it faces in the image.
//!
//! The tag frame used by [`corners_mm`] and [`white_at`]: origin at the
//! tag's centre, `x` along its columns, `y` up its rows (towards row 0),
//! millimetres; `tag_mm` is the side of the black square (8 cells).

use kornia_apriltag::family::{TagFamily, TagFamilyKind};
use kornia_apriltag::{AprilTagDecoder, DecodeTagsConfig};
use kornia_image::{Image, ImageSize};

/// The order of a [`Tag`]'s corners, in the tag's own frame.
pub const CORNER_ORDER: [&str; 4] = ["top-left", "top-right", "bottom-right", "bottom-left"];

/// The decimations (kornia `downscale_factor`) every search runs.
pub const DECIMATIONS: [usize; 6] = [1, 2, 3, 4, 6, 8];

/// Most bit errors a reported tag may have.
pub const MAX_HAMMING: u32 = 2;

/// One detected tag.
#[derive(Debug, Clone, PartialEq)]
pub struct Tag {
    /// tag36h11 id.
    pub id: usize,
    /// Bit errors of its code.
    pub hamming: u32,
    /// Whether it was found inverted (a negative print).
    pub negative: bool,
    /// The black square's corners in pixels (`x` right, `y` down, pixel
    /// centres at `i + 0.5`), in [`CORNER_ORDER`].
    pub corners: [[f64; 2]; 4],
}

impl Tag {
    /// Centre of the corners, pixels.
    pub fn centre(&self) -> [f64; 2] {
        let c = &self.corners;
        [(c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4.0, (c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4.0]
    }

    /// Mean side, pixels.
    pub fn side(&self) -> f64 {
        let c = &self.corners;
        (0..4).map(|i| (c[i][0] - c[(i + 1) % 4][0]).hypot(c[i][1] - c[(i + 1) % 4][1])).sum::<f64>() / 4.0
    }
}

/// A grey image, `f32` in `[0, 1]`.
#[derive(Clone, Debug)]
pub struct Grey {
    /// Width, height.
    pub size: (usize, usize),
    /// Pixels, row-major.
    pub px: Vec<f32>,
}

impl Grey {
    /// From 8-bit luma.
    pub fn from_luma8(w: usize, h: usize, y: &[u8]) -> Self {
        Grey { size: (w, h), px: y.iter().map(|&v| v as f32 / 255.0).collect() }
    }

    /// `1 − v`.
    pub fn inverted(&self) -> Self {
        Grey { size: self.size, px: self.px.iter().map(|v| 1.0 - v).collect() }
    }

    /// Bilinear sample at pixel coordinates (centres at `i + 0.5`).
    pub fn at(&self, x: f64, y: f64) -> f64 {
        let (w, h) = self.size;
        let (fx, fy) = (x - 0.5, y - 0.5);
        let (x0, y0) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - x0, fy - y0);
        let g = |i: f64, j: f64| {
            let (i, j) = (i.clamp(0.0, w as f64 - 1.0) as usize, j.clamp(0.0, h as f64 - 1.0) as usize);
            self.px[j * w + i] as f64
        };
        (1.0 - ty) * ((1.0 - tx) * g(x0, y0) + tx * g(x0 + 1.0, y0)) + ty * ((1.0 - tx) * g(x0, y0 + 1.0) + tx * g(x0 + 1.0, y0 + 1.0))
    }

    /// 2× bilinear upsample (pixel `u` of the result is at `(u + 0.5) / 2`
    /// here).
    fn upsample2(&self) -> Grey {
        let (w, h) = self.size;
        let (w2, h2) = (2 * w, 2 * h);
        let mut px = vec![0f32; w2 * h2];
        for v in 0..h2 {
            for u in 0..w2 {
                px[v * w2 + u] = self.at((u as f64 + 0.5) * 0.5, (v as f64 + 0.5) * 0.5) as f32;
            }
        }
        Grey { size: (w2, h2), px }
    }
}

fn family() -> TagFamily {
    TagFamily::tag36_h11().expect("tag36h11")
}

/// tag36h11's 10 × 10 cell image of `id`, row-major from the top-left:
/// `true` is white. The outer ring is the white quiet zone, the next ring
/// the black border, the inner 6 × 6 the code (AprilRobotics'
/// `apriltag_to_image`). `None` past the last id.
pub fn cells(id: usize) -> Option<[[bool; 10]; 10]> {
    let fam = family();
    let code = *fam.code_data.get(id)? as u64;
    let mut c = [[false; 10]; 10];
    for i in 0..10 {
        c[0][i] = true;
        c[9][i] = true;
        c[i][0] = true;
        c[i][9] = true;
    }
    for i in 0..fam.nbits {
        if code >> (fam.nbits - i - 1) & 1 == 1 {
            c[fam.bit_y[i] as usize + 1][fam.bit_x[i] as usize + 1] = true;
        }
    }
    Some(c)
}

/// The number of tag36h11 ids.
pub fn id_count() -> usize {
    family().code_data.len()
}

/// Whether the printed tag is white at tag-frame `(x, y)` mm (`None` off
/// the 10-cell print).
pub fn white_at(cells: &[[bool; 10]; 10], tag_mm: f64, x: f64, y: f64) -> Option<bool> {
    let cell = tag_mm / 8.0;
    let (c, r) = ((x / cell + 5.0).floor(), (5.0 - y / cell).floor());
    if !(0.0..10.0).contains(&c) || !(0.0..10.0).contains(&r) {
        return None;
    }
    Some(cells[r as usize][c as usize])
}

/// The black square's corners in the tag frame (mm, `y` up), in
/// [`CORNER_ORDER`]: `(−,+), (+,+), (+,−), (−,−)`.
pub fn corners_mm(tag_mm: f64) -> [[f64; 2]; 4] {
    let a = tag_mm / 2.0;
    [[-a, a], [a, a], [a, -a], [-a, -a]]
}

/// The homography mapping `src[i]` to `dst[i]` (four points), row-major 3 × 3.
pub fn homography(src: &[[f64; 2]; 4], dst: &[[f64; 2]; 4]) -> Option<[f64; 9]> {
    let mut a = [[0.0f64; 9]; 8];
    for i in 0..4 {
        let ([x, y], [u, v]) = (src[i], dst[i]);
        a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
        a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
    }
    // Gaussian elimination with partial pivoting on the 8 × 8 system.
    for c in 0..8 {
        let p = (c..8).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))?;
        if a[p][c].abs() < 1e-12 {
            return None;
        }
        a.swap(c, p);
        for r in 0..8 {
            if r != c {
                let k = a[r][c] / a[c][c];
                for k2 in c..9 {
                    a[r][k2] -= k * a[c][k2];
                }
            }
        }
    }
    let mut h = [0.0; 9];
    for i in 0..8 {
        h[i] = a[i][8] / a[i][i];
    }
    h[8] = 1.0;
    Some(h)
}

/// Apply homography `h` to `p`.
pub fn apply(h: &[f64; 9], p: [f64; 2]) -> [f64; 2] {
    let w = h[6] * p[0] + h[7] * p[1] + h[8];
    [(h[0] * p[0] + h[1] * p[1] + h[2]) / w, (h[3] * p[0] + h[4] * p[1] + h[5]) / w]
}

/// The unit tag frame the detector works in: the black square is 8 × 8.
const UNIT_MM: f64 = 8.0;

/// Decode the tag whose black square has corners `quad` (in
/// [`CORNER_ORDER`]): `(id, hamming)` of the nearest code.
fn decode_with(img: &Grey, quad: &[[f64; 2]; 4], fam: &TagFamily) -> Option<(usize, u32)> {
    let h = homography(&corners_mm(UNIT_MM), quad)?;
    // Cell (c, r) centre, sampled as the mean of a 3 × 3 grid over the
    // middle half of the cell: a single bilinear tap at the centre of a
    // 40-pixel cell sees JPEG ringing and paper texture.
    let at = |c: usize, r: usize| {
        let mut s = 0.0;
        for j in [-0.25, 0.0, 0.25] {
            for i in [-0.25, 0.0, 0.25] {
                let (x, y) = (c as f64 - 5.0 + 0.5 + i, 5.0 - r as f64 - 0.5 + j);
                let p = apply(&h, [x, y]);
                s += img.at(p[0], p[1]);
            }
        }
        s / 9.0
    };
    // The black border ring (always black) and the code's own white bits
    // set the threshold; the quiet zone outside the border is not used, so
    // a print trimmed at the border, or a negative print whose quiet zone
    // is the paper, decodes the same.
    let mut black = 0.0;
    for i in 1..9 {
        black += at(i, 1) + at(i, 8);
    }
    for i in 2..8 {
        black += at(1, i) + at(8, i);
    }
    let black = black / 28.0;
    let bits: Vec<f64> = (0..fam.nbits).map(|i| at(fam.bit_x[i] as usize + 1, fam.bit_y[i] as usize + 1)).collect();
    // Otsu on the 36 bits plus the border: the split that best separates two
    // levels, constrained to lie above the border's level.
    let mut v: Vec<f64> = bits.clone();
    v.sort_by(f64::total_cmp);
    let (mut best, mut thr) = (f64::NEG_INFINITY, None);
    for k in 1..v.len() {
        let (lo, hi) = (&v[..k], &v[k..]);
        let (ml, mh) = (lo.iter().sum::<f64>() / lo.len() as f64, hi.iter().sum::<f64>() / hi.len() as f64);
        let score = (lo.len() * hi.len()) as f64 * (mh - ml) * (mh - ml);
        if score > best {
            best = score;
            thr = Some((ml, mh, 0.5 * (v[k - 1] + v[k])));
        }
    }
    let (_, white, thr) = thr?;
    if white - black < 0.05 {
        return None;
    }
    // The darks of the code must look like the border.
    let thr = thr.max(black + 0.25 * (white - black)).min(white - 0.25 * (white - black));
    let mut code = 0u64;
    for b in &bits {
        code = (code << 1) | (*b > thr) as u64;
    }
    fam.code_data.iter().enumerate().map(|(id, &c)| (id, ((c as u64) ^ code).count_ones())).min_by_key(|&(_, d)| d)
}

/// Fit a line `n · p = c` (unit `n`) to points by total least squares.
fn fit_line(pts: &[[f64; 2]]) -> Option<([f64; 2], f64)> {
    if pts.len() < 4 {
        return None;
    }
    let n = pts.len() as f64;
    let (mx, my) = (pts.iter().map(|p| p[0]).sum::<f64>() / n, pts.iter().map(|p| p[1]).sum::<f64>() / n);
    let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
    for p in pts {
        let (dx, dy) = (p[0] - mx, p[1] - my);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    // The direction is the major eigenvector; the normal is the minor one.
    let th = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let nrm = [-th.sin(), th.cos()];
    Some((nrm, nrm[0] * mx + nrm[1] * my))
}

/// Sub-pixel corners (in [`CORNER_ORDER`]) of a tag with the polarity of a
/// normal print in `img`: each side of the black square is re-found as the
/// dark-to-light gradient maximum along its normal at 24 points, fitted with
/// a line, and adjacent lines are intersected. Two passes; a pass that would
/// move a corner by more than a few pixels (it found some other edge) is
/// dropped.
pub fn refine(img: &Grey, corners: &[[f64; 2]; 4]) -> [[f64; 2]; 4] {
    let mut q = *corners;
    let tm = corners_mm(UNIT_MM);
    for _ in 0..2 {
        let Some(h) = homography(&tm, &q) else { return q };
        // How far a few pixels reach: search ±3 px, or ±8 % of a side on
        // big tags whose quad came from a coarse decimation.
        let side = (0..4).map(|i| (q[i][0] - q[(i + 1) % 4][0]).hypot(q[i][1] - q[(i + 1) % 4][1])).sum::<f64>() / 4.0;
        let reach = (0.02 * side).clamp(3.0, 8.0);
        let mut lines = Vec::new();
        for e in 0..4 {
            let (a, b) = (tm[e], tm[(e + 1) % 4]);
            let mut pts = Vec::new();
            for k in 0..24 {
                let t = 0.12 + 0.76 * k as f64 / 23.0;
                let m = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
                // Outward normal in the tag frame, as an image direction.
                let out = [m[0] * 1.02, m[1] * 1.02];
                let (pm, po) = (apply(&h, m), apply(&h, out));
                let d = [po[0] - pm[0], po[1] - pm[1]];
                let dl = d[0].hypot(d[1]);
                if dl < 1e-9 {
                    continue;
                }
                let d = [d[0] / dl, d[1] / dl];
                let g = |s: f64| {
                    let p = [pm[0] + s * d[0], pm[1] + s * d[1]];
                    img.at(p[0] + 0.25 * d[0], p[1] + 0.25 * d[1]) - img.at(p[0] - 0.25 * d[0], p[1] - 0.25 * d[1])
                };
                let (mut best, mut bs) = (f64::NEG_INFINITY, 0.0);
                let mut s = -reach;
                while s <= reach {
                    let v = g(s);
                    if v > best {
                        best = v;
                        bs = s;
                    }
                    s += 0.25;
                }
                if best <= 0.02 {
                    continue;
                }
                let (l, r) = (g(bs - 0.25), g(bs + 0.25));
                let den = l - 2.0 * best + r;
                // Parabola through the three samples; its vertex, in steps.
                let off = if den.abs() > 1e-12 { (0.5 * (l - r) / den).clamp(-1.0, 1.0) } else { 0.0 };
                let s = bs + off * 0.25;
                pts.push([pm[0] + s * d[0], pm[1] + s * d[1]]);
            }
            match fit_line(&pts) {
                Some(l) => lines.push(l),
                None => return q,
            }
        }
        let mut nq = q;
        for c in 0..4 {
            // Corner c is the meeting of side c−1 (ending at c) and side c.
            let (l1, l2) = (lines[(c + 3) % 4], lines[c]);
            let det = l1.0[0] * l2.0[1] - l1.0[1] * l2.0[0];
            if det.abs() < 1e-9 {
                return q;
            }
            nq[c] = [(l1.1 * l2.0[1] - l2.1 * l1.0[1]) / det, (l1.0[0] * l2.1 - l2.0[0] * l1.1) / det];
        }
        if (0..4).any(|c| (nq[c][0] - q[c][0]).hypot(nq[c][1] - q[c][1]) > reach + 1.0) {
            return q;
        }
        q = nq;
    }
    q
}

/// Every tag36h11 in `img` (display-encoded grey), corners refined on
/// `img` itself.
pub fn detect(img: &Grey) -> Vec<Tag> {
    detect_refined(img, img)
}

/// Every tag36h11 in `img` (display-encoded, what the quad finder
/// expects), with the corners refined on `lin`, the same image in linear
/// light: blur is linear in light, so a black-to-white edge's gradient
/// maximum sits on the edge there; on the sRGB-encoded image it shifts
/// towards the dark side.
///
/// Both polarities and every decimation run and are merged (one tag per id:
/// the fewest bit errors, then the finest decimation). If that finds
/// nothing, the search is repeated at 2×. Sorted by id.
pub fn detect_refined(img: &Grey, lin: &Grey) -> Vec<Tag> {
    assert_eq!(img.size, lin.size, "img and lin differ in size");
    let mut found = search(img, lin, 1.0);
    if found.is_empty() {
        found = search(&img.upsample2(), &lin.upsample2(), 2.0);
    }
    found.sort_by_key(|t| t.id);
    found
}

/// A candidate: tag, the decimation that found it.
struct Cand {
    tag: Tag,
    down: usize,
}

fn search(img: &Grey, lin: &Grey, scale: f64) -> Vec<Tag> {
    let fam = family();
    let debug = std::env::var_os("IRYS_TAGS_DEBUG").is_some();
    let mut out: Vec<Cand> = Vec::new();
    for negative in [false, true] {
        let (img, lin) = if negative { (img.inverted(), lin.inverted()) } else { (img.clone(), lin.clone()) };
        let (w, h) = img.size;
        let bytes: Vec<u8> = img.px.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
        let size = ImageSize { width: w, height: h };
        for down in DECIMATIONS {
            // A level whose decimated image is under ~64 px finds nothing.
            if w.min(h) / down < 64 {
                continue;
            }
            for q in quads(&bytes, size, down) {
                let mut best: Option<(usize, u32, [[f64; 2]; 4])> = None;
                for wind in [false, true] {
                    for rot in 0..4 {
                        let c: [[f64; 2]; 4] = std::array::from_fn(|k| q[if wind { (rot + 4 - k) % 4 } else { (rot + k) % 4 }]);
                        if let Some((id, ham)) = decode_with(&img, &c, &fam)
                            && best.as_ref().is_none_or(|b| ham < b.1)
                        {
                            best = Some((id, ham, c));
                        }
                    }
                }
                if debug {
                    eprintln!("irys-tags: x{scale} neg {negative} down {down} quad {q:?} -> {:?}", best.map(|b| (b.0, b.1)));
                }
                let Some((id, hamming, c)) = best else { continue };
                if hamming > MAX_HAMMING {
                    continue;
                }
                let corners = refine(&lin, &c).map(|p| [p[0] / scale, p[1] / scale]);
                let cand = Cand { tag: Tag { id, hamming, negative, corners }, down };
                match out.iter_mut().find(|o| o.tag.id == id) {
                    Some(o) if (hamming, down) < (o.tag.hamming, o.down) => *o = cand,
                    Some(_) => {}
                    None => out.push(cand),
                }
            }
        }
    }
    out.into_iter().map(|c| c.tag).collect()
}

/// kornia's quads at decimation `down`, pixels (`x` right, `y` down, pixel
/// centres at `i + 0.5`).
fn quads(bytes: &[u8], size: ImageSize, down: usize) -> Vec<[[f64; 2]; 4]> {
    let Ok(mut cfg) = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11]) else { return vec![] };
    cfg.downscale_factor = down;
    let Ok(im) = Image::<u8, 1>::from_size_slice(size, bytes) else { return vec![] };
    let Ok(mut dec) = AprilTagDecoder::new(cfg, size) else { return vec![] };
    let Ok(dets) = dec.decode(&im) else { return vec![] };
    dets.iter().map(|d| std::array::from_fn(|k| [d.quad.corners[k].x as f64, d.quad.corners[k].y as f64])).collect()
}
