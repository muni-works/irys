//! v1: the AprilTag (tag36h11) that fixes the camera, the metric scale and
//! the ground plane.
//!
//! Detection is `kornia-apriltag` (pure Rust, Apache-2.0; its tag36h11 code
//! table is the AprilRobotics one, BSD-2-Clause). Everything after the quad
//! is here: the corners are matched to the tag's own frame by decoding the
//! 36 bits under each of the eight corner assignments (so no corner-order
//! convention is trusted), refined to sub-pixel by fitting a line to the
//! gradient maximum along each side of the black square, and the camera is
//! solved from them: pose `(R, t)` and the focal length `f`, the principal
//! point at the image centre, with the EXIF focal length as a prior.
//!
//! The tag frame is the world: origin at the tag's centre, `x` along its
//! columns, `y` up its rows (towards row 0), `z` out of the paper,
//! millimetres. [`TAG_MM`] is the side of the black square.

use kornia_apriltag::family::{TagFamily, TagFamilyKind};
use kornia_apriltag::{AprilTagDecoder, DecodeTagsConfig};
use kornia_image::{Image, ImageSize};

/// The black square's side the kit prints, millimetres.
pub const TAG_MM: f64 = 50.0;
/// The tag id the kit prints.
pub const TAG_ID: usize = 0;

/// tag36h11's 10 × 10 cell image of `id`, row-major from the top-left:
/// `true` is white. The outer ring is the white quiet zone, the next ring
/// the black border, the inner 6 × 6 the code (AprilRobotics'
/// `apriltag_to_image`).
pub fn cells(id: usize) -> [[bool; 10]; 10] {
    let fam = TagFamily::tag36_h11().expect("tag36h11");
    let code = fam.code_data[id] as u64;
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
    c
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

/// The tag's corners in the tag frame (mm): `(−,−), (+,−), (+,+), (−,+)`.
pub fn corners_mm(tag_mm: f64) -> [[f64; 2]; 4] {
    let a = tag_mm / 2.0;
    [[-a, -a], [a, -a], [a, a], [-a, a]]
}

/// A detected tag: id, Hamming distance of its code, and the black square's
/// corners in pixels in [`corners_mm`]'s order.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TagObs {
    /// Tag id.
    pub id: usize,
    /// Bit errors.
    pub hamming: u32,
    /// Corners, pixels (`x` right, `y` down).
    pub corners: [[f64; 2]; 4],
}

/// A grey image, `f32` in `[0, 1]`.
#[derive(Clone)]
pub struct Grey {
    /// Width, height.
    pub size: (usize, usize),
    /// Pixels, row-major.
    pub px: Vec<f32>,
}

impl Grey {
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

fn decode_with(img: &Grey, quad: &[[f64; 2]; 4], fam: &TagFamily, tag_mm: f64) -> Option<(usize, u32)> {
    let h = homography(&corners_mm(tag_mm), quad)?;
    let cell = tag_mm / 8.0;
    let at = |c: usize, r: usize| {
        let (x, y) = ((c as f64 - 5.0 + 0.5) * cell, (5.0 - r as f64 - 0.5) * cell);
        let p = apply(&h, [x, y]);
        img.at(p[0], p[1])
    };
    // Threshold between the black border ring and the white quiet zone.
    let (mut black, mut white) = (0.0, 0.0);
    for i in 1..9 {
        black += at(i, 1) + at(i, 8) + at(1, i) + at(8, i);
    }
    for i in 0..10 {
        white += at(i, 0) + at(i, 9) + at(0, i) + at(9, i);
    }
    let (black, white) = (black / 32.0, white / 40.0);
    if white - black < 0.05 {
        return None;
    }
    let thr = 0.5 * (black + white);
    let mut code = 0u64;
    for i in 0..fam.nbits {
        let b = at(fam.bit_x[i] as usize + 1, fam.bit_y[i] as usize + 1) > thr;
        code = (code << 1) | b as u64;
    }
    fam.code_data
        .iter()
        .enumerate()
        .map(|(id, &c)| (id, ((c as u64) ^ code).count_ones()))
        .min_by_key(|&(_, d)| d)
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

/// Sub-pixel corners: each side of the black square is re-found as the
/// gradient maximum along its normal at 24 points, fitted with a line, and
/// adjacent lines are intersected. Two passes.
pub fn refine(img: &Grey, corners: &[[f64; 2]; 4], tag_mm: f64) -> [[f64; 2]; 4] {
    let mut q = *corners;
    let tm = corners_mm(tag_mm);
    for _ in 0..2 {
        let Some(h) = homography(&tm, &q) else { return q };
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
                let mut s = -3.0;
                while s <= 3.0 {
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
        // A refinement that moves a corner by more than a few pixels found
        // some other edge.
        if (0..4).any(|c| (nq[c][0] - q[c][0]).hypot(nq[c][1] - q[c][1]) > 4.0) {
            return q;
        }
        q = nq;
    }
    q
}

/// Every tag36h11 in `img`, corners refined.
pub fn detect(img: &Grey, tag_mm: f64) -> Vec<TagObs> {
    detect_refined(img, img, tag_mm)
}

/// [`detect`] on `img` (display-encoded, what the detector expects), with
/// the corners refined on `lin`, the same image in linear light: blur is
/// linear in light, so a black-to-white edge's gradient maximum sits on the
/// edge there; on the sRGB-encoded image it shifts towards the dark side
/// (the tag looks smaller, the camera farther: 0.6 % in the synthetic
/// photos).
pub fn detect_refined(img: &Grey, lin: &Grey, tag_mm: f64) -> Vec<TagObs> {
    let (w, h) = img.size;
    let bytes: Vec<u8> = img.px.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
    let fam = TagFamily::tag36_h11().expect("tag36h11");
    let mut out: Vec<TagObs> = Vec::new();
    // Large photos are decimated by 2 inside the detector; small ones not.
    for down in [2usize, 3, 4, 1] {
        let Ok(mut cfg) = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11]) else { continue };
        cfg.downscale_factor = down;
        let Ok(im) = Image::<u8, 1>::from_size_slice(ImageSize { width: w, height: h }, &bytes) else { continue };
        let Ok(mut dec) = AprilTagDecoder::new(cfg, ImageSize { width: w, height: h }) else { continue };
        let Ok(dets) = dec.decode(&im) else { continue };
        for d in dets {
            let q: Vec<[f64; 2]> = d.quad.corners.iter().map(|c| [c.x as f64, c.y as f64]).collect();
            // Every assignment of the quad to the tag's corners: four
            // rotations, both windings. The one that decodes wins.
            let mut best: Option<(usize, u32, [[f64; 2]; 4])> = None;
            for wind in [false, true] {
                for rot in 0..4 {
                    let mut c = [[0.0; 2]; 4];
                    for k in 0..4 {
                        let i = if wind { (rot + 4 - k) % 4 } else { (rot + k) % 4 };
                        c[k] = q[i];
                    }
                    if let Some((id, ham)) = decode_with(img, &c, &fam, tag_mm) {
                        if best.as_ref().is_none_or(|b| ham < b.1) {
                            best = Some((id, ham, c));
                        }
                    }
                }
            }
            let Some((id, ham, c)) = best else { continue };
            if ham > 2 || out.iter().any(|o| o.id == id) {
                continue;
            }
            let corners = refine(lin, &c, tag_mm);
            out.push(TagObs { id, hamming: ham, corners });
        }
        if !out.is_empty() {
            break;
        }
    }
    out
}

/// A pinhole with square pixels: focal `f` (pixels) and principal point.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Intrinsics {
    /// Focal length, pixels.
    pub f: f64,
    /// Principal point, pixels.
    pub cx: f64,
    /// Principal point, pixels.
    pub cy: f64,
}

/// World (tag frame) → camera: `X_c = R X + t`, camera `x` right, `y`
/// down, `z` forward.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Pose {
    /// Rotation, row-major.
    pub r: [[f64; 3]; 3],
    /// Translation, mm.
    pub t: [f64; 3],
}

impl Pose {
    /// The camera centre in the world.
    pub fn eye(&self) -> [f64; 3] {
        let (r, t) = (&self.r, &self.t);
        [
            -(r[0][0] * t[0] + r[1][0] * t[1] + r[2][0] * t[2]),
            -(r[0][1] * t[0] + r[1][1] * t[1] + r[2][1] * t[2]),
            -(r[0][2] * t[0] + r[1][2] * t[1] + r[2][2] * t[2]),
        ]
    }

    /// Project world point `p` with `k`.
    pub fn project(&self, k: &Intrinsics, p: [f64; 3]) -> Option<[f64; 2]> {
        let c: Vec<f64> = (0..3).map(|i| self.r[i][0] * p[0] + self.r[i][1] * p[1] + self.r[i][2] * p[2] + self.t[i]).collect();
        (c[2] > 1e-9).then(|| [k.f * c[0] / c[2] + k.cx, k.f * c[1] / c[2] + k.cy])
    }
}

fn rodrigues(w: [f64; 3]) -> [[f64; 3]; 3] {
    let th = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
    if th < 1e-12 {
        return [[1.0, -w[2], w[1]], [w[2], 1.0, -w[0]], [-w[1], w[0], 1.0]];
    }
    let k = [w[0] / th, w[1] / th, w[2] / th];
    let (s, c) = th.sin_cos();
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = (1.0 - c) * k[i] * k[j] + if i == j { c } else { 0.0 };
        }
    }
    r[0][1] -= s * k[2];
    r[0][2] += s * k[1];
    r[1][0] += s * k[2];
    r[1][2] -= s * k[0];
    r[2][0] -= s * k[1];
    r[2][1] += s * k[0];
    r
}

fn matmul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    c
}

/// The focal length a plane homography implies with the principal point
/// at `(cx, cy)` (Zhang's two constraints, averaged); `None` if neither
/// is usable.
pub fn focal_from_homography(h: &[f64; 9], cx: f64, cy: f64) -> Option<f64> {
    // Shift the principal point to the origin.
    let col = |j: usize| [h[j] - cx * h[6 + j], h[3 + j] - cy * h[6 + j], h[6 + j]];
    let (a, b) = (col(0), col(1));
    let mut fs = Vec::new();
    let d1 = a[2] * b[2];
    if d1.abs() > 1e-12 {
        let f2 = -(a[0] * b[0] + a[1] * b[1]) / d1;
        if f2 > 0.0 {
            fs.push(f2.sqrt());
        }
    }
    let d2 = a[2] * a[2] - b[2] * b[2];
    if d2.abs() > 1e-12 {
        let f2 = -(a[0] * a[0] + a[1] * a[1] - b[0] * b[0] - b[1] * b[1]) / d2;
        if f2 > 0.0 {
            fs.push(f2.sqrt());
        }
    }
    (!fs.is_empty()).then(|| fs.iter().sum::<f64>() / fs.len() as f64)
}

/// The camera from one tag: pose and focal length. `f0` is the prior
/// focal (EXIF), `f_sigma` its relative standard deviation; the principal
/// point is held at `(cx, cy)`. Returns the camera and the corners' RMS
/// reprojection error in pixels.
pub fn solve_camera(obs: &TagObs, tag_mm: f64, f0: f64, f_sigma: f64, cx: f64, cy: f64) -> Option<(Intrinsics, Pose, f64)> {
    let tm = corners_mm(tag_mm);
    let h = homography(&tm, &obs.corners)?;
    let mut f = f0;
    // Initial pose from K⁻¹ H = λ [r1 r2 t].
    let kinv = |v: [f64; 3], f: f64| [(v[0] - cx * v[2]) / f, (v[1] - cy * v[2]) / f, v[2]];
    let init = |f: f64| -> Option<Pose> {
        let h1 = kinv([h[0], h[3], h[6]], f);
        let h2 = kinv([h[1], h[4], h[7]], f);
        let h3 = kinv([h[2], h[5], h[8]], f);
        let n1 = (h1[0] * h1[0] + h1[1] * h1[1] + h1[2] * h1[2]).sqrt();
        let n2 = (h2[0] * h2[0] + h2[1] * h2[1] + h2[2] * h2[2]).sqrt();
        let mut lam = 2.0 / (n1 + n2);
        if h3[2] * lam < 0.0 {
            lam = -lam;
        }
        let r1: Vec<f64> = h1.iter().map(|x| x * lam).collect();
        let r2: Vec<f64> = h2.iter().map(|x| x * lam).collect();
        // Make r1, r2 orthonormal symmetrically.
        let norm = |v: [f64; 3]| {
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / n, v[1] / n, v[2] / n]
        };
        let (a, b) = (norm([r1[0], r1[1], r1[2]]), norm([r2[0], r2[1], r2[2]]));
        let s = norm([a[0] + b[0], a[1] + b[1], a[2] + b[2]]);
        let d = norm([a[0] - b[0], a[1] - b[1], a[2] - b[2]]);
        let k = std::f64::consts::FRAC_1_SQRT_2;
        let x = [k * (s[0] + d[0]), k * (s[1] + d[1]), k * (s[2] + d[2])];
        let y = [k * (s[0] - d[0]), k * (s[1] - d[1]), k * (s[2] - d[2])];
        let z = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]];
        let r = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]];
        Some(Pose { r, t: [h3[0] * lam, h3[1] * lam, h3[2] * lam] })
    };
    let mut pose = init(f)?;
    // Levenberg–Marquardt on (ω, t, log f): eight corner residuals plus
    // the focal prior.
    let resid = |pose: &Pose, f: f64| -> Option<Vec<f64>> {
        let k = Intrinsics { f, cx, cy };
        let mut r = Vec::with_capacity(9);
        for (c, o) in tm.iter().zip(&obs.corners) {
            let p = pose.project(&k, [c[0], c[1], 0.0])?;
            r.push(p[0] - o[0]);
            r.push(p[1] - o[1]);
        }
        // 0.3 px per corner is the detector's noise; the prior in the same units.
        let s = 0.3;
        for v in r.iter_mut() {
            *v /= s;
        }
        r.push((f / f0).ln() / f_sigma);
        Some(r)
    };
    let step = |pose: &Pose, f: f64, d: &[f64; 7]| -> (Pose, f64) {
        let r = matmul(&rodrigues([d[0], d[1], d[2]]), &pose.r);
        (Pose { r, t: [pose.t[0] + d[3], pose.t[1] + d[4], pose.t[2] + d[5]] }, f * d[6].exp())
    };
    let cost = |r: &[f64]| r.iter().map(|x| x * x).sum::<f64>();
    let mut cur = resid(&pose, f)?;
    let mut mu = 1e-3;
    for _ in 0..100 {
        let mut jac = vec![[0.0; 7]; cur.len()];
        for k in 0..7 {
            let mut d = [0.0; 7];
            let e = if k < 3 { 1e-6 } else if k < 6 { 1e-4 } else { 1e-6 };
            d[k] = e;
            let (p2, f2) = step(&pose, f, &d);
            let r2 = resid(&p2, f2)?;
            for i in 0..cur.len() {
                jac[i][k] = (r2[i] - cur[i]) / e;
            }
        }
        let mut a = vec![0.0; 49];
        let mut g = [0.0; 7];
        for (row, ri) in jac.iter().zip(&cur) {
            for p in 0..7 {
                g[p] += row[p] * ri;
                for q in 0..7 {
                    a[p * 7 + q] += row[p] * row[q];
                }
            }
        }
        let mut improved = false;
        for _ in 0..8 {
            let mut m = a.clone();
            for p in 0..7 {
                m[p * 7 + p] *= 1.0 + mu;
                m[p * 7 + p] += 1e-12;
            }
            let Some(dx) = crate::fit::solve(&m, &g.map(|x| -x), 7) else {
                mu *= 10.0;
                continue;
            };
            let d: [f64; 7] = std::array::from_fn(|i| dx[i]);
            let (p2, f2) = step(&pose, f, &d);
            if let Some(r2) = resid(&p2, f2) {
                if cost(&r2) < cost(&cur) {
                    let rel = (cost(&cur) - cost(&r2)) / cost(&cur).max(1e-30);
                    pose = p2;
                    f = f2;
                    cur = r2;
                    mu = (mu / 3.0).max(1e-9);
                    improved = rel > 1e-12;
                    break;
                }
            }
            mu *= 10.0;
        }
        if !improved {
            break;
        }
    }
    let k = Intrinsics { f, cx, cy };
    let rms = (cur[..8].iter().map(|x| (x * 0.3).powi(2)).sum::<f64>() / 4.0).sqrt();
    Some((k, pose, rms))
}

/// What the photo's EXIF says about the lens.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct Exif {
    /// `FocalLengthIn35mmFilm`.
    pub f35: Option<f64>,
    /// `FocalLength`, mm.
    pub f_mm: Option<f64>,
    /// `Orientation` (1 = as stored).
    pub orientation: u16,
}

/// Read the EXIF of a JPEG's bytes.
pub fn read_exif(bytes: &[u8]) -> Exif {
    let mut e = Exif { orientation: 1, ..Exif::default() };
    let Ok(ex) = exif::Reader::new().read_from_container(&mut std::io::Cursor::new(bytes)) else { return e };
    let num = |tag: exif::Tag| -> Option<f64> {
        let f = ex.get_field(tag, exif::In::PRIMARY)?;
        match &f.value {
            exif::Value::Short(v) => v.first().map(|&x| x as f64),
            exif::Value::Long(v) => v.first().map(|&x| x as f64),
            exif::Value::Rational(v) => v.first().map(|r| r.to_f64()),
            _ => None,
        }
    };
    e.f35 = num(exif::Tag::FocalLengthIn35mmFilm).filter(|&f| f > 0.0);
    e.f_mm = num(exif::Tag::FocalLength).filter(|&f| f > 0.0);
    e.orientation = num(exif::Tag::Orientation).map(|o| o as u16).unwrap_or(1);
    e
}

/// The focal length in pixels for an image of `w × h` (as displayed) from
/// its EXIF: the 35 mm-equivalent focal on the diagonal (43.27 mm), else
/// `FocalLength` with `sensor_w_mm`, else `None`.
pub fn exif_focal_px(e: &Exif, w: usize, h: usize, sensor_w_mm: Option<f64>) -> Option<f64> {
    let diag = (w as f64).hypot(h as f64);
    if let Some(f35) = e.f35 {
        return Some(f35 / 36f64.hypot(24.0) * diag);
    }
    match (e.f_mm, sensor_w_mm) {
        (Some(f), Some(sw)) => Some(f / sw * w.max(h) as f64),
        _ => None,
    }
}

/// A JPEG with an EXIF APP1 segment carrying `FocalLength` and
/// `FocalLengthIn35mmFilm` inserted after SOI (for the synthetic photos,
/// so they exercise the same path as a phone's).
pub fn with_exif(jpeg: &[u8], f_mm: f64, f35: u16) -> Vec<u8> {
    // TIFF, little endian: IFD0 { ExifIFD pointer } → Exif IFD { FocalLength, FocalLengthIn35mmFilm }.
    let mut t: Vec<u8> = Vec::new();
    t.extend_from_slice(b"II*\0");
    t.extend_from_slice(&8u32.to_le_bytes());
    // IFD0 at 8: 1 entry.
    t.extend_from_slice(&1u16.to_le_bytes());
    let exif_ifd = 8 + 2 + 12 + 4;
    t.extend_from_slice(&0x8769u16.to_le_bytes());
    t.extend_from_slice(&4u16.to_le_bytes());
    t.extend_from_slice(&1u32.to_le_bytes());
    t.extend_from_slice(&(exif_ifd as u32).to_le_bytes());
    t.extend_from_slice(&0u32.to_le_bytes());
    // Exif IFD: 2 entries, then the rational.
    let rat_at = exif_ifd + 2 + 2 * 12 + 4;
    t.extend_from_slice(&2u16.to_le_bytes());
    t.extend_from_slice(&0x920Au16.to_le_bytes());
    t.extend_from_slice(&5u16.to_le_bytes());
    t.extend_from_slice(&1u32.to_le_bytes());
    t.extend_from_slice(&(rat_at as u32).to_le_bytes());
    t.extend_from_slice(&0xA405u16.to_le_bytes());
    t.extend_from_slice(&3u16.to_le_bytes());
    t.extend_from_slice(&1u32.to_le_bytes());
    t.extend_from_slice(&(f35 as u32).to_le_bytes());
    t.extend_from_slice(&0u32.to_le_bytes());
    t.extend_from_slice(&((f_mm * 1000.0).round() as u32).to_le_bytes());
    t.extend_from_slice(&1000u32.to_le_bytes());
    let mut app1 = b"Exif\0\0".to_vec();
    app1.extend_from_slice(&t);
    let mut out = jpeg[..2].to_vec();
    out.extend_from_slice(&[0xFF, 0xE1]);
    out.extend_from_slice(&((app1.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(&app1);
    out.extend_from_slice(&jpeg[2..]);
    out
}
