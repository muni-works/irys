//! v1: the AprilTag (tag36h11) that fixes the camera, the metric scale and
//! the ground plane.
//!
//! Detection is [`irys_tags`] (pure Rust, Apache-2.0): both polarities,
//! every decimation, a 2× retry for small tags, its own 36-bit decode under
//! all eight corner assignments, and sub-pixel corners refined on linear
//! light. Any tag36h11 id works. This module solves the camera from it:
//! pose `(R, t)` and the focal length `f`, the principal point at the image
//! centre, with the EXIF focal length as a prior.
//!
//! The tag frame is the world: origin at the tag's centre, `x` along its
//! columns, `y` up its rows (towards row 0), `z` out of the paper,
//! millimetres. [`TAG_MM`] is the side of the black square. Corners are in
//! [`irys_tags::CORNER_ORDER`] (top-left, top-right, bottom-right,
//! bottom-left in the tag's own frame).

pub use irys_tags::{CORNER_ORDER, Grey, apply, corners_mm, homography, white_at};

/// The black square's side the kit prints, millimetres.
pub const TAG_MM: f64 = 50.0;
/// The tag id the kit prints (any id is accepted; this one is preferred
/// when several are in view).
pub const TAG_ID: usize = 0;

/// tag36h11's 10 × 10 cell image of `id` ([`irys_tags::cells`]); panics
/// past the last id.
pub fn cells(id: usize) -> [[bool; 10]; 10] {
    irys_tags::cells(id).expect("tag36h11 id")
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

impl From<irys_tags::Tag> for TagObs {
    fn from(t: irys_tags::Tag) -> Self {
        TagObs { id: t.id, hamming: t.hamming, corners: t.corners }
    }
}

/// Every tag36h11 in `img`, corners refined. `tag_mm` only scales the tag
/// frame, which detection doesn't need; it is kept for the callers.
pub fn detect(img: &Grey, _tag_mm: f64) -> Vec<TagObs> {
    irys_tags::detect(img).into_iter().map(TagObs::from).collect()
}

/// [`detect`] on `img` (display-encoded), with the corners refined on
/// `lin`, the same image in linear light ([`irys_tags::detect_refined`]).
pub fn detect_refined(img: &Grey, lin: &Grey, _tag_mm: f64) -> Vec<TagObs> {
    irys_tags::detect_refined(img, lin).into_iter().map(TagObs::from).collect()
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
            if let Some(r2) = resid(&p2, f2)
                && cost(&r2) < cost(&cur) {
                    let rel = (cost(&cur) - cost(&r2)) / cost(&cur).max(1e-30);
                    pose = p2;
                    f = f2;
                    cur = r2;
                    mu = (mu / 3.0).max(1e-9);
                    improved = rel > 1e-12;
                    break;
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
