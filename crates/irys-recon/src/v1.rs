//! v1: from photos (each with the tag in frame) to a program.
//!
//! 1. [`load`]: decode, apply the EXIF orientation, linearise (sRGB).
//! 2. [`calibrate`]: find the tag, solve the camera ([`crate::tag`]).
//! 3. [`locate`]: segment a low-resolution copy against the backdrop and
//!    take the largest component off the tag as the part; crop a 4:3
//!    window around it.
//! 4. [`frame`]: resample the crop to the fit's film, segment it, weight
//!    out the tag and anything off the table.
//! 5. [`recover_photos`]: base box and pose from the masks, the appearance
//!    from the base box (key direction, roughness, key size, gamma), the
//!    noise scale from the residual, then v0's search with v1's fixes
//!    (Huber residual, a beam of two, finalists at full resolution) on the
//!    photo model, and a last polish: appearance, then geometry again.

use std::path::Path;

use crate::dsl::Program;
use crate::fit::{Fit, Fitter, Level, Target};
use crate::look::{self, Look};
use crate::model::Setup;
use crate::photo::{self, Camera, Rig, View};
use crate::search::{self, Algorithmic, Options, Recovery};
use crate::segment::{self, Region};
use crate::tag::{self, Exif, Grey, Intrinsics, Pose, TagObs};

/// A photo, linear.
pub struct Photo {
    /// File name.
    pub name: String,
    /// Width, height (as displayed).
    pub size: (usize, usize),
    /// Linear sRGB.
    pub rgb: Vec<[f32; 3]>,
    /// sRGB-encoded grey in `[0, 1]` (for the tag detector).
    pub grey: Grey,
    /// Linear grey (for the tag's corners).
    pub grey_lin: Grey,
    /// EXIF.
    pub exif: Exif,
}

/// Decode `path`.
pub fn load(path: &Path) -> Result<Photo, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let exif = tag::read_exif(&bytes);
    let img = image::load_from_memory(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    let img = match exif.orientation {
        3 => img.rotate180(),
        6 => img.rotate90(),
        8 => img.rotate270(),
        _ => img,
    };
    let rgb8 = img.to_rgb8();
    let (w, h) = (rgb8.width() as usize, rgb8.height() as usize);
    let lut: Vec<f32> = (0..256).map(|i| look::srgb_to_linear(i as f64 / 255.0) as f32).collect();
    let rgb: Vec<[f32; 3]> = rgb8.pixels().map(|p| [lut[p[0] as usize], lut[p[1] as usize], lut[p[2] as usize]]).collect();
    let grey = Grey { size: (w, h), px: rgb8.pixels().map(|p| (p[0] as f32 + p[1] as f32 + p[2] as f32) / 765.0).collect() };
    let grey_lin = Grey { size: (w, h), px: rgb.iter().map(|c| (c[0] + c[1] + c[2]) / 3.0).collect() };
    Ok(Photo { name: path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(), size: (w, h), rgb, grey, grey_lin, exif })
}

/// What the tag gave.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Calib {
    /// The tag.
    pub tag: TagObs,
    /// Intrinsics (full image).
    pub k: Intrinsics,
    /// World → camera.
    pub pose: Pose,
    /// Corner RMS, pixels.
    pub rms: f64,
    /// The EXIF focal in pixels, if there was one.
    pub f_exif: Option<f64>,
}

/// Find the tag and solve the camera. `sensor_w_mm` is used when the EXIF
/// has `FocalLength` but no 35 mm equivalent.
pub fn calibrate(p: &Photo, tag_mm: f64, sensor_w_mm: Option<f64>) -> Result<Calib, String> {
    let obs = tag::detect_refined(&p.grey, &p.grey_lin, tag_mm);
    let t = obs.iter().find(|o| o.id == tag::TAG_ID).or(obs.first()).ok_or(format!("{}: no tag36h11 found", p.name))?.clone();
    let (w, h) = p.size;
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    let f_exif = tag::exif_focal_px(&p.exif, w, h, sensor_w_mm);
    let (f0, sig) = match f_exif {
        // One tag constrains the focal length weakly (it trades against
        // distance); the EXIF value is the better estimate, held to 2 %.
        Some(f) => (f, 0.02),
        None => {
            let hm = tag::homography(&tag::corners_mm(tag_mm), &t.corners).ok_or("degenerate tag")?;
            (tag::focal_from_homography(&hm, cx, cy).unwrap_or(0.85 * w.max(h) as f64), 0.5)
        }
    };
    let (k, pose, rms) = tag::solve_camera(&t, tag_mm, f0, sig, cx, cy).ok_or("camera solve failed")?;
    Ok(Calib { tag: t, k, pose, rms, f_exif })
}

/// Area-resample the `crop` window of `rgb` (`size`) to `out`.
pub fn resample(rgb: &[[f32; 3]], size: (usize, usize), crop: (f64, f64, f64, f64), out: (usize, usize)) -> Vec<[f32; 3]> {
    let (w, h) = size;
    // Per output column/row: source pixel indices and weights.
    let axis = |n_out: usize, start: f64, len: f64, n_in: usize| -> Vec<Vec<(usize, f32)>> {
        (0..n_out)
            .map(|i| {
                let (a, b) = (start + len * i as f64 / n_out as f64, start + len * (i + 1) as f64 / n_out as f64);
                let mut v = Vec::new();
                let mut x = a.floor();
                while x < b {
                    let wgt = (b.min(x + 1.0) - a.max(x)).max(0.0);
                    if wgt > 0.0 {
                        v.push(((x.max(0.0) as usize).min(n_in - 1), wgt as f32));
                    }
                    x += 1.0;
                }
                let s: f32 = v.iter().map(|q| q.1).sum();
                v.iter().map(|&(i, q)| (i, q / s.max(1e-9))).collect()
            })
            .collect()
    };
    let xs = axis(out.0, crop.0, crop.2, w);
    let ys = axis(out.1, crop.1, crop.3, h);
    let mut o = vec![[0f32; 3]; out.0 * out.1];
    for (j, yw) in ys.iter().enumerate() {
        for (i, xw) in xs.iter().enumerate() {
            let mut acc = [0f32; 3];
            for &(y, wy) in yw {
                for &(x, wx) in xw {
                    let c = rgb[y * w + x];
                    for k in 0..3 {
                        acc[k] += c[k] * wx * wy;
                    }
                }
            }
            o[j * out.0 + i] = acc;
        }
    }
    o
}

/// The crop window (full-image pixels, 4:3) around the part.
pub fn locate(p: &Photo, c: &Calib, region: &Region) -> Result<(f64, f64, f64, f64), String> {
    let (w, h) = p.size;
    let k = (w as f64 / 640.0).ceil().max(1.0);
    let lw = (w as f64 / k) as usize;
    let lh = (h as f64 / k) as usize;
    let crop = (0.0, 0.0, lw as f64 * k, lh as f64 * k);
    let small = resample(&p.rgb, p.size, crop, (lw, lh));
    let cam = Camera::crop(&c.k, &c.pose, crop, (lw, lh));
    let bd = segment::backdrop(&small, &cam, region).ok_or("no backdrop around the tag")?;
    let mut m = segment::mask(&small, &cam, &bd);
    let pts = segment::table_points(&cam, &Region { radius: 1e9, ..*region });
    for (i, q) in pts.iter().enumerate() {
        // Off the table or on the tag: not the part.
        if q.is_none() {
            m[i] = 0.0;
        }
    }
    segment::drop_specks(&mut m, (lw, lh), 12);
    let comps = segment::components(&m, (lw, lh));
    // The largest component whose table centroid is within the region.
    let part = comps
        .iter()
        .find(|(px, _)| {
            let (mut sx, mut sy, mut n) = (0.0, 0.0, 0.0);
            for &q in px {
                if let Some(t) = cam.on_table((q % lw) as f64 + 0.5, (q / lw) as f64 + 0.5) {
                    sx += t.0;
                    sy += t.1;
                    n += 1.0;
                }
            }
            n > 0.0 && (sx / n).hypot(sy / n) < region.radius
        })
        .ok_or("no part found beside the tag")?;
    let bb = part.1;
    let (x0, y0, x1, y1) = (bb[0] as f64 * k, bb[1] as f64 * k, (bb[2] + 1) as f64 * k, (bb[3] + 1) as f64 * k);
    let (cxp, cyp) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    let (mut cw, mut ch) = ((x1 - x0) * 1.35, (y1 - y0) * 1.35);
    if cw / ch > 4.0 / 3.0 {
        ch = cw * 0.75;
    } else {
        cw = ch * 4.0 / 3.0;
    }
    let cw = cw.min(w as f64).max(64.0);
    let ch = (cw * 0.75).min(h as f64);
    let cw = ch * 4.0 / 3.0;
    let x = (cxp - cw / 2.0).clamp(0.0, w as f64 - cw);
    let y = (cyp - ch / 2.0).clamp(0.0, h as f64 - ch);
    Ok((x, y, cw, ch))
}

/// One photo on the fit's film.
pub struct Framed {
    /// Camera.
    pub cam: Camera,
    /// Linear sRGB.
    pub rgb: Vec<[f32; 3]>,
    /// Linear grey (mean of the channels).
    pub grey: Vec<f64>,
    /// Segmentation.
    pub mask: Vec<f64>,
    /// Pixel weights.
    pub weight: Vec<f64>,
    /// The crop (full-image pixels).
    pub crop: (f64, f64, f64, f64),
    /// Coverage from chromaticity (0 backdrop, 1 part), and its noise;
    /// `None` when the part's colour is too close to the backdrop's.
    pub cov: Option<(Vec<f64>, f64)>,
}

/// Resample, segment and weight.
pub fn frame(p: &Photo, c: &Calib, crop: (f64, f64, f64, f64), film: (usize, usize), region: &Region) -> Result<Framed, String> {
    let rgb = resample(&p.rgb, p.size, crop, film);
    let cam = Camera::crop(&c.k, &c.pose, crop, film);
    let bd = segment::backdrop(&rgb, &cam, &Region { radius: 1e9, ..*region }).ok_or("no backdrop in the crop")?;
    let mut mask = segment::mask(&rgb, &cam, &bd);
    let pts = segment::table_points(&cam, &Region { radius: 1e9, ..*region });
    let weight: Vec<f64> = pts.iter().map(|q| q.is_some() as u8 as f64).collect();
    for (m, w) in mask.iter_mut().zip(&weight) {
        if *w == 0.0 {
            *m = 0.0;
        }
    }
    segment::drop_specks(&mut mask, film, 6);
    let grey = rgb.iter().map(|c| (c[0] as f64 + c[1] as f64 + c[2] as f64) / 3.0).collect();
    let cov = chroma_coverage(&rgb, &mask, &weight, film, &bd);
    Ok(Framed { cam, rgb, grey, mask, weight, crop, cov })
}

/// Coverage measured by chromaticity: each pixel's chromaticity projected
/// on the line from the backdrop's to the part's (the median inside the
/// mask, two pixels in from its edge).
fn chroma_coverage(rgb: &[[f32; 3]], mask: &[f64], weight: &[f64], (w, h): (usize, usize), bd: &segment::Backdrop) -> Option<(Vec<f64>, f64)> {
    let ch = |c: [f32; 3]| {
        let s = (c[0] + c[1] + c[2]) as f64 + 1e-4;
        [c[0] as f64 / s, c[1] as f64 / s]
    };
    let inner = |p: usize, r: i64| {
        let (x, y) = ((p % w) as i64, (p / w) as i64);
        (-r..=r).all(|dy| (-r..=r).all(|dx| {
            let (xx, yy) = (x + dx, y + dy);
            xx >= 0 && yy >= 0 && xx < w as i64 && yy < h as i64 && mask[yy as usize * w + xx as usize] > 0.5
        }))
    };
    let med = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied()
    };
    let ins: Vec<usize> = (0..w * h).filter(|&p| inner(p, 2)).collect();
    let cr = med(ins.iter().map(|&p| ch(rgb[p])[0]).collect())?;
    let cg = med(ins.iter().map(|&p| ch(rgb[p])[1]).collect())?;
    let d = [cr - bd.chroma[0], cg - bd.chroma[1]];
    let dl2 = d[0] * d[0] + d[1] * d[1];
    if dl2.sqrt() < 8.0 * bd.chroma_sigma {
        return None;
    }
    let cov: Vec<f64> = rgb
        .iter()
        .map(|&c| {
            let q = ch(c);
            (((q[0] - bd.chroma[0]) * d[0] + (q[1] - bd.chroma[1]) * d[1]) / dl2).clamp(-0.5, 1.5)
        })
        .collect();
    // Its noise: the spread on the backdrop away from the part.
    let table: Vec<f64> = (0..w * h).filter(|&p| weight[p] > 0.0 && mask[p] < 0.5 && !inner(p, 0) && {
        let (x, y) = ((p % w) as i64, (p / w) as i64);
        (-3i64..=3).all(|dy| (-3i64..=3).all(|dx| {
            let (xx, yy) = ((x + dx).clamp(0, w as i64 - 1), (y + dy).clamp(0, h as i64 - 1));
            mask[yy as usize * w + xx as usize] < 0.5
        }))
    }).map(|p| cov[p].abs()).collect();
    let s = (1.4826 * med(table)?).max(0.03);
    Some((cov, s))
}

/// Options for [`recover_photos`].
#[derive(Debug, Clone)]
pub struct V1Options {
    /// The tag's black square, mm.
    pub tag_mm: f64,
    /// Sensor width for EXIF without a 35 mm equivalent.
    pub sensor_w_mm: Option<f64>,
    /// Film of the fit.
    pub film: (usize, usize),
    /// Huber threshold, σ.
    pub huber: f64,
    /// The search's options.
    pub search: Options,
    /// Forward-difference columns for the occluded parameters.
    pub fd_occluded: bool,
    /// Shadow (secondary) edges.
    pub secondary: bool,
    /// Where the backdrop is.
    pub region: Region,
    /// The chromatic coverage term ([`Target::cov_obs`]).
    pub chroma: bool,
}

impl Default for V1Options {
    fn default() -> Self {
        Self {
            tag_mm: tag::TAG_MM,
            sensor_w_mm: None,
            film: (256, 192),
            huber: 1.5,
            search: Options { huber: Some(1.5), ..Options::default() },
            fd_occluded: true,
            secondary: true,
            region: Region::default(),
            chroma: true,
        }
    }
}

/// What v1 recovered, and from what.
pub struct V1 {
    /// The search's result (program, full-resolution fit, score, judge).
    pub rec: Recovery,
    /// The final look per view (with the fitted colour).
    pub looks: Vec<Look>,
    /// Per photo: calibration.
    pub calib: Vec<Calib>,
    /// Per photo: framing.
    pub framed: Vec<Framed>,
    /// The setup at full film resolution, with the final looks.
    pub setup: Setup,
    /// The target at full film resolution (gamma applied).
    pub target: Target,
    /// Noise σ (search, full).
    pub sigma: (f64, f64),
}

/// Mask pixels within `r` of `p`.
fn near_mask(m: &[f64], (w, h): (usize, usize), p: usize, r: i64) -> usize {
    let (x, y) = ((p % w) as i64, (p / w) as i64);
    let mut n = 0;
    for dy in -r..=r {
        for dx in -r..=r {
            let (xx, yy) = ((x + dx).clamp(0, w as i64 - 1), (y + dy).clamp(0, h as i64 - 1));
            n += (m[yy as usize * w + xx as usize] > 0.5) as usize;
        }
    }
    n
}

/// The residual's integrated autocorrelation over the `ok` pixels, as an
/// area in pixels: `(1 + 2 Σ ρ_x(k)) (1 + 2 Σ ρ_y(k))`, each sum stopped
/// where `ρ` drops under 0.05; between 1 and 64.
pub fn correlation_area(r: &[f64], ok: &[bool], (w, h): (usize, usize)) -> f64 {
    let n = ok.iter().filter(|&&o| o).count().max(1) as f64;
    let mean = r.iter().zip(ok).filter(|(_, o)| **o).map(|(x, _)| x).sum::<f64>() / n;
    let var = r.iter().zip(ok).filter(|(_, o)| **o).map(|(x, _)| (x - mean).powi(2)).sum::<f64>() / n;
    if var <= 0.0 {
        return 1.0;
    }
    let rho = |dx: usize, dy: usize| {
        let (mut s, mut c) = (0.0, 0.0);
        for y in 0..h - dy {
            for x in 0..w - dx {
                let (p, q) = (y * w + x, (y + dy) * w + x + dx);
                if ok[p] && ok[q] {
                    s += (r[p] - mean) * (r[q] - mean);
                    c += 1.0;
                }
            }
        }
        if c > 0.0 { s / (c * var) } else { 0.0 }
    };
    let side = |horiz: bool| {
        let mut t = 1.0;
        for k in 1..32 {
            let p = if horiz { rho(k, 0) } else { rho(0, k) };
            if p < 0.05 {
                break;
            }
            t += 2.0 * p;
        }
        t
    };
    (side(true) * side(false)).clamp(1.0, 64.0)
}

fn gamma_target(grey: &[f64], g: f64) -> Vec<f64> {
    grey.iter().map(|x| x.max(0.0).powf(1.0 / g)).collect()
}

/// The whole of v1 on `photos` (one or more of the same scene).
pub fn recover_photos(photos: &[Photo], o: &V1Options) -> Result<V1, String> {
    let mut calib = Vec::new();
    let mut framed = Vec::new();
    for p in photos {
        let c = calibrate(p, o.tag_mm, o.sensor_w_mm)?;
        let crop = locate(p, &c, &o.region)?;
        framed.push(frame(p, &c, crop, o.film, &o.region)?);
        calib.push(c);
    }
    let rig = Rig {
        views: framed.iter().map(|f| View { cam: f.cam, look: Look::default() }).collect(),
        ss: 2,
        fd_occluded: o.fd_occluded,
        secondary: o.secondary,
        shadow_samples: 4000,
    };
    let mut lin = Target::from_masks(
        framed.iter().map(|f| f.grey.clone()).collect(),
        framed.iter().map(|f| f.mask.clone()).collect(),
        framed.iter().map(|f| f.weight.clone()).collect(),
        o.film,
    );
    if o.chroma && framed.iter().all(|f| f.cov.is_some()) {
        lin = lin.with_cov(framed.iter().map(|f| f.cov.clone().unwrap().0).collect(), framed.iter().map(|f| f.cov.as_ref().unwrap().1).collect());
    }
    // Appearance from the base box, at half resolution.
    let half = lin.half_masks();
    let mut hrig = rig.scaled(0.5);
    let dbg = std::env::var("RECON_DEBUG").is_ok();
    if dbg {
        if let Ok(d) = std::env::var("RECON_DEBUG") {
            let (w, h) = o.film;
            let save = |name: &str, v: &[f64]| {
                let img = image::GrayImage::from_fn(w as u32, h as u32, |x, y| image::Luma([crate::render::srgb8(v[(y as usize) * w + x as usize])]));
                img.save(std::path::Path::new(&d).join(name)).ok();
            };
            save("dbg_target.png", &framed[0].grey);
            save("dbg_mask.png", &framed[0].mask);
            save("dbg_weight.png", &framed[0].weight);
        }
        eprintln!("crops {:?}", framed.iter().map(|f| f.crop).collect::<Vec<_>>());
    }
    // Every start the silhouettes allow (one per band of heights), each
    // lit and fitted on the image; the lowest robust cost wins. (A
    // silhouette-only settle first slides every start along the
    // depth–height valley to the same place, so there is none.)
    let starts = photo::init_bases(&half, &hrig);
    let mut best: Option<(f64, Program, Rig, Vec<f64>)> = None;
    let mut tried = Vec::new();
    let mut sref: Option<f64> = None;
    for start in starts {
        let base = start.clone();
        let mut r = hrig.clone();
        let mut sig = Vec::new();
        for vi in 0..r.views.len() {
            let w: Vec<f64> = (0..half.img[vi].len()).map(|p| half.w(vi, p)).collect();
            sig.push(photo::fit_look(&base, &mut r, vi, &half.img[vi], &w, photo::LookSearch { key_grid: true, nonlinear: true }));
        }
        // A short image fit under that light.
        let mut ht = half.clone();
        for (vi, img) in ht.img.iter_mut().enumerate() {
            *img = gamma_target(img, r.views[vi].look.gamma);
        }
        let s2 = sig.iter().sum::<f64>() / sig.len() as f64;
        let mut f = Fitter::with_sigma2(Setup::photo(r.clone(), 6000), ht, s2 * s2).robust(Some(o.huber));
        let Ok(fit) = f.fit(&base, &[Level { sigma: 1.5, lambda: 0.1, iters: 5 }, Level { sigma: 0.0, lambda: 0.0, iters: 5 }]) else { continue };
        // Judge every start at one σ (the first's), so the costs compare.
        let sj = *sref.get_or_insert(s2);
        let cost = fit.nll * s2 * s2 / (sj * sj);
        tried.push((fit.prog.to_text().replace('\n', "; "), cost));
        if best.as_ref().is_none_or(|b| cost < b.0) {
            best = Some((cost, fit.prog.clone(), r, sig));
        }
    }
    if dbg {
        eprintln!("starts {tried:?}");
    }
    let (_, base, hrig, sig_half) = best.ok_or("no start fitted")?;
    let looks: Vec<Look> = hrig.views.iter().map(|v| v.look).collect();
    if dbg {
        eprintln!("looks {looks:?} sigma {sig_half:?}");
    }
    let mut rig_full = rig.clone();
    for (v, l) in rig_full.views.iter_mut().zip(&looks) {
        v.look = *l;
    }
    let tgt = |t: &Target| -> Target {
        let mut t = t.clone();
        for (vi, img) in t.img.iter_mut().enumerate() {
            *img = gamma_target(img, looks[vi].gamma);
        }
        t
    };
    let target = tgt(&lin);
    let setup = Setup::photo(rig_full, 40_000);
    // σ at full resolution: the base box's residual under the fitted look.
    let mut sig_full = Vec::new();
    {
        let mut f = Fitter::with_sigma2(setup.clone(), target.clone(), 1.0);
        let e = f.eval(&base, false)?;
        for vi in 0..e.img.len() {
            let r: Vec<f64> = e.img[vi].iter().zip(&target.img[vi]).map(|(a, b)| a - b).collect();
            let w: Vec<f64> = (0..r.len()).map(|p| target.w(vi, p)).collect();
            sig_full.push(photo::mad_sigma(&r, &w));
        }
    }
    // The residual is not white: what the appearance model cannot do
    // (a window's reflection, interreflection, a tone curve) is smooth over
    // many pixels. Counting each pixel as independent would overcount the
    // nats such structure is worth, and features would be accepted to
    // explain it. The integrated autocorrelation of the residual on the
    // table, away from the part, is the number of pixels per independent
    // one; σ² is scaled by it (half as many per axis at half resolution).
    let kappa: f64;
    {
        let mut f = Fitter::with_sigma2(setup.clone(), target.clone(), 1.0);
        let e = f.eval(&base, false)?;
        let mut ks = Vec::new();
        for vi in 0..e.img.len() {
            let r: Vec<f64> = e.img[vi].iter().zip(&target.img[vi]).map(|(a, b)| a - b).collect();
            let ok: Vec<bool> = (0..r.len()).map(|p| target.w(vi, p) > 0.0 && near_mask(&framed[vi].mask, o.film, p, 4) == 0).collect();
            ks.push(correlation_area(&r, &ok, o.film));
        }
        kappa = ks.iter().sum::<f64>() / ks.len() as f64;
    }
    let kappa_half = (kappa / 4.0).max(1.0);
    if dbg {
        eprintln!("residual correlation area: {kappa:.1} px (half: {kappa_half:.1})");
    }
    let sh = sig_half.iter().sum::<f64>() / sig_half.len() as f64 * kappa_half.sqrt();
    let sf = sig_full.iter().sum::<f64>() / sig_full.len() as f64 * kappa.sqrt();
    let mut so = o.search.clone();
    so.sigma2 = Some((sh * sh, sf * sf));
    so.huber = Some(o.huber);
    so.base = Some(base.clone());
    let rec = search::recover(&target, &[], &setup, 0, &mut Algorithmic, &so);
    // Polish: the appearance on the recovered geometry (every knob), then
    // the geometry again under it.
    let mut rig_p = (**setup.photo.as_ref().unwrap()).clone();
    for (v, l) in rig_p.views.iter_mut().zip(&rec_looks(&rec, &setup)) {
        v.look = *l;
    }
    for vi in 0..rig_p.views.len() {
        let w: Vec<f64> = (0..lin.img[vi].len()).map(|p| lin.w(vi, p)).collect();
        photo::fit_look(&rec.fit.prog, &mut rig_p, vi, &lin.img[vi], &w, photo::LookSearch { key_grid: false, nonlinear: true });
    }
    let looks2: Vec<Look> = rig_p.views.iter().map(|v| v.look).collect();
    let mut target2 = lin.clone();
    for (vi, img) in target2.img.iter_mut().enumerate() {
        *img = gamma_target(img, looks2[vi].gamma);
    }
    let setup2 = Setup::photo(rig_p, 40_000);
    let mut f = Fitter::with_sigma2(setup2.clone(), target2.clone(), sf * sf).robust(Some(o.huber));
    let polished: Option<Fit> = f.fit(&rec.fit.prog, &[Level { sigma: 1.0, lambda: 0.0, iters: 3 }, Level { sigma: 0.0, lambda: 0.0, iters: 8 }]).ok();
    let mut rec = rec;
    if let Some(pf) = polished {
        let s = crate::mdl::score(&pf);
        rec.program = crate::canon::canonical(&pf.prog);
        rec.judge = crate::mdl::judge(&pf);
        rec.score = s;
        rec.fit = pf;
    }
    // Colour: each channel's linear solve on the final geometry; the
    // part's albedo per channel relative to the table's.
    let mut looks_final = looks2.clone();
    if let Ok(m) = photo::posed(&rec.fit.prog) {
        for (vi, fr) in framed.iter().enumerate() {
            let v = View { cam: setup2.photo.as_ref().unwrap().views[vi].cam, look: looks2[vi] };
            let b = photo::bases(&m, &v, 2);
            let w: Vec<f64> = (0..fr.grey.len()).map(|p| fr.weight[p]).collect();
            let mut rgb = [0.0; 3];
            for c in 0..3 {
                let t: Vec<f64> = fr.rgb.iter().map(|x| (x[c] as f64).max(0.0).powf(1.0 / looks2[vi].gamma)).collect();
                rgb[c] = look::solve_colour(&b, &t, &w, &looks2[vi]);
            }
            looks_final[vi].rgb = rgb;
        }
    }
    Ok(V1 { rec, looks: looks_final, calib, framed, setup: setup2, target: target2, sigma: (sh, sf) })
}

fn rec_looks(_rec: &Recovery, setup: &Setup) -> Vec<Look> {
    setup.photo.as_ref().unwrap().views.iter().map(|v| v.look).collect()
}

pub use crate::export::stl;

/// An overlay: the photo's crop, the model's silhouette and creases in
/// colour, and the residual beside it.
pub fn overlay(v: &V1, vi: usize) -> image::RgbImage {
    let fr = &v.framed[vi];
    let (w, h) = (fr.cam.size.0 as u32, fr.cam.size.1 as u32);
    let mut img = image::RgbImage::new(2 * w, h);
    let enc = crate::render::srgb8;
    for y in 0..h {
        for x in 0..w {
            let c = fr.rgb[(y * w + x) as usize];
            img.put_pixel(x, y, image::Rgb([enc(c[0] as f64), enc(c[1] as f64), enc(c[2] as f64)]));
            let r = v.rec.fit.residual[vi][(y * w + x) as usize];
            let t = (r / 4.0).clamp(-1.0, 1.0);
            let px = if t >= 0.0 { [255, (255.0 * (1.0 - t)) as u8, (255.0 * (1.0 - t)) as u8] } else { [(255.0 * (1.0 + t)) as u8, (255.0 * (1.0 + t)) as u8, 255] };
            img.put_pixel(w + x, y, image::Rgb(px));
        }
    }
    // The model's edges: where the coverage of neighbouring pixels differs.
    if let Ok(m) = photo::posed(&v.rec.fit.prog) {
        let cam = &fr.cam;
        let id = |x: f64, y: f64| -> i64 {
            let d = cam.dir(x, y);
            match m.bvh.trace_closest(&irys_core::Ray::new(cam.eye(), d)) {
                Some(hh) => {
                    let n = hh.normal.as_ref();
                    ((n.x * 4.0).round() as i64) * 100 + ((n.y * 4.0).round() as i64) * 10 + (n.z * 4.0).round() as i64 + 1000
                }
                None => 0,
            }
        };
        let ids: Vec<i64> = (0..(w * h) as usize).map(|p| id((p % w as usize) as f64 + 0.5, (p / w as usize) as f64 + 0.5)).collect();
        for y in 0..h as usize - 1 {
            for x in 0..w as usize - 1 {
                let p = y * w as usize + x;
                if ids[p] != ids[p + 1] || ids[p] != ids[p + w as usize] {
                    let col = if ids[p] == 0 || ids[p + 1] == 0 || ids[p + w as usize] == 0 { [0, 255, 80] } else { [255, 200, 0] };
                    img.put_pixel(x as u32, y as u32, image::Rgb(col));
                }
            }
        }
    }
    img
}
