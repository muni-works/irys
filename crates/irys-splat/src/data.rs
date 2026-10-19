//! Posed image sets: OpenCV pinhole intrinsics (`camera_info.json`, through
//! `look::pinhole`) and one camera → world pose per line of `poses.jsonl`.
//!
//! ipse `docs/CAPTURE.md`'s export, `<capture>.train/` (`camera_info.json`,
//! `poses.jsonl` lines `{frame, t, image: "images/N.png", depth?:
//! "depth/N.u16", cam_pos, cam_rot}` with `cam_rot` row-major world ←
//! optical, `images/`, `depth/`, `align.json`), is read as is: names in a
//! line resolve against the given directory or else the poses file's own.
//!
//! A pose line is a JSON object with either
//! - `cam_pos` `[x, y, z]` and `cam_rot` (3 × 3 camera → world, row-major:
//!   nine numbers or three rows) — the K1 sweep format; or
//! - a 4 × 4 camera → world matrix under `T_world_cam`, `c2w`, `pose`,
//!   `transform` or `matrix` (sixteen numbers or four rows).
//!
//! The image is found, in order, from `image` / `file` / `jpeg` / `path`
//! (relative to the images directory); else `<frame:06>.nv12` there (the
//! K1's raw colour frames, BT.601 full range); else, when the line has `t` and a
//! `jpegs.jsonl` (`{frame, t}` per image) sits beside the images directory,
//! the image nearest in time; else `<frame:06>.jpg|.png` in the directory.
//! Depth (optional) is `depth` (a file relative to the depth directory) or
//! `<frame:06>.u16` there: mono16 millimetres (`DepthSpec::scale` changes
//! the unit), or `.f32` metres; out-of-range values are invalid.

use std::path::{Path, PathBuf};

use irys_diff::look::pinhole::Pinhole;
use serde_json::Value;

/// One posed frame.
#[derive(Debug, Clone)]
pub struct View {
    pub name: String,
    pub cam: Pinhole,
    /// sRGB in 0..1, row-major.
    pub rgb: Vec<[f32; 3]>,
    /// Optical depth (metres; 0: invalid), when given.
    pub depth: Option<Vec<f32>>,
}

/// Depth decoding.
#[derive(Debug, Clone, Copy)]
pub struct DepthSpec {
    /// Metres per unit of a `.u16` file.
    pub scale: f64,
    pub min: f64,
    pub max: f64,
}

impl Default for DepthSpec {
    fn default() -> Self {
        Self { scale: 1e-3, min: 0.1, max: 20.0 }
    }
}

fn nums(v: &Value) -> Option<Vec<f64>> {
    let a = v.as_array()?;
    let mut out = Vec::new();
    for x in a {
        if let Some(r) = x.as_array() {
            for y in r {
                out.push(y.as_f64()?);
            }
        } else {
            out.push(x.as_f64()?);
        }
    }
    Some(out)
}

/// The camera → world `(rot, pos)` of a pose line.
pub fn pose_of(v: &Value) -> Result<([[f64; 3]; 3], [f64; 3]), String> {
    if let (Some(p), Some(r)) = (v.get("cam_pos").and_then(nums), v.get("cam_rot").and_then(nums)) {
        if p.len() != 3 || r.len() != 9 {
            return Err("cam_pos needs 3 numbers and cam_rot 9".into());
        }
        return Ok((std::array::from_fn(|i| std::array::from_fn(|j| r[3 * i + j])), [p[0], p[1], p[2]]));
    }
    for k in ["T_world_cam", "c2w", "pose", "transform", "matrix"] {
        if let Some(m) = v.get(k).and_then(nums) {
            if m.len() != 16 {
                return Err(format!("{k} needs 16 numbers"));
            }
            return Ok((std::array::from_fn(|i| std::array::from_fn(|j| m[4 * i + j])), [m[3], m[7], m[11]]));
        }
    }
    Err("pose line has neither cam_pos + cam_rot nor a 4×4 matrix".into())
}

/// Decode an image file to sRGB 0..1 at `(w, h)` (refused if the size differs).
pub fn load_rgb(path: &Path, w: usize, h: usize) -> Result<Vec<[f32; 3]>, String> {
    if path.extension().is_some_and(|e| e == "nv12") {
        return load_nv12(path, w, h);
    }
    let im = image::open(path).map_err(|e| format!("{}: {e}", path.display()))?.to_rgb8();
    if im.width() as usize != w || im.height() as usize != h {
        return Err(format!("{}: {}×{}, camera says {w}×{h}", path.display(), im.width(), im.height()));
    }
    Ok(im.pixels().map(|p| p.0.map(|c| c as f32 / 255.0)).collect())
}

/// NV12 (the K1's raw colour frames): a Y plane, then interleaved CbCr at
/// half resolution; BT.601 full range (JFIF; the sweep's Y spans 0–255 and
/// matches its JPEGs' mean) → sRGB.
pub fn load_nv12(path: &Path, w: usize, h: usize) -> Result<Vec<[f32; 3]>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if b.len() < w * h * 3 / 2 {
        return Err(format!("{}: {} bytes, want {}", path.display(), b.len(), w * h * 3 / 2));
    }
    let (y, uv) = b.split_at(w * h);
    Ok((0..w * h)
        .map(|p| {
            let (x, r) = (p % w, p / w);
            let q = (r / 2) * w + (x / 2) * 2;
            let yy = y[p] as f32;
            let cb = uv[q] as f32 - 128.0;
            let cr = uv[q + 1] as f32 - 128.0;
            let c = [yy + 1.402 * cr, yy - 0.344_136 * cb - 0.714_136 * cr, yy + 1.772 * cb];
            c.map(|v| (v / 255.0).clamp(0.0, 1.0))
        })
        .collect())
}

pub fn load_depth(path: &Path, n: usize, spec: DepthSpec) -> Result<Vec<f32>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let raw: Vec<f64> = if path.extension().is_some_and(|e| e == "f32") {
        b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect()
    } else {
        b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as f64 * spec.scale).collect()
    };
    if raw.len() != n {
        return Err(format!("{}: {} values, want {n}", path.display(), raw.len()));
    }
    Ok(raw.into_iter().map(|z| if z.is_finite() && z >= spec.min && z <= spec.max { z as f32 } else { 0.0 }).collect())
}

/// A file named in a pose line: relative to `dir` if it is there, else to
/// the poses file's directory (ipse CAPTURE.md's `<capture>.train/` names
/// `images/…` and `depth/…` relative to itself).
fn resolve(s: &str, dir: &Path, root: &Path) -> PathBuf {
    let a = dir.join(s);
    if a.exists() { a } else { root.join(s) }
}

/// Read `poses.jsonl` against `images` (and an optional depth directory).
pub fn load(images: &Path, poses: &Path, camera: &Path, depth: Option<&Path>, spec: DepthSpec) -> Result<Vec<View>, String> {
    let k = Pinhole::from_camera_info(&std::fs::read_to_string(camera).map_err(|e| format!("{}: {e}", camera.display()))?)?;
    let (w, h) = (k.width, k.height);
    let text = std::fs::read_to_string(poses).map_err(|e| format!("{}: {e}", poses.display()))?;
    // optional image timestamps beside the image directory
    let times: Vec<(f64, u64)> = images
        .parent()
        .map(|p| p.join("jpegs.jsonl"))
        .filter(|p| p.exists())
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|t| {
            t.lines()
                .filter_map(|l| serde_json::from_str::<Value>(l).ok())
                .filter_map(|v| Some((v["t"].as_f64()?, v["frame"].as_u64()?)))
                .collect()
        })
        .unwrap_or_default();
    let root = poses.parent().unwrap_or(Path::new("."));
    let mut out = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let at = |e: String| format!("{}:{}: {e}", poses.display(), ln + 1);
        let v: Value = serde_json::from_str(line).map_err(|e| at(e.to_string()))?;
        let (rot, pos) = pose_of(&v).map_err(at)?;
        let frame = v["frame"].as_u64();
        let named = ["image", "file", "jpeg", "path"].iter().find_map(|k| v.get(*k).and_then(Value::as_str)).map(|s| resolve(s, images, root));
        let nv12 = frame.map(|f| images.join(format!("{f:06}.nv12"))).filter(|p| p.exists());
        let img: PathBuf = if let Some(p) = named {
            p
        } else if let Some(p) = nv12 {
            p
        } else if let (Some(t), false) = (v["t"].as_f64(), times.is_empty()) {
            let best = times.iter().min_by(|a, b| (a.0 - t).abs().total_cmp(&(b.0 - t).abs())).unwrap();
            images.join(format!("{:06}.jpg", best.1))
        } else {
            let f = frame.ok_or_else(|| at("no image, t or frame".into()))?;
            let j = images.join(format!("{f:06}.jpg"));
            if j.exists() { j } else { images.join(format!("{f:06}.png")) }
        };
        let rgb = load_rgb(&img, w, h)?;
        let dep = match depth {
            None => None,
            Some(d) => {
                let file = v.get("depth").and_then(Value::as_str).map(|s| resolve(s, d, root)).or_else(|| {
                    let f = frame?;
                    [format!("{f:06}.u16"), format!("{f:06}.f32")].into_iter().map(|n| d.join(n)).find(|p| p.exists())
                });
                match file {
                    Some(p) => Some(load_depth(&p, w * h, spec)?),
                    None => None,
                }
            }
        };
        let name = img.file_name().map_or(format!("{ln}"), |s| s.to_string_lossy().into_owned());
        out.push(View { name, cam: k.at(rot, pos), rgb, depth: dep });
    }
    Ok(out)
}

/// Write a view set in the same format (`images/<i:06>.png`, `poses.jsonl`,
/// `camera_info.json`, optional `depth/<i:06>.f32`), for synthetic sets.
pub fn save(dir: &Path, views: &[View]) -> Result<(), String> {
    let e = |e: std::io::Error| e.to_string();
    std::fs::create_dir_all(dir.join("images")).map_err(e)?;
    let k = &views[0].cam;
    let info = serde_json::json!({
        "width": k.width, "height": k.height,
        "k": [k.fx, 0.0, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0],
        "d": [0.0, 0.0, 0.0, 0.0, 0.0], "distortion_model": "plumb_bob",
    });
    std::fs::write(dir.join("camera_info.json"), serde_json::to_string_pretty(&info).unwrap()).map_err(e)?;
    let mut lines = String::new();
    for (i, v) in views.iter().enumerate() {
        let px: Vec<u8> = v.rgb.iter().flat_map(|c| c.map(|x| (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)).collect();
        let name = format!("{i:06}.png");
        image::save_buffer(dir.join("images").join(&name), &px, k.width as u32, k.height as u32, image::ColorType::Rgb8)
            .map_err(|e| e.to_string())?;
        let mut line = serde_json::json!({
            "frame": i, "image": name, "cam_pos": v.cam.pos,
            "cam_rot": v.cam.rot.iter().flatten().collect::<Vec<_>>(),
        });
        if let Some(d) = &v.depth {
            std::fs::create_dir_all(dir.join("depth")).map_err(e)?;
            let dn = format!("{i:06}.f32");
            let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(dir.join("depth").join(&dn), b).map_err(e)?;
            line["depth"] = dn.into();
        }
        lines += &line.to_string();
        lines.push('\n');
    }
    std::fs::write(dir.join("poses.jsonl"), lines).map_err(e)
}
