//! The data engine: seeded scenes → irys photos and dense labels → shards
//! on disk → the network's targets.
//!
//! A sample is the scene (its JSON is the label set: every object's class,
//! program text, parameters and pose, the camera, the lights), the 8-bit
//! photo at the net's size, and per pixel the instance id and the metric
//! depth (an irys TLAS cast). `seed` alone determines all of it.

use std::io::{Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::render::{Caster, TABLE, lin};
use crate::scene::{CLASSES, Cam, NP, NTOK, Object, Scene, VOCAB, norm_mm, param_used};

/// Image size the net sees.
pub const W: usize = 128;
pub const H: usize = 96;
/// Patch side.
pub const PS: usize = 8;
pub const GW: usize = W / PS;
pub const GH: usize = H / PS;
/// Patches.
pub const P: usize = GW * GH;
/// Patch pixels.
pub const PIX: usize = PS * PS;
/// Patch input: RGB plus the pixel's two normalised coordinates.
pub const DIN: usize = PIX * 3;


/// One sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    pub scene: Scene,
    /// Program text, one line per object.
    pub program: String,
    /// Which objects in view act as a scale gauge.
    pub gauges: Vec<usize>,
    #[serde(skip)]
    pub rgb: Vec<u8>,
    #[serde(skip)]
    pub inst: Vec<u16>,
    #[serde(skip)]
    pub depth: Vec<f32>,
}

impl Sample {
    pub fn text(scene: &Scene) -> String {
        scene.objects.iter().map(Object::to_text).collect::<Vec<_>>().join("\n")
    }

    /// Labels for an already rendered photo.
    pub fn label(scene: Scene, rgb: Vec<u8>, meshes: &[crate::scene::Tris]) -> Sample {
        // Clutter occludes like anything else and is labelled as the table.
        let mut all = meshes.to_vec();
        all.extend(crate::render::world_meshes(&scene.clutter));
        let caster = Caster::new(&all);
        let cam = scene.cam.resized((W, H));
        let mut l = caster.labels(&cam);
        let n = meshes.len();
        for v in l.inst.iter_mut() {
            if *v != TABLE && *v as usize >= n {
                *v = TABLE;
            }
        }
        let gauges = scene
            .objects
            .iter()
            .enumerate()
            .filter(|(i, o)| o.class.gauge().is_some() && l.inst.iter().filter(|&&k| k as usize == *i).count() > 20)
            .map(|(i, _)| i)
            .collect();
        Sample { program: Self::text(&scene), scene, gauges, rgb, inst: l.inst, depth: l.depth }
    }

    pub fn linear(&self) -> Vec<[f64; 3]> {
        self.rgb.chunks_exact(3).map(|c| [lin(c[0]), lin(c[1]), lin(c[2])]).collect()
    }
}

/// Write samples as a shard: `<stem>.json` (scenes) and `<stem>.bin`
/// (photo, instance, depth per sample, fixed size).
pub fn write_shard(path: &Path, samples: &[Sample]) -> std::io::Result<()> {
    let js = serde_json::to_string(samples).unwrap();
    std::fs::write(path.with_extension("json"), js)?;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path.with_extension("bin"))?);
    for s in samples {
        f.write_all(&s.rgb)?;
        for &v in &s.inst {
            f.write_all(&v.to_le_bytes())?;
        }
        for &v in &s.depth {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

pub fn read_shard(path: &Path) -> std::io::Result<Vec<Sample>> {
    let mut samples: Vec<Sample> = serde_json::from_str(&std::fs::read_to_string(path.with_extension("json"))?).map_err(std::io::Error::other)?;
    let mut f = std::io::BufReader::new(std::fs::File::open(path.with_extension("bin"))?);
    let n = W * H;
    for s in &mut samples {
        let mut rgb = vec![0u8; n * 3];
        f.read_exact(&mut rgb)?;
        let mut b = vec![0u8; n * 2];
        f.read_exact(&mut b)?;
        let inst = b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let mut b = vec![0u8; n * 4];
        f.read_exact(&mut b)?;
        let depth = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        s.rgb = rgb;
        s.inst = inst;
        s.depth = depth;
    }
    Ok(samples)
}

/// All shards under `dir` whose name starts with `prefix`, in order.
pub fn read_dir(dir: &Path, prefix: &str) -> Vec<Sample> {
    let mut names: Vec<_> = std::fs::read_dir(dir)
        .map(|r| r.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "bin")).collect())
        .unwrap_or_default();
    names.retain(|p: &std::path::PathBuf| p.file_name().unwrap().to_string_lossy().starts_with(prefix));
    names.sort();
    names.iter().flat_map(|p| read_shard(p).expect("shard")).collect()
}

/// The patchified network input `[P, DIN]`: per patch, its 64 pixels'
/// sRGB in `[−1, 1]`, channel-minor.
pub fn patches(rgb: &[u8]) -> Vec<f64> {
    let mut x = vec![0.0; P * PIX * 3];
    for py in 0..GH {
        for px in 0..GW {
            let p = py * GW + px;
            for j in 0..PS {
                for i in 0..PS {
                    let k = (py * PS + j) * W + px * PS + i;
                    for c in 0..3 {
                        x[(p * PIX + j * PS + i) * 3 + c] = rgb[k * 3 + c] as f64 / 127.5 - 1.0;
                    }
                }
            }
        }
    }
    x
}

/// Where an object stands, in net pixels: the projection of its base centre.
pub fn centre_px(cam: &Cam, o: &Object) -> Option<[f64; 2]> {
    cam.project([o.pose[0], o.pose[1], 0.0])
}

/// Per-image targets and weights, laid out as the graph's loss inputs.
#[derive(Debug, Clone, Default)]
pub struct Targets {
    pub t_pix: Vec<f64>,
    pub w_pix: Vec<f64>,
    pub t_ctr: Vec<f64>,
    pub w_ctr: Vec<f64>,
    pub t_cls: Vec<f64>,
    pub w_cls: Vec<f64>,
    pub t_tok: Vec<f64>,
    pub w_tok: Vec<f64>,
    pub t_par: Vec<f64>,
    pub w_par: Vec<f64>,
    pub t_yaw: Vec<f64>,
    pub w_yaw: Vec<f64>,
    pub t_plane: Vec<f64>,
    pub w_plane: Vec<f64>,
}

/// Heat sigma (patches) for an object of projected radius `r_px` pixels.
pub fn heat_sigma(r_px: f64) -> f64 {
    (0.3 * r_px / PS as f64).clamp(0.6, 2.5)
}

/// The supervised targets of `s`, weights already divided by `batch`.
pub fn targets(s: &Sample, batch: usize) -> Targets {
    let cam = s.scene.cam.resized((W, H));
    let nb = batch as f64;
    let mut t = Targets {
        t_pix: vec![0.0; P * 4 * PIX],
        w_pix: vec![0.0; P * 4 * PIX],
        t_ctr: vec![0.0; P],
        w_ctr: vec![0.0; P],
        t_cls: vec![0.0; P * CLASSES],
        w_cls: vec![0.0; P],
        t_tok: vec![0.0; P * NTOK * VOCAB],
        w_tok: vec![0.0; P * NTOK],
        t_par: vec![0.0; P * NP],
        w_par: vec![0.0; P * NP],
        t_yaw: vec![0.0; P * 2],
        w_yaw: vec![0.0; P * 2],
        t_plane: cam.plane().to_vec(),
        w_plane: vec![1.0 / nb; 2],
    };
    let centres: Vec<Option<[f64; 2]>> = s.scene.objects.iter().map(|o| centre_px(&cam, o)).collect();
    let visible: Vec<usize> = (0..s.scene.objects.len()).map(|i| s.inst.iter().filter(|&&k| k as usize == i).count()).collect();
    let n_fg = s.inst.iter().filter(|&&k| k != TABLE).count().max(1) as f64;
    for y in 0..H {
        for x in 0..W {
            let k = y * W + x;
            let p = (y / PS) * GW + x / PS;
            let j = (y % PS) * PS + x % PS;
            let at = |c: usize| (p * 4 + c) * PIX + j;
            let id = s.inst[k];
            let fg = id != TABLE;
            t.t_pix[at(0)] = if fg { 1.0 } else { 0.0 };
            t.w_pix[at(0)] = 1.0 / (nb * (W * H) as f64) * 4.0;
            t.t_pix[at(3)] = (s.depth[k].max(1.0) as f64 / 1000.0).ln();
            t.w_pix[at(3)] = 0.5 / (nb * (W * H) as f64);
            if fg {
                if let Some(c) = centres[id as usize] {
                    t.t_pix[at(1)] = (c[0] - (x as f64 + 0.5)) / PS as f64;
                    t.t_pix[at(2)] = (c[1] - (y as f64 + 0.5)) / PS as f64;
                    t.w_pix[at(1)] = 0.5 / (nb * n_fg);
                    t.w_pix[at(2)] = 0.5 / (nb * n_fg);
                }
            }
        }
    }
    // Centre heat and attributes.
    let mut owner = vec![(f64::INFINITY, usize::MAX); P];
    for (i, o) in s.scene.objects.iter().enumerate() {
        let Some(c) = centres[i] else { continue };
        if visible[i] < 12 {
            continue;
        }
        let r_px = (visible[i] as f64 / std::f64::consts::PI).sqrt();
        let sg = heat_sigma(r_px);
        let (gx, gy) = (c[0] / PS as f64, c[1] / PS as f64);
        for p in 0..P {
            let (px, py) = ((p % GW) as f64 + 0.5, (p / GW) as f64 + 0.5);
            let d2 = (px - gx).powi(2) + (py - gy).powi(2);
            let h = (-d2 / (2.0 * sg * sg)).exp();
            if h > t.t_ctr[p] {
                t.t_ctr[p] = h;
            }
            if d2 < owner[p].0 && d2 < (1.0f64).max(sg * sg) {
                owner[p] = (d2, i);
            }
        }
        let _ = o;
    }
    for p in 0..P {
        t.w_ctr[p] = (1.0 + 9.0 * t.t_ctr[p]) / (nb * P as f64) * 2.0;
    }
    let mut n_attr = vec![0usize; s.scene.objects.len()];
    for p in 0..P {
        if owner[p].1 != usize::MAX {
            n_attr[owner[p].1] += 1;
        }
    }
    let n_obj = n_attr.iter().filter(|&&n| n > 0).count().max(1) as f64;
    for p in 0..P {
        let (_, i) = owner[p];
        if i == usize::MAX {
            continue;
        }
        let o = &s.scene.objects[i];
        let w = 1.0 / (nb * n_obj * n_attr[i] as f64);
        t.t_cls[p * CLASSES + o.class as usize] = 1.0;
        t.w_cls[p] = w;
        let tm = o.token_mask();
        for k in 0..NTOK {
            t.t_tok[(p * NTOK + k) * VOCAB + o.tokens[k].min(VOCAB - 1)] = 1.0;
            t.w_tok[p * NTOK + k] = if tm[k] { w } else { 0.0 };
        }
        for k in 0..NP {
            if k < o.params.len() && param_used(o.family, &o.tokens, k) {
                t.t_par[p * NP + k] = norm_mm(o.params[k]);
                t.w_par[p * NP + k] = 0.3 * w;
            }
        }
        if o.family.has_yaw() {
            let a = 2.0 * o.pose[2].to_radians();
            t.t_yaw[p * 2] = a.cos();
            t.t_yaw[p * 2 + 1] = a.sin();
            t.w_yaw[p * 2] = 0.5 * w;
            t.w_yaw[p * 2 + 1] = 0.5 * w;
        }
    }
    t
}

/// The data engine: render `n` samples starting at `seed0` (GPU feature).
#[cfg(feature = "gpu")]
pub fn generate(ctx: &'static irys_gpu::GpuContext, seed0: u64, n: usize, domain: crate::scene::Domain, spp: u32) -> Vec<Sample> {
    use crate::render::{photo, world_meshes};
    use crate::scene::sample_scene;
    use rayon::prelude::*;
    let mut out = Vec::with_capacity(n);
    for chunk in (0..n as u64).collect::<Vec<_>>().chunks(32) {
        // CPU (scenes, meshes) in parallel; the device one frame at a time;
        // labels in parallel again.
        let built: Vec<_> = chunk
            .par_iter()
            .map(|&k| {
                let scene = sample_scene(seed0 + k, domain, (W, H));
                let meshes = world_meshes(&scene.objects);
                (scene, meshes)
            })
            .collect();
        let shot: Vec<_> = built
            .into_iter()
            .filter_map(|(scene, meshes)| match photo(ctx, &scene, &meshes, (W, H), spp) {
                Ok(i) => Some((scene, meshes, i)),
                Err(e) => {
                    eprintln!("seed {}: {e}", scene.seed);
                    None
                }
            })
            .collect();
        out.extend(shot.into_par_iter().map(|(scene, meshes, img)| Sample::label(scene, img.into_raw(), &meshes)).collect::<Vec<_>>());
    }
    out
}

/// v3: the photo at `hi ×` the net's size as well (the compare image's
/// target), and the net's 128 × 96 input area-averaged from it. Returns the
/// samples and their high-resolution photos.
#[cfg(feature = "gpu")]
pub fn generate_hi(ctx: &'static irys_gpu::GpuContext, seed0: u64, n: usize, domain: crate::scene::Domain, spp: u32, hi: usize) -> Vec<(Sample, image::RgbImage)> {
    use crate::render::{photo, world_meshes};
    use crate::scene::sample_scene;
    use rayon::prelude::*;
    let mut out = Vec::with_capacity(n);
    for chunk in (0..n as u64).collect::<Vec<_>>().chunks(16) {
        let built: Vec<_> = chunk
            .par_iter()
            .map(|&k| {
                let scene = sample_scene(seed0 + k, domain, (W, H));
                let meshes = world_meshes(&scene.objects);
                (scene, meshes)
            })
            .collect();
        let shot: Vec<_> = built
            .into_iter()
            .filter_map(|(scene, meshes)| match photo(ctx, &scene, &meshes, (W * hi, H * hi), spp) {
                Ok(i) => Some((scene, meshes, i)),
                Err(e) => {
                    eprintln!("seed {}: {e}", scene.seed);
                    None
                }
            })
            .collect();
        out.extend(
            shot.into_par_iter()
                .map(|(scene, meshes, img)| {
                    let small = downsample_rgb8(&img, (W, H));
                    (Sample::label(scene, small, &meshes), img)
                })
                .collect::<Vec<_>>(),
        );
    }
    out
}

/// Area-average an 8-bit sRGB photo to `size` (in linear light), back to 8 bits.
pub fn downsample_rgb8(img: &image::RgbImage, size: (usize, usize)) -> Vec<u8> {
    crate::render::resample_rgb8(img, size).iter().flat_map(|p| p.map(irys_recon::render::srgb8)).collect()
}

/// The high-resolution photo of `seed` written by `neural_data --hi`.
pub fn hi_path(dir: &Path, prefix: &str, seed: u64) -> std::path::PathBuf {
    dir.join(format!("{prefix}-hi")).join(format!("{seed}.png"))
}
