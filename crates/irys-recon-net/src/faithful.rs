//! v3's compare image: the scene as irys's OpenPBR scene kernel renders it
//! (NEE, shadows, the light tree, the world), with its appearance fitted to
//! the photo, in place of v2's linear `c₀ + c·n` on a cast.
//!
//! - [`Compare3`] holds one scene on the device. Meshes are uploaded in the
//!   object's own frame and placed by the TLAS instance transform, so a
//!   pose move is [`GpuLook::set_instances`] (no geometry re-upload); the
//!   light and the substances are [`GpuLook::set_env`], `set_sun` and
//!   `set_materials` (no rebuild). Only a change of shape rebuilds.
//! - [`Appearance`] is what is fitted: per-object OpenPBR base colour,
//!   roughness and metalness, the table's, the world as a non-negative
//!   combination of [`LOBES`] smooth lobes, one key light (direction, size,
//!   colour) and the photo's tone gamma. Exposure is the lights' scale.
//! - [`fit_appearance`] fits it: lighting by non-negative least squares
//!   over basis renders (the transport is linear in the emission), a grid
//!   over the key's direction and size, the gamma by a grid, then per-object
//!   colour by the closed-form scale on the object's pixels (instance ids
//!   from M1's sensor kernel) and roughness / metal by a choice among three.
//!
//! The comparison is in linear light against the photo linearised with the
//! fitted gamma.

use irys_diff::look::openpbr::OpenPbr;
use irys_diff::look::scene::{Camera as LCam, Equirect, Instance as LInst, LookScene, Material, Mesh, Sun, mask};
use irys_diff::look::scene_gpu::{GpuLook, Opts};
use irys_diff::look::sensor::{MISS, Sensor, aov};

use crate::scene::{Cam, Mat, Object, Tris};

/// A photo channel at or above this (8-bit) is saturated.
pub const SAT8: u8 = 254;
/// The same in linear light (for any gamma near 1).
pub const SAT: f64 = 0.985;

/// World lobes: a broad sky, the ground, and 30 directional lobes
/// (`cos⁸` about centres at elevations 75°, 45°, 20° and 5°), enough to
/// carry a window or a softbox as well as the sky's gradient.
pub const LOBES: usize = 32;
/// World texel size.
pub const ENV: (u32, u32) = (96, 48);

fn lobe_centres() -> Vec<[f64; 3]> {
    let mut v = Vec::new();
    for (el, n) in [(75.0f64, 4), (45.0, 8), (20.0, 10), (5.0, 8)] {
        for k in 0..n {
            let az = (k as f64 + if n % 4 == 0 { 0.5 } else { 0.0 }) * std::f64::consts::TAU / n as f64;
            let e = el.to_radians();
            v.push([e.cos() * az.cos(), e.cos() * az.sin(), e.sin()]);
        }
    }
    v
}

/// Lobe `k` at elevation `el` and azimuth `az` (radians).
pub fn lobe(k: usize, el: f64, az: f64) -> f64 {
    let up = el > 0.0;
    match k {
        0 => if up { el.sin() } else { 0.0 },
        1 => if up { 0.0 } else { 1.0 },
        _ => {
            if !up {
                return 0.0;
            }
            thread_local! { static C: Vec<[f64; 3]> = lobe_centres(); }
            let d = [el.cos() * az.cos(), el.cos() * az.sin(), el.sin()];
            C.with(|c| {
                let m = c[k - 2];
                (d[0] * m[0] + d[1] * m[1] + d[2] * m[2]).max(0.0).powi(8)
            })
        }
    }
}

/// One key light.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Key {
    /// Toward the light (unit).
    pub dir: [f64; 3],
    /// Angular radius, radians.
    pub radius: f64,
    /// Irradiance it delivers at normal incidence, linear RGB.
    pub rgb: [f64; 3],
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Appearance {
    /// Per object, in scene order.
    pub mats: Vec<Mat>,
    pub table: Mat,
    /// World lobe weights, linear RGB.
    pub env: [[f64; 3]; LOBES],
    pub key: Key,
    /// The photo's tone gamma (display = srgb(linear)^gamma).
    pub gamma: f64,
    /// The lens's defocus: a Gaussian σ as a fraction of the image width
    /// (the model image is blurred by it before any comparison).
    #[serde(default)]
    pub defocus: f64,
}

impl Appearance {
    pub fn grey(n: usize) -> Appearance {
        let g = Mat { rgb: [0.5; 3], rough: 0.5, metal: false };
        Appearance {
            mats: vec![g; n],
            table: g,
            env: [[0.3; 3]; LOBES],
            key: Key { dir: crate::scene::normalize([0.3, -0.3, 0.9]), radius: 5f64.to_radians(), rgb: [1.0; 3] },
            gamma: 1.0,
            defocus: 0.0,
        }
    }
}

/// The world for lobe weights `w`.
pub fn env_of(w: &[[f64; 3]; LOBES]) -> Equirect {
    let (ew, eh) = ENV;
    let mut rgb = Vec::with_capacity((ew * eh) as usize);
    for j in 0..eh {
        for i in 0..ew {
            let theta = (j as f64 + 0.5) / eh as f64 * std::f64::consts::PI;
            let phi = (i as f64 + 0.5) / ew as f64 * std::f64::consts::TAU;
            let el = std::f64::consts::FRAC_PI_2 - theta;
            let mut c = [0.0f64; 3];
            for (k, wk) in w.iter().enumerate() {
                let l = lobe(k, el, phi);
                for ch in 0..3 {
                    c[ch] += wk[ch] * l;
                }
            }
            rgb.push([c[0] as f32, c[1] as f32, c[2] as f32]);
        }
    }
    Equirect { size: ENV, rgb }
}

fn sun_of(k: &Key) -> Sun {
    let s = std::f64::consts::PI * k.radius.sin().powi(2);
    Sun { dir: k.dir, radius: k.radius, radiance: k.rgb.map(|v| v.max(0.0) / s) }
}

pub fn material(m: &Mat, name: &str) -> Material {
    Material::new(
        name,
        OpenPbr {
            base_color: m.rgb.map(|v| v.clamp(0.0, 1.0)),
            base_metalness: if m.metal { 1.0 } else { 0.0 },
            specular_roughness: m.rough,
            specular_ior: 1.5,
            ..OpenPbr::default()
        },
    )
}

/// The irys camera of `cam` at `size`.
pub fn lcam(cam: &Cam, size: (usize, usize)) -> LCam {
    let c = cam.resized(size);
    let (f, r, u, ty) = c.frame();
    LCam { eye: c.eye, forward: f, right: r, up: u, tan_half: ty }
}

fn xf(pose: [f64; 3]) -> [[f32; 4]; 3] {
    let (s, c) = pose[2].to_radians().sin_cos();
    [[c as f32, -s as f32, 0.0, pose[0] as f32], [s as f32, c as f32, 0.0, pose[1] as f32], [0.0, 0.0, 1.0, 0.0]]
}

/// One scene on the device.
pub struct Compare3 {
    pub ctx: &'static irys_gpu::GpuContext,
    pub size: (usize, usize),
    look: GpuLook,
    sensor: Sensor,
    /// Instance slot → object index (the table's slot is `usize::MAX`).
    pub slot_obj: Vec<usize>,
    /// Object → its mesh index (`None`: no mesh / absent).
    mesh_of: Vec<Option<usize>>,
    table_mesh: usize,
    n_obj: usize,
}

impl Compare3 {
    /// `meshes[i]` is object `i`'s mesh in its own frame (empty: absent).
    pub fn new(ctx: &'static irys_gpu::GpuContext, objs: &[Object], meshes: &[Tris], app: &Appearance, cam: &Cam, size: (usize, usize)) -> Option<Compare3> {
        let mut lm = Vec::new();
        let mut inst = Vec::new();
        let mut mats = Vec::new();
        let mut slot_obj = Vec::new();
        let mut mesh_of = vec![None; objs.len()];
        for (i, (o, m)) in objs.iter().zip(meshes).enumerate() {
            mats.push(material(&app.mats[i], &format!("obj{i}")));
            if m.t.is_empty() {
                continue;
            }
            lm.push(Mesh {
                positions: m.p.iter().map(|q| [q[0] as f32, q[1] as f32, q[2] as f32]).collect(),
                normals: Vec::new(),
                uvs: Vec::new(),
                indices: m.t.iter().flatten().copied().collect(),
                material: i,
            });
            mesh_of[i] = Some(lm.len() - 1);
            inst.push(LInst { mesh: lm.len() - 1, xf: xf(o.pose), mask: mask::GEOMETRY });
            slot_obj.push(i);
        }
        mats.push(material(&app.table, "table"));
        let half = 20000.0f32;
        lm.push(Mesh {
            positions: vec![[-half, -half, 0.0], [half, -half, 0.0], [half, half, 0.0], [-half, half, 0.0]],
            normals: vec![[0.0, 0.0, 1.0]; 4],
            uvs: Vec::new(),
            indices: vec![0, 1, 2, 0, 2, 3],
            material: mats.len() - 1,
        });
        let table_mesh = lm.len() - 1;
        inst.push(LInst { mesh: table_mesh, xf: xf([0.0; 3]), mask: mask::GEOMETRY });
        slot_obj.push(usize::MAX);
        let ls = LookScene {
            meshes: lm,
            instances: inst,
            materials: mats,
            textures: Vec::new(),
            env: Some(env_of(&app.env)),
            sun: Some(sun_of(&app.key)),
            portal_skies: Vec::new(),
            camera: lcam(cam, size),
            size,
            frames: Vec::new(),
            fps: 24.0,
            dropped: Vec::new(),
        };
        let look = GpuLook::new(ctx, &ls)?;
        let sensor = Sensor::new(&look, size, 4);
        Some(Compare3 { ctx, size, look, sensor, slot_obj, mesh_of, table_mesh, n_obj: objs.len() })
    }

    /// Re-place objects (TLAS only); `present[i]` false drops object `i`.
    pub fn set_poses(&mut self, objs: &[Object], present: &[bool]) {
        let mut inst = Vec::new();
        let mut slot_obj = Vec::new();
        for (i, o) in objs.iter().enumerate() {
            if let (Some(m), true) = (self.mesh_of[i], present[i]) {
                inst.push(LInst { mesh: m, xf: xf(o.pose), mask: mask::GEOMETRY });
                slot_obj.push(i);
            }
        }
        inst.push(LInst { mesh: self.table_mesh, xf: xf([0.0; 3]), mask: mask::GEOMETRY });
        slot_obj.push(usize::MAX);
        self.look.set_instances(&inst).expect("instances");
        self.slot_obj = slot_obj;
    }

    pub fn set_appearance(&mut self, app: &Appearance) {
        let mut mats: Vec<Material> = (0..self.n_obj).map(|i| material(&app.mats[i], "o")).collect();
        mats.push(material(&app.table, "table"));
        self.look.set_materials(&mats);
        self.look.set_env(&env_of(&app.env));
        self.look.set_sun(&sun_of(&app.key));
    }

    /// An exact world and key (the oracle's), scaled by `k`.
    pub fn set_exact(&mut self, e: &Equirect, sun: &Sun, k: f64) {
        let e = Equirect { size: e.size, rgb: e.rgb.iter().map(|c| c.map(|v| v * k as f32)).collect() };
        self.look.set_env(&e);
        self.look.set_sun(&Sun { radiance: sun.radiance.map(|v| v * k), ..*sun });
    }

    pub fn set_env(&mut self, w: &[[f64; 3]; LOBES]) {
        self.look.set_env(&env_of(w));
    }
    pub fn set_key(&mut self, k: &Key) {
        self.look.set_sun(&sun_of(k));
    }
    pub fn set_mats(&mut self, objs: &[Mat], table: &Mat) {
        let mut mats: Vec<Material> = objs.iter().map(|m| material(m, "o")).collect();
        mats.push(material(table, "table"));
        self.look.set_materials(&mats);
    }

    fn opts(spp: u32, seed: u64) -> Opts {
        Opts { spp, max_bounces: 3, clamp: 6.0, seed, ..Opts::default() }
    }

    /// Linear RGB, `size`.
    pub fn render(&self, cam: &Cam, spp: u32, seed: u64) -> Vec<[f64; 3]> {
        let fr = self.look.render(&lcam(cam, self.size), &Self::opts(spp, seed));
        to_px(&fr.rgb)
    }

    /// Every camera in one submit (M1's `render_batch`).
    pub fn render_batch(&self, cams: &[Cam], spp: u32, seed: u64) -> Vec<Vec<[f64; 3]>> {
        let lc: Vec<LCam> = cams.iter().map(|c| lcam(c, self.size)).collect();
        self.look.render_batch(&lc, &Self::opts(spp, seed), false).into_iter().map(|f| to_px(&f.rgb)).collect()
    }

    /// Object index per pixel (`usize::MAX`: table, `usize::MAX - 1`: miss),
    /// one unjittered ray per pixel through M1's sensor kernel.
    pub fn ids(&mut self, cam: &Cam) -> Vec<usize> {
        let a = self.sensor.render(&[lcam(cam, self.size)], aov::INST);
        a.inst.unwrap().iter().map(|&k| if k == MISS || k as usize >= self.slot_obj.len() { usize::MAX - 1 } else { self.slot_obj[k as usize] }).collect()
    }
}

fn to_px(rgb: &[f32]) -> Vec<[f64; 3]> {
    rgb.chunks_exact(3).map(|c| [c[0] as f64, c[1] as f64, c[2] as f64]).map(|c| c.map(|v| if v.is_finite() { v } else { 0.0 })).collect()
}

/// The photo (8-bit sRGB) linearised under tone gamma `g`.
pub fn linearise(rgb8: &[u8], g: f64) -> Vec<[f64; 3]> {
    let lut: Vec<f64> = (0..256).map(|v| irys_recon::look::srgb_to_linear((v as f64 / 255.0).powf(1.0 / g))).collect();
    rgb8.chunks_exact(3).map(|c| [lut[c[0] as usize], lut[c[1] as usize], lut[c[2] as usize]]).collect()
}

/// Display value of a linear model pixel under gamma `g`.
pub fn display(v: f64, g: f64) -> f64 {
    let x = v.clamp(0.0, 1.0);
    let s = if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
    s.powf(g)
}

/// PSNR (dB) of the model's display image against the 8-bit photo.
pub fn psnr(model: &[[f64; 3]], rgb8: &[u8], g: f64) -> f64 {
    let mut se = 0.0;
    for (m, t) in model.iter().zip(rgb8.chunks_exact(3)) {
        for c in 0..3 {
            let d = display(m[c], g) - t[c] as f64 / 255.0;
            se += d * d;
        }
    }
    let mse = se / (model.len() * 3) as f64;
    -10.0 * mse.max(1e-12).log10()
}

/// Non-negative least squares on the normal equations (`a` symmetric).
pub fn nnls(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for _ in 0..400 {
        let mut moved = 0.0f64;
        for k in 0..n {
            if a[k][k] <= 1e-18 {
                continue;
            }
            let r: f64 = b[k] - (0..n).map(|j| a[k][j] * x[j]).sum::<f64>();
            let nx = (x[k] + r / a[k][k]).max(0.0);
            moved = moved.max((nx - x[k]).abs() / (x[k].abs() + 1e-6));
            x[k] = nx;
        }
        if moved < 1e-7 {
            break;
        }
    }
    x
}

/// Fitting knobs.
#[derive(Debug, Clone, Copy)]
pub struct FitOpts {
    pub spp: u32,
    pub seed: u64,
    /// Outer rounds (lights → colours → materials).
    pub rounds: usize,
    /// Fit the per-object roughness / metal choice.
    pub materials: bool,
}

impl Default for FitOpts {
    fn default() -> Self {
        FitOpts { spp: 8, seed: 7, rounds: 2, materials: true }
    }
}

/// Key directions tried: elevation × azimuth.
pub fn key_grid() -> Vec<[f64; 3]> {
    let mut v = Vec::new();
    for el in [12.0f64, 25.0, 40.0, 55.0, 70.0] {
        let n = if el > 60.0 { 6 } else { 12 };
        for k in 0..n {
            let az = (k as f64 * 360.0 / n as f64).to_radians();
            let e = el.to_radians();
            v.push([e.cos() * az.cos(), e.cos() * az.sin(), e.sin()]);
        }
    }
    v.push([0.0, 0.0, 1.0]);
    v
}

/// Per-object colour: scale each object's (and the table's) base colour so
/// its pixels match in the least-squares sense.
fn rescale_colours(model: &[[f64; 3]], target: &[[f64; 3]], ids: &[usize], app: &mut Appearance) {
    let n = app.mats.len();
    let mut num = vec![[0.0; 3]; n + 1];
    let mut den = vec![[0.0; 3]; n + 1];
    for p in 0..model.len() {
        let k = match ids[p] {
            i if i < n => i,
            usize::MAX => n,
            _ => continue,
        };
        for c in 0..3 {
            if target[p][c] >= SAT && model[p][c] >= SAT {
                continue; // both clipped: says nothing
            }
            num[k][c] += model[p][c] * target[p][c];
            den[k][c] += model[p][c] * model[p][c];
        }
    }
    for k in 0..=n {
        let m = if k < n { &mut app.mats[k] } else { &mut app.table };
        for c in 0..3 {
            if den[k][c] > 1e-12 {
                m.rgb[c] = (m.rgb[c] * num[k][c] / den[k][c]).clamp(0.003, 0.98);
            }
        }
    }
}

/// What [`fit_appearance`] reports.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FitReport {
    pub renders: usize,
    pub seconds: f64,
    /// Linear-light RMS of the final model against the linearised photo.
    pub rms: f64,
}

/// Fit `app` (lights, gamma, colours, materials) for the scene in `cmp`
/// against the 8-bit photo `rgb8` at the compare size.
pub fn fit_appearance(cmp: &mut Compare3, cam: &Cam, rgb8: &[u8], app: &mut Appearance, o: &FitOpts) -> FitReport {
    let t0 = std::time::Instant::now();
    let npx = cmp.size.0 * cmp.size.1;
    // Colours and substances are read inside each object, two pixels from
    // its outline: where the geometry is only roughly right, the outline's
    // pixels belong to the neighbours as often as to the object.
    let ids = erode(&cmp.ids(cam), cmp.size, 2);
    // Saturated photo pixels (any channel at 255) say only "at least this
    // bright": they are left out of every fit.
    // Per channel: a pixel saturated in red still measures green and blue.
    let ok: [Vec<bool>; 3] = std::array::from_fn(|c| rgb8.chunks_exact(3).map(|px| px[c] < SAT8).collect());
    let use_px: Vec<bool> = (0..rgb8.len() / 3).map(|p| ok.iter().all(|o| o[p])).collect();
    let gammas = [0.8, 0.86, 0.92, 0.97, 1.0, 1.03, 1.08, 1.14, 1.2];
    let targets: Vec<Vec<[f64; 3]>> = gammas.iter().map(|&g| linearise(rgb8, g)).collect();
    let mut renders = 0;
    let zero_env = [[0.0; 3]; LOBES];
    let dark = Key { rgb: [0.0; 3], ..app.key };
    let keys = key_grid();
    for round in 0..o.rounds {
        cmp.set_mats(&app.mats, &app.table);
        // World lobes, the sun off.
        cmp.set_key(&dark);
        let mut basis: Vec<Vec<[f64; 3]>> = Vec::new();
        for k in 0..LOBES {
            let mut w = zero_env;
            w[k] = [1.0; 3];
            cmp.set_env(&w);
            basis.push(cmp.render(cam, o.spp, o.seed));
            renders += 1;
        }
        cmp.set_env(&zero_env);
        // The key: on the first round, every direction at a middle size;
        // afterwards, the neighbours of the current one.
        let dirs: Vec<[f64; 3]> = if round == 0 {
            keys.clone()
        } else {
            let d0 = app.key.dir;
            let mut v = vec![d0];
            for (da, de) in [(-8.0f64, 0.0f64), (8.0, 0.0), (0.0, -6.0), (0.0, 6.0)] {
                let el = d0[2].asin().to_degrees() + de;
                let az = d0[1].atan2(d0[0]).to_degrees() + da;
                let (e, a) = (el.clamp(3.0, 89.0).to_radians(), az.to_radians());
                v.push([e.cos() * a.cos(), e.cos() * a.sin(), e.sin()]);
            }
            v
        };
        let radii: Vec<f64> = if round == 0 { vec![5.0] } else { vec![1.5, 3.5, 7.0, 12.0] };
        // Gram blocks of the world basis, once per round; per gamma its
        // right-hand side; per key only the key's row.
        let nb = LOBES;
        let used3: [Vec<usize>; 3] = std::array::from_fn(|c| (0..npx).filter(|&p| ok[c][p]).collect());
        let gee: Vec<Vec<Vec<f64>>> = (0..3)
            .map(|c| {
                let mut a = vec![vec![0.0; nb]; nb];
                for i in 0..nb {
                    for j in 0..=i {
                        let v: f64 = used3[c].iter().map(|&p| basis[i][p][c] * basis[j][p][c]).sum();
                        a[i][j] = v;
                        a[j][i] = v;
                    }
                }
                a
            })
            .collect();
        let be: Vec<Vec<Vec<f64>>> = targets.iter().map(|t| (0..3).map(|c| (0..nb).map(|i| used3[c].iter().map(|&p| basis[i][p][c] * t[p][c]).sum()).collect()).collect()).collect();
        let tt: Vec<[f64; 3]> = targets.iter().map(|t| std::array::from_fn(|c| used3[c].iter().map(|&p| t[p][c] * t[p][c]).sum())).collect();
        let mut best: Option<(f64, usize, Key, Vec<[f64; 3]>)> = None;
        for d in &dirs {
            for &rad in &radii {
                let key = Key { dir: *d, radius: rad.to_radians(), rgb: [1.0; 3] };
                cmp.set_key(&key);
                let bk = cmp.render(cam, o.spp, o.seed);
                renders += 1;
                let gek: Vec<Vec<f64>> = (0..3).map(|c| (0..nb).map(|i| used3[c].iter().map(|&p| basis[i][p][c] * bk[p][c]).sum()).collect()).collect();
                let gkk: Vec<f64> = (0..3).map(|c| used3[c].iter().map(|&p| bk[p][c] * bk[p][c]).sum()).collect();
                for (gi, t) in targets.iter().enumerate() {
                    let mut w = vec![[0.0; 3]; nb + 1];
                    let mut sse = 0.0;
                    for c in 0..3 {
                        let mut a = gee[c].clone();
                        for i in 0..nb {
                            a[i].push(gek[c][i]);
                        }
                        let mut last = gek[c].clone();
                        last.push(gkk[c]);
                        a.push(last);
                        let mut b = be[gi][c].clone();
                        b.push(used3[c].iter().map(|&p| bk[p][c] * t[p][c]).sum());
                        let x = nnls(&a, &b);
                        let mut q = tt[gi][c];
                        for i in 0..=nb {
                            q -= 2.0 * x[i] * b[i];
                            for j in 0..=nb {
                                q += x[i] * a[i][j] * x[j];
                            }
                        }
                        sse += q.max(0.0);
                        for i in 0..=nb {
                            w[i][c] = x[i];
                        }
                    }
                    if best.as_ref().is_none_or(|x| sse < x.0) {
                        best = Some((sse, gi, key, w));
                    }
                }
            }
        }
        let (_, gi, key, mut w) = best.unwrap();
        // Saturated pixels say "at least 1": where the model falls short of
        // that, they join the fit at 1 (an active set for the inequality;
        // two rounds).
        cmp.set_key(&key);
        let bk = cmp.render(cam, o.spp, o.seed);
        renders += 1;
        let mut active = ok.clone();
        for _ in 0..2 {
            let m: Vec<[f64; 3]> = (0..npx).map(|p| std::array::from_fn(|c| (0..LOBES).map(|i| w[i][c] * basis[i][p][c]).sum::<f64>() + w[LOBES][c] * bk[p][c])).collect();
            let mut added = 0;
            for p in 0..npx {
                for c in 0..3 {
                    if !active[c][p] && m[p][c] < SAT {
                        active[c][p] = true;
                        added += 1;
                    }
                }
            }
            if added == 0 {
                break;
            }
            let t = &targets[gi];
            let mut b: Vec<&[[f64; 3]]> = basis.iter().map(|v| v.as_slice()).collect();
            b.push(&bk);
            for c in 0..3 {
                let n = b.len();
                let mut a = vec![vec![0.0; n]; n];
                let mut r = vec![0.0; n];
                for p in (0..npx).filter(|&p| active[c][p]) {
                    let tv = t[p][c].min(1.0);
                    for i in 0..n {
                        r[i] += b[i][p][c] * tv;
                        for j in 0..=i {
                            a[i][j] += b[i][p][c] * b[j][p][c];
                        }
                    }
                }
                for i in 0..n {
                    for j in 0..i {
                        a[j][i] = a[i][j];
                    }
                }
                let x = nnls(&a, &r);
                for i in 0..n {
                    w[i][c] = x[i];
                }
            }
        }
        app.gamma = gammas[gi];
        for k in 0..LOBES {
            app.env[k] = w[k];
        }
        app.key = Key { rgb: w[LOBES], ..key };
        cmp.set_appearance(app);
        let target = &targets[gi];
        // Colours, twice (the transport is close to linear in each albedo).
        for _ in 0..2 {
            let m = cmp.render(cam, o.spp, o.seed);
            renders += 1;
            rescale_colours(&m, target, &ids, app);
            cmp.set_mats(&app.mats, &app.table);
        }
        // Materials: each object picks the variant whose pixels fit best.
        if o.materials {
            let variants = [(0.2, false), (0.35, false), (0.5, false), (0.7, false), (0.25, true), (0.45, true)];
            let mut err = vec![[0.0f64; 6]; app.mats.len()];
            let mut tries: Vec<Appearance> = Vec::new();
            for (vi, &(rough, metal)) in variants.iter().enumerate() {
                let mut a = app.clone();
                for m in &mut a.mats {
                    m.rough = rough;
                    if m.metal != metal {
                        m.metal = metal;
                    }
                }
                cmp.set_mats(&a.mats, &a.table);
                let m = cmp.render(cam, o.spp, o.seed);
                rescale_colours(&m, target, &ids, &mut a);
                cmp.set_mats(&a.mats, &a.table);
                let m = cmp.render(cam, o.spp, o.seed);
                renders += 2;
                for p in 0..npx {
                    if ids[p] < app.mats.len() && use_px[p] {
                        err[ids[p]][vi] += (0..3).map(|c| (m[p][c] - target[p][c]).powi(2)).sum::<f64>();
                    }
                }
                tries.push(a);
            }
            for i in 0..app.mats.len() {
                let vi = (0..variants.len()).min_by(|&a, &b| err[i][a].total_cmp(&err[i][b])).unwrap();
                if err[i][vi] > 0.0 {
                    app.mats[i] = tries[vi].mats[i];
                }
            }
            cmp.set_mats(&app.mats, &app.table);
        }
    }
    let m = cmp.render(cam, o.spp, o.seed);
    renders += 1;
    let t = linearise(rgb8, app.gamma);
    // Defocus: the blur of the model that fits best.
    let mut best_d = (f64::INFINITY, 0.0);
    for d in [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 3.6] {
        let mb = blur_px(&m, cmp.size, d);
        let e: f64 = mb.iter().zip(&t).zip(&use_px).filter(|x| *x.1).map(|((a, b), _)| (0..3).map(|c| (a[c].min(1.0) - b[c]).powi(2)).sum::<f64>()).sum();
        if e < best_d.0 {
            best_d = (e, d);
        }
    }
    app.defocus = best_d.1 / cmp.size.0 as f64;
    let m = blur_px(&m, cmp.size, best_d.1);
    let rms = (m.iter().zip(&t).map(|(a, b)| (0..3).map(|c| (a[c] - b[c]).powi(2)).sum::<f64>()).sum::<f64>() / (npx * 3) as f64).sqrt();
    FitReport { renders, seconds: t0.elapsed().as_secs_f64(), rms }
}

/// Every object's mesh in its own frame (empty when it does not build).
pub fn local_meshes(objs: &[Object]) -> Vec<Tris> {
    use rayon::prelude::*;
    objs.par_iter().map(|o| crate::scene::mesh_fast(o).unwrap_or_default()).collect()
}

/// Gaussian blur of an RGB image (σ in pixels).
pub fn blur_px(img: &[[f64; 3]], size: (usize, usize), s: f64) -> Vec<[f64; 3]> {
    if s <= 0.0 {
        return img.to_vec();
    }
    let flat: Vec<f64> = img.iter().flat_map(|p| p.iter().copied()).collect();
    irys_recon::fit::blur(&flat, size, 3, s).chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
}

/// Keep a pixel's id only where every pixel within `r` (Chebyshev) has it;
/// else `usize::MAX - 1` (unused).
pub fn erode(ids: &[usize], (w, h): (usize, usize), r: usize) -> Vec<usize> {
    (0..w * h)
        .map(|p| {
            let (x, y) = (p % w, p / w);
            let id = ids[p];
            let ok = (y.saturating_sub(r)..(y + r + 1).min(h)).all(|yy| (x.saturating_sub(r)..(x + r + 1).min(w)).all(|xx| ids[yy * w + xx] == id));
            if ok { id } else { usize::MAX - 1 }
        })
        .collect()
}
