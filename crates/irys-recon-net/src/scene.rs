//! The v2 scene language: many objects on a support plane.
//!
//! An object is a **family** (the grammar's production), a few discrete
//! **tokens** (variant and counts), continuous **parameters** in mm, a
//! **pose** `(x, y, yaw)` on the plane, a **class** (what it is) and a
//! **material**. Five families:
//!
//! - `plate`: v0/v1's program (`irys_recon::dsl`), a filleted plate with a
//!   grid or a circle of through-holes, built as a vcad BRep — brackets,
//!   gear plates;
//! - `lathe`: a revolved profile spline through four radii — cans,
//!   bottles, cups, the humidifier;
//! - `product`: a box with filleted vertical edges or every edge rounded,
//!   optionally on a stand — monitor, power strip, speaker, mouse, webcam;
//! - `grid`: a base with a rows × columns pattern of key caps at a pitch —
//!   keyboards (the pitch is a scale gauge: 19.05 mm);
//! - `shell`: a walled shell with ribs (open tray) or a honeycomb lid of
//!   hexagonal cells — 3D-printed cases.
//!
//! Every object meshes in its own frame: centred in `x, y`, standing on
//! `z = 0`. The text form is one line per object (`Object::to_text`).

use irys_recon::dsl::{Corner, Feature, Program};
use irys_recon::jac::Mix;
use serde::{Deserialize, Serialize};

/// Token slots and their vocabulary.
pub const NTOK: usize = 4;
pub const VOCAB: usize = 8;
/// Continuous parameter slots.
pub const NP: usize = 8;
/// Class slots (some reserved).
pub const CLASSES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Family {
    Plate = 0,
    Lathe = 1,
    Product = 2,
    Grid = 3,
    Shell = 4,
}

pub const FAMILIES: [Family; 5] = [Family::Plate, Family::Lathe, Family::Product, Family::Grid, Family::Shell];

impl Family {
    pub fn from_token(t: usize) -> Option<Family> {
        FAMILIES.get(t).copied()
    }
    pub fn name(self) -> &'static str {
        ["plate", "lathe", "product", "grid", "shell"][self as usize]
    }
    /// Parameter names, in slot order.
    pub fn param_names(self) -> &'static [&'static str] {
        match self {
            Family::Plate => &["w", "d", "h", "r", "hole_r", "pitch"],
            Family::Lathe => &["h", "r0", "r1", "r2", "r3"],
            Family::Product => &["w", "d", "h", "r", "lift"],
            Family::Grid => &["w", "d", "h", "pitch", "key_h"],
            Family::Shell => &["w", "d", "h", "wall", "cell"],
        }
    }
    /// Whether a yaw is observable (a solid of revolution has none).
    pub fn has_yaw(self) -> bool {
        self != Family::Lathe
    }
}

/// What an object is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Class {
    Bracket = 0,
    GearPlate = 1,
    Can = 2,
    Bottle = 3,
    Cup = 4,
    Humidifier = 5,
    Monitor = 6,
    PowerStrip = 7,
    Speaker = 8,
    Mouse = 9,
    Keyboard = 10,
    HoneycombCase = 11,
    Tray = 12,
    Webcam = 13,
}

pub const CLASS_LIST: [Class; 14] = [
    Class::Bracket,
    Class::GearPlate,
    Class::Can,
    Class::Bottle,
    Class::Cup,
    Class::Humidifier,
    Class::Monitor,
    Class::PowerStrip,
    Class::Speaker,
    Class::Mouse,
    Class::Keyboard,
    Class::HoneycombCase,
    Class::Tray,
    Class::Webcam,
];

impl Class {
    pub fn from_index(i: usize) -> Option<Class> {
        CLASS_LIST.get(i).copied()
    }
    pub fn name(self) -> &'static str {
        [
            "bracket",
            "gear plate",
            "soda can",
            "bottle",
            "cup",
            "humidifier",
            "monitor",
            "power strip",
            "speaker",
            "mouse",
            "keyboard",
            "honeycomb case",
            "tray",
            "webcam",
        ][self as usize]
    }
    pub fn family(self) -> Family {
        use Class::*;
        match self {
            Bracket | GearPlate => Family::Plate,
            Can | Bottle | Cup | Humidifier => Family::Lathe,
            Monitor | PowerStrip | Speaker | Mouse | Webcam => Family::Product,
            Keyboard => Family::Grid,
            HoneycombCase | Tray => Family::Shell,
        }
    }
    /// A known size that sets the metric scale: `(parameter slot, mm)`.
    /// A 12 oz can is 122 mm tall; a keyboard's key pitch is 19.05 mm.
    pub fn gauge(self) -> Option<(usize, f64)> {
        match self {
            Class::Can => Some((0, 122.0)),
            Class::Keyboard => Some((3, 19.05)),
            _ => None,
        }
    }
}

/// A substance, as the renderer wants it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Mat {
    pub rgb: [f64; 3],
    pub rough: f64,
    pub metal: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Object {
    pub class: Class,
    pub family: Family,
    pub tokens: [usize; NTOK],
    /// mm, in [`Family::param_names`] order.
    pub params: Vec<f64>,
    /// `x, y` (mm) and yaw (degrees) on the support plane.
    pub pose: [f64; 3],
    pub mat: Mat,
}

/// Normalised log size: what the network regresses.
pub fn norm_mm(v: f64) -> f64 {
    (v.max(0.05).ln() - 3.5) / 1.5
}
pub fn denorm_mm(n: f64) -> f64 {
    (n * 1.5 + 3.5).exp()
}

impl Object {
    /// Which token slots mean something for this object.
    pub fn token_mask(&self) -> [bool; NTOK] {
        token_mask(self.family, &self.tokens)
    }

    /// The plate as a v0 program (`None` for other families).
    pub fn plate_program(&self) -> Option<Program> {
        if self.family != Family::Plate {
            return None;
        }
        let p = &self.params;
        let (w, d, h, r, hr, pitch) = (p[0], p[1], p[2], p[3], p[4], p[5]);
        let corner = if r >= 0.6 { Corner::Fillet(r.min(0.44 * w.min(d))) } else { Corner::Sharp };
        let mut features = Vec::new();
        if self.tokens[1] == 0 {
            let (n, m) = (self.tokens[2].max(1), self.tokens[3].max(1));
            for j in 0..m {
                let y = (j as f64 - (m as f64 - 1.0) / 2.0) * pitch;
                let x = -(n as f64 - 1.0) / 2.0 * pitch;
                if n >= 2 {
                    features.push(Feature::LinearHoles { x, y, dx: pitch, dy: 0.0, r: hr, n });
                } else {
                    features.push(Feature::Hole { x, y, r: hr });
                }
            }
        } else {
            features.push(Feature::CircularHoles { x: 0.0, y: 0.0, rad: pitch, phase: 0.0, r: hr, n: self.tokens[2].max(3) });
        }
        Some(Program { w, d, h, corner, features, pose: None })
    }

    /// The object as one line of the scene language.
    pub fn to_text(&self) -> String {
        let names = self.family.param_names();
        let mut s = format!("{} {}", self.class.name().replace(' ', "_"), self.family.name());
        let toks: Vec<String> = token_names(self.family, &self.tokens);
        for t in toks {
            s += &format!(" {t}");
        }
        for (n, v) in names.iter().zip(&self.params) {
            if param_used(self.family, &self.tokens, names.iter().position(|x| x == n).unwrap()) {
                s += &format!(" {n}={v:.2}");
            }
        }
        s += &format!(" @ {:.1} {:.1} {:.1}", self.pose[0], self.pose[1], self.pose[2]);
        s
    }

    /// A bounding radius on the plane, mm.
    pub fn radius(&self) -> f64 {
        let p = &self.params;
        match self.family {
            Family::Lathe => p[1..5].iter().cloned().fold(0.0, f64::max),
            _ => 0.5 * p[0].hypot(p[1]),
        }
    }

    /// Height, mm.
    pub fn height(&self) -> f64 {
        let p = &self.params;
        match self.family {
            Family::Lathe => p[0],
            Family::Product if self.tokens[2] == 1 => p[2] + p[4],
            Family::Grid => p[2] + p[4],
            _ => p[2],
        }
    }

    pub fn valid(&self) -> bool {
        let p = &self.params;
        if p.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return false;
        }
        match self.family {
            Family::Plate => self.plate_program().is_some_and(|q| q.valid()),
            Family::Lathe => p[0] > 5.0 && p[1..5].iter().all(|&r| r > 2.0),
            Family::Product => p[3] < 0.5 * p[0].min(p[1]).min(if self.tokens[1] == 1 { p[2] } else { 1e9 }),
            Family::Grid => p[3] > 5.0 && p[0] > 2.0 * p[3] && p[1] > 2.0 * p[3],
            Family::Shell => p[3] > 0.5 && p[3] < 0.2 * p[0].min(p[1]) && p[4] > 2.0 * p[3] && p[4] < 0.4 * p[0].min(p[1]),
        }
    }
}

pub fn token_mask(f: Family, t: &[usize; NTOK]) -> [bool; NTOK] {
    match f {
        Family::Plate => [true, true, true, t[1] == 0],
        Family::Lathe => [true, false, false, false],
        Family::Product => [true, true, true, false],
        Family::Grid => [true, false, true, false],
        Family::Shell => [true, true, t[1] == 0, false],
    }
}

pub fn param_used(f: Family, t: &[usize; NTOK], k: usize) -> bool {
    match f {
        Family::Product => k < 4 || t[2] == 1,
        Family::Shell => k < 4 || t[1] == 1,
        _ => k < f.param_names().len(),
    }
}

fn token_names(f: Family, t: &[usize; NTOK]) -> Vec<String> {
    match f {
        Family::Plate if t[1] == 0 => vec![format!("holes={}x{}", t[2], t[3])],
        Family::Plate => vec![format!("circle={}", t[2])],
        Family::Lathe => vec![],
        Family::Product => {
            let mut v = vec![if t[1] == 0 { "fillet=vertical".into() } else { "fillet=all".into() }];
            if t[2] == 1 {
                v.push("stand".into());
            }
            v
        }
        Family::Grid => vec![format!("rows={}", t[2])],
        Family::Shell if t[1] == 0 => vec![format!("open ribs={}", t[2])],
        Family::Shell => vec!["hex".into()],
    }
}

/// Program-length cost of the structure, nats: one choice per used token
/// slot over its vocabulary, plus [`irys_recon::mdl::NATS_PER_PARAM`] per
/// used continuous parameter.
pub fn program_length(o: &Object) -> f64 {
    program_length_with(o, structure_nats(o))
}

/// The structure's share under the uniform grammar: one choice per used slot.
pub fn structure_nats(o: &Object) -> f64 {
    o.token_mask().iter().filter(|&&m| m).count() as f64 * (VOCAB as f64).ln()
}

/// Program length with the structure coded by a prior of `structure` nats
/// (`−log p(structure)`: the grammar's, or the network's).
pub fn program_length_with(o: &Object, structure: f64) -> f64 {
    let pars = (0..o.params.len()).filter(|&k| param_used(o.family, &o.tokens, k)).count() as f64;
    structure + pars * irys_recon::mdl::NATS_PER_PARAM + 3.0 * irys_recon::mdl::NATS_PER_PARAM
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

fn range(r: &mut Mix, lo: f64, hi: f64) -> f64 {
    lo + (hi - lo) * r.uniform()
}
fn pick<T: Copy>(r: &mut Mix, xs: &[T]) -> T {
    xs[(r.uniform() * xs.len() as f64) as usize % xs.len()]
}

const PLA: [[f64; 3]; 8] = [
    [0.85, 0.28, 0.04],
    [0.05, 0.18, 0.65],
    [0.65, 0.05, 0.06],
    [0.08, 0.45, 0.15],
    [0.80, 0.62, 0.04],
    [0.35, 0.08, 0.45],
    [0.04, 0.04, 0.045],
    [0.80, 0.80, 0.78],
];

fn material(r: &mut Mix, class: Class) -> Mat {
    let u = r.uniform();
    match class {
        Class::Can => Mat { rgb: pick(r, &[[0.75, 0.05, 0.05], [0.85, 0.85, 0.85], [0.1, 0.2, 0.6], [0.1, 0.5, 0.15], [0.9, 0.7, 0.1]]), rough: range(r, 0.2, 0.4), metal: u < 0.4 },
        Class::Monitor | Class::Keyboard | Class::Mouse | Class::Webcam | Class::Speaker => {
            if u < 0.25 {
                Mat { rgb: [0.85, 0.86, 0.88], rough: range(r, 0.25, 0.45), metal: true }
            } else {
                let g = range(r, 0.02, 0.25);
                Mat { rgb: [g, g, g * 1.03], rough: range(r, 0.3, 0.7), metal: false }
            }
        }
        Class::Bottle | Class::Cup | Class::Humidifier | Class::PowerStrip => {
            let g = range(r, 0.6, 0.92);
            if u < 0.5 {
                Mat { rgb: [g, g, g * 0.98], rough: range(r, 0.2, 0.6), metal: false }
            } else {
                Mat { rgb: pick(r, &PLA), rough: range(r, 0.2, 0.6), metal: false }
            }
        }
        _ => {
            if u < 0.15 {
                Mat { rgb: [0.88, 0.89, 0.9], rough: range(r, 0.2, 0.5), metal: true }
            } else {
                Mat { rgb: pick(r, &PLA), rough: range(r, 0.35, 0.65), metal: false }
            }
        }
    }
}

fn q(v: f64) -> f64 {
    (v * 2.0).round() / 2.0
}

/// A random object of `class`, valid, at the origin.
pub fn sample_object(r: &mut Mix, class: Class) -> Object {
    let family = class.family();
    for _ in 0..1000 {
        let (tokens, params): ([usize; NTOK], Vec<f64>) = match class {
            Class::Bracket => {
                let n = 1 + (r.uniform() * 4.0) as usize;
                let m = 1 + (r.uniform() * 2.0) as usize;
                let pitch = q(range(r, 10.0, 22.0));
                let hr = q(range(r, 1.5, 4.0).min(0.3 * pitch));
                let w = q(n as f64 * pitch + range(r, 10.0, 30.0));
                let d = q(m as f64 * pitch + range(r, 10.0, 25.0));
                ([0, 0, n, m], vec![w, d, q(range(r, 3.0, 8.0)), q(range(r, 0.0, 6.0)), hr, pitch])
            }
            Class::GearPlate => {
                let k = 3 + (r.uniform() * 5.0) as usize;
                let rad = q(range(r, 12.0, 30.0));
                let hr = q(range(r, 1.5, 4.0));
                let s = q(2.0 * (rad + hr) + range(r, 8.0, 20.0));
                ([0, 1, k, 0], vec![s, q(s * range(r, 0.9, 1.1)), q(range(r, 2.5, 6.0)), q(range(r, 3.0, 0.3 * s)), hr, rad])
            }
            Class::Can => {
                let h = range(r, 120.0, 124.0);
                let rr = range(r, 32.5, 33.5);
                ([1, 0, 0, 0], vec![h, rr - 1.5, rr, rr, rr - 6.0])
            }
            Class::Bottle => {
                let h = range(r, 170.0, 260.0);
                let r0 = range(r, 28.0, 42.0);
                ([1, 0, 0, 0], vec![h, r0, r0 * range(r, 0.95, 1.05), r0 * range(r, 0.7, 1.0), range(r, 11.0, 16.0)])
            }
            Class::Cup => {
                let h = range(r, 75.0, 110.0);
                let r0 = range(r, 30.0, 40.0);
                ([1, 0, 0, 0], vec![h, r0, r0 * 1.08, r0 * 1.17, r0 * range(r, 1.2, 1.35)])
            }
            Class::Humidifier => {
                let h = range(r, 140.0, 250.0);
                let r0 = range(r, 50.0, 85.0);
                ([1, 0, 0, 0], vec![h, r0, r0 * range(r, 0.9, 1.05), r0 * range(r, 0.8, 1.0), r0 * range(r, 0.45, 0.8)])
            }
            Class::Monitor => (
                [2, 0, 1, 0],
                vec![range(r, 480.0, 620.0), range(r, 15.0, 40.0), range(r, 280.0, 380.0), range(r, 2.0, 6.0), range(r, 70.0, 140.0)],
            ),
            Class::PowerStrip => ([2, 0, 0, 0], vec![range(r, 250.0, 400.0), range(r, 45.0, 62.0), range(r, 28.0, 40.0), range(r, 4.0, 14.0), 50.0]),
            Class::Speaker => ([2, 0, 0, 0], vec![range(r, 80.0, 150.0), range(r, 80.0, 150.0), range(r, 150.0, 250.0), range(r, 3.0, 25.0), 50.0]),
            Class::Mouse => ([2, 1, 0, 0], vec![range(r, 60.0, 72.0), range(r, 105.0, 126.0), range(r, 34.0, 44.0), range(r, 14.0, 16.5), 50.0]),
            Class::Webcam => ([2, 1, 0, 0], vec![range(r, 70.0, 100.0), range(r, 25.0, 38.0), range(r, 25.0, 35.0), range(r, 8.0, 12.0), 50.0]),
            Class::Keyboard => {
                let rows = 4 + (r.uniform() * 3.0) as usize;
                let pitch = 19.05;
                let d = rows as f64 * pitch + range(r, 8.0, 30.0);
                ([3, 0, rows, 0], vec![range(r, 290.0, 440.0), d, range(r, 12.0, 25.0), pitch, range(r, 6.0, 10.0)])
            }
            Class::HoneycombCase => (
                [4, 1, 0, 0],
                vec![range(r, 80.0, 140.0), range(r, 60.0, 100.0), range(r, 25.0, 45.0), range(r, 1.8, 3.0), range(r, 7.0, 13.0)],
            ),
            Class::Tray => {
                let ribs = (r.uniform() * 4.0) as usize;
                ([4, 0, ribs, 0], vec![range(r, 80.0, 200.0), range(r, 60.0, 150.0), range(r, 18.0, 50.0), range(r, 2.0, 4.0), 10.0])
            }
        };
        let o = Object { class, family, tokens, params, pose: [0.0, 0.0, range(r, -180.0, 180.0)], mat: material(r, class) };
        if o.valid() {
            return o;
        }
    }
    panic!("no valid {class:?}");
}

/// A pinhole: eye, look-at, horizontal field of view (degrees), size.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Cam {
    pub eye: [f64; 3],
    pub at: [f64; 3],
    pub hfov: f64,
    pub size: (usize, usize),
}

impl Cam {
    /// forward, right, up, tan of the vertical half angle.
    pub fn frame(&self) -> ([f64; 3], [f64; 3], [f64; 3], f64) {
        let f = normalize(sub(self.at, self.eye));
        let r = normalize(cross(f, [0.0, 0.0, 1.0]));
        let u = cross(r, f);
        let tx = (self.hfov.to_radians() / 2.0).tan();
        let ty = tx * self.size.1 as f64 / self.size.0 as f64;
        (f, r, u, ty)
    }
    /// Unit ray direction through pixel coordinate `(x, y)` (top-left origin).
    pub fn dir(&self, x: f64, y: f64) -> [f64; 3] {
        let (f, r, u, ty) = self.frame();
        let (w, h) = (self.size.0 as f64, self.size.1 as f64);
        let sx = (2.0 * x / w - 1.0) * ty * w / h;
        let sy = (1.0 - 2.0 * y / h) * ty;
        normalize([f[0] + r[0] * sx + u[0] * sy, f[1] + r[1] * sx + u[1] * sy, f[2] + r[2] * sx + u[2] * sy])
    }
    /// Pixel of world point `p`.
    pub fn project(&self, p: [f64; 3]) -> Option<[f64; 2]> {
        let (f, r, u, ty) = self.frame();
        let d = sub(p, self.eye);
        let z = dot(d, f);
        if z <= 1e-6 {
            return None;
        }
        let (w, h) = (self.size.0 as f64, self.size.1 as f64);
        let sx = dot(d, r) / z / (ty * w / h);
        let sy = dot(d, u) / z / ty;
        Some([(sx + 1.0) * w / 2.0, (1.0 - sy) * h / 2.0])
    }
    /// Where pixel `(x, y)` meets the plane `z = 0`.
    pub fn on_plane(&self, x: f64, y: f64) -> Option<[f64; 2]> {
        let d = self.dir(x, y);
        if d[2] >= -1e-6 {
            return None;
        }
        let t = -self.eye[2] / d[2];
        Some([self.eye[0] + t * d[0], self.eye[1] + t * d[1]])
    }
    /// Height above the plane (m) and pitch (rad, below horizontal): what the
    /// plane head regresses.
    pub fn plane(&self) -> [f64; 2] {
        let (f, ..) = self.frame();
        [(self.eye[2] / 1000.0).ln(), (-f[2]).asin()]
    }
    /// The canonical camera of a plane estimate: at `(0, −y, H)` looking down
    /// `pitch` along `+y`.
    pub fn from_plane(plane: [f64; 2], hfov: f64, size: (usize, usize)) -> Cam {
        let h = plane[0].exp() * 1000.0;
        let p = plane[1].clamp(0.05, 1.5);
        let back = h / p.tan();
        Cam { eye: [0.0, -back, h], at: [0.0, 0.0, 0.0], hfov, size }
    }
    pub fn resized(&self, size: (usize, usize)) -> Cam {
        Cam { size, ..*self }
    }
}

/// Lighting: a procedural environment and a key light.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Lighting {
    /// Sky zenith and horizon radiance, linear RGB.
    pub zenith: [f64; 3],
    pub horizon: [f64; 3],
    /// A window (softbox) on the environment: azimuth, elevation (degrees),
    /// half size (degrees), radiance.
    pub window: [f64; 4],
    /// Key light direction (toward the light), angular radius (degrees),
    /// radiance scale.
    pub key: [f64; 3],
    pub key_radius: f64,
    pub key_radiance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    pub objects: Vec<Object>,
    pub table: Mat,
    pub cam: Cam,
    pub light: Lighting,
    pub seed: u64,
    /// Camera post: blur σ (px), gamma tweak, JPEG quality, noise gain.
    pub post: [f64; 4],
}

/// Which slice of the world a stream draws from: `A` is the supervised
/// training distribution, `B` a shifted one (other lights, lower and wider
/// cameras, stronger post) used unlabelled for renderer-as-teacher tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Domain {
    A,
    B,
}

/// A random tabletop scene, deterministic in `seed`.
pub fn sample_scene(seed: u64, domain: Domain, size: (usize, usize)) -> Scene {
    let mut r = Mix(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ if domain == Domain::A { 0xA } else { 0xB0B });
    let n = 3 + (r.uniform() * 10.0) as usize;
    // Desk-scale scenes sometimes hold a monitor and a keyboard; bench-scale
    // ones hold small parts. Sample a scale first.
    let big = r.uniform() < 0.45;
    let small_classes = [
        Class::Bracket,
        Class::GearPlate,
        Class::Can,
        Class::Cup,
        Class::Mouse,
        Class::Webcam,
        Class::HoneycombCase,
        Class::Tray,
        Class::Bottle,
        Class::Speaker,
        Class::Humidifier,
    ];
    let mut objects: Vec<Object> = Vec::new();
    let mut classes = Vec::new();
    if big {
        if r.uniform() < 0.6 {
            classes.push(Class::Monitor);
        }
        if r.uniform() < 0.7 {
            classes.push(Class::Keyboard);
        }
        if r.uniform() < 0.4 {
            classes.push(Class::PowerStrip);
        }
    }
    while classes.len() < n {
        classes.push(pick(&mut r, &small_classes));
    }
    // Place: largest first, rejection on bounding circles, in a region that
    // grows with what is placed.
    let mut objs: Vec<Object> = classes.iter().map(|&c| sample_object(&mut r, c)).collect();
    objs.sort_by(|a, b| b.radius().total_cmp(&a.radius()));
    let area: f64 = objs.iter().map(|o| (o.radius() * 1.15).powi(2) * std::f64::consts::PI).sum();
    let half = (area / 0.45).sqrt() * 0.5;
    for mut o in objs {
        let mut placed = false;
        for _ in 0..300 {
            let x = range(&mut r, -half, half);
            // Monitors go to the back.
            let y = if o.class == Class::Monitor { range(&mut r, 0.3 * half, half) } else { range(&mut r, -half * 0.8, half * 0.8) };
            let ok = objects.iter().all(|p| {
                let gap = (p.pose[0] - x).hypot(p.pose[1] - y);
                gap > 1.02 * (p.radius() + o.radius())
            });
            if ok {
                o.pose[0] = x;
                o.pose[1] = y;
                if o.class == Class::Monitor || o.class == Class::Keyboard || o.class == Class::PowerStrip {
                    o.pose[2] = range(&mut r, -20.0, 20.0);
                }
                placed = true;
                break;
            }
        }
        if placed {
            objects.push(o);
        }
    }
    // Camera: looking along +y (azimuth 0), so poses are camera-relative.
    let (el_lo, el_hi, fov_lo, fov_hi) = match domain {
        Domain::A => (28.0, 65.0, 58.0, 72.0),
        Domain::B => (22.0, 50.0, 66.0, 78.0),
    };
    let el: f64 = range(&mut r, el_lo, el_hi);
    let hfov = range(&mut r, fov_lo, fov_hi);
    let mut cx = 0.0;
    let mut cy = 0.0;
    for o in &objects {
        cx += o.pose[0];
        cy += o.pose[1];
    }
    cx /= objects.len().max(1) as f64;
    cy /= objects.len().max(1) as f64;
    let extent = objects.iter().map(|o| (o.pose[0] - cx).hypot(o.pose[1] - cy) + o.radius()).fold(50.0, f64::max);
    let tall = objects.iter().map(|o| o.height()).fold(0.0, f64::max);
    let vfov = 2.0 * ((hfov.to_radians() / 2.0).tan() * size.1 as f64 / size.0 as f64).atan();
    let dist = (extent.max(tall * 0.8) / (vfov / 2.0).tan()) * range(&mut r, 0.8, 1.15);
    let at = [cx + range(&mut r, -0.1, 0.1) * extent, cy + range(&mut r, -0.1, 0.1) * extent, range(&mut r, 0.0, 0.25) * tall];
    let e = el.to_radians();
    let eye = [at[0], at[1] - dist * e.cos(), at[2] + dist * e.sin()];
    let cam = Cam { eye, at, hfov, size };
    // Lighting.
    let warm = range(&mut r, -0.15, 0.15);
    let lvl = range(&mut r, 0.4, 1.2);
    let tint = |k: f64| [k * (1.0 + warm), k, k * (1.0 - warm)];
    let light = match domain {
        Domain::A => Lighting {
            zenith: tint(lvl),
            horizon: tint(lvl * range(&mut r, 0.4, 0.9)),
            window: [range(&mut r, 0.0, 360.0), range(&mut r, 10.0, 45.0), range(&mut r, 10.0, 30.0), lvl * range(&mut r, 2.0, 6.0)],
            key: [range(&mut r, 0.0, 360.0), range(&mut r, 35.0, 75.0), 0.0],
            key_radius: range(&mut r, 3.0, 12.0),
            key_radiance: range(&mut r, 0.6, 2.0),
        },
        Domain::B => Lighting {
            zenith: tint(lvl * 0.7),
            horizon: tint(lvl * range(&mut r, 0.8, 1.2)),
            window: [range(&mut r, 0.0, 360.0), range(&mut r, 5.0, 25.0), range(&mut r, 20.0, 40.0), lvl * range(&mut r, 4.0, 9.0)],
            key: [range(&mut r, 0.0, 360.0), range(&mut r, 15.0, 40.0), 0.0],
            key_radius: range(&mut r, 1.0, 4.0),
            key_radiance: range(&mut r, 1.5, 3.5),
        },
    };
    let table = {
        let u = r.uniform();
        let rgb = if u < 0.3 {
            [range(&mut r, 0.45, 0.65), range(&mut r, 0.30, 0.42), range(&mut r, 0.18, 0.26)]
        } else if u < 0.6 {
            let g = range(&mut r, 0.55, 0.85);
            [g, g, g * 0.98]
        } else if u < 0.85 {
            let g = range(&mut r, 0.08, 0.3);
            [g, g, g * 1.02]
        } else {
            [range(&mut r, 0.2, 0.5), range(&mut r, 0.3, 0.5), range(&mut r, 0.35, 0.6)]
        };
        Mat { rgb, rough: range(&mut r, 0.4, 0.9), metal: false }
    };
    let post = match domain {
        Domain::A => [range(&mut r, 0.3, 0.8), range(&mut r, 0.92, 1.08), range(&mut r, 82.0, 95.0), range(&mut r, 0.5, 1.5)],
        Domain::B => [range(&mut r, 0.6, 1.2), range(&mut r, 0.8, 1.2), range(&mut r, 70.0, 88.0), range(&mut r, 1.0, 3.0)],
    };
    Scene { objects, table, cam, light, seed, post }
}

// ---------------------------------------------------------------------------
// Meshes
// ---------------------------------------------------------------------------

/// Positions and triangles, object frame, mm.
#[derive(Debug, Clone, Default)]
pub struct Tris {
    pub p: Vec<[f64; 3]>,
    pub t: Vec<[u32; 3]>,
}

impl Tris {
    fn append(&mut self, o: Tris) {
        let k = self.p.len() as u32;
        self.p.extend(o.p);
        self.t.extend(o.t.into_iter().map(|[a, b, c]| [a + k, b + k, c + k]));
    }
    /// Placed by `pose` on the plane.
    pub fn posed(&self, pose: [f64; 3]) -> Tris {
        let (s, c) = pose[2].to_radians().sin_cos();
        Tris {
            p: self.p.iter().map(|q| [c * q[0] - s * q[1] + pose[0], s * q[0] + c * q[1] + pose[1], q[2]]).collect(),
            t: self.t.clone(),
        }
    }
}

/// An extruded convex polygon (counter-clockwise), `z0..z1`.
fn prism(poly: &[[f64; 2]], z0: f64, z1: f64) -> Tris {
    let n = poly.len();
    let mut m = Tris::default();
    for q in poly {
        m.p.push([q[0], q[1], z0]);
    }
    for q in poly {
        m.p.push([q[0], q[1], z1]);
    }
    let (cx, cy) = poly.iter().fold((0.0, 0.0), |a, q| (a.0 + q[0] / n as f64, a.1 + q[1] / n as f64));
    let b = m.p.len() as u32;
    m.p.push([cx, cy, z0]);
    m.p.push([cx, cy, z1]);
    for i in 0..n {
        let j = (i + 1) % n;
        let (i0, j0, i1, j1) = (i as u32, j as u32, (i + n) as u32, (j + n) as u32);
        m.t.push([i0, j0, j1]);
        m.t.push([i0, j1, i1]);
        m.t.push([b, j0, i0]);
        m.t.push([b + 1, i1, j1]);
    }
    m
}

fn rounded_rect(w: f64, d: f64, r: f64, seg: usize) -> Vec<[f64; 2]> {
    let r = r.clamp(0.0, 0.499 * w.min(d));
    let (a, b) = (w / 2.0 - r, d / 2.0 - r);
    let mut v = Vec::new();
    for (k, (cx, cy)) in [(a, -b), (a, b), (-a, b), (-a, -b)].into_iter().enumerate() {
        let start = -std::f64::consts::FRAC_PI_2 + k as f64 * std::f64::consts::FRAC_PI_2;
        let n = if r > 0.0 { seg } else { 0 };
        for i in 0..=n {
            let t = start + std::f64::consts::FRAC_PI_2 * i as f64 / n.max(1) as f64;
            v.push([cx + r * t.cos(), cy + r * t.sin()]);
            if n == 0 {
                break;
            }
        }
    }
    v
}

fn cuboid(cx: f64, cy: f64, w: f64, d: f64, z0: f64, z1: f64) -> Tris {
    let (a, b) = (w / 2.0, d / 2.0);
    prism(&[[cx - a, cy - b], [cx + a, cy - b], [cx + a, cy + b], [cx - a, cy + b]], z0, z1)
}

/// A box with every edge rounded: a sphere split at its equator and
/// meridians, each octant pushed out to a corner.
fn rounded_box(w: f64, d: f64, h: f64, r: f64, seg: usize) -> Tris {
    let r = r.clamp(0.01, 0.499 * w.min(d).min(h));
    let (a, b, c) = (w / 2.0 - r, d / 2.0 - r, h / 2.0 - r);
    let nu = 4 * seg; // longitude steps
    let nv = 2 * seg; // latitude steps
    let mut m = Tris::default();
    // Duplicate the seams: longitudes at the quadrant boundaries appear on
    // both sides, the equator on both hemispheres, so the stretched quads
    // between duplicates are the flat faces.
    let mut lons = Vec::new();
    for q in 0..4 {
        for i in 0..=seg {
            lons.push((q, (q as f64 + i as f64 / seg as f64) * std::f64::consts::FRAC_PI_2));
        }
    }
    let mut lats = Vec::new();
    for hsign in [-1.0f64, 1.0] {
        for j in 0..=seg {
            let t = j as f64 / seg as f64 * std::f64::consts::FRAC_PI_2;
            lats.push((hsign, if hsign < 0.0 { -std::f64::consts::FRAC_PI_2 + t } else { t }));
        }
    }
    let _ = (nu, nv);
    for &(hs, lat) in &lats {
        for &(q, lon) in &lons {
            let (x, y, z) = (lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin());
            let sx = if q == 0 || q == 3 { 1.0 } else { -1.0 };
            let sy = if q == 0 || q == 1 { 1.0 } else { -1.0 };
            m.p.push([x * r + sx * a, y * r + sy * b, z * r + hs * c + h / 2.0]);
        }
    }
    let cols = lons.len();
    let rows = lats.len();
    for j in 0..rows - 1 {
        for i in 0..cols {
            let i2 = (i + 1) % cols;
            let (p00, p01, p10, p11) = ((j * cols + i) as u32, (j * cols + i2) as u32, ((j + 1) * cols + i) as u32, ((j + 1) * cols + i2) as u32);
            m.t.push([p00, p01, p11]);
            m.t.push([p00, p11, p10]);
        }
    }
    m
}

fn lathe(h: f64, radii: &[f64], seg: usize) -> Tris {
    // Catmull–Rom through the four knots at z = 0, h/3, 2h/3, h.
    let k = radii.len();
    let rows = 16;
    let prof: Vec<(f64, f64)> = (0..=rows)
        .map(|i| {
            let s = i as f64 / rows as f64 * (k - 1) as f64;
            let j = (s.floor() as usize).min(k - 2);
            let t = s - j as f64;
            let p = |m: i64| radii[m.clamp(0, k as i64 - 1) as usize];
            let (p0, p1, p2, p3) = (p(j as i64 - 1), p(j as i64), p(j as i64 + 1), p(j as i64 + 2));
            let r = 0.5 * (2.0 * p1 + (-p0 + p2) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t);
            (r.max(0.5), h * i as f64 / rows as f64)
        })
        .collect();
    let mut m = Tris::default();
    for &(r, z) in &prof {
        for i in 0..seg {
            let a = std::f64::consts::TAU * i as f64 / seg as f64;
            m.p.push([r * a.cos(), r * a.sin(), z]);
        }
    }
    for j in 0..rows {
        for i in 0..seg {
            let i2 = (i + 1) % seg;
            let (a, b, c, d) = ((j * seg + i) as u32, (j * seg + i2) as u32, ((j + 1) * seg + i2) as u32, ((j + 1) * seg + i) as u32);
            m.t.push([a, b, c]);
            m.t.push([a, c, d]);
        }
    }
    let bot = m.p.len() as u32;
    m.p.push([0.0, 0.0, 0.0]);
    m.p.push([0.0, 0.0, h]);
    let top_row = (rows * seg) as u32;
    for i in 0..seg {
        let i2 = (i + 1) % seg;
        m.t.push([bot, i2 as u32, i as u32]);
        m.t.push([bot + 1, top_row + i as u32, top_row + i2 as u32]);
    }
    m
}

fn hex(cx: f64, cy: f64, r: f64) -> Vec<[f64; 2]> {
    (0..6).map(|k| {
        let a = std::f64::consts::FRAC_PI_3 * k as f64 + std::f64::consts::FRAC_PI_6;
        [cx + r * a.cos(), cy + r * a.sin()]
    })
    .collect()
}

/// A hexagonal ring prism: the wall between across-flats `cell` and
/// `cell − 2·wall`, as six slabs.
fn hex_ring(cx: f64, cy: f64, cell: f64, wall: f64, z0: f64, z1: f64) -> Tris {
    let ro = cell / 3f64.sqrt();
    let ri = (cell - 2.0 * wall).max(0.2) / 3f64.sqrt();
    let o = hex(cx, cy, ro);
    let i = hex(cx, cy, ri);
    let mut m = Tris::default();
    for k in 0..6 {
        let j = (k + 1) % 6;
        m.append(prism(&[i[k], o[k], o[j], i[j]], z0, z1));
    }
    m
}

/// The object's mesh in its own frame. Plates go through vcad.
pub fn mesh(o: &Object) -> Result<Tris, String> {
    let p = &o.params;
    let seg = 6;
    Ok(match o.family {
        Family::Plate => {
            let prog = o.plate_program().unwrap();
            let solid = prog.build()?;
            let (pts, tris) = irys_recon::geom::mesh(&solid);
            Tris { p: pts.iter().map(|q| [q.x, q.y, q.z]).collect(), t: tris }
        }
        Family::Lathe => lathe(p[0], &p[1..5], 32),
        Family::Product => {
            let (w, d, h, r) = (p[0], p[1], p[2], p[3]);
            let z0 = if o.tokens[2] == 1 { p[4] } else { 0.0 };
            let mut m = if o.tokens[1] == 0 { prism(&rounded_rect(w, d, r, seg), z0, z0 + h) } else {
                let mut b = rounded_box(w, d, h, r, seg);
                for q in &mut b.p {
                    q[2] += z0;
                }
                b
            };
            if o.tokens[2] == 1 {
                let neck = (0.12 * w).clamp(30.0, 80.0);
                m.append(cuboid(0.0, d * 0.5 + 8.0, neck, 14.0, 4.0, z0 + 0.4 * h));
                m.append(prism(&rounded_rect((0.4 * w).clamp(120.0, 260.0), 160.0f64.min(0.6 * z0 + 80.0), 20.0, seg), 0.0, 6.0).posed([0.0, d * 0.5 + 8.0, 0.0]));
            }
            m
        }
        Family::Grid => {
            let (w, d, h, pitch, kh) = (p[0], p[1], p[2], p[3], p[4]);
            let mut m = prism(&rounded_rect(w, d, 3.0, 3), 0.0, h);
            let rows = o.tokens[2].max(1);
            let cols = ((w - 10.0) / pitch).floor().max(1.0) as usize;
            let cap = 0.8 * pitch;
            for j in 0..rows {
                let y = (j as f64 - (rows as f64 - 1.0) / 2.0) * pitch;
                for i in 0..cols {
                    let x = (i as f64 - (cols as f64 - 1.0) / 2.0) * pitch;
                    m.append(cuboid(x, y, cap, cap, h - 0.5, h + kh));
                }
            }
            m
        }
        Family::Shell => {
            let (w, d, h, t, cell) = (p[0], p[1], p[2], p[3], p[4]);
            let mut m = cuboid(0.0, 0.0, w, d, 0.0, t.min(h * 0.3));
            m.append(cuboid(0.0, -d / 2.0 + t / 2.0, w, t, 0.0, h));
            m.append(cuboid(0.0, d / 2.0 - t / 2.0, w, t, 0.0, h));
            m.append(cuboid(-w / 2.0 + t / 2.0, 0.0, t, d, 0.0, h));
            m.append(cuboid(w / 2.0 - t / 2.0, 0.0, t, d, 0.0, h));
            if o.tokens[1] == 0 {
                let ribs = o.tokens[2];
                for k in 0..ribs {
                    let x = -w / 2.0 + w * (k as f64 + 1.0) / (ribs as f64 + 1.0);
                    m.append(cuboid(x, 0.0, t, d - t, 0.0, 0.7 * h));
                }
            } else {
                // A honeycomb lid: whole cells inside a border of 1.5 cells' wall.
                let lid = (2.0 * t).max(2.0);
                let dx = cell;
                let dy = cell * 3f64.sqrt() / 2.0;
                let ro = cell / 3f64.sqrt();
                let ny = ((d - 2.0 * t) / dy).floor() as i64;
                let nx = ((w - 2.0 * t) / dx).floor() as i64;
                let mut any = false;
                for j in -ny..=ny {
                    for i in -nx..=nx {
                        let x = i as f64 * dx + if j.rem_euclid(2) == 1 { dx / 2.0 } else { 0.0 };
                        let y = j as f64 * dy;
                        if x.abs() + ro < w / 2.0 - t && y.abs() + ro < d / 2.0 - t {
                            m.append(hex_ring(x, y, cell, 0.5 * t.max(1.0), h - lid, h));
                            any = true;
                        }
                    }
                }
                if !any {
                    m.append(cuboid(0.0, 0.0, w, d, h - lid, h));
                }
            }
            m
        }
    })
}

pub fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn normalize(a: [f64; 3]) -> [f64; 3] {
    let n = dot(a, a).sqrt().max(1e-300);
    [a[0] / n, a[1] / n, a[2] / n]
}

// ---------------------------------------------------------------------------
// A plate without booleans: the outline with its holes triangulated by ear
// clipping (holes bridged to the outline), extruded. Same solid as vcad's
// (to tessellation), at a fraction of the cost, for the search's inner loop.
// ---------------------------------------------------------------------------

fn area2(p: &[[f64; 2]]) -> f64 {
    (0..p.len()).map(|i| {
        let (a, b) = (p[i], p[(i + 1) % p.len()]);
        a[0] * b[1] - b[0] * a[1]
    })
    .sum()
}

fn seg_cross(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let o = |p: [f64; 2], q: [f64; 2], r: [f64; 2]| (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
    let (d1, d2, d3, d4) = (o(a, b, c), o(a, b, d), o(c, d, a), o(c, d, b));
    d1 * d2 < -1e-12 && d3 * d4 < -1e-12
}

/// Outline (ccw) with holes (any winding) → one simple polygon by bridges.
fn bridge(outer: &[[f64; 2]], holes: &[Vec<[f64; 2]>]) -> Vec<[f64; 2]> {
    let mut poly = outer.to_vec();
    let mut hs: Vec<Vec<[f64; 2]>> = holes
        .iter()
        .map(|h| {
            let mut h = h.clone();
            if area2(&h) > 0.0 {
                h.reverse();
            }
            h
        })
        .collect();
    hs.sort_by(|a, b| {
        let ma = a.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max);
        let mb = b.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max);
        mb.total_cmp(&ma)
    });
    for (hi, h) in hs.iter().enumerate() {
        let m = (0..h.len()).max_by(|&i, &j| h[i][0].total_cmp(&h[j][0])).unwrap();
        let hm = h[m];
        // The nearest polygon vertex whose bridge crosses nothing.
        let mut order: Vec<usize> = (0..poly.len()).collect();
        order.sort_by(|&i, &j| {
            let di = (poly[i][0] - hm[0]).hypot(poly[i][1] - hm[1]);
            let dj = (poly[j][0] - hm[0]).hypot(poly[j][1] - hm[1]);
            di.total_cmp(&dj)
        });
        let mut pick = order[0];
        'cand: for &i in &order {
            let pv = poly[i];
            let n = poly.len();
            for k in 0..n {
                if seg_cross(hm, pv, poly[k], poly[(k + 1) % n]) {
                    continue 'cand;
                }
            }
            for other in hs[hi..].iter() {
                for k in 0..other.len() {
                    if seg_cross(hm, pv, other[k], other[(k + 1) % other.len()]) {
                        continue 'cand;
                    }
                }
            }
            pick = i;
            break;
        }
        let mut np = poly[..=pick].to_vec();
        for k in 0..=h.len() {
            np.push(h[(m + k) % h.len()]);
        }
        np.extend_from_slice(&poly[pick..]);
        poly = np;
    }
    poly
}

/// Ear clipping of a simple ccw polygon (bridged duplicates allowed).
fn earclip(p: &[[f64; 2]]) -> Vec<[usize; 3]> {
    let mut idx: Vec<usize> = (0..p.len()).collect();
    let mut out = Vec::new();
    let cross = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    let mut guard = 0;
    while idx.len() > 3 && guard < 100_000 {
        guard += 1;
        let n = idx.len();
        let mut clipped = false;
        for k in 0..n {
            let (ia, ib, ic) = (idx[(k + n - 1) % n], idx[k], idx[(k + 1) % n]);
            let (a, b, c) = (p[ia], p[ib], p[ic]);
            if cross(a, b, c) <= 1e-12 {
                continue;
            }
            let inside = idx.iter().any(|&j| {
                if j == ia || j == ib || j == ic {
                    return false;
                }
                let q = p[j];
                if (q[0] - a[0]).abs() + (q[1] - a[1]).abs() < 1e-12 || (q[0] - b[0]).abs() + (q[1] - b[1]).abs() < 1e-12 || (q[0] - c[0]).abs() + (q[1] - c[1]).abs() < 1e-12 {
                    return false;
                }
                cross(a, b, q) >= -1e-12 && cross(b, c, q) >= -1e-12 && cross(c, a, q) >= -1e-12
            });
            if !inside {
                out.push([ia, ib, ic]);
                idx.remove(k);
                clipped = true;
                break;
            }
        }
        if !clipped {
            // Degenerate remainder: fan it.
            for k in 1..idx.len() - 1 {
                out.push([idx[0], idx[k], idx[k + 1]]);
            }
            return out;
        }
    }
    if idx.len() == 3 {
        out.push([idx[0], idx[1], idx[2]]);
    }
    out
}

fn circle(cx: f64, cy: f64, r: f64, n: usize) -> Vec<[f64; 2]> {
    (0..n).map(|k| {
        let a = std::f64::consts::TAU * k as f64 / n as f64;
        [cx + r * a.cos(), cy + r * a.sin()]
    })
    .collect()
}

/// A plate's mesh without vcad.
pub fn plate_fast(prog: &Program) -> Tris {
    let r = match prog.corner {
        Corner::Fillet(r) => r,
        _ => 0.0,
    };
    let outer = rounded_rect(prog.w, prog.d, r, 6);
    let mut holes = Vec::new();
    for f in &prog.features {
        for (x, y, hr) in f.holes() {
            holes.push(circle(x, y, hr, 16));
        }
    }
    let poly = bridge(&outer, &holes);
    let tris = earclip(&poly);
    let h = prog.h;
    let mut m = Tris::default();
    let n = poly.len() as u32;
    for q in &poly {
        m.p.push([q[0], q[1], h]);
    }
    for q in &poly {
        m.p.push([q[0], q[1], 0.0]);
    }
    for t in &tris {
        m.t.push([t[0] as u32, t[1] as u32, t[2] as u32]);
        m.t.push([n + t[0] as u32, n + t[2] as u32, n + t[1] as u32]);
    }
    let mut wall = |loop_: &[[f64; 2]]| {
        let k = loop_.len();
        let b = m.p.len() as u32;
        for q in loop_ {
            m.p.push([q[0], q[1], 0.0]);
            m.p.push([q[0], q[1], h]);
        }
        for i in 0..k {
            let j = (i + 1) % k;
            let (a0, a1, b0, b1) = (b + 2 * i as u32, b + 2 * i as u32 + 1, b + 2 * j as u32, b + 2 * j as u32 + 1);
            m.t.push([a0, b0, b1]);
            m.t.push([a0, b1, a1]);
        }
    };
    wall(&outer);
    for hl in &holes {
        let mut r = hl.clone();
        r.reverse();
        wall(&r);
    }
    m
}

/// [`mesh`], with plates through [`plate_fast`].
pub fn mesh_fast(o: &Object) -> Result<Tris, String> {
    match o.family {
        Family::Plate => Ok(plate_fast(&o.plate_program().unwrap())),
        _ => mesh(o),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn volume(m: &Tris) -> f64 {
        m.t.iter()
            .map(|t| {
                let (a, b, c) = (m.p[t[0] as usize], m.p[t[1] as usize], m.p[t[2] as usize]);
                dot(a, cross(b, c)) / 6.0
            })
            .sum()
    }

    #[test]
    fn fast_plates_match_the_program_volume() {
        let mut r = Mix(3);
        for k in 0..40 {
            let c = if k % 2 == 0 { Class::Bracket } else { Class::GearPlate };
            let o = sample_object(&mut r, c);
            let p = o.plate_program().unwrap();
            let v = volume(&plate_fast(&p));
            let want = p.volume();
            assert!((v - want).abs() / want < 0.02, "{}: {v} vs {want}", o.to_text());
        }
    }

    #[test]
    fn every_class_samples_and_meshes() {
        let mut r = Mix(5);
        for c in CLASS_LIST {
            let o = sample_object(&mut r, c);
            let m = mesh_fast(&o).unwrap();
            assert!(!m.t.is_empty() && volume(&m) > 0.0, "{c:?}");
        }
    }

    #[test]
    fn scenes_are_deterministic() {
        let a = sample_scene(11, Domain::A, (128, 96));
        let b = sample_scene(11, Domain::A, (128, 96));
        assert_eq!(serde_json::to_string(&a).unwrap(), serde_json::to_string(&b).unwrap());
        assert!(a.objects.len() >= 3);
    }
}
