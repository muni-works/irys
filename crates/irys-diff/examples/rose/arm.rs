//! rose full — the whole robot, its left arm posable — read from the rose
//! repo's own `.loon` CAD and tessellated by vcad 0.10's kernel; shared by
//! `rose`, `rose_fit` and `rose_check`.
//!
//! The source is the canonical assembly, `full/cad/rose-full.loon`: its
//! `[use links.X [..]]` / `[use actuators [..]]` forms name the modules and
//! its `[root NAME material]` forms are the parts, every one already placed
//! in the robot's frame at zero joint angles (rose_full_loon.py emits world
//! millimetres; the assembly carries no transforms). So at `φ = 0` this is
//! rose-full.loon, root for root. The modules' `[let NAME EXPR]` forms are
//! evaluated onto `vcad_kernel::Solid` over the vocabulary of vcad's
//! `hardware/rose-full/plates.loon` (`plate-*`, `bore-*`, `act-*`, ported
//! call for call because vcad-loon is not on crates.io), plus `translate`,
//! `rotate`, `union`, `difference` and `extrude`/`sketch` (the chest panels).
//!
//! The joints come from the same files: each arm joint is its actuator's
//! envelope in `actuators.loon` (`[act-AX d lo hi cu cv]`: axis `AX` through
//! `(cu, cv)`). What rides which joint is rose_full_links.py's role table —
//! shoA is the shoulder-pitch output and the roll stator, shoB the roll
//! output and the elbow stator, elbA the elbow output and the wrist stator,
//! the hand the wrist output — so each actuator rides the link it bolts its
//! stator to. The wrist is held at 0.

#![allow(dead_code)]

use std::{collections::HashMap, fs, io::Write, path::Path, time::Instant};

use irys_core::{Point3, Transform, TriMesh, Vec3};
use irys_diff::prb::{Env, Opaque, Placement, Scene, Surface, grid_floor};
use irys_diff::vis::{Rig, about};
use irys_prims::analytic::Analytic;
use vcad_kernel::Solid;
use vcad_kernel::vcad_kernel_math as km;
use vcad_kernel::vcad_kernel_sketch::{SketchProfile, SketchSegment};

/// The pose the target is rendered at: shoulder pitch, shoulder roll,
/// elbow (radians) — reaching forward and out from the side, elbow bent.
pub const TRUE_PHI: [f64; 3] = [-0.45, 0.28, -1.05];
/// Bead-blasted clear-anodised aluminium: `(α, F0 at 450, 550, 650 nm)`.
pub const TRUE_THETA: [f64; 4] = [0.30, 0.86, 0.88, 0.90];

// ---- the loon subset ----

#[derive(Debug, Clone)]
pub enum Expr {
    Num(f64),
    Sym(String),
    List(Vec<Expr>),
}

fn parse(src: &str) -> Vec<Expr> {
    let mut toks = Vec::new();
    let mut cur = String::new();
    let mut chars = src.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            ';' => {
                for c in chars.by_ref() {
                    if c == '\n' {
                        break;
                    }
                }
            }
            '"' => {
                let mut s = String::from("\"");
                for c in chars.by_ref() {
                    if c == '"' {
                        break;
                    }
                    s.push(c);
                }
                toks.push(s);
            }
            // `#[..]` is a vector literal (a sketch's segments): a list here.
            '#' if chars.peek() == Some(&'[') => {}
            '[' | ']' => {
                if !cur.is_empty() {
                    toks.push(std::mem::take(&mut cur));
                }
                toks.push(c.to_string());
            }
            c if c.is_whitespace() => {
                if !cur.is_empty() {
                    toks.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        toks.push(cur);
    }
    fn one(t: &[String], i: &mut usize) -> Expr {
        let tok = &t[*i];
        *i += 1;
        if tok == "[" {
            let mut v = Vec::new();
            while t[*i] != "]" {
                v.push(one(t, i));
            }
            *i += 1;
            Expr::List(v)
        } else if let Ok(x) = tok.parse::<f64>() {
            Expr::Num(x)
        } else {
            Expr::Sym(tok.clone())
        }
    }
    let mut i = 0;
    let mut out = Vec::new();
    while i < toks.len() {
        out.push(one(&toks, &mut i));
    }
    out
}

fn head(e: &Expr) -> Option<(&str, &[Expr])> {
    match e {
        Expr::List(v) => match v.first() {
            Some(Expr::Sym(s)) => Some((s.as_str(), &v[1..])),
            _ => None,
        },
        _ => None,
    }
}

fn sym(e: &Expr) -> &str {
    match e {
        Expr::Sym(s) => s.trim_start_matches('"'),
        _ => panic!("loon: expected a name, got {e:?}"),
    }
}

/// A module's top-level `[let NAME EXPR]`s.
fn lets(src: &str) -> HashMap<String, Expr> {
    parse(src)
        .iter()
        .filter_map(|e| match head(e) {
            Some(("let", [n, e])) => Some((sym(n).to_string(), e.clone())),
            _ => None,
        })
        .collect()
}

/// One `[root ..]` of the assembly: its name, the module that exports it,
/// its material, and its expression.
#[derive(Debug, Clone)]
pub struct Root {
    pub name: String,
    pub module: String,
    pub material: String,
    pub expr: Expr,
}

/// Read `full/cad/rose-full.loon` and resolve every root through the
/// module its `[use ..]` imports it from, in the assembly's order.
pub fn assembly(rose: &Path) -> Vec<Root> {
    let cad = rose.join("full/cad");
    let src = fs::read_to_string(cad.join("rose-full.loon")).expect("rose full/cad/rose-full.loon");
    let forms = parse(&src);
    let mut from: HashMap<String, String> = HashMap::new();
    let mut modules: HashMap<String, HashMap<String, Expr>> = HashMap::new();
    let mut roots = Vec::new();
    for f in &forms {
        match head(f) {
            Some(("use", [m, Expr::List(names)])) => {
                let m = sym(m).to_string();
                let file = cad.join(format!("{}.loon", m.replace('.', "/")));
                let text = fs::read_to_string(&file)
                    .unwrap_or_else(|_| panic!("rose module {}", file.display()));
                modules.insert(m.clone(), lets(&text));
                for n in names {
                    from.insert(sym(n).to_string(), m.clone());
                }
            }
            Some(("root", [n, mat])) => {
                let name = sym(n).to_string();
                let module = from
                    .get(&name)
                    .unwrap_or_else(|| panic!("root {name} is not imported"))
                    .clone();
                let expr = modules[&module]
                    .get(&name)
                    .unwrap_or_else(|| panic!("{module} does not export {name}"))
                    .clone();
                roots.push(Root {
                    name,
                    module: module.trim_start_matches("links.").to_string(),
                    material: sym(mat).to_string(),
                    expr,
                });
            }
            _ => {}
        }
    }
    roots
}

// plates.loon: "Conventions (vcad): Z-up, mm, degrees. cube corner at
// origin; cylinder axis +Z, base at z=0. Subject-last, so [difference tool
// subject]." The rotations put a +Z cylinder on +X ([rotate 0 90 0]) and +Y
// ([rotate -90 0 0]).

const CORNER_SEG: u32 = 24;
const BORE_SEG: u32 = 32;
const ACT_SEG: u32 = 64;

/// "Every helper takes a normal axis in its name (-x/-y/-z). u/v are the two
/// in-plane axes in x<y<z order": `(n, u, v)` → world.
fn place(axis: usize, p: [f64; 3]) -> [f64; 3] {
    let (u, v) = match axis {
        0 => (1, 2),
        1 => (0, 2),
        _ => (0, 1),
    };
    let mut w = [0.0; 3];
    w[axis] = p[0];
    w[u] = p[1];
    w[v] = p[2];
    w
}

fn cube(axis: usize, at: [f64; 3], size: [f64; 3]) -> Solid {
    let s = place(axis, size);
    let a = place(axis, at);
    Solid::cube(s[0], s[1], s[2]).translate(a[0], a[1], a[2])
}

/// A +Z cylinder turned onto `axis`.
fn cyl(axis: usize, r: f64, h: f64, seg: u32) -> Solid {
    let c = Solid::cylinder(r, h, seg);
    match axis {
        0 => c.rotate(0.0, 90.0, 0.0),
        1 => c.rotate(-90.0, 0.0, 0.0),
        _ => c,
    }
}

/// "Rounded-rectangle plate. Corner radius r. Built as two overlapping boxes
/// plus four corner cylinders."
fn plate(axis: usize, [n0, n1, u0, u1, v0, v1, r]: [f64; 7]) -> Solid {
    let t = n1 - n0;
    let core = cube(axis, [n0, u0 + r, v0], [t, u1 - u0 - 2.0 * r, v1 - v0]);
    let cross = cube(axis, [n0, u0, v0 + r], [t, u1 - u0, v1 - v0 - 2.0 * r]);
    let cn = |cu: f64, cv: f64| {
        let a = place(axis, [n0, cu, cv]);
        cyl(axis, r, t, CORNER_SEG).translate(a[0], a[1], a[2])
    };
    core.union(&cross)
        .union(&cn(u0 + r, v0 + r))
        .union(&cn(u0 + r, v1 - r))
        .union(&cn(u1 - r, v0 + r))
        .union(&cn(u1 - r, v1 - r))
}

/// "Through-bores, overshooting the plate both ways so the cut exits cleanly."
fn bore(axis: usize, [d, cu, cv, n0, n1]: [f64; 5]) -> Solid {
    let a = place(axis, [n0 - 2.0, cu, cv]);
    cyl(axis, d / 2.0, n1 - n0 + 4.0, BORE_SEG).translate(a[0], a[1], a[2])
}

/// "Actuator envelope stand-ins ... d = envelope diameter, lo..hi the axial span."
fn act(axis: usize, [d, lo, hi, cu, cv]: [f64; 5]) -> Solid {
    let a = place(axis, [lo, cu, cv]);
    cyl(axis, d / 2.0, hi - lo, ACT_SEG).translate(a[0], a[1], a[2])
}

fn axis_of(name: &str) -> usize {
    match name.as_bytes().last() {
        Some(b'x') => 0,
        Some(b'y') => 1,
        _ => 2,
    }
}

fn num(e: &Expr) -> f64 {
    match e {
        Expr::Num(x) => *x,
        _ => panic!("loon: expected a number, got {e:?}"),
    }
}

fn nums<const N: usize>(a: &[Expr]) -> [f64; N] {
    let v: Vec<f64> = a.iter().map(num).collect();
    v.try_into()
        .unwrap_or_else(|v| panic!("loon: {N} numbers, got {v:?}"))
}

/// `[sketch ox oy oz  xx xy xz  yx yy yz  #[segments]]`, segments
/// `[line x0 y0 x1 y1]` and `[arc x0 y0 x1 y1 cx cy ccw]`.
fn sketch(a: &[Expr]) -> SketchProfile {
    let o: [f64; 9] = nums(&a[..9]);
    let Expr::List(segs) = &a[9] else {
        panic!("sketch: segments")
    };
    let p2 = |x: &Expr, y: &Expr| km::Point2::new(num(x), num(y));
    let segs = segs
        .iter()
        .map(|s| match head(s) {
            Some(("line", [a, b, c, d])) => SketchSegment::Line {
                start: p2(a, b),
                end: p2(c, d),
            },
            Some(("arc", [a, b, c, d, e, f, ccw])) => SketchSegment::Arc {
                start: p2(a, b),
                end: p2(c, d),
                center: p2(e, f),
                ccw: sym(ccw) == "true",
            },
            _ => panic!("sketch: segment {s:?}"),
        })
        .collect();
    SketchProfile::new(
        km::Point3::new(o[0], o[1], o[2]),
        km::Vec3::new(o[3], o[4], o[5]),
        km::Vec3::new(o[6], o[7], o[8]),
        segs,
    )
    .expect("sketch profile")
}

pub fn eval(e: &Expr) -> Solid {
    let (h, a) = head(e).unwrap_or_else(|| panic!("loon: expected a form, got {e:?}"));
    match h {
        "union" => eval(&a[0]).union(&eval(&a[1])),
        "difference" => eval(&a[1]).difference(&eval(&a[0])),
        "translate" => {
            let [x, y, z] = nums(&a[..3]);
            eval(&a[3]).translate(x, y, z)
        }
        "rotate" => {
            let [x, y, z] = nums(&a[..3]);
            eval(&a[3]).rotate(x, y, z)
        }
        "extrude" => {
            let [x, y, z] = nums(&a[..3]);
            let Some(("sketch", s)) = head(&a[3]) else {
                panic!("extrude wants a sketch")
            };
            Solid::extrude(sketch(s), km::Vec3::new(x, y, z)).expect("extrude")
        }
        h if h.starts_with("plate-") => plate(axis_of(h), nums(a)),
        h if h.starts_with("bore-") => bore(axis_of(h), nums(a)),
        h if h.starts_with("act-") => act(axis_of(h), nums(a)),
        h => panic!("loon: {h} is outside the subset this reader takes"),
    }
}

// ---- tessellation, cached ----

fn tessellate(s: &Solid) -> TriMesh {
    let m = s.to_mesh(32);
    let positions = m
        .vertices
        .chunks_exact(3)
        .map(|p| Point3::new(p[0] as f64, p[1] as f64, p[2] as f64))
        .collect();
    TriMesh::new(positions, Vec::new(), &m.indices)
}

pub fn concat(meshes: &[&TriMesh]) -> TriMesh {
    let (mut pos, mut idx) = (Vec::new(), Vec::new());
    for m in meshes {
        let base = pos.len() as u32;
        pos.extend_from_slice(m.positions());
        idx.extend(m.triangles().iter().flatten().map(|i| i + base));
    }
    TriMesh::new(pos, Vec::new(), &idx)
}

fn save(path: &Path, m: &TriMesh) {
    let mut f = fs::File::create(path).unwrap();
    f.write_all(&(m.positions().len() as u64).to_le_bytes())
        .unwrap();
    for p in m.positions() {
        for c in [p.x, p.y, p.z] {
            f.write_all(&c.to_le_bytes()).unwrap();
        }
    }
    for i in m.triangles().iter().flatten() {
        f.write_all(&i.to_le_bytes()).unwrap();
    }
}

fn load(path: &Path) -> Option<TriMesh> {
    let b = fs::read(path).ok()?;
    let n = u64::from_le_bytes(b.get(..8)?.try_into().ok()?) as usize;
    let f = |i: usize| f64::from_le_bytes(b[8 + 8 * i..16 + 8 * i].try_into().unwrap());
    let pos = (0..n)
        .map(|i| Point3::new(f(3 * i), f(3 * i + 1), f(3 * i + 2)))
        .collect();
    let idx: Vec<u32> = b[8 + 24 * n..]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Some(TriMesh::new(pos, Vec::new(), &idx))
}

fn hash(s: &str) -> u64 {
    // FNV-1a: the cache key is the source text.
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
    })
}

/// The robot's trees: [`ARM_TREES`], then the body's (the ground is
/// appended last, at index `meshes.len()`).
pub struct Arm {
    pub meshes: Vec<TriMesh>,
    pub names: Vec<String>,
    /// Which trees are actuator envelopes (drawn dark).
    pub dark: Vec<bool>,
    /// The four arm joints at zero pose: (pivot, unit axis), shoulder
    /// pitch, shoulder roll, elbow, wrist yaw.
    pub joints: [(Point3, Vec3); 4],
    /// Millimetres the robot's frame is lifted so its soles are on `z = 0`.
    pub lift: f64,
    /// Which rose file it came from.
    pub source: String,
    /// Roots in the assembly, and how many of them were tessellated now.
    pub roots: usize,
    pub tessellated: usize,
    pub seconds: f64,
}

/// The arm's trees: its links (conductor) and its actuators (dark). The
/// rest of the robot follows, one tree per link module and side (`body
/// shoA R`, …, `body actuators L`), so no tree is deep enough to overflow
/// the GPU traversal stack; [`Arm::names`] has them all.
pub const ARM_TREES: [&str; 8] = [
    "shoA",
    "shoB",
    "elbA",
    "hand",
    "act_sho_pitch",
    "act_sho_roll",
    "act_elb_pitch",
    "act_wrist_yaw",
];

/// rose_full_links.py's chain for the left arm: each joint's actuator in
/// actuators.loon, and the link module that is its output.
const CHAIN: [(&str, &str); 4] = [
    ("act_L_sho_pitch", "shoA"),
    ("act_L_sho_roll", "shoB"),
    ("act_L_elb_pitch", "elbA"),
    ("act_L_wrist_yaw", "hand"),
];

/// Read rose-full.loon and tessellate every root (in parallel, cached by
/// expression under the temp dir), sorted into trees.
pub fn load_arm(rose: &Path) -> Arm {
    let t = Instant::now();
    let roots = assembly(rose);
    let cache = std::env::temp_dir().join("irys-rose-full");
    fs::create_dir_all(&cache).unwrap();
    let key = |r: &Root| {
        cache.join(format!(
            "{:016x}.mesh",
            hash(&format!("{:?} {CORNER_SEG} {BORE_SEG} {ACT_SEG}", r.expr))
        ))
    };
    // Tessellate what is not cached, biggest expressions first, on every core.
    let mut todo: Vec<&Root> = roots.iter().filter(|r| !key(r).exists()).collect();
    todo.sort_by_key(|r| std::cmp::Reverse(format!("{:?}", r.expr).len()));
    let tessellated = todo.len();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(r) = todo.get(i) else { break };
                    let ti = Instant::now();
                    let m = tessellate(&eval(&r.expr));
                    save(&key(r), &m);
                    eprintln!(
                        "  {:24} {:6} triangles {:6.1?}",
                        r.name,
                        m.triangles().len(),
                        ti.elapsed()
                    );
                }
            });
        }
    });
    let meshes: Vec<TriMesh> = roots
        .iter()
        .map(|r| load(&key(r)).expect("cached mesh"))
        .collect();

    let mut names: Vec<String> = ARM_TREES.iter().map(|s| s.to_string()).collect();
    let mut groups: Vec<Vec<&TriMesh>> = vec![Vec::new(); names.len()];
    for (r, m) in roots.iter().zip(&meshes) {
        let k = if let Some(k) = CHAIN.iter().position(|(a, _)| *a == r.name) {
            4 + k
        } else if let Some(k) = CHAIN
            .iter()
            .position(|(_, m)| *m == r.module && r.name.starts_with("L_"))
        {
            k
        } else {
            let side = if r.module == "actuators" {
                r.name.split('_').nth(1).unwrap_or("C")
            } else {
                r.name.split('_').next().unwrap_or("C")
            };
            let n = format!("body {} {side}", r.module);
            names.iter().position(|x| *x == n).unwrap_or_else(|| {
                names.push(n);
                groups.push(Vec::new());
                names.len() - 1
            })
        };
        groups[k].push(m);
    }
    let dark = names
        .iter()
        .enumerate()
        .map(|(k, n)| (4..8).contains(&k) || n.starts_with("body actuators"))
        .collect();
    let by_name: HashMap<&str, &Root> = roots.iter().map(|r| (r.name.as_str(), r)).collect();
    let joints = CHAIN.map(|(a, _)| {
        let (h, args) = head(&by_name[a].expr).unwrap();
        let ax = axis_of(h);
        let [_, lo, hi, cu, cv] = nums::<5>(args);
        let p = place(ax, [(lo + hi) / 2.0, cu, cv]);
        let mut d = [0.0; 3];
        d[ax] = 1.0;
        (Point3::new(p[0], p[1], p[2]), Vec3::new(d[0], d[1], d[2]))
    });
    let lift = -meshes
        .iter()
        .flat_map(|m| m.positions().iter().map(|p| p.z))
        .fold(f64::INFINITY, f64::min);
    let meshes: Vec<TriMesh> = groups.iter().map(|g| concat(g)).collect();
    let source = rose.join("full/cad/rose-full.loon").display().to_string();
    eprintln!(
        "rose full: {} roots from {source} ({tessellated} tessellated, the rest cached) in {:.1?}; lift {lift:.1} mm",
        roots.len(),
        t.elapsed()
    );
    Arm {
        meshes,
        names,
        dark,
        joints,
        lift,
        source,
        roots: roots.len(),
        tessellated,
        seconds: t.elapsed().as_secs_f64(),
    }
}

// ---- the rig and the scene ----

/// Object → world for (shoA, shoB, elbA) at joint angles
/// `(shoulder pitch, shoulder roll, elbow)`, radians, about the actuators'
/// own axes, after `base`.
fn chain_from(base: &Transform, j: &[(Point3, Vec3); 4], phi: &[f64]) -> [Transform; 3] {
    let a = base.then(&about(j[0].0, j[0].1, phi[0]));
    let b = a.then(&about(j[1].0, j[1].1, phi[1]));
    let c = b.then(&about(j[2].0, j[2].1, phi[2]));
    [a, b, c]
}

impl Arm {
    /// The robot's frame → world: soles on the ground.
    pub fn base(&self) -> Transform {
        Transform::translation(0.0, 0.0, self.lift)
    }

    pub fn chain(&self, phi: &[f64]) -> [Transform; 3] {
        chain_from(&self.base(), &self.joints, phi)
    }

    pub fn rig(&self) -> Rig<'static> {
        let (base, joints) = (self.base(), self.joints);
        Rig {
            moving: MOVING.to_vec(),
            place: Box::new(move |phi: &[f64]| {
                let c = chain_from(&base, &joints, phi);
                RIDES.iter().map(|&r| c[r].clone()).collect()
            }),
        }
    }
}

/// Placements by tree: 0 shoA, 1 shoB, 2 elbA, 3 hand, 4–7 actuators,
/// then the body's trees and the ground. Placement `k` is tree `k`.
pub const MOVING: [usize; 7] = [0, 1, 2, 3, 5, 6, 7];

/// What each moving placement rides on: 0 shoA, 1 shoB, 2 elbA (the hand
/// rides the wrist, held at 0; each actuator rides its stator's link).
const RIDES: [usize; 7] = [0, 1, 2, 2, 0, 1, 2];

/// Dark anodised actuator bodies (`actuators.loon`'s 0.10 grey).
pub const ACT_TINT: f64 = 0.19;

/// The beauty camera: eye, target, half field of view (degrees), in world
/// millimetres (soles at z = 0, the left arm at y ≈ 255).
pub const BEAUTY_CAM: ([f64; 3], [f64; 3], f64) =
    ([1000.0, 2600.0, 1250.0], [60.0, 140.0, 870.0], 21.5);
/// The fit's camera ([`arm_scene_from`]): front-left and a little below
/// the arm, so the forearm reads against the sky.
pub const FIT_CAM: ([f64; 3], [f64; 3], f64) =
    ([1074.0, 1526.0, 1021.0], [170.0, 235.0, 1270.0], 11.0);

/// The robot on the ground, its arm posed at `phi`, `spp` paths per pixel,
/// from the beauty camera.
pub fn scene(arm: &Arm, size: (usize, usize), spp: usize, phi: &[f64]) -> Scene {
    scene_from(arm, size, spp, phi, BEAUTY_CAM)
}

/// [`scene`] from camera `(eye, target, half fov°)`.
pub fn scene_from(
    arm: &Arm,
    size: (usize, usize),
    spp: usize,
    phi: &[f64],
    cam: ([f64; 3], [f64; 3], f64),
) -> Scene {
    let mut s = build_scene(arm, arm.meshes.len(), size, spp, phi, cam);
    // The sun over the robot's left shoulder, so the posed arm is lit.
    s.env.sun = Vec3::new(0.35, 0.6, 0.75).normalize();
    s
}

/// The left arm alone — its links and actuators, placed exactly as in the
/// robot — against the ground and sky: the fit's scene, where the arm is
/// the only thing in frame made of the finish being fitted.
pub fn arm_scene_from(
    arm: &Arm,
    size: (usize, usize),
    spp: usize,
    phi: &[f64],
    cam: ([f64; 3], [f64; 3], f64),
) -> Scene {
    let mut s = build_scene(arm, ARM_TREES.len(), size, spp, phi, cam);
    s.floor_checker = None;
    s
}

fn build_scene(
    arm: &Arm,
    n: usize,
    size: (usize, usize),
    spp: usize,
    phi: &[f64],
    cam: ([f64; 3], [f64; 3], f64),
) -> Scene {
    let mut trees: Vec<Opaque> = arm.meshes[..n]
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let s = if arm.dark[i] {
                Surface::Diffuse
            } else {
                Surface::Conductor
            };
            Opaque::new(m.clone(), s, Analytic::from_prims(Vec::new()), &[])
        })
        .collect();
    trees.push(Opaque::new(
        grid_floor(16, 8000.0),
        Surface::Diffuse,
        Analytic::from_prims(Vec::new()),
        &[],
    ));
    let c = arm.chain(phi);
    let mut p = Vec::new();
    for k in 0..n {
        let to_world = match MOVING.iter().position(|&m| m == k) {
            Some(i) => c[RIDES[i]].clone(),
            None => arm.base(),
        };
        let pl = Placement::new(k, to_world);
        p.push(if arm.dark[k] { pl.tinted(ACT_TINT) } else { pl });
    }
    p.push(Placement::new(n, Transform::identity()));
    let mut s = Scene::instanced(trees, p, None, size, spp, (p3(cam.0), p3(cam.1)), cam.2);
    s.nee = true;
    s.depth = 5;
    s.floor_albedo = 0.55;
    s.floor_checker = Some(250.0);
    s.env = Env {
        sun: Vec3::new(0.45, -0.55, 0.75).normalize(),
        kappa: 900.0,
        sun_radiance: 5.0 * 900.0 / (2.0 * std::f64::consts::PI),
        sky: 0.8,
    };
    s
}

fn p3(a: [f64; 3]) -> Point3 {
    Point3::new(a[0], a[1], a[2])
}

/// The rose checkout: `$ROSE`, or `~/Developer/rose`.
pub fn rose_dir() -> std::path::PathBuf {
    std::env::var_os("ROSE")
        .map(Into::into)
        .unwrap_or_else(|| Path::new(&std::env::var("HOME").unwrap()).join("Developer/rose"))
}

pub fn ppm(path: &Path, (w, h): (usize, usize), img: &[f32], exposure: f32) {
    let mut f = fs::File::create(path).unwrap();
    write!(f, "P6\n{w} {h}\n255\n").unwrap();
    let srgb = |x: f32| {
        let x = (x * exposure).clamp(0.0, 1.0);
        let s = if x <= 0.0031308 {
            12.92 * x
        } else {
            1.055 * x.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0 + 0.5) as u8
    };
    f.write_all(&img.iter().map(|&x| srgb(x)).collect::<Vec<_>>())
        .unwrap();
}

/// vcad primitives for checking the edge term on vcad's tessellation.
pub fn test_parts() -> Vec<(&'static str, TriMesh)> {
    vec![
        ("ours: cuboid", irys_diff::prb::cuboid(30.0, 20.0, 10.0)),
        ("vcad cube", tessellate(&Solid::cube(60.0, 40.0, 20.0))),
        (
            "vcad cylinder",
            tessellate(&Solid::cylinder(20.0, 40.0, 32)),
        ),
        (
            "plate",
            tessellate(&plate(2, [0.0, 3.0, 0.0, 60.0, 0.0, 60.0, 20.0])),
        ),
        (
            "plate+bore",
            tessellate(
                &plate(2, [0.0, 3.0, 0.0, 60.0, 0.0, 60.0, 20.0])
                    .difference(&bore(2, [20.0, 30.0, 30.0, 0.0, 3.0])),
            ),
        ),
    ]
}

/// Welded edge valence counts: (1, 2, more) — 1 is a crack or T-junction.
pub fn valence(m: &TriMesh) -> (usize, usize, usize, usize) {
    let key = |p: &Point3| [p.x, p.y, p.z].map(|c| (c * 1e5).round() as i64);
    let mut map: HashMap<_, usize> = HashMap::new();
    let pos = m.positions();
    for t in m.triangles() {
        for e in 0..3 {
            let (a, b) = (key(&pos[t[e] as usize]), key(&pos[t[(e + 1) % 3] as usize]));
            *map.entry(if a < b { (a, b) } else { (b, a) }).or_default() += 1;
        }
    }
    let c = |n: usize| map.values().filter(|&&v| v == n).count();
    (
        m.triangles().len(),
        c(1),
        c(2),
        map.values().filter(|&&v| v > 2).count(),
    )
}
