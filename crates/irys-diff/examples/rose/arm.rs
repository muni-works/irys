//! rose full's left arm, read from the rose repo's own `.loon` CAD and
//! tessellated by vcad 0.10's kernel; shared by `rose` and `rose_fit`.
//!
//! The link files (`full/cad/links/{shoA,shoB,elbA,hand}.loon`) and the
//! actuator envelopes (`full/cad/actuators.loon`) are parsed as they are —
//! generated `[let NAME EXPR]` forms over `plate-*`, `bore-*`, `act-*`,
//! `union` and `difference` — and evaluated onto vcad's `Part`. The
//! vocabulary those names come from, vcad's `hardware/rose-full/plates.loon`,
//! is ported here call for call (its comments are quoted at each), because
//! vcad-loon is not on crates.io; the links themselves are read, not ported.
//! Every link's plates are one bottom-level tree; the actuators are trees of
//! their own, carried by the link whose stator plate they bolt to.
//!
//! The chain (`full/rose_full_layout.py`'s `arm('L')`): shoulder pitch about
//! `y` through `(0, 255, 462)` carries shoA; shoulder roll about `x` through
//! `(·, 255, 362)` carries shoB; elbow pitch about `y` through
//! `(0, 255, 235.2)` carries elbA and the hand (wrist yaw held at 0). In the
//! scene the arm is moved to `y = 0` and lifted `LIFT` off the ground, its
//! shoulder actuator on a dark post.

#![allow(dead_code)]

use std::{collections::HashMap, fs, io::Write, path::Path, time::Instant};

use irys_core::{Point3, Transform, TriMesh, Vec3};
use irys_diff::prb::{Env, Opaque, Placement, Scene, Surface, grid_floor};
use irys_diff::vis::{Rig, about};
use irys_prims::analytic::Analytic;
use vcad::Part;

/// The pose the target is rendered at: shoulder pitch, shoulder roll,
/// elbow (radians) — reaching forward and out, elbow bent.
pub const TRUE_PHI: [f64; 3] = [-0.45, -0.28, -1.05];
/// Bead-blasted clear-anodised aluminium: `(α, F0 at 450, 550, 650 nm)`.
pub const TRUE_THETA: [f64; 4] = [0.30, 0.86, 0.88, 0.90];

/// Millimetres the arm's CAD origin is lifted off the ground.
pub const LIFT: f64 = 160.0;
/// The CAD's arm plane, moved to `y = 0`.
pub const ARM_Y: f64 = 255.0;

pub const SHO_PITCH: [f64; 3] = [0.0, ARM_Y, 462.0];
pub const SHO_ROLL: [f64; 3] = [0.0, ARM_Y, 362.0];
pub const ELBOW: [f64; 3] = [0.0, ARM_Y, 235.2];

// ---- the loon subset ----

#[derive(Debug, Clone)]
enum Expr {
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

/// The file's top-level `[let NAME EXPR]`s.
fn lets(src: &str) -> HashMap<String, Expr> {
    parse(src)
        .into_iter()
        .filter_map(|e| match e {
            Expr::List(v) if matches!(&v[0], Expr::Sym(s) if s == "let") => match (&v[1], &v[2]) {
                (Expr::Sym(n), e) => Some((n.clone(), e.clone())),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

// plates.loon: "Conventions (vcad): Z-up, mm, degrees. cube corner at
// origin; cylinder axis +Z, base at z=0. Subject-last, so [difference tool
// subject]." The rotations put a +Z cylinder on +X ([rotate 0 90 0]) and +Y
// ([rotate -90 0 0]).

const CORNER_SEG: u32 = 24;
const BORE_SEG: u32 = 32;
const ACT_SEG: u32 = 64;

/// A box from `lo` along `size`, axes in (n, u, v) order mapped to world by `ax`.
fn place(axis: usize, p: [f64; 3]) -> [f64; 3] {
    // (n, u, v) → world: n is the normal axis, u < v the other two.
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

fn cube(axis: usize, at: [f64; 3], size: [f64; 3]) -> Part {
    let s = place(axis, size);
    let a = place(axis, at);
    Part::cube("cube", s[0], s[1], s[2]).translate(a[0], a[1], a[2])
}

/// A +Z cylinder turned onto `axis`.
fn cyl(axis: usize, r: f64, h: f64, seg: u32) -> Part {
    let c = Part::cylinder("cyl", r, h, seg);
    match axis {
        0 => c.rotate(0.0, 90.0, 0.0),
        1 => c.rotate(-90.0, 0.0, 0.0),
        _ => c,
    }
}

/// "Rounded-rectangle plate. Corner radius r. Built as two overlapping boxes
/// plus four corner cylinders."
fn plate(axis: usize, [n0, n1, u0, u1, v0, v1, r]: [f64; 7]) -> Part {
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
fn bore(axis: usize, [d, cu, cv, n0, n1]: [f64; 5]) -> Part {
    let a = place(axis, [n0 - 2.0, cu, cv]);
    cyl(axis, d / 2.0, n1 - n0 + 4.0, BORE_SEG).translate(a[0], a[1], a[2])
}

/// "Actuator envelope stand-ins ... d = envelope diameter, lo..hi the axial span."
fn act(axis: usize, [d, lo, hi, cu, cv]: [f64; 5]) -> Part {
    let a = place(axis, [lo, cu, cv]);
    cyl(axis, d / 2.0, hi - lo, ACT_SEG).translate(a[0], a[1], a[2])
}

fn eval(e: &Expr) -> Part {
    let Expr::List(v) = e else {
        panic!("loon: expected a form, got {e:?}")
    };
    let Expr::Sym(head) = &v[0] else {
        panic!("loon: head {:?}", v[0])
    };
    let nums = || -> Vec<f64> {
        v[1..]
            .iter()
            .map(|a| match a {
                Expr::Num(x) => *x,
                _ => panic!("loon: {head} wants numbers"),
            })
            .collect()
    };
    let axis = |s: &str| match s.as_bytes().last() {
        Some(b'x') => 0,
        Some(b'y') => 1,
        _ => 2,
    };
    match head.as_str() {
        "union" => eval(&v[1]).union(&eval(&v[2])),
        "difference" => eval(&v[2]).difference(&eval(&v[1])),
        h if h.starts_with("plate-") => plate(axis(h), nums().try_into().unwrap()),
        h if h.starts_with("bore-") => bore(axis(h), nums().try_into().unwrap()),
        h if h.starts_with("act-") => act(axis(h), nums().try_into().unwrap()),
        h => panic!("loon: {h} is outside the subset this reader takes"),
    }
}

// ---- tessellation, cached ----

fn tessellate(part: &Part) -> TriMesh {
    let m = part.to_mesh();
    let v = m.vertices();
    let positions = v
        .chunks_exact(3)
        .map(|p| Point3::new(p[0] as f64, p[1] as f64, p[2] as f64))
        .collect();
    TriMesh::new(positions, Vec::new(), &m.indices())
}

fn concat(meshes: &[TriMesh]) -> TriMesh {
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

/// The arm's trees, in [`TREES`] order (the ground is appended last).
pub struct Arm {
    pub meshes: Vec<TriMesh>,
    /// Which rose files they came from.
    pub source: String,
    pub seconds: f64,
}

/// Trees: links (conductor), then actuators and the post (dark diffuse).
pub const TREES: [&str; 9] = [
    "shoA",
    "shoB",
    "elbA",
    "hand",
    "act_sho_pitch",
    "act_sho_roll",
    "act_elb_pitch",
    "act_wrist_yaw",
    "post",
];

fn hash(s: &str) -> u64 {
    // FNV-1a: the cache key is the source text.
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
    })
}

/// Read and tessellate the arm from the rose checkout at `rose`.
pub fn load_arm(rose: &Path) -> Arm {
    let t = Instant::now();
    let links = rose.join("full/cad/links");
    let cache = std::env::temp_dir().join("irys-rose");
    fs::create_dir_all(&cache).unwrap();
    let actuators = fs::read_to_string(rose.join("full/cad/actuators.loon"))
        .expect("rose full/cad/actuators.loon");
    let acts = lets(&actuators);
    let mut meshes = Vec::new();
    let mut tessellated = 0;
    let mut cached = |key: String, build: &dyn Fn() -> TriMesh| {
        let p = cache.join(format!("{:016x}.mesh", hash(&key)));
        load(&p).unwrap_or_else(|| {
            let m = build();
            save(&p, &m);
            tessellated += 1;
            m
        })
    };
    for link in ["shoA", "shoB", "elbA", "hand"] {
        let src = fs::read_to_string(links.join(format!("{link}.loon"))).expect("rose link");
        let l = lets(&src);
        let mut names: Vec<&String> = l.keys().filter(|n| n.starts_with("L_")).collect();
        names.sort();
        let parts: Vec<TriMesh> = names
            .iter()
            .map(|n| {
                let e = &l[*n];
                cached(format!("{n} {e:?} {CORNER_SEG} {BORE_SEG}"), &|| {
                    tessellate(&eval(e))
                })
            })
            .collect();
        meshes.push(concat(&parts));
    }
    for name in [
        "act_L_sho_pitch",
        "act_L_sho_roll",
        "act_L_elb_pitch",
        "act_L_wrist_yaw",
    ] {
        let e = &acts[name];
        meshes.push(cached(format!("{name} {e:?} {ACT_SEG}"), &|| {
            tessellate(&eval(e))
        }));
    }
    // The post the shoulder actuator mounts to: a 120 × 70 mm column from
    // the ground to above the shoulder, inboard of the pitch actuator.
    meshes.push(cached(format!("post {LIFT}"), &|| {
        tessellate(&cube(2, [-LIFT, -60.0, 106.0], [LIFT + 530.0, 120.0, 70.0]))
    }));
    eprintln!(
        "rose arm: {} trees from {} ({tessellated} tessellated, the rest cached) in {:.1?}",
        meshes.len(),
        links.display(),
        t.elapsed()
    );
    Arm {
        meshes,
        source: links.display().to_string(),
        seconds: t.elapsed().as_secs_f64(),
    }
}

// ---- the rig and the scene ----

/// The arm's base: CAD → world.
pub fn base() -> Transform {
    Transform::translation(0.0, -ARM_Y, LIFT)
}

fn p3(a: [f64; 3]) -> Point3 {
    Point3::new(a[0], a[1], a[2])
}

/// Object → world for (shoA, shoB, elbA) at joint angles
/// `(shoulder pitch, shoulder roll, elbow)`, radians.
pub fn chain(phi: &[f64]) -> [Transform; 3] {
    let y = Vec3::new(0.0, 1.0, 0.0);
    let x = Vec3::new(1.0, 0.0, 0.0);
    let a = base().then(&about(p3(SHO_PITCH), y, phi[0]));
    let b = a.then(&about(p3(SHO_ROLL), x, phi[1]));
    let c = b.then(&about(p3(ELBOW), y, phi[2]));
    [a, b, c]
}

/// Placements by tree: 0 shoA, 1 shoB, 2 elbA, 3 hand, 4–7 actuators,
/// 8 post, 9 ground. Placement `k` is tree `k`.
pub const MOVING: [usize; 7] = [0, 1, 2, 3, 5, 6, 7];

/// What each moving placement rides on: 0 shoA, 1 shoB, 2 elbA.
const RIDES: [usize; 7] = [0, 1, 2, 2, 0, 1, 2];

pub fn rig() -> Rig<'static> {
    Rig {
        moving: MOVING.to_vec(),
        place: Box::new(|phi: &[f64]| {
            let c = chain(phi);
            RIDES.iter().map(|&r| c[r].clone()).collect()
        }),
    }
}

/// Dark anodised actuator bodies (`actuators.loon`'s 0.10 grey) and post.
pub const ACT_TINT: f64 = 0.19;
pub const POST_TINT: f64 = 0.32;

/// The beauty camera: eye, target, half field of view (degrees).
pub const BEAUTY_CAM: ([f64; 3], [f64; 3], f64) =
    ([1050.0, 1250.0, 820.0], [60.0, 40.0, 470.0], 14.0);
/// The fit's camera: the same view, closer in on the arm.
pub const FIT_CAM: ([f64; 3], [f64; 3], f64) =
    ([1050.0, 1250.0, 820.0], [170.0, -20.0, 430.0], 11.0);

/// The arm on the ground, posed at `phi`, `spp` paths per pixel, from the
/// beauty camera.
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
    let mut trees: Vec<Opaque> = arm
        .meshes
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let s = if i < 4 {
                Surface::Conductor
            } else {
                Surface::Diffuse
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
    let c = chain(phi);
    let mut p = Vec::new();
    for (k, _) in TREES.iter().enumerate() {
        let to_world = match MOVING.iter().position(|&m| m == k) {
            Some(i) => c[RIDES[i]].clone(),
            None => base(),
        };
        let pl = Placement::new(k, to_world);
        p.push(match k {
            0..=3 => pl,
            8 => pl.tinted(POST_TINT),
            _ => pl.tinted(ACT_TINT),
        });
    }
    p.push(Placement::new(9, Transform::identity()));
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
    let mut v = vec![
        ("ours: cuboid", irys_diff::prb::cuboid(30.0, 20.0, 10.0)),
        ("vcad cube", tessellate(&Part::cube("c", 60.0, 40.0, 20.0))),
        (
            "vcad cylinder",
            tessellate(&Part::cylinder("c", 20.0, 40.0, 32)),
        ),
        (
            "plate",
            tessellate(&plate(2, [0.0, 3.0, 0.0, 60.0, 0.0, 60.0, 20.0])),
        ),
    ];
    v.push((
        "plate+bore",
        tessellate(
            &plate(2, [0.0, 3.0, 0.0, 60.0, 0.0, 60.0, 20.0])
                .difference(&bore(2, [20.0, 30.0, 30.0, 0.0, 3.0])),
        ),
    ));
    v
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
