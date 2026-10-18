//! The benchmark scenes. Meshes only (Cycles gets the same triangles), every
//! one built from irys-diff's stock meshes placed by transforms.
//!
//! - `pair`: phase 2c's outdoor scene — two GGX conductor balls (one
//!   squashed and turned) on a checkered floor under a sky and a soft sun.
//!   Mostly direct light; the easy case.
//! - `room`: the classroom stand-in — a closed room lit only through one
//!   window by a sharp sun and the sky, a conductor ball and a diffuse box
//!   inside. Nearly all of the image is indirect light, and most light
//!   samples from inside are occluded: the case path guiding and reservoir
//!   reuse exist for.
//! - `cornell`: a grey Cornell box (irys's diffuse albedo is a scalar, so
//!   no red and green walls) lit by one emissive quad under the ceiling, a
//!   tall box and a conductor ball, black outside. The mesh-emitter case.

use irys_core::{Point3, Transform, Vec3};
use irys_diff::ggx::PARAMS;
use irys_diff::prb::{Env, Opaque, Placement, Scene, Surface, cuboid, uv_sphere};
use irys_prims::analytic::Analytic;

/// The scenes, by name.
pub const NAMES: [&str; 3] = ["pair", "room", "cornell"];

/// The conductor every scene uses: phase 2c's gold, `(α, F0 at 450, 550,
/// 650 nm)`.
pub const GOLD: [f64; PARAMS] = [0.2, 0.34, 0.78, 1.0];

/// A benchmark scene and the θ its conductor renders at.
pub struct Bench {
    /// Its name in [`NAMES`].
    pub name: &'static str,
    /// The scene, at the requested size and spp.
    pub scene: Scene,
    /// The conductor.
    pub theta: [f64; PARAMS],
}

/// Phase 5a's real scenes: Blender demo files through
/// `scripts/blend_export.py`, in `$IRYS_SCENES/<name>` (default
/// `../irys-assets/blender-demos/export`). `<name>-grey` is the same scene
/// with every non-emitter a grey Lambertian (albedo [`GREY`]).
pub const IMPORTED: [&str; 2] = ["classroom", "pavilion"];

/// The grey benchmark's albedo.
pub const GREY: f64 = 0.8;

/// Where the exports live.
pub fn scenes_dir() -> std::path::PathBuf {
    std::env::var("IRYS_SCENES")
        .unwrap_or_else(|_| "../irys-assets/blender-demos/export".into())
        .into()
}

/// `(export name, grey)` when `name` is an imported scene.
pub fn imported(name: &str) -> Option<(&str, bool)> {
    let (base, grey) = match name.strip_suffix("-grey") {
        Some(b) => (b, true),
        None => (name, false),
    };
    IMPORTED.contains(&base).then_some((base, grey))
}

/// Scene `name` at `w × h`, `spp` paths per pixel per call.
pub fn build(name: &str, w: usize, h: usize, spp: usize) -> Bench {
    if let Some((base, grey)) = imported(name) {
        let opts = irys_diff::blend::Options {
            width: Some(w),
            spp,
            depth: 8,
            grey: grey.then_some(GREY),
        };
        let im = irys_diff::blend::load(&scenes_dir().join(base), &opts)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(im.scene.size, (w, h), "{name}: --w/--h must keep the export's aspect");
        return Bench {
            name: Box::leak(name.to_string().into_boxed_str()),
            scene: im.scene,
            theta: im.theta,
        };
    }
    match name {
        "pair" => pair(w, h, spp),
        "room" => room(w, h, spp),
        "cornell" => cornell(w, h, spp),
        _ => panic!("unknown scene {name}: one of {NAMES:?}"),
    }
}

fn pair(w: usize, h: usize, spp: usize) -> Bench {
    let mut scene = Scene::pair(w, h, spp, 1.0);
    scene.depth = 6;
    scene.complete = true;
    Bench {
        name: "pair",
        scene,
        theta: GOLD,
    }
}

fn mesh(m: irys_core::TriMesh, s: Surface) -> Opaque {
    Opaque::new(m, s, Analytic::from_prims(Vec::new()), &[])
}

/// An axis-aligned box from `lo` to `hi`, as a placement of the unit cube.
fn slab(lo: [f64; 3], hi: [f64; 3], albedo: f64) -> Placement {
    let c = |k: usize| 0.5 * (lo[k] + hi[k]);
    let s = |k: usize| hi[k] - lo[k];
    Placement::new(
        0,
        Transform::translation(c(0), c(1), c(2)).then(&Transform::scale(s(0), s(1), s(2))),
    )
    .tinted(albedo)
}

fn room(w: usize, h: usize, spp: usize) -> Bench {
    // Interior x ∈ [−3, 3], y ∈ [−2.5, 2.5], z ∈ [0, 3]; walls 0.1 thick.
    // One window in the +y wall, x ∈ [−1.2, 1.2], z ∈ [0.9, 2.3].
    let (t, x0, x1, y0, y1, z1) = (0.1, -3.0, 3.0, -2.5, 2.5, 3.0);
    let (wx0, wx1, wz0, wz1) = (-1.2, 1.2, 0.9, 2.3);
    let wall = 1.0;
    let mut p = vec![
        slab([x0 - t, y0 - t, -t], [x1 + t, y1 + t, 0.0], 0.8), // floor
        slab([x0 - t, y0 - t, z1], [x1 + t, y1 + t, z1 + t], wall), // ceiling
        slab([x0 - t, y0 - t, 0.0], [x0, y1 + t, z1], wall),    // −x
        slab([x1, y0 - t, 0.0], [x1 + t, y1 + t, z1], wall),    // +x
        slab([x0, y0 - t, 0.0], [x1, y0, z1], wall),            // −y
        // +y, around the window
        slab([x0, y1, 0.0], [wx0, y1 + t, z1], wall),
        slab([wx1, y1, 0.0], [x1, y1 + t, z1], wall),
        slab([wx0, y1, 0.0], [wx1, y1 + t, wz0], wall),
        slab([wx0, y1, wz1], [wx1, y1 + t, z1], wall),
        // a low box, darker
        slab([-2.1, -1.6, 0.0], [-1.1, -0.6, 0.9], 0.6),
    ];
    // The ball: conductor, radius 0.6, resting on the floor.
    p.push(Placement::new(
        1,
        Transform::translation(0.9, -0.5, 0.6).then(&Transform::scale(0.6, 0.6, 0.6)),
    ));
    let mut scene = Scene::instanced(
        vec![
            mesh(cuboid(0.5, 0.5, 0.5), Surface::Diffuse),
            mesh(uv_sphere(32, 64), Surface::Conductor),
        ],
        p,
        None,
        (w, h),
        spp,
        (Point3::new(2.7, -2.3, 1.7), Point3::new(-0.6, 0.9, 0.9)),
        32.0,
    );
    scene.floor_albedo = 0.7;
    scene.floor_checker = None;
    scene.nee = true;
    scene.depth = 8;
    scene.complete = true;
    scene.env = Env {
        sun: Vec3::new(0.25, 1.0, 0.75).normalize(),
        kappa: 400.0,
        sun_radiance: 300.0,
        sky: 1.0,
        ..Env::default()
    };
    Bench {
        name: "room",
        scene,
        theta: GOLD,
    }
}

fn cornell(w: usize, h: usize, spp: usize) -> Bench {
    Bench {
        name: "cornell",
        scene: Scene::cornell(w, h, spp),
        theta: GOLD,
    }
}

/// Turn on the samplers a variant names (`base`, `restir`, `guide`,
/// `sobol`, `spectral`, `rr`, joined by `+`).
pub fn apply(scene: &mut Scene, features: &[&str]) {
    for f in features {
        match *f {
            "base" => {}
            "restir" => scene.restir = Some(irys_diff::prb::Restir::default()),
            "guide" => scene.guide = Some(irys_diff::prb::Guide::default()),
            "sobol" => scene.qmc = true,
            "spectral" => scene.spectral_is = true,
            "rr" => scene.roulette = Some(irys_diff::prb::Roulette::default()),
            "sun" => scene.env = scene.env.pick_by_peak(),
            // phases 1–3's fixed half share (by peak is the default since 5a)
            "half" => scene.env = scene.env.with_pick(irys_diff::prb::SUN_PICK),
            other => panic!("unknown sampler {other}"),
        }
    }
}
