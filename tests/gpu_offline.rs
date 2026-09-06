//! The offline render path: many samples against one resident scene, read
//! back once as linear HDR.
//!
//! No client crate involved — the built-in analytic geometry module is the
//! scene, so what these pin is the renderer's own half: that the offline loop
//! converges to the number the CPU integrator converges to, that its seed
//! makes a render reproducible and a different seed makes it independent, that
//! the explicit camera basis survives a mirroring, and that the backdrop modes
//! do what `PathTraceOptions::show_background` does.
//!
//! Run with `--features gpu -- --ignored --test-threads=1`.
#![cfg(all(feature = "gpu", not(target_arch = "wasm32")))]

use std::sync::Arc;

use kosm_render::analytic::Prim;
use kosm_render::gpu::{
    AnalyticGeometry, AnalyticPrim, GpuAreaLight, GpuCamera, GpuContext, GpuMaterial,
    MAX_TRAVERSAL_DEPTH, OfflineOptions, OfflineResult, RayTracePipeline, SceneRef,
    validate_tree_depth,
};
use kosm_render::math::Aabb;
use kosm_render::pathtrace::{
    Camera, Environment, Object, PathTraceOptions, Pbr, Scene, studio_rig,
};
use kosm_render::{Analytic, Bvh, FlatBvhNode, Point3, Vec3};

const W: u32 = 64;
const H: u32 = 64;

/// The subject: a unit sphere, its centre a radius above the floor.
const SPHERE_C: [f64; 3] = [0.0, 0.0, 1.0];
const SPHERE_R: f64 = 1.0;
/// The floor plane, through the origin, facing up.
const FLOOR_Z: f64 = 0.0;

const EYE: [f64; 3] = [6.0, -6.0, 4.0];
const AT: [f64; 3] = SPHERE_C;
const FOV_DEG: f64 = 40.0;

fn ctx_or_skip(name: &str) -> Option<&'static GpuContext> {
    match GpuContext::init_blocking() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping {name}: no GPU ({e})");
            None
        }
    }
}

fn sphere_pbr() -> Pbr {
    Pbr {
        base_color: [0.8, 0.8, 0.82],
        metallic: 0.0,
        roughness: 0.4,
        ..Default::default()
    }
}

fn floor_pbr() -> Pbr {
    Pbr {
        base_color: [0.35, 0.35, 0.36],
        metallic: 0.0,
        roughness: 0.9,
        ..Default::default()
    }
}

/// The rig both tiers are lit by: the studio softboxes around the subject.
fn rig() -> Vec<kosm_render::AreaLight> {
    studio_rig(
        Point3::new(SPHERE_C[0], SPHERE_C[1], SPHERE_C[2]),
        3.0 * SPHERE_R,
    )
}

// ── the device scene ─────────────────────────────────────────────────────

struct Fixture {
    geometry: AnalyticGeometry,
    materials: Vec<GpuMaterial>,
    lights: Vec<GpuAreaLight>,
}

impl Fixture {
    fn new() -> Self {
        Self {
            geometry: AnalyticGeometry {
                prims: vec![
                    AnalyticPrim::sphere(
                        [SPHERE_C[0] as f32, SPHERE_C[1] as f32, SPHERE_C[2] as f32],
                        SPHERE_R as f32,
                        0,
                    ),
                    AnalyticPrim::plane([0.0, 0.0, FLOOR_Z as f32], [0.0, 0.0, 1.0], 1),
                ],
            },
            materials: vec![
                GpuMaterial::from_pbr(sphere_pbr()),
                GpuMaterial::from_pbr(floor_pbr()),
            ],
            lights: rig().iter().map(GpuAreaLight::from_area_light).collect(),
        }
    }

    /// The subject alone, with no floor under it: what the backdrop tests
    /// want, so a corner pixel really does miss everything.
    fn sphere_only() -> Self {
        let mut f = Self::new();
        f.geometry.prims.truncate(1);
        f
    }

    fn scene(&self) -> SceneRef<'_> {
        SceneRef {
            geometry: &self.geometry,
            materials: &self.materials,
            lights: &self.lights,
            environment: None,
        }
    }
}

fn gpu_camera(w: u32, h: u32) -> GpuCamera {
    GpuCamera::new(
        [EYE[0] as f32, EYE[1] as f32, EYE[2] as f32],
        [AT[0] as f32, AT[1] as f32, AT[2] as f32],
        [0.0, 0.0, 1.0],
        (FOV_DEG as f32).to_radians(),
        w,
        h,
    )
}

/// The same camera, spelled as an explicit basis. `derived` picks the
/// right-handed reconstruction the viewport uses; `!derived` negates `right`,
/// which is a mirrored (left-handed) basis no look-at can produce.
fn explicit_camera(w: u32, h: u32, mirrored: bool) -> GpuCamera {
    let eye = Vec3::new(EYE[0], EYE[1], EYE[2]);
    let at = Vec3::new(AT[0], AT[1], AT[2]);
    let forward = (at - eye).normalize();
    let up_hint = Vec3::new(0.0, 0.0, 1.0);
    let mut right = forward.cross(up_hint).normalize();
    let up = right.cross(forward).normalize();
    if mirrored {
        right = -right;
    }
    GpuCamera::from_basis(
        [EYE[0] as f32, EYE[1] as f32, EYE[2] as f32],
        [forward.x as f32, forward.y as f32, forward.z as f32],
        [right.x as f32, right.y as f32, right.z as f32],
        [up.x as f32, up.y as f32, up.z as f32],
        (FOV_DEG as f32).to_radians(),
        (at - eye).norm() as f32,
        w,
        h,
    )
}

fn offline_opts(spp: u32) -> OfflineOptions {
    OfflineOptions {
        width: W,
        height: H,
        spp,
        // The implicit ground is off: the floor is real geometry here, and a
        // second one underneath it is exactly the bug `ground_enabled` guards.
        ground_enabled: false,
        ..Default::default()
    }
}

fn render(
    ctx: &'static GpuContext,
    fixture: &Fixture,
    camera: &GpuCamera,
    opts: &OfflineOptions,
) -> OfflineResult {
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    pipeline
        .render_offline(ctx, fixture.scene(), camera, opts)
        .expect("offline render")
}

// ── the CPU scene ────────────────────────────────────────────────────────

fn cpu_scene() -> Scene<Analytic> {
    let sphere = Analytic::from_prims(vec![Prim::Sphere {
        center: Point3::new(SPHERE_C[0], SPHERE_C[1], SPHERE_C[2]),
        radius: SPHERE_R,
    }]);
    let floor = Analytic::from_prims(vec![Prim::Plane {
        point: Point3::new(0.0, 0.0, FLOOR_Z),
        normal: kosm_render::Dir3::new_normalize(Vec3::new(0.0, 0.0, 1.0)),
    }]);
    Scene {
        objects: vec![
            Object::new(Arc::new(Bvh::build(sphere)), sphere_pbr()),
            Object::new(Arc::new(Bvh::build(floor)), floor_pbr()),
        ],
        lights: rig(),
        env: Environment::default(),
        sun: None,
        ground: None,
        splats: None,
    }
}

fn cpu_mean_luminance(spp: u32) -> f32 {
    let cam = Camera::look_at(
        Point3::new(EYE[0], EYE[1], EYE[2]),
        Point3::new(AT[0], AT[1], AT[2]),
        Vec3::new(0.0, 0.0, 1.0),
        FOV_DEG,
    );
    let opts = PathTraceOptions {
        spp,
        max_depth: OfflineOptions::default().max_depth,
        rr_start: OfflineOptions::default().rr_start,
        denoise: false,
        show_background: true,
        seed: 7,
        ..Default::default()
    };
    let film = kosm_render::pathtrace::render(&cpu_scene(), &cam, W, H, &opts);
    let mut sum = 0.0f64;
    for p in film.rgb.chunks_exact(3) {
        sum += (0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]) as f64;
    }
    (sum / (film.rgb.len() / 3) as f64) as f32
}

// ── helpers ──────────────────────────────────────────────────────────────

fn luma(p: [f32; 3]) -> f32 {
    0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]
}

/// Mean luminance of a vertical half of the image.
fn half_mean(r: &OfflineResult, left: bool) -> f32 {
    let (lo, hi) = if left { (0, r.width / 2) } else { (r.width / 2, r.width) };
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for y in 0..r.height {
        for x in lo..hi {
            sum += luma(r.pixel(x, y)) as f64;
            n += 1;
        }
    }
    (sum / n.max(1) as f64) as f32
}

/// Mean absolute difference between a pixel's luminance and its four
/// neighbours' — a crude but monotone read on how noisy an image is.
fn roughness(r: &OfflineResult) -> f32 {
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for y in 1..r.height - 1 {
        for x in 1..r.width - 1 {
            let c = luma(r.pixel(x, y));
            for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                let nx = (x as i32 + dx) as u32;
                let ny = (y as i32 + dy) as u32;
                sum += (c - luma(r.pixel(nx, ny))).abs() as f64;
                n += 1;
            }
        }
    }
    (sum / n.max(1) as f64) as f32
}

// ── the tests ────────────────────────────────────────────────────────────

/// The claim the whole path exists for: the device's estimate of this image is
/// the CPU integrator's estimate of it.
///
/// The two tracers are not bit-identical — f32 against f64, a different RNG,
/// a different sample pattern — so this is a tolerance on the *mean* over the
/// frame, which is what a converged estimator is allowed to be judged on.
/// Ten per cent relative: comfortably inside the two tiers' agreement and far
/// outside anything a mis-wired uniform would leave.
#[test]
#[ignore = "requires GPU"]
fn offline_luminance_agrees_with_the_cpu_integrator() {
    let Some(ctx) = ctx_or_skip("offline_luminance_agrees_with_the_cpu_integrator") else {
        return;
    };
    let fixture = Fixture::new();
    let gpu = render(ctx, &fixture, &gpu_camera(W, H), &offline_opts(128)).mean_luminance();
    let cpu = cpu_mean_luminance(128);

    assert!(gpu > 0.01, "the device render is black: {gpu}");
    let rel = (gpu - cpu).abs() / cpu.max(1e-6);
    assert!(
        rel < 0.10,
        "offline mean luminance {gpu} against the CPU's {cpu} ({:.1}% apart)",
        rel * 100.0
    );
}

/// A seed makes a render reproducible, and a different seed makes it a
/// different — independent — estimate of the same image.
#[test]
#[ignore = "requires GPU"]
fn the_seed_reproduces_a_render_and_a_different_seed_does_not() {
    let Some(ctx) = ctx_or_skip("the_seed_reproduces_a_render_and_a_different_seed_does_not")
    else {
        return;
    };
    let fixture = Fixture::new();
    let cam = gpu_camera(W, H);

    let mut opts = offline_opts(8);
    opts.seed = 1234;
    let a = render(ctx, &fixture, &cam, &opts);
    let b = render(ctx, &fixture, &cam, &opts);
    assert_eq!(a.rgba, b.rgba, "the same seed gave a different image");

    opts.seed = 5678;
    let c = render(ctx, &fixture, &cam, &opts);
    assert_ne!(a.rgba, c.rgba, "a different seed gave the same image");

    // Different noise, same picture: the two seeds agree on the mean.
    let (ma, mc) = (a.mean_luminance(), c.mean_luminance());
    let rel = (ma - mc).abs() / ma.max(1e-6);
    assert!(rel < 0.05, "two seeds disagree on the mean: {ma} vs {mc}");
}

/// More samples, less noise. The point of the loop.
#[test]
#[ignore = "requires GPU"]
fn sixty_four_samples_are_smoother_than_one() {
    let Some(ctx) = ctx_or_skip("sixty_four_samples_are_smoother_than_one") else {
        return;
    };
    let fixture = Fixture::new();
    let cam = gpu_camera(W, H);
    let one = render(ctx, &fixture, &cam, &offline_opts(1));
    let many = render(ctx, &fixture, &cam, &offline_opts(64));

    assert_eq!(one.spp, 1);
    assert_eq!(many.spp, 64);
    let (r1, r64) = (roughness(&one), roughness(&many));
    assert!(
        r64 < r1 * 0.7,
        "64 spp is not meaningfully smoother than 1: {r64} against {r1}"
    );
}

/// A mirrored explicit basis reaches the shader as given, and the image comes
/// back flipped left-for-right rather than silently re-derived.
///
/// Judged on the two halves of the frame rather than pixel-for-pixel: the RNG
/// is keyed on the pixel, so mirroring the *camera* does not mirror the noise,
/// and only the picture underneath it is expected to match.
#[test]
#[ignore = "requires GPU"]
fn a_mirrored_explicit_basis_flips_the_image_left_for_right() {
    let Some(ctx) = ctx_or_skip("a_mirrored_explicit_basis_flips_the_image_left_for_right") else {
        return;
    };
    let fixture = Fixture::new();
    let opts = offline_opts(32);

    let derived = render(ctx, &fixture, &gpu_camera(W, H), &opts);
    let explicit = render(ctx, &fixture, &explicit_camera(W, H, false), &opts);
    let mirrored = render(ctx, &fixture, &explicit_camera(W, H, true), &opts);

    // An explicit basis that happens to be the derived one renders the same
    // picture — the mode is not a second projection.
    let (dl, dr) = (half_mean(&derived, true), half_mean(&derived, false));
    let (el, er) = (half_mean(&explicit, true), half_mean(&explicit, false));
    assert!(
        (dl - el).abs() < 0.02 * dl.max(1e-6) && (dr - er).abs() < 0.02 * dr.max(1e-6),
        "an explicit right-handed basis rendered something else: {dl}/{dr} vs {el}/{er}"
    );

    // The scene is not left-right symmetric, or this proves nothing.
    let asym = (dl - dr).abs() / dl.max(dr).max(1e-6);
    assert!(asym > 0.05, "the fixture is too symmetric to detect a flip");

    // And the mirrored basis swaps the halves rather than reproducing them.
    let (ml, mr) = (half_mean(&mirrored, true), half_mean(&mirrored, false));
    assert!(
        (ml - dr).abs() < 0.05 * dr.max(1e-6) && (mr - dl).abs() < 0.05 * dl.max(1e-6),
        "mirrored basis did not flip the image: {ml}/{mr} against {dl}/{dr}"
    );
}

/// The backdrop modes: black leaves a miss transparent, the environment mode
/// puts the sky the integrator lights with behind the subject.
#[test]
#[ignore = "requires GPU"]
fn the_backdrop_modes_do_what_show_background_does() {
    let Some(ctx) = ctx_or_skip("the_backdrop_modes_do_what_show_background_does") else {
        return;
    };
    // No floor: the corner really is empty space.
    let fixture = Fixture::sphere_only();
    let cam = gpu_camera(W, H);

    let mut opts = offline_opts(8);
    opts.show_background = false;
    let black = render(ctx, &fixture, &cam, &opts);
    opts.show_background = true;
    let env = render(ctx, &fixture, &cam, &opts);

    // The corner pixel misses everything on both.
    let (x, y) = (1, 1);
    let b = black.pixel(x, y);
    assert!(
        b.iter().all(|c| *c == 0.0),
        "a miss under BACKGROUND_BLACK is not black: {b:?}"
    );
    let alpha = black.rgba[((y * black.width + x) * 4 + 3) as usize];
    assert_eq!(alpha, 0.0, "a miss under BACKGROUND_BLACK has coverage");

    let e = env.pixel(x, y);
    assert!(
        luma(e) > 0.01,
        "a miss under BACKGROUND_ENVIRONMENT has no radiance: {e:?}"
    );

    // The subject itself is unaffected by the choice of backdrop, and covered.
    let (cx, cy) = (black.width / 2, black.height / 2);
    let centre_alpha = black.rgba[((cy * black.width + cx) * 4 + 3) as usize];
    assert!(
        centre_alpha > 0.5,
        "the covered centre pixel reads as transparent: {centre_alpha}"
    );
    let (cb, ce) = (luma(black.pixel(cx, cy)), luma(env.pixel(cx, cy)));
    assert!(cb > 0.0 && ce > 0.0, "the subject is black on one of the two");
}

/// `to_film` hands the accumulation buffer to the CPU `Film` so the caller's
/// output transform is the CPU one — and leaves the guide planes zeroed, which
/// is why the doc comment forbids feeding it to the CPU denoiser.
#[test]
#[ignore = "requires GPU"]
fn to_film_carries_the_radiance_and_leaves_the_guides_zeroed() {
    let Some(ctx) = ctx_or_skip("to_film_carries_the_radiance_and_leaves_the_guides_zeroed") else {
        return;
    };
    let fixture = Fixture::new();
    let r = render(ctx, &fixture, &gpu_camera(W, H), &offline_opts(4));
    let film = r.to_film();

    assert_eq!((film.width, film.height), (r.width, r.height));
    assert_eq!(film.rgb.len(), (W * H * 3) as usize);
    assert_eq!(film.alpha.len(), (W * H) as usize);
    assert_eq!(&film.rgb[..3], &r.pixel(0, 0)[..]);
    assert!(
        film.normal.iter().all(|v| *v == 0.0)
            && film.depth.iter().all(|v| *v == 0.0)
            && film.albedo.iter().all(|v| *v == 0.0)
            && film.variance.iter().all(|v| *v == 0.0),
        "to_film invented guide data it never read back"
    );
}

// ── the host-side traversal guard, which needs no device ─────────────────

/// A chain `depth` levels deep: each internal node's left child is the next
/// link, its right child a leaf.
fn deep_chain(depth: usize) -> Vec<FlatBvhNode> {
    let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
    let mut nodes: Vec<FlatBvhNode> = Vec::new();
    for level in 0..depth - 1 {
        // Internal: left = the next link, right = a leaf parked at the end.
        let left = (level + 1) * 2;
        let right = left - 1;
        nodes.push((aabb, false, left as u32, right as u32));
        nodes.push((aabb, true, 0, 1));
    }
    nodes.push((aabb, true, 0, 1));
    nodes
}

#[test]
fn validate_tree_depth_refuses_a_tree_deeper_than_the_traversal_stack() {
    assert_eq!(MAX_TRAVERSAL_DEPTH, 64);

    let ok = deep_chain(MAX_TRAVERSAL_DEPTH);
    assert_eq!(kosm_render::gpu::tree_depth(&ok), MAX_TRAVERSAL_DEPTH);
    validate_tree_depth(&ok).expect("a tree exactly as deep as the stack must be accepted");

    let too_deep = deep_chain(MAX_TRAVERSAL_DEPTH + 1);
    assert_eq!(
        kosm_render::gpu::tree_depth(&too_deep),
        MAX_TRAVERSAL_DEPTH + 1
    );
    let err = validate_tree_depth(&too_deep).expect_err("a 65-deep tree must be refused");
    let msg = err.to_string();
    assert!(msg.contains("65") && msg.contains("64"), "unhelpful: {msg}");

    // An empty tree is depth 0 and passes; so does a lone root.
    validate_tree_depth(&[]).expect("an empty tree is not too deep");
    validate_tree_depth(&deep_chain(1)).expect("a lone leaf is not too deep");
}
