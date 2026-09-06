//! The photon map, gathered on the device, against the CPU gather.
//!
//! A slab of glass hangs over a white floor under a sun. The light that
//! reaches the floor beneath the slab got there by refraction, which
//! next-event estimation cannot find on either tier — so the picture of that
//! floor is, on both, the photon map and nothing else. The map is built once
//! on the CPU and handed to both integrators; the test is that they read it
//! to the same picture.
//!
//! Run with `--features gpu -- --ignored --test-threads=1`.
#![cfg(all(feature = "gpu", not(target_arch = "wasm32")))]

use std::sync::Arc;

use kosm_render::analytic::{Frame, Prim};
use kosm_render::caustics::{self, CausticMap, CausticOptions, CausticPack};
use kosm_render::gpu::{
    AnalyticGeometry, AnalyticPrim, GpuCamera, GpuContext, GpuMaterial, GpuRenderState,
    RayTracePipeline, SceneRef,
};
use kosm_render::pathtrace::{
    Camera, Environment, GradientEnv, Ground, Object, PathTraceOptions, Pbr, Scene, Sun,
    render_with_caustics,
};
use kosm_render::{Analytic, Bvh, Point3, Vec3};

const W: u32 = 48;
const H: u32 = 48;
/// One device sample per pass; the mean over these is what the CPU's `spp`
/// is compared against.
const FRAMES: u32 = 64;

const SLAB_HALF: [f64; 3] = [1.0, 1.0, 0.1];
const FLOOR_Z: f64 = -3.0;
const RADIUS: f64 = 0.08;
const PHOTONS: usize = 200_000;

fn ctx_or_skip(name: &str) -> Option<&'static GpuContext> {
    match GpuContext::init_blocking() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping {name}: no GPU ({e})");
            None
        }
    }
}

fn white_lambert() -> Pbr {
    Pbr {
        base_color: [1.0, 1.0, 1.0],
        metallic: 0.0,
        roughness: 1.0,
        clearcoat: 0.0,
        specular: 0.0,
        ior: 1.0,
        ..Default::default()
    }
}

fn glass() -> Pbr {
    Pbr::glass(1.5, 0.02)
}

fn sun() -> Sun {
    Sun::new(Vec3::new(0.0, 0.0, 1.0), 0.005, [3.0, 3.0, 3.0])
}

/// The CPU scene: the slab as a box, the floor as the implicit ground.
fn cpu_scene() -> Scene<Analytic> {
    let slab = Analytic::from_prims(vec![Prim::Box {
        center: Point3::new(0.0, 0.0, 0.0),
        half: Vec3::new(SLAB_HALF[0], SLAB_HALF[1], SLAB_HALF[2]),
        rot: Frame::identity(),
    }]);
    Scene {
        objects: vec![Object::new(Arc::new(Bvh::build(slab)), glass())],
        lights: Vec::new(),
        env: Environment::constant([0.0; 3]),
        sun: Some(sun()),
        ground: Some(Ground {
            z: FLOOR_Z,
            material: white_lambert(),
            shadow_catcher: false,
        }),
        splats: None,
    }
}

/// Eye, target, and the camera's vertical field of view in degrees: from the
/// side and below the slab, looking at the floor under it, so no primary ray
/// crosses the glass.
const EYE: [f64; 3] = [0.0, -6.0, -1.0];
const AT: [f64; 3] = [0.0, 0.0, FLOOR_Z];
const FOV_DEG: f64 = 30.0;

/// The GPU's picture: the mean of `FRAMES` one-sample passes, linear.
fn gpu_render(ctx: &'static GpuContext, map: Option<&CausticMap>) -> Vec<f32> {
    let geometry = AnalyticGeometry {
        prims: vec![
            AnalyticPrim::aabb(
                [0.0, 0.0, 0.0],
                [SLAB_HALF[0] as f32, SLAB_HALF[1] as f32, SLAB_HALF[2] as f32],
                0,
            ),
            AnalyticPrim::plane([0.0, 0.0, FLOOR_Z as f32], [0.0, 0.0, 1.0], 1),
        ],
    };
    let materials = vec![
        GpuMaterial::from_pbr(glass()),
        GpuMaterial::from_pbr(white_lambert()),
    ];
    let scene = SceneRef {
        geometry: &geometry,
        materials: &materials,
        lights: &[],
        environment: None,
    };
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let mut res = pipeline.resident_scene(ctx, scene, W, H);
    res.set_caustics(ctx, map);
    let cam = GpuCamera::new(
        [EYE[0] as f32, EYE[1] as f32, EYE[2] as f32],
        [AT[0] as f32, AT[1] as f32, AT[2] as f32],
        [0.0, 0.0, 1.0],
        (FOV_DEG as f32).to_radians(),
        W,
        H,
    );
    let n = (W * H) as usize;
    let mut mean = vec![0.0f32; n * 3];
    for frame in 1..=FRAMES {
        let mut state = GpuRenderState::new(frame);
        state.enable_edges = 0;
        state.stylize = 0;
        state.ground_enabled = 0;
        state.max_depth = 1;
        state.light_count = 0;
        state.set_gradient_env(&GradientEnv {
            zenith: [0.0; 3],
            horizon: [0.0; 3],
            ground: [0.0; 3],
            intensity: 1.0,
        });
        state.set_sun(Some(&sun()));
        let film = pollster::block_on(pipeline.render_resident_linear(ctx, &mut res, &cam, state))
            .expect("linear pass");
        for (m, s) in mean.iter_mut().zip(&film.rgb) {
            *m += s / FRAMES as f32;
        }
    }
    mean
}

/// The CPU's picture over the same scene, with the same map.
fn cpu_render(map: Option<&CausticMap>) -> Vec<f32> {
    let cam = Camera::look_at(
        Point3::new(EYE[0], EYE[1], EYE[2]),
        Point3::new(AT[0], AT[1], AT[2]),
        Vec3::new(0.0, 0.0, 1.0),
        FOV_DEG,
    );
    let opts = PathTraceOptions {
        spp: FRAMES,
        max_depth: 1,
        denoise: false,
        firefly_clamp: None,
        seed: 7,
        ..Default::default()
    };
    render_with_caustics(&cpu_scene(), &cam, W, H, &opts, map).rgb
}

fn build_map() -> CausticMap {
    let map = caustics::trace(
        &cpu_scene(),
        &CausticOptions {
            photons: PHOTONS,
            radius: Some(RADIUS),
            ..Default::default()
        },
    );
    assert!(!map.is_empty(), "no photons landed under the slab");
    map
}

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32
}

fn mean_of(a: &[f32]) -> f32 {
    a.iter().sum::<f32>() / a.len() as f32
}

/// The packed layout gathers what the hash map gathers, before any device is
/// involved: the same photons, the same disc, the same number.
#[test]
fn the_packed_map_gathers_what_the_hash_map_gathers() {
    let map = build_map();
    let pack = CausticPack::new(&map);
    assert_eq!(pack.photon_count as usize, map.len());
    let n = Vec3::new(0.0, 0.0, 1.0);
    let mut worst = 0.0f32;
    for i in 0..25 {
        for j in 0..25 {
            let x = -1.2 + 0.1 * i as f64;
            let y = -1.2 + 0.1 * j as f64;
            let cpu = map.irradiance(Point3::new(x, y, FLOOR_Z), n);
            let dev = pack.irradiance([x as f32, y as f32, FLOOR_Z as f32], [0.0, 0.0, 1.0]);
            let rel = (cpu[0] - dev[0]).abs() / cpu[0].max(1e-3);
            worst = worst.max(rel);
        }
    }
    assert!(worst < 1e-3, "packed gather differs from the hash map by {worst}");
}

/// The claim: the two tiers, handed one map, paint the same floor.
#[test]
#[ignore = "requires GPU"]
fn the_device_gathers_the_map_the_cpu_gathers() {
    let Some(ctx) = ctx_or_skip("the_device_gathers_the_map_the_cpu_gathers") else {
        return;
    };
    let map = build_map();
    let cpu = cpu_render(Some(&map));
    let gpu = gpu_render(ctx, Some(&map));
    let cpu_plain = cpu_render(None);
    let gpu_plain = gpu_render(ctx, None);

    // The map did something on both tiers, and about the same something.
    let cpu_added = mean_of(&cpu) - mean_of(&cpu_plain);
    let gpu_added = mean_of(&gpu) - mean_of(&gpu_plain);
    let diff = mean_abs_diff(&cpu, &gpu);
    let scale = mean_of(&cpu).max(1e-6);
    eprintln!(
        "cpu mean {:.4} (+{cpu_added:.4} from the map)  gpu mean {:.4} (+{gpu_added:.4})  \
         mean |cpu - gpu| {diff:.5} = {:.2}% of the cpu mean",
        mean_of(&cpu),
        mean_of(&gpu),
        100.0 * diff / scale,
    );
    assert!(cpu_added > 0.05 * scale, "the CPU picture shows no caustic");
    assert!(gpu_added > 0.05 * scale, "the GPU picture shows no caustic");
    // The kernel is the same and the map is the same; what is left is the
    // sun's own sampling noise on the lit floor and f32 against f64 in the
    // gather. Three percent of the mean is well above both and well below a
    // missing or doubled caustic.
    assert!(
        diff < 0.03 * scale,
        "CPU and GPU caustics differ by {:.2}% of the mean",
        100.0 * diff / scale
    );
}

/// With no map — or an empty one — the device's picture is the one it was:
/// the guard is on `enabled`, and nothing past it runs.
#[test]
#[ignore = "requires GPU"]
fn an_empty_map_changes_nothing() {
    let Some(ctx) = ctx_or_skip("an_empty_map_changes_nothing") else {
        return;
    };
    let none = gpu_render(ctx, None);
    let empty = gpu_render(ctx, Some(&CausticMap::empty()));
    assert_eq!(none, empty, "an empty map changed a pixel");
    // ... and a real map does change it, so the equality above is not the
    // equality of two blank pictures.
    let map = build_map();
    let lit = gpu_render(ctx, Some(&map));
    assert!(mean_of(&lit) > mean_of(&none) * 1.05);
}
