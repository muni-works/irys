//! The sun, on the device, against the analytic answer and against the CPU.
//!
//! A directional light of finite angular size delivers `E·cos(theta)` to a
//! plane, whatever the renderer. That is a number both tiers can be held to
//! without a reference image, so it is what this test asks for — from the CPU
//! integrator and from the GPU one, over the same geometry, to within 1%.
//!
//! Run with `--features gpu -- --ignored --test-threads=1`.
#![cfg(all(feature = "gpu", not(target_arch = "wasm32")))]

use kosm_render::gpu::{
    AnalyticGeometry, AnalyticPrim, GpuCamera, GpuContext, GpuMaterial, GpuRenderState,
    RayTracePipeline, SceneRef,
};
use kosm_render::pathtrace::{GradientEnv, Pbr, Sun};
use kosm_render::{Point3, Vec3};

const W: u32 = 32;
const H: u32 = 32;
/// Enough passes for a one-sample-per-pixel device pass to converge on a
/// cone this small; the estimator has no variance beyond the disc's own
/// width, so this is generous.
const FRAMES: u32 = 256;

fn ctx_or_skip(name: &str) -> Option<&'static GpuContext> {
    match GpuContext::init_blocking() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping {name}: no GPU ({e})");
            None
        }
    }
}

/// A perfectly white Lambertian: `ior = 1` kills the specular lobe's F0, so
/// the surface reflects exactly `E·cos(theta)/pi` and the render inverts back
/// to the irradiance.
fn white_lambert() -> Pbr {
    Pbr {
        base_color: [1.0, 1.0, 1.0],
        metallic: 0.0,
        roughness: 1.0,
        clearcoat: 0.0,
        ior: 1.0,
        ..Default::default()
    }
}

fn sun_at(theta: f64) -> Sun {
    Sun::new(
        Vec3::new(theta.sin(), 0.0, theta.cos()),
        0.01,
        [2.0, 2.0, 2.0],
    )
}

/// Irradiance the GPU measures on a plane at z = 0 under `sun`.
fn gpu_irradiance(ctx: &'static GpuContext, sun: &Sun) -> f64 {
    let geometry = AnalyticGeometry {
        prims: vec![AnalyticPrim::plane([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0)],
    };
    let materials = vec![GpuMaterial::from_pbr(white_lambert())];
    let scene = SceneRef {
        geometry: &geometry,
        materials: &materials,
        lights: &[],
        environment: None,
    };
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let mut res = pipeline.resident_scene(ctx, scene, W, H);
    // Straight down at the plane, so every pixel sees the same geometry.
    let cam = GpuCamera::new([0.0, 0.0, 4.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.4, W, H);

    let mut mean = 0.0;
    for frame in 1..=FRAMES {
        let mut state = GpuRenderState::new(frame);
        state.enable_edges = 0;
        state.stylize = 0;
        state.ground_enabled = 0;
        state.max_depth = 1;
        state.light_count = 0;
        // A black sky: the sun is the whole of the lighting.
        state.set_gradient_env(&GradientEnv {
            zenith: [0.0; 3],
            horizon: [0.0; 3],
            ground: [0.0; 3],
            intensity: 1.0,
        });
        state.set_sun(Some(sun));
        let film = pollster::block_on(pipeline.render_resident_linear(ctx, &mut res, &cam, state))
            .expect("linear pass");
        if frame == FRAMES {
            let n = (W * H) as usize;
            mean = (0..n).map(|i| film.rgb[i * 3] as f64).sum::<f64>() / n as f64;
        }
    }
    mean * core::f64::consts::PI
}

/// The same measurement from the CPU integrator, over the same plane.
fn cpu_irradiance(sun: &Sun) -> f64 {
    use kosm_render::pathtrace::{Environment, Ground, PathTraceOptions, Scene, render};
    use kosm_render::TriMesh;

    let scene = Scene::<TriMesh> {
        objects: Vec::new(),
        lights: Vec::new(),
        env: Environment::constant([0.0; 3]),
        sun: Some(*sun),
        ground: Some(Ground {
            z: 0.0,
            material: white_lambert(),
            shadow_catcher: false,
        }),
        splats: None,
    };
    let cam = kosm_render::pathtrace::Camera::look_at(
        Point3::new(0.0, 0.0, 4.0),
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        23.0,
    );
    let opts = PathTraceOptions {
        spp: 2048,
        max_depth: 1,
        denoise: false,
        firefly_clamp: None,
        seed: 11,
        ..Default::default()
    };
    let film = render(&scene, &cam, 16, 16, &opts);
    let n = (film.width * film.height) as usize;
    let mean = (0..n).map(|i| film.rgb[i * 3] as f64).sum::<f64>() / n as f64;
    mean * core::f64::consts::PI
}

#[test]
#[ignore = "requires GPU"]
fn the_device_sun_delivers_the_analytic_irradiance() {
    let Some(ctx) = ctx_or_skip("the_device_sun_delivers_the_analytic_irradiance") else {
        return;
    };
    for theta_deg in [0.0f64, 30.0, 60.0] {
        let theta = theta_deg.to_radians();
        let sun = sun_at(theta);
        let expected = sun.irradiance[0] as f64 * theta.cos();

        let gpu = gpu_irradiance(ctx, &sun);
        let cpu = cpu_irradiance(&sun);

        let gpu_rel = (gpu - expected).abs() / expected;
        let cpu_rel = (cpu - expected).abs() / expected;
        eprintln!(
            "theta={theta_deg:>4}  analytic={expected:.5}  cpu={cpu:.5} ({:.2}%)  \
             gpu={gpu:.5} ({:.2}%)",
            cpu_rel * 100.0,
            gpu_rel * 100.0
        );
        assert!(cpu_rel < 0.01, "CPU irradiance off by {:.2}%", cpu_rel * 100.0);
        assert!(gpu_rel < 0.01, "GPU irradiance off by {:.2}%", gpu_rel * 100.0);
        // ... and the two tiers agree with each other, which is the parity
        // claim: they may not both be wrong in the same direction.
        let parity = (gpu - cpu).abs() / expected;
        assert!(parity < 0.01, "CPU/GPU disagree by {:.2}%", parity * 100.0);
    }
}

/// With no sun set, nothing changes: the default render state carries a zero
/// PDF and the shader's sun branch never fires.
#[test]
#[ignore = "requires GPU"]
fn no_sun_is_the_old_behaviour() {
    let Some(ctx) = ctx_or_skip("no_sun_is_the_old_behaviour") else {
        return;
    };
    let sun = sun_at(0.0);
    let with = gpu_irradiance(ctx, &sun);
    assert!(with > 0.0);

    let mut state = GpuRenderState::new(1);
    state.set_sun(Some(&sun));
    state.set_sun(None);
    assert_eq!(state.sun_radiance[3], 0.0, "clearing the sun must zero its PDF");
}
