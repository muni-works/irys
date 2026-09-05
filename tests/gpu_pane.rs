//! Next-event estimation through a thin pane, on the device.
//!
//! A shadow ray that meets a thin-walled transmissive sheet is not blocked:
//! it carries on, dimmed by `1 − F(cos θ)`, the same factor the thin-walled
//! BSDF branch applies to a refracted path. That number is analytic, so both
//! tiers can be held to it without a reference image.
//!
//! Run with `--features gpu -- --ignored --test-threads=1`.
#![cfg(all(feature = "gpu", not(target_arch = "wasm32")))]

use kosm_render::gpu::{
    AnalyticGeometry, AnalyticPrim, GpuCamera, GpuContext, GpuMaterial, GpuRenderState,
    RayTracePipeline, SceneRef,
};
use kosm_render::pathtrace::{GradientEnv, Pbr, Sun};
use kosm_render::Vec3;

const W: u32 = 32;
const H: u32 = 32;
const FRAMES: u32 = 64;

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
        ior: 1.0,
        ..Default::default()
    }
}

fn window_glass() -> Pbr {
    Pbr {
        transmission: 1.0,
        thin_walled: true,
        roughness: 0.0,
        ior: 1.5,
        ..Pbr::glass(1.5, 0.0)
    }
}

/// Irradiance the device measures on the floor, with or without a pane of
/// glass between the floor and the sun.
fn gpu_irradiance(ctx: &'static GpuContext, glazed: bool) -> f64 {
    let mut prims = vec![AnalyticPrim::plane([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0)];
    if glazed {
        prims.push(AnalyticPrim::plane([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 1));
    }
    let geometry = AnalyticGeometry { prims };
    let materials = vec![
        GpuMaterial::from_pbr(white_lambert()),
        GpuMaterial::from_pbr(window_glass()),
    ];
    let scene = SceneRef {
        geometry: &geometry,
        materials: &materials,
        lights: &[],
        environment: None,
    };
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let mut res = pipeline.resident_scene(ctx, scene, W, H);
    // The eye sits *under* the pane, so the view ray does not pick up a
    // second Fresnel factor and the measurement is the shadow ray's alone.
    let cam = GpuCamera::new([0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.4, W, H);
    let sun = Sun::new(Vec3::new(0.0, 0.0, 1.0), 0.01, [2.0, 2.0, 2.0]);

    let mut mean = 0.0;
    for frame in 1..=FRAMES {
        let mut state = GpuRenderState::new(frame);
        state.enable_edges = 0;
        state.stylize = 0;
        state.ground_enabled = 0;
        // Direct lighting only: the ratio under test is the shadow ray's.
        state.max_depth = 1;
        state.light_count = 0;
        state.set_gradient_env(&GradientEnv {
            zenith: [0.0; 3],
            horizon: [0.0; 3],
            ground: [0.0; 3],
            intensity: 1.0,
        });
        state.set_sun(Some(&sun));
        let film = pollster::block_on(pipeline.render_resident_linear(ctx, &mut res, &cam, state))
            .expect("linear pass");
        if frame == FRAMES {
            let n = (W * H) as usize;
            mean = (0..n).map(|i| film.rgb[i * 3] as f64).sum::<f64>() / n as f64;
        }
    }
    mean * core::f64::consts::PI
}

/// The device must light a floor through a pane, at `(1 − F)` of the open
/// irradiance. Before this the pane was an opaque wall and the floor was
/// black — which is the whole reason the court's clerestory had to be cut
/// open rather than glazed.
#[test]
#[ignore = "requires a GPU"]
fn the_device_lights_a_floor_through_a_pane() {
    let Some(ctx) = ctx_or_skip("the_device_lights_a_floor_through_a_pane") else {
        return;
    };
    let bare = gpu_irradiance(ctx, false);
    let glazed = gpu_irradiance(ctx, true);
    assert!(bare > 0.0, "the open floor must be lit at all: {bare}");
    // Normal incidence at n = 1.5.
    let expected = 1.0 - 0.04;
    let ratio = glazed / bare;
    assert!(
        (ratio - expected).abs() < 0.02 * expected,
        "device pane transmittance {ratio} is not within 2% of {expected}"
    );
}
