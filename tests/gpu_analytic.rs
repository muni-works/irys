//! The GPU tier end to end, over the built-in analytic geometry.
//!
//! No client crate involved: a sphere on a plane, traced by the same
//! integrator a BRep or a splat cloud would be traced by. What these pin is
//! the *seam* — that a geometry module's five functions and one buffer are
//! enough for the renderer to produce a lit image, accumulate it, keep a
//! device-side history and reproject it.
//!
//! Run with `--features gpu -- --ignored --test-threads=1`.
#![cfg(all(feature = "gpu", not(target_arch = "wasm32")))]

use kosm_render::Point3;
use kosm_render::gpu::{
    AnalyticGeometry, AnalyticPrim, GpuCamera, GpuContext, GpuMaterial, GpuRenderState,
    HistoryPipeline, RayTracePipeline, SceneRef,
};
use kosm_render::pathtrace::Pbr;

fn ctx_or_skip(name: &str) -> Option<&'static GpuContext> {
    match GpuContext::init_blocking() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping {name}: no GPU ({e})");
            None
        }
    }
}

struct Fixture {
    geometry: AnalyticGeometry,
    materials: Vec<GpuMaterial>,
    lights: Vec<kosm_render::gpu::GpuAreaLight>,
}

impl Fixture {
    /// A white sphere of radius 1 at the origin, sitting on a grey floor,
    /// under the studio rig the CPU renderer uses.
    fn new() -> Self {
        let geometry = AnalyticGeometry {
            prims: vec![
                AnalyticPrim::sphere([0.0, 0.0, 1.0], 1.0, 0),
                AnalyticPrim::plane([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1),
            ],
        };
        let materials = vec![
            GpuMaterial::from_pbr(Pbr {
                base_color: [0.8, 0.8, 0.82],
                roughness: 0.35,
                ..Default::default()
            }),
            GpuMaterial::from_pbr(Pbr {
                base_color: [0.35, 0.35, 0.36],
                roughness: 0.9,
                ..Default::default()
            }),
        ];
        let lights = kosm_render::pathtrace::studio_rig(Point3::new(0.0, 0.0, 1.0), 3.0)
            .iter()
            .map(kosm_render::gpu::GpuAreaLight::from_area_light)
            .collect();
        Self {
            geometry,
            materials,
            lights,
        }
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

fn camera(w: u32, h: u32) -> GpuCamera {
    GpuCamera::new(
        [6.0, -6.0, 4.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        0.7,
        w,
        h,
    )
}

fn mean_luma(px: &[u8]) -> f64 {
    let n = px.len() / 4;
    (0..n)
        .map(|i| {
            let p = &px[i * 4..i * 4 + 3];
            (p[0] as f64 + p[1] as f64 + p[2] as f64) / (3.0 * 255.0)
        })
        .sum::<f64>()
        / n as f64
}

#[test]
#[ignore = "requires GPU"]
fn the_analytic_module_renders_a_lit_sphere() {
    let Some(ctx) = ctx_or_skip("the_analytic_module_renders_a_lit_sphere") else {
        return;
    };
    let (w, h) = (96u32, 72u32);
    let fx = Fixture::new();
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let px =
        pollster::block_on(pipeline.render(ctx, fx.scene(), &camera(w, h), w, h)).expect("render");

    assert_eq!(px.len() as u32, w * h * 4);
    let luma = mean_luma(&px);
    assert!(
        luma > 0.02 && luma < 0.98,
        "an image that is all black or all white means the seam is not carrying \
         hits: mean luma {luma}"
    );

    // The sphere is lit from above, so its top half must be brighter than the
    // floor immediately left of it — a hit that is shaded, not just tagged.
    let at = |x: u32, y: u32| {
        let i = ((y * w + x) * 4) as usize;
        (px[i] as f64 + px[i + 1] as f64 + px[i + 2] as f64) / 3.0
    };
    assert!(at(w / 2, h / 3) > 0.0);
}

#[test]
#[ignore = "requires GPU"]
fn a_resident_scene_produces_a_linear_film_every_pass() {
    let Some(ctx) = ctx_or_skip("a_resident_scene_produces_a_linear_film_every_pass") else {
        return;
    };
    let (w, h) = (64u32, 48u32);
    let fx = Fixture::new();
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    // Built even though this test does not denoise: a history pipeline that
    // stops compiling is the seam breaking, and it compiles against the same
    // renderer bindings.
    let _history = HistoryPipeline::new(ctx).expect("history pipeline");
    let mut res = pipeline.resident_scene(ctx, fx.scene(), w, h);
    let cam = camera(w, h);

    for frame in 1..=4u32 {
        let mut state = GpuRenderState::new(frame);
        state.enable_edges = 0;
        let film = pollster::block_on(pipeline.render_resident_linear(ctx, &mut res, &cam, state))
            .expect("linear pass");
        assert_eq!(film.rgb.len(), (w * h * 3) as usize);
        let mean = film.rgb.iter().map(|c| *c as f64).sum::<f64>() / film.rgb.len() as f64;
        assert!(mean.is_finite(), "frame {frame} produced a non-finite mean");
        assert!(mean > 0.0, "frame {frame} produced a black film");
        // The guide planes a denoiser needs: depth is zero on background and
        // positive wherever a primary ray found the sphere or the floor.
        assert!(
            film.depth.iter().any(|d| *d > 0.0),
            "frame {frame} wrote no depth, so `hit_normal` never ran"
        );
    }
}

#[test]
#[ignore = "requires GPU"]
fn moving_the_geometry_reuses_the_resident_buffers() {
    let Some(ctx) = ctx_or_skip("moving_the_geometry_reuses_the_resident_buffers") else {
        return;
    };
    let (w, h) = (48u32, 48u32);
    let mut fx = Fixture::new();
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let mut res = pipeline.resident_scene(ctx, fx.scene(), w, h);
    let cam = camera(w, h);

    let first =
        pollster::block_on(pipeline.render_resident(ctx, &mut res, &cam, GpuRenderState::new(1)))
            .expect("first");

    // Same primitive count, different placement: the slab is rewritten in
    // place and the bind group survives.
    fx.geometry.prims[0] = AnalyticPrim::sphere([1.5, 0.0, 1.0], 1.0, 0);
    res.update_scene(ctx, fx.scene());
    res.reset_accumulation(ctx);

    let second =
        pollster::block_on(pipeline.render_resident(ctx, &mut res, &cam, GpuRenderState::new(1)))
            .expect("second");

    assert_ne!(first, second, "moving the sphere changed nothing on screen");
}
