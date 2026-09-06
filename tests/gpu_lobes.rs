//! The split denoiser: one sample, two lobes.
//!
//! Three claims about the diffuse/specular split in `history.wgsl`:
//!
//! * nothing is lost in the sorting — over a frame of mixed materials the two
//!   lobe planes the integrator writes sum to the summed plane, pixel for
//!   pixel, to the float;
//! * the specular demodulator the integrator writes into its guide plane is
//!   the same split-sum lookup [`kosm_render::spec_albedo`] evaluates on the
//!   CPU, at the same roughness, F0 and view cosine;
//! * a mirror sliding in its own plane keeps its reflection's history under
//!   the specular reprojection — which follows the reflected point — where
//!   the surface reprojection, which follows the mirror, drags a stale
//!   reflection under every pixel and has it clamped away.
//!
//! Run with `--features gpu -- --ignored --test-threads=1 --nocapture`.
#![cfg(all(feature = "gpu", not(target_arch = "wasm32")))]

use kosm_render::Point3;
use kosm_render::gpu::wgpu;
use kosm_render::gpu::{
    AnalyticGeometry, AnalyticPrim, GpuAreaLight, GpuCamera, GpuContext, GpuDenoiseParams,
    GpuMaterial, GpuRenderState, HistoryPipeline, InstanceMotion, RayTracePipeline, SceneRef,
};
use kosm_render::pathtrace::{self, Pbr};

const W: u32 = 64;
const H: u32 = 64;

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
    lights: Vec<GpuAreaLight>,
}

impl Fixture {
    /// A sphere on a plane, with the two materials given: index 0 is the
    /// sphere's, 1 the plane's.
    fn new(sphere: Pbr, plane: Pbr) -> Self {
        Self {
            geometry: AnalyticGeometry {
                prims: vec![
                    AnalyticPrim::sphere([0.0, 0.0, 1.0], 1.0, 0),
                    AnalyticPrim::plane([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1),
                ],
            },
            materials: vec![GpuMaterial::from_pbr(sphere), GpuMaterial::from_pbr(plane)],
            lights: pathtrace::studio_rig(Point3::new(0.0, 0.0, 1.0), 3.0)
                .iter()
                .map(GpuAreaLight::from_area_light)
                .collect(),
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

fn camera() -> GpuCamera {
    GpuCamera::new([5.0, -5.0, 3.5], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 0.7, W, H)
}

fn state(frame: u32) -> GpuRenderState {
    let mut s = GpuRenderState::new(frame);
    s.enable_edges = 0;
    s.stylize = 0;
    s.ground_enabled = 0;
    s.set_camera_visible_lights(false);
    s
}

/// A storage texture the resolve pass can write into.
fn target(ctx: &GpuContext) -> wgpu::TextureView {
    ctx.device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("lobes test target"),
            size: wgpu::Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

/// The primary ray through a pixel's centre, as `view_ray` in `history.wgsl`
/// and the ray generator build it.
fn pixel_dir(cam: &GpuCamera, x: u32, y: u32) -> [f32; 3] {
    let norm = |v: [f32; 3]| {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / l, v[1] / l, v[2] / l]
    };
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let fwd = norm([
        cam.target[0] - cam.position[0],
        cam.target[1] - cam.position[1],
        cam.target[2] - cam.position[2],
    ]);
    let right = norm(cross(fwd, [cam.up[0], cam.up[1], cam.up[2]]));
    let up = cross(right, fwd);
    let tan = (cam.fov * 0.5).tan();
    let aspect = W as f32 / H as f32;
    let ndc_x = (x as f32 + 0.5) / W as f32 * 2.0 - 1.0;
    let ndc_y = 1.0 - (y as f32 + 0.5) / H as f32 * 2.0;
    norm([
        fwd[0] + right[0] * ndc_x * tan * aspect + up[0] * ndc_y * tan,
        fwd[1] + right[1] * ndc_x * tan * aspect + up[1] * ndc_y * tan,
        fwd[2] + right[2] * ndc_x * tan * aspect + up[2] * ndc_y * tan,
    ])
}

/// The integrator sorts every sample into two planes that sum to the one it
/// always wrote — over a frame with a glossy plastic, a rough metal, a
/// background and the lit gradient between them.
#[test]
#[ignore = "requires GPU"]
fn the_two_lobes_sum_to_the_sample() {
    let Some(ctx) = ctx_or_skip("the_two_lobes_sum_to_the_sample") else {
        return;
    };
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let history = HistoryPipeline::new(ctx).expect("history pipeline");
    let view = target(ctx);
    let fx = Fixture::new(
        Pbr {
            base_color: [0.8, 0.3, 0.2],
            roughness: 0.25,
            ..Default::default()
        },
        Pbr {
            base_color: [0.7, 0.7, 0.72],
            metallic: 1.0,
            roughness: 0.6,
            ..Default::default()
        },
    );
    let mut res = pipeline.resident_scene(ctx, fx.scene(), W, H);
    // Clamp off: the clamp is per lobe by design — a stale reflection is
    // shortened without touching the diffuse history under it — so with it
    // on the two lobes' counts can legitimately part company.
    let denoise = GpuDenoiseParams {
        clamp_k: 0.0,
        ..GpuDenoiseParams::default()
    };
    let n = (W * H) as usize;

    // The check: the two lobe means sum to the summed mean, pixel for pixel.
    let check = |res: &mut _, label: &str, only_surfaces: bool| -> (f32, f64, f64) {
        let summed = pollster::block_on(pipeline.read_history(ctx, res))
            .expect("read")
            .expect("history");
        let (d, s) = pollster::block_on(pipeline.read_history_lobes(ctx, res))
            .expect("read")
            .expect("history");
        let g = pollster::block_on(pipeline.read_guides(ctx, res)).expect("guides");
        let mut worst = 0.0f32;
        let (mut de, mut se) = (0.0f64, 0.0f64);
        for i in 0..n {
            if only_surfaces && g.depth[i] <= 0.0 {
                continue;
            }
            assert_eq!(d.count[i], summed.count[i], "{label}: pixel {i}'s diffuse count");
            assert_eq!(s.count[i], summed.count[i], "{label}: pixel {i}'s specular count");
            for k in 0..3 {
                let a = d.rgb[i * 3 + k] + s.rgb[i * 3 + k];
                worst = worst.max((a - summed.rgb[i * 3 + k]).abs());
            }
            de += d.rgb[i * 3] as f64;
            se += s.rgb[i * 3] as f64;
        }
        eprintln!("lobes, {label}: worst |diffuse + specular - summed| = {worst:.2e}; red energy diffuse {de:.2} specular {se:.2}");
        (worst, de, se)
    };

    let mut pass = |res: &mut _, f: u32| {
        pipeline
            .accumulate_and_denoise_resident(
                ctx,
                &history,
                res,
                &camera(),
                state(f),
                &[],
                &denoise,
                &view,
            )
            .expect("pass");
    };

    // One pass: the trace's own split, exact to the float over the whole
    // frame, background included.
    pass(&mut res, 1);
    let (worst, de, se) = check(&mut res, "one pass", false);
    assert!(worst < 1e-5, "the lobes do not sum to the sample: {worst}");
    // Both halves carry something: the plastic's and the metal's lobes.
    assert!(de > 0.0 && se > 0.0);

    // Four passes: the fold keeps the sum, wherever both lobes may hold four
    // samples. The background cannot — its roughness is zero, so its
    // specular history is capped at one — and is left out.
    for f in 2..=4 {
        pass(&mut res, f);
    }
    let (worst, _, _) = check(&mut res, "four passes", true);
    assert!(worst < 1e-5, "the folded lobes do not sum to the folded sample: {worst}");
}

/// The specular demodulator in guide plane 4 is `spec_albedo` on the CPU, at
/// the pixel's own view cosine.
#[test]
#[ignore = "requires GPU"]
fn the_specular_albedo_guide_matches_the_cpu_lookup() {
    let Some(ctx) = ctx_or_skip("the_specular_albedo_guide_matches_the_cpu_lookup") else {
        return;
    };
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let history = HistoryPipeline::new(ctx).expect("history pipeline");
    let view = target(ctx);
    // Metals, so F0 is the base colour with nothing in between.
    let sphere = Pbr {
        base_color: [0.95, 0.64, 0.54],
        metallic: 1.0,
        roughness: 0.15,
        ..Default::default()
    };
    let plane = Pbr {
        base_color: [0.56, 0.57, 0.58],
        metallic: 1.0,
        roughness: 0.7,
        ..Default::default()
    };
    let fx = Fixture::new(sphere, plane);
    let mut res = pipeline.resident_scene(ctx, fx.scene(), W, H);
    pipeline
        .accumulate_and_denoise_resident(
            ctx,
            &history,
            &mut res,
            &camera(),
            state(1),
            &[],
            &GpuDenoiseParams::default(),
            &view,
        )
        .expect("pass");
    let g = pollster::block_on(pipeline.read_guides(ctx, &mut res)).expect("guides");

    let cam = camera();
    let mut worst = 0.0f32;
    let mut checked = 0;
    for y in 0..H {
        for x in 0..W {
            let i = (y * W + x) as usize;
            if g.depth[i] <= 0.0 {
                continue;
            }
            let m = if g.id[i] as u32 == 1 { &sphere } else { &plane };
            assert!((g.roughness[i] - m.roughness).abs() < 1e-6, "pixel {i}: roughness");
            let d = pixel_dir(&cam, x, y);
            let n = &g.normal[i * 3..i * 3 + 3];
            let mu = (-(n[0] * d[0] + n[1] * d[1] + n[2] * d[2])).max(1e-3);
            let cpu = kosm_render::spec_albedo(m.base_color, m.roughness, mu);
            for k in 0..3 {
                worst = worst.max((cpu[k] - g.spec_albedo[i * 3 + k]).abs());
            }
            checked += 1;
        }
    }
    eprintln!("spec albedo: {checked} pixels, worst |cpu - gpu| = {worst:.2e}");
    assert!(checked > 1000);
    // The view cosine is recomputed on the CPU from the pixel's *centre*
    // ray, and the trace jittered its own inside the pixel, so this is a
    // tolerance on the cosine, not on the lookup.
    assert!(worst < 5e-3, "the guide disagrees with the CPU lookup by {worst}");
}

/// A mirror sliding in its own plane: the reflection stays where it was, the
/// surface does not. The specular reprojection follows the reflected point
/// and keeps the history; the surface reprojection follows the mirror,
/// carries a shifted reflection under every pixel, and the clamp throws it
/// away.
#[test]
#[ignore = "requires GPU"]
fn a_sliding_mirror_keeps_its_reflection_under_the_specular_reprojection() {
    let Some(ctx) =
        ctx_or_skip("a_sliding_mirror_keeps_its_reflection_under_the_specular_reprojection")
    else {
        return;
    };
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let history = HistoryPipeline::new(ctx).expect("history pipeline");
    let view = target(ctx);
    // A dark matte sphere over a glossy metal floor, under the sky and
    // nothing else: the floor's specular half is the sphere's silhouette
    // against the reflected sky, smooth enough that the clamp fires on a
    // shifted reflection and not on the lobe's own noise. The floor's
    // roughness puts its history cap (about 16) well above the clamp's
    // reset length.
    let mut fx = Fixture::new(
        Pbr {
            base_color: [0.05, 0.05, 0.05],
            roughness: 0.9,
            ..Default::default()
        },
        Pbr {
            base_color: [0.9, 0.9, 0.9],
            metallic: 1.0,
            roughness: 0.5,
            ..Default::default()
        },
    );
    fx.lights.clear();
    let mut res = pipeline.resident_scene(ctx, fx.scene(), W, H);
    let denoise = GpuDenoiseParams {
        lobes: true,
        ..GpuDenoiseParams::default()
    };
    const FRAMES: u32 = 12;
    // The floor is instance 0 and slides along x by this much a frame — a
    // few pixels at this size, so a carried pixel is visibly the wrong one.
    let step = 0.4_f32;
    let motion = InstanceMotion::new(
        &[InstanceMotion::STATIC, 0],
        &[[
            1.0, 0.0, 0.0, -step, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0,
        ]],
    );
    for f in 0..FRAMES {
        pipeline
            .accumulate_and_denoise_resident_moving(
                ctx,
                &history,
                &mut res,
                &camera(),
                state(f + 1),
                &[],
                &denoise,
                &view,
                Some(&camera()),
                Some(&motion),
            )
            .expect("pass");
    }
    let summed = pollster::block_on(pipeline.read_history(ctx, &mut res))
        .expect("read")
        .expect("history");
    let (_, spec) = pollster::block_on(pipeline.read_history_lobes(ctx, &mut res))
        .expect("read")
        .expect("history");
    let g = pollster::block_on(pipeline.read_guides(ctx, &mut res)).expect("guides");

    // Over the floor's pixels: how much history each reprojection kept.
    let (mut surface, mut virtual_, mut k) = (0.0f64, 0.0f64, 0usize);
    for i in 0..(W * H) as usize {
        if g.id[i] as u32 != 2 {
            continue;
        }
        surface += summed.count[i] as f64;
        virtual_ += spec.count[i] as f64;
        k += 1;
    }
    let mut hs = [0usize; 20];
    let mut hv = [0usize; 20];
    for i in 0..(W * H) as usize {
        if g.id[i] as u32 != 2 {
            continue;
        }
        hs[(summed.count[i] as usize).min(19)] += 1;
        hv[(spec.count[i] as usize).min(19)] += 1;
    }
    eprintln!("hist surface {hs:?}\nhist virtual {hv:?}");
    let (surface, virtual_) = (surface / k as f64, virtual_ / k as f64);
    eprintln!(
        "sliding mirror: {k} floor pixels, mean history — surface reprojection {surface:.1}, \
         specular (virtual) reprojection {virtual_:.1}, of {FRAMES} frames"
    );
    assert!(k > 500);
    assert!(
        virtual_ > surface + 1.0,
        "the specular reprojection kept {virtual_:.1} against the surface's {surface:.1}"
    );
}
