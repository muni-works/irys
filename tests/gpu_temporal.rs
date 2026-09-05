//! Per-pixel temporal accumulation: what the live tier is made of.
//!
//! The rectangle mask was the old answer to "what moved": every pixel inside a
//! box around a moved object restarted from one sample, and the box was
//! visible as a rectangle of grain. These tests pin the answer that replaced
//! it — per-pixel validity, computed on the device.
//!
//! Four claims:
//!
//! * a translating sphere over a static plane keeps *both* histories, and only
//!   the trail it uncovers restarts;
//! * a light that moves is caught by the neighbourhood clamp within a few
//!   frames, even though nothing about the geometry changed;
//! * a one-frame pixel's variance is a spatial estimate and is non-zero, which
//!   is what buys it a wide filter on the frame it appears;
//! * and so the first frame carries no grain: no pixel of the resolved image
//!   is far from its own 3x3 neighbourhood.
//!
//! Plus the claim the still tier depends on: with the history capped at 64 a
//! long static render lands where an uncapped one does, to within a display
//! bit.
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

/// The sphere's index in `Fixture::geometry`. The analytic module hands a
/// primitive's index back as the hit's `face_idx`, so this *is* the id the
/// guide plane carries (biased by one) and the instance table is keyed on.
const SPHERE: usize = 0;

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
    fn new() -> Self {
        Self {
            geometry: AnalyticGeometry {
                prims: vec![
                    AnalyticPrim::sphere([0.0, 0.0, 1.0], 1.0, 0),
                    AnalyticPrim::plane([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 1),
                ],
            },
            materials: vec![
                GpuMaterial::from_pbr(Pbr {
                    base_color: [0.8, 0.78, 0.74],
                    roughness: 0.45,
                    ..Default::default()
                }),
                GpuMaterial::from_pbr(Pbr {
                    base_color: [0.35, 0.35, 0.36],
                    roughness: 0.9,
                    ..Default::default()
                }),
            ],
            lights: Self::rig(Point3::new(0.0, 0.0, 1.0)),
        }
    }

    fn rig(at: Point3) -> Vec<GpuAreaLight> {
        pathtrace::studio_rig(at, 3.0)
            .iter()
            .map(GpuAreaLight::from_area_light)
            .collect()
    }

    /// Put the sphere at `x` along the world x axis.
    fn move_sphere(&mut self, x: f32) {
        self.geometry.prims[SPHERE].a[0] = x;
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

/// A storage texture the resolve pass can write, plus a readback for it.
struct Target {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    readback: wgpu::Buffer,
    padded_bpr: u32,
}

impl Target {
    fn new(ctx: &GpuContext) -> Self {
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("temporal test target"),
            size: wgpu::Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let padded_bpr = (W * 4).div_ceil(256) * 256;
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("temporal test readback"),
            size: (padded_bpr as u64) * (H as u64),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            texture,
            view,
            readback,
            padded_bpr,
        }
    }

    fn pixels(&self, ctx: &GpuContext) -> Vec<u8> {
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bpr),
                    rows_per_image: Some(H),
                },
            },
            wgpu::Extent3d {
                width: W,
                height: H,
                depth_or_array_layers: 1,
            },
        );
        ctx.queue.submit(Some(enc.finish()));
        let slice = self.readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll");
        let data = slice.get_mapped_range().expect("mapped");
        let mut out = Vec::with_capacity((W * H * 4) as usize);
        for row in 0..H {
            let start = (row * self.padded_bpr) as usize;
            out.extend_from_slice(&data[start..start + (W * 4) as usize]);
        }
        drop(data);
        self.readback.unmap();
        out
    }
}

/// The sphere's silhouette this frame, in pixels: which of them the sphere
/// covers, worked out on the CPU from the same analytic geometry the shader
/// traces. Cheaper and more direct than reading an id plane back.
fn sphere_mask(cx: f32) -> Vec<bool> {
    let cam = camera();
    let eye = [cam.position[0], cam.position[1], cam.position[2]];
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
        cam.target[0] - eye[0],
        cam.target[1] - eye[1],
        cam.target[2] - eye[2],
    ]);
    let right = norm(cross(fwd, [cam.up[0], cam.up[1], cam.up[2]]));
    let up = cross(right, fwd);
    let tan = (cam.fov * 0.5).tan();
    let aspect = W as f32 / H as f32;
    let c = [cx, 0.0, 1.0];

    let mut out = vec![false; (W * H) as usize];
    for y in 0..H {
        for x in 0..W {
            let ndc_x = (x as f32 + 0.5) / W as f32 * 2.0 - 1.0;
            let ndc_y = 1.0 - (y as f32 + 0.5) / H as f32 * 2.0;
            let d = norm([
                fwd[0] + right[0] * ndc_x * tan * aspect + up[0] * ndc_y * tan,
                fwd[1] + right[1] * ndc_x * tan * aspect + up[1] * ndc_y * tan,
                fwd[2] + right[2] * ndc_x * tan * aspect + up[2] * ndc_y * tan,
            ]);
            let oc = [eye[0] - c[0], eye[1] - c[1], eye[2] - c[2]];
            let b = oc[0] * d[0] + oc[1] * d[1] + oc[2] * d[2];
            let cq = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - 1.0;
            out[(y * W + x) as usize] = b * b - cq > 0.0 && -b > 0.0;
        }
    }
    out
}

/// A moving sphere keeps its own history; so does the plane it slides over;
/// and the trail it uncovers does not.
///
/// This is the whole of what replaced the rectangle mask. Under the mask every
/// one of these pixels — sphere, plane and trail alike — sat inside the box
/// around the sphere's old and new poses and restarted from one sample. Here
/// the sphere is carried by its own transform, the plane by the identity, and
/// only the pixels that were sphere and are now plane have nothing to carry.
#[test]
#[ignore = "requires GPU"]
fn a_moving_sphere_keeps_its_history_and_its_trail_does_not() {
    let Some(ctx) = ctx_or_skip("a_moving_sphere_keeps_its_history_and_its_trail_does_not") else {
        return;
    };
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let history = HistoryPipeline::new(ctx).expect("history pipeline");
    let target = Target::new(ctx);
    let denoise = GpuDenoiseParams::default();

    const FRAMES: u32 = 12;
    // Fast enough that the silhouette walks several pixels a frame: a ball
    // creeping under a pixel a frame is carried by the camera-only
    // reprojection whether or not anyone tells it what moved, and this test
    // would pass without the feature it is testing.
    let step = 0.25_f32;

    // Sixteen frames of a sphere translating in x, with or without telling
    // the device that it is the sphere that is moving.
    let run = |told: bool| {
        let mut fx = Fixture::new();
        let mut res = pipeline.resident_scene(ctx, fx.scene(), W, H);
        for f in 0..FRAMES {
            fx.move_sphere(f as f32 * step);
            res.update_scene(ctx, fx.scene());
            let motion = InstanceMotion::new(
                &[0, InstanceMotion::STATIC],
                &[[
                    1.0, 0.0, 0.0, -step, //
                    0.0, 1.0, 0.0, 0.0, //
                    0.0, 0.0, 1.0, 0.0,
                ]],
            );
            pipeline
                .accumulate_and_denoise_resident_moving(
                    ctx,
                    &history,
                    &mut res,
                    &camera(),
                    state(f + 1),
                    &[],
                    &denoise,
                    &target.view,
                    // The camera never moves, so the reprojection is offered
                    // the same view either way. What differs is whether it is
                    // told the *sphere* moved: without that it validates the
                    // ball's pixels against the ball where it used to be,
                    // and every one of them fails.
                    Some(&camera()),
                    if told { Some(&motion) } else { None },
                )
                .expect("pass");
        }
        let _ = target.pixels(ctx);
        pollster::block_on(pipeline.read_history(ctx, &mut res))
            .expect("read")
            .expect("history")
    };

    let told = run(true);
    let blind = run(false);

    let now = sphere_mask((FRAMES - 1) as f32 * step);
    let before = sphere_mask(0.0);

    let mean_over = |h: &kosm_render::gpu::History, m: &dyn Fn(usize) -> bool| {
        let mut s = 0.0f64;
        let mut k = 0usize;
        for i in 0..(W * H) as usize {
            if m(i) {
                s += h.count[i] as f64;
                k += 1;
            }
        }
        (s / k.max(1) as f64, k)
    };

    let (told_sphere, sphere_n) = mean_over(&told, &|i| now[i]);
    let (blind_sphere, _) = mean_over(&blind, &|i| now[i]);
    let (told_plane, plane_n) = mean_over(&told, &|i| !now[i] && !before[i]);
    let (told_trail, trail_n) = mean_over(&told, &|i| !now[i] && before[i]);

    println!(
        "mean history over {sphere_n} sphere pixels: {told_sphere:.1} told, {blind_sphere:.1} blind; \
         {plane_n} plane pixels {told_plane:.1}; {trail_n} trail pixels {told_trail:.1}"
    );

    assert!(sphere_n > 100, "the sphere should cover real pixels");
    assert!(trail_n > 20, "the sphere did not move far enough to uncover a trail");

    // The sphere keeps a real history — most of the sixteen frames — and
    // keeps it *because* it was told what moved.
    assert!(
        told_sphere >= 8.0,
        "the moving sphere averaged only {told_sphere:.1} samples of history over sixteen frames"
    );
    assert!(
        told_sphere > blind_sphere * 2.0,
        "telling the device what moved bought nothing: {told_sphere:.1} samples against \
         {blind_sphere:.1} without the motion table"
    );
    // The plane it slides over never restarts at all.
    assert!(
        told_plane >= 8.0,
        "the static plane lost its history: {told_plane:.1} samples"
    );
    // And the strip the sphere uncovered has nothing to carry.
    assert!(
        told_trail < told_plane * 0.85,
        "the uncovered trail kept a history it could not have had: {told_trail:.1} samples \
         against the plane's {told_plane:.1}"
    );
}

/// A light that moves is caught by the clamp, without anything telling the
/// device it moved.
///
/// This is the case no geometric test can see: the floor is still the floor,
/// at the same depth with the same normal and the same id, and the only thing
/// that changed is how bright it is. Under a plain running mean a 64-sample
/// history takes a hundred frames to let go of the old shadow. The clamp reels
/// it in against what the raw sample says the pixel plausibly is.
#[test]
#[ignore = "requires GPU"]
fn a_moved_light_is_clamped_out_within_four_frames() {
    let Some(ctx) = ctx_or_skip("a_moved_light_is_clamped_out_within_four_frames") else {
        return;
    };
    let mut fx = Fixture::new();
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let history = HistoryPipeline::new(ctx).expect("history pipeline");
    let target = Target::new(ctx);
    let denoise = GpuDenoiseParams::default();
    let mut res = pipeline.resident_scene(ctx, fx.scene(), W, H);

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
                &target.view,
            )
            .expect("pass");
    };

    // Converge under the original rig.
    for f in 1..=24 {
        pass(&mut res, f);
    }
    let settled = pollster::block_on(pipeline.read_history(ctx, &mut res))
        .expect("read")
        .expect("history");

    // Now move the rig. Nothing about the geometry changes.
    fx.lights = Fixture::rig(Point3::new(2.5, 1.5, 2.0));
    res.set_lights(ctx, &fx.lights);

    // The reference: where the picture is *going*, from a history that never
    // saw the old rig at all.
    let mut fresh = pipeline.resident_scene(ctx, fx.scene(), W, H);
    for f in 1..=24 {
        pipeline
            .accumulate_and_denoise_resident(
                ctx,
                &history,
                &mut fresh,
                &camera(),
                state(f),
                &[],
                &denoise,
                &target.view,
            )
            .expect("pass");
    }
    let goal = pollster::block_on(pipeline.read_history(ctx, &mut fresh))
        .expect("read")
        .expect("history");

    // Blur both before comparing. Two independent 24-sample estimates differ
    // by their own noise everywhere, and that noise is much larger than the
    // lighting change this test is trying to see; a 5x5 box knocks it down by
    // five and leaves the change, which is smooth and large.
    let blur = |rgb: &[f32]| {
        let mut out = vec![0.0f32; rgb.len()];
        for y in 0..H as i32 {
            for x in 0..W as i32 {
                for c in 0..3 {
                    let mut s = 0.0;
                    let mut k = 0.0;
                    for dy in -2..=2 {
                        for dx in -2..=2 {
                            let (qx, qy) = (x + dx, y + dy);
                            if qx < 0 || qy < 0 || qx >= W as i32 || qy >= H as i32 {
                                continue;
                            }
                            s += rgb[((qy as u32 * W + qx as u32) as usize) * 3 + c];
                            k += 1.0;
                        }
                    }
                    out[((y as u32 * W + x as u32) as usize) * 3 + c] = s / k;
                }
            }
        }
        out
    };
    let goal_b = blur(&goal.rgb);
    let err = |h: &kosm_render::gpu::History| {
        let b = blur(&h.rgb);
        let mut s = 0.0f64;
        for i in 0..b.len() {
            let d = (b[i] - goal_b[i]) as f64;
            s += d * d;
        }
        (s / b.len() as f64).sqrt()
    };

    let before = err(&settled);
    for f in 25..=28 {
        pass(&mut res, f);
    }
    let after4 = pollster::block_on(pipeline.read_history(ctx, &mut res))
        .expect("read")
        .expect("history");
    let after = err(&after4);

    println!("stale-lighting RMS: {before:.4} before, {after:.4} after four frames");
    for f in 29..=32 {
        pass(&mut res, f);
    }
    let after8 = pollster::block_on(pipeline.read_history(ctx, &mut res))
        .expect("read")
        .expect("history");
    let eight = err(&after8);
    println!("... and {eight:.4} after eight");
    assert!(
        after < before * 0.80,
        "four frames after the light moved the picture is still {after:.4} from where it is going, \
         against {before:.4} before. The clamp is not reeling the stale history in."
    );
    assert!(
        eight < before * 0.68,
        "eight frames after the light moved the stale lighting is still {eight:.4} out of {before:.4}"
    );
}

/// A pixel with one frame of history gets a *spatial* variance, and a non-zero
/// one.
///
/// Two temporal moments over a single sample are not an error bar. Without the
/// spatial fallback the à-trous filter's luminance stop reads that pixel's own
/// sample as its own tolerance, which is the widest possible licence and the
/// least informative; with it the pixel is told how noisy its neighbourhood
/// actually is on the frame it appears.
#[test]
#[ignore = "requires GPU"]
fn a_one_frame_pixel_gets_a_spatial_variance() {
    let Some(ctx) = ctx_or_skip("a_one_frame_pixel_gets_a_spatial_variance") else {
        return;
    };
    let fx = Fixture::new();
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let history = HistoryPipeline::new(ctx).expect("history pipeline");
    let target = Target::new(ctx);
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
            &target.view,
        )
        .expect("pass");
    let h = pollster::block_on(pipeline.read_history(ctx, &mut res))
        .expect("read")
        .expect("history");

    let mut lit = 0usize;
    let mut nonzero = 0usize;
    for i in 0..(W * H) as usize {
        if h.count[i] == 1 && h.rgb[i * 3] > 0.0 {
            lit += 1;
            if h.variance[i] > 0.0 {
                nonzero += 1;
            }
        }
    }
    println!("{nonzero}/{lit} one-frame pixels carry a non-zero variance");
    assert!(lit > 500, "expected a frame's worth of one-sample pixels");
    assert_eq!(
        nonzero, lit,
        "a one-frame pixel with no variance has no error bar for the filter to widen from"
    );
}

/// The first frame reaches the screen with no more grain than a converged one.
///
/// The claim the whole live tier is for. A resolved frame has legitimate
/// high-frequency detail — silhouettes, the contact shadow's edge — so
/// "smooth" cannot be the test; what can be is that a *one-sample* frame is no
/// rougher than a 64-sample one. With the spatial variance feeding the à-trous
/// pass it is not. Without it the first frame's own single sample is its own
/// error bar, the luminance stop trusts everything, and the same measurement
/// runs several times higher.
#[test]
#[ignore = "requires GPU"]
fn the_first_frame_has_no_grain() {
    let Some(ctx) = ctx_or_skip("the_first_frame_has_no_grain") else {
        return;
    };
    let fx = Fixture::new();
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let history = HistoryPipeline::new(ctx).expect("history pipeline");
    let target = Target::new(ctx);

    let run = |passes: u32, denoise: GpuDenoiseParams| {
        let mut res = pipeline.resident_scene(ctx, fx.scene(), W, H);
        for f in 1..=passes {
            pipeline
                .accumulate_and_denoise_resident(
                    ctx,
                    &history,
                    &mut res,
                    &camera(),
                    state(f),
                    &[],
                    &denoise,
                    &target.view,
                )
                .expect("pass");
        }
        target.pixels(ctx)
    };

    // How far each pixel is from the mean of its own 3x3, per channel,
    // summarised at the 95th percentile so a silhouette does not set the
    // number on its own.
    let roughness = |px: &[u8]| {
        let mut devs: Vec<u32> = Vec::with_capacity((W * H) as usize);
        for y in 1..H - 1 {
            for x in 1..W - 1 {
                let at = |xx: u32, yy: u32, c: usize| px[((yy * W + xx) * 4) as usize + c] as i32;
                let mut worst = 0;
                for c in 0..3 {
                    let mut acc = 0;
                    for dy in 0..3 {
                        for dx in 0..3 {
                            acc += at(x + dx - 1, y + dy - 1, c);
                        }
                    }
                    worst = worst.max((at(x, y, c) - acc / 9).abs());
                }
                devs.push(worst as u32);
            }
        }
        devs.sort_unstable();
        (devs[devs.len() / 2], devs[devs.len() * 95 / 100])
    };

    let first = run(1, GpuDenoiseParams::default());
    let converged = run(64, GpuDenoiseParams::default());
    let blind = run(
        1,
        GpuDenoiseParams {
            spatial_variance: false,
            ..GpuDenoiseParams::default()
        },
    );

    let (m1, p1) = roughness(&first);
    let (mc, pc) = roughness(&converged);
    let (mb, pb) = roughness(&blind);
    println!(
        "3x3 deviation (median, p95): first frame ({m1}, {p1}), converged ({mc}, {pc}), \
         first frame without the spatial variance ({mb}, {pb})"
    );

    assert!(
        p1 <= pc + 4,
        "the first frame is grainier than a converged one: p95 {p1}/255 against {pc}/255"
    );
    assert!(
        p1 < pb,
        "the spatial variance bought nothing: p95 {p1}/255 with it, {pb}/255 without"
    );
}

/// Capping the history at 64 costs a long static render nothing.
///
/// The cap is what makes the live tier live, and the still tier — `--shot`,
/// many passes, nothing moving — has to be able to ignore it. An exponential
/// moving average with a 1/64 floor has an effective window of 127 samples, so
/// against a converged reference a 128-pass capped render is as close as a
/// 128-pass true mean is. (`--shot` also switches the clamp off: neighbourhood
/// clamping is a live-frame instrument and on a still frame it only ever costs
/// convergence.)
#[test]
#[ignore = "requires GPU"]
fn the_history_cap_does_not_move_a_still_render() {
    let Some(ctx) = ctx_or_skip("the_history_cap_does_not_move_a_still_render") else {
        return;
    };
    let fx = Fixture::new();
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let history = HistoryPipeline::new(ctx).expect("history pipeline");
    let target = Target::new(ctx);

    let shot = GpuDenoiseParams {
        clamp_k: 0.0,
        ..GpuDenoiseParams::default()
    };
    let run = |denoise: GpuDenoiseParams, passes: u32| {
        let mut res = pipeline.resident_scene(ctx, fx.scene(), W, H);
        for f in 1..=passes {
            pipeline
                .accumulate_and_denoise_resident(
                    ctx,
                    &history,
                    &mut res,
                    &camera(),
                    state(f),
                    &[],
                    &denoise,
                    &target.view,
                )
                .expect("pass");
        }
        pollster::block_on(pipeline.read_history(ctx, &mut res))
            .expect("read")
            .expect("history")
            .rgb
    };

    let uncapped = GpuDenoiseParams {
        history_cap: u32::MAX,
        ..shot
    };
    let reference = run(uncapped, 640);
    let capped_128 = run(shot, 128);
    let uncapped_128 = run(uncapped, 128);

    let rmse = |a: &[f32]| {
        let mut s = 0.0f64;
        for i in 0..a.len() {
            let d = (a[i] - reference[i]) as f64;
            s += d * d;
        }
        (s / a.len() as f64).sqrt()
    };
    let (c, u) = (rmse(&capped_128), rmse(&uncapped_128));
    println!("128 passes against a 640-pass reference: capped {c:.5}, uncapped {u:.5}");
    assert!(
        c < u * 1.25,
        "the 1/64 cap cost a still render real convergence: {c:.5} against {u:.5}"
    );
}
