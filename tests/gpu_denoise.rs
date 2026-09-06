//! The device denoiser against the CPU one, and what the filter is worth.
//!
//! Two things live here. The first is the parity claim that makes the GPU
//! history trustworthy at all: with a one-sample history the device à-trous
//! pass must produce, filter weight for filter weight, the frame
//! [`kosm_render::pathtrace::denoise`] would have produced from the same
//! `Film`. Kept in the renderer's own suite over `AnalyticGeometry`, so no
//! client crate is needed to run it.
//!
//! The second is a measurement rather than an assertion: RMSE against a
//! converged reference at 1, 4 and 16 samples, with and without the filter,
//! and the per-pixel iteration budget the filter now spends.
//!
//! Run with `--features gpu -- --ignored --test-threads=1 --nocapture`.
#![cfg(all(feature = "gpu", not(target_arch = "wasm32")))]

use kosm_render::Point3;
use kosm_render::gpu::wgpu;
use kosm_render::gpu::{
    AnalyticGeometry, AnalyticPrim, GpuAreaLight, GpuCamera, GpuContext, GpuDenoiseParams,
    GpuMaterial, GpuRenderState, HistoryPipeline, RayTracePipeline, SceneRef, atrous_iters_for,
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

/// A sphere on a floor under the studio rig — enough geometry for a
/// silhouette, a contact shadow and a lit gradient, which is what an
/// edge-stopping filter has to get right.
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
    GpuCamera::new(
        [5.0, -5.0, 3.5],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        0.7,
        W,
        H,
    )
}

/// Path trace only: no edge overlay, no stylisation.
fn state(frame: u32) -> GpuRenderState {
    let mut s = GpuRenderState::new(frame);
    s.enable_edges = 0;
    s.stylize = 0;
    s.ground_enabled = 0;
    s.set_camera_visible_lights(false);
    s
}

/// A storage texture the history pass can resolve into, plus its readback.
struct Target {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    readback: wgpu::Buffer,
    padded_bpr: u32,
    width: u32,
    height: u32,
}

impl Target {
    fn new(ctx: &GpuContext, width: u32, height: u32) -> Self {
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("denoise test target"),
            size: wgpu::Extent3d {
                width,
                height,
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
        let padded_bpr = (width * 4).div_ceil(256) * 256;
        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("denoise test readback"),
            size: (padded_bpr as u64) * (height as u64),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self {
            texture,
            view,
            readback,
            padded_bpr,
            width,
            height,
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
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
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
        let mut out = Vec::with_capacity((self.width * self.height * 4) as usize);
        for row in 0..self.height {
            let start = (row * self.padded_bpr) as usize;
            out.extend_from_slice(&data[start..start + (self.width * 4) as usize]);
        }
        drop(data);
        self.readback.unmap();
        out
    }
}

/// The device à-trous pass must be `pathtrace::denoise`, weight for weight.
///
/// One pass, so every pixel's history is a single sample: the fade that scales
/// the filter down as a pixel converges is at full strength, and the per-pixel
/// iteration budget is at its maximum, so the shader is doing exactly what the
/// CPU filter does over the same `Film`.
#[test]
#[ignore = "requires GPU"]
fn the_device_denoise_matches_the_cpu_filter() {
    let Some(ctx) = ctx_or_skip("the_device_denoise_matches_the_cpu_filter") else {
        return;
    };
    let fx = Fixture::new();
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let history = HistoryPipeline::new(ctx).expect("history pipeline");
    let n = (W * H) as usize;
    // `spatial_variance` off. The default is on, and on purpose: SVGF's
    // spatial estimate is what lets a one-frame pixel be filtered as wide as
    // its neighbours say it needs, instead of showing the error bar its own
    // single sample gives — which is no error bar at all. The CPU filter has
    // no such estimator, so the two tiers cannot agree byte for byte with it
    // on. This test is about the *filter* being the same filter, so it turns
    // the estimator off and pins the à-trous weights; the live tier runs with
    // it on.
    // The disocclusion fallback off, for the same reason: it widens a
    // one-sample pixel's filter past what the CPU filter runs.
    let denoise = GpuDenoiseParams {
        spatial_variance: false,
        fresh_extra_iters: 0,
        fresh_lum_relax: 1.0,
        ..GpuDenoiseParams::default()
    };

    // The CPU reference: the same single raw sample, filtered and tonemapped
    // through the CPU tier.
    let mut ref_res = pipeline.resident_scene(ctx, fx.scene(), W, H);
    let mut film =
        pollster::block_on(pipeline.render_resident_linear(ctx, &mut ref_res, &camera(), state(1)))
            .expect("linear pass");
    let opts = pathtrace::PathTraceOptions {
        denoise_iters: denoise.iters,
        sigma_normal: denoise.sigma_normal,
        sigma_depth: denoise.sigma_depth,
        sigma_lum: denoise.sigma_lum,
        ..Default::default()
    };
    pathtrace::denoise(&mut film, &opts);
    let want = film.to_srgb8(denoise.exposure, false);

    // The device: the same pass, filtered and tonemapped on the GPU.
    let mut res = pipeline.resident_scene(ctx, fx.scene(), W, H);
    let target = Target::new(ctx, W, H);
    pipeline
        .accumulate_and_denoise_resident(
            ctx,
            &history,
            &mut res,
            &camera(),
            state(1),
            &[],
            &denoise,
            &target.view,
        )
        .expect("history pass");
    let got = target.pixels(ctx);

    // Tolerance. Both tiers run the same filter in f32, so the only
    // disagreement available is arithmetic: `exp` and `pow` are
    // implementation-defined to a couple of ULP either way, and the 25 taps
    // are summed in a different order. Through ACES and the sRGB transfer that
    // is worth well under one code; two is generous, and a real disagreement
    // about a filter weight moves whole regions by tens.
    let mut worst = 0u8;
    let mut sum = 0.0f64;
    let mut over = 0usize;
    for i in 0..n {
        for k in 0..3 {
            let d = got[i * 4 + k].abs_diff(want[i * 4 + k]);
            worst = worst.max(d);
            sum += d as f64;
            if d > 2 {
                over += 1;
            }
        }
    }
    let mean = sum / (n * 3) as f64;
    eprintln!("worst channel difference {worst}/255, mean {mean:.4}, {over} channels over 2");
    assert!(
        want.chunks(4).any(|p| p[0] > 8),
        "the CPU reference frame is black — nothing was rendered",
    );
    assert_eq!(
        over, 0,
        "{over} channels differ from the CPU filter by more than 2/255 (worst \
         {worst}). The device à-trous pass is not the same filter as \
         `pathtrace::denoise`.",
    );
    assert!(mean < 0.2, "mean channel difference {mean:.4} is too large");
}

/// The per-pixel iteration budget: full on the first sample — which is what
/// keeps the parity test above honest — and none once a pixel has converged.
#[test]
fn the_iteration_budget_falls_with_the_history() {
    let (iters, cutoff) = (5u32, 32u32);
    assert_eq!(atrous_iters_for(1.0, iters, cutoff), iters);
    assert_eq!(atrous_iters_for(cutoff as f32, iters, cutoff), 0);
    assert_eq!(atrous_iters_for(cutoff as f32 + 100.0, iters, cutoff), 0);
    // Monotone in between, and never more than the caller asked for.
    let mut prev = iters;
    for c in 1..=cutoff {
        let k = atrous_iters_for(c as f32, iters, cutoff);
        assert!(k <= prev, "budget rose from {prev} to {k} at count {c}");
        assert!(k <= iters);
        prev = k;
    }
    // Zero iterations in means zero out, whatever the history.
    assert_eq!(atrous_iters_for(1.0, 0, cutoff), 0);
}

/// RMSE against a converged reference at 1, 4 and 16 samples, with the filter
/// and without it. A measurement, printed with `--nocapture`.
#[test]
#[ignore = "requires GPU"]
fn measure_the_denoiser() {
    let Some(ctx) = ctx_or_skip("measure_the_denoiser") else {
        return;
    };
    let fx = Fixture::new();
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let history = HistoryPipeline::new(ctx).expect("history pipeline");
    let denoise = GpuDenoiseParams::default();

    // The reference: a long accumulation, unfiltered.
    let reference = accumulate(ctx, &pipeline, &history, &fx, 512, None);

    eprintln!("  samples      raw RMSE   denoised RMSE   iterations/pixel");
    for samples in [1u32, 4, 16] {
        let raw = accumulate(ctx, &pipeline, &history, &fx, samples, None);
        let filtered = accumulate(ctx, &pipeline, &history, &fx, samples, Some(&denoise));
        let budget = atrous_iters_for(samples as f32, denoise.iters, denoise.count_cutoff);
        eprintln!(
            "  {samples:>7}   {:>11.5}   {:>13.5}   {budget:>16}",
            rmse(&raw, &reference),
            rmse(&filtered, &reference),
        );
    }
}

/// Accumulate `samples` passes and return the resolved frame, filtered when
/// `denoise` is `Some` and left as the bare running mean when it is `None`.
fn accumulate(
    ctx: &'static GpuContext,
    pipeline: &RayTracePipeline,
    history: &HistoryPipeline,
    fx: &Fixture,
    samples: u32,
    denoise: Option<&GpuDenoiseParams>,
) -> Vec<u8> {
    let mut res = pipeline.resident_scene(ctx, fx.scene(), W, H);
    let target = Target::new(ctx, W, H);
    // Zero iterations is the shader's own "do not filter" path, so the
    // unfiltered case runs through exactly the same accumulation.
    let mut params = denoise.copied().unwrap_or(GpuDenoiseParams {
        iters: 0,
        ..GpuDenoiseParams::default()
    });
    if denoise.is_none() {
        params.iters = 0;
    }
    for frame in 1..=samples {
        pipeline
            .accumulate_and_denoise_resident(
                ctx,
                history,
                &mut res,
                &camera(),
                state(frame),
                &[],
                &params,
                &target.view,
            )
            .expect("history pass");
    }
    target.pixels(ctx)
}

/// RMSE in 8-bit codes over the RGB channels.
fn rmse(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len() / 4;
    let mut s = 0.0f64;
    for i in 0..n {
        for k in 0..3 {
            let d = a[i * 4 + k] as f64 - b[i * 4 + k] as f64;
            s += d * d;
        }
    }
    (s / (n * 3) as f64).sqrt()
}

// ─── the split: accumulate per box, denoise once ──────────────────────────

/// A tiling of the 64x64 frame into four boxes on deliberately unaligned
/// seams.
///
/// 27 and 20 are not multiples of the 8x8 workgroup, so three of the four
/// boxes start mid-workgroup and all four end mid-workgroup. That is the case
/// the dispatch origin has to get right: the accumulate pass covers the box's
/// workgroups rather than the frame's, so its invocation ids are shifted, and
/// a shift that is off by anything at all folds the sample into the wrong
/// pixels.
const TILING: [[u32; 4]; 4] = [
    [0, 0, 27, 20],
    [27, 0, 37, 20],
    [0, 20, 27, 44],
    [27, 20, 37, 44],
];

/// `k` boxes accumulated and denoised once is one full-frame pass, byte for
/// byte, when the boxes tile the frame.
///
/// This is the whole claim the split rests on. `accumulate_resident` folds a
/// sample into the pixels of one box and `denoise_and_resolve_resident` runs
/// the filter chain over the frame afterwards, so `k` dirty rectangles cost
/// `k` traces and *one* denoise rather than `k` of each. If that is really
/// only a regrouping of the same work, tiling the frame with boxes has to
/// land on the same frame the fused call produces — not close, identical: the
/// same raw samples fold into the same pixels in the same order and the same
/// filter runs over the result.
#[test]
#[ignore = "requires GPU"]
fn boxes_that_tile_the_frame_are_one_full_pass() {
    let Some(ctx) = ctx_or_skip("boxes_that_tile_the_frame_are_one_full_pass") else {
        return;
    };
    let fx = Fixture::new();
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let history = HistoryPipeline::new(ctx).expect("history pipeline");
    let denoise = GpuDenoiseParams::default();

    // The reference: one unscissored fused pass.
    let mut whole = pipeline.resident_scene(ctx, fx.scene(), W, H);
    let whole_target = Target::new(ctx, W, H);
    pipeline
        .accumulate_and_denoise_resident(
            ctx,
            &history,
            &mut whole,
            &camera(),
            state(1),
            &[],
            &denoise,
            &whole_target.view,
        )
        .expect("full pass");
    let want = whole_target.pixels(ctx);

    // The split: the same frame index for every box, so every box traces the
    // sample the full pass would have traced for its pixels, then one denoise
    // over the lot.
    let mut tiled = pipeline.resident_scene(ctx, fx.scene(), W, H);
    let tiled_target = Target::new(ctx, W, H);
    for rect in TILING {
        let mut s = state(1);
        s.set_scissor(rect);
        pipeline
            .accumulate_resident(ctx, &history, &mut tiled, &camera(), s, &[], None)
            .expect("box accumulate");
    }
    pipeline
        .denoise_and_resolve_resident(ctx, &history, &mut tiled, &denoise, &tiled_target.view)
        .expect("denoise");
    let got = tiled_target.pixels(ctx);

    assert!(
        want.chunks(4).any(|p| p[0] > 8),
        "the reference frame is black — nothing was rendered",
    );
    let differing = got
        .chunks(4)
        .zip(want.chunks(4))
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing,
        0,
        "{differing} of {} pixels differ between four tiling boxes and one full \
         pass. The split is not a regrouping of the same work — check the \
         accumulate dispatch's origin against its scissor.",
        (W * H) as usize,
    );
}

/// A box accumulate touches its box and nothing else.
///
/// The scissor already stopped the *fold* outside the rectangle; what is new
/// is that the dispatch does not cover the frame at all, and a dispatch origin
/// that is wrong by a workgroup would quietly fold this pass's sample into
/// somebody else's pixels. Counting samples is the sharpest way to see it: a
/// pixel outside the box must still be on one sample after a second pass, and
/// its mean must be the float it already held.
#[test]
#[ignore = "requires GPU"]
fn a_box_accumulate_leaves_the_rest_of_the_frame_alone() {
    let Some(ctx) = ctx_or_skip("a_box_accumulate_leaves_the_rest_of_the_frame_alone") else {
        return;
    };
    let fx = Fixture::new();
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let history = HistoryPipeline::new(ctx).expect("history pipeline");
    let mut res = pipeline.resident_scene(ctx, fx.scene(), W, H);

    // One full pass, so every pixel has exactly one sample.
    pipeline
        .accumulate_resident(ctx, &history, &mut res, &camera(), state(1), &[], None)
        .expect("full accumulate");
    let before = pollster::block_on(pipeline.read_history(ctx, &mut res))
        .expect("read")
        .expect("a history");
    assert!(
        before.count.iter().all(|&c| c == 1),
        "the first pass did not leave every pixel on one sample",
    );

    // A second pass over one unaligned box.
    let rect = [11u32, 7, 29, 23];
    let mut s = state(2);
    s.set_scissor(rect);
    pipeline
        .accumulate_resident(ctx, &history, &mut res, &camera(), s, &[], None)
        .expect("box accumulate");
    let after = pollster::block_on(pipeline.read_history(ctx, &mut res))
        .expect("read")
        .expect("a history");

    let inside = |x: u32, y: u32| {
        x >= rect[0] && x < rect[0] + rect[2] && y >= rect[1] && y < rect[1] + rect[3]
    };
    let mut wrong_in = 0usize;
    let mut wrong_out = 0usize;
    let mut moved_out = 0usize;
    for y in 0..H {
        for x in 0..W {
            let i = (y * W + x) as usize;
            if inside(x, y) {
                if after.count[i] != 2 {
                    wrong_in += 1;
                }
            } else {
                if after.count[i] != 1 {
                    wrong_out += 1;
                }
                if after.rgb[i * 3..i * 3 + 3] != before.rgb[i * 3..i * 3 + 3] {
                    moved_out += 1;
                }
            }
        }
    }
    assert_eq!(
        wrong_in, 0,
        "{wrong_in} pixels inside the box did not take the sample"
    );
    assert_eq!(
        wrong_out, 0,
        "{wrong_out} pixels outside the box took a sample they were not offered",
    );
    assert_eq!(
        moved_out, 0,
        "{moved_out} pixels outside the box had their running mean moved",
    );
}

/// What the split is worth: four boxes of a tenth of the frame each, against
/// four full passes, at the viewer's own size.
///
/// A measurement, not an assertion — the numbers are the point. Three ways of
/// spending a frame:
///
/// * four fused full-frame calls, which is what a viewer with no scissor pays;
/// * four fused *box* calls, which is what it paid before this split: the
///   trace shrank to the box but the whole denoise chain ran four times;
/// * four box accumulates and one denoise, which is the split.
#[test]
#[ignore = "requires GPU"]
fn measure_the_box_split() {
    let Some(ctx) = ctx_or_skip("measure_the_box_split") else {
        return;
    };
    const TW: u32 = 512;
    const TH: u32 = 288;
    // Four boxes of about a tenth of the frame each: 162x91 is 14 742 px
    // against the frame's 147 456.
    const BOXES: [[u32; 4]; 4] = [
        [10, 10, 162, 91],
        [200, 40, 162, 91],
        [60, 150, 162, 91],
        [330, 180, 162, 91],
    ];

    let fx = Fixture::new();
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let history = HistoryPipeline::new(ctx).expect("history pipeline");
    let denoise = GpuDenoiseParams::default();
    let cam = GpuCamera::new(
        [5.0, -5.0, 3.5],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        0.7,
        TW,
        TH,
    );
    let target = Target::new(ctx, TW, TH);

    let wait = || {
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll")
    };

    // Everything is timed after a warm-up pass, so no pipeline compile or
    // first-touch allocation lands inside a measurement.
    let mut res = pipeline.resident_scene(ctx, fx.scene(), TW, TH);
    for frame in 1..=2 {
        pipeline
            .accumulate_and_denoise_resident(
                ctx,
                &history,
                &mut res,
                &cam,
                state(frame),
                &[],
                &denoise,
                &target.view,
            )
            .expect("warm-up");
    }
    wait();

    let time = |f: &mut dyn FnMut()| {
        let t = std::time::Instant::now();
        for _ in 0..REPEATS {
            f();
        }
        wait();
        t.elapsed().as_secs_f64() * 1e3 / REPEATS as f64
    };
    const REPEATS: u32 = 20;

    let mut frame = 100u32;
    let full = time(&mut || {
        for _ in 0..4 {
            frame += 1;
            pipeline
                .accumulate_and_denoise_resident(
                    ctx,
                    &history,
                    &mut res,
                    &cam,
                    state(frame),
                    &[],
                    &denoise,
                    &target.view,
                )
                .expect("full pass");
        }
    });

    let fused_boxes = time(&mut || {
        for rect in BOXES {
            frame += 1;
            let mut s = state(frame);
            s.set_scissor(rect);
            pipeline
                .accumulate_and_denoise_resident(
                    ctx,
                    &history,
                    &mut res,
                    &cam,
                    s,
                    &[],
                    &denoise,
                    &target.view,
                )
                .expect("fused box pass");
        }
    });

    let split = time(&mut || {
        for rect in BOXES {
            frame += 1;
            let mut s = state(frame);
            s.set_scissor(rect);
            pipeline
                .accumulate_resident(ctx, &history, &mut res, &cam, s, &[], None)
                .expect("box accumulate");
        }
        pipeline
            .denoise_and_resolve_resident(ctx, &history, &mut res, &denoise, &target.view)
            .expect("denoise");
    });

    eprintln!(
        "{TW}x{TH}, four boxes of 10%: {full:.2} ms four full fused passes, \
         {fused_boxes:.2} ms four fused box passes, {split:.2} ms four box \
         accumulates and one denoise ({:.2}x the fused boxes, {:.2}x the full \
         passes)",
        fused_boxes / split,
        full / split,
    );

    // Again with the fade off. Everything above ran on a history hundreds of
    // samples deep, where `atrous_iters_for` has already faded the filter out
    // and each of its five iterations is a read and a write rather than 25
    // taps — so four denoise chains were four cheap chains. `--shot` turns the
    // fade off (`CourtGpu::always_denoise`), and a viewport that has just been
    // orbited is near enough the same thing: every pixel back at a full
    // budget. That is where paying for the chain once instead of four times is
    // worth what it sounds like it should be worth.
    let hot = GpuDenoiseParams {
        count_cutoff: u32::MAX,
        ..denoise
    };
    let hot_fused = time(&mut || {
        for rect in BOXES {
            frame += 1;
            let mut s = state(frame);
            s.set_scissor(rect);
            pipeline
                .accumulate_and_denoise_resident(
                    ctx,
                    &history,
                    &mut res,
                    &cam,
                    s,
                    &[],
                    &hot,
                    &target.view,
                )
                .expect("fused box pass");
        }
    });
    let hot_split = time(&mut || {
        for rect in BOXES {
            frame += 1;
            let mut s = state(frame);
            s.set_scissor(rect);
            pipeline
                .accumulate_resident(ctx, &history, &mut res, &cam, s, &[], None)
                .expect("box accumulate");
        }
        pipeline
            .denoise_and_resolve_resident(ctx, &history, &mut res, &hot, &target.view)
            .expect("denoise");
    });
    eprintln!(
        "{TW}x{TH}, the same boxes with the fade off: {hot_fused:.2} ms fused, \
         {hot_split:.2} ms split ({:.2}x)",
        hot_fused / hot_split,
    );
}
