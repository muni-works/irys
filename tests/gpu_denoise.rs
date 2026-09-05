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
    GpuCamera::new([5.0, -5.0, 3.5], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 0.7, W, H)
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
    let denoise = GpuDenoiseParams::default();

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
