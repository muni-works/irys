//! The four stabilizers in the à-trous path: what stops the live tier from
//! boiling.
//!
//! Watching the court, the image crawled — grain that moved frame to frame
//! over surfaces whose radiance did not — and popped whenever a firefly
//! landed. Four claims, one per cause:
//!
//! * a sample a thousand times brighter than its neighbourhood moves the mean
//!   by no more than the firefly cap allows, where uncapped it resets it;
//! * a history grossly outside the raw neighbourhood's variance box is
//!   clipped into it in one frame, where the bounded fold alone would take a
//!   history length of frames;
//! * the second temporal pass over the filtered output takes the
//!   frame-to-frame movement of a still picture down by a measured factor,
//!   without moving where it converges;
//! * and a pixel restarted among converged ones comes out blurry rather than
//!   grainy.
//!
//! Plus the ramps the shader and the host both compute, pinned so neither can
//! drift.
//!
//! Run with `--features gpu -- --ignored --test-threads=1 --nocapture`.
#![cfg(all(feature = "gpu", not(target_arch = "wasm32")))]

use kosm_render::Point3;
use kosm_render::gpu::wgpu;
use kosm_render::gpu::{
    AnalyticGeometry, AnalyticPrim, GpuAreaLight, GpuCamera, GpuContext, GpuDenoiseParams,
    GpuMaterial, GpuRenderState, History, HistoryPipeline, RayTracePipeline, SceneRef,
    firefly_k_for, fresh_extra_iters_for, fresh_lum_relax_for, variance_gamma_for,
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
            label: Some("stabilizer test target"),
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
            label: Some("stabilizer test readback"),
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

/// The whole of a converged run: `passes` passes on the still fixture.
struct Rig<'a> {
    ctx: &'a GpuContext,
    pipeline: RayTracePipeline,
    history: HistoryPipeline,
    target: Target,
    fx: Fixture,
}

impl<'a> Rig<'a> {
    fn new(ctx: &'a GpuContext) -> Self {
        Self {
            ctx,
            pipeline: RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline"),
            history: HistoryPipeline::new(ctx).expect("history pipeline"),
            target: Target::new(ctx),
            fx: Fixture::new(),
        }
    }

    fn pass(
        &self,
        res: &mut kosm_render::gpu::ResidentScene,
        f: u32,
        keep: &[u8],
        denoise: &GpuDenoiseParams,
    ) {
        self.pipeline
            .accumulate_and_denoise_resident(
                self.ctx,
                &self.history,
                res,
                &camera(),
                state(f),
                keep,
                denoise,
                &self.target.view,
            )
            .expect("pass");
    }

    fn converge(
        &self,
        passes: u32,
        denoise: &GpuDenoiseParams,
    ) -> kosm_render::gpu::ResidentScene {
        let mut res = self
            .pipeline
            .resident_scene(self.ctx, self.fx.scene(), W, H);
        for f in 1..=passes {
            self.pass(&mut res, f, &[], denoise);
        }
        res
    }

    fn read(&self, res: &mut kosm_render::gpu::ResidentScene) -> History {
        pollster::block_on(self.pipeline.read_history(self.ctx, res))
            .expect("read")
            .expect("history")
    }

    /// The history's mean as a raw frame the fold would accept: what a
    /// noiseless sample of exactly the converged picture looks like.
    fn raw_of(h: &History) -> Vec<f32> {
        let n = (W * H) as usize;
        let mut raw = vec![0.0f32; n * 4];
        for i in 0..n {
            raw[i * 4..i * 4 + 3].copy_from_slice(&h.rgb[i * 3..i * 3 + 3]);
            raw[i * 4 + 3] = h.alpha[i];
        }
        raw
    }

    fn fold(
        &self,
        res: &mut kosm_render::gpu::ResidentScene,
        f: u32,
        raw: &[f32],
        denoise: &GpuDenoiseParams,
    ) {
        self.pipeline
            .fold_resident_sample(
                self.ctx,
                &self.history,
                res,
                &camera(),
                state(f),
                raw,
                denoise,
            )
            .expect("fold");
    }
}

fn lum(rgb: &[f32]) -> f32 {
    0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

/// The pixel the tests aim at: the brightest lit pixel near the frame's
/// middle, which is the top of the sphere.
fn bright_pixel(h: &History) -> usize {
    let mut best = 0;
    let mut best_l = -1.0;
    for y in H / 4..3 * H / 4 {
        for x in W / 4..3 * W / 4 {
            let i = (y * W + x) as usize;
            let l = lum(&h.rgb[i * 3..i * 3 + 3]);
            if h.count[i] > 0 && l > best_l {
                best_l = l;
                best = i;
            }
        }
    }
    best
}

/// The ramps the host computes are the ramps the shader computes.
///
/// Each has a mirror in `history.wgsl`; the numbers at the ends and the knee
/// are pinned here so a change to one side has to be a change to both.
#[test]
fn the_stabilizer_ramps_are_pinned() {
    assert_eq!(firefly_k_for(1.0, 12.0), 12.0);
    assert_eq!(firefly_k_for(32.0, 12.0), 24.0);
    assert_eq!(firefly_k_for(200.0, 12.0), 24.0);
    assert!((firefly_k_for(16.5, 12.0) - 18.0).abs() < 1e-5);

    assert_eq!(variance_gamma_for(1.0, 2.5), 1.0);
    assert_eq!(variance_gamma_for(32.0, 2.5), 2.5);
    assert!((variance_gamma_for(16.5, 2.5) - 1.75).abs() < 1e-5);

    assert_eq!(fresh_extra_iters_for(1.0, 1), 1);
    assert_eq!(fresh_extra_iters_for(3.0, 1), 1);
    assert_eq!(fresh_extra_iters_for(4.0, 1), 0);
    assert_eq!(fresh_extra_iters_for(2.0, 0), 0);

    assert_eq!(fresh_lum_relax_for(1.0, 3.0), 3.0);
    assert_eq!(fresh_lum_relax_for(4.0, 3.0), 1.0);
    assert!((fresh_lum_relax_for(2.5, 3.0) - 2.0).abs() < 1e-5);
    assert_eq!(fresh_lum_relax_for(1.0, 0.5), 1.0, "a relax under 1 is no relax");
}

/// A firefly moves the mean by no more than the cap allows.
///
/// Sixteen passes converge the fixture; then a frame is folded that is the
/// converged picture exactly, except one pixel at a thousand times its
/// brightness. Uncapped, the bounded fold puts a sixtieth of a thousand into
/// the mean and the pixel is sixty times too bright for the next sixty
/// frames — the pop. Capped, the sample is at most `k(n)` times the
/// neighbourhood and the mean moves by at most that much over `n`.
#[test]
#[ignore = "requires GPU"]
fn a_firefly_moves_the_mean_by_no_more_than_the_cap() {
    let Some(ctx) = ctx_or_skip("a_firefly_moves_the_mean_by_no_more_than_the_cap") else {
        return;
    };
    let rig = Rig::new(ctx);
    // The other clamps off, so what is measured is the cap alone.
    let capped = GpuDenoiseParams {
        clamp_k: 0.0,
        variance_gamma: 0.0,
        ..GpuDenoiseParams::default()
    };
    let uncapped = GpuDenoiseParams {
        firefly_k: 0.0,
        ..capped
    };
    const PASSES: u32 = 16;
    const FIREFLY: f32 = 1000.0;

    let run = |denoise: &GpuDenoiseParams| {
        let mut res = rig.converge(PASSES, denoise);
        let before = rig.read(&mut res);
        let p = bright_pixel(&before);
        let mut raw = Rig::raw_of(&before);
        for c in 0..3 {
            raw[p * 4 + c] *= FIREFLY;
        }
        rig.fold(&mut res, PASSES + 1, &raw, denoise);
        let after = rig.read(&mut res);
        (before, after, p)
    };

    let (before, after, p) = run(&capped);
    let (before_u, after_u, p_u) = run(&uncapped);
    assert_eq!(p, p_u, "the two runs converged to different pictures");

    let l0 = lum(&before.rgb[p * 3..p * 3 + 3]);
    let l1 = lum(&after.rgb[p * 3..p * 3 + 3]);
    let l1_u = lum(&after_u.rgb[p_u * 3..p_u * 3 + 3]);
    let n = before.count[p] as f32;
    // The most the cap lets through: a sample of k(n) times the local
    // estimate — which on a smooth surface is the pixel's own luminance to
    // within a few percent — folded at 1/(n + 1).
    let k = firefly_k_for(n, capped.firefly_k);
    let bound = l0 * (1.0 + (k * 1.15 - 1.0) / (n + 1.0));
    println!(
        "pixel {p} at n = {n}: {l0:.4} before, {l1:.4} capped (bound {bound:.4}), \
         {l1_u:.4} uncapped"
    );
    assert!(l0 > 0.0, "the target pixel is dark");
    assert!(
        (l1_u / l0) > 20.0,
        "the uncapped fold did not pop: {l1_u:.4} against {l0:.4} — the synthetic firefly \
         never reached the mean"
    );
    assert!(
        l1 <= bound,
        "the capped fold moved the mean from {l0:.4} to {l1:.4}, past the cap's bound {bound:.4}"
    );
    assert_eq!(
        after.count[p], before.count[p] + 1,
        "the cap should fold the sample, not throw it away"
    );

    // Everywhere else the fold was the picture itself, and nothing moved.
    let mut worst = 0.0f32;
    for i in 0..(W * H) as usize {
        if i == p {
            continue;
        }
        let a = lum(&after.rgb[i * 3..i * 3 + 3]);
        let b = lum(&before.rgb[i * 3..i * 3 + 3]);
        worst = worst.max((a - b).abs() / b.max(1e-3));
    }
    assert!(
        worst < 1e-3,
        "the cap touched a pixel that was not the firefly: {worst:.5} relative"
    );
    let _ = before_u;
}

/// A history grossly outside the variance box is clipped into it on the
/// frame it falls out.
///
/// The fixture converges, then folds frames of the picture at five times the
/// brightness — a light came on, and every pixel's history is now a ghost of
/// the dark room, well outside what the raw neighbourhood says the pixel is.
/// Under the bounded fold alone the ghost fades at 1/n a frame; the box clips
/// it to the edge and shortens it, and the next fold carries it most of the
/// rest of the way.
///
/// Five times, not a fraction: the box's σ is floored at the temporal spread
/// the neighbourhood has seen, so that nine taps of heavy-tailed noise cannot
/// narrow it onto a biased mean, and on this fixture that spread is over half
/// the signal. A change *inside* two of those σ is the smooth-change clamp's
/// to catch (`clamp_k`, off here), not the box's.
#[test]
#[ignore = "requires GPU"]
fn a_ghost_outside_the_variance_box_is_clipped_in_one_frame() {
    let Some(ctx) = ctx_or_skip("a_ghost_outside_the_variance_box_is_clipped_in_one_frame") else {
        return;
    };
    let rig = Rig::new(ctx);
    // The smooth-change test off, so what is measured is the box alone.
    let boxed = GpuDenoiseParams {
        clamp_k: 0.0,
        firefly_k: 0.0,
        ..GpuDenoiseParams::default()
    };
    let unboxed = GpuDenoiseParams {
        variance_gamma: 0.0,
        ..boxed
    };
    const PASSES: u32 = 16;
    const GAIN: f32 = 5.0;

    let run = |denoise: &GpuDenoiseParams| {
        let mut res = rig.converge(PASSES, denoise);
        let before = rig.read(&mut res);
        let mut raw = Rig::raw_of(&before);
        for i in 0..(W * H) as usize {
            for c in 0..3 {
                raw[i * 4 + c] *= GAIN;
            }
        }
        rig.fold(&mut res, PASSES + 1, &raw, denoise);
        let after = rig.read(&mut res);
        (before, after)
    };

    let (before, after) = run(&boxed);
    let (_, after_u) = run(&unboxed);

    // How far each lit pixel still is from where the picture went, as a
    // fraction of the jump, averaged.
    let residue = |h: &History| {
        let mut s = 0.0f64;
        let mut k = 0usize;
        for i in 0..(W * H) as usize {
            let b = lum(&before.rgb[i * 3..i * 3 + 3]);
            if before.count[i] == 0 || b < 0.05 {
                continue;
            }
            let a = lum(&h.rgb[i * 3..i * 3 + 3]);
            s += ((b * GAIN - a) / (b * (GAIN - 1.0))) as f64;
            k += 1;
        }
        (s / k.max(1) as f64, k)
    };
    let (r, k) = residue(&after);
    let (r_u, _) = residue(&after_u);
    println!("ghost residue over {k} lit pixels after one frame: {r:.3} boxed, {r_u:.3} unboxed");
    assert!(k > 500, "expected a frame's worth of lit pixels");
    assert!(
        r_u > 0.8,
        "the bounded fold alone let the ghost go too fast ({r_u:.3}); the test is not measuring the box"
    );
    assert!(
        r < 0.35,
        "one frame after the light came on, {r:.3} of the ghost is still in the history"
    );
    // ... and the history was shortened, so the next frames are worth more.
    let mut shortened = 0usize;
    for i in 0..(W * H) as usize {
        if before.count[i] > 0 && after.count[i] <= boxed.clamp_reset + 1 {
            shortened += 1;
        }
    }
    assert!(
        shortened > k / 2,
        "only {shortened} of {k} clipped pixels had their history shortened"
    );
}

/// The second temporal pass takes the flicker out of a still picture — and
/// does not move where it converges.
///
/// Forty one-sample passes on the still fixture, the presented frame read
/// after each. Every one of them is a different picture, because the mean
/// moved by 1/n and the à-trous kernel's edge-stops read that move as a
/// change of edge. The pass over the filtered output blends each frame toward
/// the last presented one, and the frame-to-frame movement drops by a
/// measured factor; the fortieth frame is still the fortieth frame.
#[test]
#[ignore = "requires GPU"]
fn the_second_temporal_pass_takes_the_flicker_out() {
    let Some(ctx) = ctx_or_skip("the_second_temporal_pass_takes_the_flicker_out") else {
        return;
    };
    let rig = Rig::new(ctx);
    const PASSES: u32 = 40;
    const FROM: u32 = 4;

    let run = |denoise: &GpuDenoiseParams| {
        let mut res = rig
            .pipeline
            .resident_scene(rig.ctx, rig.fx.scene(), W, H);
        let mut frames = Vec::new();
        for f in 1..=PASSES {
            rig.pass(&mut res, f, &[], denoise);
            if f >= FROM {
                frames.push(rig.target.pixels(rig.ctx));
            }
        }
        frames
    };
    let on = run(&GpuDenoiseParams::default());
    let off = run(&GpuDenoiseParams {
        temporal_filter: 0.0,
        ..GpuDenoiseParams::default()
    });

    // Mean absolute frame-to-frame movement, in 8-bit codes, over the frame.
    let movement = |frames: &[Vec<u8>]| {
        let mut s = 0.0f64;
        let mut n = 0usize;
        for pair in frames.windows(2) {
            for i in 0..(W * H) as usize {
                for c in 0..3 {
                    s += (pair[1][i * 4 + c] as f64 - pair[0][i * 4 + c] as f64).abs();
                    n += 1;
                }
            }
        }
        s / n as f64
    };
    let (m_on, m_off) = (movement(&on), movement(&off));

    // And the last frames agree: the pass smooths the path, not the
    // destination.
    let last_on = on.last().unwrap();
    let last_off = off.last().unwrap();
    let mut diff = 0.0f64;
    let mut worst = 0u8;
    for i in 0..(W * H * 4) as usize {
        if i % 4 == 3 {
            continue;
        }
        let d = last_on[i].abs_diff(last_off[i]);
        diff += d as f64;
        worst = worst.max(d);
    }
    diff /= (W * H * 3) as f64;
    println!(
        "frame-to-frame movement over passes {FROM}..={PASSES}: {m_off:.3} codes without the \
         temporal pass, {m_on:.3} with; the last frames differ by {diff:.3} codes mean, {worst} worst"
    );
    assert!(m_off > 0.0, "the still sequence did not move at all; nothing to measure");
    assert!(
        m_on < m_off * 0.6,
        "the second temporal pass took the flicker from {m_off:.3} only to {m_on:.3} codes"
    );
    assert!(
        diff < 1.5,
        "the temporal pass moved where the picture converges: {diff:.3} codes mean, {worst} worst"
    );
}

/// A pixel restarted among converged ones comes out blurry rather than
/// grainy.
///
/// Twenty-four passes converge the fixture; then a 16x16 box of the frame
/// is restarted through the keep mask, so its pixels have one sample each
/// and sit among pixels with twenty-four. With the fallback they get one
/// more wavelet iteration and a looser luminance stop than the budget gives
/// them, and the box is smoother than without.
#[test]
#[ignore = "requires GPU"]
fn a_fresh_pixel_is_blurry_rather_than_grainy() {
    let Some(ctx) = ctx_or_skip("a_fresh_pixel_is_blurry_rather_than_grainy") else {
        return;
    };
    let rig = Rig::new(ctx);
    const PASSES: u32 = 24;
    let (bx0, by0, bx1, by1) = (24u32, 20u32, 40u32, 36u32);
    let mut keep = vec![1u8; (W * H) as usize];
    for y in by0..by1 {
        for x in bx0..bx1 {
            keep[(y * W + x) as usize] = 0;
        }
    }

    let run = |denoise: &GpuDenoiseParams| {
        let mut res = rig.converge(PASSES, denoise);
        rig.pass(&mut res, PASSES + 1, &keep, denoise);
        let h = rig.read(&mut res);
        let px = rig.target.pixels(rig.ctx);
        (h, px)
    };
    let with = GpuDenoiseParams::default();
    let without = GpuDenoiseParams {
        fresh_extra_iters: 0,
        fresh_lum_relax: 1.0,
        ..with
    };
    let (h_with, px_with) = run(&with);
    let (_, px_without) = run(&without);

    // The box really did restart.
    let mut fresh = 0usize;
    for y in by0..by1 {
        for x in bx0..bx1 {
            if h_with.count[(y * W + x) as usize] == 1 {
                fresh += 1;
            }
        }
    }
    assert_eq!(fresh, ((bx1 - bx0) * (by1 - by0)) as usize, "the keep mask did not restart the box");

    // Mean 3x3 deviation inside the box, per channel, in codes.
    let roughness = |px: &[u8]| {
        let at = |xx: u32, yy: u32, c: usize| px[((yy * W + xx) * 4) as usize + c] as i32;
        let mut s = 0.0f64;
        let mut n = 0usize;
        for y in by0 + 1..by1 - 1 {
            for x in bx0 + 1..bx1 - 1 {
                for c in 0..3 {
                    let mut acc = 0;
                    for dy in 0..3 {
                        for dx in 0..3 {
                            acc += at(x + dx - 1, y + dy - 1, c);
                        }
                    }
                    s += (at(x, y, c) - acc / 9).abs() as f64;
                    n += 1;
                }
            }
        }
        s / n as f64
    };
    let (r_with, r_without) = (roughness(&px_with), roughness(&px_without));
    println!("3x3 deviation inside the restarted box: {r_without:.3} codes without the fallback, {r_with:.3} with");
    assert!(
        r_with < r_without * 0.85,
        "the disocclusion fallback bought nothing: {r_with:.3} codes with it, {r_without:.3} without"
    );
}
