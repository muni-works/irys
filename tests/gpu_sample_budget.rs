//! Gradient-directed sampling: where the frame's rays go.
//!
//! A uniform spend gives the wall that has not changed in four hundred frames
//! exactly what it gives the ball that crossed it on this one. These tests pin
//! the four claims that replaced it:
//!
//! * the budget is a *budget* — with the scene static it lands on
//!   `rays_per_frame` to within a percent, and it lands on the pixels whose own
//!   error bars are largest;
//! * with one ball moving, the ball's footprint and the neighbourhood the
//!   filter will drag with it take the frame;
//! * at **equal samples folded per frame**, sixteen frames of a directed spend
//!   are closer to a converged reference than sixteen frames of a uniform one;
//! * and no pixel starves: over `floor_k` frames every pixel is sampled.
//!
//! Run with `--features gpu -- --ignored --test-threads=1 --nocapture`.
#![cfg(all(feature = "gpu", not(target_arch = "wasm32")))]

use kosm_render::Point3;
use kosm_render::gpu::wgpu;
use kosm_render::gpu::{
    AnalyticGeometry, AnalyticPrim, Budget, GpuAreaLight, GpuCamera, GpuContext, GpuDenoiseParams,
    GpuMaterial, GpuRenderState, HistoryPipeline, InstanceMotion, RayTracePipeline, SampleBudget,
    SceneRef,
};
use kosm_render::pathtrace::{self, Pbr};

const W: u32 = 64;
const H: u32 = 64;
const N: usize = (W * H) as usize;

/// The sphere's index in `Fixture::geometry`, and so the id the guide plane
/// carries (biased by one) and the motion table is keyed on.
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
            lights: pathtrace::studio_rig(Point3::new(0.0, 0.0, 1.0), 3.0)
                .iter()
                .map(GpuAreaLight::from_area_light)
                .collect(),
        }
    }

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
    GpuCamera::new(
        [5.0, -5.0, 3.5],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        0.7,
        W,
        H,
    )
}

fn state(frame: u32) -> GpuRenderState {
    let mut s = GpuRenderState::new(frame);
    s.enable_edges = 0;
    s.stylize = 0;
    s.ground_enabled = 0;
    s.set_camera_visible_lights(false);
    s
}

/// A storage texture the resolve pass can write. Nothing here reads it back —
/// every claim is about the history and the budget — but the denoise chain
/// needs somewhere to put the frame.
struct Target {
    view: wgpu::TextureView,
}

impl Target {
    fn new(ctx: &GpuContext) -> Self {
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("budget test target"),
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
        Self {
            view: texture.create_view(&Default::default()),
        }
    }
}

/// The sphere's silhouette at `cx`, worked out on the CPU from the same
/// analytic geometry the shader traces.
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

    let mut out = vec![false; N];
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

/// The instance motion for a sphere that just stepped `step` along +x, with
/// the plane static.
fn motion_for(step: f32) -> InstanceMotion {
    InstanceMotion::new(
        &[0, InstanceMotion::STATIC],
        &[[
            1.0, 0.0, 0.0, -step, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0,
        ]],
    )
}

/// One budgeted frame, end to end: budget, `rounds` trace rounds, denoise.
#[allow(clippy::too_many_arguments)]
fn frame(
    ctx: &'static GpuContext,
    pipeline: &RayTracePipeline,
    history: &HistoryPipeline,
    res: &mut kosm_render::gpu::ResidentScene,
    target: &Target,
    denoise: &GpuDenoiseParams,
    budget: &SampleBudget,
    motion: Option<&InstanceMotion>,
    f: u32,
) {
    pipeline
        .budget_frame(ctx, history, res, &camera(), budget, motion, f)
        .expect("budget");
    let cam = camera();
    for r in 0..budget.rounds.max(1) {
        pipeline
            .accumulate_resident_round(
                ctx,
                history,
                res,
                &camera(),
                // Each round is an independent sample of the same picture, so
                // each needs its own frame index: the jitter and the RNG ride
                // on it.
                state(f * budget.rounds.max(1) + r + 1),
                &[],
                if r == 0 { Some(&cam) } else { None },
                denoise,
                if r == 0 { motion } else { None },
                budget,
                r,
                f,
            )
            .expect("round");
    }
    pipeline
        .denoise_and_resolve_resident(ctx, history, res, denoise, &target.view)
        .expect("denoise");
}

fn mean_over(v: &[f32], mask: &dyn Fn(usize) -> bool) -> (f64, usize) {
    let mut s = 0.0f64;
    let mut k = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if mask(i) {
            s += x as f64;
            k += 1;
        }
    }
    (s / k.max(1) as f64, k)
}

/// With nothing moving, the budget is still a budget — it sums to what the
/// tuner asked for — and it goes to the pixels whose own error bars are
/// widest.
///
/// This is the claim that the *history* input works on its own, with the
/// physics contributing nothing: there is no motion table at all, so every
/// sample the budget moves is moved on the strength of sigma/sqrt(n) and the
/// last sample's disagreement.
#[test]
#[ignore = "requires GPU"]
fn a_static_frame_spends_its_budget_on_its_noisiest_pixels() {
    let Some(ctx) = ctx_or_skip("a_static_frame_spends_its_budget_on_its_noisiest_pixels") else {
        return;
    };
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let history = HistoryPipeline::new(ctx).expect("history pipeline");
    let target = Target::new(ctx);
    let denoise = GpuDenoiseParams::default();
    let fx = Fixture::new();
    let mut res = pipeline.resident_scene(ctx, fx.scene(), W, H);

    // A budget that never fires the starvation floor, so the total is the
    // weight field's alone and the test is measuring what it means to.
    let budget = SampleBudget {
        floor_k: 1_000_000,
        ..SampleBudget::directed(W, H, 1.0, 4)
    };

    // Enough frames that every pixel is past the short-history term and what
    // is left to separate them is their own error bars; the last frame leaves
    // the budget on the device.
    for f in 0..24 {
        frame(
            ctx, &pipeline, &history, &mut res, &target, &denoise, &budget, None, f,
        );
    }

    let b: Budget = pollster::block_on(pipeline.read_budget(ctx, &mut res))
        .expect("read")
        .expect("budget");
    let hist = pollster::block_on(pipeline.read_history(ctx, &mut res))
        .expect("read")
        .expect("history");

    let total = b.total();
    let want = budget.rays_per_frame as f64;
    println!("total budget {total:.0} samples against a target of {want:.0}");
    assert!(
        (total - want).abs() <= want * 0.01,
        "the frame's budget is {total:.0} samples, {:.2}% off the {want:.0} it was given",
        (total - want).abs() / want * 100.0,
    );

    // Split the frame by its own relative error bar and ask where the samples
    // went. `variance` is the variance of the mean, so its root over the mean
    // is the fraction the extra samples would buy down.
    let mut rel: Vec<(usize, f64)> = (0..N)
        .map(|i| {
            let lum = (0.2126 * hist.rgb[i * 3]
                + 0.7152 * hist.rgb[i * 3 + 1]
                + 0.0722 * hist.rgb[i * 3 + 2]) as f64;
            (i, (hist.variance[i].max(0.0) as f64).sqrt() / lum.max(1e-4))
        })
        .collect();
    rel.sort_by(|a, c| c.1.partial_cmp(&a.1).unwrap());
    let q = N / 4;
    let noisy: f64 = rel[..q]
        .iter()
        .map(|&(i, _)| b.samples[i] as f64)
        .sum::<f64>()
        / q as f64;
    let calm: f64 = rel[N - q..]
        .iter()
        .map(|&(i, _)| b.samples[i] as f64)
        .sum::<f64>()
        / q as f64;
    println!("noisiest quarter {noisy:.2} samples/pixel, calmest quarter {calm:.2}");
    assert!(
        noisy > calm * 1.5,
        "the budget is barely directed: {noisy:.2} samples on the noisiest quarter against \
         {calm:.2} on the calmest"
    );
}

/// One ball moving, and the frame's samples follow it — the ball's own
/// footprint first, the neighbourhood the filter will drag with it second, and
/// the far side of the frame last.
#[test]
#[ignore = "requires GPU"]
fn a_moving_ball_takes_the_frame() {
    let Some(ctx) = ctx_or_skip("a_moving_ball_takes_the_frame") else {
        return;
    };
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let history = HistoryPipeline::new(ctx).expect("history pipeline");
    let target = Target::new(ctx);
    let denoise = GpuDenoiseParams::default();
    let mut fx = Fixture::new();
    let mut res = pipeline.resident_scene(ctx, fx.scene(), W, H);

    let step = 0.25_f32;
    // A modest dilation, so "near the ball" and "far from it" are separable at
    // 64x64. The viewer's is the à-trous footprint, 32 pixels, which at this
    // size would be most of the frame.
    let budget = SampleBudget {
        radius: 6,
        floor_k: 1_000_000,
        ..SampleBudget::directed(W, H, 1.0, 4)
    };

    // Settle first with nothing moving, so the history term is quiet
    // everywhere and what is left is the physics.
    for f in 0..8 {
        frame(
            ctx, &pipeline, &history, &mut res, &target, &denoise, &budget, None, f,
        );
    }
    // Then one frame in which the ball moves.
    let f = 8;
    fx.move_sphere(step);
    res.update_scene(ctx, fx.scene());
    frame(
        ctx,
        &pipeline,
        &history,
        &mut res,
        &target,
        &denoise,
        &budget,
        Some(&motion_for(step)),
        f,
    );

    let b = pollster::block_on(pipeline.read_budget(ctx, &mut res))
        .expect("read")
        .expect("budget");

    // The ball's own pixels, as it stood when the budget was computed — the
    // budget runs *before* the trace, so it sees the previous frame's guides.
    let ball = sphere_mask(0.0);
    // Its neighbourhood: within the dilation radius of a ball pixel.
    let r = budget.radius as i32;
    let mut near = vec![false; N];
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            let mut hit = false;
            for dy in -r..=r {
                for dx in -r..=r {
                    let (qx, qy) = (x + dx, y + dy);
                    if qx >= 0
                        && qy >= 0
                        && qx < W as i32
                        && qy < H as i32
                        && ball[(qy * W as i32 + qx) as usize]
                    {
                        hit = true;
                    }
                }
            }
            near[(y * W as i32 + x) as usize] = hit && !ball[(y * W as i32 + x) as usize];
        }
    }

    let (on_ball, n_ball) = mean_over(&b.samples, &|i| ball[i]);
    let (halo, n_halo) = mean_over(&b.samples, &|i| near[i]);
    let (far, n_far) = mean_over(&b.samples, &|i| !ball[i] && !near[i]);
    println!(
        "budget: {on_ball:.2} samples/pixel over {n_ball} ball pixels, {halo:.2} over {n_halo} \
         halo pixels, {far:.2} over {n_far} elsewhere"
    );

    assert!(
        n_ball > 100 && n_halo > 100 && n_far > 100,
        "degenerate split"
    );
    assert!(
        on_ball > far * 3.0,
        "the moving ball got {on_ball:.2} samples/pixel against {far:.2} for the static \
         background — the physics bought nothing"
    );
    assert!(
        halo > far * 2.0,
        "the ball's neighbourhood got {halo:.2} against {far:.2}: the dilation is not \
         reaching the shadow and the filter footprint"
    );
    let (motion_on, _) = mean_over(&b.motion, &|i| ball[i]);
    assert!(
        motion_on > 0.5,
        "the motion drive over the ball is only {motion_on:.2}: the screen-space \
         displacement is not being read out of the motion table"
    );
}

/// The claim the whole thing is for: at **equal samples folded per frame**,
/// sixteen frames of a directed spend land closer to the converged picture
/// than sixteen frames of a uniform one.
///
/// Both arms fold `W * H` samples a frame. The uniform arm puts one on every
/// pixel; the directed arm puts four on the ball and its wake and none on the
/// wall it has already resolved.
#[test]
#[ignore = "requires GPU"]
fn a_directed_spend_beats_a_uniform_one_at_equal_samples() {
    let Some(ctx) = ctx_or_skip("a_directed_spend_beats_a_uniform_one_at_equal_samples") else {
        return;
    };
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let history = HistoryPipeline::new(ctx).expect("history pipeline");
    let target = Target::new(ctx);
    let denoise = GpuDenoiseParams::default();

    const FRAMES: u32 = 16;
    let step = 0.06_f32;

    // Sixteen frames of a ball crossing the plane, under one budget — from a
    // frame that has already converged.
    //
    // The settle is the point, not a convenience. Directing samples is worth
    // nothing on a frame where every pixel is equally unresolved: there is
    // nowhere to take them from. What it is *for* is the live case — a room
    // that has been standing still for a hundred frames and one ball crossing
    // it — so that is what both arms are given, identically, before the
    // budgets diverge.
    const SETTLE: u32 = 64;
    let run = |budget: SampleBudget| {
        let mut fx = Fixture::new();
        let mut res = pipeline.resident_scene(ctx, fx.scene(), W, H);
        for f in 0..SETTLE {
            pipeline
                .accumulate_and_denoise_resident(
                    ctx,
                    &history,
                    &mut res,
                    &camera(),
                    state(f + 1),
                    &[],
                    &denoise,
                    &target.view,
                )
                .expect("settle pass");
        }
        for f in 0..FRAMES {
            fx.move_sphere(f as f32 * step);
            res.update_scene(ctx, fx.scene());
            frame(
                ctx,
                &pipeline,
                &history,
                &mut res,
                &target,
                &denoise,
                &budget,
                Some(&motion_for(step)),
                f,
            );
        }
        pollster::block_on(pipeline.read_history(ctx, &mut res))
            .expect("read")
            .expect("history")
    };

    // The reference: the last frame's pose, held still and converged with a
    // long uniform spend. Both arms are trying to reach this.
    let reference = {
        let mut fx = Fixture::new();
        fx.move_sphere((FRAMES - 1) as f32 * step);
        let mut res = pipeline.resident_scene(ctx, fx.scene(), W, H);
        let long = GpuDenoiseParams {
            history_cap: 4096,
            clamp_k: 0.0,
            ..denoise
        };
        for f in 0..512 {
            pipeline
                .accumulate_and_denoise_resident(
                    ctx,
                    &history,
                    &mut res,
                    &camera(),
                    state(f + 1),
                    &[],
                    &long,
                    &target.view,
                )
                .expect("reference pass");
        }
        pollster::block_on(pipeline.read_history(ctx, &mut res))
            .expect("read")
            .expect("history")
    };

    let uniform = run(SampleBudget::uniform(W, H));
    let directed = run(SampleBudget {
        radius: 8,
        floor_k: 8,
        ..SampleBudget::directed(W, H, 1.0, 4)
    });

    let rmse = |h: &kosm_render::gpu::History| {
        let mut s = 0.0f64;
        for k in 0..N * 3 {
            let d = (h.rgb[k] - reference.rgb[k]) as f64;
            s += d * d;
        }
        (s / (N * 3) as f64).sqrt()
    };
    let ball = sphere_mask((FRAMES - 1) as f32 * step);
    let rmse_over = |h: &kosm_render::gpu::History, m: &dyn Fn(usize) -> bool| {
        let mut s = 0.0f64;
        let mut k = 0usize;
        for i in 0..N {
            if m(i) {
                for c in 0..3 {
                    let d = (h.rgb[i * 3 + c] - reference.rgb[i * 3 + c]) as f64;
                    s += d * d;
                }
                k += 3;
            }
        }
        (s / k.max(1) as f64).sqrt()
    };
    println!(
        "  ball: uniform {:.5} directed {:.5}; elsewhere: uniform {:.5} directed {:.5}",
        rmse_over(&uniform, &|i| ball[i]),
        rmse_over(&directed, &|i| ball[i]),
        rmse_over(&uniform, &|i| !ball[i]),
        rmse_over(&directed, &|i| !ball[i]),
    );
    let (ru, rd) = (rmse(&uniform), rmse(&directed));
    println!(
        "RMSE against the converged reference after {FRAMES} frames at {N} samples/frame: \
         uniform {ru:.5}, directed {rd:.5} — a ratio of {:.3}",
        rd / ru
    );
    assert!(
        rd < ru,
        "directing the samples made it worse: {rd:.5} against {ru:.5}"
    );
}

/// No pixel starves.
///
/// A budget that can hand a pixel nothing can hand it nothing forever, and a
/// wall that stopped being a wall would never be noticed. `floor_k` is the
/// hard guarantee: every pixel is sampled at least once in every `floor_k`
/// frames, on a phase of its own so the cost is spread rather than periodic.
#[test]
#[ignore = "requires GPU"]
fn every_pixel_is_sampled_within_the_floor_period() {
    let Some(ctx) = ctx_or_skip("every_pixel_is_sampled_within_the_floor_period") else {
        return;
    };
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let history = HistoryPipeline::new(ctx).expect("history pipeline");
    let target = Target::new(ctx);
    let denoise = GpuDenoiseParams::default();
    let fx = Fixture::new();
    let mut res = pipeline.resident_scene(ctx, fx.scene(), W, H);

    const K: u32 = 8;
    let budget = SampleBudget {
        floor_k: K,
        ..SampleBudget::directed(W, H, 1.0, 4)
    };

    // Settle, so the budget has every reason to abandon most of the frame.
    for f in 0..16 {
        frame(
            ctx, &pipeline, &history, &mut res, &target, &denoise, &budget, None, f,
        );
    }
    let before = pollster::block_on(pipeline.read_history(ctx, &mut res))
        .expect("read")
        .expect("history");

    // One floor period. The floor's phase is `(frame + pixel) % k`, so every
    // pixel's turn comes up exactly once in k frames.
    for f in 16..16 + K {
        frame(
            ctx, &pipeline, &history, &mut res, &target, &denoise, &budget, None, f,
        );
    }
    let after = pollster::block_on(pipeline.read_history(ctx, &mut res))
        .expect("read")
        .expect("history");

    // The count is capped, so a pixel already at the cap cannot show a rise.
    // What it can show is that it was folded into: the mean moved.
    let cap = denoise.history_cap;
    let mut starved = 0usize;
    for i in 0..N {
        let grew = after.count[i] > before.count[i] || before.count[i] >= cap;
        let moved = (0..3).any(|c| after.rgb[i * 3 + c] != before.rgb[i * 3 + c]);
        if !grew && !moved {
            starved += 1;
        }
    }
    println!("{starved} of {N} pixels went untouched over {K} frames");
    assert_eq!(
        starved, 0,
        "{starved} pixels were never sampled in a whole floor period of {K} frames"
    );
}
