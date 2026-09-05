//! ReSTIR DI on the device.
//!
//! Bitterli et al. 2020. The claims, in order: that resampling does not move
//! the answer (the temporal path with the paper's M clamp is unbiased in
//! expectation, and the spatial pass's bias is small and measured), that it
//! buys a large variance reduction at one sample a frame, and that a light
//! that moves does not leave its old reservoirs lying around.
//!
//! Run with `--features gpu -- --ignored --test-threads=1`.
#![cfg(all(feature = "gpu", not(target_arch = "wasm32")))]

use kosm_render::gpu::{
    AnalyticGeometry, AnalyticPrim, GpuAreaLight, GpuCamera, GpuContext, GpuMaterial,
    GpuRenderState, RayTracePipeline, ResidentScene, SceneRef,
};
use kosm_render::pathtrace::{AreaLight, Pbr};
use kosm_render::{Point3, Vec3};

/// Path length these tests trace. Three bounces of global illumination.
const DEPTH: u32 = 3;

/// The configuration these tests treat as the shipped one: sixteen candidates,
/// one round of spatial reuse, and a *small* radius. Wide reuse measured
/// worse than narrow at every candidate count on this room — 2.9x quieter at
/// 4 px, 2.7x at 16 px — because a neighbour four pixels away is looking at
/// very nearly the same integral and one thirty pixels away is not.
const SHIPPED: Option<(u32, u32, f32)> = Some((16, 1, 4.0));

fn ctx_or_skip(name: &str) -> Option<&'static GpuContext> {
    match GpuContext::init_blocking() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping {name}: no GPU ({e})");
            None
        }
    }
}

/// A closed room: six planes facing in, a sphere on the floor, and three
/// ceiling panels. Closed so nothing escapes to the environment and every
/// photon in the picture came off a panel — which is what makes the direct
/// term the whole of the interesting variance.
struct Room {
    geometry: AnalyticGeometry,
    materials: Vec<GpuMaterial>,
    lights: Vec<GpuAreaLight>,
}

fn panel(center: [f64; 3], half: f64, emission: f32) -> AreaLight {
    AreaLight {
        center: Point3::new(center[0], center[1], center[2]),
        // Normal is u x v; with u = +x and v = +y that is +z, so flip v to
        // aim the emitting face down into the room.
        u: Vec3::new(half, 0.0, 0.0),
        v: Vec3::new(0.0, -half, 0.0),
        emission: [emission, emission * 0.97, emission * 0.9],
    }
}

/// Ten ceiling panels of widely different power, which is the rig ReSTIR
/// exists for: one uniformly-drawn light sample a pixel has to guess which of
/// ten matters here, and resampling does not have to guess.
fn ceiling_rig(shift: f64) -> Vec<AreaLight> {
    let mut out = Vec::new();
    for i in 0..24usize {
        let col = (i % 6) as f64;
        let row = (i / 6) as f64;
        let x = -2.5 + col * 1.0 + shift;
        let y = -3.2 + row * 1.7;
        // Two orders of magnitude between the dimmest panel and the
        // brightest, scattered rather than graded, so which panel matters at
        // a shading point is a fact about that point and not about the rig.
        let e = 1.5 * f32::powf(1.28, ((i * 11) % 24) as f32);
        out.push(panel([x, y, 3.9], 0.22, e));
    }
    out
}

impl Room {
    fn new(panel_shift: f64) -> Self {
        let geometry = AnalyticGeometry {
            prims: vec![
                AnalyticPrim::plane([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0),
                AnalyticPrim::plane([0.0, 0.0, 4.0], [0.0, 0.0, -1.0], 1),
                AnalyticPrim::plane([-3.0, 0.0, 0.0], [1.0, 0.0, 0.0], 2),
                AnalyticPrim::plane([3.0, 0.0, 0.0], [-1.0, 0.0, 0.0], 2),
                AnalyticPrim::plane([0.0, 3.0, 0.0], [0.0, -1.0, 0.0], 2),
                AnalyticPrim::plane([0.0, -4.0, 0.0], [0.0, 1.0, 0.0], 2),
                AnalyticPrim::sphere([0.0, 0.5, 1.0], 1.0, 3),
            ],
        };
        let mat = |c: [f32; 3], r: f32| {
            GpuMaterial::from_pbr(Pbr {
                base_color: c,
                roughness: r,
                ..Default::default()
            })
        };
        Self {
            geometry,
            materials: vec![
                mat([0.6, 0.58, 0.55], 0.85),
                mat([0.75, 0.75, 0.75], 0.9),
                mat([0.55, 0.56, 0.6], 0.9),
                mat([0.8, 0.75, 0.7], 0.3),
            ],
            lights: ceiling_rig(panel_shift)
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

fn camera(w: u32, h: u32) -> GpuCamera {
    GpuCamera::new(
        [0.0, -3.6, 1.9],
        [0.0, 0.6, 1.2],
        [0.0, 0.0, 1.0],
        0.9,
        w,
        h,
    )
}

/// The render state both tiers share; `restir` is the only difference.
fn state(frame: u32, restir: Option<(u32, u32, f32)>, depth: u32) -> GpuRenderState {
    let mut s = GpuRenderState::new(frame);
    s.enable_edges = 0;
    s.stylize = 0;
    s.ground_enabled = 0;
    s.max_depth = depth;
    s.rr_start = 8;
    s.firefly_clamp = 0.0;
    if let Some((m, sp, r)) = restir {
        s.set_restir(m, sp, r);
    }
    s
}

/// Sum `frames` independent linear passes into a per-channel mean.
fn accumulate(
    pipeline: &RayTracePipeline,
    ctx: &GpuContext,
    res: &mut ResidentScene,
    cam: &GpuCamera,
    frames: u32,
    restir: Option<(u32, u32, f32)>,
    depth: u32,
) -> Vec<f64> {
    let mut sum: Vec<f64> = Vec::new();
    for f in 1..=frames {
        let film = pollster::block_on(pipeline.render_resident_linear(
            ctx,
            res,
            cam,
            state(f, restir, depth),
        ))
        .expect("linear pass");
        if sum.is_empty() {
            sum = vec![0.0; film.rgb.len()];
        }
        for (a, b) in sum.iter_mut().zip(film.rgb.iter()) {
            *a += *b as f64;
        }
    }
    for v in sum.iter_mut() {
        *v /= frames as f64;
    }
    sum
}

fn one_frame(
    pipeline: &RayTracePipeline,
    ctx: &GpuContext,
    res: &mut ResidentScene,
    cam: &GpuCamera,
    frame: u32,
    restir: Option<(u32, u32, f32)>,
    depth: u32,
) -> Vec<f32> {
    pollster::block_on(pipeline.render_resident_linear(ctx, res, cam, state(frame, restir, depth)))
        .expect("linear pass")
        .rgb
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Root-mean-square difference of a single frame from a converged reference —
/// the noise the frame carries, in linear radiance.
fn rms_vs(frame: &[f32], reference: &[f64]) -> f64 {
    let n = frame.len().min(reference.len());
    let s: f64 = (0..n)
        .map(|i| {
            let d = frame[i] as f64 - reference[i];
            d * d
        })
        .sum();
    (s / n as f64).sqrt()
}

fn rel_diff(a: &[f64], b: &[f64]) -> f64 {
    (mean(a) - mean(b)).abs() / mean(b)
}

#[test]
#[ignore = "requires GPU"]
fn restir_and_plain_nee_converge_to_the_same_picture() {
    let Some(ctx) = ctx_or_skip("restir_and_plain_nee_converge_to_the_same_picture") else {
        return;
    };
    let (w, h) = (128u32, 96u32);
    let room = Room::new(0.0);
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let cam = camera(w, h);
    let frames = 160;

    let mut plain = pipeline.resident_scene(ctx, room.scene(), w, h);
    let reference = accumulate(&pipeline, ctx, &mut plain, &cam, frames, None, DEPTH);

    // Temporal only. This is the path the paper proves unbiased in
    // expectation; the M clamp bounds how long a reservoir may claim to be,
    // not what it estimates.
    let mut temporal = pipeline.resident_scene(ctx, room.scene(), w, h);
    let t = accumulate(
        &pipeline,
        ctx,
        &mut temporal,
        &cam,
        frames,
        Some((16, 0, 0.0)),
        DEPTH,
    );
    let d_temporal = rel_diff(&t, &reference);

    // With spatial reuse (the shipped configuration). Biased on purpose — the neighbour's sample is
    // re-weighted against this pixel's p̂ with no MIS weights and no
    // visibility ray of its own — and this is the number that says by how
    // much.
    let mut spatial = pipeline.resident_scene(ctx, room.scene(), w, h);
    let s = accumulate(&pipeline, ctx, &mut spatial, &cam, frames, SHIPPED, DEPTH);
    let d_spatial = rel_diff(&s, &reference);

    eprintln!(
        "mean radiance: plain {:.5}, restir temporal {:.5} ({:+.2}%), \
         restir + spatial {:.5} ({:+.2}%)",
        mean(&reference),
        mean(&t),
        100.0 * (mean(&t) - mean(&reference)) / mean(&reference),
        mean(&s),
        100.0 * (mean(&s) - mean(&reference)) / mean(&reference),
    );

    assert!(
        d_temporal < 0.02,
        "temporal-only ReSTIR moved the mean by {:.2}%, which is more than the \
         2% an unbiased estimator is allowed",
        100.0 * d_temporal
    );
    // The spatial pass is the biased combination. It is allowed to move the
    // answer; it is not allowed to move it far.
    assert!(
        d_spatial < 0.05,
        "spatial reuse moved the mean by {:.2}%",
        100.0 * d_spatial
    );
}

#[test]
#[ignore = "requires GPU"]
fn restir_is_much_quieter_at_one_sample_a_frame() {
    let Some(ctx) = ctx_or_skip("restir_is_much_quieter_at_one_sample_a_frame") else {
        return;
    };
    let (w, h) = (128u32, 96u32);
    let room = Room::new(0.0);
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let cam = camera(w, h);
    let cfg = SHIPPED;

    // Measured twice: once over the direct term alone, which is all ReSTIR
    // touches and where the whole of its variance reduction shows, and once
    // over the three-bounce path the viewport actually traces, where the
    // indirect noise ReSTIR does nothing about is most of what is left.
    let mut ratios = Vec::new();
    for depth in [1u32, DEPTH] {
        let mut conv = pipeline.resident_scene(ctx, room.scene(), w, h);
        let reference = accumulate(&pipeline, ctx, &mut conv, &cam, 192, None, depth);

        // Eight frames of each, then look at the eighth — one sample per
        // pixel, no host-side history, the frame a viewport would put on the
        // glass.
        let mut plain = pipeline.resident_scene(ctx, room.scene(), w, h);
        let mut plain_frame = Vec::new();
        for f in 1..=8u32 {
            plain_frame = one_frame(&pipeline, ctx, &mut plain, &cam, f, None, depth);
        }
        let mut rst = pipeline.resident_scene(ctx, room.scene(), w, h);
        let mut restir_frame = Vec::new();
        for f in 1..=8u32 {
            restir_frame = one_frame(&pipeline, ctx, &mut rst, &cam, f, cfg, depth);
        }

        let plain_rms = rms_vs(&plain_frame, &reference);
        let restir_rms = rms_vs(&restir_frame, &reference);
        let ratio = plain_rms / restir_rms;
        eprintln!(
            "max_depth {depth}: 1 spp after 8 frames, rms from the converged \
             reference — plain NEE {plain_rms:.5}, ReSTIR (M=16, 1 spatial) \
             {restir_rms:.5}, {ratio:.2}x quieter"
        );
        ratios.push(ratio);
    }

    assert!(
        ratios[0] > 2.5,
        "over the direct term ReSTIR was only {:.2}x quieter than plain NEE",
        ratios[0]
    );
    assert!(
        ratios[1] > 1.2,
        "over the full path ReSTIR was only {:.2}x quieter than plain NEE",
        ratios[1]
    );
}

#[test]
#[ignore = "requires GPU"]
fn a_moved_panel_does_not_leave_stale_reservoirs() {
    let Some(ctx) = ctx_or_skip("a_moved_panel_does_not_leave_stale_reservoirs") else {
        return;
    };
    let (w, h) = (96u32, 72u32);
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let cam = camera(w, h);
    let before = Room::new(0.0);
    let after = Room::new(2.4);
    let restir = SHIPPED;

    // What the moved rig looks like with no history at all to be stale: a
    // scene that has only ever seen the panels where they now are.
    let mut fresh = pipeline.resident_scene(ctx, after.scene(), w, h);
    let mut fresh_frames = Vec::new();
    for f in 1..=6u32 {
        fresh_frames.push(one_frame(
            &pipeline, ctx, &mut fresh, &cam, f, restir, DEPTH,
        ));
    }

    // Eight frames of settled reservoirs under the old rig, then move the
    // panels — and change nothing else. The accumulation is *not* reset: the
    // question is exactly whether the reservoirs notice on their own.
    let mut moved = pipeline.resident_scene(ctx, before.scene(), w, h);
    for f in 1..=8u32 {
        one_frame(&pipeline, ctx, &mut moved, &cam, f, restir, DEPTH);
    }
    moved.set_lights(ctx, &after.lights);

    let mut errs = Vec::new();
    for (i, f) in (9..=14u32).enumerate() {
        let frame = one_frame(&pipeline, ctx, &mut moved, &cam, f, restir, DEPTH);
        // Against the *same* frame index of the fresh run, so the comparison
        // is like for like in how much noise each still carries.
        let fresh_ref: Vec<f64> = fresh_frames[i.min(fresh_frames.len() - 1)]
            .iter()
            .map(|v| *v as f64)
            .collect();
        let m = mean(&fresh_ref);
        let mv = mean(&frame.iter().map(|v| *v as f64).collect::<Vec<_>>());
        errs.push((mv - m).abs() / m);
    }
    eprintln!(
        "frames after the panels moved, mean error against a from-scratch run: {:?}",
        errs.iter()
            .map(|e| (e * 100.0).round() / 100.0)
            .collect::<Vec<_>>()
    );
    // Three frames is "a few". By then a reservoir that still pointed at the
    // old panel would have had its p̂ re-evaluated against the new one three
    // times over, and the M clamp bounds how much of the old one is left.
    assert!(
        errs[0] < 0.08,
        "the very first frame after the move was {:.1}% off a from-scratch \
         render: the stale reservoirs were still being believed",
        errs[0] * 100.0
    );
    assert!(
        errs[2] < 0.05,
        "three frames after the move the picture was still {:.1}% off a \
         from-scratch render: reservoirs went stale",
        errs[2] * 100.0
    );
}

#[test]
#[ignore = "requires GPU"]
fn restir_pass_time_at_the_live_tier_size() {
    let Some(ctx) = ctx_or_skip("restir_pass_time_at_the_live_tier_size") else {
        return;
    };
    let (w, h) = (512u32, 288u32);
    let room = Room::new(0.0);
    let pipeline = RayTracePipeline::new(ctx, &AnalyticGeometry::module()).expect("pipeline");
    let cam = camera(w, h);

    let time = |label: &str, restir: Option<(u32, u32, f32)>| {
        let mut res = pipeline.resident_scene(ctx, room.scene(), w, h);
        for f in 1..=3u32 {
            one_frame(&pipeline, ctx, &mut res, &cam, f, restir, DEPTH);
        }
        let t0 = std::time::Instant::now();
        let n = 10u32;
        for f in 4..4 + n {
            one_frame(&pipeline, ctx, &mut res, &cam, f, restir, DEPTH);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        eprintln!("{label}: {ms:.2} ms/pass at {w}x{h}");
        ms
    };
    let plain = time("plain NEE", None);
    let one = time("ReSTIR M=16, 1 spatial", SHIPPED);
    let two = time("ReSTIR M=16, 2 spatial", Some((16, 2, 4.0)));
    eprintln!(
        "ReSTIR costs {:.2}x a plain pass at one spatial pass, {:.2}x at two",
        one / plain,
        two / plain
    );
}
