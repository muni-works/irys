//! The learned denoiser's WGSL against its Rust.
//!
//! [`kosm_render::neural::Weights::forward`] is the definition of the network
//! and `neural.wgsl` is the thing that actually runs, and the only way the
//! second stays the first is a test that evaluates both on the same numbers.
//! It is worth more here than for most shaders: a network whose features are
//! a hair off, or whose weight indexing transposes an axis, does not crash
//! and does not obviously misbehave. It produces a slightly wrong picture
//! everywhere, forever.
//!
//! The fixture is synthetic on purpose — random weights, a made-up frame with
//! a depth discontinuity and a background band through it. Trained weights
//! would test the training as much as the shader, and random ones exercise
//! every path the softmax has.
//!
//! Run with `--features gpu`.
#![cfg(all(feature = "gpu", not(target_arch = "wasm32")))]

use kosm_render::gpu::wgpu;
use kosm_render::gpu::{GpuContext, NeuralDenoiser, NeuralPipeline};
use kosm_render::neural::{C_IN, K, KS, Weights};

const W: u32 = 24;
const H: u32 = 16;

fn ctx_or_skip(name: &str) -> Option<&'static GpuContext> {
    match GpuContext::init_blocking() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("skipping {name}: no GPU ({e})");
            None
        }
    }
}

/// Weights with the shape header a trainer writes and arbitrary values.
fn synthetic(hidden: usize, seed: u64) -> Weights {
    let n = (hidden * C_IN * KS * KS + hidden)
        + (hidden * hidden * KS * KS + hidden)
        + (K * hidden * KS * KS + K);
    let mut s = seed | 1;
    let mut b = b"KOSMKPN1".to_vec();
    for v in [C_IN as u32, hidden as u32, K as u32, KS as u32] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    for _ in 0..n {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // small, so three layers of ReLU neither die nor blow up
        let x = ((s >> 40) as f32 / 8_388_608.0 - 0.5) * 0.6;
        b.extend_from_slice(&x.to_le_bytes());
    }
    Weights::from_bytes(&b).expect("synthetic blob is well formed")
}

/// One frame's worth of history and guides: a lit gradient over two depths
/// with a background band, and per-pixel history lengths from 1 to 12 so the
/// count feature is exercised rather than constant.
struct Frame {
    mean: Vec<f32>,
    variance: Vec<f32>,
    normal: Vec<f32>,
    depth: Vec<f32>,
    albedo: Vec<f32>,
    count: Vec<f32>,
    id: Vec<f32>,
}

impl Frame {
    fn new() -> Self {
        let n = (W * H) as usize;
        let mut f = Self {
            mean: vec![0.0; n * 3],
            variance: vec![0.0; n],
            normal: vec![0.0; n * 3],
            depth: vec![0.0; n],
            albedo: vec![0.0; n * 3],
            count: vec![0.0; n],
            id: vec![0.0; n],
        };
        for y in 0..H as usize {
            for x in 0..W as usize {
                let p = y * W as usize + x;
                let near = x < W as usize / 2;
                // a band of background across the middle, so the shader's
                // depth <= 0 passthrough is on the tested path
                let bg = y >= 7 && y < 9;
                f.depth[p] = if bg {
                    0.0
                } else if near {
                    900.0
                } else {
                    4200.0
                };
                let nz = if near { 1.0 } else { 0.3 };
                f.normal[p * 3] = if near { 0.0 } else { 0.95 };
                f.normal[p * 3 + 2] = nz;
                for c in 0..3 {
                    f.albedo[p * 3 + c] = 0.15 + 0.25 * ((p + c * 5) % 7) as f32 / 7.0;
                    f.mean[p * 3 + c] =
                        0.05 + 1.6 * ((p * 13 + c * 29) % 31) as f32 / 31.0 + if bg { 3.0 } else { 0.0 };
                }
                f.variance[p] = 0.002 + 0.05 * ((p * 7) % 11) as f32 / 11.0;
                f.count[p] = 1.0 + ((p * 3) % 12) as f32;
                // Several materials, with the background at the zero
                // sentinel, so the id feature is a seam rather than a
                // constant both implementations agree about by accident.
                f.id[p] = if bg { 0.0 } else { (1 + (p * 5) % 4) as f32 };
            }
        }
        f
    }

    /// The scene's guide buffer: three `vec4` planes, plane 1 (normal, depth)
    /// and plane 2 (albedo, biased id).
    fn guides(&self) -> Vec<f32> {
        let n = (W * H) as usize;
        let mut v = vec![0.0f32; n * 4 * 3];
        for p in 0..n {
            let g1 = (n + p) * 4;
            v[g1] = self.normal[p * 3];
            v[g1 + 1] = self.normal[p * 3 + 1];
            v[g1 + 2] = self.normal[p * 3 + 2];
            v[g1 + 3] = self.depth[p];
            let g2 = (2 * n + p) * 4;
            v[g2] = self.albedo[p * 3];
            v[g2 + 1] = self.albedo[p * 3 + 1];
            v[g2 + 2] = self.albedo[p * 3 + 2];
            v[g2 + 3] = self.id[p];
        }
        v
    }

    /// (radiance, coverage).
    fn mean_buffer(&self) -> Vec<f32> {
        let n = (W * H) as usize;
        let mut v = vec![0.0f32; n * 4];
        for p in 0..n {
            v[p * 4] = self.mean[p * 3];
            v[p * 4 + 1] = self.mean[p * 3 + 1];
            v[p * 4 + 2] = self.mean[p * 3 + 2];
            v[p * 4 + 3] = 1.0;
        }
        v
    }

    /// (count, Σl, Σl², variance of the mean).
    fn stats_buffer(&self) -> Vec<f32> {
        let n = (W * H) as usize;
        let mut v = vec![0.0f32; n * 4];
        for p in 0..n {
            v[p * 4] = self.count[p];
            v[p * 4 + 3] = self.variance[p];
        }
        v
    }
}

fn storage(ctx: &GpuContext, label: &str, data: &[f32], readable: bool) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | if readable {
                    wgpu::BufferUsages::COPY_SRC
                } else {
                    wgpu::BufferUsages::empty()
                },
        })
}

fn read_back(ctx: &GpuContext, src: &wgpu::Buffer, len: usize) -> Vec<f32> {
    let size = (len * 4) as u64;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("neural parity readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
    ctx.queue.submit(Some(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let view = slice.get_mapped_range().expect("the staging buffer maps");
    let out = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
    drop(view);
    staging.unmap();
    out
}

/// The shader and the reference forward, on the same frame and the same
/// weights.
///
/// Every input plane varies across the frame — the history length pixel by
/// pixel, the hit id pixel by pixel, a band of background through the middle
/// — because a feature that is constant over the test frame is a feature the
/// test cannot tell is wired to the wrong index. Both sides now take the
/// count and the id per pixel, as the device always had them, so this is one
/// call and not one call per distinct count.
#[test]
fn the_shader_matches_the_reference_forward() {
    let Some(ctx) = ctx_or_skip("gpu_neural") else {
        return;
    };
    let n = (W * H) as usize;
    let frame = Frame::new();
    let weights = synthetic(8, 20250904);

    let pipeline = NeuralPipeline::new(ctx).expect("the neural shader compiles");
    let mut denoiser = NeuralDenoiser::new(ctx, &weights, W, H);
    // No pass-through: the point is to compare the network everywhere it can
    // run, and counts here go to 12.
    denoiser.set_count_cutoff(1_000_000);

    let guides = storage(ctx, "guides", &frame.guides(), false);
    let mean = storage(ctx, "mean", &frame.mean_buffer(), false);
    let stats = storage(ctx, "stats", &frame.stats_buffer(), false);
    let scratch = storage(ctx, "scratch", &vec![0.0f32; n * 4], true);

    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    denoiser.record(ctx, &pipeline, &mut enc, &guides, &mean, &stats, &scratch);
    ctx.queue.submit(Some(enc.finish()));
    let got = read_back(ctx, &scratch, n * 4);

    let want = weights.forward(
        W as usize,
        H as usize,
        &frame.mean,
        &frame.variance,
        &frame.normal,
        &frame.depth,
        &frame.albedo,
        &frame.id,
        &frame.count,
    );

    // The reference produces remodulated radiance; the shader leaves
    // demodulated illumination in the scratch for `resolve` to remodulate.
    let mut compared = 0usize;
    let mut worst = 0.0f32;
    for p in 0..n {
        if frame.depth[p] <= 0.0 {
            continue;
        }
        for ch in 0..3 {
            let a = frame.albedo[p * 3 + ch].max(kosm_render::neural::DEMOD_FLOOR);
            let mine = got[p * 4 + ch] * a;
            let theirs = want[p * 3 + ch];
            let rel = (mine - theirs).abs() / theirs.abs().max(1e-3);
            worst = worst.max(rel);
            assert!(
                rel < 2e-3,
                "pixel {p} channel {ch}: shader {mine}, reference {theirs}"
            );
        }
        compared += 1;
    }
    assert!(compared > n / 2, "only {compared} of {n} pixels compared");
    eprintln!("neural parity: {compared} pixels, worst relative error {worst:.2e}");
}

/// A pixel with no geometry behind it, and one whose history is past the
/// cutoff, are both passed through rather than filtered.
#[test]
fn background_and_converged_pixels_are_passed_through() {
    let Some(ctx) = ctx_or_skip("gpu_neural passthrough") else {
        return;
    };
    let n = (W * H) as usize;
    let frame = Frame::new();
    let weights = synthetic(6, 7);
    let pipeline = NeuralPipeline::new(ctx).expect("the neural shader compiles");
    let mut denoiser = NeuralDenoiser::new(ctx, &weights, W, H);
    // every pixel here has a history of at least 1, so a cutoff of 1 passes
    // the whole frame through
    denoiser.set_count_cutoff(1);

    let guides = storage(ctx, "guides", &frame.guides(), false);
    let mean = storage(ctx, "mean", &frame.mean_buffer(), false);
    let stats = storage(ctx, "stats", &frame.stats_buffer(), false);
    let scratch = storage(ctx, "scratch", &vec![0.0f32; n * 4], true);

    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    denoiser.record(ctx, &pipeline, &mut enc, &guides, &mean, &stats, &scratch);
    ctx.queue.submit(Some(enc.finish()));
    let got = read_back(ctx, &scratch, n * 4);

    for p in 0..n {
        for ch in 0..3 {
            let want = frame.mean[p * 3 + ch]
                / frame.albedo[p * 3 + ch].max(kosm_render::neural::DEMOD_FLOOR);
            assert!(
                (got[p * 4 + ch] - want).abs() <= want.abs() * 1e-4 + 1e-5,
                "pixel {p} channel {ch}: {} is not the unfiltered {want}",
                got[p * 4 + ch]
            );
        }
    }
}
