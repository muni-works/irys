//! The WGSL sample pattern is a port of `sampler.rs`. This holds it to that,
//! number for number: both patterns, over a grid of (pixel, frame, dimension),
//! on real hardware.
//!
//! Bit-exact, not approximately. Everything in the construction is integer
//! until the last two operations — a 24-bit value scaled by a power of two
//! and a fract of its sum with an 8-bit mask value — and those are exact in
//! f32 on any IEEE device, so a difference of one ulp is a bug in the port.
//!
//! `#[ignore]`d like the other GPU tests; run with
//! `cargo test -p kosm-render --features gpu --test gpu_sampler -- --ignored`.

#![cfg(all(feature = "gpu", not(target_arch = "wasm32")))]

use kosm_render::gpu::{GpuContext, shaders};
use kosm_render::sampler::SamplePattern;

const HARNESS: &str = r#"
@group(0) @binding(0) var<storage, read> keys: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec2<f32>>;

@compute @workgroup_size(64)
fn parity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&keys) {
        return;
    }
    let k = keys[i];
    out[i] = vec2<f32>(
        blue_noise_sample(k.xy, k.z, k.w, 0u),
        white_noise_sample(k.xy, k.z, k.w, 0u),
    );
}
"#;

fn run(ctx: &GpuContext, keys: &[[u32; 4]]) -> Vec<[f32; 2]> {
    use wgpu::util::DeviceExt;
    let device = &ctx.device;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("sampler parity"),
        source: wgpu::ShaderSource::Wgsl(
            format!("{}\n{HARNESS}", shaders::sampler_shader()).into(),
        ),
    });
    let out_size = (keys.len() * 8) as u64;
    let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("keys"),
        contents: bytemuck::cast_slice(keys),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("read"),
        size: out_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[entry(0, true), entry(1, false)],
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("parity"),
        compilation_options: Default::default(),
        cache: None,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups((keys.len() as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, out_size);
    ctx.queue.submit(Some(encoder.finish()));
    let slice = read_buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let data = slice.get_mapped_range().expect("the readback maps");
    let out: Vec<[f32; 2]> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    read_buf.unmap();
    out
}

#[test]
#[ignore = "needs a GPU adapter"]
fn gpu_sampler_matches_the_cpu_sampler_bit_for_bit() {
    let ctx = match pollster::block_on(GpuContext::init()) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("skipped: {e}");
            return;
        }
    };
    // Pixels across and beyond the mask tile, frames across the low bits
    // and high, dimensions across the first bounces and the ReSTIR salts.
    let mut keys = Vec::new();
    for &frame in &[0u32, 1, 2, 3, 7, 8, 15, 16, 63, 64, 255, 1000, 65537] {
        for &dim in &[
            0u32, 1, 2, 3, 4, 5, 17, 194, 195, 202, 203, 1222, 30011, 32769,
        ] {
            for py in (0..300).step_by(13) {
                for px in (0..300).step_by(11) {
                    keys.push([px, py, frame, dim]);
                }
            }
        }
    }
    let gpu = run(&ctx, &keys);
    let mut mismatches = 0;
    for (k, g) in keys.iter().zip(&gpu) {
        let blue = SamplePattern::BlueNoise.sample([k[0], k[1]], k[2], k[3]);
        let white = SamplePattern::White.sample([k[0], k[1]], k[2], k[3]);
        if g[0].to_bits() != blue.to_bits() || g[1].to_bits() != white.to_bits() {
            if mismatches < 10 {
                eprintln!("key {k:?}: gpu {g:?}, cpu ({blue}, {white})");
            }
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "of {} keys", keys.len());
}
