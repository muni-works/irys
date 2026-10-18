//! The look's graphs on the device: [`TensorGraph::to_wgsl`]'s kernels run
//! once over host-written inputs and read back. This is how the OpenPBR
//! stack, the volume event and the hair BCSDF are held against their f64
//! twins on the GPU (the integrators themselves are CPU f64; design.md,
//! Phase 4, says why). Additive: no change to `gpu.rs` or its shaders.

use irys_gpu::GpuContext;
use irys_gpu::wgpu::{self, util::DeviceExt};
use tang_compile::tensor::{Format, TensorGraph, TensorId};

/// Run `g`'s kernels for `outputs` once over `inputs` (graph input order,
/// f32 on the device) and return each output.
pub fn run(ctx: &GpuContext, g: &TensorGraph, outputs: &[TensorId], inputs: &[Vec<f64>]) -> Vec<Vec<f32>> {
    let dev = &ctx.device;
    let prog = g.to_wgsl(outputs);
    let mk = |data: &[f32]| {
        dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("look ir"),
            contents: bytemuck::cast_slice(if data.is_empty() { &[0.0f32] } else { data }),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
    };
    let bufs: Vec<(TensorId, wgpu::Buffer, usize)> = prog
        .buffers
        .iter()
        .map(|b| {
            assert_eq!(b.format, Format::F32, "f32 only");
            let data: Vec<f32> = match b.input {
                Some(k) => inputs[k].iter().map(|&v| v as f32).collect(),
                None => vec![0.0; b.len.max(1)],
            };
            (b.node, mk(&data), b.len)
        })
        .collect();
    let find = |t: TensorId| {
        let t = prog.storage(t);
        bufs.iter().find(|(n, _, _)| *n == t).expect("buffer")
    };
    let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        for k in &prog.kernels {
            let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("look ir"),
                source: wgpu::ShaderSource::Wgsl(k.source.as_str().into()),
            });
            let pipe = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("look ir"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let uni = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("look ir"),
                contents: bytemuck::cast_slice(&[k.points, 0, 0, 0]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let mut entries: Vec<wgpu::BindGroupEntry> = k
                .reads
                .iter()
                .enumerate()
                .map(|(i, &t)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: find(t).1.as_entire_binding(),
                })
                .collect();
            entries.push(wgpu::BindGroupEntry {
                binding: k.reads.len() as u32,
                resource: find(k.node).1.as_entire_binding(),
            });
            entries.push(wgpu::BindGroupEntry {
                binding: k.reads.len() as u32 + 1,
                resource: uni.as_entire_binding(),
            });
            let bg = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("look ir"),
                layout: &pipe.get_bind_group_layout(0),
                entries: &entries,
            });
            pass.set_pipeline(&pipe);
            pass.set_bind_group(0, &bg, &[]);
            let groups = k.count.div_ceil(k.workgroup_size);
            assert!(groups <= 65535, "{groups} workgroups");
            pass.dispatch_workgroups(groups, 1, 1);
        }
    }
    let outs: Vec<(&wgpu::Buffer, usize)> = outputs
        .iter()
        .map(|&t| {
            let (_, b, n) = find(t);
            (b, *n)
        })
        .collect();
    let bytes: u64 = outs.iter().map(|(_, n)| *n as u64 * 4).sum();
    let staging = dev.create_buffer(&wgpu::BufferDescriptor {
        label: Some("look readback"),
        size: bytes.max(4),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut at = 0;
    for (b, n) in &outs {
        enc.copy_buffer_to_buffer(b, 0, &staging, at, *n as u64 * 4);
        at += *n as u64 * 4;
    }
    ctx.queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    loop {
        dev.poll(wgpu::PollType::wait_indefinitely()).expect("device lost");
        match rx.try_recv() {
            Ok(r) => break r.expect("map failed"),
            Err(std::sync::mpsc::TryRecvError::Empty) => std::thread::yield_now(),
            Err(e) => panic!("map: {e}"),
        }
    }
    let all = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range().expect("mapped")).to_vec();
    staging.unmap();
    let mut out = Vec::new();
    let mut at = 0;
    for (_, n) in outs {
        out.push(all[at..at + n].to_vec());
        at += n;
    }
    out
}
