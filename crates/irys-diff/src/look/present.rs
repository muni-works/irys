//! M1: a frame straight into a texture on the device (docs/belief-codec.md,
//! M1 item 7): the path tracer's accumulator, divided by the sample count
//! and sRGB-encoded, written into an `rgba8unorm` storage texture — an
//! IOSurface-backed one through `irys-ffi` — with no readback and no copy.

use irys_gpu::GpuContext;
use irys_gpu::wgpu;

use super::scene::Camera;
use super::scene_gpu::{GpuLook, Opts};

const SHADER: &str = r#"
struct PP { size: vec4<u32>, scale: vec4<f32> }
@group(0) @binding(0) var<uniform> Q: PP;
@group(0) @binding(1) var<storage, read> accum: array<vec4<f32>>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba8unorm, write>;

fn enc(c: f32) -> f32 {
    let x = clamp(c, 0.0, 1.0);
    if (x <= 0.0031308) { return 12.92 * x; }
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

@compute @workgroup_size(8, 8)
fn present(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= Q.size.x || gid.y >= Q.size.y) { return; }
    let a = accum[gid.y * Q.size.x + gid.x].xyz * Q.scale.x;
    textureStore(dst, vec2<i32>(gid.xy), vec4<f32>(enc(a.x), enc(a.y), enc(a.z), 1.0));
}
"#;

/// The conversion pass, for one [`GpuLook`] and one target texture.
pub struct Present {
    ctx: &'static GpuContext,
    pipe: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    params: wgpu::Buffer,
    size: (usize, usize),
}

impl Present {
    /// Bind `look`'s accumulator to `target`, an `Rgba8Unorm` texture of the
    /// scene's size with `STORAGE_BINDING`.
    pub fn new(look: &GpuLook, target: &wgpu::Texture) -> Self {
        let ctx = look.ctx();
        let dev = &ctx.device;
        let size = look.size();
        assert_eq!((target.width() as usize, target.height() as usize), size, "target is not the scene's size");
        assert_eq!(target.format(), wgpu::TextureFormat::Rgba8Unorm);
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("present"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let pipe = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("present"),
            layout: None,
            module: &module,
            entry_point: Some("present"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("present params"),
            size: 32,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let view = target.create_view(&Default::default());
        let group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("present"),
            layout: &pipe.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: look.accum_buf().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&view) },
            ],
        });
        Self { ctx, pipe, group, params, size }
    }

    /// Render `cam` and write the frame into the target: one submit, one
    /// wait (so the caller may present it at once), nothing read back.
    pub fn render(&self, look: &GpuLook, cam: &Camera, o: &Opts) -> f64 {
        let t0 = std::time::Instant::now();
        let (dev, q) = (&self.ctx.device, &self.ctx.queue);
        let p: [u32; 8] = [self.size.0 as u32, self.size.1 as u32, 0, 0, (1.0 / o.spp.max(1) as f32).to_bits(), 0, 0, 0];
        q.write_buffer(&self.params, 0, bytemuck::cast_slice(&p));
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut slot = 0;
        look.encode_frame(&mut enc, cam, o, &mut slot);
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipe);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(self.size.0.div_ceil(8) as u32, self.size.1.div_ceil(8) as u32, 1);
        }
        q.submit([enc.finish()]);
        dev.poll(wgpu::PollType::wait_indefinitely()).expect("device lost");
        t0.elapsed().as_secs_f64()
    }
}
