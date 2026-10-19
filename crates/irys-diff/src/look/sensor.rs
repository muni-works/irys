//! M1: the scene kernel as a sensor (docs/belief-codec.md, M1 item 2, and
//! the batching half of item 3).
//!
//! [`Sensor`] traces one primary ray per pixel per camera, no jitter, no
//! shading (`sensor.wgsl`): first-hit distance (metric, `+∞` at a miss),
//! instance id, material id, shading normal, the caller's σ, and the splat
//! layer — each in its own buffer, N cameras in one submit, one wait and one
//! readback, each AOV read back only when asked for.

use bytemuck::{Pod, Zeroable};
use irys_gpu::GpuContext;
use irys_gpu::wgpu;

use super::scene::{Camera, Instance};
use super::scene_gpu::{GpuLook, SPLAT};
use super::splat_gpu::GpuSplats;

const SCENE: &str = include_str!("scene.wgsl");
const SENSOR: &str = include_str!("sensor.wgsl");

/// What `GpuLook` keeps for the sensor kernel, the batch and `set_instances`.
pub(crate) struct Shared {
    pub ctx: &'static GpuContext,
    pub tris: wgpu::Buffer,
    pub inst: wgpu::Buffer,
    pub mats: wgpu::Buffer,
    pub texels: wgpu::Buffer,
    pub textab: wgpu::Buffer,
    pub tlas: wgpu::Tlas,
    pub blas: Vec<Option<wgpu::Blas>>,
    pub splats: GpuSplats,
    pub instances: Vec<Instance>,
    pub first_tri: Vec<u32>,
    pub mesh_mat: Vec<u32>,
    pub emit_base: Vec<u32>,
    /// Instance slots in the TLAS (the splat proxy is the slot after).
    pub cap: usize,
}

impl GpuLook {
    pub(crate) fn ctx(&self) -> &'static GpuContext {
        self.m1.ctx
    }
}

/// The id [`Aovs::inst`] and [`Aovs::mat`] report where the splat layer
/// crossed half coverage before any surface.
pub const SPLAT_ID: u32 = 0xffff_fffe;
/// The id at a miss.
pub const MISS: u32 = u32::MAX;

/// AOV selection bits for [`Sensor::trace`] and [`Sensor::read`].
pub mod aov {
    /// First-hit distance along the unit ray, metres (`+∞`: miss).
    pub const DIST: u32 = 1;
    /// Instance index ([`super::SPLAT_ID`], [`super::MISS`]).
    pub const INST: u32 = 2;
    /// Material index (same markers).
    pub const MAT: u32 = 4;
    /// World shading normal, facing the camera (zero at a miss or a splat).
    pub const NORMAL: u32 = 8;
    /// The caller's σ of what the pixel sees ([`super::Sensor::set_sigma`]).
    pub const SIGMA: u32 = 16;
    /// The splat layer: linear Rec.709 premultiplied, and its coverage.
    pub const RGB: u32 = 32;
    /// Everything.
    pub const ALL: u32 = 63;
}


/// AOVs read back from a [`Sensor`]: camera-major, `cameras × w × h`, each
/// present only if it was asked for.
#[derive(Debug, Clone, Default)]
pub struct Aovs {
    /// Image size.
    pub size: (usize, usize),
    /// Cameras in the batch.
    pub cameras: usize,
    /// [`aov::DIST`].
    pub dist: Option<Vec<f32>>,
    /// [`aov::INST`].
    pub inst: Option<Vec<u32>>,
    /// [`aov::MAT`].
    pub mat: Option<Vec<u32>>,
    /// Unit normals (zero where there is none).
    pub normal: Option<Vec<[f32; 3]>>,
    /// [`aov::SIGMA`].
    pub sigma: Option<Vec<f32>>,
    /// Premultiplied linear rgb and coverage.
    pub rgb: Option<Vec<[f32; 4]>>,
}

/// Decode `pack2x16snorm(oct(n))`.
pub fn oct_unpack(v: u32) -> [f32; 3] {
    if v == 0 {
        return [0.0; 3];
    }
    let s = |x: u16| ((x as i16) as f32 / 32767.0).clamp(-1.0, 1.0);
    let (x, y) = (s(v as u16), s((v >> 16) as u16));
    let z = 1.0 - x.abs() - y.abs();
    let (x, y) = if z < 0.0 {
        ((1.0 - y.abs()) * x.signum_or_one(), (1.0 - x.abs()) * y.signum_or_one())
    } else {
        (x, y)
    };
    let n = (x * x + y * y + z * z).sqrt();
    [x / n, y / n, z / n]
}

trait SignumOrOne {
    fn signum_or_one(self) -> Self;
}
impl SignumOrOne for f32 {
    fn signum_or_one(self) -> f32 {
        if self >= 0.0 { 1.0 } else { -1.0 }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SensorParams {
    size: [u32; 4],
    misc: [f32; 4],
    count: [u32; 4],
}

/// A camera's angular pixel size at the image centre (radians), what the
/// splat dilation scales with: scene.wgsl's `tan_half · |right| · 2 / h`
/// over `|forward|`.
pub fn pixel_angle(c: &Camera, h: usize) -> f64 {
    let n = |a: [f64; 3]| (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    c.tan_half * n(c.right) * 2.0 / h as f64 / n(c.forward)
}

fn cam_rows(c: &Camera, h: usize) -> [[f32; 4]; 4] {
    let v = |a: [f64; 3]| [a[0] as f32, a[1] as f32, a[2] as f32, 0.0];
    let t = c.tan_half;
    // scene.wgsl scales right and up by tan_half; the sensor kernel's
    // convention is tan_half = 1, so fold it in here.
    let mut eye = v(c.eye);
    eye[3] = pixel_angle(c, h) as f32;
    [eye, v(c.forward), v(c.right.map(|x| x * t)), v(c.up.map(|x| x * t))]
}

/// The AOV kernel over one [`GpuLook`] (and its splats), for batches of up
/// to `max_cameras` cameras of one size.
pub struct Sensor {
    ctx: &'static GpuContext,
    pipe: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    params: wgpu::Buffer,
    cams: wgpu::Buffer,
    out: [wgpu::Buffer; 6],
    sigma_in: wgpu::Buffer,
    size: (usize, usize),
    max: usize,
    cap: usize,
    miss_sigma: f32,
    last: (usize, u32),
}

const OUT_BYTES: [u64; 6] = [4, 4, 4, 4, 4, 16];

impl Sensor {
    /// The kernel, its buffers for `max_cameras` × `size`, bound to `look`.
    pub fn new(look: &GpuLook, size: (usize, usize), max_cameras: usize) -> Self {
        let ctx = look.ctx();
        let dev = &ctx.device;
        let sh = &look.m1;
        let scope = dev.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = dev.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("look sensor"),
            // the sensor's splat walk keeps 64 per round (one round per ray
            // on the garage, against 3–4 at 16); the scene kernel keeps 16
            source: wgpu::ShaderSource::Wgsl(
                format!("enable wgpu_ray_query;\n{SCENE}\n{}\n{SENSOR}", SPLAT.replace("const SPLAT_K: u32 = 16u;", "const SPLAT_K: u32 = 64u;")).into(),
            ),
        });
        let ro = |b| matches!(b, 1..=5 | 60 | 61 | 71 | 76);
        let bindings: Vec<u32> = (1..=5).chain([11, 60, 61, 62]).chain(70..=78).collect();
        let layout = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("look sensor"),
            entries: &bindings
                .iter()
                .map(|&b| wgpu::BindGroupLayoutEntry {
                    binding: b,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: match b {
                        11 => wgpu::BindingType::AccelerationStructure { vertex_return: false },
                        62 | 70 => wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        _ => wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: ro(b) },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("look sensor"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipe = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("look sensor"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("aov"),
            compilation_options: Default::default(),
            cache: None,
        });
        let n = (size.0 * size.1 * max_cameras) as u64;
        let buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            dev.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes.max(16), usage, mapped_at_creation: false })
        };
        let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let out = OUT_BYTES.map(|b| buf("look sensor aov", n * b, st));
        let params = buf("look sensor params", 48, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let cams = buf("look sensor cameras", max_cameras as u64 * 64, st);
        let sigma_in = buf("look sensor sigma in", (sh.cap as u64 + 1) * 4, st);
        let mut ents: Vec<wgpu::BindGroupEntry> = Vec::new();
        for (b, r) in [
            (1, &sh.tris),
            (2, &sh.inst),
            (3, &sh.mats),
            (4, &sh.texels),
            (5, &sh.textab),
            (60, &sh.splats.rec),
            (61, &sh.splats.sh),
            (62, &sh.splats.params),
            (70, &params),
            (71, &cams),
            (72, &out[0]),
            (73, &out[1]),
            (74, &out[2]),
            (75, &out[3]),
            (76, &sigma_in),
            (77, &out[4]),
            (78, &out[5]),
        ] {
            ents.push(wgpu::BindGroupEntry { binding: b, resource: r.as_entire_binding() });
        }
        ents.push(wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::AccelerationStructure(&sh.tlas) });
        let group = dev.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("look sensor"), layout: &layout, entries: &ents });
        if let Some(e) = pollster::block_on(scope.pop()) {
            panic!("look sensor: {e}");
        }
        Self { ctx, pipe, group, params, cams, out, sigma_in, size, max: max_cameras, cap: sh.cap, miss_sigma: 0.0, last: (0, 0) }
    }

    /// The belief's σ per instance slot, the splat layer's, and a miss's;
    /// [`aov::SIGMA`] reports the σ of whatever each pixel sees.
    pub fn set_sigma(&mut self, per_instance: &[f32], splat: f32, miss: f32) {
        assert!(per_instance.len() <= self.cap, "more σ than instance slots");
        let mut v = vec![0f32; self.cap + 1];
        v[..per_instance.len()].copy_from_slice(per_instance);
        v[self.cap] = splat;
        self.ctx.queue.write_buffer(&self.sigma_in, 0, bytemuck::cast_slice(&v));
        self.miss_sigma = miss;
    }

    /// Encode the AOVs `which` ([`aov`]) of every camera into `enc`.
    pub fn encode(&mut self, enc: &mut wgpu::CommandEncoder, cams: &[Camera], which: u32) {
        assert!(cams.len() <= self.max, "batch larger than the sensor was built for");
        assert!(!cams.is_empty());
        let q = &self.ctx.queue;
        let rows: Vec<[[f32; 4]; 4]> = cams.iter().map(|c| cam_rows(c, self.size.1)).collect();
        q.write_buffer(&self.cams, 0, bytemuck::cast_slice(&rows));
        let (w, h) = self.size;
        let p = SensorParams {
            size: [w as u32, h as u32, cams.len() as u32, which],
            misc: [self.miss_sigma, 0.0, 0.0, 0.0],
            count: [self.cap as u32, 0, 0, 0],
        };
        q.write_buffer(&self.params, 0, bytemuck::bytes_of(&p));
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        pass.set_pipeline(&self.pipe);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups((w * h).div_ceil(64) as u32, cams.len() as u32, 1);
        drop(pass);
        self.last = (cams.len(), which);
    }

    /// Trace the AOVs `which` for every camera: one submit, no wait.
    pub fn trace(&mut self, cams: &[Camera], which: u32) {
        let mut enc = self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.encode(&mut enc, cams, which);
        self.ctx.queue.submit([enc.finish()]);
    }

    /// Read back the AOVs `which` of the last [`Sensor::trace`]: one wait
    /// and one mapped copy for all of them.
    pub fn read(&self, which: u32) -> Aovs {
        let (n, traced) = self.last;
        assert_eq!(which & !traced, 0, "reading an AOV that was not traced");
        let px = (self.size.0 * self.size.1 * n) as u64;
        let mut spans = Vec::new();
        let mut off = 0u64;
        for (k, b) in OUT_BYTES.iter().enumerate() {
            if which & (1 << k) != 0 {
                spans.push((k, off, px * b));
                off += px * b;
            }
        }
        let mut a = Aovs { size: self.size, cameras: n, ..Default::default() };
        if spans.is_empty() {
            return a;
        }
        let dev = &self.ctx.device;
        let staging = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("look sensor readback"),
            size: off,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for &(k, o, len) in &spans {
            enc.copy_buffer_to_buffer(&self.out[k], 0, &staging, o, len);
        }
        self.ctx.queue.submit([enc.finish()]);
        let bytes = map_read(self.ctx, &staging);
        for &(k, o, len) in &spans {
            let b = &bytes[o as usize..(o + len) as usize];
            match k {
                0 => a.dist = Some(bytemuck::cast_slice(b).to_vec()),
                1 => a.inst = Some(bytemuck::cast_slice(b).to_vec()),
                2 => a.mat = Some(bytemuck::cast_slice(b).to_vec()),
                3 => a.normal = Some(bytemuck::cast_slice::<u8, u32>(b).iter().map(|&v| oct_unpack(v)).collect()),
                4 => a.sigma = Some(bytemuck::cast_slice(b).to_vec()),
                _ => a.rgb = Some(bytemuck::cast_slice(b).to_vec()),
            }
        }
        a
    }

    /// [`Sensor::trace`] then [`Sensor::read`].
    pub fn render(&mut self, cams: &[Camera], which: u32) -> Aovs {
        self.trace(cams, which);
        self.read(which)
    }

    /// The device buffer of one AOV (bit of [`aov`]), camera-major, for a
    /// consumer on the device (the residual pass).
    pub fn buffer(&self, which: u32) -> &wgpu::Buffer {
        assert_eq!(which.count_ones(), 1);
        &self.out[which.trailing_zeros() as usize]
    }

    /// Image size.
    pub fn size(&self) -> (usize, usize) {
        self.size
    }
}

/// Copy-to-staging has been submitted; wait for it and read it.
pub(crate) fn map_read(ctx: &GpuContext, staging: &wgpu::Buffer) -> Vec<u8> {
    let dev = &ctx.device;
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    dev.poll(wgpu::PollType::wait_indefinitely()).expect("device lost");
    rx.recv().expect("map callback").expect("map failed");
    let out = slice.get_mapped_range().expect("mapped").to_vec();
    staging.unmap();
    out
}
