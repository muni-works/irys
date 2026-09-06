//! The learned denoiser as a compute pass.
//!
//! [`crate::neural::Weights`] says what the network is; this says how to run
//! it on the device, in the same place in the frame the à-trous chain runs.
//! It reads the history's running mean, its per-pixel statistics and the
//! resident scene's guide planes, and it writes the filter's answer into the
//! same `(illumination, variance)` scratch buffer the wavelet iterations
//! write — so `history.wgsl`'s `resolve` is untouched and remodulates and
//! tonemaps whichever filter ran.
//!
//! That is the whole of the seam.
//! [`crate::gpu::RayTracePipeline::denoise_and_resolve_resident_neural`] is
//! the à-trous call with the wavelet loop swapped out; everything either side
//! of it — the trace, the fold, the reprojection, the clamp, the resolve — is
//! unchanged and unaware.
//!
//! # Three dispatches
//!
//! One per convolution, with the hidden activations in storage buffers. A
//! fused pass would need about 39 KB of workgroup memory at 32 hidden
//! channels against a 16 KB budget, and would recompute every pixel outside
//! its own 8x8 tile twice; three passes read their 3x3 neighbourhood out of
//! cache and cost two buffers of `hidden * width * height` floats. See
//! `neural.wgsl`.
//!
//! # What it costs
//!
//! Multiply-accumulates per pixel are `9·(10·h + h·h + 25·h)`: about 19,300
//! at 32 hidden channels, against roughly 5 x 25 x 25 taps for a five
//! iteration à-trous. It is not a cheaper filter. It is a filter that was
//! shown the answer.

use super::context::{GpuContext, GpuError};
use crate::neural::Weights;
use bytemuck::{Pod, Zeroable};

/// Workgroup size in each dimension; mirrors `neural.wgsl`.
const WG: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct NeuralParams {
    width: u32,
    height: u32,
    hidden: u32,
    /// Unread by the shader; see `neural.wgsl`. Present so the struct's word
    /// layout is stated rather than padded into existence.
    _reserved: u32,
    w1: u32,
    b1: u32,
    w2: u32,
    b2: u32,
    w3: u32,
    b3: u32,
    count_cutoff: u32,
    /// Where the firefly veto's three scalars start; see
    /// [`crate::neural::Veto`]. `tap_major` leaves them alone, so this is the
    /// file's own offset.
    veto: u32,
}

/// The three compute pipelines and their shared layout, built once.
pub struct NeuralPipeline {
    conv1: wgpu::ComputePipeline,
    conv2: wgpu::ComputePipeline,
    conv3: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

impl NeuralPipeline {
    /// Compile the network's passes.
    pub fn new(ctx: &GpuContext) -> Result<Self, GpuError> {
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Neural Denoise Shader"),
                source: wgpu::ShaderSource::Wgsl(super::shaders::NEURAL_SHADER.into()),
            });
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Neural Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                std::mem::size_of::<NeuralParams>() as u64,
                            ),
                        },
                        count: None,
                    },
                    storage(1, true),  // guides
                    storage(2, true),  // history mean
                    storage(3, true),  // history stats
                    storage(4, false), // activation A
                    storage(5, false), // activation B
                    storage(6, false), // the à-trous scratch the resolve reads
                    storage(7, true),  // weights
                ],
            });
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Neural Pipeline Layout"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        let mk = |entry: &str| {
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        Ok(Self {
            conv1: mk("conv1"),
            conv2: mk("conv2"),
            conv3: mk("conv3_apply"),
            layout,
        })
    }
}

/// The blob with each convolution's weights transposed from the file's
/// `[out][in][tap]` to the shader's `[out][tap][in]`.
///
/// The one thing that is rearranged on the way to the device, and the reason
/// is the inner loop: every convolution in `neural.wgsl` walks the input
/// channel at a fixed neighbour, so tap-major turns a stride-9 gather into a
/// contiguous run beside the equally contiguous activation read. The biases
/// have no tap axis and are copied straight through.
///
/// The veto's three scalars sit past the last bias and are not a matrix, so
/// the clone carries them through untouched.
fn tap_major(w: &Weights) -> Vec<f32> {
    use crate::neural::{C_IN, K, KS};
    let taps = KS * KS;
    let mut out = w.data.clone();
    let off = w.offsets();
    let mut permute = |base: usize, c_out: usize, c_in: usize| {
        for o in 0..c_out {
            for i in 0..c_in {
                for t in 0..taps {
                    out[base + (o * taps + t) * c_in + i] =
                        w.data[base + (o * c_in + i) * taps + t];
                }
            }
        }
    };
    permute(off[0] as usize, w.hidden, C_IN);
    permute(off[2] as usize, w.hidden, w.hidden);
    permute(off[4] as usize, K, w.hidden);
    out
}

/// One network's weights and one frame size's activation buffers, resident on
/// the device.
///
/// Built from a [`Weights`] once and kept for the life of the viewport; the
/// weights never change and the activations are sized by the frame.
pub struct NeuralDenoiser {
    hidden: u32,
    offsets: [u32; 7],
    width: u32,
    height: u32,
    weights: wgpu::Buffer,
    act_a: wgpu::Buffer,
    act_b: wgpu::Buffer,
    params: wgpu::Buffer,
    /// How long a pixel's history may get before the network stops filtering
    /// it; see [`NeuralDenoiser::set_count_cutoff`].
    cutoff: u32,
}

impl NeuralDenoiser {
    /// Upload `weights` and allocate the activations for a `width` x `height`
    /// frame.
    pub fn new(ctx: &GpuContext, weights: &Weights, width: u32, height: u32) -> Self {
        let n = (width as u64) * (height as u64);
        let hidden = weights.hidden as u64;
        let storage = |label: &str, size: u64| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let w = storage("Neural Weights", (weights.data.len() * 4) as u64);
        ctx.queue
            .write_buffer(&w, 0, bytemuck::cast_slice(&tap_major(weights)));
        Self {
            hidden: weights.hidden as u32,
            offsets: weights.offsets(),
            width,
            height,
            weights: w,
            act_a: storage("Neural Activation A", n * hidden * 4),
            act_b: storage("Neural Activation B", n * hidden * 4),
            params: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Neural Params"),
                size: std::mem::size_of::<NeuralParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            cutoff: 32,
        }
    }

    /// The frame size the activations are allocated for.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Hidden channels, and therefore how much of the per-pixel cost this
    /// network is.
    pub fn hidden(&self) -> u32 {
        self.hidden
    }

    /// The history length at which the network stops filtering a pixel and
    /// passes its running mean through.
    ///
    /// The mirror of `GpuDenoiseParams::count_cutoff`, and for the same
    /// reason: once the temporal mean is cleaner than any spatial filter
    /// would leave it, filtering only softens the picture. Default 32.
    pub fn set_count_cutoff(&mut self, cutoff: u32) {
        self.cutoff = cutoff.max(1);
    }

    /// Reallocate for a new frame size, if it changed.
    pub fn ensure(&mut self, ctx: &GpuContext, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }
        let n = (width as u64) * (height as u64) * (self.hidden as u64) * 4;
        let mk = |label: &str| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: n.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        self.act_a = mk("Neural Activation A");
        self.act_b = mk("Neural Activation B");
        self.width = width;
        self.height = height;
    }

    /// Record the three convolutions into `encoder`, leaving the filtered
    /// illumination in `scratch`.
    ///
    /// `guides`, `mean` and `stats` are the resident scene's guide planes and
    /// the history's two buffers, in the layouts `history.wgsl` documents:
    /// `guides` is three `vec4` planes with (normal, depth) at plane 1 and
    /// (albedo, biased id) at plane 2, `mean` is (radiance, coverage) and
    /// `stats` is (count, Σl, Σl², variance of the mean). `scratch` is one of
    /// the history's à-trous ping-pong buffers, in its
    /// `(illumination, variance)` layout, so the caller's `resolve` reads it
    /// exactly as it would read a wavelet iteration's output.
    ///
    /// Public because a host that drives its own encoder — or a parity test
    /// that has no resident scene at all, only the six buffers — has as much
    /// business calling this as
    /// [`super::RayTracePipeline::denoise_and_resolve_resident_neural`] does.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        ctx: &GpuContext,
        pipeline: &NeuralPipeline,
        encoder: &mut wgpu::CommandEncoder,
        guides: &wgpu::Buffer,
        mean: &wgpu::Buffer,
        stats: &wgpu::Buffer,
        scratch: &wgpu::Buffer,
    ) {
        let p = NeuralParams {
            width: self.width,
            height: self.height,
            hidden: self.hidden,
            _reserved: 0,
            w1: self.offsets[0],
            b1: self.offsets[1],
            w2: self.offsets[2],
            b2: self.offsets[3],
            w3: self.offsets[4],
            b3: self.offsets[5],
            count_cutoff: self.cutoff,
            veto: self.offsets[6],
        };
        ctx.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&p));

        let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Neural Bind Group"),
            layout: &pipeline.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: guides.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: mean.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: stats.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.act_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.act_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: scratch.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: self.weights.as_entire_binding(),
                },
            ],
        });

        let groups = (
            self.width.div_ceil(WG).max(1),
            self.height.div_ceil(WG).max(1),
        );
        for (pipe, label) in [
            (&pipeline.conv1, "Neural Conv 1"),
            (&pipeline.conv2, "Neural Conv 2"),
            (&pipeline.conv3, "Neural Conv 3 and Apply"),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        }
    }
}
