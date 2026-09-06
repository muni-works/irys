//! wgpu compute pipeline for ray tracing.

use super::buffers::{
    GpuCamera, GpuRenderState, RESTIR_STAGE_INITIAL, RESTIR_STAGE_SHADE, RESTIR_STAGE_SPATIAL,
};
use super::context::{GpuContext, GpuError};
use super::geometry::GeometryModule;
use super::scene::SceneRef;

/// Ray tracing compute pipeline.
///
/// One pipeline per geometry module: the client's WGSL is compiled into the
/// shader and its bindings into the layout, both fixed for the pipeline's
/// life. Scenes come and go against it.
pub struct RayTracePipeline {
    pipeline: wgpu::ComputePipeline,
    /// Second pass that refines edge pixels with additional stratified samples.
    refine_pipeline: wgpu::ComputePipeline,
    /// ReSTIR DI: candidate generation, temporal reuse and the survivor's one
    /// shadow ray.
    restir_initial_pipeline: wgpu::ComputePipeline,
    /// ReSTIR DI: one round of spatial reuse.
    restir_spatial_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl RayTracePipeline {
    /// The compute pipeline the main pass runs.
    pub(super) fn compute_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.pipeline
    }

    /// The edge-refinement pass's pipeline.
    pub(super) fn refine_compute_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.refine_pipeline
    }

    /// ReSTIR's candidate-generation and temporal-reuse pass.
    pub(super) fn restir_initial_compute_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.restir_initial_pipeline
    }

    /// ReSTIR's spatial-reuse pass.
    pub(super) fn restir_spatial_compute_pipeline(&self) -> &wgpu::ComputePipeline {
        &self.restir_spatial_pipeline
    }

    /// The bind group layout both passes share.
    pub(super) fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Encode and submit ReSTIR's resampling dispatches for one frame.
    ///
    /// One `restir_initial` over the frame, then `spatial_passes` rounds of
    /// `restir_spatial`, ping-ponging between the two reservoir slots. Each
    /// stage is its own submission because they differ only in the render
    /// state uniform, and a queue write applies to *every* command buffer in
    /// the submission it precedes — four submissions a frame is a rounding
    /// error against a path-traced pass, and a dynamic-offset uniform would
    /// have changed the bind group layout every client shares.
    ///
    /// Returns `(temporal slot, shading slot)`: the reservoirs the *next*
    /// frame's temporal reuse should read — the ones as they were before any
    /// spatial reuse — and the ones this frame should shade with.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_restir_stages(
        &self,
        ctx: &GpuContext,
        state: &GpuRenderState,
        state_buffer: &wgpu::Buffer,
        bind_group: &wgpu::BindGroup,
        dispatch_w: u32,
        dispatch_h: u32,
    ) -> (u32, u32) {
        let temporal_read = state.restir[2] & 0xFF;
        let temporal_write = 1 - temporal_read;
        let mut read = temporal_read;
        let mut write = temporal_write;
        // Spatial output goes to slots 2 and 3, never back over the temporal
        // pair: see `restir_read_slot` in the shader for what feeding one
        // into the other does by frame eight.
        let mut spatial_slot = 2;
        let stages = 1 + state.restir_spatial_passes();
        for stage in 0..stages {
            if stage > 0 {
                read = write;
                write = spatial_slot;
                spatial_slot = if spatial_slot == 2 { 3 } else { 2 };
            }
            let mut st = *state;
            st.restir[2] = read | (write << 8);
            st.restir[3] = if stage == 0 {
                RESTIR_STAGE_INITIAL
            } else {
                RESTIR_STAGE_SPATIAL
            };
            ctx.queue
                .write_buffer(state_buffer, 0, bytemuck::bytes_of(&st));
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ReSTIR Encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ReSTIR Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(if stage == 0 {
                    self.restir_initial_compute_pipeline()
                } else {
                    self.restir_spatial_compute_pipeline()
                });
                pass.set_bind_group(0, bind_group, &[]);
                pass.dispatch_workgroups(dispatch_w.div_ceil(8), dispatch_h.div_ceil(8), 1);
            }
            ctx.queue.submit(Some(encoder.finish()));
        }
        // What next frame's temporal reuse reads, and what this frame's
        // shading pass reads. They are the same slot only when no spatial
        // pass ran.
        (temporal_write, write)
    }

    /// Create a new ray trace pipeline over a client's geometry module.
    ///
    /// Fails if the module declares more than the five bindings the browser
    /// storage-buffer budget leaves for geometry, or declares one outside
    /// `1..=5`. See [`super::geometry`] for the split.
    pub fn new(ctx: &GpuContext, geometry: &GeometryModule) -> Result<Self, GpuError> {
        if geometry.layout.len() > super::geometry::MAX_GEOMETRY_BINDINGS {
            return Err(GpuError::InvalidInput(format!(
                "a geometry module may declare at most {} bindings, got {}",
                super::geometry::MAX_GEOMETRY_BINDINGS,
                geometry.layout.len()
            )));
        }
        if let Some(bad) = geometry
            .layout
            .iter()
            .find(|e| !(1..=5).contains(&e.binding))
        {
            return Err(GpuError::InvalidInput(format!(
                "geometry binding {} is outside the 1..=5 the renderer reserves",
                bad.binding
            )));
        }

        let shader_module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Ray Trace Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    super::shaders::trace_shader(&geometry.wgsl).into(),
                ),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Ray Trace Bind Group Layout"),
                    entries: &[
                        // Camera uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Output texture
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: wgpu::TextureFormat::Rgba8Unorm,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        // Render state uniform (for progressive rendering)
                        wgpu::BindGroupLayoutEntry {
                            binding: 7,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Accumulation buffer (for progressive rendering)
                        // Using storage buffer instead of ReadWrite texture for WebGPU compatibility
                        wgpu::BindGroupLayoutEntry {
                            binding: 8,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Materials storage
                        wgpu::BindGroupLayoutEntry {
                            binding: 9,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Depth/normal buffer (for edge detection)
                        // Using storage buffer instead of ReadWrite texture for WebGPU compatibility
                        wgpu::BindGroupLayoutEntry {
                            binding: 10,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Area lights (softboxes) for MIS-weighted direct lighting
                        wgpu::BindGroupLayoutEntry {
                            binding: 11,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // HDR environment. Textures, not storage buffers: the
                        // ten bindings above already exhaust the browser's
                        // maxStorageBuffersPerShaderStage.
                        wgpu::BindGroupLayoutEntry {
                            binding: 13,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 14,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // Feature ID buffer (per-pixel face_idx for analytic crease detection)
                        wgpu::BindGroupLayoutEntry {
                            binding: 12,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ]
                    .into_iter()
                    // The photon map: a uniform and two textures at 15..=17.
                    .chain(super::resident::CausticBinding::layout_entries())
                    .chain(geometry.layout.iter().copied())
                    .collect::<Vec<_>>(),
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Ray Trace Pipeline Layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Ray Trace Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        let refine_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Ray Trace Refine Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("refine"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let mk = |label: &str, entry: &str| {
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };
        let restir_initial_pipeline = mk("ReSTIR Initial Pipeline", "restir_initial");
        let restir_spatial_pipeline = mk("ReSTIR Spatial Pipeline", "restir_spatial");

        Ok(Self {
            pipeline,
            refine_pipeline,
            restir_initial_pipeline,
            restir_spatial_pipeline,
            bind_group_layout,
        })
    }

    /// Render a scene to an output texture with progressive accumulation.
    ///
    /// This function is async to support WASM's single-threaded environment where
    /// blocking GPU buffer readback causes deadlocks. The async wrapper allows
    /// wasm-bindgen-futures to yield control back to the browser event loop.
    ///
    /// # Arguments
    /// * `ctx` - GPU context
    /// * `scene` - Scene data to render
    /// * `camera` - Camera parameters
    /// * `width` - Output width in pixels
    /// * `height` - Output height in pixels
    /// * `frame_index` - Frame number for progressive accumulation (1 = first frame/reset)
    /// * `accum_buffer` - Optional accumulation buffer from previous frames
    ///
    /// # Returns
    /// A tuple of (pixels, accumulation_buffer) for progressive rendering.
    #[allow(clippy::too_many_arguments)]
    pub async fn render_progressive<'s>(
        &self,
        ctx: &GpuContext,
        scene: impl Into<SceneRef<'s>>,
        camera: &GpuCamera,
        width: u32,
        height: u32,
        frame_index: u32,
        accum_buffer: Option<wgpu::Buffer>,
    ) -> Result<(Vec<u8>, wgpu::Buffer), GpuError> {
        self.render_progressive_with_debug(
            ctx,
            scene,
            camera,
            width,
            height,
            frame_index,
            accum_buffer,
            0,
        )
        .await
    }

    /// Render a scene with debug visualization mode.
    ///
    /// # Arguments
    /// * Same as render_progressive, plus:
    /// * `debug_mode` - Debug visualization: 0=normal, 1=normals as RGB, 2=face_id, 3=n_dot_l, 4=orientation
    #[allow(clippy::too_many_arguments)]
    pub async fn render_progressive_with_debug<'s>(
        &self,
        ctx: &GpuContext,
        scene: impl Into<SceneRef<'s>>,
        camera: &GpuCamera,
        width: u32,
        height: u32,
        frame_index: u32,
        accum_buffer: Option<wgpu::Buffer>,
        debug_mode: u32,
    ) -> Result<(Vec<u8>, wgpu::Buffer), GpuError> {
        // Delegate to full settings with default edge, AO, and refinement parameters
        self.render_with_full_settings(
            ctx,
            scene,
            camera,
            width,
            height,
            frame_index,
            accum_buffer,
            debug_mode,
            true,
            0.1,
            30.0,
            0,
            0,
        )
        .await
    }

    /// Render a scene with full control over all settings.
    ///
    /// # Arguments
    /// * Same as render_progressive_with_debug, plus:
    /// * `enable_edges` - Whether to show edge detection overlay
    /// * `edge_depth_threshold` - Depth discontinuity threshold for edges
    /// * `edge_normal_threshold` - Normal angle threshold (degrees) for edges
    /// * `theme` - Visual theme (0=dark, 1=light)
    /// * `refine_sample_count` - Additional rays per edge pixel (0=disabled, 4/9/16 typical)
    #[allow(clippy::too_many_arguments)]
    pub async fn render_with_full_settings<'s>(
        &self,
        ctx: &GpuContext,
        scene: impl Into<SceneRef<'s>>,
        camera: &GpuCamera,
        width: u32,
        height: u32,
        frame_index: u32,
        accum_buffer: Option<wgpu::Buffer>,
        debug_mode: u32,
        enable_edges: bool,
        edge_depth_threshold: f32,
        edge_normal_threshold: f32,
        theme: u32,
        refine_sample_count: u32,
    ) -> Result<(Vec<u8>, wgpu::Buffer), GpuError> {
        let render_state = GpuRenderState::with_refinement(
            frame_index,
            debug_mode,
            enable_edges,
            edge_depth_threshold,
            edge_normal_threshold,
            theme,
            refine_sample_count,
        );
        let (pixels, accum) = self
            .render_with_render_state(
                ctx,
                scene,
                camera,
                width,
                height,
                accum_buffer,
                render_state,
            )
            .await?;
        Ok((pixels, accum))
    }

    /// Render with a fully-constructed `GpuRenderState` (supports per-type edge style).
    #[allow(clippy::too_many_arguments)]
    pub async fn render_with_render_state<'s>(
        &self,
        ctx: &GpuContext,
        scene: impl Into<SceneRef<'s>>,
        camera: &GpuCamera,
        width: u32,
        height: u32,
        accum_buffer: Option<wgpu::Buffer>,
        render_state: GpuRenderState,
    ) -> Result<(Vec<u8>, wgpu::Buffer), GpuError> {
        use wgpu::util::DeviceExt;

        let scene: SceneRef<'s> = scene.into();

        // Derive the scene-dependent state here rather than trusting callers to
        // keep it in sync with the buffers we are about to bind.
        let mut render_state = render_state;
        render_state.light_count = scene.lights.len() as u32;
        match scene.environment {
            Some(e) => {
                render_state.env_mode = 1;
                render_state.env_width = e.width;
                render_state.env_height = e.height;
                render_state.env_intensity = e.intensity;
                render_state.env_rotation = e.rotation;
                render_state.env_marg_int = e.marg_int;
            }
            None => render_state.env_mode = 0,
        }

        // Create camera buffer
        let camera_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::bytes_of(camera),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create render state buffer
        let render_state_buffer =
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Render State Buffer"),
                    contents: bytemuck::bytes_of(&render_state),
                    usage: wgpu::BufferUsages::UNIFORM.union(wgpu::BufferUsages::COPY_DST),
                });

        // The client's geometry, whatever it is: opaque slabs, bound in the
        // order its module declared them.
        let geometry_buffers: Vec<wgpu::Buffer> = scene
            .geometry
            .slabs()
            .iter()
            .map(|slab| {
                ctx.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(slab.label),
                        contents: slab.bytes,
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            })
            .collect();

        let materials = if scene.materials.is_empty() {
            vec![super::buffers::GpuMaterial::default()]
        } else {
            scene.materials.to_vec()
        };
        let materials_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Materials Buffer"),
                contents: bytemuck::cast_slice(&materials),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Create output texture
        let output_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Output Texture"),
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
        let output_view = output_texture.create_view(&Default::default());

        // Create or reuse accumulation buffer (4 floats per pixel: r, g, b, count)
        let accum_buf_size = (width * height * 16) as u64; // 4 * sizeof(f32)
        let accum = accum_buffer.unwrap_or_else(|| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Accumulation Buffer"),
                size: accum_buf_size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        });

        // Depth/normal buffer for edge detection (vec4 per pixel: normal.xyz,
        // depth), plus the two guide planes a raw-sample pass fills. See the
        // binding's comment in `raytrace.wgsl`.
        // Three planes plus the sample budget's selection mask, and two
        // ReSTIR slots of three each when the reservoirs are on. See the
        // `ReSTIR DI` block in `integrator.wgsl` for why they live here and
        // not in storage buffers of their own.
        let dn_planes: u64 = if render_state.restir_enabled() { 16 } else { 4 };
        let depth_normal_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Depth Normal Buffer"),
            size: accum_buf_size * dn_planes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Area lights. WGSL cannot bind a zero-length storage array, so an
        // unlit scene still gets one dummy entry; `light_count` is what the
        // shader actually loops over.
        let mut lights: Vec<super::buffers::GpuAreaLight> = if scene.lights.is_empty() {
            vec![super::buffers::GpuAreaLight::default()]
        } else {
            scene.lights.to_vec()
        };
        // Built here rather than in `GpuScene`, because callers are free to
        // replace `scene.lights` after the scene is packed — a table baked
        // earlier would describe a rig that is no longer there.
        super::buffers::pack_light_power_table(&mut lights);
        let light_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Area Light Buffer"),
                contents: bytemuck::cast_slice(&lights),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // HDR environment. A gradient-lit scene still gets 1x1 dummies — a
        // texture binding cannot be null — and `env_mode` is what the shader
        // actually branches on.
        let mk_tex = |label: &str, w: u32, h: u32, fmt: wgpu::TextureFormat, data: &[f32]| {
            let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: fmt,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let bytes_per_px = if fmt == wgpu::TextureFormat::Rgba32Float {
                16
            } else {
                4
            };
            ctx.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(data),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * bytes_per_px),
                    rows_per_image: Some(h),
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            tex.create_view(&wgpu::TextureViewDescriptor::default())
        };

        let (env_pixels_view, env_cdf_view) = match scene.environment {
            Some(e) if e.width > 0 && e.height > 0 => (
                mk_tex(
                    "Environment Pixels",
                    e.width,
                    e.height,
                    wgpu::TextureFormat::Rgba32Float,
                    &e.pixels,
                ),
                mk_tex(
                    "Environment CDF",
                    e.width + 1,
                    e.height + 1,
                    wgpu::TextureFormat::R32Float,
                    &e.cdf,
                ),
            ),
            _ => (
                mk_tex(
                    "Environment Pixels (unused)",
                    1,
                    1,
                    wgpu::TextureFormat::Rgba32Float,
                    &[0.0, 0.0, 0.0, 1.0],
                ),
                mk_tex(
                    "Environment CDF (unused)",
                    1,
                    1,
                    wgpu::TextureFormat::R32Float,
                    &[0.0],
                ),
            ),
        };

        // Feature ID buffer: one u32 per pixel storing face_idx (0xFFFFFFFF = background).
        // Written at frame 1 and reused by the crease detector on subsequent frames.
        let feature_id_buf_size = (width * height * 4) as u64;
        let feature_id_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Feature ID Buffer"),
            size: feature_id_buf_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Create readback buffer
        let output_size = (width * height * 4) as u64;
        let padded_bytes_per_row = (width * 4).div_ceil(256) * 256;
        let readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback Buffer"),
            size: (padded_bytes_per_row * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // The one-shot path binds no photon map; the shader's guard stays off.
        let no_caustics = super::resident::CausticBinding::new(ctx, None);

        // Create bind group
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ray Trace Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: render_state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: accum.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: materials_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: depth_normal_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: light_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: feature_id_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::TextureView(&env_pixels_view),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: wgpu::BindingResource::TextureView(&env_cdf_view),
                },
            ]
            .into_iter()
            .chain(no_caustics.entries())
            .chain(
                geometry_buffers
                    .iter()
                    .enumerate()
                    .map(|(i, buf)| wgpu::BindGroupEntry {
                        binding: 1 + i as u32,
                        resource: buf.as_entire_binding(),
                    }),
            )
            .collect::<Vec<_>>(),
        });

        // Dispatch compute shader
        // A scissored pass dispatches only over its rectangle; the shader adds
        // the origin back. That is what makes a masked re-render cost in
        // proportion to what it redraws instead of always paying for the frame.
        let (dispatch_w, dispatch_h) = match render_state.scissor() {
            Some([x, y, w, h]) => (
                w.min(width.saturating_sub(x)),
                h.min(height.saturating_sub(y)),
            ),
            None => (width, height),
        };

        // ReSTIR's resampling runs before the shading pass and submits its own
        // command buffers. In a one-shot render there is nothing behind the
        // frame to reproject, so this is candidate generation plus whatever
        // spatial reuse was asked for — single-frame ReSTIR, which is exactly
        // what a parity test wants to look at.
        let mut render_state = render_state;
        if render_state.restir_enabled() && render_state.scissor().is_none() {
            let (_, shade_slot) = self.encode_restir_stages(
                ctx,
                &render_state,
                &render_state_buffer,
                &bind_group,
                dispatch_w,
                dispatch_h,
            );
            render_state.restir[2] = shade_slot;
            render_state.restir[3] = RESTIR_STAGE_SHADE;
            ctx.queue
                .write_buffer(&render_state_buffer, 0, bytemuck::bytes_of(&render_state));
        }

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Ray Trace Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Ray Trace Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_w.div_ceil(8), dispatch_h.div_ceil(8), 1);
        }

        // The SSAO pass used to run here. Real multi-bounce GI computes contact
        // occlusion correctly, so a screen-space proxy on top of it would only
        // double-darken concave regions.

        // Adaptive refinement pass: fires extra stratified rays at edge pixels.
        // The main pass must fully complete before refine reads depth_normal_buffer,
        // which is guaranteed by wgpu's sequential command encoding.
        if render_state.refine_sample_count > 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Ray Trace Refine Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.refine_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch_w.div_ceil(8), dispatch_h.div_ceil(8), 1);
        }

        // Copy texture to readback buffer
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &output_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        ctx.queue.submit(Some(encoder.finish()));

        let result =
            read_back_rgba(ctx, &readback_buffer, width, height, padded_bytes_per_row).await?;
        let _ = output_size;

        Ok((result, accum))
    }

    /// One pass in linear space, with the denoiser's guide buffers.
    ///
    /// The one-shot twin of [`RayTracePipeline::render_resident_linear`], and
    /// implemented as one: the scene is uploaded, rendered once and dropped.
    /// If you are going to call this more than once on the same geometry, keep
    /// a [`ResidentScene`](super::ResidentScene) instead — this pays the full
    /// upload every time, which is the cost the resident path exists to avoid.
    ///
    /// `state` is forced to raw-sample mode, so the returned radiance is this
    /// pass's own sample rather than a running average. See
    /// [`GpuRenderState::set_raw_sample`] for what that means, and
    /// [`RayTracePipeline::render_resident_linear`] for the Film's conventions.
    pub async fn render_linear_with_render_state<'s>(
        &self,
        ctx: &GpuContext,
        scene: impl Into<SceneRef<'s>>,
        camera: &GpuCamera,
        width: u32,
        height: u32,
        state: GpuRenderState,
    ) -> Result<crate::pathtrace::Film, GpuError> {
        let mut res = self.resident_scene(ctx, scene, width, height);
        self.render_resident_linear(ctx, &mut res, camera, state)
            .await
    }

    /// Render a scene to an output texture (single-frame, non-progressive).
    ///
    /// This is a convenience wrapper around render_progressive for backward compatibility.
    pub async fn render<'s>(
        &self,
        ctx: &GpuContext,
        scene: impl Into<SceneRef<'s>>,
        camera: &GpuCamera,
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, GpuError> {
        let (pixels, _accum) = self
            .render_progressive(ctx, scene, camera, width, height, 1, None)
            .await?;
        Ok(pixels)
    }
}

/// Map a readback buffer and strip wgpu's 256-byte row padding.
///
/// Both the one-shot path and the resident one end here, so the wasm/native
/// split over `map_async` lives in exactly one place.
pub(super) async fn read_back_rgba(
    ctx: &GpuContext,
    readback_buffer: &wgpu::Buffer,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
) -> Result<Vec<u8>, GpuError> {
    wait_for_map(ctx, readback_buffer).await?;

    let buffer_slice = readback_buffer.slice(..);
    let data = buffer_slice
        .get_mapped_range()
        .map_err(|_| GpuError::BufferMapping)?;

    // Remove padding from rows
    let mut result = Vec::with_capacity((width * height * 4) as usize);
    for row in 0..height {
        let row_start = (row * padded_bytes_per_row) as usize;
        let row_end = row_start + (width * 4) as usize;
        result.extend_from_slice(&data[row_start..row_end]);
    }

    drop(data);
    readback_buffer.unmap();

    Ok(result)
}

/// Read a whole mapped buffer back as `f32`s.
///
/// The linear exits copy buffer-to-buffer rather than texture-to-buffer, so
/// there is no 256-byte row padding to strip — the bytes are exactly the
/// `vec4<f32>` planes the shader wrote.
pub(super) async fn read_back_f32(
    ctx: &GpuContext,
    readback_buffer: &wgpu::Buffer,
) -> Result<Vec<f32>, GpuError> {
    wait_for_map(ctx, readback_buffer).await?;
    let buffer_slice = readback_buffer.slice(..);
    let data = buffer_slice
        .get_mapped_range()
        .map_err(|_| GpuError::BufferMapping)?;
    let out = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
    drop(data);
    readback_buffer.unmap();
    Ok(out)
}

/// Await `map_async` on a buffer's whole range.
///
/// The wasm and native halves of that wait are the only difference between the
/// two readback paths, so they live here once.
async fn wait_for_map(ctx: &GpuContext, readback_buffer: &wgpu::Buffer) -> Result<(), GpuError> {
    let buffer_slice = readback_buffer.slice(..);

    // On WASM, use a Promise that resolves when the callback fires
    // This properly yields to the browser event loop
    #[cfg(target_arch = "wasm32")]
    let map_result = {
        use wasm_bindgen::prelude::*;
        use wasm_bindgen_futures::JsFuture;

        // Create a Promise that resolves when map_async callback fires
        let (promise, resolve, reject) = {
            let resolve_ref = std::rc::Rc::new(std::cell::RefCell::new(None::<js_sys::Function>));
            let reject_ref = std::rc::Rc::new(std::cell::RefCell::new(None::<js_sys::Function>));
            let resolve_clone = resolve_ref.clone();
            let reject_clone = reject_ref.clone();

            let promise = js_sys::Promise::new(&mut |resolve, reject| {
                *resolve_clone.borrow_mut() = Some(resolve);
                *reject_clone.borrow_mut() = Some(reject);
            });

            let resolve = resolve_ref.borrow().clone().unwrap();
            let reject = reject_ref.borrow().clone().unwrap();
            (promise, resolve, reject)
        };

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| match result {
            Ok(()) => {
                let _ = resolve.call0(&JsValue::undefined());
            }
            Err(_) => {
                let _ = reject.call1(
                    &JsValue::undefined(),
                    &JsValue::from_str("Buffer mapping failed"),
                );
            }
        });

        // Single poll to submit the mapping request
        let _ = ctx.device.poll(wgpu::PollType::Poll);

        // Await the promise - this yields to browser event loop properly
        match JsFuture::from(promise).await {
            Ok(_) => Ok(()),
            Err(_) => Err(GpuError::BufferMapping),
        }
    };

    #[cfg(not(target_arch = "wasm32"))]
    let map_result = {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let success = Arc::new(AtomicBool::new(false));
        let success_clone = success.clone();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            if result.is_ok() {
                success_clone.store(true, Ordering::SeqCst);
            }
        });

        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

        if success.load(Ordering::SeqCst) {
            Ok(())
        } else {
            Err(GpuError::BufferMapping)
        }
    };

    if map_result.is_err() {
        return Err(GpuError::BufferMapping);
    }
    Ok(())
}

/// Settings for an offline (non-interactive) GPU render.
///
/// Deliberately narrow compared with [`GpuRenderState`]: the viewport-only
/// knobs (edge overlay, stylisation, debug modes, adaptive refinement, the
/// device-side history and ReSTIR) are all forced off, because an offline
/// render wants the integrator's estimate and nothing painted on top of it.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, Copy)]
pub struct OfflineOptions {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Samples per pixel. Each is one dispatch of the main kernel.
    pub spp: u32,
    /// Maximum path length. Held constant across all samples — the viewport's
    /// [`super::depth_for_frame`] escalation exists to make the *first* frame
    /// land fast, which an offline render does not care about, and it would
    /// bias the running mean towards the shallow early samples.
    pub max_depth: u32,
    /// Depth at which Russian roulette begins.
    pub rr_start: u32,
    /// Clamp on indirect radiance to kill fireflies (0 disables).
    pub firefly_clamp: f32,
    /// Overall multiplier on the analytic studio environment. Ignored when the
    /// scene carries an HDR environment, which brings its own intensity.
    pub env_intensity: f32,
    /// Whether the implicit ground plane participates in the path trace.
    pub ground_enabled: bool,
    /// The exact counterpart of
    /// [`crate::pathtrace::PathTraceOptions::show_background`].
    ///
    /// `true` puts the *lighting environment* behind the subject; `false`
    /// leaves it black, which paired with the film's coverage alpha gives a
    /// transparent RGBA render. Either way the viewport's themed `sky_color`
    /// backdrop — a UI choice unrelated to the sky the integrator samples — is
    /// out of the picture.
    ///
    /// Defaults to `true`: an offline render is not a viewport.
    pub show_background: bool,
    /// RNG decorrelation seed. The same seed and the same scene give the same
    /// image, every run, on the same adapter.
    pub seed: u32,
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for OfflineOptions {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            spp: 64,
            max_depth: super::buffers::DEFAULT_MAX_DEPTH,
            rr_start: super::buffers::DEFAULT_RR_START,
            firefly_clamp: super::buffers::DEFAULT_FIREFLY_CLAMP,
            env_intensity: super::buffers::DEFAULT_ENV_INTENSITY,
            ground_enabled: true,
            show_background: true,
            seed: 0,
        }
    }
}

/// The HDR result of an offline render.
///
/// This is the accumulation buffer verbatim: **linear radiance**, not
/// tonemapped and not gamma-encoded. Exposure, ACES and sRGB encoding belong
/// to the caller — feed [`Self::to_film`] into the CPU renderer's output
/// transform rather than writing a second, subtly different one.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone)]
pub struct OfflineResult {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Samples per pixel that were actually accumulated.
    pub spp: u32,
    /// Linear RGB radiance plus coverage, 4 floats per pixel, row-major
    /// top-to-bottom. Alpha is the integrator's coverage estimate: 0 on a
    /// pixel every sample of which missed the geometry, 1 on a fully covered
    /// one, and the fraction between on a silhouette.
    pub rgba: Vec<f32>,
}

#[cfg(not(target_arch = "wasm32"))]
impl OfflineResult {
    /// Linear RGB at one pixel.
    pub fn pixel(&self, x: u32, y: u32) -> [f32; 3] {
        let i = ((y * self.width + x) * 4) as usize;
        [self.rgba[i], self.rgba[i + 1], self.rgba[i + 2]]
    }

    /// Mean relative luminance over the whole image.
    pub fn mean_luminance(&self) -> f32 {
        if self.rgba.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0f64;
        let mut n = 0usize;
        for p in self.rgba.chunks_exact(4) {
            sum += (0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]) as f64;
            n += 1;
        }
        (sum / n.max(1) as f64) as f32
    }

    /// Repackage as a [`crate::pathtrace::Film`] so the CPU renderer's
    /// exposure/ACES/sRGB path can be reused unchanged.
    ///
    /// The normal, depth, albedo and variance guide buffers are left **zeroed**
    /// — an offline render never asks the shader for them, since the guide
    /// planes are only written in raw-sample mode. The result must therefore
    /// not be fed to [`crate::pathtrace::denoise`], which reads all four; use
    /// [`super::ResidentScene`] with `render_resident_linear` if guides are
    /// what you want.
    pub fn to_film(&self) -> crate::pathtrace::Film {
        let n = (self.width as usize) * (self.height as usize);
        let mut rgb = Vec::with_capacity(n * 3);
        let mut alpha = Vec::with_capacity(n);
        for p in self.rgba.chunks_exact(4) {
            rgb.extend_from_slice(&p[..3]);
            alpha.push(p[3]);
        }
        crate::pathtrace::Film {
            width: self.width,
            height: self.height,
            rgb,
            alpha,
            normal: vec![0.0; n * 3],
            depth: vec![0.0; n],
            albedo: vec![0.0; n * 3],
            variance: vec![0.0; n],
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl RayTracePipeline {
    /// Render `spp` samples per pixel offline and read back the HDR
    /// accumulation buffer once.
    ///
    /// [`Self::render_with_render_state`] is shaped for a 1-spp progressive
    /// viewport: every call recreates every scene buffer, rebuilds the bind
    /// group, and reads back the tonemapped `Rgba8Unorm` texture. At one frame
    /// per gesture that is free; at 512 spp it is 512 scene uploads and 512
    /// GPU→CPU round trips, and the round trips alone dominate the render.
    ///
    /// Here the scene goes up once as a [`super::ResidentScene`] and the
    /// per-sample loop rewrites only the render-state uniform — frame index and
    /// Halton jitter — before dispatching the main kernel. Refinement, the
    /// device-side history and denoiser, and ReSTIR are all off: they are
    /// interactivity aids that trade bias for speed at low sample counts, which
    /// is the wrong trade when the whole point is to converge.
    ///
    /// The readback is the f32 accumulation buffer, not the tonemapped texture,
    /// so the caller gets linear HDR radiance and applies its own exposure.
    ///
    /// Native only — it blocks on the buffer mapping, which would deadlock the
    /// browser's single-threaded event loop.
    pub fn render_offline<'s>(
        &self,
        ctx: &GpuContext,
        scene: impl Into<SceneRef<'s>>,
        camera: &GpuCamera,
        opts: &OfflineOptions,
    ) -> Result<OfflineResult, GpuError> {
        let (width, height) = (opts.width.max(1), opts.height.max(1));
        let spp = opts.spp.max(1);

        let mut res = self.resident_scene(ctx, scene, width, height);
        let plane = (width as u64) * (height as u64) * 16;

        // One submit per sample. `write_buffer` is staged and applied at the
        // next submit, so the frame index cannot be advanced several times
        // inside a single encoder — every dispatch would read the same uniform
        // and collapse the running mean.
        for frame_index in 1..=spp {
            let state = self.offline_render_state(opts, frame_index);
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Offline Ray Trace Encoder"),
                });
            self.encode_resident(ctx, &mut res, camera, state, None, &mut encoder);
            ctx.queue.submit(Some(encoder.finish()));
        }

        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Offline HDR Readback Buffer"),
            size: plane,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Offline Readback Encoder"),
            });
        // Only the first plane: the two lobe planes behind it are written in
        // raw-sample mode, which an offline render never asks for.
        encoder.copy_buffer_to_buffer(res.raw_and_guide_buffers().0, 0, &readback, 0, plane);
        ctx.queue.submit(Some(encoder.finish()));

        let rgba = pollster::block_on(read_back_f32(ctx, &readback))?;

        Ok(OfflineResult {
            width,
            height,
            spp,
            rgba,
        })
    }

    /// The render state for one sample of an offline render.
    ///
    /// Everything the viewport uses to stay responsive is off: no edge overlay,
    /// no stylisation, no debug mode, no adaptive refinement, no ReSTIR, and a
    /// fixed `max_depth` instead of the per-frame escalation. The scene-derived
    /// fields (light count, environment) are filled in by
    /// [`super::ResidentScene`] from the buffers actually bound.
    fn offline_render_state(&self, opts: &OfflineOptions, frame_index: u32) -> GpuRenderState {
        let mut s = GpuRenderState::new(frame_index);
        let (jx, jy) = super::buffers::halton_jitter(frame_index);
        s.jitter_x = jx;
        s.jitter_y = jy;
        s.enable_edges = 0;
        s.stylize = 0;
        s.debug_mode = 0;
        s.refine_sample_count = 0;
        s.restir[0] = 0;
        s.max_depth = opts.max_depth.max(1);
        s.rr_start = opts.rr_start;
        s.firefly_clamp = opts.firefly_clamp;
        s.env_intensity = opts.env_intensity;
        s.ground_enabled = u32::from(opts.ground_enabled);
        s.seed = opts.seed;
        s.background_mode = if opts.show_background {
            super::buffers::BACKGROUND_ENVIRONMENT
        } else {
            super::buffers::BACKGROUND_BLACK
        };
        s
    }
}
