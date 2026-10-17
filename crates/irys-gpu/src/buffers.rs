//! The renderer's GPU-side data: materials, area lights, the camera and the
//! per-frame render state.
//!
//! Geometry is not here. A client packs its own primitives and hands them over
//! as opaque slabs — see [`super::GpuGeometry`].

use bytemuck::{Pod, Zeroable};

/// GPU-compatible material representation (PBR).
///
/// Mirrors [`crate::pathtrace::Pbr`] field-for-field so the GPU path tracer and
/// the CPU reference shade identically. The WGSL `GpuMaterial` struct in
/// `shaders/raytrace.wgsl` must match this layout exactly.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuMaterial {
    /// Base color (linear RGB + alpha).
    pub color: [f32; 4],
    /// Metallic factor (0 = dielectric, 1 = metal).
    pub metallic: f32,
    /// Roughness factor (0 = smooth, 1 = rough).
    pub roughness: f32,
    /// Strength of the clearcoat layer (0 = none, 1 = full).
    pub clearcoat: f32,
    /// Perceptual roughness of the clearcoat layer.
    pub clearcoat_roughness: f32,
    /// Dielectric index of refraction, drives the base specular reflectance.
    pub ior: f32,
    /// Signed anisotropy in -1..1: positive stretches the specular highlight
    /// along the local tangent, negative along the bitangent, 0 = isotropic.
    pub anisotropy: f32,
    /// Incident specular amount, Disney's normalised parameter (0.5 = F0 0.04).
    pub specular: f32,
    /// Tint of the dielectric F0 towards the base hue, 0..1.
    pub specular_tint: f32,
    /// Roughness of the diffuse (EON) lobe. 0 is Lambert.
    pub diffuse_roughness: f32,
    /// Hanrahan-Krueger subsurface blend, 0..1.
    pub subsurface: f32,
    /// Strength of the sheen layer, 0 = none.
    pub sheen: f32,
    /// Roughness of the sheen layer, the alpha axis of the LTC fit.
    pub sheen_roughness: f32,
    /// Colour of the sheen layer.
    pub sheen_color: [f32; 3],
    /// Weight of the dielectric transmission lobe, 0 = opaque.
    pub transmission: f32,
    /// Colour transmitted through one attenuation distance of the interior.
    pub attenuation_color: [f32; 3],
    /// Distance over which the interior attenuates to `attenuation_color`.
    /// A non-positive value means no absorption — the shader has no
    /// infinity to compare against, so "off" is spelled `0` on this side.
    pub attenuation_distance: f32,
    /// Abbe number for Cauchy dispersion; 0 = none.
    pub abbe: f32,
    /// Non-zero when the surface is an infinitely thin sheet.
    pub thin_walled: f32,
    /// Non-zero when `sellmeier_b`/`sellmeier_c` carry a real glass, which
    /// overrides `abbe`.
    pub has_sellmeier: f32,
    /// Padding to keep `sellmeier_b` on its 16-byte boundary.
    pub _pad0: f32,
    /// Sellmeier `B` coefficients.
    pub sellmeier_b: [f32; 3],
    /// Padding for 16-byte alignment.
    pub _pad1: f32,
    /// Sellmeier `C` coefficients, in µm².
    pub sellmeier_c: [f32; 3],
    /// Thickness of the thin film in nanometres; 0 = no film.
    pub thin_film_thickness: f32,
    /// Index of refraction of that film.
    pub thin_film_ior: f32,
    /// Padding to 16 bytes.
    pub _pad2: [f32; 3],
    /// Surface colour the subsurface random walk is asked to produce.
    pub subsurface_color: [f32; 3],
    /// Padding for 16-byte alignment.
    pub _pad3: f32,
    /// Mean free path inside the medium, per channel, in scene units.
    pub subsurface_radius: [f32; 3],
    /// Padding for 16-byte alignment. The struct is 192 bytes.
    pub _pad4: f32,
}

impl Default for GpuMaterial {
    fn default() -> Self {
        Self {
            color: [0.7, 0.7, 0.7, 1.0], // Neutral gray
            metallic: 0.0,
            roughness: 0.5,
            clearcoat: 0.0,
            clearcoat_roughness: 0.1,
            ior: 1.5,
            anisotropy: 0.0,
            specular: 0.5,
            specular_tint: 0.0,
            diffuse_roughness: 0.0,
            subsurface: 0.0,
            sheen: 0.0,
            sheen_roughness: 0.3,
            sheen_color: [1.0; 3],
            transmission: 0.0,
            attenuation_color: [1.0; 3],
            attenuation_distance: 0.0,
            abbe: 0.0,
            thin_walled: 0.0,
            has_sellmeier: 0.0,
            _pad0: 0.0,
            sellmeier_b: [0.0; 3],
            _pad1: 0.0,
            sellmeier_c: [0.0; 3],
            thin_film_thickness: 0.0,
            thin_film_ior: 1.5,
            _pad2: [0.0; 3],
            subsurface_color: [1.0; 3],
            _pad3: 0.0,
            subsurface_radius: [1.0; 3],
            _pad4: 0.0,
        }
    }
}

impl GpuMaterial {
    /// Create a new material with the given color.
    pub fn with_color(r: f32, g: f32, b: f32) -> Self {
        Self {
            color: [r, g, b, 1.0],
            ..Default::default()
        }
    }

    /// Create a metallic material.
    pub fn metal(r: f32, g: f32, b: f32, roughness: f32) -> Self {
        Self {
            color: [r, g, b, 1.0],
            metallic: 1.0,
            roughness,
            ..Default::default()
        }
    }

    /// Create a plastic material, optionally clearcoated.
    pub fn plastic(r: f32, g: f32, b: f32, roughness: f32) -> Self {
        Self {
            color: [r, g, b, 1.0],
            metallic: 0.0,
            roughness,
            ..Default::default()
        }
    }

    /// Add a clearcoat layer of the given strength and roughness.
    pub fn with_clearcoat(mut self, clearcoat: f32, clearcoat_roughness: f32) -> Self {
        self.clearcoat = clearcoat;
        self.clearcoat_roughness = clearcoat_roughness;
        self
    }

    /// Build from the CPU reference material.
    ///
    /// Paired with a client's own IR-to-[`crate::pathtrace::Pbr`] conversion
    /// (in vcad, `pathtrace::from_material_def`), this is what makes the
    /// viewport and an offline render derive the SAME material from the same
    /// definition — clearcoat heuristic, IOR and grain included.
    pub fn from_pbr(p: crate::pathtrace::Pbr) -> Self {
        Self {
            color: [p.base_color[0], p.base_color[1], p.base_color[2], 1.0],
            metallic: p.metallic,
            roughness: p.roughness,
            clearcoat: p.clearcoat,
            clearcoat_roughness: p.clearcoat_roughness,
            ior: p.ior,
            anisotropy: p.anisotropy,
            specular: p.specular,
            specular_tint: p.specular_tint,
            diffuse_roughness: p.diffuse_roughness,
            subsurface: p.subsurface,
            sheen: p.sheen,
            sheen_roughness: p.sheen_roughness,
            sheen_color: p.sheen_color,
            transmission: p.transmission,
            attenuation_color: p.attenuation_color,
            // Infinity does not survive a `-ffast-math`-shaped shader as a
            // comparison; "no absorption" is 0 on the GPU and the shader
            // tests for it that way.
            attenuation_distance: if p.attenuation_distance.is_finite() {
                p.attenuation_distance.max(0.0)
            } else {
                0.0
            },
            abbe: p.abbe,
            thin_walled: if p.thin_walled { 1.0 } else { 0.0 },
            has_sellmeier: if p.sellmeier.is_some() { 1.0 } else { 0.0 },
            _pad0: 0.0,
            sellmeier_b: p
                .sellmeier
                .map_or([0.0; 3], |(b, _)| [b[0] as f32, b[1] as f32, b[2] as f32]),
            _pad1: 0.0,
            sellmeier_c: p
                .sellmeier
                .map_or([0.0; 3], |(_, c)| [c[0] as f32, c[1] as f32, c[2] as f32]),
            thin_film_thickness: p.thin_film_thickness,
            thin_film_ior: p.thin_film_ior,
            _pad2: [0.0; 3],
            subsurface_color: p.subsurface_color,
            _pad3: 0.0,
            subsurface_radius: [
                p.subsurface_radius[0] as f32,
                p.subsurface_radius[1] as f32,
                p.subsurface_radius[2] as f32,
            ],
            _pad4: 0.0,
        }
    }

    /// Convert to the CPU reference material, for cross-checking the two
    /// shading paths against each other.
    pub fn to_pbr(self) -> crate::pathtrace::Pbr {
        crate::pathtrace::Pbr {
            base_color: [self.color[0], self.color[1], self.color[2]],
            metallic: self.metallic,
            roughness: self.roughness,
            clearcoat: self.clearcoat,
            clearcoat_roughness: self.clearcoat_roughness,
            ior: self.ior,
            anisotropy: self.anisotropy,
            specular: self.specular,
            specular_tint: self.specular_tint,
            diffuse_roughness: self.diffuse_roughness,
            subsurface: self.subsurface,
            // The device has no phase-function knob: its walk is isotropic,
            // which is exactly `0`. Round-tripping a CPU material with an
            // anisotropy through the GPU layout therefore loses it, and
            // saying so here is better than carrying a field the shader
            // would ignore.
            subsurface_anisotropy: 0.0,
            sheen: self.sheen,
            sheen_roughness: self.sheen_roughness,
            sheen_color: self.sheen_color,
            transmission: self.transmission,
            abbe: self.abbe,
            sellmeier: if self.has_sellmeier != 0.0 {
                Some((
                    [
                        self.sellmeier_b[0] as f64,
                        self.sellmeier_b[1] as f64,
                        self.sellmeier_b[2] as f64,
                    ],
                    [
                        self.sellmeier_c[0] as f64,
                        self.sellmeier_c[1] as f64,
                        self.sellmeier_c[2] as f64,
                    ],
                ))
            } else {
                None
            },
            attenuation_color: self.attenuation_color,
            attenuation_distance: if self.attenuation_distance > 0.0 {
                self.attenuation_distance
            } else {
                f32::INFINITY
            },
            thin_walled: self.thin_walled != 0.0,
            thin_film_thickness: self.thin_film_thickness,
            thin_film_ior: self.thin_film_ior,
            subsurface_color: self.subsurface_color,
            subsurface_radius: [
                self.subsurface_radius[0] as f64,
                self.subsurface_radius[1] as f64,
                self.subsurface_radius[2] as f64,
            ],
            emissive: [0.0; 3],
        }
    }
}

/// GPU-compatible rectangular area light ("softbox").
///
/// Mirrors the WGSL `GpuAreaLight`. Built from [`crate::pathtrace::AreaLight`]
/// via [`GpuAreaLight::from_area_light`] so the GPU and CPU renderers light the
/// scene with the same rig — that is what makes specular highlights on metal
/// match between the viewport and `--photoreal`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuAreaLight {
    /// Centre of the rectangle (w unused).
    pub center: [f32; 4],
    /// Half-extent along the rectangle's first axis (w unused).
    pub u: [f32; 4],
    /// Half-extent along the second axis (w unused).
    pub v: [f32; 4],
    /// Emitted radiance (w unused).
    pub emission: [f32; 4],
}

impl GpuAreaLight {
    /// Convert a CPU reference area light to its GPU representation.
    ///
    /// The power-table fields (`center.w`, `emission.w`) are left at zero;
    /// [`pack_light_power_table`] fills them at upload, after the caller has
    /// finished editing the list.
    pub fn from_area_light(l: &crate::pathtrace::AreaLight) -> Self {
        Self {
            center: [l.center.x as f32, l.center.y as f32, l.center.z as f32, 0.0],
            u: [l.u.x as f32, l.u.y as f32, l.u.z as f32, 0.0],
            v: [l.v.x as f32, l.v.y as f32, l.v.z as f32, 0.0],
            emission: [l.emission[0], l.emission[1], l.emission[2], 0.0],
        }
    }

    /// Area of the emitting rectangle, matching `AreaLight::area`.
    fn area(&self) -> f32 {
        let u = [self.u[0], self.u[1], self.u[2]];
        let v = [self.v[0], self.v[1], self.v[2]];
        let c = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        4.0 * (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt()
    }
}

/// Fill each light's power-table fields in place: `center.w` gets the
/// probability of drawing that light, `emission.w` the running CDF.
///
/// The shader draws one light per bounce from this table instead of
/// shadow-raying all of them, exactly as the CPU integrator does. The table
/// lives in the two spare `w` lanes rather than a binding of its own because
/// the ten storage-buffer slots browsers guarantee are already spoken for.
///
/// Weights come from [`crate::pathtrace::power_table_from_weights`], the same
/// function `SceneAccel` uses, so a CPU-vs-GPU parity test is comparing two
/// renderers sampling one distribution and not two tables that merely look
/// alike.
pub fn pack_light_power_table(lights: &mut [GpuAreaLight]) {
    // Rec. 709 luminance, matching `pathtrace::luminance`.
    let powers: Vec<f32> = lights
        .iter()
        .map(|l| {
            let lum = 0.2126 * l.emission[0] + 0.7152 * l.emission[1] + 0.0722 * l.emission[2];
            (lum * l.area()).max(0.0)
        })
        .collect();
    let (cdf, pick) = crate::pathtrace::power_table_from_weights(&powers);
    for (i, l) in lights.iter_mut().enumerate() {
        l.center[3] = pick[i];
        l.emission[3] = cdf[i];
    }
}

/// Camera parameters for the ray tracer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuCamera {
    /// Camera position.
    pub position: [f32; 4],
    /// Look-at target.
    pub target: [f32; 4],
    /// Up vector.
    pub up: [f32; 4],
    /// World direction mapping to screen +x. Only read when
    /// [`basis_mode`](Self::basis_mode) is [`CAMERA_BASIS_EXPLICIT`].
    pub right: [f32; 4],
    /// Field of view in radians.
    pub fov: f32,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// How the shader builds the screen basis.
    ///
    /// * [`CAMERA_BASIS_DERIVED`] — build it right-handedly from `position`,
    ///   `target` and `up` (`right = forward x up`). What the viewport has
    ///   always done, and what [`GpuCamera::new`] still sets.
    /// * [`CAMERA_BASIS_EXPLICIT`] — use [`right`](Self::right) and
    ///   [`up`](Self::up) *verbatim*, with `forward = normalize(target -
    ///   position)`.
    ///
    /// The explicit mode exists because a client's camera can carry a
    /// **mirrored** (left-handed) screen basis, and no `look_at`-plus-up-hint
    /// construction can reproduce one — rebuilding such a view right-handedly
    /// flips the image left-for-right.
    ///
    /// Occupies what used to be the trailing padding word, so every field
    /// before `right` keeps its offset.
    pub basis_mode: u32,
}

/// [`GpuCamera::basis_mode`]: derive the screen basis from the up hint.
pub const CAMERA_BASIS_DERIVED: u32 = 0;

/// [`GpuCamera::basis_mode`]: use the supplied `right`/`up` verbatim, so a
/// mirrored basis survives the trip to the shader.
pub const CAMERA_BASIS_EXPLICIT: u32 = 1;

/// Render state for progressive rendering.
///
/// Layout (128 bytes, 16-byte aligned — matches `RenderState` in raytrace.wgsl):
/// offset  0–31:  eight u32/f32 scalars (frame_index … theme)
/// offset 32–47:  path tracing (max_depth, rr_start, light_count, env_intensity)
/// offset 48–63:  refine_sample_count, firefly_clamp, ground_enabled, stylize
/// offset 64–79:  silhouette_color vec4
/// offset 80–95:  crease_color vec4
/// offset 96–111: boundary_color vec4
/// offset 112–127: four f32 width/softness scalars
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuRenderState {
    /// Current frame index for accumulation (1-based).
    pub frame_index: u32,
    /// Jitter X offset for anti-aliasing (-0.5 to 0.5).
    pub jitter_x: f32,
    /// Jitter Y offset for anti-aliasing (-0.5 to 0.5).
    pub jitter_y: f32,
    /// Edge-type bit-flags: 0=off, bit0=silhouette, bit1=crease, bit2=boundary.
    pub enable_edges: u32,
    /// Edge detection threshold for depth discontinuity.
    pub edge_depth_threshold: f32,
    /// Edge detection threshold for normal discontinuity (degrees).
    pub edge_normal_threshold: f32,
    /// Debug render mode: 0=normal, 1=show normals, 2=show face_id, 3=show n_dot_l,
    /// 4=show orientation, 5=sample-count heatmap (blue=1 ray, red=max rays).
    pub debug_mode: u32,
    /// Theme: 0 = dark (default), 1 = light. Drives the visible background
    /// palette in `sky_color`; the IBL panels and direct lighting stay
    /// constant across themes so the model itself looks the same.
    pub theme: u32,
    // Path tracing
    /// Maximum path length (1 = direct lighting only). Escalated by the
    /// refinement scheduler: shallow on the draft frame, deeper as
    /// accumulation proceeds, so the first frame stays interactive.
    pub max_depth: u32,
    /// Depth at which Russian roulette begins.
    pub rr_start: u32,
    /// Number of valid entries in the area-light buffer.
    pub light_count: u32,
    /// Overall multiplier on the analytic studio environment.
    pub env_intensity: f32,
    // refinement + path tracing continued
    /// Number of additional refinement rays per edge pixel (0 = disabled).
    /// Actual rays fired = floor(sqrt(refine_sample_count))^2.
    pub refine_sample_count: u32,
    /// Clamp on indirect radiance to kill fireflies (0 = disabled).
    pub firefly_clamp: f32,
    /// Whether the implicit ground plane participates in the path trace.
    pub ground_enabled: u32,
    /// Non-zero enables non-photoreal stylisation (the Sobel edge overlay).
    /// Off in a photoreal viewport: edge lines fight photorealism.
    pub stylize: u32,
    // --- edge style (added for Fusion-style edge lines) ---
    /// Silhouette line color (RGBA linear, depth-gradient edges).
    pub silhouette_color: [f32; 4],
    /// Crease line color (RGBA linear, face-ID boundary edges).
    pub crease_color: [f32; 4],
    /// Boundary line color (RGBA linear, foreground→background edges).
    pub boundary_color: [f32; 4],
    /// Silhouette line apparent width (1.0 = one pixel).
    pub silhouette_width: f32,
    /// Crease line apparent width.
    pub crease_width: f32,
    /// Boundary line apparent width.
    pub boundary_width: f32,
    /// Sub-pixel softness factor (higher = softer AA transition).
    pub edge_softness: f32,
    /// Environment mode: 0 = analytic gradient, 1 = lat-long HDR image.
    pub env_mode: u32,
    /// Environment image width in texels (image mode only).
    pub env_width: u32,
    /// Environment image height in texels (image mode only).
    pub env_height: u32,
    /// Environment rotation about +Z, in radians.
    pub env_rotation: f32,
    /// Normaliser for the environment's uv-space PDF.
    pub env_marg_int: f32,
    /// Scissor origin, `x | (y << 16)` in pixels.
    ///
    /// The compute pass dispatches only over the scissor and offsets every
    /// invocation by this, so a masked pass costs in proportion to the
    /// rectangle rather than the frame. Zero-size means "the whole frame", set
    /// for you by [`GpuRenderState::new`]; use
    /// [`GpuRenderState::set_scissor`] rather than packing it by hand.
    ///
    /// Carved out of what used to be padding, so the struct's size and the
    /// layout of every field before it are unchanged.
    pub scissor_xy: u32,
    /// Scissor size, `w | (h << 16)` in pixels. Zero means the whole frame.
    pub scissor_wh: u32,
    /// Reserved word, kept for layout. It now carries a bit-field of shader
    /// flags — [`FLAG_RAW_SAMPLE`] and [`FLAG_CAMERA_VISIBLE_LIGHTS`]. Set
    /// them with [`GpuRenderState::set_raw_sample`] and
    /// [`GpuRenderState::set_camera_visible_lights`] rather than by hand; the
    /// field name stays for source compatibility.
    pub _pad3: [u32; 1],
    /// Radiance straight up under the analytic gradient environment, in
    /// `.rgb`; `.w` is unused.
    ///
    /// These three mirror [`crate::pathtrace::GradientEnv`]'s fields, and used
    /// to be compiled into the shader instead. A CPU scene lit by any gradient
    /// other than the default — [`crate::pathtrace::Environment::constant`]
    /// most of all, which is a gradient whose three colours are equal — was
    /// therefore lit by a different sky on the GPU, however carefully
    /// `env_intensity` was matched. Set them together with
    /// [`GpuRenderState::set_gradient_env`]; they default to
    /// `GradientEnv::default()`, so a caller who says nothing gets the studio
    /// gradient the shader used to have.
    ///
    /// Ignored when `env_mode` is 1: an HDR image carries its own radiance.
    pub env_zenith: [f32; 4],
    /// Radiance at the horizon under the analytic gradient. See
    /// [`GpuRenderState::env_zenith`].
    pub env_horizon: [f32; 4],
    /// Radiance straight down under the analytic gradient — the bounce off the
    /// studio floor. See [`GpuRenderState::env_zenith`].
    pub env_ground: [f32; 4],
    /// The sun: unit direction **towards** it in `.xyz`, the cosine of its
    /// angular radius in `.w`.
    ///
    /// Mirrors [`crate::pathtrace::Sun`]. A zero `sun_radiance.w` (the PDF)
    /// means there is no sun, which is the default — a caller who says
    /// nothing gets exactly the lighting it always had.
    pub sun_direction: [f32; 4],
    /// The sun's radiance in `.rgb` (irradiance over its solid angle), and
    /// the solid-angle PDF of the NEE strategy — `1 / solid_angle` — in `.w`.
    /// `.w <= 0` disables the sun.
    pub sun_radiance: [f32; 4],
    /// ReSTIR DI: `[candidates M, spatial passes, slot to read, stage]`.
    ///
    /// `candidates == 0` — the default — is the old path: the shader takes the
    /// `sample_lights` + `sample_sun` NEE branch at every depth and none of
    /// the reservoir code runs. Set it with [`GpuRenderState::set_restir`].
    /// `stage` is written by the renderer, not the caller: it names which of
    /// the three ReSTIR dispatches is running (see `RESTIR_STAGE_*`).
    pub restir: [u32; 4],
    /// ReSTIR DI: `[spatial radius px, previous-M cap factor, neighbours k,
    /// unused]`.
    pub restir_params: [f32; 4],
    /// The camera the *previous* pass was rendered from, for ReSTIR's temporal
    /// reprojection: position in `.xyz`, and `.w` non-zero when there is one.
    ///
    /// Filled in by [`crate::gpu::ResidentScene`] from the camera it was handed
    /// last pass — a caller never sets these, and cannot get them wrong.
    pub prev_cam_position: [f32; 4],
    /// The previous pass's camera target in `.xyz`, its vertical field of view
    /// in radians in `.w`.
    pub prev_cam_look_at: [f32; 4],
    /// The previous pass's camera up vector in `.xyz`.
    pub prev_cam_up: [f32; 4],
    /// Extra decorrelation term folded into the WGSL per-pixel RNG seed.
    ///
    /// The shader's white-noise hash is `pixel.x*1973 + pixel.y*9277 +
    /// dim*26699 + frame*12345 + seed*2654435761 + 1`, and the blue-noise
    /// sampler mixes the same `seed*2654435761` into its Owen scramble keys.
    /// Zero — the value the viewport uses and the value every existing
    /// constructor sets — reproduces the pre-seed behaviour bit for bit, so
    /// the browser path is unchanged. An offline render sets it to get a
    /// different but *reproducible* sample sequence for the same frames.
    pub seed: u32,
    /// What a camera ray that hits nothing returns.
    ///
    /// * [`BACKGROUND_SKY`] (0) — `sky_color`, the *themed viewport backdrop*.
    ///   The historical behaviour, and what every viewport constructor sets.
    /// * [`BACKGROUND_ENVIRONMENT`] (1) — `env_radiance`, the same sky the
    ///   integrator lights with. This is [`crate::pathtrace::PathTraceOptions::show_background`]
    ///   on the CPU, so an offline render asks for it.
    /// * [`BACKGROUND_BLACK`] (2) — black with zero coverage, the CPU's
    ///   `show_background` off. Paired with the film's coverage alpha this is
    ///   what makes a transparent PNG.
    ///
    /// It has to be a shader-side choice rather than a CPU composite: a pixel
    /// on the subject's silhouette averages background and surface samples
    /// together, and once that mean exists the two cannot be separated again.
    pub background_mode: u32,
    /// Padding to a 16-byte multiple (required for uniform buffers).
    pub _pad_bg: [u32; 2],
    /// The previous pass's screen `+x` in `.xyz` and its
    /// [`GpuCamera::basis_mode`] in `.w`, so ReSTIR's temporal reprojection
    /// rebuilds the same — possibly mirrored — basis the pass was rendered
    /// with. Filled in by [`crate::gpu::ResidentScene`]; a caller never sets it.
    pub prev_cam_right: [f32; 4],
}

/// [`GpuRenderState::background_mode`]: draw the themed viewport backdrop.
pub const BACKGROUND_SKY: u32 = 0;

/// [`GpuRenderState::background_mode`]: draw the lighting environment, as the
/// CPU renderer does with `PathTraceOptions::show_background`.
pub const BACKGROUND_ENVIRONMENT: u32 = 1;

/// [`GpuRenderState::background_mode`]: leave the backdrop black, matching the
/// CPU renderer with `show_background` off. Paired with the film's coverage
/// alpha this is what makes a transparent PNG.
pub const BACKGROUND_BLACK: u32 = 2;

/// ReSTIR stage: generate candidates, resample temporally, test the survivor.
pub const RESTIR_STAGE_INITIAL: u32 = 0;
/// ReSTIR stage: one round of spatial reuse.
pub const RESTIR_STAGE_SPATIAL: u32 = 1;
/// ReSTIR stage: the shading pass reads the final reservoir.
pub const RESTIR_STAGE_SHADE: u32 = 2;

/// How much longer than this frame's own candidate count a reused temporal
/// reservoir may claim to be. Bitterli et al. use 20x.
pub const RESTIR_DEFAULT_M_CAP: f32 = 20.0;
/// Neighbours drawn per spatial reuse pass.
pub const RESTIR_DEFAULT_NEIGHBOURS: f32 = 4.0;

/// Default silhouette line color: near-black, slightly cool.
const DEFAULT_SILHOUETTE_COLOR: [f32; 4] = [0.08, 0.08, 0.10, 1.0];
/// Default crease line color: slightly lighter than silhouette.
const DEFAULT_CREASE_COLOR: [f32; 4] = [0.12, 0.12, 0.14, 1.0];
/// Default boundary line color: darkest of the three types.
const DEFAULT_BOUNDARY_COLOR: [f32; 4] = [0.06, 0.06, 0.08, 1.0];

/// All three edge types on: bits 0 (silhouette) | 1 (crease) | 2 (boundary).
const EDGES_ALL: u32 = 7;

/// `GpuRenderState`'s flag word, bit 0: write this pass's own raw sample
/// rather than folding it into the running average.
pub const FLAG_RAW_SAMPLE: u32 = 1 << 0;

/// `GpuRenderState`'s flag word, bit 1: area lights are visible to camera
/// rays, as they are to [`crate::pathtrace::render`]'s.
pub const FLAG_CAMERA_VISIBLE_LIGHTS: u32 = 1 << 1;

/// `GpuRenderState`'s flag word, bit 2: the sample budget's per-pixel
/// selection mask is live, and this dispatch is to trace only the pixels it
/// selects for [`GpuRenderState::budget_round`].
///
/// The mask is written by `budget.wgsl`'s `budget_select` into the fourth
/// plane of the depth/normal buffer, one `u32` per pixel with bit `r` set when
/// the pixel folds round `r`. Without this flag the shader traces every pixel
/// in the scissor, which is what every caller that never asked for a budget
/// gets.
pub const FLAG_BUDGET_MASK: u32 = 1 << 2;

/// `GpuRenderState`'s flag word, bit 3: a pixel the mask skips still writes
/// its guide planes — the primary hit and nothing else.
///
/// Set on round 0 only. The history pass's reprojection reads the guides of
/// *every* pixel, folded or not, so leaving a skipped pixel's guides a frame
/// stale would reproject it through last frame's geometry. A guides-only
/// invocation pays for the primary ray and skips the shading, which is where
/// a path-traced sample's cost is.
pub const FLAG_BUDGET_GUIDES: u32 = 1 << 3;

/// `GpuRenderState`'s flag word, bit 4: draw the path's random numbers from
/// [`crate::sampler::SamplePattern::BlueNoise`] rather than the white-noise
/// hash. See [`GpuRenderState::set_sample_pattern`].
pub const FLAG_BLUE_NOISE: u32 = 1 << 4;

/// Where the budget round index sits in `GpuRenderState`'s flag word.
const BUDGET_ROUND_SHIFT: u32 = 8;

/// Full path depth, matching `PathTraceOptions::default().max_depth` so the
/// converged viewport image matches `vcad-render --photoreal`.
pub const DEFAULT_MAX_DEPTH: u32 = 6;
/// Depth at which Russian roulette begins, matching the CPU renderer.
pub const DEFAULT_RR_START: u32 = 3;
/// Environment multiplier, matching `Environment::default().intensity`.
pub const DEFAULT_ENV_INTENSITY: f32 = 0.35;

/// The analytic gradient a render state describes until a caller says
/// otherwise, matching `GradientEnv::default()` field for field. Duplicated as
/// a `const` rather than called, because a struct literal's fields must be:
/// `the_default_render_state_carries_the_default_gradient` in
/// `tests/env_parity.rs` pins the two together.
const DEFAULT_GRADIENT: crate::pathtrace::GradientEnv = crate::pathtrace::GradientEnv {
    zenith: [0.34, 0.42, 0.55],
    horizon: [0.62, 0.64, 0.68],
    ground: [0.18, 0.17, 0.16],
    intensity: DEFAULT_ENV_INTENSITY,
};

/// Widen an RGB triple to the `vec4` the uniform's layout wants.
const fn rgba(c: [f32; 3]) -> [f32; 4] {
    [c[0], c[1], c[2], 0.0]
}
/// Indirect-radiance clamp, matching the CPU renderer's firefly clamp.
pub const DEFAULT_FIREFLY_CLAMP: f32 = 12.0;

/// Path depth to trace on a given accumulation frame.
///
/// Full path tracing is too slow for the viewport's draft frame, so depth
/// escalates with accumulation: the first frame traces shallow (direct
/// lighting plus one bounce) and lands fast, and by the time the `high` tier
/// is accumulating we are at the full depth that matches the CPU renderer.
/// The refinement scheduler in `RayTracedViewport.tsx` resets `frame_index` on
/// every camera change, so each gesture gets a cheap first frame.
///
/// Depth only ever increases, and the accumulation buffer is a running average,
/// so early shallow frames are progressively outweighed by deeper ones.
pub fn depth_for_frame(frame_index: u32, ceiling: u32) -> u32 {
    let d = match frame_index {
        0 | 1 => 2,
        2..=4 => 4,
        _ => DEFAULT_MAX_DEPTH,
    };
    d.min(ceiling.max(1))
}

impl GpuRenderState {
    /// Write each pass's own sample rather than a running average.
    ///
    /// The shader normally folds every pass into `accum_buffer` as
    /// `mix(prev, new, 1/frame_index)`, which is what a viewport converging on
    /// its own wants. A host that keeps its own per-pixel history wants the
    /// opposite: one independent, unweighted sample per pass, plus the guide
    /// buffers to reproject it with. Setting this
    ///
    /// * writes `new_color` straight to the accumulation buffer, keeping the
    ///   path tracer's coverage in alpha instead of the sample count,
    /// * rewrites the depth/normal and feature-ID buffers every pass rather
    ///   than only on frame 1, and fills the guide planes (face-forwarded
    ///   normal, distance from the eye, denoise albedo),
    /// * skips the in-shader spatial denoise, which exists to hide the noise
    ///   in a *converging* average and would correlate samples the host is
    ///   about to average itself.
    ///
    /// `frame_index` still drives the jitter and the RNG, so successive raw
    /// passes at increasing `frame_index` are independent samples of the same
    /// image. Leave `refine_sample_count` at 0: the refinement pass blends
    /// into the same buffer with weights of its own.
    pub fn set_raw_sample(&mut self, on: bool) {
        if on {
            self._pad3[0] |= FLAG_RAW_SAMPLE;
        } else {
            self._pad3[0] &= !FLAG_RAW_SAMPLE;
        }
    }

    /// Whether [`GpuRenderState::set_raw_sample`] is on.
    pub fn raw_sample(&self) -> bool {
        self._pad3[0] & FLAG_RAW_SAMPLE != 0
    }

    /// Trace only the pixels the sample budget selected for `round`.
    ///
    /// `guides` asks the skipped pixels for their primary hit anyway, so the
    /// guide planes come out of the pass whole; set it on round 0, which is
    /// the round the reprojection runs behind. See [`FLAG_BUDGET_MASK`] and
    /// [`FLAG_BUDGET_GUIDES`].
    ///
    /// Set by [`crate::gpu::RayTracePipeline::accumulate_resident_round`] out
    /// of the budget it was handed; a caller driving the rounds itself never
    /// needs to.
    pub fn set_budget_mask(&mut self, on: bool, round: u32, guides: bool) {
        self._pad3[0] &= !(FLAG_BUDGET_MASK | FLAG_BUDGET_GUIDES | (0xFu32 << BUDGET_ROUND_SHIFT));
        if on {
            self._pad3[0] |= FLAG_BUDGET_MASK | ((round & 0xF) << BUDGET_ROUND_SHIFT);
            if guides {
                self._pad3[0] |= FLAG_BUDGET_GUIDES;
            }
        }
    }

    /// Whether [`GpuRenderState::set_budget_mask`] is on.
    pub fn budget_mask(&self) -> bool {
        self._pad3[0] & FLAG_BUDGET_MASK != 0
    }

    /// The round [`GpuRenderState::set_budget_mask`] was given.
    pub fn budget_round(&self) -> u32 {
        (self._pad3[0] >> BUDGET_ROUND_SHIFT) & 0xF
    }

    /// Let camera rays see the area lights.
    ///
    /// The shader has always dropped an emitter hit at depth 0, so a softbox
    /// never appears in frame as a white slab. That is right for the viewport,
    /// whose rig is sized to the scene bounds and swings through frame as the
    /// camera orbits — and wrong for any scene whose lights are part of the
    /// set: [`crate::pathtrace::render`] renders them, so the two tiers
    /// disagree by the whole emission wherever a panel is visible. In Kosm's
    /// closed court, with ten ceiling panels of radiance 18 in frame, the GPU
    /// image came out at 56% of the CPU's — all of it those pixels, the walls
    /// between them agreeing to a tenth of a percent.
    ///
    /// Off by default, so a caller that says nothing renders exactly what it
    /// rendered before. Turn it on for parity with the CPU renderer.
    pub fn set_camera_visible_lights(&mut self, on: bool) {
        if on {
            self._pad3[0] |= FLAG_CAMERA_VISIBLE_LIGHTS;
        } else {
            self._pad3[0] &= !FLAG_CAMERA_VISIBLE_LIGHTS;
        }
    }

    /// Which pattern the shader draws its random numbers from.
    ///
    /// [`crate::sampler::SamplePattern::White`] is the hash the shader has
    /// always used. [`crate::sampler::SamplePattern::BlueNoise`] is the
    /// Owen-scrambled Sobol sequence shifted per pixel by the blue-noise
    /// mask. The sub-pixel jitter stays the host's `jitter_x`/`jitter_y`
    /// either way; see `ray_origin_and_direction` in the integrator for why.
    pub fn set_sample_pattern(&mut self, pattern: crate::sampler::SamplePattern) {
        match pattern {
            crate::sampler::SamplePattern::White => self._pad3[0] &= !FLAG_BLUE_NOISE,
            crate::sampler::SamplePattern::BlueNoise => self._pad3[0] |= FLAG_BLUE_NOISE,
        }
    }

    /// What [`GpuRenderState::set_sample_pattern`] was last given.
    pub fn sample_pattern(&self) -> crate::sampler::SamplePattern {
        if self._pad3[0] & FLAG_BLUE_NOISE != 0 {
            crate::sampler::SamplePattern::BlueNoise
        } else {
            crate::sampler::SamplePattern::White
        }
    }

    /// Whether [`GpuRenderState::set_camera_visible_lights`] is on.
    pub fn camera_visible_lights(&self) -> bool {
        self._pad3[0] & FLAG_CAMERA_VISIBLE_LIGHTS != 0
    }

    /// Create a new render state for the given frame with default edge style.
    pub fn new(frame_index: u32) -> Self {
        let (jitter_x, jitter_y) = halton_2_3(frame_index);
        Self {
            frame_index,
            jitter_x,
            jitter_y,
            enable_edges: EDGES_ALL,
            edge_depth_threshold: 0.1,
            edge_normal_threshold: 30.0,
            debug_mode: 0,
            theme: 0,
            max_depth: depth_for_frame(frame_index, DEFAULT_MAX_DEPTH),
            rr_start: DEFAULT_RR_START,
            light_count: 0,
            env_intensity: DEFAULT_ENV_INTENSITY,
            refine_sample_count: 0,
            firefly_clamp: DEFAULT_FIREFLY_CLAMP,
            ground_enabled: 1,
            stylize: 1,
            silhouette_color: DEFAULT_SILHOUETTE_COLOR,
            crease_color: DEFAULT_CREASE_COLOR,
            boundary_color: DEFAULT_BOUNDARY_COLOR,
            silhouette_width: 1.0,
            crease_width: 0.75,
            boundary_width: 1.25,
            edge_softness: 1.5,
            env_mode: 0,
            env_width: 0,
            env_height: 0,
            env_rotation: 0.0,
            env_marg_int: 0.0,
            scissor_xy: 0,
            scissor_wh: 0,
            _pad3: [match crate::sampler::SamplePattern::default() {
                crate::sampler::SamplePattern::White => 0,
                crate::sampler::SamplePattern::BlueNoise => FLAG_BLUE_NOISE,
            }; 1],
            env_zenith: rgba(DEFAULT_GRADIENT.zenith),
            env_horizon: rgba(DEFAULT_GRADIENT.horizon),
            env_ground: rgba(DEFAULT_GRADIENT.ground),
            sun_direction: [0.0, 0.0, 1.0, 1.0],
            sun_radiance: [0.0; 4],
            restir: [0, 0, 0, RESTIR_STAGE_SHADE],
            restir_params: [0.0, RESTIR_DEFAULT_M_CAP, RESTIR_DEFAULT_NEIGHBOURS, 0.0],
            prev_cam_position: [0.0; 4],
            prev_cam_look_at: [0.0; 4],
            prev_cam_up: [0.0; 4],
            seed: 0,
            background_mode: BACKGROUND_SKY,
            _pad_bg: [0; 2],
            prev_cam_right: [0.0; 4],
        }
    }

    /// Light the direct term with ReSTIR DI (Bitterli et al. 2020) instead of
    /// one next-event sample per pixel.
    ///
    /// `candidates` is M, the number of light samples resampled per pixel per
    /// frame (8-32 is the paper's range); 0 — the default — turns the whole
    /// thing off and leaves the shader on the path it has always taken, which
    /// is why every existing test and `--shot` are unchanged by this landing.
    /// `spatial_passes` is how many rounds of neighbour reuse to run (0-2) and
    /// `spatial_radius` is their radius in pixels (~16 is a good default at
    /// 512x288).
    ///
    /// Only the primary hit's direct lighting from the area lights and the sun
    /// goes through the reservoir. Indirect bounces, and the environment at
    /// every depth, keep the one-light NEE they had.
    pub fn set_restir(&mut self, candidates: u32, spatial_passes: u32, spatial_radius: f32) {
        self.restir[0] = candidates;
        self.restir[1] = spatial_passes;
        self.restir_params[0] = spatial_radius;
    }

    /// Whether [`GpuRenderState::set_restir`] asked for reservoirs.
    pub fn restir_enabled(&self) -> bool {
        self.restir[0] > 0
    }

    /// M, the candidate count per pixel per frame.
    pub fn restir_candidates(&self) -> u32 {
        self.restir[0]
    }

    /// How many spatial reuse passes each frame runs.
    pub fn restir_spatial_passes(&self) -> u32 {
        self.restir[1]
    }

    /// How many neighbours each spatial pass draws (k). Defaults to 4.
    pub fn set_restir_neighbours(&mut self, k: u32) {
        self.restir_params[2] = k as f32;
    }

    /// The cap on a reused temporal reservoir's M, as a multiple of this
    /// frame's candidate count. Defaults to 20, as in the paper.
    pub fn set_restir_m_cap(&mut self, cap: f32) {
        self.restir_params[1] = cap;
    }

    /// Light the scene with `g`, the same analytic gradient the CPU renderer
    /// would integrate.
    ///
    /// This is how a caller whose environment is not the studio default — a
    /// constant sky, a warmer horizon, an unlit white-furnace test — gets the
    /// GPU to agree with [`crate::pathtrace::render`] on what the sky is. It
    /// also sets `env_mode` back to the gradient and clears the image fields,
    /// since the two are alternatives.
    pub fn set_gradient_env(&mut self, g: &crate::pathtrace::GradientEnv) {
        self.env_mode = 0;
        self.env_intensity = g.intensity;
        self.env_zenith = rgba(g.zenith);
        self.env_horizon = rgba(g.horizon);
        self.env_ground = rgba(g.ground);
    }

    /// Place this pass's primary rays under `filter` rather than uniformly
    /// in the pixel.
    ///
    /// The device's sub-pixel jitter is a Halton pair computed *here*, on the
    /// host, so importance-sampling the reconstruction filter is a warp of
    /// two numbers before they are uploaded — and it is
    /// [`crate::pathtrace::PixelFilter::warp`] itself doing it, the same
    /// function the CPU integrator calls. There is no filter code in the
    /// shader at all, and no way for the two tiers to drift.
    ///
    /// [`crate::pathtrace::PixelFilter::Box`] restores the uniform jitter
    /// exactly, which is what [`GpuRenderState::new`] leaves in place.
    pub fn set_pixel_filter(&mut self, filter: crate::pathtrace::PixelFilter) {
        let (u, v) = halton_unit(self.frame_index);
        self.jitter_x = filter.warp(u as f64) as f32;
        self.jitter_y = filter.warp(v as f64) as f32;
    }

    /// Light the scene with a directional sun, as
    /// [`crate::pathtrace::Scene::sun`] does on the CPU. Pass `None` to
    /// remove it.
    pub fn set_sun(&mut self, sun: Option<&crate::pathtrace::Sun>) {
        match sun {
            None => {
                self.sun_direction = [0.0, 0.0, 1.0, 1.0];
                self.sun_radiance = [0.0; 4];
            }
            Some(s) => {
                let d = s.direction.normalize();
                self.sun_direction = [d.x as f32, d.y as f32, d.z as f32, s.cos_radius() as f32];
                let r = s.radiance();
                self.sun_radiance = [r[0], r[1], r[2], (1.0 / s.solid_angle().max(1e-12)) as f32];
            }
        }
    }

    /// Restrict the pass to `[x, y, w, h]` in pixels.
    ///
    /// The dispatch is sized to the rectangle and every invocation is offset
    /// into it, so a masked pass does the work of the rectangle and not of the
    /// frame. Pixels outside keep whatever the accumulation buffer already
    /// holds; the output texture is likewise only written inside.
    ///
    /// Coordinates are packed into 16 bits each, which is the frame size the
    /// rest of the pipeline can address anyway. A zero-area rect clears the
    /// scissor rather than rendering nothing — "no restriction" is the useful
    /// reading of an empty one here, and callers with genuinely nothing to
    /// draw skip the dispatch.
    pub fn set_scissor(&mut self, rect: [u32; 4]) {
        if rect[2] == 0 || rect[3] == 0 {
            self.scissor_xy = 0;
            self.scissor_wh = 0;
            return;
        }
        let clamp = |v: u32| v.min(0xFFFF);
        self.scissor_xy = clamp(rect[0]) | (clamp(rect[1]) << 16);
        self.scissor_wh = clamp(rect[2]) | (clamp(rect[3]) << 16);
    }

    /// The scissor as `[x, y, w, h]`, or `None` when the pass covers the whole
    /// frame.
    pub fn scissor(&self) -> Option<[u32; 4]> {
        if self.scissor_wh == 0 {
            return None;
        }
        Some([
            self.scissor_xy & 0xFFFF,
            self.scissor_xy >> 16,
            self.scissor_wh & 0xFFFF,
            self.scissor_wh >> 16,
        ])
    }

    /// Create a new render state with a specific debug mode.
    pub fn with_debug_mode(frame_index: u32, debug_mode: u32) -> Self {
        let mut state = Self::new(frame_index);
        state.debug_mode = debug_mode;
        state
    }

    /// Create a render state with edge detection disabled.
    #[allow(dead_code)]
    pub fn without_edges(frame_index: u32) -> Self {
        let mut state = Self::new(frame_index);
        state.enable_edges = 0;
        state
    }

    /// Create a render state with custom edge settings.
    pub fn with_edge_settings(
        frame_index: u32,
        debug_mode: u32,
        enable_edges: bool,
        edge_depth_threshold: f32,
        edge_normal_threshold: f32,
    ) -> Self {
        Self::with_full_settings(
            frame_index,
            debug_mode,
            enable_edges,
            edge_depth_threshold,
            edge_normal_threshold,
            0,
            0,
        )
    }

    /// Create a render state with all settings including theme and refinement.
    #[allow(clippy::too_many_arguments)]
    pub fn with_full_settings(
        frame_index: u32,
        debug_mode: u32,
        enable_edges: bool,
        edge_depth_threshold: f32,
        edge_normal_threshold: f32,
        theme: u32,
        refine_sample_count: u32,
    ) -> Self {
        let mut state = Self::new(frame_index);
        state.enable_edges = if enable_edges { EDGES_ALL } else { 0 };
        state.edge_depth_threshold = edge_depth_threshold;
        state.edge_normal_threshold = edge_normal_threshold;
        state.debug_mode = debug_mode;
        state.theme = theme;
        state.refine_sample_count = refine_sample_count;
        state
    }

    /// Create a fully-styled render state.
    ///
    /// `enable_silhouette`, `enable_crease`, `enable_boundary` control which
    /// edge types are rendered independently.
    #[allow(clippy::too_many_arguments)]
    pub fn new_styled(
        frame_index: u32,
        debug_mode: u32,
        enable_silhouette: bool,
        enable_crease: bool,
        enable_boundary: bool,
        edge_depth_threshold: f32,
        edge_normal_threshold: f32,
        theme: u32,
        silhouette_color: [f32; 4],
        crease_color: [f32; 4],
        boundary_color: [f32; 4],
        silhouette_width: f32,
        crease_width: f32,
        boundary_width: f32,
        edge_softness: f32,
    ) -> Self {
        let (jitter_x, jitter_y) = halton_2_3(frame_index);
        let enable_edges = (enable_silhouette as u32)
            | ((enable_crease as u32) << 1)
            | ((enable_boundary as u32) << 2);
        Self {
            frame_index,
            jitter_x,
            jitter_y,
            enable_edges,
            edge_depth_threshold,
            edge_normal_threshold,
            debug_mode,
            theme,
            max_depth: depth_for_frame(frame_index, DEFAULT_MAX_DEPTH),
            rr_start: DEFAULT_RR_START,
            light_count: 0,
            env_intensity: DEFAULT_ENV_INTENSITY,
            refine_sample_count: 0,
            firefly_clamp: DEFAULT_FIREFLY_CLAMP,
            ground_enabled: 1,
            stylize: 1,
            silhouette_color,
            crease_color,
            boundary_color,
            silhouette_width,
            crease_width,
            boundary_width,
            edge_softness,
            env_mode: 0,
            env_width: 0,
            env_height: 0,
            env_rotation: 0.0,
            env_marg_int: 0.0,
            scissor_xy: 0,
            scissor_wh: 0,
            _pad3: [match crate::sampler::SamplePattern::default() {
                crate::sampler::SamplePattern::White => 0,
                crate::sampler::SamplePattern::BlueNoise => FLAG_BLUE_NOISE,
            }; 1],
            env_zenith: rgba(DEFAULT_GRADIENT.zenith),
            env_horizon: rgba(DEFAULT_GRADIENT.horizon),
            env_ground: rgba(DEFAULT_GRADIENT.ground),
            sun_direction: [0.0, 0.0, 1.0, 1.0],
            sun_radiance: [0.0; 4],
            restir: [0, 0, 0, RESTIR_STAGE_SHADE],
            restir_params: [0.0, RESTIR_DEFAULT_M_CAP, RESTIR_DEFAULT_NEIGHBOURS, 0.0],
            prev_cam_position: [0.0; 4],
            prev_cam_look_at: [0.0; 4],
            prev_cam_up: [0.0; 4],
            seed: 0,
            background_mode: BACKGROUND_SKY,
            _pad_bg: [0; 2],
            prev_cam_right: [0.0; 4],
        }
    }

    /// Create a render state with adaptive refinement enabled.
    pub fn with_refinement(
        frame_index: u32,
        debug_mode: u32,
        enable_edges: bool,
        edge_depth_threshold: f32,
        edge_normal_threshold: f32,
        theme: u32,
        refine_sample_count: u32,
    ) -> Self {
        Self::with_full_settings(
            frame_index,
            debug_mode,
            enable_edges,
            edge_depth_threshold,
            edge_normal_threshold,
            theme,
            refine_sample_count,
        )
    }
}

/// Sub-pixel jitter for one accumulation frame, in `[-0.5, 0.5]`.
///
/// The same low-discrepancy offsets [`GpuRenderState::new`] bakes in, exposed
/// so an offline sample loop can advance the jitter without rebuilding the
/// whole render state each sample.
pub fn halton_jitter(frame_index: u32) -> (f32, f32) {
    halton_2_3(frame_index)
}

/// Generate Halton sequence sample for bases 2 and 3.
/// Returns values in range [-0.5, 0.5] for sub-pixel jittering.
fn halton_2_3(index: u32) -> (f32, f32) {
    (halton(index, 2) - 0.5, halton(index, 3) - 0.5)
}

/// The same pair, before centring: two uniforms in [0, 1), which is what a
/// reconstruction filter's warp takes.
fn halton_unit(index: u32) -> (f32, f32) {
    (halton(index, 2), halton(index, 3))
}

/// Halton sequence generator for a given base.
fn halton(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0f32;
    let mut r = 0.0f32;
    let base_f = base as f32;
    while index > 0 {
        f /= base_f;
        r += f * (index % base) as f32;
        index /= base;
    }
    r
}

impl GpuCamera {
    /// Create a new camera for rendering.
    pub fn new(
        position: [f32; 3],
        target: [f32; 3],
        up: [f32; 3],
        fov: f32,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            position: [position[0], position[1], position[2], 1.0],
            target: [target[0], target[1], target[2], 1.0],
            up: [up[0], up[1], up[2], 0.0],
            // Unread in derived mode; zero rather than a made-up axis so a
            // stale value can never be mistaken for a real basis.
            right: [0.0; 4],
            fov,
            width,
            height,
            basis_mode: CAMERA_BASIS_DERIVED,
        }
    }

    /// Create a camera from an explicit — possibly mirrored — screen basis.
    ///
    /// `forward`, `right` and `up` are used as given (the shader normalises
    /// them but does not re-orthogonalise), so a left-handed view reaches the
    /// GPU unflipped. `focus_dist` only positions the `target` point the
    /// shader derives `forward` from; it does not focus anything, since the
    /// GPU tracer is a pinhole.
    #[allow(clippy::too_many_arguments)]
    pub fn from_basis(
        position: [f32; 3],
        forward: [f32; 3],
        right: [f32; 3],
        up: [f32; 3],
        fov: f32,
        focus_dist: f32,
        width: u32,
        height: u32,
    ) -> Self {
        let d = focus_dist.max(1.0);
        Self {
            position: [position[0], position[1], position[2], 1.0],
            target: [
                position[0] + forward[0] * d,
                position[1] + forward[1] * d,
                position[2] + forward[2] * d,
                1.0,
            ],
            up: [up[0], up[1], up[2], 0.0],
            right: [right[0], right[1], right[2], 0.0],
            fov,
            width,
            height,
            basis_mode: CAMERA_BASIS_EXPLICIT,
        }
    }
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    /// The WGSL `GpuMaterial` is laid out by the same rules from the same field
    /// list, and nothing checks that by inspection — a mismatched stride makes
    /// every material after the first read someone else's bytes, which shades
    /// plausibly and wrongly. `tests/gpu_bsdf.rs` would catch it on hardware;
    /// this catches it without a GPU.
    #[test]
    fn the_material_stride_is_what_the_shader_expects() {
        // vec4 color, twelve scalars, then four more 16-byte rows: sheen_color
        // + transmission, attenuation_color + distance, the four dispersion
        // scalars, the two Sellmeier triples with their padding, and the
        // thin film's thickness and index, and the subsurface medium's
        // colour and per-channel mean free path.
        assert_eq!(std::mem::size_of::<GpuMaterial>(), 192);
        assert_eq!(std::mem::align_of::<GpuMaterial>(), 4);
    }
}
