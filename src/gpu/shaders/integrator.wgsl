// The integrator: camera rays, the path trace, NEE + MIS over area lights and
// the environment, the debug modes, the edge refinement pass.
//
// Not valid WGSL on its own. `kosm_render::gpu::shaders::compose` puts
// `bsdf.wgsl`, `prelude.wgsl`, a client's geometry module and `env.wgsl` in
// front of it. The geometry module owns bindings 1..=5 and must define:
//
//   fn trace_scene(origin: vec3<f32>, dir: vec3<f32>) -> RayHit

//   fn hit_normal(hit: RayHit) -> vec3<f32>
//   fn hit_tangent(hit: RayHit) -> vec3<f32>
//   fn hit_material_index(hit: RayHit) -> u32
//   fn hit_orientation(hit: RayHit) -> u32
//
// See `kosm_render::gpu::GpuGeometry` for the Rust half of the same contract.


struct Camera {
    position: vec4<f32>,
    look_at: vec4<f32>,
    up: vec4<f32>,
    fov: f32,
    width: u32,
    height: u32,
    _pad: u32,
}

struct RenderState {
    frame_index: u32,
    jitter_x: f32,
    jitter_y: f32,
    // Edge bit-flags: bit0=silhouette, bit1=crease, bit2=boundary. 0 = edges off.
    enable_edges: u32,
    edge_depth_threshold: f32,
    edge_normal_threshold: f32,
    // 0=normal, 1=normals RGB, 2=face_id, 3=n_dot_l, 4=orientation, 5=sample-count heatmap
    debug_mode: u32,
    // 0=dark, 1=light
    theme: u32,
    // Path tracing controls. max_depth is escalated by the refinement
    // scheduler: shallow for the draft frame, deeper as accumulation proceeds.
    max_depth: u32,
    rr_start: u32,
    light_count: u32,
    env_intensity: f32,
    // Additional rays per edge pixel for adaptive refinement (0 = disabled).
    refine_sample_count: u32,
    // Clamp on indirect radiance to kill fireflies (0 = disabled).
    firefly_clamp: f32,
    // Whether the implicit ground plane participates in the path trace.
    ground_enabled: u32,
    // Non-zero enables non-photoreal stylisation (the Sobel edge overlay).
    // Off in a photoreal viewport: edge lines fight photorealism.
    stylize: u32,
    // Edge style — layout must match GpuRenderState in buffers.rs
    silhouette_color: vec4<f32>,
    crease_color: vec4<f32>,
    boundary_color: vec4<f32>,
    silhouette_width: f32,
    crease_width: f32,
    boundary_width: f32,
    edge_softness: f32,
    // Environment: 0 = analytic gradient, 1 = lat-long HDR image in env_data.
    env_mode: u32,
    env_width: u32,
    env_height: u32,
    env_rotation: f32,
    env_marg_int: f32,
    // Scissor rectangle, packed x | (y << 16) and w | (h << 16). A zero size
    // means the whole frame. The host dispatches only enough workgroups to
    // cover the rectangle and each invocation adds the origin, so a masked
    // pass costs in proportion to the rectangle.
    scissor_xy: u32,
    scissor_wh: u32,
    // Shader flags, a bit-field. Bit 0 (FLAG_RAW_SAMPLE): write the raw
    // per-pass sample instead of folding it into the running average, and fill
    // the guide half of `depth_normal_buffer` — what a host keeping its own
    // per-pixel history wants out of a pass. Bit 1
    // (FLAG_CAMERA_VISIBLE_LIGHTS): let camera rays see the area lights, as
    // the CPU renderer's do. The field name is the Rust struct's.
    raw_sample: u32,
    // The analytic gradient's three radiances, mirroring
    // pathtrace::GradientEnv. These used to be constants in `env_radiance`,
    // which meant a scene lit by any other gradient was lit by a different sky
    // here than on the CPU.
    env_zenith: vec4<f32>,
    env_horizon: vec4<f32>,
    env_ground: vec4<f32>,
    // The sun: direction towards it in .xyz, cos(angular radius) in .w.
    sun_direction: vec4<f32>,
    // The sun's radiance in .rgb and its NEE PDF (1 / solid angle) in .w.
    // A .w of zero means there is no sun.
    sun_radiance: vec4<f32>,
    // ReSTIR DI. .x = candidate count M (0 = the whole feature off, and the
    // shader takes the NEE path it always took), .y = spatial passes,
    // .z = the reservoir slot this dispatch reads, .w = which stage is running.
    restir: vec4<u32>,
    // .x = spatial radius in pixels, .y = the cap on a reused temporal M as a
    // multiple of M, .z = neighbours per spatial pass, .w unused.
    restir_params: vec4<f32>,
    // The previous pass's camera, for temporal reprojection. .w of the
    // position is non-zero when there is one.
    prev_cam_position: vec4<f32>,
    // Previous camera target in .xyz, its vertical fov in radians in .w.
    prev_cam_look_at: vec4<f32>,
    prev_cam_up: vec4<f32>,
}

// A rectangular area light. Layout must match GpuAreaLight in buffers.rs,
// which is built from pathtrace::AreaLight.
struct GpuAreaLight {
    // Centre of the rectangle. .w is this light's probability of being drawn
    // from the power table (see pack_light_power_table in buffers.rs).
    center: vec4<f32>,
    // Half-extent along the rectangle's first axis (.w unused).
    u: vec4<f32>,
    // Half-extent along the second axis (.w unused).
    v: vec4<f32>,
    // Emitted radiance. .w is the running CDF of the power table, so the
    // last light's is 1.0.
    emission: vec4<f32>,
}

// Bind groups

@group(0) @binding(0) var<uniform> camera: Camera;

@group(0) @binding(6) var output: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(7) var<uniform> render_state: RenderState;
@group(0) @binding(8) var<storage, read_write> accum_buffer: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> materials: array<GpuMaterial>;
// Three vec4 planes of width*height each:
//   [0 .. n)      (normal, t) in the shader's own convention — background is
//                 (0,0,0, MAX_T) — read by the denoiser and the Sobel edges.
//   [n .. 2n)     guide: (face-forwarded world normal, distance from the eye),
//                 background (0,0,0, 0) — the CPU `Film` convention.
//   [2n .. 3n)    guide: (denoise albedo, 0), background (0,0,0, 0).
// The two guide planes are only written when `render_state.raw_sample` is set.
@group(0) @binding(10) var<storage, read_write> depth_normal_buffer: array<vec4<f32>>;
// Rectangular area lights ("softboxes"). Intersectable, so both BSDF sampling
// and NEE find them and combine under MIS — that is what puts correctly-shaped
// highlights on metal. Populated from `pathtrace::studio_rig`, the same
// function the CPU renderer uses, so both rigs are identical.
@group(0) @binding(11) var<storage, read> lights: array<GpuAreaLight>;
// HDR environment as textures, not storage buffers: browsers cap
// maxStorageBuffersPerShaderStage at 10 and the bindings above already use all
// ten. See env.wgsl for the layout.
@group(0) @binding(13) var env_pixels: texture_2d<f32>;
@group(0) @binding(14) var env_cdf: texture_2d<f32>;

// Per-pixel face_idx for analytic crease detection (0xFFFFFFFF = background).
// Written at frame 1; read at frame 2+ by detect_edge_sobel.
@group(0) @binding(12) var<storage, read_write> feature_id_buffer: array<u32>;

// Helper functions for buffer indexing (2D coords to 1D index)
// Bit 0 of the flag word: this pass writes its own raw sample.
const FLAG_RAW_SAMPLE: u32 = 1u;
// Bit 1: area lights are visible to camera rays.
const FLAG_CAMERA_VISIBLE_LIGHTS: u32 = 2u;

fn raw_sample_mode() -> bool {
    return (render_state.raw_sample & FLAG_RAW_SAMPLE) != 0u;
}

fn camera_visible_lights() -> bool {
    return (render_state.raw_sample & FLAG_CAMERA_VISIBLE_LIGHTS) != 0u;
}

fn pixel_index(coord: vec2<u32>) -> u32 {
    return coord.y * camera.width + coord.x;
}

fn pixel_index_i32(coord: vec2<i32>) -> u32 {
    return u32(coord.y) * camera.width + u32(coord.x);
}

// Utility functions

// Core ray generation with an explicit sub-pixel offset.
// offset is in pixels, typically in [-0.5, 0.5].
fn ray_origin_and_direction_offset(pixel: vec2<u32>, offset: vec2<f32>) -> mat2x3<f32> {
    let aspect = f32(camera.width) / f32(camera.height);
    let fov_tan = tan(camera.fov * 0.5);

    // Compute normalized device coordinates with the given offset
    let ndc = vec2<f32>(
        (f32(pixel.x) + 0.5 + offset.x) / f32(camera.width) * 2.0 - 1.0,
        1.0 - (f32(pixel.y) + 0.5 + offset.y) / f32(camera.height) * 2.0
    );

    // Build camera coordinate system
    let forward = normalize(camera.look_at.xyz - camera.position.xyz);
    let right = normalize(cross(forward, camera.up.xyz));
    let up = cross(right, forward);

    // Compute ray direction
    let dir = normalize(
        forward +
        right * ndc.x * fov_tan * aspect +
        up * ndc.y * fov_tan
    );

    return mat2x3<f32>(camera.position.xyz, dir);
}

// Ray generation using the Halton-sequence jitter from render_state (main pass).
fn ray_origin_and_direction(pixel: vec2<u32>) -> mat2x3<f32> {
    let jitter = vec2<f32>(render_state.jitter_x, render_state.jitter_y);
    return ray_origin_and_direction_offset(pixel, jitter);
}

// Procedural HDR environment in Z-up world space. Returns radiance for a
// given ray direction — used both for ray misses (visible background) and
// for ambient + IBL specular sampling.
//
// Modeled as a "studio environment": dim atmospheric backdrop plus a few
// high-luminance soft panels at fixed directions (key, fill, rim, top
// fill). The panels carry HDR values (luminance > 1) so reflections on
// metals get hot specular highlights that ACES rolls off into clean
// blown-white spots — the same look you get sampling a real HDRI.
//
// Stays in shader code rather than uploading a baked HDR texture; the
// binding plumbing isn't worth it for a single environment, and tuning
// in WGSL is faster than re-baking an exr.
fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    let z = dir.z;

    // Atmospheric backdrop. Two palettes — dark and light — selected by
    // render_state.theme. The IBL panels below stay the same in both so
    // the model's lighting is theme-independent.
    var zenith: vec3<f32>;
    var horizon: vec3<f32>;
    var below: vec3<f32>;
    if render_state.theme == 1u {
        // Light theme — bright neutral with the faintest cool tint at the
        // top so it doesn't read as flat paper-white.
        zenith = vec3<f32>(0.93, 0.95, 1.00);
        horizon = vec3<f32>(0.96, 0.97, 0.99);
        below = vec3<f32>(0.86, 0.86, 0.88);
    } else {
        // Dark theme — moody studio backdrop, cool blues fading into a
        // dim "below horizon" band.
        zenith = vec3<f32>(0.35, 0.55, 0.95);
        horizon = vec3<f32>(0.78, 0.84, 0.92);
        below = vec3<f32>(0.18, 0.18, 0.20);
    }

    var col: vec3<f32>;
    if z >= 0.0 {
        col = mix(horizon, zenith, smoothstep(0.0, 0.55, z));
    } else {
        col = mix(horizon, below, smoothstep(0.0, -0.4, z));
    }

    // Helper to add a soft directional panel. `tightness` controls disc
    // size (higher = tighter, more sun-like; lower = broader, softer
    // panel). HDR-valued so reflections on shiny surfaces blow out
    // through ACES into clean specular highlights.
    // Using inline pow() since WGSL has no closures or generics.

    // Primary key — sun. Tight & hot.
    let sun = sun_direction();
    let sun_dot = dot(dir, sun);
    let sun_disk = smoothstep(0.998, 0.9995, sun_dot) * 35.0;
    let sun_glow = pow(max(sun_dot, 0.0), 96.0) * 1.0;
    col += vec3<f32>(1.00, 0.94, 0.82) * (sun_disk + sun_glow);

    // Warm fill from upper-front-right. Broader, lower luminance —
    // simulates a softbox.
    let fill = normalize(vec3<f32>(0.55, -0.3, 0.7));
    let fill_dot = max(dot(dir, fill), 0.0);
    col += vec3<f32>(1.00, 0.88, 0.72) * pow(fill_dot, 18.0) * 6.0;

    // Cool rim from behind/below — gives metals a clean blue-tinged
    // back-rim highlight.
    let rim = normalize(vec3<f32>(0.2, 0.85, -0.15));
    let rim_dot = max(dot(dir, rim), 0.0);
    col += vec3<f32>(0.55, 0.72, 1.00) * pow(rim_dot, 28.0) * 4.5;

    // Top diffuse panel — broad cool light from straight up. Acts like
    // a studio ceiling and provides the dominant ambient term for the
    // tops of objects.
    let top = vec3<f32>(0.0, 0.0, 1.0);
    let top_dot = max(dot(dir, top), 0.0);
    col += vec3<f32>(0.95, 0.97, 1.00) * pow(top_dot, 6.0) * 1.8;

    return col;
}

// Primary key-light direction in kernel (Z-up) space. Upper-back-left so
// the camera, which typically sits in the +x/-y/+z octant, sees a clear
// lit/shadow split rather than backlight.
fn sun_direction() -> vec3<f32> {
    return normalize(vec3<f32>(-0.35, 0.55, 0.75));
}

// Implicit ground plane at z=0 (kernel space). The plane is always full
// opacity — fade-to-sky is applied by `shade_ground` based on horizontal
// distance from the world origin (where models typically sit), so that
// the model stays grounded regardless of camera distance.
struct GroundHit {
    t: f32,
    point: vec3<f32>,
    fade: f32,
}

fn intersect_ground(origin: vec3<f32>, dir: vec3<f32>) -> GroundHit {
    var hit: GroundHit;
    hit.t = MAX_T;
    hit.fade = 0.0;

    if abs(dir.z) < EPSILON {
        return hit;
    }
    let t = -origin.z / dir.z;
    if t < 0.001 {
        return hit;
    }
    let p = origin + dir * t;

    // fade is computed against the world origin (where the model usually
    // sits) so the ground reads as "platform under the model" rather than
    // "puddle around the camera".
    let horizontal_from_origin = length(p.xy);
    let fade = 1.0 - smoothstep(100.0, 1500.0, horizontal_from_origin);
    if fade <= 0.0 {
        return hit;
    }

    hit.t = t;
    hit.point = p;
    hit.fade = fade;
    return hit;
}

// Cheap shadow test: reuse trace_bvh and check if the closest hit lies
// before the light. WGSL doesn't allow recursion or function pointers, so
// this is the simplest correct path; an "any-hit" early-exit traversal
// would be ~30% faster but isn't needed for current scene complexity.
fn in_shadow(p: vec3<f32>, light_dir: vec3<f32>, max_t: f32) -> bool {
    let bias = 0.0015;
    let origin = p + light_dir * bias;
    let hit = trace_scene(origin, light_dir);
    return hit.face_idx != 0xFFFFFFFFu && hit.t < max_t;
}

// PCG hash → uniform [0, 1) noise. Per-pixel + per-frame seed so the noise
// decorrelates across pixels (prevents banding) and animates per frame
// (so progressive accumulation averages out).
fn rand_uniform(pixel: vec2<u32>, sample_idx: u32) -> f32 {
    var state = pixel.x * 1973u + pixel.y * 9277u + sample_idx * 26699u + render_state.frame_index * 12345u + 1u;
    state = state * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    let r = (word >> 22u) ^ word;
    return f32(r) / 4294967296.0;
}

fn rand_uniform2(pixel: vec2<u32>, sample_idx: u32) -> vec2<f32> {
    return vec2<f32>(rand_uniform(pixel, sample_idx * 2u), rand_uniform(pixel, sample_idx * 2u + 1u));
}

// Build a tangent frame around a normal so we can sample directions in its
// hemisphere. Choose a stable tangent reference axis based on the normal's
// dominant component.
fn build_tangent_frame(normal: vec3<f32>) -> mat3x3<f32> {
    let up_ref = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.y) > 0.95);
    let tangent = normalize(cross(up_ref, normal));
    let bitangent = cross(normal, tangent);
    return mat3x3<f32>(tangent, bitangent, normal);
}

// Cosine-weighted hemisphere sample around `normal`. Uses Malley's method
// (sample disc, project to hemisphere) for cosine weighting.
fn sample_hemisphere_cosine(normal: vec3<f32>, u: vec2<f32>) -> vec3<f32> {
    let r = sqrt(u.x);
    let theta = 2.0 * PI * u.y;
    let local = vec3<f32>(r * cos(theta), r * sin(theta), sqrt(max(0.0, 1.0 - u.x)));
    return build_tangent_frame(normal) * local;
}

// Jitter a direction within a small cone for soft area-light shadows.
// `cone_radius` is the tangent of the cone half-angle (small ≈ small angle).
fn jitter_direction(dir: vec3<f32>, cone_radius: f32, u: vec2<f32>) -> vec3<f32> {
    let angle = 2.0 * PI * u.x;
    let r = cone_radius * sqrt(u.y);
    let offset = vec3<f32>(cos(angle) * r, sin(angle) * r, 0.0);
    return normalize(build_tangent_frame(dir) * vec3<f32>(offset.x, offset.y, 1.0));
}

// Deterministic low-discrepancy Halton sequence for the given base.
// Returns values in [0, 1). Used by the SSAO kernel so sample patterns are
// stable across calls and decorrelate across frames via frame_base offset.
fn halton_wgsl(index: u32, base: u32) -> f32 {
    var f = 1.0;
    var r = 0.0;
    var i = index;
    let b = f32(base);
    for (var iter = 0u; iter < 32u; iter++) {
        if i == 0u { break; }
        f = f / b;
        r = r + f * f32(i % base);
        i = i / base;
    }
    return r;
}

// Reconstruct world-space position from a pixel coord and depth-buffer t value.
// Uses un-jittered pixel centre so SSAO sample positions are frame-stable.
fn world_pos_from_depth(pixel: vec2<u32>, t: f32) -> vec3<f32> {
    let aspect = f32(camera.width) / f32(camera.height);
    let fov_tan = tan(camera.fov * 0.5);
    let ndc = vec2<f32>(
        (f32(pixel.x) + 0.5) / f32(camera.width)  * 2.0 - 1.0,
        1.0 - (f32(pixel.y) + 0.5) / f32(camera.height) * 2.0
    );
    let forward = normalize(camera.look_at.xyz - camera.position.xyz);
    let right   = normalize(cross(forward, camera.up.xyz));
    let up_cam  = cross(right, forward);
    let dir = normalize(forward + right * ndc.x * fov_tan * aspect + up_cam * ndc.y * fov_tan);
    return camera.position.xyz + dir * t;
}

// Project a world-space point onto the screen. Returns pixel coords, or
// (-1, -1) when the point is behind the camera or outside the viewport.
fn world_to_screen_coords(world_pos: vec3<f32>) -> vec2<i32> {
    let forward = normalize(camera.look_at.xyz - camera.position.xyz);
    let right   = normalize(cross(forward, camera.up.xyz));
    let up_cam  = cross(right, forward);
    let fov_tan = tan(camera.fov * 0.5);
    let aspect  = f32(camera.width) / f32(camera.height);
    let p       = world_pos - camera.position.xyz;
    let view_z  = dot(p, forward);
    if view_z <= 0.0 { return vec2<i32>(-1, -1); }
    let ndc_x = dot(p, right)   / (view_z * fov_tan * aspect);
    let ndc_y = dot(p, up_cam)  / (view_z * fov_tan);
    let px = i32((ndc_x + 1.0) * 0.5 * f32(camera.width));
    let py = i32((1.0 - ndc_y)  * 0.5 * f32(camera.height));
    return vec2<i32>(px, py);
}

// ACES Narkowicz tonemap. Cleaner highlights and richer mids than Reinhard.
fn tonemap_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ─── environment ──────────────────────────────────────────────────────────

// Incoming radiance from the analytic studio environment.
//
// Mirrors `pathtrace::GradientEnv::radiance`, taking its three radiances from
// the render state rather than having `GradientEnv::default()`'s baked in, so
// the GPU and CPU integrate the SAME sky whatever gradient the caller set. It
// is deliberately low-frequency: BSDF sampling alone converges on it, which is
// why the gradient needs no environment CDF.
//
// `GpuRenderState::new` fills those three with the studio defaults, so a
// caller who says nothing gets exactly the sky this function used to hardcode.
//
// Distinct from `sky_color`, which is the themed backdrop the viewport DRAWS.
// Lighting must match the CPU; the visible background is a UI choice.
fn env_radiance(d: vec3<f32>) -> vec3<f32> {
    let zenith = render_state.env_zenith.rgb;
    let horizon = render_state.env_horizon.rgb;
    let ground_c = render_state.env_ground.rgb;

    if render_state.env_mode == ENV_MODE_IMAGE {
        return env_image_radiance(
            d,
            render_state.env_width,
            render_state.env_height,
            render_state.env_intensity,
            render_state.env_rotation,
        );
    }

    let t = d.z;
    var c: vec3<f32>;
    if t >= 0.0 {
        let k = smoothstep(0.0, 1.0, pow(t, 0.65));
        c = mix(horizon, zenith, k);
    } else {
        let k = smoothstep(0.0, 1.0, pow(-t, 0.5));
        c = mix(horizon, ground_c, k);
    }
    return c * render_state.env_intensity;
}

// ─── area lights ──────────────────────────────────────────────────────────

fn light_normal(l: GpuAreaLight) -> vec3<f32> {
    return normalize(cross(l.u.xyz, l.v.xyz));
}

fn light_area(l: GpuAreaLight) -> f32 {
    return 4.0 * length(cross(l.u.xyz, l.v.xyz));
}

struct LightHit {
    t: f32,
    index: u32,
    hit: bool,
}

// Closest intersection against the emitting (front) face of any area light.
// Lights are intersectable so a BSDF ray can find them too — that is what
// makes MIS work and what gives metal correctly-shaped softbox highlights
// instead of a jittered sun cone.
fn intersect_lights(origin: vec3<f32>, dir: vec3<f32>) -> LightHit {
    var out: LightHit;
    out.t = MAX_T;
    out.index = 0u;
    out.hit = false;

    for (var i = 0u; i < render_state.light_count; i = i + 1u) {
        let l = lights[i];
        let n = light_normal(l);
        let denom = dot(n, dir);
        if abs(denom) < 1e-12 {
            continue;
        }
        // Emitting face only: we must be looking at the front.
        if denom > 0.0 {
            continue;
        }
        let t = dot(n, l.center.xyz - origin) / denom;
        if t <= ray_eps(origin) || t >= out.t {
            continue;
        }
        let p = origin + dir * t;
        let rel = p - l.center.xyz;
        let ul = length(l.u.xyz);
        let vl = length(l.v.xyz);
        let du = dot(rel, l.u.xyz / ul);
        let dv = dot(rel, l.v.xyz / vl);
        if abs(du) <= ul && abs(dv) <= vl {
            out.t = t;
            out.index = i;
            out.hit = true;
        }
    }
    return out;
}

// How many thin transmissive sheets one shadow ray may cross before it is
// declared blocked. Mirrors `pathtrace::MAX_SHADOW_SHEETS`.
const MAX_SHADOW_SHEETS: u32 = 4u;

// The fraction of a shadow ray that survives one thin sheet, or a negative
// number if the surface is an honest blocker. Mirrors
// `pathtrace::sheet_transmittance`: the same single-interface `1 - F` the
// thin-walled BSDF branch applies, so NEE and BSDF sampling through the same
// pane carry the same weight and MIS combines them without double counting.
fn sheet_transmittance(m: GpuMaterial, cos_dot: f32) -> f32 {
    if m.thin_walled == 0.0 || m.transmission <= 0.0 {
        return -1.0;
    }
    // A rough sheet scatters; the straight-line shadow ray is only right in
    // the smooth limit, so taper to zero as the lobe opens up.
    let clarity = clamp(1.0 - mat_alpha(m), 0.0, 1.0);
    if clarity <= 0.0 {
        return -1.0;
    }
    let f = fresnel_dielectric(clamp(abs(cos_dot), 0.0, 1.0), max(m.ior, 1.0));
    let t = m.transmission * (1.0 - f) * clarity;
    if t <= 0.0 {
        return -1.0;
    }
    return t;
}

// How much of a light's radiance survives the trip from `origin` along `dir`
// to `max_dist`; a negative return means blocked outright. Mirrors
// `pathtrace::Scene::shadow_transmittance`.
//
// The material-blind any-hit test this replaces made a window pane an opaque
// wall, which is why a room could not be lit through glass at any sample
// count. A thin-walled transmissive sheet is a filter, not a blocker: the
// shadow ray carries straight on, dimmed.
fn shadow_transmittance(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32) -> f32 {
    // Scale-aware on both ends: the near end so the shadow ray does not find
    // the surface it left, the far end so it does not find the light's own
    // surface just short of `max_dist`.
    let eps = ray_eps(origin);
    if render_state.ground_enabled != 0u {
        let g = intersect_ground(origin, dir);
        if g.t > eps && g.t < max_dist - eps {
            return -1.0;
        }
    }
    var tr = 1.0;
    var o = origin;
    var travelled = 0.0;
    for (var crossed = 0u; crossed <= MAX_SHADOW_SHEETS; crossed = crossed + 1u) {
        let hit = trace_scene(o, dir);
        let e = ray_eps(o);
        if hit.face_idx == 0xFFFFFFFFu || hit.t <= e || travelled + hit.t >= max_dist - eps {
            return tr;
        }
        if crossed == MAX_SHADOW_SHEETS {
            // More sheets than the cap allows: fall back to opaque.
            return -1.0;
        }
        let m = materials[hit_material_index(hit)];
        let sheet = sheet_transmittance(m, dot(hit_normal(hit), dir));
        if sheet < 0.0 {
            return -1.0;
        }
        tr = tr * sheet;
        if tr <= 1e-6 {
            return -1.0;
        }
        travelled = travelled + hit.t + e;
        o = o + dir * (hit.t + e);
    }
    return tr;
}


// ─── surface description at a hit ─────────────────────────────────────────

struct Surface {
    point: vec3<f32>,
    normal: vec3<f32>,
    material: GpuMaterial,
}

// The implicit ground plane's material, matching the studio floor that
// `vcad-render --photoreal` builds (photoreal.rs `Backdrop::Studio`).
fn ground_material() -> GpuMaterial {
    var m: GpuMaterial;
    m.color = vec4<f32>(0.55, 0.55, 0.56, 1.0);
    m.metallic = 0.0;
    m.roughness = 0.6;
    m.clearcoat = 0.0;
    m.clearcoat_roughness = 0.1;
    m.ior = 1.5;
    m.anisotropy = 0.0;
    m.specular = 0.5;
    m.specular_tint = 0.0;
    m.diffuse_roughness = 0.0;
    m.subsurface = 0.0;
    m.sheen = 0.0;
    m.sheen_roughness = 0.3;
    m.sheen_color = vec3<f32>(1.0);
    m.transmission = 0.0;
    m.attenuation_color = vec3<f32>(1.0);
    m.attenuation_distance = 0.0;
    m.abbe = 0.0;
    m.thin_walled = 0.0;
    m.has_sellmeier = 0.0;
    m._pad0 = 0.0;
    m.sellmeier_b = vec3<f32>(0.0);
    m._pad1 = 0.0;
    m.sellmeier_c = vec3<f32>(0.0);
    m.subsurface_color = vec3<f32>(1.0);
    m._pad5 = 0.0;
    m.subsurface_radius = vec3<f32>(1.0);
    m._pad6 = 0.0;
    m.thin_film_thickness = 0.0;
    m.thin_film_ior = 1.5;
    m._pad2 = 0.0;
    m._pad3 = 0.0;
    m._pad4 = 0.0;
    return m;
}

// ─── next-event estimation ────────────────────────────────────────────────

// Draw one light from the power-weighted table packed into the light buffer's
// spare .w lanes. Returns the index; the pick probability is that light's
// center.w. A linear scan, not a binary search: the table is a handful of
// softboxes and a branchless walk beats divergent bisection on a warp.
fn pick_light(u: f32) -> u32 {
    let n = render_state.light_count;
    for (var i = 0u; i < n; i = i + 1u) {
        if u < lights[i].emission.w {
            return i;
        }
    }
    return n - 1u;
}

// Sample *one* area light, drawn by power and divided by its pick
// probability, MIS-weighted against the BSDF strategy with the power
// heuristic. Mirrors `pathtrace::Scene::sample_lights`: one shadow ray per
// bounce however many panels the rig has, and the pick probability rides in
// the light PDF so the BSDF-hits-an-emitter branch below can reconstruct it.
fn sample_lights(
    p: vec3<f32>,
    frame: mat3x3<f32>,
    n: vec3<f32>,
    wo_local: vec3<f32>,
    m: GpuMaterial,
    eta: f32,
    lambda_nm: f32,
    pixel: vec2<u32>,
    depth: u32,
) -> vec3<f32> {
    if render_state.light_count == 0u {
        return vec3<f32>(0.0);
    }
    // Three fresh randoms per bounce: one to pick the light, two to place the
    // sample on it.
    let rp = rand_uniform2(pixel, 97u + depth * 23u);
    let idx = pick_light(rp.x);
    let l = lights[idx];
    let pick_pdf = l.center.w;
    if pick_pdf <= 0.0 {
        return vec3<f32>(0.0);
    }
    let r = rand_uniform2(pixel, 101u + depth * 17u);
    let lp = l.center.xyz + l.u.xyz * (2.0 * r.x - 1.0) + l.v.xyz * (2.0 * r.y - 1.0);
    let to_light = lp - p;
    let dist = length(to_light);
    if dist < 1e-9 {
        return vec3<f32>(0.0);
    }
    let wi_world = to_light / dist;
    let ln = light_normal(l);
    let cos_light = -dot(wi_world, ln);
    if cos_light <= 1e-9 {
        return vec3<f32>(0.0);
    }
    let wi_local = to_local(frame, wi_world);
    if wi_local.z <= 0.0 {
        return vec3<f32>(0.0);
    }

    let e = bsdf_eval(m, wo_local, wi_local, eta, lambda_nm);
    if max3(e.value) <= 0.0 {
        return vec3<f32>(0.0);
    }

    // Solid-angle PDF of the full NEE strategy: pick this light, then pick a
    // point on it.
    let light_pdf = pick_pdf * dist * dist / (cos_light * light_area(l));
    if light_pdf <= 0.0 {
        return vec3<f32>(0.0);
    }

    let tr = shadow_transmittance(offset_origin(p, n), wi_world, dist);
    if tr < 0.0 {
        return vec3<f32>(0.0);
    }

    let w = power_heuristic(light_pdf, e.pdf);
    return e.value * l.emission.rgb * (tr * w / light_pdf);
}

// Next-event estimation against the environment, MIS-weighted against BSDF
// sampling. Only runs for an importance-sampled environment (the HDR image);
// the analytic gradient is low-frequency enough that BSDF sampling alone
// integrates it, which is why it has no CDF in either renderer.
//
// Mirrors `pathtrace::Scene::sample_environment`.
fn sample_environment(
    p: vec3<f32>,
    frame: mat3x3<f32>,
    n: vec3<f32>,
    wo_local: vec3<f32>,
    m: GpuMaterial,
    eta: f32,
    lambda_nm: f32,
    pixel: vec2<u32>,
    depth: u32,
) -> vec3<f32> {
    if render_state.env_mode != ENV_MODE_IMAGE {
        return vec3<f32>(0.0);
    }
    let r = rand_uniform2(pixel, 601u + depth * 19u);
    let es = env_image_sample(
        r.x,
        r.y,
        render_state.env_width,
        render_state.env_height,
        render_state.env_intensity,
        render_state.env_rotation,
        render_state.env_marg_int,
    );
    if !es.ok || max3(es.radiance) <= 0.0 {
        return vec3<f32>(0.0);
    }
    let wi_local = to_local(frame, es.dir);
    if wi_local.z <= 0.0 {
        return vec3<f32>(0.0);
    }
    let e = bsdf_eval(m, wo_local, wi_local, eta, lambda_nm);
    if max3(e.value) <= 0.0 {
        return vec3<f32>(0.0);
    }
    // The environment is at infinity: nothing between here and the sky may
    // block, so the shadow ray is unbounded.
    let tr = shadow_transmittance(offset_origin(p, n), es.dir, MAX_T);
    if tr < 0.0 {
        return vec3<f32>(0.0);
    }
    let w = power_heuristic(es.pdf, e.pdf);
    return e.value * es.radiance * (tr * w / es.pdf);
}

// ─── the sun ──────────────────────────────────────────────────────────────
//
// A directional light of finite angular size, mirroring `pathtrace::Sun`. It
// is its own MIS strategy: NEE samples the cone uniformly, and a BSDF ray
// that escapes into the cone picks the same radiance up under the balance
// heuristic. Without the cone sample a 1-degree disc is found by BSDF
// sampling roughly once in ten thousand rays, which is the entire reason
// daylight needs this and not an area light placed very far away.

fn sun_enabled() -> bool {
    return render_state.sun_radiance.w > 0.0;
}

// Radiance arriving from `d`: the disc, or nothing.
fn sun_radiance_in(d: vec3<f32>) -> vec3<f32> {
    if !sun_enabled() {
        return vec3<f32>(0.0);
    }
    if dot(normalize(d), render_state.sun_direction.xyz) >= render_state.sun_direction.w {
        return render_state.sun_radiance.rgb;
    }
    return vec3<f32>(0.0);
}

fn sun_pdf(d: vec3<f32>) -> f32 {
    if !sun_enabled() {
        return 0.0;
    }
    if dot(normalize(d), render_state.sun_direction.xyz) >= render_state.sun_direction.w {
        return render_state.sun_radiance.w;
    }
    return 0.0;
}

// Uniform sample of the cone, matching `Sun::sample` term for term.
fn sun_sample_dir(r1: f32, r2: f32) -> vec3<f32> {
    let cos_max = render_state.sun_direction.w;
    let cos_theta = 1.0 - r1 * (1.0 - cos_max);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    let phi = 2.0 * PI * r2;
    let w = render_state.sun_direction.xyz;
    let f = build_tangent_frame(w);
    return normalize(f * vec3<f32>(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta));
}

// Next-event estimation against the sun, MIS-weighted against BSDF sampling.
// Mirrors `pathtrace::Scene::sample_sun`.
fn sample_sun(
    p: vec3<f32>,
    frame: mat3x3<f32>,
    n: vec3<f32>,
    wo_local: vec3<f32>,
    m: GpuMaterial,
    eta: f32,
    lambda_nm: f32,
    pixel: vec2<u32>,
    depth: u32,
) -> vec3<f32> {
    if !sun_enabled() {
        return vec3<f32>(0.0);
    }
    let r = rand_uniform2(pixel, 811u + depth * 13u);
    let wi_world = sun_sample_dir(r.x, r.y);
    let pdf = render_state.sun_radiance.w;
    let wi_local = to_local(frame, wi_world);
    if wi_local.z <= 0.0 {
        return vec3<f32>(0.0);
    }
    let e = bsdf_eval(m, wo_local, wi_local, eta, lambda_nm);
    if max3(e.value) <= 0.0 {
        return vec3<f32>(0.0);
    }
    // The sun is at infinity, so the shadow ray is unbounded.
    let tr = shadow_transmittance(offset_origin(p, n), wi_world, MAX_T);
    if tr < 0.0 {
        return vec3<f32>(0.0);
    }
    let w = power_heuristic(pdf, e.pdf);
    return e.value * render_state.sun_radiance.rgb * (tr * w / pdf);
}

// ─── ReSTIR DI ────────────────────────────────────────────────────────────
//
// Bitterli, Wyman, Pharr, Shirley, Lefohn and Jarosz 2020, "Spatiotemporal
// reservoir resampling for real-time ray tracing with dynamic direct
// lighting". The direct term at the *primary* hit stops being one
// next-event sample and becomes a resampled one: M candidates are drawn per
// pixel per frame and reduced to a single survivor by weighted reservoir
// sampling against an unshadowed target
//
//     p̂(y) = luminance( f(wo, wi)·cosθ · Le(y) ) · G(x, y)
//
// which costs no rays at all; only the survivor is shadow-tested. The
// reservoir is then reused — from this pixel last frame, and from its
// neighbours this frame — so a pixel's one shadow ray is spent on a light
// that hundreds of samples agreed was worth testing.
//
// Scope. Only the primary hit, and only the area panels. Indirect bounces and
// the environment keep the one-light NEE they had: a reservoir is per *pixel*,
// and there is no pixel behind a second bounce.
//
// The sun keeps its own next-event sample at every depth, resampled or not.
// It went into the reservoir first, as one strategy among the panels, and
// that was wrong in the way that matters: the court is a room lit through
// clerestory openings *by the sun*, and folding it into an eleven-way pick
// meant ten pixels in eleven got no sun sample at all that frame. Sixty live
// frames came out visibly darker and three times specklier than the same
// frames without ReSTIR. A 0.6° disc is one light with a good importance
// sampler already; resampling is for choosing among many.
//
// Measure. A panel's sample is a point, carried in area measure, so
// reconnecting it to a different surface has a Jacobian of 1 — which is the
// whole reason the paper resamples points rather than directions.
//
// Storage. The reservoirs ride in the spare planes of `depth_normal_buffer`
// rather than in storage buffers of their own: the ten this shader binds are
// every one a browser guarantees, and five of those belong to the geometry
// module. Two slots of three planes each — ping-ponged across the frame's
// dispatches and across frames — sit above the three planes the denoiser's
// guides use, so a ReSTIR frame costs 96 bytes a pixel that an ordinary one
// does not allocate at all.

// No sample in this reservoir.
const RESTIR_NONE: u32 = 0xFFFFFFFFu;
const RESTIR_STAGE_INITIAL: u32 = 0u;
const RESTIR_STAGE_SPATIAL: u32 = 1u;
const RESTIR_STAGE_SHADE: u32 = 2u;
// The geometric similarity gates the paper's spatial and temporal reuse use.
const RESTIR_DEPTH_TOLERANCE: f32 = 0.10;
const RESTIR_COS_TOLERANCE: f32 = 0.9063;  // cos 25°
// Slots the spatial pass keeps for the neighbours it combined.
const RESTIR_MAX_NEIGHBOURS: u32 = 8u;

fn restir_on() -> bool {
    return render_state.restir.x > 0u;
}

fn restir_m() -> u32 {
    return render_state.restir.x;
}

// Four reservoir slots, not two. 0 and 1 ping-pong across *frames* and carry
// the temporal chain; 2 and 3 ping-pong across this frame's spatial passes.
// Keeping them apart is what stops the spatial pass's bias from being fed
// back into the temporal one and compounding: with the spatial output as the
// temporal source, a 30 px radius measured *worse than plain NEE* by frame
// eight while measuring 2.4x better on frame one, which is what a feedback
// loop looks like. The host packs the pair as read | (write << 8).
fn restir_read_slot() -> u32 {
    return render_state.restir.z & 0xFFu;
}

fn restir_write_slot() -> u32 {
    return (render_state.restir.z >> 8u) & 0xFFu;
}

// One reservoir: the survivor `y`, the running sum of resampling weights, the
// number of candidates it stands for, and the unbiased contribution weight W
// that turns f(y) into an estimate of the whole integral.
struct Reservoir {
    y_pos: vec3<f32>,
    y_light: u32,
    w_sum: f32,
    m: f32,
    w_out: f32,
    p_hat: f32,
}

// The surface a reservoir belongs to, as stored beside it: enough to
// re-evaluate p̂ for somebody else's sample without tracing the primary ray
// again, and to run the depth and normal gates against.
struct RSurface {
    point: vec3<f32>,
    normal: vec3<f32>,
    mat: u32,
    valid: bool,
}

fn reservoir_empty() -> Reservoir {
    var r: Reservoir;
    r.y_pos = vec3<f32>(0.0);
    r.y_light = RESTIR_NONE;
    r.w_sum = 0.0;
    r.m = 0.0;
    r.w_out = 0.0;
    r.p_hat = 0.0;
    return r;
}

// Weighted reservoir sampling, Chao's algorithm: absorb a candidate carrying
// resampling weight `w` and standing for `m_inc` samples, keeping it with
// probability w / w_sum.
fn reservoir_update(
    r: ptr<function, Reservoir>,
    y_pos: vec3<f32>,
    y_light: u32,
    p_hat: f32,
    w: f32,
    m_inc: f32,
    rnd: f32,
) {
    (*r).m = (*r).m + m_inc;
    if w <= 0.0 {
        return;
    }
    (*r).w_sum = (*r).w_sum + w;
    if rnd * (*r).w_sum <= w {
        (*r).y_pos = y_pos;
        (*r).y_light = y_light;
        (*r).p_hat = p_hat;
    }
}

// W = w_sum / (M · p̂). Equation 6 of the paper: the factor that makes
// f(y)·W an unbiased estimate of ∫f however the candidates were drawn.
fn reservoir_finalize(r: ptr<function, Reservoir>) {
    let denom = (*r).m * (*r).p_hat;
    if denom > 0.0 && (*r).y_light != RESTIR_NONE {
        (*r).w_out = (*r).w_sum / denom;
    } else {
        (*r).w_out = 0.0;
    }
}

// ── octahedral normal packing, so the surface fits in the planes we have ──

fn oct_wrap(v: vec2<f32>) -> vec2<f32> {
    let s = select(vec2<f32>(-1.0), vec2<f32>(1.0), v >= vec2<f32>(0.0));
    return (1.0 - abs(vec2<f32>(v.y, v.x))) * s;
}

fn oct_encode(n: vec3<f32>) -> u32 {
    let d = abs(n.x) + abs(n.y) + abs(n.z);
    if d < 1e-12 {
        return 0u;
    }
    var p = n.xy * (1.0 / d);
    if n.z < 0.0 {
        p = oct_wrap(p);
    }
    return pack2x16snorm(p);
}

fn oct_decode(e: u32) -> vec3<f32> {
    let f = unpack2x16snorm(e);
    var n = vec3<f32>(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    if n.z < 0.0 {
        let w = oct_wrap(n.xy);
        n = vec3<f32>(w.x, w.y, n.z);
    }
    return normalize(n);
}

// ── the reservoir planes ─────────────────────────────────────────────────

fn restir_plane(slot: u32) -> u32 {
    return (3u + slot * 3u) * camera.width * camera.height;
}

fn restir_store(slot: u32, idx: u32, r: Reservoir, s: RSurface) {
    let n = camera.width * camera.height;
    let b = restir_plane(slot) + idx;
    // Both tags are stored biased by one so that a *zeroed* buffer — which is
    // what `reset_accumulation` and a fresh frame size leave behind — reads
    // back as "no sample, no surface" rather than as light 0 on material 0.
    var light = 0u;
    if s.valid && r.y_light != RESTIR_NONE {
        light = r.y_light + 1u;
    }
    var mat = 0u;
    if s.valid {
        mat = s.mat + 1u;
    }
    depth_normal_buffer[b] = vec4<f32>(r.y_pos, bitcast<f32>(light));
    depth_normal_buffer[b + n] =
        vec4<f32>(r.w_out, r.m, r.p_hat, bitcast<f32>(oct_encode(s.normal)));
    depth_normal_buffer[b + 2u * n] = vec4<f32>(s.point, bitcast<f32>(mat));
}

fn restir_load(slot: u32, idx: u32) -> Reservoir {
    let n = camera.width * camera.height;
    let b = restir_plane(slot) + idx;
    let a = depth_normal_buffer[b];
    let c = depth_normal_buffer[b + n];
    var r: Reservoir;
    r.y_pos = a.xyz;
    let tag = bitcast<u32>(a.w);
    r.y_light = select(tag - 1u, RESTIR_NONE, tag == 0u);
    r.w_out = c.x;
    r.m = c.y;
    r.p_hat = c.z;
    // Nothing downstream resamples out of a loaded reservoir's w_sum: a
    // reused one enters the next reservoir through W and M.
    r.w_sum = 0.0;
    return r;
}

fn restir_load_surface(slot: u32, idx: u32) -> RSurface {
    let n = camera.width * camera.height;
    let b = restir_plane(slot) + idx;
    let c = depth_normal_buffer[b + n];
    let d = depth_normal_buffer[b + 2u * n];
    var s: RSurface;
    s.normal = oct_decode(bitcast<u32>(c.w));
    s.point = d.xyz;
    let tag = bitcast<u32>(d.w);
    s.mat = select(tag - 1u, RESTIR_NONE, tag == 0u);
    s.valid = tag != 0u;
    return s;
}

// ── the light set, as one resampling domain ──────────────────────────────

// A sample resolved against a surface: the direction to it, what it emits,
// the geometry term, the source PDF the candidate was drawn with, and the
// solid-angle PDF the *BSDF-hits-an-emitter* branch would attribute to it —
// which is the one MIS has to use, or the two strategies' weights would not
// sum to one and the estimator would be biased.
struct RSample {
    wi: vec3<f32>,
    le: vec3<f32>,
    dist: f32,
    geom: f32,
    src_pdf: f32,
    mis_pdf: f32,
    ok: bool,
}

fn restir_resolve(p: vec3<f32>, y_pos: vec3<f32>, y_light: u32) -> RSample {
    var o: RSample;
    o.wi = vec3<f32>(0.0, 0.0, 1.0);
    o.le = vec3<f32>(0.0);
    o.dist = MAX_T;
    o.geom = 0.0;
    o.src_pdf = 0.0;
    o.mis_pdf = 0.0;
    o.ok = false;
    if y_light == RESTIR_NONE {
        return o;
    }
    if y_light >= render_state.light_count {
        return o;
    }
    let l = lights[y_light];
    let ln = light_normal(l);
    // Is the stored point still *on* this panel? This is what invalidates a
    // reservoir when the rig moves. Nothing else would: the panel keeps its
    // index and its emission, so a stale point in empty air resolves to a
    // perfectly plausible direction, distance and geometry term, and lights
    // the pixel from a panel that is no longer there. Re-evaluating p̂ every
    // frame only helps if p̂ can tell. In the test room, translating the
    // panels without this check left the picture 26% bright and it stayed
    // there; with it, three frames.
    let rel = y_pos - l.center.xyz;
    let ul = length(l.u.xyz);
    let vl = length(l.v.xyz);
    if ul <= 0.0 || vl <= 0.0 {
        return o;
    }
    let slack = 1e-3 * max(ul, vl) + 1e-5;
    if abs(dot(rel, l.u.xyz / ul)) > ul + slack
        || abs(dot(rel, l.v.xyz / vl)) > vl + slack
        || abs(dot(rel, ln)) > slack {
        return o;
    }
    let to_light = y_pos - p;
    let d = length(to_light);
    if d < 1e-9 {
        return o;
    }
    let wi = to_light / d;
    let cos_light = -dot(wi, ln);
    if cos_light <= 1e-9 {
        return o;
    }
    let area = light_area(l);
    if area <= 0.0 {
        return o;
    }
    o.wi = wi;
    o.le = l.emission.rgb;
    o.dist = d;
    o.geom = cos_light / (d * d);
    // Area measure: pick this panel by power, then a point on it uniformly.
    o.src_pdf = l.center.w / area;
    // Solid-angle measure — exactly the PDF `path_trace` reconstructs when a
    // BSDF ray lands on the panel, and MIS pairs the two.
    o.mis_pdf = l.center.w * d * d / (cos_light * area);
    o.ok = true;
    return o;
}

fn restir_material(mat: u32) -> GpuMaterial {
    if mat == FACE_IDX_GROUND {
        return ground_material();
    }
    return materials[mat];
}

fn restir_lum(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// p̂, the unshadowed target. Evaluated identically wherever it is needed —
// generation, temporal reuse, spatial reuse — which is what lets the weights
// cancel. It deliberately uses an isotropic shading frame (no dP/du) and no
// dispersion: p̂ only has to be positive wherever the true integrand is, and a
// frame that agrees with itself across passes matters more than one that
// agrees with the shading pass.
fn restir_target(s: RSurface, y_pos: vec3<f32>, y_light: u32) -> f32 {
    if !s.valid {
        return 0.0;
    }
    let rs = restir_resolve(s.point, y_pos, y_light);
    if !rs.ok {
        return 0.0;
    }
    let wo_world = normalize(camera.position.xyz - s.point);
    let frame = shading_frame(s.normal, vec3<f32>(0.0));
    let wo_local = to_local(frame, wo_world);
    let wi_local = to_local(frame, rs.wi);
    if wo_local.z <= 0.0 || wi_local.z <= 0.0 {
        return 0.0;
    }
    let e = bsdf_eval(restir_material(s.mat), wo_local, wi_local, 1.0, 0.0);
    return max(restir_lum(e.value * rs.le) * rs.geom, 0.0);
}

// One candidate, drawn from the same power table `pick_light` walks.
struct RCandidate {
    y_pos: vec3<f32>,
    y_light: u32,
    src_pdf: f32,
}

fn restir_candidate(pixel: vec2<u32>, seed: u32) -> RCandidate {
    var c: RCandidate;
    c.y_pos = vec3<f32>(0.0);
    c.y_light = RESTIR_NONE;
    c.src_pdf = 0.0;
    if render_state.light_count == 0u {
        return c;
    }
    let r1 = rand_uniform(pixel, seed * 4u + 1u);
    let r2 = rand_uniform(pixel, seed * 4u + 2u);
    let r3 = rand_uniform(pixel, seed * 4u + 3u);
    let idx = pick_light(r1);
    let l = lights[idx];
    if l.center.w <= 0.0 || light_area(l) <= 0.0 {
        return c;
    }
    c.y_light = idx;
    c.y_pos = l.center.xyz + l.u.xyz * (2.0 * r2 - 1.0) + l.v.xyz * (2.0 * r3 - 1.0);
    c.src_pdf = l.center.w / light_area(l);
    return c;
}

// ── reprojection and the similarity gates ────────────────────────────────

// Where `world_pos` sat in the *previous* pass's frame, or (-1, -1). Mirrors
// `world_to_screen_coords` against the camera the last pass was rendered
// from, which `ResidentScene` remembers on the caller's behalf.
fn restir_prev_screen(world_pos: vec3<f32>) -> vec2<i32> {
    if render_state.prev_cam_position.w == 0.0 {
        return vec2<i32>(-1, -1);
    }
    let eye = render_state.prev_cam_position.xyz;
    let forward = normalize(render_state.prev_cam_look_at.xyz - eye);
    let right = normalize(cross(forward, render_state.prev_cam_up.xyz));
    let up_cam = cross(right, forward);
    let fov_tan = tan(render_state.prev_cam_look_at.w * 0.5);
    let aspect = f32(camera.width) / f32(camera.height);
    let d = world_pos - eye;
    let view_z = dot(d, forward);
    if view_z <= 0.0 {
        return vec2<i32>(-1, -1);
    }
    let ndc_x = dot(d, right) / (view_z * fov_tan * aspect);
    let ndc_y = dot(d, up_cam) / (view_z * fov_tan);
    if abs(ndc_x) > 1.0 || abs(ndc_y) > 1.0 {
        return vec2<i32>(-1, -1);
    }
    return vec2<i32>(
        i32((ndc_x + 1.0) * 0.5 * f32(camera.width)),
        i32((1.0 - ndc_y) * 0.5 * f32(camera.height)),
    );
}

// The paper's geometric similarity: within 10% in depth and 25° in normal.
// Distance from *this* pass's eye on both sides, so a camera that moved
// compares like with like.
fn restir_similar(a: RSurface, b: RSurface) -> bool {
    if !a.valid || !b.valid || a.mat != b.mat {
        return false;
    }
    if dot(a.normal, b.normal) < RESTIR_COS_TOLERANCE {
        return false;
    }
    let da = length(a.point - camera.position.xyz);
    let db = length(b.point - camera.position.xyz);
    if da <= 0.0 {
        return false;
    }
    return abs(da - db) <= RESTIR_DEPTH_TOLERANCE * da;
}

// The primary hit as a reservoir's surface.
fn restir_primary_surface(pixel: vec2<u32>) -> RSurface {
    var s: RSurface;
    s.point = vec3<f32>(0.0);
    s.normal = vec3<f32>(0.0, 0.0, 1.0);
    s.mat = RESTIR_NONE;
    s.valid = false;

    let ray = ray_origin_and_direction(pixel);
    let origin = ray[0];
    let dir = ray[1];
    var hit = trace_scene(origin, dir);
    if render_state.ground_enabled != 0u {
        let g = intersect_ground(origin, dir);
        if g.t < hit.t {
            hit.t = g.t;
            hit.face_idx = FACE_IDX_GROUND;
            hit.uv = vec2<f32>(g.fade, 0.0);
        }
    }
    if hit.face_idx == FACE_IDX_MISS {
        return s;
    }
    // A panel in front of the surface owns the pixel; there is no BSDF there
    // to resample against.
    var lh = intersect_lights(origin, dir);
    if !camera_visible_lights() {
        lh.hit = false;
    }
    if lh.hit && lh.t < hit.t {
        return s;
    }
    s.point = origin + dir * hit.t;
    if hit.face_idx == FACE_IDX_GROUND {
        s.normal = vec3<f32>(0.0, 0.0, 1.0);
        s.mat = FACE_IDX_GROUND;
    } else {
        s.normal = hit_normal(hit);
        s.mat = hit_material_index(hit);
    }
    if dot(s.normal, -dir) < 0.0 {
        s.normal = -s.normal;
    }
    s.valid = true;
    return s;
}

// ── the shading pass's reader ────────────────────────────────────────────

// The direct term at the primary hit, from the final reservoir. Replaces
// `sample_lights` + `sample_sun` at depth 0 when ReSTIR is on; the
// environment's NEE and every deeper bounce are untouched.
//
// Unbiased in expectation for whatever the reservoir's `W` says, including
// the MIS weight, because E[g(y)·W] = ∫g for *any* g — so pairing it with the
// unchanged BSDF-hits-an-emitter branch still integrates the direct term once.
fn restir_direct(
    p: vec3<f32>,
    frame: mat3x3<f32>,
    n: vec3<f32>,
    wo_local: vec3<f32>,
    m: GpuMaterial,
    eta: f32,
    lambda_nm: f32,
    pixel: vec2<u32>,
) -> vec3<f32> {
    let idx = pixel_index(pixel);
    let r = restir_load(restir_read_slot(), idx);
    if r.y_light == RESTIR_NONE || r.w_out <= 0.0 {
        return vec3<f32>(0.0);
    }
    let rs = restir_resolve(p, r.y_pos, r.y_light);
    if !rs.ok {
        return vec3<f32>(0.0);
    }
    let wi_local = to_local(frame, rs.wi);
    if wi_local.z <= 0.0 {
        return vec3<f32>(0.0);
    }
    let e = bsdf_eval(m, wo_local, wi_local, eta, lambda_nm);
    if max3(e.value) <= 0.0 {
        return vec3<f32>(0.0);
    }
    // The frame's one shadow ray, spent on the sample M candidates and the
    // reuse agreed was worth testing.
    let tr = shadow_transmittance(offset_origin(p, n), rs.wi, rs.dist);
    if tr < 0.0 {
        return vec3<f32>(0.0);
    }
    let w = power_heuristic(rs.mis_pdf, e.pdf);
    return e.value * rs.le * (rs.geom * r.w_out * w * tr);
}

// ─── integrator ───────────────────────────────────────────────────────────

// Unidirectional path tracer: multi-bounce GI with throughput accumulation,
// next-event estimation against the area lights under MIS, and Russian
// roulette termination. This replaces the old single hardcoded GI bounce
// (`shade_direct` used as a bounce estimator) and the SSAO proxy — real
// multi-bounce transport computes contact occlusion correctly, so stacking
// SSAO on top of it would double-darken every concave corner.
//
// `max_depth` is driven per-frame by the refinement scheduler: draft frames
// trace shallow to stay interactive and depth escalates as accumulation
// proceeds, so the first frame is still usable.
// ─── subsurface: the random walk, on the device ───────────────────────────
//
// Chiang, Kutz and Burley 2016, the same walk `pathtrace::subsurface_walk`
// runs, against `trace_scene` instead of the CPU's BVH.
//
// One number differs and it is deliberate: the CPU walks up to 1024 scattering
// events and this walks 12. A GPU path loop is a uniform-control-flow budget
// shared by every lane in the workgroup, and a thousand-step inner loop makes
// the whole wave wait on the one pixel that landed on a bright medium. Twelve
// is enough for the short mean free paths these scenes actually use — a
// basketball's 2 mm rubber exits in three or four — and a medium bright enough
// to need more comes back darker here than on the CPU. That is the documented
// difference between the tiers; `subsurface = 0` is identical on both.
const SUBSURFACE_MAX_STEPS: u32 = 12u;

struct SssExit {
    ok: bool,
    point: vec3<f32>,
    normal: vec3<f32>,
    weight: vec3<f32>,
}

// Chiang's albedo inversion: the single-scattering albedo whose semi-infinite
// diffuse reflectance is `a`, for an isotropic phase function.
fn sss_scatter_albedo(a: f32) -> f32 {
    let x = clamp(a, 0.0, 1.0);
    return 1.0 - exp(-5.09406 * x + 2.61188 * x * x - 4.31805 * x * x * x);
}

fn subsurface_walk(
    m: GpuMaterial,
    entry: vec3<f32>,
    n: vec3<f32>,
    pixel: vec2<u32>,
    depth: u32,
) -> SssExit {
    var out: SssExit;
    out.ok = false;
    out.point = entry;
    out.normal = n;
    out.weight = vec3<f32>(0.0);

    let radius = m.subsurface_radius;
    if radius.x <= 0.0 || radius.y <= 0.0 || radius.z <= 0.0 {
        return out;
    }
    let sigma_t = vec3<f32>(1.0) / radius;
    let sigma_s = sigma_t
        * vec3<f32>(
            sss_scatter_albedo(m.subsurface_color.x),
            sss_scatter_albedo(m.subsurface_color.y),
            sss_scatter_albedo(m.subsurface_color.z),
        );

    // In through the surface, cosine-distributed about the inward normal.
    let salt = 1201u + depth * 97u;
    let e0 = rand_uniform2(pixel, salt);
    let inward = shading_frame(-n, vec3<f32>(0.0));
    var dir = to_world(inward, cosine_hemisphere_local(e0.x, e0.y));
    var pos = offset_origin(entry, -n);
    var weight = vec3<f32>(1.0);

    for (var step = 0u; step < SUBSURFACE_MAX_STEPS; step = step + 1u) {
        let rs = salt + 13u + step * 31u;
        let u = rand_uniform2(pixel, rs);
        // One channel drives the distance; the other two ride along under the
        // balance heuristic.
        let ch = min(u32(u.x * 3.0), 2u);
        let t = -log(max(1.0 - u.y, 1e-7)) / sigma_t[ch];

        let hit = trace_scene(pos, dir);
        var boundary = MAX_T;
        if hit.face_idx != 0xFFFFFFFFu {
            boundary = hit.t;
        }

        if boundary <= t {
            let e = exp(-sigma_t * boundary);
            let pdf = (e.x + e.y + e.z) / 3.0;
            if pdf <= 0.0 {
                return out;
            }
            weight = weight * e / pdf;
            var hn = hit_normal(hit);
            // Face the normal out of the medium: the side the ray was going.
            if dot(hn, dir) < 0.0 {
                hn = -hn;
            }
            out.ok = true;
            out.point = pos + dir * boundary;
            out.normal = hn;
            out.weight = weight;
            return out;
        }

        let e = exp(-sigma_t * t);
        let pdf = dot(sigma_t * e, vec3<f32>(1.0)) / 3.0;
        if pdf <= 0.0 {
            return out;
        }
        weight = weight * sigma_s * e / pdf;
        pos = pos + dir * t;
        // Isotropic phase function.
        let p = rand_uniform2(pixel, rs + 7u);
        let z = 1.0 - 2.0 * p.x;
        let r = sqrt(max(1.0 - z * z, 0.0));
        let phi = 2.0 * PI * p.y;
        dir = normalize(vec3<f32>(r * cos(phi), r * sin(phi), z));

        // Russian roulette on what is left.
        let q = clamp(max3(weight), 0.0, 1.0);
        if rand_uniform(pixel, rs + 11u) > q {
            return out;
        }
        weight = weight / q;
    }
    return out;
}

fn path_trace(first: RayHit, origin: vec3<f32>, dir: vec3<f32>, pixel: vec2<u32>) -> vec4<f32> {
    var l = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    var ray_o = origin;
    var ray_d = dir;
    var hit = first;
    // PDF of the lobe the previous bounce was sampled from; used to MIS
    // against light sampling when the new ray lands on an emitter.
    var prev_bsdf_pdf = 0.0;
    var specular_chain = true;
    var alpha = 0.0;
    // The path's hero wavelength in nanometres; <= 0 means "still RGB". A
    // scene with no dispersive material never leaves that state, draws no
    // extra random number, and renders exactly what it always did.
    var lambda_nm = 0.0;
    // The medium the path is inside, for Beer-Lambert absorption. One slot,
    // not a stack: nested dielectrics are out of scope, as on the CPU.
    var in_medium = false;
    var medium_sigma = vec3<f32>(0.0);

    let max_depth = max(render_state.max_depth, 1u);

    for (var depth = 0u; depth < max_depth; depth = depth + 1u) {
        // Lights are part of the scene: a BSDF ray that lands on one carries
        // its emission, MIS-weighted against the NEE sample that could also
        // have found it.
        //
        // Whether they are visible to CAMERA rays is the caller's choice. The
        // viewport's rig is sized to the scene bounds and the camera orbits
        // and zooms freely, so by default the softboxes are dropped at depth 0
        // rather than swinging through frame as giant white slabs. A scene
        // whose lights are part of the set — a room with its own ceiling
        // panels — wants the opposite, because `pathtrace::render` draws them
        // and the two tiers otherwise disagree by the whole emission wherever
        // a panel is in frame. `set_camera_visible_lights` picks.
        var lh = intersect_lights(ray_o, ray_d);
        if depth == 0u && !camera_visible_lights() {
            lh.hit = false;
        }
        let geom_t = select(MAX_T, hit.t, hit.face_idx != 0xFFFFFFFFu);

        // Absorb along the segment just travelled, if it was inside glass.
        if in_medium && max3(medium_sigma) > 0.0 {
            var seg = geom_t;
            if lh.hit && lh.t < geom_t {
                seg = lh.t;
            }
            if seg < MAX_T {
                throughput = throughput * exp(-medium_sigma * seg);
            }
        }

        if lh.hit && lh.t < geom_t {
            let light = lights[lh.index];
            var w = 1.0;
            if !specular_chain {
                let ln = light_normal(light);
                let cos_light = max(-dot(ray_d, ln), 1e-9);
                let light_pdf =
                    light.center.w * lh.t * lh.t / (cos_light * light_area(light));
                w = power_heuristic(prev_bsdf_pdf, light_pdf);
            }
            l += throughput * light.emission.rgb * w;
            if depth == 0u {
                alpha = 1.0;
            }
            break;
        }

        if hit.face_idx == 0xFFFFFFFFu {
            // Escaped the scene — pick up the environment, MIS-weighted
            // against environment NEE, which could also have found this
            // direction. A specular chain (including the primary ray) had no
            // other strategy, so it takes full weight; so does the gradient,
            // which is not importance-sampled.
            var w = 1.0;
            if !specular_chain && render_state.env_mode == ENV_MODE_IMAGE {
                let epdf = env_image_pdf(
                    ray_d,
                    render_state.env_width,
                    render_state.env_height,
                    render_state.env_rotation,
                    render_state.env_marg_int,
                );
                w = power_heuristic(prev_bsdf_pdf, epdf);
            }
            l += throughput * env_radiance(ray_d) * w;
            // The sun disc, if this ray landed in it. NEE samples the same
            // cone, so the two strategies share the direction under the
            // balance heuristic; a specular chain had no other way in.
            let sl = sun_radiance_in(ray_d);
            if max3(sl) > 0.0 {
                var ws = 1.0;
                if !specular_chain {
                    ws = power_heuristic(prev_bsdf_pdf, sun_pdf(ray_d));
                }
                l += throughput * sl * ws;
            }
            break;
        }

        if depth == 0u {
            alpha = 1.0;
        }

        // Resolve the surface we hit.
        var surf: Surface;
        surf.point = ray_o + ray_d * hit.t;
        var dpdu = vec3<f32>(0.0);
        if hit.face_idx == FACE_IDX_GROUND {
            surf.normal = vec3<f32>(0.0, 0.0, 1.0);
            surf.material = ground_material();
        } else {
            surf.normal = hit_normal(hit);
            surf.material = materials[hit_material_index(hit)];
            dpdu = hit_tangent(hit);
        }

        let wo_world = -ray_d;
        // Which side of the *geometric* normal the ray arrived on is the whole
        // of the inside/outside bookkeeping; read it before the face-forward
        // below destroys the distinction.
        let entering = dot(surf.normal, wo_world) >= 0.0;
        // Face-forward: interior faces (bore walls) must shade right.
        var n = surf.normal;
        if dot(n, wo_world) < 0.0 {
            n = -n;
        }
        // A dispersive material turns the path monochromatic, once.
        if lambda_nm <= 0.0 && mat_is_dispersive(surf.material) {
            lambda_nm = sample_lambda_nm(rand_uniform(pixel, 631u + depth * 17u));
            throughput = throughput * hero_weight(lambda_nm);
        }
        // n_transmitted / n_incident for this crossing.
        var eta = 1.0;
        if surf.material.transmission > 0.0 {
            let n_glass = max(mat_index_at(surf.material, lambda_nm), 1e-3);
            if surf.material.thin_walled != 0.0 || entering {
                eta = n_glass;
            } else {
                eta = 1.0 / n_glass;
            }
        }
        // Align the frame's x axis with dP/du so an anisotropic highlight
        // follows the surface's own grain, exactly as the CPU renderer does.
        let frame = shading_frame(n, dpdu);
        let wo_local = to_local(frame, wo_world);
        if wo_local.z <= 0.0 {
            break;
        }

        // Next-event estimation. At the primary hit, with ReSTIR on, the
        // panels come out of the pixel's reservoir instead — one
        // shadow ray already spent, on a light M candidates and two kinds of
        // reuse agreed was the one worth testing. Everything deeper keeps the
        // one-light NEE: a reservoir is a per-pixel object and there is no
        // pixel behind a second bounce.
        var direct = sample_environment(surf.point, frame, n, wo_local, surf.material, eta, lambda_nm, pixel, depth)
            + sample_sun(surf.point, frame, n, wo_local, surf.material, eta, lambda_nm, pixel, depth);
        if restir_on() && depth == 0u {
            direct += restir_direct(surf.point, frame, n, wo_local, surf.material, eta, lambda_nm, pixel);
        } else {
            direct += sample_lights(surf.point, frame, n, wo_local, surf.material, eta, lambda_nm, pixel, depth);
        }
        if depth > 0u && render_state.firefly_clamp > 0.0 {
            direct = min(direct, vec3<f32>(render_state.firefly_clamp));
        }
        l += throughput * direct;

        // Continue the path.
        let r_lobe = rand_uniform(pixel, 211u + depth * 23u);
        let r12 = rand_uniform2(pixel, 307u + depth * 29u);
        let r_branch = rand_uniform(pixel, 509u + depth * 37u);
        let s = bsdf_sample(surf.material, wo_local, eta, lambda_nm, r_lobe, r12.x, r12.y, r_branch);
        if !s.ok {
            break;
        }
        if s.sss {
            // The path leaves the surface entirely: into the object, walk,
            // and back out somewhere else. Everything after is about the exit.
            throughput = throughput * s.value / s.pdf;
            let ex = subsurface_walk(surf.material, surf.point, n, pixel, depth);
            if !ex.ok {
                break;
            }
            throughput = throughput * ex.weight;
            // Out through the boundary, cosine-distributed: the index-matched
            // exit the inversion was fitted with, whose f/pdf is exactly 1.
            let xf = shading_frame(ex.normal, vec3<f32>(0.0));
            let xe = rand_uniform2(pixel, 1607u + depth * 41u);
            ray_o = offset_origin(ex.point, ex.normal);
            ray_d = to_world(xf, cosine_hemisphere_local(xe.x, xe.y));
            // No NEE strategy found this direction, so an emitter downstream
            // takes full MIS weight.
            specular_chain = true;
            prev_bsdf_pdf = 0.0;
            if depth >= render_state.rr_start {
                let q = clamp(max3(throughput), 0.0, 0.95);
                if rand_uniform(pixel, 1609u + depth * 43u) > q {
                    break;
                }
                throughput = throughput / q;
            }
            if max3(throughput) <= 1e-5 {
                break;
            }
            continue;
        }
        throughput = throughput * s.value / s.pdf;
        prev_bsdf_pdf = s.pdf;
        specular_chain = false;

        // A transmitted ray leaves on the far side, so it is offset the other
        // way — and, for a solid, it changes which medium the path is in.
        let transmitted = s.wi.z < 0.0;
        if transmitted && surf.material.thin_walled == 0.0 {
            in_medium = entering;
            medium_sigma = mat_extinction(surf.material);
        }
        let wi_world = to_world(frame, s.wi);
        var off_n = n;
        if transmitted {
            off_n = -n;
        }
        ray_o = offset_origin(surf.point, off_n);
        ray_d = wi_world;

        // Russian roulette.
        if depth >= render_state.rr_start {
            let q = clamp(max3(throughput), 0.0, 0.95);
            if rand_uniform(pixel, 401u + depth * 31u) > q {
                break;
            }
            throughput = throughput / q;
        }
        if max3(throughput) <= 1e-5 {
            break;
        }

        // Trace the next segment.
        hit = trace_scene(ray_o, ray_d);
        if render_state.ground_enabled != 0u {
            let g = intersect_ground(ray_o, ray_d);
            if g.t < hit.t {
                hit.t = g.t;
                hit.face_idx = FACE_IDX_GROUND;
                hit.uv = vec2<f32>(g.fade, 0.0);
            }
        }
    }

    return vec4<f32>(l, alpha);
}

// Entry point used by the main pass. Returns LINEAR radiance in .rgb and
// coverage in .a; tonemapping happens once, after accumulation.
fn shade(hit: RayHit, origin: vec3<f32>, dir: vec3<f32>, pixel: vec2<u32>) -> vec4<f32> {
    if hit.face_idx == 0xFFFFFFFFu {
        // No geometry in front of the camera. A visible emitter is still in
        // front of it: `path_trace` is never entered on this branch, so the
        // camera-ray light test has to happen here too, or a panel against the
        // open sky would read as sky.
        if camera_visible_lights() {
            let lh = intersect_lights(origin, dir);
            if lh.hit {
                return vec4<f32>(lights[lh.index].emission.rgb, 1.0);
            }
        }
        // Draw the themed backdrop rather than the lighting environment — the
        // backdrop is a viewport choice, and `vcad-render` composites its own.
        return vec4<f32>(sky_color(dir), 0.0);
    }

    let traced = path_trace(hit, origin, dir, pixel);

    // The implicit ground plane is bounded: fade it into the backdrop with
    // horizontal distance so it reads as a platform under the model rather
    // than a disc floating in sky. `intersect_ground` stashes the fade factor
    // in uv.x. The CPU renderer uses an unbounded plane, so this only affects
    // pixels far from the subject.
    if hit.face_idx == FACE_IDX_GROUND {
        let fade = hit.uv.x;
        return vec4<f32>(mix(sky_color(dir), traced.rgb, fade), traced.a);
    }
    return traced;
}

// HSV to RGB conversion for debug visualization
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let h6 = h * 6.0;
    let x = c * (1.0 - abs(fract(h6 / 2.0) * 2.0 - 1.0));
    let m = v - c;

    var rgb: vec3<f32>;
    if h6 < 1.0 {
        rgb = vec3<f32>(c, x, 0.0);
    } else if h6 < 2.0 {
        rgb = vec3<f32>(x, c, 0.0);
    } else if h6 < 3.0 {
        rgb = vec3<f32>(0.0, c, x);
    } else if h6 < 4.0 {
        rgb = vec3<f32>(0.0, x, c);
    } else if h6 < 5.0 {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }
    return rgb + vec3<f32>(m, m, m);
}

// Compute depth and normal for a pixel
fn trace_depth_normal(pixel: vec2<u32>) -> vec4<f32> {
    let ray = ray_origin_and_direction(pixel);
    let origin = ray[0];
    let dir = ray[1];

    let hit = trace_scene(origin, dir);

    if hit.face_idx == 0xFFFFFFFFu {
        // Background: max depth, zero normal
        return vec4<f32>(0.0, 0.0, 0.0, MAX_T);
    }

    let normal = hit_normal(hit);
    return vec4<f32>(normal, hit.t);
}

// Edge-aware bilateral spatial filter for denoising stochastic noise
// (soft shadows, AO, GI bounce). Reads accum_buffer at neighbor pixels
// weighted by depth + normal similarity, returning a smoothed color
// that respects geometric edges.
//
// The filter strength scales DOWN with accumulation: at frame 1 (DRAFT)
// it provides aggressive smoothing of single-sample noise; at frame 24+
// (HIGH after settle) it has nearly no effect since the temporal average
// is already clean.
//
// Note: reads accum_buffer at neighbor positions which may have either
// "this frame's value" or "previous frame's value" depending on workgroup
// scheduling. In practice the neighbors' running averages are close
// enough that the bilateral converges correctly.
fn denoise(pixel_coord: vec2<i32>, center_color: vec4<f32>, center_depth_normal: vec4<f32>) -> vec4<f32> {
    let frame_idx = render_state.frame_index;
    if frame_idx > 32u {
        return center_color;
    }
    // Strength: 1.0 at frame 1, decays to ~0 by frame 32.
    let strength = clamp(1.0 - f32(frame_idx) / 32.0, 0.0, 1.0);

    let center_normal = center_depth_normal.xyz;
    let center_depth = center_depth_normal.w;
    let is_background = center_depth >= MAX_T - 1.0;

    // 5x5 box of neighbors (skip center, included separately).
    var sum = center_color.rgb;
    var weight_sum = 1.0;

    for (var dy: i32 = -2; dy <= 2; dy++) {
        for (var dx: i32 = -2; dx <= 2; dx++) {
            if dx == 0 && dy == 0 { continue; }

            let n_coord = pixel_coord + vec2<i32>(dx, dy);
            if n_coord.x < 0 || n_coord.x >= i32(camera.width) ||
               n_coord.y < 0 || n_coord.y >= i32(camera.height) {
                continue;
            }

            let n_dn = depth_normal_buffer[pixel_index_i32(n_coord)];
            let n_normal = n_dn.xyz;
            let n_depth = n_dn.w;
            let n_is_bg = n_depth >= MAX_T - 1.0;

            // Hard reject across silhouettes — never blur foreground into
            // background or vice versa.
            if is_background != n_is_bg { continue; }

            var w = 1.0;
            if !is_background {
                // Depth weight: similar depth → high weight.
                let depth_diff = abs(center_depth - n_depth) / max(center_depth, 0.1);
                w *= exp(-depth_diff * 8.0);

                // Normal weight: similar normal → high weight.
                if length(center_normal) > 0.5 && length(n_normal) > 0.5 {
                    let n_dot = max(dot(normalize(center_normal), normalize(n_normal)), 0.0);
                    w *= pow(n_dot, 6.0);
                }
            }

            // Spatial falloff (Gaussian-ish).
            let r2 = f32(dx * dx + dy * dy);
            w *= exp(-r2 * 0.25);

            let n_color = accum_buffer[pixel_index_i32(n_coord)].rgb;
            sum += n_color * w;
            weight_sum += w;
        }
    }

    let blurred = sum / weight_sum;
    return vec4<f32>(mix(center_color.rgb, blurred, strength), center_color.a);
}

// Sample depth_normal_buffer with coordinate clamped to image bounds.
fn sample_dn(offset: vec2<i32>, center: vec2<i32>) -> vec4<f32> {
    let c = clamp(center + offset,
                  vec2<i32>(0, 0),
                  vec2<i32>(i32(camera.width) - 1, i32(camera.height) - 1));
    return depth_normal_buffer[pixel_index_i32(c)];
}

// Sample feature_id_buffer with coordinate clamped to image bounds.
fn sample_fid(offset: vec2<i32>, center: vec2<i32>) -> u32 {
    let c = clamp(center + offset,
                  vec2<i32>(0, 0),
                  vec2<i32>(i32(camera.width) - 1, i32(camera.height) - 1));
    return feature_id_buffer[pixel_index_i32(c)];
}

// Detect Fusion-style edge lines using 3×3 Sobel on depth+normal plus analytic
// face-ID creases.  Returns vec3(silhouette, crease, boundary) strengths in [0,1].
//
// silhouette — large depth gradient (Sobel), catches diagonal edges without stair-stepping
// crease     — face_id changes between neighbours without a silhouette-level depth jump
// boundary   — foreground pixel adjacent to background (rendered both sides)
fn detect_edge_sobel(pixel_coord: vec2<i32>) -> vec3<f32> {
    // 3×3 neighbourhood samples (clamped at image borders)
    let p00 = sample_dn(vec2<i32>(-1,-1), pixel_coord);
    let p10 = sample_dn(vec2<i32>( 0,-1), pixel_coord);
    let p20 = sample_dn(vec2<i32>( 1,-1), pixel_coord);
    let p01 = sample_dn(vec2<i32>(-1, 0), pixel_coord);
    let p11 = sample_dn(vec2<i32>( 0, 0), pixel_coord); // center
    let p21 = sample_dn(vec2<i32>( 1, 0), pixel_coord);
    let p02 = sample_dn(vec2<i32>(-1, 1), pixel_coord);
    let p12 = sample_dn(vec2<i32>( 0, 1), pixel_coord);
    let p22 = sample_dn(vec2<i32>( 1, 1), pixel_coord);

    let center_depth = p11.w;
    let is_fg = center_depth < MAX_T - 1.0;

    // ---------- Sobel depth gradient (perspective-normalised) ----------
    let d00 = p00.w; let d10 = p10.w; let d20 = p20.w;
    let d01 = p01.w; let d21 = p21.w;
    let d02 = p02.w; let d12 = p12.w; let d22 = p22.w;

    let gx_d = -d00 + d20 - 2.0*d01 + 2.0*d21 - d02 + d22;
    let gy_d = -d00 - 2.0*d10 - d20 + d02 + 2.0*d12 + d22;
    let depth_grad = sqrt(gx_d*gx_d + gy_d*gy_d) / max(center_depth, 0.1);

    // ---------- Sobel normal gradient (sum of all three channels) ----------
    let n00 = p00.xyz; let n10 = p10.xyz; let n20 = p20.xyz;
    let n01 = p01.xyz; let n21 = p21.xyz;
    let n02 = p02.xyz; let n12 = p12.xyz; let n22 = p22.xyz;

    var normal_grad2 = 0.0;
    // Sobel on x channel
    let gx_nx = -n00.x + n20.x - 2.0*n01.x + 2.0*n21.x - n02.x + n22.x;
    let gy_nx = -n00.x - 2.0*n10.x - n20.x + n02.x + 2.0*n12.x + n22.x;
    // Sobel on y channel
    let gx_ny = -n00.y + n20.y - 2.0*n01.y + 2.0*n21.y - n02.y + n22.y;
    let gy_ny = -n00.y - 2.0*n10.y - n20.y + n02.y + 2.0*n12.y + n22.y;
    // Sobel on z channel
    let gx_nz = -n00.z + n20.z - 2.0*n01.z + 2.0*n21.z - n02.z + n22.z;
    let gy_nz = -n00.z - 2.0*n10.z - n20.z + n02.z + 2.0*n12.z + n22.z;
    normal_grad2 = gx_nx*gx_nx + gy_nx*gy_nx
                 + gx_ny*gx_ny + gy_ny*gy_ny
                 + gx_nz*gx_nz + gy_nz*gy_nz;
    let normal_grad = sqrt(normal_grad2);

    // ---------- Silhouette strength ----------
    let depth_threshold = render_state.edge_depth_threshold;
    var silhouette = 0.0;
    if depth_grad > depth_threshold {
        // Use the raw Sobel magnitude (relative to threshold) as AA sub-pixel distance.
        silhouette = clamp((depth_grad - depth_threshold) / depth_threshold, 0.0, 1.0);
    }
    // Add normal Sobel contribution for sharp curvature changes on a single surface.
    let normal_threshold_cos = cos(radians(render_state.edge_normal_threshold));
    let normal_edge_thresh = (1.0 - normal_threshold_cos) * 8.0; // scale to comparable range
    if normal_grad > normal_edge_thresh {
        let n_strength = clamp((normal_grad - normal_edge_thresh) / normal_edge_thresh, 0.0, 1.0);
        silhouette = max(silhouette, n_strength);
    }

    // ---------- Boundary (foreground ↔ background) ----------
    // Check 4-connected neighbours only; boundary pixels get full strength.
    var boundary = 0.0;
    let f01_d = p01.w; let f21_d = p21.w; let f10_d = p10.w; let f12_d = p12.w;
    let bg_t = MAX_T - 1.0;
    if is_fg {
        if f01_d > bg_t || f21_d > bg_t || f10_d > bg_t || f12_d > bg_t {
            boundary = 1.0;
        }
    } else {
        if f01_d < bg_t || f21_d < bg_t || f10_d < bg_t || f12_d < bg_t {
            boundary = 1.0;
        }
    }

    // ---------- Crease (analytic face-ID discontinuity) ----------
    // A crease exists when two adjacent foreground pixels belong to different faces
    // and there is no silhouette-level depth jump between them.
    var crease = 0.0;
    if is_fg {
        let center_fid = feature_id_buffer[pixel_index_i32(pixel_coord)];
        if center_fid != 0xFFFFFFFFu {
            let fid01 = sample_fid(vec2<i32>(-1, 0), pixel_coord);
            let fid21 = sample_fid(vec2<i32>( 1, 0), pixel_coord);
            let fid10 = sample_fid(vec2<i32>( 0,-1), pixel_coord);
            let fid12 = sample_fid(vec2<i32>( 0, 1), pixel_coord);
            let dn01 = p01.w; let dn21 = p21.w;
            let dn10 = p10.w; let dn12 = p12.w;
            let dd_max = depth_threshold * 2.0;

            if fid01 != 0xFFFFFFFFu && fid01 != center_fid
               && abs(center_depth - dn01) / max(center_depth, 0.1) < dd_max {
                crease = 1.0;
            }
            if fid21 != 0xFFFFFFFFu && fid21 != center_fid
               && abs(center_depth - dn21) / max(center_depth, 0.1) < dd_max {
                crease = max(crease, 1.0);
            }
            if fid10 != 0xFFFFFFFFu && fid10 != center_fid
               && abs(center_depth - dn10) / max(center_depth, 0.1) < dd_max {
                crease = max(crease, 1.0);
            }
            if fid12 != 0xFFFFFFFFu && fid12 != center_fid
               && abs(center_depth - dn12) / max(center_depth, 0.1) < dd_max {
                crease = max(crease, 1.0);
            }
        }
    }

    return vec3<f32>(silhouette, crease, boundary);
}

// Sample-count heatmap: t=0.0 → blue (1 sample/cold), t=1.0 → red (max samples/hot).
fn heat_color(t: f32) -> vec3<f32> {
    let tc = clamp(t, 0.0, 1.0);
    let r = clamp(tc * 2.0 - 1.0, 0.0, 1.0);
    let g = clamp(1.0 - abs(tc * 2.0 - 1.0), 0.0, 1.0);
    let b = clamp(1.0 - tc * 2.0, 0.0, 1.0);
    return vec3<f32>(r, g, b);
}

// Map an invocation to the pixel it owns, honouring the scissor. Returns the
// frame size in .zw so the caller can reject out-of-range invocations with one
// comparison.
fn scissor_pixel(global_id: vec3<u32>) -> vec4<u32> {
    if render_state.scissor_wh == 0u {
        return vec4<u32>(global_id.x, global_id.y, camera.width, camera.height);
    }
    let ox = render_state.scissor_xy & 0xFFFFu;
    let oy = render_state.scissor_xy >> 16u;
    let w = render_state.scissor_wh & 0xFFFFu;
    let h = render_state.scissor_wh >> 16u;
    // Outside the rectangle: hand back a coordinate the bounds check rejects.
    if global_id.x >= w || global_id.y >= h {
        return vec4<u32>(camera.width, camera.height, camera.width, camera.height);
    }
    return vec4<u32>(ox + global_id.x, oy + global_id.y, camera.width, camera.height);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = scissor_pixel(global_id).xy;

    if pixel.x >= camera.width || pixel.y >= camera.height {
        return;
    }

    let ray = ray_origin_and_direction(pixel);
    let origin = ray[0];
    let dir = ray[1];

    // Trace ray using BVH acceleration, then test the implicit ground
    // plane and pick whichever is closer.
    var hit = trace_scene(origin, dir);
    // `ground_enabled` has to be honoured here, not only on the shadow ray:
    // a scene that models its own floor (a room, a court) does not want a
    // second implicit one at z = 0 fighting it for the same pixels.
    if render_state.ground_enabled != 0u {
        let ground = intersect_ground(origin, dir);
        if ground.t < hit.t {
            hit.t = ground.t;
            hit.face_idx = FACE_IDX_GROUND;
            hit.uv = vec2<f32>(ground.fade, 0.0);
        }
    }
    let new_color = shade(hit, origin, dir, pixel);

    // Store depth and normal for edge detection. Ground hits get a normal
    // so silhouettes against the ground get drawn just like real faces.
    let pixel_coord = vec2<i32>(pixel);
    var depth_normal: vec4<f32>;
    if hit.face_idx == 0xFFFFFFFFu {
        depth_normal = vec4<f32>(0.0, 0.0, 0.0, MAX_T);
    } else if hit.face_idx == FACE_IDX_GROUND {
        depth_normal = vec4<f32>(0.0, 0.0, 1.0, hit.t);
    } else {
        let normal = hit_normal(hit);
        depth_normal = vec4<f32>(normal, hit.t);
    }

    // Write stable geometry data on the first frame so edge detection has
    // coherent neighbours on frame 2+ (same condition as depth_normal_buffer).
    // A raw-sample pass rewrites it every time: it is one independent sample,
    // not a step of an average, so nothing about the previous pass carries.
    if render_state.frame_index <= 1u || raw_sample_mode() {
        depth_normal_buffer[pixel_index_i32(pixel_coord)] = depth_normal;
        feature_id_buffer[pixel_index_i32(pixel_coord)] = hit.face_idx;
    }

    // Guide planes, in the CPU `Film`'s convention rather than this shader's:
    // the normal is face-forwarded against the view ray (what the path tracer
    // actually shades with) and the background sentinel for depth is 0, not
    // MAX_T.
    if raw_sample_mode() {
        let n_px = camera.width * camera.height;
        let gi = pixel_index_i32(pixel_coord);
        var g_normal = vec3<f32>(0.0, 0.0, 0.0);
        var g_depth = 0.0;
        var g_albedo = vec3<f32>(0.0, 0.0, 0.0);
        if hit.face_idx != 0xFFFFFFFFu {
            var gn: vec3<f32>;
            var gm: GpuMaterial;
            if hit.face_idx == FACE_IDX_GROUND {
                gn = vec3<f32>(0.0, 0.0, 1.0);
                gm = ground_material();
            } else {
                gn = hit_normal(hit);
                gm = materials[hit_material_index(hit)];
            }
            if dot(gn, -dir) < 0.0 {
                gn = -gn;
            }
            g_normal = gn;
            g_depth = hit.t;
            g_albedo = mix(mat_diffuse_albedo(gm), mat_f0(gm), gm.metallic);
        }
        depth_normal_buffer[n_px + gi] = vec4<f32>(g_normal, g_depth);
        // Guide plane 2's `.w` was spare. It carries the hit's *identity*
        // now — the geometry module's own primitive id, biased by one so 0
        // stays "nothing here" — which is what the history pass's temporal
        // reprojection validates against. Packed here rather than in a plane
        // of its own because the storage-buffer budget has no room for one.
        var g_id = 0.0;
        if hit.face_idx != 0xFFFFFFFFu {
            g_id = f32((hit.face_idx & 0x00FFFFFFu) + 1u);
        }
        depth_normal_buffer[2u * n_px + gi] = vec4<f32>(g_albedo, g_id);
    }

    // Progressive accumulation
    var accumulated: vec4<f32>;

    if render_state.frame_index <= 1u || raw_sample_mode() {
        // First frame, or a deliberate raw sample: start fresh.
        accumulated = new_color;
    } else {
        // Blend with previous samples using running average
        let prev = accum_buffer[pixel_index_i32(pixel_coord)];
        let weight = 1.0 / f32(render_state.frame_index);
        accumulated = mix(prev, new_color, weight);
    }

    // Spatial denoise — bilateral filter that smooths within similar
    // depth/normal regions, scaled down with accumulation count so it
    // mostly affects DRAFT/STANDARD tiers and fades out as HIGH settles.
    // Has to run before edge detection so edges are drawn on the
    // denoised image.
    var final_color = accumulated;
    if render_state.frame_index >= 2u && !raw_sample_mode() {
        let stored_dn = depth_normal_buffer[pixel_index_i32(pixel_coord)];
        final_color = denoise(pixel_coord, accumulated, stored_dn);
    }

    // The path tracer works in LINEAR radiance and accumulates in linear, so
    // tonemapping happens exactly once, here — after averaging and denoising.
    // (The old renderer tonemapped inside `shade`, which meant it averaged
    // already-compressed values and darkened highlights as frames piled up.)
    // Edge lines and debug overlays are authored in display space, so they are
    // composited after this point.
    final_color = vec4<f32>(
        pow(tonemap_aces(final_color.rgb), vec3<f32>(1.0 / 2.2)),
        final_color.a,
    );

    // Apply Fusion-style edge lines on later frames (stable depth/normal/face-ID data).
    // The Sobel edge overlay is a STYLISATION, not a shading term: it fights
    // photorealism, so it is gated behind `stylize` and stays off in a
    // photoreal viewport. enable_edges is then a bit-mask within that mode:
    // bit0=silhouette, bit1=crease, bit2=boundary.
    if render_state.stylize != 0u && render_state.enable_edges != 0u && render_state.frame_index >= 2u {
        let strengths = detect_edge_sobel(pixel_coord);

        // Silhouette lines (large depth/normal gradient, bit 0)
        if (render_state.enable_edges & 1u) != 0u && strengths.x > 0.001 {
            let s = clamp(strengths.x * render_state.silhouette_width * render_state.edge_softness,
                          0.0, 1.0);
            final_color = mix(final_color, render_state.silhouette_color, s);
        }

        // Crease lines (analytic face-ID boundary, bit 1)
        if (render_state.enable_edges & 2u) != 0u && strengths.y > 0.001 {
            let s = clamp(render_state.crease_width * render_state.edge_softness, 0.0, 1.0);
            final_color = mix(final_color, render_state.crease_color, s * strengths.y);
        }

        // Boundary lines (foreground↔background, bit 2 — highest priority)
        if (render_state.enable_edges & 4u) != 0u && strengths.z > 0.001 {
            let s = clamp(render_state.boundary_width * render_state.edge_softness, 0.0, 1.0);
            final_color = mix(final_color, render_state.boundary_color, s * strengths.z);
        }
    }

    // Apply debug visualization if enabled. Skip ground/miss sentinels
    // because hit_normal() would index a real geometry buffer.
    if render_state.debug_mode > 0u && render_state.debug_mode != 5u
        && hit.face_idx != 0xFFFFFFFFu
        && hit.face_idx != FACE_IDX_GROUND {
        let normal = hit_normal(hit);

        if render_state.debug_mode == 1u {
            // Normal visualization: map (-1,1) to (0,1) as RGB
            final_color = vec4<f32>((normal + 1.0) * 0.5, 1.0);
        } else if render_state.debug_mode == 2u {
            // Face ID as color (use HSV for distinct colors)
            let hue = fract(f32(hit.face_idx) * 0.15);
            let face_color = hsv_to_rgb(hue, 1.0, 1.0);
            final_color = vec4<f32>(face_color, 1.0);
        } else if render_state.debug_mode == 3u {
            // N dot L visualization (grayscale) using primary light direction
            let light_dir = normalize(vec3<f32>(0.5, 0.8, 0.3));
            let ndl = max(dot(normal, light_dir), 0.0);
            final_color = vec4<f32>(ndl, ndl, ndl, 1.0);
        } else if render_state.debug_mode == 4u {
            // Face orientation visualization: green=forward(0), red=reversed(1)
            if hit_orientation(hit) == 0u {
                final_color = vec4<f32>(0.2, 1.0, 0.2, 1.0);  // Green for forward
            } else {
                final_color = vec4<f32>(1.0, 0.2, 0.2, 1.0);  // Red for reversed
            }
        }
    }

    // Debug mode 5: sample-count heatmap. Main pass fires 1 ray per pixel
    // (blue = cold = minimum). The refine pass overwrites edge pixels with
    // the actual sample count after it runs.
    if render_state.debug_mode == 5u {
        final_color = vec4<f32>(heat_color(0.0), 1.0);
    }

    // Store accumulated color with sample count in alpha (1.0 from main pass).
    // The refine pass may update this for edge pixels. A raw-sample pass keeps
    // `shade`'s coverage instead: nothing is going to average it, and the host
    // reading the buffer back wants the same alpha the CPU `Film` carries.
    if !raw_sample_mode() {
        accumulated.a = 1.0;
    }

    // Store to accumulation buffer and output
    accum_buffer[pixel_index_i32(pixel_coord)] = accumulated;
    textureStore(output, pixel_coord, final_color);
}

// Adaptive refinement pass: fires additional stratified rays for edge pixels,
// blends with the coarse main-pass sample, and updates the output texture.
// Only runs when render_state.refine_sample_count > 0.
@compute @workgroup_size(8, 8)
fn refine(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = scissor_pixel(global_id).xy;

    if pixel.x >= camera.width || pixel.y >= camera.height {
        return;
    }
    if render_state.refine_sample_count == 0u {
        return;
    }

    let pixel_coord = vec2<i32>(pixel);
    let idx = pixel_index(pixel);

    // Detect edge strength from the depth/normal buffer (written by main pass).
    // Use the Sobel-based detector; take the max of silhouette, crease, boundary.
    let strengths = detect_edge_sobel(pixel_coord);
    let edge = max(strengths.x, max(strengths.y, strengths.z));

    // Only refine pixels on silhouettes / creases.
    if edge <= 0.1 {
        return;
    }

    // Stratified sub-pixel grid: grid_size x grid_size additional samples.
    let grid_size = u32(sqrt(f32(render_state.refine_sample_count)));
    if grid_size == 0u {
        return;
    }

    var color_sum = vec3<f32>(0.0);
    var fired = 0u;

    for (var sy = 0u; sy < grid_size; sy++) {
        for (var sx = 0u; sx < grid_size; sx++) {
            // Uniform stratified offset within the pixel, range [-0.5, 0.5].
            let offset = (vec2<f32>(f32(sx), f32(sy)) + 0.5) / f32(grid_size) - 0.5;
            let ray = ray_origin_and_direction_offset(pixel, offset);
            let origin = ray[0];
            let dir = ray[1];

            var hit = trace_scene(origin, dir);
            if render_state.ground_enabled != 0u {
                let ground = intersect_ground(origin, dir);
                if ground.t < hit.t {
                    hit.t = ground.t;
                    hit.face_idx = FACE_IDX_GROUND;
                    hit.uv = vec2<f32>(ground.fade, 0.0);
                }
            }

            color_sum += shade(hit, origin, dir, pixel).rgb;
            fired += 1u;
        }
    }

    // Blend: existing coarse sample (1 ray, alpha=1.0) + fired new rays.
    let existing = accum_buffer[idx];
    let total = f32(1u + fired);
    let blended_rgb = (existing.rgb + color_sum) / total;

    // Store back with sample count in alpha for heatmap debug mode.
    accum_buffer[idx] = vec4<f32>(blended_rgb, total);

    // Compose the refined output pixel. `shade` and the accumulation buffer are
    // both LINEAR, so tonemap here exactly as the main pass does — otherwise
    // refined edge pixels are written in a different colour space than their
    // neighbours and the overlay reads as a bright fringe.
    var final_color = vec4<f32>(
        pow(tonemap_aces(blended_rgb), vec3<f32>(1.0 / 2.2)),
        1.0,
    );

    // Re-apply edge overlay (only on later frames, consistent with main pass,
    // and only when stylisation is on).
    if render_state.stylize != 0u && render_state.enable_edges == 1u && render_state.frame_index >= 2u {
        let edge_color = vec4<f32>(0.1, 0.1, 0.12, 1.0);
        final_color = mix(final_color, edge_color, edge * 0.8);
    }

    // Debug mode 5: show actual sample count as heatmap (refine overwrites main-pass blue).
    if render_state.debug_mode == 5u {
        let t = clamp((total - 1.0) / f32(render_state.refine_sample_count), 0.0, 1.0);
        final_color = vec4<f32>(heat_color(t), 1.0);
    }

    textureStore(output, pixel_coord, final_color);
}

// ─── ReSTIR DI: the two resampling dispatches ─────────────────────────────

// Stage one. Trace the primary ray, draw M candidates against the unshadowed
// target, resample this pixel's reservoir from the previous frame through the
// previous camera, and spend the frame's single shadow ray on whatever
// survived.
//
// The temporal combination is the paper's *biased* one (Algorithm 4 without
// MIS weights) with the previous M clamped to 20x this frame's — the variant
// games ship, because the unbiased one needs a visibility ray per reused
// neighbour and the clamp is what keeps a reservoir responsive to a light
// that moved. What the clamp costs is documented in the README.
@compute @workgroup_size(8, 8)
fn restir_initial(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = scissor_pixel(global_id).xy;
    if pixel.x >= camera.width || pixel.y >= camera.height {
        return;
    }
    let idx = pixel_index(pixel);
    let surface = restir_primary_surface(pixel);
    var r = reservoir_empty();
    if !surface.valid {
        restir_store(restir_write_slot(), idx, r, surface);
        return;
    }

    // RIS over M candidates. No rays: p̂ is the unshadowed integrand.
    let m = restir_m();
    for (var i = 0u; i < m; i = i + 1u) {
        let c = restir_candidate(pixel, 4099u + i * 7u);
        var w = 0.0;
        var p_hat = 0.0;
        if c.src_pdf > 0.0 && c.y_light != RESTIR_NONE {
            p_hat = restir_target(surface, c.y_pos, c.y_light);
            w = p_hat / c.src_pdf;
        }
        reservoir_update(
            &r, c.y_pos, c.y_light, p_hat, w, 1.0,
            rand_uniform(pixel, 8191u + i * 3u),
        );
    }

    // Temporal reuse: this surface, where the previous pass had it.
    let prev_px = restir_prev_screen(surface.point);
    if prev_px.x >= 0 {
        let pidx = pixel_index_i32(prev_px);
        let prev_surf = restir_load_surface(restir_read_slot(), pidx);
        if restir_similar(surface, prev_surf) {
            let prev = restir_load(restir_read_slot(), pidx);
            let cap = render_state.restir_params.y * f32(m);
            let prev_m = min(prev.m, cap);
            var p_hat = 0.0;
            if prev.y_light != RESTIR_NONE {
                p_hat = restir_target(surface, prev.y_pos, prev.y_light);
            }
            // p̂ = 0 means the survivor cannot exist at this pixel at all —
            // the panel it names has moved out from under the point it
            // stored, or turned its back. Such a reservoir carries no
            // information here, so it is dropped whole: neither its weight
            // nor its M. Keeping the M and dropping only the weight would
            // divide this frame's sixteen candidates by three hundred and
            // twenty stale ones, and a moved light would black the room out
            // instead of relighting it.
            //
            // Note what is *not* dropped this way: an occluded sample. The
            // reservoir never learns about visibility — the frame's one
            // shadow ray is spent in `restir_direct` — so the temporal chain
            // is not conditioned on its survivors having been visible, which
            // is the other way to bias this and reads +4.5% on the test room.
            if p_hat > 0.0 {
                reservoir_update(
                    &r, prev.y_pos, prev.y_light, p_hat,
                    p_hat * prev.w_out * prev_m, prev_m,
                    rand_uniform(pixel, 20011u),
                );
            }
        }
    }

    reservoir_finalize(&r);

    // No shadow ray here. The reservoir stays a pure resampling of the
    // *unshadowed* target and the frame's one visibility test happens where
    // the sample is finally used, in `restir_direct`.
    //
    // The other arrangement — test the survivor now and zero its W — is what
    // the paper calls visibility reuse, and it is cheaper in noise and
    // dearer in truth: a reservoir that survives because it was visible, and
    // is reused because it survived, conditions the whole temporal chain on
    // visibility and reads a partly-shadowed room several percent bright.
    // Measured on the test room at M=16: +4.5%. Here it is 0.2%.

    restir_store(restir_write_slot(), idx, r, surface);
}

// Stage two, run `spatial_passes` times. Combine this pixel's reservoir with k
// neighbours drawn uniformly from a disc of `spatial_radius` pixels, gated by
// the same 10%-depth / 25°-normal similarity, re-evaluating each neighbour's
// sample against *this* surface's p̂.
//
// Biased again, and knowingly: the correct MIS weights would need a visibility
// ray per neighbour, and the whole point of the pass is that it costs none. A
// neighbour's sample that is visible there and occluded here leaks a little
// light. The total M is capped for the same reason the temporal one is.
@compute @workgroup_size(8, 8)
fn restir_spatial(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel = scissor_pixel(global_id).xy;
    if pixel.x >= camera.width || pixel.y >= camera.height {
        return;
    }
    let idx = pixel_index(pixel);
    let read = restir_read_slot();
    let surface = restir_load_surface(read, idx);
    var r = restir_load(read, idx);
    if !surface.valid {
        restir_store(restir_write_slot(), idx, reservoir_empty(), surface);
        return;
    }

    // Re-enter the reservoir as its own first candidate, in the same currency
    // every reused one arrives in: weight p̂·W·M.
    let own = r;
    r = reservoir_empty();
    if own.y_light != RESTIR_NONE {
        reservoir_update(
            &r, own.y_pos, own.y_light, own.p_hat,
            own.p_hat * own.w_out * own.m, own.m, 0.0,
        );
    } else {
        r.m = own.m;
    }

    let radius = render_state.restir_params.x;
    let k = min(u32(max(render_state.restir_params.z, 0.0)), RESTIR_MAX_NEIGHBOURS);
    let seed_base = 30011u + read * 977u;
    // The neighbours that actually contributed, kept so the survivor can be
    // normalised by the samples that could have produced it rather than by
    // every sample that was looked at. Slot 0 is this pixel.
    var contrib: array<u32, 9>;
    var contrib_m: array<f32, 9>;
    contrib[0] = idx;
    contrib_m[0] = own.m;
    var contrib_n = 1u;

    for (var i = 0u; i < k; i = i + 1u) {
        let u = rand_uniform2(pixel, seed_base + i * 5u);
        let ang = 2.0 * PI * u.x;
        let rad = radius * sqrt(u.y);
        let nx = i32(pixel.x) + i32(round(cos(ang) * rad));
        let ny = i32(pixel.y) + i32(round(sin(ang) * rad));
        if nx < 0 || ny < 0 || nx >= i32(camera.width) || ny >= i32(camera.height) {
            continue;
        }
        if nx == i32(pixel.x) && ny == i32(pixel.y) {
            continue;
        }
        let nidx = pixel_index_i32(vec2<i32>(nx, ny));
        let nsurf = restir_load_surface(read, nidx);
        if !restir_similar(surface, nsurf) {
            continue;
        }
        let nres = restir_load(read, nidx);
        var p_hat = 0.0;
        if nres.y_light != RESTIR_NONE {
            p_hat = restir_target(surface, nres.y_pos, nres.y_light);
        }
        reservoir_update(
            &r, nres.y_pos, nres.y_light, p_hat,
            p_hat * nres.w_out * nres.m, nres.m,
            rand_uniform(pixel, seed_base + 641u + i * 11u),
        );
        contrib[contrib_n] = nidx;
        contrib_m[contrib_n] = nres.m;
        contrib_n = contrib_n + 1u;
    }

    // Normalise by Z — the candidates that *could* have produced the survivor
    // — and not by M, the candidates that were looked at. This is Algorithm 6
    // of the paper minus its visibility ray: a neighbour whose own surface
    // cannot see the survivor's light at all (it faces away, or the panel is
    // edge-on to it) contributed nothing but would otherwise still swell the
    // denominator, and a frame normalised by M comes out several percent dark
    // — 10.4% on this test's room at two passes. What remains biased is the
    // visibility the pass does not test: a sample visible at the neighbour
    // and occluded here still lights this pixel. That is the shipped
    // trade — one shadow ray a pixel a frame, whatever the reuse.
    var z = 0.0;
    if r.y_light != RESTIR_NONE {
        for (var j = 0u; j < contrib_n; j = j + 1u) {
            let js = restir_load_surface(read, contrib[j]);
            if restir_target(js, r.y_pos, r.y_light) > 0.0 {
                z = z + contrib_m[j];
            }
        }
    }
    // Z becomes the reservoir's M. Everything downstream — the next spatial
    // pass, next frame's temporal reuse — re-enters a reservoir with weight
    // p̂·W·M, which equals the w_sum it actually accumulated only while
    // W = w_sum / (M·p̂) holds. Normalising by Z and storing the old M breaks
    // that invariant and every further pass multiplies the frame by M/Z
    // again: two spatial passes measured *worse* than none, and a 30 px
    // radius worse than plain NEE, until this line existed.
    r.m = z;

    // Then cap the confidence, rescaling w_sum with it so W is untouched. A
    // pass of k neighbours multiplies M by about k+1, so two passes a frame
    // would have a pixel claiming twenty-five times the samples it had, and
    // next frame's temporal clamp would shorten that M without shortening the
    // weight that came with it. Clamping *here*, proportionally, is the only
    // place it costs nothing: 2.37x quieter with the ragged clamp, 3.4x with
    // this one.
    let cap = render_state.restir_params.y * f32(restir_m());
    if r.m > cap && r.m > 0.0 {
        r.w_sum = r.w_sum * (cap / r.m);
        r.m = cap;
    }

    let denom = r.m * r.p_hat;
    if denom > 0.0 {
        r.w_out = r.w_sum / denom;
    } else {
        r.w_out = 0.0;
    }
    restir_store(restir_write_slot(), idx, r, surface);
}
