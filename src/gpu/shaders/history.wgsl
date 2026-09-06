//! Device-side per-pixel history and à-trous denoise.
//!
//! A host driving `render_resident_linear` gets one raw sample per pass and
//! has to do three things with it on the CPU: fold it into a running mean,
//! filter that mean, and tonemap the result. At 512x288 the filter alone costs
//! an order of magnitude more than the trace it is cleaning up. All three fit
//! in compute passes over buffers that already live on the device.
//!
//! Five entry points, run in this order once per pass:
//!
//! * `reproject` (optional) — carry each pixel's history across the frame's
//!   motion: unproject through this pass's depth, carry the world point back
//!   through its own instance's transform (`motion`), project into the
//!   previous view, and take the previous pixel's mean and count where the
//!   previous *id*, depth and normal agree that it is the same surface. A
//!   scene with nothing moving supplies no instances and this is the
//!   camera-only reprojection it used to be. The gather reads pixels other
//!   invocations would be writing, so it lands in the scratch pair rather
//!   than in `mean`/`stats`, and `accumulate` reads it from there — see
//!   `params.reprojected`.
//! * `accumulate` — fold `raw` into `mean`/`stats`, honouring the caller's
//!   per-pixel keep mask. A zero mask entry restarts that pixel's history.
//!   The fold is bounded (`history_cap`), so it is an exponential moving
//!   average rather than a mean that can no longer be moved, and a pixel
//!   whose history disagrees with what this pass's neighbourhood says it
//!   should be has that history *shortened* rather than trusted — which is
//!   what catches a shadow the ball has already left.
//! * `demodulate` — divide the mean by the albedo guide and prefilter the
//!   variance, writing `scratch_src`. A pixel with fewer than four samples
//!   gets SVGF's spatial estimate — a 7x7 luminance variance over its
//!   neighbours — because its own two temporal moments say nothing yet.
//! * `atrous` — one 5x5 B3-spline wavelet iteration at `params.stride`,
//!   reading `scratch_src` and writing `scratch_dst`. Dispatched once per
//!   iteration with the two scratch buffers swapped between them.
//! * `resolve` — re-modulate, blend by history length, tonemap, and store
//!   into the caller's texture.
//!
//! This is a port of `pathtrace::denoise`, filter weight for filter weight, so
//! a one-sample history denoises to what the CPU would have produced from the
//! same `Film`. `tests/gpu_history.rs` pins that.

const PI: f32 = 3.14159265359;

// Floor on the demodulation divisor. Mirrors `pathtrace::DEMOD_FLOOR`.
const DEMOD_FLOOR: f32 = 0.05;

// 5x5 separable B3-spline kernel, [1 4 6 4 1] / 16.
const B3_0: f32 = 0.0625;
const B3_1: f32 = 0.25;
const B3_2: f32 = 0.375;

struct HistoryParams {
    width: u32,
    height: u32,
    // History length at which the filter has fully faded out. A pixel with
    // this many samples is left exactly as the mean found it.
    count_cutoff: u32,
    // Number of à-trous iterations the host is about to dispatch. Zero means
    // "no filtering at all", and `resolve` then tonemaps the bare mean.
    iters: u32,
    sigma_lum: f32,
    sigma_depth: f32,
    sigma_normal: f32,
    exposure: f32,
    // Tap spacing for this à-trous iteration: 1, 2, 4, ...
    stride: u32,
    // `resolve` only: non-zero when the final iteration left its result in
    // `scratch_dst` rather than `scratch_src`.
    src_is_b: u32,
    // The trace pass's scissor rectangle, packed x | (y << 16) and
    // w | (h << 16), as `GpuRenderState` packs it. A zero size means the pass
    // covered the whole frame.
    //
    // `accumulate` needs it because a scissored trace only rewrites `raw`
    // inside the rectangle: outside it, `raw` still holds whatever the last
    // unscissored pass left there, and folding that in again would count one
    // sample as many and drag the mean towards it. Skipping those pixels
    // leaves their mean and count exactly as they were, which is what a viewer
    // tracing only the part of the frame that moved is asking for.
    scissor_xy: u32,
    scissor_wh: u32,

    // ─── `reproject` only ────────────────────────────────────────────────
    // Both views, as the ray generator in `raytrace.wgsl` builds them:
    // `.xyz` is eye / right / up / forward, and `view_params` is
    // (tan(fov/2), aspect) for the current view then the previous one.
    cur_eye: vec4<f32>,
    cur_right: vec4<f32>,
    cur_up: vec4<f32>,
    cur_forward: vec4<f32>,
    prev_eye: vec4<f32>,
    prev_right: vec4<f32>,
    prev_up: vec4<f32>,
    prev_forward: vec4<f32>,
    view_params: vec4<f32>,
    // Non-zero when `reproject` ran this pass, in which case `accumulate`
    // takes each pixel's history out of the scratch pair rather than out of
    // `mean`/`stats`. Folding the write-back into `accumulate` rather than
    // giving it a dispatch of its own saves a full-frame round trip.
    reprojected: u32,
    // Which à-trous iteration this slot drives, 0-based. `atrous` compares it
    // against the per-pixel iteration budget below.
    iter_index: u32,
    // The frame-space pixel this dispatch's (0, 0) invocation stands on.
    // `accumulate` dispatches over its scissor box's workgroups rather than
    // the frame's — a box worth a tenth of the frame is a tenth of the
    // workgroups, not a full grid that returns early nine times out of ten —
    // so its invocation ids are shifted onto the box's corner. Every other
    // pass covers the frame and leaves these zero.
    //
    // Explicit u32s, not a vec2<u32> pair: a vec3 in WGSL is 16-byte aligned
    // and would put this struct's size past the Rust one.
    origin_x: u32,
    origin_y: u32,

    // ─── temporal accumulation knobs ─────────────────────────────────────
    // Longest history a pixel may hold. Past it the fold is an exponential
    // moving average with a fixed 1/`history_cap` weight rather than a true
    // mean, so lighting that changed a hundred frames ago cannot outvote what
    // the pixel is seeing now.
    history_cap: u32,
    // Neighbourhood colour clamping, in standard deviations of this pass's
    // raw 3x3 neighbourhood. Zero turns it off.
    clamp_k: f32,
    // The history length a clamped pixel drops to. Not 1: a clamped pixel is
    // one whose history was *stale*, not one that has never been seen, and
    // restarting it from a single sample puts the grain back.
    clamp_reset: u32,
    // How many instances `motion` carries. Zero means every surface is
    // static and the reprojection is the camera-only one.
    motion_instances: u32,
    // How many entries the id -> instance table at the head of `motion` has.
    motion_ids: u32,
    // Non-zero to estimate a short-history pixel's variance spatially (SVGF's
    // 7x7 fallback) rather than from its own two temporal moments, which say
    // nothing until there are a few of them. Off reproduces
    // `pathtrace::denoise` exactly, which is what the parity test wants.
    spatial_variance: u32,

    // ─── gradient-directed sampling ──────────────────────────────────────
    // See `budget.wgsl`. Non-zero puts `accumulate` behind the per-pixel
    // budget: pixel p folds this round's sample with probability
    // b(p)/`budget_rounds`, and otherwise commits whatever the reprojection
    // carried and leaves the count alone. Zero is one sample everywhere,
    // which is what every caller that never asked for a budget gets.
    budget_enabled: u32,
    // 0 spends the frame's rays uniformly — today's behaviour exactly — and
    // 1 spends them entirely where the budget says. In between is the linear
    // blend of the two weight fields.
    budget_bias: f32,
    // How many trace rounds the host will dispatch this frame. The budget is
    // clamped to it, and it is the denominator of the per-round coin.
    budget_rounds: u32,
    // Which of those rounds `accumulate` is folding, 0-based. Part of the
    // coin's seed, so a pixel with b = 2 out of 4 rounds does not take the
    // same two rounds every frame.
    budget_round: u32,
    // The frame's total sample budget, in samples. `sum(b) == rays_per_frame`
    // to within the clamp; a frame's uniform spend is width * height.
    rays_per_frame: f32,
    // How far the motion drive is dilated, in pixels. The à-trous filter's
    // footprint: a moved object drags everything within it.
    budget_radius: u32,
    // Every pixel is guaranteed one sample once every this many frames,
    // whatever its budget. 1 is "every pixel every frame", which turns the
    // starvation floor into no budget at all.
    budget_floor_k: u32,
    // The frame counter the floor's phase and the coin's seed ride on.
    budget_frame: u32,
    // Out to a multiple of 16 bytes, so the Rust struct and this one agree.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
}

// How far this frame's surface point may lie off the plane the previous
// frame's surface point sat in, as a fraction of the distance to it.
//
// Measured along the *normal*, not along the ray. A room seen from inside is
// mostly grazing — the floor, the ceiling and the side walls all run away
// from the eye — and on a grazing surface the depth changes by far more than
// 2% across a single pixel, so a plain depth-ratio test throws away a third
// of a frame that has not moved. Along the normal there is no such slope:
// the same wall reads the same distance however obliquely it is seen, and a
// disocclusion still reads as the whole gap between the two surfaces.
const REPROJ_DEPTH_TOL: f32 = 0.02;

// How closely the two frames' normals must agree to be the same surface.
const REPROJ_NORMAL_DOT: f32 = 0.9;

@group(0) @binding(0) var<uniform> params: HistoryParams;
// This pass's own raw linear sample: the ray tracer's accumulation buffer
// after a `raw_sample` pass, which is (radiance, coverage).
@group(0) @binding(1) var<storage, read> raw: array<vec4<f32>>;
// The resident depth/normal buffer's planes. Plane 1 is (face-forwarded
// normal, distance from the eye) and plane 2 is (denoise albedo, 0), both in
// the CPU `Film`'s conventions — background depth is 0, not MAX_T. Plane 3 is
// the sample budget's per-pixel selection mask, which is the one thing here
// this shader *writes*: see `budget_select` in `budget.wgsl`.
@group(0) @binding(2) var<storage, read_write> guides: array<vec4<f32>>;
// Running mean: (linear radiance, coverage).
@group(0) @binding(3) var<storage, read_write> mean: array<vec4<f32>>;
// (count, luminance sum, luminance-squared sum, variance of the mean).
@group(0) @binding(4) var<storage, read_write> stats: array<vec4<f32>>;
// One entry per pixel: 0 restarts that pixel's history, non-zero keeps it.
@group(0) @binding(5) var<storage, read> keep: array<u32>;
// (illumination, variance) ping-pong for the wavelet iterations.
@group(0) @binding(6) var<storage, read_write> scratch_src: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> scratch_dst: array<vec4<f32>>;
@group(0) @binding(8) var out_tex: texture_storage_2d<rgba8unorm, write>;
// The previous pass's guide plane 1 — (face-forwarded normal, distance from
// the previous eye) — one vec4 per pixel, copied out of `guides` at the end
// of the pass that wrote it. Zeroed depth means "the previous pass had
// nothing there", which reads as a restart.
// The previous pass's guide planes 1 and 2 — (face-forwarded normal, distance
// from the previous eye) for `n` pixels, then (albedo, biased hit id) for `n`
// more — copied out of `guides` at the end of the pass that wrote them.
// Zeroed depth means "the previous pass had nothing there", which reads as a
// restart.
@group(0) @binding(9) var<storage, read> prev_guides: array<vec4<f32>>;
// Object motion, packed by `gpu::history::InstanceMotion`:
//
//   [0 .. ceil(motion_ids/4))   the id -> instance slot table, four bitcast
//                               u32s to a vec4, indexed by the *unbiased*
//                               hit id. 0xFFFFFFFF means "static".
//   then three vec4s per instance: the rows of the 3x4 matrix
//   `prev_T · cur_T⁻¹`, which takes a point where this frame put it to
//   where the previous frame had it.
//
// A frame with nothing moving binds a stub and sets `motion_instances` to 0.
@group(0) @binding(10) var<storage, read> motion: array<vec4<f32>>;

// ─── the split denoiser's own buffers ─────────────────────────────────────
// Two planes each, diffuse then specular, in `mean`/`stats`' layouts: the
// running mean of each lobe's history and its statistics. One departure:
// the specular mean's `.w` is not coverage — the diffuse plane has that —
// but the running mean of the reflection's distance, which the specular
// reprojection steers by. The summed `mean`/`stats` above keep going
// exactly as they did, so the neural path and every existing reader see
// nothing new.
@group(0) @binding(14) var<storage, read_write> lobe_mean: array<vec4<f32>>;
@group(0) @binding(15) var<storage, read_write> lobe_stats: array<vec4<f32>>;
// Where `reproject_lobes` gathers to: four planes, (diffuse mean, diffuse
// stats, specular mean, specular stats), read back by `accumulate_lobes`
// exactly as `accumulate` reads the scratch pair.
@group(0) @binding(16) var<storage, read_write> lobe_gather: array<vec4<f32>>;

fn luminance(c: vec3<f32>) -> f32 {
    return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
}

fn n_pixels() -> u32 {
    return params.width * params.height;
}

fn guide_depth(i: u32) -> f32 {
    return guides[n_pixels() + i].w;
}

fn guide_normal(i: u32) -> vec3<f32> {
    return guides[n_pixels() + i].xyz;
}

fn guide_albedo(i: u32) -> vec3<f32> {
    return guides[2u * n_pixels() + i].xyz;
}

// The hit's identity at pixel `i`, biased by one so 0 is "nothing here".
// Written into guide plane 2's spare lane by the integrator.
fn guide_id(i: u32) -> u32 {
    return u32(max(guides[2u * n_pixels() + i].w, 0.0));
}

// The same, off the previous pass's copy.
fn prev_guide_id(j: u32) -> u32 {
    return u32(max(prev_guides[n_pixels() + j].w, 0.0));
}

// ─── object motion ────────────────────────────────────────────────────────

// vec4s the id -> instance table occupies at the head of `motion`.
fn motion_table_len() -> u32 {
    return (params.motion_ids + 3u) / 4u;
}

// Which instance the (unbiased) hit id `id` belongs to, or 0xFFFFFFFF for a
// surface with no motion of its own.
fn instance_of(id: u32) -> u32 {
    if id >= params.motion_ids {
        return 0xFFFFFFFFu;
    }
    let v = motion[id / 4u];
    let lane = id % 4u;
    var f = v.x;
    if lane == 1u {
        f = v.y;
    } else if lane == 2u {
        f = v.z;
    } else if lane == 3u {
        f = v.w;
    }
    return bitcast<u32>(f);
}

// Where instance `inst` had the world point `p` on the previous frame.
fn motion_point(inst: u32, p: vec3<f32>) -> vec3<f32> {
    let b = motion_table_len() + inst * 3u;
    let r0 = motion[b];
    let r1 = motion[b + 1u];
    let r2 = motion[b + 2u];
    return vec3<f32>(
        dot(r0.xyz, p) + r0.w,
        dot(r1.xyz, p) + r1.w,
        dot(r2.xyz, p) + r2.w,
    );
}

// The same matrix's rotation acting on a direction, for carrying the normal
// into the previous frame so the normal gate compares like with like.
fn motion_dir(inst: u32, d: vec3<f32>) -> vec3<f32> {
    let b = motion_table_len() + inst * 3u;
    let r0 = motion[b];
    let r1 = motion[b + 1u];
    let r2 = motion[b + 2u];
    let v = vec3<f32>(dot(r0.xyz, d), dot(r1.xyz, d), dot(r2.xyz, d));
    if length(v) < 1e-12 {
        return d;
    }
    return normalize(v);
}

// The demodulation divisor, per channel and floored, as `pathtrace::denoise`
// applies it on the way in and the way out.
fn demod_albedo(i: u32) -> vec3<f32> {
    return max(guide_albedo(i), vec3<f32>(DEMOD_FLOOR));
}

// The scalar the demodulated variance is divided by: the luminance of the
// floored albedo, itself floored.
fn demod_lum(i: u32) -> f32 {
    return max(luminance(demod_albedo(i)), DEMOD_FLOOR);
}

fn in_bounds(gid: vec3<u32>) -> bool {
    return gid.x < params.width && gid.y < params.height;
}

fn flat_index(gid: vec3<u32>) -> u32 {
    return gid.y * params.width + gid.x;
}

// Whether this pixel is one the trace pass just wrote a fresh sample for.
fn in_scissor(gid: vec3<u32>) -> bool {
    if params.scissor_wh == 0u {
        return true;
    }
    let ox = params.scissor_xy & 0xFFFFu;
    let oy = params.scissor_xy >> 16u;
    let w = params.scissor_wh & 0xFFFFu;
    let h = params.scissor_wh >> 16u;
    return gid.x >= ox && gid.x < ox + w && gid.y >= oy && gid.y < oy + h;
}

// ─── pass 0: carry the history across a camera move ───────────────────────
//
// Without this, a camera move can only be expressed through the keep mask,
// and a host with no reprojection of its own uploads an all-restart mask: one
// orbit throws the whole frame away and every pixel starts again from a
// single sample. Most of those pixels are the same surface seen from a hair
// to the left.
//
// The test is deliberately conservative — nearest tap, no bilinear blend, and
// both a depth and a normal gate — because a history carried onto the wrong
// surface does not look like noise. It looks like the previous frame smeared
// across this one, and it takes `count_cutoff` samples to wash out.

// The primary ray direction for a pixel centre under one view, matching
// `ray_origin_and_direction_offset` in `raytrace.wgsl` with a zero offset.
fn view_ray(right: vec3<f32>, up: vec3<f32>, fwd: vec3<f32>, tan_fov: f32, aspect: f32, x: u32, y: u32) -> vec3<f32> {
    let ndc = vec2<f32>(
        (f32(x) + 0.5) / f32(params.width) * 2.0 - 1.0,
        1.0 - (f32(y) + 0.5) / f32(params.height) * 2.0
    );
    return normalize(fwd + right * ndc.x * tan_fov * aspect + up * ndc.y * tan_fov);
}

@compute @workgroup_size(8, 8)
fn reproject(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !in_bounds(gid) {
        return;
    }
    let i = flat_index(gid);
    // Whatever happens, this pixel's slot in the scratch pair is written, so
    // `reproject_commit` never copies a stale gather back into the history.
    scratch_src[i] = vec4<f32>(0.0);
    scratch_dst[i] = vec4<f32>(0.0);

    let depth = guide_depth(i);
    if depth <= 0.0 {
        return; // background: nothing to carry
    }

    // Where this pixel's surface is, in world space.
    let dir = view_ray(
        params.cur_right.xyz, params.cur_up.xyz, params.cur_forward.xyz,
        params.view_params.x, params.view_params.y, gid.x, gid.y,
    );
    let p_now = params.cur_eye.xyz + depth * dir;

    // Where the *instance* this pixel is looking at had that point last
    // frame. A ball in flight moves with its own transform; a static surface
    // has none and this is the identity, which is the camera-only
    // reprojection this pass used to be.
    let id = guide_id(i);
    var inst = 0xFFFFFFFFu;
    if params.motion_instances > 0u && id > 0u {
        let cand = instance_of(id - 1u);
        if cand < params.motion_instances {
            inst = cand;
        }
    }
    var p = p_now;
    var n_now = guide_normal(i);
    if inst != 0xFFFFFFFFu {
        p = motion_point(inst, p_now);
        n_now = motion_dir(inst, n_now);
    }

    // ... and where it was on the previous frame's film.
    let v = p - params.prev_eye.xyz;
    let z = dot(v, params.prev_forward.xyz);
    if z <= 0.0 {
        return; // behind the previous eye
    }
    let tan_fov = params.view_params.z;
    let aspect = params.view_params.w;
    let ndc_x = dot(v, params.prev_right.xyz) / (z * tan_fov * aspect);
    let ndc_y = dot(v, params.prev_up.xyz) / (z * tan_fov);
    let fx = (ndc_x + 1.0) * 0.5 * f32(params.width) - 0.5;
    let fy = (1.0 - ndc_y) * 0.5 * f32(params.height) - 0.5;
    let qx = i32(round(fx));
    let qy = i32(round(fy));
    if qx < 0 || qy < 0 || qx >= i32(params.width) || qy >= i32(params.height) {
        return; // it was off the previous frame
    }
    let j = u32(qy) * params.width + u32(qx);

    // Was the previous frame looking at *this* surface, or at something in
    // front of it? Reconstruct the point it had there and ask how far this
    // frame's point lies off its tangent plane.
    let prev_depth = prev_guides[j].w;
    if prev_depth <= 0.0 {
        return;
    }
    // The strongest gate there is: was the previous frame looking at the same
    // *object*? A ball crossing the floor and the floor behind it agree on
    // depth to well inside the tolerance for a frame or two, and disagree
    // here on every one of them.
    if prev_guide_id(j) != id {
        return;
    }
    let prev_n = prev_guides[j].xyz;
    let prev_dir = view_ray(
        params.prev_right.xyz, params.prev_up.xyz, params.prev_forward.xyz,
        tan_fov, aspect, u32(qx), u32(qy),
    );
    let q = params.prev_eye.xyz + prev_depth * prev_dir;
    let expected = length(v);
    if abs(dot(p - q, prev_n)) > REPROJ_DEPTH_TOL * expected {
        return; // disoccluded: something else was there
    }
    if dot(n_now, prev_n) < REPROJ_NORMAL_DOT {
        return; // the same plane, a different surface — a silhouette edge
    }

    scratch_src[i] = mean[j];
    scratch_dst[i] = stats[j];
}

// ─── pass 1: fold this sample into the history ────────────────────────────

@compute @workgroup_size(8, 8)
fn accumulate(@builtin(global_invocation_id) lid: vec3<u32>) {
    // The dispatch covers the scissor box's workgroups, not the frame's, so
    // the invocation id is box-relative. Everything below wants frame
    // coordinates.
    let gid = vec3<u32>(lid.x + params.origin_x, lid.y + params.origin_y, lid.z);
    if !in_bounds(gid) {
        return;
    }
    // The box is not a whole number of workgroups, and outside the trace
    // pass's scissor there is no new sample to fold in either way.
    if !in_scissor(gid) {
        return;
    }
    let i = flat_index(gid);
    let c = raw[i];
    let l = luminance(c.rgb);

    // Where this pixel's history is: in the buffers, or — if `reproject` ran
    // this pass — in the scratch pair it gathered into.
    var m = mean[i];
    var st = stats[i];
    if params.reprojected != 0u {
        m = scratch_src[i];
        st = scratch_dst[i];
    }
    if keep[i] == 0u {
        m = vec4<f32>(0.0);
        st = vec4<f32>(0.0);
    }

    // ─── the sample budget ───────────────────────────────────────────────
    //
    // Whether this pixel got one of this frame's rays at all. `budget.wgsl`
    // decided that before the trace ran, from the physics, the history and
    // the last image, and wrote the answer where the trace could read it —
    // so a pixel with b = 2 out of 4 rounds was traced on two of them and a
    // converged, static one was not traced at all.
    //
    // The selection is independent of what the sample turned out to be, so
    // the mean over the folded samples is still an unbiased estimate of the
    // pixel. That is the whole reason to select rather than weight.
    //
    // A skipped pixel is not simply abandoned: whatever the reprojection
    // carried for it still has to be committed, or a pixel that skips every
    // round of a frame silently loses the history the camera move carried
    // onto it.
    if params.budget_enabled != 0u {
        // Which of this frame's rounds this pixel takes.
        //
        // Not a coin. A coin at probability b/rounds folds b samples in
        // expectation and is unbiased in the *mean*, which is the only thing
        // an unbiasedness argument covers — but a pixel's error goes as
        // 1/sqrt(count), and 1/sqrt is convex, so a count that scatters around
        // b is worse than a count that is b. Measured on a converged frame
        // with one ball crossing it, the coin gave up a third of what the
        // whole feature was buying.
        //
        // So the rounds are *stratified*. Walk the interval [u, u + b) in
        // steps of b/rounds and take a round each time the walk crosses an
        // integer: the pixel takes exactly floor(b) or ceil(b) rounds, the
        // fractional part decided by a per-pixel dither u so that the
        // expectation is still exactly b. Unbiased, and the count never
        // strays by more than one.
        //
        // Decided before the trace, by `budget_select`, and read back here
        // rather than re-derived: the trace skipped the pixels this mask does
        // not name, so a second derivation that disagreed by one float would
        // fold a sample that was never taken.
        if (budget_mask_load(i) & (1u << params.budget_round)) == 0u {
            // Skipped — but not abandoned. Whatever the reprojection carried
            // for this pixel, and a zeroed keep entry, are decisions this pass
            // still has to commit: a pixel that skips every round of a frame
            // would otherwise silently lose the history the camera move
            // carried onto it. Where neither applies, `m` and `st` are what is
            // already there and this writes them back unchanged.
            mean[i] = m;
            stats[i] = st;
            return;
        }
    }

    // ─── neighbourhood clamping ──────────────────────────────────────────
    //
    // A carried history can be *stale* without being wrong about which
    // surface it is on: the ball has moved off the floor and the floor is
    // still holding the ball's shadow. Nothing in the depth/normal/id gates
    // sees that, because the floor is still the floor.
    //
    // What does see it is this pass's own sample. Clamp the history into the
    // range the raw 3x3 neighbourhood says the pixel plausibly is, and a
    // shadow that has gone is out of range on the first frame and reeled in
    // over the next few. The clamped pixel's history length drops with it, so
    // the à-trous filter widens there and the reeling-in is not visible as
    // noise.
    if params.clamp_k > 0.0 && st.x > 0.5 {
        var s1 = vec3<f32>(0.0);
        var s2 = vec3<f32>(0.0);
        var hsum = vec3<f32>(0.0);
        var k = 0.0;
        let x = i32(gid.x);
        let y = i32(gid.y);
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            let qy = y + dy;
            if qy < 0 || qy >= i32(params.height) {
                continue;
            }
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let qx = x + dx;
                if qx < 0 || qx >= i32(params.width) {
                    continue;
                }
                let q = u32(qy) * params.width + u32(qx);
                let t = raw[q].rgb;
                s1 = s1 + t;
                s2 = s2 + t * t;
                hsum = hsum + mean[q].rgb;
                k = k + 1.0;
            }
        }
        if k > 0.0 {
            let mu = s1 / k;
            let sd = sqrt(max(s2 / k - mu * mu, vec3<f32>(0.0)));
            // What is compared is the *history's* 3x3 mean against the raw
            // sample's 3x3 mean, not the history's own pixel against the raw
            // neighbourhood. The two are blurred by the same kernel, so the
            // spatial bias that makes a plain TAA clamp fire on every gradient
            // in the frame cancels, and what is left is the error bar on a
            // nine-sample mean — three times tighter than one sample's, which
            // is three times the sensitivity to lighting that really did
            // change.
            let hmu = hsum / k;
            let tol = params.clamp_k * sd / sqrt(k) + 1e-5;
            let d = abs(hmu - mu);
            if max(d.x, max(d.y, d.z)) > max(tol.x, max(tol.y, tol.z)) {
                // The history is *shortened*, not overwritten. Snapping the
                // colour to the edge of the neighbourhood would be the usual
                // TAA move and it is a biased one: on a still frame the
                // clamp fires by chance a few percent of the time, always
                // pulling towards a nine-sample mean, and a hundred passes of
                // that is a visible tint. Shortening the history is unbiased —
                // the next few samples are simply worth much more — and the
                // stale value is gone in about `clamp_reset` frames either
                // way.
                st.x = min(st.x, f32(max(params.clamp_reset, 1u)));
            }
            // `mean[q]` for a neighbour is read while other invocations of
            // this same dispatch may be folding their own sample into it. The
            // race is deliberate and harmless: what comes back is that
            // neighbour's history either side of one sample out of `n`, and
            // the test is a comparison of nine-pixel means against an error
            // bar. Synchronising it would cost a full-frame dispatch to buy a
            // difference under the noise floor.
        }
    }

    // ─── the bounded fold ────────────────────────────────────────────────
    //
    // A true running mean over an unbounded history is the right estimator
    // for a still picture and the wrong one for a live scene: at n = 400 a
    // pixel that has just been lit differently moves a quarter of a percent
    // a frame. Cap n and the fold becomes an exponential moving average with
    // a floor on its weight, which converges just as far and then keeps up.
    let cap = f32(max(params.history_cap, 1u));
    let n = min(st.x + 1.0, cap);
    let inv = 1.0 / n;
    m = m + (c - m) * inv;
    // The luminance moments are means, not sums, so they ride the same cap.
    let mu1 = st.y + (l - st.y) * inv;
    let mu2 = st.z + (l * l - st.z) * inv;

    // Variance of the *mean*, matching `pathtrace::trace_pixel`: sample
    // variance over n, with a single sample falling back to its own magnitude
    // because it says nothing about its own spread.
    var v: f32;
    if n > 1.5 {
        let sample_var = max(mu2 - mu1 * mu1, 0.0) * n / (n - 1.0);
        v = sample_var * inv;
    } else {
        v = mu1 * mu1;
    }

    mean[i] = m;
    stats[i] = vec4<f32>(n, mu1, mu2, v);
}

// ─── pass 2: demodulate and prefilter the variance ────────────────────────

@compute @workgroup_size(8, 8)
fn demodulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !in_bounds(gid) {
        return;
    }
    let i = flat_index(gid);
    let illum = mean[i].rgb / demod_albedo(i);

    // A 3x3 box over the demodulated variance. The per-pixel estimate is
    // itself noisy at low sample counts, and a noisy error bar makes the
    // luminance weight jitter between "trust" and "reject" pixel to pixel.
    // Background taps are excluded, exactly as the CPU filter excludes them.
    var s = 0.0;
    var k = 0.0;
    let x = i32(gid.x);
    let y = i32(gid.y);
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        let qy = y + dy;
        if qy < 0 || qy >= i32(params.height) {
            continue;
        }
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let qx = x + dx;
            if qx < 0 || qx >= i32(params.width) {
                continue;
            }
            let q = u32(qy) * params.width + u32(qx);
            if guide_depth(q) <= 0.0 {
                continue;
            }
            let lq = demod_lum(q);
            s = s + stats[q].w / (lq * lq);
            k = k + 1.0;
        }
    }

    let own = stats[i].w / (demod_lum(i) * demod_lum(i));
    var v = own;
    if k > 0.0 {
        v = s / k;
    }

    // ─── SVGF's spatial fallback ─────────────────────────────────────────
    //
    // Two temporal moments over one, two or three samples are not an error
    // bar; they are three numbers. A pixel that short has to be told how
    // noisy it is by its *neighbours* instead — a 7x7 luminance variance over
    // the demodulated illumination, taps rejected on the same depth and
    // normal the filter itself rejects on, so the estimate does not straddle
    // an edge and hand the filter licence to blur across it.
    //
    // This is what lets a one-frame pixel be filtered wide and correctly on
    // the frame it appears, rather than showing its single sample and then
    // settling. Off, `demodulate` is exactly the CPU filter's prefilter,
    // which is what the parity test pins.
    if params.spatial_variance != 0u && stats[i].x < 4.0 {
        var s1 = 0.0;
        var s2 = 0.0;
        var kk = 0.0;
        let z_p = guide_depth(i);
        let n_p = guide_normal(i);
        for (var dy = -3; dy <= 3; dy = dy + 1) {
            let qy = y + dy;
            if qy < 0 || qy >= i32(params.height) {
                continue;
            }
            for (var dx = -3; dx <= 3; dx = dx + 1) {
                let qx = x + dx;
                if qx < 0 || qx >= i32(params.width) {
                    continue;
                }
                let q = u32(qy) * params.width + u32(qx);
                let z_q = guide_depth(q);
                if z_q <= 0.0 {
                    continue;
                }
                if abs(z_p - z_q) > 0.1 * z_p {
                    continue;
                }
                if dot(n_p, guide_normal(q)) < 0.8 {
                    continue;
                }
                let lq = luminance(mean[q].rgb / demod_albedo(q));
                s1 = s1 + lq;
                s2 = s2 + lq * lq;
                kk = kk + 1.0;
            }
        }
        if kk > 1.0 {
            let mu = s1 / kk;
            // The variance of the neighbourhood, widened as the history
            // shortens: a 3-sample pixel is nearly there and a 1-sample one
            // is not, and the filter should know the difference.
            let spatial = max(s2 / kk - mu * mu, 0.0) * (4.0 - stats[i].x);
            v = max(spatial, 1e-8);
        }
    }

    scratch_src[i] = vec4<f32>(illum, v);
}

// ─── pass 3: one à-trous wavelet iteration ────────────────────────────────

// Iterations a pixel with `count` samples of history still gets: the full
// `params.iters` on its first sample, falling linearly to none at
// `count_cutoff`. Mirrored in Rust as `gpu::history::atrous_iters_for`.
fn atrous_iters_for(count: f32) -> u32 {
    let cutoff = f32(max(params.count_cutoff, 1u));
    let t = clamp((cutoff - count) / max(cutoff - 1.0, 1e-6), 0.0, 1.0);
    return u32(ceil(f32(params.iters) * t));
}

fn b3(k: i32) -> f32 {
    if k == 0 || k == 4 {
        return B3_0;
    }
    if k == 1 || k == 3 {
        return B3_1;
    }
    return B3_2;
}

@compute @workgroup_size(8, 8)
fn atrous(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !in_bounds(gid) {
        return;
    }
    let p = flat_index(gid);
    let z_p = guide_depth(p);
    let centre = scratch_src[p];

    // Background is analytic and noise-free: pass it through, and never let it
    // bleed onto a surface below.
    if z_p <= 0.0 {
        scratch_dst[p] = centre;
        return;
    }
    // How many wavelet iterations *this* pixel still deserves.
    //
    // `resolve` already fades the filter's strength out as a pixel's history
    // grows, but a faded filter costs exactly what a full one does: every
    // pixel ran every iteration and the widest ones — stride 16, reaching a
    // 65-pixel footprint — are the expensive ones, 25 scattered taps apiece.
    // Letting the *count* fall too means a nearly-converged pixel does the
    // first pass or two and drops out, and a fully converged one does none.
    // The budget is full at a single sample, so a one-sample history is
    // filtered exactly as `pathtrace::denoise` filters a `Film` — which is
    // what the parity test pins.
    if params.iter_index >= atrous_iters_for(stats[p].x) {
        scratch_dst[p] = centre;
        return;
    }

    let stride = i32(max(params.stride, 1u));
    let sigma_n2 = max(params.sigma_normal, 1e-4) * max(params.sigma_normal, 1e-4);
    let sigma_l = max(params.sigma_lum, 1e-6);
    let sigma_z = max(params.sigma_depth, 1e-6) * f32(stride);

    let n_p = guide_normal(p);
    let c_p = centre.xyz;
    let l_p = luminance(c_p);
    // The estimator's own error bar sets how much luminance disagreement
    // counts as signal rather than noise, so a firefly — which has an enormous
    // error bar — stops protecting itself and gets filtered.
    let l_tol = sigma_l * sqrt(max(centre.w, 0.0)) + 1e-4;

    var sum = vec3<f32>(0.0);
    var vsum = 0.0;
    var wsum = 0.0;

    let x = i32(gid.x);
    let y = i32(gid.y);
    for (var ky = 0; ky < 5; ky = ky + 1) {
        let qy = y + (ky - 2) * stride;
        if qy < 0 || qy >= i32(params.height) {
            continue;
        }
        for (var kx = 0; kx < 5; kx = kx + 1) {
            let qx = x + (kx - 2) * stride;
            if qx < 0 || qx >= i32(params.width) {
                continue;
            }
            let q = u32(qy) * params.width + u32(qx);
            let z_q = guide_depth(q);
            if z_q <= 0.0 {
                continue;
            }

            let dn = n_p - guide_normal(q);
            let w_n = exp(-dot(dn, dn) / sigma_n2);
            let w_z = exp(-abs(z_p - z_q) / (sigma_z * z_p));
            let tap = scratch_src[q];
            let w_l = exp(-abs(l_p - luminance(tap.xyz)) / l_tol);

            let weight = b3(kx) * b3(ky) * w_n * w_z * w_l;
            if weight <= 0.0 {
                continue;
            }
            sum = sum + tap.xyz * weight;
            // The variance of a weighted mean of independent estimates carries
            // the squared weights.
            vsum = vsum + weight * weight * tap.w;
            wsum = wsum + weight;
        }
    }

    if wsum > 0.0 {
        scratch_dst[p] = vec4<f32>(sum / wsum, vsum / (wsum * wsum));
    } else {
        scratch_dst[p] = centre;
    }
}

// ─── pass 4: re-modulate, tonemap, present ────────────────────────────────

// ACES filmic tonemap (Narkowicz fit), matching `pathtrace::tonemap_aces`.
fn tonemap_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Linear to sRGB transfer, matching `pathtrace::linear_to_srgb` — the exact
// curve, not the 1/2.2 approximation the main pass uses, so a frame that came
// out of here is the frame `Film::to_srgb8` would have written.
fn linear_to_srgb1(x: f32) -> f32 {
    if x <= 0.0031308 {
        return 12.92 * x;
    }
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(linear_to_srgb1(c.x), linear_to_srgb1(c.y), linear_to_srgb1(c.z));
}

@compute @workgroup_size(8, 8)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !in_bounds(gid) {
        return;
    }
    let i = flat_index(gid);
    let m = mean[i];
    var rgb = m.rgb;

    if params.iters > 0u && guide_depth(i) > 0.0 {
        var filt: vec4<f32>;
        if params.src_is_b != 0u {
            filt = scratch_dst[i];
        } else {
            filt = scratch_src[i];
        }
        let remod = filt.xyz * demod_albedo(i);
        // The filter fades out as the history grows: full strength on the
        // first sample, nothing at all once the pixel has `count_cutoff` of
        // them and the temporal mean is doing the work.
        let cnt = stats[i].x;
        let span = max(f32(params.count_cutoff) - 1.0, 1e-6);
        let strength = clamp((f32(params.count_cutoff) - cnt) / span, 0.0, 1.0);
        rgb = mix(rgb, remod, strength);
    }

    let mapped = linear_to_srgb(tonemap_aces(rgb * params.exposure));
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(mapped, m.w));
}

// ─── the split denoiser ───────────────────────────────────────────────────
//
// Everything above filters one buffer: the sample as the integrator summed
// it. What follows filters it as *two*, the way NRD, Lumen and FidelityFX
// do: the part that left the primary hit through a diffuse lobe, demodulated
// by the albedo and held for a long history, and the part that left through
// a specular lobe, demodulated by the lobe's own directional albedo, held
// for a history no longer than its roughness warrants, reprojected through
// the point the reflection actually shows, and filtered with an edge-stop
// that is sharp on a mirror and wide on a rough floor. The two are
// recombined in `resolve_lobes`.
//
// Five more entry points, dispatched in the same slots as their unsplit
// twins: `reproject_lobes` (with `reproject`), `accumulate_lobes` (after
// `accumulate`), then `demodulate_lobes`, `atrous_lobes` and `resolve_lobes`
// *instead of* `demodulate`, `atrous` and `resolve`. Nothing above changes.

// The specular history's bounds, in samples, interpolated by roughness: a
// mirror keeps a handful, a rough surface keeps as many as a diffuse one.
//
// The floor is not 1. With the reflection reprojected through its own point
// a mirror's history is *right* until the camera moves, and on the court a
// floor of one left the backboard glass with no temporal history at all —
// measured over the `--denoise-eval` sequence it was 21.8 codes off the
// reference against 18.7 unsplit; a floor of 8 brought it to 19.4, and
// lifting the cap altogether (32) drifted back out to 20.0.
const SPEC_HISTORY_MIN: f32 = 8.0;
const SPEC_HISTORY_MAX: f32 = 32.0;
// What the luminance edge-stop is scaled by on the specular half, by
// roughness: a mirror's reflection is an image and must not be blurred
// across; a rough lobe's is a smear and should be.
const SPEC_SIGMA_MIRROR: f32 = 0.25;
const SPEC_SIGMA_ROUGH: f32 = 2.0;

const LOBE_DIFFUSE: u32 = 0u;
const LOBE_SPECULAR: u32 = 1u;

fn guide_spec_albedo(i: u32) -> vec3<f32> {
    return guides[4u * n_pixels() + i].xyz;
}

fn guide_roughness(i: u32) -> f32 {
    return guides[4u * n_pixels() + i].w;
}

// A reflection further than this many times the surface's own distance is
// the sky as far as parallax goes.
const SPEC_FAR: f32 = 32.0;

// How far *this pass's* specular bounce off pixel `i`'s surface went, made
// usable: a hit's distance capped at `SPEC_FAR` surface distances, an
// escaped bounce read as that cap, and a diffuse bounce — which says nothing
// about the reflection — as 0.
fn guide_spec_dist(i: u32) -> f32 {
    let t = guides[5u * n_pixels() + i].x;
    let far = SPEC_FAR * guide_depth(i);
    if t < 0.0 {
        return far;
    }
    return min(t, far);
}

// The reflection's distance as the specular history has averaged it: the
// specular lobe mean's spare lane. One sample's bounce is one direction out
// of a rough lobe, and the reprojection wants the lobe's centre, not its
// last draw.
fn spec_dist_smoothed(i: u32) -> f32 {
    return lobe_mean[n_pixels() + i].w;
}

// How much of the reflection's parallax a lobe of this roughness shows:
// all of it on a mirror, none of it on a Lambertian smear. The factor
// NRD scales its virtual position by.
fn spec_dominant(roughness: f32) -> f32 {
    let r = clamp(roughness, 0.0, 1.0);
    return (1.0 - r) * (sqrt(1.0 - r) + r);
}

// The demodulator for one lobe, floored as `demod_albedo` is.
fn lobe_demod(lobe: u32, i: u32) -> vec3<f32> {
    if lobe == LOBE_DIFFUSE {
        return demod_albedo(i);
    }
    return max(guide_spec_albedo(i), vec3<f32>(DEMOD_FLOOR));
}

fn lobe_demod_lum(lobe: u32, i: u32) -> f32 {
    return max(luminance(lobe_demod(lobe, i)), DEMOD_FLOOR);
}

// The longest history one lobe may hold at pixel `i`.
fn lobe_history_cap(lobe: u32, i: u32) -> f32 {
    let cap = f32(max(params.history_cap, 1u));
    if lobe == LOBE_DIFFUSE {
        return cap;
    }
    return min(cap, mix(SPEC_HISTORY_MIN, SPEC_HISTORY_MAX, guide_roughness(i)));
}

// This pass's raw sample for one lobe: the integrator's planes 1 and 2.
fn raw_lobe(lobe: u32, i: u32) -> vec4<f32> {
    return raw[(1u + lobe) * n_pixels() + i];
}

// Where pixel `i` was on the previous frame's film, validated as the same
// surface, or -1 if there is nothing to carry. `reproject`'s test, made a
// function so the specular half can run it twice.
//
// `virtual_t` is the specular twist. A reflection is not *on* the mirror:
// it is the reflected scene seen through it, `t` further along the view
// ray, and under a camera move it slides across the mirror with that
// point's parallax rather than the mirror's. So the specular half projects
// the point `virtual_t` behind the surface — surface plus the reflected ray
// extended by the specular bounce's hit distance — and takes the history
// from wherever *that* landed, still gated on the surface: the previous
// pixel has to have been looking at this same object, in this same plane.
// The virtual point is not carried by the object's own motion: a mirror
// sliding in its own plane shows the same static world it did, and it is
// the surface, not the reflection, that moved.
fn reproject_find(gid: vec3<u32>, i: u32, virtual_t: f32) -> i32 {
    let depth = guide_depth(i);
    if depth <= 0.0 {
        return -1;
    }
    let dir = view_ray(
        params.cur_right.xyz, params.cur_up.xyz, params.cur_forward.xyz,
        params.view_params.x, params.view_params.y, gid.x, gid.y,
    );
    let p_now = params.cur_eye.xyz + depth * dir;

    let id = guide_id(i);
    var inst = 0xFFFFFFFFu;
    if params.motion_instances > 0u && id > 0u {
        let cand = instance_of(id - 1u);
        if cand < params.motion_instances {
            inst = cand;
        }
    }
    var p = p_now;
    var n_now = guide_normal(i);
    if inst != 0xFFFFFFFFu {
        p = motion_point(inst, p_now);
        n_now = motion_dir(inst, n_now);
    }
    // What is projected: the surface, or the point the reflection shows.
    var proj = p;
    if virtual_t > 0.0 {
        proj = params.cur_eye.xyz + (depth + virtual_t) * dir;
    }

    let v = proj - params.prev_eye.xyz;
    let z = dot(v, params.prev_forward.xyz);
    if z <= 0.0 {
        return -1;
    }
    let tan_fov = params.view_params.z;
    let aspect = params.view_params.w;
    let ndc_x = dot(v, params.prev_right.xyz) / (z * tan_fov * aspect);
    let ndc_y = dot(v, params.prev_up.xyz) / (z * tan_fov);
    let fx = (ndc_x + 1.0) * 0.5 * f32(params.width) - 0.5;
    let fy = (1.0 - ndc_y) * 0.5 * f32(params.height) - 0.5;
    let qx = i32(round(fx));
    let qy = i32(round(fy));
    if qx < 0 || qy < 0 || qx >= i32(params.width) || qy >= i32(params.height) {
        return -1;
    }
    let j = u32(qy) * params.width + u32(qx);

    let prev_depth = prev_guides[j].w;
    if prev_depth <= 0.0 {
        return -1;
    }
    if prev_guide_id(j) != id {
        return -1;
    }
    let prev_n = prev_guides[j].xyz;
    let prev_dir = view_ray(
        params.prev_right.xyz, params.prev_up.xyz, params.prev_forward.xyz,
        tan_fov, aspect, u32(qx), u32(qy),
    );
    let q = params.prev_eye.xyz + prev_depth * prev_dir;
    // The surface's own distance sets the tolerance, whatever was projected.
    let expected = length(p - params.prev_eye.xyz);
    if abs(dot(p - q, prev_n)) > REPROJ_DEPTH_TOL * expected {
        return -1;
    }
    if dot(n_now, prev_n) < REPROJ_NORMAL_DOT {
        return -1;
    }
    return i32(j);
}

// ─── pass 0b: carry both lobes' histories across the frame ───────────────

@compute @workgroup_size(8, 8)
fn reproject_lobes(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !in_bounds(gid) {
        return;
    }
    let i = flat_index(gid);
    let n = n_pixels();
    lobe_gather[i] = vec4<f32>(0.0);
    lobe_gather[n + i] = vec4<f32>(0.0);
    lobe_gather[2u * n + i] = vec4<f32>(0.0);
    lobe_gather[3u * n + i] = vec4<f32>(0.0);

    // The diffuse half follows the surface, exactly as `reproject` does.
    let jd = reproject_find(gid, i, 0.0);
    if jd >= 0 {
        lobe_gather[i] = lobe_mean[u32(jd)];
        lobe_gather[n + i] = lobe_stats[u32(jd)];
    }
    // The specular half follows the reflection — and, where the virtual
    // point lands on nothing it can use (a curved reflector, a point off the
    // previous film), the surface, which is what it had before. The
    // distance is the history's own average where it has one, and this
    // pass's bounce where it does not.
    var t = spec_dist_smoothed(i);
    if t <= 0.0 {
        t = guide_spec_dist(i);
    }
    var js = reproject_find(gid, i, t * spec_dominant(guide_roughness(i)));
    if js < 0 {
        js = jd;
    }
    if js >= 0 {
        lobe_gather[2u * n + i] = lobe_mean[n + u32(js)];
        lobe_gather[3u * n + i] = lobe_stats[n + u32(js)];
    }
}

// ─── pass 1b: fold this sample into each lobe's history ──────────────────

// `accumulate`'s fold, for one lobe. The budget's skip, the keep mask, the
// neighbourhood clamp and the bounded fold are all as above; what differs is
// the cap, which the specular half takes from its roughness.
fn fold_lobe(gid: vec3<u32>, i: u32, lobe: u32) {
    let n = n_pixels();
    let slot = lobe * n + i;
    let c = raw_lobe(lobe, i);
    let l = luminance(c.rgb);

    var m = lobe_mean[slot];
    var st = lobe_stats[slot];
    if params.reprojected != 0u {
        m = lobe_gather[(2u * lobe) * n + i];
        st = lobe_gather[(2u * lobe + 1u) * n + i];
    }
    if keep[i] == 0u {
        m = vec4<f32>(0.0);
        st = vec4<f32>(0.0);
    }

    if params.budget_enabled != 0u {
        if (budget_mask_load(i) & (1u << params.budget_round)) == 0u {
            lobe_mean[slot] = m;
            lobe_stats[slot] = st;
            return;
        }
    }

    if params.clamp_k > 0.0 && st.x > 0.5 {
        var s1 = vec3<f32>(0.0);
        var s2 = vec3<f32>(0.0);
        var hsum = vec3<f32>(0.0);
        var k = 0.0;
        let x = i32(gid.x);
        let y = i32(gid.y);
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            let qy = y + dy;
            if qy < 0 || qy >= i32(params.height) {
                continue;
            }
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let qx = x + dx;
                if qx < 0 || qx >= i32(params.width) {
                    continue;
                }
                let q = u32(qy) * params.width + u32(qx);
                let t = raw_lobe(lobe, q).rgb;
                s1 = s1 + t;
                s2 = s2 + t * t;
                hsum = hsum + lobe_mean[lobe * n + q].rgb;
                k = k + 1.0;
            }
        }
        if k > 0.0 {
            let mu = s1 / k;
            let sd = sqrt(max(s2 / k - mu * mu, vec3<f32>(0.0)));
            let hmu = hsum / k;
            let tol = params.clamp_k * sd / sqrt(k) + 1e-5;
            let d = abs(hmu - mu);
            if max(d.x, max(d.y, d.z)) > max(tol.x, max(tol.y, tol.z)) {
                st.x = min(st.x, f32(max(params.clamp_reset, 1u)));
            }
        }
    }

    let cap = lobe_history_cap(lobe, i);
    let cnt = min(st.x + 1.0, cap);
    let inv = 1.0 / cnt;
    if lobe == LOBE_SPECULAR {
        // The specular mean's spare lane is the reflection's distance, folded
        // with the same weight — and left alone by a diffuse bounce, which
        // saw no reflection to measure.
        let t = guide_spec_dist(i);
        var td = m.w;
        if t > 0.0 {
            td = select(td + (t - td) * inv, t, td <= 0.0);
        }
        m = vec4<f32>(m.rgb + (c.rgb - m.rgb) * inv, td);
    } else {
        m = m + (c - m) * inv;
    }
    let mu1 = st.y + (l - st.y) * inv;
    let mu2 = st.z + (l * l - st.z) * inv;
    var v: f32;
    if cnt > 1.5 {
        let sample_var = max(mu2 - mu1 * mu1, 0.0) * cnt / (cnt - 1.0);
        v = sample_var * inv;
    } else {
        v = mu1 * mu1;
    }
    lobe_mean[slot] = m;
    lobe_stats[slot] = vec4<f32>(cnt, mu1, mu2, v);
}

@compute @workgroup_size(8, 8)
fn accumulate_lobes(@builtin(global_invocation_id) lid: vec3<u32>) {
    let gid = vec3<u32>(lid.x + params.origin_x, lid.y + params.origin_y, lid.z);
    if !in_bounds(gid) {
        return;
    }
    if !in_scissor(gid) {
        return;
    }
    let i = flat_index(gid);
    fold_lobe(gid, i, LOBE_DIFFUSE);
    fold_lobe(gid, i, LOBE_SPECULAR);
}

// ─── pass 2b: demodulate each lobe and prefilter its variance ────────────

// `demodulate`, for one lobe, writing plane `lobe` of `scratch_src`.
fn demod_lobe(gid: vec3<u32>, i: u32, lobe: u32) {
    let n = n_pixels();
    let slot = lobe * n + i;
    let illum = lobe_mean[slot].rgb / lobe_demod(lobe, i);

    var s = 0.0;
    var k = 0.0;
    let x = i32(gid.x);
    let y = i32(gid.y);
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        let qy = y + dy;
        if qy < 0 || qy >= i32(params.height) {
            continue;
        }
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let qx = x + dx;
            if qx < 0 || qx >= i32(params.width) {
                continue;
            }
            let q = u32(qy) * params.width + u32(qx);
            if guide_depth(q) <= 0.0 {
                continue;
            }
            let lq = lobe_demod_lum(lobe, q);
            s = s + lobe_stats[lobe * n + q].w / (lq * lq);
            k = k + 1.0;
        }
    }
    let own_l = lobe_demod_lum(lobe, i);
    var v = lobe_stats[slot].w / (own_l * own_l);
    if k > 0.0 {
        v = s / k;
    }

    if params.spatial_variance != 0u && lobe_stats[slot].x < 4.0 {
        var s1 = 0.0;
        var s2 = 0.0;
        var kk = 0.0;
        let z_p = guide_depth(i);
        let n_p = guide_normal(i);
        for (var dy = -3; dy <= 3; dy = dy + 1) {
            let qy = y + dy;
            if qy < 0 || qy >= i32(params.height) {
                continue;
            }
            for (var dx = -3; dx <= 3; dx = dx + 1) {
                let qx = x + dx;
                if qx < 0 || qx >= i32(params.width) {
                    continue;
                }
                let q = u32(qy) * params.width + u32(qx);
                let z_q = guide_depth(q);
                if z_q <= 0.0 {
                    continue;
                }
                if abs(z_p - z_q) > 0.1 * z_p {
                    continue;
                }
                if dot(n_p, guide_normal(q)) < 0.8 {
                    continue;
                }
                let lq = luminance(lobe_mean[lobe * n + q].rgb / lobe_demod(lobe, q));
                s1 = s1 + lq;
                s2 = s2 + lq * lq;
                kk = kk + 1.0;
            }
        }
        if kk > 1.0 {
            let mu = s1 / kk;
            let spatial = max(s2 / kk - mu * mu, 0.0) * (4.0 - lobe_stats[slot].x);
            v = max(spatial, 1e-8);
        }
    }

    scratch_src[slot] = vec4<f32>(illum, v);
}

@compute @workgroup_size(8, 8)
fn demodulate_lobes(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !in_bounds(gid) {
        return;
    }
    let i = flat_index(gid);
    demod_lobe(gid, i, LOBE_DIFFUSE);
    demod_lobe(gid, i, LOBE_SPECULAR);
}

// ─── pass 3b: one à-trous iteration over each lobe ───────────────────────

// `atrous`, for one lobe, over plane `lobe` of the scratch pair. The
// specular half's luminance tolerance is scaled by its roughness.
fn atrous_lobe(gid: vec3<u32>, p: u32, lobe: u32) {
    let n = n_pixels();
    let slot = lobe * n + p;
    let z_p = guide_depth(p);
    let centre = scratch_src[slot];
    if z_p <= 0.0 {
        scratch_dst[slot] = centre;
        return;
    }
    if params.iter_index >= atrous_iters_for(lobe_stats[slot].x) {
        scratch_dst[slot] = centre;
        return;
    }

    let stride = i32(max(params.stride, 1u));
    let sigma_n2 = max(params.sigma_normal, 1e-4) * max(params.sigma_normal, 1e-4);
    var sigma_l = max(params.sigma_lum, 1e-6);
    if lobe == LOBE_SPECULAR {
        sigma_l = sigma_l * mix(SPEC_SIGMA_MIRROR, SPEC_SIGMA_ROUGH, guide_roughness(p));
    }
    let sigma_z = max(params.sigma_depth, 1e-6) * f32(stride);

    let n_p = guide_normal(p);
    let c_p = centre.xyz;
    let l_p = luminance(c_p);
    let l_tol = sigma_l * sqrt(max(centre.w, 0.0)) + 1e-4;

    var sum = vec3<f32>(0.0);
    var vsum = 0.0;
    var wsum = 0.0;
    let x = i32(gid.x);
    let y = i32(gid.y);
    for (var ky = 0; ky < 5; ky = ky + 1) {
        let qy = y + (ky - 2) * stride;
        if qy < 0 || qy >= i32(params.height) {
            continue;
        }
        for (var kx = 0; kx < 5; kx = kx + 1) {
            let qx = x + (kx - 2) * stride;
            if qx < 0 || qx >= i32(params.width) {
                continue;
            }
            let q = u32(qy) * params.width + u32(qx);
            let z_q = guide_depth(q);
            if z_q <= 0.0 {
                continue;
            }
            let dn = n_p - guide_normal(q);
            let w_n = exp(-dot(dn, dn) / sigma_n2);
            let w_z = exp(-abs(z_p - z_q) / (sigma_z * z_p));
            let tap = scratch_src[lobe * n + q];
            let w_l = exp(-abs(l_p - luminance(tap.xyz)) / l_tol);
            let weight = b3(kx) * b3(ky) * w_n * w_z * w_l;
            if weight <= 0.0 {
                continue;
            }
            sum = sum + tap.xyz * weight;
            vsum = vsum + weight * weight * tap.w;
            wsum = wsum + weight;
        }
    }
    if wsum > 0.0 {
        scratch_dst[slot] = vec4<f32>(sum / wsum, vsum / (wsum * wsum));
    } else {
        scratch_dst[slot] = centre;
    }
}

@compute @workgroup_size(8, 8)
fn atrous_lobes(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !in_bounds(gid) {
        return;
    }
    let p = flat_index(gid);
    atrous_lobe(gid, p, LOBE_DIFFUSE);
    atrous_lobe(gid, p, LOBE_SPECULAR);
}

// ─── pass 4b: recombine, tonemap, present ────────────────────────────────

// `resolve`'s fade, for one lobe's history length.
fn lobe_strength(count: f32) -> f32 {
    let span = max(f32(params.count_cutoff) - 1.0, 1e-6);
    return clamp((f32(params.count_cutoff) - count) / span, 0.0, 1.0);
}

@compute @workgroup_size(8, 8)
fn resolve_lobes(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !in_bounds(gid) {
        return;
    }
    let i = flat_index(gid);
    let n = n_pixels();
    let md = lobe_mean[i];
    let ms = lobe_mean[n + i];
    var rgb = md.rgb + ms.rgb;

    if params.iters > 0u && guide_depth(i) > 0.0 {
        var fd: vec4<f32>;
        var fs: vec4<f32>;
        if params.src_is_b != 0u {
            fd = scratch_dst[i];
            fs = scratch_dst[n + i];
        } else {
            fd = scratch_src[i];
            fs = scratch_src[n + i];
        }
        // diffuse * albedo + specular * specular albedo, each faded by its
        // own history length: a rough floor's specular half is filtered long
        // after its diffuse half has converged.
        let d = mix(md.rgb, fd.xyz * lobe_demod(LOBE_DIFFUSE, i), lobe_strength(lobe_stats[i].x));
        let s = mix(ms.rgb, fs.xyz * lobe_demod(LOBE_SPECULAR, i), lobe_strength(lobe_stats[n + i].x));
        rgb = d + s;
    }

    let mapped = linear_to_srgb(tonemap_aces(rgb * params.exposure));
    textureStore(out_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(mapped, md.w));
}
