// ─── where the samples go ─────────────────────────────────────────────────
//
// Appended to `history.wgsl`, and reads its bindings and its `params`. Not
// valid WGSL on its own; see `shaders::history_shader`.
//
// A game spends its samples uniformly: one ray per pixel per frame, whether
// the pixel is a wall that has not changed in four hundred frames or the ball
// that crossed it this one. Everything needed to do better is already on the
// device *before* the trace:
//
//   * the physics — every instance's transform this frame, so the exact
//     screen-space displacement of the surface each pixel is looking at;
//   * the history — each pixel's sample count and the variance of its own
//     mean, so how converged it is;
//   * the image — the last raw sample against the running mean, which is the
//     same disagreement the neighbourhood clamp measures.
//
// These four passes turn that into a per-pixel sample budget b(p): how many
// of this frame's rays pixel p should get. Normalised so the frame's total is
// the tuner's `rays_per_frame`, so directing the samples never spends more of
// them — it only moves them.
//
//   budget_weight   raw drive per pixel, from the three inputs above
//   budget_blur_x   dilate the motion drive by the filter radius, horizontally
//   budget_blur_y   ...and vertically, then sum the frame's weight
//   budget_normalize  scale the weights so they sum to `rays_per_frame`
//
// `accumulate` then folds pixel p's sample on round r only when p's coin comes
// up, at probability b(p)/rounds. Selection is independent of the sample's
// value, so the mean of what is folded is unbiased — the 1/p reweighting a
// value-dependent scheme would need is not needed here, and would only add
// variance.

// One pixel's four drives and its budget:
//   .x  the motion drive, dilated by the two blur passes
//   .y  the history drive: relative error bar, plus a short-history term
//   .z  the image drive: the last sample's disagreement with the mean
//   .w  the budget b(p), in samples — written by `budget_normalize`
@group(0) @binding(11) var<storage, read_write> budget: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read_write> budget_scratch: array<vec4<f32>>;

// The frame's total weight, in 1/BUDGET_FIXED units. Cleared by the host each
// frame; `budget_blur_y` adds every pixel's weight into it and
// `budget_normalize` divides by it.
@group(0) @binding(13) var<storage, read_write> budget_total: array<atomic<u32>>;

// Fixed point for the weight sum. A weight is clamped to WEIGHT_MAX, so the
// sum fits a u32 for any frame up to about four megapixels.
const BUDGET_FIXED: f32 = 32.0;
const WEIGHT_MAX: f32 = 16.0;

// The most any one drive may claim, in units of a uniform pixel's share.
//
// The drives are *ratios* against their targets, not switches, and clamping
// them at one was the whole difference between a budget and a flat field: a
// frame in which every pixel is above its noise target is a frame in which
// every pixel asks for exactly one share, which is the uniform spend wearing
// a disguise. Left as ratios, a pixel eight times over its target outbids one
// just past it, which is what "directed" has to mean. The ceiling is only
// there to stop a single outlier taking the frame.
const DRIVE_MAX: f32 = 16.0;

// A pixel whose surface moved this far across the film, in pixels, is fully
// weighted. Half a pixel already puts a new surface under half the filter's
// taps.
const MOTION_PIXELS: f32 = 0.5;

// The relative error bar a converged pixel is aiming at: one standard
// deviation of the mean, as a fraction of the mean's own luminance.
//
// Not the error a *finished* picture would want — that is well under a
// percent, and every pixel of a live frame is over it, which makes the term
// a constant. This is set where a live frame's pixels actually sit, so that
// the ratio spreads them out instead of pinning them all to the ceiling.
const NOISE_TARGET: f32 = 0.15;

// The most the error-bar term may claim, in shares.
//
// It has to be bounded and it has to be small. Every pixel of a path-traced
// frame is over a 2% error bar for a long time, so an unbounded term is a
// flat field wearing a disguise — and a term that dominates the short-history
// one starves the pixels that most need samples. Three shares is enough range
// for the noisiest quarter of a still frame to outbid the calmest and not
// enough to outbid a pixel whose history was just thrown away.
const ERR_MAX: f32 = 3.0;

// How many of a single sample's standard deviations the last sample may sit
// from the running mean before the pixel is treated as having changed.
// Looser than the clamp's threshold because this is one sample against a
// mean rather than two nine-pixel means.
const DISAGREE_SIGMAS: f32 = 2.0;

// A pixel with fewer than this many samples is short: nothing it reports
// about its own variance is worth anything yet, and what it needs is samples.
const SHORT_HISTORY: f32 = 8.0;

// What a pixel with no history at all is worth, in shares.
const SHORT_GAIN: f32 = 6.0;

// The share of a uniform pixel's weight that every pixel keeps however
// converged and however static it is. Without it a wall that has been still
// for a thousand frames is never sampled again and can never notice that it
// stopped being still — and the floor test wants a hard guarantee as well as
// this soft one.
const BUDGET_FLOOR: f32 = 0.03;

// What a fully moving pixel is worth on its own, before the multiplicative
// term, in units of a pixel that is one error target off.
const MOTION_GAIN: f32 = 3.0;
const IMAGE_GAIN: f32 = 3.0;

fn budget_drive(x: f32) -> f32 {
    return clamp(x, 0.0, DRIVE_MAX);
}

// Pass 0: what each pixel's three inputs say, before any dilation.
//
// Everything read here is known *before* this frame is traced: the guide
// planes and the raw sample are the previous pass's, and the motion table is
// this frame's, written by the host out of the same transforms it just posed
// the scene with. That is the whole point — a budget computed from this
// frame's sample would arrive one pass too late to spend it.
@compute @workgroup_size(8, 8)
fn budget_weight(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !in_bounds(gid) {
        return;
    }
    let i = flat_index(gid);

    // ─── the physics ─────────────────────────────────────────────────────
    //
    // How far the surface under this pixel moves across the film this frame.
    // The motion table gives the world displacement of the instance the pixel
    // was looking at; the previous depth turns it into pixels.
    var motion_drive = 0.0;
    let depth = guide_depth(i);
    if params.motion_instances > 0u && depth > 0.0 {
        let id = guide_id(i);
        if id > 0u {
            let inst = instance_of(id - 1u);
            if inst < params.motion_instances {
                let dir = view_ray(
                    params.cur_right.xyz, params.cur_up.xyz, params.cur_forward.xyz,
                    params.view_params.x, params.view_params.y, gid.x, gid.y,
                );
                let p = params.cur_eye.xyz + depth * dir;
                let world = length(motion_point(inst, p) - p);
                // Pixels per world unit at this depth, for a pinhole whose
                // half-height subtends tan(fov/2).
                let focal_px = 0.5 * f32(params.height) / max(params.view_params.x, 1e-6);
                let px = world * focal_px / max(depth, 1e-6);
                motion_drive = budget_drive(px / MOTION_PIXELS);
            }
        }
    }

    // ─── the history ─────────────────────────────────────────────────────
    let st = stats[i];
    let n = st.x;
    let m = mean[i];
    let lum = max(luminance(m.rgb), 1e-4);
    // `stats.w` is already the variance *of the mean*, so its root is the
    // error bar the extra samples would shrink — the sigma/sqrt(n) the tuner
    // is trying to buy down.
    let rel_err = sqrt(max(st.w, 0.0)) / lum;
    // Two terms, added rather than maxed, because they say different things:
    // how far off the mean is, and how little there is behind it.
    //
    // The second is the one that carries a live frame, and it is not a small
    // correction. A pixel the neighbourhood clamp has just shortened — a
    // floor the ball's shadow has moved off — is holding two samples and is
    // the worst pixel in the frame; a converged neighbour of it is the best.
    // Capping the short-history term at one share tells the budget they are
    // equally deserving, which is the whole difference between a directed
    // spend and a rounding error.
    let history_drive = min(rel_err / NOISE_TARGET, ERR_MAX)
        + SHORT_GAIN * clamp((SHORT_HISTORY - n) / SHORT_HISTORY, 0.0, 1.0);

    // ─── the image ───────────────────────────────────────────────────────
    //
    // The last raw sample against the mean it was folded into. This is what
    // catches lighting that went stale with nothing moving under the pixel —
    // the ball's shadow left on the floor — one frame after the clamp does,
    // and spends samples there rather than merely shortening the history.
    // Measured in units of *one sample's own* standard deviation, not as a
    // fraction of the mean. A path-traced sample at 1 spp sits half its own
    // magnitude away from the truth on a good day, so a relative test reads
    // the Monte Carlo noise on every pixel in the frame far louder than it
    // reads the shadow that actually moved — and the budget follows the
    // noise. Against the pixel's own error bar the noise reads one, by
    // construction, and only a real change reads more.
    var image_drive = 0.0;
    if n > 1.5 {
        let d = abs(luminance(raw[i].rgb) - luminance(m.rgb));
        let sample_sigma = sqrt(max(st.w, 0.0) * n);
        image_drive = budget_drive(
            max(d / max(sample_sigma * DISAGREE_SIGMAS, 1e-6) - 1.0, 0.0),
        );
    }

    budget[i] = vec4<f32>(motion_drive, history_drive, image_drive, 0.0);
}

// One separable dilation tap set: the maximum of the motion drive over
// `budget_radius` pixels either side, at a stride that keeps the tap count
// bounded for a wide radius.
fn dilate_motion(x: i32, y: i32, dx: i32, dy: i32) -> f32 {
    let r = i32(params.budget_radius);
    if r <= 0 {
        let i = u32(y) * params.width + u32(x);
        return budget[i].x;
    }
    // At most 17 taps whatever the radius: a dilation is a maximum, so a
    // coarse comb over the footprint finds the same peak a dense one would
    // unless the moving object is thinner than the stride, and the other
    // axis's pass closes that gap.
    let step = max(1, (r + 7) / 8);
    var best = 0.0;
    var k = -r;
    loop {
        if k > r {
            break;
        }
        let qx = clamp(x + dx * k, 0, i32(params.width) - 1);
        let qy = clamp(y + dy * k, 0, i32(params.height) - 1);
        best = max(best, budget[u32(qy) * params.width + u32(qx)].x);
        k = k + step;
    }
    return best;
}

// Pass 1: dilate the motion drive horizontally.
//
// A moved instance changes more than the pixels it covers: its shadow moves,
// the light it bounced moves, and the à-trous filter will reach `2^iters`
// pixels off its silhouette and pull whatever is there into it. Spending
// samples only on the silhouette buys a clean ball inside a dirty halo. The
// radius is the host's filter footprint.
@compute @workgroup_size(8, 8)
fn budget_blur_x(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !in_bounds(gid) {
        return;
    }
    let i = flat_index(gid);
    var v = budget[i];
    v.x = dilate_motion(i32(gid.x), i32(gid.y), 1, 0);
    budget_scratch[i] = v;
}

// Pass 2: dilate vertically, combine the three drives into one weight, and
// add it to the frame's total.
//
// The weight is written back into `.w` and the total is a fixed-point atomic
// sum, so `budget_normalize` needs no reduction of its own.
@compute @workgroup_size(8, 8)
fn budget_blur_y(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !in_bounds(gid) {
        return;
    }
    let i = flat_index(gid);
    // The horizontal pass left its result in the scratch copy; read the
    // vertical taps out of that.
    let r = i32(params.budget_radius);
    var motion_drive = budget_scratch[i].x;
    if r > 0 {
        let step = max(1, (r + 7) / 8);
        var k = -r;
        loop {
            if k > r {
                break;
            }
            let qy = clamp(i32(gid.y) + k, 0, i32(params.height) - 1);
            motion_drive = max(motion_drive, budget_scratch[u32(qy) * params.width + gid.x].x);
            k = k + step;
        }
    }
    let v = budget_scratch[i];

    // What the three drives are estimating between them is one number: the
    // error this pixel will still be carrying at the end of the frame if it
    // is given nothing.
    //
    // The history and the image are two readings of the same quantity — how
    // far the running mean is from the truth — so they combine with a
    // maximum: whichever noticed it. The physics is not a reading of it at
    // all; it is the news that the reading is about to stop being true,
    // because the surface under the pixel is somewhere else. So it multiplies
    // rather than competes: a pixel that was converged and is about to be
    // disoccluded needs a fresh pixel's worth of samples, not a converged
    // one's.
    let directed = BUDGET_FLOOR
        + v.y
        + MOTION_GAIN * motion_drive
        + IMAGE_GAIN * v.z;

    // bias 0 is one unit of weight everywhere, which is exactly today's
    // uniform spend; bias 1 is the directed field; in between is the honest
    // interpolation of the two, because the normalisation below is linear in
    // the weight.
    let w = clamp(mix(1.0, directed, clamp(params.budget_bias, 0.0, 1.0)), 0.0, WEIGHT_MAX);
    budget[i] = vec4<f32>(motion_drive, v.y, v.z, w);
        // Rounded, not truncated. Truncation loses half a fixed-point unit per
    // pixel on average, and half a unit over a whole frame is a systematic
    // under-count of the total — which the normalisation then divides by, so
    // the frame comes out a percent or two over the budget it was given.
    atomicAdd(&budget_total[0], u32(w * BUDGET_FIXED + 0.5));
}

// Pass 3: scale the weights into samples.
//
// b(p) = w(p) * rays_per_frame / sum(w), clamped to the rounds the host is
// actually going to dispatch. `.w` ends up holding b(p) — samples, not
// weight — which is what `accumulate` reads and what `read_budget` hands
// back.
@compute @workgroup_size(8, 8)
fn budget_normalize(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !in_bounds(gid) {
        return;
    }
    let i = flat_index(gid);
    let total = max(f32(atomicLoad(&budget_total[0])) / BUDGET_FIXED, 1e-6);
    let v = budget[i];
    var b = v.w * params.rays_per_frame / total;

    // The starvation floor. A pixel whose weight rounds to nothing would
    // never be looked at again, and a scene it stopped agreeing with would
    // never be noticed. Every `budget_floor_k` frames each pixel is
    // guaranteed one sample; the phase is the pixel's own, so the cost is
    // spread evenly over the k frames rather than landing on one of them.
    let k = max(params.budget_floor_k, 1u);
    if k == 1u || (params.budget_frame + i) % k == 0u {
        b = max(b, 1.0);
    }

    budget[i] = vec4<f32>(v.x, v.y, v.z, min(b, f32(max(params.budget_rounds, 1u))));
}

// Pass 4: put back what the clamp took.
//
// `budget_normalize` clamps b(p) to the rounds the host will dispatch and
// raises the floor pixels to one, and both of those move the total off
// `rays_per_frame` — down where a hot pixel wanted six samples out of four
// rounds, up where the floor fired. One rescale pass closes it: sum what was
// actually assigned, scale every pixel that is neither clamped nor floored by
// the ratio, and the total lands within a fraction of a percent of the
// target. A second round would buy nothing a test could see.
@compute @workgroup_size(8, 8)
fn budget_assigned(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !in_bounds(gid) {
        return;
    }
    let i = flat_index(gid);
    atomicAdd(&budget_total[1], u32(budget[i].w * BUDGET_FIXED + 0.5));
}

@compute @workgroup_size(8, 8)
fn budget_rescale(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !in_bounds(gid) {
        return;
    }
    let i = flat_index(gid);
    let assigned = max(f32(atomicLoad(&budget_total[1])) / BUDGET_FIXED, 1e-6);
    let scale = params.rays_per_frame / assigned;
    let cap = f32(max(params.budget_rounds, 1u));
    let v = budget[i];
    var b = v.w * scale;
    let k = max(params.budget_floor_k, 1u);
    if k == 1u || (params.budget_frame + i) % k == 0u {
        b = max(b, 1.0);
    }
    budget[i] = vec4<f32>(v.x, v.y, v.z, clamp(b, 0.0, cap));
}
