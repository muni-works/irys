// The learned denoiser, on the device.
//
// Three dispatches, one per convolution, with the two hidden activations in
// global storage rather than in workgroup memory. That is a deliberate
// trade. A single fused pass would have to hold a 14x14 feature tile, a
// 12x12 first activation and a 10x10 second in `var<workgroup>` — about 39 KB
// at 32 hidden channels, against a 16 KB budget — and every pixel outside a
// workgroup's own 8x8 would be computed twice over. Three passes read their
// 3x3 neighbourhood out of L2 instead, which the cache is good at, and cost
// one buffer of `hidden * n` floats each.
//
// The weights arrive as one flat `array<f32>` of `W1 b1 W2 b2 W3 b3` with
// the six offsets in the uniform, but each `W` is **tap-major** here —
// `[out][tap][in]` rather than the file's `[out][in][tap]`. That is the one
// thing `NeuralDenoiser::new` rearranges on upload, and it is worth a
// rearrangement: the inner loop of every convolution walks the input channel
// at a fixed neighbour, so tap-major makes both the weight read and the
// activation read a contiguous run.
//
// Every read of the history's `mean`, `stats` and guide planes matches
// `history.wgsl`'s, and the output lands in the same `(illumination,
// variance)` scratch buffer the à-trous iterations write, so `resolve` is
// untouched and remodulates and tonemaps whichever filter ran.

const C_IN: u32 = 11u;
const TAPS: i32 = 5;
const K: u32 = 25u;
const KS: i32 = 3;
const DEMOD_FLOOR: f32 = 0.01;
const DEPTH_SCALE: f32 = 3000.0;

struct NeuralParams {
    width: u32,
    height: u32,
    hidden: u32,
    // Unread here; the history length is per pixel and comes out of `stats`.
    // Present so the struct's word layout is stated rather than padded into
    // existence.
    reserved: u32,
    // `Weights::offsets()`, in floats.
    w1: u32,
    b1: u32,
    w2: u32,
    b2: u32,
    w3: u32,
    b3: u32,
    // Below this history length the network runs; at or above it the pixel is
    // passed through, exactly as `atrous_iters_for` fades the à-trous filter
    // out. A converged pixel needs no filter.
    count_cutoff: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: NeuralParams;
// (face-forwarded normal, distance) at plane 1, (albedo, biased id) at plane 2
@group(0) @binding(1) var<storage, read> guides: array<vec4<f32>>;
// the running mean: (linear radiance, coverage)
@group(0) @binding(2) var<storage, read> mean: array<vec4<f32>>;
// (count, luminance sum, luminance-squared sum, variance of the mean)
@group(0) @binding(3) var<storage, read> stats: array<vec4<f32>>;
// `hidden * n` floats each, ping-ponged between the two convolution passes.
//
// Interleaved by pixel — `act[p * hidden + i]` — and not planar. The second
// convolution reads every input channel at each of nine neighbours, so the
// inner loop walks `i` at a fixed `q`; interleaved that is `hidden`
// consecutive floats and one cache line, planar it is `hidden` reads a frame
// apart. Measured at 512x288 on Metal it is most of the pass.
@group(0) @binding(4) var<storage, read_write> act_a: array<f32>;
@group(0) @binding(5) var<storage, read_write> act_b: array<f32>;
// where the filtered illumination lands, in `history.wgsl`'s scratch layout
@group(0) @binding(6) var<storage, read_write> filtered: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> weights: array<f32>;

fn n_pixels() -> u32 {
    return params.width * params.height;
}

fn luminance(c: vec3<f32>) -> f32 {
    return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
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

fn guide_id(i: u32) -> f32 {
    return guides[2u * n_pixels() + i].w;
}

// `crate::neural::id_feature`, tap for tap.
//
// The id is a label and not a quantity, so it is hashed rather than fed: the
// only thing the convolution wants from it is whether two pixels are the same
// surface, and a hash makes that a difference it can see. Zero — background —
// stays zero.
fn id_feature(id: f32) -> f32 {
    if id <= 0.0 {
        return 0.0;
    }
    var h = u32(id) * 0x9E3779B9u;
    h = h ^ (h >> 15u);
    h = h * 0x85EBCA6Bu;
    h = h ^ (h >> 13u);
    return f32(h >> 8u) / 16777216.0;
}

fn demod_albedo(i: u32) -> vec3<f32> {
    return max(guide_albedo(i), vec3<f32>(DEMOD_FLOOR));
}

fn clamp_x(x: i32) -> u32 {
    return u32(clamp(x, 0, i32(params.width) - 1));
}

fn clamp_y(y: i32) -> u32 {
    return u32(clamp(y, 0, i32(params.height) - 1));
}

fn pixel_at(x: i32, y: i32) -> u32 {
    return clamp_y(y) * params.width + clamp_x(x);
}

// The demodulated illumination the predicted kernel averages.
fn illum(i: u32) -> vec3<f32> {
    return mean[i].rgb / demod_albedo(i);
}

// One input feature, mirroring `Weights::features_at` plane for plane.
//
// `conv1` evaluates the ninety it needs — ten planes over nine neighbours —
// into registers once, so this is called nine times an invocation and not
// nine times per output channel.
fn feature(c: u32, i: u32) -> f32 {
    let a = demod_albedo(i);
    let la = max(luminance(a), DEMOD_FLOOR);
    if c < 3u {
        let m = mean[i].rgb;
        var v: f32;
        if c == 0u { v = m.x / a.x; } else if c == 1u { v = m.y / a.y; } else { v = m.z / a.z; }
        return log(max(1.0 + v, 1e-8));
    }
    if c == 3u {
        return 1.0 / sqrt(max(stats[i].x, 1.0));
    }
    if c == 4u {
        return log(1.0 + sqrt(max(stats[i].w, 0.0) / (la * la)));
    }
    if c < 8u {
        let n = guide_normal(i);
        if c == 5u { return n.x; } else if c == 6u { return n.y; } else { return n.z; }
    }
    if c == 8u {
        let d = guide_depth(i);
        if d > 0.0 { return d / (d + DEPTH_SCALE); }
        return 0.0;
    }
    if c == 9u {
        return la;
    }
    return id_feature(guide_id(i));
}

// ─── pass 1: features → hidden, ReLU ──────────────────────────────────────

@compute @workgroup_size(8, 8)
fn conv1(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }
    let p = gid.y * params.width + gid.x;
    let x = i32(gid.x);
    let y = i32(gid.y);
    let h = params.hidden;

    // The nine neighbours, and the ninety-nine features over them, once for
    // the whole invocation rather than once per output channel. `feature` is a
    // couple of buffer reads and a log, and at 32 output channels the naive
    // loop evaluates every one of them thirty-two times.
    var nb: array<f32, 99>;
    for (var t = 0u; t < 9u; t = t + 1u) {
        let q = pixel_at(x + i32(t % 3u) - 1, y + i32(t / 3u) - 1);
        for (var i = 0u; i < C_IN; i = i + 1u) {
            nb[t * C_IN + i] = feature(i, q);
        }
    }

    for (var o = 0u; o < h; o = o + 1u) {
        var acc = weights[params.b1 + o];
        let row = params.w1 + o * 9u * C_IN;
        for (var t = 0u; t < 9u; t = t + 1u) {
            let wbase = row + t * C_IN;
            let nbase = t * C_IN;
            for (var i = 0u; i < C_IN; i = i + 1u) {
                acc = acc + weights[wbase + i] * nb[nbase + i];
            }
        }
        act_a[p * h + o] = max(acc, 0.0);
    }
}

// ─── pass 2: hidden → hidden, ReLU ────────────────────────────────────────

@compute @workgroup_size(8, 8)
fn conv2(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }
    let p = gid.y * params.width + gid.x;
    let x = i32(gid.x);
    let y = i32(gid.y);
    let h = params.hidden;
    var q: array<u32, 9>;
    for (var t = 0u; t < 9u; t = t + 1u) {
        q[t] = pixel_at(x + i32(t % 3u) - 1, y + i32(t / 3u) - 1) * h;
    }
    for (var o = 0u; o < h; o = o + 1u) {
        var acc = weights[params.b2 + o];
        let row = params.w2 + o * 9u * h;
        for (var t = 0u; t < 9u; t = t + 1u) {
            let wbase = row + t * h;
            let abase = q[t];
            for (var i = 0u; i < h; i = i + 1u) {
                acc = acc + weights[wbase + i] * act_a[abase + i];
            }
        }
        act_b[p * h + o] = max(acc, 0.0);
    }
}

// ─── pass 3: hidden → 25 logits, softmax, apply ───────────────────────────

@compute @workgroup_size(8, 8)
fn conv3_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }
    let p = gid.y * params.width + gid.x;
    let x = i32(gid.x);
    let y = i32(gid.y);
    let h = params.hidden;

    // A background pixel has no albedo to demodulate by and no geometry to
    // steer on; the à-trous pass skips it too. Pass the mean through.
    if guide_depth(p) <= 0.0 || stats[p].x >= f32(params.count_cutoff) {
        filtered[p] = vec4<f32>(illum(p), stats[p].w);
        return;
    }

    var q: array<u32, 9>;
    for (var t = 0u; t < 9u; t = t + 1u) {
        q[t] = pixel_at(x + i32(t % 3u) - 1, y + i32(t / 3u) - 1) * h;
    }
    var z: array<f32, 25>;
    var mx = -3.4e38;
    for (var o = 0u; o < K; o = o + 1u) {
        var acc = weights[params.b3 + o];
        let row = params.w3 + o * 9u * h;
        for (var t = 0u; t < 9u; t = t + 1u) {
            let wbase = row + t * h;
            let abase = q[t];
            for (var i = 0u; i < h; i = i + 1u) {
                acc = acc + weights[wbase + i] * act_b[abase + i];
            }
        }
        z[o] = acc;
        mx = max(mx, acc);
    }

    var sum = 0.0;
    for (var o = 0u; o < K; o = o + 1u) {
        let e = exp(z[o] - mx);
        z[o] = e;
        sum = sum + e;
    }
    let inv = 1.0 / sum;

    let r = TAPS / 2;
    var acc = vec3<f32>(0.0);
    for (var ky = 0; ky < TAPS; ky = ky + 1) {
        for (var kx = 0; kx < TAPS; kx = kx + 1) {
            let q = pixel_at(x + kx - r, y + ky - r);
            acc = acc + (z[u32(ky * TAPS + kx)] * inv) * illum(q);
        }
    }
    filtered[p] = vec4<f32>(acc, stats[p].w);
}
