// M1: the residual pass (docs/belief-codec.md, M1 item 6).
//
// classify: a joint colour / depth noise model decides, per pixel, whether
// the prediction explains the real frame. Colour is compared in sRGB with a
// floor plus a shot-like term, depth as optical z with a stereo-like
// σ_z = a + b z² widened by the belief's own σ; e = mean over channels of
// (Δc / σ_c)² + (Δz / σ_z)², explained when e < τ. A pixel the prediction
// misses but the camera ranges is unexplained in depth outright.
//
// cc_*: 4-connected components of the unexplained pixels, joined only across
// depth-continuous neighbours: min-label propagation with pointer jumping.
// The fixed point is the smallest pixel index of each component whatever
// order the threads ran in, so the labels are deterministic. stats / compact
// reduce each component with integer atomics only (count, bbox, Σ z in mm)
// and list the components above a size floor; the host sorts the list by
// root, so its order is deterministic too.

struct RP {
    size: vec4<u32>,    // w, h, first pixel of the predicted camera, flags (1: σ)
    cam: vec4<f32>,     // fx, fy, cx, cy
    noise: vec4<f32>,   // σ_c floor, shot k (σ_c² += k c), σ_z a, σ_z b
    misc: vec4<f32>,    // τ, jump abs (m), jump rel, -
    lim: vec4<u32>,     // min pixels per listed component, list capacity, -, -
}

@group(0) @binding(0) var<uniform> R: RP;
@group(0) @binding(1) var<storage, read> real_rgb: array<u32>;      // sRGB8, r | g << 8 | b << 16
@group(0) @binding(2) var<storage, read> real_z: array<f32>;        // optical z, m (≤ 0: none)
@group(0) @binding(3) var<storage, read> pred_rgb: array<vec4<f32>>; // linear rgb
@group(0) @binding(4) var<storage, read> pred_dist: array<f32>;     // ray distance, m (+∞: miss)
@group(0) @binding(5) var<storage, read> pred_sigma: array<f32>;    // m
@group(0) @binding(6) var<storage, read_write> mask: array<u32>;    // 1 explained, 0 not
@group(0) @binding(7) var<storage, read_write> label: array<u32>;
@group(0) @binding(8) var<storage, read_write> stats: array<atomic<u32>>;   // 7 per pixel
@group(0) @binding(9) var<storage, read_write> list: array<atomic<u32>>;    // count, then 8 per entry
@group(0) @binding(10) var<storage, read_write> flag: array<atomic<u32>>;             // changed, explained count
@group(0) @binding(11) var<storage, read_write> err: array<f32>;            // e per pixel

const NONE: u32 = 0xffffffffu;
const BIG: u32 = 65535u;

fn srgb_enc(c: f32) -> f32 {
    let x = clamp(c, 0.0, 1.0);
    if (x <= 0.0031308) { return 12.92 * x; }
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

fn pred_z(p: u32) -> f32 {
    let d = pred_dist[R.size.z + p];
    let u = f32(p % R.size.x);
    let v = f32(p / R.size.x);
    let a = (u - R.cam.z) / R.cam.x;
    let b = (v - R.cam.w) / R.cam.y;
    return d / sqrt(1.0 + a * a + b * b);
}

@compute @workgroup_size(64)
fn classify(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    let n = R.size.x * R.size.y;
    if (p >= n) { return; }
    let rc = real_rgb[p];
    let real = vec3<f32>(f32(rc & 255u), f32((rc >> 8u) & 255u), f32((rc >> 16u) & 255u)) / 255.0;
    let pl = pred_rgb[R.size.z + p].xyz;
    let pred = vec3<f32>(srgb_enc(pl.x), srgb_enc(pl.y), srgb_enc(pl.z));
    var e = 0.0;
    for (var k = 0u; k < 3u; k = k + 1u) {
        let s2 = R.noise.x * R.noise.x + R.noise.y * real[k];
        let dc = real[k] - pred[k];
        e = e + dc * dc / s2;
    }
    e = e / 3.0;
    let zr = real_z[p];
    let zp = pred_z(p);
    if (zr > 0.0) {
        if (zp > 0.0 && zp < 3.0e38) {
            var sz = R.noise.z + R.noise.w * zr * zr;
            var s2 = sz * sz;
            if ((R.size.w & 1u) != 0u) {
                let sb = pred_sigma[R.size.z + p];
                s2 = s2 + sb * sb;
            }
            let dz = zr - zp;
            e = e + dz * dz / s2;
        } else {
            e = 3.0e38;   // the camera ranges something the prediction misses
        }
    }
    err[p] = e;
    let ok = e < R.misc.x;
    if (ok) { atomicAdd(&flag[1], 1u); }
    mask[p] = select(0u, 1u, ok);
    label[p] = select(p, NONE, ok);
}

fn joined(p: u32, q: u32) -> bool {
    if (label[q] == NONE) { return false; }
    let a = real_z[p];
    let b = real_z[q];
    if (a > 0.0 && b > 0.0) {
        return abs(a - b) <= R.misc.y + R.misc.z * min(a, b);
    }
    return true;
}

@compute @workgroup_size(64)
fn cc_min(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    let w = R.size.x;
    let n = w * R.size.y;
    if (p >= n) { return; }
    let l = label[p];
    if (l == NONE) { return; }
    var m = l;
    let x = p % w;
    if (x > 0u && joined(p, p - 1u)) { m = min(m, label[p - 1u]); }
    if (x + 1u < w && joined(p, p + 1u)) { m = min(m, label[p + 1u]); }
    if (p >= w && joined(p, p - w)) { m = min(m, label[p - w]); }
    if (p + w < n && joined(p, p + w)) { m = min(m, label[p + w]); }
    if (m < l) {
        label[p] = m;
        atomicStore(&flag[0], 1u);
    }
}

@compute @workgroup_size(64)
fn cc_jump(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= R.size.x * R.size.y) { return; }
    let l = label[p];
    if (l == NONE) { return; }
    let ll = label[l];
    if (ll < l) {
        label[p] = ll;
        atomicStore(&flag[0], 1u);
    }
}

@compute @workgroup_size(64)
fn cc_stats(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    let w = R.size.x;
    if (p >= w * R.size.y) { return; }
    let r = label[p];
    if (r == NONE) { return; }
    let x = p % w;
    let y = p / w;
    let o = 7u * r;
    atomicAdd(&stats[o], 1u);
    atomicMax(&stats[o + 1u], BIG - x);
    atomicMax(&stats[o + 2u], BIG - y);
    atomicMax(&stats[o + 3u], x);
    atomicMax(&stats[o + 4u], y);
    let z = real_z[p];
    if (z > 0.0) {
        atomicAdd(&stats[o + 5u], u32(round(z * 1000.0)));
        atomicAdd(&stats[o + 6u], 1u);
    }
}

@compute @workgroup_size(64)
fn cc_compact(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= R.size.x * R.size.y) { return; }
    if (label[p] != p) { return; }
    let o = 7u * p;
    let c = atomicLoad(&stats[o]);
    if (c < R.lim.x) { return; }
    let i = atomicAdd(&list[0], 1u);
    if (i >= R.lim.y) { return; }
    let b = 1u + 8u * i;
    atomicStore(&list[b], p);
    atomicStore(&list[b + 1u], c);
    atomicStore(&list[b + 2u], BIG - atomicLoad(&stats[o + 1u]));
    atomicStore(&list[b + 3u], BIG - atomicLoad(&stats[o + 2u]));
    atomicStore(&list[b + 4u], atomicLoad(&stats[o + 3u]));
    atomicStore(&list[b + 5u], atomicLoad(&stats[o + 4u]));
    atomicStore(&list[b + 6u], atomicLoad(&stats[o + 5u]));
    atomicStore(&list[b + 7u], atomicLoad(&stats[o + 6u]));
}
