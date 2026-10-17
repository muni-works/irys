// The phase 2a wavefront: everything a path does except evaluate the BSDF.
//
// `f` is not here. It is the tang IR's own WGSL (`ggx::Graph`'s eval
// lowering), dispatched between `bounce` and `update` as its own kernels and
// read back through binding 9. Forward and replay dispatch the same
// pipelines over the same bind groups, so `f` is the same instructions on
// the same inputs in both passes, and so is everything in this file — the
// replay differs from the forward pass only in `params.replay`, which gates
// the `δf` write and nothing that feeds the path.
//
// Scene: `prb::Scene`'s ball and floor, as two analytic primitives.

struct Params {
    seed: vec2<u32>,      // (lo, hi) of the u64 frame seed
    size: vec2<u32>,      // image w, h
    spp: u32,
    nw: u32,              // wavelengths per path, 1..4
    paths: u32,
    replay: u32,
    eye: vec4<f32>,
    fwd: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    tan_half: f32,
    albedo: f32,
    alpha: f32,           // the roughness the sampler draws at
    eps: f32,             // ray offset off a surface, and the hit's t_min / 10
}

struct Path {
    o: vec4<f32>,
    d: vec4<f32>,
    tp: vec4<f32>,        // throughput per wavelength
    rad: vec4<f32>,       // L_j per wavelength
    lam: vec4<f32>,       // wavelengths, nm
    smp: vec4<f32>,       // (cos_i, pdf, on the ball this bounce, _)
    rng: vec2<u32>,       // SplitMix64 state (lo, hi)
    alive: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> paths: array<Path>;
@group(0) @binding(2) var<storage, read_write> c_ci: array<f32>;
@group(0) @binding(3) var<storage, read_write> c_co: array<f32>;
@group(0) @binding(4) var<storage, read_write> c_ch: array<f32>;
@group(0) @binding(5) var<storage, read_write> c_oh: array<f32>;
@group(0) @binding(6) var<storage, read_write> c_b0: array<f32>;
@group(0) @binding(7) var<storage, read_write> c_b1: array<f32>;
@group(0) @binding(8) var<storage, read_write> c_b2: array<f32>;
@group(0) @binding(9) var<storage, read> f_in: array<f32>;
@group(0) @binding(10) var<storage, read> l_in: array<f32>;
@group(0) @binding(11) var<storage, read> dl_in: array<f32>;
@group(0) @binding(12) var<storage, read_write> df_out: array<f32>;
@group(0) @binding(13) var<storage, read_write> l_out: array<f32>;
@group(0) @binding(14) var<storage, read_write> grad: array<f32>;
@group(0) @binding(15) var<storage, read> dtheta: array<f32>;

const PI: f32 = 3.14159265358979;

// ---- prb::Rng, exactly: SplitMix64 on (lo, hi) u32 pairs ----

const GOLDEN: vec2<u32> = vec2<u32>(0x7f4a7c15u, 0x9e3779b9u);
const M1: vec2<u32> = vec2<u32>(0x1ce4e5b9u, 0xbf58476du);
const M2: vec2<u32> = vec2<u32>(0x133111ebu, 0x94d049bbu);

fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}

// Full 32×32 → 64 product through 16-bit halves.
fn mul32(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xffffu;
    let a1 = a >> 16u;
    let b0 = b & 0xffffu;
    let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);
    return vec2<u32>((p00 & 0xffffu) | (mid << 16u), p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u));
}

// Low 64 bits of a 64×64 product (wrapping_mul).
fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let p = mul32(a.x, b.x);
    return vec2<u32>(p.x, p.y + a.x * b.y + a.y * b.x);
}

// Shift right by 0 < s < 32.
fn shr64(a: vec2<u32>, s: u32) -> vec2<u32> {
    return vec2<u32>((a.x >> s) | (a.y << (32u - s)), a.y >> s);
}

fn rng_next(s: ptr<function, vec2<u32>>) -> vec2<u32> {
    *s = add64(*s, GOLDEN);
    var z = *s;
    z = mul64(z ^ shr64(z, 30u), M1);
    z = mul64(z ^ shr64(z, 27u), M2);
    return z ^ shr64(z, 31u);
}

fn rng_new(path: u32) -> vec2<u32> {
    var s = params.seed ^ mul64(vec2<u32>(path, 0u), GOLDEN);
    _ = rng_next(&s);
    return s;
}

// The CPU's `(z >> 11) · 2⁻⁵³`, cut to the 24 bits an f32 holds: the same
// draw, rounded toward zero.
fn rand(s: ptr<function, vec2<u32>>) -> f32 {
    return f32(rng_next(s).y >> 8u) * (1.0 / 16777216.0);
}

// ---- the scene ----

fn sky(d: vec3<f32>, lambda: f32) -> f32 {
    let sun = normalize(vec3<f32>(0.5, -0.35, 0.8));
    let tint = 1.0 + 0.4 * (550.0 - lambda) / 200.0;
    return tint * (0.15 + 0.25 * max(d.z, 0.0)) + 6.0 * exp(40.0 * (dot(d, sun) - 1.0));
}

struct Hit {
    t: f32,
    prim: u32,   // 0 ball, 1 floor, 2 miss
}

fn intersect(o: vec3<f32>, d: vec3<f32>) -> Hit {
    let tmin = params.eps * 0.1;
    var best = Hit(3.4e38, 2u);
    let oc = o - vec3<f32>(0.0, 0.0, 1.0);
    let b = dot(oc, d);
    let c = dot(oc, oc) - 1.0;
    let disc = b * b - c;
    if (disc >= 0.0) {
        let sq = sqrt(disc);
        let t0 = -b - sq;
        let t1 = -b + sq;
        if (t0 > tmin) {
            best = Hit(t0, 0u);
        } else if (t1 > tmin) {
            best = Hit(t1, 0u);
        }
    }
    if (abs(d.z) > 1e-12) {
        let t = -o.z / d.z;
        let p = o + d * t;
        if (t > tmin && t < best.t && p.x * p.x + p.y * p.y <= 900.0) {
            best = Hit(t, 1u);
        }
    }
    return best;
}

// Duff et al. 2017, as `prb::basis`.
fn basis_t(n: vec3<f32>) -> vec3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
}

fn basis_b(n: vec3<f32>) -> vec3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return vec3<f32>(b, s + n.y * n.y * a, -n.y);
}

fn smith_lambda(a2: f32, c: f32) -> f32 {
    let c2 = c * c;
    return (sqrt(1.0 + a2 * (1.0 - c2) / c2) - 1.0) * 0.5;
}

// `ggx::pdf`: the sampler's own density, detached.
fn ggx_pdf(alpha: f32, co: f32, ch: f32) -> f32 {
    let a2 = alpha * alpha;
    let k = ch * ch * (a2 - 1.0) + 1.0;
    let d = a2 / (PI * k * k);
    return d / (1.0 + smith_lambda(a2, co)) / (4.0 * co);
}

// `ggx::sample` (Heitz 2018), f32. Returns (wi, pdf).
fn ggx_sample(alpha: f32, wo: vec3<f32>, u: vec2<f32>) -> vec4<f32> {
    let vh = normalize(vec3<f32>(alpha * wo.x, alpha * wo.y, wo.z));
    let len2 = vh.x * vh.x + vh.y * vh.y;
    var t1 = vec3<f32>(1.0, 0.0, 0.0);
    if (len2 > 0.0) {
        t1 = vec3<f32>(-vh.y, vh.x, 0.0) * (1.0 / sqrt(len2));
    }
    let t2 = cross(vh, t1);
    let r = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    let p1 = r * cos(phi);
    let s = 0.5 * (1.0 + vh.z);
    let p2 = (1.0 - s) * sqrt(max(1.0 - p1 * p1, 0.0)) + s * r * sin(phi);
    let p3 = sqrt(max(1.0 - p1 * p1 - p2 * p2, 0.0));
    let nh = t1 * p1 + t2 * p2 + vh * p3;
    let h = normalize(vec3<f32>(alpha * nh.x, alpha * nh.y, max(nh.z, 1e-12)));
    let oh = dot(wo, h);
    let wi = 2.0 * oh * h - wo;
    return vec4<f32>(wi, ggx_pdf(alpha, wo.z, h.z));
}

fn band_weights(l: f32) -> vec3<f32> {
    if (l <= 450.0) {
        return vec3<f32>(1.0, 0.0, 0.0);
    } else if (l <= 550.0) {
        let t = (l - 450.0) / 100.0;
        return vec3<f32>(1.0 - t, t, 0.0);
    } else if (l <= 650.0) {
        let t = (l - 550.0) / 100.0;
        return vec3<f32>(0.0, 1.0 - t, t);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

fn write_lane(j: u32, ci: f32, co: f32, ch: f32, oh: f32, b: vec3<f32>) {
    c_ci[j] = ci;
    c_co[j] = co;
    c_ch[j] = ch;
    c_oh[j] = oh;
    c_b0[j] = b.x;
    c_b1[j] = b.y;
    c_b2[j] = b.z;
}

// ---- the kernels ----

@compute @workgroup_size(64)
fn gen(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.paths) { return; }
    var s = rng_new(i);
    let u = rand(&s);
    var lam = vec4<f32>(0.0);
    for (var k = 0u; k < params.nw; k = k + 1u) {
        let v = u + f32(k) / f32(params.nw);
        lam[k] = 380.0 + 400.0 * (v - floor(v));
    }
    let w = params.size.x;
    let h = params.size.y;
    let px = i / params.spp;
    let x = f32(px % w) + rand(&s);
    let y = f32(px / w) + rand(&s);
    let sx = (2.0 * x / f32(w) - 1.0) * params.tan_half * f32(w) / f32(h);
    let sy = (1.0 - 2.0 * y / f32(h)) * params.tan_half;
    let d = normalize(params.fwd.xyz + params.right.xyz * sx + params.up.xyz * sy);
    paths[i] = Path(
        vec4<f32>(params.eye.xyz, 0.0),
        vec4<f32>(d, 0.0),
        vec4<f32>(1.0),
        vec4<f32>(0.0),
        lam,
        vec4<f32>(0.0),
        s,
        1u,
        0u,
    );
}

@compute @workgroup_size(64)
fn bounce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.paths) { return; }
    var p = paths[i];
    let base = i * params.nw;
    // Inactive lanes get a harmless vertex; `update` gives them no adjoint.
    for (var k = 0u; k < params.nw; k = k + 1u) {
        write_lane(base + k, 1.0, 1.0, 1.0, 1.0, vec3<f32>(0.0));
    }
    p.smp = vec4<f32>(0.0);
    if (p.alive == 0u) {
        paths[i] = p;
        return;
    }
    let o = p.o.xyz;
    let d = p.d.xyz;
    let hit = intersect(o, d);
    if (hit.prim == 2u) {
        for (var k = 0u; k < params.nw; k = k + 1u) {
            p.rad[k] = p.rad[k] + p.tp[k] * sky(d, p.lam[k]);
        }
        p.alive = 0u;
        paths[i] = p;
        return;
    }
    let x = o + d * hit.t;
    var n = vec3<f32>(0.0, 0.0, 1.0);
    if (hit.prim == 0u) {
        n = normalize(x - vec3<f32>(0.0, 0.0, 1.0));
    }
    let t = basis_t(n);
    let b = basis_b(n);
    let origin = x + n * params.eps;
    var s = p.rng;
    if (hit.prim == 1u) {
        let u1 = rand(&s);
        let u2 = rand(&s);
        let r = sqrt(u1);
        let phi = 2.0 * PI * u2;
        let wi = t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(1.0 - u1, 0.0));
        for (var k = 0u; k < params.nw; k = k + 1u) {
            p.tp[k] = p.tp[k] * params.albedo;
        }
        p.o = vec4<f32>(origin, 0.0);
        p.d = vec4<f32>(normalize(wi), 0.0);
        p.rng = s;
        paths[i] = p;
        return;
    }
    let wo = -d;
    let lo = vec3<f32>(dot(wo, t), dot(wo, b), dot(wo, n));
    let u = vec2<f32>(rand(&s), rand(&s));
    p.rng = s;
    let sm = ggx_sample(params.alpha, lo, u);
    let li = sm.xyz;
    let pdf = sm.w;
    if (li.z <= 0.0 || lo.z <= 0.0 || !(pdf > 0.0) || !(pdf < 3.4e38)) {
        p.alive = 0u;
        paths[i] = p;
        return;
    }
    let hh = normalize(li + lo);
    let oh = dot(lo, hh);
    for (var k = 0u; k < params.nw; k = k + 1u) {
        write_lane(base + k, li.z, lo.z, hh.z, oh, band_weights(p.lam[k]));
    }
    p.smp = vec4<f32>(li.z, pdf, 1.0, 0.0);
    p.o = vec4<f32>(origin, 0.0);
    p.d = vec4<f32>(normalize(t * li.x + b * li.y + n * li.z), 0.0);
    paths[i] = p;
}

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.paths) { return; }
    var p = paths[i];
    let base = i * params.nw;
    for (var k = 0u; k < params.nw; k = k + 1u) {
        df_out[base + k] = 0.0;
    }
    if (p.smp.z == 0.0) { return; }
    var lit = false;
    for (var k = 0u; k < params.nw; k = k + 1u) {
        let j = base + k;
        let f = f_in[j];
        if (params.replay != 0u && f > 0.0) {
            df_out[j] = dl_in[j] * (l_in[j] - p.rad[k]) / f;
        }
        p.tp[k] = p.tp[k] * (f * p.smp.x / p.smp.y);
        lit = lit || p.tp[k] > 0.0;
    }
    if (!lit) {
        p.alive = 0u;
    }
    paths[i] = p;
}

@compute @workgroup_size(64)
fn finish(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.paths) { return; }
    let p = paths[i];
    for (var k = 0u; k < params.nw; k = k + 1u) {
        l_out[i * params.nw + k] = p.rad[k];
    }
}

// δθ over bounces: the IR reduced this bounce's lanes to four numbers (one
// workgroup tree per parameter); add them in. One thread per parameter, in
// bounce order, so the gradient is deterministic — no float atomics.
@compute @workgroup_size(4)
fn accumulate(@builtin(local_invocation_id) lid: vec3<u32>) {
    grad[lid.x] = grad[lid.x] + dtheta[lid.x];
}
