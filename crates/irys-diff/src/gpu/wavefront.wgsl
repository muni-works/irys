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
// Scene (phase 2b): `prb::Scene`'s two trees, flattened from irys-core's
// `Bvh::flatten` into one node array — the opaque tree (triangles + analytic
// spheres and discs) rooted at node 0, the splat tree at `splat_root` — and
// walked with a fixed-order stack, so a ray visits the same nodes in the same
// order every time it is traced. Splats are 3DGRT-style: their 3σ boxes are
// the proxies in the tree, and a ray collects them in (t, index) order, K at a
// time, compositing front to back up to the opaque hit.

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
    resp: vec4<f32>,      // spectrum::RESPONSE_NORM × band / (spp × nw)
    pixels: u32,
    inv_n: f32,           // 1 / (pixels × 3), the L2 mean
    checker: f32,         // checker square size, 0 for plain
    splat_root: u32,      // node index of the splat tree, NONE without one
    splat_base: u32,      // first splat record in `geo`, in vec4s
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

const NONE: u32 = 0xffffffffu;
const LEAF: u32 = 0x80000000u;

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
// Scoring on the device (piece 1): image, L2 loss, per-lane δL.
@group(0) @binding(17) var<storage, read_write> img: array<f32>;
@group(0) @binding(18) var<storage, read> target_img: array<f32>;
@group(0) @binding(19) var<storage, read_write> d_img: array<f32>;
@group(0) @binding(20) var<storage, read_write> sq: array<f32>;
@group(0) @binding(21) var<storage, read_write> loss_out: array<f32>;
@group(0) @binding(22) var<storage, read_write> dl_out: array<f32>;
// The scene: nodes (min.xyz, a) (max.xyz, b | LEAF), leaf prim indices, and
// prim records (opaque: three vec4 each, splats: four from `splat_base`).
@group(0) @binding(23) var<storage, read> nodes: array<vec4<f32>>;
@group(0) @binding(24) var<storage, read> prim_ids: array<u32>;
@group(0) @binding(25) var<storage, read> geo: array<vec4<f32>>;

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
    prim: u32,   // combined opaque index, NONE on a miss
}

// The slab test, as `Ray::intersect_aabb`: (enter, exit), enter > exit on a
// miss.
fn slab(o: vec3<f32>, inv: vec3<f32>, n: u32) -> vec2<f32> {
    let lo = (nodes[2u * n].xyz - o) * inv;
    let hi = (nodes[2u * n + 1u].xyz - o) * inv;
    let a = min(lo, hi);
    let b = max(lo, hi);
    let enter = max(max(a.x, a.y), max(a.z, 0.0));
    let exit = min(b.x, min(b.y, b.z));
    return vec2<f32>(enter, exit);
}

fn node_a(n: u32) -> u32 {
    return bitcast<u32>(nodes[2u * n].w);
}

fn node_b(n: u32) -> u32 {
    return bitcast<u32>(nodes[2u * n + 1u].w);
}

// One opaque prim: kind in `r0.w` (0 triangle, 1 sphere, 2 disc).
fn hit_prim(o: vec3<f32>, d: vec3<f32>, i: u32, tmin: f32) -> f32 {
    let r0 = geo[3u * i];
    let r1 = geo[3u * i + 1u];
    let r2 = geo[3u * i + 2u];
    let kind = u32(r0.w);
    if (kind == 0u) {
        // Möller–Trumbore, double-sided.
        let e1 = r1.xyz - r0.xyz;
        let e2 = r2.xyz - r0.xyz;
        let pv = cross(d, e2);
        let det = dot(e1, pv);
        if (abs(det) < 1e-12) { return -1.0; }
        let inv = 1.0 / det;
        let tv = o - r0.xyz;
        let u = dot(tv, pv) * inv;
        if (u < 0.0 || u > 1.0) { return -1.0; }
        let qv = cross(tv, e1);
        let v = dot(d, qv) * inv;
        if (v < 0.0 || u + v > 1.0) { return -1.0; }
        return dot(e2, qv) * inv;
    }
    if (kind == 1u) {
        let oc = o - r0.xyz;
        let b = dot(oc, d);
        let c = dot(oc, oc) - r2.w * r2.w;
        let disc = b * b - c;
        if (disc < 0.0) { return -1.0; }
        let sq = sqrt(disc);
        let t0 = -b - sq;
        if (t0 > tmin) { return t0; }
        return -b + sq;
    }
    let n = r1.xyz;
    let den = dot(d, n);
    if (den == 0.0) { return -1.0; }
    let q = o - r0.xyz;
    let t = -dot(q, n) / den;
    let p = q + d * t;
    if (dot(p, p) > r2.w * r2.w) { return -1.0; }
    return t;
}

// Closest opaque hit. Near child first (left on a tie), far child pushed
// under it; the walk is a function of the ray alone, so replay retraces it.
fn intersect(o: vec3<f32>, d: vec3<f32>) -> Hit {
    let tmin = params.eps * 0.1;
    let inv = 1.0 / d;
    var best = Hit(3.4e38, NONE);
    var stack: array<u32, 64>;
    var sp = 1u;
    stack[0] = 0u;
    loop {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        let n = stack[sp];
        let e = slab(o, inv, n);
        if (e.x > e.y || e.x >= best.t || e.y <= tmin) { continue; }
        let a = node_a(n);
        let b = node_b(n);
        if ((b & LEAF) != 0u) {
            for (var k = a; k < a + (b & ~LEAF); k = k + 1u) {
                let i = prim_ids[k];
                let t = hit_prim(o, d, i, tmin);
                if (t > tmin && t < best.t) {
                    best = Hit(t, i);
                }
            }
            continue;
        }
        let ea = slab(o, inv, a);
        let eb = slab(o, inv, b);
        let ha = ea.x <= ea.y;
        let hb = eb.x <= eb.y;
        if (ha && hb) {
            if (ea.x <= eb.x) {
                stack[sp] = b;
                stack[sp + 1u] = a;
            } else {
                stack[sp] = a;
                stack[sp + 1u] = b;
            }
            sp = sp + 2u;
        } else if (ha) {
            stack[sp] = a;
            sp = sp + 1u;
        } else if (hb) {
            stack[sp] = b;
            sp = sp + 1u;
        }
    }
    return best;
}

fn prim_normal(i: u32, x: vec3<f32>) -> vec3<f32> {
    let r0 = geo[3u * i];
    let r1 = geo[3u * i + 1u];
    let kind = u32(r0.w);
    if (kind == 0u) {
        return normalize(cross(r1.xyz - r0.xyz, geo[3u * i + 2u].xyz - r0.xyz));
    }
    if (kind == 1u) {
        return normalize(x - r0.xyz);
    }
    return r1.xyz;
}

fn prim_conductor(i: u32) -> bool {
    return geo[3u * i + 1u].w == 0.0;
}

fn albedo_at(x: vec3<f32>) -> f32 {
    if (params.checker > 0.0) {
        let c = floor(x.x / params.checker) + floor(x.y / params.checker);
        if (c - 2.0 * floor(c * 0.5) == 1.0) {
            return params.albedo * 0.55;
        }
    }
    return params.albedo;
}

// ---- splats: 3DGRT-style ordered hit collection ----

const K: u32 = 16u;

// `Splats::intersect`: the Gaussian's peak along the ray, `(t, α)`; α < 0
// for a miss or past 3σ.
fn hit_splat(o: vec3<f32>, d: vec3<f32>, i: u32) -> vec2<f32> {
    let base = params.splat_base + 4u * i;
    let c = geo[base];
    let a0 = geo[base + 1u];
    let a1 = geo[base + 2u];
    let delta = o - c.xyz;
    let ad = vec3<f32>(
        a0.x * d.x + a0.y * d.y + a0.z * d.z,
        a0.y * d.x + a0.w * d.y + a1.x * d.z,
        a0.z * d.x + a1.x * d.y + a1.y * d.z,
    );
    let den = dot(d, ad);
    if (!(den > 0.0)) { return vec2<f32>(0.0, -1.0); }
    let t = -dot(delta, ad) / den;
    let q = delta + t * d;
    let aq = vec3<f32>(
        a0.x * q.x + a0.y * q.y + a0.z * q.z,
        a0.y * q.x + a0.w * q.y + a1.x * q.z,
        a0.z * q.x + a1.x * q.y + a1.y * q.z,
    );
    let d2 = max(dot(q, aq), 0.0);
    if (d2 > 9.0) { return vec2<f32>(t, -1.0); }
    return vec2<f32>(t, c.w * exp(-0.5 * d2));
}

// (t, i) strictly after (t0, i0): the order splats are composited in, a
// total order, so ties never depend on the walk.
fn after(t: f32, i: u32, t0: f32, i0: u32) -> bool {
    return t > t0 || (t == t0 && i0 != NONE && i > i0);
}

struct Segment {
    rad: vec3<f32>,   // Σ T α c, linear sRGB
    tr: f32,          // Π (1 − α)
}

// `splats::composite` over (tmin, tmax): rounds of "the K nearest splats
// after the last one composited", each a full walk of the splat tree that
// keeps a sorted K-buffer and culls boxes beyond its last entry once full.
fn composite(o: vec3<f32>, d: vec3<f32>, tmin: f32, tmax: f32) -> Segment {
    var seg = Segment(vec3<f32>(0.0), 1.0);
    if (params.splat_root == NONE) { return seg; }
    let inv = 1.0 / d;
    var t0 = tmin;
    var i0 = NONE;
    for (var round = 0u; round < 256u; round = round + 1u) {
        var kt: array<f32, 16>;
        var ki: array<u32, 16>;
        var ka: array<f32, 16>;
        var cnt = 0u;
        var stack: array<u32, 64>;
        var sp = 1u;
        stack[0] = params.splat_root;
        loop {
            if (sp == 0u) { break; }
            sp = sp - 1u;
            let n = stack[sp];
            let e = slab(o, inv, n);
            if (e.x > e.y || e.y < t0 || e.x >= tmax) { continue; }
            if (cnt == K && e.x > kt[K - 1u]) { continue; }
            let a = node_a(n);
            let b = node_b(n);
            if ((b & LEAF) != 0u) {
                for (var k = a; k < a + (b & ~LEAF); k = k + 1u) {
                    let i = prim_ids[k];
                    let h = hit_splat(o, d, i);
                    if (h.y <= 0.0 || h.x >= tmax || !after(h.x, i, t0, i0)) { continue; }
                    if (cnt == K && !(h.x < kt[K - 1u] || (h.x == kt[K - 1u] && i < ki[K - 1u]))) {
                        continue;
                    }
                    // Sorted insert by (t, i).
                    var j = min(cnt, K - 1u);
                    loop {
                        if (j == 0u) { break; }
                        let pt = kt[j - 1u];
                        if (pt < h.x || (pt == h.x && ki[j - 1u] < i)) { break; }
                        kt[j] = kt[j - 1u];
                        ki[j] = ki[j - 1u];
                        ka[j] = ka[j - 1u];
                        j = j - 1u;
                    }
                    kt[j] = h.x;
                    ki[j] = i;
                    ka[j] = h.y;
                    cnt = min(cnt + 1u, K);
                }
                continue;
            }
            // Children in index order: the K-buffer's order is the (t, i)
            // order whatever the walk, so no near-first sort is needed.
            stack[sp] = b;
            stack[sp + 1u] = a;
            sp = sp + 2u;
        }
        for (var k = 0u; k < cnt; k = k + 1u) {
            let alpha = clamp(ka[k], 0.0, 1.0);
            let c = geo[params.splat_base + 4u * ki[k] + 3u].xyz;
            seg.rad = seg.rad + seg.tr * alpha * c;
            seg.tr = seg.tr * (1.0 - alpha);
            if (seg.tr <= 1e-4) {
                seg.tr = 0.0;
                return seg;
            }
        }
        if (cnt < K) { return seg; }
        t0 = kt[K - 1u];
        i0 = ki[K - 1u];
    }
    return seg;
}

// `prb::splat_radiance`: sRGB onto the three hats (450 b, 550 g, 650 r).
fn splat_lambda(rgb: vec3<f32>, l: f32) -> f32 {
    let w = band_weights(l);
    return w.x * rgb.z + w.y * rgb.y + w.z * rgb.x;
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
    if (params.splat_root != NONE) {
        let seg = composite(o, d, params.eps * 0.1, hit.t);
        for (var k = 0u; k < params.nw; k = k + 1u) {
            p.rad[k] = p.rad[k] + p.tp[k] * splat_lambda(seg.rad, p.lam[k]);
            p.tp[k] = p.tp[k] * seg.tr;
        }
        if (seg.tr == 0.0) {
            p.alive = 0u;
            paths[i] = p;
            return;
        }
    }
    if (hit.prim == NONE) {
        for (var k = 0u; k < params.nw; k = k + 1u) {
            p.rad[k] = p.rad[k] + p.tp[k] * sky(d, p.lam[k]);
        }
        p.alive = 0u;
        paths[i] = p;
        return;
    }
    let x = o + d * hit.t;
    let n = prim_normal(hit.prim, x);
    let conductor = prim_conductor(hit.prim);
    let t = basis_t(n);
    let b = basis_b(n);
    let origin = x + n * params.eps;
    var s = p.rng;
    if (!conductor) {
        let albedo = albedo_at(x);
        let u1 = rand(&s);
        let u2 = rand(&s);
        let r = sqrt(u1);
        let phi = 2.0 * PI * u2;
        let wi = t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(1.0 - u1, 0.0));
        for (var k = 0u; k < params.nw; k = k + 1u) {
            p.tp[k] = p.tp[k] * albedo;
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

// ---- scoring: the image, the L2 loss and δL per lane, on the device ----
//
// `spectrum::hero_weight`, ported: the Wyman/Sloan/Shirley CIE fit, XYZ to
// linear sRGB, and the host's normalisation constants through `params.resp`
// (which also carries the 1 / (spp × nw) the image averages by).

fn skew_gauss(x: f32, mu: f32, s1: f32, s2: f32) -> f32 {
    let s = select(s2, s1, x < mu);
    let t = (x - mu) / s;
    return exp(-0.5 * t * t);
}

fn lane_rgb(l: f32) -> vec3<f32> {
    let x = 1.056 * skew_gauss(l, 599.8, 37.9, 31.0)
        + 0.362 * skew_gauss(l, 442.0, 16.0, 26.7)
        - 0.065 * skew_gauss(l, 501.1, 20.4, 26.2);
    let y = 0.821 * skew_gauss(l, 568.8, 46.9, 40.5) + 0.286 * skew_gauss(l, 530.9, 16.3, 31.1);
    let z = 1.217 * skew_gauss(l, 437.0, 11.8, 36.0) + 0.681 * skew_gauss(l, 459.0, 26.0, 13.8);
    let rgb = vec3<f32>(
        3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
        -0.969266 * x + 1.8760108 * y + 0.041556 * z,
        0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
    );
    return rgb * params.resp.xyz;
}

// One thread per pixel: its `spp × nw` lanes are contiguous and summed in
// lane order (as `prb::image`), then the residual against the target, the
// pixel's share of `∂loss/∂image`, and its squared error for `loss_sum`.
@compute @workgroup_size(64)
fn score(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    if (px >= params.pixels) { return; }
    var c = vec3<f32>(0.0);
    for (var s = 0u; s < params.spp; s = s + 1u) {
        let i = px * params.spp + s;
        let lam = paths[i].lam;
        for (var k = 0u; k < params.nw; k = k + 1u) {
            c = c + l_in[i * params.nw + k] * lane_rgb(lam[k]);
        }
    }
    let r = c - vec3<f32>(target_img[px * 3u], target_img[px * 3u + 1u], target_img[px * 3u + 2u]);
    for (var ch = 0u; ch < 3u; ch = ch + 1u) {
        img[px * 3u + ch] = c[ch];
        d_img[px * 3u + ch] = 2.0 * r[ch] * params.inv_n;
    }
    sq[px] = dot(r, r);
}

var<workgroup> red: array<f32, 256>;

// The L2 mean in one workgroup: each thread sums a fixed stride of pixels in
// order, then a fixed tree. No atomics, so the loss is the same bits every
// run.
@compute @workgroup_size(256)
fn loss_sum(@builtin(local_invocation_id) lid: vec3<u32>) {
    var acc = 0.0;
    for (var p = lid.x; p < params.pixels; p = p + 256u) {
        acc = acc + sq[p];
    }
    red[lid.x] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) {
            red[lid.x] = red[lid.x] + red[lid.x + s];
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        loss_out[0] = red[0] * params.inv_n;
    }
}

// `prb::lane_adjoint`: δL per lane from the pixel adjoint and the lane's own
// wavelength response. Runs after the adjoint seed's forward pass, so the
// wavelengths in `paths` are that frame's.
@compute @workgroup_size(64)
fn lane_adjoint(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.paths) { return; }
    let px = i / params.spp;
    let g = vec3<f32>(d_img[px * 3u], d_img[px * 3u + 1u], d_img[px * 3u + 2u]);
    let lam = paths[i].lam;
    for (var k = 0u; k < params.nw; k = k + 1u) {
        dl_out[i * params.nw + k] = dot(g, lane_rgb(lam[k]));
    }
}
