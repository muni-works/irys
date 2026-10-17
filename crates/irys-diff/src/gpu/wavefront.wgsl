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
    paths: u32,           // paths in this chunk
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
    inst_base: u32,       // first instance record in `geo`, in vec4s
    nee: u32,             // next-event estimation + MIS on
    cap: u32,             // paths a chunk holds: light lanes start at cap × nw
    base: u32,            // this chunk's first path in the frame
    total: u32,           // paths in the frame
    _p0: u32,
    _p1: u32,
    sun: vec4<f32>,       // unit sun direction, κ
    env: vec4<f32>,       // sun radiance, sky scale, SUN_PICK, vMF normaliser κ / (2π (1 − e^{−2κ}))
}

const NONE: u32 = 0xffffffffu;
const LEAF: u32 = 0x80000000u;

struct Path {
    o: vec4<f32>,
    d: vec4<f32>,
    tp: vec4<f32>,        // throughput per wavelength
    rad: vec4<f32>,       // L_j per wavelength
    lam: vec4<f32>,       // wavelengths, nm
    smp: vec4<f32>,       // (cos_i, pdf, conductor vertex this bounce, pdf of the ray's own BSDF sample)
    ldir: vec4<f32>,      // light sample at this conductor vertex: (ω_L, s = cos w_L / p_L), s = 0 for none
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
// records: opaque prims (three vec4 each, from 0), instances (four from
// `inst_base`: world → local rows, then (blas root, prim base, albedo, _)),
// splats (four from `splat_base`). Node 0 is the top-level tree's root.
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

// `prb::Env::radiance`.
fn sky(d: vec3<f32>, lambda: f32) -> f32 {
    let tint = 1.0 + 0.4 * (550.0 - lambda) / 200.0;
    return tint * params.env.y * (0.15 + 0.25 * max(d.z, 0.0))
        + params.env.x * exp(params.sun.w * (dot(d, params.sun.xyz) - 1.0));
}

// `prb::Env::pdf`: SUN_PICK × vMF + the rest uniform over the sphere.
fn light_pdf(d: vec3<f32>) -> f32 {
    let vmf = params.env.w * exp(params.sun.w * (dot(d, params.sun.xyz) - 1.0));
    return params.env.z * vmf + (1.0 - params.env.z) * (0.25 / PI);
}

// `prb::Env::sample`.
fn light_sample(u: vec3<f32>) -> vec3<f32> {
    let phi = 2.0 * PI * u.z;
    if (u.x < params.env.z) {
        let k = params.sun.w;
        let c = clamp(1.0 + log(u.y + (1.0 - u.y) * exp(-2.0 * k)) / k, -1.0, 1.0);
        let r = sqrt(max(1.0 - c * c, 0.0));
        let m = params.sun.xyz;
        return basis_t(m) * (r * cos(phi)) + basis_b(m) * (r * sin(phi)) + m * c;
    }
    let z = 1.0 - 2.0 * u.y;
    let r = sqrt(max(1.0 - z * z, 0.0));
    return vec3<f32>(r * cos(phi), r * sin(phi), z);
}

// The power heuristic.
fn mis(a: f32, b: f32) -> f32 {
    let s = a * a + b * b;
    return select(0.0, a * a / s, s > 0.0);
}

struct Hit {
    t: f32,
    prim: u32,   // opaque prim record, NONE on a miss
    inst: u32,   // instance record
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
        // `d` is an instance-local direction, not unit: t stays the world
        // ray's parameter.
        let oc = o - r0.xyz;
        let a = dot(d, d);
        let b = dot(oc, d);
        let c = dot(oc, oc) - r2.w * r2.w;
        let disc = b * b - a * c;
        if (disc < 0.0) { return -1.0; }
        let sq = sqrt(disc);
        let t0 = (-b - sq) / a;
        if (t0 > tmin) { return t0; }
        return (-b + sq) / a;
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

// One bottom-level tree, from `root`, for a ray already in its instance's
// space: the closest prim in (tmin, tmax). Near child first (left on a tie),
// far child pushed under it; the walk is a function of the ray alone, so
// replay retraces it.
fn walk_blas(o: vec3<f32>, d: vec3<f32>, root: u32, prim_base: u32, tmin: f32, tmax: f32) -> Hit {
    let inv = 1.0 / d;
    var best = Hit(tmax, NONE, NONE);
    var stack: array<u32, 64>;
    var sp = 1u;
    stack[0] = root;
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
                let i = prim_base + prim_ids[k];
                let t = hit_prim(o, d, i, tmin);
                if (t > tmin && t < best.t) {
                    best = Hit(t, i, NONE);
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

// The world ray in instance `k`'s space, `(o, d)` with `d` not renormalised,
// so a local hit's t is the world ray's.
fn to_local(k: u32, o: vec3<f32>, d: vec3<f32>) -> array<vec3<f32>, 2> {
    let r = params.inst_base + 4u * k;
    let m0 = geo[r];
    let m1 = geo[r + 1u];
    let m2 = geo[r + 2u];
    return array<vec3<f32>, 2>(
        vec3<f32>(dot(m0.xyz, o) + m0.w, dot(m1.xyz, o) + m1.w, dot(m2.xyz, o) + m2.w),
        vec3<f32>(dot(m0.xyz, d), dot(m1.xyz, d), dot(m2.xyz, d)),
    );
}

// Closest opaque hit in (tmin, tmax), or with `any` the first found: the
// top-level tree (`irys_core::Tlas::flatten`, rooted at node 0) walked like
// a bottom-level one, each leaf's instances in leaf order, each through its
// own bottom-level walk.
fn intersect(o: vec3<f32>, d: vec3<f32>, tmax: f32, any: bool) -> Hit {
    // Origins are pushed off their surface by `offset_ray`.
    let tmin = 0.0;
    let inv = 1.0 / d;
    var best = Hit(tmax, NONE, NONE);
    var stack: array<u32, 32>;
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
                let inst = prim_ids[k];
                let info = geo[params.inst_base + 4u * inst + 3u];
                let l = to_local(inst, o, d);
                let h = walk_blas(l[0], l[1], bitcast<u32>(info.x), bitcast<u32>(info.y), tmin, best.t);
                if (h.prim != NONE) {
                    best = Hit(h.t, h.prim, inst);
                    if (any) { return best; }
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

// Off-surface origin, `prb::offset`: along the normal by
// ε = max(2⁻¹⁵ |x|∞, 2⁻¹⁷ (|o|∞ + t)). The first term is Wächter & Binder's
// ("A Fast and Robust Method for Avoiding Self-Intersection", Ray Tracing
// Gems ch. 6) 256 ulps of the position; their per-coordinate integer form
// assumes a barycentric hit point, and ours is o + t d, whose rounding grows
// with |o| + t — a floor at z = 0 seen from 5 m has z error ~ulp(5 m), not
// ulp(0) — hence the second. Both scale with the scene; hits need only t > 0.
fn offset_ray(x: vec3<f32>, n: vec3<f32>, o: vec3<f32>, t: f32) -> vec3<f32> {
    let ax = abs(x);
    let ao = abs(o);
    let e = max(max(ax.x, max(ax.y, ax.z)) * (1.0 / 32768.0), (max(ao.x, max(ao.y, ao.z)) + t) * (1.0 / 131072.0));
    return x + n * e;
}

// Anything in the way of a shadow ray to the environment.
fn occluded(o: vec3<f32>, d: vec3<f32>) -> bool {
    return intersect(o, d, 3.4e38, true).prim != NONE;
}

// The world normal at a hit: the prim's local normal by the inverse
// transpose of the instance's linear part (the transpose of world → local).
fn hit_normal(h: Hit, o: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    let l = to_local(h.inst, o, d);
    let nl = prim_normal(h.prim, l[0] + l[1] * h.t);
    let r = params.inst_base + 4u * h.inst;
    return normalize(geo[r].xyz * nl.x + geo[r + 1u].xyz * nl.y + geo[r + 2u].xyz * nl.z);
}

fn inst_albedo(k: u32) -> f32 {
    return geo[params.inst_base + 4u * k + 3u].z;
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
//
// A frame is traced in chunks of `cap` paths (the most one 1-D dispatch of
// the IR's kernels holds). Path `i` of a chunk is path `base + i` of the
// frame: its generator, pixel, and `L`/`δL` slots are the frame's; its
// state, graph columns and `δf` are the chunk's. Slots past the chunk's
// `paths` are dead paths with harmless lanes, so a short last chunk feeds
// the graph nothing.

// A 2-D grid flattened: kernels over a whole frame can need more than 65535
// workgroups (`GpuPrb::dispatch`).
fn flat(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * 64u;
}

// The path's wavelengths: the first draw of its generator, as `gen` makes
// them, so scoring can rebuild them for any path of the frame.
fn lambdas(s: ptr<function, vec2<u32>>) -> vec4<f32> {
    let u = rand(s);
    var lam = vec4<f32>(0.0);
    for (var k = 0u; k < params.nw; k = k + 1u) {
        let v = u + f32(k) / f32(params.nw);
        lam[k] = 380.0 + 400.0 * (v - floor(v));
    }
    return lam;
}

fn path_lambdas(path: u32) -> vec4<f32> {
    var s = rng_new(path);
    return lambdas(&s);
}

@compute @workgroup_size(64)
fn gen(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.cap) { return; }
    if (i >= params.paths) {
        paths[i].smp = vec4<f32>(0.0);
        paths[i].alive = 0u;
        return;
    }
    let g = params.base + i;
    var s = rng_new(g);
    let lam = lambdas(&s);
    let w = params.size.x;
    let h = params.size.y;
    let px = g / params.spp;
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
        vec4<f32>(0.0),
        s,
        1u,
        0u,
    );
}

// The harmless vertex: `f` of it is 0 and `update` seeds it no adjoint.
fn harmless(j: u32) {
    write_lane(j, 1.0, 1.0, 1.0, 1.0, vec3<f32>(0.0));
}

// See `prb::trace` for the estimator, NEE and MIS included, and what the
// replay owes each vertex; this is that, step for step.
@compute @workgroup_size(64)
fn bounce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.cap) { return; }
    var p = paths[i];
    let base = i * params.nw;
    let light_base = params.cap * params.nw + base;
    // Inactive lanes get a harmless vertex; `update` gives them no adjoint.
    for (var k = 0u; k < params.nw; k = k + 1u) {
        harmless(base + k);
        if (params.nee != 0u) {
            harmless(light_base + k);
        }
    }
    let prev_pdf = p.smp.w;
    p.smp = vec4<f32>(0.0, 0.0, 0.0, prev_pdf);
    p.ldir = vec4<f32>(0.0);
    if (p.alive == 0u) {
        paths[i] = p;
        return;
    }
    let o = p.o.xyz;
    let d = p.d.xyz;
    let hit = intersect(o, d, 3.4e38, false);
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
        var w = 1.0;
        if (params.nee != 0u && prev_pdf > 0.0) {
            w = mis(prev_pdf, light_pdf(d));
        }
        for (var k = 0u; k < params.nw; k = k + 1u) {
            p.rad[k] = p.rad[k] + p.tp[k] * sky(d, p.lam[k]) * w;
        }
        p.alive = 0u;
        paths[i] = p;
        return;
    }
    let x = o + d * hit.t;
    var n = hit_normal(hit, o, d);
    if (dot(n, d) > 0.0) {
        n = -n;
    }
    let conductor = prim_conductor(hit.prim);
    let t = basis_t(n);
    let b = basis_b(n);
    let origin = offset_ray(x, n, o, hit.t);
    var s = p.rng;
    // The light sample: three uniforms, drawn before the BSDF's whether or
    // not it lands.
    var lit = false;
    var ld = vec3<f32>(0.0);
    var lcos = 0.0;
    var lpdf = 0.0;
    if (params.nee != 0u) {
        let u = vec3<f32>(rand(&s), rand(&s), rand(&s));
        ld = light_sample(u);
        lcos = dot(ld, n);
        lpdf = light_pdf(ld);
        lit = lcos > 0.0 && !occluded(origin, ld);
    }
    if (!conductor) {
        let albedo = albedo_at(x) * inst_albedo(hit.inst);
        if (lit) {
            let w = mis(lpdf, lcos / PI);
            for (var k = 0u; k < params.nw; k = k + 1u) {
                p.rad[k] = p.rad[k] + p.tp[k] * albedo / PI * lcos * sky(ld, p.lam[k]) * w / lpdf;
            }
        }
        let u1 = rand(&s);
        let u2 = rand(&s);
        let r = sqrt(u1);
        let phi = 2.0 * PI * u2;
        let cz = sqrt(max(1.0 - u1, 0.0));
        let wi = t * (r * cos(phi)) + b * (r * sin(phi)) + n * cz;
        for (var k = 0u; k < params.nw; k = k + 1u) {
            p.tp[k] = p.tp[k] * albedo;
        }
        p.smp.w = cz / PI;
        p.o = vec4<f32>(origin, 0.0);
        p.d = vec4<f32>(normalize(wi), 0.0);
        p.rng = s;
        paths[i] = p;
        return;
    }
    let wo = -d;
    let lo = vec3<f32>(dot(wo, t), dot(wo, b), dot(wo, n));
    if (lit && lo.z > 0.0) {
        let li = vec3<f32>(dot(ld, t), dot(ld, b), lcos);
        let hl = normalize(li + lo);
        let ohl = dot(lo, hl);
        for (var k = 0u; k < params.nw; k = k + 1u) {
            write_lane(light_base + k, li.z, lo.z, hl.z, ohl, band_weights(p.lam[k]));
        }
        let pb = ggx_pdf(params.alpha, lo.z, hl.z);
        p.ldir = vec4<f32>(ld, lcos * mis(lpdf, pb) / lpdf);
        p.smp.z = 1.0;
    }
    let u = vec2<f32>(rand(&s), rand(&s));
    p.rng = s;
    let sm = ggx_sample(params.alpha, lo, u);
    let li = sm.xyz;
    let pdf = sm.w;
    if (li.z <= 0.0 || lo.z <= 0.0 || !(pdf > 0.0) || !(pdf < 3.4e38)) {
        // Dead after this vertex; its light sample still counts.
        p.alive = 0u;
        paths[i] = p;
        return;
    }
    let hh = normalize(li + lo);
    let oh = dot(lo, hh);
    for (var k = 0u; k < params.nw; k = k + 1u) {
        write_lane(base + k, li.z, lo.z, hh.z, oh, band_weights(p.lam[k]));
    }
    p.smp = vec4<f32>(li.z, pdf, 1.0, pdf);
    p.o = vec4<f32>(origin, 0.0);
    p.d = vec4<f32>(normalize(t * li.x + b * li.y + n * li.z), 0.0);
    paths[i] = p;
}

@compute @workgroup_size(64)
fn update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.cap) { return; }
    var p = paths[i];
    let base = i * params.nw;
    let frame = (params.base + i) * params.nw;
    let light_base = params.cap * params.nw + base;
    for (var k = 0u; k < params.nw; k = k + 1u) {
        df_out[base + k] = 0.0;
        if (params.nee != 0u) {
            df_out[light_base + k] = 0.0;
        }
    }
    if (p.smp.z == 0.0) { return; }
    var live = false;
    for (var k = 0u; k < params.nw; k = k + 1u) {
        let j = base + k;
        let fj = frame + k;
        // NEE first: C_j joins L_j before the BSDF lane's δf reads it.
        if (p.ldir.w > 0.0) {
            let g = p.tp[k] * p.ldir.w * sky(p.ldir.xyz, p.lam[k]);
            if (params.replay != 0u) {
                df_out[light_base + k] = dl_in[fj] * g;
            }
            p.rad[k] = p.rad[k] + g * f_in[light_base + k];
        }
        if (p.smp.y > 0.0) {
            let f = f_in[j];
            if (params.replay != 0u && f > 0.0) {
                df_out[j] = dl_in[fj] * (l_in[fj] - p.rad[k]) / f;
            }
            p.tp[k] = p.tp[k] * (f * p.smp.x / p.smp.y);
            live = live || p.tp[k] > 0.0;
        }
    }
    if (!live) {
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
        l_out[(params.base + i) * params.nw + k] = p.rad[k];
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
fn score(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let px = flat(gid, nwg);
    if (px >= params.pixels) { return; }
    var c = vec3<f32>(0.0);
    for (var s = 0u; s < params.spp; s = s + 1u) {
        let i = px * params.spp + s;
        let lam = path_lambdas(i);
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
// wavelength response, over every path of the frame. The wavelengths are
// rebuilt from the generators of `params.seed`, which is the adjoint frame's
// when this runs.
@compute @workgroup_size(64)
fn lane_adjoint(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat(gid, nwg);
    if (i >= params.total) { return; }
    let px = i / params.spp;
    let g = vec3<f32>(d_img[px * 3u], d_img[px * 3u + 1u], d_img[px * 3u + 2u]);
    let lam = path_lambdas(i);
    for (var k = 0u; k < params.nw; k = k + 1u) {
        dl_out[i * params.nw + k] = dot(g, lane_rgb(lam[k]));
    }
}
