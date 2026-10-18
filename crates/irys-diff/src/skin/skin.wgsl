// The skin integrator on the device: `skin::render::trace_path` in f32, one
// thread per pixel per sample, the whole path in one invocation (a megakernel,
// not a wavefront). Every function here has a same-named f64 twin in
// `skin/render.rs`, `skin/mesh.rs`, `skin/optics.rs`, `skin/graph.rs` or
// `ggx.rs`; the RNG is `prb::Rng` exactly, so a pixel walks the CPU's path
// until f32 rounding moves a branch. The specular `f` is the `Spec` graph's
// formula written out (graph.rs), not the IR's emitted WGSL: this kernel is
// forward-only, and the gradient is the CPU's replay.

struct Light {
    c: vec4<f32>,   // centre, radiance
    u: vec4<f32>,   // half-edge, tint r
    v: vec4<f32>,   // half-edge, tint g
    n: vec4<f32>,   // unit normal (u × v), tint b
}

struct Params {
    size: vec4<u32>,      // w, h, full w, full h
    crop: vec4<u32>,      // x0, y0, spp, depth
    seed: vec4<u32>,      // lo, hi, sample index, first pixel of this chunk
    flags: vec4<u32>,     // sss, lights, max events, uniform chromophores
    eye: vec4<f32>,       // xyz, tan_half
    fwd: vec4<f32>,       // xyz, lens radius
    right: vec4<f32>,     // xyz, focus distance
    up: vec4<f32>,        // xyz, mm per unit
    th0: vec4<f32>,       // model θ0..3
    th1: vec4<f32>,       // θ4, env, bump, spec gain
    sm0: vec4<f32>,       // sampling θ0..3
    sm1: vec4<f32>,       // sampling θ4, -, -, -
    tex: vec4<u32>,       // texture w, h, pixels in the frame, write lanes
    resp: vec4<f32>,      // RESPONSE_NORM × band / (spp × NW)
    hair0: vec4<f32>,     // fibre: eta, beta_m, beta_n, alpha°
    hair1: vec4<f32>,     // eumelanin, pheomelanin, strands on, -
    lights: array<Light, 4>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> nodes: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> prim_ids: array<u32>;
// Nine vec4 per triangle: three positions (w = tangent sign), then per
// corner (normal, u) and (tangent, v).
@group(0) @binding(3) var<storage, read> tris: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> tex_a: array<vec4<f32>>;   // albedo rgb, spec
@group(0) @binding(5) var<storage, read> tex_b: array<vec4<f32>>;   // normal rgb
@group(0) @binding(6) var<storage, read> chromo: array<vec2<f32>>;  // m, h
@group(0) @binding(7) var<storage, read> table: array<vec4<f32>>;   // per nm: μa mel, μa blood, μa base, σs
@group(0) @binding(8) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> lanes: array<vec4<f32>>;

const PI: f32 = 3.14159265358979;
const NONE: u32 = 0xffffffffu;
const LEAF: u32 = 0x80000000u;
const NW: u32 = 4u;
const F0: f32 = 0.027777778;      // ((1.4 − 1)/(1.4 + 1))²
const MIX0: f32 = 0.85;
const MIX1: f32 = 0.15;
const G_HG: f32 = 0.8;

// ---- prb::Rng ----

const GOLDEN: vec2<u32> = vec2<u32>(0x7f4a7c15u, 0x9e3779b9u);
const M1: vec2<u32> = vec2<u32>(0x1ce4e5b9u, 0xbf58476du);
const M2: vec2<u32> = vec2<u32>(0x133111ebu, 0x94d049bbu);

fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}

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

fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let p = mul32(a.x, b.x);
    return vec2<u32>(p.x, p.y + a.x * b.y + a.y * b.x);
}

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
    var s = P.seed.xy ^ mul64(vec2<u32>(path, 0u), GOLDEN);
    _ = rng_next(&s);
    return s;
}

fn rand(s: ptr<function, vec2<u32>>) -> f32 {
    return f32(rng_next(s).y >> 8u) * (1.0 / 16777216.0);
}

// ---- spectra ----

// ggx::band_weights through render::lift: rgb on the 450/550/650 hats.
fn lift(rgb: vec3<f32>, l: f32) -> f32 {
    if (l <= 450.0) { return rgb.z; }
    if (l <= 550.0) {
        let t = (l - 450.0) / 100.0;
        return (1.0 - t) * rgb.z + t * rgb.y;
    }
    if (l <= 650.0) {
        let t = (l - 550.0) / 100.0;
        return (1.0 - t) * rgb.y + t * rgb.x;
    }
    return rgb.x;
}

// optics::coefficients.
fn coefficients(l: f32) -> vec4<f32> {
    let x = clamp(l - 380.0, 0.0, 400.0);
    let i = min(u32(floor(x)), 399u);
    let f = x - f32(i);
    return table[i] * (1.0 - f) + table[i + 1u] * f;
}

fn skew_gauss(x: f32, mu: f32, s1: f32, s2: f32) -> f32 {
    let s = select(s2, s1, x < mu);
    let t = (x - mu) / s;
    return exp(-0.5 * t * t);
}

// spectrum::hero_weight × 1 / (spp × NW).
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
    return rgb * P.resp.xyz;
}

fn schlick(c: f32) -> f32 {
    let m = 1.0 - clamp(c, 0.0, 1.0);
    let m2 = m * m;
    return F0 + (1.0 - F0) * m2 * m2 * m;
}

// ---- frames and sampling ----

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

fn cosine(n: vec3<f32>, u1: f32, u2: f32) -> vec4<f32> {
    let r = sqrt(u1);
    let phi = 2.0 * PI * u2;
    let cz = sqrt(max(1.0 - u1, 0.0));
    let d = basis_t(n) * (r * cos(phi)) + basis_b(n) * (r * sin(phi)) + n * cz;
    return vec4<f32>(d, cz);
}

fn hg_sample(w: vec3<f32>, u1: f32, u2: f32) -> vec3<f32> {
    let g = G_HG;
    let s = (1.0 - g * g) / (1.0 - g + 2.0 * g * u1);
    let c = clamp((1.0 + g * g - s * s) / (2.0 * g), -1.0, 1.0);
    let sn = sqrt(max(1.0 - c * c, 0.0));
    let phi = 2.0 * PI * u2;
    return basis_t(w) * (sn * cos(phi)) + basis_b(w) * (sn * sin(phi)) + w * c;
}

fn concentric(u1: f32, u2: f32) -> vec2<f32> {
    let a = 2.0 * u1 - 1.0;
    let b = 2.0 * u2 - 1.0;
    if (a == 0.0 && b == 0.0) { return vec2<f32>(0.0); }
    var r: f32;
    var phi: f32;
    if (abs(a) > abs(b)) {
        r = a;
        phi = 0.25 * PI * (b / a);
    } else {
        r = b;
        phi = 0.5 * PI - 0.25 * PI * (a / b);
    }
    return vec2<f32>(r * cos(phi), r * sin(phi));
}

// ggx::pdf at α (a² = α²): the reflected-direction density.
fn ggx_pdf(alpha: f32, co: f32, ch: f32) -> f32 {
    let a2 = alpha * alpha;
    let k = ch * ch * (a2 - 1.0) + 1.0;
    let d = a2 / (PI * k * k);
    let lam = (sqrt(1.0 + a2 * (1.0 - co * co) / (co * co)) - 1.0) * 0.5;
    return d / (1.0 + lam) / (4.0 * co);
}

// ggx::sample (Heitz 2018) → local wi.
fn ggx_sample(alpha: f32, wo: vec3<f32>, u1: f32, u2: f32) -> vec3<f32> {
    let vh = normalize(vec3<f32>(alpha * wo.x, alpha * wo.y, wo.z));
    let len2 = vh.x * vh.x + vh.y * vh.y;
    var t1 = vec3<f32>(1.0, 0.0, 0.0);
    if (len2 > 0.0) { t1 = vec3<f32>(-vh.y, vh.x, 0.0) / sqrt(len2); }
    let t2 = cross(vh, t1);
    let r = sqrt(u1);
    let phi = 2.0 * PI * u2;
    let p1 = r * cos(phi);
    let s = 0.5 * (1.0 + vh.z);
    let p2 = (1.0 - s) * sqrt(max(1.0 - p1 * p1, 0.0)) + s * r * sin(phi);
    let p3 = sqrt(max(1.0 - p1 * p1 - p2 * p2, 0.0));
    let nh = p1 * t1 + p2 * t2 + p3 * vh;
    let h = normalize(vec3<f32>(alpha * nh.x, alpha * nh.y, max(nh.z, 1e-12)));
    let oh = dot(wo, h);
    return 2.0 * oh * h - wo;
}

fn smith_lambda(a2: f32, c: f32) -> f32 {
    return (sqrt(1.0 + a2 * (1.0 - c * c) / (c * c)) - 1.0) * 0.5;
}

// graph::spec_f64.
fn spec_f(r1: f32, r2: f32, ci: f32, co: f32, ch: f32, oh: f32, k: f32) -> f32 {
    let m = 1.0 - oh;
    let m2 = m * m;
    let fr = F0 + (1.0 - F0) * m2 * m2 * m;
    var s = 0.0;
    for (var i = 0u; i < 2u; i = i + 1u) {
        let r = select(r2, r1, i == 0u);
        let a2 = r * r * r * r;
        let q = ch * ch * (a2 - 1.0) + 1.0;
        let d = a2 / (PI * q * q) / (1.0 + smith_lambda(a2, ci) + smith_lambda(a2, co));
        s = s + select(MIX1, MIX0, i == 0u) * d;
    }
    return k * fr * s / (4.0 * ci * co);
}

// graph::spec_pdf at the sampling θ.
fn spec_pdf(co: f32, ch: f32) -> f32 {
    return MIX0 * ggx_pdf(P.sm0.w * P.sm0.w, co, ch) + MIX1 * ggx_pdf(P.sm1.x * P.sm1.x, co, ch);
}

fn mis(a: f32, b: f32) -> f32 {
    let s = a * a + b * b;
    return select(0.0, a * a / s, s > 0.0);
}

// ---- the mesh ----

struct Hit {
    t: f32,
    prim: u32,
    u: f32,
    v: f32,
}

fn slab(o: vec3<f32>, inv: vec3<f32>, n: u32) -> vec2<f32> {
    let lo = (nodes[2u * n].xyz - o) * inv;
    let hi = (nodes[2u * n + 1u].xyz - o) * inv;
    let a = min(lo, hi);
    let b = max(lo, hi);
    return vec2<f32>(max(max(a.x, a.y), max(a.z, 0.0)), min(b.x, min(b.y, b.z)));
}

fn tri_hit(o: vec3<f32>, d: vec3<f32>, i: u32) -> vec3<f32> {
    let p0 = tris[9u * i].xyz;
    let e1 = tris[9u * i + 1u].xyz - p0;
    let e2 = tris[9u * i + 2u].xyz - p0;
    let pv = cross(d, e2);
    let det = dot(e1, pv);
    if (abs(det) <= 1e-12 * max(length(e1) * length(e2), 1.0)) { return vec3<f32>(-1.0, 0.0, 0.0); }
    let inv = 1.0 / det;
    let tv = o - p0;
    let u = dot(tv, pv) * inv;
    if (u < -1e-7 || u > 1.0 + 1e-7) { return vec3<f32>(-1.0, 0.0, 0.0); }
    let qv = cross(tv, e1);
    let v = dot(d, qv) * inv;
    if (v < -1e-7 || u + v > 1.0 + 1e-7) { return vec3<f32>(-1.0, 0.0, 0.0); }
    return vec3<f32>(dot(e2, qv) * inv, u, v);
}

// Closest hit in (0, tmax), or with `any` the first.
fn intersect(o: vec3<f32>, d: vec3<f32>, tmax: f32, any: bool) -> Hit {
    let inv = 1.0 / d;
    var best = Hit(tmax, NONE, 0.0, 0.0);
    var stack: array<u32, 64>;
    var sp = 1u;
    stack[0] = 0u;
    loop {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        let n = stack[sp];
        let e = slab(o, inv, n);
        if (e.x > e.y || e.x >= best.t) { continue; }
        let a = bitcast<u32>(nodes[2u * n].w);
        let b = bitcast<u32>(nodes[2u * n + 1u].w);
        if ((b & LEAF) != 0u) {
            for (var k = a; k < a + (b & ~LEAF); k = k + 1u) {
                let i = prim_ids[k];
                let h = tri_hit(o, d, i);
                if (h.x > 0.0 && h.x < best.t) {
                    best = Hit(h.x, i, h.y, h.z);
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

fn box_d2(p: vec3<f32>, n: u32) -> f32 {
    let lo = nodes[2u * n].xyz;
    let hi = nodes[2u * n + 1u].xyz;
    let d = max(max(lo - p, p - hi), vec3<f32>(0.0));
    return dot(d, d);
}

// accel::tri_d2 (Ericson 5.1.5).
fn tri_d2(p: vec3<f32>, i: u32) -> f32 {
    let a = tris[9u * i].xyz;
    let b = tris[9u * i + 1u].xyz;
    let c = tris[9u * i + 2u].xyz;
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { return dot(ap, ap); }
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { return dot(bp, bp); }
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let q = ap - ab * (d1 / (d1 - d3));
        return dot(q, q);
    }
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { return dot(cp, cp); }
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let q = ap - ac * (d2 / (d2 - d6));
        return dot(q, q);
    }
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let q = bp - (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
        return dot(q, q);
    }
    let den = 1.0 / (va + vb + vc);
    let q = ap - ab * (vb * den) - ac * (vc * den);
    return dot(q, q);
}

// accel::Accel::distance: the nearest triangle, nearer child first.
fn distance(p: vec3<f32>) -> f32 {
    var best = 3.4e38;
    var stack: array<u32, 64>;
    var sp = 1u;
    stack[0] = 0u;
    loop {
        if (sp == 0u) { break; }
        sp = sp - 1u;
        let n = stack[sp];
        if (box_d2(p, n) >= best) { continue; }
        let a = bitcast<u32>(nodes[2u * n].w);
        let b = bitcast<u32>(nodes[2u * n + 1u].w);
        if ((b & LEAF) != 0u) {
            for (var k = a; k < a + (b & ~LEAF); k = k + 1u) {
                best = min(best, tri_d2(p, prim_ids[k]));
            }
            continue;
        }
        if (box_d2(p, a) <= box_d2(p, b)) {
            stack[sp] = b;
            stack[sp + 1u] = a;
        } else {
            stack[sp] = a;
            stack[sp + 1u] = b;
        }
        sp = sp + 2u;
    }
    return sqrt(best);
}

fn occluded(o: vec3<f32>, d: vec3<f32>, dist: f32) -> bool {
    if (intersect(o, d, dist * (1.0 - 1e-4), true).prim != NONE) { return true; }
    // Fibres shadow too (no-ops without strands: `strands_none.wgsl`).
    return P.hair1.z != 0.0 && strand_hit(o, d, 1e-7, dist * (1.0 - 1e-4), true).seg != NONE;
}

fn offset_ray(x: vec3<f32>, n: vec3<f32>, o: vec3<f32>, t: f32) -> vec3<f32> {
    let ax = abs(x);
    let ao = abs(o);
    let e = max(max(ax.x, max(ax.y, ax.z)) * (1.0 / 32768.0), (max(ao.x, max(ao.y, ao.z)) + t) * (1.0 / 131072.0));
    return x + n * e;
}

fn geo_normal(i: u32) -> vec3<f32> {
    let p0 = tris[9u * i].xyz;
    return normalize(cross(tris[9u * i + 1u].xyz - p0, tris[9u * i + 2u].xyz - p0));
}

// mesh::shade's interpolated normal (unflipped, normalised; the geometric
// one where the blend cancels).
fn smooth_normal(h: Hit) -> vec3<f32> {
    let i = h.prim;
    let n = tris[9u * i + 3u].xyz * (1.0 - h.u - h.v) + tris[9u * i + 5u].xyz * h.u + tris[9u * i + 7u].xyz * h.v;
    if (length(n) > 1e-12) { return normalize(n); }
    return geo_normal(i);
}

// Bilinear, wrapped: mesh::Texture::sample.
fn texel(which: u32, u: f32, v: f32) -> vec4<f32> {
    let w = P.tex.x;
    let h = P.tex.y;
    let x = (u - floor(u)) * f32(w) - 0.5;
    let y = (v - floor(v)) * f32(h) - 0.5;
    let x0 = floor(x);
    let y0 = floor(y);
    let fx = x - x0;
    let fy = y - y0;
    let xa = u32((i32(x0) % i32(w) + i32(w)) % i32(w));
    let xb = u32((i32(x0) + 1 + i32(w)) % i32(w));
    let ya = u32((i32(y0) % i32(h) + i32(h)) % i32(h));
    let yb = u32((i32(y0) + 1 + i32(h)) % i32(h));
    var a: vec4<f32>;
    var b: vec4<f32>;
    var c: vec4<f32>;
    var d: vec4<f32>;
    if (which == 0u) {
        a = tex_a[ya * w + xa]; b = tex_a[ya * w + xb]; c = tex_a[yb * w + xa]; d = tex_a[yb * w + xb];
    } else {
        a = tex_b[ya * w + xa]; b = tex_b[ya * w + xb]; c = tex_b[yb * w + xa]; d = tex_b[yb * w + xb];
    }
    let top = a * (1.0 - fx) + b * fx;
    let bot = c * (1.0 - fx) + d * fx;
    return top * (1.0 - fy) + bot * fy;
}

struct Surf {
    ng: vec3<f32>,
    n: vec3<f32>,
    uv: vec2<f32>,
    k: f32,
    albedo: vec3<f32>,
}

// render::surface: mesh::shade, the normal map (mesh::bump), the spec map.
fn surface(h: Hit, d: vec3<f32>) -> Surf {
    let i = h.prim;
    let w0 = 1.0 - h.u - h.v;
    let c0a = tris[9u * i + 3u];
    let c0b = tris[9u * i + 4u];
    let c1a = tris[9u * i + 5u];
    let c1b = tris[9u * i + 6u];
    let c2a = tris[9u * i + 7u];
    let c2b = tris[9u * i + 8u];
    var ns = c0a.xyz * w0 + c1a.xyz * h.u + c2a.xyz * h.v;
    let uv = vec2<f32>(c0a.w * w0 + c1a.w * h.u + c2a.w * h.v, c0b.w * w0 + c1b.w * h.u + c2b.w * h.v);
    let tg = c0b.xyz * w0 + c1b.xyz * h.u + c2b.xyz * h.v;
    let sign = tris[9u * i].w;
    var ng = geo_normal(i);
    if (length(ns) > 1e-12) { ns = normalize(ns); } else { ns = ng; }
    var flip = 1.0;
    if (dot(ng, d) > 0.0) { flip = -1.0; }
    ng = ng * flip;
    ns = ns * flip;
    var n = ns;
    if (P.th1.z > 0.0) {
        let t0 = tg - ns * dot(ns, tg);
        if (length(t0) >= 1e-9) {
            let t = normalize(t0);
            let b = cross(ns, t) * sign;
            let rgb = texel(1u, uv.x, uv.y).xyz;
            let dd = rgb * 2.0 - vec3<f32>(1.0);
            let s = P.th1.z;
            let m = vec3<f32>(dd.x * s, dd.y * s, 1.0 - s + max(dd.z, 0.0) * s);
            let o = t * m.x + b * m.y + ns * m.z;
            if (length(o) >= 1e-9) { n = normalize(o); }
        }
    }
    if (dot(n, -d) <= 1e-4) { n = ng; }
    let ta = texel(0u, uv.x, uv.y);
    return Surf(ng, n, uv, P.th1.w * ta.w, ta.xyz);
}

// ---- lights ----

fn light_probs(x: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
    var p = vec4<f32>(0.0);
    var sum = 0.0;
    for (var i = 0u; i < P.flags.y; i = i + 1u) {
        let l = P.lights[i];
        let to = l.c.xyz - x;
        let d2 = max(dot(to, to), 1e-12);
        let d = to / sqrt(d2);
        let cr = max(dot(n, d), 0.02);
        let cl = max(-dot(d, l.n.xyz), 0.02);
        let area = 4.0 * length(cross(l.u.xyz, l.v.xyz));
        p[i] = l.c.w * area * cr * cl / d2;
        sum = sum + p[i];
    }
    return p / sum;
}

// light_probs at a fibre: power × the emitter's cosine / d², no receiver
// cosine (a fibre scatters into the whole sphere).
fn light_probs_fibre(x: vec3<f32>) -> vec4<f32> {
    var p = vec4<f32>(0.0);
    var sum = 0.0;
    for (var i = 0u; i < P.flags.y; i = i + 1u) {
        let l = P.lights[i];
        let to = l.c.xyz - x;
        let d2 = max(dot(to, to), 1e-12);
        let d = to / sqrt(d2);
        let cl = max(-dot(d, l.n.xyz), 0.02);
        let area = 4.0 * length(cross(l.u.xyz, l.v.xyz));
        p[i] = l.c.w * area * cl / d2;
        sum = sum + p[i];
    }
    return p / sum;
}

fn sample_light_fibre(x: vec3<f32>, u: vec3<f32>) -> LightSample {
    var out = LightSample(false, 0u, vec3<f32>(0.0), 0.0, 0.0);
    if (P.flags.y == 0u) { return out; }
    let p = light_probs_fibre(x);
    var i = P.flags.y - 1u;
    var acc = 0.0;
    for (var k = 0u; k < P.flags.y; k = k + 1u) {
        acc = acc + p[k];
        if (u.x < acc) {
            i = k;
            break;
        }
    }
    let l = P.lights[i];
    let y = l.c.xyz + l.u.xyz * (2.0 * u.y - 1.0) + l.v.xyz * (2.0 * u.z - 1.0);
    let to = y - x;
    let dist = length(to);
    let wi = to / dist;
    let cl = -dot(wi, l.n.xyz);
    if (cl <= 0.0) { return out; }
    let area = 4.0 * length(cross(l.u.xyz, l.v.xyz));
    return LightSample(true, i, wi, dist, p[i] * dist * dist / (cl * area));
}

fn light_le(i: u32, lam: f32) -> f32 {
    let l = P.lights[i];
    return l.c.w * lift(vec3<f32>(l.u.w, l.v.w, l.n.w), lam);
}

struct LightSample {
    ok: bool,
    i: u32,
    wi: vec3<f32>,
    dist: f32,
    pdf: f32,
}

fn sample_light(x: vec3<f32>, n: vec3<f32>, u: vec3<f32>) -> LightSample {
    var out = LightSample(false, 0u, vec3<f32>(0.0), 0.0, 0.0);
    if (P.flags.y == 0u) { return out; }
    let p = light_probs(x, n);
    var i = P.flags.y - 1u;
    var acc = 0.0;
    for (var k = 0u; k < P.flags.y; k = k + 1u) {
        acc = acc + p[k];
        if (u.x < acc) {
            i = k;
            break;
        }
    }
    let l = P.lights[i];
    let y = l.c.xyz + l.u.xyz * (2.0 * u.y - 1.0) + l.v.xyz * (2.0 * u.z - 1.0);
    let to = y - x;
    let dist = length(to);
    let wi = to / dist;
    let cl = -dot(wi, l.n.xyz);
    if (cl <= 0.0 || dot(wi, n) <= 0.0) { return out; }
    let area = 4.0 * length(cross(l.u.xyz, l.v.xyz));
    return LightSample(true, i, wi, dist, p[i] * dist * dist / (cl * area));
}

fn light_hit_t(i: u32, o: vec3<f32>, d: vec3<f32>, tmax: f32) -> f32 {
    let l = P.lights[i];
    let den = dot(d, l.n.xyz);
    if (den >= 0.0) { return -1.0; }
    let t = dot(l.c.xyz - o, l.n.xyz) / den;
    if (!(t > 0.0 && t < tmax)) { return -1.0; }
    let p = o + d * t - l.c.xyz;
    let a = dot(p, l.u.xyz) / dot(l.u.xyz, l.u.xyz);
    let b = dot(p, l.v.xyz) / dot(l.v.xyz, l.v.xyz);
    if (abs(a) <= 1.0 && abs(b) <= 1.0) { return t; }
    return -1.0;
}

fn light_pdf(x: vec3<f32>, n: vec3<f32>, i: u32, d: vec3<f32>, t: f32) -> f32 {
    let l = P.lights[i];
    let c = -dot(d, l.n.xyz);
    if (c <= 0.0) { return 0.0; }
    let area = 4.0 * length(cross(l.u.xyz, l.v.xyz));
    return light_probs(x, n)[i] * t * t / (c * area);
}

// ---- the medium ----

struct Walk {
    exited: bool,
    weight: f32,
    hit: Hit,
    w: vec3<f32>,
    x: vec3<f32>,
}

// render::walk_g at G_HG.
fn walk(x0: vec3<f32>, w0: vec3<f32>, model: vec2<f32>, samp: vec2<f32>, rng: ptr<function, vec2<u32>>) -> Walk {
    let st_m = model.x + model.y;
    let st_s = samp.x + samp.y;
    var x = x0;
    var w = w0;
    var weight = 1.0;
    // The safe sphere (accel.rs): centre, radius²; −1 for none yet.
    var sc = x0;
    var sr2 = -1.0;
    for (var e = 0u; e < P.flags.z; e = e + 1u) {
        let tt = -log(1.0 - rand(rng)) / st_s;
        let end = x + w * tt;
        var h = Hit(0.0, NONE, 0.0, 0.0);
        if (!(dot(end - sc, end - sc) < sr2)) {
            let r = distance(x) * (1.0 - 1e-4);
            sc = x;
            sr2 = r * r;
            if (!(dot(end - sc, end - sc) < sr2)) {
                // Only a boundary nearer than t matters: walk to t alone.
                h = intersect(x, w, tt, false);
            }
        }
        if (h.prim != NONE) {
            weight = weight * exp(-st_m * h.t) / exp(-st_s * h.t);
            return Walk(true, weight, h, w, x);
        }
        weight = weight * model.y * exp(-st_m * tt) / (st_s * exp(-st_s * tt));
        x = end;
        let ua = rand(rng);
        let ub = rand(rng);
        w = normalize(hg_sample(w, ua, ub));
    }
    return Walk(false, 0.0, Hit(0.0, NONE, 0.0, 0.0), w, x);
}

// ---- the path ----

fn spec_cols(li: vec3<f32>, lo: vec3<f32>) -> vec4<f32> {
    let hv = normalize(li + lo);
    return vec4<f32>(li.z, lo.z, hv.z, dot(lo, hv));
}

@compute @workgroup_size(64)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let px = P.seed.w + gid.x + gid.y * nwg.x * 64u;
    if (px >= P.tex.z) { return; }
    let spp = P.crop.z;
    let index = px * spp + P.seed.z;
    var rng = rng_new(index);
    let u0 = rand(&rng);
    var lam: vec4<f32>;
    for (var k = 0u; k < NW; k = k + 1u) {
        let v = u0 + f32(k) / f32(NW);
        lam[k] = 380.0 + 400.0 * (v - floor(v));
    }
    // render::Scene::camera_ray.
    let w = P.size.x;
    let fw = f32(P.size.z);
    let fh = f32(P.size.w);
    let xs = f32(P.crop.x + px % w) + rand(&rng);
    let ys = f32(P.crop.y + px / w) + rand(&rng);
    let sx = (2.0 * xs / fw - 1.0) * P.eye.w * fw / fh;
    let sy = (1.0 - 2.0 * ys / fh) * P.eye.w;
    let dpin = P.fwd.xyz + P.right.xyz * sx + P.up.xyz * sy;
    let lu1 = rand(&rng);
    let lu2 = rand(&rng);
    var ro = P.eye.xyz;
    var rd = normalize(dpin);
    if (P.fwd.w > 0.0) {
        let dk = concentric(lu1, lu2);
        let lens = P.eye.xyz + P.right.xyz * (dk.x * P.fwd.w) + P.up.xyz * (dk.y * P.fwd.w);
        let focus = P.eye.xyz + dpin * P.right.w;
        ro = lens;
        rd = normalize(focus - lens);
    }
    var tp = vec4<f32>(1.0);
    var rad = vec4<f32>(0.0);
    var prev_x = ro;
    var prev_n = vec3<f32>(0.0, 0.0, 1.0);
    var prev_pdf = 0.0;
    var prev_fibre = false;
    let r1 = P.th0.w;
    let r2 = P.th1.x;
    let mm = P.up.w;
    for (var depth = 0u; depth < P.crop.w; depth = depth + 1u) {
        let h = intersect(ro, rd, 3.4e38, false);
        var tmax = h.t;
        // A fibre nearer than the skin.
        var sh = SHit(tmax, NONE, vec3<f32>(0.0));
        if (P.hair1.z != 0.0) {
            sh = strand_hit(ro, rd, 1e-7, tmax, false);
            if (sh.seg != NONE) { tmax = sh.t; }
        }
        // Closest light before the surface.
        var li = NONE;
        var lt = tmax;
        for (var i = 0u; i < P.flags.y; i = i + 1u) {
            let t = light_hit_t(i, ro, rd, lt);
            if (t > 0.0) {
                li = i;
                lt = t;
            }
        }
        if (li != NONE) {
            var wm = 1.0;
            if (prev_pdf > 0.0) {
                var pl = light_pdf(prev_x, prev_n, li, rd, lt);
                if (prev_fibre) {
                    let l = P.lights[li];
                    let area = 4.0 * length(cross(l.u.xyz, l.v.xyz));
                    pl = light_probs_fibre(prev_x)[li] * lt * lt / (max(-dot(rd, l.n.xyz), 1e-12) * area);
                }
                wm = mis(prev_pdf, pl);
            }
            for (var k = 0u; k < NW; k = k + 1u) {
                rad[k] = rad[k] + tp[k] * light_le(li, lam[k]) * wm;
            }
            break;
        }
        if (sh.seg != NONE) {
            // A fibre: Chiang's BCSDF (bcsdf.wgsl) per hero wavelength,
            // lobes picked by the lanes' mean A_p, NEE through skin and
            // fibres, then a sampled direction — `hair::HairScene::trace`'s
            // vertex with four wavelengths.
            let x = ro + rd * sh.t;
            let tx = strand_tangent(sh.seg);
            var z = -(rd - tx * dot(rd, tx));
            if (length(z) < 1e-9) { break; }
            z = normalize(z);
            let y = cross(z, tx);
            let frad = strand_radius(sh.seg);
            let mel = strand_melanin(sh.seg);
            let hh = clamp(dot(sh.n, y), -1.0, 1.0);
            let wo = vec3<f32>(dot(-rd, tx), dot(-rd, y), dot(-rd, z));
            let fb = fib_new(P.hair0, hh);
            let g = fib_geometry(fb, wo);
            var sa = vec4<f32>(0.0);
            var ap = vec4<f32>(0.0);
            for (var k = 0u; k < NW; k = k + 1u) {
                sa[k] = fib_sigma_a(P.hair1.xy * mel, lam[k]);
                ap = ap + fib_ap(g.f, g.l, sa[k]) * (1.0 / f32(NW));
            }
            let off = sh.n * (2.0 * frad);
            let u3 = vec3<f32>(rand(&rng), rand(&rng), rand(&rng));
            let ls = sample_light_fibre(x, u3);
            if (ls.ok) {
                let xs = x + select(-off, off, dot(ls.wi, sh.n) > 0.0);
                if (!occluded(xs, ls.wi, ls.dist)) {
                    let wl = vec3<f32>(dot(ls.wi, tx), dot(ls.wi, y), dot(ls.wi, z));
                    let mn = fib_mn(fb, g, wl);
                    let wm = mis(ls.pdf, fib_pdf(fb, g, wl, ap));
                    for (var k = 0u; k < NW; k = k + 1u) {
                        rad[k] = rad[k] + tp[k] * fib_f(mn, g.f, g.l, sa[k]) * light_le(ls.i, lam[k]) * wm / ls.pdf;
                    }
                }
            }
            let u4 = vec4<f32>(rand(&rng), rand(&rng), rand(&rng), rand(&rng));
            let wl = fib_sample(fb, g, ap, u4);
            let pdf = fib_pdf(fb, g, wl, ap);
            if (!(pdf > 0.0) || pdf > 3.0e38) { break; }
            let mn = fib_mn(fb, g, wl);
            for (var k = 0u; k < NW; k = k + 1u) {
                tp[k] = tp[k] * fib_f(mn, g.f, g.l, sa[k]) / pdf;
            }
            let nd = normalize(tx * wl.x + y * wl.y + z * wl.z);
            prev_x = x;
            prev_n = sh.n;
            prev_pdf = pdf;
            prev_fibre = true;
            ro = x + select(-off, off, dot(nd, sh.n) > 0.0);
            rd = nd;
            continue;
        }
        prev_fibre = false;
        if (h.prim == NONE) {
            let env = P.th1.y * (0.55 + 0.45 * rd.z);
            rad = rad + tp * env;
            break;
        }
        let s = surface(h, rd);
        let ng = s.ng;
        let n = s.n;
        let wo = -rd;
        let origin = offset_ray(ro + rd * h.t, ng, ro, h.t);
        let bt = basis_t(n);
        let bb = basis_b(n);
        let lo = vec3<f32>(dot(wo, bt), dot(wo, bb), dot(wo, n));
        let co = max(lo.z, 1e-6);
        let pr = clamp(4.0 * schlick(co) * s.k, 0.1, 0.6);
        let fr_o = schlick(co);
        var fd = vec4<f32>(0.0);
        if (P.flags.x == 0u) {
            for (var k = 0u; k < NW; k = k + 1u) {
                fd[k] = lift(s.albedo, lam[k]) * (1.0 - fr_o) / PI;
            }
        }
        // 1. NEE.
        let u3 = vec3<f32>(rand(&rng), rand(&rng), rand(&rng));
        let ls = sample_light(origin, n, u3);
        if (ls.ok && dot(ls.wi, ng) > 0.0 && !occluded(origin, ls.wi, ls.dist)) {
            let lv = vec3<f32>(dot(ls.wi, bt), dot(ls.wi, bb), dot(ls.wi, n));
            let c = spec_cols(lv, lo);
            var pb = 0.0;
            if (lv.z > 0.0) {
                pb = pr * spec_pdf(c.y, c.z);
                if (P.flags.x == 0u) { pb = pb + (1.0 - pr) * lv.z / PI; }
            }
            let wm = mis(ls.pdf, pb);
            let fs = spec_f(r1, r2, c.x, c.y, c.z, c.w, s.k);
            let sc = lv.z * wm / ls.pdf;
            for (var k = 0u; k < NW; k = k + 1u) {
                rad[k] = rad[k] + tp[k] * sc * light_le(ls.i, lam[k]) * (fs + fd[k]);
            }
        }
        // 2. Reflect, or go through.
        let uc = rand(&rng);
        let um = rand(&rng);
        let u1 = rand(&rng);
        let u2 = rand(&rng);
        if (uc < pr || P.flags.x == 0u) {
            var lv: vec3<f32>;
            if (uc < pr) {
                let r = select(P.sm1.x, P.sm0.w, um < MIX0);
                lv = ggx_sample(r * r, lo, u1, u2);
            } else {
                let cw = cosine(n, u1, u2).xyz;
                lv = vec3<f32>(dot(cw, bt), dot(cw, bb), dot(cw, n));
            }
            if (lv.z <= 0.0) { break; }
            let c = spec_cols(lv, lo);
            var pdf = pr * spec_pdf(c.y, c.z);
            if (P.flags.x == 0u) { pdf = pdf + (1.0 - pr) * lv.z / PI; }
            if (!(pdf > 0.0) || pdf > 3.0e38) { break; }
            let wi = bt * lv.x + bb * lv.y + n * lv.z;
            if (dot(wi, ng) <= 0.0) { break; }
            let fs = spec_f(r1, r2, c.x, c.y, c.z, c.w, s.k);
            for (var k = 0u; k < NW; k = k + 1u) {
                tp[k] = tp[k] * (fs + fd[k]) * lv.z / pdf;
            }
            prev_x = origin;
            prev_n = n;
            prev_pdf = pdf;
            ro = origin;
            rd = normalize(wi);
            continue;
        }
        // 3. Into the medium: four walks.
        tp = tp * ((1.0 - fr_o) / (1.0 - pr));
        var m = 1.0;
        var hb = 1.0;
        if (P.flags.w == 0u) {
            let tw = P.tex.x;
            let th = P.tex.y;
            let xq = min(u32((s.uv.x - floor(s.uv.x)) * f32(tw)), tw - 1u);
            let yq = min(u32((s.uv.y - floor(s.uv.y)) * f32(th)), th - 1u);
            let cm = chromo[yq * tw + xq];
            m = cm.x;
            hb = cm.y;
        }
        let x0 = offset_ray(ro + rd * h.t, -ng, ro, h.t);
        let w0 = cosine(-ng, u1, u2).xyz;
        let cpick = min(u32(rand(&rng) * f32(NW)), NW - 1u);
        var cont = false;
        var cx = vec3<f32>(0.0);
        var cn = vec3<f32>(0.0);
        var cs = vec3<f32>(0.0);
        var cw = 0.0;
        for (var k = 0u; k < NW; k = k + 1u) {
            let cf = coefficients(lam[k]);
            let model = vec2<f32>(P.th0.z * (P.th0.x * m * cf.x + P.th0.y * hb * cf.y + cf.z) * mm, P.th0.z * cf.w * mm);
            let samp = vec2<f32>(P.sm0.z * (P.sm0.x * m * cf.x + P.sm0.y * hb * cf.y + cf.z) * mm, P.sm0.z * cf.w * mm);
            let wk = walk(x0, w0, model, samp, &rng);
            if (wk.exited) {
                var ne = geo_normal(wk.hit.prim);
                if (dot(ne, wk.w) < 0.0) { ne = -ne; }
                let xo = offset_ray(wk.x + wk.w * wk.hit.t, ne, wk.x, wk.hit.t);
                var ns = smooth_normal(wk.hit);
                if (dot(geo_normal(wk.hit.prim), ne) < 0.0) { ns = -ns; }
                if (dot(ns, ne) <= 0.0) { ns = ne; }
                let ue = vec3<f32>(rand(&rng), rand(&rng), rand(&rng));
                let el = sample_light(xo, ns, ue);
                if (el.ok && dot(el.wi, ne) > 0.0 && !occluded(xo, el.wi, el.dist)) {
                    let ci = dot(el.wi, ns);
                    let wm = mis(el.pdf, ci / PI);
                    rad[k] = rad[k] + tp[k] * wk.weight / PI * ci * light_le(el.i, lam[k]) * wm / el.pdf;
                }
                if (k == cpick) {
                    cont = true;
                    cx = xo;
                    cn = ne;
                    cs = ns;
                    cw = wk.weight;
                }
            }
        }
        if (!cont) { break; }
        for (var k = 0u; k < NW; k = k + 1u) {
            tp[k] = select(0.0, tp[k] * cw * f32(NW), k == cpick);
        }
        let v1 = rand(&rng);
        let v2 = rand(&rng);
        let cd = cosine(cs, v1, v2);
        if (dot(cd.xyz, cn) <= 0.0) { break; }
        prev_x = cx;
        prev_n = cs;
        prev_pdf = cd.w / PI;
        ro = cx;
        rd = normalize(cd.xyz);
    }
    var rgb = vec3<f32>(0.0);
    for (var k = 0u; k < NW; k = k + 1u) {
        rgb = rgb + rad[k] * lane_rgb(lam[k]);
    }
    accum[px] = accum[px] + vec4<f32>(rgb, 0.0);
    if (P.tex.w != 0u) {
        lanes[index] = rad;
    }
}
