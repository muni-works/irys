// Phase 5b: full scenes with OpenPBR materials on the device.
//
// A megakernel like skin.wgsl: one thread per pixel per sample, the whole
// path in one invocation, samples as separate dispatches added in order, so
// a frame is bit-deterministic. Hardware ray queries (the host prepends
// `enable wgpu_ray_query;`), instance masks for per-ray visibility (portals,
// hidden lights), and non-opaque triangles for cut-outs and thin glass.
//
// The material is look::openpbr's, written out: `At::new` (`at_new`),
// `At::columns` + `graph::stack` (`eval_dir`), `At::lobe_weights`,
// `At::pdf`, `At::sample`, over the same albedo tables (uploaded), with
// irys's three additions (metal colour, thin translucency, thin-walled pass)
// and subsurface as a Lambertian lobe of the subsurface colour (no random
// walk in full scenes). `probe` evaluates lanes for the f64 twin's test.
//
// Four hero wavelengths per path; the hero samples the direction, every
// wavelength is weighted by its own f over the hero's pdf, so dispersion
// needs no special case: a smooth dispersive refraction is a near-delta
// lobe the other wavelengths evaluate to ~0 off their own direction.

struct Params {
    size: vec4<u32>,      // w, h, spp, max bounces
    seed: vec4<u32>,      // lo, hi, sample, first pixel of this chunk
    flags: vec4<u32>,     // pixels, n emitters, n portal tris, rr depth
    eye: vec4<f32>,       // xyz, tan_half
    fwd: vec4<f32>,       // xyz, clamp (0 = off)
    right: vec4<f32>,
    up: vec4<f32>,
    resp: vec4<f32>,      // RESPONSE_NORM × band / NW
    sun_dir: vec4<f32>,   // xyz, cos r (0: no sun)
    sun_l: vec4<f32>,     // radiance hats, solid angle
    picks: vec4<f32>,     // cumulative: sun, env, emitters, portals
    env: vec4<u32>,       // w, h, on, rows offset
    minv0: vec4<f32>,     // M⁻¹ rows (rgb → hats)
    minv1: vec4<f32>,
    minv2: vec4<f32>,
    emit_off: vec4<u32>,  // emitter cdf offset 0, portal first index, -, -
    prof: vec4<u32>,      // profiling: x bits (PF_*), -, -, -
    samp: vec4<u32>,      // sampler: x bits (S_*), spp of the frame, -, -
}

@group(0) @binding(0) var<uniform> P: Params;
// 7 vec4 per triangle: positions (w: -), then per corner (n.xyz, u) and
// (v0, v1, v2, -).
@group(0) @binding(1) var<storage, read> tris: array<vec4<f32>>;
// Per instance: (first triangle, material, first emitter or NONE, -).
@group(0) @binding(2) var<storage, read> inst: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> mats: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> texels: array<u32>;
@group(0) @binding(5) var<storage, read> textab: array<vec4<u32>>;   // offset, w, h, -
@group(0) @binding(6) var<storage, read> tabs: array<f32>;           // diel, metal ×3, sheen
// Emitter triangles (world): 3 vec4: (p0, material bits), (p1, area), (p2, cdf).
@group(0) @binding(7) var<storage, read> emit: array<vec4<f32>>;
// The world: per texel (rgb, conditional cdf); then per row (marginal cdf,
// row probability, -, -); then portal skies (rgb, -).
@group(0) @binding(8) var<storage, read> envt: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> spec: array<f32>;
@group(0) @binding(11) var accel: acceleration_structure;
@group(0) @binding(12) var<storage, read_write> probe_io: array<vec4<f32>>;
// Profiling counters (`Opts::prof` with PF_STATS): see `scene_gpu::Stats`.
@group(0) @binding(13) var<storage, read_write> stats: array<atomic<u32>>;

const PF_STATS: u32 = 1u;      // count rays, candidates, classes, occupancy
const PF_NO_NEE: u32 = 2u;     // no next-event estimation
const PF_NO_SHADOW: u32 = 4u;  // NEE without its shadow ray (unoccluded)
const PF_LAMBERT: u32 = 8u;    // every material a Lambertian of its base colour
const PF_OPAQUE: u32 = 16u;    // no alpha test in traversal (cut-outs solid)

var<private> st_cand: u32 = 0u;
var<private> st_scand: u32 = 0u;
var<workgroup> wg_alive: array<atomic<u32>, 16>;
var<workgroup> wg_cls: array<atomic<u32>, 16>;

fn prof_on(b: u32) -> bool {
    return (P.prof.x & b) != 0u;
}

const PI: f32 = 3.14159265358979;
const NONE: u32 = 0xffffffffu;
const NW: u32 = 4u;
const BINS: u32 = 40u;
const MS: u32 = 20u;          // vec4 per material
const TN: u32 = 32u;
const NE: u32 = 33u;
const LN3: f32 = 1.0986122886681098;
const A_MIN: f32 = 1e-3;
const MU_BAR: f32 = 0.14285714285714285;
const DIEL_N: u32 = 33792u;   // TN·TN·NE
const SHEEN_OFF: u32 = 36864u; // DIEL_N + 3·TN·TN

const M_GEOMETRY: u32 = 0x01u;
const M_PORTAL: u32 = 0x02u;
const M_LIGHT_HIDDEN: u32 = 0x04u;
const M_LIGHT: u32 = 0x08u;

const F_THIN: u32 = 1u;
const F_CUTOUT: u32 = 2u;
const F_NONOPAQUE: u32 = 4u;
const F_EMIT_BACK: u32 = 8u;
const F_PORTAL: u32 = 16u;
const F_SPOT: u32 = 32u;
const F_EMIT_BOTH: u32 = 64u;
const F_EMITTER: u32 = 128u;
const F_METAL_SEP: u32 = 256u;

// ---- prb::Rng (as skin.wgsl), keyed by a 64-bit path index ----

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

fn rng_new(path: vec2<u32>) -> vec2<u32> {
    var s = P.seed.xy ^ mul64(path, GOLDEN);
    _ = rng_next(&s);
    return s;
}

fn rand(s: ptr<function, vec2<u32>>) -> f32 {
    return f32(rng_next(s).y >> 8u) * (1.0 / 16777216.0);
}

// ---- speed round 4: phase 3's sampler (`qmc.rs`, the same integers) ----
//
// Burley's shuffled, Owen-scrambled Sobol: a path draws 4-tuples, each
// tuple its own scramble of dimensions 0-3 keyed by (pixel, tuple), the
// sample index shuffled per tuple. `SOBOL_T`, `QKEY` (the host splices them
// in from `qmc::byte_tables` / `value_keys`) and `SPEC_CDF`
// (`qmc::Spectral`, the wavelength CDF by film response) are module constants.

const S_SOBOL: u32 = 1u;
const S_SPECTRAL: u32 = 2u;

fn qhash(x0: u32) -> u32 {
    var x = x0;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    return x ^ (x >> 16u);
}

fn qhash2(a: u32, b: u32) -> u32 {
    return qhash(a ^ (qhash(b) + 0x9e3779b9u));
}

fn owen(x0: u32, seed: u32) -> u32 {
    var x = reverseBits(x0);
    x = x + seed;
    x = x ^ (x * 0x6c50b47cu);
    x = x ^ (x * 0xb82f1e52u);
    x = x ^ (x * 0xc7afe638u);
    x = x ^ (x * 0x8d22f6e6u);
    return reverseBits(x);
}

fn sobol(i: u32, d: u32) -> u32 {
    if (d == 0u) {
        return reverseBits(i);
    }
    let t = (d - 1u) * 1024u;
    return SOBOL_T[t + (i & 255u)]
        ^ SOBOL_T[t + 256u + ((i >> 8u) & 255u)]
        ^ SOBOL_T[t + 512u + ((i >> 16u) & 255u)]
        ^ SOBOL_T[t + 768u + (i >> 24u)];
}

// The path's tuple counter and pixel key.
var<private> q_tuple: u32 = 0u;
var<private> q_pixel: u32 = 0u;

// The next 4-tuple: Sobol, or four of the path's hash generator.
fn draw4(rng: ptr<function, vec2<u32>>) -> vec4<f32> {
    if ((P.samp.x & S_SOBOL) == 0u) {
        return vec4<f32>(rand(rng), rand(rng), rand(rng), rand(rng));
    }
    let seed = qhash2(q_pixel, q_tuple);
    q_tuple = q_tuple + 1u;
    let j = owen(P.seed.z, seed);
    var u = vec4<f32>(0.0);
    for (var d = 0u; d < 4u; d = d + 1u) {
        let x = owen(sobol(j, d), qhash(seed ^ QKEY[d]));
        u[d] = f32(x >> 8u) * (1.0 / 16777216.0);
    }
    return u;
}

// `Spectral::lambda` and `Spectral::pdf` (relative to uniform).
fn spec_lambda(u: f32) -> f32 {
    var lo = 1u;
    var hi = 256u;
    loop {
        if (lo >= hi) { break; }
        let mid = (lo + hi) / 2u;
        if (SPEC_CDF[mid] <= u) { lo = mid + 1u; } else { hi = mid; }
    }
    let b = min(lo - 1u, 255u);
    let c0 = SPEC_CDF[b];
    let c1 = SPEC_CDF[b + 1u];
    let t = clamp((u - c0) / (c1 - c0), 0.0, 1.0);
    return 380.0 + 400.0 * (f32(b) + t) / 256.0;
}

fn spec_pdf(l: f32) -> f32 {
    if ((P.samp.x & S_SPECTRAL) == 0u) { return 1.0; }
    let b = min(u32(max((l - 380.0) / 400.0 * 256.0, 0.0)), 255u);
    return (SPEC_CDF[b + 1u] - SPEC_CDF[b]) * 256.0;
}

// ---- spectra ----

fn bands(l: f32) -> vec3<f32> {
    if (l <= 450.0) { return vec3<f32>(1.0, 0.0, 0.0); }
    if (l <= 550.0) {
        let t = (l - 450.0) / 100.0;
        return vec3<f32>(1.0 - t, t, 0.0);
    }
    if (l <= 650.0) {
        let t = (l - 550.0) / 100.0;
        return vec3<f32>(0.0, 1.0 - t, t);
    }
    return vec3<f32>(0.0, 0.0, 1.0);
}

// A colour already on the hats (450, 550, 650) at `l`.
fn lifth(h: vec3<f32>, l: f32) -> f32 {
    return dot(bands(l), h);
}

// Linear sRGB → hats (look::openpbr::hats).
fn hats(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(P.minv0.xyz, rgb), dot(P.minv1.xyz, rgb), dot(P.minv2.xyz, rgb));
}

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
    return rgb * P.resp.xyz;
}

// ---- small helpers ----

fn mis(a: f32, b: f32) -> f32 {
    let a2 = a * a;
    return a2 / max(a2 + b * b, 1e-30);
}

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

fn offset_ray(x: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let m = max(max(abs(x.x), abs(x.y)), abs(x.z));
    return x + n * max(m * 3e-6, 2e-5);
}

fn srgb_dec(c: f32) -> f32 {
    if (c <= 0.04045) { return c / 12.92; }
    return pow((c + 0.055) / 1.055, 2.4);
}

fn texel_rgba(t: u32, x: u32, y: u32) -> vec4<f32> {
    let e = textab[t];
    let v = texels[e.x + y * e.y + x];
    return vec4<f32>(f32(v & 0xffu), f32((v >> 8u) & 0xffu), f32((v >> 16u) & 0xffu), f32(v >> 24u)) / 255.0;
}

// Bilinear, wrapped; rgb decoded from sRGB, alpha linear.
fn tex(t: u32, uv: vec2<f32>) -> vec4<f32> {
    let e = textab[t];
    let w = f32(e.y);
    let h = f32(e.z);
    let x = (uv.x - floor(uv.x)) * w - 0.5;
    let y = (uv.y - floor(uv.y)) * h - 0.5;
    let x0 = floor(x);
    let y0 = floor(y);
    let fx = x - x0;
    let fy = y - y0;
    let xi0 = u32((i32(x0) % i32(e.y) + i32(e.y)) % i32(e.y));
    let yi0 = u32((i32(y0) % i32(e.z) + i32(e.z)) % i32(e.z));
    let xi1 = (xi0 + 1u) % e.y;
    let yi1 = (yi0 + 1u) % e.z;
    var a = texel_rgba(t, xi0, yi0) * (1.0 - fx) + texel_rgba(t, xi1, yi0) * fx;
    var b = texel_rgba(t, xi0, yi1) * (1.0 - fx) + texel_rgba(t, xi1, yi1) * fx;
    let c = a * (1.0 - fy) + b * fy;
    return vec4<f32>(srgb_dec(c.x), srgb_dec(c.y), srgb_dec(c.z), c.w);
}

// ---- the albedo tables (look::openpbr::Tables, look::sheen) ----

fn cell(x: f32) -> vec2<f32> {
    let s = clamp(x, 0.0, 1.0) * f32(TN - 1u);
    let i = min(floor(s), f32(TN - 2u));
    return vec2<f32>(i, s - i);
}

fn diel_slab(mu: f32, r: f32, k: u32) -> f32 {
    let a = cell(mu);
    let b = cell(r);
    let i = u32(a.x);
    let j = u32(b.x);
    let v00 = tabs[(i * TN + j) * NE + k];
    let v10 = tabs[((i + 1u) * TN + j) * NE + k];
    let v01 = tabs[(i * TN + j + 1u) * NE + k];
    let v11 = tabs[((i + 1u) * TN + j + 1u) * NE + k];
    let lo = v00 * (1.0 - a.y) + v10 * a.y;
    let hi = v01 * (1.0 - a.y) + v11 * a.y;
    return lo * (1.0 - b.y) + hi * b.y;
}

fn diel(mu: f32, r: f32, eta: f32) -> f32 {
    let s = clamp((log(eta) / LN3 + 1.0) * 0.5, 0.0, 1.0) * f32(NE - 1u);
    let k = min(u32(floor(s)), NE - 2u);
    let c = s - f32(k);
    return diel_slab(mu, r, k) * (1.0 - c) + diel_slab(mu, r, k + 1u) * c;
}

fn metal_tab(mu: f32, r: f32, w: u32) -> f32 {
    let a = cell(mu);
    let b = cell(r);
    let i = u32(a.x);
    let j = u32(b.x);
    let o = DIEL_N + w * TN * TN;
    let lo = tabs[o + i * TN + j] * (1.0 - a.y) + tabs[o + (i + 1u) * TN + j] * a.y;
    let hi = tabs[o + i * TN + j + 1u] * (1.0 - a.y) + tabs[o + (i + 1u) * TN + j + 1u] * a.y;
    return lo * (1.0 - b.y) + hi * b.y;
}

fn sheen_read1(off: u32, row: u32, x0: f32) -> f32 {
    let x = clamp(x0, 0.0, 1.0) * 31.0;
    let i = min(u32(x), 31u);
    let n = min(i + 1u, 31u);
    let t = x - f32(i);
    let d0 = tabs[SHEEN_OFF + off + row * 32u + i];
    return (1.0 - t) * d0 + t * tabs[SHEEN_OFF + off + row * 32u + n];
}

fn sheen_lookup(off: u32, mu: f32, r: f32) -> f32 {
    let y = clamp(r, 0.0, 1.0) * 31.0;
    let j = min(u32(y), 31u);
    let n = min(j + 1u, 31u);
    let t = y - f32(j);
    return (1.0 - t) * sheen_read1(off, j, mu) + t * sheen_read1(off, n, mu);
}

fn sheen_ltc(mu: f32, rf: f32) -> vec3<f32> {
    let r = clamp(rf, 1e-3, 1.0);
    let a = sheen_lookup(0u, mu, r);
    let b = sheen_lookup(1024u, mu, r);
    let rr = sheen_lookup(2048u, mu, r);
    if (abs(a) < 1e-5 || rr < 1e-5) { return vec3<f32>(1.0, 0.0, 0.0); }
    return vec3<f32>(a, b, rr);
}

fn sheen_brdf(rf: f32, wo: vec3<f32>, wi: vec3<f32>) -> f32 {
    if (wi.z <= 0.0 || wo.z <= 0.0) { return 0.0; }
    let l = sheen_ltc(wo.z, rf);
    let p = sqrt(wo.x * wo.x + wo.y * wo.y);
    var tx = 1.0;
    var ty = 0.0;
    if (p > 1e-9) { tx = wo.x / p; ty = wo.y / p; }
    let x = wi.x * tx + wi.y * ty;
    let y = -wi.x * ty + wi.y * tx;
    let z = wi.z;
    let ax = l.x * x + l.y * z;
    let len2 = ax * ax + (l.x * y) * (l.x * y) + z * z;
    let q = l.x / len2;
    return l.z * (z * q * q / PI) / wi.z;
}

// ---- OpenPBR pieces (look::openpbr) ----

fn alpha_of(r: f32) -> f32 {
    return A_MIN + (1.0 - A_MIN) * r * r;
}

fn fresnel(c0: f32, eta: f32) -> f32 {
    let c = clamp(c0, 0.0, 1.0);
    let g2 = eta * eta - 1.0 + c * c;
    if (g2 <= 0.0) { return 1.0; }
    let g = sqrt(g2);
    let a = (g - c) / (g + c);
    let b = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
    return 0.5 * a * a * (1.0 + b * b);
}

fn modulated_ior(eta: f32, xi0: f32) -> f32 {
    let fs = ((1.0 - eta) / (1.0 + eta)) * ((1.0 - eta) / (1.0 + eta));
    var hi = 1.0;
    if (fs > 0.0) { hi = 1.0 / fs; }
    let xi = clamp(xi0, 0.0, hi);
    let e = sign(eta - 1.0) * sqrt(xi * fs);
    return (1.0 + e) / max(1.0 - e, 1e-9);
}

fn airy(c: f32, nf: f32, d_nm: f32, ns: f32, lam: f32) -> f32 {
    let c1 = clamp(c, 1e-6, 1.0);
    let s1 = sqrt(max(1.0 - c1 * c1, 0.0));
    let s2 = s1 / nf;
    let c2 = sqrt(max(1.0 - s2 * s2, 0.0));
    let s3 = s1 / ns;
    let c3 = sqrt(max(1.0 - s3 * s3, 0.0));
    let delta = 4.0 * PI * nf * d_nm * c2 / lam;
    let cd = cos(delta);
    let rs12 = (c1 - nf * c2) / (c1 + nf * c2);
    let rs23 = (nf * c2 - ns * c3) / (nf * c2 + ns * c3);
    let rp12 = (nf * c1 - c2) / (nf * c1 + c2);
    let rp23 = (ns * c2 - nf * c3) / (ns * c2 + nf * c3);
    let ks = 2.0 * rs12 * rs23 * cd;
    let kp = 2.0 * rp12 * rp23 * cd;
    let s = (rs12 * rs12 + rs23 * rs23 + ks) / (1.0 + rs12 * rs12 * rs23 * rs23 + ks);
    let p = (rp12 * rp12 + rp23 * rp23 + kp) / (1.0 + rp12 * rp12 * rp23 * rp23 + kp);
    return 0.5 * (s + p);
}

fn ggx_d(a: f32, ch: f32) -> f32 {
    let a2 = a * a;
    let k = ch * ch * (a2 - 1.0) + 1.0;
    return a2 / (PI * k * k);
}

fn ggx_lambda(a: f32, c: f32) -> f32 {
    let c2 = max(c * c, 1e-12);
    return (sqrt(1.0 + a * a * (1.0 - c2) / c2) - 1.0) * 0.5;
}

fn vndf(a: f32, wo: vec3<f32>, u: vec2<f32>) -> vec3<f32> {
    let vh = normalize(vec3<f32>(a * wo.x, a * wo.y, wo.z));
    let len2 = vh.x * vh.x + vh.y * vh.y;
    var t1 = vec3<f32>(1.0, 0.0, 0.0);
    if (len2 > 0.0) { t1 = vec3<f32>(-vh.y, vh.x, 0.0) / sqrt(len2); }
    let t2 = cross(vh, t1);
    let r = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    let p1 = r * cos(phi);
    let s = 0.5 * (1.0 + vh.z);
    let p2 = (1.0 - s) * sqrt(max(1.0 - p1 * p1, 0.0)) + s * r * sin(phi);
    let p3 = sqrt(max(1.0 - p1 * p1 - p2 * p2, 0.0));
    let nh = p1 * t1 + p2 * t2 + p3 * vh;
    return normalize(vec3<f32>(a * nh.x, a * nh.y, max(nh.z, 1e-12)));
}

fn vndf_pdf(a: f32, co: f32, ch: f32, oh: f32) -> f32 {
    return ggx_d(a, ch) * abs(oh) / ((1.0 + ggx_lambda(a, co)) * co);
}

fn reflect_h(wo: vec3<f32>, h: vec3<f32>) -> vec3<f32> {
    return 2.0 * dot(wo, h) * h - wo;
}

// Refraction; zero under total internal reflection.
fn refract_h(wo: vec3<f32>, h: vec3<f32>, etap: f32) -> vec3<f32> {
    let c = dot(wo, h);
    let s2 = max(1.0 - c * c, 0.0) / (etap * etap);
    if (s2 >= 1.0) { return vec3<f32>(0.0); }
    let ct = sqrt(1.0 - s2);
    return -wo / etap + (c / etap - ct) * h;
}

fn cosine_dir(u: vec2<f32>) -> vec3<f32> {
    let r = sqrt(u.x);
    let phi = 2.0 * PI * u.y;
    return vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(1.0 - u.x, 0.0)));
}

// ---- the material ----

struct Mat {
    base: vec3<f32>, wb: f32,
    metal: vec3<f32>, met: f32,
    spec: vec3<f32>, sw: f32,
    r: f32, ior: f32, tw: f32, tdepth: f32,
    trans: vec3<f32>, abbe: f32,
    ssc: vec3<f32>, ss: f32,
    coat: vec3<f32>, cw: f32,
    cr: f32, cior: f32, cdark: f32, fw: f32,
    fuzz: vec3<f32>, fr: f32,
    filmw: f32, film_nm: f32, film_ior: f32, ef_coat: f32,
    emis: vec3<f32>, flags: u32,
    transl: vec3<f32>, tl: f32,
    base_tex: u32, metal_tex: u32, alpha_tex: u32, opacity: f32,
    xf: vec4<f32>,
    portal: u32, spot_c: vec3<f32>,
    spot_a: vec3<f32>, spot_cos: f32,
    spot_smooth: f32,
}

fn load_mat(i: u32) -> Mat {
    let o = i * MS;
    var m: Mat;
    let a0 = mats[o]; m.base = a0.xyz; m.wb = a0.w;
    let a1 = mats[o + 1u]; m.metal = a1.xyz; m.met = a1.w;
    let a2 = mats[o + 2u]; m.spec = a2.xyz; m.sw = a2.w;
    let a3 = mats[o + 3u]; m.r = a3.x; m.ior = a3.y; m.tw = a3.z; m.tdepth = a3.w;
    let a4 = mats[o + 4u]; m.trans = a4.xyz; m.abbe = a4.w;
    let a5 = mats[o + 5u]; m.ssc = a5.xyz; m.ss = a5.w;
    let a6 = mats[o + 6u]; m.coat = a6.xyz; m.cw = a6.w;
    let a7 = mats[o + 7u]; m.cr = a7.x; m.cior = a7.y; m.cdark = a7.z; m.fw = a7.w;
    let a8 = mats[o + 8u]; m.fuzz = a8.xyz; m.fr = a8.w;
    let a9 = mats[o + 9u]; m.filmw = a9.x; m.film_nm = a9.y; m.film_ior = a9.z; m.ef_coat = a9.w;
    let a10 = mats[o + 10u]; m.emis = a10.xyz; m.flags = bitcast<u32>(a10.w);
    let a11 = mats[o + 11u]; m.transl = a11.xyz; m.tl = a11.w;
    let a12 = mats[o + 12u];
    m.base_tex = bitcast<u32>(a12.x); m.metal_tex = bitcast<u32>(a12.y);
    m.alpha_tex = bitcast<u32>(a12.z); m.opacity = a12.w;
    m.xf = mats[o + 13u];
    let a14 = mats[o + 14u]; m.portal = bitcast<u32>(a14.x); m.spot_c = a14.yzw;
    let a15 = mats[o + 15u]; m.spot_a = a15.xyz; m.spot_cos = a15.w;
    m.spot_smooth = mats[o + 16u].x;
    return m;
}

// The material at one wavelength on one side (At), with everything that
// depends on ω_o only (the albedo tables at μ_o) computed once per vertex.
struct At {
    lam: f32, inside: bool, co: f32,
    eta_f: f32, eta_c: f32, eta_raw: f32, etap: f32,
    base: f32, metal: f32, sc: f32, tc: f32, ssc: f32, cc: f32, fc: f32, tlc: f32,
    es_u: f32, es_c: f32,     // E_s(μ_o) at η_f, η_c, the film's ratio in
    es_lw: f32,               // E_s(μ_o, η_f) without it (lobe weights)
    eb: f32,                  // the coated base's normal-incidence albedo
    e_coat: f32, k_s: f32,    // the coat's E(μ_o) and F(μ_o)
    e_fuzz: f32,              // fuzz colour × R(μ_o)
    em: f32,                  // the metal's E(μ_o)
}

fn ior_at(m: Mat, lam: f32) -> f32 {
    if (m.abbe <= 0.0) { return m.ior; }
    // spectrum::cauchy_index(n_d, V, λ µm)
    let lf = 0.48613;
    let ld = 0.58756;
    let lc = 0.65627;
    let b = (m.ior - 1.0) / (m.abbe * (1.0 / (lf * lf) - 1.0 / (lc * lc)));
    let a = m.ior - b / (ld * ld);
    let l = lam * 1e-3;
    return a + b / (l * l);
}

fn film_on(m: Mat, a: At) -> bool {
    return m.filmw > 0.0 && !a.inside;
}

fn fd_at(m: Mat, a: At, c: f32, eta: f32) -> f32 {
    let f = fresnel(c, eta);
    if (!film_on(m, a)) { return f; }
    let film = airy(c, m.film_ior, m.film_nm, eta, a.lam);
    return f + (film - f) * m.filmw;
}

fn film_metal(m: Mat, a: At, c: f32) -> f32 {
    let f0 = clamp(m.wb * a.metal, 0.0, 0.98);
    let n = (1.0 + sqrt(f0)) / (1.0 - sqrt(f0));
    return airy(c, m.film_ior, m.film_nm, n, a.lam);
}

fn xi_metal(m: Mat) -> f32 {
    return select(m.sw, 1.0, (m.flags & F_METAL_SEP) != 0u);
}

fn at_new(m: Mat, lam: f32, inside: bool, co0: f32) -> At {
    var a: At;
    let co = max(co0, 1e-9);
    a.lam = lam;
    a.inside = inside;
    a.co = co;
    let eta = ior_at(m, lam);
    let rel = max(eta / m.cior, m.cior / eta);
    let eu = modulated_ior(eta, m.sw);
    let ec = modulated_ior(rel, m.sw);
    if (inside) {
        a.eta_f = 1.0 / eu; a.eta_c = 1.0 / ec; a.etap = 1.0 / eta;
    } else {
        a.eta_f = eu; a.eta_c = ec; a.etap = eta;
    }
    a.eta_raw = eta;
    a.base = lifth(m.base, lam);
    a.metal = lifth(m.metal, lam);
    a.sc = lifth(m.spec, lam);
    a.tc = select(lifth(m.trans, lam), 1.0, m.tdepth > 0.0);
    a.ssc = lifth(m.ssc, lam);
    a.cc = lifth(m.coat, lam);
    a.fc = lifth(m.fuzz, lam);
    a.tlc = lifth(m.transl, lam);
    let fon = film_on(m, a);
    a.es_lw = diel(co, m.r, a.eta_f);
    var ru = 1.0;
    if (fon) { ru = fd_at(m, a, co, a.eta_f) / max(fresnel(co, a.eta_f), 1e-9); }
    a.es_u = a.es_lw * ru;
    a.es_c = a.es_u;
    a.eb = 0.0;
    a.e_coat = 0.0;
    a.k_s = 0.0;
    if (m.cw > 0.0) {
        var rc = 1.0;
        if (fon) { rc = fd_at(m, a, co, a.eta_c) / max(fresnel(co, a.eta_c), 1e-9); }
        a.es_c = diel(co, m.r, a.eta_c) * rc;
        // stack's eb: the coated base at normal incidence
        let es1 = diel(1.0, m.r, a.eta_c);
        let f0 = m.wb * a.metal;
        let kk = (1.0 - a.sc) * (f0 + (1.0 - f0) * pow(1.0 - MU_BAR, 5.0));
        let e5 = metal_tab(1.0, m.r, 1u);
        let em1 = xi_metal(m) * (f0 * (metal_tab(1.0, m.r, 0u) - e5) + e5 - kk * metal_tab(1.0, m.r, 2u));
        let bd = (1.0 - m.ss) * m.wb * ((1.0 - m.tl) * a.base + m.tl * a.tlc) + m.ss * a.ssc;
        a.eb = m.met * em1 + (1.0 - m.met) * (a.sc * es1 + (1.0 - a.sc * es1) * (1.0 - m.tw) * bd);
        let etac_side = select(m.cior, 1.0 / m.cior, inside);
        a.e_coat = diel(co, m.cr, etac_side);
        a.k_s = fresnel(co, etac_side);
    }
    a.e_fuzz = 0.0;
    if (m.fw > 0.0) { a.e_fuzz = a.fc * sheen_ltc(co, m.fr).z; }
    a.em = 0.0;
    if (m.met > 0.0) {
        let f0 = m.wb * a.metal;
        let e5 = metal_tab(co, m.r, 1u);
        a.em = max(f0 * (metal_tab(co, m.r, 0u) - e5) + e5, 0.0);
    }
    return a;
}

// The coat's transmission and darkening, `under`, for a path of `path`
// (1/μ'_i + 1/μ'_o).
fn coat_under(m: Mat, a: At, path: f32) -> f32 {
    let t_coat = pow(max(a.cc, 1e-12), 0.5 * path);
    let k_r = 1.0 - (1.0 - m.ef_coat) / (m.cior * m.cior);
    let fs = ((1.0 - a.eta_raw) / (1.0 + a.eta_raw)) * ((1.0 - a.eta_raw) / (1.0 + a.eta_raw));
    let xs_fs = clamp(m.sw * fs, 0.0, 1.0);
    let rd = 1.0 + (m.r - 1.0) * xs_fs;
    let rb = rd + (m.r - rd) * m.met;
    let kr = a.k_s + (k_r - a.k_s) * rb;
    let delta = (1.0 - kr) / (1.0 - a.eb * kr);
    let dark = 1.0 + (delta - 1.0) * m.cdark;
    return t_coat * (1.0 - a.e_coat) * dark;
}

fn coat_mu(m: Mat, mu: f32) -> f32 {
    return sqrt(max(1.0 - (1.0 - mu * mu) / (m.cior * m.cior), 1e-6));
}

// f(ω_o → ω_i) at one wavelength, local frame (ω_o.z = a.co > 0): the
// reflective lobes above, the refraction and the thin translucency below.
fn eval_dir(m: Mat, a: At, wo: vec3<f32>, wi: vec3<f32>) -> f32 {
    let co = a.co;
    let thin = (m.flags & F_THIN) != 0u;
    let up = wi.z > 0.0;
    if (!up && (m.tw <= 0.0 || thin) && m.tl <= 0.0) { return 0.0; }
    var h: vec3<f32>;
    var ci: f32;
    if (up) {
        h = normalize(wi + wo);
        ci = wi.z;
    } else {
        h = normalize(wo + wi * a.etap);
        if (h.z < 0.0) { h = -h; }
        ci = -wi.z;
    }
    let oh = max(abs(dot(wo, h)), 1e-9);
    let ih = abs(dot(wi, h));
    let cI = max(abs(ci), 1e-9);
    let ch = clamp(h.z, 1e-9, 1.0);
    // GGX at the specular roughness.
    let al = alpha_of(m.r);
    let a2 = al * al;
    let kq = ch * ch * (a2 - 1.0) + 1.0;
    let d = a2 / (PI * kq * kq);
    let g2 = 1.0 / (1.0 + ggx_lambda(al, cI) + ggx_lambda(al, co));
    let dg = d * g2 / (4.0 * cI * co);
    let bd = (1.0 - m.ss) * m.wb * (1.0 - m.tl) * a.base;
    let coated = m.cw > 0.0;
    var f_metal = 0.0;
    if (up && m.met > 0.0) {
        var film_m = 0.0;
        var wfm = 0.0;
        if (film_on(m, a)) { film_m = film_metal(m, a, oh); wfm = m.filmw; }
        let f0 = m.wb * a.metal;
        let fs_oh = f0 + (1.0 - f0) * pow(1.0 - oh, 5.0);
        let fs_bar = f0 + (1.0 - f0) * pow(1.0 - MU_BAR, 5.0);
        let kk = (1.0 - a.sc) * fs_bar;
        let mo = 1.0 - oh;
        let w82 = oh * mo * mo * mo * mo * mo * mo / (MU_BAR * pow(1.0 - MU_BAR, 6.0));
        f_metal = xi_metal(m) * ((1.0 - wfm) * (fs_oh - w82 * kk) + wfm * film_m);
    }
    // The two branches: uncoated (η_f) and, under a coat, coated (η_c).
    var rr = vec2<f32>(0.0);
    var tt = vec2<f32>(0.0);
    let nb = select(1u, 2u, coated);
    for (var k = 0u; k < nb; k = k + 1u) {
        let eta = select(a.eta_f, a.eta_c, k == 1u);
        let es = select(a.es_u, a.es_c, k == 1u);
        let fd = fd_at(m, a, oh, eta);
        let kd = (1.0 - m.tw) * (1.0 - a.sc * es);
        if (up) {
            let diffuse = kd * (bd + m.ss * a.ssc) / PI;
            rr[k] = m.met * f_metal * dg + (1.0 - m.met) * (a.sc * fd * dg + diffuse);
        } else {
            var t = 0.0;
            if (!thin && m.tw > 0.0) {
                let den = dot(wi, h) + dot(wo, h) / a.etap;
                let denom2 = max(den * den, 1e-12);
                let btdf = d * g2 * ih * oh / (cI * co * denom2 * a.etap * a.etap);
                t = m.tw * a.tc * (1.0 - fd) * btdf;
            }
            t = t + kd * (1.0 - m.ss) * m.wb * m.tl * a.tlc / PI;
            tt[k] = (1.0 - m.met) * t;
        }
    }
    var under = 0.0;
    if (coated) { under = coat_under(m, a, 1.0 / coat_mu(m, cI) + 1.0 / coat_mu(m, co)); }
    let fuzzmul = 1.0 - m.fw * a.e_fuzz;
    if (up) {
        var f_coat = 0.0;
        if (coated) {
            let etac_side = select(m.cior, 1.0 / m.cior, a.inside);
            let ac = alpha_of(m.cr);
            f_coat = fresnel(oh, etac_side) * ggx_d(ac, ch)
                / (1.0 + ggx_lambda(ac, cI) + ggx_lambda(ac, co)) / (4.0 * cI * co);
        }
        var f_fuzz = 0.0;
        if (m.fw > 0.0) { f_fuzz = a.fc * sheen_brdf(m.fr, wo, wi); }
        return m.fw * f_fuzz + fuzzmul * (m.cw * f_coat + (1.0 - m.cw) * rr.x + m.cw * under * rr.y);
    }
    return fuzzmul * ((1.0 - m.cw) * tt.x + m.cw * under * tt.y);
}

// Thin-walled glass: what passes straight through at cosine a.co.
fn thin_pass(m: Mat, a: At) -> f32 {
    let fd = fd_at(m, a, max(a.co, 1e-4), a.eta_f);
    var under = 0.0;
    if (m.cw > 0.0) { under = coat_under(m, a, 2.0 / coat_mu(m, a.co)); }
    let base = (1.0 - m.met) * m.tw * a.tc * (1.0 - fd);
    return (1.0 - m.fw * a.e_fuzz) * ((1.0 - m.cw) + m.cw * under) * base;
}

// Lobe weights at ω_o (detached): spec GGX, coat GGX, cosine (diffuse +
// fuzz + subsurface), transmission (refraction or thin pass), translucency.
struct Lobes { w: array<f32, 5> }

fn lobe_weights(m: Mat, a: At) -> Lobes {
    let es = a.es_lw;
    let ec = a.e_coat;
    let ef = a.e_fuzz;
    let above = max(1.0 - m.fw * ef, 0.0) * max(1.0 - m.cw * ec, 0.0);
    let die = 1.0 - m.met;
    let kd = (1.0 - m.tw) * max(1.0 - a.sc * es, 0.0);
    let spec = above * (m.met * a.em + die * a.sc * es);
    let diff = above * die * kd * ((1.0 - m.ss) * m.wb * (1.0 - m.tl) * a.base + m.ss * a.ssc);
    let trans = above * die * m.tw * (1.0 - es) * a.tc;
    let tl = above * die * kd * (1.0 - m.ss) * m.wb * m.tl * a.tlc;
    var l: Lobes;
    l.w[0] = spec + 1e-3;
    l.w[1] = m.cw * ec;
    l.w[2] = diff + m.fw * ef;
    l.w[3] = trans;
    l.w[4] = tl;
    if (a.inside) {
        l.w[0] = l.w[0] + l.w[2];
        l.w[1] = 0.0;
        l.w[2] = 0.0;
        l.w[4] = 0.0;
    }
    let s = l.w[0] + l.w[1] + l.w[2] + l.w[3] + l.w[4];
    l.w[0] = l.w[0] / s;
    l.w[1] = l.w[1] / s;
    l.w[2] = l.w[2] / s;
    l.w[3] = l.w[3] / s;
    l.w[4] = l.w[4] / s;
    return l;
}

// Directional pdf of ω_i under the lobe mixture (the thin pass is a
// discrete event and has no density).
fn lobe_pdf(m: Mat, a: At, l: Lobes, wo: vec3<f32>, wi: vec3<f32>) -> f32 {
    return lobe_pdf_e(m, a.etap, l, wo, wi);
}

fn lobe_pdf_e(m: Mat, etap: f32, l: Lobes, wo: vec3<f32>, wi: vec3<f32>) -> f32 {
    let co = wo.z;
    let lw = l.w;
    if (wi.z > 0.0) {
        let h = normalize(wi + wo);
        let oh = dot(wo, h);
        let jac = 1.0 / (4.0 * max(abs(oh), 1e-12));
        let ps = vndf_pdf(alpha_of(m.r), co, h.z, oh) * jac;
        let pc = vndf_pdf(alpha_of(m.cr), co, h.z, oh) * jac;
        return lw[0] * ps + lw[1] * pc + lw[2] * wi.z / PI;
    }
    var p = lw[4] * (-wi.z) / PI;
    if ((m.flags & F_THIN) == 0u) {
        let e = etap;
        var h = normalize(wo + wi * e);
        if (h.z < 0.0) { h = -h; }
        let oh = dot(wo, h);
        let ih = dot(wi, h);
        if (oh > 0.0 && ih < 0.0) {
            let den = ih + oh / e;
            p = p + lw[3] * vndf_pdf(alpha_of(m.r), co, h.z, oh) * abs(ih) / (den * den);
        }
    }
    return p;
}

// The BSDF density of ω_i as the lanes' path densities weigh it (for MIS
// against a light sample): Σ q_j p_j / Σ q_j.
fn spectral_pdf(m: Mat, at: ptr<function, array<At, 4>>, l: Lobes, wo: vec3<f32>, wi: vec3<f32>, q: vec4<f32>) -> f32 {
    var p = 0.0;
    for (var k = 0u; k < NW; k = k + 1u) {
        p = p + q[k] * lobe_pdf(m, (*at)[k], l, wo, wi);
    }
    return p / (q.x + q.y + q.z + q.w);
}

fn qbar(q: vec4<f32>) -> f32 {
    return max((q.x + q.y + q.z + q.w) * 0.25, 1e-30);
}

// Sample: (ω_i, lobe). Lobe 3 on a thin material is the pass (ω_i = −ω_o).
struct Pick { wi: vec3<f32>, lobe: u32 }

fn lobe_sample(m: Mat, etap: f32, l: Lobes, wo: vec3<f32>, u: vec3<f32>) -> Pick {
    var acc = 0.0;
    var k = 4u;
    var lw = l.w;
    for (var i = 0u; i < 5u; i = i + 1u) {
        acc = acc + lw[i];
        if (u.x < acc) { k = i; break; }
    }
    let uv = u.yz;
    var p: Pick;
    p.lobe = k;
    if (k == 0u) {
        p.wi = reflect_h(wo, vndf(alpha_of(m.r), wo, uv));
    } else if (k == 1u) {
        p.wi = reflect_h(wo, vndf(alpha_of(m.cr), wo, uv));
    } else if (k == 2u) {
        p.wi = cosine_dir(uv);
    } else if (k == 3u) {
        if ((m.flags & F_THIN) != 0u) {
            p.wi = -wo;
        } else {
            p.wi = refract_h(wo, vndf(alpha_of(m.r), wo, uv), etap);
        }
    } else {
        let c = cosine_dir(uv);
        p.wi = vec3<f32>(c.x, c.y, -c.z);
    }
    return p;
}


// ---- the four lanes at once ----
//
// Where the index does not depend on the wavelength (no dispersion, no thin
// film), every geometric factor (GGX, Fresnel, the tables at μ_o) is the
// same for the four lanes and only the colours differ: At4 / eval4 / lobes4
// / thin4 are At / eval_dir / lobe_weights / thin_pass with the colours as
// vec4, computed once. Same arithmetic per lane as the scalar path, which
// stays for dispersive and filmed materials (`fast_ok`).

struct At4 {
    inside: bool, co: f32,
    eta_f: f32, eta_c: f32, eta_raw: f32, etap: f32,
    base: vec4<f32>, metal: vec4<f32>, sc: vec4<f32>, tc: vec4<f32>,
    ssc: vec4<f32>, cc: vec4<f32>, fc: vec4<f32>, tlc: vec4<f32>,
    es_u: f32, es_c: f32,
    eb: vec4<f32>,
    e_coat: f32, k_s: f32,
    e_fuzz: vec4<f32>,
    em: vec4<f32>,
}

fn fast_ok(m: Mat) -> bool {
    return m.abbe <= 0.0 && m.filmw <= 0.0;
}

fn lift4(h: vec3<f32>, lam: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(lifth(h, lam.x), lifth(h, lam.y), lifth(h, lam.z), lifth(h, lam.w));
}

fn metal4(mu: f32, r: f32, w: u32) -> f32 {
    return metal_tab(mu, r, w);
}

fn at4_new(m: Mat, lam: vec4<f32>, inside: bool, co0: f32) -> At4 {
    var a: At4;
    let co = max(co0, 1e-9);
    a.inside = inside;
    a.co = co;
    let eta = m.ior;
    let rel = max(eta / m.cior, m.cior / eta);
    let eu = modulated_ior(eta, m.sw);
    let ec = modulated_ior(rel, m.sw);
    if (inside) {
        a.eta_f = 1.0 / eu; a.eta_c = 1.0 / ec; a.etap = 1.0 / eta;
    } else {
        a.eta_f = eu; a.eta_c = ec; a.etap = eta;
    }
    a.eta_raw = eta;
    a.base = lift4(m.base, lam);
    a.metal = lift4(m.metal, lam);
    a.sc = lift4(m.spec, lam);
    a.tc = select(lift4(m.trans, lam), vec4<f32>(1.0), m.tdepth > 0.0);
    a.ssc = lift4(m.ssc, lam);
    a.cc = lift4(m.coat, lam);
    a.fc = lift4(m.fuzz, lam);
    a.tlc = lift4(m.transl, lam);
    a.es_u = diel(co, m.r, a.eta_f);
    a.es_c = a.es_u;
    a.eb = vec4<f32>(0.0);
    a.e_coat = 0.0;
    a.k_s = 0.0;
    if (m.cw > 0.0) {
        a.es_c = diel(co, m.r, a.eta_c);
        let es1 = diel(1.0, m.r, a.eta_c);
        let f0 = m.wb * a.metal;
        let kk = (1.0 - a.sc) * (f0 + (1.0 - f0) * pow(1.0 - MU_BAR, 5.0));
        let e5 = metal4(1.0, m.r, 1u);
        let em1 = xi_metal(m) * (f0 * (metal4(1.0, m.r, 0u) - e5) + e5 - kk * metal4(1.0, m.r, 2u));
        let bd = (1.0 - m.ss) * m.wb * ((1.0 - m.tl) * a.base + m.tl * a.tlc) + m.ss * a.ssc;
        a.eb = m.met * em1 + (1.0 - m.met) * (a.sc * es1 + (1.0 - a.sc * es1) * (1.0 - m.tw) * bd);
        let etac_side = select(m.cior, 1.0 / m.cior, inside);
        a.e_coat = diel(co, m.cr, etac_side);
        a.k_s = fresnel(co, etac_side);
    }
    a.e_fuzz = vec4<f32>(0.0);
    if (m.fw > 0.0) { a.e_fuzz = a.fc * sheen_ltc(co, m.fr).z; }
    a.em = vec4<f32>(0.0);
    if (m.met > 0.0) {
        let f0 = m.wb * a.metal;
        let e5 = metal4(co, m.r, 1u);
        a.em = max(f0 * (metal4(co, m.r, 0u) - e5) + e5, vec4<f32>(0.0));
    }
    return a;
}

fn coat_under4(m: Mat, a: At4, path: f32) -> vec4<f32> {
    let t_coat = pow(max(a.cc, vec4<f32>(1e-12)), vec4<f32>(0.5 * path));
    let k_r = 1.0 - (1.0 - m.ef_coat) / (m.cior * m.cior);
    let fs = ((1.0 - a.eta_raw) / (1.0 + a.eta_raw)) * ((1.0 - a.eta_raw) / (1.0 + a.eta_raw));
    let xs_fs = clamp(m.sw * fs, 0.0, 1.0);
    let rd = 1.0 + (m.r - 1.0) * xs_fs;
    let rb = rd + (m.r - rd) * m.met;
    let kr = a.k_s + (k_r - a.k_s) * rb;
    let delta = (1.0 - kr) / (1.0 - a.eb * kr);
    let dark = 1.0 + (delta - 1.0) * m.cdark;
    return t_coat * (1.0 - a.e_coat) * dark;
}

fn eval4(m: Mat, a: At4, wo: vec3<f32>, wi: vec3<f32>) -> vec4<f32> {
    let co = a.co;
    let thin = (m.flags & F_THIN) != 0u;
    let up = wi.z > 0.0;
    if (!up && (m.tw <= 0.0 || thin) && m.tl <= 0.0) { return vec4<f32>(0.0); }
    var h: vec3<f32>;
    var ci: f32;
    if (up) {
        h = normalize(wi + wo);
        ci = wi.z;
    } else {
        h = normalize(wo + wi * a.etap);
        if (h.z < 0.0) { h = -h; }
        ci = -wi.z;
    }
    let oh = max(abs(dot(wo, h)), 1e-9);
    let ih = abs(dot(wi, h));
    let cI = max(abs(ci), 1e-9);
    let ch = clamp(h.z, 1e-9, 1.0);
    let al = alpha_of(m.r);
    let a2 = al * al;
    let kq = ch * ch * (a2 - 1.0) + 1.0;
    let d = a2 / (PI * kq * kq);
    let g2 = 1.0 / (1.0 + ggx_lambda(al, cI) + ggx_lambda(al, co));
    let dg = d * g2 / (4.0 * cI * co);
    let bd = (1.0 - m.ss) * m.wb * (1.0 - m.tl) * a.base;
    let coated = m.cw > 0.0;
    var f_metal = vec4<f32>(0.0);
    if (up && m.met > 0.0) {
        let f0 = m.wb * a.metal;
        let p5 = pow(1.0 - oh, 5.0);
        let fs_oh = f0 + (1.0 - f0) * p5;
        let fs_bar = f0 + (1.0 - f0) * pow(1.0 - MU_BAR, 5.0);
        let kk = (1.0 - a.sc) * fs_bar;
        let mo = 1.0 - oh;
        let w82 = oh * mo * mo * mo * mo * mo * mo / (MU_BAR * pow(1.0 - MU_BAR, 6.0));
        f_metal = xi_metal(m) * (fs_oh - w82 * kk);
    }
    var ru = vec4<f32>(0.0);
    var rc = vec4<f32>(0.0);
    var tu = vec4<f32>(0.0);
    var tcv = vec4<f32>(0.0);
    let nb = select(1u, 2u, coated);
    for (var k = 0u; k < nb; k = k + 1u) {
        let eta = select(a.eta_f, a.eta_c, k == 1u);
        let es = select(a.es_u, a.es_c, k == 1u);
        let fd = fresnel(oh, eta);
        let kd = (1.0 - m.tw) * (1.0 - a.sc * es);
        if (up) {
            let diffuse = kd * (bd + m.ss * a.ssc) / PI;
            let r = m.met * f_metal * dg + (1.0 - m.met) * (a.sc * fd * dg + diffuse);
            if (k == 0u) { ru = r; } else { rc = r; }
        } else {
            var t = vec4<f32>(0.0);
            if (!thin && m.tw > 0.0) {
                let den = dot(wi, h) + dot(wo, h) / a.etap;
                let denom2 = max(den * den, 1e-12);
                let btdf = d * g2 * ih * oh / (cI * co * denom2 * a.etap * a.etap);
                t = m.tw * a.tc * (1.0 - fd) * btdf;
            }
            t = t + kd * (1.0 - m.ss) * m.wb * m.tl * a.tlc / PI;
            if (k == 0u) { tu = (1.0 - m.met) * t; } else { tcv = (1.0 - m.met) * t; }
        }
    }
    var under = vec4<f32>(0.0);
    if (coated) { under = coat_under4(m, a, 1.0 / coat_mu(m, cI) + 1.0 / coat_mu(m, co)); }
    let fuzzmul = 1.0 - m.fw * a.e_fuzz;
    if (up) {
        var f_coat = 0.0;
        if (coated) {
            let etac_side = select(m.cior, 1.0 / m.cior, a.inside);
            let ac = alpha_of(m.cr);
            f_coat = fresnel(oh, etac_side) * ggx_d(ac, ch)
                / (1.0 + ggx_lambda(ac, cI) + ggx_lambda(ac, co)) / (4.0 * cI * co);
        }
        var f_fuzz = vec4<f32>(0.0);
        if (m.fw > 0.0) { f_fuzz = a.fc * sheen_brdf(m.fr, wo, wi); }
        return m.fw * f_fuzz + fuzzmul * (m.cw * f_coat + (1.0 - m.cw) * ru + m.cw * under * rc);
    }
    return fuzzmul * ((1.0 - m.cw) * tu + m.cw * under * tcv);
}

fn thin4(m: Mat, a: At4) -> vec4<f32> {
    let fd = fresnel(max(a.co, 1e-4), a.eta_f);
    var under = vec4<f32>(0.0);
    if (m.cw > 0.0) { under = coat_under4(m, a, 2.0 / coat_mu(m, a.co)); }
    let base = (1.0 - m.met) * m.tw * a.tc * (1.0 - fd);
    return (1.0 - m.fw * a.e_fuzz) * ((1.0 - m.cw) + m.cw * under) * base;
}

// lobe_weights per lane, normalised per lane, averaged over the lanes.
fn lobes4(m: Mat, a: At4) -> Lobes {
    let es = a.es_u;
    let ec = a.e_coat;
    let ef = a.e_fuzz;
    let above = max(1.0 - m.fw * ef, vec4<f32>(0.0)) * max(1.0 - m.cw * ec, 0.0);
    let die = 1.0 - m.met;
    let kd = (1.0 - m.tw) * max(1.0 - a.sc * es, vec4<f32>(0.0));
    var w0 = above * (m.met * a.em + die * a.sc * es) + 1e-3;
    var w1 = vec4<f32>(m.cw * ec);
    var w2 = above * die * kd * ((1.0 - m.ss) * m.wb * (1.0 - m.tl) * a.base + m.ss * a.ssc) + m.fw * ef;
    let w3 = above * die * m.tw * (1.0 - es) * a.tc;
    var w4 = above * die * kd * (1.0 - m.ss) * m.wb * m.tl * a.tlc;
    if (a.inside) {
        w0 = w0 + w2;
        w1 = vec4<f32>(0.0);
        w2 = vec4<f32>(0.0);
        w4 = vec4<f32>(0.0);
    }
    let s = w0 + w1 + w2 + w3 + w4;
    let q = vec4<f32>(0.25);
    var l: Lobes;
    l.w[0] = dot(w0 / s, q);
    l.w[1] = dot(w1 / s, q);
    l.w[2] = dot(w2 / s, q);
    l.w[3] = dot(w3 / s, q);
    l.w[4] = dot(w4 / s, q);
    return l;
}

// ---- the world ----

fn env_uv(d: vec3<f32>) -> vec2<f32> {
    var phi = atan2(-d.x, -d.y);
    if (phi < 0.0) { phi = phi + 2.0 * PI; }
    return vec2<f32>(phi / (2.0 * PI), acos(clamp(d.z, -1.0, 1.0)) / PI);
}

fn env_dir(uv: vec2<f32>) -> vec3<f32> {
    let phi = 2.0 * PI * uv.x;
    let th = PI * uv.y;
    let s = sin(th);
    return vec3<f32>(-s * sin(phi), -s * cos(phi), cos(th));
}

fn env_texel(d: vec3<f32>) -> u32 {
    let uv = env_uv(d);
    let w = P.env.x;
    let h = P.env.y;
    let i = min(u32(uv.x * f32(w)), w - 1u);
    let j = min(u32(uv.y * f32(h)), h - 1u);
    return j * w + i;
}

fn env_rgb(d: vec3<f32>) -> vec3<f32> {
    if (P.env.z == 0u) { return vec3<f32>(0.0); }
    return envt[env_texel(d)].xyz;
}

// Solid-angle pdf of the world's importance sampling.
fn env_pdf(d: vec3<f32>) -> f32 {
    if (P.env.z == 0u) { return 0.0; }
    let w = P.env.x;
    let t = env_texel(d);
    let j = t / w;
    let i = t % w;
    let c1 = envt[t].w;
    var c0 = 0.0;
    if (i > 0u) { c0 = envt[t - 1u].w; }
    let pr = envt[P.env.w + j].y;
    let s = sqrt(max(1.0 - d.z * d.z, 1e-12));
    return (c1 - c0) * pr * f32(w * P.env.y) / (2.0 * PI * PI * s);
}

fn env_sample(u: vec4<f32>) -> vec4<f32> {
    let w = P.env.x;
    let h = P.env.y;
    // row: first marginal cdf > u.x
    var lo = 0u;
    var hi = h - 1u;
    loop {
        if (lo >= hi) { break; }
        let mid = (lo + hi) / 2u;
        if (envt[P.env.w + mid].x > u.x) { hi = mid; } else { lo = mid + 1u; }
    }
    let j = lo;
    var a = 0u;
    var b = w - 1u;
    loop {
        if (a >= b) { break; }
        let mid = (a + b) / 2u;
        if (envt[j * w + mid].w > u.y) { b = mid; } else { a = mid + 1u; }
    }
    let d = env_dir(vec2<f32>((f32(a) + u.z) / f32(w), (f32(j) + u.w) / f32(h)));
    return vec4<f32>(d, env_pdf(d));
}

fn sun_pdf() -> f32 {
    return 1.0 / P.sun_l.w;
}

// What a ray escaping along `d` sees, and the light strategy's pdf of it.
fn sky(d: vec3<f32>) -> vec4<f32> {
    var rgbh = hats(env_rgb(d));
    var pl = P.picks.y - P.picks.x;
    var p = pl * env_pdf(d);
    if (P.sun_dir.w > 0.0 && dot(d, P.sun_dir.xyz) >= P.sun_dir.w) {
        rgbh = rgbh + P.sun_l.xyz;
        p = p + P.picks.x * sun_pdf();
    }
    // hats in xyz; pdf in w
    return vec4<f32>(rgbh, p);
}

// A portal sky: one header vec4 (w, h as bits) at `m.portal`, then texels.
fn portal_rgb(m: Mat, d: vec3<f32>) -> vec3<f32> {
    let hd = envt[m.portal];
    let w = bitcast<u32>(hd.x);
    let h = bitcast<u32>(hd.y);
    let uv = env_uv(d);
    let i = min(u32(uv.x * f32(w)), w - 1u);
    let j = min(u32(uv.y * f32(h)), h - 1u);
    return envt[m.portal + 1u + j * w + i].xyz;
}

fn spot_mask(m: Mat, x: vec3<f32>) -> f32 {
    if ((m.flags & F_SPOT) == 0u) { return 1.0; }
    let t = dot(normalize(x - m.spot_c), m.spot_a);
    if (t <= m.spot_cos) { return 0.0; }
    return smoothstep(0.0, 1.0, (t - m.spot_cos) / m.spot_smooth);
}

// ---- geometry ----

struct Surf {
    ok: bool,
    t: f32,
    inst: u32,
    tri: u32,
    mat: u32,
    x: vec3<f32>,
    ng: vec3<f32>,
    ns: vec3<f32>,
    uv: vec2<f32>,
    emit_index: u32,
}

fn tri_uv(tri: u32, b: vec2<f32>) -> vec2<f32> {
    let o = tri * 7u;
    let vv = tris[o + 6u];
    let u = tris[o + 3u].w * (1.0 - b.x - b.y) + tris[o + 4u].w * b.x + tris[o + 5u].w * b.y;
    let v = vv.x * (1.0 - b.x - b.y) + vv.y * b.x + vv.z * b.y;
    return vec2<f32>(u, v);
}

fn mat_uv(m: Mat, uv: vec2<f32>) -> vec2<f32> {
    return uv * m.xf.xy + m.xf.zw;
}

// A candidate's material, the little a traversal needs of it: flags,
// textures, the UV transform and the constant opacity.
struct Cand { flags: u32, alpha_tex: u32, opacity: f32, xf: vec4<f32> }

fn load_cand(i: u32) -> Cand {
    let o = i * MS;
    var c: Cand;
    c.flags = bitcast<u32>(mats[o + 10u].w);
    let t = mats[o + 12u];
    c.alpha_tex = select(bitcast<u32>(t.x), bitcast<u32>(t.z), bitcast<u32>(t.z) != NONE);
    c.opacity = t.w;
    c.xf = mats[o + 13u];
    return c;
}

fn cand_alpha(c: Cand, uv: vec2<f32>) -> f32 {
    if (c.alpha_tex == NONE) { return 1.0; }
    return tex(c.alpha_tex, uv * c.xf.xy + c.xf.zw).w;
}

fn alpha_at(m: Mat, uv: vec2<f32>) -> f32 {
    var t = m.alpha_tex;
    if (t == NONE) { t = m.base_tex; }
    if (t == NONE) { return 1.0; }
    return tex(t, mat_uv(m, uv)).w;
}

// The closest hit along a ray under `msk`, cut-outs alpha-tested.
fn trace_ray(o: vec3<f32>, d: vec3<f32>, msk: u32) -> Surf {
    var rq: ray_query;
    rayQueryInitialize(&rq, accel, RayDesc(0u, msk, 0.0, 3.4e38, o, d));
    loop {
        if (!rayQueryProceed(&rq)) { break; }
        let c = rayQueryGetCandidateIntersection(&rq);
        if (c.kind != RAY_QUERY_INTERSECTION_TRIANGLE) { continue; }
        st_cand = st_cand + 1u;
        let ir = inst[c.instance_custom_data];
        let m = load_cand(ir.y);
        if ((m.flags & F_CUTOUT) != 0u && !prof_on(PF_OPAQUE)) {
            let uv = tri_uv(ir.x + c.primitive_index, c.barycentrics);
            if (cand_alpha(m, uv) < 0.5) { continue; }
        }
        rayQueryConfirmIntersection(&rq);
    }
    let h = rayQueryGetCommittedIntersection(&rq);
    var s: Surf;
    s.ok = h.kind == RAY_QUERY_INTERSECTION_TRIANGLE;
    if (!s.ok) { return s; }
    let ir = inst[h.instance_custom_data];
    s.t = h.t;
    s.inst = h.instance_custom_data;
    s.tri = ir.x + h.primitive_index;
    s.mat = ir.y;
    s.emit_index = select(NONE, ir.z + h.primitive_index, ir.z != NONE);
    s.x = o + d * h.t;
    let q = s.tri * 7u;
    let ow = h.object_to_world;
    let p0 = ow * vec4<f32>(tris[q].xyz, 1.0);
    let p1 = ow * vec4<f32>(tris[q + 1u].xyz, 1.0);
    let p2 = ow * vec4<f32>(tris[q + 2u].xyz, 1.0);
    s.ng = normalize(cross(p1 - p0, p2 - p0));
    let b = h.barycentrics;
    let n0 = tris[q + 3u].xyz;
    let n1 = tris[q + 4u].xyz;
    let n2 = tris[q + 5u].xyz;
    let no = n0 * (1.0 - b.x - b.y) + n1 * b.x + n2 * b.y;
    let wo2 = h.world_to_object;
    let l = mat3x3<f32>(wo2[0], wo2[1], wo2[2]);
    var ns = transpose(l) * no;
    if (dot(ns, ns) < 1e-20) { ns = s.ng; }
    s.ns = normalize(ns);
    s.uv = tri_uv(s.tri, b);
    return s;
}

// Transmittance per wavelength along a shadow ray (geometry only): cut-outs
// alpha-tested, thin glass and constant opacity attenuate.
fn shadow(o: vec3<f32>, d: vec3<f32>, tmax: f32, lam: vec4<f32>) -> vec4<f32> {
    var rq: ray_query;
    rayQueryInitialize(&rq, accel, RayDesc(RAY_FLAG_TERMINATE_ON_FIRST_HIT, M_GEOMETRY, 0.0, tmax, o, d));
    var tr = vec4<f32>(1.0);
    loop {
        if (!rayQueryProceed(&rq)) { break; }
        let c = rayQueryGetCandidateIntersection(&rq);
        if (c.kind != RAY_QUERY_INTERSECTION_TRIANGLE) { continue; }
        st_scand = st_scand + 1u;
        let ir = inst[c.instance_custom_data];
        let mc = load_cand(ir.y);
        if ((mc.flags & F_CUTOUT) != 0u && !prof_on(PF_OPAQUE)) {
            let uv = tri_uv(ir.x + c.primitive_index, c.barycentrics);
            if (cand_alpha(mc, uv) < 0.5) { continue; }
            rayQueryConfirmIntersection(&rq);
            break;
        }
        if ((mc.flags & F_THIN) != 0u) {
            let m = load_mat(ir.y);
            let q = (ir.x + c.primitive_index) * 7u;
            let ng = normalize(cross(tris[q + 1u].xyz - tris[q].xyz, tris[q + 2u].xyz - tris[q].xyz));
            let ow = c.object_to_world;
            let ngw = normalize((ow * vec4<f32>(ng, 0.0)).xyz);
            let cs = abs(dot(ngw, d));
            if (fast_ok(m)) {
                tr = tr * thin4(m, at4_new(m, lam, false, cs));
            } else {
                for (var k = 0u; k < NW; k = k + 1u) {
                    tr[k] = tr[k] * thin_pass(m, at_new(m, lam[k], false, cs));
                }
            }
            if (max(max(tr.x, tr.y), max(tr.z, tr.w)) <= 0.0) {
                rayQueryConfirmIntersection(&rq);
                break;
            }
            continue;
        }
        if (mc.opacity < 1.0) {
            tr = tr * (1.0 - mc.opacity);
            continue;
        }
        rayQueryConfirmIntersection(&rq);
        break;
    }
    let h = rayQueryGetCommittedIntersection(&rq);
    if (h.kind != RAY_QUERY_INTERSECTION_NONE) { return vec4<f32>(0.0); }
    return tr;
}

// ---- lights ----

struct LS {
    ok: bool,
    wi: vec3<f32>,
    dist: f32,
    pdf: f32,           // solid angle, with the group pick
    le: vec3<f32>,      // radiance on the hats
}

fn emitter_tri(i: u32) -> array<vec4<f32>, 3> {
    return array<vec4<f32>, 3>(emit[3u * i], emit[3u * i + 1u], emit[3u * i + 2u]);
}

fn emitter_prob(i: u32, first: u32) -> f32 {
    let c1 = emit[3u * i + 2u].w;
    var c0 = 0.0;
    if (i > first) { c0 = emit[3u * (i - 1u) + 2u].w; }
    return c1 - c0;
}

// Which side of an emitter triangle emits toward a viewer along -d.
fn emits_toward(m: Mat, ng: vec3<f32>, d: vec3<f32>) -> bool {
    if ((m.flags & F_EMIT_BOTH) != 0u) { return true; }
    let front = dot(ng, d) < 0.0;
    return select(front, !front, (m.flags & F_EMIT_BACK) != 0u);
}

fn emitter_le(m: Mat, xl: vec3<f32>, ng: vec3<f32>, d: vec3<f32>, x: vec3<f32>) -> vec3<f32> {
    if (!emits_toward(m, ng, d)) { return vec3<f32>(0.0); }
    if ((m.flags & F_PORTAL) != 0u) {
        let r = d - 2.0 * dot(d, ng) * ng;
        return hats(portal_rgb(m, r));
    }
    return m.emis * spot_mask(m, x);
}

fn pick_cdf(first: u32, n: u32, u: f32) -> u32 {
    var lo = first;
    var hi = first + n - 1u;
    loop {
        if (lo >= hi) { break; }
        let mid = (lo + hi) / 2u;
        if (emit[3u * mid + 2u].w > u) { hi = mid; } else { lo = mid + 1u; }
    }
    return lo;
}

fn sample_light(x: vec3<f32>, u: vec4<f32>, u2: f32) -> LS {
    var s: LS;
    s.ok = false;
    let pk = P.picks;
    if (u.x < pk.x) {
        // the sun: a uniform cone
        let cr = P.sun_dir.w;
        let z = 1.0 - u.y * (1.0 - cr);
        let r = sqrt(max(1.0 - z * z, 0.0));
        let ph = 2.0 * PI * u.z;
        let n = P.sun_dir.xyz;
        let wi = basis_t(n) * (r * cos(ph)) + basis_b(n) * (r * sin(ph)) + n * z;
        s.ok = true;
        s.wi = wi;
        s.dist = 3.4e38;
        // the world is sampled by the other strategy too: count both
        s.pdf = pk.x * sun_pdf() + (pk.y - pk.x) * env_pdf(wi);
        s.le = P.sun_l.xyz + hats(env_rgb(wi));
        return s;
    }
    if (u.x < pk.y) {
        let e = env_sample(vec4<f32>(u.y, u.z, u.w, u2));
        s.ok = e.w > 0.0;
        s.wi = e.xyz;
        s.dist = 3.4e38;
        var p = (pk.y - pk.x) * e.w;
        var le = hats(env_rgb(e.xyz));
        if (P.sun_dir.w > 0.0 && dot(e.xyz, P.sun_dir.xyz) >= P.sun_dir.w) {
            p = p + pk.x * sun_pdf();
            le = le + P.sun_l.xyz;
        }
        s.pdf = p;
        s.le = le;
        return s;
    }
    var first = 0u;
    var n = P.flags.y;
    var g = pk.z - pk.y;
    var uu = (u.x - pk.y) / max(g, 1e-12);
    if (u.x >= pk.z) {
        first = P.emit_off.y;
        n = P.flags.z;
        g = pk.w - pk.z;
        uu = (u.x - pk.z) / max(g, 1e-12);
    }
    if (n == 0u) { return s; }
    let i = pick_cdf(first, n, clamp(uu, 0.0, 0.99999994));
    let t = emitter_tri(i);
    let su = sqrt(u.y);
    let b0 = 1.0 - su;
    let b1 = u.z * su;
    let xl = t[0].xyz * b0 + t[1].xyz * b1 + t[2].xyz * (1.0 - b0 - b1);
    let ng = normalize(cross(t[1].xyz - t[0].xyz, t[2].xyz - t[0].xyz));
    let dv = xl - x;
    let d2 = dot(dv, dv);
    let dist = sqrt(d2);
    let wi = dv / dist;
    let cl = abs(dot(ng, wi));
    if (cl < 1e-6) { return s; }
    let m = load_mat(bitcast<u32>(t[0].w));
    s.le = emitter_le(m, xl, ng, wi, x);
    s.ok = max(max(s.le.x, s.le.y), s.le.z) > 0.0;
    s.wi = wi;
    s.dist = dist;
    s.pdf = g * emitter_prob(i, first) / t[1].w * d2 / cl;
    return s;
}

// The light strategy's pdf of hitting emitter triangle `i` at distance
// `dist` with cosine `cl` (solid angle, from the previous vertex).
fn emitter_pdf(i: u32, dist: f32, cl: f32) -> f32 {
    let portal = i >= P.emit_off.y && P.flags.z > 0u;
    let first = select(0u, P.emit_off.y, portal);
    let g = select(P.picks.z - P.picks.y, P.picks.w - P.picks.z, portal);
    let t1 = emit[3u * i + 1u];
    return g * emitter_prob(i, first) / t1.w * dist * dist / max(cl, 1e-6);
}

// ---- the integrator ----

var<workgroup> wg_val: array<f32, 256>;
var<workgroup> wg_bin: array<u32, 256>;

fn local_frame(ns: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(v, basis_t(ns)), dot(v, basis_b(ns)), dot(v, ns));
}

fn world_of(ns: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    return basis_t(ns) * v.x + basis_b(ns) * v.y + ns * v.z;
}

@compute @workgroup_size(64)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32,
         @builtin(workgroup_id) wid: vec3<u32>) {
    let px = P.seed.w + gid.x;
    var rad = vec4<f32>(0.0);
    var lam = vec4<f32>(0.0);
    let live = px < P.flags.x;
    let pstats = prof_on(PF_STATS);
    if (pstats && li < 16u) {
        atomicStore(&wg_alive[li], 0u);
        atomicStore(&wg_cls[li], 0u);
    }
    workgroupBarrier();
    var iter = 0u;
    if (live) {
        var rng = rng_new(vec2<u32>(px, P.seed.z));
        q_pixel = qhash2(px, P.seed.x ^ qhash(P.seed.y));
        q_tuple = 0u;
        let u_cam = draw4(&rng);
        for (var k = 0u; k < NW; k = k + 1u) {
            let v = u_cam.x + f32(k) / f32(NW);
            if ((P.samp.x & S_SPECTRAL) != 0u) {
                lam[k] = spec_lambda(v - floor(v));
            } else {
                lam[k] = 380.0 + 400.0 * (v - floor(v));
            }
        }
        let w = P.size.x;
        let fw = f32(P.size.x);
        let fh = f32(P.size.y);
        let xs = f32(px % w) + u_cam.y;
        let ys = f32(px / w) + u_cam.z;
        let sx = (2.0 * xs / fw - 1.0) * P.eye.w * fw / fh;
        let sy = (1.0 - 2.0 * ys / fh) * P.eye.w;
        var ro = P.eye.xyz;
        var rd = normalize(P.fwd.xyz + P.right.xyz * sx + P.up.xyz * sy);
        var tp = vec4<f32>(1.0);
        // Spectral MIS over whole paths: q_j = P_j / P_hero, the lanes'
        // path densities relative to the hero's; a contribution is divided
        // by their mean (balance heuristic over the four wavelengths).
        var q = vec4<f32>(1.0);
        var prev_x = ro;
        var prev_pdf = 0.0;      // 0: camera ray, no rival strategy
        var msk = M_GEOMETRY | M_LIGHT;
        var bounce = 0u;
        var passes = 0u;
        let clampv = P.fwd.w;
        loop {
            let s = trace_ray(ro, rd, msk);
            let it = min(iter, 7u);
            if (pstats) {
                atomicAdd(&stats[select(1u, 0u, iter == 0u)], 1u);
                atomicAdd(&stats[8u + it], 1u);
                atomicAdd(&wg_alive[(li / 32u) * 8u + it], 1u);
            }
            iter = iter + 1u;
            if (!s.ok) {
                let e = sky(rd);
                var wm = 1.0;
                if (prev_pdf > 0.0) { wm = mis(prev_pdf, e.w); }
                for (var k = 0u; k < NW; k = k + 1u) {
                    var c = tp[k] * lifth(e.xyz, lam[k]) * wm / qbar(q);
                    if (clampv > 0.0 && bounce > 0u) { c = min(c, clampv); }
                    rad[k] = rad[k] + c;
                }
                break;
            }
            var m = load_mat(s.mat);
            if (prof_on(PF_LAMBERT) && (m.flags & F_EMITTER) == 0u) {
                m.met = 0.0; m.sw = 0.0; m.tw = 0.0; m.cw = 0.0; m.fw = 0.0; m.ss = 0.0;
                m.tl = 0.0; m.filmw = 0.0; m.abbe = 0.0; m.metal_tex = NONE;
                m.flags = m.flags & ~(F_THIN | F_METAL_SEP);
            }
            if (P.size.z != 0u) {
                // debug views: 1 material hash, 2 shading normal, 3 transmission weight
                var c = vec3<f32>(0.0);
                if (P.size.z == 1u) {
                    let hsh = s.mat * 2654435761u;
                    c = vec3<f32>(f32(hsh & 255u), f32((hsh >> 8u) & 255u), f32((hsh >> 16u) & 255u)) / 255.0;
                } else if (P.size.z == 2u) {
                    c = s.ns * 0.5 + 0.5;
                } else {
                    c = vec3<f32>(m.tw, f32(dot(s.ng, rd) < 0.0), s.t * 0.05);
                }
                accum[px] = accum[px] + vec4<f32>(c, 0.0);
                break;
            }
            // Emitters: collect what faces the ray, then stop (or, for a
            // portal's transparent side, pass).
            if ((m.flags & F_EMITTER) != 0u) {
                let le = emitter_le(m, s.x, s.ng, rd, prev_x);
                if (max(max(le.x, le.y), le.z) > 0.0) {
                    var wm = 1.0;
                    if (prev_pdf > 0.0 && s.emit_index != NONE) {
                        let dv = s.x - prev_x;
                        wm = mis(prev_pdf, emitter_pdf(s.emit_index, length(dv), abs(dot(s.ng, rd))));
                    }
                    for (var k = 0u; k < NW; k = k + 1u) {
                        var c = tp[k] * lifth(le, lam[k]) * wm / qbar(q);
                        if (clampv > 0.0 && bounce > 0u) { c = min(c, clampv); }
                        rad[k] = rad[k] + c;
                    }
                    break;
                }
                if ((m.flags & F_PORTAL) != 0u && passes < 32u) {
                    passes = passes + 1u;
                    ro = offset_ray(s.x, select(s.ng, -s.ng, dot(s.ng, rd) < 0.0));
                    continue;
                }
                break;
            }
            if (bounce >= P.size.w) { break; }
            // Constant opacity: a stochastic pass (not a vertex).
            if (m.opacity < 1.0 && (m.flags & F_CUTOUT) == 0u) {
                if (rand(&rng) >= m.opacity) {
                    if (passes >= 32u) { break; }
                    passes = passes + 1u;
                    ro = offset_ray(s.x, select(s.ng, -s.ng, dot(s.ng, rd) < 0.0));
                    continue;
                }
            }
            let wo_w = -rd;
            let thin = (m.flags & F_THIN) != 0u;
            let transmissive = m.tw > 0.0 && !thin;
            var ng = s.ng;
            var ns = s.ns;
            var inside = false;
            if (dot(ng, wo_w) < 0.0) {
                ng = -ng;
                ns = -ns;
                inside = transmissive;
            }
            // A shading normal at a grazing or back-facing angle to the
            // viewer: bend it toward the geometric normal until the view is
            // at least τ above it (Cycles' ensure_valid_specular_reflection
            // has the same aim), or a GGX lobe at μ_o → 0 makes fireflies.
            let tau = min(0.1, 0.9 * dot(ng, wo_w));
            if (dot(ns, wo_w) < tau) {
                var lo = 0.0;
                var hi = 1.0;
                for (var it = 0u; it < 12u; it = it + 1u) {
                    let t = 0.5 * (lo + hi);
                    if (dot(normalize(mix(ns, ng, t)), wo_w) < tau) { lo = t; } else { hi = t; }
                }
                ns = normalize(mix(ns, ng, hi));
            }
            var mt = m;
            if (m.base_tex != NONE) {
                mt.base = hats(tex(m.base_tex, mat_uv(m, s.uv)).xyz);
            }
            if (m.metal_tex != NONE) {
                mt.metal = hats(tex(m.metal_tex, mat_uv(m, s.uv)).xyz);
            }
            let wo = local_frame(ns, wo_w);
            // The four-lane path where the index is the same at every
            // wavelength; per lane otherwise (and when P.up.w forces it).
            let fast = fast_ok(mt) && P.up.w == 0.0;
            if (pstats) {
                let cls = select(0u, 1u, mt.tw > 0.0) | select(0u, 2u, mt.met > 0.0) | select(0u, 4u, mt.cw > 0.0)
                    | select(0u, 8u, mt.fw > 0.0 || mt.tl > 0.0 || thin) | select(0u, 16u, !fast);
                atomicAdd(&stats[32u + cls], 1u);
                atomicAdd(&stats[7u], 1u);
                if (!fast) { atomicAdd(&stats[6u], 1u); }
                atomicOr(&wg_cls[(li / 32u) * 8u + it], 1u << cls);
            }
            var at: array<At, 4>;
            var a4: At4;
            var lw: Lobes;
            var etap0 = 1.0;
            if (fast) {
                a4 = at4_new(mt, lam, inside, max(wo.z, 1e-6));
                lw = lobes4(mt, a4);
                etap0 = a4.etap;
            } else {
                for (var k = 0u; k < NW; k = k + 1u) { at[k] = at_new(mt, lam[k], inside, max(wo.z, 1e-6)); }
                // Lobe weights averaged over the four wavelengths: the hero's
                // alone would give a lane whose albedo the hero does not share
                // (a green leaf at a blue hero) f / pdf without bound.
                lw = lobe_weights(mt, at[0]);
                for (var k = 1u; k < NW; k = k + 1u) {
                    let lk = lobe_weights(mt, at[k]);
                    lw.w[0] = lw.w[0] + lk.w[0];
                    lw.w[1] = lw.w[1] + lk.w[1];
                    lw.w[2] = lw.w[2] + lk.w[2];
                    lw.w[3] = lw.w[3] + lk.w[3];
                    lw.w[4] = lw.w[4] + lk.w[4];
                }
                lw.w[0] = lw.w[0] * 0.25;
                lw.w[1] = lw.w[1] * 0.25;
                lw.w[2] = lw.w[2] * 0.25;
                lw.w[3] = lw.w[3] * 0.25;
                lw.w[4] = lw.w[4] * 0.25;
                etap0 = at[0].etap;
            }
            let xo_up = offset_ray(s.x, ng);
            let xo_dn = offset_ray(s.x, -ng);
            // Next-event estimation.
            let ul = draw4(&rng);
            let ub4 = draw4(&rng);
            let ul2 = ub4.x;
            if (!prof_on(PF_NO_NEE)) {
                let l = sample_light(s.x, ul, ul2);
                if (l.ok && l.pdf > 0.0) {
                    let above = dot(l.wi, ng) > 0.0;
                    let xo = select(xo_dn, xo_up, above);
                    let wi = local_frame(ns, l.wi);
                    if ((above && wi.z > 0.0) || (!above && wi.z < 0.0)) {
                        var dist = l.dist;
                        if (dist < 3e38) { dist = length(s.x + l.wi * l.dist - xo) * (1.0 - 1e-4); }
                        var tr = vec4<f32>(1.0);
                        if (!prof_on(PF_NO_SHADOW)) {
                            tr = shadow(xo, l.wi, dist, lam);
                            if (pstats) { atomicAdd(&stats[2u], 1u); }
                        }
                        if (max(max(tr.x, tr.y), max(tr.z, tr.w)) > 0.0) {
                            var pb: f32;
                            var f4: vec4<f32>;
                            if (fast) {
                                pb = lobe_pdf_e(mt, a4.etap, lw, wo, wi);
                                f4 = eval4(mt, a4, wo, wi);
                            } else {
                                pb = spectral_pdf(mt, &at, lw, wo, wi, q);
                                for (var k = 0u; k < NW; k = k + 1u) { f4[k] = eval_dir(mt, at[k], wo, wi); }
                            }
                            let wm = mis(l.pdf, pb);
                            for (var k = 0u; k < NW; k = k + 1u) {
                                let fk = f4[k];
                                var c = tp[k] * fk * abs(wi.z) * lifth(l.le, lam[k]) * tr[k] * wm / (l.pdf * qbar(q));
                                if (clampv > 0.0 && bounce > 0u) { c = min(c, clampv); }
                                rad[k] = rad[k] + c;
                            }
                        }
                    }
                }
            }
            // The BSDF sample.
            let ub = ub4.yzw;
            let pick = lobe_sample(mt, etap0, lw, wo, ub);
            if (pick.lobe == 3u && thin) {
                // The thin pass: straight on, not a vertex (shadow rays
                // took the same pass at the previous vertex).
                if (fast) {
                    tp = tp * thin4(mt, a4) / lw.w[3];
                } else {
                    for (var k = 0u; k < NW; k = k + 1u) {
                        tp[k] = tp[k] * thin_pass(mt, at[k]) / lw.w[3];
                    }
                }
                if (passes >= 32u) { break; }
                passes = passes + 1u;
                ro = xo_dn;
                continue;
            }
            let wi = pick.wi;
            if (dot(wi, wi) < 0.5 || wi.z == 0.0) { break; }
            let wi_w = normalize(world_of(ns, wi));
            let above = dot(wi_w, ng) > 0.0;
            if (above != (wi.z > 0.0)) { break; }
            // Spectral MIS (Wilkie+ 2014): every lane over the mean of the
            // four wavelengths' densities, so a dispersive near-delta
            // refraction carries its hero ×4 and the rest ~0, unbiased.
            var pdf: f32;
            if (fast) {
                // every lane's density is the hero's: q is unchanged
                pdf = lobe_pdf_e(mt, a4.etap, lw, wo, wi);
                if (!(pdf > 0.0) || pdf > 3.0e38) { break; }
                tp = tp * eval4(mt, a4, wo, wi) * (abs(wi.z) / pdf);
            } else {
                pdf = spectral_pdf(mt, &at, lw, wo, wi, q);
                let p0 = lobe_pdf(mt, at[0], lw, wo, wi);
                if (!(pdf > 0.0) || pdf > 3.0e38 || !(p0 > 0.0)) { break; }
                for (var k = 0u; k < NW; k = k + 1u) {
                    tp[k] = tp[k] * eval_dir(mt, at[k], wo, wi) * abs(wi.z) / p0;
                    if (k > 0u) { q[k] = q[k] * lobe_pdf(mt, at[k], lw, wo, wi) / p0; }
                }
            }
            bounce = bounce + 1u;
            if (bounce >= P.flags.w) {
                let q = clamp(max(max(tp.x, tp.y), max(tp.z, tp.w)), 0.05, 1.0);
                if (rand(&rng) >= q) { break; }
                tp = tp / q;
            }
            if (max(max(tp.x, tp.y), max(tp.z, tp.w)) <= 0.0) { break; }
            prev_x = s.x;
            prev_pdf = pdf;
            msk = M_GEOMETRY | M_LIGHT | M_LIGHT_HIDDEN | M_PORTAL;
            ro = select(xo_dn, xo_up, above);
            rd = wi_w;
        }
        var rgb = vec3<f32>(0.0);
        // A guard, not a clamp: a path whose radiance is beyond anything
        // the scene can emit (1e9; the brightest pixel of a converged
        // pavilion is ~170) came from a vanishing pdf in f32 and is dropped
        // (one sample in ~10^10 on the pavilion at 1080p).
        for (var k = 0u; k < NW; k = k + 1u) {
            if (!(rad[k] == rad[k]) || abs(rad[k]) > 1.0e9) { rad = vec4<f32>(0.0); }
            rad[k] = rad[k] / spec_pdf(lam[k]);
            rgb = rgb + rad[k] * lane_rgb(lam[k]);
        }
        accum[px] = accum[px] + vec4<f32>(rgb, 0.0);
    }
    if (pstats) {
        atomicAdd(&stats[3u], st_cand);
        atomicAdd(&stats[4u], st_scand);
        workgroupBarrier();
        if (li < 16u) {
            let n = atomicLoad(&wg_alive[li]);
            if (n > 0u) {
                atomicAdd(&stats[16u + (li % 8u)], 1u);
                atomicAdd(&stats[24u + (li % 8u)], countOneBits(atomicLoad(&wg_cls[li])));
            }
        }
    }
    // The frame's mean spectrum: each workgroup sums its lanes into BINS
    // slots of its own, in lane order (no float atomics: deterministic).
    for (var k = 0u; k < NW; k = k + 1u) {
        wg_val[li * NW + k] = select(0.0, rad[k], live);
        wg_bin[li * NW + k] = min(u32((lam[k] - 380.0) / 400.0 * f32(BINS)), BINS - 1u);
    }
    workgroupBarrier();
    if (li == 0u) {
        let g = (P.seed.w / 64u) + wid.x;
        for (var i = 0u; i < 256u; i = i + 1u) {
            let b = wg_bin[i];
            spec[g * BINS + b] = spec[g * BINS + b] + wg_val[i];
        }
    }
}

// Lanes for the f64 twin: per lane 4 vec4 in (material, λ, inside, -),
// wo, wi, u; out (f, pdf, lobe weights 0..1), (lobe weights 2..4, thin pass),
// (sample wi, lobe).
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = bitcast<u32>(probe_io[0].x);
    if (i >= n) { return; }
    let base = 1u + i * 7u;
    let hd = probe_io[base];
    let m = load_mat(bitcast<u32>(hd.x));
    let wo = probe_io[base + 1u].xyz;
    let a = at_new(m, hd.y, hd.z > 0.5, wo.z);
    let wi = probe_io[base + 2u].xyz;
    let u = probe_io[base + 3u].xyz;
    let lw = lobe_weights(m, a);
    let f = eval_dir(m, a, wo, wi);
    let p = lobe_pdf(m, a, lw, wo, wi);
    probe_io[base + 4u] = vec4<f32>(f, p, lw.w[0], lw.w[1]);
    probe_io[base + 5u] = vec4<f32>(lw.w[2], lw.w[3], lw.w[4], thin_pass(m, a));
    let pk = lobe_sample(m, a.etap, lw, wo, u);
    probe_io[base + 6u] = vec4<f32>(pk.wi, f32(pk.lobe));
}
