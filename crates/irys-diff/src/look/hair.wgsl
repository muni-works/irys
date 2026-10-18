// `look::hair::HairScene::trace` on the device, forward only: one thread per
// path, the CPU's `prb::Rng` stream draw for draw, the fibres through
// `strands.wgsl`'s ray query, the scalp and the quads analytic, the BCSDF
// from `bcsdf.wgsl`. Writes (L, λ) per path; the host bins the image with
// `HairScene::image`, so a path is comparable lane for lane with the CPU.

struct Light {
    c: vec4<f32>,   // centre, radiance
    u: vec4<f32>,   // half-edge, tint r
    v: vec4<f32>,   // half-edge, tint g
    n: vec4<f32>,   // unit normal, tint b
}

struct HP {
    size: vec4<u32>,     // w, h, spp, depth
    seed: vec4<u32>,     // lo, hi, first path, lights
    eye: vec4<f32>,      // xyz, tan(fov/2)
    fwd: vec4<f32>,      // xyz, aspect
    right: vec4<f32>,
    up: vec4<f32>,
    scalp: vec4<f32>,    // centre, radius (≤ 0: none)
    rho: vec4<f32>,      // scalp albedo on the hats (450, 550, 650)
    fiber: vec4<f32>,    // eta, beta_m, beta_n, alpha°
    theta: vec4<f32>,    // θ, θ_s
    env: vec4<f32>,      // env, paths in this dispatch
    lights: array<Light, 4>,
}

@group(0) @binding(0) var<uniform> P: HP;
@group(0) @binding(1) var<storage, read_write> lanes: array<vec2<f32>>;

const PI: f32 = 3.14159265358979;

// ---- prb::Rng ----

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
    let lo = (p00 & 0xffffu) | (mid << 16u);
    let hi = p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
    return vec2<u32>(lo, hi);
}

fn mul64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let p = mul32(a.x, b.x);
    return vec2<u32>(p.x, p.y + a.x * b.y + a.y * b.x);
}

fn shr64(a: vec2<u32>, s: u32) -> vec2<u32> {
    return vec2<u32>((a.x >> s) | (a.y << (32u - s)), a.y >> s);
}

const GOLDEN: vec2<u32> = vec2<u32>(0x7f4a7c15u, 0x9e3779b9u);
const M1: vec2<u32> = vec2<u32>(0x1ce4e5b9u, 0xbf58476du);
const M2: vec2<u32> = vec2<u32>(0x133111ebu, 0x94d049bbu);

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

// ---- helpers ----

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

// render::lift: rgb (r, g, b) on the 450/550/650 hats.
fn lift(rgb: vec3<f32>, l: f32) -> f32 {
    let w = bands(l);
    return w.x * rgb.z + w.y * rgb.y + w.z * rgb.x;
}

fn mis(a: f32, b: f32) -> f32 {
    let s = a * a + b * b;
    return select(0.0, a * a / s, s > 0.0);
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

fn light_le(i: u32, lam: f32) -> f32 {
    let l = P.lights[i];
    return l.c.w * lift(vec3<f32>(l.u.w, l.v.w, l.n.w), lam);
}

fn light_area(i: u32) -> f32 {
    let l = P.lights[i];
    return 4.0 * length(cross(l.u.xyz, l.v.xyz));
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

struct LS {
    ok: bool,
    d: vec3<f32>,
    dist: f32,
    pdf: f32,
    i: u32,
}

// HairScene::sample_light: uniform over the lights, uniform on the quad.
fn sample_light(x: vec3<f32>, u: vec3<f32>) -> LS {
    let n = P.seed.w;
    let i = min(u32(u.x * f32(n)), n - 1u);
    let l = P.lights[i];
    let p = l.c.xyz + l.u.xyz * (2.0 * u.y - 1.0) + l.v.xyz * (2.0 * u.z - 1.0);
    let dv = p - x;
    let d2 = dot(dv, dv);
    let d = dv / sqrt(d2);
    let cl = -dot(d, l.n.xyz);
    return LS(cl > 0.0, d, sqrt(d2), d2 / (cl * light_area(i) * f32(n)), i);
}

fn scalp_t(o: vec3<f32>, d: vec3<f32>) -> f32 {
    if (P.scalp.w <= 0.0) { return -1.0; }
    let oc = o - P.scalp.xyz;
    let b = dot(oc, d);
    let q = dot(oc, oc) - P.scalp.w * P.scalp.w;
    let disc = b * b - q;
    if (disc < 0.0) { return -1.0; }
    let s = sqrt(disc);
    // The CPU's 1e-7 floor is below f32's noise for a ray leaving the
    // sphere it starts on (|oc|² − r² rounds at ~1e-7): 1e-4 here.
    if (-b - s > 1e-4) { return -b - s; }
    if (-b + s > 1e-4) { return -b + s; }
    return -1.0;
}

fn visible(o: vec3<f32>, d: vec3<f32>, dist: f32) -> bool {
    if (strand_hit(o, d, 1e-7, dist, true).seg != 0xffffffffu) { return false; }
    let t = scalp_t(o, d);
    return !(t > 0.0) || t >= dist;
}

@compute @workgroup_size(64)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let k = gid.x + gid.y * nwg.x * 64u;
    if (k >= u32(P.env.y)) { return; }
    let path = P.seed.z + k;
    var rng = rng_new(path);
    let spp = P.size.z;
    let lam = 380.0 + 400.0 * ((f32(path % spp) + rand(&rng)) / f32(spp));
    let w = P.size.x;
    let h = P.size.y;
    let px = path / spp;
    let ix = f32(px % w) + rand(&rng);
    let iy = f32(px / w) + rand(&rng);
    let th = P.eye.w;
    var d = normalize(P.fwd.xyz + P.right.xyz * ((2.0 * ix / f32(w) - 1.0) * th * P.fwd.w)
        + P.up.xyz * ((1.0 - 2.0 * iy / f32(h)) * th));
    var o = P.eye.xyz;
    var beta = 1.0;
    var l = 0.0;
    var last_pdf = -1.0;
    let theta = P.theta.xy;
    let theta_s = P.theta.zw;
    for (var depth = 0u; depth < P.size.w; depth = depth + 1u) {
        // closest: fibre, scalp, light.
        let sh = strand_hit(o, d, 1e-7, 3.4e38, false);
        var tb = 3.4e38;
        var kind = 0u;   // 0 none, 1 hair, 2 scalp, 3 light
        if (sh.seg != 0xffffffffu) {
            tb = sh.t;
            kind = 1u;
        }
        let ts = scalp_t(o, d);
        if (ts > 0.0 && ts < tb) {
            tb = ts;
            kind = 2u;
        }
        var li = 0u;
        let lim = tb;
        for (var i = 0u; i < P.seed.w; i = i + 1u) {
            let t = light_hit_t(i, o, d, lim);
            if (t > 0.0 && t < tb) {
                tb = t;
                kind = 3u;
                li = i;
            }
        }
        if (kind == 0u) {
            l = l + beta * P.env.x;
            break;
        }
        if (kind == 3u) {
            let pl = tb * tb / (-dot(d, P.lights[li].n.xyz) * light_area(li) * f32(P.seed.w));
            var wm = 1.0;
            if (last_pdf >= 0.0) { wm = mis(last_pdf, pl); }
            l = l + beta * light_le(li, lam) * wm;
            break;
        }
        if (kind == 2u) {
            let x = o + d * tb;
            let n = normalize(x - P.scalp.xyz);
            let wb = bands(lam);
            let rho = dot(wb, P.rho.xyz);
            let un = vec3<f32>(rand(&rng), rand(&rng), rand(&rng));
            let ls = sample_light(x, un);
            if (ls.ok) {
                let c = dot(ls.d, n);
                if (c > 0.0 && visible(x + n * 1e-6, ls.d, ls.dist)) {
                    l = l + beta * rho / PI * c * light_le(ls.i, lam) / ls.pdf * mis(ls.pdf, c / PI);
                }
            }
            let u1 = rand(&rng);
            let u2 = rand(&rng);
            let r = sqrt(u1);
            let ph = 2.0 * PI * u2;
            let cz = sqrt(max(1.0 - u1, 0.0));
            d = basis_t(n) * (r * cos(ph)) + basis_b(n) * (r * sin(ph)) + n * cz;
            o = x + n * 1e-6;
            beta = beta * rho;
            last_pdf = cz / PI;
        } else {
            let x = o + d * tb;
            let tx = strand_tangent(sh.seg);
            var z = -(d - tx * dot(d, tx));
            if (length(z) < 1e-9) { break; }
            z = normalize(z);
            let y = cross(z, tx);
            let rad = strand_segs[2u * sh.seg].w;
            let hh = clamp(dot(sh.n, y), -1.0, 1.0);
            let wo = vec3<f32>(dot(-d, tx), dot(-d, y), dot(-d, z));
            let f = fib_new(P.fiber, hh);
            let g = fib_geometry(f, wo);
            let mel = strand_segs[2u * sh.seg + 1u].w;
            let ap_s = fib_ap(g.f, g.l, fib_sigma_a(theta_s * mel, lam));
            let sa = fib_sigma_a(theta * mel, lam);
            let un = vec3<f32>(rand(&rng), rand(&rng), rand(&rng));
            let off = sh.n * (2.0 * rad);
            let ls = sample_light(x, un);
            if (ls.ok) {
                let side = select(-off, off, dot(ls.d, sh.n) > 0.0);
                if (visible(x + side, ls.d, ls.dist)) {
                    let wi = vec3<f32>(dot(ls.d, tx), dot(ls.d, y), dot(ls.d, z));
                    let fv = fib_f(fib_mn(f, g, wi), g.f, g.l, sa);
                    let pb = fib_pdf(f, g, wi, ap_s);
                    l = l + beta * fv * light_le(ls.i, lam) / ls.pdf * mis(ls.pdf, pb);
                }
            }
            let us = vec4<f32>(rand(&rng), rand(&rng), rand(&rng), rand(&rng));
            let wi = fib_sample(f, g, ap_s, us);
            let pdf = fib_pdf(f, g, wi, ap_s);
            if (!(pdf > 0.0)) { break; }
            let fv = fib_f(fib_mn(f, g, wi), g.f, g.l, sa);
            beta = beta * fv / pdf;
            d = normalize(tx * wi.x + y * wi.y + z * wi.z);
            o = x + select(-off, off, dot(d, sh.n) > 0.0);
            last_pdf = pdf;
        }
        if (!(beta > 0.0) || beta > 3.0e38) { break; }
    }
    lanes[path] = vec2<f32>(l, lam);
}
