// M1: 3DGS splats in the scene kernel's TLAS (3DGRT proxies).
//
// One TLAS instance (mask SPLAT_MASK) holds an AABB BLAS of every splat's
// proxy box. A segment is composited in rounds: each round runs one ray query
// over the splat instance and keeps a sorted buffer of the K nearest
// responses strictly after the last one composited, ordered by (t, index) —
// a total order, so the result does not depend on the order the hardware
// reports candidates in. Once the buffer is full, the K-th entry's t is
// committed (rayQueryGenerateIntersection), so the traversal culls boxes
// that start past it; a culled box's peak lies past its entry, so the set
// is unchanged. `C += T α c; T *= 1 − α`, out at T ≤ 1e-4, exactly
// splat_gpu::composite_cpu.
//
// The response: the peak along the ray is at t* = −dᵀAΔ / dᵀAd (A = Σ⁻¹,
// 3DGRT's sort key), and α = min(0.99, o · exp(−½ q)) with q the Mahalanobis
// distance of the ray from μ under Σ's marginal across the ray, plus the
// trainer's 0.3 px² dilation at that distance (without it a ray tracer
// aliases the needles a rasteriser-trained cloud relies on being widened).
// Dropped below 1/255, the trainer's own cut, and when the peak lies outside
// the proxy box (√(2 ln 255 o) σ + a margin for the dilation), so culling
// by box is exact.
// Colour is the cloud's SH at the ray direction (+ ½, clamped), an sRGB
// value: composited as is (as the trainer does) and decoded per layer.

struct SplatParams {
    info: vec4<u32>,     // count, degree, u32 words of SH per splat, rounds
}

@group(0) @binding(60) var<storage, read> splat_rec: array<vec4<f32>>;  // 5 per splat: (μ, opacity), A, Σ, box
@group(0) @binding(61) var<storage, read> splat_sh: array<u32>;         // f16 pairs, coefficient-major RGB
@group(0) @binding(62) var<uniform> SP: SplatParams;

const SPLAT_MASK: u32 = 0x10u;
const SPLAT_K: u32 = 16u;
const SPLAT_T_CUT: f32 = 1e-4;

struct SplatSeg {
    rgb: vec3<f32>,      // Σ T α c, sRGB-valued
    trans: f32,          // T after the segment
    t_half: f32,         // t at which T first fell to ≤ ½ (−1: never)
    first: u32,          // the splat that crossed ½ (NONE: none)
}

// (t*, α) of splat i along o + t d (d unit); α = 0 when it does not respond.
// `pix` is the ray's angular pixel size (radians; 0: none): the trainer's
// 0.3 px² screen-space dilation becomes an isotropic 0.3 (pix t*)² in the
// plane across the ray, where the response is the 2D marginal's.
fn splat_eval(i: u32, o: vec3<f32>, d: vec3<f32>, pix: f32) -> vec2<f32> {
    let r0 = splat_rec[5u * i];
    let a0 = splat_rec[5u * i + 1u];     // A00 A01 A02 A11
    let a1 = splat_rec[5u * i + 2u];     // A12 A22 Σ00 Σ01
    let s1 = splat_rec[5u * i + 3u];     // Σ02 Σ11 Σ12 Σ22
    let ext = splat_rec[5u * i + 4u].xyz;
    let del = o - r0.xyz;
    let ad = vec3<f32>(a0.x * d.x + a0.y * d.y + a0.z * d.z,
                       a0.y * d.x + a0.w * d.y + a1.x * d.z,
                       a0.z * d.x + a1.x * d.y + a1.y * d.z);
    let dad = dot(d, ad);
    if (!(dad > 0.0)) { return vec2<f32>(0.0, 0.0); }
    let t = -dot(ad, del) / dad;
    let x = del + t * d;                     // the peak, relative to μ
    if (any(abs(x) > ext)) { return vec2<f32>(t, 0.0); }
    // the marginal across the ray: basis (e1, e2) ⟂ d
    let e1 = basis_t(d);
    let e2 = basis_b(d);
    let se1 = vec3<f32>(a1.z * e1.x + a1.w * e1.y + s1.x * e1.z,
                        a1.w * e1.x + s1.y * e1.y + s1.z * e1.z,
                        s1.x * e1.x + s1.z * e1.y + s1.w * e1.z);
    let se2 = vec3<f32>(a1.z * e2.x + a1.w * e2.y + s1.x * e2.z,
                        a1.w * e2.x + s1.y * e2.y + s1.z * e2.z,
                        s1.x * e2.x + s1.z * e2.y + s1.w * e2.z);
    let fp = pix * max(t, 0.0);
    let dil = 0.3 * fp * fp;
    let c00 = dot(e1, se1) + dil;
    let c01 = dot(e1, se2);
    let c11 = dot(e2, se2) + dil;
    let det = c00 * c11 - c01 * c01;
    if (!(det > 0.0)) { return vec2<f32>(t, 0.0); }
    let u = dot(e1, x);
    let v = dot(e2, x);
    let q = (c11 * u * u - 2.0 * c01 * u * v + c00 * v * v) / det;
    let a = min(0.99, r0.w * exp(-0.5 * q));
    if (a < 1.0 / 255.0) { return vec2<f32>(t, 0.0); }
    return vec2<f32>(t, a);
}

fn sh_coef(base: u32, k: u32) -> vec3<f32> {
    let h = 3u * k;
    var c: vec3<f32>;
    for (var j = 0u; j < 3u; j = j + 1u) {
        let x = h + j;
        let w = unpack2x16float(splat_sh[base + x / 2u]);
        c[j] = select(w.x, w.y, (x & 1u) == 1u);
    }
    return c;
}

// The 3DGS view-dependent colour (degree ≤ 3), d from the camera to the splat.
fn splat_colour(i: u32, d: vec3<f32>) -> vec3<f32> {
    let base = i * SP.info.z;
    var c = vec3<f32>(0.5) + 0.28209479177387814 * sh_coef(base, 0u);
    let deg = SP.info.y;
    if (deg >= 1u) {
        let x = d.x;
        let y = d.y;
        let z = d.z;
        let c1 = 0.4886025119029199;
        c = c - c1 * y * sh_coef(base, 1u) + c1 * z * sh_coef(base, 2u) - c1 * x * sh_coef(base, 3u);
        if (deg >= 2u) {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            c = c + 1.0925484305920792 * x * y * sh_coef(base, 4u)
                  - 1.0925484305920792 * y * z * sh_coef(base, 5u)
                  + 0.31539156525252005 * (2.0 * zz - xx - yy) * sh_coef(base, 6u)
                  - 1.0925484305920792 * x * z * sh_coef(base, 7u)
                  + 0.5462742152960396 * (xx - yy) * sh_coef(base, 8u);
            if (deg >= 3u) {
                c = c - 0.5900435899266435 * y * (3.0 * xx - yy) * sh_coef(base, 9u)
                      + 2.890611442640554 * x * y * z * sh_coef(base, 10u)
                      - 0.4570457994644658 * y * (4.0 * zz - xx - yy) * sh_coef(base, 11u)
                      + 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(base, 12u)
                      - 0.4570457994644658 * x * (4.0 * zz - xx - yy) * sh_coef(base, 13u)
                      + 1.445305721320277 * z * (xx - yy) * sh_coef(base, 14u)
                      - 0.5900435899266435 * x * (xx - 3.0 * yy) * sh_coef(base, 15u);
            }
        }
    }
    return max(c, vec3<f32>(0.0));
}

struct SplatRound {
    kt: array<f32, SPLAT_K>,
    ki: array<u32, SPLAT_K>,
    n: u32,
}

// One round: the K nearest responses after (t_last, i_last) (after tmin
// when not `started`). Its own function, so each round's ray query is a
// fresh variable (Metal will not re-initialise one in a loop).
fn splat_round(o: vec3<f32>, d: vec3<f32>, pix: f32, tmin: f32, tmax: f32, started: bool, t_last: f32, i_last: u32) -> SplatRound {
    var r: SplatRound;
    r.n = 0u;
    var rq: ray_query;
    rayQueryInitialize(&rq, accel, RayDesc(0u, SPLAT_MASK, max(select(tmin, t_last, started), 0.0), tmax, o, d));
    loop {
        if (!rayQueryProceed(&rq)) { break; }
        let c = rayQueryGetCandidateIntersection(&rq);
        if (c.kind != RAY_QUERY_INTERSECTION_AABB) { continue; }
        let i = c.primitive_index;
        let e = splat_eval(i, o, d, pix);
        if (e.y <= 0.0 || !(e.x < tmax)) { continue; }
        // strictly after the last composited (t, i)
        if (started) {
            if (e.x < t_last || (e.x == t_last && i <= i_last)) { continue; }
        } else if (!(e.x > tmin)) { continue; }
        if (r.n == SPLAT_K && (e.x > r.kt[SPLAT_K - 1u] || (e.x == r.kt[SPLAT_K - 1u] && i > r.ki[SPLAT_K - 1u]))) {
            continue;
        }
        // insertion into the sorted buffer
        var j = min(r.n, SPLAT_K - 1u);
        loop {
            if (j == 0u) { break; }
            let pt = r.kt[j - 1u];
            let pi = r.ki[j - 1u];
            if (pt < e.x || (pt == e.x && pi < i)) { break; }
            r.kt[j] = pt;
            r.ki[j] = pi;
            j = j - 1u;
        }
        r.kt[j] = e.x;
        r.ki[j] = i;
        r.n = min(r.n + 1u, SPLAT_K);
        if (r.n == SPLAT_K) {
            let tk = r.kt[SPLAT_K - 1u];
            let cm = rayQueryGetCommittedIntersection(&rq);
            if (cm.kind == RAY_QUERY_INTERSECTION_NONE || tk < cm.t) {
                rayQueryGenerateIntersection(&rq, tk);
            }
        }
    }
    return r;
}

// Composite the splats along o + t d, t in (tmin, tmax), d unit.
fn splat_segment(o: vec3<f32>, d: vec3<f32>, pix: f32, tmin: f32, tmax: f32) -> SplatSeg {
    var seg: SplatSeg;
    seg.rgb = vec3<f32>(0.0);
    seg.trans = 1.0;
    seg.t_half = -1.0;
    seg.first = NONE;
    if (SP.info.x == 0u) { return seg; }
    var t_last = tmin;
    var i_last = NONE;
    var started = false;
    for (var round = 0u; round < SP.info.w; round = round + 1u) {
        let r = splat_round(o, d, pix, tmin, tmax, started, t_last, i_last);
        for (var k = 0u; k < r.n; k = k + 1u) {
            let i = r.ki[k];
            let e = splat_eval(i, o, d, pix);
            let w = seg.trans * e.y;
            seg.rgb = seg.rgb + w * splat_colour(i, d);
            seg.trans = seg.trans * (1.0 - e.y);
            if (seg.t_half < 0.0 && seg.trans <= 0.5) {
                seg.t_half = r.kt[k];
                seg.first = i;
            }
            if (seg.trans <= SPLAT_T_CUT) {
                seg.trans = 0.0;
                return seg;
            }
        }
        if (r.n < SPLAT_K) { break; }
        t_last = r.kt[SPLAT_K - 1u];
        i_last = r.ki[SPLAT_K - 1u];
        started = true;
    }
    return seg;
}

fn srgb_dec3(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(srgb_dec(c.x), srgb_dec(c.y), srgb_dec(c.z));
}

// The segment's linear Rec.709 radiance: the layer's sRGB colour decoded
// and weighted by its coverage.
fn splat_linear(g: SplatSeg) -> vec3<f32> {
    let a = 1.0 - g.trans;
    if (a <= 0.0) { return vec3<f32>(0.0); }
    return srgb_dec3(g.rgb / a) * a;
}
