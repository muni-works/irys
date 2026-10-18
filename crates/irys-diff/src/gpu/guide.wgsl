// Phase 3: online path guiding at diffuse vertices — a fixed-resolution
// cousin of practical path guiding (Müller et al. 2017): a uniform spatial
// grid over the scene, split six ways by the normal's major axis, each cell
// a 16 × 16 equal-area histogram over the sphere of incident radiance.
//
// Appended to `wavefront.wgsl` + `restir.wgsl` (one module).
//
// Sampling (in `bounce`): with probability c (the cell trained, else 0) a
// direction from the cell's histogram, otherwise the cosine lobe; the pdf
// is the mixture q = c g(ω) + (1 − c) cos⁺/π, the throughput takes
// albedo cos/π / q, and q is the BSDF strategy's pdf for every MIS weight
// (escapes, emitter hits, the NEE weight at the vertex). The cosine
// component covers every direction the integrand lives on, so the forward
// pass stays unbiased whatever the histogram holds; the guide is a constant
// between `guide_build` calls, so forward and replay see the same pdfs, and
// it is never differentiated (sampling is detached).
//
// Training (forward passes with `params.gres.w & 2`): every diffuse vertex
// leaves a record — cell and bin of the sampled direction, the radiance
// collected so far (NEE included), the throughput after it, q — and
// `finish` turns each into Li / q with Li = (L − L⁺) / T, summed over the
// path's wavelengths: an unbiased estimate of ∫_bin of the radiance the
// BSDF strategy is responsible for (direct light arrives MIS-weighted by
// w_B, so the guide does not learn what NEE already samples well — training
// on NEE's own share as well was measured worse on the box, 2.3e-3 against
// 1.7e-3 relMSE at 256 spp, and only 10 % better on the room). Sums are
// 64-bit fixed point in two u32 atomics, so the histogram is the same
// integers whatever order the threads add in: deterministic.
//
// `guide_build` (one workgroup per cell, one thread per bin): the
// histogram normalised, mixed with 10 % uniform, prefix-summed into a CDF.

const GBINS: u32 = 256u;
const GSIDE: u32 = 16u;
const GFIX: f32 = 65536.0;       // fixed-point scale of the histogram

@group(0) @binding(50) var<storage, read_write> g_hist: array<atomic<u32>>;   // (lo, hi) per bin
@group(0) @binding(51) var<storage, read_write> g_cdf: array<f32>;            // per bin
@group(0) @binding(52) var<storage, read_write> g_ok: array<u32>;             // per cell: trained
@group(0) @binding(53) var<storage, read_write> g_rec: array<vec4<f32>>;      // per path × depth
@group(0) @binding(54) var<storage, read_write> g_nrec: array<u32>;           // per path
@group(0) @binding(55) var<storage, read_write> g_cnt: array<atomic<u32>>;    // per cell: samples splatted

fn guide_on() -> bool {
    return (params.gres.w & 1u) != 0u;
}

fn guide_training() -> bool {
    return (params.gres.w & 2u) != 0u;
}

// The cell of a point with normal n: grid voxel × major axis of n (6).
fn guide_cell(x: vec3<f32>, n: vec3<f32>) -> u32 {
    let r = vec3<i32>(params.gres.xyz);
    let v = vec3<i32>(floor((x - params.gmin.xyz) * params.gmin.w));
    let c = clamp(v, vec3<i32>(0), r - vec3<i32>(1));
    let a = abs(n);
    var axis = 0u;
    if (a.y > a.x && a.y >= a.z) {
        axis = 2u;
    } else if (a.z > a.x && a.z > a.y) {
        axis = 4u;
    }
    let comp = select(select(n.x, n.y, axis == 2u), n.z, axis == 4u);
    axis = axis + select(0u, 1u, comp < 0.0);
    let vox = (u32(c.z) * u32(r.y) + u32(c.y)) * u32(r.x) + u32(c.x);
    return vox * 6u + axis;
}

fn guide_bin(d: vec3<f32>) -> u32 {
    let row = min(u32((d.z + 1.0) * 0.5 * f32(GSIDE)), GSIDE - 1u);
    let phi = atan2(d.y, d.x);
    let col = min(u32((phi + PI) / (2.0 * PI) * f32(GSIDE)), GSIDE - 1u);
    return row * GSIDE + col;
}

fn guide_mix(cell: u32) -> f32 {
    return select(0.0, params.guide.x, g_ok[cell] != 0u);
}

fn guide_pdf(cell: u32, d: vec3<f32>) -> f32 {
    let b = guide_bin(d);
    let base = cell * GBINS;
    let lo = select(0.0, g_cdf[base + b - 1u], b > 0u);
    return (g_cdf[base + b] - lo) * f32(GBINS) / (4.0 * PI);
}

// A direction from the cell's histogram: the bin by the CDF, uniform in it.
fn guide_sample(cell: u32, u1: f32, u2: f32) -> vec3<f32> {
    let base = cell * GBINS;
    var lo = 0u;
    var hi = GBINS - 1u;
    loop {
        if (lo >= hi) { break; }
        let mid = (lo + hi) / 2u;
        if (g_cdf[base + mid] <= u1) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    let b = lo;
    let c0 = select(0.0, g_cdf[base + b - 1u], b > 0u);
    let c1 = g_cdf[base + b];
    let v = clamp((u1 - c0) / max(c1 - c0, 1e-12), 0.0, 1.0);
    let z = clamp(-1.0 + 2.0 * (f32(b / GSIDE) + v) / f32(GSIDE), -1.0, 1.0);
    let phi = -PI + 2.0 * PI * (f32(b % GSIDE) + u2) / f32(GSIDE);
    let r = sqrt(max(1.0 - z * z, 0.0));
    return vec3<f32>(r * cos(phi), r * sin(phi), z);
}

// Add `v` (radiance per pdf) to a bin, in 64-bit fixed point.
fn guide_splat(cell: u32, b: u32, v: f32) {
    if (!(v > 0.0)) { return; }
    let f = u32(min(v, 60000.0) * GFIX);
    let k = 2u * (cell * GBINS + b);
    let old = atomicAdd(&g_hist[k], f);
    if (old > 0xffffffffu - f) {
        atomicAdd(&g_hist[k + 1u], 1u);
    }
    atomicAdd(&g_cnt[cell], 1u);
}

fn guide_record(i: u32, cell: u32, b: u32, lplus: f32, t_next: f32, q: f32) {
    let n = g_nrec[i];
    if (n >= params.gres2.x) { return; }
    g_rec[i * params.gres2.x + n] = vec4<f32>(bitcast<f32>(cell * GBINS + b), lplus, t_next, q);
    g_nrec[i] = n + 1u;
}

// `finish`'s training half: every record of path `i` into the histogram.
fn guide_flush(i: u32, l: f32) {
    let n = g_nrec[i];
    for (var j = 0u; j < n; j = j + 1u) {
        let r = g_rec[i * params.gres2.x + j];
        let cb = bitcast<u32>(r.x);
        let li = max(l - r.y, 0.0) / r.z;
        if (r.z > 0.0 && r.w > 0.0) {
            guide_splat(cb / GBINS, cb % GBINS, li / r.w);
        }
    }
    g_nrec[i] = 0u;
}

var<workgroup> gsum: array<f32, 256>;

@compute @workgroup_size(256)
fn guide_build(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let cell = wid.x + wid.y * nwg.x;
    if (cell >= params.gres2.y) { return; }
    let k = 2u * (cell * GBINS + lid.x);
    let v = f32(atomicLoad(&g_hist[k])) / GFIX + f32(atomicLoad(&g_hist[k + 1u])) * (4294967296.0 / GFIX);
    gsum[lid.x] = v;
    workgroupBarrier();
    // Hillis–Steele inclusive scan, fixed order.
    for (var s = 1u; s < 256u; s = s << 1u) {
        var add = 0.0;
        if (lid.x >= s) {
            add = gsum[lid.x - s];
        }
        workgroupBarrier();
        gsum[lid.x] = gsum[lid.x] + add;
        workgroupBarrier();
    }
    let total = gsum[255];
    let eps = 0.1;
    var c = f32(lid.x + 1u) / 256.0;
    if (total > 0.0) {
        c = (1.0 - eps) * gsum[lid.x] / total + eps * f32(lid.x + 1u) / 256.0;
    }
    if (lid.x == 255u) {
        c = 1.0;
        g_ok[cell] = select(0u, 1u, total > 0.0 && atomicLoad(&g_cnt[cell]) >= params.gres2.z);
    }
    g_cdf[cell * GBINS + lid.x] = c;
}
