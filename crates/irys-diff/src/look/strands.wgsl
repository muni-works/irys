// Fibres on the device through wgpu 30 ray queries (`look::hair_gpu`).
//
// The host prepends `enable wgpu_ray_query;` and `const STRAND_SIDES: u32`:
// 0 for one AABB per segment, intersected here as the exact round cylinder
// `hair::Strands::intersect` is (`rayQueryGenerateIntersection`); n ≥ 3 for
// an n-sided tessellated tube per segment in a triangle BLAS, committed by
// the hardware, its fibre found from the triangle (`primitive_index /
// 2n`), the tangent and offset from that segment's axis.
//
// `strand_segs`: two vec4 per segment, (a, radius) and (b, melanin scale).

@group(0) @binding(10) var strand_accel: acceleration_structure;
@group(0) @binding(11) var<storage, read> strand_segs: array<vec4<f32>>;

struct SHit {
    t: f32,
    seg: u32,
    n: vec3<f32>,
}

// The cylinder of segment `i` on `o + t d` (`|d| = 1`): the nearest t in
// (tmin, tmax), or −1. The quadratic is solved about the ray's closest
// approach to `a`, so `|o|` far larger than the radius loses nothing in f32.
fn strand_cyl(o: vec3<f32>, d: vec3<f32>, i: u32, tmin: f32, tmax: f32) -> f32 {
    let a = strand_segs[2u * i].xyz;
    let r = strand_segs[2u * i].w;
    let b = strand_segs[2u * i + 1u].xyz;
    let ax = b - a;
    let len = length(ax);
    let tx = ax / len;
    let t0 = dot(a - o, d);
    let oc = (o + d * t0) - a;
    let dp = d - tx * dot(d, tx);
    let op = oc - tx * dot(oc, tx);
    let qa = dot(dp, dp);
    if (qa < 1e-12) { return -1.0; }
    let qb = 2.0 * dot(dp, op);
    let qc = dot(op, op) - r * r;
    let disc = qb * qb - 4.0 * qa * qc;
    if (disc < 0.0) { return -1.0; }
    let sq = sqrt(disc);
    for (var k = 0; k < 2; k = k + 1) {
        let tl = select((-qb + sq) / (2.0 * qa), (-qb - sq) / (2.0 * qa), k == 0);
        let t = t0 + tl;
        // A ray leaving through the far side of its own fibre starts on
        // that fibre's surface (hair::HairScene's offsets): its own root is
        // ≈ 1e-16 in f64 but f32 noise of order 1e-3 r here, so roots
        // within 1e-2 r of the origin are the surface it left.
        if (t <= max(tmin, 1e-2 * r) || t >= tmax) { continue; }
        let s = dot(oc + d * tl, tx);
        if (s < 0.0 || s > len) { continue; }
        return t;
    }
    return -1.0;
}

// The outward normal at `p` on segment `i`'s axis.
fn strand_normal(i: u32, p: vec3<f32>) -> vec3<f32> {
    let a = strand_segs[2u * i].xyz;
    let b = strand_segs[2u * i + 1u].xyz;
    let tx = normalize(b - a);
    let s = clamp(dot(p - a, tx), 0.0, length(b - a));
    return normalize(p - (a + tx * s));
}

fn strand_radius(i: u32) -> f32 {
    return strand_segs[2u * i].w;
}

fn strand_melanin(i: u32) -> f32 {
    return strand_segs[2u * i + 1u].w;
}

fn strand_tangent(i: u32) -> vec3<f32> {
    return normalize(strand_segs[2u * i + 1u].xyz - strand_segs[2u * i].xyz);
}

fn strand_hit(o: vec3<f32>, d: vec3<f32>, tmin: f32, tmax: f32, any: bool) -> SHit {
    var rq: ray_query;
    let flags = select(0u, RAY_FLAG_TERMINATE_ON_FIRST_HIT, any);
    rayQueryInitialize(&rq, strand_accel, RayDesc(flags, 0xffu, tmin, tmax, o, d));
    var best_t = tmax;
    loop {
        if (!rayQueryProceed(&rq)) { break; }
        let c = rayQueryGetCandidateIntersection(&rq);
        if (c.kind != RAY_QUERY_INTERSECTION_AABB) { continue; }
        let t = strand_cyl(o, d, c.primitive_index, tmin, best_t);
        if (t > 0.0) {
            // A shadow query answers here: a generated intersection is only
            // committed by the next `rayQueryProceed`, which `any` skips.
            if (any) {
                return SHit(t, c.primitive_index, vec3<f32>(0.0));
            }
            best_t = t;
            rayQueryGenerateIntersection(&rq, t);
        }
    }
    let h = rayQueryGetCommittedIntersection(&rq);
    if (h.kind == RAY_QUERY_INTERSECTION_NONE) {
        return SHit(tmax, 0xffffffffu, vec3<f32>(0.0));
    }
    var seg = h.primitive_index;
    if (STRAND_SIDES > 0u) { seg = seg / (2u * max(STRAND_SIDES, 1u)); }
    return SHit(h.t, seg, strand_normal(seg, o + d * h.t));
}
