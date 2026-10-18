// Speed piece 1: the opaque scene through the hardware's ray queries
// (wgpu 30's `EXPERIMENTAL_RAY_QUERY`, Metal's intersection queries).
//
// Compiled in only when the device has the feature (`rt::Accel`), after
// `enable wgpu_ray_query;` at the top of the module. One TLAS instance per
// (placement, kind): a triangle BLAS of the bottom-level tree's mesh, opaque,
// and an AABB BLAS of its analytic prims, intersected here by the same
// `hit_prim` the software walk uses, in the instance space `to_local` gives.
// `rt_inst[instance_index]` is (instance record, first prim record), so a
// hit maps to exactly the `Hit` the software walk returns.
//
// Determinism: the acceleration structure is built once per geometry and
// the hardware's traversal is a function of (structure, ray), so forward and
// replay — and two runs — commit the same primitive. A tie in t between two
// triangles is resolved by the hardware, not by prim index as the software
// walk does; `phase_speed` tests replay bit-exactness and run-to-run
// equality on the bench scenes under both walks.

@group(0) @binding(57) var accel: acceleration_structure;
@group(0) @binding(58) var<storage, read> rt_inst: array<vec2<u32>>;

fn intersect_hw(o: vec3<f32>, d: vec3<f32>, tmax: f32, any: bool) -> Hit {
    var rq: ray_query;
    let flags = select(0u, RAY_FLAG_TERMINATE_ON_FIRST_HIT, any);
    rayQueryInitialize(&rq, accel, RayDesc(flags, 0xffu, 0.0, tmax, o, d));
    var best_t = tmax;
    var best = Hit(tmax, NONE, NONE);
    loop {
        if (!rayQueryProceed(&rq)) { break; }
        let c = rayQueryGetCandidateIntersection(&rq);
        if (c.kind != RAY_QUERY_INTERSECTION_AABB) { continue; }
        let m = rt_inst[c.instance_index];
        let i = m.y + c.primitive_index;
        let l = to_local(m.x, o, d);
        let t = hit_prim(l[0], l[1], i, 0.0);
        let cm = rayQueryGetCommittedIntersection(&rq);
        if (cm.kind != RAY_QUERY_INTERSECTION_NONE) {
            best_t = min(best_t, cm.t);
        }
        if (t > 0.0 && t < best_t) {
            best_t = t;
            rayQueryGenerateIntersection(&rq, t);
            if (any) {
                break;
            }
        }
    }
    let h = rayQueryGetCommittedIntersection(&rq);
    if (h.kind == RAY_QUERY_INTERSECTION_NONE) {
        return best;
    }
    let m = rt_inst[h.instance_index];
    return Hit(h.t, m.y + h.primitive_index, m.x);
}

fn intersect(o: vec3<f32>, d: vec3<f32>, tmax: f32, any: bool) -> Hit {
    if (params.rt != 0u) {
        return intersect_hw(o, d, tmax, any);
    }
    return intersect_sw(o, d, tmax, any);
}

