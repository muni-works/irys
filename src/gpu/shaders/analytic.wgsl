// A small built-in geometry module: spheres and planes, traced linearly.
//
// Implements kosm-render's geometry contract (see `gpu::geometry`) so the
// renderer's own tests have something to trace without depending on a client.
// One storage buffer, at binding 1 — four to spare for a real client.

struct AnalyticPrim {
    // 0 = sphere, 1 = plane.
    kind: u32,
    material_idx: u32,
    // 0 = forward, 1 = reversed.
    orientation: u32,
    _pad: u32,
    // Sphere: centre.xyz, radius in .w. Plane: a point on it.
    a: vec4<f32>,
    // Sphere: unused. Plane: unit normal.
    b: vec4<f32>,
}

@group(0) @binding(1) var<storage, read> prims: array<AnalyticPrim>;

fn analytic_hit_sphere(origin: vec3<f32>, dir: vec3<f32>, p: AnalyticPrim) -> RayHit {
    var hit: RayHit;
    hit.t = MAX_T;
    hit.face_idx = FACE_IDX_MISS;

    // Re-origin at the closest approach, as the BRep solves do: the quartic's
    // coefficients then scale with the sphere, not with the scene.
    let oc0 = origin - p.a.xyz;
    let t0 = closest_approach_local(oc0, dir);
    let oc = oc0 + dir * t0;

    let a = dot(dir, dir);
    let b = 2.0 * dot(oc, dir);
    let c = dot(oc, oc) - p.a.w * p.a.w;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return hit;
    }
    let sq = sqrt(disc);
    var t = (-b - sq) / (2.0 * a);
    if t < -t0 {
        t = (-b + sq) / (2.0 * a);
    }
    if t < -t0 {
        return hit;
    }
    hit.t = t + t0;

    let n = normalize((origin + dir * hit.t) - p.a.xyz);
    var phi = atan2(n.y, n.x);
    if phi < 0.0 { phi += 2.0 * PI; }
    hit.uv = vec2<f32>(phi, acos(clamp(n.z, -1.0, 1.0)));
    return hit;
}

fn closest_approach_local(oc: vec3<f32>, dir: vec3<f32>) -> f32 {
    return max(-dot(oc, dir) / dot(dir, dir), 0.0);
}

fn analytic_hit_plane(origin: vec3<f32>, dir: vec3<f32>, p: AnalyticPrim) -> RayHit {
    var hit: RayHit;
    hit.t = MAX_T;
    hit.face_idx = FACE_IDX_MISS;

    let denom = dot(dir, p.b.xyz);
    if abs(denom) < EPSILON {
        return hit;
    }
    let t = dot(p.a.xyz - origin, p.b.xyz) / denom;
    if t <= 0.0 {
        return hit;
    }
    hit.t = t;
    // A plane is unbounded here, so its uv is only ever used for a tangent.
    let f = onb(p.b.xyz);
    let q = (origin + dir * t) - p.a.xyz;
    hit.uv = vec2<f32>(dot(q, f[0]), dot(q, f[1]));
    return hit;
}

fn trace_scene(origin: vec3<f32>, dir: vec3<f32>) -> RayHit {
    var best: RayHit;
    best.t = MAX_T;
    best.face_idx = FACE_IDX_MISS;

    let floor_t = ray_eps(origin);
    let n = arrayLength(&prims);
    for (var i = 0u; i < n; i++) {
        let p = prims[i];
        var h: RayHit;
        if p.kind == 0u {
            h = analytic_hit_sphere(origin, dir, p);
        } else {
            h = analytic_hit_plane(origin, dir, p);
        }
        // `h.t` is MAX_T on a miss; face_idx is filled in here, not by the
        // per-primitive solves, which do not know their own index.
        if h.t < MAX_T && h.t > floor_t && h.t < best.t {
            best = h;
            best.face_idx = i;
        }
    }
    return best;
}

fn hit_normal(hit: RayHit) -> vec3<f32> {
    let p = prims[hit.face_idx];
    if p.kind == 0u {
        let n = vec3<f32>(
            sin(hit.uv.y) * cos(hit.uv.x),
            sin(hit.uv.y) * sin(hit.uv.x),
            cos(hit.uv.y),
        );
        if p.orientation == 1u { return -n; }
        return n;
    }
    if p.orientation == 1u { return -p.b.xyz; }
    return p.b.xyz;
}

fn hit_tangent(hit: RayHit) -> vec3<f32> {
    let p = prims[hit.face_idx];
    if p.kind == 0u {
        // dP/dphi, normalised; degenerate at the poles.
        let s = sin(hit.uv.y);
        if abs(s) < 1e-6 { return vec3<f32>(0.0); }
        return vec3<f32>(-sin(hit.uv.x), cos(hit.uv.x), 0.0);
    }
    return onb(p.b.xyz)[0];
}

fn hit_material_index(hit: RayHit) -> u32 {
    return prims[hit.face_idx].material_idx;
}

fn hit_orientation(hit: RayHit) -> u32 {
    return prims[hit.face_idx].orientation;
}
