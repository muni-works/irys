// The renderer's prelude: the types and helpers that BOTH halves of a
// composed trace shader need — the integrator above and whatever geometry
// module a client supplies below.
//
// Composed first, after `bsdf.wgsl`, by `kosm_render::gpu::shaders::compose`.

// A ray's closest hit. `face_idx` is the geometry module's own primitive id;
// the integrator only ever compares it against the two sentinels below and
// hands it back to the module's accessors.
struct RayHit {
    t: f32,
    face_idx: u32,
    uv: vec2<f32>,
}

// No hit. Every geometry module must return this in `face_idx` on a miss.
const FACE_IDX_MISS: u32 = 0xFFFFFFFFu;
// The integrator's implicit ground plane. Reserved: a geometry module must
// never produce it.
const FACE_IDX_GROUND: u32 = 0xFFFFFFFEu;

// Far plane and the generic "is this t meaningful" floor.
const MAX_T: f32 = 1e10;
const EPSILON: f32 = 1e-6;


fn intersect_aabb(origin: vec3<f32>, inv_dir: vec3<f32>, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> vec2<f32> {
    let t1 = (aabb_min - origin) * inv_dir;
    let t2 = (aabb_max - origin) * inv_dir;

    let t_min = min(t1, t2);
    let t_max = max(t1, t2);

    let t_enter = max(max(t_min.x, t_min.y), t_min.z);
    let t_exit = min(min(t_max.x, t_max.y), t_max.z);

    return vec2<f32>(t_enter, t_exit);
}

// ─── scale-aware self-intersection epsilon ────────────────────────────────
//
// Every ray that leaves a surface has to clear that surface by more than the
// float grid at that point, and on the GPU that grid is f32. A model authored
// in millimetres puts a wall at a coordinate of 2.6e4, where one ulp is about
// 2e-3 — so the fixed `p + n * 1e-4` this shader used to apply rounded away
// entirely, the ray started *on* the wall, and the wall shadowed itself. The
// CPU tier never showed it: f64 has ten million ulps of headroom at the same
// coordinate.
//
// `RAY_EPS_REL` is ~16 ulps of f32 (1 ulp is 2^-23 relative), enough to clear
// the accumulated error in an intersection while staying far below any
// feature a renderer resolves. `RAY_EPS_ABS` keeps small scenes — and the
// origin itself — on the behaviour the small-unit tests pin.
const RAY_EPS_ABS: f32 = 1e-4;
const RAY_EPS_REL: f32 = 2e-6;

// The interval floor for a ray leaving `p`: absolute near the origin, relative
// once coordinates are large.
fn ray_eps(p: vec3<f32>) -> f32 {
    let scale = max(max(abs(p.x), abs(p.y)), abs(p.z));
    return max(RAY_EPS_ABS, scale * RAY_EPS_REL);
}

// Lift a hit point off its surface along the normal, by the same amount.
fn offset_origin(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return p + n * ray_eps(p);
}


// The shading frame at a hit.
// Mirrors `pathtrace::shading_frame`: when the hit carries a surface tangent it
// is Gram-Schmidt orthogonalised against the (face-forwarded) normal and used
// as the frame's x axis, so the anisotropic lobe follows the surface's own
// parameterisation. Otherwise fall back to the arbitrary `onb` basis — which is
// exactly what an isotropic material wants, since its BSDF is invariant to the
// choice.
fn shading_frame(n: vec3<f32>, dpdu: vec3<f32>) -> mat3x3<f32> {
    let t_raw = dpdu - n * dot(dpdu, n);
    // A tangent (numerically) parallel to the normal carries no direction;
    // fall back rather than normalising noise.
    if length(t_raw) > 1e-9 {
        let t = normalize(t_raw);
        return mat3x3<f32>(t, cross(n, t), n);
    }
    return onb(n);
}
