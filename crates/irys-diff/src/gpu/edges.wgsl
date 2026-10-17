// Phase 2e: the edge term's sampling on the device (`vis::EdgeSamples` in
// f32). Appended to wavefront.wgsl, whose scene bindings (23–25) and
// traversal (`intersect`, `to_local`) it uses; `params` is bound for
// `inst_base` only. Three stages, every one a pure function of its inputs:
//
// 1. `edge_cand`, one thread per (moving instance, triangle, edge) slot:
//    project, classify, clip, and write `len × importance` (0 for no
//    candidate) and the clipped range.
// 2. `scan_block`, `scan_tops`, `scan_fill`: the inclusive prefix sum of
//    those weights, in a fixed order (each block summed sequentially by one
//    thread, the block totals by one thread), so the same weights give the
//    same bits every time.
// 3. `edge_sample`, one thread per sample: stratified `u`, binary search,
//    the edge point, its material point, the ownership ray through the
//    device's own tree, and the weight per parameter.

struct EdgeParams {
    seed: vec2<u32>,
    n: u32,          // samples
    slots: u32,
    moving: u32,     // moving instances
    np: u32,         // parameters
    policy: u32,     // 0: 2d's importance; 1: 2e's front-only silhouettes
    blocks: u32,
    eps_px: f32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}

// Per moving instance: (instance record, first prim record, triangles,
// first slot), then the first triangle's index into `nbr`/`dihedral`.
struct Mover {
    rec: u32,
    prim_base: u32,
    tris: u32,
    slot_base: u32,
    edge_base: u32,
    _a: u32,
    _b: u32,
    _c: u32,
}

@group(0) @binding(27) var<uniform> ep: EdgeParams;
@group(0) @binding(28) var<storage, read> movers: array<Mover>;
// Per moving instance: to_world rows (3), then per parameter its
// derivative's rows (3 each): `4 × ... ` vec4s, stride 3 (1 + np).
@group(0) @binding(29) var<storage, read> xf: array<vec4<f32>>;
// Per triangle of a moving tree, per edge: neighbour (or NONE / FLAT).
@group(0) @binding(30) var<storage, read> nbr: array<u32>;
@group(0) @binding(31) var<storage, read> dihedral: array<f32>;
@group(0) @binding(32) var<storage, read_write> wt: array<f32>;      // slot weight, then cum
@group(0) @binding(33) var<storage, read_write> rng_t: array<vec4<f32>>; // slot (t0, t1, imp, _)
@group(0) @binding(34) var<storage, read_write> btot: array<f32>;
@group(0) @binding(35) var<storage, read_write> pts_out: array<vec2<f32>>;
@group(0) @binding(36) var<storage, read_write> smp_out: array<f32>;    // per sample: pixel, visible, then np weights

const FLAT: u32 = 0xfffffffeu;
const BACK_WEIGHT: f32 = 0.02;
const SCAN_BLOCK: u32 = 1024u;

fn flat_e(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * 64u;
}

fn mover_of(slot: u32) -> u32 {
    var m = 0u;
    for (var i = 1u; i < ep.moving; i = i + 1u) {
        if (slot >= movers[i].slot_base) { m = i; }
    }
    return m;
}

fn world(mi: u32, p: vec3<f32>) -> vec3<f32> {
    let r = mi * 3u * (1u + ep.np);
    return vec3<f32>(dot(xf[r].xyz, p) + xf[r].w, dot(xf[r + 1u].xyz, p) + xf[r + 1u].w, dot(xf[r + 2u].xyz, p) + xf[r + 2u].w);
}

fn dworld(mi: u32, j: u32, p: vec3<f32>) -> vec3<f32> {
    let r = mi * 3u * (1u + ep.np) + 3u * (1u + j);
    return vec3<f32>(dot(xf[r].xyz, p) + xf[r].w, dot(xf[r + 1u].xyz, p) + xf[r + 1u].w, dot(xf[r + 2u].xyz, p) + xf[r + 2u].w);
}

fn obj(prim: u32, k: u32) -> vec3<f32> {
    return geo[3u * prim + k].xyz;
}

// `Scene::project`: pixel coordinates, and `z` (≤ 0 behind the eye).
fn project(x: vec3<f32>) -> vec3<f32> {
    let d = x - params.eye.xyz;
    let z = dot(d, params.fwd.xyz);
    let w = f32(params.size.x);
    let h = f32(params.size.y);
    let kx = w / (2.0 * params.tan_half * w / h);
    let ky = h / (2.0 * params.tan_half);
    return vec3<f32>(dot(d, params.right.xyz) / z * kx + w / 2.0, h / 2.0 - dot(d, params.up.xyz) / z * ky, z);
}

// The world normal of a moving triangle (inverse transpose: orientation
// kept under a mirror, as `apply_normal`), unnormalised: the cofactor
// matrix over the determinant's sign.
fn wnormal(mi: u32, prim: u32) -> vec3<f32> {
    let r = mi * 3u * (1u + ep.np);
    let a = xf[r].xyz;
    let b = xf[r + 1u].xyz;
    let c = xf[r + 2u].xyz;
    // Columns of the linear part are (a.x, b.x, c.x) etc.; its cofactor
    // matrix maps normals.
    let c0 = vec3<f32>(a.x, b.x, c.x);
    let c1 = vec3<f32>(a.y, b.y, c.y);
    let c2 = vec3<f32>(a.z, b.z, c.z);
    let nl = cross(obj(prim, 1u) - obj(prim, 0u), obj(prim, 2u) - obj(prim, 0u));
    let det = dot(c0, cross(c1, c2));
    return (cross(c1, c2) * nl.x + cross(c2, c0) * nl.y + cross(c0, c1) * nl.z) * sign(det);
}

fn facing(mi: u32, prim: u32) -> f32 {
    let x = world(mi, obj(prim, 0u));
    let n = wnormal(mi, prim);
    let v = params.eye.xyz - x;
    return dot(n, v) / max(length(n) * length(v), 1e-30);
}

fn crease_weight(c: f32) -> f32 {
    return clamp(4.0 * (1.0 - c), 0.02, 1.0);
}

// Liang–Barsky: the part of a → b inside the frame, (t0, t1), t0 ≥ t1 empty.
fn clip(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let w = f32(params.size.x);
    let h = f32(params.size.y);
    var t0 = 0.0;
    var t1 = 1.0;
    let d = b - a;
    let ps = array<f32, 4>(-d.x, d.x, -d.y, d.y);
    let qs = array<f32, 4>(a.x, w - a.x, a.y, h - a.y);
    for (var i = 0u; i < 4u; i = i + 1u) {
        let p = ps[i];
        let q = qs[i];
        if (p == 0.0) {
            if (q < 0.0) { return vec2<f32>(1.0, 0.0); }
        } else {
            let r = q / p;
            if (p < 0.0) { t0 = max(t0, r); } else { t1 = min(t1, r); }
        }
    }
    return vec2<f32>(t0, t1);
}

@compute @workgroup_size(64)
fn edge_cand(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let slot = flat_e(gid, nwg);
    if (slot >= ep.slots) { return; }
    wt[slot] = 0.0;
    rng_t[slot] = vec4<f32>(0.0);
    let mi = mover_of(slot);
    let mv = movers[mi];
    let local = slot - mv.slot_base;
    let tri = local / 3u;
    let e = local % 3u;
    let code = nbr[mv.edge_base + local];
    if (code == FLAT) { return; }
    let prim = mv.prim_base + tri;
    let f = facing(mi, prim) > 0.0;
    var imp = 1.0;
    if (code == NONE) {
        imp = select(BACK_WEIGHT, 1.0, f);
    } else {
        let fn_ = facing(mi, mv.prim_base + code);
        let fo = fn_ > 0.0;
        if (f && fo) {
            let a = facing(mi, prim);
            if (a < fn_ || (a == fn_ && tri > code)) { return; }
            imp = crease_weight(dihedral[mv.edge_base + local]);
        } else if (!f && !fo) {
            imp = BACK_WEIGHT;
        } else if (f) {
            imp = 1.0;
        } else {
            imp = select(1.0, BACK_WEIGHT, ep.policy == 1u);
        }
    }
    let pa = project(world(mi, obj(prim, e)));
    let pb = project(world(mi, obj(prim, (e + 1u) % 3u)));
    var pc = project(world(mi, obj(prim, (e + 2u) % 3u)));
    if (pa.z <= 0.0 || pb.z <= 0.0) { return; }
    if (pc.z <= 0.0) {
        // The third corner behind the eye: a point just inside the
        // triangle next to the edge says which side is its, as on the host.
        let m = 0.5 * (world(mi, obj(prim, e)) + world(mi, obj(prim, (e + 1u) % 3u)));
        pc = project(m + (world(mi, obj(prim, (e + 2u) % 3u)) - m) * 1e-3);
        if (pc.z <= 0.0) { return; }
    }
    let t = clip(pa.xy, pb.xy);
    if (t.x >= t.y) { return; }
    let len = length(pb.xy - pa.xy) * (t.y - t.x);
    if (len <= 0.0) { return; }
    wt[slot] = len * imp;
    rng_t[slot] = vec4<f32>(t.x, t.y, imp, 0.0);
}

@compute @workgroup_size(64)
fn scan_block(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let b = flat_e(gid, nwg);
    if (b >= ep.blocks) { return; }
    var s = 0.0;
    for (var i = b * SCAN_BLOCK; i < min((b + 1u) * SCAN_BLOCK, ep.slots); i = i + 1u) {
        s = s + wt[i];
    }
    btot[b] = s;
}

@compute @workgroup_size(1)
fn scan_tops() {
    var s = 0.0;
    for (var b = 0u; b < ep.blocks; b = b + 1u) {
        let x = btot[b];
        btot[b] = s;
        s = s + x;
    }
    btot[ep.blocks] = s;
}

@compute @workgroup_size(64)
fn scan_fill(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let b = flat_e(gid, nwg);
    if (b >= ep.blocks) { return; }
    var s = btot[b];
    for (var i = b * SCAN_BLOCK; i < min((b + 1u) * SCAN_BLOCK, ep.slots); i = i + 1u) {
        s = s + wt[i];
        wt[i] = s;
    }
}

fn erng(i: u32) -> vec2<u32> {
    var s = (ep.seed ^ vec2<u32>(0x65646765u, 0u)) ^ mul64(vec2<u32>(i, 0u), GOLDEN);
    _ = rng_next(&s);
    return s;
}

fn screen(p: vec2<f32>) -> vec2<f32> {
    let w = f32(params.size.x);
    let h = f32(params.size.y);
    return vec2<f32>((2.0 * p.x / w - 1.0) * params.tan_half * w / h, (1.0 - 2.0 * p.y / h) * params.tan_half);
}

fn ray_dir(q: vec2<f32>) -> vec3<f32> {
    return normalize(params.fwd.xyz + params.right.xyz * q.x + params.up.xyz * q.y);
}

@compute @workgroup_size(64)
fn edge_sample(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = flat_e(gid, nwg);
    if (i >= ep.n) { return; }
    let stride = 2u + ep.np;
    for (var j = 0u; j < stride; j = j + 1u) { smp_out[i * stride + j] = 0.0; }
    let total = btot[ep.blocks];
    if (total <= 0.0) {
        pts_out[2u * i] = vec2<f32>(0.0);
        pts_out[2u * i + 1u] = vec2<f32>(0.0);
        return;
    }
    var s = erng(i);
    let u = (f32(i) + rand(&s)) / f32(ep.n) * total;
    // partition_point(c <= u)
    var lo = 0u;
    var hi = ep.slots;
    loop {
        if (lo >= hi) { break; }
        let mid = (lo + hi) / 2u;
        if (wt[mid] <= u) { lo = mid + 1u; } else { hi = mid; }
    }
    let si = min(lo, ep.slots - 1u);
    let start = select(0.0, wt[si - 1u], si > 0u);
    let frac = clamp((u - start) / (wt[si] - start), 0.0, 1.0);
    let r = rng_t[si];
    // f32 cumulative sums can put `u` past the last slot's sum, onto a
    // trailing slot that is no candidate (importance 0): its weight would
    // be `x / 0`. Score it 0 (probability ~1e-7 per sample).
    if (r.z <= 0.0 || wt[si] <= start) {
        pts_out[2u * i] = vec2<f32>(0.0);
        pts_out[2u * i + 1u] = vec2<f32>(0.0);
        return;
    }
    let mi = mover_of(si);
    let mv = movers[mi];
    let local = si - mv.slot_base;
    let tri = local / 3u;
    let e = local % 3u;
    let prim = mv.prim_base + tri;
    let la = obj(prim, e);
    let lb = obj(prim, (e + 1u) % 3u);
    let a = world(mi, la);
    let b = world(mi, lb);
    let pa = project(a).xy;
    let pb = project(b).xy;
    var pc3 = project(world(mi, obj(prim, (e + 2u) % 3u)));
    if (pc3.z <= 0.0) {
        let m = 0.5 * (a + b);
        pc3 = project(m + (world(mi, obj(prim, (e + 2u) % 3u)) - m) * 1e-3);
    }
    let pc = pc3.xy;
    let t = r.x + frac * (r.y - r.x);
    let d = pb - pa;
    let sp = pa + t * d;
    let dl = length(d);
    var nrm = vec2<f32>(d.y / dl, -d.x / dl);
    if (dot(nrm, pc - pa) > 0.0) { nrm = -nrm; }
    let inner = screen(sp - ep.eps_px * nrm);
    let outer = screen(sp + ep.eps_px * nrm);
    pts_out[2u * i] = inner;
    pts_out[2u * i + 1u] = outer;
    let w = f32(params.size.x);
    let h = f32(params.size.y);
    let px = u32(clamp(floor(sp.x), 0.0, w - 1.0));
    let py = u32(clamp(floor(sp.y), 0.0, h - 1.0));
    smp_out[i * stride] = f32(py * params.size.x + px);
    // The material point: the edge line's closest approach to the ray.
    let eye = params.eye.xyz;
    let sc = screen(sp);
    let rr = params.fwd.xyz + params.right.xyz * sc.x + params.up.xyz * sc.y;
    let ev = b - a;
    let w0 = eye - a;
    let aa = dot(rr, rr);
    let bb = dot(rr, ev);
    let cc = dot(ev, ev);
    let dd = dot(rr, w0);
    let ee = dot(ev, w0);
    let den = aa * cc - bb * bb;
    var v = t;
    if (abs(den) > 0.0) { v = clamp((aa * ee - bb * dd) / den, 0.0, 1.0); }
    let x = a + ev * v;
    let xl = la + (lb - la) * v;
    // Ownership: the inner ray lands on this triangle, or a coplanar one
    // of the same face of the same instance.
    let dir = ray_dir(inner);
    let hit = intersect(eye, dir, 3.4e38, false);
    if (hit.prim == NONE || hit.inst != mv.rec) { return; }
    if (hit.prim != prim) {
        let nt = cross(lb - la, obj(prim, (e + 2u) % 3u) - la);
        let nh = cross(obj(hit.prim, 1u) - obj(hit.prim, 0u), obj(hit.prim, 2u) - obj(hit.prim, 0u));
        let ln = length(nt);
        let lh = length(nh);
        if (!(ln > 0.0 && lh > 0.0 && dot(nt, nh) / (ln * lh) > 1.0 - 1e-6)) { return; }
        let l = to_local(hit.inst, eye, dir);
        let q = l[0] + l[1] * hit.t;
        if (abs(dot(nt / ln, q - la)) >= 1e-4 * max(length(lb - la), 1e-9)) { return; }
    }
    // The pinhole's Jacobian at x.
    let dx = x - eye;
    let z = dot(dx, params.fwd.xyz);
    if (z <= 0.0) { return; }
    let kx = w / (2.0 * params.tan_half * w / h);
    let ky = h / (2.0 * params.tan_half);
    let ax = dot(dx, params.right.xyz);
    let by = dot(dx, params.up.xyz);
    let jx = (params.right.xyz * z - params.fwd.xyz * ax) * (kx / (z * z));
    let jy = (params.up.xyz * z - params.fwd.xyz * by) * (-ky / (z * z));
    smp_out[i * stride + 1u] = 1.0;
    for (var j = 0u; j < ep.np; j = j + 1u) {
        let vel = dworld(mi, j, xl);
        let vn = dot(jx, vel) * nrm.x + dot(jy, vel) * nrm.y;
        smp_out[i * stride + 2u + j] = vn * total / (r.z * f32(ep.n));
    }
}
