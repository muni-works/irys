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

// Barycentrics of `p` in the screen triangle (a, b, c) (any winding).
fn bary_px(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> vec3<f32> {
    let ar = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    if (ar == 0.0) { return vec3<f32>(-1.0); }
    let la = ((b.x - p.x) * (c.y - p.y) - (b.y - p.y) * (c.x - p.x)) / ar;
    let lb = ((c.x - p.x) * (a.y - p.y) - (c.y - p.y) * (a.x - p.x)) / ar;
    return vec3<f32>(la, lb, 1.0 - la - lb);
}

// Per corner, its distance in pixels from the opposite edge (λ = 1 there),
// so `λ × margin` is the distance from that edge.
fn margin_px(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> vec3<f32> {
    let ar = abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
    return vec3<f32>(ar / max(length(c - b), 1e-30), ar / max(length(a - c), 1e-30), ar / max(length(b - a), 1e-30));
}

// A point of instance `k`'s space in the world: the instance record holds
// world → local rows, inverted here.
fn inst_world(k: u32, p: vec3<f32>) -> vec3<f32> {
    let r = params.inst_base + 4u * k;
    let m0 = geo[r];
    let m1 = geo[r + 1u];
    let m2 = geo[r + 2u];
    let q = p - vec3<f32>(m0.w, m1.w, m2.w);
    // Rows of A are m0, m1, m2; A⁻¹ = adj(A) / det(A), adj's columns the
    // rows' cross products.
    let c0 = cross(m1.xyz, m2.xyz);
    let c1 = cross(m2.xyz, m0.xyz);
    let c2 = cross(m0.xyz, m1.xyz);
    let det = dot(m0.xyz, c0);
    return (c0 * q.x + c1 * q.y + c2 * q.z) / det;
}

// The sign of instance `k`'s determinant (a mirror flips the winding).
fn inst_sign(k: u32) -> f32 {
    let r = params.inst_base + 4u * k;
    return sign(dot(geo[r].xyz, cross(geo[r + 1u].xyz, geo[r + 2u].xyz)));
}

// Diagnostics (`GpuEdges::debug`): why a sample was refused, and which
// (slot, t) it was, in the weight lanes (which a refused sample leaves 0).
fn refuse(i: u32, stride: u32, c: f32, si: u32, t: f32) {
    smp_out[i * stride + 1u] = c;
    if (ep._p0 != 0u && ep.np >= 2u) {
        smp_out[i * stride + 2u] = bitcast<f32>(si);
        smp_out[i * stride + 3u] = t;
    }
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
    // One rounding (2f): `u` on the same f32 grid as the cumulative sums,
    // so a slot's chance is its realised width `wt[si] − start` over the
    // total exactly (`(f32(i) + r) / n × total` put `u` on a grid ~2× as
    // coarse and not aligned with the sums, which skewed sub-0.01 px
    // slots: rose's arm read 1.6× high on roll).
    let step = total / f32(ep.n);
    let u = fma(f32(i), step, rand(&s) * step);
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
    // Ownership (2f fix): is this triangle the first surface along the
    // inner ray? At a silhouette the triangle is nearly edge-on, and an f32
    // ray-triangle test's barycentrics err by ~ulp(|o|)/sin(grazing angle):
    // the inner ray, 1e-3 px inside the edge, leaked past its own triangle
    // (or, for a sliver, past the coplanar neighbour the host lands on) to
    // the sky, another link or the wall behind — on rose, the samples with
    // the largest jumps — and the device's edge term read 30 % low. Decided
    // in screen space instead, where projecting a vertex is well
    // conditioned whatever the angle: a test point inside the triangle's
    // projection next to the sample, and nothing in front of the
    // triangle's depth there whose own projection contains it. Hits whose
    // triangle does not contain the point are the same leaks, stepped over.
    let dir = ray_dir(inner);
    // A back face (2e's 0.02-importance side of a silhouette) is hidden by
    // its front neighbour a few µm in front at the inner point — closer
    // than any depth tolerance resolves — so it keeps the ray test, whose
    // leaks there only refuse what the host would refuse anyway.
    let own2d = pc3.z > 0.0 && facing(mi, prim) > 0.0;
    // The test point: the inner point if it is inside the triangle (the
    // host's own question); if not — a sliver thinner than `eps` on screen
    // there, or a sample by a sharp corner, where the host asks a coplanar
    // neighbour — the point halfway across the triangle along the same
    // normal, which is inside it and as close to the sample.
    let l0 = vec2<f32>(1.0 - t, t);
    let l1 = bary_px(sp - nrm, pa, pb, pc).xy;
    let dlam = l1 - l0;
    var wpx = 3.4e38;
    if (dlam.x < 0.0) { wpx = min(wpx, l0.x / -dlam.x); }
    if (dlam.y < 0.0) { wpx = min(wpx, l0.y / -dlam.y); }
    let ip = sp - min(ep.eps_px, 0.5 * wpx) * nrm;
    let idir = ray_dir(screen(ip));
    if (own2d) {
        // λ is in the edge's frame (a = corner e); depth by 1/z.
        let bary = bary_px(ip, pa, pb, pc);
        let za = project(a).z;
        let zb = project(b).z;
        let zt = 1.0 / (bary.x / za + bary.y / zb + bary.z / pc3.z);
        let tmax = zt * (1.0 - 1e-6) / dot(idir, params.fwd.xyz);
        var o = eye;
        var left = tmax;
        for (var k = 0u; k < 8u; k = k + 1u) {
            let hh = intersect(o, idir, left, false);
            if (hh.prim == NONE) { break; }
            var real = true;
            if (hh.inst == mv.rec && hh.prim == prim) {
                real = false;
            } else if (u32(geo[3u * hh.prim].w) == 0u) {
                let wa = inst_world(hh.inst, obj(hh.prim, 0u));
                let wb = inst_world(hh.inst, obj(hh.prim, 1u));
                let wc = inst_world(hh.inst, obj(hh.prim, 2u));
                let qa = project(wa);
                let qb = project(wb);
                let qc = project(wc);
                // A back face never hides anything on a closed solid (the
                // solid's front face is in front of it); the one just
                // behind a sharp silhouette is within any depth tolerance.
                let front = dot(cross(wb - wa, wc - wa), eye - wa) * inst_sign(hh.inst) > 0.0;
                if (!front) {
                    real = false;
                } else if (qa.z > 0.0 && qb.z > 0.0 && qc.z > 0.0) {
                    let l2 = bary_px(ip, qa.xy, qb.xy, qc.xy);
                    let m2 = margin_px(qa.xy, qb.xy, qc.xy);
                    let mm = ep.eps_px * 0.25;
                    real = l2.x * m2.x >= mm && l2.y * m2.y >= mm && l2.z * m2.z >= mm;
                }
            }
            if (real) { refuse(i, stride, -6.0, si, t); return; }
            let step = hh.t * (1.0 + 1e-6);
            o = o + idir * step;
            left = left - step;
            if (left <= 0.0) { break; }
        }
    } else {
        let hit = intersect(eye, dir, 3.4e38, false);
        if (hit.prim == NONE) { refuse(i, stride, -1.0, si, t); return; }
        if (hit.inst != mv.rec) { refuse(i, stride, -2.0, si, t); return; }
        if (hit.prim != prim) {
            let nt = cross(lb - la, obj(prim, (e + 2u) % 3u) - la);
            let nh = cross(obj(hit.prim, 1u) - obj(hit.prim, 0u), obj(hit.prim, 2u) - obj(hit.prim, 0u));
            let ln = length(nt);
            let lh = length(nh);
            // f32 (2f): a sliver's cross product carries its corners' rounding
            // (~1e-7 of their magnitude over its own size), so the host's
            // 1e-6 on the cosine and 1e-4 of the edge on the plane distance
            // refused real coplanar neighbours on rose's mm-scale CAD — 7 % of
            // its visible samples, a quarter of the shoulder pitch's mask
            // gradient. Tolerances in units of the coordinates' own rounding.
            let mag = max(max(abs(la.x), abs(la.y)), max(abs(la.z), 1e-9));
            let cos_tol = max(1e-6, 64.0 * 1.2e-7 * mag * mag / max(ln * lh, 1e-30) * 4.0);
            if (!(ln > 0.0 && lh > 0.0 && dot(nt, nh) / (ln * lh) > 1.0 - min(cos_tol, 1e-3))) {
                refuse(i, stride, -3.0, si, t);
                return;
            }
            let l = to_local(hit.inst, eye, dir);
            let q = l[0] + l[1] * hit.t;
            let dtol = max(1e-4 * max(length(lb - la), 1e-9), 64.0 * 1.2e-7 * (mag + length(q)));
            if (abs(dot(nt / ln, q - la)) >= dtol) { refuse(i, stride, -4.0, si, t); return; }
        }
    }
    // The pinhole's Jacobian at x.
    let dx = x - eye;
    let z = dot(dx, params.fwd.xyz);
    if (z <= 0.0) { refuse(i, stride, -5.0, si, t); return; }
    let kx = w / (2.0 * params.tan_half * w / h);
    let ky = h / (2.0 * params.tan_half);
    let ax = dot(dx, params.right.xyz);
    let by = dot(dx, params.up.xyz);
    let jx = (params.right.xyz * z - params.fwd.xyz * ax) * (kx / (z * z));
    let jy = (params.up.xyz * z - params.fwd.xyz * by) * (-ky / (z * z));
    smp_out[i * stride + 1u] = 1.0;
    if (ep._p0 != 0u && ep.np >= 2u) {
        smp_out[i * stride + 2u] = bitcast<f32>(si);
        smp_out[i * stride + 3u] = t;
        return;
    }
    // The density per pixel of edge is the slot's realised width (f32
    // sums: not exactly `len × importance` for a slot much narrower than
    // the sums' rounding) over its length, over the total.
    let wreal = wt[si] - start;
    let lslot = dl * (r.y - r.x);
    for (var j = 0u; j < ep.np; j = j + 1u) {
        let vel = dworld(mi, j, xl);
        let vn = dot(jx, vel) * nrm.x + dot(jy, vel) * nrm.y;
        smp_out[i * stride + 2u + j] = vn * total * lslot / (wreal * f32(ep.n));
    }
}
