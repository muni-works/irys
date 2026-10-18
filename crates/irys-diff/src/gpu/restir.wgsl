// Phase 3: ReSTIR DI (Bitterli et al. 2020) at the primary vertex, in the
// generalized-RIS form of Lin et al. 2022 with pairwise MIS — unbiased.
//
// Appended to `wavefront.wgsl` (one module). Three kernels run after `gen`
// and before the first `bounce`, over one chunk that holds the whole frame:
//
//   ris_init      the primary hit's G-buffer, and M0 candidates from the
//                 base light sampler (`sample_light`) resampled to one;
//   ris_temporal  combine with the same slot's reservoir of the previous
//                 frame (its own G-buffer: a different sub-pixel point);
//   ris_spatial   combine with k neighbours' post-temporal reservoirs.
//
// A sample is a direction (the environment) or a point on an emitter with
// its normal (area measure), so reuse is the identity shift in either
// sub-domain and needs no Jacobian. Targets are unshadowed:
//
//   p̂_i(y) = Le(y, 550 nm) · f̃_i(ω) · cos_i(ω) · w_L,i(ω) · G(y)
//
// with f̃ the BSDF without Fresnel (diffuse: albedo/π), w_L the power
// heuristic of the *base* light pdf against the BSDF pdf (so the reservoir
// estimates the light strategy's share of the MIS pair, and BSDF escapes keep
// their w_B unchanged), G = cos_l / d² for emitter points and 1 otherwise.
// Every number here is detached: the reservoir hands the bounce kernel a
// sample and its unbiased contribution weight W, and the light lane's
// `s = cos · w_L · W · G` is a constant to ∂θ like the base `w_L / p_L`.
//
// Pairwise MIS (Bitterli 2022 §9.1.3; Lin et al. 2022 eq. 37): for the
// canonical domain c with confidence M_c and k neighbours q,
//
//   π_q(y) = M_q p̂_q(y) / (M_q p̂_q(y) + (M_c / k) p̂_c(y)),
//   m_q(y) = π_q(y) / k,   m_c(y) = (1/k) Σ_q (1 − π_q(y)),
//
// which sum to one for every y over the domains whose p̂ covers it; each
// candidate gets w = m(y) p̂_c(y) W, one is kept with probability ∝ w, and
// W_out = Σw / p̂_c(y_kept). Targets without visibility need no shadow rays
// in the MIS weights; neighbour rejection uses G-buffers only (never
// samples), so it changes which domains are combined and not the bias.

struct Gb {
    o: vec3<f32>,      // shadow-ray origin (the offset hit point)
    kind: u32,         // 0 nothing to shade, 1 diffuse, 2 conductor
    n: vec3<f32>,      // shading normal, towards the camera
    albedo: f32,
    wo: vec3<f32>,
    t: f32,            // primary hit distance
}

struct Res {
    y: vec3<f32>,      // direction (env) or emitter point
    w: f32,            // unbiased contribution weight W; 0 is empty
    m: f32,            // confidence
    nl: vec3<f32>,     // emitter normal; 0 for the environment
}

@group(0) @binding(41) var<storage, read_write> g_cur: array<vec4<f32>>;
@group(0) @binding(42) var<storage, read> g_prev: array<vec4<f32>>;
@group(0) @binding(43) var<storage, read_write> r_a: array<vec4<f32>>;
@group(0) @binding(44) var<storage, read_write> r_b: array<vec4<f32>>;
@group(0) @binding(45) var<storage, read> r_prev: array<vec4<f32>>;

fn gb_load_cur(i: u32) -> Gb {
    let a = g_cur[3u * i];
    let b = g_cur[3u * i + 1u];
    let c = g_cur[3u * i + 2u];
    return Gb(a.xyz, u32(a.w), b.xyz, b.w, c.xyz, c.w);
}

fn gb_load_prev(i: u32) -> Gb {
    let a = g_prev[3u * i];
    let b = g_prev[3u * i + 1u];
    let c = g_prev[3u * i + 2u];
    return Gb(a.xyz, u32(a.w), b.xyz, b.w, c.xyz, c.w);
}

fn res_pack(r: Res) -> array<vec4<f32>, 2> {
    return array<vec4<f32>, 2>(vec4<f32>(r.y, r.w), vec4<f32>(r.m, r.nl));
}

fn res_a(i: u32) -> Res {
    let a = r_a[2u * i];
    let b = r_a[2u * i + 1u];
    return Res(a.xyz, a.w, b.x, b.yzw);
}

fn res_b(i: u32) -> Res {
    let a = r_b[2u * i];
    let b = r_b[2u * i + 1u];
    return Res(a.xyz, a.w, b.x, b.yzw);
}

fn res_prev(i: u32) -> Res {
    let a = r_prev[2u * i];
    let b = r_prev[2u * i + 1u];
    return Res(a.xyz, a.w, b.x, b.yzw);
}

fn put_a(i: u32, r: Res) {
    let v = res_pack(r);
    r_a[2u * i] = v[0];
    r_a[2u * i + 1u] = v[1];
}

fn put_b(i: u32, r: Res) {
    let v = res_pack(r);
    r_b[2u * i] = v[0];
    r_b[2u * i + 1u] = v[1];
}

fn empty_res() -> Res {
    return Res(vec3<f32>(0.0), 0.0, 0.0, vec3<f32>(0.0));
}

// The BSDF sampler's pdf at the vertex, as `bounce` computes it for MIS.
fn gb_bsdf_pdf(g: Gb, dir: vec3<f32>) -> f32 {
    let ci = dot(g.n, dir);
    if (g.kind == 1u) {
        return ci / PI;
    }
    let co = dot(g.wo, g.n);
    let h = normalize(dir + g.wo);
    return ggx_pdf(params.alpha, co, dot(h, g.n));
}

// The BSDF without Fresnel: what the target ranks samples by.
fn gb_bsdf(g: Gb, dir: vec3<f32>) -> f32 {
    if (g.kind == 1u) {
        return g.albedo / PI;
    }
    let ci = dot(g.n, dir);
    let co = dot(g.wo, g.n);
    if (co <= 0.0) {
        return 0.0;
    }
    let h = normalize(dir + g.wo);
    let ch = dot(h, g.n);
    let a2 = params.alpha * params.alpha;
    let k = ch * ch * (a2 - 1.0) + 1.0;
    let d = a2 / (PI * k * k);
    let g2 = 1.0 / (1.0 + smith_lambda(a2, ci) + smith_lambda(a2, co));
    return d * g2 / (4.0 * ci * co);
}

// Direction, distance, and G (cos_l / d² for an emitter point, 1 for the
// environment) of sample `y` seen from `o`; G = 0 when the emitter's back
// faces `o`.
struct Seen {
    dir: vec3<f32>,
    dist: f32,
    geom: f32,
    pl: f32,       // the base light sampler's solid-angle pdf
}

fn seen(o: vec3<f32>, y: vec3<f32>, nl: vec3<f32>) -> Seen {
    if (all(nl == vec3<f32>(0.0))) {
        return Seen(y, 3.4e38, 1.0, light_pdf_env(y));
    }
    let v = y - o;
    let d2 = dot(v, v);
    let dist = sqrt(d2);
    let dir = v / dist;
    let cl = -dot(nl, dir);
    if (!(cl > 0.0) || !(d2 > 0.0)) {
        return Seen(dir, dist, 0.0, 0.0);
    }
    return Seen(dir, dist, cl / d2, light_pdf_emitter(dist, cl));
}

fn le_grey(y: vec3<f32>, nl: vec3<f32>) -> f32 {
    return select(params.light.w, sky(y, 550.0), all(nl == vec3<f32>(0.0)));
}

// p̂ of domain g at sample (y, nl), unshadowed, in the sample's measure.
fn target_at(g: Gb, y: vec3<f32>, nl: vec3<f32>) -> f32 {
    if (g.kind == 0u) {
        return 0.0;
    }
    let s = seen(g.o, y, nl);
    let ci = dot(g.n, s.dir);
    if (!(ci > 0.0) || !(s.geom > 0.0)) {
        return 0.0;
    }
    let pb = gb_bsdf_pdf(g, s.dir);
    let v = le_grey(y, nl) * gb_bsdf(g, s.dir) * ci * mis(s.pl, pb) * s.geom;
    return select(0.0, v, v > 0.0 && v < 3.4e38);
}

// The ReSTIR streams: keyed apart from every path's own (bit 31 / 30).
fn ris_rng(g: u32, which: u32) -> vec2<u32> {
    return rng_new(g | (0x80000000u >> which));
}

@compute @workgroup_size(64)
fn ris_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.cap) { return; }
    g_cur[3u * i] = vec4<f32>(0.0);
    put_a(i, empty_res());
    if (i >= params.paths) { return; }
    let p = paths[i];
    let o = p.o.xyz;
    let d = p.d.xyz;
    let hit = intersect(o, d, 3.4e38, false);
    if (hit.prim == NONE || prim_emitter(hit.prim)) { return; }
    let x = o + d * hit.t;
    var n = hit_normal(hit, o, d);
    if (dot(n, d) > 0.0) {
        n = -n;
    }
    let origin = offset_ray(x, n, o, hit.t);
    var kind = 2u;
    var albedo = 0.0;
    if (!prim_conductor(hit.prim)) {
        kind = 1u;
        let info = geo[params.inst_base + 4u * hit.inst + 3u];
        albedo = albedo_at(x, info.w != 0.0) * info.z;
    }
    let g = Gb(origin, kind, n, albedo, -d, hit.t);
    g_cur[3u * i] = vec4<f32>(origin, f32(kind));
    g_cur[3u * i + 1u] = vec4<f32>(n, albedo);
    g_cur[3u * i + 2u] = vec4<f32>(-d, hit.t);
    var s = ris_rng(params.base + i, 0u);
    let m0 = params.restir.y;
    var wsum = 0.0;
    var keep = empty_res();
    var keep_p = 0.0;
    for (var j = 0u; j < m0; j = j + 1u) {
        let u = vec3<f32>(rand(&s), rand(&s), rand(&s));
        let pick = rand(&s);
        let dr = sample_light(origin, u);
        if (!dr.ok) { continue; }
        var y = dr.dir;
        var nl = vec3<f32>(0.0);
        var ps = dr.pdf;
        if (dr.emitter) {
            y = origin + dr.dir * dr.dist;
            nl = dr.nl;
            ps = (1.0 - params.light.y) / params.light.z;
        }
        let ph = target_at(g, y, nl);
        let w = ph / ps;
        if (!(w > 0.0) || !(w < 3.4e38)) { continue; }
        wsum = wsum + w;
        if (pick * wsum < w) {
            keep = Res(y, 0.0, 1.0, nl);
            keep_p = ph;
        }
    }
    if (keep_p > 0.0) {
        keep.w = wsum / (f32(m0) * keep_p);
    }
    keep.m = 1.0;
    put_a(i, keep);
}

// Pairwise MIS over the canonical reservoir `c` (domain gc) and `nq`
// neighbours, resampled into one. Neighbour domains and reservoirs come in
// through the `nb_*` arrays.
const KMAX: u32 = 9u;

struct Combine {
    gq: array<Gb, KMAX>,
    rq: array<Res, KMAX>,
    n: u32,
}

fn combine(gc: Gb, c: Res, nb: ptr<function, Combine>, s: ptr<function, vec2<u32>>) -> Res {
    let k = (*nb).n;
    if (k == 0u) {
        return c;
    }
    let mc = c.m / f32(k);
    let cap = f32(params.restir.w);
    var wsum = 0.0;
    var keep = c;
    var keep_p = 0.0;
    var mtot = c.m;
    // the canonical candidate
    let pc_c = target_at(gc, c.y, c.nl);
    if (c.w > 0.0 && pc_c > 0.0) {
        var m = 0.0;
        for (var q = 0u; q < k; q = q + 1u) {
            let pq = (*nb).rq[q].m * target_at((*nb).gq[q], c.y, c.nl);
            m = m + select(1.0, mc * pc_c / (pq + mc * pc_c), pq + mc * pc_c > 0.0);
        }
        let w = m / f32(k) * pc_c * c.w;
        if (w > 0.0) {
            wsum = w;
            keep_p = pc_c;
        }
    }
    for (var q = 0u; q < k; q = q + 1u) {
        let r = (*nb).rq[q];
        mtot = mtot + r.m;
        let u = rand(s);
        if (!(r.w > 0.0)) { continue; }
        let pc = target_at(gc, r.y, r.nl);
        if (!(pc > 0.0)) { continue; }
        let pq = r.m * target_at((*nb).gq[q], r.y, r.nl);
        let pi = select(0.0, pq / (pq + mc * pc), pq + mc * pc > 0.0);
        let w = pi / f32(k) * pc * r.w;
        if (!(w > 0.0) || !(w < 3.4e38)) { continue; }
        wsum = wsum + w;
        if (u * wsum < w) {
            keep = r;
            keep_p = pc;
        }
    }
    var out = keep;
    out.w = select(0.0, wsum / keep_p, keep_p > 0.0);
    out.m = min(mtot, cap);
    return out;
}

// Whether domain q is close enough to c to be worth combining: normals
// within ~25°, primary depth within 10 %. G-buffers only.
fn similar(c: Gb, q: Gb) -> bool {
    return q.kind != 0u && dot(c.n, q.n) > 0.9 && abs(q.t - c.t) < 0.1 * c.t;
}

@compute @workgroup_size(64)
fn ris_temporal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.cap) { return; }
    let c = res_a(i);
    let gc = gb_load_cur(i);
    if (gc.kind == 0u || params.restir.z == 0u) {
        put_b(i, c);
        return;
    }
    var nb: Combine;
    nb.n = 0u;
    let gp = gb_load_prev(i);
    var rp = res_prev(i);
    if (similar(gc, gp) && rp.w > 0.0) {
        rp.m = min(rp.m, f32(params.restir.w));
        nb.gq[0] = gp;
        nb.rq[0] = rp;
        nb.n = 1u;
    }
    var s = ris_rng(params.base + i, 1u);
    put_b(i, combine(gc, c, &nb, &s));
}

@compute @workgroup_size(64)
fn ris_spatial(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.cap) { return; }
    let c = res_b(i);
    let gc = gb_load_cur(i);
    if (gc.kind == 0u) {
        put_a(i, c);
        return;
    }
    let g = params.base + i;
    let spp = params.spp;
    let px = g / spp;
    let sub = g % spp;
    let w = params.size.x;
    let h = params.size.y;
    let x0 = i32(px % w);
    let y0 = i32(px / w);
    var s = ris_rng(g, 2u);
    var nb: Combine;
    nb.n = 0u;
    let k = min(params.restir.x >> 8u, KMAX);
    let radius = f32(params.restir.x & 0xffu);
    for (var q = 0u; q < k; q = q + 1u) {
        let a = 2.0 * PI * rand(&s);
        let r = radius * sqrt(rand(&s));
        let xq = clamp(x0 + i32(round(r * cos(a))), 0, i32(w) - 1);
        let yq = clamp(y0 + i32(round(r * sin(a))), 0, i32(h) - 1);
        let gq = (u32(yq) * w + u32(xq)) * spp + sub;
        if (gq == g || gq < params.base || gq - params.base >= params.paths) { continue; }
        let j = gq - params.base;
        let gj = gb_load_cur(j);
        if (!similar(gc, gj)) { continue; }
        nb.gq[nb.n] = gj;
        nb.rq[nb.n] = res_b(j);
        nb.n = nb.n + 1u;
    }
    put_a(i, combine(gc, c, &nb, &s));
}
