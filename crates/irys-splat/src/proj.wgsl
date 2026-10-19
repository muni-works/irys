// irys-splat: per-splat projection, its adjoint fused with Adam, and the
// loss (L1 + D-SSIM + depth L1) with its gradient — the device ports of
// project.rs and loss.rs, f32. Appended to raster.wgsl (one module, one
// bind group).
//
// Parameters: 64 f32 per splat: mean [0, 3), log scale [3, 6), quaternion
// [6, 10) (w x y z, unnormalised), opacity logit 10, SH [12, 60) as
// 3 k + channel; 11 and 60..63 are padding (never updated).

@group(0) @binding(11) var<storage, read_write> prm: array<f32>;
@group(0) @binding(12) var<storage, read_write> adm: array<f32>;
@group(0) @binding(13) var<storage, read_write> adv: array<f32>;
@group(0) @binding(14) var<storage, read_write> keys: array<vec4<f32>>;   // (u, v, radius, z); radius 0: culled
@group(0) @binding(15) var<storage, read_write> dstat: array<vec2<f32>>;  // Σ |∂L/∂uv|_ndc, views
@group(0) @binding(16) var<storage, read_write> gt: array<vec4<f32>>;     // sRGB, depth (≤ 0 invalid)
@group(0) @binding(17) var<storage, read_write> sa: array<f32>;           // 15 per pixel
@group(0) @binding(18) var<storage, read_write> sb: array<f32>;
@group(0) @binding(19) var<storage, read_write> gdbg: array<f32>;         // 64 per splat: the raw gradient

const NP: u32 = 64u;
const DIL: f32 = 0.3;

struct F {
    ok: bool,
    pc: vec3<f32>,
    clampx: f32,
    clampy: f32,
    t: vec2<f32>,
    qn: vec4<f32>,
    qlen: f32,
    r: array<vec3<f32>, 3>,
    s: vec3<f32>,
    m: array<vec3<f32>, 3>,
    sig: array<vec3<f32>, 3>,
    tm0: vec3<f32>,
    tm1: vec3<f32>,
    s2: vec3<f32>,
    det: f32,
    o: f32,
    dir: vec3<f32>,
    vlen: f32,
    raw: vec3<f32>,
    uv: vec2<f32>,
    conic: vec3<f32>,
    radius: f32,
}

fn wrow(k: u32) -> vec3<f32> {
    // world → camera row k = column k of the camera → world rotation
    return vec3<f32>(P.rot0[k], P.rot1[k], P.rot2[k]);
}

fn sh_basis(d: vec3<f32>, deg: u32, b: ptr<function, array<f32, 16>>, g: ptr<function, array<vec3<f32>, 16>>) {
    let x = d.x;
    let y = d.y;
    let z = d.z;
    for (var k = 0u; k < 16u; k = k + 1u) {
        (*b)[k] = 0.0;
        (*g)[k] = vec3<f32>(0.0);
    }
    (*b)[0] = 0.28209479177387814;
    if (deg >= 1u) {
        let c1 = 0.4886025119029199;
        (*b)[1] = -c1 * y; (*g)[1] = vec3<f32>(0.0, -c1, 0.0);
        (*b)[2] = c1 * z;  (*g)[2] = vec3<f32>(0.0, 0.0, c1);
        (*b)[3] = -c1 * x; (*g)[3] = vec3<f32>(-c1, 0.0, 0.0);
    }
    if (deg >= 2u) {
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let a0 = 1.0925484305920792;
        let a1 = -1.0925484305920792;
        let a2 = 0.31539156525252005;
        let a3 = -1.0925484305920792;
        let a4 = 0.5462742152960396;
        (*b)[4] = a0 * x * y; (*g)[4] = vec3<f32>(a0 * y, a0 * x, 0.0);
        (*b)[5] = a1 * y * z; (*g)[5] = vec3<f32>(0.0, a1 * z, a1 * y);
        (*b)[6] = a2 * (2.0 * zz - xx - yy); (*g)[6] = vec3<f32>(-2.0 * a2 * x, -2.0 * a2 * y, 4.0 * a2 * z);
        (*b)[7] = a3 * x * z; (*g)[7] = vec3<f32>(a3 * z, 0.0, a3 * x);
        (*b)[8] = a4 * (xx - yy); (*g)[8] = vec3<f32>(2.0 * a4 * x, -2.0 * a4 * y, 0.0);
        if (deg >= 3u) {
            let c0 = -0.5900435899266435;
            let c1 = 2.890611442640554;
            let c2 = -0.4570457994644658;
            let c3 = 0.3731763325901154;
            let c4 = -0.4570457994644658;
            let c5 = 1.445305721320277;
            let c6 = -0.5900435899266435;
            (*b)[9] = c0 * y * (3.0 * xx - yy); (*g)[9] = vec3<f32>(c0 * 6.0 * x * y, c0 * (3.0 * xx - 3.0 * yy), 0.0);
            (*b)[10] = c1 * x * y * z; (*g)[10] = vec3<f32>(c1 * y * z, c1 * x * z, c1 * x * y);
            (*b)[11] = c2 * y * (4.0 * zz - xx - yy); (*g)[11] = vec3<f32>(c2 * -2.0 * x * y, c2 * (4.0 * zz - xx - 3.0 * yy), c2 * 8.0 * y * z);
            (*b)[12] = c3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy); (*g)[12] = vec3<f32>(c3 * -6.0 * x * z, c3 * -6.0 * y * z, c3 * (6.0 * zz - 3.0 * xx - 3.0 * yy));
            (*b)[13] = c4 * x * (4.0 * zz - xx - yy); (*g)[13] = vec3<f32>(c4 * (4.0 * zz - 3.0 * xx - yy), c4 * -2.0 * x * y, c4 * 8.0 * x * z);
            (*b)[14] = c5 * z * (xx - yy); (*g)[14] = vec3<f32>(c5 * 2.0 * x * z, c5 * -2.0 * y * z, c5 * (xx - yy));
            (*b)[15] = c6 * x * (xx - 3.0 * yy); (*g)[15] = vec3<f32>(c6 * (3.0 * xx - 3.0 * yy), c6 * -6.0 * x * y, 0.0);
        }
    }
}

fn shc(i: u32, k: u32) -> vec3<f32> {
    let b = NP * i + 12u + 3u * k;
    return vec3<f32>(prm[b], prm[b + 1u], prm[b + 2u]);
}

fn n_coef(deg: u32) -> u32 {
    return (deg + 1u) * (deg + 1u);
}

fn fwd(i: u32, basis: ptr<function, array<f32, 16>>, dbasis: ptr<function, array<vec3<f32>, 16>>) -> F {
    var f: F;
    f.ok = false;
    let b = NP * i;
    let mu = vec3<f32>(prm[b], prm[b + 1u], prm[b + 2u]);
    let pos = vec3<f32>(P.rot0.w, P.rot1.w, P.rot2.w);
    let v = mu - pos;
    let w0 = wrow(0u);
    let w1 = wrow(1u);
    let w2 = wrow(2u);
    let pc = vec3<f32>(dot(w0, v), dot(w1, v), dot(w2, v));
    f.pc = pc;
    let z = pc.z;
    if (!(z > P.misc.x)) { return f; }
    let rx = pc.x / z;
    let rxc = clamp(rx, P.lim.x, P.lim.y);
    f.clampx = select(0.0, rxc, rxc != rx);
    let ry = pc.y / z;
    let ryc = clamp(ry, P.lim.z, P.lim.w);
    f.clampy = select(0.0, ryc, ryc != ry);
    f.t = vec2<f32>(rxc * z, ryc * z);
    let fx = P.intr.x;
    let fy = P.intr.y;
    let j0 = vec3<f32>(fx / z, 0.0, -fx * f.t.x / (z * z));
    let j1 = vec3<f32>(0.0, fy / z, -fy * f.t.y / (z * z));
    let q = vec4<f32>(prm[b + 6u], prm[b + 7u], prm[b + 8u], prm[b + 9u]);
    f.qlen = max(length(q), 1e-12);
    let qn = q / f.qlen;
    f.qn = qn;
    let qw = qn.x;
    let qx = qn.y;
    let qy = qn.z;
    let qz = qn.w;
    f.r[0] = vec3<f32>(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy));
    f.r[1] = vec3<f32>(2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx));
    f.r[2] = vec3<f32>(2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy));
    f.s = exp(vec3<f32>(prm[b + 3u], prm[b + 4u], prm[b + 5u]));
    for (var a = 0u; a < 3u; a = a + 1u) { f.m[a] = f.r[a] * f.s; }
    for (var a = 0u; a < 3u; a = a + 1u) {
        f.sig[a] = vec3<f32>(dot(f.m[a], f.m[0]), dot(f.m[a], f.m[1]), dot(f.m[a], f.m[2]));
    }
    f.tm0 = j0.x * w0 + j0.y * w1 + j0.z * w2;
    f.tm1 = j1.x * w0 + j1.y * w1 + j1.z * w2;
    let st0 = vec3<f32>(dot(f.sig[0], f.tm0), dot(f.sig[1], f.tm0), dot(f.sig[2], f.tm0));
    let st1 = vec3<f32>(dot(f.sig[0], f.tm1), dot(f.sig[1], f.tm1), dot(f.sig[2], f.tm1));
    f.s2 = vec3<f32>(dot(f.tm0, st0) + DIL, dot(f.tm0, st1), dot(f.tm1, st1) + DIL);
    f.det = f.s2.x * f.s2.z - f.s2.y * f.s2.y;
    if (!(f.det > 0.0)) { return f; }
    f.conic = vec3<f32>(f.s2.z / f.det, -f.s2.y / f.det, f.s2.x / f.det);
    f.o = 1.0 / (1.0 + exp(-prm[b + 10u]));
    if (f.o * 255.0 <= 1.0) { return f; }
    let mid = 0.5 * (f.s2.x + f.s2.z);
    let lmax = mid + sqrt(max(mid * mid - f.det, 0.1));
    f.radius = sqrt(2.0 * log(255.0 * f.o) * lmax);
    f.uv = vec2<f32>(fx * pc.x / z + P.intr.z, fy * pc.y / z + P.intr.w);
    let wd = f32(P.size.x);
    let ht = f32(P.size.y);
    if (f.uv.x + f.radius < -0.5 || f.uv.x - f.radius > wd - 0.5 || f.uv.y + f.radius < -0.5 || f.uv.y - f.radius > ht - 0.5) {
        return f;
    }
    f.vlen = max(length(v), 1e-12);
    f.dir = v / f.vlen;
    let deg = u32(P.misc.y);
    sh_basis(f.dir, deg, basis, dbasis);
    var raw = vec3<f32>(0.5);
    for (var k = 0u; k < n_coef(deg); k = k + 1u) { raw = raw + (*basis)[k] * shc(i, k); }
    f.raw = raw;
    f.ok = true;
    return f;
}

@compute @workgroup_size(256, 1, 1)
fn project(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = gid.x + gid.y * nw.x * 256u;
    if (i >= P.size.w) { return; }
    var basis: array<f32, 16>;
    var dbasis: array<vec3<f32>, 16>;
    let f = fwd(i, &basis, &dbasis);
    if (!f.ok) {
        keys[i] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        proj[3u * i] = vec4<f32>(0.0);
        proj[3u * i + 1u] = vec4<f32>(0.0);
        proj[3u * i + 2u] = vec4<f32>(0.0);
        return;
    }
    let col = max(f.raw, vec3<f32>(0.0));
    keys[i] = vec4<f32>(f.uv, f.radius, f.pc.z);
    proj[3u * i] = vec4<f32>(f.uv, f.conic.x, f.conic.y);
    proj[3u * i + 1u] = vec4<f32>(f.conic.z, f.o, col.x, col.y);
    proj[3u * i + 2u] = vec4<f32>(col.z, f.pc.z, 0.0, 0.0);
}

fn lr_of(k: u32) -> f32 {
    if (k < 3u) { return P.lr0.x; }
    if (k < 6u) { return P.lr0.y; }
    if (k < 10u) { return P.lr0.z; }
    if (k == 10u) { return P.lr0.w; }
    if (k < 12u) { return 0.0; }
    if (k < 15u) { return P.lr1.x; }
    if (k < 60u) { return P.lr1.y; }
    return 0.0;
}

// The adjoint of `project` from the reduced screen-space gradient g2, then
// Adam on all 60 live parameters (a zero gradient still moves the moments,
// as torch's dense Adam does), and the densification statistics.
@compute @workgroup_size(256, 1, 1)
fn adjoint(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = gid.x + gid.y * nw.x * 256u;
    if (i >= P.size.w) { return; }
    var basis: array<f32, 16>;
    var dbasis: array<vec3<f32>, 16>;
    let f = fwd(i, &basis, &dbasis);
    var gr: array<f32, 64>;
    for (var k = 0u; k < 64u; k = k + 1u) { gr[k] = 0.0; }
    let ga = g2[3u * i];
    let gb = g2[3u * i + 1u];
    let gc = g2[3u * i + 2u];
    if (f.ok) {
        let fx = P.intr.x;
        let fy = P.intr.y;
        let z = f.pc.z;
        let dcol = vec3<f32>(gb.z, gb.w, gc.x);
        let deg = u32(P.misc.y);
        var ddir = vec3<f32>(0.0);
        for (var ch = 0u; ch < 3u; ch = ch + 1u) {
            if (f.raw[ch] <= 0.0) { continue; }
            let dc = dcol[ch];
            for (var k = 0u; k < n_coef(deg); k = k + 1u) {
                gr[12u + 3u * k + ch] = basis[k] * dc;
                ddir = ddir + dbasis[k] * (shc(i, k)[ch] * dc);
            }
        }
        let dd = dot(ddir, f.dir);
        var dmu = (ddir - f.dir * dd) / f.vlen;
        var dpc = vec3<f32>(0.0, 0.0, gc.y);
        gr[10] = gb.y * f.o * (1.0 - f.o);
        dpc.x = dpc.x + ga.x * fx / z;
        dpc.y = dpc.y + ga.y * fy / z;
        dpc.z = dpc.z - (ga.x * fx * f.pc.x / (z * z) + ga.y * fy * f.pc.y / (z * z));
        // conic = Σ₂⁻¹
        let k00 = f.s2.z / f.det;
        let k01 = -f.s2.y / f.det;
        let k11 = f.s2.x / f.det;
        let gk00 = ga.z;
        let gk01 = 0.5 * ga.w;
        let gk11 = gb.x;
        // KG
        let a00 = k00 * gk00 + k01 * gk01;
        let a01 = k00 * gk01 + k01 * gk11;
        let a10 = k01 * gk00 + k11 * gk01;
        let a11 = k01 * gk01 + k11 * gk11;
        let g00 = -(a00 * k00 + a01 * k01);
        let g01 = -(a00 * k01 + a01 * k11);
        let g10 = -(a10 * k00 + a11 * k01);
        let g11 = -(a10 * k01 + a11 * k11);
        let tm0 = f.tm0;
        let tm1 = f.tm1;
        var dsig: array<vec3<f32>, 3>;
        for (var a = 0u; a < 3u; a = a + 1u) {
            dsig[a] = tm0[a] * (g00 * tm0 + g01 * tm1) + tm1[a] * (g10 * tm0 + g11 * tm1);
        }
        let ts0 = vec3<f32>(dot(f.sig[0], tm0), dot(f.sig[1], tm0), dot(f.sig[2], tm0));
        let ts1 = vec3<f32>(dot(f.sig[0], tm1), dot(f.sig[1], tm1), dot(f.sig[2], tm1));
        let dt0 = 2.0 * (g00 * ts0 + g01 * ts1);
        let dt1 = 2.0 * (g10 * ts0 + g11 * ts1);
        let w0 = wrow(0u);
        let w1 = wrow(1u);
        let w2 = wrow(2u);
        let dj0 = vec3<f32>(dot(dt0, w0), dot(dt0, w1), dot(dt0, w2));
        let dj1 = vec3<f32>(dot(dt1, w0), dot(dt1, w1), dot(dt1, w2));
        let z2 = z * z;
        let z3 = z2 * z;
        dpc.z = dpc.z + dj0.x * (-fx / z2) + dj0.z * (2.0 * fx * f.t.x / z3) + dj1.y * (-fy / z2) + dj1.z * (2.0 * fy * f.t.y / z3);
        let dtx = dj0.z * (-fx / z2);
        let dty = dj1.z * (-fy / z2);
        if (f.clampx == 0.0) { dpc.x = dpc.x + dtx; } else { dpc.z = dpc.z + dtx * f.clampx; }
        if (f.clampy == 0.0) { dpc.y = dpc.y + dty; } else { dpc.z = dpc.z + dty * f.clampy; }
        dmu = dmu + w0 * dpc.x + w1 * dpc.y + w2 * dpc.z;
        gr[0] = dmu.x;
        gr[1] = dmu.y;
        gr[2] = dmu.z;
        // Σ = M Mᵀ, M = R S
        var dr: array<vec3<f32>, 3>;
        var ds = vec3<f32>(0.0);
        for (var a = 0u; a < 3u; a = a + 1u) {
            let dm = 2.0 * (dsig[a].x * f.m[0] + dsig[a].y * f.m[1] + dsig[a].z * f.m[2]);
            dr[a] = dm * f.s;
            ds = ds + dm * f.r[a];
        }
        gr[3] = ds.x * f.s.x;
        gr[4] = ds.y * f.s.y;
        gr[5] = ds.z * f.s.z;
        let w = f.qn.x;
        let x = f.qn.y;
        let y = f.qn.z;
        let zq = f.qn.w;
        let dqn = vec4<f32>(
            2.0 * (-zq * dr[0].y + y * dr[0].z + zq * dr[1].x - x * dr[1].z - y * dr[2].x + x * dr[2].y),
            2.0 * (y * dr[0].y + zq * dr[0].z + y * dr[1].x - 2.0 * x * dr[1].y - w * dr[1].z + zq * dr[2].x + w * dr[2].y - 2.0 * x * dr[2].z),
            2.0 * (-2.0 * y * dr[0].x + x * dr[0].y + w * dr[0].z + x * dr[1].x + zq * dr[1].z - w * dr[2].x + zq * dr[2].y - 2.0 * y * dr[2].z),
            2.0 * (-2.0 * zq * dr[0].x - w * dr[0].y + x * dr[0].z + w * dr[1].x - 2.0 * zq * dr[1].y + y * dr[1].z + x * dr[2].x + y * dr[2].y));
        let dq = (dqn - f.qn * dot(dqn, f.qn)) / f.qlen;
        gr[6] = dq.x;
        gr[7] = dq.y;
        gr[8] = dq.z;
        gr[9] = dq.w;
        if (csr_off[i + 1u] > csr_off[i]) {
            let g = vec2<f32>(ga.x * f32(P.size.x) * 0.5, ga.y * f32(P.size.y) * 0.5);
            dstat[i] = dstat[i] + vec2<f32>(length(g), 1.0);
        }
    }
    let b = NP * i;
    for (var k = 0u; k < 60u; k = k + 1u) {
        gdbg[b + k] = gr[k];
        let lr = lr_of(k);
        if (lr == 0.0) { continue; }
        let m = 0.9 * adm[b + k] + 0.1 * gr[k];
        let v = 0.999 * adv[b + k] + 0.001 * gr[k] * gr[k];
        adm[b + k] = m;
        adv[b + k] = v;
        prm[b + k] = prm[b + k] - lr * (m / P.lr1.z) / (sqrt(v / P.lr1.w) + 1e-15);
    }
}

// ---- the loss -------------------------------------------------------------

fn win(j: u32) -> f32 {
    let w = array<f32, 12>(P.win0.x, P.win0.y, P.win0.z, P.win0.w, P.win1.x, P.win1.y, P.win1.z, P.win1.w, P.win2.x, P.win2.y, P.win2.z, P.win2.w);
    return w[j];
}

const C1: f32 = 0.0001;
const C2: f32 = 0.0009;

// a, b, a², b², ab per channel, blurred along x → sa
@compute @workgroup_size(16, 16, 1)
fn ssim_h1(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = P.size.x;
    let h = P.size.y;
    if (gid.x >= w || gid.y >= h) { return; }
    var acc: array<f32, 15>;
    for (var k = 0u; k < 15u; k = k + 1u) { acc[k] = 0.0; }
    for (var j = 0u; j < 11u; j = j + 1u) {
        let xx = i32(gid.x) + i32(j) - 5;
        if (xx < 0 || xx >= i32(w)) { continue; }
        let p = gid.y * w + u32(xx);
        let a = out_c[p];
        let b = gt[p];
        let wj = win(j);
        for (var ch = 0u; ch < 3u; ch = ch + 1u) {
            let av = a[ch];
            let bv = b[ch];
            acc[5u * ch] += wj * av;
            acc[5u * ch + 1u] += wj * bv;
            acc[5u * ch + 2u] += wj * av * av;
            acc[5u * ch + 3u] += wj * bv * bv;
            acc[5u * ch + 4u] += wj * av * bv;
        }
    }
    let o = 15u * (gid.y * w + gid.x);
    for (var k = 0u; k < 15u; k = k + 1u) { sa[o + k] = acc[k]; }
}

// blur along y: src → dst, `n` maps of the 15
fn vblur(gid: vec3<u32>, n: u32, from_a: bool) {
    let w = P.size.x;
    let h = P.size.y;
    if (gid.x >= w || gid.y >= h) { return; }
    var acc: array<f32, 15>;
    for (var k = 0u; k < 15u; k = k + 1u) { acc[k] = 0.0; }
    for (var j = 0u; j < 11u; j = j + 1u) {
        let yy = i32(gid.y) + i32(j) - 5;
        if (yy < 0 || yy >= i32(h)) { continue; }
        let o = 15u * (u32(yy) * w + gid.x);
        let wj = win(j);
        for (var k = 0u; k < n; k = k + 1u) {
            acc[k] += wj * select(sb[o + k], sa[o + k], from_a);
        }
    }
    let o = 15u * (gid.y * w + gid.x);
    for (var k = 0u; k < n; k = k + 1u) {
        if (from_a) { sb[o + k] = acc[k]; } else { sa[o + k] = acc[k]; }
    }
}

@compute @workgroup_size(16, 16, 1)
fn ssim_v1(@builtin(global_invocation_id) gid: vec3<u32>) {
    vblur(gid, 15u, true);
}

// ∂(mean SSIM)/∂(μ₁, σ₁₁, σ₁₂) per pixel, scaled by −λ / (3N) → sa (9)
@compute @workgroup_size(16, 16, 1)
fn ssim_g(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = P.size.x;
    if (gid.x >= w || gid.y >= P.size.y) { return; }
    let o = 15u * (gid.y * w + gid.x);
    let scale = -P.bg.w * P.misc.w;
    for (var ch = 0u; ch < 3u; ch = ch + 1u) {
        let m1 = sb[o + 5u * ch];
        let m2 = sb[o + 5u * ch + 1u];
        let s11 = sb[o + 5u * ch + 2u] - m1 * m1;
        let s22 = sb[o + 5u * ch + 3u] - m2 * m2;
        let s12 = sb[o + 5u * ch + 4u] - m1 * m2;
        let n1 = 2.0 * m1 * m2 + C1;
        let n2 = 2.0 * s12 + C2;
        let d1 = m1 * m1 + m2 * m2 + C1;
        let d2 = s11 + s22 + C2;
        let s = n1 * n2 / (d1 * d2);
        let g11 = scale * (-s / d2);
        let g12 = scale * (2.0 * n1 / (d1 * d2));
        let gmu = scale * (2.0 * m2 * n2 / (d1 * d2) - s * 2.0 * m1 / d1) - 2.0 * m1 * g11 - m2 * g12;
        sa[o + 3u * ch] = gmu;
        sa[o + 3u * ch + 1u] = g11;
        sa[o + 3u * ch + 2u] = g12;
    }
}

@compute @workgroup_size(16, 16, 1)
fn ssim_h2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = P.size.x;
    if (gid.x >= w || gid.y >= P.size.y) { return; }
    var acc: array<f32, 9>;
    for (var k = 0u; k < 9u; k = k + 1u) { acc[k] = 0.0; }
    for (var j = 0u; j < 11u; j = j + 1u) {
        let xx = i32(gid.x) + i32(j) - 5;
        if (xx < 0 || xx >= i32(w)) { continue; }
        let o = 15u * (gid.y * w + u32(xx));
        let wj = win(j);
        for (var k = 0u; k < 9u; k = k + 1u) { acc[k] += wj * sa[o + k]; }
    }
    let o = 15u * (gid.y * w + gid.x);
    for (var k = 0u; k < 9u; k = k + 1u) { sb[o + k] = acc[k]; }
}

// the last blur, then ∂loss/∂(rgb, depth) per pixel → dl
@compute @workgroup_size(16, 16, 1)
fn ssim_v2(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = P.size.x;
    let h = P.size.y;
    if (gid.x >= w || gid.y >= h) { return; }
    var acc: array<f32, 9>;
    for (var k = 0u; k < 9u; k = k + 1u) { acc[k] = 0.0; }
    for (var j = 0u; j < 11u; j = j + 1u) {
        let yy = i32(gid.y) + i32(j) - 5;
        if (yy < 0 || yy >= i32(h)) { continue; }
        let o = 15u * (u32(yy) * w + gid.x);
        let wj = win(j);
        for (var k = 0u; k < 9u; k = k + 1u) { acc[k] += wj * sb[o + k]; }
    }
    let p = gid.y * w + gid.x;
    let a = out_c[p];
    let b = gt[p];
    let l1 = (1.0 - P.bg.w) * P.misc.w;
    var g = vec4<f32>(0.0);
    for (var ch = 0u; ch < 3u; ch = ch + 1u) {
        g[ch] = acc[3u * ch] + 2.0 * a[ch] * acc[3u * ch + 1u] + b[ch] * acc[3u * ch + 2u] + l1 * sign(a[ch] - b[ch]);
    }
    if (b.w > 0.0 && P.misc.z > 0.0) { g.w = P.misc.z * sign(a.w - b.w); }
    dl[p] = g;
}
