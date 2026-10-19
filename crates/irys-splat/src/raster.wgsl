// irys-splat: the tile rasteriser and its adjoint (3DGS / gsplat's model).
//
// One workgroup per 16 × 16 tile, one thread per pixel, the tile's splats
// in (depth, index) order staged through workgroup memory 256 at a time.
// Pixel (i, j)'s centre is (i, j) (OpenCV, look::pinhole).
//
// forward:  C = Σ αᵢ Tᵢ cᵢ + T_N bg (rgb, plus the optical depth as a 4th
//           channel), α = min(0.99, o·exp(power)), skipped below 1/255,
//           stopped before T would fall under 1e-4. Writes C, T_N and the
//           count of list entries consumed (`last`).
// backward: the same list back to front from T_N: Tᵢ = Tᵢ₊₁ / (1 − αᵢ),
//           ∂C/∂αᵢ = Tᵢ (cᵢ − Sᵢ) with S the colour behind (S_N = bg).
//           Each splat's per-pixel gradient is summed over the tile by a
//           fixed tree in workgroup memory and written to its slot — one
//           record per (tile, splat), no float atomics.
// reduce:   per splat, its slots in the host's fixed (tile) order.
//
// raster_cpu.rs is the f64 twin of all three.

struct U {
    size: vec4<u32>,   // w, h, tiles_x, splats
    bg: vec4<f32>,
}

@group(0) @binding(0) var<uniform> P: U;
@group(0) @binding(1) var<storage, read> proj: array<vec4<f32>>;        // 3 per splat: (u, v, A, B) (C, o, r, g) (b, z, -, -)
@group(0) @binding(2) var<storage, read> ranges: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_c: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> out_t: array<vec2<u32>>;  // (bits of T_N, last)
@group(0) @binding(6) var<storage, read> dl: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> slot_g: array<vec4<f32>>; // 3 per slot
@group(0) @binding(8) var<storage, read> csr_off: array<u32>;
@group(0) @binding(9) var<storage, read> csr_slots: array<u32>;
@group(0) @binding(10) var<storage, read_write> g2: array<vec4<f32>>;    // 3 per splat

const B: u32 = 256u;
const ALPHA_MIN: f32 = 0.003921569;  // 1/255
const ALPHA_MAX: f32 = 0.99;
const T_MIN: f32 = 1e-4;

var<workgroup> s_a: array<vec4<f32>, 256>;
var<workgroup> s_b: array<vec4<f32>, 256>;
var<workgroup> s_c: array<vec4<f32>, 256>;
var<workgroup> r_a: array<vec4<f32>, 256>;
var<workgroup> r_b: array<vec4<f32>, 256>;
var<workgroup> r_c: array<vec4<f32>, 256>;
var<workgroup> n_done: atomic<u32>;
var<workgroup> all_done: u32;

// (α, G, dx, dy); α = 0: no response. Shared by both passes so they agree.
fn response(a: vec4<f32>, b: vec4<f32>, px: f32, py: f32) -> vec4<f32> {
    let dx = px - a.x;
    let dy = py - a.y;
    let power = -0.5 * (a.z * dx * dx + b.x * dy * dy) - a.w * dx * dy;
    if (power > 0.0) { return vec4<f32>(0.0); }
    let g = exp(power);
    let al = min(ALPHA_MAX, b.y * g);
    if (al < ALPHA_MIN) { return vec4<f32>(0.0); }
    return vec4<f32>(al, g, dx, dy);
}

@compute @workgroup_size(16, 16, 1)
fn forward(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(local_invocation_index) li: u32) {
    let tile = wg.y * P.size.z + wg.x;
    let pxu = wg.x * 16u + lid.x;
    let pyu = wg.y * 16u + lid.y;
    let inside = pxu < P.size.x && pyu < P.size.y;
    let px = f32(pxu);
    let py = f32(pyu);
    let rg = ranges[tile];
    var t = 1.0;
    var c = vec4<f32>(0.0);
    var last = 0u;
    var done = !inside;
    if (li == 0u) { atomicStore(&n_done, 0u); }
    workgroupBarrier();
    if (done) { atomicAdd(&n_done, 1u); }
    let n = rg.y - rg.x;
    let rounds = (n + B - 1u) / B;
    for (var r = 0u; r < rounds; r = r + 1u) {
        workgroupBarrier();
        if (li == 0u) { all_done = atomicLoad(&n_done); }
        let k = rg.x + r * B + li;
        if (k < rg.y) {
            let i = ids[k];
            s_a[li] = proj[3u * i];
            s_b[li] = proj[3u * i + 1u];
            s_c[li] = proj[3u * i + 2u];
        }
        if (workgroupUniformLoad(&all_done) == 256u) { break; }
        let cnt = min(B, rg.y - rg.x - r * B);
        for (var j = 0u; j < cnt; j = j + 1u) {
            if (done) { break; }
            let a = s_a[j];
            let b = s_b[j];
            let e = response(a, b, px, py);
            if (e.x <= 0.0) { continue; }
            let nt = t * (1.0 - e.x);
            if (nt < T_MIN) {
                done = true;
                atomicAdd(&n_done, 1u);
                break;
            }
            let cc = s_c[j];
            c = c + (e.x * t) * vec4<f32>(b.z, b.w, cc.x, cc.y);
            t = nt;
            last = r * B + j + 1u;
        }
    }
    if (inside) {
        let pix = pyu * P.size.x + pxu;
        out_c[pix] = c + t * vec4<f32>(P.bg.xyz, 0.0);
        out_t[pix] = vec2<u32>(bitcast<u32>(t), last);
    }
}

@compute @workgroup_size(16, 16, 1)
fn backward(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
            @builtin(local_invocation_index) li: u32) {
    let tile = wg.y * P.size.z + wg.x;
    let pxu = wg.x * 16u + lid.x;
    let pyu = wg.y * 16u + lid.y;
    let inside = pxu < P.size.x && pyu < P.size.y;
    let px = f32(pxu);
    let py = f32(pyu);
    let rg = ranges[tile];
    var t = 1.0;
    var last = 0u;
    var d = vec4<f32>(0.0);
    if (inside) {
        let pix = pyu * P.size.x + pxu;
        let tt = out_t[pix];
        t = bitcast<f32>(tt.x);
        last = tt.y;
        d = dl[pix];
    }
    var s = vec4<f32>(P.bg.xyz, 0.0);
    let n = rg.y - rg.x;
    let rounds = (n + B - 1u) / B;
    for (var rr = 0u; rr < rounds; rr = rr + 1u) {
        let r = rounds - 1u - rr;
        let base = rg.x + r * B;
        let cnt = min(B, rg.y - base);
        workgroupBarrier();
        if (li < cnt) {
            let i = ids[base + li];
            s_a[li] = proj[3u * i];
            s_b[li] = proj[3u * i + 1u];
            s_c[li] = proj[3u * i + 2u];
        }
        workgroupBarrier();
        for (var jj = 0u; jj < cnt; jj = jj + 1u) {
            let j = cnt - 1u - jj;
            let k = r * B + j;          // index in the tile's list
            var ga = vec4<f32>(0.0);
            var gb = vec4<f32>(0.0);
            var gc = vec4<f32>(0.0);
            if (k < last) {
                let a = s_a[j];
                let b = s_b[j];
                let e = response(a, b, px, py);
                if (e.x > 0.0) {
                    let al = e.x;
                    let cc = s_c[j];
                    let col = vec4<f32>(b.z, b.w, cc.x, cc.y);
                    let ti = t / (1.0 - al);
                    let gcol = (al * ti) * d;
                    let dal = ti * dot(col - s, d);
                    s = al * col + (1.0 - al) * s;
                    t = ti;
                    var go = 0.0;
                    var gu = vec2<f32>(0.0);
                    var gq = vec3<f32>(0.0);
                    if (b.y * e.y < ALPHA_MAX) {
                        go = e.y * dal;
                        let dp = b.y * e.y * dal;
                        let dx = e.z;
                        let dy = e.w;
                        gu = vec2<f32>(dp * (a.z * dx + a.w * dy), dp * (a.w * dx + b.x * dy));
                        gq = vec3<f32>(-0.5 * dx * dx * dp, -dx * dy * dp, -0.5 * dy * dy * dp);
                    }
                    ga = vec4<f32>(gu, gq.x, gq.y);
                    gb = vec4<f32>(gq.z, go, gcol.x, gcol.y);
                    gc = vec4<f32>(gcol.z, gcol.w, 0.0, 0.0);
                }
            }
            r_a[li] = ga;
            r_b[li] = gb;
            r_c[li] = gc;
            workgroupBarrier();
            for (var st = 128u; st > 0u; st = st >> 1u) {
                if (li < st) {
                    r_a[li] = r_a[li] + r_a[li + st];
                    r_b[li] = r_b[li] + r_b[li + st];
                    r_c[li] = r_c[li] + r_c[li + st];
                }
                workgroupBarrier();
            }
            if (li == 0u) {
                let slot = base + j;
                slot_g[3u * slot] = r_a[0];
                slot_g[3u * slot + 1u] = r_b[0];
                slot_g[3u * slot + 2u] = r_c[0];
            }
            workgroupBarrier();
        }
    }
}

@compute @workgroup_size(256, 1, 1)
fn reduce(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let i = gid.x + gid.y * nw.x * 256u;
    if (i >= P.size.w) { return; }
    var a = vec4<f32>(0.0);
    var b = vec4<f32>(0.0);
    var c = vec4<f32>(0.0);
    for (var k = csr_off[i]; k < csr_off[i + 1u]; k = k + 1u) {
        let sl = csr_slots[k];
        a = a + slot_g[3u * sl];
        b = b + slot_g[3u * sl + 1u];
        c = c + slot_g[3u * sl + 2u];
    }
    g2[3u * i] = a;
    g2[3u * i + 1u] = b;
    g2[3u * i + 2u] = c;
}
