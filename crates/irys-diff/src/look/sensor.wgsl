// M1: the sensor kernel. One primary ray per pixel per camera, no jitter,
// no shading: first-hit distance, instance id, material id, shading normal,
// the caller's per-instance σ, and the splat layer — each to its own buffer,
// N cameras in one dispatch (x: pixel, y: camera).
//
// Appended to scene.wgsl + splat.wgsl: it traces with the scene kernel's own
// `trace_ray` (cut-outs alpha-tested) and `splat_segment`.

struct SensorParams {
    size: vec4<u32>,   // w, h, cameras, flags (A_*)
    misc: vec4<f32>,   // σ at a miss, -, -, -
    count: vec4<u32>,  // σ entries before the splat one (instance capacity), -, -, -
}

@group(0) @binding(70) var<uniform> S: SensorParams;
@group(0) @binding(71) var<storage, read> cams: array<vec4<f32>>;      // 4 per camera: (eye, pixel angle), fwd, right, up
@group(0) @binding(72) var<storage, read_write> aov_dist: array<f32>;
@group(0) @binding(73) var<storage, read_write> aov_inst: array<u32>;
@group(0) @binding(74) var<storage, read_write> aov_mat: array<u32>;
@group(0) @binding(75) var<storage, read_write> aov_normal: array<u32>;
@group(0) @binding(76) var<storage, read> sigma_in: array<f32>;
@group(0) @binding(77) var<storage, read_write> aov_sigma: array<f32>;
@group(0) @binding(78) var<storage, read_write> aov_rgb: array<vec4<f32>>;

const A_DIST: u32 = 1u;
const A_INST: u32 = 2u;
const A_MAT: u32 = 4u;
const A_NORMAL: u32 = 8u;
const A_SIGMA: u32 = 16u;
const A_RGB: u32 = 32u;
const SPLAT_ID: u32 = 0xfffffffeu;

fn oct_pack(n: vec3<f32>) -> u32 {
    let p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    var e = p;
    if (n.z < 0.0) {
        let s = select(vec2<f32>(-1.0), vec2<f32>(1.0), p >= vec2<f32>(0.0));
        e = (vec2<f32>(1.0) - abs(p.yx)) * s;
    }
    return pack2x16snorm(e);
}

@compute @workgroup_size(64)
fn aov(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = S.size.x;
    let h = S.size.y;
    let px = gid.x;
    let cam = gid.y;
    if (px >= w * h || cam >= S.size.z) { return; }
    let o = cam * w * h + px;
    let fw = f32(w);
    let fh = f32(h);
    let xs = f32(px % w) + 0.5;
    let ys = f32(px / w) + 0.5;
    // scene.wgsl's camera with tan_half = 1: a pinhole's principal point and
    // focal lengths live in the (unnormalised) fwd / right / up.
    let sx = (2.0 * xs / fw - 1.0) * fw / fh;
    let sy = 1.0 - 2.0 * ys / fh;
    let ro = cams[4u * cam].xyz;
    let rd = normalize(cams[4u * cam + 1u].xyz + cams[4u * cam + 2u].xyz * sx + cams[4u * cam + 3u].xyz * sy);
    let s = trace_ray(ro, rd, M_GEOMETRY | M_LIGHT);
    let g = splat_segment(ro, rd, cams[4u * cam].w, 0.0, select(3.4e38, s.t, s.ok));
    let fl = S.size.w;
    let inf = bitcast<f32>(0x7f800000u);
    var dist = inf;
    var id = NONE;
    var mat = NONE;
    var nrm = 0u;
    var sig = S.misc.x;
    if (g.t_half >= 0.0) {
        dist = g.t_half;
        id = SPLAT_ID;
        mat = SPLAT_ID;
        if ((fl & A_SIGMA) != 0u) { sig = sigma_in[S.count.x]; }
    } else if (s.ok) {
        dist = s.t;
        id = s.inst;
        mat = s.mat;
        nrm = oct_pack(select(s.ns, -s.ns, dot(s.ns, rd) > 0.0));
        if ((fl & A_SIGMA) != 0u) { sig = sigma_in[s.inst]; }
    }
    if ((fl & A_DIST) != 0u) { aov_dist[o] = dist; }
    if ((fl & A_INST) != 0u) { aov_inst[o] = id; }
    if ((fl & A_MAT) != 0u) { aov_mat[o] = mat; }
    if ((fl & A_NORMAL) != 0u) { aov_normal[o] = nrm; }
    if ((fl & A_SIGMA) != 0u) { aov_sigma[o] = sig; }
    if ((fl & A_RGB) != 0u) { aov_rgb[o] = vec4<f32>(splat_linear(g), 1.0 - g.trans); }
}
