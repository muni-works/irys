// Speed piece 2: lane compaction between `bounce` and the eval IR.
//
// The scan is integer and fixed-shape — a workgroup Hillis–Steele scan per
// block of 256 paths, then one workgroup over the block totals — so a
// path's compact slot is a function of the frame's lane bits alone: the
// forward pass and its replay compact identically, and so do two runs. No
// atomics. `ind` is the eval IR's indirect dispatch: (workgroups, 1, 1,
// compact lanes).

@group(0) @binding(63) var<storage, read_write> ind: array<u32>;

var<workgroup> csum: array<u32, 256>;

@compute @workgroup_size(256)
fn cscan_block(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let t = lid.x;
    let i = wid.x * 256u + t;
    var v = 0u;
    if (i < params.cap) {
        v = countOneBits(lane_bits[i]);
    }
    csum[t] = v;
    workgroupBarrier();
    for (var o = 1u; o < 256u; o = o * 2u) {
        var a = 0u;
        if (t >= o) {
            a = csum[t - o];
        }
        workgroupBarrier();
        csum[t] = csum[t] + a;
        workgroupBarrier();
    }
    if (i < params.cap) {
        lane_off[i] = csum[t] - v;
    }
    if (t == 255u) {
        lane_top[wid.x] = csum[255];
    }
}

@compute @workgroup_size(256)
fn cscan_tops(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    let nb = (params.cap + 255u) / 256u;
    let per = (nb + 255u) / 256u;
    let b0 = min(t * per, nb);
    let b1 = min(b0 + per, nb);
    var v = 0u;
    for (var b = b0; b < b1; b = b + 1u) {
        v = v + lane_top[b];
    }
    csum[t] = v;
    workgroupBarrier();
    for (var o = 1u; o < 256u; o = o * 2u) {
        var a = 0u;
        if (t >= o) {
            a = csum[t - o];
        }
        workgroupBarrier();
        csum[t] = csum[t] + a;
        workgroupBarrier();
    }
    var run = csum[t] - v;
    for (var b = b0; b < b1; b = b + 1u) {
        let c = lane_top[b];
        lane_top[b] = run;
        run = run + c;
    }
    if (t == 255u) {
        let lanes = csum[255] * params.nw;
        ind[0] = max((lanes + 255u) / 256u, 1u);
        ind[1] = 1u;
        ind[2] = 1u;
        ind[3] = lanes;
    }
}

// The graph's columns, at the compact lanes only.
@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.cap) { return; }
    let bits = lane_bits[i];
    if (bits == 0u) { return; }
    let lam = lams[params.base + i];
    var j = lane_slot(i) * params.nw;
    if ((bits & 1u) != 0u) {
        let a = stage[2u * i];
        for (var k = 0u; k < params.nw; k = k + 1u) {
            write_lane(j + k, a.x, a.y, a.z, a.w, band_weights(lam[k]));
        }
        j = j + params.nw;
    }
    if ((bits & 2u) != 0u) {
        let a = stage[2u * i + 1u];
        for (var k = 0u; k < params.nw; k = k + 1u) {
            write_lane(j + k, a.x, a.y, a.z, a.w, band_weights(lam[k]));
        }
    }
}

// δf = 0 past the compact lanes. Unused since speed round 2 (the replay's
// δθ sums the compact lanes only, `lane_sum`); kept for the pipeline table.
@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let j = flat(gid, nwg);
    if (j < ind[3] || j >= arrayLength(&df_out)) { return; }
    df_out[j] = 0.0;
}

// Speed round 2: the replay's δθ over the compact lanes only. The lane
// adjoint IR (`ggx::LaneAdjoint`) leaves `δf_v ∂f_v/∂θ_k` per lane in four
// columns; `lane_sum` sums each workgroup's 256 lanes by a fixed tree (one
// workgroup per 256 lanes of the wavefront; those past the compact lanes
// sum zeros), `lane_sum_top` sums the partials in a fixed order in one
// workgroup. A function of the
// compact lanes alone, so forward-replay and rerun give the same bits.
@group(0) @binding(65) var<storage, read> dz0: array<f32>;
@group(0) @binding(66) var<storage, read> dz1: array<f32>;
@group(0) @binding(67) var<storage, read> dz2: array<f32>;
@group(0) @binding(68) var<storage, read> dz3: array<f32>;
@group(0) @binding(69) var<storage, read_write> lpart: array<vec4<f32>>;
@group(0) @binding(70) var<storage, read_write> dsum: array<f32>;

var<workgroup> lsum: array<vec4<f32>, 256>;

fn lsum_tree(t: u32) {
    workgroupBarrier();
    for (var o = 128u; o > 0u; o = o >> 1u) {
        if (t < o) {
            lsum[t] = lsum[t] + lsum[t + o];
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn lane_sum(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let t = lid.x;
    let j = wid.x * 256u + t;
    var v = vec4<f32>(0.0);
    if (j < ind[3]) {
        v = vec4<f32>(dz0[j], dz1[j], dz2[j], dz3[j]);
    }
    lsum[t] = v;
    lsum_tree(t);
    if (t == 0u) {
        lpart[wid.x] = lsum[0];
    }
}

@compute @workgroup_size(256)
fn lane_sum_top(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    let nb = arrayLength(&lpart);
    var v = vec4<f32>(0.0);
    for (var b = t; b < nb; b = b + 256u) {
        v = v + lpart[b];
    }
    lsum[t] = v;
    lsum_tree(t);
    if (t < 4u) {
        dsum[t] = lsum[0][t];
    }
}
