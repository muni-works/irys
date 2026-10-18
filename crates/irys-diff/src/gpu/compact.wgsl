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
    let lam = paths[i].lam;
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

// Replay only: δf = 0 past the compact lanes, so the adjoint IR (which
// reduces over every lane) sees no stale adjoint from an earlier bounce.
@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let j = flat(gid, nwg);
    if (j < ind[3] || j >= arrayLength(&df_out)) { return; }
    df_out[j] = 0.0;
}
