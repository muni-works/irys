# The belief codec

The robot and the operator share one world, and the network only carries
surprise. This document specifies milestones M0–M3. M4–M6 (the Dojo view, the
residual lifecycle, commands in belief) are sketched at the end.

## Principle

Robot (K1) and operator (Dojo) run the same world model. The robot sends only
the **innovation**: what its sensors saw that the model did not predict. The
belief *state* is kept in lockstep with deterministic f64 CPU math. Each side
renders its own pixels from it (Jetson CUDA and Mac Metal will never agree bit
for bit; the state does). Unexplained pixels are shown as raw camera texture at
their measured depth until the belief absorbs them.

---

## M0: prove it offline (go/no-go)

**Data on disk**
- Map `ipse/maps/garage-perim`: `mesh.stl`, `sdf.bin` (cell 0.02 m),
  `splat.ply` (+ 7k/30k/clean decimations), `map.toml` (identity align; built
  from `perim.derived`, 144 views, scale 2.7077 m/unit).
- Capture `ipse-captures/2026-08-14-garage-sweep`: 1832 colour JPEGs (219 MB),
  2657 StereoNet depth frames (mono16 mm, valid 0.4–4.0 m), 544×448,
  fx = fy = 204.64, cx = 277.81, cy = 221.25, no distortion,
  frame `head_color_optical_frame`. `poses.jsonl` has **35** posed keyframes
  (cam_pos, 3×3 cam_rot); `frames.jsonl` / `jpegs.jsonl` give timestamps.

**Steps**
1. **Register** the capture frame to the map frame. Align the 35 keyframes'
   back-projected depth to the map SDF (point-to-SDF ICP, plus scale if the
   capture's units differ). Report the residual RMS in mm.
2. **Predict.** At each keyframe, render the map with irys from the camera
   (RGB + depth). Use the mesh first; splats once M1's loader lands.
3. **Track.** For every frame between keyframes, start from a constant-velocity
   prediction and refine SE(3) by render-and-compare: irys's rigid edge and
   interior gradients (2d–2f) on colour + depth, with a robust loss. Report
   drift against the held-out keyframes; hold out every other keyframe.
4. **Residual.** Per frame, compute the explained mask (|real − predicted| below
   a noise threshold, in colour and depth jointly) and the unexplained pixels.
5. **Code it two ways.**
   - (a) Mask + unexplained pixels, entropy-coded (run-length mask + JPEG/AVIF
     of patches, or a simple arithmetic coder over quantised residual).
   - (b) The pose update alone (6 floats) for frames whose residual is below
     threshold, i.e. the "nothing new" case.
6. **Baseline.** H.264 and H.265 (ffmpeg x264/x265) of the same JPEG sequence,
   at the same PSNR on the *reconstructed* frame. For the belief codec, the
   reconstruction is the prediction composited with the patches.
7. **Report.**
   - Bits/s over time for codec vs H.264/H.265.
   - PSNR/SSIM of the reconstruction.
   - Fraction of pixels explained.
   - Tracking drift with and without render refinement.
   - The top surprise events with thumbnails.

**Gate:** in well-mapped stretches the belief codec is ≥10× under H.264 at
equal reconstructed PSNR, and render tracking cuts drift vs odometry or
keyframe interpolation. If it misses, find out why (map quality? lighting
mismatch? tracking?) before M2.

**Where:** `ipse/crates/ipse-belief/examples/m0_garage.rs` (or a `tools/m0`
binary), irys as a git dependency at a fixed rev. No writes to the capture or
map directories; outputs go to `ipse/out/m0/`.

---

## M1: irys prerequisites

Each item lands on its own irys branch, gated by the usual invariants.

**Status (2026-10-19, irys `claude/m1`):** built, every item a commit;
numbers, API and deviations in irys `docs/design.md`, "M1".

1. [x] **Splats unparked.**
   - A `.ply` loader for 3DGS: position, scale, rotation, opacity, SH up to
     degree 3.
   - Splats in the OpenPBR scene kernel (3DGRT proxies in the same TLAS as
     meshes, the sorted k-buffer composite from 2b, SH evaluated per view).
   - Optional per-splat material for relighting later. *(not started)*
   - Gate: the garage splat renders within N dB PSNR of gsplat's own render at
     held-out capture views. *(gsplat is CUDA-only and the capture is not
     yet registered to the map — M0 step 1 — so the gate ran against a CPU
     EWA rasteriser at the map's held-out training cameras; see design.md.)*
2. [x] **AOVs.** First-hit distance (metric, with a real miss marker), instance id,
   material id, normal, and the per-pixel σ of the belief (the σ input comes
   from the caller). Separate buffers, read back on demand.
3. [x] **Batched cameras.** N cameras × M scenes in one submit / one wait / one
   readback; spectrum readback optional. Target: per-camera overhead under
   0.5 ms at 544×448 depth-only.
4. [x] **TLAS-only rebuild.** `set_instances` moves placements without re-uploading
   BLAS or geometry. Target: under 1 ms for 30 instances.
5. [x] **Camera principal point** (cx, cy offsets) plus OpenCV pinhole
   conventions, matching `camera_info.json`.
6. [x] **Residual pass.** The GPU computes |real − predicted| with a joint
   colour/depth noise model, giving the explained mask and the unexplained
   patch list. That feeds M0's coder and M4's patches.
7. [x] **irys-ffi + IOSurface spike.** A C ABI that renders into an IOSurface-backed
   texture that Swift/Metal presents with no copy. First answer whether wgpu 30
   exposes the Metal texture (`wgpu::hal` interop); fall back to a single
   copy-to-shared-surface if not. *(wgpu 30 does: no copy.)*

---

## M2: the belief engine (`ipse-belief`)

**State** (all f64, serialisable, hashed):
- `map`: a reference to an immutable base map (mesh + SDF + splats) by content
  hash, plus a delta log.
- `robot`: SE(3) pose, velocity, joint angles (from the K1 bus), and covariance.
- `objects`: a list of { id, kind (splat-blob | mesh | cad-program), pose,
  shape params, material, covariance, born_at, last_seen, provenance }.
- `env`: lighting parameters (sky/sun or SH) and exposure.
- `clock`: frame index and time.

**Predict** (deterministic):
- Robot: integrate IMU/odometry and joint FK; process-noise growth in σ.
- Objects: static by default. Dynamic ones get a phyz step (rigid, contact
  with the map SDF) plus a constant-velocity fallback.
- σ grows for regions and objects not observed (staleness).
- **Deformable and chaotic things** (snow, cables, cloth, a plowed windrow)
  get a *generative* learned residual: a latent diffusion prior, not a
  regressor. Following PhysLDM (2610.07609), regression on chaotic dynamics
  predicts a damped, unphysical average; diffusion models the distribution.
  Lockstep still holds: the sampler is seeded from the delta's seq and the
  state hash, and evaluated in deterministic f64 on CPU (or its quantised
  outputs are carried in the delta), so both ends sample the same future.

**Update** (the robot side only; it needs the camera):
1. Render the expected frame (irys, M1 batched + AOVs).
2. Pose refinement: Gauss–Newton/LM on SE(3) with irys Jacobians.
3. Residual: the explained mask, then unexplained blobs (connected components in
   colour + depth).
4. Each blob is matched to an existing object (update its pose/shape) or
   **born** as a splat-blob object from its back-projected pixels.
5. MDL promotion (M5): a blob is promoted to mesh, then to a CAD program, when
   that shortens the description. irys-recon supplies the program fit.
6. Objects not explained for T seconds and contradicted by observation (seen
   through) are retired.

Output per step: a **state delta** (the innovation) plus the new state hash.

**Determinism.** Every update that changes state is computed in f64 on the CPU
from inputs that are themselves part of the delta. GPU-derived quantities enter
the state only as **quantised values carried inside the delta**. The receiver
never recomputes them, so Jetson/Mac GPU differences can't desync the state.

**API sketch**

```rust
let mut belief = Belief::new(map_ref, robot_model, cfg);
// robot side
let delta = belief.step(&sensors, &mut irys_ctx)?;   // predict + update
send(delta);
// operator side
belief.apply(&delta)?;                                // predict + apply, no camera
assert_eq!(belief.hash(), delta.state_hash);
```

---

## M3: the innovation transport

**Messages** (CBOR or postcard, versioned):
- `Delta { seq, t, parent_hash, ops: [Op], state_hash, budget_hint }`
- `Op`:
  - `RobotPose(se3_q)`, `Joints(q16)`
  - `ObjBorn { id, kind, params_q, patch_ref? }`
  - `ObjUpdate { id, dparams_q }`, `ObjRetire { id }`
  - `EnvUpdate`
  - `Patch { id, depth_q, rgb_avif }` (residual pixels until absorbed)
- `Keyframe { full state snapshot }`: on connect, on desync, and every N s.
- `Ack { seq, state_hash }` from the operator.

**Quantisation.** Poses at 0.1 mm / 0.01°, params per type. The quantised value
*is* the state on both sides, so quantisation never desyncs.

**Lockstep and recovery.**
- The receiver applies deltas in seq order and checks `state_hash`.
- On a mismatch or gap it requests a `Keyframe`.
- Deltas are idempotent by seq.

**Rate control.**
- A budget in kb/s. Patches are prioritised by expected operator relevance:
  near the robot's path, large depth discontinuity, new.
- Low-priority patches are deferred or downsampled.
- Pose and safety-critical ops always go first.

**Transport.** Over ipse-view's bridge (QUIC/WebSocket). Patches go on
unreliable datagrams, ops on the reliable stream.

**Raw-video inset.** A separate low-rate H.264 stream (e.g. 160×128 at 5 fps)
stays on as ground truth. It's small, but always present.

**Gate:**
- A recorded garage replay through a simulated link (latency, loss) keeps the
  operator state hash equal 99.9% of the time, and resyncs within one RTT +
  keyframe otherwise.
- The bandwidth matches M0's numbers within 20%.

---

## M4–M6 (sketch)

- **M4, Dojo belief view:**
  - irys-ffi viewport rendering the operator-side belief, predicted to "now"
    (client-side prediction).
  - Residual patches as textured quads at depth.
  - σ haze, a surprise timeline, and the raw-video inset.
- **M5, residual lifecycle:** pixels → splats → mesh → CAD via MDL and
  irys-recon. The operator watches the world sharpen.
- **M6, commands in belief space:**
  - Select an object to act on: "pick up", "go behind".
  - An operator edit to an object is a strong prior op sent upstream; the
    robot's update must explain it or reject it, visibly.
