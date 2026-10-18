# irys — design

One renderer for the stack. vcad, kosm, ipse, rose and muni each draw pixels
today, in five different ways. irys is where the light lives; every repo above
it is a client that hands over geometry and a camera and gets a frame back.

Status: proposal, 2026-10-17. Phase 0 (seed + split) has landed; see below.

## Why now

The survey (2026-10-17) found:

| Where | What | Problem |
|---|---|---|
| `kosm/crates/kosm-render` (41k LOC) | spectral path tracer, BVH/TLAS, caustics, splats, heightfield, blue noise, GPU tier (wgpu 30) | already geometry-agnostic — but lives in a game's repo and is versioned with it |
| `kosm/crates/kosm-view` (10k) | winit viewer, temporal history, à-trous, raster live tier | duplicates `kosm-render::gpu::{history,budget}` |
| `vcad/crates/vcad-kernel-raytrace` (16k) | direct BRep ray tracing, its own `Bvh`/`Tlas`/`Instance`/`pathtrace`, wgpu 23 | second copy of the same acceleration stack, same type names |
| `vcad/crates/vcad-kernel-gpu` (4k) | wgpu context, wavefront | third GPU context layer |
| `vcad/crates/vcad-render` (13k) | hidden-line SVG/PNG, `photoreal`, `materials`, `envmap` | second material/env model; rose's only renderer |
| `phyz/crates/phyz-camera` (3k) | wgpu 23 rasterizer for sim cameras | fourth GPU context, own mesh/scene types |
| `muni/depot/splat-worker`, ipse `scripts/splat_*` | Python gsplat on CUDA | splats rendered by a stack irys can't call |
| `muni/depot/console`, `vcad/packages/app` | three.js | out of scope for v1 (see below) |

`kosm-view` building vcad-raytrace with `gpu` means one binary links wgpu 23
*and* 30. That alone is worth fixing.

## Principles

1. **irys owns no geometry.** It never depends on vcad, phyz, kosm or ipse.
   Clients implement `Geometry`; irys bounds it, traverses it and shades it.
   This is the rule `kosm-render` already follows — keep it.
2. **A substance is not a look.** irys takes optical constants (spectral
   albedo, IOR/Sellmeier, roughness, emission). kosm's `material` library
   stays in kosm and derives them; irys never carries a second brass.
3. **CPU is the reference, GPU is a tier.** Every GPU result has a CPU path
   it is tested against. Headless and CI build with no device.
4. **One wgpu.** The whole graph agrees on one major version, bumped in irys
   first.
5. **Pure frame.** `render(scene, camera, budget) -> Frame` is a function.
   Temporal state (history, reprojection) is an explicit value the caller
   threads, not hidden in a viewer — so kosm's "same id, same bytes" holds.
6. **Differentiable where it is cheap.** Scalars are `tang::Scalar` so duals
   flow through optics and the frame (kosm's marble already depends on this).

## Crate split

```
irys/
  crates/
    irys-core     math, Ray/Hit, Aabb, Geometry trait, Bvh/Sah, Tlas/Instance,
                  Camera, Frame (rgba + depth + normal + id AOVs). no deps but tang.
    irys-light    spectrum (CIE, Sellmeier), optics (Fresnel), BSDFs, env/HDR,
                  sampler (blue noise), CPU pathtrace, caustics/photon map.
    irys-prims    stock Geometry impls: TriMesh, Analytic, HeightField, Splats.
    irys-gpu      one wgpu context, buffers, resident scene, WGSL integrator,
                  history/budget/à-trous/firefly cap, neural denoiser. feature-
                  gated; the only crate that names wgpu.
    irys-raster   fast live tier: forward pass, shadow map, SH probes (from
                  kosm-view/raster) + a sim-camera mode (absorbs phyz-camera).
    irys-line     hidden-line / flat-shaded SVG+PNG (from vcad-render's core).
    irys-view     winit window over irys-gpu/raster. optional, never in CI.
    irys          facade: `use irys::prelude::*;`, re-exports by feature.
```

Rule of thumb for a boundary: a crate exists if some client wants it without
the rest. rose wants `irys-line` only. phyz wants `irys-raster` only. A
headless kosm test wants `irys-core + irys-light` only.

## The Geometry trait

Keep `kosm-render`'s trait as is — it has survived BRep, meshes, splats and
heightfields:

```rust
pub trait Geometry {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn bounds(&self, i: usize) -> Aabb;
    fn intersect(&self, ray: &Ray, i: usize, t_min: f64, t_max: f64) -> Option<Hit>;
    fn intersect_all(&self, ray: &Ray, i: usize, out: &mut Vec<Hit>) { /* convex default */ }
    fn occludes(&self, ray: &Ray, i: usize, t_min: f64, t_max: f64) -> bool { /* via intersect */ }
}
```

Additions, each optional with a default so no existing impl breaks:

- `fn material(&self, i: usize) -> MaterialId` — today materials ride in
  `Hit`; making it a trait method lets the GPU tier upload it once.
- `fn gpu(&self) -> Option<GpuGeometry<'_>>` — how a primitive set
  describes itself to `irys-gpu` (triangle soup, analytic prim table, splat
  buffer, heightfield texture). `None` means CPU-only, and the GPU tier
  falls back or refuses loudly, never silently drops it.
- `fn tessellate(&self) -> Option<TriMesh>` — for `irys-raster` and
  `irys-line`, which need triangles. BRep impls in vcad return the
  tessellator's mesh; analytic prims tessellate themselves.

What stays out: transforms (that's `Tlas`/`Instance`), animation, and any
notion of a document or scene graph. A client's scene graph builds a `Tlas`.

**vcad's BRep** becomes `impl Geometry for BrepFaces` in vcad (in
`vcad-kernel-raytrace`, shrunk to the surface intersectors + trimming).
Its `Bvh`, `Tlas`, `pathtrace` and `gpu/` are deleted in favour of irys's.

## Clients after the move

| Client | Depends on | Keeps locally |
|---|---|---|
| kosm | `irys` (light, gpu, view) | `Camera` as a `Lens`, materials → optics constants, `brep::Scene` adapter |
| vcad | `irys-core`, `irys-light`, `irys-line`, `irys-gpu` (wasm) | BRep `Geometry` impl, `.vcad` → scene, CLI flags |
| rose | `vcad-render` → `irys-line` | nothing new; `render.sh` unchanged |
| phyz | `irys-raster` | `phyz-camera` becomes a thin adapter |
| ipse | `irys-prims::Splats` + `irys-gpu` | map → splat buffer adapter |
| muni | see below | |

## muni

muni is the reason this has to be a library and not a kosm crate. What it
needs, in order:

1. **sim-world depth/RGB from 3DGS** (`docs/sim-service-plan.md`). Today that
   plan is Python gsplat in a container. With `irys-prims::Splats` +
   `irys-gpu`, sim-bridge renders depth from arbitrary poses in Rust, in
   process, no ZMQ hop, and the same renderer trains the world model and
   tests the policy. This is the first muni deliverable.
2. **Sim cameras for bvr's `sim` crate** — `irys-raster`'s sim-camera mode,
   batched like kosm's `step_batch`.
3. **Depot digital twin.** The console stays three.js for now. Later,
   `irys-gpu` on wasm/WebGPU can render the splat map behind the three.js
   overlays. Not v1.
4. **Splat training** stays Python (gsplat/CUDA). irys consumes the `.ply`;
   it does not train splats in v1.

Out of scope: teleop video (GStreamer), Rerun replay, the marketing site.

## Migration order

Each step lands green on its own; nothing is a flag day.

1. **Seed.** `git subtree split` `kosm/crates/kosm-render` into
   `irys/crates/irys-core` with history; republish nothing yet. kosm points at
   it by path via `.cargo/config.toml`, CI by git rev.
2. **Split** that crate into `irys-core`, `irys-light`, `irys-prims`,
   `irys-gpu` along the module lines above. Pure moves; `kosm-render` becomes
   a re-export shim so kosm and vcad 0.10 keep compiling.
3. **Hoist kosm-view's** history/budget/temporal into `irys-gpu` (dedupe with
   `kosm-render::gpu`), raster into `irys-raster`, window into `irys-view`.
4. **Publish** `irys-*` 0.1 to crates.io. kosm drops the shim and depends on
   irys directly.
5. **vcad onto wgpu 30 + irys.** `vcad-kernel-raytrace` keeps surface
   intersectors and implements `Geometry`; its BVH/TLAS/pathtrace/gpu go.
   `vcad-kernel-gpu`'s context is replaced by `irys-gpu`'s for rendering
   (its compute-only parts — narrowphase, router — stay in vcad).
   Snapshot every vcad render test before and after.
6. **vcad-render → irys-line + thin CLI.** rose is checked with its
   `render.sh` output diffed pixel-wise.
7. **phyz-camera → irys-raster** adapter.
8. **muni sim-bridge** renders splat depth through irys (muni #1 above).
9. **ipse** splat viewer/compare scripts move to a irys example.

Steps 1–4 are kosm-only and low risk. Step 5 is the big one: two BVHs with
the same names, a wgpu major bump, and wasm in the vcad app.

## Phase 0: the split as built (2026-10-17)

Pure moves (`git mv`), no behaviour change. What landed differs from the
sketch above in three places, all because a module-line split would have
made a cycle:

- **Splats live in `irys-light`, not `irys-prims`.** `splats.rs` shades
  through `pathtrace`'s surface model and the CPU integrator composites
  splats itself (`Scene::splats`), so prims → light → prims. `irys-prims` is
  `analytic` + `heightfield`.
- **TriMesh stays in `irys-core`.** It is defined in `geometry.rs` beside
  the trait, and the integrator, the GPU tier and the tests all name it.
- **The sampler (blue-noise mask, Sobol) is in `irys-core`**, because both
  tiers read it; `spectrum`/`optics` call each other and go to `irys-light`
  together with `tables`, `env`, `caustics`, the CPU tier (`cpu/`,
  `pathtrace`) and the CPU half of the neural denoiser (`neural`). There is
  no `irys-neural`: the device half is `irys-gpu::neural`.

```
irys-core   (tang)                       math ray geometry bvh sah tlas sampler
irys-prims  -> core                      analytic heightfield
irys-light  -> core   (rayon native)     spectrum optics tables env cpu pathtrace
                                         caustics splats neural
irys-gpu    -> core, light (wgpu 30)     the old gpu/ module + shaders
irys        -> core, light, prims, gpu?  facade, kosm-render 0.2's public paths;
                                         `gpu` feature pulls irys-gpu
kosm-render -> irys                      0.2.1 shim, `pub use irys::*`
```

Each moved module keeps its `crate::x` paths through a private
`use irys_core::{..}` at its new crate root. The integration tests live in
the facade (`crates/irys/tests`) since they span tiers.

## Phase 1 spike: the contract on CPU (2026-10-17)

`crates/irys-diff`, its own crate because tang-compile is a git dependency
(`bfc8003`, kosm's rev) and irys-light has to stay publishable. The
workspace `[patch.crates-io]` sends `tang`/`tang-expr` to that rev so the
graph holds one tang; it folds into irys-light when the IR publishes.

- `ggx.rs` — a rough conductor (GGX, height-correlated Smith, Schlick with a
  three-hat spectral F0), `θ = (α, F0₀, F0₁, F0₂)`, written once as one
  `TensorGraph` batched over a wavefront: inputs are `[n]` columns of vertex
  cosines and hero-wavelength hat weights. Lowered to eval (`f`, `pdf`) and
  adjoint (`backward` of `Σ δf·f`, so the batch sum is the reduction).
  Sampling (VNDF) is f64 CPU code and detached.
- `prb.rs` — forward pass keeps `L` per path; replay reseeds, re-walks, and
  adds `δL (L − L_j) ∂f_j/f_j`. The derivation is the module doc.
- `ir.rs` — compiled C (f32) or interpreter (f64), `IRYS_IR=interp` forces
  the latter.

Scene: GGX ball on a grey Lambertian disc under a smooth sky + soft sun,
`irys_core::Bvh` over `irys_prims::Analytic`, one hero wavelength per path
through `irys_light::spectrum`. Results (debug build, M-series):

| Gate | Result |
|---|---|
| gradient vs CRN central differences, sampler frozen, L2, f64 | rel error 1.6e-10 … 1.9e-9 on all four θ |
| same, sampler at current α (paths move), mean pixel, 64 seeds | α: −5.84e-2 ± 0.40e-2 vs −5.79e-2 ± 1.2e-2 (0.0σ); F0 bands rel ≤ 1e-6 |
| compiled C f32 vs interpreter f64 | f 1.6e-6, pdf 1.7e-6, δθ 9.4e-7 worst rel; interpreter = hand f64 to 1e-12 |
| replay vs forward | bit-identical L on 512/512 paths, both backends |
| memory | 30.1 f64 words per path at depth 2 and 8, nothing per vertex |
| inverse: α 0.6 → 0.25, 32×32×4, Adam | within 0.02 at iteration ~20; last-10 mean 0.265 |
| timing | gradient (render + forward + replay) 4096 paths ≈ 35 ms native, ≈ 1.3 s interpreted; 512 paths interpreted 0.17 s |

What the spike says the design got wrong or left out:

1. **The replay weight has a `1/f`.** It is `δL (L − L_j) ∂θ log f_j`; the
   contract's "`(L − L_so_far) · ∂f/∂θ`" is off by `f_j`.
2. **A nonlinear loss needs two seeds.** With one, the L2 gradient
   `2 (I − T) ∂I` correlates the noise in both factors and descent also
   minimises variance: the first inverse run walked α from 0.6 to 1.0.
   `prb::gradient` takes `(primal, adjoint)` seeds; same seed only for the
   CRN check.
3. **Memory is paths × 1, not pixels × 1**, once forward and replay are
   separate wavefront passes (`L` per path between them). Pixels × 1 needs
   the replay fused into the forward kernel per path, or a third pass.
4. **The pdf the integrator divides by is the sampler's**, not the eval
   graph's: the two agree (tested to 1e-9), but the one that drew the
   direction is the one that counts (frozen sampling makes that explicit).
   The graph's pdf is for MIS.
5. **Replay bit-exactness needs one program for `f`.** The eval and adjoint
   lowerings fuse differently and may round differently in f32, so replay
   re-evaluates `f` with the eval program and runs the adjoint only for `δθ`.
   The C is built with `-ffp-contract=off` for the same reason.
6. **tang-gpu is out.** Its `CpuProgram` links wgpu 25 (principle 4); `ir.rs`
   carries the forty-line loader instead.
7. Not done: PRB inside `irys-light`'s integrator (the spike has its own
   200-line wavefront), the floor's BSDF as a graph, NEE/MIS, Russian roulette, visibility (reparam) gradients.

### Hero wavelengths ×4 (2026-10-17)

Each path now carries four stratified wavelengths, the hero and three
rotated by `k/4` of the band (Wilkie+ 2014). The hero alone draws the
direction; the graph runs per *lane* (path × wavelength, so the wavefront is
4× wider) and the image splats each lane at `hero_weight(λ_k) / (4 spp)`.
GGX's sampler is λ-free, so all four share the hero's pdf and the MIS weight
is a flat ¼. Each lane is its own PRB path (`L`, `L_j`, `δL`).

| | |
|---|---|
| per-pixel variance, 24×24×2, 16 seeds, 1 λ → 4 λ | total 7.4× lower, chroma (channel − grey) 28.7× lower |
| frozen-sampler gradient vs CRN central differences | rel ≤ 6.1e-10, unchanged gate |
| live sampler, 64 seeds | α 0.4σ, F0 bands rel ≤ 1e-6 |
| replay vs forward | bit-identical on every lane, both backends; 63.1 words/path at depth 2 and 8 |
| inverse α 0.6 → 0.25 | within 0.02 at iteration 13; last-10 mean 0.234 |
| `examples/fit`, 96×96×8, 60 its | θ = [0.188, 0.334, 0.783, 0.997] vs truth [0.2, 0.34, 0.78, 1.0] |

No tolerance changed.

## Phase 2a: the PRB wavefront on the GPU (2026-10-17)

`irys-diff::gpu` behind a `gpu` feature, on `irys-gpu`'s wgpu 30 context.
The phase-2 risk was replay bit-exactness on a device; it held first time.

**The IR does emit WGSL.** tang-compile at `bfc8003` has
`TensorGraph::to_wgsl` (one kernel per materialised comp, views as index
arithmetic, `split` schedules for reductions). So nothing is hand-lowered:
the GGX eval lowering (`f`, `pdf`) and adjoint lowering (`δθ`) are the IR's
own kernels, dispatched as wavefront stages. Only the path logic is hand
WGSL (`src/gpu/wavefront.wgsl`): `gen`, `bounce` (intersect the two
analytic prims, VNDF-sample at the hero, write the seven graph columns per
lane), `update` (`T *= f cos/p`, and in replay `δf = δL (L − L_j)/f`),
`finish` (`L` per lane), `accumulate`. A bounce is bounce → eval IR →
update [→ adjoint IR → accumulate], one compute pass per frame.

**Bit-exactness, by construction.** Forward and replay dispatch the same
`ComputePipeline`s over the same bind groups — one shader module per
kernel, built once — so `f` and everything feeding a path is the same
machine code on the same inputs. The replay differs only in a uniform that
gates the `δf` store. The adjoint program recomputes `f` internally; that
copy never touches throughput (note 5 above). RNG: `prb::Rng` ported
exactly (SplitMix64 on `u32` pairs, 64-bit add/mul by 16-bit limbs); a
uniform keeps the top 24 of the CPU's 53 bits, so CPU and GPU walk the same
paths up to f32 rounding. Nothing broke; the only f32 accommodation is the
surface offset, 1e-4 (t_min 1e-5) instead of the CPU's 1e-6.

**The reduction: no atomics.** WGSL has no float atomics and a CAS loop on
`atomic<u32>` makes the sum order (so the gradient's low bits) depend on
scheduling. Instead the adjoint's `Σ_lanes` stays the IR's own monoid
reduction, scheduled `split = 256` (one workgroup per parameter, threads
stride the lanes, tree combine); `accumulate` adds the four results per
bounce in bounce order. The gradient is bit-deterministic run to run
(tested).

| Gate (debug build, M-series, Metal) | Result |
|---|---|
| replay `L` vs forward `L`, 64×64×4 × 4 λ | 65536/65536 lanes bit-identical, depth 2 and 8; `δθ` identical twice |
| GPU f32 vs CPU f64 interpreter, frozen sampler, 32×32×4, one seed | `L` median rel 1.2e-7, p99 1.5e-4 (14/16384 lanes > 1e-3: silhouette rays); loss rel 5.8e-7; `δθ` rel 8.3e-5, 7.0e-6, 1.9e-5, 3.5e-4 |
| GPU PRB vs GPU central differences (h = 1e-3, CRN) | rel 1.1e-6, 9.9e-6, 6.9e-6, 1.9e-3 (θ₃, the smallest gradient, is f32 FD noise) |
| throughput, 96×96×8 (73 728 paths × 4 λ, depth 4) | forward 1.1 ms incl. readback (≈ 66 Mpaths/s); replay 20.6 ms; gradient with host scoring 166 ms |
| CPU compiled-C path, same frame, same profile | forward 453 ms (0.16 Mpaths/s), gradient 1543 ms: GPU 9× end to end, ~400× on the forward pass |
| `examples/fit_gpu`, same seeds and Adam as `fit` | θ = [0.188, 0.334, 0.783, 0.997] vs truth [0.2, 0.34, 0.78, 1.0] — the CPU run's numbers to three places; 60 iterations + previews in 14 s vs ≈ 2 min |

![phase 2a fit](phase2a-fit.png) (`phase2a-loss.csv` is its loss and θ.)

Remaining risks and follow-ups:

1. **Host scoring dominates.** ~140 ms of the 166 ms gradient is the host
   building the image, the loss and `δL` per lane from `L` (debug Rust).
   Image, loss and `lane_adjoint` belong on the device; then the host reads
   back four floats. *Done in phase 2b, piece 1.*
2. **Replay cost is the adjoint.** Four split reductions per bounce, each
   recomputing `f` and its derivative per lane: 20 ms vs 1 ms forward. A
   fused one-pass reduction over all four parameters (one kernel, width-4
   accumulator) is an IR schedule, not a design change.
3. **The scene is hand-written in WGSL.** Two analytic prims, no BVH. Phase
   2 proper needs `irys-gpu`'s scene traversal inside `bounce`; bit-exactness
   then also depends on traversal order being deterministic (it is, for a
   stackful BVH walk, but test it). *Done in phase 2b, piece 2; tested.*
4. **Lane layout.** Graph inputs are seven separate `[lanes]` columns because
   that is how `to_wgsl` binds inputs; that is 9 storage bindings in `bounce`.
   A struct-of-columns input or a `[lanes, 7]` view would cut it.
5. **Cross-vendor.** Bit-exactness is per device and driver: the same
   pipeline on the same GPU. Replaying on another adapter is not promised,
   and nothing needs it.
6. f32 everywhere on the device; long paths and big scenes will want the
   offset scaled to the hit distance. *Done in phase 2c, piece 2.*

## Phase 2b: the real scene on the GPU wavefront (2026-10-17)

Two pieces, both in `irys-diff::gpu`.

**Piece 1 — scoring on the device** (phase 2a risk 1). Three more wavefront
kernels: `score` (one thread per pixel sums its `spp × 4` contiguous lanes
in lane order through a WGSL port of `spectrum::hero_weight`, then writes
the residual's `∂loss/∂image` and squared error), `loss_sum` (one workgroup:
a fixed per-thread stride, then a fixed tree), `lane_adjoint` (δL per lane
from the adjoint frame's own wavelengths). No float atomics anywhere; the
host writes uniforms and reads back five floats, `loss` and `δθ`
(`GpuPrb::gradient_l2`). The host-scored `gpu::gradient` stays as the
reference and for non-L2 losses.

| Gate | Result |
|---|---|
| device image / δL vs host f64 scoring of the same `L` | max err / max 1.0e-6 / 1.6e-6; loss rel 5.3e-8; `δθ` rel ≤ 5.3e-5 |
| device-scored gradient, two seeds, run twice | bit-identical loss and `δθ` |
| gradient, 96×96×8, debug | 170–178 ms host-scored → **11.6–12.8 ms** device-scored; CPU compiled C 1583 ms (124×) |
| same, `--release` | 27–42 ms → **7.0–7.8 ms**; CPU 116–129 ms (17×) |
| `fit_gpu` | same θ to three places, 60 iterations 14 s → 5 s |

**Piece 2 — irys's BVH on the wavefront.** `irys-gpu` has no traversal of
its own (its geometry seam leaves `trace_scene` to the client), so the
wavefront ports `irys-core`'s flat BVH: `Bvh::flatten` of two trees packed
into one node array (opaque tree at 0, splat tree after it), one
leaf-index array, one `vec4` record array — three bindings. `prb::Scene`
is now that: `Bvh<Opaque>` (a `TriMesh` and an `Analytic` behind one
`Geometry`, combined indices, a `Surface` per prim) plus an optional
`Bvh<Splats>`; the CPU reference traces the same trees. `Scene::new` is the
phase 1/2a scene, now traced through the BVH on the device too (its tests
are unchanged and pass); `Scene::garden` is the ball on a 512-triangle
checkered floor in front of a 900-splat procedural bush.

- **Walk order.** A 64-entry stack, near child first by slab entry (left on
  a tie), the far child pushed under it; nodes are culled against the
  current closest `t`. The walk is a function of the ray, so replay
  retraces it; tested bitwise across runs and across two wavefronts.
- **Splats, 3DGRT-style.** The 3σ boxes are the proxies in the splat tree.
  A segment (up to the opaque hit, or ∞) composites in rounds: each round
  walks the tree and keeps a sorted K = 16 buffer of the nearest hits
  strictly after the last one composited, ordered by `(t, index)` — a total
  order, so the result does not depend on which nodes the walk visits
  first — and culls boxes past the buffer's last entry once full.
  `C += T α c; T *= 1 − α`, out at `T ≤ 1e-4`, exactly
  `irys_light::splats::composite`'s model (emissive and absorbing, baked
  lighting, no shading, no secondary rays). Spectral radiance is the sRGB
  colour lifted onto the conductor's three hats (`prb::splat_radiance`).
- **Differentiable in θ only.** Splats are θ-free factors on the path
  (`L` gains `T_path · C`, the throughput gains `T`), so the PRB estimator
  is unchanged. Splat parameters (means, covariances, opacity, colour) are
  the follow-up: 3DGRT's backward is a second ordered walk per segment.

| Gate (debug, M-series, Metal) | Result |
|---|---|
| replay vs forward, garden 64×64×4 × 4 λ | 65536/65536 lanes bit-identical at depth 2 and 6; `δθ` identical twice |
| walk-order determinism | two wavefronts × two runs: `L`, image, loss, `δθ` bitwise identical |
| GPU f32 vs CPU f64 (same trees, CPU `splats::composite`), frozen sampler, 32×32×4 | `L` median rel 1.2e-7, p99 2.5e-4, 31/16384 lanes > 1e-3; loss rel 5.2e-6; `δθ` rel 4.6e-5, 1.2e-5, 2.1e-5, 2.4e-4 |
| GPU PRB vs GPU central differences (h = 4e-3, CRN, device loss) | rel 2.0e-4, 3.3e-5, 4.8e-5, 3.8e-3 (θ₃ again the f32 floor) |
| the bush is composited | 277/2304 pixels change with the splats removed |
| throughput, garden 96×96×8 | forward 5.6 ms (13 Mpaths/s), gradient 19.6 ms; release 4.6 / 18.4 ms; CPU compiled forward 665 ms debug, 56 ms release |
| `examples/fit_scene`, 96×96×8, 60 its, same seeds and Adam | θ = [0.208, 0.339, 0.788, 0.998] vs truth [0.2, 0.34, 0.78, 1.0], 60 iterations in 1.3 s |
| beauty, 384×384, 256 spp (64 × 4 spp), depth 6, fitted θ | 1.3 s |

![phase 2b fit](phase2b-fit.png) (`phase2b-loss.csv` is its loss and θ.)

![phase 2b beauty](phase2b-beauty.png)

Deviations:

1. **No reuse of irys-gpu traversal**, because there is none: irys-gpu's
   integrator takes `trace_scene` from the client. The port is ~150 lines of
   WGSL in `wavefront.wgsl`. Once a second client needs it, it belongs in
   irys-gpu as the stock `GeometryModule` for `Bvh` + `TriMesh` + `Analytic`.
2. **One BVH per kind, not a TLAS.** Two trees in one node buffer; no
   instance transforms. `Tlas` is generic over one geometry type, and mixing
   kinds needs either an enum geometry (what `Opaque` is) or per-instance
   kinds in the WGSL. The traversal is the same code either way. *A TLAS over
   instances in phase 2c, piece 1 (one `Opaque` kind per tree).*
3. **GPU analytic prims are spheres and discs**, and splat colour is
   degree 0; `pack` panics on anything else rather than dropping it.
4. **Splats are not differentiated**, and they cast no shadows (there are
   no shadow rays: no NEE yet). The CPU and GPU agree on that model.
5. Splat proxies use the CPU's 3σ box; the peak `t` is reported in f32, and
   a peak computed a hair outside its own box can be culled by the K-buffer
   — deterministic, so it never breaks replay, but CPU and GPU may differ on
   such a splat.

Risks and next:

1. **Splat cost.** The garden's forward is 4.6× the two-prim scene's,
   nearly all of it the compositing rounds (every round re-walks the splat
   tree). Real captures are 10⁵–10⁶ splats: K = 16 with a re-walk per round
   will not hold; 3DGRT's answer is hardware any-hit with a k-buffer, ours
   is probably a larger K in workgroup memory and a per-ray resume point.
2. **Replay cost is still the adjoint** (phase 2a risk 2): the fused
   four-parameter reduction is unchanged work.
3. **The IR's grid limit.** `to_wgsl` dispatches one-dimensional grids, so a
   frame is capped at 65535 × 64 lanes (the beauty frame renders 4 spp at a
   time). A 2D grid split, or chunking lanes, lifts it. *Done in phase 2c,
   piece 2 (chunks).*
4. **f32 offsets** (phase 2a risk 6) bite sooner in a mesh: the fixed
   1e-4 offset is fine at unit scale; a CAD-scale scene needs it scaled to
   the hit distance. *Done in phase 2c, piece 2.*
5. **Splat gradients** need the backward walk per segment and per-splat
   parameter accumulation, which brings back the reduction problem at
   10⁵ parameters: per-splat accumulators can't be one workgroup each.

## Phase 2c: instances, NEE, robustness, the rover (2026-10-17)

Three pieces, all in `irys-diff`; splats are untouched (they work as in 2b,
and refuse NEE).

**Piece 1 — TLAS + instances, NEE and MIS.** `prb::Scene` is now
bottom-level trees (`Bvh<Opaque>`, shared by `Arc`) placed by
`irys_core::Tlas` (`Placement`: tree, object → world, diffuse tint); the
CPU reference traces the `Tlas` itself. The wavefront flattens the top
level into the node array (root 0), every bottom-level tree after it, and
one four-`vec4` record per instance (world → local rows, root, first prim,
tint). The walk is the 2b walk twice: top-level leaves in leaf order, each
instance through its own tree with the ray mapped into local space and
*not* renormalised, so a local `t` is the world ray's and hits compare
across instances with no rescale; the normal comes back by the transpose
of world → local (the inverse transpose of the placement). Shading is
two-sided on both tiers.

NEE: at every surface vertex, three uniforms (always drawn, before the
BSDF's two) pick a light direction from `Env::sample` — `SUN_PICK = ½` the
sun lobe, which is a von Mises–Fisher density up to scale and so sampled
exactly, the rest uniform over the sphere for the gradient — and a shadow
ray (any-hit through the same walk) decides visibility. The power heuristic
weighs it against BSDF sampling; an escape after a BSDF bounce carries
`w_B = p_B² / (p_B² + p_L²)` with `p_B` kept in the path state, and the
camera ray's escape has no rival (`w = 1`). `Env::default()` is 2b's sky to
the bit, so with `Scene::nee = false` every earlier estimator and test is
unchanged.

The replay (derived in full on `prb::trace`). Every term of `L` is a
product of BSDF values: the BSDF-sampled `f_i` of each earlier vertex and,
for a light term `C_j = T_j f_j(ω_L) cos Le V w_L / p_L`, the light-sampled
`f_j(ω_L)` once. Regrouping `∂θ log` by `f`,

```text
∂L/∂θ = Σ_j [ ∂θ log f_j(ω_B,j) · (L − L_j⁺)  +  ∂θ log f_j(ω_L,j) · C_j ]
```

with `L_j⁺` the radiance collected *including* `C_j`. So at a conductor
vertex the replay adds `C_j` to its running `L_j` first, then seeds two
lanes: `δf_L = δL · T_j · s · Le` (`s = cos w_L / p_L`; no division, `f_L`
may be 0) and `δf_B = δL (L − L_j⁺) / f_B`. The light lanes are a second
half of the same graph batch (`Scene::graph_width` = 2 × lanes), so one
eval and one adjoint reduction take both; MIS weights are functions of the
detached pdfs and are constants to `∂θ` like `p` (they sum to one for every
`ω`, so this stays unbiased with a live sampler too). Diffuse vertices are
θ-free: their `C_j` lands in `L_j` and nothing else.

**Piece 2 — robustness.** Origins leave a surface along the normal by
`ε = max(2⁻¹⁵ |x|∞, 2⁻¹⁷ (|o|∞ + t))` on both tiers (`prb::offset`,
`offset_ray`), and hits need only `t > 0`. The first term is Wächter &
Binder's 256 ulps of the position; their per-coordinate integer form
assumes a hit point interpolated from vertices, and ours is `o + t d`,
whose rounding grows with `|o| + t` — a floor at `z = 0` seen from 5 m has a
`z` error of `ulp(5 m)`, not `ulp(0)`; the integer form alone still left
the ×1000 floor speckled — hence the second term. The CPU uses the same
constants in f64, so both tiers offset by the same relative amount.

Frames of any size are one call: `GpuPrb` traces a frame in chunks of as
many paths as one 1-D dispatch of the IR's kernels holds (65535 × 64 graph
lanes; with NEE, 524 280 paths at 4 λ), one submit per chunk after that
chunk's uniforms, and scores the whole frame after with 2-D grids (`flat`).
`L`, `δL` and the image are frame-wide; path state, columns and `δf` are
the chunk's, and a short last chunk's tail is dead paths with harmless
lanes. Scoring rebuilds each path's wavelengths from its generator rather
than reading the (chunk-local) path state. The gradient adds chunk by chunk
in order, so it is still deterministic.

**Piece 3 — the rover.** `examples/rover.rs`: muni's BVR1 (`bvr/cad`) —
the 2020-extrusion frame of `BVR1Frame::generate` (T-slots and bore cut by
vcad's booleans), four `UUMotor::kn6104` hub motors, dropout plates, base
tray and a lid — authored on vcad 0.10's `Part` and tessellated by vcad's
kernel, in millimetres. Each repeated part is one tree: a 500 mm rail
placed 8 times, a 20 mm corner block 8, a 200 mm post 4, the wheel 4, the
plate 4 (31 instances, 18 966 triangles stored, 84 808 placed). The frame
and lid are the anodised-gold GGX conductor (θ), wheels and plates tinted
diffuse, the ground a checkered mesh, sun (κ = 900, ≈ 2° across) and sky
with NEE. The fit starts at θ = (0.6, 0.6, 0.6, 0.6) and recovers the
body's roughness and F0 from a 256-spp target render with 8-spp gradients.

| Gate (debug, M-series, Metal; scene `Scene::pair` unless named) | Result |
|---|---|
| instances | one mesh ball (16 × 32 UV sphere) placed twice — uniform scale, and squashed + turned — on a mesh floor of a second tree |
| replay vs forward, NEE, 64×64×4 × 4 λ | 65536/65536 lanes bit-identical at depth 2 and 6; `δθ` identical twice |
| determinism | two wavefronts × two runs: `L`, image, loss, `δθ` bitwise identical |
| CPU f64 PRB vs CPU central differences (h = 1e-5, CRN, frozen sampler, 16×16×2) | rel 3.1e-9, 7.0e-11, 1.8e-10, 6.0e-12 — the NEE replay formula, light lanes and `L_j⁺` included |
| GPU f32 vs CPU f64, frozen, 32×32×4 | `L` median rel 1.1e-7, p99 4.1e-6, 0/16384 lanes > 1e-3; loss rel 1.8e-7; `δθ` rel 3.6e-6, 7.4e-7, 2.7e-7, 4.3e-7 |
| GPU PRB vs GPU central differences (h = 4e-3, device loss) | rel 9.3e-4, 4.0e-6, 1.2e-5, 1.8e-5 |
| **NEE + MIS vs BSDF only, equal spp** (64×64×4, 48 frames) | per-pixel variance **4.5× lower**; image means agree to 2.4e-3, worst 8×8 block 2.7 σ |
| throughput 96×96×8, NEE off / on | debug: forward 1.75 / 2.08 ms, gradient 11.7 / 18.4 ms; release: 0.87 / 1.21 ms, 7.5 / 14.9 ms |
| ×1000 and ×1e5 scene vs unit (images normalised by their mean) | mean \|Δ\| / mean 2.1e-7 and 1.8e-7, 0/9216 pixels > 20 % darker; the fixed 1e-4 offset at ×1000: 0.21 and 1825/9216 (20 %) |
| chunked (10 chunks, last short) vs one dispatch | `L`, image, loss bitwise identical; `δθ` rel ≤ 1.2e-7; chunked replay bit-exact |
| 1024×1024 × 4 spp, NEE, one call | 9 chunks; image 106 ms (release 85 ms), replay 1.0 s, 16 777 216/16 777 216 lanes bit-identical; mean within 5e-4 of 128² × 64 spp |
| rover: vcad tessellation | 7 parts in 109 s debug, 8 s release |
| **rover fit**, 192×128×8, NEE, depth 5, 80 its, same Adam as 2b | **θ = [0.182, 0.417, 0.768, 0.987]** (mean of the last 20 iterates) vs truth [0.18, 0.42, 0.78, 0.98]; image MSE 2.70e-3, the same as the truth re-rendered (the noise floor); ≈ 62 ms per gradient, debug and release alike (GPU-bound) |
| rover beauty, 768×512, 256 spp (16 calls × 16 spp, 13 chunks each), depth 6 | 2.7 s debug, 2.6 s release |

![phase 2c rover](phase2c-rover.png)

![phase 2c fit](phase2c-fit.png) (`phase2c-loss.csv` is every iteration's
8-spp loss and θ.)

Deviations:

1. **The rover CAD is ported, not imported.** `bvr/cad` is a crate on a
   path dependency of a private checkout, so it cannot be a dependency here;
   `examples/rover.rs` repeats its calls (same dimensions, same booleans) on
   crates.io vcad 0.10, a dev-dependency only (no irys crate depends on
   vcad). The vcad graph is wgpu-free (vcad-kernel-raytrace's `gpu` feature
   is off) but brings crates.io kosm-render 0.2.0 as a second copy beside
   the workspace shim, dev-only. Simplified against the full assembly: plates
   for the dropout mounts, a box lid for the shell, no electronics.
2. **The sun is the environment's lobe**, sampled exactly, not a separate
   area emitter; there are no mesh emitters yet.
3. **The phase-2 gate is half met**: the rover is in a scene and its
   material is fitted from a *render*. A photo needs a camera pose and the
   lighting, and pose needs visibility gradients (below).
4. Wächter & Binder's per-coordinate integer offset is replaced by a scalar
   ε with a distance term (piece 2 above, and why).
5. Chunking is the wavefront's; the IR's grid is still one-dimensional
   (tang unchanged).
6. NEE with splats is refused: shadow rays would have to composite them.

Risks and next:

1. **Visibility gradients.** θ moves no geometry, so the estimator has no
   boundary term. Fitting pose or shape from a photo (the actual gate)
   needs the reparameterisation kernel (warped area, Bangaru+ 2020), and
   then edges of 85 k triangles are the expensive part.
2. **NEE doubles the graph batch**: release gradient 7.5 → 14.9 ms. The
   fused four-parameter reduction (2a risk 2) matters twice as much now.
3. **One submit per chunk**, uniforms by queue writes; at hundreds of chunks
   a dynamic-offset uniform array would put a frame in one submit.
4. **Fixed traversal stacks** (32 top, 64 bottom): `pack` refuses deeper
   trees rather than overflowing; a TLAS of 10⁵ instances will want more.
5. Light sampling knows only the environment; emissive meshes (and the
   rover's LEDs) need a light tree.
6. f32 holds at ×1e5 about the origin; a kilometre scene with millimetre
   parts wants camera-relative positions.

## Phase 2d: visibility gradients, rose's arm (2026-10-17)

The hero subject is now rose full's left arm (the rover stays as an
example). Three pieces: pose gradients through primary visibility, the arm
from rose's own CAD, and a fit that recovers three joint angles and the
finish together. Splats are untouched.

**The choice: edge sampling, not a reparameterisation.** The brief named
warped-area sampling (Bangaru, Li & Durand 2020) or Loubet et al. 2019's
reparameterisation. Both were weighed against what the wavefront is:

- *Warped area* needs a warp field `V` that is continuous across every
  discontinuity and equals its velocity there. With a box pixel filter the
  pixel borders are discontinuities of the integrand too (velocity 0), so
  `V` must also vanish on them — which contradicts matching a silhouette
  crossing a pixel border; the paper gets out of this with a smooth
  reconstruction filter spanning pixels. So: a new image formation (tent or
  Gaussian splatting) on both tiers, auxiliary rays per primary ray, and the
  harmonic boundary-test weights — and its estimate of `V` is a ratio of
  sums, consistent rather than unbiased at finite auxiliary counts.
- *Loubet* is cheaper (one rotation per ray, picked from auxiliary rays)
  but biased by construction wherever more than one discontinuity is in the
  kernel — on an arm of 550 k triangles, everywhere near the silhouette.

The parameters here are rigid instance placements and the camera is fixed,
so the discontinuity set is known exactly: the projected edges of the moving
triangles. Reynolds' transport over that set — the boundary integral both
methods above are ways of *not* evaluating — is directly samplable: Li et
al. 2018's edge sampling restricted to primary rays. It is exact, needs no
change to image formation or to the path kernel (the two sides of an edge
are two ordinary camera paths), and is tested against central differences
to the noise. `vis.rs` derives it in full:

```text
∂I_p/∂φ ⊇ ∫_{E ∩ p} (f(s − εn) − f(s + εn)) · v_n(s) dℓ(s),   v_n = n · ∂π(X(s))/∂φ
```

- **Candidates.** Every triangle edge of every moving instance, projected
  and clipped to the frame. `Edges` welds each tree once (positions to
  1e-7 of the diagonal) to find each edge's neighbour: shared with a
  coplanar triangle → never a candidate; otherwise an importance —
  silhouette (one face front, one back) and boundary 1, a crease between
  front faces `clamp(4 (1 − cos), 0.02, 1)` (a cylinder facet ≈ 0.08, a
  right angle 1), back-only 0.02 (winding is a hint: nothing gets weight 0,
  so it stays unbiased). `n` samples per call, stratified over
  `Σ ℓ w`, each weighted `Σℓw / (w_e n)`.
- **The two sides** are two full paths through `s ∓ εn`, `ε = 10⁻³ px`,
  traced by the existing wavefront in a new *point mode*
  (`Scene::set_points`, WGSL binding 26): path `i` leaves through a given
  camera-plane point and is keyed `i / 2`, so a pair shares wavelengths and
  every uniform after the camera — the difference is common random numbers.
- **Ownership.** Each triangle owns its edges: the inner ray must hit that
  triangle, *or a coplanar one of the same face* — CAD tessellations are
  full of slivers thinner than ε, and without this the vcad plate's
  edge term came out 60 % high; a test now holds it. Where the outer ray
  lands on the same instance within 10⁻⁴ of the distance (a crease, which
  the neighbour samples from its side), each side takes half. A silhouette's
  back face sees its inner ray hit the front face (not coplanar) and is 0.
- **Velocity.** `X(s)` is the edge line's closest approach to the camera ray
  through `s`, carried back to object space; `v = J_π · ∂X/∂φ`, with
  `∂X/∂φ` from central differences (h = 1e-6) of the rig's closed-form
  placement — a derivative of the rig, exact to ~1e-10, not of the image.
- **What is left out.** The interior derivative of `f` at fixed `x` (shading
  at the hit turning with the link) and every secondary discontinuity
  (shadows, reflections). Against the sky at depth 1 the edge term is the
  whole derivative; shaded, it is not (below), and the fit converges anyway
  because the target is a fixed point of any gradient linear in `δI`.

GPU: the edge term is host bookkeeping (sampling, ownership tests on the CPU
TLAS, weights) around one point-mode forward pass on the device;
`GpuPrb::update_geometry` re-uploads the top-level tree and instance records
after `Scene::place` (buffers carry room for the tree to change shape) and
is tested bitwise against a rebuild.

**The arm, from the assembly.** `examples/rose/arm.rs` reads rose's
canonical assembly, `full/cad/rose-full.loon` (generated by
`full/rose_full_loon.py`): its `[use links.X [..]]` / `[use actuators [..]]`
forms name the modules and its 160 `[root NAME material]` forms are the
parts, each already in the robot's frame at zero joint angles — the
assembly carries no transforms, so placing a root is reading it. Each root
resolves to its module's `[let NAME EXPR]` and is evaluated onto
`vcad_kernel::Solid` (0.10, a dev-dependency like `vcad`; the chest panels
are `extrude`/`sketch` forms only the kernel exposes) over the vocabulary of
vcad's `hardware/rose-full/plates.loon`, ported call for call because
vcad-loon is not on crates.io, plus `translate` and `rotate`. The joints come
from the same files: each arm joint is its actuator envelope in
`actuators.loon` (`[act-AX d lo hi cu cv]` — axis `AX` through `(cu, cv)`:
shoulder pitch `y` through (·, 462), roll `x` through (255, 362), elbow `y`
through (·, 235.2), wrist `z` through (0, 255)); what rides each joint is
`rose_full_links.py`'s role table (shoA is the pitch output and roll stator,
shoB the roll output and elbow stator, elbA the elbow output and wrist
stator, the hand the wrist output), so each actuator rides the link its
stator bolts to. The robot is lifted 1029 mm so its soles are on the ground.
Trees: the left arm's four links (shoA 223 304 triangles, shoB 145 818, elbA
175 934, hand 5 104) and four actuators, then the rest of the body one tree
per link module and side (31 trees; one 1.8 M-triangle body tree was 63
levels deep, past the WGSL traversal stack): 2.4 M triangles. A cold cache is
199 s release on every core (vcad's booleans on bored plates, up to 200 s for
one plate), then 50 ms. Sheet is clear-anodised aluminium (GGX
`θ = (0.30, 0.86, 0.88, 0.90)`), actuators at `actuators.loon`'s 0.10 grey.

*What the first version got wrong.* It read the same four link modules and
the right pivots, but hand-assembled them into a scene of its own: the arm
moved to `y = 0` on an invented post, no torso, and a target pose rolling
the arm 16° *inward* — through where the chest is — with the forearm
pitched 86° forward and seen end-on from the camera. Without the body
nothing said "arm", and the forearm's two 3 mm fork plates read as bars
sticking out sideways. (Nothing came from `mini/` or `mega/`: mini's arm is
`uarm`/`farm` cylinders at x = ±132 and mega has no plate vocabulary at
all.) `examples/rose_check.rs` now holds the reading to rose's own render:
an exact orthographic raster from vcad-render's `--view iso` axis and
framing (`--fill 0.92 --auto-aspect`, 1400 px) comes out 437 × 1400 like
`full/render/rose-full-iso.png`, its silhouette has IoU 0.980 with that
image's alpha (bounding boxes within 1 px), and every one of the arm's
26 970 pixels lands inside rose's silhouette; irys's own render from a
pinhole 40 m out on the same axis overlays it. The target pose now rolls
the arm 16° *out*.

| Gate (M-series, Metal) | Result |
|---|---|
| box vs sky, depth 1, 40×30, edge 24 × 8192 samples vs CD over 48² fixed points per pixel (h 0.05) | rel 6.7e-2, 1.1e-1, 4.1e-3 — within 3.3, 0.8, 1.3 σ of the edge estimate (the CD reference is good to ~1 %) |
| vcad rounded plate (5 110 triangles) vs sky, same | rel 6.7e-2, 9.2e-4, 2.4e-3 (z 0.7, 0.05, 0.9) |
| edge term on the device (f32) vs CPU f64, same samples, depth 3 | rel 1.8e-6, 7.3e-7, 2.6e-7 |
| point-mode replay | 65536/65536 lanes bit-identical; every 2a–2c replay/determinism test unchanged |
| `update_geometry` vs rebuilding the wavefront | image bitwise identical |
| **arm mask** (whole robot, beauty camera, depth 1, 96², 64 × 16384 samples vs CD over 32² points, h 0.01) | edge (−6.89 ± 0.97, −9.26 ± 0.22, +4.46 ± 0.49) vs CD (−5.72, −8.83, +4.87): within 1.2, 2.0, 0.85 σ |
| arm shaded (depth 5, NEE) | edge (+48.8, +7.4, +3.5) vs CD (+91.7, +24.4, −40.5): the interior and shadow terms dominate — measured, not hidden |
| arm GPU vs CPU edge term, mask | \|Δ\|/\|g\| 5.3e-8, 8.2e-8 |
| rose beauty, the whole robot, 768² × 256 spp, depth 6 | 4.1 s release (19 chunks per call) |
| **rose fit** (the arm alone, 192², 8 spp, depth 5, 16 384 edge pairs, 300 its, pose alone for 100) | φ = (−25.76, +15.97, −60.23)° vs truth (−25.78, +16.04, −60.16)° from (−8.6, +3.4, −37.2)°: error ≤ 0.07°; θ = (0.293, 0.844, 0.864, 0.885) vs (0.30, 0.86, 0.88, 0.90); image MSE 2.081e-3 vs truth re-rendered 2.062e-3; 87 ms θ gradient + 75 ms edge term per iteration, 54 s total |

![phase 2d rose](phase2d-rose.png)

![phase 2d fit](phase2d-fit.png) (`phase2d-loss.csv`: every iteration's
8-spp loss, the three angles, θ.)

Deviations:

1. **Edge sampling instead of warped area or Loubet**, for the reasons
   above. Primary visibility only, rigid instance parameters only.
2. **The interior pose derivative is not computed.** Shading at a hit
   point turning with its link, and shadows/reflections moving, are absent
   from `∂/∂φ`; the shaded arm's gradient is wrong by O(1) in magnitude and
   sign on two of three angles. The fit still lands on the truth — the
   silhouette term is zero there and points the right way near it — but a
   fit that needs shading cues for pose (a low-contrast silhouette) would
   not.
3. **The loon is read, the plates vocabulary ported** (vcad-loon is not
   published). The fit stages pose before finish (100 pose-only
   iterations). On the beauty camera, with both live from the first step,
   `F0` saturated at 1 to brighten a misplaced arm; whether the staging is
   still needed on the fit's camera was not separately tested.
4. **The fit is the arm alone, not the robot.** It runs on the left arm's
   eight trees, placed exactly as in the assembly, over a plain floor, from
   front-left and a little below so the forearm reads against the sky (11°
   half-angle). With the body in frame, or the forearm against the checker,
   the 3 mm fork plates give the silhouette term too little to pull on and
   the elbow stalled 27–68° short (the missing interior term, deviation 2,
   is what would have to carry it); pitch and roll converged either way.

5. **"From a photo" not done.** The obvious non-irys target is
   `vcad-render`'s output from `rose/render.sh`, but it is an auto-framed
   (`--fill --auto-aspect`) orthographic flat/line render with its own
   lights: matching it needs an orthographic camera in irys, a camera-pose
   parameter, and a lighting model — none of which this phase has. Not
   attempted rather than shown badly.

Risks and next:

1. **Variance.** 76 % of the arm's edge samples are hidden (inner ray
   blocked); per-run rel σ is 0.8–2.4 on the mask. A visibility-aware
   importance (a cheap ID-buffer pass to keep only edges that border a
   different instance or the sky) is the obvious next 10×.
2. **The interior term.** Differentiating the hit's shading frame with the
   link's motion (the GGX graph takes the cosines as columns: `∂cos/∂φ`
   feeds the adjoint as extra lanes) — then shadows need edge sampling of
   *secondary* rays, which is where warped-area's per-ray machinery starts
   to pay.
3. **Host-side cost.** The 87 ms edge term is mostly host work —
   projecting 227 k candidate edges and two CPU ownership rays per sample,
   around one small device pass (the split is not measured). It belongs on
   the device, where a TLAS walk already exists.
4. **Camera pose** is the next parameter (the camera is a moving "instance"
   for every edge), and then a real photo with a measured camera.
5. vcad tessellation density (550 k triangles for three sheet-metal links)
   costs traversal and edge count; a coarser tolerance for rendering would
   cut both.

## Phase 2e: the interior term, cheaper edges, the camera (2026-10-17)

Continues 2d's pose gradients. Scenes are the tests' own (`tests/phase2e.rs`):
an arm of boxes — a static post, a conductor upper link, a diffuse forearm,
each pre-rolled so no face is axis-aligned, gaps at both joints — at depth 4
with NEE, rigged by shoulder pitch, shoulder yaw and elbow.

### 1. The interior term (`interior.rs`)

`∂I/∂φ` at fixed random numbers, where the integrand is smooth: the hit
point moving on a moving triangle, its normal turning, the next direction
(drawn in the hit's frame) turning with it, and every cosine, GGX value,
pdf, MIS weight and sky lookup downstream. **Hand-derived forward mode**: a
dual number (`D`, ≤ 8 tangents) re-traces the frame with the primal trace's
control flow — Möller–Trumbore against the triangle the primal hit, corners
as duals; GGX visible-normal sampling and eval in duals; NEE and the
environment in duals. Not the IR: the GGX graph differentiates to `θ` with
the columns as data; making it take `∂cos/∂φ` lanes would need a second
graph for one factor of many, and the duals cover all of them. The dual
trace's values equal the forward pass's `L` to 9e-15.

*Attached frames.* The renderer's `basis(n)` flips where `n_z` crosses 0 — a
discontinuity in `φ` no edge sees. The derivative taken is that of the
estimator whose frame on a moving instance is `R(φ)R(φ₀)ᵀ basis(n(φ₀))`:
equal to the rendered one at `φ₀`, a valid estimator of the same `I(φ)` for
every `φ`, so its derivative is unbiased from the very same paths. Rigid
placements assumed (the frame's angular velocity is the skew part of
`L̇L⁻¹`).

*A 2d bug, found by this gate.* 2d sampled a crease between two front faces
from both sides and halved where the outer ray landed next to the edge
point. When the grazing side's ownership test failed (its inner ray at
10⁻³ px onto a nearly edge-on face) half the jump was lost. A mask has no
jump at creases, so every 2d gate passed; shaded, the edge term on the arm
was +15.7 against a true total of −3.0 (z ≈ 25), and this — not the missing
interior term — is what 2d's "shaded arm disagrees in sign" was. Now a
crease has one owner, the face meeting the eye more squarely, at full
weight, and the outer ray's `f` is whatever it lands on.

| Gate (CPU f64; CD over 16² fixed points per pixel, h 0.01, 4 seeds) | Result |
|---|---|
| dual values vs `prb::forward` (interpreter), 3456 lanes | worst rel 9.3e-15 |
| per-lane tangent vs CD at one seed, translations (frames fixed), depth 4 | 92/92, 20/20 moving lanes agree to 1e-4 |
| per-lane, rotations, depth 1 + NEE | 44/44, 44/44, 12/12 |
| **arm vs sky, depth 4, NEE**, 48×36, edge 24 × 8192 pairs + interior 24 × 4 spp | sum (−3.17 ± 0.67, −3.65 ± 0.28, −0.06 ± 0.18) vs CD (−5.41 ± 1.35, −2.83 ± 0.94, −0.26 ± 0.54): z 1.5, 0.8, 0.4. Edge alone z 2.5, 0.2, 2.0; interior (−1.49, −1.02, +1.31) |
| same arm over a floor it shadows (ignored test, measured) | sum (−31.1, +8.5, −6.3) vs CD (−36.9 ± 1.4, +11.7 ± 2.1, −8.7 ± 1.3): z 3.6, 1.5, 1.9 — the floor shadow moving is ~15 % of φ0 and neither term sees it |
| a conductor slab filling the frame (scratch, not kept) | edge + interior (523, 90, 11.3) vs CD (519 ± 12, 94 ± 8, 9.0 ± 2.7) |

Deviations: CPU only (the device has no dual trace); triangles only on the
path (analytic prims refused), no splats; the checker's edges ignored.
Interpenetrating instances are a discontinuity (their intersection curve)
that is no triangle's edge — the first arm, with links sunk into each other
and into the post, disagreed by 10σ before the gaps; rose's links bolt
plate-to-plate, so this matters there.

### 2. Visibility-aware edge importance (`vis::Visibility`)

An ID + depth buffer (one ray per pixel centre, 0.5 ms at 48×36) and two
changes to the candidates' importance, both unbiased (nothing gets weight
0): a silhouette is sampled from its **front face only** — 2d gave the back
face's side weight 1, and its inner ray always lands on the front face and
scores 0, so half of every silhouette sample was wasted — and each
candidate, probed every pixel along its screen length, keeps
`HIDDEN_WEIGHT = 0.01` of its importance over runs where it is deeper than
the farthest surface in the surrounding 3 × 3 pixels. Visible creases keep
their crease weight: the brief said to drop front-front edges, but shaded
they carry real jumps (piece 1's bug is exactly that term), so dropping
them would bias every shaded gradient.

| Plate arm (links are three parallel plates in one tree, the post in front), 48×36, 24 × 4096 pairs | 2d importance | 2e | variance ratio |
|---|---|---|---|
| depth 1, visible samples | 35 % | 57 % | φ 2.2, 1.4, 2.3 (geo-mean **1.9**) |
| depth 2 + NEE | 35 % | 57 % | 1.8, 1.2, 2.1 (geo-mean **1.7**) |

Means agree to |z| < 1.6. **Not the 10× 2d expected.** Removing waste
caps the gain at `1 / visible fraction`; what is left invisible (43 %) is
mostly edges that coincide with a nearer edge to within a pixel (stacked
plates share their outline), which no pixel-resolution buffer can tell
apart, and the remaining variance is per-sample: `v_n · Σℓw / w_e` swings
with crease weight and edge speed, and shaded the two paths' difference is
itself noisy. A velocity-aware importance (`∝ |v_n|`, per parameter) is the
next lever. The rose arm itself was not re-measured (its files are off
limits here).

### 3. The edge term's host work on the device (`gpu/edges.wgsl`, `GpuEdges`)

`edges.wgsl` is appended to the wavefront's source and walks its tree.
`edge_cand` (a thread per moving triangle edge: project, facing, crease
owner, importance, clip, `len × w`), a fixed-order prefix sum (`scan_block`
sums 1024 slots sequentially per thread, `scan_tops` the block totals in
one thread, `scan_fill` writes the inclusive sums — no atomics, so the same
weights give the same bits), and `edge_sample` (a thread per sample:
stratified `u`, binary search, edge point, closest-approach material point,
the ownership ray through the device's TLAS with the coplanar-sliver rule,
the pinhole Jacobian, `v_n` per parameter from the rig's derivative
matrices). The samples land straight in the wavefront's point buffer; the
host keeps the rig (placements at `φ ± RIG_H`, a few matrices) and the
final `O(n)` weighted sum. The host now draws each sample's `u` from its
own generator (`Rng(seed ^ "edge", i)`), as the device does, so the two
draw the same samples.

| Gate (M-series, Metal) | Result |
|---|---|
| device vs host samples, plate arm, 16 384, both importances | 98.9–99.6 % of points within 1e-5 (the rest: f32 cumulative sums moving a `u` across a segment end); weights 16384/16384 within 2e-3; visible counts equal |
| edge term from device samples vs host samples, both traced on the device | rel ≤ 2.1e-3 |
| two device draws | bit-identical points and weights |
| **split, 522 k triangles, 16 384 pairs, 192², depth 5 (release)** | before: host sampling **102.5 ms**, device pairs 2.7 ms, host sum 0.45 ms. After: host 0.04 ms, device sampling **3.0 ms**, device pairs 3.9 ms, host sum 0.43 ms — 105 → 7.4 ms |

Deviations: the ID-buffer demotion (piece 2) is host-only; the device does
2d's importance or 2e's front-only silhouettes. Velocities still come from
central differences of the rig's closed form on the host (exact to 1e-10;
f32 on the device).

### 4. The camera as a parameter (`vis::CameraRig`, `Tangents::camera`)

`c = (t, ω, δ)`: eye translated by `t` (world), the frame turned about the
eye by `R_z R_y R_x(ω)` (world axes), the half field of view widened by `δ`
— seven parameters about a base pinhole (`Scene::set_camera` is new). Edge
term: every instance's edges are candidates (front-only silhouettes), and a
fixed world point's screen velocity is the pinhole's derivative (central
differences of the closed form). Interior: the eye, axes and `tan` of the
half-angle are duals; the primary ray and everything downstream follow.
Edges with a corner behind the eye are now kept on both tiers (the
triangle's side is read from a point just inside it next to the edge); 2d
dropped every triangle with a corner behind the eye, which a camera near a
floor makes common. Edge ends behind the eye are still dropped (no near
clip).

| Gate | Result |
|---|---|
| per-lane, all 7 parameters, arm on floor, depth 4, NEE, one seed | rotations and fov: 376/376, 376/376, 44/44, 376/376 moving lanes agree to 1e-4; translations move no lane (flat faces, directional light, no second object on the path) |
| **two boxes, conductor + diffuse, apart, depth 3 + NEE**, 48×36, 24 × (8192 pairs + 4 spp) vs CD (16² points, h 0.05 / 0.004 rad / 0.01 rad, 4 seeds) | z −0.54, +1.01, −2.34, −0.40, −0.59, +0.96, −1.57; e.g. fov: edge 55.5 + interior 89.9 = 145.4 ± 1.3 vs 150.1 ± 2.7 |
| arm mask, depth 1 | z −0.52, +1.69, +1.72, +1.25, −2.74, −0.28, +0.80. The reference itself moves by a few per cent with the rotation step (c4: 6.5, 5.3, 4.3 at h = 0.01, 0.004, 0.025 rad) |
| shaded arm, depth 4 + NEE (ignored test, measured) | translations and the roll disagree by 5–14σ (c0: 5.16 ± 0.25 vs 0.57 ± 0.21) — the arm's shadows on itself and the post are image discontinuities that are no triangle's edge, and every one of them moves when the camera does |

### 5. Shadow edges: deferred

Not built. What the gates measured it is worth: ~15 % of the shoulder
pitch on the arm over a floor (−31.1 vs −36.9 ± 1.4), and for the camera on
the shaded arm it dominates the translations (c0: 5.2 vs 0.6). Why not now:

- The light is an environment (a vMF sun lobe plus the sky), so a shadow
  edge is not a curve on a light source but the occluders' silhouettes as
  seen from each shading point, over the whole sphere. Li et al. 2018's
  secondary edge sampling needs, per shading point, a sample of a point on
  an occluder silhouette edge with a density over all edges of the scene —
  a per-vertex edge hierarchy (their 3D edge tree with cone bounds), not
  the per-frame screen-space list pieces 2–3 built.
- Each sample needs its own pair of shadow rays (`ω ± ε`), a weight with
  the edge's motion projected onto the unit sphere around the shading
  point, and the path's throughput up to it — so the replay's per-vertex
  state, or a third pass. On the device that is a new kernel per bounce.
- With a smooth environment the visibility integral is smooth in `x`; its
  per-sample discontinuity is an artefact of point-sampling the light.
  Warped-area reparameterisation of the shadow ray alone (Bangaru et al.
  2020, restricted to NEE) is the cheaper candidate here, because a shadow
  ray has one discontinuity family and no image-formation change.

The camera and the floor arm stay as measured (ignored) tests, so the term
has its gates waiting.

### Phase 2e risks and next

1. **Shadow edges** (above): the largest error left on any shaded scene
   with contact or cast shadows, and for every camera parameter.
2. **Interpenetration.** Intersection curves of overlapping instances are
   discontinuities no edge list contains. Rose's links bolt plate to plate;
   if they overlap in the tessellation, the edge term is biased there. A
   check (instances' meshes intersecting) belongs in `Edges::new`.
3. **Variance.** Piece 2 gave 1.7–1.9×, not 10×: per-sample variance now
   dominates. Importance ∝ `|v_n|` per parameter, and fewer, fatter crease
   samples, are next.
4. **The interior term is CPU f64 only** (~0.25 s for 48×36×4 spp × depth 4
   in debug); the fit on the device would need a dual (or adjoint) trace in
   WGSL, or the IR graph extended to take the shading frame's tangents.
5. **The CD references** are good to a few per cent at best (the rotation
   step moves them by that much); gates at 3σ of combined error are as
   tight as they can be made without a much costlier reference.
6. The rose arm itself was not re-run (its files belonged to another
   agent); the crease bug fixed in piece 1 changes 2d's shaded rose
   numbers (`edge (+650, +479, +167)`), which should be re-measured.

### Phase 2e: rose in context (2026-10-17)

The question: with 2e's pose gradient (edge term on the device,
`GpuEdges` with front-only silhouettes, plus the interior term on the CPU),
does the rose arm fit converge with the body behind the forearm? **No.**
`examples/rose_fit.rs` now runs the whole robot from rose-full.loon on the
checker floor by default, from 2d's fit camera so the only change from 2d is
what is behind the forearm (the torso); `--control` is 2d's arm-alone
scene, `--edge-only` 2d's gradient. Same start, truth, θ, and schedule (300
iterations, pose alone for 100); interior at 1 spp per iteration.

| Run (192², 8 spp, depth 5 + NEE) | φ (last-40 mean) vs (−25.78, +16.04, −60.16)° | θ vs (0.30, 0.86, 0.88, 0.90) |
|---|---|---|
| control, 2e gradient | (−25.78, +16.02, −60.16) | (0.293, 0.845, 0.864, 0.886) |
| control, edge only (2d) | (−25.78, +16.00, −60.17) | (0.293, 0.845, 0.865, 0.886) |
| **context, 2e gradient** | (−2.66, −40.74, +0.49): diverges | (0.432, 0.678, 0.679, 0.681) |
| context, edge only | (−9.19, +60.34, +13.37) | (0.289, 0.818, 0.834, 0.846) |
| context, camera looking down on the checker | (+4.09, +36.43, +12.78) | (0.318, 0.833, 0.847, 0.862) |
| context, no sun (sky only) | (−25.81, +16.21, **−20.12**) | (0.262, 0.804, 0.823, 0.842) |
| context, depth 1 + NEE | (−20.31, −15.91, −79.08) | (0.346, 0.702, 0.713, 0.723) |
| context, 16 px box-blurred pose loss for the first 100 (schedule change) | (−9.09, +57.00, +17.07) | (0.287, 0.816, 0.832, 0.843) |

Per iteration, context (release, M-series): θ gradient 122 ms, edge term
93 ms (including re-uploading the whole robot's top-level tree to the pairs'
wavefront and tracing 16 384 pairs at depth 5; `GpuEdges`'s own draw is
the 3 ms of piece 3), interior 15 ms (36 864 paths, 1 spp, depth 5, 16
threads; 4 spp is 67 ms and changes nothing), 94 s for 300. Control: 87 /
21 / 7 ms. The interior term was not too slow at full resolution.

Why, measured with `--check` (the pose gradient at the start pose, θ at
truth, against central differences of the 256-spp MSE, h = 0.01):

1. **Discontinuities seen through a bounce.** Context with the sun, 256
   seeds: edge + interior (−0.099 ± 0.010, −0.024 ± 0.004, −0.052 ± 0.006)
   vs CD (+0.017 ± 0.002, −0.004 ± 0.001, +0.010 ± 0.002): z −11, −5, −9,
   the wrong sign on pitch and elbow. With no sun it agrees (z 1, 1, 2); at
   depth 1 with NEE (so primary cast shadows are in the image) it agrees (z
   −1, 2, −1); at depth 2 with NEE it is already off (z −3, −3, −2). So the
   missing term is not primary visibility and not the shadows the arm casts
   on what the camera sees directly: it is the arm moving *in reflection* in
   the polished torso (and the torso in the arm), and shadow boundaries
   seen through that reflection. Neither 2e term samples those. The control
   has the same bias in kind (z 6, 6, 6) but there the target is still the
   fixed point and the reflections are of sky; in context the bias is 6× the
   true gradient. The per-sample distribution is heavy-tailed on top of that
   (median −0.098, p90 +0.020, max +0.66 on pitch).
2. **The loss itself.** Even where the gradient is right, the image L2 along
   the elbow (pitch, roll at truth) has a basin only ±5° wide around −60°:
   with the sky only, a local minimum at −20° (MSE 2.81e-4 vs 3.01e-4 at
   −40°), which is exactly where the no-sun fit stopped with pitch and roll
   exact; at depth 1, one near −40°. Aluminium forearm over aluminium torso
   gives the edge term little contrast to pull on away from the target
   silhouette. A 16 px blur makes the sun-lit elbow scan monotone from −10° to
   −60°, but with the biased gradient the blurred fit diverges anyway.
3. Not interpenetration: the arm's mask gradient (depth 1, sky only,
   control) is 27 % low on pitch (z −7) — a sign of 2e's interpenetration
   risk or of stacked plates — but it is the same in both scenes and the
   control converges.

Fixed on the way: `edges.wgsl`'s sampler could land past the last
cumulative sum (f32) on a trailing zero-importance slot and return `x / 0`;
one such sample in a few thousand iterations NaN'd every fit. It now scores
0. Next: secondary edges (reflection and shadow) for the conductor, or
warped-area reparameterisation of the secondary rays (5. above), then the
basin, by a pyramid or a silhouette/feature loss.

![phase 2e rose in context](phase2e-rose-fit.png) (`phase2e-rose-loss.csv`:
every run, every iteration, both pose terms and the time split.)

## Phase 2f: secondary visibility, rose in context (2026-10-17)

The question 2e left: does the rose arm fit converge with the whole robot,
the checker floor and the forearm in front of the polished torso? **Yes,
with 600 iterations**: φ = (−25.73, +16.07, −60.23)° vs truth (−25.78,
+16.04, −60.16)°, θ = (0.303, 0.855, 0.876, 0.895) vs (0.30, 0.86, 0.88,
0.90), image MSE 8.57e-4 against the truth re-rendered at 8.60e-4. Same
start, truth, θ, learning rates and schedule as 2e; the difference is the
pose gradient. Four pieces, each a commit.

### 1. Secondary edge sampling (`secondary.rs`, `interior::interior_with`)

A path vertex `x` sees discontinuities of its own: an occluder's outline
across the shadow ray, the arm's outline in the torso's reflection, a crease
seen through a bounce. The forward pass estimates the vertex's direction
integral twice — a light sample `ω_L` (fixed in the world at fixed random
numbers) and a BSDF sample `ω_B` (drawn in the vertex's attached frame from
`wo`) — with integrands `F_L = g Le V w_L` and `F_B = g C [w_B on escape]`.
Both jump across every edge `x` sees, and Reynolds gives each a boundary
term with the edge's angular speed *relative to its sampler's map*:

```text
∂/∂φ ⊇ ∮ ΔF_L · n·V + ΔF_B · n·(V − ω̇_B) dℓ_ω,    V = (I − ωωᵀ)(Ẋ_p − ẋ)/r
```

`ω̇_B` is the velocity of the direction the BSDF sampler draws at the `u`
that draws the edge point: the GGX visible-normal sampler (and the cosine
one) is inverted for that `u` and re-drawn in duals (round trip to 1e-7,
unit test). One edge point per vertex from `Secondary` — one edge BVH per
bottom-level tree in object space (built once: 1.34 M candidate edges for
the whole robot in 0.65 s), nodes weighed `Σℓ / max(d², r²)`, leaves by 2e's
owner rules seen from `x` — by one-sample MIS over three importances
(geometry; the sun lobe over each node's cone of directions; at a
conductor, its mirror lobe), exact density, the point along the edge
piecewise-constant in 16 pieces. The two sides are continuation paths from
`x` with common random numbers; the inner one must land on the owner (2e's
coplanar-sliver rule). Candidates are the moving instances, and every
instance when `x` or its `wo` moves. CPU f64, from the interior term's own
paths; `IRYS_SEC_N` edge samples per vertex (4 in the fits).

**The choice, measured** (`secondary_estimators_compared`, ignored): on 2e's
arm over a floor, the light sample's share of the boundary term by edge
sampling and by warped-area reparameterisation of the shadow ray (Bangaru
et al. 2020: `K` auxiliary rays from a vMF around it, harmonic weights
`1/(D² + B²)` with `B` the hit's angular distance to its triangle's
candidate edges, the warp's divergence by central differences), 400 seeds
of the same paths:

| estimator | φ0 | φ1 | φ2 | time / estimate |
|---|---|---|---|---|
| edge sampling, light share | −5.68 ± 0.47 | +3.51 ± 0.56 | −0.33 ± 0.16 | 2.5 ms |
| warp K=4, κ=400 | +3.3 ± 4.3 | +7.8 ± 6.4 | +7.1 ± 4.9 | 2.4 ms |
| warp K=8, κ=400 | +0.9 ± 2.7 | +5.2 ± 3.7 | +3.8 ± 2.4 | 2.8 ms |
| warp K=8, κ=100 | +17.6 ± 13 | +22 ± 16 | +18 ± 16 | 3.8 ms |
| warp K=8, κ=1600 | −6.7 ± 1.8 | +8.8 ± 2.5 | −0.1 ± 0.7 | 4.1 ms |
| warp K=32, κ=400 | −4.1 ± 1.9 | +8.0 ± 2.4 | +1.2 ± 0.9 | 8.3 ms |

The warp's best setting has 14–19× edge sampling's variance at 1.6× its
cost, heavy tails (an auxiliary ray crossing an edge jumps its weight) and
a bias that moves with `κ`; edge sampling is unbiased by construction.
Edge sampling, and on the CPU: the device has no dual trace (2e risk 4),
and at 53 ms per iteration (below) the CPU is not the bottleneck.

| Gate (`tests/phase2f.rs`, release; CD over 16² points per pixel, now Cranley–Patterson shifted per seed, 8 seeds) | Result |
|---|---|
| arm over a floor it shadows (2e's ignored gate), depth 4 + NEE, 24 seeds | edge −29.5, interior −1.5, secondary −4.2 ± 1.8: sum (−35.3, +12.6, −8.3) vs CD (−37.1 ± 1.4, +15.1 ± 1.5, −8.4 ± 0.6): **z +0.77, −1.07, +0.06** (2e alone +3.8, −3.9, +3.4) |
| the same arm against the sky (2e's gated scene) | z +0.61, −1.12, +0.81 |
| shaded arm under camera motion (2e's ignored gate), 7 parameters | **z +0.10, −1.07, −2.64, −2.16, −0.03, −0.38, −1.09** (2e alone +16, +2.3, −0.6, −0.3, −4.8, −9.2, −9.9) |
| **rose in context, sun, `--check` at the start pose** (256-spp MSE, h 0.01), 64 seeds | edge −0.087 ± 0.009, interior +0.0046, secondary +0.149 ± 0.065, curves +0.0005: sum (+0.067 ± 0.067, −0.001 ± 0.030, +0.010 ± 0.034) vs CD (+0.0167, −0.0037, +0.0098): **z 1, 0, 0** (2e: −11, −5, −9) |
| same, 256 seeds | secondary +0.014 ± 0.030: z −3, −2, −4 |
| rose, loss only on pixels ≥ 3 px from the arm (`MASK=3`: what is left is the arm's shadows and reflections), 64 seeds | secondary (−3.8 ± 6.8, −17.7 ± 4.1, −5.6 ± 1.6)e-4 vs CD (−5.9 ± 1.2, −9.7 ± 0.8, −5.7 ± 0.5)e-4: z 0, −2, 0 |

The rose gate is met at 64 seeds and missed at 256: the secondary term on
rose is heavy-tailed (per-seed σ ≈ 30× the gradient at 4 spp), so its
stderr is not a reliable error bar and neither number is a clean pass. What
is solid: off the arm (`MASK`) it matches; on the arm the biggest samples
are vertices on one aluminium plate a few millimetres from another link's
edge (`IRYS_SEC_DUMP`: r ≈ 4 mm, |V| ≈ 100, pdf ≈ 5e-6 per mm) — joints and
bolted plates in contact, where `m·V ∝ 1/r²` and the node bound caps at the
node's size.

*Two 2e residuals found on the way.* (a) **`GpuEdges` is biased low on
rose**: on the control mask (depth 1, sky only) it gives 1.84e-3 on pitch
where host samples under every importance (2d, front-only, ID buffer;
traced on the device or in f64 on the host) give 2.43–2.57e-3 and central
differences 2.62e-3 (h 0.003) to 2.82e-3 (h 0.01). The device refused real
coplanar slivers (f32 tolerances; fixed, 5616 → 5869 of the host's 6068
visible samples) and still weighs ~8 % of samples 5–20 % differently;
not found. 2f's rose runs use host samples (`--vis`). (b) The CD reference
on rose moves by ~7 % with the step (2.62 vs 2.82e-3); the box gates' did
with its fixed grid points (now shifted per seed).

### 2. Interpenetration (`secondary::pierces`, `vis::curves`, `EdgeSamples::for_curves`)

Rose's arm does interpenetrate: at the start pose 6 instance pairs (shoA↔shoB,
shoB↔elbA, shoA into two body trees), a few dozen edge crossings, 5 713
triangle-pair intersection segments, 143 px of curve on screen. The
intersection curves are now sampled like edges (two rays either side, each
on one of the two instances at the curve's depth; normal speed from
re-intersecting the two planes at φ ± h). On a box arm sunk 6 cm at both
joints the curves carry −2.89 of φ0's −3.55 and the total gates (z +1.2,
+0.5, −1.9). **On rose they are 0.1–1 % of the gradient** (mask: +2.9e-6 of
2.8e-3 on pitch; in context +5e-4 of 0.017) — not 2e's 27 % mask deficit,
which was (a) above. They are in `--check`, not in the fit. Not done: the
curves as seen through a bounce (secondary rays), where a curve point near
`x` is a `1/r²` singularity.

### 3. Basin width (`--pyramid`, `SCAN`)

MSE along the elbow (pitch and roll at truth, 4 × 32 spp), per level of a
block-average pyramid and per box blur:

| elbow | −80 | −70 | −65 | **−60** | −55 | −50 | −45 | −40 | −30 | −20 | −10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 3.85e-3 | 4.22e-3 | 2.95e-3 | **9.3e-4** | 2.52e-3 | 2.81e-3 | 2.70e-3 | 2.85e-3 | 3.38e-3 | 3.45e-3 | 3.93e-3 |
| pyr2 | 2.38e-3 | 2.32e-3 | 1.38e-3 | **2.5e-4** | 1.23e-3 | 1.68e-3 | 1.57e-3 | 1.70e-3 | 2.20e-3 | 2.25e-3 | 2.45e-3 |
| pyr4 | 1.63e-3 | 1.43e-3 | 8.42e-4 | **6.0e-5** | 7.23e-4 | 1.14e-3 | 1.07e-3 | 1.17e-3 | 1.64e-3 | 1.64e-3 | 1.74e-3 |
| pyr8 | 8.73e-4 | 7.13e-4 | 3.04e-4 | **1.5e-5** | 3.00e-4 | 6.70e-4 | 5.77e-4 | 6.91e-4 | 1.01e-3 | 9.86e-4 | 1.04e-3 |
| blur16 | 6.67e-5 | 2.82e-5 | 1.16e-5 | **1.1e-6** | 1.21e-5 | 3.16e-5 | 6.10e-5 | 9.20e-5 | 1.09e-4 | 1.40e-4 | 1.53e-4 |

(`docs/phase2f-scan.txt` has every 5°.) Downsampling sharpens the minimum
relative to its surroundings but does not widen it: every pyramid level
keeps the local minimum at −45° and the −80/−70 shoulder; only the 16-px
blur is monotone from −10° to −60° (2e's finding). The pyramid schedule
(pose-only phase on 8×, 4×, 2×, a third each) **diverged** (elbow +25°,
pitch −2°). Not on by default; the fit converged without it because the
start (−37°) is inside the region where the corrected gradient points the
right way, not because the basin is wide.

### 4. The fit

`rose_fit --vis` with `IRYS_SEC_N=4` (192², 8 spp, depth 5 + NEE, host edge
samples with the ID buffer, interior + secondary at 1 spp), Adam as 2e:

| Run | φ (last-40 mean) vs (−25.78, +16.04, −60.16)° | θ vs (0.30, 0.86, 0.88, 0.90) | MSE (truth 8.60e-4) |
|---|---|---|---|
| **context, 2f, 600 iterations** | **(−25.73, +16.07, −60.23)** | (0.303, 0.855, 0.876, 0.895) | 8.57e-4 |
| context, 2f, 300 | (−25.06, +16.12, −57.50) | (0.311, 0.841, 0.857, 0.873) | 2.57e-3 |
| context, host edges + interior, no secondary, 300 | (−25.56, +8.04, +55.68): diverges | (0.336, 0.783, 0.791, 0.801) | 4.19e-3 |
| context, 2f + pyramid, 300 | (−2.34, +19.05, +25.40): diverges | (0.364, 0.701, 0.702, 0.705) | 6.64e-3 |
| control, 2f, 300 | (−25.74, +15.91, −60.34) | (0.257, 0.769, 0.788, 0.806) | 2.24e-3 (2.06e-3) |

Per iteration, context, 600-iteration run (release, M-series, nothing else
running): θ gradient 111.6 ms (GPU), edge term 166.8 ms (host sampling with
the ID buffer + 16 384 pairs on the device), interior + secondary 66.2 ms
(36 864 paths at 1 spp, 4 edge samples per vertex; interior alone is 13 ms,
so the secondary term costs ~53 ms), 244 s in all. Control: 88 / 98 / 22 ms.

![phase 2f rose fit](phase2f-rose-fit.png) (`phase2f-rose-loss.csv`: every
run above, every iteration, the three pose terms and the time split.)

Deviations:

1. Edge sampling, not warped area, for the measured reason above; CPU only.
2. The fit uses host edge samples (`--vis`), not `GpuEdges` (2e's default),
   because the device's are biased on rose (piece 1 (a)). That alone does
   not converge (table: no secondary).
3. 600 iterations, not 300: at 300 the elbow is still 2.7° short (the
   schedule's learning rate decays, so the extra 300 are small steps).
4. One run per configuration: the fit's convergence is not measured over
   seeds.
5. The intersection-curve term is in `--check`, not the fit (it is ≤ 1 %).

### Phase 2f risks and next

1. **Variance of the secondary term on rose.** Heavy-tailed, from vertices
   millimetres from another link's edge; the 256-seed check misses at z −3
   to −4 while 64 seeds pass. An edge tree whose node bound does not cap at
   the node's size near `x`, or a near-field strategy (edges within a few
   `r` of `x`, by an exact local search), is the next lever; then fewer
   iterations would do.
2. **`GpuEdges`' remaining bias** on rose (weights on ~8 % of samples).
3. The basin is still ±5° on the elbow; a fit started outside it fails.
   The blur is the only lever measured monotone, and needs the corrected
   gradient (2e's blurred fit had the biased one) — not re-tried here.
4. Secondary intersection curves and the checker's albedo edges are not
   sampled; the camera gate's c2/c3 sit at z −2.6/−2.2.
5. The secondary term is CPU f64; at 4 samples per vertex it is a quarter
   of an iteration. A device port needs the dual trace first.

## Phase 3: sampling (2026-10-17)

The gate: equal-time error ≤ Cycles on standard scenes. **Not met.** At
equal *samples* irys now matches or beats Cycles on all three benchmark
scenes (relMSE at 1024 spp: pair 4.2e-5 vs 4.0e-5, room 4.8e-4 vs 9.5e-4,
cornell 1.8e-4 vs 3.2e-4); at equal *time* it is 1.5–3.4× behind (below
Cycles only on cornell's smallest budget), and the whole remaining gap is
throughput — Cycles takes a sample 2.4–3.6× faster on this GPU. Before
phase 3 the equal-time gap was 2.2–26×. Seven commits before this note:

| piece | commit | what |
|---|---|---|
| 0 | `irys-bench` | the harness: matched scenes in Cycles, references, equal-time relMSE |
| 1 | emitters, tail, ReSTIR DI | mesh lights + NEE, `Scene::complete`, ReSTIR DI on the wavefront |
| 2 | path guiding | grid × histogram guiding at diffuse vertices, trained online |
| 3 | Sobol + spectral IS | Burley's scrambled Sobol, wavelengths by the film's response |
| 3′ | Sobol constant | direction numbers as a WGSL constant (sampler overhead halved) |
| — | Russian roulette | detached, θ-free; measured, off (below) |
| — | bench timing | fastest-of-repeats and interleaved re-timing on a shared GPU |

### 0. The benchmark (`crates/irys-bench`, `scripts/`)

**Cycles is here**: Blender 4.4.3 at `/Applications/Blender.app` (not on
`PATH`), Cycles on Metal, the M4 Max's 40-core GPU — the same device irys
uses. **Classroom and bistro are not**: nothing was downloaded, and irys
cannot read a `.blend` (phase 5), so the scenes are irys's own, rebuilt
in Cycles from exported triangles:

- **pair** — phase 2c's outdoor scene: two GGX gold balls (one squashed and
  turned) on a checker floor, sky and a soft sun (κ = 40). Direct light.
- **room** — the classroom stand-in: a closed 6 × 5 × 3 m room lit only
  through one window by a sharp sun (κ = 400) and the sky; a gold ball and a
  box inside. Almost all indirect, most light samples occluded.
- **cornell** — a grey Cornell box under one emissive quad (phase 3's mesh
  emitter), a tall box and a gold ball. Grey because irys's diffuse albedo
  is a scalar.

`irys-bench export` writes world-space triangles per placement, materials,
the pinhole and the sky; `scripts/cycles_bench.py` builds them in Cycles.
Matched: geometry, camera (vertical FOV, box filter one pixel wide), the
sky (irys's `Env::radiance` as shader nodes, the gradient's spectral tint
projected to sRGB; Cycles importance-samples a 2048² map of it), Lambert
diffuse and irys's checker as nodes, the conductor as Principled metallic
with single-scatter GGX, roughness √α and base colour = irys's spectral F0
projected to sRGB (exact for Schlick, which is linear in F0), one-sided
emitters, depth `d` = Cycles `max_bounces = d − 1` (irys's
`Scene::complete`, piece 1). No clamping, denoising, or adaptive sampling;
Cycles' light tree and default (Sobol–Burley/blue-noise) sampling stay on.
Not matched: Cycles is RGB and irys spectral; Cycles' metallic Fresnel is
F82-tint, irys's Schlick; Cycles uses Russian roulette, irys a fixed depth.

**The metric.** Each renderer is scored against **its own** 32768-spp
reference — relMSE = mean (x − r)² / (r² + 10⁻²) over pixels and channels —
because the models differ slightly; the references agree anyway (channel
means equal to four digits on all three scenes, cross relMSE ≤ 1e-4), so
neither renderer is scored on an easier picture. 400 × 300, sweeps of
1 … 1024 spp. **Time** is wall time of the render calls on a built, warm
scene: irys's `GpuPrb` calls (calls of 16 spp, or 4 with ReSTIR/guiding);
Cycles' `render()` with persistent data after a warm-up, so its ~45–90 ms
per-render overhead is in. Equal-time error is log–log interpolation of each
curve at Cycles' own times for 64, 256 and 1024 spp. **The GPU was shared**
with other sessions' work for most of the runs, and single sweeps scattered
by up to 2×, so every time reported is the fastest observation of that
level across all sweeps (2–5 per level, `--reps 2` each, plus interleaved
`irys-bench retime` rounds), for Cycles per level and for irys as the
fastest seconds-per-spp at ≥ 64 spp (its time is linear in spp). Both
renderers get their best case; relMSE does not depend on any of it.

### Results

Equal-time relMSE (each renderer against its own reference) at Cycles'
own times for 64, 256 and 1024 spp; *ratio* is irys / Cycles, < 1 is the
gate. "phase 2" is the estimator this phase started from (NEE + MIS,
random numbers, uniform wavelengths); "best" is the best phase 3
configuration on that scene — Sobol + spectral IS on pair and cornell,
plus guiding on room:

| scene | budget | Cycles | irys phase 2 | ratio | irys best | ratio |
|---|---|---|---|---|---|---|
| pair | 0.089 s | 8.8e-4 | 8.6e-3 | 9.8 | 1.4e-3 | 1.6 |
| pair | 0.220 s | 1.7e-4 | 3.5e-3 | 20.6 | 5.0e-4 | 3.0 |
| pair | 0.744 s | 4.0e-5 | 1.0e-3 | 25.9 | 1.4e-4 | 3.4 |
| room | 0.168 s | 1.6e-2 | 4.1e-2 | 2.6 | 2.6e-2 | 1.6 |
| room | 0.559 s | 3.8e-3 | 1.2e-2 | 3.3 | 6.3e-3 | 1.7 |
| room | 2.307 s | 9.5e-4 | 3.1e-3 | 3.2 | 1.4e-3 | 1.5 |
| cornell | 0.165 s | 5.7e-3 | 1.2e-2 | 2.2 | 5.2e-3 | **0.9** |
| cornell | 0.453 s | 1.3e-3 | 4.6e-3 | 3.5 | 2.0e-3 | 1.5 |
| cornell | 1.766 s | 3.2e-4 | 1.2e-3 | 3.7 | 4.7e-4 | 1.5 |

Each piece alone and together, as the equal-time ratio at Cycles' 256-spp
time:

| scene | phase 2 | ReSTIR DI | guiding | Sobol | spectral IS | Sobol + spectral | + guiding |
|---|---|---|---|---|---|---|---|
| pair | 20.6 | 15.0 | 19.9 | 4.2 | 13.6 | **3.0** | 3.7 |
| room | 3.3 | 3.6 | 1.9 | 2.8 | 3.0 | 2.6 | **1.7** |
| cornell | 3.5 | 4.0 | 3.2 | 1.9 | 3.0 | **1.5** | 1.8 |

Time per spp at 400 × 300 (fastest observed): irys 2.6 / 5.7 / 3.9 ms
(phase 2) and 2.6 / 6.5 / 4.1 ms (best) on pair / room / cornell; Cycles
0.73 / 2.25 / 1.72 ms at 1024 spp, overhead included. irys is 2.4–3.6×
slower per sample, which is the whole gap: at **equal spp** the best
configuration is level with Cycles on pair and twice as good on room and
cornell (relMSE × spp at 1024 spp, lower is better):

| scene | Cycles | phase 2 | ReSTIR DI | guiding | Sobol | spectral IS | Sobol + spectral | + guiding |
|---|---|---|---|---|---|---|---|---|
| pair | 0.041 | 0.299 | 0.210 | 0.314 | 0.058 | 0.214 | **0.043** | 0.048 |
| room | 0.973 | 1.240 | 1.250 | 0.698 | 0.933 | 1.127 | 0.834 | **0.494** |
| cornell | 0.331 | 0.538 | 0.534 | 0.451 | 0.237 | 0.444 | 0.202 | **0.184** |

![pair](phase3-pair.png)
![room](phase3-room.png)
![cornell](phase3-cornell.png)

(Each image: Cycles at its 64-spp time; irys's phase 2 estimator and
Sobol + spectral IS + guiding at their levels nearest that time; both
references. Every panel is labelled with spp, seconds and relMSE.
`scripts/bench.sh` and `irys-bench report` regenerate them.)

### 1. Mesh emitters, the tail, ReSTIR DI

**Emitters.** `Surface::Emitter` is a black one-sided area light of
`Scene::emission`. NEE picks the environment or an emitter triangle
(`Scene::lights`, a CDF by area — with one emission, area is power and a
light tree would pick the same triangles) and MIS weighs it against BSDF
hits on both tiers. The PRB replay is unchanged in form: emission is θ-free.
**The tail** (`Scene::complete`): the last vertex's BSDF ray is traced for
emission only, so a path of `d` vertices collects exactly the light of ≤ `d`
scatterings by both strategies. Without it the last vertex had only its
NEE share, ~3 % dark on the box — which is how this was found: NEE and
BSDF-only images disagreed.

**ReSTIR DI** (`gpu/restir.wgsl`), at the first vertex: M0 = 8 candidates
from the base light sampler, temporal reuse of the same slot's previous
reservoir (its own G-buffer: a different sub-pixel point), k = 3 spatial
neighbours within 16 px, all in generalized RIS (Lin et al. 2022) with
pairwise MIS and unshadowed targets — **unbiased** (tested block by block
against plain NEE). A sample is an environment direction or an emitter
point, so reuse is the identity shift. The reservoir estimates the light
strategy's *share* of the NEE/BSDF MIS pair (`w_L` is in its integrand), so
BSDF escapes keep their `w_B` and the rest of the estimator is untouched;
`W` is a detached number and PRB's light lane takes `cos · w_L · W · G` in
place of `cos · w_L / p_L`. Replay is exact because the reservoirs are a
function of the seed and the *previous* frame's buffers, which only
`GpuPrb::advance` changes.

It buys 1.4× at equal spp on pair (direct light) and nothing on room and
cornell (indirect-bound), for 5–16 % more time — **net loss everywhere
once the sampler is in** (Sobol already stratifies the direct light, and
ReSTIR's candidates use their own random stream). Unshadowed targets are
why it cannot help the room: the sun wins every reservoir, also where it is
occluded.

### 2. Path guiding (`gpu/guide.wgsl`)

**Choice.** A fixed-resolution cousin of practical path guiding (Müller et
al. 2017), not online neural guiding (NASG, 2303.08064): the question was
how much guiding buys on these scenes, and a grid of histograms answers it
without a second network in the bounce loop; it also kept the invariants
for free (integer atomics make training deterministic; the guide is a
constant between builds, so replay is exact). A neural guide is the next
step if the grid's resolution turns out to be the limit (it was not: 10, 16
and 24 cells per side measured within 6 %).

A uniform grid (16 cells on the scene's longest side) × 6 normal classes,
each a 16 × 16 equal-area histogram of incident radiance. Diffuse vertices
sample a one-sample MIS mixture `q = c g + (1 − c) cos⁺/π`, c = 0.5 once a
cell has 64 samples; `q` is the BSDF strategy's pdf for every MIS weight.
The cosine half keeps it unbiased whatever the histogram holds. Training
runs in forward passes only: each diffuse vertex records `Li / q` with
`Li = (L − L⁺) / T`; `guide_build` normalises, mixes 10 % uniform and
prefix-sums. In the bench the first half of a render's calls train
(rebuilding after each) and every call counts towards the image. Learning
NEE's share as well (Müller's choice) was measured worse: 2.3e-3 against
1.7e-3 on the box at 256 spp.

Room: 1.8× lower error at equal spp, 1.7× at equal time. Cornell: 1.2×.
Pair: nothing (direct light). Conductors are not guided.

### 3. The sampler (`qmc.rs`)

**Choice over ReSTIR PT and NRC**, by the benchmark: at equal spp Cycles'
error on pair was 7× lower than irys's, and the gap was where the samples
go — independent random numbers against Cycles' Sobol–Burley, and a film
whose four stratified hero wavelengths leave a *white* pixel with a
relative variance of 0.152 per path (pair's whole error at 1 spp was 0.29).
Path reuse addresses neither; NRC would add bias for an indirect-bound gap
the guide already halves.

**Sobol, shuffled and Owen-scrambled** (Burley, JCGT 2020; Cycles'
Sobol–Burley). A path's draws come in tuples of up to four dimensions —
camera `(λ, x, y)`, then a light tuple and a BSDF tuple per vertex
(`PathRng::align`) — and each draw is a pure integer function of (pixel,
sample, dimension): `owen(sobol_d(owen(i, seed)), hash(seed, d + 1))`.
Owen-scrambling the index maps each aligned power-of-two block of indices
onto another, so every tuple's first 2^k samples are still a (0, k, 2)-net
(unit test, every elementary interval) while tuples are decorrelated.
The device and the CPU compute the same integers (GPU vs CPU f64 per lane:
median rel 1.9e-7, no lane off by 1e-3); replay is exact by construction.
The frame seed's low word is the frame's place in the sequence.

**Wavelengths by importance**: the hero uniform through the inverse CDF of
`p(λ) ∝ |r̄| + |ḡ| + |b̄|` (256 bins), the other three rotated by `k/4` *in
CDF space* so each lane is marginally `p` and the set stays stratified; the
film divides by `p`. White's per-path variance 0.152 → 0.054.

Together: pair 7× lower error at equal spp, cornell 2.7×, room 1.5×, for
0–12 % more time (after the direction numbers became a WGSL constant;
+12–20 % before).

### Russian roulette (measured, off)

`Scene::roulette`: from vertex 3, continue with `q = clamp(proxy, 0.1, 1)`
and divide by `q`, the proxy being the diffuse albedos over the earlier
survivals' `q` — θ-free, so the decision is detached and CRN differences
see the same paths (CPU and device; tested like the rest). On the room it
saves 28 % of the time and costs 43 % more error per sample, neutral alone
and a loss with the guide (it cuts the guided paths that carry the light).
With lane compaction (risk 1) a dead lane would also stop costing an IR
evaluation, and the trade changes.

### Phase 3 deviations

1. **Scenes**: no classroom or bistro (no download without asking, and no
   `.blend` import before phase 5). The stand-ins test the same things —
   indirect light through a small opening, a small area light, direct sun —
   but are smaller and simpler than the real ones.
2. **The Cornell box is grey** (scalar diffuse albedo); emitters are
   spectrally flat.
3. **ReSTIR DI and guiding are GPU-only**; the CPU reference refuses a
   scene that asks for them, and their tests are GPU against GPU (plain NEE
   or cosine sampling for unbiasedness, central differences for PRB). The
   sampler is on both tiers.
4. **ReSTIR needs the frame in one chunk** (≤ 524 280 paths: 4 spp at
   400 × 300), and its candidates draw from the random stream even under
   Sobol.
5. **No light tree**: one emission means area is power. It belongs with
   the first scene whose emitters differ.
6. **Timing on a shared GPU**: other sessions used the device during most
   runs; the reported times are best observations for both renderers (see
   the metric). An idle machine would give tighter numbers, not different
   conclusions: the per-sample errors carry the argument.

### Phase 3 risks and next

1. **Throughput is the gate now.** Cycles traces 2.5–4× more paths per
   second here. irys walks its BVH in WGSL (≈ 60 % of a frame), Cycles uses
   Metal's ray-tracing hardware; irys also evaluates the IR's BSDF over
   every lane every bounce, dead or diffuse (≈ 25 %, bandwidth-bound), and
   has no Russian roulette (fixed depth). Hardware ray queries (wgpu's ray
   tracing on Metal), lane compaction before the IR eval, and roulette are
   the levers, roughly in that order.
2. **ReSTIR DI is a net loss** with the sampler on. Visibility-aware reuse
   (shadow rays in the pairwise MIS weights) is what helps interiors; it
   costs O(k) shadow rays per pixel and was not tried.
3. **Bind groups must list exactly the bindings each entry point uses**
   (wgpu's derived layouts): a kernel edit that stops using a buffer fails
   validation at `GpuPrb::new`. Explicit layouts would make it robust.
4. **The guide's grid is uniform**: fine for these rooms, wasteful for a
   street; an adaptive tree (SD-tree) or a neural guide when scenes grow.
5. ReSTIR PT/GRIS and NRC were not built.

## Speed: hardware ray queries, lane compaction (2026-10-17)

Phase 3 left the gate on throughput: irys took a sample 2.4–3.6× slower
than Cycles on the same M4 Max. Two commits on `claude/speed`:

| piece | commit | what |
|---|---|---|
| 1 | hardware ray queries | BLAS per tree, TLAS of the placements, `rayQuery` for closest and shadow rays |
| 2 | lane compaction | stage → deterministic scan → scatter → eval IR over the compact lanes only |

### 1. Hardware ray tracing (`gpu/rt.rs`, `gpu/rt.wgsl`)

**What wgpu 30 offers on Metal** (read in the registry source, then
probed on the device): `wgpu-hal-30.0.1/src/metal/adapter.rs` sets
`Features::EXPERIMENTAL_RAY_QUERY` from `supportsRaytracing` and
`supportsRaytracingFromRender` on macOS ≥ 15; `metal/command.rs` builds
BLASes and TLASes through `MTLAccelerationStructureCommandEncoder` (triangle
and bounding-box geometry, indirect instance descriptors whose transform
is the instance's row-major 3 × 4); `naga-30.0.1/src/back/msl/ray.rs`
lowers WGSL `rayQuery*` to Metal's `intersection_query`, procedural AABB
candidates and `rayQueryGenerateIntersection` included. The M4 Max reports
the feature. A Metal-native backend (objc2-metal intersector, behind a
feature) and a faster software walk (CWBVH) were therefore not needed:
the hardware is reachable from WGSL, behind the one experimental-features
token, and the shader stays one module.

**Build.** Per bottom-level tree, up to two BLASes: its triangles (opaque,
in mesh order, so `primitive_index` is the prim record's offset) and its
analytic prims as AABBs (padded by a relative margin), intersected in the
candidate loop by the software walk's own `hit_prim` in the instance space
`to_local` gives. One TLAS instance per (placement, BLAS), its transform
the placement's object → world, `rt_inst` mapping it back to (instance
record, first prim record) — so a hardware hit is exactly the `Hit` the
WGSL walk returns. `update_geometry` rebuilds the TLAS. Splats keep the
software K-buffer (they are parked); the edge sampler keeps the WGSL walk
(one eye ray per edge point).

**Two device facts found the hard way.** The first frames after a build
were missing a whole BLAS, a different one run to run, until the BLAS
builds were waited on before the TLAS build and first use: Metal does not
order an acceleration-structure build against the work that reads it
here. The builds also go one per submit (wgpu's Metal
`place_acceleration_structure_barrier` is empty). And naga rejects
`rayQueryGenerateIntersection(&q, expr)` with an inline expression ("used
before it was introduced"); a `let` first works.

**Determinism.** Ray queries are a function of (structure, ray) on this
device: replay is bit-exact and two wavefronts agree bit for bit under the
hardware walk (`tests/speed.rs`, four scenes: instanced meshes, the
emitter box, analytic prims alone, triangles + a sphere behind splats).
Ties in t between two triangles are the hardware's to break, not prim
order as in the WGSL walk; nothing in these scenes has coplanar
overlapping triangles, and a tie would be resolved the same way every
trace, which is what replay needs. The hardware triangle test is
watertight where Möller–Trumbore leaves cracks: on 120 000 primary rays
pair and cornell differ from the WGSL walk on one ray each (an edge, the
hardware hitting what the software missed). Images agree to 1e-3 in mean
and relMSE; on the bench the relMSE at every level is the software walk's
to four digits. `IRYS_RT=sw` (or `GpuPrb::set_hardware_rt(false)`) keeps
the WGSL walk as fallback and reference.

Incoherent rays (8 bounces from the primary hits, a microbenchmark): the
hardware walk traces 3.4 G rays/s in the room against 0.57 G for the WGSL
walk (6×), 3.6 G against 0.68 G in the box, 2.4 G against 1.6 G on pair;
shadow (terminate-on-first-hit) rays 4.5 G against 0.62 G.

### 2. Lane compaction (`gpu/compact.wgsl`)

`bounce` used to write seven graph columns for every lane of every path
every bounce (harmless ones for diffuse and dead vertices), and the eval IR
read them all: ~800 bytes per path per bounce, bandwidth-bound. Now a
conductor vertex stages its two lane groups (BSDF sample, light sample) as
two `vec4`s per path and sets lane bits; a fixed-shape integer scan
(Hillis–Steele per 256 paths, then one workgroup over the block totals —
no atomics) gives each path its compact slot; `scatter` writes the columns
there; the eval IR (pointwise over lanes) runs by **indirect dispatch** over
the compact lanes alone, and `update` reads `f` there and skips paths with
no lanes without loading them. The GGX conductor is the only IR BSDF, so
this is also the material bin: diffuse vertices never reach the graph.
Replay compacts identically (same kernels, same bits) and additionally
zeroes `δf` past the compact lanes, so the adjoint's full-width reduction
sees no stale adjoint; columns start harmless so never-written lanes stay
finite. Chunked and whole-frame renders are bit-identical.

**Russian roulette, remeasured** (it was off because a dead lane still
cost an IR evaluation, which compaction removes): room −25 % time, +56 %
error per sample; cornell −18 %, +62 %. Error × time is worse on both, with
and without the guide. It stays off.

### Results

Same protocol and references as phase 3 (each renderer against its own
32768-spp reference; equal-time error by log–log interpolation at Cycles'
own 64/256/1024-spp times; fastest observation per level). Cycles' times
are the fastest of the phase 3 sweeps and three sweeps run here,
interleaved with irys's; irys's are the fastest per level of five sweeps
× 2 repetitions. **Raw** times are used, not phase 3's "fastest s/spp
extrapolated linearly": on the closed scenes irys's per-sample time grows
with the render's length (cornell 2.14 ms/spp at 64 spp, 2.88 at 1024;
room 2.55 → 2.85), reproducibly and with the software walk too, so the GPU
sustains less than its first second — linear extrapolation would credit
irys with a speed it does not keep. The machine was shared (load average
10–45 from other sessions' builds and tests during the runs).

Time per sample, 400 × 300, ms (irys: fastest at ≥ 64 spp / at 1024 spp):

| scene | Cycles | irys phase 3 best | irys now: Sobol + spectral | irys now: + guiding |
|---|---|---|---|---|
| pair | 0.71 | 2.6 | **0.88** / 0.89 | 1.03 / 1.03 |
| room | 2.14 | 6.5 | 2.55 / 2.85 | **2.92** / 3.53 |
| cornell | 1.44 | 4.1 | **2.14** / 2.88 | 2.77 / 3.55 |

Equal-time relMSE, irys / Cycles (< 1 is the gate), the best
configuration per scene in bold as in phase 3:

| scene | budget | Cycles | Sobol + spectral | ratio | + guiding | ratio | phase 3 best ratio |
|---|---|---|---|---|---|---|---|
| pair | 0.089 s | 8.8e-4 | 4.1e-4 | **0.47** | 5.8e-4 | 0.66 | 1.6 |
| pair | 0.220 s | 1.7e-4 | 1.6e-4 | **0.93** | 2.2e-4 | 1.29 | 3.0 |
| pair | 0.729 s | 4.0e-5 | 5.1e-5 | **1.27** | 6.5e-5 | 1.64 | 3.4 |
| room | 0.167 s | 1.6e-2 | 1.3e-2 | 0.83 | 9.8e-3 | **0.62** | 1.6 |
| room | 0.534 s | 3.8e-3 | 4.1e-3 | 1.09 | 3.0e-3 | **0.79** | 1.7 |
| room | 2.190 s | 9.5e-4 | 1.1e-3 | 1.12 | 7.8e-4 | **0.82** | 1.5 |
| cornell | 0.144 s | 5.7e-3 | 3.4e-3 | **0.60** | 4.6e-3 | 0.81 | 0.9 |
| cornell | 0.419 s | 1.3e-3 | 1.3e-3 | **0.98** | 1.5e-3 | 1.17 | 1.5 |
| cornell | 1.474 s | 3.2e-4 | 4.0e-4 | **1.23** | 4.7e-4 | 1.45 | 1.5 |

With phase 3's linear-time protocol the room ratios are 0.62 / 0.73 / 0.69
and cornell's 0.60 / 0.81 / **0.90**; pair is unchanged (its time per
sample does not drift).

![pair](speed-pair.png)
![room](speed-room.png)
![cornell](speed-cornell.png)

(Cycles at its 64-spp time, both irys configurations at their nearest
level, both references. `irys-bench report --prefix speed` regenerates
them; `irys-bench speed` prints ms/spp per variant.)

**Gate verdict: met on room at every budget; not met overall.** irys is
ahead of Cycles at every small budget (0.47–0.62) and on the room
throughout, level at the middle budgets of pair and cornell (0.93, 0.98),
and behind at the largest budget on pair (1.27) and cornell (1.23 raw,
0.90 by phase 3's protocol). The time per sample came down 2.9× on pair,
2.2× on room and 1.9× on cornell; irys is still 1.25× (pair), 1.2–1.65×
(room, depending on configuration and length) and 1.5–2.0× (cornell)
slower per sample than Cycles, and the largest-budget gap is that, against
an error per sample that is level (pair) or 1.6–2× better (room, cornell).

### Where a frame goes now, and the next lever

Measured by removing kernels from the trace at 64 spp (room, ms/spp, the
frame is 2.36): the pipeline floor (gen, finish, scoring, readback, the
per-call submit) 0.21; `bounce` ≈ 2.0; scan + scatter + eval + update
≈ 0.2. Inside `bounce`, the hardware trace of ~2 M rays/spp at the
microbenchmark's 3.4–4.5 G rays/s is ≈ 0.5–0.6. The rest of `bounce` —
≈ 1.4 ms/spp, more than half the frame — is shading and state, and the
largest named part of it is the **sampler**: Sobol + spectral IS against
plain random numbers costs 0.16 / 0.55 / 0.48 ms/spp on pair / room /
cornell (18–23 % of the frame; phase 3 measured 0–12 % of a frame 2–3×
longer). Each draw hashes six times, Owen-scrambles twice and walks up to
32 Sobol columns for the scrambled index's set bits. **Next lever**: hoist
the per-path and per-tuple hashes (four of a tuple's draws share the
scrambled index), and replace the 32-step column walk by four byte-table
lookups per dimension (3 × 4 × 256 words, XOR is exact, so device and CPU
still agree bit for bit). Expected: most of the 0.5 ms/spp on the
interiors, which is the remaining gap on cornell and pair at the largest
budget. Then the per-call floor (0.21 ms/spp, a quarter of pair's frame):
the bench reads the image back after every 16-spp call; accumulating on
the device would remove most of it. Neither was started.

### Speed risks

1. **`EXPERIMENTAL_RAY_QUERY` is experimental** in wgpu 30 and needs the
   `ExperimentalFeatures` token in `irys-gpu`'s device request; the
   build/use ordering had to be enforced by hand (a wait after the BLAS
   builds). A wgpu upgrade should rerun `tests/speed.rs` first.
2. **Ties** between triangles at equal t resolve in the hardware's order,
   not by prim index. Deterministic on this device and covered by the
   replay and rerun tests; coplanar overlapping geometry would show it.
3. **Sustained throughput** on the closed scenes is 10–35 % below the first
   second's, on both walks; the bench reports raw times for that reason.
   Whether it is power management under the other sessions' load or
   intrinsic is not separated.
4. **The replay is not compacted**: it zeroes `δf` and runs the adjoint IR
   over every lane, so gradient passes gained the trace speed-up but not
   the eval one.
5. The edge sampler and splats still walk in WGSL.

## Open questions

- **Name.** `irys` is a placeholder; check crates.io before step 4.
- **f64 vs f32.** kosm-render is f64 on CPU, f32 on GPU. Keep it, or go
  `Scalar`-generic all the way (costs compile time)?
- **Spectral vs RGB.** vcad's photoreal path may be RGB. irys is spectral;
  RGB is a 3-band spectrum. Confirm vcad looks don't shift.
- **Where Camera lives.** irys owns the pinhole/thin-lens model; kosm's
  `Camera` lens wraps it. Equirect (muni's 360 camera) goes in irys.
- **Licence.** Match kosm's.

## Roadmap to Cycles parity (2026-10-17)

Each phase ends on a measured gate, not a demo.

| Phase | Work | Exit gate |
|---|---|---|
| 0 seed | split kosm-render into crates, reference CI | kosm builds on irys, snapshots byte-identical |
| 1 differentiable core | tang IR BSDFs, path replay backprop, hero wavelength | marble gradients match central differences, O(1) memory |
| 2 unified scene + wavefront GPU | 3DGRT splats + BRep + mesh in one BVH, vcad on wgpu 30 | CAD rover in a splat street; pose fit from a photo |
| 3 sampling | ReSTIR PT, neural guiding, NRC | equal-time error ≤ Cycles on classroom, bistro (2026-10-17: not met; equal-spp error ≤ Cycles, throughput 2.4–3.6× short — see Phase 3; after Speed: met on room, ahead at small budgets everywhere, 1.23–1.27× behind at the largest budget on pair and cornell) |
| 4 the look | OpenPBR, layering, volumes, SSS, hair | OpenPBR reference scenes match |
| 5 product | USD + .blend import, denoiser, RL batch lane | Blender demo files render; public bench vs Cycles |

muni lands after phase 2. Phases 1–2 are the research risk; 3–5 are
known techniques and a lot of engineering.

## The kernel contract

Every BSDF, texture, emitter and the denoiser is a tang IR graph, written
once and pure. The compiler lowers each graph three ways:

- `eval`: f and pdf
- `sample`: u → wo, **detached**. Sampling is never differentiated.
- `adjoint`: δf → δθ, from `TensorGraph::backward`

The integrator runs a forward pass that keeps only L per path, then a
**path replay** pass (Vicini+ 2021): same seed, same path, and at each vertex
δθ += δL · (L − L_so_far) · ∂f/∂θ, atomically added. Memory is
pixels × 1, not pixels × spp × depth. Visibility discontinuities are handled
by a separate reparam kernel (warped-area, Bangaru+ 2020).

Oracle: the f64 tang interpreter runs the same graph; every backend kernel is
checked against it and against central differences.

## The look: OpenPBR, position-free

OpenPBR's slab (fuzz, coat, thin film, base metal/dielectric, subsurface) is
evaluated by position-free Monte Carlo (Guo+ 2018): an unbiased random walk
between interfaces, each interface one tang IR BSDF. Spectral by default, so
thin film and dispersion are exact. Risk: replay through a walk of random
length must consume the RNG stream identically. Test that first.

## The film: KPN-driven adaptive sampling

Render a few spp → film keeps mean, variance and AOVs → KPN denoises and
predicts its residual error → the sampler puts the next spp where that error
is → repeat until the predicted error is under target. KPN and sampler train
together as one tang IR graph (nonlinear Noise2Noise, no clean references).

## Skin (2026-10-17)

A lifelike head, as a first instance of "the look" (phase 4) on the PRB
machinery: `irys-diff::skin`, additive to everything above (no change to
`prb`, `gpu.rs` or `wavefront.wgsl`). The subject is the Lee Perry-Smith scan
— "Lee Perry-Smith" by Infinite-Realities (ir-ltd.net), CC BY 3.0
(creativecommons.org/licenses/by/3.0), via the three.js examples — read from
`$IRYS_ASSETS/lee-perry-smith` (default `/Users/cam/Developer/irys-assets`)
and never copied into the repository. The renders below are of it.

![front](skin-head-front.png)

![three-quarter](skin-head-3q.png)

![no SSS · SSS + one lobe · SSS + two lobes](skin-compare.png)

### Method

- **Mesh and maps** (`skin::mesh`). The glb's first primitive (9 279
  vertices, 17 684 triangles, a bust) through `gltf` + `image` behind an
  `assets` feature; per-vertex tangents by Lengyel's method; bilinear,
  wrapped, linear-f32 textures (sRGB decoded for colour). Two facts about
  this file, both settled by the maps rather than by convention: its UVs
  run `v` *up* the image (read as glTF says, the lips land on the chin and
  the eyebrows on the neck; `load` flips them, `the_eyebrows_are_above_the_eyes`
  holds it), and its normal map is green-up (as a height gradient, the
  green-up reading is 2.26× closer to curl-free than green-down;
  `the_normal_map_is_opengl`). 1 unit = 50 mm (4.3 units nose to occiput).
- **The medium** (`skin::optics`). One homogeneous layer per entry point,
  per mm: `σa = s (m μa_mel + h μa_blood + μa_base)`, `σs = s μs'/(1 − g)`,
  Henyey–Greenstein `g = 0.8`. Measured constituents: melanosome absorption
  `51.9 (λ/500)^−3.5` and the average-skin reduced scattering
  `4.60 (0.409 (λ/500)^−4 + 0.591 (λ/500)^−0.562)` (Jacques 2013), whole
  blood from Prahl's HbO₂/Hb extinction at 75 % saturation, Jacques' bloodless
  baseline. One 1-nm table feeds CPU and GPU. `θ = (m-scale, h-scale, s, r₁,
  r₂)`.
- **Albedo → chromophores.** The map's colour is inverted per texel to
  `(m, h)`: the walk's own total albedo (below) through the similarity
  relation, times `1 − F̄` (the entry Fresnel, cosine-averaged), integrated
  against the sRGB response, searched on a 96² grid at the nodes of a 48³
  `√rgb` grid and trilinearly interpolated (0.6 s for 1024²; mean RGB error
  0.030; 0.6 % of texels at a bound — painted stubble and lips). The head
  renders at `θ = (1, 1, 1, 0.35, 0.6)`: the inverted maps as they are.
- **The walk** (`render::walk_g`). Analog, one wavelength per walk: distances
  drawn at the sampler's `σt` (detached), `f = σs e^{−σt t}` per scatter and
  `e^{−σt d}` at the boundary, HG turns. Boundaries are any triangle of the
  closed mesh; entry and exit are diffuse transmissions (cosine in, cosine
  out about the smooth normal), `(1 − F(cos_o))` at entry, nothing at the
  inner boundary — Cycles' random walk. Each entry runs **four walks**, one
  per hero wavelength, each with NEE at its own exit; the path continues
  from one (picked uniformly, weight 4). A walk that runs out of events
  (4096) is absorbed.
- **The layer** (`skin::graph::Spec`). Dual-lobe GGX dielectric, Schlick at
  `η = 1.4`, height-correlated Smith, lobes 0.85 / 0.15 at perceptual
  roughness `r₁ = 0.35`, `r₂ = 0.6` (`α = r²`), scaled by Map-SPEC
  normalised to a mean of 0.7. VNDF sampling of the lobe picked by its
  weight, MIS against the lights with the mixture's pdf. Reflect-or-enter at
  `P_r = clamp(4 F k, 0.1, 0.6)`, detached.
- **Portrait** (`skin::head`, `render`). Three softbox quads (warm key
  7 × 7, cool fill 10 × 10, rim strip), NEE picking a light by
  power × the two cosines / d² (detached; the MIS pdf recomputes it at the
  vertex the ray left), MIS with every BSDF-sampled light hit; a dark
  studio gradient; thin lens (Shirley–Chiu); ACES (Narkowicz's fit) and
  the sRGB curve on display only, exposure from the lit pixels' median.
- **Differentiable** (`skin::graph`, `render::replay`). Both factors are tang
  IR graphs. `Medium` is written in the log domain, `ℓ = s ln σs − t σt`,
  and `ℓ` is linear in `(s, t)` for fixed coefficients, so every event of a
  walk — which all share `W = δL (L − L_entry⁺)`, nothing being collected
  inside — reduces exactly to one adjoint lane with `s = W · #scatters`,
  `t = W · Σt`: the replay runs the adjoint once per walk, not once per
  event (155 records for 43 k events in the test). Records are reduced in
  path order through the compiled adjoints: the gradient is bit-identical
  run to run.
- **GPU** (`skin/gpu.rs`, `skin.wgsl`). The same path as one WGSL
  megakernel, one thread per pixel per sample, samples as separate
  dispatches in order (no atomics), the frame chunked at 64 k pixels per
  command buffer. Forward only.
- **Safe sphere** (`skin::accel`). A walk asks once how far the nearest
  triangle is; every later step ending inside that sphere cannot reach the
  boundary and is not traced. Outcomes are unchanged (a step is either
  provably clear or traced as before); the GPU walk got 3.8× faster
  (256² × 64 spp: 38 s → 9.9 s).

### Validation

| Gate (release, M-series) | Result |
|---|---|
| walk vs Chandrasekhar's `H` (semi-infinite, isotropic, index-matched), `ϖ` 0.3 … 0.999, diffuse and normal incidence, 50 k–400 k walks | every case within 1.7 σ; rel error ≤ 4.3e-3 (e.g. `ϖ = 0.99`: 0.79404 ± 0.00096 vs 0.79456 diffuse, 0.75268 ± 0.00103 vs 0.75272 normal) |
| `g = 0.8` walk vs the similarity relation (what the inversion assumes) | absolute albedo error ≤ 0.0025 at `ϖ` 0.9 … 0.999 |
| `∫ H dμ` vs `(2/ϖ)(1 − √(1 − ϖ))` | < 1e-10 |
| `Spec`, `Medium` graphs vs their f64 twins (interpreter) | ≤ 1e-12; `Medium` adjoint vs analytic `∂ℓ` ≤ 1e-10 |
| compiled C (f32) vs interpreter adjoint, full replay | rel ≤ 3.1e-6 on all five θ |
| replay vs forward, SSS ball (16² × 2 spp × 4 λ, 1000 walks, 199 k events) and Lambert | 2048 / 2048 lanes bit-identical, both; δθ bit-identical twice |
| PRB vs central differences, frozen sampler, CRN, f64 | mean loss: rel ≤ 6.3e-9 on all five θ; L2 to a target: ≤ 3.2e-8 |
| live sampler (distances drawn at the current θ), melanin, 24 seeds | PRB −0.370 ± 0.011 vs CD −0.388 ± 0.041 (0.4 σ) |
| GPU f32 vs CPU f64, per lane, ball, Lambert / SSS | median rel 8.3e-8 / 8.4e-8; > 1e-3: 0 / 8192 and 17 / 73 728 — the walks follow the same paths in f32 |
| same, a 32² crop of the face, Lambert / SSS | median 1.3e-6 / 2.7e-6; frame sum rel 3e-7 / 2.5e-4 |
| GPU frame twice | bit-identical |

### Renders and timings

`examples/skin` (GPU, release, Metal, M-series; depth 5, 4 λ per path):

| Frame | Size × spp | Time |
|---|---|---|
| `skin-head-front.png` | 1024² × 256 | 260 s (1.03 Mpaths/s) |
| `skin-head-3q.png` | 1024² × 256 | 252 s (1.07 Mpaths/s) |
| `skin-compare.png`, per panel: Lambert + 1 GGX / SSS + 1 GGX / SSS + 2 GGX | 512 × 640 × 256 | 0.9 s / 98 s / ≈ 100 s |

SSS costs ≈ 100× the surface model: four walks per entry, ≈ 300 events each.

### The fit

`examples/skin_fit` (CPU f64 forward, compiled-C adjoints): the head with
uniform chromophores, target rendered at truth `(m, h, r₁) = (0.030, 0.012,
0.30)` at 96² × 32 spp; start `(0.090, 0.004, 0.50)` (3× melanin, ⅓ blood,
rougher); Adam on `(ln m, ln h, r₁)`, 96² × 16 spp per gradient with
decorrelated primal/adjoint seeds, 300 steps, cosine decay; 340 s.
Last-10 mean: **melanin 0.0306, blood 0.0103, r₁ 0.311** — melanin and
roughness within 4 %, blood 14 % low (its signal is the 540–580 nm band, and
it trades against melanin there).

![fit](skin-fit.png) (`skin-fit.csv`: every iteration's loss and θ.)

### Deviations

1. **One layer, not two.** Donner & Jensen's epidermis (melanin) over
   dermis (blood) is folded into one bulk medium per entry point: the
   inversion can only identify two numbers from a colour anyway, and a
   layered walk is the position-free machinery this section did not build.
   The chromophores are the entry texel's for the whole walk.
2. **Diffuse transmission at the boundary**, no inner Fresnel, no rough
   refraction — Cycles' convention, and the reason the walk's albedo has an
   exact reference.
3. **One wavelength per walk.** The first version walked all four hero
   wavelengths together with spectral MIS (one-sample balance over lanes,
   throughput-weighted): the per-event weight's variance compounds over
   hundreds of events and the frame was chroma noise. Four separate walks
   cost 4× the events and keep the four-wavelength colour of the direct
   light; the indirect continuation is one wavelength.
4. **`f` is computed in Rust f64 on the CPU**, the graphs' twins (held equal
   to 1e-12 against the interpreter); the IR runs the adjoints. Phase 1's
   note 5 (one eval program for `f`) is about f32 rounding between lowerings;
   here both passes run the same Rust, and replay is bit-exact.
5. **The GPU is forward only** and its `f` is hand-written WGSL of the same
   formulas, not the IR's emitted kernels. The gradients are the CPU's.
6. **Capped walks.** 3–4 % of walks on the head reach 4096 events (deep red
   transport through the skull's volume, which is not skin anyway) and are
   absorbed: a small darkening bias in the reds.
7. **No eyes.** The scan's eyelids are closed; there is no socket to put an
   eyeball in, so no sclera or cornea. Deferred to a subject with open eyes.
8. **The normal map** bends the specular layer and the entry Fresnel only;
   the exit lobe uses the smooth interpolated normal (the geometric normal
   showed the facets; a bumped exit would put back the pores the medium
   blurs).

### Gaps to MetaHuman-level realism

- **Hair.** None as geometry: eyebrows, lashes and stubble are paint in a
  1K albedo, the scalp is bald. Strand hair (Marschner/d'Eon fibres, a
  Chiang-style BCSDF) is a phase-4 item.
- **Eyes.** Closed in the scan; no eyeball, cornea, tear line or caruncle.
- **Maps.** 1024² colour, specular and normal. MetaHumans carry 4–8K
  albedo, cavity, roughness and per-region normal maps plus animated
  wrinkle maps; here the pores are a few texels wide.
- **No displacement.** 17.7 k triangles; silhouettes (ears, nostrils, lips)
  are polygonal and all fine detail is the normal map.
- **No measured SSS maps.** Chromophores come from inverting an albedo that
  has lighting and occlusion baked in; there is no measured scatter radius,
  no spatially varying density, no epidermis/dermis split, no
  hemoglobin-oxygenation map, no peach fuzz / sheen.
- **No mouth interior, teeth, or neck/shoulder cloth**, and the bust's
  bottom is open (walks that leave through it are lost — rare).
- **Lighting** is three analytic quads and a gradient, not a captured HDRI
  stage; there is no denoiser, and 256 spp still shows SSS grain in the
  shadows.

### Risks and next

1. **Walk cost** dominates (≈ 300 events per walk, four walks per entry). A
   majorant/null-collision walk over a coarse density grid, or a
   diffusion-guided exit (Křivánek & d'Eon's zero-variance walks), would
   cut it; so would path guiding at the exit (the phase-3 branch's work).
2. **Position-free layering** (the design's flagged risk) is now shown
   safe for a random walk of random length: replay re-consumes the stream
   identically. A layered OpenPBR slab is the same machinery with more than
   one medium.
3. **GPU gradients.** The megakernel is forward-only; the walk's adjoint
   reduces per walk, so a device replay needs only one record per walk per
   lane — the wavefront's reduction scheme applies directly.
4. **Fit from a photo** needs a camera and lighting fit first (phases 2d–2f
   have both), and the albedo's baked lighting separated from the
   chromophores.
