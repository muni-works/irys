# irys — design

One renderer for the stack. vcad, kosm, ipse, rose and muni each draw pixels
today, in five different ways. irys is where the light lives; every repo above
it is a client that hands over geometry and a camera and gets a frame back.

Status: proposal, 2026-10-17. Nothing has moved yet.

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
| 3 sampling | ReSTIR PT, neural guiding, NRC | equal-time error ≤ Cycles on classroom, bistro |
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
