# irys

The renderer for vcad, kosm, ipse, phyz and muni. Owns no geometry: clients
implement `Geometry`, irys does the light.

Start with [docs/design.md](docs/design.md).

| Crate | What |
|---|---|
| `irys-core` | math, rays, `Geometry`, `TriMesh`, BVH/SAH, TLAS, sampler |
| `irys-prims` | analytic prims, heightfield |
| `irys-light` | spectrum, optics, env, CPU path tracer, caustics, splats, neural (CPU) |
| `irys-gpu` | the wgpu 30 tier: context, buffers, WGSL integrator, history, denoiser |
| `irys` | facade (`use irys::*`), kosm-render 0.2's public paths |
| `kosm-render` | 0.2.1 re-export shim for kosm |
| `irys-diff` | research: BSDFs as tang IR graphs, path replay backprop on CPU and a GPU wavefront (unpublished; tang-compile is a git dep) |

Status: phases 0, 1, 2a–2d of the design. `irys-diff` fits a GGX material's
roughness and colour by path replay backprop on the GPU, through a TLAS of
instanced meshes with next-event estimation and MIS, and — phase 2d — the
*pose* of instances by edge-sampled visibility gradients. The hero is rose
full's left arm, read from rose's `.loon` CAD and tessellated by vcad: three
joint angles and the anodised finish recovered together from a target render
(to 0.06° and 0.005):

```
cargo test -p irys-diff --features gpu            # GPU tests skip without an adapter
cargo run --release -p irys-diff --features gpu --example rose -- out/rose   # needs ../rose (or $ROSE)
cargo run --release -p irys-diff --features gpu --example rose_fit -- out/rose_fit
cargo run -p irys-diff --features gpu --example rover -- out/rover
cargo run -p irys-diff --features gpu --example fit_scene -- out/fit_scene
```

![phase 2d rose](docs/phase2d-rose.png)

![phase 2d fit](docs/phase2d-fit.png)

![phase 2c rover](docs/phase2c-rover.png)

