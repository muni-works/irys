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

Status: phases 0, 1, 2a and 2b of the design. `irys-diff` fits a GGX ball's
roughness and colour by path replay backprop on the GPU, in a scene of a
triangle mesh, analytic prims and Gaussian splats:

```
cargo test -p irys-diff --features gpu            # GPU tests skip without an adapter
cargo run -p irys-diff --features gpu --example fit_scene -- out/fit_scene
```

![phase 2b beauty](docs/phase2b-beauty.png)
