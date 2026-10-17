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

Status: phases 0, 1, 2a, 2b and 2c of the design. `irys-diff` fits a GGX
material's roughness and colour by path replay backprop on the GPU, through a
TLAS of instanced meshes with next-event estimation and MIS, at any scale and
frame size — here on muni's BVR1 rover, built from vcad parts:

```
cargo test -p irys-diff --features gpu            # GPU tests skip without an adapter
cargo run -p irys-diff --features gpu --example rover -- out/rover
cargo run -p irys-diff --features gpu --example fit_scene -- out/fit_scene
```

![phase 2c rover](docs/phase2c-rover.png)

