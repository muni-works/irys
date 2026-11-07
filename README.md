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
| `irys-diff` | research: BSDFs as tang IR graphs, path replay backprop on CPU and a GPU wavefront |
| `irys-splat` | research: 3DGS trainer on irys's GPU tier (tile rasteriser forward + adjoint) |
| `irys-recon` | research: image to CAD, recover a vcad program and its dimensions from a photo |
| `irys-recon-net` | research: learned scene proposer seeding the recon MDL search |
| `irys-ffi` | C ABI rendering into an IOSurface-backed Metal texture |
| `irys-bench` | equal-time benchmark against Cycles |

Research crates are unpublished until tang is on crates.io (it is a git dependency); build them from this repo.

Status: phases 0–4b of the design (inverse rendering through 2f, the equal-time benchmark against Cycles, the OpenPBR look with volumes and hair), the belief codec's M1 prerequisites, splat training, and image-to-CAD v0/v1 ([docs/image-to-cad.md](docs/image-to-cad.md)). Highlights: `irys-diff` fits a GGX material's
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


## License

Apache-2.0, see [LICENSE](LICENSE).
