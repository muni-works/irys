# kosm-render

The light.

Rays, acceleration structures and (soon) the integrator, over anything that
can be bounded and hit.

## The seam

`Geometry` is the entire boundary between this crate and geometry:

```rust
pub trait Geometry {
    fn len(&self) -> usize;
    fn bounds(&self, i: usize) -> Aabb;
    fn intersect(&self, ray: &Ray, i: usize, t_min: f64, t_max: f64) -> Option<Hit>;
    fn intersect_all(&self, ray: &Ray, i: usize, out: &mut Vec<Hit>) { .. }
    fn occludes(&self, ray: &Ray, i: usize, t_min: f64, t_max: f64) -> bool { .. }
}
```

How many primitives you have, where each one is, and what a ray finds when it
meets one. Implement it and you get `Bvh<G>` over your primitives and
`Tlas<G>` over placed instances of them.

Primitives are addressed by a flat index. That index comes back on every
`Hit`, which is how the caller recovers whatever *it* calls the thing — a
`FaceId`, a triangle, a body, a splat. The renderer never learns the name.

vcad implements it over the trimmed analytic faces of a B-rep solid
(`vcad-kernel-raytrace::BrepGeom`), which is why `intersect/` and `trim.rs`
stayed there: they are geometry, not light. `TriMesh` is built in, because
every renderer needs triangles at least once.

## The dependency rule

`kosm-render` depends on **`tang`**, **`rayon`**, and later **`wgpu`** /
**`bytemuck`**. Never on `vcad-*`, `phyz-*`, or any other Kosm crate. It is a
leaf, and it has to stay one — vcad is a *client* of this crate, not a
sibling.

It must build for the browser:

```bash
cargo check -p kosm-render --target wasm32-unknown-unknown
```

`rayon` is therefore behind a `cfg(not(target_arch = "wasm32"))` dependency,
not a hard one.

## Math

`Point3`, `Vec3`, `Dir3`, `Point2`, `Vec2` are `tang`'s types monomorphised
to `f64` — the same types vcad's kernel math and phyz's poses are already
made of, so a geometry crate handing us a point does not have to convert one.
`Transform` is a single `tang::Mat4<f64>`: full affine, because an instance
may be scaled or mirrored and the tracer has to map rays and normals through
both.
