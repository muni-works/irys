//! Speed piece 1: the scene's opaque geometry as hardware acceleration
//! structures (wgpu 30 ray queries; Metal's on Apple silicon).
//!
//! Per bottom-level tree, up to two BLASes: its triangles (opaque, in mesh
//! order, so a hit's `primitive_index` is the prim record's offset from the
//! tree's first) and its analytic prims as AABBs (in `analytic` order, after
//! the triangles in the records), intersected in the shader by the software
//! walk's own `hit_prim`. One TLAS instance per (placement, BLAS), its
//! transform the placement's object → world; `rt_inst` maps a TLAS instance
//! back to (instance record, first prim record). `gpu/rt.wgsl` is the
//! shader side.

use irys_gpu::wgpu::{self, util::DeviceExt};
use irys_gpu::GpuContext;

use irys_core::Geometry;

use crate::prb::Scene;

/// The acceleration structures of one [`Scene`].
pub(crate) struct Accel {
    /// Per bottom-level tree: (triangle BLAS, AABB BLAS), either absent
    /// when the tree has none of that kind.
    blas: Vec<(Option<wgpu::Blas>, Option<wgpu::Blas>)>,
    pub(crate) tlas: wgpu::Tlas,
    pub(crate) map: wgpu::Buffer,
    /// TLAS instances the current build uses.
    max: u32,
    /// The BLASes' vertex, index and box buffers.
    _inputs: Vec<wgpu::Buffer>,
}

/// Whether this device traces through ray queries.
pub(crate) fn available(ctx: &GpuContext) -> bool {
    ctx.device
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
}

const FAST: wgpu::AccelerationStructureFlags = wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE;

impl Accel {
    /// Build every BLAS and the TLAS for `scene`. `None` without the feature.
    pub(crate) fn new(ctx: &GpuContext, scene: &Scene) -> Option<Self> {
        if !available(ctx) {
            return None;
        }
        use irys_prims::analytic::Prim;
        let dev = &ctx.device;
        let mut blas = Vec::new();
        let mut keep = Vec::new();
        for b in &scene.blas {
            let o = b.geometry();
            let mesh = o.mesh();
            let tri = (mesh.len() > 0).then(|| {
                let v: Vec<[f32; 3]> = mesh
                    .positions()
                    .iter()
                    .map(|p| [p.x as f32, p.y as f32, p.z as f32])
                    .collect();
                let idx: Vec<u32> = mesh.triangles().iter().flatten().copied().collect();
                let vb = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rt vertices"),
                    contents: bytemuck::cast_slice(&v),
                    usage: wgpu::BufferUsages::BLAS_INPUT,
                });
                let ib = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rt indices"),
                    contents: bytemuck::cast_slice(&idx),
                    usage: wgpu::BufferUsages::BLAS_INPUT,
                });
                let size = wgpu::BlasTriangleGeometrySizeDescriptor {
                    vertex_format: wgpu::VertexFormat::Float32x3,
                    vertex_count: v.len() as u32,
                    index_format: Some(wgpu::IndexFormat::Uint32),
                    index_count: Some(idx.len() as u32),
                    flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
                };
                let bl = dev.create_blas(
                    &wgpu::CreateBlasDescriptor {
                        label: Some("rt triangles"),
                        flags: FAST,
                        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                    },
                    wgpu::BlasGeometrySizeDescriptors::Triangles {
                        descriptors: vec![size.clone()],
                    },
                );
                (bl, vb, ib, size)
            });
            let prims = o.analytic().prims();
            let aabb = (!prims.is_empty()).then(|| {
                let boxes: Vec<[f32; 6]> = prims
                    .iter()
                    .map(|p| {
                        let (c, e) = match *p {
                            Prim::Sphere { center, radius } => (center, [radius; 3]),
                            Prim::Disc {
                                center,
                                normal,
                                radius,
                            } => {
                                let n = [normal.x, normal.y, normal.z];
                                (center, n.map(|k| radius * (1.0 - k * k).max(0.0).sqrt()))
                            }
                            other => panic!("irys-diff rt: {other:?} has no intersector"),
                        };
                        // Padded by a relative margin so the f32 box never
                        // clips the prim's own f32 intersection.
                        let c = [c.x, c.y, c.z];
                        let pad = |i: usize| e[i] * 1e-4 + 1e-6 * (c[i].abs() + e[i]);
                        let lo = |i: usize| (c[i] - e[i] - pad(i)) as f32;
                        let hi = |i: usize| (c[i] + e[i] + pad(i)) as f32;
                        [lo(0), lo(1), lo(2), hi(0), hi(1), hi(2)]
                    })
                    .collect();
                let ab = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rt aabbs"),
                    contents: bytemuck::cast_slice(&boxes),
                    usage: wgpu::BufferUsages::BLAS_INPUT,
                });
                let size = wgpu::BlasAABBGeometrySizeDescriptor {
                    primitive_count: boxes.len() as u32,
                    flags: wgpu::AccelerationStructureGeometryFlags::empty(),
                };
                let bl = dev.create_blas(
                    &wgpu::CreateBlasDescriptor {
                        label: Some("rt aabbs"),
                        flags: FAST,
                        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                    },
                    wgpu::BlasGeometrySizeDescriptors::AABBs {
                        descriptors: vec![size.clone()],
                    },
                );
                (bl, ab, size)
            });
            keep.push((tri, aabb));
        }
        {
            let entries: Vec<wgpu::BlasBuildEntry> = keep
                .iter()
                .flat_map(|(tri, aabb)| {
                    let t = tri.as_ref().map(|(bl, vb, ib, size)| wgpu::BlasBuildEntry {
                        blas: bl,
                        geometry: wgpu::BlasGeometries::TriangleGeometries(vec![
                            wgpu::BlasTriangleGeometry {
                                size,
                                vertex_buffer: vb,
                                first_vertex: 0,
                                vertex_stride: 12,
                                index_buffer: Some(ib),
                                first_index: Some(0),
                                transform_buffer: None,
                                transform_buffer_offset: None,
                            },
                        ]),
                    });
                    let a = aabb.as_ref().map(|(bl, ab, size)| wgpu::BlasBuildEntry {
                        blas: bl,
                        geometry: wgpu::BlasGeometries::AabbGeometries(vec![
                            wgpu::BlasAabbGeometry {
                                size,
                                stride: 24,
                                aabb_buffer: ab,
                                primitive_offset: 0,
                            },
                        ]),
                    });
                    t.into_iter().chain(a)
                })
                .collect();
            // One submit per BLAS, conservatively: wgpu 30's Metal backend
            // places no barrier between the builds of one call
            // (`place_acceleration_structure_barrier` is empty).
            for e in &entries {
                let mut enc =
                    dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                enc.build_acceleration_structures(std::iter::once(e), std::iter::empty());
                ctx.queue.submit([enc.finish()]);
            }
        }
        // And finished before anything reads them: without this wait a
        // whole BLAS was missing from the traced frames, varying run to run
        // (measured on the M4 Max; the TLAS build and the trace are not
        // ordered after the BLAS builds).
        wait(ctx);
        let mut inputs = Vec::new();
        for (tri, aabb) in keep {
            blas.push((
                tri.map(|t| {
                    inputs.extend([t.1, t.2]);
                    t.0
                }),
                aabb.map(|a| {
                    inputs.push(a.1);
                    a.0
                }),
            ));
        }
        let max = (2 * scene.placements().len()).max(1) as u32;
        let tlas = dev.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("rt tlas"),
            max_instances: max,
            flags: FAST,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });
        let map = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rt instance map"),
            size: max as u64 * 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut a = Self {
            blas,
            tlas,
            map,
            max,
            _inputs: inputs,
        };
        a.update(ctx, scene);
        Some(a)
    }

    /// Rebuild the TLAS from the scene's placements (after
    /// [`Scene::place`]); the BLASes are the trees this was built with.
    pub(crate) fn update(&mut self, ctx: &GpuContext, scene: &Scene) {
        // The first prim record of each bottom-level tree, as `gpu::pack`
        // lays them out: triangles then analytic prims, tree after tree.
        let mut first = Vec::new();
        let mut at = 0u32;
        for b in &scene.blas {
            let o = b.geometry();
            first.push((at, at + o.mesh().len() as u32));
            at += (o.mesh().len() + o.analytic().prims().len()) as u32;
        }
        let mut map: Vec<[u32; 2]> = Vec::new();
        let mut n = 0usize;
        for k in 0..self.max as usize {
            self.tlas[k] = None;
        }
        for (k, i) in scene.tlas.instances().iter().enumerate() {
            let place = &scene.placements[i.payload()];
            let m = i.to_world().matrix;
            let mut xf = [0f32; 12];
            for r in 0..3 {
                for c in 0..4 {
                    xf[4 * r + c] = m.get(r, c) as f32;
                }
            }
            let (tri, aabb) = &self.blas[place.blas];
            let (ft, fa) = first[place.blas];
            for (bl, f) in [(tri, ft), (aabb, fa)] {
                if let Some(bl) = bl {
                    self.tlas[n] = Some(wgpu::TlasInstance::new(bl, xf, k as u32, 0xff));
                    map.push([k as u32, f]);
                    n += 1;
                }
            }
        }
        assert!(n <= self.max as usize, "rt: more instances than built for");
        if map.is_empty() {
            map.push([0, 0]);
        }
        ctx.queue
            .write_buffer(&self.map, 0, bytemuck::cast_slice(&map));
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.build_acceleration_structures(std::iter::empty(), [&self.tlas]);
        ctx.queue.submit([enc.finish()]);
        wait(ctx);
    }
}

/// Block until the queue is idle.
fn wait(ctx: &GpuContext) {
    ctx.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("device lost");
}
