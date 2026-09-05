//! The top level: a hierarchy over placed instances, each pointing at a
//! shared bottom-level tree.
//!
//! # Why two levels
//!
//! A scene traced by looping over every object's tree and keeping the nearest
//! hit is O(objects) per ray, with no spatial culling *between* objects. The
//! top level is itself a SAH tree, built over per-instance world boxes, so a
//! ray descends only into the handful of objects whose bounds it crosses.
//!
//! # Why instancing
//!
//! Rather than baking each placement into a cloned copy of the geometry, the
//! ray is transformed into the instance's local space and traced against the
//! *shared* tree. A linear pattern of a hundred identical bolts builds one
//! hierarchy, not a hundred.
//!
//! The hit is mapped back out:
//!
//! - **`t`**: the local ray is renormalised, so `t_local` is in local units.
//!   With `L = |M⁻¹ d_world|` and `d_world` a unit vector, `t_world =
//!   t_local / L`. For a rigid placement `L == 1`; for a scaled one this is
//!   what keeps depth comparisons between instances meaningful.
//! - **normal**: by the inverse transpose of the upper-left 3×3, which is the
//!   covector rule. It is also what makes mirrored (negative-determinant)
//!   instances come out right: `M⁻ᵀ n` preserves outward-ness under any
//!   invertible map.

use std::sync::Arc;

use crate::bvh::Bvh;
use crate::geometry::Geometry;
use crate::math::{transform_aabb, Aabb, Dir3, Transform};
use crate::ray::{Hit, Ray};
use crate::sah::{item_bounds, sah_split, SahItem};

/// One placed instance of a shared bottom-level hierarchy.
#[derive(Debug, Clone)]
pub struct Instance<G> {
    /// The shared bottom-level tree.
    blas: Arc<Bvh<G>>,
    /// Object → world.
    to_world: Transform,
    /// World → object, precomputed for ray transformation.
    to_local: Transform,
    /// This instance's world bounds.
    world_aabb: Aabb,
    /// Caller-supplied index, echoed back on every hit.
    payload: usize,
}

impl<G: Geometry> Instance<G> {
    /// Place a shared tree with an object→world transform.
    ///
    /// `None` when the tree is empty or the transform is singular — a ray
    /// cannot be mapped into a collapsed space.
    pub fn new(blas: Arc<Bvh<G>>, to_world: Transform, payload: usize) -> Option<Self> {
        let local_bounds = blas.bounds()?;
        let to_local = to_world.inverse()?;
        let world_aabb = transform_aabb(&local_bounds, &to_world);
        Some(Self { blas, to_world, to_local, world_aabb, payload })
    }

    /// Place a shared tree at the identity.
    pub fn identity(blas: Arc<Bvh<G>>, payload: usize) -> Option<Self> {
        Self::new(blas, Transform::identity(), payload)
    }

    /// The caller-supplied payload index.
    pub fn payload(&self) -> usize {
        self.payload
    }

    /// This instance's world bounds.
    pub fn world_aabb(&self) -> Aabb {
        self.world_aabb
    }

    /// The object→world placement.
    pub fn to_world(&self) -> &Transform {
        &self.to_world
    }

    /// The bottom-level tree this instance shares.
    pub fn blas(&self) -> &Arc<Bvh<G>> {
        &self.blas
    }

    /// Map a world ray into local space.
    ///
    /// Returns the local ray plus `L`, the length of the un-normalised local
    /// direction: `t_local = L * t_world`.
    fn local_ray(&self, ray: &Ray) -> Option<(Ray, f64)> {
        let origin = self.to_local.apply_point(&ray.origin);
        let dir = self.to_local.apply_vec(&ray.direction.into_inner());
        let len = dir.norm();
        if !(len.is_finite() && len > 0.0) {
            return None;
        }
        Some((Ray::new(origin, dir), len))
    }

    /// Closest hit on this instance inside `(t_min, t_max)`, world units.
    ///
    /// Public because a scene may want to bypass the top level entirely —
    /// a single-object preview, or a test that checks the hierarchy against
    /// the brute-force answer.
    ///
    /// Both bounds scale by `len` on the way in, which is the whole reason
    /// `t_world = t_local / len` holds on the way out.
    pub fn trace_closest(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<InstanceHit> {
        let (local, len) = self.local_ray(ray)?;
        let local_hit = self.blas.trace_closest_range(&local, t_min * len, t_max * len)?;

        let t = local_hit.t / len;
        let normal = self
            .to_world
            .apply_normal(&local_hit.normal.into_inner())
            .try_normalize()
            .map(Dir3::new_unchecked)
            .unwrap_or(local_hit.normal);

        // `dpdu` is a *tangent*, not a normal: it transforms by the linear
        // part of the object→world matrix, not its inverse transpose.
        // Dropping it here would silently demote every anisotropic material
        // to isotropic, since the shading frame falls back when it is absent.
        let dpdu = local_hit
            .dpdu
            .map(|v| self.to_world.apply_vec(&v))
            .filter(|v| v.norm() > 0.0 && v.norm().is_finite());

        Some(InstanceHit {
            hit: Hit {
                t,
                point: ray.at(t),
                normal,
                uv: local_hit.uv,
                prim: local_hit.prim,
                payload: local_hit.payload,
                dpdu,
            },
            payload: self.payload,
        })
    }

    /// Any-hit against this instance over `(t_min, t_max)`, world units.
    pub fn occluded(&self, ray: &Ray, t_min: f64, t_max: f64) -> bool {
        match self.local_ray(ray) {
            Some((local, len)) => self.blas.occluded_range(&local, t_min * len, t_max * len),
            None => false,
        }
    }
}

/// A hit, tagged with the instance that produced it.
#[derive(Debug, Clone, Copy)]
pub struct InstanceHit {
    /// The intersection, in world space.
    pub hit: Hit,
    /// Payload of the instance that was hit.
    pub payload: usize,
}

/// A node of the top level.
#[derive(Debug, Clone)]
enum TlasNode {
    Leaf { aabb: Aabb, instances: Vec<u32> },
    Internal { aabb: Aabb, left: Box<TlasNode>, right: Box<TlasNode> },
}

impl TlasNode {
    fn aabb(&self) -> &Aabb {
        match self {
            TlasNode::Leaf { aabb, .. } | TlasNode::Internal { aabb, .. } => aabb,
        }
    }
}

/// A flattened top-level node, mirroring [`crate::bvh::FlatBvhNode`].
pub type FlatTlasNode = (Aabb, bool, u32, u32);

/// The scene: a hierarchy over placed [`Instance`]s.
#[derive(Debug, Clone)]
pub struct Tlas<G> {
    root: Option<TlasNode>,
    instances: Vec<Instance<G>>,
}

impl<G> Default for Tlas<G> {
    fn default() -> Self {
        Self { root: None, instances: Vec::new() }
    }
}

impl<G: Geometry> Tlas<G> {
    /// Build over the given instances, with the same SAH search the
    /// bottom level uses, applied to instance world boxes.
    pub fn build(instances: Vec<Instance<G>>) -> Self {
        if instances.is_empty() {
            return Self { root: None, instances };
        }

        let mut items: Vec<SahItem<u32>> = instances
            .iter()
            .enumerate()
            .map(|(i, inst)| {
                let aabb = inst.world_aabb;
                (i as u32, aabb, aabb.center())
            })
            .collect();

        let root = Some(build_node(&mut items));
        Self { root, instances }
    }

    /// Number of placed instances. Empty or singular ones are dropped at
    /// build time, so this can be fewer than were handed in.
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Whether the structure holds no instances.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// The instances, in build order.
    pub fn instances(&self) -> &[Instance<G>] {
        &self.instances
    }

    /// World bounds of the whole scene, if non-empty.
    pub fn bounds(&self) -> Option<Aabb> {
        self.root.as_ref().map(|n| *n.aabb())
    }

    /// Closest hit in the scene, in world space.
    pub fn trace_closest(&self, ray: &Ray) -> Option<InstanceHit> {
        self.trace_closest_range(ray, 0.0, f64::INFINITY)
    }

    /// Closest hit within the open interval `(t_min, t_max)`.
    ///
    /// `t_min` lets a caller skip the surface a ray just left without moving
    /// the origin — and because it is pushed down into the bottom level
    /// rather than applied as a post-filter, a surface hidden behind the
    /// skipped one is still found.
    pub fn trace_closest_range(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<InstanceHit> {
        let root = self.root.as_ref()?;
        let mut best: Option<InstanceHit> = None;
        let mut best_t = t_max;
        self.closest_node(ray, root, t_min, &mut best, &mut best_t);
        best
    }

    fn closest_node(
        &self,
        ray: &Ray,
        node: &TlasNode,
        t_min: f64,
        best: &mut Option<InstanceHit>,
        best_t: &mut f64,
    ) {
        let Some((enter, exit)) = ray.intersect_aabb(node.aabb()) else {
            return;
        };
        if enter >= *best_t || exit <= t_min {
            return;
        }

        match node {
            TlasNode::Leaf { instances, .. } => {
                for &i in instances {
                    let inst = &self.instances[i as usize];
                    if let Some(found) = inst.trace_closest(ray, t_min, *best_t) {
                        if found.hit.t < *best_t {
                            *best_t = found.hit.t;
                            *best = Some(found);
                        }
                    }
                }
            }
            TlasNode::Internal { left, right, .. } => {
                // Descend into the nearer child first so the far one is more
                // likely to be culled outright.
                let lt = ray.intersect_aabb(left.aabb()).map(|(t, _)| t);
                let rt = ray.intersect_aabb(right.aabb()).map(|(t, _)| t);
                let (first, second) = match (lt, rt) {
                    (Some(l), Some(r)) if r < l => (Some(right), Some(left)),
                    (Some(_), Some(_)) => (Some(left), Some(right)),
                    (Some(_), None) => (Some(left), None),
                    (None, Some(_)) => (Some(right), None),
                    (None, None) => (None, None),
                };
                if let Some(n) = first {
                    self.closest_node(ray, n, t_min, best, best_t);
                }
                if let Some(n) = second {
                    self.closest_node(ray, n, t_min, best, best_t);
                }
            }
        }
    }

    /// Any-hit: is anything in `(0, t_max)` along the ray?
    ///
    /// Returns on the first blocker found, at both levels.
    pub fn occluded(&self, ray: &Ray, t_max: f64) -> bool {
        self.occluded_range(ray, 0.0, t_max)
    }

    /// Any-hit over the open interval `(t_min, t_max)`.
    pub fn occluded_range(&self, ray: &Ray, t_min: f64, t_max: f64) -> bool {
        match self.root {
            Some(ref root) => self.occluded_node(ray, root, t_min, t_max),
            None => false,
        }
    }

    fn occluded_node(&self, ray: &Ray, node: &TlasNode, t_min: f64, t_max: f64) -> bool {
        let Some((enter, exit)) = ray.intersect_aabb(node.aabb()) else {
            return false;
        };
        if enter >= t_max || exit <= t_min {
            return false;
        }
        match node {
            TlasNode::Leaf { instances, .. } => instances
                .iter()
                .any(|&i| self.instances[i as usize].occluded(ray, t_min, t_max)),
            TlasNode::Internal { left, right, .. } => {
                self.occluded_node(ray, left, t_min, t_max)
                    || self.occluded_node(ray, right, t_min, t_max)
            }
        }
    }

    /// Flatten the top level for GPU upload, mirroring [`Bvh::flatten`].
    ///
    /// Returns the node array plus the instance indices in leaf order; a
    /// leaf's `(left_or_first, right_or_count)` slices into that list.
    pub fn flatten(&self) -> (Vec<FlatTlasNode>, Vec<u32>) {
        let mut nodes = Vec::new();
        let mut indices = Vec::new();
        if let Some(root) = &self.root {
            flatten_node(root, &mut nodes, &mut indices);
        }
        (nodes, indices)
    }
}

fn flatten_node(node: &TlasNode, nodes: &mut Vec<FlatTlasNode>, indices: &mut Vec<u32>) -> usize {
    let idx = nodes.len();
    match node {
        TlasNode::Leaf { aabb, instances } => {
            let start = indices.len() as u32;
            indices.extend(instances.iter().copied());
            nodes.push((*aabb, true, start, instances.len() as u32));
        }
        TlasNode::Internal { aabb, left, right } => {
            nodes.push((*aabb, false, 0, 0));
            let l = flatten_node(left, nodes, indices);
            let r = flatten_node(right, nodes, indices);
            nodes[idx].2 = l as u32;
            nodes[idx].3 = r as u32;
        }
    }
    idx
}

/// Recursively build the top level. Leaves hold few instances, since each one
/// is itself an expensive descent.
fn build_node(items: &mut [SahItem<u32>]) -> TlasNode {
    let bounds = item_bounds(items);

    if items.len() <= 2 {
        return TlasNode::Leaf {
            aabb: bounds,
            instances: items.iter().map(|(i, _, _)| *i).collect(),
        };
    }

    let mid = sah_split(items, &bounds);
    let (left, right) = items.split_at_mut(mid);
    TlasNode::Internal {
        aabb: bounds,
        left: Box::new(build_node(left)),
        right: Box::new(build_node(right)),
    }
}
