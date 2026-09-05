//! The bottom level: one hierarchy over one geometry's primitives.

use crate::geometry::Geometry;
use crate::math::Aabb;
use crate::ray::{Hit, Ray};
use crate::sah::{item_bounds, sah_split, SahItem};

/// A flattened node for GPU upload: `(AABB, is_leaf, left_or_first,
/// right_or_count)`.
pub type FlatBvhNode = (Aabb, bool, u32, u32);

/// A node: a leaf holding primitive indices, or an internal node with two
/// children.
#[derive(Debug, Clone)]
pub enum BvhNode {
    /// Leaf node holding primitive indices into the owning geometry.
    Leaf {
        /// Bounds of this node.
        aabb: Aabb,
        /// The primitives gathered here.
        prims: Vec<u32>,
    },
    /// Internal node with two children.
    Internal {
        /// Bounds of this node.
        aabb: Aabb,
        /// Left child.
        left: Box<BvhNode>,
        /// Right child.
        right: Box<BvhNode>,
    },
}

impl BvhNode {
    /// This node's bounds.
    #[inline]
    pub fn aabb(&self) -> &Aabb {
        match self {
            BvhNode::Leaf { aabb, .. } | BvhNode::Internal { aabb, .. } => aabb,
        }
    }
}

/// One primitive's build-time record.
type PrimData = SahItem<u32>;

/// A bounding volume hierarchy over a [`Geometry`], built with the surface
/// area heuristic.
///
/// It owns the geometry, because a hierarchy over primitives someone else may
/// mutate is a hierarchy that is wrong. Share it by sharing the whole thing:
/// an `Arc<Bvh<G>>` is what an instance holds.
#[derive(Debug, Clone)]
pub struct Bvh<G> {
    root: Option<BvhNode>,
    geom: G,
}

impl<G: Geometry> Bvh<G> {
    /// Build a hierarchy over a geometry.
    pub fn build(geom: G) -> Self {
        let mut prim_data: Vec<PrimData> = (0..geom.len())
            .map(|i| {
                let aabb = geom.bounds(i);
                (i as u32, aabb, aabb.center())
            })
            .collect();

        let root = if prim_data.is_empty() {
            None
        } else {
            Some(build_node(&mut prim_data))
        };

        Self { root, geom }
    }

    /// The geometry this was built over.
    pub fn geometry(&self) -> &G {
        &self.geom
    }

    /// The root node, if the geometry had any primitives.
    pub fn root(&self) -> Option<&BvhNode> {
        self.root.as_ref()
    }

    /// Bounds of the whole hierarchy, in the geometry's own space.
    pub fn bounds(&self) -> Option<Aabb> {
        self.root.as_ref().map(|n| *n.aabb())
    }

    /// Every intersection along the ray, sorted by `t`.
    pub fn trace(&self, ray: &Ray) -> Vec<Hit> {
        let mut hits = Vec::new();

        if let Some(ref root) = self.root {
            self.trace_node(ray, root, &mut hits);
        }

        hits.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));
        hits
    }

    /// The closest hit.
    pub fn trace_closest(&self, ray: &Ray) -> Option<Hit> {
        self.trace_closest_limit(ray, f64::INFINITY)
    }

    /// The closest hit strictly nearer than `t_max`.
    ///
    /// Seeding the search with a known upper bound lets the early-outs prune
    /// subtrees a caller has already beaten — the basis of TLAS traversal,
    /// where each instance inherits the best `t` found so far.
    pub fn trace_closest_limit(&self, ray: &Ray, t_max: f64) -> Option<Hit> {
        self.trace_closest_range(ray, 0.0, t_max)
    }

    /// The closest hit in the open interval `(t_min, t_max)`.
    ///
    /// `t_min` is how callers dodge self-intersection without nudging the ray
    /// origin along the normal. It is a genuine interval search, not a
    /// post-filter: a hit at or before `t_min` is skipped and the search
    /// continues past it, so a surface hidden behind one is still found.
    pub fn trace_closest_range(&self, ray: &Ray, t_min: f64, t_max: f64) -> Option<Hit> {
        let mut closest: Option<Hit> = None;
        let mut closest_t = t_max;

        if let Some(ref root) = self.root {
            self.trace_node_closest(ray, root, t_min, &mut closest, &mut closest_t);
        }

        closest
    }

    /// Any-hit: does the ray hit anything in `(0, t_max)`?
    ///
    /// Returns as soon as one hit is found — strictly less work than finding
    /// the nearest, and the traversal shadow rays want.
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

    fn occluded_node(&self, ray: &Ray, node: &BvhNode, t_min: f64, t_max: f64) -> bool {
        let Some((enter, exit)) = ray.intersect_aabb(node.aabb()) else {
            return false;
        };
        // Prune boxes wholly outside the interval on either side.
        if enter >= t_max || exit <= t_min {
            return false;
        }
        match node {
            BvhNode::Leaf { prims, .. } => prims
                .iter()
                .any(|&prim| self.geom.occludes(ray, prim as usize, t_min, t_max)),
            BvhNode::Internal { left, right, .. } => {
                self.occluded_node(ray, left, t_min, t_max)
                    || self.occluded_node(ray, right, t_min, t_max)
            }
        }
    }

    fn trace_node(&self, ray: &Ray, node: &BvhNode, hits: &mut Vec<Hit>) {
        match node {
            BvhNode::Leaf { aabb, prims } => {
                if ray.intersect_aabb(aabb).is_some() {
                    for &prim in prims {
                        self.geom.intersect_all(ray, prim as usize, hits);
                    }
                }
            }
            BvhNode::Internal { aabb, left, right } => {
                if ray.intersect_aabb(aabb).is_some() {
                    self.trace_node(ray, left, hits);
                    self.trace_node(ray, right, hits);
                }
            }
        }
    }

    fn trace_node_closest(
        &self,
        ray: &Ray,
        node: &BvhNode,
        t_min: f64,
        closest: &mut Option<Hit>,
        closest_t: &mut f64,
    ) {
        let Some((enter, exit)) = ray.intersect_aabb(node.aabb()) else {
            return;
        };
        // Early out if the box is beyond the current closest, or entirely
        // behind `t_min`.
        if enter >= *closest_t || exit <= t_min {
            return;
        }

        match node {
            BvhNode::Leaf { prims, .. } => {
                for &prim in prims {
                    // `closest_t` is the upper bound the geometry may prune
                    // against: nothing further away can win.
                    if let Some(hit) = self.geom.intersect(ray, prim as usize, t_min, f64::INFINITY) {
                        if hit.t < *closest_t {
                            *closest_t = hit.t;
                            *closest = Some(hit);
                        }
                    }
                }
            }
            BvhNode::Internal { left, right, .. } => {
                // Descend into the nearer child first, so the far one is more
                // likely to be culled outright.
                let left_t = ray.intersect_aabb(left.aabb()).map(|(t, _)| t);
                let right_t = ray.intersect_aabb(right.aabb()).map(|(t, _)| t);

                match (left_t, right_t) {
                    (Some(lt), Some(rt)) => {
                        if lt < rt {
                            self.trace_node_closest(ray, left, t_min, closest, closest_t);
                            self.trace_node_closest(ray, right, t_min, closest, closest_t);
                        } else {
                            self.trace_node_closest(ray, right, t_min, closest, closest_t);
                            self.trace_node_closest(ray, left, t_min, closest, closest_t);
                        }
                    }
                    (Some(_), None) => {
                        self.trace_node_closest(ray, left, t_min, closest, closest_t);
                    }
                    (None, Some(_)) => {
                        self.trace_node_closest(ray, right, t_min, closest, closest_t);
                    }
                    (None, None) => {}
                }
            }
        }
    }

    /// Flatten into a node array for GPU upload.
    ///
    /// Returns `(AABB, is_leaf, left_or_first, right_or_count)` tuples — for
    /// an internal node the two child indices, for a leaf the start and count
    /// into the returned primitive-index list, which is the leaf order.
    pub fn flatten(&self) -> (Vec<FlatBvhNode>, Vec<u32>) {
        let mut nodes = Vec::new();
        let mut prims = Vec::new();

        if let Some(root) = &self.root {
            flatten_node(root, &mut nodes, &mut prims);
        }

        (nodes, prims)
    }
}

/// Recursively flatten a node.
fn flatten_node(node: &BvhNode, nodes: &mut Vec<FlatBvhNode>, prims: &mut Vec<u32>) -> usize {
    let idx = nodes.len();

    match node {
        BvhNode::Leaf { aabb, prims: leaf } => {
            let start = prims.len() as u32;
            let count = leaf.len() as u32;
            prims.extend(leaf.iter().copied());
            nodes.push((*aabb, true, start, count));
        }
        BvhNode::Internal { aabb, left, right } => {
            // Reserve this node's slot, then patch in the child indices.
            nodes.push((*aabb, false, 0, 0));
            let left_idx = flatten_node(left, nodes, prims);
            let right_idx = flatten_node(right, nodes, prims);
            nodes[idx].2 = left_idx as u32;
            nodes[idx].3 = right_idx as u32;
        }
    }

    idx
}

/// Build a node recursively.
fn build_node(data: &mut [PrimData]) -> BvhNode {
    let bounds = item_bounds(data);

    if data.len() <= 4 {
        return BvhNode::Leaf {
            aabb: bounds,
            prims: data.iter().map(|(id, _, _)| *id).collect(),
        };
    }

    let mid = sah_split(data, &bounds);
    let (left, right) = data.split_at_mut(mid);

    BvhNode::Internal {
        aabb: bounds,
        left: Box::new(build_node(left)),
        right: Box::new(build_node(right)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::TriMesh;
    use crate::math::{Point3, Vec3};

    /// A unit cube as twelve triangles.
    fn cube() -> TriMesh {
        let p = |x, y, z| Point3::new(x, y, z);
        let positions = vec![
            p(0.0, 0.0, 0.0), p(1.0, 0.0, 0.0), p(1.0, 1.0, 0.0), p(0.0, 1.0, 0.0),
            p(0.0, 0.0, 1.0), p(1.0, 0.0, 1.0), p(1.0, 1.0, 1.0), p(0.0, 1.0, 1.0),
        ];
        let indices = [
            0, 2, 1, 0, 3, 2, // -z
            4, 5, 6, 4, 6, 7, // +z
            0, 1, 5, 0, 5, 4, // -y
            3, 7, 6, 3, 6, 2, // +y
            0, 4, 7, 0, 7, 3, // -x
            1, 2, 6, 1, 6, 5, // +x
        ];
        TriMesh::new(positions, Vec::new(), &indices)
    }

    #[test]
    fn a_ray_through_a_cube_enters_and_leaves() {
        let bvh = Bvh::build(cube());
        let ray = Ray::new(Point3::new(-1.0, 0.35, 0.6), Vec3::new(1.0, 0.0, 0.0));
        let hits = bvh.trace(&ray);
        assert_eq!(hits.len(), 2, "in one face and out the other");
        assert!((hits[0].t - 1.0).abs() < 1e-9);
        assert!((hits[1].t - 2.0).abs() < 1e-9);
        assert!(hits[0].t <= hits[1].t, "sorted by t");
    }

    #[test]
    fn the_closest_hit_is_the_near_face() {
        let bvh = Bvh::build(cube());
        let ray = Ray::new(Point3::new(-1.0, 0.35, 0.6), Vec3::new(1.0, 0.0, 0.0));
        let hit = bvh.trace_closest(&ray).expect("the cube is in the way");
        assert!((hit.t - 1.0).abs() < 1e-9);
    }

    #[test]
    fn t_min_skips_past_the_near_face_without_losing_the_far_one() {
        let bvh = Bvh::build(cube());
        let ray = Ray::new(Point3::new(-1.0, 0.35, 0.6), Vec3::new(1.0, 0.0, 0.0));
        let hit = bvh
            .trace_closest_range(&ray, 1.5, f64::INFINITY)
            .expect("the far face is still there");
        assert!((hit.t - 2.0).abs() < 1e-9);
    }

    #[test]
    fn occlusion_agrees_with_the_closest_hit() {
        let bvh = Bvh::build(cube());
        let ray = Ray::new(Point3::new(-1.0, 0.35, 0.6), Vec3::new(1.0, 0.0, 0.0));
        assert!(bvh.occluded(&ray, 10.0));
        assert!(!bvh.occluded(&ray, 0.5), "the box is beyond the limit");
    }

    #[test]
    fn an_empty_geometry_traces_nothing() {
        let bvh = Bvh::build(TriMesh::default());
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0));
        assert!(bvh.bounds().is_none());
        assert!(bvh.trace_closest(&ray).is_none());
        assert!(!bvh.occluded(&ray, f64::INFINITY));
    }

    #[test]
    fn flatten_round_trips_every_primitive_once() {
        let bvh = Bvh::build(cube());
        let (nodes, prims) = bvh.flatten();
        assert!(!nodes.is_empty());
        let mut sorted = prims.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 12, "all twelve triangles, each once");
    }
}
