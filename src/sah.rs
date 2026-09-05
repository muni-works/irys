//! The split search both levels of the hierarchy share.
//!
//! A bucketed surface-area heuristic over items that carry a payload, their
//! bounds and their centroid. Generic over the payload so the per-object BLAS
//! (primitives) and the scene TLAS (instances) run one implementation rather
//! than two copies that drift.

use crate::math::{Aabb, Point3, Vec3};

/// An item to be partitioned: a payload, its bounds, its centroid.
pub type SahItem<T> = (T, Aabb, Point3);

/// Bounds of a slice of items.
pub fn item_bounds<T>(items: &[SahItem<T>]) -> Aabb {
    let mut bounds = Aabb::empty();
    for (_, aabb, _) in items {
        bounds.include(aabb);
    }
    bounds
}

/// Split a slice in two, returning the midpoint index.
///
/// Runs the bucketed SAH search and falls back to a median split when the
/// chosen plane leaves one side empty. Always returns `1..items.len()`, so
/// callers can recurse unconditionally.
pub fn sah_split<T>(items: &mut [SahItem<T>], bounds: &Aabb) -> usize {
    let (axis, pos) = find_best_split(items, bounds);
    let mid = partition_items(items, axis, pos);
    if mid == 0 || mid == items.len() {
        items.len() / 2
    } else {
        mid
    }
}

/// Find the best split axis and position.
fn find_best_split<T>(data: &[SahItem<T>], bounds: &Aabb) -> (usize, f64) {
    const NUM_BUCKETS: usize = 12;

    let extent = Vec3::new(
        bounds.max.x - bounds.min.x,
        bounds.max.y - bounds.min.y,
        bounds.max.z - bounds.min.z,
    );

    let mut best_cost = f64::INFINITY;
    let mut best_axis = 0;
    let mut best_pos = 0.0;

    for axis in 0..3 {
        let axis_extent = match axis {
            0 => extent.x,
            1 => extent.y,
            _ => extent.z,
        };

        if axis_extent < 1e-10 {
            continue;
        }

        let axis_min = match axis {
            0 => bounds.min.x,
            1 => bounds.min.y,
            _ => bounds.min.z,
        };

        let mut bucket_counts = [0usize; NUM_BUCKETS];
        let mut bucket_bounds = [Aabb::empty(); NUM_BUCKETS];

        for (_, aabb, centroid) in data {
            let c = match axis {
                0 => centroid.x,
                1 => centroid.y,
                _ => centroid.z,
            };

            let b = ((c - axis_min) / axis_extent * NUM_BUCKETS as f64) as usize;
            let b = b.min(NUM_BUCKETS - 1);

            bucket_counts[b] += 1;
            bucket_bounds[b].include(aabb);
        }

        for split in 1..NUM_BUCKETS {
            let mut left_count = 0;
            let mut left_bounds = Aabb::empty();
            for i in 0..split {
                left_count += bucket_counts[i];
                if bucket_counts[i] > 0 {
                    left_bounds.include(&bucket_bounds[i]);
                }
            }

            let mut right_count = 0;
            let mut right_bounds = Aabb::empty();
            for i in split..NUM_BUCKETS {
                right_count += bucket_counts[i];
                if bucket_counts[i] > 0 {
                    right_bounds.include(&bucket_bounds[i]);
                }
            }

            if left_count == 0 || right_count == 0 {
                continue;
            }

            // Traversal cost, plus each side's area share times its load.
            let cost = 0.125
                + left_bounds.surface_area() / bounds.surface_area() * left_count as f64
                + right_bounds.surface_area() / bounds.surface_area() * right_count as f64;

            if cost < best_cost {
                best_cost = cost;
                best_axis = axis;
                best_pos = axis_min + (split as f64 / NUM_BUCKETS as f64) * axis_extent;
            }
        }
    }

    (best_axis, best_pos)
}

/// Partition items by centroid along an axis.
fn partition_items<T>(data: &mut [SahItem<T>], axis: usize, pos: f64) -> usize {
    let mut left = 0;
    let mut right = data.len();

    while left < right {
        let c = match axis {
            0 => data[left].2.x,
            1 => data[left].2.y,
            _ => data[left].2.z,
        };

        if c < pos {
            left += 1;
        } else {
            right -= 1;
            data.swap(left, right);
        }
    }

    left
}
