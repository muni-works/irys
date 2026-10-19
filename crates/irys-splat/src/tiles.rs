//! Tile binning on the host: every visible splat is listed in each 16 × 16
//! tile its 1/255 contour's bounding square touches, each tile's list in
//! `(depth, index)` order. No key sort: the splats are depth-sorted once
//! (n log n over splats, not over intersections) and scattered into tile
//! buckets in that order by a counting pass, so each bucket comes out
//! sorted. Chunked over rayon with per-chunk histograms and a prefix, so
//! the result does not depend on the thread count.
//!
//! The scatter also writes the splat-major view (`csr_*`): for splat `i`,
//! the slots it occupies, in tile order. The device's adjoint writes one
//! gradient per slot and the reduction sums each splat's slots in that
//! fixed order (the "sorted segments" pattern: no float atomics).

use rayon::prelude::*;

use crate::project::Proj;

pub const TILE: usize = 16;

#[derive(Debug, Default, Clone)]
pub struct Bins {
    pub tiles_x: usize,
    pub tiles_y: usize,
    /// `[start, end)` into `ids` per tile.
    pub ranges: Vec<[u32; 2]>,
    /// Splat index per slot.
    pub ids: Vec<u32>,
    /// Per splat `[csr_off[i], csr_off[i + 1])` into `csr_slots`.
    pub csr_off: Vec<u32>,
    pub csr_slots: Vec<u32>,
}

struct Ptr(*mut u32);
unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}
impl Ptr {
    /// # Safety
    /// Every index is written by one thread only.
    unsafe fn set(&self, i: usize, v: u32) {
        unsafe { *self.0.add(i) = v }
    }
}

/// The tile rectangle `[x0, x1) × [y0, y1)` of a splat.
pub fn rect(k: [f32; 4], w: usize, h: usize) -> Option<[usize; 4]> {
    let (u, v, r) = (k[0] as f64, k[1] as f64, k[2] as f64);
    if !(r > 0.0) {
        return None;
    }
    let lo_x = (u - r).ceil().max(0.0);
    let hi_x = (u + r).floor().min(w as f64 - 1.0);
    let lo_y = (v - r).ceil().max(0.0);
    let hi_y = (v + r).floor().min(h as f64 - 1.0);
    if !(lo_x <= hi_x && lo_y <= hi_y) {
        return None;
    }
    let t = |v: f64| v as usize / TILE;
    Some([t(lo_x), t(hi_x) + 1, t(lo_y), t(hi_y) + 1])
}

/// `(u, v, radius, z)` per splat from host projections (radius 0: culled).
pub fn keys(p: &[Option<Proj>]) -> Vec<[f32; 4]> {
    p.iter().map(|q| q.map_or([0.0; 4], |q| [q.uv[0] as f32, q.uv[1] as f32, q.radius as f32, q.colour[3] as f32])).collect()
}

pub fn bin(p: &[Option<Proj>], w: usize, h: usize) -> Bins {
    bin_keys(&keys(p), w, h)
}

/// Bin splats given as `(u, v, radius, z)` (the device's `project` output).
pub fn bin_keys(k: &[[f32; 4]], w: usize, h: usize) -> Bins {
    let (tx, ty) = (w.div_ceil(TILE), h.div_ceil(TILE));
    let nt = tx * ty;
    let n = k.len();
    let rects: Vec<Option<[usize; 4]>> = k.par_iter().map(|q| rect(*q, w, h)).collect();
    let mut order: Vec<(f32, u32)> = (0..n).filter(|&i| rects[i].is_some()).map(|i| (k[i][3], i as u32)).collect();
    order.par_sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    // splat-major offsets
    let mut csr_off = vec![0u32; n + 1];
    for i in 0..n {
        let c = rects[i].map_or(0, |r| (r[1] - r[0]) * (r[3] - r[2]));
        csr_off[i + 1] = csr_off[i] + c as u32;
    }
    let total = csr_off[n] as usize;
    // per-chunk tile histograms
    let chunk = order.len().div_ceil(64).max(1024);
    let chunks: Vec<&[(f32, u32)]> = order.chunks(chunk).collect();
    let hist: Vec<Vec<u32>> = chunks
        .par_iter()
        .map(|c| {
            let mut hh = vec![0u32; nt];
            for &(_, i) in c.iter() {
                let r = rects[i as usize].unwrap();
                for y in r[2]..r[3] {
                    for x in r[0]..r[1] {
                        hh[y * tx + x] += 1;
                    }
                }
            }
            hh
        })
        .collect();
    let mut ranges = vec![[0u32; 2]; nt];
    let mut base = vec![vec![0u32; nt]; chunks.len()];
    let mut at = 0u32;
    for t in 0..nt {
        ranges[t][0] = at;
        for (c, hh) in hist.iter().enumerate() {
            base[c][t] = at;
            at += hh[t];
        }
        ranges[t][1] = at;
    }
    let mut ids = vec![0u32; total];
    let mut csr_slots = vec![0u32; total];
    let (pi, ps) = (Ptr(ids.as_mut_ptr()), Ptr(csr_slots.as_mut_ptr()));
    chunks.par_iter().zip(base.into_par_iter()).for_each(|(c, mut b)| {
        for &(_, i) in c.iter() {
            let r = rects[i as usize].unwrap();
            let mut k = csr_off[i as usize] as usize;
            for y in r[2]..r[3] {
                for x in r[0]..r[1] {
                    let t = y * tx + x;
                    let slot = b[t];
                    b[t] += 1;
                    // SAFETY: slots are disjoint across chunks by the prefix,
                    // and each splat's csr range is its own.
                    unsafe {
                        pi.set(slot as usize, i);
                        ps.set(k, slot);
                    }
                    k += 1;
                }
            }
        }
    });
    Bins { tiles_x: tx, tiles_y: ty, ranges, ids, csr_off, csr_slots }
}
