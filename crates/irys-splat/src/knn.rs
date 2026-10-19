//! Mean distance to the k nearest neighbours, by a uniform grid (the initial
//! splat size, as 3DGS's `distCUDA2`).

use std::collections::HashMap;

use rayon::prelude::*;

pub fn mean_nn_dist(p: &[[f32; 3]], k: usize) -> Vec<f32> {
    let n = p.len();
    if n <= 1 {
        return vec![0.01; n];
    }
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for q in p {
        for a in 0..3 {
            lo[a] = lo[a].min(q[a]);
            hi[a] = hi[a].max(q[a]);
        }
    }
    let vol: f32 = (0..3).map(|a| (hi[a] - lo[a]).max(1e-3)).product();
    let cell = (vol / n as f32).cbrt() * 2.0;
    let key = |q: &[f32; 3]| -> [i32; 3] { std::array::from_fn(|a| ((q[a] - lo[a]) / cell).floor() as i32) };
    let mut grid: HashMap<[i32; 3], Vec<u32>> = HashMap::new();
    for (i, q) in p.iter().enumerate() {
        grid.entry(key(q)).or_default().push(i as u32);
    }
    p.par_iter()
        .enumerate()
        .map(|(i, q)| {
            let c = key(q);
            let mut best: Vec<f32> = Vec::with_capacity(k + 1);
            let mut r = 1i32;
            loop {
                best.clear();
                for dx in -r..=r {
                    for dy in -r..=r {
                        for dz in -r..=r {
                            if let Some(v) = grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) {
                                for &j in v {
                                    if j as usize == i {
                                        continue;
                                    }
                                    let o = p[j as usize];
                                    best.push((0..3).map(|a| (o[a] - q[a]).powi(2)).sum::<f32>());
                                }
                            }
                        }
                    }
                }
                // the k nearest are certain once they lie within r cells
                best.sort_by(f32::total_cmp);
                let reach = (r as f32 * cell).powi(2);
                if (best.len() >= k && best[k - 1] <= reach) || r >= 8 {
                    break;
                }
                r += 1;
            }
            let m = best.len().min(k).max(1);
            best.iter().take(m).map(|d| d.sqrt()).sum::<f32>() / m as f32
        })
        .collect()
}
