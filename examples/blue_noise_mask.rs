//! Bakes the sampler's blue-noise mask: `src/blue_noise_128.bin`.
//!
//! A 128×128 tile of 2D *vectors*, not scalars. The sampler adds a texel's
//! vector to a dimension pair's Sobol point (`sampler.rs`), so what has to be
//! blue is the vector field: two neighbouring texels must hold vectors that
//! differ, in the toroidal value space, in proportion to how close they are.
//! A scalar blue-noise mask read at two offsets gives two blue components
//! whose *product* is white — the spectrum of a product is the convolution of
//! the spectra — and a frame under it measured no bluer than white noise.
//!
//! Georgiev & Fajardo 2016 (*Blue-noise Dithered Sampling*): the values are
//! a fixed stratified set — here the first 16384 points of the (0, 2)-
//! sequence the sampler itself uses, so the tile's shifts cover the unit
//! square uniformly — assigned to texels by simulated annealing over swaps,
//! against the energy
//!
//! ```text
//! E = Σ_{p≠q} exp(−|p − q|² / σ_i²) · D(v_p, v_q)
//! ```
//!
//! with σ_i = 2.1 and `D` a value-space dissimilarity, by Metropolis over
//! swaps under a falling temperature. Georgiev and Fajardo's
//! `D` is exp(−‖v_p − v_q‖ / σ_s²), and a mask annealed against it alone
//! measured blue for a smooth integrand and *whiter than white* for a step —
//! which is the finding behind Heitz et al. 2019: a step's error at a pixel
//! is which side of the step the pixel's shift lands on, and two shifts can
//! be far apart in value and still land on the same side. So `D` here is
//! half that term and half the fraction of a fixed family of step
//! integrands — square waves along lattice directions and discs, all
//! periodic on the value torus — on which the two shifts agree, weighted
//! 3:7 towards the steps.
//! Every value is stored as two bytes, `u` then `v`, row-major.
//!
//! Deterministic: a fixed seed, so the file is reproducible. Run with
//! `cargo run -p kosm-render --release --example blue_noise_mask`.

use kosm_render::sampler::{nested_uniform_scramble, sobol_dim0, sobol_dim1};

const N: usize = 128;
const SIGMA_I: f64 = 2.1;
const SIGMA_S: f64 = 1.0;
/// The neighbourhood a swap is scored over; the kernel is 2e-4 at its edge.
const R: i32 = 6;
const SWAPS: usize = 12_000_000;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() as usize) % n
    }
}

fn torus(a: f64, b: f64) -> f64 {
    let d = (a - b).abs();
    d.min(1.0 - d)
}

/// The step family: which side of each of 64 periodic steps a value is on,
/// one bit each.
fn step_bits(v: (f64, f64)) -> u64 {
    let mut bits = 0u64;
    let mut k = 0;
    // Square waves: 8 lattice directions, 6 phases each.
    for (dx, dy) in [
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (1.0, -1.0),
        (2.0, 1.0),
        (1.0, 2.0),
        (2.0, -1.0),
        (-1.0, 2.0),
    ] {
        for phase in 0..6 {
            let t = v.0 * dx + v.1 * dy + phase as f64 / 6.0;
            if t - t.floor() < 0.5 {
                bits |= 1 << k;
            }
            k += 1;
        }
    }
    // Discs: 16 centres and radii, fixed.
    for (cx, cy, r) in [
        (0.1, 0.2, 0.25),
        (0.6, 0.7, 0.35),
        (0.3, 0.8, 0.45),
        (0.9, 0.4, 0.3),
        (0.5, 0.1, 0.5),
        (0.2, 0.5, 0.4),
        (0.7, 0.3, 0.2),
        (0.8, 0.9, 0.55),
        (0.05, 0.6, 0.3),
        (0.45, 0.45, 0.15),
        (0.35, 0.15, 0.4),
        (0.95, 0.85, 0.45),
        (0.65, 0.05, 0.25),
        (0.15, 0.95, 0.5),
        (0.55, 0.55, 0.6),
        (0.85, 0.25, 0.35),
    ] {
        let d = (torus(v.0, cx).powi(2) + torus(v.1, cy).powi(2)).sqrt();
        if d < r {
            bits |= 1 << k;
        }
        k += 1;
    }
    bits
}

/// The energy between texel `i` and its neighbourhood, given its value.
fn local(values: &[(f64, f64, u64)], kernel: &[f64], i: usize, v: (f64, f64, u64)) -> f64 {
    let (px, py) = ((i % N) as i32, (i / N) as i32);
    let mut e = 0.0;
    for dy in -R..=R {
        for dx in -R..=R {
            if dx == 0 && dy == 0 {
                continue;
            }
            let q = ((py + dy).rem_euclid(N as i32) as usize) * N
                + (px + dx).rem_euclid(N as i32) as usize;
            let w = values[q];
            let dv = (torus(v.0, w.0).powi(2) + torus(v.1, w.1).powi(2)).sqrt();
            let agree = (!(v.2 ^ w.2)).count_ones() as f64 / 64.0;
            let k = kernel[((dy + R) * (2 * R + 1) + dx + R) as usize];
            e += k * (0.3 * (-dv / (SIGMA_S * SIGMA_S)).exp() + 0.7 * agree);
        }
    }
    e
}

fn main() {
    let total = N * N;
    let kernel: Vec<f64> = (-R..=R)
        .flat_map(|dy| {
            (-R..=R).map(move |dx| (-((dx * dx + dy * dy) as f64) / (SIGMA_I * SIGMA_I)).exp())
        })
        .collect();

    // The values: a stratified set, shuffled onto the texels.
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
    let mut values: Vec<(f64, f64, u64)> = (0..total as u32)
        .map(|i| {
            let idx = nested_uniform_scramble(i, 0xB1_E5);
            let v = (
                (sobol_dim0(idx) >> 8) as f64 / 16777216.0,
                (sobol_dim1(idx) >> 8) as f64 / 16777216.0,
            );
            (v.0, v.1, step_bits(v))
        })
        .collect();
    for i in (1..total).rev() {
        let j = rng.below(i + 1);
        values.swap(i, j);
    }

    // Metropolis: a swap that raises the energy is kept with probability
    // exp(-ΔE / T), T falling geometrically from T0 to T1 over the run, so
    // the field can leave the local minimum a greedy walk stops in.
    let (t0, t1) = (0.2f64, 0.002f64);
    let mut kept = 0usize;
    for step in 0..SWAPS {
        let a = rng.below(total);
        let b = rng.below(total);
        if a == b {
            continue;
        }
        let (va, vb) = (values[a], values[b]);
        let before = local(&values, &kernel, a, va) + local(&values, &kernel, b, vb);
        values[a] = vb;
        values[b] = va;
        let after = local(&values, &kernel, a, vb) + local(&values, &kernel, b, va);
        let t = t0 * (t1 / t0).powf(step as f64 / SWAPS as f64);
        let u = rng.next() as f64 / (1u64 << 31) as f64;
        if after < before || u < ((before - after) / t).exp() {
            kept += 1;
        } else {
            values[a] = va;
            values[b] = vb;
        }
        if step % 1_000_000 == 0 {
            eprintln!("swap {step}: {kept} kept, T = {t:.4}");
        }
    }

    let mut bytes = Vec::with_capacity(total * 2);
    for &(u, v, _) in &values {
        bytes.push((u * 256.0) as u8);
        bytes.push((v * 256.0) as u8);
    }
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/blue_noise_128.bin");
    std::fs::write(path, &bytes).unwrap();
    println!(
        "wrote {path}: {} texels, {kept} of {SWAPS} swaps kept",
        total
    );
}
