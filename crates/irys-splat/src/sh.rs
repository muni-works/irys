//! Real spherical harmonics up to degree 3, the 3DGS convention (the same
//! constants and signs as `splat.wgsl`'s `splat_colour`), with the basis's
//! derivative in the direction for the colour's view-direction gradient.

/// Coefficients per channel at degree 3.
pub const COEFS: usize = 16;

/// `(degree + 1)²`.
pub fn count(degree: u32) -> usize {
    ((degree + 1) * (degree + 1)) as usize
}

const C0: f64 = 0.28209479177387814;
const C1: f64 = 0.4886025119029199;
const C2: [f64; 5] = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396];
const C3: [f64; 7] = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
];

/// The basis at unit direction `d` (entries past `count(degree)` are 0) and
/// its Jacobian `∂b_k/∂d` (of the polynomial, not through normalisation).
pub fn basis(d: [f64; 3], degree: u32) -> ([f64; COEFS], [[f64; 3]; COEFS]) {
    let [x, y, z] = d;
    let mut b = [0.0; COEFS];
    let mut g = [[0.0; 3]; COEFS];
    b[0] = C0;
    if degree >= 1 {
        b[1] = -C1 * y;
        g[1] = [0.0, -C1, 0.0];
        b[2] = C1 * z;
        g[2] = [0.0, 0.0, C1];
        b[3] = -C1 * x;
        g[3] = [-C1, 0.0, 0.0];
    }
    if degree >= 2 {
        let (xx, yy, zz) = (x * x, y * y, z * z);
        b[4] = C2[0] * x * y;
        g[4] = [C2[0] * y, C2[0] * x, 0.0];
        b[5] = C2[1] * y * z;
        g[5] = [0.0, C2[1] * z, C2[1] * y];
        b[6] = C2[2] * (2.0 * zz - xx - yy);
        g[6] = [-2.0 * C2[2] * x, -2.0 * C2[2] * y, 4.0 * C2[2] * z];
        b[7] = C2[3] * x * z;
        g[7] = [C2[3] * z, 0.0, C2[3] * x];
        b[8] = C2[4] * (xx - yy);
        g[8] = [2.0 * C2[4] * x, -2.0 * C2[4] * y, 0.0];
        if degree >= 3 {
            b[9] = C3[0] * y * (3.0 * xx - yy);
            g[9] = [C3[0] * 6.0 * x * y, C3[0] * (3.0 * xx - 3.0 * yy), 0.0];
            b[10] = C3[1] * x * y * z;
            g[10] = [C3[1] * y * z, C3[1] * x * z, C3[1] * x * y];
            b[11] = C3[2] * y * (4.0 * zz - xx - yy);
            g[11] = [C3[2] * -2.0 * x * y, C3[2] * (4.0 * zz - xx - 3.0 * yy), C3[2] * 8.0 * y * z];
            b[12] = C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
            g[12] = [C3[3] * -6.0 * x * z, C3[3] * -6.0 * y * z, C3[3] * (6.0 * zz - 3.0 * xx - 3.0 * yy)];
            b[13] = C3[4] * x * (4.0 * zz - xx - yy);
            g[13] = [C3[4] * (4.0 * zz - 3.0 * xx - yy), C3[4] * -2.0 * x * y, C3[4] * 8.0 * x * z];
            b[14] = C3[5] * z * (xx - yy);
            g[14] = [C3[5] * 2.0 * x * z, C3[5] * -2.0 * y * z, C3[5] * (xx - yy)];
            b[15] = C3[6] * x * (xx - 3.0 * yy);
            g[15] = [C3[6] * (3.0 * xx - 3.0 * yy), C3[6] * -6.0 * x * y, 0.0];
        }
    }
    (b, g)
}

/// The DC coefficient for a target sRGB colour: `c = ½ + C0 · f_dc`.
pub fn dc_of(rgb: f64) -> f64 {
    (rgb - 0.5) / C0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobian_matches_differences() {
        let d = [0.3, -0.5, 0.81];
        let (_, g) = basis(d, 3);
        let h = 1e-6;
        for a in 0..3 {
            let mut p = d;
            let mut m = d;
            p[a] += h;
            m[a] -= h;
            let (bp, _) = basis(p, 3);
            let (bm, _) = basis(m, 3);
            for k in 0..COEFS {
                let cd = (bp[k] - bm[k]) / (2.0 * h);
                assert!((cd - g[k][a]).abs() < 1e-7, "k {k} axis {a}: {cd} vs {}", g[k][a]);
            }
        }
    }

    #[test]
    fn matches_irys_colour() {
        // The same polynomial as splat_gpu::colour_cpu (and splat.wgsl).
        use irys_light::PlyCloud;
        let mut sh = vec![[0.0f32; 3]; 16];
        for (k, s) in sh.iter_mut().enumerate() {
            *s = [0.01 * k as f32, -0.02 * k as f32 + 0.1, 0.03];
        }
        let c = PlyCloud { positions: vec![[0.0; 3]], scales: vec![[0.1; 3]], quats: vec![[1.0, 0.0, 0.0, 0.0]], opacities: vec![0.5], sh: sh.clone(), degree: 3 };
        let d = [0.48f64, -0.6, 0.64];
        let want = irys_diff::look::splat_gpu::colour_cpu(&c, 0, d);
        let (b, _) = basis(d, 3);
        for ch in 0..3 {
            let half = |v: f32| irys_diff::look::splat_gpu::f16_value(irys_diff::look::splat_gpu::f16_bits(v)) as f64;
            let got = 0.5 + (0..16).map(|k| b[k] * half(sh[k][ch])).sum::<f64>();
            assert!((got.max(0.0) - want[ch]).abs() < 1e-9, "{ch}: {got} vs {}", want[ch]);
        }
    }
}
