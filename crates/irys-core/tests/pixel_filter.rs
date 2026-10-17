//! The reconstruction filter: what it puts where, and that both tiers agree.
//!
//! Primary rays used to be jittered uniformly in the pixel — a box filter,
//! which is why a thin bright feature against a dark background stairsteps
//! however many samples it gets. The filter is importance-sampled instead of
//! weighted, so the accumulation rule never changes; what these tests hold is
//! that the sampling really does reproduce the kernel, that a converged edge
//! comes out as the kernel's own footprint, and that the device aims at the
//! same point the CPU does.

use kosm_render::pathtrace::{GAUSSIAN_SIGMA, PixelFilter};

const FILTERS: [PixelFilter; 3] = [
    PixelFilter::Box,
    PixelFilter::Gaussian,
    PixelFilter::BlackmanHarris,
];

/// The kernel's own normalised CDF, integrated numerically — deliberately by
/// a different route than the closed form `warp` inverts, so agreement means
/// something.
fn analytic_cdf(f: PixelFilter, x: f64) -> f64 {
    let r = f.radius();
    let n = 200_000;
    let step = 2.0 * r / n as f64;
    let (mut below, mut total) = (0.0, 0.0);
    for i in 0..n {
        let t = -r + (i as f64 + 0.5) * step;
        let w = f.weight(t) * step;
        total += w;
        if t <= x {
            below += w;
        }
    }
    below / total
}

/// `warp` must be the inverse CDF of `weight`: the fraction of warped samples
/// landing left of `x` has to be the kernel's own mass left of `x`.
#[test]
fn the_warp_reproduces_the_kernel() {
    for f in FILTERS {
        for k in 1..20 {
            let u = k as f64 / 20.0;
            let x = f.warp(u);
            let got = analytic_cdf(f, x);
            assert!(
                (got - u).abs() < 2e-3,
                "{f:?}: warp({u}) = {x}, whose CDF is {got}",
            );
        }
        // The support is respected exactly at both ends.
        assert!(f.warp(0.0) >= -f.radius() - 1e-9);
        assert!(f.warp(1.0) <= f.radius() + 1e-9);
    }
}

/// The box filter's warp is `u - 0.5` exactly, which is the sample position
/// the integrator used before there was a filter at all. Anything else here
/// would silently change every existing render.
#[test]
fn box_is_the_old_jitter_to_the_bit() {
    for k in 0..1000 {
        let u = k as f64 / 1000.0;
        assert_eq!(PixelFilter::Box.warp(u), u - 0.5);
    }
}

/// A converged high-contrast edge must come out as the filter's own
/// footprint.
///
/// A step edge at `e` pixels from the pixel centre, sampled through the
/// filter, converges to the kernel's mass to the right of `e`. That is the
/// filter's analytic footprint, and it is what a rendered silhouette running
/// across a pixel actually looks like — so it is checked directly, over the
/// same `warp` the integrator calls.
#[test]
fn the_converged_edge_profile_matches_the_analytic_footprint() {
    for f in FILTERS {
        for step in -6..=6 {
            let e = step as f64 * 0.25;
            let n = 200_000;
            // Stratified uniforms, as spp samples of a pixel effectively are.
            let mut lit = 0.0f64;
            for i in 0..n {
                let u = (i as f64 + 0.5) / n as f64;
                if f.warp(u) > e {
                    lit += 1.0;
                }
            }
            let got = lit / n as f64;
            let want = 1.0 - analytic_cdf(f, e);
            assert!(
                (got - want).abs() < 3e-3,
                "{f:?}: an edge at {e} px converged to {got}, footprint says {want}",
            );
        }
    }
}

/// The two wide filters must actually be wider than the box: an edge they
/// reconstruct spans more than one pixel, which is the whole point.
#[test]
fn the_wide_filters_span_more_than_a_pixel() {
    let width = |f: PixelFilter| {
        // 10-90% width of the edge footprint, in pixels.
        let at = |q: f64| {
            let mut lo = -f.radius();
            let mut hi = f.radius();
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                if 1.0 - analytic_cdf(f, mid) > q {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            0.5 * (lo + hi)
        };
        at(0.1) - at(0.9)
    };
    let b = width(PixelFilter::Box);
    let g = width(PixelFilter::Gaussian);
    let bh = width(PixelFilter::BlackmanHarris);
    assert!(b > 0.75 && b < 0.85, "box 10-90 width {b}");
    assert!(g > b, "the Gaussian ({g}) is no wider than the box ({b})");
    assert!(
        bh > b,
        "Blackman-Harris ({bh}) is no wider than the box ({b})"
    );
    // A Gaussian of sigma 0.4 has a 10-90 edge width of 2*0.4*1.2816 ~ 1.025.
    let expect = 2.0 * GAUSSIAN_SIGMA * 1.281_551_6;
    assert!(
        (g - expect).abs() < 0.02,
        "Gaussian 10-90 width {g}, analytic {expect}",
    );
    // Over the same 1.5-pixel support the two land within a tenth of a pixel
    // of each other — Blackman-Harris buys its lower sidelobes rather than a
    // narrower main lobe, so it should not be much sharper *or* much softer.
    assert!(
        (bh - g).abs() < 0.1,
        "Blackman-Harris ({bh}) and the Gaussian ({g}) should have similar edge widths",
    );
}

/// The device aims where the CPU aims.
///
/// The GPU's sub-pixel jitter is a Halton pair computed on the host, so the
/// filter is applied there — by the same `warp` — and the shader never learns
/// what a reconstruction filter is. This checks the two really are one
/// function and not two that happen to look alike.
#[cfg(feature = "gpu")]
#[test]
fn the_device_jitter_is_the_cpu_warp() {
    use kosm_render::gpu::GpuRenderState;

    for f in FILTERS {
        for frame in 1..64u32 {
            let plain = GpuRenderState::new(frame);
            let mut filtered = GpuRenderState::new(frame);
            filtered.set_pixel_filter(f);

            // The uniforms the plain state's centred jitter came from.
            let (u, v) = (plain.jitter_x as f64 + 0.5, plain.jitter_y as f64 + 0.5);
            let (wx, wy) = (f.warp(u) as f32, f.warp(v) as f32);
            assert!(
                (filtered.jitter_x - wx).abs() <= f32::EPSILON * 4.0,
                "{f:?} frame {frame}: device x {} vs warp {wx}",
                filtered.jitter_x,
            );
            assert!(
                (filtered.jitter_y - wy).abs() <= f32::EPSILON * 4.0,
                "{f:?} frame {frame}: device y {} vs warp {wy}",
                filtered.jitter_y,
            );
            if f == PixelFilter::Box {
                // The compatibility default: bit-identical to no filter.
                assert_eq!(filtered.jitter_x, plain.jitter_x);
                assert_eq!(filtered.jitter_y, plain.jitter_y);
            }
            assert!(filtered.jitter_x.abs() <= f.radius() as f32 + 1e-6);
        }
    }
}
