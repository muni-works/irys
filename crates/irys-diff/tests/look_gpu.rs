//! Phase 4 on the device: the look's IR graphs as tang-compile emits them
//! in WGSL, against their f64 twins. Skips without a device.
#![cfg(feature = "gpu")]

use irys_diff::look::graph::{self, Stack};
use irys_diff::look::openpbr::{At, COLUMNS, Kind, OpenPbr, PARAMS};
use irys_diff::prb::Rng;

fn dir(rng: &mut Rng) -> [f64; 3] {
    let z = 0.05 + 0.95 * rng.uniform();
    let p = 2.0 * std::f64::consts::PI * rng.uniform();
    let s = (1.0 - z * z).sqrt();
    [s * p.cos(), s * p.sin(), z]
}

#[test]
fn stack_on_the_gpu_is_the_f64_twin() {
    let Ok(ctx) = irys_diff::gpu::context() else {
        eprintln!("no device: skipped");
        return;
    };
    let m = OpenPbr {
        base_color: [0.7, 0.45, 0.25],
        base_metalness: 0.35,
        coat_weight: 0.4,
        coat_roughness: 0.15,
        fuzz_weight: 0.3,
        thin_film_weight: 0.5,
        transmission_weight: 0.2,
        subsurface_weight: 0.2,
        ..Default::default()
    };
    let theta = m.theta();
    let n = 4096;
    let mut rng = Rng::new(2, 0);
    let rows: Vec<[f64; COLUMNS]> = (0..n)
        .map(|k| {
            let at = At::new(&m, 380.0 + 400.0 * rng.uniform(), false);
            let (wo, mut wi) = (dir(&mut rng), dir(&mut rng));
            let kind = match k % 3 {
                0 => Kind::Reflect,
                1 => {
                    wi[2] = -wi[2];
                    Kind::Transmit
                }
                _ => Kind::Subsurface,
            };
            at.columns(kind, wo, wi)
        })
        .collect();
    let df: Vec<f64> = (0..n).map(|_| rng.uniform() - 0.5).collect();
    let mut inputs: Vec<Vec<f64>> = vec![rows.iter().flatten().copied().collect()];
    inputs.push(theta.to_vec());
    inputs.push(df.clone());
    let g = Stack::new(n);
    let out = irys_diff::look::gpu::run(ctx, &g.g, &[g.f, g.dtheta], &inputs);
    let mut worst: f64 = 0.0;
    let mut over = 0;
    for (r, &fg) in rows.iter().zip(&out[0]) {
        let ft = graph::stack(r, &theta);
        let e = (fg as f64 - ft).abs() / ft.abs().max(1e-3);
        worst = worst.max(e);
        over += (e > 1e-4) as usize;
    }
    let mut ad = [0.0; PARAMS];
    let mut p = Stack::new(n).adjoint(false);
    let cpu = graph::reduce(&mut p, n, &rows, &df, &theta);
    for k in 0..PARAMS {
        ad[k] = (out[1][k] as f64 - cpu[k]).abs() / cpu[k].abs().max(1e-3);
    }
    println!("GPU f vs f64: worst rel {worst:.2e}, {over}/{n} over 1e-4; δθ rel {ad:?}");
    assert!(worst < 1e-3 && over < n / 100);
    assert!(ad.iter().all(|&e| e < 1e-3));
}

#[test]
fn volume_and_hair_graphs_on_the_gpu() {
    use irys_diff::look::{hair, volume};
    let Ok(ctx) = irys_diff::gpu::context() else {
        eprintln!("no device: skipped");
        return;
    };
    let n = 2048;
    let mut rng = Rng::new(5, 0);
    // Volume events.
    let rows: Vec<[f64; volume::COLUMNS]> = (0..n)
        .map(|_| {
            let real = (rng.uniform() < 0.4) as u8 as f64;
            let b = irys_diff::ggx::band_weights(380.0 + 400.0 * rng.uniform());
            [real, 1.0 - real, 0.2 + rng.uniform(), 0.3 + rng.uniform(), 6.0, b[0], b[1], b[2]]
        })
        .collect();
    let th = [1.2, 0.7, 0.8, 0.9];
    let mut inputs: Vec<Vec<f64>> = (0..volume::COLUMNS).map(|k| rows.iter().map(|r| r[k]).collect()).collect();
    inputs.push(th.to_vec());
    inputs.push(vec![0.0; n]);
    let g = volume::VolEvent::new(n);
    let out = irys_diff::look::gpu::run(ctx, &g.g, &[g.ell], &inputs);
    let wv = rows.iter().zip(&out[0]).map(|(r, &v)| (v as f64 - volume::ell(r, &th)).abs()).fold(0.0, f64::max);
    // Hair lanes.
    let bc = hair::Bcsdf::new(&hair::Fiber::default(), 0.4);
    let hr: Vec<[f64; hair::COLUMNS]> = (0..n)
        .map(|_| bc.columns(dir(&mut rng), dir(&mut rng), 380.0 + 400.0 * rng.uniform()))
        .collect();
    let ht = [1.3, 0.4];
    let mut inputs: Vec<Vec<f64>> = (0..hair::COLUMNS).map(|k| hr.iter().map(|r| r[k]).collect()).collect();
    inputs.push(ht.to_vec());
    inputs.push(vec![0.0; n]);
    let g = hair::Hair::new(n);
    let out = irys_diff::look::gpu::run(ctx, &g.g, &[g.f], &inputs);
    let wh = hr
        .iter()
        .zip(&out[0])
        .map(|(r, &v)| (v as f64 - hair::hair_f(r, &ht)).abs() / hair::hair_f(r, &ht).abs().max(1e-3))
        .fold(0.0, f64::max);
    println!("GPU volume ℓ worst abs {wv:.2e}; hair f worst rel {wh:.2e}");
    assert!(wv < 1e-5 && wh < 1e-4);
}
