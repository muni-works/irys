//! Checks the graph: native C against the interpreter, `backward` against
//! central differences (interpreter, f64), and times one full-size step.

use irys_recon::jac::Mix;
use irys_recon_net::data::Targets;
use irys_recon_net::net::{Arch, DIN, P, PIX};
use irys_recon_net::scene::{CLASSES, NP, NTOK, VOCAB};
use irys_recon_net::train::{Batch, Trainer};

fn rnd(r: &mut Mix, n: usize, lo: f64, hi: f64) -> Vec<f64> {
    (0..n).map(|_| lo + (hi - lo) * r.uniform()).collect()
}

fn batch(r: &mut Mix, b: usize) -> Batch {
    let s = 1.0 / (b * P) as f64;
    let t = Targets {
        t_pix: rnd(r, b * P * 4 * PIX, 0.0, 1.0),
        w_pix: rnd(r, b * P * 4 * PIX, 0.0, s / 64.0),
        t_ctr: rnd(r, b * P, 0.0, 1.0),
        w_ctr: rnd(r, b * P, 0.0, s),
        t_cls: rnd(r, b * P * CLASSES, 0.0, 0.2),
        w_cls: rnd(r, b * P, 0.0, s),
        t_tok: rnd(r, b * P * NTOK * VOCAB, 0.0, 0.2),
        w_tok: rnd(r, b * P * NTOK, 0.0, s),
        t_par: rnd(r, b * P * NP, -1.0, 1.0),
        w_par: rnd(r, b * P * NP, 0.0, s),
        t_yaw: rnd(r, b * P * 2, -1.0, 1.0),
        w_yaw: rnd(r, b * P * 2, 0.0, s),
        t_plane: rnd(r, b * 2, -1.0, 1.0),
        w_plane: rnd(r, b * 2, 0.0, 1.0),
    };
    let mut bt = Batch::new(&[rnd(r, b * P * DIN, -1.0, 1.0)], &[t]);
    bt.x = rnd(r, b * P * DIN, -1.0, 1.0);
    bt.g_par = rnd(r, b * P * 2 * NP, -s, s);
    bt.g_yaw = rnd(r, b * P * 2, -s, s);
    bt.g_plane = rnd(r, b * 2, -0.1, 0.1);
    bt
}

fn main() {
    let tiny = Arch { d: 8, tok_hidden: 8, ch_hidden: 8, blocks: 1 };
    let mut r = Mix(7);
    let b = batch(&mut r, 1);
    let mut nat = Trainer::new(tiny, 1, 1, None, 1);
    assert!(nat.native, "no C compiler");
    unsafe { std::env::set_var("IRYS_IR", "interp") };
    let mut itp = Trainer::new(tiny, 1, 1, Some(nat.params.clone()), 1);
    unsafe { std::env::remove_var("IRYS_IR") };
    let (ln, gn) = nat.grads(std::slice::from_ref(&b));
    let (li, gi) = itp.grads(std::slice::from_ref(&b));
    println!("loss native {:.6} interp {:.6}", ln[0], li[0]);
    let mut worst: f64 = 0.0;
    for (a, c) in gn.iter().flatten().zip(gi.iter().flatten()) {
        worst = worst.max((a - c).abs() / (c.abs() + 1e-4));
    }
    println!("grad native vs interp: worst rel {worst:.2e}");
    // Central differences on a few entries of every tensor.
    let mut wfd: f64 = 0.0;
    for k in 0..itp.params.len() {
        for j in [0usize, itp.params[k].len() / 2] {
            let h = 1e-5;
            let base = itp.params[k][j];
            itp.params[k][j] = base + h;
            let (lp, _) = itp.grads(std::slice::from_ref(&b));
            itp.params[k][j] = base - h;
            let (lm, _) = itp.grads(std::slice::from_ref(&b));
            itp.params[k][j] = base;
            let fd = (lp[0] - lm[0]) / (2.0 * h);
            let e = (fd - gi[k][j]).abs() / (fd.abs().max(gi[k][j].abs()) + 1e-6);
            wfd = wfd.max(e);
        }
    }
    println!("backward vs central differences: worst rel {wfd:.2e}");

    // Full size timing.
    let arch = Arch::default();
    let per: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(2);
    let workers: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let t0 = std::time::Instant::now();
    let mut tr = Trainer::new(arch, per, workers, None, 1);
    println!("compiled {} workers × {per} images in {:.1}s, {} parameters, native {}", workers, t0.elapsed().as_secs_f64(), tr.parameter_count(), tr.native);
    let bs: Vec<Batch> = (0..workers).map(|_| batch(&mut r, per)).collect();
    for _ in 0..3 {
        let t0 = std::time::Instant::now();
        let (l, g) = tr.grads(&bs);
        let dt = t0.elapsed().as_secs_f64();
        tr.adam(&g, 1e-3, 1.0, 0.0);
        println!("step {:.2}s ({:.1} images/s) loss {:.4}", dt, (per * workers) as f64 / dt, l[0]);
    }
}
