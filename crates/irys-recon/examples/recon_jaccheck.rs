//! v1: the photo model's Jacobian against central differences of its own
//! (deterministic) image, column by column.
//!
//! `cargo run --release -p irys-recon --example recon_jaccheck -- [--fd] [--no-secondary]`

use irys_core::Vec3;
use irys_recon::dsl::{Axis, Corner, Feature, Program};
use irys_recon::look::Look;
use irys_recon::photo::{self, Camera, Rig, View};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let fd = args.iter().any(|a| a == "--fd");
    let sec = !args.iter().any(|a| a == "--no-secondary");
    let eye = Vec3::new(150.0, -190.0, 170.0);
    let at = Vec3::new(5.0, 0.0, 5.0);
    let f = (at - eye).normalize();
    let r = f.cross(&Vec3::new(0.0, 0.0, 1.0)).normalize();
    let d = f.cross(&r);
    let size = (200, 150);
    let cam = Camera { f: 330.0, cx: 100.0, cy: 75.0, r: [[r.x, r.y, r.z], [d.x, d.y, d.z], [f.x, f.y, f.z]], eye: [eye.x, eye.y, eye.z], size };
    let look = Look { key_dir: [-0.5, 0.3, 0.8], key: 0.9, ..Look::default() };
    let rig = Rig { views: vec![View { cam, look }], ss: 3, fd_occluded: fd, secondary: sec, shadow_samples: 20_000 };
    let prog = Program {
        w: 60.0,
        d: 44.0,
        h: 10.0,
        corner: Corner::Fillet(5.0),
        features: vec![
            Feature::Hole { x: -18.0, y: 9.0, r: 4.0 },
            Feature::Counterbore { x: 12.0, y: 8.0, r: 3.0, cr: 6.0, cd: 4.0 },
            Feature::Slot { x: -12.0, y: -11.0, len: 10.0, r: 2.5, axis: Axis::X },
            Feature::Boss { x: 14.0, y: -10.0, r: 5.0, bh: 6.0 },
        ],
        pose: Some([10.0, -5.0, 20.0]),
    };
    let t0 = std::time::Instant::now();
    let e = photo::eval(&prog, &rig, 60_000, 7, true).unwrap();
    println!("eval with jacobian {:.2}s", t0.elapsed().as_secs_f64());
    let t1 = std::time::Instant::now();
    let _ = photo::eval(&prog, &rig, 0, 7, false).unwrap();
    println!("eval image {:.2}s", t1.elapsed().as_secs_f64());
    let names: Vec<String> = prog.param_info().iter().map(|i| i.name.clone()).collect();
    let np = names.len();
    let j = &e.jac.as_ref().unwrap()[0];
    let th = prog.params();
    for k in 0..np {
        let h = if names[k].ends_with("yaw") { 0.2 } else { 0.1 };
        let mut a = th.clone();
        a[k] += h;
        let mut b = th.clone();
        b[k] -= h;
        let (mut pa, mut pb) = (prog.clone(), prog.clone());
        pa.set_params(&a);
        pb.set_params(&b);
        let (Ok(ea), Ok(eb)) = (photo::eval(&pa, &rig, 0, 7, false), photo::eval(&pb, &rig, 0, 7, false)) else {
            println!("{:12} build failed", names[k]);
            continue;
        };
        let cd: Vec<f64> = ea.img[0].iter().zip(&eb.img[0]).map(|(x, y)| (x - y) / (2.0 * h)).collect();
        let jc: Vec<f64> = (0..cd.len()).map(|p| j.img[p * np + k]).collect();
        // Compare smoothed (a 2-px box): edge samples are per pixel noisy.
        let sm = |v: &[f64]| irys_recon::fit::blur(v, size, 1, 1.5);
        let (cd, jc) = (sm(&cd), sm(&jc));
        let dot: f64 = cd.iter().zip(&jc).map(|(a, b)| a * b).sum();
        let (nc, nj) = (cd.iter().map(|x| x * x).sum::<f64>().sqrt(), jc.iter().map(|x| x * x).sum::<f64>().sqrt());
        println!("{:12} ratio {:6.3}  cosine {:6.3}  |cd| {:.3e}", names[k], dot / (nc * nc).max(1e-30), dot / (nc * nj).max(1e-30), nc);
    }
}
