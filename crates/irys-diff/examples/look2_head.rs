//! Phase 4b: the Lee Perry-Smith head with strands — eyebrows, stubble and
//! a short scalp cut (`skin::groom`) — rendered with the SSS skin in one
//! frame on the GPU (`GpuSkin::with_strands`).
//!
//! ```text
//! cargo run --release -p irys-diff --features gpu,assets --example look2_head -- docs [size] [spp] [cyl|tubeN] [bare]
//! ```
//!
//! writes `look2-head-hair-front.png` and `look2-head-hair-3q.png` and
//! prints timings (`bare`: also each frame without the hair). The head is
//! "Lee Perry-Smith" by Infinite-Realities (ir-ltd.net), CC BY 3.0
//! (creativecommons.org/licenses/by/3.0), via the three.js examples; it is
//! read from `$IRYS_ASSETS/lee-perry-smith` and never copied into the
//! repository.

use std::path::Path;
use std::time::Instant;

use irys_diff::look::hair::Fiber;
use irys_diff::look::hair_gpu::{GpuStrands, StrandPrim};
use irys_diff::skin::gpu::{GpuSkin, SkinHair};
use irys_diff::skin::groom::{GroomSpec, groom};
use irys_diff::skin::render::{Chromophores, Mode, THETA, tonemap};
use irys_diff::skin::{head, mesh, optics};

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next().unwrap_or_else(|| "docs".into());
    let size: usize = args.next().map_or(1024, |s| s.parse().expect("size"));
    let spp: usize = args.next().map_or(256, |s| s.parse().expect("spp"));
    let prim = match args.next().as_deref() {
        None | Some("cyl") => StrandPrim::Cylinders,
        Some(t) => StrandPrim::Tubes(t.trim_start_matches("tube").parse().expect("tubeN")),
    };
    let bare = args.next().is_some_and(|a| a == "bare");
    let out = Path::new(&out);
    std::fs::create_dir_all(out).expect("out dir");
    let ctx = irys_diff::gpu::context().expect("a GPU adapter");

    let asset = mesh::load(&mesh::asset_dir()).expect("the Lee Perry-Smith asset");
    let (chromo, _) = optics::invert(&asset.albedo);
    let s0 = head::scene(&asset, Chromophores::Map(chromo.clone()), head::View::Front);
    let t = Instant::now();
    let g = groom(&s0.mesh, &asset.albedo, &GroomSpec::default());
    println!(
        "groom: {} scalp, {} brow, {} stubble strands, {} segments, {:.2} s",
        g.counts[0],
        g.counts[1],
        g.counts[2],
        g.strands.segs.len(),
        t.elapsed().as_secs_f64()
    );
    let t = Instant::now();
    let strands = GpuStrands::new(ctx, &g.strands, Some(&g.melanin), prim).expect("ray queries");
    println!("{prim:?}: {} primitives, built in {:.2} s", strands.primitives, t.elapsed().as_secs_f64());
    // Dark brown-black (Chiang: eumelanin 8 is black, 1.3 brown).
    let hair = || SkinHair {
        strands: &strands,
        fiber: Fiber::default(),
        melanin: [5.0, 0.4],
    };
    let mut exposure = 0.0;
    for (name, view) in [
        ("look2-head-hair-front", head::View::Front),
        ("look2-head-hair-3q", head::View::ThreeQuarter),
    ] {
        let mut s = head::scene(&asset, Chromophores::Map(chromo.clone()), view);
        s.set_size(size, size);
        s.spp = spp;
        s.mode = Mode { sss: true, dual: true };
        if bare {
            let gb = GpuSkin::new(ctx, &s, false);
            let t = Instant::now();
            let _ = gb.render(&s, &THETA, 1, false);
            let secs = t.elapsed().as_secs_f64();
            println!("{name} without hair: {secs:.1} s ({:.2} Mpaths/s)", s.paths() as f64 / secs / 1e6);
        }
        let gs = GpuSkin::with_strands(ctx, &s, false, Some(hair()));
        let t = Instant::now();
        let (img, _) = gs.render(&s, &THETA, 1, false);
        let secs = t.elapsed().as_secs_f64();
        let img: Vec<f64> = img.iter().map(|&x| x as f64).collect();
        if exposure == 0.0 {
            exposure = auto_exposure(&img);
        }
        println!(
            "{name}: {size}² × {spp} spp, depth {}, {secs:.1} s ({:.2} Mpaths/s)",
            s.depth,
            s.paths() as f64 / secs / 1e6
        );
        let path = out.join(format!("{name}.png"));
        image::save_buffer(&path, &tonemap(&img, exposure), size as u32, size as u32, image::ExtendedColorType::Rgb8)
            .expect("png");
        println!("wrote {}", path.display());
    }
}

/// `examples/skin.rs`'s exposure: the lit pixels' median luminance at 0.24.
fn auto_exposure(img: &[f64]) -> f64 {
    let mut lum: Vec<f64> = img
        .chunks_exact(3)
        .map(|c| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])
        .collect();
    lum.sort_by(f64::total_cmp);
    let top = lum[lum.len() * 999 / 1000];
    let lit: Vec<f64> = lum.into_iter().filter(|&l| l > 0.05 * top).collect();
    0.24 / lit[lit.len() / 2]
}
