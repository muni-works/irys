//! Photoreal skin: the Lee Perry-Smith head in two portraits and a
//! comparison strip, on the GPU.
//!
//! ```text
//! cargo run --release -p irys-diff --features gpu,assets --example skin -- docs
//! cargo run --release -p irys-diff --features gpu,assets --example skin -- out/look 384 64
//! ```
//!
//! writes `skin-head-front.png`, `skin-head-3q.png` and `skin-compare.png`
//! (no SSS, SSS with one specular lobe, SSS with two) into the directory,
//! at the given size (default 1024) and samples per pixel (default 256),
//! and prints timings.
//!
//! The head is "Lee Perry-Smith" by Infinite-Realities (ir-ltd.net),
//! licensed CC BY 3.0 (creativecommons.org/licenses/by/3.0), from the
//! three.js examples. It is read from `$IRYS_ASSETS/lee-perry-smith`
//! (default `/Users/cam/Developer/irys-assets`) and never copied into the
//! repository.

use std::path::Path;
use std::time::Instant;

use irys_diff::skin::gpu::GpuSkin;
use irys_diff::skin::render::{Chromophores, Mode, Scene, THETA, tonemap};
use irys_diff::skin::{head, mesh, optics};

#[path = "common/font.rs"]
mod font;
use font::text;

fn main() {
    let mut args = std::env::args().skip(1);
    let out = args.next().unwrap_or_else(|| "docs".into());
    let size: usize = args.next().map_or(1024, |s| s.parse().expect("size"));
    let spp: usize = args.next().map_or(256, |s| s.parse().expect("spp"));
    let only = args.next();
    let out = Path::new(&out);
    std::fs::create_dir_all(out).expect("out dir");
    let ctx = irys_diff::gpu::context().expect("a GPU adapter");

    let t = Instant::now();
    let asset = mesh::load(&mesh::asset_dir()).expect("the Lee Perry-Smith asset");
    let (chromo, err) = optics::invert(&asset.albedo);
    println!(
        "asset + chromophore inversion: {:.2} s (mean rgb error {err:.4})",
        t.elapsed().as_secs_f64()
    );

    let render = |s: &Scene| {
        let g = GpuSkin::new(ctx, s, false);
        let t = Instant::now();
        let (img, _) = g.render(s, &THETA, 1, false);
        let secs = t.elapsed().as_secs_f64();
        (img.iter().map(|&x| x as f64).collect::<Vec<f64>>(), secs)
    };
    let dual = Mode {
        sss: true,
        dual: true,
    };
    let want = |name: &str| only.as_deref().is_none_or(|o| name.contains(o));

    // The two portraits share the front's exposure.
    let mut exposure = 0.0;
    for (name, view) in [
        ("skin-head-front", head::View::Front),
        ("skin-head-3q", head::View::ThreeQuarter),
    ] {
        if !want(name) && exposure != 0.0 {
            continue;
        }
        let mut s = head::scene(&asset, Chromophores::Map(chromo.clone()), view);
        s.set_size(size, size);
        s.spp = spp;
        s.mode = dual;
        let (img, secs) = render(&s);
        if exposure == 0.0 {
            exposure = auto_exposure(&img);
        }
        println!(
            "{name}: {size}² × {spp} spp, depth {}, {:.1} s ({:.2} Mpaths/s), exposure {exposure:.3}",
            s.depth,
            secs,
            (size * size * spp) as f64 / secs / 1e6
        );
        png(
            &out.join(format!("{name}.png")),
            s.size,
            &tonemap(&img, exposure),
        );
    }
    if !want("skin-compare") {
        return;
    }

    // The strip: one framing, three models, the same exposure.
    let (pw, ph) = (size / 2, size * 5 / 8);
    let panels = [
        (
            "NO SSS: LAMBERT + 1 GGX",
            Mode {
                sss: false,
                dual: false,
            },
        ),
        (
            "SSS + 1 GGX",
            Mode {
                sss: true,
                dual: false,
            },
        ),
        ("SSS + DUAL GGX", dual),
    ];
    let mut strip = vec![0u8; 3 * pw * ph * 3];
    for (i, (label, mode)) in panels.iter().enumerate() {
        let mut s = head::scene(&asset, Chromophores::Map(chromo.clone()), head::View::Front);
        head::compare_frame(&mut s, (pw, ph));
        s.spp = spp;
        s.mode = *mode;
        let (img, secs) = render(&s);
        println!("compare {label}: {pw}×{ph} × {spp} spp, {secs:.1} s");
        let mut px = tonemap(&img, exposure);
        text(&mut px, pw, 10, 10, label, (pw / 240).max(1));
        for y in 0..ph {
            let row = &px[y * pw * 3..(y + 1) * pw * 3];
            strip[(y * 3 * pw + i * pw) * 3..(y * 3 * pw + (i + 1) * pw) * 3].copy_from_slice(row);
        }
    }
    png(&out.join("skin-compare.png"), (3 * pw, ph), &strip);
}

/// The exposure that puts the median of the lit pixels' luminance at 0.24
/// before the curve (skin a little over middle grey; highlights roll off in
/// the curve), the background — under 5% of the near-peak — not counted.
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

fn png(path: &Path, (w, h): (usize, usize), rgb: &[u8]) {
    image::save_buffer(
        path,
        rgb,
        w as u32,
        h as u32,
        image::ExtendedColorType::Rgb8,
    )
    .expect("png");
    println!("wrote {}", path.display());
}
