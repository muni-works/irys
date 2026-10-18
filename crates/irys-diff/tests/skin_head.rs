//! The Lee Perry-Smith head (CC BY 3.0 Infinite-Realities): the loader, the
//! maps, the inversion, and the two tiers on a crop of the face. Needs the
//! asset (`$IRYS_ASSETS/lee-perry-smith`) and the `assets` feature; skips
//! without them.
#![cfg(feature = "assets")]

use irys_diff::skin::{head, mesh, optics, render};

fn asset() -> Option<mesh::Asset> {
    let dir = mesh::asset_dir();
    if !dir.join("LeePerrySmith.glb").exists() {
        eprintln!("{} missing: skipped", dir.display());
        return None;
    }
    Some(mesh::load(&dir).expect("load"))
}

#[test]
fn loads_the_scan_and_its_maps() {
    let Some(a) = asset() else { return };
    let m = &a.mesh;
    assert_eq!(m.positions.len(), 9279);
    assert_eq!(m.indices.len() / 3, 17684);
    assert!(
        m.uvs
            .iter()
            .all(|t| (0.0..=1.0).contains(&t[0]) && (0.0..=1.0).contains(&t[1]))
    );
    // Tangents: unit, ⊥ the normal, sign ±1.
    for (t, n) in m.tangents.iter().zip(&m.normals) {
        let v = irys_core::Vec3::new(t[0], t[1], t[2]);
        assert!((v.norm() - 1.0).abs() < 1e-9 && v.dot(n).abs() < 1e-9 && t[3].abs() == 1.0);
    }
    for t in [&a.albedo, &a.spec, &a.normal] {
        assert_eq!(t.size, (1024, 1024));
    }
    let mut s = m.clone();
    head::to_scene(&mut s);
    let (lo, hi) = s.bounds();
    println!("bust bounds {lo:?} … {hi:?}");
    assert!(hi.z - lo.z > 7.9 && hi.x - lo.x > 8.5);
}

#[test]
fn the_normal_map_is_opengl() {
    // A tangent-space normal map is a height gradient: `(r, ±g)` is
    // curl-free for the right sign. Image `y` runs down, so OpenGL's
    // green-up map is the one where `(r, −g)` wins.
    let Some(a) = asset() else { return };
    let (w, h) = a.normal.size;
    let n = |x: usize, y: usize| {
        let t = a.normal.texels[y * w + x];
        (2.0 * t[0] as f64 - 1.0, 2.0 * t[1] as f64 - 1.0)
    };
    let curl = |s: f64| {
        let mut c = 0.0;
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let dy = (n(x + 1, y).1 - n(x - 1, y).1) * s;
                let dx = n(x, y + 1).0 - n(x, y - 1).0;
                c += (dy - dx).powi(2);
            }
        }
        c
    };
    let (down, up) = (curl(1.0), curl(-1.0));
    println!(
        "curl² green-down {down:.1}, green-up {up:.1} ({:.2}×)",
        down / up
    );
    assert!(down > 2.0 * up);
}

#[test]
fn chromophore_inversion_fits_the_albedo_map() {
    let Some(a) = asset() else { return };
    let (map, err) = optics::invert(&a.albedo);
    let n = map.len() as f64;
    let (m, h) = map.iter().fold((0.0, 0.0), |(m, h), t| {
        (m + t[0] as f64 / n, h + t[1] as f64 / n)
    });
    let edge = map
        .iter()
        .filter(|t| t[0] as f64 >= 0.999 * optics::MEL_MAX || t[1] as f64 >= 0.999 * optics::HB_MAX)
        .count();
    println!(
        "inversion: mean rgb error {err:.4}, mean melanin {m:.4}, mean blood {h:.4}, {edge} texels at a bound"
    );
    assert!(err < 0.05);
    assert!(m > 0.0 && h > 0.0);
}

#[cfg(feature = "gpu")]
#[test]
fn the_tiers_agree_on_a_crop_of_the_face() {
    let Some(a) = asset() else { return };
    let Ok(ctx) = irys_diff::gpu::context() else {
        eprintln!("no GPU: skipped");
        return;
    };
    let (chromo, _) = optics::invert(&a.albedo);
    for sss in [false, true] {
        let mut s = head::scene(
            &a,
            render::Chromophores::Map(chromo.clone()),
            head::View::Front,
        );
        s.set_size(512, 512);
        // A 32² window on the left cheek.
        s.size = (32, 32);
        s.crop = (200, 270);
        s.spp = 4;
        s.depth = 4;
        s.mode = render::Mode { sss, dual: true };
        let (l, c) = render::forward(&s, &render::THETA, 2);
        let g = irys_diff::skin::gpu::GpuSkin::new(ctx, &s, true);
        let (_, lg) = g.render(&s, &render::THETA, 2, true);
        let mut r: Vec<f64> = l
            .iter()
            .zip(&lg)
            .map(|(&a, &b)| (a - b as f64).abs() / a.abs().max(1e-3))
            .collect();
        let big = r.iter().filter(|&&x| x > 1e-3).count();
        r.sort_by(f64::total_cmp);
        let (sc, sg) = (
            l.iter().sum::<f64>(),
            lg.iter().map(|&x| x as f64).sum::<f64>(),
        );
        println!(
            "sss {sss}: {} lanes, median rel {:.1e}, {big} > 1e-3, sum rel {:.1e}, {c:?}",
            l.len(),
            r[r.len() / 2],
            (sc - sg).abs() / sc
        );
        assert!(r[r.len() / 2] < 1e-4);
        assert!((sc - sg).abs() / sc < 2e-2);
    }
}
