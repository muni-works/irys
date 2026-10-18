//! Phase 4: the OpenPBR material sweep, and its match against Cycles.
//!
//! ```text
//! cargo run --release -p irys-diff --example look_sweep -- <dir> [size] [spp]
//! blender -b --factory-startup -P scripts/openpbr_cycles.py -- <dir> [spp]
//! cargo run --release -p irys-diff --example look_sweep -- <dir> [size] [spp]
//! ```
//!
//! The first run renders every swatch (cached as `irys_<i>.f32`, linear
//! sRGB) and writes `sweep.json`, which the Blender script reads to build
//! the same sphere, quads, environment and orthographic camera in Cycles
//! with each swatch's parameters on a Principled BSDF (`cycles_<i>.f32`).
//! A run with both present prints the per-swatch error and writes
//! `docs/look-openpbr-sweep.png` (irys, labelled) and
//! `<dir>/compare.png` (irys over Cycles, row by row).

use std::fmt::Write as _;
use std::path::Path;

use irys_diff::look::openpbr::OpenPbr;
use irys_diff::look::swatch::{Swatch, studio};
use irys_diff::skin::render::{Light, tonemap};

fn swatches() -> Vec<(String, String, OpenPbr)> {
    let d = OpenPbr::default();
    let six = |f: &dyn Fn(f64) -> f64| (0..6).map(|i| f(i as f64 / 5.0)).collect::<Vec<_>>();
    let mut out = Vec::new();
    let mut row = |name: &str, vals: Vec<f64>, lab: &dyn Fn(f64) -> String, m: &dyn Fn(f64) -> OpenPbr| {
        for v in vals {
            out.push((name.to_string(), lab(v), m(v)));
        }
    };
    let f2 = |v: f64| format!("{v:.2}");
    row("DIELECTRIC ROUGHNESS", six(&|t| t), &f2, &|v| OpenPbr {
        base_color: [0.8, 0.3, 0.2],
        specular_roughness: v,
        ..d
    });
    row("METAL ROUGHNESS", six(&|t| t), &f2, &|v| OpenPbr {
        base_color: [1.0, 0.77, 0.34],
        base_metalness: 1.0,
        specular_roughness: v,
        ..d
    });
    row("METALNESS", six(&|t| t), &f2, &|v| OpenPbr {
        base_color: [0.9, 0.55, 0.45],
        base_metalness: v,
        specular_roughness: 0.3,
        ..d
    });
    row("COAT WEIGHT", six(&|t| t), &f2, &|v| OpenPbr {
        base_color: [0.6, 0.08, 0.08],
        specular_roughness: 0.6,
        coat_weight: v,
        coat_roughness: 0.05,
        ..d
    });
    row("COAT DARKENING 0", six(&|t| t), &f2, &|v| OpenPbr {
        base_color: [0.6, 0.08, 0.08],
        specular_roughness: 0.6,
        coat_weight: v,
        coat_roughness: 0.05,
        coat_darkening: 0.0,
        ..d
    });
    row("THIN FILM UM", six(&|t| 0.1 + 0.1 * t * 5.0), &f2, &|v| OpenPbr {
        base_color: [0.05; 3],
        specular_roughness: 0.15,
        thin_film_weight: 1.0,
        thin_film_thickness: v,
        thin_film_ior: 1.33,
        ..d
    });
    row("SPECULAR IOR", vec![1.1, 1.3, 1.5, 1.8, 2.2, 3.0], &f2, &|v| OpenPbr {
        base_color: [0.0; 3],
        specular_roughness: 0.2,
        specular_ior: v,
        ..d
    });
    row("TRANSMISSION V20", six(&|t| 0.5 * t), &f2, &|v| OpenPbr {
        base_color: [1.0; 3],
        transmission_weight: 1.0,
        specular_roughness: v,
        transmission_dispersion_scale: 1.0,
        transmission_dispersion_abbe_number: 20.0,
        ..d
    });
    row("FUZZ WEIGHT", six(&|t| t), &f2, &|v| OpenPbr {
        base_color: [0.1, 0.15, 0.5],
        specular_roughness: 0.7,
        fuzz_weight: v,
        fuzz_color: [1.0, 0.9, 0.8],
        ..d
    });
    row("SUBSURFACE WEIGHT", six(&|t| t), &f2, &|v| OpenPbr {
        base_color: [0.9, 0.6, 0.45],
        subsurface_color: [0.9, 0.6, 0.45],
        subsurface_weight: v,
        subsurface_radius: 0.3,
        specular_roughness: 0.4,
        ..d
    });
    out
}

fn material_json(m: &OpenPbr) -> String {
    let c = |v: [f64; 3]| format!("[{}, {}, {}]", v[0], v[1], v[2]);
    format!(
        concat!(
            "{{\"base_weight\": {}, \"base_color\": {}, \"base_metalness\": {}, ",
            "\"specular_weight\": {}, \"specular_color\": {}, \"specular_roughness\": {}, \"specular_ior\": {}, ",
            "\"transmission_weight\": {}, \"transmission_color\": {}, ",
            "\"transmission_dispersion_scale\": {}, \"transmission_dispersion_abbe_number\": {}, ",
            "\"subsurface_weight\": {}, \"subsurface_color\": {}, \"subsurface_radius\": {}, \"subsurface_radius_scale\": {}, ",
            "\"coat_weight\": {}, \"coat_color\": {}, \"coat_roughness\": {}, \"coat_ior\": {}, \"coat_darkening\": {}, ",
            "\"fuzz_weight\": {}, \"fuzz_color\": {}, \"fuzz_roughness\": {}, ",
            "\"thin_film_weight\": {}, \"thin_film_thickness\": {}, \"thin_film_ior\": {}}}"
        ),
        m.base_weight,
        c(m.base_color),
        m.base_metalness,
        m.specular_weight,
        c(m.specular_color),
        m.specular_roughness,
        m.specular_ior,
        m.transmission_weight,
        c(m.transmission_color),
        m.transmission_dispersion_scale,
        m.transmission_dispersion_abbe_number,
        m.subsurface_weight,
        c(m.subsurface_color),
        m.subsurface_radius,
        c(m.subsurface_radius_scale),
        m.coat_weight,
        c(m.coat_color),
        m.coat_roughness,
        m.coat_ior,
        m.coat_darkening,
        m.fuzz_weight,
        c(m.fuzz_color),
        m.fuzz_roughness,
        m.thin_film_weight,
        m.thin_film_thickness,
        m.thin_film_ior,
    )
}

fn light_json(l: &Light) -> String {
    let p = |v: [f64; 3]| format!("[{}, {}, {}]", v[0], v[1], v[2]);
    format!(
        "{{\"center\": {}, \"u\": {}, \"v\": {}, \"radiance\": {}, \"tint\": {}}}",
        p([l.center.x, l.center.y, l.center.z]),
        p([l.u.x, l.u.y, l.u.z]),
        p([l.v.x, l.v.y, l.v.z]),
        l.radiance,
        p(l.tint)
    )
}

fn read_f32(p: &Path) -> Option<Vec<f64>> {
    let b = std::fs::read(p).ok()?;
    Some(
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect(),
    )
}

fn write_f32(p: &Path, v: &[f64]) {
    let b: Vec<u8> = v.iter().flat_map(|&x| (x as f32).to_le_bytes()).collect();
    std::fs::write(p, b).unwrap();
}

// ─── a 5×7 font for the labels ───────────────────────────────────────────

fn glyph(c: char) -> [u8; 7] {
    match c {
        'A' => [14, 17, 17, 31, 17, 17, 17],
        'B' => [30, 17, 17, 30, 17, 17, 30],
        'C' => [14, 17, 16, 16, 16, 17, 14],
        'D' => [30, 17, 17, 17, 17, 17, 30],
        'E' => [31, 16, 16, 30, 16, 16, 31],
        'F' => [31, 16, 16, 30, 16, 16, 16],
        'G' => [14, 17, 16, 23, 17, 17, 15],
        'H' => [17, 17, 17, 31, 17, 17, 17],
        'I' => [14, 4, 4, 4, 4, 4, 14],
        'K' => [17, 18, 20, 24, 20, 18, 17],
        'L' => [16, 16, 16, 16, 16, 16, 31],
        'M' => [17, 27, 21, 21, 17, 17, 17],
        'N' => [17, 17, 25, 21, 19, 17, 17],
        'O' => [14, 17, 17, 17, 17, 17, 14],
        'P' => [30, 17, 17, 30, 16, 16, 16],
        'R' => [30, 17, 17, 30, 20, 18, 17],
        'S' => [15, 16, 16, 14, 1, 1, 30],
        'T' => [31, 4, 4, 4, 4, 4, 4],
        'U' => [17, 17, 17, 17, 17, 17, 14],
        'V' => [17, 17, 17, 17, 17, 10, 4],
        'W' => [17, 17, 17, 21, 21, 21, 10],
        'Y' => [17, 17, 10, 4, 4, 4, 4],
        'Z' => [31, 1, 2, 4, 8, 16, 31],
        '0' => [14, 17, 19, 21, 25, 17, 14],
        '1' => [4, 12, 4, 4, 4, 4, 14],
        '2' => [14, 17, 1, 2, 4, 8, 31],
        '3' => [31, 2, 4, 2, 1, 17, 14],
        '4' => [2, 6, 10, 18, 31, 2, 2],
        '5' => [31, 16, 30, 1, 1, 17, 14],
        '6' => [6, 8, 16, 30, 17, 17, 14],
        '7' => [31, 1, 2, 4, 8, 8, 8],
        '8' => [14, 17, 17, 14, 17, 17, 14],
        '9' => [14, 17, 17, 15, 1, 2, 12],
        '.' => [0, 0, 0, 0, 0, 12, 12],
        '-' => [0, 0, 0, 31, 0, 0, 0],
        '=' => [0, 0, 31, 0, 31, 0, 0],
        '/' => [1, 1, 2, 4, 8, 16, 16],
        '%' => [24, 25, 2, 4, 8, 19, 3],
        _ => [0; 7],
    }
}

fn text(img: &mut image::RgbImage, x: u32, y: u32, s: &str, scale: u32, col: [u8; 3]) {
    for (i, ch) in s.chars().enumerate() {
        let g = glyph(ch.to_ascii_uppercase());
        for (r, bits) in g.iter().enumerate() {
            for b in 0..5 {
                if bits >> (4 - b) & 1 == 1 {
                    for dy in 0..scale {
                        for dx in 0..scale {
                            let px = x + (i as u32 * 6 + b) * scale + dx;
                            let py = y + r as u32 * scale + dy;
                            if px < img.width() && py < img.height() {
                                img.put_pixel(px, py, image::Rgb(col));
                            }
                        }
                    }
                }
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).map(String::as_str).unwrap_or("out/look-sweep"));
    let size: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(96);
    let spp: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);
    std::fs::create_dir_all(dir).unwrap();
    let sw = swatches();
    let lights: Vec<Light> = studio()
        .into_iter()
        .map(|mut l| {
            l.tint = [1.0; 3];
            l
        })
        .collect();
    let proto = Swatch::new(OpenPbr::default(), (size, size), spp);
    // The scene for Blender.
    let mut js = String::new();
    write!(
        js,
        "{{\"size\": {size}, \"width\": {}, \"env\": {}, \"depth\": {}, \"lights\": [{}], \"swatches\": [",
        proto.width,
        proto.env,
        proto.depth,
        lights.iter().map(light_json).collect::<Vec<_>>().join(", ")
    )
    .unwrap();
    js += &sw
        .iter()
        .map(|(r, c, m)| format!("{{\"row\": \"{r}\", \"label\": \"{c}\", \"m\": {}}}", material_json(m)))
        .collect::<Vec<_>>()
        .join(",\n");
    js += "]}\n";
    std::fs::write(dir.join("sweep.json"), js).unwrap();

    let t0 = std::time::Instant::now();
    let imgs: Vec<Vec<f64>> = sw
        .iter()
        .enumerate()
        .map(|(i, (_, _, m))| {
            let p = dir.join(format!("irys_{i}.f32"));
            if let Some(v) = read_f32(&p).filter(|v| v.len() == size * size * 3) {
                return v;
            }
            let mut s = Swatch::new(*m, (size, size), spp);
            s.lights = lights.clone();
            let img = s.render(&m.theta(), 0x5eed + i as u64);
            write_f32(&p, &img);
            img
        })
        .collect();
    println!("irys: {} swatches {size}² × {spp} spp in {:.1} s", sw.len(), t0.elapsed().as_secs_f64());

    // The sphere's pixels (orthographic: |x| < 1 in film units).
    let fw = proto.width;
    let mask: Vec<bool> = (0..size * size)
        .map(|p| {
            let (x, y) = ((p % size) as f64 + 0.5, (p / size) as f64 + 0.5);
            let (u, v) = (-0.5 * fw + x / size as f64 * fw, 0.5 * fw - y / size as f64 * fw);
            u * u + v * v < 0.97 * 0.97
        })
        .collect();
    let cycles: Vec<Option<Vec<f64>>> = (0..sw.len())
        .map(|i| read_f32(&dir.join(format!("cycles_{i}.f32"))).filter(|v| v.len() == size * size * 3))
        .collect();
    let have_cycles = cycles.iter().all(Option::is_some);
    let mut table = String::from("row,label,irys_mean_r,irys_mean_g,irys_mean_b,cycles_mean_r,cycles_mean_g,cycles_mean_b,mean_rel_err,pixel_rel_mae\n");
    let mut errs = Vec::new();
    if have_cycles {
        println!("{:>22} {:>6}  {:>8}  {:>8}", "row", "value", "mean err", "pixel MAE");
        for (i, (r, l, _)) in sw.iter().enumerate() {
            let (a, b) = (&imgs[i], cycles[i].as_ref().unwrap());
            let mut ma = [0.0; 3];
            let mut mb = [0.0; 3];
            let (mut num, mut den) = (0.0, 0.0);
            let mut n = 0.0;
            for p in 0..size * size {
                if !mask[p] {
                    continue;
                }
                n += 1.0;
                for c in 0..3 {
                    ma[c] += a[p * 3 + c];
                    mb[c] += b[p * 3 + c];
                    num += (a[p * 3 + c] - b[p * 3 + c]).abs();
                    den += b[p * 3 + c];
                }
            }
            let (sa, sb): (f64, f64) = (ma.iter().sum(), mb.iter().sum());
            let mean = (sa - sb) / sb;
            let mae = num / den;
            errs.push((r.clone(), l.clone(), mean, mae));
            println!("{r:>22} {l:>6}  {:+7.2}%  {:7.2}%", 100.0 * mean, 100.0 * mae);
            writeln!(
                table,
                "{r},{l},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5}",
                ma[0] / n, ma[1] / n, ma[2] / n, mb[0] / n, mb[1] / n, mb[2] / n, mean, mae
            )
            .unwrap();
        }
        std::fs::write(dir.join("match.csv"), &table).unwrap();
    }

    // The labelled grid.
    let rows: Vec<&String> = {
        let mut v: Vec<&String> = Vec::new();
        for (r, _, _) in &sw {
            if v.last() != Some(&r) {
                v.push(r);
            }
        }
        v
    };
    let cols = 6;
    let (pad, top, left) = (8u32, 44u32, 190u32);
    let cell = size as u32 + pad;
    let label_h = 16u32;
    let rows_n = rows.len() as u32;
    let width = left + cols * cell + pad;
    let height = top + rows_n * (cell + label_h) + pad;
    let compose = |srcs: &dyn Fn(usize) -> Vec<f64>, title: &str, with_err: bool| {
        let mut img = image::RgbImage::from_pixel(width, height, image::Rgb([24, 24, 26]));
        text(&mut img, pad, 12, title, 2, [230, 230, 230]);
        for (i, (_, l, _)) in sw.iter().enumerate() {
            let (r, c) = ((i / cols as usize) as u32, (i % cols as usize) as u32);
            let x0 = left + c * cell;
            let y0 = top + r * (cell + label_h);
            let px = tonemap(&srcs(i), 1.0);
            for y in 0..size as u32 {
                for x in 0..size as u32 {
                    let k = ((y as usize) * size + x as usize) * 3;
                    img.put_pixel(x0 + x, y0 + y, image::Rgb([px[k], px[k + 1], px[k + 2]]));
                }
            }
            let lab = if with_err && !errs.is_empty() {
                format!("{l} {:+.1}%", 100.0 * errs[i].2)
            } else {
                l.clone()
            };
            text(&mut img, x0, y0 + size as u32 + 3, &lab, 1, [200, 200, 200]);
            if c == 0 {
                text(&mut img, pad, y0 + size as u32 / 2 - 4, &sw[i].0, 1, [230, 230, 230]);
            }
        }
        img
    };
    let docs = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/look-openpbr-sweep.png");
    let title = if have_cycles {
        "OPENPBR IN IRYS - MEAN ERR VS CYCLES"
    } else {
        "OPENPBR IN IRYS"
    };
    compose(&|i| imgs[i].clone(), title, have_cycles).save(&docs).unwrap();
    println!("wrote {}", docs.display());
    if have_cycles {
        compose(&|i| cycles[i].clone().unwrap(), "CYCLES PRINCIPLED - SAME SCENE", false)
            .save(dir.join("cycles.png"))
            .unwrap();
        println!("wrote {}", dir.join("cycles.png").display());
    }
}
