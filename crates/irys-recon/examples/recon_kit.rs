//! The physical kit for v1: a printable tag, three test parts and how to
//! shoot them.
//!
//! `cargo run --release -p irys-recon --example recon_kit -- [OUT]`
//! (default `../irys-assets/recon-v1-kit`) writes:
//!
//! - `tag36h11-id0-50mm.pdf`: the tag at an exact size (the black square
//!   50.0 mm), a 100 mm scale bar, a 90 mm cut line, printing notes;
//! - `parts/<name>.stl` (vcad's exporter) and `parts/<name>.txt` (the
//!   ground-truth program) for three printable parts;
//! - `SHOOTING.md`.

use std::fmt::Write as _;
use std::path::PathBuf;

use irys_recon::dsl::{Axis, Corner, Feature, Program};
use irys_recon::{export, tag};

/// The test parts: each exercises a different mix of the family. The first
/// three are hand-made; the last two are synthetic-gate parts (seeds 3004 and
/// 3012) that pass at 0.5 mm in the 3-view gate, so a real photo can be held
/// against the same part's synthetic result.
const PARTS: [(&str, &str); 5] = [
    (
        "kit-a-holes-counterbore-slot",
        "base 72.000 52.000 8.000 fillet 5.000\nhole -24.000 12.000 3.200\ncounterbore 17.000 11.000 2.200 4.600 3.000\nslotx -6.000 -13.000 14.000 2.600",
    ),
    (
        "kit-b-boss-circular",
        "base 64.000 48.000 10.000 chamfer 4.000\nboss 15.000 4.000 7.000 8.000\ncircular 3 -14.000 -2.000 9.000 90.000 1.800\nhole 16.000 -15.000 2.600",
    ),
    (
        "kit-c-linear-boss-counterbore-slot",
        "base 84.000 56.000 6.000\nlinear 3 -28.000 16.000 13.000 0.000 2.000\nboss -14.000 -12.000 6.000 6.000\ncounterbore 14.000 -12.000 2.700 5.200 2.500\nsloty 30.000 4.000 12.000 2.200",
    ),
    (
        "kit-d-gate3004-linear-boss",
        "base 58.800 56.700 8.300 chamfer 5.500\nhole -16.400 6.200 5.000\nlinear 2 -7.700 -0.400 9.900 0.000 2.600\nboss 1.000 -13.200 7.900 3.800",
    ),
    (
        "kit-e-gate3012-circular-counterbore",
        "base 65.400 61.100 11.000\ncircular 4 15.900 -13.000 14.500 30.800 2.500\ncounterbore -1.500 7.900 2.400 5.600 3.100",
    ),
];

fn main() {
    let out = PathBuf::from(std::env::args().nth(1).unwrap_or("../irys-assets/recon-v1-kit".into()));
    std::fs::create_dir_all(out.join("parts")).unwrap();
    std::fs::write(out.join("tag36h11-id0-50mm.pdf"), tag_pdf()).unwrap();
    for (name, text) in PARTS {
        let p = Program::from_text(text).unwrap();
        assert!(p.valid(), "{name} is not a valid program");
        let s = p.build().unwrap_or_else(|e| panic!("{name}: {e}"));
        std::fs::write(out.join("parts").join(format!("{name}.stl")), export::stl(&p).unwrap()).unwrap();
        std::fs::write(out.join("parts").join(format!("{name}.txt")), p.to_text() + "\n").unwrap();
        std::fs::write(out.join("parts").join(format!("{name}.calipers.csv")), calipers(&p)).unwrap();
        std::fs::write(out.join("parts").join(format!("{name}.drawing.svg")), drawing(&p, name)).unwrap();
        println!("{name}: {} params, volume {:.0} mm³ (vcad {:.0})", p.params().len(), p.volume(), s.volume());
    }
    std::fs::write(out.join("SHOOTING.md"), SHOOTING).unwrap();
    std::fs::write(out.join("CALIPERS.md"), CALIPERS).unwrap();
    println!("kit written to {}", out.display());
}

/// One A4 page: the tag, a scale bar, a cut line, notes. Hand-written PDF
/// 1.4 (vector rectangles, base-14 Helvetica, nothing embedded).
fn tag_pdf() -> Vec<u8> {
    let mm = 72.0 / 25.4;
    let (pw, ph) = (210.0 * mm, 297.0 * mm);
    let mut c = String::new();
    let text = |c: &mut String, x: f64, y: f64, size: f64, s: &str| {
        let esc = s.replace('\\', "\\\\").replace('(', "\\(").replace(')', "\\)");
        let _ = writeln!(c, "BT /F1 {size} Tf {x:.2} {y:.2} Td ({esc}) Tj ET");
    };
    // The tag: the black square 50 mm, centred 175 mm above the bottom.
    let (cx, cy) = (pw / 2.0, 175.0 * mm);
    let cell = tag::TAG_MM / 8.0 * mm;
    let cells = tag::cells(tag::TAG_ID);
    // The black square whole, then its white cells on top: no seams
    // between black cells.
    let b = 4.0 * cell;
    let _ = writeln!(c, "0 0 0 rg {:.3} {:.3} {:.3} {:.3} re f", cx - b, cy - b, 2.0 * b, 2.0 * b);
    c += "1 1 1 rg\n";
    for (r, row) in cells.iter().enumerate().take(9).skip(1) {
        for (k, &white) in row.iter().enumerate().take(9).skip(1) {
            if white {
                let x = cx + (k as f64 - 5.0) * cell;
                let y = cy + (5.0 - r as f64 - 1.0) * cell;
                let _ = writeln!(c, "{x:.3} {y:.3} {cell:.3} {cell:.3} re f");
            }
        }
    }
    c += "0 0 0 rg\n";
    // The 90 mm cut square.
    let half = 45.0 * mm;
    let _ = writeln!(c, "0.4 G 0.5 w [4 3] 0 d {:.2} {:.2} {:.2} {:.2} re S [] 0 d", cx - half, cy - half, 2.0 * half, 2.0 * half);
    c += "0 G 0 0 0 rg\n";
    text(&mut c, cx - half, cy - half - 12.0, 8.0, "cut along the dashed line (90 mm square); keep the white border");
    // A 100 mm scale bar with 10 mm ticks.
    let (x0, y0) = (pw / 2.0 - 50.0 * mm, 95.0 * mm);
    let _ = writeln!(c, "0.8 w {x0:.2} {y0:.2} m {:.2} {y0:.2} l S", x0 + 100.0 * mm);
    for i in 0..=10 {
        let x = x0 + i as f64 * 10.0 * mm;
        let t = if i % 5 == 0 { 4.0 * mm } else { 2.5 * mm };
        let _ = writeln!(c, "{x:.2} {y0:.2} m {x:.2} {:.2} l S", y0 + t);
        text(&mut c, x - 3.0, y0 - 11.0, 8.0, &format!("{}", i * 10));
    }
    for i in 0..100 {
        let x = x0 + i as f64 * mm;
        let _ = writeln!(c, "0.3 w {x:.2} {y0:.2} m {x:.2} {:.2} l S 0.8 w", y0 + 1.2 * mm);
    }
    text(&mut c, x0, y0 + 7.0 * mm, 9.0, "scale bar: 100 mm (ticks every 10 mm, fine ticks every 1 mm)");
    // Notes.
    let lines = [
        ("irys recon v1 - AprilTag tag36h11, id 0", 15.0),
        ("Black square: exactly 50.0 mm a side (the white margin around it is part of the tag).", 9.5),
        ("", 6.0),
        ("Print at 100 % / Actual size. Turn off 'Fit to page', 'Shrink oversized pages' and any scaling.", 9.5),
        ("Check with a ruler: the black square must measure 50.0 mm and the scale bar 100 mm.", 9.5),
        ("If either is off by more than 0.5 mm, fix the print settings and print again: the tag is the", 9.5),
        ("metric reference, and every recovered dimension inherits its error.", 9.5),
        ("Matte paper, black ink at full density; keep it flat (tape the corners down).", 9.5),
        ("Cut along the dashed 90 mm square. Lay it flat beside the part on the same surface.", 9.5),
    ];
    let mut y = ph - 22.0 * mm;
    for (s, size) in lines {
        text(&mut c, 18.0 * mm, y, size, s);
        y -= size * 1.55;
    }
    text(&mut c, 18.0 * mm, 18.0 * mm, 7.5, "tag36h11 code table: AprilRobotics apriltag (BSD-2-Clause), via kornia-apriltag (Apache-2.0).");
    // Assemble the PDF.
    let objs = [
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
        format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw:.2} {ph:.2}] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"),
        format!("<< /Length {} >>\nstream\n{c}endstream", c.len()),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>".to_string(),
    ];
    let mut pdf = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n".to_vec();
    let mut offs = Vec::new();
    for (i, o) in objs.iter().enumerate() {
        offs.push(pdf.len());
        pdf.extend_from_slice(format!("{} 0 obj\n{o}\nendobj\n", i + 1).as_bytes());
    }
    let xref = pdf.len();
    let mut x = format!("xref\n0 {}\n0000000000 65535 f \n", objs.len() + 1);
    for o in offs {
        let _ = writeln!(x, "{o:010} 00000 n ");
    }
    let _ = write!(x, "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", objs.len() + 1);
    pdf.extend_from_slice(x.as_bytes());
    pdf
}

const SHOOTING: &str = r#"# Shooting a part for irys recon v1

One photo is enough; three are better. Read this once, then shoot.

## What you need

- `tag36h11-id0-50mm.pdf`, printed at **100 % / Actual size** (no "fit to
  page"). Measure it: the black square must be **50.0 mm** and the scale
  bar **100 mm**. Cut along the dashed 90 mm square.
- One of the test parts in `parts/` (STL), printed in a **saturated colour**
  of PLA (orange, blue, red, green, yellow, purple). Not white, grey or
  black: the segmenter tells the part from the backdrop by colour.
  0.2 mm layers, no brim (it changes the outline), supports off (none are
  needed). Its ground truth is the `.txt` beside the STL.
- A **plain matte backdrop**, lighter than the part: a sheet of white
  printer paper or poster board, bigger than about 30 × 25 cm. No pattern,
  no wood grain, no glossy table.

## Set up

1. Lay the backdrop flat. Put the tag on it and the part beside it, **on
   the same surface**, standing on its flat bottom, with **2–5 cm** between
   the part and the tag's paper. Nothing on top of the tag; nothing
   touching the part.
2. Light: **diffuse room light plus one soft key** (a window off to one
   side, or a desk lamp bounced off a wall). No direct sun, no hard spot
   right overhead, no flash. A soft shadow beside the part is fine — the
   model fits it.
3. Don't move the part or the tag between shots.

## Shoot

- **Three photos** from three sides, each at **30–60° above the table**
  (not straight down, not grazing), 25–40 cm away, with both the tag and
  the whole part in frame and the part roughly in the middle third.
- Phone camera app, 1× lens (the main camera, not ultra-wide), **no
  portrait mode, no zoom, no filters**. Hold still; let it focus on the
  part.
- **Keep the EXIF** (focal length). AirDrop or a cable keeps it; some
  messaging apps strip it. HEIC is fine if you export to JPEG with
  metadata.

## Where the files go

Drop them in `/Users/cam/Developer/irys-assets/recon-m2-photos/<part name>/`
(e.g. `kit-a-holes-counterbore-slot/IMG_0412.jpg`). Never commit photos.

Then:

```
cargo run --release -p irys-recon --example recon_photo -- \
  ../irys-assets/recon-m2-photos/kit-a-holes-counterbore-slot/*.jpg \
  --truth ../irys-assets/recon-m2-kit/parts/kit-a-holes-counterbore-slot.txt
```

It writes `program.txt`, `part.vcad`, `part.stl`, `overlay_*.png` and
`report.md` (each dimension with its ±σ, and against the truth when given).
The v1 gate: reprint the recovered part and check it with calipers —
every dimension within 1 mm.
"#;

/// What to measure on the printed part, as a sheet to fill in: each quantity a
/// caliper can reach, how to reach it, and the nominal value. Positions are in
/// the drawing's frame: from the LEFT edge (−x) and the FRONT edge (−y), with
/// the part oriented as `<name>.drawing.svg` shows it.
fn calipers(p: &Program) -> String {
    let mut s = String::from("quantity,how,nominal_mm,measured_1,measured_2,measured_3\n");
    let mut row = |q: &str, how: &str, v: f64| {
        let _ = writeln!(s, "\"{q}\",\"{how}\",{v:.2},,,");
    };
    let (hw, hd) = (p.w / 2.0, p.d / 2.0);
    row("plate width", "outside jaws, left to right edge", p.w);
    row("plate depth", "outside jaws, front to back edge", p.d);
    row("plate thickness", "outside jaws, away from features", p.h);
    match p.corner {
        Corner::Fillet(r) => row("corner fillet radius", "radius gauge, or compare to the drawing", r),
        Corner::Chamfer(c) => row("corner chamfer leg", "depth rod or outside jaws along an edge", c),
        Corner::Sharp => {}
    }
    for (i, f) in p.features.iter().enumerate() {
        let n = i + 1;
        let pos = |row: &mut dyn FnMut(&str, &str, f64), what: &str, x: f64, y: f64, r: f64| {
            row(&format!("f{n} {what} centre from left edge"), "edge to near side of the bore + bore radius", hw + x);
            row(&format!("f{n} {what} centre from front edge"), "edge to near side of the bore + bore radius", hd + y);
            let _ = r;
        };
        match *f {
            Feature::Hole { x, y, r } => {
                row(&format!("f{n} hole diameter"), "inside jaws", 2.0 * r);
                pos(&mut row, "hole", x, y, r);
            }
            Feature::Counterbore { x, y, r, cr, cd } => {
                row(&format!("f{n} counterbore through-hole diameter"), "inside jaws, below the step", 2.0 * r);
                row(&format!("f{n} counterbore diameter"), "inside jaws, at the top", 2.0 * cr);
                row(&format!("f{n} counterbore depth"), "depth rod, top face to the step", cd);
                pos(&mut row, "counterbore", x, y, cr);
            }
            Feature::Slot { x, y, len, r, axis } => {
                row(&format!("f{n} slot width"), "inside jaws, across the slot", 2.0 * r);
                row(&format!("f{n} slot overall length"), "inside jaws, end to end", len + 2.0 * r);
                pos(&mut row, if axis == Axis::X { "slot (along x)" } else { "slot (along y)" }, x, y, r);
            }
            Feature::Boss { x, y, r, bh } => {
                row(&format!("f{n} boss diameter"), "outside jaws", 2.0 * r);
                row(&format!("f{n} boss height above the plate"), "depth rod, boss top to plate top", bh);
                pos(&mut row, "boss", x, y, r);
            }
            Feature::LinearHoles { x, y, dx, dy, r, n: k } => {
                row(&format!("f{n} {k} holes in a row: diameter"), "inside jaws, each hole", 2.0 * r);
                row(&format!("f{n} hole pitch"), "near side of hole 1 to near side of hole 2", dx.hypot(dy));
                pos(&mut row, "first hole of the row", x, y, r);
            }
            Feature::CircularHoles { x, y, rad, r, n: k, .. } => {
                row(&format!("f{n} {k} holes on a circle: diameter"), "inside jaws, each hole", 2.0 * r);
                row(&format!("f{n} bolt-circle diameter"), "across two opposite holes, near side to far side, minus one hole diameter", 2.0 * rad);
                row(&format!("f{n} bolt circle centre from left edge"), "midpoint of two opposite holes: average their centre distances", hw + x);
                row(&format!("f{n} bolt circle centre from front edge"), "midpoint of two opposite holes: average their centre distances", hd + y);
            }
        }
    }
    s
}

/// A top view at 1:1 (1 px per 0.25 mm), with the edges named the way the
/// caliper sheet names them.
fn drawing(p: &Program, name: &str) -> String {
    let k = 4.0;
    let (w, d) = (p.w * k, p.d * k);
    let (ox, oy) = (60.0, 60.0);
    let cx = |x: f64| ox + (x + p.w / 2.0) * k;
    let cy = |y: f64| oy + (p.d / 2.0 - y) * k;
    let mut s = format!(r##"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" font-family="monospace" font-size="14"><rect width="100%" height="100%" fill="white"/>"##, (w + 2.0 * ox).max(760.0), d + 2.0 * oy + 40.0);
    let rr = match p.corner { Corner::Fillet(r) => r * k, _ => 0.0 };
    let _ = write!(s, r##"<rect x="{ox}" y="{oy}" width="{w}" height="{d}" rx="{rr}" fill="none" stroke="black" stroke-width="2"/>"##);
    let circ = |s: &mut String, x: f64, y: f64, r: f64, dash: bool| {
        let _ = write!(s, r##"<circle cx="{:.1}" cy="{:.1}" r="{:.1}" fill="none" stroke="black" stroke-width="1.5"{}/>"##, cx(x), cy(y), r * k, if dash { r#" stroke-dasharray="6 4""# } else { "" });
    };
    for (i, f) in p.features.iter().enumerate() {
        let (lx, ly) = match *f {
            Feature::Hole { x, y, r } => { circ(&mut s, x, y, r, false); (x, y) }
            Feature::Counterbore { x, y, r, cr, .. } => { circ(&mut s, x, y, r, false); circ(&mut s, x, y, cr, false); (x, y) }
            Feature::Boss { x, y, r, .. } => { circ(&mut s, x, y, r, true); (x, y) }
            Feature::Slot { x, y, len, r, axis } => {
                let (hx, hy) = if axis == Axis::X { (len / 2.0, 0.0) } else { (0.0, len / 2.0) };
                let _ = write!(s, r##"<line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="black" stroke-width="{:.1}" stroke-linecap="round" stroke-opacity="0.25"/>"##, cx(x - hx), cy(y - hy), cx(x + hx), cy(y + hy), 2.0 * r * k);
                (x, y)
            }
            Feature::LinearHoles { x, y, dx, dy, r, n } => { for j in 0..n { circ(&mut s, x + dx * j as f64, y + dy * j as f64, r, false); } (x, y) }
            Feature::CircularHoles { x, y, rad, phase, r, n } => {
                for j in 0..n { let a = (phase + 360.0 * j as f64 / n as f64).to_radians(); circ(&mut s, x + rad * a.cos(), y + rad * a.sin(), r, false); }
                (x, y)
            }
        };
        let _ = write!(s, r##"<text x="{:.1}" y="{:.1}" fill="#c00">f{}</text>"##, cx(lx) + 6.0, cy(ly) - 6.0, i + 1);
    }
    let _ = write!(s, r##"<text x="{ox}" y="{}" text-anchor="start">FRONT EDGE (-y)</text><text x="{}" y="{}" transform="rotate(-90 {} {})" text-anchor="middle">LEFT EDGE (-x)</text><text x="{ox}" y="30">{name}: top view, bosses dashed; dimensions in .calipers.csv</text></svg>"##,
        oy + d + 24.0, ox - 16.0, oy + d / 2.0, ox - 16.0, oy + d / 2.0);
    s
}

const CALIPERS: &str = r#"# Measuring the printed parts (the M2 gate)

The paper's claim is that a photograph measures a part. This is the check: the
same parts, measured with calipers, are the standard the photos are read against.

1. **Before photographing**, measure each part with digital calipers and fill in
   `parts/<name>.calipers.csv`: three readings per quantity, in the three blank
   columns. Hold the part as `parts/<name>.drawing.svg` shows it, so "left edge"
   and "front edge" mean the same thing as in the sheet.
2. Positions: measure from the edge to the **near side** of the bore (inside of
   the hole), then add the bore's radius (half its measured diameter).
3. Thickness: away from bosses, chamfers and the bottom edge (a printed part's
   first layer flares out by 0.2 to 0.4 mm, the "elephant's foot"; measure above it).
4. Note anything odd (warping, a stringy hole) in a last column.

The recovery writes `report.md` per part; the comparison against these sheets is
the M2 table: recovered − measured, per quantity, with the recovery's ±σ.
"#;
