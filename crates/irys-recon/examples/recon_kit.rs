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

use irys_recon::dsl::Program;
use irys_recon::{export, tag};

/// The test parts: each exercises a different mix of the family.
const PARTS: [(&str, &str); 3] = [
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
        println!("{name}: {} params, volume {:.0} mm³ (vcad {:.0})", p.params().len(), p.volume(), s.volume());
    }
    std::fs::write(out.join("SHOOTING.md"), SHOOTING).unwrap();
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

Drop them in `/Users/cam/Developer/irys-assets/recon-v1-photos/<part name>/`
(e.g. `kit-a-holes-counterbore-slot/IMG_0412.jpg`). Never commit photos.

Then:

```
cargo run --release -p irys-recon --example recon_photo -- \
  ../irys-assets/recon-v1-photos/kit-a-holes-counterbore-slot/*.jpg \
  --truth ../irys-assets/recon-v1-kit/parts/kit-a-holes-counterbore-slot.txt
```

It writes `program.txt`, `part.vcad`, `part.stl`, `overlay_*.png` and
`report.md` (each dimension with its ±σ, and against the truth when given).
The v1 gate: reprint the recovered part and check it with calipers —
every dimension within 1 mm.
"#;
