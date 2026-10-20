//! Detect tag36h11 tags in a photo (either polarity) and print id, Hamming
//! distance, centre and apparent size.
//! `cargo run --release -p irys-recon --example tag_probe -- photo.jpg [tag_mm]`

use irys_recon::tag::{Grey, detect};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("photo path");
    let tag_mm: f64 = args.next().map(|s| s.parse().expect("tag_mm")).unwrap_or(150.0);
    let img = image::open(&path).expect("open").to_luma8();
    let (w, h) = img.dimensions();
    let px: Vec<f32> = img.pixels().map(|p| p.0[0] as f32 / 255.0).collect();
    let g = Grey { size: (w as usize, h as usize), px };
    let tags = detect(&g, tag_mm);
    println!("{} tags in {w}x{h}", tags.len());
    for t in tags {
        let c = t.corners;
        let cx = c.iter().map(|p| p[0]).sum::<f64>() / 4.0;
        let cy = c.iter().map(|p| p[1]).sum::<f64>() / 4.0;
        let side = (0..4).map(|i| {
            let (a, b) = (c[i], c[(i + 1) % 4]);
            (a[0] - b[0]).hypot(a[1] - b[1])
        }).sum::<f64>() / 4.0;
        println!("  id {} hamming {} centre ({cx:.0}, {cy:.0}) side {side:.0} px", t.id, t.hamming);
    }
}
