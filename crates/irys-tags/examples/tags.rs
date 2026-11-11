//! Detect tag36h11 tags in photos and print id, polarity, Hamming distance,
//! centre, side and corners. `IRYS_TAGS_DEBUG=1` prints every quad tried.
//! `cargo run --release -p irys-tags --example tags -- photo.jpg [...]`

fn main() {
    for path in std::env::args().skip(1) {
        let img = image::open(&path).expect("open").to_luma8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let g = irys_tags::Grey::from_luma8(w, h, img.as_raw());
        let t0 = std::time::Instant::now();
        let tags = irys_tags::detect(&g);
        println!("{path}: {} tags in {w}x{h} ({:.0?})", tags.len(), t0.elapsed());
        for t in tags {
            let [cx, cy] = t.centre();
            println!("  id {} {} hamming {} centre ({cx:.0}, {cy:.0}) side {:.0} px corners {:?}", t.id, if t.negative { "negative" } else { "normal" }, t.hamming, t.side(), t.corners.map(|p| [p[0].round(), p[1].round()]));
        }
    }
}
