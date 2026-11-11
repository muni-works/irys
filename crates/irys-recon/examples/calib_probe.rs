//! Find the tag in photos and solve the camera, as photo import does.
//! `cargo run --release -p irys-recon --example calib_probe -- photo.jpg [...]`
fn main() {
    for p in std::env::args().skip(1) {
        let ph = irys_recon::v1::load(std::path::Path::new(&p)).expect("load");
        match irys_recon::v1::calibrate(&ph, irys_recon::tag::TAG_MM, None) {
            Ok(c) => println!("{p}: {}x{} id {} hamming {} f {:.0} (exif {:?}) rms {:.2} px", ph.size.0, ph.size.1, c.tag.id, c.tag.hamming, c.k.f, c.f_exif.map(|f| f.round()), c.rms),
            Err(e) => println!("{p}: {e}"),
        }
    }
}
