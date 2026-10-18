use irys_recon::{dsl::Program, generate, model::*, render};
fn main() {
    let a = std::env::args().nth(1).unwrap();
    let truth = a.parse::<u64>().map(generate::program).unwrap_or_else(|_| Program::from_text(&a.replace(';', "\n")).unwrap());
    let setup = Setup::new(render::views());
    let t = target(&truth, &setup, 64, 1).unwrap();
    let (w, h) = setup.film.size;
    let mut img = image::GrayImage::new(w as u32 * 3, h as u32);
    for (v, im) in t.iter().enumerate() { for p in 0..w*h { img.put_pixel((v*w + p % w) as u32, (p / w) as u32, image::Luma([render::srgb8(im[p])])); } }
    img.save(std::env::args().nth(2).unwrap()).unwrap();
    let mx = t[0].iter().cloned().fold(0.0, f64::max);
    println!("max {mx}");
}
