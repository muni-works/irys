//! The tag: print it in a synthetic photo, find it, solve the camera.

use irys_recon::tag::{self, Grey, Intrinsics, Pose};

fn look_at(eye: [f64; 3], at: [f64; 3]) -> Pose {
    let f = norm(sub(at, eye));
    let r = norm(cross(f, [0.0, 0.0, 1.0]));
    let d = cross(f, r);
    let rm = [r, d, f];
    let t = [-dot(rm[0], eye), -dot(rm[1], eye), -dot(rm[2], eye)];
    Pose { r: rm, t }
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }
fn norm(a: [f64; 3]) -> [f64; 3] { let n = dot(a, a).sqrt(); [a[0] / n, a[1] / n, a[2] / n] }

/// A tag on grey paper seen by `pose`, 4 × 4 supersampled.
fn photo(k: &Intrinsics, pose: &Pose, size: (usize, usize)) -> Grey {
    let cells = tag::cells(tag::TAG_ID);
    let eye = pose.eye();
    let mut px = vec![0f32; size.0 * size.1];
    for y in 0..size.1 {
        for x in 0..size.0 {
            let mut acc = 0.0;
            for j in 0..4 {
                for i in 0..4 {
                    let u = (x as f64 + (i as f64 + 0.5) / 4.0 - k.cx) / k.f;
                    let v = (y as f64 + (j as f64 + 0.5) / 4.0 - k.cy) / k.f;
                    // Camera ray in the world: Rᵀ (u, v, 1).
                    let d: Vec<f64> = (0..3).map(|c| pose.r[0][c] * u + pose.r[1][c] * v + pose.r[2][c]).collect();
                    let t = -eye[2] / d[2];
                    let (wx, wy) = (eye[0] + t * d[0], eye[1] + t * d[1]);
                    acc += match tag::white_at(&cells, tag::TAG_MM, wx, wy) {
                        Some(true) => 0.85,
                        Some(false) => 0.06,
                        None => 0.6,
                    };
                }
            }
            px[y * size.0 + x] = (acc / 16.0) as f32;
        }
    }
    Grey { size, px }
}

#[test]
fn finds_the_tag_and_the_camera() {
    let size = (1200, 900);
    let k = Intrinsics { f: 1000.0, cx: 600.0, cy: 450.0 };
    for (eye, at) in [([120.0, -260.0, 220.0], [10.0, 5.0, 0.0]), ([-200.0, -150.0, 140.0], [-5.0, 10.0, 0.0])] {
        let pose = look_at(eye, at);
        let img = photo(&k, &pose, size);
        let obs = tag::detect(&img, tag::TAG_MM);
        assert_eq!(obs.len(), 1, "one tag");
        let o = &obs[0];
        assert_eq!(o.id, tag::TAG_ID);
        let mut worst = 0.0f64;
        for (c, q) in tag::corners_mm(tag::TAG_MM).iter().zip(&o.corners) {
            let p = pose.project(&k, [c[0], c[1], 0.0]).unwrap();
            worst = worst.max((p[0] - q[0]).hypot(p[1] - q[1]));
        }
        println!("corner error {worst:.3} px");
        assert!(worst < 0.25, "corners within a quarter pixel: {worst}");
        // Solve with a focal prior 4 % off.
        let (k2, p2, rms) = tag::solve_camera(o, tag::TAG_MM, 1040.0, 0.05, 600.0, 450.0).unwrap();
        let (e1, e2) = (pose.eye(), p2.eye());
        let de = ((e1[0] - e2[0]).powi(2) + (e1[1] - e2[1]).powi(2) + (e1[2] - e2[2]).powi(2)).sqrt();
        println!("f {:.1} eye error {de:.2} mm rms {rms:.3}", k2.f);
        assert!(rms < 0.3);
        assert!((k2.f - 1000.0).abs() < 40.0);
    }
}

#[test]
fn finds_a_negative_tag() {
    let size = (1200, 900);
    let k = Intrinsics { f: 1000.0, cx: 600.0, cy: 450.0 };
    let pose = look_at([120.0, -260.0, 220.0], [10.0, 5.0, 0.0]);
    let mut img = photo(&k, &pose, size);
    for v in &mut img.px {
        *v = 1.0 - *v;
    }
    let obs = tag::detect(&img, tag::TAG_MM);
    assert_eq!(obs.len(), 1, "one negative tag");
    assert_eq!(obs[0].id, tag::TAG_ID);
    let mut worst = 0.0f64;
    for (c, q) in tag::corners_mm(tag::TAG_MM).iter().zip(&obs[0].corners) {
        let p = pose.project(&k, [c[0], c[1], 0.0]).unwrap();
        worst = worst.max((p[0] - q[0]).hypot(p[1] - q[1]));
    }
    assert!(worst < 0.25, "negative tag corners within a quarter pixel: {worst}");
}

#[test]
fn exif_round_trip() {
    let img = image::RgbImage::from_pixel(16, 8, image::Rgb([90, 90, 90]));
    let mut buf = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 90).encode_image(&img).unwrap();
    let j = tag::with_exif(&buf, 5.96, 26);
    let e = tag::read_exif(&j);
    assert_eq!(e.f35, Some(26.0));
    assert!((e.f_mm.unwrap() - 5.96).abs() < 1e-9);
    assert!(image::load_from_memory(&j).is_ok());
}
