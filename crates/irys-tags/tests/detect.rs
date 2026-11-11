//! Synthetic tags at known corners, and two real phone photos.

use irys_tags::{CORNER_ORDER, Grey, apply, cells, corners_mm, detect, homography, white_at};

/// Render tag `id` (8-cell black square plus its quiet zone) so its black
/// square's corners land on `quad` (in [`CORNER_ORDER`]), 4 × 4
/// supersampled, on a 0.55 grey ground.
fn render(id: usize, quad: [[f64; 2]; 4], size: (usize, usize), negative: bool) -> Grey {
    let c = cells(id).unwrap();
    let h = homography(&quad, &corners_mm(8.0)).unwrap();
    let mut px = vec![0f32; size.0 * size.1];
    for y in 0..size.1 {
        for x in 0..size.0 {
            let mut acc = 0.0;
            for j in 0..4 {
                for i in 0..4 {
                    let p = apply(&h, [x as f64 + (i as f64 + 0.5) / 4.0, y as f64 + (j as f64 + 0.5) / 4.0]);
                    acc += match white_at(&c, 8.0, p[0], p[1]) {
                        Some(w) if w != negative => 0.88,
                        Some(_) => 0.07,
                        None => 0.55,
                    };
                }
            }
            px[y * size.0 + x] = (acc / 16.0) as f32;
        }
    }
    Grey { size, px }
}

/// A square of side `s` centred at `c`, turned by `deg`, with a little
/// perspective, in [`CORNER_ORDER`].
fn quad(c: [f64; 2], s: f64, deg: f64) -> [[f64; 2]; 4] {
    let (sn, cs) = deg.to_radians().sin_cos();
    // Tag frame (y up) to image (y down); the top edge a bit shorter.
    let pts = [[-0.47, 0.5], [0.47, 0.5], [0.5, -0.5], [-0.5, -0.5]];
    pts.map(|[x, y]| {
        let (x, y) = (x * s, -y * s);
        [c[0] + cs * x - sn * y, c[1] + sn * x + cs * y]
    })
}

fn worst(a: &[[f64; 2]; 4], b: &[[f64; 2]; 4]) -> f64 {
    (0..4).map(|i| (a[i][0] - b[i][0]).hypot(a[i][1] - b[i][1])).fold(0.0, f64::max)
}

#[test]
fn corners_come_back_in_corner_order_at_any_rotation() {
    assert_eq!(CORNER_ORDER[0], "top-left");
    for deg in [0.0, 90.0, 180.0, 270.0, 33.0] {
        let q = quad([320.0, 240.0], 200.0, deg);
        let tags = detect(&render(7, q, (640, 480), false));
        assert_eq!(tags.len(), 1, "{deg}°");
        assert_eq!(tags[0].id, 7);
        assert!(!tags[0].negative);
        let e = worst(&tags[0].corners, &q);
        assert!(e < 0.3, "{deg}°: corners off by {e} px");
    }
}

#[test]
fn finds_negative_prints() {
    let q = quad([300.0, 260.0], 240.0, 20.0);
    let tags = detect(&render(2, q, (640, 520), true));
    assert_eq!(tags.len(), 1);
    assert_eq!((tags[0].id, tags[0].hamming, tags[0].negative), (2, 0, true));
    assert!(worst(&tags[0].corners, &q) < 0.3);
}

#[test]
fn reports_any_id_and_several_at_once() {
    let n = irys_tags::id_count();
    assert_eq!(n, 587);
    let size = (900, 400);
    let (a, b) = (quad([220.0, 200.0], 220.0, 10.0), quad([660.0, 200.0], 160.0, -25.0));
    let (ia, ib) = (n - 1, 300);
    let (ga, gb) = (render(ia, a, size, false), render(ib, b, size, false));
    // Composite: the right half from the second render.
    let px = (0..size.0 * size.1).map(|k| if k % size.0 < 450 { ga.px[k] } else { gb.px[k] }).collect();
    let tags = detect(&Grey { size, px });
    assert_eq!(tags.iter().map(|t| t.id).collect::<Vec<_>>(), vec![ib, ia]);
}

#[test]
fn finds_small_tags() {
    // ~20 px across: below what kornia finds at 1×.
    let q = quad([80.0, 60.0], 20.0, 15.0);
    let tags = detect(&render(11, q, (160, 120), false));
    assert_eq!(tags.len(), 1, "small tag");
    assert_eq!(tags[0].id, 11);
    assert!(worst(&tags[0].corners, &q) < 0.75);
}

#[test]
fn finds_the_negative_tag_in_phone_photos() {
    // Crops (downscaled) of two iPhone photos of a negative-printed id 2,
    // in display orientation: at full size kornia found this tag only at
    // decimation 6 and 8.
    for f in ["tag36h11_id2_negative_0943.jpg", "tag36h11_id2_negative_0944.jpg"] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(f);
        let img = image::open(&path).unwrap().to_luma8();
        let g = Grey::from_luma8(img.width() as usize, img.height() as usize, img.as_raw());
        let tags = detect(&g);
        assert_eq!(tags.len(), 1, "{f}");
        assert_eq!((tags[0].id, tags[0].hamming, tags[0].negative), (2, 0, true), "{f}");
        assert!(tags[0].side() > 300.0, "{f}");
    }
}
