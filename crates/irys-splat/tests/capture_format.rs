//! The ipse CAPTURE.md export layout (`<capture>.train/`) loads as is.

use irys_splat::data::{self, DepthSpec};

#[test]
fn reads_capture_md_train_dir() {
    let d = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("fixture.train");
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(d.join("images")).unwrap();
    std::fs::create_dir_all(d.join("depth")).unwrap();
    let (w, h) = (8u32, 6u32);
    std::fs::write(d.join("camera_info.json"), format!(
        r#"{{"width": {w}, "height": {h}, "k": [5.0, 0.0, 3.5, 0.0, 5.0, 2.5, 0.0, 0.0, 1.0], "d": [0,0,0,0,0], "distortion_model": "plumb_bob", "frame_id": "head_color_optical_frame"}}"#)).unwrap();
    std::fs::write(d.join("align.json"), r#"{"scale": 1.0}"#).unwrap();
    let mut lines = String::new();
    for f in [1234u32, 1240] {
        let px: Vec<u8> = (0..w * h * 3).map(|i| (i * 7 % 255) as u8).collect();
        image::save_buffer(d.join(format!("images/{f:06}.png")), &px, w, h, image::ColorType::Rgb8).unwrap();
        let dep: Vec<u8> = (0..w * h).flat_map(|i| (1000u16 + i as u16).to_le_bytes()).collect();
        let depth = if f == 1234 {
            std::fs::write(d.join(format!("depth/{f:06}.u16")), dep).unwrap();
            format!(r#", "depth": "depth/{f:06}.u16""#)
        } else {
            String::new()
        };
        lines += &format!(
            r#"{{"frame": {f}, "t": 1786642516.94, "image": "images/{f:06}.png"{depth}, "cam_pos": [1.0, 2.0, 0.5], "cam_rot": [1, 0, 0, 0, 0, 1, 0, -1, 0]}}"#
        );
        lines.push('\n');
    }
    std::fs::write(d.join("poses.jsonl"), lines).unwrap();
    // --train DIR: images, poses and camera from the dir; depth from the lines
    let v = data::load(&d, &d.join("poses.jsonl"), &d.join("camera_info.json"), Some(&d), DepthSpec::default()).unwrap();
    assert_eq!(v.len(), 2);
    assert_eq!(v[0].cam.pos, [1.0, 2.0, 0.5]);
    assert_eq!(v[0].cam.rot, [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]);
    assert_eq!((v[0].cam.fx, v[0].cam.cx, v[0].cam.width), (5.0, 3.5, 8));
    assert_eq!(v[0].rgb[1], [21.0 / 255.0, 28.0 / 255.0, 35.0 / 255.0]);
    let dz = v[0].depth.as_ref().unwrap();
    assert!((dz[0] - 1.0).abs() < 1e-6 && (dz[5] - 1.005).abs() < 1e-6);
    assert!(v[1].depth.is_none());
    // the same with --images pointing at images/ (names still resolve)
    let v2 = data::load(&d.join("images"), &d.join("poses.jsonl"), &d.join("camera_info.json"), None, DepthSpec::default()).unwrap();
    assert_eq!(v2[1].rgb, v[1].rgb);
}
