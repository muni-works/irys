//! M1: the OpenCV pinhole (docs/belief-codec.md, M1 item 5).
//!
//! Intrinsics `fx, fy, cx, cy` with pixel centres at integer coordinates,
//! x right, y down, z forward, as a ROS `camera_info.json` gives them, and a
//! camera → world pose as a `poses.jsonl` line gives it. It becomes a kernel
//! [`Camera`] with no shader change: the principal point and the focal
//! lengths live in the unnormalised `forward`, `right` and `up`.

use super::scene::Camera;

/// OpenCV pinhole intrinsics plus a camera → world pose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pinhole {
    /// Image width, pixels.
    pub width: usize,
    /// Image height, pixels.
    pub height: usize,
    /// Focal length along x, pixels.
    pub fx: f64,
    /// Focal length along y, pixels.
    pub fy: f64,
    /// Principal point x, pixels (pixel `(0, 0)`'s centre is `(0, 0)`).
    pub cx: f64,
    /// Principal point y, pixels.
    pub cy: f64,
    /// Camera → world rotation, row-major; its columns are the optical
    /// frame's axes (x right, y down, z forward) in the world.
    pub rot: [[f64; 3]; 3],
    /// Camera centre in the world.
    pub pos: [f64; 3],
}

impl Pinhole {
    /// From a ROS `camera_info.json` (`width`, `height`, `k`; a non-zero
    /// distortion is refused) at the identity pose.
    pub fn from_camera_info(json: &str) -> Result<Self, String> {
        let v: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let num = |x: &serde_json::Value| x.as_f64().ok_or("camera_info: not a number");
        let k: Vec<f64> = v["k"].as_array().ok_or("camera_info: no k")?.iter().map(num).collect::<Result<_, _>>()?;
        if k.len() != 9 {
            return Err("camera_info: k is not 3 × 3".into());
        }
        if let Some(d) = v["d"].as_array() {
            if d.iter().any(|x| x.as_f64().unwrap_or(1.0) != 0.0) {
                return Err("camera_info: distortion is not supported".into());
            }
        }
        Ok(Self {
            width: v["width"].as_u64().ok_or("camera_info: no width")? as usize,
            height: v["height"].as_u64().ok_or("camera_info: no height")? as usize,
            fx: k[0],
            fy: k[4],
            cx: k[2],
            cy: k[5],
            rot: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            pos: [0.0; 3],
        })
    }

    /// The same intrinsics at another pose.
    pub fn at(&self, rot: [[f64; 3]; 3], pos: [f64; 3]) -> Self {
        Self { rot, pos, ..*self }
    }

    /// The kernel camera that traces exactly these pixels (`tan_half` = 1;
    /// scene.wgsl's pixel `(i, j)` + jitter `(½, ½)` lands on OpenCV `(i, j)`).
    pub fn camera(&self) -> Camera {
        let col = |k: usize| [self.rot[0][k], self.rot[1][k], self.rot[2][k]];
        let (x, y, z) = (col(0), col(1), col(2));
        let (w, h) = (self.width as f64, self.height as f64);
        let ax = (w / 2.0 - 0.5 - self.cx) / self.fx;
        let ay = (h / 2.0 - 0.5 - self.cy) / self.fy;
        Camera {
            eye: self.pos,
            forward: std::array::from_fn(|a| z[a] + x[a] * ax + y[a] * ay),
            right: x.map(|v| v * h / (2.0 * self.fx)),
            up: y.map(|v| -v * h / (2.0 * self.fy)),
            tan_half: 1.0,
        }
    }

    /// OpenCV projection of a world point: `(u, v, z)` with `z` the optical
    /// depth (`None` behind the camera).
    pub fn project(&self, p: [f64; 3]) -> Option<[f64; 3]> {
        let d = [p[0] - self.pos[0], p[1] - self.pos[1], p[2] - self.pos[2]];
        let c: [f64; 3] = std::array::from_fn(|k| (0..3).map(|a| self.rot[a][k] * d[a]).sum());
        (c[2] > 0.0).then(|| [self.fx * c[0] / c[2] + self.cx, self.fy * c[1] / c[2] + self.cy, c[2]])
    }

    /// The unit world ray through pixel coordinates `(u, v)`.
    pub fn ray(&self, u: f64, v: f64) -> [f64; 3] {
        let c = [(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0];
        let d: [f64; 3] = std::array::from_fn(|a| (0..3).map(|k| self.rot[a][k] * c[k]).sum());
        let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        d.map(|x| x / n)
    }

    /// `z / distance` along pixel `(u, v)`'s ray: multiply a [`super::sensor::aov::DIST`]
    /// value by it for the optical (OpenCV) depth.
    pub fn z_per_dist(&self, u: f64, v: f64) -> f64 {
        let a = (u - self.cx) / self.fx;
        let b = (v - self.cy) / self.fy;
        1.0 / (1.0 + a * a + b * b).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k1() -> Pinhole {
        Pinhole::from_camera_info(
            r#"{"width": 544, "height": 448, "k": [204.64, 0.0, 277.81, 0.0, 204.64, 221.25, 0.0, 0.0, 1.0], "d": [0,0,0,0,0], "distortion_model": "plumb_bob"}"#,
        )
        .unwrap()
        .at([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]], [1.0, 2.0, 0.87])
    }

    #[test]
    fn camera_info_and_projection() {
        let k = k1();
        assert_eq!((k.width, k.height, k.fx, k.cx, k.cy), (544, 448, 204.64, 277.81, 221.25));
        // the principal point projects to itself, along the optical axis (world +x here)
        let c = k.project([3.0, 2.0, 0.87]).unwrap();
        assert!((c[0] - 277.81).abs() < 1e-9 && (c[1] - 221.25).abs() < 1e-9 && (c[2] - 2.0).abs() < 1e-12);
        // world +z (up) is image −v
        let up = k.project([3.0, 2.0, 1.87]).unwrap();
        assert!((up[1] - (221.25 - 204.64 * 0.5)).abs() < 1e-9);
        // world +y (left of a camera facing +x) is image −u
        let left = k.project([3.0, 3.0, 0.87]).unwrap();
        assert!((left[0] - (277.81 - 204.64 * 0.5)).abs() < 1e-9);
        assert!(k.project([0.0, 2.0, 0.87]).is_none());
        assert!(Pinhole::from_camera_info(r#"{"width": 4, "height": 4, "k": [1,0,0,0,1,0,0,0,1], "d": [0.1,0,0,0,0]}"#).is_err());
    }

    #[test]
    fn the_kernel_camera_traces_opencv_pixels() {
        let k = k1();
        let cam = k.camera();
        // scene.wgsl: pixel (i, j) + jitter (½, ½) → fwd + right·sx + up·sy
        let ray = |u: f64, v: f64| {
            let (w, h) = (k.width as f64, k.height as f64);
            let sx = (2.0 * (u + 0.5) / w - 1.0) * cam.tan_half * w / h;
            let sy = (1.0 - 2.0 * (v + 0.5) / h) * cam.tan_half;
            let r: [f64; 3] = std::array::from_fn(|a| cam.forward[a] + cam.right[a] * sx + cam.up[a] * sy);
            let l = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
            r.map(|x| x / l)
        };
        for (u, v) in [(0.0, 0.0), (543.0, 447.0), (277.81, 221.25), (100.0, 400.0)] {
            let (a, b) = (ray(u, v), k.ray(u, v));
            assert!((0..3).all(|i| (a[i] - b[i]).abs() < 1e-12), "{u},{v}: {a:?} {b:?}");
            let p: [f64; 3] = std::array::from_fn(|i| k.pos[i] + 3.0 * b[i]);
            let q = k.project(p).unwrap();
            assert!((q[0] - u).abs() < 1e-9 && (q[1] - v).abs() < 1e-9);
            assert!((q[2] - 3.0 * k.z_per_dist(u, v)).abs() < 1e-9);
        }
    }
}
