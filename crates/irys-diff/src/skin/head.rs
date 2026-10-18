//! The Lee Perry-Smith head as a portrait: the scan turned `+z` up, its maps,
//! a key / fill / rim of softboxes, and a thin-lens camera.
//!
//! The asset is "Lee Perry-Smith" by Infinite-Realities (ir-ltd.net),
//! CC BY 3.0 (creativecommons.org/licenses/by/3.0), via the three.js
//! examples; it lives outside the repository ([`super::mesh::asset_dir`]).

use irys_core::{Point3, Vec3};

use super::mesh::{Asset, Mesh};
use super::render::{Camera, Chromophores, Light, Scene};

/// Millimetres per unit of the scan. The file is a bust (8.55 units is the
/// shoulders); the head is 4.3 units from nose tip to the back of the skull
/// and 3.6 across the ears, an adult's ≈ 215 mm and ≈ 175 mm.
pub const MM_PER_UNIT: f64 = 50.0;

/// The middle of the head (not the bust), scene units.
pub const HEAD: [f64; 3] = [-0.08, -0.3, 1.9];

/// glTF is `+y` up, `+z` toward the viewer; the scene is `+z` up, the face
/// looking down `−y`.
pub fn to_scene(m: &mut Mesh) {
    let f = |p: Point3| Point3::new(p.x, -p.z, p.y);
    let g = |v: Vec3| Vec3::new(v.x, -v.z, v.y);
    m.transform(f, g);
}

/// Which portrait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    /// Straight on, key from camera left.
    Front,
    /// Three-quarter, key from camera right, a strong rim from behind left.
    ThreeQuarter,
}

/// The head, lit and framed.
pub fn scene(asset: &Asset, chromo: Chromophores, view: View) -> Scene {
    let mut mesh = asset.mesh.clone();
    to_scene(&mut mesh);
    let mut s = Scene::new(
        mesh,
        asset.albedo.clone(),
        asset.spec.clone(),
        asset.normal.clone(),
        chromo,
        MM_PER_UNIT,
    );
    frame(&mut s, view);
    s
}

/// Set lights and camera for `view`.
pub fn frame(s: &mut Scene, view: View) {
    let c = Point3::new(HEAD[0], HEAD[1], HEAD[2]);
    let warm = [1.0, 0.93, 0.84];
    let cool = [0.85, 0.92, 1.0];
    let at = |x: f64, y: f64, z: f64| Point3::new(c.x + x, c.y + y, c.z + z);
    match view {
        View::Front => {
            s.lights = vec![
                Light::facing(at(-6.5, -8.5, 4.5), at(0.0, -1.0, 0.3), 6.0, 6.0, 2.4, warm),
                Light::facing(
                    at(9.0, -7.0, -0.5),
                    at(0.0, -1.0, 0.0),
                    8.0,
                    8.0,
                    0.35,
                    cool,
                ),
                Light::facing(
                    at(5.0, 7.0, 4.0),
                    at(0.0, 0.0, 0.5),
                    2.0,
                    7.0,
                    5.0,
                    [1.0, 0.97, 0.92],
                ),
            ];
            s.camera = Camera::look_at(at(0.0, -30.0, 0.4), at(0.0, -2.0, 0.1), 10.5, 0.06);
        }
        View::ThreeQuarter => {
            s.lights = vec![
                Light::facing(at(8.0, -7.5, 4.5), at(0.0, -1.0, 0.3), 6.0, 6.0, 2.4, warm),
                Light::facing(
                    at(-9.0, -6.0, 0.0),
                    at(0.0, -1.0, 0.0),
                    8.0,
                    8.0,
                    0.25,
                    cool,
                ),
                Light::facing(
                    at(-6.0, 7.0, 4.0),
                    at(0.0, 0.0, 0.5),
                    2.0,
                    7.0,
                    7.0,
                    [1.0, 0.97, 0.92],
                ),
            ];
            s.camera = Camera::look_at(at(-16.0, -24.0, 0.8), at(-0.4, -1.4, 0.2), 10.5, 0.06);
        }
    }
    s.env = 0.03;
    s.lights_changed();
}
