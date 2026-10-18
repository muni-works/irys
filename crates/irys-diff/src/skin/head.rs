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
    // Map-SPEC is relative: scale it so its mean is 0.7 of skin's own F0
    // (k = 1 everywhere reads as oiled under softboxes this large).
    s.spec_gain = 0.7 / asset.spec.mean()[0];
    frame(&mut s, view);
    s
}

/// Set lights and camera for `view`: a large warm key, a dim cool fill, a
/// strip rim behind, all softboxes out of frame, a dark grey studio around.
pub fn frame(s: &mut Scene, view: View) {
    let c = Point3::new(HEAD[0], HEAD[1], HEAD[2]);
    let at = |x: f64, y: f64, z: f64| Point3::new(c.x + x, c.y + y, c.z + z);
    let (key, fill, rim, eye, target) = match view {
        View::Front => (
            at(-7.0, -8.0, 6.0),
            at(9.0, -8.0, 0.0),
            at(8.5, 6.0, 3.5),
            at(0.0, -30.0, 0.2),
            at(0.0, -2.0, -0.35),
        ),
        View::ThreeQuarter => (
            at(8.0, -8.5, 5.5),
            at(-10.0, -6.0, 0.5),
            at(-8.0, 6.5, 3.0),
            at(-16.0, -24.5, 0.6),
            at(-0.5, -1.3, -0.3),
        ),
    };
    s.lights = vec![
        Light::facing(key, at(0.0, -1.0, 0.3), 7.0, 7.0, 6.0, WARM),
        Light::facing(fill, at(0.0, -1.0, 0.0), 10.0, 10.0, 0.2, COOL),
        Light::facing(rim, at(0.0, 0.0, 0.6), 1.6, 8.0, 9.0, [1.0, 0.97, 0.92]),
    ];
    s.camera = Camera::look_at(eye, target, 11.0, 0.05);
    s.env = 0.006;
    s.depth = 5;
    s.lights_changed();
}

/// Warm key: about 4500 K on the three hats.
pub const WARM: [f64; 3] = [1.0, 0.92, 0.80];
/// Cool fill.
pub const COOL: [f64; 3] = [0.82, 0.90, 1.0];

/// The comparison strip's framing at `size`: close on the cheek, the ear
/// and the nose from three-quarter left, with a backlight behind the ear —
/// where light through thin tissue shows what a surface model cannot.
pub fn compare_frame(s: &mut Scene, size: (usize, usize)) {
    let c = Point3::new(HEAD[0], HEAD[1], HEAD[2]);
    let at = |x: f64, y: f64, z: f64| Point3::new(c.x + x, c.y + y, c.z + z);
    s.lights = vec![
        Light::facing(
            at(-5.0, -9.0, 6.0),
            at(-0.5, -1.0, 0.3),
            6.0,
            6.0,
            4.0,
            WARM,
        ),
        Light::facing(
            at(8.0, -8.0, 0.0),
            at(0.0, -1.0, 0.0),
            10.0,
            10.0,
            0.15,
            COOL,
        ),
        Light::facing(
            at(-3.5, 5.5, 0.8),
            at(-1.6, 0.2, 0.4),
            2.5,
            3.0,
            25.0,
            [1.0, 0.95, 0.88],
        ),
    ];
    s.camera = Camera::look_at(at(-17.0, -21.0, 0.8), at(-0.8, -0.9, -0.2), 11.0, 0.04);
    s.env = 0.006;
    s.depth = 5;
    s.set_size(size.0, size.1);
    s.lights_changed();
}
