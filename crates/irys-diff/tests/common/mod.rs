//! The synthetic skin scene the skin tests share.
#![allow(dead_code)]

use irys_core::Point3;
use irys_diff::skin::graph::PARAMS;
use irys_diff::skin::mesh::{Mesh, Texture};
use irys_diff::skin::render::{Camera, Chromophores, Light, Mode, Scene};

/// A 10 mm-radius ball of skin, key and fill.
pub fn ball(size: usize, spp: usize, mode: Mode) -> Scene {
    let mesh = Mesh::uv_sphere(Point3::origin(), 1.0, 24, 48);
    let mut s = Scene::new(
        mesh,
        Texture::constant([0.6, 0.4, 0.33]),
        Texture::constant([0.6, 0.6, 0.6]),
        Texture::constant([0.5, 0.5, 1.0]),
        Chromophores::Uniform,
        10.0,
    );
    s.lights = vec![
        Light::facing(
            Point3::new(-2.5, -3.0, 2.5),
            Point3::origin(),
            2.0,
            2.0,
            3.0,
            [1.0, 0.95, 0.9],
        ),
        Light::facing(
            Point3::new(3.0, -2.0, 0.0),
            Point3::origin(),
            3.0,
            3.0,
            0.6,
            [0.9, 0.95, 1.0],
        ),
        Light::facing(
            Point3::new(1.0, 3.0, 2.0),
            Point3::origin(),
            1.0,
            2.0,
            4.0,
            [1.0, 1.0, 1.0],
        ),
    ];
    s.lights_changed();
    s.env = 0.05;
    s.camera = Camera::look_at(Point3::new(0.0, -6.0, 0.5), Point3::origin(), 22.0, 0.0);
    s.set_size(size, size);
    s.spp = spp;
    s.depth = 4;
    s.mode = mode;
    s
}

/// Uniform chromophores: θ₀, θ₁ are the fractions themselves.
pub const THETA: [f64; PARAMS] = [0.03, 0.02, 1.0, 0.35, 0.6];
