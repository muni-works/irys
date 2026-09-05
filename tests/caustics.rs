//! The caustic pass, held to the one number it cannot fudge: energy.
//!
//! A photon pass has exactly one job that can be checked without a reference
//! image — the light that goes into the glass has to come out of it. These
//! tests set up a case where the answer is analytic (an index-matched sphere,
//! which bends nothing and loses nothing), check the pass against it, and
//! then check that a real glass sphere loses only what Fresnel says it should
//! and concentrates the rest into a spot.

use std::sync::Arc;

use kosm_render::caustics::{self, CausticOptions};
use kosm_render::pathtrace::{
    Camera, Environment, Ground, Object, PathTraceOptions, Pbr, Scene, Sun, render,
    render_with_caustics,
};
use kosm_render::analytic::{Frame, Prim};
use kosm_render::{Analytic, Bvh, Point3, Vec3};

const SPHERE_R: f64 = 1.0;
const FLOOR_Z: f64 = -1.5;

fn white_lambert() -> Pbr {
    Pbr {
        base_color: [1.0, 1.0, 1.0],
        metallic: 0.0,
        roughness: 1.0,
        clearcoat: 0.0,
        specular: 0.0,
        ior: 1.0,
        ..Default::default()
    }
}

/// A glass ball hanging over a white floor, lit from straight overhead.
fn ball_over_floor(ior: f32) -> Scene<Analytic> {
    Scene {
        objects: vec![Object::new(
            Arc::new(Bvh::build(Analytic::sphere(
                Point3::new(0.0, 0.0, 0.0),
                SPHERE_R,
            ))),
            Pbr::glass(ior, 0.0),
        )],
        lights: Vec::new(),
        env: Environment::constant([0.0; 3]),
        sun: Some(Sun::new(Vec3::new(0.0, 0.0, 1.0), 0.01, [1.0, 1.0, 1.0])),
        ground: Some(Ground {
            z: FLOOR_Z,
            material: white_lambert(),
            shadow_catcher: false,
        }),
        splats: None,
    }
}

fn opts(photons: usize, radius: f64) -> CausticOptions {
    CausticOptions {
        photons,
        radius: Some(radius),
        ..Default::default()
    }
}

/// A flat slab of glass at normal incidence has an analytic transmittance:
/// `(1 − F)²`, one Fresnel interface on the way in and one on the way out,
/// which at `n = 1.5` is `0.96² = 0.9216`. The photon pass has to reproduce
/// it — and doing so exercises every part of the pass at once: the emission
/// disc's normalisation and area, the aim at the refractive bounds, the
/// first-hit filter that throws away photons which missed the glass, the walk
/// through two interfaces, and the deposit itself.
///
/// This is the pass's energy conservation, stated as a number rather than as
/// a picture.
#[test]
fn the_photon_pass_conserves_the_energy_that_entered_the_glass() {
    // A 2x2 slab, 0.2 thick, hanging over the floor.
    let slab = Analytic::from_prims(vec![Prim::Box {
        center: Point3::new(0.0, 0.0, 0.0),
        half: Vec3::new(1.0, 1.0, 0.1),
        rot: Frame::identity(),
    }]);
    let scene = Scene {
        objects: vec![Object::new(
            Arc::new(Bvh::build(slab)),
            Pbr::glass(1.5, 0.02),
        )],
        lights: Vec::new(),
        env: Environment::constant([0.0; 3]),
        sun: Some(Sun::new(Vec3::new(0.0, 0.0, 1.0), 0.001, [1.0, 1.0, 1.0])),
        ground: Some(Ground {
            z: -3.0,
            material: white_lambert(),
            shadow_catcher: false,
        }),
        splats: None,
    };
    let map = caustics::trace(&scene, &opts(200_000, 0.05));
    assert!(!map.is_empty(), "no photons landed");

    // The sun is straight down and its irradiance is 1, so the power that
    // entered the glass is the slab's silhouette area.
    let entered = 4.0f32;
    let f = 0.04f32; // Fresnel at normal incidence, n = 1.5.
    let expected = entered * (1.0 - f) * (1.0 - f);
    let deposited = map.deposited_power()[0];
    let err = (deposited - expected).abs() / expected;
    assert!(
        err < 0.05,
        "deposited {deposited} is not within 5% of the {expected} that \
         (1 - F)^2 says should have come through"
    );

    // And the pass did not invent light: it aimed a disc covering the slab's
    // bounds, so it emitted strictly more than it deposited.
    assert!(map.emitted_power()[0] > deposited);
}

/// Real glass loses a few percent to Fresnel at each of the two interfaces
/// and a little more to total internal reflection near the rim — and focuses
/// what is left. Both halves of that, in one test.
#[test]
fn a_glass_ball_focuses_what_fresnel_lets_through() {
    let scene = ball_over_floor(1.5);
    let radius = 0.05;
    let map = caustics::trace(&scene, &opts(400_000, radius));
    assert!(!map.is_empty());

    let entered = (std::f64::consts::PI * SPHERE_R * SPHERE_R) as f32;
    let ratio = map.deposited_power()[0] / entered;
    assert!(
        (0.70..0.98).contains(&ratio),
        "a glass ball transmitted {ratio} of what entered it, which is not a \
         Fresnel loss"
    );

    // Focused: the axis is far brighter than a patch a sphere-radius out.
    let n = Vec3::new(0.0, 0.0, 1.0);
    let axis = map.irradiance(Point3::new(0.0, 0.0, FLOOR_Z), n)[0];
    let off = map.irradiance(Point3::new(2.5, 0.0, FLOOR_Z), n)[0];
    assert!(
        axis > 10.0 * (off + 1e-6),
        "the spot ({axis}) is not brighter than its surroundings ({off})"
    );
    // The sun's own irradiance is 1; a caustic is light gathered from an
    // area much larger than the spot, so it must beat that by a lot.
    assert!(axis > 5.0, "the caustic ({axis}) is not a caustic");
}

/// Away from the spot, the caustic map must change nothing. The photon pass
/// adds the refracted share and only the refracted share; a floor that never
/// saw the glass renders exactly as it did.
#[test]
fn the_floor_far_from_the_spot_is_unchanged() {
    let scene = ball_over_floor(1.5);
    let map = caustics::trace(&scene, &opts(100_000, 0.05));

    // Look at a patch of floor well outside anything the ball can reach.
    let cam = Camera::look_at(
        Point3::new(12.0, 0.0, 2.0),
        Point3::new(12.0, 0.0, FLOOR_Z),
        Vec3::new(0.0, 1.0, 0.0),
        20.0,
    );
    let o = PathTraceOptions {
        spp: 32,
        max_depth: 2,
        denoise: false,
        ..Default::default()
    };
    let plain = render(&scene, &cam, 16, 16, &o);
    let lit = render_with_caustics(&scene, &cam, 16, 16, &o, Some(&map));
    for i in 0..plain.rgb.len() {
        assert!(
            (plain.rgb[i] - lit.rgb[i]).abs() <= 1e-6,
            "the caustic map changed a pixel it could not reach"
        );
    }
}

/// Not an assertion — a printout, so the numbers behind the two tests above
/// are visible with `--nocapture` rather than only in a commit message.
#[test]
fn report_the_energy_numbers() {
    let scene = ball_over_floor(1.5);
    let map = caustics::trace(&scene, &opts(400_000, 0.05));
    let entered = (std::f64::consts::PI * SPHERE_R * SPHERE_R) as f32;
    println!(
        "ball: emitted {:.4}, entered {:.4}, deposited {:.4} ({:.1}% of what entered), \
         {} photons, axis irradiance {:.3}",
        map.emitted_power()[0],
        entered,
        map.deposited_power()[0],
        100.0 * map.deposited_power()[0] / entered,
        map.len(),
        map.irradiance(Point3::new(0.0, 0.0, FLOOR_Z), Vec3::new(0.0, 0.0, 1.0))[0],
    );
}
