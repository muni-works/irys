//! Shared fixtures for the tier's unit tests.

#![cfg(test)]

use super::*;
#[allow(unused_imports)]
use crate::geometry::TriMesh;

pub(crate) fn cube_mesh() -> TriMesh {
    let p = |x, y, z| Point3::new(x, y, z);
    let positions = vec![
        p(0.0, 0.0, 0.0),
        p(10.0, 0.0, 0.0),
        p(10.0, 10.0, 0.0),
        p(0.0, 10.0, 0.0),
        p(0.0, 0.0, 10.0),
        p(10.0, 0.0, 10.0),
        p(10.0, 10.0, 10.0),
        p(0.0, 10.0, 10.0),
    ];
    let indices = [
        0, 2, 1, 0, 3, 2, // -z
        4, 5, 6, 4, 6, 7, // +z
        0, 1, 5, 0, 5, 4, // -y
        3, 7, 6, 3, 6, 2, // +y
        0, 4, 7, 0, 7, 3, // -x
        1, 2, 6, 1, 6, 5, // +x
    ];
    TriMesh::new(positions, Vec::new(), &indices)
}

pub(crate) fn test_scene() -> Scene<TriMesh> {
    Scene {
        objects: vec![Object::new(
            Arc::new(Bvh::build(cube_mesh())),
            Pbr::plastic([0.8, 0.3, 0.2], 0.35, 0.0),
        )],
        lights: studio_rig(Point3::new(5.0, 5.0, 5.0), 9.0),
        env: Environment::default(),
        sun: None,
        ground: None,
        splats: None,
    }
}

/// The old estimator: shadow-ray every light, each with its own
/// area-sampling PDF. Kept here as the reference the one-light-per-bounce
/// importance sampler must match in expectation.
pub(crate) fn sample_all_lights_reference(
    scene: &Scene<TriMesh>,
    accel: &SceneAccel<TriMesh>,
    p: Point3,
    frame: &Frame,
    wo_local: Vec3,
    m: &Pbr,
    eta: f32,
    rng: &mut Rng,
) -> [f32; 3] {
    let Frame { t, b, n } = *frame;
    let mut sum = [0.0f32; 3];
    for light in &scene.lights {
        let lp = light.sample(rng.f64(), rng.f64());
        let to_light = lp - p;
        let dist = to_light.norm();
        if dist < 1e-9 {
            continue;
        }
        let wi_world = to_light / dist;
        let cos_light = -wi_world.dot(light.normal());
        if cos_light <= 1e-9 {
            continue;
        }
        let wi_local = to_local(t, b, n, wi_world);
        if wi_local.z <= 0.0 {
            continue;
        }
        let (f, bsdf_pdf) = bsdf_eval(m, wo_local, wi_local, eta, 0.0);
        if max3(f) <= 0.0 {
            continue;
        }
        let light_pdf = (dist * dist / (cos_light * light.area())) as f32;
        if !light_pdf.is_finite() || light_pdf <= 0.0 {
            continue;
        }
        if scene.occluded(accel, p + n * 1e-5, wi_world, dist) {
            continue;
        }
        // The reference's MIS partner must be *its* own light pdf, so
        // this is the old weighting verbatim.
        let w = power_heuristic(light_pdf, bsdf_pdf);
        sum = add3(sum, scale3(mul3(f, light.emission), w / light_pdf));
    }
    sum
}

/// Both estimators are MIS-weighted, and the two weightings differ per
/// sample (the pick probability enters the light pdf). What must agree is
/// the *total* direct-lighting estimate — NEE plus the BSDF-sampled hits
/// on emitters — so this compares the unweighted NEE integral by driving
/// both with `power_heuristic` replaced by 1: i.e. the plain estimator
/// `f * Le * cos / pdf`, which is what unbiasedness is about.
pub(crate) fn nee_unweighted_mean(
    scene: &Scene<TriMesh>,
    accel: &SceneAccel<TriMesh>,
    pick_one: bool,
    n: usize,
) -> [f64; 3] {
    let p = Point3::new(0.0, 0.0, 0.0);
    let nrm = Vec3::new(0.0, 0.0, 1.0);
    let frame = shading_frame(nrm, None);
    let wo_world = Vec3::new(0.3, 0.2, 0.9).normalize();
    let wo_local = to_local(frame.t, frame.b, nrm, wo_world);
    let m = Pbr {
        base_color: [0.8, 0.7, 0.6],
        roughness: 0.6,
        ..Default::default()
    };
    let mut rng = Rng::new(0xA11CE);
    let mut sum = [0.0f64; 3];
    for _ in 0..n {
        let est = if pick_one {
            let Some((i, pick_pdf)) = accel.pick_light(rng.f64() as f32) else {
                continue;
            };
            one_light_unweighted(
                &scene.lights[i],
                pick_pdf,
                p,
                &frame,
                wo_local,
                &m,
                1.0,
                &mut rng,
            )
        } else {
            let mut acc = [0.0f32; 3];
            for light in &scene.lights {
                acc = add3(
                    acc,
                    one_light_unweighted(light, 1.0, p, &frame, wo_local, &m, 1.0, &mut rng),
                );
            }
            acc
        };
        for c in 0..3 {
            sum[c] += est[c] as f64;
        }
    }
    [sum[0] / n as f64, sum[1] / n as f64, sum[2] / n as f64]
}

pub(crate) fn one_light_unweighted(
    light: &AreaLight,
    pick_pdf: f32,
    p: Point3,
    frame: &Frame,
    wo_local: Vec3,
    m: &Pbr,
    eta: f32,
    rng: &mut Rng,
) -> [f32; 3] {
    let Frame { t, b, n } = *frame;
    let lp = light.sample(rng.f64(), rng.f64());
    let to_light = lp - p;
    let dist = to_light.norm();
    if dist < 1e-9 {
        return [0.0; 3];
    }
    let wi_world = to_light / dist;
    let cos_light = -wi_world.dot(light.normal());
    if cos_light <= 1e-9 {
        return [0.0; 3];
    }
    let wi_local = to_local(t, b, n, wi_world);
    if wi_local.z <= 0.0 {
        return [0.0; 3];
    }
    let (f, _) = bsdf_eval(m, wo_local, wi_local, eta, 0.0);
    let pdf = pick_pdf * (dist * dist / (cos_light * light.area())) as f32;
    if !pdf.is_finite() || pdf <= 0.0 {
        return [0.0; 3];
    }
    scale3(mul3(f, light.emission), 1.0 / pdf)
}

pub(crate) fn open_scene(lights: Vec<AreaLight>) -> Scene<TriMesh> {
    Scene {
        objects: Vec::new(),
        lights,
        env: Environment::default(),
        sun: None,
        ground: None,
        splats: None,
    }
}

pub(crate) fn panel(center: Point3, emission: [f32; 3], half: f64) -> AreaLight {
    // Faces -Z, i.e. down at the origin.
    AreaLight {
        center,
        u: Vec3::new(half, 0.0, 0.0),
        v: Vec3::new(0.0, -half, 0.0),
        emission,
    }
}

/// An axis-aligned quad in the z = `z` plane, spanning ±`half` in x and y.
pub(crate) fn pane_mesh(z: f64, half: f64) -> TriMesh {
    let p = |x, y| Point3::new(x, y, z);
    let positions = vec![
        p(-half, -half),
        p(half, -half),
        p(half, half),
        p(-half, half),
    ];
    TriMesh::new(positions, Vec::new(), &[0, 1, 2, 0, 2, 3])
}

/// A smooth thin-walled pane of ordinary window glass.
pub(crate) fn window_glass() -> Pbr {
    Pbr {
        transmission: 1.0,
        thin_walled: true,
        roughness: 0.0,
        ior: 1.5,
        ..Pbr::glass(1.5, 0.0)
    }
}

/// Mean NEE estimate at the origin on a white Lambertian floor facing +z.
pub(crate) fn nee_mean(scene: &Scene<TriMesh>, n: usize) -> f64 {
    let accel = SceneAccel::build(scene);
    let m = Pbr {
        base_color: [1.0; 3],
        metallic: 0.0,
        roughness: 1.0,
        ..Pbr::default()
    };
    let frame = shading_frame(Vec3::new(0.0, 0.0, 1.0), None);
    let wo_local = Vec3::new(0.0, 0.0, 1.0);
    let mut rng = Rng::new(0x9e3779b97f4a7c15);
    let mut sum = 0.0f64;
    for _ in 0..n {
        let e = scene.sample_lights(
            &accel,
            Point3::new(0.0, 0.0, 0.0),
            &frame,
            wo_local,
            &m,
            1.0,
            0.0,
            &mut rng,
        );
        sum += luminance(e) as f64;
    }
    sum / n as f64
}

/// Nothing about an opaque scene changed. The shadow ray that walks
/// through sheets has to agree, hit for hit, with the any-hit traversal
/// it replaced wherever there are no sheets to walk through — which is
/// what makes every render that predates panes bit-identical.

pub(crate) fn test_camera() -> Camera {
    Camera::look_at(
        Point3::new(30.0, -34.0, 24.0),
        Point3::new(5.0, 5.0, 5.0),
        Vec3::new(0.0, 0.0, 1.0),
        32.0,
    )
}
