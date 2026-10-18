// `strands.wgsl`'s interface without fibres: a kernel that can trace strands
// compiles against these when the scene has none (no ray-query binding).

struct SHit {
    t: f32,
    seg: u32,
    n: vec3<f32>,
}

fn strand_hit(o: vec3<f32>, d: vec3<f32>, tmin: f32, tmax: f32, any: bool) -> SHit {
    return SHit(tmax, 0xffffffffu, vec3<f32>(0.0));
}

fn strand_tangent(i: u32) -> vec3<f32> {
    return vec3<f32>(1.0, 0.0, 0.0);
}

fn strand_radius(i: u32) -> f32 {
    return 0.0;
}

fn strand_melanin(i: u32) -> f32 {
    return 1.0;
}
