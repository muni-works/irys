//! The WGSL BSDF is a port of the Rust one. This checks it, number by number.
//!
//! `pathtrace.rs` is the reference and `gpu/shaders/bsdf.wgsl` is the port, and
//! nothing about a rendered image makes a divergence between them obvious — a
//! dropped compensation factor or a sheen table read off by a row just shifts
//! energy a little and still looks like a plausible picture. So the shader is
//! compiled standalone against a harness entry point, driven on real hardware,
//! and compared against the Rust for the same inputs.
//!
//! Three properties:
//!
//! 1. `gpu_bsdf_eval_matches_the_cpu_reference` — `(f·cos, pdf)` agrees across
//!    a sweep of every parameter, the new ones included.
//! 2. `gpu_bsdf_sample_pdf_matches_eval_pdf` — the PDF `bsdf_sample` returns is
//!    the PDF `bsdf_eval` reports for the direction it drew. This is the MIS
//!    invariant; when it breaks the image is energy-wrong and still plausible.
//! 3. `gpu_furnace_closes_on_a_rough_metal` — the device's own estimate of a
//!    white metal's directional albedo, which is what actually exercises the
//!    baked `GGX_E` table through the shader's interpolation rather than the
//!    CPU's.
//!
//! `#[ignore]`d like the other GPU tests; run with
//! `cargo test -p kosm-render --features gpu --test gpu_bsdf -- --ignored`.

#![cfg(all(feature = "gpu", not(target_arch = "wasm32")))]

use bytemuck::{Pod, Zeroable};
use kosm_render::gpu::{GpuContext, GpuMaterial, shaders};
use kosm_render::math::Vec3;
use kosm_render::pathtrace::{Pbr, reference_bsdf_eval};

/// Mirrors `ParityIn` in [`HARNESS`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ParityIn {
    material: GpuMaterial,
    wo: [f32; 4],
    wi: [f32; 4],
    rnd: [f32; 4],
}

/// Mirrors `ParityOut` in [`HARNESS`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct ParityOut {
    /// `(f·cos, pdf)` from `bsdf_eval` at the given `wi`.
    eval: [f32; 4],
    /// `(wi, pdf)` from `bsdf_sample`.
    sampled: [f32; 4],
    /// `(f·cos, pdf)` from re-evaluating at the sampled direction.
    resampled: [f32; 4],
}

/// The compute half: one invocation per input, every entry point the tests use.
const HARNESS: &str = r#"
struct ParityIn {
    material: GpuMaterial,
    wo: vec4<f32>,
    wi: vec4<f32>,
    rnd: vec4<f32>,
}

struct ParityOut {
    eval: vec4<f32>,
    sampled: vec4<f32>,
    resampled: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> parity_in: array<ParityIn>;
@group(0) @binding(1) var<storage, read_write> parity_out: array<ParityOut>;

@compute @workgroup_size(64)
fn parity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= arrayLength(&parity_in) {
        return;
    }
    let p = parity_in[i];
    var o: ParityOut;

    let e = bsdf_eval(p.material, p.wo.xyz, p.wi.xyz);
    o.eval = vec4<f32>(e.value, e.pdf);

    let s = bsdf_sample(p.material, p.wo.xyz, p.rnd.x, p.rnd.y, p.rnd.z);
    if s.ok {
        o.sampled = vec4<f32>(s.wi, s.pdf);
        let r = bsdf_eval(p.material, p.wo.xyz, s.wi);
        o.resampled = vec4<f32>(r.value, r.pdf);
    } else {
        o.sampled = vec4<f32>(0.0);
        o.resampled = vec4<f32>(0.0);
    }
    parity_out[i] = o;
}
"#;

fn ctx_or_skip(name: &str) -> Option<&'static GpuContext> {
    match pollster::block_on(GpuContext::init()) {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("[{name}] skipped: {e}");
            None
        }
    }
}

/// Run the harness over `inputs` and read the results back.
fn run(ctx: &GpuContext, inputs: &[ParityIn]) -> Vec<ParityOut> {
    use wgpu::util::DeviceExt;

    let device = &ctx.device;
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("bsdf parity"),
        // The BSDF alone, not `shaders::compose`: the environment module in
        // the full composition wants texture bindings this harness has no use
        // for, and the point here is to isolate the shading model.
        source: wgpu::ShaderSource::Wgsl(
            format!("{}\n{HARNESS}", shaders::BSDF_SHADER).into(),
        ),
    });

    let out_size = (inputs.len() * std::mem::size_of::<ParityOut>()) as u64;
    let in_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("parity in"),
        contents: bytemuck::cast_slice(inputs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("parity out"),
        size: out_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("parity read"),
        size: out_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("parity layout"),
        entries: &[entry(0, true), entry(1, false)],
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("parity bind"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: in_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: out_buf.as_entire_binding(),
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("parity pipeline layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("parity pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("parity"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(inputs.len().div_ceil(64) as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, out_size);
    ctx.queue.submit(Some(enc.finish()));

    let slice = read_buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().expect("readback buffer did not map");
    let out: Vec<ParityOut> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    read_buf.unmap();
    out
}

/// A sweep over every parameter the model has, one axis at a time plus a
/// couple of everything-at-once materials.
fn materials() -> Vec<Pbr> {
    let base = Pbr {
        base_color: [0.72, 0.55, 0.38],
        roughness: 0.35,
        ..Default::default()
    };
    let mut out = vec![base, Pbr::default()];
    for r in [0.05f32, 0.3, 0.7, 1.0] {
        out.push(Pbr { roughness: r, ..base });
    }
    for v in [0.0f32, 0.3, 0.7, 1.0] {
        out.push(Pbr { metallic: v, ..base });
        out.push(Pbr { diffuse_roughness: v, ..base });
        out.push(Pbr { subsurface: v, diffuse_roughness: 0.5, ..base });
        out.push(Pbr { specular_tint: v, specular: 0.9, ..base });
        out.push(Pbr { sheen: v, sheen_roughness: 0.4, ..base });
        out.push(Pbr { sheen: 0.8, sheen_roughness: v.max(0.05), ..base });
        out.push(Pbr { clearcoat: v, clearcoat_roughness: 0.12, ..base });
        out.push(Pbr { anisotropy: v * 2.0 - 1.0, ..base });
    }
    for s in [0.0f32, 0.25, 0.5, 1.0] {
        out.push(Pbr { specular: s, ..base });
    }
    for ior in [1.0f32, 1.33, 1.52, 2.4] {
        out.push(Pbr { ior, ..base });
    }
    // Everything at once, twice, so cross-terms between the layers are hit.
    out.push(Pbr {
        metallic: 0.4,
        roughness: 0.28,
        diffuse_roughness: 0.7,
        subsurface: 0.35,
        specular: 0.85,
        specular_tint: 0.6,
        sheen: 0.7,
        sheen_color: [0.9, 0.95, 1.0],
        sheen_roughness: 0.45,
        anisotropy: 0.6,
        clearcoat: 0.8,
        clearcoat_roughness: 0.1,
        ..base
    });
    out.push(Pbr {
        metallic: 0.9,
        roughness: 0.8,
        diffuse_roughness: 1.0,
        subsurface: 1.0,
        specular: 0.2,
        specular_tint: 1.0,
        sheen: 1.0,
        sheen_color: [1.0, 0.7, 0.5],
        sheen_roughness: 0.9,
        anisotropy: -0.8,
        clearcoat: 0.4,
        clearcoat_roughness: 0.3,
        ior: 1.7,
        ..base
    });
    out
}

fn directions() -> Vec<Vec3> {
    [
        (0.0, 0.0, 1.0),
        (0.3, 0.15, 0.94),
        (0.6, -0.2, 0.77),
        (0.85, 0.1, 0.52),
        (-0.5, 0.6, 0.62),
        (0.94, 0.2, 0.27),
    ]
    .iter()
    .map(|&(x, y, z)| Vec3::new(x, y, z).normalize())
    .collect()
}

fn v4(v: Vec3) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, 0.0]
}

/// Build the full cross product of materials, view and light directions, with
/// a cheap deterministic sample seed riding along in `rnd`.
fn sweep() -> (Vec<ParityIn>, Vec<(Pbr, Vec3, Vec3)>) {
    let mut ins = Vec::new();
    let mut meta = Vec::new();
    let dirs = directions();
    let mut k = 0u32;
    for m in materials() {
        for &wo in &dirs {
            for &wi in &dirs {
                k = k.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let r = |s: u32| ((k >> s) & 0xFFFF) as f32 / 65536.0;
                ins.push(ParityIn {
                    material: GpuMaterial::from_pbr(m),
                    wo: v4(wo),
                    wi: v4(wi),
                    rnd: [r(0), r(8), r(16), 0.0],
                });
                meta.push((m, wo, wi));
            }
        }
    }
    (ins, meta)
}

#[test]
#[ignore = "requires GPU"]
fn gpu_bsdf_eval_matches_the_cpu_reference() {
    let Some(ctx) = ctx_or_skip("gpu_bsdf_eval_matches_the_cpu_reference") else {
        return;
    };
    let (ins, meta) = sweep();
    let outs = run(ctx, &ins);

    let mut worst = 0.0f32;
    for (o, (m, wo, wi)) in outs.iter().zip(&meta) {
        let (cf, cpdf) = reference_bsdf_eval(m, *wo, *wi);
        for c in 0..3 {
            // The CPU runs f64 and the device f32, and the tables are
            // interpolated on both sides, so this is an f32 tolerance, not
            // bit-equality.
            let scale = cf[c].abs().max(o.eval[c].abs()).max(1e-3);
            let rel = (cf[c] - o.eval[c]).abs() / scale;
            worst = worst.max(rel);
            assert!(
                rel <= 2e-3,
                "value channel {c} differs by {rel}: cpu {cf:?} gpu {:?}\n  {m:?}\n  \
                 wo {wo:?} wi {wi:?}",
                &o.eval[..3]
            );
        }
        let scale = cpdf.abs().max(o.eval[3].abs()).max(1e-3);
        let rel = (cpdf - o.eval[3]).abs() / scale;
        worst = worst.max(rel);
        assert!(
            rel <= 2e-3,
            "pdf differs by {rel}: cpu {cpdf} gpu {}\n  {m:?}\n  wo {wo:?} wi {wi:?}",
            o.eval[3]
        );
    }
    eprintln!("worst relative disagreement over {} cases: {worst:e}", meta.len());
}

#[test]
#[ignore = "requires GPU"]
fn gpu_bsdf_sample_pdf_matches_eval_pdf() {
    let Some(ctx) = ctx_or_skip("gpu_bsdf_sample_pdf_matches_eval_pdf") else {
        return;
    };
    let (ins, meta) = sweep();
    let outs = run(ctx, &ins);
    for (o, (m, wo, _)) in outs.iter().zip(&meta) {
        let sampled = o.sampled[3];
        if sampled <= 0.0 {
            continue; // the sampler rejected this draw
        }
        let evaluated = o.resampled[3];
        assert!(
            (sampled - evaluated).abs() <= 1e-4 * sampled.max(1.0),
            "device PDF mismatch: sampled {sampled}, evaluated {evaluated}\n  {m:?}\n  wo {wo:?}"
        );
    }
}

#[test]
#[ignore = "requires GPU"]
fn gpu_furnace_closes_on_a_rough_metal() {
    let Some(ctx) = ctx_or_skip("gpu_furnace_closes_on_a_rough_metal") else {
        return;
    };
    // A white metal under Turquin's compensation should return every photon
    // regardless of how rough it is. Estimated on the device, so the shader's
    // own GGX_E interpolation is what is on trial.
    for alpha in [0.2f32, 0.5, 1.0] {
        let m = Pbr {
            base_color: [1.0; 3],
            metallic: 1.0,
            roughness: alpha.sqrt(),
            ..Default::default()
        };
        for mu in [1.0f64, 0.7, 0.3] {
            let s = (1.0 - mu * mu).max(0.0).sqrt();
            let wo = Vec3::new(s, 0.0, mu);
            let n = 65_536usize;
            let ins: Vec<ParityIn> = (0..n)
                .map(|i| {
                    // Stratified in the lobe-choice and the two lobe
                    // variates, so the estimate is stable enough for a 1%
                    // bound without a million invocations.
                    let a = (i % 256) as f32 / 256.0 + 1.0 / 512.0;
                    let b = (i / 256) as f32 / 256.0 + 1.0 / 512.0;
                    ParityIn {
                        material: GpuMaterial::from_pbr(m),
                        wo: v4(wo),
                        wi: v4(wo),
                        rnd: [0.9, a, b, 0.0],
                    }
                })
                .collect();
            let outs = run(ctx, &ins);
            let mut sum = 0.0f64;
            for o in &outs {
                if o.sampled[3] > 0.0 {
                    sum += (o.resampled[0] / o.sampled[3]) as f64;
                }
            }
            let albedo = sum / n as f64;
            assert!(
                (0.99..=1.01).contains(&albedo),
                "device albedo {albedo} at alpha {alpha}, mu {mu}"
            );
        }
    }
}
