//! The renderer's half of the bind group has to leave room for a client's.
//!
//! WebGPU's `maxStorageBuffersPerShaderStage` is **10** in Chrome, and a
//! composed trace shader sits exactly at that limit: five bindings for the
//! renderer, five for the geometry module. Native Metal allows far more, so a
//! regression here passes every GPU test on this machine while the browser
//! rejects the bind group layout and paints nothing. That happened once; this
//! is the guard.
//!
//! If the renderer needs more data, it goes in a texture (48 slots), as the
//! HDR environment does — not in an eleventh storage buffer.
#![cfg(feature = "gpu")]

use kosm_render::gpu::geometry::MAX_GEOMETRY_BINDINGS;
use kosm_render::gpu::shaders;

const BROWSER_LIMIT: usize = 10;

fn storage_bindings(src: &str) -> usize {
    src.lines()
        .filter(|l| {
            let t = l.trim_start();
            !t.starts_with("//") && t.contains("var<storage")
        })
        .count()
}

#[test]
fn the_renderer_leaves_five_storage_bindings_for_geometry() {
    let renderer = storage_bindings(&shaders::trace_shader(""));
    assert_eq!(
        renderer + MAX_GEOMETRY_BINDINGS,
        BROWSER_LIMIT,
        "the renderer declares {renderer} storage buffers and promises clients \
         {MAX_GEOMETRY_BINDINGS} more, which is not the {BROWSER_LIMIT} a \
         browser guarantees. Either move the renderer's new data into a \
         texture or lower MAX_GEOMETRY_BINDINGS — and if you lower it, say so \
         in `gpu::geometry`, because a client built against the old number \
         will be rejected by WebGPU with the pipeline valid and the viewport \
         blank."
    );
}

#[test]
fn a_full_client_still_fits() {
    let src = shaders::trace_shader(&kosm_render::gpu::AnalyticGeometry::module().wgsl);
    assert!(storage_bindings(&src) <= BROWSER_LIMIT);
}

#[test]
fn the_analytic_module_declares_one_binding() {
    let m = kosm_render::gpu::AnalyticGeometry::module();
    assert_eq!(m.layout.len(), 1);
    assert_eq!(m.layout[0].binding, 1);
}
