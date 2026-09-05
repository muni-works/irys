//! WGSL for the renderer. Geometry is the client's; see
//! [`super::geometry`] for the contract the two halves meet on.

/// The shared BSDF: the single shading model used by the GPU path tracer and,
/// as a port, by the CPU renderer in [`crate::pathtrace`].
///
/// Composed first, so the BSDF exists in exactly one place in the codebase.
/// It also defines `PI` and the `GpuMaterial` struct, since those are part of
/// the shading contract.
pub const BSDF_SHADER: &str = include_str!("bsdf.wgsl");

/// What both halves of a composed shader share: `RayHit`, the face-index
/// sentinels, `MAX_T`/`EPSILON`, the scale-aware ray epsilon, `intersect_aabb`
/// and `shading_frame`.
pub const PRELUDE_SHADER: &str = include_str!("prelude.wgsl");

/// Lat-long HDR environment: nearest-texel lookup, CDF importance sampling and
/// the solid-angle PDF, ported from `pathtrace::EnvMap`.
pub const ENV_SHADER: &str = include_str!("env.wgsl");

/// The built-in analytic geometry module: spheres and planes at binding 1.
/// See [`super::analytic`].
pub const ANALYTIC_SHADER: &str = include_str!("analytic.wgsl");

/// The integrator. Not valid WGSL on its own — see [`compose`].
pub const INTEGRATOR_SHADER: &str = include_str!("integrator.wgsl");

/// Device-side per-pixel history and à-trous denoise.
///
/// Self-contained — it shades nothing, so unlike the others it needs no
/// prefix.
pub const HISTORY_SHADER: &str = include_str!("history.wgsl");

/// The learned denoiser: three convolutions, a softmax and a 5x5 apply,
/// standing where the à-trous chain stands. See [`super::neural`].
pub const NEURAL_SHADER: &str = include_str!("neural.wgsl");

/// The gradient-directed sample budget: which pixels this frame's rays go to.
///
/// A fragment, not a module — it reads [`HISTORY_SHADER`]'s bindings and its
/// `params`, and is compiled behind it by [`history_shader`].
pub const BUDGET_SHADER: &str = include_str!("budget.wgsl");

/// The history module the device-side passes are compiled from: the history
/// and denoise passes, then the budget passes that share their bindings.
pub fn history_shader() -> String {
    format!("{HISTORY_SHADER}\n{BUDGET_SHADER}")
}

/// Put the renderer's prelude in front of a body, and the BSDF in front of
/// that. For shader harnesses that shade but do not trace.
pub fn compose(body: &str) -> String {
    compose_with("", body)
}

/// [`compose`] with a client's own WGSL spliced in after the prelude — what a
/// client's parity harness wants when it drives its geometry's own functions.
pub fn compose_with(client_wgsl: &str, body: &str) -> String {
    format!("{BSDF_SHADER}\n{PRELUDE_SHADER}\n{client_wgsl}\n{ENV_SHADER}\n{body}")
}

/// The full trace shader: BSDF, prelude, the client's geometry module, the
/// environment, then the integrator.
pub fn trace_shader(geometry_wgsl: &str) -> String {
    format!("{BSDF_SHADER}\n{PRELUDE_SHADER}\n{geometry_wgsl}\n{ENV_SHADER}\n{INTEGRATOR_SHADER}")
}
