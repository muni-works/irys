//! Gradient-directed sampling: where the frame's rays go.
//!
//! A game spends its samples uniformly. One ray per pixel per frame, whether
//! the pixel is a wall that has looked the same for four hundred frames or the
//! ball that crossed it on this one. The wall's ray buys nothing — its mean is
//! already inside a display bit — and the ball's one ray is not nearly enough.
//!
//! And the renderer *knows*, before it traces anything, which is which:
//!
//! * **the physics.** Every instance's transform this frame is already on the
//!   device, packed for the reprojection as [`super::InstanceMotion`]. From it
//!   and the previous frame's depth plane, the exact screen-space displacement
//!   of the surface under each pixel — a motion vector, not an estimate.
//! * **the history.** Each pixel carries its own sample count and the variance
//!   of its own mean. sigma/sqrt(n) is the error bar another sample would buy
//!   down, and it is per pixel.
//! * **the image.** The last raw sample against the running mean is the same
//!   disagreement the neighbourhood clamp measures — and it is the only thing
//!   that sees lighting going stale under a surface that did not move.
//!
//! [`RayTracePipeline::budget_frame`] turns those three into a per-pixel
//! budget `b(p)`, in samples, normalised so the frame's total is
//! [`SampleBudget::rays_per_frame`]. Directing samples never spends more of
//! them; it moves them from the wall to the ball.
//!
//! # Deterministic or stochastic
//!
//! The natural implementation is the deterministic one: `accumulate` loops
//! `b(p)` samples at pixel `p`. That needs the *trace* to loop per pixel, and
//! the trace's entry point takes one sample per invocation per dispatch —
//! changing that is a change to `integrator.wgsl`, which this does not touch.
//!
//! So the implementation here is the stochastic one. The host dispatches
//! [`SampleBudget::rounds`] trace rounds; on round `r` pixel `p` folds its
//! sample with probability `b(p)/rounds`, and so takes `b(p)` samples in
//! expectation. The coin is a hash of `(pixel, frame, round)` and has nothing
//! to do with what the sample turned out to be, so the mean over the folded
//! samples is an unbiased estimate of the pixel — no `1/p` reweighting is
//! needed, and a reweighting would only add variance. What a skipped pixel
//! keeps is its history, exactly.
//!
//! The accounting that follows is therefore in **samples folded**, which is
//! what `rays_per_frame` bounds. The rays *dispatched* are still one per pixel
//! per round: a per-pixel skip inside the trace is one line in the integrator
//! and would make the two numbers the same.
//!
//! # The knobs
//!
//! [`SampleBudget::bias`] is the whole of it. Zero is a flat weight field and
//! therefore today's uniform spend, exactly; one is the directed field; in
//! between is the linear blend, because the normalisation is linear in the
//! weight. [`SampleBudget::floor_k`] is the guarantee that nothing starves: a
//! pixel gets at least one sample every `floor_k` frames whatever its budget
//! says, on a phase of its own so the cost is spread rather than periodic.

use super::context::{GpuContext, GpuError};
use super::history::{
    BudgetFields, HistoryParams, HistoryPipeline, NO_BUDGET, PARAM_STRIDE, dispatch,
    history_bind_group, view_basis,
};
use super::pipeline::{RayTracePipeline, read_back_f32};
use super::resident::ResidentScene;
use super::wgpu;
use super::{GpuCamera, GpuDenoiseParams, GpuRenderState, InstanceMotion};

/// The parameter slot the budget passes drive.
///
/// Slot 0 belongs to `accumulate`, which is dispatched once per round with a
/// different round index and cannot share one. The budget passes run once at
/// the head of the frame and want a slot that no round rewrites; the à-trous
/// slots are free until the denoise call, which runs after every round.
const BUDGET_SLOT: u32 = 1;

/// How many times the budget's total is re-summed and rescaled; see
/// [`RayTracePipeline::budget_frame`].
const RESCALE_ROUNDS: usize = 3;

/// How a frame's rays are to be spent.
///
/// The defaults are a working directed budget at the renderer's own filter
/// footprint: pass `bias: 0.0` for the uniform spend the renderer had before
/// any of this, and `rays_per_frame: (w * h) as f32` for the same *number* of
/// samples a uniform frame would have folded.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleBudget {
    /// 0 spends the frame's samples uniformly — one per pixel when
    /// `rays_per_frame` is the pixel count, which is what the renderer did
    /// before — and 1 spends them entirely where the budget says. In between
    /// is the linear blend of the two weight fields.
    pub bias: f32,
    /// The frame's total sample budget, in samples folded. `sum(b(p))` lands
    /// on this to within a fraction of a percent.
    pub rays_per_frame: f32,
    /// How many trace rounds the host will dispatch this frame, and so the
    /// most samples any one pixel may take. Four lets a moving silhouette be
    /// four times better served than the frame average; past about eight the
    /// per-round dispatch overhead is the frame.
    pub rounds: u32,
    /// How far the motion drive is dilated, in pixels: a moved object drags
    /// its shadow, its bounce and everything the à-trous filter will reach
    /// for. The filter's own footprint, `2^iters`, is the right number.
    pub radius: u32,
    /// Every pixel is guaranteed one sample once every this many frames,
    /// whatever its budget. This is the hard starvation floor; the soft one
    /// is a share of the weight every pixel keeps (see `budget.wgsl`).
    /// 1 guarantees every pixel every frame, which is no budget at all.
    pub floor_k: u32,
    /// Trace every pixel on every round and fold only the selected ones —
    /// what a budgeted frame did before the trace learned to skip.
    ///
    /// The control arm, and nothing else: it places exactly the same samples
    /// the skipping path does and pays `rounds` full frames of rays to do it.
    /// `the_skip_folds_the_same_samples` runs the two against each other.
    pub trace_all: bool,
}

impl Default for SampleBudget {
    fn default() -> Self {
        Self {
            bias: 1.0,
            rays_per_frame: 0.0,
            rounds: 4,
            radius: 32,
            floor_k: 16,
            trace_all: false,
        }
    }
}

impl SampleBudget {
    /// The uniform spend: one sample per pixel per frame, in one round.
    ///
    /// Byte for byte what a caller who never asked for a budget gets, and the
    /// control arm of every comparison.
    pub fn uniform(width: u32, height: u32) -> Self {
        Self {
            bias: 0.0,
            rays_per_frame: (width as f32) * (height as f32),
            rounds: 1,
            radius: 0,
            floor_k: 1,
            trace_all: false,
        }
    }

    /// A directed budget spending the same number of samples a uniform frame
    /// would, over `rounds` rounds.
    pub fn directed(width: u32, height: u32, bias: f32, rounds: u32) -> Self {
        Self {
            bias,
            rays_per_frame: (width as f32) * (height as f32),
            rounds: rounds.max(1),
            ..Default::default()
        }
    }

    fn fields(&self, round: u32, frame: u32) -> BudgetFields {
        BudgetFields {
            enabled: 1,
            bias: self.bias.clamp(0.0, 1.0),
            rounds: self.rounds.max(1),
            round,
            rays_per_frame: self.rays_per_frame.max(0.0),
            radius: self.radius,
            floor_k: self.floor_k.max(1),
            frame,
        }
    }
}

/// The per-pixel budget read back off the device, for tests and for a tuner.
#[derive(Debug, Clone)]
pub struct Budget {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// `b(p)`: how many of this frame's samples pixel `p` is to get, in
    /// expectation. Sums to [`SampleBudget::rays_per_frame`].
    pub samples: Vec<f32>,
    /// The dilated motion drive, 0..1 — what the physics said.
    pub motion: Vec<f32>,
    /// The history drive, 0..1 — the relative error bar and the short-history
    /// term.
    pub history: Vec<f32>,
    /// The image drive, 0..1 — the last sample's disagreement with the mean.
    pub image: Vec<f32>,
    /// Which rounds each pixel takes: bit `r` set when the pixel folds — and
    /// so is traced on — round `r`. What `budget_select` wrote and what both
    /// the trace and the fold read.
    pub rounds: Vec<u32>,
    /// The guide depth each pixel's primary hit left behind: distance from
    /// the eye, 0 for background. Read back beside the budget because it is
    /// the thing a skipped pixel still owes — see [`FLAG_BUDGET_GUIDES`].
    ///
    /// [`FLAG_BUDGET_GUIDES`]: super::FLAG_BUDGET_GUIDES
    pub depth: Vec<f32>,
}

impl Budget {
    /// The frame's total budget, in samples.
    pub fn total(&self) -> f64 {
        self.samples.iter().map(|&b| b as f64).sum()
    }
}

impl RayTracePipeline {
    /// Compute this frame's per-pixel sample budget, before tracing it.
    ///
    /// Everything the passes read is already on the device and already true:
    /// the guide planes and the raw sample are the *previous* frame's, and
    /// `motion` is *this* frame's, written out of the same transforms the host
    /// just posed the scene with. That is what "before tracing" means — a
    /// budget computed from this frame's sample would arrive a pass too late
    /// to spend it.
    ///
    /// Call once at the head of a frame, then
    /// [`RayTracePipeline::accumulate_resident_round`] once per round in
    /// `0..budget.rounds`, then
    /// [`RayTracePipeline::denoise_and_resolve_resident`] once.
    pub fn budget_frame(
        &self,
        ctx: &GpuContext,
        history_pipeline: &HistoryPipeline,
        res: &mut ResidentScene,
        camera: &GpuCamera,
        budget: &SampleBudget,
        motion: Option<&InstanceMotion>,
        frame: u32,
    ) -> Result<(), GpuError> {
        let (w, h) = res.size();
        res.ensure_history(ctx, w, h);

        // The motion table is the reprojection's, uploaded the same way: the
        // budget wants the displacement it encodes and the reprojection wants
        // the transform, and they are the same bytes.
        let (motion_ids, motion_instances) = match motion {
            Some(m) if m.instances() > 0 => {
                let hist = res.history_mut().expect("history was just ensured");
                hist.upload_motion(ctx, m);
                (m.ids(), m.instances())
            }
            _ => (0, 0),
        };

        let cur = view_basis(camera);
        let bud = budget.fields(0, frame);
        {
            let hist = res.history().expect("history was just ensured");
            let p = HistoryParams {
                width: w,
                height: h,
                count_cutoff: 1,
                iters: 0,
                sigma_lum: 0.0,
                sigma_depth: 0.0,
                sigma_normal: 0.0,
                exposure: 1.0,
                stride: 1,
                src_is_b: 0,
                scissor_xy: 0,
                scissor_wh: 0,
                cur_eye: cur.0,
                cur_right: cur.1,
                cur_up: cur.2,
                cur_forward: cur.3,
                prev_eye: cur.0,
                prev_right: cur.1,
                prev_up: cur.2,
                prev_forward: cur.3,
                view_params: [cur.4, cur.5, cur.4, cur.5],
                reprojected: 0,
                iter_index: 0,
                origin_x: 0,
                origin_y: 0,
                history_cap: 1,
                clamp_k: 0.0,
                clamp_reset: 1,
                motion_instances,
                motion_ids,
                spatial_variance: 0,
                budget_enabled: bud.enabled,
                budget_bias: bud.bias,
                budget_rounds: bud.rounds,
                budget_round: bud.round,
                rays_per_frame: bud.rays_per_frame,
                budget_radius: bud.radius,
                budget_floor_k: bud.floor_k,
                budget_frame: bud.frame,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
                _pad3: 0,
                _pad4: 0,
                _pad5: 0,
            };
            ctx.queue.write_buffer(
                &hist.params_buffer(),
                PARAM_STRIDE * BUDGET_SLOT as u64,
                bytemuck::bytes_of(&p),
            );
        }

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sample Budget Encoder"),
            });

        let (raw, guides) = res.raw_and_guide_buffers();
        let hist = res.history().expect("history was just ensured");
        // Both sums start each frame at zero; the passes only ever add.
        encoder.clear_buffer(hist.budget_buffers().1, 0, None);

        let group = history_bind_group(
            ctx,
            history_pipeline,
            hist,
            raw,
            guides,
            &hist.scratch_a_buffer(),
            &hist.scratch_b_buffer(),
            None,
            "Sample Budget Bind Group",
        );
        let groups = (w.div_ceil(8), h.div_ceil(8));
        for (pipeline, label) in [
            (&history_pipeline.budget_weight, "Budget Weight"),
            (&history_pipeline.budget_blur_x, "Budget Dilate X"),
            (&history_pipeline.budget_blur_y, "Budget Dilate Y"),
            (&history_pipeline.budget_normalize, "Budget Normalize"),
        ] {
            dispatch(&mut encoder, pipeline, &group, BUDGET_SLOT, groups, label);
        }
        // The normalisation is exact in the weights and inexact in the
        // samples, because `budget_normalize` then clamps a pixel to the
        // rounds the host will actually dispatch and raises a floored pixel to
        // one. Both move the total. Each rescale pass sums what was really
        // assigned and scales by the ratio; the clamps bite less each time, so
        // three rounds put the total inside a tenth of a percent of the target
        // where one leaves it a couple of percent out.
        for _ in 0..RESCALE_ROUNDS {
            encoder.clear_buffer(hist.budget_buffers().1, 4, Some(4));
            dispatch(
                &mut encoder,
                &history_pipeline.budget_assigned,
                &group,
                BUDGET_SLOT,
                groups,
                "Budget Assigned",
            );
            dispatch(
                &mut encoder,
                &history_pipeline.budget_rescale,
                &group,
                BUDGET_SLOT,
                groups,
                "Budget Rescale",
            );
        }
        // Last: turn b(p) into the set of rounds each pixel takes, and write
        // it where the trace can read it. Everything above is a weight field;
        // this is the decision, and it is made here rather than in the fold so
        // that a pixel which will not fold this round is never traced.
        dispatch(
            &mut encoder,
            &history_pipeline.budget_select,
            &group,
            BUDGET_SLOT,
            groups,
            "Budget Select",
        );
        ctx.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// One of a budgeted frame's trace rounds.
    ///
    /// [`RayTracePipeline::accumulate_resident_temporal`] with the budget
    /// gate on: the trace covers the frame (or the state's scissor box) as it
    /// always did, and the fold takes pixel `p`'s sample with probability
    /// `b(p)/rounds`. Pass `prev_view` and `motion` on **round 0 only**, for
    /// the same reason the unbudgeted call wants them on the first box only:
    /// the reprojection gathers over the whole frame and a later round would
    /// re-gather from a history it has already folded into.
    #[allow(clippy::too_many_arguments)]
    pub fn accumulate_resident_round(
        &self,
        ctx: &GpuContext,
        history_pipeline: &HistoryPipeline,
        res: &mut ResidentScene,
        camera: &GpuCamera,
        state: GpuRenderState,
        keep: &[u8],
        prev_view: Option<&GpuCamera>,
        denoise: &GpuDenoiseParams,
        motion: Option<&InstanceMotion>,
        budget: &SampleBudget,
        round: u32,
        frame: u32,
    ) -> Result<(), GpuError> {
        let bud = if budget.bias <= 0.0 && budget.rounds <= 1 {
            // Nothing to gate: one round at zero bias is the uniform spend,
            // and skipping the selection keeps it bit-identical to the old
            // path.
            NO_BUDGET
        } else {
            budget.fields(round, frame)
        };
        // Tell the trace about the selection. Round 0 asks the pixels it skips
        // for their guide planes anyway: the reprojection runs behind that
        // round and reads every pixel's guides, folded or not.
        let mut state = state;
        state.set_budget_mask(bud.enabled != 0 && !budget.trace_all, round, round == 0);
        self.accumulate_resident_inner(
            ctx,
            history_pipeline,
            res,
            camera,
            state,
            keep,
            prev_view,
            denoise,
            motion,
            bud,
        )
    }

    /// Read this frame's budget back.
    ///
    /// For tests and for a tuner; the render path never needs it. Returns
    /// `None` if no pass has built a history for this scene yet.
    pub async fn read_budget(
        &self,
        ctx: &GpuContext,
        res: &mut ResidentScene,
    ) -> Result<Option<Budget>, GpuError> {
        let Some((w, h)) = res.history().map(|hi| hi.size()) else {
            return Ok(None);
        };
        let plane = (w as u64) * (h as u64) * 16;
        // The budget itself, then guide planes 1 and 3 — the depth a pass
        // left behind, and the selection mask that decided which passes it
        // was.
        let bytes = plane * 3;
        res.history_mut()
            .expect("just checked")
            .ensure_budget_readback(ctx, bytes);

        let guides = res.raw_and_guide_buffers().1;
        let hist = res.history().expect("just checked");
        let staging = hist.budget_readback_buffer().expect("just ensured");
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Sample Budget Readback Encoder"),
            });
        encoder.copy_buffer_to_buffer(hist.budget_buffers().0, 0, staging, 0, plane);
        encoder.copy_buffer_to_buffer(guides, plane, staging, plane, plane);
        encoder.copy_buffer_to_buffer(guides, plane * 3, staging, plane * 2, plane);
        ctx.queue.submit(Some(encoder.finish()));

        let raw = read_back_f32(ctx, staging).await?;
        let n = (w as usize) * (h as usize);
        let mut out = Budget {
            width: w,
            height: h,
            samples: vec![0.0; n],
            motion: vec![0.0; n],
            history: vec![0.0; n],
            image: vec![0.0; n],
            rounds: vec![0; n],
            depth: vec![0.0; n],
        };
        for i in 0..n {
            out.motion[i] = raw[i * 4];
            out.history[i] = raw[i * 4 + 1];
            out.image[i] = raw[i * 4 + 2];
            out.samples[i] = raw[i * 4 + 3];
            out.depth[i] = raw[(n + i) * 4 + 3];
            out.rounds[i] = raw[(2 * n + i) * 4].to_bits();
        }
        Ok(Some(out))
    }
}
