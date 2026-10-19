//! M1: the path tracer over N cameras in one submit (docs/belief-codec.md,
//! M1 item 3). The sensor kernel ([`super::sensor::Sensor`]) batches the AOVs.

use std::time::Instant;

use irys_gpu::wgpu;

use super::scene::Camera;
use super::scene_gpu::{Frame, GpuLook, Opts, RING};
use super::sensor::map_read;

impl GpuLook {
    /// Encode one frame of `cam` into `enc`: clear the accumulators, then
    /// every sample and chunk, each dispatch with its own ring slot
    /// (from `*slot` on). The accumulator holds the frame afterwards.
    pub(crate) fn encode_frame(&self, enc: &mut wgpu::CommandEncoder, cam: &Camera, o: &Opts, slot: &mut usize) {
        let pixels = self.pixels();
        let chunk = super::scene_gpu::CHUNK;
        let k = o.regen.max(1);
        assert!(*slot + o.spp.div_ceil(k) as usize * pixels.div_ceil(chunk) <= RING, "more dispatches than the ring holds");
        enc.clear_buffer(self.accum_buf(), 0, None);
        enc.clear_buffer(self.spec_buf(), 0, None);
        // render()'s schedule (k samples per dispatch, path regeneration),
        // without guiding: a guide is rebuilt between dispatches.
        for s in (0..o.spp).step_by(k as usize) {
            let ns = k.min(o.spp - s);
            for first in (0..pixels).step_by(chunk) {
                let n = (pixels - first).min(chunk);
                let p = self.params_bytes(cam, o, s, ns, first as u32);
                let at = (*slot * self.slot_bytes()) as u64;
                self.ctx().queue.write_buffer(self.params_buf(), at, &p);
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
                pass.set_pipeline(self.trace_pipe());
                pass.set_bind_group(0, self.bind_group(), &[at as u32]);
                pass.dispatch_workgroups(n.div_ceil(64) as u32, 1, 1);
                drop(pass);
                *slot += 1;
            }
        }
    }

    /// [`GpuLook::render`] (guiding off) for every camera in `cams`: every sample of every
    /// camera encoded in one submit (at most [`RING`] dispatches each), one
    /// wait, one readback. The spectrum is read back only when `spectrum`.
    pub fn render_batch(&self, cams: &[Camera], o: &Opts, spectrum: bool) -> Vec<Frame> {
        let (dev, queue) = (&self.ctx().device, &self.ctx().queue);
        let pixels = self.pixels();
        let chunk = super::scene_gpu::CHUNK;
        let per_cam = o.spp.div_ceil(o.regen.max(1)) as usize * pixels.div_ceil(chunk);
        assert!(per_cam <= RING, "one camera needs more dispatches than the ring holds");
        let groups = pixels.div_ceil(64);
        let bins = super::scene_gpu::BINS;
        let acc_b = pixels as u64 * 16;
        let spec_b = (groups * bins) as u64 * 4;
        let per_b = acc_b + if spectrum { spec_b } else { 0 };
        let out = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("look batch"),
            size: (per_b * cams.len() as u64).max(16),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let t0 = Instant::now();
        let mut slot = 0usize;
        let mut enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for (c, cam) in cams.iter().enumerate() {
            if slot + per_cam > RING {
                // the ring is full: flush (one more submit and wait)
                queue.submit([enc.finish()]);
                dev.poll(wgpu::PollType::wait_indefinitely()).expect("device lost");
                enc = dev.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                slot = 0;
            }
            self.encode_frame(&mut enc, cam, o, &mut slot);
            let base = c as u64 * per_b;
            enc.copy_buffer_to_buffer(self.accum_buf(), 0, &out, base, acc_b);
            if spectrum {
                enc.copy_buffer_to_buffer(self.spec_buf(), 0, &out, base + acc_b, spec_b);
            }
        }
        queue.submit([enc.finish()]);
        let bytes = map_read(self.ctx(), &out);
        let seconds = t0.elapsed().as_secs_f64();
        let inv = 1.0 / o.spp.max(1) as f32;
        (0..cams.len())
            .map(|c| {
                let b = &bytes[c * per_b as usize..(c + 1) * per_b as usize];
                let acc: &[f32] = bytemuck::cast_slice(&b[..acc_b as usize]);
                let rgb = acc.chunks_exact(4).flat_map(|p| [p[0] * inv, p[1] * inv, p[2] * inv]).collect();
                let spectrum = if spectrum {
                    let sp: &[f32] = bytemuck::cast_slice(&b[acc_b as usize..]);
                    let per_bin = (pixels as f64 * o.spp as f64 * 4.0) / bins as f64;
                    (0..bins).map(|k| (0..groups).map(|g| sp[g * bins + k] as f64).sum::<f64>() / per_bin).collect()
                } else {
                    Vec::new()
                };
                Frame { rgb, spectrum, seconds: seconds / cams.len() as f64, stats: None }
            })
            .collect()
    }
}
