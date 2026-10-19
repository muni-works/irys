//! M1: TLAS-only rebuilds (docs/belief-codec.md, M1 item 4).

use std::time::Instant;

use irys_gpu::wgpu;

use super::scene::Instance;
use super::scene_gpu::GpuLook;

/// Why [`GpuLook::set_instances`] refused.
#[derive(Debug, Clone, PartialEq)]
pub enum InstanceError {
    /// More placements than the TLAS was built with.
    Capacity {
        /// Placements asked for.
        asked: usize,
        /// Slots the TLAS was built with.
        cap: usize,
    },
    /// A placement names a mesh the scene does not have.
    Mesh(usize),
    /// An emitting placement moved or changed: its light samples are baked
    /// in world space at build time.
    Emitter(usize),
}

impl std::fmt::Display for InstanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl GpuLook {
    /// The placements the TLAS holds now.
    pub fn instances(&self) -> &[Instance] {
        &self.m1.instances
    }

    /// Move, add or drop placements (up to the count the scene was built
    /// with) by rebuilding the TLAS only: no BLAS, vertex, material or
    /// texture is touched; the instance table (16 B each) is rewritten.
    /// Emitting placements must stay as they were. Returns the wall time,
    /// seconds, including the wait that orders the build before any trace.
    pub fn set_instances(&mut self, instances: &[Instance]) -> Result<f64, InstanceError> {
        let t0 = Instant::now();
        let s = &mut self.m1;
        if instances.len() > s.cap {
            return Err(InstanceError::Capacity { asked: instances.len(), cap: s.cap });
        }
        for (k, i) in instances.iter().enumerate() {
            if i.mesh >= s.blas.len() {
                return Err(InstanceError::Mesh(i.mesh));
            }
            let was = s.emit_base.get(k).copied().unwrap_or(u32::MAX) != u32::MAX;
            let emits_now = s.emit_base.iter().zip(&s.instances).any(|(&e, o)| e != u32::MAX && o.mesh == i.mesh);
            if was || emits_now {
                let same = s.instances.get(k).is_some_and(|o| o.mesh == i.mesh && o.xf == i.xf && o.mask == i.mask);
                if !same {
                    return Err(InstanceError::Emitter(k));
                }
            }
        }
        let ctx = s.ctx;
        let rows: Vec<[u32; 4]> = instances
            .iter()
            .enumerate()
            .map(|(k, i)| [s.first_tri[i.mesh], s.mesh_mat[i.mesh], s.emit_base.get(k).copied().unwrap_or(u32::MAX), 0])
            .collect();
        if !rows.is_empty() {
            ctx.queue.write_buffer(&s.inst, 0, bytemuck::cast_slice(&rows));
        }
        for k in 0..s.cap {
            s.tlas[k] = None;
        }
        for (k, i) in instances.iter().enumerate() {
            if let Some(b) = &s.blas[i.mesh] {
                let x = &i.xf;
                let xf = [x[0][0], x[0][1], x[0][2], x[0][3], x[1][0], x[1][1], x[1][2], x[1][3], x[2][0], x[2][1], x[2][2], x[2][3]];
                s.tlas[k] = Some(wgpu::TlasInstance::new(b, xf, k as u32, i.mask as u8));
            }
        }
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.build_acceleration_structures(std::iter::empty(), [&s.tlas]);
        ctx.queue.submit([enc.finish()]);
        // The build is waited on before anything traces: wgpu 30's Metal
        // backend places no barrier between an acceleration-structure build
        // and the work that reads it (design.md, "Speed", device facts).
        ctx.device.poll(wgpu::PollType::wait_indefinitely()).expect("device lost");
        s.instances = instances.to_vec();
        Ok(t0.elapsed().as_secs_f64())
    }
}
