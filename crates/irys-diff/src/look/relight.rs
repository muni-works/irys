//! Relighting and re-materialling a built scene in place: the light and
//! the substances change, the geometry (BLAS, TLAS, triangles) does not.
//! What the image-to-CAD compare loop (irys-recon-net, v3) fits per scene:
//! the lighting basis renders and the per-object OpenPBR colour and
//! roughness are each one buffer write, not a rebuild.

use super::scene::{Equirect, Material, Sun};
use super::scene_gpu::{GpuLook, pack_env, pack_material};

impl GpuLook {
    /// Replace every material (same count and order as the scene was built
    /// with; no portal skies). Emitting materials must not change: their
    /// light samples are baked at build time.
    pub fn set_materials(&mut self, mats: &[Material]) {
        let rows: Vec<[f32; 4]> = mats.iter().flat_map(|m| pack_material(m, None)).collect();
        let ctx = self.m1.ctx;
        assert!(rows.len() * 16 <= self.m1.mats.size() as usize, "more materials than the scene was built with");
        ctx.queue.write_buffer(&self.m1.mats, 0, bytemuck::cast_slice(&rows));
    }

    /// Replace the sun's direction, size and radiance. The light-group
    /// shares stay those of the build (an estimator choice, not a bias), so
    /// a scene meant to be relit should be built with a sun present.
    pub fn set_sun(&mut self, s: &Sun) {
        let omega = std::f64::consts::TAU * (1.0 - s.radius.cos());
        let h = super::openpbr::hats(s.radiance);
        self.sun = ([s.dir[0] as f32, s.dir[1] as f32, s.dir[2] as f32, s.radius.cos() as f32], [h[0] as f32, h[1] as f32, h[2] as f32, omega as f32]);
    }

    /// Replace the world (same texel size as the build's; its importance
    /// tables are rebuilt).
    pub fn set_env(&mut self, e: &Equirect) {
        let (texels, word) = pack_env(e);
        assert_eq!(word, self.env_word(), "the world's size changed");
        assert!(texels.len() * 16 <= self.b_env.size() as usize);
        self.m1.ctx.queue.write_buffer(&self.b_env, 0, bytemuck::cast_slice(&texels));
    }
}
