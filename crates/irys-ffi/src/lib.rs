//! irys over a C ABI, rendering into an IOSurface with no copy (M1 item 7).
//!
//! The finding: wgpu 30 can do it. `wgpu::Device::as_hal::<Metal>()` hands
//! back the `MTLDevice`; `newTextureWithDescriptor:iosurface:plane:` makes an
//! `MTLTexture` whose storage *is* the IOSurface; `hal::metal::Device::
//! texture_from_raw` + `wgpu::Device::create_texture_from_hal` wrap it as an
//! ordinary `wgpu::Texture`. The path tracer accumulates in its own buffer
//! and one compute pass ([`irys_diff::look::present::Present`]) writes the
//! sRGB frame into that texture as a storage texture. Swift / Metal then
//! presents the IOSurface (a `CALayer.contents`, or its own `MTLTexture` of
//! the same surface). Nothing crosses the CPU.
//!
//! `include/irys.h` is the header; `harness/` is a Swift program that makes
//! an IOSurface, renders into it through this ABI, and checks its pixels
//! against the readback path.
//!
//! Every function returns 0 / non-null on success; [`irys_last_error`] has
//! the message otherwise. Handles are not thread-safe; use one per thread.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char};
use std::path::Path;
use std::ptr::NonNull;

use irys_diff::look::openpbr::OpenPbr;
use irys_diff::look::pinhole::Pinhole;
use irys_diff::look::present::Present;
use irys_diff::look::scene::{Camera, LookScene, Material};
use irys_diff::look::scene_gpu::{GpuLook, Opts};
use irys_gpu::{GpuContext, wgpu};

thread_local! {
    static ERR: RefCell<CString> = RefCell::new(CString::default());
}

fn fail(msg: impl Into<String>) -> i32 {
    let m = msg.into().replace('\0', " ");
    ERR.with(|e| *e.borrow_mut() = CString::new(m).unwrap_or_default());
    -1
}

/// The message of the last failure on this thread (valid until the next call).
#[unsafe(no_mangle)]
pub extern "C" fn irys_last_error() -> *const c_char {
    ERR.with(|e| e.borrow().as_ptr())
}

/// The device.
pub struct IrysContext {
    ctx: &'static GpuContext,
}

/// One scene on the device, its camera and its IOSurface targets.
pub struct IrysScene {
    look: GpuLook,
    camera: Camera,
    targets: HashMap<usize, (wgpu::Texture, Present)>,
}

/// The process's GPU (shared by every context); null without one, or
/// without hardware ray queries.
#[unsafe(no_mangle)]
pub extern "C" fn irys_context_create() -> *mut IrysContext {
    match irys_diff::gpu::context() {
        Ok(ctx) if ctx.device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) => Box::into_raw(Box::new(IrysContext { ctx })),
        Ok(_) => {
            fail("the GPU has no ray queries");
            std::ptr::null_mut()
        }
        Err(e) => {
            fail(format!("{e}"));
            std::ptr::null_mut()
        }
    }
}

/// Free a context (the device itself lives for the process).
///
/// # Safety
/// `c` is null or from [`irys_context_create`], freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn irys_context_destroy(c: *mut IrysContext) {
    if !c.is_null() {
        drop(unsafe { Box::from_raw(c) });
    }
}

fn opt_path<'a>(p: *const c_char) -> Result<Option<&'a Path>, String> {
    if p.is_null() {
        return Ok(None);
    }
    let s = unsafe { CStr::from_ptr(p) }.to_str().map_err(|e| e.to_string())?;
    Ok(Some(Path::new(s)))
}

/// Load a scene: a Blender export directory (`look::scene::load`; null for
/// none), a 3DGS `.ply` (null for none), at `width × height`. With neither,
/// a demo scene (four spheres under a white world).
///
/// # Safety
/// `c` from [`irys_context_create`]; the paths null or NUL-terminated UTF-8.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn irys_scene_load(
    c: *const IrysContext,
    export_dir: *const c_char,
    splat_ply: *const c_char,
    width: u32,
    height: u32,
) -> *mut IrysScene {
    let Some(c) = (unsafe { c.as_ref() }) else {
        fail("null context");
        return std::ptr::null_mut();
    };
    let r = (|| -> Result<IrysScene, String> {
        let size = (width as usize, height as usize);
        let (dir, ply) = (opt_path(export_dir)?, opt_path(splat_ply)?);
        let mut scene = match dir {
            Some(d) => {
                let mut s = irys_diff::look::scene::load(d, Some(size.0))?;
                s.size = size;
                s
            }
            None if ply.is_some() => {
                let mut s = LookScene::spheres(vec![], 0.0, size);
                s.env = None;
                s
            }
            None => LookScene::spheres(
                (0..4)
                    .map(|k| {
                        Material::new("demo", OpenPbr { base_color: [0.8, 0.25 * k as f64, 0.2], specular_roughness: 0.3, ..OpenPbr::default() })
                    })
                    .collect(),
                1.0,
                size,
            ),
        };
        scene.size = size;
        let cloud = ply.map(|p| irys_light::PlyCloud::load(p)).transpose()?;
        let look = GpuLook::with_splats(c.ctx, &scene, cloud.as_ref()).ok_or("no ray queries")?;
        Ok(IrysScene { look, camera: scene.camera, targets: HashMap::new() })
    })();
    match r {
        Ok(s) => Box::into_raw(Box::new(s)),
        Err(e) => {
            fail(e);
            std::ptr::null_mut()
        }
    }
}

/// Free a scene and its IOSurface wrappers (the surfaces are the caller's).
///
/// # Safety
/// `s` null or from [`irys_scene_load`], freed once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn irys_scene_destroy(s: *mut IrysScene) {
    if !s.is_null() {
        drop(unsafe { Box::from_raw(s) });
    }
}

/// Set the camera: OpenCV pinhole intrinsics (pixels) and the camera →
/// world pose (`rot` row-major, columns = optical x right, y down, z
/// forward; `pos` the centre).
///
/// # Safety
/// `s` from [`irys_scene_load`]; `rot` 9 doubles, `pos` 3.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn irys_scene_set_camera(s: *mut IrysScene, fx: f64, fy: f64, cx: f64, cy: f64, rot: *const f64, pos: *const f64) -> i32 {
    let Some(s) = (unsafe { s.as_mut() }) else { return fail("null scene") };
    if rot.is_null() || pos.is_null() {
        return fail("null pose");
    }
    let r = unsafe { std::slice::from_raw_parts(rot, 9) };
    let p = unsafe { std::slice::from_raw_parts(pos, 3) };
    let (w, h) = s.look.size();
    s.camera = Pinhole {
        width: w,
        height: h,
        fx,
        fy,
        cx,
        cy,
        rot: std::array::from_fn(|i| std::array::from_fn(|j| r[3 * i + j])),
        pos: [p[0], p[1], p[2]],
    }
    .camera();
    0
}

#[cfg(target_vendor = "apple")]
mod surface {
    use super::*;
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_io_surface::IOSurfaceRef;
    use objc2_metal::{MTLDevice, MTLPixelFormat, MTLStorageMode, MTLTexture, MTLTextureDescriptor, MTLTextureType, MTLTextureUsage};

    /// An `Rgba8Unorm` storage texture whose storage is `surface`.
    pub fn texture(ctx: &GpuContext, surface: &IOSurfaceRef) -> Result<wgpu::Texture, String> {
        let (w, h) = (surface.width(), surface.height());
        let fmt = surface.pixel_format();
        // 'RGBA' (kCVPixelFormatType_32RGBA)
        if fmt != u32::from_be_bytes(*b"RGBA") || surface.bytes_per_element() != 4 {
            return Err(format!("the IOSurface must be 'RGBA', 4 bytes per element (is {fmt:#x})"));
        }
        let hal = unsafe { ctx.device.as_hal::<wgpu::hal::api::Metal>() }.ok_or("not a Metal device")?;
        let dev: &ProtocolObject<dyn MTLDevice> = hal.raw_device();
        let desc = unsafe { MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(MTLPixelFormat::RGBA8Unorm, w, h, false) };
        desc.setUsage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite);
        desc.setStorageMode(MTLStorageMode::Shared);
        let raw: Retained<ProtocolObject<dyn MTLTexture>> =
            dev.newTextureWithDescriptor_iosurface_plane(&desc, surface, 0).ok_or("newTextureWithDescriptor:iosurface:plane: failed")?;
        let size = wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 };
        let ht = unsafe {
            wgpu::hal::metal::Device::texture_from_raw(
                raw,
                wgpu::TextureFormat::Rgba8Unorm,
                MTLTextureType::Type2D,
                1,
                1,
                wgpu::hal::CopyExtent { width: w as u32, height: h as u32, depth: 1 },
                None,
            )
        };
        drop(hal);
        Ok(unsafe {
            ctx.device.create_texture_from_hal::<wgpu::hal::api::Metal>(
                ht,
                &wgpu::TextureDescriptor {
                    label: Some("irys iosurface"),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::TextureUses::UNINITIALIZED,
            )
        })
    }
}

fn render(s: &mut IrysScene, key: usize, make: impl FnOnce(&GpuContext) -> Result<wgpu::Texture, String>, spp: u32) -> i32 {
    let ctx = s.look_ctx();
    if !s.targets.contains_key(&key) {
        match make(ctx) {
            Ok(t) => {
                if (t.width() as usize, t.height() as usize) != s.look.size() {
                    return fail(format!("surface is {}×{}, scene {:?}", t.width(), t.height(), s.look.size()));
                }
                let p = Present::new(&s.look, &t);
                s.targets.insert(key, (t, p));
            }
            Err(e) => return fail(e),
        }
    }
    let (_, p) = &s.targets[&key];
    p.render(&s.look, &s.camera, &Opts { spp: spp.max(1), ..Opts::default() });
    0
}

impl IrysScene {
    fn look_ctx(&self) -> &'static GpuContext {
        irys_diff::gpu::context().expect("the context outlives every scene")
    }
}

/// Render `spp` samples into `surface` (an `IOSurfaceRef`, 'RGBA', the
/// scene's size): written on the device, no readback, finished on return.
/// The surface's Metal texture is made once and kept until the scene is
/// destroyed; pass the same surface each frame.
///
/// # Safety
/// `s` from [`irys_scene_load`]; `surface` a live `IOSurfaceRef` that
/// outlives the scene.
#[cfg(target_vendor = "apple")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn irys_scene_render_iosurface(s: *mut IrysScene, surface: *const std::ffi::c_void, spp: u32) -> i32 {
    let Some(s) = (unsafe { s.as_mut() }) else { return fail("null scene") };
    let Some(sp) = NonNull::new(surface as *mut objc2_io_surface::IOSurfaceRef) else { return fail("null surface") };
    let r: &objc2_io_surface::IOSurfaceRef = unsafe { sp.as_ref() };
    render(s, surface as usize, |ctx| surface::texture(ctx, r), spp)
}

/// [`irys_scene_render_iosurface`] by global IOSurface id
/// (`IOSurfaceGetID`; `IOSurfaceLookup` finds it in this process).
///
/// # Safety
/// `s` from [`irys_scene_load`].
#[cfg(target_vendor = "apple")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn irys_scene_render_iosurface_id(s: *mut IrysScene, id: u32, spp: u32) -> i32 {
    let Some(sr) = objc2_io_surface::IOSurfaceRef::lookup(id) else { return fail(format!("no IOSurface {id}")) };
    let ptr = &*sr as *const objc2_io_surface::IOSurfaceRef as *const std::ffi::c_void;
    let r = unsafe { irys_scene_render_iosurface(s, ptr, spp) };
    // the scene keeps a texture of the surface; keep the surface alive too
    std::mem::forget(sr);
    r
}

/// The readback path, for comparison: render and copy the sRGB8 frame
/// (`width · height · 4` bytes, RGBA, rows packed) into `out`.
///
/// # Safety
/// `s` from [`irys_scene_load`]; `out` holds `width · height · 4` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn irys_scene_render_rgba8(s: *mut IrysScene, out: *mut u8, spp: u32) -> i32 {
    let Some(s) = (unsafe { s.as_mut() }) else { return fail("null scene") };
    if out.is_null() {
        return fail("null output");
    }
    let (w, h) = s.look.size();
    let f = s.look.render(&s.camera, &Opts { spp: spp.max(1), ..Opts::default() });
    let o = unsafe { std::slice::from_raw_parts_mut(out, w * h * 4) };
    let enc = |c: f32| {
        let x = c.clamp(0.0, 1.0);
        let v = if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
        (v * 255.0).round() as u8
    };
    for p in 0..w * h {
        o[4 * p..4 * p + 4].copy_from_slice(&[enc(f.rgb[3 * p]), enc(f.rgb[3 * p + 1]), enc(f.rgb[3 * p + 2]), 255]);
    }
    0
}
