# Phase 5b: one Cycles frame of a .blend, matched to irys's beauty render.
#
#   blender -b -Y file.blend -P scripts/cycles_look.py -- <out prefix> <export dir> [--scene NAME]
#           [--spp N] [--depth N] [--device GPU|CPU] [--denoise 0|1] [--width W] [--camera cams.json --index K]
#
# The file's own materials, lights and world; the export's camera and
# aspect at --width; max_bounces = depth - 1 for every bounce type; a box
# filter one pixel wide; Standard view; no adaptive sampling, clamping,
# motion blur or depth of field. --denoise 1 turns on OIDN (Cycles'
# default denoiser on this machine). Writes <prefix>.f32 (linear RGB),
# <prefix>.png (Standard view) and <prefix>.json (seconds, spp).

import json
import os
import sys
import time

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blend_export  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
prefix, export = argv[0], argv[1]
opts = dict(zip(argv[2::2], argv[3::2]))
spp = int(opts.get("--spp", 1024))
depth = int(opts.get("--depth", 8))
device = opts.get("--device", "GPU")
denoise = opts.get("--denoise", "0") == "1"
if "--scene" in opts:
    bpy.context.window.scene = bpy.data.scenes[opts["--scene"]]
sc = bpy.context.scene
side = json.load(open(os.path.join(export, "irys.json")))
w0, h0 = side["camera"]["size"]
w = int(opts.get("--width", w0))
h = int(round(w * h0 / w0))
blend_export.render_settings_like_render(sc)
if "--frame" in opts:
    sc.frame_set(int(opts["--frame"]))

rd = sc.render
rd.engine = "CYCLES"
rd.resolution_x, rd.resolution_y, rd.resolution_percentage = w, h, 100
rd.pixel_aspect_x = rd.pixel_aspect_y = 1.0
rd.use_persistent_data = True
rd.use_compositing = False
rd.use_sequencer = False
rd.use_motion_blur = False
rd.film_transparent = False
rd.image_settings.file_format = "OPEN_EXR"
rd.image_settings.color_depth = "32"
rd.use_border = False
sc.view_settings.view_transform = "Standard"
sc.view_settings.look = "None"
sc.view_settings.exposure = 0.0
sc.view_settings.gamma = 1.0
sc.view_settings.use_curve_mapping = False
sc.camera.data.dof.use_dof = False
cy = sc.cycles
if device == "GPU":
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "METAL"
    cy.device = "GPU"
else:
    cy.device = "CPU"
cy.use_adaptive_sampling = False
cy.use_denoising = denoise
if denoise:
    cy.denoiser = "OPENIMAGEDENOISE"
    cy.denoising_input_passes = "RGB_ALBEDO_NORMAL"
cy.pixel_filter_type = "BOX"
cy.filter_width = 1.0
cy.max_bounces = depth - 1
for k in ("diffuse_bounces", "glossy_bounces", "transmission_bounces", "transparent_max_bounces"):
    setattr(cy, k, depth - 1)
cy.volume_bounces = 0
cy.sample_clamp_direct = 0.0
cy.sample_clamp_indirect = float(opts.get("--clamp", 0.0))
cy.blur_glossy = 0.0
cy.use_auto_tile = False
cy.samples = spp
cy.seed = int(opts.get("--seed", 1))
first = [v for v in sc.view_layers if v.use][0]
for vl in sc.view_layers:
    vl.use = vl == first

t = time.perf_counter()
bpy.ops.render.render(write_still=False)
secs = time.perf_counter() - t
path = prefix + "_tmp.exr"
bpy.data.images["Render Result"].save_render(path)
img = bpy.data.images.load(path)
px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1, :, :3]
bpy.data.images.remove(img)
os.remove(path)
np.ascontiguousarray(px).astype("<f4").tofile(prefix + ".f32")
srgb = np.where(px <= 0.0031308, 12.92 * px, 1.055 * np.power(np.clip(px, 0, None), 1 / 2.4) - 0.055)
im = bpy.data.images.new("irys_cycles_png", w, h, alpha=False)
im.colorspace_settings.name = "Non-Color"
rgba = np.ones((h, w, 4), dtype=np.float32)
rgba[..., :3] = np.clip(srgb, 0, 1)
im.pixels.foreach_set(rgba[::-1].ravel())
im.filepath_raw = prefix + ".png"
im.file_format = "PNG"
im.save()
json.dump({"spp": spp, "seconds": secs, "size": [w, h], "denoise": denoise, "device": device, "depth": depth,
           "mean": float(px.mean())},
          open(prefix + ".json", "w"))
print(f"cycles {w}x{h} {spp} spp denoise={denoise} {secs:.2f} s mean {px.mean():.5f}", flush=True)
