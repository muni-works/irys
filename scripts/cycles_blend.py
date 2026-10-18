# Phase 5a: the Cycles side of the real-scene benchmark, on the .blend itself.
#
#   blender -b -Y file.blend -P scripts/cycles_blend.py -- <bench dir> <export dir> [--scene NAME]
#           [--ref-spp N] [--max-spp N] [--device GPU|CPU]
#
# The file's own materials, lights and world, so this is what irys's import
# is measured against for the "true materials" numbers. Matched to irys's
# render: the camera and size of the export's sidecar (the file's camera at
# irys's resolution), depth = the bench's (Cycles max_bounces = depth - 1,
# for every bounce type), the render-time depsgraph the exporter saw, no
# denoising / adaptive sampling / clamping / compositing / motion blur /
# depth of field, a box filter one pixel wide, Standard view transform.
# Writes the same files as cycles_bench.py: cycles_ref.f32, cycles_<spp>.f32
# and cycles.json.

import json
import os
import sys
import time

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blend_export  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1 :]
root, export = argv[0], argv[1]
opts = dict(zip(argv[2::2], argv[3::2]))
ref_spp = int(opts.get("--ref-spp", 4096))
max_spp = int(opts.get("--max-spp", 1024))
device = opts.get("--device", "GPU")
depth = int(opts.get("--depth", 8))
if "--scene" in opts:
    bpy.context.window.scene = bpy.data.scenes[opts["--scene"]]
sc = bpy.context.scene
side = json.load(open(os.path.join(export, "irys.json")))
w, h = side["camera"]["size"]
os.makedirs(root, exist_ok=True)
blend_export.render_settings_like_render(sc)

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
cy.use_denoising = False
cy.pixel_filter_type = "BOX"
cy.filter_width = 1.0
cy.max_bounces = depth - 1
for k in ("diffuse_bounces", "glossy_bounces", "transmission_bounces", "transparent_max_bounces"):
    setattr(cy, k, depth - 1)
cy.volume_bounces = 0
cy.sample_clamp_direct = 0.0
cy.sample_clamp_indirect = 0.0
cy.blur_glossy = 0.0
cy.use_auto_tile = False
# The first view layer the file renders (the exporter measured that one).
first = [v for v in sc.view_layers if v.use][0]
for vl in sc.view_layers:
    vl.use = vl == first


def render(spp, seed):
    cy.samples = spp
    cy.seed = seed
    t = time.perf_counter()
    bpy.ops.render.render(write_still=False)
    secs = time.perf_counter() - t
    path = os.path.join(root, "_cycles.exr")
    bpy.data.images["Render Result"].save_render(path)
    img = bpy.data.images.load(path)
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1, :, :3]
    bpy.data.images.remove(img)
    os.remove(path)
    return np.ascontiguousarray(px).astype("<f4"), secs


render(1, 0)  # warm: kernels, BVH, the world's importance map
ref_path = os.path.join(root, "cycles_ref.f32")
if not os.path.exists(ref_path):
    img, secs = render(ref_spp, 1 << 20)
    img.tofile(ref_path)
    json.dump({"spp": ref_spp, "seconds": secs}, open(os.path.join(root, "cycles_ref.json"), "w"))
    print(f"cycles reference {ref_spp} spp in {secs:.1f} s", flush=True)
rows = []
spp = 1
while spp <= max_spp:
    render(spp, 7 + spp)
    img, secs = render(spp, 1000 + spp)
    secs = min(secs, render(spp, 1000 + spp)[1])
    img.tofile(os.path.join(root, f"cycles_{spp}.f32"))
    rows.append({"spp": spp, "seconds": secs})
    print(f"cycles {spp:5d} spp {secs:8.4f} s", flush=True)
    spp *= 2
json.dump({"size": [w, h], "device": device, "rows": rows, "source": "blend"}, open(os.path.join(root, "cycles.json"), "w"), indent=1)
