# Cycles side of phase 4's OpenPBR reference match.
#
#   blender -b --factory-startup -P scripts/openpbr_cycles.py -- <dir> [spp] [CPU|GPU]
#
# Reads <dir>/sweep.json (written by `examples/look_sweep`), builds the same
# unit sphere, emissive quads, constant world and orthographic camera, and
# renders every swatch with its OpenPBR parameters mapped onto Blender's
# Principled BSDF (4.x, which follows OpenPBR's structure): base colour x
# base weight, metallic, roughness, IOR, specular IOR level = weight / 2,
# specular tint, transmission, coat (weight, roughness, IOR, tint), sheen
# (= fuzz), thin film (thickness in nm, IOR), random-walk subsurface (radius
# = radius_scale, scale = radius, colour = base colour). Single-scatter GGX,
# box pixel filter one pixel wide, no clamping, no denoising, caustics on,
# max bounces = irys's depth - 1. Writes cycles_<i>.f32 (linear sRGB,
# little-endian f32, w*h*3, rows top down).
#
# Not mapped (Principled 4.4 has no input for them): transmission
# dispersion, coat darkening, OpenPBR's fuzz model (Blender's sheen is the
# microfibre LTC fit the spec recommends; irys uses Charlie).

import json
import math
import os
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1 :]
root = argv[0]
spp = int(argv[1]) if len(argv) > 1 else 1024
device = argv[2] if len(argv) > 2 else "GPU"
cfg = json.load(open(os.path.join(root, "sweep.json")))
size = cfg["size"]

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
cy = scene.cycles
if device == "GPU":
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for t in ("METAL", "OPTIX", "CUDA", "HIP"):
        try:
            prefs.compute_device_type = t
            prefs.get_devices()
            if any(d.type == t for d in prefs.devices):
                for d in prefs.devices:
                    d.use = True
                cy.device = "GPU"
                break
        except TypeError:
            continue
cy.samples = spp
cy.use_adaptive_sampling = False
cy.use_denoising = False
cy.pixel_filter_type = "BOX"
cy.filter_width = 1.0
d = cfg["depth"] - 1
cy.max_bounces = d
cy.diffuse_bounces = d
cy.glossy_bounces = d
cy.transmission_bounces = d
cy.volume_bounces = 0
cy.transparent_max_bounces = d
cy.sample_clamp_direct = 0.0
cy.sample_clamp_indirect = 0.0
cy.blur_glossy = 0.0
cy.caustics_reflective = True
cy.caustics_refractive = True
cy.seed = 7
scene.render.resolution_x = size
scene.render.resolution_y = size
scene.render.resolution_percentage = 100
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"

world = bpy.data.worlds.new("env")
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (cfg["env"], cfg["env"], cfg["env"], 1.0)
bg.inputs["Strength"].default_value = 1.0
scene.world = world

cam_data = bpy.data.cameras.new("cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = cfg["width"]
cam = bpy.data.objects.new("cam", cam_data)
cam.location = (0.0, -5.0, 0.0)
cam.rotation_euler = (math.pi / 2, 0.0, 0.0)
scene.collection.objects.link(cam)
scene.camera = cam

for i, l in enumerate(cfg["lights"]):
    c, u, v = (np.array(l[k]) for k in ("center", "u", "v"))
    verts = [tuple(c - u - v), tuple(c + u - v), tuple(c + u + v), tuple(c - u + v)]
    me = bpy.data.meshes.new(f"light{i}")
    me.from_pydata(verts, [], [(0, 1, 2, 3)])  # normal = u x v, the emitting side
    ob = bpy.data.objects.new(f"light{i}", me)
    mat = bpy.data.materials.new(f"light{i}")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    t = l["tint"]
    em.inputs["Color"].default_value = (t[0], t[1], t[2], 1.0)
    em.inputs["Strength"].default_value = l["radiance"]
    # One-sided, like irys's quads: the back is black.
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    mix = nt.nodes.new("ShaderNodeMixShader")
    black = nt.nodes.new("ShaderNodeEmission")
    black.inputs["Strength"].default_value = 0.0
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(geo.outputs["Backfacing"], mix.inputs["Fac"])
    nt.links.new(em.outputs[0], mix.inputs[1])
    nt.links.new(black.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs["Surface"])
    me.materials.append(mat)
    scene.collection.objects.link(ob)

bpy.ops.mesh.primitive_uv_sphere_add(segments=256, ring_count=128, radius=1.0, location=(0, 0, 0))
sphere = bpy.context.active_object
bpy.ops.object.shade_smooth()
smat = bpy.data.materials.new("swatch")
smat.use_nodes = True
sphere.data.materials.append(smat)
p = smat.node_tree.nodes["Principled BSDF"]
p.distribution = "GGX"
p.subsurface_method = "RANDOM_WALK"


def rgba(c):
    return (c[0], c[1], c[2], 1.0)


for i, s in enumerate(cfg["swatches"]):
    m = s["m"]
    bw = m["base_weight"]
    base = [bw * x for x in m["base_color"]]
    if m["subsurface_weight"] > 0.0:
        base = m["subsurface_color"]
    p.inputs["Base Color"].default_value = rgba(base)
    p.inputs["Metallic"].default_value = m["base_metalness"]
    p.inputs["Roughness"].default_value = m["specular_roughness"]
    p.inputs["IOR"].default_value = m["specular_ior"]
    p.inputs["Specular IOR Level"].default_value = 0.5 * m["specular_weight"]
    p.inputs["Specular Tint"].default_value = rgba(m["specular_color"])
    p.inputs["Transmission Weight"].default_value = m["transmission_weight"]
    p.inputs["Coat Weight"].default_value = m["coat_weight"]
    p.inputs["Coat Roughness"].default_value = m["coat_roughness"]
    p.inputs["Coat IOR"].default_value = m["coat_ior"]
    p.inputs["Coat Tint"].default_value = rgba(m["coat_color"])
    p.inputs["Sheen Weight"].default_value = m["fuzz_weight"]
    p.inputs["Sheen Roughness"].default_value = m["fuzz_roughness"]
    p.inputs["Sheen Tint"].default_value = rgba(m["fuzz_color"])
    tf = m["thin_film_thickness"] * 1000.0 if m["thin_film_weight"] > 0.0 else 0.0
    p.inputs["Thin Film Thickness"].default_value = tf
    p.inputs["Thin Film IOR"].default_value = m["thin_film_ior"]
    p.inputs["Subsurface Weight"].default_value = m["subsurface_weight"]
    p.inputs["Subsurface Radius"].default_value = tuple(m["subsurface_radius_scale"])
    p.inputs["Subsurface Scale"].default_value = m["subsurface_radius"]
    if "Subsurface IOR" in p.inputs:  # only shown for the skin walk
        p.inputs["Subsurface IOR"].default_value = m["specular_ior"]
    p.inputs["Subsurface Anisotropy"].default_value = 0.0
    path = os.path.join(root, f"cycles_{i}.exr")
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path)
    px = np.array(img.pixels[:], dtype=np.float32).reshape(size, size, 4)[::-1, :, :3]
    px.astype("<f4").tofile(os.path.join(root, f"cycles_{i}.f32"))
    bpy.data.images.remove(img)
    os.remove(path)
    print(f"swatch {i}: {s['row']} {s['label']}", flush=True)
