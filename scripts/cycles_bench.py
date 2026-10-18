# Cycles side of phase 3's equal-time benchmark.
#
#   blender -b --factory-startup -P scripts/cycles_bench.py -- <scene dir> [--ref-spp N] [--max-spp N] [--device GPU|CPU]
#
# Reads <scene dir>/scene.json (written by `irys-bench export`), builds the
# same triangles, camera, materials and sky in Cycles, renders a reference
# (cached as cycles_ref.f32) and a sweep of 1, 2, 4, ... spp, and writes
# cycles_<spp>.f32 (linear sRGB, little-endian f32, w*h*3, rows top down)
# and cycles.json (spp and wall seconds per level). relMSE is computed by
# `irys-bench report`, against this reference, so both renderers are scored
# the same way.
#
# Matched against irys:
# - geometry: irys's placements, flattened to world-space triangles;
# - camera: pinhole, vertical field of view, box pixel filter one pixel wide;
# - sky: irys's `Env::radiance` built as shader nodes, the spectral tint of
#   the gradient projected to linear sRGB by irys-bench (`spectrum_rgb`), the
#   sun a white vMF lobe; Cycles importance-samples it from a baked map;
# - diffuse: Lambert (Diffuse BSDF), irys's checker as nodes;
# - conductor: Principled, metallic 1, single-scatter GGX, roughness
#   sqrt(alpha), base colour = irys's spectral F0 projected to sRGB;
# - depth: irys's `depth` surface vertices, each with NEE = Cycles
#   max_bounces = depth - 1. No clamping, no denoising, no adaptive sampling.
# Not matched: Cycles is RGB, irys spectral; Cycles' metallic Fresnel is
# F82-tint, irys's Schlick; Cycles uses Russian roulette, irys a fixed depth.

import json
import os
import sys
import time

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1 :]
root = argv[0]
opts = dict(zip(argv[1::2], argv[2::2]))
ref_spp = int(opts.get("--ref-spp", 8192))
max_spp = int(opts.get("--max-spp", 256))
device = opts.get("--device", "GPU")
cfg = json.load(open(os.path.join(root, "scene.json")))
w, h = cfg["size"]

sc = bpy.context.scene
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)


def node_math(nt, op, a=None, b=None):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    for i, v in enumerate((a, b)):
        if v is None:
            continue
        if isinstance(v, (int, float)):
            n.inputs[i].default_value = v
        else:
            nt.links.new(v, n.inputs[i])
    return n.outputs[0]


def material(spec, name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    if spec["kind"] == "conductor":
        p = nt.nodes.new("ShaderNodeBsdfPrincipled")
        p.distribution = "GGX"
        f0 = spec["f0"]
        p.inputs["Base Color"].default_value = (f0[0], f0[1], f0[2], 1.0)
        p.inputs["Metallic"].default_value = 1.0
        p.inputs["Roughness"].default_value = spec["roughness"]
        nt.links.new(p.outputs[0], out.inputs["Surface"])
        return m
    if spec["kind"] == "emitter":
        e = nt.nodes.new("ShaderNodeEmission")
        r = spec["radiance"]
        e.inputs["Color"].default_value = (r[0], r[1], r[2], 1.0)
        # one-sided, as irys's: the back face is black
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        nt.links.new(node_math(nt, "SUBTRACT", 1.0, geo.outputs["Backfacing"]), e.inputs["Strength"])
        nt.links.new(e.outputs[0], out.inputs["Surface"])
        return m
    d = nt.nodes.new("ShaderNodeBsdfDiffuse")
    d.inputs["Roughness"].default_value = 0.0
    a = spec["albedo"]
    if spec.get("checker"):
        s = spec["checker"]
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(geo.outputs["Position"], sep.inputs[0])
        fx = node_math(nt, "FLOOR", node_math(nt, "DIVIDE", sep.outputs[0], s))
        fy = node_math(nt, "FLOOR", node_math(nt, "DIVIDE", sep.outputs[1], s))
        odd = node_math(nt, "FLOORED_MODULO", node_math(nt, "ADD", fx, fy), 2.0)
        # albedo * (1 - odd * (1 - dark))
        k = node_math(nt, "MULTIPLY", odd, 1.0 - spec["checker_dark"])
        alb = node_math(nt, "MULTIPLY", node_math(nt, "SUBTRACT", 1.0, k), a)
        comb = nt.nodes.new("ShaderNodeCombineXYZ")
        for i in range(3):
            nt.links.new(alb, comb.inputs[i])
        nt.links.new(comb.outputs[0], d.inputs["Color"])
    else:
        d.inputs["Color"].default_value = (a, a, a, 1.0)
    nt.links.new(d.outputs[0], out.inputs["Surface"])
    return m


def mesh_from_corners(name, corners):
    """A mesh of independent triangles from f32 xyz per corner."""
    me = bpy.data.meshes.new(name)
    n = corners.shape[0]
    me.vertices.add(n)
    me.vertices.foreach_set("co", corners.reshape(-1))
    me.loops.add(n)
    me.loops.foreach_set("vertex_index", np.arange(n, dtype=np.int32))
    me.polygons.add(n // 3)
    me.polygons.foreach_set("loop_start", np.arange(0, n, 3, dtype=np.int32))
    me.polygons.foreach_set("loop_total", np.full(n // 3, 3, dtype=np.int32))
    me.update()
    return me


if "meshes" in cfg:
    # Phase 5a's instanced export: each irys tree once, placed per object,
    # the material on the object so instances can differ.
    blob = np.fromfile(os.path.join(root, "meshes.f32"), dtype="<f4").reshape(-1, 3)
    data = []
    for k, m in enumerate(cfg["meshes"]):
        me = mesh_from_corners(f"tree{k}", blob[m["offset"] : m["offset"] + m["corners"]])
        me.materials.append(None)
        data.append(me)
    cache = {}
    from mathutils import Matrix as _M

    def instancer(mesh_k, mat, mats):
        """Many placements of one tree as one object: geometry nodes'
        Instance on Points from a point cloud carrying each placement's
        rotation and scale. Linked duplicates are re-synced on every render
        (seconds for the pavilion's 20 000 pebbles), these are not."""
        src = bpy.data.objects.new(f"src{mesh_k}_{len(cache)}", data[mesh_k].copy())
        src.data.materials[0] = mat
        sc.collection.objects.link(src)
        src.hide_render = True
        n = len(mats)
        pts = bpy.data.meshes.new(f"pts{mesh_k}")
        pts.vertices.add(n)
        loc, rot, scl = [], [], []
        for m in mats:
            M = _M((m[0:4], m[4:8], m[8:12], (0, 0, 0, 1)))
            l, q, s3 = M.decompose()
            loc += list(l)
            rot += list(q.to_euler())
            scl += list(s3)
        pts.vertices.foreach_set("co", loc)
        for name, vals in (("irys_rot", rot), ("irys_scl", scl)):
            a = pts.attributes.new(name, "FLOAT_VECTOR", "POINT")
            a.data.foreach_set("vector", vals)
        o = bpy.data.objects.new(f"inst{mesh_k}_{len(cache)}", pts)
        sc.collection.objects.link(o)
        g = bpy.data.node_groups.new("irys_inst", "GeometryNodeTree")
        g.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
        g.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
        gi, go = g.nodes.new("NodeGroupInput"), g.nodes.new("NodeGroupOutput")
        ip = g.nodes.new("GeometryNodeInstanceOnPoints")
        oi = g.nodes.new("GeometryNodeObjectInfo")
        oi.inputs["Object"].default_value = src
        oi.transform_space = "ORIGINAL"
        ra, sa = g.nodes.new("GeometryNodeInputNamedAttribute"), g.nodes.new("GeometryNodeInputNamedAttribute")
        for na, nm in ((ra, "irys_rot"), (sa, "irys_scl")):
            na.data_type = "FLOAT_VECTOR"
            na.inputs["Name"].default_value = nm
        g.links.new(gi.outputs[0], ip.inputs["Points"])
        g.links.new(oi.outputs["Geometry"], ip.inputs["Instance"])
        g.links.new(ra.outputs["Attribute"], ip.inputs["Rotation"])
        g.links.new(sa.outputs["Attribute"], ip.inputs["Scale"])
        g.links.new(ip.outputs["Instances"], go.inputs[0])
        mod = o.modifiers.new("irys_inst", "NODES")
        mod.node_group = g

    groups = {}
    for ob in cfg["objects"]:
        key = json.dumps(ob["material"], sort_keys=True)
        if key not in cache:
            cache[key] = material(ob["material"], f"m{len(cache)}")
        groups.setdefault((ob["mesh"], key), []).append(ob["matrix"])
    n_obj = 0
    for (mk, key), mats in groups.items():
        # Shear-free placements (all of an import's) instance exactly as TRS.
        if len(mats) >= 2:
            instancer(mk, cache[key], mats)
            continue
        for m in mats:
            o = bpy.data.objects.new(f"o{n_obj}", data[mk])
            n_obj += 1
            o.material_slots[0].link = "OBJECT"
            o.material_slots[0].material = cache[key]
            o.matrix_world = _M((m[0:4], m[4:8], m[8:12], (0, 0, 0, 1)))
            sc.collection.objects.link(o)
    print(f"cycles: {len(data)} meshes, {len(cfg['objects'])} objects, {len(cache)} materials", flush=True)

for ob in cfg.get("objects", []) if "meshes" not in cfg else []:
    me = bpy.data.meshes.new(ob["name"])
    v = np.array(ob["verts"], dtype=np.float64).reshape(-1, 3)
    t = np.array(ob["tris"], dtype=np.int64).reshape(-1, 3)
    me.from_pydata(v.tolist(), [], t.tolist())
    me.update()
    me.materials.append(material(ob["material"], ob["name"] + "_m"))
    o = bpy.data.objects.new(ob["name"], me)
    sc.collection.objects.link(o)

# The sky: tint * sky * (0.15 + 0.25 max(d_z, 0)) + S exp(kappa (d . sun - 1)).
env = cfg["env"]
world = bpy.data.worlds.new("irys")
sc.world = world
world.use_nodes = True
nt = world.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
tc = nt.nodes.new("ShaderNodeTexCoord")
# In a world shader the generated coordinate is the ray direction.
dirn = nt.nodes.new("ShaderNodeVectorMath")
dirn.operation = "NORMALIZE"
nt.links.new(tc.outputs["Generated"], dirn.inputs[0])
dot = nt.nodes.new("ShaderNodeVectorMath")
dot.operation = "DOT_PRODUCT"
nt.links.new(dirn.outputs[0], dot.inputs[0])
dot.inputs[1].default_value = env["sun"]
lobe = node_math(
    nt,
    "MULTIPLY",
    node_math(nt, "EXPONENT", node_math(nt, "MULTIPLY", node_math(nt, "SUBTRACT", dot.outputs["Value"], 1.0), env["kappa"])),
    0.0 if "sun_lamp" in cfg else env["sun_radiance"],
)
if "sun_lamp" in cfg:
    # Phase 5a: the lobe as a sun lamp of the same irradiance and solid angle.
    from mathutils import Vector as _V

    sl = bpy.data.lights.new("irys_sun", "SUN")
    sl.energy = cfg["sun_lamp"]["irradiance"]
    sl.angle = 2.0 * cfg["sun_lamp"]["radius"]
    so = bpy.data.objects.new("irys_sun", sl)
    so.rotation_euler = _V(env["sun"]).to_track_quat("Z", "Y").to_euler()
    sc.collection.objects.link(so)
sep = nt.nodes.new("ShaderNodeSeparateXYZ")
nt.links.new(dirn.outputs[0], sep.inputs[0])
grad = node_math(nt, "MULTIPLY_ADD", node_math(nt, "MAXIMUM", sep.outputs[2], 0.0), 0.25)
grad.node.inputs[2].default_value = 0.15
tint = env["sky_tint_rgb"]
comb = nt.nodes.new("ShaderNodeCombineXYZ")
for i in range(3):
    nt.links.new(
        node_math(nt, "ADD", node_math(nt, "MULTIPLY", grad, tint[i] * env["sky"]), lobe), comb.inputs[i]
    )
bg = nt.nodes.new("ShaderNodeBackground")
bg.inputs["Strength"].default_value = 1.0
nt.links.new(comb.outputs[0], bg.inputs["Color"])
wo = nt.nodes.new("ShaderNodeOutputWorld")
nt.links.new(bg.outputs[0], wo.inputs["Surface"])
world.cycles.sampling_method = "MANUAL"
world.cycles.sample_map_resolution = 2048

# The pinhole.
cam = cfg["camera"]
cd = bpy.data.cameras.new("cam")
cd.sensor_fit = "VERTICAL"
cd.angle_y = 2.0 * np.arctan(cam["tan_half"])
co = bpy.data.objects.new("cam", cd)
sc.collection.objects.link(co)
from mathutils import Matrix, Vector

r, u, f, e = (Vector(cam[k]) for k in ("right", "up", "forward", "eye"))
co.matrix_world = Matrix(
    ((r.x, u.x, -f.x, e.x), (r.y, u.y, -f.y, e.y), (r.z, u.z, -f.z, e.z), (0, 0, 0, 1))
)
sc.camera = co

rd = sc.render
rd.engine = "CYCLES"
rd.resolution_x, rd.resolution_y, rd.resolution_percentage = w, h, 100
rd.use_persistent_data = True
rd.image_settings.file_format = "OPEN_EXR"
rd.image_settings.color_depth = "32"
sc.view_settings.view_transform = "Standard"
sc.view_settings.look = "None"
sc.view_settings.exposure = 0.0
sc.view_settings.gamma = 1.0
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
depth = cfg["depth"]
cy.max_bounces = depth - 1
cy.diffuse_bounces = depth - 1
cy.glossy_bounces = depth - 1
cy.transmission_bounces = 0
cy.volume_bounces = 0
cy.transparent_max_bounces = 0
cy.sample_clamp_direct = 0.0
cy.sample_clamp_indirect = 0.0
cy.blur_glossy = 0.0
cy.caustics_reflective = True
cy.caustics_refractive = True
cy.use_auto_tile = False


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
    render(spp, 7 + spp)  # once to settle, then the fastest of two timed runs
    img, secs = render(spp, 1000 + spp)
    secs = min(secs, render(spp, 1000 + spp)[1])
    img.tofile(os.path.join(root, f"cycles_{spp}.f32"))
    rows.append({"spp": spp, "seconds": secs})
    print(f"cycles {spp:5d} spp {secs:8.4f} s", flush=True)
    spp *= 2
json.dump({"size": [w, h], "device": device, "rows": rows}, open(os.path.join(root, "cycles.json"), "w"), indent=1)
