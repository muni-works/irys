# Phase 5a: a .blend as a scene irys reads.
#
#   blender -b -Y file.blend -P scripts/blend_export.py -- <out dir> [--scene NAME] [--width 480]
#           [--look 1] [--frames a:b[:step]] [--reel fps:a:b] [--env-width 1024] [--measure-width 480]
#
# Phase 5b (--look 1, the default) adds, per material, its OpenPBR
# parameters and baked textures (scripts/blend_openpbr.py: textures/*.png,
# glTF baseColorTexture + KHR_texture_transform, the metal colour and the
# cut-out named in the sidecar), each light's RGB colour (blackbody tints
# included), spot cones and camera visibility, light portals as geometry with
# their own sky (portal_*.f32), the world as an equirect (env.f32), and with
# --frames the file's camera frame by frame. `irys_diff::look::scene::load`
# reads it; the phase 5a fields below are unchanged for `irys_diff::blend`.
# 5b also stops exporting particle emitters a render hides
# (show_instancer_for_render), which 5a did: the pavilion's lotus emitter
# plane covered its pool.
#
# Writes <out dir>/scene.glb (glTF 2.0: every unique evaluated mesh once,
# triangulated, with normals and UVs, one primitive per material, and one
# node per instance, so instancing is kept) and <out dir>/irys.json (the
# sidecar: camera, lights, the environment fit, each glTF material's irys
# BSDF, and what was dropped). `irys_diff::blend::load` reads both.
#
# Why not Blender's own glTF exporter: it does not write particle-system or
# geometry-node instances as shared meshes (the pavilion's 20 000 pebbles
# would be 20 000 copies or nothing), it has no area lights, and it
# exports a Principled BSDF's factors but not the older node graphs
# (Diffuse/Anisotropic/Mix) the demo files are built from. This writes the
# same container (positions, normals, TEXCOORD_0, u32 indices, nodes with
# matrices, pbrMetallicRoughness factors, a perspective camera) from the
# render-time depsgraph, and keeps what glTF has no word for in the sidecar.
#
# Coordinates: glTF is Y-up, so nodes carry Blender's Z-up world matrix
# premultiplied by (x, y, z) -> (x, z, -y); the sidecar is in Blender's
# Z-up metres, irys's frame. `-Y` keeps the file's own scripts from running.
#
# irys's BSDFs today (phase 5a): a grey Lambertian (one scalar albedo per
# placement), one GGX conductor (one theta per scene), and a one-sided
# emitter (one radiance per scene). Each material is reduced to those:
#
# - the surface output is walked through Mix/Add shaders into weighted
#   lobes: diffuse (Diffuse, Principled base, Translucent, Toon, Oren-Nayar
#   roughness ignored), glossy (Glossy/Anisotropic/Principled metallic or
#   specular), transmission (Glass/Refraction/Transparent/Principled
#   transmission), emission (Emission, Principled emission), subsurface;
# - a colour input is a constant, an image (its mean, linear), an RGB node,
#   a MixRGB/Mix of colours (by its factor), or else whatever image feeds it
#   upstream, else 0.8 and the input is reported as unresolved;
# - kind: emitter if emission dominates the reflected albedo, hidden if
#   transmission weighs >= 1/2 (glass: irys has no dielectric, so a window
#   lets light through rather than stopping it), conductor if glossy weighs
#   more than diffuse and the glossy colour is not grey-dark, else diffuse
#   with the luminance of the diffuse colour; skin flagged where
#   Principled subsurface > 0 (irys's skin is its own renderer, so it maps
#   to diffuse here and is reported).
#
# Lights (Cycles units, measured: sun irradiance = strength; area radiance
# = P / (pi A); point/spot radiance = P / (4 pi^2 r^2) on a sphere of the
# soft radius) become emitter quads, discs and spheres, and the sun the
# environment's vMF lobe (kappa = 2 / r^2 for angular radius r, the same
# solid angle; peak = S kappa / 2 pi, so the irradiance is S). irys has one
# emitter radiance, so every emitter is shrunk about its centre by
# sqrt(L_i / L_max): the same power from the same place.
#
# The world is baked by Cycles to a small equirect and fitted to irys's
# sky gradient `sky (0.15 + 0.25 max(z, 0))` by least squares over solid
# angle (texels above 20x the median are clipped and reported: a sky
# texture's sun disc belongs to the sun light, not the gradient).

import json
import math
import os
import struct
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blend_openpbr  # noqa: E402

LUM = np.array([0.2126, 0.7152, 0.0722])
DIFFUSE_NODES = {"ShaderNodeBsdfDiffuse", "ShaderNodeBsdfTranslucent", "ShaderNodeBsdfToon", "ShaderNodeBsdfSheen"}
GLOSSY_NODES = {"ShaderNodeBsdfGlossy", "ShaderNodeBsdfAnisotropic", "ShaderNodeBsdfMetallic"}
TRANSMIT_NODES = {"ShaderNodeBsdfGlass", "ShaderNodeBsdfRefraction", "ShaderNodeBsdfTransparent"}
PASS_NODES = {"NodeReroute"}


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


class Reducer:
    """Material node graphs to irys's BSDFs, with a log of what fell away."""

    def __init__(self, measured=None):
        self.measured = measured or {}
        self.images = {}
        self.dropped = {}  # material name -> sorted list of dropped features

    def image_mean(self, img):
        if img is None:
            return None
        if img.name in self.images:
            return self.images[img.name]
        c = None
        try:
            if img.size[0] > 0:
                small = img.copy()
                small.scale(min(64, img.size[0]), min(64, img.size[1]))
                px = np.array(small.pixels[:], dtype=np.float64).reshape(-1, 4)[:, :3]
                bpy.data.images.remove(small)
                if img.colorspace_settings.name in ("sRGB",) and not img.is_float:
                    px = srgb_to_linear(px)
                c = px.mean(axis=0)
        except Exception:
            c = None
        self.images[img.name] = c
        return c

    def upstream_image(self, node, seen=None):
        seen = seen or set()
        if node in seen:
            return None
        seen.add(node)
        if node.bl_idname == "ShaderNodeTexImage":
            return self.image_mean(node.image)
        for i in node.inputs:
            for l in i.links:
                c = self.upstream_image(l.from_node, seen)
                if c is not None:
                    return c
        return None

    def colour(self, sock, drop):
        """The mean linear RGB a colour socket evaluates to."""
        if not sock.is_linked:
            v = sock.default_value
            try:
                return np.array([v[0], v[1], v[2]], dtype=np.float64)
            except TypeError:
                return np.array([v, v, v], dtype=np.float64)
        n = sock.links[0].from_node
        while n.bl_idname in PASS_NODES:
            n = n.inputs[0].links[0].from_node
        t = n.bl_idname
        if t == "ShaderNodeTexImage":
            c = self.image_mean(n.image)
            drop.add("texture averaged")
            return c if c is not None else np.array([0.8] * 3)
        if t == "ShaderNodeRGB":
            v = n.outputs[0].default_value
            return np.array(v[:3], dtype=np.float64)
        if t == "ShaderNodeMixRGB" or (t == "ShaderNodeMix" and getattr(n, "data_type", "") == "RGBA"):
            if t == "ShaderNodeMixRGB":
                f, a, b = n.inputs[0], n.inputs[1], n.inputs[2]
            else:
                f, a, b = n.inputs["Factor"], n.inputs[6], n.inputs[7]
            fac = self.scalar(f, drop)
            ca, cb = self.colour(a, drop), self.colour(b, drop)
            bt = getattr(n, "blend_type", "MIX")
            if bt == "MULTIPLY":
                return ca * (1 - fac + fac * cb)
            if bt != "MIX":
                drop.add(f"mix {bt.lower()} as mix")
            return (1 - fac) * ca + fac * cb
        if t in ("ShaderNodeHueSaturation", "ShaderNodeBrightContrast", "ShaderNodeGamma", "ShaderNodeInvert", "ShaderNodeRGBCurve"):
            drop.add(f"{t[10:]} ignored")
            key = "Color"
            return self.colour(n.inputs[key], drop) if key in n.inputs else np.array([0.8] * 3)
        if t == "ShaderNodeTexSky":
            drop.add("sky texture as its upper-hemisphere mean")
            return sky_node_mean(n)
        c = self.upstream_image(n)
        drop.add(f"{t[10:]} colour {'via its image' if c is not None else 'unresolved'}")
        return c if c is not None else np.array([0.8] * 3)

    def scalar(self, sock, drop):
        if not sock.is_linked:
            v = sock.default_value
            return float(v if isinstance(v, (int, float)) else v[0])
        n = sock.links[0].from_node
        if n.bl_idname == "ShaderNodeValue":
            return float(n.outputs[0].default_value)
        if n.bl_idname == "ShaderNodeLayerWeight" or n.bl_idname == "ShaderNodeFresnel":
            drop.add("fresnel/layer weight as 0.2")
            return 0.2
        c = self.upstream_image(n)
        drop.add(f"{n.bl_idname[10:]} factor {'as its image mean' if c is not None else 'as 0.5'}")
        return float(c @ LUM) if c is not None else 0.5

    def lobes(self, sock, w, out, drop, depth=0):
        """Accumulate weighted lobes reached from shader socket `sock`."""
        if not sock.is_linked or depth > 32 or w <= 0:
            return
        n = sock.links[0].from_node
        while n.bl_idname in PASS_NODES:
            n = n.inputs[0].links[0].from_node
        t = n.bl_idname
        if t == "ShaderNodeMixShader" and n.inputs[0].is_linked and n.inputs[0].links[0].from_node.bl_idname == "ShaderNodeNewGeometry" \
                and n.inputs[0].links[0].from_socket.name == "Backfacing":
            # One side one shader, the other side another: irys surfaces are
            # one-sided emitters or two-sided BSDFs, so keep the side that
            # emits (if either does) and say which.
            front = {"diffuse": [], "glossy": [], "transmit": [], "emit": [], "skin": False}
            back = {"diffuse": [], "glossy": [], "transmit": [], "emit": [], "skin": False}
            self.lobes(n.inputs[1], w, front, drop, depth + 1)
            self.lobes(n.inputs[2], w, back, drop, depth + 1)
            side, keep = ("back", back) if back["emit"] and not front["emit"] else ("front", front)
            out["emit_side"] = side
            drop.add("backfacing mix: one side kept")
            for k in ("diffuse", "glossy", "transmit", "emit"):
                out[k] += keep[k]
        elif t == "ShaderNodeMixShader":
            f = min(max(self.scalar(n.inputs[0], drop), 0.0), 1.0)
            self.lobes(n.inputs[1], w * (1 - f), out, drop, depth + 1)
            self.lobes(n.inputs[2], w * f, out, drop, depth + 1)
        elif t == "ShaderNodeAddShader":
            self.lobes(n.inputs[0], w, out, drop, depth + 1)
            self.lobes(n.inputs[1], w, out, drop, depth + 1)
        elif t in DIFFUSE_NODES:
            if t != "ShaderNodeBsdfDiffuse":
                drop.add(f"{t[14:]} as diffuse")
            out["diffuse"].append((w, self.colour(n.inputs["Color"], drop)))
        elif t in GLOSSY_NODES:
            if t == "ShaderNodeBsdfAnisotropic" or ("Anisotropy" in n.inputs and n.inputs["Anisotropy"].default_value != 0):
                drop.add("anisotropy")
            r = self.scalar(n.inputs["Roughness"], drop) if "Roughness" in n.inputs else 0.2
            col = self.colour(n.inputs["Color"], drop) if "Color" in n.inputs else np.array([0.9] * 3)
            out["glossy"].append((w, col, r))
        elif t in TRANSMIT_NODES:
            out["transmit"].append(w)
        elif t == "ShaderNodeEmission":
            c = self.colour(n.inputs["Color"], drop) * self.scalar(n.inputs["Strength"], drop)
            out["emit"].append((w, c))
        elif t == "ShaderNodeBsdfPrincipled":
            base = self.colour(n.inputs["Base Color"], drop)
            met = self.scalar(n.inputs["Metallic"], drop)
            rough = self.scalar(n.inputs["Roughness"], drop)
            tr = self.scalar(n.inputs["Transmission Weight"], drop)
            ss = self.scalar(n.inputs["Subsurface Weight"], drop)
            spec = self.scalar(n.inputs["Specular IOR Level"], drop)
            if ss > 0:
                out["skin"] = True
            if self.scalar(n.inputs["Coat Weight"], drop) > 0:
                drop.add("coat")
            if self.scalar(n.inputs["Sheen Weight"], drop) > 0:
                drop.add("sheen")
            if spec > 0 and met < 1:
                drop.add("dielectric specular")
            out["transmit"].append(w * tr * (1 - met))
            out["glossy"].append((w * met, base, rough))
            out["diffuse"].append((w * (1 - met) * (1 - tr), base))
            e = self.colour(n.inputs["Emission Color"], drop) * self.scalar(n.inputs["Emission Strength"], drop)
            if e.max() > 0:
                out["emit"].append((w, e))
        elif t == "ShaderNodeHoldout":
            drop.add("holdout")
        else:
            drop.add(f"{t[10:]} shader as grey diffuse")
            out["diffuse"].append((w, np.array([0.8] * 3)))

    def reduce(self, mat):
        """`mat` as an irys material record."""
        drop = set()
        out = {"diffuse": [], "glossy": [], "transmit": [], "emit": [], "skin": False}
        if mat is None:
            rec = {"kind": "diffuse", "albedo": 0.8, "rgb": [0.8] * 3}
            return rec, ["no material"]
        if not mat.use_nodes:
            c = np.array(mat.diffuse_color[:3])
            return {"kind": "diffuse", "albedo": float(c @ LUM), "rgb": c.tolist()}, ["no nodes"]
        outs = [n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"]
        act = [n for n in outs if n.is_active_output] or outs
        if not act:
            return {"kind": "diffuse", "albedo": 0.8, "rgb": [0.8] * 3}, ["no output"]
        o = act[0]
        self.lobes(o.inputs["Surface"], 1.0, out, drop)
        if o.inputs["Volume"].is_linked:
            drop.add("volume")
        if o.inputs["Displacement"].is_linked:
            drop.add("displacement")
        for n in mat.node_tree.nodes:
            if n.bl_idname in ("ShaderNodeBump", "ShaderNodeNormalMap"):
                drop.add("bump/normal map")
                break
        wd = sum(w for w, _ in out["diffuse"])
        wg = sum(w for w, _, _ in out["glossy"])
        wt = sum(out["transmit"])
        cd = sum(w * c for w, c in out["diffuse"]) / wd if wd > 0 else np.zeros(3)
        cg = sum(w * c for w, c, _ in out["glossy"]) / wg if wg > 0 else np.zeros(3)
        rg = sum(w * r for w, _, r in out["glossy"]) / wg if wg > 0 else 0.5
        em = sum(w * c for w, c in out["emit"]) if out["emit"] else np.zeros(3)
        rec = {"weights": {"diffuse": wd, "glossy": wg, "transmit": wt}}
        # The diffuse albedo and glossy tint as reflected energy, weights in.
        dcol, gcol = wd * cd, wg * cg
        # Metal-like: the graph has (almost) no diffuse lobe. A glossy coat
        # over a diffuse base (Mix by Fresnel/Layer Weight, Principled
        # dielectric) is not a conductor however bright its highlight.
        metal = wg > 0 and wd <= 0.1 * (wd + wg)
        meas = self.measured.get(mat.name)
        if meas is not None and meas[2] >= 16 and not out["emit"] and wt < 0.5:
            # What Cycles shaded the camera's pixels with: its Diffuse and
            # Glossy Color passes are exactly these, textures, vertex
            # colours, procedurals and mix weights evaluated.
            rec["measured"] = {"diffuse": meas[0].tolist(), "glossy": meas[1].tolist(), "pixels": meas[2],
                               "reduced_diffuse": dcol.tolist(), "reduced_glossy": gcol.tolist()}
            dcol, gcol = meas[0], meas[1]
            wd, wg = float(dcol @ LUM > 0), float(gcol @ LUM > 0)
            cd, cg = dcol, gcol
        refl = wd * float(cd @ LUM) + wg * float(cg @ LUM)
        if float(em @ LUM) > max(refl, 1e-3):
            rec.update(kind="emitter", radiance=float(em @ LUM), rgb=em.tolist(), side=out.get("emit_side", "front"))
            if wd + wg > 0:
                drop.add("reflection under emission")
        elif wt >= 0.5:
            rec.update(kind="hidden", rgb=[1.0, 1.0, 1.0])
            drop.add("transmission (object hidden)")
        elif metal and float(cg @ LUM) > 0.1:
            rec.update(kind="conductor", rgb=cg.tolist(), roughness=float(rg), albedo=float(dcol @ LUM))
            if wd > 0:
                drop.add("diffuse under conductor")
        else:
            a = float(dcol @ LUM) if wd > 0 else 0.8
            rec.update(kind="diffuse", albedo=min(max(a, 0.0), 0.99), rgb=dcol.tolist())
            if wg > 0:
                drop.add("glossy over diffuse")
            if wt > 0:
                drop.add("partial transmission")
        if out["skin"]:
            rec["skin"] = True
            drop.add("subsurface (irys skin is its own renderer)")
        if rec["kind"] != "emitter" and out["emit"]:
            drop.add("weak emission")
        if float(np.max(np.abs(np.asarray(rec["rgb"]) - np.mean(rec["rgb"])))) > 0.02 and rec["kind"] in ("diffuse", "conductor"):
            drop.add("colour (irys albedo is grey)")
        return rec, sorted(drop)


_SKY = {}


def sky_node_equirect(node, w=64, h=32, spp=16):
    """A Sky Texture baked as a world of its own: an equirect (rows from +z
    down; column u at azimuth atan2(-x, -y) = 2 pi u, see `bake_world`)."""
    key = (node.id_data.name + "/" + node.name, w, h)
    if key in _SKY:
        return _SKY[key]
    wd = bpy.data.worlds.new("irys_sky_probe")
    wd.use_nodes = True
    nt = wd.node_tree
    sky = nt.nodes.new("ShaderNodeTexSky")
    for a in ("sky_type", "sun_direction", "turbidity", "ground_albedo", "sun_disc", "sun_size", "sun_intensity",
              "sun_elevation", "sun_rotation", "altitude", "air_density", "dust_density", "ozone_density"):
        if hasattr(node, a):
            try:
                setattr(sky, a, getattr(node, a))
            except (AttributeError, TypeError):
                pass
    bg = nt.nodes["Background"]
    nt.links.new(sky.outputs[0], bg.inputs[0])
    tmp = bpy.data.scenes.new("irys_sky_probe")
    tmp.world = wd
    px = bake_world(tmp, w, h, spp)
    bpy.data.scenes.remove(tmp)
    bpy.data.worlds.remove(wd)
    _SKY[key] = px
    return px


def sky_node_mean(node):
    """A Sky Texture's mean linear RGB over the upper hemisphere."""
    px = sky_node_equirect(node)
    up = px[: px.shape[0] // 2]
    el = (0.5 - (np.arange(up.shape[0]) + 0.5) / px.shape[0]) * math.pi
    wgt = np.cos(el)[:, None, None]
    return (up * wgt).sum(axis=(0, 1)) / (wgt.sum() * up.shape[1])


def light_node_factor(l, drop, name, rgb_out=None):
    """What a light's node tree multiplies its power by: the Emission node's
    strength (a Light Falloff feeding it gives its own strength) times its
    colour's luminance (a Blackbody is luminance 1). `rgb_out` (a list)
    receives the tint as linear RGB of luminance 1 (phase 5b)."""
    if not (l.use_nodes and l.node_tree):
        return 1.0
    em = [n for n in l.node_tree.nodes if n.bl_idname == "ShaderNodeEmission"]
    if not em:
        return 1.0
    e = em[0]
    st = e.inputs["Strength"]
    if st.is_linked:
        src = st.links[0].from_node
        f = src.inputs["Strength"].default_value if "Strength" in src.inputs and not src.inputs["Strength"].is_linked else 1.0
        if src.bl_idname != "ShaderNodeLightFalloff":
            drop.append(f"{name}: light strength node {src.bl_idname[10:]}")
    else:
        f = st.default_value
    col = e.inputs["Color"]
    if col.is_linked:
        src = col.links[0].from_node
        if src.bl_idname != "ShaderNodeBlackbody":
            drop.append(f"{name}: light colour node {src.bl_idname[10:]}")
        else:
            drop.append(f"{name}: blackbody tint (luminance kept; phase 5b: its colour)")
            if rgb_out is not None:
                rgb_out.extend(blend_openpbr.blackbody_rgb(src.inputs["Temperature"].default_value).tolist())
        c = 1.0
    else:
        c = float(np.array(col.default_value[:3]) @ LUM)
        if rgb_out is not None and c > 0:
            rgb_out.extend((np.array(col.default_value[:3]) / c).tolist())
    return f * c


def measure_colours(sc, w, h, spp=16):
    """Per material, the mean Diffuse Color and Glossy Color passes over the
    pixels the camera sees it in (Cycles evaluates the textures, vertex
    colours, procedurals and mix weights the reducer can only guess at).
    Returns {material name: (diffuse rgb, glossy rgb, pixels)}."""
    names = [m.name for m in bpy.data.materials]
    old = {m.name: m.pass_index for m in bpy.data.materials}
    for i, m in enumerate(bpy.data.materials):
        m.pass_index = i + 1
    vl = [v for v in sc.view_layers if v.use][0]
    vl.use_pass_diffuse_color = True
    vl.use_pass_glossy_color = True
    vl.use_pass_material_index = True
    rd = sc.render
    keep = (rd.engine, rd.resolution_x, rd.resolution_y, rd.resolution_percentage, sc.cycles.samples,
            sc.cycles.use_denoising, sc.cycles.use_adaptive_sampling, sc.cycles.filter_width, sc.use_nodes, rd.use_compositing)
    rd.engine = "CYCLES"
    rd.resolution_x, rd.resolution_y, rd.resolution_percentage = w, h, 100
    sc.cycles.samples = spp
    sc.cycles.use_denoising = False
    sc.cycles.use_adaptive_sampling = False
    sc.cycles.filter_width = 0.01
    sc.cycles.device = "CPU"
    sc.use_nodes = True
    rd.use_compositing = True
    tree = sc.node_tree
    saved_links = [(l.from_socket, l.to_socket) for l in tree.links]
    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.layer = vl.name
    fo = tree.nodes.new("CompositorNodeOutputFile")
    base = os.path.join(bpy.app.tempdir, "irys_passes")
    fo.base_path = base
    fo.format.file_format = "OPEN_EXR"
    fo.format.color_depth = "32"
    fo.file_slots.clear()
    for k in ("DiffCol", "GlossCol", "IndexMA"):
        fo.file_slots.new(k)
        tree.links.new(rl.outputs[k], fo.inputs[k])
    bpy.ops.render.render(write_still=False, scene=sc.name, layer=vl.name)
    out = {}
    if not os.path.isdir(base):
        print("irys export: the colour passes did not render; reducer colours only", flush=True)
    try:
        px = {}
        for k in ("DiffCol", "GlossCol", "IndexMA"):
            f = sorted(x for x in os.listdir(base) if x.startswith(k))[-1]
            img = bpy.data.images.load(os.path.join(base, f))
            px[k] = np.array(img.pixels[:], dtype=np.float64).reshape(-1, 4)[:, :3]
            bpy.data.images.remove(img)
        idx = px["IndexMA"][:, 0]
        ok = np.abs(idx - np.round(idx)) < 1e-3
        idx = np.round(idx).astype(np.int64)
        for i, n in enumerate(names):
            sel = ok & (idx == i + 1)
            c = int(sel.sum())
            if c:
                out[n] = (px["DiffCol"][sel].mean(axis=0), px["GlossCol"][sel].mean(axis=0), c)
    except (FileNotFoundError, IndexError):
        out = {}
    finally:
        import shutil
        shutil.rmtree(base, ignore_errors=True)
        tree.nodes.remove(fo)
        tree.nodes.remove(rl)
        (rd.engine, rd.resolution_x, rd.resolution_y, rd.resolution_percentage, sc.cycles.samples,
         sc.cycles.use_denoising, sc.cycles.use_adaptive_sampling, sc.cycles.filter_width, sc.use_nodes, rd.use_compositing) = keep
        for m in bpy.data.materials:
            m.pass_index = old[m.name]
    return out


def render_settings_like_render(sc):
    """Make the viewport depsgraph see what a render sees."""
    for o in bpy.data.objects:
        o.hide_viewport = o.hide_render
        if hasattr(o, "show_instancer_for_viewport"):
            # the emitter of a particle system shows itself in a render only
            # if it says so for renders (phase 5b; 5a missed it)
            o.show_instancer_for_viewport = o.show_instancer_for_render
        for m in o.modifiers:
            if hasattr(m, "render_levels") and hasattr(m, "levels"):
                m.levels = m.render_levels
            m.show_viewport = m.show_render
        for ps in o.particle_systems:
            s = ps.settings
            s.display_percentage = 100
            if hasattr(s, "rendered_child_count"):
                s.child_percent = s.rendered_child_count
    for c in bpy.data.collections:
        c.hide_viewport = c.hide_render

    def lc(layer):
        layer.hide_viewport = False
        for ch in layer.children:
            lc(ch)

    for vl in sc.view_layers:
        lc(vl.layer_collection)


def camera_record(sc, w):
    cam = sc.camera
    d = cam.data
    rx, ry = sc.render.resolution_x, sc.render.resolution_y
    h = int(round(w * ry / rx))
    aspect = (rx * sc.render.pixel_aspect_x) / (ry * sc.render.pixel_aspect_y)
    fit = d.sensor_fit
    if fit == "AUTO":
        fit = "HORIZONTAL" if aspect >= 1 else "VERTICAL"
    sensor = d.sensor_height if (fit == "VERTICAL" and d.sensor_fit != "AUTO") else d.sensor_width
    if d.lens_unit == "FOV" or True:
        half = math.atan(sensor / (2 * d.lens))
    tan_half = math.tan(half) / aspect if fit == "HORIZONTAL" else math.tan(half)
    m = cam.matrix_world
    eye = m.translation
    r = (m.to_3x3() @ Vector((1, 0, 0))).normalized()
    u = (m.to_3x3() @ Vector((0, 1, 0))).normalized()
    f = (m.to_3x3() @ Vector((0, 0, -1))).normalized()
    drop = []
    if d.type != "PERSP":
        drop.append(f"camera type {d.type}")
    if d.shift_x or d.shift_y:
        drop.append("lens shift")
    if d.dof.use_dof:
        drop.append("depth of field")
    return {
        "eye": list(eye), "forward": list(f), "right": list(r), "up": list(u),
        "tan_half": tan_half, "size": [w, h], "lens_mm": d.lens, "dropped": drop,
    }, h


def bake_world(sc, w=128, h=64, spp=64):
    """The world as an equirect (rows from +z down), via Cycles."""
    tmp = bpy.data.scenes.new("irys_world_bake")
    tmp.world = sc.world
    cd = bpy.data.cameras.new("irys_pano")
    cd.type = "PANO"
    try:
        cd.panorama_type = "EQUIRECTANGULAR"
    except AttributeError:
        cd.cycles.panorama_type = "EQUIRECTANGULAR"
    co = bpy.data.objects.new("irys_pano", cd)
    co.rotation_euler = (math.pi / 2, 0, 0)
    tmp.collection.objects.link(co)
    tmp.camera = co
    tmp.render.engine = "CYCLES"
    tmp.cycles.device = "CPU"
    tmp.cycles.samples = spp
    tmp.cycles.use_denoising = False
    tmp.cycles.use_adaptive_sampling = False
    tmp.render.resolution_x, tmp.render.resolution_y, tmp.render.resolution_percentage = w, h, 100
    tmp.render.image_settings.file_format = "OPEN_EXR"
    tmp.render.image_settings.color_depth = "32"
    tmp.view_settings.view_transform = "Standard"
    tmp.render.filepath = os.path.join(bpy.app.tempdir, "irys_world.exr")
    bpy.ops.render.render(write_still=True, scene=tmp.name)
    img = bpy.data.images.load(tmp.render.filepath)
    px = np.array(img.pixels[:], dtype=np.float64).reshape(h, w, 4)[::-1, :, :3]
    bpy.data.images.remove(img)
    bpy.data.scenes.remove(tmp)
    return px


def fit_env(sc, sun_records):
    if sc.world is None:
        return {"sky": 0.0, "sky_rgb": [0, 0, 0], "clipped": 0.0, "fit_rel_rms": 0.0}
    px = bake_world(sc)
    h, w, _ = px.shape
    lum = px @ LUM
    el = (0.5 - (np.arange(h) + 0.5) / h) * math.pi  # +pi/2 at the top row
    z = np.sin(el)[:, None] * np.ones((1, w))
    dw = np.cos(el)[:, None] * np.ones((1, w))  # solid angle per texel, up to a constant
    med = np.median(lum)
    cap = 20 * max(med, 1e-6)
    clipped = float((np.clip(lum - cap, 0, None) * dw).sum() / max((lum * dw).sum(), 1e-12))
    lc = np.minimum(lum, cap)
    g = 0.15 + 0.25 * np.maximum(z, 0)
    sky = float((lc * g * dw).sum() / (g * g * dw).sum())
    resid = lc - sky * g
    rel = float(np.sqrt((resid ** 2 * dw).sum() / max((lc ** 2 * dw).sum(), 1e-12)))
    rgb = (np.minimum(px, cap) * dw[..., None]).sum(axis=(0, 1)) / dw.sum()
    return {"sky": sky, "sky_rgb": rgb.tolist(), "clipped": clipped, "fit_rel_rms": rel, "median": float(med)}


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    out = argv[0]
    opts = dict(zip(argv[1::2], argv[2::2]))
    if "--scene" in opts:
        bpy.context.window.scene = bpy.data.scenes[opts["--scene"]]
    sc = bpy.context.scene
    width = int(opts.get("--width", 480))
    os.makedirs(out, exist_ok=True)
    render_settings_like_render(sc)
    cam0, h0 = camera_record(sc, width)
    mw = int(opts.get("--measure-width", min(width, 480)))
    mh = int(round(mw * h0 / width))
    measured = measure_colours(sc, 2 * mw, 2 * mh) if opts.get("--measure", "1") == "1" else {}
    dg = bpy.context.evaluated_depsgraph_get()
    red = Reducer(measured)

    # Materials, by name, reduced once.
    mats = {}  # name -> (glTF index, record)
    gl_mats = []

    def mat_key(m):
        return m.name if m else "__none__"

    def mat_index(m):
        key = mat_key(m)
        if key not in mats:
            rec, drop = red.reduce(m)
            rec["name"] = key
            rec["dropped"] = drop
            mats[key] = (len(gl_mats), rec)
            rgb = [float(min(max(c, 0), 1)) for c in rec["rgb"]]
            g = {"name": key, "pbrMetallicRoughness": {
                "baseColorFactor": rgb + [1.0],
                "metallicFactor": 1.0 if rec["kind"] == "conductor" else 0.0,
                "roughnessFactor": float(min(max(rec.get("roughness", 1.0), 0), 1)),
            }, "doubleSided": True}
            if rec["kind"] == "emitter":
                e = np.asarray(rec["rgb"])
                s = float(e.max())
                g["emissiveFactor"] = (e / s).tolist()
                g["extensions"] = {"KHR_materials_emissive_strength": {"emissiveStrength": s}}
            if rec["kind"] == "hidden":
                g["alphaMode"] = "BLEND"
                g["pbrMetallicRoughness"]["baseColorFactor"][3] = 0.1
            if look:
                look_material(m, key, rec, g, len(gl_mats))
            gl_mats.append(g)
        return mats[key][0]

    # Phase 5b: each material as OpenPBR, its colour fields baked to PNGs
    # under textures/ (glTF baseColorTexture + KHR_texture_transform; the
    # metal colour and opacity are named in the sidecar).
    gl_images, gl_textures = [], []
    look = opts.get("--look", "1") == "1"
    ev = blend_openpbr.Evaluator(env_bake=lambda n: sky_node_equirect(n, 512, 256, 16))
    tex_dir = os.path.join(out, "textures")

    def add_texture(path):
        gl_images.append({"uri": os.path.relpath(path, out)})
        gl_textures.append({"source": len(gl_images) - 1, "sampler": 0})
        return len(gl_textures) - 1

    def look_material(m, key, rec, g, index):
        params, tex, em, drop = blend_openpbr.openpbr(ev, m)
        meas = measured.get(key)
        approx = any(("as 0.5" in d or "as a constant" in d or "non-UV" in d or "its mean" in d) for d in drop)
        if meas is not None and meas[2] >= 16 and approx and params["base_metalness"] < 0.5 \
                and params["transmission_weight"] < 0.5 and params["translucency"] <= 0.0:
            # Inputs the UVs cannot carry (attributes, procedurals) were
            # constants; Cycles' Diffuse Color pass says what the camera's
            # pixels had. Scale the diffuse colour to that mean.
            have = np.asarray(params["base_color"], dtype=np.float64)
            if "base_color" in tex:
                have = tex["base_color"].mean()
            want = np.asarray(meas[0], dtype=np.float64) / max(1 - params["base_metalness"], 1e-3)
            k = np.clip(want / np.maximum(have, 1e-4), 1 / 3, 3.0)
            params["base_color"] = (np.asarray(params["base_color"]) * k).tolist()
            if "base_color" in tex:
                tex["base_color"] = tex["base_color"].map(lambda a, k=k: a * k)
            drop.append(f"diffuse colour scaled to Cycles' colour pass ({k.round(2).tolist()})")
        side = {}
        os.makedirs(tex_dir, exist_ok=True)
        for which in ("base_color", "metal_color"):
            f = tex.get(which)
            if f is None or f.img is None or f.xf == "equirect":
                continue
            path = os.path.join(tex_dir, f"{index:03d}_{which}.png")
            alpha = tex.get("opacity") if which == "base_color" else None
            blend_openpbr.save_texture(f, path, alpha)
            t = add_texture(path)
            side[which] = {"path": os.path.relpath(path, out), "texture": t, "xf": list(f.xf),
                           "alpha": alpha is not None and alpha.img is not None}
            if which == "base_color":
                g["pbrMetallicRoughness"]["baseColorTexture"] = {
                    "index": t, "extensions": {"KHR_texture_transform": {"scale": [f.xf[0], f.xf[1]], "offset": [f.xf[2], f.xf[3]]}}}
        op = tex.get("opacity")
        if op is not None and op.img is not None and "base_color" not in side:
            # Opacity alone: a white texture carrying it in alpha.
            white = blend_openpbr.Field(img=np.full(op.img.shape, 1.0), xf=op.xf)
            path = os.path.join(tex_dir, f"{index:03d}_opacity.png")
            blend_openpbr.save_texture(white, path, op)
            t = add_texture(path)
            side["opacity"] = {"path": os.path.relpath(path, out), "texture": t, "xf": list(op.xf), "alpha": True}
        if "opacity" in tex and tex["opacity"].img is not None:
            g["alphaMode"] = "MASK"
            g["alphaCutoff"] = 0.5
        rec["openpbr"] = params
        rec["textures"] = side
        rec["dropped_openpbr"] = drop
        if em is not None:
            rec["emission"] = {"rgb": em["rgb"], "side": em["side"]}
            if "equirect" in em:
                path = os.path.join(out, f"portal_{index:03d}.f32")
                e = np.ascontiguousarray(em["equirect"]).astype("<f4")
                e.tofile(path)
                rec["emission"]["equirect"] = {"path": os.path.basename(path), "size": [e.shape[1], e.shape[0]],
                                               "vector": "reflection"}

    # Unique meshes, keyed by evaluated data and the object's materials.
    bin_chunks = []
    offset = [0]
    accessors, views = [], []

    def add_view(arr, target=None):
        b = arr.tobytes()
        pad = (-len(b)) % 4
        views.append({"buffer": 0, "byteOffset": offset[0], "byteLength": len(b), **({"target": target} if target else {})})
        bin_chunks.append(b + b"\0" * pad)
        offset[0] += len(b) + pad
        return len(views) - 1

    def add_acc(arr, typ, comp, minmax=False, target=None):
        v = add_view(arr, target)
        a = {"bufferView": v, "componentType": comp, "count": int(arr.shape[0]), "type": typ}
        if minmax:
            a["min"] = arr.min(axis=0).tolist()
            a["max"] = arr.max(axis=0).tolist()
        accessors.append(a)
        return len(accessors) - 1

    meshes, gl_meshes = {}, []
    nodes = []
    tris_total, tris_unique = 0, 0
    skipped = {}
    zup_to_yup = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, -1, 0, 0), (0, 0, 0, 1)))
    lights = []
    portals = []
    portal_recs = []
    for oi in dg.object_instances:
        ob = oi.object
        if not oi.show_self:
            # A particle / geometry-node emitter whose own geometry the render
            # hides ("Show Emitter" off). Phase 5a exported these: the
            # pavilion's pool was covered by its lotus emitter plane.
            skipped["hidden instancers"] = skipped.get("hidden instancers", 0) + 1
            continue
        if ob.type == "LIGHT":
            lights.append((ob.data, oi.matrix_world.copy(), ob.name, bool(ob.visible_camera)))
            continue
        if ob.type not in ("MESH", "CURVE", "SURFACE", "META", "FONT", "CURVES") or ob.type == "CURVES":
            if ob.type in ("CURVES",):
                skipped["hair curves"] = skipped.get("hair curves", 0) + 1
            continue
        recs = [mats[mat_key(sl.material)][1] if mat_key(sl.material) in mats else red.reduce(sl.material)[0] for sl in ob.material_slots]
        if recs and all(r["kind"] == "emitter" for r in recs) and not ob.visible_camera and not ob.visible_shadow:
            # A light portal: an emitter no camera ray sees and no shadow
            # ray stops. irys's emitters are geometry that every ray sees,
            # so a portal at a window would shadow the sun it frames; its
            # light is folded into the environment instead.
            portals.append((ob.name, max(r["radiance"] for r in recs)))
            if look:
                # Phase 5b keeps the portal as what it is: a window onto its
                # emission that shadow and camera rays pass through.
                me = ob.to_mesh()
                me.calc_loop_triangles()
                mw_ = oi.matrix_world
                tri = [[list(mw_ @ me.vertices[v].co) for v in t.vertices] for t in me.loop_triangles]
                mi_ = ob.material_slots[0].material if ob.material_slots else None
                portal_recs.append({"name": ob.name, "material": mat_index(mi_), "triangles": tri})
                ob.to_mesh_clear()
            continue
        if not (ob.visible_camera and ob.visible_shadow and ob.visible_diffuse and ob.visible_glossy):
            # irys has no per-ray visibility: such an object is seen by
            # every ray (a light portal then shadows what it lights)
            skipped["ray visibility flags ignored"] = skipped.get("ray visibility flags ignored", 0) + 1
        slots = tuple((s.material.name if s.material else None) for s in ob.material_slots)
        dkey = ob.data.name if ob.type == "MESH" else "obj:" + ob.name
        key = (dkey, slots)
        if key not in meshes:
            try:
                me = ob.to_mesh()
            except RuntimeError:
                me = None
            if me is None or len(me.polygons) == 0:
                meshes[key] = None
                if me is not None:
                    ob.to_mesh_clear()
                continue
            me.calc_loop_triangles()
            nt = len(me.loop_triangles)
            loops = np.empty(nt * 3, dtype=np.int64)
            me.loop_triangles.foreach_get("loops", loops)
            vidx = np.empty(nt * 3, dtype=np.int64)
            me.loop_triangles.foreach_get("vertices", vidx)
            mi = np.empty(nt, dtype=np.int64)
            me.loop_triangles.foreach_get("material_index", mi)
            co = np.empty(len(me.vertices) * 3, dtype=np.float32)
            me.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3)[vidx]
            cn = np.empty(len(me.loops) * 3, dtype=np.float32)
            try:
                me.corner_normals.foreach_get("vector", cn)
            except AttributeError:
                me.calc_normals_split()
                me.loops.foreach_get("normal", cn)
            cn = cn.reshape(-1, 3)[loops]
            uv = None
            if me.uv_layers.active is not None:
                u = np.empty(len(me.loops) * 2, dtype=np.float32)
                me.uv_layers.active.data.foreach_get("uv", u)
                uv = u.reshape(-1, 2)[loops]
                uv[:, 1] = 1.0 - uv[:, 1]  # glTF's v runs down
            ob.to_mesh_clear()
            prims = []
            for m in np.unique(mi):
                sel = np.nonzero(mi == m)[0]
                corner = (sel[:, None] * 3 + np.arange(3)[None, :]).reshape(-1)
                mat = ob.material_slots[m].material if m < len(ob.material_slots) else None
                p = {"attributes": {
                    "POSITION": add_acc(np.ascontiguousarray(co[corner]), "VEC3", 5126, True, 34962),
                    "NORMAL": add_acc(np.ascontiguousarray(cn[corner]), "VEC3", 5126, False, 34962),
                }, "indices": add_acc(np.arange(len(corner), dtype=np.uint32), "SCALAR", 5125, False, 34963),
                    "material": mat_index(mat)}
                if uv is not None:
                    p["attributes"]["TEXCOORD_0"] = add_acc(np.ascontiguousarray(uv[corner]), "VEC2", 5126, False, 34962)
                prims.append(p)
            gl_meshes.append({"name": dkey, "primitives": prims})
            meshes[key] = (len(gl_meshes) - 1, nt)
            tris_unique += nt
        if meshes[key] is None:
            continue
        idx, nt = meshes[key]
        mw = zup_to_yup @ oi.matrix_world
        nodes.append({"mesh": idx, "matrix": [mw[r][c] for c in range(4) for r in range(4)]})
        tris_total += nt

    cam, h = camera_record(sc, width)
    m = zup_to_yup @ sc.camera.matrix_world
    nodes.append({"camera": 0, "name": "camera", "matrix": [m[r][c] for c in range(4) for r in range(4)]})

    # Lights.
    rec_lights, light_drop = [], []
    for l, mw, name, vis_cam in lights:
        tint = []
        col = float(np.array(l.color[:3]) @ LUM) * light_node_factor(l, light_drop, name, tint)
        lc = np.array(l.color[:3], dtype=np.float64)
        lc = lc / max(float(lc @ LUM), 1e-12)
        rgb = (lc * (np.asarray(tint) if tint else 1.0))
        rgb = (rgb / max(float(rgb @ LUM), 1e-12)).tolist()
        n_before = len(rec_lights)
        if l.type == "SUN":
            d = (mw.to_3x3() @ Vector((0, 0, 1))).normalized()
            r = max(l.angle / 2, 1e-3)
            rec_lights.append({"name": name, "kind": "sun", "dir": list(d), "irradiance": l.energy * col, "radius": r})
            if l.angle / 2 < 1e-3:
                light_drop.append(f"{name}: sun angle below 1 mrad, widened")
        elif l.type == "AREA":
            sx = l.size
            sy = l.size if l.shape in ("SQUARE", "DISK") else l.size_y
            disk = l.shape in ("DISK", "ELLIPSE")
            area = (math.pi / 4 if disk else 1.0) * sx * sy
            L = l.energy * col / (math.pi * area)
            m3 = mw.to_3x3()
            sc_ = Vector((m3.col[0].length, m3.col[1].length))
            rec_lights.append({"name": name, "kind": "disk" if disk else "quad", "center": list(mw.translation),
                               "u": list(m3.col[0].normalized() * sx * sc_[0] / 2), "v": list(m3.col[1].normalized() * sy * sc_[1] / 2),
                               "normal": list((m3 @ Vector((0, 0, -1))).normalized()), "radiance": L})
            if getattr(l, "spread", math.pi) < math.pi - 1e-3:
                light_drop.append(f"{name}: area spread")
        elif l.type in ("POINT", "SPOT"):
            r = max(l.shadow_soft_size, 0.02)
            L = l.energy * col / (4 * math.pi ** 2 * r * r)
            if l.type == "SPOT":
                # a spot is a point light restricted to a cone: the same radiance, less power
                light_drop.append(f"{name}: spot cone (5a: emits all round; 5b honours it)")
            rec_lights.append({"name": name, "kind": "sphere", "center": list(mw.translation), "radius": r, "radiance": L})
            if l.type == "SPOT":
                rec_lights[-1]["spot"] = {"axis": list((mw.to_3x3() @ Vector((0, 0, -1))).normalized()),
                                          "size": l.spot_size, "blend": l.spot_blend}
            if l.shadow_soft_size < 0.02:
                light_drop.append(f"{name}: radius {l.shadow_soft_size:.3f} widened to 0.02")
        for rl in rec_lights[n_before:]:
            rl["rgb"] = rgb  # linear Rec.709, luminance 1 (phase 5b)
            rl["visible_camera"] = vis_cam
    env = fit_env(sc, rec_lights)
    if look and sc.world is not None:
        # Phase 5b: the world itself, as an equirect irys importance-samples.
        ew, eh = int(opts.get("--env-width", 1024)), int(opts.get("--env-width", 1024)) // 2
        px = bake_world(sc, ew, eh, int(opts.get("--env-spp", 16)))
        px = np.ascontiguousarray(px).astype("<f4")
        px.tofile(os.path.join(out, "env.f32"))
        env["equirect"] = {"path": "env.f32", "size": [ew, eh], "mean": px.reshape(-1, 3).mean(axis=0).tolist()}
    if portals:
        # The gradient seen through a window (z ~ 0 .. 0.5) is ~0.2 sky.
        lp = max(L for _, L in dict(portals).items())
        env["portal_sky"] = lp / 0.2
        if env["sky"] < env["portal_sky"]:
            env["sky"] = env["portal_sky"]
        light_drop.append(f"light portals {sorted(set(n for n, _ in portals))} folded into the sky ({lp:.2f} -> sky {lp / 0.2:.2f})")

    gltf = {
        "asset": {"version": "2.0", "generator": "irys scripts/blend_export.py"},
        "extensionsUsed": ["KHR_materials_emissive_strength"] + (["KHR_texture_transform"] if gl_images else []),
        "scene": 0, "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes, "meshes": gl_meshes, "materials": gl_mats,
        "cameras": [{"type": "perspective", "perspective": {
            "yfov": 2 * math.atan(cam["tan_half"]), "aspectRatio": width / h, "znear": sc.camera.data.clip_start}}],
        "accessors": accessors, "bufferViews": views,
        **({"images": gl_images, "textures": gl_textures, "samplers": [{"wrapS": 10497, "wrapT": 10497}]} if gl_images else {}),
        "buffers": [{"byteLength": offset[0]}],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    blob = b"".join(bin_chunks)
    with open(os.path.join(out, "scene.glb"), "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(blob), 0x004E4942) + blob)

    dropped = {}
    for _, rec in mats.values():
        for d in rec["dropped"]:
            dropped[d] = dropped.get(d, 0) + 1
    side = {
        "source": bpy.data.filepath, "blender": bpy.app.version_string, "scene": sc.name,
        "camera": cam, "lights": rec_lights, "env": env,
        "materials": [rec for _, rec in sorted(mats.values(), key=lambda x: x[0])],
        "stats": {"instances": len(nodes) - 1, "unique_meshes": len(gl_meshes), "triangles_unique": tris_unique,
                  "triangles_instanced": tris_total, "materials": len(gl_mats), "skipped_objects": skipped},
        "dropped": {"materials": dropped, "lights": light_drop, "camera": cam["dropped"],
                    "world": [f"clipped {env['clipped']:.1%} of the world's power (sun disc / hot texels)",
                              f"gradient fit rel rms {env['fit_rel_rms']:.2f}", "world colour (irys sky tint is fixed)"]},
        "cycles": {"samples": sc.cycles.samples, "max_bounces": sc.cycles.max_bounces},
    }
    if look:
        dl = {}
        for _, rec in mats.values():
            for d in rec.get("dropped_openpbr", []):
                dl[d] = dl.get(d, 0) + 1
        side["dropped"]["openpbr"] = dl
        side["portals"] = portal_recs
        side["openpbr_version"] = 1
    if "--frames" in opts:
        # The file's camera, frame by frame (an animated camera's keyframes,
        # evaluated): the reel's camera moves.
        a, b, step = (int(x) for x in (opts["--frames"].split(":") + ["1"])[:3])
        frames = []
        for fr in range(a, b + 1, step):
            sc.frame_set(fr)
            c, _ = camera_record(sc, width)
            frames.append({"frame": fr, **{k: c[k] for k in ("eye", "forward", "right", "up", "tan_half")}})
        side["camera_frames"] = {"fps": sc.render.fps / sc.render.fps_base, "frames": frames}
    if "--reel" in opts:
        # The camera resampled to another frame rate (subframes): the reel's
        # 30 fps from a 24/25 fps animation, frames a..b of the file.
        rfps, a, b = (float(x) for x in opts["--reel"].split(":"))
        sfps = sc.render.fps / sc.render.fps_base
        n = int(round((b - a) / sfps * rfps)) + 1
        frames = []
        for k in range(n):
            t = a + k * sfps / rfps
            sc.frame_set(int(math.floor(t)), subframe=t - math.floor(t))
            c, _ = camera_record(sc, width)
            frames.append({"t": t, **{key: c[key] for key in ("eye", "forward", "right", "up", "tan_half")}})
        side["reel_frames"] = {"fps": rfps, "from": a, "to": b, "scene_fps": sfps, "frames": frames}
    json.dump(side, open(os.path.join(out, "irys.json"), "w"), indent=1)
    print(f"irys export: {len(nodes) - 1} instances of {len(gl_meshes)} meshes, {tris_unique} unique / {tris_total} placed triangles, "
          f"{len(gl_mats)} materials, {len(rec_lights)} lights, sky {env['sky']:.3f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
