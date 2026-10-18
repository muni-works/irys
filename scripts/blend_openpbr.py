# Phase 5b: Blender materials as OpenPBR Surface parameters, with baked
# textures. Imported by scripts/blend_export.py (Blender's own Python, numpy).
#
# Blender 4.x's Principled BSDF follows OpenPBR, so it maps 1:1. The older
# graphs the demo files are built from map to the closest OpenPBR material:
#
#   Diffuse(c)                         base_color c, specular_weight 0
#   Mix(k, Diffuse c, Glossy g, r)     base_color c, metalness k, metal colour g
#     (constant k: an energy mix of a  (OpenPBR's metalness *is* that mix; the
#      Lambertian and a GGX lobe)       metal's F0 gets its own colour, see below)
#   Mix(Fresnel/LayerWeight, D, G)     dielectric: base_color c, specular_weight 1,
#                                      specular_color g, specular_ior from the
#                                      Fresnel IOR (LayerWeight blend b: 1/(1 − b))
#   Glass(c, r, n) / Refraction        transmission_weight 1, transmission_color c,
#                                      specular_roughness r, specular_ior n
#   Add/Mix(Glossy, Transparent)       thin-walled transmission (straight through,
#                                      shadow rays pass), the glossy as specular
#   Mix(image alpha, Transparent, X)   a cut-out: geometry_opacity from the image
#   Translucent(c)                     thin diffuse transmission (OpenPBR's
#                                      thin-walled subsurface), colour c
#   Emission(c, s)                     emission colour c, luminance s (radiance)
#
# One extension of OpenPBR's parameter set is needed: `metal_color`, the F0
# of the metal slab, which OpenPBR ties to base_color. A Blender Mix of a
# Diffuse (colour c) and a Glossy (colour g) is exactly OpenPBR's metalness
# mix with the two slabs' colours apart; irys keeps both.
#
# Colours are fields: a constant, or an image on the mesh's UVs. A socket is
# evaluated through the image's chain (Mix in its blend modes, RGB Curves,
# Hue/Saturation/Value, Bright/Contrast, Gamma, Invert, Color Ramp, Math,
# Mapping's scale and location) into a baked texture, so the texture irys
# samples is the one Cycles would have shaded with wherever only the UVs
# matter. Inputs that do not depend on the UVs (Attribute, Object Info,
# Noise/Wave/Voronoi in generated coordinates, Geometry) become their
# constant mean, and are reported.

import colorsys
import math
import os

import bpy
import numpy as np

LUM = np.array([0.2126, 0.7152, 0.0722])
MAX_TEX = int(os.environ.get("IRYS_MAX_TEX", "1024"))
PASS = {"NodeReroute"}


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


class Field:
    """A colour over the UVs: `c` (3,) constant, or `img` (h, w, 3) with
    the glTF-UV transform `xf` = (sx, sy, ox, oy): texel = uv · s + o."""

    def __init__(self, c=None, img=None, xf=None):
        self.c = None if c is None else np.asarray(c, dtype=np.float64).reshape(3)
        self.img = img
        self.xf = xf

    @staticmethod
    def k(v):
        return Field(np.array([v, v, v], dtype=np.float64))

    def is_const(self):
        return self.img is None

    def mean(self):
        return self.c if self.img is None else self.img.reshape(-1, 3).mean(axis=0)

    def scalar_mean(self):
        return float(self.mean() @ np.array([1 / 3, 1 / 3, 1 / 3]))

    def arr(self):
        return self.c if self.img is None else self.img

    def map(self, fn):
        if self.img is None:
            return Field(fn(self.c[None, None, :])[0, 0])
        return Field(img=fn(self.img), xf=self.xf)


def combine(fields, fn, drop):
    """Apply `fn` to the fields' arrays, broadcasting constants. Images
    with different shapes are resampled to the first; with different UV
    transforms, the later ones become their means."""
    base = next((f for f in fields if f.img is not None), None)
    if base is None:
        return Field(fn(*[f.c for f in fields]))
    h, w, _ = base.img.shape
    arrs = []
    for f in fields:
        if f.img is None:
            arrs.append(f.c[None, None, :])
        elif f.xf != base.xf:
            drop.add("two UV mappings in one chain (one as its mean)")
            arrs.append(f.mean()[None, None, :])
        elif f.img.shape != base.img.shape:
            arrs.append(resample(f.img, h, w))
        else:
            arrs.append(f.img)
    out = fn(*arrs)
    out = np.broadcast_to(out, (h, w, 3)).astype(np.float64)
    return Field(img=out, xf=base.xf)


def resample(img, h, w):
    sh, sw, _ = img.shape
    ys = np.minimum((np.arange(h) + 0.5) * sh / h, sh - 1).astype(np.int64)
    xs = np.minimum((np.arange(w) + 0.5) * sw / w, sw - 1).astype(np.int64)
    return img[ys][:, xs]


def lut(fn, n=256):
    xs = np.linspace(0, 1, n)
    return np.array([fn(float(x)) for x in xs])


def apply_lut(table, x):
    n = len(table)
    s = np.clip(x, 0.0, 1.0) * (n - 1)
    i = np.minimum(np.floor(s).astype(np.int64), n - 2)
    t = s - i
    return table[i] * (1 - t) + table[i + 1] * t


def rgb_to_hsv(a):
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = np.max(a, axis=-1)
    mn = np.min(a, axis=-1)
    d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-12
    rc = np.where(m, (mx - r) / np.where(m, d, 1), 0)
    gc = np.where(m, (mx - g) / np.where(m, d, 1), 0)
    bc = np.where(m, (mx - b) / np.where(m, d, 1), 0)
    h = np.where(r == mx, bc - gc, np.where(g == mx, 2.0 + rc - bc, 4.0 + gc - rc))
    h = np.where(m, (h / 6.0) % 1.0, 0.0)
    s = np.where(mx > 1e-12, d / np.where(mx > 1e-12, mx, 1), 0)
    return np.stack([h, s, mx], axis=-1)


def hsv_to_rgb(a):
    h, s, v = a[..., 0] % 1.0, a[..., 1], a[..., 2]
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    i = i.astype(np.int64) % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def blackbody_rgb(t_k):
    """Linear Rec.709 colour of a blackbody at `t_k`, luminance 1 (Cycles
    normalises its Blackbody node the same way)."""
    lam = np.linspace(380, 780, 401) * 1e-9
    c1, c2 = 3.741771852e-16, 1.438776877e-2
    b = c1 / (lam ** 5 * (np.exp(c2 / (lam * t_k)) - 1))
    l = lam * 1e9

    def g(x, mu, s1, s2):
        s = np.where(x < mu, s1, s2)
        return np.exp(-0.5 * ((x - mu) / s) ** 2)

    x = 1.056 * g(l, 599.8, 37.9, 31.0) + 0.362 * g(l, 442.0, 16.0, 26.7) - 0.065 * g(l, 501.1, 20.4, 26.2)
    y = 0.821 * g(l, 568.8, 46.9, 40.5) + 0.286 * g(l, 530.9, 16.3, 31.1)
    z = 1.217 * g(l, 437.0, 11.8, 36.0) + 0.681 * g(l, 459.0, 26.0, 13.8)
    X, Y, Z = (b * x).sum(), (b * y).sum(), (b * z).sum()
    rgb = np.array([3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z,
                    -0.969266 * X + 1.8760108 * Y + 0.041556 * Z,
                    0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z])
    rgb = np.maximum(rgb, 0)
    return rgb / max(float(rgb @ LUM), 1e-12)


class Evaluator:
    """Socket fields, with a log of approximations per material."""

    def __init__(self, env_bake=None):
        self.cache = {}
        self.env_bake = env_bake  # fn(sky node) -> equirect (h, w, 3)

    # ── images ──
    def image(self, img, drop):
        if img is None:
            return None
        key = img.name
        if key in self.cache:
            return self.cache[key]
        px = None
        try:
            w, h = img.size
            if w > 0 and h > 0:
                s = max(w, h)
                if s > MAX_TEX:
                    k = MAX_TEX / s
                    small = img.copy()
                    small.scale(max(1, int(w * k)), max(1, int(h * k)))
                    src = small
                else:
                    src, small = img, None
                ww, hh = src.size
                buf = np.empty(ww * hh * 4, dtype=np.float32)
                src.pixels.foreach_get(buf)
                px = buf.reshape(hh, ww, 4).astype(np.float64)[::-1]  # top row first (glTF v down)
                if small is not None:
                    bpy.data.images.remove(small)
                if img.colorspace_settings.name == "sRGB" and not img.is_float:
                    px[..., :3] = srgb_to_linear(px[..., :3])
        except Exception as e:  # noqa: BLE001
            drop.add(f"image {img.name} unreadable ({type(e).__name__})")
            px = None
        self.cache[key] = px
        return px

    def uv_xf(self, sock, drop):
        """The glTF-UV transform an image's Vector input implies, or None
        when it is not the UVs (then the image is used as its mean)."""
        if not sock.is_linked:
            return (1.0, 1.0, 0.0, 0.0)
        n = sock.links[0].from_node
        s = sock.links[0].from_socket.name
        if n.bl_idname == "ShaderNodeTexCoord" and s == "UV":
            return (1.0, 1.0, 0.0, 0.0)
        if n.bl_idname == "ShaderNodeUVMap":
            return (1.0, 1.0, 0.0, 0.0)
        if n.bl_idname == "ShaderNodeMapping" and n.vector_type in ("POINT", "TEXTURE"):
            inner = self.uv_xf(n.inputs["Vector"], drop)
            if inner is None:
                return None
            sc = n.inputs["Scale"].default_value
            lo = n.inputs["Location"].default_value
            rot = n.inputs["Rotation"].default_value
            if abs(rot[2]) > 1e-6:
                drop.add("mapping rotation")
            sx, sy, ox, oy = sc[0], sc[1], lo[0], lo[1]
            if n.vector_type == "TEXTURE":
                sx, sy = 1 / sx, 1 / sy
                ox, oy = -ox * sx, -oy * sy
            # Blender's v up, glTF's down: v_b' = sy v_b + oy, v_b = 1 − v_g.
            isx, isy, iox, ioy = inner
            return (isx * sx, isy * sy, iox * sx + ox, 1.0 - sy - oy + ioy * sy)
        return None

    def tex_image(self, n, out_name, drop):
        px = self.image(n.image, drop)
        if px is None:
            return Field.k(0.8)
        if n.projection != "FLAT":
            drop.add(f"image projection {n.projection} as its mean")
            xf = None
        else:
            xf = self.uv_xf(n.inputs["Vector"], drop)
        if out_name == "Alpha":
            a = np.repeat(px[..., 3:4], 3, axis=-1)
        else:
            a = px[..., :3]
        if xf is None:
            if n.inputs["Vector"].is_linked:
                drop.add("image on non-UV coordinates as its mean")
            return Field(a.reshape(-1, 3).mean(axis=0))
        return Field(img=a, xf=xf)

    # ── sockets ──
    def colour(self, sock, drop):
        if not sock.is_linked:
            v = sock.default_value
            try:
                return Field(np.array([v[0], v[1], v[2]], dtype=np.float64))
            except TypeError:
                return Field.k(float(v))
        link = sock.links[0]
        n, out = link.from_node, link.from_socket
        while n.bl_idname in PASS:
            if not n.inputs[0].is_linked:
                return Field.k(0.8)
            link = n.inputs[0].links[0]
            n, out = link.from_node, link.from_socket
        return self.node(n, out.name, drop)

    def scalar(self, sock, drop):
        f = self.colour(sock, drop)
        return f.map(lambda a: np.repeat(np.mean(a, axis=-1, keepdims=True), 3, axis=-1)) \
            if not _is_grey(f) else f

    def node(self, n, out, drop):
        t = n.bl_idname
        if t == "ShaderNodeTexImage":
            return self.tex_image(n, out, drop)
        if t == "ShaderNodeRGB":
            return Field(np.array(n.outputs[0].default_value[:3], dtype=np.float64))
        if t == "ShaderNodeValue":
            return Field.k(float(n.outputs[0].default_value))
        if t in ("ShaderNodeMixRGB", "ShaderNodeMix"):
            return self.mix(n, drop)
        if t == "ShaderNodeHueSaturation":
            hue = self.scalar(n.inputs["Hue"], drop).scalar_mean()
            sat = self.scalar(n.inputs["Saturation"], drop).scalar_mean()
            val = self.scalar(n.inputs["Value"], drop).scalar_mean()
            fac = self.scalar(n.inputs["Fac"], drop)
            c = self.colour(n.inputs["Color"], drop)

            def hsv(a):
                x = rgb_to_hsv(np.maximum(a, 0))
                x[..., 0] = (x[..., 0] + hue + 0.5) % 1.0
                x[..., 1] = np.clip(x[..., 1] * sat, 0, 1)
                x[..., 2] = x[..., 2] * val
                return hsv_to_rgb(x)

            return combine([c, fac], lambda a, f: (1 - f) * a + f * hsv(np.broadcast_to(a, np.broadcast(a, f).shape)), drop)
        if t == "ShaderNodeBrightContrast":
            b = self.scalar(n.inputs["Bright"], drop).scalar_mean()
            k = self.scalar(n.inputs["Contrast"], drop).scalar_mean()
            c = self.colour(n.inputs["Color"], drop)
            return c.map(lambda a: np.maximum((1 + k) * a + (b - k / 2), 0))
        if t == "ShaderNodeGamma":
            g = self.scalar(n.inputs["Gamma"], drop).scalar_mean()
            return self.colour(n.inputs["Color"], drop).map(lambda a: np.power(np.maximum(a, 0), g))
        if t == "ShaderNodeInvert":
            fac = self.scalar(n.inputs["Fac"], drop)
            c = self.colour(n.inputs["Color"], drop)
            return combine([c, fac], lambda a, f: (1 - f) * a + f * (1 - a), drop)
        if t == "ShaderNodeRGBCurve":
            m = n.mapping
            m.initialize()
            cc = lut(lambda x: m.evaluate(m.curves[3], x))
            per = [lut(lambda x, i=i: m.evaluate(m.curves[i], x)) for i in range(3)]
            fac = self.scalar(n.inputs["Fac"], drop)
            c = self.colour(n.inputs["Color"], drop)

            def curve(a):
                o = np.empty(np.broadcast(a, np.zeros(3)).shape)
                a = np.broadcast_to(a, o.shape)
                for i in range(3):
                    o[..., i] = apply_lut(cc, apply_lut(per[i], a[..., i]))
                return o

            return combine([c, fac], lambda a, f: (1 - f) * a + f * curve(a), drop)
        if t == "ShaderNodeValToRGB":
            r = n.color_ramp
            tab = lut(lambda x: list(r.evaluate(x))[:3])
            fs = n.inputs["Fac"]
            if fs.is_linked and fs.links[0].from_node.bl_idname == "ShaderNodeObjectInfo" \
                    and fs.links[0].from_socket.name == "Random":
                # A per-object random colour: the ramp's mean over [0, 1].
                drop.add("object random colour as its ramp mean")
                if out == "Alpha":
                    return Field.k(float(lut(lambda x: r.evaluate(x)[3]).mean()))
                return Field(tab.mean(axis=0))
            f = self.scalar(n.inputs["Fac"], drop)
            if out == "Alpha":
                ta = lut(lambda x: r.evaluate(x)[3])
                return f.map(lambda a: np.repeat(apply_lut(ta, a[..., 0])[..., None], 3, axis=-1))
            return f.map(lambda a: np.stack([apply_lut(tab[:, i], a[..., 0]) for i in range(3)], axis=-1))
        if t == "ShaderNodeMath":
            return self.math(n, drop)
        if t == "ShaderNodeRGBToBW":
            c = self.colour(n.inputs["Color"], drop)
            return c.map(lambda a: np.repeat((a @ LUM)[..., None], 3, axis=-1))
        if t in ("ShaderNodeSeparateColor", "ShaderNodeSeparateRGB"):
            c = self.colour(n.inputs[0], drop)
            i = {"Red": 0, "R": 0, "Green": 1, "G": 1, "Blue": 2, "B": 2}.get(out, 0)
            return c.map(lambda a: np.repeat(a[..., i:i + 1], 3, axis=-1))
        if t == "ShaderNodeBlackbody":
            k = self.scalar(n.inputs["Temperature"], drop).scalar_mean()
            return Field(blackbody_rgb(k))
        if t == "ShaderNodeTexSky" and self.env_bake is not None:
            e = self.env_bake(n)
            return Field(img=e, xf="equirect")
        if t == "ShaderNodeLayerWeight" or t == "ShaderNodeFresnel":
            drop.add("fresnel/layer weight as a factor: 0.2")
            return Field.k(0.2)
        if t == "ShaderNodeAttribute":
            drop.add("attribute colour as 0.5 grey")
            return Field.k(0.5)
        if t == "ShaderNodeObjectInfo":
            drop.add("object info as 0.5")
            return Field.k(0.5)
        drop.add(f"{t[10:]} as a constant")
        return Field.k(0.5)

    def mix(self, n, drop):
        if n.bl_idname == "ShaderNodeMixRGB":
            f, a, b = n.inputs[0], n.inputs[1], n.inputs[2]
            clamp = n.use_clamp
        else:
            dt = getattr(n, "data_type", "RGBA")
            if dt == "FLOAT":
                f, a, b = n.inputs[0], n.inputs[2], n.inputs[3]
            elif dt == "VECTOR":
                drop.add("vector mix as a constant")
                return Field.k(0.5)
            else:
                f, a, b = n.inputs[0], n.inputs[6], n.inputs[7]
            clamp = getattr(n, "clamp_result", False)
        fac = self.scalar(f, drop)
        ca, cb = self.colour(a, drop), self.colour(b, drop)
        bt = getattr(n, "blend_type", "MIX")

        def blend(x, y, k):
            if bt == "MIX":
                o = (1 - k) * x + k * y
            elif bt == "MULTIPLY":
                o = x * (1 - k + k * y)
            elif bt == "ADD":
                o = x + k * y
            elif bt == "SUBTRACT":
                o = x - k * y
            elif bt == "SCREEN":
                o = 1 - (1 - k + k * (1 - y)) * (1 - x)
            elif bt == "OVERLAY":
                o = np.where(x < 0.5, x * (1 - k + 2 * k * y), 1 - (1 - k + 2 * k * (1 - y)) * (1 - x))
            elif bt == "DARKEN":
                o = (1 - k) * x + k * np.minimum(x, y)
            elif bt == "LIGHTEN":
                o = (1 - k) * x + k * np.maximum(x, y)
            elif bt == "DIFFERENCE":
                o = (1 - k) * x + k * np.abs(x - y)
            elif bt == "SOFT_LIGHT":
                scr = 1 - (1 - y) * (1 - x)
                o = (1 - k) * x + k * ((1 - x) * y * x + x * scr)
            else:
                drop.add(f"mix {bt.lower()} as mix")
                o = (1 - k) * x + k * y
            return np.clip(o, 0, 1) if clamp else o

        return combine([ca, cb, fac], blend, drop)

    def math(self, n, drop):
        op = n.operation
        a = self.scalar(n.inputs[0], drop)
        b = self.scalar(n.inputs[1], drop)
        ops = {
            "ADD": lambda x, y: x + y, "SUBTRACT": lambda x, y: x - y, "MULTIPLY": lambda x, y: x * y,
            "DIVIDE": lambda x, y: x / np.where(np.abs(y) > 1e-12, y, 1e-12),
            "POWER": lambda x, y: np.power(np.maximum(x, 0), y), "MINIMUM": np.minimum, "MAXIMUM": np.maximum,
            "GREATER_THAN": lambda x, y: (x > y).astype(np.float64), "LESS_THAN": lambda x, y: (x < y).astype(np.float64),
            "ABSOLUTE": lambda x, y: np.abs(x), "SQRT": lambda x, y: np.sqrt(np.maximum(x, 0)),
        }
        if op not in ops:
            drop.add(f"math {op.lower()} as its first input")
            fn = lambda x, y: x  # noqa: E731
        else:
            fn = ops[op]
        out = combine([a, b], fn, drop)
        if n.use_clamp:
            out = out.map(lambda x: np.clip(x, 0, 1))
        return out


def _is_grey(f):
    a = f.arr()
    return np.allclose(a[..., 0], a[..., 1]) and np.allclose(a[..., 1], a[..., 2])


# ── shader graphs → lobes ────────────────────────────────────────────────

def fresnel_factor_ior(sock):
    """If `sock` is a Fresnel / Layer Weight factor, its IOR."""
    if not sock.is_linked:
        return None
    n = sock.links[0].from_node
    if n.bl_idname == "ShaderNodeFresnel":
        return float(n.inputs["IOR"].default_value)
    if n.bl_idname == "ShaderNodeLayerWeight":
        b = float(n.inputs["Blend"].default_value)
        return min(max(1.0 / max(1.0 - b, 1e-3), 1.01), 4.0)
    return None


def backfacing(sock):
    if not sock.is_linked:
        return False
    l = sock.links[0]
    return l.from_node.bl_idname == "ShaderNodeNewGeometry" and l.from_socket.name == "Backfacing"


def lobes(ev, sock, w, out, drop, fres=None, depth=0):
    """Accumulate (kind, weight field, ...) reached from shader `sock`."""
    if not sock.is_linked or depth > 32:
        return
    n = sock.links[0].from_node
    while n.bl_idname in PASS:
        n = n.inputs[0].links[0].from_node
    t = n.bl_idname
    if t == "ShaderNodeMixShader" and backfacing(n.inputs[0]):
        # front: input 1, back: input 2. irys surfaces are two-sided, so keep
        # the emitting side if one emits (one-sided emitter), else the front.
        f, b = [], []
        lobes(ev, n.inputs[1], w, f, drop, fres, depth + 1)
        lobes(ev, n.inputs[2], w, b, drop, fres, depth + 1)
        fe = any(x[0] == "emit" for x in f)
        be = any(x[0] == "emit" for x in b)
        if be and not fe:
            out.append(("side", "back"))
            out.extend(b)
        else:
            out.append(("side", "front"))
            out.extend(f)
        drop.add("backfacing mix: one side kept")
    elif t == "ShaderNodeMixShader":
        ior = fresnel_factor_ior(n.inputs[0])
        if ior is not None:
            # A Fresnel-weighted layer: the second shader on top of the first.
            lobes(ev, n.inputs[1], w, out, drop, fres, depth + 1)
            lobes(ev, n.inputs[2], w, out, drop, ior, depth + 1)
            if n.inputs[0].links[0].from_socket.name == "Facing":
                drop.add("layer weight facing as fresnel")
        else:
            k = ev.scalar(n.inputs[0], drop)
            k = k.map(lambda a: np.clip(a, 0, 1))
            lobes(ev, n.inputs[1], combine([w, k], lambda a, b: a * (1 - b), drop), out, drop, fres, depth + 1)
            lobes(ev, n.inputs[2], combine([w, k], lambda a, b: a * b, drop), out, drop, fres, depth + 1)
    elif t == "ShaderNodeAddShader":
        # Both inputs get the whole weight; "addx" remembers the excess so
        # the energy partition counts this node's weight once.
        k = len(out)
        lobes(ev, n.inputs[0], w, out, drop, fres, depth + 1)
        k1 = len(out)
        lobes(ev, n.inputs[1], w, out, drop, fres, depth + 1)
        solid = ("diffuse", "translucent", "metal", "glass", "transparent")
        if any(x[0] in solid for x in out[k:k1]) and any(x[0] in solid for x in out[k1:]):
            out.append(("addx", w))
    elif t == "ShaderNodeBsdfDiffuse" or t == "ShaderNodeBsdfToon":
        if t != "ShaderNodeBsdfDiffuse":
            drop.add("toon as diffuse")
        if "Roughness" in n.inputs and (n.inputs["Roughness"].is_linked or n.inputs["Roughness"].default_value > 0):
            drop.add("oren-nayar roughness")
        out.append(("diffuse", w, ev.colour(n.inputs["Color"], drop)))
    elif t == "ShaderNodeBsdfTranslucent":
        out.append(("translucent", w, ev.colour(n.inputs["Color"], drop)))
    elif t == "ShaderNodeBsdfSheen":
        out.append(("fuzz", w, ev.colour(n.inputs["Color"], drop),
                    ev.scalar(n.inputs["Roughness"], drop).scalar_mean()))
    elif t in ("ShaderNodeBsdfGlossy", "ShaderNodeBsdfAnisotropic", "ShaderNodeBsdfMetallic"):
        if "Anisotropy" in n.inputs and (n.inputs["Anisotropy"].is_linked or abs(n.inputs["Anisotropy"].default_value) > 1e-3):
            drop.add("anisotropy")
        r = ev.scalar(n.inputs["Roughness"], drop).scalar_mean() if "Roughness" in n.inputs else 0.2
        if "Roughness" in n.inputs and n.inputs["Roughness"].is_linked:
            drop.add("roughness texture as its mean")
        col = ev.colour(n.inputs["Color"], drop) if "Color" in n.inputs else (
            ev.colour(n.inputs["Base Color"], drop) if "Base Color" in n.inputs else Field.k(0.9))
        out.append(("spec" if fres else "metal", w, col, r, fres))
    elif t in ("ShaderNodeBsdfGlass", "ShaderNodeBsdfRefraction"):
        r = ev.scalar(n.inputs["Roughness"], drop).scalar_mean()
        out.append(("glass", w, ev.colour(n.inputs["Color"], drop), r, float(n.inputs["IOR"].default_value)))
        if t == "ShaderNodeBsdfRefraction":
            drop.add("refraction BSDF as glass (adds its Fresnel reflection)")
    elif t == "ShaderNodeBsdfTransparent":
        out.append(("transparent", w, ev.colour(n.inputs["Color"], drop)))
    elif t == "ShaderNodeEmission":
        c = ev.colour(n.inputs["Color"], drop)
        s = ev.scalar(n.inputs["Strength"], drop).scalar_mean()
        out.append(("emit", w, c, s))
    elif t == "ShaderNodeBsdfPrincipled":
        out.append(("principled", w, n))
    elif t == "ShaderNodeHoldout":
        drop.add("holdout")
    elif t == "ShaderNodeBsdfHairPrincipled" or t == "ShaderNodeBsdfHair":
        drop.add("hair BSDF on a mesh as diffuse")
        out.append(("diffuse", w, Field.k(0.3)))
    elif t == "ShaderNodeGroup":
        # Descend into a group with a single shader output.
        inner = [x for x in n.node_tree.nodes if x.bl_idname == "NodeGroupOutput"]
        if inner and inner[0].inputs and inner[0].inputs[0].is_linked:
            drop.add("node group (inputs as their defaults)")
            lobes(ev, inner[0].inputs[0], w, out, drop, fres, depth + 1)
        else:
            out.append(("diffuse", w, Field.k(0.8)))
            drop.add("node group shader as grey diffuse")
    else:
        drop.add(f"{t[10:]} shader as grey diffuse")
        out.append(("diffuse", w, Field.k(0.8)))


def wsum(items, drop):
    """Σ w and Σ w·c / Σ w over (w, c) field pairs."""
    if not items:
        return Field.k(0.0), Field.k(0.0)
    W = combine([w for w, _ in items], lambda *a: sum(a), drop)
    S = combine([x for w, c in items for x in (w, c)],
                lambda *a: sum(a[2 * i] * a[2 * i + 1] for i in range(len(a) // 2)), drop)
    C = combine([S, W], lambda s, w: s / np.maximum(w, 1e-9), drop)
    return W, C


def openpbr(ev, mat):
    """`mat` as OpenPBR parameters: (params, textures, emission, dropped).
    Colour params are constants; `textures` maps 'base_color', 'metal_color'
    and 'opacity' to Fields that override the constant."""
    drop = set()
    p = default_params()
    tex = {}
    if mat is None:
        return p, tex, None, ["no material"]
    if not mat.use_nodes:
        p["base_color"] = list(mat.diffuse_color[:3])
        p["specular_weight"] = 0.0
        return p, tex, None, ["no nodes"]
    outs = [n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"]
    act = [n for n in outs if n.is_active_output and n.target in ("ALL", "CYCLES")] or outs
    if not act:
        return p, tex, None, ["no output"]
    o = act[0]
    if o.inputs["Volume"].is_linked:
        drop.add("volume")
    if o.inputs["Displacement"].is_linked:
        drop.add("displacement")
    for n in mat.node_tree.nodes:
        if n.bl_idname in ("ShaderNodeBump", "ShaderNodeNormalMap") and n.outputs[0].is_linked:
            drop.add("bump/normal map")
            break
    L = []
    lobes(ev, o.inputs["Surface"], Field.k(1.0), L, drop)
    side = next((x[1] for x in L if x[0] == "side"), "front")
    L = [x for x in L if x[0] != "side"]
    excess = sum(x[1].scalar_mean() for x in L if x[0] == "addx")
    L = [x for x in L if x[0] != "addx"]
    prin = [x for x in L if x[0] == "principled"]
    if prin:
        # Principled, mixed or not: its own lobes join the others.
        for _, w, n in prin:
            L.extend(principled_lobes(ev, n, w, p, drop))
        L = [x for x in L if x[0] != "principled"]
    by = lambda k: [x for x in L if x[0] == k]  # noqa: E731
    Wd, Cd = wsum([(x[1], x[2]) for x in by("diffuse")], drop)
    Wm, Cm = wsum([(x[1], x[2]) for x in by("metal")], drop)
    Wg, Cg = wsum([(x[1], x[2]) for x in by("glass")], drop)
    Wtr, Ctr = wsum([(x[1], x[2]) for x in by("transparent")], drop)
    Wtl, Ctl = wsum([(x[1], x[2]) for x in by("translucent")], drop)
    spec = by("spec")
    fuzz = by("fuzz")
    emit = by("emit")
    wd, wm, wg, wtr, wtl = (f.scalar_mean() for f in (Wd, Wm, Wg, Wtr, Wtl))
    # Opacity: a transparent share driven by an image is a cut-out.
    solid = wd + wm + wg + wtl
    if wtr > 1e-4:
        if not Wtr.is_const() and solid > 1e-4:
            op = combine([Wtr], lambda a: np.clip(1 - a, 0, 1), drop)
            tex["opacity"] = op
            p["geometry_opacity"] = float(op.scalar_mean())
            drop.add("cut-out (alpha test at 0.5)")
        elif wm + len(spec) > 0 and wd + wg + wtl < 1e-3:
            # Glossy + transparent: architectural (thin) glass.
            p["geometry_thin_walled"] = 1.0
            p["transmission_weight"] = 1.0
            p["transmission_color"] = Ctr.mean().tolist()
            p["specular_weight"] = 1.0
            ior = spec[0][4] if spec else 1.5
            p["specular_ior"] = float(ior)
            src = spec if spec else by("metal")
            p["specular_color"] = (sum(x[1].scalar_mean() * x[2].mean() for x in src)
                                   / max(sum(x[1].scalar_mean() for x in src), 1e-9)).tolist()
            p["specular_roughness"] = float(np.mean([x[3] for x in src]))
            wm = 0.0
            spec = []
            drop.add("glossy + transparent as thin-walled glass")
        else:
            p["geometry_opacity"] = float(np.clip(1 - wtr / max(wtr + solid, 1e-9), 0, 1))
            if solid > 1e-4:
                drop.add("constant partial transparency as opacity")
    # The energy partition: Mix weights sum to one, an Add's inputs share
    # one weight (its excess is counted once).
    tot = max(wd + wm + wg + wtl - excess, 1e-9)
    if wd + wm + wg + wtl > 1e-4:
        p["base_metalness"] = float(min(wm / tot, 1.0))
        die = max(tot - wm, 1e-9)
        tw = min(wg / die, 1.0) if wg > 0 else 0.0
        p["transmission_weight"] = max(p["transmission_weight"], float(tw))
        # Diffuse reflection and transmission, relative to the dielectric
        # slab: the kernel reflects (1 − tl) base and transmits tl · colour,
        # both under (1 − tw); Add(Diffuse, Translucent) keeps both whole.
        rd, rt = wd / die, wtl / die
        S = rd + rt
        g = S / max(1.0 - tw, 1e-6)
        p["base_color"] = (Cd.mean() * g).tolist() if wd > 0 else [0.0, 0.0, 0.0]
        if wd > 0 and not Cd.is_const():
            tex["base_color"] = Cd.map(lambda a, g=g: a * g)
        if wd > 0 and g > 1.0 + 1e-3:
            drop.add(f"added diffuse + translucent: albedos kept whole (x{g:.2f})")
        if wm > 0:
            p["metal_color"] = Cm.mean().tolist()
            if not Cm.is_const():
                tex["metal_color"] = Cm
            ms = by("metal")
            p["specular_roughness"] = float(sum(x[1].scalar_mean() * x[3] for x in ms) / max(wm, 1e-9))
        if wtl > 0:
            p["translucency"] = float(rt / max(S, 1e-9))
            p["translucent_color"] = (Ctl.mean() * g).tolist()
            if wd <= 0:
                p["base_color"] = [0.0, 0.0, 0.0]
        if wg > 0:
            g = by("glass")
            p["transmission_color"] = (sum(x[1].scalar_mean() * x[2].mean() for x in g) / wg).tolist()
            p["specular_roughness"] = float(sum(x[1].scalar_mean() * x[3] for x in g) / wg)
            p["specular_ior"] = float(sum(x[1].scalar_mean() * x[4] for x in g) / wg)
            p["specular_weight"] = 1.0
            p["specular_color"] = [1.0, 1.0, 1.0]
        elif not prin:
            # Blender's Diffuse has no specular of its own.
            p["specular_weight"] = 0.0
    if spec:
        ws = sum(x[1].scalar_mean() for x in spec)
        p["specular_weight"] = 1.0
        p["specular_color"] = (sum(x[1].scalar_mean() * x[2].mean() for x in spec) / max(ws, 1e-9)).tolist()
        p["specular_ior"] = float(sum(x[1].scalar_mean() * x[4] for x in spec) / max(ws, 1e-9))
        r = float(sum(x[1].scalar_mean() * x[3] for x in spec) / max(ws, 1e-9))
        if p["base_metalness"] > 0 and abs(r - p["specular_roughness"]) > 0.05:
            drop.add("one roughness for metal and dielectric specular")
        if p["base_metalness"] < 0.5 and not by("glass"):
            p["specular_roughness"] = r
    if fuzz:
        wf = sum(x[1].scalar_mean() for x in fuzz)
        p["fuzz_weight"] = float(min(wf, 1.0))
        p["fuzz_color"] = (sum(x[1].scalar_mean() * x[2].mean() for x in fuzz) / max(wf, 1e-9)).tolist()
        p["fuzz_roughness"] = float(sum(x[1].scalar_mean() * x[3] for x in fuzz) / max(wf, 1e-9))
    em = None
    if emit:
        rad = sum(x[1].scalar_mean() * x[2].mean() * x[3] for x in emit)
        envs = [x for x in emit if x[2].xf == "equirect"]
        em = {"rgb": np.asarray(rad).tolist(), "side": side}
        if envs:
            em["equirect"] = envs[0][2].img * (envs[0][3] * envs[0][1].scalar_mean())
            em["rgb"] = (envs[0][2].mean() * envs[0][3]).tolist()
        p["emission_color"] = (np.asarray(rad) / max(float(np.max(rad)), 1e-12)).tolist()
        p["emission_luminance"] = float(np.max(rad))
    return p, tex, em, sorted(drop)


def principled_lobes(ev, n, w, p, drop):
    """A Principled BSDF: 1:1 onto `p` for the layers OpenPBR shares
    (coat, fuzz, film, specular, SSS, emission), and lobes for the parts a
    Mix can weigh (diffuse/metal/glass)."""
    g = lambda name: n.inputs[name]  # noqa: E731
    s = lambda name: ev.scalar(g(name), drop).scalar_mean()  # noqa: E731
    base = ev.colour(g("Base Color"), drop)
    met, rough, tr = s("Metallic"), s("Roughness"), s("Transmission Weight")
    ior = s("IOR")
    p["specular_weight"] = 2.0 * s("Specular IOR Level")  # 0.5 is the spec's 1
    p["specular_color"] = ev.colour(g("Specular Tint"), drop).mean().tolist()
    p["specular_roughness"] = rough
    p["specular_ior"] = ior
    if g("Normal").is_linked:
        drop.add("bump/normal map")
    if s("Anisotropic") > 0:
        drop.add("anisotropy")
    ss = s("Subsurface Weight")
    if ss > 0:
        p["subsurface_weight"] = ss
        p["subsurface_color"] = base.mean().tolist()
        p["subsurface_radius"] = s("Subsurface Scale")
        rr = g("Subsurface Radius").default_value
        p["subsurface_radius_scale"] = [rr[0], rr[1], rr[2]]
        p["subsurface_scatter_anisotropy"] = s("Subsurface Anisotropy")
        p["subsurface_mode"] = "cycles"
    cw = s("Coat Weight")
    if cw > 0:
        p["coat_weight"] = cw
        p["coat_roughness"] = s("Coat Roughness")
        p["coat_ior"] = s("Coat IOR")
        p["coat_color"] = ev.colour(g("Coat Tint"), drop).mean().tolist()
        p["coat_darkening"] = 0.0  # Cycles does not darken (design.md, phase 4)
    sh = s("Sheen Weight")
    if sh > 0:
        p["fuzz_weight"] = sh
        p["fuzz_color"] = ev.colour(g("Sheen Tint"), drop).mean().tolist()
        p["fuzz_roughness"] = s("Sheen Roughness")
    tf = s("Thin Film Thickness") if "Thin Film Thickness" in n.inputs else 0.0
    if tf > 0:
        p["thin_film_weight"] = 1.0
        p["thin_film_thickness"] = tf * 1e-3  # nm -> µm
        p["thin_film_ior"] = s("Thin Film IOR")
    a = ev.scalar(g("Alpha"), drop)
    out = []
    if a.scalar_mean() < 0.999:
        out.append(("transparent", combine([w, a], lambda x, y: x * (1 - y), drop), Field.k(1.0)))
        w = combine([w, a], lambda x, y: x * y, drop)
    out.append(("diffuse", combine([w], lambda x: x * (1 - met) * (1 - tr), drop), base))
    if met > 0:
        out.append(("metal", combine([w], lambda x: x * met, drop), base, rough, None))
    if tr > 0:
        out.append(("glass", combine([w], lambda x: x * tr * (1 - met), drop), base, rough, ior))
    e = ev.colour(g("Emission Color"), drop)
    es = s("Emission Strength")
    if es > 0 and float(e.mean().max()) > 0:
        out.append(("emit", w, e, es))
    return out


def default_params():
    """OpenPBR 1.x's defaults (look::openpbr::OpenPbr::default), plus irys's
    `metal_color` (None: base_color), `translucency` and thin-walled."""
    return {
        "base_weight": 1.0, "base_color": [0.8, 0.8, 0.8], "base_metalness": 0.0, "metal_color": None,
        "specular_weight": 1.0, "specular_color": [1.0, 1.0, 1.0], "specular_roughness": 0.3, "specular_ior": 1.5,
        "transmission_weight": 0.0, "transmission_color": [1.0, 1.0, 1.0], "transmission_depth": 0.0,
        "transmission_dispersion_scale": 0.0, "transmission_dispersion_abbe_number": 20.0,
        "subsurface_weight": 0.0, "subsurface_color": [0.8, 0.8, 0.8], "subsurface_radius": 1.0,
        "subsurface_radius_scale": [1.0, 0.5, 0.25], "subsurface_scatter_anisotropy": 0.0, "subsurface_mode": "physical",
        "coat_weight": 0.0, "coat_color": [1.0, 1.0, 1.0], "coat_roughness": 0.0, "coat_ior": 1.6, "coat_darkening": 1.0,
        "fuzz_weight": 0.0, "fuzz_color": [1.0, 1.0, 1.0], "fuzz_roughness": 0.5,
        "thin_film_weight": 0.0, "thin_film_thickness": 0.5, "thin_film_ior": 1.4,
        "emission_luminance": 0.0, "emission_color": [1.0, 1.0, 1.0],
        "geometry_opacity": 1.0, "geometry_thin_walled": 0.0, "translucency": 0.0, "translucent_color": [0.8, 0.8, 0.8],
    }


def save_texture(field, path, alpha=None):
    """A field's image as an 8-bit sRGB PNG (alpha: a scalar field)."""
    img = np.clip(field.img, 0, None)
    h, w, _ = img.shape
    rgba = np.ones((h, w, 4))
    rgba[..., :3] = linear_to_srgb(img)
    if alpha is not None and alpha.img is not None:
        a = alpha.img if alpha.img.shape == img.shape else resample(alpha.img, h, w)
        rgba[..., 3] = np.clip(a[..., 0], 0, 1)
    bi = bpy.data.images.new("irys_tex", w, h, alpha=True)
    bi.colorspace_settings.name = "Non-Color"
    bi.pixels.foreach_set(rgba[::-1].astype(np.float32).ravel())
    bi.filepath_raw = path
    bi.file_format = "PNG"
    bi.save()
    bpy.data.images.remove(bi)
