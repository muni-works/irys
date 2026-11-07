"""Mitsuba 3's half of the irys baseline (see README.md).

    python run_mitsuba.py OUT [material] [pose] [--variant metal_ad_spectral]

OUT holds what `mitsuba_bench` (irys) exported: material.json / pose.json,
their PLYs. Writes mitsuba_<bench>.csv (one row per spp and seed: gradient,
loss, milliseconds) and mitsuba_<bench>_cd.csv (central differences).

The BSDF is irys's rough conductor exactly (`crates/irys-diff/src/ggx.rs`):
isotropic GGX, height-correlated Smith, Schlick's Fresnel, F0 a spectrum of
three hats peaking at 450/550/650 nm (in an RGB variant: F0 per channel,
R = the 650 nm hat, B = 450), visible-normal sampling. The environment is
irys's analytic sky and sun lobe baked into a lat-long map.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import mitsuba as mi
import drjit as dr

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
OPTS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
VARIANT = OPTS.get("variant", "metal_ad_spectral")
mi.set_variant(VARIANT)
SPECTRAL = "spectral" in VARIANT
OUT = Path(ARGS[0] if ARGS else "out")
TAG = OPTS.get("tag", "")
WHICH = ARGS[1:] or ["material", "pose"]
CD_ONLY = "cd-only" in sys.argv


class IrysGGX(mi.BSDF):
    """irys's `ggx` graph: f = F D G2 / (4 cos_i cos_o), pdf = D G1(o) / (4 cos_o)."""

    def __init__(self, props):
        mi.BSDF.__init__(self, props)
        self.alpha = mi.Float(props.get("alpha", 0.35))
        self.f0 = mi.Vector3f(props.get("f0_450", 0.95), props.get("f0_550", 0.75), props.get("f0_650", 0.45))
        self.m_flags = mi.BSDFFlags.GlossyReflection | mi.BSDFFlags.FrontSide
        self.m_components = [self.m_flags]

    def _F0(self, si):
        if SPECTRAL:
            lam = si.wavelengths
            b0 = dr.clip((550.0 - lam) / 100.0, 0.0, 1.0)
            b2 = dr.clip((lam - 550.0) / 100.0, 0.0, 1.0)
            b1 = 1.0 - b0 - b2
            return b0 * self.f0.x + b1 * self.f0.y + b2 * self.f0.z
        return mi.Color3f(self.f0.z, self.f0.y, self.f0.x)

    @staticmethod
    def _lam(a2, c):
        c2 = dr.maximum(c * c, 1e-12)
        return (dr.sqrt(1.0 + a2 * (1.0 - c2) / c2) - 1.0) * 0.5

    def _f_cos(self, si, wo, m):
        """f · cos_i (Mitsuba's eval convention) for the half vector m."""
        ci = mi.Frame3f.cos_theta(wo)
        co = mi.Frame3f.cos_theta(si.wi)
        a2 = self.alpha * self.alpha
        ch = mi.Frame3f.cos_theta(m)
        den = ch * ch * (a2 - 1.0) + 1.0
        D = a2 / (dr.pi * den * den)
        G2 = 1.0 / (1.0 + self._lam(a2, ci) + self._lam(a2, co))
        oh = dr.dot(si.wi, m)
        F0 = self._F0(si)
        F = F0 + (1.0 - F0) * dr.power(dr.maximum(1.0 - oh, 0.0), 5.0)
        return F * D * G2 / (4.0 * co)

    def _pdf(self, si, wo, m):
        co = mi.Frame3f.cos_theta(si.wi)
        a2 = self.alpha * self.alpha
        ch = mi.Frame3f.cos_theta(m)
        den = ch * ch * (a2 - 1.0) + 1.0
        D = a2 / (dr.pi * den * den)
        G1 = 1.0 / (1.0 + self._lam(a2, co))
        return D * G1 / (4.0 * co)

    def sample(self, ctx, si, sample1, sample2, active=True):
        co = mi.Frame3f.cos_theta(si.wi)
        active = active & (co > 0)
        distr = mi.MicrofacetDistribution(mi.MicrofacetType.GGX, dr.detach(self.alpha), True)
        m, _ = distr.sample(si.wi, sample2)
        wo = mi.reflect(si.wi, m)
        active = active & (mi.Frame3f.cos_theta(wo) > 0) & (dr.dot(si.wi, m) > 0)
        bs = dr.zeros(mi.BSDFSample3f)
        bs.wo = wo
        bs.pdf = dr.select(active, self._pdf(si, wo, m), 0.0)
        bs.eta = 1.0
        bs.sampled_type = mi.UInt32(+mi.BSDFFlags.GlossyReflection)
        bs.sampled_component = mi.UInt32(0)
        val = self._f_cos(si, wo, m) / dr.maximum(bs.pdf, 1e-20)
        return (bs, dr.select(active, mi.Spectrum(val), mi.Spectrum(0.0)))

    def eval(self, ctx, si, wo, active=True):
        ci = mi.Frame3f.cos_theta(wo)
        co = mi.Frame3f.cos_theta(si.wi)
        active = active & (ci > 0) & (co > 0)
        m = dr.normalize(wo + si.wi)
        return dr.select(active, mi.Spectrum(self._f_cos(si, wo, m)), mi.Spectrum(0.0))

    def pdf(self, ctx, si, wo, active=True):
        ci = mi.Frame3f.cos_theta(wo)
        co = mi.Frame3f.cos_theta(si.wi)
        active = active & (ci > 0) & (co > 0)
        m = dr.normalize(wo + si.wi)
        return dr.select(active, self._pdf(si, wo, m), 0.0)

    def eval_pdf(self, ctx, si, wo, active=True):
        return self.eval(ctx, si, wo, active), self.pdf(ctx, si, wo, active)

    def traverse(self, callback):
        callback.put("alpha", self.alpha, mi.ParamFlags.Differentiable)
        callback.put("f0", self.f0, mi.ParamFlags.Differentiable)

    def parameters_changed(self, keys):
        pass

    def to_string(self):
        return "IrysGGX"


mi.register_bsdf("irys_ggx", lambda props: IrysGGX(props))

# irys's environment (prb::Env): sky · tint(λ) (0.15 + 0.25 max(d_z, 0)) +
# sun_radiance · exp(κ (d·sun − 1)); tint(λ) = 1 + 0.4 (550 − λ)/200.
# Mitsuba's envmap: local d = (sin φ sin θ, cos θ, −cos φ sin θ), θ = vπ,
# φ = 2πu; to_world maps local +y to world +z.
TO_WORLD = mi.ScalarTransform4f([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
RGB_NM = (610.0, 550.0, 465.0)


def env_bitmap(env, w=2048, h=1024):
    v, u = np.meshgrid((np.arange(h) + 0.5) / h, (np.arange(w) + 0.5) / w, indexing="ij")
    th, ph = v * math.pi, u * 2 * math.pi
    dl = np.stack([np.sin(ph) * np.sin(th), np.cos(th), -np.cos(ph) * np.sin(th)], -1)
    d = np.stack([dl[..., 0], -dl[..., 2], dl[..., 1]], -1)  # TO_WORLD
    sun = np.array(env["sun"])
    lobe = env["sun_radiance"] * np.exp(env["kappa"] * (d @ sun - 1.0))
    grad = env["sky"] * (0.15 + 0.25 * np.maximum(d[..., 2], 0.0))
    img = np.stack([(1.0 + 0.4 * (550.0 - nm) / 200.0) * grad + lobe for nm in RGB_NM], -1)
    return mi.Bitmap(img.astype(np.float32))


def welded_mesh(path, name, bsdf):
    """The PLY as written — shared vertices, flat shading — built directly.

    Mitsuba's PLY loader with `face_normals` gave a mesh whose projective
    (boundary) derivatives came out 2–3× the finite differences: its seams
    count as boundary edges. The welded mesh matches them (README)."""
    lines = Path(path).read_text().split("\n")
    nv = int(lines[2].split()[2])
    nf = int(next(l for l in lines if l.startswith("element face")).split()[2])
    h = lines.index("end_header")
    v = np.array([[float(x) for x in l.split()] for l in lines[h + 1:h + 1 + nv]], np.float32)
    f = np.array([[int(x) for x in l.split()[1:]] for l in lines[h + 1 + nv:h + 1 + nv + nf]], np.uint32)
    props = mi.Properties()
    props["bsdf"] = mi.load_dict(bsdf)
    props["face_normals"] = True
    m = mi.Mesh(name, nv, nf, props, has_vertex_normals=False)
    mp = mi.traverse(m)
    mp["vertex_positions"] = dr.ravel(mi.Point3f(v[:, 0], v[:, 1], v[:, 2]))
    mp["faces"] = dr.ravel(mi.Vector3u(f[:, 0], f[:, 1], f[:, 2]))
    mp.update()
    return m


def load_scene(js, out, integrator, theta=None, border=False):
    w, h = js["size"]
    d = {
        "type": "scene",
        "integrator": integrator,
        "sensor": {
            "type": "perspective",
            "fov": js["fov_y_deg"],
            "fov_axis": "y",
            "to_world": mi.ScalarTransform4f().look_at(js["eye"], js["target"], js["up"]),
            "film": {"type": "hdrfilm", "width": w, "height": h, "rfilter": {"type": "box"}, "pixel_format": "rgb", "sample_border": border},
            "sampler": {"type": "independent", "sample_count": 1},
        },
        "env": {"type": "envmap", "bitmap": env_bitmap(js["env"]), "to_world": TO_WORLD},
    }
    th = theta if theta is not None else js["theta0"]
    for i, s in enumerate(js["shapes"]):
        if s["bsdf"] == "conductor" and OPTS.get("conductor") == "diffuse":
            b = {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.5] * 3}}
        elif s["bsdf"] == "conductor":
            b = {"type": "irys_ggx", "alpha": th[0], "f0_450": th[1], "f0_550": th[2], "f0_650": th[3]}
        else:
            # irys: Lambertian, albedo = floor_albedo × the placement's tint,
            # spectrally flat.
            b = {"type": "diffuse", "reflectance": {"type": "spectrum" if SPECTRAL else "rgb", "value": js["floor_albedo"] * s["albedo"]} if SPECTRAL else {"type": "rgb", "value": [js["floor_albedo"] * s["albedo"]] * 3}}
        d[f"shape_{i}"] = welded_mesh(out / s["ply"], f"shape_{i}", b)
    return mi.load_dict(d)


def mse(img, target):
    return dr.mean(dr.square(img - target), axis=None)


def product_loss(a, b, target):
    """mean((I_a − T)(I_b − T)), two independent renders: unbiased for the
    loss of the expected image, which the decorrelated gradient estimates
    (one noisy image's MSE adds Var I, which depends on the parameters)."""
    return dr.mean((a - target) * (b - target), axis=None)


def num(x):
    """A Python float from a drjit scalar, array or 0-d tensor."""
    return float(x.array[0]) if isinstance(x, mi.TensorXf) else float(x[0])


def render_mean(scene, spp, frames, seed0, params=None):
    acc = None
    for f in range(frames):
        im = mi.render(scene, params, spp=spp, seed=seed0 + f)
        acc = im if acc is None else acc + im
    acc = acc / frames
    dr.eval(acc)
    return mi.TensorXf(acc)


def material():
    js = json.loads((OUT / "material.json").read_text())
    md = js["depth"] + 1  # irys `depth` vertices, `complete`: Mitsuba max_depth
    integ = {"type": "prb", "max_depth": md, "rr_depth": 1000}
    if CD_ONLY:
        target = mi.TensorXf(mi.Bitmap(str(OUT / "mitsuba_material_target.exr")))
    else:
        scene = load_scene(js, OUT, integ, theta=js["theta_target"])
        target = render_mean(scene, 256, 16, 1 << 20)  # 4096 spp
        mi.util.write_bitmap(str(OUT / "mitsuba_material_target.exr"), target)
    scene = load_scene(js, OUT, integ)
    params = mi.traverse(scene)
    ka = [k for k in params.keys() if k.endswith(".alpha")]
    kf = [k for k in params.keys() if k.endswith(".f0")]
    th0 = js["theta0"]
    if not CD_ONLY:
        start = render_mean(scene, 256, 4, 1 << 21)
        mi.util.write_bitmap(str(OUT / "mitsuba_material_start.exr"), start)

    def set_theta(th):
        for k in ka:
            params[k] = th[0]
        for k in kf:
            params[k] = mi.Vector3f(th[1], th[2], th[3])
        params.update()

    rows = ["system,bench,spp,seed,g0,g1,g2,g3,loss,ms"]
    seeds = int(OPTS.get("seeds", 64))
    for spp in [] if CD_ONLY else [1, 4, 16]:
        for it in range(-3, seeds):  # three warm-ups (kernel compilation)
            set_theta(th0)
            for k in ka + kf:
                dr.enable_grad(params[k])
            params.update()
            t0 = time.perf_counter()
            img = mi.render(scene, params, spp=spp, seed=2 * max(it, 0) + 1000 * (it < 0), seed_grad=2 * max(it, 0) + 1 + 1000 * (it < 0))
            loss = mse(img, target)
            dr.backward(loss)
            ga = sum(dr.grad(params[k]) for k in ka)
            gf = sum(dr.grad(params[k]) for k in kf)
            g = [num(ga), num(gf.x), num(gf.y), num(gf.z)]
            lv = num(loss)
            ms = (time.perf_counter() - t0) * 1e3
            if it >= 0:
                rows.append(f"mitsuba,material,{spp},{it},{g[0]},{g[1]},{g[2]},{g[3]},{lv},{ms}")
        print(f"material spp {spp}: done", file=sys.stderr)
    if not CD_ONLY:
        (OUT / "mitsuba_material.csv").write_text("\n".join(rows) + "\n")

    cd = ["system,bench,param,h,seed,dloss"]
    hh = 0.01
    with dr.suspend_grad():
        for p in range(4):
            for sd in range(64):
                a, b = list(th0), list(th0)
                a[p] += hh
                b[p] -= hh
                set_theta(a)
                la = num(product_loss(mi.render(scene, params, spp=16, seed=5000 + sd), mi.render(scene, params, spp=16, seed=50000 + sd), target))
                set_theta(b)
                lb = num(product_loss(mi.render(scene, params, spp=16, seed=5000 + sd), mi.render(scene, params, spp=16, seed=50000 + sd), target))
                cd.append(f"mitsuba,material,{p},{hh},{sd},{(la - lb) / (2 * hh)}")
    (OUT / "mitsuba_material_cd.csv").write_text("\n".join(cd) + "\n")


# --- pose: phase 2e's arm rig (irys `vis::about`, `Transform::then`) ---------


def _rot(axis, ang):
    """Right-handed rotation about a unit axis (Rodrigues), as a 3x3 of drjit Floats."""
    x, y, z = axis
    c, s = dr.cos(ang), dr.sin(ang)
    t = 1.0 - c
    return [[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c]]


class Aff:
    """x ↦ R x + t, with drjit entries; a.then(b) = a ∘ b (b first), as irys."""

    def __init__(self, R, t):
        self.R, self.t = R, t

    @staticmethod
    def about(p, axis, ang):
        R = _rot(axis, ang)
        t = [p[i] - sum(R[i][j] * p[j] for j in range(3)) for i in range(3)]
        return Aff(R, t)

    @staticmethod
    def trans(v):
        one = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
        return Aff([[mi.Float(x) for x in r] for r in one], [mi.Float(x) for x in v])

    @staticmethod
    def rot_x(a):
        return Aff(_rot((1.0, 0.0, 0.0), mi.Float(a)), [mi.Float(0.0)] * 3)

    def then(self, b):
        R = [[sum(self.R[i][k] * b.R[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        t = [sum(self.R[i][k] * b.t[k] for k in range(3)) + self.t[i] for i in range(3)]
        return Aff(R, t)

    def apply(self, P):  # P: mi.Point3f (all vertices)
        return mi.Point3f(*(self.R[i][0] * P.x + self.R[i][1] * P.y + self.R[i][2] * P.z + self.t[i] for i in range(3)))


def arm_place(phi, rig):
    s = rig["shoulder"]
    U, F, G = rig["upper"], rig["fore"], rig["gap"]
    sh = Aff.about(s, (0.0, 0.0, 1.0), phi[1]).then(Aff.about(s, (0.0, 1.0, 0.0), phi[0]))
    e = [s[0] + 2 * U + 2 * G, s[1], s[2]]
    upper = sh.then(Aff.trans([s[0] + U + G, s[1], s[2]])).then(Aff.rot_x(rig["roll"][0]))
    fore = sh.then(Aff.about(e, (0.0, 1.0, 0.0), phi[2])).then(Aff.trans([e[0] + F + G, e[1], e[2]])).then(Aff.rot_x(rig["roll"][1]))
    return [upper, fore]


def pose():
    js = json.loads((OUT / "pose.json").read_text())
    md = js["depth"] + 1
    rig = js["rig"]
    th = js["theta_arm"]
    integ = {"type": "prb_projective", "max_depth": md, "rr_depth": 1000}
    if OPTS.get("integrator") == "direct_projective":
        integ = {"type": "direct_projective"}
    if OPTS.get("integrator") == "prb":
        integ = {"type": "prb", "max_depth": md, "rr_depth": 1000}
    for k in ("sppc", "sppp", "sppi", "guiding", "guiding_rounds"):
        if k in OPTS:
            integ[k] = OPTS[k] if k == "guiding" else int(OPTS[k])
    scene = load_scene(js, OUT, integ, theta=th, border=True)
    params = mi.traverse(scene)
    moving = {}
    for i, s in enumerate(js["shapes"]):
        if s["moving"] >= 0:
            key = f"shape_{i}.vertex_positions"
            moving[s["moving"]] = (key, dr.unravel(mi.Point3f, dr.detach(params[key])))

    def set_pose(phi):
        for m, A in enumerate(arm_place(phi, rig)):
            key, P = moving[m]
            params[key] = dr.ravel(A.apply(P))
        params.update()

    if CD_ONLY:
        target = mi.TensorXf(mi.Bitmap(str(OUT / f"mitsuba_pose{TAG}_target.exr")))
    else:
        with dr.suspend_grad():
            set_pose([mi.Float(x) for x in js["phi_target"]])
            target = render_mean(scene, 256, 16, 1 << 22)
        mi.util.write_bitmap(str(OUT / f"mitsuba_pose{TAG}_target.exr"), target)

    rows = ["system,bench,spp,seed,g0,g1,g2,loss,ms"]
    seeds = int(OPTS.get("seeds", 40))
    for spp in [] if CD_ONLY and "grad" not in sys.argv else [int(x) for x in OPTS.get("spps", "1,4,16").split(",")]:
        for it in range(-3, seeds):
            phi = [mi.Float(x) for x in js["phi0"]]
            for p in phi:
                dr.enable_grad(p)
            t0 = time.perf_counter()
            set_pose(phi)
            sd = max(it, 0) + 1000 * (it < 0)
            img = mi.render(scene, params, spp=spp, seed=2 * sd, seed_grad=2 * sd + 1)
            loss = mse(img, target)
            dr.backward(loss)
            g = [num(dr.grad(p)) for p in phi]
            lv = num(loss)
            ms = (time.perf_counter() - t0) * 1e3
            if it >= 0:
                rows.append(f"mitsuba,pose,{spp},{it},{g[0]},{g[1]},{g[2]},{lv},{ms}")
        print(f"pose spp {spp}: done", file=sys.stderr)
    if len(rows) > 1:
        (OUT / f"mitsuba_pose{TAG}.csv").write_text("\n".join(rows) + "\n")

    if "skip-cd" in sys.argv:
        return
    cd = ["system,bench,param,h,seed,dloss"]
    hh = 0.01
    with dr.suspend_grad():
        for p in range(3):
            for sd in range(32):
                a, b = list(js["phi0"]), list(js["phi0"])
                a[p] += hh
                b[p] -= hh
                set_pose([mi.Float(x) for x in a])
                la = num(product_loss(mi.render(scene, params, spp=64, seed=7000 + sd), mi.render(scene, params, spp=64, seed=70000 + sd), target))
                set_pose([mi.Float(x) for x in b])
                lb = num(product_loss(mi.render(scene, params, spp=64, seed=7000 + sd), mi.render(scene, params, spp=64, seed=70000 + sd), target))
                cd.append(f"mitsuba,pose,{p},{hh},{sd},{(la - lb) / (2 * hh)}")
    (OUT / f"mitsuba_pose{TAG}_cd.csv").write_text("\n".join(cd) + "\n")


if __name__ == "__main__":
    if "material" in WHICH:
        material()
    if "pose" in WHICH:
        pose()
