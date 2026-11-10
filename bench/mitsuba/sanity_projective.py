"""Is Mitsuba's projective sampling set up right here? (README: pose.)

    python sanity_projective.py > out/sanity_projective.txt

A box translating in x in front of an environment, `direct_projective`,
the derivative of Σ w·I (w a horizontal ramp) by reverse mode against
central differences of the 1024-spp image. Mitsuba's built-in `cube` has
split vertices along its edges; the welded box (8 vertices, the faces irys's
`prb::cuboid` writes) does not. Run on both backends.
"""

import sys

import numpy as np
import mitsuba as mi
import drjit as dr

FACES = np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
                  [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]], np.uint32)


def welded(half, rot=np.eye(3)):
    v = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)], float) * np.array(half)
    v = (v @ rot.T + np.array([0, 0, 0.5])).astype(np.float32)
    props = mi.Properties()
    props["bsdf"] = mi.load_dict({"type": "diffuse", "reflectance": {"type": "rgb", "value": 0.3}})
    props["face_normals"] = True
    m = mi.Mesh("box", 8, 12, props, has_vertex_normals=False)
    mp = mi.traverse(m)
    mp["vertex_positions"] = dr.ravel(mi.Point3f(v[:, 0], v[:, 1], v[:, 2]))
    mp["faces"] = dr.ravel(mi.Vector3u(FACES[:, 0], FACES[:, 1], FACES[:, 2]))
    mp.update()
    return m


def check(label, shape):
    sc = mi.load_dict({
        "type": "scene", "integrator": {"type": "direct_projective"},
        "sensor": {"type": "perspective", "fov": 30, "fov_axis": "y",
                   "to_world": mi.ScalarTransform4f().look_at([0, -6, 1], [0, 0, 0.5], [0, 0, 1]),
                   "film": {"type": "hdrfilm", "width": 96, "height": 72, "rfilter": {"type": "box"}, "sample_border": True},
                   "sampler": {"type": "independent"}},
        "env": {"type": "constant", "radiance": {"type": "rgb", "value": 1.0}},
        "box": shape,
        "floor": {"type": "rectangle", "to_world": mi.ScalarTransform4f().scale(4), "bsdf": {"type": "diffuse"}},
    })
    p = mi.traverse(sc)
    key = "box.vertex_positions"
    w = np.ascontiguousarray((np.linspace(0, 1, 96)[None, :, None] * np.ones((72, 96, 3))).astype(np.float32))
    wt = mi.TensorXf(w)
    p0 = dr.unravel(mi.Point3f, dr.detach(p[key]))

    def setx(x):
        p[key] = dr.ravel(p0 + mi.Vector3f(x, 0, 0))
        p.update()

    gs = []
    for s in range(8):
        x = mi.Float(0.0)
        dr.enable_grad(x)
        setx(x)
        img = mi.render(sc, p, spp=64, seed=s, seed_grad=100 + s)
        dr.backward(dr.sum(wt * img, axis=None))
        gs.append(float(dr.grad(x)[0]))
    with dr.suspend_grad():
        h = 0.01
        setx(mi.Float(h))
        a = np.array(mi.render(sc, p, spp=1024, seed=1))
        setx(mi.Float(-h))
        b = np.array(mi.render(sc, p, spp=1024, seed=1))
    fd = float((w * (a - b) / (2 * h)).sum())
    print(f"{mi.variant()}  {label:36s} reverse mode {np.mean(gs):+9.2f} ± {np.std(gs) / np.sqrt(8):.2f}   CD {fd:+9.2f}   ratio {np.mean(gs) / fd:.3f}")


if __name__ == "__main__":
    for v in ["metal_ad_rgb", "llvm_ad_rgb"]:
        mi.set_variant(v)
        check("built-in cube (split vertices)", {"type": "cube", "to_world": mi.ScalarTransform4f().translate([0, 0, 0.5]).scale(0.5),
                                                  "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": 0.3}}})
        check("welded cube", welded([0.5, 0.5, 0.5]))
        c, s = np.cos(0.35), np.sin(0.35)
        check("welded thin box, rolled 0.35", welded([0.5, 0.15, 0.15], np.array([[1, 0, 0], [0, c, -s], [0, s, c]])))
    sys.stdout.flush()
