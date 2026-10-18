#!/usr/bin/env python3
# Phase 5b: camera moves for the reel, as the JSON lists
# `examples/blend_look --sequence` renders.
#
#   python3 scripts/reel_cameras.py <export dir> keyframed <out.json>
#   python3 scripts/reel_cameras.py <export dir> orbit <out.json> [--seconds 5] [--fps 30]
#                                   [--distance D] [--degrees A] [--rise R]
#   python3 scripts/reel_cameras.py <export dir> dolly <out.json> [--seconds 5] [--fps 30]
#                                   [--distance D] [--rise R] [--distance-to-target T]
#
# keyframed: the .blend's own animated camera, resampled to the reel's
# frame rate by the exporter (`blend_export.py --reel 30:a:b`).
# orbit: an authored move: the export's camera swung about a vertical axis
# through a pivot D metres ahead of it, A degrees in all, eased in and out
# (smoothstep), rising R metres, always looking at the pivot.
# dolly: the export's camera pushed D metres along its view, eased, rising
# R metres, looking at a point T metres ahead of where it started.

import json
import math
import sys


def sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def add(a, b):
    return [a[i] + b[i] for i in range(3)]


def mul(a, k):
    return [x * k for x in a]


def norm(a):
    l = math.sqrt(sum(x * x for x in a)) or 1.0
    return [x / l for x in a]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def look(eye, target, tan_half):
    f = norm(sub(target, eye))
    r = norm(cross(f, [0.0, 0.0, 1.0]))
    u = cross(r, f)
    return {"eye": eye, "forward": f, "right": r, "up": u, "tan_half": tan_half}


def main():
    d, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
    o = dict(zip(sys.argv[4::2], sys.argv[5::2]))
    side = json.load(open(d + "/irys.json"))
    if mode == "keyframed":
        r = side["reel_frames"]
        cams = [{k: f[k] for k in ("eye", "forward", "right", "up", "tan_half")} for f in r["frames"]]
    elif mode == "orbit":
        c = side["camera"]
        secs = float(o.get("--seconds", 5))
        fps = float(o.get("--fps", 30))
        dist = float(o.get("--distance", 4))
        deg = math.radians(float(o.get("--degrees", 14)))
        rise = float(o.get("--rise", 0.0))
        pivot = add(c["eye"], mul(c["forward"], dist))
        n = int(round(secs * fps)) + 1
        cams = []
        for k in range(n):
            s = k / (n - 1)
            e = s * s * (3 - 2 * s)
            a = (e - 0.5) * deg
            v = sub(c["eye"], pivot)
            ca, sa = math.cos(a), math.sin(a)
            vr = [v[0] * ca - v[1] * sa, v[0] * sa + v[1] * ca, v[2] + rise * (e - 0.5)]
            cams.append(look(add(pivot, vr), pivot, c["tan_half"]))
    elif mode == "dolly":
        c = side["camera"]
        secs = float(o.get("--seconds", 5))
        fps = float(o.get("--fps", 30))
        dist = float(o.get("--distance", 3))
        rise = float(o.get("--rise", 0.0))
        tgt = add(c["eye"], mul(c["forward"], float(o.get("--distance-to-target", 20))))
        n = int(round(secs * fps)) + 1
        cams = []
        for k in range(n):
            s = k / (n - 1)
            e = s * s * (3 - 2 * s)
            eye = add(add(c["eye"], mul(c["forward"], dist * e)), [0.0, 0.0, rise * e])
            cams.append(look(eye, tgt, c["tan_half"]))
    else:
        raise SystemExit(f"unknown mode {mode}")
    json.dump(cams, open(out, "w"))
    print(f"{len(cams)} cameras -> {out}")


if __name__ == "__main__":
    main()
