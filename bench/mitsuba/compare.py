"""Summarise bench/mitsuba/out/*.csv into results.md (see README.md).

    python compare.py out > results.md

Per system, bench, sample budget and parameter: the gradient's mean and
standard error over seeds, its z against the system's own central
differences of the loss, the per-gradient relative variance
(var / CD mean²), the median milliseconds per gradient, and the efficiency
1 / (relative variance × seconds) — the inverse of the time to a given
relative variance, so the equal-variance time ratio is its ratio.
Each system is held against its own CD: the two renderers' images differ
slightly (README: mismatches), so their gradients are compared in relative
terms, never value against value.
"""

import csv
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "out")


def rows(name):
    p = OUT / name
    return list(csv.DictReader(p.open())) if p.exists() else []


def cd_table(name):
    t = defaultdict(list)
    for r in rows(name):
        t[int(r["param"])].append(float(r["dloss"]))
    return {p: (st.mean(v), st.stdev(v) / math.sqrt(len(v))) for p, v in t.items()}


def summary(bench, cols, nparam, label):
    out = []
    for system in ["irys", "mitsuba"]:
        budget_cols = cols[system]
        data = rows(f"{system}_{bench}.csv")
        cd = cd_table(f"{system}_{bench}_cd.csv")
        if not data or not cd:
            continue
        groups = defaultdict(list)
        for r in data:
            groups[tuple(r[c] for c in budget_cols)].append(r)
        for key, rs in sorted(groups.items(), key=lambda kv: [int(x) for x in kv[0]]):
            ms = st.median(float(r["ms"]) for r in rs)
            for p in range(nparam):
                g = [float(r[f"g{p}"]) for r in rs]
                m, sd = st.mean(g), st.stdev(g)
                se = sd / math.sqrt(len(g))
                cm, cs = cd[p]
                z = (m - cm) / math.hypot(se, cs)
                rv = sd * sd / (cm * cm) if cm else float("nan")
                eff = 1.0 / (rv * ms / 1e3) if rv > 0 else float("nan")
                out.append(dict(system=system, budget="/".join(key), param=p, n=len(g), mean=m, se=se, cd=cm, cd_se=cs,
                                z=z, relvar=rv, ms=ms, eff=eff))
    print(f"\n### {label}\n")
    print("| system | budget | param | seeds | gradient (mean ± se) | own CD | z | rel. var | ms / gradient | efficiency 1/(relvar·s) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for o in out:
        print(f"| {o['system']} | {o['budget']} | θ{o['param']} | {o['n']} | {o['mean']:+.4e} ± {o['se']:.1e} | {o['cd']:+.4e} ± {o['cd_se']:.1e} | {o['z']:+.1f} | {o['relvar']:.3e} | {o['ms']:.1f} | {o['eff']:.3g} |")
    return out


def ratios(out, label):
    """Efficiency ratio irys / Mitsuba per parameter, best budget of each."""
    print(f"\n{label}: best efficiency per system and parameter (equal-variance time ratio = Mitsuba eff / irys eff inverse)\n")
    print("| param | irys best (budget, eff) | Mitsuba best (budget, eff) | irys / Mitsuba |")
    print("|---|---|---|---|")
    ps = sorted({o["param"] for o in out})
    for p in ps:
        best = {}
        for s in ["irys", "mitsuba"]:
            c = [o for o in out if o["system"] == s and o["param"] == p and not math.isnan(o["eff"])]
            if c:
                best[s] = max(c, key=lambda o: o["eff"])
        if len(best) == 2:
            i, m = best["irys"], best["mitsuba"]
            print(f"| θ{p} | {i['budget']}, {i['eff']:.3g} | {m['budget']}, {m['eff']:.3g} | {i['eff'] / m['eff']:.2f}× |")


def images():
    """The two renderers' targets side by side: how closely the scenes match."""
    try:
        import numpy as np
        import mitsuba as mi
        mi.set_variant("scalar_rgb")
    except Exception as e:  # noqa: BLE001
        print(f"\n(image check skipped: {e})")
        return

    def pfm(path):
        b = path.read_bytes()
        parts = b.split(b"\n", 3)
        w, h = map(int, parts[1].split())
        a = np.frombuffer(parts[3][: w * h * 12], np.float32).reshape(h, w, 3)
        return a[::-1].astype(float)

    print("\n### The two renderers' target images\n")
    print("| bench | irys mean RGB | Mitsuba mean RGB | rel. L1 difference (per pixel, mean) |")
    print("|---|---|---|---|")
    for bench in ["material", "pose"]:
        ip, mp = OUT / f"irys_{bench}_target.pfm", OUT / f"mitsuba_{bench}_target.exr"
        if not (ip.exists() and mp.exists()):
            continue
        a = pfm(ip)
        b = np.array(mi.Bitmap(str(mp)))[..., :3].astype(float)
        rel = np.abs(a - b).sum() / np.abs(a).sum()
        print(f"| {bench} | {np.round(a.mean((0, 1)), 4).tolist()} | {np.round(b.mean((0, 1)), 4).tolist()} | {rel:.3f} |")


def hi_spp():
    data = rows("mitsuba_pose_hi.csv")
    cd = cd_table("mitsuba_pose_cd.csv")
    if not data or not cd:
        return
    print("\n### Mitsuba pose gradient at higher spp (does it converge to its own CD?)\n")
    print("| spp | param | seeds | gradient (mean ± se) | own CD | z | ms / gradient |")
    print("|---|---|---|---|---|---|---|")
    for spp in sorted({r["spp"] for r in data}, key=int):
        rs = [r for r in data if r["spp"] == spp]
        for p in range(3):
            g = [float(r[f"g{p}"]) for r in rs]
            m, se = st.mean(g), st.stdev(g) / math.sqrt(len(g))
            cm, cs = cd[p]
            print(f"| {spp} | φ{p} | {len(g)} | {m:+.4e} ± {se:.1e} | {cm:+.4e} ± {cs:.1e} | {(m - cm) / math.hypot(se, cs):+.1f} | {st.median(float(r['ms']) for r in rs):.0f} |")


def irys_split():
    data = rows("irys_pose.csv")
    if not data or "edge0" not in data[0]:
        return
    print("\n### irys pose gradient split (65 536 edge samples, 16 interior spp)\n")
    print("| param | total | edge term | interior term |")
    print("|---|---|---|---|")
    rs = [r for r in data if r["n_edge"] == "65536"]
    for p in range(3):
        t = st.mean(float(r[f"g{p}"]) for r in rs)
        e = st.mean(float(r[f"edge{p}"]) for r in rs)
        print(f"| φ{p} | {t:+.4e} | {e:+.4e} | {t - e:+.4e} |")


if __name__ == "__main__":
    mat = summary("material", {"irys": ["spp"], "mitsuba": ["spp"]}, 4,
                  "Material gradient (pair, GGX conductor; θ = α, F0₄₅₀, F0₅₅₀, F0₆₅₀; budget = spp)")
    ratios(mat, "Material")
    pose = summary("pose", {"irys": ["n_edge", "spp_int"], "mitsuba": ["spp"]}, 3,
                   "Pose gradient (two-box arm; φ = pitch, yaw, elbow; budget: irys edge samples / interior spp, Mitsuba prb_projective spp)")
    ratios(pose, "Pose")
    irys_split()
    hi_spp()
    images()
