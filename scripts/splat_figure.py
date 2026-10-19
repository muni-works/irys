#!/usr/bin/env python3
"""Figures for docs/design.md, "Splat training" (PIL only).

    python3 scripts/splat_figure.py OUT.png REPORT_DIR[:label] [REPORT_DIR:label ...] \
        [--views 0,5,10] [--irys] [--title TEXT]

Rows: held-out views of the first report, ground truth | splat render |
|error| ×4 (| irys ray-traced render with --irys). Below: held-out PSNR over
iterations for every report (log.csv).
"""
import json
import sys

from PIL import Image, ImageDraw, ImageFont

args = sys.argv[1:]
out = args.pop(0)
opts = {}
reps = []
i = 0
while i < len(args):
    if args[i].startswith("--"):
        if args[i] == "--irys":
            opts["irys"] = True
            i += 1
        else:
            opts[args[i][2:]] = args[i + 1]
            i += 2
    else:
        d, _, lab = args[i].partition(":")
        reps.append((d, lab or d))
        i += 1
views = [int(v) for v in opts.get("views", "0,6,12").split(",")]
cols = ["gt", "render", "error"] + (["irys"] if opts.get("irys") else [])
heads = {"gt": "held-out ground truth", "render": "irys-trained splats (raster)", "error": "|error| x4", "irys": "same map, irys ray tracer"}
d0 = reps[0][0]
m0 = json.load(open(d0 + "/metrics.json"))
w, h = Image.open(f"{d0}/test{views[0]:03d}_gt.png").size
s = 0.5
tw, th = int(w * s), int(h * s)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
except OSError:
    font = small = ImageFont.load_default()
pad, top = 6, 24
plot_h = 260
W = pad + len(cols) * (tw + pad)
H = top + len(views) * (th + pad) + plot_h + 40
img = Image.new("RGB", (W, H), "white")
dr = ImageDraw.Draw(img)
for c, k in enumerate(cols):
    dr.text((pad + c * (tw + pad), 4), heads[k], fill="black", font=font)
for r, v in enumerate(views):
    pv = m0["per_view"][v]
    for c, k in enumerate(cols):
        t = Image.open(f"{d0}/test{v:03d}_{k}.png").resize((tw, th))
        x, y = pad + c * (tw + pad), top + r * (th + pad)
        img.paste(t, (x, y))
        if k == "render":
            dr.text((x + 4, y + 4), f"{pv['psnr']:.1f} dB  SSIM {pv['ssim']:.3f}", fill="yellow", font=small)
# PSNR curves
y0 = top + len(views) * (th + pad) + 20
x0, x1 = pad + 50, W - pad - 10
py0, py1 = y0 + 10, y0 + plot_h - 30
series = []
for d, lab in reps:
    rows = [l.split(",") for l in open(d + "/log.csv").read().strip().splitlines()[1:]]
    pts = [(int(r[0]), float(r[4])) for r in rows if r[4] not in ("nan", "NaN")]
    series.append((lab, pts))
xmax = max(p[0] for _, s_ in series for p in s_)
ys = [p[1] for _, s_ in series for p in s_]
lo, hi = int(min(ys)) - 1, int(max(ys)) + 2
fx = lambda it: x0 + (x1 - x0) * it / xmax
fy = lambda v: py1 - (py1 - py0) * (v - lo) / (hi - lo)
dr.rectangle([x0, py0, x1, py1], outline="black")
for v in range(lo, hi + 1, max(1, (hi - lo) // 6)):
    dr.line([x0, fy(v), x1, fy(v)], fill=(225, 225, 225))
    dr.text((x0 - 30, fy(v) - 7), f"{v}", fill="black", font=small)
if lo <= 28 <= hi:
    dr.line([x0, fy(28), x1, fy(28)], fill=(200, 60, 60))
    dr.text((x1 - 60, fy(28) - 15), "gate 28 dB", fill=(200, 60, 60), font=small)
for k in range(0, 6):
    it = xmax * k // 5
    dr.text((fx(it) - 12, py1 + 4), f"{it}", fill="black", font=small)
palette = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (148, 103, 189)]
for n, (lab, pts) in enumerate(series):
    col = palette[n % 4]
    dr.line([(fx(a), fy(b)) for a, b in pts], fill=col, width=2)
    dr.text((x0 + 10, py0 + 6 + 16 * n), f"{lab}: {pts[-1][1]:.2f} dB", fill=col, font=small)
dr.text((x0, py1 + 18), opts.get("title", "held-out PSNR (dB) vs iteration"), fill="black", font=small)
img.save(out)
print(out, img.size)
