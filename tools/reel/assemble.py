#!/usr/bin/env python3
"""Lay the reel's frames out on the soundtrack's grid.

    python3 assemble.py MANIFEST_DIR [out/beats.json] [out/reel-frames]

Reads the shots in beats.json (written by `irys-reel render`), and writes one
30 fps frame per output time as NNNNN.jpg: a symlink to the source frame, a
copy with the shot's caption drawn on, or a title card. ImageMagick draws the
text (SF Pro, macOS's system font). Standard library only.
"""
import json, os, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

FONT = os.environ.get("FONT", "/System/Library/Fonts/SFNS.ttf")
W, H = 1920, 1080
BG = "#0a0a0c"

root = sys.argv[1]
beats = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "out/beats.json"))
out = sys.argv[3] if len(sys.argv) > 3 else "out/reel-frames"
man = json.load(open(os.path.join(root, "manifest.json")))
paths = {s["name"]: os.path.join(root, s["path"]) for s in man["sequences"]}
fps = beats["fps"]

shutil.rmtree(out, ignore_errors=True)
os.makedirs(out)
cards = os.path.join(out, "_cards")
os.makedirs(cards)


def magick(*a):
    subprocess.run(["magick", *a], check=True)


def card(text, path):
    lines = text.split("\n")
    a = ["-size", f"{W}x{H}", f"xc:{BG}", "-font", FONT, "-fill", "white",
         "-gravity", "center", "-pointsize", "240", "-kerning", "-4", "-annotate", "+0-30", lines[0]]
    if len(lines) > 1:
        a += ["-kerning", "0", "-fill", "#9a9aa2", "-pointsize", "26", "-gravity", "south",
              "-annotate", "+0+84", lines[1]]
    magick(*a, "-quality", "95", path)


def captioned(src, text, dst):
    # A soft shadow under the text so it reads on sky and water.
    magick(src, "(", "-size", f"{W}x{H}", "xc:none", "-font", FONT, "-pointsize", "36", "-fill", "#000000b0",
           "-gravity", "southwest", "-annotate", "+97+76", text.lower(), "-blur", "0x4", ")", "-composite",
           "-font", FONT, "-pointsize", "36", "-fill", "#ffffffee", "-gravity", "southwest",
           "-annotate", "+96+78", text.lower(), "-quality", "93", dst)


shots = [g for s in beats["sections"] for g in s["shots"]]
n = int(round(beats["duration"] * fps))
jobs, card_files = [], {}
for j in range(n):
    t = j / fps
    g = next((g for g in reversed(shots) if t >= g["start"] - 1e-9), shots[0])
    dst = os.path.join(out, f"{j:05d}.jpg")
    if g.get("card"):
        key = g["card"]
        if key not in card_files:
            card_files[key] = os.path.join(cards, f"card{len(card_files)}.jpg")
            card(key, card_files[key])
        os.symlink(os.path.abspath(card_files[key]), dst)
        continue
    last = g["frames"] - 1
    hold = g.get("hold")
    if hold == "last":
        f = last
    elif hold == "first":
        f = min(g.get("from", 0), last)
    else:
        f = min(g.get("from", 0) + int((t - g["start"]) * g["fps"] * g["speed"]), last)
    src = os.path.join(paths[g["id"]], f"{f:04d}.jpg")
    if g.get("caption") and t >= g["start"] + 0.2:
        jobs.append((src, g["caption"], dst))
    else:
        os.symlink(os.path.abspath(src), dst)

with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as ex:
    list(ex.map(lambda a: captioned(*a), jobs))
print(f"{out}: {n} frames, {len(jobs)} captioned, {len(card_files)} cards")
