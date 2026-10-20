# Desk photo: v3 inventory

Photo `desk-IMG_0579.jpg` (1600×1200, f35 24 mm, horizontal FOV 71.6°). Net input 128×96; appearance fitted at 256×192; search at 512×384 on irys's OpenPBR render.

- Net (`out/neural/v3/desk2-run/weights.json`): 11 detections in 0.025 s; support plane from the plane head: camera 280 mm above it, pitch 17.6°.
- Net gauges: none recognised with confidence by the net.
- Births from the residual: 0 objects in 108 s.
- Scale after the search: ×1.000 from nothing (no gauge found; the plane head's scale); every length below is scaled by it.
- Appearance: 170 renders in 6.2 s; tone γ 0.80; defocus σ 6.00 px at 512.
- Search: 399 shape rebuilds, 140 TLAS-only moves, 539 renders, 366 edge Jacobians in 160 s; kept 8 of 11; scene MDL 21875 → 21804 nats (τ 90.7).

**Read this first.** The desk is still mostly not recovered. Checked by eye
against the photo (the numbers are the labels in the bottom-right panel of
[neural-v3-desk.png](neural-v3-desk.png)):

- **Not found:** the keyboard, the white mouse, the can (so no gauge set
  the scale: it is the plane head's), the monitor, the humidifier, the
  orange honeycomb case, the black printed bracket in the foreground. The
  success criterion (keyboard, can, mouse and monitor with sensible sizes
  from the can and key-pitch gauges) is **not met**.
- **Roughly placed:** #3 (an arm with a speaker head) over the left wall,
  where a speaker and webcams hang on arms; #6 (an arm) and #2 over the
  right wall near the real power strip and the right speaker's arm; #4 at
  the frame's left edge (wall clutter).
- **Wrong:** #1 (a power strip on the wall) and #5 (a speaker) on the
  monitor's face; #7 (a mouse) on the paper and the red knife at the bottom
  left; #8 (a webcam) on the empty desk between the can and the keyboard.
- **What changed from v2:** the desk-trained network now proposes wall
  hardware (two arms, a power strip on the wall) and a plausible camera
  (280 mm above the desk, pitch 17.6°, against v2's 378 mm and 35.7°), but it
  still does not see the desk's own objects at 128 × 96. Births from the
  residual found 26 unexplained components on the desk, and none paid for
  its program: the photo's residual dispersion is τ ≈ 91 (synthetic scenes:
  9–15), so after tempering no single object shortens the description by
  its length. The appearance model is the bottleneck here as it is on the
  synthetic scenes: the fitted defocus (σ 6 px at 512) and tone (γ 0.80, the
  grid's edge) show the fit compensating for what the scene language does
  not hold (the screen's light, the wall's clutter, the lamp bar).
- **±σ** are from the search's Fisher information in log units at the
  plane head's scale; with this model error they are optimistic.

| # | class | family | program (search's scale) | x, y (mm), yaw (°) | dimensions mm ± σ (gauge scale) | born |
|---|---|---|---|---|---|---|
| 1 | power strip | product | `power_strip product fillet=all on-wall w=231.93 d=72.49 h=36.85 r=12.45 lift=63.16 @ 197.3 -130.2 -2.1` | 197, -130, -2 | w 232 ± 15, d 72 ± 41, h 37 ± 14, r 12 ± 12, lift 63 ± 6 |  |
| 2 | mouse | product | `mouse product fillet=all w=69.08 d=131.54 h=47.38 r=12.14 @ 372.1 -185.4 -23.8` | 372, -185, -24 | w 69 ± 21, d 132 ± 10, h 47 ± 18, r 12 ± 6 |  |
| 3 | wall arm | arm | `wall_arm arm head=speaker l1=209.14 l2=72.30 r=18.47 z=132.23 hw=46.42 hh=105.27 hd=130.44 a=46.99 @ -306.6 -111.7 3.3` | -307, -112, 3 | l1 209 ± 9, l2 72 ± 5, r 18 ± 6, z 132 ± 4, hw 46 ± 8, hh 105 ± 12, hd 130 ± 18, a 47 ± 2 |  |
| 4 | speaker | product | `speaker product fillet=vertical w=109.86 d=116.85 h=155.88 r=17.98 @ -401.4 -351.5 52.3` | -401, -352, 52 | w 110 ± 36, d 117 ± 48, h 156 ± 12, r 18 ± 12 |  |
| 5 | speaker | product | `speaker product fillet=vertical w=105.38 d=102.83 h=139.29 r=14.90 @ -135.6 -4.3 32.8` | -136, -4, 33 | w 105 ± 38, d 103 ± 92, h 139 ± 8, r 15 ± 36 |  |
| 6 | wall arm | arm | `wall_arm arm head=speaker l1=202.80 l2=53.64 r=40.00 z=71.00 hw=32.35 hh=93.74 hd=95.95 a=48.98 @ 218.4 -231.4 2.6` | 218, -231, 3 | l1 203 ± 40, l2 54 ± 57, r 40 ± 6, z 71 ± 6, hw 32 ± 51, hh 94 ± 56, hd 96 ± 3034164, a 49 ± 10 |  |
| 7 | mouse | product | `mouse product fillet=vertical w=72.31 d=72.98 h=33.00 r=9.76 @ -215.1 -587.6 -0.0` | -215, -588, -0 | w 72 ± 14, d 73 ± 6, h 33 ± 3, r 10 ± 16 |  |
| 8 | webcam | product | `webcam product fillet=all w=88.76 d=60.81 h=27.92 r=7.74 @ 75.0 -450.8 -21.9` | 75, -451, -22 | w 89 ± 45, d 61 ± 14, h 28 ± 2, r 8 ± 5 |  |
