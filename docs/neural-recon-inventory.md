# Desk photo: neural recon inventory

Photo `desk-IMG_0579.jpg` (1600×1200, f35 24 mm, horizontal FOV 71.6°). Net input 128×96; search at 256×192.

- Net: 12 detections in 0.029 s. Support plane from the plane head: camera 378 mm above it, pitch 35.7°.
- Scale: no known-size object was recognised with confidence; the scale is the plane head's alone.
- MDL search: 2853 irys renders in 18.5 s; kept 12 of 12 detections; full-frame residual 10516846 → 9637043 nats.
- Unexplained: 43.2 % of the frame is table-labelled yet differs from the model by more than 0.15 (linear).

**Read this first.** The recovered scene is mostly wrong. The network was
trained only on synthetic tabletops (objects standing on one plane, seen
from 22–65° above). In this photo the camera is low, half the frame is a
wall and pegboard, and the monitor, speakers, webcams and the power strip
hang on that wall. Cables, the screen content, the soft items and the
pegboard clutter are outside the scene language, so they are unmodelled.
Checked by eye against the photo:

- **Not found:** the monitor, the keyboard, the MX-style mouse, the game
  controller, the power strip, the orange honeycomb case, the gear plates.
  The soda can (a scale gauge) was not recognised as one, so no gauge set
  the scale; it comes from the plane head alone and is not trustworthy.
- **Roughly placed:** #9 (tray, p 0.79) sits on the black 3D-printed bracket
  in the left foreground, which is the closest thing in the language; #5
  (humidifier, p 0.99) is a large lathe hypothesis over the monitor's right
  half, not over the real humidifier; #6/#7 cover the can and the cable
  clutter left of it; #10–#12 cover the mouse and the right-hand desk items.
- **Unmodelled:** cables, the screen image, the wall and pegboard, the lamp
  rail, soft and translucent items. There is no splat or mesh fallback in
  this pipeline yet; those pixels are left to the background model (43 % of
  the frame is background-labelled yet off the model by >0.15 linear).
- **±σ:** from the search's Fisher information in log units. 0 means the
  window pins the parameter far below 1 mm *given this model*, which with
  this model error says nothing; the huge values are parameters the photo
  does not constrain (fillet radii, a lathe's top radius).

| # | class (p) | family | program | x, y (mm), yaw (°) | dimensions mm ± σ | gauge |
|---|---|---|---|---|---|---|
| 1 | speaker (0.12) | product | `speaker product fillet=vertical w=88.29 d=50.63 h=87.31 r=19.29 @ -87.6 16.2 -13.7` | -88, 16, -14 | w 88 ± 0, d 51 ± 0, h 87 ± 0, r 19 ± 0 |  |
| 2 | speaker (0.01) | product | `speaker product fillet=vertical w=105.41 d=15.55 h=45.03 r=7.00 @ 101.6 37.5 -5.4` | 102, 38, -5 | w 105 ± 0, d 16 ± 0, h 45 ± 0, r 7 ± 221248 |  |
| 3 | webcam (0.27) | product | `webcam product fillet=vertical w=83.47 d=54.73 h=56.13 r=18.41 @ 298.2 -77.6 62.8` | 298, -78, 63 | w 83 ± 0, d 55 ± 0, h 56 ± 0, r 18 ± 0 |  |
| 4 | soda can (0.06) | lathe | `soda_can lathe h=191.96 r0=19.67 r1=28.86 r2=33.86 r3=2.50 @ -255.9 11.9 0.0` | -256, 12, 0 | h 192 ± 0, r0 20 ± 0, r1 29 ± 0, r2 34 ± 0, r3 2 ± 79057 |  |
| 5 | humidifier (0.99) | lathe | `humidifier lathe h=206.05 r0=79.11 r1=67.28 r2=56.06 r3=44.92 @ 182.1 -56.0 0.0` | 182, -56, 0 | h 206 ± 0, r0 79 ± 0, r1 67 ± 0, r2 56 ± 0, r3 45 ± 0 |  |
| 6 | soda can (0.53) | lathe | `soda_can lathe h=109.41 r0=82.00 r1=62.85 r2=35.99 r3=39.05 @ -118.6 -131.3 0.0` | -119, -131, 0 | h 109 ± 0, r0 82 ± 0, r1 63 ± 0, r2 36 ± 0, r3 39 ± 0 |  |
| 7 | bottle (0.29) | lathe | `bottle lathe h=110.51 r0=75.37 r1=48.86 r2=23.19 r3=48.36 @ -313.2 -130.4 0.0` | -313, -130, 0 | h 111 ± 0, r0 75 ± 0, r1 49 ± 0, r2 23 ± 0, r3 48 ± 0 |  |
| 8 | webcam (0.88) | product | `webcam product fillet=all stand w=51.75 d=54.70 h=43.46 r=2.82 lift=3.09 @ 34.3 -116.3 83.9` | 34, -116, 84 | w 52 ± 0, d 55 ± 0, h 43 ± 0, r 3 ± 0, lift 3 ± 0 |  |
| 9 | tray (0.79) | shell | `tray shell open ribs=1 w=132.73 d=75.23 h=27.42 wall=6.54 @ -145.8 -293.2 72.6` | -146, -293, 73 | w 133 ± 0, d 75 ± 0, h 27 ± 0, wall 7 ± 0 |  |
| 10 | speaker (0.44) | product | `speaker product fillet=all w=173.73 d=77.90 h=64.86 r=20.44 @ 197.8 -250.0 -10.7` | 198, -250, -11 | w 174 ± 0, d 78 ± 0, h 65 ± 0, r 20 ± 0 |  |
| 11 | speaker (0.81) | product | `speaker product fillet=vertical w=107.32 d=58.10 h=107.28 r=22.25 @ 102.9 -285.2 19.6` | 103, -285, 20 | w 107 ± 0, d 58 ± 0, h 107 ± 0, r 22 ± 0 |  |
| 12 | speaker (0.47) | product | `speaker product fillet=vertical w=167.07 d=67.90 h=88.65 r=30.56 @ 238.6 -337.7 7.3` | 239, -338, 7 | w 167 ± 0, d 68 ± 0, h 89 ± 0, r 31 ± 966269 |  |
