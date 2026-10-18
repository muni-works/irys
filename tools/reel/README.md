# irys reel soundtrack

A 54-second, 124 BPM soundtrack (plus a 2.5 s bell tail) made by the stack
itself. Nothing is sampled: every sound is synthesised by `irys-reel`, a small
Rust tool (`publish = false`, its own workspace, so none of its dependencies
reach the irys library crates).

- **Harmony is the frames' light.** Each frame's mean spectrum (radiance in
  bins over 380–780 nm) is folded onto twelve pitch classes. Light at λ has
  frequency c/λ; forty octaves down, the visible band is F4–F5, so one octave
  of light is one octave of sound and A4 = 440 Hz sits at 619.7 nm (orange).
  Red 700 nm is G, yellow 580 nm A#, green 530 nm C, cyan 490 nm C#, blue
  450 nm D#, violet 400 nm F. Chroma is taken relative to equal-energy white
  (the red end spans more nanometres per semitone). One chord per beat:
  the root is the strongest pitch class, three more tones are picked by chroma
  weight plus a small consonance bonus; the chroma is exponentially smoothed
  (`chroma_smoothing`), the four pad voices are voice-led by least motion and
  glide (90 ms) between chords. Gold comes out around A♭/B♭, the sky around E,
  a thin film's interference peak walks the band and the chords walk with it.
- **Percussion is struck kosm materials.** Free–free Euler–Bernoulli bar
  modes from each material's ρ and E (`f_n = (β_nL)²/(2πL²)·√(E t²/12ρ)`),
  decays `π f η` (kosm's `audio::bar_modes`), strike weights from the mode
  shapes, a Hertz-contact rolloff and a millisecond of contact noise. The
  constants are kosm's library values, copied into `src/modal.rs`:

  | voice | material | geometry | modes (Hz) |
  |---|---|---|---|
  | kick | skin | 70 × 20 mm pad (+ cavity, onset chirp) | 92, 252, 495 |
  | low knock | limestone | 1200 × 80 mm slab | 250, 690, 1353, 2237 |
  | clap | oak | 250 × 20 mm plank | 1260, 3472, 6807 |
  | hat | glass | 50 × 4 mm rod | 8703 |
  | ping | anodised Al (CRC, η est.) | 120 × 3 mm strip | 1083, 2984, 5850 |
  | crash-bell | brass | 300 × 10 mm bar | 392, 1080, 2117, 3499 |
  | bell | gold | 200 × 8 mm bar | 416, 1147, 2248, 3716 |

- **Risers are the fits converging.** A riser reads a loss column from
  `docs/*.csv`, takes the best-so-far loss, normalises it to progress 0→1, and
  calls the fit converged where progress first reaches 0.98. The riser is
  stretched so that iteration lands on the drop: progress drives a band-passed
  noise sweep (300 Hz → 13 kHz) and a two-octave pitch rise, so a plateau in
  the loss is a plateau in the riser. Build → drop 1 is phase 2d (the rose,
  converged at iteration 112 of 300); break → drop 2 is the skin fit (48 of
  300); the intro's faint sweep is phase 2a.

## Steps

```bash
cd tools/reel
export CARGO_TARGET_DIR=../../target
cargo build --release
BIN=$CARGO_TARGET_DIR/release/irys-reel

# 1. manifest: the real one, or the committed stub (regenerate with `$BIN stub stub-manifest.json`)
M=/Users/cam/Developer/irys-assets/reel/manifest.json
[ -e "$M" ] || M=stub-manifest.json

# 2. render: out/soundtrack.wav (48 kHz, 24-bit stereo), out/preview.wav
#    (build + drop 1), out/spectrogram.png, out/beats.json. Seeded: same inputs, same bytes.
$BIN render --manifest "$M" --arrangement arrangement.json --docs ../../docs --out out

# 3. frames: cut to out/beats.json (beats, downbeats, drops, sections, and which
#    sequence plays when). Test frames for the mux:
mkdir -p out/stub-frames/reel
ffmpeg -v error -y -f lavfi -i testsrc2=size=480x270:rate=30 -t 20 -start_number 0 out/stub-frames/reel/%05d.png

# 4. mux: frames (%05d.png or .jpg from 0, FPS=30) + soundtrack -> out/irys-reel.mp4
./mux.sh out/stub-frames/reel
```

## Manifest

```json
{ "stub": false, "wavelengths_nm": [380, 390, ..., 780],
  "sequences": [ { "id": "gold", "path": "gold/%05d.png", "frames": 48, "fps": 24,
                   "caption": "...", "spectra": [[...41 radiances...], ...] } ] }
```

`frame_count` and `mean_spectra` are accepted as aliases; a sequence may carry
its own `wavelengths_nm`. Spectra are any linear radiance scale. Inside a
section a sequence's frame is `t · fps`, looping. `stub-manifest.json` is a
**stub**: illuminant × reflectance for gold, sky, skin, a thinning soap film,
the classroom and the pavilion, not irys renders.

## Timeline (124 BPM, bar = 1.935 s)

| bars | section | seconds | sequences | drums |
|---|---|---|---|---|
| 0–4 | intro | 0.00–7.74 | sky, pavilion | stone, glass ticks, gold |
| 4–8 | build | 7.74–15.48 | classroom, skin | kick, oak roll; phase 2d riser |
| 8–16 | drop 1 | 15.48–30.97 | gold, thin film | four on the floor, arp |
| 16–20 | break | 30.97–38.71 | sky, thin film | gold bells; skin-fit riser (bars 18–20) |
| 20–26 | drop 2 | 38.71–50.32 | skin, gold, pavilion | full, more arp |
| 26–28 | outro | 50.32–54.19 (+2.5 s ring) | gold | one last strike |

Edit `arrangement.json` to change any of it; `beats.json` follows.
