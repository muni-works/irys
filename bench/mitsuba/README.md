# Mitsuba 3 baseline

irys's differentiable renderer (`crates/irys-diff`) against Mitsuba 3 on the
same scenes, the same loss and the same machine: the material gradient (path
replay on both sides) and the pose gradient (irys's edge sampling against
Mitsuba's projective sampling). Every number below is in `results.md`,
regenerated from the raw CSVs in `out/` by `compare.py`.

## Machine and versions

| | |
|---|---|
| machine | Apple M4 Max (16 CPU cores, 40-core GPU, Metal 4), 48 GB |
| OS | macOS 27.0.1 (26A434) |
| irys | this repository at the commit that adds this directory; `rustc 1.98.1`, release, wgpu on Metal, hardware ray queries on |
| Mitsuba | 3.9.1 (pip wheel), Dr.Jit 1.5.0, Python 3.13.3; variant `metal_ad_spectral` (the GPU); `llvm_ad_rgb` only for the sanity check |

Both run on the same GPU, one at a time, nothing else running.

## Reproduce

```bash
cargo run --release -p irys-diff --features gpu --example mitsuba_bench -- bench/mitsuba/out
python -m venv mi && mi/bin/pip install mitsuba==3.9.1 numpy
cd bench/mitsuba && ../../mi/bin/python run_mitsuba.py out material pose
../../mi/bin/python run_mitsuba.py out pose cd-only grad skip-cd --tag=_hi --spps=64,256 --seeds=24
../../mi/bin/python sanity_projective.py > out/sanity_projective.txt
../../mi/bin/python compare.py out > results.md
```

(For `--tag=_hi`, copy `mitsuba_pose_target.exr` and `mitsuba_pose_cd.csv` to
the `_hi` names first.) Diagnostics used while chasing the pose bias:
`--integrator=prb|direct_projective`, `--guiding=none`, `--guiding_rounds=N`,
`--conductor=diffuse`, `--variant=llvm_ad_rgb`, `--spps=…`, `--seeds=N`,
`--tag=_x` (separate output files).

## What is compared

**Protocol.** Each system renders its own target (4096 spp) and takes the
gradient of `mean((I − T)²)` over pixels and channels — the same
normalisation on both sides — at the same start, with decorrelated primal and
adjoint seeds (irys `gradient_l2(seeds = (2s, 2s+1))`, Mitsuba
`render(seed = 2s, seed_grad = 2s+1)`), so both estimate the gradient of the
expected image's loss. Per budget, 64 seeds (material) or 40 (pose), after
three warm-up gradients (Mitsuba compiles its kernels then). Wall-clock is
host time per gradient, from scene update to the gradient on the host.

**Reference.** Each system is checked against its own central differences
(h = 0.01) of `mean((I_a − T)(I_b − T))` over two independent renders,
common seeds at ±h: that is unbiased for the expected image's loss. A first
version took differences of one noisy image's MSE, which adds `Var I` — and
`Var I` depends on roughness and pose — and put *both* renderers 6–150 % off
their references in the same direction; it was the reference, not the
renderers.

**Figures of merit.** Per parameter: z of the mean against the system's own
CD; relative variance `var / CD²` of one gradient; milliseconds per gradient;
and efficiency `1 / (relative variance × seconds)`. Efficiency is the inverse
of the time to reach a given relative variance, so the ratio of two
efficiencies is the equal-variance time ratio. The two renderers' images are
close but not identical (below), so gradients are compared in relative terms,
never value against value.

### 1. Material gradient

Scene: irys `Scene::pair` — one 16 × 32 UV-sphere mesh placed twice (one
squashed and turned) as a GGX conductor on a 16 × 16 grid floor (Lambertian,
albedo 0.5), lit by irys's sky gradient and sun lobe; 128 × 128, depth 4
with NEE and MIS (irys `complete`, i.e. Mitsuba `max_depth = 5`), no Russian
roulette. θ = (α, F0₄₅₀, F0₅₅₀, F0₆₅₀) at (0.35, 0.95, 0.75, 0.45); target
(0.2, 0.34, 0.78, 1.0). Mitsuba integrator `prb`.

### 2. Pose gradient

Scene: irys phase 2e's arm — a static post, a conductor upper link and a
diffuse forearm (boxes, each pre-rolled), on a floor; 96 × 72, depth 2 with
NEE and MIS (`max_depth = 3`). φ = (shoulder pitch, shoulder yaw, elbow) at
(0.30, 0.40, −0.60) rad; target (0.35, 0.45, −0.70). The rig is written out
(`pose.json`) and rebuilt in Dr.Jit (`run_mitsuba.py: arm_place`); its
vertices match irys's to 1.5e-7. irys: device edge samples (front-only
silhouettes) traced on the GPU plus the CPU interior term (f64 duals) at
budgets (4096 samples, 1 spp), (16 384, 4), (65 536, 16). Mitsuba:
`prb_projective` (default octree guiding) at 1, 4, 16 spp (`sppc = sppp =
sppi = spp`), film `sample_border` on.

## Mismatches (everything that is not the same)

| | irys | Mitsuba | effect |
|---|---|---|---|
| spectrum | hero wavelengths, 4 per path, its own lane → RGB weights | `metal_ad_spectral`, 4 wavelengths per path, CIE | colours differ slightly |
| environment | analytic: `sky · tint(λ) (0.15 + 0.25 max(d_z, 0)) + 6 e^{40 (d·sun − 1)}` | the same baked to a 2048 × 1024 lat-long map, RGB from `tint` at 610/550/465 nm, spectrum by Mitsuba's RGB upsampling | sun lobe resampled (≈ 0.18° texels against a 9° lobe); sky tint approximated |
| light sampling | the sun lobe sampled exactly (vMF) mixed with the uniform sphere, share by peak radiance | envmap importance sampling of the bitmap | different variance, both unbiased |
| BSDF | `ggx.rs` graph | `IrysGGX` Python plugin: the same formulas (height-correlated Smith, Schlick, three hat bands, visible-normal sampling via Mitsuba's GGX sampler) | none intended; a plugin runs as JIT code, not Mitsuba's C++ |
| diffuse | Lambertian, two-sided | Mitsuba `diffuse`, one-sided | identical on these closed meshes and the up-facing floor |
| pixel filter | box, one jittered sample per lane | `hdrfilm` with a box filter | none |
| shading normals | geometric (meshes carry none) | `face_normals`, welded mesh | none |
| sampler | irys's own per-lane generator | `independent` | different noise |
| pose terms | primary-visibility edges + interior; no shadow or reflection edges (deferred in phase 2e) | primary and indirect visibility (projective) + continuous | Mitsuba estimates more of the gradient; on this scene the indirect part is small (irys alone matches CD) |
| pose timing | edge term on the GPU, interior term on the CPU (16 threads) | all on the GPU | irys's interior is CPU time |

The targets differ by the mean relative L1 per pixel in `results.md`
("target images"): close enough that each system's own CD agrees with the
other's to 2 % on the pose's pitch, not identical.

## Results

See `results.md` for every row. Summary:

**Material gradient: irys wins, by 7–26× at equal variance.** Both
renderers are unbiased against their own references at every budget (|z| ≤
1.7, 64 seeds). At equal spp irys's gradient has 1.2–4.6× less relative variance
(sun lobe sampled exactly against Mitsuba's bitmap importance sampling,
mostly) and takes 2.7–5.5 ms against Mitsuba's 21–31 ms (128², depth 4, 1–16
spp). Equal-variance time ratio, best budget of each: α 7.0×, F0₄₅₀ 12.6×,
F0₅₅₀ 25.5×, F0₆₅₀ 17.0×. Mitsuba's wall-clock moves between runs (an
earlier identical run: 13.8 / 18.2 / 22.3 ms, so 4.8–17.6×); irys's moved
by < 0.6 ms. Part of Mitsuba's time is per-call overhead (16× the spp costs
1.5× the time), which a larger image would amortise.

**Pose gradient: no valid efficiency comparison — Mitsuba's projective
gradient is biased on this scene in my setup.** irys matches its CD on pitch
and yaw (z +0.8 and −2.6 at the largest budget) and reads 15 % high on the
elbow (z +3.3; irys has no shadow or reflection edges, phase 2e). Mitsuba's
`prb_projective` converges to the wrong value: pitch −2.1e-4 at 256 spp
against its own CD −6.45e-4 (z +13), yaw and elbow off the other way; its
continuous part alone (`prb`) is small (pitch −8e-6, like irys's interior
−2e-5), so the error is in the boundary term. The same holds with guiding off
or four guiding rounds, with `direct_projective`, on the CPU backend, with the
conductor made diffuse, and with welded meshes. Mitsuba's CD agrees with
irys's to 2 %, so the scenes match. `sanity_projective.py` shows the same
Mitsuba setup reproducing CD on a box translating in front of an environment
(welded boxes: ratio 0.99–1.03, both backends); a box whose seams split its
vertices (Mitsuba's own `cube`) reads 1.86×. I did not find what in the arm scene breaks it, so the
pose rows report Mitsuba's numbers as measured and claim nothing from them.
For the record, Mitsuba took 260–340 ms per pose gradient (1–256 spp) and
irys 8.6–60 ms (of which the CPU interior term is most of the larger
budgets).

## Files

- `run_mitsuba.py` — Mitsuba's side: BSDF plugin, environment, scenes, gradients, CD.
- `compare.py` — `results.md` from `out/*.csv`.
- `sanity_projective.py` — Mitsuba's projective gradient against CD on boxes.
- `out/` — the exported scenes (JSON, PLY), the targets (irys PFM, Mitsuba
  EXR), and the raw CSVs: `{irys,mitsuba}_{material,pose}.csv` (one row per
  budget and seed: gradient, loss, ms), `*_cd.csv` (one row per parameter and
  seed), `mitsuba_pose_hi.csv` (64 and 256 spp), `sanity_projective.txt`.
- irys's side is `crates/irys-diff/examples/mitsuba_bench.rs`.
