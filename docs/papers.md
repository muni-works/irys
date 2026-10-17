# The Pareto frontier, 2026-10-17

What irys implements, in order. ✓ = arXiv id verified; others are venue-only
papers, cited from memory — check before quoting.

| # | Layer | Papers |
|---|---|---|
| 1 | adjoint contract | Path Replay Backprop (Vicini+ TOG 2021); Dr.Jit 2202.01284 ✓; Radiative Backprop (Nimier-David+ 2020); RB w/ non-static geometry 2504.05750 ✓ |
| 2 | spectral core | Hero wavelength (Wilkie+ EGSR 2014); Jakob–Hanika sigmoid upsampling (EG 2019); one-to-many upsampling 2306.11464 ✓ |
| 3 | unified BVH | 3DGRT 2407.07090 ✓; EVER 2410.01804 ✓; 3DGUT 2412.12507 ✓ |
| 4 | many lights | ReSTIR DI (Bitterli+ 2020); GRIS / ReSTIR PT (Lin+ 2022); MCMC decorrelation 2211.00166 ✓; differentiable reservoir PT 2406.16360 ✓ |
| 5 | visibility gradients | warped-area reparam (Bangaru+ 2020); projective sampling (Zhang+ 2023); rasterized edge gradients 2405.02508 ✓ |
| 6 | path guiding | online neural NASG 2303.08064 ✓; factorized 2506.00839 ✓; parametric mixtures 2504.04315 ✓; parity: practical path guiding (Müller+ 2017) |
| 7 | radiance cache | NRC 2106.12372 ✓ |
| 8 | RL batch lane | large-batch sim 2103.07013 ✓; GPUDrive 2408.01584 ✓; PyBatchRender 2601.01288 ✓; Madrona (SIGGRAPH 2023) |
| — | materials | OpenPBR (ASWF 2024); position-free MC layers (Guo+ 2018); diffuse-interface layers 2203.11835 ✓; neural appearance 2305.02678 ✓ |
| — | film | WS-KPN 2202.05977 ✓; two-stage adaptive 2103.16115 ✓; nonlinear Noise2Noise 2512.24794 ✓ |
| — | compiler | SLANG.D (Bangaru+ TOG 2023) — closest analogue to tang IR; megakernels considered harmful (Laine+ 2013) |

Rule: 1 and 2 go in before anything else. Both are cheap on day one and
painful to retrofit.
