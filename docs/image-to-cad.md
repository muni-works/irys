# Image to CAD, video to CAD

Research synthesis from 2026-10-18. Nothing here is built yet.

The goal is a photo in and an editable vcad program out; later, a video in and vcad plus a phyz scene out.

## The one equation

    score(program, θ) = −log p(photo | irys(phyz(vcad(program, θ))))   residual
                      + ½ log det F(θ)                                  parameter cost (Laplace)
                      + −log p(program)                                 program length (LLM / grammar)

Each term is in nats, so the shortest true description of the photo wins (MDL).
An edit to the program is accepted only if it shortens the description. That
stops the agent from adding a feature for every pixel.

## The loop

- Outer loop, discrete: an agent proposes edits to the vcad program. The program
  is canonicalised by an e-graph (Szalinski, 1909.12252) so equivalent programs
  are deduplicated. Programs with recurring structure are compressed into
  library primitives (DreamCoder, 2006.08381).
- Inner loop, continuous: Levenberg–Marquardt on SE(3) × shape params using irys
  Jacobians (bundle adjustment as the template; Solà 1812.01537). It runs
  coarse-to-fine as a blur homotopy, with OT or distance-transform edge losses
  for wide basins (this fixes the ±5° basin found in phase 2e). The implicit
  function theorem gives gradients through vcad's constraint solver and phyz's
  contact solver (Blondel 2105.15183). Relaxations (DiffCSG 2409.01421, soft
  CSG, Gumbel) are used for proposals only; every candidate is snapped back to
  an exact BRep before it is scored.
- Judge: the Fisher information F = JᵀΣ⁻¹J gives covariances (Cramér–Rao). Its
  near-null eigenvectors are exactly what the photo cannot determine. The next
  view or motion is chosen to maximise log det F (D-optimal).

## Identifiability

- One image: the scale gauge (fixed by an AprilTag; MoGe-2 2507.02546 as a
  prior), hidden faces, and hole depth.
- Video: gravity becomes a metric ruler. Masses come only as ratios unless a
  known force acts. Inertia shows only under rotation, friction only under sliding.

## Steps and gates

| step | input | gate |
|---|---|---|
| v0 sim-to-sim | an irys render of a random vcad part | dimensions within 0.5 mm, topology correct on 20 parts |
| v1 one photo + AprilTag | a printed part with known CAD | reprint it; calipers agree within 1 mm |
| v2 multi-view | 3–10 photos | the same gate on occluded features |
| v3 video → CAD + phyz | a moving mechanism | the re-simulated video matches; it predicts a new motion |

## Where we're ahead, where the literature is

No published image-to-CAD system refines through an exact BRep with a physical,
differentiable renderer, or reports parameter uncertainty. The closest are
CAD-Recode (2412.14042), CADCodeVerify (2410.05340), Img2CAD (2410.03417) and
Articulate-Anything (2410.13882), which are feed-forward or rely on gradient-free
VLM loops. gradSim (2104.02646) is the v3 architecture built on weaker physics.

The literature is ahead on learned proposal priors (DeepCAD 2105.09492, Fusion
360 Gallery 2010.02392), VLM articulation priors, and monocular metric depth.
Use those to initialise; don't rebuild them.

Metrics: Chamfer, IoU, invalid-program rate, and parameter error on the
e-graph canonical form. The last is the editability metric the field lacks.
