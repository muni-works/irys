# recon v0 gate (3 views)

**17/20 pass** (topology correct and every dimension within 0.5 mm); topology correct 17/20; 5488 s total, 274 s per part.

| seed | truth | recovered | topo | max err mm | worst param | MDL (truth) | iters | renders | s | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 3000 | `chamfer[hole,boss]` | `chamfer[boss]` | no | - | - | 87015 (74718) | 133 | 380 | 68 | FAIL |
| 3001 | `sharp[hole]` | `sharp[hole]` | yes | 0.021 | f0.r | 75338 (75499) | 79 | 258 | 38 | PASS |
| 3002 | `fillet[boss,boss]` | `fillet[boss,boss]` | yes | 0.109 | f1.x | 74780 (74687) | 296 | 796 | 142 | PASS |
| 3003 | `sharp[hole,counterbore,boss]` | `sharp[hole,counterbore,boss]` | yes | 0.102 | f2.x | 74720 (74459) | 436 | 1333 | 448 | PASS |
| 3004 | `chamfer[hole,linear_holes×2,boss]` | `chamfer[hole,linear_holes×2,boss]` | yes | 0.076 | f2.bh | 75120 (74978) | 760 | 2023 | 547 | PASS |
| 3005 | `sharp[counterbore,slotX]` | `sharp[counterbore,slotX]` | yes | 0.178 | f0.cd | 75317 (75281) | 151 | 456 | 114 | PASS |
| 3006 | `chamfer[hole,hole,boss]` | `chamfer[hole,hole,boss]` | yes | 0.090 | f2.bh | 74478 (74276) | 495 | 1497 | 315 | PASS |
| 3007 | `chamfer[hole,counterbore,boss]` | `fillet[hole,counterbore,boss]` | no | - | - | 75361 (74666) | 279 | 849 | 400 | FAIL |
| 3008 | `sharp[boss,boss]` | `sharp[boss,boss]` | yes | 0.072 | f1.y | 74318 (74301) | 151 | 438 | 172 | PASS |
| 3009 | `sharp[circular_holes×3]` | `sharp[circular_holes×3]` | yes | 0.014 | f0.phase | 74783 (74953) | 121 | 376 | 102 | PASS |
| 3010 | `sharp[slotY]` | `sharp[slotY]` | yes | 0.077 | f0.len | 74714 (74824) | 62 | 213 | 116 | PASS |
| 3011 | `fillet[counterbore,slotX,boss]` | `fillet[slotX,boss]` | no | - | - | 92892 (74558) | 166 | 516 | 313 | FAIL |
| 3012 | `sharp[circular_holes×4,counterbore]` | `sharp[circular_holes×4,counterbore]` | yes | 0.136 | f1.cd | 75031 (74702) | 360 | 1040 | 820 | PASS |
| 3013 | `sharp[hole,hole,slotX]` | `sharp[hole,hole,slotX]` | yes | 0.078 | f2.len | 74260 (74361) | 233 | 688 | 166 | PASS |
| 3014 | `chamfer[linear_holes×3,boss]` | `chamfer[linear_holes×3,boss]` | yes | 0.070 | f1.x | 74984 (74970) | 490 | 1573 | 668 | PASS |
| 3015 | `fillet[hole,boss]` | `fillet[hole,boss]` | yes | 0.100 | f1.x | 75111 (75137) | 393 | 1168 | 401 | PASS |
| 3016 | `chamfer[hole,hole]` | `chamfer[hole,hole]` | yes | 0.022 | base.w | 74746 (74761) | 224 | 680 | 134 | PASS |
| 3017 | `fillet[counterbore]` | `fillet[counterbore]` | yes | 0.176 | f0.cd | 74963 (75065) | 117 | 384 | 231 | PASS |
| 3018 | `fillet[counterbore]` | `fillet[counterbore]` | yes | 0.142 | f0.cd | 75395 (75179) | 190 | 535 | 207 | PASS |
| 3019 | `sharp[slotX]` | `sharp[slotX]` | yes | 0.023 | base.w | 74694 (74794) | 99 | 302 | 87 | PASS |
