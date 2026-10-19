# recon v0 gate (1 view)

**17/20 pass** (topology correct and every dimension within 0.5 mm); topology correct 17/20; 6764 s total, 338 s per part.

| seed | truth | recovered | topo | max err mm | worst param | MDL (truth) | iters | renders | s | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 3000 | `chamfer[hole,boss]` | `chamfer[hole,boss]` | yes | 0.294 | f1.bh | 26209 (26021) | 349 | 1012 | 269 | PASS |
| 3001 | `sharp[hole]` | `sharp[hole]` | yes | 0.027 | f0.y | 26005 (25998) | 77 | 275 | 109 | PASS |
| 3002 | `fillet[boss,boss]` | `fillet[boss,boss]` | yes | 0.199 | f1.bh | 25779 (25952) | 242 | 710 | 299 | PASS |
| 3003 | `sharp[hole,counterbore,boss]` | `sharp[hole,counterbore,boss]` | yes | 0.250 | f2.r | 25725 (25618) | 436 | 1335 | 953 | PASS |
| 3004 | `chamfer[hole,linear_holes×2,boss]` | `chamfer[hole,hole,hole,boss]` | no | - | - | 27888 (25669) | 605 | 1680 | 718 | FAIL |
| 3005 | `sharp[counterbore,slotX]` | `sharp[counterbore,slotX]` | yes | 0.420 | f0.cd | 26133 (26142) | 198 | 599 | 131 | PASS |
| 3006 | `chamfer[hole,hole,boss]` | `chamfer[hole,hole,slotY]` | no | - | - | 31321 (25424) | 491 | 1458 | 267 | FAIL |
| 3007 | `chamfer[hole,counterbore,boss]` | `chamfer[hole,counterbore]` | no | - | - | 37847 (25792) | 341 | 1056 | 397 | FAIL |
| 3008 | `sharp[boss,boss]` | `sharp[boss,boss]` | yes | 0.324 | f0.x | 25660 (25647) | 135 | 427 | 115 | PASS |
| 3009 | `sharp[circular_holes×3]` | `sharp[circular_holes×3]` | yes | 0.052 | base.w | 25775 (25849) | 137 | 495 | 124 | PASS |
| 3010 | `sharp[slotY]` | `sharp[slotY]` | yes | 0.139 | f0.len | 26080 (26146) | 82 | 280 | 64 | PASS |
| 3011 | `fillet[counterbore,slotX,boss]` | `fillet[counterbore,slotX,boss]` | yes | 0.365 | f0.cd | 25565 (25618) | 253 | 805 | 395 | PASS |
| 3012 | `sharp[circular_holes×4,counterbore]` | `sharp[circular_holes×4,counterbore]` | yes | 0.118 | f1.cd | 25706 (25597) | 494 | 1670 | 715 | PASS |
| 3013 | `sharp[hole,hole,slotX]` | `sharp[hole,hole,slotX]` | yes | 0.048 | f2.y | 25591 (25641) | 453 | 1423 | 301 | PASS |
| 3014 | `chamfer[linear_holes×3,boss]` | `chamfer[linear_holes×3,boss]` | yes | 0.077 | f1.bh | 25608 (25657) | 695 | 2211 | 1300 | PASS |
| 3015 | `fillet[hole,boss]` | `fillet[hole,boss]` | yes | 0.357 | f1.y | 26028 (25962) | 282 | 901 | 128 | PASS |
| 3016 | `chamfer[hole,hole]` | `chamfer[hole,hole]` | yes | 0.070 | f0.r | 26067 (26002) | 390 | 1252 | 232 | PASS |
| 3017 | `fillet[counterbore]` | `fillet[counterbore]` | yes | 0.329 | f0.cd | 25691 (25750) | 171 | 589 | 116 | PASS |
| 3018 | `fillet[counterbore]` | `fillet[counterbore]` | yes | 0.043 | f0.x | 25584 (25620) | 121 | 390 | 91 | PASS |
| 3019 | `sharp[slotX]` | `sharp[slotX]` | yes | 0.208 | f0.len | 25702 (25814) | 74 | 242 | 40 | PASS |
