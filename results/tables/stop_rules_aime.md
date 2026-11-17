# Stopping rules on AIME (from the fixed-cap runs)

| Rule | Accuracy (all copies) | Copy 1: kJ/query | Copy 2: kJ/query | Copy 1: saved vs full | Copy 2: saved vs full | Stopped early |
|---|---|---|---|---|---|---|
| full thinking | 68% (81/120) | 117.0 | 140.5 | +0% | +0% | 0 |
| fixed cap 4k | 41% (49/120) | 39.5 | 43.0 | +66% | +69% | 115 |
| fixed cap 8k | 56% (67/120) | 62.2 | 67.1 | +47% | +52% | 84 |
| fixed cap 16k | 68% (81/120) | 90.7 | 104.3 | +22% | +26% | 48 |
| agree-twice (probe cost paid) | 65% (78/120) | 136.9 | 149.2 | -17% | -6% | 52 |
| agree-twice (probes free) | 65% (78/120) | 101.4 | 111.1 | +13% | +21% | 52 |
| oracle: first correct probe | 73% (88/120) | 96.0 | 101.8 | +18% | +28% | 81 |

## agree-twice by difficulty (probe cost paid)

| Group | Full: acc | Rule: acc | Stopped at 4k / 8k / 16k / not early |
|---|---|---|---|
| 1-5 (easier) | 82% | 85% | 14 / 6 / 1 / 19 |
| 6-10 (medium) | 68% | 62% | 12 / 5 / 1 / 22 |
| 11-15 (hardest) | 52% | 48% | 5 / 7 / 1 / 27 |

Notes:
- Probes are the forced answers of `run_caps.py` (answer up to 4,096 tokens each).
- 'Stopped early' = the rule stopped before the model finished thinking on its own.
- Oracle uses the gold answer; no real rule can do better than it with these probe points.
