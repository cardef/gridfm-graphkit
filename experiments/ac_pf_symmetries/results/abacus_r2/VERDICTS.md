# R2: M0+Canon vs M1 on held-out grids — verdicts (2026-10-03)

Predictions: `PREREGISTRATION.md` (fixed before launch). Tables and the scoring: `REPORT.md`, written by
`r2_report.py`. 12 runs (2 folds × 2 arms × 3 seeds), all on abacus-004, identical code hashes.

| id | verdict | what the numbers say |
|---|---|---|
| P17 held-out grids | PASS (no separation) | M1's means are lower on every (grid, channel): case30 VA 3.45 vs 5.34 deg, VM 0.0082 vs 0.0109; case57 VA 5.55 vs 8.26 deg, VM 0.0177 vs 0.0180; but M1's seed spread is large (VA ± 2-3 deg) and every range overlaps canon's |
| P18 training grids | FAIL: 4 of 6 (grid, fold) separate | case118, both folds: M1 better, VA [0.69, 0.88] vs [1.23, 1.99] and [0.79, 0.99] vs [1.66, 2.09] deg (PG 27-34 vs 73-76 MW, PBE 0.61-0.63 vs 1.1-1.2 MVA); case30 trained in fold 1: M1 better, [0.19, 0.27] vs [0.38, 0.47]; case14 in fold 1: M1 worse, [0.14, 0.25] vs [0.108, 0.118]; case14 and case57 in fold 0 overlap |
| P19 (exploratory) | — | canon trained on three grids vs on case14 alone (single-grid replicate, different data sizes): held-out VM lower (case30 0.011 vs 0.016, case57 0.018 vs 0.025), held-out VA on case30 higher (5.3 vs 1.7 deg, almost all offset) |

R7 on case118 (in distribution, both folds): canon's VA error is mostly the per-graph offset relative to REF
(1.40-1.57 of 1.65-1.87 deg); M1 cuts the offset to 0.55-0.69 deg and the branch differences from 0.25-0.27 to
0.13-0.14 deg. Canon's offset grows with grid size (0.11-0.15 deg on case14, 0.24 on case57, 1.4-1.6 on case118).

## Reading

With the physics correction in every layer, the branch representation is worth most where node angles are
furthest from the reference: on the largest training grid it halves the angle error, and the error it removes is
the offset that R7 found dominant. On the smallest grid it can be worse. Zero-shot on the two small held-out grids
the means favour M1 but stay inside seed variability. Not tested yet, and the direct test of H1 that this suggests:
a held-out grid larger than the training ones (e.g. train {14, 30, 57}, hold out 118), where the offset the
representation removes is largest.
