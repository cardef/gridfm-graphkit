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

## Fold 3: case118 held out (2026-10-05)

Six runs (2 arms × 3 seeds, train {14, 30, 57}) on abacus-007, same code hashes as folds 1-2. The fold-3 section of
`PREREGISTRATION.md` (09:03) and its scoring in `r2_report.py` (09:06) predate the first result (10:30). As in
folds 1-2, every run reached the 100-epoch cap without stopping early: both arms are budget-limited alike.

| id | verdict | what the numbers say |
|---|---|---|
| P20 case118 held out, VA | FAIL | m1 [10.37, 12.74] vs canon [10.06, 12.41] deg (means 11.2 vs 11.5). No zero-shot channel separates: VM [0.0059, 0.0083] vs [0.0054, 0.0069] p.u., PBE [17.4, 26.3] vs [13.7, 20.2] MVA, M1's means higher on both |
| P21 | not scored (P20 failed) | m1/canon ratio 0.96 on the offset, 0.99 on the rest: no gain to attribute |
| P22 (exploratory) | — | zero-shot case118, both arms: offset 6.9-7.1 deg, rest 8.8-8.9 deg, branch differences θ_f − θ_t wrong by 2.0-2.1 deg, 8× (canon) and 15× (M1) the same arm trained on case118 in folds 1-2 (0.25-0.27 and 0.13-0.14). In distribution M1 separates below canon on case30 (VA [0.20, 0.31] vs [0.64, 0.66] deg, VM, QG), case14 (VM, QG) and case57 (QG), but canon is the outlier: its case14 errors are 2-9× those of folds 1-2 (VA, VM, QG), case30 1.4-4×, while M1's are in line with folds 1-2. Training composition and node type changed together; cause not identified |

### Trivial references (not pre-registered; `r2_baselines.py`, output in `BASELINES.txt`)

Same scenarios as the zero-shot evaluation, inputs only. flat: every angle = θ_ref, VM = 1. DC: one linear solve
B θ = P per graph (P from Pd and the Pg of PV buses, B_ft = Im Y_ft), θ_ref fixed, S1/S3/S4-exact by construction.
VA in deg; learned models: means over 3 seeds, in distribution the range of the fold means where the grid was trained.

| grid | flat | DC | canon zero-shot | M1 zero-shot | canon in dist | M1 in dist |
|---|---|---|---|---|---|---|
| case14 | 12.5 | 1.51 | — | — | 0.11-0.33 | 0.17-0.21 |
| case30 | 11.7 | 1.56 | 5.34 | 3.45 | 0.42-0.65 | 0.24 |
| case57 | 5.51 | 1.08 | 8.26 | 5.55 | 0.25-0.29 | 0.22 |
| case118 | 14.2 | 1.95 | 11.5 | 11.2 | 1.65-1.87 | 0.77-0.89 |

Branch differences: DC wrong by 0.34-0.53 deg, the learned models zero-shot by 0.73-2.07 (2-4× DC). DC's error on
case14/30 is almost all offset (1.46-1.51 of 1.51-1.56 deg), consistent with the losses it ignores. Zero-shot VM is
informative (0.006-0.018 against 0.027-0.038 for VM = 1).

## Reading (all three folds)

H1 for M1 is not supported on any held-out grid, including the one where the pre-registration expected it to be
easiest. The references put the representation question in its place: zero-shot, neither arm comes near a DC power
flow (2-8× its angle error on every held-out grid; on case57 canon is worse than predicting θ_ref everywhere, M1 equal
to it), and in distribution on case118 canon is only 4-15% below DC; M1 is below half of it. What fails on an
unseen grid is the branch-level quantity, which M1 takes as given: it changes how node angles are assembled from
branch differences, which pays once those are right (in distribution, large grid) and not otherwise. Hypothesis, not
tested: the branch differences need the global, topology-dependent solve (in DC, L⁻¹P), which 12 message-passing
layers with per-layer physics feedback approximate well only on the topologies they were trained on.

Suggested next test (not run): give the model θ_DC (computed from inputs, exact under S1/S3/S4/S5, so it stays on
the canonical section) and learn the AC correction. Prediction for fold 3: zero-shot VA on case118 below DC's
1.95 deg for every seed; if not, the learned correction does not transfer and the gain is DC's alone.

## M3 on fold 3 (2026-10-05)

Three runs (`m3canon`, seeds 0-2, array 61962) on abacus-007 with 6 CPUs each, HEAD 30a2148 (the pre-registration of
P23-P25 is in that commit, before the runs); every run reached the 100-epoch cap. Compared with fold 3's canon runs.

| id | verdict | what the numbers say |
|---|---|---|
| P23 case118 held out, VA | FAIL | m3 [9.28, 12.96] vs canon [10.06, 12.41] deg (means 10.6 vs 11.5). Seeds 0-1 (9.45, 9.28) were below canon's range; seed 2 (12.96, offset 10.0 deg) is above it |
| P24 vs DC | PASS (as predicted) | every seed 5-7× DC's 1.95 deg |
| P25 (exploratory) | — | zero-shot case118: VM [0.0024, 0.0043] vs [0.0054, 0.0069] p.u., PBE [9.5, 13.2] vs [13.7, 20.2] MVA, QG [16, 21] vs [22, 73] Mvar, angle rest [7.6, 8.2] vs [8.5, 9.5] deg: all separated in M3's favour; branch differences [1.79, 2.02] vs [1.78, 2.25] deg, offset and PG overlap. In distribution: PBE lower on all three grids (separated), VM on case30 and case57, VA on case14 by a hair ([0.078, 0.274] vs [0.276, 0.395]); the rest overlap. Canon's fold-3 excess on case30 VA is there in M3 too (0.43-0.65 deg): not specific to canon's frame |

Reading: the per-bus frame does not move the angles (the branch differences are as wrong as canon's, and the offset is
seed-dependent), but it lowers the voltage-magnitude and power-balance errors, zero-shot and in distribution. These
channels were exploratory here, on one fold. Two explanations are confounded by construction (pre-registered side
effect): the local inputs themselves (q_i = Q_i/D_i is the natural variable of a local voltage deviation) or the
per-bus normalization of the residual that GNS feeds back into every layer and puts in its physics loss.
Suggested next: (a) pre-register "M3 below canon on zero-shot VM and PBE" on folds 1-2 (held out case30, case57);
(b) an ablation with canon's inputs and the per-bus residual normalization, to separate the two explanations.
