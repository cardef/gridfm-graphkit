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

## Residual-normalization ablation on fold 3 (P28, 2026-10-05)

Three runs (`canonresnorm`, seeds 0-2, array 61992) on abacus-007 with 6 CPUs each, HEAD 9c74f7c (the pre-registration
of P26-P28 is in that commit, before the runs); every run reached the 100-epoch cap. Compared with fold 3's canon and
m3canon runs.

| id | verdict | what the numbers say |
|---|---|---|
| P28, VM (decisive) | FAIL: unresolved at 3 seeds | resnorm [0.0034, 0.0066] p.u. overlaps canon [0.0054, 0.0069] and M3 [0.0024, 0.0043] (seeds 0.0034, 0.0039, 0.0066) |
| P28, PBE (alongside) | (i) residual normalization | resnorm [8.6, 10.7] MVA: below canon [13.7, 20.2], overlapping M3 [9.5, 13.2] |

Exploratory, same runs. Zero-shot: angle rest [7.2, 7.9] deg below canon [8.5, 9.5], overlapping M3; QG [23, 36] Mvar
overlapping canon and above M3 [16, 21]; VA, offset and branch differences overlap everything. In distribution: VM and
PBE below canon on case30 and case57 and overlapping M3 there; on case14 everything overlaps.

Reading: dividing the residual GNS feeds back into every layer (and puts in its physics loss) by the local admittance
scale reproduces most of what M3 gained: the power balance zero-shot, VM and PBE in distribution on case30/57, the
non-offset part of the angle error. M3's local inputs add the zero-shot QG gain; the zero-shot VM gain is split or
unresolved at 3 seeds. The ablation, like M3, changes the feedback and the weighting of the physics loss together,
so which of the two matters is not separated. It is the test Piano A §9 suggested for GNS's physics feedback
("residuo normalizzato in ingresso a physics_mlp"), here on transfer instead of augmentation.

## M3 on folds 1-2 (P26-P27, 2026-10-06)

Six runs (`m3canon`, folds 1-2 × seeds 0-2, arrays 61990/61991) on abacus-004 with 5 CPUs each, HEAD 9c74f7c; every
run reached the 100-epoch cap. Compared with those folds' canon runs.

| id | verdict | what the numbers say |
|---|---|---|
| P26 zero-shot VM | FAIL | case30 m3 [0.0093, 0.0131] vs canon [0.0086, 0.0132]; case57 [0.0161, 0.0306] vs [0.0149, 0.0216], one seed far above canon |
| P27 zero-shot PBE | PASS | case30 [3.29, 4.97] vs [5.39, 5.82] MVA; case57 [4.91, 7.61] vs [8.27, 12.63] |

Exploratory, same runs: the angle error that is not a common offset (VA rest) is below canon on both grids (case30
[1.43, 1.92] vs [2.32, 2.81] deg, case57 [1.90, 2.45] vs [3.36, 4.13]), as on case118; branch differences below on
case57 ([1.07, 1.76] vs [1.98, 2.15]); total VA, offset, PG and QG overlap. In distribution PBE is below canon on 4 of
6 (fold, grid) pairs, VM on 2.

## Reading (M3, all folds)

On the three held-out grids M3 lowers the zero-shot power-balance error (P27, pre-registered on case30 and case57;
the same on case118) and the part of the angle error that is not a common offset (exploratory, all three grids). It
lowers neither the total angle error, dominated by the per-graph offset relative to REF and seed-dependent, nor VM
beyond case118 (P26). On case118 the per-bus residual normalization alone reproduces the PBE and angle-rest gains
(P28, alongside): what transfers is mostly how GNS's physics feedback is normalized, not the local inputs.

Suggested next (not run): (a) the residual normalization inside M1, which removes the offset in distribution (P18)
while the normalization lowers the rest: the two act on the two parts of the angle error; (b) whether the normalized
feedback lets the wide augmentation train (Piano A §9 hypothesis).

## DC prior (P29-P33, 2026-10-08)

Nine runs (`dccanon`, 3 folds × seeds 0-2; arrays 62474/62475 on abacus-004 with 5 CPUs per task, 62538 on
abacus-007 with 4 instead of the registered 6), code as registered (HEAD 665c298 for folds 1-2, 37eb1ae for fold 3;
the commits in between touched only the pre-registration and the report); every run reached the 100-epoch cap.
Compared with DC (`BASELINES.txt`) and with each fold's canon runs.

| id | verdict | what the numbers say |
|---|---|---|
| P29 | FAIL | case118 [1.90, 2.00] deg vs DC 1.95: DC level |
| P30 | FAIL | case30 [1.98, 2.35] vs DC 1.56; case57 [1.45, 1.81] vs DC 1.08: worse than DC |
| P31 | PASS (gate) | far below canon on all three grids |
| P29m, P30m | FAIL | no seed within 0.75 × DC on any grid |
| P33 | FAIL | case118 offset [1.46, 1.61] vs DC's 1.6 (two seeds below, one at 1.61): DC level |

Exploratory (P32), with the decomposition the amendment fixed in advance (DC's own value in brackets, deg):

| held out | offset | rest | branch differences |
|---|---|---|---|
| case30 | 1.93-2.31 (1.51) | 0.33-0.43 (0.40) | 0.34-0.43 (0.34) |
| case57 | 1.37-1.72 (0.39) | 0.48-0.57 (1.01) | 0.37-0.39 (0.53) |
| case118 | 1.46-1.61 (1.6) | 1.10-1.37 (1.11) | 0.39-0.48 (0.48) |

The local part of the learned correction transfers: rest and branch differences at or below DC's on every grid,
halved on case57. The global part does not: the correction adds a common angle shift learned on the training grids,
which on case30 and case57 is far worse than DC's own offset. PBE is far below canon's on all three grids (case118
7.4-7.6 vs 13.7-20.2 MVA; PBE is blind to a common shift); VM stays at canon's level (the prior touches only the
angles). In distribution the DC prior is the best arm on angles (case118 0.59-0.65 deg in folds 1-2, against canon
1.23-2.09 and M1 0.69-0.99). Training time is not comparable with canon's (different CPU counts per task).

Reading, as the amendment fixed it for a P33 failure: the model cannot fix the loss-induced offset even when given
the DC solution. What is missing is a global quantity, how the losses are carried through the network to the slack,
which local message passing cannot infer on an unseen grid. By the pre-registered decision rule there is no separate
DC-prior paper. Suggested next: compute that global quantity physically (DC with loss compensation, one Newton or
fast-decoupled step), and as its learned counterpart a model whose learned part is local (injection corrections) and
whose global part is a second exact solve.

## Transfer step 1: classical references (C1-C4, 2026-10-08)

`r2_classical.py` (job 62783, code as committed in 1284efc), output in `CLASSICAL.txt`. First run (job 62780)
failed the gate because the PF mask hides the REF voltage; see the amendment in `PREREGISTRATION.md`.

| id | verdict | what the numbers say |
|---|---|---|
| C1 gate | PASS | Newton to convergence with the labelled REF voltage matches the labels: VA ≤ 6e-6 deg, VM ≤ 2e-7 p.u. on all four grids |
| C2 | FAIL (seen before the amendment, disclosed) | DC with loss compensation: VA 1.87 / 1.94 / 1.56 / 1.48 deg vs DC 1.51 / 1.56 / 1.08 / 1.95 (case14 / 30 / 57 / 118); offset higher on three grids |
| C3 | FAIL | one Newton step from DC (oracle REF voltage): VA 0.20 / 0.13 / 0.22 / 2.53 deg, VM 0.002 / 0.003 / 0.012 / 0.002; above DC with loss compensation on case118 and above 0.005 p.u. VM on case57; with the inputs-only REF voltage also above 0.25 deg on case30 |
| C4 | holds on case14 only | Newton from DC: 3.2 iterations on case14, 4.0 elsewhere; from a flat start 4.0 everywhere |

What the references say:
- Physics solves these grids exactly in at most four Newton iterations once the REF voltage is known; DC saves at
  most one iteration. A learned model has a practical role here only as a faster approximation, not as a better one.
- One Newton step from DC is not a uniformly strong reference: on case118 it overshoots (2.5 deg, worse than DC).
- With the same information as the models (REF voltage estimated from the PV setpoints), exact physics still errs
  by 0.43 / 0.56 / 0.07 / 0.23 deg and 0.006 / 0.005 / 0.0008 / 0.002 p.u.: that is the floor the hidden REF voltage
  puts under any method using these inputs (the setpoint estimate is off by 0.019 / 0.024 / 0.004 / 0.015 p.u.).
  The learned models are above it zero-shot: the best arm on angles (DC prior) by 3.5-20×, the best on VM (M3) by
  1.3-18×.
- DC's offset is not the series losses at |V| = 1 (C2).
