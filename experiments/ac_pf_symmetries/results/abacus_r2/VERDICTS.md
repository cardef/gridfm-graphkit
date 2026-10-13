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

## Transfer step 1b (C5-C7, 2026-10-08)

`r2_classical.py` at 47451d3 (job 62854), output in `CLASSICAL_1b.txt`. The P37s threshold (the 12-step P-theta
iteration: case30 1.003, case57 1.445, case118 2.997 deg) was set in `r2_report.py` at 19:16, when one dcinjcanon
result existed and had not been read.

| id | verdict | what the numbers say |
|---|---|---|
| C5 | FAIL | DC's signed offset is -1.46 / -1.51 / **+0.38** / +1.49 deg (case14 / 30 / 57 / 118): negative on case14 and case30 only. V-weighted DC lowers the offset to 0.85 (case14) and 1.10 (case30), not to <= 0.5; on case118 it raises it (1.94) |
| C6 | FAIL | the P-theta iteration with V held at the setpoints: VA 0.51 / 1.00 / 1.45 / 3.00 deg against DC's 1.51 / 1.56 / 1.08 / 1.95: better than DC on case14 and case30, worse on case57 and case118 |
| C7 | PASS | REF voltage on the PV setpoints and the total loads, in sample: R^2 0.991 / 0.947 / 0.973 on case14 / 57 / 118 (RMSE 0.0002 / 0.0013 / 0.0025 p.u.; constant 1.06 on case30) |

Reading:
- The REF voltage is a near-deterministic function of the inputs within a grid (the OPF the generator runs),
  which is why the released models predict it in distribution; the "inputs" floor of step 1 (mean PV setpoint) is
  a crude estimator's, not the Bayes floor. Zero-shot, the function is grid-specific (it goes through costs and
  limits the PF task masks).
- DC's offset is not one global quantity. The voltage level explains part of it on case14 and case30 (V-weighting
  removes 42% and 27%), losses explain part of it on case118, and on case57 the two cancel. With the REF voltage
  known, the 12-step iteration still leaves 0.06-0.38 deg of offset on case14 and case30 because the PQ voltages
  are held at 1.0 there: the offset tracks the whole voltage profile, which only the full AC iteration (fast-decoupled
  or Newton) gets right (fast-decoupled from the inputs converges to the same 0.07-0.56 deg floor as Newton).
- P37s is therefore a weak test on case57 and case118, where the P-theta reference is worse than DC; P35's margin
  over DC is what binds there.

## Transfer step 2: zero-initialized DC prior and injection correction (P34-P38, 2026-10-08)

18 runs (dczcanon, dcinjcanon; 3 folds × seeds 0-2; arrays 62785-62790; code at 1284efc; folds 1-2 on abacus-004,
fold 3 on abacus-007, 4 CPUs per task); every run reached the 100-epoch cap. P34s and P37s are the amendment's
criteria (f4743ea, before any dcinjcanon result existed; P37s threshold set at 19:16, before one existing result was
read).

| id | verdict | what the numbers say (zero-shot VA, deg) |
|---|---|---|
| P34 | FAIL | offset: case30 [0.20, 1.35] vs DC 1.51 (ok); case57 [0.27, 0.68] vs 0.39; case118 [0.97, 4.77] vs 1.6 |
| P34s | FAIL | the same against 0.75 × DC's offset |
| P35 | FAIL | case30 [0.27, 1.45] vs 1.17; case57 [0.37, 0.73] vs 0.81 (ok); case118 [1.06, 5.32] vs 1.46 |
| P36 | FAIL | below dczcanon on case30 ([0.27, 1.45] vs [1.56, 2.06]) and case57 ([0.37, 0.73] vs [1.46, 1.54]), overlapping on case118 ([1.06, 5.32] vs [1.64, 2.32]) |
| P37, P37s | FAIL | on case118 (one seed at 5.32); met on case57 for both, on case30 for P37 only |
| P38 (exploratory) | — | dczcanon vs dccanon (zero init only): no systematic change (case30 [1.56, 2.06] vs [1.98, 2.35], case57 [1.46, 1.54] vs [1.45, 1.81], case118 [1.64, 2.32] vs [1.90, 2.00]) |

Per seed, dcinjcanon zero-shot (DC's value in brackets): case30 VA 0.35 / 0.27 / 1.45 (1.56), branch differences 0.12
/ 0.10 / 0.31 (0.34); case57 VA 0.37 / 0.73 / 0.67 (1.08), branch differences 0.15 / 0.23 / 0.23 (0.53); case118 VA
1.94 / 5.32 / 1.06 (1.95), branch differences 0.36 / 0.70 / 0.18 (0.48). In distribution on case118 (folds 1-2)
0.10-0.30 deg, the best arm so far (dccanon 0.59-0.65, canon 1.23-2.09).

Reading: the pre-registered transfer criterion fails, on case118 and on one case30 seed. The exploratory pattern is
the first positive transfer result of the campaign. On held-out grids of the training size, reading the learned
correction as an injection and computing its effect with the exact solve beats both DC (all seeds on case57, two
of three on case30, which on case30 also go below the inputs-only physics floor of 0.56 deg because the model infers
the REF voltage better than the mean PV setpoint does) and the same correction applied to the angles directly (P36
on two grids). On the larger held-out grid it is unstable across seeds (1.06 to 5.32 deg, the large errors in the
common offset), consistent with the review's concern: L^-1 amplifies the smooth part of c, and that amplification
grows with grid size beyond what training saw. Zero initialization alone changes nothing (P38).

## REF voltage given vs hidden, and the two extra arms (P39-P46, C8; 2026-10-09)

27 runs, every one at the 100-epoch cap. `canonvref` (3 folds × seeds 0-2; arrays 62812/62813 on abacus-004,
62814 on abacus-007; 4 CPUs per task; code 5447821 on the vref branch, merged into main as 6a72b06), `dcinjvref`
(62860-62862) and `dcinj1hop` (62863-62865), both launched 2026-10-08 ~19:20 with the code of 99d201d, which added
the arms. canon's VM on PQ buses comes from its re-evaluation (62815-62817, `result_eval.json`), which reproduces
`result.json` on every shared metric to 4e-5 relative: the gate of the P39-P41 block holds. dccanon's signed offsets
from its re-evaluation (62866-62868). Scored with `r2_report.py` at HEAD 9fc600a, run from the workstation against the
GPFS mount (Python 3.11, numpy 2.3; the script needs only numpy): a deviation from the cluster-side scoring of the
earlier blocks, with no effect on the numbers.

| id | verdict | what the numbers say (zero-shot, deg / p.u. / MVA) |
|---|---|---|
| P39 | FAIL, in the direction the registration did not anticipate | VA: case30 canonvref [7.69, 8.18] vs canon [3.89, 6.17], separated with canonvref ABOVE; case57 [7.45, 9.21] vs [7.44, 8.97] and case118 [9.90, 11.35] vs [10.06, 12.41] overlap. The registered failure reading (canonvref entirely below canon somewhere) does not apply. On case30 the whole increase is the common offset ([7.26, 7.75] vs [3.13, 5.49]); rest and branch differences overlap |
| P40 | FAIL | VM on PQ buses: case30 [0.0067, 0.0134] vs canon [0.0086, 0.0135]; case118 [0.0051, 0.0070] vs [0.0049, 0.0064]; case57 [0.0159, 0.0271] vs [0.0144, 0.0216]: overlap everywhere. Registered reading: the learned models' zero-shot VM error is not the voltage level |
| P41 (exploratory) | — | zero-shot QG below canon on case57 ([14.0, 21.9] vs [41.4, 95.0] Mvar) and case118 ([16.6, 20.2] vs [21.9, 73.5]); PBE below on case30 ([3.89, 4.61] vs [5.39, 5.82] MVA), overlapping elsewhere. In distribution canon's VM on all buses is 2-4× its VM on PQ buses on case14 and case118 (fold 3 case14 [0.0024, 0.0051] vs [0.0009, 0.0013]; fold 1 case118 [0.0012, 0.0018] vs [0.0006, 0.0009]): the REF bus carries most of canon's in-dist VM error there; on PQ buses canonvref and canon overlap on every (fold, grid). In-dist VA: canonvref below canon on case14 and case30 in fold 3 ([0.12, 0.24] vs [0.28, 0.40]; [0.45, 0.60] vs [0.64, 0.66]), overlapping elsewhere (case118 means higher, 1.9-2.5 vs 1.7-1.9). In-dist PBE above canon on case14 in fold 2 and case30 in fold 3 |
| P42 | FAIL | dcinjvref VA vs 0.75 × DC: case30 [0.23, 2.16] vs 1.17; case57 [0.53, 1.37] vs 0.81; case118 [1.66, 11.33] vs 1.46. Every grid has a seed above DC itself (1.56 / 1.08 / 1.95). Registered reading: even on the well-posed task the learned correction does not transfer |
| P43 | FAIL | dcinjvref offset vs dcinjcanon: case30 [0.19, 2.03] vs [0.20, 1.35]; case57 [0.37, 1.25] vs [0.27, 0.68]: not below. Registered reading: the shift is not the hidden setpoint |
| P44 (exploratory) | — | against the oracle references of `CLASSICAL_1b.txt` (REF voltage given to physics): Newton is exact; the 12-step P-θ iteration errs 0.54 / 1.41 / 2.87 deg (case30 / 57 / 118); dcinjvref per seed 0.23, 1.30, 2.16 / 0.53, 1.37, 0.80 / 11.3, 3.75, 1.66: at or below the iteration on case57 (all seeds) and case30 (one seed); one case118 seed diverged (VA 11.3 deg, VM 0.041, PBE 123 MVA, QG 870 Mvar; fine in distribution). VM on PQ buses overlaps canonvref's on all three grids. QG far below dcinjcanon on case30 ([2.2, 3.5] vs [7.1, 11.6] Mvar) and case57 ([5.2, 11.0] vs [33.7, 43.7]); branch differences above it on case57 ([0.24, 0.29] vs [0.15, 0.23]). In distribution VM below dcinjcanon on case14 and case118 (the REF bus is given), VA below on case14 in fold 3, overlapping elsewhere |
| P45 | FAIL | dcinj1hop rest / branch differences vs DC: case30 [0.44, 0.83] vs 0.40, [0.28, 0.52] vs 0.34; case57 [1.10, 2.72] vs 1.01, [0.45, 1.01] vs 0.53; case118 [1.17, 3.59] vs 1.11, [0.58, 1.75] vs 0.48: above DC on every grid. Registered reading: the local transfer seen with dccanon needed multi-hop context |
| P46 (exploratory) | — | dcinj1hop vs dcinjcanon: zero-shot VM and PBE separated above on all three grids (PBE [14.4, 22.6] vs [1.07, 2.38] MVA on case30, [57.7, 60.4] vs [2.95, 3.86] on case57, [48.2, 65.4] vs [3.7, 10.2] on case118), QG above on case57 and case118, VA above on case57 ([1.58, 4.82] vs [0.37, 0.73]) and overlapping on case30 and case118 where dcinjcanon's own spread is wide. In distribution above on VM, VA and PBE on nearly every (fold, grid), PBE by 5-20×. Training 1,000-1,400 s against 7,000-17,000 s |
| C8 (exploratory) | — | dccanon's signed zero-shot offset: case30 −2.06 / −1.93 / −2.31 deg against DC's −1.51; case57 +1.50 / +1.72 / +1.37 against +0.38; case118 +1.37 / +1.32 / +1.51 against +1.49. Same sign as DC's on every grid: the learned correction does not carry a constant of the training grids; it pushes the offset further in DC's own direction on case30 (by 0.4-0.8 deg) and case57 (by 1.0-1.3), and leaves it at DC's on case118 |

Reading: the REF voltage is not what the learned models lack. Given as an input it leaves the zero-shot angle error
where it was on case57 and case118 and makes it worse on case30 (P39, opposite to the registered direction; the whole
increase is the common offset), and lowers the voltage error on the PQ buses nowhere (P40). The same holds for the
injection-correction arm (P42, P43): the shift that does not transfer is not the hidden setpoint, and C8 says it is
not a constant of the training grids either (same sign as DC's offset on every grid). What the setpoint buys is in
distribution and in the task definition: the REF bus carried most of canon's in-dist VM error on case14 and case118,
and with the setpoint given the task is well posed, so the floor that physics puts under any method from these inputs
(0.07-0.56 deg) goes to zero. The local part of the learned correction needs the multi-hop context: with one layer and
no physics feedback, rest and branch differences are worse than DC's everywhere (P45) and every other channel is
worse than dcinjcanon's (P46). Protocol consequence, carried into Piano T: give the REF voltage to every method as an
input (an OPF setpoint, like the PV ones), compare VM on PQ buses, keep the multi-layer corrector. By the decision rule
of the pre-registration there is no claim about the hidden setpoint; the zero-shot problem stays the global shift, now
shown to be neither the voltage level nor a constant.

## Size extrapolation of the trained arms (P47-P50, 2026-10-09 evening)

Every saved model of the three folds (84 per grid; canonresnorm exists in fold 3 only), evaluated zero-shot and
unchanged on case500_goc and case2000_goc: arrays 62987 and 62988 (60 tasks, 4 CPUs each, code 60c4da6, `multigrid.py
--eval-only --extrap`, `result_extrap_<grid>.json` in the registered run dirs); the fixed 1024 scenarios of a
2048-scenario random subset (seed 0) drawn from 20 of the ~1000 partitions of the public `pf_small` sets (data job
62985). References: `BASELINES_extrap.txt` (flat, DC; computed before any evaluation) and `CLASSICAL_extrap.txt`
(case500 from the dense `r2_classical.py` at 60c4da6; case2000 from its sparse rewrite, job 63051, identical to the
dense output on case14 and case500 at every printed digit; the dense run on case2000 was cancelled after 4 h 27 min).
Deviation: the
arrays were released from their dependency on the data job once the processed caches and the DC references existed,
so the classical floors of case500 were computed concurrently with the evaluations, not before; the pre-registered
criteria use only DC. Scored with `r2_extrap.py` (`REPORT_extrap.md`) from the workstation, as the previous block.

| grid | flat VA | DC VA | DC offset | DC rest | DC θ_f − θ_t | VM flat | physics from the inputs only |
|---|---|---|---|---|---|---|---|
| case500_goc | 10.1 | 2.51 | 2.18 | 1.24 | 0.274 | 0.0648 | fast-decoupled and Newton 0.070 deg, VM 0.0009; Newton converges in 5 iterations from flat or DC; DC + loss compensation 1.73; one Newton step from DC 26.8 (overshoot) |
| case2000_goc | 16.9 | 9.75 | 9.08 | 3.54 | 0.291 | 0.0628 | fast-decoupled from the inputs 0.33 deg (0.33 on the 1010 scenarios where the oracle Newton converges), VM on PQ buses 0.0013; DC + loss compensation 2.91; the oracle Newton reproduces the labels on 1010 of 1024 scenarios (max 1.1e-5 deg) and diverges on 14 from the DC start and from flat (plain Newton without step control; no islands, no PV voltage off its setpoint, labels' mismatch in our Ybus 1.5e-4 p.u. everywhere), so its aggregate rows in `CLASSICAL_extrap.txt` are contaminated by those 14 (`gate_extrap.py`, job 63052) |

| id | verdict | what the numbers say (zero-shot VA in deg unless stated) |
|---|---|---|
| P47 | FAIL, below the band in one fold | dccanon / DC on case500: fold 1 (train 14/57/118) [0.75, 0.92], fold 2 [0.90, 0.98], fold 3 [0.99, 1.02]; dczcanon [0.73, 0.81], [0.84, 0.93], [0.98, 1.10]. The registered reading for "below 0.9" (a gain that grows with size) holds only in the fold whose training set holds the 118, and is 8-25%. On case2000 both arms are DC: [0.94, 1.08] x, with DC's own offset (8.5-9.9 against 9.08) and rest (3.4-3.6 against 3.54) |
| P48 | FAIL by the spread clause; the substantive prediction holds everywhere | dcinjcanon on case500: fold 1 [6.96, 45.6] (spread 6.5x), fold 2 [37.2, 53.6] (1.4x), fold 3 [27.8, 128] (4.6x): every seed above DC's 2.51 in every fold, 3-50x; fold 2's spread is below 2 because its three seeds fail alike. Neither registered failure reading applies (no fold below DC; the spread did not shrink through stability). On case2000: 20-136 (8-12x DC). dcinjvref 12-109 (500), 51-130 (2000); dcinj1hop 11-118 (500), 59-112 (2000) |
| P49 | FAIL | canon, m1fullcanon, m3canon, canonvref on case500: 5.5-15.3 deg, mean ratios to DC 2.4-3.9x, per seed 2.2-6.1x (one m1fullcanon seed in fold 2 above 4x); offset²/VA² 0.42-0.78, the larger part in 10 of 12 cells but never dominant; rest 3.6-5.7 against DC's 1.24, branch differences 0.74-2.29 against DC's 0.27: the local part is 3-8x worse than DC's. Registered reading: the canon-like arms lose less with size than the DC-based ones gain because DC itself worsens (1.95 -> 2.51 -> 9.75 deg), and the local part degrades too. On case2000 1.2-2.2x DC (11.3-21.5 deg; rest 8.3-9.9 against DC's 3.54; branch differences 0.95-2.7 against 0.29) |
| P50 (exploratory) | — | VM on case500: every arm 0.36-0.83 of flat and 26-60x the physics floor (0.0009); m3canon has the lowest means but overlaps canon's range in every fold; on case2000 0.5-1.6 of flat, dcinjcanon above flat. PBE on case500: m3canon and canonresnorm 27-43 MVA, dccanon 30-66, canon 59-149, dcinj1hop 260-515; on case2000 m3canon 41-64, dccanon 47-98, canon 84-221. Cost of extrapolation (zero-shot VA on case500 / in-distribution VA on the fold's largest training grid, means): canon 4-28x, m1fullcanon 11-30x, m3canon 5-29x, dccanon 3-21x, dcinjcanon 130-1100x. The canon-like arms' absolute angle error is not monotone in size: 6-8 deg on case500 against 10-12 on case118 held out and 5-8 on case30/57 held out; it is DC that degrades with size. The DC prior's gain over DC: 8-25% at 500 in the fold that trained on the 118, none at 2000 |

Reading: with size, what transfers is the physics operator and nothing learned. The DC-anchored arms are DC, plus a
small gain at 4x the training size when the largest training grid was the 118 and none at 17x. Every arm without the
anchor is 1.2-3.9x worse than DC in angles, with a local part 3-9x worse than DC's branch differences and a voltage
channel at 0.4-0.9 of a flat profile. The injection arms, the closest realization of "local corrections behind an
exact global solve", collapse: L⁻¹ amplifies the smooth part of a local error, more on larger grids. And the DC
operator is itself the wrong global operator with size: its offset against the slack grows from 1.6 deg (118) to 9.1
(2000) while its branch differences stay at 0.3, and the fast-decoupled iteration from the same inputs reaches 0.07 deg
on case500 and 0.33 on case2000 (30x below DC and below every learned arm there). The transferable global operator is
the AC iteration; plain Newton needs step control on the largest grid (14 of 1024 scenarios diverge from the DC start). Carried into Piano T: its strong form (beat DC
zero-shot with a DC anchor, learned local corrections and exact symmetries) is contradicted at PoC scale (three
training grids of at most 118 buses, 1024 scenarios each, a 20M-parameter GNS); the one experiment that can revive it
is T1w trained on the 100 public pretraining subgrids and evaluated zero-shot on 118/500/2000, pre-registered with the
expectation "DC level". Failing that, the question becomes what a learned component adds to an AC iteration.

## Piano K: topology transfer on case118 (K-P1-K-P5, 2026-10-10, scored 2026-10-13)

One grid, the public N-1 set of case118 (`gridfm/pf_small_case118_ieee`): training on the official seed-0 split
(5,000 train, 500 val, the first 2,000 official test ids as the in-distribution set; `data/case118_k`), zero-shot on
six N-k sets we generated with gridfm-datakit from the public recipe with all outage mass on exactly k outages
(`cluster/nk/case118_k<k>.yaml`, 2,048 scenarios each minus the few that did not converge: 2048/2048/2047/2047/2044/
2021 for k = 1/2/3/5/10/20; islanded topologies are rejected by datakit; the fixed 1024 of each are evaluated).
Arms: plain (the backbone with the public config, no canonicalization, our recipe), dccanon, dcinjcanon, gp (+vref
+canon +resnorm +dc), 3 seeds each, 100 epochs, 4-6 CPUs, 7-15.5 h per run (jobs 63095, eval 63096, code 393e669);
the released checkpoints base/small/tiny seed 0 (trained on the full public N-1 set, ~175k scenarios) on the same
sets (job 63140, `genco_released/`); references `BASELINES_nk.txt`, `CLASSICAL_nk.txt` (damped Newton), PH =
linear regressor of the REF voltage on the PV setpoints and the total loads fitted on the 5,000 training scenarios
(`vref_coef.json`, in-sample R^2 0.40, RMSE 0.0137 p.u.) + 12 fast-decoupled steps (`CLASSICAL_nk_ph.txt`).
Tables: `results/pianok/REPORT_k.md` (`r2_k.py`). Gate first.

| id | verdict | what the numbers say (zero-shot VA in deg) |
|---|---|---|
| K-P5 | PASS | damped Newton with the labelled REF voltage reproduces the labels at every k (VA 2-5e-6 deg, max 5-9 iterations); on our k = 1 set the released checkpoints err within 0.6-3.2% of their official-test error (base 0.482 vs 0.479, small 0.516 vs 0.533, tiny 0.635 vs 0.628): the generated sets are comparable |
| K-P1 | FAIL | the degradation with k is not global: from k = 1 to k = 10 the offset grows x1.7-2.0 and the branch differences x1.6-1.8 for plain; x5.1 and x5.1 for GENCO base (small x4.3 / x4.3, tiny x4.3 / x3.9). Every component of the error grows alike. Registered reading: the learned local physics does not survive deep contingencies either |
| K-P2 | FAIL | dccanon's VA doubles from k = 1 to k = 10 (x1.94-2.04) and its VA/DC ratio rises from 0.41-0.44 to 0.60-0.61 (x1.40-1.47): the anchored arm degrades faster than DC. plain's VA/DC rises too (x1.25-1.41): at this budget plain is above DC already at k = 1 (1.2-1.7 x). Registered reading: the anchor does not protect against deep contingencies; the learned correction degrades like the backbone |
| K-P3 | PASS | dccanon / DC: 0.41-0.44 (k=1), 0.44-0.46 (2), 0.39-0.42 (3), 0.47-0.49 (5), 0.60-0.61 (10), 0.70 (20): a floor at every k, every seed, and well below DC |
| K-P4 | FAIL | PH: VA 0.236 / 0.294 / 0.246 / 0.252 / 0.387 / 0.298 deg for k = 1/2/3/5/10/20 (above 0.3 at k = 10), VM on PQ buses 0.00046-0.00059 (ok); below every learned arm only for k >= 5 and k = 2: dcinjcanon is below PH at k = 1 ([0.197, 0.223]) and k = 3 ([0.180, 0.191]). PH's error is almost all offset (0.23 of 0.24): the REF-voltage regressor is weak on N-1 data (R^2 0.40 against 0.95-0.99 on the base-topology data of R2). Registered reading: the REF voltage depends on the topology more than the in-distribution R^2 suggested |

Exploratory (not pre-registered), the findings that matter:

| arm | in-dist (official test) | k = 1 | k = 5 | k = 10 | k = 20 | k=20 / k=1 | k=20 / DC |
|---|---|---|---|---|---|---|---|
| GENCO base, released (175k scenarios) | 0.479 | 0.482 | 1.87 | 2.50 | 4.68 | x9.7 | 1.10 |
| plain (5,000 scenarios) | [2.44, 3.85] | [2.84, 4.10] | [4.02, 5.56] | [5.48, 7.13] | [8.24, 10.2] | x2.5 | 1.9-2.4 |
| dccanon | [0.81, 0.90] | [0.97, 1.03] | [1.26, 1.31] | [1.97, 2.00] | [2.95, 2.99] | x3.0 | 0.70 |
| gp | [0.69, 0.79] | [0.88, 0.96] | [1.17, 1.26] | [1.88, 1.98] | [2.83, 2.99] | x3.2 | 0.67-0.70 |
| dcinjcanon | [0.155, 0.167] | [0.197, 0.223] | [0.345, 0.369] | [0.444, 0.514] | [0.616, 0.680] | x3.1 | 0.145-0.16 |
| DC | — | 2.36 | 2.68 | 3.27 | 4.25 | x1.8 | 1 |
| fast-decoupled from the inputs | — | 0.316 | 0.326 | 0.445 | 0.408 | x1.3 | 0.10 |
| PH | — | 0.236 | 0.252 | 0.387 | 0.298 | x1.3 | 0.07 |

- In distribution, dcinjcanon trained on 5,000 scenarios errs 0.155-0.167 deg on the official test against 0.479 for
  the released GENCO base trained on ~175k: 3x lower with 35x less data; dccanon and gp (0.7-0.9) are within 2x of it,
  plain (2.4-3.9) is 5-8x above it. The DC anchor is worth more than the data budget here.
- Every learned arm degrades with k; the full-budget released models degrade fastest (x8-10 from k = 1 to 20, reaching
  DC level at k = 20 in angles while their residual stays 0.1-0.7 MVA: the residual is blind to the shift), our
  low-budget arms x2.5-3.2, the physics references x1.3-1.8.
- dcinjcanon is the only learned arm that stays near the physics floor at every k: 0.62-0.68 deg at k = 20 against
  0.41 for fast-decoupled from the inputs and 0.30 for PH, 7x below GENCO base and 6-7x below DC; its branch
  differences are 5-13x below DC's at every k. The same arm collapses when the grid changes (Piano T, P48): it is a
  same-grid method. Its k = 2 error (0.40-0.42) is above its k = 3 error (0.18-0.19) for every seed: a property of
  that set's scenarios, not noise.
- VM on PQ buses: gp lowest among the learned arms (0.0003 at k = 1, 0.0022 at k = 20; the REF voltage given), GENCO
  base 0.00026 -> 0.0028, dccanon 0.0009 -> 0.0035; PH flat at 0.0005-0.0006.

Reading: on the same grid, deep contingencies break the learned local physics as much as the learned global part
(K-P1), and an exact per-topology solve inside the model does not make the learned part k-invariant (K-P2); what it
does is bound the error by DC's (K-P3) and, read as an injection correction, keep it 6-7x below DC up to N-20 while
cutting the in-distribution error 3x below the full-budget public model with 35x less data. That is the concrete
improvement to GENCO this campaign has found, with its scope: one grid, its own N-k distribution, no transfer to other
grids. The hybrid PH shows what a learned component competes against on topologies: a 0.3 deg floor that is entirely
the REF-voltage guess, and that a 0.40-R^2 regressor already reaches; dcinjcanon is below it only at k <= 3. Next,
not run: dcinjcanon with the REF voltage given (its VM is 2x gp's), its stability margin against k and against the
training budget, and the released architecture retrained with the injection decoder at full budget.

## Piano X, X0: the physics frontier on new grids (X0-P1-X0-P4, 2026-10-13)

`frontier.py` at ab78fab (srun job 63189), the fixed 1024 zero-shot scenarios of each grid, DC start, REF voltage given
and from the inputs; `results/pianox/FRONTIER.txt` and `frontier_<case>.json`. Costs are medians per scenario with
the same sparse code on one thread; the backbone's forward time uses 4 threads on the same CPU.

| id | verdict | what the numbers say (VA in deg, case14 / 30 / 57 / 118 / 500 / 2000) |
|---|---|---|
| X0-P1 | FAIL | fast-decoupled with the REF voltage given, after 2 iterations: 0.155 / 0.104 / 0.439 / 0.305 / **0.529** / **1.61**: above 0.5 on the two GOC grids, where it is 94-97% common offset (0.50 of 0.53, 1.56 of 1.61); after 5: 0.0148 / 0.0087 / **0.069** / 0.00052 / 0.00054 / 0.0061: above 0.05 on case57, where the iteration converges slowly and does not settle (0.038 at 8 iterations, 0.061 at 12). Below 0.1 deg after 3 / 3 / 5 / 3 / 3 / 4 iterations. Registered reading: the room for acceleration is the offset mode at 2-3 iterations on the large grids, and case57 |
| X0-P2 | PASS | case2000: DC 3.92 ms (factor 3.58, solve 0.34), B'' factor 3.34, one fast-decoupled iteration 0.63: DC plus 2 iterations 2.17 x DC, plus 5 iterations 2.66 x. On every grid, 5 iterations cost 2.4-2.7 x DC |
| X0-P3 | FAIL on case57 only | REF voltage from the inputs, after 5 vs 12 iterations: 0.429 / 0.432, 0.565 / 0.562, **0.115 / 0.088**, 0.235 / 0.235, 0.0709 / 0.0704, 0.329 / 0.332: within 1% except case57 (+30%), where the iteration itself is slow; elsewhere the inputs-only error is the hidden setpoint |
| X0-P4 (exploratory) | — | damped Newton from the DC start, REF voltage given: below 0.01 deg after 2 iterations on case14/30/57, 3 on case118, 4 on case500 after a first step that overshoots to 26.8 deg, not within 6 on case2000 (88, 35, 5.2, 0.43, 0.14, 0.15 deg; one scenario unconverged); one Newton iteration costs 2.2-12.8 ms against 0.04-0.63 for fast-decoupled. GENCO backbone forward (12 layers, random weights): case2000 hidden 12 / 24 / 48 = 115 / 194 / 490 ms per graph at batch 1, 74 / 163 / 417 at batch 32; case118 46 / 54 / 75 and 5.6 / 11.7 / 29.5. At 2000 buses one forward of the smallest backbone costs 7x the converged fast-decoupled (DC + B'' + 5 iterations = 10.4 ms) and 19x DC |

Reading: on every grid, with the REF voltage given, 4-5 fast-decoupled iterations from the DC start reach 0.0005-0.07
deg at 2.4-2.7x the DC cost, against 1-10 deg for every learned model evaluated zero-shot on new grids so far. The two
factorizations are two thirds of that cost on case2000 (6.9 of 10.4 ms): saving iterations can save at most the other
third, and a learned accelerator pays off only if it costs less than one iteration (0.63 ms there). On this CPU even
the smallest backbone's forward costs more than the converged physics; GPU not measured (the paper's speed-ups are GPU
against PowerModels' AC-PF). The slowest mode of the iteration is the common offset relative to the slack, the same
global quantity the learned models fail to infer: before any learning, a classical acceleration is the baseline (X0b).

## Piano X, X0b: Anderson-accelerated fast-decoupled (X0b-P1-X0b-P3, 2026-10-13)

`frontier.py` at 7ff3319 (srun job 63194), same scenarios, starts and REF-voltage modes as X0;
`results/pianox/FRONTIER_anderson.txt`, `frontier_anderson_<case>.json`. The plain fast-decoupled rerun in the same job
reproduces X0 at every printed digit.

| id | verdict | what the numbers say (VA in deg, case14 / 30 / 57 / 118 / 500 / 2000) |
|---|---|---|
| X0b-P1 | FAIL | Anderson with memory 3, REF voltage given, after 3 iterations: 0.0098 / 0.0101 / 0.057 / 0.074 / **0.123** / **0.241**, against plain 0.055 / 0.041 / 0.213 / 0.022 / 0.044 / 0.148: below 0.1 on the four IEEE grids, above it on case500 and case2000, where Anderson is worse than plain at iterations 2-4. Below 0.1 deg after 2 / 3 / 3 / 3 / 4 / 4 iterations, against plain's 3 / 3 / 5 / 3 / 3 / 4. Registered reading: on the large grids the early offset mode is not taken by classical acceleration; a learned accelerator would have room there |
| X0b-P2 | PASS | case57 after 12 iterations 6.0e-6 deg, against plain 0.061 and rising: the slow mode is accelerated away |
| X0b-P3 (exploratory) | — | memory 2 and 3 alike. Anderson reaches the exact solution on every grid within 7-10 iterations (VA at most 6e-6 deg; no scenario with mismatch above 1e-4 after 12, where plain leaves 981-1024 of 1024 above it on case14, case30, case57 and case2000). Its overhead in this Python code is 0.7-0.8x a plain iteration on case2000 (1.07-1.13 ms against 0.63). The inputs-only runs reach the same floors |

Reading: on every new grid tested the physics iteration is itself the zero-shot solver: 0.1 deg in 2-5 iterations and
the exact solution in 7-10 with Anderson, at 2.4-2.9x the DC cost. The margin left to a learned accelerator is bounded
and small. On case2000 with the REF voltage given, reaching 0.1 deg takes 4 plain iterations, 9.8 ms or 2.5x DC; a
perfect accelerator reaching it in 2 would save 1.26 ms, 13% of the cost of a new topology, because the two
factorizations take 6.9 ms; only when many scenarios share a topology and the factorizations are amortized could the
saving approach 2x. It must cost less than 0.63 ms per iteration it replaces; the smallest GENCO backbone costs 74 ms per
graph on this CPU.

## Piano X, X3: GENCO with the injection decoder on the 31 public subgrids (X3-P1-X3-P6, 2026-10-13)

GPU jobs 63215-63218 (A40, abacus-002/003, 2 runs per job) and 63219 (H100, abacus-005: dcinjvref seed 2), code
17bab77; the end-of-training `--eval-only --extrap` steps of all runs but dcinjvref seed 2 ran with dd16233-e8af9ae,
whose changes leave evaluation unchanged (`--inj-gain` defaults to 1). 31 subgrids, 1,200 scenarios each, up to 100
epochs, patience 40. Scored with `r2_pt.py` (`results/pianox_pt/REPORT_pt.md`). Zero-shot VA in deg, source
normalizer (the canonicalized arms are invariant to it; plain is scored with both).

| id | verdict | what the numbers say |
|---|---|---|
| X3-P1 | PASS | dcinjcanon below DC on both case118 sets, every seed: case118_ieee [1.57, 1.89] vs 1.95; case118_tr [1.87, 2.20] vs 2.30 (0.80-0.97x DC) |
| X3-P2 | FAIL | case500_goc [1.60, 3.21] vs 2.52: seeds 1.60, 1.91, 3.21 |
| X3-P3 | PASS | plain above DC on case118_ieee (11.0-20.4 source, 11.4-26.7 refit) and case500_goc (8.0-9.4, 7.8-9.7), every seed, either normalizer |
| X3-P4 | PASS | case2000_goc [3.09, 15.8] vs 9.75: seeds 3.09, 3.68, 15.8 |
| X3-P5 | PASS | no learned arm below fast-decoupled from the same inputs after 5 iterations: on 118 / 500 / 2000 the best learned arm is 1.49 / 1.19 / 2.25 against 0.235 / 0.071 / 0.329 (inputs) and 0.0005 / 0.0005 / 0.006 (REF voltage given) |
| X3-P6 (exploratory) | — | below |

Per seed (case118 / case118_tr / case500 / case2000; best validation loss and its epoch):

| arm | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| plain | 14.1 / 16.2 / 9.4 / 16.5 (0.182, ep 11) | 20.4 / 22.0 / 9.0 / 15.9 (0.146, ep 96) | 11.0 / 13.5 / 8.0 / 15.1 (0.144, ep 97) |
| dcinjcanon | 1.89 / 2.20 / 3.21 / 3.09 (0.150, ep 0) | 1.57 / 1.87 / 1.60 / 3.68 (0.115, ep 5) | 1.86 / 2.15 / 1.91 / 15.8 (0.094, ep 97) |
| dcinjvref | 1.49 / 1.81 / 1.19 / 6.72 (0.095, ep 87) | 1.69 / 2.01 / 3.06 / 2.25 (0.119, ep 22) | 1.67 / 1.98 / 3.10 / 3.28 (0.126, ep 3) |
| DC | 1.95 / 2.30 / 2.52 / 9.75 | | |
| GENCO public pretrained | 11.3 / 13.3 / 6.3 / 13.4 | | |

Exploratory (X3-P6): on case57, case30 and case14, below the training sizes, both injection arms sit at DC
(0.98-1.01x); plain at 2.7-12x DC. dcinjvref against dcinjcanon: overlapping ranges on every grid. In distribution the
injection arms reach validation losses of 0.094-0.15 against plain's 0.144-0.182. Training is unstable: every injection
run has epochs above 0.5 after the fifth (X3a-P0's criterion; dcinjcanon peaks 3.3, 12.8, 1.4), plain none (at most
0.40); diagnosis in the X3a block of PREREGISTRATION.md (`diag_x3.py`). Time per epoch: plain about 4.5 min, injection
arms about 9 min on an A40, 5.4 min on the H100. Hardware split: eight runs on A40, dcinjvref seed 2 on H100.

Reading: the decisive prediction holds. Trained on 31 topologies, the injection decoder brings GENCO below DC on the
benchmark where the paper reports its failure, every seed, while GENCO trained on the same data stays 5.6-14x above DC on case118,
as the public pretrained model does: topology diversity alone does not make GENCO transfer, the architecture does. The
margin on case118 is modest (0.80-0.97x DC) and far from physics from the same inputs (0.24 deg). Beyond case118 the
open-loop decoder is unreliable: on case500 and case2000 single seeds of the two injection arms range from 0.23x to 1.6x DC, and the worst
case2000 seed (15.8, offset 14.9) is the best-trained one, which is the amplification of a slightly wrong smooth
injection by L^-1 on a high-gain grid. The case2000 result also contradicts the PoC reading of P48 (collapse beyond the
training size, 8-12x DC): with training grids of 126-987 buses, 5 of the 6 injection seeds are 1.5-4x below DC there. The closed loop
(dcloop, coded at e8af9ae, not run) removes both the training collapse mechanism and the test-time amplification.

## Piano X, X3a: the injection decoder with gain 0.01 (X3a-P0-X3a-P6, 2026-10-13)

GPU jobs 63231 (H100 abacus-006: seeds 0, 1) and 63232 (H100 abacus-013: seed 2), code dd16233, `--inj-gain 0.01`,
otherwise X3's setting; all three ran the full 100 epochs. Scored with `r2_pt.py --runs results/pianox_pt_a --out
REPORT_pt_a.md`. Zero-shot VA in deg.

| id | verdict | what the numbers say |
|---|---|---|
| X3a-P0 | FAIL | seed 1 stays in the initial stuck state up to epoch 9 (validation 1.0-1.7, peak 1.12 after epoch 5); seeds 0 and 2 never exceed 0.42 and 0.26 after epoch 5. No seed relapses once healthy, against relapses in every X3 injection run |
| X3a-P1 | PASS | below DC on both case118 sets, every seed: case118_ieee [1.51, 1.88] vs 1.95; case118_tr [1.83, 2.22] vs 2.30 |
| X3a-P2 | FAIL | case500_goc [1.09, 3.71] vs 2.52: seeds 2.84, 3.71, 1.09 |
| X3a-P4 | FAIL | no seed above DC on case2000_goc: [3.18, 6.95] vs 9.75 (seeds 3.18, 6.95, 4.97) |
| X3a-P5 | PASS | best seed 1.51 / 1.09 / 3.18 on 118 / 500 / 2000 against fast-decoupled from the inputs 0.235 / 0.071 / 0.329 |
| X3a-P6 (exploratory) | — | below |

Exploratory: best validation loss 0.086, 0.079, 0.073 (epochs 97, 94, 97), below every X3 run (injection 0.094-0.150,
plain 0.144-0.182). Against X3's dcinjcanon, seed by seed: case118 alike (X3 1.57-1.89); case500 alike in spread (X3
1.60-3.21); case2000 without the outlier (X3 3.09-15.8). VM on PQ buses on case118 0.0011-0.0024, the public pretrained
model 0.0022. The error on case500 and case2000 is mostly offset (case500 seed 1: 3.50 of 3.71; case2000 seed 1: 6.69 of
6.95). Below the training sizes (57, 30, 14) the arm sits at DC (0.98-1.01x).

Reading: the gain removes the relapses, as the head-step diagnosis predicted, but not the initial stuck state, which
has the separate voltage-collapse mechanism; X3a-P0 fails on that state alone. With stable training the decoder
extrapolates beyond the training sizes better than X3a-P4 expected: every seed is 1.4-3x below DC on case2000. The
registered reading of that failure applies: once trained on many topologies, the open-loop injection correction is
amplified less than the PoC suggested. Its weak point is the global offset on case500, where seeds land on either
side of DC; that is the quantity the closed loop computes from the residual instead of predicting it.
