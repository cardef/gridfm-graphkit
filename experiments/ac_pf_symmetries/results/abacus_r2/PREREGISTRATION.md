# R2: M0+Canon vs M1 on held-out grids — predictions fixed 2026-10-02 before launch

Setting (`multigrid.py`, `cluster/train_r2.sbatch`): GNS (GENCO Base architecture) trained on three IEEE grids at
once, batches mixing grids, then evaluated zero-shot on a fourth. Two folds: train {14, 57, 118} / held out 30,
and train {14, 30, 118} / held out 57. Per grid 1024 scenarios (a fixed random subset of the 2048-scenario sets of
the replicate; base topology, θ_ref = 0), split 80/10/10 by the run seed, each training grid with its own
normalizer fitted on its train split; the held-out grid uses the first training grid's normalizer, which cannot
matter for these two arms (both exactly S3-invariant; E5b). Max 100 epochs, patience 40, batch 64. Seeds 0-2.
All 12 runs on the same CPU node type (abacus-004).

Arms: `canon` = Canonicalize(GNS), the Piano A baseline; `m1fullcanon` = Canonicalize(BranchAngleLayers(GNS)), the
branch representation with the reconstruction in every layer. Same frames, so a difference is one of
representation (H1).

Why this test: in the single-grid replicate M1 was on par with M0+Canon in distribution (P14) and zero-shot (P15),
with one hint: case30 zero-shot VM 0.011 vs 0.016, ranges overlapping by 3e-4. H1 is about unseen grids, which a
single source grid probes weakly.

Rule as before: with 3 seeds a difference is called only when the [min, max] ranges do not overlap.

| id | question | prediction | reading if it fails |
|---|---|---|---|
| P17 | H1 for M1 on held-out grids (case30 in fold 0, case57 in fold 1; VA and VM) | no separation on any (held-out grid, channel) | m1fullcanon entirely below canon on some (grid, channel) = H1 supported there; entirely above = the branch representation transfers worse |
| P18 | in distribution, every training grid of both folds, VA | no separation | a separation on a training grid is a representation effect in distribution |
| P19 | exploratory | — | canon's held-out error here vs the single-grid replicate (case14-only training, case30/57 zero-shot), descriptive: does adding training grids help this backbone transfer? |

## Fold 3, fixed 2026-10-05 after reading folds 1-2, before it runs: a held-out grid larger than the training ones

Motivation (R2 VERDICTS): in distribution M1 halved the angle error on case118 in both folds, by removing the
per-graph offset relative to REF that grows with grid size (canon: 0.11-0.15 deg on case14, 1.4-1.6 deg on
case118), while the two small held-out grids did not separate. If the branch representation is what removes that
offset, the gain must also appear zero-shot on a large grid the model never saw. Same protocol as folds 1-2
(1024 scenarios per grid, max 100 epochs, patience 40, 3 seeds, canon vs m1fullcanon), train {14, 30, 57},
held out case118. All six runs on one node: abacus-007 (Skylake, 12 CPUs each; moved there at launch, 09:10,
because abacus-004 was being held for a W28 job; the earlier folds ran on abacus-004 with 16).

| id | question | prediction | reading if it fails |
|---|---|---|---|
| P20 | H1 for M1 under size extrapolation | m1fullcanon's zero-shot VA range on case118 lies entirely below canon's | overlap: the branch representation's gain is an in-distribution effect only (H1 for M1 not supported even where it should be easiest) |
| P21 | where does it come from (only if P20 passes) | the gain is mostly in the per-graph offset (VA_cm ratio m1/canon below the VA_diff ratio) | a uniform gain = not the offset mechanism |
| P22 | exploratory | — | zero-shot VM and PBE on case118, in-distribution errors on 14/30/57 |

## M3 on fold 3, fixed 2026-10-05 after reading fold 3 and the DC/flat references, before it runs

Arm `m3canon` = Canonicalize(GNS, local=True), Piano A's M3: the S3 frame per bus instead of per graph. Every
power/admittance input of bus i, of its generators and of the edge rows leaving it is divided by s_i = D_i / D_ref,
with D_i = Σ_j |Y_ij| over the rows leaving i and D_ref the mean over training buses; predicted PG/QG are multiplied
back. These are M3's local dimensionless features (p_i = P_i/D_i; Y_ij/D_i per directed row, so the two rows of a
branch give D_j/D_i and only the global scale is lost), keeping Y_ij complex instead of |Y_ij|. Bus i's power
balance only involves its own quantities, so GNS's physics holds exactly in these variables (test
`test_m3_frame_divides_each_bus_power_balance_by_its_own_scale`). Side effect: the physics feedback and the
physics loss see bus i's residual divided by s_i (canon: by the graph's scale). S1 and S4 frames as in canon;
exactly equivariant (`test_m3_local_frame_is_exactly_equivariant`).

Same protocol as fold 3 (train {14, 30, 57}, held out case118, 1024 scenarios per grid, max 100 epochs, patience
40, seeds 0-2), compared with fold 3's canon runs. Node abacus-007 like those, with 6 CPUs per task instead of 12:
the node is shared, and the thread count changes the order of float sums, not the model.

| id | question | prediction | reading if it fails |
|---|---|---|---|
| P23 | H1 for M3 under size extrapolation | m3canon's zero-shot VA range on case118 lies entirely below canon's [10.06, 12.41] deg | overlap: local dimensionless inputs do not transfer to a larger grid better than a per-graph frame |
| P24 | does M3 close the gap to DC (1.95 deg on case118, `BASELINES.txt`)? | no: some seed's zero-shot VA stays above 1.95 deg | every seed below DC: the local frame fixes size extrapolation outright |
| P25 | exploratory | — | zero-shot branch differences θ_f − θ_t (canon 2.0 deg), VM and PBE on case118; in distribution on 14/30/57, including whether canon's fold-3 anomaly on case14/30 appears in M3 |

P23 states H1. After fold 3 (what fails zero-shot are the branch differences, which a frame does not compute) we
expect it to fail; the scoring does not depend on this expectation.

## M3 confirmation and ablation, fixed 2026-10-05 after reading M3 on fold 3, before they run

On fold 3, M3 lowered zero-shot VM and PBE (exploratory, P25). Two questions follow.

**Does it hold on other held-out grids?** `m3canon` on folds 1-2 (held out case30 / case57), seeds 0-2, compared
with those folds' canon runs. Node abacus-004 like them, 5 CPUs per task instead of 16 (shared node; the thread
count changes the order of float sums, not the model).

**Which of the two confounded parts of M3 does it?** Arm `canonresnorm` = Canonicalize(BusResidualScale(GNS))
(`resnorm.py`): canon's per-graph frame for every input, but the residual GNS feeds back into every layer and puts
in its physics loss is divided by D_i / D_ref, which makes it equal to M3's (D_i is degree-1 homogeneous, so canon's
frame cancels; D_ref is M3's constant). Tests: exact equivariance, and the first fed-back residual equals the
bare model's divided by D_i / D_ref bus by bus. Fold 3, seeds 0-2, abacus-007 with 6 CPUs (as M3).

| id | question | prediction | reading if it fails |
|---|---|---|---|
| P26 | M3's zero-shot VM gain on other held-out grids | m3canon's VM range entirely below canon's on case30 ([0.008638, 0.01324]) and on case57 ([0.01489, 0.02155]) | the fold-3 VM gain does not generalize across held-out grids |
| P27 | same for PBE | m3canon's PBE range entirely below canon's on case30 ([5.386, 5.823] MVA) and on case57 ([8.266, 12.63] MVA) | as above, for the power balance |
| P28 | where M3's VM gain comes from (case118, zero-shot VM; PBE reported alongside) | (i) the residual normalization: canonresnorm's VM range entirely below canon's [0.005432, 0.006928] and not entirely above M3's [0.002398, 0.004323] | (ii) overlapping canon's and entirely above M3's: the local inputs; (iii) below canon and above M3: both; otherwise unresolved at 3 seeds |

P28's prediction is a low-confidence call: the in-distribution PBE improved on all three grids, a feedback-like
signature, but q_i = Q_i/D_i is also the natural input for a local voltage deviation.

## DC prior, fixed 2026-10-07 before it runs

Arm `dccanon` = Canonicalize(DCPriorLayers(GNS)) (`dcprior.py`): in every layer GNS decodes its bus angles as
theta_DC + a learned correction. theta_DC is the DC power flow of `r2_baselines.py` (B theta = P from the inputs,
anchored at REF, one solve per graph, no gradient), exact under S1/S3/S4/S5, so the arm stays on canon's section and
differs from canon only in how the angles are represented; VM, PG and QG as in canon. Tests: theta_DC solves the DC
system and is far closer to the AC angles than the REF angle is; with a zero correction the model outputs theta_DC;
exact equivariance. theta_DC is computed in the forward pass (one dense solve per graph, as M1 does in each of its
12 layers), not precomputed: same numbers, simpler code; the training-time cost is measured (P32).

All three folds (train on three IEEE grids, hold out the fourth; 1024 scenarios per grid, max 100 epochs, patience
40, seeds 0-2), compared with canon and with DC on the same zero-shot scenarios (`BASELINES.txt`: 1.56, 1.08 and
1.95 deg on case30, case57 and case118). Nodes as for the M3 runs: folds 1-2 on abacus-004 (5 CPUs per task), fold
3 on abacus-007 (6 CPUs).

| id | question | prediction | reading if it fails |
|---|---|---|---|
| P29 | does the learned correction transfer to a larger grid? | zero-shot VA on case118 below DC's 1.95 deg for every seed | under size extrapolation the correction does not transfer: the gain over the learned models is DC's alone |
| P30 | the same on the held-out grids of training size | zero-shot VA below DC on case30 (1.56) and case57 (1.08) for every seed | as above, without the size shift |
| P31 | does the DC representation help a learned model? | zero-shot VA range entirely below canon's on all three held-out grids | the correction is so wrong on unseen grids that it undoes DC's advantage |
| P32 | exploratory | — | zero-shot VM, PBE, offset, rest and branch differences; in distribution vs canon and DC; training time vs canon |

Expectation, not part of the scoring: P31 is near certain (DC alone is 2-8x below canon's ranges); P29 and P30 are
the real test and I would not bet either way: the correction must stay small on grids it never saw.

### Amendment to the DC prior, fixed 2026-10-07 ~16:53 (committed in f44e933), before any dccanon result exists

After an adversarial review (the criterion "below DC" has no margin; P30 lumps two grids whose DC errors differ in
structure). No dccanon result file existed when this was written (checked by listing, not reading). Deviation logged:
fold 3 runs with 4 CPUs per task, not 6 (node occupancy; float summation order only).

DC's own split on the zero-shot scenarios (`BASELINES.txt`, deg): case118 offset 1.6, rest 1.11, branch differences
0.48; case30 1.51 / 0.40 / 0.34 (almost all offset: the losses DC ignores); case57 0.39 / 1.01 / 0.53 (mostly rest).

| id | question | prediction | reading if it fails |
|---|---|---|---|
| P29m | a margin over DC on case118 | zero-shot VA <= 0.75 × DC (1.46 deg) for every seed | P29 without P29m: "DC plus a small correction", no transfer claim |
| P30m | the same on case30 and case57 | <= 1.17 deg on case30 and <= 0.81 deg on case57, every seed | scored per grid; see the readings below |
| P33 | does the correction do the global part? | zero-shot offset on case118 below DC's 1.6 deg for every seed | the model cannot fix the loss-induced offset even when given DC: what is missing is loss allocation, not the linear solve |

Readings fixed now for the ambiguous patterns: 2 of 3 seeds below DC = "DC level", no claim; pass on case30 but not
case57 (or the reverse) is read through DC's error structure above (case30: global offset; case57: local rest); P29
pass with P30 fail = the correction helps where DC is worse, no general claim.

Decision rule for a separate Paper 2 (from the review): all of (i) P29m and P30m pass on case118 and case30, and
case57 is not above DC; (ii) P33 passes; (iii) dccanon beats DC + one Newton or fast-decoupled iteration on the same
scenarios (to be computed; it does not depend on the dccanon runs); (iv) the result holds with a correction
initialized at zero (to be run). Otherwise the DC reference and the DC prior go into Paper 1.

## Transfer, step 1 and 2: classical references and a learned local correction behind an exact solve, fixed 2026-10-08 before either is computed

After the DC prior: its learned correction transfers in its local part (rest and branch differences at or below DC's)
but not in the common shift relative to REF, which comes from how the losses are carried to the slack, a global
quantity. Two steps.

Step 1, classical references on the zero-shot scenarios (`r2_classical.py`, no training, inputs only): DC; DC with
loss compensation (three passes adding each branch's series loss at |V| = 1 as load, half per end); one full AC
Newton step from the DC angles and the voltage setpoints; Newton to convergence (iterations from the DC start and
from a flat start).

| id | question | prediction | reading if it fails |
|---|---|---|---|
| C1 | gate: does the solver reproduce the data's physics? | Newton to convergence matches the labels (VA < 1e-3 deg, VM < 1e-5 p.u.) on all four grids | the Ybus or the bus types differ from the data's physics: nothing below is read until fixed |
| C2 | does a physical loss allocation fix DC's offset? | DC with loss compensation: offset below DC's on case14, case30 and case118, total VA below DC on all four grids | the offset is not the series losses at |V| = 1 |
| C3 | how good is one classical step? | one Newton step from DC: VA below DC with loss compensation on all four grids, and below 0.25 deg; VM below 0.005 p.u. | if it holds, a learned model has an accuracy role zero-shot only if it beats one Newton step, otherwise only as a warm start |
| C4 | descriptive | Newton from the DC start needs fewer iterations than from a flat start | — |

Step 2, two arms (`dcprior.py`), both with the correction initialized at zero (the review's point: dccanon started
from a random correction), all three folds, seeds 0-2, nodes as dccanon:
- `dczcanon`: dccanon with the angle correction initialized at zero.
- `dcinjcanon`: the network's angle output is read as a per-bus injection correction c and turned into angles by
  the same anchored Laplacian solve as theta_DC: theta = theta_DC + L^-1 c, in every layer. The learned part is
  local (c, e.g. the losses at a bus); its global effect, including the common shift, comes out of the exact solve.

| id | question | prediction | reading if it fails |
|---|---|---|---|
| P34 | does the global part transfer when it comes out of the solve? | dcinjcanon's zero-shot offset at or below DC's on all three held-out grids, every seed | the local injection correction does not carry the loss allocation to unseen grids |
| P35 | margin over DC | dcinjcanon zero-shot VA <= 0.75 × DC on all three held-out grids, every seed | no useful transfer |
| P36 | correction through the solve vs angle correction | dcinjcanon's zero-shot VA range entirely below dczcanon's on all three grids | the solve does not help the learned correction transfer |
| P37 | does learning add to physics? | dcinjcanon zero-shot VA below DC with loss compensation (step 1) on all three grids, every seed | the learned correction adds nothing to a physical loss allocation |
| P38 | exploratory | — | dczcanon vs dccanon (does zero initialization change transfer?); VM, PBE, in distribution; one Newton step and Newton iterations started from each arm's prediction |

Decision rule for a transfer result: P34, P35 and P37 pass; if C3 holds, also fewer Newton iterations from
dcinjcanon's prediction than from DC (to be measured). Expectation: C1 and C3 likely; P34 and P37 open.

### Amendment to step 1, 2026-10-08 ~15:50 (committed in bddf446), after a failed gate and before the corrected run

The first run of `r2_classical.py` (job 62780) failed C1: Newton diverged. Cause, found on one scenario: the
framework's PF mask hides the REF bus's voltage magnitude as well as its QG (`gridfm_graphkit/datasets/masking.py`,
the "REF buses" block), so the input holds 0 there and Newton started from V = 0 at REF. The labelled state has zero
mismatch with this solver's Ybus and injections (checked), so the physics matches. Consequence beyond the bug: with
P and Q at REF unknown and its voltage hidden, the PF the models are asked to solve is one degree of freedom short
(unknowns N - 1 angles + the PQ and REF voltages; equations N - 1 P balances + the PQ Q balances): the REF voltage
level is not determined by the inputs, which bears on every zero-shot VM number in this file.

Corrected step 1: Newton twice, with the labelled REF voltage ("oracle", one input more than the learned models; C1
is checked on it) and with the mean of the graph's PV voltage setpoints ("inputs", the models' information). C3 and
C4 are read on both, with C3's thresholds unchanged. Disclosure: the first run's DC and DC-with-loss-compensation
rows do not depend on the bug and were seen: C2 fails (loss compensation raises the offset on case14, case30 and
case57 and lowers it only on case118; VA 1.87 / 1.94 / 1.56 / 1.48 deg against DC's 1.51 / 1.56 / 1.08 / 1.95).
Its pre-registered reading stands: DC's offset is not the series losses at |V| = 1. P37 keeps its definition (below
DC with loss compensation) and is now a weak test on three of the four grids.

## REF voltage given vs hidden (P39-P41), fixed 2026-10-08 before the runs

Arm `canonvref`: canon with the REF bus's voltage magnitude given as an input (`UnmaskRefVoltage`: restored from the
labels and marked known, so the model clamps it like the PV setpoints; an operator setpoint in practice). Everything
else as canon; all three folds, seeds 0-2; folds 1-2 on abacus-004, fold 3 on abacus-007, 4 CPUs per task. The code
is on branch `research/ac-pf-symmetries-vref` in a separate worktree (`SymmetricalFM/wt-vref`), so that the pending
step-2 runs keep importing the registered code of this checkout. VM is compared on the PQ buses only (`VM_PQ`, new in
`run.accuracy`), since canonvref does not predict the REF voltage and canon does; canon's comes from re-evaluating its
saved models with that code (`--eval-only`, `result_eval.json`), whose other metrics must reproduce `result.json`.

| id | question | prediction | reading if it fails |
|---|---|---|---|
| P39 | does the hidden REF voltage explain the zero-shot angle error? | no separation between canonvref and canon on zero-shot VA on any held-out grid: exact physics loses only 0.07-0.56 deg to it (`CLASSICAL.txt`), canon errs by 3.9-12.4 deg | canonvref entirely below canon somewhere: the learned model's guess of the voltage level drives its angle error far more than it does for physics |
| P40 | does it explain the zero-shot voltage error? | canonvref's zero-shot VM_PQ range entirely below canon's on case30 and case118, where the REF voltage estimate from the PV setpoints is off by 0.024 and 0.015 p.u. | the learned models' zero-shot VM error is not the voltage level |
| P41 | exploratory | — | case57 VM (the estimate is off by only 0.004 p.u. there); in distribution VM_PQ and VA (the REF voltage varies across scenarios on case14, case57 and case118); PBE; canon's VM error at the REF bus |

### Amendment to step 2, 2026-10-08 ~18:25, after a review and before any dcinjcanon result exists

Checked by listing file names only: no dcinjcanon result existed; two dczcanon results (fold 3, seeds 1-2) existed
and were not read. Changes, all to dcinjcanon:
- P34 as registered ("offset at or below DC's") passes with a zero correction, which reproduces DC exactly: it is not
  informative alone. Added P34s: zero-shot offset <= 0.75 × DC's offset on all three held-out grids, every seed. P34
  is still reported as registered.
- P37's threshold (DC with loss compensation) is weaker than DC on case30 and case57 (C2). Added P37s: zero-shot VA
  below a non-learned iteration from the same inputs, the P-theta fixed-point iteration theta <- theta + L^-1 (P_spec -
  P_ac(V, theta)) with L the DC matrix, 12 steps (one per GNS layer) from theta_DC, V fixed at the setpoints (PQ at
  1.0, REF at the mean PV setpoint), every seed on all three grids. GNS's per-layer physics feedback could learn
  exactly this iteration; if the model does not beat it, learning adds nothing. It will be computed by `r2_classical.py`
  before any dcinjcanon result is read (it does not depend on the runs); the full fast-decoupled iteration is reported
  alongside.
- "Local" in "local injection correction" describes the output (one value per bus), not its inputs (a 12-hop
  embedding with physics feedback): a test of locality needs a corrector without message passing, planned separately.
- The decision rule for a transfer result becomes P34s, P35 and P37s.

## Step 1b and two more arms (C5-C8, P42-P46), fixed 2026-10-08 before computing or running anything

After the third review: C2's failure fits the voltage level better than losses (with V_f V_t ~ 1.06-1.12, the same P
needs smaller angles than DC gives, so DC's angles are already too negative, and adding losses makes them more so).
And the REF voltage is an OPF output: in PF mode `gridfm_datakit` runs an OPF on every scenario and uses its voltages
as the setpoints of all generators, REF included (`process_network.py`, `process_scenario_pf_mode` and
`pf_preprocessing`), so it is a setpoint like the PV voltages the models already get, and a function of the loads
through an OPF whose costs and voltage limits the PF task masks.

Step 1b (`r2_classical.py`, inputs only, same zero-shot scenarios): DC with susceptances weighted by V_f V_t (V from
the setpoints, PQ at 1.0, REF at the mean PV setpoint), with and without loss compensation; the P-theta fixed-point
iteration of P37s (12 steps from theta_DC, V fixed as above); the full fast-decoupled iteration (12 steps); the signed
per-graph offset of every reference; the in-sample R^2 of the REF voltage on the PV setpoints (mean, min, max) and the
total loads, per grid. Plus a re-evaluation of the dccanon models with a signed offset metric.

| id | question | prediction | reading if it fails |
|---|---|---|---|
| C5 | is DC's offset the voltage level? | DC's signed offset is negative on case14, case30 and case57, and the V-weighted DC brings the offset to <= 0.5 deg on case14 and case30 | the offset is neither losses nor the voltage level: back to nonlocality or the slack balance |
| C6 | the P37s reference | the 12-step P-theta iteration is below DC on VA on all four grids | the fixed-point iteration does not help with V held at the setpoints |
| C7 | is the REF voltage predictable within a grid from the inputs? | R^2 >= 0.9 on case14, case57 and case118 (on case30 it is constant at 1.06) | the OPF's REF voltage depends on what the PF task hides, even within a grid |
| C8 | exploratory | — | dccanon's signed zero-shot offset on case30 and case57: if opposite in sign to DC's, the learned correction carries a constant of the training grids |

Two arms, all three folds, seeds 0-2, CPU (abacus-004 for folds 1-2, abacus-007 for fold 3, 4 CPUs per task):
- `dcinjvref`: dcinjcanon with the REF voltage given (UnmaskRefVoltage), the best angle arm on the well-posed task.
- `dcinj1hop`: dcinjcanon with a one-layer GNS: the injection correction sees one hop and gets no physics feedback.
  The test of whether the local part of the correction transfers without message passing.

| id | question | prediction | reading if it fails |
|---|---|---|---|
| P42 | with the setpoint known, does the correction behind the solve beat DC? | dcinjvref zero-shot VA <= 0.75 × DC on all three held-out grids, every seed | even on the well-posed task the learned correction does not transfer |
| P43 | is the untransferred shift the voltage level? | dcinjvref's zero-shot offset range entirely below dcinjcanon's on case30 and case57 | the shift is not the hidden setpoint |
| P44 | exploratory | — | dcinjvref against the oracle physics (exact) and the P-theta iteration with the oracle REF voltage; VM_PQ |
| P45 | does the local part transfer without message passing? | dcinj1hop's zero-shot rest and branch differences at or below DC's on all three grids, every seed | the local transfer seen with dccanon needed multi-hop context |
| P46 | exploratory | — | dcinj1hop vs dcinjcanon on every channel: what depth adds or costs in transfer |

## Size extrapolation of the trained arms (P47-P50), fixed 2026-10-09 ~16:00 before any model is evaluated there

Question: how do the arms trained on grids of at most 118 buses do on grids 4x and 17x larger? Setting: every saved
model of the three folds (arms canon, m1fullcanon, m3canon, canonresnorm, dccanon, dczcanon, dcinjcanon, canonvref,
dcinjvref, dcinj1hop; seeds 0-2; canonresnorm exists in fold 3 only) is evaluated zero-shot and unchanged on
case500_goc and case2000_goc (`multigrid.py --eval-only --extrap`, writes `result_extrap_<grid>.json` in the
registered run dir): the fixed 1024-scenario subset (seed 0) of a 2048-scenario random subset (seed 0) of the public
`gridfm/pf_small_<grid>` datasets, drawn from 20 of their ~1000 partitions spread over the whole set (partition 0 and
every 50th; the full sets are 13 and 80 GB and the filesystem is 98% full), bus/branch/gen tables only, same masks
and metrics as R2. References on the same scenarios and inputs: flat and DC (`r2_baselines.py`), the oracle Newton,
the fast-decoupled iteration and the inputs-only physics floor (`r2_classical.py`), computed by the data job before
any model result exists. Rule as before: ranges over the 3 seeds; a difference is called only when the ranges do not
overlap; "every fold" = the models of all three folds, scored fold by fold.

| id | question | prediction | reading if it fails |
|---|---|---|---|
| P47 | does the DC prior stay at DC level with size? | dccanon's zero-shot VA on case500_goc within [0.9, 1.1] x DC's VA on that grid for every seed of every fold, as on case118 (1.00 x) | above 1.1: the learned correction harms with size (the amplification the review feared); below 0.9: a gain that grows with size |
| P48 | does the injection correction's instability grow with size? | dcinjcanon does not beat DC on case500_goc: in every fold at least one seed has VA above DC's, and the seed max/min ratio of VA exceeds 2 in every fold | if some fold has every seed below DC: the injection correction transfers to 4x the training size; if the spread shrinks: the case118 instability was not a size effect |
| P49 | do the canon-like arms stay far from DC with size? | zero-shot VA of canon, m1fullcanon, m3canon and canonvref on case500_goc at least 4 x DC's for every seed of every fold (5.4-5.9 x on case118), with the per-graph offset the larger part of VA^2 | ratio below 4 somewhere: the canon-like arms lose less with size than the DC-based ones; offset not dominant: the local part degrades too |
| P50 | exploratory | — | the same on case2000_goc; offset / rest / branch differences against grid size; VM and VM_PQ against flat and the physics floor; PBE; dczcanon, dcinjvref and dcinj1hop; the cost of extrapolation (zero-shot on the large grid / in-distribution on the largest training grid); whether m3canon's and canonresnorm's VM and PBE gains over canon hold with size |

## Piano K, topology transfer on one grid (K-P1-K-P5), fixed 2026-10-09 ~23:55 before any N-k scenario exists

Question: on the same grid, how do GENCO and its DC-anchored variant degrade zero-shot with the depth of the
contingency, trained on N-1? Setting: case118_ieee; training data = the public `gridfm/pf_small_case118_ieee` set
(N-1), official seed-0 split (5000 train / 500 val, first 2000 official test ids); test sets = N-k generated by us with
gridfm-datakit from the public recipe (`cluster/nk/case118_k<k>.yaml`: 512 load scenarios x 4 topology variants, all
outage mass on exactly k, k in 1, 2, 3, 5, 10, 20, islanded topologies rejected by datakit). Arms: the released GENCO
checkpoints (base/small/tiny, seed 0; zero training), G0 (the backbone with the public config, our recipe), +dc,
+dcinj, GP (+vref +canon +resnorm +dc), 3 seeds each; PH = linear regressor of the REF voltage on the PV setpoints and
loads, fitted on train, followed by 12 fast-decoupled steps; references flat, DC, DC + losses, fast-decoupled from the
mean PV setpoint, damped Newton oracle. Metrics as R2 (VA with offset / rest / branch differences, VM on PQ buses,
PBE). Rule: ranges over 3 seeds; thresholds below are provisional until the k = 1 gate measures the seed noise, then
fixed. Full text: `SymmetricalFM/Piano K — Transfer zero-shot a nuove topologie.md`.

| id | question | prediction | reading if it fails |
|---|---|---|---|
| K-P1 | is the degradation with k global? | for G0 and the released checkpoints, from k = 1 to k = 10 the offset grows by >= 2x and the branch differences by < 1.5x | the learned local physics does not survive deep contingencies: the problem is not the global memory |
| K-P2 | does the DC anchor remove the k-dependence? | +dc's VA at k = 10 within 1.3x its VA at k = 1, and its VA/DC ratio shrinks by less than 20%; G0's VA/DC ratio halves | the anchor does not protect against deep contingencies: the learned correction degrades like the backbone |
| K-P3 | is +dc a floor? | +dc <= 1.1 x DC at every k, every seed | the anchor stops being a floor on stressed topologies |
| K-P4 | the hybrid | PH below 0.3 deg VA and 0.003 p.u. VM on PQ buses at every k, below every learned arm | the REF voltage depends on the topology more than the in-distribution R^2 suggests |
| K-P5 | gate | damped Newton with the labelled REF voltage reproduces the labels to 1e-5 deg on every scenario at every k; on our k = 1 set the released checkpoints err within 10% of their official-test error | our N-k sets are not comparable with the public set: regenerate |

## Piano X, X0: the physics frontier on new grids (X0-P1-X0-P4), fixed 2026-10-13 before running

Question: on grids no model was trained on, how accurate is physics from the same inputs at a learned surrogate's
compute budget, and how many iterations are left for a learned component to save? Setting: `frontier.py` on the fixed
1024 zero-shot scenarios of case14_ieee, case30_ieee, case57_ieee, case118_ieee, case500_goc, case2000_goc; DC start;
fast-decoupled (12 iterations) and damped Newton (6) recorded after every iteration; REF voltage given and from the
inputs; costs per scenario with the same sparse code (scipy SuperLU, one thread), factorizations included; the GENCO
backbone's forward time on the same graphs (random weights, 4 threads, batch 1 and 32) for scale. Thresholds apply to
the RMS over scenarios. Full text: `SymmetricalFM/Piano X — Estrapolazione zero-shot su topologie interamente nuove.md`.

| id | question | prediction | reading if it fails |
|---|---|---|---|
| X0-P1 | does physics already transfer at a surrogate's budget? | with the REF voltage given, fast-decoupled from the DC start below 0.5 deg VA after 2 iterations on every grid, and below 0.05 deg after 5 | where it needs more iterations, a learned accelerator has room: those grids become X1's targets |
| X0-P2 | cost | on case2000_goc, the DC start plus 2 fast-decoupled iterations cost at most 4x the DC solve (median per scenario, factorizations included) | fast-decoupled is too expensive at a surrogate's budget: a single-pass learned model keeps a cost niche |
| X0-P3 | where the inputs-only floor comes from | with the REF voltage from the inputs, fast-decoupled after 5 iterations within 20% of its 12-iteration VA on every grid | the inputs-only error is not only the hidden setpoint: the iteration itself is slow there |
| X0-P4 | exploratory | — | Newton per iteration and its divergences; worst scenarios; iterations to 0.1 and 0.01 deg; the backbone's forward time against fast-decoupled's |

## Piano X, X0b: Anderson-accelerated fast-decoupled (X0b-P1-X0b-P3), fixed 2026-10-13 after X0, before running

Question: does a classical, training-free acceleration remove the slow modes X0 found (the common offset on the large
grids at 2-3 iterations, case57's slow convergence), leaving nothing for a learned accelerator? Setting:
`frontier.py --methods anderson2 anderson3`: Anderson acceleration (type II, memory 2 and 3, full mixing, memory
restart when the fixed-point residual grows) of one full fast-decoupled iteration, so its fixed points are the exact
AC solutions and each step costs one iteration plus a least-squares problem of size m; same grids, scenarios and DC
start; REF voltage given and from the inputs.

| id | question | prediction | reading if it fails |
|---|---|---|---|
| X0b-P1 | does classical acceleration take the offset mode? | with the REF voltage given, Anderson with memory 3 below 0.1 deg after 3 iterations on every grid, including case2000 (plain 0.148) and case57 (plain 0.213) | where it does not, a learned accelerator has room that classical acceleration leaves |
| X0b-P2 | case57 | Anderson with memory 3 below 0.01 deg after 12 iterations on case57 (plain 0.061 and rising) | case57's problem is a map that does not contract, not a slow mode: it needs Newton or damping |
| X0b-P3 | exploratory | — | memory 2 vs 3; per-iteration overhead of the least squares; the inputs-only runs |

## Piano X, X3: GENCO with the injection decoder trained on the public subgrids (X3-P1-X3-P6), fixed 2026-10-13 before any X3 model exists

Question: does the architecture change that beat DC zero-shot on unseen grids within the training sizes at PoC scale
(the injection decoder: a learned per-bus injection correction whose global effect is computed by the exact solve on
the actual topology) make GENCO beat DC zero-shot on the benchmark where the paper reports its failure, once trained
on many topologies around that size? Setting: the 31 public pretraining subgrids of `gridfm/genco-pf-pretraining`
(the dataset lists 100 training, 10 validation and 10 test subgrids; only these 31 training ones are uploaded), 126-987
buses, 800-1,200 scenarios each from partitions 0, 8, 16, 24, 32, 40 (36,400 in all, 80/10/10 per grid by the run
seed); the PF task with the framework's mask; the GENCO Base backbone (hidden 48, 12 layers, 8 heads), 100 epochs,
batch 64, patience 40, the base config's optimizer; 3 seeds per arm; A40 GPUs. Arms: plain (GENCO as released, a
normalizer per training grid, no canonicalization), dcinjcanon (injection decoder and canonicalization), dcinjvref
(the same with the REF voltage given). Zero-shot sets, fixed 1024 scenarios (seed 0): case118_ieee, case118_tr (a
2,048-scenario subset, seed 0, of the 9,952 evaluation scenarios of GENCO's transfer experiment), case500_goc,
case2000_goc, case57_ieee, case30_ieee, case14_ieee. plain is evaluated with the source normalizer (the first training
grid's) and with one fitted on the target, and its predictions below hold for either; the canonicalized arms are
invariant to that choice by construction. References computed before any X3 model: DC and fast-decoupled from the same
inputs (`frontier.py`, X0, and `results/pianox_pt/FRONTIER_case118_tr.txt`), GENCO's public pretrained checkpoint
(100 subgrids, about 900,000 scenarios, masked pretraining, 300 epochs) and its case118 fine-tuned checkpoints at 100
and 1,000 samples (`results/pianox_pt/PUBLIC.txt`). Rule: 3 seeds; "every seed" means all three.

| id | question | prediction | reading if it fails |
|---|---|---|---|
| X3-P1 | the decisive one: does the injection decoder make GENCO beat DC where the paper fails? | dcinjcanon's zero-shot VA below DC's on case118_ieee (1.95 deg) and on case118_tr (DC from FRONTIER_case118_tr.txt), every seed | the injection correction does not transfer to an unseen grid even with 31 training topologies around its size: the PoC result on case30 and case57 was a small-grid effect |
| X3-P2 | inside the training size range | dcinjcanon below DC on case500_goc (2.51 deg), every seed | as X3-P1, at a size the training grids cover |
| X3-P3 | the control: data or architecture? | plain above DC on case118_ieee and case500_goc for every seed, with either normalizer | topology diversity alone makes GENCO transfer; the architecture change is not needed |
| X3-P4 | beyond the training size range | on case2000_goc, twice the largest training subgrid, at least one dcinjcanon seed above DC (9.75 deg) | once trained on many topologies, the injection correction extrapolates beyond the training sizes: the amplification argument is weaker than thought |
| X3-P5 | against physics | on case118_ieee, case500_goc and case2000_goc no learned arm below fast-decoupled from the same inputs after 5 iterations, REF voltage from the inputs for dcinjcanon and plain, given for dcinjvref | a learned model beats physics from the same inputs on an unseen grid |
| X3-P6 | exploratory | — | the public checkpoints against our arms; dcinjvref against dcinjcanon; the small grids below the training sizes; the offset / rest split; the in-distribution error on the training subgrids; time per epoch and per forward |
