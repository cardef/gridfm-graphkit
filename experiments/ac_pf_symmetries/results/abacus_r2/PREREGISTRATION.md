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
