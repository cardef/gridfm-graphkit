# Cluster replicate of the case14 campaign — predictions fixed before launch (2026-10-01)

Same protocol as the PoC (`run.py` defaults: 80/10/10 random split per seed, normalizer fitted on train,
patience 40, 150 epochs, batch 64, GENCO Base architecture), new data: 2048 random scenarios (seed 0) of
HierarchicalFM's 10k-scenario datakit sets for case14/30/57 (base topology, generated 2026-07-08), built by
`subset_data.py random`. The PoC's Mac data is not available here, so this is a replicate on independent
draws, not a re-run. Zero-shot keeps the source normalizer. Seeds 0, 1, 2; T4 GPUs.

Arms: `m0`, `canon`, `aug` (wide, loss in the augmented frame: the documented failure), `augcov` (E6b),
`augnophys` and its control `m0nophys` (E6a), `augmild`, `augphase`, `augscale`.

Reported as mean ± std over seeds; with 3 seeds a difference is called only when the ranges do not overlap.

| id | question | prediction | reading if it fails |
|---|---|---|---|
| P1 | Prop. 5 in distribution | `canon` and `m0` overlap on every in-dist channel (`Canonicalize` is the identity on these grids: θ_ref = 0, fixed topology, taps < 1) | a systematic in-dist gap = training-trajectory effect of the wrapper (e.g. `fit_scale_ref` consuming RNG), not of the function |
| P2 | E6b: is the wide-augmentation failure the loss frame? | `augcov` trains: in-dist VA within 2x of `augmild` and below 1 deg (PoC `aug`: 5.6 deg) | the frame of the loss is not the cause |
| P3 | E6a: is it the 1/k physics term? | `augnophys` in-dist VA within 2x of `m0nophys` | the PG/QG MSE drift or the range itself is the cause |
| P4 | R7: where does the phase-augmentation gain sit? | `augphase` beats `m0` on VA (PoC 0.09 vs 0.51 deg); "reference routing" predicts the gain is mostly in the common offset: VA_cm ratio (augphase/m0) below the VA_diff and θ_f − θ_t ratios | similar ratios on all three = a general regularization effect, not reference routing |
| P5 | H4 extension to 3 seeds (zero-shot, source normalizer) | `canon` VM error below `m0` on case30 and case57 (PoC seed 0: 0.012 vs 0.023, 0.017 vs 0.032) | the PoC's single-seed gain was seed noise |
| P6 | exactness | `canon` EE/RMSE < 1e-3 on every probe; `m0`: EE_S1(π)/RMSE > 10, `flip` (lines) < 1e-3, `fliptrafo` between 1e-3 and 1 | an implementation error in the actions or the wrapper |

## Follow-up, fixed 2026-10-01 before it runs: H4 at PoC scale

Two more canonicalized arms differing only in the S3 representative, s_a = P95(|input injections|)^a ·
mean(|Yff|)^(1−a): `canonmix` (a = 0.5) and `canonp95` (a = 1); `canon` is a = 0. Every a is an exact frame
(tests: degree-1 homogeneity, S1/S4 invariance, exact equivariance of the wrapped model). On this data mean |Yff|
is constant per grid, so `canon` is the identity in distribution, while a > 0 rescales every sample by its own load
level (exact, but the model then sees load-normalized injections with correspondingly rescaled admittances).

| id | question | prediction | reading if it fails |
|---|---|---|---|
| P7 | H4: does the representative along the S3 orbit change zero-shot error? | for at least one target, the zero-shot VM means of the three a-arms span more than 2x the largest seed std among them | the choice of a is irrelevant at this scale (one source grid, two small targets) |
| P8 | in-distribution cost of a > 0 | none predicted (exploratory) | — |

## Follow-up, fixed 2026-10-01 21:05 after P2/P3 failed (all 9 arms in except m0 seed 0), before it runs

Facts that motivate it: wide augmentation fails with the loss in the sample frame (`augcov` 4.2-7.5 deg) and
without the physics loss (`augnophys` 3.7-6.0 deg vs `m0nophys` 0.14-0.23). GNS feeds every layer's power
residual back into the latent state (`h_bus += physics_mlp(residuals)`), and that residual scales as 1/k, so a
wide S3 range changes the forward pass itself, whatever the loss. And `canon` beats `m0` in distribution in
every paired seed so far (PoC 3/3, here 2/2), although the wrapper is the identity function on these data.

| id | question | prediction | reading if it fails |
|---|---|---|---|
| P9 | which axis of `augcov` breaks training? | `augcovscale` (k 0.1-10, α = 0) fails (in-dist VA > 1 deg); `augcovphase` (α ±π, k = 1) trains (VA < 1 deg) | `augcovphase` failing too = the full-circle phase range is a problem of its own |
| P10 | exploratory: is canon < m0 in distribution data order? | `m0warm` (m0 with canon's RNG path: one shuffled train pass after init) is drawn from the same distribution as `canon` and `m0`; with seeds 0-4 the per-seed sign of canon − m0 is a fair coin | canon below both `m0` and `m0warm` on every seed = the wrapper changes training (unexplained by Prop. 5) |

Run from `SymmetricalFM/dev/gridfm-graphkit` (same data, results here) because the audit array still imports
this checkout's `run.py`/`symmetries.py`; seeds 3-4 of `m0`/`canon` run from this checkout.

## F1, first representation arm, fixed 2026-10-02 before it runs: M1 (Piano A §6)

`m1canon` = `Canonicalize(BranchAngleHead(GNS))` (`hodge.py`): an antisymmetric edge head on the backbone's final
bus embeddings predicts δ̂ per branch; bus angles are the REF-anchored least-squares (Hodge) reconstruction with
weights |Y_ft|; PG at REF and QG at PV/REF are recomputed from that state by the model's physics decoder, whose
residual replaces the last layer's in the physics loss. Same frames as `canon` (S1+S3+S4 exact; tests), so by Prop. 5
any difference from `canon` is a difference of representation (H1). Seeds 0-4, compared with `canon` seeds 0-4.

Motivation: R7 found 81-96% of the node-angle error to be a per-graph offset relative to REF, and branch differences
~3.6x more accurate than node angles. Against it: the plan's toy result (the best K-hop linear filter is not more
accurate on branch differences) and the fact that the REF-branch differences need the same global information
(the slack injection) as the offset itself.

| id | question | prediction | reading if it fails |
|---|---|---|---|
| P11 | H1 for M1 in distribution | no separation: m1canon and canon VA ranges overlap over seeds 0-4 | m1canon below canon on every seed = a representation gain in distribution |
| P12 | H1 for M1 zero-shot (case30, case57; VA and VM) | no separation on any (target, channel) | m1canon's range below canon's on some (target, channel) = a transfer gain of the representation |
| P13 | exactness | m1canon EE/RMSE < 1e-3 frame-consistent on every probe | implementation error |

R7 metrics (`run.angle_error_split`): VA² = VA_cm² + VA_diff² over the predicted buses of each graph (VA_cm =
per-graph mean error: an offset relative to the clamped REF angle, seen only by the branches incident to REF);
dVA = error of θ_f − θ_t, each branch once.
