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
