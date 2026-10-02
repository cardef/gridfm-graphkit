# Cluster replicate of the case14 campaign — verdicts (2026-10-01/02)

Predictions: `PREREGISTRATION.md` (fixed before each batch ran). Scores: `prereg_output.txt`, produced by
`python -m experiments.ac_pf_symmetries.prereg` from the result files; P1-P9 on seeds 0-2 as registered.
Tables: `case14_ieee/REPORT.md` (all runs; m0 and canon have 5 seeds there). 46 runs (27 + 15 follow-up + seeds 3-4 of
m0/canon), 150 epochs unless early-stopped, T4.

| id | verdict | what the numbers say |
|---|---|---|
| P1 Prop. 5 in dist. | FAIL, narrowly (VA only) | canon VA [0.151, 0.183] deg vs m0 [0.183, 0.441]; with seeds 3-4 (exploratory) every channel overlaps |
| P2 E6b loss frame | FAIL | augcov VA 4.2-7.5 deg vs augmild 0.16: the loss in the sample frame does not rescue wide augmentation |
| P3 E6a physics term | FAIL | augnophys 3.7-6.0 deg vs its control m0nophys 0.14-0.23: neither does dropping the physics loss |
| P4 R7 | gain PASS, routing FAIL | augphase beats m0 (VA [0.085, 0.145] vs [0.183, 0.441]); the gain is uniform: ratios 0.42 (offset), 0.48 (rest), 0.41 (branch angles) |
| P5 H4 zero-shot | FAIL | canon's zero-shot VM is lower than m0's on average (case30 0.016 vs 0.019, case57 0.019 vs 0.033) but the ranges overlap; the PoC's single-seed gap does not separate over 3 seeds |
| P6 exactness | FAIL as registered, PASS frame-consistent | the registered ratio compared EE of PG/QG under S3 (measured in the transformed frame, powers / k) with an RMSE in the original frame; x k gives canon 1.5e-4 (rel. EE ~1e-6). m0: EE_S1(π)/RMSE ≥ 387, line flip ≤ 9e-5, transformer flip 0.03-0.11 |
| P7 H4 representative | FAIL | zero-shot VM means for a = 0 / 0.5 / 1 span less than 2 sd on both targets |
| P8 (exploratory) | — | a > 0 roughly doubles in-dist VA (canonmix 0.25-0.60, canonp95 0.30-0.45 vs canon 0.15-0.18 deg) |
| P9 which axis | FAIL, narrowly | augcovscale 1.00 deg (0.90-1.08, threshold > 1), augcovphase 0.47: each axis alone trains at 1.7x / 3.6x m0's error, both together give 21x (augcov): an interaction, not one broken axis |
| P10 (exploratory) | no wrapper effect | canon < m0 in 4/5 seeds; m0warm (canon's RNG path, no wrapper) 0.140-0.200 deg sits with canon; seed spread of one function class is 0.14-0.44 deg |

## What survives

1. Exactness and its cost: `Canonicalize` is exactly equivariant (all probes ~1e-4 frame-consistent) and costs
   nothing in distribution on base-topology data (P1 extension, P10). M0 ignores the REF-angle input (S1, never
   varied in training) and breaks S3 by 7-51x its own RMSE; line flips are exact, transformer flips break weakly.
2. Wide augmentation (α ±π, k 0.1-10) fails for a reason that is neither the loss frame nor the physics loss. A
   mechanism consistent with all four arms (not tested directly): GNS adds `physics_mlp(residual)` to the latent
   state in every layer and the residual scales as 1/k, so the S3 range changes the forward pass itself.
3. Mild phase augmentation is the best in-distribution arm (VA 0.115 deg), by a uniform reduction of every
   angle-error component; it is not "reference routing" in distribution.
4. Zero-shot (case14 -> case30/57, source normalizer) does not separate canon from m0 over 3-5 seeds; H4's choice
   of S3 representative does not matter at this scale. Per Prop. 5 neither result is a gain of symmetry: on
   these grids the conventions are identical, so the arms can differ only through training.

## Post-hoc observations (not pre-registered; to confirm before use)

- In every trained-well arm 81-96% of the node VA MSE is a per-graph common offset relative to the clamped
  REF angle (`VA_cm`); branch angle differences are ~3.6x more accurate than node angles.
- Phase augmentation degrades zero-shot VA on case30 through that offset: augphase offset 5.7 deg and augmild
  7.1 deg vs m0 1.5 deg (rest 2.2-2.6 vs 1.3); wide phase augmentation also destroys zero-shot VM (0.10-0.19
  p.u. vs 0.02). Reading: the network learns to carry the REF angle's value across case14's topology and that
  transport does not transfer; canonicalization never asks for it. Scale-only arms show no such effect.
- The replicate reproduces the PoC's ordering in distribution (augphase < augmild < canon < m0 on VA) on
  independent data; absolute errors are ~2x smaller (PoC m0 0.51 deg, here 0.28).

## F1, M1 (2026-10-02; predictions P11-P13 fixed before the runs)

`m1canon` = `Canonicalize(BranchAngleHead(GNS))`, 5 seeds, against `canon` seeds 0-4.

| id | verdict | what the numbers say |
|---|---|---|
| P11 in dist. | FAIL, in the wrong direction | m1canon is worse on every seed: VA [0.384, 0.945] deg vs canon [0.151, 0.327]; mean 0.73 vs 0.22; VM 0.0038 vs 0.0016, PG 6.9 vs 2.1 MW, PBE 0.51 vs 0.16 MVA |
| P12 zero-shot | PASS | no separation on case30/case57 × VA/VM (case30 VM 0.015 vs 0.016, case57 0.026 vs 0.025); zero-shot PBE much worse (46 vs 7 MVA on case30) |
| P13 exactness | PASS | worst frame-consistent EE/RMSE 5.8e-5 |

R7: the offset is not removed (0.70 deg of 0.73) and the branch differences themselves are ~3.3x worse than the
ones implied by canon's node angles (0.193 vs 0.059 deg). Confound, identified after the fact: in this variant the
reconstructed angles do not go through GNS's per-layer physics correction, which keeps acting on the backbone's own
node angles; the final angles come from one edge decoder plus a least-squares solve. The result reads "a single-shot
edge decoder on the final embeddings is worse than the 12-layer physics-corrected node decoder", not yet "the branch
representation is worse". A clean test of M1 needs the reconstruction inside every layer.
