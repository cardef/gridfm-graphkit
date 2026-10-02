# Piano A F0 on the released GENCO checkpoints — verdicts (2026-10-01/02)

Predictions: `GENCO_EXPERIMENTS.md` §5 and its amendments of 2026-10-01 (before any audit; E5d after seeing only
the 27 case14-sourced pairs, scored on the 81 others). Tables: `REPORT.md` (`genco_report.py`). Population: the
36 released IEEE PF checkpoints (14/30/57/118 × base/small/tiny × seeds 0/1/42), each on its own official split.

| id | verdict | numbers |
|---|---|---|
| G1-G3 | PASS 36/36 | PBE within ±0.5% of the released "PBE Mean" (in MVA), official test sizes, strict load |
| refit gate | PASS | the framework's P95 fit on the first 5000 official train ids reproduces the released baseMVA within ±0.5% |
| E0p | S1/S2 untestable, as predicted | REF angle 0 in every scenario, no phase shifter, on all 4 grids |
| E0p (topology) | canon not the identity in dist. | every scenario has one branch (77-93%) or generator out; CV of mean \|Yff\| 4.7 / 3.2 / 1.9 / 1.5% |
| E0p-dup | no leakage | exact duplicate scenarios 34.7 / 21.0 / 11.6 / 4.5%, all inside one load scenario; 0 test graphs with a train twin in all 36 splits |
| E1 | FAIL, kill fires | relative EE_S3 on VM 0.22-0.80 at k = 0.1 (predicted ≤ 0.1); at k = 10 up to 0.345 (case57 base; kill ≥ 0.3), VA up to 1.3; line flips ~1e-6 as predicted; transformer flips 3e-5 to 3e-4 |
| E1-S1 (control) | as expected | EE_S1(π)/RMSE 300-6700: the REF-angle input never varies in training |
| E1c (exploratory) | — | at k = 0.1, base breaks S3 2-3x more than small/tiny on case14 and case118 (rel 0.80/0.40/0.36, 0.66/0.23/0.22), not on case30/57 |
| E2 | not free (as amended for CV > 1e-3) | in-dist VA cost of post-hoc canon falls with grid size, tracking the CV above: +73% (case14, seed spread −32% to +210%), +9-18% (case30), +2-10% (case57), +1-1.5% (case118); worst EE rel 4e-5 (float32; above the registered 1e-5) |
| E5a | PASS | 540 checks: the S3 probe at k = refit/source base equals the refit run to 1.6e-4; \|ΔRMSE\| ≤ EE_S3(k) with 0 violations |
| E5b | PASS | canon's zero-shot predictions are identical under source and refit normalizers (7.5e-5, 108 pairs) |
| E5c | FAIL | ρ(EE_S3,VM(k=10) on the target, zero-shot VM RMSE) = 0.90, residualized on (source, target, size) 0.84; predicted \|ρ_partial\| < 0.3 |
| E5d-1 | PASS | held-out ρ_partial 0.756, 95% CI [0.57, 0.87] |
| E5d-2 | FAIL at the threshold (inconclusive) | ρ_partial(EE, raw − canon error) 0.499 [0.27, 0.66] vs > 0.5 |
| E5d-3 | PASS formally, not resolved | ρ_partial(EE, canon error) 0.694 [0.50, 0.82] vs raw 0.756; difference 0.063 [−0.10, 0.21] |

Zero-shot effect of post-hoc canon (source normalizer): VM better in 79/108 pairs (median error ratio 0.85; all
27 case14-sourced pairs, median 0.61; case57 sources 14/27, median 1.00); VA 59/108, median 0.99.

## What survives

1. The normalizer refit is an exact S3 action, so its effect on any zero-shot metric is bounded by EE_S3 at the
   refit's ratio, computed without labels (E5a, an identity verified 540 times). Protocol differences such as
   the PoC's "M0 moves up to 2x between normalizers" are symmetry breaking, nothing else.
2. Canonicalization makes zero-shot numbers independent of the normalizer (E5b) and, applied post hoc to the
   released weights, lowers zero-shot VM error on most pairs; in distribution it costs as much as the frame
   statistic varies there (E2): free on fixed topology, not on N-1 data.
3. The released models are far more sensitive to the MVA base than the PoC's M0 (E1).
4. EE_S3 on the target ranks released checkpoints by zero-shot VM error (E5d-1), but predicts the error left after
   canonicalization almost as well (E5d-3 unresolved). The simplest reading is generic sensitivity, not the
   S3-orbit mechanism: in this population EE works as a label-free warning sign, not as a measure of the error
   that the symmetry removes.

## Consequences for the text of Piano A (not applied; for the author)

- H2: the 2026-09-30 revision ("ρ ≈ 0 without an orbit component") is contradicted in this population; EE_S3
  predicts zero-shot error even where canonicalization has removed the orbit part.
- §3 GENCO row and §8: released checkpoints break S4 on transformers at EE/RMSE 0.05-0.37 (rel 3e-5 to 3e-4) and S3
  at rel 0.2-0.8 (k = 0.1); the PoC numbers (0.02) understate both.
- §7.1 E0: the public datasets are N-1 with parameter variation, so Prop. 5's "M0+Canon ≡ M0 in distribution"
  holds only on fixed-topology data; on the public sets post-hoc canonicalization has a measurable cost.
- §7.2: refitting the normalizer on the target reads labels (Qg, slack Pg) *and* acts on the inputs as an exact
  S3 transformation; without canonicalization the metric difference it causes is the model's S3 breaking at that
  k, with it the choice of normalizer is irrelevant.
