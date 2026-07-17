# Experiment Tracker: Kron–Schur GridFM Scaling

**Date:** 2026-07-17

**Proposal SHA-256:** `385cd3ba44995ff8ddaa3250abbefc941f566b27f9cd2da61821bf601abf9547`

**Campaign status:** BLOCKED until I001–I010 and R014 are PASS.

**Proposal review:** PENDING for the G8/G16/G26 contract. The 9.05 / 10 READY
verdict covered G28 and is historical; the attempted G26 re-review failed
authentication before a verdict. No treatment is authorized.

**Status vocabulary:** `TODO`, `RUNNING`, `PASS`, `FAIL`, `BLOCKED`, `SKIPPED`.
**Legacy rule:** existing M0/M1 artifacts cannot change a status below.

## Implementation Gates

| ID | Milestone | Purpose | System / Variant | Split | Required Evidence | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| I001 | M0 | Freeze fork/upstream provenance | repository | n/a | origin, upstream ref, merge base, clean-clone instructions, exact environment/datakit source, artifact-store ownership | MUST | PASS | formal typed record at clean reachable GraphKit `c690d6d`, editable Datakit `b0d55d0`, pinned upstream/merge base `b3d663b`; MLflow child-store smoke passes |
| I002 | M1 | Define immutable geometry contracts | common | topology only | typed partition/operator/graph/provenance schemas; no scenario fields | MUST | PASS | formal typed hashed `contract-tests` record at clean reachable `93815a6`; 3/3 selected tests pass (job 54416) |
| I003 | M1 | Implement deterministic partition | Kron + Quotient | source topology only | stable-ID ordering, fixed METIS seed, anchor tie-break, permutation test | MUST | PASS | formal typed hashed `partition-tests` record at clean reachable `93815a6`; 2/2 selected tests pass (job 54416) |
| I004 | M1 | Implement Kron builder | Kron | source/target topology only | dense-reference Schur parity, coverage-or-fail, residual, conditioning, resource gates | MUST | PASS | formal typed hashed `kron-tests` record at clean reachable `93815a6`; 2/2 selected tests pass (job 54416); source policy calibration remains R003 |
| I005 | M1 | Implement Quotient builder | Quotient | source/target topology only | one-hot assignment, complex cut sums, four-channel schema, no Schur fill | MUST | PASS | formal typed hashed `quotient-tests` record at clean reachable `93815a6`; 2/2 selected tests pass (job 54416) |
| I006 | M1 | Implement content-addressed registry | all | multi-topology | immutable cache key/hash/device tests; no sample paths or copied operators | MUST | PASS | formal typed hashed `registry-tests` record at clean reachable `93815a6`; 2/2 selected tests pass (job 54416) |
| I007 | M2 | Extract one communication seam | Flat/Global/Kron/Quotient | synthetic + source | shared encoder/stem/slot/readout; output/gradient schemas; parameter report | MUST | PASS | formal typed hashed `seam-tests` record at clean reachable `93815a6`; 8/8 selected tests pass (job 54416); R004 is frozen |
| I008 | M3 | Implement portable PF data contract | all | source + target metadata | case-declared `baseMVA`, source-only optional stats, target-output unreadability test | MUST | PASS | formal typed hashed `data-tests` record at clean reachable `93815a6`; 8/8 selected tests pass (job 54416); no target outcomes were read |
| I009 | M3 | Implement balanced training/evaluation | all | G8/G16/G26 | provenance/case sampler, per-graph/component loss, known-value projection, metric unit tests | MUST | PASS | formal typed hashed `training-tests` record at clean reachable `93815a6`; 22/22 selected tests pass (job 54414) |
| I010 | M3 | Implement compute and compatibility gates | all | synthetic + largest grids | cumulative-FLOP checkpoint tests, profiler cross-check, compile parity, upstream-flat load, clean clone, MLflow child-store create/search smoke | MUST | BLOCKED | fresh CPU audit at clean `4a293b2` passes upstream identity, MLflow child-store smoke, clean-clone import, and flat schema/FLOP checks; blocked by absent reviewed upstream-flat checkpoint, unavailable CUDA compile/FLOP parity, and absent largest-grid data/peaks |

## Readiness and Freeze Gates

| ID | Milestone | Purpose | System / Variant | Split | Required Evidence | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | Build candidate topology inventory | data | all available cases | case IDs, provenance groups, bus counts, `baseMVA`, integrity status | MUST | PASS | formal record at clean reachable `93815a6` (job 54414); PGLib v23 commit `dc6be4b`: 66 cases, 65 eligible, 15 conservative provenance groups, 3–78,484 buses; one preserved integrity warning |
| R002 | M0 | Freeze source-development split rules | data | source candidates | G8⊂G16⊂G26 rule, fixed `S_total`, source-dev groups, no outage pseudo-systems | MUST | PASS | formal topology-only audit at clean reachable `93815a6` (job 54414): `S_total=11,655`; G26 maximal; PSERC/ACTIV source development; six target groups and 27 targets |
| R003 | M1 | Evaluate geometry candidates | Kron | source-development topology only | ≤12-policy table: residual, conditioning, nnz, FLOPs, build time, host peak | MUST | BLOCKED | 12-policy topology-only grid and explicit width-64 projected-FLOP model frozen at SHA-256 `3cb2a7a`; waits for audited source-development data |
| R004 | M2 | Match common capacity | all | source topology only | widths and Flat `q`; all parameter counts within 2% | MUST | PASS | formal typed record at clean reachable `93815a6` (job 54414): Flat 122/q1=898,655; Global 118=898,101; Kron/Quotient 123=897,657; relative gap 0.1112% |
| R005 | M4 | Freeze `C_cal` | Flat | source-development | treatment-blind throughput probe and 3-hour aggregate upper bound | MUST | TODO | probe is charged to Flat calibration bucket |
| C001 | M4 | Loss candidate 1 | Flat seed 0 | source-development | `C_cal` checkpoint, error, residual, GPU-hours | MUST | BLOCKED | waits for R005 |
| C002 | M4 | Loss candidate 2 | Flat seed 0 | source-development | `C_cal` checkpoint, error, residual, GPU-hours | MUST | BLOCKED | waits for R005 |
| C003 | M4 | Loss candidate 3 | Flat seed 0 | source-development | `C_cal` checkpoint, error, residual, GPU-hours | MUST | BLOCKED | waits for R005 |
| R006 | M4 | Select common loss vector | Flat only | source-development | deterministic candidate decision and config hash | MUST | BLOCKED | reducer and 3-hour aggregate guard ready; waits for C001–C003 |
| R007 | M4 | Estimate design dispersion | selected Flat seed 0 | held-out source-development groups | group errors, `s_Flat`, `sigma_design=sqrt(2)s_Flat` | MUST | BLOCKED | deterministic reducer ready; selected `C_cal` evidence absent |
| R008 | M4 | Freeze powered target-group count | analysis | candidate target groups | 1M-draw PCG64 power report, seed 20260714, ≥80% power, count ≥6 | MUST | BLOCKED | vectorized exact-sign power reducer ready; source-development dispersion absent |
| R009 | M4 | Freeze target manifest and terciles | data | held-out targets | selected groups/topologies, ≥4 per tercile, source extrema, extrapolation subset | MUST | BLOCKED | target outputs unreadable |
| P001 | M4 | Profile Local core | Flat | G26 source / largest source shapes | counted FLOPs, GPU-hours, wall time, peaks | MUST | BLOCKED | treatment-blind |
| P002 | M4 | Profile Global core | Global | G26 source / largest source shapes | counted FLOPs, GPU-hours, wall time, peaks | MUST | BLOCKED | exactly one summary slot |
| P003 | M4 | Profile Kron core and build | Kron | G26 source / largest source shapes | counted FLOPs, runtime fit, build time, host/GPU peaks | MUST | BLOCKED | failures preserved |
| P004 | M4 | Profile Quotient core and build | Quotient | G26 source / largest source shapes | counted FLOPs, runtime fit, build time, host/GPU peaks | MUST | BLOCKED | no padded edges |
| R010 | M4 | Fit core-specific runtime upper bounds | all | profiles | upper-bound model and uncertainty audit | MUST | BLOCKED | feeds common `C` |
| S001 | M5 | Treatment-blind execution smoke | common seam | G8 source-only | short run, checkpoints, hashes, no tuning | MUST | BLOCKED | included in remaining 7-hour pre-campaign bucket |
| R011 | M5 | Dry-run locked evaluator | analysis | synthetic immutable artifacts | aggregation, exact inversion, failure denominator, figure-table schema | MUST | BLOCKED | no target outcomes |
| R012 | M5 | Freeze common `C` and budget | all | campaign manifest | `10 + S + 0.2S ≤ 230`, learning-horizon gate, per-run upper bounds | MUST | BLOCKED | deterministic 20-bound budget reducer ready; core profiles and learning horizon absent |
| R013 | M5 | Freeze configs and hashes | all | all manifests | code/data/topology/geometry/config/environment/analysis hashes | MUST | BLOCKED | explicit manifest; no glob discovery |
| R014 | M5 | Final campaign authorization | all | n/a | I001–I010, R001–R013, C001–C003, P001–P004, and S001 PASS; reviewer-readable freeze packet | MUST | BLOCKED | sole gate to E001–E020 |

## Confirmatory Training Runs

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| E001 | M6 | C1 | Flat, G8, seed 0 | source G8 → frozen targets | error, residual, FLOPs, GPU-hours, failures | MUST | BLOCKED | checkpoints C/4,C/2,C |
| E002 | M6 | C1 | Flat, G8, seed 1 | source G8 → frozen targets | same | MUST | BLOCKED | waits for R014 |
| E003 | M6 | C1 | Global, G8, seed 0 | source G8 → frozen targets | same | MUST | BLOCKED | one global-summary slot |
| E004 | M6 | C1 | Global, G8, seed 1 | source G8 → frozen targets | same | MUST | BLOCKED | waits for R014 |
| E005 | M6 | C1 | Kron, G8, seed 0 | source G8 → frozen targets | same + geometry costs | MUST | BLOCKED | waits for R014 |
| E006 | M6 | C1 | Kron, G8, seed 1 | source G8 → frozen targets | same + geometry costs | MUST | BLOCKED | waits for R014 |
| E007 | M6 | C1 | Flat, G16, seed 0 | source G16 → frozen targets | error, residual, FLOPs, GPU-hours, failures | MUST | BLOCKED | checkpoints C/4,C/2,C |
| E008 | M6 | C1 | Flat, G16, seed 1 | source G16 → frozen targets | same | MUST | BLOCKED | waits for R014 |
| E009 | M6 | C1 | Global, G16, seed 0 | source G16 → frozen targets | same | MUST | BLOCKED | waits for R014 |
| E010 | M6 | C1 | Global, G16, seed 1 | source G16 → frozen targets | same | MUST | BLOCKED | waits for R014 |
| E011 | M6 | C1 | Kron, G16, seed 0 | source G16 → frozen targets | same + geometry costs | MUST | BLOCKED | waits for R014 |
| E012 | M6 | C1 | Kron, G16, seed 1 | source G16 → frozen targets | same + geometry costs | MUST | BLOCKED | waits for R014 |
| E013 | M6 | C1 | Flat, G26, seed 0 | source G26 → frozen targets | error, residual, FLOPs, GPU-hours, failures | MUST | BLOCKED | decisive setting |
| E014 | M6 | C1 | Flat, G26, seed 1 | source G26 → frozen targets | same | MUST | BLOCKED | decisive setting |
| E015 | M6 | C1 | Global, G26, seed 0 | source G26 → frozen targets | same | MUST | BLOCKED | decisive setting |
| E016 | M6 | C1 | Global, G26, seed 1 | source G26 → frozen targets | same | MUST | BLOCKED | decisive setting |
| E017 | M6 | C1/C2 | Kron, G26, seed 0 | source G26 → frozen targets | same + geometry costs | MUST | BLOCKED | decisive setting |
| E018 | M6 | C1/C2 | Kron, G26, seed 1 | source G26 → frozen targets | same + geometry costs | MUST | BLOCKED | decisive setting |
| E019 | M6 | C2 | Quotient, G26, seed 0 | source G26 → frozen targets | same + realized nnz/build cost | MUST | BLOCKED | same cap, no fill padding |
| E020 | M6 | C2 | Quotient, G26, seed 1 | source G26 → frozen targets | same + realized nnz/build cost | MUST | BLOCKED | same cap, no fill padding |

## Locked Analysis and Paper Artifacts

| ID | Milestone | Purpose | Inputs | Required Output | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| A001 | M7 | Aggregate immutable metrics | E001–E020 | scenario→topology→seed→group metrics; component errors | MUST | BLOCKED | failures retained |
| A002 | M7 | Test C1 and C2 | A001 | exact p-values, inverted bounds, all conjunctive gates | MUST | BLOCKED | wild bootstrap is sensitivity only |
| A003 | M7 | Evaluate size/diversity behavior | A001 | frozen-tercile and G8/G16/G26 panels | MUST | BLOCKED | G16 reported, not pass gate |
| A004 | M7 | Report systems envelope | P001–P004, E001–E020 | build/inference/memory/nonzero/failure table | MUST | BLOCKED | amortization at 1 and 1000 |
| X001 | post-M7 | Exploratory few-shot appendix | completed mandatory campaign | descriptive curves only | OPTIONAL | BLOCKED | outside 230-hour confirmatory budget; cannot rescue claims |

## Immediate Queue

### Reconciled performed evidence

| Evidence | Observed state | Gate effect |
|---|---|---|
| Legacy M0 summaries | CPU wiring, profiling, overfit, reconstruction, mmap, and prototype hierarchy checks completed | engineering hints only; no status change |
| Legacy M1 MLflow experiment `702378410004452588` | 30 records: 11 finished, 19 stale; only Flat/case2000 finished; 238.675 one-GPU elapsed hours across finished records | not confirmatory; no I/R/E credit |
| Legacy M1 SLURM logs | 40 `.stfolder`/MLflow discovery failures plus cancellation, OOM, and Triton resource failures | adds fail-closed store smoke to I010 |
| Current Abacus backend | shared Python 3.12 environment validated on an RTX A4000 with NumPy 2.4.6, PyTorch 2.12.1+cu126, PyG 2.8.0, and PyMETIS 2025.2.2; independent documentation replay passed | accelerator access is usable; I002–I009 and R001/R002/R004 have formal PASS receipts at `93815a6`; commit `e2ab5a8` clears the spawn-before-Julia deadlock and bounds derived chunk seeds to uint32; end-to-end smoke 54456 passes; G26 review, I010, R003, and later gates remain blocked; no treatment launch is authorized |

1. **Data generation:** requeue-enabled jobs 54458 and 54459 generate the frozen static-topology pools (2,331 scenarios for each source; 512 for each source-development/target) from the R002-selected members; dependent job 54460 audits all 55 outputs and runs R003 only after generation completes; target outcomes remain unreadable to selection code.
2. **R003:** run the bounded source-development-only geometry calibration after audited data are available and write its typed gate record.
3. **I010:** add a reviewed upstream-flat checkpoint fixture, then run CUDA compile/FLOP parity and largest-grid host/accelerator peaks on a GPU node after admissible data exist.
4. **R014:** only after every required I/R/C/P/S record is hashed PASS, materialize the explicit E001–E020 campaign and launch by manifest ID.


No GPU treatment job is authorized.
