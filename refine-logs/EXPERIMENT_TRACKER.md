# Experiment Tracker: Kron–Schur GridFM Scaling

**Date:** 2026-07-14

**Proposal SHA-256:** `8063430ccb337f9e78be1c71374299288a485418a7e101a9e70e1a58b4410678`

**Campaign status:** BLOCKED until I001–I010 and R014 are PASS.

**Status vocabulary:** `TODO`, `RUNNING`, `PASS`, `FAIL`, `BLOCKED`, `SKIPPED`.
**Legacy rule:** existing M0/M1 artifacts cannot change a status below.

## Implementation Gates

| ID | Milestone | Purpose | System / Variant | Split | Required Evidence | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| I001 | M0 | Freeze fork/upstream provenance | repository | n/a | origin, upstream ref, merge base, clean-clone instructions, environment lock | MUST | TODO | origin is `cardef/gridfm-graphkit`; upstream remote/reference still must be recorded |
| I002 | M1 | Define immutable geometry contracts | common | topology only | typed partition/operator/graph/provenance schemas; no scenario fields | MUST | TODO | new additive path; do not reuse legacy hierarchy object |
| I003 | M1 | Implement deterministic partition | Kron + Quotient | source topology only | stable-ID ordering, fixed METIS seed, anchor tie-break, permutation test | MUST | TODO | cannot read bus class, labels, or target results |
| I004 | M1 | Implement Kron builder | Kron | source/target topology only | dense-reference Schur parity, coverage-or-fail, residual, conditioning, resource gates | MUST | TODO | runtime stores sparse geometry only |
| I005 | M1 | Implement Quotient builder | Quotient | source/target topology only | one-hot assignment, complex cut sums, four-channel schema, no Schur fill | MUST | TODO | same partition/coarse count/sparsity cap; no edge padding |
| I006 | M1 | Implement content-addressed registry | all | multi-topology | immutable cache key/hash/device tests; no sample paths or copied operators | MUST | TODO | clean-cache and collision tests required |
| I007 | M2 | Extract one communication seam | Flat/Global/Kron/Quotient | synthetic + source | shared encoder/stem/slot/readout; output/gradient schemas; parameter report | MUST | TODO | start `MetaPathFinder` legacy-import denial at first commit |
| I008 | M3 | Implement portable PF data contract | all | source + target metadata | case-declared `baseMVA`, source-only optional stats, target-output unreadability test | MUST | TODO | fitted per-grid and per-sample normalizers forbidden |
| I009 | M3 | Implement balanced training/evaluation | all | G8/G16/G32 | provenance/case sampler, per-graph/component loss, known-value projection, metric unit tests | MUST | TODO | no coarse/reconstruction/hierarchy-specific loss |
| I010 | M3 | Implement compute and compatibility gates | all | synthetic + largest grids | cumulative-FLOP checkpoint tests, profiler cross-check, compile parity, upstream-flat load, clean clone | MUST | TODO | include host/GPU peak and multi-topology batching |

## Readiness and Freeze Gates

| ID | Milestone | Purpose | System / Variant | Split | Required Evidence | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | Build candidate topology inventory | data | all available cases | case IDs, provenance groups, bus counts, `baseMVA`, integrity status | MUST | TODO | whole groups held out |
| R002 | M0 | Freeze source-development split rules | data | source candidates | G8⊂G16⊂G32 rule, fixed `S_total`, source-dev groups, no outage pseudo-systems | MUST | TODO | exact members freeze at R014 |
| R003 | M1 | Evaluate geometry candidates | Kron | source-development topology only | ≤12-policy table: residual, conditioning, nnz, FLOPs, build time, host peak | MUST | TODO | deterministic choice rule |
| R004 | M2 | Match common capacity | all | source topology only | widths and Flat `q`; all parameter counts within 2% | MUST | TODO | no dummy parameters |
| R005 | M4 | Freeze `C_cal` | Flat | source-development | treatment-blind throughput probe and 3-hour aggregate upper bound | MUST | TODO | probe is charged to Flat calibration bucket |
| C001 | M4 | Loss candidate 1 | Flat seed 0 | source-development | `C_cal` checkpoint, error, residual, GPU-hours | MUST | BLOCKED | waits for R005 |
| C002 | M4 | Loss candidate 2 | Flat seed 0 | source-development | `C_cal` checkpoint, error, residual, GPU-hours | MUST | BLOCKED | waits for R005 |
| C003 | M4 | Loss candidate 3 | Flat seed 0 | source-development | `C_cal` checkpoint, error, residual, GPU-hours | MUST | BLOCKED | waits for R005 |
| R006 | M4 | Select common loss vector | Flat only | source-development | deterministic candidate decision and config hash | MUST | BLOCKED | total C001–C003 plus probe ≤3 GPU-hours |
| R007 | M4 | Estimate design dispersion | selected Flat seed 0 | held-out source-development groups | group errors, `s_Flat`, `sigma_design=sqrt(2)s_Flat` | MUST | BLOCKED | selected `C_cal` checkpoint only |
| R008 | M4 | Freeze powered target-group count | analysis | candidate target groups | 1M-draw PCG64 power report, seed 20260714, ≥80% power, count ≥6 | MUST | BLOCKED | insufficient groups blocks campaign |
| R009 | M4 | Freeze target manifest and terciles | data | held-out targets | selected groups/topologies, ≥4 per tercile, source extrema, extrapolation subset | MUST | BLOCKED | target outputs unreadable |
| P001 | M4 | Profile Local core | Flat | G32 source / largest source shapes | counted FLOPs, GPU-hours, wall time, peaks | MUST | BLOCKED | treatment-blind |
| P002 | M4 | Profile Global core | Global | G32 source / largest source shapes | counted FLOPs, GPU-hours, wall time, peaks | MUST | BLOCKED | exactly one summary slot |
| P003 | M4 | Profile Kron core and build | Kron | G32 source / largest source shapes | counted FLOPs, runtime fit, build time, host/GPU peaks | MUST | BLOCKED | failures preserved |
| P004 | M4 | Profile Quotient core and build | Quotient | G32 source / largest source shapes | counted FLOPs, runtime fit, build time, host/GPU peaks | MUST | BLOCKED | no padded edges |
| R010 | M4 | Fit core-specific runtime upper bounds | all | profiles | upper-bound model and uncertainty audit | MUST | BLOCKED | feeds common `C` |
| S001 | M5 | Treatment-blind execution smoke | common seam | G8 source-only | short run, checkpoints, hashes, no tuning | MUST | BLOCKED | included in remaining 7-hour pre-campaign bucket |
| R011 | M5 | Dry-run locked evaluator | analysis | synthetic immutable artifacts | aggregation, exact inversion, failure denominator, figure-table schema | MUST | BLOCKED | no target outcomes |
| R012 | M5 | Freeze common `C` and budget | all | campaign manifest | `10 + S + 0.2S ≤ 230`, learning-horizon gate, per-run upper bounds | MUST | BLOCKED | planned `S ≤ 183.33` GPU-hours |
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
| E013 | M6 | C1 | Flat, G32, seed 0 | source G32 → frozen targets | error, residual, FLOPs, GPU-hours, failures | MUST | BLOCKED | decisive setting |
| E014 | M6 | C1 | Flat, G32, seed 1 | source G32 → frozen targets | same | MUST | BLOCKED | decisive setting |
| E015 | M6 | C1 | Global, G32, seed 0 | source G32 → frozen targets | same | MUST | BLOCKED | decisive setting |
| E016 | M6 | C1 | Global, G32, seed 1 | source G32 → frozen targets | same | MUST | BLOCKED | decisive setting |
| E017 | M6 | C1/C2 | Kron, G32, seed 0 | source G32 → frozen targets | same + geometry costs | MUST | BLOCKED | decisive setting |
| E018 | M6 | C1/C2 | Kron, G32, seed 1 | source G32 → frozen targets | same + geometry costs | MUST | BLOCKED | decisive setting |
| E019 | M6 | C2 | Quotient, G32, seed 0 | source G32 → frozen targets | same + realized nnz/build cost | MUST | BLOCKED | same cap, no fill padding |
| E020 | M6 | C2 | Quotient, G32, seed 1 | source G32 → frozen targets | same + realized nnz/build cost | MUST | BLOCKED | same cap, no fill padding |

## Locked Analysis and Paper Artifacts

| ID | Milestone | Purpose | Inputs | Required Output | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|
| A001 | M7 | Aggregate immutable metrics | E001–E020 | scenario→topology→seed→group metrics; component errors | MUST | BLOCKED | failures retained |
| A002 | M7 | Test C1 and C2 | A001 | exact p-values, inverted bounds, all conjunctive gates | MUST | BLOCKED | wild bootstrap is sensitivity only |
| A003 | M7 | Evaluate size/diversity behavior | A001 | frozen-tercile and G8/G16/G32 panels | MUST | BLOCKED | G16 reported, not pass gate |
| A004 | M7 | Report systems envelope | P001–P004, E001–E020 | build/inference/memory/nonzero/failure table | MUST | BLOCKED | amortization at 1 and 1000 |
| X001 | post-M7 | Exploratory few-shot appendix | completed mandatory campaign | descriptive curves only | OPTIONAL | BLOCKED | outside 230-hour confirmatory budget; cannot rescue claims |

## Immediate Queue

1. **I001:** record upstream reference/merge base and clean-clone contract.
2. **R001:** build the case/provenance/`baseMVA` candidate inventory.
3. **I002:** implement the immutable, topology-only geometry schema.

No GPU treatment job is authorized.
