# Experiment Plan: Kron–Schur Communication for GridFM Scaling

**Problem:** Determine whether deterministic, topology-specific Kron–Schur communication improves zero-shot power-flow transfer per cumulative FLOP in one parameter-shared model across unseen grid topologies and sizes.

**Method thesis:** A sparse, parameter-free electrical hierarchy should provide more useful nonlocal communication than matched local, typewise-global-summary, and same-partition generic-hierarchy alternatives.

**Date:** 2026-07-16

**Implementation boundary:** `cardef/gridfm-graphkit`, current evidence commit `c690d6d6fd6e71187f0f4659c8daf52becbba69a`; upstream `gridfm/gridfm-graphkit` reference `b3d663b62179222c1ebec00ee29f67ea50e68c0b`; merge base `b3d663b62179222c1ebec00ee29f67ea50e68c0b`. I001–I009 and R001–R002 have typed hashed PASS records at the current evidence commit.

**Proposal source:** `refine-logs/FINAL_PROPOSAL.md`, SHA-256 `1f0aa148ac2a973b60d9c472e896a1394c48047ac566e801ae38dd7d388bbde1`.
**Current status:** the proposal is amended from `G32` to `G28` after a label-blind feasibility audit; targeted external re-review is pending. The pinned PGLib inventory yields one deterministic minimum-group assignment with 28 sources, 27 targets across six held-out groups, source maximum 4,917 buses, and 10 size-extrapolative targets; R002 formally PASS. I010, R003–R014, and all confirmatory runs remain BLOCKED until their prerequisites pass under the amended committed contract.

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| **C1 — compute-transfer scaling** | This is the paper's dominant contribution. It asks whether electrical hierarchy improves a domain-specific foundation model rather than a single-grid specialist. | At common cumulative-FLOP checkpoints, Kron must beat both Flat and Global at `G28/C` with an inverted one-sided 95% error bound below zero, retain a physical-residual upper bound no larger than `log(1.05)`, remain better at `C/2` and `C`, and show non-contracting point estimates from `G8` to `G28` and from the smallest to largest frozen size tercile. | B1, B3, B4 |
| **C2 — electrical operator rather than generic hierarchy** | Without this separation, a positive result supports only generic multiscale communication. | At `G28/C`, the inverted one-sided 95% bound for `log(error_Kron)-log(error_Quotient)` must be below zero with the same 5% residual gate. Partition, adapter, channel schema, coarse-node count, trainable capacity, data, and sparsity cap must match; realized nonzeros must be reported and charged in FLOPs rather than falsely equalized. | B2, B3, B4 |

### Anti-claims to rule out

| Anti-claim | Control |
|---|---|
| The gain comes from more parameters or training work. | Trainable parameters within 2% without dummy parameters; checkpoints at common counted cumulative FLOPs; exact parameters and work reported. |
| The gain comes from more data. | One fixed total scenario count across nested `G8 subset G16 subset G28`; only topology diversity changes. |
| The gain comes from a larger search budget. | At most twelve topology-only geometry policies and three Flat-only loss candidates; no arm-specific tuning. |
| Any hierarchy or cheap global mixer would work equally well. | Same-partition Quotient and released GridSFM-style typewise Global controls. |
| Target information leaked into preprocessing or model selection. | Target outputs are unreadable to normalization, partitioning, geometry, checkpointing, and hyperparameter code; violations invalidate the zero-shot block. |
| The current M0/M1 prototype already validates the claim. | M0/M1 code, configurations, caches, and results remain legacy evidence and cannot satisfy an I- or R-gate. |

### Reconciled legacy evidence (2026-07-15)

The migrated M1 payload is useful as failure evidence, not as a partial confirmatory campaign:

- MLflow experiment `702378410004452588` contains 30 run records: 11 `FINISHED` and 19 stale `RUNNING`; six stale records contain partial metrics and thirteen contain none.
- All 11 finished records are Flat/case2000 depth-16 or depth-32 variants. No Kron, Quotient, Global, case500, multi-topology, or held-out-topology comparison completed.
- The 11 finished records sum to 238.675 one-GPU elapsed hours. Several `FINISHED` depth-32 endpoints are numerically divergent, so MLflow lifecycle status is not a scientific success criterion.
- Forty first-wave SLURM logs fail before training with `Invalid experiment ID: '.stfolder'`; the Syncthing marker was inside the MLflow file-store root. Other logs record cancellation, host OOM kills, and Triton resource failures.
- The executed Git SHA is absent, per-run parameters are not logged, fitted per-grid normalizer artifacts are present, and the legacy objective/configuration violates the communication-only contract.

Consequences: legacy outcomes may size resource guards and regression tests, but they consume zero confirmatory budget, satisfy no I/R/E item, and cannot select a treatment, loss, checkpoint, geometry policy, or target set. The final runner must isolate the MLflow store below the Syncthing root and fail its preflight before reserving a GPU.

## Paper Storyline

### Main paper must prove

1. **B1:** Kron changes the held-out error–physics–compute frontier relative to Flat and Global.
2. **B2:** the full electrical operator family matters relative to the same-partition Quotient control.
3. **B3:** the benefit does not contract over the preregistered topology-diversity and target-size ranges, while build and inference costs remain visible.
4. **B4:** inference, failures, and resource accounting are topology-level, provenance-balanced, and fail closed.

### Appendix can support

- per-seed dispersion, every topology and provenance-group contrast;
- the reported-only wild-cluster-bootstrap-t sensitivity;
- raw VM and VA errors, bus-type residuals, operator diagnostics, and a sample-matched endpoint;
- optional few-shot adaptation only after the mandatory campaign and reserve close.

### Experiments intentionally cut

- model-size or dataset-size scaling laws;
- recursive or deeper hierarchies;
- HELM reconstruction, affine physical unpooling, `v_aff`, `cbus_x`, or coarse supervision;
- learned tokenizers, Graph Mamba, GraphGPS, linear attention, or another global-mixer axis;
- OPF, state estimation, dynamics, topology control, or multitask training;
- artificial padding of Quotient edges to mimic Schur fill;
- a third seed obtained by deleting a baseline, diversity level, or reserve.

No LLM, VLM, diffusion, or RL primitive is central. A frontier-necessity block is therefore omitted. The released GridSFM-style global summary is the strongest necessary domain-specific simple alternative.

## Frozen Estimands and Aggregation

### Per-scenario metrics

For each graph, compute errors only on masked unknowns after projecting known quantities:

```text
e_VM = RMSE(VM_hat - VM) / 0.01 p.u.
e_VA = RMSE(wrap(VA_hat - VA)) / (pi / 180 radians)
error = sqrt((e_VM^2 + e_VA^2) / 2)

residual = mean_bus sqrt(delta_P^2 + delta_Q^2) / baseMVA
```

The fixed scales make the headline error dimensionless and give 0.01 p.u. and 1 degree equal weight. Report `RMSE_VM_pu` and `RMSE_VA_rad` separately so the scalar cannot conceal a component failure. `wrap` maps angular error to `[-pi, pi)`. The residual uses case-declared `baseMVA`; missing or inconsistent metadata is a data failure.

### Aggregation order

1. Arithmetic-mean scenario metrics within each topology, checkpoint, arm, and seed.
2. Arithmetic-mean the two preregistered seed metrics within each topology.
3. Form paired topology log contrasts with `epsilon = 1e-12` only to protect the logarithm.
4. Arithmetic-mean topology contrasts within each `provenance_group`.
5. Use the unweighted arithmetic mean of equal-weight group contrasts as the exact sign-flip statistic.

Seeds and scenarios are never inferential replicates. Report scenario P95 and between-seed dispersion as diagnostics only.

### Inference

- Primary test: exact one-sided sign-flip randomization over provenance-group contrasts, `alpha = 0.05`.
- Interval: one-sided 95% bound obtained by inversion of the same test.
- Sensitivity only: wild-cluster-bootstrap-t with 9,999 fixed-seed Rademacher draws; it cannot rescue a failed exact test.
- Multiplicity: the claims use conjunctive intersection-union gates. No favorable checkpoint, baseline, target subset, or size bin may be selected after results.
- Construction failures stay in the denominator and are listed explicitly.

## Preregistered Constants and Freeze Rules

| Quantity | Rule | Freeze evidence |
|---|---|---|
| Source sets | Nested intact PGLib base cases: `G8 subset G16 subset G28`; outage variants do not count as systems. | `topology_manifest.yaml` hash |
| Total training scenarios | One exact count `S_total`, identical at G8/G16/G28; balance provenance groups then cases. | data-manifest hash |
| Target pool | At least 12 intact topologies, at least 6 independent provenance groups, roughly 0.5k–13.7k buses, and at least 4 targets in each frozen bus-count tercile. | target-manifest hash |
| Size extrapolation | `N_target > N_source_max`; at least 4 such targets across at least 2 target groups for the claim. | source extrema and target manifest |
| Geometry policy | At most 12 joint `(rho, k_P, k_C, kappa)` policies; deterministic residual/FLOP choice from source topology only. | candidate table and selected-policy hash |
| Capacity | Common model-size tier; widths and Flat depth `q` chosen deterministically to match trainable parameters within 2%. | parameter report |
| Loss weights | At most 3 candidates, Flat-HGNS seed 0 only, common `C_cal`, total Flat calibration bucket at most 3 GPU-hours. | calibration table and config hash |
| Design effect | `delta_min = -log(0.95)`. | analysis-config hash |
| Design dispersion | `sigma_design = sqrt(2) s_Flat` from the selected-loss Flat seed-0 `C_cal` checkpoint on held-out source-development groups. | power report |
| Power simulation | PCG64 seed `20260714`, 1,000,000 draws; smallest available group count at least 6 with at least 80% power. | power-report hash |
| Checkpoints | First crossing of counted cumulative work `{C/4, C/2, C}`; no target-selected checkpoint. | profiler and callback test |
| Training seeds | Exactly `0` and `1` for every mandatory arm/configuration. | run manifest |
| Compile mode | Shared across arms; enable only if output, gradient, and FLOP-counter parity all pass. | parity-test report |

Any unresolved quantity above blocks R014. A value may be determined by its frozen rule; it may not remain a free choice when the campaign starts.

## Experiment Blocks

### B0 — Integrity, calibration, and freeze

- **Claim tested:** all anti-claims; prerequisite to C1 and C2.
- **Why this block exists:** the current repository prototype violates the final no-leak and communication-only contracts. Training before these checks would generate uninterpretable evidence.
- **Dataset / split / task:** source topology metadata, source-training/source-development PF data, and target metadata only. Target outputs remain inaccessible.
- **Compared systems:** constructors and one common backbone configuration for Flat, Global, Kron, and Quotient; no efficacy comparison.
- **Metrics:** import-denial result, upstream-flat parity, row/column coverage, harmonic residual, conditioning, nonzeros, build time, host memory, parameter counts, FLOP parity, normalization provenance, target-output access audit, and power calculation.
- **Setup details:** complete I001–I010 and R001–R014. The `MetaPathFinder` deny-list begins at the first I007/F2 communication-seam commit and runs continuously.
- **Success criterion:** every implementation gate passes; `S_total`, topology manifests, geometry policy, loss vector, target-group count, `C`, runtime upper bounds, analysis code, and hashes are frozen at R014.
- **Failure interpretation:** no confirmatory treatment run may launch. Repair the contract or label subsequent work a pilot.
- **Table / figure target:** reproducibility/integrity table in the main paper; full gate ledger in the appendix.
- **Priority:** MUST-RUN.

### B1 — Main compute-transfer frontier

- **Claim tested:** C1.
- **Why this block exists:** it directly tests whether electrical nonlocal communication changes transfer at fixed capacity, scenarios, and compute.
- **Dataset / split / task:** PF on nested `G8/G16/G28` sources; zero-shot evaluation on the frozen held-out target manifest.
- **Compared systems:** Flat-HGNS, Global-HGNS, and Kron-HGNS.
- **Metrics:** headline dimensionless error, `RMSE_VM_pu`, `RMSE_VA_rad`, dimensionless physical residual, cumulative training FLOPs, GPU-hours, parameter count, warm inference latency, and failures.
- **Setup details:** 3 cores × 3 diversity levels × seeds `{0,1}` = 18 runs. Same scenario count, data order policy, objective, optimizer family, local stem/readout, checkpoints, and model-size tier.
- **Success criterion:** all five C1 conjunctive gates from the canonical proposal pass. The decisive comparison is `G28/C`; `C/2`, G8/G28, and size-tercile gates prevent checkpoint and scaling cherry-picking.
- **Failure interpretation:** Global tie/win favors the simpler global mechanism; Flat tie/win rejects hierarchy necessity; loss of the effect at larger size/diversity rejects the scalability thesis; a FLOP-matched failure rejects compute efficiency.
- **Table / figure target:** main Table 1 and error-versus-cumulative-FLOPs Figure 2.
- **Priority:** MUST-RUN.

### B2 — Electrical mechanism isolation

- **Claim tested:** C2.
- **Why this block exists:** a hierarchy gain alone is not evidence that circuit-derived geometry matters.
- **Dataset / split / task:** identical `G28` training data and frozen zero-shot targets used by B1.
- **Compared systems:** Kron-HGNS versus Quotient-HGNS; B1 Flat and Global remain contextual references.
- **Metrics:** paired error and residual contrasts, exact bounds, realized cross-level/coarse nonzeros, cumulative FLOPs, parameters, build cost, and inference cost.
- **Setup details:** two additional Quotient runs, seeds `{0,1}`; identical partition, adapter, channel schema, coarse-node count, sparsity cap, and training recipe. Quotient is not padded to imitate Schur fill.
- **Success criterion:** at `G28/C`, the inverted upper 95% error bound is below zero and the residual bound is no larger than `log(1.05)`.
- **Failure interpretation:** a tie collapses C2 to generic multiscale communication; a Quotient win makes the electrical operator unnecessary in the tested regime.
- **Table / figure target:** main mechanism panel in Figure 3 and Table 2.
- **Priority:** MUST-RUN.

### B3 — Size, diversity, and systems envelope

- **Claim tested:** the scaling and feasibility parts of C1; anti-claim that a quality gain hides unacceptable static or runtime cost.
- **Why this block exists:** the method is topology-specific and may fail during dense Kron construction even when sparse runtime inference is cheap.
- **Dataset / split / task:** all frozen target topologies, with predeclared size terciles and size-extrapolative subset.
- **Compared systems:** reuse all B1/B2 checkpoints; no new training runs.
- **Metrics:** group-balanced error contrasts by size tercile and G-level; `N_source_min/max`; static build wall time; dense-intermediate host peak; sparse operator bytes/nonzeros; accelerator peak; warm inference median/P95; end-to-end latency amortized over 1 and 1000 scenarios.
- **Setup details:** one clean-cache build measurement per topology plus three repeats for timing; batch size 1 for latency; 30 warm-up and 200 measured inference iterations when feasible, with the exact fallback count disclosed for very large grids.
- **Success criterion:** C1 size/diversity non-contraction gates pass; every failure remains visible; measured costs stay within R014's frozen resource envelope.
- **Failure interpretation:** resource-gate failures define the supported regime. The study reports the two-level limit and does not add recursive levels post hoc.
- **Table / figure target:** Figure 3 size/diversity panel and Figure 4 systems envelope.
- **Priority:** MUST-RUN.

### B4 — Robustness and failure accounting

- **Claim tested:** validity and interpretability of C1/C2.
- **Why this block exists:** with few independent groups, transparent contrasts and failure coverage matter more than many auxiliary benchmarks.
- **Dataset / split / task:** every confirmatory topology and checkpoint.
- **Compared systems:** reuse all 20 mandatory runs.
- **Metrics:** every topology contrast, equal-weight group contrasts, exact-test attainable levels, seed dispersion, scenario P95, wild-cluster sensitivity, construction failures, NaN/divergence events, and missing-data integrity failures.
- **Setup details:** analysis reads only immutable run manifests and evaluation artifacts. Target-arm labels remain blinded during campaign monitoring; monitoring may stop only for preregistered integrity or resource failures, never for efficacy.
- **Success criterion:** exact analysis reproduces from a clean checkout; no target, seed, checkpoint, or failure is silently omitted.
- **Failure interpretation:** exact-test failure stands even if a sensitivity analysis is favorable. Missing or mutable evidence invalidates the associated claim.
- **Table / figure target:** main failure-coverage row plus appendix per-topology forest plots.
- **Priority:** MUST-RUN.

## Mandatory Run Matrix

| Core | G8 | G16 | G28 | Seeds | Runs | Role |
|---|---:|---:|---:|---|---:|---|
| Flat-HGNS | yes | yes | yes | 0, 1 | 6 | local baseline |
| Global-HGNS | yes | yes | yes | 0, 1 | 6 | released-domain global-summary baseline |
| Kron-HGNS | yes | yes | yes | 0, 1 | 6 | proposed method |
| Quotient-HGNS | no | no | yes | 0, 1 | 2 | electrical-mechanism control |
| **Total** |  |  |  |  | **20** | fixed confirmatory matrix |

All 20 runs are mandatory once R014 passes. Operational launch order may prioritize G28 to expose resource failures, but target efficacy outputs remain sealed until the matrix completes. No arm may be dropped because an early result is unfavorable.

## Run Order and Milestones

| Milestone | Goal | Runs / artifacts | Decision gate | Cost / turnaround | Principal risk |
|---|---|---|---|---|---|
| **M0 — repository and protocol** | Establish the fork/upstream boundary, reconcile prior runs, and validate artifact-store isolation. | I001, R001–R002; legacy audit; MLflow child-store preflight | Repository, topology, data, provenance, and output-store contracts are explicit. | CPU only; 2–3 days | stale prototype assumptions or Syncthing markers leak into the new path |
| **M1 — geometry domain** | Implement label-blind partition, Kron, Quotient, conservative transport, and registry. | I002–I006; R003–R004 | Algebra, coverage, determinism, permutation, and resource tests pass. | CPU; about 1 week | dense `P` or Schur fill exceeds host limits |
| **M2 — common communication seam** | Make communication the sole learned treatment. | I007; continuous import denial | Flat compatibility and all four cores share one slot and output schema. | CPU plus tiny synthetic GPU smoke; about 1 week | legacy `GNS_hetero_hier` reuse |
| **M3 — portable data and accounting** | Make multi-topology PF evaluation zero-shot-safe and comparable. | I008–I010; R005 | `baseMVA`, masks, aggregation, FLOPs, checkpointing, and batching tests pass. | CPU; about 1 week | fitted per-grid normalization or inconsistent metric units |
| **M4 — calibration and power** | Freeze common choices without treatment information. | C001–C003, P001–P004, R006–R010 | Loss, geometry, capacity, target-group count, and runtime models fixed. | At most 10 GPU-hours total; about 1 week | insufficient independent target groups or low feasible `C` |
| **M5 — smoke and final freeze** | Validate execution and freeze the campaign. | S001, R011–R014 | G8 smoke passes without tuning; `C`, hashes, budget, and analysis dry-run pass. | Included in M4's 10-hour cap; 1–2 days | smoke reveals a cross-topology wiring defect |
| **M6 — confirmatory campaign** | Produce all mandatory evidence. | E001–E020 | All runs complete or failures are recorded; no efficacy stopping. | Planned training sum at most 183.3 GPU-hours; roughly 4 weeks | runtime tails or failed largest-grid construction |
| **M7 — locked analysis** | Evaluate C1/C2 and assemble paper artifacts. | A001–A004 | Tables/figures reproduce from immutable artifacts and claim language follows outcomes. | CPU plus evaluation GPU time already budgeted; 1–2 weeks | accidental target/checkpoint selection |

## Compute and Data Budget

### GPU arithmetic

Let `S_campaign` be the sum of the 20 per-run upper bounds after profiling. The binding rule is

```text
10 pre-campaign GPU-hours + S_campaign + 0.20 S_campaign <= 230 GPU-hours.
```

Therefore

```text
S_campaign <= (230 - 10) / 1.20 = 183.33 GPU-hours,
reserve       = 0.20 S_campaign <= 36.67 GPU-hours,
mean run cap  = 183.33 / 20 = 9.17 GPU-hours.
```

The mean is diagnostic, not a per-run quota; R012 uses the sum of core-specific upper bounds. The Flat throughput probe, `C_cal`, and three loss candidates share the 3-hour Flat calibration bucket. G28 profiles and S001 share the remaining 7-hour pre-campaign bucket. CPU data generation and geometry construction are measured separately.

If the largest common `C` satisfying the equation is below the preregistered source-only learning horizon, R014 fails. Do not delete a seed, baseline, diversity level, Quotient control, or reserve.

### Data needs

- open PGLib case metadata and synthetically generated PF scenarios;
- exactly frozen nested source sets and a provenance-group-held-out target pool;
- one fixed scenario total across G8/G16/G28;
- case-declared `baseMVA`, canonical schemas, and explicit missing-field masks;
- immutable topology, scenario, split, geometry, config, environment, and code hashes.

### Human evaluation

None.

### Biggest bottlenecks

1. obtaining enough independent provenance groups to achieve 80% design power for a 5% effect;
2. dense construction memory/time at the largest grids;
3. keeping the shared 3090-class campaign below the frozen runtime upper bound;
4. preventing legacy hierarchy and fitted-normalizer imports.

## Artifact Contract

Every run or gate writes one machine-readable record beneath the new confirmatory namespace `experiments/fm_scaling/`; legacy M0/M1 paths are read-only evidence. Mutable outputs live below `mlruns/fm-scaling/`, with the MLflow file store fixed at `mlruns/fm-scaling/mlflow-store/`, scheduler logs at `mlruns/fm-scaling/slurm-logs/`, and compact gate/run records at `mlruns/fm-scaling/result-summaries/`. MLflow must never scan repository-level `mlruns/`, because Syncthing owns `mlruns/.stfolder`. Each record contains:

- run/gate ID and status;
- fork commit, upstream reference, merge base, and worktree state;
- topology/data/geometry/config/environment hashes;
- system, G-level, seed, `C` thresholds, parameters, and counted FLOPs;
- wall time, GPU-hours, accelerator/host peaks, and failure reason;
- checkpoint and evaluation artifact hashes.

Campaign launch consumes an explicit run manifest. Wildcard YAML discovery is forbidden. Before any GPU reservation, the runner verifies that the MLflow store is a strict child of the Syncthing root, contains no `.stfolder`, can create/search a disposable experiment, and is not the legacy experiment store. Failure blocks the job and writes a machine-readable failure record.

The implemented runner also requires per-network hashed split tensors. Source cases alone may enter the balanced training sampler; every frozen target has empty train/validation splits and a complete test split. Training stops at the first batch crossing common `C`, persists weights at the first crossings of `C/4`, `C/2`, and `C`, and then evaluates every frozen target scenario from each saved checkpoint. The locked analyzer rejects a run matrix with any missing launch record, checkpoint, target topology, seed, or scenario.

## Stop / Go Rules

- **STOP before R014:** any I001–I010 failure, target leakage, missing group power, unbounded choice, incompatible Flat path, failed algebra, or infeasible budget.
- **STOP a job:** NaN/divergence, corrupted data, resource limit, or hash mismatch under a preregistered rule. Preserve the failure artifact.
- **DO NOT STOP for efficacy:** no interim target comparison can cancel or modify the fixed matrix.
- **GO to claims:** only locked A001–A004 outputs determine which claim boundary is supported.
- **OPTIONAL:** few-shot work begins only after all mandatory artifacts and reserve accounting close; it cannot rescue C1 or C2.

## Risks and Mitigations

| Risk | Failure condition | Mitigation |
|---|---|---|
| Power assumptions understate treatment dispersion. | Actual hierarchy group dispersion exceeds Flat or paired correlations are negative. | Treat the model as design-only; report achieved contrasts and exact inference; failed power/effect gate remains negative evidence. |
| Quotient comparison still confounds support density. | Reviewer interprets a Kron win as coefficient-only evidence. | Report both support densities and costs; claim only the full electrical operator family. |
| Metric scalar hides VM or VA failure. | Headline error improves while one physical component degrades materially. | Report both raw component RMSEs and require the preregistered scalar only for inference; discuss component divergence explicitly. |
| Large-grid build fails. | Coverage, conditioning, host-memory, or time gate fails. | Keep failure in denominator and report the operating boundary; do not tune targets or add levels. |
| Parameter/FLOP matching is approximate. | Parameter gap exceeds 2% or counter parity fails. | Deterministic width/`q` matching; block R014; report exact parameters and profiler cross-check. |
| Existing prototype contaminates results. | Forbidden module import or legacy artifact path appears. | F2-onward `MetaPathFinder` subprocess denial and clean-clone run. |
| Syncthing metadata contaminates MLflow discovery. | The configured MLflow store equals the Syncthing root, contains `.stfolder`, or fails a create/search smoke. | Fixed child store `mlruns/fm-scaling/mlflow-store/`; CPU preflight before `sbatch`; no wildcard experiment discovery. |
| Budget tail risk. | Sum of profiled upper bounds plus reserve exceeds 230 hours. | Lower common `C` only if above the frozen learning horizon; otherwise classify as pilot. |
| Fast-moving prior art. | A later work occupies the controlled study. | Repeat the primary-source novelty sweep before submission; do not add treatments mid-campaign. |

## Paper Artifact Map

| Artifact | Evidence source |
|---|---|
| Figure 1 — common backbone and fork boundary | I001–I010 implementation contract |
| Figure 2 — held-out error versus cumulative FLOPs for G8/G16/G28 | E001–E018, A001–A002 |
| Figure 3 — size-tercile and Kron-versus-Quotient contrasts | E001–E020, A002–A003 |
| Figure 4 — build, inference, memory, and failure envelope | P001–P004, E001–E020, A004 |
| Table 1 — main error/residual/compute results | A001 |
| Table 2 — mechanism isolation and matching audit | A002, A004 |
| Appendix — per-topology/group contrasts and sensitivities | A002–A004 |

## Final Checklist

- [ ] I001–I010 pass on a clean checkout.
- [ ] R014 freezes all manifests, choices, hashes, analysis, and budget.
- [ ] Main-paper tables are covered by the fixed 20-run matrix.
- [ ] C1 uses exact group-level inference and all conjunctive gates.
- [ ] C2 is isolated without false edge-count matching.
- [ ] Simplicity is defended by Flat, Global, and Quotient rather than extra variants.
- [ ] Frontier primitives are explicitly outside the claim.
- [ ] Target labels never affect preprocessing, geometry, tuning, or checkpoint selection.
- [ ] Failures remain in the denominator.
- [ ] MLflow child-store preflight passes before any GPU allocation.
- [ ] Nice-to-have few-shot work cannot delay or rescue the core evidence.
