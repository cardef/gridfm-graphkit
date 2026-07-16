# Experiment Results and Bridge Status

## 2026-07-16 maximal local-CPU pass

**Confirmatory status:** 0/20 E-runs authorized; 0/20 completed. R014 remains
BLOCKED. No efficacy result exists.

The local CPU environment was rebound to clean GraphKit commit `4a293b2` and
clean editable Datakit commit `b0d55d0`. Environment spec `f5f0604c` passed
imports, the seeded matrix witness, and an independent agent replay of the
documented command with checksum `882.053564644979`.

Fresh clean-commit evidence under
`mlruns/fm-scaling/result-summaries/evidence/` records:

- I001 PASS at `I001-preflight-4a293b2-b0d55d0.json`;
- I002--I009 PASS at `I002-4a293b2.json` through `I009-4a293b2.json`, with
  49 selected tests in total;
- R001 PASS at `R001-pglib-v23-4a293b2.json` and R002 PASS at
  `R002-split-audit-4a293b2.json` from the clean pinned PGLib v23 checkout;
- I010 remains BLOCKED at `I010-cpu-4a293b2.json`: upstream identity, the
  MLflow child-store smoke, clean-clone import, and Flat schema/FLOP checks
  pass; a reviewed upstream-Flat checkpoint, CUDA compile/FLOP parity, and
  largest-grid host/accelerator measurements remain absent.

Repository-wide CPU verification on the final hardened worktree passed
134/134 unit tests in 191.65 s. The focused FM-scaling subset passes 35/35;
focused pre-commit hooks pass. The documented Bandit command was found to scan
zero files because it omitted recursion. After changing it to `bandit -r` and
removing two `shell=True` integration-tool call sites, the effective scan
covered 22,053 lines and found zero high-severity issues. Integration-test
collection succeeds (2 tests); the training integration tests were not run
because they download 10,000-scenario archives, modify example configs, and
require a machine-specific calibration baseline.

The CPU pass also found that the old R003 selector contradicted the proposal:
it minimized residual first instead of choosing the least projected sparse
message work inside the 5%-of-best worst-residual band. The worktree now
implements the residual-band rule and requires the hashed candidate artifact
to provide an explicit cross-edge/coarse-edge/coarse-node FLOP model. There is
no hidden default. The evidence validator independently recomputes the choice.

No final data generation or R003/R004 evidence was produced. Four
load-bearing inputs are not frozen anywhere in the canonical contract:

1. exact `S_total` and hence per-topology scenario counts;
2. the whole-provenance-group source-training/source-development split;
3. the at-most-12 geometry candidate table and its explicit projected-FLOP
   model;
4. the deterministic width/Flat-`q` capacity-search domain.

Guessing any of these would violate the plan's unbounded-choice stop rule.
Only 15 GiB of local disk was free during this pass, so generating an
unfrozen 55-topology payload would also be operationally unsafe. Once these
four inputs are reviewed and frozen, CPU can render/generate/audit the data,
run R003/R004, build geometry, and materialize split artifacts. GPU-only I010,
M4 calibration/profiling, S001, and R014 remain outside this host's capability.

## 2026-07-15 implementation completion

**Confirmatory status:** 0/20 E-runs authorized; 0/20 completed. R014 remains BLOCKED. No efficacy result exists.

The communication-only experiment implementation is now present: immutable Kron/Quotient geometry, Flat/Global/Kron/Quotient communication cores, case-declared normalization, target-safe data/split tooling, provenance/case-balanced sampling, graph-balanced objective, masked wrapped VM/VA evaluation, first-crossing cumulative-FLOP checkpoints, strict campaign manifests, fail-closed launch, and locked C1/C2 analysis.

Verification on the current CPU host:

- focused post-review bridge/geometry/model/data/analysis suite: 42 passed;
- full repository unit suite before the final contract-only hardening pass: 116 passed, 184 warnings, 311.28 s; the changed bridge subset was rerun afterward;
- Ruff and flake8 on every changed/new Python path: passed;
- real PyMetis contiguous deterministic smoke: passed;
- subprocess legacy-import denial through the full confirmatory task entry point: passed.

This is code readiness, not gate readiness. I001 is still blocked because the shared `../.venv` imports `gridfm-datakit` from `.claude/worktrees/fix-branch-limit-cache`, not the exact sibling root, and the sibling checkout is dirty/unpublished for confirmatory provenance. I010 still needs clean-clone/upstream-flat compatibility, CUDA compile/FLOP parity, and largest-grid resource evidence. R001-R014, C001-C003, P001-P004, and S001 have not been executed under the final contract.

### Post-review hardening

The first full code review was correctly BLOCKED. The implementation was then changed so that:

- the real CLI applies distinct checkpoint/output overrides and passes the subprocess legacy-import denial;
- training is train-only, seals immutable checkpoint/FLOP/runtime hashes, and target evaluation is impossible until all 20 training records are sealed and unchanged;
- confirmatory dataset construction performs no outcome-dependent scenario filtering; the fixed degenerate-solution rule rejects and records the whole topology before target freeze;
- target size terciles and extrapolation flags are derived from frozen bus counts, with the required case/group minima enforced;
- geometry identity excludes timing, construction failures remain in a per-topology report, and datakit generation emits an executing-process provenance sidecar from the exact sibling fork;
- gate records require gate-specific evidence kinds; locked analysis now includes component metrics, scenario P95, seed dispersion, wild-cluster sensitivity, resource records, and sealed metric hashes;
- deterministic reducers exist for bounded geometry selection, capacity matching, loss selection, design dispersion/power, and the 20-run budget equation.

These changes close code-review defects; they do not manufacture missing source-only calibration, CUDA profiling, or clean-SHA evidence. R014 remains BLOCKED.

## 2026-07-15 bridge reconciliation

**Confirmatory status:** 0/20 E-runs authorized; 0/20 completed. R014 remains BLOCKED.

The migrated M1 payload was inspected directly rather than inferred from the old plan:

| Item | Observed result | Interpretation |
|---|---|---|
| MLflow experiment | 30 records: 11 `FINISHED`, 19 stale `RUNNING`; six stale runs have partial metrics | lifecycle metadata is incomplete and stale |
| Completed systems | Flat/case2000 depth-16 and depth-32 variants only | no treatment comparison exists |
| Recorded finished elapsed time | 238.675 one-GPU hours across 11 records | legacy spend only; outside the new 230-hour confirmatory ledger |
| Numerical quality | multiple depth-32 `FINISHED` endpoints diverged; other runs produced ground-truth test CSVs | `FINISHED` cannot be used as a pass verdict |
| Launch integrity | 40 first-wave logs failed on `Invalid experiment ID: '.stfolder'` | MLflow must use a child store below the Syncthing root |
| Additional failures | cancellations, host OOM kills, and Triton resource errors are present | resource failures need pre-registered fail-closed records |
| Provenance | executed SHA absent; fitted per-grid normalizer artifacts present | incompatible with the final zero-shot contract |

No legacy metric is used to choose final geometry, loss, capacity, checkpoints, or targets. The immediate bridge scope is CPU-only I001/R001/I002 work and an output-store preflight. This host has no CUDA or MPS backend. The shared environment imports datakit from nested worktree `7c9b93a`; I001 requires the requested sibling checkout at `1640668`, and that sibling commit is not yet reachable from `origin/main`. The saved provenance record correctly remains BLOCKED until the editable link is repointed and both repositories are clean and reproducible from their recorded remote refs.

Bridge verification: 9 focused preflight tests and all pre-commit hooks pass. The full unit suite passes 83/83 when run outside the restricted sandbox; the sandbox-only failures were denied PyTorch shared-memory and Lightning localhost-socket operations.

## Historical M0 prototype record

> **Legacy prototype evidence.** These results establish wiring and rough cost for the removed v2 plan. They do not satisfy the final two-map, strict-validity, boundary-projection, objective, or launch contracts. The current plan and tracker govern all new work.

**Date**: 2026-07-07
**Original plan**: superseded v2 experiment plan, removed during canonical cleanup
**Historical scope**: M0 research-gate runs R001–R004 + engineering-track items E001–E003 (all old M0 MUST items DONE; E004/E005 were deferred).

## Environment note (resolved during the session)

The local disk hit **ENOSPC twice** mid-session (killed a 512-scenario
generation; later blocked shell commands). Initial M0 gates were met at
reduced scale + from `.m` files. After the user freed space (~53 GB), R001
was regenerated at full local scale and R002 recomputed on it — the tables
below reflect the final full-scale state. Julia depot note: deleting
`/tmp/julia_depot` under disk pressure required a juliapkg force-resolve to
rebuild (`JULIA_DEPOT_PATH=/tmp/julia_depot` is set in `~/.zshrc`, so the
depot does not survive reboots either). **Cluster-scale generation
(10k/grid) is still required before M1** — GPU milestones run there.

## Results by run

### R001 — data generation (DONE at local scale)

- datakit configs: `experiments/m0/datakit_configs/*.yaml` — perturbations
  (topology/generation/admittance) **off** so Y is fixed per grid, matching
  the frozen per-grid-operator design; mode=pf, seed=0.
- Generated (requested → solved scenarios): case14/30/57 2048→2048,
  case118 1024→1024, case500_goc 512→512, **case2000_goc 256→64**,
  Texas2k 256→254. ~214 MB in gitignored `data/`. Verified: Y identical
  across scenarios; ids contiguous after drops; `load_scenario_idx` present.
- **case2000_goc PF yield is 25%**: the fast Newton PF diverges on most
  stressed load scalings (datakit drops + renumbers them). For cluster-scale
  generation either request ~4× the target, soften the load-scaling range
  (Texas2k's datakit config uses start 0.8 / step 0.05), or fall back to
  `pf_fast: false` (Ipopt, slower).
- pglib case9241_pegase `.m` downloaded (2.5 MB, corrected variant).

### R002 — HierarchyTransform precompute (DONE, ALL GATES PASS)

`gridfm_graphkit/datasets/hierarchy.py` (new): boundary selection
(REF ∪ PV ∪ HV-kV closest to 27%, gen-only fallback for constant-kV grids),
Schur `Y_red` (|Y|>1e-3 directed edges), prolongation `P = −Y_ii⁻¹Y_ib`
top-k(16→32) row-sparsified with the **hard assert ≥95% retained |P| mass**,
Ward `W_int` (provenance), per-scenario `V_aff = Y_ii⁻¹ conj(S_I)` + coarse
injections (PF-input masked convention). Cache per grid under `processed/`;
`AddHierarchy` transform attaches cbus node type + seeds/connects/prolong
relations at runtime (PyG hetero collate handles batching; verified in a
4-graph batch). Full numbers: `experiments/m0/results/r002_precompute.json`.

| Grid | N | boundary | P mass kept (k) | Y_red offdiag density | op time |
|---|---|---|---|---|---|
| case14 | 14 | 5 (36%) | 1.000 (5) | 100% (N_c=5) | 5 ms |
| case30 | 30 | 6 (20%) | 1.000 (6) | 100% (N_c=6) | 1 ms |
| case57 | 57 | 7 (12%) | 1.000 (7) | 81% (N_c=7) | 1 ms |
| case118 | 118 | 54 (46%) | 1.000 (16) | 11.0% | 3 ms |
| case500_goc | 500 | 113 (23%) | 0.991 (32) | 52.7% | 13 ms |
| case2000_goc | 2000 | 348 (17%) | 0.966 (16) | 6.6% | 85 ms |
| Texas2k | 2000 | 499 (25%) | 0.985 (32) | ~6% | 150 ms |
| case9241_pegase | 9241 | 2904 (31%) | 0.994 (16) | 0.39% | 3.7 s |

- **Hard assert PASS on all 8 grids** (worst 96.6%, case2000_goc @ k=16).
- Precompute ≤3.9 s/grid operators vs the <10 CPU-min gate; per-scenario
  V_aff + coarse injections cost ≤0.7 s per grid for the FULL scenario set
  (e.g. 2048 scenarios of case14 in 0.65 s — amortized ~0.3 ms/scenario).
  Thresholded Y_red keeps ≥99.98% of off-diagonal admittance mass.
- Harmonic-extension diagnostic (true V_B + sparsified P, vs true AC
  interior voltages), median |V̂−V| p.u.: case14 0.0076, case30 0.0152,
  case57 0.0086, case118 0.0020, case500 0.0185, case2000 0.0309,
  Texas2k 0.0164 — the linear iterate degrades with size/stress as expected
  but stays a useful prior everywhere; the model consumes it through a
  residual stream, so it only needs to be informative, not exact.
- Caveats: case118's boundary is 46% (its 54 gen buses alone exceed the
  25–30% target — expected for generator-dense grids); case500's coarse
  density 52.7% is benign in absolute terms (N_c=113 → ~6.6k edges);
  case9241 remains operators-only (no local scenario data).

### Cross-model code review (GPT-5.x via codex MCP, read-only)

0 CRITICAL. Findings, all verified and fixed before the final gate run:
- MAJOR: Y_red diagonal was cached but never consumed → now appended to
  `cbus_x` (self-admittance as coarse node features, `input_cbus_dim` 6→8);
  still cached for the post-gate coarse-PBE ablation.
- MAJOR (latent): boundary Pg used `is_slack_gen==0` while the PF mask
  masks ALL gens at REF buses → now excludes every generator on a REF bus.
  Verified it never fired on current grids (each REF bus hosts exactly the
  slack gen), so no result was contaminated.
- MINOR: coarse edge orientation flipped to j→i carrying Y_ij (matters only
  under non-symmetric Y_red from phase shifters).
Signs/conjugates of Ward/Schur/harmonic extension, no-leakage of masked
quantities, unit/normalization consistency, ground-truth-only evaluation,
and hetero-batching indexing were all confirmed OK by the review.

### R003 — pre-registered FLOP accounting (DONE, committed before M1)

`gridfm_graphkit/utils/flops.py` (new): analytic per-relation forward FLOPs
for `GNS_heterogeneous` and `GNS_hetero_hier`, formulas traced from the
installed PyG `TransformerConv` (query/key/value/edge/skip/beta) and the
model code (per-layer decode heads + physics loop included).
`matched_flat_depth()` defines the M1 matched-FLOP pairs (target ratio
[0.9, 1.1]). Driver: `experiments/m0/r003_flops.py`; table:
`experiments/m0/results/r003_flops.json`.

Reference point (hidden 48, heads 8; KS split L_f/L_c/L_f' = 4/8/4):
KS @ case2000_goc = 46.1 GFLOP/sample ≈ 0.84× flat depth-8 (254.6 GFLOP at
depth 48). No swept flat depth lands inside [0.9,1.1] of this particular KS
config — M1 closes the pairing by tuning the KS depth split/width (both
reported), exactly the plan's fallback.

### R004 — KS-2-level model + case14 overfit (DONE, gate PASS)

`gridfm_graphkit/models/gnn_hetero_hier.py` (new, registered
`GNS_hetero_hier`): fine HGNS stack → latent restriction (seeds relation +
`MLP_in` with Ward-restricted physical features) → coarse HGNS stack on
thresholded Y_red → coarse decoder D_c → physical prolongation
`V̂ = v_aff + P_sp·V_B` (complex 2-channel scatter) + latent merge `MLP_out`
→ fine HGNS stack → baseline-identical decode. New loss `CoarseVoltageMSE`
(λ_v boundary-voltage supervision, restricted fine labels via seeds).

Overfit runs (CPU, hidden 32, heads 4, 2/2/2 layers, MaskedBusMSE +
0.2·CoarseVoltageMSE):

| Run | train scen. | epochs | weights | train RMSE VM (p.u.) | VA (rad) |
|---|---|---|---|---|---|
| KS, Lightning, 58-scen (pre-fix) | 58 | 1500 | best-val | 1.05e-3 | 2.68e-3 |
| KS, Lightning, 16-scen (pre-fix) | 12 | 3000 | best-val | 1.29e-3 | 3.15e-3 |
| **KS, direct (post-fix)** | 58 | 3000 | **final** | **1.22e-3** | **3.35e-3** |
| flat GNS_heterogeneous, direct, same recipe | 58 | 3000 | final | 2.00e-3 | 4.18e-3 |

Interpretation. All runs floor near 1e-3 at this budget; the direct
final-weights run refutes the hypothesis that best-val checkpoint selection
was the limiter. The identical-recipe **flat baseline floors HIGHER**
(2.0e-3/4.2e-3 vs 1.22e-3/3.35e-3), so ~1e-3 is the recipe/data floor for
this budget (hidden 32, 3000 epochs, cosine to 1e-5; losses still slowly
descending at cutoff), not a KS wiring defect — the hierarchical path
trains end-to-end and fits *better* than the established baseline on the
same data. Wiring gate PASS. Literal solver accuracy (1e-6) was not
reached and is not required for M0; capacity/schedule scaling is M1 work.
Scripts: `experiments/m0/r004_overfit_direct.py` (final-weights trainer),
`r004_eval.py` (checkpoint eval); JSONs in `experiments/m0/results/`.
Provenance: these runs used the interim 64-scenario case14 dataset, later
replaced by the 2048-scenario regeneration (same datakit config except
`load.scenarios`; seed 0) — rerunning on the new data would only tighten
the fit (more data, same recipe).

### E001 — torch.profiler baseline (DONE, CPU)

`experiments/m0/e001_profile.py`; JSON in `experiments/m0/results/`.
Flat GNS_heterogeneous (hidden 48, heads 8, 12 layers), default PF losses,
10 profiled steps after warmup. CPU-only — GPU rankings will differ; the
cluster reruns this before any GPU-targeted E-item lands.

| Grid (batch) | train s/step | infer s/step | top self-CPU train ops |
|---|---|---|---|
| case14 (32) | 0.43 | 0.11 | mm > addmm > scatter_add_ > mul |
| case118 (16) | 0.85 | 0.20 | mm > scatter_add_ > addmm > mul |
| case500 (8) | 1.51 | 0.34 | mm > scatter_add_ > mul > addmm |
| case2000 (4) | 3.93 | 0.66 | **mul > mm** > scatter_add_ > addmm |

Read-out for the E-track: GEMM dominates at small N; `scatter_add_` is a
solid #2 from case118 up (E002 migration targets a real cost); at case2000
elementwise `mul` overtakes GEMM (attention message ops — E006's
attention-free layer targets exactly this). E002–E004 remain TODO.

### E002 — native scatter migration (DONE, plan E1)

torch_scatter / torch_sparse fully removed from the package: 12 files now
import from `gridfm_graphkit/utils/scatter.py` (native `index_add_` /
`scatter_reduce_` / `sparse_coo.coalesce` implementations of the used
subset, dim=0 only, fail-loud otherwise). CI's wheel-install step, the
CLAUDE.md/README/docs install special-cases, and the tasks/utils.py MPS
CPU-detour are deleted.

Gates:
- **Full suite 49/49 green** (42 baseline + 7 new parity tests in
  `tests/test_native_scatter.py`, which compare against
  torch_scatter/torch_sparse where installed and keep fixed-value
  assertions once the wheels are gone).
- **Op-level parity + timing**: exact value parity on random inputs
  (incl. empty segments, out= accumulation, hybrid sparse coalesce);
  micro-benchmark at model shapes: native = 0.95–0.99× torch_scatter
  (marginally faster). Whole-model 10-step A/B deltas were within machine
  noise (−15%…+20%) — CPU wall-clock is neutral, as expected; the wins are
  structural (no version-pinned wheel, no custom-op compile graph breaks,
  correct MPS dispatch). GPU A/B at case500 deferred to the cluster.
- `scatter_max` returns `(values, None)` — no call site consumes the
  argmax, and computing it requires an int64 `scatter_reduce` that MPS
  does not support (the same dispatch gap the deleted workaround papered
  over).
- **Pre-existing flaw found & fixed**: `torch_sparse.SparseTensor` is
  internally inconsistent on multigraphs — `to_dense()` overwrites
  parallel edges while `sum(dim=1)` adds them, so the old RRWP transition
  matrix had rows summing ≠ 1 on grids with parallel branches. The native
  path accumulates consistently (deg ≡ row-sum); RRWP outputs change
  (correctly) on such grids. Covered by an explicit multigraph test.

### E003 — mmap consolidated store A/B (DONE, plan E2)

The plan's store already exists in the repo as ``HeteroGridDatasetMMap``
(``data.consolidated: true``, one ``consolidated.pt`` per grid served via
``torch.load(mmap=True)``); this run is its validation gate
(`experiments/m0/e003_mmap_ab.py`, JSON in `experiments/m0/results/`):

| Grid | samples | byte-identical | samples/s disk→mmap (w=0) | (w=4) |
|---|---|---|---|---|
| case118 (1024 scen) | all | **yes** | 778 → 1752 (**2.25×**) | 2728 → 4807 (1.76×) |
| case2000 (64 scen) | all | **yes** | 616 → 1202 (**1.95×**) | 1765 → 2708 (1.54×) |

Named deviations from the plan text (both justified, not fixed): one file
per grid rather than per (grid, split) — split handling lives in Subset
indices, a per-split file would add nothing; ``get()`` clones the mmap
slice rather than zero-copy — required because the normalizer mutates
tensors in place. Recommendation: use ``data.consolidated: true`` for all
M1+ runs.

### R006 — P reconstruction gate + k escalation (DONE local, Amendment A1)

`experiments/m0/r006_p_recon_gate.py`; gate now lives in
`build_operators` (`RECON_ASSERT = 1e-3` p.u. median over ≤64 true-V_b
scenarios, k ladder 16→32→64→dense). All local caches rebuilt through it:

| Grid | k before → after | recon med @k16 | recon med @chosen k | nnz_prolong |
|---|---|---|---|---|
| case14/30/57/118 | unchanged (≤16) | 0 (P effectively dense/blockwise) | 0 | unchanged |
| case500_goc | 32 → **64** | 4.42e-2 | **2.23e-4** | 12 497 → 24 881 |
| case2000_goc | 16 → **64** | 2.48e-2 | **3.95e-5** | 26 780 → 106 076 |
| Texas2k | 32 → **64** | 6.62e-2 | **6.90e-4** | 48 531 → 96 563 |

Dense P was never needed — k=64 clears the gate everywhere. As
pre-registered, the affine unpool error did NOT move (case2000
3.09e-2 → 3.03e-2; linearization dominates): R006 buys falsifiability
for R020/R021, not metrics. Consequences handled: R003/R010 regenerated
(KS@2000 46.06 → 46.24 GFLOP, iso widths unchanged, all ratios still in
[0.9, 1.1]). Follow-up fork (named, not taken): the latent merge now
scatters over the same enlarged P (≈4× nnz at case2000, ~2 fine-layers'
FLOPs); a physical/latent k split would spend those FLOPs on fine capacity
instead — decide only if M2 iso-matching gets tight. case9241: mass-only
(no local scenarios), cluster re-run required.

### R007 — case2000 outlier triage (DONE, Amendment A1)

`experiments/m0/r007_outlier_triage.py` → `results/r007_outlier_triage.json`.
The ≈1.07 p.u. outlier is **bus 1261** — leaf (degree 1), 13.8 kV, zero
load, gen-free — whose true solution is V≈0 in exactly 32/64 scenarios
(only ever-dead bus in the grid; alive at ~1.05 p.u. otherwise). The
suspected cause is REFUTED: the sample IS converged in power coordinates
(mismatch ~1e-9), but the power residual is **blind at V=0** (S = V·I* ≡ 0);
the KCL current residual there is 0.246 p.u. → the fast-PF solver landed on
the degenerate V=0 branch of a zero-injection leaf. Data artifact, not
model error. M1 data prep action: flag scenarios where any zero-load bus
has |V| < 0.1 p.u. Gate metrics stay median/p90 (pre-registered).

### R021 — HELM2 unpool implementation + wiring smoke (local part DONE)

Cluster arm stays TODO (M2-adjacent, gated on R006 — now satisfied).
Implemented: `model.unpool: helm2` in `GNS_hetero_hier`
(`c₀+c₁+c₂`, batched dense-LU solves; static per-grid operators in a
separate `helm_runtime_*.pt` written by `build_grid_hierarchy`, loaded in
the model's own process keyed by a per-sample path string — worker/DDP
safe; divergence canary median |c₂|/|c₁| in `model.helm_canary`; |c₀|≥0.5
floor on the reciprocal-series weights so an untrained coarse decoder
cannot blow up gradients). FLOP term in `utils/flops.py`
(≈24·n_int² + 40·n_int; ≪1% of forward at h=48). Unit tests:
`tests/test_helm_unpool.py` (reference recurrence, batched=per-sample,
backward finite incl. degenerate c₀).

Wiring smoke `experiments/m0/r021_wiring_smoke.py` (case14):
- **Phase A (operator ceiling, TRUE V_b through the model's runtime
  path):** affine 7.48e-3 → helm2 8.12e-5 median (**92×**, canary 0.042)
  — reproduces the pilot (8.8e-3 → 1.1e-4, ~80×): the model computes the
  same series the pilot validated.
- **Phase B (300-epoch train smoke, helm2 vs affine arms, PASS):**
  losses finite for both; canary median 0.044, max 0.186 (convergent);
  train RMSE helm2 VM 2.34e-3 / VA 8.43e-3 vs affine 3.42e-3 / 1.12e-2.
  PASS criteria are plumbing-level, NOT performance — case14 is too small
  and this is a single seed (pre-registered). One observation worth
  carrying to M2, stated without weight: the R005 load-tercile VA curve is
  FLAT for helm2 (9.6e-3 → 7.7e-3 → 7.7e-3) and RISING for affine
  (6.5e-3 → 8.9e-3 → 1.6e-2) — the exact discriminating signature R021
  pre-registers (kill criterion (c) is its absence at scale). Suggestive,
  not evidence; the cluster arm decides. `results/r021_wiring_smoke.json`.

## M0 decision gate

| Gate | Status |
|---|---|
| P retained mass ≥95% @ k 16→32, all grids | **PASS** (worst 96.6%) |
| Precompute <10 CPU-min/grid | **PASS** (≤3.9 s) |
| FLOP script pre-registered before M1 | **PASS** (committed) |
| case14 KS overfit VM/VA RMSE → ~0 | **PASS** (1.22e-3 p.u. / 3.35e-3 rad final weights; beats identical-recipe flat baseline at 2.0e-3/4.2e-3 — floor is the recipe, not the wiring) |
| Unit suite green after integration | **PASS** (42/42 post-fix; pre-commit clean) |

## Summary

- 4/4 M0 research-gate runs + E001 completed; **all hard gates pass** — the
  kill-criterion (P fill-in) is retired on all 8 grids including case9241.
- Main risk carried forward: none new at M0. The M2 falsifier (matched-FLOP
  deep flat ties at case2000) remains the declared kill risk. New operational
  risk noted: case2000_goc fast-PF scenario yield is 25% — plan cluster
  generation volume accordingly.
- **Deferred**: cluster-scale R001 (10k/grid); E004 (fused AdamW — GPU-only
  one-liner) and E005 (M1, lands with the size-balanced sampler); GPU
  reruns of E001/E002 A/Bs on the cluster.

## Next step

→ M1 (R010 flat depth sweep + R011 GRIT memory curve) on the cluster via
`/run-experiment`, after full-scale data generation there (use
`data.consolidated: true` per E003). On the cluster, re-run
`r006_p_recon_gate.py` (or `build_grid_hierarchy`) on the freshly generated
data before anything consumes the caches — the R006 gate and the HELM
runtime files are built there — and apply the R007 filter (drop scenarios
where a zero-load bus has |V| < 0.1 p.u.) at data prep.
