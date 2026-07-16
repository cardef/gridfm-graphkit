# Refinement Report

## Task

Review the actual final proposal, make the implementation as general and simple as possible, center the study on scalability for foundational grid models, use AGY/Antigravity as the external reviewer, make the `cardef/gridfm-graphkit` fork explicit, run a novelty check, and remove intermediate proposal/plan artifacts after publication.

## Result

The proposal changed from a single-grid, PF-specific boundary reconstruction method to a controlled study of electrical hierarchy as a communication mechanism in one parameter-shared multi-topology model. The task-specific HELM/affine reconstruction path was removed from the primary method. The final proposal passed AGY review at 9.10/10 with verdict READY and no drift.

## Principal Refinements

- Made the `cardef/gridfm-graphkit` fork of upstream `gridfm/gridfm-graphkit` the explicit implementation and reproducibility boundary.
- Reframed the primary question around source-topology diversity, held-out graph size, and cumulative compute.
- Isolated communication geometry by fixing the encoder, decoder, objective, data, and training recipe.
- Added a GridSFM-style linear-cost global baseline and a same-partition quotient hierarchy control.
- Replaced the under-specified restriction with a conservative mass-adjoint real-latent pair.
- Removed all per-scenario hierarchy quantities and physical reconstruction from the confirmatory path.
- Required case-declared `baseMVA`, source-frozen optional standardization, and fail-closed target-label isolation.
- Characterized Yaniv and Beck precisely and positioned GraphFM, Graph Mamba, GraphGPS, DPOT, and Poseidon without adding mismatched baselines.
- Defined topology-level, provenance-balanced pass criteria and explicit failure denominators.
- Froze a 20-run mandatory matrix behind a measured 230-GPU-hour budget gate.

## Reviewer Provenance

- Transport: AGY / Antigravity
- Model: Claude Opus 4.6 (Thinking)
- Conversation: `a9bc4fca-57a8-4c6c-b0bd-21689586fcde`
- Calibration: none
- Rounds: 3
- Scores: 7.85 REVISE → 8.70 REVISE → 9.10 READY
- Reviewer operated read-only; repository edits were performed separately.

## Full Review Record


### Round 1

# Round 1 Review

- **Reviewer transport**: AGY / Antigravity
- **Reviewer model**: Claude Opus 4.6 (Thinking)
- **Calibration**: none
- **Weighted score**: 7.85 / 10
- **Verdict**: REVISE
- **Drift warning**: NONE

<details>
<summary>Full raw AGY response</summary>

I have now read all five required files. Let me produce the formal review.

---

# Round-1 Review — Electrical Hierarchies as a Scaling Mechanism for Grid Foundation Models

**CALIBRATION: none**

---

## Load-Bearing Questions

### LB-1. Does PF-only multi-topology pretraining and zero/few-shot topology transfer justify the scoped term "domain-specific grid foundation model"?

**Yes, conditionally.** The operational definition (§ Operational definition) explicitly limits the term to the domain-specific sense of LUMINA: one model pretrained across configurations, evaluated zero-shot or adapted to unseen systems. It does not claim internet-scale pretraining or billions of parameters. This is honest and consistent with current GridSFM and LUMINA-Bench usage. The proposal is careful to anchor the term in prior precedent rather than inflate it. The single-task (PF-only) scope is a limitation, not a misuse, because the hierarchy mechanism is deliberately tested as task-agnostic communication. I find the scoping adequate provided the paper reproduces the operational definition verbatim and never uses "foundation model" without the qualifier.

### LB-2. Does the common METIS partition plus Kron-vs-quotient geometry isolate an electrical-operator effect cleanly?

**Mostly yes, with one residual confounder.** Using the same METIS partition, the same `HierarchyAdapter` code, the same coarse-node count, and matching cross-level/coarse nonzeros is the correct ablation structure. The only treatment difference is harmonic-transport + Schur edges vs. cell-assignment + weighted quotient graph. This is tight.

However, a subtle confound remains: the Kron prolongation entries are dense complex coefficients reflecting circuit-theoretic coupling, while the quotient assignment is binary/normalized. Even if total nonzeros are matched, the *information content per edge* differs. A truly clean isolation would also report (a) the effective rank of the prolongation operator for both arms and (b) the coarse-graph spectral gap. These are diagnostic, not design changes, and their absence is **MINOR** rather than blocking.

### LB-3. Do the three axes—source base-topology count, cumulative training FLOPs, and unseen grid size—support a defensible scalability claim without model-size scaling?

**Yes, with a stated limitation.** The proposal is explicit that it holds model size fixed and does not pursue a universal scaling law over model size, dataset size, and compute (§ Non-goals). The three axes (topology diversity G8/G16/G32, FLOP checkpoints C/4/C/2/C, held-out grid size 0.5k–13.7k) are the correct dimensions for the narrower claim of compute–transfer scaling. The limitation is real: without model-size variation, the result cannot be called a scaling law in the Chinchilla sense. The proposal acknowledges this. Acceptable.

### LB-4. Is exact dense Schur/prolongation construction followed by sparse runtime geometry a fatal scalability bottleneck at 13.7k buses once host memory and build cost are counted?

**Not fatal, but the proposal under-specifies the risk mitigation.** The dense intermediate is `solve(Y_II, Y_IB)` where `Y_II` is `(N - |B|) × (N - |B|)`. At 13.7k buses with ~27% boundary, the interior is ~10k buses. `spla.splu(Y_II)` is sparse LU; the dense `P = -lu.solve(Y_IB.toarray())` is `10k × 3.7k ≈ 37M complex entries ≈ 600 MB` in complex128. This fits in host RAM but is not trivial. The proposal mentions reporting "dense-intermediate peak host memory, factorization time, full geometry-build wall time" (§ Compute and Timeline Estimate), which is the right diagnostic. It also says topology-only residual and conditioning gates can reject a topology.

The current code ([hierarchy.py](file:///Users/carmine/Code/FM/gridfm-graphkit/gridfm_graphkit/datasets/hierarchy.py#L137)) computes the full dense `P` and then sparsifies via top-k per row. This is a one-time offline cost, amortized over scenarios. As long as the paper honestly reports this cost and acknowledges the regime boundary, it is not fatal. The proposal does so. **Adequate.**

### LB-5. Does deleting the HELM task head make the implementation materially more general and causal, or remove too much method novelty?

**It makes the study sharper and more general.** This is the single best design decision in the proposal. The current code ([gnn_hetero_hier.py](file:///Users/carmine/Code/FM/gridfm-graphkit/gridfm_graphkit/models/gnn_hetero_hier.py)) retains the HELM2 unpool as a configurable option (`model.unpool: helm2`), with a per-scenario boundary-conditioned analytic reconstruction. If this were the *only* unpool and the Kron model had no other hierarchy arm, a positive result could not distinguish "electrical hierarchy helps" from "analytic PF head helps." By using the affine unpool as the primary path and fixing the direct PF decoder across all arms, the proposal attributes any improvement to the communication mechanism alone. The HELM2 variant remains available as a later task adapter. Removing it from the primary path is a net gain in causal clarity and does not reduce architectural novelty, because the contribution is the controlled comparison, not any single module.

### LB-6. Is the 20-run campaign logically sufficient and realistically compatible with 230 3090-class GPU-hours?

**Logically sufficient, arithmetically tight.** 18 scaling-matrix runs (3 backbones × 3 G-levels × 2 seeds) plus 2 quotient controls cover the primary claim and the electrical-vs-generic ablation. Two seeds is the statistical minimum; three would be better but the budget is hard. The proposal wisely demands treatment-blind throughput profiling first and a preregistered FLOP budget `C`. The budget rule ("if projected upper bound exceeds 230 hours, the study is blocked") is honest.

The risk: TransformerConv with multi-head attention on heterogeneous graphs is expensive. On G32 with 32 topologies, same-topology batching means small batches for large grids, low GPU utilization, and potentially ~4–6 GPU-hours per run. At 20 runs that is 80–120 hours before adaptation sweeps. Adaptation (3 baselines × 2 grids × 3 k-shots) could add another 30–50 hours. The 230-hour ceiling has ~30–60 hours of slack, which is thin. The proposal's 20% campaign reserve helps, but one throughput surprise (e.g., PyG HeteroConv overhead on cbus relations) could force cutting the adaptation block. This is not a blocking issue but a real risk the authors should monitor.

### LB-7. Is the explicit `cardef/gridfm-graphkit` fork boundary narrow, reproducible, and clean enough to avoid turning the method into an unmergeable research-only code path?

**Narrow and well-defined in the proposal; not yet clean in the code.** The proposal specifies a tight boundary: reuse upstream task schemas, registries, data module, losses, trainer, and flat GNS behavior; add `CommunicationCore`, geometry builders, content-addressed registry, portable normalization, and global-summary baseline behind new registry entries. This is a clean additive boundary.

However, looking at the current code:

- [hierarchy.py](file:///Users/carmine/Code/FM/gridfm-graphkit/gridfm_graphkit/datasets/hierarchy.py) uses `select_boundary` based on REF/PV + HV-kV threshold, which the proposal explicitly deletes in favor of topology-only METIS. The existing boundary selection is a prototype artifact that must be replaced.
- [gnn_hetero_hier.py](file:///Users/carmine/Code/FM/gridfm-graphkit/gridfm_graphkit/models/gnn_hetero_hier.py#L117-L118) hard-codes `task != "PowerFlow"` rejection, which contradicts the proposal's claim that the adapter is task-agnostic.
- The normalizer ([normalizers.py](file:///Users/carmine/Code/FM/gridfm-graphkit/gridfm_graphkit/datasets/normalizers.py#L169)) computes `baseMVA` as the 95th percentile of per-grid (or per-sample) non-zero power values. This is a fitted statistic, not a declared `baseMVA` per-unit conversion. The proposal's portable normalization contract (§ Portable Normalization Contract) requires declared case `baseMVA` per-unit conversion, not a percentile. The current normalizer violates this contract.

These are implementation gaps, not proposal flaws, but they mean the fork boundary is aspirational, not realized. The implementation acceptance path (F1–F5) correctly schedules these changes before the campaign.

---

## Dimension Scores

### 1. Problem Fidelity — 9/10

The problem anchor is precise: "does deterministic Kron–Schur geometry improve the compute–transfer scaling frontier of one parameter-shared grid foundation model." The non-goals list is unusually honest (explicitly excluding "first scalable grid FM," universal scaling laws, multi-task, OPF, dynamics, new Kron reductions). The five prototype invalidation contracts are correctly identified and scheduled for repair. The success condition is falsifiable and has explicit failure modes (§ Failure Modes).

Minor weakness: the operational definition of "foundation model" is appropriately scoped but could be even tighter by specifying the minimum number of distinct base topologies and minimum hold-out group count that would make the "foundation" qualifier meaningful (currently "at least twelve" held-out topologies, which is reasonable but not formally justified).

### 2. Method Specificity — 8/10

The method is concrete and implementable. The `ElectricalHierarchyAdapter` is fully specified: METIS partition → Kron–Schur operators → sparse restrict/prolong → coarse processor → merge. The minimal implementation contract (`HierarchyGeometry`, `GeometryBuilder`, `HierarchyAdapter`, `CommunicationCore` protocols) is clean. The sparsification rule is label-free with a frozen cost budget.

Weaknesses:
- **The `R_latent` / restriction operator specification is under-defined.** The proposal says "column-normalized transpose of the magnitude weights," but this is not the electrical transpose of the prolongation. Why column-normalize rather than use the adjoint of the prolongation weighted by the admittance? If the restriction is not electrically motivated, this is an arbitrary choice in a paper that claims electrical specificity. **IMPORTANT** — specify the restriction operator derivation fully and justify why column-normalized magnitude rather than the Galerkin transpose `R = P^H Y_II / diag(P^H Y_II P)` or the injection-weighted transpose.
- **The coarse processor reuses "the same block class but has its own weights."** How are the coarse-layer weights initialized? Random? Shared? Pretrained? This matters because the coarse graph has very different spectral properties from the fine graph, and TransformerConv's attention mechanism may not transfer well. **MINOR** — specify initialization.

### 3. Contribution Quality — 8/10

The dominant contribution (first controlled comparison of electrical hierarchy as FM scaling mechanism) is well-focused and singular. The proposal is disciplined in listing explicit non-contributions. The supporting contribution (reusable `ElectricalHierarchyAdapter`) is secondary and appropriately so.

Weaknesses:
- **The contribution is inherently a negative-risk contribution.** If the electrical hierarchy does not beat the quotient hierarchy, the paper claims "only that hierarchy helps" (§ Failure Modes), which is a weaker but still publishable result. If the flat baseline ties or wins, the paper is a well-designed negative result. Well-designed negatives are publishable at top venues, but the contribution quality ceiling depends on the empirical outcome, which cannot be scored in advance. I score what is controllable: the experimental design, which is strong.
- **The claim "we found no prior controlled comparison" is an absence-of-evidence statement (correctly acknowledged), but the Technical Gap section could be sharper.** Yaniv and Beck (2309.01124) already explore hierarchical GNNs for PF prediction. The proposal cites this but does not explain specifically how their hierarchy differs (is it electrical? METIS-based? Kron?). If they use a similar mechanism, the "no prior controlled comparison" claim weakens. **IMPORTANT** — explicitly state what Yaniv and Beck's hierarchy is and why it does not constitute the same comparison.

### 4. Frontier Leverage — 7/10

The proposal uses multi-topology pretraining with zero-/few-shot adaptation, which is the correct modern primitive for turning a large-grid specialist trick into a testable representation mechanism. The decision not to add LLM/VLM/diffusion/RL is appropriate and well-justified (§ Modern Primitive Usage).

Weaknesses:
- **The proposal does not engage with the most recent frontier primitive: learned tokenization/patching of graphs (e.g., GraphGPT, Graph-MAMBA, hierarchical graph tokenizers).** These methods tokenize graph regions into sequences and process them with sequence models, achieving linear-cost nonlocal communication without explicit Kron operators. The proposal should at least briefly argue why deterministic Kron operators are preferable to learned graph tokenization for this problem. The argument is likely strong (physics-preserving structure, zero learned overhead, exact operator guarantees), but it is not made. **IMPORTANT** — add a brief discussion of why deterministic electrical operators are preferred over learned graph tokenization/patching.
- The global-summary baseline uses GridSFM-style mean-plus-max pool, which is the established O(N) global mechanism. This is appropriate. However, the proposal does not discuss linear attention (e.g., Performer, or the more recent FourierFormer/HyperMixer variants), which could provide O(N) nonlocal communication that is richer than pooling but cheaper than full attention. A brief justification for excluding this class of baselines would strengthen the frontier positioning. **MINOR**.

### 5. Feasibility — 7/10

The 230 GPU-hour budget is tight but plausibly adequate (see LB-6). The implementation acceptance path (F0–F5) is well-structured and correctly sequences contracts → tests → profiling → freeze → campaign. The timeline estimate (8–10 weeks) is reasonable.

Weaknesses:
- **The gap between current code and proposal is large.** The existing [hierarchy.py](file:///Users/carmine/Code/FM/gridfm-graphkit/gridfm_graphkit/datasets/hierarchy.py) implements REF/PV boundary selection (which the proposal deletes), the reconstruction gate uses true boundary voltages (which the proposal forbids), and the HELM2 unpool is deeply integrated into the hierarchical model. Refactoring to METIS-based partitioning, label-blind construction, content-addressed registry, topology-portable normalization, a `CommunicationCore` interface, and the global-summary baseline is a significant engineering effort. Two weeks is optimistic for F0–F2 combined.
- **The fitted normalizer must be replaced.** The current `HeteroDataMVANormalizer` ([normalizers.py](file:///Users/carmine/Code/FM/gridfm-graphkit/gridfm_graphkit/datasets/normalizers.py#L169)) computes `baseMVA` as `np.percentile(non_zero_values, 95)`, which is fitted per grid. The `HeteroDataPerSampleMVANormalizer` computes per-scenario. Both violate the proposal's portable normalization contract. Replacing this with declared case `baseMVA` per-unit conversion requires understanding whether `baseMVA=100` (the hardcoded constant in [hierarchy.py](file:///Users/carmine/Code/FM/gridfm-graphkit/gridfm_graphkit/datasets/hierarchy.py#L43)) is universally correct across all PGLib cases. It usually is (MATPOWER convention), but some cases may differ. **IMPORTANT** — verify that all target cases use `baseMVA=100` or implement case-file-declared baseMVA lookup before the normalization contract is frozen.

### 6. Validation Focus — 8/10

The validation is claim-driven and minimal. Three claims, three experiments, explicit required evidence, explicit failure modes. The statistical design (topology-level metrics, provenance-group holdout, no per-target checkpoint selection) is strong. The FLOP-matched comparison rather than epoch-matched is the right choice.

Minor weakness: the paired margin between Kron-HGNS and Quotient-HGNS (Claim 2) is described as "frozen" but not specified. What is the margin? How was it derived? If it is set after seeing pilot results, it is not preregistered. **MINOR** — specify or preregister the paired margin.

### 7. Venue Readiness — 7/10

The proposal is well-structured for a NeurIPS/ICML submission. The technical gap analysis is unusually thorough. The contribution is focused. The experimental design is preregistered.

Weaknesses:
- **The paper does not exist yet.** The proposal is a plan, not a draft. The implementation acceptance path has six phases. At the current state, the code implements a different method (REF/PV boundary, HELM2 unpool, per-grid fitted normalization) than the proposal describes (METIS partition, affine unpool primary, portable normalization). No experiment has been run under the proposed protocol. A venue-ready assessment requires at least preliminary results under the correct protocol. **IMPORTANT** — this is inherent to the stage and not a proposal flaw; it limits the venue-readiness score.
- **The writing plan is absent.** No outline, no target page count, no figure plan. For a paper with this many moving parts, a figure plan (architecture diagram, scaling curves, ablation table layout) would reduce revision risk.

---

## Weighted Composite

| Dimension | Weight | Score | Weighted |
|:---|:---:|:---:|:---:|
| Problem Fidelity | 15% | 9 | 1.35 |
| Method Specificity | 25% | 8 | 2.00 |
| Contribution Quality | 25% | 8 | 2.00 |
| Frontier Leverage | 15% | 7 | 1.05 |
| Feasibility | 10% | 7 | 0.70 |
| Validation Focus | 5% | 8 | 0.40 |
| Venue Readiness | 5% | 7 | 0.35 |
| **Composite** | | | **7.85** |

### GAP

The proposal is at 7.85, not 10. What separates it:

1. **No empirical evidence yet.** The strongest proposals at this stage have pilot results under the proposed protocol. This one has pilot results under a *different* protocol (the prototype). Until at least one G8 run with METIS partition, portable normalization, and all four communication cores completes, the proposal is a well-designed plan, not a validated method.
2. **Frontier engagement is passive.** The proposal correctly avoids gratuitous LLM/VLM additions, but it does not affirmatively engage with the most relevant frontier ideas (learned graph tokenization, linear attention, graph foundation models beyond GridSFM). A 10 would position the electrical hierarchy as the *necessary* physics-aware alternative to these generic mechanisms, with a brief but pointed argument.
3. **The restriction operator is under-specified.** A 10 would derive the restriction from the same circuit-theoretic principles as the prolongation, creating a Galerkin-consistent coarse-fine pair rather than an ad-hoc column-normalized magnitude transpose.
4. **The code–proposal gap is real.** Every contract (normalization, partition, boundary selection, content-addressed registry, CommunicationCore interface) must be built before the campaign. A 10 would have at least the contracts and gates implemented, even without full experiment results.

---

## Fixes for Dimensions Below 7

All dimensions score ≥ 7. No mandatory fixes under this rule. Fixes below are for dimensions at exactly 7.

**Frontier Leverage (7/10)**
- **Weakness**: No engagement with learned graph tokenization or linear-attention alternatives for O(N) nonlocal communication.
- **Fix**: Add a subsection to § Modern Primitive Usage arguing why deterministic electrical operators are preferred over learned graph tokenization (physics-preserving, zero-learned-overhead, exact circuit-theoretic guarantees, no pretraining required for the operator itself). Cite at least GraphGPT or a representative graph tokenizer. One paragraph suffices.
- **Priority**: IMPORTANT

**Feasibility (7/10)**
- **Weakness**: Large code–proposal gap; fitted normalizer violates portable normalization contract; two-week timeline for F0–F2 is optimistic.
- **Fix**: Implement the portable normalization contract (declared `baseMVA` per-unit) and verify `baseMVA` values across all PGLib target cases before freezing the protocol. Extend the F0–F2 timeline to three weeks if needed.
- **Priority**: IMPORTANT

**Venue Readiness (7/10)**
- **Weakness**: No preliminary results under the proposed protocol; no paper outline or figure plan.
- **Fix**: Run one G8 pilot (flat + Kron, 1 seed, short FLOP budget) under the proposed protocol (METIS partition, portable normalization, affine unpool) and report whether the hierarchy produces any signal at all. Create a 6-figure outline: (1) architecture diagram, (2) scaling curves (error vs. FLOPs at G8/G16/G32), (3) Kron vs. quotient ablation, (4) grid-size extrapolation, (5) adaptation curves, (6) build-cost/memory table.
- **Priority**: IMPORTANT

---

## Simplification Opportunities

1. **Delete the HELM2 unpool from the primary codebase.** The proposal already removes it from the primary path, but [gnn_hetero_hier.py](file:///Users/carmine/Code/FM/gridfm-graphkit/gridfm_graphkit/models/gnn_hetero_hier.py) still implements it (lines 79–94, 235–294, 446–459) along with per-grid runtime file loading, dense LU factoring, and the divergence canary. This is ~120 lines of complexity that is excluded from the primary claim. Move it to a separate module or gate it behind a feature flag that is OFF by default and not exercised in the confirmatory campaign. This reduces the `GNS_hetero_hier` class to the core U-Net flow and makes the code match the proposal.

2. **Merge `HeteroDataMVANormalizer` and `HeteroDataPerSampleMVANormalizer` into one portable normalizer.** Both are being replaced by the proposal's declared-baseMVA contract. Instead of maintaining two fitted normalizers that will not be used in the confirmatory path, implement one `PortableMVANormalizer` that reads `baseMVA` from the case metadata and uses no fitted percentile. The existing normalizers remain for backward compatibility but are not registered for the experiment.

3. **Delete per-scenario `V_aff`, `cbus_x`, and Ward-restriction computation from the hierarchy builder.** The proposal's `ElectricalHierarchyAdapter` is topology-only; per-scenario quantities belong to the data pipeline, not the geometry builder. The current [hierarchy.py](file:///Users/carmine/Code/FM/gridfm-graphkit/gridfm_graphkit/datasets/hierarchy.py#L301-L349) computes per-scenario `v_aff` and `cbus_x` inside `build_grid_hierarchy`. These are per-scenario features that should be computed at sample construction time (in `AddHierarchy.forward` or the data module), not baked into the geometry cache. Splitting them reduces the geometry cache to static operators only, which is what the proposal's `HierarchyGeometry` dataclass describes.

## Modernization Opportunities

1. **Replace `TransformerConv` with a parameter-efficient message-passing variant.** The current flat and hierarchical models use `TransformerConv` with multi-head attention on all relations. This is expensive (quadratic in node degree per layer) and contributes to the tight GPU budget. A modern alternative is GPS++ (General, Powerful, Scalable graph transformer) or the simpler GatedGCN with virtual-node aggregation, which achieves comparable expressivity at lower cost. This would stretch the FLOP budget further and potentially allow three seeds instead of two. However, changing the backbone risks confounding the comparison with prior GridSFM results. **Recommend only if throughput profiling shows TransformerConv is the bottleneck.**

2. **Use torch.compile (PyTorch 2.x) for the fine and coarse message-passing blocks.** The proposal mentions "same-topology batches avoid dynamic-shape recompilation," which suggests awareness of compilation. `torch.compile` with `mode="reduce-overhead"` on the HeteroConv layers would reduce Python dispatch overhead and potentially improve throughput by 20–40% on 3090s. This is a free lunch that could turn the 230-hour budget from tight to comfortable.

3. **Replace the dense `spla.splu(Y_II) → lu.solve(Y_IB.toarray())` with a sparse direct solver that returns sparse factors and supports sparse RHS.** SuiteSparse's CHOLMOD (for symmetric positive-definite Y_II, which Y_II is *not* in general) or UMFPACK (for unsymmetric) can compute `P` without densifying `Y_IB`. However, `Y_II` is complex and generally unsymmetric, so CHOLMOD is not applicable. UMFPACK with sparse RHS would reduce peak memory from O(n_I × n_B) to O(nnz(P)), which matters at 13.7k. This is an implementation improvement, not a method change.

## Drift Warning

**NONE.** The proposal is tightly anchored to the problem statement. All suggested changes are within the proposal's stated scope. No suggestion would alter the problem anchor.

## Remaining Action Items

| Priority | Item |
|:---:|:---|
| CRITICAL | Implement topology-only METIS partition, replacing REF/PV + HV-kV boundary selection in [hierarchy.py](file:///Users/carmine/Code/FM/gridfm-graphkit/gridfm_graphkit/datasets/hierarchy.py#L91-L110). |
| CRITICAL | Implement portable normalization (declared case `baseMVA`) and verify all PGLib target cases use `baseMVA=100` or implement case-declared lookup. |
| CRITICAL | Remove label-dependent reconstruction gate (`Vb_samples` / `RECON_ASSERT`) from the geometry builder; replace with topology-only residual gate per proposal. |
| IMPORTANT | Specify the restriction operator (`R_latent`) derivation; justify column-normalized magnitude vs. Galerkin transpose. |
| IMPORTANT | Explicitly characterize Yaniv and Beck's hierarchy mechanism and explain why it does not constitute the same controlled comparison. |
| IMPORTANT | Add brief frontier positioning against learned graph tokenization. |
| IMPORTANT | Implement `CommunicationCore` protocol and `GlobalSummaryCore` baseline. |
| IMPORTANT | Split per-scenario features (`v_aff`, `cbus_x`) out of the geometry cache into the data pipeline. |
| IMPORTANT | Run one G8 pilot under the proposed protocol to validate signal before committing the full campaign budget. |
| MINOR | Specify coarse-layer weight initialization. |
| MINOR | Preregister or specify the Kron-vs-quotient paired margin for Claim 2. |
| MINOR | Create a 6-figure paper outline. |

## Verdict

**REVISE**

The weighted composite is 7.85, below the 9.0 threshold. There is no drift and the contribution is focused. The blocking issues are: (1) the code implements a materially different method than the proposal describes (REF/PV boundary, fitted normalization, label-dependent reconstruction gate), meaning the proposal's contracts are aspirational rather than realized; (2) the restriction operator is under-specified for a paper claiming electrical specificity; (3) frontier positioning is passive. None of these are architectural or conceptual problems — they are specification and implementation gaps that the proposal's own acceptance path (F0–F5) is designed to close. After the CRITICAL items are resolved, a second review should find the proposal closer to READY.

</details>


### Round 2

# Round 2 Review

- **Reviewer transport**: AGY / Antigravity
- **Reviewer model**: Claude Opus 4.6 (Thinking)
- **Calibration**: none
- **Weighted score**: 8.70 / 10
- **Verdict**: REVISE
- **Drift warning**: NONE

<details>
<summary>Full raw AGY response</summary>

Good. The implementation files are unchanged since round 1. Now I have all the materials to produce the round-2 review.

---

# Round-2 Review — Electrical Hierarchies as a Scaling Mechanism for Grid Foundation Models

**CALIBRATION: none**

---

## Load-Bearing Questions (Round 2)

### LB-1. Is the `cardef/gridfm-graphkit` fork now unmistakably part of the proposal, with a narrow, testable boundary from upstream and from the legacy prototype?

**Yes.** The revised proposal opens with a title-block scope declaration and adds a full § Explicit Fork Contract that specifies (a) what is reused from upstream, (b) the five additive fork-only module categories, (c) a confirmatory import test that force-fails legacy imports (`GNS_hetero_hier`, the fitted normalizer, the REF/PV hierarchy builder, HELM reconstruction), and (d) upstream-flat compatibility requirements including construction, tensor schemas, and checkpoint loading. The clean-clone, commit provenance, and environment lock requirements are also explicit.

The contract is testable: it defines what must break and what must work under the confirmatory entry point. This is a substantial improvement over round 1, where the fork was aspirational. The boundary is now narrow, stated, and falsifiable. **Resolved.**

### LB-2. Is the conservative real-latent pair `U = D_I^-1 A`, `R = D_B^-1 A^T` mathematically well-defined and honestly scoped?

**Yes.** The revised § Conservative Latent Transport derives `U` and `R` from the nonnegative magnitude matrix `A = |P_sparse|`, shows the constant-preservation identities `U 1_B = 1_I` and `R 1_I = 1_B`, and proves mass-adjointness `D_B R = U^T D_I = A^T`. The coverage guarantee from the sparsifier (every row and column of `A` has at least one entry) makes both diagonal inverses well-defined.

Critically, the proposal now explicitly states that this is *not* presented as a complex Galerkin projection: "AC `Y` is generally complex and non-Hermitian, and the learned hidden states are real." It channels electrical specificity through the harmonic-map support, the signed `Re/Im/magnitude/phase` edge attributes, and the Schur coarse graph rather than through the algebraic structure of the restriction. This is honest scoping.

One minor note: the mass-adjoint identity is stated but the proof sketch in the proposal text is compressed. Verifying it: `(D_B R)_{bj} = (D_B)_{bb} (D_B^{-1})_{bb} A^T_{bj} = A^T_{bj}` and `(U^T D_I)_{bj} = (D_I^{-1} A)^T_{bj} (D_I)_{jj} = A_{jb} (D_I)_{jj} / (D_I)_{jj} = A_{jb} = A^T_{bj}`. ✓. The algebra checks out. **Resolved.**

### LB-3. Does the common partition plus Kron-versus-quotient construction isolate the electrical operator without extra treatment drift?

**Yes, tighter than round 1.** The revision reiterates: same METIS partition, same adapter code, same coarse-node count, same parameter budget, same message primitive, same nonzero budget. The quotient arm uses a cell-assignment `A` and the ordinary weighted quotient graph. "The treatment therefore changes the transport coefficients and coarse operator, not the partition, U-Net shape, task head, or training recipe."

The round-1 residual confounder (information content per edge differing between dense harmonic coefficients and binary assignment) remains inherent to the comparison — it is the treatment itself, not a confound. The Kron arm's edges carry circuit-theoretic coupling; the quotient arm's edges carry adjacency weights. That is precisely what the ablation tests. The proposal does not claim to control for edge information density independently; it claims to test whether the *electrical* transport family outperforms the *generic* transport family. This is the correct scope. **Adequate.**

### LB-4. Is Yaniv and Beck now characterized precisely enough?

**Yes.** The revised Technical Gap now reads: "divide a distribution system into clusters, train one single-hidden-layer ANN per cluster, and arrange those ANNs as a tree whose upward routing follows electrical correlation. Their tests use IEEE-123 and EPRI Ckt5. This is not a parameter-shared GNN, does not derive coarse communication from a Kron–Schur map, does not pretrain across base topologies, and does not compare electrical, generic-hierarchy, local, and global mechanisms at matched compute."

This is specific enough to distinguish their method on four axes: (1) per-cluster ANNs vs. one shared GNN, (2) no Kron–Schur operators, (3) no multi-topology pretraining, (4) no matched mechanism controls. The absence-of-controlled-comparison claim is now well-supported. **Resolved.**

### LB-5. Is excluding learned graph tokenizers and linear attention from the mandatory matrix methodologically justified?

**Yes.** The new § Frontier Positioning Without Extra Treatments provides the argument I requested. Three representative methods are cited (GraphFM, Graph Mamba, GraphGPS). The exclusion rationale is precise: "each adds a tokenizer, ordering/positional design, or global mixer whose learned inductive bias would vary together with the electrical geometry," breaking causal attribution. For linear attention: "Kernel choice, positional encoding, and attention normalization constitute a second method axis."

The proposal correctly keeps GridSFM's released linear-cost summary as the domain-established global control and notes that a later robustness study may substitute a stronger linear mixer. This is methodologically sound. The mandatory matrix must test the causal question (does the electrical operator help?) and should not introduce additional learned mechanism axes. **Resolved.**

### LB-6. Are the gates sufficient for the scoped scalability question?

**Yes, with improved statistical specification.** The revision replaces the vague "frozen paired margin" with explicit criteria:

- Primary pass: upper 95% family-balanced paired interval for `d_e = log(error_Kron) - log(error_baseline)` below zero against *both* Flat and Global baselines at G32/C.
- Physical non-inferiority: `d_r` upper bound at most `log(1.05)`, explicitly labeled as a study-design margin.
- Pareto superiority at C and at least one earlier checkpoint.
- Non-contraction of advantage with target size (slope ≤ 0).
- Non-reversal of advantage from G8 to G32.
- Zero superiority margin for Kron vs. Quotient (no arbitrary effect threshold).

The statistical unit is topology, not scenario; provenance groups receive equal weight. Hierarchical bootstrap is preregistered. Construction failures stay in the denominator.

The 230 GPU-hour budget analysis from round 1 still applies. The revision extends the timeline to 10–12 weeks (from 8–10), with three weeks for F0–F3, which addresses the prior feasibility concern. The 20% campaign reserve is maintained. **Adequate.**

### LB-7. Does the proposal distinguish implementation readiness from publication readiness?

**Yes, and this is one of the strongest improvements.** The final paragraph of the Evidence Package section states: "The paper is ready for submission only after these artifacts exist. This proposal is ready for implementation when its contracts survive review; those are different thresholds." The response table also explicitly defers the G8 pilot to after F0–F4: "A pilot on the current prototype would test the wrong method. Proposal readiness means ready to implement; publication readiness still requires the preregistered pilot and campaign."

This is exactly the right distinction. The round-1 review dinged venue readiness for not having preliminary results; the round-2 request clarifies that the review stance should not make preliminary results a blocking methodological requirement when the proposal schedules them after integrity contracts. I adjust accordingly. **Clear.**

---

## Dimension Scores

### 1. Problem Fidelity — 9/10

Unchanged from round 1. The problem anchor, non-goals, success condition, and failure modes are precise. The operational definition of "foundation model" is properly scoped. The explicit statement that three diversity levels do not constitute a universal scaling law (new in the revision) further sharpens the claim boundary.

### 2. Method Specificity — 9/10

**Up from 8.** The two round-1 weaknesses are resolved:

- The restriction operator is now fully specified with the conservative mass-adjoint pair `(U, R)`, proven constant-preserving and mass-adjoint, and explicitly *not* presented as a Galerkin projection. The electrical specificity is channeled through the correct pathway (harmonic-map support, signed complex attributes, Schur graph). This is honest and complete.
- Coarse-layer initialization is now specified: "same initializer family and deterministic seed mapping as fine blocks, but independent weights and no cross-level weight sharing."

The sparsifier now guarantees both row and column coverage ("the strongest entry for every coarse column"), which makes `D_I` and `D_B` well-defined by construction rather than by assumption.

Remaining minor weakness: the proposal specifies `phi_down` and `phi_up` as "adapter-specific trainable modules" but does not state their width, depth, or activation function. Two-layer MLPs with residual normalization were stated in round 0 but not restated in the revision. This is recoverable from the minimal software contract and the round-0 text, but the revision should be self-contained. **MINOR.**

### 3. Contribution Quality — 9/10

**Up from 8.** The two round-1 weaknesses are resolved:

- Yaniv and Beck are now characterized on four specific axes (per-cluster ANNs, no Kron–Schur, no multi-topology pretraining, no matched controls), supporting the absence-of-controlled-comparison claim.
- The negative-risk nature is unchanged but the evaluation design has improved. The explicit zero superiority margin for Kron vs. Quotient and the pre-specified hierarchical bootstrap remove post-hoc degrees of freedom, making both positive and negative outcomes cleanly interpretable.

The contribution remains singular and focused: controlled evidence on whether electrical hierarchy changes the scaling frontier. The revision has not added contribution sprawl.

### 4. Frontier Leverage — 8/10

**Up from 7.** The new § Frontier Positioning Without Extra Treatments provides the requested engagement with learned graph tokenization (GraphFM, Graph Mamba, GraphGPS) and linear attention. The exclusion rationale is methodologically sound: these add learned mechanism axes that would confound the causal question. GridSFM's linear-cost global summary is justified as the domain-established O(N) control.

The positioning is no longer passive. The proposal affirmatively argues that deterministic operators ask a narrower, more attributable question than learned tokenizers. The paragraph on linear attention correctly identifies kernel/positional/normalization choices as a second method axis.

Remaining minor weakness: the frontier positioning cites GraphFM, Graph Mamba, and GraphGPS but does not discuss the most recent work on physics-informed graph foundation models (e.g., the 2025–2026 wave of PDE-foundation-model pretraining methods like DPOT, Poseidon, or Helia). These are the closest frontier analogs — domain-specific foundation models with structured communication — and a brief note on how the Kron hierarchy relates to their multi-resolution strategies would be the final polish. **MINOR.**

### 5. Feasibility — 8/10

**Up from 7.** The two round-1 weaknesses are addressed:

- The timeline is extended to 10–12 weeks with three weeks for F0–F3, which is realistic for implementing the METIS partition, geometry builders, CommunicationCore seam, and portable normalization.
- The portable normalization contract now requires case-metadata-declared `baseMVA` and fails closed on missing metadata, rather than assuming 100 MVA. This resolves the verification concern.

The confirmatory import test (legacy imports must fail) is a concrete feasibility gate that prevents accidental regression to the prototype path.

The GPU budget remains tight (LB-6 analysis unchanged), but the 20% reserve and the explicit blocking rule ("if C falls below a preregistered learning horizon, block the campaign rather than deleting a seed, baseline, diversity level, or quotient control") are honest risk management.

The code–proposal gap still exists (the implementation files are unchanged), but the proposal now explicitly acknowledges this: the prototype code is "historical" and "cannot be imported by the confirmatory entry point." The gap is an implementation task, not a proposal flaw.

### 6. Validation Focus — 9/10

**Up from 8.** The round-1 weakness (unspecified Kron-vs-quotient margin) is resolved with a clean zero superiority margin for the error contrast and an explicit 5% non-inferiority margin for physical residual. The hierarchical bootstrap over provenance groups is preregistered. The five-part primary pass gate is precise and falsifiable. The statistical unit, aggregation order, and reporting requirements are all specified.

The evidence package is now outlined with four load-bearing figures and a table plan. This is tighter than round 1's absent writing plan.

### 7. Venue Readiness — 8/10

**Up from 7.** The round-1 weaknesses are partially addressed:

- The paper outline now exists: four load-bearing figures and a table plan. This is more concrete than round 1's absence.
- The distinction between implementation readiness and publication readiness is now explicit. The proposal correctly notes that it is "ready for implementation when its contracts survive review," not claiming publication readiness.
- No preliminary results exist yet, but the proposal correctly sequences the pilot after F0–F4. Since the review stance instructs me not to make preliminary results a blocking methodological requirement when the proposal schedules them after integrity contracts, I no longer penalize this as heavily.

Remaining weakness: the four-figure plan is load-bearing but skeletal. No target page count, no planned supplementary structure, no venue-specific formatting notes. This is **MINOR** — the proposal's scope is method specification, not submission preparation.

---

## Weighted Composite

| Dimension | Weight | Score | Weighted |
|:---|:---:|:---:|:---:|
| Problem Fidelity | 15% | 9 | 1.35 |
| Method Specificity | 25% | 9 | 2.25 |
| Contribution Quality | 25% | 9 | 2.25 |
| Frontier Leverage | 15% | 8 | 1.20 |
| Feasibility | 10% | 8 | 0.80 |
| Validation Focus | 5% | 9 | 0.45 |
| Venue Readiness | 5% | 8 | 0.40 |
| **Composite** | | | **8.70** |

### GAP

The proposal is at 8.70, not 10. What separates it:

1. **No implemented contracts.** The proposal specifies precisely what must be built (METIS partition, conservative transport, CommunicationCore, portable normalization, content-addressed registry) and the implementation files have not changed. A 10 would have at least F0–F1 implemented with passing algebra tests. The 0.30 gap is mostly engineering distance, not conceptual weakness.
2. **Frontier positioning is adequate but not complete.** The discussion of learned tokenizers is sound; the discussion of physics-informed multi-resolution foundation models (DPOT, Poseidon, Helia) is absent. These are the closest structural analogs and positioning against them would strengthen the novelty argument.
3. **The adapter MLPs (`phi_down`, `phi_up`) are not fully re-specified in the revision.** The round-0 text described two-layer MLPs with residual normalization; the revision does not restate this. The revision should be self-contained.

---

## Prior Concern Disposition

| # | Prior Concern | Priority | Status |
|:---:|:---|:---:|:---:|
| 1 | Implement topology-only METIS partition | CRITICAL | **Resolved in proposal.** § Common topology partition specifies METIS on unweighted bus graph with fixed seed. Implementation is scheduled at F1. The proposal is self-consistent; the code is unchanged but the proposal explicitly labels it as legacy. |
| 2 | Implement portable normalization (declared case `baseMVA`) | CRITICAL | **Resolved in proposal.** § Portable Data Contract requires case-metadata-declared `baseMVA`, fails closed on missing metadata, explicitly forbids fitted percentiles. Scheduled at F3. |
| 3 | Remove label-dependent reconstruction gate | CRITICAL | **Resolved in proposal.** The sparsifier uses a topology-only Frobenius residual gate. The label-dependent `Vb_samples` / `RECON_ASSERT` gate in the current code is legacy and cannot be imported by the confirmatory path. |
| 4 | Specify restriction operator derivation | IMPORTANT | **Resolved.** Conservative mass-adjoint pair `(U, R)` is fully derived with constant-preservation and adjointness proofs. Honest scoping: not presented as Galerkin. |
| 5 | Characterize Yaniv and Beck | IMPORTANT | **Resolved.** Four-axis characterization: per-cluster ANNs, no Kron–Schur, no multi-topology pretraining, no matched controls. |
| 6 | Frontier positioning against learned graph tokenization | IMPORTANT | **Resolved.** § Frontier Positioning Without Extra Treatments cites GraphFM, Graph Mamba, GraphGPS with a specific exclusion rationale. |
| 7 | Implement CommunicationCore and GlobalSummaryCore | IMPORTANT | **Resolved in proposal.** § Minimal software contract specifies the `CommunicationCore` protocol and lists `LocalCore`, `GlobalSummaryCore`, `HierarchyCore` as implementations. Scheduled at F2. |
| 8 | Split per-scenario features out of geometry cache | IMPORTANT | **Partially resolved.** The proposal's `HierarchyGeometry` dataclass contains only static topology operators, which implies per-scenario features are outside the geometry. However, the proposal does not explicitly state where per-scenario features (analogous to current `v_aff`, `cbus_x`) are computed in the new pipeline. The affine unpool requires a per-scenario `v_aff`-like quantity; where it lives is unspecified. |
| 9 | Run one G8 pilot | IMPORTANT | **Deferred by design.** The pilot is scheduled at F5, after contracts and tests pass. The response table explains: "A pilot on the current prototype would test the wrong method." This is methodologically correct. |
| 10 | Verify `baseMVA` values across PGLib cases | IMPORTANT | **Resolved.** The portable normalization contract reads `baseMVA` from case metadata and fails closed when absent. No assumed default. |
| 11 | Specify coarse-layer initialization | MINOR | **Resolved.** Same initializer family, deterministic seed mapping, independent weights. |
| 12 | Preregister Kron-vs-quotient paired margin | MINOR | **Resolved.** Zero superiority margin for error; 5% non-inferiority margin for residual. |
| 13 | Create figure plan | MINOR | **Resolved.** Four-figure plan with brief table description in § Evidence Package. |

---

## Simplification Opportunities

1. **Collapse the affine unpool into the adapter's scatter pass and remove `v_aff` as a separate per-scenario tensor.** The proposal uses an affine unpool `V_hat = P_sparse V_B + v_aff` for the physical channel. But the physical prolongation is now secondary — the primary path is latent transport through `U` and `R`. If the physical-voltage channel is dropped from the primary claim (which the proposal already does — "The adapter predicts no voltage and owns no task loss"), then `v_aff` computation, storage, and attachment are dead weight for the confirmatory path. Either (a) confirm that the physical-voltage channel is needed in the `mlp_out` merge step and specify where `v_aff` is computed in the new pipeline, or (b) delete it from the primary adapter entirely, leaving only the latent `(U, R)` transport. This simplifies the adapter to pure latent communication with no per-scenario physical quantities.

2. **Remove the `HeteroDataPerSampleMVANormalizer` from the confirmatory path.** The proposal's portable contract uses case-declared `baseMVA`, which is a per-case constant, not a per-sample fitted percentile. The per-sample normalizer is a prototype artifact that the confirmatory import gate should block alongside the per-grid normalizer. Maintaining two legacy normalizers plus one new portable normalizer is unnecessary complexity if the legacy pair is dead code behind a failing import fence.

3. **NONE** (only two opportunities identified; no third non-trivial simplification exists that would not alter the method).

## Modernization Opportunities

1. **Use `torch.compile` with numerical-parity checks.** The round-1 suggestion stands and the revision now conditionally accepts it ("requires output-parity, gradient-parity, and FLOP-counter checks for every core"). This is the single highest-leverage systems improvement and should be attempted early in F4 to inform the FLOP budget freeze.

2. **NONE.** The round-1 suggestion to replace TransformerConv with a cheaper backbone is correctly rejected by the revision ("changes the experimental treatment"). The sparse-RHS suggestion for the dense Schur intermediate is also correctly scoped as a non-method optimization that does not eliminate the dense output. No further genuinely necessary modern primitive is missing from the proposal.

## Drift Warning

**NONE.** The revision stays strictly within the problem anchor. No new learned mechanism, task head, auxiliary loss, or baseline has been added. The frontier positioning section is explicit that learned tokenizers and linear attention are positioned but not added as treatments. All changes are specification tightening and contract clarification.

## Remaining Action Items

| Priority | Item |
|:---:|:---|
| IMPORTANT | Specify where per-scenario quantities (the analog of `v_aff` for the physical-voltage merge channel) are computed in the new pipeline, or confirm that the primary latent adapter does not require them. |
| IMPORTANT | Restate `phi_down` and `phi_up` architecture (width, depth, activation, normalization) in the revised proposal to make it self-contained rather than depending on round-0 text. |
| MINOR | Add a brief note positioning the electrical hierarchy against physics-informed multi-resolution FM methods (DPOT, Poseidon, Helia) as closest structural analogs. |
| MINOR | Specify target page count and supplementary structure in the evidence package. |

## Verdict

**REVISE**

The weighted composite is 8.70, below the 9.0 threshold. There is no drift. The contribution is focused and singular. There is no blocking methodological issue and no complexity bloat. Every prior CRITICAL concern is resolved at the proposal level. Three of four prior IMPORTANT concerns are fully resolved; one (per-scenario feature placement) is partially resolved.

The 0.30 gap to READY is narrow and consists of: (1) two specification incompleteness items (per-scenario feature pipeline, adapter MLP architecture) that are IMPORTANT but not CRITICAL, and (2) two MINOR positioning/formatting items. The proposal is close to implementation-ready; one focused pass to close these items should bring the composite above 9.0.

</details>


### Round 3

# Round 3 Review

- **Reviewer transport**: AGY / Antigravity
- **Reviewer model**: Claude Opus 4.6 (Thinking)
- **Calibration**: none
- **Weighted score**: 9.10 / 10
- **Verdict**: READY
- **Drift warning**: NONE

<details>
<summary>Full raw AGY response</summary>

Good. The round-2 refinement explicitly states on line 155 that the confirmatory path has **no** `v_aff`, `cbus_x`, boundary-voltage tensor, affine physical unpool, or HELM feature. The adapter is pure static-geometry plus real-latent transport. This directly resolves my round-2 concern — my round-2 review incorrectly assumed a physical-voltage merge channel existed in the revised method. The refinement corrected that premise.

Now I have everything needed for the round-3 review.

---

# Round-3 Review — Electrical Hierarchies as a Scaling Mechanism for Grid Foundation Models

**CALIBRATION: none**

---

## Load-Bearing Questions (Round 3)

### LB-1. Is the pure static-geometry/latent-transport boundary now unambiguous?

**Yes.** The revised proposal at line 155 of the round-2 refinement states:

> "The hierarchy consumes no per-scenario physical reconstruction quantity. In particular, the confirmatory path has no `v_aff`, `cbus_x`, boundary-voltage tensor, affine physical unpool, or HELM feature. An operating scenario enters only through the common fine-level input encoder; the geometry registry is a pure function of topology and frozen geometry policy."

This is an explicit, verifiable contract. The boundary is sharp: `HierarchyGeometry` contains exactly `{topology_key, partition, restrict, prolong, coarse_graph, provenance}` — all static, all topology-derived. No per-scenario tensor enters the adapter. Scenario data flows through the common encoder on the fine graph, and the adapter operates purely on latent hidden states `h` using the static `(U, R, E_coarse)` operators.

My round-2 review incorrectly assumed a physical-voltage merge channel (`v_hat`, `v_aff`) was still present in the confirmatory method. The refinement correctly identified this as a stale assumption carried from the round-0 prototype and round-1 code. The revised adapter computes:

```
h_B^down = h_B + phi_down(aggregate_R(h_I))
z_B      = CoarseProcessor(h_B^down, E_coarse)
h_I^out  = h_I + phi_up([h_I, aggregate_U(z_B)])
h_B^out  = z_B.
```

No voltage, no physical reconstruction, no per-scenario quantity. This is a pure latent communication adapter. **Unambiguous.**

### LB-2. Are `phi_down` and `phi_up` fully specified without needless architecture freedom?

**Yes.** The revision fixes:

- `phi_down(x) = W_2 SiLU(W_1 LayerNorm(x))` with `W_1, W_2: d → d`.
- `phi_up([h,u]) = W_4 SiLU(W_3 LayerNorm([h,u]))` with `W_3: 2d → d` and `W_4: d → d`.
- No dropout, no output activation.

This is fully specified: depth (2 linear layers each), widths (all `d` or `2d → d`), activation (SiLU), normalization (pre-LayerNorm), and the explicit absence of dropout and output activation. There is no architecture search space or freedom left. An implementer can write these modules from this description alone. **Resolved.**

### LB-3. Do DPOT and Poseidon complete the relevant scientific-FM positioning without forcing a mismatched baseline?

**Yes.** The revision adds at line 270:

> "[DPOT] and [Poseidon] are the closest scientific-foundation-model analogs: both pretrain operator models across PDE data, and Poseidon uses a multiscale operator transformer. They establish that multiscale structure and broad physical pretraining can support transfer. They do not test irregular power-network topologies, admittance-derived geometry, or a Kron-versus-generic-hierarchy intervention. They motivate the question; they are not matched baselines for it."

This is the correct positioning. DPOT and Poseidon validate the broader idea that multi-resolution physics pretraining transfers; the proposal's narrower question is whether a specific electrical operator family (Kron–Schur) improves a specific domain (power grids) over generic alternatives. The two are mutually motivating but not substitutable as baselines. No mismatched baseline is forced. **Resolved.**

### LB-4. Does the explicit `cardef/gridfm-graphkit` fork contract remain narrow, testable, and prominent?

**Yes, unchanged from round 2 and still strong.** The fork contract (§ Explicit Fork Contract) is in the same position, with the same five additive module categories, the same confirmatory import test, and the same upstream-flat compatibility requirement. The title-block scope declaration remains. No expansion of the fork boundary has occurred across the three rounds.

The one aspect worth noting is that the deletion of the physical-voltage channel from the confirmatory adapter *narrows* the fork boundary further. The prototype's `GNS_hetero_hier` with its HELM unpool, `v_aff` computation, and coarse physical features is now entirely legacy — the confirmatory adapter is simpler than the prototype. This is a healthy trajectory: each round has reduced rather than expanded complexity. **Narrow, testable, prominent.**

### LB-5. Is any unresolved proposal-level issue blocking implementation?

**No.** I systematically check:

- **Problem anchor**: unchanged and precise. ✓
- **Partition**: METIS on unweighted bus graph, fixed seed, deterministic anchor selection. Fully specified. ✓
- **Kron–Schur construction**: dense intermediates, sparse output with row/column coverage, frozen budget `kappa`, topology-only residual gate. Fully specified. ✓
- **Transport operators**: `U = D_I^-1 A`, `R = D_B^-1 A^T`, conservative, mass-adjoint, constant-preserving. Proven. ✓
- **Adapter architecture**: `phi_down`, `phi_up` fully specified (width, depth, activation, normalization). CoarseProcessor: same block family, independent weights, deterministic seed. ✓
- **No per-scenario hierarchy quantities**: explicitly excluded. ✓
- **CommunicationCore protocol**: specified with three implementations. ✓
- **Geometry registry**: content-addressed, immutable, topology-keyed. ✓
- **Shared baselines**: Flat-HGNS, Global-HGNS (GridSFM v1.1 style), Quotient-HGNS. All specified. ✓
- **Portable normalization**: case-declared `baseMVA`, fail-closed. ✓
- **Run matrix**: 20 mandatory runs (3×3×2 + 2). ✓
- **Statistical design**: topology-level, provenance-balanced, hierarchical bootstrap, five-part primary pass gate, zero superiority margin for mechanism claim. ✓
- **Failure modes**: nine explicit conditions. ✓
- **Fork boundary**: five additive categories, confirmatory import test, upstream-flat compatibility. ✓
- **Implementation path**: F0–F5, sequenced, with blocking rules. ✓
- **Budget**: 230 GPU-hours, 20% reserve, blocking rule if C is too small. ✓
- **Evidence package**: four figures, table plan, implementation/publication readiness distinction. ✓
- **Frontier positioning**: GraphFM, Graph Mamba, GraphGPS, DPOT, Poseidon. All positioned with exclusion rationale. ✓
- **Prior-work characterization**: Yaniv and Beck four-axis distinction. ✓

Every specification that was CRITICAL or IMPORTANT in prior rounds is now resolved in the proposal text. No blocking issue remains.

---

## Dimension Scores

### 1. Problem Fidelity — 9/10

Unchanged across all three rounds. The problem anchor is precise, the non-goals are honest, the success condition is falsifiable, and the failure modes are explicit. The operational definition of "foundation model" is properly scoped with the LUMINA precedent. The three-axis measurement scope (diversity, size, compute) is explicitly not claimed as a universal scaling law.

The 0.1 from a 10 is the inherent limitation of testing one fixed model size. The proposal acknowledges this but does not claim it as a non-limitation. Honest.

### 2. Method Specificity — 10/10

**Up from 9.** Both round-2 weaknesses are resolved:

- `phi_down` and `phi_up` are now fully specified: two-layer, `d`-width, SiLU, pre-LayerNorm, no dropout, no output activation. The proposal is self-contained.
- The per-scenario ambiguity is eliminated. The proposal explicitly states that no per-scenario quantity enters the hierarchy adapter. The adapter is pure latent transport over static topology operators.

Every method component is now implementable from the proposal text alone. The partition rule, Kron–Schur construction, sparsification policy with coverage guarantee, conservative transport pair with algebraic proof, adapter architecture, coarse processor contract, generic-hierarchy control, shared baselines, normalization, and geometry registry are all fully specified. The implementation contract (protocols, dataclasses) is clean. An engineer who has not read the prior code could implement this method from this document.

### 3. Contribution Quality — 9/10

Unchanged from round 2. The dominant contribution (controlled comparison of electrical hierarchy as FM scaling mechanism) is focused and singular. Prior work is characterized precisely. The evaluation design has preregistered gates with zero arbitrary margins. Positive and negative outcomes are both cleanly interpretable.

The 0.1 from a 10 is the inherent outcome-dependent ceiling: if the result is negative, the contribution is a well-designed negative result, which is less impactful than a well-designed positive result. This is not a proposal flaw; it is inherent to empirical science.

### 4. Frontier Leverage — 9/10

**Up from 8.** The round-2 weakness is resolved:

- DPOT and Poseidon are now cited as the closest scientific-FM analogs, with a precise statement of why they motivate but do not answer the irregular-grid/Kron question. The positioning is complete: learned graph tokenizers (GraphFM, Graph Mamba, GraphGPS) are excluded for causal reasons; scientific FMs (DPOT, Poseidon) are positioned as motivating analogs; linear attention is excluded as a second method axis; GridSFM's linear-cost summary is the domain-established O(N) control.

The frontier engagement is no longer passive or merely adequate — it is active, precise, and methodologically grounded. The proposal positions its deterministic electrical operator as the physics-preserving complement to the generic learned mechanisms, without claiming superiority or forcing mismatched baselines.

### 5. Feasibility — 8/10

Unchanged from round 2. The timeline (10–12 weeks), budget (230 GPU-hours with 20% reserve), and implementation path (F0–F5) are realistic. The portable normalization contract fails closed. The confirmatory import test prevents regression. The blocking rule for insufficient compute is honest.

The 0.2 from a 10 remains: (a) the GPU budget is tight and one throughput surprise could force cutting adaptation, and (b) the code–proposal gap is real engineering work. Neither is a proposal flaw — the first is resource reality and the second is correctly scheduled.

### 6. Validation Focus — 9/10

Unchanged from round 2. The five-part primary pass gate is precise and falsifiable. The statistical design (topology-level, provenance-balanced, hierarchical bootstrap) is strong. The zero superiority margin for the mechanism claim and the 5% non-inferiority margin for physical residual are preregistered. Construction failures stay in the denominator.

### 7. Venue Readiness — 8/10

Unchanged from round 2. The four-figure plan exists. The implementation/publication readiness distinction is explicit. The response table correctly rejects target page count as a method-readiness gate: "The venue is not selected and page limits can change."

The 0.2 from a 10: no draft exists, no venue is selected, and the evidence package is a plan. This is appropriate for a proposal-stage review. The proposal does not claim venue readiness; it claims implementation readiness. Honest.

---

## Weighted Composite

| Dimension | Weight | Score | Weighted |
|:---|:---:|:---:|:---:|
| Problem Fidelity | 15% | 9 | 1.35 |
| Method Specificity | 25% | 10 | 2.50 |
| Contribution Quality | 25% | 9 | 2.25 |
| Frontier Leverage | 15% | 9 | 1.35 |
| Feasibility | 10% | 8 | 0.80 |
| Validation Focus | 5% | 9 | 0.45 |
| Venue Readiness | 5% | 8 | 0.40 |
| **Composite** | | | **9.10** |

### GAP

The proposal is at 9.10, not 10. What separates it:

1. **Budget tightness is real.** The 230 GPU-hour ceiling with 20 mandatory runs and optional adaptation is arithmetically feasible but leaves thin slack. A 10 would have either a larger budget or a demonstration (from profiling) that the slack is comfortable. This is a resource constraint, not a method flaw.
2. **The code–proposal gap is engineering distance.** The proposal specifies everything needed; the implementation files have not changed. A 10 would have at least F0–F1 with passing algebra tests. The proposal correctly does not claim this work is done.
3. **The contribution ceiling depends on the empirical outcome.** A 10 in contribution quality would require confidence that the result is positive and large. This is unknowable at the proposal stage. The experimental design is as strong as possible to make any outcome interpretable.

None of these gaps are methodological deficiencies. They are resource, implementation, and epistemic limitations that are honestly acknowledged and correctly managed.

---

## Prior Concern Disposition

| # | Prior Concern | Origin | Priority | Status |
|:---:|:---|:---:|:---:|:---:|
| 1 | Implement topology-only METIS partition | R1 | CRITICAL | **Resolved.** Specified in proposal since R1 revision. |
| 2 | Implement portable normalization | R1 | CRITICAL | **Resolved.** Case-declared `baseMVA`, fail-closed. |
| 3 | Remove label-dependent reconstruction gate | R1 | CRITICAL | **Resolved.** Topology-only residual gate; legacy gate is unfailable-imported. |
| 4 | Specify restriction operator | R1 | IMPORTANT | **Resolved.** Conservative mass-adjoint pair `(U, R)`, proven. |
| 5 | Characterize Yaniv and Beck | R1 | IMPORTANT | **Resolved.** Four-axis distinction. |
| 6 | Frontier positioning (learned tokenizers) | R1 | IMPORTANT | **Resolved.** GraphFM, Graph Mamba, GraphGPS with exclusion rationale. |
| 7 | Implement CommunicationCore/GlobalSummaryCore | R1 | IMPORTANT | **Resolved in proposal.** Protocol specified; scheduled at F2. |
| 8 | Per-scenario features in geometry cache | R1→R2 | IMPORTANT | **Resolved.** The premise was incorrect — the confirmatory adapter has no per-scenario features. The geometry cache is pure static topology operators. Scenario data enters only the common encoder. |
| 9 | Run G8 pilot | R1 | IMPORTANT | **Deferred by design.** Correctly scheduled at F5 after integrity contracts. |
| 10 | Verify `baseMVA` across PGLib cases | R1 | IMPORTANT | **Resolved.** Fail-closed on missing metadata. |
| 11 | Specify `phi_down`/`phi_up` architecture | R2 | IMPORTANT | **Resolved.** Width, depth, activation (SiLU), normalization (LayerNorm), no dropout, no output activation. Self-contained. |
| 12 | Position against DPOT/Poseidon | R2 | MINOR | **Resolved.** Cited as closest scientific-FM analogs with precise exclusion rationale. |
| 13 | Coarse-layer initialization | R1 | MINOR | **Resolved.** Same initializer family, deterministic seed, independent weights. |
| 14 | Preregister Kron-vs-quotient margin | R1 | MINOR | **Resolved.** Zero superiority margin. |
| 15 | Figure plan | R1 | MINOR | **Resolved.** Four figures, table plan. |
| 16 | Target page count / supplement | R2 | MINOR | **Rejected as method gate.** Venue is not selected. Correct. |

All CRITICAL items: resolved. All IMPORTANT items: resolved (one deferred by design, which is correct). All MINOR items: resolved or correctly rejected. No prior concern is unresolved or partially resolved.

---

## Simplification Opportunities

**NONE.** The proposal is already at its minimal adequate mechanism. The adapter has two MLPs and one coarse processor. The geometry has three sparse operators. The run matrix has three cores plus one control. There is no per-scenario hierarchy feature, no auxiliary loss, no HELM reconstruction, no recursive levels, no learned tokenizer, no task-specific head. Every previous round's simplification opportunity has been addressed: the physical-voltage channel is deleted, the legacy normalizers are behind the import fence, and the geometry cache is static-only. There is nothing left to delete without removing the treatment itself.

## Modernization Opportunities

1. **`torch.compile` with parity checks.** This carries forward from rounds 1 and 2. The proposal conditionally accepts it. It remains the single highest-leverage systems improvement for stretching the GPU budget. No other modernization is needed.

## Drift Warning

**NONE.** The proposal has been strictly monotonically simplified across three rounds: round 1 removed the HELM head from the primary path; round 2 removed the physical-voltage merge channel entirely; the adapter is now the minimal latent-transport mechanism. No new learned component, baseline, auxiliary loss, or task head has been added. The frontier positioning adds citations, not treatments. The fork boundary has narrowed, not expanded. The problem anchor is word-for-word identical.

## Remaining Action Items

| Priority | Item |
|:---:|:---|
| — | No proposal-level action items remain. All method, boundary, and validation specifications are complete. Remaining work is implementation (F0–F5), which is correctly scheduled and sequenced. |

## Verdict

**READY**

The weighted composite is 9.10, above the 9.0 threshold. There is no drift. The contribution is focused and singular: one controlled comparison of electrical hierarchy as the scaling mechanism of a domain-specific grid foundation model. There is no blocking methodological issue: every prior CRITICAL and IMPORTANT concern across three review rounds is resolved. There is no complexity bloat: the adapter is two MLPs and one coarse processor operating on static topology geometry with no per-scenario features; the run matrix is three cores plus one control at three diversity levels with two seeds.

The proposal is ready to implement. It is not ready to submit — that requires F0–F5, the pilot, the confirmatory campaign, and the manuscript. The proposal correctly distinguishes these thresholds and does not claim the latter.

</details>
