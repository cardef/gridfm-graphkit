# Research Idea Report

> **Historical discovery artifact.** This report records the 2026-07-06 idea-generation stage; it is not an implementation or experiment specification. The canonical reviewed artifacts are `refine-logs/FINAL_PROPOSAL.md`, `refine-logs/EXPERIMENT_PLAN.md`, and `refine-logs/EXPERIMENT_TRACKER.md`. References below to the removed technical plan, the old three-seed core, auxiliary losses, or earlier exactness language are historical only.

**Direction**: Hierarchical GNN scaling for gridfm-graphkit (PF/OPF/SE at 2k–10k+ buses; cross-size transfer)
**Generated**: 2026-07-06
**Ideas evaluated**: 35 generated (5 lens shards × ~5 + 10 Codex brainstorm) → 25 after mechanical dedup → 25 through feasibility gate (0 dropped) → cross-model triage top-10 → **3 recommended** → 2 CPU pilots run (GPU pilots skipped: no GPU on this machine)
**Reviewer backend**: Codex MCP, gpt-5.5, xhigh (threadId `019f3673-cae9-78d1-83a0-2295561296c4`; traces in `.aris/traces/idea-creator/2026-07-06_run01/`)

## Historical Status Addendum (2026-07-10) — superseded by the canonical proposal

Idea 3 (I9, Kron-Schur U-Net) became the active program (`refine-logs/FINAL_PROPOSAL.md`, now v3). The repo-reviewed implementation audit `docs/kron_hierarchical_gridfm_technical_plan.md` (2026-07-10) corrects the exactness language used below; the historical text is preserved unchanged, read it with these corrections:

- **"Prolongation = the exact harmonic extension"** (Idea 3, step 2): exact only when evaluated on the **unsparsified** operators and with zero interior injections ("zero-injection harmonic extension"). The prototype's row-sparsified P is a *latent transport* operator and carries no exactness claim; v3 splits the two (`P_phys` vs `P_latent`).
- **"Linearized reconstruction" / affine correction**: the v1 affine design silently sets w₀ = 1 and is **not** the true first-order term of the boundary-conditioned HELM series; it is renamed `affine_flat` and kept as a legacy control.
- **Sparsity selection**: any rule that picks operator sparsity by evaluating reconstruction against true boundary voltages (as the later R006 gate did) is label leakage; production sparsity policies are topology-only, frozen on development grids.
- **"Electrically exact arm" (Idea 2 cross-reference)**: the exactness sentence is scoped to the physical path at numerical solver tolerance; the hierarchical GNN's latent path is honestly approximate.
- **Claim language**: "beats flat at any depth" is retired in favor of the measured depth/width frontier; a credible scalable-global baseline (HH-MPNN-like) joins the comparison set so GRIT's memory wall is not over-read.

## Landscape Summary

The learned-power-flow field has flat-GNN solvers (PowerFlowNet, CANOS, TypedGNN, this repo's heterogeneous GNS) that work in-distribution but degrade on unseen topologies and grid sizes — the PFΔ benchmark (rivera2025) documents this against the solvers' own robustness claims. The graph-transformer branch hit a compute wall instead: GRIT-class models need O(N²) attention plus dense O(N²K) RRWP precompute, which is why gridfm-graphkit has GRIT configs only for case14 while HGNS configs reach case2000/Texas.

The newest relevant result reframes the problem: HH-MPNN (arXiv 2510.06860) bolted Performer linear attention and an effective-resistance encoding onto a heterogeneous MPNN — i.e., removed the receptive-field limit — and zero-shot size transfer *stayed* broken ("orders of magnitude worse", rescued only by fine-tuning on 5% of target-grid data). The field's two stock explanations for transfer failure (receptive field; vague "scale-dependent features") are therefore both unproven attributions. Generic-GNN theory (Yehudai et al., ICML 2021; spectral bio-graph work, arXiv 2305.15611) characterizes size-generalization failure but has never been tested interventionally in a physics-regression domain.

Meanwhile the mesh-simulation world (MultiScale MeshGraphNets, BSMS-GNN, EvoMesh, M4GN) treats hierarchy as the solved answer to long-range propagation — but all their coarsening assumes spatial, homogeneous meshes. Power grids are heterogeneous and non-spatial, yet possess something meshes lack: a *domain-native* hierarchy (voltage levels, substations, zones) and an electrically exact reduction calculus (Kron/Ward = Schur complements on the admittance matrix, Dörfler–Bullo). No published PF/OPF model uses either (wiki gaps G1/G2, re-verified today; NDP uses Kron reduction only as a generic pooling operator for graph classification).

The four wiki gaps (G1 electrical hierarchy, G2 mesh-multiscale adaptation, G3 size transfer, G4 sub-quadratic transformers) all survive the 2025–26 survey, with one caveat: G4's "linear attention on grids" slot is taken by HH-MPNN, so differentiation must come from *structured* (hierarchical/physics-derived) attention, sparse positional encodings, or coarse-token designs.

Our two CPU pilots (below) add local evidence: Kron reduction of real 1354–2000-bus grids produces coarse graphs whose fill-in is electrically negligible (hierarchical coarsening is structurally cheap), and DC PTDF sensitivities decay *slowly* (one decade per ~10 hops on grids ~25–28 hops across) and correlate better with hop distance than with effective resistance — long-range coupling is real, and the folklore "electrical distance gives sharp exponential locality" is at minimum incomplete.

## Recommended Ideas (ranked)

### Idea 1 (I3+I4): Factorial attribution of zero-shot size-transfer failure
- **Method (what we actually do)**: (1) Train the repo's existing heterogeneous GNS on small grids (case14–118 pool) with the existing pipeline. (2) Evaluate zero-shot on case500/case2000, then intervene on one factor at a time: swap the normalizer (train-fit MVA vs per-sample vs an *oracle* arm where normalizer stats are refit on the target grid — a zero-gradient, config-only change via the repo's `--normalizer_stats` path); vary receptive range (depth sweep + a virtual global node); quantile-match test-grid input marginals to the training distribution; evaluate on Ward/Kron-reduced intermediates of case2000 that interpolate grid size. (3) For the fine-tuning side (I4): adapt to case2000 under a matched 5%-sample budget with four arms — zero-shot / normalizer-refit-only / affine input-output adapters / full fine-tune. (4) Report one table: fraction of the transfer gap closed per factor.
- **Hypothesis**: normalization + feature-marginal shift account for most of the gap; receptive field for little (consistent with HH-MPNN's global attention failing to fix zero-shot transfer).
- **Minimum experiment**: the two normalizer arms alone (~4 runs, config-only) already give the headline signal.
- **Expected outcome**: either "the field is building architectures to fix a preprocessing/calibration bug" (high-impact negative) or a quantified confirmation that structure, not calibration, is the bottleneck — both redirect G3 work.
- **Novelty**: 8/10 — closest: Yehudai et al. ICML 2021 (generic, characterization not intervention), PFΔ / HH-MPNN (document failure, don't attribute). Cite both early; claim "controlled attribution in physics-regression grid solvers", not "first size-gen study" and not causal proof.
- **Feasibility**: ~16–20 short single-GPU runs on existing configs/data; the repo's two normalizers and `--normalizer_stats` restore path make the key arms config-only. No new model code except the virtual-node variant.
- **Risk**: MEDIUM (factors may interact; use factorial design, report interactions).
- **Contribution type**: diagnostic / empirical.
- **Pilot result**: SKIPPED — needs GPU. (CPU Pilot B is consistent with "range alone is not the story": coupling is long-range yet HH-MPNN showed adding range doesn't fix transfer.)
- **Reviewer's likely objection**: oracle interventions (stats transplant, marginal matching) are "artificial". Answer: that is what makes them attributions rather than methods.
- **Why we should do this**: highest information-per-GPU-hour on the list; its outcome decides whether Ideas 2–3 are even aimed at the right problem. Codex triage rank #1.

### Idea 2 (I6): Domain-given vs learned coarsening — the controlled hierarchy benchmark
- **Method (what we actually do)**: (1) Build ONE U-Net-style hierarchical wrapper around the repo's existing HGNS layers (encode → pool → coarse MP → unpool → decode, with skips). (2) Swap ONLY the pooling assignment matrix between arms: learned (TopK/DiffPool-style), METIS, voltage-level/zone partition read from the grid metadata, Kron/Ward-based, and random balanced clusters as the control. (3) Train each at matched parameter count on case500 and case2000 PF (one OPF arm), measure accuracy, physics-residual violations, and small→large transfer. (4) The random/shuffled control separates "hierarchy helps because it's electrical" from "hierarchy helps because it shortens diameter".
- **Hypothesis**: domain-given partitions match or beat learned pooling — grid hierarchy is physically fixed, not task-emergent; if random matches everything, hierarchy's value is mere diameter-shortening.
- **Minimum experiment**: 5 conditions × case500 PF first; extend winners to case2000/OPF.
- **Expected outcome**: a decision table telling the field WHICH hierarchy (if any) to build — direct resolution of gap G1's actionable half; every outcome (electrical wins / learned wins / random ties) is a publishable, load-bearing result.
- **Novelty**: 8.5/10 (highest of the three) — no pooling-family ablation exists for grid state regression; generic pooling papers and M2NO don't answer the grid question.
- **Feasibility**: 5 conditions × 2 grids ≈ 1–2 weeks on 2–4 GPUs; the METIS/zone super-node transform doubles as infrastructure for I7/I8 follow-ups.
- **Risk**: MEDIUM (all hierarchy variants could lose to flat HGNS — which is itself the strongest possible negative result for the whole G1/G2 agenda).
- **Contribution type**: empirical / benchmark.
- **Pilot result**: SKIPPED — needs GPU. (CPU Pilot A de-risks the Kron/Ward arm: coarse graphs are structurally sparse after thresholding.)
- **Reviewer's likely objection**: pooling implementation details dominate the comparison. Mitigate: identical wrapper, matched params, multiple seeds, report per-arm tuning budget.
- **Why we should do this**: it is the empirical spine any hierarchy paper needs; Codex triage rank #2 and least novelty risk.

### Idea 3 (I9): Kron-Schur U-Net — electrically exact restriction/prolongation
- **Method (what we actually do)**: (1) Precompute, per grid, a 2-level hierarchy: boundary set = HV backbone ∪ generator buses (~25–30% of nodes); restriction = Kron reduction (Schur complement of the admittance matrix onto the boundary), with Ward-style redistribution of interior injections handling gen/load heterogeneity. (2) Prolongation = the exact harmonic extension V_int = −Y_ii⁻¹ Y_ib V_bnd (linearized reconstruction — a fixed sparse operator, not an MLP). (3) Wrap existing HGNS layers in a fine→coarse→fine V-cycle using these fixed operators; coarse-level edges are the thresholded Kron-reduced admittances. (4) Compare against matched-parameter flat HGNS and the generic-pooling arms of Idea 2 on case500/case2000 PF, with voltage-angle (long-range) error broken out.
- **Hypothesis**: because every coarse level is a true (DC-exact, AC-approximate) power-flow problem, the operator — not the loss — encodes the physics; largest gains on long-range quantities.
- **Minimum experiment**: 2-level version on case500 vs flat + mean-pool baselines; days on 1 GPU once the Schur precompute (scipy, CPU-minutes — now already prototyped in `idea-stage/pilot_cpu.py`) is wired into a task transform.
- **Expected outcome**: the first electrically grounded hierarchical PF/OPF architecture (G1+G2), or a clean negative showing electrical exactness buys nothing over generic coarsening (feeds Idea 2's table either way).
- **Novelty**: 8/10 — closest: NDP (Kron pooling, generic graph classification), Dörfler–Bullo/Opti-KRON (no learning), admittance-GP arXiv 2606.03717 (GP, no hierarchy). Frame as "Kron/Schur as restriction/prolongation *inside* a heterogeneous encoder-decoder GNN for state regression"; never claim "first Kron in ML".
- **Feasibility**: precompute validated on real grids today (Pilot A); model code is a wrapper over existing HGNS layers + 2 fixed sparse operators.
- **Risk**: MEDIUM — Schur unpooling may be brittle for gen/load heterogeneity; AC nonlinearity means harmonic extension is only a linearized prior.
- **Contribution type**: new method.
- **Pilot result**: **POSITIVE (CPU Pilot A)** — declared kill criterion ("Kron fill-in erases the advantage") cleared on both real grids: case1354_pegase boundary 31% → coarse off-diag density 2.5%; Texas2k boundary 24.9% → 19.4% raw but 95% of off-diagonal admittance mass concentrated in <0.5% of possible coarse edges (thresholding at |Y|>1e-3 p.u. keeps ~100% of mass at 7.2% density). Two-level coarsening is structurally cheap on real grids.
- **Reviewer's likely objection**: "NDP already did Kron pooling." Differentiation is task (continuous state regression), heterogeneity handling (Ward injection redistribution), and the exact Schur prolongation — none in NDP.
- **Why we should do this**: strongest constructive method candidate; its main structural risk is now retired by pilot; slots directly into Idea 2's benchmark as the electrically-exact arm. Codex triage rank #3; build after Idea 2's wrapper exists.

## Secondary tier (validated, not scheduled — pursue opportunistically)

| ID | Idea | Triage rank | One-line status |
|----|------|-------------|-----------------|
| I11 | Ward-crop training augmentation (electrically valid size continuum from one large case) | 4 | Novel (verified no prior art); LOW risk; shares Ward tooling with I3/I9 |
| I1 | Physics information-range atlas (Jacobian/PTDF decay per task and size) | 5 | Pilot B ran a v0: found *slow* decay and hop>Reff correlation — the surprise makes the full AC/per-task atlas MORE interesting, but the "exponential electrical locality" hypothesis needs revision |
| I7 | Hierarchy vs depth vs virtual node at matched compute (+ transfer) | 6 | Natural extension of I6; absorb I8 (levels sweep) and I19 (Dirichlet logging) into it |
| I13 | Schwarz/Ward inference-time decomposition (train small, solve big by fixed-point iteration) | 7 | Bold G3 attack; convergence with inexact learned inner solver is the crux |
| I17 | Topology vs size vs grid-identity robustness protocol (case2000 vs Texas same-size pair) | 8 | Cheap off shared checkpoints; reframes every robustness table in the field |
| I12 | H-matrix/Z_bus-structured attention (stage-1 compressibility diagnostic is standalone) | 9 | HIGH risk stage 2; stage 1 is CPU-days and publishable alone |
| I2, I24, I14, I15, I10, I22, I5, I18 | (mid-tier: ERF-vs-physics, task-dependence, depth-vs-perturbation-class, sparse RRWP, elimination-tree sweeps, coarse-token GRIT, 9241 scaling law, physics-loss regime split) | 11–18 | Not scheduled; several are cheap add-ons to the recommended experiments |

## Eliminated Ideas (cross-model kill-list — not executor taste)

| Idea | Reason eliminated |
|------|-------------------|
| I25 electrically-regularized learned coarsening | Longest build; pointless before I6 shows whether learned coarsening is even needed |
| I23 Ward-reduced teacher supervision | Depends on I9 existing; second-order refinement |
| I20 linear-probing for implicit hierarchy | Too interpretive standalone; keep as free appendix on any checkpoint |
| I19 over-smoothing Dirichlet sweep | Instrumentation, not contribution; folded into I7 |
| I8 L*(N) hierarchy-levels law | Parameter sweep; keep only as an I7 arm |
| I21 multigrid coarse-correction head | Less crisp than I9, less diagnostic than I6 |
| I16 contingency amortization break-even | Sound domain economics, off-center for the hierarchy agenda |

## Pilot Experiment Results

| Pilot | Idea | Hardware | Time | Key metric | Signal |
|-------|------|----------|------|-----------|--------|
| A: Kron fill-in density | I9 (gate) | CPU (this Mac) | ~1 min | Texas2k: 95% of off-diag mass in 0.31% of possible coarse edges; density 7.2% at 1e-3 tol; case1354: 2.5% density | **POSITIVE** — kill criterion cleared |
| B: PTDF locality (v0 of I1) | I1 premise / I3 context | CPU (this Mac) | ~2 min | Spearman(log10\|PTDF\|): hop −0.42 vs Reff −0.16 (Texas2k); 10× decay per ~10 hops, diameter ~28 | **SURPRISING** — locality is weak and hop-dominated; contradicts electrical-exponential-decay folklore; long-range coupling confirmed |
| I3 normalizer arms | I3 | — | — | — | SKIPPED — needs GPU |
| I6 pooling arms | I6 | — | — | — | SKIPPED — needs GPU |

Pilot caveats (honest limits): DC/PTDF linearization, effective resistance from the DC Laplacian, flat-start structure only; Pilot B's hop-vs-Reff comparison doesn't decorrelate the two distance notions. The full I1 atlas would fix all three. Script: `idea-stage/pilot_cpu.py`; raw output: `idea-stage/pilot_results.txt`.

## Suggested Execution Order

1. **I3 normalizer arms first** (days, config-only): the cheapest decisive experiment on the list; its answer conditions everything downstream.
2. **I6 wrapper + 5 pooling arms** on case500 (the shared hierarchical infrastructure), extending to case2000/OPF for survivors.
3. **I9 Kron-Schur arms** into I6's benchmark (precompute already prototyped) — this is also where I23/I25 get re-evaluated if fixed pooling underperforms.
4. Opportunistic: I17 protocol numbers off I3's checkpoints; I1 full atlas (CPU) whenever GPUs are busy; I11 as the data-side counterfactual if I3 blames distribution shift.

## Next Steps

- [ ] Secure GPU access (remote; ~30–60 GPU-hours covers I3's full factorial + I6's case500 arms)
- [ ] Run I3's two normalizer arms (`HeteroDataMVANormalizer` vs `HeteroDataPerSampleMVANormalizer` vs oracle `--normalizer_stats` refit) — config-only
- [ ] `/novelty-check` re-run before any writeup (survey freshness: 2026-07-06)
- [ ] If I3/I6 results land, invoke `/experiment-plan` for the full ablation matrix and `/auto-review-loop` toward submission
