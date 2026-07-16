---
type: paper
node_id: paper:piloto2024_canos_fast_scalable
title: "CANOS: A Fast and Scalable Neural AC-OPF Solver Robust To N-1 Perturbations"
authors: ["Luis Piloto", "Sofia Liguori", "Sephora Madjiheurem", "Miha Zgubic", "Sean Lovett", "Hamish Tomlinson", "Sophie Elster", "Chris Apps", "Sims Witherspoon"]
year: 2024
venue: "arXiv"
external_ids:
  arxiv: "2403.17660"
  doi: null
  s2: null
tags: ["opf", "power-grid", "scalability", "gnn"]
added: 2026-07-06T07:25:36Z
---

# CANOS: A Fast and Scalable Neural AC-OPF Solver Robust To N-1 Perturbations

## One-line thesis
A constraint-augmented, encode-process-decode Interaction-Network GNN predicts near-optimal, near-feasible AC-OPF solutions in 33–65 ms up to 10,000 buses, robust to N-1 perturbations.

## Problem / Gap
Exact AC-OPF is too slow to solve close to real-time (needed every 5–15 minutes), so operators fall back on linear DC-OPF, which is AC-infeasible and drives costly uplift payments and higher emissions, worst on large grids; prior ML approaches to topology-robust OPF either targeted smaller grids, did not report constraint-satisfaction metrics, or represented topology changes only by zeroing out line admittance features — architecturally limited to dropping existing lines, unable to handle adding new ones.

## Method
CANOS (Constraint-Augmented Neural OPF Solver) is an encode-process-decode Graph Network over a heterogeneous bus/generator/load/shunt graph with directed AC-line and transformer edges propagated in both directions. The processor stacks 36–60 residual Interaction-Network blocks (2-layer MLPs, ReLU + layer norm; hidden size 128 for the 48/60-step "Deep" variants, 384 for the 36-step "Wide" variant), with separate update-function parameters per node/edge type. The decoder enforces generator-power and voltage-magnitude box bounds exactly via a sigmoid squash on the raw output (Eq. 13), and branch flows (pf, qf, pt, qt) are not learned at all but derived analytically from predicted bus voltages via the AC branch-flow (Ohm's) equations — guaranteeing those constraints by construction. Training minimizes supervised L2 loss plus a weighted (C=0.1) constraint-violation-degree penalty over the remaining equality/inequality constraints (reference-angle, power/voltage bounds, power balance), rather than relying on supervision alone.

## Key Results
- Pre-power-flow optimality: within ~0.1% of AC-IPOPT cost for the Deep-CANOS variants (99.98–100.04% cost ratio) and ~1% for Wide-CANOS, versus DC-IPOPT's 96.8–99.4% (worse on TopDrop); Deep-CANOS48 runs in 33 ms (500-bus) to 54 ms (10,000-bus) — sub-linear scaling — while DC-IPOPT itself takes 1.3 s to 50 s over the same range.
- Pre-power-flow constraint violations are negligible (~1e-9) for branch thermal and angle-difference constraints across all grid sizes; the only substantial violations are in real/reactive bus power balance (order 1e-2), which the optional power-flow post-processing step resolves (≥98% convergence rate from either CANOS or DC-IPOPT initialization).
- After power-flow post-processing, CANOS's worst remaining violation (slack-generator real-power bound) is far smaller than DC-IPOPT's: on the 500-bus TopDrop set, 0.154 p.u. vs. 3.86 p.u. (15 MW vs. 403 MW on a 1164 MW slack generator); the gap persists at 2,000- and 10,000-bus scale.

## Assumptions
- Supervised training requires large solved-instance corpora: 300k examples per grid size, generated with PowerModels.jl + Ipopt on PGLib-OPF base cases (500/2,000/10,000-bus), split 90/5/5.
- Robustness is evaluated only under single-element N-1 perturbations of a fixed base topology per grid size (drop one non-slack-connected generator or one branch, resampled if it disconnects the graph) plus independent ±20% load scaling — not arbitrary or unseen networks, and not simultaneous multi-element outages.
- Assumes access to multi-GPU training compute (8×A100 40GB, up to 800k steps for the 10,000-bus models) and, for feasibility restoration, a power-flow solver (pandapower with numba acceleration).

## Limitations / Failure Modes
- No hard AC-feasibility guarantee: only the reference-angle, box-bound, and branch-flow constraints are exact by construction; power-balance, thermal, and angle-difference constraints are merely penalized in the loss, not enforced — a limitation the authors say is shared by all ML-for-OPF methods (and by DC-OPF, structurally).
- The authors explicitly flag data drift/distribution shift as an open problem: a model trained under one regime may degrade on future grid conditions, and they call for further work on measuring this in practice and on monitoring/fine-tuning pipelines to correct for it.
- Perturbation diversity used for both training and robustness evaluation is narrow — independent per-load scaling plus single-element N-1 drops only; broader perturbation classes (generator capacity or thermal-rating changes, correlated demand patterns, multi-element outages) are named as future work, as is direct comparison against competing topology-aware ML-OPF architectures (e.g., 3–4-layer GCN models).

## Reusable Ingredients
- Encode-process-decode Graph Network with typed (heterogeneous) residual Interaction-Network blocks — directly analogous to graphkit's HeteroData/typed-message-passing design (`GNS_heterogeneous`).
- Constraint-augmented loss (supervised L2 + weighted constraint-violation-degree penalty, C=0.1) combined with sigmoid-squashed hard box-bound enforcement and analytic (non-learned) derivation of branch flow from predicted voltages — shrinks the constraint surface the network actually has to learn, rather than penalizing everything equally.
- N-1 "TopDrop" augmentation recipe: uniformly drop one non-slack-connected generator or one branch per example (50% of the dataset), resampling on disconnection, as a lightweight topology-robustness training signal.

## Open Questions
- Does CANOS's accuracy hold under OOD grid-size or topology transfer (train on one network, evaluate on a larger or structurally different one), rather than N-1 perturbations of the same base grid it trained on? (PFΔ later reports degradation here — ties to G3.)
- Appendix D shows accuracy/feasibility improving sharply with message-passing depth up to ~16–36 steps on the 500-bus grid, and the larger 2,000-/10,000-bus models use even deeper stacks (48–60) chosen empirically — is there a principled relationship between required depth and grid diameter/size, or was this simply tuned per grid?

## Claims
_No claims tracked yet — populate via /proof-checker._

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._

## Relevance to This Project
Strongest published flat (non-hierarchical, non-attention), typed-message-passing AC-OPF baseline at realistic transmission scale — its ~1% cost gap, ~50 ms inference, and sub-linear 500-to-10,000-bus scaling set the bar a hierarchical graphkit model (G1) must match on speed, while its untested OOD/topology-transfer robustness (per PFΔ, G3) is the gap such a model should close.

## Abstract (original)

> Optimal Power Flow (OPF) refers to a wide range of related optimization problems with the goal of operating power systems efficiently and securely. In the simplest setting, OPF determines how much power to generate in order to minimize costs while meeting demand for power and satisfying physical and operational constraints. In even the simplest case, power grid operators use approximations of the AC-OPF problem because solving the exact problem is prohibitively slow with state-of-the-art solvers. These approximations sacrifice accuracy and operational feasibility in favor of speed. This trade-off leads to costly "uplift payments" and increased carbon emissions, especially for large power grids. In the present work, we train a deep learning system (CANOS) to predict near-optimal solutions (within 1% of the true AC-OPF cost) without compromising speed (running in as little as 33--65 ms). Importantly, CANOS scales to realistic grid sizes with promising empirical results on grids containing as many as 10,000 buses. Finally, because CANOS is a Graph Neural Network, it is robust to changes in topology. We show that CANOS is accurate across N-1 topological perturbations of a base grid typically used in security-constrained analysis. This paves the way for more efficient optimization of more complex OPF problems which alter grid connectivity such as unit commitment, topology optimization and security-constrained OPF.
