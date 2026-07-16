---
type: idea
node_id: idea:domain-vs-learned-coarsening
title: "Domain-given vs learned coarsening: controlled pooling-family benchmark for grid state regression"
stage: proposed
outcome: pending
added: 2026-07-06T12:26:13Z
based_on: ["paper:gao2019_graph_unets", "paper:cao2022_efficient_learning_meshbased", "paper:fortunato2022_multiscale_meshgraphnets"]
target_gaps: ["gap:G1", "gap:G2"]
tags: []
---

# Domain-given vs learned coarsening: controlled pooling-family benchmark for grid state regression

**stage:** `proposed`  ·  **outcome:** `pending`

## Thesis
One fixed hierarchical U-Net wrapper over HGNS; swap only the pooling assignment (learned TopK/DiffPool vs METIS vs voltage-level/zone vs Kron-Ward vs random control) at matched params on case500/2000 PF+OPF, incl. small-to-large transfer. Expect domain-given partitions to match or beat learned pooling.

## Key risks
Pooling implementation details may dominate; all hierarchy arms may lose to flat HGNS (itself a decisive negative). Codex triage rank 2, novelty 8.5/10 (least novelty risk).

**Status update (2026-07-10)**: the METIS/generic-pooling arm is now a *required mechanism-gate control* in the kron-schur-unet campaign (technical plan §Fair comparison protocol) — Kron latent hierarchy must beat it at comparable compute to support the electrical-hierarchy claim. The standalone benchmark remains a separate idea; comparisons must match both params AND FLOPs (±10%) or report iso-param and iso-FLOP side by side.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
