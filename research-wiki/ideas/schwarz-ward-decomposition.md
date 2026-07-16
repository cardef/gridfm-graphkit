---
type: idea
node_id: idea:schwarz-ward-decomposition
title: "Schwarz/Ward inference-time decomposition: train small, solve big via fixed-point iteration"
stage: proposed
outcome: pending
added: 2026-07-06T12:26:15Z
based_on: ["paper:pfaff2020_learning_meshbased_simulation"]
target_gaps: ["gap:G3"]
tags: []
---

# Schwarz/Ward inference-time decomposition: train small, solve big via fixed-point iteration

**stage:** `proposed`  ·  **outcome:** `pending`

## Thesis
Solve 2k-10k-bus PF with a small-grid-trained model via outer Gauss-Seidel/Schwarz over overlapping METIS partitions with Ward-compressed exteriors — size-OOD becomes in-distribution subproblems. Training-free G3 answer composable with any backbone.

## Key risks
Convergence of fixed-point loop with inexact learned inner solver is the crux; may read as classical DDM + learned solver. Codex triage rank 7. Learned-DDM prior exists for PDEs (arXiv 2312.14050), none for PF.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
