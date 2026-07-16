---
type: idea
node_id: idea:hmatrix-zbus-attention
title: "H-matrix/Z_bus-structured attention (+ standalone compressibility diagnostic)"
stage: proposed
outcome: pending
added: 2026-07-06T12:26:15Z
based_on: ["paper:ma2023_graph_inductive_biases"]
target_gaps: ["gap:G4"]
tags: []
---

# H-matrix/Z_bus-structured attention (+ standalone compressibility diagnostic)

**stage:** `proposed`  ·  **outcome:** `pending`

## Thesis
Stage 1 (CPU, standalone): measure H-matrix block-low-rank compressibility of transmission Z_bus on electrical-distance cluster trees — unmeasured in ML literature; go/no-go gate. Stage 2: inherit attention sparsity from Z_bus admissibility blocks for sub-quadratic GRIT-class transformers past case14.

## Key risks
HIGH: Z_bus block ranks may not be low; numerics community may consider structure known. Differentiates from HH-MPNN (generic Performer + scalar encoding). Codex triage rank 9. Pilot B weak Reff-locality is a caution flag for the cluster-tree premise.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
