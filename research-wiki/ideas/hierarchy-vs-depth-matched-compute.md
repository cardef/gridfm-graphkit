---
type: idea
node_id: idea:hierarchy-vs-depth-matched-compute
title: "Hierarchy vs depth vs virtual node at matched compute, in-distribution + transfer"
stage: proposed
outcome: pending
added: 2026-07-06T12:26:15Z
based_on: ["paper:fortunato2022_multiscale_meshgraphnets", "paper:cao2022_efficient_learning_meshbased"]
target_gaps: ["gap:G1", "gap:G2", "gap:G3"]
tags: []
---

# Hierarchy vs depth vs virtual node at matched compute, in-distribution + transfer

**stage:** `proposed`  ·  **outcome:** `pending`

## Thesis
Matched-FLOPs: deep residual HGNS (~50 layers) vs one METIS/zone super-node level vs global virtual node on case2000/Texas + trained-on-118 zero-shot; shuffled-pooling control; Dirichlet-energy + error-vs-electrical-distance instrumentation. Absorbs hierarchy-levels-scaling-law (L* sweep) and oversmoothing-dirichlet-sweep as arms.

## Key risks
Matched FLOPs never perfectly fair; deep baseline and hierarchy may tie (informative negative). Codex triage rank 6.

**Status update (2026-07-10)**: partially absorbed into the kron-schur-unet campaign's required baseline family (flat depth/width frontier + virtual node + one credible scalable-global arm, technical plan §Fair comparison protocol). Comparison standard tightened there: params AND FLOPs both within ±10% (else dual iso-param/iso-FLOP reporting), and "beats flat at any depth" phrasing retired in favor of the measured frontier.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
