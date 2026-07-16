---
type: idea
node_id: idea:contingency-amortization-breakeven
title: "Amortization break-even surface for learned contingency screening"
stage: archived
outcome: pending
added: 2026-07-06T12:26:43Z
based_on: ["paper:piloto2024_canos_fast_scalable"]
target_gaps: ["gap:G3"]
tags: []
---

# Amortization break-even surface for learned contingency screening

**stage:** `archived`  ·  **outcome:** `pending`

## Thesis
Measure the (N, contingency-count) region where a trained surrogate beats Newton-Raphson once label-generation + training cost are counted.

## Key risks
Killed for this agenda (Codex triage): sound domain economics but off-center for hierarchical scaling; good TPS-style side paper.

**Status update (2026-07-10)**: named in the kron-schur-unet proposal as the natural successor paper (fixed-Y scoping remark, SMW/LODF low-rank updates) — but the technical plan bars stating an O(n_I) per-contingency update cost without a concrete factorization, sparsity, and accuracy analysis; that analysis would be this idea's first work item if revived.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
