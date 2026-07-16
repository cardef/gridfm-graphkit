---
type: idea
node_id: idea:pf-information-range-atlas
title: "Physics information-range atlas: sensitivity decay vs hop and electrical distance, per task and size"
stage: proposed
outcome: pending
added: 2026-07-06T12:26:14Z
based_on: ["paper:ma2023_graph_inductive_biases"]
target_gaps: ["gap:G1", "gap:G3"]
tags: []
---

# Physics information-range atlas: sensitivity decay vs hop and electrical distance, per task and size

**stage:** `proposed`  ·  **outcome:** `pending`

## Thesis
CPU-only: Jacobian/PTDF-LODF decay vs hop vs effective-resistance distance across case14-9241, PF vs OPF. v0 pilot (2026-07-06, DC PTDF on case1354/Texas2k) SURPRISED: decay is slow (10x per ~10 hops, diameter ~25-28) and hop correlates BETTER than Reff (Spearman -0.42 vs -0.16) — electrical-exponential-locality folklore incomplete; long-range coupling real.

## Key risks
Reviewers may say diagnostic-not-ML; no clean decay law may exist (the pilot suggests exactly that — which is the finding). Codex triage rank 5. Full atlas needs AC Jacobian + decorrelated distance comparison.

## Connections
_Edges are recorded in `graph/edges.jsonl`; summarize here for human readers._
